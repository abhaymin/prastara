//! Inkling 2's blocks on the device (docs/INKLING-PLAN.md I8), for strata-hybrid's graph_inkling.rs: depth mixing,
//! the loop controllers, the short causal convolutions with their state held on the device, clipping, attention with
//! a learned relative bias and a sink over a (ring) cache, the sigmoid router with its selection bias, SiTU and the
//! final soft cap. Each kernel computes what model/inkling.rs computes on the CPU, in the same order of operations
//! where that is cheap (so the graph's logits are the CPU path's within float rounding); K, V and the attention mask
//! are rounded to f16 as the CPU path (and llama.cpp) keep them.
//!
//! Every function here only queues its kernel: nothing waits for the device.

#![allow(unused_assignments)]

use super::*;
use crate::kernels::CUBE;

/// The most states one depth mix reads (the earlier blocks and the current one).
pub const MIX_STATES: u32 = 64;

/// The sum of `v` over the cube (every unit gets it; the whole cube must call it).
#[cube]
fn cube_sum(v: f32) -> f32 {
    let mut sh = SharedMemory::<f32>::new(CUBE as usize);
    sh[UNIT_POS as usize] = v;
    sync_cube();
    let mut stride = u32::new(64);
    while stride > 0 {
        if UNIT_POS < stride {
            sh[UNIT_POS as usize] = sh[UNIT_POS as usize] + sh[(UNIT_POS + stride) as usize];
        }
        sync_cube();
        stride /= 2;
    }
    let r = sh[0usize];
    sync_cube();
    r
}

/// The largest `v` over the cube (every unit gets it).
#[cube]
fn cube_max(v: f32) -> f32 {
    let mut sh = SharedMemory::<f32>::new(CUBE as usize);
    sh[UNIT_POS as usize] = v;
    sync_cube();
    let mut stride = u32::new(64);
    while stride > 0 {
        if UNIT_POS < stride {
            sh[UNIT_POS as usize] = f32::max(sh[UNIT_POS as usize], sh[(UNIT_POS + stride) as usize]);
        }
        sync_cube();
        stride /= 2;
    }
    let r = sh[0usize];
    sync_cube();
    r
}

/// x rounded to the nearest f16 (ties to even), kept as f32: the normal range, which is all K, V and the mask use
/// (a subnormal keeps 10 bits here, within 6e-8 of the f16 value).
#[cube]
fn f16r(x: f32) -> f32 {
    let b = u32::reinterpret(x);
    f32::reinterpret((b + 0xFFFu32 + ((b >> 13) & 1)) & 0xFFFF_E000u32)
}

/// tanh to about an ulp: an odd series near 0 (where e^2x - 1 cancels), the exponential form elsewhere.
#[cube]
fn tanh_p(x: f32) -> f32 {
    let a = f32::abs(x);
    let mut t = f32::new(0.0);
    if a < f32::new(0.125) {
        let x2 = x * x;
        t = x * (f32::new(1.0) + x2 * (f32::new(-0.33333334) + x2 * (f32::new(0.13333334) + x2 * (f32::new(-0.053968254) + x2 * f32::new(0.021869488)))));
    } else {
        let e = f32::exp(f32::new(-2.0) * a);
        t = (f32::new(1.0) - e) / (f32::new(1.0) + e);
        if x < f32::new(0.0) {
            t = -t;
        }
    }
    t
}

#[cube]
fn sigm(x: f32) -> f32 {
    f32::new(1.0) / (f32::new(1.0) + f32::exp(-x))
}

/// y = v over n values.
#[cube(launch_unchecked)]
pub fn ink_fill(y: &mut Array<f32>, v: f32, n: u32) {
    let j = (CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X) * CUBE + UNIT_POS;
    if j < n {
        y[j as usize] = v;
    }
}

/// The depth mix of one query (inkling.rs `mix`), a cube per token (CUBE_POS_X): the `ns` states
/// st[off + i * stride + tok * d ..] and cur[tok * d ..], each RMS-normed (eps 1e-6), mixed by the softmax of their
/// dot with q[qo .. qo + d] / sqrt(d).
#[cube(launch_unchecked)]
pub fn ink_mix(st: &Array<f32>, cur: &Array<f32>, q: &Array<f32>, y: &mut Array<f32>, off: u32, stride: u32, ns: u32, qo: u32, d: u32) {
    let tok = CUBE_POS_X;
    let mut kk = SharedMemory::<f32>::new(MIX_STATES as usize);
    let mut lg = SharedMemory::<f32>::new(MIX_STATES as usize);
    let scale = f32::new(1.0) / f32::sqrt(f32::cast_from(d));
    for s in 0..ns + 1 {
        let mut ss = f32::new(0.0);
        let mut dq = f32::new(0.0);
        let mut j = UNIT_POS;
        while j < d {
            let mut v = f32::new(0.0);
            if s < ns {
                v = st[(off + s * stride + tok * d + j) as usize];
            } else {
                v = cur[(tok * d + j) as usize];
            }
            ss += v * v;
            dq += v * q[(qo + j) as usize];
            j += CUBE;
        }
        let ss = cube_sum(ss);
        let dq = cube_sum(dq);
        let k = f32::new(1.0) / f32::sqrt(ss / f32::cast_from(d) + f32::new(1.0e-6));
        if UNIT_POS == 0 {
            kk[s as usize] = k;
            lg[s as usize] = dq * k * scale;
        }
        sync_cube();
    }
    let mut m = f32::new(-1.0e30);
    for s in 0..ns + 1 {
        m = f32::max(m, lg[s as usize]);
    }
    let mut tot = f32::new(0.0);
    for s in 0..ns + 1 {
        tot += f32::exp(lg[s as usize] - m);
    }
    let mut j = UNIT_POS;
    while j < d {
        let mut acc = f32::new(0.0);
        for s in 0..ns + 1 {
            let mut v = f32::new(0.0);
            if s < ns {
                v = st[(off + s * stride + tok * d + j) as usize];
            } else {
                v = cur[(tok * d + j) as usize];
            }
            acc += f32::exp(lg[s as usize] - m) / tot * (v * kk[s as usize]);
        }
        y[(tok * d + j) as usize] = acc;
        j += CUBE;
    }
}

/// A controller's input per token (CUBE_POS_X): y[tok * 2d ..] = [rmsn(a[tok]), rmsn(b[tok])], eps 1e-6.
#[cube(launch_unchecked)]
pub fn ink_cat_norm(a: &Array<f32>, b: &Array<f32>, y: &mut Array<f32>, d: u32) {
    let tok = CUBE_POS_X;
    let mut sa = f32::new(0.0);
    let mut sb = f32::new(0.0);
    let mut j = UNIT_POS;
    while j < d {
        let va = a[(tok * d + j) as usize];
        let vb = b[(tok * d + j) as usize];
        sa += va * va;
        sb += vb * vb;
        j += CUBE;
    }
    let ka = f32::new(1.0) / f32::sqrt(cube_sum(sa) / f32::cast_from(d) + f32::new(1.0e-6));
    let kb = f32::new(1.0) / f32::sqrt(cube_sum(sb) / f32::cast_from(d) + f32::new(1.0e-6));
    let mut j = UNIT_POS;
    while j < d {
        y[(tok * 2 * d + j) as usize] = a[(tok * d + j) as usize] * ka;
        y[(tok * 2 * d + d + j) as usize] = b[(tok * d + j) as usize] * kb;
        j += CUBE;
    }
}

/// A controller's output: y = c - tanh(f) c + tanh(g) m, over n values.
#[cube(launch_unchecked)]
pub fn ink_ctrl_out(c: &Array<f32>, m: &Array<f32>, g: &Array<f32>, f: &Array<f32>, y: &mut Array<f32>, n: u32) {
    let j = (CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X) * CUBE + UNIT_POS;
    if j < n {
        let cv = c[j as usize];
        y[j as usize] = cv - tanh_p(f[j as usize]) * cv + tanh_p(g[j as usize]) * m[j as usize];
    }
}

/// A causal short convolution with a residual (inkling.rs `convolve`) for `nt` tokens of `n` channels, a unit per
/// channel: each token's earlier inputs come from the chunk or from `hist` (per channel the n_conv - 1 before the
/// chunk, oldest first, at hoff), which then takes the chunk's last ones. `dynamic`: dy holds per token and head the
/// input-dependent taps (lag 0 first), `heads` heads.
#[cube(launch_unchecked)]
pub fn ink_conv(x: &Array<f32>, w: &Array<f32>, dy: &Array<f32>, hist: &mut Array<f32>, y: &mut Array<f32>, n: u32, nc: u32, nt: u32,
                hoff: u32, heads: u32, k: f32, #[comptime] dynamic: bool) {
    let c = (CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X) * CUBE + UNIT_POS;
    if c < n {
        let past = nc - 1;
        let hb = hoff + c * past;
        let mut ph = u32::new(1);
        if dynamic {
            ph = n / heads;
        }
        for t in 0..nt {
            let cur = x[(t * n + c) as usize];
            let mut acc = cur * w[(c * nc + past) as usize];
            for j in 0..past {
                let l = past - j;
                let mut v = f32::new(0.0);
                if t >= l {
                    v = x[((t - l) * n + c) as usize];
                } else {
                    v = hist[(hb + past - (l - t)) as usize];
                }
                acc += v * w[(c * nc + j) as usize];
            }
            if dynamic {
                let tb = (t * heads + c / ph) * nc;
                acc += cur * dy[tb as usize];
                for l in 1..nc {
                    let mut v = f32::new(0.0);
                    if t >= l {
                        v = x[((t - l) * n + c) as usize];
                    } else {
                        v = hist[(hb + past - (l - t)) as usize];
                    }
                    acc += v * dy[(tb + l) as usize];
                }
            }
            y[(t * n + c) as usize] = (cur + acc) * k;
        }
        // the new history: lag past - j from the next position; the old value it reads (index j + nt) is not yet
        // overwritten
        for j in 0..past {
            let l = past - j;
            let mut v = f32::new(0.0);
            if nt >= l {
                v = x[((nt - l) * n + c) as usize];
            } else {
                v = hist[(hb + past - (l - nt)) as usize];
            }
            hist[(hb + j) as usize] = v;
        }
    }
}

/// Each row of `d` values (a cube per row): times s[(row % rpt) * d + j] when `scaled`, scaled down to an L2 norm of
/// at most `thr` (inkling.rs `clip`), times tau[row / rpt] when `taus`, rounded to f16 when `round`.
#[cube(launch_unchecked)]
pub fn ink_clip(x: &mut Array<f32>, s: &Array<f32>, tau: &Array<f32>, d: u32, rpt: u32, thr: f32, #[comptime] scaled: bool,
                #[comptime] taus: bool, #[comptime] round: bool) {
    let row = CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X;
    let base = row * d;
    let sb = (row % rpt) * d;
    let mut acc = f32::new(0.0);
    let mut j = UNIT_POS;
    while j < d {
        let mut v = x[(base + j) as usize];
        if scaled {
            v = v * s[(sb + j) as usize];
        }
        acc += v * v;
        j += CUBE;
    }
    let norm = f32::sqrt(cube_sum(acc)) + f32::new(1.0e-6);
    let f = f32::clamp(thr / norm, f32::new(0.0), f32::new(1.0));
    let mut t = f32::new(1.0);
    if taus {
        t = tau[(row / rpt) as usize];
    }
    let mut j = UNIT_POS;
    while j < d {
        let mut v = x[(base + j) as usize];
        if scaled {
            v = v * s[(sb + j) as usize];
        }
        v = v * f;
        if taus {
            v = v * t;
        }
        if round {
            v = f16r(v);
        }
        x[(base + j) as usize] = v;
        j += CUBE;
    }
}

/// Each row of `d` values RMS-normed without a weight (a cube per row), rounded to f16 when `round`.
#[cube(launch_unchecked)]
pub fn ink_rmsn_rows(x: &mut Array<f32>, d: u32, eps: f32, #[comptime] round: bool) {
    let row = CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X;
    let base = row * d;
    let mut acc = f32::new(0.0);
    let mut j = UNIT_POS;
    while j < d {
        let v = x[(base + j) as usize];
        acc += v * v;
        j += CUBE;
    }
    let k = f32::new(1.0) / f32::sqrt(cube_sum(acc) / f32::cast_from(d) + eps);
    let mut j = UNIT_POS;
    while j < d {
        let mut v = x[(base + j) as usize] * k;
        if round {
            v = f16r(v);
        }
        x[(base + j) as usize] = v;
        j += CUBE;
    }
}

/// `nt` rows of `row` values into a cache of `wring` positions (a ring for a windowed layer): position base + tok at
/// (base + tok) % wring.
#[cube(launch_unchecked)]
pub fn ink_kv_put(x: &Array<f32>, cache: &mut Array<f32>, base: u32, row: u32, wring: u32, n: u32) {
    let j = (CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X) * CUBE + UNIT_POS;
    if j < n {
        let tok = j / row;
        cache[(((base + tok) % wring) * row + j % row) as usize] = x[j as usize];
    }
}

/// Attention with Inkling's relative bias and sink (inkling.rs `layer`), a cube per (head, token): the token at
/// position p = base + tok reads positions max(0, p + 1 - win)..=p (win 0: all), cached at t % wring. Its score of t is
/// q.k * scale + f16(bias(p - t) - sink), bias(dist) = rb[dist] . r (0 from `extent` on); the softmax has one more
/// logit of 0 (the sink). `s` is scratch of nt * n_head * span scores.
#[cube(launch_unchecked)]
pub fn ink_attend(q: &Array<f32>, kc: &Array<f32>, vc: &Array<f32>, r: &Array<f32>, rb: &Array<f32>, sk: &Array<f32>, s: &mut Array<f32>,
                  o: &mut Array<f32>, base: u32, nh: u32, nkv: u32, hd: u32, nrel: u32, extent: u32, win: u32, wring: u32, span: u32,
                  scale: f32) {
    let h = CUBE_POS_X;
    let tok = CUBE_POS_Y;
    let row = tok * nh + h;
    let p = base + tok;
    let mut first = u32::new(0);
    if win > 0 && p + 1 > win {
        first = p + 1 - win;
    }
    let cnt = p + 1 - first;
    let kh = h / (nh / nkv);
    let qo = row * hd;
    let ro = row * nrel;
    let so = row * span;
    let sink = sk[row as usize];
    let mut top = f32::new(0.0);
    let mut i = UNIT_POS;
    while i < cnt {
        let t = first + i;
        let ko = ((t % wring) * nkv + kh) * hd;
        let mut acc = f32::new(0.0);
        for j in 0..hd {
            acc += q[(qo + j) as usize] * kc[(ko + j) as usize];
        }
        let dist = p - t;
        let mut b = f32::new(0.0);
        if dist < extent {
            for j in 0..nrel {
                b += rb[(dist * nrel + j) as usize] * r[(ro + j) as usize];
            }
        }
        let v = acc * scale + f16r(b - sink);
        s[(so + i) as usize] = v;
        top = f32::max(top, v);
        i += CUBE;
    }
    let top = cube_max(top);
    let mut sum = f32::new(0.0);
    let mut i = UNIT_POS;
    while i < cnt {
        let e = f32::exp(s[(so + i) as usize] - top);
        s[(so + i) as usize] = e;
        sum += e;
        i += CUBE;
    }
    let total = f32::exp(-top) + cube_sum(sum);
    let mut j = UNIT_POS;
    while j < hd {
        let mut acc = f32::new(0.0);
        for i in 0..cnt {
            let t = first + i;
            acc += s[(so + i) as usize] / total * vc[(((t % wring) * nkv + kh) * hd + j) as usize];
        }
        o[(qo + j) as usize] = acc;
        j += CUBE;
    }
}

/// a *= sigmoid(g), over n values.
#[cube(launch_unchecked)]
pub fn ink_sig_mul(a: &mut Array<f32>, g: &Array<f32>, n: u32) {
    let j = (CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X) * CUBE + UNIT_POS;
    if j < n {
        a[j as usize] = a[j as usize] * sigm(g[j as usize]);
    }
}

/// Inkling's router per token (a cube per token): sigmoid probabilities, the `k` best by probability plus the
/// selection bias (ties to the lower index, as a stable sort), weights p / (sum of the chosen p + 1e-20) * ws.
/// -> out[tok * 2k ..]: the indices (f32), then the weights (route_topk's layout, for moe_combine).
#[cube(launch_unchecked)]
pub fn ink_route(logits: &Array<f32>, bias: &Array<f32>, out: &mut Array<f32>, n_e: u32, k: u32, ws: f32, #[comptime] biased: bool) {
    if UNIT_POS == 0 {
        let tok = CUBE_POS_X;
        let base = tok * n_e;
        let ob = tok * 2 * k;
        let mut prev_s = f32::new(1.0e30);
        let mut prev_e = u32::new(0);
        let mut first = true;
        let mut wsum = f32::new(0.0);
        for s in 0..k {
            let mut best = f32::new(-1.0e30);
            let mut be = u32::new(0);
            for e in 0..n_e {
                let mut sel = sigm(logits[(base + e) as usize]);
                if biased {
                    sel += bias[e as usize];
                }
                let below = first || sel < prev_s || (sel == prev_s && e > prev_e);
                if below && sel > best {
                    best = sel;
                    be = e;
                }
            }
            let p = sigm(logits[(base + be) as usize]);
            out[(ob + s) as usize] = f32::cast_from(be);
            out[(ob + k + s) as usize] = p;
            wsum += p;
            prev_s = best;
            prev_e = be;
            first = false;
        }
        let d = wsum + f32::new(1.0e-20);
        for s in 0..k {
            out[(ob + k + s) as usize] = out[(ob + k + s) as usize] / d * ws;
        }
    }
}

/// SiTU on a fused gate/up product: per token the 2ff values [gate, up] -> ff values
/// beta tanh(g / beta) sigmoid(g) * lb tanh(u / lb) (u itself when lb is 0), over n = tokens * ff.
#[cube(launch_unchecked)]
pub fn ink_situ(gu: &Array<f32>, y: &mut Array<f32>, ff: u32, beta: f32, lb: f32, n: u32) {
    let j = (CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X) * CUBE + UNIT_POS;
    if j < n {
        let tok = j / ff;
        let i = j % ff;
        let g = gu[(tok * 2 * ff + i) as usize];
        let mut u = gu[(tok * 2 * ff + ff + i) as usize];
        if lb > f32::new(0.0) {
            u = lb * tanh_p(u / lb);
        }
        y[j as usize] = beta * tanh_p(g / beta) * sigm(g) * u;
    }
}

/// x = cap tanh(x / cap), over n values.
#[cube(launch_unchecked)]
pub fn ink_softcap(x: &mut Array<f32>, cap: f32, n: u32) {
    let j = (CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X) * CUBE + UNIT_POS;
    if j < n {
        x[j as usize] = cap * tanh_p(x[j as usize] / cap);
    }
}

/// A grid of `n` units in cubes of CUBE, over x and y (each axis has a limit of 65535 on some backends).
fn flat(n: usize) -> CubeCount {
    let cubes = (n as u32).div_ceil(CUBE).max(1);
    let cx = cubes.min(65535);
    CubeCount::Static(cx, cubes.div_ceil(cx), 1)
}

/// A grid of `rows` cubes over x and y.
fn rows_grid(rows: usize) -> CubeCount {
    let r = (rows as u32).max(1);
    let cx = r.min(65535);
    CubeCount::Static(cx, r.div_ceil(cx), 1)
}

impl Gpu {
    /// Whether this device has the Inkling kernels (CubeCL; not the native Metal backend).
    pub fn has_ink_kernels(&self) -> bool {
        !self.is_native()
    }

    /// y = v everywhere, queued.
    pub fn ink_fill(&self, y: &Buf, v: f32) {
        stats::launch(1);
        with_client!(self, c, R, unsafe {
            ink_fill::launch_unchecked::<R>(c, flat(y.len), CubeDim::new_1d(CUBE), y.arg(), v, y.len as u32);
        });
    }

    /// A buffer of `n` zeros, queued (no upload).
    pub fn ink_zeros(&self, n: usize) -> Buf {
        let y = self.scratch(n.max(1));
        self.ink_fill(&y, 0.0);
        y
    }

    /// The depth mix of query `qi` (d values at queries[qi * d]) for `nt` tokens: the `ns` states in `st` (state i of
    /// token tok at off + i * stride + tok * d) and `cur`. -> nt * d values.
    #[allow(clippy::too_many_arguments)]
    pub fn ink_mix(&self, st: &Buf, off: usize, stride: usize, ns: usize, cur: &Buf, queries: &Buf, qi: usize, d: usize, nt: usize) -> Buf {
        assert!(ns < MIX_STATES as usize, "ink_mix: {ns} states (at most {})", MIX_STATES - 1);
        stats::launch(1);
        let y = self.scratch(nt * d);
        with_client!(self, c, R, unsafe {
            ink_mix::launch_unchecked::<R>(c, CubeCount::Static(nt as u32, 1, 1), CubeDim::new_1d(CUBE), st.arg(), cur.arg(), queries.arg(), y.arg(),
                off as u32, stride as u32, ns as u32, (qi * d) as u32, d as u32);
        });
        y
    }

    /// [rmsn(a), rmsn(b)] per token (rows of d): the controllers' input.
    pub fn ink_cat_norm(&self, a: &Buf, b: &Buf, d: usize) -> Buf {
        stats::launch(1);
        let nt = a.len / d;
        let y = self.scratch(2 * a.len);
        with_client!(self, c, R, unsafe {
            ink_cat_norm::launch_unchecked::<R>(c, CubeCount::Static(nt as u32, 1, 1), CubeDim::new_1d(CUBE), a.arg(), b.arg(), y.arg(), d as u32);
        });
        y
    }

    /// c - tanh(f) c + tanh(g) m.
    pub fn ink_ctrl_out(&self, cur: &Buf, m: &Buf, g: &Buf, f: &Buf) -> Buf {
        stats::launch(1);
        let y = self.scratch(cur.len);
        with_client!(self, c, R, unsafe {
            ink_ctrl_out::launch_unchecked::<R>(c, flat(cur.len), CubeDim::new_1d(CUBE), cur.arg(), m.arg(), g.arg(), f.arg(), y.arg(), cur.len as u32);
        });
        y
    }

    /// The short convolution of `nt` tokens of `n` channels (x), with the taps `w` (n * nc), the dynamic taps `dy`
    /// (per token, `heads` heads of nc) when given, and the history in `hist` at `hoff` (updated). -> nt * n values.
    #[allow(clippy::too_many_arguments)]
    pub fn ink_conv(&self, x: &Buf, w: &Buf, dy: Option<&Buf>, hist: &Buf, hoff: usize, n: usize, nc: usize, nt: usize) -> Buf {
        stats::launch(1);
        let y = self.scratch(nt * n);
        let heads = dy.map_or(1, |d| d.len / nt / nc);
        let k = 1.0 / (((if dy.is_some() { 2 } else { 1 }) * nc + 1) as f32).sqrt();
        with_client!(self, c, R, unsafe {
            ink_conv::launch_unchecked::<R>(c, flat(n), CubeDim::new_1d(CUBE), x.arg(), w.arg(), dy.unwrap_or(x).arg(), hist.arg(), y.arg(), n as u32,
                nc as u32, nt as u32, hoff as u32, heads as u32, k, dy.is_some());
        });
        y
    }

    /// Clips each row of `d` values of x in place (see the kernel): `scale` per channel of a token's `rpt` rows,
    /// `tau` per token, f16 rounding when `round`.
    pub fn ink_clip(&self, x: &Buf, scale: Option<&Buf>, tau: Option<&Buf>, d: usize, rpt: usize, thr: f32, round: bool) {
        stats::launch(1);
        with_client!(self, c, R, unsafe {
            ink_clip::launch_unchecked::<R>(c, rows_grid(x.len / d), CubeDim::new_1d(CUBE), x.arg(), scale.unwrap_or(x).arg(), tau.unwrap_or(x).arg(),
                d as u32, rpt as u32, thr, scale.is_some(), tau.is_some(), round);
        });
    }

    /// Each row of `d` values RMS-normed in place without a weight, f16-rounded when `round`.
    pub fn ink_rmsn_rows(&self, x: &Buf, d: usize, eps: f32, round: bool) {
        stats::launch(1);
        with_client!(self, c, R, unsafe {
            ink_rmsn_rows::launch_unchecked::<R>(c, rows_grid(x.len / d), CubeDim::new_1d(CUBE), x.arg(), d as u32, eps, round);
        });
    }

    /// Rows of `row` values for positions base.. into a cache of `wring` positions.
    pub fn ink_kv_put(&self, x: &Buf, cache: &Buf, base: usize, row: usize, wring: usize) {
        stats::launch(1);
        with_client!(self, c, R, unsafe {
            ink_kv_put::launch_unchecked::<R>(c, flat(x.len), CubeDim::new_1d(CUBE), x.arg(), cache.arg(), base as u32, row as u32, wring as u32,
                x.len as u32);
        });
    }

    /// Inkling attention for `nt` tokens at positions base.. (see the kernel). -> nt * n_head * hd values.
    #[allow(clippy::too_many_arguments)]
    pub fn ink_attend(&self, q: &Buf, kc: &Buf, vc: &Buf, r: &Buf, rb: &Buf, sinks: &Buf, base: usize, nt: usize, n_head: usize, n_kv: usize, hd: usize,
                      n_rel: usize, extent: usize, win: usize, wring: usize) -> Buf {
        stats::launch(1);
        let n = base + nt;
        let span = if win > 0 { n.min(win) } else { n };
        let s = self.scratch(nt * n_head * span);
        let o = self.scratch(nt * n_head * hd);
        with_client!(self, c, R, unsafe {
            ink_attend::launch_unchecked::<R>(c, CubeCount::Static(n_head as u32, nt as u32, 1), CubeDim::new_1d(CUBE), q.arg(), kc.arg(), vc.arg(),
                r.arg(), rb.arg(), sinks.arg(), s.arg(), o.arg(), base as u32, n_head as u32, n_kv as u32, hd as u32, n_rel as u32, extent as u32,
                win as u32, wring as u32, span as u32, 1.0 / (hd as f32).sqrt());
        });
        o
    }

    /// a *= sigmoid(g), in place.
    pub fn ink_sig_mul(&self, a: &Buf, g: &Buf) {
        stats::launch(1);
        with_client!(self, c, R, unsafe {
            ink_sig_mul::launch_unchecked::<R>(c, flat(a.len), CubeDim::new_1d(CUBE), a.arg(), g.arg(), a.len as u32);
        });
    }

    /// The router's choice per token from its logits (rows of n_e): -> per token k indices, then k weights.
    pub fn ink_route(&self, logits: &Buf, bias: Option<&Buf>, n_e: usize, k: usize, ws: f32) -> Buf {
        stats::launch(1);
        let nt = logits.len / n_e;
        let out = self.scratch(nt * 2 * k);
        with_client!(self, c, R, unsafe {
            ink_route::launch_unchecked::<R>(c, CubeCount::Static(nt as u32, 1, 1), CubeDim::new_1d(32), logits.arg(), bias.unwrap_or(logits).arg(),
                out.arg(), n_e as u32, k as u32, ws, bias.is_some());
        });
        out
    }

    /// SiTU of fused gate/up rows (2ff per token) -> ff per token.
    pub fn ink_situ(&self, gu: &Buf, ff: usize, beta: f32, lb: f32) -> Buf {
        stats::launch(1);
        let n = gu.len / 2;
        let y = self.scratch(n);
        with_client!(self, c, R, unsafe {
            ink_situ::launch_unchecked::<R>(c, flat(n), CubeDim::new_1d(CUBE), gu.arg(), y.arg(), ff as u32, beta, lb, n as u32);
        });
        y
    }

    /// x = cap tanh(x / cap), in place.
    pub fn ink_softcap(&self, x: &Buf, cap: f32) {
        stats::launch(1);
        with_client!(self, c, R, unsafe {
            ink_softcap::launch_unchecked::<R>(c, flat(x.len), CubeDim::new_1d(CUBE), x.arg(), cap, x.len as u32);
        });
    }
}
