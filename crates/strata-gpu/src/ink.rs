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
use crate::kernels::{row_part, warp_sum, F4, MV_ROWS, CUBE};

/// The most states one depth mix reads (the earlier blocks and the current one).
pub const MIX_STATES: u32 = 16;
/// The most experts the router kernel takes.
pub const ROUTE_EXPERTS: u32 = 256;
/// Attention's positions per split (flash decoding): one token's, and a chunk's (fewer partial results).
const SPLIT_ONE: u32 = 64;
const SPLIT_CHUNK: u32 = 256;

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
/// dot with q[qo .. qo + d] / sqrt(d). Every state's two sums are reduced together.
#[cube(launch_unchecked)]
pub fn ink_mix(st: &Array<f32>, cur: &Array<f32>, q: &Array<f32>, y: &mut Array<f32>, off: u32, stride: u32, ns: u32, qo: u32, d: u32) {
    let tok = CUBE_POS_X;
    let mut pss = SharedMemory::<f32>::new((MIX_STATES * CUBE) as usize);
    let mut pdq = SharedMemory::<f32>::new((MIX_STATES * CUBE) as usize);
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
        pss[(s * CUBE + UNIT_POS) as usize] = ss;
        pdq[(s * CUBE + UNIT_POS) as usize] = dq;
    }
    sync_cube();
    let mut stride2 = u32::new(64);
    while stride2 > 0 {
        if UNIT_POS < stride2 {
            for s in 0..ns + 1 {
                let a = (s * CUBE + UNIT_POS) as usize;
                let b = (s * CUBE + UNIT_POS + stride2) as usize;
                pss[a] = pss[a] + pss[b];
                pdq[a] = pdq[a] + pdq[b];
            }
        }
        sync_cube();
        stride2 /= 2;
    }
    let mut m = f32::new(-1.0e30);
    for s in 0..ns + 1 {
        let k = f32::new(1.0) / f32::sqrt(pss[(s * CUBE) as usize] / f32::cast_from(d) + f32::new(1.0e-6));
        m = f32::max(m, pdq[(s * CUBE) as usize] * k * scale);
    }
    let mut tot = f32::new(0.0);
    for s in 0..ns + 1 {
        let k = f32::new(1.0) / f32::sqrt(pss[(s * CUBE) as usize] / f32::cast_from(d) + f32::new(1.0e-6));
        tot += f32::exp(pdq[(s * CUBE) as usize] * k * scale - m);
    }
    let mut j = UNIT_POS;
    while j < d {
        let mut acc = f32::new(0.0);
        for s in 0..ns + 1 {
            let k = f32::new(1.0) / f32::sqrt(pss[(s * CUBE) as usize] / f32::cast_from(d) + f32::new(1.0e-6));
            let mut v = f32::new(0.0);
            if s < ns {
                v = st[(off + s * stride + tok * d + j) as usize];
            } else {
                v = cur[(tok * d + j) as usize];
            }
            acc += f32::exp(pdq[(s * CUBE) as usize] * k * scale - m) / tot * (v * k);
        }
        y[(tok * d + j) as usize] = acc;
        j += CUBE;
    }
}

/// Units per cube of `ink_mix_q`.
const MIX_CUBE: u32 = 1024;

/// The depth mix of up to four queries at once (q[qo + i * d ..], i < nq) over the same states, into y0..y3: a cube of
/// 32 warps per token, the sums reduced by plane operations (devices with 32-wide planes; `ink_mix` elsewhere).
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn ink_mix_q(st: &Array<f32>, cur: &Array<f32>, q: &Array<f32>, y0: &mut Array<f32>, y1: &mut Array<f32>, y2: &mut Array<f32>, y3: &mut Array<f32>,
                 off: u32, stride: u32, ns: u32, qo: u32, nq: u32, d: u32) {
    let tok = CUBE_POS_X;
    let warp = UNIT_POS / 32;
    let lane = UNIT_POS % 32;
    let mut part = SharedMemory::<f32>::new((MIX_STATES * 5 * 32) as usize);
    let mut fin = SharedMemory::<f32>::new((MIX_STATES * 5) as usize);
    for s in 0..ns + 1 {
        let mut ss = f32::new(0.0);
        let mut d0 = f32::new(0.0);
        let mut d1 = f32::new(0.0);
        let mut d2 = f32::new(0.0);
        let mut d3 = f32::new(0.0);
        let mut j = UNIT_POS;
        while j < d {
            let mut v = f32::new(0.0);
            if s < ns {
                v = st[(off + s * stride + tok * d + j) as usize];
            } else {
                v = cur[(tok * d + j) as usize];
            }
            ss += v * v;
            d0 += v * q[(qo + j) as usize];
            if nq > 1 {
                d1 += v * q[(qo + d + j) as usize];
            }
            if nq > 2 {
                d2 += v * q[(qo + 2 * d + j) as usize];
            }
            if nq > 3 {
                d3 += v * q[(qo + 3 * d + j) as usize];
            }
            j += MIX_CUBE;
        }
        let ss = plane_sum(ss);
        let d0 = plane_sum(d0);
        let d1 = plane_sum(d1);
        let d2 = plane_sum(d2);
        let d3 = plane_sum(d3);
        if lane == 0 {
            let b = s * 5 * 32 + warp;
            part[b as usize] = ss;
            part[(b + 32) as usize] = d0;
            part[(b + 64) as usize] = d1;
            part[(b + 96) as usize] = d2;
            part[(b + 128) as usize] = d3;
        }
    }
    sync_cube();
    if UNIT_POS < (ns + 1) * 5 {
        let mut acc = f32::new(0.0);
        for w in 0..32u32 {
            acc += part[(UNIT_POS * 32 + w) as usize];
        }
        fin[UNIT_POS as usize] = acc;
    }
    sync_cube();
    let scale = f32::new(1.0) / f32::sqrt(f32::cast_from(d));
    for qi in 0..nq {
        let mut m = f32::new(-1.0e30);
        for s in 0..ns + 1 {
            let k = f32::new(1.0) / f32::sqrt(fin[(s * 5) as usize] / f32::cast_from(d) + f32::new(1.0e-6));
            m = f32::max(m, fin[(s * 5 + 1 + qi) as usize] * k * scale);
        }
        let mut tot = f32::new(0.0);
        for s in 0..ns + 1 {
            let k = f32::new(1.0) / f32::sqrt(fin[(s * 5) as usize] / f32::cast_from(d) + f32::new(1.0e-6));
            tot += f32::exp(fin[(s * 5 + 1 + qi) as usize] * k * scale - m);
        }
        let mut j = UNIT_POS;
        while j < d {
            let mut acc = f32::new(0.0);
            for s in 0..ns + 1 {
                let k = f32::new(1.0) / f32::sqrt(fin[(s * 5) as usize] / f32::cast_from(d) + f32::new(1.0e-6));
                let mut v = f32::new(0.0);
                if s < ns {
                    v = st[(off + s * stride + tok * d + j) as usize];
                } else {
                    v = cur[(tok * d + j) as usize];
                }
                acc += f32::exp(fin[(s * 5 + 1 + qi) as usize] * k * scale - m) / tot * (v * k);
            }
            let at = (tok * d + j) as usize;
            if qi == 0 {
                y0[at] = acc;
            } else if qi == 1 {
                y1[at] = acc;
            } else if qi == 2 {
                y2[at] = acc;
            } else {
                y3[at] = acc;
            }
            j += MIX_CUBE;
        }
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

/// Attention with Inkling's relative bias and sink (inkling.rs `layer`), a cube per (head, token) (the mask rounded to f16 when
/// `round`): the token at
/// position p = base + tok reads positions max(0, p + 1 - win)..=p (win 0: all), cached at t % wring. Its score of t is
/// q.k * scale + f16(bias(p - t) - sink), bias(dist) = rb[dist] . r (0 from `extent` on); the softmax has one more
/// logit of 0 (the sink). `s` is scratch of nt * n_head * span scores.
#[cube(launch_unchecked)]
pub fn ink_attend(q: &Array<f32>, kc: &Array<f32>, vc: &Array<f32>, r: &Array<f32>, rb: &Array<f32>, sk: &Array<f32>, s: &mut Array<f32>,
                  o: &mut Array<f32>, base: u32, nh: u32, nkv: u32, hd: u32, nrel: u32, extent: u32, win: u32, wring: u32, span: u32,
                  scale: f32, #[comptime] round: bool) {
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
        let mut mask = b - sink;
        if round {
            mask = f16r(mask);
        }
        let v = acc * scale + mask;
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

/// K and V rows for positions base.. into their caches of `wring` positions (two ink_kv_put in one launch).
#[cube(launch_unchecked)]
pub fn ink_kv_put2(k: &Array<f32>, v: &Array<f32>, kc: &mut Array<f32>, vc: &mut Array<f32>, base: u32, row: u32, wring: u32, n: u32) {
    let j = (CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X) * CUBE + UNIT_POS;
    if j < 2 * n {
        let i = j % n;
        let at = (((base + i / row) % wring) * row + i % row) as usize;
        if j < n {
            kc[at] = k[i as usize];
        } else {
            vc[at] = v[i as usize];
        }
    }
}

/// y[tok * dim + j] = the chosen experts' outputs weighted and added in routing order (moe_combine's sum, written
/// rather than added).
#[cube(launch_unchecked)]
pub fn ink_combine(slots: &Array<f32>, routes: &Array<f32>, y: &mut Array<f32>, k: u32, dim: u32, n: u32) {
    let i = (CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X) * CUBE + UNIT_POS;
    if i < n {
        let tok = i / dim;
        let j = i % dim;
        let mut acc = f32::new(0.0);
        for s in 0..k {
            acc += routes[(tok * 2 * k + k + s) as usize] * slots[((tok * k + s) * dim + j) as usize];
        }
        y[i as usize] = acc;
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

/// Inkling's router per token (a warp per token): sigmoid probabilities, the `k` best by probability plus the
/// selection bias (ties to the lower index, as a stable sort), weights p / (sum of the chosen p + 1e-20) * ws.
/// -> out[tok * 2k ..]: the indices (f32), then the weights (route_topk's layout, for moe_combine).
#[cube(launch_unchecked)]
pub fn ink_route(logits: &Array<f32>, bias: &Array<f32>, out: &mut Array<f32>, n_e: u32, k: u32, ws: f32, #[comptime] biased: bool) {
    let tok = CUBE_POS_X;
    let lane = UNIT_POS;
    let base = tok * n_e;
    let ob = tok * 2 * k;
    let mut sel = SharedMemory::<f32>::new(ROUTE_EXPERTS as usize);
    let mut bv = SharedMemory::<f32>::new(32usize);
    let mut bi = SharedMemory::<u32>::new(32usize);
    let mut e = lane;
    while e < n_e {
        let mut v = sigm(logits[(base + e) as usize]);
        if biased {
            v += bias[e as usize];
        }
        sel[e as usize] = v;
        e += 32;
    }
    sync_cube();
    let mut wsum = f32::new(0.0);
    for s in 0..k {
        let mut best = f32::new(-3.0e38);
        let mut be = n_e;
        let mut e = lane;
        while e < n_e {
            if sel[e as usize] > best {
                best = sel[e as usize];
                be = e;
            }
            e += 32;
        }
        bv[lane as usize] = best;
        bi[lane as usize] = be;
        sync_cube();
        let mut stride = u32::new(16);
        while stride > 0 {
            if lane < stride {
                let ov = bv[(lane + stride) as usize];
                let oi = bi[(lane + stride) as usize];
                let mv = bv[lane as usize];
                let mi = bi[lane as usize];
                if ov > mv || (ov == mv && oi < mi) {
                    bv[lane as usize] = ov;
                    bi[lane as usize] = oi;
                }
            }
            sync_cube();
            stride /= 2;
        }
        let pick = bi[0usize];
        let p = sigm(logits[(base + pick) as usize]);
        wsum += p;
        sync_cube();
        if lane == 0 {
            out[(ob + s) as usize] = f32::cast_from(pick);
            out[(ob + k + s) as usize] = p;
            sel[pick as usize] = f32::new(-3.0e38);
        }
        sync_cube();
    }
    if lane == 0 {
        let d = wsum + f32::new(1.0e-20);
        for s in 0..k {
            out[(ob + k + s) as usize] = out[(ob + k + s) as usize] / d * ws;
        }
    }
}

/// y[t * rows + r] = W[r] . x[t] for an F32 matrix held as a buffer (the router, which stays F32): a warp per row,
/// t = CUBE_POS_Z.
#[cube(launch_unchecked)]
pub fn ink_mv_f32(w: &Array<f32>, x: &Array<f32>, y: &mut Array<f32>, rows: u32, cols: u32, #[comptime] plane: bool) {
    let lane = UNIT_POS % 32;
    let row = (CUBE_POS_X + CUBE_POS_Y * CUBE_COUNT_X) * MV_ROWS + UNIT_POS / 32;
    let t = CUBE_POS_Z;
    let mut acc = f32::new(0.0);
    if row < rows {
        let mut j = lane;
        while j < cols {
            acc += w[(row * cols + j) as usize] * x[(t * cols + j) as usize];
            j += 32;
        }
    }
    let total = warp_sum(acc, plane, comptime![MV_ROWS * 32]);
    if lane == 0 && row < rows {
        y[(t * rows + row) as usize] = total;
    }
}

/// A fused gate/up matrix (rows 2ff: the gates, then the ups) and SiTU in one decode kernel: y[t * ff + r] =
/// beta tanh(g / beta) sigmoid(g) * lb tanh(u / lb), g = W[r] . x[t], u = W[ff + r] . x[t]; a warp per row r.
#[cube(launch_unchecked)]
pub fn ink_mv_situ(w: &Array<crate::kernels::V4>, wd: &Array<u32>, x: &Array<F4>, y: &mut Array<f32>, ff: u32, cols: u32, beta: f32, lb: f32,
                   #[comptime] k: u32, #[comptime] plane: bool) {
    let lane = UNIT_POS % 32;
    let row = (CUBE_POS_X + CUBE_POS_Y * CUBE_COUNT_X) * MV_ROWS + UNIT_POS / 32;
    let t = CUBE_POS_Z;
    let mut pg = f32::new(0.0);
    let mut pu = f32::new(0.0);
    if row < ff {
        pg = row_part(w, wd, x, row, cols, t, lane, k);
        pu = row_part(w, wd, x, row + ff, cols, t, lane, k);
    }
    let g = warp_sum(pg, plane, comptime![MV_ROWS * 32]);
    let u = warp_sum(pu, plane, comptime![MV_ROWS * 32]);
    if lane == 0 && row < ff {
        let mut uu = u;
        if lb > f32::new(0.0) {
            uu = lb * tanh_p(u / lb);
        }
        y[(t * ff + row) as usize] = beta * tanh_p(g / beta) * sigm(g) * uu;
    }
}

/// Expert `ex`'s part of row `row` (row_part), its matrix one of eight (the branch is the same for a whole cube).
#[cube]
#[allow(clippy::too_many_arguments)]
fn part8(ex: u32, w0: &Array<crate::kernels::V4>, d0: &Array<u32>, w1: &Array<crate::kernels::V4>, d1: &Array<u32>, w2: &Array<crate::kernels::V4>,
         d2: &Array<u32>, w3: &Array<crate::kernels::V4>, d3: &Array<u32>, w4: &Array<crate::kernels::V4>, d4: &Array<u32>,
         w5: &Array<crate::kernels::V4>, d5: &Array<u32>, w6: &Array<crate::kernels::V4>, d6: &Array<u32>, w7: &Array<crate::kernels::V4>,
         d7: &Array<u32>, x: &Array<F4>, row: u32, cols: u32, t: u32, lane: u32, #[comptime] k: u32) -> f32 {
    let mut v = f32::new(0.0);
    if ex == 0 {
        v = row_part(w0, d0, x, row, cols, t, lane, k);
    } else if ex == 1 {
        v = row_part(w1, d1, x, row, cols, t, lane, k);
    } else if ex == 2 {
        v = row_part(w2, d2, x, row, cols, t, lane, k);
    } else if ex == 3 {
        v = row_part(w3, d3, x, row, cols, t, lane, k);
    } else if ex == 4 {
        v = row_part(w4, d4, x, row, cols, t, lane, k);
    } else if ex == 5 {
        v = row_part(w5, d5, x, row, cols, t, lane, k);
    } else if ex == 6 {
        v = row_part(w6, d6, x, row, cols, t, lane, k);
    } else {
        v = row_part(w7, d7, x, row, cols, t, lane, k);
    }
    v
}

/// Up to eight experts' fused gate/up products of one token and SiTU in one launch: expert ex (the cubes' ex-th share)
/// writes y[ex * ff + r] (ink_mv_situ's values).
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn ink_mv8_situ(w0: &Array<crate::kernels::V4>, d0: &Array<u32>, w1: &Array<crate::kernels::V4>, d1: &Array<u32>, w2: &Array<crate::kernels::V4>,
                    d2: &Array<u32>, w3: &Array<crate::kernels::V4>, d3: &Array<u32>, w4: &Array<crate::kernels::V4>, d4: &Array<u32>,
                    w5: &Array<crate::kernels::V4>, d5: &Array<u32>, w6: &Array<crate::kernels::V4>, d6: &Array<u32>, w7: &Array<crate::kernels::V4>,
                    d7: &Array<u32>, x: &Array<F4>, y: &mut Array<f32>, n: u32, ff: u32, cols: u32, beta: f32, lb: f32, #[comptime] k: u32,
                    #[comptime] plane: bool) {
    let lane = UNIT_POS % 32;
    let per = ff.div_ceil(MV_ROWS);
    let cube = CUBE_POS_X + CUBE_POS_Y * CUBE_COUNT_X;
    let ex = cube / per;
    let row = (cube % per) * MV_ROWS + UNIT_POS / 32;
    let mut pg = f32::new(0.0);
    let mut pu = f32::new(0.0);
    if ex < n && row < ff {
        pg = part8(ex, w0, d0, w1, d1, w2, d2, w3, d3, w4, d4, w5, d5, w6, d6, w7, d7, x, row, cols, 0, lane, k);
        pu = part8(ex, w0, d0, w1, d1, w2, d2, w3, d3, w4, d4, w5, d5, w6, d6, w7, d7, x, row + ff, cols, 0, lane, k);
    }
    let g = warp_sum(pg, plane, comptime![MV_ROWS * 32]);
    let u = warp_sum(pu, plane, comptime![MV_ROWS * 32]);
    if lane == 0 && ex < n && row < ff {
        let mut uu = u;
        if lb > f32::new(0.0) {
            uu = lb * tanh_p(u / lb);
        }
        y[(ex * ff + row) as usize] = beta * tanh_p(g / beta) * sigm(g) * uu;
    }
}

/// Up to eight experts' down products in one launch: expert ex reads x's row ex (its SiTU output) and writes its
/// output row of `rows` at y[o_ex ..] (its (token, slot) row).
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn ink_mv8_down(w0: &Array<crate::kernels::V4>, d0: &Array<u32>, w1: &Array<crate::kernels::V4>, d1: &Array<u32>, w2: &Array<crate::kernels::V4>,
                    d2: &Array<u32>, w3: &Array<crate::kernels::V4>, d3: &Array<u32>, w4: &Array<crate::kernels::V4>, d4: &Array<u32>,
                    w5: &Array<crate::kernels::V4>, d5: &Array<u32>, w6: &Array<crate::kernels::V4>, d6: &Array<u32>, w7: &Array<crate::kernels::V4>,
                    d7: &Array<u32>, x: &Array<F4>, y: &mut Array<f32>, offs: &Array<u32>, n: u32, rows: u32, cols: u32, #[comptime] k: u32,
                    #[comptime] plane: bool) {
    let lane = UNIT_POS % 32;
    let per = rows.div_ceil(MV_ROWS);
    let cube = CUBE_POS_X + CUBE_POS_Y * CUBE_COUNT_X;
    let ex = cube / per;
    let row = (cube % per) * MV_ROWS + UNIT_POS / 32;
    let mut acc = f32::new(0.0);
    if ex < n && row < rows {
        acc = part8(ex, w0, d0, w1, d1, w2, d2, w3, d3, w4, d4, w5, d5, w6, d6, w7, d7, x, row, cols, ex, lane, k);
    }
    let total = warp_sum(acc, plane, comptime![MV_ROWS * 32]);
    if lane == 0 && ex < n && row < rows {
        let o = offs[ex as usize];
        y[(o + row) as usize] = total;
    }
}

/// Attention's partial results per split of `len` positions (flash decoding), a cube of 4 warps per (head, split,
/// token): the scores of the split's positions as `ink_attend` computes them (a warp per position, its lanes over the
/// head's values), then the split's max, its sum of exponentials and its weighted sum of V: part[(row * splits + sp) *
/// (hd + 2)..] = [max, sum, hd values].
#[cube(launch_unchecked)]
pub fn ink_attend_part(q: &Array<f32>, kc: &Array<f32>, vc: &Array<f32>, r: &Array<f32>, rb: &Array<f32>, sk: &Array<f32>, part: &mut Array<f32>,
                       base: u32, nh: u32, nkv: u32, hd: u32, nrel: u32, extent: u32, win: u32, wring: u32, len: u32, splits: u32, scale: f32,
                       #[comptime] plane: bool, #[comptime] round: bool) {
    let h = CUBE_POS_X;
    let sp = CUBE_POS_Y;
    let tok = CUBE_POS_Z;
    let row = tok * nh + h;
    let p = base + tok;
    let mut first = u32::new(0);
    if win > 0 && p + 1 > win {
        first = p + 1 - win;
    }
    let cnt = p + 1 - first;
    let lo = sp * len;
    let mut n_here = u32::new(0);
    if cnt > lo {
        n_here = u32::min(len, cnt - lo);
    }
    let kh = h / (nh / nkv);
    let qo = row * hd;
    let ro = row * nrel;
    let sink = sk[row as usize];
    let mut qs = SharedMemory::<f32>::new(512usize);
    let mut sc = SharedMemory::<f32>::new(SPLIT_CHUNK as usize);
    let mut j = UNIT_POS;
    while j < hd {
        qs[j as usize] = q[(qo + j) as usize];
        j += CUBE;
    }
    sync_cube();
    let warp = UNIT_POS / 32;
    let lane = UNIT_POS % 32;
    for st in 0..len / 4 {
        let i = st * 4 + warp;
        let mut acc = f32::new(0.0);
        if i < n_here {
            let t = first + lo + i;
            let ko = ((t % wring) * nkv + kh) * hd;
            let mut j = lane;
            while j < hd {
                acc += qs[j as usize] * kc[(ko + j) as usize];
                j += 32;
            }
        }
        let tot = warp_sum(acc, plane, CUBE);
        if lane == 0 && i < n_here {
            let t = first + lo + i;
            let dist = p - t;
            let mut b = f32::new(0.0);
            if dist < extent {
                for jj in 0..nrel {
                    b += rb[(dist * nrel + jj) as usize] * r[(ro + jj) as usize];
                }
            }
            let mut mask = b - sink;
            if round {
                mask = f16r(mask);
            }
            sc[i as usize] = tot * scale + mask;
        }
    }
    sync_cube();
    let mut m = f32::new(-1.0e30);
    let mut i = UNIT_POS;
    while i < n_here {
        m = f32::max(m, sc[i as usize]);
        i += CUBE;
    }
    let m = cube_max(m);
    let mut l = f32::new(0.0);
    let mut i = UNIT_POS;
    while i < n_here {
        let e = f32::exp(sc[i as usize] - m);
        sc[i as usize] = e;
        l += e;
        i += CUBE;
    }
    let l = cube_sum(l);
    let pb = (row * splits + sp) * (hd + 2);
    let mut j = UNIT_POS;
    while j < hd {
        let mut acc = f32::new(0.0);
        for ii in 0..n_here {
            let t = first + lo + ii;
            acc += sc[ii as usize] * vc[(((t % wring) * nkv + kh) * hd + j) as usize];
        }
        part[(pb + 2 + j) as usize] = acc;
        j += CUBE;
    }
    if UNIT_POS == 0 {
        part[pb as usize] = m;
        part[(pb + 1) as usize] = l;
    }
}

/// The splits merged per row (a cube per query head of a token), with the sink's logit of 0: o = sum over splits of
/// e^(m_s - M) o_s / (e^-M + sum of e^(m_s - M) l_s).
#[cube(launch_unchecked)]
pub fn ink_attend_merge(part: &Array<f32>, o: &mut Array<f32>, splits: u32, hd: u32) {
    let row = CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X;
    let w = hd + 2;
    let mut mx = f32::new(0.0);
    for s in 0..splits {
        mx = f32::max(mx, part[((row * splits + s) * w) as usize]);
    }
    let mut total = f32::exp(-mx);
    for s in 0..splits {
        let b = (row * splits + s) * w;
        total += part[(b + 1) as usize] * f32::exp(part[b as usize] - mx);
    }
    let mut j = UNIT_POS;
    while j < hd {
        let mut acc = f32::new(0.0);
        for s in 0..splits {
            let b = (row * splits + s) * w;
            acc += f32::exp(part[b as usize] - mx) / total * part[(b + 2 + j) as usize];
        }
        o[(row * hd + j) as usize] = acc;
        j += CUBE;
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

    /// The depth mixes of `nq` (1 to 4) consecutive queries from queries[q0 * d ..] over the same states (see
    /// `ink_mix`): -> one buffer of nt * d per query.
    #[allow(clippy::too_many_arguments)]
    pub fn ink_mix_n(&self, st: &Buf, off: usize, stride: usize, ns: usize, cur: &Buf, queries: &Buf, q0: usize, nq: usize, d: usize, nt: usize) -> Vec<Buf> {
        assert!(ns < MIX_STATES as usize && (1..=4).contains(&nq), "ink_mix_n: {ns} states, {nq} queries");
        if !self.planes() {
            return (0..nq).map(|i| self.ink_mix(st, off, stride, ns, cur, queries, q0 + i, d, nt)).collect();
        }
        stats::launch(1);
        let ys: Vec<Buf> = (0..nq).map(|_| self.scratch(nt * d)).collect();
        let y = |i: usize| &ys[i.min(nq - 1)];
        with_client!(self, c, R, unsafe {
            ink_mix_q::launch_unchecked::<R>(c, CubeCount::Static(nt as u32, 1, 1), CubeDim::new_1d(MIX_CUBE), st.arg(), cur.arg(), queries.arg(),
                y(0).arg(), y(1).arg(), y(2).arg(), y(3).arg(), off as u32, stride as u32, ns as u32, (q0 * d) as u32, nq as u32, d as u32);
        });
        ys
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

    /// Whether the device's planes are 32 units wide with plane operations (the warp reductions take them).
    fn planes(&self) -> bool {
        with_client!(self, c, R, {
            let p = c.properties();
            p.hardware.plane_size_min == 32 && p.hardware.plane_size_max == 32 && p.features.plane.contains(cubecl::ir::features::Plane::Ops)
        })
    }

    /// Inkling attention for `nt` tokens at positions base.. (see the kernels): flash decoding, splits of the
    /// positions then their merge; the one-cube-per-head kernel for heads over 512 values. -> nt * n_head * hd values.
    #[allow(clippy::too_many_arguments)]
    pub fn ink_attend(&self, q: &Buf, kc: &Buf, vc: &Buf, r: &Buf, rb: &Buf, sinks: &Buf, base: usize, nt: usize, n_head: usize, n_kv: usize, hd: usize,
                      n_rel: usize, extent: usize, win: usize, wring: usize, round: bool) -> Buf {
        let n = base + nt;
        let span = if win > 0 { n.min(win) } else { n };
        let scale = 1.0 / (hd as f32).sqrt();
        if hd <= 512 && std::env::var_os("AM2_INK_ATTEND_ONE").is_none() {
            stats::launch(2);
            let len = if nt == 1 { SPLIT_ONE } else { SPLIT_CHUNK } as usize;
            let splits = span.div_ceil(len);
            let part = self.scratch(nt * n_head * splits * (hd + 2));
            let o = self.scratch(nt * n_head * hd);
            let plane = self.planes();
            with_client!(self, c, R, unsafe {
                ink_attend_part::launch_unchecked::<R>(c, CubeCount::Static(n_head as u32, splits as u32, nt as u32), CubeDim::new_1d(CUBE), q.arg(),
                    kc.arg(), vc.arg(), r.arg(), rb.arg(), sinks.arg(), part.arg(), base as u32, n_head as u32, n_kv as u32, hd as u32, n_rel as u32,
                    extent as u32, win as u32, wring as u32, len as u32, splits as u32, scale, plane, round);
                ink_attend_merge::launch_unchecked::<R>(c, rows_grid(nt * n_head), CubeDim::new_1d(CUBE), part.arg(), o.arg(), splits as u32, hd as u32);
            });
            return o;
        }
        stats::launch(1);
        let s = self.scratch(nt * n_head * span);
        let o = self.scratch(nt * n_head * hd);
        with_client!(self, c, R, unsafe {
            ink_attend::launch_unchecked::<R>(c, CubeCount::Static(n_head as u32, nt as u32, 1), CubeDim::new_1d(CUBE), q.arg(), kc.arg(), vc.arg(),
                r.arg(), rb.arg(), sinks.arg(), s.arg(), o.arg(), base as u32, n_head as u32, n_kv as u32, hd as u32, n_rel as u32, extent as u32,
                win as u32, wring as u32, span as u32, scale, round);
        });
        o
    }

    /// W x for `tokens` vectors, W an F32 matrix held as a buffer of rows x cols.
    pub fn ink_mv_f32(&self, w: &Buf, x: &Buf, rows: usize, cols: usize) -> Buf {
        stats::launch(1);
        let tokens = x.len / cols;
        let y = self.scratch(tokens * rows);
        let cubes = (rows as u32).div_ceil(MV_ROWS);
        let plane = self.planes();
        with_client!(self, c, R, unsafe {
            ink_mv_f32::launch_unchecked::<R>(c, CubeCount::Static(cubes.min(65535), cubes.div_ceil(cubes.min(65535)), tokens as u32),
                CubeDim::new_1d(MV_ROWS * 32), w.arg(), x.arg(), y.arg(), rows as u32, cols as u32, plane);
        });
        y
    }

    /// SiTU of a fused gate/up matrix's product with `tokens` vectors: one decode kernel for a fast matrix and a few
    /// tokens, else the product (the tiled kernel for a chunk) and then `ink_situ`. -> tokens * ff values.
    pub fn ink_gate_up_situ(&self, w: &Weights, x: &Buf, tokens: usize, beta: f32, lb: f32) -> Buf {
        let ff = w.rows as usize / 2;
        if !self.fused(&[w], tokens as u32) {
            return self.ink_situ(&self.mv_tokens(w, x, tokens as u32), ff, beta, lb);
        }
        stats::launch(1);
        let y = self.scratch(tokens * ff);
        let cubes = (ff as u32).div_ceil(MV_ROWS);
        with_client!(self, c, R, unsafe {
            ink_mv_situ::launch_unchecked::<R>(c, CubeCount::Static(cubes.min(65535), cubes.div_ceil(cubes.min(65535)), tokens as u32),
                CubeDim::new_1d(MV_ROWS * 32), w.v4(), w.sc(), ArrayArg::from_raw_parts(x.handle.cube(), x.len / 4), y.arg(), ff as u32, w.cols,
                beta, lb, w.geometry.2, w.plane);
        });
        y
    }

    /// Whether `ink_experts8` takes these experts' matrices: fast ones of one kind and width each (decode kernels).
    pub fn ink_experts8_take(&self, gus: &[&Weights], dns: &[&Weights]) -> bool {
        !gus.is_empty() && gus.len() <= 8 && gus.len() == dns.len() && self.fused(gus, 1) && self.fused(dns, 1)
            && gus.iter().all(|w| w.rows == gus[0].rows) && dns.iter().all(|w| w.rows == dns[0].rows)
    }

    /// Up to eight experts of one token in two launches (`ink_experts8_take` must hold): their gate/up products and
    /// SiTU, then their down products, expert i's output written at slots[offs[i] ..].
    #[allow(clippy::too_many_arguments)]
    pub fn ink_experts8(&self, gus: &[&Weights], dns: &[&Weights], x: &Buf, slots: &Buf, offs: &[usize], beta: f32, lb: f32) {
        stats::launch(2);
        let mut o: Vec<u32> = offs.iter().map(|&v| v as u32).collect();
        o.resize(8, 0);
        stats::h2d(32);
        let offs = with_client!(self, c, R, c.create_from_slice(u32::as_bytes(&o)));
        let n = gus.len();
        let (ff, cols, rows) = (gus[0].rows / 2, gus[0].cols, dns[0].rows);
        let mid = self.scratch(n * ff as usize);
        let g = |i: usize| gus[i.min(n - 1)];
        let d = |i: usize| dns[i.min(n - 1)];
        let grid = |cubes: u32| CubeCount::Static(cubes.min(65535), cubes.div_ceil(cubes.min(65535)), 1);
        with_client!(self, c, R, unsafe {
            ink_mv8_situ::launch_unchecked::<R>(c, grid(ff.div_ceil(MV_ROWS) * n as u32), CubeDim::new_1d(MV_ROWS * 32),
                g(0).v4(), g(0).sc(), g(1).v4(), g(1).sc(), g(2).v4(), g(2).sc(), g(3).v4(), g(3).sc(), g(4).v4(), g(4).sc(), g(5).v4(), g(5).sc(),
                g(6).v4(), g(6).sc(), g(7).v4(), g(7).sc(), ArrayArg::from_raw_parts(x.handle.cube(), x.len / 4), mid.arg(), n as u32, ff, cols,
                beta, lb, gus[0].geometry.2, gus[0].plane);
            ink_mv8_down::launch_unchecked::<R>(c, grid(rows.div_ceil(MV_ROWS) * n as u32), CubeDim::new_1d(MV_ROWS * 32),
                d(0).v4(), d(0).sc(), d(1).v4(), d(1).sc(), d(2).v4(), d(2).sc(), d(3).v4(), d(3).sc(), d(4).v4(), d(4).sc(), d(5).v4(), d(5).sc(),
                d(6).v4(), d(6).sc(), d(7).v4(), d(7).sc(), ArrayArg::from_raw_parts(mid.handle.cube(), mid.len / 4), slots.arg(),
                ArrayArg::from_raw_parts(offs.clone(), 8), n as u32, rows, ff, dns[0].geometry.2, dns[0].plane);
        });
    }

    /// K and V for positions base.. into their caches, one launch.
    #[allow(clippy::too_many_arguments)]
    pub fn ink_kv_put2(&self, k: &Buf, v: &Buf, kc: &Buf, vc: &Buf, base: usize, row: usize, wring: usize) {
        stats::launch(1);
        with_client!(self, c, R, unsafe {
            ink_kv_put2::launch_unchecked::<R>(c, flat(2 * k.len), CubeDim::new_1d(CUBE), k.arg(), v.arg(), kc.arg(), vc.arg(), base as u32, row as u32,
                wring as u32, k.len as u32);
        });
    }

    /// The routed experts' outputs (`slots`: tokens * k rows of `dim`) weighted by `routes` and summed in routing order.
    pub fn ink_combine(&self, slots: &Buf, routes: &Buf, k: usize, dim: usize) -> Buf {
        stats::launch(1);
        let n = routes.len / (2 * k) * dim;
        let y = self.scratch(n);
        with_client!(self, c, R, unsafe {
            ink_combine::launch_unchecked::<R>(c, flat(n), CubeDim::new_1d(CUBE), slots.arg(), routes.arg(), y.arg(), k as u32, dim as u32, n as u32);
        });
        y
    }

    /// A x and B x (the same width and kind), one decode launch when the fast kernels take both. -> (A x, B x).
    pub fn ink_mv2(&self, a: &Weights, b: &Weights, x: &Buf, tokens: usize) -> (Buf, Buf) {
        if !self.fused(&[a, b], tokens as u32) {
            return (self.mv_tokens(a, x, tokens as u32), self.mv_tokens(b, x, tokens as u32));
        }
        stats::launch(1);
        let (ya, yb) = (self.scratch(tokens * a.rows as usize), self.scratch(tokens * b.rows as usize));
        with_client!(self, c, R, unsafe {
            crate::kernels::mv3::launch_unchecked::<R>(c, mv_count(a.rows, b.rows, 0, tokens as u32), CubeDim::new_1d(MV_ROWS * 32), a.v4(), a.sc(),
                b.v4(), b.sc(), b.v4(), b.sc(), ArrayArg::from_raw_parts(x.handle.cube(), x.len / 4), ya.arg(), yb.arg(), yb.arg(), a.rows, b.rows,
                0, 0, a.cols, a.geometry.2, a.plane);
        });
        (ya, yb)
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
        assert!(n_e <= ROUTE_EXPERTS as usize, "ink_route: {n_e} experts (at most {ROUTE_EXPERTS})");
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
