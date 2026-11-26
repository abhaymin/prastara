//! Inkling 2 on the device graph (docs/INKLING-PLAN.md I8): a whole token, or a prompt chunk, through
//! `inkling2` / `inkling2_loop` as one queue of kernels on the device that holds its attention and dense matrices,
//! instead of the matrix-by-matrix path's round trip per product (about a thousand per token on the real file).
//!
//! On the device: the depth mixing of the residual streams (strata-gpu ink.rs `ink_mix`), the loop's messages and
//! controllers, the short convolutions with their state (each cache slot's on the device), clipping and the log scale,
//! attention with the relative bias and the sink over each slot's cache (a ring of the window plus a chunk for the
//! windowed layers), the gate, the sigmoid router with its selection bias, SiTU, and the soft cap. Every slot (a layer
//! in one pass of the loop) has its own K, V and convolution state, as model/inkling.rs `InkSlot`.
//!
//! The experts are where placement put them (gpu.rs, the expert profile's order, a Kosha pack's file order): those on
//! the graph's device run there, those on another device run there with only the activations crossing (one read per
//! layer and device), the rest on the CPU (decode and prompt alike) with only the activations crossing. The one wait
//! per layer reads the router's choice (and, when some experts are off the device, the layer's normed input with it
//! in the same read).
//!
//! K, V and the attention mask are rounded to f16 as the CPU path keeps them, so the logits are the CPU path's within
//! float rounding (the tests below). Activations are not rounded to 8 bits as the CPU path rounds them for quantized
//! products (ggml's way).

use crate::graph::Graph;
use crate::model::inkling::{f32s, InkCtrl, InkLayer, Inkling};
use crate::model::{Model, QMat, State};
use strata_gpu::{Buf, Gpu, Weights};

/// The most tokens one run takes: a windowed slot's ring holds the window and one chunk.
const CHUNK: usize = 128;

/// A matrix on the graph's device: placement's copy, or one uploaded here (the small ones placement leaves out).
enum Mat<'a> {
    On(&'a Weights),
    Own(Weights),
}

impl std::ops::Deref for Mat<'_> {
    type Target = Weights;
    fn deref(&self) -> &Weights {
        match self {
            Mat::On(w) => w,
            Mat::Own(w) => w,
        }
    }
}

/// Where an expert's (gate/up, down) are.
enum Where<'a> {
    Here(&'a Weights, &'a Weights),
    Other(usize, &'a Weights, &'a Weights),
    Cpu,
}

struct GCtrl<'a> {
    down: Mat<'a>,
    gain: Mat<'a>,
    forget: Mat<'a>,
}

struct GLayer<'a> {
    l: &'a InkLayer,
    wq: &'a Weights,
    wk: &'a Weights,
    wv: &'a Weights,
    wo: &'a Weights,
    gate: &'a Weights,
    rel: &'a Weights,
    k_proj: Option<&'a Weights>,
    v_proj: Option<&'a Weights>,
    sink: Mat<'a>,
    q_clip: Buf,
    rel_clip: Buf,
    rel_bias: Buf,
    k_conv: Buf,
    v_conv: Buf,
    post_conv: Buf,
    ffn_conv: Buf,
    norms: Vec<Buf>,
    res_query: Option<Buf>,
    ffn_query: Buf,
    ffn_norm: Buf,
    router: Mat<'a>,
    probs_b: Option<Buf>,
    experts: Vec<Where<'a>>,
    /// a looped layer's message queries and controllers (Q, K, V, FFN)
    msg_query: Option<Buf>,
    ctrl: Vec<GCtrl<'a>>,
}

/// A cache slot on the device: K and V (`wring` positions), and the convolutions' state.
struct Slot {
    kc: Buf,
    vc: Buf,
    hist: Buf,
    wring: usize,
}

pub struct InkGraph<'a> {
    g: &'static Gpu,
    dev: usize,
    ink: &'a Inkling,
    layers: Vec<GLayer<'a>>,
    slots: Vec<Slot>,
    output: &'a Weights,
    out_norm: Buf,
    out_query: Buf,
    max_ctx: usize,
    max_blocks: usize,
}

fn on_device<'a>(q: &'a QMat, dev: &mut Option<usize>) -> Result<&'a Weights, String> {
    let d = q.gpu.as_ref().ok_or("--graph needs every attention and dense matrix on a device: give --devices room for them")?;
    match *dev {
        Some(x) if x != d.device() => Err("--graph needs every attention and dense matrix on one device".to_string()),
        _ => {
            *dev = Some(d.device());
            Ok(d.weights())
        }
    }
}

/// The matrix on device `dev`: placement's copy there, else uploaded in its encoding (or as F32 when the device has
/// no kernel for it).
fn small<'a>(g: &Gpu, q: &'a QMat, dev: usize) -> Result<Mat<'a>, String> {
    if let Some(d) = q.gpu.as_ref().filter(|d| d.device() == dev) {
        return Ok(Mat::On(d.weights()));
    }
    match g.upload(q.ty(), q.data, q.rows, q.cols) {
        Ok(w) => Ok(Mat::Own(w)),
        Err(_) => {
            let b: Vec<u8> = f32s(q).iter().flat_map(|v| v.to_le_bytes()).collect();
            g.upload("F32", &b, q.rows, q.cols).map(Mat::Own)
        }
    }
}

/// The slots' ring sizes: the window and a chunk for a windowed layer, the context for a full one.
fn ring(ink: &Inkling, slot_layer: usize, max_ctx: usize) -> usize {
    if ink.layers[slot_layer].swa { (ink.n_swa + CHUNK).min(max_ctx) } else { max_ctx }
}

/// Each slot's physical layer.
fn slot_layers(ink: &Inkling) -> Vec<usize> {
    let n = ink.layers.len();
    let mut v: Vec<usize> = (0..n).collect();
    if let Some((s, e, p)) = ink.core {
        for _ in 1..p {
            v.extend(s..e);
        }
    }
    v
}

/// Why the graph cannot run this Inkling model, if it cannot.
pub fn ink_refusal(m: &Model) -> Option<String> {
    let ink = m.ink.as_ref()?;
    if ink.layers.iter().any(|l| l.n_kv == 0 || l.n_head % l.n_kv != 0) {
        return Some("--graph: this Inkling's query heads are not whole groups of its KV heads".into());
    }
    None
}

/// The bytes the graph's slots take on its device for `max_ctx` tokens (K and V f32, the convolutions' state), what
/// placement keeps free for them.
pub fn ink_kv_need(m: &Model, max_ctx: usize) -> u64 {
    let Some(ink) = m.ink.as_ref() else { return 0 };
    let past = ink.n_conv - 1;
    slot_layers(ink).iter().map(|&l| {
        let ly = &ink.layers[l];
        let kvd = (ly.hd * ly.n_kv) as u64;
        2 * kvd * ring(ink, l, max_ctx) as u64 * 4 + (2 * kvd + 2 * m.n_embd as u64) * past as u64 * 4
    }).sum::<u64>()
        // the small matrices uploaded here (sink, router, controllers, tables), roughly
        + ink.layers.iter().map(|l| (l.router.data.len() + l.sink.data.len() + l.rel_bias.data.len()) as u64 * 2).sum::<u64>()
        + ink.loops.iter().flat_map(|lp| lp.ctrl.iter()).map(|c| (c.down.data.len() + c.gain.data.len() + c.forget.data.len()) as u64 * 2).sum::<u64>()
}

impl<'a> InkGraph<'a> {
    pub fn new(m: &'a Model, max_ctx: usize) -> Result<InkGraph<'a>, String> {
        let ink: &Inkling = m.ink.as_deref().ok_or("not an Inkling model")?;
        if let Some(why) = ink_refusal(m) {
            return Err(why);
        }
        let mut dev = None;
        let output = on_device(&m.output, &mut dev)?;
        for ly in &ink.layers {
            for q in [&ly.wq, &ly.wk, &ly.wv, &ly.wo, &ly.gate, &ly.rel].into_iter().chain(ly.k_proj.iter()).chain(ly.v_proj.iter()) {
                on_device(q, &mut dev)?;
            }
        }
        let dev = dev.expect("the output's device");
        let g = crate::gpu::device(dev);
        if !g.has_ink_kernels() || !g.has_moe_kernels() {
            return Err("--graph: Inkling's kernels are not on the native Metal backend yet (AM2_METAL_NATIVE=0 runs them on wgpu)".into());
        }
        let need = ink_kv_need(m, max_ctx);
        if let Some(room) = crate::gpu::ROOM.get().and_then(|r| r.get(dev)) {
            if need > *room {
                let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
                return Err(format!("Inkling's caches for {max_ctx} tokens need {:.2} GiB on the device and its --devices budget has {:.2} GiB left \
                                    (a smaller --max-context or a larger budget)", gib(need), gib(*room)));
            }
        }
        let w = |q: &'a QMat| q.gpu.as_ref().expect("checked above").weights();
        let mut layers = Vec::with_capacity(ink.layers.len());
        for (il, ly) in ink.layers.iter().enumerate() {
            let lp = ink.core.filter(|(s, e, _)| (*s..*e).contains(&il)).map(|(s, _, _)| &ink.loops[il - s]);
            let ctrl = lp.map_or(Ok(vec![]), |lp| lp.ctrl.iter().map(|c: &'a InkCtrl| Ok(GCtrl {
                down: small(g, &c.down, dev)?, gain: small(g, &c.gain, dev)?, forget: small(g, &c.forget, dev)?,
            })).collect::<Result<Vec<_>, String>>())?;
            let experts = (0..ly.gate_up.0.len()).map(|e| match (ly.gate_up.0[e].gpu.as_ref(), ly.down.0[e].gpu.as_ref()) {
                (Some(a), Some(b)) if a.device() == dev && b.device() == dev => Where::Here(a.weights(), b.weights()),
                (Some(a), Some(b)) if a.device() == b.device() => Where::Other(a.device(), a.weights(), b.weights()),
                _ => Where::Cpu,
            }).collect();
            layers.push(GLayer {
                l: ly, wq: w(&ly.wq), wk: w(&ly.wk), wv: w(&ly.wv), wo: w(&ly.wo), gate: w(&ly.gate), rel: w(&ly.rel),
                k_proj: ly.k_proj.as_ref().map(w), v_proj: ly.v_proj.as_ref().map(w),
                sink: small(g, &ly.sink, dev)?,
                q_clip: g.buffer(&ly.q_clip), rel_clip: g.buffer(&ly.rel_clip), rel_bias: g.buffer(&f32s(&ly.rel_bias)),
                k_conv: g.buffer(&ly.k_conv), v_conv: g.buffer(&ly.v_conv), post_conv: g.buffer(&ly.post_conv), ffn_conv: g.buffer(&ly.ffn_conv),
                norms: ly.norms.iter().map(|n| g.buffer(n)).collect(),
                res_query: ly.res_query.as_ref().map(|q| g.buffer(q)), ffn_query: g.buffer(&ly.ffn_query), ffn_norm: g.buffer(&ly.ffn_norm),
                router: small(g, &ly.router, dev)?,
                probs_b: ly.probs_b.as_ref().map(|b| g.buffer(b)),
                experts,
                msg_query: lp.map(|lp| g.buffer(&lp.msg_query)),
                ctrl,
            });
        }
        let past = ink.n_conv - 1;
        let slots = slot_layers(ink).iter().map(|&l| {
            let ly = &ink.layers[l];
            let (kvd, wring) = (ly.hd * ly.n_kv, ring(ink, l, max_ctx));
            Slot { kc: g.kv_cache(kvd * wring), vc: g.kv_cache(kvd * wring), hist: g.ink_zeros((2 * kvd + 2 * m.n_embd) * past), wring }
        }).collect();
        // the most block outputs a token keeps: one per block start in the prelude, the core's last pass and after it
        let n = ink.layers.len();
        let max_blocks = (0..n).filter(|l| l % ink.block == 0).count() + 1;
        Ok(InkGraph { g, dev, ink, layers, slots, output, out_norm: g.buffer(&m.out_norm), out_query: g.buffer(&ink.out_query), max_ctx, max_blocks })
    }

    /// How many routed experts are on the graph's device, and how many there are.
    pub fn experts_on(&self) -> (usize, usize) {
        self.layers.iter().flat_map(|l| l.experts.iter()).fold((0, 0), |(a, b), e| (a + matches!(e, Where::Here(..)) as usize, b + 1))
    }

    /// One token: the logits of the next one.
    pub fn forward(&self, m: &Model, st: &mut State, token: u32) -> Vec<f32> {
        self.run(m, st, &[self.ink.embed(m, token)], false).pop().expect("a token")
    }

    /// One input row (a picture's, from the encoder) in place of a token: no token norm (Model::forward_embd).
    pub fn forward_embd(&self, m: &Model, st: &mut State, row: Vec<f32>) -> Vec<f32> {
        self.run(m, st, &[row], false).pop().expect("a row")
    }

    /// A prompt chunk: -> the logits after its last token.
    pub fn forward_batch(&self, m: &Model, st: &mut State, tokens: &[u32]) -> Vec<f32> {
        self.chunks(m, st, tokens, false).pop().expect("a token")
    }

    /// A prompt chunk: -> the logits after each token.
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn forward_batch_all(&self, m: &Model, st: &mut State, tokens: &[u32]) -> Vec<Vec<f32>> {
        self.chunks(m, st, tokens, true)
    }

    /// Runs of at most CHUNK tokens whose attention scores (tokens x heads x positions, f32) stay within 128 MiB.
    fn chunks(&self, m: &Model, st: &mut State, tokens: &[u32], all: bool) -> Vec<Vec<f32>> {
        let heads = self.ink.layers.iter().map(|l| l.n_head).max().unwrap_or(1);
        let mut outs = Vec::new();
        let mut i = 0;
        while i < tokens.len() {
            let mut n = (tokens.len() - i).min(CHUNK);
            while n > 1 && n * heads * (st.pos + n) * 4 > 128 << 20 {
                n /= 2;
            }
            let rows: Vec<Vec<f32>> = tokens[i..i + n].iter().map(|&t| self.ink.embed(m, t)).collect();
            let mut got = self.run(m, st, &rows, all);
            i += n;
            if all || i == tokens.len() {
                outs.append(&mut got);
            }
        }
        outs
    }

    /// The rows (embeddings after the token norm, or pictures' rows) through the model at positions st.pos..:
    /// -> the logits after each (`all`) or after the last.
    fn run(&self, m: &Model, st: &mut State, rows: &[Vec<f32>], all: bool) -> Vec<Vec<f32>> {
        let (g, ink) = (self.g, self.ink);
        let base = st.pos;
        let nt = rows.len();
        assert!(nt <= CHUNK, "--graph: {nt} rows in one run (at most {CHUNK})");
        assert!(base + nt <= self.max_ctx, "--graph: {} tokens are past the context of {}", base + nt, self.max_ctx);
        let d = m.n_embd;
        if base == 0 {
            // a new conversation: the convolutions start from zeros (the caches are written before they are read)
            for s in &self.slots {
                g.ink_fill(&s.hist, 0.0);
            }
        }
        let x = g.buffer(&rows.iter().flatten().copied().collect::<Vec<f32>>());
        // the full layers' log scale per position
        let taus = g.buffer(&(0..nt).map(|i| 1.0 + ink.alpha * crate::model::ln_f32(((base + i + 1) as f32 / ink.floor).max(1.0))).collect::<Vec<f32>>());
        let blk = g.scratch(self.max_blocks * nt * d);
        let mut nb = 0usize;
        let n_phys = ink.layers.len();
        let (start, end, passes) = ink.core.unwrap_or((n_phys, n_phys, 1));
        let cx = Cx { base, nt, d, taus: &taus, blk: &blk };
        let mut cur = x;
        for il in 0..start {
            cur = self.layer(m, &cx, il, il, cur, &mut nb, None);
        }
        let prelude = cur.clone();
        let pc = nb;
        for pass in 0..passes {
            let mut bank: Option<(Buf, usize, Buf)> = None;
            if pass > 0 {
                let n_bank = nb.saturating_sub(pc + 1);
                let b = g.scratch((n_bank * nt * d).max(1));
                if n_bank > 0 {
                    g.copy(&blk, &b, (pc + 1) * nt * d, 0, n_bank * nt * d);
                }
                bank = Some((b, n_bank, std::mem::replace(&mut cur, prelude.clone())));
                nb = pc;
            }
            for ip in start..end {
                match &bank {
                    None => cur = self.layer(m, &cx, ip, ip, cur, &mut nb, None),
                    Some((b, n_bank, out)) => {
                        let slot = n_phys + (pass - 1) * (end - start) + (ip - start);
                        let mq = self.layers[ip].msg_query.as_ref().expect("a looped layer's message queries");
                        let msgs: Vec<Buf> = (0..4).map(|j| g.ink_mix(b, 0, nt * d, *n_bank, out, mq, j, d, nt)).collect();
                        cur = self.layer(m, &cx, slot, ip, cur, &mut nb, Some(&msgs));
                    }
                }
            }
        }
        for il in end..n_phys {
            cur = self.layer(m, &cx, il, il, cur, &mut nb, None);
        }
        let xo = g.ink_mix(&blk, 0, nt * d, nb, &cur, &self.out_query, 0, d, nt);
        st.pos = base + nt;
        let h = g.rms_norm(&xo, &self.out_norm, m.eps);
        let (h, n_out) = if all || nt == 1 { (h, nt) } else { (g.row(&h, (nt - 1) * d, d), 1) };
        let logits = g.mv_tokens(self.output, &h, n_out as u32);
        if ink.softcap != 0.0 {
            g.ink_softcap(&logits, ink.softcap);
        }
        let v = g.read(&logits);
        v.chunks_exact(v.len() / n_out).map(|c| c[..ink.n_vocab].to_vec()).collect()
    }

    /// cur - tanh(forget) cur + tanh(gain) msg, the gates from the normed pair (inkling.rs `control`).
    fn control(&self, cur: &Buf, msg: &Buf, c: &GCtrl, d: usize, nt: usize) -> Buf {
        let g = self.g;
        let h = g.mv_tokens(&c.down, &g.ink_cat_norm(cur, msg, d), nt as u32);
        g.ink_ctrl_out(cur, msg, &g.mv_tokens(&c.gain, &h, nt as u32), &g.mv_tokens(&c.forget, &h, nt as u32))
    }

    /// One layer at cache slot `slot` (physical layer `il`), as inkling.rs `layer`, for the run's tokens.
    #[allow(clippy::too_many_arguments)]
    fn layer(&self, m: &Model, cx: &Cx, slot: usize, il: usize, inp_sa: Buf, nb: &mut usize, msgs: Option<&[Buf]>) -> Buf {
        let (g, ink) = (self.g, self.ink);
        let gl = &self.layers[il];
        let ly = gl.l;
        let (d, nt, base) = (cx.d, cx.nt, cx.base);
        let (hd, nh, nkv) = (ly.hd, ly.n_head, ly.n_kv);
        let kvd = hd * nkv;
        let (nc, past) = (ink.n_conv, ink.n_conv - 1);
        let sl = &self.slots[slot];
        let n = nt as u32;

        let mut inputs: Vec<Buf> = match &gl.res_query {
            Some(q) if ly.swa => vec![g.ink_mix(cx.blk, 0, nt * d, *nb, &inp_sa, q, 0, d, nt); 3],
            Some(q) => (0..3).map(|j| g.ink_mix(cx.blk, 0, nt * d, *nb, &inp_sa, q, j, d, nt)).collect(),
            None => vec![inp_sa.clone(); 3],
        };
        if let Some(ms) = msgs {
            for j in 0..3 {
                inputs[j] = self.control(&inputs[j], &ms[j], &gl.ctrl[j], d, nt);
            }
        }
        let normed: Vec<Buf> = if ly.swa && msgs.is_none() {
            vec![g.rms_norm(&inputs[0], &gl.norms[0], m.eps); 3]
        } else {
            (0..3).map(|j| g.rms_norm(&inputs[j], &gl.norms[if ly.swa { 0 } else { j }], m.eps)).collect()
        };
        let (in_q, in_k, in_v) = (&normed[0], &normed[1], &normed[2]);

        let q = g.mv_tokens(gl.wq, in_q, n);
        let kd = gl.k_proj.map(|p| g.mv_tokens(p, in_k, n));
        let vd = gl.v_proj.map(|p| g.mv_tokens(p, in_v, n));
        let kc = g.ink_conv(&g.mv_tokens(gl.wk, in_k, n), &gl.k_conv, kd.as_ref(), &sl.hist, 0, kvd, nc, nt);
        let vc = g.ink_conv(&g.mv_tokens(gl.wv, in_v, n), &gl.v_conv, vd.as_ref(), &sl.hist, kvd * past, kvd, nc, nt);
        let r = g.mv_tokens(gl.rel, in_q, n);
        let tau = (!ly.swa).then_some(cx.taus);
        g.ink_clip(&q, Some(&gl.q_clip), tau, hd, nh, ink.qk_clip, false);
        g.ink_clip(&kc, None, None, hd, nkv, ink.qk_clip, true);
        g.ink_rmsn_rows(&vc, hd, m.eps, true);
        g.ink_clip(&r, Some(&gl.rel_clip), tau, ink.n_rel, nh, ink.rel_clip, false);
        let sinks = g.mv_tokens(&gl.sink, in_q, n);
        g.ink_kv_put(&kc, &sl.kc, base, kvd, sl.wring);
        g.ink_kv_put(&vc, &sl.vc, base, kvd, sl.wring);
        let (extent, win) = if ly.swa { (ink.n_swa, ink.n_swa) } else { (ink.extent, 0) };
        let att = g.ink_attend(&q, &sl.kc, &sl.vc, &r, &gl.rel_bias, &sinks, base, nt, nh, nkv, hd, ink.n_rel, extent, win, sl.wring);
        g.ink_sig_mul(&att, &g.mv_tokens(gl.gate, in_q, n));
        let cur = g.ink_conv(&g.mv_tokens(gl.wo, &att, n), &gl.post_conv, None, &sl.hist, 2 * kvd * past, d, nc, nt);

        // the first layer of a block starts a new residual stream; the one before it joins the mixed blocks
        if il % ink.block == 0 {
            g.copy(&inp_sa, cx.blk, 0, *nb * nt * d, nt * d);
            *nb += 1;
        } else {
            g.add_into(&inp_sa, &cur);
        }
        let ffn_inp = cur;
        let mut x = g.ink_mix(cx.blk, 0, nt * d, *nb, &ffn_inp, &gl.ffn_query, 0, d, nt);
        if let Some(ms) = msgs {
            x = self.control(&x, &ms[3], &gl.ctrl[3], d, nt);
        }
        let x = g.rms_norm(&x, &gl.ffn_norm, m.eps);
        let acc = self.experts(m, il, gl, &x, nt);
        let out = g.ink_conv(&acc, &gl.ffn_conv, None, &sl.hist, (2 * kvd + d) * past, d, nc, nt);
        g.add_into(&ffn_inp, &out);
        out
    }

    /// The routed experts of `x` (nt tokens), weighted and summed in routing order: the routing on the device, one
    /// read, then each chosen expert once for the tokens that chose it, where placement put it.
    fn experts(&self, m: &Model, il: usize, gl: &GLayer, x: &Buf, nt: usize) -> Buf {
        let (g, ink) = (self.g, self.ink);
        let ly = gl.l;
        let (k, d, n_e) = (ly.top_k, m.n_embd, gl.experts.len());
        let logits = g.mv_tokens(&gl.router, x, nt as u32);
        let ws = if ink.w_scale != 0.0 { ink.w_scale } else { 1.0 };
        let routes = g.ink_route(&logits, gl.probs_b.as_ref(), n_e, k, ws);
        // the one wait of the layer: the choice, and the input when some experts are off the device
        let off = gl.experts.iter().any(|e| !matches!(e, Where::Here(..)));
        let (r, xv) = if off {
            let both = g.scratch(routes.len + x.len);
            g.copy(&routes, &both, 0, 0, routes.len);
            g.copy(x, &both, 0, routes.len, x.len);
            let mut v = g.read(&both);
            let xv = v.split_off(routes.len);
            (v, Some(xv))
        } else {
            (g.read(&routes), None)
        };
        let mut users: Vec<Vec<usize>> = vec![Vec::new(); n_e];
        for t in 0..nt {
            for s in 0..k {
                let e = r[t * 2 * k + s] as usize;
                users[e].push(t * k + s);
                m.heat[il * m.profile_experts + e].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
        }
        let slots = g.scratch(nt * k * d);
        let idx = |us: &[usize], f: &dyn Fn(usize) -> usize| us.iter().map(|&ts| f(ts) as f32).collect::<Vec<f32>>();
        // the device's experts, queued
        for (e, us) in users.iter().enumerate().filter(|(_, us)| !us.is_empty()) {
            if let Where::Here(gu, dn) = gl.experts[e] {
                let n = us.len();
                let xin = if nt == 1 { x.clone() } else { g.gather_rows(x, &g.buffer(&idx(us, &|ts| ts / k)), d) };
                let o = g.mv_tokens(dn, &g.ink_situ(&g.mv_tokens(gu, &xin, n as u32), ly.gate_up.0[e].rows / 2, ink.situ_beta, ink.situ_lb), n as u32);
                if n == 1 {
                    g.copy(&o, &slots, 0, us[0] * d, d);
                } else {
                    g.scatter_rows(&o, &g.buffer(&idx(us, &|ts| ts)), &slots, d);
                }
            }
        }
        if let Some(xv) = xv {
            // the other devices' experts, queued there with the input sent once per device
            let mut others: Vec<(usize, Buf, Vec<usize>)> = Vec::new();
            for (e, us) in users.iter().enumerate().filter(|(_, us)| !us.is_empty()) {
                if let Where::Other(od, gu, dn) = gl.experts[e] {
                    let o2 = crate::gpu::device(od);
                    let at = match others.iter().position(|o| o.0 == od) {
                        Some(i) => i,
                        None => {
                            others.push((od, o2.scratch(nt * k * d), Vec::new()));
                            others.len() - 1
                        }
                    };
                    let xin = o2.buffer(&us.iter().flat_map(|&ts| xv[ts / k * d..(ts / k + 1) * d].iter().copied()).collect::<Vec<f32>>());
                    let n = us.len() as u32;
                    let o = o2.mv_tokens(dn, &o2.ink_situ(&o2.mv_tokens(gu, &xin, n), ly.gate_up.0[e].rows / 2, ink.situ_beta, ink.situ_lb), n);
                    o2.scatter_rows(&o, &o2.buffer(&idx(us, &|ts| ts)), &others[at].1, d);
                    others[at].2.extend_from_slice(us);
                }
            }
            // the CPU's experts meanwhile
            let here: Vec<usize> = (0..n_e).filter(|&e| !users[e].is_empty() && matches!(gl.experts[e], Where::Cpu)).collect();
            if !here.is_empty() {
                if m.prefetch.load(std::sync::atomic::Ordering::Relaxed) {
                    here.iter().for_each(|&e| { crate::pack::wanted_off_device(&ly.gate_up.0[e]); crate::pack::wanted_off_device(&ly.down.0[e]); });
                }
                use rayon::prelude::*;
                let outs: Vec<Vec<Vec<f32>>> = here.par_iter().map(|&e| {
                    let xs: Vec<Vec<f32>> = users[e].iter().map(|&ts| xv[ts / k * d..(ts / k + 1) * d].to_vec()).collect();
                    ink.expert_rows(il, e, &xs)
                }).collect();
                let rows: Vec<f32> = outs.iter().flatten().flatten().copied().collect();
                let dst: Vec<f32> = here.iter().flat_map(|&e| users[e].iter().map(|&ts| ts as f32)).collect();
                g.scatter_rows(&g.buffer(&rows), &g.buffer(&dst), &slots, d);
            }
            for (od, buf, us) in others {
                let v = crate::gpu::device(od).read(&buf);
                let rows: Vec<f32> = us.iter().flat_map(|&ts| v[ts * d..(ts + 1) * d].iter().copied()).collect();
                g.scatter_rows(&g.buffer(&rows), &g.buffer(&idx(&us, &|ts| ts)), &slots, d);
            }
        }
        let acc = g.ink_zeros(nt * d);
        g.moe_combine(&slots, &routes, &acc, k as u32);
        acc
    }

    /// The graph's device.
    #[allow(dead_code)]
    pub fn device(&self) -> usize {
        self.dev
    }
}

/// What every layer of a run shares.
struct Cx<'b> {
    base: usize,
    nt: usize,
    d: usize,
    taus: &'b Buf,
    blk: &'b Buf,
}

/// The device graph of either kind: main.rs's one entry point.
pub enum AnyGraph<'a> {
    Std(Graph<'a>),
    Ink(InkGraph<'a>),
}

/// Why the graph cannot run this model whatever the placement, if it cannot.
pub fn refusal(m: &Model) -> Option<String> {
    if m.ink.is_some() { ink_refusal(m) } else { crate::graph::refusal(m) }
}

/// The bytes the graph keeps free on the first device for its caches.
pub fn kv_need(m: &Model, max_ctx: usize, per_value: u64) -> u64 {
    if m.ink.is_some() { ink_kv_need(m, max_ctx) } else { crate::graph::kv_need(m, max_ctx, per_value) }
}

impl<'a> AnyGraph<'a> {
    pub fn new(m: &'a Model, max_ctx: usize) -> Result<AnyGraph<'a>, String> {
        if m.ink.is_some() { InkGraph::new(m, max_ctx).map(AnyGraph::Ink) } else { Graph::new(m, max_ctx).map(AnyGraph::Std) }
    }

    pub fn experts_on(&self) -> (usize, usize) {
        match self { AnyGraph::Std(g) => g.experts_on(), AnyGraph::Ink(g) => g.experts_on() }
    }

    pub fn bench(&self, m: &Model) {
        match self { AnyGraph::Std(g) => g.bench(m), AnyGraph::Ink(_) => eprintln!("GPU_BENCH: not for Inkling's graph") }
    }

    pub fn batches(&self) -> bool {
        match self { AnyGraph::Std(g) => g.batches(), AnyGraph::Ink(_) => true }
    }

    /// Whether `forward_embd` takes input rows (GENI's pictures) on the graph: Inkling's.
    pub fn takes_rows(&self) -> bool {
        matches!(self, AnyGraph::Ink(_))
    }

    pub fn forward(&self, m: &Model, st: &mut State, token: u32) -> Vec<f32> {
        match self { AnyGraph::Std(g) => g.forward(m, st, token), AnyGraph::Ink(g) => g.forward(m, st, token) }
    }

    pub fn forward_batch(&self, m: &Model, st: &mut State, tokens: &[u32]) -> Vec<f32> {
        match self { AnyGraph::Std(g) => g.forward_batch(m, st, tokens), AnyGraph::Ink(g) => g.forward_batch(m, st, tokens) }
    }

    pub fn forward_embd(&self, m: &Model, st: &mut State, row: Vec<f32>) -> Vec<f32> {
        match self { AnyGraph::Std(_) => unreachable!("takes_rows"), AnyGraph::Ink(g) => g.forward_embd(m, st, row) }
    }
}

#[cfg(test)]
mod tests {
    //! Inkling's graph against the CPU path (model/inkling.rs, the oracle), on AM2_TEST_DEVICE (default cuda:1) when it
    //! opens, else the tests say so and pass: the logits token by token, a chunk at a time, a chunk then tokens and a
    //! chunk's last, with all, half and none of the experts on the device, within graph.rs's tolerance and with the
    //! same best token at every step.
    use super::*;
    use crate::testgguf::{tmp_inkling, InkSpec};
    use std::path::Path;

    /// graph.rs's: the CPU keeps K and V in f16, as this graph does, so the gap here is the sums' order.
    const TOL: f32 = 3e-3;

    fn device() -> Option<String> {
        let d = std::env::var("AM2_TEST_DEVICE").unwrap_or_else(|_| "cuda:1".into());
        if d == "none" || strata_gpu::Gpu::open(&d).is_err() {
            eprintln!("no device {d}: the graph tests are skipped");
            return None;
        }
        Some(d)
    }

    /// Everything placed on the device, then only this share of the experts kept there (the first layers' first).
    fn placed(path: &Path, dev: &str, experts: f64) -> Model {
        let (mut m, _) = Model::load(path, None).unwrap();
        // the caches' room kept, as main.rs keeps it (gpu.rs ROOM is the process's first placement's)
        let reserve = ink_kv_need(&m, 256);
        crate::gpu::place(&mut m, &[(dev.to_string(), 2 << 30)], None, reserve).unwrap();
        let (_, mut ex) = m.matrices_mut();
        let mut keys: Vec<(usize, usize)> = ex.keys().copied().collect();
        keys.sort();
        let keep = (keys.len() as f64 * experts).round() as usize;
        for k in &keys[keep..] {
            for q in ex.get_mut(k).unwrap().iter_mut() {
                q.gpu = None;
            }
        }
        m
    }

    fn max_diff(a: &[Vec<f32>], b: &[Vec<f32>]) -> f32 {
        assert_eq!(a.len(), b.len());
        a.iter().zip(b).map(|(x, y)| { assert_eq!(x.len(), y.len()); x.iter().zip(y).map(|(p, q)| (p - q).abs()).fold(0.0, f32::max) }).fold(0.0, f32::max)
    }

    fn argmax(v: &[f32]) -> usize {
        v.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|(i, _)| i).unwrap()
    }

    /// 40 tokens: past the 16-token window and the 24-position bias table.
    fn tokens() -> Vec<u32> {
        (0..40).map(|i| (i * 37 + 5) % 256).collect()
    }

    /// The CPU's logits token by token against the graph's four ways. -> (the worst difference, experts on / all).
    fn check(name: &str, cpu: &Model, m: &Model) -> (f32, (usize, usize)) {
        let toks = tokens();
        let mut st = cpu.new_state();
        let want: Vec<Vec<f32>> = toks.iter().map(|&t| cpu.forward(&mut st, t)).collect();
        let gr = InkGraph::new(m, 256).unwrap();
        let mut st = m.new_state();
        let one: Vec<Vec<f32>> = toks.iter().map(|&t| gr.forward(m, &mut st, t)).collect();
        let mut st = m.new_state();
        let chunk = gr.forward_batch_all(m, &mut st, &toks);
        let mut st = m.new_state();
        let mut mixed = gr.forward_batch_all(m, &mut st, &toks[..25]);
        mixed.extend(toks[25..].iter().map(|&t| gr.forward(m, &mut st, t)));
        let mut st = m.new_state();
        let last = gr.forward_batch(m, &mut st, &toks);
        let d = [max_diff(&one, &want), max_diff(&chunk, &want), max_diff(&mixed, &want), max_diff(&[last], &want[want.len() - 1..])];
        eprintln!("{name}: experts {:?} on the device: max |graph - cpu| one={:.2e} chunk={:.2e} mixed={:.2e} last={:.2e}", gr.experts_on(), d[0], d[1], d[2], d[3]);
        if std::env::var_os("AM2_INK_DIFFS").is_some() {
            let per: Vec<String> = one.iter().zip(&want).map(|(a, b)| format!("{:.1e}", max_diff(&[a.clone()], &[b.clone()]))).collect();
            eprintln!("{name}: per token {}", per.join(" "));
        }
        for (i, (a, b)) in one.iter().zip(&want).enumerate() {
            assert_eq!(argmax(a), argmax(b), "{name}: token {i}'s best logit");
        }
        let worst = d.iter().copied().fold(0.0, f32::max);
        assert!(worst < TOL, "{name}: {d:?}");
        (worst, gr.experts_on())
    }

    #[test]
    fn inkling_on_the_graph_equals_the_cpu() {
        let _one = crate::graph::tests::ONE.lock().unwrap_or_else(|e| e.into_inner());
        let Some(dev) = device() else { return };
        for (name, spec) in [("tiny inkling2_loop", InkSpec::tiny()), ("tiny inkling2", InkSpec { loop_: None, seed: 0x1_4C1E, ..InkSpec::tiny() })] {
            let p = tmp_inkling(&name.replace(' ', "-"), &spec);
            let (cpu, _) = Model::load(&p, None).unwrap();
            let (_, on) = check(name, &cpu, &placed(&p, &dev, 1.0));
            assert_eq!(on.0, on.1, "every expert on the device");
            let (_, on) = check(&format!("{name}, half the experts on the device"), &cpu, &placed(&p, &dev, 0.5));
            assert!(on.0 > 0 && on.0 < on.1, "{on:?}");
            let (_, on) = check(&format!("{name}, the experts on the CPU"), &cpu, &placed(&p, &dev, 0.0));
            assert_eq!(on.0, 0);
        }
    }

    /// One wait per layer evaluation (the router's choice, with the input when experts are off the device) and one
    /// for the logits, per token and per chunk.
    #[test]
    fn inkling_graph_waits_once_per_layer() {
        let _one = crate::graph::tests::ONE.lock().unwrap_or_else(|e| e.into_inner());
        let Some(dev) = device() else { return };
        let p = tmp_inkling("tiny-inkling2_loop", &InkSpec::tiny());
        let evals = 6 + 2 * 2;
        for share in [1.0, 0.0] {
            let m = placed(&p, &dev, share);
            let gr = InkGraph::new(&m, 256).unwrap();
            let mut st = m.new_state();
            let s0 = strata_gpu::stats::snapshot();
            gr.forward_batch(&m, &mut st, &tokens()[..20]);
            let s1 = strata_gpu::stats::snapshot();
            gr.forward(&m, &mut st, 7);
            let s2 = strata_gpu::stats::snapshot();
            let (chunk, tok) = (s1 - s0, s2 - s1);
            eprintln!("{share} of the experts on the device: a chunk of 20: {} syncs, {} launches; a token: {} syncs, {} launches",
                      chunk.syncs, chunk.launches, tok.syncs, tok.launches);
            assert_eq!(chunk.syncs, evals + 1);
            assert_eq!(tok.syncs, evals + 1);
        }
    }

    /// A Kosha pack of the fixture with an expert profile: placement puts the profile's hottest experts on the device
    /// (the budget takes about half of them), the graph runs them there and the rest on the CPU, and the logits are the
    /// CPU path's on the same pack.
    #[cfg(feature = "aipac")]
    #[test]
    fn a_kosha_pack_places_its_experts_by_the_profile_on_the_graph() {
        let _one = crate::graph::tests::ONE.lock().unwrap_or_else(|e| e.into_inner());
        let Some(dev) = device() else { return };
        let gguf = tmp_inkling("tiny-inkling2_loop", &InkSpec::tiny());
        let dir = gguf.parent().unwrap().to_path_buf();
        // a profile that ranks the experts in an order of its own: the last layer's last expert first
        let (probe, _) = Model::load(&gguf, None).unwrap();
        let (counts, nl) = probe.expert_layers();
        let ne = probe.profile_experts;
        let mut ranked: Vec<(u16, u16)> = counts.iter().flat_map(|&(l, n)| (0..n).map(move |e| (l as u16, e as u16))).collect();
        ranked.sort_by_key(|&(l, e)| (std::cmp::Reverse((l as usize * 7 + e as usize * 3) % 11), l, e));
        let prof = crate::pack::Profile { n_layers: nl, n_expert: ne, ranked: ranked.clone() };
        let prof_path = dir.join("tiny-inkling.strp");
        prof.write(&prof_path).unwrap();
        let kosha = dir.join("tiny-inkling.kosha");
        if !kosha.exists() {
            am2pack::build(&am2pack::BuildOptions { model: gguf.clone(), profile: Some(prof_path.clone()), out: kosha.clone(), ..Default::default() }, &mut |_| {}).unwrap();
        }
        let (cpu, _) = Model::load(&kosha, None).unwrap();
        assert!(!cpu.file_order.is_empty(), "the experts come from the pack's records");
        // a budget for the core and half the experts
        let (mut m, _) = Model::load(&kosha, None).unwrap();
        let (every, ex) = m.matrices_mut();
        let core: u64 = every.iter().map(|q| q.data.len() as u64).sum();
        let one: u64 = ex.values().next().unwrap().iter().map(|q| q.data.len() as u64).sum();
        let half = ex.len() as u64 / 2;
        let budget = core + half * one + one / 2;
        let p = crate::pack::Profile::read(&prof_path, nl, ne).unwrap();
        let reserve = ink_kv_need(&m, 256);
        crate::gpu::place(&mut m, &[(dev.clone(), budget + reserve)], Some(&p), reserve).unwrap();
        let ink = m.ink.as_ref().unwrap();
        let on: Vec<(u16, u16)> = ranked.iter().copied().filter(|&(l, e)| ink.layers[l as usize].gate_up.0[e as usize].gpu.is_some()).collect();
        assert_eq!(on.len() as u64, half, "half the experts on the device");
        assert_eq!(on, ranked[..half as usize], "the profile's hottest");
        let (_, n) = check("tiny inkling2_loop from a Kosha pack, by its profile", &cpu, &m);
        assert_eq!(n.0 as u64, half);
    }

    /// The speed run (docs/INKLING-PLAN.md I8) on testgguf's `write_inkling_speed_model` (the real shapes, 32 experts):
    /// matrix by matrix (the CPU path with every product on the device, what `--no-graph` runs) against the graph, with
    /// every expert on the device and with half of them on the CPU: prompt and decode tokens/s, waits and launches per
    /// token. AM2_INK_BENCH=<the model> cargo test --release inkling_speed -- --ignored --nocapture
    #[test]
    #[ignore]
    fn inkling_speed() {
        let _one = crate::graph::tests::ONE.lock().unwrap_or_else(|e| e.into_inner());
        let Some(dev) = device() else { return };
        let path = std::path::PathBuf::from(std::env::var("AM2_INK_BENCH").expect("AM2_INK_BENCH=<model>"));
        let ctx = 2048;
        let (mut m, _) = Model::load(&path, None).unwrap();
        let reserve = ink_kv_need(&m, ctx);
        let t0 = std::time::Instant::now();
        let (summary, _, _) = crate::gpu::place(&mut m, &[(dev.clone(), 7 << 30)], None, reserve).unwrap();
        eprintln!("placed in {:.1} s: {summary}", t0.elapsed().as_secs_f64());
        let prompt: Vec<u32> = (0..512).map(|i| (i * 7919 + 13) % 200000).collect();
        let per = |s: strata_gpu::stats::Stats, n: usize| (s.syncs as f64 / n as f64, s.launches as f64 / n as f64);
        // matrix by matrix: a short prompt and a few tokens (each product a round trip)
        {
            let mut st = m.new_state();
            let t = std::time::Instant::now();
            let s0 = strata_gpu::stats::snapshot();
            for &tk in &prompt[..16] {
                m.forward(&mut st, tk);
            }
            let pp = 16.0 / t.elapsed().as_secs_f64();
            let t = std::time::Instant::now();
            let s1 = strata_gpu::stats::snapshot();
            let mut tk = 5u32;
            for _ in 0..16 {
                let l = m.forward(&mut st, tk);
                tk = argmax(&l) as u32;
            }
            let tg = 16.0 / t.elapsed().as_secs_f64();
            let (sy, la) = per(strata_gpu::stats::snapshot() - s1, 16);
            let _ = s0;
            eprintln!("INK_BENCH matrix-by-matrix all experts on the device: prompt {pp:.1} tok/s (token by token), decode {tg:.1} tok/s, {sy:.0} syncs and {la:.0} launches per token");
        }
        for share in [1.0, 0.5] {
            if share < 1.0 {
                let (_, mut ex) = m.matrices_mut();
                let mut keys: Vec<(usize, usize)> = ex.keys().copied().collect();
                keys.sort_by_key(|&(l, e)| (e, l));
                let keep = (keys.len() as f64 * share).round() as usize;
                for k in &keys[keep..] {
                    for q in ex.get_mut(k).unwrap().iter_mut() {
                        q.gpu = None;
                    }
                }
            }
            let gr = InkGraph::new(&m, ctx).unwrap();
            let mut st = m.new_state();
            // warm the kernels (CubeCL compiles each on its first launch)
            gr.forward_batch(&m, &mut st, &prompt[..130]);
            gr.forward(&m, &mut st, 3);
            let mut st = m.new_state();
            let t = std::time::Instant::now();
            let s0 = strata_gpu::stats::snapshot();
            let mut l = vec![];
            for c in prompt.chunks(128) {
                l = gr.forward_batch(&m, &mut st, c);
            }
            let pp = prompt.len() as f64 / t.elapsed().as_secs_f64();
            let (psy, pla) = per(strata_gpu::stats::snapshot() - s0, prompt.len());
            let mut tk = argmax(&l) as u32;
            let n = 64;
            let t = std::time::Instant::now();
            let s1 = strata_gpu::stats::snapshot();
            for _ in 0..n {
                tk = argmax(&gr.forward(&m, &mut st, tk)) as u32;
            }
            let tg = n as f64 / t.elapsed().as_secs_f64();
            let s = strata_gpu::stats::snapshot() - s1;
            let (sy, la) = per(s, n);
            eprintln!("INK_BENCH graph, experts {:?} on the device: prompt {pp:.1} tok/s ({psy:.2} syncs, {pla:.1} launches per token), decode {tg:.1} tok/s \
                       ({sy:.0} syncs, {la:.0} launches, {:.2} ms waiting per token) at {} tokens", gr.experts_on(), s.wait_ns as f64 / 1e6 / n as f64, st.pos);
        }
    }
}
