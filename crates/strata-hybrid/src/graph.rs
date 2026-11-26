//! A whole token on the device (docs/ENGINE-PLAN.md phase 1): the norms, RoPE, attention over a KV cache that stays on the
//! device, the feed-forwards and the additions run as one queue of kernels, and only the logits are read back. The CPU
//! path crosses to the device and back for every product, which is what limited it.
//!
//! It runs dense attention models (`llama`, `mistral3`, `qwen3` with its per-head q/k norms and NEOX RoPE), the loop
//! (Avarta: the core's extra passes, each with its own KV cache, and the gate mix x + g (y - x)) and softmax-routed
//! experts (`qwen3moe`, Mixtral-style `llama`; docs/MULA-PLAN.md M8 and the plan's Part D). Attention and dense matrices
//! must all be on one device. Experts may be anywhere: the router and its top-k run on the device, and the one wait per
//! MoE layer reads the choice back (2k values, not one wait per product). Experts on the device run there; the others run
//! on the CPU (decode and prompt alike), with only the activations crossing: the layer's normed input down, their
//! outputs up. They are never streamed to the device per token.
//!
//! A prompt chunk runs as a whole: each matrix once for all its tokens, and each chosen expert once for the tokens that
//! chose it (gathered into one block, the outputs scattered back to their (token, slot) rows, then summed in routing
//! order as model.rs sums them).
//!
//! Activations are not rounded to 8 bits here as the CPU path rounds them (ggml's way), and K and V stay f32 (the CPU
//! path keeps f16), so a logit can differ a little more; the parity tests (below) decide.

use crate::model::{Act, Block, Gating, Model, Moe, QMat, State, Step};
use strata_gpu::{Buf, Gpu, Weights};

struct Attn<'a> {
    norm: Buf,
    q: &'a Weights,
    k: &'a Weights,
    v: &'a Weights,
    o: &'a Weights,
    q_norm: Option<Buf>,
    k_norm: Option<Buf>,
    n_head: u32,
    n_kv: u32,
    hd: u32,
}

/// One attention cache on the device: a layer's in one pass of the loop.
struct Kv {
    kc: Buf,
    vc: Buf,
    s: Buf,
}

/// A routed expert layer: the router on the device, and each expert's (gate, up, down) when they are all there.
struct GMoe<'a> {
    m: &'a Moe,
    router: Weights,
    on: Vec<Option<(&'a Weights, &'a Weights, &'a Weights)>>,
}

enum Ffn<'a> {
    Dense { gate: &'a Weights, up: &'a Weights, down: &'a Weights },
    Moe(GMoe<'a>),
}

struct Layer<'a> {
    attn: Attn<'a>,
    ffn_norm: Buf,
    ffn: Ffn<'a>,
}

pub struct Graph<'a> {
    g: &'static Gpu,
    layers: Vec<Layer<'a>>,
    /// per (pass, layer): pass * layers + layer; the extra passes have caches for the loop's core only
    kv: Vec<Option<Kv>>,
    /// the loop's gates, pass 1 first
    gates: Vec<Buf>,
    out_norm: Buf,
    output: &'a Weights,
    n_rot: u32,
    neox: bool,
    max_ctx: usize,
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

/// The matrix's copy on device `dev`, if it has one there.
fn there(q: &QMat, dev: usize) -> Option<&Weights> {
    q.gpu.as_ref().filter(|d| d.device() == dev).map(|d| d.weights())
}

/// Why the graph cannot run this MoE layer, if it cannot.
fn moe_refusal(m: &Moe) -> Option<&'static str> {
    if m.gating != Gating::Softmax || m.bias.is_some() {
        Some("--graph runs softmax-routed experts only so far (not sigmoid gating or a selection bias)")
    } else if m.latent.is_some() || m.shared.is_some() || m.shared_gate.is_some() {
        Some("--graph does not run latent or shared experts yet")
    } else if m.act != Act::Silu || m.gate_exps.is_none() {
        Some("--graph needs SiLU-gated experts")
    } else {
        None
    }
}

/// Why the graph cannot run this model's layers whatever the placement, if it cannot.
pub fn refusal(m: &Model) -> Option<String> {
    if m.arch.starts_with("inkling2") {
        return Some("--graph does not run Inkling yet".into());
    }
    if m.rope.is_none() {
        return Some("--graph needs a model with RoPE".into());
    }
    for l in &m.layers {
        if l.hc.is_some() || l.ple.is_some() {
            return Some("--graph does not run hyper-connections yet".into());
        }
        let [(_, Block::Attn(a)), (_, ffn)] = &l.parts[..] else {
            return Some("--graph runs attention layers with a feed-forward only so far".into());
        };
        if a.q_b.is_some() || a.k_b.is_some() || a.v_b.is_some() || a.o_b.is_some() || !a.rope || a.gated || a.qsa.is_some() {
            return Some("--graph: this model's attention has parts the device path does not have yet".into());
        }
        match ffn {
            Block::Dense(f) if f.gate.is_none() || f.act != Act::Silu => return Some("--graph needs SiLU-gated feed-forwards".into()),
            Block::Dense(_) => {}
            Block::Moe(e) => if let Some(why) = moe_refusal(e) { return Some(why.into()) },
            _ => return Some("--graph runs dense and expert feed-forwards only so far".into()),
        }
    }
    None
}

/// The bytes the graph's KV caches and score scratch take for `max_ctx` tokens at `per_value` bytes a K or V value:
/// what placement keeps free on the first device for them (main.rs), so the experts do not fill the budget first.
pub fn kv_need(m: &Model, max_ctx: usize, per_value: u64) -> u64 {
    let passes = m.loop_.as_ref().map_or(1, |l| l.passes);
    let mut b = 0u64;
    for pass in 0..passes {
        for (l, layer) in m.layers.iter().enumerate() {
            if pass > 0 && !m.loop_.as_ref().is_some_and(|lp| (lp.start..lp.end).contains(&l)) {
                continue;
            }
            if let Some((_, Block::Attn(a))) = layer.parts.first() {
                b += (2 * a.n_kv * a.head_dim) as u64 * max_ctx as u64 * per_value + a.n_head as u64 * max_ctx as u64 * 4;
            }
        }
    }
    b
}

impl<'a> Graph<'a> {
    pub fn new(m: &'a Model, max_ctx: usize) -> Result<Graph<'a>, String> {
        if let Some(why) = refusal(m) {
            return Err(why);
        }
        let mut dev = None;
        let rope = m.rope.as_ref().ok_or("--graph needs a model with RoPE")?;
        let output = on_device(&m.output, &mut dev)?;
        // first the attention and dense matrices, which fix the device; the experts need not be on it
        let mut raw = Vec::new();
        for l in &m.layers {
            if l.hc.is_some() || l.ple.is_some() {
                return Err("--graph does not run hyper-connections yet".into());
            }
            let [(n1, Block::Attn(a)), (n2, ffn)] = &l.parts[..] else {
                return Err("--graph runs attention layers with a feed-forward only so far".into());
            };
            if a.q_b.is_some() || a.k_b.is_some() || a.v_b.is_some() || a.o_b.is_some() || !a.rope || a.gated || a.qsa.is_some() {
                return Err("--graph: this model's attention has parts the device path does not have yet".into());
            }
            let dense = match ffn {
                Block::Dense(f) => {
                    let gate = f.gate.as_ref().ok_or("--graph needs gated feed-forwards")?;
                    if f.act != Act::Silu {
                        return Err("--graph needs SiLU feed-forwards".into());
                    }
                    Some((on_device(gate, &mut dev)?, on_device(&f.up, &mut dev)?, on_device(&f.down, &mut dev)?))
                }
                Block::Moe(e) => {
                    if let Some(why) = moe_refusal(e) {
                        return Err(why.into());
                    }
                    None
                }
                _ => return Err("--graph runs dense and expert feed-forwards only so far".into()),
            };
            raw.push((n1, a, on_device(&a.q, &mut dev)?, on_device(&a.k, &mut dev)?, on_device(&a.v, &mut dev)?, on_device(&a.o, &mut dev)?,
                      n2, ffn, dense));
        }
        let dev = dev.ok_or("--graph: no layers")?;
        let g = crate::gpu::device(dev);
        let looped = m.loop_.as_ref();
        if !g.has_moe_kernels() && (looped.is_some() || m.moes().next().is_some()) {
            return Err("--graph: the loop and experts are not on the native Metal backend yet (AM2_METAL_NATIVE=0 runs them on wgpu)".into());
        }
        // the caches: every layer's for the base pass, the loop core's for each extra pass (as many as the model has:
        // --loop-passes may ask fewer per request). They and the score scratch count against the --devices budget: K and
        // V at Gpu::kv_bytes each (half on native Metal, f32 on CubeCL), the scores f32.
        let n_layers = m.layers.len();
        let passes = looped.map_or(1, |l| l.passes);
        let wanted = |pass: usize, l: usize| pass == 0 || looped.is_some_and(|lp| (lp.start..lp.end).contains(&l));
        let kvb = g.kv_bytes();
        let mut kv_bytes = 0u64;
        for pass in 0..passes {
            for (_, r) in raw.iter().enumerate().filter(|(l, _)| wanted(pass, *l)) {
                kv_bytes += (2 * r.1.n_kv * r.1.head_dim) as u64 * max_ctx as u64 * kvb + r.1.n_head as u64 * max_ctx as u64 * 4;
            }
        }
        if let Some(room) = crate::gpu::ROOM.get().and_then(|r| r.get(dev)) {
            if kv_bytes > *room {
                let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
                return Err(format!("the KV cache for {max_ctx} tokens needs {:.2} GiB on the device and its --devices budget has {:.2} GiB left \
                                    (a smaller --max-context or a larger budget)", gib(kv_bytes), gib(*room)));
            }
        }
        let mut kv = Vec::with_capacity(passes * n_layers);
        for pass in 0..passes {
            for (l, r) in raw.iter().enumerate() {
                let row = r.1.n_kv * r.1.head_dim;
                kv.push(wanted(pass, l).then(|| Kv { kc: g.kv_cache(row * max_ctx), vc: g.kv_cache(row * max_ctx), s: g.scratch(r.1.n_head * max_ctx) }));
            }
        }
        let mut layers = Vec::with_capacity(n_layers);
        for (n1, a, q, k, v, o, n2, ffn, dense) in raw {
            let ffn = match (ffn, dense) {
                (_, Some((gate, up, down))) => Ffn::Dense { gate, up, down },
                (Block::Moe(e), None) => {
                    let bytes: Vec<u8> = e.router.iter().flat_map(|v| v.to_le_bytes()).collect();
                    let router = g.upload("F32", &bytes, e.n_expert, m.n_embd)?;
                    let gx = e.gate_exps.as_ref().expect("gated experts");
                    let on = (0..e.n_expert).map(|x| Some((there(&gx.0[x], dev)?, there(&e.up_exps.0[x], dev)?, there(&e.down_exps.0[x], dev)?))).collect();
                    Ffn::Moe(GMoe { m: e, router, on })
                }
                _ => unreachable!("checked above"),
            };
            layers.push(Layer {
                attn: Attn { norm: g.buffer(n1), q, k, v, o, q_norm: a.q_norm.as_ref().map(|w| g.buffer(w)), k_norm: a.k_norm.as_ref().map(|w| g.buffer(w)),
                             n_head: a.n_head as u32, n_kv: a.n_kv as u32, hd: a.head_dim as u32 },
                ffn_norm: g.buffer(n2),
                ffn,
            });
        }
        let gates = looped.map_or(vec![], |l| l.gates.iter().map(|w| g.buffer(w)).collect());
        Ok(Graph { g, layers, kv, gates, out_norm: g.buffer(&m.out_norm), output, n_rot: rope.n_dims as u32, neox: rope.neox, max_ctx })
    }

    /// How many routed experts are on the graph's device, and how many there are (for the INFO line).
    pub fn experts_on(&self) -> (usize, usize) {
        self.layers.iter().filter_map(|l| match &l.ffn { Ffn::Moe(e) => Some(e), _ => None })
            .fold((0, 0), |(a, b), e| (a + e.on.iter().filter(|o| o.is_some()).count(), b + e.on.len()))
    }

    /// AM2_GPU_BENCH=1: how long each kind of kernel takes on this device, each queued `n` times before one read (so the
    /// time per call is the device's plus the queueing, without a wait per call). Prints one line per kernel to stderr.
    pub fn bench(&self, m: &Model) {
        let g = self.g;
        let n = 50;
        let time = |what: &str, bytes: usize, f: &dyn Fn() -> Buf| {
            let warm = f();
            g.read(&warm);
            let t = std::time::Instant::now();
            let mut last = f();
            for _ in 1..n {
                last = f();
            }
            g.read(&last);
            let us = t.elapsed().as_secs_f64() * 1e6 / n as f64;
            let gbs = if bytes > 0 { format!(" {:.1} GB/s", bytes as f64 / us / 1e3) } else { String::new() };
            eprintln!("GPU_BENCH {what:<28} {us:9.1} us/call{gbs}");
        };
        let l0 = &self.layers[0];
        let a = &l0.attn;
        let kv = self.kv[0].as_ref().expect("layer 0's cache");
        let dim = m.n_embd;
        let x = g.buffer(&vec![0.01f32; dim]);
        let mut mats = vec![("attn_q", a.q), ("attn_k", a.k), ("attn_v", a.v), ("attn_o", a.o), ("output", self.output)];
        if let Ffn::Dense { gate, up, down } = &l0.ffn {
            mats.extend([("ffn_gate", *gate), ("ffn_up", *up), ("ffn_down", *down)]);
        }
        for (name, w) in &mats {
            let xin = g.buffer(&vec![0.01f32; w.cols as usize]);
            time(&format!("mv {name} {}x{}", w.rows, w.cols), w.bytes, &|| g.mv(w, &xin));
        }
        time("rms_norm", 0, &|| g.rms_norm(&x, &a.norm, m.eps));
        time("add_into", 0, &|| { g.add_into(&x, &x); x.clone() });
        let rope = m.rope.as_ref().expect("rope");
        let tab = g.buffer(&rope.table(5));
        let q = g.scratch(a.q.rows as usize);
        time("rope q", 0, &|| { g.rope(&tab, &q, a.hd, self.n_rot, self.neox); q.clone() });
        let k = g.scratch(a.k.rows as usize);
        time("kv_append", 0, &|| { g.kv_append(&k, &kv.kc, 0); k.clone() });
        for ctx in [16u32, 128, 1024] {
            time(&format!("attend n={ctx}"), 0, &|| g.attend(&q, &kv.kc, &kv.vc, &kv.s, ctx, a.n_head, a.n_kv, a.hd, 0.1));
        }
        time("buffer upload (rope table)", 0, &|| g.buffer(&rope.table(5)));
        let mut st = m.new_state();
        let t = std::time::Instant::now();
        for i in 0..20 {
            self.forward(m, &mut st, (1000 + i) % m.embd.rows as u32);
        }
        eprintln!("GPU_BENCH whole token (forward + read)  {:9.1} us/token, {} layers", t.elapsed().as_secs_f64() * 1e6 / 20.0, self.layers.len());
    }

    /// Whether prompts can go through [`Graph::forward_batch`]: CubeCL devices always; the native Metal backend when every
    /// head fits its chunk attention kernel (at most 256 values, a divisor of its 256 threads).
    pub fn batches(&self) -> bool {
        !self.g.is_native() || self.layers.iter().all(|l| l.attn.hd <= 256 && 256 % l.attn.hd == 0)
    }

    /// One token: the logits of the next one. The state only counts positions: K and V are on the device.
    pub fn forward(&self, m: &Model, st: &mut State, token: u32) -> Vec<f32> {
        self.run(m, st, &[token], false).pop().expect("a token")
    }

    /// A chunk of prompt tokens at once: each matrix applied to all of them in one dispatch (its weights read once per
    /// chunk), each chosen expert once for the tokens that chose it, RMS norm and RoPE per token with its own position,
    /// K and V appended for the whole chunk, then causal attention per token. -> the logits after the last token, as
    /// `forward` would give them token by token.
    pub fn forward_batch(&self, m: &Model, st: &mut State, tokens: &[u32]) -> Vec<f32> {
        self.chunks(m, st, tokens, false).pop().expect("a token")
    }

    /// `forward_batch` with the logits after every token.
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn forward_batch_all(&self, m: &Model, st: &mut State, tokens: &[u32]) -> Vec<Vec<f32>> {
        self.chunks(m, st, tokens, true)
    }

    /// A chunk split so that one layer's attention scores (tokens x heads x positions, f32) stay within 128 MiB.
    fn chunks(&self, m: &Model, st: &mut State, tokens: &[u32], all: bool) -> Vec<Vec<f32>> {
        let heads = self.layers.iter().map(|l| l.attn.n_head as usize).max().unwrap_or(1);
        let mut outs = Vec::new();
        let mut i = 0;
        while i < tokens.len() {
            let mut n = tokens.len() - i;
            while n > 1 && n * heads * (st.pos + n) * 4 > 128 << 20 {
                n /= 2;
            }
            let mut got = self.run(m, st, &tokens[i..i + n], all);
            i += n;
            if all || i == tokens.len() {
                outs.append(&mut got);
            }
        }
        outs
    }

    /// `tokens` through the model at positions st.pos..: -> the logits after each (`all`) or after the last.
    fn run(&self, m: &Model, st: &mut State, tokens: &[u32], all: bool) -> Vec<Vec<f32>> {
        let g = self.g;
        let base = st.pos;
        let nt = tokens.len();
        assert!(base + nt <= self.max_ctx, "--graph: {} tokens are past the context of {}", base + nt, self.max_ctx);
        let rope = m.rope.as_ref().expect("rope");
        let tab = g.buffer(&(0..nt).flat_map(|i| rope.table(base + i)).collect::<Vec<f32>>());
        let x = g.buffer(&tokens.iter().flat_map(|&t| m.embd.row(t as usize)).collect::<Vec<f32>>());
        let dim = m.n_embd;
        // the attention's scale per token (the temperature of Mistral 3 at its position): a buffer for a chunk
        let hd0 = self.layers[0].attn.hd;
        let scales = (nt > 1).then(|| g.buffer(&(0..nt).map(|i| m.temp_scale_at(base + i) / (hd0 as f32).sqrt()).collect::<Vec<f32>>()));
        let n_layers = self.layers.len();
        let mut saved: Option<Buf> = None;
        for step in m.schedule() {
            let (l, pass) = match step {
                Step::Layer(l, p) => (l, p),
                Step::Save => {
                    let s = g.scratch(x.len);
                    g.copy(&x, &s, 0, 0, x.len);
                    saved = Some(s);
                    continue;
                }
                Step::Mix(p) => {
                    g.loop_mix(saved.as_ref().expect("saved before the pass"), &x, &self.gates[p - 1]);
                    continue;
                }
            };
            let layer = &self.layers[l];
            let kv = self.kv[pass * n_layers + l].as_ref().expect("a cache for each layer of each pass");
            self.attention(m, &layer.attn, kv, &tab, scales.as_ref(), &x, base, nt);
            let h = g.rms_norm(&x, &layer.ffn_norm, m.eps);
            match &layer.ffn {
                Ffn::Dense { gate, up, down } => {
                    let (u, gv) = (g.mv_tokens(up, &h, nt as u32), g.mv_tokens(gate, &h, nt as u32));
                    g.silu_gate(&gv, &u);
                    g.add_into(&g.mv_tokens(down, &u, nt as u32), &x);
                }
                Ffn::Moe(e) => self.moe(m, e, &h, &x, nt),
            }
        }
        st.pos = base + nt;
        if all {
            let logits = g.read(&g.mv_tokens(self.output, &g.rms_norm(&x, &self.out_norm, m.eps), nt as u32));
            return logits.chunks_exact(logits.len() / nt).map(|c| c.to_vec()).collect();
        }
        let last = if nt == 1 { x } else { g.row(&x, (nt - 1) * dim, dim) };
        vec![g.read(&g.mv(self.output, &g.rms_norm(&last, &self.out_norm, m.eps)))]
    }

    #[allow(clippy::too_many_arguments)]
    fn attention(&self, m: &Model, a: &Attn, kv: &Kv, tab: &Buf, scales: Option<&Buf>, x: &Buf, base: usize, nt: usize) {
        let g = self.g;
        let n = nt as u32;
        let h = g.rms_norm(x, &a.norm, m.eps);
        let (mut q, mut k, v) = (g.mv_tokens(a.q, &h, n), g.mv_tokens(a.k, &h, n), g.mv_tokens(a.v, &h, n));
        if let Some(w) = &a.q_norm {
            q = g.rms_norm(&q, w, m.eps);
        }
        if let Some(w) = &a.k_norm {
            k = g.rms_norm(&k, w, m.eps);
        }
        let row = (a.n_kv * a.hd) as usize;
        let att = match scales {
            None => {
                g.rope(tab, &q, a.hd, self.n_rot, self.neox);
                g.rope(tab, &k, a.hd, self.n_rot, self.neox);
                g.kv_append(&k, &kv.kc, base * row);
                g.kv_append(&v, &kv.vc, base * row);
                let ts = m.temp_scale_at(base);
                g.attend(&q, &kv.kc, &kv.vc, &kv.s, base as u32 + 1, a.n_head, a.n_kv, a.hd, ts / (a.hd as f32).sqrt())
            }
            Some(sc) => {
                g.rope_tokens(tab, &q, a.hd, self.n_rot, self.neox, n);
                g.rope_tokens(tab, &k, a.hd, self.n_rot, self.neox, n);
                g.kv_append(&k, &kv.kc, base * row);
                g.kv_append(&v, &kv.vc, base * row);
                g.attend_tokens(&q, &kv.kc, &kv.vc, sc, base as u32, n, a.n_head, a.n_kv, a.hd)
            }
        };
        g.add_into(&g.mv_tokens(a.o, &att, n), x);
    }

    /// x += the routed experts of `h` (nt tokens): routing on the device, one read of the choice, the experts on the device
    /// queued (each once for all the tokens that chose it), the others computed on the CPU meanwhile, every output put in
    /// its (token, slot) row, then the weighted sum in routing order.
    fn moe(&self, m: &Model, e: &GMoe, h: &Buf, x: &Buf, nt: usize) {
        let g = self.g;
        let mo = e.m;
        let (k, dim) = (mo.top_k, m.n_embd);
        let logits = g.mv_tokens(&e.router, h, nt as u32);
        let routes = g.route_topk(&logits, mo.n_expert as u32, k as u32, mo.w_norm, mo.w_scale);
        // the one wait of the layer: 2k values per token
        let r = g.read(&routes);
        let mut users: Vec<Vec<usize>> = vec![Vec::new(); mo.n_expert];
        for t in 0..nt {
            for s in 0..k {
                let ex = r[t * 2 * k + s] as usize;
                users[ex].push(t * k + s);
                m.heat[mo.layer * m.profile_experts + ex].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
        }
        let slots = g.scratch(nt * k * dim);
        let here: Vec<usize> = (0..mo.n_expert).filter(|&ex| !users[ex].is_empty() && e.on[ex].is_none()).collect();
        // the CPU's experts need the input: read now (the device is idle after the read above, so no wait), computed
        // after the device's experts are queued, so both run at once
        let hv = (!here.is_empty()).then(|| g.read(h));
        for (ex, (gate, up, down)) in e.on.iter().enumerate().filter_map(|(ex, o)| o.map(|o| (ex, o))) {
            let us = &users[ex];
            if us.is_empty() {
                continue;
            }
            let n = us.len() as u32;
            let xin = if nt == 1 { h.clone() } else { g.gather_rows(h, &g.buffer(&us.iter().map(|&ts| (ts / k) as f32).collect::<Vec<_>>()), dim) };
            let (u, gv) = (g.mv_tokens(up, &xin, n), g.mv_tokens(gate, &xin, n));
            g.silu_gate(&gv, &u);
            let o = g.mv_tokens(down, &u, n);
            if n == 1 {
                g.copy(&o, &slots, 0, us[0] * dim, dim);
            } else {
                g.scatter_rows(&o, &g.buffer(&us.iter().map(|&ts| ts as f32).collect::<Vec<_>>()), &slots, dim);
            }
        }
        if let Some(hv) = hv {
            m.prefetch_experts(mo, here.iter().copied());
            let ins: Vec<Vec<Vec<f32>>> = here.iter().map(|&ex| users[ex].iter().map(|&ts| hv[ts / k * dim..(ts / k + 1) * dim].to_vec()).collect()).collect();
            let jobs: Vec<(usize, &[Vec<f32>])> = here.iter().zip(&ins).map(|(&ex, xs)| (ex, &xs[..])).collect();
            let outs = mo.experts_here(&jobs);
            let rows: Vec<f32> = outs.iter().flatten().flatten().copied().collect();
            let dst: Vec<f32> = here.iter().flat_map(|&ex| users[ex].iter().map(|&ts| ts as f32)).collect();
            g.scatter_rows(&g.buffer(&rows), &g.buffer(&dst), &slots, dim);
        }
        g.moe_combine(&slots, &routes, x, k as u32);
    }
}

#[cfg(test)]
mod tests {
    //! The graph against the CPU path, on a device: AM2_TEST_DEVICE (default cuda:1) when it opens, else the tests say so
    //! and pass. Each fixture's logits token by token, a chunk at a time and a chunk then tokens, on the device, equal
    //! the CPU path's token by token within TOL, with the same best token at every step.
    use super::*;
    use crate::testgguf::{self, Spec};
    use std::path::{Path, PathBuf};
    use std::sync::Mutex;

    /// The largest difference allowed between the graph's logits and the CPU path's: the CPU keeps K and V in f16 (the
    /// graph f32); on these f32 fixtures that is about 1e-3.
    const TOL: f32 = 3e-3;

    /// One placement at a time: the devices are opened once per process (gpu.rs DEVICES).
    static ONE: Mutex<()> = Mutex::new(());

    fn device() -> Option<String> {
        let d = std::env::var("AM2_TEST_DEVICE").unwrap_or_else(|_| "cuda:1".into());
        if d == "none" || strata_gpu::Gpu::open(&d).is_err() {
            eprintln!("no device {d}: the graph tests are skipped");
            return None;
        }
        Some(d)
    }

    /// The model placed on the test device: everything, then this share of the routed experts kept there (1.0 all of
    /// them, 0.0 none: they are the CPU's), the first layers' first.
    fn placed(path: &Path, dev: &str, experts: f64) -> Model {
        let (mut m, _) = Model::load(path, None).unwrap();
        crate::gpu::place(&mut m, &[(dev.to_string(), 2 << 30)], None, 0).unwrap();
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

    const TOKENS: [u32; 11] = [1, 17, 200, 3, 42, 42, 99, 7, 128, 5, 64];

    /// The CPU path's logits token by token against the graph's token by token, a chunk at a time, a chunk then
    /// tokens, and a chunk's last. -> (the worst difference, the experts on the device and in all).
    fn check(name: &str, path: &Path, dev: &str, experts: f64, passes: usize) -> (f32, (usize, usize)) {
        let (cpu, _) = Model::load(path, None).unwrap();
        cpu.loop_passes.store(passes, std::sync::atomic::Ordering::Relaxed);
        let toks: Vec<u32> = TOKENS.iter().map(|&t| t % cpu.embd.rows as u32).collect();
        let mut st = cpu.new_state();
        let want: Vec<Vec<f32>> = toks.iter().map(|&t| cpu.forward(&mut st, t)).collect();
        let m = placed(path, dev, experts);
        m.loop_passes.store(passes, std::sync::atomic::Ordering::Relaxed);
        let gr = Graph::new(&m, 64).unwrap();
        let mut st = m.new_state();
        let one: Vec<Vec<f32>> = toks.iter().map(|&t| gr.forward(&m, &mut st, t)).collect();
        let mut st = m.new_state();
        let chunk = gr.forward_batch_all(&m, &mut st, &toks);
        let mut st = m.new_state();
        let mut mixed = gr.forward_batch_all(&m, &mut st, &toks[..5]);
        mixed.extend(toks[5..].iter().map(|&t| gr.forward(&m, &mut st, t)));
        let mut st = m.new_state();
        let last = gr.forward_batch(&m, &mut st, &toks);
        let d = [max_diff(&one, &want), max_diff(&chunk, &want), max_diff(&mixed, &want), max_diff(&[last], &want[want.len() - 1..])];
        eprintln!("{name}: experts {:?} on the device, passes={passes}: max |graph - cpu| one={:.2e} chunk={:.2e} mixed={:.2e} last={:.2e}",
                  gr.experts_on(), d[0], d[1], d[2], d[3]);
        for (i, (a, b)) in one.iter().zip(&want).enumerate() {
            assert_eq!(argmax(a), argmax(b), "{name}: token {i}'s best logit");
        }
        let worst = d.iter().copied().fold(0.0, f32::max);
        assert!(worst < TOL, "{name}: {d:?}");
        (worst, gr.experts_on())
    }

    fn workshop_loop() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../strata-workshop/tests/fixtures/tiny-qwen3-loop/model.f32.gguf")
    }

    #[test]
    fn qwen3_and_its_loop_on_the_graph_equal_the_cpu() {
        let _one = ONE.lock().unwrap_or_else(|e| e.into_inner());
        let Some(dev) = device() else { return };
        let plain = testgguf::tmp("tiny-qwen3", &Spec::tiny_qwen3());
        check("tiny qwen3", &plain, &dev, 1.0, 0);
        let lp = workshop_loop();
        check("tiny qwen3 loop", &lp, &dev, 1.0, 0);
        check("tiny qwen3 loop, one pass", &lp, &dev, 1.0, 1);
        check("tiny qwen3 loop, two passes", &lp, &dev, 1.0, 2);
        let own = testgguf::tmp("tiny-qwen3-loop", &Spec { loop_: Some((1, 3, 3)), seed: 0x5EED_0003, ..Spec::tiny_qwen3() });
        check("tiny qwen3 loop (testgguf)", &own, &dev, 1.0, 0);
        let llama = testgguf::tmp("tiny-llama", &Spec { arch: "llama", seed: 0x5EED_0004, ..Spec::tiny_qwen3() });
        check("tiny llama", &llama, &dev, 1.0, 0);
    }

    /// The looped fixture on the graph against the Workshop's own (f32) forward pass, which the CPU path matches
    /// within 2e-3 (model/loop_tests.rs).
    #[test]
    fn the_graph_runs_the_loop_as_the_workshop_trains_it() {
        let _one = ONE.lock().unwrap_or_else(|e| e.into_inner());
        let Some(dev) = device() else { return };
        let d = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../strata-workshop/tests/fixtures/tiny-qwen3-loop");
        let want: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(d.join("logits.json")).unwrap()).unwrap();
        let toks: Vec<u32> = want["tokens"].as_array().unwrap().iter().map(|t| t.as_u64().unwrap() as u32).collect();
        let looped: Vec<Vec<f32>> = want["looped"].as_array().unwrap().iter()
            .map(|r| r.as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect()).collect();
        let m = placed(&d.join("model.f32.gguf"), &dev, 1.0);
        let gr = Graph::new(&m, 64).unwrap();
        let mut st = m.new_state();
        let one: Vec<Vec<f32>> = toks.iter().map(|&t| gr.forward(&m, &mut st, t)).collect();
        let mut st = m.new_state();
        let chunk = gr.forward_batch_all(&m, &mut st, &toks);
        let (a, b) = (max_diff(&one, &looped), max_diff(&chunk, &looped));
        eprintln!("graph vs the Workshop's looped logits: one={a:.2e} chunk={b:.2e}");
        assert!(a < 2e-3 && b < 2e-3, "{a} {b}");
    }

    /// Softmax-routed experts on the graph: all on the device, none (the CPU computes them, the activations cross), and
    /// some of each; also with the loop.
    #[test]
    fn experts_on_the_graph_equal_the_cpu() {
        let _one = ONE.lock().unwrap_or_else(|e| e.into_inner());
        let Some(dev) = device() else { return };
        let moe = testgguf::tmp("tiny-qwen3moe", &Spec::tiny_qwen3moe());
        let (_, on) = check("tiny qwen3moe", &moe, &dev, 1.0, 0);
        assert_eq!(on.0, on.1, "every expert on the device");
        let (_, on) = check("tiny qwen3moe, experts on the CPU", &moe, &dev, 0.0, 0);
        assert_eq!(on.0, 0);
        let (_, on) = check("tiny qwen3moe, half the experts on the device", &moe, &dev, 0.5, 0);
        assert!(on.0 > 0 && on.0 < on.1, "{on:?}");
        let lp = testgguf::tmp("tiny-qwen3moe-loop", &Spec { loop_: Some((1, 3, 2)), seed: 0x5EED_0005, ..Spec::tiny_qwen3moe() });
        check("tiny qwen3moe loop, half on the device", &lp, &dev, 0.5, 0);
        let mixtral = testgguf::tmp("tiny-mixtral", &Spec { arch: "llama", seed: 0x5EED_0006, ..Spec::tiny_qwen3moe() });
        check("tiny mixtral-style llama", &mixtral, &dev, 1.0, 0);
    }
}
