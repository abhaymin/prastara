//! `inkling2` and `inkling2_loop` (Inkling 2): attention without RoPE (a learned relative-position bias, an attention
//! sink, clipped queries and keys), short causal convolutions with a recurrent state on K, V, the attention output and
//! the expert output, residual streams that restart every block and are mixed over the depth by learned queries, SiTU
//! experts with a fused gate/up matrix, and - in the loop variant - a core of layers run several times with the same
//! weights, each pass with its own caches, the passes joined by small controllers' messages.
//! One token at a time, as llama.cpp's src/models/inkling2.cpp and inkling2-loop.cpp compute it (the parity oracle).

use super::*;

pub(crate) struct InkLayer {
    pub(crate) swa: bool,
    pub(crate) n_head: usize,
    pub(crate) n_kv: usize,
    pub(crate) hd: usize,
    pub(crate) wq: QMat,
    pub(crate) wk: QMat,
    pub(crate) wv: QMat,
    pub(crate) wo: QMat,
    pub(crate) gate: QMat,
    pub(crate) q_clip: Vec<f32>,
    pub(crate) sink: QMat,
    pub(crate) rel: QMat,
    /// rows: a distance; columns: the relative features
    pub(crate) rel_bias: QMat,
    pub(crate) rel_clip: Vec<f32>,
    pub(crate) k_conv: Vec<f32>,
    pub(crate) v_conv: Vec<f32>,
    pub(crate) post_conv: Vec<f32>,
    pub(crate) ffn_conv: Vec<f32>,
    /// one norm (windowed layers) or one each for the Q, K and V inputs (full-attention layers)
    pub(crate) norms: Vec<Vec<f32>>,
    pub(crate) k_proj: Option<QMat>,
    pub(crate) v_proj: Option<QMat>,
    /// the depth-mixing queries of the attention inputs (1 or 3), none in the first layer
    pub(crate) res_query: Option<Vec<f32>>,
    pub(crate) ffn_query: Vec<f32>,
    pub(crate) ffn_norm: Vec<f32>,
    pub(crate) router: QMat,
    pub(crate) probs_b: Option<Vec<f32>>,
    pub(crate) gate_up: Experts,
    pub(crate) down: Experts,
    pub(crate) top_k: usize,
}

pub(crate) struct InkCtrl {
    pub(crate) down: QMat,
    pub(crate) gain: QMat,
    pub(crate) forget: QMat,
}

/// A looped layer's controllers: the queries that read the previous pass (Q, K, V, FFN) and the gates that mix it in.
pub(crate) struct InkLoop {
    pub(crate) msg_query: Vec<f32>,
    pub(crate) ctrl: Vec<InkCtrl>,
}

pub(crate) struct Inkling {
    pub(crate) layers: Vec<InkLayer>,
    pub(crate) loops: Vec<InkLoop>,
    /// (start, end, passes) of the looped core
    pub(crate) core: Option<(usize, usize, usize)>,
    pub(crate) block: usize,
    pub(crate) n_conv: usize,
    pub(crate) n_rel: usize,
    pub(crate) extent: usize,
    pub(crate) n_swa: usize,
    pub(crate) qk_clip: f32,
    pub(crate) rel_clip: f32,
    pub(crate) floor: f32,
    pub(crate) alpha: f32,
    pub(crate) situ_beta: f32,
    pub(crate) situ_lb: f32,
    pub(crate) w_scale: f32,
    pub(crate) softcap: f32,
    pub(crate) n_vocab: usize,
    pub(crate) tok_norm: Vec<f32>,
    pub(crate) out_query: Vec<f32>,
}

/// One cache slot (a layer in one pass): the keys and values of every token so far, and the convolutions' last inputs
/// (per channel the n_conv - 1 before this token, oldest first; K, then V, then the attention and expert outputs).
#[derive(Clone, Default)]
pub(crate) struct InkSlot {
    k: Vec<f32>,
    v: Vec<f32>,
    pub(crate) hist: Vec<f32>,
}

impl InkSlot {
    pub(crate) fn cut(&mut self, from: usize, to: usize) {
        self.k.truncate(self.k.len() / from * to);
        self.v.truncate(self.v.len() / from * to);
    }
}

fn tanh_f32(x: f32) -> f32 {
    (x as f64).tanh() as f32
}

/// x / sqrt(mean(x^2) + eps)
fn rmsn(x: &[f32], eps: f32) -> Vec<f32> {
    let ms = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let k = 1.0 / (ms + eps).sqrt();
    x.iter().map(|v| v * k).collect()
}

/// Each `d`-long row scaled down to an L2 norm of at most `thr`, after the optional per-channel scale.
fn clip(x: &mut [f32], d: usize, scale: Option<&[f32]>, thr: f32) {
    if let Some(s) = scale {
        x.iter_mut().zip(s).for_each(|(v, s)| *v *= s);
    }
    for row in x.chunks_mut(d) {
        let norm = row.iter().map(|v| v * v).sum::<f32>().sqrt() + 1e-6;
        let f = (thr / norm).clamp(0.0, 1.0);
        row.iter_mut().for_each(|v| *v *= f);
    }
}

/// Attention over the depth: the earlier block outputs and `cur`, normalized, mixed by each query's softmax.
fn mix(blocks: &[Vec<f32>], cur: &[f32], queries: &[f32]) -> Vec<Vec<f32>> {
    let d = cur.len();
    let states: Vec<Vec<f32>> = blocks.iter().map(|b| rmsn(b, 1e-6)).chain(std::iter::once(rmsn(cur, 1e-6))).collect();
    let scale = 1.0 / (d as f32).sqrt();
    queries.chunks(d).map(|q| {
        let mut p: Vec<f32> = states.iter().map(|s| dot(s, q) * scale).collect();
        softmax(&mut p);
        let mut out = vec![0f32; d];
        for (s, p) in states.iter().zip(&p) {
            out.iter_mut().zip(s).for_each(|(o, v)| *o += p * v);
        }
        out
    }).collect()
}

/// A causal short convolution with a residual. `hist` holds each channel's n_conv - 1 earlier inputs and takes this
/// one; `dynamic` adds input-dependent taps per head (lag 0 is this token).
fn convolve(cur: &[f32], weight: &[f32], dynamic: Option<&[f32]>, hist: &mut [f32], n_conv: usize) -> Vec<f32> {
    let n = cur.len();
    let past = n_conv - 1;
    let k = 1.0 / (((if dynamic.is_some() { 2 } else { 1 }) * n_conv + 1) as f32).sqrt();
    let per_head = dynamic.map(|d| n / (d.len() / n_conv));
    let mut out = vec![0f32; n];
    for c in 0..n {
        let h = &mut hist[c * past..(c + 1) * past];
        let w = &weight[c * n_conv..(c + 1) * n_conv];
        let mut acc = cur[c] * w[past];
        for j in 0..past {
            acc += h[j] * w[j];
        }
        if let (Some(d), Some(ph)) = (dynamic, per_head) {
            let taps = &d[(c / ph) * n_conv..(c / ph + 1) * n_conv];
            acc += cur[c] * taps[0];
            for lag in 1..n_conv {
                acc += h[past - lag] * taps[lag];
            }
        }
        out[c] = (cur[c] + acc) * k;
        h.copy_within(1.., 0);
        h[past - 1] = cur[c];
    }
    out
}

fn dump(name: &str, slot: usize, v: &[f32]) {
    static DIR: std::sync::OnceLock<Option<std::path::PathBuf>> = std::sync::OnceLock::new();
    if let Some(dir) = DIR.get_or_init(|| std::env::var_os("AM2_INK_DUMP").map(std::path::PathBuf::from)) {
        use std::io::Write;
        let mut f = std::fs::OpenOptions::new().create(true).append(true).open(dir.join(format!("{name}-{slot}.f32"))).expect("dump file");
        let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
        f.write_all(&bytes).expect("dump");
    }
}

/// Whether K, V and the attention mask are rounded to f16 as llama.cpp keeps them (the default); AM2_INK_F32_CACHE=1
/// keeps them f32, on the CPU path and the device graph alike: the graph's parity tests use it to separate the
/// arithmetic from the f16 roundings, which turn the paths' last-bit differences into a few 1e-3 now and then.
pub(crate) fn f16_cache() -> bool {
    match F16_CACHE.load(Ordering::Relaxed) {
        0 => {
            let on = !std::env::var("AM2_INK_F32_CACHE").is_ok_and(|v| v == "1");
            F16_CACHE.store(if on { 1 } else { 2 }, Ordering::Relaxed);
            on
        }
        v => v == 1,
    }
}

/// 0: not read yet, 1: f16, 2: f32.
static F16_CACHE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

/// Tests: the f16 roundings on or off (hold CACHE_MODE meanwhile).
#[cfg(test)]
pub(crate) fn set_f16_cache(on: bool) {
    F16_CACHE.store(if on { 1 } else { 2 }, Ordering::Relaxed);
}

/// Tests that run either path hold this, so that one switching the roundings does not change another's run.
#[cfg(test)]
pub(crate) static CACHE_MODE: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// A matrix's values as f32 (the device graph uploads the small ones this way).
pub(crate) fn f32s(q: &QMat) -> Vec<f32> {
    q.all_f32()
}

impl Inkling {
    pub(crate) fn load(g: &Gguf, arch: &str, tensors: &HashMap<String, QMat>) -> Result<Inkling, String> {
        let k = |n: &str| format!("{arch}.{n}");
        let u = |n: &str| g.u64(&k(n)).ok_or_else(|| format!("{} is missing", k(n)));
        let fl = |n: &str| g.metadata.get(&k(n)).and_then(|v| v.as_f64()).map(|v| v as f32).ok_or_else(|| format!("{} is missing", k(n)));
        let take = |name: &str| tensors.get(name).cloned().ok_or_else(|| format!("tensor {name} is missing"));
        let vecf = |name: &str| take(name).map(|t| t.all_f32());
        let n_layer = u("block_count")? as usize;
        let hd = u("attention.key_length")? as usize;
        let pattern: Vec<bool> = match g.metadata.get(&k("attention.sliding_window_pattern")) {
            Some(serde_json::Value::Array(a)) => a.iter().map(|v| v.as_bool().unwrap_or_else(|| v.as_u64().unwrap_or(0) != 0)).collect(),
            _ => return Err(format!("{} is missing", k("attention.sliding_window_pattern"))),
        };
        if pattern.len() != n_layer {
            return Err(format!("{arch}: the sliding-window pattern has {} entries for {n_layer} layers", pattern.len()));
        }
        let mut layers = Vec::with_capacity(n_layer);
        for l in 0..n_layer {
            let p = |n: &str| format!("blk.{l}.{n}.weight");
            let swa = pattern[l];
            let router = take(&p("ffn_gate_inp"))?;
            let n_expert = router.rows;
            let norms = if swa {
                vec![vecf(&p("attn_norm"))?]
            } else {
                vec![vecf(&p("attn_q_inp_norm"))?, vecf(&p("attn_k_inp_norm"))?, vecf(&p("attn_v_inp_norm"))?]
            };
            layers.push(InkLayer {
                swa,
                n_head: arr_u64(g, &k("attention.head_count"), l) as usize,
                n_kv: arr_u64(g, &k("attention.head_count_kv"), l) as usize,
                hd,
                wq: take(&p("attn_q"))?, wk: take(&p("attn_k"))?, wv: take(&p("attn_v"))?, wo: take(&p("attn_output"))?,
                gate: take(&p("attn_gate"))?, q_clip: vecf(&p("attn_q_clip"))?, sink: take(&p("attn_sink_proj"))?,
                rel: take(&p("attn_rel"))?, rel_bias: take(&p("attn_rel_bias"))?, rel_clip: vecf(&p("attn_rel_clip"))?,
                k_conv: vecf(&p("attn_k_conv"))?, v_conv: vecf(&p("attn_v_conv"))?, post_conv: vecf(&p("attn_post_conv"))?,
                ffn_conv: vecf(&p("ffn_post_conv"))?, norms,
                k_proj: if swa { None } else { Some(take(&p("attn_k_conv_proj"))?) },
                v_proj: if swa { None } else { Some(take(&p("attn_v_conv_proj"))?) },
                res_query: if l > 0 { Some(vecf(&p("attn_res_query"))?) } else { None },
                ffn_query: vecf(&p("ffn_res_query"))?, ffn_norm: vecf(&p("ffn_norm"))?,
                probs_b: tensors.get(&format!("blk.{l}.exp_probs_b.bias")).map(|t| t.all_f32()),
                gate_up: Experts::named(tensors, &p("ffn_gate_up_exps"), n_expert).ok_or_else(|| format!("tensor {} is missing", p("ffn_gate_up_exps")))?,
                down: Experts::named(tensors, &p("ffn_down_exps"), n_expert).ok_or_else(|| format!("tensor {} is missing", p("ffn_down_exps")))?,
                router,
                top_k: arr_u64(g, &k("expert_used_count"), l) as usize,
            });
        }
        let (core, loops) = if arch == "inkling2_loop" {
            let (start, end, passes) = (u("loop.start")? as usize, u("loop.end")? as usize, u("loop.count")? as usize);
            if start == 0 || start >= end || end > n_layer || passes < 1 {
                return Err(format!("{arch}: invalid loop core [{start}, {end}) with {passes} passes for {n_layer} layers"));
            }
            let mut loops = Vec::new();
            for l in start..end {
                let p = |n: &str| format!("blk.{l}.{n}.weight");
                let ctrl = ["q", "k", "v", "ffn"].iter().map(|c| Ok(InkCtrl {
                    down: take(&p(&format!("loop_{c}_ctrl_down")))?, gain: take(&p(&format!("loop_{c}_ctrl_gain")))?,
                    forget: take(&p(&format!("loop_{c}_ctrl_forget")))?,
                })).collect::<Result<Vec<_>, String>>()?;
                loops.push(InkLoop { msg_query: vecf(&p("loop_msg_query"))?, ctrl });
            }
            (Some((start, end, passes)), loops)
        } else {
            (None, vec![])
        };
        Ok(Inkling {
            layers, loops, core,
            block: u("attn_res.block_size")? as usize,
            n_conv: u("shortconv.l_cache")? as usize,
            n_rel: u("attention.relative_length")? as usize,
            extent: u("attention.relative_extent")? as usize,
            n_swa: u("attention.sliding_window")? as usize,
            qk_clip: fl("attention.qk_clip")?, rel_clip: fl("attention.relative_clip")?,
            floor: u("attention.log_scale_floor")? as f32, alpha: fl("attention.log_scale_alpha")?,
            situ_beta: fl("activation.situ_beta")?, situ_lb: fl("activation.situ_linear_beta")?,
            w_scale: fl("expert_weights_scale").unwrap_or(0.0), softcap: fl("final_logit_softcapping").unwrap_or(0.0),
            n_vocab: u("vocab_size_unpadded")? as usize,
            tok_norm: vecf("token_embd_norm.weight")?, out_query: vecf("output_res_query.weight")?,
        })
    }

    /// The cache slots: one per layer, and one more per looped layer for each pass after the first.
    pub(crate) fn n_slots(&self) -> usize {
        self.layers.len() + self.core.map_or(0, |(s, e, p)| (p - 1) * (e - s))
    }

    pub(crate) fn n_expert(&self) -> usize {
        self.layers.iter().map(|l| l.router.rows).max().unwrap_or(0)
    }

    /// Every matrix a token uses, and each (layer, expert)'s own, for the GPU placement (gpu.rs).
    pub(crate) fn matrices_mut(&mut self) -> (Vec<&mut QMat>, Vec<((usize, usize), Vec<&mut QMat>)>) {
        let mut every: Vec<&mut QMat> = Vec::new();
        let mut experts = Vec::new();
        for (l, ly) in self.layers.iter_mut().enumerate() {
            let InkLayer { wq, wk, wv, wo, gate, rel, k_proj, v_proj, gate_up, down, .. } = ly;
            every.extend([wq, wk, wv, wo, gate, rel]);
            every.extend(k_proj.as_mut());
            every.extend(v_proj.as_mut());
            for (e, (gu, dn)) in gate_up.0.iter_mut().zip(down.0.iter_mut()).enumerate() {
                experts.push(((l, e), vec![gu, dn]));
            }
        }
        (every, experts)
    }

    /// (layer, experts) of every layer, for the placement's order.
    pub(crate) fn expert_counts(&self) -> Vec<(usize, usize)> {
        self.layers.iter().enumerate().map(|(l, ly)| (l, ly.router.rows)).collect()
    }

    /// situ(gate, up) = beta tanh(gate / beta) sigmoid(gate) * lb tanh(up / lb), of a fused gate/up product.
    fn situ(&self, gu: &[f32]) -> Vec<f32> {
        let n_ff = gu.len() / 2;
        let (beta, lb) = (self.situ_beta, self.situ_lb);
        (0..n_ff).map(|i| {
            let (gt, up) = (gu[i], gu[n_ff + i]);
            let up = if lb > 0.0 { lb * tanh_f32(up / lb) } else { up };
            beta * tanh_f32(gt / beta) * sigmoid(gt) * up
        }).collect()
    }

    /// Expert `e` of layer `il` for several inputs, unweighted (the device graph's CPU experts): down(situ(gate_up x)).
    pub(crate) fn expert_rows(&self, il: usize, e: usize, xs: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let ly = &self.layers[il];
        let mids: Vec<Vec<f32>> = ly.gate_up.0[e].matmul(xs).iter().map(|g| self.situ(g)).collect();
        ly.down.0[e].matmul(&mids)
    }

    fn control(&self, cur: &[f32], message: &[f32], c: &InkCtrl) -> Vec<f32> {
        let mut gate = rmsn(cur, 1e-6);
        gate.extend(rmsn(message, 1e-6));
        let g = c.down.mul(&gate);
        let mut v = QMat::mul_many(&[&c.gain, &c.forget], &g);
        let forget = v.pop().unwrap();
        let gain = v.pop().unwrap();
        (0..cur.len()).map(|i| cur[i] - tanh_f32(forget[i]) * cur[i] + tanh_f32(gain[i]) * message[i]).collect()
    }

    /// One layer at cache slot `slot` (physical layer `il`, depth `ip`); `messages` in the passes after the first.
    #[allow(clippy::too_many_arguments)]
    fn layer(&self, m: &Model, slot: usize, il: usize, ip: usize, cur: Vec<f32>, blocks: &mut Vec<Vec<f32>>,
             messages: Option<(&[Vec<f32>], &InkLoop)>, st: &mut InkSlot, pos: usize) -> Vec<f32> {
        let t_layer = std::time::Instant::now();
        let ly = &self.layers[il];
        let d = m.n_embd;
        let (hd, n_head, n_kv) = (ly.hd, ly.n_head, ly.n_kv);
        let kgqa = hd * n_kv;
        let (n_conv, past) = (self.n_conv, self.n_conv - 1);
        if st.hist.is_empty() {
            st.hist = vec![0f32; (2 * kgqa + 2 * d) * past];
        }
        let inp_sa = cur;
        let mut inputs: Vec<Vec<f32>> = match &ly.res_query {
            Some(q) => {
                let mixed = mix(blocks, &inp_sa, q);
                (0..3).map(|j| mixed[if ly.swa { 0 } else { j }].clone()).collect()
            }
            None => vec![inp_sa.clone(); 3],
        };
        if let Some((msgs, lp)) = messages {
            for j in 0..3 {
                inputs[j] = self.control(&inputs[j], &msgs[j], &lp.ctrl[j]);
            }
        }
        let inputs: Vec<Vec<f32>> = (0..3).map(|j| {
            if ly.swa && messages.is_none() && j > 0 {
                return vec![];
            }
            rms_norm(&inputs[j], &ly.norms[if ly.swa { 0 } else { j }], m.eps)
        }).collect();
        let (in_q, in_k, in_v) = (&inputs[0], if inputs[1].is_empty() { &inputs[0] } else { &inputs[1] },
                                  if inputs[2].is_empty() { &inputs[0] } else { &inputs[2] });

        let mut q = ly.wq.mul(in_q);
        let k_dyn = ly.k_proj.as_ref().map(|p| p.mul(in_k));
        let v_dyn = ly.v_proj.as_ref().map(|p| p.mul(in_v));
        let mut kc = convolve(&ly.wk.mul(in_k), &ly.k_conv, k_dyn.as_deref(), &mut st.hist[..kgqa * past], n_conv);
        let vc = convolve(&ly.wv.mul(in_v), &ly.v_conv, v_dyn.as_deref(), &mut st.hist[kgqa * past..2 * kgqa * past], n_conv);
        let mut r = ly.rel.mul(in_q);
        clip(&mut q, hd, Some(&ly.q_clip), self.qk_clip);
        clip(&mut kc, hd, None, self.qk_clip);
        let mut vc: Vec<f32> = vc.chunks(hd).flat_map(|h| rmsn(h, m.eps)).collect();
        clip(&mut r, self.n_rel, Some(&ly.rel_clip), self.rel_clip);
        if !ly.swa {
            let tau = 1.0 + self.alpha * ln_f32(((pos + 1) as f32 / self.floor).max(1.0));
            q.iter_mut().for_each(|v| *v *= tau);
            r.iter_mut().for_each(|v| *v *= tau);
        }
        dump("Qcur", slot, &q);
        dump("Kcur", slot, &kc);
        dump("Vcur", slot, &vc);
        dump("Rcur", slot, &r);
        let sinks = ly.sink.mul(in_q);
        // the cache holds f16 values (llama.cpp's default cache type)
        if f16_cache() {
            f16_round(&mut kc);
            f16_round(&mut vc);
        }
        st.k.extend_from_slice(&kc);
        st.v.extend_from_slice(&vc);
        let n = st.k.len() / kgqa;
        debug_assert_eq!(n, pos + 1);
        let extent = if ly.swa { self.n_swa } else { self.extent };
        let first = if ly.swa { n.saturating_sub(self.n_swa) } else { 0 };
        let scale = 1.0 / (hd as f32).sqrt();
        let group = n_head / n_kv;
        let (kcache, vcache) = (&st.k, &st.v);
        let mut att = vec![0f32; n_head * hd];
        att.par_chunks_mut(hd).enumerate().for_each(|(h, oh)| {
            let kh = h / group;
            let qh = &q[h * hd..(h + 1) * hd];
            // the bias by distance: the relative features on the bias table; past the table's extent it is zero
            let table = ly.rel_bias.mul(&r[h * self.n_rel..(h + 1) * self.n_rel]);
            let mut s: Vec<f32> = (first..n).map(|t| {
                let dist = pos - t;
                // the bias less the sink is the attention mask, which llama.cpp keeps in f16
                let mut bias = [if dist < extent { table[dist] } else { 0.0 } - sinks[h]];
                if f16_cache() {
                    f16_round(&mut bias);
                }
                dot(qh, &kcache[(t * n_kv + kh) * hd..(t * n_kv + kh + 1) * hd]) * scale + bias[0]
            }).collect();
            // a softmax with one more logit of zero: the sink, which takes its share and gives nothing
            let mx = s.iter().cloned().fold(0f32, f32::max);
            let mut sum = exp_f32(-mx);
            for v in s.iter_mut() {
                *v = exp_f32(*v - mx);
                sum += *v;
            }
            for (i, p) in s.iter().enumerate() {
                let t = first + i;
                let vt = &vcache[(t * n_kv + kh) * hd..(t * n_kv + kh + 1) * hd];
                let p = p / sum;
                oh.iter_mut().zip(vt).for_each(|(o, v)| *o += p * v);
            }
        });
        dump("kqv_out", slot, &att);
        let g = ly.gate.mul(in_q);
        att.iter_mut().zip(&g).for_each(|(o, g)| *o *= sigmoid(*g));
        let mut cur = convolve(&ly.wo.mul(&att), &ly.post_conv, None, &mut st.hist[2 * kgqa * past..(2 * kgqa + d) * past], n_conv);
        dump("attn_out", slot, &cur);

        // the first layer of a block starts a new residual stream; the one before it joins the mixed blocks
        if ip % self.block == 0 {
            blocks.push(inp_sa);
        } else {
            cur.iter_mut().zip(&inp_sa).for_each(|(c, x)| *c += x);
        }
        let ffn_inp = cur;
        dump("ffn_inp", slot, &ffn_inp);
        if times::on() {
            times::add(2, t_layer);
        }
        let t_ffn = std::time::Instant::now();

        let mut x = mix(blocks, &ffn_inp, &ly.ffn_query).pop().unwrap();
        if let Some((msgs, lp)) = messages {
            x = self.control(&x, &msgs[3], &lp.ctrl[3]);
        }
        let x = rms_norm(&x, &ly.ffn_norm, m.eps);
        let logits = ly.router.mul(&x);
        dump("ffn_moe_logits", slot, &logits);
        let probs: Vec<f32> = logits.iter().map(|&v| sigmoid(v)).collect();
        let sel = |e: usize| probs[e] + ly.probs_b.as_ref().map_or(0.0, |b| b[e]);
        let mut order: Vec<usize> = (0..probs.len()).collect();
        order.sort_by(|&a, &b| sel(b).total_cmp(&sel(a)));
        order.truncate(ly.top_k);
        let sum = order.iter().map(|&e| probs[e]).sum::<f32>() + 1e-20;
        let ws = if self.w_scale != 0.0 { self.w_scale } else { 1.0 };
        let mut acc = vec![0f32; d];
        if m.prefetch.load(Ordering::Relaxed) {
            // the chosen experts that are on the SSD are asked for together, before the first is multiplied
            order.iter().for_each(|&e| { crate::pack::wanted_off_device(&ly.gate_up.0[e]); crate::pack::wanted_off_device(&ly.down.0[e]); });
        }
        for &e in &order {
            m.heat[il * m.profile_experts + e].fetch_add(1, Ordering::Relaxed);
            let mid = self.situ(&ly.gate_up.0[e].mul(&x));
            let w = probs[e] / sum * ws;
            acc.iter_mut().zip(&ly.down.0[e].mul(&mid)).for_each(|(a, v)| *a += w * v);
        }
        let out = convolve(&acc, &ly.ffn_conv, None, &mut st.hist[(2 * kgqa + d) * past..], n_conv);
        dump("ffn_out", slot, &out);
        if times::on() {
            times::add(5, t_ffn);
        }
        let out: Vec<f32> = ffn_inp.iter().zip(&out).map(|(a, b)| a + b).collect();
        dump("l_out", slot, &out);
        out
    }

    /// One position through the model: `x` is the token's embedding after the token norm, or an image row as it is.
    pub(crate) fn forward(&self, m: &Model, slots: &mut [InkSlot], pos: usize, x: Vec<f32>) -> Vec<f32> {
        let n_phys = self.layers.len();
        let (start, end, passes) = self.core.unwrap_or((n_phys, n_phys, 1));
        let mut blocks: Vec<Vec<f32>> = Vec::new();
        let mut cur = x;
        for il in 0..start {
            cur = self.layer(m, il, il, il, cur, &mut blocks, None, &mut slots[il], pos);
        }
        // every pass starts from the prelude's output; a pass reads the one before it through the messages
        let prelude = cur.clone();
        let prelude_blocks = blocks.clone();
        for pass in 0..passes {
            let mut bank: Vec<Vec<f32>> = Vec::new();
            let mut bank_out = Vec::new();
            if pass > 0 {
                bank = blocks[prelude_blocks.len() + 1..].to_vec();
                bank_out = std::mem::replace(&mut cur, prelude.clone());
                blocks = prelude_blocks.clone();
            }
            for ip in start..end {
                if pass == 0 {
                    cur = self.layer(m, ip, ip, ip, cur, &mut blocks, None, &mut slots[ip], pos);
                    continue;
                }
                let slot = n_phys + (pass - 1) * (end - start) + (ip - start);
                let lp = &self.loops[ip - start];
                let msgs = mix(&bank, &bank_out, &lp.msg_query);
                cur = self.layer(m, slot, ip, ip, cur, &mut blocks, Some((&msgs, lp)), &mut slots[slot], pos);
            }
        }
        for il in end..n_phys {
            cur = self.layer(m, il, il, il, cur, &mut blocks, None, &mut slots[il], pos);
        }
        let x = mix(&blocks, &cur, &self.out_query).pop().unwrap();
        let t_out = std::time::Instant::now();
        let h = rms_norm(&x, &m.out_norm, m.eps);
        let mut logits = m.output.mul(&h);
        if times::on() {
            times::add(times::OUTPUT, t_out);
        }
        if self.softcap != 0.0 {
            logits.iter_mut().for_each(|v| *v = self.softcap * tanh_f32(*v / self.softcap));
        }
        logits.truncate(self.n_vocab);
        logits
    }

    pub(crate) fn embed(&self, m: &Model, token: u32) -> Vec<f32> {
        rms_norm(&m.embd.row(token as usize), &self.tok_norm, m.eps)
    }
}

#[cfg(test)]
mod tests {
    //! The CPU path on a synthetic Inkling (testgguf `InkSpec::tiny`: the loop over layers 2-3 three times, windowed and
    //! full layers, a 16-token window and a 24-position bias table, both crossed by the 40-token prompt): the oracle the
    //! device graph (graph_inkling.rs) is checked against. Its own parity with llama.cpp is I6's (the Workshop fixtures).
    use crate::model::Model;
    use crate::testgguf::{tmp_inkling, InkSpec};

    #[test]
    fn the_cpu_path_runs_a_synthetic_inkling_consistently() {
        let _mode = super::CACHE_MODE.lock().unwrap_or_else(|e| e.into_inner());
        let p = tmp_inkling("tiny-inkling-cpu", &InkSpec::tiny());
        let (m, _) = Model::load(&p, None).unwrap();
        let ink = m.ink.as_ref().unwrap();
        assert_eq!(ink.n_slots(), 6 + 2 * 2, "6 layers, and the core's 2 layers twice more");
        let toks: Vec<u32> = (0..40).map(|i| (i * 37 + 5) % 256).collect();
        let mut st = m.new_state();
        let one: Vec<Vec<f32>> = toks.iter().map(|&t| m.forward(&mut st, t)).collect();
        assert!(one.iter().all(|l| l.len() == 250 && l.iter().all(|v| v.is_finite() && v.abs() <= 30.0)), "250 logits, soft-capped at 30");
        let spread = one.iter().map(|l| l.iter().cloned().fold(f32::MIN, f32::max) - l.iter().cloned().fold(f32::MAX, f32::min)).fold(f32::MAX, f32::min);
        assert!(spread > 0.5, "the logits are spread: {spread}");
        // the same again, as a chunk, and after a rollback
        let mut st = m.new_state();
        assert_eq!(m.forward_batch_all(&mut st, &toks), one, "a chunk is token by token");
        let mut st = m.new_state();
        toks[..20].iter().for_each(|&t| { m.forward(&mut st, t); });
        let ck = st.checkpoint();
        toks[20..30].iter().for_each(|&t| { m.forward(&mut st, t); });
        st.rollback(ck);
        let again: Vec<Vec<f32>> = toks[20..].iter().map(|&t| m.forward(&mut st, t)).collect();
        assert_eq!(again, one[20..], "a rollback forgets the tokens after it");
        // the loop's later passes change the logits: the same model without its core run again differs
        let flat = tmp_inkling("tiny-inkling-cpu-noloop", &InkSpec { loop_: None, ..InkSpec::tiny() });
        let (m2, _) = Model::load(&flat, None).unwrap();
        assert_eq!(m2.arch, "inkling2");
        let mut st = m2.new_state();
        let two: Vec<Vec<f32>> = toks.iter().map(|&t| m2.forward(&mut st, t)).collect();
        let d = one.iter().flatten().zip(two.iter().flatten()).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
        assert!(d > 0.1, "the loop moves the logits: {d}");
    }
}
