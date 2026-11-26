//! Tiny synthetic GGUFs for the device graph's tests and its speed runs: dense `qwen3` / `llama`, a `qwen3moe` with
//! softmax-routed experts, and either with the loop (Avarta). Random weights at sizes where the activations stay
//! O(1), so the logits are well spread and a router's choices are not near ties.
//!
//! `AM2_WRITE_FIXTURES=<dir> cargo test write_speed_fixtures -- --ignored` writes the larger ones the speed runs use.

use std::path::Path;
use strata_pack::gguf_write::{Kv, TensorSpec, Writer};

#[derive(Clone)]
pub struct Spec {
    pub arch: &'static str,
    pub n_embd: usize,
    pub n_head: usize,
    pub n_kv: usize,
    pub head_dim: usize,
    pub n_layer: usize,
    pub vocab: usize,
    /// a dense layer's feed-forward width
    pub ffn: usize,
    /// experts per layer (0: dense) and how many each token takes
    pub n_expert: usize,
    pub top_k: usize,
    pub expert_ffn: usize,
    /// the matrices' encoding: F32, Q8_0 or Q4_K (norms and the router are always F32)
    pub ty: &'static str,
    /// the loop: blocks start..end, passes
    pub loop_: Option<(usize, usize, usize)>,
    pub seed: u64,
    /// the matrices quantized to `ty` and written back as F32: the same model without the quantized products, the
    /// reference the device's quantized kernels are checked against
    pub dequantized: bool,
    /// sliding-window layers (0: none): the even layers but the last attend to this many tokens
    pub window: usize,
}

impl Spec {
    pub fn tiny_qwen3() -> Spec {
        Spec { arch: "qwen3", n_embd: 64, n_head: 4, n_kv: 2, head_dim: 32, n_layer: 4, vocab: 256, ffn: 96, n_expert: 0, top_k: 0,
               expert_ffn: 0, ty: "F32", loop_: None, seed: 0x5EED_0001,
               dequantized: false, window: 0 }
    }

    pub fn tiny_qwen3moe() -> Spec {
        Spec { arch: "qwen3moe", n_expert: 8, top_k: 2, expert_ffn: 32, seed: 0x5EED_0002, ..Spec::tiny_qwen3() }
    }
}

struct Rng(u64);
impl Rng {
    fn unit(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

fn ggml_type(ty: &str) -> u32 {
    match ty { "F32" => 0, "Q5_0" => 6, "Q8_0" => 8, "Q4_K" => 12, _ => panic!("testgguf: {ty}") }
}

/// Writes the model to `path`.
pub fn write(s: &Spec, path: &Path) {
    let mut rng = Rng(s.seed | 1);
    let a = s.arch;
    let qwen = a.starts_with("qwen");
    let mut kv: Vec<(String, Kv)> = vec![
        ("general.architecture".into(), Kv::Str(a.into())),
        ("general.name".into(), Kv::Str(format!("tiny-{a}"))),
        (format!("{a}.block_count"), Kv::U32(s.n_layer as u32)),
        (format!("{a}.context_length"), Kv::U32(4096)),
        (format!("{a}.embedding_length"), Kv::U32(s.n_embd as u32)),
        (format!("{a}.feed_forward_length"), Kv::U32(s.ffn as u32)),
        (format!("{a}.attention.head_count"), Kv::U32(s.n_head as u32)),
        (format!("{a}.attention.head_count_kv"), Kv::U32(s.n_kv as u32)),
        (format!("{a}.attention.key_length"), Kv::U32(s.head_dim as u32)),
        (format!("{a}.attention.value_length"), Kv::U32(s.head_dim as u32)),
        (format!("{a}.attention.layer_norm_rms_epsilon"), Kv::F32(1e-6)),
        (format!("{a}.rope.freq_base"), Kv::F32(10000.0)),
    ];
    if s.n_expert > 0 {
        kv.push((format!("{a}.expert_count"), Kv::U32(s.n_expert as u32)));
        kv.push((format!("{a}.expert_used_count"), Kv::U32(s.top_k as u32)));
        kv.push((format!("{a}.expert_feed_forward_length"), Kv::U32(s.expert_ffn as u32)));
    }
    if let Some((start, end, passes)) = s.loop_ {
        kv.push((format!("{a}.loop.start"), Kv::U32(start as u32)));
        kv.push((format!("{a}.loop.end"), Kv::U32(end as u32)));
        kv.push((format!("{a}.loop.count"), Kv::U32(passes as u32)));
    }
    if s.window > 0 {
        kv.push((format!("{a}.attention.sliding_window"), Kv::U32(s.window as u32)));
        kv.push((format!("{a}.attention.sliding_window_pattern"), Kv::Bools((0..s.n_layer).map(|l| l % 2 == 0 && l + 1 < s.n_layer).collect())));
    }
    // (name, numpy shape, F32 forced, values' scale; 0: a norm, 1 + noise)
    let mut ts: Vec<(String, Vec<usize>, bool, f32)> = Vec::new();
    let (d, q, kvd) = (s.n_embd, s.n_head * s.head_dim, s.n_kv * s.head_dim);
    let sc = |cols: usize| 1.0 / (cols as f32).sqrt();
    ts.push(("token_embd.weight".into(), vec![s.vocab, d], false, 1.0));
    ts.push(("output_norm.weight".into(), vec![d], true, 0.0));
    ts.push(("output.weight".into(), vec![s.vocab, d], false, sc(d)));
    for l in 0..s.n_layer {
        let p = |n: &str| format!("blk.{l}.{n}");
        ts.push((p("attn_norm.weight"), vec![d], true, 0.0));
        ts.push((p("attn_q.weight"), vec![q, d], false, sc(d)));
        ts.push((p("attn_k.weight"), vec![kvd, d], false, sc(d)));
        ts.push((p("attn_v.weight"), vec![kvd, d], false, sc(d)));
        ts.push((p("attn_output.weight"), vec![d, q], false, sc(q)));
        if qwen {
            ts.push((p("attn_q_norm.weight"), vec![s.head_dim], true, 0.0));
            ts.push((p("attn_k_norm.weight"), vec![s.head_dim], true, 0.0));
        }
        ts.push((p("ffn_norm.weight"), vec![d], true, 0.0));
        if s.n_expert > 0 {
            let (e, f) = (s.n_expert, s.expert_ffn);
            ts.push((p("ffn_gate_inp.weight"), vec![e, d], true, 3.0 * sc(d)));
            ts.push((p("ffn_gate_exps.weight"), vec![e, f, d], false, sc(d)));
            ts.push((p("ffn_up_exps.weight"), vec![e, f, d], false, sc(d)));
            ts.push((p("ffn_down_exps.weight"), vec![e, d, f], false, sc(f)));
        } else {
            ts.push((p("ffn_gate.weight"), vec![s.ffn, d], false, sc(d)));
            ts.push((p("ffn_up.weight"), vec![s.ffn, d], false, sc(d)));
            ts.push((p("ffn_down.weight"), vec![d, s.ffn], false, sc(s.ffn)));
        }
    }
    if let Some((_, _, passes)) = s.loop_ {
        for p in 1..passes {
            // gates part open, different per channel: the loop moves the logits
            ts.push((format!("loop.gate.{p}"), vec![d], true, -0.5));
        }
    }
    let bytes = |shape: &[usize], f32_: bool| -> (u32, u64) {
        let n: usize = shape.iter().product();
        let ty = if f32_ || s.dequantized { "F32" } else { s.ty };
        let (be, bb) = strata_gguf::block_geometry(ty).unwrap();
        (ggml_type(ty), (n as u64 / be) * bb)
    };
    let specs: Vec<TensorSpec> = ts.iter().map(|(n, shape, f, _)| {
        let (t, nb) = bytes(shape, *f);
        TensorSpec { name: n.clone(), shape: shape.iter().map(|&v| v as u64).collect(), ggml_type: t, nbytes: nb }
    }).collect();
    let kvr: Vec<(&str, Kv)> = kv.iter().map(|(k, v)| (k.as_str(), v.clone())).collect();
    let mut w = Writer::create_typed(path, &kvr, specs).unwrap();
    for (_, shape, f32_, scale) in &ts {
        let n: usize = shape.iter().product();
        let vals: Vec<f32> = (0..n).map(|_| {
            let u = rng.unit();
            if *scale == 0.0 { 1.0 + 0.1 * u } else if *scale < 0.0 { 0.5 + 0.4 * u } else { u * scale * 1.7 }
        }).collect();
        if *f32_ || s.ty == "F32" {
            let b: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
            w.tensor_data(&b).unwrap();
        } else {
            let mut out = Vec::new();
            strata_pack::quantize::quantize(s.ty, &vals, &mut out).unwrap();
            if s.dequantized {
                let mut back = vec![0f32; n];
                strata_pack::quants::dequantize_into(s.ty, &out, &mut back).unwrap();
                out = back.iter().flat_map(|v| v.to_le_bytes()).collect();
            }
            w.tensor_data(&out).unwrap();
        }
        w.end_tensor().unwrap();
    }
    w.finish().unwrap();
}

/// A synthetic Inkling 2 (`inkling2` / `inkling2_loop`, model/inkling.rs): the same tensor names, shapes and metadata
/// keys as the real file and the Workshop's TinyInkling, random weights at sizes where the activations stay O(1).
#[derive(Clone)]
pub struct InkSpec {
    pub n_embd: usize,
    pub n_head: usize,
    pub n_kv: usize,
    pub head_dim: usize,
    pub n_layer: usize,
    /// the padded vocabulary (the matrices' rows) and the real one (the logits kept)
    pub vocab: usize,
    pub vocab_unpadded: usize,
    pub n_expert: usize,
    pub top_k: usize,
    pub ffn: usize,
    /// the looped core: layers start..end, passes; None writes `inkling2`
    pub loop_: Option<(usize, usize, usize)>,
    /// per layer: a windowed layer (true) or full attention
    pub swa: Vec<bool>,
    pub n_swa: usize,
    pub extent: usize,
    pub n_rel: usize,
    pub n_conv: usize,
    pub rank: usize,
    pub block: usize,
    /// the encodings: attention and dense matrices, the vocabulary, the experts' gate/up and down, the controllers
    pub ty: &'static str,
    pub ty_vocab: &'static str,
    pub ty_gate_up: &'static str,
    pub ty_down: &'static str,
    pub ty_ctrl: &'static str,
    pub seed: u64,
    /// the matrices quantized and written back as F32 (Spec's `dequantized`): the reference for the device's kernels
    pub dequantized: bool,
}

impl InkSpec {
    /// 6 layers, the loop over 2..4 three times, blocks of 2, a 16-token window and a 24-position bias (shorter than
    /// the tests' prompts, so the ring and the extent are crossed), everything F32.
    pub fn tiny() -> InkSpec {
        InkSpec { n_embd: 128, n_head: 4, n_kv: 2, head_dim: 32, n_layer: 6, vocab: 256, vocab_unpadded: 250, n_expert: 8, top_k: 2, ffn: 64,
                  loop_: Some((2, 4, 3)), swa: vec![true, true, true, false, true, false], n_swa: 16, extent: 24, n_rel: 16, n_conv: 4,
                  rank: 32, block: 2, ty: "F32", ty_vocab: "F32", ty_gate_up: "F32", ty_down: "F32", ty_ctrl: "F32", seed: 0x1_4C1D,
                  dequantized: false }
    }
}

/// Writes the Inkling model to `path`.
pub fn write_inkling(s: &InkSpec, path: &Path) {
    let mut rng = Rng(s.seed | 1);
    let a = if s.loop_.is_some() { "inkling2_loop" } else { "inkling2" };
    let nl = s.n_layer;
    let mut kv: Vec<(String, Kv)> = vec![
        ("general.architecture".into(), Kv::Str(a.into())),
        ("general.name".into(), Kv::Str(format!("synthetic-{a}"))),
        (format!("{a}.block_count"), Kv::U32(nl as u32)),
        (format!("{a}.context_length"), Kv::U32(1 << 20)),
        (format!("{a}.embedding_length"), Kv::U32(s.n_embd as u32)),
        (format!("{a}.feed_forward_length"), Kv::U32(s.ffn as u32)),
        (format!("{a}.attention.head_count"), Kv::Ints(vec![s.n_head as i32; nl])),
        (format!("{a}.attention.head_count_kv"), Kv::Ints(vec![s.n_kv as i32; nl])),
        (format!("{a}.attention.layer_norm_rms_epsilon"), Kv::F32(1e-6)),
        (format!("{a}.expert_count"), Kv::U32(s.n_expert as u32)),
        (format!("{a}.expert_used_count"), Kv::U32(s.top_k as u32)),
        (format!("{a}.attention.key_length"), Kv::U32(s.head_dim as u32)),
        (format!("{a}.attention.value_length"), Kv::U32(s.head_dim as u32)),
        (format!("{a}.vocab_size"), Kv::U32(s.vocab as u32)),
        (format!("{a}.vocab_size_unpadded"), Kv::U32(s.vocab_unpadded as u32)),
        (format!("{a}.expert_feed_forward_length"), Kv::U32(s.ffn as u32)),
        (format!("{a}.rope.dimension_count"), Kv::U32(0)),
        (format!("{a}.attention.sliding_window"), Kv::U32(s.n_swa as u32)),
        (format!("{a}.attention.sliding_window_pattern"), Kv::Bools(s.swa.clone())),
        (format!("{a}.shortconv.l_cache"), Kv::U32(s.n_conv as u32)),
        (format!("{a}.expert_weights_scale"), Kv::F32(8.0)),
        (format!("{a}.activation.situ_beta"), Kv::F32(4.0)),
        (format!("{a}.activation.situ_linear_beta"), Kv::F32(25.0)),
        (format!("{a}.attn_res.block_size"), Kv::U32(s.block as u32)),
        (format!("{a}.final_logit_softcapping"), Kv::F32(30.0)),
        (format!("{a}.attention.relative_length"), Kv::U32(s.n_rel as u32)),
        (format!("{a}.attention.relative_extent"), Kv::U32(s.extent as u32)),
        (format!("{a}.attention.qk_clip"), Kv::F32((s.head_dim as f32).sqrt() * 4.0)),
        (format!("{a}.attention.relative_clip"), Kv::F32(8.0)),
        // a small floor, so the log scale of the full layers moves within the tests' prompts
        (format!("{a}.attention.log_scale_floor"), Kv::U32(8)),
        (format!("{a}.attention.log_scale_alpha"), Kv::F32(0.1)),
        ("tokenizer.ggml.eos_token_id".into(), Kv::U32(1)),
    ];
    if let Some((start, end, passes)) = s.loop_ {
        kv.push((format!("{a}.loop.start"), Kv::U32(start as u32)));
        kv.push((format!("{a}.loop.end"), Kv::U32(end as u32)));
        kv.push((format!("{a}.loop.count"), Kv::U32(passes as u32)));
        kv.push((format!("{a}.loop.gate_rank"), Kv::U32(s.rank as u32)));
    }
    // (name, numpy shape, type, fill): fill > 0 a uniform of that scale, 0 a norm (1 + noise), < 0 a uniform of -fill
    // around 1 (clip scales)
    let (d, q, kvd) = (s.n_embd, s.n_head * s.head_dim, s.n_kv * s.head_dim);
    let sc = |cols: usize| 1.0 / (cols as f32).sqrt();
    let mut ts: Vec<(String, Vec<usize>, &str, f32)> = vec![
        ("token_embd.weight".into(), vec![s.vocab, d], s.ty_vocab, 1.0),
        ("token_embd_norm.weight".into(), vec![d], "F32", 0.0),
        ("output.weight".into(), vec![s.vocab, d], s.ty_vocab, sc(d)),
        ("output_norm.weight".into(), vec![d], "F32", 0.0),
        ("output_res_query.weight".into(), vec![d], "F32", 0.3),
    ];
    for l in 0..nl {
        let p = |n: &str| format!("blk.{l}.{n}");
        let swa = s.swa[l];
        for (n, rows) in [("attn_q", q), ("attn_k", kvd), ("attn_v", kvd), ("attn_gate", q), ("attn_sink_proj", s.n_head), ("attn_rel", s.n_rel * s.n_head)] {
            ts.push((p(&format!("{n}.weight")), vec![rows, d], s.ty, sc(d)));
        }
        ts.push((p("attn_output.weight"), vec![d, q], s.ty, sc(q)));
        ts.push((p("attn_q_clip.weight"), vec![s.n_head, s.head_dim], "F32", -0.3));
        ts.push((p("attn_rel_clip.weight"), vec![s.n_head, s.n_rel], "F32", -0.3));
        ts.push((p("attn_rel_bias.weight"), vec![if swa { s.n_swa } else { s.extent }, s.n_rel], "F32", 0.5));
        for (n, ch) in [("attn_k_conv", kvd), ("attn_v_conv", kvd), ("attn_post_conv", d), ("ffn_post_conv", d)] {
            ts.push((p(&format!("{n}.weight")), vec![ch, s.n_conv], "F32", 0.5));
        }
        if swa {
            ts.push((p("attn_norm.weight"), vec![d], "F32", 0.0));
        } else {
            for n in ["attn_q_inp_norm", "attn_k_inp_norm", "attn_v_inp_norm"] {
                ts.push((p(&format!("{n}.weight")), vec![d], "F32", 0.0));
            }
            for n in ["attn_k_conv_proj", "attn_v_conv_proj"] {
                ts.push((p(&format!("{n}.weight")), vec![s.n_conv * s.n_kv, d], s.ty, sc(d)));
            }
        }
        if l > 0 {
            ts.push((p("attn_res_query.weight"), vec![if swa { 1 } else { 3 }, d], "F32", 0.3));
        }
        ts.push((p("ffn_res_query.weight"), vec![d], "F32", 0.3));
        ts.push((p("ffn_norm.weight"), vec![d], "F32", 0.0));
        ts.push((p("ffn_gate_inp.weight"), vec![s.n_expert, d], "F32", 3.0 * sc(d)));
        ts.push((p("exp_probs_b.bias"), vec![s.n_expert], "F32", 0.05));
        ts.push((p("ffn_gate_up_exps.weight"), vec![s.n_expert, 2 * s.ffn, d], s.ty_gate_up, sc(d)));
        ts.push((p("ffn_down_exps.weight"), vec![s.n_expert, d, s.ffn], s.ty_down, sc(s.ffn)));
        if s.loop_.is_some_and(|(a, b, _)| (a..b).contains(&l)) {
            ts.push((p("loop_msg_query.weight"), vec![4, d], "F32", 0.3));
            for c in ["q", "k", "v", "ffn"] {
                ts.push((p(&format!("loop_{c}_ctrl_down.weight")), vec![s.rank, 2 * d], s.ty_ctrl, sc(2 * d)));
                ts.push((p(&format!("loop_{c}_ctrl_gain.weight")), vec![d, s.rank], s.ty_ctrl, sc(s.rank)));
                ts.push((p(&format!("loop_{c}_ctrl_forget.weight")), vec![d, s.rank], s.ty_ctrl, sc(s.rank)));
            }
        }
    }
    if s.dequantized {
        ts.iter_mut().for_each(|t| if t.2 != "F32" { t.2 = Box::leak(format!("F32:{}", t.2).into_boxed_str()) });
    }
    let specs: Vec<TensorSpec> = ts.iter().map(|(n, shape, ty, _)| {
        let count: usize = shape.iter().product();
        let ty = if ty.starts_with("F32") { "F32" } else { ty };
        let (be, bb) = strata_gguf::block_geometry(ty).unwrap();
        TensorSpec { name: n.clone(), shape: shape.iter().map(|&v| v as u64).collect(), ggml_type: ggml_type(ty), nbytes: (count as u64 / be) * bb }
    }).collect();
    let kvr: Vec<(&str, Kv)> = kv.iter().map(|(k, v)| (k.as_str(), v.clone())).collect();
    let mut w = Writer::create_typed(path, &kvr, specs).unwrap();
    for (_, shape, ty, fill) in &ts {
        let n: usize = shape.iter().product();
        let vals: Vec<f32> = (0..n).map(|_| {
            let u = rng.unit();
            if *fill == 0.0 { 1.0 + 0.1 * u } else if *fill < 0.0 { 1.0 + fill * u } else { u * fill * 1.7 }
        }).collect();
        if *ty == "F32" {
            w.tensor_data(&vals.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
        } else if let Some(q) = ty.strip_prefix("F32:") {
            let mut out = Vec::new();
            strata_pack::quantize::quantize(q, &vals, &mut out).unwrap();
            let mut back = vec![0f32; n];
            strata_pack::quants::dequantize_into(q, &out, &mut back).unwrap();
            w.tensor_data(&back.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
        } else {
            let mut out = Vec::new();
            strata_pack::quantize::quantize(ty, &vals, &mut out).unwrap();
            w.tensor_data(&out).unwrap();
        }
        w.end_tensor().unwrap();
    }
    w.finish().unwrap();
}

/// An Inkling fixture as a file in the test process's temp dir.
pub fn tmp_inkling(name: &str, s: &InkSpec) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("am2-testgguf-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let p = dir.join(format!("{name}.gguf"));
    if !p.exists() {
        write_inkling(s, &p);
    }
    p
}

/// The fixtures as files in a temp dir of their own (one per test process and name).
pub fn tmp(name: &str, s: &Spec) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("am2-testgguf-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let p = dir.join(format!("{name}.gguf"));
    if !p.exists() {
        write(s, &p);
    }
    p
}

/// The speed runs' models (docs/MULA-PLAN.md M8): a small dense Qwen3, the same looped, and a small Qwen3-MoE, in Q4_K
/// and Q8_0 (about 0.2-0.6 GB each).
#[test]
#[ignore]
fn write_speed_fixtures() {
    let dir = std::path::PathBuf::from(std::env::var("AM2_WRITE_FIXTURES").expect("AM2_WRITE_FIXTURES=<dir>"));
    std::fs::create_dir_all(&dir).unwrap();
    let dense = Spec { arch: "qwen3", n_embd: 1024, n_head: 16, n_kv: 8, head_dim: 128, n_layer: 16, vocab: 32768, ffn: 3072,
                       n_expert: 0, top_k: 0, expert_ffn: 0, ty: "Q4_K", loop_: None, seed: 0xD1, dequantized: false, window: 0 };
    let looped = Spec { loop_: Some((4, 12, 2)), seed: 0xD2, ..dense.clone() };
    let moe = Spec { arch: "qwen3moe", n_expert: 32, top_k: 4, expert_ffn: 512, seed: 0xD3, ..dense.clone() };
    let moe_q8 = Spec { ty: "Q8_0", seed: 0xD4, ..moe.clone() };
    let tiny_moe = Spec::tiny_qwen3moe();
    let tiny_loop = Spec { loop_: Some((1, 3, 3)), seed: 0x5EED_0003, ..Spec::tiny_qwen3() };
    for (n, s) in [("tiny-qwen3moe", &tiny_moe), ("tiny-qwen3-loop", &tiny_loop),
                   ("small-qwen3-q4k", &dense), ("small-qwen3-loop-q4k", &looped), ("small-qwen3moe-q4k", &moe), ("small-qwen3moe-q8_0", &moe_q8)] {
        let p = dir.join(format!("{n}.gguf"));
        write(s, &p);
        eprintln!("{} {:.1} MB", p.display(), std::fs::metadata(&p).unwrap().len() as f64 / 1e6);
    }
}

/// The kernel speed runs' models (docs/MULA-PLAN.md M8, the kernels): Qwen3-8B's shapes (4096 wide, 32 heads of 128, 8
/// KV heads, 12288 in the feed-forward, a 151936-token vocabulary) with 8 layers, in Q8_0 (2.3 GB on the card) and Q4_K
/// (1.2 GB), and the Q8_0 one looped (blocks 2..6, 2 passes: 12 layer passes per token), small enough for a card that
/// already holds another model, large enough that a token's weights are read at the card's bandwidth.
///   AM2_WRITE_FIXTURES=<dir> cargo test --release write_kernel_bench_models -- --ignored
#[test]
#[ignore]
fn write_kernel_bench_models() {
    let dir = std::path::PathBuf::from(std::env::var("AM2_WRITE_FIXTURES").expect("AM2_WRITE_FIXTURES=<dir>"));
    std::fs::create_dir_all(&dir).unwrap();
    let q8 = Spec { arch: "qwen3", n_embd: 4096, n_head: 32, n_kv: 8, head_dim: 128, n_layer: 8, vocab: 151936, ffn: 12288,
                    n_expert: 0, top_k: 0, expert_ffn: 0, ty: "Q8_0", loop_: None, seed: 0xB1, dequantized: false, window: 0 };
    let q4 = Spec { ty: "Q4_K", seed: 0xB2, ..q8.clone() };
    let looped = Spec { loop_: Some((2, 6, 2)), seed: 0xB3, ..q8.clone() };
    for (n, s) in [("bench-qwen3-4096-q8_0", &q8), ("bench-qwen3-4096-q4_k", &q4), ("bench-qwen3-4096-loop-q8_0", &looped)] {
        let p = dir.join(format!("{n}.gguf"));
        if !p.exists() {
            write(s, &p);
        }
        eprintln!("{} {:.2} GB", p.display(), std::fs::metadata(&p).unwrap().len() as f64 / 1e9);
    }
}

/// The Inkling speed run's model (docs/INKLING-PLAN.md I8): the real Inkling 2 Mini's shapes and encodings (2304 wide,
/// 24 heads of 128, 8 KV heads, 24 layers with the loop over 6..18 three times, blocks of 6, a 1024-token window and
/// bias, 8 experts of 1152 per token, a 201,024-row vocabulary; Q8_0 attention, vocabulary and controllers, Q4_K
/// gate/up and Q5_0 down) with 32 experts per layer instead of 128: a token reads what a real token reads, and the
/// whole model fits a card beside another (about 4.9 GB).
///   AM2_WRITE_FIXTURES=<dir> cargo test --release write_inkling_speed_model -- --ignored
#[test]
#[ignore]
fn write_inkling_speed_model() {
    let dir = std::path::PathBuf::from(std::env::var("AM2_WRITE_FIXTURES").expect("AM2_WRITE_FIXTURES=<dir>"));
    std::fs::create_dir_all(&dir).unwrap();
    let p = dir.join("bench-inkling2-2304x32e.gguf");
    if !p.exists() {
        write_inkling(&inkling_speed_spec(), &p);
    }
    eprintln!("{} {:.2} GB", p.display(), std::fs::metadata(&p).unwrap().len() as f64 / 1e9);
}

pub fn inkling_speed_spec() -> InkSpec {
    let swa: Vec<bool> = (0..24).map(|l| l % 6 != 5).collect();
    InkSpec { n_embd: 2304, n_head: 24, n_kv: 8, head_dim: 128, n_layer: 24, vocab: 201024, vocab_unpadded: 200058, n_expert: 32, top_k: 8,
              ffn: 1152, loop_: Some((6, 18, 3)), swa, n_swa: 1024, extent: 1024, n_rel: 16, n_conv: 4, rank: 64, block: 6, ty: "Q8_0",
              ty_vocab: "Q8_0", ty_gate_up: "Q4_K", ty_down: "Q5_0", ty_ctrl: "Q8_0", seed: 0x1_4C20,
              dequantized: false }
}
