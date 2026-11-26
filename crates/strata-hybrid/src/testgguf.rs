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
    match ty { "F32" => 0, "Q8_0" => 8, "Q4_K" => 12, _ => panic!("testgguf: {ty}") }
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
