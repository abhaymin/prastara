//! 1M-token context, the Workshop's part (docs/MULA-PLAN.md L1-L3): YaRN positions, sliding-window layers.

use std::path::PathBuf;

use candle_core::Device;
use strata_workshop::model::{BaseDtype, Ctx, LlamaModel, Yarn};

fn fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny-qwen3-hf")
}

/// Qwen3's head (128) and base (1e6) stretched 32x over 32,768 positions: the values transformers' formula gives
/// (computed independently in float64 from `_compute_yarn_parameters`).
#[test]
fn yarn_frequencies_match_the_reference() {
    let y = Yarn::new(32.0, 32768);
    let f = y.inv_freq(128, 1e6);
    for (i, want) in [(0, 1.0), (10, 0.1154781984689458), (23, 0.006978305848598663), (28, 0.0016957065652617347), (40, 5.5571231563716334e-06),
                      (63, 3.8779305023491235e-08)] {
        assert!((f[i] - want).abs() <= want * 1e-12, "frequency {i}: {} against {want}", f[i]);
    }
    assert!((y.attention_factor - 1.3465735902799727).abs() < 1e-12);
    // factor 1 is the plain rotation
    let plain = Yarn::new(1.0, 32768).inv_freq(128, 1e6);
    for (i, p) in plain.iter().enumerate() {
        let want = 1.0 / 1e6f64.powf(2.0 * i as f64 / 128.0);
        assert!((p - want).abs() <= want * 1e-12);
    }
}

fn with_config(tag: &str, edit: impl Fn(&mut serde_json::Value)) -> PathBuf {
    let d = std::env::temp_dir().join(format!("longctx-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    for e in std::fs::read_dir(fixture()).unwrap().flatten() {
        std::fs::copy(e.path(), d.join(e.file_name())).unwrap();
    }
    let mut c: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(d.join("config.json")).unwrap()).unwrap();
    edit(&mut c);
    std::fs::write(d.join("config.json"), c.to_string()).unwrap();
    d
}

const TOKENS: [u32; 8] = [1, 7, 42, 3, 59, 5, 11, 2];

fn logits(dir: &std::path::Path) -> Vec<Vec<f32>> {
    LlamaModel::load(dir, BaseDtype::F32, &Device::Cpu).unwrap().forward(&TOKENS, &Ctx::eval()).unwrap().to_vec2().unwrap()
}

#[test]
fn a_window_wider_than_the_text_changes_nothing_and_a_narrow_one_only_later_positions() {
    let layers = |c: &serde_json::Value| c["num_hidden_layers"].as_u64().unwrap() as usize;
    let base = logits(&fixture());
    let wide = with_config("wide", |c| { let n = layers(c); c["layer_types"] = serde_json::json!(vec!["sliding_attention"; n]); c["sliding_window"] = serde_json::json!(64); });
    assert_eq!(logits(&wide), base);
    let narrow = with_config("narrow", |c| { let n = layers(c); c["layer_types"] = serde_json::json!(vec!["sliding_attention"; n]); c["sliding_window"] = serde_json::json!(3); });
    let got = logits(&narrow);
    assert_eq!(got[..3], base[..3], "the first 3 positions see the same keys");
    assert_ne!(got[3..], base[3..], "later positions lose the keys beyond the window");
    // a mixed model: only the sliding layers lose them
    let mixed = with_config("mixed", |c| { let n = layers(c); let t: Vec<&str> = (0..n).map(|i| if i % 2 == 0 { "full_attention" } else { "sliding_attention" }).collect();
                                           c["layer_types"] = serde_json::json!(t); c["sliding_window"] = serde_json::json!(3); });
    let m = logits(&mixed);
    assert_eq!(m[..3], base[..3]);
    assert_ne!(m[3..], got[3..]);
    for d in [wide, narrow, mixed] { let _ = std::fs::remove_dir_all(d); }
}

#[test]
fn a_yarn_config_loads_and_keeps_short_positions_close() {
    let y = with_config("yarn", |c| { c["rope_scaling"] = serde_json::json!({"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 64}); });
    let m = LlamaModel::load(&y, BaseDtype::F32, &Device::Cpu).unwrap();
    assert_eq!(m.cfg.yarn.unwrap().factor, 4.0);
    let got = logits(&y);
    assert!(got.iter().flatten().all(|v| v.is_finite()));
    // refused: a type the forward pass does not do
    let bad = with_config("bad", |c| { c["rope_scaling"] = serde_json::json!({"rope_type": "longrope", "factor": 4.0}); });
    assert!(LlamaModel::load(&bad, BaseDtype::F32, &Device::Cpu).is_err());
    for d in [y, bad] { let _ = std::fs::remove_dir_all(d); }
}

#[test]
fn the_context_command_writes_a_long_context_copy() {
    let out = std::env::temp_dir().join(format!("longctx-conv-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&out);
    strata_workshop::model::make_long_context(&fixture(), &out, 1 << 20, Some(32768), 4096, 4).unwrap();
    let m = LlamaModel::load(&out, BaseDtype::F32, &Device::Cpu).unwrap();
    assert_eq!(m.cfg.yarn.unwrap().factor, 32.0);
    assert_eq!(m.cfg.max_position_embeddings, 1 << 20);
    assert_eq!(m.cfg.sliding_window, Some(4096));
    let full: Vec<usize> = m.cfg.layer_sliding.iter().enumerate().filter(|(_, s)| !**s).map(|(i, _)| i).collect();
    let n = m.cfg.num_hidden_layers;
    assert!(!full.is_empty() && full.iter().all(|i| (i + 1) % 4 == 0 || i + 1 == n), "{full:?}");
    assert!(logits(&out).iter().flatten().all(|v| v.is_finite()));
    // twice is refused, and so is a length that is not longer
    assert!(strata_workshop::model::make_long_context(&out, &out.with_extension("2"), 1 << 21, None, 4096, 4).is_err());
    assert!(strata_workshop::model::make_long_context(&fixture(), &out.with_extension("3"), 1000, Some(32768), 4096, 4).is_err());
    let _ = std::fs::remove_dir_all(&out);
}

/// AM2_MULA_DIRS=<folder>[,<folder>...]: each model's top tokens after "The capital of France is" (Qwen3 ids;
/// 12095 is " Paris"), on the CPU in BF16. A check that a conversion still answers, not a quality measure.
#[test]
#[ignore]
fn real_models_answer_a_short_prompt() {
    for dir in std::env::var("AM2_MULA_DIRS").unwrap().split(',') {
        let m = LlamaModel::load(std::path::Path::new(dir), BaseDtype::BF16, &Device::Cpu).unwrap();
        let tokens = [785u32, 6722, 315, 9625, 374];
        let last: Vec<f32> = m.forward(&tokens, &Ctx::eval()).unwrap().get(tokens.len() - 1).unwrap().to_vec1().unwrap();
        let max = last.iter().cloned().fold(f32::MIN, f32::max);
        let z: f32 = last.iter().map(|v| (v - max).exp()).sum();
        let mut idx: Vec<usize> = (0..last.len()).collect();
        idx.sort_by(|a, b| last[*b].total_cmp(&last[*a]));
        let top: Vec<String> = idx[..5].iter().map(|i| format!("{i}:{:.3}", (last[*i] - max).exp() / z)).collect();
        println!("{dir}: {}", top.join(" "));
    }
}

/// The fixture with a tokenizer (the tiny Llama's: 32 tokens, the rest of the 64 padding), ready to convert.
fn with_tokenizer(tag: &str, edit: impl Fn(&mut serde_json::Value)) -> PathBuf {
    let d = with_config(tag, edit);
    let tok = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny-llama-hf");
    for f in ["tokenizer.json", "tokenizer_config.json"] {
        std::fs::copy(tok.join(f), d.join(f)).unwrap();
    }
    d
}

/// The positions the engine's long-context fixture reads: longer than its window of 5.
const LONG: [u32; 24] = [1, 7, 42, 3, 59, 5, 11, 2, 30, 17, 8, 61, 4, 23, 9, 50, 12, 6, 33, 19, 44, 2, 27, 15];

fn long_logits(dir: &std::path::Path) -> Vec<Vec<f32>> {
    LlamaModel::load(dir, BaseDtype::F32, &Device::Cpu).unwrap().forward(&LONG, &Ctx::eval()).unwrap().to_vec2().unwrap()
}

/// `convert --yarn` and a `context` copy to GGUF: the keys the engine reads (crates/strata-hybrid model.rs
/// Rope::from_gguf, the sliding-window pattern).
#[test]
fn long_context_models_convert_with_their_keys() {
    use strata_workshop::convert::{convert_with, AddYarn, OutType};
    let base = with_tokenizer("conv-base", |_| {});
    let g1 = base.join("yarn.gguf");
    convert_with(&base, &g1, OutType::F32, Some("gpt-2"), Some(AddYarn { factor: 4.0, original: 32 })).unwrap();
    let g = strata_gguf::Gguf::open(&g1).unwrap();
    assert_eq!(g.str("qwen3.rope.scaling.type"), Some("yarn"));
    assert_eq!(g.u64("qwen3.rope.scaling.original_context_length"), Some(32));
    assert_eq!(g.u64("qwen3.context_length"), Some(128));
    assert_eq!(g.metadata["qwen3.rope.scaling.factor"].as_f64(), Some(4.0));
    assert!(!g.metadata.contains_key("qwen3.rope.scaling.attn_factor"), "YaRN's own attention factor is the engine's default");
    assert!(!g.metadata.contains_key("qwen3.attention.sliding_window"));
    // a context copy: its YaRN from the config, its sliding layers
    let long = base.with_extension("long");
    let _ = std::fs::remove_dir_all(&long);
    strata_workshop::model::make_long_context(&base, &long, 512, Some(32), 5, 3).unwrap();
    let g2 = base.join("long.gguf");
    convert_with(&long, &g2, OutType::F32, Some("gpt-2"), None).unwrap();
    let g = strata_gguf::Gguf::open(&g2).unwrap();
    assert_eq!(g.metadata["qwen3.rope.scaling.factor"].as_f64(), Some(16.0));
    assert_eq!(g.u64("qwen3.context_length"), Some(512));
    assert_eq!(g.u64("qwen3.attention.sliding_window"), Some(5));
    let pattern: Vec<bool> = g.metadata["qwen3.attention.sliding_window_pattern"].as_array().unwrap().iter().map(|v| v.as_bool().unwrap()).collect();
    assert_eq!(pattern, vec![true, true, false]);
    // --yarn on a model that has YaRN already is refused
    assert!(convert_with(&long, &base.join("twice.gguf"), OutType::F32, Some("gpt-2"), Some(AddYarn { factor: 4.0, original: 32 })).is_err());
    for d in [base, long] { let _ = std::fs::remove_dir_all(d); }
}

/// The engine's long-context fixture (crates/strata-hybrid model/loop_tests.rs `long_context`): the tiny Qwen3 with
/// YaRN added at conversion (4 over 32), and a `context` copy (YaRN 16 over 32, layers 0 and 1 sliding over 5 tokens),
/// as F32 GGUFs, and the Workshop's logits for both and the base over 24 tokens. --ignored writes it again.
#[test]
#[ignore]
fn write_the_engines_long_context_fixture() {
    use strata_workshop::convert::{convert, convert_with, AddYarn, OutType};
    let fx = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny-qwen3-long");
    std::fs::create_dir_all(&fx).unwrap();
    let base = with_tokenizer("fx-base", |_| {});
    convert_with(&base, &fx.join("yarn.f32.gguf"), OutType::F32, Some("gpt-2"), Some(AddYarn { factor: 4.0, original: 32 })).unwrap();
    // the Workshop's view of the same YaRN: the config's rope_scaling
    let yarn = with_tokenizer("fx-yarn", |c| { c["rope_scaling"] = serde_json::json!({"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32}); });
    let long = base.with_extension("long");
    let _ = std::fs::remove_dir_all(&long);
    strata_workshop::model::make_long_context(&base, &long, 512, Some(32), 5, 3).unwrap();
    convert(&long, &fx.join("swa.f32.gguf"), OutType::F32, Some("gpt-2")).unwrap();
    let (b, y, s) = (long_logits(&base), long_logits(&yarn), long_logits(&long));
    let d = |a: &[Vec<f32>], c: &[Vec<f32>]| a.iter().flatten().zip(c.iter().flatten()).map(|(x, z)| (x - z).abs()).fold(0f32, f32::max);
    println!("yarn moves the logits by {:.3}, the context copy by {:.3}", d(&y, &b), d(&s, &b));
    let json = serde_json::json!({"tokens": LONG, "base": b, "yarn": y, "swa": s, "window": 5, "sliding": [true, true, false],
                                  "note": "written by crates/strata-workshop/tests/long_context.rs write_the_engines_long_context_fixture"});
    std::fs::write(fx.join("logits.json"), serde_json::to_string(&json).unwrap()).unwrap();
    for d in [base, yarn, long] { let _ = std::fs::remove_dir_all(d); }
}
