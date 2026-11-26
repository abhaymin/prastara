//! The loop (Avarta) in the engine against the Workshop's forward pass: the looped tiny Qwen3 with open gates
//! (crates/strata-workshop/tests/fixtures/tiny-qwen3-loop, written by the Workshop's avarta.rs) gives the Workshop's
//! logits token by token and a chunk at a time, and with one pass the base's.
//!
//! The tolerance is the engine's own: it keeps keys and values in F16 as llama.cpp does, so even the base differs
//! from the Workshop's F32 forward pass by about 6e-4 here. The loop moves these logits by up to 1.7.

/// The largest difference allowed from the Workshop's logits (about 3x what F16 keys and values give the base).
const TOL: f32 = 2e-3;

use std::path::PathBuf;
use std::sync::atomic::Ordering;
use super::Model;

fn fixture() -> (Model, serde_json::Value) {
    let d = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../strata-workshop/tests/fixtures/tiny-qwen3-loop");
    let (m, _) = Model::load(&d.join("model.f32.gguf"), None).unwrap();
    let want: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(d.join("logits.json")).unwrap()).unwrap();
    (m, want)
}

fn rows(v: &serde_json::Value) -> Vec<Vec<f32>> {
    v.as_array().unwrap().iter().map(|r| r.as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect()).collect()
}

fn max_diff(a: &[Vec<f32>], b: &[Vec<f32>]) -> f32 {
    a.iter().flatten().zip(b.iter().flatten()).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

fn tokens(want: &serde_json::Value) -> Vec<u32> {
    want["tokens"].as_array().unwrap().iter().map(|t| t.as_u64().unwrap() as u32).collect()
}

#[test]
fn the_engine_runs_the_loop_as_the_workshop_trains_it() {
    let (m, want) = fixture();
    assert_eq!(m.passes(), 3);
    assert!(m.summary().contains("loop=1..3x3/3"), "{}", m.summary());
    let toks = tokens(&want);
    let (looped, base) = (rows(&want["looped"]), rows(&want["base"]));
    assert!(max_diff(&looped, &base) > 0.5, "the fixture's gates are open: the loop changes the logits");

    let mut st = m.new_state();
    let one: Vec<Vec<f32>> = toks.iter().map(|&t| m.forward(&mut st, t)).collect();
    let d = max_diff(&one, &looped);
    assert!(d < TOL, "token by token: {d}");

    let mut st = m.new_state();
    let chunk = m.forward_batch_all(&mut st, &toks);
    let d = max_diff(&chunk, &looped);
    assert!(d < TOL, "a chunk at a time: {d}");

    // a chunk, then one token more: the extra passes' caches carry on
    let mut st = m.new_state();
    let mut mixed = m.forward_batch_all(&mut st, &toks[..4]);
    mixed.extend(toks[4..].iter().map(|&t| m.forward(&mut st, t)));
    let d = max_diff(&mixed, &looped);
    assert!(d < TOL, "a chunk then single tokens: {d}");
}

#[test]
fn one_pass_is_the_base() {
    let (m, want) = fixture();
    m.loop_passes.store(1, Ordering::Relaxed);
    assert_eq!(m.passes(), 1);
    let mut st = m.new_state();
    let got: Vec<Vec<f32>> = tokens(&want).iter().map(|&t| m.forward(&mut st, t)).collect();
    let d = max_diff(&got, &rows(&want["base"]));
    assert!(d < TOL, "--loop-passes 1 runs the base: {d}");
}

/// LoRA upcycling (plan Part C): the Workshop's `export-moe` tiny qwen3moe GGUFs against its own forward pass
/// (crates/strata-workshop/tests/fixtures/tiny-qwen3-moe, written by its upcycle.rs): 4 experts top-2 with trained
/// experts and routers, the same quantized to Q8_0, a looped model upcycled to top-1 (the MoE blocks follow the
/// loop's schedule like the attention ones), and the untrained export, which is the dense base.
mod moe {
    use super::{max_diff, rows, tokens, TOL};
    use crate::model::Model;
    use std::path::PathBuf;

    fn load(name: &str) -> (Model, serde_json::Value) {
        let d = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../strata-workshop/tests/fixtures/tiny-qwen3-moe");
        let (m, _) = Model::load(&d.join(name), None).unwrap();
        (m, serde_json::from_str(&std::fs::read_to_string(d.join("logits.json")).unwrap()).unwrap())
    }

    /// Token by token and a chunk at a time, against `want`.
    fn check(m: &Model, toks: &[u32], want: &[Vec<f32>], tol: f32, what: &str) -> f32 {
        let mut st = m.new_state();
        let one: Vec<Vec<f32>> = toks.iter().map(|&t| m.forward(&mut st, t)).collect();
        let d1 = max_diff(&one, want);
        let mut st = m.new_state();
        let d2 = max_diff(&m.forward_batch_all(&mut st, toks), want);
        println!("{what}: token by token {d1:.2e}, a chunk {d2:.2e}");
        assert!(d1 < tol && d2 < tol, "{what}: {d1} / {d2}");
        d1.max(d2)
    }

    #[test]
    fn the_engine_runs_the_upcycled_moe_as_the_workshop_trains_it() {
        let (m, want) = load("moe.f32.gguf");
        assert_eq!(m.arch, "qwen3moe");
        let (moe, base) = (rows(&want["moe"]), rows(&want["base"]));
        assert!(max_diff(&moe, &base) > 0.1, "the fixture's experts are trained: they change the logits");
        check(&m, &tokens(&want), &moe, TOL, "4 experts top-2");
    }

    #[test]
    fn a_quantized_export_stays_close() {
        let (m, want) = load("moe.q8_0.gguf");
        check(&m, &tokens(&want), &rows(&want["moe"]), 5e-2, "Q8_0");
    }

    #[test]
    fn a_looped_moe_runs_its_experts_in_every_pass() {
        let (m, want) = load("moe-loop.f32.gguf");
        assert_eq!(m.passes(), 3);
        let looped = rows(&want["moe_loop"]);
        assert!(max_diff(&looped, &rows(&want["base"])) > 0.1);
        check(&m, &tokens(&want), &looped, TOL, "looped, top-1");
        // a chunk, then single tokens: each pass's caches carry on
        let toks = tokens(&want);
        let mut st = m.new_state();
        let mut mixed = m.forward_batch_all(&mut st, &toks[..4]);
        mixed.extend(toks[4..].iter().map(|&t| m.forward(&mut st, t)));
        assert!(max_diff(&mixed, &looped) < TOL);
    }

    #[test]
    fn the_untrained_export_is_the_dense_base() {
        let (m, want) = load("init.f32.gguf");
        check(&m, &tokens(&want), &rows(&want["base"]), TOL, "untrained, 4 experts top-1");
    }
}

/// The compact caches (kvcache.rs, `--kv`) on the loop and MoE fixtures: f16 (the default, the engine's values since
/// the start) and q8 against an f32 cache and against the Workshop's own f32 logits, token by token and a chunk at a
/// time.
mod compact_cache {
    use super::{max_diff, rows, tokens};
    use crate::kvcache::KvType;
    use crate::model::Model;
    use std::path::PathBuf;

    fn logits(m: &Model, toks: &[u32], chunk: bool) -> Vec<Vec<f32>> {
        let mut st = m.new_state();
        if chunk { m.forward_batch_all(&mut st, toks) } else { toks.iter().map(|&t| m.forward(&mut st, t)).collect() }
    }

    #[test]
    fn f16_and_q8_caches_keep_the_logits() {
        let fx = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../strata-workshop/tests/fixtures");
        for (file, key, json) in [("tiny-qwen3-loop/model.f32.gguf", "looped", "tiny-qwen3-loop/logits.json"),
                                  ("tiny-qwen3-moe/moe.f32.gguf", "moe", "tiny-qwen3-moe/logits.json"),
                                  ("tiny-qwen3-moe/moe-loop.f32.gguf", "moe_loop", "tiny-qwen3-moe/logits.json")] {
            let want: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(fx.join(json)).unwrap()).unwrap();
            let (toks, ws) = (tokens(&want), rows(&want[key]));
            let (mut m, _) = Model::load(&fx.join(file), None).unwrap();
            let mut got = vec![];
            for kv in [KvType::F32, KvType::F16, KvType::Q8] {
                m.kv_type = kv;
                let (one, chunk) = (logits(&m, &toks, false), logits(&m, &toks, true));
                assert_eq!(one, chunk, "{file} {kv:?}: a chunk is token by token, bit for bit");
                got.push(one);
            }
            let d = |i: usize| (max_diff(&got[i], &got[0]), max_diff(&got[i], &ws));
            let (f32w, f16, q8) = (d(0).1, d(1), d(2));
            println!("{file}: vs the f32 cache: f16 {:.2e}, q8 {:.2e}; vs the Workshop: f32 {f32w:.2e}, f16 {:.2e}, q8 {:.2e}", f16.0, q8.0, f16.1, q8.1);
            assert!(f32w < 2e-3 && f16.0 < 2e-3 && f16.1 < 2e-3, "{file}");
            assert!(q8.0 < 3e-2 && q8.1 < 3e-2, "{file}: q8 {q8:?}");
        }
    }
}

/// Long context (docs/MULA-PLAN.md L5) against the Workshop's forward pass (crates/strata-workshop/tests/fixtures/
/// tiny-qwen3-long, written by its long_context.rs): YaRN added at conversion, and the `context` copy with YaRN and
/// two of three layers sliding over 5 tokens, over 24 tokens; token by token, a chunk at a time, and a chunk then
/// tokens, in every cache encoding.
pub(crate) mod long_context {
    use super::{max_diff, rows, tokens};
    use crate::kvcache::KvType;
    use crate::model::{Block, Model};
    use std::path::PathBuf;

    pub fn dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../strata-workshop/tests/fixtures/tiny-qwen3-long")
    }

    pub fn want() -> serde_json::Value {
        serde_json::from_str(&std::fs::read_to_string(dir().join("logits.json")).unwrap()).unwrap()
    }

    /// (f32, f16, q8) worst differences from `want` over the three ways of reading.
    fn run(m: &mut Model, toks: &[u32], want: &[Vec<f32>]) -> Vec<f32> {
        [KvType::F32, KvType::F16, KvType::Q8].iter().map(|&kv| {
            m.kv_type = kv;
            let mut st = m.new_state();
            let one: Vec<Vec<f32>> = toks.iter().map(|&t| m.forward(&mut st, t)).collect();
            let mut st = m.new_state();
            let chunk = m.forward_batch_all(&mut st, toks);
            let mut st = m.new_state();
            let mut mixed = m.forward_batch_all(&mut st, &toks[..9]);
            mixed.extend(toks[9..].iter().map(|&t| m.forward(&mut st, t)));
            assert_eq!(one, chunk, "{kv:?}: a chunk is token by token");
            max_diff(&one, want).max(max_diff(&mixed, want))
        }).collect()
    }

    #[test]
    fn yarn_and_sliding_windows_run_as_the_workshop_trains_them() {
        let w = want();
        let toks = tokens(&w);
        let (base, yarn, swa) = (rows(&w["base"]), rows(&w["yarn"]), rows(&w["swa"]));
        assert!(max_diff(&yarn, &base) > 1e-2 && max_diff(&swa, &yarn) > 1e-2, "the fixture's YaRN and windows change the logits");
        let (mut y, _) = Model::load(&dir().join("yarn.f32.gguf"), None).unwrap();
        let dy = run(&mut y, &toks, &yarn);
        let (mut s, _) = Model::load(&dir().join("swa.f32.gguf"), None).unwrap();
        let windows: Vec<Option<usize>> = s.layers.iter().map(|l| match &l.parts[0].1 { Block::Attn(a) => a.window, _ => None }).collect();
        assert_eq!(windows, vec![Some(5), Some(5), None]);
        let ds = run(&mut s, &toks, &swa);
        println!("YaRN: max |engine - Workshop| f32 {:.2e} f16 {:.2e} q8 {:.2e}; YaRN + windows: f32 {:.2e} f16 {:.2e} q8 {:.2e}",
                 dy[0], dy[1], dy[2], ds[0], ds[1], ds[2]);
        for d in [&dy, &ds] {
            assert!(d[0] < 1e-4 && d[1] < 2e-3 && d[2] < 3e-2, "{d:?}");
        }
    }
}
