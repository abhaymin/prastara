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
