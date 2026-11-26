//! Answers through Prastara's Rust engine (am2strata-hybrid --serve) on the tiny fixtures, on the CPU: the same greedy
//! answers as the Workshop's own model (the chat template is rendered here, the engine continues the same ids), a
//! looped Qwen3 at each pass count, and a whole regression suite through the engine with the loss tests on the
//! Workshop's model. Needs the engine binary: `AM2STRATA_HYBRID_EXE`, or am2strata-hybrid on the PATH (a CPU build:
//! `cargo build --release --manifest-path crates/strata-hybrid/Cargo.toml --no-default-features`); without it these
//! tests say so and check nothing.

mod common;

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use candle_core::{Device, Tensor};
use common::*;
use serde_json::json;
use strata_workshop::chatdata::ChatTokenizer;
use strata_workshop::engine::{find_exe, Engine, EngineGenerator, EngineSpec};
use strata_workshop::model::{make_looped, BaseDtype, LlamaModel, LoopSpec, LOOP_FILE};
use strata_workshop::regress::{self, Answer, Generator};

fn engine_exe() -> Option<PathBuf> {
    match find_exe(None) {
        Ok(p) => Some(p),
        Err(e) => {
            eprintln!("skipped: {e:#}");
            None
        }
    }
}

fn to_gguf(dir: &Path) -> PathBuf {
    let g = dir.join("model.f32.gguf");
    strata_workshop::convert::convert(dir, &g, strata_workshop::convert::OutType::F32, Some("gpt-2")).unwrap();
    g
}

const PROMPTS: [&str; 5] = ["What colour is the sky?", "Write add(a, b).", "Tell me a story about a fox and a crow.", "snow: which colour?", "2 + 2 = ?"];

/// Greedy answers of both paths to `prompts`, and how long each took.
fn both(model_dir: &Path, gguf: &Path, exe: &Path, passes: Option<usize>, prompts: &[&str], max_new: usize) -> (Vec<Vec<u32>>, Vec<Vec<u32>>, f64, f64) {
    let tok = ChatTokenizer::load(model_dir, None).unwrap();
    let mut m = LlamaModel::load(model_dir, BaseDtype::F32, &Device::Cpu).unwrap();
    if let Some(p) = passes {
        m.set_passes(p).unwrap();
    }
    let w = Generator::new(&m, &tok, 4096);
    let t0 = Instant::now();
    let ws: Vec<Vec<u32>> = prompts.iter().map(|p| w.generate(&w.prompt(p).unwrap().unwrap(), max_new, None).unwrap()).collect();
    let wt = t0.elapsed().as_secs_f64();
    let mut spec = EngineSpec::new(exe.to_path_buf(), gguf.to_path_buf(), vec!["--threads".into(), "2".into()]);
    spec.log = Some(model_dir.join("engine.log"));
    let e = EngineGenerator::new(Engine::start(&spec, passes).unwrap(), &tok, 4096);
    let t0 = Instant::now();
    let es: Vec<Vec<u32>> = prompts.iter().map(|p| e.generate(&e.prompt(p).unwrap().unwrap(), max_new, None).unwrap()).collect();
    let et = t0.elapsed().as_secs_f64();
    let sp = e.into_speed();
    eprintln!("workshop {wt:.2} s, engine {et:.2} s ({} tokens at {:.0} tokens/s decoding, loaded in {:.2} s)", sp.generated, sp.tokens_per_second(), sp.load_seconds);
    (ws, es, wt, et)
}

#[test]
fn the_engine_gives_the_workshops_greedy_answers_with_the_same_chat_template() {
    let Some(exe) = engine_exe() else { return };
    let root = temp_dir("engine-parity");
    let model = root.join("model");
    chat_model(&model);
    let gguf = to_gguf(&model);
    let cfg = strata_workshop::model::ModelConfig::load(&model).unwrap();
    assert_eq!(strata_workshop::engine::check_gguf_matches(&gguf, &cfg).unwrap(), "llama");
    let (ws, es, _, _) = both(&model, &gguf, &exe, None, &PROMPTS, 24);
    assert!(ws.iter().any(|a| !a.is_empty()), "the tiny model says something");
    assert_eq!(ws, es, "greedy answers agree token for token");
}

/// The looped tiny chat model with open gates (different per pass and channel): each pass count gives the same
/// answers on both paths, and the pass counts differ from each other (the loop really runs).
#[test]
fn a_looped_model_answers_the_same_at_each_pass_count() {
    let Some(exe) = engine_exe() else { return };
    let root = temp_dir("engine-loop");
    let (base, dir) = (root.join("base"), root.join("looped"));
    chat_model(&base);
    make_looped(&base, &dir, LoopSpec { start: 1, end: 3, passes: 3 }).unwrap();
    let hidden = 32;
    let gate = |p: usize| Tensor::new((0..hidden).map(|c| 0.1 + 0.3 * p as f32 + 0.01 * c as f32).collect::<Vec<f32>>(), &Device::Cpu).unwrap();
    let gates: HashMap<String, Tensor> = [("loop.gate.1".to_string(), gate(1)), ("loop.gate.2".to_string(), gate(2))].into();
    candle_core::safetensors::save(&gates, dir.join(LOOP_FILE)).unwrap();
    let gguf = to_gguf(&dir);
    let mut per_pass = Vec::new();
    for p in 1..=3 {
        let (ws, es, _, _) = both(&dir, &gguf, &exe, Some(p), &PROMPTS[..3], 16);
        assert_eq!(ws, es, "{p} pass(es): the same greedy answers");
        per_pass.push(ws);
    }
    assert!(per_pass[0] != per_pass[2], "the loop changes the answers: {per_pass:?}");
}

/// A suite with every kind but judge, the generation tests answered by the engine: the same answers and scores as the
/// Workshop's own run, the loss test on the Workshop's model either way, and repeat/samples in the report.
#[test]
fn a_suite_runs_through_the_engine_with_the_workshops_numbers() {
    let Some(exe) = engine_exe() else { return };
    let root = temp_dir("engine-suite");
    let model = root.join("model");
    chat_model(&model);
    let gguf = to_gguf(&model);
    write_jsonl(&root.join("heldout.jsonl"), &chat_rows(4));
    write_jsonl(&root.join("code.jsonl"), &[json!({"id": "add", "prompt": "Write add(a, b).", "test": "assert add(1, 2) == 3"})]);
    write_jsonl(&root.join("qa.jsonl"), &[json!({"id": "q1", "context": "The sky is blue.", "question": "What colour is the sky?", "answers": ["blue"]})]);
    write_jsonl(&root.join("prompts.jsonl"), &PROMPTS.iter().enumerate().map(|(i, p)| json!({"id": format!("p{i}"), "prompt": p})).collect::<Vec<_>>());
    let suite = root.join("suite.json");
    std::fs::write(&suite, json!({"name": "engine", "tests": [
        {"name": "held-out", "kind": "loss", "data": "heldout.jsonl"},
        {"name": "code", "kind": "code", "format": "jsonl", "data": "code.jsonl", "max_tokens": 16},
        {"name": "qa", "kind": "qa", "format": "jsonl", "data": "qa.jsonl", "max_tokens": 8},
        {"name": "repeat", "kind": "repeat", "data": "prompts.jsonl", "max_tokens": 48},
        {"name": "samples", "kind": "samples", "data": "prompts.jsonl", "max_tokens": 24},
    ]}).to_string()).unwrap();
    let plain = regress::Options::new(&model, &suite, root.join("workshop"));
    let t0 = Instant::now();
    let a = regress::run(&plain, None, &mut |_| {}, &|| false).unwrap();
    let wt = t0.elapsed().as_secs_f64();
    let mut via = regress::Options::new(&model, &suite, root.join("engine"));
    via.engine = Some(EngineSpec::new(exe, gguf, vec!["--threads".into(), "2".into()]));
    via.against = vec![root.join("workshop")];
    let mut log = vec![];
    let t0 = Instant::now();
    let b = regress::run(&via, None, &mut |v| log.push(v), &|| false).unwrap();
    eprintln!("the suite: workshop {wt:.2} s, engine {:.2} s", t0.elapsed().as_secs_f64());
    assert!(b.results.iter().all(|r| r.error.is_none()), "{:?}", b.results.iter().map(|r| &r.error).collect::<Vec<_>>());
    for (x, y) in a.results.iter().zip(&b.results) {
        assert_eq!((&x.test, &x.items), (&y.test, &y.items), "the same scores");
    }
    let by = |r: &regress::Report, t: &str| r.results.iter().find(|x| x.test == t).unwrap().clone();
    assert_eq!(by(&b, "code").extra["answered_by"], "engine");
    assert!(by(&b, "held-out").extra.get("answered_by").is_none(), "the loss test is the Workshop's");
    assert_eq!(by(&a, "samples").extra["answers"], by(&b, "samples").extra["answers"], "the same sample answers");
    let rep = by(&b, "repeat");
    assert!(!rep.higher_is_better && rep.items.len() == PROMPTS.len() && rep.extra["distinct_2"].is_number());
    assert!(b.regressions().is_empty(), "the same answers regress nothing: {:?}", b.json["regressions"]);
    let md = std::fs::read_to_string(root.join("engine/report.md")).unwrap();
    assert!(md.contains("## Samples: samples") && md.contains("**Before**") && md.contains("**Now**"), "{md}");
    assert!(md.contains("Engine at 1 pass(es)"), "{md}");
    assert!(b.json["engine"]["speed"][0]["generated"].as_u64().unwrap() > 0);
    assert!(root.join("engine/engine.log").exists());
    assert!(log.iter().any(|e| e["message"].as_str().is_some_and(|m| m.contains("answered by the engine"))));
}

/// A GGUF of another model is refused before anything runs.
#[test]
fn a_gguf_of_another_model_is_refused() {
    let root = temp_dir("engine-mismatch");
    let (a, b) = (root.join("a"), root.join("b"));
    chat_model(&a);
    chat_model_sized(&b, 2, 32, 64);
    let gguf = to_gguf(&b);
    let cfg = strata_workshop::model::ModelConfig::load(&a).unwrap();
    let e = strata_workshop::engine::check_gguf_matches(&gguf, &cfg).unwrap_err();
    assert!(format!("{e:#}").contains("blocks"), "{e:#}");
}
