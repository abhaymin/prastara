//! Avarta loop conversion (docs/MULA-PLAN.md M1) on the tiny Qwen3 fixture: a converted model gives exactly the
//! base's logits; an open gate changes them; the gates get a gradient and train with an adapter, and an adapter saved
//! with its gates loads back to the same logits.

use std::collections::HashMap;
use std::path::PathBuf;

use candle_core::{DType, Device, Tensor};
use strata_workshop::model::{adapter_config, adapter_tensors, make_looped, BaseDtype, Ctx, LlamaModel, LoopSpec, LoraSpec, Target, LOOP_FILE};

fn fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny-qwen3-hf")
}

fn scratch(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("avarta-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    d
}

const TOKENS: [u32; 6] = [1, 7, 42, 3, 59, 5];

fn logits(dir: &std::path::Path) -> Vec<Vec<f32>> {
    LlamaModel::load(dir, BaseDtype::F32, &Device::Cpu).unwrap().forward(&TOKENS, &Ctx::eval()).unwrap().to_vec2().unwrap()
}

#[test]
fn a_converted_model_is_exactly_the_base() {
    let out = scratch("exact");
    make_looped(&fixture(), &out, LoopSpec { start: 1, end: 3, passes: 3 }).unwrap();
    assert!(out.join(LOOP_FILE).exists());
    let cfg: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(out.join("config.json")).unwrap()).unwrap();
    assert_eq!(cfg["prastara_loop"]["passes"], 3);
    assert_eq!(logits(&fixture()), logits(&out), "closed gates must leave every logit bit for bit");
    // converting twice is refused, and so is a loop outside the model
    assert!(make_looped(&out, &scratch("twice"), LoopSpec { start: 0, end: 1, passes: 2 }).is_err());
    assert!(make_looped(&fixture(), &scratch("bad"), LoopSpec { start: 2, end: 9, passes: 2 }).is_err());
    let _ = std::fs::remove_dir_all(&out);
}

#[test]
fn an_open_gate_changes_the_logits() {
    let out = scratch("open");
    make_looped(&fixture(), &out, LoopSpec { start: 0, end: 2, passes: 2 }).unwrap();
    let hidden = LlamaModel::load(&out, BaseDtype::F32, &Device::Cpu).unwrap().cfg.hidden_size;
    let gates: HashMap<String, Tensor> = [("loop.gate.1".to_string(), Tensor::full(0.5f32, hidden, &Device::Cpu).unwrap())].into();
    candle_core::safetensors::save(&gates, out.join(LOOP_FILE)).unwrap();
    let (base, looped) = (logits(&fixture()), logits(&out));
    let diff = base.iter().flatten().zip(looped.iter().flatten()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    assert!(diff > 1e-4 && diff.is_finite(), "an open gate must run the core again (max difference {diff})");
    let _ = std::fs::remove_dir_all(&out);
}

#[test]
fn the_gates_train_with_an_adapter_and_load_back() {
    let out = scratch("train");
    make_looped(&fixture(), &out, LoopSpec { start: 1, end: 3, passes: 2 }).unwrap();
    let mut m = LlamaModel::load(&out, BaseDtype::F32, &Device::Cpu).unwrap();
    // training from inside the loop is refused: its blocks must be trained together
    let inside = LoraSpec { targets: vec![Target::Q, Target::V], train_layers: Some(1), ..LoraSpec::default() };
    assert!(m.add_lora(&inside).is_err());
    let spec = LoraSpec { targets: vec![Target::Q, Target::V], ..LoraSpec::default() };
    m.add_lora(&spec).unwrap();
    let vars = m.lora_vars();
    let gate = vars.iter().find(|(n, _)| n == "base_model.model.model.loop.gate.1").expect("the gate is a trained variable").1.clone();
    let supervised = vec![true; TOKENS.len()];
    let grads = m.loss(&TOKENS, &supervised, &Ctx::eval()).unwrap().backward().unwrap();
    let g = grads.get(&gate).expect("the gate gets a gradient");
    let norm: f32 = g.sqr().unwrap().sum_all().unwrap().to_scalar().unwrap();
    assert!(norm > 0.0, "a closed gate still learns: its gradient is (y - x) times the loss's");

    // one step on the gate, then the adapter (with the gate) is saved and loaded into a fresh model
    gate.set(&(gate.as_tensor() - (g * 0.1).unwrap()).unwrap()).unwrap();
    let stepped = m.forward(&TOKENS, &Ctx::eval()).unwrap().to_vec2::<f32>().unwrap();
    let ad = scratch("adapter");
    std::fs::create_dir_all(&ad).unwrap();
    let tensors: HashMap<String, Tensor> = adapter_tensors(&m).into_iter().collect();
    assert!(tensors.contains_key("base_model.model.model.loop.gate.1"));
    candle_core::safetensors::save(&tensors, ad.join("adapter_model.safetensors")).unwrap();
    std::fs::write(ad.join("adapter_config.json"), serde_json::to_string(&adapter_config(&spec, &m.cfg, "base")).unwrap()).unwrap();
    let mut fresh = LlamaModel::load(&out, BaseDtype::F32, &Device::Cpu).unwrap();
    fresh.load_adapter(&ad).unwrap();
    assert_eq!(fresh.forward(&TOKENS, &Ctx::eval()).unwrap().to_vec2::<f32>().unwrap(), stepped);
    assert_eq!(fresh.loop_gate_tensors()["loop.gate.1"].to_dtype(DType::F32).unwrap().to_vec1::<f32>().unwrap(),
               gate.as_tensor().to_vec1::<f32>().unwrap());
    // merged: the adapter's matrices folded in and its gates in loop.safetensors give the same logits
    let merged = scratch("merged");
    strata_workshop::merge::merge(&out, &ad, &merged).unwrap();
    let got = logits(&merged);
    let diff = got.iter().flatten().zip(stepped.iter().flatten()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    assert!(diff < 1e-4, "the merged looped model differs by {diff}");
    for d in [out, ad, merged] {
        let _ = std::fs::remove_dir_all(d);
    }
}

/// The real base (docs/MULA-PLAN.md M0 and M1): AM2_MULA_BASE=<Qwen3-8B folder> AM2_MULA_OUT=<empty folder>. The base
/// answers "The capital of France is" with " Paris", and its looped copy (blocks 12..24, two passes) gives the same
/// logits bit for bit. About 17 GB of RAM per model, one at a time.
#[test]
#[ignore]
fn the_real_base_runs_and_its_looped_copy_is_exact() {
    let (base, out) = (PathBuf::from(std::env::var("AM2_MULA_BASE").unwrap()), PathBuf::from(std::env::var("AM2_MULA_OUT").unwrap()));
    let tokens = [785u32, 6722, 315, 9625, 374];
    let run = |dir: &std::path::Path| -> Vec<f32> {
        let t0 = std::time::Instant::now();
        let m = LlamaModel::load(dir, BaseDtype::BF16, &Device::Cpu).unwrap();
        let l = m.forward(&tokens, &Ctx::eval()).unwrap();
        let last: Vec<f32> = l.get(tokens.len() - 1).unwrap().to_vec1().unwrap();
        println!("{}: {:.1} s", dir.display(), t0.elapsed().as_secs_f64());
        last
    };
    let a = run(&base);
    let top = a.iter().enumerate().max_by(|x, y| x.1.total_cmp(y.1)).unwrap().0;
    println!("top token {top} (12095 is \" Paris\")");
    assert_eq!(top, 12095);
    if !out.join("config.json").exists() {
        make_looped(&base, &out, LoopSpec { start: 12, end: 24, passes: 2 }).unwrap();
    }
    assert_eq!(a, run(&out), "the looped copy's logits");
}
