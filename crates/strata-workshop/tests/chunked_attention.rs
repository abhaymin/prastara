//! Chunked attention (docs/MULA-PLAN.md L3): the same logits and gradients as the seq x seq table, for full and
//! sliding layers, on a looped model, with and without checkpointing. One test: it sets the process-wide thresholds.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::Ordering;

use candle_core::{Device, Tensor};
use strata_workshop::model::{make_looped, BaseDtype, Ctx, LlamaModel, LoopSpec, LoraSpec, Target, ATTENTION_CHUNK, CHUNKED_ABOVE, LOOP_FILE};

const TOKENS: [u32; 11] = [1, 7, 42, 3, 59, 5, 11, 2, 9, 33, 4];

fn model_dir(tag: &str, sliding: bool) -> PathBuf {
    let src = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny-qwen3-hf");
    let d = std::env::temp_dir().join(format!("chunked-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    for e in std::fs::read_dir(&src).unwrap().flatten() {
        std::fs::copy(e.path(), d.join(e.file_name())).unwrap();
    }
    if sliding {
        let mut c: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(d.join("config.json")).unwrap()).unwrap();
        let n = c["num_hidden_layers"].as_u64().unwrap() as usize;
        c["layer_types"] = serde_json::json!((0..n).map(|i| if i == 0 { "full_attention" } else { "sliding_attention" }).collect::<Vec<_>>());
        c["sliding_window"] = serde_json::json!(4);
        std::fs::write(d.join("config.json"), c.to_string()).unwrap();
    }
    let looped = d.with_extension("loop");
    let _ = std::fs::remove_dir_all(&looped);
    make_looped(&d, &looped, LoopSpec { start: 1, end: 3, passes: 2 }).unwrap();
    let hidden = LlamaModel::load(&looped, BaseDtype::F32, &Device::Cpu).unwrap().cfg.hidden_size;
    let g: HashMap<String, Tensor> = [("loop.gate.1".to_string(), Tensor::full(0.3f32, hidden, &Device::Cpu).unwrap())].into();
    candle_core::safetensors::save(&g, looped.join(LOOP_FILE)).unwrap();
    looped
}

/// logits and every adapter's gradient (with B set non-zero so A's show)
fn run(dir: &PathBuf, ckpt: bool) -> (Vec<f32>, Vec<f32>) {
    let mut m = LlamaModel::load(dir, BaseDtype::F32, &Device::Cpu).unwrap();
    m.add_lora(&LoraSpec { targets: vec![Target::Q, Target::K, Target::V, Target::O], ..LoraSpec::default() }).unwrap();
    for (n, v) in m.lora_vars() {
        if n.ends_with("lora_B.weight") {
            v.set(&Tensor::full(0.02f32, v.as_tensor().shape(), &Device::Cpu).unwrap()).unwrap();
        }
    }
    m.set_checkpointing(ckpt);
    let logits: Vec<f32> = m.forward(&TOKENS, &Ctx::eval()).unwrap().flatten_all().unwrap().to_vec1().unwrap();
    let g = m.loss(&TOKENS, &vec![true; TOKENS.len()], &Ctx::eval()).unwrap().backward().unwrap();
    let extra = m.take_checkpoint_grads();
    let mut grads = Vec::new();
    for (_, v) in m.lora_vars() {
        let mut t = g.get(&v).cloned().unwrap_or_else(|| v.as_tensor().zeros_like().unwrap());
        for (w, e) in &extra {
            if w.as_tensor().id() == v.as_tensor().id() {
                t = (t + e).unwrap();
            }
        }
        grads.extend(t.flatten_all().unwrap().to_vec1::<f32>().unwrap());
    }
    (logits, grads)
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

#[test]
fn chunked_attention_equals_the_table() {
    for sliding in [false, true] {
        let dir = model_dir(if sliding { "sliding" } else { "full" }, sliding);
        CHUNKED_ABOVE.store(usize::MAX, Ordering::Relaxed);
        let (tl, tg) = run(&dir, false);
        CHUNKED_ABOVE.store(2, Ordering::Relaxed);
        for chunk in [1, 3, 4, 16] {
            ATTENTION_CHUNK.store(chunk, Ordering::Relaxed);
            for ckpt in [false, true] {
                let (cl, cg) = run(&dir, ckpt);
                let (dl, dg) = (max_diff(&tl, &cl), max_diff(&tg, &cg));
                let big = tg.iter().fold(0.0f32, |m, x| m.max(x.abs()));
                println!("sliding {sliding} chunk {chunk} checkpoint {ckpt}: logits {dl:.2e}, gradients {dg:.2e} (largest {big:.2e})");
                assert!(dl < 1e-5 && dg <= big * 1e-5 + 1e-7, "logits {dl}, gradients {dg}");
            }
        }
        CHUNKED_ABOVE.store(2048, Ordering::Relaxed);
        ATTENTION_CHUNK.store(512, Ordering::Relaxed);
        let _ = std::fs::remove_dir_all(dir.with_extension(""));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
