//! Abliteration that survives training (strata_workshop::carry), on the tiny fixtures and the CPU: the direction is
//! saved beside an abliterated model and linked into its upcycled copy; `--direction-only` recomputes the same r for a
//! folder made without it and refuses another site; a merge of a LoRA whose o_proj and down_proj B write along r gives
//! writers with ‖rᵀW‖ ≈ 0; export-moe of an upcycled abliterated model gives experts' down tensors (and attn_output)
//! orthogonal to r, read back from the GGUF; the marks are carried into the config and the GGUF; a base without its
//! direction file is refused; refusal-check answers through the Rust engine on a GGUF and a Kosha pack. The prompts
//! are placeholders, never real requests.

mod common;

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use candle_core::{Device, Tensor};
use serde_json::{json, Value};
use strata_workshop::abliterate::{self, orthogonalize, residual_write, role, LayerChoice, Role, Site};
use strata_workshop::carry::{self, Mode, DIRECTION_FILE};
use strata_workshop::export_moe::{export_moe_carry, ExportType};
use strata_workshop::merge::Merger;
use strata_workshop::model::{adapter_config, adapter_tensors, make_upcycled, BaseDtype, LlamaModel, LoraSpec, MoeSpec, Target};
use strata_workshop::st_writer;

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(name)
}

fn config(dir: &Path) -> Value {
    serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap()
}

fn unit(v: Vec<f32>) -> Vec<f32> {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    v.into_iter().map(|x| x / n).collect()
}

/// A fixed direction for a model `hidden` wide.
fn some_r(hidden: usize) -> Vec<f32> {
    unit((0..hidden).map(|i| ((i * 7 % 5) as f32) - 2.0 + 0.1 * i as f32).collect())
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let d: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
    let n = |v: &[f32]| v.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    d / (n(a) * n(b))
}

/// Every tensor of a folder as f32.
fn tensors(dir: &Path) -> HashMap<String, Tensor> {
    let mut out = HashMap::new();
    for f in strata_workshop::model::weight_files(dir).unwrap() {
        for (n, t) in candle_core::safetensors::load(&f, &Device::Cpu).unwrap() {
            out.insert(n, t.to_dtype(candle_core::DType::F32).unwrap());
        }
    }
    out
}

/// An abliterated copy of `src` along `r` (block 1, pass 0, scale 1), with the tokenizer of `tok` when it has none.
fn abliterated(src: &Path, out: &Path, r: &[f32]) {
    orthogonalize(src, out, r, 1.0, json!({"layer": 1, "pass": 0, "scale": 1.0, "separation": 3.5, "prompts": {"harmful": 4, "harmless": 4}, "chosen": "given"}),
                  st_writer::SHARD_BYTES).unwrap();
    let tok = fixture("tiny-llama-hf");
    for f in ["tokenizer.json", "tokenizer_config.json"] {
        if !out.join(f).exists() {
            std::fs::copy(tok.join(f), out.join(f)).unwrap();
        }
    }
}

#[test]
fn the_direction_is_saved_and_reloads_and_an_upcycle_links_it() {
    let tmp = common::temp_dir("carry-saved");
    let src = fixture("tiny-qwen3-hf");
    let r = some_r(32);
    let abl = tmp.join("abl");
    abliterated(&src, &abl, &r);
    let d = carry::load(&abl.join(DIRECTION_FILE)).unwrap();
    assert_eq!(d.r, r, "the direction reads back exactly");
    assert_eq!((d.scale, d.layer, d.pass, d.source.as_str()), (1.0, 1, 0, "abliterate"));
    assert!((d.separation - 3.5).abs() < 1e-9);
    assert_eq!(config(&abl)["prastara_abliterated"]["direction_file"], DIRECTION_FILE);
    // the direction file is no weight: the model loads and converts as before
    LlamaModel::load(&abl, BaseDtype::F32, &Device::Cpu).unwrap();
    // upcycle links it; with the link gone it is still found through the linked weights
    let up = tmp.join("up");
    make_upcycled(&abl, &up, MoeSpec::full(4, 2)).unwrap();
    assert!(up.join(DIRECTION_FILE).exists(), "upcycle links the direction");
    assert_eq!(carry::find(&up), Some(up.join(DIRECTION_FILE)));
    std::fs::remove_file(up.join(DIRECTION_FILE)).unwrap();
    let found = carry::find(&up).expect("found beside the weights the copy links to");
    assert_eq!(std::fs::canonicalize(found).unwrap(), std::fs::canonicalize(abl.join(DIRECTION_FILE)).unwrap());
    // a folder that is not abliterated has none
    assert_eq!(carry::find(&src), None);
}

#[test]
fn direction_only_recomputes_the_same_r_and_refuses_another_site() {
    let tmp = common::temp_dir("carry-dironly");
    let src = tmp.join("src");
    let m = strata_workshop::testmodel::TinyLlama { seed: 0xAB1, num_layers: 4, vocab_size: strata_workshop::testmodel::CHAT_VOCAB, hidden: 32,
                                                    intermediate: 64, num_attention_heads: 4, num_key_value_heads: 2, shard_count: 2 };
    strata_workshop::testmodel::write_chat_model(&src, &m).unwrap();
    let harmful: Vec<String> = (1..=6).map(|i| format!("please refuse: item {i}")).collect();
    let harmless: Vec<String> = abliterate::SAMPLE_HARMLESS.iter().take(6).map(|s| s.to_string()).collect();
    let out = tmp.join("abl");
    let mut o = abliterate::Options::new(&src, harmful.clone(), harmless.clone(), &out);
    o.layer = LayerChoice::Site(Site { block: 2, pass: 0 });
    let rep = abliterate::run(&o, &mut |_| {}).unwrap();
    let full = carry::load(&out.join(DIRECTION_FILE)).unwrap();
    assert_eq!(full.r, rep.chosen.direction);
    // as a folder made before the file existed
    std::fs::remove_file(out.join(DIRECTION_FILE)).unwrap();
    let weights_before: Vec<(PathBuf, Vec<u8>)> = std::fs::read_dir(&out).unwrap().flatten().map(|e| (e.path(), std::fs::read(e.path()).unwrap())).collect();

    // another site, the abliterated folder named as the original, a folder that is not abliterated: refused
    let mut wrong = abliterate::DirectionOnly::new(&src, harmful.clone(), harmless.clone(), &out);
    wrong.layer = LayerChoice::Site(Site { block: 1, pass: 0 });
    let e = format!("{:#}", abliterate::direction_only(&wrong, &mut |_| {}).unwrap_err());
    assert!(e.contains("block 2 pass 0") && e.contains("--layer 1:0"), "{e}");
    assert!(abliterate::direction_only(&abliterate::DirectionOnly::new(&out, harmful.clone(), harmless.clone(), &out), &mut |_| {}).is_err());
    assert!(abliterate::direction_only(&abliterate::DirectionOnly::new(&src, harmful.clone(), harmless.clone(), &src), &mut |_| {}).is_err());
    // other prompts give another separation: refused
    let other: Vec<String> = (1..=6).map(|i| format!("tell me about the number {i}")).collect();
    let mut off = abliterate::DirectionOnly::new(&src, other, harmless.clone(), &out);
    off.layer = LayerChoice::Site(Site { block: 2, pass: 0 });
    assert!(abliterate::direction_only(&off, &mut |_| {}).is_err(), "other prompts are not the direction that was removed");
    assert!(!out.join(DIRECTION_FILE).exists(), "nothing written by a refused run");

    // the recorded site (given, then taken from the record): the same r, and only the direction file is written
    let mut right = abliterate::DirectionOnly::new(&src, harmful.clone(), harmless.clone(), &out);
    right.layer = LayerChoice::Site(Site { block: 2, pass: 0 });
    let got = abliterate::direction_only(&right, &mut |_| {}).unwrap();
    let again = carry::load(&got.file).unwrap();
    assert!(cosine(&again.r, &full.r) > 0.9999, "cosine {}", cosine(&again.r, &full.r));
    assert_eq!((again.layer, again.pass, again.source.as_str()), (2, 0, "direction-only"));
    assert!(got.separation_error < 1e-6, "{}", got.separation_error);
    assert!(got.writers.iter().all(|(_, b, a)| *b > 1e-3 && *a < 1e-5), "{:?}", got.writers);
    for (p, bytes) in &weights_before {
        assert_eq!(&std::fs::read(p).unwrap(), bytes, "{} is untouched", p.display());
    }
    // a second time: refused, the file is there
    assert!(abliterate::direction_only(&right, &mut |_| {}).is_err());
    std::fs::remove_file(out.join(DIRECTION_FILE)).unwrap();
    abliterate::direction_only(&abliterate::DirectionOnly::new(&src, harmful, harmless, &out), &mut |_| {}).unwrap();
    assert!(cosine(&carry::load(&out.join(DIRECTION_FILE)).unwrap().r, &full.r) > 0.9999, "auto takes the recorded site");
}

/// A PEFT adapter for the chat model whose o_proj and down_proj B write along `r` (and a q_proj one, a reader).
fn adapter_writing_r(dir: &Path, cfg: &strata_workshop::model::ModelConfig, r: &[f32]) {
    std::fs::create_dir_all(dir).unwrap();
    let rank = 4;
    let (h, ff) = (cfg.hidden_size, cfg.intermediate_size);
    let q = cfg.num_attention_heads * cfg.head_dim;
    let mut t: HashMap<String, Tensor> = HashMap::new();
    let val = |seed: usize, i: usize| (((i * 31 + seed * 17) % 23) as f32 - 11.0) / 40.0;
    for b in 0..cfg.num_hidden_layers {
        for (k, out, inn) in [("self_attn.o_proj", h, q), ("mlp.down_proj", h, ff), ("self_attn.q_proj", q, h)] {
            let stem = format!("base_model.model.model.layers.{b}.{k}");
            let a: Vec<f32> = (0..rank * inn).map(|i| val(b + 1, i)).collect();
            // B = r uᵀ + noise: the change writes along r
            let bm: Vec<f32> = (0..out * rank).map(|i| { let (o, c) = (i / rank, i % rank); (if out == h { 2.0 * r[o] * (c as f32 + 1.0) } else { 0.0 }) + val(b + 7, i) }).collect();
            t.insert(format!("{stem}.lora_A.weight"), Tensor::from_vec(a, (rank, inn), &Device::Cpu).unwrap());
            t.insert(format!("{stem}.lora_B.weight"), Tensor::from_vec(bm, (out, rank), &Device::Cpu).unwrap());
        }
    }
    candle_core::safetensors::save(&t, dir.join("adapter_model.safetensors")).unwrap();
    std::fs::write(dir.join("adapter_config.json"), json!({"r": rank, "lora_alpha": 8, "target_modules": ["q_proj", "o_proj", "down_proj"]}).to_string()).unwrap();
}

fn merge_with(base: &Path, ad: &Path, out: &Path, mode: Mode) -> anyhow::Result<strata_workshop::merge::MergeReport> {
    let mut m = Merger::new(base, ad, out);
    m.set_carry(carry::Options { mode, direction: None });
    m.run()
}

#[test]
fn a_merge_keeps_every_writer_off_r_and_carries_the_marks() {
    let tmp = common::temp_dir("carry-merge");
    let src = tmp.join("src");
    common::chat_model(&src);
    let cfg = strata_workshop::model::ModelConfig::load(&src).unwrap();
    let r = some_r(cfg.hidden_size);
    let abl = tmp.join("abl");
    abliterated(&src, &abl, &r);
    let ad = tmp.join("adapter");
    adapter_writing_r(&ad, &cfg, &r);

    let carried = tmp.join("carried");
    let rep = merge_with(&abl, &ad, &carried, Mode::Carry).unwrap();
    assert_eq!(rep.carry.state, "carried");
    assert_eq!(rep.carry.projected.len(), 2 * cfg.num_hidden_layers, "every o_proj and down_proj, not q_proj: {:?}", rep.carry.projected);
    let skipped = tmp.join("skipped");
    let rep_skip = merge_with(&abl, &ad, &skipped, Mode::Skip).unwrap();
    assert_eq!((rep_skip.carry.state.as_str(), rep_skip.carry.projected.len()), ("dropped", 0));
    assert!(rep_skip.carry_lines[0].starts_with("WARNING"));

    let (a, c, s) = (tensors(&abl), tensors(&carried), tensors(&skipped));
    let mut writers = 0;
    for (name, t) in &c {
        match role(name, t.dims(), cfg.hidden_size).unwrap() {
            Role::Writer { axis } => {
                writers += 1;
                let w = residual_write(t, axis, &r).unwrap();
                assert!(w < 1e-5, "{name}: the merged writer writes {w} along r");
                let changed = (t - &a[name]).unwrap().abs().unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap();
                if name.contains("o_proj") || name.contains("down_proj.weight") {
                    assert!(changed > 0.0, "{name}: the adapter changed it");
                    // without the carry the same merge writes along r again
                    assert!(residual_write(&s[name], axis, &r).unwrap() > 1e-3, "{name}: the test's adapter writes along r");
                }
            }
            Role::Embed => assert!(residual_write(t, 1, &r).unwrap() < 1e-5, "{name}"),
            Role::Untouched => {
                let same = (t - &s[name]).unwrap().abs().unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap();
                assert_eq!(same, 0.0, "{name}: a reader is merged as without the carry");
            }
        }
    }
    assert_eq!(writers, 2 * cfg.num_hidden_layers);

    // the marks: carried, private, the direction beside the result; dropped without the carry (still private)
    let cc = config(&carried);
    assert_eq!((cc["prastara_abliterated"]["carried"].clone(), cc["prastara_abliterated"]["layer"].clone(), cc["prastara_private"].clone()), (json!(true), json!(1), json!(true)), "{cc}");
    assert_eq!(carry::load(&carried.join(DIRECTION_FILE)).unwrap().r, r);
    let sc = config(&skipped);
    assert!(sc["prastara_abliterated"].is_null() && sc["prastara_private"] == true, "{sc}");
    assert!(!skipped.join(DIRECTION_FILE).exists());
    // ... and into the GGUF (and its quantized copy)
    let gguf = tmp.join("carried.f32.gguf");
    strata_workshop::convert::convert(&carried, &gguf, strata_workshop::convert::OutType::F32, Some("gpt-2")).unwrap();
    let q8 = tmp.join("carried.q8_0.gguf");
    strata_workshop::quantize_gguf::quantize(&gguf, &q8, strata_workshop::quantize_gguf::PureType::from_s("q8_0").unwrap()).unwrap();
    for f in [&gguf, &q8] {
        let g = strata_gguf::Gguf::open(f).unwrap();
        for k in [strata_workshop::convert::PRIVATE_KEY, strata_workshop::convert::ABLITERATED_KEY, "prastara.abliterated.carried"] {
            assert_eq!(g.metadata.get(k), Some(&json!(true)), "{k} in {}", f.display());
        }
    }
    let gs = tmp.join("skipped.f32.gguf");
    strata_workshop::convert::convert(&skipped, &gs, strata_workshop::convert::OutType::F32, Some("gpt-2")).unwrap();
    let g = strata_gguf::Gguf::open(&gs).unwrap();
    assert_eq!(g.metadata.get(strata_workshop::convert::PRIVATE_KEY), Some(&json!(true)));
    assert!(g.metadata.get(strata_workshop::convert::ABLITERATED_KEY).is_none());
    // a base that is not abliterated merges as it always did
    let plain = tmp.join("plain");
    assert_eq!(merge_with(&src, &ad, &plain, Mode::Carry).unwrap().carry.state, "none");
    assert!(config(&plain)["prastara_abliterated"].is_null());
}

#[test]
fn a_base_without_its_direction_file_is_refused_unless_told() {
    let tmp = common::temp_dir("carry-missing");
    let src = tmp.join("src");
    common::chat_model(&src);
    let cfg = strata_workshop::model::ModelConfig::load(&src).unwrap();
    let r = some_r(cfg.hidden_size);
    let abl = tmp.join("abl");
    abliterated(&src, &abl, &r);
    std::fs::remove_file(abl.join(DIRECTION_FILE)).unwrap();
    let ad = tmp.join("adapter");
    adapter_writing_r(&ad, &cfg, &r);
    let out = tmp.join("merged");
    let e = format!("{:#}", merge_with(&abl, &ad, &out, Mode::Carry).err().expect("refused"));
    assert!(e.contains("--direction-only") && e.contains("--layer 1 --pass 0") && e.contains("--allow-unprojected"), "{e}");
    assert!(!out.exists(), "nothing written");
    let rep = merge_with(&abl, &ad, &out, Mode::AllowUnprojected).unwrap();
    assert_eq!(rep.carry.state, "unprojected");
    let c = config(&out);
    assert_eq!((c["prastara_abliterated"]["carried"].clone(), c["prastara_private"].clone()), (json!(false), json!(true)), "{c}");
    // a direction named outright that does not fit the model is refused
    let wrong = tmp.join("wrong");
    std::fs::create_dir_all(&wrong).unwrap();
    carry::save(&wrong, &carry::Direction { r: some_r(cfg.hidden_size), scale: 1.0, layer: 3, pass: 0, separation: 1.0, source: "abliterate".into() }).unwrap();
    let merged2 = tmp.join("merged-2");
    let mut m = Merger::new(&abl, &ad, &merged2);
    m.set_carry(carry::Options { mode: Mode::Carry, direction: Some(wrong.join(DIRECTION_FILE)) });
    assert!(format!("{:#}", m.run().err().unwrap()).contains("block 3"));
    // export-moe refuses the same way
    let up = tmp.join("up");
    make_upcycled(&abl, &up, MoeSpec::full(4, 2)).unwrap();
    let e = format!("{:#}", export_moe_carry(&up, None, &tmp.join("x.gguf"), ExportType::F32, Some("gpt-2"), &Default::default(), &mut |_, _, _| {}).unwrap_err());
    assert!(e.contains("--direction-only"), "{e}");
    assert!(!tmp.join("x.gguf").exists() && !tmp.join("x.gguf.part").exists());
}

/// A deterministic value in [-1, 1) for index `i` of the tensor named `name`.
fn noise(name: &str, i: usize) -> f32 {
    let mut s = name.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| (h ^ b as u64).wrapping_mul(0x100_0000_01b3)) ^ (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    s ^= s >> 33;
    s = s.wrapping_mul(0xff51_afd7_ed55_8ccd);
    s ^= s >> 33;
    (s >> 40) as f32 / 8_388_608.0 - 1.0
}

/// One tensor of an F32 GGUF as row-major f32 values with its numpy-order shape.
fn gguf_f32(path: &Path, name: &str) -> (Vec<usize>, Vec<f32>) {
    let g = strata_gguf::Gguf::open(path).unwrap();
    let t = g.tensors.iter().find(|t| t.name == name).unwrap_or_else(|| panic!("{name}"));
    assert_eq!(t.ggml_type, 0, "{name} is F32");
    let bytes = std::fs::read(path).unwrap();
    let start = (g.data_start + t.offset) as usize;
    let n = t.dims.iter().product::<u64>() as usize;
    (t.dims.iter().rev().map(|d| *d as usize).collect(), bytes[start..start + 4 * n].chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
}

#[test]
fn export_moe_keeps_the_experts_down_projections_off_r() {
    let tmp = common::temp_dir("carry-moe");
    let r = some_r(32);
    let abl = tmp.join("abl");
    abliterated(&fixture("tiny-qwen3-hf"), &abl, &r);
    let up = tmp.join("up");
    make_upcycled(&abl, &up, MoeSpec::full(4, 2)).unwrap();
    // "trained": the shared attention adapters (o_proj among them) and every expert's LoRA set to fixed noise
    let spec = LoraSpec { targets: vec![Target::Q, Target::K, Target::V, Target::O], train_layers: Some(2), ..LoraSpec::default() };
    let mut m = LlamaModel::load(&up, BaseDtype::F32, &Device::Cpu).unwrap();
    m.add_lora(&spec).unwrap();
    for (n, v) in m.lora_vars() {
        let scale = if n.ends_with("lora_B.weight") { 0.2 } else if n.ends_with(".mlp.gate.weight") { 1.0 } else { continue };
        let vals: Vec<f32> = (0..v.as_tensor().elem_count()).map(|i| scale * noise(&n, i)).collect();
        v.set(&Tensor::from_vec(vals, v.as_tensor().shape(), &Device::Cpu).unwrap()).unwrap();
    }
    let ad = tmp.join("ad");
    std::fs::create_dir_all(&ad).unwrap();
    let t: HashMap<String, Tensor> = adapter_tensors(&m).into_iter().collect();
    candle_core::safetensors::save(&t, ad.join("adapter_model.safetensors")).unwrap();
    std::fs::write(ad.join("adapter_config.json"), serde_json::to_string(&adapter_config(&spec, &m.cfg, "base")).unwrap()).unwrap();

    let gguf = tmp.join("carried.gguf");
    let rep = export_moe_carry(&up, Some(&ad), &gguf, ExportType::F32, Some("gpt-2"), &Default::default(), &mut |_, _, _| {}).unwrap();
    assert_eq!(rep.carry.state, "carried");
    let skip = tmp.join("skipped.gguf");
    export_moe_carry(&up, Some(&ad), &skip, ExportType::F32, Some("gpt-2"), &carry::Options { mode: Mode::Skip, direction: None }, &mut |_, _, _| {}).unwrap();
    let write = |shape: &[usize], v: Vec<f32>, axis: usize| residual_write(&Tensor::from_vec(v, shape.to_vec(), &Device::Cpu).unwrap(), axis, &r).unwrap();
    for b in 0..3 {
        // experts' down: [E, hidden, ff], each expert's hidden axis 1; attn_output: [hidden, q], axis 0
        let (shape, v) = gguf_f32(&gguf, &format!("blk.{b}.ffn_down_exps.weight"));
        assert_eq!(shape, vec![4, 32, 64]);
        for e in 0..4 {
            let one = v[e * 32 * 64..(e + 1) * 32 * 64].to_vec();
            assert!(write(&[32, 64], one, 0) < 1e-5, "block {b} expert {e} writes along r");
        }
        let (s2, o) = gguf_f32(&gguf, &format!("blk.{b}.attn_output.weight"));
        assert!(write(&s2, o, 0) < 1e-5, "block {b} attn_output");
        if b >= 1 {
            // the trained blocks: without the carry the experts' LoRAs and o_proj write along r again
            let (shape, v) = gguf_f32(&skip, &format!("blk.{b}.ffn_down_exps.weight"));
            assert!(write(&shape, v, 1) > 1e-3, "block {b}: the test's expert LoRAs write along r");
            let (s2, o) = gguf_f32(&skip, &format!("blk.{b}.attn_output.weight"));
            assert!(write(&s2, o, 0) > 1e-3, "block {b}: the test's o_proj adapter writes along r");
            // the readers are exported as without the carry
            assert_eq!(gguf_f32(&gguf, &format!("blk.{b}.ffn_up_exps.weight")), gguf_f32(&skip, &format!("blk.{b}.ffn_up_exps.weight")));
            assert_eq!(gguf_f32(&gguf, &format!("blk.{b}.attn_q.weight")), gguf_f32(&skip, &format!("blk.{b}.attn_q.weight")));
        }
    }
    // the marks
    let g = strata_gguf::Gguf::open(&gguf).unwrap();
    for k in [strata_workshop::convert::PRIVATE_KEY, strata_workshop::convert::ABLITERATED_KEY, "prastara.abliterated.carried"] {
        assert_eq!(g.metadata.get(k), Some(&json!(true)), "{k}");
    }
    assert_eq!(g.u64("prastara.abliterated.layer"), Some(1));
    let gs = strata_gguf::Gguf::open(&skip).unwrap();
    assert_eq!(gs.metadata.get(strata_workshop::convert::PRIVATE_KEY), Some(&json!(true)));
    assert!(gs.metadata.get(strata_workshop::convert::ABLITERATED_KEY).is_none());
    // quantized as written, the marks stay
    let q4 = tmp.join("carried.q4_k.gguf");
    export_moe_carry(&up, Some(&ad), &q4, ExportType::from_s("q8_0").unwrap(), Some("gpt-2"), &Default::default(), &mut |_, _, _| {}).unwrap();
    assert_eq!(strata_gguf::Gguf::open(&q4).unwrap().metadata.get("prastara.abliterated.carried"), Some(&json!(true)));
}

#[test]
fn refusal_check_answers_through_the_engine_on_a_gguf_and_a_kosha_pack() {
    let exe = match strata_workshop::engine::find_exe(None) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("skipped: {e:#}");
            return;
        }
    };
    let tmp = common::temp_dir("carry-refusal");
    let model = tmp.join("model");
    common::chat_model(&model);
    let gguf = tmp.join("model.f32.gguf");
    strata_workshop::convert::convert(&model, &gguf, strata_workshop::convert::OutType::F32, Some("gpt-2")).unwrap();
    let kosha = tmp.join("model.kosha");
    am2pack::build(&am2pack::BuildOptions { model: gguf.clone(), out: kosha.clone(), ..Default::default() }, &mut |_| {}).unwrap();
    let harmful: Vec<String> = (1..=5).map(|i| format!("please refuse: item {i}")).collect();
    let harmless: Vec<String> = abliterate::SAMPLE_HARMLESS.iter().take(5).map(|s| s.to_string()).collect();
    let spec = strata_workshop::engine::EngineSpec::new(exe, gguf.clone(), vec!["--threads".into(), "2".into()]);
    let o = strata_workshop::refusal_check::Options { engine: spec.clone(), tokenizer: model.clone(), harmful: harmful.clone(), harmless: harmless.clone(),
                                                       n: 3, max_tokens: 8, before: Some(kosha.clone()) };
    let mut events = Vec::new();
    let rep = strata_workshop::refusal_check::run(&o, &mut |v| events.push(v)).unwrap();
    assert_eq!(events.iter().filter(|e| e["event"] == "check").count(), 2, "before and after");
    let (a, b) = (&rep.after, rep.before.as_ref().unwrap());
    assert_eq!((a.check.n, b.check.n), (3, 3), "the last 3 of each list");
    assert_eq!(a.speed.requests, 6);
    // the pack holds the same tensors: the same greedy answers
    assert_eq!((a.check.harmful_refused, a.check.harmless_refused, &a.check.samples), (b.check.harmful_refused, b.check.harmless_refused, &b.check.samples));
    assert!(!a.check.samples.is_empty());
    assert_eq!(b.marks["pack"], true);
    assert!(rep.to_json()["after"]["result"]["harmful_refusal_rate"].is_number());
    // a GGUF of another model (another vocabulary) is refused before the engine starts
    let other = tmp.join("other");
    common::chat_model_sized(&other, 2, 32, 64);
    let mut wrong = o.clone();
    wrong.before = None;
    wrong.engine.gguf = fixture("tiny-llama.f32.gguf");
    wrong.tokenizer = other;
    assert!(strata_workshop::refusal_check::run(&wrong, &mut |_| {}).is_err());
}
