//! LoRA upcycling (plan Part C, Mula-MoE) on the tiny Qwen3 fixture: `upcycle` marks a copy; trained, every block
//! is a mixture of experts over the shared feed-forward that starts exactly as the dense base; the router and the
//! experts get gradients (the same with checkpointing), the load-balancing loss is in the loss; the adapter saves and
//! loads back; `export-moe` writes a qwen3moe GGUF. The engine's side of the gate is
//! crates/strata-hybrid/src/model/loop_tests.rs (fixtures written here by `write_the_engines_moe_fixture`).

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use candle_core::{Device, Tensor};
use strata_workshop::export_moe::{export_bytes, export_moe, ExportType};
use strata_workshop::model::{adapter_config, adapter_tensors, make_looped, make_upcycled, moe_router_name, BaseDtype, Ctx, LlamaModel, LoopSpec,
                             LoraSpec, ModelConfig, MoeSpec, Target, LOOP_FILE};

fn fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny-qwen3-hf")
}

fn scratch(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("upcycle-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    d
}

const TOKENS: [u32; 8] = [1, 7, 42, 3, 59, 5, 18, 30];

fn logits(m: &LlamaModel) -> Vec<Vec<f32>> {
    m.forward(&TOKENS, &Ctx::eval()).unwrap().to_vec2().unwrap()
}

fn load(dir: &Path) -> LlamaModel {
    LlamaModel::load(dir, BaseDtype::F32, &Device::Cpu).unwrap()
}

fn max_diff(a: &[Vec<f32>], b: &[Vec<f32>]) -> f32 {
    a.iter().flatten().zip(b.iter().flatten()).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

fn upcycled(tag: &str, src: &Path, experts: usize, top_k: usize) -> PathBuf {
    let out = scratch(tag);
    make_upcycled(src, &out, MoeSpec { experts, top_k }).unwrap();
    out
}

/// A deterministic value in [-1, 1) for index `i` of the tensor named `name`.
fn noise(name: &str, i: usize) -> f32 {
    let mut s = name.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| (h ^ b as u64).wrapping_mul(0x100_0000_01b3)) ^ (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    s ^= s >> 33;
    s = s.wrapping_mul(0xff51_afd7_ed55_8ccd);
    s ^= s >> 33;
    (s >> 40) as f32 / 8_388_608.0 - 1.0
}

/// "Trained" values: every B and router set to fixed noise, so the experts differ and the router spreads the tokens.
fn pretend_trained(m: &LlamaModel, b_scale: f32, router_scale: f32) {
    for (n, v) in m.lora_vars() {
        let scale = if n.ends_with("lora_B.weight") { b_scale } else if n.ends_with(".mlp.gate.weight") { router_scale } else { continue };
        let vals: Vec<f32> = (0..v.as_tensor().elem_count()).map(|i| scale * noise(&n, i)).collect();
        v.set(&Tensor::from_vec(vals, v.as_tensor().shape(), &Device::Cpu).unwrap()).unwrap();
    }
}

fn save_adapter(m: &LlamaModel, spec: &LoraSpec, dir: &Path) {
    std::fs::create_dir_all(dir).unwrap();
    let tensors: HashMap<String, Tensor> = adapter_tensors(m).into_iter().collect();
    candle_core::safetensors::save(&tensors, dir.join("adapter_model.safetensors")).unwrap();
    std::fs::write(dir.join("adapter_config.json"), serde_json::to_string(&adapter_config(spec, &m.cfg, "base")).unwrap()).unwrap();
}

fn with_tokenizer(dir: &Path) {
    let tok = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny-llama-hf");
    for f in ["tokenizer.json", "tokenizer_config.json"] {
        std::fs::copy(tok.join(f), dir.join(f)).unwrap();
    }
}

#[test]
fn upcycle_marks_a_linked_copy_and_counts_the_parameters() {
    let out = upcycled("mark", &fixture(), 10, 1);
    let cfg: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(out.join("config.json")).unwrap()).unwrap();
    assert_eq!(cfg["prastara_moe"], serde_json::json!({"experts": 10, "top_k": 1}));
    assert!(std::fs::symlink_metadata(out.join("model.safetensors")).unwrap().file_type().is_symlink(), "the weights are linked, not copied");
    let parsed = ModelConfig::load(&out).unwrap();
    let m = parsed.moe.unwrap();
    // hidden 32, intermediate 64, 3 blocks: a feed-forward is 3·32·64 = 6144, a router 10·32
    let base = parsed.total_params();
    assert_eq!(m.params(&parsed), (base + 3 * (9 * 6144 + 320), base + 3 * 320));
    assert_eq!(MoeSpec { experts: 10, top_k: 2 }.params(&parsed).1, base + 3 * (6144 + 320));
    // upcycling twice, an MoE input, and a top-k above the experts are refused
    assert!(make_upcycled(&out, &scratch("twice"), MoeSpec { experts: 4, top_k: 1 }).is_err());
    assert!(make_upcycled(&fixture(), &scratch("k"), MoeSpec { experts: 4, top_k: 5 }).is_err());
    let moe_in = scratch("moe-in");
    std::fs::create_dir_all(&moe_in).unwrap();
    let mut c = cfg.clone();
    c.as_object_mut().unwrap().remove("prastara_moe");
    c["num_experts"] = serde_json::json!(8);
    std::fs::write(moe_in.join("config.json"), c.to_string()).unwrap();
    let e = make_upcycled(&moe_in, &scratch("moe-out"), MoeSpec { experts: 4, top_k: 1 }).unwrap_err();
    assert!(format!("{e:#}").contains("mixture-of-experts"), "{e:#}");
    // the Qwen3-8B shape of the plan: 57.2 B in all, 8.2 B active (top-1), 13.6 B (top-2)
    let q8b = ModelConfig::from_json(&serde_json::json!({"model_type": "qwen3", "vocab_size": 151936, "hidden_size": 4096, "intermediate_size": 12288,
        "num_hidden_layers": 36, "num_attention_heads": 32, "num_key_value_heads": 8, "head_dim": 128, "tie_word_embeddings": false})).unwrap();
    let (t, a) = MoeSpec { experts: 10, top_k: 1 }.params(&q8b);
    assert!((57.0e9..57.4e9).contains(&(t as f64)) && (8.1e9..8.3e9).contains(&(a as f64)), "{t} {a}");
    assert!((13.5e9..13.7e9).contains(&(MoeSpec { experts: 10, top_k: 2 }.params(&q8b).1 as f64)));
    for d in [out, moe_in] {
        let _ = std::fs::remove_dir_all(d);
    }
}

/// The parity gate: with B = 0 every expert is the base feed-forward and the top-k weights sum to 1.
#[test]
fn at_the_start_the_mixture_is_the_dense_base() {
    let base = logits(&load(&fixture()));
    for k in [1, 2, 4] {
        let out = upcycled(&format!("init{k}"), &fixture(), 4, k);
        let mut m = load(&out);
        assert_eq!(logits(&m), base, "before an adapter the upcycled copy is the base");
        m.add_lora(&LoraSpec { targets: vec![Target::Q, Target::V, Target::Up], ..LoraSpec::default() }).unwrap();
        assert!(m.lora_vars().iter().any(|(n, _)| n == &moe_router_name(2)));
        assert!(!m.lora_vars().iter().any(|(n, _)| n.ends_with("layers.0.mlp.up_proj.lora_A.weight")), "the feed-forward adapters are the experts'");
        let d = max_diff(&logits(&m), &base);
        if k == 1 {
            assert_eq!(d, 0.0, "top-1: the weight is exactly 1, the logits are the base's bit for bit");
        } else {
            assert!(d < 1e-5, "top-{k}: {d}");
        }
        let _ = std::fs::remove_dir_all(&out);
    }
}

fn grads(m: &LlamaModel, ctx: &Ctx) -> (Vec<(String, Vec<f32>)>, Option<f64>) {
    let supervised = vec![true; TOKENS.len()];
    let g = m.loss(&TOKENS, &supervised, ctx).unwrap().backward().unwrap();
    let extra = m.take_checkpoint_grads();
    let out = m.lora_vars().iter().map(|(n, v)| {
        let mut t = g.get(v).cloned().unwrap_or_else(|| v.as_tensor().zeros_like().unwrap());
        for (w, e) in &extra {
            if w.as_tensor().id() == v.as_tensor().id() {
                t = (t + e).unwrap();
            }
        }
        (n.clone(), t.flatten_all().unwrap().to_vec1().unwrap())
    }).collect();
    (out, ctx.aux_loss())
}

/// The router and every expert get gradients; the same with gradient checkpointing (dropout and a loop included);
/// the load-balancing loss is in the loss.
#[test]
fn routers_and_experts_train_and_checkpointing_gives_the_same_gradients() {
    let looped = scratch("ckpt-loop");
    make_looped(&fixture(), &looped, LoopSpec { start: 1, end: 3, passes: 2 }).unwrap();
    let hidden = load(&looped).cfg.hidden_size;
    let gates: HashMap<String, Tensor> = [("loop.gate.1".to_string(), Tensor::full(0.4f32, hidden, &Device::Cpu).unwrap())].into();
    candle_core::safetensors::save(&gates, looped.join(LOOP_FILE)).unwrap();
    for (k, src) in [(1, fixture()), (2, looped.clone())] {
        let out = upcycled(&format!("ckpt{k}"), &src, 4, k);
        let spec = LoraSpec { targets: vec![Target::Q, Target::V], dropout: 0.1, ..LoraSpec::default() };
        let run = |ckpt: bool| {
            let mut m = load(&out);
            m.add_lora(&spec).unwrap();
            pretend_trained(&m, 0.01, 1.0);
            m.set_checkpointing(ckpt);
            grads(&m, &Ctx::train(7))
        };
        let ((plain, aux_a), (ckpt, aux_b)) = (run(false), run(true));
        let aux = aux_a.expect("the load-balancing loss of an upcycled model");
        assert!(aux >= 1.0 - 1e-6 && aux < 4.0, "Switch's loss is at least 1 (E · Σ f·P with Σ f = Σ P = 1 is at its least when even): {aux}");
        assert!((aux - aux_b.unwrap()).abs() < 1e-6);
        let mut biggest = 0.0f32;
        for ((n, a), (_, b)) in plain.iter().zip(&ckpt) {
            let big = a.iter().fold(0.0f32, |m, x| m.max(x.abs()));
            let d = a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
            assert!(d <= big.max(1e-6) * 1e-4, "top-{k} {n}: checkpointing differs by {d} (largest {big})");
            biggest = biggest.max(big);
            if n.contains(".mlp.gate.weight") {
                // every router learns, top-1 too: its probability rides along as Switch's does
                assert!(big > 0.0, "top-{k}: router {n} gets a gradient");
            }
        }
        // each block's experts together, B side: the ones that got tokens learn
        for b in 0..3 {
            let learned = plain.iter().filter(|(n, g)| n.contains(&format!(".layers.{b}.mlp.experts.")) && n.ends_with("lora_B.weight") && g.iter().any(|x| *x != 0.0)).count();
            assert!(learned >= 3 * k.min(2), "top-{k} block {b}: {learned} expert adapters learned");
        }
        println!("top-{k}: aux {aux:.4}, largest gradient {biggest:.2e}");
        let _ = std::fs::remove_dir_all(&out);
    }
    let _ = std::fs::remove_dir_all(&looped);
}

#[test]
fn an_adapter_saves_and_loads_back_with_its_experts() {
    let out = upcycled("adapter", &fixture(), 4, 2);
    let spec = LoraSpec { targets: vec![Target::Q, Target::V], train_layers: Some(2), ..LoraSpec::default() };
    let mut m = load(&out);
    m.add_lora(&spec).unwrap();
    pretend_trained(&m, 0.05, 1.0);
    let want = logits(&m);
    assert!(max_diff(&want, &logits(&load(&fixture()))) > 1e-3, "the pretend-trained experts change the logits");
    // the counted parameters are the variables
    let n: usize = m.lora_vars().iter().map(|(_, v)| v.as_tensor().elem_count()).sum();
    assert_eq!(spec.trainable_params(&m.cfg), n as u64);
    let ad = scratch("adapter-out");
    save_adapter(&m, &spec, &ad);
    let c: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(ad.join("adapter_config.json")).unwrap()).unwrap();
    assert_eq!(c["prastara_moe"], serde_json::json!({"experts": 4, "top_k": 2, "rank": 16}));
    let mut fresh = load(&out);
    let got = fresh.load_adapter(&ad).unwrap();
    assert_eq!(got.train_layers, Some(2));
    assert_eq!(fresh.first_trained(), 1);
    assert_eq!(logits(&fresh), want);
    // on the dense base it does not load (its routers and experts fit nothing there)
    assert!(load(&fixture()).load_adapter(&ad).is_err());
    for d in [out, ad] {
        let _ = std::fs::remove_dir_all(d);
    }
}

/// A few plain gradient steps on the experts and the router lower the loss.
#[test]
fn a_few_steps_lower_the_loss() {
    let out = upcycled("steps", &fixture(), 4, 1);
    let mut m = load(&out);
    m.add_lora(&LoraSpec { targets: vec![Target::Q, Target::V], ..LoraSpec::default() }).unwrap();
    let supervised = vec![true; TOKENS.len()];
    let mut losses = vec![];
    for step in 0..12 {
        let ctx = Ctx::train(step);
        let loss = m.loss(&TOKENS, &supervised, &ctx).unwrap();
        losses.push(loss.to_scalar::<f32>().unwrap());
        let g = loss.backward().unwrap();
        for (_, v) in m.lora_vars() {
            if let Some(gv) = g.get(&v) {
                v.set(&(v.as_tensor() - (gv * 0.05).unwrap()).unwrap()).unwrap();
            }
        }
    }
    println!("losses {losses:?}");
    assert!(losses[11] < losses[0] - 0.05, "{losses:?}");
    let _ = std::fs::remove_dir_all(&out);
}

/// One tensor's f32 values from an F32 GGUF.
fn gguf_f32(path: &Path, name: &str) -> (Vec<u64>, Vec<f32>) {
    let g = strata_gguf::Gguf::open(path).unwrap();
    let t = g.tensors.iter().find(|t| t.name == name).unwrap_or_else(|| panic!("{name}"));
    assert_eq!(t.ggml_type, 0, "{name} is F32");
    let bytes = std::fs::read(path).unwrap();
    let start = (g.data_start + t.offset) as usize;
    let n = t.dims.iter().product::<u64>() as usize;
    (t.dims.clone(), bytes[start..start + 4 * n].chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
}

#[test]
fn export_moe_writes_a_qwen3moe_with_each_experts_own_weights() {
    let out = upcycled("export", &fixture(), 4, 2);
    with_tokenizer(&out);
    let spec = LoraSpec { targets: vec![Target::Q, Target::V], train_layers: Some(2), ..LoraSpec::default() };
    let mut m = load(&out);
    m.add_lora(&spec).unwrap();
    pretend_trained(&m, 0.05, 1.0);
    let ad = scratch("export-ad");
    save_adapter(&m, &spec, &ad);
    let gguf = scratch("export-gguf").join("moe.f32.gguf");
    let r = export_moe(&out, Some(&ad), &gguf, ExportType::F32, Some("gpt-2")).unwrap();
    assert_eq!(r.trained_blocks, 2);
    let g = strata_gguf::Gguf::open(&gguf).unwrap();
    assert_eq!(g.architecture(), Some("qwen3moe"));
    for (k, v) in [("qwen3moe.expert_count", 4), ("qwen3moe.expert_used_count", 2), ("qwen3moe.expert_feed_forward_length", 64), ("qwen3moe.block_count", 3)] {
        assert_eq!(g.u64(k), Some(v), "{k}");
    }
    // ggml order: ne [h, ff, E] for gate and up, [ff, h, E] for down, [h, E] for the router
    let dims = |n: &str| g.tensors.iter().find(|t| t.name == n).unwrap().dims.clone();
    assert_eq!(dims("blk.1.ffn_up_exps.weight"), vec![32, 64, 4]);
    assert_eq!(dims("blk.1.ffn_down_exps.weight"), vec![64, 32, 4]);
    assert_eq!(dims("blk.1.ffn_gate_inp.weight"), vec![32, 4]);
    assert_eq!(std::fs::metadata(&gguf).unwrap().len() as f64 / export_bytes(&m.cfg, ExportType::F32).unwrap() as f64 > 1.0, true);
    // expert 3's up projection in block 2 is W + s · B A; block 0 (not trained) has four copies of W and a zero router
    let w = |n: &str| -> Vec<f32> { let t = candle_core::safetensors::load(fixture().join("model.safetensors"), &Device::Cpu).unwrap(); t[n].flatten_all().unwrap().to_vec1().unwrap() };
    let a = candle_core::safetensors::load(ad.join("adapter_model.safetensors"), &Device::Cpu).unwrap();
    let stem = "base_model.model.model.layers.2.mlp.experts.3.up_proj";
    let delta = (a[&format!("{stem}.lora_B.weight")].matmul(&a[&format!("{stem}.lora_A.weight")]).unwrap() * spec.scaling()).unwrap();
    let want: Vec<f32> = w("model.layers.2.mlp.up_proj.weight").iter().zip(delta.flatten_all().unwrap().to_vec1::<f32>().unwrap()).map(|(x, d)| x + d).collect();
    let (_, up) = gguf_f32(&gguf, "blk.2.ffn_up_exps.weight");
    let d = up[3 * 64 * 32..].iter().zip(&want).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
    assert!(d < 1e-6, "expert 3: {d}");
    let (_, up0) = gguf_f32(&gguf, "blk.0.ffn_up_exps.weight");
    let base_up0 = w("model.layers.0.mlp.up_proj.weight");
    assert!((0..4).all(|e| up0[e * 2048..(e + 1) * 2048] == base_up0[..]));
    assert!(gguf_f32(&gguf, "blk.0.ffn_gate_inp.weight").1.iter().all(|x| *x == 0.0));
    // q_proj carries the shared adapter folded in
    let (_, q) = gguf_f32(&gguf, "blk.2.attn_q.weight");
    assert_ne!(q, w("model.layers.2.self_attn.q_proj.weight"));
    // quantized as written: the matrices in Q8_0, the routers and norms F32
    let q8 = gguf.with_file_name("moe.q8_0.gguf");
    export_moe(&out, Some(&ad), &q8, ExportType::from_s("q8_0").unwrap(), Some("gpt-2")).unwrap();
    let g8 = strata_gguf::Gguf::open(&q8).unwrap();
    let ty = |n: &str| strata_gguf::type_name(g8.tensors.iter().find(|t| t.name == n).unwrap().ggml_type);
    assert_eq!((ty("blk.1.ffn_down_exps.weight").as_str(), ty("blk.1.ffn_gate_inp.weight").as_str(), ty("blk.1.ffn_norm.weight").as_str()), ("Q8_0", "F32", "F32"));
    assert!(std::fs::metadata(&q8).unwrap().len() < std::fs::metadata(&gguf).unwrap().len() / 2);
    // an adapter trained for another shape, and a folder that is not upcycled, are refused
    let other = upcycled("export-other", &fixture(), 3, 1);
    assert!(export_moe(&other, Some(&ad), &gguf.with_file_name("x.gguf"), ExportType::F32, Some("gpt-2")).is_err());
    assert!(export_moe(&fixture(), None, &gguf.with_file_name("y.gguf"), ExportType::F32, Some("gpt-2")).is_err());
    for d in [out, ad, other, gguf.parent().unwrap().to_path_buf()] {
        let _ = std::fs::remove_dir_all(d);
    }
}

/// The engine's parity fixtures (crates/strata-hybrid/src/model/loop_tests.rs): tiny qwen3moe GGUFs with the
/// Workshop's logits. `moe`: 4 experts top-2 with pretend-trained experts in the top 2 blocks; `moe_loop`: a looped
/// (1..3 x3, open gates) model upcycled to 4 experts top-1, every block trained; `init`: the upcycled model with no
/// adapter (must be the dense base). Run with --ignored to write them again after a change.
#[test]
#[ignore]
fn write_the_engines_moe_fixture() {
    let fx = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny-qwen3-moe");
    std::fs::create_dir_all(&fx).unwrap();
    // plain: top-2, the top two blocks trained
    let out = upcycled("fx-moe", &fixture(), 4, 2);
    with_tokenizer(&out);
    let spec = LoraSpec { targets: vec![Target::Q, Target::V], train_layers: Some(2), ..LoraSpec::default() };
    let mut m = load(&out);
    m.add_lora(&spec).unwrap();
    pretend_trained(&m, 0.05, 2.0);
    let moe = logits(&m);
    let ad = scratch("fx-moe-ad");
    save_adapter(&m, &spec, &ad);
    export_moe(&out, Some(&ad), &fx.join("moe.f32.gguf"), ExportType::F32, Some("gpt-2")).unwrap();
    export_moe(&out, Some(&ad), &fx.join("moe.q8_0.gguf"), ExportType::from_s("q8_0").unwrap(), Some("gpt-2")).unwrap();
    export_moe(&out, None, &fx.join("init.f32.gguf"), ExportType::F32, Some("gpt-2")).unwrap();
    // looped: open gates per pass and channel, top-1, every block trained (the trained gates come from the adapter)
    let looped = scratch("fx-loop");
    make_looped(&fixture(), &looped, LoopSpec { start: 1, end: 3, passes: 3 }).unwrap();
    let lup = upcycled("fx-loop-moe", &looped, 4, 1);
    with_tokenizer(&lup);
    let lspec = LoraSpec { targets: vec![Target::Q, Target::V], ..LoraSpec::default() };
    let mut lm = load(&lup);
    lm.add_lora(&lspec).unwrap();
    pretend_trained(&lm, 0.05, 2.0);
    for (n, v) in lm.lora_vars() {
        if let Some(p) = n.strip_prefix("base_model.model.model.loop.gate.") {
            let p: f32 = p.parse().unwrap();
            let vals: Vec<f32> = (0..v.as_tensor().elem_count()).map(|c| 0.1 + 0.3 * p + 0.01 * c as f32).collect();
            v.set(&Tensor::from_vec(vals, v.as_tensor().shape(), &Device::Cpu).unwrap()).unwrap();
        }
    }
    let moe_loop = logits(&lm);
    let lad = scratch("fx-loop-ad");
    save_adapter(&lm, &lspec, &lad);
    export_moe(&lup, Some(&lad), &fx.join("moe-loop.f32.gguf"), ExportType::F32, Some("gpt-2")).unwrap();
    let base = logits(&load(&fixture()));
    assert!(max_diff(&moe, &base) > 0.1 && max_diff(&moe_loop, &base) > 0.1);
    let json = serde_json::json!({"tokens": TOKENS, "moe": moe, "moe_loop": moe_loop, "base": base,
                                  "note": "written by crates/strata-workshop/tests/upcycle.rs write_the_engines_moe_fixture"});
    std::fs::write(fx.join("logits.json"), serde_json::to_string(&json).unwrap()).unwrap();
    for d in [out, ad, looped, lup, lad] {
        let _ = std::fs::remove_dir_all(d);
    }
}
