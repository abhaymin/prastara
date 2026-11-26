//! Fine-grained upcycling (the 56B-A5B design) on the tiny fixtures: each expert is a slice of the dense
//! feed-forward's neurons (`--expert-ff`), copies of a group each with their own LoRA, top-k > 1 over different
//! groups. The parity gate for the slicing: one copy per group and every group chosen (E = G = top-k, a uniform
//! router) is the dense model, whatever the neuron permutation. Then: gradients (shared A or not, with a loop, the
//! same with checkpointing), the adapter round trip, the calibration (cluster split, router start, dense-start check),
//! and export-moe's fine-grained tensors. The engine's side is crates/strata-hybrid/src/model/loop_tests.rs (fixtures
//! written by `write_the_engines_fine_fixture`).

mod common;

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use candle_core::{Device, Tensor};
use strata_workshop::export_moe::{export_bytes, export_moe, ExportType};
use strata_workshop::model::{adapter_config, adapter_tensors, make_looped, make_upcycled, moe_group_stem, moe_router_name, BaseDtype, Ctx, LlamaModel,
                             LoopSpec, LoraSpec, ModelConfig, MoeSpec, MoeSplit, Target, LOOP_FILE, MOE_INIT_FILE};
use strata_workshop::moe_init::{self, balanced_kmeans, InitOptions};

fn fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny-qwen3-hf")
}

fn scratch(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("upcycle-fine-{tag}-{}", std::process::id()));
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

/// The tiny model's shape: hidden 32, ff 64; experts of 16 neurons are 4 groups.
fn fine(experts: usize, top_k: usize, shared_a: bool, rank: Option<usize>) -> MoeSpec {
    MoeSpec { expert_ff: Some(16), shared_a, expert_rank: rank, ..MoeSpec::full(experts, top_k) }
}

/// A deterministic value in [-1, 1) for index `i` of the tensor named `name`.
fn noise(name: &str, i: usize) -> f32 {
    let mut s = name.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| (h ^ b as u64).wrapping_mul(0x100_0000_01b3)) ^ (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    s ^= s >> 33;
    s = s.wrapping_mul(0xff51_afd7_ed55_8ccd);
    s ^= s >> 33;
    (s >> 40) as f32 / 8_388_608.0 - 1.0
}

/// Rewrite an upcycled folder's start: a seeded shuffle of every block's neurons, and routers from `router`.
fn set_init(dir: &Path, seed: u64, router: impl Fn(usize, usize, usize) -> f32) {
    let cfg = ModelConfig::load(dir).unwrap();
    let m = cfg.moe.unwrap();
    let mut t: HashMap<String, Tensor> = HashMap::new();
    for b in 0..cfg.num_hidden_layers {
        let perm: Vec<u32> = strata_workshop::finetune::permutation(seed, b as u64, cfg.intermediate_size).into_iter().map(|x| x as u32).collect();
        t.insert(format!("perm.{b}"), Tensor::from_vec(perm, (cfg.intermediate_size,), &Device::Cpu).unwrap());
        let r: Vec<f32> = (0..m.experts * cfg.hidden_size).map(|i| router(b, i / cfg.hidden_size, i % cfg.hidden_size)).collect();
        t.insert(format!("router.{b}"), Tensor::from_vec(r, (m.experts, cfg.hidden_size), &Device::Cpu).unwrap());
    }
    candle_core::safetensors::save(&t, dir.join(MOE_INIT_FILE)).unwrap();
}

fn upcycled(tag: &str, src: &Path, m: MoeSpec) -> PathBuf {
    let out = scratch(tag);
    make_upcycled(src, &out, m).unwrap();
    out
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

fn q8b() -> ModelConfig {
    ModelConfig::from_json(&serde_json::json!({"model_type": "qwen3", "vocab_size": 151936, "hidden_size": 4096, "intermediate_size": 12288,
        "num_hidden_layers": 36, "num_attention_heads": 32, "num_key_value_heads": 8, "head_dim": 128, "tie_word_embeddings": false})).unwrap()
}

#[test]
fn the_56b_a5b_shape_counts_and_checks() {
    let mut cfg = q8b();
    let m = MoeSpec { expert_ff: Some(1024), ..MoeSpec::full(120, 5) };
    m.check_with(&cfg).unwrap();
    assert_eq!((m.groups(&cfg), m.ff_e(&cfg), m.out_scale(&cfg)), (12, 1024, 5.0));
    let (total, active) = m.params(&cfg);
    println!("56B-A5B: {total} total, {active} active");
    assert!((57.0e9..57.3e9).contains(&(total as f64)) && (4.95e9..5.1e9).contains(&(active as f64)), "{total} {active}");
    // the full-width design is unchanged: 57.2 B, 8.2 B active
    let (t, a) = MoeSpec::full(10, 1).params(&cfg);
    assert!((57.0e9..57.4e9).contains(&(t as f64)) && (8.1e9..8.3e9).contains(&(a as f64)));
    // the experts' trainable parameters per block at rank 8: 120 · 3 · 8 · (4096 + 1024) + the router; shared A
    assert_eq!(MoeSpec { expert_rank: Some(8), ..m }.block_trainable(&cfg, 16), 120 * 3 * 8 * (4096 + 1024) + 120 * 4096);
    assert_eq!(MoeSpec { expert_rank: Some(8), shared_a: true, ..m }.block_trainable(&cfg, 16),
               12 * 8 * (2 * 4096 + 1024) + 120 * 8 * (2 * 1024 + 4096) + 120 * 4096);
    // over 36 blocks at rank 8 that is the plan's ~531 M (8.5 GB with gradients and AdamW in f32)
    assert!((530e6..535e6).contains(&(36.0 * (120 * 3 * 8 * (4096 + 1024)) as f64)));
    // the shapes refused: ff not divided, experts not whole copies, top-k above the groups
    for bad in [MoeSpec { expert_ff: Some(1000), ..m }, MoeSpec { experts: 100, ..m }, MoeSpec { top_k: 13, expert_ff: Some(1024), ..MoeSpec::full(120, 13) }] {
        assert!(bad.check_with(&cfg).is_err(), "{bad:?}");
    }
    // the config round trip
    let j = MoeSpec { split: MoeSplit::Cluster, expert_rank: Some(4), shared_a: true, ..m }.to_json();
    assert_eq!(j, serde_json::json!({"experts": 120, "top_k": 5, "expert_ff": 1024, "split": "cluster", "expert_rank": 4, "shared_a": true}));
    assert_eq!(MoeSpec::from_json(&j).unwrap().unwrap(), MoeSpec { split: MoeSplit::Cluster, expert_rank: Some(4), shared_a: true, ..m });
    assert_eq!(MoeSpec::full(10, 1).to_json(), serde_json::json!({"experts": 10, "top_k": 1}), "a full-width config is written as before");
    // the export: Q4_K about 32 GB
    cfg.moe = Some(m);
    let q4 = export_bytes(&cfg, ExportType::from_s("q4_k").unwrap()).unwrap();
    assert!((31e9..33.5e9).contains(&(q4 as f64)), "{q4}");
    // the memory schemes on two 16 GB cards: the default fits with the margin, rank 8 unshared at 1,024 does not
    let schemes = strata_workshop::profile::moe_schemes(&cfg, 2, strata_workshop::profile::DEFAULT_CARD_BYTES);
    let d = strata_workshop::profile::moe_default_scheme(&cfg, 2, strata_workshop::profile::DEFAULT_CARD_BYTES).unwrap();
    for s in &schemes {
        println!("rank {} shared {} len {}: {} trainable, {:.2} GB a card{}", s.expert_rank, s.shared_a, s.max_len, s.trainable_params, s.card_bytes as f64 / 1e9,
                 if s.fits { " fits" } else { "" });
    }
    println!("default: {d:?}");
    assert!(d.fits && d.card_bytes + strata_workshop::profile::MOE_CARD_MARGIN_BYTES <= strata_workshop::profile::DEFAULT_CARD_BYTES);
    assert!(!schemes.iter().find(|s| s.expert_rank == 8 && !s.shared_a && s.max_len == 1024).unwrap().fits);
}

/// The slicing's parity gate: E = G groups, one copy each, top-k = G and a uniform router: the out-scale G times the
/// even weights 1/G sum every group once, which is the dense feed-forward for any neuron permutation.
#[test]
fn every_group_once_is_the_dense_model() {
    let dense = logits(&load(&fixture()));
    let out = upcycled("parity", &fixture(), fine(4, 4, false, None));
    set_init(&out, 7, |_, _, _| 0.0);
    let mut m = load(&out);
    let d0 = max_diff(&logits(&m), &dense);
    // run dense (the permuted weights are the same function) and with adapters at B = 0
    let dd = max_diff(&m.forward(&TOKENS, &Ctx::dense_inference()).unwrap().to_vec2().unwrap(), &dense);
    m.add_lora(&LoraSpec { targets: vec![Target::Q, Target::V], ..LoraSpec::default() }).unwrap();
    let d1 = max_diff(&logits(&m), &dense);
    println!("E = G = top-k = 4: {d0:.2e} as loaded, {dd:.2e} run dense, {d1:.2e} with fresh adapters");
    assert!(d0 < 1e-5 && dd < 1e-5 && d1 < 1e-5, "{d0} {dd} {d1}");
    // a top-k below the groups is NOT the dense model (MoEfication's approximation)
    let part = upcycled("parity-k2", &fixture(), fine(4, 2, false, None));
    set_init(&part, 7, |_, _, _| 0.0);
    let dp = max_diff(&logits(&load(&part)), &dense);
    println!("top-2 of 4 groups: {dp:.3e} from the dense logits");
    assert!(dp > 1e-3);
    for d in [out, part] {
        let _ = std::fs::remove_dir_all(d);
    }
}

fn grads(m: &LlamaModel, ctx: &Ctx) -> (Vec<(String, Vec<f32>)>, Option<f64>, Option<f64>) {
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
    (out, ctx.aux_loss(), ctx.z_loss())
}

/// Routers, the experts' B and (shared or not) A get gradients; checkpointing (with dropout and a loop) gives the
/// same; the load-balancing loss and the z-loss are in the loss.
#[test]
fn fine_experts_train_and_checkpointing_gives_the_same_gradients() {
    let looped = scratch("ckpt-loop");
    make_looped(&fixture(), &looped, LoopSpec { start: 1, end: 3, passes: 2 }).unwrap();
    let hidden = load(&looped).cfg.hidden_size;
    let gates: HashMap<String, Tensor> = [("loop.gate.1".to_string(), Tensor::full(0.4f32, hidden, &Device::Cpu).unwrap())].into();
    candle_core::safetensors::save(&gates, looped.join(LOOP_FILE)).unwrap();
    for (tag, src, m) in [("plain", fixture(), fine(8, 2, false, None)), ("shared-loop", looped.clone(), fine(12, 3, true, Some(4)))] {
        let out = upcycled(&format!("ckpt-{tag}"), &src, m);
        let spec = LoraSpec { targets: vec![Target::Q, Target::V], dropout: 0.1, ..LoraSpec::default() };
        let run = |ckpt: bool| {
            let mut lm = load(&out);
            lm.add_lora(&spec).unwrap();
            pretend_trained(&lm, 0.01, 1.0);
            lm.set_checkpointing(ckpt);
            grads(&lm, &Ctx::train(7))
        };
        let ((plain, aux_a, z_a), (ckpt, aux_b, z_b)) = (run(false), run(true));
        let (aux, z) = (aux_a.expect("the load-balancing loss"), z_a.expect("the z-loss of a fine-grained model"));
        assert!(aux >= 1.0 - 1e-6 && aux < m.experts as f64, "{tag}: Switch's loss {aux}");
        assert!(z > 0.0 && (aux - aux_b.unwrap()).abs() < 1e-6 && (z - z_b.unwrap()).abs() < 1e-6);
        let mut seen_a = false;
        for ((n, a), (_, b)) in plain.iter().zip(&ckpt) {
            let big = a.iter().fold(0.0f32, |m, x| m.max(x.abs()));
            let d = a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
            assert!(d <= big.max(1e-6) * 1e-4, "{tag} {n}: checkpointing differs by {d} (largest {big})");
            if n.contains(".mlp.gate.weight") {
                assert!(big > 0.0, "{tag}: router {n} gets a gradient");
            }
            if n.contains(".mlp.groups.") || n.contains(".mlp.experts.") && n.ends_with("lora_A.weight") {
                seen_a |= big > 0.0;
            }
        }
        assert!(seen_a, "{tag}: the experts' A learn");
        if m.shared_a {
            assert!(plain.iter().any(|(n, _)| n.contains(&moe_group_stem(2, 3, Target::Down))), "a shared A is named by its group");
            assert!(!plain.iter().any(|(n, _)| n.contains(".mlp.experts.") && n.ends_with("lora_A.weight")));
        }
        for b in 0..3 {
            let learned = plain.iter().filter(|(n, g)| n.contains(&format!(".layers.{b}.mlp.experts.")) && n.ends_with("lora_B.weight") && g.iter().any(|x| *x != 0.0)).count();
            assert!(learned >= 3 * m.top_k, "{tag} block {b}: {learned} expert adapters learned");
        }
        println!("{tag}: aux {aux:.4}, z {z:.4}");
        let _ = std::fs::remove_dir_all(&out);
    }
    let _ = std::fs::remove_dir_all(&looped);
}

#[test]
fn a_fine_adapter_saves_and_loads_back() {
    for shared in [false, true] {
        let out = upcycled(&format!("adapter-{shared}"), &fixture(), fine(8, 2, shared, Some(4)));
        let spec = LoraSpec { targets: vec![Target::Q, Target::V], train_layers: Some(2), ..LoraSpec::default() };
        let mut m = load(&out);
        let start = logits(&m);
        m.add_lora(&spec).unwrap();
        assert_eq!(logits(&m), start, "fresh adapters (B = 0) start from the upcycled start");
        assert!(!m.lora_vars().iter().any(|(n, _)| n == &moe_router_name(0)), "block 0 is below the trained ones: its router stays at its start");
        pretend_trained(&m, 0.05, 1.0);
        let want = logits(&m);
        assert!(max_diff(&want, &start) > 1e-3);
        let n: usize = m.lora_vars().iter().map(|(_, v)| v.as_tensor().elem_count()).sum();
        assert_eq!(spec.trainable_params(&m.cfg), n as u64, "shared {shared}: the counted parameters are the variables");
        let ad = scratch(&format!("adapter-out-{shared}"));
        save_adapter(&m, &spec, &ad);
        let c: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(ad.join("adapter_config.json")).unwrap()).unwrap();
        assert_eq!(c["prastara_moe"]["expert_ff"], 16);
        assert_eq!(c["prastara_moe"]["expert_rank"], 4);
        let mut fresh = load(&out);
        fresh.load_adapter(&ad).unwrap();
        assert_eq!(fresh.first_trained(), 1);
        assert_eq!(logits(&fresh), want);
        // the router-only scheme: no expert adapters at all
        let mut r0 = load(&upcycled(&format!("r0-{shared}"), &fixture(), fine(8, 2, false, Some(0))));
        r0.add_lora(&spec).unwrap();
        assert!(r0.lora_vars().iter().all(|(n, _)| !n.contains(".mlp.experts.")));
        assert!(r0.lora_vars().iter().any(|(n, _)| n.contains(".mlp.gate.weight")));
        for d in [out, ad, scratch(&format!("r0-{shared}"))] {
            let _ = std::fs::remove_dir_all(d);
        }
    }
}

/// Plain gradient steps on the fine experts and routers lower the loss.
#[test]
fn a_few_fine_steps_lower_the_loss() {
    let out = upcycled("steps", &fixture(), fine(8, 2, true, Some(8)));
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

#[test]
fn balanced_kmeans_finds_planted_groups_of_equal_size() {
    // 24 rows near 3 directions, shuffled: three groups of 8, each one direction
    let (n, d) = (24usize, 6usize);
    let order = strata_workshop::finetune::permutation(3, 1, n);
    let mut v = vec![0f32; n * d];
    for (row, &i) in order.iter().enumerate() {
        let c = i % 3;
        for j in 0..d {
            v[row * d + j] = if j / 2 == c { 1.0 } else { 0.0 } + 0.05 * noise("k", row * d + j);
        }
    }
    let t = Tensor::from_vec(v, (n, d), &Device::Cpu).unwrap();
    let t = t.broadcast_div(&t.sqr().unwrap().sum_keepdim(1).unwrap().sqrt().unwrap()).unwrap();
    let a = balanced_kmeans(&t, 3, 11, 20).unwrap();
    for g in 0..3 {
        assert_eq!(a.iter().filter(|x| **x == g).count(), 8);
    }
    for (row, &i) in order.iter().enumerate() {
        for (row2, &i2) in order.iter().enumerate() {
            assert_eq!(a[row] == a[row2], i % 3 == i2 % 3, "rows {row} and {row2}");
        }
    }
    assert_eq!(a, balanced_kmeans(&t, 3, 11, 20).unwrap(), "deterministic");
}

/// The calibrated upcycle on a tiny chat model: clustered groups (a permutation, every group exactly ff_e), the
/// router's start (tokens pick different groups), and the dense-start check (how far the start is from the dense
/// model, measured, not assumed).
#[test]
fn a_calibrated_upcycle_groups_routes_and_measures_its_start() {
    let root = common::temp_dir("fine-calib");
    let (model, data) = (root.join("model"), root.join("chat.jsonl"));
    common::chat_model_sized(&model, 3, 32, 64);
    common::write_jsonl(&data, &common::chat_rows(64));
    let out = root.join("up");
    let m = MoeSpec { split: MoeSplit::Cluster, ..fine(8, 2, true, Some(4)) };
    let opts = InitOptions { calib: Some(data.clone()), calib_tokens: 600, max_len: 64, check_tokens: 200, base_dtype: BaseDtype::F32, ..Default::default() };
    let mut log = Vec::new();
    let r = moe_init::upcycle(&model, &out, m, &opts, &mut |l| log.push(l)).unwrap();
    for l in r.lines() {
        println!("{l}");
    }
    assert!(r.calibrated && r.calib_tokens >= 600 && r.blocks.len() == 3);
    let init = candle_core::safetensors::load(out.join(MOE_INIT_FILE), &Device::Cpu).unwrap();
    for b in 0..3 {
        let mut p = init[&format!("perm.{b}")].to_vec1::<u32>().unwrap();
        p.sort();
        assert_eq!(p, (0..64).collect::<Vec<u32>>(), "block {b}: a permutation of the neurons");
    }
    for s in &r.blocks {
        assert!(s.cluster_similarity >= s.contiguous_similarity - 1e-6, "block {}: clustering is at least as tight as the contiguous split", s.block);
        assert!(s.distinct > 0.9, "block {}: {:.3} of tokens pick k different groups", s.block, s.distinct);
        assert!(s.coverage_router > s.coverage_random - 0.05 && s.coverage_best >= s.coverage_router - 1e-9);
    }
    let (dense, up, n) = r.check.unwrap();
    println!("tiny dense-start gap: dense {dense:.4}, upcycled {up:.4} ({:+.4}) on {n} tokens", up - dense);
    assert!(n >= 200 && dense.is_finite() && up.is_finite());
    // the start, loaded, is the mixture (not the dense model), and the dense run of the same folder is the base
    let tokens: Vec<u32> = (0..12).map(|i| (i * 37 + 11) % 259).collect();
    let base = LlamaModel::load(&model, BaseDtype::F32, &Device::Cpu).unwrap().forward(&tokens, &Ctx::eval()).unwrap().to_vec2::<f32>().unwrap();
    let u = LlamaModel::load(&out, BaseDtype::F32, &Device::Cpu).unwrap();
    let ud = u.forward(&tokens, &Ctx::dense_inference()).unwrap().to_vec2::<f32>().unwrap();
    assert!(max_diff(&ud, &base) < 1e-5, "the permuted weights run dense are the base");
    assert!(max_diff(&u.forward(&tokens, &Ctx::eval()).unwrap().to_vec2::<f32>().unwrap(), &base) > 1e-4);
    // the same seed, the same start
    let again = root.join("up2");
    moe_init::upcycle(&model, &again, m, &InitOptions { check_tokens: 0, ..opts.clone() }, &mut |_| {}).unwrap();
    let init2 = candle_core::safetensors::load(again.join(MOE_INIT_FILE), &Device::Cpu).unwrap();
    for k in ["perm.1", "router.2"] {
        assert_eq!(init[k].flatten_all().unwrap().to_vec1::<f32>().ok(), init2[k].flatten_all().unwrap().to_vec1::<f32>().ok(), "{k}");
    }
    assert_eq!(init["perm.1"].to_vec1::<u32>().unwrap(), init2["perm.1"].to_vec1::<u32>().unwrap());
    // a cluster split without text is refused; contiguous works
    assert!(moe_init::upcycle(&model, &root.join("up3"), m, &InitOptions::default(), &mut |_| {}).is_err());
    moe_init::upcycle(&model, &root.join("up4"), MoeSpec { split: MoeSplit::Contiguous, ..m }, &InitOptions::default(), &mut |_| {}).unwrap();
    assert!(root.join("up4").join(MOE_INIT_FILE).exists());
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
fn export_moe_writes_each_fine_experts_slice() {
    let m = fine(8, 2, true, Some(4));
    let out = upcycled("export", &fixture(), m);
    set_init(&out, 5, |b, e, j| 0.3 * noise(&format!("r{b}"), e * 32 + j));
    with_tokenizer(&out);
    let spec = LoraSpec { targets: vec![Target::Q, Target::V], train_layers: Some(2), ..LoraSpec::default() };
    let mut lm = load(&out);
    lm.add_lora(&spec).unwrap();
    pretend_trained(&lm, 0.05, 2.0);
    let ad = scratch("export-ad");
    save_adapter(&lm, &spec, &ad);
    let gguf = scratch("export-gguf").join("fine.f32.gguf");
    let r = export_moe(&out, Some(&ad), &gguf, ExportType::F32, Some("gpt-2")).unwrap();
    assert_eq!(r.trained_blocks, 2);
    let g = strata_gguf::Gguf::open(&gguf).unwrap();
    for (k, v) in [("qwen3moe.expert_count", 8), ("qwen3moe.expert_used_count", 2), ("qwen3moe.expert_feed_forward_length", 16)] {
        assert_eq!(g.u64(k), Some(v), "{k}");
    }
    let dims = |n: &str| g.tensors.iter().find(|t| t.name == n).unwrap().dims.clone();
    assert_eq!(dims("blk.1.ffn_gate_exps.weight"), vec![32, 16, 8]);
    assert_eq!(dims("blk.1.ffn_down_exps.weight"), vec![16, 32, 8]);
    assert!(std::fs::metadata(&gguf).unwrap().len() as f64 / export_bytes(&lm.cfg, ExportType::F32).unwrap() as f64 > 1.0);
    // expert 5 (group 1, copy 1) of block 2: gate rows perm[16..32] + s·B_5·A_group1; down columns perm[16..32] + its LoRA, times top-k
    let base = candle_core::safetensors::load(fixture().join("model.safetensors"), &Device::Cpu).unwrap();
    let a = candle_core::safetensors::load(ad.join("adapter_model.safetensors"), &Device::Cpu).unwrap();
    let init = candle_core::safetensors::load(out.join(MOE_INIT_FILE), &Device::Cpu).unwrap();
    let perm = init["perm.2"].to_vec1::<u32>().unwrap();
    let idx = Tensor::from_vec(perm[16..32].to_vec(), (16,), &Device::Cpu).unwrap();
    let s = spec.scaling_at(4);
    let delta = |t: Target| (a[&format!("base_model.model.model.layers.2.mlp.experts.5.{}.lora_B.weight", t.as_str())]
        .matmul(&a[&format!("{}.lora_A.weight", moe_group_stem(2, 1, t))]).unwrap() * s).unwrap();
    let gate = (base["model.layers.2.mlp.gate_proj.weight"].index_select(&idx, 0).unwrap() + delta(Target::Gate)).unwrap();
    let down = ((base["model.layers.2.mlp.down_proj.weight"].index_select(&idx, 1).unwrap() + delta(Target::Down)).unwrap() * 2.0).unwrap();
    let close = |got: &[f32], want: &Tensor| got.iter().zip(want.flatten_all().unwrap().to_vec1::<f32>().unwrap()).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
    let (_, ge) = gguf_f32(&gguf, "blk.2.ffn_gate_exps.weight");
    let (_, de) = gguf_f32(&gguf, "blk.2.ffn_down_exps.weight");
    assert!(close(&ge[5 * 512..6 * 512], &gate) < 1e-6 && close(&de[5 * 512..6 * 512], &down) < 1e-6);
    // block 0 is not trained: its router is the start's, its experts the slices (times top-k for down)
    let (_, r0) = gguf_f32(&gguf, "blk.0.ffn_gate_inp.weight");
    assert_eq!(r0, init["router.0"].flatten_all().unwrap().to_vec1::<f32>().unwrap());
    let p0 = init["perm.0"].to_vec1::<u32>().unwrap();
    let up0 = base["model.layers.0.mlp.up_proj.weight"].index_select(&Tensor::from_vec(p0[48..64].to_vec(), (16,), &Device::Cpu).unwrap(), 0).unwrap();
    let (_, ue) = gguf_f32(&gguf, "blk.0.ffn_up_exps.weight");
    assert!(close(&ue[7 * 512..8 * 512], &up0) == 0.0, "expert 7 of block 0 is group 3 as it is");
    for d in [out, ad, gguf.parent().unwrap().to_path_buf()] {
        let _ = std::fs::remove_dir_all(d);
    }
}

/// The engine's fine-grained parity fixtures: `fine.f32.gguf` (8 experts of 16 neurons = 4 groups x 2 copies, top-2,
/// shuffled neurons, the top two blocks trained, block 0 routed by its start), `fine-loop.f32.gguf` (a looped model,
/// 1..3 x 3 with open gates, 12 experts = 4 groups x 3 copies, top-3, A shared per group, every block trained),
/// `fine-all.f32.gguf` (E = G = top-k = 4, a uniform router: the dense base), and the Workshop's logits in
/// `fine-logits.json`. Run with --ignored to write them again after a change.
#[test]
#[ignore]
fn write_the_engines_fine_fixture() {
    let fx = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny-qwen3-moe");
    std::fs::create_dir_all(&fx).unwrap();
    let out = upcycled("fx-fine", &fixture(), fine(8, 2, false, Some(8)));
    set_init(&out, 5, |b, e, j| 0.5 * noise(&format!("r{b}"), e * 32 + j));
    with_tokenizer(&out);
    let spec = LoraSpec { targets: vec![Target::Q, Target::V], train_layers: Some(2), ..LoraSpec::default() };
    let mut m = load(&out);
    m.add_lora(&spec).unwrap();
    pretend_trained(&m, 0.05, 2.0);
    let fine_logits = logits(&m);
    let ad = scratch("fx-fine-ad");
    save_adapter(&m, &spec, &ad);
    export_moe(&out, Some(&ad), &fx.join("fine.f32.gguf"), ExportType::F32, Some("gpt-2")).unwrap();
    // looped, shared A, top-3 of 12
    let looped = scratch("fx-fine-loop");
    make_looped(&fixture(), &looped, LoopSpec { start: 1, end: 3, passes: 3 }).unwrap();
    let lup = upcycled("fx-fine-loop-moe", &looped, fine(12, 3, true, Some(4)));
    set_init(&lup, 9, |b, e, j| 0.5 * noise(&format!("loop{b}"), e * 32 + j));
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
    let fine_loop = logits(&lm);
    let lad = scratch("fx-fine-loop-ad");
    save_adapter(&lm, &lspec, &lad);
    export_moe(&lup, Some(&lad), &fx.join("fine-loop.f32.gguf"), ExportType::F32, Some("gpt-2")).unwrap();
    // every group once: the dense base
    let all = upcycled("fx-fine-all", &fixture(), fine(4, 4, false, None));
    set_init(&all, 3, |_, _, _| 0.0);
    with_tokenizer(&all);
    let fine_all = logits(&load(&all));
    export_moe(&all, None, &fx.join("fine-all.f32.gguf"), ExportType::F32, Some("gpt-2")).unwrap();
    let base = logits(&load(&fixture()));
    assert!(max_diff(&fine_logits, &base) > 0.1 && max_diff(&fine_loop, &base) > 0.1 && max_diff(&fine_all, &base) < 1e-5);
    let json = serde_json::json!({"tokens": TOKENS, "fine": fine_logits, "fine_loop": fine_loop, "fine_all": fine_all, "base": base,
                                  "note": "written by crates/strata-workshop/tests/upcycle_fine.rs write_the_engines_fine_fixture"});
    std::fs::write(fx.join("fine-logits.json"), serde_json::to_string(&json).unwrap()).unwrap();
    for d in [out, ad, looped, lup, lad, all] {
        let _ = std::fs::remove_dir_all(d);
    }
}
