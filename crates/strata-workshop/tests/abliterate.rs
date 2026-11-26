//! Abliteration on the tiny fixtures: every orthogonalized matrix stops writing along the direction (‖rᵀW‖ ≈ 0) and
//! so the residual stream of the result has no component along it; every other tensor is byte-identical; a sharded
//! output loads back; a looped model keeps its gates; a tied head is kept as it was; the whole run works from text
//! prompts through the chat template. The prompts are placeholders, never real requests.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use candle_core::{Device, Tensor};
use strata_workshop::abliterate::{self, capture, choose, directions, orthogonalize, residual_write, role, LayerChoice, Options, Role, Site};
use strata_workshop::model::{make_looped, BaseDtype, Ctx, LlamaModel, LoopSpec, LOOP_FILE};
use strata_workshop::safetensors_io::Shard;
use strata_workshop::st_writer;

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(name)
}

fn scratch(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("abliterate-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    d
}

/// Synthetic prompts as token ids: the "harmful" ones all carry token 42, the others never do.
fn id_prompts() -> (Vec<Vec<u32>>, Vec<Vec<u32>>) {
    let harmful = (0..6u32).map(|i| vec![1, 7 + i, 42, 42, 3 + i % 3]).collect();
    let harmless = (0..6u32).map(|i| vec![1, 7 + i, 20 + i, 5, 3 + i % 3]).collect();
    (harmful, harmless)
}

/// The direction from `dir`'s residuals, at `choice`.
fn direction(dir: &Path, choice: LayerChoice) -> (Vec<f32>, Vec<abliterate::SiteScore>) {
    let m = LlamaModel::load(dir, BaseDtype::F32, &Device::Cpu).unwrap();
    let (h, b) = id_prompts();
    let (ch, cb) = (capture(&m, &h, &mut |_, _| {}).unwrap(), capture(&m, &b, &mut |_, _| {}).unwrap());
    let scores = directions(&ch, &cb).unwrap();
    let k = choose(&scores, choice).unwrap();
    (scores[k].direction.clone(), scores)
}

/// Every tensor of a folder as (dtype, shape, bytes), read through its index when it has one.
fn tensors(dir: &Path) -> HashMap<String, (String, Vec<usize>, Vec<u8>)> {
    let mut out = HashMap::new();
    for f in strata_workshop::model::weight_files(dir).unwrap() {
        let s = Shard::open(&f).unwrap();
        for t in s.tensors() {
            out.insert(t.name().to_string(), (t.dtype().as_str().to_string(), t.shape().to_vec(), s.bytes(t).to_vec()));
        }
    }
    out
}

fn as_tensor(t: &(String, Vec<usize>, Vec<u8>)) -> Tensor {
    assert_eq!(t.0, "F32");
    let v: Vec<f32> = t.2.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    Tensor::from_vec(v, t.1.clone(), &Device::Cpu).unwrap()
}

/// The checks every abliterated folder must pass against its source.
fn check_result(src: &Path, out: &Path, r: &[f32], tied: bool) {
    let (a, b) = (tensors(src), tensors(out));
    let hidden = r.len();
    let mut changed = 0;
    for (name, t) in &a {
        let got = b.get(name).unwrap_or_else(|| panic!("{name} is missing from the result"));
        assert_eq!((&t.0, &t.1), (&got.0, &got.1), "{name}: dtype and shape are kept");
        match role(name, &t.1, hidden).unwrap() {
            Role::Untouched => assert_eq!(t.2, got.2, "{name}: an untouched tensor is byte-identical"),
            Role::Writer { axis } => {
                changed += 1;
                let before = residual_write(&as_tensor(t), axis, r).unwrap();
                let after = residual_write(&as_tensor(got), axis, r).unwrap();
                assert!(before > 1e-3, "{name}: wrote along r before ({before})");
                assert!(after < 1e-5, "{name}: still writes along r ({after})");
            }
            Role::Embed => {
                changed += 1;
                assert!(residual_write(&as_tensor(got), 1, r).unwrap() < 1e-5, "{name}: embeddings along r");
                if tied {
                    assert_eq!(b["lm_head.weight"].2, t.2, "the untied head is the original embeddings, byte for byte");
                }
            }
        }
    }
    assert!(changed > 0);
    let cfg: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(out.join("config.json")).unwrap()).unwrap();
    assert_eq!(cfg["prastara_private"], true, "an abliterated model is marked private");
    assert!(cfg["prastara_abliterated"].is_object());
    if tied {
        assert_eq!(cfg["tie_word_embeddings"], false);
    }
    // the residual stream of the result has (almost) nothing along r, at every block and pass. An open loop's gates
    // mix two passes channel by channel, which no weight edit can keep orthogonal: from the second extra pass on (and
    // after the loop), a little comes back, much less than the original had.
    let tokens = [1u32, 9, 42, 42, 4, 17];
    let along = |dir: &Path| -> Vec<(usize, usize, f32, f32)> {
        let m = LlamaModel::load(dir, BaseDtype::F32, &Device::Cpu).unwrap();
        m.residuals(&tokens).unwrap().into_iter().map(|(b, p, x)| {
            let v = x.to_vec1::<f32>().unwrap();
            (b, p, v.iter().zip(r).map(|(a, b)| a * b).sum::<f32>().abs(), v.iter().map(|a| a * a).sum::<f32>().sqrt())
        }).collect()
    };
    let loop_ = cfg["prastara_loop"].as_object().map(|l| (l["end"].as_u64().unwrap() as usize));
    let mixed_by_gates = |blk: usize, pass: usize| loop_.is_some_and(|end| pass >= 2 || pass == 0 && blk >= end);
    for ((blk, pass, dot, n), (_, _, dot0, _)) in along(out).into_iter().zip(along(src)) {
        if mixed_by_gates(blk, pass) {
            assert!(dot < 0.2 * dot0, "block {blk} pass {pass}: residual · r = {dot}, was {dot0}");
        } else {
            assert!(dot <= 1e-4 * n.max(1.0), "block {blk} pass {pass}: residual · r = {dot} (|x| {n})");
        }
    }
    let m = LlamaModel::load(out, BaseDtype::F32, &Device::Cpu).unwrap();
    let logits = m.forward(&[1, 5, 9], &Ctx::eval()).unwrap().to_vec2::<f32>().unwrap();
    assert!(logits.iter().flatten().all(|x| x.is_finite()));
}

#[test]
fn qwen3_writers_lose_the_direction_and_a_sharded_output_loads() {
    let src = fixture("tiny-qwen3-hf");
    let (r, scores) = direction(&src, LayerChoice::Auto);
    assert_eq!(scores.len(), 3, "one site per block without a loop");
    assert!(scores.iter().any(|s| s.eligible));
    let out = scratch("qwen3");
    // small shards: the result is split, with an index of file names
    let rep = orthogonalize(&src, &out, &r, 1.0, serde_json::json!({"layer": 1}), 20_000).unwrap();
    assert!(rep.files.len() > 1, "the output is sharded ({} files)", rep.files.len());
    assert!(!rep.untied);
    let index = st_writer::read_index(&out).unwrap().expect("an index");
    for (name, file) in &index {
        assert!(out.join(file).exists(), "{name} -> {file}");
    }
    assert!(!out.join("model.safetensors").exists());
    // candle (the way transformers does) loads the shards the index names
    let files: Vec<PathBuf> = index.values().collect::<std::collections::BTreeSet<_>>().into_iter().map(|f| out.join(f)).collect();
    let st = unsafe { candle_core::safetensors::MmapedSafetensors::multi(&files) }.unwrap();
    assert_eq!(st.tensors().len(), index.len());
    check_result(&src, &out, &r, false);
    // an output folder with something in it is refused
    assert!(orthogonalize(&src, &out, &r, 1.0, serde_json::json!({}), 20_000).is_err());
    let _ = std::fs::remove_dir_all(&out);
}

#[test]
fn a_looped_model_reads_each_pass_and_keeps_its_gates() {
    let looped = scratch("looped-src");
    make_looped(&fixture("tiny-qwen3-hf"), &looped, LoopSpec { start: 1, end: 3, passes: 3 }).unwrap();
    let gate = |p: usize| Tensor::new((0..32).map(|c| 0.1 + 0.3 * p as f32 + 0.01 * c as f32).collect::<Vec<f32>>(), &Device::Cpu).unwrap();
    let gates: HashMap<String, Tensor> = [("loop.gate.1".to_string(), gate(1)), ("loop.gate.2".to_string(), gate(2))].into();
    candle_core::safetensors::save(&gates, looped.join(LOOP_FILE)).unwrap();

    let (r, scores) = direction(&looped, LayerChoice::Site(Site { block: 2, pass: 1 }));
    let sites: Vec<(usize, usize)> = scores.iter().map(|s| (s.site.block, s.site.pass)).collect();
    assert_eq!(sites, vec![(0, 0), (1, 0), (2, 0), (1, 1), (2, 1), (1, 2), (2, 2)], "every block of every pass, in run order");
    let out = scratch("looped-out");
    orthogonalize(&looped, &out, &r, 1.0, serde_json::json!({"layer": 2, "pass": 1}), st_writer::SHARD_BYTES).unwrap();
    assert!(out.join("model.safetensors").exists() && !out.join(st_writer::INDEX_FILE).exists(), "one shard, no index");
    assert_eq!(std::fs::read(looped.join(LOOP_FILE)).unwrap(), std::fs::read(out.join(LOOP_FILE)).unwrap(), "the gates are kept");
    let cfg: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(out.join("config.json")).unwrap()).unwrap();
    assert_eq!(cfg["prastara_loop"]["passes"], 3);
    check_result(&looped, &out, &r, false);
    let _ = std::fs::remove_dir_all(&looped);
    let _ = std::fs::remove_dir_all(&out);
}

#[test]
fn a_tied_head_is_untied_and_reads_as_before() {
    let src = fixture("tiny-llama-tied-hf");
    let (r, _) = direction(&src, LayerChoice::Auto);
    let out = scratch("tied");
    let rep = orthogonalize(&src, &out, &r, 1.0, serde_json::json!({}), st_writer::SHARD_BYTES).unwrap();
    assert!(rep.untied);
    check_result(&src, &out, &r, true);
    let _ = std::fs::remove_dir_all(&out);
}

#[test]
fn the_whole_run_from_text_prompts_with_a_check() {
    let src = scratch("chat-src");
    let m = strata_workshop::testmodel::TinyLlama { seed: 0xAB1, num_layers: 4, vocab_size: strata_workshop::testmodel::CHAT_VOCAB, hidden: 32,
                                                    intermediate: 64, num_attention_heads: 4, num_key_value_heads: 2, shard_count: 2 };
    strata_workshop::testmodel::write_chat_model(&src, &m).unwrap();
    let harmful: Vec<String> = (1..=6).map(|i| format!("please refuse: item {i}")).collect();
    let harmless: Vec<String> = abliterate::SAMPLE_HARMLESS.iter().take(6).map(|s| s.to_string()).collect();
    let out = scratch("chat-out");
    let mut o = Options::new(&src, harmful, harmless, &out);
    o.check = 2;
    o.check_tokens = 4;
    let mut events = Vec::new();
    let rep = abliterate::run(&o, &mut |v| events.push(v)).unwrap();
    assert!(events.iter().any(|e| e["event"] == "layers" && e["layers"].as_array().map(|a| a.len()) == Some(4)), "the score table is reported");
    assert_eq!(events.iter().filter(|e| e["event"] == "check").count(), 2, "before and after");
    let (before, after) = (rep.before.unwrap(), rep.after.unwrap());
    assert_eq!((before.n, after.n), (2, 2));
    assert!(before.harmless_loss.is_some() && after.harmless_loss.is_some());
    let cfg: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(out.join("config.json")).unwrap()).unwrap();
    assert_eq!(cfg["prastara_abliterated"]["prompts"]["harmful"], 4, "two of six were held out for the check");
    assert_eq!(cfg["prastara_private"], true);
    assert!(out.join("tokenizer.json").exists() && out.join("tokenizer_config.json").exists());
    check_result(&src, &out, &rep.chosen.direction, false);
    // too few prompts for the direction and the check together
    let mut short = Options::new(&src, vec!["a".into(), "b".into()], vec!["c".into(), "d".into()], scratch("short"));
    short.check = 1;
    assert!(abliterate::run(&short, &mut |_| {}).is_err());
    let _ = std::fs::remove_dir_all(&src);
    let _ = std::fs::remove_dir_all(&out);
}

#[test]
fn the_private_mark_survives_convert_and_quantize() {
    // a Qwen3 (convert_qwen3, the path the real model takes) and a Llama (the other converter)
    for (name, tag) in [("tiny-qwen3-hf", "mark-qwen3"), ("tiny-llama-hf", "mark-llama")] {
        let src = fixture(name);
        let hidden = strata_workshop::model::ModelConfig::load(&src).unwrap().hidden_size;
        let r: Vec<f32> = { let v: Vec<f32> = (0..hidden).map(|i| ((i * 7 % 5) as f32) - 2.0).collect(); let n = v.iter().map(|x| x * x).sum::<f32>().sqrt(); v.iter().map(|x| x / n).collect() };
        let out = scratch(tag);
        orthogonalize(&src, &out, &r, 1.0, serde_json::json!({"layer": 1, "scale": 1.0}), st_writer::SHARD_BYTES).unwrap();
        let tok = fixture("tiny-llama-hf");
        for f in ["tokenizer.json", "tokenizer_config.json"] {
            if !out.join(f).exists() {
                std::fs::copy(tok.join(f), out.join(f)).unwrap();
            }
        }
        let gguf = out.join("model.f32.gguf");
        strata_workshop::convert::convert(&out, &gguf, strata_workshop::convert::OutType::F32, Some("gpt-2")).unwrap();
        let q8 = out.join("model.q8_0.gguf");
        strata_workshop::quantize_gguf::quantize(&gguf, &q8, strata_workshop::quantize_gguf::PureType::from_s("q8_0").unwrap()).unwrap();
        for f in [&gguf, &q8] {
            let g = strata_gguf::Gguf::open(f).unwrap();
            assert_eq!(g.metadata.get(strata_workshop::convert::PRIVATE_KEY), Some(&serde_json::json!(true)), "{name}: {}", f.display());
            assert_eq!(g.metadata.get(strata_workshop::convert::ABLITERATED_KEY), Some(&serde_json::json!(true)), "{name}: {}", f.display());
            assert_eq!(g.u64("prastara.abliterated.layer"), Some(1));
        }
        // the source model carries no mark
        let plain = scratch(&format!("{tag}-plain"));
        std::fs::create_dir_all(&plain).unwrap();
        let pg = plain.join("plain.gguf");
        let srcdir = if name == "tiny-qwen3-hf" { let d = scratch(&format!("{tag}-src")); std::fs::create_dir_all(&d).unwrap();
            for e in std::fs::read_dir(&src).unwrap().flatten() { std::fs::copy(e.path(), d.join(e.file_name())).unwrap(); }
            for f in ["tokenizer.json", "tokenizer_config.json"] { std::fs::copy(tok.join(f), d.join(f)).unwrap(); } d } else { src.clone() };
        strata_workshop::convert::convert(&srcdir, &pg, strata_workshop::convert::OutType::F32, Some("gpt-2")).unwrap();
        assert!(strata_gguf::Gguf::open(&pg).unwrap().metadata.get(strata_workshop::convert::PRIVATE_KEY).is_none());
        for d in [out, plain] {
            let _ = std::fs::remove_dir_all(d);
        }
        if name == "tiny-qwen3-hf" {
            let _ = std::fs::remove_dir_all(srcdir);
        }
    }
}

#[test]
fn harmless_prompts_come_from_our_own_chat_data() {
    let d = scratch("harvest");
    std::fs::create_dir_all(d.join("sub")).unwrap();
    let rec = |u: &str| serde_json::json!({"messages": [{"role": "system", "content": "be kind"}, {"role": "user", "content": u}, {"role": "assistant", "content": "ok"}]}).to_string();
    std::fs::write(d.join("a.jsonl"), [rec("Explain how a\nrainbow forms."), rec("नमस्ते, आप कैसे हैं? मुझे बताइए"), rec("Explain how a rainbow forms."), rec("hi")].join("\n")).unwrap();
    std::fs::write(d.join("sub/b.jsonl"), [rec("Write a Rust function that adds two numbers."), rec("List four fruits that are yellow."), "not json".into()].join("\n")).unwrap();
    let all = abliterate::harvest_prompts(&[d.clone()], 100, 300, false).unwrap();
    assert_eq!(all, vec!["Explain how a rainbow forms.", "Write a Rust function that adds two numbers.", "नमस्ते, आप कैसे हैं? मुझे बताइए", "List four fruits that are yellow."],
               "newlines folded, duplicates and too-short ones dropped, the files taken in turn");
    let latin = abliterate::harvest_prompts(&[d.clone()], 2, 300, true).unwrap();
    assert_eq!(latin, vec!["Explain how a rainbow forms.", "Write a Rust function that adds two numbers."]);
    // a .jsonl list is read the same way by abliterate
    assert_eq!(abliterate::read_prompt_list(&d.join("sub/b.jsonl")).unwrap().len(), 2);
    let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn the_estimate_counts_the_real_looped_8b() {
    // the Mula base: Qwen3-8B, blocks 12..24 looped twice, untied head, bf16
    let cfg = strata_workshop::model::ModelConfig::from_json(&serde_json::json!({
        "architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3", "hidden_size": 4096, "intermediate_size": 12288, "num_hidden_layers": 36,
        "num_attention_heads": 32, "num_key_value_heads": 8, "head_dim": 128, "vocab_size": 151936, "rms_norm_eps": 1e-6, "rope_theta": 1000000,
        "max_position_embeddings": 40960, "tie_word_embeddings": false, "prastara_loop": {"start": 12, "end": 24, "passes": 2}})).unwrap();
    let src = 16_381_470_720u64;
    let cpu = abliterate::estimate(&cfg, BaseDtype::BF16, 0, 976, 21, 32, 24, src);
    assert_eq!(cpu.sites, 48, "36 blocks and the 12 looped ones once more");
    assert!((16.0e9..16.8e9).contains(&(cpu.model_bytes as f64)), "{}", cpu.model_bytes);
    assert!(cpu.ram_bytes > cpu.model_bytes && cpu.ram_bytes < 24_000_000_000, "{}", cpu.ram_bytes);
    assert_eq!(cpu.card_bytes, 0);
    let gpu = abliterate::estimate(&cfg, BaseDtype::BF16, 2, 976, 21, 32, 24, src);
    assert!(gpu.card_bytes < 14_000_000_000 && gpu.card_bytes > gpu.model_bytes / 2, "{}", gpu.card_bytes);
    assert!(gpu.ram_bytes < 6_000_000_000, "the GPUs hold the model: {}", gpu.ram_bytes);
    assert_eq!(gpu.disk_bytes, src);
    // the check (no cache) costs more than reading the prompts
    let no_check = abliterate::estimate(&cfg, BaseDtype::BF16, 2, 976, 21, 0, 24, src);
    assert!(gpu.flops > 2.0 * no_check.flops);
}
