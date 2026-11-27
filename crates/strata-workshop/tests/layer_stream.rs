//! Layer by layer (after Soup's `stream_layers`, `lisa` and `freeze_layers`, Apache-2.0): a run with the base streamed
//! block by block takes the same steps as one with the base held (dense, looped and upcycled tiny models, the RAM and
//! disk tiers, one and two devices); LISA trains only the blocks it samples, anew every interval, the same ones for the
//! same seed, and resumes exactly; `--freeze-bottom` is `--train-layers` counted from the bottom.

mod common;

use std::cell::Cell;
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

use candle_core::{Device, Tensor};
use strata_workshop::finetune::{self, lisa_sample, Event, Lisa, Resume, TrainReport, TrainSettings};
use strata_workshop::layer_stream::StreamMode;
use strata_workshop::model::{adapter_tensors, make_looped, make_upcycled, BaseDtype, LlamaModel, LoopSpec, LoraSpec, ModelConfig, MoeSpec, Target, MOE_INIT_FILE};

fn settings(model: &Path, data: &Path, out: &Path) -> TrainSettings {
    let mut s = TrainSettings::new(model, data, out);
    // dropout on: the replay must draw the same masks from the streamed weights
    s.lora = LoraSpec { r: 4, alpha: 8.0, dropout: 0.1, use_rslora: false, targets: Target::ALL.to_vec(), train_layers: None, seed: 7 };
    s.lr = 2e-2;
    s.epochs = 1.0;
    s.max_steps = Some(4);
    s.accumulation = 2;
    s.max_len = 64;
    s.val_split = 0.125;
    s.save_steps = 0;
    s.logging_steps = 1;
    s.seed = 11;
    s.gradient_checkpointing = true;
    s
}

fn run(s: &TrainSettings) -> (TrainReport, Vec<String>) {
    let mut logs = Vec::new();
    let r = finetune::train(s, &mut |e| if let Event::Log(l) = e { logs.push(l) }, &|| false).unwrap();
    (r, logs)
}

fn adapter(dir: &Path) -> BTreeMap<String, Vec<f32>> {
    candle_core::safetensors::load(dir.join(finetune::ADAPTER_FILE), &Device::Cpu).unwrap().into_iter()
        .map(|(k, v)| (k, v.flatten_all().unwrap().to_vec1::<f32>().unwrap())).collect()
}

fn max_diff(a: &BTreeMap<String, Vec<f32>>, b: &BTreeMap<String, Vec<f32>>) -> f32 {
    assert_eq!(a.keys().collect::<Vec<_>>(), b.keys().collect::<Vec<_>>(), "the same tensors");
    a.iter().map(|(k, x)| x.iter().zip(&b[k]).map(|(p, q)| (p - q).abs()).fold(0.0, f32::max)).fold(0.0, f32::max)
}

/// The base held, then streamed as `mode` (`devices` CPU devices, `buffers`): the same losses and adapter.
fn same_with_streaming(tag: &str, model: &Path, data: &Path, root: &Path, mode: StreamMode, devices: usize, buffers: usize) {
    let mut held = settings(model, data, &root.join(format!("{tag}-held")));
    held.devices = vec!["cpu".into(); devices];
    let (a, _) = run(&held);
    let mut streamed = held.clone();
    streamed.out = root.join(format!("{tag}-streamed"));
    streamed.stream_layers = Some(mode);
    streamed.stream_buffers = buffers;
    streamed.stream_cache = Some(root.join("shards"));
    let (b, logs) = run(&streamed);
    let dl = a.losses.iter().zip(&b.losses).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max);
    let da = max_diff(&adapter(&held.out), &adapter(&streamed.out));
    println!("{tag}: losses {:?} / {:?}, max loss difference {dl:.2e}, adapter {da:.2e}", a.losses, b.losses);
    assert_eq!(a.losses.len(), b.losses.len());
    assert!(dl <= 1e-6 && da <= 1e-6, "{tag}: losses differ by {dl}, the adapter by {da}");
    assert_eq!(a.eval_loss.map(|l| (l * 1e6).round()), b.eval_loss.map(|l| (l * 1e6).round()), "{tag}: the held-out loss");
    let st = b.stream.expect("streamed stats");
    assert!(st.loads > 0 && st.prefetch_hits > 0, "{tag}: {st:?}");
    assert!(a.stream.is_none());
    assert!(logs.iter().any(|l| l.contains("layer streaming (")), "{tag}: {logs:?}");
    // the adapter is still a plain PEFT adapter: it loads on the held base and gives the same logits
    let tokens: Vec<u32> = (1..20).collect();
    let logits = |dir: &Path| {
        let mut m = LlamaModel::load(model, BaseDtype::F32, &Device::Cpu).unwrap();
        m.load_adapter(dir).unwrap();
        m.forward(&tokens, &strata_workshop::model::Ctx::inference()).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap()
    };
    let (la, lb) = (logits(&held.out), logits(&streamed.out));
    assert!(la.iter().zip(&lb).all(|(x, y)| (x - y).abs() <= 1e-5));
}

#[test]
fn a_streamed_dense_model_trains_the_same() {
    let (root, model, data) = common::model_and_data("stream-dense");
    same_with_streaming("dense-ram", &model, &data, &root, StreamMode::Ram, 1, 2);
    same_with_streaming("dense-disk", &model, &data, &root, StreamMode::Disk, 1, 3);
    // the cache is made once: a second disk run finds it complete
    let mut again = settings(&model, &data, &root.join("again"));
    again.stream_layers = Some(StreamMode::Disk);
    again.stream_cache = Some(root.join("shards"));
    let (_, logs) = run(&again);
    assert!(logs.iter().any(|l| l.contains("is complete")), "{logs:?}");
    // two (CPU) devices: each with its own buffers
    same_with_streaming("dense-two", &model, &data, &root, StreamMode::Auto, 2, 2);
}

#[test]
fn a_streamed_looped_model_trains_the_same() {
    let root = common::temp_dir("stream-loop");
    let (base, data) = (root.join("base"), root.join("train.jsonl"));
    common::chat_model_sized(&base, 4, 32, 64);
    common::write_jsonl(&data, &common::chat_rows(24));
    let looped = root.join("looped");
    // blocks 1-2 run three times: the stream sees blocks again in the later passes
    make_looped(&base, &looped, LoopSpec { start: 1, end: 3, passes: 3 }).unwrap();
    same_with_streaming("loop", &looped, &data, &root, StreamMode::Ram, 1, 2);
    same_with_streaming("loop-two", &looped, &data, &root, StreamMode::Ram, 2, 4);
}

/// A fine-grained upcycle whose neurons are shuffled (the streamed slices must be the permuted ones).
fn upcycled(root: &Path, base: &Path, m: MoeSpec) -> PathBuf {
    let out = root.join(format!("up-{}-{}", m.experts, m.expert_ff.unwrap_or(0)));
    make_upcycled(base, &out, m).unwrap();
    if m.expert_ff.is_some() {
        let cfg = ModelConfig::load(&out).unwrap();
        let f = out.join(MOE_INIT_FILE);
        let mut t: HashMap<String, Tensor> = candle_core::safetensors::load(&f, &Device::Cpu).unwrap();
        for b in 0..cfg.num_hidden_layers {
            let perm: Vec<u32> = finetune::permutation(5, b as u64, cfg.intermediate_size).into_iter().map(|x| x as u32).collect();
            t.insert(format!("perm.{b}"), Tensor::from_vec(perm, (cfg.intermediate_size,), &Device::Cpu).unwrap());
        }
        std::fs::remove_file(&f).unwrap();
        candle_core::safetensors::save(&t, &f).unwrap();
    }
    out
}

#[test]
fn a_streamed_upcycled_model_trains_the_same() {
    let (root, model, data) = common::model_and_data("stream-moe");
    // 56B-A5B's design in small: 8 experts of 16 neurons (4 groups), top-2, expert rank 2 with a shared A
    let fine = upcycled(&root, &model, MoeSpec { expert_ff: Some(16), shared_a: true, expert_rank: Some(2), ..MoeSpec::full(8, 2) });
    same_with_streaming("moe-fine", &fine, &data, &root, StreamMode::Ram, 1, 2);
    same_with_streaming("moe-fine-two", &fine, &data, &root, StreamMode::Disk, 2, 2);
    // the full-width design (every expert the whole feed-forward)
    let full = upcycled(&root, &model, MoeSpec::full(4, 2));
    same_with_streaming("moe-full", &full, &data, &root, StreamMode::Ram, 1, 2);
}

#[test]
fn a_bf16_base_streams_the_same() {
    let (root, model, data) = common::model_and_data("stream-bf16");
    let mut held = settings(&model, &data, &root.join("held"));
    held.base_dtype = BaseDtype::BF16;
    let (a, _) = run(&held);
    let mut st = held.clone();
    st.out = root.join("streamed");
    st.stream_layers = Some(StreamMode::Ram);
    let (b, _) = run(&st);
    assert_eq!(a.losses, b.losses, "bit for bit");
    assert_eq!(max_diff(&adapter(&held.out), &adapter(&st.out)), 0.0);
}

/// Streaming without asking for checkpointing turns it on (and says so); bad buffer counts are refused.
#[test]
fn streaming_checks_its_settings() {
    let (root, model, data) = common::model_and_data("stream-settings");
    let mut s = settings(&model, &data, &root.join("o"));
    s.gradient_checkpointing = false;
    s.stream_layers = Some(StreamMode::Ram);
    s.max_steps = Some(1);
    let (_, logs) = run(&s);
    assert!(logs.iter().any(|l| l.contains("turns gradient checkpointing on")), "{logs:?}");
    for bad in [1, 9] {
        let mut b = s.clone();
        b.stream_buffers = bad;
        assert!(finetune::train(&b, &mut |_| {}, &|| false).is_err(), "{bad} buffers");
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// LISA
// ---------------------------------------------------------------------------------------------------------------------

/// The adapter a run starts from (the same seed's fresh adapter), per tensor.
fn fresh_adapter(model: &Path, spec: &LoraSpec) -> BTreeMap<String, Vec<f32>> {
    let mut m = LlamaModel::load(model, BaseDtype::F32, &Device::Cpu).unwrap();
    m.add_lora(spec).unwrap();
    adapter_tensors(&m).into_iter().map(|(k, v)| (k, v.flatten_all().unwrap().to_vec1::<f32>().unwrap())).collect()
}

fn block_of(name: &str) -> Option<usize> {
    let rest = &name[name.find("layers.")? + 7..];
    rest[..rest.find('.')?].parse().ok()
}

/// The blocks whose adapter tensors moved from `start`.
fn changed_blocks(start: &BTreeMap<String, Vec<f32>>, end: &BTreeMap<String, Vec<f32>>) -> Vec<usize> {
    let mut out: Vec<usize> = end.iter().filter(|(k, v)| start[*k] != **v).filter_map(|(k, _)| block_of(k)).collect();
    out.sort_unstable();
    out.dedup();
    out
}

fn lisa_model(tag: &str) -> (common::Tmp, PathBuf, PathBuf) {
    let root = common::temp_dir(tag);
    let (model, data) = (root.join("model"), root.join("train.jsonl"));
    common::chat_model_sized(&model, 6, 32, 64);
    common::write_jsonl(&data, &common::chat_rows(24));
    (root, model, data)
}

#[test]
fn lisa_samples_the_same_blocks_for_a_seed_and_new_ones_each_interval() {
    let blocks: Vec<usize> = (0..12).collect();
    let a: Vec<Vec<usize>> = (0..10).map(|i| lisa_sample(3, i, &blocks, 2)).collect();
    let b: Vec<Vec<usize>> = (0..10).map(|i| lisa_sample(3, i, &blocks, 2)).collect();
    let c: Vec<Vec<usize>> = (0..10).map(|i| lisa_sample(4, i, &blocks, 2)).collect();
    assert_eq!(a, b, "deterministic by seed");
    assert_ne!(a, c, "another seed, other blocks");
    assert!(a.iter().all(|s| s.len() == 2 && s[0] < s[1] && s.iter().all(|x| blocks.contains(x))));
    assert!(a.windows(2).filter(|w| w[0] != w[1]).count() >= 7, "the sample changes from interval to interval: {a:?}");
    // more than there are: all of them
    assert_eq!(lisa_sample(1, 0, &[4, 5], 3), vec![4, 5]);
}

#[test]
fn lisa_trains_only_the_sampled_blocks() {
    let (root, model, data) = lisa_model("lisa");
    let mut s = settings(&model, &data, &root.join("one"));
    s.lisa = Some(Lisa { layers: 2, interval: 2, seed: 9 });
    s.max_steps = Some(2);
    let start = fresh_adapter(&model, &s.lora);
    let all: Vec<usize> = (0..6).collect();
    // one interval: only its two blocks moved, every other block's adapter is as it started, to the bit
    let (_, logs) = run(&s);
    let first = lisa_sample(9, 0, &all, 2);
    assert_eq!(changed_blocks(&start, &adapter(&s.out)), first, "{logs:?}");
    assert!(logs.iter().any(|l| l.starts_with("LISA: blocks")), "{logs:?}");
    // three intervals: the blocks of each, and no others
    let mut three = s.clone();
    three.out = root.join("three");
    three.max_steps = Some(6);
    let (_, logs) = run(&three);
    let mut want: Vec<usize> = (0..3).flat_map(|i| lisa_sample(9, i, &all, 2)).collect();
    want.sort_unstable();
    want.dedup();
    assert_eq!(changed_blocks(&start, &adapter(&three.out)), want);
    assert_eq!(logs.iter().filter(|l| l.starts_with("LISA: blocks")).count(), 3, "{logs:?}");
    // with --train-layers 3 only the top three blocks are candidates
    let mut top = s.clone();
    top.out = root.join("top");
    top.lora.train_layers = Some(3);
    top.max_steps = Some(6);
    run(&top);
    let start_top = fresh_adapter(&model, &top.lora);
    let moved = changed_blocks(&start_top, &adapter(&top.out));
    assert!(!moved.is_empty() && moved.iter().all(|b| *b >= 3), "{moved:?}");
}

#[test]
fn lisa_with_streaming_and_after_a_resume_takes_the_same_steps() {
    let (root, model, data) = lisa_model("lisa-stream");
    let mut s = settings(&model, &data, &root.join("plain"));
    s.lisa = Some(Lisa { layers: 2, interval: 2, seed: 4 });
    s.max_steps = Some(6);
    let (a, _) = run(&s);
    // streamed: the not-sampled blocks are read as constants, the same arithmetic otherwise
    let mut st = s.clone();
    st.out = root.join("streamed");
    st.stream_layers = Some(StreamMode::Ram);
    let (b, _) = run(&st);
    let dl = a.losses.iter().zip(&b.losses).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max);
    assert!(dl <= 1e-6 && max_diff(&adapter(&s.out), &adapter(&st.out)) <= 1e-6, "{dl}");
    // stopped at step 2 (an interval's end: the next step samples anew) and resumed: the uninterrupted run's steps
    let mut part = s.clone();
    part.out = root.join("resumed");
    let steps = Cell::new(0);
    let r = finetune::train(&part, &mut |e| if let Event::Step { .. } = e { steps.set(steps.get() + 1) }, &|| steps.get() >= 2).unwrap();
    assert!(r.stopped && r.steps_done == 2);
    part.resume = Some(Resume::Auto);
    let (c, _) = run(&part);
    assert_eq!(c.steps_done, 6);
    let dl = a.losses.iter().zip(&c.losses).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max);
    let da = max_diff(&adapter(&s.out), &adapter(&part.out));
    println!("LISA resumed: losses differ by {dl:.2e}, adapter by {da:.2e}");
    assert!(dl <= 1e-6 && da <= 1e-6, "{dl} {da}");
    // a checkpoint made without LISA is not resumed with it (another run's steps)
    let mut other = s.clone();
    other.lisa = Some(Lisa { layers: 3, interval: 2, seed: 4 });
    other.out = part.out.clone();
    other.resume = Some(Resume::Dir(part.out.join("checkpoint-2")));
    assert!(finetune::train(&other, &mut |_| {}, &|| false).is_err());
}

#[test]
fn lisa_on_an_upcycled_model_trains_the_sampled_routers_only() {
    let (root, model, data) = common::model_and_data("lisa-moe");
    let fine = upcycled(&root, &model, MoeSpec { expert_ff: Some(16), shared_a: true, expert_rank: Some(2), ..MoeSpec::full(8, 2) });
    let mut s = settings(&fine, &data, &root.join("o"));
    s.lisa = Some(Lisa { layers: 1, interval: 3, seed: 2 });
    s.max_steps = Some(3);
    let start = fresh_adapter(&fine, &s.lora);
    run(&s);
    let end = adapter(&s.out);
    let chosen = lisa_sample(2, 0, &[0, 1, 2], 1);
    assert_eq!(changed_blocks(&start, &end), chosen);
    let router = strata_workshop::model::moe_router_name(chosen[0]);
    assert_ne!(start[&router], end[&router], "the sampled block's router trained");
}

#[test]
fn freeze_bottom_is_train_layers_from_the_bottom() {
    let (root, model, data) = common::model_and_data("freeze");
    let mut s = settings(&model, &data, &root.join("o"));
    s.freeze_bottom = Some(1);
    s.apply_freeze_bottom(3).unwrap();
    assert_eq!((s.lora.train_layers, s.freeze_bottom), (Some(2), None));
    let mut zero = settings(&model, &data, &root.join("z"));
    zero.freeze_bottom = Some(0);
    zero.apply_freeze_bottom(3).unwrap();
    assert_eq!(zero.lora.train_layers, None);
    for (n, t) in [(3, None), (1, Some(1))] {
        let mut bad = settings(&model, &data, &root.join("b"));
        bad.freeze_bottom = Some(n);
        bad.lora.train_layers = t;
        assert!(bad.apply_freeze_bottom(3).is_err(), "{n} {t:?}");
    }
    // a run with it trains the top two blocks, as --train-layers 2 does
    let mut a = settings(&model, &data, &root.join("fb"));
    a.freeze_bottom = Some(1);
    a.max_steps = Some(2);
    let mut b = settings(&model, &data, &root.join("tl"));
    b.lora.train_layers = Some(2);
    b.max_steps = Some(2);
    let (ra, _) = run(&a);
    let (rb, _) = run(&b);
    assert_eq!(ra.losses, rb.losses);
    assert_eq!(max_diff(&adapter(&a.out), &adapter(&b.out)), 0.0);
    assert!(adapter(&a.out).keys().all(|k| block_of(k).is_none_or(|b| b >= 1)));
}

#[test]
fn a_soup_config_streams_freezes_and_with_relax_samples() {
    let dir = common::temp_dir("soup-stream");
    let yaml = "base: ./m\ndata:\n  train: d.jsonl\ntraining:\n  quantization: none\n  stream_layers: true\n  stream_source: disk\n  stream_buffers: 3\n  freeze_layers: 4\n  lora:\n    r: 8\n";
    let imp = strata_workshop::soup::import(yaml, &dir, false).unwrap();
    assert_eq!((imp.settings.stream_layers, imp.settings.stream_buffers, imp.settings.freeze_bottom), (Some(StreamMode::Disk), 3, Some(4)));
    assert!(imp.settings.gradient_checkpointing);
    let lisa = "base: ./m\ndata:\n  train: d.jsonl\ntraining:\n  quantization: none\n  lisa_enabled: true\n  lisa_num_layers: 3\n  lisa_interval_steps: 5\n";
    assert!(strata_workshop::soup::import(lisa, &dir, false).is_err(), "Soup's LISA is whole-layer fine-tuning");
    let relaxed = strata_workshop::soup::import(lisa, &dir, true).unwrap();
    assert_eq!(relaxed.settings.lisa, Some(Lisa { layers: 3, interval: 5, seed: 42 }));
    let bad = "base: ./m\ndata:\n  train: d.jsonl\ntraining:\n  quantization: none\n  stream_layers: true\n  stream_source: nvme\n";
    assert!(strata_workshop::soup::import(bad, &dir, false).is_err());
}

/// On the GPUs (`--features cuda`, run with `-- --ignored`; never in the default run): a streamed run on cuda:0 (and on
/// cuda:0,cuda:1 with AM2_TEST_GPUS=2) takes the steps a held one does there, within float noise, and its uploads were
/// prefetched.
#[cfg(feature = "cuda")]
#[test]
#[ignore]
fn on_the_gpus_a_streamed_run_matches_the_held_one() {
    let (root, model, data) = common::model_and_data("stream-gpu");
    let devices: Vec<String> = match std::env::var("AM2_TEST_GPUS").as_deref() {
        Ok("2") => vec!["cuda:0".into(), "cuda:1".into()],
        _ => vec!["cuda:0".into()],
    };
    for dtype in [BaseDtype::F32, BaseDtype::BF16] {
        let mut held = settings(&model, &data, &root.join(format!("held-{}", dtype.as_str())));
        held.devices = devices.clone();
        held.base_dtype = dtype;
        let (a, _) = run(&held);
        for mode in [StreamMode::Ram, StreamMode::Disk] {
            let mut st = held.clone();
            st.out = root.join(format!("streamed-{}-{}", dtype.as_str(), mode.as_str()));
            st.stream_layers = Some(mode);
            st.stream_cache = Some(root.join("shards"));
            let (b, logs) = run(&st);
            let dl = a.losses.iter().zip(&b.losses).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max);
            let da = max_diff(&adapter(&held.out), &adapter(&st.out));
            println!("{devices:?} {} {}: losses differ by {dl:.2e}, adapter by {da:.2e}; {:?}", dtype.as_str(), mode.as_str(), b.stream);
            for l in logs.iter().filter(|l| l.contains("layer streaming")) {
                println!("  {l}");
            }
            assert!(dl <= 1e-4 && da <= 1e-4, "{dl} {da}");
            assert!(b.stream.unwrap().loads > 0);
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// gradients, streamed against held: the real 56B-A5B run is an upcycled MoE over a looped base on two devices
// ---------------------------------------------------------------------------------------------------------------------

/// A looped base (blocks 1-2 twice), then a fine-grained upcycle of it (8 experts of 16 neurons, top-2, shared A):
/// the 56B-A5B's shape in small.
fn looped_moe(root: &Path, hidden: usize, ff: usize) -> (PathBuf, PathBuf) {
    let (base, data) = (root.join("base"), root.join("train.jsonl"));
    common::chat_model_sized(&base, 4, hidden, ff);
    common::write_jsonl(&data, &common::chat_rows(24));
    let looped = root.join("looped");
    make_looped(&base, &looped, LoopSpec { start: 1, end: 3, passes: 2 }).unwrap();
    let moe = upcycled(root, &looped, MoeSpec { expert_ff: Some(ff / 4), shared_a: true, expert_rank: Some(2), ..MoeSpec::full(8, 2) });
    (moe, data)
}

fn open(devices: &[&str]) -> Vec<Device> {
    devices.iter().map(|d| match *d {
        "cpu" => Device::Cpu,
        #[cfg(feature = "cuda")]
        d if d.starts_with("cuda:") => Device::new_cuda(d[5..].parse().unwrap()).unwrap(),
        other => panic!("{other}"),
    }).collect()
}

/// One loss and every adapter variable's gradient (B set non-zero first, so A, the router and the loop gates all
/// get one), for the model `how` loads: "held", "held-ckpt", or a streamed tier ("ram", "disk").
fn grads_of(model: &Path, root: &Path, how: &str, devices: &[&str], buffers: usize, dtype: BaseDtype, tokens: &[u32]) -> (f64, BTreeMap<String, Vec<f32>>) {
    let devs = open(devices);
    let n = ModelConfig::load(model).unwrap().num_hidden_layers;
    let mut m = match how {
        "held" | "held-ckpt" => LlamaModel::load_parts_on(model, dtype, &devs, 0..n, true, true).unwrap(),
        tier => {
            let opts = strata_workshop::layer_stream::StreamOptions { mode: StreamMode::from_s(tier).unwrap(), buffers, cache_root: Some(root.join("shards")) };
            LlamaModel::load_streamed(model, dtype, &devs, 0..n, true, true, &opts, &mut |_| {}).unwrap()
        }
    };
    m.add_lora(&LoraSpec { r: 4, alpha: 8.0, dropout: 0.1, use_rslora: false, targets: Target::ALL.to_vec(), train_layers: None, seed: 7 }).unwrap();
    m.set_checkpointing(how != "held");
    let vars = m.lora_vars();
    for (k, (name, v)) in vars.iter().enumerate() {
        if name.contains("lora_B") {
            let t = v.as_tensor();
            let vals: Vec<f32> = (0..t.elem_count()).map(|i| (((i * 7919 + k * 104_729) % 1000) as f32 / 1000.0 - 0.5) * 0.2).collect();
            v.set(&Tensor::from_vec(vals, t.dims(), t.device()).unwrap()).unwrap();
        }
    }
    let supervised: Vec<bool> = (0..tokens.len()).map(|p| p > tokens.len() / 3).collect();
    let mut total = 0.0;
    let mut acc: BTreeMap<String, Vec<f32>> = BTreeMap::new();
    // two micro-batches in a row (the second finds the buffers the first left), each with its own backward pass
    for rep in 0..2u64 {
        let ctx = strata_workshop::model::Ctx::train(13 + rep);
        let loss = m.loss(tokens, &supervised, &ctx).unwrap();
        total += loss.to_dtype(candle_core::DType::F32).unwrap().to_scalar::<f32>().unwrap() as f64;
        let g = loss.backward().unwrap();
        let ck = m.take_checkpoint_grads();
        for (name, v) in &vars {
            let mut sum: Option<Tensor> = g.get(v.as_tensor()).map(|t| t.to_device(&Device::Cpu).unwrap());
            for (cv, ct) in &ck {
                if cv.as_tensor().id() == v.as_tensor().id() {
                    let ct = ct.to_device(&Device::Cpu).unwrap();
                    sum = Some(match sum { Some(s) => (s + ct).unwrap(), None => ct });
                }
            }
            let vals = sum.map(|s| s.to_dtype(candle_core::DType::F32).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap())
                .unwrap_or_else(|| vec![0.0; v.as_tensor().elem_count()]);
            let e = acc.entry(name.clone()).or_insert_with(|| vec![0.0; vals.len()]);
            e.iter_mut().zip(vals).for_each(|(a, b)| *a += b);
        }
    }
    (total, acc)
}

/// Gradients of `b` against `a`: the largest difference over the largest magnitude, and the tensor where it is.
fn grad_gap(a: &BTreeMap<String, Vec<f32>>, b: &BTreeMap<String, Vec<f32>>) -> (f32, String) {
    assert_eq!(a.keys().collect::<Vec<_>>(), b.keys().collect::<Vec<_>>());
    let scale = a.values().flatten().fold(0.0f32, |m, x| m.max(x.abs())).max(1e-12);
    let mut worst = (0.0f32, String::new());
    for (k, x) in a {
        let d = x.iter().zip(&b[k]).map(|(p, q)| (p - q).abs()).fold(0.0, f32::max) / scale;
        if d > worst.0 {
            worst = (d, k.clone());
        }
    }
    worst
}

/// Every variant's loss and gradients against the held, un-checkpointed model's.
fn gradients_agree(devices: &[&str], dtype: BaseDtype, tol: f32, hidden: usize, ff: usize) {
    let root = common::temp_dir("stream-grads");
    let (moe, _) = looped_moe(&root, hidden, ff);
    let tokens: Vec<u32> = (0..48).map(|i| (i * 37 % 200 + 3) as u32).collect();
    let (l0, g0) = grads_of(&moe, &root, "held", devices, 2, dtype, &tokens);
    assert!(g0.values().flatten().any(|x| *x != 0.0));
    let nonzero = g0.iter().filter(|(_, v)| v.iter().any(|x| *x != 0.0)).count();
    assert!(nonzero * 10 >= g0.len() * 9, "most variables get a gradient: {nonzero} of {}", g0.len());
    for (how, buffers) in [("held-ckpt", 2), ("ram", 2), ("ram", 3), ("disk", 2)] {
        let (l, g) = grads_of(&moe, &root, how, devices, buffers, dtype, &tokens);
        let (gap, at) = grad_gap(&g0, &g);
        println!("{devices:?} {} {how} x{buffers}: loss {l0:.6} / {l:.6}, gradients differ by {gap:.2e} of the largest (at {at})", dtype.as_str());
        assert!((l - l0).abs() <= tol as f64 * l0.abs().max(1.0), "{how}: loss {l} vs {l0}");
        assert!(gap <= tol, "{devices:?} {how} x{buffers}: gradients differ by {gap} at {at}");
    }
}

#[test]
fn a_streamed_looped_moe_has_the_held_gradients() {
    gradients_agree(&["cpu"], BaseDtype::F32, 1e-5, 32, 64);
    gradients_agree(&["cpu", "cpu"], BaseDtype::F32, 1e-5, 32, 64);
}

/// The 56B-A5B run's shape in small through the whole trainer: several steps at lr > 0 after a warm-up step, the
/// held-out loss, streamed against held, on one and two devices.
fn looped_moe_trains_the_same(devices: &[&str], dtype: BaseDtype, tol: f64, hidden: usize, ff: usize) {
    let root = common::temp_dir("stream-loop-moe");
    let (moe, data) = looped_moe(&root, hidden, ff);
    let mut held = settings(&moe, &data, &root.join("held"));
    held.devices = devices.iter().map(|d| d.to_string()).collect();
    held.base_dtype = dtype;
    held.lr = 2e-3;
    held.max_steps = Some(6);
    held.warmup_ratio = 0.17;
    let (a, _) = run(&held);
    let mut plain = held.clone();
    plain.out = root.join("plain");
    plain.gradient_checkpointing = false;
    let (p, _) = run(&plain);
    for (mode, buffers) in [(StreamMode::Ram, 2), (StreamMode::Disk, 2)] {
        let mut st = held.clone();
        st.out = root.join(format!("streamed-{}", mode.as_str()));
        st.stream_layers = Some(mode);
        st.stream_buffers = buffers;
        st.stream_cache = Some(root.join("shards"));
        let (b, _) = run(&st);
        let dl = a.losses.iter().zip(&b.losses).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max);
        let de = (a.eval_loss.unwrap() - b.eval_loss.unwrap()).abs();
        println!("{devices:?} {} {}: losses {:?} / {:?} (no checkpointing {:?}), eval {:?} / {:?}", dtype.as_str(), mode.as_str(), a.losses, b.losses, p.losses,
                 a.eval_loss, b.eval_loss);
        assert!(dl <= tol && de <= tol, "{devices:?} {}: losses differ by {dl}, eval by {de}", mode.as_str());
    }
    let dp = a.losses.iter().zip(&p.losses).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max);
    // a bf16 base on a GPU: checkpointed and not differ by ~2.5e-3 after a few steps. Not a bug: under checkpointing a
    // looped block's adapters get their gradient in two parts (one per pass, from the replays) and without it in one,
    // so the sums associate differently; AdamW turns those last-bit differences on near-zero gradients into full-size
    // steps. Changing only the order the trainer adds a micro-step's parts in moved the drift from the two-GPU run to
    // the one-GPU one. f32 and the CPU agree to 1e-7; streamed against held (both checkpointed) stays exact.
    let loose = dtype != BaseDtype::F32 && devices.iter().all(|d| d.starts_with("cuda"));
    let ptol = if loose { 5e-3 } else { tol };
    assert!(dp <= ptol, "checkpointing changes the losses by {dp}");
    // training helps: the held-out loss after training is in line with the last train losses, not far above them
    let (first, last, eval) = (a.losses[0], *a.losses.last().unwrap(), a.eval_loss.unwrap());
    assert!(last < first && eval < first * 1.2, "losses {:?}, eval {eval}", a.losses);
}

#[test]
fn a_streamed_looped_moe_trains_and_evaluates_the_same() {
    looped_moe_trains_the_same(&["cpu"], BaseDtype::F32, 1e-6, 32, 64);
    looped_moe_trains_the_same(&["cpu", "cpu"], BaseDtype::F32, 1e-6, 32, 64);
}

/// AM2_TEST_GPUS=2: cuda:0 and cuda:1; AM2_TEST_DEVICES=cuda:1 (a comma list) names them; else cuda:0.
fn test_gpus() -> Vec<&'static str> {
    if let Ok(list) = std::env::var("AM2_TEST_DEVICES") {
        return list.split(',').map(|d| &*Box::leak(d.trim().to_string().into_boxed_str())).collect();
    }
    match std::env::var("AM2_TEST_GPUS").as_deref() {
        Ok("2") => vec!["cuda:0", "cuda:1"],
        _ => vec!["cuda:0"],
    }
}

/// On the GPUs (`-- --ignored`): the looped MoE's gradients and steps, streamed against held, f32 and bf16, with
/// blocks large enough (hidden 512, 2,048 neurons: 17 MB a block in f32) that an upload takes longer than the
/// kernels of a block on a few tokens, so a copy the compute does not wait for shows.
#[cfg(feature = "cuda")]
#[test]
#[ignore]
fn on_the_gpus_a_streamed_looped_moe_has_the_held_gradients() {
    let gpus = test_gpus();
    gradients_agree(&gpus, BaseDtype::F32, 1e-4, 512, 2048);
    gradients_agree(&gpus, BaseDtype::BF16, 1e-4, 512, 2048);
}

#[cfg(feature = "cuda")]
#[test]
#[ignore]
fn on_the_gpus_a_streamed_looped_moe_trains_and_evaluates_the_same() {
    let gpus = test_gpus();
    looped_moe_trains_the_same(&gpus, BaseDtype::F32, 1e-4, 512, 2048);
    looped_moe_trains_the_same(&gpus, BaseDtype::BF16, 1e-4, 512, 2048);
}

/// On two GPUs (`AM2_TEST_GPUS=2 -- --ignored`): the looped MoE's logits match the CPU's, block by block and through
/// the loop's gate. The gate lives with the loop's last block, on the second card; it was copied there GPU to GPU
/// (candle's dtod copy between two cards' contexts), which did not wait for its source and gave a wrong gate.
#[cfg(feature = "cuda")]
#[test]
#[ignore]
fn on_the_gpus_a_split_looped_moe_computes_what_the_cpu_does() {
    let root = common::temp_dir("split-forward");
    let (moe, _) = looped_moe(&root, 512, 2048);
    let tokens: Vec<u32> = (0..48).map(|i| (i * 37 % 200 + 3) as u32).collect();
    let n = ModelConfig::load(&moe).unwrap().num_hidden_layers;
    let logits = |d: &[&str]| {
        let m = LlamaModel::load_parts_on(&moe, BaseDtype::F32, &open(d), 0..n, true, true).unwrap();
        let gate = m.loop_gate_tensors()["loop.gate.1"].to_device(&Device::Cpu).unwrap().to_vec1::<f32>().unwrap();
        (m.forward(&tokens, &strata_workshop::model::Ctx::inference()).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap(), gate)
    };
    let (want, gate) = logits(&["cpu"]);
    for d in [test_gpus(), test_gpus().into_iter().rev().collect()] {
        let (got, g) = logits(&d);
        let gap = got.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
        println!("{d:?}: logits differ from the CPU's by {gap:.2e}");
        assert_eq!(g, gate, "{d:?}: the loop's gate");
        assert!(gap <= 1e-4, "{d:?}: {gap}");
    }
}

/// On the GPUs (`-- --ignored`; AM2_TEST_DEVICES=cuda:1 picks a card): 8 blocks of hidden 512 (the loop 2..6 across
/// the two cards), 16 experts, bf16, attention LoRA rank 16, conversations of 600-1,000 tokens (the held-out pass takes
/// the chunked attention): held, held checkpointed and streamed take the same five steps (warm-up, lr 2e-4) and give the
/// same held-out loss. The blocks are big enough that a block's replay outlasts the next block's upload, which is what
/// showed the streamed run's buffer being freed and refilled under a replay still reading it.
#[cfg(feature = "cuda")]
#[test]
#[ignore]
fn on_the_gpus_a_larger_streamed_looped_moe_trains_the_same() {
    let root = common::temp_dir("probe-scale");
    let (base, data) = (root.join("base"), root.join("train.jsonl"));
    common::chat_model_sized(&base, 8, 512, 2048);
    let rows: Vec<serde_json::Value> = (0..24).map(|i| {
        let (thing, colour) = common::THINGS[i % common::THINGS.len()];
        let story = format!("The {thing} is {colour}. ").repeat(20 + (i % 5) * 6);
        serde_json::json!({"messages": [{"role": "user", "content": format!("{story} What colour is {thing}?")}, {"role": "assistant", "content": format!("It is {colour}. {story}")}]})
    }).collect();
    common::write_jsonl(&data, &rows);
    let looped = root.join("looped");
    make_looped(&base, &looped, LoopSpec { start: 2, end: 6, passes: 2 }).unwrap();
    let moe = upcycled(&root, &looped, MoeSpec { expert_ff: Some(512), shared_a: true, expert_rank: Some(2), ..MoeSpec::full(16, 4) });
    let gpus = test_gpus();
    let mut held = settings(&moe, &data, &root.join("held"));
    held.devices = gpus.iter().map(|d| d.to_string()).collect();
    held.base_dtype = BaseDtype::BF16;
    held.lr = 2e-4;
    held.max_steps = Some(5);
    held.warmup_ratio = 0.2;
    held.max_len = 1024;
    held.lora.targets = vec![Target::Q, Target::K, Target::V, Target::O];
    held.lora.r = 16;
    held.lora.alpha = 32.0;
    held.lora.dropout = 0.05;
    let mut plain = held.clone();
    plain.out = root.join("plain");
    plain.gradient_checkpointing = false;
    let mut st = held.clone();
    st.out = root.join("streamed");
    st.stream_layers = Some(StreamMode::Ram);
    let (rp, _) = run(&plain);
    let (rh, _) = run(&held);
    let (rs, _) = run(&st);
    for (tag, r) in [("held", &rp), ("held, checkpointed", &rh), ("streamed", &rs)] {
        println!("{gpus:?} {tag}: losses {:?}, eval {:?}, {:?}", r.losses, r.eval_loss, r.stream);
    }
    let gap = |a: &TrainReport, b: &TrainReport| a.losses.iter().zip(&b.losses).map(|(x, y)| (x - y).abs()).fold((a.eval_loss.unwrap() - b.eval_loss.unwrap()).abs(), f64::max);
    // before the fix the streamed run left the others from step 3 on (its first update): 5.717 against 5.682
    assert!(gap(&rp, &rs) <= 1e-4 && gap(&rh, &rs) <= 1e-4, "streamed {:?} / held {:?}", rs.losses, rh.losses);
    assert!(rs.stream.unwrap().prefetch_hits > 0);
}

/// On the GPUs (`-- --ignored`): the streamed buffers are made once and written in place, so the device's memory pool
/// does not grow over many micro-steps of varying length: each micro-step's reserve (high-water mark, the pool trimmed
/// in between as the trainer does) stays within that of the first rounds, and the pool holds only `buffers` blocks
/// beyond the activations.
#[cfg(feature = "cuda")]
#[test]
#[ignore]
fn on_the_gpus_the_stream_buffers_do_not_grow_the_pool() {
    let root = common::temp_dir("stream-pool");
    let (moe, _) = looped_moe(&root, 512, 2048);
    let devs = open(&test_gpus());
    let n = ModelConfig::load(&moe).unwrap().num_hidden_layers;
    let opts = strata_workshop::layer_stream::StreamOptions { mode: StreamMode::Ram, buffers: 2, cache_root: None };
    let mut m = LlamaModel::load_streamed(&moe, BaseDtype::BF16, &devs, 0..n, true, true, &opts, &mut |_| {}).unwrap();
    m.add_lora(&LoraSpec { r: 4, alpha: 8.0, dropout: 0.0, use_rslora: false, targets: Target::ALL.to_vec(), train_layers: None, seed: 7 }).unwrap();
    m.set_checkpointing(true);
    let block = m.stream().unwrap().block_bytes();
    let lengths = [40usize, 200, 90, 300, 20, 250, 130, 60];
    let mut marks: Vec<Vec<(u64, u64)>> = Vec::new();
    m.synchronize().unwrap();
    m.memory_marks(true);
    for round in 0..3 {
        for (k, len) in lengths.iter().enumerate() {
            let tokens: Vec<u32> = (0..*len).map(|i| ((i * 37 + k * 11 + round) % 200 + 3) as u32).collect();
            let supervised: Vec<bool> = (0..tokens.len()).map(|p| p > tokens.len() / 2).collect();
            let loss = m.loss(&tokens, &supervised, &strata_workshop::model::Ctx::train(k as u64)).unwrap();
            let g = loss.backward().unwrap();
            drop((g, m.take_checkpoint_grads(), loss));
            m.synchronize().unwrap();
            marks.push(m.memory_marks(true));
        }
    }
    let per = lengths.len();
    for d in 0..devs.len() {
        let mib = |b: u64| b >> 20;
        let first = marks[..per].iter().map(|v| v[d].1).max().unwrap();
        let later = marks[per..].iter().map(|v| v[d].1).max().unwrap();
        let slack = marks.iter().map(|v| v[d].1.saturating_sub(v[d].0)).max().unwrap();
        println!("dev{d}: reserve-high first round {} MiB, later rounds {} MiB; reserve above use at most {} MiB (a block is {} MiB); {:?}",
                 mib(first), mib(later), mib(slack), mib(block), marks.iter().map(|v| (mib(v[d].0), mib(v[d].1))).collect::<Vec<_>>());
        assert!(later <= first + (first >> 5), "dev{d}: the pool's reserve grew from {first} to {later}");
        assert!(slack <= 2 * block + (48 << 20), "dev{d}: the pool reserved {slack} bytes above its use");
    }
}

/// On the GPUs (`-- --ignored`): after a streamed load the process holds the page-locked store and nothing of the
/// mapped checkpoint (its pages are released block by block), and the kernel's count of the page-locked pages is
/// double (file-backed and shared, through /dev/nvidiactl), which the log's note takes out.
#[cfg(feature = "cuda")]
#[test]
#[ignore]
fn on_the_gpus_a_streamed_load_keeps_no_checkpoint_resident() {
    let kb = |k: &str| -> u64 {
        std::fs::read_to_string("/proc/self/status").unwrap().lines().find(|l| l.starts_with(k)).unwrap().split_whitespace().nth(1).unwrap().parse::<u64>().unwrap() * 1024
    };
    let root = common::temp_dir("stream-rss");
    let base = root.join("base");
    common::chat_model_sized(&base, 4, 2048, 8192);
    let file: u64 = std::fs::read_dir(&base).unwrap().flatten().filter(|e| e.path().extension().is_some_and(|x| x == "safetensors")).map(|e| e.metadata().unwrap().len()).sum();
    let devs = open(&test_gpus());
    for d in &devs {
        Tensor::zeros(8, candle_core::DType::F32, d).unwrap().sum_all().unwrap();
    }
    let before = kb("VmRSS:") - kb("RssAnon:");
    let opts = strata_workshop::layer_stream::StreamOptions { mode: StreamMode::Ram, buffers: 2, cache_root: None };
    let m = LlamaModel::load_streamed(&base, BaseDtype::BF16, &devs, 0..4, true, true, &opts, &mut |_| {}).unwrap();
    let pinned = m.stream().unwrap().pinned_bytes;
    let twice = strata_workshop::layer_stream::driver_mapped_bytes();
    // what is resident and not anonymous, beyond what was there before: the pinned store, counted twice
    let after = kb("VmRSS:") - kb("RssAnon:");
    let left = (after - before) as i64 - 2 * pinned as i64;
    let note = strata_workshop::layer_stream::host_memory_note().unwrap();
    println!("checkpoint {:.2} GB, pinned {:.2} GB, driver-mapped {:.2} GB, mapped beyond the pinned store {:.2} GB; {note}",
             file as f64 / 1e9, pinned as f64 / 1e9, twice as f64 / 1e9, left as f64 / 1e9);
    // the driver maps a little of its own per card (about 60 MB) beside the page-locked store
    assert!(pinned > 0 && twice >= pinned && twice <= pinned + devs.len() as u64 * (128 << 20), "pinned {pinned}, driver-mapped {twice}");
    assert!(left < (file / 10) as i64, "{left} bytes of the {file}-byte checkpoint still resident");
    drop(m);
}

/// A short run (up to `SHORT_RUN_STEPS`) reports every step's loss, whatever `logging_steps` says.
#[test]
fn a_short_run_reports_every_step() {
    let (root, model, data) = common::model_and_data("short-run-steps");
    let mut s = settings(&model, &data, &root.join("o"));
    s.logging_steps = 10;
    s.max_steps = Some(5);
    let mut steps = Vec::new();
    finetune::train(&s, &mut |e| if let Event::Step { step, loss, .. } = e { steps.push((step, loss)) }, &|| false).unwrap();
    assert_eq!(steps.iter().map(|(s, _)| *s).collect::<Vec<_>>(), vec![1, 2, 3, 4, 5]);
    assert!(steps.iter().all(|(_, l)| l.is_finite()));
}
