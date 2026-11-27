//! Sequence packing and several sequences per streamed pass: a packed pass gives each conversation the loss and the
//! LoRA (and router, and loop-gate) gradients it has alone, MoE router losses included; N sequences in one pass give
//! the gradients of N passes; packing never splits a conversation, positions restart and no row sees another
//! conversation; a resumed packed run takes the same next batch and the same steps; an out-of-memory pass is retried
//! with fewer sequences and gives the same run.

mod common;

use std::cell::Cell;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use candle_core::{Device, Tensor, Var};
use strata_workshop::finetune::{self, Event, Resume, TrainReport, TrainSettings};
use strata_workshop::layer_stream::{StreamMode, StreamOptions};
use strata_workshop::model::{make_looped, make_upcycled, BaseDtype, Ctx, LlamaModel, LoopSpec, LoraSpec, ModelConfig, MoeSpec, PassLayout, Target};
use strata_workshop::packing::{pack, Schedule};

/// The tests share the trainer's global test hooks (out-of-memory injection, the chunking threshold).
static LOCK: Mutex<()> = Mutex::new(());

fn lock() -> std::sync::MutexGuard<'static, ()> {
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// `n` conversations of varied lengths: random tokens, the second half of each supervised.
fn conversations(vocab: usize, lens: &[usize], seed: u64) -> Vec<(Vec<u32>, Vec<bool>)> {
    let mut st = seed;
    lens.iter().map(|&n| {
        let t: Vec<u32> = (0..n).map(|_| (splitmix(&mut st) % vocab as u64) as u32).collect();
        let s: Vec<bool> = (0..n).map(|p| p >= n / 2 && p > 0).collect();
        (t, s)
    }).collect()
}

/// Every trained variable moved off its start (B is zero at the start: A would get no gradient), by the seed.
fn perturb(m: &LlamaModel, seed: u64) {
    let mut st = seed;
    for (_, v) in m.lora_vars() {
        let n = v.as_tensor().elem_count();
        let vals: Vec<f32> = (0..n).map(|_| ((splitmix(&mut st) % 2001) as f32 / 1000.0 - 1.0) * 0.3).collect();
        v.set(&Tensor::from_vec(vals, v.as_tensor().shape(), v.as_tensor().device()).unwrap()).unwrap();
    }
}

/// Each variable's gradient: the graph's and the checkpoint replays' parts summed.
fn grads_of(m: &LlamaModel, loss: &Tensor) -> BTreeMap<String, Vec<f32>> {
    let g = loss.backward().unwrap();
    let extra = m.take_checkpoint_grads();
    m.lora_vars().into_iter().filter_map(|(name, v): (String, Var)| {
        let id = v.as_tensor().id();
        let parts: Vec<Tensor> = g.get(&v).cloned().into_iter().chain(extra.iter().filter(|(x, _)| x.as_tensor().id() == id).map(|(_, t)| t.clone())).collect();
        let mut it = parts.into_iter();
        let first = it.next()?;
        let total = it.fold(first, |a, b| (a + b).unwrap());
        Some((name, total.flatten_all().unwrap().to_vec1::<f32>().unwrap()))
    }).collect()
}

fn add(acc: &mut BTreeMap<String, Vec<f32>>, g: BTreeMap<String, Vec<f32>>) {
    for (k, v) in g {
        let e = acc.entry(k).or_insert_with(|| vec![0.0; v.len()]);
        e.iter_mut().zip(v).for_each(|(a, b)| *a += b);
    }
}

/// The largest difference relative to the largest value.
fn rel_diff(a: &BTreeMap<String, Vec<f32>>, b: &BTreeMap<String, Vec<f32>>) -> f32 {
    assert_eq!(a.keys().collect::<Vec<_>>(), b.keys().collect::<Vec<_>>(), "the same variables got gradients");
    let scale = b.values().flatten().fold(0f32, |m, x| m.max(x.abs())).max(1e-6);
    a.iter().map(|(k, x)| x.iter().zip(&b[k]).map(|(p, q)| (p - q).abs()).fold(0.0, f32::max)).fold(0.0, f32::max) / scale
}

fn load(dir: &Path, stream: Option<&Path>) -> LlamaModel {
    let cfg = ModelConfig::load(dir).unwrap();
    let mut m = match stream {
        None => LlamaModel::load(dir, BaseDtype::F32, &Device::Cpu).unwrap(),
        Some(cache) => LlamaModel::load_streamed(dir, BaseDtype::F32, &[Device::Cpu], 0..cfg.num_hidden_layers, true, true,
                                                 &StreamOptions { mode: StreamMode::Ram, buffers: 2, cache_root: Some(cache.to_path_buf()) }, &mut |_| {}).unwrap(),
    };
    m.add_lora(&LoraSpec { r: 4, alpha: 8.0, dropout: 0.0, use_rslora: false, targets: Target::ALL.to_vec(), train_layers: None, seed: 3 }).unwrap();
    m.set_checkpointing(true);
    perturb(&m, 0x5EED);
    m
}

/// Each conversation alone vs packed as `layout_seqs` (indices into `convs`): the loss of the pack is the sum of the
/// conversations' own (router losses included), and so are the gradients.
fn packed_equals_alone(tag: &str, dir: &Path, stream: Option<&Path>, lens: &[usize], layout_seqs: &[Vec<usize>]) {
    let m = load(dir, stream);
    let convs = conversations(m.cfg.vocab_size, lens, 0xC0FFEE);
    let ctx_seed = 99;
    // alone, one by one
    let (mut loss_alone, mut g_alone, mut aux_alone) = (0.0f64, BTreeMap::new(), 0.0f64);
    for (t, s) in &convs {
        let ctx = Ctx::train(ctx_seed);
        let loss = m.loss_from_hidden(&m.lower(t).unwrap(), t, s, &ctx).unwrap();
        loss_alone += loss.to_scalar::<f32>().unwrap() as f64;
        aux_alone += ctx.aux_loss().unwrap_or(0.0);
        add(&mut g_alone, grads_of(&m, &loss));
    }
    // packed, all in one pass
    let layout = PassLayout { seqs: layout_seqs.iter().map(|s| s.iter().map(|i| lens[*i]).collect()).collect() };
    let order: Vec<usize> = layout_seqs.iter().flatten().copied().collect();
    let tokens: Vec<u32> = order.iter().flat_map(|i| convs[*i].0.clone()).collect();
    let sup: Vec<bool> = order.iter().flat_map(|i| convs[*i].1.clone()).collect();
    let ctx = Ctx::train(ctx_seed);
    let loss = m.loss_packed(&m.lower_packed(&tokens, &layout).unwrap(), &layout, &tokens, &sup, &ctx).unwrap();
    let loss_packed = loss.to_scalar::<f32>().unwrap() as f64;
    let g_packed = grads_of(&m, &loss);
    let dl = (loss_packed - loss_alone).abs() / loss_alone.abs().max(1.0);
    let dg = rel_diff(&g_packed, &g_alone);
    println!("{tag}: loss alone {loss_alone:.6} packed {loss_packed:.6} (rel {dl:.2e}); gradients rel {dg:.2e}; router {:?} vs {:.4}", ctx.aux_loss(), aux_alone / lens.len() as f64);
    assert!(dl < 1e-5, "{tag}: the packed loss {loss_packed} is not the sum {loss_alone}");
    assert!(dg < 1e-4, "{tag}: the gradients differ by {dg} (relative)");
    if m.cfg.moe.is_some() {
        let a = ctx.aux_loss().expect("a router loss");
        assert!((a - aux_alone / lens.len() as f64).abs() < 1e-5, "{tag}: the mean router loss {a} vs {}", aux_alone / lens.len() as f64);
    }
}

fn upcycle(root: &Path, base: &Path, m: MoeSpec, tag: &str) -> PathBuf {
    let out = root.join(tag);
    make_upcycled(base, &out, m).unwrap();
    out
}

const LENS: [usize; 6] = [9, 17, 5, 23, 12, 30];

/// Three ways to lay the six conversations: one sequence of all, two sequences, and every one alone in its own sequence.
fn layouts() -> Vec<Vec<Vec<usize>>> {
    vec![vec![vec![0, 1, 2, 3, 4, 5]], vec![vec![0, 1, 2], vec![3, 4, 5]], (0..6).map(|i| vec![i]).collect()]
}

#[test]
fn a_packed_pass_is_each_conversation_alone_dense_moe_and_looped() {
    let _g = lock();
    let root = common::temp_dir("pack-equal");
    let dense = root.join("dense");
    common::chat_model_sized(&dense, 3, 32, 64);
    let looped = root.join("looped");
    make_looped(&dense, &looped, LoopSpec { start: 1, end: 3, passes: 3 }).unwrap();
    let fine = upcycle(&root, &dense, MoeSpec { expert_ff: Some(16), shared_a: true, expert_rank: Some(2), ..MoeSpec::full(8, 2) }, "moe-fine");
    let full = upcycle(&root, &dense, MoeSpec::full(4, 2), "moe-full");
    let looped_moe = upcycle(&root, &looped, MoeSpec { expert_ff: Some(16), shared_a: false, expert_rank: Some(2), ..MoeSpec::full(8, 2) }, "moe-loop");
    for (tag, dir) in [("dense", &dense), ("looped", &looped), ("moe-fine", &fine), ("moe-full", &full), ("moe-looped", &looped_moe)] {
        for (k, l) in layouts().iter().enumerate() {
            packed_equals_alone(&format!("{tag}/{k}"), dir, None, &LENS, l);
        }
        // streamed: each block's weights fetched once for the whole pass, and again once in its replay
        packed_equals_alone(&format!("{tag}/streamed"), dir, Some(&root.join("shards")), &LENS, &layouts()[1]);
    }
}

/// Above the chunking threshold a long sequence attends a conversation at a time (and a long conversation by the
/// chunked rule): still each conversation alone.
#[test]
fn a_long_packed_sequence_goes_a_conversation_at_a_time() {
    let _g = lock();
    let root = common::temp_dir("pack-long");
    let dense = root.join("dense");
    common::chat_model_sized(&dense, 2, 32, 64);
    let fine = upcycle(&root, &dense, MoeSpec { expert_ff: Some(16), shared_a: true, expert_rank: Some(2), ..MoeSpec::full(8, 2) }, "moe-fine");
    let before = strata_workshop::model::CHUNKED_ABOVE.load(std::sync::atomic::Ordering::Relaxed);
    let chunk = strata_workshop::model::ATTENTION_CHUNK.load(std::sync::atomic::Ordering::Relaxed);
    strata_workshop::model::CHUNKED_ABOVE.store(20, std::sync::atomic::Ordering::Relaxed);
    strata_workshop::model::ATTENTION_CHUNK.store(8, std::sync::atomic::Ordering::Relaxed);
    for dir in [&dense, &fine] {
        packed_equals_alone("long", dir, None, &LENS, &layouts()[0]);
    }
    strata_workshop::model::CHUNKED_ABOVE.store(before, std::sync::atomic::Ordering::Relaxed);
    strata_workshop::model::ATTENTION_CHUNK.store(chunk, std::sync::atomic::Ordering::Relaxed);
}

#[test]
fn positions_restart_and_the_mask_keeps_conversations_apart() {
    let l = PassLayout { seqs: vec![vec![3, 2], vec![4]] };
    assert_eq!(l.positions(), vec![0, 1, 2, 0, 1, 0, 1, 2, 3]);
    assert_eq!(l.segments(), vec![(0, 3), (3, 2), (5, 4)]);
    assert_eq!(l.sequences(), vec![(0, 5), (5, 4)]);
    assert_eq!((l.rows(), l.conversations(), l.is_single()), (9, 3, false));
    let m = PassLayout::mask_values(&[3, 2], usize::MAX);
    let sees = |i: usize, j: usize| m[i * 5 + j] == 0.0;
    for i in 0..5 {
        for j in 0..5 {
            let same = (i < 3) == (j < 3);
            assert_eq!(sees(i, j), same && j <= i, "query {i} key {j}");
        }
    }
    // a window too
    let w = PassLayout::mask_values(&[4], 2);
    assert!(w[3 * 4 + 2] == 0.0 && w[3 * 4 + 1] == f32::NEG_INFINITY);

    // the hidden states of a packed sequence are each conversation's own: position 0 of the second conversation is
    // what a conversation starting there computes alone
    let _g = lock();
    let root = common::temp_dir("pack-states");
    let dense = root.join("dense");
    common::chat_model_sized(&dense, 2, 32, 64);
    let m = LlamaModel::load(&dense, BaseDtype::F32, &Device::Cpu).unwrap();
    let convs = conversations(m.cfg.vocab_size, &[7, 11], 5);
    let tokens: Vec<u32> = convs.iter().flat_map(|c| c.0.clone()).collect();
    let layout = PassLayout { seqs: vec![vec![7, 11]] };
    let packed = m.upper_hidden_packed(&m.lower_packed(&tokens, &layout).unwrap(), &layout, &Ctx::inference()).unwrap();
    let alone = m.upper_hidden(&m.lower(&convs[1].0).unwrap(), &Ctx::inference()).unwrap();
    let d = (packed.narrow(0, 7, 11).unwrap() - alone).unwrap().abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap();
    assert!(d < 1e-5, "the second conversation's states differ by {d}");
}

#[test]
fn packing_never_splits_and_the_schedule_is_a_pure_function() {
    let lens: Vec<usize> = (0..500).map(|i| 5 + (i * 7919) % 120).collect();
    let s = Schedule::new(&lens, 11, 2.5, 128, true);
    let again = Schedule::new(&lens, 11, 2.5, 128, true);
    assert_eq!(s, again, "the same order for the same seed");
    assert_ne!(s, Schedule::new(&lens, 12, 2.5, 128, true));
    let per = s.per_epoch();
    assert!(per < 500 && per >= lens.iter().sum::<usize>().div_ceil(128));
    // every epoch: each conversation exactly once, no sequence over the cap
    let mut m = 0;
    for _epoch in 0..2 {
        let mut seen = Vec::new();
        loop {
            let u = s.unit(m);
            let n: usize = u.iter().map(|i| lens[*i]).sum();
            assert!(n <= 128, "unit {m}: {n} tokens");
            seen.extend_from_slice(u);
            m += 1;
            if seen.len() == lens.len() {
                break;
            }
            assert!(seen.len() < lens.len(), "an epoch holds each conversation once");
        }
        seen.sort_unstable();
        assert_eq!(seen, (0..lens.len()).collect::<Vec<_>>());
    }
    // 2.5 epochs: two whole epochs and half the third's sequences
    assert!(s.total() >= m + 1 && s.total() <= m + per);
    // a conversation of the cap or more stays alone
    let packs = pack(&[200, 10, 10, 128], &[0, 1, 2, 3], 128);
    assert_eq!(packs, vec![vec![0], vec![1, 2], vec![3]]);
    // unpacked: the order an unpacked run has always had
    let u = Schedule::new(&lens, 11, 1.0, 128, false);
    let order = finetune::permutation(11, 1, lens.len());
    assert!((0..lens.len()).all(|m| u.unit(m) == [order[m]]));
    assert_eq!(u.total(), 500);
}

// ---------------------------------------------------------------------------------------------------------------------
// training runs
// ---------------------------------------------------------------------------------------------------------------------

fn rows(n: usize) -> Vec<serde_json::Value> {
    // conversations of varied lengths, so packing has something to do
    (0..n).map(|i| {
        let (thing, colour) = common::THINGS[i % common::THINGS.len()];
        let pad = " and so on".repeat(i % 5);
        serde_json::json!({"messages": [
            {"role": "user", "content": format!("What colour is {thing}{pad}?")},
            {"role": "assistant", "content": format!("It is {colour}.")},
        ]})
    }).collect()
}

fn settings(model: &Path, data: &Path, out: &Path) -> TrainSettings {
    let mut s = TrainSettings::new(model, data, out);
    s.lora = LoraSpec { r: 4, alpha: 8.0, dropout: 0.0, use_rslora: false, targets: Target::ALL.to_vec(), train_layers: None, seed: 7 };
    s.lr = 2e-2;
    s.epochs = 2.0;
    s.accumulation = 4;
    s.max_len = 160;
    s.val_split = 0.125;
    s.save_steps = 0;
    s.logging_steps = 1;
    s.seed = 11;
    s.gradient_checkpointing = true;
    s.pack = true;
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

fn max_abs(a: &BTreeMap<String, Vec<f32>>, b: &BTreeMap<String, Vec<f32>>) -> f32 {
    a.iter().map(|(k, x)| x.iter().zip(&b[k]).map(|(p, q)| (p - q).abs()).fold(0.0, f32::max)).fold(0.0, f32::max)
}

fn moe_and_data(tag: &str) -> (common::Tmp, PathBuf, PathBuf) {
    let root = common::temp_dir(tag);
    let (dense, data) = (root.join("dense"), root.join("train.jsonl"));
    common::chat_model_sized(&dense, 3, 32, 64);
    common::write_jsonl(&data, &rows(40));
    let looped = root.join("looped");
    make_looped(&dense, &looped, LoopSpec { start: 1, end: 3, passes: 2 }).unwrap();
    let moe = upcycle(&root, &looped, MoeSpec { expert_ff: Some(16), shared_a: true, expert_rank: Some(2), ..MoeSpec::full(8, 2) }, "moe");
    (root, moe, data)
}

/// The same packed run with 1, 2 and 4 sequences a pass, streamed or not: the same losses and adapter (the step's
/// gradients are the same sums, in another grouping).
#[test]
fn several_sequences_a_pass_train_the_same() {
    let _g = lock();
    let (root, model, data) = moe_and_data("pack-spp");
    let mut base = settings(&model, &data, &root.join("one"));
    base.max_steps = Some(4);
    let (a, logs) = run(&base);
    assert!(logs.iter().any(|l| l.starts_with("packing: ")), "{logs:?}");
    for (n, streamed) in [(2, false), (4, false), (4, true)] {
        let mut s = base.clone();
        s.out = root.join(format!("n{n}-{streamed}"));
        s.seqs_per_pass = n;
        if streamed {
            s.stream_layers = Some(StreamMode::Ram);
            s.stream_cache = Some(root.join("shards"));
        }
        let (b, logs) = run(&s);
        let dl = a.losses.iter().zip(&b.losses).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max);
        let da = max_abs(&adapter(&base.out), &adapter(&s.out));
        println!("{n} a pass (streamed {streamed}): losses {:?} vs {:?}, adapter {da:.2e}; eval {:?} vs {:?}", a.losses, b.losses, a.eval_loss, b.eval_loss);
        assert_eq!(a.losses.len(), b.losses.len());
        assert!(dl < 1e-5 && da < 1e-5, "{n} a pass: losses differ by {dl}, the adapter by {da}");
        assert!((a.eval_loss.unwrap() - b.eval_loss.unwrap()).abs() < 1e-5, "the held-out loss");
        assert!(logs.iter().any(|l| l.contains(&format!("{n} sequences per pass"))), "{logs:?}");
        if streamed {
            // one copy of each block forward and one in each replay a pass: fewer copies than one pass a sequence
            let st = b.stream.unwrap();
            assert!(st.loads > 0, "{st:?}");
        }
    }
}

/// A packed step is the unpacked step over the same conversations: at a vanishing rate the first packed step's loss
/// is the mean of its conversations' unpacked losses.
#[test]
fn a_packed_step_weighs_each_conversation_as_unpacked() {
    let _g = lock();
    let (root, model, data) = moe_and_data("pack-weigh");
    let mut s = settings(&model, &data, &root.join("packed"));
    s.lr = 1e-12;
    s.warmup_ratio = 0.0;
    s.max_steps = Some(1);
    s.accumulation = 2;
    let (packed, _) = run(&s);
    let d = finetune::prepare_data(&s).unwrap();
    let batch = finetune::step_batch(&s, &d, 0);
    let convs: Vec<usize> = batch.iter().flatten().copied().collect();
    assert!(convs.len() > batch.len(), "the step packs several conversations a sequence: {batch:?}");
    // each of those conversations alone, at the start (the same adapter: the same seed)
    let mut m = LlamaModel::load(&model, BaseDtype::F32, &Device::Cpu).unwrap();
    m.add_lora(&s.lora).unwrap();
    let mut sum = 0.0;
    for i in &convs {
        let ex = &d.train[*i];
        sum += m.loss(&ex.tokens, &ex.supervised, &Ctx::train(1)).unwrap().to_scalar::<f32>().unwrap() as f64;
    }
    let mean = sum / convs.len() as f64;
    println!("packed step loss {:.6}, the mean of its {} conversations alone {mean:.6}", packed.losses[0], convs.len());
    assert!((packed.losses[0] - mean).abs() < 1e-5, "{} vs {mean}", packed.losses[0]);
}

#[test]
fn a_resumed_packed_run_takes_the_same_batches_and_steps() {
    let _g = lock();
    let (root, model, data) = moe_and_data("pack-resume");
    let mut s = settings(&model, &data, &root.join("straight"));
    s.max_steps = Some(6);
    s.seqs_per_pass = 2;
    let (straight, _) = run(&s);

    let mut s2 = s.clone();
    s2.out = root.join("stopped");
    let calls = Cell::new(0);
    let first = finetune::train(&s2, &mut |_| {}, &|| { calls.set(calls.get() + 1); calls.get() == 3 }).unwrap();
    assert!(first.stopped && first.steps_done == 3);
    // the next batch, as the resumed run computes it, is the one the straight run took at step 4
    let d = finetune::prepare_data(&s2).unwrap();
    assert_eq!(finetune::step_batch(&s2, &d, 3), finetune::step_batch(&s, &finetune::prepare_data(&s).unwrap(), 3));
    assert!(!finetune::step_batch(&s2, &d, 3).is_empty());
    s2.resume = Some(Resume::Auto);
    let (second, logs) = run(&s2);
    assert!(logs.iter().any(|l| l.contains("resumed from")), "{logs:?}");
    assert_eq!(second.steps_done, 6);
    assert_eq!(second.losses, straight.losses, "every step's loss, before and after the stop");
    assert_eq!(max_abs(&adapter(&s.out), &adapter(&s2.out)), 0.0);
    // a checkpoint made packed is not continued unpacked
    let mut other = s2.clone();
    other.pack = false;
    other.resume = Some(Resume::Dir(s2.out.join("checkpoint-3")));
    let e = finetune::train(&other, &mut |_| {}, &|| false).unwrap_err().to_string();
    assert!(e.contains("other settings"), "{e}");
}

/// A pass of 4 sequences that runs out of device memory is retried as passes of 2 (and stays at 2): the same run.
#[test]
fn an_out_of_memory_pass_is_retried_with_fewer_sequences() {
    let _g = lock();
    let (root, model, data) = moe_and_data("pack-oom");
    let mut s = settings(&model, &data, &root.join("clean"));
    s.max_steps = Some(3);
    s.seqs_per_pass = 4;
    let (clean, _) = run(&s);
    let mut o = s.clone();
    o.out = root.join("oom");
    // micro-batch 5 is in step 2's pass of four (units 4..8)
    finetune::INJECT_OOM_TIMES.store(1, std::sync::atomic::Ordering::Relaxed);
    finetune::INJECT_OOM.store(5, std::sync::atomic::Ordering::Relaxed);
    let (r, logs) = run(&o);
    finetune::INJECT_OOM.store(usize::MAX, std::sync::atomic::Ordering::Relaxed);
    let line = logs.iter().find(|l| l.contains("out of device memory")).unwrap_or_else(|| panic!("{logs:?}"));
    assert!(line.contains("(a pass of 4)") && line.contains("with 2 sequences per pass instead of 4"), "{line}");
    let dl = clean.losses.iter().zip(&r.losses).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max);
    let da = max_abs(&adapter(&s.out), &adapter(&o.out));
    assert!(dl < 1e-5 && da < 1e-5, "losses differ by {dl}, the adapter by {da}");
    let report: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(o.out.join(finetune::REPORT_FILE)).unwrap()).unwrap();
    assert_eq!(report["packing"]["seqs_per_pass"], 2, "{report}");
}

/// A single packed sequence that runs out of device memory is retried as two passes, its conversations split in
/// halves: the same sums, so the same run.
#[test]
fn an_out_of_memory_sequence_is_retried_in_halves() {
    let _g = lock();
    let (root, model, data) = moe_and_data("pack-oom-split");
    let mut s = settings(&model, &data, &root.join("clean"));
    s.max_steps = Some(2);
    s.seqs_per_pass = 1;
    let (clean, _) = run(&s);
    let mut o = s.clone();
    o.out = root.join("oom");
    finetune::INJECT_OOM_TIMES.store(1, std::sync::atomic::Ordering::Relaxed);
    finetune::INJECT_OOM.store(5, std::sync::atomic::Ordering::Relaxed);
    let (r, logs) = run(&o);
    finetune::INJECT_OOM.store(usize::MAX, std::sync::atomic::Ordering::Relaxed);
    let line = logs.iter().find(|l| l.contains("out of device memory")).unwrap_or_else(|| panic!("{logs:?}"));
    assert!(line.contains("micro-step 5") && line.contains("as two passes of"), "{line}");
    let dl = clean.losses.iter().zip(&r.losses).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max);
    let da = max_abs(&adapter(&s.out), &adapter(&o.out));
    println!("{line}\nlosses {:?} vs {:?}", clean.losses, r.losses);
    assert!(dl < 1e-5 && da < 1e-5, "losses differ by {dl}, the adapter by {da}");
}
