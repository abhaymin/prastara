//! A micro-step that runs out of device memory is dropped whole and run once more after the pools are trimmed: the
//! run then takes exactly the steps it would have taken (nothing counted twice: the loss, the adapters' gradients,
//! those the checkpoint replays left behind), and says so in the log; a second failure ends the run. The failure is
//! injected (`finetune::INJECT_OOM`), on the CPU. On the GPUs (ignored): trimming the pools at block boundaries changes
//! nothing of the arithmetic and does keep the reserve near the use. Their own test binary: the hooks are
//! process-wide.

mod common;

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::atomic::Ordering;

use candle_core::Device;
use strata_workshop::finetune::{self, Event, TrainReport, TrainSettings, INJECT_OOM, INJECT_OOM_TIMES};
use strata_workshop::layer_stream::StreamMode;
use strata_workshop::model::{LoraSpec, Target};

fn settings(model: &Path, data: &Path, out: &Path) -> TrainSettings {
    let mut s = TrainSettings::new(model, data, out);
    s.lora = LoraSpec { r: 4, alpha: 8.0, dropout: 0.1, use_rslora: false, targets: Target::ALL.to_vec(), train_layers: None, seed: 7 };
    s.lr = 2e-2;
    s.epochs = 1.0;
    s.max_steps = Some(4);
    s.accumulation = 3;
    s.max_len = 64;
    s.save_steps = 0;
    s.seed = 11;
    s.gradient_checkpointing = true;
    s
}

fn run(s: &TrainSettings) -> anyhow::Result<(TrainReport, Vec<String>)> {
    let mut logs = Vec::new();
    let r = finetune::train(s, &mut |e| if let Event::Log(l) = e { logs.push(l) }, &|| false)?;
    Ok((r, logs))
}

fn adapter(dir: &Path) -> BTreeMap<String, Vec<f32>> {
    candle_core::safetensors::load(dir.join(finetune::ADAPTER_FILE), &Device::Cpu).unwrap().into_iter()
        .map(|(k, v)| (k, v.flatten_all().unwrap().to_vec1::<f32>().unwrap())).collect()
}

#[test]
fn an_out_of_memory_micro_step_is_retried_once_and_counted_once() {
    let (root, model, data) = common::model_and_data("oom-retry");
    for (tag, stream) in [("held", None), ("streamed", Some(StreamMode::Ram))] {
        let mut base = settings(&model, &data, &root.join(format!("{tag}-plain")));
        base.stream_layers = stream;
        let (a, _) = run(&base).unwrap();
        // micro-step 4: the second of step 2, after the first one's gradients are already in the step's sum
        let mut hit = base.clone();
        hit.out = root.join(format!("{tag}-retried"));
        INJECT_OOM_TIMES.store(1, Ordering::SeqCst);
        INJECT_OOM.store(4, Ordering::SeqCst);
        let (b, logs) = run(&hit).unwrap();
        assert_eq!(INJECT_OOM.load(Ordering::SeqCst), usize::MAX, "the failure was injected");
        assert_eq!(a.losses, b.losses, "{tag}: the same losses, bit for bit");
        assert_eq!(a.tokens_seen, b.tokens_seen, "{tag}: the retried conversation counted once");
        assert_eq!(adapter(&base.out), adapter(&hit.out), "{tag}: the same adapter");
        assert!(logs.iter().any(|l| l.contains("micro-step 4: out of device memory") && l.contains("trying it once more")), "{logs:?}");
        // twice in a row: the run fails, as it did before the retry
        let mut twice = base.clone();
        twice.out = root.join(format!("{tag}-twice"));
        INJECT_OOM_TIMES.store(2, Ordering::SeqCst);
        INJECT_OOM.store(4, Ordering::SeqCst);
        let e = run(&twice).err().expect("a second failure ends the run");
        assert!(finetune::is_out_of_memory(&e), "{e:#}");
        INJECT_OOM.store(usize::MAX, Ordering::SeqCst);
    }
    assert!(!finetune::is_out_of_memory(&anyhow::anyhow!("the loss became NaN")));
}

/// On the GPUs (`-- --ignored --test-threads=1`; AM2_TEST_DEVICES=cuda:1 or AM2_TEST_GPUS=2): a streamed bf16 looped
/// MoE over conversations of varied length, with the block-boundary trim off, at its default gap and at every block:
/// the same losses; with it, each micro-step's reserve stays nearer its use. Prints the pool's release threshold.
#[cfg(feature = "cuda")]
#[test]
#[ignore]
fn on_the_gpus_block_boundary_trims_keep_the_steps_and_the_reserve_down() {
    use strata_workshop::model::{make_looped, make_upcycled, LoopSpec, MoeSpec, TRIMS, TRIM_GAP};
    let devices: Vec<String> = match (std::env::var("AM2_TEST_DEVICES"), std::env::var("AM2_TEST_GPUS").as_deref()) {
        (Ok(l), _) => l.split(',').map(|d| d.trim().to_string()).collect(),
        (_, Ok("2")) => vec!["cuda:0".into(), "cuda:1".into()],
        _ => vec!["cuda:0".into()],
    };
    let root = common::temp_dir("trim-gpu");
    let (base, data) = (root.join("base"), root.join("train.jsonl"));
    common::chat_model_sized(&base, 8, 512, 2048);
    let rows: Vec<serde_json::Value> = (0..32).map(|i| {
        let (thing, colour) = common::THINGS[i % common::THINGS.len()];
        let story = format!("The {thing} is {colour}. ").repeat(2 + (i * 7) % 40);
        serde_json::json!({"messages": [{"role": "user", "content": format!("{story} What colour is {thing}?")}, {"role": "assistant", "content": format!("It is {colour}. {story}")}]})
    }).collect();
    common::write_jsonl(&data, &rows);
    let looped = root.join("looped");
    make_looped(&base, &looped, LoopSpec { start: 2, end: 6, passes: 2 }).unwrap();
    let moe = root.join("moe");
    make_upcycled(&looped, &moe, MoeSpec { expert_ff: Some(512), shared_a: true, expert_rank: Some(2), ..MoeSpec::full(16, 4) }).unwrap();
    let mut results = Vec::new();
    for (tag, gap) in [("no trim", u64::MAX - 1), ("default gap", strata_workshop::model::DEFAULT_TRIM_GAP), ("every block", 0)] {
        TRIM_GAP.store(gap, Ordering::SeqCst);
        let t0 = TRIMS.load(Ordering::SeqCst);
        let mut s = settings(&moe, &data, &root.join(format!("out-{}", gap.min(1 << 40))));
        s.devices = devices.clone();
        s.base_dtype = strata_workshop::model::BaseDtype::BF16;
        s.stream_layers = Some(StreamMode::Ram);
        s.max_len = 1024;
        s.max_steps = Some(4);
        s.accumulation = 4;
        s.lr = 2e-4;
        let trace = s.out.join("memory-trace.log");
        let (r, _) = run(&s).unwrap();
        let text = std::fs::read_to_string(&trace).unwrap_or_default();
        // the largest reserve above use over the micro-steps, per device (MiB)
        let mut gaps: Vec<i64> = Vec::new();
        for l in text.lines() {
            for (d, part) in l.split(" | ").skip(1).enumerate() {
                let nums: Vec<i64> = part.split_whitespace().filter_map(|w| w.parse().ok()).collect();
                if nums.len() >= 2 {
                    if gaps.len() <= d { gaps.push(0); }
                    gaps[d] = gaps[d].max(nums[1] - nums[0]);
                }
            }
        }
        let thr: Vec<Option<u64>> = devices.iter().map(|d| {
            let dev = Device::new_cuda(d[5..].parse().unwrap()).unwrap();
            strata_workshop::model::pool_release_threshold(&dev, None)
        }).collect();
        println!("{tag}: trims {}, reserve above use at most {gaps:?} MiB, release threshold {thr:?}, losses {:?}", TRIMS.load(Ordering::SeqCst) - t0, r.losses);
        results.push((r.losses, TRIMS.load(Ordering::SeqCst) - t0, gaps));
    }
    TRIM_GAP.store(u64::MAX, Ordering::SeqCst);
    assert_eq!(results[0].0, results[2].0, "trimming changes no arithmetic");
    assert_eq!(results[0].0, results[1].0);
    assert!(results[0].1 == 0 && results[2].1 > 0);
    // summed over the cards, one pool chunk (32 MiB) of noise allowed: the reserve is a high-water mark
    let (none, every): (i64, i64) = (results[0].2.iter().sum(), results[2].2.iter().sum());
    assert!(every <= none + 32, "trimming at every block keeps the reserve nearer the use: {:?}", results.iter().map(|r| &r.2).collect::<Vec<_>>());
}
