//! On the GPUs (`-- --ignored`; AM2_TEST_DEVICES=cuda:1): the device-memory peak of a micro-step with gradient
//! accumulation is the peak of the same micro-step without it plus one set of the step's gradient sums (the sums the
//! later micro-steps add into), not more. Each micro-step `m` trains on the same conversation with the same dropout
//! whatever the accumulation, so the two memory traces compare micro-step by micro-step. Uses only what the trainer
//! has had since layer streaming came in, so the same file measures older versions too. The conversations are short
//! and the adapter large, so the moment the gradients are added up is each micro-step's peak: a trainer that held a
//! second set there (the micro-step's own sums beside the step's) shows two sets more, not one.

#![cfg(feature = "cuda")]

mod common;

use std::path::Path;

use strata_workshop::finetune::{self, TrainSettings};
use strata_workshop::layer_stream::StreamMode;
use strata_workshop::model::{BaseDtype, LoraSpec, Target};

fn used_high(trace: &Path) -> Vec<Vec<u64>> {
    std::fs::read_to_string(trace).unwrap().lines().map(|l| {
        l.split(" | ").skip(1).map(|part| part.split_whitespace().filter_map(|w| w.parse::<u64>().ok()).next().unwrap()).collect()
    }).collect()
}

#[test]
#[ignore]
fn on_the_gpus_accumulation_holds_one_set_of_sums_more() {
    let devices: Vec<String> = std::env::var("AM2_TEST_DEVICES").map(|l| l.split(',').map(|d| d.trim().to_string()).collect()).unwrap_or_else(|_| vec!["cuda:0".into()]);
    let root = common::temp_dir("mem-peak");
    let (model, data) = (root.join("model"), root.join("train.jsonl"));
    common::chat_model_sized(&model, 8, 512, 2048);
    let rows: Vec<serde_json::Value> = (0..24).map(|i| {
        let (thing, colour) = common::THINGS[i % common::THINGS.len()];
        // short conversations: the activations stay below the gradient sets, so the adding-up is the peak
        let story = format!("The {thing} is {colour}. ").repeat(1 + i % 3);
        serde_json::json!({"messages": [{"role": "user", "content": format!("{story} What colour is {thing}?")}, {"role": "assistant", "content": format!("It is {colour}.")}]})
    }).collect();
    common::write_jsonl(&data, &rows);
    let run = |acc: usize, steps: usize, tag: &str| {
        let mut s = TrainSettings::new(&model, &data, &root.join(tag));
        // a large adapter (about 100 MB of f32 gradients), so one set of sums stands out of the noise
        s.lora = LoraSpec { r: 256, alpha: 256.0, dropout: 0.1, use_rslora: false, targets: Target::ALL.to_vec(), train_layers: None, seed: 7 };
        s.devices = devices.clone();
        s.base_dtype = BaseDtype::BF16;
        s.stream_layers = Some(StreamMode::Ram);
        s.gradient_checkpointing = true;
        s.max_len = 128;
        s.accumulation = acc;
        s.max_steps = Some(steps);
        s.val_split = 0.0;
        s.eval_data = None;
        s.save_steps = 0;
        s.lr = 2e-5;
        let r = finetune::train(&s, &mut |_| {}, &|| false).unwrap();
        (used_high(&s.out.join("memory-trace.log")), r.trainable_params)
    };
    let (one, params) = run(1, 12, "acc1");
    let (four, _) = run(4, 3, "acc4");
    let set = params * 4;
    assert_eq!(one.len(), 12);
    assert_eq!(four.len(), 12);
    let mut worst: i64 = i64::MIN;
    for m in 0..12 {
        for d in 0..one[m].len() {
            let extra = (four[m][d] as i64 - one[m][d] as i64) << 20;
            if m % 4 != 0 {
                worst = worst.max(extra);
            }
        }
    }
    println!("one set of gradient sums: {} MiB; a later micro-step's peak with accumulation is at most {} MiB above the same micro-step's without; \
              used-high per micro-step (MiB): accumulation 1 {:?}, accumulation 4 {:?}", set >> 20, worst >> 20, one, four);
    assert!(worst <= (set + set / 4 + (8 << 20)) as i64, "accumulating holds {} MiB more than one set of sums ({} MiB)", worst >> 20, set >> 20);
}
