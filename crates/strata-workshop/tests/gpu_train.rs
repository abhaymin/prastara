//! Training on GPUs (docs/MULA-PLAN.md M2): the same run on the CPU and split over cuda:0 and cuda:1 gives the same
//! losses and the same adapter, on a looped chat model. Needs the `cuda` feature and two NVIDIA cards with a little
//! free memory: `cargo test --features cuda --test gpu_train -- --ignored`.
#![cfg(feature = "cuda")]

mod common;

use strata_workshop::finetune::{self, TrainSettings};
use strata_workshop::model::{make_looped, LoopSpec, LoraSpec, Target};

#[test]
#[ignore]
fn a_looped_model_trains_the_same_on_two_gpus_as_on_the_cpu() {
    // AM2_TEST_CHUNKED=1: every attention chunked (3 queries at a time), so its GPU side is checked too
    if std::env::var("AM2_TEST_CHUNKED").as_deref() == Ok("1") {
        strata_workshop::model::CHUNKED_ABOVE.store(4, std::sync::atomic::Ordering::Relaxed);
        strata_workshop::model::ATTENTION_CHUNK.store(3, std::sync::atomic::Ordering::Relaxed);
    }
    let (root, model, data) = common::model_and_data("gpu-train");
    let looped = root.join("looped");
    make_looped(&model, &looped, LoopSpec { start: 1, end: 3, passes: 2 }).unwrap();
    let run = |devices: &[&str], out: &str, ckpt: bool| {
        let mut s = TrainSettings::new(&looped, &data, root.join(out));
        s.lora = LoraSpec { r: 4, alpha: 8.0, targets: vec![Target::Q, Target::V, Target::Down], ..LoraSpec::default() };
        s.max_steps = Some(6);
        s.save_steps = 0;
        s.devices = devices.iter().map(|d| d.to_string()).collect();
        s.gradient_checkpointing = ckpt;
        let t0 = std::time::Instant::now();
        let r = finetune::train(&s, &mut |_| {}, &|| false).unwrap();
        println!("{devices:?}: {} steps in {:.2} s, losses {:?}", r.steps_done, t0.elapsed().as_secs_f64(), r.losses);
        let adapter = candle_core::safetensors::load(root.join(out).join(finetune::ADAPTER_FILE), &candle_core::Device::Cpu).unwrap();
        (r.losses, adapter)
    };
    let (cpu_l, cpu_a) = run(&[], "cpu", false);
    let (gpu_l, gpu_a) = run(&["cuda:0", "cuda:1"], "gpu", false);
    let (ck_l, ck_a) = run(&["cuda:0", "cuda:1"], "gpu-ckpt", true);
    assert!(cpu_a.contains_key("base_model.model.model.loop.gate.1"), "the loop's gate trains");
    let dl = cpu_l.iter().zip(&gpu_l).map(|(a, b)| (a - b).abs()).fold(0.0f64, f64::max);
    let mut da = 0.0f32;
    for (k, a) in &cpu_a {
        let b = &gpu_a[k];
        let d = (a - b).unwrap().abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap();
        da = da.max(d);
    }
    for (k, a) in &cpu_a {
        let d = (a - &ck_a[k]).unwrap().abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap();
        da = da.max(d);
    }
    let dl = cpu_l.iter().zip(&ck_l).map(|(a, b)| (a - b).abs()).fold(dl, f64::max);
    println!("losses differ by {dl:.2e}, adapters by {da:.2e} (two GPUs, with and without checkpointing)");
    assert!(cpu_l.iter().all(|l| l.is_finite()) && cpu_l.len() == 6);
    assert!(dl < 1e-4 && da < 1e-4, "losses {dl}, adapters {da}");
}
