//! One block of the 56B-A5B's shape, synthetic weights, on one device: the time and memory of a training pass
//! (forward, checkpoint replay, backward) and, with `AM2_PROFILE_OPS=1`, where the forward's time goes.
//!
//! ```text
//! cargo run --release --features cuda --example moe_block_bench -- <scratch-dir> [device cuda:1] [tokens 512] [conversations 3] [reps 3]
//! ```
//!
//! Qwen3-8B's block (hidden 4096, 32 heads over 8, head_dim 128, ff 12288) upcycled fine-grained (120 experts of 1,024
//! neurons, top-5, expert rank 4 with a shared A), rank 16 on q and v, a bf16 base. The vocabulary is small (512): the
//! head is not what is measured. About 0.4 GB of weights on the device, plus the pass.

use std::time::Instant;

use candle_core::{Device, Tensor};
use strata_workshop::model::{self, make_upcycled, BaseDtype, Ctx, LlamaModel, LoraSpec, MoeSpec, PassLayout, Target};
use strata_workshop::testmodel::TinyLlama;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let scratch = std::path::PathBuf::from(args.get(1).expect("a scratch folder"));
    let dev_name = args.get(2).cloned().unwrap_or_else(|| "cuda:1".into());
    let tokens: usize = args.get(3).and_then(|v| v.parse().ok()).unwrap_or(512);
    let convs: usize = args.get(4).and_then(|v| v.parse().ok()).unwrap_or(3);
    let reps: usize = args.get(5).and_then(|v| v.parse().ok()).unwrap_or(3);
    let layers: usize = std::env::var("BENCH_LAYERS").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
    let base = scratch.join(format!("dense-{layers}"));
    let up = scratch.join(format!("moe-{layers}"));
    if !up.join("config.json").exists() {
        std::fs::create_dir_all(&scratch)?;
        let t = TinyLlama { seed: 7, num_layers: layers, vocab_size: 512, hidden: 4096, intermediate: 12288, num_attention_heads: 32, num_key_value_heads: 8, shard_count: 1 };
        t.write(&base)?;
        make_upcycled(&base, &up, MoeSpec { experts: 120, top_k: 5, expert_ff: Some(1024), expert_rank: Some(4), shared_a: true, ..MoeSpec::full(120, 5) })?;
    }
    let dev = strata_workshop::finetune::open_devices(&[dev_name.clone()])?.remove(0);
    if std::env::var("BENCH_COPY").is_ok_and(|v| v == "1") {
        // one block's streamed bytes, host to device (pageable memory: a lower bound of the stream's pinned rate)
        let host = Tensor::zeros(193_000_000usize, candle_core::DType::BF16, &Device::Cpu)?;
        let _warm = host.narrow(0, 0, 1 << 20)?.to_device(&dev)?;
        dev.synchronize()?;
        let t = Instant::now();
        let on = host.to_device(&dev)?;
        dev.synchronize()?;
        let s = t.elapsed().as_secs_f64();
        println!("host to {dev_name}: {:.0} MB in {:.0} ms = {:.2} GB/s (pageable)", 386.0, s * 1e3, 0.386 / s);
        drop(on);
        return Ok(());
    }
    let mut m = LlamaModel::load(&up, BaseDtype::BF16, &dev)?;
    m.add_lora(&LoraSpec { r: 16, alpha: 32.0, dropout: 0.0, use_rslora: false, targets: vec![Target::Q, Target::V], train_layers: None, seed: 3 })?;
    m.set_checkpointing(true);
    // conversations of about equal length, packed into one sequence
    let lens: Vec<usize> = (0..convs).map(|i| tokens / convs + usize::from(i < tokens % convs)).collect();
    let layout = PassLayout { seqs: vec![lens.clone()] };
    let mut st = 1u64;
    let ids: Vec<u32> = (0..tokens).map(|_| { st = st.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); ((st >> 33) % 512) as u32 }).collect();
    let mut sup = Vec::new();
    for n in &lens {
        sup.extend((0..*n).map(|p| p >= n / 2));
    }
    let profile = std::env::var("AM2_PROFILE_OPS").is_ok_and(|v| v == "1");
    if std::env::var("AM2_GROUPED_LORA").is_ok_and(|v| v == "0") {
        model::GROUPED_LORA.store(false, std::sync::atomic::Ordering::Relaxed);
    }
    #[cfg(feature = "cuda")]
    if std::env::var("AM2_TF32").is_ok_and(|v| v == "1") {
        candle_core::cuda::set_gemm_reduced_precision_f32(true);
        println!("TF32 products on");
    }
    if std::env::var("AM2_HOST_GRADS").is_ok_and(|v| v == "0") {
        model::HOST_REPLAY_GRADS.store(false, std::sync::atomic::Ordering::Relaxed);
    }
    if std::env::var("AM2_OFFLOAD").is_ok_and(|v| v == "1") {
        model::OFFLOAD_CHECKPOINTS.store(true, std::sync::atomic::Ordering::Relaxed);
    }
    println!("offloaded checkpoints: {}", model::OFFLOAD_CHECKPOINTS.load(std::sync::atomic::Ordering::Relaxed));
    if std::env::var("BENCH_CHECK").is_ok_and(|v| v == "1") {
        // the loss and every adapter gradient, the old way (copy by copy, inputs on the card) and the new
        let run = |grouped: bool, offload: bool| -> anyhow::Result<(f32, Vec<(String, Vec<f32>)>)> {
            model::GROUPED_LORA.store(grouped, std::sync::atomic::Ordering::Relaxed);
            model::OFFLOAD_CHECKPOINTS.store(offload, std::sync::atomic::Ordering::Relaxed);
            let ctx = Ctx::train(0);
            let h = m.lower_packed(&ids, &layout)?;
            let loss = m.loss_packed(&h, &layout, &ids, &sup, &ctx)?;
            let g = loss.backward()?;
            let extra = m.take_checkpoint_grads();
            let mut out = Vec::new();
            for (name, v) in m.lora_vars() {
                let id = v.as_tensor().id();
                let mut t: Option<Tensor> = g.get(&v).map(|t| t.to_device(&Device::Cpu)).transpose()?;
                for (x, e) in &extra {
                    if x.as_tensor().id() == id {
                        let e = e.to_device(&Device::Cpu)?;
                        t = Some(match t { Some(a) => (a + e)?, None => e });
                    }
                }
                if let Some(t) = t {
                    out.push((name, t.flatten_all()?.to_vec1::<f32>()?));
                }
            }
            Ok((loss.to_scalar::<f32>()?, out))
        };
        let (l0, g0) = run(false, false)?;
        for (grouped, offload) in [(true, false), (false, true), (true, true)] {
            let (l, g) = run(grouped, offload)?;
            let scale = g0.iter().flat_map(|(_, v)| v.iter()).fold(0f32, |a, x| a.max(x.abs()));
            let d = g.iter().zip(&g0).map(|((na, a), (nb, b))| { assert_eq!(na, nb); a.iter().zip(b).fold(0f32, |m, (x, y)| m.max((x - y).abs())) }).fold(0f32, f32::max);
            println!("grouped {grouped} offload {offload}: loss {l} vs {l0} ({:.2e}), gradients {} tensors, max diff {d:.3e} of {scale:.3e} ({:.2e} relative)",
                     (l - l0).abs(), g.len(), d / scale.max(1e-30));
        }
        return Ok(());
    }
    println!("grouped expert LoRA: {}", model::GROUPED_LORA.load(std::sync::atomic::Ordering::Relaxed));
    let mut times = Vec::new();
    for rep in 0..reps + 1 {
        model::PROFILE_OPS.store(profile && rep > 0, std::sync::atomic::Ordering::Relaxed);
        let _ = m.memory_marks(true);
        dev.synchronize()?;
        let t0 = Instant::now();
        let ctx = Ctx::train(rep as u64);
        let h = m.lower_packed(&ids, &layout)?;
        let loss = m.loss_packed(&h, &layout, &ids, &sup, &ctx)?;
        let _v = loss.to_scalar::<f32>()?;
        dev.synchronize()?;
        let fwd = t0.elapsed().as_secs_f64();
        let t1 = Instant::now();
        let g = loss.backward()?;
        let extra = m.take_checkpoint_grads();
        dev.synchronize()?;
        let bwd = t1.elapsed().as_secs_f64();
        let marks = m.memory_marks(false);
        drop((g, extra, loss, h));
        m.synchronize()?;
        let now = m.memory_now();
        if rep > 0 {
            times.push((fwd, bwd));
        }
        println!("rep {rep}: forward {:.1} ms, backward (replay + gradients) {:.1} ms; high {:?} MiB used/reserved; after release {:?}", fwd * 1e3, bwd * 1e3,
                 marks.iter().map(|(u, r)| (u >> 20, r >> 20)).collect::<Vec<_>>(), now.iter().map(|(u, r)| (u >> 20, r >> 20)).collect::<Vec<_>>());
    }
    let n = times.len().max(1) as f64;
    let (f, b) = times.iter().fold((0.0, 0.0), |a, t| (a.0 + t.0, a.1 + t.1));
    println!("{layers} block(s), {tokens} tokens in {convs} conversations on {dev_name}: forward {:.1} ms, backward {:.1} ms, pass {:.1} ms (mean of {reps}{})",
             f / n * 1e3, b / n * 1e3, (f + b) / n * 1e3, if profile { ", profiled: slower" } else { "" });
    if profile {
        let ops = model::take_op_times();
        let total: f64 = ops.iter().filter(|(k, _)| !k.contains('.') || k.starts_with("attn")).map(|(_, v)| v.0).sum();
        for (k, (s, c)) in &ops {
            println!("  {k:<22} {:>9.1} ms {:>7} calls", s / n * 1e3, c / reps.max(1) as u64);
        }
        println!("  (top-level sections {:.1} ms a rep, forward and replay)", total / n * 1e3);
    }
    let _ = Tensor::zeros(1, candle_core::DType::F32, &Device::Cpu)?;
    Ok(())
}
