//! `profile` and `autopilot`: the arithmetic against numbers worked out by hand, for the tiny fixture's config and for
//! Llama-2-7B's (config only — no weights are read anywhere here).

use std::path::PathBuf;

use strata_workshop::model::{BaseDtype, LoraSpec, ModelConfig, Target};
use strata_workshop::profile::{self, AutopilotInput, Machine, ProfileInput};

fn tiny() -> ModelConfig {
    ModelConfig::load(&PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny-llama-hf")).unwrap()
}

fn lora(r: usize, train_layers: Option<usize>) -> LoraSpec {
    LoraSpec { r, alpha: 2.0 * r as f64, dropout: 0.0, use_rslora: false, targets: vec![Target::Q, Target::V], train_layers, seed: 0 }
}

fn input(cfg: ModelConfig, r: usize, train_layers: Option<usize>, dtype: BaseDtype, max_len: usize, cache: bool) -> ProfileInput {
    ProfileInput { cfg, lora: lora(r, train_layers), base_dtype: dtype, max_len, accumulation: 4, cache, examples: 20, epochs: 3.0, avg_tokens: 12, checkpointing: false, devices: 1, stream: None, batch: Default::default() }
}

#[test]
fn the_tiny_model_by_hand() {
    // hidden 16, intermediate 32, 2 blocks, vocab 32, 2 heads of 8, head not tied
    let cfg = tiny();
    // a block: q, k, v, o 16×16 = 4 · 256, gate and up 32×16, down 16×32 = 3 · 512 → 2560, and two norms of 16
    assert_eq!(cfg.block_params(), 2560 + 32);
    // embeddings 32·16 + 2 blocks + final norm 16 + head 32·16
    assert_eq!(cfg.total_params(), 512 + 2 * 2592 + 16 + 512);

    let e = profile::estimate(&input(cfg.clone(), 4, None, BaseDtype::F32, 16, false));
    assert_eq!(e.total_params, 6224);
    // rank 4 on q and v (16 → 16): 4 · (16 + 16) each, two targets, two blocks
    assert_eq!(e.trainable_params, 512);
    assert_eq!(e.weights_bytes, 6224 * 4);
    assert_eq!((e.adapter_bytes, e.optimizer_bytes, e.gradient_bytes), (2048, 4096, 4096));
    // kept per token and block: 10·16 + 11.5·16 + 8.5·16 + 4·32 + 2·(2 heads · 16 tokens) + 2 adapters · (4 + 3·16) = 776 values
    assert_eq!(profile::graph_floats_per_token(&cfg, &lora(4, None), 16), 776.0);
    // × 4 bytes × 16 tokens × 2 blocks, twice (the gradients coming back) = 198 656; the head's sliced loss: 3 tensors of 16 × 32 values = 6 144
    assert_eq!(e.activation_bytes, 198_656 + 6_144);
    assert_eq!((e.widening_bytes, e.fill_bytes, e.cache_disk_bytes), (0, 0, 0));
    assert_eq!(e.total_bytes, 24_896 + 10_240 + 204_800);
    // 20 conversations × 3 epochs / 4 per step
    assert_eq!(e.steps, 15);
    // per token: a block forward is 2 · 2560 + 4 · 16 · 12 = 5888 operations, the head 2 · 512; forward and backward ≈ 3 ×;
    // 60 passes of 12 tokens: 12 · 60 · 3 · (2 · 5888 + 1024)
    assert_eq!(e.flops, 27_648_000.0);

    // held in 16 bits: the matrices (2 · 2560 + 2 · 512 values) at 2 bytes, the norms (2 · 32 + 16 values) at 4;
    // and the largest matrix (512 values, three ways) widened to f32 for its product
    let h = profile::estimate(&input(cfg.clone(), 4, None, BaseDtype::F16, 16, false));
    assert_eq!(h.weights_bytes, 6144 * 2 + 80 * 4);
    assert_eq!(h.widening_bytes, 512 * 4);

    // the top block only, no cache: half the adapters and half the graph, plus one block's worth for the frozen part
    // (776 − 104 adapter values = 672 per token, kept once: 4 · 16 · 672)
    let top = profile::estimate(&input(cfg.clone(), 4, Some(1), BaseDtype::F32, 16, false));
    assert_eq!((top.trainable_params, top.trained_blocks), (256, 1));
    assert_eq!(top.activation_bytes, 99_328 + 6_144 + 43_008);
    assert_eq!(top.total_bytes, 24_896 + 5 * 1024 + 148_480);
    // the frozen block forward only: 12 · (60 · 5888 + 60 · 3 · (5888 + 1024))
    assert_eq!(top.flops, 19_169_280.0);

    // with the cache: only the trained block, the head and their norms stay; the frozen block runs once per
    // conversation (20 passes, not 60); the states take 20 × 12 tokens × 16 values × 4 bytes on disk
    let cached = profile::estimate(&input(cfg, 4, Some(1), BaseDtype::F32, 16, true));
    assert_eq!(cached.weights_bytes, (2560 + 512) * 4 + (32 + 16) * 4);
    assert_eq!(cached.activation_bytes, 99_328 + 6_144);
    assert_eq!(cached.total_bytes, 12_480 + 5 * 1024 + 105_472);
    // while filling: one block (2560 · 4 + 32 · 4, more than the embeddings' 2048) and one conversation passing it
    assert_eq!(cached.fill_bytes, 10_368 + 43_008);
    assert_eq!(cached.cache_disk_bytes, 15_360);
    assert_eq!(cached.flops, 12.0 * (20.0 * 5888.0 + 60.0 * 3.0 * 6912.0));
}

#[test]
fn llama_2_7b_by_hand() {
    let cfg = profile::preset("llama-2-7b").unwrap();
    // the well-known count
    assert_eq!(cfg.total_params(), 6_738_415_616);
    // a block: 4 · 4096² attention + 3 · 4096 · 11008 MLP + two norms
    assert_eq!(cfg.block_params(), 67_108_864 + 135_266_304 + 8192);

    let e = profile::estimate(&input(cfg.clone(), 16, None, BaseDtype::F32, 1024, false));
    // PEFT's figure for r 16 on q_proj and v_proj: 16 · 8192 · 2 · 32
    assert_eq!(e.trainable_params, 8_388_608);
    assert_eq!(e.weights_bytes, 26_953_662_464);
    // per token and block: 10·4096 + 11.5·4096 + 8.5·4096 + 4·11008 + 2·32·1024 + 2·(16 + 3·4096) = 257 056 values
    assert_eq!(profile::graph_floats_per_token(&cfg, &lora(16, None), 1024), 257_056.0);
    // 8 bytes (value and gradient) · 1024 tokens · 257 056 · 32 blocks, and the head's sliced loss, 3 · 1024 · 32000 · 4 (under the 32 M-logit slice)
    assert_eq!(e.activation_bytes, 67_385_688_064 + 393_216_000);
    assert_eq!(e.total_bytes, 26_953_662_464 + 167_772_160 + 67_778_904_064);

    // the limited-resource plan: 16-bit base, top 2 blocks, cache — 6.6 GB where the f32 run of every block is 95 GB
    let small = profile::estimate(&input(cfg.clone(), 16, Some(2), BaseDtype::F16, 1024, true));
    assert_eq!(small.trainable_params, 524_288);
    // two blocks' matrices and the head at 2 bytes; two blocks' norms and the final one at 4
    assert_eq!(small.weights_bytes, (2 * 202_375_168 + 131_072_000) * 2 + (2 * 8192 + 4096) * 4);
    assert_eq!(small.activation_bytes, 2 * 2_105_802_752 + 393_216_000);
    // the embeddings (32000 × 4096, 131 M values) are the largest matrix, widened 64 M values at a time
    assert_eq!(small.widening_bytes, 268_435_456);
    assert_eq!(small.total_bytes, 1_071_726_592 + 10_485_760 + 4_604_821_504 + 268_435_456);
    // filling the cache: one block in 16 bits (404 783 104, more than the embeddings' 262 144 000), a conversation
    // passing it (4 · 1024 · 232 448) and the widened matrix
    assert_eq!(small.fill_bytes, 404_783_104 + 952_107_008 + 268_435_456);
    assert!(small.fill_bytes < small.total_bytes);

    // 16 bits alone, every block: the matrices at 2 bytes, 266 240 norm values at 4
    let half = profile::estimate(&input(cfg, 16, None, BaseDtype::F16, 1024, false));
    assert_eq!(half.weights_bytes, 6_738_149_376 * 2 + 266_240 * 4);
}

fn machine(ram_gb: u64) -> Machine {
    Machine { ram_total: 64 << 30, ram_available: ram_gb << 30, cpus: 16, gpus: vec![], cpu_gflops: 100.0 }
}

#[test]
fn the_verdict_compares_with_free_memory() {
    let cfg = profile::preset("llama-2-7b").unwrap();
    let p = input(cfg, 16, Some(2), BaseDtype::F16, 1024, true);
    let e = profile::estimate(&p);
    // 5 955 469 312 + 10 % + 512 MiB
    assert_eq!(profile::need_bytes(&e), 5_955_469_312 + 595_546_931 + (512 << 20));
    assert_eq!(profile::verdict(&e, &machine(16)).fits_ram, Some(true));
    assert_eq!(profile::verdict(&e, &machine(6)).fits_ram, Some(false));
    // a system that does not say how much is free is not refused
    let mut unknown = machine(0);
    unknown.ram_total = 0;
    assert_eq!(profile::verdict(&e, &unknown).fits_ram, None);
    // a GPU with 15 GB free would hold it; one with 4 GB would not
    let mut m = machine(16);
    m.gpus = profile::parse_nvidia_smi("0, NVIDIA GeForce RTX 5060 Ti, 16311, 15360\n1, NVIDIA GeForce RTX 5060 Ti, 16311, 4096\n");
    assert_eq!(m.gpus[1].free_bytes, 4096 << 20);
    assert_eq!(profile::verdict(&e, &m).fits_gpu, vec![(0, true), (1, false)]);
    // time: operations over (GFLOP/s × the share training reaches)
    assert_eq!(profile::cpu_seconds(&e, 100.0), e.flops / (100.0 * profile::cpu_training_efficiency(BaseDtype::F16) * 1e9));
    let lines = profile::report_lines(&p, &e, &m).join("\n");
    assert!(lines.contains("524288 adapter parameters") && lines.contains("fits: needs") && lines.contains("would not fit") && lines.contains("measured on a two-card"), "{lines}");
    assert_eq!(profile::parse_meminfo("MemTotal:       61712345 kB\nMemFree: 1 kB\nMemAvailable:   20000000 kB\n"), (61712345 * 1024, 20000000 * 1024));
}

#[test]
fn the_time_estimate_depends_on_how_the_base_is_held() {
    // the same work: f16 ran twice as long as f32 and bf16 on this CPU (docs/DEPLOYMENT.md), and the estimate says so
    let at = |dtype| profile::cpu_seconds(&profile::estimate(&input(tiny(), 4, None, dtype, 16, false)), 100.0);
    let (f32_s, bf16_s, f16_s) = (at(BaseDtype::F32), at(BaseDtype::BF16), at(BaseDtype::F16));
    assert_eq!(f32_s, 27_648_000.0 / (100.0 * 0.6 * 1e9));
    assert_eq!(bf16_s, f32_s, "the same operations at the same speed");
    assert!((f16_s / f32_s - 2.0).abs() < 1e-12, "f16 {f16_s} s against f32 {f32_s} s");
}

#[test]
fn the_autopilot_fits_the_machine_and_says_why() {
    let big = profile::preset("llama-2-7b").unwrap();
    let ask = |cfg: &ModelConfig, ram_gb: u64, hours: f64, examples: usize| profile::autopilot(&AutopilotInput {
        cfg: cfg.clone(), source_dtype: BaseDtype::F32, examples, p95_tokens: 400, avg_tokens: 250, budget_bytes: ram_gb << 30,
        time_budget_seconds: hours * 3600.0, cpu_gflops: 100.0,
    }).unwrap();

    // a tiny model on a large machine: nothing is held back
    let easy = ask(&tiny(), 32, 1.0, 200);
    assert_eq!((easy.base_dtype, easy.lora.train_layers, easy.cache, easy.lora.r, easy.epochs), (BaseDtype::F32, None, false, 8, 5.0));
    assert_eq!(easy.max_len, 440, "the 95th percentile and a tenth more");
    assert_eq!(easy.accumulation, 16, "200 / 8, capped at 16");
    assert!(easy.fits_memory && easy.fits_time);

    // a 7B model in 16 GB of RAM, on a CPU, with 200 hours (the frozen blocks alone take about 53 of them at this
    // speed): 16 bits (f32 alone is 27 GB), fewer blocks, the cache — and the plan really fits
    let tight = ask(&big, 16, 200.0, 2000);
    assert_eq!(tight.base_dtype, BaseDtype::F16);
    let n = tight.lora.train_layers.expect("not every block");
    assert!(n < 32 && tight.cache);
    assert!(tight.fits_memory, "{:?}", tight.choices);
    assert!(profile::need_bytes(&tight.estimate) <= 16 << 30);
    assert_eq!((tight.lora.r, tight.lora.alpha, tight.lr, tight.epochs), (16, 32.0, 2e-4, 3.0));
    // the largest number of blocks that fits: twice as many would not
    let twice = profile::estimate(&ProfileInput { cfg: big.clone(), lora: LoraSpec { train_layers: Some(n * 2), ..tight.lora.clone() }, base_dtype: tight.base_dtype,
                                                  max_len: tight.max_len, accumulation: tight.accumulation, cache: true, examples: 2000, epochs: tight.epochs, avg_tokens: 250, checkpointing: false, devices: 1, stream: None, batch: Default::default() });
    assert!(profile::need_bytes(&twice) > 16 << 30 || profile::cpu_seconds(&twice, 100.0) > 200.0 * 3600.0);

    // every choice has its one-line reason
    let settings: Vec<&str> = tight.choices.iter().map(|c| c.setting).collect();
    for s in ["rank", "alpha", "targets", "learning rate", "accumulation", "base dtype", "sequence length", "trained blocks", "activation cache", "epochs"] {
        assert!(settings.contains(&s), "no reason given for {s}");
    }
    assert!(tight.choices.iter().all(|c| !c.why.is_empty() && !c.why.contains('\n')));
    let why = |s: &str| tight.choices.iter().find(|c| c.setting == s).unwrap().why.clone();
    assert!(why("base dtype").contains("f32 would need"), "{}", why("base dtype"));
    assert!(why("trained blocks").contains("all 32 would need"), "{}", why("trained blocks"));

    // a time budget too short for anything: said, with the closest plan (one block, one epoch), not hidden
    let rushed = ask(&big, 16, 0.01, 2000);
    assert!(!rushed.fits_time && rushed.fits_memory);
    assert_eq!((rushed.lora.train_layers, rushed.epochs), (Some(1), 1.0));
    assert!(rushed.choices.iter().any(|c| c.why.contains("longer than the time budget")));
    // memory too small for even the smallest plan: said too
    let starved = ask(&big, 1, 200.0, 2000);
    assert!(!starved.fits_memory);
    assert_eq!(starved.max_len, 128, "the sequence was shortened as far as it goes");
}

#[test]
fn the_cpu_benchmark_gives_a_number() {
    // not a speed test: only that the measurement runs and is sane
    let g = profile::measure_cpu_gflops();
    assert!(g >= 0.1 && g < 1e5, "{g} GFLOP/s");
}

// ---------------------------------------------------------------------------------------------------------------------
// layer streaming
// ---------------------------------------------------------------------------------------------------------------------

use strata_workshop::layer_stream::{StreamMode, Tier};
use strata_workshop::model::MoeSpec;
use strata_workshop::profile::StreamProfile;

fn streamed(p: ProfileInput, tier: Tier, buffers: usize, rates: Vec<f64>) -> ProfileInput {
    ProfileInput { stream: Some(StreamProfile { mode: if tier == Tier::Ram { StreamMode::Ram } else { StreamMode::Disk }, tier, buffers, rates, on_gpu: true }), ..p }
}

#[test]
fn the_streamed_tiny_model_by_hand() {
    let cfg = tiny();
    let base = input(cfg.clone(), 4, None, BaseDtype::F32, 16, false);
    let e = profile::estimate(&streamed(base.clone(), Tier::Ram, 2, vec![1e9]));
    let st = e.stream.clone().expect("a streamed estimate");
    // a block's seven matrices: 2560 f32 values
    assert_eq!((st.block_bytes, st.buffer_bytes), (10_240, 20_480));
    // the device: both blocks' norms 2 · 32 · 4, the adapters with AdamW and gradients 5 · 2048, each block's saved input 4 · 16 · 16,
    // two buffers, one block's activations (checkpointing: 2 · 4 · 16 tokens · 776), the head's loss slice 6144, the final norm 64 and
    // the untied head 32 · 16 · 4
    // (a GPU run: the saved inputs and the replay times the measured calibration, and the softmax backward's temporaries)
    let cal = |b: u64| (profile::ACTIVATION_CALIBRATION * b as f64) as u64;
    assert_eq!(st.device_bytes, vec![256 + 10_240 + 2 * cal(1024) + 20_480 + cal(99_328) + profile::softmax_backward_bytes(&cfg, 16) + 6_144 + 64 + 2_048]);
    assert_eq!(e.per_device_bytes, st.device_bytes[0]);
    // the RAM tier holds both blocks' matrices and the embedding table (looked up in RAM on a GPU)
    assert_eq!((st.host_bytes, st.disk_bytes), (20_480 + 2_048, 0));
    assert_eq!(e.total_bytes, st.host_bytes, "a GPU run's RAM is the streamed base");
    // each block read forward and again in its replay; four conversations a step
    assert_eq!((st.bytes_per_conversation, st.bytes_per_step), (2 * 2 * 10_240, 4 * 2 * 2 * 10_240));
    assert!((st.transfer_seconds_per_step - 163_840.0 / 1e9).abs() < 1e-12);
    assert!(st.slowdown_serial >= st.slowdown_overlapped && st.slowdown_overlapped >= 1.0 && st.slowdown == st.slowdown_overlapped);
    // the disk tier: the shard cache on disk, two blocks in flight in RAM, and no overlap expected
    let d = profile::estimate(&streamed(base.clone(), Tier::Disk, 2, vec![1e9])).stream.unwrap();
    assert_eq!((d.disk_bytes, d.host_bytes), (20_480, 2 * 10_240 + 2_048));
    assert_eq!(d.slowdown, d.slowdown_serial);
    // more buffers: more device memory, nothing else
    let four = profile::estimate(&streamed(base.clone(), Tier::Ram, 4, vec![1e9])).stream.unwrap();
    assert_eq!(four.device_bytes[0] - st.device_bytes[0], 2 * 10_240);
    // with --train-layers 1 and the cache only the top block is loaded and streamed (and replayed)
    let top = profile::estimate(&streamed(input(cfg.clone(), 4, Some(1), BaseDtype::F32, 16, true), Tier::Ram, 2, vec![1e9])).stream.unwrap();
    assert_eq!((top.host_bytes, top.bytes_per_conversation), (10_240, 2 * 10_240));
    // the JSON carries it
    let j = profile::estimate_json(&streamed(base, Tier::Ram, 2, vec![1e9]), &e, &Machine { ram_total: 0, ram_available: 0, cpus: 1, gpus: vec![], cpu_gflops: 1.0 });
    assert_eq!(j["stream"]["block_bytes"], 10_240);
    assert_eq!(j["stream"]["tier"], "ram");
}

/// The upcycled 56B-A5B (Qwen3-8B: 36 blocks, a 16.4 GB bf16 base; 120 experts of 1,024 neurons, top-5, expert rank 4 with
/// a shared A, rank 16 on the attention) on two 16 GB cards, the OCuLink one (PCIe 4.0 x4) and the USB4 one.
#[test]
fn the_56b_a5b_streams_on_two_16_gb_cards() {
    let mut cfg = ModelConfig::from_json(&serde_json::json!({"model_type": "qwen3", "vocab_size": 151936, "hidden_size": 4096, "intermediate_size": 12288,
        "num_hidden_layers": 36, "num_attention_heads": 32, "num_key_value_heads": 8, "head_dim": 128, "tie_word_embeddings": false})).unwrap();
    cfg.moe = Some(MoeSpec { expert_ff: Some(1024), expert_rank: Some(4), shared_a: true, ..MoeSpec::full(120, 5) });
    let card = profile::DEFAULT_CARD_BYTES;
    let rates = vec![strata_workshop::layer_stream::link_bytes_s(16.0, 4), strata_workshop::layer_stream::USB4_TUNNEL_BYTES_S];
    let machine = Machine { ram_total: 64 << 30, ram_available: 30 << 30, cpus: 16, cpu_gflops: 100.0,
                            gpus: (0..2).map(|i| profile::Gpu { index: i, name: "RTX 5060 Ti".into(), total_bytes: card, free_bytes: card }).collect() };
    for len in [512, 1024] {
        let held = profile::moe_profile(&cfg, len, 2);
        let e_held = profile::estimate(&held);
        let p = streamed(held, Tier::Ram, 2, rates.clone());
        let e = profile::estimate(&p);
        let st = e.stream.clone().unwrap();
        let need = profile::card_bytes(&e);
        let fits = need + profile::MOE_CARD_MARGIN_BYTES <= card;
        println!("56B-A5B at {len} tokens, streamed: per card {} + CUDA {} = {:.2} GB of {:.2} GB ({}); held: {:.2} GB a card; host RAM {:.2} GB; \
                  PCIe {:.1} GB a conversation, {:.1} s a step of 4 against {:.1} s of compute: {:.2}x overlapped, {:.2}x serial",
                 st.device_bytes.iter().map(|b| format!("{:.2} GB", *b as f64 / 1e9)).collect::<Vec<_>>().join(" / "), profile::human_bytes(profile::CUDA_OVERHEAD_BYTES),
                 need as f64 / 1e9, card as f64 / 1e9, if fits { "fits with the 1.5 GB margin" } else { "does NOT fit" }, profile::card_bytes(&e_held) as f64 / 1e9,
                 st.host_bytes as f64 / 1e9, st.bytes_per_conversation as f64 / 1e9, st.transfer_seconds_per_step, st.compute_seconds_per_step,
                 st.slowdown_overlapped, st.slowdown_serial);
        for l in profile::report_lines(&p, &e, &machine) {
            println!("  {l}");
        }
        // a block's streamed matrices: q, o 4096², k, v 1024 × 4096, gate, up, down 12288 × 4096, in bf16
        assert_eq!(st.block_bytes, 2 * (2 * 4096 * 4096 + 2 * 1024 * 4096 + 3 * 12288 * 4096));
        // of the 16.4 GB base the RAM holds the 36 streamed blocks (13.9 GB) and the embedding table (1.24 GB); the head (1.24 GB)
        // stays on the last card
        assert!((15.1e9..15.2e9).contains(&(st.host_bytes as f64)), "{}", st.host_bytes);
        assert_eq!(st.bytes_per_conversation, 2 * 36 * st.block_bytes);
        // the busiest card holds much less than without streaming, and it fits
        assert!(need < profile::card_bytes(&e_held));
        assert!(fits, "{len} tokens: {need}");
        assert!(st.slowdown_overlapped >= 1.0 && st.slowdown_serial > st.slowdown_overlapped);
    }
}

/// Packing and several sequences a pass: the same PCIe bytes a pass, fewer passes an epoch, more activations on the
/// cards; auto picks the most that fit.
#[test]
fn packing_and_sequences_a_pass_in_the_estimate() {
    let mut cfg = ModelConfig::from_json(&serde_json::json!({"model_type": "qwen3", "vocab_size": 151936, "hidden_size": 4096, "intermediate_size": 12288,
        "num_hidden_layers": 36, "num_attention_heads": 32, "num_key_value_heads": 8, "head_dim": 128, "tie_word_embeddings": false})).unwrap();
    cfg.moe = Some(MoeSpec { expert_ff: Some(1024), expert_rank: Some(4), shared_a: true, ..MoeSpec::full(120, 5) });
    let rates = vec![strata_workshop::layer_stream::link_bytes_s(16.0, 4), strata_workshop::layer_stream::USB4_TUNNEL_BYTES_S];
    let mut base = streamed(profile::moe_profile(&cfg, 1024, 2), Tier::Ram, 2, rates);
    base.examples = 100_000;
    base.avg_tokens = 320;
    let one = profile::estimate(&base);
    let packed = ProfileInput { batch: profile::BatchProfile { packed_sequences: Some(33_000), seqs_per_pass: 4 }, ..base.clone() };
    let four = profile::estimate(&packed);
    let (a, b) = (one.stream.unwrap(), four.stream.clone().unwrap());
    assert_eq!(a.bytes_per_pass, b.bytes_per_pass, "a pass moves the base twice, whatever it holds");
    assert_eq!((a.passes_per_epoch, b.passes_per_epoch), (100_000, 33_000 / 4));
    assert_eq!(b.bytes_per_step, b.bytes_per_pass, "a step of 4 sequences is one pass");
    assert!(b.epoch_seconds < a.epoch_seconds / 3.0, "{} vs {}", b.epoch_seconds, a.epoch_seconds);
    assert!(b.device_bytes.iter().zip(&a.device_bytes).all(|(x, y)| x > y), "four sequences' activations");
    assert_eq!((four.units_per_epoch, four.seqs_per_pass, four.steps), (33_000, 4, 33_000 / 4));
    // auto: all four on empty 16 GB cards; on cards with almost nothing free, one
    let empty = [profile::DEFAULT_CARD_BYTES; 2];
    let n = profile::auto_seqs_per_pass(&packed, &empty);
    assert!((1..=4).contains(&n));
    assert_eq!(profile::auto_seqs_per_pass(&packed, &[1 << 30, 1 << 30]), 1);
    for l in profile::report_lines(&packed, &four, &Machine { ram_total: 64 << 30, ram_available: 30 << 30, cpus: 16, cpu_gflops: 100.0, gpus: vec![] }) {
        println!("  {l}");
    }
    println!("auto on two empty 16 GB cards: {n}");
}
