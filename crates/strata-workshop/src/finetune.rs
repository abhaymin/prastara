//! LoRA fine-tuning of a real model ([`crate::model::LlamaModel`]) on chat data — the Workshop's `train`.
//!
//! One conversation at a time (batch 1). A step is `accumulation` conversations: each one's loss is back-propagated
//! on its own and the gradients are summed, so only one conversation's graph is ever alive; then the mean gradient is
//! clipped and AdamW ([`crate::train::AdamWState`], the crate's own, with its moments in the open) takes one step at
//! the rate of a warm-up + cosine schedule. The run is a pure function of the settings and the seed: the order of the
//! conversations, the dropout masks and the rate depend only on (seed, step), which is what makes a resumed run take
//! exactly the steps an uninterrupted one would.
//!
//! **Output.** `<out>/adapter_model.safetensors` + `adapter_config.json` in PEFT's format (the Workshop's `merge`,
//! PEFT and Unsloth read it), `training_report.json`, and `checkpoint-<step>/` folders holding the adapter, AdamW's
//! moments and the step, to resume from.
//!
//! **Limited resources.** See [`TrainSettings`]: a 16-bit base, LoRA in the top N blocks only (the blocks below run
//! as plain inference), and the activation cache ([`ActCache`]): the hidden state entering the first trained block is
//! computed once per conversation and stored on disk, so later epochs never run the lower blocks — which can then
//! leave memory, or never be loaded when the cache is already complete.
//!
//! **Layer by layer** (after Soup, Apache-2.0: its `training.stream_layers`, `lisa` and `freeze_layers`). Layer
//! streaming ([`crate::layer_stream`], `--stream-layers ram|disk|auto`) keeps the frozen base off the device and copies
//! each block in just before it runs; LISA (`--lisa-layers K --lisa-interval N`, [`lisa_sample`]) trains only K sampled
//! blocks' adapters for N steps at a time and clears the moments of the blocks it re-freezes; `--freeze-bottom N` is
//! Soup's name for training the top `blocks − N` ([`TrainSettings::apply_freeze_bottom`]).

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{anyhow, bail, Context as _, Result};
use candle_core::{DType, Device, Tensor, Var};
use candle_nn::ParamsAdamW;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use crate::chatdata::{self, ChatTokenizer, DataFormat, Example};
use crate::layer_stream::{StreamMode, StreamOptions, StreamStats};
use crate::model::{self, BaseDtype, Ctx, LlamaModel, LoraSpec, ModelConfig, PassLayout};
use crate::packing::Schedule;
use crate::train::{hf_cosine_lr, AdamWState};

pub const ADAPTER_FILE: &str = "adapter_model.safetensors";
pub const CONFIG_FILE: &str = "adapter_config.json";
pub const OPTIMIZER_FILE: &str = "optimizer.safetensors";
pub const STATE_FILE: &str = "trainer_state.json";
pub const REPORT_FILE: &str = "training_report.json";

/// Where a run continues from.
#[derive(Debug, Clone, PartialEq)]
pub enum Resume {
    /// the newest `checkpoint-<step>` in the output folder, if there is one
    Auto,
    Dir(PathBuf),
}

/// Everything a training run depends on.
#[derive(Debug, Clone)]
pub struct TrainSettings {
    /// the base model: a Hugging Face folder
    pub model: PathBuf,
    pub data: PathBuf,
    pub data_format: DataFormat,
    /// a second file to measure on; otherwise `val_split` of the data is held out
    pub eval_data: Option<PathBuf>,
    pub val_split: f64,
    pub out: PathBuf,
    pub lora: LoraSpec,
    /// f16 / bf16 hold the frozen base in half the memory (LoRA and the loss stay f32)
    pub base_dtype: BaseDtype,
    /// peak learning rate
    pub lr: f64,
    pub epochs: f64,
    /// stop after this many optimizer steps, whatever `epochs` says
    pub max_steps: Option<usize>,
    /// conversations per optimizer step (the batch is always 1; this is the effective batch)
    pub accumulation: usize,
    /// longer conversations are cut to this many tokens
    pub max_len: usize,
    pub warmup_ratio: f64,
    pub weight_decay: f64,
    /// clip the gradient's norm to this (0: no clipping)
    pub max_grad_norm: f64,
    pub seed: u64,
    /// the activation cache's folder (needs `lora.train_layers`)
    pub cache_dir: Option<PathBuf>,
    /// write a checkpoint every N steps (0: only when stopped and at the end)
    pub save_steps: usize,
    /// measure the held-out loss every N steps (0: only at the end)
    pub eval_steps: usize,
    pub logging_steps: usize,
    pub resume: Option<Resume>,
    /// a chat template file, for a model that has none
    pub chat_template: Option<PathBuf>,
    /// score every token, not only the assistant's (Soup's `train_on_responses_only: false`)
    pub score_everything: bool,
    /// where the model trains: `cpu` (the default), or GPUs such as `cuda:0,cuda:1`, the blocks split between them
    pub devices: Vec<String>,
    /// recompute each trained block's activations in the backward pass instead of keeping them (less memory, about
    /// one more forward pass of work)
    pub gradient_checkpointing: bool,
    /// stream the blocks' frozen matrices from host memory or disk ([`crate::layer_stream`]); turns checkpointing on
    pub stream_layers: Option<StreamMode>,
    /// device buffers per device for the streamed blocks (2 to 8)
    pub stream_buffers: usize,
    /// the disk tier's shard cache root (default [`crate::layer_stream::default_cache_root`])
    pub stream_cache: Option<PathBuf>,
    /// LISA: only a few sampled blocks train at a time
    pub lisa: Option<Lisa>,
    /// Soup's `freeze_layers`: the bottom N blocks carry no adapters (resolved into `lora.train_layers` once the model's
    /// block count is known, [`TrainSettings::apply_freeze_bottom`])
    pub freeze_bottom: Option<usize>,
    /// sequence packing ([`crate::packing`]): whole conversations laid end to end into sequences of up to `max_len`
    /// tokens, each conversation attending only to itself from position 0; a step is then `accumulation` sequences.
    /// The command turns it on by default when the base is streamed.
    pub pack: bool,
    /// sequences (or, unpacked, conversations) that go through each block together in one pass: a streamed block's
    /// weights are copied once for all of them, forward and in the replay. The gradients are those of running them one
    /// by one (dropout's masks aside: they are drawn per pass). 0: auto, which the command resolves from the cards'
    /// free memory ([`crate::profile::auto_seqs_per_pass`]); here it is 1. At most `accumulation` are used.
    pub seqs_per_pass: usize,
}

/// LISA (Layerwise Importance Sampled AdamW, after Soup's `lisa`): every `interval` optimizer steps, `layers` of the
/// blocks that carry adapters are sampled (by `seed` and the interval's number alone, so a resumed run samples the
/// same ones); only their adapters (and an upcycled block's router and experts) train until the next sample, and the
/// AdamW moments of the blocks that stop training are cleared. The loop's gates, when they train, train throughout
/// (they are not a block's), as Soup keeps the embeddings and head on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Lisa {
    pub layers: usize,
    pub interval: usize,
    pub seed: u64,
}

/// The blocks LISA trains in interval `interval`: `k` of `candidates` (all of them when k is more), sorted.
pub fn lisa_sample(seed: u64, interval: usize, candidates: &[usize], k: usize) -> Vec<usize> {
    let order = permutation(seed ^ 0x115A_5A11, interval as u64 + 1, candidates.len());
    let mut chosen: Vec<usize> = order.into_iter().take(k.min(candidates.len())).map(|i| candidates[i]).collect();
    chosen.sort_unstable();
    chosen
}

/// The devices named by `--devices` (`cpu`, `cuda:N`, `metal[:N]`), opened.
pub fn open_devices(names: &[String]) -> Result<Vec<Device>> {
    if names.is_empty() {
        return Ok(vec![Device::Cpu]);
    }
    names.iter().map(|n| match n.trim() {
        "cpu" => Ok(Device::Cpu),
        g if g == "metal" || g.starts_with("metal:") => {
            let i: usize = if g == "metal" { 0 } else { g["metal:".len()..].parse().map_err(|_| anyhow!("device {g:?}: name metal or metal:N"))? };
            #[cfg(feature = "metal")]
            { Device::new_metal(i).map_err(|e| anyhow!("metal:{i}: {e}")) }
            #[cfg(not(feature = "metal"))]
            { let _ = i; bail!("device {g}: this Workshop was built without the metal feature (Apple Silicon builds have it)") }
        }
        g => {
            let i: usize = g.strip_prefix("cuda:").and_then(|i| i.parse().ok())
                .ok_or_else(|| anyhow!("device {g:?}: name cpu, cuda:N or metal[:N]"))?;
            #[cfg(feature = "cuda")]
            { Device::new_cuda(i).map_err(|e| anyhow!("cuda:{i}: {e}")) }
            #[cfg(not(feature = "cuda"))]
            { let _ = i; bail!("device {g}: this Workshop was built without the cuda feature") }
        }
    }).collect()
}

impl TrainSettings {
    /// Soup's defaults where it has them (epochs 3, warm-up 3 %, weight decay 0.01, clip 1.0, accumulation 4,
    /// 10 % held out, 2048 tokens), LoRA's usual rate.
    pub fn new(model: impl Into<PathBuf>, data: impl Into<PathBuf>, out: impl Into<PathBuf>) -> Self {
        Self {
            model: model.into(), data: data.into(), data_format: DataFormat::Auto, eval_data: None, val_split: 0.1, out: out.into(),
            lora: LoraSpec::default(), base_dtype: BaseDtype::F32, lr: 2e-4, epochs: 3.0, max_steps: None, accumulation: 4, max_len: 2048,
            warmup_ratio: 0.03, weight_decay: 0.01, max_grad_norm: 1.0, seed: 42, cache_dir: None, save_steps: 100, eval_steps: 0,
            logging_steps: 10, resume: None, chat_template: None, score_everything: false, devices: Vec::new(), gradient_checkpointing: false,
            stream_layers: None, stream_buffers: crate::layer_stream::DEFAULT_BUFFERS, stream_cache: None, lisa: None, freeze_bottom: None,
            pack: false, seqs_per_pass: 1,
        }
    }

    /// Turn `freeze_bottom` into `lora.train_layers` for a model of `layers` blocks (a no-op without it).
    pub fn apply_freeze_bottom(&mut self, layers: usize) -> Result<()> {
        let Some(n) = self.freeze_bottom else { return Ok(()) };
        if n >= layers {
            bail!("--freeze-bottom {n}: the model has {layers} blocks; freeze 0 to {}", layers.saturating_sub(1));
        }
        match self.lora.train_layers {
            Some(t) if t != layers - n => bail!("--freeze-bottom {n} and --train-layers {t} disagree on a {layers}-block model: give one of them"),
            _ => self.lora.train_layers = (n > 0).then_some(layers - n),
        }
        self.freeze_bottom = None;
        Ok(())
    }

    pub fn validate(&self) -> Result<()> {
        if !(self.lr > 0.0 && self.lr.is_finite()) {
            bail!("the learning rate must be above 0");
        }
        if !(self.epochs > 0.0) {
            bail!("epochs must be above 0");
        }
        if self.accumulation == 0 {
            bail!("gradient accumulation must be 1 or more");
        }
        if self.max_len < 8 {
            bail!("a sequence length of {} is too short to hold a question and an answer", self.max_len);
        }
        if !(0.0..=0.5).contains(&self.val_split) {
            bail!("the validation split must be between 0 and 0.5");
        }
        if !(0.0..=0.5).contains(&self.warmup_ratio) {
            bail!("the warm-up ratio must be between 0 and 0.5");
        }
        crate::layer_stream::validate_buffers(self.stream_buffers)?;
        if let Some(l) = self.lisa {
            if l.layers == 0 || l.interval == 0 {
                bail!("LISA needs --lisa-layers and --lisa-interval of 1 or more");
            }
        }
        Ok(())
    }

    /// The facts a checkpoint must share with the run that resumes it, or the steps would not be the same steps.
    fn fingerprint(&self, n_train: usize, total_steps: usize, units: usize) -> Value {
        let mut v = json!({
            "lr": self.lr, "accumulation": self.accumulation, "max_len": self.max_len, "seed": self.seed, "warmup_ratio": self.warmup_ratio,
            "weight_decay": self.weight_decay, "max_grad_norm": self.max_grad_norm, "train_examples": n_train, "total_steps": total_steps,
            "r": self.lora.r, "alpha": self.lora.alpha, "dropout": self.lora.dropout, "use_rslora": self.lora.use_rslora,
            "targets": self.lora.targets.iter().map(|t| t.as_str()).collect::<Vec<_>>(), "train_layers": self.lora.train_layers,
            "base_dtype": self.base_dtype.as_str(), "score_everything": self.score_everything,
        });
        // LISA changes which steps are taken (streaming does not: the same arithmetic, other memory)
        if let Some(l) = self.lisa {
            v["lisa_layers"] = json!(l.layers);
            v["lisa_interval"] = json!(l.interval);
            v["lisa_seed"] = json!(l.seed);
        }
        // packing changes the steps (sequences per pass does not: the same sums, other memory)
        if self.pack {
            v["pack"] = json!(true);
            v["packed_sequences"] = json!(units);
        }
        v
    }
}

/// What a run tells its caller while it works.
#[derive(Debug, Clone)]
pub enum Event {
    Log(String),
    /// an optimizer step finished: `loss` is the mean over its conversations, before the update
    /// `aux`: on an upcycled model, the mean load-balancing loss of the step (1 when the router spreads the tokens
    /// evenly), which `loss` includes times [`crate::model::MOE_AUX_COEF`]
    Step { step: usize, total: usize, loss: f64, lr: f64, eta_seconds: f64, aux: Option<f64> },
    Eval { step: usize, loss: f64 },
    Checkpoint { step: usize, dir: PathBuf },
    /// the activation cache is being filled
    Cache { done: usize, total: usize },
}

/// The outcome of [`train`].
#[derive(Debug, Clone)]
pub struct TrainReport {
    pub out: PathBuf,
    pub steps_done: usize,
    pub total_steps: usize,
    /// true when a stop request ended the run early (a checkpoint was written)
    pub stopped: bool,
    /// the loss of every step taken in this process (before its update)
    pub losses: Vec<f64>,
    pub eval_loss: Option<f64>,
    pub train_examples: usize,
    pub eval_examples: usize,
    pub trainable_params: u64,
    pub seconds: f64,
    pub tokens_seen: u64,
    pub cache_hits: usize,
    pub cache_misses: usize,
    pub cache_bytes: u64,
    /// the lower blocks were never loaded (the cache was complete) or were freed after filling it
    pub lower_in_memory: bool,
    pub peak_rss_bytes: Option<u64>,
    /// what layer streaming copied (None: not streamed)
    pub stream: Option<StreamStats>,
}

/// A test hook: the micro-step with this number fails, after its backward pass, as if the device had run out of
/// memory (`usize::MAX`: never), [`INJECT_OOM_TIMES`] times. One failure is retried and must give the run it would
/// have had; two end the run.
pub static INJECT_OOM: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(usize::MAX);
pub static INJECT_OOM_TIMES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(1);
/// A test hook: the micro-step with this number fails while its gradients are being added into the step's sums,
/// after the first [`INJECT_COMMIT_AFTER`] parts (once; `usize::MAX`: never).
pub static INJECT_COMMIT_OOM: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(usize::MAX);
pub static INJECT_COMMIT_AFTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(3);

/// Whether `e` is a device's out-of-memory error (CUDA's or cuBLAS's, as candle and cudarc word them, or Metal's).
/// Not sticky on CUDA: the context stays usable, so the micro-step can run again.
pub fn is_out_of_memory(e: &anyhow::Error) -> bool {
    let t = format!("{e:#}");
    t.contains("CUDA_ERROR_OUT_OF_MEMORY") || t.contains("CUBLAS_STATUS_ALLOC_FAILED") || t.contains("out of memory") && (t.contains("CUDA") || t.contains("cuda") || t.contains("Metal"))
}

fn first_line(e: &anyhow::Error) -> String {
    format!("{e:#}").lines().next().unwrap_or("").chars().take(160).collect()
}

/// A run of at most this many steps reports every step's loss, whatever `logging_steps` says.
pub const SHORT_RUN_STEPS: usize = 50;

/// The process's peak resident memory (Linux `VmHWM`), when the OS says.
pub fn peak_rss_bytes() -> Option<u64> {
    let s = std::fs::read_to_string("/proc/self/status").ok()?;
    let kb: u64 = s.lines().find(|l| l.starts_with("VmHWM:"))?.split_whitespace().nth(1)?.parse().ok()?;
    Some(kb * 1024)
}

fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// A permutation of `0..n` that depends only on `(seed, salt)` (Fisher–Yates).
pub fn permutation(seed: u64, salt: u64, n: usize) -> Vec<usize> {
    let mut state = seed ^ salt.wrapping_mul(0xC2B2_AE3D_27D4_EB4F) ^ 0x51_7C_C1_B7_27_22_0A_95;
    let mut v: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        let j = (splitmix(&mut state) % (i as u64 + 1)) as usize;
        v.swap(i, j);
    }
    v
}

// ---------------------------------------------------------------------------------------------------------------------
// the activation cache
// ---------------------------------------------------------------------------------------------------------------------

/// The hidden states entering the first trained block, one file per conversation, keyed by its token ids.
///
/// The folder's `manifest.json` records what the states depend on — the model's files, the first trained block and
/// the base dtype — and a cache made for something else is refused, not reused. A state is stored as f32, exactly as
/// computed, so a run reading the cache has the same losses as one computing the states.
pub struct ActCache {
    dir: PathBuf,
    pub hits: usize,
    pub misses: usize,
    pub bytes_written: u64,
}

/// What the cached states depend on: the config, each weight file's name, size and first mebibyte, the first trained
/// block and the base dtype.
pub fn cache_fingerprint(model_dir: &Path, first_trained: usize, base_dtype: BaseDtype) -> Result<Value> {
    use std::io::Read as _;
    let mut weights = Vec::new();
    for f in model::weight_files(model_dir)? {
        let mut head = vec![0u8; 1 << 20];
        let mut file = std::fs::File::open(&f)?;
        let mut n = 0;
        while n < head.len() {
            let got = file.read(&mut head[n..])?;
            if got == 0 {
                break;
            }
            n += got;
        }
        weights.push(json!({"file": f.file_name().and_then(|n| n.to_str()), "bytes": file.metadata()?.len(), "head_sha256": format!("{:x}", Sha256::digest(&head[..n]))}));
    }
    let config = std::fs::read(model_dir.join("config.json"))?;
    Ok(json!({"format": 1, "config_sha256": format!("{:x}", Sha256::digest(&config)), "weights": weights,
              "first_trained_block": first_trained, "base_dtype": base_dtype.as_str()}))
}

impl ActCache {
    /// Open (or start) a cache folder for `fingerprint`.
    pub fn open(dir: &Path, fingerprint: &Value) -> Result<Self> {
        std::fs::create_dir_all(dir).with_context(|| format!("{}", dir.display()))?;
        let manifest = dir.join("manifest.json");
        match std::fs::read_to_string(&manifest) {
            Ok(text) => {
                let have: Value = serde_json::from_str(&text).with_context(|| format!("{}", manifest.display()))?;
                if &have != fingerprint {
                    bail!("the activation cache in {} was made for another model, base dtype or number of trained blocks; choose another folder or delete this one", dir.display());
                }
            }
            Err(_) => std::fs::write(&manifest, serde_json::to_string_pretty(fingerprint)?)?,
        }
        Ok(Self { dir: dir.to_path_buf(), hits: 0, misses: 0, bytes_written: 0 })
    }

    fn path(&self, tokens: &[u32]) -> PathBuf {
        // FNV-1a over the ids; the file also holds the ids, so a clash is detected, never used
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for t in tokens {
            for b in t.to_le_bytes() {
                h = (h ^ b as u64).wrapping_mul(0x0000_0100_0000_01B3);
            }
        }
        self.dir.join(format!("{h:016x}-{}.safetensors", tokens.len()))
    }

    pub fn has(&self, tokens: &[u32]) -> bool {
        self.path(tokens).exists()
    }

    pub fn get(&mut self, tokens: &[u32], dev: &Device) -> Result<Option<Tensor>> {
        let p = self.path(tokens);
        if !p.exists() {
            return Ok(None);
        }
        let mut t = candle_core::safetensors::load(&p, dev).with_context(|| format!("{}", p.display()))?;
        let ids: Vec<u32> = t.remove("tokens").ok_or_else(|| anyhow!("{}: no tokens", p.display()))?.to_vec1()?;
        if ids != tokens {
            return Ok(None);
        }
        self.hits += 1;
        Ok(Some(t.remove("h").ok_or_else(|| anyhow!("{}: no hidden state", p.display()))?))
    }

    /// Compute the states of `examples` without ever holding the frozen part in memory: the embeddings first, then
    /// one block at a time — load it, pass every conversation's state through it (the states wait on disk in
    /// `filling/`), drop it. Memory: the embeddings or one block, plus one conversation. The states are the same bits
    /// as [`LlamaModel::lower`] computes with everything loaded: the same operations in the same order.
    pub fn fill(&mut self, model_dir: &Path, dtype: BaseDtype, dev: &Device, first_trained: usize, examples: &[&Example], on_event: &mut dyn FnMut(Event)) -> Result<()> {
        let work = self.dir.join("filling");
        // a fill that was interrupted starts again: half-way states are of no use
        std::fs::remove_dir_all(&work).ok();
        std::fs::create_dir_all(&work)?;
        let part = |i: usize| work.join(format!("{i}.safetensors"));
        let save = |i: usize, h: &Tensor| -> Result<()> { Ok(candle_core::safetensors::save(&HashMap::from([("h".to_string(), h.clone())]), part(i))?) };
        // per conversation: the embeddings pass, then one pass per frozen block
        let passes_per_example = 1 + first_trained;
        let total = examples.len() * passes_per_example;
        let mut done = 0;
        let mut tick = |on_event: &mut dyn FnMut(Event)| {
            done += 1;
            if done % 16 == 0 || done == total {
                on_event(Event::Cache { done, total });
            }
        };
        {
            let embeddings = LlamaModel::load_parts(model_dir, dtype, dev, 0..0, true, false)?;
            for (i, ex) in examples.iter().enumerate() {
                save(i, &embeddings.embed_tokens(&ex.tokens)?)?;
                tick(on_event);
            }
        }
        for block in 0..first_trained {
            let one = LlamaModel::load_parts(model_dir, dtype, dev, block..block + 1, false, false)?;
            for i in 0..examples.len() {
                let h = candle_core::safetensors::load(part(i), dev)?.remove("h").ok_or_else(|| anyhow!("{}: no state", part(i).display()))?;
                save(i, &one.frozen_blocks(&h, block..block + 1)?)?;
                tick(on_event);
            }
        }
        for (i, ex) in examples.iter().enumerate() {
            let h = candle_core::safetensors::load(part(i), dev)?.remove("h").ok_or_else(|| anyhow!("{}: no state", part(i).display()))?;
            self.put(&ex.tokens, &h)?;
        }
        std::fs::remove_dir_all(&work).ok();
        Ok(())
    }

    pub fn put(&mut self, tokens: &[u32], h: &Tensor) -> Result<()> {
        let p = self.path(tokens);
        let tmp = p.with_extension("tmp");
        let map: HashMap<String, Tensor> = [("h".to_string(), h.to_dtype(DType::F32)?), ("tokens".to_string(), Tensor::from_vec(tokens.to_vec(), (tokens.len(),), h.device())?)].into();
        candle_core::safetensors::save(&map, &tmp)?;
        self.bytes_written += std::fs::metadata(&tmp)?.len();
        std::fs::rename(&tmp, &p)?;
        self.misses += 1;
        Ok(())
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// data
// ---------------------------------------------------------------------------------------------------------------------

/// The tokenised training and held-out conversations of a run.
#[derive(Debug, Clone)]
pub struct Data {
    pub train: Vec<Example>,
    pub eval: Vec<Example>,
    /// what was left out or cut, for the log
    pub notes: Vec<String>,
}

/// Read, tokenise and split the data as `settings` says.
pub fn prepare_data(s: &TrainSettings) -> Result<Data> {
    let tok = ChatTokenizer::load(&s.model, s.chat_template.as_deref())?;
    let mut notes = Vec::new();
    let mut load = |path: &Path| -> Result<Vec<Example>> {
        let (records, bad) = chatdata::read_jsonl(path, s.data_format)?;
        if !bad.is_empty() {
            notes.push(format!("{}: {} unreadable rows left out (line {}: {})", path.display(), bad.len(), bad[0].0, bad[0].1));
        }
        let t = chatdata::tokenize(&tok, &records, s.max_len, s.score_everything);
        if t.skipped_count > 0 {
            notes.push(format!("{}: {} conversations left out ({})", path.display(), t.skipped_count, t.skipped.join("; ")));
        }
        if t.truncated > 0 {
            notes.push(format!("{}: {} conversations reach the {}-token cap and were cut there", path.display(), t.truncated, s.max_len));
        }
        if t.examples.is_empty() {
            bail!("{}: no usable conversation", path.display());
        }
        Ok(t.examples)
    };
    let all = load(&s.data)?;
    let (train, eval) = match &s.eval_data {
        Some(p) => (all, load(p)?),
        None => {
            let n = all.len();
            let held = if s.val_split > 0.0 && n >= 2 { ((n as f64 * s.val_split).round() as usize).clamp(1, n - 1) } else { 0 };
            let order = permutation(s.seed, 0xE7A1, n);
            let is_eval: std::collections::HashSet<usize> = order[n - held..].iter().copied().collect();
            let (mut tr, mut ev) = (Vec::new(), Vec::new());
            for (i, ex) in all.into_iter().enumerate() {
                if is_eval.contains(&i) { ev.push(ex) } else { tr.push(ex) }
            }
            (tr, ev)
        }
    };
    Ok(Data { train, eval, notes })
}

/// Optimizer steps for `n` conversations: `ceil(ceil(n · epochs) / accumulation)`, capped by `max_steps`.
pub fn total_steps(n: usize, epochs: f64, accumulation: usize, max_steps: Option<usize>) -> usize {
    total_units_steps((n as f64 * epochs).ceil() as usize, accumulation, max_steps)
}

/// Optimizer steps for a run of `units` units (conversations, or packed sequences): `ceil(units / accumulation)`,
/// capped by `max_steps`.
pub fn total_units_steps(units: usize, accumulation: usize, max_steps: Option<usize>) -> usize {
    let steps = units.div_ceil(accumulation.max(1)).max(1);
    max_steps.map_or(steps, |m| steps.min(m.max(1)))
}

/// The training units (each a list of indices into `data.train`) of optimizer step `step` (0-based) of a run with
/// these settings: what a run and its resumption both take at that step.
pub fn step_batch(s: &TrainSettings, data: &Data, step: usize) -> Vec<Vec<usize>> {
    let lens: Vec<usize> = data.train.iter().map(|e| e.tokens.len()).collect();
    Schedule::new(&lens, s.seed, s.epochs, s.max_len, s.pack).step_units(step, s.accumulation).into_iter().map(<[usize]>::to_vec).collect()
}

// ---------------------------------------------------------------------------------------------------------------------
// the run
// ---------------------------------------------------------------------------------------------------------------------

/// The conversations of one pass, end to end, and how they lie ([`PassLayout`]).
struct Pass {
    examples: Vec<Example>,
    layout: PassLayout,
    tokens: Vec<u32>,
    supervised: Vec<bool>,
}

impl Pass {
    /// The pass of `units` (each a sequence: indices into `pool`).
    fn of(pool: &[Example], units: &[Vec<usize>]) -> Pass {
        let examples: Vec<Example> = units.iter().flatten().map(|i| pool[*i].clone()).collect();
        let layout = PassLayout { seqs: units.iter().map(|u| u.iter().map(|i| pool[*i].tokens.len()).collect()).collect() };
        let tokens = examples.iter().flat_map(|e| e.tokens.iter().copied()).collect();
        let supervised = examples.iter().flat_map(|e| e.supervised.iter().copied()).collect();
        Pass { examples, layout, tokens, supervised }
    }
}

/// What a pass gives the step: the sum of its conversations' losses, their mean router loss, its tokens and
/// conversations, and its gradients.
struct PassOut {
    value: f64,
    aux: Option<f64>,
    tokens: u64,
    conversations: usize,
    grads: candle_core::backprop::GradStore,
    extra: Vec<(Var, Tensor)>,
}

/// A loaded model with its adapters, optimizer and data: what `train` and `find_lr` both drive.
struct Runner {
    model: LlamaModel,
    names: Vec<String>,
    vars: Vec<Var>,
    opt: AdamWState,
    data: Data,
    cache: Option<ActCache>,
    seed: u64,
    accumulation: usize,
    max_grad_norm: f64,
    /// the run's training units in order: conversations, or packed sequences of them
    sched: Schedule,
    /// units per pass through the blocks (lowered for the rest of the run after a pass runs out of device memory)
    seqs_per_pass: usize,
    /// pack the held-out conversations too (the same mean, fewer passes)
    pack_eval: bool,
    max_len: usize,
    tokens_seen: u64,
    lower_in_memory: bool,
    /// the last step's mean load-balancing loss (an upcycled model only)
    last_aux: Option<f64>,
    /// the data is a private dataset (crate::private_data): the adapter is marked `prastara_private`
    private: bool,
    /// where each micro-step's memory high-water marks go on CUDA (`AM2_MEM_TRACE`, else `<out>/memory-trace.log`)
    mem_trace: Option<PathBuf>,
    lisa: Option<LisaRun>,
    /// each trainable variable's block (None: the loop's gates, which are no block's)
    var_block: Vec<Option<usize>>,
    /// lines for the log from inside a step (LISA's samples)
    pending: Vec<String>,
}

struct LisaRun {
    settings: Lisa,
    /// the blocks that carry adapters
    candidates: Vec<usize>,
    /// the interval sampled now, and its blocks
    current: Option<(usize, Vec<usize>)>,
}

/// The block an adapter variable belongs to, from its name (`…layers.N.…`), as Soup finds a parameter's layer.
fn block_of(name: &str) -> Option<usize> {
    let rest = &name[name.find("layers.")? + "layers.".len()..];
    rest[..rest.find('.')?].parse().ok()
}

impl Runner {
    fn new(s: &TrainSettings, data: Data, on_event: &mut dyn FnMut(Event)) -> Result<Self> {
        let dev = Device::Cpu;
        let cfg = ModelConfig::load(&s.model)?;
        let first = s.lora.first_trained(cfg.num_hidden_layers);
        if let Some(n) = s.lora.train_layers {
            if n == 0 || n > cfg.num_hidden_layers {
                bail!("--train-layers {n}: the model has {} blocks; choose 1 to {}", cfg.num_hidden_layers, cfg.num_hidden_layers);
            }
        }
        let mut cache = match &s.cache_dir {
            Some(_) if first == 0 => {
                on_event(Event::Log("the activation cache is off: every block trains, so there is no frozen part to cache (use --train-layers N)".into()));
                None
            }
            Some(dir) => Some(ActCache::open(dir, &cache_fingerprint(&s.model, first, s.base_dtype)?)?),
            None => None,
        };
        // with the cache, the frozen blocks are never all in memory: what is missing is computed one block at a time
        let mut lower_in_memory = true;
        if let Some(c) = cache.as_mut() {
            let missing: Vec<&Example> = data.train.iter().chain(&data.eval).filter(|ex| !c.has(&ex.tokens)).collect();
            if missing.is_empty() {
                on_event(Event::Log(format!("activation cache complete: blocks 0 to {} were not loaded", first - 1)));
            } else {
                c.fill(&s.model, s.base_dtype, &dev, first, &missing, on_event)?;
                on_event(Event::Log(format!("activation cache filled block by block ({} states, {:.1} MB): never more than one of the {first} frozen blocks was in memory",
                                            c.misses, c.bytes_written as f64 / 1e6)));
            }
            lower_in_memory = false;
        }
        let devs = open_devices(&s.devices)?;
        let skip = if cache.is_some() { first } else { 0 };
        let mut model = match s.stream_layers {
            Some(mode) => {
                let opts = StreamOptions { mode, buffers: s.stream_buffers, cache_root: s.stream_cache.clone() };
                LlamaModel::load_streamed(&s.model, s.base_dtype, &devs, skip..cfg.num_hidden_layers, skip == 0, true, &opts, &mut |l| on_event(Event::Log(l)))?
            }
            None => LlamaModel::load_parts_on(&s.model, s.base_dtype, &devs, skip..cfg.num_hidden_layers, skip == 0, true)?,
        };
        if let (Some(_), Some(note)) = (s.stream_layers, crate::layer_stream::host_memory_note()) {
            on_event(Event::Log(format!("host memory after the streamed load: {note}")));
        }
        model.add_lora(&s.lora)?;
        if s.stream_layers.is_some() && !s.gradient_checkpointing {
            on_event(Event::Log("layer streaming turns gradient checkpointing on: the backward pass replays each block from its streamed weights".into()));
        }
        model.set_checkpointing(s.gradient_checkpointing || s.stream_layers.is_some());
        let (names, vars): (Vec<String>, Vec<Var>) = model.lora_vars().into_iter().unzip();
        let var_block: Vec<Option<usize>> = names.iter().map(|n| block_of(n)).collect();
        let lisa = s.lisa.map(|l| {
            let candidates: Vec<usize> = (first..cfg.num_hidden_layers).collect();
            on_event(Event::Log(format!("LISA: {} of the {} blocks with adapters train at a time, sampled anew every {} steps (seed {})",
                                        l.layers.min(candidates.len()), candidates.len(), l.interval, l.seed)));
            LisaRun { settings: l, candidates, current: None }
        });
        let mut params = ParamsAdamW::default();
        params.lr = s.lr;
        params.weight_decay = s.weight_decay;
        let opt = AdamWState::new(vars.clone(), params).map_err(|e| anyhow!("{e}"))?;
        let lens: Vec<usize> = data.train.iter().map(|e| e.tokens.len()).collect();
        let sched = Schedule::new(&lens, s.seed, s.epochs, s.max_len, s.pack);
        Ok(Self { model, names, vars, opt, data, cache, seed: s.seed, accumulation: s.accumulation, max_grad_norm: s.max_grad_norm, sched,
                  seqs_per_pass: s.seqs_per_pass.clamp(1, s.accumulation.max(1)), pack_eval: s.pack || s.seqs_per_pass > 1, max_len: s.max_len,
                  tokens_seen: 0, lower_in_memory, last_aux: None, private: false,
                  mem_trace: std::env::var_os("AM2_MEM_TRACE").map(PathBuf::from).or_else(|| Some(s.out.join("memory-trace.log"))), lisa, var_block,
                  pending: Vec::new() })
    }

    /// LISA: the blocks of the interval step `step` is in. Moving to a new interval clears the moments of the blocks
    /// that stop training and starts those that begin again from zero (`restart`; off when a resumed run takes up the
    /// interval its checkpoint was in, whose moments it has just restored).
    fn lisa_select(&mut self, step: usize, restart: bool) -> Result<()> {
        let Some(l) = self.lisa.as_mut() else { return Ok(()) };
        let interval = step / l.settings.interval;
        if l.current.as_ref().is_some_and(|(c, _)| *c == interval) {
            return Ok(());
        }
        let new = lisa_sample(l.settings.seed, interval, &l.candidates, l.settings.layers);
        let old = l.current.take().map(|(_, b)| b);
        if restart {
            for (slot, b) in self.var_block.iter().enumerate() {
                let Some(b) = b else { continue };
                let (was, now) = (old.as_ref().is_some_and(|o| o.contains(b)), new.contains(b));
                if was && !now {
                    self.opt.clear_slot(slot).map_err(|e| anyhow!("{e}"))?;
                } else if now && !was {
                    self.opt.restart_slot(slot).map_err(|e| anyhow!("{e}"))?;
                }
            }
        }
        self.model.set_active_blocks(Some(&new));
        let until = (interval + 1) * l.settings.interval;
        self.pending.push(format!("LISA: blocks {new:?} train for steps {} to {until}", interval * l.settings.interval + 1));
        l.current = Some((interval, new));
        Ok(())
    }

    fn hidden(&mut self, ex: &Example) -> Result<Tensor> {
        if let Some(c) = self.cache.as_mut() {
            return c.get(&ex.tokens, self.model.device())?.ok_or_else(|| anyhow!("a hidden state is missing from the activation cache (was the folder changed during the run?)"));
        }
        self.model.lower(&ex.tokens)
    }

    /// The hidden state entering the first trained block for a pass: each conversation's cached state end to end, or
    /// the lower blocks run once over the whole pass (a block-diagonal mask: the same states).
    fn hidden_pass(&mut self, pass: &Pass) -> Result<Tensor> {
        if pass.examples.len() == 1 {
            return self.hidden(&pass.examples[0]);
        }
        if self.cache.is_some() {
            let parts = pass.examples.iter().map(|e| self.hidden(e)).collect::<Result<Vec<_>>>()?;
            return Ok(Tensor::cat(&parts, 0)?);
        }
        self.model.lower_packed(&pass.tokens, &pass.layout)
    }

    /// One pass through the blocks: units `m .. m + n` of the run (conversations, or packed sequences), all together.
    /// Its loss is the sum of each conversation's own, so the step divides by its conversations; the gradients are as
    /// the backward pass left them (the graph's store and the checkpoint replays' parts, no copies). Nothing of the
    /// run's state changes here, so a pass that fails can be run again.
    fn micro(&mut self, step: usize, m: usize, n: usize, lr: f64) -> Result<PassOut> {
        let units: Vec<Vec<usize>> = (m..m + n).map(|j| self.sched.unit(j).to_vec()).collect();
        let pass = Pass::of(&self.data.train, &units);
        let h = self.hidden_pass(&pass)?;
        let mut dropout_seed = self.seed ^ (m as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let ctx = Ctx::train(splitmix(&mut dropout_seed));
        let loss = self.model.loss_packed(&h, &pass.layout, &pass.tokens, &pass.supervised, &ctx)?;
        let value = loss.to_scalar::<f32>()? as f64;
        if !value.is_finite() {
            bail!("the loss became {value} at step {} (rate {lr:.3e}): lower the learning rate", step + 1);
        }
        let grads = loss.backward()?;
        let hook = INJECT_OOM.load(std::sync::atomic::Ordering::Relaxed);
        if (m..m + n).contains(&hook) {
            if INJECT_OOM_TIMES.fetch_sub(1, std::sync::atomic::Ordering::Relaxed) <= 1 {
                INJECT_OOM.store(usize::MAX, std::sync::atomic::Ordering::Relaxed);
            }
            bail!("injected for a test: DriverError(CUDA_ERROR_OUT_OF_MEMORY, \"out of memory\")");
        }
        // with checkpointing, the adapters inside a recomputed block get their gradient there
        let extra = self.model.take_checkpoint_grads();
        drop((loss, h));
        Ok(PassOut { value, aux: ctx.aux_loss(), tokens: pass.tokens.len() as u64, conversations: pass.examples.len(), grads, extra })
    }

    /// Add a micro-step's gradient parts into the step's sums, one part at a time, as the trainer always has (the
    /// extra memory is one variable's new sum). `added` records each `(slot, part)` for [`Runner::uncommit`].
    fn commit(&self, m: usize, sum: &mut [Option<Tensor>], grads: &candle_core::backprop::GradStore, extra: &[(Var, Tensor)],
              added: &mut Vec<(usize, Tensor)>) -> Result<()> {
        for (slot, var) in self.vars.iter().enumerate() {
            let id = var.as_tensor().id();
            for g in grads.get(var).into_iter().chain(extra.iter().filter(|(v, _)| v.as_tensor().id() == id).map(|(_, g)| g)) {
                if INJECT_COMMIT_OOM.load(std::sync::atomic::Ordering::Relaxed) == m && added.len() == INJECT_COMMIT_AFTER.load(std::sync::atomic::Ordering::Relaxed) {
                    INJECT_COMMIT_OOM.store(usize::MAX, std::sync::atomic::Ordering::Relaxed);
                    bail!("injected for a test: DriverError(CUDA_ERROR_OUT_OF_MEMORY, \"out of memory\")");
                }
                let next = match &sum[slot] {
                    Some(acc) => (acc + g)?,
                    None => g.clone(),
                };
                sum[slot] = Some(next);
                added.push((slot, g.clone()));
            }
        }
        Ok(())
    }

    /// Take back the parts a [`Runner::commit`] that failed half-way added (after the pools are trimmed, so the
    /// subtraction has room): exact for a slot that was empty before, one rounding of `(s + g) − g` otherwise. Only
    /// needed when the device runs out of memory while the gradients are being added up, after the whole micro-step.
    fn uncommit(sum: &mut [Option<Tensor>], before_empty: &[bool], added: Vec<(usize, Tensor)>) -> Result<()> {
        let mut first_part: Vec<bool> = before_empty.to_vec();
        let mut plan: Vec<(usize, Tensor, bool)> = Vec::with_capacity(added.len());
        for (slot, g) in added {
            let first = std::mem::replace(&mut first_part[slot], false);
            plan.push((slot, g, first));
        }
        for (slot, g, first) in plan.into_iter().rev() {
            let Some(s) = sum[slot].take() else { continue };
            // the slot's first part into an empty sum is the sum: dropping it is exact
            sum[slot] = if first { None } else { Some((s - g)?) };
        }
        Ok(())
    }

    /// One optimizer step at rate `lr` over units `step·accumulation ..`, as many as `count`, `seqs_per_pass` of them
    /// a pass. Returns the mean loss over the step's conversations before the update (and the gradient is the mean
    /// over them too: one conversation's tokens weigh what they weigh unpacked). A loss that is not a number stops
    /// the run without touching the adapters.
    fn step(&mut self, step: usize, count: usize, lr: f64) -> Result<f64> {
        self.lisa_select(step, true)?;
        let mut sum: Vec<Option<Tensor>> = vec![None; self.vars.len()];
        let mut loss_sum = 0.0;
        let mut aux_sum: Option<f64> = None;
        let mut conversations = 0usize;
        let mut k = 0;
        while k < count {
            let m = step * self.accumulation + k;
            // a pass that runs out of device memory (anywhere in its forward or backward pass, a checkpoint replay
            // included, or while its gradients are added up) is thrown away whole and run once more after the pools
            // are trimmed, with half the sequences when it had several (and so for the rest of the run). The step's
            // sums take its gradients only after its backward pass has finished, so nothing is counted twice, and they
            // are added in place as always: no second set of gradients is held.
            let mut attempt = 0;
            let (n, value, aux, tokens, convs) = loop {
                let n = self.seqs_per_pass.min(count - k).max(1);
                let before_empty: Vec<bool> = sum.iter().map(Option::is_none).collect();
                let mut added = Vec::new();
                let tried = self.micro(step, m, n, lr).and_then(|o| {
                    self.commit(m, &mut sum, &o.grads, &o.extra, &mut added)?;
                    Ok((o.value, o.aux, o.tokens, o.conversations))
                });
                match tried {
                    Ok((value, aux, tokens, convs)) => break (n, value, aux, tokens, convs),
                    Err(e) if attempt == 0 && is_out_of_memory(&e) => {
                        attempt += 1;
                        drop(self.model.take_checkpoint_grads());
                        self.model.synchronize()?;
                        let undone = added.len();
                        Self::uncommit(&mut sum, &before_empty, added)?;
                        let fewer = if n > 1 {
                            self.seqs_per_pass = (n / 2).max(1);
                            format!(" with {} sequences per pass instead of {n} (for the rest of the run)", self.seqs_per_pass)
                        } else {
                            String::new()
                        };
                        // what the cards hold now (not the high-water marks): after the trim, the reserve should be
                        // back at the use, which is the run's fixed state (adapters, moments, stream buffers)
                        let now = self.model.memory_now();
                        let line = format!("step {} micro-step {m}{}: out of device memory ({}); its tensors dropped{}, the pools trimmed{}, trying it once more{fewer}",
                            step + 1, if n > 1 { format!(" (a pass of {n})") } else { String::new() }, first_line(&e),
                            if undone > 0 { format!(", {undone} gradient parts taken back") } else { String::new() },
                            if now.is_empty() { String::new() } else {
                                format!(" (now {})", now.iter().enumerate().map(|(d, (u, r))| format!("dev{d} {} MiB reserved, {} MiB in use", r >> 20, u >> 20)).collect::<Vec<_>>().join("; ")) });
                        // (train() says it even when the retry fails too and the step ends the run)
                        self.pending.push(line);
                    }
                    Err(e) => return Err(e),
                }
            };
            loss_sum += value;
            if let Some(a) = aux {
                aux_sum = Some(aux_sum.unwrap_or(0.0) + a * convs as f64);
            }
            conversations += convs;
            self.tokens_seen += tokens;
            // the backward pass's buffers are free now: let the devices' memory pools take them back
            self.model.synchronize()?;
            let marks = self.model.memory_marks(true);
            if let Some(path) = self.mem_trace.as_ref().filter(|_| !marks.is_empty()) {
                let mib = |b: u64| b >> 20;
                let which = if n > 1 { format!("{m}-{}", m + n - 1) } else { m.to_string() };
                let line = format!("step {} micro {which} tokens {tokens} | {}\n", step + 1,
                    marks.iter().enumerate().map(|(i, (u, r))| format!("dev{i} used-high {} MiB reserved-high {} MiB", mib(*u), mib(*r))).collect::<Vec<_>>().join(" | "));
                use std::io::Write as _;
                if let Some(d) = path.parent() { let _ = std::fs::create_dir_all(d); }
                if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(path) {
                    let _ = f.write_all(line.as_bytes());
                }
            }
            k += n;
        }
        let conversations = conversations.max(1);
        // the mean gradient over the step's conversations, clipped by its norm over all adapters
        let mut norm_sq = 0.0f64;
        for g in sum.iter_mut().flatten() {
            *g = (&*g / conversations as f64)?;
            norm_sq += g.sqr()?.sum_all()?.to_scalar::<f32>()? as f64;
        }
        let norm = norm_sq.sqrt();
        if self.max_grad_norm > 0.0 && norm > self.max_grad_norm {
            for g in sum.iter_mut().flatten() {
                *g = (&*g * (self.max_grad_norm / norm))?;
            }
        }
        self.opt.params.lr = lr;
        self.opt.step_grads(&sum).map_err(|e| anyhow!("{e}"))?;
        drop(sum);
        self.model.synchronize()?;
        self.last_aux = aux_sum.map(|a| a / conversations as f64);
        Ok(loss_sum / conversations as f64)
    }

    /// The mean loss over the held-out conversations (each its own mean); packed and several sequences a pass when the
    /// training is, which gives the same mean in fewer passes.
    fn eval_loss(&mut self) -> Result<Option<f64>> {
        if self.data.eval.is_empty() {
            return Ok(None);
        }
        let n = self.data.eval.len();
        let mut sum = 0.0;
        if !self.pack_eval {
            for i in 0..n {
                let ex = self.data.eval[i].clone();
                let h = self.hidden(&ex)?;
                sum += self.model.loss_from_hidden(&h, &ex.tokens, &ex.supervised, &Ctx::inference())?.to_scalar::<f32>()? as f64;
            }
            return Ok(Some(sum / n as f64));
        }
        let order: Vec<usize> = (0..n).collect();
        let units = if self.sched.packed {
            let lens: Vec<usize> = self.data.eval.iter().map(|e| e.tokens.len()).collect();
            crate::packing::pack(&lens, &order, self.max_len)
        } else {
            order.into_iter().map(|i| vec![i]).collect()
        };
        for chunk in units.chunks(self.seqs_per_pass.max(1)) {
            let pass = Pass::of(&self.data.eval, chunk);
            let h = self.hidden_pass(&pass)?;
            sum += self.model.loss_packed(&h, &pass.layout, &pass.tokens, &pass.supervised, &Ctx::inference())?.to_scalar::<f32>()? as f64;
        }
        Ok(Some(sum / n as f64))
    }

    fn save_adapter(&self, dir: &Path, base: &Path, spec: &LoraSpec) -> Result<()> {
        std::fs::create_dir_all(dir)?;
        let tensors: HashMap<String, Tensor> = model::adapter_tensors(&self.model).into_iter().collect();
        candle_core::safetensors::save(&tensors, dir.join(ADAPTER_FILE))?;
        std::fs::write(dir.join(CONFIG_FILE), serde_json::to_string_pretty(&model::adapter_config(spec, &self.model.cfg, &base.display().to_string()))?)?;
        if self.private {
            crate::private_data::mark_config_private(&dir.join(CONFIG_FILE))?;
        }
        Ok(())
    }

    fn save_checkpoint(&self, dir: &Path, s: &TrainSettings, step: usize, fingerprint: &Value, losses: &[f64]) -> Result<()> {
        self.save_adapter(dir, &s.model, &s.lora)?;
        let mut moments: HashMap<String, Tensor> = HashMap::new();
        for (name, (_, m, v)) in self.names.iter().zip(&self.opt.slots) {
            moments.insert(format!("{name}.exp_avg"), m.as_tensor().detach());
            moments.insert(format!("{name}.exp_avg_sq"), v.as_tensor().detach());
        }
        candle_core::safetensors::save(&moments, dir.join(OPTIMIZER_FILE))?;
        let mut state = json!({"global_step": step, "adam_step": self.opt.step_t, "settings": fingerprint, "losses": losses});
        if self.lisa.is_some() {
            state["slot_reset"] = json!(self.opt.slot_reset);
        }
        std::fs::write(dir.join(STATE_FILE), serde_json::to_string_pretty(&state)?)?;
        Ok(())
    }

    /// Restore adapters, moments and the step from a checkpoint made with the same settings; returns the step and
    /// the losses so far. Everything is checked before anything is changed.
    fn load_checkpoint(&mut self, dir: &Path, fingerprint: &Value) -> Result<(usize, Vec<f64>)> {
        let state: Value = serde_json::from_str(&std::fs::read_to_string(dir.join(STATE_FILE)).with_context(|| format!("{}: not a checkpoint (no {STATE_FILE})", dir.display()))?)?;
        if &state["settings"] != fingerprint {
            let differ: Vec<String> = fingerprint.as_object().into_iter().flatten()
                .filter(|(k, v)| &&state["settings"][k.as_str()] != v)
                .map(|(k, v)| format!("{k}: checkpoint {}, now {v}", state["settings"][k.as_str()])).collect();
            bail!("{} was made with other settings ({}); a resumed run must repeat them", dir.display(), differ.join("; "));
        }
        // on the host, then to each variable's device: a variable on the second card never gets a GPU-to-GPU copy
        // (candle's dtod between two cards' contexts does not wait for its source; see `model::move_to`)
        let adapter = candle_core::safetensors::load(dir.join(ADAPTER_FILE), &Device::Cpu)?;
        let moments = candle_core::safetensors::load(dir.join(OPTIMIZER_FILE), &Device::Cpu)?;
        for (name, var) in self.names.iter().zip(&self.vars) {
            for (key, map) in [(name.clone(), &adapter), (format!("{name}.exp_avg"), &moments), (format!("{name}.exp_avg_sq"), &moments)] {
                let t = map.get(&key).ok_or_else(|| anyhow!("{}: no tensor {key}", dir.display()))?;
                if t.dims() != var.as_tensor().dims() {
                    bail!("{key}: shape {:?} in the checkpoint, {:?} in the model", t.dims(), var.as_tensor().dims());
                }
            }
        }
        let step = state["global_step"].as_u64().ok_or_else(|| anyhow!("{STATE_FILE}: no global_step"))? as usize;
        let adam = state["adam_step"].as_u64().ok_or_else(|| anyhow!("{STATE_FILE}: no adam_step"))? as usize;
        for (name, (theta, m, v)) in self.names.iter().zip(&self.opt.slots) {
            theta.set(&model::move_to(&adapter[name], theta.device())?)?;
            m.set(&model::move_to(&moments[&format!("{name}.exp_avg")], m.device())?)?;
            v.set(&model::move_to(&moments[&format!("{name}.exp_avg_sq")], v.device())?)?;
        }
        self.opt.step_t = adam;
        if let Some(r) = state["slot_reset"].as_array() {
            if r.len() != self.opt.slot_reset.len() {
                bail!("{STATE_FILE}: slot_reset has {} entries for {} variables", r.len(), self.opt.slot_reset.len());
            }
            self.opt.slot_reset = r.iter().map(|v| v.as_u64().unwrap_or(0) as usize).collect();
        }
        // LISA takes up the interval the checkpoint's last step was in, its moments as restored
        if step > 0 {
            self.lisa_select(step - 1, false)?;
            self.pending.clear();
        }
        let losses = state["losses"].as_array().map(|a| a.iter().filter_map(Value::as_f64).collect()).unwrap_or_default();
        Ok((step, losses))
    }
}

/// The newest `checkpoint-<step>` folder of an output folder.
pub fn latest_checkpoint(out: &Path) -> Option<PathBuf> {
    std::fs::read_dir(out).ok()?.flatten()
        .filter_map(|e| {
            let step: usize = e.file_name().to_str()?.strip_prefix("checkpoint-")?.parse().ok()?;
            e.path().join(STATE_FILE).exists().then_some((step, e.path()))
        })
        .max_by_key(|(step, _)| *step).map(|(_, p)| p)
}

/// Train a LoRA adapter. `on_event` hears the progress; `should_stop` is asked after every step, and a yes writes a
/// checkpoint and ends the run (`stopped` in the report).
pub fn train(s: &TrainSettings, on_event: &mut dyn FnMut(Event), should_stop: &dyn Fn() -> bool) -> Result<TrainReport> {
    let mut owned;
    let s = if s.freeze_bottom.is_some() {
        owned = s.clone();
        owned.apply_freeze_bottom(ModelConfig::load(&s.model)?.num_hidden_layers)?;
        &owned
    } else {
        s
    };
    s.validate()?;
    let t0 = Instant::now();
    let trims_at_start = model::TRIMS.load(std::sync::atomic::Ordering::Relaxed);
    let data = prepare_data(s)?;
    for n in &data.notes {
        on_event(Event::Log(n.clone()));
    }
    let (n_train, n_eval) = (data.train.len(), data.eval.len());
    let train_tokens: usize = data.train.iter().map(|e| e.tokens.len()).sum();
    if s.seqs_per_pass == 0 {
        on_event(Event::Log("--seqs-per-pass auto is resolved by the command from the cards' free memory; 1 here".into()));
    }
    let mut run = Runner::new(s, data, on_event)?;
    // the units (conversations, or packed sequences) the run takes, and its steps
    let units = run.sched.total();
    let total = total_units_steps(units, s.accumulation, s.max_steps);
    let total_micro = units.min(total * s.accumulation);
    let warmup = (s.warmup_ratio * total as f64).ceil() as usize;
    let fingerprint = s.fingerprint(n_train, total, units);
    if s.pack {
        let per = run.sched.per_epoch();
        on_event(Event::Log(format!("packing: {n_train} conversations ({train_tokens} tokens) into {per} sequences of up to {} tokens an epoch ({:.1} conversations and {:.0} % of the cap a sequence); a step is {} sequences",
                                    s.max_len, n_train as f64 / per.max(1) as f64, 100.0 * train_tokens as f64 / (per.max(1) * s.max_len) as f64, s.accumulation)));
    }
    if run.seqs_per_pass > 1 || s.pack {
        on_event(Event::Log(format!("{} {} per pass through the blocks (each streamed block copied once for all of them, forward and in the replay)",
                                    run.seqs_per_pass, if s.pack { "sequences" } else { "conversations" })));
    }
    // trained on a private dataset: the adapter (and what is merged, converted and installed from it) is private too
    run.private = crate::private_data::is_private_dataset(&s.data) || s.eval_data.as_deref().is_some_and(crate::private_data::is_private_dataset);
    if run.private {
        on_event(Event::Log("the data is private: the adapter is marked private (prastara_private), and so is what is made from it".into()));
    }
    let trainable: u64 = run.vars.iter().map(|v| v.as_tensor().elem_count() as u64).sum();
    on_event(Event::Log(format!(
        "{} conversations to train on, {} held out; {} steps of {} {} ({} warm-up); {} adapter parameters in blocks {} to {} of a {} base",
        n_train, n_eval, total, s.accumulation, if s.pack { "sequences" } else { "conversations" }, warmup, trainable, run.model.first_trained(), run.model.cfg.num_hidden_layers - 1, s.base_dtype.as_str())));

    let (mut step, mut losses) = (0usize, Vec::new());
    let resume_dir = match &s.resume {
        Some(Resume::Dir(d)) => Some(d.clone()),
        Some(Resume::Auto) => latest_checkpoint(&s.out),
        None => None,
    };
    if let Some(dir) = resume_dir {
        (step, losses) = run.load_checkpoint(&dir, &fingerprint)?;
        on_event(Event::Log(format!("resumed from {} at step {step}", dir.display())));
    }
    let start_step = step;
    let mut stopped = false;
    let mut eval = None;
    let t_steps = Instant::now();
    while step < total {
        let count = s.accumulation.min(total_micro - step * s.accumulation).max(1);
        let lr = hf_cosine_lr(s.lr, step, warmup, total);
        let stepped = run.step(step, count, lr);
        for l in std::mem::take(&mut run.pending) {
            on_event(Event::Log(l));
        }
        let loss = stepped?;
        losses.push(loss);
        step += 1;
        let done = (step - start_step) as f64;
        let eta = t_steps.elapsed().as_secs_f64() / done * (total - step) as f64;
        // a short run (up to SHORT_RUN_STEPS) reports every step: its few losses are the whole story
        if s.logging_steps <= 1 || total <= SHORT_RUN_STEPS || step % s.logging_steps == 0 || step == total || step == start_step + 1 {
            on_event(Event::Step { step, total, loss, lr, eta_seconds: eta, aux: run.last_aux });
        }
        if s.eval_steps > 0 && step % s.eval_steps == 0 && step < total {
            if let Some(l) = run.eval_loss()? {
                on_event(Event::Eval { step, loss: l });
            }
        }
        let stop_now = step < total && should_stop();
        if stop_now || (s.save_steps > 0 && step % s.save_steps == 0 && step < total) {
            let dir = s.out.join(format!("checkpoint-{step}"));
            run.save_checkpoint(&dir, s, step, &fingerprint, &losses)?;
            on_event(Event::Checkpoint { step, dir });
        }
        if stop_now {
            stopped = true;
            break;
        }
    }
    if !stopped {
        eval = run.eval_loss()?;
        if let Some(l) = eval {
            on_event(Event::Eval { step, loss: l });
        }
        // the finished adapter, in PEFT's format, at the top of the output folder
        run.save_adapter(&s.out, &s.model, &s.lora)?;
    }
    let (hits, misses, bytes) = run.cache.as_ref().map_or((0, 0, 0), |c| (c.hits, c.misses, c.bytes_written));
    let stream = run.model.stream().map(|st| st.stats());
    if let (Some(st), Some(ls)) = (&stream, run.model.stream()) {
        on_event(Event::Log(format!("layer streaming: {} block copies ({:.2} GB), {} of the runs found their block prefetched, {} waited for it ({} tier)",
                                    st.loads, st.bytes as f64 / 1e9, st.prefetch_hits, st.demand_loads, ls.tier.as_str())));
        if let Some(note) = crate::layer_stream::host_memory_note() {
            on_event(Event::Log(format!("host memory at the end: {note}")));
        }
    }
    let trims = model::TRIMS.load(std::sync::atomic::Ordering::Relaxed) - trims_at_start;
    if trims > 0 {
        on_event(Event::Log(format!("{trims} block-boundary trims of the devices' memory pools (reserve more than {} MiB above use)",
                                    model::trim_gap_bytes() >> 20)));
    }
    let report = TrainReport {
        out: s.out.clone(), steps_done: step, total_steps: total, stopped, losses: losses.clone(), eval_loss: eval, train_examples: n_train,
        eval_examples: n_eval, trainable_params: trainable, seconds: t0.elapsed().as_secs_f64(), tokens_seen: run.tokens_seen, cache_hits: hits,
        cache_misses: misses, cache_bytes: bytes, lower_in_memory: run.lower_in_memory, peak_rss_bytes: peak_rss_bytes(), stream: stream.clone(),
    };
    std::fs::create_dir_all(&s.out)?;
    std::fs::write(s.out.join(REPORT_FILE), serde_json::to_string_pretty(&json!({
        "steps_done": step, "total_steps": total, "stopped": stopped, "losses": losses, "eval_loss": eval, "train_examples": n_train,
        "eval_examples": n_eval, "trainable_params": trainable, "seconds": report.seconds, "tokens_seen": report.tokens_seen,
        "peak_rss_bytes": report.peak_rss_bytes,
        // page-locked memory the kernel counts twice in VmRSS / VmHWM (see layer_stream::driver_mapped_bytes)
        "rss_counted_twice_bytes": crate::layer_stream::driver_mapped_bytes(), "private": run.private, "settings": fingerprint, "base_model": s.model.display().to_string(),
        "activation_cache": {"hits": hits, "written": misses, "bytes_written": bytes, "lower_blocks_in_memory": report.lower_in_memory},
        "layer_stream": stream.map(|st| json!({"tier": run.model.stream().map(|l| l.tier.as_str()), "buffers": s.stream_buffers, "block_copies": st.loads,
                                               "bytes_copied": st.bytes, "prefetch_hits": st.prefetch_hits, "demand_loads": st.demand_loads})),
        "lisa": s.lisa.map(|l| json!({"layers": l.layers, "interval": l.interval, "seed": l.seed})),
        "packing": {"packed": s.pack, "units": units, "units_per_epoch": run.sched.per_epoch(), "seqs_per_pass": run.seqs_per_pass},
    }))?)?;
    Ok(report)
}

// ---------------------------------------------------------------------------------------------------------------------
// the learning-rate range test
// ---------------------------------------------------------------------------------------------------------------------

/// The range test's settings: the rate rises geometrically from `start` to `end` over `steps` optimizer steps.
#[derive(Debug, Clone, Copy)]
pub struct FindLr {
    pub start: f64,
    pub end: f64,
    pub steps: usize,
}

impl Default for FindLr {
    fn default() -> Self {
        Self { start: 1e-7, end: 1.0, steps: 40 }
    }
}

/// What the curve says.
#[derive(Debug, Clone, PartialEq)]
pub struct LrSuggestion {
    /// the rate where the smoothed loss was falling fastest; `None` when it never fell
    pub suggested: Option<f64>,
    /// the rate at the lowest smoothed loss (too high to train at: the loss turns there)
    pub min_loss_lr: f64,
    /// the first rate after the minimum whose smoothed loss is over 4× the minimum
    pub diverged_at: Option<f64>,
    pub smoothed: Vec<f64>,
}

/// The outcome of [`find_lr`].
#[derive(Debug, Clone)]
pub struct LrReport {
    pub lrs: Vec<f64>,
    pub losses: Vec<f64>,
    pub suggestion: LrSuggestion,
    /// the run ended early because the loss blew up (which is what a range test is for)
    pub diverged: bool,
}

/// Read a suggestion off a range test's (rate, loss) curve: smooth the losses (an exponential average, bias
/// corrected), find the minimum, and before it the place where the loss falls fastest per unit of log-rate. The rate
/// at the start of that fall is the suggestion (a step back from the steepest point, as Soup and fastai advise).
pub fn suggest_lr(lrs: &[f64], losses: &[f64]) -> Result<LrSuggestion> {
    if lrs.len() != losses.len() {
        bail!("{} rates for {} losses", lrs.len(), losses.len());
    }
    if lrs.len() < 4 {
        bail!("a range test needs at least 4 steps, this one has {}", lrs.len());
    }
    const BETA: f64 = 0.7;
    let mut avg = 0.0;
    let smoothed: Vec<f64> = losses.iter().enumerate().map(|(i, l)| {
        avg = BETA * avg + (1.0 - BETA) * l;
        avg / (1.0 - BETA.powi(i as i32 + 1))
    }).collect();
    let min_idx = (0..smoothed.len()).min_by(|a, b| smoothed[*a].total_cmp(&smoothed[*b])).unwrap_or(0);
    let diverged_at = (min_idx + 1..smoothed.len()).find(|i| smoothed[*i].abs() > 4.0 * smoothed[min_idx].abs()).map(|i| lrs[i]);
    let mut best = (0.0f64, 0usize);
    for i in 1..=min_idx {
        let d_lr = lrs[i].ln() - lrs[i - 1].ln();
        if d_lr > 0.0 {
            let slope = (smoothed[i] - smoothed[i - 1]) / d_lr;
            if slope < best.0 {
                best = (slope, i);
            }
        }
    }
    Ok(LrSuggestion { suggested: (best.0 < 0.0).then(|| lrs[best.1 - 1]), min_loss_lr: lrs[min_idx], diverged_at, smoothed })
}

/// The learning-rate range test: a short run whose rate rises geometrically, recording each step's loss. Nothing is
/// written; the adapters trained here are thrown away.
pub fn find_lr(s: &TrainSettings, opts: FindLr, on_event: &mut dyn FnMut(Event)) -> Result<LrReport> {
    let mut owned;
    let s = if s.freeze_bottom.is_some() {
        owned = s.clone();
        owned.apply_freeze_bottom(ModelConfig::load(&s.model)?.num_hidden_layers)?;
        &owned
    } else {
        s
    };
    s.validate()?;
    if !(opts.start > 0.0 && opts.end > opts.start) || opts.steps < 4 {
        bail!("a range test needs 0 < start < end and at least 4 steps");
    }
    let data = prepare_data(s)?;
    for n in &data.notes {
        on_event(Event::Log(n.clone()));
    }
    let mut run = Runner::new(s, data, on_event)?;
    let (mut lrs, mut losses) = (Vec::new(), Vec::new());
    let mut best = f64::INFINITY;
    let mut diverged = false;
    for step in 0..opts.steps {
        let lr = opts.start * (opts.end / opts.start).powf(step as f64 / (opts.steps - 1) as f64);
        let loss = match run.step(step, s.accumulation, lr) {
            Ok(l) => l,
            // a loss that is not a number is the end of the curve, not a failure
            Err(e) if e.to_string().contains("the loss became") => {
                diverged = true;
                break;
            }
            Err(e) => return Err(e),
        };
        lrs.push(lr);
        losses.push(loss);
        on_event(Event::Step { step: step + 1, total: opts.steps, loss, lr, eta_seconds: 0.0, aux: run.last_aux });
        best = best.min(loss);
        if step >= 4 && loss > 4.0 * best {
            diverged = true;
            break;
        }
    }
    let suggestion = suggest_lr(&lrs, &losses)?;
    Ok(LrReport { lrs, losses, suggestion, diverged })
}
