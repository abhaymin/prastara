//! A real Llama-family decoder in candle, with LoRA hooks — the model the Workshop trains.
//!
//! Loads a Hugging Face folder (`config.json` + safetensors in F32, F16 or BF16) for the dense decoders the converter
//! handles: Llama and Mistral (RMSNorm, RoPE, grouped-query attention, SwiGLU), Qwen2 (the same with q/k/v biases) and
//! Qwen3 (the same with a per-head RMSNorm on q and k and an explicit `head_dim`); embeddings tied to the output head
//! or not. One sequence at a time (batch 1), causal mask, logits out. It follows transformers' modules step by step and
//! is checked against transformers' own logits (`tests/model_parity.rs`, fixtures made once with
//! `tests/fixtures/make-forward-reference.py`).
//!
//! **LoRA.** Any of `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` gets `y = W x + s · B A drop(x)`
//! with `s = alpha / r`, or `alpha / sqrt(r)` with rsLoRA. A is random (PEFT's bound, seeded), B is zero, so an
//! untrained adapter leaves the model unchanged. Only A and B are `Var`s; the base never is.
//!
//! **What costs no gradient.** candle records an operation only when one of its inputs descends from a `Var`. With
//! LoRA in the top N blocks only ([`LoraSpec::train_layers`]), the embedding and the blocks below hold no `Var`, so
//! they run as plain inference: nothing is kept for a backward pass. [`LlamaModel::lower`] returns the hidden state
//! entering the first trained block; [`LlamaModel::upper`] continues from it. That split is what the activation cache
//! (`crate::finetune`) stores, and what lets the lower blocks leave memory ([`LlamaModel::drop_lower`]).
//!
//! **A half-precision base.** With [`BaseDtype::F16`] or [`BaseDtype::BF16`] every weight matrix stays in 16 bits and
//! is widened to f32 only for the one product that uses it; activations, LoRA and the loss are f32. Inside the trained
//! region a plain `matmul` would keep that widened copy alive in the graph until the backward pass (f16 + f32 of the
//! whole region: worse than f32 alone), so there the product goes through [`FrozenLinear`], a custom operation that
//! keeps only the 16-bit weight and widens it again, briefly, to send the gradient back. Every frozen product goes
//! through it, in f32 too: candle's `matmul` backward would also compute a (useless, weight-sized) gradient for W.

use std::cell::{Cell, RefCell};
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use anyhow::{anyhow, bail, Context as _, Result};
use candle_core::{CpuStorage, CustomOp1, CustomOp3, DType, Device, Layout, Shape, Tensor, Var, D};
use serde_json::Value;

/// The decoder families this forward pass covers (each checked against transformers).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Family {
    Llama,
    Mistral,
    Qwen2,
    Qwen3,
}

impl Family {
    pub fn as_str(self) -> &'static str {
        match self {
            Family::Llama => "llama",
            Family::Mistral => "mistral",
            Family::Qwen2 => "qwen2",
            Family::Qwen3 => "qwen3",
        }
    }
}

/// What `config.json` says about the model, as far as the forward pass and the memory estimate need it.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelConfig {
    pub family: Family,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    /// `head_dim` when the config names it (Qwen3, some Mistral), else hidden / heads
    pub head_dim: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub max_position_embeddings: usize,
    pub tie_word_embeddings: bool,
    /// Avarta: blocks `start..end` run `passes` times with the same weights (`"prastara_loop"` in config.json)
    pub loop_: Option<LoopSpec>,
    /// YaRN position scaling (`rope_scaling` / `rope_parameters` with `rope_type: yarn`)
    pub yarn: Option<Yarn>,
    /// the window of the sliding-attention layers (`sliding_window`), with `layer_types` naming which layers slide
    pub sliding_window: Option<usize>,
    /// per layer: true for `sliding_attention` (empty: every layer attends to everything)
    pub layer_sliding: Vec<bool>,
    /// LoRA upcycling: every trained block's feed-forward becomes a mixture of experts (`"prastara_moe"` in config.json)
    pub moe: Option<MoeSpec>,
}

/// YaRN (transformers' `_compute_yarn_parameters`): rotary frequencies past the original length are interpolated
/// by `factor`, the fast ones kept, a linear ramp between `beta_fast` and `beta_slow` rotations; cos and sin are
/// scaled by `attention_factor` (0.1 ln(factor) + 1 unless given).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Yarn {
    pub factor: f64,
    pub original_max_position_embeddings: usize,
    pub beta_fast: f64,
    pub beta_slow: f64,
    pub attention_factor: f64,
    pub truncate: bool,
}

impl Yarn {
    pub fn new(factor: f64, original: usize) -> Yarn {
        Yarn { factor, original_max_position_embeddings: original, beta_fast: 32.0, beta_slow: 1.0,
               attention_factor: if factor <= 1.0 { 1.0 } else { 0.1 * factor.ln() + 1.0 }, truncate: true }
    }

    fn from_json(r: &Value, max_pos: usize) -> Result<Yarn> {
        let factor = r["factor"].as_f64().ok_or_else(|| anyhow!("rope_scaling yarn: no factor"))?;
        if factor < 1.0 {
            bail!("rope_scaling yarn: factor {factor} is below 1");
        }
        let mut y = Yarn::new(factor, r["original_max_position_embeddings"].as_u64().map_or(max_pos, |x| x as usize));
        if let Some(b) = r["beta_fast"].as_f64() { y.beta_fast = b; }
        if let Some(b) = r["beta_slow"].as_f64() { y.beta_slow = b; }
        if let Some(a) = r["attention_factor"].as_f64() { y.attention_factor = a; }
        if r.get("mscale").is_some_and(|v| !v.is_null()) || r.get("mscale_all_dim").is_some_and(|v| !v.is_null()) {
            bail!("rope_scaling yarn with mscale / mscale_all_dim (DeepSeek's form) is not implemented");
        }
        if let Some(t) = r["truncate"].as_bool() { y.truncate = t; }
        Ok(y)
    }

    /// The rotary inverse frequencies for a head of `dim` with base `theta`.
    pub fn inv_freq(&self, dim: usize, theta: f64) -> Vec<f64> {
        let correction = |rot: f64| dim as f64 * (self.original_max_position_embeddings as f64 / (rot * 2.0 * std::f64::consts::PI)).ln() / (2.0 * theta.ln());
        let (mut low, mut high) = (correction(self.beta_fast), correction(self.beta_slow));
        if self.truncate {
            low = low.floor();
            high = high.ceil();
        }
        let (low, high) = (low.max(0.0), high.min(dim as f64 - 1.0));
        let high = if low == high { high + 0.001 } else { high };
        (0..dim / 2).map(|i| {
            let pos = theta.powf(2.0 * i as f64 / dim as f64);
            let (extra, inter) = (1.0 / pos, 1.0 / (self.factor * pos));
            let ramp = ((i as f64 - low) / (high - low)).clamp(0.0, 1.0);
            let keep = 1.0 - ramp;
            inter * (1.0 - keep) + extra * keep
        }).collect()
    }
}

/// A loop over the core blocks (Avarta, docs/MULA-PLAN.md M1). Pass 1 is the base's own pass; every later pass p
/// takes the previous pass's output x, runs the core again to y, and keeps `x + g_p ⊙ (y - x)`, where the gate g_p
/// (one value per hidden channel, `loop.gate.<p>` in `loop.safetensors`) starts at zero: a model just converted
/// gives exactly the base's logits, and training opens the gates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LoopSpec {
    pub start: usize,
    pub end: usize,
    pub passes: usize,
}

/// The file beside the weights that holds the loop's gates.
pub const LOOP_FILE: &str = "loop.safetensors";

/// The file beside the weights of a fine-grained upcycled folder: per block `perm.<b>` (U32 `[ff]`, the dense
/// neurons in group order: group g is `perm[g·ff_e .. (g+1)·ff_e]`) and `router.<b>` (F32 `[experts, hidden]`, the
/// router's start), written by [`crate::moe_init`].
pub const MOE_INIT_FILE: &str = "moe_init.safetensors";

impl LoopSpec {
    pub fn from_json(v: &Value, layers: usize) -> Result<Option<LoopSpec>> {
        if v.is_null() {
            return Ok(None);
        }
        let n = |k: &str| v[k].as_u64().map(|x| x as usize).ok_or_else(|| anyhow!("config.prastara_loop.{k} is missing or not a number"));
        let l = LoopSpec { start: n("start")?, end: n("end")?, passes: n("passes")? };
        l.check(layers)?;
        Ok(Some(l))
    }

    pub fn check(&self, layers: usize) -> Result<()> {
        if self.start >= self.end || self.end > layers {
            bail!("the loop's blocks {}..{} must be a non-empty range inside the model's {layers}", self.start, self.end);
        }
        if self.passes < 2 || self.passes > 16 {
            bail!("the loop's passes must be 2 to 16 (got {})", self.passes);
        }
        Ok(())
    }

    pub fn to_json(&self) -> Value {
        serde_json::json!({"start": self.start, "end": self.end, "passes": self.passes})
    }
}

/// LoRA upcycling (Mula-MoE, plan Part C): the dense feed-forward of every trained block becomes `experts` experts
/// that share its frozen weights, each with its own LoRA on gate, up and down, behind a router `[experts, hidden]`
/// (softmax, the best `top_k`, their weights renormalised to sum to 1, as the engine's qwen3moe routes).
///
/// **Full width** (`expert_ff` None, the 57B-A8B design): every expert is the whole dense feed-forward; with B = 0
/// every expert is the base feed-forward and the weights sum to 1, so an upcycled model starts exactly as its base.
///
/// **Fine-grained** (`expert_ff` = ff_e below the intermediate size ff, the 56B-A5B design): the dense feed-forward's
/// ff neurons are split into G = ff / ff_e groups (`split`: contiguous, or clustered by co-activation on calibration
/// text, see [`crate::moe_init`]); expert e is group `e mod G`, copy `e div G` (so `experts` is a multiple of G). An
/// expert's gate and up rows and down columns are its group's slice of the frozen dense weights (a permutation held
/// in `moe_init.safetensors`, applied to the weights once at load: nothing is copied), plus its own LoRA. A token's
/// `top_k` experts are summed with their renormalised weights times `top_k` (so k equal weights give the plain sum of
/// k groups, as the dense model sums all G); `export-moe` folds that factor into the down weights. With top_k < G a
/// token uses only k·ff_e of the ff neurons, so **at the start the model is NOT the dense model**: it is the
/// MoEfication approximation (the k groups the router's start prefers), and `upcycle --dense-start-check` measures
/// how far it is. With E = G and top_k = G (and a uniform router) it is the dense model exactly.
///
/// `export-moe` folds each expert's LoRA into its own weights: a real `qwen3moe` model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MoeSpec {
    pub experts: usize,
    pub top_k: usize,
    /// fine-grained: each expert is `expert_ff` of the dense feed-forward's neurons (None: full width)
    pub expert_ff: Option<usize>,
    /// the experts' LoRA rank (None: the adapter's rank; 0: the experts are not adapted, only the router trains)
    pub expert_rank: Option<usize>,
    /// one LoRA A per neuron group, shared by its copies, each copy with its own B (a fine-grained memory saver)
    pub shared_a: bool,
    /// how the neurons were grouped (fine-grained only)
    pub split: MoeSplit,
}

/// How a fine-grained upcycle groups the dense feed-forward's neurons.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MoeSplit {
    /// neurons `g·ff_e .. (g+1)·ff_e` form group g
    Contiguous,
    /// balanced k-means over the neurons' co-activation on calibration text (MoEfication), groups of exactly ff_e
    Cluster,
}

impl MoeSplit {
    pub fn from_s(s: &str) -> Result<Self> {
        match s {
            "contiguous" => Ok(MoeSplit::Contiguous),
            "cluster" => Ok(MoeSplit::Cluster),
            _ => bail!("--split {s:?}: cluster or contiguous"),
        }
    }
    pub fn as_str(self) -> &'static str {
        match self {
            MoeSplit::Contiguous => "contiguous",
            MoeSplit::Cluster => "cluster",
        }
    }
}

/// The weight of the Switch-style load-balancing loss added to the training loss (`E · Σ_e f_e · P_e`, averaged over
/// the MoE blocks a forward pass ran: 1 when the router spreads the tokens evenly).
pub const MOE_AUX_COEF: f64 = 0.01;
/// The weight of ST-MoE's router z-loss (the mean squared log-sum-exp of the router's logits), fine-grained only: it
/// keeps 120 experts' logits from drifting large.
pub const MOE_Z_COEF: f64 = 0.001;

impl MoeSpec {
    /// The full-width design (every expert the whole feed-forward), the adapter's rank, no sharing.
    pub fn full(experts: usize, top_k: usize) -> MoeSpec {
        MoeSpec { experts, top_k, expert_ff: None, expert_rank: None, shared_a: false, split: MoeSplit::Contiguous }
    }

    pub fn from_json(v: &Value) -> Result<Option<MoeSpec>> {
        if v.is_null() {
            return Ok(None);
        }
        let n = |k: &str| v[k].as_u64().map(|x| x as usize).ok_or_else(|| anyhow!("config.prastara_moe.{k} is missing or not a number"));
        let opt = |k: &str| -> Result<Option<usize>> {
            match &v[k] {
                Value::Null => Ok(None),
                x => x.as_u64().map(|x| Some(x as usize)).ok_or_else(|| anyhow!("config.prastara_moe.{k} is not a number")),
            }
        };
        let m = MoeSpec { experts: n("experts")?, top_k: n("top_k")?, expert_ff: opt("expert_ff")?, expert_rank: opt("expert_rank")?,
                          shared_a: v["shared_a"].as_bool().unwrap_or(false),
                          split: v["split"].as_str().map(MoeSplit::from_s).transpose()?.unwrap_or(MoeSplit::Contiguous) };
        m.check()?;
        Ok(Some(m))
    }

    pub fn check(&self) -> Result<()> {
        if self.experts < 2 || self.experts > 256 {
            bail!("an upcycled model needs 2 to 256 experts (got {})", self.experts);
        }
        if self.top_k == 0 || self.top_k > self.experts {
            bail!("top-k {} must be 1 to the number of experts ({})", self.top_k, self.experts);
        }
        if self.expert_ff == Some(0) {
            bail!("--expert-ff 0: an expert needs at least one neuron");
        }
        Ok(())
    }

    /// The checks that need the dense model's shape: ff_e divides ff, the experts are whole copies of the G groups,
    /// and a token's k experts can be k different groups.
    pub fn check_with(&self, cfg: &ModelConfig) -> Result<()> {
        self.check()?;
        let ff = cfg.intermediate_size;
        if let Some(fe) = self.expert_ff {
            if fe > ff || ff % fe != 0 {
                bail!("--expert-ff {fe} must divide the intermediate size {ff} (e.g. {})", ff / 12);
            }
            let g = ff / fe;
            if g > 1 {
                if self.experts % g != 0 {
                    bail!("{} experts are not whole copies of the {g} neuron groups of {fe} (ff {ff} / {fe}): use a multiple of {g}", self.experts);
                }
                if self.top_k > g {
                    bail!("top-k {} is more than the {g} neuron groups: a token's experts are different groups", self.top_k);
                }
            }
        }
        Ok(())
    }

    /// Neurons per expert: `expert_ff`, or the whole intermediate size.
    pub fn ff_e(&self, cfg: &ModelConfig) -> usize {
        self.expert_ff.unwrap_or(cfg.intermediate_size).min(cfg.intermediate_size)
    }

    /// Neuron groups G (1: full width).
    pub fn groups(&self, cfg: &ModelConfig) -> usize {
        cfg.intermediate_size / self.ff_e(cfg).max(1)
    }

    /// Fine-grained: the experts are slices of the dense feed-forward (G > 1).
    pub fn fine(&self, cfg: &ModelConfig) -> bool {
        self.groups(cfg) > 1
    }

    /// What each chosen expert's output is multiplied by: `top_k` when fine-grained (k weights of about 1/k then give
    /// the plain sum of k groups), 1 at full width (identical copies, weights summing to 1).
    pub fn out_scale(&self, cfg: &ModelConfig) -> f64 {
        if self.fine(cfg) { self.top_k as f64 } else { 1.0 }
    }

    /// The experts' LoRA rank given the adapter's.
    pub fn rank(&self, r: usize) -> usize {
        self.expert_rank.unwrap_or(r)
    }

    pub fn to_json(&self) -> Value {
        let mut v = serde_json::json!({"experts": self.experts, "top_k": self.top_k});
        if let Some(fe) = self.expert_ff {
            v["expert_ff"] = serde_json::json!(fe);
            v["split"] = serde_json::json!(self.split.as_str());
        }
        if let Some(r) = self.expert_rank {
            v["expert_rank"] = serde_json::json!(r);
        }
        if self.shared_a {
            v["shared_a"] = serde_json::json!(true);
        }
        v
    }

    /// `(total, active per token)` parameters of the exported mixture: each block's feed-forward becomes `experts`
    /// experts of ff_e neurons (`top_k` of them active) and a router, everything else as the base.
    pub fn params(&self, cfg: &ModelConfig) -> (u64, u64) {
        let layers = cfg.num_hidden_layers as u64;
        let ffn = 3 * (cfg.hidden_size * cfg.intermediate_size) as u64;
        let expert = 3 * (cfg.hidden_size * self.ff_e(cfg)) as u64;
        let router = (self.experts * cfg.hidden_size) as u64;
        let base = cfg.total_params() - layers * ffn;
        (base + layers * (self.experts as u64 * expert + router), base + layers * (self.top_k as u64 * expert + router))
    }

    /// Trainable parameters of one upcycled block at adapter rank `r`: the router and the experts' LoRAs (gate and up
    /// `[ff_e, h]`, down `[h, ff_e]`), with A shared per group when `shared_a`.
    pub fn block_trainable(&self, cfg: &ModelConfig, r: usize) -> u64 {
        let (h, fe, e) = (cfg.hidden_size as u64, self.ff_e(cfg) as u64, self.experts as u64);
        let re = self.rank(r) as u64;
        let lora = if self.shared_a {
            let g = self.groups(cfg) as u64;
            g * re * (2 * h + fe) + e * re * (2 * fe + h)
        } else {
            e * 3 * re * (h + fe)
        };
        e * h + lora
    }
}

impl ModelConfig {
    pub fn load(dir: &Path) -> Result<Self> {
        let p = dir.join("config.json");
        let text = std::fs::read_to_string(&p).with_context(|| format!("{}: not a Hugging Face model folder (no config.json)", dir.display()))?;
        let v: Value = serde_json::from_str(&text).with_context(|| format!("{}", p.display()))?;
        Self::from_json(&v).with_context(|| format!("{}", p.display()))
    }

    /// Read a config, refusing by name what this trainer cannot run (rather than training something else quietly).
    pub fn from_json(c: &Value) -> Result<Self> {
        let arch = c["architectures"].as_array().and_then(|a| a.first()).and_then(Value::as_str).unwrap_or("");
        let model_type = c["model_type"].as_str().unwrap_or("");
        for key in ["num_experts", "num_local_experts", "num_experts_per_tok", "moe_intermediate_size"] {
            if c.get(key).is_some_and(|v| !v.is_null() && v.as_u64() != Some(0)) {
                bail!("a mixture-of-experts model ({key} in config.json): training expert adapters is not done; only dense Llama, Mistral, Qwen2 and Qwen3 decoders train here");
            }
        }
        for key in ["vision_config", "audio_config", "vision_tower"] {
            if c.get(key).is_some_and(|v| !v.is_null()) {
                bail!("a multimodal model ({key} in config.json): vision and audio towers are not trained here; only text decoders are");
            }
        }
        let family = match (model_type, arch) {
            ("llama", _) => Family::Llama,
            ("mistral", _) => Family::Mistral,
            ("qwen2", _) => Family::Qwen2,
            ("qwen3", _) => Family::Qwen3,
            ("", a) if a.starts_with("Llama") => Family::Llama,
            ("", a) if a.starts_with("Mistral") => Family::Mistral,
            ("", a) if a.starts_with("Qwen2") => Family::Qwen2,
            ("", a) if a.starts_with("Qwen3") => Family::Qwen3,
            _ => bail!("architecture {arch:?} (model_type {model_type:?}) is not one this trainer runs: Llama, Mistral, Qwen2, Qwen3"),
        };
        let n = |k: &str| c[k].as_u64().map(|x| x as usize).ok_or_else(|| anyhow!("config.{k} is missing or not a number"));
        // transformers 5 writes rope_parameters {rope_theta, rope_type}; before that rope_theta and rope_scaling
        let rope = &c["rope_parameters"];
        let mut yarn = None;
        let max_pos_cfg = c["max_position_embeddings"].as_u64().map(|x| x as usize).unwrap_or(2048);
        // rope_scaling (transformers 4) wins over rope_parameters (transformers 5) when both are written
        let scaling = &c["rope_scaling"];
        let (src, kind) = match scaling["rope_type"].as_str().or(scaling["type"].as_str()) {
            Some(t) => (scaling, Some(t)),
            None => (rope, rope["rope_type"].as_str()),
        };
        if let Some(t) = kind {
            match t {
                "default" => {}
                "yarn" => yarn = Some(Yarn::from_json(src, max_pos_cfg)?),
                _ => bail!("rope scaling of type {t:?} is not implemented in this forward pass (default and yarn are)"),
            }
        } else if c.get("rope_scaling").is_some_and(|v| !v.is_null()) {
            bail!("config.rope_scaling is set, which this forward pass does not implement");
        }
        let layers_n = n("num_hidden_layers")?;
        let layer_sliding: Vec<bool> = match c["layer_types"].as_array() {
            Some(a) => {
                if a.len() != layers_n {
                    bail!("config.layer_types names {} layers, the model has {layers_n}", a.len());
                }
                a.iter().map(|t| match t.as_str() {
                    Some("full_attention") => Ok(false),
                    Some("sliding_attention") => Ok(true),
                    other => bail!("layer type {other:?}: full_attention or sliding_attention"),
                }).collect::<Result<_>>()?
            }
            None => Vec::new(),
        };
        let sliding_window = c["sliding_window"].as_u64().map(|x| x as usize).filter(|_| layer_sliding.iter().any(|s| *s));
        if layer_sliding.iter().any(|s| *s) && sliding_window.unwrap_or(0) == 0 {
            bail!("config.layer_types has sliding layers but no sliding_window");
        }
        if let Some(act) = c["hidden_act"].as_str() {
            if act != "silu" {
                bail!("hidden_act {act:?} is not implemented (silu only)");
            }
        }
        let hidden_size = n("hidden_size")?;
        let num_attention_heads = n("num_attention_heads")?;
        if num_attention_heads == 0 {
            bail!("config.num_attention_heads is 0");
        }
        let num_key_value_heads = c["num_key_value_heads"].as_u64().map(|x| x as usize).unwrap_or(num_attention_heads);
        if num_key_value_heads == 0 || num_attention_heads % num_key_value_heads != 0 {
            bail!("num_attention_heads {num_attention_heads} is not a multiple of num_key_value_heads {num_key_value_heads}");
        }
        let head_dim = c["head_dim"].as_u64().map(|x| x as usize).unwrap_or(hidden_size / num_attention_heads);
        if head_dim == 0 || head_dim % 2 != 0 {
            bail!("head_dim {head_dim} must be even and above 0 (rotary embeddings pair its halves)");
        }
        let cfg = ModelConfig {
            family,
            vocab_size: n("vocab_size")?,
            hidden_size,
            intermediate_size: n("intermediate_size")?,
            num_hidden_layers: n("num_hidden_layers")?,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            rms_norm_eps: c["rms_norm_eps"].as_f64().unwrap_or(1e-6),
            rope_theta: rope["rope_theta"].as_f64().or(c["rope_theta"].as_f64()).unwrap_or(10000.0),
            max_position_embeddings: c["max_position_embeddings"].as_u64().map(|x| x as usize).unwrap_or(2048),
            tie_word_embeddings: c["tie_word_embeddings"].as_bool().unwrap_or(false),
            loop_: LoopSpec::from_json(&c["prastara_loop"], n("num_hidden_layers")?)?,
            moe: MoeSpec::from_json(&c["prastara_moe"])?,
            yarn, sliding_window, layer_sliding,
        };
        if let Some(m) = cfg.moe {
            m.check_with(&cfg)?;
        }
        Ok(cfg)
    }

    /// `(out, in)` of one target's weight.
    pub fn target_shape(&self, t: Target) -> (usize, usize) {
        let (h, i) = (self.hidden_size, self.intermediate_size);
        let (q, kv) = (self.num_attention_heads * self.head_dim, self.num_key_value_heads * self.head_dim);
        match t {
            Target::Q => (q, h),
            Target::K | Target::V => (kv, h),
            Target::O => (h, q),
            Target::Gate | Target::Up => (i, h),
            Target::Down => (h, i),
        }
    }

    /// Parameters of one decoder block (matrices, norms, and the family's biases and q/k norms).
    pub fn block_params(&self) -> u64 {
        let matrices: u64 = Target::ALL.iter().map(|t| { let (o, i) = self.target_shape(*t); (o * i) as u64 }).sum();
        matrices + self.block_small_params()
    }

    /// The 1-D parameters of one block: two norms, plus Qwen2's q/k/v biases or Qwen3's q/k norms.
    pub fn block_small_params(&self) -> u64 {
        let (q, kv) = (self.num_attention_heads * self.head_dim, self.num_key_value_heads * self.head_dim);
        let extra = match self.family {
            Family::Qwen2 => q + 2 * kv,
            Family::Qwen3 => 2 * self.head_dim,
            _ => 0,
        };
        (2 * self.hidden_size + extra) as u64
    }

    /// All parameters: embeddings, blocks, the final norm and the output head unless it is tied to the embeddings.
    pub fn total_params(&self) -> u64 {
        let emb = (self.vocab_size * self.hidden_size) as u64;
        emb + self.num_hidden_layers as u64 * self.block_params() + self.hidden_size as u64 + if self.tie_word_embeddings { 0 } else { emb }
    }
}

/// A projection a LoRA adapter can sit on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Target {
    Q,
    K,
    V,
    O,
    Gate,
    Up,
    Down,
}

impl Target {
    pub const ALL: [Target; 7] = [Target::Q, Target::K, Target::V, Target::O, Target::Gate, Target::Up, Target::Down];

    pub fn as_str(self) -> &'static str {
        match self {
            Target::Q => "q_proj",
            Target::K => "k_proj",
            Target::V => "v_proj",
            Target::O => "o_proj",
            Target::Gate => "gate_proj",
            Target::Up => "up_proj",
            Target::Down => "down_proj",
        }
    }

    pub fn from_s(s: &str) -> Result<Target> {
        Target::ALL.into_iter().find(|t| t.as_str() == s.trim())
            .ok_or_else(|| anyhow!("{s:?} is not a LoRA target; choose from q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj"))
    }

    /// `q_proj,v_proj` → targets, in the fixed module order, without repeats.
    pub fn parse_list(s: &str) -> Result<Vec<Target>> {
        let mut out: Vec<Target> = s.split(',').filter(|p| !p.trim().is_empty()).map(Target::from_s).collect::<Result<_>>()?;
        out.sort();
        out.dedup();
        if out.is_empty() {
            bail!("no LoRA target modules named");
        }
        Ok(out)
    }

    /// The module path inside a block, as Hugging Face names it.
    pub fn module(self) -> &'static str {
        match self {
            Target::Q | Target::K | Target::V | Target::O => "self_attn",
            _ => "mlp",
        }
    }
}

/// LoRA settings (what `adapter_config.json` stores, plus which blocks train).
#[derive(Debug, Clone, PartialEq)]
pub struct LoraSpec {
    pub r: usize,
    pub alpha: f64,
    pub dropout: f64,
    pub use_rslora: bool,
    pub targets: Vec<Target>,
    /// LoRA only in the top N blocks; `None` trains every block
    pub train_layers: Option<usize>,
    /// seeds A's initial values
    pub seed: u64,
}

impl Default for LoraSpec {
    fn default() -> Self {
        Self { r: 16, alpha: 32.0, dropout: 0.0, use_rslora: false, targets: vec![Target::Q, Target::V], train_layers: None, seed: 42 }
    }
}

impl LoraSpec {
    /// `alpha / r`, or `alpha / sqrt(r)` with rsLoRA.
    pub fn scaling(&self) -> f64 {
        self.scaling_at(self.r)
    }

    /// The scaling of an adapter of rank `r` under these settings (an upcycled model's experts may have their own
    /// rank, as PEFT's `rank_pattern` gives a module: `alpha / r`, or `alpha / sqrt(r)` with rsLoRA).
    pub fn scaling_at(&self, r: usize) -> f64 {
        let r = r.max(1) as f64;
        if self.use_rslora { self.alpha / r.sqrt() } else { self.alpha / r }
    }

    /// The index of the first block that carries adapters.
    pub fn first_trained(&self, num_layers: usize) -> usize {
        match self.train_layers {
            Some(n) => num_layers.saturating_sub(n.max(1)),
            None => 0,
        }
    }

    /// Adapter parameters for `cfg`: per trained block and target, `r · (in + out)`.
    /// On an upcycled model the feed-forward targets are each expert's own: every trained block has `experts`
    /// LoRAs on gate, up and down and a router `[experts, hidden]`, and the attention targets are shared.
    pub fn trainable_params(&self, cfg: &ModelConfig) -> u64 {
        let blocks = (cfg.num_hidden_layers - self.first_trained(cfg.num_hidden_layers)) as u64;
        let shared = |t: &&Target| cfg.moe.is_none() || t.module() == "self_attn";
        let per: u64 = self.targets.iter().filter(shared).map(|t| { let (o, i) = cfg.target_shape(*t); (self.r * (o + i)) as u64 }).sum();
        let moe = cfg.moe.map_or(0, |m| m.block_trainable(cfg, self.r));
        blocks * (per + moe)
    }
}

/// How the frozen base is held in memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BaseDtype {
    F32,
    F16,
    BF16,
}

impl BaseDtype {
    pub fn from_s(s: &str) -> Result<Self> {
        match s.to_ascii_lowercase().as_str() {
            "f32" | "float32" => Ok(BaseDtype::F32),
            "f16" | "fp16" | "float16" => Ok(BaseDtype::F16),
            "bf16" | "bfloat16" => Ok(BaseDtype::BF16),
            "4bit" | "8bit" | "q4" | "int4" | "int8" => bail!("a {s} base (QLoRA) is not done: the base trains in f32, f16 or bf16"),
            _ => bail!("base dtype {s:?}: choose f32, f16 or bf16"),
        }
    }
    pub fn as_str(self) -> &'static str {
        match self {
            BaseDtype::F32 => "f32",
            BaseDtype::F16 => "f16",
            BaseDtype::BF16 => "bf16",
        }
    }
    pub fn bytes(self) -> u64 {
        if self == BaseDtype::F32 { 4 } else { 2 }
    }
    fn dtype(self) -> DType {
        match self {
            BaseDtype::F32 => DType::F32,
            BaseDtype::F16 => DType::F16,
            BaseDtype::BF16 => DType::BF16,
        }
    }
}

/// A frozen product `x Wᵀ` as one graph node that sends the gradient to `x` only. The value is computed outside the
/// graph and handed over here; the backward pass computes `grad · W` (widening a 16-bit `W` for that product only).
/// Without it, candle's `matmul` node would compute a full gradient for every frozen weight on the path (its backward
/// does both operands), and with a 16-bit base would also hold the widened f32 copy of every trained block's weights.
struct FrozenLinear {
    /// the forward value, taken (not copied twice) by the forward call
    y: Mutex<Option<Tensor>>,
    /// `[out, in]`, 16-bit
    w: Tensor,
}

impl CustomOp1 for FrozenLinear {
    fn name(&self) -> &'static str {
        "frozen-half-linear"
    }

    fn cpu_fwd(&self, _storage: &CpuStorage, _layout: &Layout) -> candle_core::Result<(CpuStorage, Shape)> {
        take_value(&self.y, "frozen-half-linear")
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(&self, _storage: &candle_core::CudaStorage, _layout: &Layout) -> candle_core::Result<(candle_core::CudaStorage, Shape)> {
        take_cuda(&self.y, "frozen-half-linear")
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(&self, _storage: &candle_core::MetalStorage, _layout: &Layout) -> candle_core::Result<(candle_core::MetalStorage, Shape)> {
        take_metal(&self.y, "frozen-half-linear")
    }

    fn bwd(&self, _arg: &Tensor, _res: &Tensor, grad_res: &Tensor) -> candle_core::Result<Option<Tensor>> {
        // grad · W, a slice of W's rows at a time when W is large (see `WIDEN_ROWS_MAX`)
        let rows = self.w.dim(0)?;
        let step = widen_rows(&self.w);
        let mut acc: Option<Tensor> = None;
        for r0 in (0..rows).step_by(step) {
            let n = step.min(rows - r0);
            let w = widen(&self.w.narrow(0, r0, n)?)?;
            let part = grad_res.narrow(D::Minus1, r0, n)?.contiguous()?.matmul(&w)?;
            acc = Some(match acc { Some(a) => (a + part)?, None => part });
        }
        Ok(acc)
    }
}

/// The most values of a 16-bit matrix widened to f32 at once (256 MB): a larger one (the output head: 152k x 4096
/// for Qwen3-8B, 2.5 GB as f32) is multiplied a slice of rows at a time, each row's dot product as before.
static WIDEN_VALUES_MAX: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(64 << 20);

/// A weight (or a slice of one) as f32 in a layout a product takes: a 16-bit one widened (a fresh, contiguous copy),
/// an f32 one as it is, or made contiguous when it is a column slice (a fine-grained expert's down projection).
fn widen(w: &Tensor) -> candle_core::Result<Tensor> {
    if w.dtype() == DType::F32 { w.contiguous() } else { w.to_dtype(DType::F32) }
}

fn widen_rows(w: &Tensor) -> usize {
    let max = WIDEN_VALUES_MAX.load(std::sync::atomic::Ordering::Relaxed);
    let (rows, cols) = (w.dim(0).unwrap_or(1), w.elem_count() / w.dim(0).unwrap_or(1).max(1));
    if w.dtype() == DType::F32 || rows * cols <= max { rows.max(1) } else { (max / cols.max(1)).max(1) }
}

/// The most logits the head's loss holds at once (32 M values, 128 MB as f32): Qwen3's 151,936-entry vocabulary gives
/// about 220 rows per slice.
static HEAD_SLICE_VALUES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(32 << 20);

/// `x Wᵀ` for a detached `x` (no graph), a slice of W's rows widened at a time, as [`Linear::frozen`] does.
fn frozen_product(x: &Tensor, w: &Tensor) -> candle_core::Result<Tensor> {
    let rows = w.dim(0)?;
    let step = widen_rows(w);
    let mut parts = Vec::new();
    for r0 in (0..rows).step_by(step) {
        let s = w.narrow(0, r0, step.min(rows - r0))?;
        let s = if s.dtype() == DType::F32 { s } else { s.to_dtype(DType::F32)? };
        parts.push(x.matmul(&s.t()?)?);
    }
    if parts.len() == 1 { Ok(parts.pop().unwrap()) } else { Tensor::cat(&parts, D::Minus1) }
}

/// The mean cross-entropy of the frozen head `h Wᵀ` against `targets`, as ONE graph node. Built from the head's
/// product and `cross_entropy` it keeps the whole `[rows, vocab]` logits and their log-softmax for the backward pass:
/// 3.7 GB at 1,024 rows of a 151,936-entry vocabulary, all on the device that holds the head (the first real Mula
/// run ran out of memory there). This keeps the loss only: the forward pass goes a slice of rows at a time, and the
/// backward pass recomputes each slice's softmax for `dh = (softmax − onehot) · W / rows`.
struct HeadLoss {
    y: Mutex<Option<Tensor>>,
    /// `[vocab, hidden]`, frozen
    w: Tensor,
    /// `[rows]` u32
    targets: Tensor,
}

fn head_slice_rows(vocab: usize) -> usize {
    (HEAD_SLICE_VALUES.load(std::sync::atomic::Ordering::Relaxed) / vocab.max(1)).max(1)
}

/// One slice's logits and their log-sum-exp `[c, 1]`.
fn head_slice(h: &Tensor, w: &Tensor) -> candle_core::Result<(Tensor, Tensor)> {
    let logits = frozen_product(h, w)?;
    let m = logits.max_keepdim(D::Minus1)?;
    let lse = (logits.broadcast_sub(&m)?.exp()?.sum_keepdim(D::Minus1)?.log()? + m)?;
    Ok((logits, lse))
}

/// The loss for `h` `[rows, hidden]` (f32) against `targets`, through the head `w`.
pub fn head_loss(h: &Tensor, w: &Tensor, targets: &[u32]) -> Result<Tensor> {
    let n = h.dim(0)?;
    if n != targets.len() || n == 0 {
        bail!("head loss: {} rows and {} targets", n, targets.len());
    }
    let tg = Tensor::from_vec(targets.to_vec(), (n,), h.device())?;
    let step = head_slice_rows(w.dim(0)?);
    let hd = h.detach();
    let mut sum: Option<Tensor> = None;
    for r0 in (0..n).step_by(step) {
        let c = step.min(n - r0);
        let (logits, lse) = head_slice(&hd.narrow(0, r0, c)?, w)?;
        let picked = logits.gather(&tg.narrow(0, r0, c)?.unsqueeze(1)?, 1)?;
        let part = (lse - picked)?.sum_all()?;
        sum = Some(match sum { Some(s) => (s + part)?, None => part });
    }
    let value = (sum.unwrap() / n as f64)?;
    Ok(if h.track_op() { h.apply_op1(HeadLoss { y: Mutex::new(Some(value)), w: w.clone(), targets: tg })? } else { value })
}

impl CustomOp1 for HeadLoss {
    fn name(&self) -> &'static str {
        "head-loss"
    }

    fn cpu_fwd(&self, _storage: &CpuStorage, _layout: &Layout) -> candle_core::Result<(CpuStorage, Shape)> {
        take_value(&self.y, "head-loss")
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(&self, _storage: &candle_core::CudaStorage, _layout: &Layout) -> candle_core::Result<(candle_core::CudaStorage, Shape)> {
        take_cuda(&self.y, "head-loss")
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(&self, _storage: &candle_core::MetalStorage, _layout: &Layout) -> candle_core::Result<(candle_core::MetalStorage, Shape)> {
        take_metal(&self.y, "head-loss")
    }

    fn bwd(&self, h: &Tensor, _res: &Tensor, grad_res: &Tensor) -> candle_core::Result<Option<Tensor>> {
        let (n, vocab) = (h.dim(0)?, self.w.dim(0)?);
        let g = (grad_res.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?[0] as f64) / n as f64;
        let step = head_slice_rows(vocab);
        let hd = h.detach();
        let ids = Tensor::arange(0u32, vocab as u32, h.device())?.unsqueeze(0)?;
        let wrows = widen_rows(&self.w);
        let mut parts = Vec::new();
        for r0 in (0..n).step_by(step) {
            let c = step.min(n - r0);
            let (logits, lse) = head_slice(&hd.narrow(0, r0, c)?, &self.w)?;
            let onehot = self.targets.narrow(0, r0, c)?.unsqueeze(1)?.broadcast_eq(&ids)?.to_dtype(DType::F32)?;
            let d = (logits.broadcast_sub(&lse)?.exp()? - onehot)?;
            drop(logits);
            // d · W, a slice of W's rows (the vocabulary) at a time
            let mut acc: Option<Tensor> = None;
            for v0 in (0..vocab).step_by(wrows) {
                let k = wrows.min(vocab - v0);
                let w = self.w.narrow(0, v0, k)?;
                let w = if w.dtype() == DType::F32 { w } else { w.to_dtype(DType::F32)? };
                let part = d.narrow(1, v0, k)?.contiguous()?.matmul(&w)?;
                acc = Some(match acc { Some(a) => (a + part)?, None => part });
            }
            parts.push((acc.unwrap() * g)?);
        }
        Ok(Some(if parts.len() == 1 { parts.pop().unwrap() } else { Tensor::cat(&parts, 0)? }))
    }
}

/// `softmax(x · scale + mask)` over the last dimension as ONE node of the graph. Built from candle's differentiable
/// operations it would keep five `[heads, seq, seq]` tensors per block for the backward pass (scaled, masked, shifted,
/// exponentiated, normalised) — the largest part of a long sequence's memory; this keeps the result only, and sends
/// the gradient back with the closed form `scale · y · (g − Σ g·y)`. (candle_nn's own `softmax_last_dim` is no
/// substitute: it has no backward pass and cuts the graph without an error, so q and k adapters would never train.)
struct MaskedSoftmax {
    y: Mutex<Option<Tensor>>,
    scale: f64,
}

impl CustomOp1 for MaskedSoftmax {
    fn name(&self) -> &'static str {
        "masked-softmax"
    }

    fn cpu_fwd(&self, _storage: &CpuStorage, _layout: &Layout) -> candle_core::Result<(CpuStorage, Shape)> {
        take_value(&self.y, "masked-softmax")
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(&self, _storage: &candle_core::CudaStorage, _layout: &Layout) -> candle_core::Result<(candle_core::CudaStorage, Shape)> {
        take_cuda(&self.y, "masked-softmax")
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(&self, _storage: &candle_core::MetalStorage, _layout: &Layout) -> candle_core::Result<(candle_core::MetalStorage, Shape)> {
        take_metal(&self.y, "masked-softmax")
    }

    fn bwd(&self, _arg: &Tensor, res: &Tensor, grad_res: &Tensor) -> candle_core::Result<Option<Tensor>> {
        let gy = grad_res.mul(res)?;
        let inner = gy.broadcast_sub(&res.broadcast_mul(&gy.sum_keepdim(D::Minus1)?)?)?;
        Ok(Some((inner * self.scale)?))
    }
}

/// The value a precomputed custom operation hands to the graph: taken once, as a plain CPU tensor.
fn take_value(slot: &Mutex<Option<Tensor>>, name: &str) -> candle_core::Result<(CpuStorage, Shape)> {
    let y = slot.lock().map_err(|_| candle_core::Error::Msg(format!("{name}: lock")))?.take()
        .ok_or_else(|| candle_core::Error::Msg(format!("{name}: forward ran twice")))?;
    // one copy of the values into a buffer of their own (the tensor may be a view of a larger one)
    let shape = y.shape().clone();
    Ok((CpuStorage::F32(y.flatten_all()?.to_vec1::<f32>()?), shape))
}

/// The same on a GPU: the value already sits on the card, and is handed over as a copy of its own (contiguous).
#[cfg(feature = "cuda")]
fn take_cuda(slot: &Mutex<Option<Tensor>>, name: &str) -> candle_core::Result<(candle_core::CudaStorage, Shape)> {
    use candle_core::backend::BackendStorage;
    let y = slot.lock().map_err(|_| candle_core::Error::Msg(format!("{name}: lock")))?.take()
        .ok_or_else(|| candle_core::Error::Msg(format!("{name}: forward ran twice")))?
        .contiguous()?;
    let (storage, layout) = y.storage_and_layout();
    match &*storage {
        candle_core::Storage::Cuda(c) => Ok((c.try_clone(layout)?, y.shape().clone())),
        _ => Err(candle_core::Error::Msg(format!("{name}: the value is not on a GPU"))),
    }
}

/// The same for a value on Apple's GPU (Metal).
#[cfg(feature = "metal")]
fn take_metal(slot: &Mutex<Option<Tensor>>, name: &str) -> candle_core::Result<(candle_core::MetalStorage, Shape)> {
    use candle_core::backend::BackendStorage;
    let y = slot.lock().map_err(|_| candle_core::Error::Msg(format!("{name}: lock")))?.take()
        .ok_or_else(|| candle_core::Error::Msg(format!("{name}: forward ran twice")))?
        .contiguous()?;
    let (storage, layout) = y.storage_and_layout();
    match &*storage {
        candle_core::Storage::Metal(m) => Ok((m.try_clone(layout)?, y.shape().clone())),
        _ => Err(candle_core::Error::Msg(format!("{name}: the value is not on Apple's GPU"))),
    }
}

/// `x` on `dev`, its gradient sent back the same way. Between two GPUs the move goes through host memory: candle's
/// direct copy from one card to another is not ordered against the next kernel, and in the backward pass that gave
/// gradients a few percent wrong (the forward values were right), measured on two RTX 5060 Ti. The host copy waits
/// for the card, so the hop is ordered; the hidden state is small next to the work of a block.
pub fn move_to(x: &Tensor, dev: &Device) -> Result<Tensor> {
    if x.device().same_device(dev) {
        return Ok(x.clone());
    }
    Ok(if x.device().is_cuda() && dev.is_cuda() { x.to_device(&Device::Cpu)?.to_device(dev)? } else { x.to_device(dev)? })
}

/// The plain version, from differentiable operations (what [`masked_softmax`] is checked against).
pub fn masked_softmax_reference(x: &Tensor, scale: f64, mask: &Tensor) -> Result<Tensor> {
    Ok(candle_nn::ops::softmax(&(x * scale)?.broadcast_add(mask)?, D::Minus1)?)
}

/// See [`MaskedSoftmax`]. Without a graph to keep (`x` descends from no variable) it is just the value.
pub fn masked_softmax(x: &Tensor, scale: f64, mask: &Tensor) -> Result<Tensor> {
    let value = masked_softmax_reference(&x.detach(), scale, mask)?;
    if !x.track_op() {
        return Ok(value);
    }
    Ok(x.apply_op1(MaskedSoftmax { y: Mutex::new(Some(value)), scale })?)
}

/// Which keys a query sees: as a table (short sequences), or as a rule, for sequences too long for a seq x seq table.
#[derive(Debug, Clone)]
pub enum Mask {
    Table(Tensor),
    /// causal, and only the last `window` keys (usize::MAX: all of them)
    Causal { window: usize },
}

/// Sequences longer than this use [`chunked_attention`] (set lower in tests).
pub static CHUNKED_ABOVE: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(2048);
/// Queries per chunk of [`chunked_attention`].
pub static ATTENTION_CHUNK: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(512);

/// The key range `[k0, k1)` query rows `[r0, r1)` see under a causal rule with `window`, and its mask `[r1-r0, k1-k0]`.
fn chunk_keys(r0: usize, r1: usize, window: usize, dev: &Device) -> Result<(usize, usize, Tensor)> {
    let k1 = r1;
    let k0 = if window == usize::MAX { 0 } else { (r0 + 1).saturating_sub(window) };
    let m: Vec<f32> = (r0..r1).flat_map(|i| (k0..k1).map(move |j| if j > i || (window != usize::MAX && i - j >= window) { f32::NEG_INFINITY } else { 0.0 })).collect();
    Ok((k0, k1, Tensor::from_vec(m, (r1 - r0, k1 - k0), dev)?))
}

/// softmax(q kᵀ · scale + mask) v for `[heads, seq, hd]`, a chunk of queries at a time against only the keys the
/// rule lets them see. The value is computed outside the graph; the backward pass recomputes each chunk's
/// probabilities and sends exact gradients to q, k and v. Memory: one chunk's `[heads, chunk, keys]`, never seq x seq.
struct ChunkedAttention {
    y: Mutex<Option<Tensor>>,
    scale: f64,
    window: usize,
    chunk: usize,
}

fn chunk_probs(q: &Tensor, k: &Tensor, r0: usize, r1: usize, window: usize, scale: f64) -> Result<(usize, usize, Tensor)> {
    let (k0, k1, mask) = chunk_keys(r0, r1, window, q.device())?;
    let qc = q.narrow(1, r0, r1 - r0)?;
    let kc = k.narrow(1, k0, k1 - k0)?;
    let s = (qc.matmul(&kc.transpose(1, 2)?.contiguous()?)? * scale)?.broadcast_add(&mask)?;
    Ok((k0, k1, candle_nn::ops::softmax_last_dim(&s)?))
}

impl CustomOp3 for ChunkedAttention {
    fn name(&self) -> &'static str {
        "chunked-attention"
    }

    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout) -> candle_core::Result<(CpuStorage, Shape)> {
        take_value(&self.y, "chunked-attention")
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(&self, _: &candle_core::CudaStorage, _: &Layout, _: &candle_core::CudaStorage, _: &Layout, _: &candle_core::CudaStorage, _: &Layout)
        -> candle_core::Result<(candle_core::CudaStorage, Shape)> {
        take_cuda(&self.y, "chunked-attention")
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(&self, _: &candle_core::MetalStorage, _: &Layout, _: &candle_core::MetalStorage, _: &Layout, _: &candle_core::MetalStorage, _: &Layout)
        -> candle_core::Result<(candle_core::MetalStorage, Shape)> {
        take_metal(&self.y, "chunked-attention")
    }

    fn bwd(&self, q: &Tensor, k: &Tensor, v: &Tensor, res: &Tensor, grad: &Tensor) -> candle_core::Result<(Option<Tensor>, Option<Tensor>, Option<Tensor>)> {
        let err = |e: anyhow::Error| candle_core::Error::Msg(format!("chunked attention: {e}"));
        let (q, k, v, o, g) = (q.detach(), k.detach(), v.detach(), res.detach(), grad.detach());
        let seq = q.dim(1)?;
        let dq = q.zeros_like()?;
        let dk = k.zeros_like()?;
        let dv = v.zeros_like()?;
        for r0 in (0..seq).step_by(self.chunk) {
            let r1 = (r0 + self.chunk).min(seq);
            let (k0, k1, p) = chunk_probs(&q, &k, r0, r1, self.window, self.scale).map_err(err)?;
            let (gc, oc) = (g.narrow(1, r0, r1 - r0)?, o.narrow(1, r0, r1 - r0)?);
            let (kc, vc) = (k.narrow(1, k0, k1 - k0)?, v.narrow(1, k0, k1 - k0)?);
            // dV += Pᵀ dO; dP = dO Vᵀ; dS = P (dP - rowsum(dO ∘ O)); dQ = dS K s; dK += dSᵀ Q s
            let dv_part = p.transpose(1, 2)?.contiguous()?.matmul(&gc.contiguous()?)?;
            dv.slice_set(&(dv.narrow(1, k0, k1 - k0)? + dv_part)?.contiguous()?, 1, k0)?;
            let dp = gc.contiguous()?.matmul(&vc.transpose(1, 2)?.contiguous()?)?;
            let d = (&gc * &oc)?.sum_keepdim(D::Minus1)?;
            let ds = (p * dp.broadcast_sub(&d)?)?;
            dq.slice_set(&(ds.matmul(&kc.contiguous()?)? * self.scale)?.contiguous()?, 1, r0)?;
            let dk_part = (ds.transpose(1, 2)?.contiguous()?.matmul(&q.narrow(1, r0, r1 - r0)?.contiguous()?)? * self.scale)?;
            dk.slice_set(&(dk.narrow(1, k0, k1 - k0)? + dk_part)?.contiguous()?, 1, k0)?;
        }
        Ok((Some(dq), Some(dk), Some(dv)))
    }
}

/// See [`ChunkedAttention`]: `[heads, seq, hd]` in, `[heads, seq, hd]` out.
pub fn chunked_attention(q: &Tensor, k: &Tensor, v: &Tensor, scale: f64, window: usize) -> Result<Tensor> {
    let seq = q.dim(1)?;
    let chunk = ATTENTION_CHUNK.load(std::sync::atomic::Ordering::Relaxed).max(1);
    let (qd, kd, vd) = (q.detach(), k.detach(), v.detach());
    let mut parts = Vec::new();
    for r0 in (0..seq).step_by(chunk) {
        let r1 = (r0 + chunk).min(seq);
        let (k0, k1, p) = chunk_probs(&qd, &kd, r0, r1, window, scale)?;
        parts.push(p.matmul(&vd.narrow(1, k0, k1 - k0)?.contiguous()?)?);
    }
    let value = Tensor::cat(&parts, 1)?;
    if !(q.track_op() || k.track_op() || v.track_op()) {
        return Ok(value);
    }
    Ok(q.apply_op3(k, v, ChunkedAttention { y: Mutex::new(Some(value)), scale, window, chunk })?)
}

/// One LoRA adapter: `A` `[r, in]`, `B` `[out, r]`.
#[derive(Debug, Clone)]
pub struct Lora {
    pub a: Var,
    pub b: Var,
    pub scaling: f64,
    pub dropout: f64,
}

/// A frozen projection, with an optional bias (Qwen2) and an optional adapter.
#[derive(Debug, Clone)]
pub struct Linear {
    /// `[out, in]` in the base dtype
    w: Tensor,
    bias: Option<Tensor>,
    pub lora: Option<Lora>,
}

/// Per-forward switches: training turns dropout on, and `seed` makes its masks repeat exactly. A training pass
/// through an upcycled model also collects each MoE block's load-balancing loss here.
#[derive(Debug, Default)]
pub struct Ctx {
    pub train: bool,
    pub seed: u64,
    /// plain inference (regression runs, sampling, validation): the trained variables (adapters, routers, loop gates)
    /// are read detached, so nothing is recorded for a backward pass, and attention goes a chunk of queries at a time
    /// above [`ATTENTION_CHUNK`] tokens. Without it a loop gate (a `Var`) made every block after the loop record its
    /// graph, attention tables included: the first GPU regression run of the looped Qwen3-8B ran out of memory on it.
    pub no_grad: bool,
    calls: Cell<u64>,
    /// each MoE block's router losses of this pass (training only), `[2]`: the load-balancing loss and the z-loss
    /// (0 at full width), summed into the loss by `loss_from_hidden`
    aux: RefCell<Vec<Tensor>>,
    /// the mean load-balancing loss the last `loss_from_hidden` added (before its coefficient), for the log
    aux_value: Cell<Option<f64>>,
    /// the mean router z-loss the last `loss_from_hidden` added (fine-grained upcycling only)
    z_value: Cell<Option<f64>>,
    /// run every feed-forward as the dense one, an upcycled block's too (its weights are the dense weights, at most
    /// permuted): the dense side of `upcycle --dense-start-check`
    pub dense_ffn: bool,
    /// when set, every dense feed-forward pushes `(its input, silu(gate)·up)` here (detached): the calibration of a
    /// fine-grained upcycle ([`crate::moe_init`])
    pub probe: Option<RefCell<Vec<(Tensor, Tensor)>>>,
}

impl Ctx {
    pub fn eval() -> Self {
        Self::default()
    }
    /// Inference with nothing recorded for a backward pass (see [`Ctx::no_grad`]).
    pub fn inference() -> Self {
        Self { no_grad: true, ..Self::default() }
    }
    /// A trained variable as this pass reads it: detached under [`Ctx::no_grad`].
    fn var(&self, v: &Var) -> Tensor {
        if self.no_grad { v.as_tensor().detach() } else { v.as_tensor().clone() }
    }
    pub fn train(seed: u64) -> Self {
        Self { train: true, seed, ..Self::default() }
    }
    /// The mean load-balancing loss the last [`LlamaModel::loss_from_hidden`] with this context added (None: no MoE).
    pub fn aux_loss(&self) -> Option<f64> {
        self.aux_value.get()
    }
    /// The mean router z-loss the last [`LlamaModel::loss_from_hidden`] added (None: not a fine-grained MoE).
    pub fn z_loss(&self) -> Option<f64> {
        self.z_value.get()
    }
    /// Inference with every feed-forward run dense (see [`Ctx::dense_ffn`]).
    pub fn dense_inference() -> Self {
        Self { no_grad: true, dense_ffn: true, ..Self::default() }
    }
    /// Inference that collects each dense feed-forward's input and activation (see [`Ctx::probe`]).
    pub fn probing() -> Self {
        Self { no_grad: true, probe: Some(RefCell::new(Vec::new())), ..Self::default() }
    }
    /// What the probe collected since the last call (emptied).
    pub fn take_probe(&self) -> Vec<(Tensor, Tensor)> {
        self.probe.as_ref().map(|p| std::mem::take(&mut *p.borrow_mut())).unwrap_or_default()
    }
}

fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Uniform in [0, 1) from the top 24 bits.
fn unit(state: &mut u64) -> f32 {
    (splitmix(state) >> 40) as f32 / 16_777_216.0
}

impl Linear {
    fn frozen(&self, x: &Tensor) -> Result<Tensor> {
        // The value is computed outside the graph in every dtype. candle's own `matmul` node, in its backward pass,
        // computes the gradient of BOTH operands whether or not they need one: a full `[out, in]` f32 gradient for
        // every frozen weight on the path, every step (measured: 542 MB peak against 180 MB on a 12-block model).
        // FrozenLinear sends the gradient to the input only.
        let value = {
            let rows = self.w.dim(0)?;
            let step = widen_rows(&self.w);
            let xd = x.detach();
            let mut parts = Vec::new();
            for r0 in (0..rows).step_by(step) {
                let w = self.w.narrow(0, r0, step.min(rows - r0))?;
                // each slice's f32 copy is freed before the next is made (a lower peak than the whole matrix)
                let wide = widen(&w)?;
                parts.push(xd.matmul(&wide.t()?)?);
            }
            if parts.len() == 1 { parts.pop().unwrap() } else { Tensor::cat(&parts, D::Minus1)? }
        };
        let y = if x.track_op() { x.apply_op1(FrozenLinear { y: Mutex::new(Some(value)), w: self.w.clone() })? } else { value };
        Ok(match &self.bias {
            Some(b) => y.broadcast_add(b)?,
            None => y,
        })
    }

    /// `x` is `[rows, in]`, f32.
    pub fn forward(&self, x: &Tensor, ctx: &Ctx) -> Result<Tensor> {
        let base = self.frozen(x)?;
        let Some(l) = &self.lora else { return Ok(base) };
        Ok(base.add(&lora_delta(l, x, ctx)?)?)
    }

    /// `W x + s · B A drop(x)` with an adapter that is not this projection's own (an expert's, over the shared W).
    pub fn forward_with(&self, x: &Tensor, l: &Lora, ctx: &Ctx) -> Result<Tensor> {
        Ok(self.frozen(x)?.add(&lora_delta(l, x, ctx)?)?)
    }
}

/// An adapter's term `s · B A drop(x)`.
fn lora_delta(l: &Lora, x: &Tensor, ctx: &Ctx) -> Result<Tensor> {
    let x = if ctx.train && l.dropout > 0.0 {
        // our own mask (candle's CPU random numbers cannot be seeded): the same seed gives the same run
        let n = ctx.calls.get();
        ctx.calls.set(n + 1);
        let mut state = ctx.seed ^ n.wrapping_mul(0xD6E8_FEB8_6659_FD93);
        let keep = 1.0 - l.dropout as f32;
        let mask: Vec<f32> = (0..x.elem_count()).map(|_| if unit(&mut state) < keep { 1.0 / keep } else { 0.0 }).collect();
        x.mul(&Tensor::from_vec(mask, x.shape(), x.device())?)?
    } else {
        x.clone()
    };
    Ok((x.matmul(&ctx.var(&l.a).t()?)?.matmul(&ctx.var(&l.b).t()?)? * l.scaling)?)
}

/// An upcycled block's feed-forward: the router and each expert's adapters on gate, up and down; the experts share
/// the block's frozen gate, up and down weights (fine-grained: each its group's slice of them).
#[derive(Debug, Clone)]
struct MoeBlock {
    /// `[experts, hidden]`, f32
    router: Var,
    /// per expert: gate, up, down (empty: the experts carry no adapter, only their slices). With a shared A, the
    /// copies of a group hold clones of one A variable.
    experts: Vec<[Lora; 3]>,
    top_k: usize,
    /// neurons per expert (the intermediate size at full width)
    ff_e: usize,
    /// neuron groups (1 at full width); expert e is group `e mod groups`
    groups: usize,
    /// each chosen expert's output times this ([`MoeSpec::out_scale`])
    out_scale: f64,
    /// the A of copy 0 of each group stands for all its copies ([`MoeSpec::shared_a`])
    shared_a: bool,
    /// the router and experts train (false: a fine-grained block below the trained ones, routed as initialised)
    trained: bool,
}

impl MoeBlock {
    /// A fine-grained block routed by `router` with no adapters, not trained.
    fn frozen(cfg: &ModelConfig, m: MoeSpec, router: &Tensor) -> Result<MoeBlock> {
        Ok(MoeBlock { router: Var::from_tensor(&router.copy()?)?, experts: Vec::new(), top_k: m.top_k, ff_e: m.ff_e(cfg), groups: m.groups(cfg),
                      out_scale: m.out_scale(cfg), shared_a: m.shared_a, trained: false })
    }

    /// The trained variables, each once (a shared A only under copy 0, which is expert `g` of group g).
    fn vars(&self) -> Vec<Var> {
        if !self.trained {
            return Vec::new();
        }
        let mut v = vec![self.router.clone()];
        for (e, ls) in self.experts.iter().enumerate() {
            for l in ls {
                if !self.shared_a || e < self.groups {
                    v.push(l.a.clone());
                }
                v.push(l.b.clone());
            }
        }
        v
    }
}

/// Each row's best `k` experts (softmax order, ties to the lower index, as the engine's `route` sorts).
fn top_k_rows(probs: &[Vec<f32>], k: usize) -> Vec<Vec<usize>> {
    probs.iter().map(|p| {
        let mut order: Vec<usize> = (0..p.len()).collect();
        order.sort_by(|&a, &b| p[b].total_cmp(&p[a]));
        order.truncate(k);
        order
    }).collect()
}

/// The router losses of a block for `ctx` (training only): Switch's `E · Σ_e (share of the assignments to e) · (mean
/// probability of e)`, and the z-loss `mean(logsumexp(logits)²)` when `logits` are given, as a `[2]` tensor.
fn push_router_losses(ctx: &Ctx, probs: &Tensor, chosen: &[Vec<usize>], logits: Option<&Tensor>) -> Result<()> {
    if !ctx.train {
        return Ok(());
    }
    let (seq, n_exp) = probs.dims2()?;
    let k = chosen.first().map_or(1, |c| c.len()).max(1);
    let dev = probs.device();
    let mut f = vec![0f32; n_exp];
    chosen.iter().flatten().for_each(|e| f[*e] += 1.0 / (seq * k) as f32);
    let aux = ((probs.mean(0)? * Tensor::from_vec(f, (n_exp,), dev)?)?.sum_all()? * n_exp as f64)?;
    let z = match logits {
        Some(l) => {
            let m = l.max_keepdim(D::Minus1)?.detach();
            let lse = (l.broadcast_sub(&m)?.exp()?.sum_keepdim(D::Minus1)?.log()? + m)?;
            lse.sqr()?.mean_all()?
        }
        None => Tensor::zeros((), DType::F32, dev)?,
    };
    ctx.aux.borrow_mut().push(Tensor::stack(&[&aux, &z], 0)?);
    Ok(())
}

/// The chosen experts' outputs `all` (rows in the order `pos` points into) weighed by the renormalised router
/// probabilities: `[seq, hidden]`. With one expert per token the renormalised weight is exactly 1 and would teach the
/// router nothing: its probability is added and taken away again (`w + p - stop(p)`, the same value) so the router
/// learns as Switch's does.
fn combine(all: Tensor, pos: Vec<u32>, probs: &Tensor, chosen: &[Vec<usize>], k: usize) -> Result<Tensor> {
    let (seq, n_exp) = probs.dims2()?;
    let hid = all.dim(1)?;
    let dev = probs.device().clone();
    let picked = all.index_select(&Tensor::from_vec(pos, (seq * k,), &dev)?, 0)?.reshape((seq, k, hid))?;
    let flat: Vec<u32> = chosen.iter().enumerate().flat_map(|(r, c)| c.iter().map(move |e| (r * n_exp + e) as u32)).collect();
    let sel = probs.flatten_all()?.index_select(&Tensor::from_vec(flat, (seq * k,), &dev)?, 0)?.reshape((seq, k))?;
    let mut w = sel.broadcast_div(&sel.sum_keepdim(1)?)?;
    if k == 1 {
        w = (w + (&sel - sel.detach())?)?;
    }
    Ok(picked.broadcast_mul(&w.unsqueeze(2)?)?.sum(1)?)
}

/// The mixture on `h` `[seq, hidden]` (the block's post-attention norm): the router's softmax picks each token's
/// `top_k` experts, their probabilities renormalised to sum to 1 weigh the experts' outputs. Gate and up are applied
/// once to every token; each chosen expert adds its own LoRA terms on its tokens, and the down projection runs once
/// per (token, expert). In training the block's load-balancing loss goes to `ctx`. (Full width with adapters: the
/// original 57B-A8B path, bit for bit; everything else goes through [`moe_ffn_grouped`].)
fn moe_ffn(block: &Block, m: &MoeBlock, h: &Tensor, ctx: &Ctx) -> Result<Tensor> {
    if m.groups > 1 || m.experts.is_empty() {
        return moe_ffn_grouped(block, m, h, ctx);
    }
    let seq = h.dim(0)?;
    let (k, dev) = (m.top_k, h.device().clone());
    let probs = candle_nn::ops::softmax(&h.matmul(&ctx.var(&m.router).t()?)?, D::Minus1)?;
    let chosen = top_k_rows(&probs.detach().to_device(&Device::Cpu)?.to_vec2::<f32>()?, k);
    let (base_g, base_u) = (block.lin[4].forward(h, ctx)?, block.lin[5].forward(h, ctx)?);
    let mut outs = Vec::new();
    let mut pos = vec![0u32; seq * k];
    let mut offset = 0usize;
    for (e, [lg, lu, ld]) in m.experts.iter().enumerate() {
        let rows: Vec<u32> = (0..seq).filter(|r| chosen[*r].contains(&e)).map(|r| r as u32).collect();
        if rows.is_empty() {
            continue;
        }
        for (j, r) in rows.iter().enumerate() {
            let slot = chosen[*r as usize].iter().position(|x| *x == e).unwrap();
            pos[*r as usize * k + slot] = (offset + j) as u32;
        }
        let n = rows.len();
        let idx = Tensor::from_vec(rows, (n,), &dev)?;
        let he = h.index_select(&idx, 0)?;
        let g = (base_g.index_select(&idx, 0)? + lora_delta(lg, &he, ctx)?)?;
        let u = (base_u.index_select(&idx, 0)? + lora_delta(lu, &he, ctx)?)?;
        let act = (candle_nn::ops::silu(&g)? * u)?;
        outs.push(block.lin[6].forward_with(&act, ld, ctx)?);
        offset += n;
    }
    let all = if outs.len() == 1 { outs.pop().unwrap() } else { Tensor::cat(&outs, 0)? };
    let out = combine(all, pos, &probs, &chosen, k)?;
    push_router_losses(ctx, &probs, &chosen, None)?;
    Ok(out)
}

/// The fine-grained mixture (and a full-width one without adapters): group by group, the (token, slot) pairs whose
/// expert is a copy of the group are gathered copy by copy; the group's frozen gate, up and down slices run once on
/// all of them (one product each, whatever the number of copies), and each copy adds its own LoRA terms to its rows.
/// The block's weights hold the groups contiguously (permuted at load), so a slice is a view of them: gate and up
/// rows `g·ff_e ..`, down columns `g·ff_e ..`. Each expert's output is multiplied by `out_scale`.
fn moe_ffn_grouped(block: &Block, m: &MoeBlock, h: &Tensor, ctx: &Ctx) -> Result<Tensor> {
    let seq = h.dim(0)?;
    let (n_exp, k, groups, fe, dev) = (m.router.as_tensor().dim(0)?, m.top_k, m.groups, m.ff_e, h.device().clone());
    let logits = h.matmul(&ctx.var(&m.router).t()?)?;
    let probs = candle_nn::ops::softmax(&logits, D::Minus1)?;
    let chosen = top_k_rows(&probs.detach().to_device(&Device::Cpu)?.to_vec2::<f32>()?, k);
    let copies = n_exp / groups;
    let mut outs = Vec::new();
    let mut pos = vec![0u32; seq * k];
    let mut offset = 0usize;
    for g in 0..groups {
        // this group's rows, copy by copy: (expert, first row, rows)
        let mut rows: Vec<u32> = Vec::new();
        let mut spans: Vec<(usize, usize, usize)> = Vec::new();
        for c in 0..copies {
            let e = c * groups + g;
            let start = rows.len();
            for (r, ch) in chosen.iter().enumerate() {
                if let Some(slot) = ch.iter().position(|x| *x == e) {
                    pos[r * k + slot] = (offset + rows.len()) as u32;
                    rows.push(r as u32);
                }
            }
            if rows.len() > start {
                spans.push((e, start, rows.len() - start));
            }
        }
        if rows.is_empty() {
            continue;
        }
        let n = rows.len();
        let he = h.index_select(&Tensor::from_vec(rows, (n,), &dev)?, 0)?;
        let slice = |i: usize, dim: usize| -> Result<Linear> { Ok(Linear { w: block.lin[i].w.narrow(dim, g * fe, fe)?, bias: None, lora: None }) };
        let (wg, wu, wd) = (slice(4, 0)?, slice(5, 0)?, slice(6, 1)?);
        // each copy's LoRA term on its own rows, stacked in row order
        let deltas = |which: usize, x: &Tensor| -> Result<Option<Tensor>> {
            if m.experts.is_empty() {
                return Ok(None);
            }
            let parts = spans.iter().map(|(e, s, l)| lora_delta(&m.experts[*e][which], &x.narrow(0, *s, *l)?, ctx)).collect::<Result<Vec<_>>>()?;
            Ok(Some(if parts.len() == 1 { parts.into_iter().next().unwrap() } else { Tensor::cat(&parts, 0)? }))
        };
        let plus = |base: Tensor, d: Option<Tensor>| -> Result<Tensor> { Ok(match d { Some(d) => (base + d)?, None => base }) };
        let gt = plus(wg.frozen(&he)?, deltas(0, &he)?)?;
        let ut = plus(wu.frozen(&he)?, deltas(1, &he)?)?;
        let act = (candle_nn::ops::silu(&gt)? * ut)?;
        let mut y = plus(wd.frozen(&act)?, deltas(2, &act)?)?;
        if m.out_scale != 1.0 {
            y = (y * m.out_scale)?;
        }
        outs.push(y);
        offset += n;
    }
    let all = if outs.len() == 1 { outs.pop().unwrap() } else { Tensor::cat(&outs, 0)? };
    let out = combine(all, pos, &probs, &chosen, k)?;
    push_router_losses(ctx, &probs, &chosen, if m.groups > 1 { Some(&logits) } else { None })?;
    Ok(out)
}

/// One decoder block.
#[derive(Debug, Clone)]
struct Block {
    /// q, k, v, o, gate, up, down - in [`Target::ALL`] order
    lin: Vec<Linear>,
    norm1: Tensor,
    norm2: Tensor,
    q_norm: Option<Tensor>,
    k_norm: Option<Tensor>,
    /// upcycled: the feed-forward is a mixture of experts over the shared gate, up and down
    moe: Option<MoeBlock>,
    /// fine-grained upcycling: the router `moe_init.safetensors` starts this block with (`[experts, hidden]`, f32)
    init_router: Option<Tensor>,
}

impl Block {
    fn has_lora(&self) -> bool {
        self.lin.iter().any(|l| l.lora.is_some()) || self.moe.as_ref().is_some_and(|m| m.trained)
    }
    fn lora_vars(&self) -> Vec<Var> {
        let mut v: Vec<Var> = self.lin.iter().filter_map(|l| l.lora.as_ref()).flat_map(|l| [l.a.clone(), l.b.clone()]).collect();
        if let Some(m) = &self.moe {
            v.extend(m.vars());
        }
        v
    }
}

/// A block run without keeping its activations: the forward value is computed and its graph dropped; the backward
/// pass runs the block again from its saved input, with the same dropout masks, and returns the input's gradient,
/// leaving the adapters' gradients in `grads`.
struct CheckpointBlock {
    y: Mutex<Option<Tensor>>,
    cfg: ModelConfig,
    block: Block,
    rope: (Tensor, Tensor),
    mask: Mask,
    train: bool,
    seed: u64,
    calls: u64,
    grads: CkptGrads,
    /// the block's router losses ride in one extra row of the output (`[seq + 1, hidden]`, the values at `[seq, 0..2]`),
    /// so their gradient comes back through this node too
    with_aux: bool,
}

impl CustomOp1 for CheckpointBlock {
    fn name(&self) -> &'static str {
        "checkpointed-block"
    }

    fn cpu_fwd(&self, _storage: &CpuStorage, _layout: &Layout) -> candle_core::Result<(CpuStorage, Shape)> {
        take_value(&self.y, "checkpointed-block")
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(&self, _storage: &candle_core::CudaStorage, _layout: &Layout) -> candle_core::Result<(candle_core::CudaStorage, Shape)> {
        take_cuda(&self.y, "checkpointed-block")
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(&self, _storage: &candle_core::MetalStorage, _layout: &Layout) -> candle_core::Result<(candle_core::MetalStorage, Shape)> {
        take_metal(&self.y, "checkpointed-block")
    }

    fn bwd(&self, arg: &Tensor, _res: &Tensor, grad_res: &Tensor) -> candle_core::Result<Option<Tensor>> {
        let err = |e: anyhow::Error| candle_core::Error::Msg(format!("checkpointed block: {e}"));
        let x = Var::from_tensor(&arg.detach())?;
        let ctx = Ctx { train: self.train, seed: self.seed, calls: Cell::new(self.calls), ..Ctx::default() };
        let y = block_forward(&self.cfg, &self.block, x.as_tensor(), &self.rope, &self.mask, &ctx).map_err(err)?;
        let (seq, g) = (y.dim(0)?, grad_res.detach());
        let mut total = (y * g.narrow(0, 0, seq)?)?.sum_all()?;
        if self.with_aux {
            let aux = ctx.aux.borrow_mut().pop().ok_or_else(|| candle_core::Error::Msg("checkpointed block: no load-balancing loss in the replay".into()))?;
            total = (total + (aux * g.get(seq)?.narrow(0, 0, 2)?)?.sum_all()?)?;
        }
        let grads = total.backward()?;
        let mut out = self.grads.lock().map_err(|_| candle_core::Error::Msg("checkpoint gradients: lock".into()))?;
        for v in self.block.lora_vars() {
            if let Some(g) = grads.get(&v) {
                out.push((v, g.clone()));
            }
        }
        Ok(grads.get(&x).cloned())
    }
}

/// See [`CheckpointBlock`].
fn checkpointed(cfg: &ModelConfig, block: &Block, x: &Tensor, rope: &(Tensor, Tensor), mask: &Mask, ctx: &Ctx, grads: &CkptGrads) -> Result<Tensor> {
    let calls = ctx.calls.get();
    let sub = Ctx { train: ctx.train, seed: ctx.seed, calls: Cell::new(calls), ..Ctx::default() };
    let value = block_forward(cfg, block, &x.detach(), rope, mask, &sub)?.detach();
    ctx.calls.set(sub.calls.get());
    let aux = sub.aux.borrow_mut().pop().map(|a| a.detach());
    if !x.track_op() && !block.has_lora() {
        return Ok(value);
    }
    let seq = value.dim(0)?;
    let packed = match &aux {
        Some(a) => {
            let hid = value.dim(1)?;
            let row = Tensor::cat(&[&a.reshape((1, 2))?, &Tensor::zeros((1, hid - 2), DType::F32, value.device())?], 1)?;
            Tensor::cat(&[&value, &row], 0)?
        }
        None => value,
    };
    // the input may not descend from a variable (the first trained block): give the node something to hang from
    let x_in = if x.track_op() { x.clone() } else { Var::from_tensor(&x.detach())?.as_tensor().clone() };
    let out = x_in.apply_op1(CheckpointBlock { y: Mutex::new(Some(packed)), cfg: cfg.clone(), block: block.clone(), rope: rope.clone(), mask: mask.clone(),
                                               train: ctx.train, seed: ctx.seed, calls, grads: grads.clone(), with_aux: aux.is_some() })?;
    if aux.is_none() {
        return Ok(out);
    }
    ctx.aux.borrow_mut().push(out.get(seq)?.narrow(0, 0, 2)?);
    Ok(out.narrow(0, 0, seq)?)
}

/// The decoder: frozen base, adapters in the trained blocks.
pub struct LlamaModel {
    pub cfg: ModelConfig,
    pub base_dtype: BaseDtype,
    /// `[vocab, hidden]`; absent once the lower part was dropped and the head is not tied to it
    embed: Option<Tensor>,
    blocks: Vec<Option<Block>>,
    norm: Tensor,
    /// absent when tied (the embeddings are the head)
    lm_head: Option<Tensor>,
    first_trained: usize,
    spec: Option<LoraSpec>,
    dev: Device,
    /// the loop's gates, one `[hidden]` vector per pass after the first (empty without a loop)
    loop_gates: Vec<Var>,
    /// every device the model is spread over (one, unless loaded with [`LlamaModel::load_on`]); `dev` is the first
    devs: Vec<Device>,
    /// the index into `devs` of each block's device; the embeddings sit on the first, the final norm and head on the last
    block_dev: Vec<usize>,
    /// gradient checkpointing ([`LlamaModel::set_checkpointing`])
    checkpoint: bool,
    /// the adapter gradients a checkpointed block's backward pass computes, waiting for [`LlamaModel::take_checkpoint_grads`]
    ckpt_grads: CkptGrads,
    /// run at most this many passes of the loop (1: the base's own pass only); `None`: all of them
    passes_cap: Option<usize>,
}

/// Gradients of adapter variables found while recomputing checkpointed blocks: (variable, gradient).
type CkptGrads = std::sync::Arc<Mutex<Vec<(Var, Tensor)>>>;

/// The weight files of a model folder, in order: `model.safetensors`, or the `model-0000i-of-0000n` shards.
/// Which of `devices` each block goes on: the first device holds the embeddings and the last the output head (its
/// loss is sliced, so no full logits). With an untied head the two ends weigh the same and the blocks split evenly; a
/// tied head is the embedding matrix on the first device, so the first device takes that many blocks' worth fewer.
pub fn block_devices(cfg: &ModelConfig, devices: usize) -> Vec<usize> {
    let (n, d) = (cfg.num_hidden_layers.max(1), devices.max(1));
    let block = (cfg.block_params() - cfg.block_small_params()).max(1) as f64;
    let extra = if d > 1 && cfg.tie_word_embeddings { ((cfg.vocab_size * cfg.hidden_size) as f64 / block).round() as usize } else { 0 };
    (0..n).map(|i| ((i + extra) * d / (n + extra)).min(d - 1)).collect()
}

pub fn weight_files(dir: &Path) -> Result<Vec<PathBuf>> {
    let single = dir.join("model.safetensors");
    if single.exists() {
        return Ok(vec![single]);
    }
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir).with_context(|| format!("{}", dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
            name.starts_with("model-") && name.ends_with(".safetensors")
        })
        .collect();
    files.sort();
    if files.is_empty() {
        bail!("{}: no model.safetensors or model-*.safetensors (a GGUF or .bin model has to be converted to safetensors first)", dir.display());
    }
    Ok(files)
}

fn rms_norm(x: &Tensor, w: &Tensor, eps: f64) -> Result<Tensor> {
    let var = x.sqr()?.mean_keepdim(D::Minus1)?;
    Ok(x.broadcast_div(&(var + eps)?.sqrt()?)?.broadcast_mul(w)?)
}

impl LlamaModel {
    /// Load the frozen base, holding its matrices as `base_dtype`.
    pub fn load(dir: &Path, base_dtype: BaseDtype, dev: &Device) -> Result<Self> {
        let layers = ModelConfig::load(dir)?.num_hidden_layers;
        Self::load_parts(dir, base_dtype, dev, 0..layers, true, true)
    }

    /// As [`load`], without the blocks below `skip_below` (and without the embeddings when the head is its own
    /// tensor): for a run whose activation cache holds what those blocks would compute.
    pub fn load_from(dir: &Path, base_dtype: BaseDtype, dev: &Device, skip_below: usize) -> Result<Self> {
        let layers = ModelConfig::load(dir)?.num_hidden_layers;
        Self::load_parts(dir, base_dtype, dev, skip_below..layers, skip_below == 0, true)
    }

    /// Load only some of the model: the blocks in `blocks`, the embeddings and the output head each on request (a
    /// tied head needs the embeddings and loads them). The activation cache is filled this way, one block at a time.
    pub fn load_parts(dir: &Path, base_dtype: BaseDtype, dev: &Device, blocks_wanted: std::ops::Range<usize>, with_embeddings: bool, with_head: bool) -> Result<Self> {
        Self::load_parts_on(dir, base_dtype, std::slice::from_ref(dev), blocks_wanted, with_embeddings, with_head)
    }

    /// Waits for every device's queued work. On CUDA that is also when the stream's memory pool gives freed blocks
    /// back (its release threshold is 0): without it the pool only grows, and a two-GPU run of the looped Qwen3-8B
    /// went from 13.7 to 15.8 GB a card in 20 steps, then ran out of memory at step 10 of the real run.
    /// Each CUDA device's memory-pool high-water marks since the last call, in bytes: (used, reserved). With
    /// `reset` the marks start again from the current use. Empty off CUDA. (`AM2_MEM_TRACE` training traces.)
    pub fn memory_marks(&self, reset: bool) -> Vec<(u64, u64)> {
        #[cfg(feature = "cuda")]
        {
            use candle_core::cuda_backend::cudarc::driver::sys;
            let mut out = Vec::new();
            for d in &self.devs {
                let Device::Cuda(cd) = d else { continue };
                let stream = cd.cuda_stream();
                let ctx = stream.context();
                if ctx.bind_to_thread().is_err() { continue; }
                unsafe {
                    let mut pool: sys::CUmemoryPool = std::ptr::null_mut();
                    if sys::cuDeviceGetDefaultMemPool(&mut pool, ctx.cu_device()) != sys::CUresult::CUDA_SUCCESS { continue; }
                    let get = |a: sys::CUmemPool_attribute| { let mut v: u64 = 0; sys::cuMemPoolGetAttribute(pool, a, &mut v as *mut u64 as *mut _); v };
                    let used = get(sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_USED_MEM_HIGH);
                    let reserved = get(sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH);
                    if reset {
                        let mut z: u64 = 0;
                        sys::cuMemPoolSetAttribute(pool, sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_USED_MEM_HIGH, &mut z as *mut u64 as *mut _);
                        sys::cuMemPoolSetAttribute(pool, sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH, &mut z as *mut u64 as *mut _);
                    }
                    out.push((used, reserved));
                }
            }
            out
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = reset;
            Vec::new()
        }
    }

    pub fn synchronize(&self) -> Result<()> {
        for d in &self.devs {
            d.synchronize()?;
        }
        self.trim_pools();
        Ok(())
    }

    /// Give each CUDA device's memory pool back what it holds unused (cuMemPoolTrimTo 0). Between conversations the
    /// pool kept 12-14 GB reserved for 8-11 GB in use (fragmentation: every conversation has its own shapes), and the
    /// upcycled 56B-A5B ran out of memory on that reserve, not on its use.
    fn trim_pools(&self) {
        #[cfg(feature = "cuda")]
        for d in &self.devs {
            use candle_core::cuda_backend::cudarc::driver::sys;
            let Device::Cuda(cd) = d else { continue };
            let stream = cd.cuda_stream();
            let ctx = stream.context();
            if ctx.bind_to_thread().is_err() { continue; }
            unsafe {
                let mut pool: sys::CUmemoryPool = std::ptr::null_mut();
                if sys::cuDeviceGetDefaultMemPool(&mut pool, ctx.cu_device()) == sys::CUresult::CUDA_SUCCESS {
                    sys::cuMemPoolTrimTo(pool, 0);
                }
            }
        }
    }

    /// The whole model spread over `devs` (e.g. two GPUs): the blocks are split into equal runs in order, the first
    /// run with the embeddings on the first device, the last with the final norm and the head on the last. Hidden
    /// states move between devices, and so do their gradients.
    pub fn load_on(dir: &Path, base_dtype: BaseDtype, devs: &[Device]) -> Result<Self> {
        let layers = ModelConfig::load(dir)?.num_hidden_layers;
        Self::load_parts_on(dir, base_dtype, devs, 0..layers, true, true)
    }

    pub fn load_parts_on(dir: &Path, base_dtype: BaseDtype, devs: &[Device], blocks_wanted: std::ops::Range<usize>, with_embeddings: bool, with_head: bool) -> Result<Self> {
        if devs.is_empty() {
            bail!("no device to load the model on");
        }
        let dev = &devs[0];
        let last = &devs[devs.len() - 1];
        let cfg = ModelConfig::load(dir)?;
        // the first device holds the embeddings and the last the output head (its loss is sliced, so no full logits):
        // with an untied head the two ends weigh the same and the blocks split evenly; a tied head is the embedding
        // matrix on the first device, so the first device takes that many blocks' worth fewer
        let block_dev = block_devices(&cfg, devs.len());
        let files = weight_files(dir)?;
        // mapped, not read: each tensor is copied out (and narrowed to the base dtype) one at a time
        let st = unsafe { candle_core::safetensors::MmapedSafetensors::multi(&files) }.with_context(|| format!("{}", dir.display()))?;
        let names: HashMap<String, ()> = st.tensors().into_iter().map(|(n, _)| (n, ())).collect();
        let prefix = if names.contains_key("model.embed_tokens.weight") { "model." } else if names.contains_key("embed_tokens.weight") { "" }
            else { bail!("{}: no model.embed_tokens.weight (not a Llama-family checkpoint)", dir.display()) };
        let matrix_on = |name: &str, dev: &Device| -> Result<Tensor> {
            let t = st.load(name, dev).with_context(|| format!("{}: tensor {name}", dir.display()))?;
            if !matches!(t.dtype(), DType::F32 | DType::F16 | DType::BF16) {
                bail!("{name} is {:?}: the base must be F32, F16 or BF16 safetensors", t.dtype());
            }
            Ok(t.to_dtype(base_dtype.dtype())?)
        };
        let matrix = |name: &str| matrix_on(name, dev);
        let vector_on = |name: &str, dev: &Device| -> Result<Tensor> { Ok(st.load(name, dev).with_context(|| format!("{}: tensor {name}", dir.display()))?.to_dtype(DType::F32)?) };
        let expect = |t: &Tensor, dims: &[usize], name: &str| -> Result<()> {
            if t.dims() != dims { bail!("{name} has shape {:?}, the config says {dims:?}", t.dims()) } else { Ok(()) }
        };
        // fine-grained upcycling: the neuron permutation and the router's start, per block
        let fine = cfg.moe.filter(|m| m.fine(&cfg));
        let init = match fine {
            Some(_) => {
                let f = dir.join(MOE_INIT_FILE);
                if !f.exists() {
                    bail!("{}: a fine-grained upcycled folder without {MOE_INIT_FILE} (run `am2strata-workshop upcycle` again)", dir.display());
                }
                candle_core::safetensors::load(&f, &Device::Cpu).with_context(|| format!("{}", f.display()))?
            }
            None => HashMap::new(),
        };

        let mut blocks = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            if !blocks_wanted.contains(&i) {
                blocks.push(None);
                continue;
            }
            let p = |k: &str| format!("{prefix}layers.{i}.{k}");
            let bdev = &devs[block_dev[i]];
            let matrix = |name: &str| matrix_on(name, bdev);
            let vector = |name: &str| vector_on(name, bdev);
            let mut lin = Vec::with_capacity(7);
            let perm = match fine {
                Some(_) => {
                    let perm = init.get(&format!("perm.{i}")).ok_or_else(|| anyhow!("{MOE_INIT_FILE}: no perm.{i}"))?;
                    expect(perm, &[cfg.intermediate_size], &format!("perm.{i}"))?;
                    Some(perm.to_dtype(DType::U32)?)
                }
                None => None,
            };
            for t in Target::ALL {
                let name = p(&format!("{}.{}.weight", t.module(), t.as_str()));
                let w = match (&perm, t.module()) {
                    // the feed-forward's neurons in group order: gate and up rows, down columns (the same function)
                    (Some(perm), "mlp") => {
                        let w = st.load(&name, &Device::Cpu).with_context(|| format!("{}: tensor {name}", dir.display()))?;
                        let w = w.index_select(perm, if t == Target::Down { 1 } else { 0 })?;
                        w.to_dtype(base_dtype.dtype())?.to_device(bdev)?
                    }
                    _ => matrix(&name)?,
                };
                let (o, in_) = cfg.target_shape(t);
                expect(&w, &[o, in_], &name)?;
                let bias_name = p(&format!("{}.{}.bias", t.module(), t.as_str()));
                let bias = if names.contains_key(&bias_name) { Some(vector(&bias_name)?) } else { None };
                lin.push(Linear { w, bias, lora: None });
            }
            let opt = |k: &str| -> Result<Option<Tensor>> { let n = p(k); if names.contains_key(&n) { Ok(Some(vector(&n)?)) } else { Ok(None) } };
            let (q_norm, k_norm) = (opt("self_attn.q_norm.weight")?, opt("self_attn.k_norm.weight")?);
            if cfg.family == Family::Qwen3 && (q_norm.is_none() || k_norm.is_none()) {
                bail!("a Qwen3 model without self_attn.q_norm / k_norm in block {i}");
            }
            let mut block = Block { lin, norm1: vector(&p("input_layernorm.weight"))?, norm2: vector(&p("post_attention_layernorm.weight"))?, q_norm, k_norm,
                                    moe: None, init_router: None };
            if let Some(m) = fine {
                if block.lin[4..].iter().any(|l| l.bias.is_some()) {
                    bail!("block {i}: a feed-forward with biases is not split into fine-grained experts here");
                }
                let r = init.get(&format!("router.{i}")).ok_or_else(|| anyhow!("{MOE_INIT_FILE}: no router.{i}"))?;
                expect(r, &[m.experts, cfg.hidden_size], &format!("router.{i}"))?;
                let r = r.to_dtype(DType::F32)?.to_device(bdev)?;
                block.moe = Some(MoeBlock::frozen(&cfg, m, &r)?);
                block.init_router = Some(r);
            }
            blocks.push(Some(block));
        }
        let has_head = names.contains_key("lm_head.weight");
        if !has_head && !cfg.tie_word_embeddings {
            bail!("{}: no lm_head.weight and tie_word_embeddings is false", dir.display());
        }
        let embed = if with_embeddings || (with_head && !has_head) {
            // an untied table on a GPU is only looked up (a few rows a step, frozen): it stays in RAM, which leaves the
            // first card its 1.2 GB (Qwen3-8B) for activations; a tied one is the head and stays on the device
            let e = if has_head && dev.is_cuda() { matrix_on(&format!("{prefix}embed_tokens.weight"), &Device::Cpu)? }
                    else { matrix(&format!("{prefix}embed_tokens.weight"))? };
            expect(&e, &[cfg.vocab_size, cfg.hidden_size], "embed_tokens.weight")?;
            Some(e)
        } else {
            None
        };
        let lm_head = if has_head && with_head { Some(matrix_on("lm_head.weight", last)?) } else { None };
        let norm = vector_on(&format!("{prefix}norm.weight"), last)?;
        let gate_dev = cfg.loop_.map_or(dev, |l| &devs[block_dev[l.end - 1]]);
        let mut loop_gates = Vec::new();
        if let Some(l) = cfg.loop_ {
            let saved = if dir.join(LOOP_FILE).exists() { candle_core::safetensors::load(dir.join(LOOP_FILE), dev)? } else { HashMap::new() };
            for p in 1..l.passes {
                let g = match saved.get(&format!("loop.gate.{p}")) {
                    Some(t) => { expect(t, &[cfg.hidden_size], "loop gate")?; t.to_dtype(DType::F32)? }
                    None => Tensor::zeros(cfg.hidden_size, DType::F32, gate_dev)?,
                };
                let g = g.to_device(gate_dev)?;
                loop_gates.push(Var::from_tensor(&g)?);
            }
        }
        Ok(Self { cfg, base_dtype, embed, blocks, norm, lm_head, first_trained: 0, spec: None, dev: dev.clone(), loop_gates,
                  devs: devs.to_vec(), block_dev, checkpoint: false, ckpt_grads: Default::default(), passes_cap: None })
    }

    pub fn device(&self) -> &Device {
        &self.dev
    }

    /// The first block carrying adapters (0 without adapters or when every block trains).
    pub fn first_trained(&self) -> usize {
        self.first_trained
    }

    pub fn lora_spec(&self) -> Option<&LoraSpec> {
        self.spec.as_ref()
    }

    /// Put fresh adapters on the chosen targets of the trained blocks: A random in ±1/sqrt(in) (PEFT's
    /// `kaiming_uniform_(a=√5)` bound), seeded by `spec.seed`, the block and the target; B zero.
    pub fn add_lora(&mut self, spec: &LoraSpec) -> Result<()> {
        if spec.r == 0 {
            bail!("LoRA rank 0 would be full fine-tuning, which this trainer does not do; use a rank of 1 or more");
        }
        if !(0.0..1.0).contains(&spec.dropout) {
            bail!("LoRA dropout {} must be in [0, 1)", spec.dropout);
        }
        if let Some(n) = spec.train_layers {
            if n == 0 || n > self.cfg.num_hidden_layers {
                bail!("--train-layers {n}: the model has {} blocks; choose 1 to {}", self.cfg.num_hidden_layers, self.cfg.num_hidden_layers);
            }
        }
        let first = spec.first_trained(self.cfg.num_hidden_layers);
        if let Some(l) = self.cfg.loop_ {
            if first > l.start && first < l.end {
                bail!("--train-layers {}: the first trained block {first} falls inside the loop {}..{}; train from block {} or above {}",
                      spec.train_layers.unwrap_or(0), l.start, l.end, l.start, l.end - 1);
            }
        }
        let moe = self.cfg.moe;
        let h = self.cfg.hidden_size;
        let cfg = self.cfg.clone();
        for (i, block) in self.blocks.iter_mut().enumerate() {
            let Some(block) = block else { continue };
            let bdev = &self.devs[self.block_dev[i]];
            // back to the start: a fine-grained block to its initial router without adapters, a full-width one dense
            block.moe = match (moe, &block.init_router) {
                (Some(m), Some(r)) => Some(MoeBlock::frozen(&cfg, m, r)?),
                _ => None,
            };
            if let (Some(m), true) = (moe, i >= first) {
                let (fe, groups, re) = (m.ff_e(&cfg), m.groups(&cfg), m.rank(spec.r));
                let router = match &block.init_router {
                    Some(r) => r.copy()?,
                    None => {
                        // the router: small seeded values, so the first choices are near-uniform but not all ties
                        let mut state = spec.seed ^ ((i as u64) << 32) ^ 0x0E7E_C0DE;
                        let router: Vec<f32> = (0..m.experts * h).map(|_| 0.02 * (2.0 * unit(&mut state) - 1.0)).collect();
                        Tensor::from_vec(router, (m.experts, h), bdev)?
                    }
                };
                // A random in ±1/sqrt(in), seeded by the block, the target and the expert (or the group, shared)
                let fresh_a = |ti: usize, tag: u64, in_: usize| -> Result<Var> {
                    let mut state = spec.seed ^ ((i as u64) << 32) ^ ((ti as u64) << 24) ^ tag;
                    let bound = 1.0 / (in_ as f32).sqrt();
                    let a: Vec<f32> = (0..re * in_).map(|_| bound * (2.0 * unit(&mut state) - 1.0)).collect();
                    Ok(Var::from_tensor(&Tensor::from_vec(a, (re, in_), bdev)?)?)
                };
                let shapes = [(4usize, fe, h), (5, fe, h), (6, h, fe)];
                let shared: Vec<Vec<Var>> = if m.shared_a && re > 0 {
                    (0..groups).map(|g| shapes.iter().map(|(ti, _, in_)| fresh_a(*ti, ((g as u64 + 1) << 12) ^ 0x5A5A_A5A5, *in_)).collect::<Result<Vec<_>>>())
                        .collect::<Result<_>>()?
                } else {
                    Vec::new()
                };
                let experts = if re == 0 { Vec::new() } else {
                    (0..m.experts).map(|e| -> Result<[Lora; 3]> {
                        let one = |slot: usize| -> Result<Lora> {
                            let (ti, out, in_) = shapes[slot];
                            let a = if m.shared_a { shared[e % groups][slot].clone() } else { fresh_a(ti, ((e as u64 + 1) << 12) ^ 0xA5A5_5A5A, in_)? };
                            Ok(Lora { a, b: Var::from_tensor(&Tensor::zeros((out, re), DType::F32, bdev)?)?, scaling: spec.scaling_at(re), dropout: spec.dropout })
                        };
                        Ok([one(0)?, one(1)?, one(2)?])
                    }).collect::<Result<Vec<_>>>()?
                };
                block.moe = Some(MoeBlock { router: Var::from_tensor(&router)?, experts, top_k: m.top_k, ff_e: fe, groups, out_scale: m.out_scale(&cfg),
                                            shared_a: m.shared_a, trained: true });
            }
            for (ti, t) in Target::ALL.into_iter().enumerate() {
                block.lin[ti].lora = None;
                // an upcycled block's feed-forward adapters are its experts'
                if i < first || !spec.targets.contains(&t) || block.moe.is_some() && t.module() == "mlp" {
                    continue;
                }
                let (out, in_) = self.cfg.target_shape(t);
                let mut state = spec.seed ^ ((i as u64) << 32) ^ ((ti as u64) << 24) ^ 0xA5A5_5A5A;
                let bound = 1.0 / (in_ as f32).sqrt();
                let a: Vec<f32> = (0..spec.r * in_).map(|_| bound * (2.0 * unit(&mut state) - 1.0)).collect();
                block.lin[ti].lora = Some(Lora {
                    a: Var::from_tensor(&Tensor::from_vec(a, (spec.r, in_), &self.devs[self.block_dev[i]])?)?,
                    b: Var::from_tensor(&Tensor::zeros((out, spec.r), DType::F32, &self.devs[self.block_dev[i]])?)?,
                    scaling: spec.scaling(),
                    dropout: spec.dropout,
                });
            }
        }
        self.first_trained = first;
        self.spec = Some(spec.clone());
        Ok(())
    }

    /// Every adapter matrix under its PEFT name (`base_model.model.model.layers.N.self_attn.q_proj.lora_A.weight`),
    /// block by block, target by target, A then B.
    pub fn lora_vars(&self) -> Vec<(String, Var)> {
        let mut out = Vec::new();
        for (i, block) in self.blocks.iter().enumerate() {
            let Some(block) = block else { continue };
            for (ti, t) in Target::ALL.into_iter().enumerate() {
                if let Some(l) = &block.lin[ti].lora {
                    let stem = format!("base_model.model.model.layers.{i}.{}.{}", t.module(), t.as_str());
                    out.push((format!("{stem}.lora_A.weight"), l.a.clone()));
                    out.push((format!("{stem}.lora_B.weight"), l.b.clone()));
                }
            }
            if let Some(m) = block.moe.as_ref().filter(|m| m.trained) {
                out.push((moe_router_name(i), m.router.clone()));
                for (e, ls) in m.experts.iter().enumerate() {
                    for (l, t) in ls.iter().zip(MOE_TARGETS) {
                        let stem = moe_expert_stem(i, e, t);
                        // a shared A once, under its group's name (copy 0 of group g is expert g)
                        if !m.shared_a {
                            out.push((format!("{stem}.lora_A.weight"), l.a.clone()));
                        } else if e < m.groups {
                            out.push((format!("{}.lora_A.weight", moe_group_stem(i, e, t)), l.a.clone()));
                        }
                        out.push((format!("{stem}.lora_B.weight"), l.b.clone()));
                    }
                }
            }
        }
        out.extend(self.trained_loop_gates());
        out
    }

    /// The loop's gates under their adapter names (`base_model.model.model.loop.gate.<p>`), when an adapter is
    /// attached and the loop lies in its trained blocks: they train with it.
    pub fn trained_loop_gates(&self) -> Vec<(String, Var)> {
        match (self.cfg.loop_, &self.spec) {
            (Some(l), Some(_)) if l.start >= self.first_trained => self.loop_gates.iter().enumerate()
                .map(|(p, g)| (format!("base_model.model.model.loop.gate.{}", p + 1), g.clone())).collect(),
            _ => Vec::new(),
        }
    }

    /// Gradient checkpointing: each trained block keeps only its input for the backward pass, and its activations
    /// are recomputed there, one block at a time. The adapters' gradients from that recomputation are collected by
    /// [`LlamaModel::take_checkpoint_grads`]; add them to the graph's own. Memory falls from every block's activations
    /// to one block's plus each block's input, for about one more forward pass of work.
    /// Run the loop `passes` times in all (1: the base's own pass, as if there were no loop), for comparing a looped
    /// model at each pass count; `passes` above the model's own is refused. A model without a loop takes only 1.
    pub fn set_passes(&mut self, passes: usize) -> Result<()> {
        let most = self.cfg.loop_.map_or(1, |l| l.passes);
        if passes == 0 || passes > most {
            bail!("{passes} passes: this model runs 1{}", if most > 1 { format!(" to {most}") } else { " (it has no loop)".into() });
        }
        self.passes_cap = Some(passes);
        Ok(())
    }

    /// The passes the loop runs now (1 without a loop).
    pub fn passes(&self) -> usize {
        let most = self.cfg.loop_.map_or(1, |l| l.passes);
        self.passes_cap.unwrap_or(most).min(most)
    }

    pub fn set_checkpointing(&mut self, on: bool) {
        self.checkpoint = on;
    }

    /// The adapter gradients the last backward pass computed inside checkpointed blocks (emptied by the call).
    pub fn take_checkpoint_grads(&self) -> Vec<(Var, Tensor)> {
        std::mem::take(&mut *self.ckpt_grads.lock().unwrap())
    }

    /// The loop's gates as they are now, under their `loop.safetensors` names.
    pub fn loop_gate_tensors(&self) -> BTreeMap<String, Tensor> {
        self.loop_gates.iter().enumerate().map(|(p, g)| (format!("loop.gate.{}", p + 1), g.as_tensor().detach())).collect()
    }

    /// Attach a PEFT adapter folder (`adapter_config.json` + `adapter_model.safetensors`) as it is: the blocks and
    /// targets its tensors name, its rank, alpha and rsLoRA flag. Returns the settings it carried.
    pub fn load_adapter(&mut self, dir: &Path) -> Result<LoraSpec> {
        let cfg: Value = serde_json::from_str(&std::fs::read_to_string(dir.join("adapter_config.json"))
            .with_context(|| format!("{}: no adapter_config.json", dir.display()))?)?;
        if cfg["use_dora"].as_bool() == Some(true) {
            bail!("a DoRA adapter: only plain LoRA and rsLoRA adapters load here");
        }
        let r = cfg["r"].as_u64().ok_or_else(|| anyhow!("adapter_config.json has no r"))? as usize;
        let alpha = cfg["lora_alpha"].as_f64().ok_or_else(|| anyhow!("adapter_config.json has no lora_alpha"))?;
        let mut spec = LoraSpec { r, alpha, dropout: cfg["lora_dropout"].as_f64().unwrap_or(0.0), use_rslora: cfg["use_rslora"].as_bool().unwrap_or(false),
                                  targets: Vec::new(), train_layers: None, seed: 0 };
        let tensors = candle_core::safetensors::load(dir.join("adapter_model.safetensors"), &self.dev)?;
        let mut used = 0;
        let mut first = self.cfg.num_hidden_layers;
        let saved_moe = MoeSpec::from_json(&cfg["prastara_moe"])?;
        if saved_moe.is_some() && saved_moe != self.cfg.moe {
            bail!("the adapter was trained on an upcycled model ({:?}); this model is {:?}", saved_moe.unwrap(), self.cfg.moe);
        }
        let h = self.cfg.hidden_size;
        let cfg = self.cfg.clone();
        for (i, block) in self.blocks.iter_mut().enumerate() {
            let Some(block) = block else { continue };
            block.moe = match (self.cfg.moe, &block.init_router) {
                (Some(m), Some(r)) => Some(MoeBlock::frozen(&cfg, m, r)?),
                _ => None,
            };
            let bdev = &self.devs[self.block_dev[i]];
            if let (Some(m), Some(router)) = (self.cfg.moe, tensors.get(&moe_router_name(i))) {
                if router.dims() != [m.experts, h] {
                    bail!("{}: the router has shape {:?}, the model has {} experts of hidden size {h}", moe_router_name(i), router.dims(), m.experts);
                }
                let (fe, groups, re) = (m.ff_e(&cfg), m.groups(&cfg), m.rank(r));
                let var = |t: &Tensor| -> Result<Var> { Ok(Var::from_tensor(&t.to_dtype(DType::F32)?.to_device(bdev)?)?) };
                // a shared A: one variable per group and target, cloned into each copy
                let mut shared: HashMap<(usize, usize), Var> = HashMap::new();
                let mut experts = Vec::with_capacity(if re == 0 { 0 } else { m.experts });
                for e in (0..m.experts).filter(|_| re > 0) {
                    let mut three = Vec::with_capacity(3);
                    for (slot, (t, (out, in_))) in MOE_TARGETS.into_iter().zip([(fe, h), (fe, h), (h, fe)]).enumerate() {
                        let stem = moe_expert_stem(i, e, t);
                        let a_name = if m.shared_a { format!("{}.lora_A.weight", moe_group_stem(i, e % groups, t)) } else { format!("{stem}.lora_A.weight") };
                        let (Some(a), Some(b)) = (tensors.get(&a_name), tensors.get(&format!("{stem}.lora_B.weight"))) else {
                            bail!("{stem}: the adapter has a router for block {i} but not this expert's lora_A and lora_B");
                        };
                        if a.dims() != [re, in_] || b.dims() != [out, re] {
                            bail!("{stem}: adapter shapes {:?} and {:?} do not fit a [{out}, {in_}] weight at rank {re}", a.dims(), b.dims());
                        }
                        let a = if m.shared_a {
                            match shared.get(&(e % groups, slot)) {
                                Some(v) => v.clone(),
                                None => { let v = var(a)?; shared.insert((e % groups, slot), v.clone()); used += 1; v }
                            }
                        } else {
                            used += 1;
                            var(a)?
                        };
                        three.push(Lora { a, b: var(b)?, scaling: spec.scaling_at(re), dropout: spec.dropout });
                        used += 1;
                    }
                    let [g, u, d]: [Lora; 3] = three.try_into().map_err(|_| anyhow!("three adapters"))?;
                    experts.push([g, u, d]);
                }
                block.moe = Some(MoeBlock { router: var(router)?, experts, top_k: m.top_k, ff_e: fe, groups, out_scale: m.out_scale(&cfg),
                                            shared_a: m.shared_a, trained: true });
                used += 1;
                first = first.min(i);
            }
            for (ti, t) in Target::ALL.into_iter().enumerate() {
                let stem = format!("base_model.model.model.layers.{i}.{}.{}", t.module(), t.as_str());
                let (a, b) = (tensors.get(&format!("{stem}.lora_A.weight")), tensors.get(&format!("{stem}.lora_B.weight")));
                block.lin[ti].lora = match (a, b) {
                    (Some(a), Some(b)) => {
                        let (out, in_) = self.cfg.target_shape(t);
                        if a.dims() != [r, in_] || b.dims() != [out, r] {
                            bail!("{stem}: adapter shapes {:?} and {:?} do not fit a [{out}, {in_}] weight at rank {r}", a.dims(), b.dims());
                        }
                        used += 2;
                        first = first.min(i);
                        if !spec.targets.contains(&t) {
                            spec.targets.push(t);
                        }
                        let bdev = &self.devs[self.block_dev[i]];
                        Some(Lora { a: Var::from_tensor(&a.to_dtype(DType::F32)?.to_device(bdev)?)?, b: Var::from_tensor(&b.to_dtype(DType::F32)?.to_device(bdev)?)?,
                                    scaling: spec.scaling(), dropout: spec.dropout })
                    }
                    (None, None) => None,
                    _ => bail!("{stem}: the adapter has only one of lora_A and lora_B"),
                };
            }
        }
        for p in 1..self.cfg.loop_.map_or(0, |l| l.passes) {
            if let Some(g) = tensors.get(&format!("base_model.model.model.loop.gate.{p}")) {
                if g.dims() != [self.cfg.hidden_size] {
                    bail!("loop gate {p} has shape {:?}, the model's hidden size is {}", g.dims(), self.cfg.hidden_size);
                }
                let gdev = self.loop_gates[p - 1].as_tensor().device().clone();
                self.loop_gates[p - 1] = Var::from_tensor(&g.to_dtype(DType::F32)?.to_device(&gdev)?)?;
                used += 1;
            }
        }
        if used != tensors.len() {
            let unknown: Vec<&String> = tensors.keys().filter(|k| !k.contains(".lora_A.weight") && !k.contains(".lora_B.weight") || !k.contains(".layers.") || k.contains(".experts.")).take(3).collect();
            bail!("the adapter holds {} tensors, {used} of them fit this model (for example not: {unknown:?}); it was made for another model or is not a plain LoRA", tensors.len());
        }
        spec.targets.sort();
        self.first_trained = if used == 0 { 0 } else { first };
        if self.first_trained > 0 {
            spec.train_layers = Some(self.cfg.num_hidden_layers - self.first_trained);
        }
        self.spec = Some(spec.clone());
        Ok(spec)
    }

    /// Free the embeddings (when the head is its own tensor) and the blocks below the first trained one. After this
    /// only [`upper`] works: the hidden states must come from the activation cache.
    pub fn drop_lower(&mut self) {
        for b in self.blocks.iter_mut().take(self.first_trained) {
            *b = None;
        }
        if self.lm_head.is_some() && self.first_trained > 0 {
            self.embed = None;
        }
    }

    /// Bytes the frozen base holds in memory right now (matrices in the base dtype, norms and biases in f32).
    pub fn base_bytes(&self) -> u64 {
        let size = |t: &Tensor| (t.elem_count() * t.dtype().size_in_bytes()) as u64;
        let mut n = self.embed.as_ref().map_or(0, size) + self.lm_head.as_ref().map_or(0, size) + size(&self.norm);
        for b in self.blocks.iter().flatten() {
            n += b.lin.iter().map(|l| size(&l.w) + l.bias.as_ref().map_or(0, size)).sum::<u64>();
            n += size(&b.norm1) + size(&b.norm2) + b.q_norm.as_ref().map_or(0, size) + b.k_norm.as_ref().map_or(0, size);
        }
        n
    }

    /// Whether the embeddings and every block below the first trained one are in memory.
    pub fn has_lower(&self) -> bool {
        self.embed.is_some() && self.blocks.iter().take(self.first_trained).all(Option::is_some)
    }

    fn rope(&self, seq: usize, dev: &Device) -> Result<(Tensor, Tensor)> {
        let hd = self.cfg.head_dim;
        // as transformers: inv_freq in f32, angles = position × inv_freq, each repeated over both halves
        let (inv, scale): (Vec<f32>, f32) = match self.cfg.yarn {
            Some(y) => (y.inv_freq(hd, self.cfg.rope_theta).into_iter().map(|f| f as f32).collect(), y.attention_factor as f32),
            None => ((0..hd / 2).map(|i| 1.0 / (self.cfg.rope_theta as f32).powf(2.0 * i as f32 / hd as f32)).collect(), 1.0),
        };
        let mut cos = Vec::with_capacity(seq * hd);
        let mut sin = Vec::with_capacity(seq * hd);
        for p in 0..seq {
            for half in 0..2 {
                let _ = half;
                for f in &inv {
                    let a = p as f32 * f;
                    cos.push(a.cos() * scale);
                    sin.push(a.sin() * scale);
                }
            }
        }
        Ok((Tensor::from_vec(cos, (seq, hd), dev)?, Tensor::from_vec(sin, (seq, hd), dev)?))
    }

    fn block_forward(&self, block: &Block, x: &Tensor, rope: &(Tensor, Tensor), mask: &Mask, ctx: &Ctx) -> Result<Tensor> {
        if !ctx.no_grad && (self.checkpoint && x.track_op() || self.checkpoint && block.has_lora()) {
            return checkpointed(&self.cfg, block, x, rope, mask, ctx, &self.ckpt_grads);
        }
        block_forward(&self.cfg, block, x, rope, mask, ctx)
    }
}

/// One decoder block, given the model's settings (what [`LlamaModel::block_forward`] and a checkpointed block's
/// recomputation both run).
fn block_forward(c: &ModelConfig, block: &Block, x: &Tensor, rope: &(Tensor, Tensor), mask: &Mask, ctx: &Ctx) -> Result<Tensor> {
    {
        let (seq, hd, nh, nkv) = (x.dim(0)?, c.head_dim, c.num_attention_heads, c.num_key_value_heads);
        let h = rms_norm(x, &block.norm1, c.rms_norm_eps)?;
        // [seq, heads·hd] → [heads, seq, hd], with Qwen3's per-head norm before the rotation
        let heads = |t: Tensor, n: usize, norm: &Option<Tensor>| -> Result<Tensor> {
            let t = t.reshape((seq, n, hd))?;
            let t = match norm {
                Some(w) => rms_norm(&t, w, c.rms_norm_eps)?,
                None => t,
            };
            Ok(t.transpose(0, 1)?.contiguous()?)
        };
        let rotate = |t: &Tensor| -> Result<Tensor> {
            let (x1, x2) = (t.narrow(D::Minus1, 0, hd / 2)?, t.narrow(D::Minus1, hd / 2, hd / 2)?);
            let rot = Tensor::cat(&[&x2.neg()?, &x1], D::Minus1)?;
            Ok((t.broadcast_mul(&rope.0)? + rot.broadcast_mul(&rope.1)?)?)
        };
        let q = rotate(&heads(block.lin[0].forward(&h, ctx)?, nh, &block.q_norm)?)?;
        let k = rotate(&heads(block.lin[1].forward(&h, ctx)?, nkv, &block.k_norm)?)?;
        let v = heads(block.lin[2].forward(&h, ctx)?, nkv, &None)?;
        // grouped-query attention: each key/value head serves heads/kv_heads query heads
        let rep = nh / nkv;
        let spread = |t: Tensor| -> Result<Tensor> {
            if rep == 1 { Ok(t) } else { Ok(t.unsqueeze(1)?.broadcast_as((nkv, rep, seq, hd))?.reshape((nh, seq, hd))?) }
        };
        let (k, v) = (spread(k)?, spread(v)?);
        let scale = 1.0 / (hd as f64).sqrt();
        let attended = match mask {
            Mask::Table(m) => masked_softmax(&q.matmul(&k.transpose(1, 2)?.contiguous()?)?, scale, m)?.matmul(&v)?,
            Mask::Causal { window } => chunked_attention(&q, &k.contiguous()?, &v.contiguous()?, scale, *window)?,
        };
        let out = attended.transpose(0, 1)?.reshape((seq, nh * hd))?;
        let x = (x + block.lin[3].forward(&out, ctx)?)?;

        let h = rms_norm(&x, &block.norm2, c.rms_norm_eps)?;
        if let (Some(m), false) = (&block.moe, ctx.dense_ffn) {
            return Ok((x + moe_ffn(block, m, &h, ctx)?)?);
        }
        let gated = (candle_nn::ops::silu(&block.lin[4].forward(&h, ctx)?)? * block.lin[5].forward(&h, ctx)?)?;
        if let Some(p) = &ctx.probe {
            p.borrow_mut().push((h.detach(), gated.detach()));
        }
        Ok((x + block.lin[6].forward(&gated, ctx)?)?)
    }
}

impl LlamaModel {

    fn causal_mask(&self, seq: usize, dev: &Device) -> Result<Tensor> {
        self.window_mask(seq, usize::MAX, dev)
    }

    /// Causal, and a key more than `window - 1` positions back is hidden (transformers' sliding window: query i sees
    /// keys j with i - window < j <= i).
    fn window_mask(&self, seq: usize, window: usize, dev: &Device) -> Result<Tensor> {
        let m: Vec<f32> = (0..seq).flat_map(|i| (0..seq).map(move |j| if j > i || i - j >= window { f32::NEG_INFINITY } else { 0.0 })).collect();
        Ok(Tensor::from_vec(m, (seq, seq), dev)?)
    }

    fn run_blocks(&self, x: Tensor, range: std::ops::Range<usize>, ctx: &Ctx) -> Result<Tensor> {
        self.run_blocks_seen(x, range, ctx, None)
    }

    /// [`run_blocks`](Self::run_blocks), handing `seen` the residual stream after every block of every pass, in the
    /// order they run: `(block, pass, state)`, pass 0 being the base's own pass and 1.. the loop's extra passes (the
    /// state there is the pass's own, before its gate mixes it in).
    fn run_blocks_seen(&self, mut x: Tensor, range: std::ops::Range<usize>, ctx: &Ctx, mut seen: Option<&mut dyn FnMut(usize, usize, &Tensor) -> Result<()>>) -> Result<Tensor> {
        let seq = x.dim(0)?;
        if seq == 0 {
            bail!("an empty token sequence");
        }
        // the rotary tables and the mask, once per device the blocks sit on
        // (rope, the causal mask, the sliding layers' mask when there are any) per device
        // inference has no backward pass to keep a table for: chunks of queries above one chunk's length, so a
        // 4,096-token prompt holds [heads, 512, keys] at a time instead of [heads, seq, seq] (2 GB on Qwen3-8B)
        let above = if ctx.no_grad { ATTENTION_CHUNK.load(std::sync::atomic::Ordering::Relaxed) } else { CHUNKED_ABOVE.load(std::sync::atomic::Ordering::Relaxed) };
        let long = seq > above;
        let mut per_dev: Vec<Option<((Tensor, Tensor), Mask, Option<Mask>)>> = vec![None; self.devs.len()];
        for i in range.clone() {
            let d = self.block_dev[i];
            if per_dev[d].is_none() {
                // a long sequence never builds a seq x seq table: the chunked attention applies the rule itself
                let (full, sw) = if long {
                    (Mask::Causal { window: usize::MAX }, self.cfg.sliding_window.map(|w| Mask::Causal { window: w }))
                } else {
                    let sw = match self.cfg.sliding_window {
                        Some(w) if w < seq => Some(Mask::Table(self.window_mask(seq, w, &self.devs[d])?)),
                        _ => None,
                    };
                    (Mask::Table(self.causal_mask(seq, &self.devs[d])?), sw)
                };
                per_dev[d] = Some((self.rope(seq, &self.devs[d])?, full, sw));
            }
        }
        // one block on its own device: the hidden state (and, backwards, its gradient) moves when the device changes
        let step = |j: usize, x: &Tensor| -> Result<Tensor> {
            let b = self.blocks[j].as_ref().ok_or_else(|| anyhow!("block {j} is not in memory (the activation cache stands in for it)"))?;
            let (rope, full, sliding) = per_dev[self.block_dev[j]].as_ref().ok_or_else(|| anyhow!("block {j}: no tables for its device"))?;
            let mask = match sliding {
                Some(m) if self.cfg.layer_sliding.get(j) == Some(&true) => m,
                _ => full,
            };
            self.block_forward(b, &move_to(x, &self.devs[self.block_dev[j]])?, rope, mask, ctx)
        };
        if let Some(l) = self.cfg.loop_ {
            if range.start > l.start && range.start < l.end {
                bail!("blocks {}.. start inside the loop {}..{}: the loop's blocks must be run together", range.start, l.start, l.end);
            }
        }
        for i in range.clone() {
            x = step(i, &x)?;
            if let Some(f) = seen.as_mut() {
                f(i, 0, &x)?;
            }
            match self.cfg.loop_ {
                Some(l) if i + 1 == l.end && range.start <= l.start => {
                    let extra = self.passes_cap.map_or(self.loop_gates.len(), |c| c.saturating_sub(1));
                    for (p, g) in self.loop_gates.iter().enumerate().take(extra) {
                        let mut y = x.clone();
                        for j in l.start..l.end {
                            y = step(j, &y)?;
                            if let Some(f) = seen.as_mut() {
                                f(j, p + 1, &y)?;
                            }
                        }
                        let y = move_to(&y, x.device())?;
                        x = (&x + (y - &x)?.broadcast_mul(&ctx.var(g))?)?;
                    }
                }
                _ => {}
            }
        }
        Ok(x)
    }

    /// Block `i` alone on `x` `[seq, hidden]` (its own rotary tables and mask; the loop is the caller's business: run
    /// its blocks again and mix with [`LlamaModel::loop_gate_tensors`]). What a block-by-block sweep runs
    /// ([`crate::moe_init`]): only that block need be in memory.
    pub fn run_one_block(&self, i: usize, x: &Tensor, ctx: &Ctx) -> Result<Tensor> {
        let b = self.blocks.get(i).and_then(Option::as_ref).ok_or_else(|| anyhow!("block {i} is not in memory"))?;
        let dev = &self.devs[self.block_dev[i]];
        let seq = x.dim(0)?;
        let window = match self.cfg.sliding_window {
            Some(w) if self.cfg.layer_sliding.get(i) == Some(&true) => w,
            _ => usize::MAX,
        };
        let above = if ctx.no_grad { ATTENTION_CHUNK.load(std::sync::atomic::Ordering::Relaxed) } else { CHUNKED_ABOVE.load(std::sync::atomic::Ordering::Relaxed) };
        let mask = if seq > above { Mask::Causal { window } } else { Mask::Table(self.window_mask(seq, window, dev)?) };
        self.block_forward(b, &move_to(x, dev)?, &self.rope(seq, dev)?, &mask, ctx)
    }

    /// The mean next-token cross-entropy of the final hidden state `x` (before the final norm) over the supervised
    /// positions, and how many there were: the end of a block-by-block sweep. Needs the final norm and the head.
    pub fn loss_of_state(&self, x: &Tensor, tokens: &[u32], supervised: &[bool]) -> Result<(f64, usize)> {
        let rows: Vec<u32> = (1..tokens.len()).filter(|p| supervised[*p]).map(|p| (p - 1) as u32).collect();
        if rows.is_empty() {
            return Ok((0.0, 0));
        }
        let targets: Vec<u32> = rows.iter().map(|r| tokens[*r as usize + 1]).collect();
        let n = rows.len();
        let x = move_to(x, self.norm.device())?;
        let hidden = rms_norm(&x.index_select(&Tensor::from_vec(rows, (n,), x.device())?, 0)?, &self.norm, self.cfg.rms_norm_eps)?;
        let w = self.lm_head.as_ref().or(self.embed.as_ref()).ok_or_else(|| anyhow!("no output head in memory"))?;
        Ok((head_loss(&move_to(&hidden, w.device())?, w, &targets)?.to_scalar::<f32>()? as f64, n))
    }

    /// A block's frozen weight for `t` as held (fine-grained: the feed-forward's neurons in group order), when the
    /// block is in memory.
    pub fn block_weight(&self, i: usize, t: Target) -> Option<Tensor> {
        let ti = Target::ALL.iter().position(|x| *x == t)?;
        self.blocks.get(i)?.as_ref().map(|b| b.lin[ti].w.clone())
    }

    /// The router a block routes with now (`[experts, hidden]`), when it is an upcycled block in memory.
    pub fn router(&self, i: usize) -> Option<Tensor> {
        self.blocks.get(i)?.as_ref()?.moe.as_ref().map(|m| m.router.as_tensor().detach())
    }

    /// The hidden state `[seq, hidden]` entering the first trained block: embeddings and the frozen blocks below it.
    /// No adapter is involved, so nothing is recorded for a backward pass; the result is detached.
    pub fn lower(&self, tokens: &[u32]) -> Result<Tensor> {
        self.frozen_blocks(&self.embed_tokens(tokens)?, 0..self.first_trained)
    }

    /// The embeddings of `tokens`, `[seq, hidden]` in f32.
    pub fn embed_tokens(&self, tokens: &[u32]) -> Result<Tensor> {
        let embed = self.embed.as_ref().ok_or_else(|| anyhow!("the embeddings are not in memory (the activation cache stands in for them)"))?;
        if let Some(bad) = tokens.iter().find(|t| **t as usize >= self.cfg.vocab_size) {
            bail!("token id {bad} is outside the vocabulary of {}", self.cfg.vocab_size);
        }
        let ids = Tensor::from_vec(tokens.to_vec(), (tokens.len(),), embed.device())?;
        let rows = embed.index_select(&ids, 0)?.to_dtype(DType::F32)?;
        // the table may be in RAM (see load_parts_on): the rows go where the first block runs
        match self.devs.first() {
            Some(d) if !d.same_device(embed.device()) => Ok(rows.to_device(d)?),
            _ => Ok(rows),
        }
    }

    /// Run `h` through the blocks in `range` as plain inference (they must carry no adapter); the result is detached.
    pub fn frozen_blocks(&self, h: &Tensor, range: std::ops::Range<usize>) -> Result<Tensor> {
        Ok(self.run_blocks(h.clone(), range, &Ctx::inference())?.detach())
    }

    /// The residual stream at the last token after every block, and after every block of each extra loop pass, in the
    /// order they run: `(block, pass, [hidden] f32 on the CPU)`. Plain inference, no adapter involved (abliteration
    /// reads its refusal direction from these).
    pub fn residuals(&self, tokens: &[u32]) -> Result<Vec<(usize, usize, Tensor)>> {
        let mut out = Vec::new();
        let last = tokens.len().checked_sub(1).ok_or_else(|| anyhow!("an empty token sequence"))?;
        let mut keep = |b: usize, p: usize, x: &Tensor| -> Result<()> {
            out.push((b, p, x.get(last)?.to_dtype(DType::F32)?.to_device(&Device::Cpu)?.detach()));
            Ok(())
        };
        self.run_blocks_seen(self.embed_tokens(tokens)?, 0..self.cfg.num_hidden_layers, &Ctx::inference(), Some(&mut keep))?;
        Ok(out)
    }

    /// From that hidden state through the trained blocks to the final norm: `[seq, hidden]`.
    pub fn upper_hidden(&self, h: &Tensor, ctx: &Ctx) -> Result<Tensor> {
        let x = self.run_blocks(h.clone(), self.first_trained..self.cfg.num_hidden_layers, ctx)?;
        rms_norm(&move_to(&x, self.norm.device())?, &self.norm, self.cfg.rms_norm_eps)
    }

    /// The output head on `[rows, hidden]` → `[rows, vocab]`.
    pub fn head(&self, h: &Tensor) -> Result<Tensor> {
        let w = self.lm_head.as_ref().or(self.embed.as_ref()).ok_or_else(|| anyhow!("no output head in memory"))?.clone();
        let h = move_to(h, w.device())?;
        Linear { w, bias: None, lora: None }.forward(&h, &Ctx::inference())
    }

    /// From the cached hidden state to logits `[seq, vocab]`.
    pub fn upper(&self, h: &Tensor, ctx: &Ctx) -> Result<Tensor> {
        self.head(&self.upper_hidden(h, ctx)?)
    }

    /// Token ids → logits `[seq, vocab]` (f32).
    pub fn forward(&self, tokens: &[u32], ctx: &Ctx) -> Result<Tensor> {
        self.upper(&self.lower(tokens)?, ctx)
    }

    /// The mean next-token cross-entropy over the supervised positions: `supervised[p]` says token `p` is a target
    /// (predicted from position `p − 1`). The head runs only on the rows that are scored.
    pub fn loss_from_hidden(&self, h: &Tensor, tokens: &[u32], supervised: &[bool], ctx: &Ctx) -> Result<Tensor> {
        let rows: Vec<u32> = (1..tokens.len()).filter(|p| supervised[*p]).map(|p| (p - 1) as u32).collect();
        if rows.is_empty() {
            bail!("an example with no supervised token");
        }
        let targets: Vec<u32> = rows.iter().map(|r| tokens[*r as usize + 1]).collect();
        let n = rows.len();
        let hidden = self.upper_hidden(h, ctx)?;
        let hidden = hidden.index_select(&Tensor::from_vec(rows, (n,), hidden.device())?, 0)?;
        let w = self.lm_head.as_ref().or(self.embed.as_ref()).ok_or_else(|| anyhow!("no output head in memory"))?;
        let ce = head_loss(&move_to(&hidden, w.device())?, w, &targets)?;
        // the upcycled blocks' load-balancing losses, averaged over the blocks (and loop passes) that ran
        let aux = std::mem::take(&mut *ctx.aux.borrow_mut());
        if aux.is_empty() {
            return Ok(ce);
        }
        let n_aux = aux.len() as f64;
        let mut sum = move_to(&aux[0], ce.device())?;
        for a in &aux[1..] {
            sum = (sum + move_to(a, ce.device())?)?;
        }
        let mean = (sum / n_aux)?;
        let both = mean.to_vec1::<f32>()?;
        ctx.aux_value.set(Some(both[0] as f64));
        if self.cfg.moe.is_some_and(|m| m.fine(&self.cfg)) {
            ctx.z_value.set(Some(both[1] as f64));
        }
        let coef = Tensor::new(&[MOE_AUX_COEF as f32, MOE_Z_COEF as f32], mean.device())?;
        Ok((ce + (mean * coef)?.sum_all()?)?)
    }

    pub fn loss(&self, tokens: &[u32], supervised: &[bool], ctx: &Ctx) -> Result<Tensor> {
        self.loss_from_hidden(&self.lower(tokens)?, tokens, supervised, ctx)
    }
}

/// What PEFT reads from `adapter_config.json`, for `spec` on `cfg` (`base` is recorded as the base model's path).
pub fn adapter_config(spec: &LoraSpec, cfg: &ModelConfig, base: &str) -> Value {
    let first = spec.first_trained(cfg.num_hidden_layers);
    let layers: Value = if first == 0 { Value::Null } else { serde_json::json!((first..cfg.num_hidden_layers).collect::<Vec<_>>()) };
    serde_json::json!({
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "base_model_name_or_path": base,
        "r": spec.r,
        "lora_alpha": spec.alpha,
        "lora_dropout": spec.dropout,
        "use_rslora": spec.use_rslora,
        "use_dora": false,
        "target_modules": spec.targets.iter().map(|t| t.as_str()).collect::<Vec<_>>(),
        "layers_to_transform": layers,
        "bias": "none",
        "fan_in_fan_out": false,
        "inference_mode": true,
        "init_lora_weights": true,
        "modules_to_save": Value::Null,
        "prastara_moe": cfg.moe.map_or(Value::Null, |m| { let mut v = m.to_json(); v["rank"] = serde_json::json!(spec.r); v }),
    })
}

/// The feed-forward projections every expert of an upcycled block adapts, in its `[Lora; 3]` order.
pub const MOE_TARGETS: [Target; 3] = [Target::Gate, Target::Up, Target::Down];

/// An upcycled block's router in the adapter (`…layers.N.mlp.gate.weight`, `[experts, hidden]`: Qwen3-MoE's name).
pub fn moe_router_name(block: usize) -> String {
    format!("base_model.model.model.layers.{block}.mlp.gate.weight")
}

/// An expert's adapter stem (`…layers.N.mlp.experts.E.gate_proj`), `.lora_A.weight` / `.lora_B.weight` after it.
pub fn moe_expert_stem(block: usize, expert: usize, t: Target) -> String {
    format!("base_model.model.model.layers.{block}.mlp.experts.{expert}.{}", t.as_str())
}

/// A neuron group's shared adapter stem (`…layers.N.mlp.groups.G.gate_proj`): its `.lora_A.weight` serves every copy
/// of the group ([`MoeSpec::shared_a`]); each copy's `.lora_B.weight` stays under its expert stem.
pub fn moe_group_stem(block: usize, group: usize, t: Target) -> String {
    format!("base_model.model.model.layers.{block}.mlp.groups.{group}.{}", t.as_str())
}

/// The adapter tensors as an ordered map, for saving.
pub fn adapter_tensors(model: &LlamaModel) -> BTreeMap<String, Tensor> {
    model.lora_vars().into_iter().map(|(n, v)| (n, v.as_tensor().detach())).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_sliced_head_loss_equals_cross_entropy_and_its_gradient() {
        let dev = Device::Cpu;
        // 37 rows over a 1,000-entry vocabulary, in slices of 8 rows (8,000 values), so the last slice is short
        HEAD_SLICE_VALUES.store(8_000, std::sync::atomic::Ordering::Relaxed);
        for dt in [DType::F32, DType::BF16] {
            let h = Var::from_tensor(&(Tensor::randn(0f32, 1.0, (37, 24), &dev).unwrap())).unwrap();
            let w = (Tensor::randn(0f32, 0.5, (1000, 24), &dev).unwrap()).to_dtype(dt).unwrap();
            let targets: Vec<u32> = (0..37u32).map(|i| (i * 97 + 5) % 1000).collect();
            let ours = head_loss(h.as_tensor(), &w, &targets).unwrap();
            let wf = w.to_dtype(DType::F32).unwrap();
            let reference = candle_nn::loss::cross_entropy(&h.as_tensor().matmul(&wf.t().unwrap()).unwrap(),
                &Tensor::from_vec(targets.clone(), (37,), &dev).unwrap()).unwrap();
            let (a, b) = (ours.to_scalar::<f32>().unwrap(), reference.to_scalar::<f32>().unwrap());
            assert!((a - b).abs() < 1e-5 * b.abs().max(1.0), "{dt:?}: loss {a} vs {b}");
            let ga = ours.backward().unwrap().get(h.as_tensor()).unwrap().clone();
            let gb = reference.backward().unwrap().get(h.as_tensor()).unwrap().clone();
            let diff = (ga - &gb).unwrap().abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap();
            let scale = gb.abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap();
            assert!(diff < 1e-5 * scale.max(1e-3) + 1e-7, "{dt:?}: gradient off by {diff} (largest {scale})");
        }
        HEAD_SLICE_VALUES.store(32 << 20, std::sync::atomic::Ordering::Relaxed);
    }

    /// candle's `matmul` backward computes a gradient for both operands: through it, every frozen weight on the path
    /// got a full-size f32 gradient each step (542 MB peak instead of 228 MB on a 12-block model). No gradient may be
    /// bigger than the largest adapter matrix or activation; a weight-sized one means a frozen weight got one.
    /// A large 16-bit matrix is widened a slice of rows at a time: the same values and input gradient as all at once.
    #[test]
    fn a_sliced_widening_gives_the_same_product_and_gradient() {
        let (o, i, rows) = (37, 24, 5);
        let w = Tensor::from_vec((0..o * i).map(|k| ((k * 7) % 31) as f32 / 31.0 - 0.5).collect::<Vec<f32>>(), (o, i), &Device::Cpu).unwrap()
            .to_dtype(DType::BF16).unwrap();
        let run = || {
            let x = Var::from_tensor(&Tensor::from_vec((0..rows * i).map(|k| (k % 5) as f32 * 0.3 - 0.6).collect::<Vec<f32>>(), (rows, i), &Device::Cpu).unwrap()).unwrap();
            let y = Linear { w: w.clone(), bias: None, lora: None }.forward(x.as_tensor(), &Ctx::eval()).unwrap();
            let g = y.sqr().unwrap().sum_all().unwrap().backward().unwrap();
            (y.to_vec2::<f32>().unwrap(), g.get(&x).unwrap().to_vec2::<f32>().unwrap())
        };
        let whole = run();
        WIDEN_VALUES_MAX.store(10 * i, std::sync::atomic::Ordering::Relaxed); // slices of 10 rows: 4 slices, the last of 7
        let sliced = run();
        WIDEN_VALUES_MAX.store(64 << 20, std::sync::atomic::Ordering::Relaxed);
        assert_eq!(whole.0, sliced.0, "each row's product is computed as before");
        let d = whole.1.iter().flatten().zip(sliced.1.iter().flatten()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(d < 1e-5, "the input gradient sums the slices: {d}");
    }

    #[test]
    fn regress_frozen_weights_get_no_gradient() {
        let (o, i, rows) = (64, 48, 3);
        let w = Tensor::from_vec((0..o * i).map(|k| (k % 13) as f32 / 13.0 - 0.5).collect::<Vec<f32>>(), (o, i), &Device::Cpu).unwrap();
        for dtype in [DType::F32, DType::F16, DType::BF16] {
            let lin = Linear { w: w.to_dtype(dtype).unwrap(), bias: None, lora: None };
            let x = Var::from_tensor(&Tensor::ones((rows, i), DType::F32, &Device::Cpu).unwrap()).unwrap();
            let grads = lin.forward(x.as_tensor(), &Ctx::eval()).unwrap().sqr().unwrap().sum_all().unwrap().backward().unwrap();
            assert!(grads.get(&x).is_some(), "{dtype:?}: the input's gradient");
            let biggest = grads.get_ids().filter_map(|id| grads.get_id(*id)).map(|g| g.elem_count()).max().unwrap();
            assert!(biggest <= rows * o, "{dtype:?}: a gradient of {biggest} values, the weight has {}", o * i);
        }
    }
}

/// Avarta conversion (docs/MULA-PLAN.md M1): `out` becomes a copy of the model folder `model` whose blocks
/// `start..end` run `passes` times. The weights are linked, not copied (copied where links are not possible);
/// `config.json` gains `"prastara_loop"`, and `loop.safetensors` holds the gates, all zero, so the converted model's
/// logits are exactly the base's until training opens them.
pub fn make_looped(model: &Path, out: &Path, l: LoopSpec) -> Result<()> {
    let mut cfg: Value = serde_json::from_str(&std::fs::read_to_string(model.join("config.json"))
        .with_context(|| format!("{}: no config.json", model.display()))?)?;
    let parsed = ModelConfig::from_json(&cfg)?;
    if parsed.loop_.is_some() {
        bail!("{} is already looped ({:?}); convert its base instead", model.display(), parsed.loop_.unwrap());
    }
    l.check(parsed.num_hidden_layers)?;
    if out.exists() && std::fs::read_dir(out)?.next().is_some() {
        bail!("{} exists and is not empty", out.display());
    }
    std::fs::create_dir_all(out)?;
    link_model_files(model, out, &["config.json", LOOP_FILE])?;
    cfg["prastara_loop"] = l.to_json();
    std::fs::write(out.join("config.json"), serde_json::to_string_pretty(&cfg)?)?;
    let zeros: HashMap<String, Tensor> = (1..l.passes)
        .map(|p| Ok((format!("loop.gate.{p}"), Tensor::zeros(parsed.hidden_size, DType::F32, &Device::Cpu)?)))
        .collect::<Result<_>>()?;
    candle_core::safetensors::save(&zeros, out.join(LOOP_FILE))?;
    Ok(())
}

/// LoRA upcycling (plan Part C): `out` becomes a copy of the dense model folder `model` (looped or not) whose
/// config carries `"prastara_moe": {experts, top_k}`; the weights are linked, as [`make_looped`] does. Training then
/// gives every trained block a router and `experts` expert LoRAs over the shared feed-forward; until then the copy
/// is the base. Returns the mixture's `(total, active)` parameter counts once exported.
pub fn make_upcycled(model: &Path, out: &Path, m: MoeSpec) -> Result<(u64, u64)> {
    let dense = start_upcycled(model, out, m)?;
    if m.fine(&dense) {
        // no calibration text: a contiguous split and a seeded router (crate::moe_init::upcycle calibrates)
        crate::moe_init::write_uncalibrated(out, &dense, m, 42)?;
    }
    Ok(m.params(&dense))
}

/// The common start of an upcycle: checks, the linked copy and its config with `"prastara_moe"`. Returns the dense
/// model's config.
pub fn start_upcycled(model: &Path, out: &Path, m: MoeSpec) -> Result<ModelConfig> {
    let mut cfg: Value = serde_json::from_str(&std::fs::read_to_string(model.join("config.json"))
        .with_context(|| format!("{}: no config.json", model.display()))?)?;
    // a mixture of experts is refused by name in from_json (num_experts, num_local_experts, ...)
    let parsed = ModelConfig::from_json(&cfg).with_context(|| format!("{}: upcycling starts from a dense Llama, Mistral, Qwen2 or Qwen3", model.display()))?;
    if parsed.moe.is_some() {
        bail!("{} is already upcycled ({:?}); upcycle its base instead", model.display(), parsed.moe.unwrap());
    }
    m.check_with(&parsed)?;
    if out.exists() && std::fs::read_dir(out)?.next().is_some() {
        bail!("{} exists and is not empty", out.display());
    }
    std::fs::create_dir_all(out)?;
    link_model_files(model, out, &["config.json", MOE_INIT_FILE])?;
    cfg["prastara_moe"] = m.to_json();
    std::fs::write(out.join("config.json"), serde_json::to_string_pretty(&cfg)?)?;
    Ok(parsed)
}

/// Every file of `model` except `skip`, linked into `out` (copied where links are not possible).
fn link_model_files(model: &Path, out: &Path, skip: &[&str]) -> Result<()> {
    for e in std::fs::read_dir(model)? {
        let e = e?;
        let name = e.file_name();
        if skip.iter().any(|s| name == *s) || !e.file_type()?.is_file() && !e.file_type()?.is_symlink() {
            continue;
        }
        let src = std::fs::canonicalize(e.path())?;
        let dst = out.join(&name);
        #[cfg(unix)]
        let linked = std::os::unix::fs::symlink(&src, &dst).is_ok();
        #[cfg(not(unix))]
        let linked = std::fs::hard_link(&src, &dst).is_ok();
        if !linked {
            std::fs::copy(&src, &dst).with_context(|| format!("copying {}", src.display()))?;
        }
    }
    Ok(())
}

/// A long-context copy of a model folder (docs/MULA-PLAN.md L1, L2): YaRN positions stretched to `length` over the
/// model's original training length, and every `full_every`-th layer kept at full attention while the others
/// attend to the last `window` tokens. The weights are linked; a loop and its gates are kept. The copy still needs
/// long-context training (L4) before it is used at that length.
pub fn make_long_context(model: &Path, out: &Path, length: usize, original: Option<usize>, window: usize, full_every: usize) -> Result<()> {
    let mut cfg: Value = serde_json::from_str(&std::fs::read_to_string(model.join("config.json"))
        .with_context(|| format!("{}: no config.json", model.display()))?)?;
    let parsed = ModelConfig::from_json(&cfg)?;
    if parsed.yarn.is_some() || parsed.layer_sliding.iter().any(|x| *x) {
        bail!("{} already has a position scaling or sliding layers; start from its base", model.display());
    }
    let original = original.unwrap_or(parsed.max_position_embeddings);
    if length <= original {
        bail!("{length} tokens is not past the model's {original}");
    }
    if full_every == 0 || window == 0 {
        bail!("--full-every and --window must be above 0");
    }
    if out.exists() && std::fs::read_dir(out)?.next().is_some() {
        bail!("{} exists and is not empty", out.display());
    }
    std::fs::create_dir_all(out)?;
    link_model_files(model, out, &["config.json"])?;
    let factor = length as f64 / original as f64;
    let yarn = serde_json::json!({"rope_type": "yarn", "factor": factor, "original_max_position_embeddings": original});
    cfg["rope_scaling"] = yarn.clone();
    if cfg.get("rope_parameters").is_some_and(|v| v.is_object()) {
        let theta = cfg["rope_parameters"]["rope_theta"].clone();
        let mut rp = yarn;
        rp["rope_theta"] = theta;
        cfg["rope_parameters"] = rp;
    }
    cfg["max_position_embeddings"] = serde_json::json!(length);
    // every full_every-th layer full, and the last one always (it reads the whole text before the output)
    let n = parsed.num_hidden_layers;
    let types: Vec<&str> = (0..n).map(|i| if (i + 1) % full_every == 0 || i + 1 == n { "full_attention" } else { "sliding_attention" }).collect();
    cfg["layer_types"] = serde_json::json!(types);
    cfg["sliding_window"] = serde_json::json!(window);
    cfg["use_sliding_window"] = serde_json::json!(true);
    std::fs::write(out.join("config.json"), serde_json::to_string_pretty(&cfg)?)?;
    ModelConfig::from_json(&cfg).context("the written config")?;
    Ok(())
}
