//! Step 5 — fold a LoRA adapter into its base safetensors, shard by shard.
//!
//! The base is opened with [`crate::safetensors_io`] (each shard mmap'd, never the whole model loaded at once). Every
//! base weight gets each trained adapter's delta `scaling * B @ A` added to it (PEFT folds the adapter into the base on
//! export); everything else — norms, MLP, embeddings, and untargeted projections — is copied back byte for byte. The
//! output is written by the streaming writer, so a merged checkpoint drops straight into transformers and the engine.
//!
//! * **Scaling** (`LoraConfig::scaling`): `alpha / r`, or `alpha / sqrt(r)` with rsLoRA, read from `adapter_config.json`.
//! * **Name mapping** ([`peft_to_base_name`]): drop `base_model.model.` and the `.lora_A.weight`/`.lora_B.weight` suffix,
//!   then add `.weight`, so `...q_proj.lora_B.weight` -> `...q_proj.weight`.
//! * **2-D targets** (the q/k/v/o projections): `delta[o][i] = scaling * sum_r B[o][r] * A[r][i]`.
//! * **Fused MoE experts** ([`add_expert_delta`]): a 3-D `[E, d1, d2]` target takes the expert einsum `einsum("o r e, e r i -> e*")`, per-expert
//!   `scaling * B_e @ A_e`. PEFT has two layouts (`-> e i o` by default, `-> e o i` after `_did_swap_in_out_features`) and the fold picks the one
//!   whose output shape matches the parameter's own `[E, d1, d2]`, refusing a square target its layouts are indistinguishable on.
//! * **Output** ([`crate::st_writer`]): written a tensor at a time in ~5 GB shards, `model.safetensors` or
//!   `model-0000i-of-0000n.safetensors` with an index whose `weight_map` names each tensor's file.
//! * **Report** ([`MergeReport`]): per tensor the relative-norm delta `|ΔW|/|W|` — targeted tensors are > 0, untargeted
//!   ones are exactly 0 (their bytes never left disk).

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, Result};
use candle_core::{Device, Tensor};
use serde_json::Value;

use crate::checkpoint;
use crate::safetensors_io::{Dtype, Shard};
use crate::st_writer::{self, StWriter, TensorSpec};

/// One base weight and what it became after the merge: same shape, plus how big its relative change was.
#[derive(Debug, Clone)]
pub struct TensorDelta {
    pub name: String,
    pub shape: Vec<usize>,
    /// |ΔW| / |W| over the original base weight; 0.0 for a tensor the adapter never touched.
    pub delta_rel_norm: f64,
}

/// The outcome of [`merge`]: where the merged checkpoint was written and a per-tensor relative-norm summary.
pub struct MergeReport {
    pub out_dir: PathBuf,
    pub per_tensor: Vec<TensorDelta>,
}

/// A LoRA delta pair for one base weight, resolved from the adapter's A/B matrices. The 2-D vs fused-MoE path is chosen by
/// the base weight's rank at merge time ([`run`]).
#[derive(Clone)]
struct Adapter {
    a: Vec<f32>,
    b: Vec<f32>,
}

/// A base weight read from a shard: its name, dtype, shape, and the raw bytes straight off the mmap.
struct BaseTensor<'s> {
    name: String,
    dtype: Dtype,
    shape: Vec<usize>,
    bytes: &'s [u8],
}

/// Merge `adapter_dir`'s adapter into every base shard under `base_dir`, writing the result to `out_dir`. Scaling is read
/// from `adapter_config.json`; set a [`Merger`] with [`Merger::set_scaling`] to fold in a deliberately wrong factor.
pub struct Merger<'a> {
    base_dir: &'a Path,
    adapter_dir: &'a Path,
    out_dir: &'a Path,
    scaling_override: Option<f32>,
    shard_bytes: u64,
}

impl<'a> Merger<'a> {
    pub fn new(base_dir: &'a Path, adapter_dir: &'a Path, out_dir: &'a Path) -> Self {
        Self { base_dir, adapter_dir, out_dir, scaling_override: None, shard_bytes: st_writer::SHARD_BYTES }
    }

    /// Force a scaling factor instead of the one in `adapter_config.json` (used by the rsLoRA gate to prove the default
    /// scaling is what makes the merged logits match). The wrong factor must fail the logit-parity gate.
    pub fn set_scaling(&mut self, s: f32) -> &mut Self {
        self.scaling_override = Some(s);
        self
    }

    /// The largest output shard (default [`st_writer::SHARD_BYTES`], about 5 GB).
    pub fn set_shard_bytes(&mut self, n: u64) -> &mut Self {
        self.shard_bytes = n;
        self
    }

    pub fn run(self) -> Result<MergeReport> {
        let scaling = match self.scaling_override {
            Some(s) => s,
            None => load_scaling(self.adapter_dir)?,
        };
        // The whole model is never resident at once: one tensor at a time is read off its shard's mmap and written out.
        // Only the tiny adapter (already on the CPU) is fully materialised into Rust.
        // `model.safetensors`, or the `model-0000i-of-0000n` shards, as the folder has them
        let shard_files = crate::model::weight_files(self.base_dir)?;

        let adapters = load_adapters(self.adapter_dir)?;
        // every output tensor keeps its name, dtype and shape: the writer plans ~5 GB shards from the list, then each
        // tensor is read off its shard's mmap, folded when targeted, and written before the next is read
        let mut specs = Vec::new();
        for sf in &shard_files {
            let shard = Shard::open(sf)?;
            specs.extend(shard.tensors().map(|t| TensorSpec::new(t.name(), t.dtype(), t.shape())));
        }
        st_writer::prepare_out(self.base_dir, self.out_dir)?;
        let mut writer = StWriter::create(self.out_dir, specs, self.shard_bytes)?;
        let mut report: Vec<TensorDelta> = Vec::new();
        for sf in &shard_files {
            let shard = Shard::open(sf)?;
            for t in shard.tensors() {
                let wt = BaseTensor { name: t.name().to_string(), dtype: t.dtype(), shape: t.shape().to_vec(), bytes: shard.bytes(t) };
                let rel_norm = match adapters.get(&wt.name) {
                    None => {
                        writer.write(&wt.name, wt.bytes)?; // untouched: the mmap bytes go back out verbatim
                        0.0
                    }
                    Some(ad) => {
                        let base = to_f32s(&wt.name, wt.dtype, wt.bytes)?;
                        let (w, delta) = match wt.shape.len() {
                            2 => {
                                let (o, in_) = (wt.shape[0], wt.shape[1]);
                                let r = ad.a.len().div_ceil(in_);
                                add_delta_2d(&base, &ad.a, &ad.b, o, in_, r, scaling)
                            }
                            3 => add_expert_delta(&wt.name, &base, &ad.a, &ad.b, wt.shape[0], wt.shape[1], wt.shape[2], scaling)?,
                            other => return Err(anyhow!("{}: base weight of rank {other} is not a LoRA target", wt.name)),
                        };
                        writer.write(&wt.name, &from_f32s(wt.dtype, &w))?;
                        rel_norm(&base, &delta)
                    }
                };
                report.push(TensorDelta { name: wt.name.clone(), shape: wt.shape.clone(), delta_rel_norm: rel_norm });
            }
        }
        writer.finish()?;

        // what makes the folder a model again: config.json, the tokenizer and its template, copied as they are
        for e in std::fs::read_dir(self.base_dir)?.flatten() {
            let (path, name) = (e.path(), e.file_name().to_string_lossy().to_string());
            if path.is_file() && !name.ends_with(".safetensors") && name != "model.safetensors.index.json" && !self.out_dir.join(&name).exists() {
                std::fs::copy(&path, self.out_dir.join(&name))?;
            }
        }

        // an adapter trained on private data (crate::private_data) makes a private model: its config says so, and the
        // GGUF converted from it carries prastara.private
        let acfg: serde_json::Value = std::fs::read_to_string(self.adapter_dir.join(checkpoint::CONFIG_FILE)).ok()
            .and_then(|t| serde_json::from_str(&t).ok()).unwrap_or_default();
        if acfg["prastara_private"] == true && self.out_dir.join("config.json").exists() {
            crate::private_data::mark_config_private(&self.out_dir.join("config.json"))?;
        }

        // a looped model (Avarta): its gates are the adapter's when it trained them, else the base's as they were
        let base_gates = self.base_dir.join(crate::model::LOOP_FILE);
        let adps = candle_core::safetensors::load(self.adapter_dir.join(checkpoint::ADAPTER_FILE), &Device::Cpu)?;
        let trained: std::collections::HashMap<String, Tensor> = adps.into_iter()
            .filter_map(|(n, t)| n.split_once(".loop.gate.").map(|(_, p)| (format!("loop.gate.{p}"), t))).collect();
        if !trained.is_empty() {
            let mut gates = if base_gates.exists() { candle_core::safetensors::load(&base_gates, &Device::Cpu)? } else { Default::default() };
            gates.extend(trained);
            candle_core::safetensors::save(&gates, self.out_dir.join(crate::model::LOOP_FILE))?;
        } else if base_gates.exists() {
            std::fs::copy(&base_gates, self.out_dir.join(crate::model::LOOP_FILE))?;
        }

        Ok(MergeReport { out_dir: self.out_dir.to_path_buf(), per_tensor: report })
    }
}

/// Shorthand for [`Merger::run`]: fold `adapter_dir`'s adapter into the base under `base_dir`, writing to `out_dir`.
pub fn merge(base_dir: &Path, adapter_dir: &Path, out_dir: &Path) -> Result<MergeReport> {
    Merger::new(base_dir, adapter_dir, out_dir).run()
}

/// Map a PEFT adapter tensor name to the base weight it folds into. Drop `base_model.model.` and the `.lora_{A,B}.weight`
/// suffix, then re-add `.weight`: `...q_proj.lora_B.weight` -> `...q_proj.weight`. Returns `None` for anything not a LoRA
/// matrix (a plain base weight never carries this shape anyway).
pub fn peft_to_base_name(peft: &str) -> Option<String> {
    let s = peft.strip_prefix("base_model.model.")?;
    if s.ends_with(".lora_A.weight") || s.ends_with(".lora_B.weight") {
        Some(format!("{}.weight", s.strip_suffix(".lora_A.weight").or_else(|| s.strip_suffix(".lora_B.weight"))?))
    } else {
        None
    }
}

/// Build the adapter map (PEFT name -> [`Adapter`]) from the loaded A/B matrices. Only fused MoE targets are 3-D; the merge
/// routes by the base weight's rank at fold time, so here both shapes arrive as flat f32 and dispatch happens later.
fn build_adapters(adps: &std::collections::HashMap<String, Tensor>) -> Result<HashMap<String, Adapter>> {
    let mut a_of: HashMap<String, Vec<f32>> = HashMap::new();
    let mut b_of: HashMap<String, Vec<f32>> = HashMap::new();
    for (name, t) in adps {
        if name.contains(".loop.gate.") {
            continue; // a looped model's gates: written to loop.safetensors, not folded into a weight
        }
        let flat = t.flatten_all()?.to_vec1::<f32>()?;
        if name.ends_with(".lora_A.weight") {
            a_of.insert(peft_to_base_name(name).ok_or_else(|| anyhow!("{name}: not a LoRA matrix"))?, flat);
        } else {
            b_of.insert(peft_to_base_name(name).ok_or_else(|| anyhow!("{name}: not a LoRA matrix"))?, flat);
        }
    }
    let mut out = HashMap::new();
    for (base, a) in &a_of {
        let b = b_of.get(base).ok_or_else(|| anyhow!("adapter has lora_A for {base} but no lora_B"))?;
        out.insert(base.clone(), Adapter { a: a.clone(), b: b.clone() });
    }
    Ok(out)
}

/// Fold `scaling * B @ A` into a 2-D weight. Row-major: `A:[r,in]`, `B:[out,r]`, `W:[out,in]`. Returns the new weight and
/// the delta (same length) so the caller can report the relative norm.
fn add_delta_2d(base: &[f32], a: &[f32], b: &[f32], out: usize, in_: usize, r: usize, scaling: f32) -> (Vec<f32>, Vec<f32>) {
    let mut w = base.to_vec();
    let mut delta = vec![0f32; out * in_];
    for o in 0..out {
        for i in 0..in_ {
            let mut acc = 0f32;
            for rr in 0..r {
                acc += b[o * r + rr] * a[rr * in_ + i];
            }
            delta[o * in_ + i] = scaling * acc;
        }
    }
    for (wi, di) in w.iter_mut().zip(delta.iter()) {
        *wi += *di;
    }
    (w, delta)
}

/// Fold `scaling * B @ A` into a fused-MoE expert weight of the parameter's own shape `[E, d1, d2]` — which is also the delta's.
/// PEFT computes each expert `scaling * B_e @ A_e` with one of two layouts: default `einsum("o r e, e r i -> e i o")`, a delta of
/// shape `[E, in, out]`, and after `_did_swap_in_out_features` `einsum("o r e, e r i -> e o i")`, a delta of `[E, out, in]`. The
/// flattened A is `[E*r, a_in]` and B is `[b_out, r*E]` either way, so `(a_in, b_out)` is the parameter's inner dims in some order:
/// pick the layout matching it against `(d1, d2)`, refuse when `d1 == d2` (both layouts look identical from the files), and refuse
/// anything that matches neither. `name` names the tensor in those error messages.
fn add_expert_delta(name: &str, base: &[f32], a: &[f32], b: &[f32], e: usize, d1: usize, d2: usize, scaling: f32) -> Result<(Vec<f32>, Vec<f32>)> {
    let len_a = a.len();
    let len_b = b.len();

    // Default layout `-> e i o`: (a_in, b_out) == (d1, d2). The result is placed at delta[e][ii][oo], so base dim d1 is `in`, d2 is `out`.
    if let Some(r) = layout_rank(len_a, e, d1, len_b, d2) {
        if d1 == d2 {
            return Err(anyhow!("{name}: fused-expert parameter is square [{e}, {d1}, {d2}]; PEFT's two expert layouts are indistinguishable from the shapes alone, so refuse rather than guess"));
        }
        return Ok(expert_delta(base, a, b, e, d2, d1, false, r, scaling));
    }

    // Swapped layout `-> e o i`: (a_in, b_out) == (d2, d1). The result is placed at delta[e][oo][ii], so base dim d1 is `out`, d2 is `in`.
    if let Some(r) = layout_rank(len_a, e, d2, len_b, d1) {
        return Ok(expert_delta(base, a, b, e, d1, d2, true, r, scaling));
    }

    if d1 == d2 {
        return Err(anyhow!("{name}: fused-expert parameter is square [{e}, {d1}, {d2}]; PEFT's two expert layouts are indistinguishable from the shapes alone, so refuse rather than guess"));
    }
    Err(anyhow!("{name}: fused-expert adapter A[E*r, {len_a}/{e}] and B[{len_b}/{e}, r*E] don't fit parameter shape [{e}, {d1}, {d2}]"))
}

/// Consistent rank when A is `[E*r, a_in]` and B is `[b_out, r*E]`: returns `r` only if both flat lengths divide evenly on it and the
/// two independent estimates agree — this is what disambiguates the two PEFT layouts from shapes alone.
fn layout_rank(len_a: usize, e: usize, a_in: usize, len_b: usize, out_b: usize) -> Option<usize> {
    if e == 0 || a_in == 0 || out_b == 0 {
        return None;
    }
    if len_a % (e * a_in) != 0 || len_b % (e * out_b) != 0 {
        return None;
    }
    let r_a = len_a / (e * a_in);
    let r_b = len_b / (e * out_b);
    if r_a == r_b { Some(r_a) } else { None }
}

/// Per-expert `scaling * B_e @ A_e` folded into `base`, writing the result as `[E, out_dim, in_dim]` when `o_first` is true (delta[e][oo][ii],
/// base dim d1 is the outer `out`) or `[E, in_dim, out_dim]` when false (delta[e][ii][oo]). A is `[E*r, a_in=in_dim]`, B is `[b_out=out_dim, r*E]`;
/// both axes are contracted over rank `r`.
fn expert_delta(base: &[f32], a: &[f32], b: &[f32], e: usize, out_dim: usize, in_dim: usize, o_first: bool, r: usize, scaling: f32) -> (Vec<f32>, Vec<f32>) {
    let mut w = base.to_vec();
    let mut delta = vec![0f32; e * out_dim * in_dim];
    let re = b.len() / out_dim; // r*E: stride pairing B's contracted rank with its expert axis
    for ee in 0..e {
        for oo in 0..out_dim {
            for ii in 0..in_dim {
                let mut acc = 0f32;
                // b reshaped [out,r,E]: element[oo][rr][ee] lives at flat oo*(r*E) + rr*E + ee
                for rr in 0..r {
                    let be = b[oo * re + rr * e + ee];
                    // a reshaped [E,r,in]: element[ee][rr][ii] lives at flat (ee*r + rr)*in + ii
                    let ae = a[(ee * r + rr) * in_dim + ii];
                    acc += be * ae;
                }
                if o_first {
                    delta[(ee * out_dim + oo) * in_dim + ii] = scaling * acc; // delta[e][oo][ii], base dim d1 is `out`
                } else {
                    delta[(ee * in_dim + ii) * out_dim + oo] = scaling * acc; // delta[e][ii][oo], base dim d1 is `in`
                }
            }
        }
    }
    for (wi, di) in w.iter_mut().zip(delta.iter()) {
        *wi += *di;
    }
    (w, delta)
}

/// Relative L2 norm of `delta` over the original weight `w` (unchanged by the caller).
fn rel_norm(w: &[f32], delta: &[f32]) -> f64 {
    let wn = w.iter().fold(0.0_f64, |a, &x| a + x as f64 * x as f64);
    let dn = delta.iter().fold(0.0_f64, |a, &x| a + x as f64 * x as f64);
    (dn.sqrt() / wn.sqrt()).clamp(0.0, f64::INFINITY)
}

/// Scaling from `adapter_config.json`: `alpha / r`, or `alpha / sqrt(r)` when rsLoRA is on.
fn load_scaling(adapter_dir: &Path) -> Result<f32> {
    let cfg: Value = serde_json::from_str(&std::fs::read_to_string(adapter_dir.join(checkpoint::CONFIG_FILE))?)?;
    let r = cfg["r"].as_u64().ok_or_else(|| anyhow!("adapter_config.json has no r"))?;
    let alpha = cfg["lora_alpha"].as_f64().ok_or_else(|| anyhow!("adapter_config.json has no lora_alpha"))? as f32;
    Ok(apply_scaling(alpha, r, cfg["use_rslora"].as_bool().unwrap_or(false)))
}

/// The scaling factor for rank `r`, alpha and rsLoRA flag — duplicated from [`crate::lora::LoraConfig`] so the merge works
/// off the on-disk config alone.
fn apply_scaling(alpha: f32, r: u64, use_rslora: bool) -> f32 {
    let r = r as f32;
    if use_rslora {
        alpha / r.sqrt()
    } else {
        alpha / r
    }
}

/// Load the adapter's A/B matrices (flat f32) under their PEFT names.
fn load_adapters(adapter_dir: &Path) -> Result<HashMap<String, Adapter>> {
    let adps = candle_core::safetensors::load(adapter_dir.join(checkpoint::ADAPTER_FILE), &Device::Cpu)?;
    build_adapters(&adps)
}

/// A weight's bytes as f32 values: F32 as stored, F16 and BF16 widened (exactly). Real Hugging Face models are mostly
/// BF16; reading their bytes as f32 would merge into garbage, so any other dtype is refused by name.
fn to_f32s(name: &str, dtype: Dtype, b: &[u8]) -> Result<Vec<f32>> {
    Ok(match dtype {
        Dtype::F32 => b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
        Dtype::F16 => b.chunks_exact(2).map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32()).collect(),
        Dtype::BF16 => b.chunks_exact(2).map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32()).collect(),
        other => return Err(anyhow!("{name}: a {} weight cannot take a LoRA delta (F32, F16 and BF16 can)", other.as_str())),
    })
}

/// f32 values back to the weight's own dtype (F16 and BF16 round to nearest, as a cast does).
fn from_f32s(dtype: Dtype, v: &[f32]) -> Vec<u8> {
    match dtype {
        Dtype::F16 => v.iter().flat_map(|x| half::f16::from_f32(*x).to_le_bytes()).collect(),
        Dtype::BF16 => v.iter().flat_map(|x| half::bf16::from_f32(*x).to_le_bytes()).collect(),
        _ => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
    }
}

#[cfg(test)]
#[path = "merge_unit.rs"]
mod unit_tests;
