//! `export-moe`: an upcycled model folder (`"prastara_moe"` in its config, [`crate::model::make_upcycled`]) and its
//! trained adapter to a real `qwen3moe` GGUF (plan Part C). Every expert becomes a full copy of the block's
//! feed-forward with its own LoRA folded in (`W + s · B_e A_e`); the router is the adapter's; attention and norms are
//! written as `convert_qwen3` writes them, with the shared attention adapters folded in; a loop keeps its keys and its
//! (trained) gates.
//!
//! The file is streamed: one base matrix is read at a time, and each expert's slice of the stacked tensors is made,
//! encoded (quantized when `--type` asks for it, by the same block quantizers as `quantize`) and written before the
//! next. Memory is about one feed-forward matrix in f32 and the adapter; the disk is the output only (a 57B Q4_K is
//! about 32 GB instead of a 114 GB F16 first).
//!
//! A block the adapter has no router for (one below `--train-layers`) is written with a zero router and identical
//! experts: every choice gives the base's feed-forward, as the Workshop's dense block does. Without an adapter the
//! whole model is exported that way, which is exactly the dense base.
//!
//! The rotary layout: the engine runs `qwen*` architectures with NEOX rotation, which is Hugging Face's own layout, so
//! Llama and Mistral bases are written without llama.cpp's q/k permute and run as `qwen3moe` (without q/k norms).

use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context as _, Result};
use candle_core::{Device, Tensor};
use serde_json::Value as Json;
use strata_pack::gguf_write::{Kv, TensorSpec, Writer};

use crate::convert::{long_context_kv, loop_parts, qwen3_header, tensor_f32};
use crate::model::{moe_expert_stem, moe_router_name, ModelConfig, MoeSpec, Target, LOOP_FILE, MOE_TARGETS};
use crate::quantize_gguf::{encode, resolve, type_id, type_nbytes, PureType};
use crate::safetensors_io::{Shard, TensorDesc};

/// The precision the exported tensors are written in (1-D tensors and the routers always stay F32).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExportType {
    F32,
    Pure(PureType),
}

impl ExportType {
    pub fn from_s(s: &str) -> Result<Self> {
        if s.eq_ignore_ascii_case("f32") {
            return Ok(ExportType::F32);
        }
        PureType::from_s(s).map(ExportType::Pure).map_err(|_| anyhow!("--type {s:?}: choose f32, f16, q8_0, q6_k, q5_k, q4_k or q4_0"))
    }
    fn target(self) -> &'static str {
        match self {
            ExportType::F32 => "F32",
            ExportType::Pure(p) => p.ggml_name(),
        }
    }
    fn file_type(self) -> u32 {
        match self {
            ExportType::F32 => 0,
            ExportType::Pure(p) => p.file_type(),
        }
    }
}

/// What was written.
#[derive(Debug, Clone)]
pub struct ExportReport {
    pub out: PathBuf,
    pub moe: MoeSpec,
    pub total_params: u64,
    pub active_params: u64,
    /// blocks whose router and experts came from the adapter (the rest: identical experts)
    pub trained_blocks: usize,
    pub bytes: u64,
}

/// The adapter: its tensors, mapped, and its scaling.
struct Adapter {
    shard: Shard,
    file: PathBuf,
    scaling: f32,
}

impl Adapter {
    fn open(dir: &Path, moe: MoeSpec) -> Result<Adapter> {
        let cfg: Json = serde_json::from_str(&std::fs::read_to_string(dir.join("adapter_config.json"))
            .with_context(|| format!("{}: no adapter_config.json", dir.display()))?)?;
        let r = cfg["r"].as_f64().ok_or_else(|| anyhow!("adapter_config.json has no r"))?;
        let alpha = cfg["lora_alpha"].as_f64().ok_or_else(|| anyhow!("adapter_config.json has no lora_alpha"))?;
        let scaling = if cfg["use_rslora"].as_bool() == Some(true) { alpha / r.sqrt() } else { alpha / r } as f32;
        match MoeSpec::from_json(&cfg["prastara_moe"])? {
            Some(m) if m == moe => {}
            Some(m) => bail!("{}: the adapter was trained for {} experts top-{}, the model is {} top-{}", dir.display(), m.experts, m.top_k, moe.experts, moe.top_k),
            None => bail!("{}: not an upcycled model's adapter (no prastara_moe in adapter_config.json); train it on the upcycled folder", dir.display()),
        }
        let file = dir.join("adapter_model.safetensors");
        Ok(Adapter { shard: Shard::open(&file)?, file, scaling })
    }

    fn get(&self, name: &str) -> Result<Option<(Vec<usize>, Vec<f32>)>> {
        match self.shard.get(name) {
            Some(t) => Ok(Some((t.shape().to_vec(), tensor_f32(&self.shard, t, &self.file)?))),
            None => Ok(None),
        }
    }

    /// `w += s · B A` for the adapter at `stem` (`…lora_A.weight` / `…lora_B.weight`), when it has one.
    fn fold(&self, stem: &str, w: &mut [f32], out: usize, in_: usize) -> Result<bool> {
        let (Some((sa, a)), Some((sb, b))) = (self.get(&format!("{stem}.lora_A.weight"))?, self.get(&format!("{stem}.lora_B.weight"))?) else {
            return Ok(false);
        };
        let r = sa[0];
        if sa != [r, in_] || sb != [out, r] {
            bail!("{stem}: adapter shapes {sa:?} and {sb:?} do not fit a [{out}, {in_}] weight");
        }
        let delta = Tensor::from_vec(b, (out, r), &Device::Cpu)?.matmul(&Tensor::from_vec(a, (r, in_), &Device::Cpu)?)?;
        let delta = (delta * self.scaling as f64)?.flatten_all()?.to_vec1::<f32>()?;
        w.iter_mut().zip(delta).for_each(|(x, d)| *x += d);
        Ok(true)
    }
}

/// Where one output tensor's values come from.
enum Source {
    /// a base tensor as it is (norms, embeddings, the head, biases)
    Plain(usize, TensorDesc),
    /// a base attention matrix, with the shared adapter at this stem folded in
    Attn(usize, TensorDesc, String),
    /// a block's router: the adapter's, or zeros
    Router(usize),
    /// a block's stacked experts for gate (0), up (1) or down (2), from this base matrix
    Experts(usize, usize, usize, TensorDesc),
    /// a loop gate's values
    Values(Vec<f32>),
}

/// See the module documentation. `adapter`: the folder `train` wrote on the upcycled model (None: every expert is the
/// base feed-forward). `pre_tokenizer`: as `convert`'s, for a tokenizer whose hash is not a known one.
pub fn export_moe(dir: &Path, adapter: Option<&Path>, out: &Path, ty: ExportType, pre_tokenizer: Option<&str>) -> Result<ExportReport> {
    export_moe_with(dir, adapter, out, ty, pre_tokenizer, &mut |_, _, _| {})
}

/// [`export_moe`], calling `progress(bytes written, bytes in all, tensor)` after each tensor.
pub fn export_moe_with(dir: &Path, adapter: Option<&Path>, out: &Path, ty: ExportType, pre_tokenizer: Option<&str>,
                       progress: &mut dyn FnMut(u64, u64, &str)) -> Result<ExportReport> {
    let d = dir.display();
    let c: Json = serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).with_context(|| format!("reading config for {d}"))?)?;
    let cfg = ModelConfig::from_json(&c).with_context(|| format!("{d}"))?;
    let moe = cfg.moe.ok_or_else(|| anyhow!("{d} is not an upcycled model (no prastara_moe in config.json): run `am2strata-workshop upcycle` on it first"))?;
    // long context (the `context` copy, or a base with YaRN in its config): the keys convert_qwen3 writes
    let yarn = cfg.yarn.map(|y| (y, cfg.max_position_embeddings));
    if let Some((y, _)) = &yarn {
        if !y.truncate {
            bail!("{d}: rope_scaling yarn with truncate false: the engine rounds the ramp's ends as llama.cpp does");
        }
    }
    let (h, ff, layers, e_n) = (cfg.hidden_size, cfg.intermediate_size, cfg.num_hidden_layers, moe.experts);
    let arch = "qwen3moe";
    let ad = adapter.map(|a| Adapter::open(a, moe)).transpose()?;

    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)?.filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("safetensors") && p.file_name().and_then(|s| s.to_str()) != Some(LOOP_FILE)).collect();
    files.sort();
    anyhow::ensure!(!files.is_empty(), "{d}: no *.safetensors shards found");
    let shards: Vec<Shard> = files.iter().map(|f| Shard::open(f)).collect::<Result<_>>()?;
    let find = |name: &str| shards.iter().enumerate().find_map(|(i, s)| s.get(name).cloned().map(|t| (i, t)));
    let need = |name: &str| find(name).ok_or_else(|| anyhow!("{d}: tensor {name} is missing"));
    let prefix = if find("model.embed_tokens.weight").is_some() { "model." } else { "" };

    let mut plan: Vec<(String, Vec<u64>, Source)> = Vec::new();
    let plain = |g: &str, (i, t): (usize, TensorDesc)| (g.to_string(), t.shape().iter().map(|&x| x as u64).collect::<Vec<_>>(), Source::Plain(i, t));
    plan.push(plain("token_embd.weight", need(&format!("{prefix}embed_tokens.weight"))?));
    plan.push(plain("output_norm.weight", need(&format!("{prefix}norm.weight"))?));
    if let (Some(t), false) = (find("lm_head.weight"), cfg.tie_word_embeddings) {
        plan.push(plain("output.weight", t));
    }
    let mut trained = 0;
    for b in 0..layers {
        let hf = |k: &str| format!("{prefix}layers.{b}.{k}");
        plan.push(plain(&format!("blk.{b}.attn_norm.weight"), need(&hf("input_layernorm.weight"))?));
        for (g, t) in [("attn_q", Target::Q), ("attn_k", Target::K), ("attn_v", Target::V), ("attn_output", Target::O)] {
            let (i, desc) = need(&hf(&format!("self_attn.{}.weight", t.as_str())))?;
            let stem = format!("base_model.model.model.layers.{b}.self_attn.{}", t.as_str());
            plan.push((format!("blk.{b}.{g}.weight"), desc.shape().iter().map(|&x| x as u64).collect(), Source::Attn(i, desc, stem)));
            if let Some(bias) = find(&hf(&format!("self_attn.{}.bias", t.as_str()))) {
                plan.push(plain(&format!("blk.{b}.{g}.bias"), bias));
            }
        }
        for (g, k) in [("attn_q_norm", "self_attn.q_norm.weight"), ("attn_k_norm", "self_attn.k_norm.weight")] {
            if let Some(t) = find(&hf(k)) {
                plan.push(plain(&format!("blk.{b}.{g}.weight"), t));
            }
        }
        plan.push(plain(&format!("blk.{b}.ffn_norm.weight"), need(&hf("post_attention_layernorm.weight"))?));
        if ad.as_ref().is_some_and(|a| a.shard.get(&moe_router_name(b)).is_some()) {
            trained += 1;
        }
        plan.push((format!("blk.{b}.ffn_gate_inp.weight"), vec![e_n as u64, h as u64], Source::Router(b)));
        for (role, (g, k)) in [("ffn_gate_exps", "mlp.gate_proj"), ("ffn_up_exps", "mlp.up_proj"), ("ffn_down_exps", "mlp.down_proj")].into_iter().enumerate() {
            let (i, desc) = need(&hf(&format!("{k}.weight")))?;
            let shape = if role == 2 { vec![e_n as u64, h as u64, ff as u64] } else { vec![e_n as u64, ff as u64, h as u64] };
            if desc.shape() != [shape[1] as usize, shape[2] as usize] {
                bail!("{d}: {k} of block {b} has shape {:?}, the config says {:?}", desc.shape(), &shape[1..]);
            }
            plan.push((format!("blk.{b}.{g}.weight"), shape, Source::Experts(b, role, i, desc)));
        }
    }
    // the loop: its keys, and its gates (the adapter's trained ones over loop.safetensors)
    let lp = loop_parts(dir, arch, h, layers)?;
    let (loop_kv, gates) = lp.unwrap_or_default();
    for (name, vals) in gates {
        let trained_gate = match &ad {
            Some(a) => a.get(&format!("base_model.model.model.{name}"))?.map(|(_, v)| v),
            None => None,
        };
        let vals = trained_gate.unwrap_or(vals);
        anyhow::ensure!(vals.len() == h, "{name} has {} values, the hidden size is {h}", vals.len());
        plan.push((name, vec![h as u64], Source::Values(vals)));
    }
    // routers that the plan does not expect (another model's adapter) are refused before anything is written
    if let Some(a) = &ad {
        for n in a.shard.names() {
            if n.contains(".mlp.gate.weight") && !(0..layers).any(|b| moe_router_name(b) == n) {
                bail!("the adapter's {n} is not a router of this model's {layers} blocks");
            }
        }
    }

    let (total, active) = moe.params(&cfg);
    let mut kv = qwen3_header(dir, &c, arch, total, ty.file_type(), pre_tokenizer)?;
    kv.push((format!("{arch}.expert_count"), Kv::U32(e_n as u32)));
    kv.push((format!("{arch}.expert_used_count"), Kv::U32(moe.top_k as u32)));
    kv.push((format!("{arch}.expert_feed_forward_length"), Kv::U32(ff as u32)));
    kv.extend(loop_kv);
    kv.extend(long_context_kv(arch, yarn, &cfg.layer_sliding, cfg.sliding_window));
    // the config's max_position_embeddings is the stretched length already (convert_qwen3 does the same)

    // each tensor's type: 1-D and the routers F32 (llama.cpp never quantizes ffn_gate_inp), the rest the target,
    // demoted by its row length as `llama-quantize --pure` does
    let type_of = |name: &str, shape: &[u64]| -> &'static str {
        if shape.len() <= 1 || name.ends_with("ffn_gate_inp.weight") { "F32" } else { resolve(ty.target(), *shape.last().unwrap() as usize) }
    };
    let specs: Vec<TensorSpec> = plan.iter().map(|(g, shape, _)| {
        let t = type_of(g, shape);
        TensorSpec { name: g.clone(), shape: shape.clone(), ggml_type: type_id(t), nbytes: type_nbytes(t, shape.iter().product::<u64>() as usize) }
    }).collect();
    let kv_ref: Vec<(&str, Kv)> = kv.iter().map(|(k, v)| (k.as_str(), v.clone())).collect();
    if let Some(p) = out.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(p)?;
    }
    let partial = out.with_extension("gguf.part");
    let all_bytes: u64 = specs.iter().map(|t| t.nbytes).sum();
    let sizes: Vec<u64> = specs.iter().map(|t| t.nbytes).collect();
    let mut done_bytes = 0u64;
    let mut w = Writer::create_typed(&partial, &kv_ref, specs)?;
    let base = |i: usize, t: &TensorDesc| tensor_f32(&shards[i], t, &files[i]);
    for (k, (g, shape, src)) in plan.iter().enumerate() {
        let t = type_of(g, shape);
        match src {
            Source::Plain(i, desc) => w.tensor_data(&encode_par(&base(*i, desc)?, t, *desc.shape().last().unwrap_or(&0)))?,
            Source::Attn(i, desc, stem) => {
                let mut v = base(*i, desc)?;
                if let Some(a) = &ad {
                    a.fold(stem, &mut v, desc.shape()[0], desc.shape()[1])?;
                }
                w.tensor_data(&encode_par(&v, t, desc.shape()[1]))?;
            }
            Source::Router(b) => {
                let v = match &ad {
                    Some(a) => match a.get(&moe_router_name(*b))? {
                        Some((s, v)) if s == [e_n, h] => v,
                        Some((s, _)) => bail!("{}: shape {s:?}, the model has {e_n} experts of hidden size {h}", moe_router_name(*b)),
                        None => vec![0.0; e_n * h],
                    },
                    None => vec![0.0; e_n * h],
                };
                w.tensor_data(&encode(&v, t))?;
            }
            Source::Experts(b, role, i, desc) => {
                let w0 = base(*i, desc)?;
                let (o, inn) = (desc.shape()[0], desc.shape()[1]);
                let has_router = ad.as_ref().is_some_and(|a| a.shard.get(&moe_router_name(*b)).is_some());
                for e in 0..e_n {
                    let mut v = w0.clone();
                    if let (Some(a), true) = (&ad, has_router) {
                        if !a.fold(&moe_expert_stem(*b, e, MOE_TARGETS[*role]), &mut v, o, inn)? {
                            bail!("block {b} has a router in the adapter but expert {e} has no {} adapter", MOE_TARGETS[*role].as_str());
                        }
                    }
                    w.tensor_data(&encode_par(&v, t, inn))?;
                }
            }
            Source::Values(v) => w.tensor_data(&encode(v, t))?,
        }
        w.end_tensor()?;
        done_bytes += sizes[k];
        progress(done_bytes, all_bytes, g);
    }
    w.finish()?;
    std::fs::rename(&partial, out).with_context(|| format!("moving the finished file to {}", out.display()))?;
    let bytes = std::fs::metadata(out).map(|m| m.len()).unwrap_or(0);
    Ok(ExportReport { out: out.to_path_buf(), moe, total_params: total, active_params: active, trained_blocks: trained, bytes })
}

/// [`encode`] with the rows split over the CPUs: block quantizers work row by row (a row's blocks never depend on
/// another row), so the bytes are the same as one thread's. `cols`: the row length. Small tensors stay on one thread.
pub fn encode_par(v: &[f32], ty: &str, cols: usize) -> Vec<u8> {
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get()).min(8);
    let rows = if cols == 0 { 0 } else { v.len() / cols };
    if threads <= 1 || v.len() < (1 << 22) || cols == 0 || rows * cols != v.len() || ty == "F32" {
        return encode(v, ty);
    }
    let per = rows.div_ceil(threads) * cols;
    let parts: Vec<Vec<u8>> = std::thread::scope(|sc| {
        let hs: Vec<_> = v.chunks(per).map(|c| sc.spawn(move || encode(c, ty))).collect();
        hs.into_iter().map(|h| h.join().expect("an encoding thread")).collect()
    });
    parts.concat()
}

/// What an export needs, before it runs: the file, and the memory (one base matrix as f32, one expert's copy of it,
/// their encoded bytes, the adapter's largest pair widened, and the threads' share).
pub fn export_needs(cfg: &ModelConfig, ty: ExportType, adapter_bytes: u64) -> Result<(u64, u64)> {
    let disk = export_bytes(cfg, ty)?;
    let mat = (cfg.intermediate_size.max(cfg.vocab_size) * cfg.hidden_size) as u64 * 4;
    Ok((disk, 3 * mat + adapter_bytes + (256 << 20)))
}

/// The exported file's size for `cfg` (an upcycled config) at `ty`, before writing it: what the UI and the disk check
/// show. Counts each tensor as `export_moe` writes it (metadata excluded: a few MB at most).
pub fn export_bytes(cfg: &ModelConfig, ty: ExportType) -> Result<u64> {
    let moe = cfg.moe.ok_or_else(|| anyhow!("not an upcycled config"))?;
    let (h, ff, e) = (cfg.hidden_size, cfg.intermediate_size, moe.experts);
    let mat = |rows: usize, cols: usize| type_nbytes(resolve(ty.target(), cols), rows * cols);
    let (q, kv) = (cfg.num_attention_heads * cfg.head_dim, cfg.num_key_value_heads * cfg.head_dim);
    let block = mat(q, h) + 2 * mat(kv, h) + mat(h, q) + (e * h * 4) as u64 + 2 * mat(e * ff, h) + mat(e * h, ff) + cfg.block_small_params() * 4;
    let emb = mat(cfg.vocab_size, h);
    let loop_gates = cfg.loop_.map_or(0, |l| (l.passes - 1) * h * 4) as u64;
    Ok(emb * if cfg.tie_word_embeddings { 1 } else { 2 } + cfg.num_hidden_layers as u64 * block + h as u64 * 4 + loop_gates)
}
