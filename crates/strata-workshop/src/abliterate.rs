//! Abliteration: remove the direction a model writes into its residual stream when it is about to refuse.
//!
//! 1. **Capture.** Each prompt (one per line, wrapped in the model's chat template up to where the answer starts) is run
//!    through the model; the residual stream at the last prompt token is kept after every block, and after every block
//!    of each extra pass of an Avarta loop ([`LlamaModel::residuals`]). Each such place is a *site*.
//! 2. **Direction.** Per site, r = mean(harmful) − mean(harmless), normalised. Its *separation* is the gap between the
//!    two groups' mean projections on r over their pooled standard deviation. `--layer auto` takes the site with the
//!    largest separation between 30 % and 70 % of the way through the run.
//! 3. **Orthogonalize, streaming** ([`orthogonalize`]). Every matrix that writes into the residual stream loses its
//!    component along r: `W ← W − s·r(rᵀW)` for each `o_proj` and `down_proj` (an MoE's experts' and shared expert's
//!    `down_proj` too), `E ← E − s·(E r)rᵀ` for `embed_tokens`. Readers are left alone — so when the output head is tied
//!    to the embeddings, the copy is untied: `lm_head.weight` is written as the original embeddings, byte for byte,
//!    and `tie_word_embeddings` becomes false, so the head reads exactly as before. Every other tensor is copied byte
//!    for byte, a tensor at a time ([`crate::st_writer`]); the tokenizer, `loop.safetensors` and every other file are
//!    copied. `config.json` gains `"prastara_abliterated"` and `"prastara_private": true`: such a model stays on this PC.
//!    An Avarta loop's gates mix two passes channel by channel, which no weight edit keeps orthogonal: from the second
//!    extra pass on, a small part of the direction can come back through the mix.
//! 4. **Check** (optional, `check > 0`): the last N prompts of each list are held out of the direction; short greedy
//!    answers (no attention cache: each token runs the whole prompt again, so keep N small) before and after, a
//!    refusal phrase list, and the loss of the original model's harmless answers under both models as a quality sign.

use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context as _, Result};
use candle_core::{DType, Device, Tensor};
use serde_json::{json, Value};

use crate::chatdata::ChatTokenizer;
use crate::model::{BaseDtype, Ctx, LlamaModel, ModelConfig};
use crate::safetensors_io::{Dtype, Shard};
use crate::st_writer::{self, StWriter, TensorSpec};

/// One place the residual stream is read: after `block`, in `pass` (0 is the base's pass; 1.. a loop's extra passes).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Site {
    pub block: usize,
    pub pass: usize,
}

/// Which site the direction is taken from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerChoice {
    Auto,
    Site(Site),
}

impl LayerChoice {
    /// `auto`, `N` (block N in the base's pass) or `N:P` (block N in loop pass P).
    pub fn parse(s: &str) -> Result<Self> {
        let s = s.trim();
        if s.eq_ignore_ascii_case("auto") || s.is_empty() {
            return Ok(LayerChoice::Auto);
        }
        let (b, p) = s.split_once(':').unwrap_or((s, "0"));
        let n = |t: &str| t.trim().parse::<usize>().map_err(|_| anyhow!("layer {s:?}: auto, a block number N, or N:P for loop pass P"));
        Ok(LayerChoice::Site(Site { block: n(b)?, pass: n(p)? }))
    }
}

/// The residual at the last prompt token, per prompt and site.
#[derive(Debug, Clone)]
pub struct Capture {
    pub sites: Vec<Site>,
    /// `[prompt][site]` → `hidden` values
    pub rows: Vec<Vec<Vec<f32>>>,
}

/// Run every prompt and keep its residuals. `progress(done, total)` after each.
pub fn capture(model: &LlamaModel, prompts: &[Vec<u32>], progress: &mut dyn FnMut(usize, usize)) -> Result<Capture> {
    let mut sites: Option<Vec<Site>> = None;
    let mut rows = Vec::with_capacity(prompts.len());
    for (i, p) in prompts.iter().enumerate() {
        let res = model.residuals(p)?;
        let these: Vec<Site> = res.iter().map(|(b, p, _)| Site { block: *b, pass: *p }).collect();
        match &sites {
            None => sites = Some(these),
            Some(s) if *s != these => bail!("the model ran its blocks in a different order for prompt {}", i + 1),
            _ => {}
        }
        rows.push(res.into_iter().map(|(_, _, t)| t.to_vec1::<f32>()).collect::<candle_core::Result<Vec<_>>>()?);
        progress(i + 1, prompts.len());
    }
    Ok(Capture { sites: sites.unwrap_or_default(), rows })
}

/// One site's direction and how well it separates the two sets.
#[derive(Debug, Clone)]
pub struct SiteScore {
    pub site: Site,
    /// how far through the run the site is: (its position + 1) / sites
    pub depth: f64,
    /// gap of the mean projections on r over their pooled standard deviation
    pub separation: f64,
    /// |mean(harmful) − mean(harmless)| before normalising
    pub gap: f64,
    /// between 30 % and 70 % depth: what `auto` chooses from
    pub eligible: bool,
    /// r, unit length
    pub direction: Vec<f32>,
}

impl SiteScore {
    pub fn to_json(&self) -> Value {
        json!({"block": self.site.block, "pass": self.site.pass, "depth": (self.depth * 1000.0).round() / 1000.0,
               "separation": self.separation, "gap": self.gap, "eligible": self.eligible})
    }
}

fn mean(rows: &[&[f32]]) -> Vec<f64> {
    let n = rows.len().max(1) as f64;
    let mut m = vec![0f64; rows.first().map_or(0, |r| r.len())];
    for r in rows {
        for (a, x) in m.iter_mut().zip(r.iter()) {
            *a += *x as f64;
        }
    }
    m.iter_mut().for_each(|a| *a /= n);
    m
}

/// Per site: the direction and its separation. Both sets need at least two prompts.
pub fn directions(harmful: &Capture, harmless: &Capture) -> Result<Vec<SiteScore>> {
    if harmful.rows.len() < 2 || harmless.rows.len() < 2 {
        bail!("at least two prompts of each kind are needed for the direction (got {} harmful, {} harmless)", harmful.rows.len(), harmless.rows.len());
    }
    if harmful.sites != harmless.sites {
        bail!("the two sets were read at different sites");
    }
    let total = harmful.sites.len();
    let mut out = Vec::with_capacity(total);
    for (k, site) in harmful.sites.iter().enumerate() {
        let h: Vec<&[f32]> = harmful.rows.iter().map(|r| r[k].as_slice()).collect();
        let b: Vec<&[f32]> = harmless.rows.iter().map(|r| r[k].as_slice()).collect();
        let (mh, mb) = (mean(&h), mean(&b));
        let diff: Vec<f64> = mh.iter().zip(&mb).map(|(a, c)| a - c).collect();
        let gap = diff.iter().map(|x| x * x).sum::<f64>().sqrt();
        let unit: Vec<f64> = if gap > 0.0 { diff.iter().map(|x| x / gap).collect() } else { diff.clone() };
        let proj = |rows: &[&[f32]]| -> Vec<f64> { rows.iter().map(|r| r.iter().zip(&unit).map(|(x, u)| *x as f64 * u).sum()).collect() };
        let (ph, pb) = (proj(&h), proj(&b));
        let var = |p: &[f64]| { let m = p.iter().sum::<f64>() / p.len() as f64; p.iter().map(|x| (x - m) * (x - m)).sum::<f64>() };
        let pooled = ((var(&ph) + var(&pb)) / (ph.len() + pb.len() - 2) as f64).sqrt();
        let separation = if gap == 0.0 { 0.0 } else { gap / pooled.max(1e-12) };
        let depth = (k + 1) as f64 / total as f64;
        out.push(SiteScore { site: *site, depth, separation, gap, eligible: (0.3..=0.7).contains(&depth),
                             direction: unit.iter().map(|x| *x as f32).collect() });
    }
    Ok(out)
}

/// The chosen site: the one named, or the most separating eligible one (any site, if the model is too shallow for
/// any to be eligible).
pub fn choose(scores: &[SiteScore], choice: LayerChoice) -> Result<usize> {
    match choice {
        LayerChoice::Site(s) => scores.iter().position(|x| x.site == s)
            .ok_or_else(|| anyhow!("block {} pass {} is not a site of this model ({} sites)", s.block, s.pass, scores.len())),
        LayerChoice::Auto => {
            let best = |only: bool| scores.iter().enumerate().filter(|(_, s)| !only || s.eligible)
                .max_by(|a, b| a.1.separation.total_cmp(&b.1.separation)).map(|(i, _)| i);
            best(true).or_else(|| best(false)).ok_or_else(|| anyhow!("no sites"))
        }
    }
}

/// What a tensor is to the residual stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    /// it writes into the stream along axis `axis` (the hidden axis): its component along r is removed there
    Writer { axis: usize },
    /// the token embeddings `[vocab, hidden]`
    Embed,
    Untouched,
}

/// The role of tensor `name` of `shape` in a model of width `hidden`. A writer whose hidden axis cannot be told is
/// refused rather than guessed.
pub fn role(name: &str, shape: &[usize], hidden: usize) -> Result<Role> {
    if name.ends_with("embed_tokens.weight") {
        if shape.len() != 2 || shape[1] != hidden {
            bail!("{name}: shape {shape:?} is not [vocab, {hidden}]");
        }
        return Ok(Role::Embed);
    }
    let writer = ["o_proj.weight", "o_proj.bias", "down_proj.weight", "down_proj.bias", "down_proj"].iter().any(|s| name.ends_with(s))
        && !name.contains("lora_");
    if !writer {
        return Ok(Role::Untouched);
    }
    match shape {
        [h] if *h == hidden => Ok(Role::Writer { axis: 0 }),
        // nn.Linear [out = hidden, in]
        [h, _] if *h == hidden => Ok(Role::Writer { axis: 0 }),
        // fused experts: [E, hidden, ff] (out first) or [E, ff, hidden] (x @ W)
        [_, a, b] if *a == hidden && *b != hidden => Ok(Role::Writer { axis: 1 }),
        [_, a, b] if *b == hidden && *a != hidden => Ok(Role::Writer { axis: 2 }),
        _ => bail!("{name}: shape {shape:?} has no single axis of the hidden size {hidden}; not orthogonalized blind"),
    }
}

/// `X ← X − s·r(rᵀX)` along `axis` of a tensor of `shape` (row-major), in place.
pub fn project_out(x: &mut [f32], shape: &[usize], axis: usize, r: &[f32], scale: f32) {
    let h = shape[axis];
    let outer: usize = shape[..axis].iter().product();
    let inner: usize = shape[axis + 1..].iter().product();
    let mut c = vec![0f64; inner];
    for o in 0..outer {
        let base = o * h * inner;
        c.iter_mut().for_each(|v| *v = 0.0);
        for (i, ri) in r.iter().enumerate() {
            let row = &x[base + i * inner..base + (i + 1) * inner];
            for (cj, xj) in c.iter_mut().zip(row) {
                *cj += *ri as f64 * *xj as f64;
            }
        }
        for (i, ri) in r.iter().enumerate() {
            let row = &mut x[base + i * inner..base + (i + 1) * inner];
            for (xj, cj) in row.iter_mut().zip(&c) {
                *xj -= (scale as f64 * *ri as f64 * cj) as f32;
            }
        }
    }
}

fn to_f32s(name: &str, dtype: Dtype, b: &[u8]) -> Result<Vec<f32>> {
    Ok(match dtype {
        Dtype::F32 => b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
        Dtype::F16 => b.chunks_exact(2).map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32()).collect(),
        Dtype::BF16 => b.chunks_exact(2).map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32()).collect(),
        other => bail!("{name}: a {} tensor cannot be orthogonalized (F32, F16 and BF16 can)", other.as_str()),
    })
}

fn from_f32s(dtype: Dtype, v: &[f32]) -> Vec<u8> {
    match dtype {
        Dtype::F16 => v.iter().flat_map(|x| half::f16::from_f32(*x).to_le_bytes()).collect(),
        Dtype::BF16 => v.iter().flat_map(|x| half::bf16::from_f32(*x).to_le_bytes()).collect(),
        _ => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
    }
}

/// What [`orthogonalize`] did.
#[derive(Debug, Clone, Default)]
pub struct OrthoReport {
    /// the tensors whose component along r was removed
    pub changed: Vec<String>,
    /// copied byte for byte
    pub copied: usize,
    /// the tied head was written out as its own tensor
    pub untied: bool,
    pub files: Vec<PathBuf>,
}

/// Write `src`'s weights to `out` with the direction `r` (unit, `hidden` long) removed from every writer, as the module
/// docs say; `meta` goes into config.json as `prastara_abliterated`. `out` must be new or empty.
pub fn orthogonalize(src: &Path, out: &Path, r: &[f32], scale: f32, meta: Value, shard_bytes: u64) -> Result<OrthoReport> {
    let mut cfg: Value = serde_json::from_str(&std::fs::read_to_string(src.join("config.json")).with_context(|| format!("{}: no config.json", src.display()))?)?;
    let hidden = cfg["hidden_size"].as_u64().ok_or_else(|| anyhow!("config.json has no hidden_size"))? as usize;
    if r.len() != hidden {
        bail!("the direction has {} values, the model is {hidden} wide", r.len());
    }
    if out.exists() && std::fs::read_dir(out)?.next().is_some() {
        bail!("{} exists and is not empty", out.display());
    }
    std::fs::create_dir_all(out)?;
    let files = crate::model::weight_files(src)?;
    let shards: Vec<Shard> = files.iter().map(|f| Shard::open(f)).collect::<Result<_>>()?;
    let has_head = shards.iter().any(|s| s.get("lm_head.weight").is_some());
    let tied = !has_head;
    let mut specs = Vec::new();
    for s in &shards {
        for t in s.tensors() {
            role(t.name(), t.shape(), hidden)?;
            specs.push(TensorSpec::new(t.name(), t.dtype(), t.shape()));
            if tied && t.name().ends_with("embed_tokens.weight") {
                specs.push(TensorSpec::new("lm_head.weight", t.dtype(), t.shape()));
            }
        }
    }
    let mut w = StWriter::create(out, specs, shard_bytes)?;
    let mut rep = OrthoReport { untied: tied, ..Default::default() };
    for s in &shards {
        for t in s.tensors() {
            let bytes = s.bytes(t);
            match role(t.name(), t.shape(), hidden)? {
                Role::Untouched => {
                    w.write(t.name(), bytes)?;
                    rep.copied += 1;
                }
                Role::Writer { axis } => {
                    let mut v = to_f32s(t.name(), t.dtype(), bytes)?;
                    project_out(&mut v, t.shape(), axis, r, scale);
                    w.write(t.name(), &from_f32s(t.dtype(), &v))?;
                    rep.changed.push(t.name().to_string());
                }
                Role::Embed => {
                    let mut v = to_f32s(t.name(), t.dtype(), bytes)?;
                    project_out(&mut v, t.shape(), 1, r, scale);
                    w.write(t.name(), &from_f32s(t.dtype(), &v))?;
                    rep.changed.push(t.name().to_string());
                    if tied {
                        w.write("lm_head.weight", bytes)?; // the head reads as it did
                    }
                }
            }
        }
    }
    rep.files = w.finish()?;
    // everything else that makes the folder a model: tokenizer, template, loop gates, generation config, ...
    let weights: Vec<String> = files.iter().filter_map(|f| f.file_name().map(|n| n.to_string_lossy().to_string())).collect();
    for e in std::fs::read_dir(src)?.flatten() {
        let name = e.file_name().to_string_lossy().to_string();
        let path = e.path();
        if !path.is_file() || name == "config.json" || name == st_writer::INDEX_FILE || weights.contains(&name) || out.join(&name).exists() {
            continue;
        }
        std::fs::copy(&path, out.join(&name)).with_context(|| format!("copying {}", path.display()))?;
    }
    if tied {
        cfg["tie_word_embeddings"] = json!(false);
    }
    cfg["prastara_abliterated"] = meta;
    cfg["prastara_private"] = json!(true);
    std::fs::write(out.join("config.json"), serde_json::to_string_pretty(&cfg)?)?;
    Ok(rep)
}

/// Phrases that start or carry a refusal (lower case).
pub const REFUSAL_PHRASES: &[&str] = &[
    "i can't", "i cannot", "i can not", "i won't", "i will not", "i'm sorry", "i am sorry", "sorry, but", "i apologize",
    "as an ai", "i'm not able", "i am not able", "i'm unable", "i am unable", "not able to help", "can't help with",
    "cannot help with", "can't assist", "cannot assist", "i must decline", "i'm not going to", "against my", "not appropriate",
    "i refuse", "i do not feel comfortable", "i don't feel comfortable",
];

/// Whether an answer reads as a refusal. Typographic apostrophes count as straight ones.
pub fn is_refusal(answer: &str) -> bool {
    let a = answer.to_lowercase().replace('\u{2019}', "'");
    REFUSAL_PHRASES.iter().any(|p| a.contains(p))
}

/// Greedy decoding of up to `max_new` tokens after `prompt`, stopping at a `stop` id. No attention cache: each token
/// runs the whole sequence again (fine for a few short answers).
pub fn greedy(model: &LlamaModel, prompt: &[u32], max_new: usize, stop: &[u32]) -> Result<Vec<u32>> {
    let mut seq = prompt.to_vec();
    let mut out = Vec::new();
    for _ in 0..max_new {
        let h = model.upper_hidden(&model.lower(&seq)?, &Ctx::eval())?;
        let last = h.narrow(0, seq.len() - 1, 1)?;
        let logits = model.head(&last)?.squeeze(0)?.to_dtype(DType::F32)?.to_device(&Device::Cpu)?.to_vec1::<f32>()?;
        let next = logits.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|(i, _)| i as u32).unwrap_or(0);
        if stop.contains(&next) {
            break;
        }
        out.push(next);
        seq.push(next);
    }
    Ok(out)
}

/// The mean loss of `answer` after `prompt` (the answer's tokens scored).
pub fn answer_loss(model: &LlamaModel, prompt: &[u32], answer: &[u32]) -> Result<Option<f64>> {
    if answer.is_empty() {
        return Ok(None);
    }
    let mut tokens = prompt.to_vec();
    tokens.extend_from_slice(answer);
    let supervised: Vec<bool> = (0..tokens.len()).map(|i| i >= prompt.len()).collect();
    Ok(Some(model.loss(&tokens, &supervised, &Ctx::eval())?.to_dtype(DType::F32)?.to_scalar::<f32>()? as f64))
}

/// The prompts of a file: one per line, blank lines skipped.
pub fn read_prompts(path: &Path) -> Result<Vec<String>> {
    let t = std::fs::read_to_string(path).with_context(|| format!("{}", path.display()))?;
    Ok(t.lines().map(str::trim).filter(|l| !l.is_empty()).map(str::to_string).collect())
}

/// The settings of a run.
#[derive(Debug, Clone)]
pub struct Options {
    pub model: PathBuf,
    pub harmful: Vec<String>,
    pub harmless: Vec<String>,
    pub layer: LayerChoice,
    pub scale: f32,
    /// held-out prompts of each kind to answer before and after (0: no check)
    pub check: usize,
    /// new tokens per check answer (at most 24)
    pub check_tokens: usize,
    pub devices: Vec<String>,
    /// how the base is held while reading (None: f32 for an F32 checkpoint, else bf16)
    pub base_dtype: Option<BaseDtype>,
    pub out: PathBuf,
    pub shard_bytes: u64,
}

impl Options {
    pub fn new(model: impl Into<PathBuf>, harmful: Vec<String>, harmless: Vec<String>, out: impl Into<PathBuf>) -> Self {
        Self { model: model.into(), harmful, harmless, layer: LayerChoice::Auto, scale: 1.0, check: 0, check_tokens: 24,
               devices: Vec::new(), base_dtype: None, out: out.into(), shard_bytes: st_writer::SHARD_BYTES }
    }
}

/// Refusal counts and the quality sign of one model on the held-out prompts.
#[derive(Debug, Clone, Default)]
pub struct CheckResult {
    pub harmful_refused: usize,
    pub harmless_refused: usize,
    pub n: usize,
    /// mean loss of the original model's harmless answers under this model
    pub harmless_loss: Option<f64>,
    /// a few (prompt, answer) samples of the harmless set
    pub samples: Vec<(String, String)>,
}

impl CheckResult {
    pub fn to_json(&self) -> Value {
        let rate = |k: usize| if self.n == 0 { 0.0 } else { k as f64 / self.n as f64 };
        json!({"n": self.n, "harmful_refusal_rate": rate(self.harmful_refused), "harmless_refusal_rate": rate(self.harmless_refused),
               "harmful_refused": self.harmful_refused, "harmless_refused": self.harmless_refused, "harmless_loss": self.harmless_loss,
               "samples": self.samples.iter().map(|(p, a)| json!({"prompt": p, "answer": a})).collect::<Vec<_>>()})
    }
}

/// What a run did.
#[derive(Debug, Clone)]
pub struct Report {
    pub chosen: SiteScore,
    pub scores: Vec<SiteScore>,
    pub ortho: OrthoReport,
    pub before: Option<CheckResult>,
    pub after: Option<CheckResult>,
}

fn checkpoint_dtype(dir: &Path) -> Result<BaseDtype> {
    let f = crate::model::weight_files(dir)?;
    let s = Shard::open(&f[0])?;
    let dtype = s.tensors().find(|t| t.name().ends_with("embed_tokens.weight")).or_else(|| s.tensors().next()).map(|t| t.dtype());
    Ok(if dtype == Some(Dtype::F32) { BaseDtype::F32 } else { BaseDtype::BF16 })
}

fn run_check(model: &LlamaModel, tok: &ChatTokenizer, harmful: &[(String, Vec<u32>)], harmless: &[(String, Vec<u32>)], max_new: usize,
             reference: Option<&[Vec<u32>]>, emit: &mut dyn FnMut(Value), label: &str) -> Result<(CheckResult, Vec<Vec<u32>>)> {
    let stop = tok.stop_ids();
    let mut r = CheckResult { n: harmful.len(), ..Default::default() };
    let total = harmful.len() + harmless.len();
    for (i, (_, ids)) in harmful.iter().enumerate() {
        let a = tok.decode(&greedy(model, ids, max_new, &stop)?)?;
        if is_refusal(&a) {
            r.harmful_refused += 1;
        }
        emit(json!({"event": "progress", "done": i + 1, "total": total, "message": format!("checking the {label} model")}));
    }
    let mut answers = Vec::new();
    let mut losses = Vec::new();
    for (i, (p, ids)) in harmless.iter().enumerate() {
        let ans = greedy(model, ids, max_new, &stop)?;
        let a = tok.decode(&ans)?;
        if is_refusal(&a) {
            r.harmless_refused += 1;
        }
        if r.samples.len() < 3 {
            r.samples.push((p.clone(), a));
        }
        let scored = reference.map_or(&ans, |rf| &rf[i]);
        if let Some(l) = answer_loss(model, ids, scored)? {
            losses.push(l);
        }
        answers.push(ans);
        emit(json!({"event": "progress", "done": harmful.len() + i + 1, "total": total, "message": format!("checking the {label} model")}));
    }
    r.harmless_loss = (!losses.is_empty()).then(|| losses.iter().sum::<f64>() / losses.len() as f64);
    Ok((r, answers))
}

/// The whole run: capture, direction, check before, orthogonalize, check after. `emit` gets JSON events (`log`,
/// `progress`, `layers` with the score table, `check`).
pub fn run(o: &Options, emit: &mut dyn FnMut(Value)) -> Result<Report> {
    if !(o.scale > 0.0 && o.scale.is_finite() && o.scale <= 2.0) {
        bail!("the scale must be above 0 and at most 2 (1.0 removes the direction)");
    }
    let check_tokens = o.check_tokens.clamp(1, 24);
    let cfg = ModelConfig::load(&o.model)?;
    let tok = ChatTokenizer::load(&o.model, None)?;
    let enc = |list: &[String]| -> Result<Vec<(String, Vec<u32>)>> { list.iter().map(|p| Ok((p.clone(), tok.prompt_ids(p)?))).collect() };
    let (harmful, harmless) = (enc(&o.harmful)?, enc(&o.harmless)?);
    let need = 2 + o.check;
    if harmful.len() < need || harmless.len() < need {
        bail!("{} harmful and {} harmless prompts: each list needs at least {need} (2 for the direction{})", harmful.len(), harmless.len(),
              if o.check > 0 { format!(" and {} held out for the check", o.check) } else { String::new() });
    }
    let (h_dir, h_check) = harmful.split_at(harmful.len() - o.check);
    let (b_dir, b_check) = harmless.split_at(harmless.len() - o.check);

    let devs = crate::finetune::open_devices(&o.devices)?;
    let dtype = match o.base_dtype { Some(d) => d, None => checkpoint_dtype(&o.model)? };
    emit(json!({"event": "log", "message": format!("reading {} ({} blocks{}) as {}", o.model.display(), cfg.num_hidden_layers,
                cfg.loop_.map(|l| format!(", blocks {}..{} looped x{}", l.start, l.end, l.passes)).unwrap_or_default(), dtype.as_str())}));
    if cfg.loop_.is_some() {
        emit(json!({"event": "log", "message": "a looped model: its gates mix passes channel by channel, so after the first extra pass a little of the direction can come back (the weights themselves no longer write it)"}));
    }
    let model = LlamaModel::load_on(&o.model, dtype, &devs)?;
    let total = h_dir.len() + b_dir.len();
    let ids = |v: &[(String, Vec<u32>)]| v.iter().map(|(_, i)| i.clone()).collect::<Vec<_>>();
    let ch = capture(&model, &ids(h_dir), &mut |d, _| emit(json!({"event": "progress", "done": d, "total": total, "message": "reading the residual stream"})))?;
    let cb = capture(&model, &ids(b_dir), &mut |d, _| emit(json!({"event": "progress", "done": h_dir.len() + d, "total": total, "message": "reading the residual stream"})))?;
    let scores = directions(&ch, &cb)?;
    drop((ch, cb));
    let k = choose(&scores, o.layer)?;
    let chosen = scores[k].clone();
    emit(json!({"event": "layers", "chosen": k, "layers": scores.iter().map(SiteScore::to_json).collect::<Vec<_>>(),
                "message": format!("direction from block {} pass {} (separation {:.2}, {:.0}% deep)", chosen.site.block, chosen.site.pass, chosen.separation, chosen.depth * 100.0)}));

    let (before, reference) = if o.check > 0 {
        let (r, answers) = run_check(&model, &tok, h_check, b_check, check_tokens, None, emit, "original")?;
        emit(json!({"event": "check", "which": "before", "result": r.to_json()}));
        (Some(r), Some(answers))
    } else {
        (None, None)
    };
    drop(model);

    let meta = json!({"layer": chosen.site.block, "pass": chosen.site.pass, "scale": o.scale, "separation": chosen.separation,
                      "prompts": {"harmful": h_dir.len(), "harmless": b_dir.len()}, "chosen": if o.layer == LayerChoice::Auto { "auto" } else { "given" }});
    emit(json!({"event": "log", "message": format!("writing {} with the direction removed", o.out.display())}));
    let ortho = orthogonalize(&o.model, &o.out, &chosen.direction, o.scale, meta, o.shard_bytes)?;
    emit(json!({"event": "log", "message": format!("{} tensors orthogonalized, {} copied as they were{}", ortho.changed.len(), ortho.copied,
                if ortho.untied { "; the tied output head was kept as its own tensor" } else { "" })}));

    let after = if o.check > 0 {
        let model = LlamaModel::load_on(&o.out, dtype, &devs)?;
        let (r, _) = run_check(&model, &tok, h_check, b_check, check_tokens, reference.as_deref(), emit, "abliterated")?;
        emit(json!({"event": "check", "which": "after", "result": r.to_json()}));
        Some(r)
    } else {
        None
    };
    Ok(Report { chosen, scores, ortho, before, after })
}

/// A few harmless prompts, for tests and a first try (the harmful list is the person's own file).
pub const SAMPLE_HARMLESS: &[&str] = &[
    "Give me three tips for keeping a houseplant healthy.",
    "Explain how a rainbow forms.",
    "Write a short poem about the sea.",
    "What is the capital of France?",
    "Suggest a name for a friendly cat.",
    "How do I boil an egg?",
    "Summarise the water cycle in two sentences.",
    "List four fruits that are yellow.",
];

/// How much of `t` still writes along `r` on its hidden `axis`: `‖rᵀW‖ / ‖W‖` (0 once orthogonalized).
pub fn residual_write(t: &Tensor, axis: usize, r: &[f32]) -> Result<f64> {
    let shape = t.dims().to_vec();
    let v = t.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let h = shape[axis];
    if h != r.len() {
        bail!("axis {axis} is {h} wide, the direction {}", r.len());
    }
    let outer: usize = shape[..axis].iter().product();
    let inner: usize = shape[axis + 1..].iter().product();
    let (mut proj, mut norm) = (0f64, 0f64);
    for o in 0..outer {
        for j in 0..inner {
            let c: f64 = (0..h).map(|i| r[i] as f64 * v[(o * h + i) * inner + j] as f64).sum();
            proj += c * c;
        }
    }
    for x in &v {
        norm += *x as f64 * *x as f64;
    }
    Ok((proj / norm.max(1e-30)).sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layer_choices_parse() {
        assert_eq!(LayerChoice::parse("auto").unwrap(), LayerChoice::Auto);
        assert_eq!(LayerChoice::parse("7").unwrap(), LayerChoice::Site(Site { block: 7, pass: 0 }));
        assert_eq!(LayerChoice::parse("2:1").unwrap(), LayerChoice::Site(Site { block: 2, pass: 1 }));
        assert!(LayerChoice::parse("x").is_err());
    }

    #[test]
    fn roles_follow_the_writers() {
        assert_eq!(role("model.layers.3.self_attn.o_proj.weight", &[8, 16], 8).unwrap(), Role::Writer { axis: 0 });
        assert_eq!(role("model.layers.3.mlp.down_proj.weight", &[8, 32], 8).unwrap(), Role::Writer { axis: 0 });
        assert_eq!(role("model.layers.3.mlp.experts.4.down_proj.weight", &[8, 32], 8).unwrap(), Role::Writer { axis: 0 });
        assert_eq!(role("model.layers.3.mlp.experts.down_proj", &[4, 8, 32], 8).unwrap(), Role::Writer { axis: 1 });
        assert_eq!(role("model.layers.3.mlp.experts.down_proj", &[4, 32, 8], 8).unwrap(), Role::Writer { axis: 2 });
        assert!(role("model.layers.3.mlp.experts.down_proj", &[4, 8, 8], 8).is_err());
        assert_eq!(role("model.embed_tokens.weight", &[64, 8], 8).unwrap(), Role::Embed);
        assert_eq!(role("lm_head.weight", &[64, 8], 8).unwrap(), Role::Untouched);
        assert_eq!(role("model.layers.0.self_attn.q_proj.weight", &[8, 8], 8).unwrap(), Role::Untouched);
        assert_eq!(role("model.layers.0.mlp.up_proj.weight", &[32, 8], 8).unwrap(), Role::Untouched);
    }

    #[test]
    fn projecting_out_removes_the_direction_on_every_axis() {
        let r: Vec<f32> = { let v = [0.3f32, -0.5, 0.2, 0.7]; let n = v.iter().map(|x| x * x).sum::<f32>().sqrt(); v.iter().map(|x| x / n).collect() };
        for (shape, axis) in [(vec![4usize, 5], 0usize), (vec![6, 4], 1), (vec![3, 4, 5], 1), (vec![3, 5, 4], 2), (vec![4], 0)] {
            let n: usize = shape.iter().product();
            let mut x: Vec<f32> = (0..n).map(|i| ((i * 37 % 11) as f32 - 5.0) / 3.0).collect();
            project_out(&mut x, &shape, axis, &r, 1.0);
            let t = Tensor::from_vec(x, shape.clone(), &Device::Cpu).unwrap();
            assert!(residual_write(&t, axis, &r).unwrap() < 1e-6, "{shape:?} axis {axis}");
        }
        assert!(is_refusal("I’m sorry, but I can't do that"));
        assert!(!is_refusal("Sure! Here are three tips."));
    }
}
