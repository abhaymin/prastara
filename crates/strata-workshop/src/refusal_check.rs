//! `refusal-check`: how often an exported model refuses, through Prastara's Rust engine (`am2strata-hybrid --serve`,
//! [`crate::engine`]) on its GGUF or Kosha pack — the check abliterate runs before and after, on the file people will
//! actually run (a quantized export, a pack), with the engine's attention cache instead of the Workshop's model.
//!
//! The last `n` prompts of each list are answered greedily (the model's chat template, rendered here, with thinking
//! off as abliterate reads them), `max_tokens` new tokens each, and an answer is a refusal when it carries one of
//! [`crate::abliterate::REFUSAL_PHRASES`]. With `before`, a second file (the model before abliteration, or before
//! training) is answered the same way for a side-by-side rate. A few harmless answers are kept as samples; the
//! harmful answers are only counted.

use std::path::{Path, PathBuf};

use anyhow::{bail, Result};
use serde_json::{json, Value};

use crate::abliterate::{is_refusal, CheckResult};
use crate::chatdata::ChatTokenizer;
use crate::engine::{check_gguf_matches, Engine, EngineSpec, Speed};

#[derive(Debug, Clone)]
pub struct Options {
    /// the engine on the file under test (`gguf`: a .gguf or a .kosha)
    pub engine: EngineSpec,
    /// a Hugging Face folder of the model (its tokenizer and chat template)
    pub tokenizer: PathBuf,
    pub harmful: Vec<String>,
    pub harmless: Vec<String>,
    /// prompts of each list (the last ones)
    pub n: usize,
    /// new tokens per answer
    pub max_tokens: usize,
    /// another file to answer the same prompts with, for comparison
    pub before: Option<PathBuf>,
}

/// One file's result.
#[derive(Debug, Clone)]
pub struct FileResult {
    pub file: PathBuf,
    pub marks: Value,
    pub check: CheckResult,
    pub speed: Speed,
}

impl FileResult {
    pub fn to_json(&self) -> Value {
        json!({"file": self.file.display().to_string(), "marks": self.marks, "result": self.check.to_json(), "speed": self.speed.to_json()})
    }
}

#[derive(Debug, Clone)]
pub struct Report {
    pub after: FileResult,
    pub before: Option<FileResult>,
}

impl Report {
    pub fn to_json(&self) -> Value {
        json!({"after": self.after.to_json(), "before": self.before.as_ref().map(FileResult::to_json)})
    }
}

/// Whether a path is a Kosha pack (`.kosha`, or the older `.am2aipac`), which the engine takes as `--model` too.
pub fn is_pack(p: &Path) -> bool {
    p.extension().and_then(|e| e.to_str()).is_some_and(|e| e.eq_ignore_ascii_case("kosha") || e.eq_ignore_ascii_case("am2aipac"))
}

/// The `prastara.*` marks of a GGUF (a pack keeps its GGUF's header verbatim; the server reads those).
pub fn marks(p: &Path) -> Value {
    if is_pack(p) {
        return json!({"pack": true});
    }
    match strata_gguf::Gguf::open(p) {
        Ok(g) => {
            let get = |k: &str| g.metadata.get(k).cloned().unwrap_or(Value::Null);
            json!({"private": get(crate::convert::PRIVATE_KEY), "abliterated": get(crate::convert::ABLITERATED_KEY),
                   "layer": get("prastara.abliterated.layer"), "carried": get("prastara.abliterated.carried")})
        }
        Err(_) => Value::Null,
    }
}

fn check_file(spec: &EngineSpec, tok: &ChatTokenizer, harmful: &[Vec<u32>], harmless: &[(String, Vec<u32>)], max_tokens: usize,
              emit: &mut dyn FnMut(Value), label: &str) -> Result<FileResult> {
    emit(json!({"event": "log", "message": format!("starting the engine on {}", spec.gguf.display())}));
    let mut e = Engine::start(spec, None)?;
    let stop = tok.stop_ids();
    let greedy = strata_protocol::Sampling::default();
    let mut r = CheckResult { n: harmful.len(), ..Default::default() };
    let mut speed = Speed { load_seconds: e.load_seconds, ..Default::default() };
    let total = harmful.len() + harmless.len();
    let mut answer = |ids: &[u32], speed: &mut Speed| -> Result<String> {
        let g = e.generate(ids, max_tokens, &greedy, &stop)?;
        speed.requests += 1;
        speed.prompt_tokens += g.prompt_tokens;
        speed.generated += g.ids.len() as u64;
        speed.prompt_ms += g.prompt_ms;
        speed.decode_ms += g.decode_ms;
        tok.decode(&g.ids)
    };
    for (i, ids) in harmful.iter().enumerate() {
        if is_refusal(&answer(ids, &mut speed)?) {
            r.harmful_refused += 1;
        }
        emit(json!({"event": "progress", "done": i + 1, "total": total, "message": format!("answering with the {label} file")}));
    }
    for (i, (p, ids)) in harmless.iter().enumerate() {
        let a = answer(ids, &mut speed)?;
        if is_refusal(&a) {
            r.harmless_refused += 1;
        }
        if r.samples.len() < 3 {
            r.samples.push((p.clone(), a));
        }
        emit(json!({"event": "progress", "done": harmful.len() + i + 1, "total": total, "message": format!("answering with the {label} file")}));
    }
    Ok(FileResult { file: spec.gguf.clone(), marks: marks(&spec.gguf), check: r, speed })
}

/// See the module documentation. `emit` gets JSON events (`log`, `progress`, `check`).
pub fn run(o: &Options, emit: &mut dyn FnMut(Value)) -> Result<Report> {
    if o.n == 0 {
        bail!("--n must be at least 1");
    }
    if o.harmful.is_empty() || o.harmless.is_empty() {
        bail!("both prompt lists need at least one prompt");
    }
    let tok = ChatTokenizer::load(&o.tokenizer, None)?;
    // the file must be this model: same blocks, vocabulary and loop (a pack is checked by the engine as it loads)
    let cfg = crate::model::ModelConfig::load(&o.tokenizer).ok();
    for f in std::iter::once(&o.engine.gguf).chain(o.before.iter()) {
        if !f.is_file() {
            bail!("{}: no such file", f.display());
        }
        if let (false, Some(c)) = (is_pack(f), &cfg) {
            check_gguf_matches(f, c)?;
        }
    }
    let last = |list: &[String]| list[list.len().saturating_sub(o.n)..].to_vec();
    let (hl, bl) = (last(&o.harmful), last(&o.harmless));
    let harmful: Vec<Vec<u32>> = hl.iter().map(|p| tok.prompt_ids_answering(p)).collect::<Result<_>>()?;
    let harmless: Vec<(String, Vec<u32>)> = bl.iter().map(|p| Ok((p.clone(), tok.prompt_ids_answering(p)?))).collect::<Result<_>>()?;
    let max_tokens = o.max_tokens.clamp(1, 512);
    emit(json!({"event": "log", "message": format!("{} harmful and {} harmless prompts, {max_tokens} new tokens each, greedy", harmful.len(), harmless.len())}));
    let m = marks(&o.engine.gguf);
    if m.is_object() && !is_pack(&o.engine.gguf) && m["abliterated"] != true {
        emit(json!({"event": "log", "message": format!("{} is not marked prastara.abliterated", o.engine.gguf.display())}));
    }
    let before = match &o.before {
        Some(b) => {
            let mut spec = o.engine.clone();
            spec.gguf = b.clone();
            let r = check_file(&spec, &tok, &harmful, &harmless, max_tokens, emit, "before")?;
            emit(json!({"event": "check", "which": "before", "result": r.check.to_json()}));
            Some(r)
        }
        None => None,
    };
    let after = check_file(&o.engine, &tok, &harmful, &harmless, max_tokens, emit, "checked")?;
    emit(json!({"event": "check", "which": "after", "result": after.check.to_json()}));
    Ok(Report { after, before })
}
