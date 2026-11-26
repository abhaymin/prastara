//! Training by grading, in rounds (plan Part G): sample N answers per prompt, grade them (run the code, or ask the
//! judge), keep the best answer of each prompt, fine-tune a LoRA adapter on what was kept (best-of-N fine-tuning,
//! [`crate::finetune`]), run the regression suite on it ([`crate::regress`]), and promote the new adapter only when
//! nothing regressed against the adapter promoted before (the base, at first).
//!
//! Each round samples with the promoted adapter and trains a fresh adapter on everything kept so far (a prompt keeps
//! its best answer across rounds), so a rejected round costs nothing but its time. Every round writes its folder
//! (`round-N/`: `samples.jsonl`, `best.jsonl`, `adapter/`, `report/`), one line in `rounds.jsonl` and a row in
//! `rounds.md`; `promoted.json` names the adapter to use.
//!
//! DPO (pairs of the best and worst answers, the base with the adapter off as the reference) is not done yet.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use serde_json::{json, Value};

use crate::chatdata::ChatTokenizer;
use crate::finetune::{self, TrainSettings};
use crate::model::{BaseDtype, LlamaModel};
use crate::regress::{self, CodeRunner, CodeTask, Generator, Judge, JudgeItem, Sampling};

/// A prompt to answer, with what its grader needs.
#[derive(Debug, Clone)]
pub struct Prompt {
    pub id: String,
    pub text: String,
    pub code: Option<CodeTask>,
    pub judge: Option<JudgeItem>,
}

/// Scores an answer; higher is better. An answer is good enough to train on at [`Grader::passing`] or above.
pub trait Grader {
    fn grade(&self, prompt: &Prompt, answer: &str) -> Result<f64>;
    fn passing(&self) -> f64;
    fn name(&self) -> String;
}

/// Runs the answer with the prompt's tests: the share of its tests passed; only a full pass is good enough.
pub struct CodeGrader(pub CodeRunner);

impl Grader for CodeGrader {
    fn grade(&self, p: &Prompt, answer: &str) -> Result<f64> {
        let task = p.code.as_ref().with_context(|| format!("prompt {}: no tests to run (a code prompt needs \"test\" or \"cases\")", p.id))?;
        let g = self.0.grade(task, answer);
        Ok(if g.passed { 1.0 } else { g.fraction * 0.99 })
    }
    fn passing(&self) -> f64 {
        1.0
    }
    fn name(&self) -> String {
        "code".into()
    }
}

/// The judge's 1–5 score; `min` or more is good enough.
pub struct JudgeGrader<'a> {
    pub judge: &'a dyn Judge,
    pub min: f64,
}

impl Grader for JudgeGrader<'_> {
    fn grade(&self, p: &Prompt, answer: &str) -> Result<f64> {
        let item = p.judge.clone().unwrap_or(JudgeItem { id: p.id.clone(), prompt: p.text.clone(), reference: None, source: None, to: None });
        Ok(self.judge.score(&regress::judge_request(&item, answer))? as f64)
    }
    fn passing(&self) -> f64 {
        self.min
    }
    fn name(&self) -> String {
        format!("judge ({})", self.judge.name())
    }
}

/// The prompts of a JSONL file: code prompts as `regress` reads them (`{"id", "prompt", "language", "test" |
/// "cases"}`), others as judge items (`{"id", "prompt", "reference"?, "source"?, "to"?}`).
pub fn load_prompts(path: &Path, code: bool) -> Result<Vec<Prompt>> {
    if code {
        let (tasks, _, _) = regress::load_code(path, "jsonl", regress::CodeLang::Python)?;
        Ok(tasks.into_iter().map(|t| Prompt { id: t.id.clone(), text: t.prompt.clone(), code: Some(t), judge: None }).collect())
    } else {
        Ok(regress::load_judge(path)?.into_iter().map(|j| Prompt { id: j.id.clone(), text: j.prompt.clone(), code: None, judge: Some(j) }).collect())
    }
}

/// What to run.
#[derive(Debug, Clone)]
pub struct Options {
    pub model: PathBuf,
    /// an adapter to start from (the first round samples with it, and it is the first one to beat)
    pub adapter: Option<PathBuf>,
    pub prompts: Vec<Prompt>,
    pub samples: usize,
    pub rounds: usize,
    pub suite: PathBuf,
    pub out: PathBuf,
    pub temperature: f64,
    pub top_p: f64,
    pub max_tokens: usize,
    pub seed: u64,
    pub passes: Vec<usize>,
    /// the fine-tuning settings (model, data and out are set per round)
    pub train: TrainSettings,
    pub devices: Vec<String>,
    pub base_dtype: Option<BaseDtype>,
    pub max_prompt_tokens: usize,
}

impl Options {
    pub fn new(model: impl Into<PathBuf>, prompts: Vec<Prompt>, suite: impl Into<PathBuf>, out: impl Into<PathBuf>) -> Self {
        let model = model.into();
        let mut train = TrainSettings::new(model.clone(), PathBuf::new(), PathBuf::new());
        train.lora.r = 16;
        train.lora.alpha = 32.0;
        train.save_steps = 0;
        Options { model, adapter: None, prompts, samples: 4, rounds: 1, suite: suite.into(), out: out.into(), temperature: 0.8, top_p: 0.95, max_tokens: 512,
                  seed: 1, passes: vec![], train, devices: vec![], base_dtype: None, max_prompt_tokens: 4096 }
    }
}

/// One round's outcome.
#[derive(Debug, Clone)]
pub struct Round {
    pub round: usize,
    pub sampled: usize,
    pub passing: usize,
    /// prompts with a kept answer so far
    pub kept: usize,
    pub adapter: Option<PathBuf>,
    pub regressions: usize,
    pub promoted: bool,
    pub note: String,
}

#[derive(Debug, Clone)]
pub struct Outcome {
    pub rounds: Vec<Round>,
    /// the adapter to use (None: nothing beat the start)
    pub promoted: Option<PathBuf>,
}

fn regress_opts(o: &Options, adapter: Option<&Path>, out: PathBuf, against: Vec<PathBuf>) -> regress::Options {
    let mut r = regress::Options::new(&o.model, &o.suite, out);
    r.adapter = adapter.map(Path::to_path_buf);
    r.passes = o.passes.clone();
    r.against = against;
    r.devices = o.devices.clone();
    r.base_dtype = o.base_dtype;
    r.max_prompt_tokens = o.max_prompt_tokens;
    r
}

/// Run the rounds. `judge` serves the suite's judge tests (and may be the grader's own).
pub fn run(o: &Options, grader: &dyn Grader, judge: Option<&dyn Judge>, emit: &mut dyn FnMut(Value), should_stop: &dyn Fn() -> bool) -> Result<Outcome> {
    if o.prompts.is_empty() {
        bail!("no prompts to answer");
    }
    if o.samples == 0 || o.rounds == 0 {
        bail!("--samples and --rounds must be 1 or more");
    }
    std::fs::create_dir_all(&o.out).with_context(|| format!("{}", o.out.display()))?;
    let log_path = o.out.join("rounds.jsonl");
    std::fs::write(&log_path, "")?;
    let mut md = String::from("# Grade and train\n\n| round | sampled | passing | prompts kept | regressions | promoted | note |\n|---|---|---|---|---|---|---|\n");
    // the start: its report is the one the first round must not fall below
    emit(json!({"event": "log", "message": "round 0: the regression suite on the starting model"}));
    let base_report = o.out.join("round-0").join("report");
    regress::run(&regress_opts(o, o.adapter.as_deref(), base_report.clone(), vec![]), judge, emit, should_stop)?;
    let mut promoted = o.adapter.clone();
    let mut promoted_report = base_report;
    let mut kept: BTreeMap<String, (f64, String, String)> = BTreeMap::new(); // id -> (score, prompt, answer)
    let mut rounds = Vec::new();
    for round in 1..=o.rounds {
        if should_stop() {
            bail!("stopped");
        }
        let dir = o.out.join(format!("round-{round}"));
        std::fs::create_dir_all(&dir)?;
        emit(json!({"event": "log", "message": format!("round {round}: {} answers to each of {} prompts{}", o.samples, o.prompts.len(),
                                                       promoted.as_ref().map_or(String::new(), |a| format!(" with {}", a.display())))}));
        // 1-2. sample and grade
        let (mut sampled, mut passing) = (0, 0);
        let mut samples = String::new();
        {
            let tok = ChatTokenizer::load(&o.model, None)?;
            let devs = finetune::open_devices(&o.devices)?;
            let dtype = o.base_dtype.unwrap_or(if devs.iter().any(|d| !d.is_cpu()) { BaseDtype::BF16 } else { BaseDtype::F32 });
            let mut model = LlamaModel::load_on(&o.model, dtype, &devs)?;
            if let Some(a) = &promoted {
                model.load_adapter(a)?;
            }
            let gen = Generator::new(&model, &tok, o.max_prompt_tokens);
            let total = o.prompts.len() * o.samples;
            for (pi, p) in o.prompts.iter().enumerate() {
                let Some(ids) = gen.prompt(&p.text)? else {
                    emit(json!({"event": "log", "message": format!("prompt {}: longer than {} tokens, left out", p.id, o.max_prompt_tokens)}));
                    continue;
                };
                for k in 0..o.samples {
                    if should_stop() {
                        bail!("stopped");
                    }
                    let seed = o.seed ^ ((round as u64) << 40) ^ ((pi as u64) << 16) ^ k as u64;
                    let a = tok.decode(&gen.generate(&ids, o.max_tokens, Some(Sampling { temperature: o.temperature, top_p: o.top_p, seed }))?)?;
                    let score = grader.grade(p, &a)?;
                    sampled += 1;
                    if score >= grader.passing() {
                        passing += 1;
                    }
                    samples.push_str(&format!("{}\n", json!({"id": p.id, "sample": k, "score": score, "answer": a})));
                    if score >= grader.passing() && kept.get(&p.id).is_none_or(|(s, _, _)| score > *s) {
                        kept.insert(p.id.clone(), (score, p.text.clone(), a));
                    }
                    emit(json!({"event": "progress", "done": pi * o.samples + k + 1, "total": total, "message": format!("round {round}: sampling and grading ({passing} good so far)")}));
                }
            }
        }
        std::fs::write(dir.join("samples.jsonl"), &samples)?;
        let best: String = kept.values().map(|(_, q, a)| format!("{}\n", json!({"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": a}]}))).collect();
        std::fs::write(dir.join("best.jsonl"), &best)?;
        let mut r = Round { round, sampled, passing, kept: kept.len(), adapter: None, regressions: 0, promoted: false, note: String::new() };
        if kept.is_empty() {
            r.note = "no answer was good enough to train on".into();
        } else {
            // 3. fine-tune on the best
            let mut s = o.train.clone();
            s.model = o.model.clone();
            s.data = dir.join("best.jsonl");
            s.out = dir.join("adapter");
            s.eval_data = None;
            if kept.len() < 10 {
                s.val_split = 0.0;
            }
            s.devices = o.devices.clone();
            if let Some(d) = o.base_dtype {
                s.base_dtype = d;
            }
            s.resume = None;
            emit(json!({"event": "log", "message": format!("round {round}: fine-tuning on the best answers of {} prompts", kept.len())}));
            let tr = finetune::train(&s, &mut |e| if let finetune::Event::Step { step, total, loss, .. } = e {
                emit(json!({"event": "progress", "done": step, "total": total, "step": step, "loss": loss, "message": format!("round {round}: training, loss {loss:.4}")}));
            }, should_stop)?;
            if tr.stopped {
                bail!("stopped");
            }
            r.adapter = Some(s.out.clone());
            // 4. the suite on it, against the promoted one
            let rep = regress::run(&regress_opts(o, Some(&s.out), dir.join("report"), vec![promoted_report.clone()]), judge, emit, should_stop)?;
            r.regressions = rep.regressions().len();
            // 5. promote only with nothing worse
            if r.regressions == 0 {
                r.promoted = true;
                promoted = Some(s.out.clone());
                promoted_report = dir.join("report");
                r.note = "promoted: nothing regressed".into();
            } else {
                let names: Vec<String> = rep.regressions().iter().map(|(i, _, _)| format!("{} at {}", rep.results[*i].test, rep.results[*i].passes)).collect();
                r.note = format!("kept the earlier adapter: {} regressed", names.join(", "));
            }
        }
        emit(json!({"event": "log", "message": format!("round {round}: {}", r.note)}));
        let line = json!({"round": round, "sampled": r.sampled, "passing": r.passing, "kept": r.kept, "adapter": r.adapter.as_ref().map(|a| a.display().to_string()),
                          "regressions": r.regressions, "promoted": r.promoted, "note": r.note,
                          "report": r.adapter.as_ref().map(|_| dir.join("report").join("report.json").display().to_string())});
        let mut log = std::fs::read_to_string(&log_path).unwrap_or_default();
        log.push_str(&format!("{line}\n"));
        std::fs::write(&log_path, log)?;
        md.push_str(&format!("| {round} | {} | {} | {} | {} | {} | {} |\n", r.sampled, r.passing, r.kept, r.regressions, if r.promoted { "yes" } else { "no" }, r.note));
        std::fs::write(o.out.join("rounds.md"), &md)?;
        std::fs::write(o.out.join("promoted.json"), serde_json::to_string_pretty(&json!({
            "adapter": promoted.as_ref().map(|a| a.display().to_string()), "report": promoted_report.join("report.json").display().to_string(), "after_round": round}))?)?;
        rounds.push(r);
    }
    Ok(Outcome { rounds, promoted })
}
