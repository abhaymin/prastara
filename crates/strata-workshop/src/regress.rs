//! Regression runs (plan Part G): the same fixed tests on each version of a model — the base, a looped copy at each
//! pass count, every trained adapter — and a report card that says what got worse.
//!
//! A **suite** (JSON) lists the tests:
//!
//! ```json
//! {"name": "mula-core", "judge_model": null, "tests": [
//!   {"name": "held-out", "kind": "loss", "data": "heldout.jsonl", "limit": 200},
//!   {"name": "humaneval", "kind": "code", "format": "humaneval", "data": "humaneval.jsonl", "limit": 50, "max_tokens": 384},
//!   {"name": "mbpp", "kind": "code", "format": "mbpp", "data": "mbpp-sanitized.jsonl"},
//!   {"name": "lcb", "kind": "code", "format": "livecodebench", "data": "livecodebench-v6.jsonl"},
//!   {"name": "codeforces", "kind": "code", "format": "codeforces", "data": "codeforces-test.jsonl"},
//!   {"name": "strandset", "kind": "code", "format": "strandset", "data": "strandset-rust-test.jsonl"},
//!   {"name": "indicqa-hi", "kind": "qa", "format": "squad", "data": "indicqa.hi.json", "limit": 100},
//!   {"name": "translate", "kind": "judge", "data": "translate.jsonl"}
//! ]}
//! ```
//!
//! Relative paths are read from the suite's folder. Tests:
//! - **loss**: the mean loss of the answers of a chat dataset (lower is better);
//! - **code**: a greedy answer to each task, run with its tests in the [sandbox](crate::sandbox) — Python with
//!   `python3 -I`, Rust compiled with `rustc` — pass@1 (formats: humaneval, mbpp, livecodebench (its stdin tasks),
//!   codeforces (its examples), strandset (its std-only tasks), jsonl: `{"id", "prompt", "language", "test" | "cases"}`);
//! - **qa**: a short answer read from a passage, exact match and token F1 (SQuAD files such as IndicQA, or jsonl
//!   `{"id", "context", "question", "answers": [...]}`); unanswerable questions are left out;
//! - **judge**: an answer scored 1–5 by a model through the server's API (`AM2STRATA_URL`, `AM2STRATA_API_KEY`, as
//!   datagen reaches it): jsonl `{"id", "prompt", "reference"?, "source"?, "to"?}` (`to`: a translation into it).
//!
//! The report keeps every item's score, so a later report pairs its items with an earlier one's and a **paired
//! bootstrap** of the difference says whether a drop is real: a test regressed when the whole 95 % interval of the
//! change lies on the worse side of zero, against the best earlier report for it.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context, Result};
use candle_core::{DType, Device};
use serde_json::{json, Value};

use crate::chatdata::{ChatTokenizer, DataFormat};
use crate::model::{BaseDtype, Ctx, LlamaModel};
use crate::recipes::{code_blocks, Draw};
use crate::sandbox::{self, Limits, Scratch};

// ------------------------------------------------------------------------------------------------------ generator

/// How an answer is drawn: greedy (None), or sampled at a temperature with nucleus `top_p`.
#[derive(Debug, Clone, Copy)]
pub struct Sampling {
    pub temperature: f64,
    pub top_p: f64,
    pub seed: u64,
}

/// Answers from the Workshop's model: no attention cache (each token runs the whole sequence again), bounded by
/// `max_new` tokens, so it suits short answers and small test sets; prompts longer than `max_prompt` are left out.
pub struct Generator<'a> {
    pub model: &'a LlamaModel,
    pub tok: &'a ChatTokenizer,
    pub stop: Vec<u32>,
    pub max_prompt: usize,
}

impl<'a> Generator<'a> {
    pub fn new(model: &'a LlamaModel, tok: &'a ChatTokenizer, max_prompt: usize) -> Self {
        Self { model, tok, stop: tok.stop_ids(), max_prompt }
    }

    /// The prompt as token ids, or None when it is longer than `max_prompt`.
    pub fn prompt(&self, user: &str) -> Result<Option<Vec<u32>>> {
        let ids = self.tok.prompt_ids(user)?;
        Ok((ids.len() <= self.max_prompt).then_some(ids))
    }

    /// Up to `max_new` tokens after `prompt`, stopping at an end-of-turn id.
    pub fn generate(&self, prompt: &[u32], max_new: usize, sampling: Option<Sampling>) -> Result<Vec<u32>> {
        let mut seq = prompt.to_vec();
        let mut out = Vec::new();
        let mut draw = sampling.map(|s| Draw::new(s.seed));
        let room = self.model.cfg.max_position_embeddings.max(prompt.len() + 1);
        for _ in 0..max_new {
            if seq.len() >= room {
                break;
            }
            let h = self.model.upper_hidden(&self.model.lower(&seq)?, &Ctx::eval())?;
            let last = h.narrow(0, seq.len() - 1, 1)?;
            let logits = self.model.head(&last)?.squeeze(0)?.to_dtype(DType::F32)?.to_device(&Device::Cpu)?.to_vec1::<f32>()?;
            let next = match (sampling, draw.as_mut()) {
                (Some(s), Some(d)) if s.temperature > 0.0 => sample(&logits, s, d),
                _ => argmax(&logits),
            };
            if self.stop.contains(&next) {
                break;
            }
            out.push(next);
            seq.push(next);
        }
        Ok(out)
    }

    /// One answer to one user message, as text; None when the prompt is too long.
    pub fn answer(&self, user: &str, max_new: usize, sampling: Option<Sampling>) -> Result<Option<String>> {
        match self.prompt(user)? {
            Some(ids) => Ok(Some(self.tok.decode(&self.generate(&ids, max_new, sampling)?)?)),
            None => Ok(None),
        }
    }
}

fn argmax(logits: &[f32]) -> u32 {
    logits.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|(i, _)| i as u32).unwrap_or(0)
}

/// A token drawn from the softmax of `logits / temperature`, kept to the smallest set holding `top_p` of the mass.
fn sample(logits: &[f32], s: Sampling, d: &mut Draw) -> u32 {
    let t = s.temperature.max(1e-4);
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
    let mut p: Vec<(usize, f64)> = logits.iter().enumerate().map(|(i, &l)| (i, ((l as f64 - max) / t).exp())).collect();
    p.sort_by(|a, b| b.1.total_cmp(&a.1));
    let total: f64 = p.iter().map(|x| x.1).sum();
    let mut kept = 0.0;
    let mut n = 0;
    for (_, w) in &p {
        kept += w / total;
        n += 1;
        if kept >= s.top_p.clamp(0.01, 1.0) {
            break;
        }
    }
    let p = &p[..n.max(1)];
    let mass: f64 = p.iter().map(|x| x.1).sum();
    let mut r = (d.next() >> 11) as f64 / (1u64 << 53) as f64 * mass;
    for (i, w) in p {
        if r < *w {
            return *i as u32;
        }
        r -= w;
    }
    p[p.len() - 1].0 as u32
}

// ------------------------------------------------------------------------------------------------- python literals

/// A Python literal as `repr` writes it (dicts, lists, tuples, strings in either quote, numbers, True/False/None) as
/// JSON: some datasets store their fields that way.
pub fn python_literal(s: &str) -> Result<Value> {
    let c: Vec<char> = s.chars().collect();
    let mut i = 0;
    let v = py_value(&c, &mut i)?;
    py_ws(&c, &mut i);
    if i != c.len() {
        bail!("python literal: unexpected text at {i}");
    }
    Ok(v)
}

fn py_ws(c: &[char], i: &mut usize) {
    while *i < c.len() && c[*i].is_whitespace() {
        *i += 1;
    }
}

fn py_value(c: &[char], i: &mut usize) -> Result<Value> {
    py_ws(c, i);
    let Some(&ch) = c.get(*i) else { bail!("python literal: ends early") };
    match ch {
        '{' | '[' | '(' => {
            let close = match ch { '{' => '}', '[' => ']', _ => ')' };
            *i += 1;
            let mut list = Vec::new();
            let mut map = serde_json::Map::new();
            loop {
                py_ws(c, i);
                if c.get(*i) == Some(&close) {
                    *i += 1;
                    break;
                }
                let v = py_value(c, i)?;
                py_ws(c, i);
                if ch == '{' {
                    if c.get(*i) != Some(&':') {
                        bail!("python literal: a dict entry without ':' at {i}");
                    }
                    *i += 1;
                    let val = py_value(c, i)?;
                    let k = match v { Value::String(s) => s, other => other.to_string() };
                    map.insert(k, val);
                } else {
                    list.push(v);
                }
                py_ws(c, i);
                match c.get(*i) {
                    Some(',') => *i += 1,
                    Some(x) if *x == close => {}
                    _ => bail!("python literal: expected ',' or '{close}' at {i}"),
                }
            }
            Ok(if ch == '{' { Value::Object(map) } else { Value::Array(list) })
        }
        '\'' | '"' => {
            let q = ch;
            *i += 1;
            let mut out = String::new();
            while let Some(&x) = c.get(*i) {
                *i += 1;
                if x == q {
                    return Ok(Value::String(out));
                }
                if x != '\\' {
                    out.push(x);
                    continue;
                }
                let Some(&e) = c.get(*i) else { bail!("python literal: a string ends in '\\'") };
                *i += 1;
                let hex = |n: usize, i: &mut usize| -> Result<char> {
                    let h: String = c.get(*i..*i + n).ok_or_else(|| anyhow!("python literal: short escape"))?.iter().collect();
                    *i += n;
                    char::from_u32(u32::from_str_radix(&h, 16)?).ok_or_else(|| anyhow!("python literal: bad escape {h}"))
                };
                match e {
                    'n' => out.push('\n'),
                    't' => out.push('\t'),
                    'r' => out.push('\r'),
                    '0' => out.push('\0'),
                    'x' => out.push(hex(2, i)?),
                    'u' => out.push(hex(4, i)?),
                    'U' => out.push(hex(8, i)?),
                    '\n' => {}
                    other => out.push(other),
                }
            }
            bail!("python literal: an unterminated string")
        }
        _ => {
            let start = *i;
            while *i < c.len() && !matches!(c[*i], ',' | ':' | ']' | '}' | ')') && !c[*i].is_whitespace() {
                *i += 1;
            }
            let w: String = c[start..*i].iter().collect();
            Ok(match w.as_str() {
                "True" => Value::Bool(true),
                "False" => Value::Bool(false),
                "None" => Value::Null,
                n => match n.parse::<i64>() {
                    Ok(x) => json!(x),
                    Err(_) => json!(n.parse::<f64>().map_err(|_| anyhow!("python literal: {n:?}"))?),
                },
            })
        }
    }
}

/// A field that is JSON already, or a string holding JSON or a Python literal.
fn structured(v: &Value) -> Value {
    match v {
        Value::String(s) => serde_json::from_str(s).or_else(|_| python_literal(s)).unwrap_or(Value::Null),
        other => other.clone(),
    }
}

// ------------------------------------------------------------------------------------------------------ code tests

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CodeLang {
    Python,
    Rust,
}

impl CodeLang {
    pub fn from_s(s: &str) -> Result<CodeLang> {
        match s.to_lowercase().as_str() {
            "python" | "py" | "python3" => Ok(CodeLang::Python),
            "rust" | "rs" => Ok(CodeLang::Rust),
            o => bail!("language {o:?}: python or rust"),
        }
    }

    fn tags(self) -> &'static [&'static str] {
        match self {
            CodeLang::Python => &["python", "py", "python3"],
            CodeLang::Rust => &["rust", "rs"],
        }
    }
}

/// How an answer is checked.
#[derive(Debug, Clone)]
pub enum CodeCheck {
    /// `header + code + suffix` must run (or, for Rust with tests, its tests pass). With `stub` (a function's
    /// signature and docstring, its name): an answer that does not define the function is taken as its body.
    Program { header: String, stub: Option<(String, String)>, suffix: String },
    /// the program reads each input on stdin and must print the output (compared word by word)
    Stdio(Vec<(String, String)>),
}

/// One coding task.
#[derive(Debug, Clone)]
pub struct CodeTask {
    pub id: String,
    /// what the model is asked
    pub prompt: String,
    pub lang: CodeLang,
    pub check: CodeCheck,
}

/// What running an answer gave.
#[derive(Debug, Clone)]
pub struct CodeGrade {
    pub passed: bool,
    /// the share of its test cases passed (1 or 0 for a program with its own tests)
    pub fraction: f64,
    pub ending: String,
}

/// The tools and limits code runs with.
#[derive(Debug, Clone)]
pub struct CodeRunner {
    pub python: Option<PathBuf>,
    pub rustc: Option<PathBuf>,
    pub run: Limits,
    pub compile: Limits,
}

impl CodeRunner {
    pub fn detect() -> CodeRunner {
        CodeRunner { python: sandbox::find_program("python3"), rustc: crate::recipes::find_rustc(), run: Limits::run(), compile: Limits::compile() }
    }

    /// Whether this machine has the tool for `lang`.
    pub fn can(&self, lang: CodeLang) -> bool {
        match lang {
            CodeLang::Python => self.python.is_some(),
            CodeLang::Rust => self.rustc.is_some(),
        }
    }

    /// Run an answer to `task` with its tests.
    pub fn grade(&self, task: &CodeTask, answer: &str) -> CodeGrade {
        let fail = |why: &str| CodeGrade { passed: false, fraction: 0.0, ending: why.to_string() };
        let code = extract_code(answer, task.lang);
        if code.trim().is_empty() {
            return fail("no code in the answer");
        }
        let Ok(dir) = Scratch::new("code") else { return fail("no scratch folder") };
        let program = match &task.check {
            CodeCheck::Program { header, stub, suffix } => {
                let body = match stub {
                    Some((sig, name)) if !defines(&code, name, task.lang) => format!("{sig}{code}"),
                    _ => code.clone(),
                };
                format!("{header}\n{body}\n{suffix}\n")
            }
            CodeCheck::Stdio(_) => code.clone(),
        };
        // a Rust program with tests runs them; one without is run as it is
        let rust_tests = task.lang == CodeLang::Rust && program.contains("#[test]");
        let exe: (PathBuf, Vec<String>) = match task.lang {
            CodeLang::Python => {
                let Some(py) = &self.python else { return fail("no python3 on this machine") };
                let src = dir.0.join("prog.py");
                if std::fs::write(&src, &program).is_err() { return fail("cannot write the program"); }
                (py.clone(), vec!["-I".into(), src.display().to_string()])
            }
            CodeLang::Rust => {
                let Some(rustc) = &self.rustc else { return fail("no rustc on this machine") };
                let src = dir.0.join("prog.rs");
                if std::fs::write(&src, &program).is_err() { return fail("cannot write the program"); }
                let bin = dir.0.join("prog");
                let (s, b) = (src.display().to_string(), bin.display().to_string());
                let mut args = vec!["--edition", "2021", "-A", "warnings", "-C", "opt-level=1", "-o", &b];
                if rust_tests { args.push("--test"); }
                args.push(&s);
                let o = sandbox::run(rustc, &args, &dir.0, b"", &self.compile);
                if !o.ok() {
                    return fail(&format!("does not compile ({})", o.ending.label()));
                }
                (bin, if rust_tests { vec!["--test-threads=1".into()] } else { vec![] })
            }
        };
        let args: Vec<&str> = exe.1.iter().map(String::as_str).collect();
        match &task.check {
            CodeCheck::Program { .. } => {
                let o = sandbox::run(&exe.0, &args, &dir.0, b"", &self.run);
                CodeGrade { passed: o.ok(), fraction: if o.ok() { 1.0 } else { 0.0 }, ending: o.ending.label() }
            }
            CodeCheck::Stdio(cases) => {
                let mut good = 0;
                let mut ending = "ok".to_string();
                for (input, want) in cases {
                    let o = sandbox::run(&exe.0, &args, &dir.0, input.as_bytes(), &self.run);
                    if o.ok() && same_words(&o.stdout, want) {
                        good += 1;
                    } else if ending == "ok" {
                        ending = if o.ok() { "wrong output".into() } else { o.ending.label() };
                    }
                }
                let n = cases.len().max(1);
                CodeGrade { passed: good == cases.len() && !cases.is_empty(), fraction: good as f64 / n as f64, ending }
            }
        }
    }
}

fn same_words(got: &str, want: &str) -> bool {
    got.split_whitespace().eq(want.split_whitespace())
}

fn defines(code: &str, name: &str, lang: CodeLang) -> bool {
    match lang {
        CodeLang::Python => code.contains(&format!("def {name}(")),
        CodeLang::Rust => code.contains(&format!("fn {name}")),
    }
}

/// The code in an answer: its first fenced block in the language (else its first block, else all of it), after any
/// thinking.
pub fn extract_code(answer: &str, lang: CodeLang) -> String {
    let a = match answer.rfind("</think>") { Some(i) => &answer[i + 8..], None => answer };
    let blocks = code_blocks(a);
    if let Some((_, c)) = blocks.iter().find(|(t, _)| lang.tags().contains(&t.as_str())) {
        return c.clone();
    }
    if let Some((_, c)) = blocks.first() {
        return c.clone();
    }
    // an unclosed fence: what follows it
    if let Some(i) = a.find("```") {
        let rest = &a[i + 3..];
        return rest.split_once('\n').map_or("", |x| x.1).to_string();
    }
    a.to_string()
}

fn s(v: &Value, k: &str) -> String {
    match &v[k] {
        Value::String(x) => x.clone(),
        Value::Null => String::new(),
        o => o.to_string(),
    }
}

fn read_rows(path: &Path) -> Result<Vec<Value>> {
    let t = std::fs::read_to_string(path).with_context(|| format!("{}", path.display()))?;
    let mut out = Vec::new();
    for (i, l) in t.lines().enumerate() {
        if l.trim().is_empty() {
            continue;
        }
        out.push(serde_json::from_str(l).with_context(|| format!("{} line {}", path.display(), i + 1))?);
    }
    Ok(out)
}

const PY_ASK: &str = "Reply with the complete code in one ```python code block.";

/// The coding tasks of a file in a known format, and how many were left out (and why, the first reason).
pub fn load_code(path: &Path, format: &str, default_lang: CodeLang) -> Result<(Vec<CodeTask>, usize, Option<String>)> {
    let rows = read_rows(path)?;
    let mut tasks = Vec::new();
    let mut skipped = 0;
    let mut why = None;
    let mut skip = |w: &str| { skipped += 1; if why.is_none() { why = Some(w.to_string()); } };
    for (n, r) in rows.iter().enumerate() {
        match format {
            "humaneval" => {
                let (prompt, entry) = (s(r, "prompt"), s(r, "entry_point"));
                let header: String = prompt.lines().filter(|l| l.starts_with("import ") || l.starts_with("from ")).map(|l| format!("{l}\n")).collect();
                tasks.push(CodeTask {
                    id: s(r, "task_id"),
                    prompt: format!("Complete this Python function. {PY_ASK}\n\n```python\n{prompt}```"),
                    lang: CodeLang::Python,
                    check: CodeCheck::Program { header, stub: Some((prompt.clone(), entry.clone())), suffix: format!("{}\n\ncheck({entry})\n", s(r, "test")) },
                });
            }
            "mbpp" => {
                let tests: Vec<String> = structured(&r["test_list"]).as_array().into_iter().flatten().filter_map(|t| t.as_str().map(str::to_string)).collect();
                let imports: Vec<String> = structured(&r["test_imports"]).as_array().into_iter().flatten().filter_map(|t| t.as_str().map(str::to_string)).collect();
                if tests.is_empty() { skip("no tests"); continue; }
                let text = if r["prompt"].is_string() { s(r, "prompt") } else { s(r, "text") };
                tasks.push(CodeTask {
                    id: s(r, "task_id"),
                    prompt: format!("{text}\nYour code should pass this test:\n{}\n{PY_ASK}", tests[0]),
                    lang: CodeLang::Python,
                    check: CodeCheck::Program { header: imports.join("\n"), stub: None, suffix: tests.join("\n") },
                });
            }
            "livecodebench" => {
                let cases = structured(&r["public_test_cases"]);
                let cases: Vec<&Value> = cases.as_array().into_iter().flatten().collect();
                if cases.is_empty() || cases.iter().any(|c| c["testtype"] != "stdin") { skip("a function-call task (only stdin tasks run here)"); continue; }
                tasks.push(CodeTask {
                    id: s(r, "question_id"),
                    prompt: format!("{}\n\nWrite a Python 3 program that reads the input from standard input and prints the answer. {PY_ASK}", s(r, "question_content")),
                    lang: CodeLang::Python,
                    check: CodeCheck::Stdio(cases.iter().map(|c| (s(c, "input"), s(c, "output"))).collect()),
                });
            }
            "codeforces" => {
                let ex = structured(&r["examples"]);
                let ex: Vec<(String, String)> = ex.as_array().into_iter().flatten().map(|c| (s(c, "input"), s(c, "output"))).collect();
                if ex.is_empty() { skip("no examples"); continue; }
                let shown: String = ex.iter().map(|(i, o)| format!("\nExample input:\n{i}\nExample output:\n{o}\n")).collect();
                tasks.push(CodeTask {
                    id: s(r, "id"),
                    prompt: format!("{}\n{shown}\nWrite a Python 3 program that reads the input from standard input and prints the answer. {PY_ASK}", s(r, "description")),
                    lang: CodeLang::Python,
                    check: CodeCheck::Stdio(ex),
                });
            }
            "strandset" => {
                let t = s(r, "test");
                let t = t.trim().trim_start_matches("```json").trim_end_matches("```").trim();
                let Ok(test) = serde_json::from_str::<Value>(t) else { skip("no runnable test"); continue };
                let main = s(&test, "main_rs");
                if test["dependencies"].as_array().is_some_and(|d| !d.is_empty()) || !main.contains("{generated_code}")
                    || !crate::recipes::self_contained_rust(&main) || main.contains("#[tokio::") {
                    skip("needs crates beyond std (only std-only tasks build with rustc alone)");
                    continue;
                }
                let input = structured(&r["input_data"]);
                let (head, tail) = main.split_once("{generated_code}").unwrap();
                let what = match s(r, "task_category").as_str() {
                    "bug_detection" => "Fix the bug in this Rust code.",
                    "code_optimization" => "Make this Rust code faster, keeping what it does.",
                    "code_refactoring" => "Refactor this Rust code, keeping what it does.",
                    "code_review" => "Review this Rust code and write the improved version.",
                    "test_generation" => "Write Rust unit tests for this code.",
                    _ => "Write the Rust code asked for.",
                };
                let fields: String = input.as_object().into_iter().flatten()
                    .map(|(k, v)| format!("\n{}:\n{}\n", k.replace('_', " "), v.as_str().map(str::to_string).unwrap_or_else(|| v.to_string()))).collect();
                tasks.push(CodeTask {
                    id: format!("{}#{n}", s(r, "crate_name")),
                    prompt: format!("{what}\n{fields}\nReply with only the new code in one ```rust code block."),
                    lang: CodeLang::Rust,
                    check: CodeCheck::Program { header: head.to_string(), stub: None, suffix: tail.to_string() },
                });
            }
            "jsonl" | "" => {
                let lang = match r["language"].as_str() { Some(l) => CodeLang::from_s(l)?, None => default_lang };
                let id = if r["id"].is_null() { format!("{n}") } else { s(r, "id") };
                let check = match r["cases"].as_array() {
                    Some(c) => CodeCheck::Stdio(c.iter().map(|c| (s(c, "input"), s(c, "output"))).collect()),
                    None => CodeCheck::Program { header: s(r, "prefix"), stub: None, suffix: s(r, "test") },
                };
                tasks.push(CodeTask { id, prompt: s(r, "prompt"), lang, check });
            }
            other => bail!("code format {other:?}: humaneval, mbpp, livecodebench, codeforces, strandset or jsonl"),
        }
    }
    Ok((tasks, skipped, why))
}

// -------------------------------------------------------------------------------------------------------- QA tests

#[derive(Debug, Clone)]
pub struct QaItem {
    pub id: String,
    pub context: String,
    pub question: String,
    pub answers: Vec<String>,
}

/// SQuAD-style questions (IndicQA), or jsonl; unanswerable ones left out (their count returned).
pub fn load_qa(path: &Path, format: &str) -> Result<(Vec<QaItem>, usize)> {
    let mut items = Vec::new();
    let mut skipped = 0;
    let mut push = |id: String, context: String, question: String, answers: Vec<String>| {
        let answers: Vec<String> = answers.into_iter().filter(|a| !a.trim().is_empty()).collect();
        if answers.is_empty() { skipped += 1; } else { items.push(QaItem { id, context, question, answers }); }
    };
    match format {
        "squad" | "" => {
            let v: Value = serde_json::from_str(&std::fs::read_to_string(path).with_context(|| format!("{}", path.display()))?)?;
            for d in v["data"].as_array().ok_or_else(|| anyhow!("{}: no \"data\" list (a SQuAD file)", path.display()))? {
                for p in d["paragraphs"].as_array().into_iter().flatten() {
                    for q in p["qas"].as_array().into_iter().flatten() {
                        let a = q["answers"].as_array().into_iter().flatten().filter_map(|a| a["text"].as_str().map(str::to_string)).collect();
                        push(s(q, "id"), s(p, "context"), s(q, "question"), a);
                    }
                }
            }
        }
        "jsonl" => {
            for (n, r) in read_rows(path)?.iter().enumerate() {
                let a = r["answers"].as_array().into_iter().flatten().filter_map(|a| a.as_str().map(str::to_string).or_else(|| a["text"].as_str().map(str::to_string))).collect();
                push(if r["id"].is_null() { n.to_string() } else { s(r, "id") }, s(r, "context"), s(r, "question"), a);
            }
        }
        o => bail!("qa format {o:?}: squad or jsonl"),
    }
    Ok((items, skipped))
}

pub fn qa_prompt(q: &QaItem) -> String {
    format!("Read the passage and answer the question with a short phrase copied from the passage. Reply with the answer only.\n\n\
             Passage:\n{}\n\nQuestion: {}", q.context, q.question)
}

/// Lower case, punctuation (Latin and Indic) and English articles out, split on white space.
pub fn qa_tokens(t: &str) -> Vec<String> {
    let t = match t.rfind("</think>") { Some(i) => &t[i + 8..], None => t };
    let cleaned: String = t.to_lowercase().chars()
        .map(|c| if c.is_ascii_punctuation() || "।॥‘’“”«»–—…¿¡·".contains(c) { ' ' } else { c }).collect();
    cleaned.split_whitespace().filter(|w| !matches!(*w, "a" | "an" | "the")).map(str::to_string).collect()
}

/// (exact match, token F1) of an answer against the best of the references.
pub fn qa_score(answer: &str, refs: &[String]) -> (f64, f64) {
    let a = qa_tokens(answer);
    let mut best = (0.0f64, 0.0f64);
    for r in refs {
        let b = qa_tokens(r);
        let em = if a == b { 1.0 } else { 0.0 };
        let mut pool: BTreeMap<&str, usize> = BTreeMap::new();
        for w in &b {
            *pool.entry(w).or_default() += 1;
        }
        let mut common = 0;
        for w in &a {
            if let Some(n) = pool.get_mut(w.as_str()).filter(|n| **n > 0) {
                *n -= 1;
                common += 1;
            }
        }
        let f1 = if common == 0 { 0.0 } else {
            let (p, r) = (common as f64 / a.len() as f64, common as f64 / b.len() as f64);
            2.0 * p * r / (p + r)
        };
        best = (best.0.max(em), best.1.max(f1));
    }
    best
}

// ----------------------------------------------------------------------------------------------------- judge tests

#[derive(Debug, Clone)]
pub struct JudgeItem {
    pub id: String,
    pub prompt: String,
    pub reference: Option<String>,
    pub source: Option<String>,
    /// a translation into this language: the judge scores faithfulness and fluency
    pub to: Option<String>,
}

pub fn load_judge(path: &Path) -> Result<Vec<JudgeItem>> {
    let opt = |r: &Value, k: &str| r[k].as_str().filter(|x| !x.is_empty()).map(str::to_string);
    Ok(read_rows(path)?.iter().enumerate().map(|(n, r)| JudgeItem {
        id: if r["id"].is_null() { n.to_string() } else { s(r, "id") },
        prompt: s(r, "prompt"), reference: opt(r, "reference"), source: opt(r, "source"), to: opt(r, "to"),
    }).collect())
}

/// The judge's request for an answer to `item`.
pub fn judge_request(item: &JudgeItem, answer: &str) -> String {
    match &item.to {
        Some(to) => crate::recipes::judge_translation(item.source.as_deref().unwrap_or(&item.prompt), answer, to),
        None => crate::datagen::judge_prompt(&item.prompt, answer, item.reference.as_deref().or(item.source.as_deref())),
    }
}

/// A judge: the request text in, a 1–5 score out (0: no score).
pub trait Judge {
    fn score(&self, request: &str) -> Result<u8>;
    fn name(&self) -> String;
}

impl Judge for crate::datagen::Model {
    fn score(&self, request: &str) -> Result<u8> {
        let reply = self.ask("You are a strict, fair grader. Answer only with the JSON asked for.", request, 0.0, 200, false)?;
        Ok(crate::datagen::judge_score(&reply))
    }
    fn name(&self) -> String {
        self.name.clone()
    }
}

// ---------------------------------------------------------------------------------------------------------- suite

/// One test of a suite, as the file gives it.
#[derive(Debug, Clone)]
pub struct TestSpec {
    pub name: String,
    pub kind: String,
    pub format: String,
    pub data: PathBuf,
    pub limit: Option<usize>,
    pub max_tokens: Option<usize>,
    pub language: Option<String>,
    pub cpu_seconds: Option<u64>,
    pub memory_mb: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct Suite {
    pub path: PathBuf,
    pub name: String,
    pub judge_model: Option<String>,
    pub tests: Vec<TestSpec>,
}

impl Suite {
    pub fn load(path: &Path) -> Result<Suite> {
        let v: Value = serde_json::from_str(&std::fs::read_to_string(path).with_context(|| format!("{}", path.display()))?)
            .with_context(|| format!("{}: not JSON", path.display()))?;
        Suite::from_json(&v, path)
    }

    pub fn from_json(v: &Value, path: &Path) -> Result<Suite> {
        let base = path.parent().unwrap_or(Path::new("."));
        let mut tests = Vec::new();
        let mut names = std::collections::BTreeSet::new();
        for (i, t) in v["tests"].as_array().ok_or_else(|| anyhow!("{}: no \"tests\" list", path.display()))?.iter().enumerate() {
            let kind = t["kind"].as_str().ok_or_else(|| anyhow!("test {}: no \"kind\" (loss, code, qa or judge)", i + 1))?.to_string();
            if !["loss", "code", "qa", "judge"].contains(&kind.as_str()) {
                bail!("test {}: kind {kind:?} is not loss, code, qa or judge", i + 1);
            }
            let name = t["name"].as_str().map(str::to_string).unwrap_or_else(|| format!("{kind}-{}", i + 1));
            if !names.insert(name.clone()) {
                bail!("two tests are named {name:?}");
            }
            let data = PathBuf::from(t["data"].as_str().ok_or_else(|| anyhow!("test {name}: no \"data\" file"))?);
            let data = if data.is_absolute() { data } else { base.join(data) };
            let n = |k: &str| t[k].as_u64();
            tests.push(TestSpec { name, kind, format: t["format"].as_str().unwrap_or("").to_string(), data, limit: n("limit").map(|x| x as usize),
                                  max_tokens: n("max_tokens").map(|x| x as usize), language: t["language"].as_str().map(str::to_string),
                                  cpu_seconds: n("cpu_seconds"), memory_mb: n("memory_mb") });
        }
        if tests.is_empty() {
            bail!("{}: the suite has no tests", path.display());
        }
        Ok(Suite { path: path.to_path_buf(), name: v["name"].as_str().unwrap_or("suite").to_string(),
                   judge_model: v["judge_model"].as_str().filter(|m| !m.is_empty()).map(str::to_string), tests })
    }
}

// -------------------------------------------------------------------------------------------------------- results

/// One test at one pass count.
#[derive(Debug, Clone)]
pub struct TestResult {
    pub test: String,
    pub kind: String,
    pub passes: usize,
    pub metric: String,
    pub higher_is_better: bool,
    pub value: Option<f64>,
    pub ci: Option<(f64, f64)>,
    /// (item id, score)
    pub items: Vec<(String, f64)>,
    pub skipped: usize,
    pub extra: Value,
    pub error: Option<String>,
}

impl TestResult {
    fn new(spec: &TestSpec, passes: usize, metric: &str, higher: bool) -> Self {
        TestResult { test: spec.name.clone(), kind: spec.kind.clone(), passes, metric: metric.into(), higher_is_better: higher, value: None, ci: None,
                     items: vec![], skipped: 0, extra: json!({}), error: None }
    }

    fn finish(&mut self) {
        let xs: Vec<f64> = self.items.iter().map(|x| x.1).collect();
        if !xs.is_empty() {
            self.value = Some(mean(&xs));
            self.ci = Some(bootstrap_ci(&xs, BOOTSTRAP_ROUNDS, 7));
        }
    }

    pub fn to_json(&self) -> Value {
        json!({"test": self.test, "kind": self.kind, "passes": self.passes, "metric": self.metric, "higher_is_better": self.higher_is_better,
               "value": self.value, "ci": self.ci.map(|c| [c.0, c.1]), "n": self.items.len(), "skipped": self.skipped, "extra": self.extra,
               "error": self.error, "items": self.items.iter().map(|(id, s)| json!({"id": id, "score": s})).collect::<Vec<_>>()})
    }

    pub fn from_json(v: &Value) -> Option<TestResult> {
        Some(TestResult {
            test: v["test"].as_str()?.to_string(), kind: v["kind"].as_str().unwrap_or("").to_string(), passes: v["passes"].as_u64().unwrap_or(1) as usize,
            metric: v["metric"].as_str().unwrap_or("").to_string(), higher_is_better: v["higher_is_better"].as_bool().unwrap_or(true),
            value: v["value"].as_f64(), ci: v["ci"].as_array().and_then(|c| Some((c.first()?.as_f64()?, c.get(1)?.as_f64()?))),
            items: v["items"].as_array().into_iter().flatten().filter_map(|i| Some((i["id"].as_str()?.to_string(), i["score"].as_f64()?))).collect(),
            skipped: v["skipped"].as_u64().unwrap_or(0) as usize, extra: v["extra"].clone(), error: v["error"].as_str().map(str::to_string),
        })
    }
}

pub const BOOTSTRAP_ROUNDS: usize = 2000;

pub fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() { 0.0 } else { xs.iter().sum::<f64>() / xs.len() as f64 }
}

/// The 95 % percentile-bootstrap interval of the mean of `xs` (`rounds` resamples, seeded: the same input gives the
/// same interval).
pub fn bootstrap_ci(xs: &[f64], rounds: usize, seed: u64) -> (f64, f64) {
    if xs.len() < 2 {
        let m = mean(xs);
        return (m, m);
    }
    let mut d = Draw::new(seed);
    let n = xs.len();
    let mut means: Vec<f64> = (0..rounds.max(10)).map(|_| (0..n).map(|_| xs[(d.next() % n as u64) as usize]).sum::<f64>() / n as f64).collect();
    means.sort_by(|a, b| a.total_cmp(b));
    let at = |q: f64| means[((q * (means.len() - 1) as f64).round() as usize).min(means.len() - 1)];
    (at(0.025), at(0.975))
}

/// A test against an earlier result for it: items paired by id; `delta` and its interval are oriented so that a
/// positive number is better.
#[derive(Debug, Clone)]
pub struct Comparison {
    pub paired: usize,
    pub before: f64,
    pub after: f64,
    pub delta: f64,
    pub ci: (f64, f64),
    pub regressed: bool,
    pub improved: bool,
}

pub fn compare(new: &TestResult, old: &TestResult) -> Option<Comparison> {
    let old_by: BTreeMap<&str, f64> = old.items.iter().map(|(i, s)| (i.as_str(), *s)).collect();
    let pairs: Vec<(f64, f64)> = new.items.iter().filter_map(|(i, s)| old_by.get(i.as_str()).map(|o| (*s, *o))).collect();
    if pairs.is_empty() {
        return None;
    }
    let sign = if new.higher_is_better { 1.0 } else { -1.0 };
    let diffs: Vec<f64> = pairs.iter().map(|(n, o)| sign * (n - o)).collect();
    let ci = bootstrap_ci(&diffs, BOOTSTRAP_ROUNDS, 11);
    let (after, before) = (mean(&pairs.iter().map(|p| p.0).collect::<Vec<_>>()), mean(&pairs.iter().map(|p| p.1).collect::<Vec<_>>()));
    // one pair says nothing about noise: it is never called a change
    let judged = pairs.len() >= 2;
    Some(Comparison { paired: pairs.len(), before, after, delta: mean(&diffs), ci, regressed: judged && ci.1 < 0.0, improved: judged && ci.0 > 0.0 })
}

// ---------------------------------------------------------------------------------------------------------- run

/// What to run.
#[derive(Debug, Clone)]
pub struct Options {
    pub model: PathBuf,
    pub adapter: Option<PathBuf>,
    /// pass counts to run a looped model at (empty: its own)
    pub passes: Vec<usize>,
    pub suite: PathBuf,
    pub out: PathBuf,
    /// earlier reports (report.json or their folders); each test is compared with the best of them
    pub against: Vec<PathBuf>,
    pub devices: Vec<String>,
    pub base_dtype: Option<BaseDtype>,
    /// prompts longer than this are left out
    pub max_prompt_tokens: usize,
}

impl Options {
    pub fn new(model: impl Into<PathBuf>, suite: impl Into<PathBuf>, out: impl Into<PathBuf>) -> Self {
        Options { model: model.into(), adapter: None, passes: vec![], suite: suite.into(), out: out.into(), against: vec![], devices: vec![],
                  base_dtype: None, max_prompt_tokens: 4096 }
    }
}

/// The finished report.
#[derive(Debug, Clone)]
pub struct Report {
    pub results: Vec<TestResult>,
    /// (result index, the report it was compared with, the comparison)
    pub comparisons: Vec<(usize, String, Comparison)>,
    pub json: Value,
}

impl Report {
    pub fn regressions(&self) -> Vec<&(usize, String, Comparison)> {
        self.comparisons.iter().filter(|c| c.2.regressed).collect()
    }
}

fn report_file(p: &Path) -> PathBuf {
    if p.is_dir() { p.join("report.json") } else { p.to_path_buf() }
}

/// Run a suite on a model (and its adapter), at each pass count, and write `report.json`, `report.md` and
/// `answers.jsonl` into `o.out`. `judge` scores the judge tests (None: they report an error).
pub fn run(o: &Options, judge: Option<&dyn Judge>, emit: &mut dyn FnMut(Value), should_stop: &dyn Fn() -> bool) -> Result<Report> {
    let suite = Suite::load(&o.suite)?;
    let tok = ChatTokenizer::load(&o.model, None)?;
    let devs = crate::finetune::open_devices(&o.devices)?;
    let dtype = o.base_dtype.unwrap_or(if devs.iter().any(|d| !d.is_cpu()) { BaseDtype::BF16 } else { BaseDtype::F32 });
    let mut model = LlamaModel::load_on(&o.model, dtype, &devs)?;
    if let Some(a) = &o.adapter {
        model.load_adapter(a).with_context(|| format!("the adapter {}", a.display()))?;
    }
    let own = model.passes();
    let passes = if o.passes.is_empty() { vec![own] } else { o.passes.clone() };
    for p in &passes {
        if *p == 0 || *p > own {
            bail!("--passes {p}: this model runs 1{}", if own > 1 { format!(" to {own}") } else { " (it has no loop)".into() });
        }
    }
    let runner = CodeRunner::detect();
    std::fs::create_dir_all(&o.out).with_context(|| format!("{}", o.out.display()))?;
    let mut answers = String::new();
    let mut results = Vec::new();
    // the items each test reads, loaded once
    let total_tests = suite.tests.len() * passes.len();
    let mut done_tests = 0;
    for &p in &passes {
        model.set_passes(p)?;
        let gen = Generator::new(&model, &tok, o.max_prompt_tokens);
        for spec in &suite.tests {
            if should_stop() {
                bail!("stopped");
            }
            done_tests += 1;
            emit(json!({"event": "log", "message": format!("[{done_tests}/{total_tests}] {} ({}) at {p} pass{}", spec.name, spec.kind, if p == 1 { "" } else { "es" })}));
            let mut progress = |i: usize, n: usize| emit(json!({"event": "progress", "done": i, "total": n, "message": format!("{} at {p} pass{}", spec.name, if p == 1 { "" } else { "es" })}));
            let r = run_test(spec, p, &gen, &runner, judge, &mut progress, &mut answers, should_stop);
            let r = r.unwrap_or_else(|e| {
                let mut t = TestResult::new(spec, p, "", true);
                t.error = Some(format!("{e:#}"));
                t
            });
            if let Some(e) = &r.error {
                emit(json!({"event": "log", "message": format!("{}: {e}", spec.name)}));
            } else {
                emit(json!({"event": "log", "message": format!("{}: {} {} over {} items", spec.name, r.metric, r.value.map_or("-".into(), |v| format!("{v:.4}")), r.items.len())}));
            }
            results.push(r);
        }
    }
    std::fs::write(o.out.join("answers.jsonl"), &answers)?;
    // against the best earlier result of each test
    let mut earlier: Vec<(String, Vec<TestResult>)> = Vec::new();
    for a in &o.against {
        let f = report_file(a);
        let v: Value = serde_json::from_str(&std::fs::read_to_string(&f).with_context(|| format!("--against {}", f.display()))?)
            .with_context(|| format!("{}: not a report", f.display()))?;
        earlier.push((f.display().to_string(), v["results"].as_array().into_iter().flatten().filter_map(TestResult::from_json).collect()));
    }
    let mut comparisons = Vec::new();
    for (i, r) in results.iter().enumerate() {
        let best = earlier.iter().flat_map(|(f, rs)| rs.iter().map(move |x| (f, x)))
            .filter(|(_, x)| x.test == r.test && x.passes == r.passes && x.metric == r.metric && x.value.is_some())
            .max_by(|a, b| {
                let (va, vb) = (a.1.value.unwrap(), b.1.value.unwrap());
                if r.higher_is_better { va.total_cmp(&vb) } else { vb.total_cmp(&va) }
            });
        if let Some((f, old)) = best {
            if let Some(c) = compare(r, old) {
                comparisons.push((i, f.clone(), c));
            }
        }
    }
    let report_json = report_to_json(o, &suite, &results, &comparisons, runner.can(CodeLang::Python), judge.map(|j| j.name()));
    std::fs::write(o.out.join("report.json"), serde_json::to_string_pretty(&report_json)?)?;
    std::fs::write(o.out.join("report.md"), report_markdown(&report_json))?;
    Ok(Report { results, comparisons, json: report_json })
}

fn report_to_json(o: &Options, suite: &Suite, results: &[TestResult], comps: &[(usize, String, Comparison)], python: bool, judge: Option<String>) -> Value {
    let comp_json = |c: &Comparison, f: &str| json!({"against": f, "paired": c.paired, "before": c.before, "after": c.after, "delta": c.delta,
                                                      "ci": [c.ci.0, c.ci.1], "regressed": c.regressed, "improved": c.improved});
    let mut rs: Vec<Value> = results.iter().map(|r| r.to_json()).collect();
    for (i, f, c) in comps {
        rs[*i]["compared"] = comp_json(c, f);
    }
    let regressions: Vec<Value> = comps.iter().filter(|c| c.2.regressed).map(|(i, f, c)| {
        let r = &results[*i];
        json!({"test": r.test, "passes": r.passes, "metric": r.metric, "before": c.before, "after": c.after, "delta": c.delta, "ci": [c.ci.0, c.ci.1], "against": f})
    }).collect();
    json!({
        "model": o.model.display().to_string(), "adapter": o.adapter.as_ref().map(|a| a.display().to_string()),
        "suite": suite.path.display().to_string(), "suite_name": suite.name, "passes": results.iter().map(|r| r.passes).collect::<std::collections::BTreeSet<_>>(),
        "created": std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0),
        "sandbox": {"network_isolated": sandbox::network_isolation_available(), "python": python, "limits": {"cpu_seconds": Limits::run().cpu_seconds, "memory_bytes": Limits::run().memory_bytes}},
        "judge": judge, "against": o.against.iter().map(|a| report_file(a).display().to_string()).collect::<Vec<_>>(),
        "bootstrap": {"rounds": BOOTSTRAP_ROUNDS, "level": 0.95, "rule": "regressed: the paired 95% interval of the change lies wholly on the worse side of zero"},
        "results": rs, "regressions": regressions,
    })
}

/// The report as a Markdown table, regressions first.
pub fn report_markdown(r: &Value) -> String {
    let mut m = format!("# Regression report: {}\n\nModel `{}`{}; suite `{}`.\n\n", r["suite_name"].as_str().unwrap_or(""), r["model"].as_str().unwrap_or(""),
                        r["adapter"].as_str().map_or(String::new(), |a| format!(" with adapter `{a}`")), r["suite"].as_str().unwrap_or(""));
    let regs = r["regressions"].as_array().cloned().unwrap_or_default();
    if regs.is_empty() {
        m.push_str(if r["against"].as_array().is_some_and(|a| !a.is_empty()) { "**No regressions** beyond noise.\n\n" } else { "No earlier report to compare with.\n\n" });
    } else {
        m.push_str(&format!("**{} regression{}:**\n\n", regs.len(), if regs.len() == 1 { "" } else { "s" }));
        for g in &regs {
            m.push_str(&format!("- {} at {} pass(es): {} {:.4} → {:.4} (change {:+.4}, 95% interval {:+.4} to {:+.4})\n", g["test"].as_str().unwrap_or(""), g["passes"],
                                g["metric"].as_str().unwrap_or(""), g["before"].as_f64().unwrap_or(0.0), g["after"].as_f64().unwrap_or(0.0),
                                g["delta"].as_f64().unwrap_or(0.0), g["ci"][0].as_f64().unwrap_or(0.0), g["ci"][1].as_f64().unwrap_or(0.0)));
        }
        m.push('\n');
    }
    m.push_str("| test | passes | metric | value | 95% interval | n | skipped | vs earlier |\n|---|---|---|---|---|---|---|---|\n");
    let f = |v: &Value| v.as_f64().map_or("-".to_string(), |x| format!("{x:.4}"));
    for t in r["results"].as_array().into_iter().flatten() {
        let vs = match &t["compared"] {
            Value::Null => String::new(),
            c if c["regressed"] == true => format!("**REGRESSED** {:+.4}", c["delta"].as_f64().unwrap_or(0.0)),
            c if c["improved"] == true => format!("better {:+.4}", c["delta"].as_f64().unwrap_or(0.0)),
            c => format!("same within noise ({:+.4})", c["delta"].as_f64().unwrap_or(0.0)),
        };
        let value = match t["error"].as_str() { Some(e) => format!("error: {}", e.chars().take(80).collect::<String>()), None => f(&t["value"]) };
        m.push_str(&format!("| {} | {} | {} | {} | {} to {} | {} | {} | {} |\n", t["test"].as_str().unwrap_or(""), t["passes"], t["metric"].as_str().unwrap_or(""),
                            value, f(&t["ci"][0]), f(&t["ci"][1]), t["n"], t["skipped"], vs));
    }
    m.push_str(&format!("\nSandbox: network {}; code runs with CPU and memory limits in its own process group.\n",
                        if r["sandbox"]["network_isolated"] == true { "isolated (own namespace)" } else { "NOT isolated (this kernel refuses an unprivileged network namespace)" }));
    m
}

fn limited<T>(mut v: Vec<T>, limit: Option<usize>) -> Vec<T> {
    if let Some(n) = limit {
        v.truncate(n);
    }
    v
}

#[allow(clippy::too_many_arguments)]
fn run_test(spec: &TestSpec, passes: usize, gen: &Generator, runner: &CodeRunner, judge: Option<&dyn Judge>, progress: &mut dyn FnMut(usize, usize),
            answers: &mut String, should_stop: &dyn Fn() -> bool) -> Result<TestResult> {
    let mut log_answer = |id: &str, a: &str, score: f64, note: &str| {
        answers.push_str(&format!("{}\n", json!({"test": spec.name, "passes": passes, "id": id, "answer": a, "score": score, "note": note})));
    };
    match spec.kind.as_str() {
        "loss" => {
            let mut r = TestResult::new(spec, passes, "loss", false);
            let (records, bad) = crate::chatdata::read_jsonl(&spec.data, DataFormat::from_s(if spec.format.is_empty() { "auto" } else { &spec.format })?)?;
            r.skipped += bad.len();
            let records = limited(records, spec.limit);
            for (i, rec) in records.iter().enumerate() {
                if should_stop() { bail!("stopped"); }
                match gen.tok.encode(rec, gen.max_prompt, false) {
                    Ok(ex) => {
                        let l = gen.model.loss(&ex.tokens, &ex.supervised, &Ctx::eval())?.to_dtype(DType::F32)?.to_scalar::<f32>()? as f64;
                        r.items.push((i.to_string(), l));
                    }
                    Err(_) => r.skipped += 1,
                }
                progress(i + 1, records.len());
            }
            r.finish();
            Ok(r)
        }
        "code" => {
            let mut r = TestResult::new(spec, passes, "pass@1", true);
            let lang = spec.language.as_deref().map(CodeLang::from_s).transpose()?.unwrap_or(CodeLang::Python);
            let (tasks, skipped, why) = load_code(&spec.data, &spec.format, lang)?;
            r.skipped = skipped;
            let mut run = runner.clone();
            if let Some(c) = spec.cpu_seconds { run.run.cpu_seconds = c; run.run.wall_seconds = 2.0 * c as f64 + 5.0; }
            if let Some(mb) = spec.memory_mb { run.run.memory_bytes = mb << 20; }
            let tasks = limited(tasks, spec.limit);
            let mut endings: BTreeMap<String, usize> = BTreeMap::new();
            let max_new = spec.max_tokens.unwrap_or(512);
            let mut partial = Vec::new();
            for (i, t) in tasks.iter().enumerate() {
                if should_stop() { bail!("stopped"); }
                if !run.can(t.lang) {
                    r.skipped += 1;
                    *endings.entry(format!("no {:?} tool", t.lang)).or_default() += 1;
                    continue;
                }
                match gen.answer(&t.prompt, max_new, None)? {
                    Some(a) => {
                        let g = run.grade(t, &a);
                        *endings.entry(g.ending.clone()).or_default() += 1;
                        r.items.push((t.id.clone(), if g.passed { 1.0 } else { 0.0 }));
                        partial.push(g.fraction);
                        log_answer(&t.id, &a, if g.passed { 1.0 } else { 0.0 }, &g.ending);
                    }
                    None => r.skipped += 1,
                }
                progress(i + 1, tasks.len());
            }
            r.extra = json!({"endings": endings, "tests_passed_share": mean(&partial), "skipped_why": why});
            r.finish();
            Ok(r)
        }
        "qa" => {
            let mut r = TestResult::new(spec, passes, "f1", true);
            let (items, unanswerable) = load_qa(&spec.data, &spec.format)?;
            r.skipped = unanswerable;
            let items = limited(items, spec.limit);
            let mut ems = Vec::new();
            for (i, q) in items.iter().enumerate() {
                if should_stop() { bail!("stopped"); }
                match gen.answer(&qa_prompt(q), spec.max_tokens.unwrap_or(48), None)? {
                    Some(a) => {
                        let (em, f1) = qa_score(&a, &q.answers);
                        ems.push(em);
                        r.items.push((q.id.clone(), f1));
                        log_answer(&q.id, &a, f1, "");
                    }
                    None => r.skipped += 1,
                }
                progress(i + 1, items.len());
            }
            r.extra = json!({"exact_match": mean(&ems), "unanswerable_left_out": unanswerable});
            r.finish();
            Ok(r)
        }
        "judge" => {
            let mut r = TestResult::new(spec, passes, "judge", true);
            let judge = judge.ok_or_else(|| anyhow!("no judge: the server's API is not reachable (AM2STRATA_URL, AM2STRATA_API_KEY)"))?;
            let items = limited(load_judge(&spec.data)?, spec.limit);
            let mut unscored = 0;
            for (i, it) in items.iter().enumerate() {
                if should_stop() { bail!("stopped"); }
                match gen.answer(&it.prompt, spec.max_tokens.unwrap_or(256), None)? {
                    Some(a) => {
                        let sc = judge.score(&judge_request(it, &a))?;
                        if sc == 0 { unscored += 1; } else { r.items.push((it.id.clone(), sc as f64)); }
                        log_answer(&it.id, &a, sc as f64, "");
                    }
                    None => r.skipped += 1,
                }
                progress(i + 1, items.len());
            }
            r.skipped += unscored;
            r.extra = json!({"judge": judge.name(), "unscored": unscored, "scale": "1-5"});
            r.finish();
            Ok(r)
        }
        k => bail!("kind {k:?}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn python_literals_read_as_json() {
        let v = python_literal(r#"{'a': 'it\'s', "b": [1, 2.5, True, None], 'c': ('x',), 'd': '\né'}"#).unwrap();
        assert_eq!(v, json!({"a": "it's", "b": [1, 2.5, true, null], "c": ["x"], "d": "\né"}));
        assert!(python_literal("{'a': ").is_err());
    }

    #[test]
    fn qa_scores_match_and_overlap() {
        assert_eq!(qa_score("The Blue sky.", &["blue sky".into()]), (1.0, 1.0));
        let (em, f1) = qa_score("very blue sky", &["blue sky".into()]);
        assert_eq!(em, 0.0);
        assert!((f1 - 0.8).abs() < 1e-9);
        assert_eq!(qa_score("दिल्ली।", &["दिल्ली".into()]).0, 1.0, "the danda is punctuation");
    }

    #[test]
    fn code_is_taken_from_the_fence_in_its_language() {
        let a = "Sure:\n```text\nnot this\n```\n```python\ndef f():\n    return 1\n```\n";
        assert_eq!(extract_code(a, CodeLang::Python), "def f():\n    return 1\n");
        assert_eq!(extract_code("<think>```python\nx\n```</think>plain", CodeLang::Python), "plain");
    }
}
