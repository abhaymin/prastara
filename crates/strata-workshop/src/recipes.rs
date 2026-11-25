//! The Workshop's built-in data recipes: **code**, **indic** and **translate**. Each carries its own catalogue (task
//! kinds, programming or Indian languages, topics), compiled into the binary, so it needs no seed file; documents are
//! optional and, when given, ground the requests in them (your own code, your own texts to translate).
//!
//! Each request draws one plan from the catalogue (the same `--seed` gives the same plans), and each recipe has its
//! own quality checks on top of the common filters:
//! - **code**: a task that asks for code must answer with a fenced code block; with `--check-code`, every
//!   self-contained Rust block (standard library only) is compiled with `rustc` and the record is dropped if it fails.
//! - **indic**: the text must really be in the language's script (Devanagari for Hindi, Tamil script for Tamil, …),
//!   not transliterated into Latin letters.
//! - **translate**: the translation must be in the target's script, of a plausible length for its source, and not a
//!   copy of it; with a judge pass, the model scores how faithful each translation is.

use std::path::Path;

use serde_json::Value;

use crate::dataset::Record;

// ------------------------------------------------------------------------------------------------------ catalogue

/// An Indian language the indic and translate recipes write: its name, its name in itself, and its script's
/// Unicode block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Lang {
    pub code: &'static str,
    pub name: &'static str,
    pub native: &'static str,
    pub script: &'static str,
    pub block: (u32, u32),
}

const DEVANAGARI: (u32, u32) = (0x0900, 0x097F);

pub const LANGS: &[Lang] = &[
    Lang { code: "hi", name: "Hindi", native: "हिन्दी", script: "Devanagari", block: DEVANAGARI },
    Lang { code: "bn", name: "Bengali", native: "বাংলা", script: "Bengali", block: (0x0980, 0x09FF) },
    Lang { code: "mr", name: "Marathi", native: "मराठी", script: "Devanagari", block: DEVANAGARI },
    Lang { code: "te", name: "Telugu", native: "తెలుగు", script: "Telugu", block: (0x0C00, 0x0C7F) },
    Lang { code: "ta", name: "Tamil", native: "தமிழ்", script: "Tamil", block: (0x0B80, 0x0BFF) },
    Lang { code: "gu", name: "Gujarati", native: "ગુજરાતી", script: "Gujarati", block: (0x0A80, 0x0AFF) },
    Lang { code: "ur", name: "Urdu", native: "اردو", script: "Arabic (Nastaliq)", block: (0x0600, 0x06FF) },
    Lang { code: "kn", name: "Kannada", native: "ಕನ್ನಡ", script: "Kannada", block: (0x0C80, 0x0CFF) },
    Lang { code: "or", name: "Odia", native: "ଓଡ଼ିଆ", script: "Odia", block: (0x0B00, 0x0B7F) },
    Lang { code: "ml", name: "Malayalam", native: "മലയാളം", script: "Malayalam", block: (0x0D00, 0x0D7F) },
    Lang { code: "pa", name: "Punjabi", native: "ਪੰਜਾਬੀ", script: "Gurmukhi", block: (0x0A00, 0x0A7F) },
    Lang { code: "as", name: "Assamese", native: "অসমীয়া", script: "Bengali-Assamese", block: (0x0980, 0x09FF) },
    Lang { code: "sa", name: "Sanskrit", native: "संस्कृतम्", script: "Devanagari", block: DEVANAGARI },
    Lang { code: "ne", name: "Nepali", native: "नेपाली", script: "Devanagari", block: DEVANAGARI },
    Lang { code: "kok", name: "Konkani", native: "कोंकणी", script: "Devanagari", block: DEVANAGARI },
];

/// The languages a `--language` value names: codes or names, comma-separated (empty: all of them).
pub fn langs(spec: Option<&str>) -> Result<Vec<Lang>, String> {
    let spec = spec.unwrap_or("").trim();
    if spec.is_empty() || spec.eq_ignore_ascii_case("all") {
        return Ok(LANGS.to_vec());
    }
    let mut out = vec![];
    for w in spec.split(',').map(str::trim).filter(|w| !w.is_empty()) {
        let l = LANGS.iter().find(|l| l.code.eq_ignore_ascii_case(w) || l.name.eq_ignore_ascii_case(w) || l.native == w)
            .ok_or_else(|| format!("{w:?} is not one of the Indian languages: {}",
                                   LANGS.iter().map(|l| format!("{} ({})", l.name, l.code)).collect::<Vec<_>>().join(", ")))?;
        if !out.contains(l) { out.push(*l); }
    }
    Ok(out)
}

/// Programming languages for the code recipe, with how often each is drawn (Rust most: Prastara is all Rust).
pub const CODE_LANGS: &[(&str, u32)] = &[("Rust", 6), ("Python", 2), ("TypeScript", 2), ("Go", 1), ("C", 1), ("SQL", 1),
                                         ("Bash", 1), ("HTML and CSS", 1)];

/// Kinds of coding task, and whether the answer must contain code.
pub const CODE_KINDS: &[(&str, bool)] = &[
    ("write a function or small module that does the task, with a short explanation", true),
    ("find and fix the bug in a short piece of code the user pastes (include the buggy code in the request)", true),
    ("explain what a piece of code the user pastes does, step by step (include the code in the request)", false),
    ("write unit tests for a function the user pastes (include the function in the request)", true),
    ("refactor code the user pastes to be clearer or faster, and say why (include the code in the request)", true),
    ("review code the user pastes: correctness, edge cases, safety and style (include the code in the request)", false),
    ("design question: compare two approaches and recommend one, with a short code sketch", true),
    ("a compiler or runtime error message the user hits and how to fix it (include the error and the code)", true),
    ("a command-line tool: parse arguments, read input, print results", true),
    ("an agentic task: plan the steps, name the tool calls (read a file, run a command, edit a file) and the final change", false),
];

pub const CODE_TOPICS: &[&str] = &[
    "string parsing", "file input and output", "error handling", "iterators and collections", "hash maps",
    "sorting and searching", "binary search", "graphs and shortest paths", "dynamic programming", "recursion",
    "concurrency with threads and channels", "async networking", "an HTTP JSON API", "a REST client with retries",
    "a rate limiter", "an LRU cache", "a tokenizer for a small language", "a ring buffer", "matrix maths",
    "date and time handling", "Unicode text and Indic scripts", "a small state machine", "command-line arguments",
    "config files (TOML, JSON)", "serialization", "logging and metrics", "a test harness", "property tests",
    "memory safety and lifetimes", "traits and generics", "pattern matching", "a binary file format", "checksums and hashing",
    "a job queue", "streaming large files", "a key-value store", "SQL queries and indexes", "shell scripting a build",
    "a web page form with validation", "performance profiling", "numerical stability (softmax, log-sum-exp)",
    "a tiny neural network layer", "quantizing numbers to 8 bits", "a chat template renderer", "a server-sent events stream",
];

/// Kinds of native writing for the indic recipe.
pub const INDIC_KINDS: &[&str] = &[
    "a clear explanation of a concept for a curious adult",
    "a simple explanation for a ten-year-old",
    "a short original story (150-300 words) with a title",
    "a short original poem with a title",
    "a polite formal letter or email",
    "an everyday conversation between two people (a few lines each)",
    "practical step-by-step instructions",
    "the meaning and use of a well-known proverb or saying in this language, with an example",
    "a short summary of a paragraph the user gives (write the paragraph in the request, in the same language)",
    "an answer to a factual question, with the reasoning",
    "advice on a common everyday problem",
    "a short news-style report about an invented local event",
];

pub const INDIC_TOPICS: &[&str] = &[
    "monsoon and farming", "a village fair", "family and festivals", "school and exams", "trains and travel",
    "cooking a regional dish", "health and sleep", "saving money", "the night sky", "rivers and water",
    "mobile phones and the internet", "a computer program and how it works", "the human body", "kindness to animals",
    "a market day", "classical music and dance", "cricket", "a historical monument", "trees and the environment",
    "learning a new skill", "a library", "electricity at home", "the post office", "a wedding", "rain and childhood",
    "a temple, mosque, church or gurdwara visit", "first day at work", "a letter to a grandparent", "yoga and exercise",
    "science: why the sky is blue", "maths: fractions", "safe use of money apps", "a river journey", "a lost and found pet",
];

/// Kinds of text the translate recipe writes when it has no documents.
pub const TRANSLATE_KINDS: &[&str] = &[
    "an everyday sentence or two", "a short paragraph from a news report", "a customer support message",
    "instructions from a product manual", "a line of dialogue from a story", "a polite request or invitation",
    "a short explanation from a school textbook", "a government or bank notice", "a medical appointment note",
    "a software error message and help text", "a recipe step", "a travel question and answer",
];

/// The ways a translation request is worded (varied so the model learns the task, not one phrasing).
const TRANSLATE_ASKS: &[&str] = &[
    "Translate into {to}:\n\n{text}",
    "Please translate this {from} text into {to}.\n\n{text}",
    "{text}\n\nTranslate the above into {to}.",
    "What is this in {to}?\n\n{text}",
    "Translate from {from} to {to}, keeping the meaning and tone:\n\n{text}",
];

// ---------------------------------------------------------------------------------------------------- the draw

/// A small fixed pseudo-random generator (splitmix64): the same seed gives the same plans on every machine.
#[derive(Debug, Clone)]
pub struct Draw(u64);

impl Draw {
    pub fn new(seed: u64) -> Draw { Draw(seed ^ 0x9E37_79B9_7F4A_7C15) }
    pub fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    pub fn pick<T: Copy>(&mut self, xs: &[T]) -> T { xs[(self.next() % xs.len() as u64) as usize] }
    fn weighted(&mut self, xs: &[(&'static str, u32)]) -> &'static str {
        let total: u32 = xs.iter().map(|x| x.1).sum();
        let mut r = (self.next() % total as u64) as u32;
        for (x, w) in xs {
            if r < *w { return x; }
            r -= w;
        }
        xs[0].0
    }
}

/// One request's plan, drawn from the catalogue.
#[derive(Debug, Clone)]
pub enum Plan {
    Code { lang: &'static str, kind: &'static str, needs_code: bool, topic: &'static str },
    Indic { lang: Lang, kind: &'static str, topic: &'static str },
    /// `to_indic`: English into `lang` (else `lang` into English)
    Translate { lang: Lang, to_indic: bool, kind: &'static str, topic: &'static str },
}

impl Plan {
    /// For the record's meta.
    pub fn meta(&self) -> Value {
        match self {
            Plan::Code { lang, topic, .. } => serde_json::json!({"language": lang, "topic": topic}),
            Plan::Indic { lang, kind, topic } => serde_json::json!({"language": lang.code, "kind": kind, "topic": topic}),
            Plan::Translate { lang, to_indic, .. } => serde_json::json!({
                "from": if *to_indic { "en" } else { lang.code }, "to": if *to_indic { lang.code } else { "en" }}),
        }
    }
}

pub fn draw_code(d: &mut Draw, langs_override: Option<&str>) -> Plan {
    let lang: &'static str = match langs_override.map(str::trim).filter(|s| !s.is_empty()) {
        Some(spec) => {
            let named: Vec<&'static str> = CODE_LANGS.iter().map(|x| x.0)
                .filter(|l| spec.split(',').any(|w| w.trim().eq_ignore_ascii_case(l))).collect();
            if named.is_empty() { d.weighted(CODE_LANGS) } else { d.pick(&named) }
        }
        None => d.weighted(CODE_LANGS),
    };
    let (kind, needs_code) = d.pick(CODE_KINDS);
    Plan::Code { lang, kind, needs_code, topic: d.pick(CODE_TOPICS) }
}

pub fn draw_indic(d: &mut Draw, langs: &[Lang]) -> Plan {
    Plan::Indic { lang: d.pick(langs), kind: d.pick(INDIC_KINDS), topic: d.pick(INDIC_TOPICS) }
}

/// The direction alternates by request (half into the Indian language, half out of it).
pub fn draw_translate(d: &mut Draw, langs: &[Lang], i: usize) -> Plan {
    Plan::Translate { lang: d.pick(langs), to_indic: i % 2 == 0, kind: d.pick(TRANSLATE_KINDS), topic: d.pick(INDIC_TOPICS) }
}

// ------------------------------------------------------------------------------------------------------ prompts

/// The request for one plan, writing `n` examples; `chunk` is a piece of a document when there are documents.
pub fn prompt(plan: &Plan, n: usize, chunk: Option<&str>, topic: Option<&str>) -> String {
    let focus = topic.map(|t| format!(" Focus on: {t}.")).unwrap_or_default();
    match plan {
        Plan::Code { lang, kind, needs_code, topic: t } => {
            let code_rule = if *needs_code {
                " Every answer must contain the complete code in a fenced block marked with the language (```rust, \
                 ```python, …); Rust must compile on stable Rust, and use only the standard library unless the request \
                 names a crate."
            } else { "" };
            match chunk {
                Some(src) => format!(
                    "Below is real code from a project. Write {n} realistic requests a developer working on this code \
                     might make, of this kind: {kind}. The requests must be about this code (quote the parts they need) \
                     and the answers must be correct for it: expert, precise, and concise.{code_rule}{focus}\n\nAnswer \
                     as a JSON array: [{{\"question\": \"...\", \"answer\": \"...\"}}]\n\nCODE:\n{src}"),
                None => format!(
                    "Write {n} realistic, varied programming requests in {lang} about {t}, of this kind: {kind}. Make \
                     them like real users' requests (some short, some detailed, a range of difficulty) and write expert \
                     answers: correct, idiomatic, with edge cases handled, and concise.{code_rule}{focus}\n\nAnswer as a \
                     JSON array: [{{\"question\": \"...\", \"answer\": \"...\"}}]"),
            }
        }
        Plan::Indic { lang, kind, topic: t } => format!(
            "Write {n} examples of a user asking an assistant for {kind}, about {t}. Both the request and the answer \
             must be written entirely in {name} ({native}) in the {script} script, never transliterated into Latin \
             letters; use natural, fluent, correct {name} as a native speaker writes it, not a word-by-word \
             translation from English. Vary the wording and length.{focus}\n\nAnswer as a JSON array: \
             [{{\"question\": \"...\", \"answer\": \"...\"}}]",
            name = lang.name, native = lang.native, script = lang.script),
        Plan::Translate { lang, to_indic, kind, topic: t } => {
            let (from, to) = if *to_indic { ("English".to_string(), lang_label(lang)) } else { (lang_label(lang), "English".to_string()) };
            match chunk {
                Some(src) => format!(
                    "Split the text below into {n} passages of one to four sentences (in order, skipping headings and \
                     anything that is not prose) and translate each passage into {to}. The source passage must be copied \
                     exactly from the text; the translation must be faithful, complete and natural, as a professional \
                     translator writes it, in the {script} script.{focus}\n\nAnswer as a JSON array: \
                     [{{\"source\": \"...\", \"translation\": \"...\"}}]\n\nTEXT:\n{src}",
                    script = if *to_indic { lang.script } else { "Latin" }),
                None => format!(
                    "Write {n} different passages in {from} — each {kind}, about {t} — and a translation of each into \
                     {to}. The passages must be natural {from} as a native speaker writes it{src_script}; the \
                     translations faithful, complete and natural, as a professional translator writes them{tgt_script}. \
                     Vary length (one to five sentences) and register.{focus}\n\nAnswer as a JSON array: \
                     [{{\"source\": \"...\", \"translation\": \"...\"}}]",
                    src_script = if *to_indic { String::new() } else { format!(", in the {} script", lang.script) },
                    tgt_script = if *to_indic { format!(", in the {} script", lang.script) } else { String::new() }),
            }
        }
    }
}

fn lang_label(l: &Lang) -> String { format!("{} ({})", l.name, l.native) }

/// The judge's request for a translation: how faithful and fluent it is, 1–5.
pub fn judge_translation(source: &str, translation: &str, to: &str) -> String {
    format!(
        "Score this translation into {to} from 1 (wrong, missing parts, or not in {to}) to 5 (faithful, complete and \
         natural). Answer as JSON: {{\"score\": N, \"why\": \"...\"}}\n\nSOURCE:\n{source}\n\nTRANSLATION:\n{translation}")
}

// ------------------------------------------------------------------------------------------------------ records

/// The records in a reply for a plan: question–answer pairs, or source–translation pairs made into a translation
/// request (worded one of several ways) and its answer — each with the text it translates (None for the others).
pub fn records_from(plan: &Plan, v: &Value, d: &mut Draw) -> Vec<(Record, Option<String>)> {
    let items = match v { Value::Array(a) => a.clone(), other => vec![other.clone()] };
    let mut out = vec![];
    for it in items {
        let pair = match plan {
            Plan::Translate { lang, to_indic, .. } => match (it["source"].as_str(), it["translation"].as_str()) {
                (Some(s), Some(t)) if !s.trim().is_empty() && !t.trim().is_empty() => {
                    let (from, to) = if *to_indic { ("English", lang.name) } else { (lang.name, "English") };
                    let ask = d.pick(TRANSLATE_ASKS).replace("{from}", from).replace("{to}", to).replace("{text}", s.trim());
                    Some((ask, t.trim().to_string(), Some(s.trim().to_string())))
                }
                _ => None,
            },
            _ => match (it["question"].as_str(), it["answer"].as_str()) {
                (Some(q), Some(a)) if !q.trim().is_empty() => Some((q.trim().to_string(), a.trim().to_string(), None)),
                _ => None,
            },
        };
        if let Some((q, a, src)) = pair {
            out.push((Record { messages: vec![("user".into(), q), ("assistant".into(), a)] }, src));
        }
    }
    out
}

// ------------------------------------------------------------------------------------------------------- checks

/// The share of a text's letters that are in a script's block (letters of other scripts count against it; digits,
/// punctuation, spaces and code — inside backticks or fences — do not count). 1.0 for text with no letters.
pub fn script_share(text: &str, block: (u32, u32)) -> f64 {
    let (mut inside, mut letters) = (0usize, 0usize);
    let mut in_code = false;
    for c in text.chars() {
        if c == '`' { in_code = !in_code; continue; }
        if in_code { continue; }
        let u = c as u32;
        if (block.0..=block.1).contains(&u) {
            inside += 1;
            letters += 1;
        } else if c.is_alphabetic() {
            letters += 1;
        }
    }
    if letters == 0 { 1.0 } else { inside as f64 / letters as f64 }
}

/// How much of a text must be in the language's script (code, names and numbers may be in Latin letters).
pub const MIN_SCRIPT_SHARE: f64 = 0.7;

/// The fenced code blocks in an answer: (language tag, code).
pub fn code_blocks(answer: &str) -> Vec<(String, String)> {
    let mut out = vec![];
    let mut lines = answer.lines();
    while let Some(l) = lines.next() {
        let t = l.trim_start();
        if let Some(tag) = t.strip_prefix("```") {
            let mut code = String::new();
            let mut closed = false;
            for l in lines.by_ref() {
                if l.trim_start().starts_with("```") { closed = true; break; }
                code.push_str(l);
                code.push('\n');
            }
            if closed { out.push((tag.trim().to_lowercase(), code)); }
        }
    }
    out
}

/// Why a record fails its recipe's checks, or None. `source` is the text a translation translates; `rustc` is the
/// compiler to check Rust with (None: no compile).
pub fn check(plan: &Plan, r: &Record, source: Option<&str>, rustc: Option<&Path>) -> Option<&'static str> {
    let user = r.messages.iter().find(|m| m.0 == "user").map(|m| m.1.as_str()).unwrap_or("");
    let answer = r.messages.iter().rev().find(|m| m.0 == "assistant").map(|m| m.1.as_str()).unwrap_or("");
    match plan {
        Plan::Code { lang, needs_code, .. } => {
            let blocks = code_blocks(answer);
            if *needs_code && blocks.is_empty() {
                return Some("code: no fenced code block");
            }
            if let (Some(rustc), true) = (rustc, *lang == "Rust") {
                for (_, code) in blocks.iter().filter(|b| b.0 == "rust" || b.0 == "rs") {
                    if self_contained_rust(code) && !compiles(rustc, code) {
                        return Some("code: Rust does not compile");
                    }
                }
            }
            None
        }
        Plan::Indic { lang, .. } => {
            if script_share(user, lang.block) < MIN_SCRIPT_SHARE { return Some("indic: request not in the script"); }
            if script_share(answer, lang.block) < MIN_SCRIPT_SHARE { return Some("indic: answer not in the script"); }
            None
        }
        Plan::Translate { lang, to_indic, .. } => {
            let src = source.unwrap_or(user);
            let (src_ok, tgt_ok) = if *to_indic {
                (script_share(&src, lang.block) < 0.2, script_share(answer, lang.block) >= MIN_SCRIPT_SHARE)
            } else {
                (script_share(&src, lang.block) >= MIN_SCRIPT_SHARE, script_share(answer, lang.block) < 0.2)
            };
            if !src_ok { return Some("translate: source in the wrong script"); }
            if !tgt_ok { return Some("translate: translation in the wrong script"); }
            if src.trim() == answer.trim() { return Some("translate: copy of the source"); }
            let (a, b) = (src.chars().filter(|c| !c.is_whitespace()).count() as f64,
                          answer.chars().filter(|c| !c.is_whitespace()).count() as f64);
            let ratio = b / a.max(1.0);
            if !(0.3..=3.0).contains(&ratio) { return Some("translate: length far from the source's"); }
            None
        }
    }
}

/// Rust code that only uses the standard library (so `rustc` alone can check it).
pub fn self_contained_rust(code: &str) -> bool {
    !code.lines().any(|l| {
        let t = l.trim_start();
        let path = t.strip_prefix("use ").or_else(|| t.strip_prefix("pub use ")).or_else(|| t.strip_prefix("extern crate "));
        match path {
            Some(p) => {
                let root = p.trim_start_matches("::").split([':', ';', ' ', '{']).next().unwrap_or("");
                !matches!(root, "std" | "core" | "alloc" | "crate" | "self" | "super")
            }
            None => t.starts_with("#[tokio::") || t.contains("#[derive(Serialize") || t.contains("#[derive(Deserialize"),
        }
    })
}

/// Whether `code` compiles as a library crate (with a `main`, as a binary) on the given `rustc`; tests are compiled too.
pub fn compiles(rustc: &Path, code: &str) -> bool {
    let dir = std::env::temp_dir().join(format!("am2-codecheck-{}-{}", std::process::id(),
                                                 std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0)));
    if std::fs::create_dir_all(&dir).is_err() { return false; }
    let src = dir.join("check.rs");
    let ok = std::fs::write(&src, code).is_ok() && {
        let bin = code.contains("fn main(");
        let mut c = std::process::Command::new(rustc);
        c.args(["--edition", "2021", "--emit=metadata", "--test", "-A", "warnings", "-o"]).arg(dir.join("out"));
        if !bin { c.args(["--crate-type", "lib"]); }
        c.arg(&src).stdout(std::process::Stdio::null()).stderr(std::process::Stdio::null());
        c.status().map(|s| s.success()).unwrap_or(false)
    };
    let _ = std::fs::remove_dir_all(&dir);
    ok
}

/// `rustc` on the PATH (or `$RUSTC`), if there is one.
pub fn find_rustc() -> Option<std::path::PathBuf> {
    if let Ok(r) = std::env::var("RUSTC") { return Some(r.into()); }
    std::env::var_os("PATH").and_then(|p| std::env::split_paths(&p).map(|d| d.join("rustc")).find(|f| f.is_file()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn languages_are_named_by_code_name_or_native_name() {
        assert_eq!(langs(None).unwrap().len(), LANGS.len());
        let l = langs(Some("hi, Tamil,తెలుగు")).unwrap();
        assert_eq!(l.iter().map(|l| l.code).collect::<Vec<_>>(), ["hi", "ta", "te"]);
        assert!(langs(Some("klingon")).unwrap_err().contains("Hindi (hi)"));
    }

    #[test]
    fn script_share_tells_native_text_from_transliteration() {
        let hi = langs(Some("hi")).unwrap()[0];
        assert!(script_share("भारत एक विशाल देश है।", hi.block) > 0.99);
        assert!(script_share("Bharat ek vishal desh hai.", hi.block) < 0.01);
        assert!(script_share("Rust में `Vec<u8>` का उपयोग करें", hi.block) > MIN_SCRIPT_SHARE, "code and a few Latin words are fine");
        assert!(script_share("उदाहरण:\n```python\nprint('hello world, this is long code')\n```\nबस।", hi.block) > 0.99, "fences skip");
        let ta = langs(Some("ta")).unwrap()[0];
        assert!(script_share("வணக்கம் உலகம்", ta.block) > 0.99);
        assert!(script_share("नमस्ते", ta.block) < 0.01, "Devanagari is not Tamil script");
    }

    #[test]
    fn the_draw_repeats_for_a_seed_and_covers_the_catalogue() {
        let (mut a, mut b) = (Draw::new(7), Draw::new(7));
        let pa: Vec<String> = (0..20).map(|_| format!("{:?}", draw_code(&mut a, None))).collect();
        let pb: Vec<String> = (0..20).map(|_| format!("{:?}", draw_code(&mut b, None))).collect();
        assert_eq!(pa, pb);
        let mut d = Draw::new(1);
        let rust = (0..1000).filter(|_| matches!(draw_code(&mut d, None), Plan::Code { lang: "Rust", .. })).count();
        assert!((300..500).contains(&rust), "Rust is drawn about 6 times in 15: {rust}");
        let only_py = (0..50).all(|_| matches!(draw_code(&mut d, Some("python")), Plan::Code { lang: "Python", .. }));
        assert!(only_py);
        let ls = langs(Some("hi,ta")).unwrap();
        let dirs: Vec<bool> = (0..4).map(|i| matches!(draw_translate(&mut d, &ls, i), Plan::Translate { to_indic: true, .. })).collect();
        assert_eq!(dirs, [true, false, true, false]);
    }

    #[test]
    fn prompts_name_the_language_script_and_format() {
        let hi = langs(Some("hi")).unwrap()[0];
        let p = prompt(&Plan::Indic { lang: hi, kind: INDIC_KINDS[0], topic: "rivers" }, 3, None, None);
        assert!(p.contains("3 examples") && p.contains("Hindi (हिन्दी)") && p.contains("Devanagari") && p.contains("never transliterated"));
        let p = prompt(&Plan::Translate { lang: hi, to_indic: false, kind: "a recipe step", topic: "t" }, 2, None, None);
        assert!(p.contains("passages in Hindi (हिन्दी)") && p.contains("into English") && p.contains("\"translation\""));
        let p = prompt(&Plan::Translate { lang: hi, to_indic: true, kind: "k", topic: "t" }, 2, Some("The text."), Some("banking"));
        assert!(p.contains("copied exactly") && p.contains("TEXT:\nThe text.") && p.contains("banking"));
        let p = prompt(&Plan::Code { lang: "Rust", kind: CODE_KINDS[0].0, needs_code: true, topic: "hash maps" }, 4, None, None);
        assert!(p.contains("in Rust about hash maps") && p.contains("```rust"));
        let p = prompt(&Plan::Code { lang: "Rust", kind: CODE_KINDS[2].0, needs_code: false, topic: "x" }, 1, Some("fn a() {}"), None);
        assert!(p.contains("CODE:\nfn a() {}") && !p.contains("fenced block"));
    }

    #[test]
    fn translation_records_and_their_checks() {
        let hi = langs(Some("hi")).unwrap()[0];
        let plan = Plan::Translate { lang: hi, to_indic: true, kind: "k", topic: "t" };
        let mut d = Draw::new(3);
        let v = json!([{"source": "The train leaves at six in the morning.", "translation": "ट्रेन सुबह छह बजे निकलती है।"},
                       {"source": "Please close the door.", "translation": "Please close the door."},
                       {"source": "Good night, see you tomorrow.", "translation": "Shubh ratri, kal milte hain."},
                       {"source": "", "translation": "खाली"}]);
        let recs = records_from(&plan, &v, &mut d);
        assert_eq!(recs.len(), 3, "an empty source is no record");
        assert_eq!(recs[1].1.as_deref(), Some("Please close the door."));
        assert!(recs[0].0.messages[0].1.contains("The train leaves") && recs[0].0.messages[0].1.contains("Hindi"));
        let c = |i: usize| check(&plan, &recs[i].0, recs[i].1.as_deref(), None);
        assert_eq!(c(0), None);
        assert_eq!(c(1), Some("translate: translation in the wrong script"));
        assert_eq!(c(2), Some("translate: translation in the wrong script"), "transliteration is refused");
        let back = Plan::Translate { lang: hi, to_indic: false, kind: "k", topic: "t" };
        let src = Some("मैं घर जा रहा हूँ।");
        let r = Record { messages: vec![("user".into(), "Translate into English:\n\nमैं घर जा रहा हूँ।".into()), ("assistant".into(), "I am going home.".into())] };
        assert_eq!(check(&back, &r, src, None), None);
        let r = Record { messages: vec![r.messages[0].clone(), ("assistant".into(), "I am going home. ".repeat(20))] };
        assert_eq!(check(&back, &r, src, None), Some("translate: length far from the source's"));
        let r = Record { messages: vec![r.messages[0].clone(), ("assistant".into(), "मैं घर जा रहा हूँ।".into())] };
        assert_eq!(check(&back, &r, src, None), Some("translate: translation in the wrong script"));
    }

    #[test]
    fn indic_records_must_be_in_the_script() {
        let ta = langs(Some("ta")).unwrap()[0];
        let plan = Plan::Indic { lang: ta, kind: "k", topic: "t" };
        let ok = Record { messages: vec![("user".into(), "மழை ஏன் பெய்கிறது?".into()), ("assistant".into(), "மேகங்களில் உள்ள நீர்த்துளிகள் கனமாகும்போது மழை பெய்கிறது.".into())] };
        assert_eq!(check(&plan, &ok, None, None), None);
        let latin = Record { messages: vec![("user".into(), "Mazhai en peygirathu?".into()), ok.messages[1].clone()] };
        assert_eq!(check(&plan, &latin, None, None), Some("indic: request not in the script"));
    }

    #[test]
    fn code_answers_need_a_block_and_rust_is_compiled() {
        let plan = Plan::Code { lang: "Rust", kind: "k", needs_code: true, topic: "t" };
        let rec = |a: &str| Record { messages: vec![("user".into(), "Write add.".into()), ("assistant".into(), a.into())] };
        assert_eq!(check(&plan, &rec("Just add them."), None, None), Some("code: no fenced code block"));
        assert_eq!(code_blocks("a\n```rust\nfn f() {}\n```\nb\n```\nunclosed").len(), 1);
        assert!(self_contained_rust("use std::collections::HashMap;\nfn f() {}"));
        assert!(!self_contained_rust("use serde::Serialize;\n"));
        assert!(!self_contained_rust("#[tokio::main]\nasync fn main() {}"));
        let Some(rustc) = find_rustc() else { eprintln!("no rustc: compile check not tested"); return };
        let good = "Here:\n```rust\npub fn add(a: i32, b: i32) -> i32 { a + b }\n#[test]\nfn t() { assert_eq!(add(1, 2), 3); }\n```";
        let bad = "Here:\n```rust\npub fn add(a: i32, b: i32) -> i32 { a + \"b\" }\n```";
        let crate_use = "Here:\n```rust\nuse serde::Serialize;\n#[derive(Serialize)] struct A;\n```";
        assert_eq!(check(&plan, &rec(good), None, Some(&rustc)), None);
        assert_eq!(check(&plan, &rec(bad), None, Some(&rustc)), Some("code: Rust does not compile"));
        assert_eq!(check(&plan, &rec(crate_use), None, Some(&rustc)), None, "code needing crates is not compiled");
        let main = "```rust\nfn main() { println!(\"hi\"); }\n```";
        assert_eq!(check(&plan, &rec(main), None, Some(&rustc)), None);
    }
}
