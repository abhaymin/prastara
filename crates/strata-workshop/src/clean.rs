//! `clean`: a chat dataset made fit to train on. Every row is read from one or more JSONL files (any shape the
//! trainer reads), checked against the rules below in order, and kept or dropped for the **first** rule it breaks;
//! the kept rows are written in a seeded shuffle (the same inputs and seed give the same file), with a card that
//! lists the sources, the count per rule and a few dropped examples of each.
//!
//! The rules, in the order they are checked:
//! - `bad_row`: not JSON, or no conversation with an answer in it.
//! - `held_out`: the row is in an `--exclude` file (a regression test's held-out set), the same prompt or a
//!   near-copy of the conversation: training on it would make the test meaningless.
//! - `prompt_too_long`: with a tokenizer, the trainer's rule: the prompt alone fills `--max-len`, so no answer token
//!   is left to learn from. Rows that fit but are longer than `--max-len` are kept and counted as `cut`.
//! - `empty_answer`: an answer with nothing in it.
//! - `template_junk`: an answer with leftover `<think>` tags, chat-template markers (`<|im_end|>`, `[INST]`, ...) or
//!   the JSON wrapper a data generator asked for.
//! - `echo`: the answer only repeats the prompt (or the text it was asked to translate).
//! - `too_short`: fewer answer tokens than `--min-answer-tokens`, or fewer than `--min-answer-ratio` × the prompt's.
//! - `repetition`: the answer repeats itself: the share of its word 3-grams that occur more than once is over
//!   `--max-repeat`, or the whole answer is one phrase said again and again.
//! - `script_mismatch` (with `--lang-match`): the answer is not in the script it should be — the translation's
//!   target language when the prompt asks for one (or the row's `meta.to`), else the prompt's own script or the
//!   row's `meta.language`.
//! - `exact_duplicate`, `near_duplicate`: the same conversation once its text is normalised (case, punctuation,
//!   white space), or a near-copy: MinHash (64 hashes of word 3-grams) Jaccard of at least `--near` on the prompt
//!   **and** on the answer (the same passage with a different question is not a duplicate).
//! - `language_cap`: with `--cap-per-language N`, rows past N of one language (`meta.language`, ISO codes of two or
//!   three letters taken as the same language), drawn by the seed — from the `--cap-only` inputs when given (the
//!   others are kept whole and count towards the N).

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context as _, Result};
use serde_json::{json, Value};

use crate::chatdata::{self, ChatTokenizer, DataFormat};
use crate::dataset::Record;
use crate::recipes::{Draw, LANGS};

// ------------------------------------------------------------------------------------------------------- settings

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dedupe {
    Off,
    Exact,
    Near,
}

impl Dedupe {
    pub fn from_s(s: &str) -> Result<Self> {
        match s.to_ascii_lowercase().as_str() {
            "off" | "none" | "no" => Ok(Dedupe::Off),
            "exact" => Ok(Dedupe::Exact),
            "near" => Ok(Dedupe::Near),
            _ => bail!("--dedupe {s:?}: exact, near or off"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Settings {
    pub inputs: Vec<PathBuf>,
    pub out: PathBuf,
    /// a Hugging Face model folder: its tokenizer and chat template measure the rows in tokens (else in words, and
    /// the length rule is not checked)
    pub model: Option<PathBuf>,
    pub chat_template: Option<PathBuf>,
    pub max_len: usize,
    /// drop rows longer than `max_len` (the trainer would cut their tail, so the answer is never seen to end)
    pub drop_cut: bool,
    pub min_answer_tokens: usize,
    pub min_answer_ratio: f64,
    pub max_repeat: f64,
    pub dedupe: Dedupe,
    pub near: f64,
    pub lang_match: bool,
    pub exclude: Vec<PathBuf>,
    pub cap_per_language: Option<usize>,
    /// the cap drops rows only from these inputs (the others are kept whole, but count towards it); empty: all
    pub cap_only: Vec<PathBuf>,
    pub seed: u64,
    /// the card; None: `<out stem>.card.md` beside the output
    pub report: Option<PathBuf>,
}

impl Settings {
    pub fn new(inputs: Vec<PathBuf>, out: PathBuf) -> Self {
        Settings { inputs, out, model: None, chat_template: None, max_len: 1024, drop_cut: false, min_answer_tokens: 4, min_answer_ratio: 0.0,
                   max_repeat: 0.5, dedupe: Dedupe::Near, near: 0.8, lang_match: false, exclude: vec![], cap_per_language: None, cap_only: vec![],
                   seed: 7, report: None }
    }

    pub fn card_path(&self) -> PathBuf {
        self.report.clone().unwrap_or_else(|| crate::datagen::card_path(&self.out))
    }
}

/// The rules, in the order they are checked (and shown).
pub const RULES: &[(&str, &str)] = &[
    ("bad_row", "not JSON, or no conversation with an answer"),
    ("held_out", "in an --exclude (held-out) file: same prompt or a near-copy"),
    ("prompt_too_long", "the prompt alone fills --max-len: the trainer would drop it (no answer token left)"),
    ("too_long", "longer than --max-len with --drop-cut: the trainer would cut the answer before it ends"),
    ("empty_answer", "the answer is empty"),
    ("template_junk", "leftover <think> tags, chat-template markers or a JSON wrapper in the answer"),
    ("echo", "the answer only repeats the prompt"),
    ("too_short", "too few answer tokens (absolute or for the prompt's length)"),
    ("repetition", "the answer repeats itself"),
    ("script_mismatch", "the answer is not in the expected script"),
    ("exact_duplicate", "the same conversation, once normalised"),
    ("near_duplicate", "a near-copy (prompt and answer) of a kept row"),
    ("language_cap", "past --cap-per-language for its language"),
];

// ---------------------------------------------------------------------------------------------- text and scripts

/// A character that belongs to a word: letters, digits and the combining marks Indic scripts write vowels with.
fn word_char(c: char) -> bool {
    if c.is_alphanumeric() {
        return true;
    }
    let u = c as u32;
    // combining diacritics; Indic blocks (vowel signs, viramas, nuktas) except the danda and double danda; Arabic marks
    (0x0300..=0x036F).contains(&u) || ((0x0900..=0x0DFF).contains(&u) && u != 0x0964 && u != 0x0965)
        || (0x064B..=0x065F).contains(&u) || u == 0x0670 || (0x200C..=0x200D).contains(&u)
}

/// The words of a text, lower-cased, punctuation stripped.
pub fn words(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(|w| w.chars().filter(|c| word_char(*c)).flat_map(char::to_lowercase).collect::<String>())
        .filter(|w| !w.is_empty())
        .collect()
}

/// Text for comparing: its words joined by single spaces.
pub fn normalise(text: &str) -> String {
    words(text).join(" ")
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Script {
    Latin,
    Devanagari,
    Bengali,
    Gurmukhi,
    Gujarati,
    Odia,
    Tamil,
    Telugu,
    Kannada,
    Malayalam,
    Arabic,
    Other,
}

impl Script {
    pub fn as_str(self) -> &'static str {
        match self {
            Script::Latin => "Latin", Script::Devanagari => "Devanagari", Script::Bengali => "Bengali", Script::Gurmukhi => "Gurmukhi",
            Script::Gujarati => "Gujarati", Script::Odia => "Odia", Script::Tamil => "Tamil", Script::Telugu => "Telugu",
            Script::Kannada => "Kannada", Script::Malayalam => "Malayalam", Script::Arabic => "Arabic", Script::Other => "other",
        }
    }
}

/// The script of a letter or combining mark (None: not a letter — digits, punctuation, space).
pub fn script_of(c: char) -> Option<Script> {
    let u = c as u32;
    Some(match u {
        0x0900..=0x097F | 0xA8E0..=0xA8FF => Script::Devanagari,
        0x0980..=0x09FF => Script::Bengali,
        0x0A00..=0x0A7F => Script::Gurmukhi,
        0x0A80..=0x0AFF => Script::Gujarati,
        0x0B00..=0x0B7F => Script::Odia,
        0x0B80..=0x0BFF => Script::Tamil,
        0x0C00..=0x0C7F => Script::Telugu,
        0x0C80..=0x0CFF => Script::Kannada,
        0x0D00..=0x0D7F => Script::Malayalam,
        0x0600..=0x06FF | 0x0750..=0x077F | 0xFB50..=0xFDFF | 0xFE70..=0xFEFF => Script::Arabic,
        _ if c.is_ascii_alphabetic() || (0x00C0..=0x024F).contains(&u) && c.is_alphabetic() => Script::Latin,
        _ if c.is_alphabetic() => Script::Other,
        _ => return None,
    }).filter(|_| !(u == 0x0964 || u == 0x0965 || c.is_numeric()))
}

/// The script most of a text's letters are in, and how many letters it has (code in backticks does not count).
pub fn dominant_script(text: &str) -> (Option<Script>, usize) {
    let mut counts: BTreeMap<Script, usize> = BTreeMap::new();
    let mut in_code = false;
    for c in text.chars() {
        if c == '`' {
            in_code = !in_code;
            continue;
        }
        if in_code {
            continue;
        }
        if let Some(s) = script_of(c) {
            *counts.entry(s).or_default() += 1;
        }
    }
    let total = counts.values().sum();
    (counts.into_iter().max_by_key(|(s, n)| (*n, std::cmp::Reverse(*s))).map(|(s, _)| s), total)
}

/// The script a language (an ISO 639 code of two or three letters, or its English name) is written in.
pub fn script_of_language(lang: &str) -> Option<Script> {
    let l = lang.trim().to_ascii_lowercase();
    let code = iso2(&l);
    if matches!(code.as_str(), "en" | "english") {
        return Some(Script::Latin);
    }
    let found = LANGS.iter().find(|x| x.code == code || x.name.eq_ignore_ascii_case(&l))?;
    Some(match found.block.0 {
        0x0900 => Script::Devanagari, 0x0980 => Script::Bengali, 0x0A00 => Script::Gurmukhi, 0x0A80 => Script::Gujarati,
        0x0B00 => Script::Odia, 0x0B80 => Script::Tamil, 0x0C00 => Script::Telugu, 0x0C80 => Script::Kannada,
        0x0D00 => Script::Malayalam, 0x0600 => Script::Arabic, _ => Script::Other,
    })
}

/// A language code as two letters where it has them (Aya's `ben`, `npi`, `pan` ... are `bn`, `ne`, `pa`).
pub fn iso2(code: &str) -> String {
    let c = code.trim().to_ascii_lowercase();
    let two = match c.as_str() {
        "eng" => "en", "hin" => "hi", "ben" => "bn", "mar" => "mr", "tel" => "te", "tam" => "ta", "guj" => "gu", "urd" => "ur",
        "kan" => "kn", "ory" | "ori" => "or", "mal" => "ml", "pan" => "pa", "asm" => "as", "san" => "sa", "npi" | "nep" => "ne",
        "gom" => "kok",
        other => other,
    };
    two.to_string()
}

/// The language a translation request asks for, from its wording ("Translate into Hindi", "What is this in
/// English?", "... from Tamil to English"): the last language named after into/to/in on a line that asks to
/// translate.
pub fn translation_target(prompt: &str) -> Option<String> {
    let names: Vec<&str> = std::iter::once("English").chain(LANGS.iter().map(|l| l.name)).collect();
    for line in prompt.lines() {
        let low = line.to_lowercase();
        if !(low.contains("translat") || low.contains("what is this in")) {
            continue;
        }
        let mut best: Option<(usize, &str)> = None;
        for kw in ["into ", "to ", "in "] {
            let mut from = 0;
            while let Some(p) = low[from..].find(kw) {
                let at = from + p;
                from = at + kw.len();
                // a whole word before it
                if at > 0 && low[..at].chars().next_back().is_some_and(|c| c.is_alphanumeric()) {
                    continue;
                }
                let rest = &low[at + kw.len()..];
                if let Some(n) = names.iter().find(|n| rest.starts_with(&n.to_lowercase())
                    && !rest[n.len()..].chars().next().is_some_and(|c| c.is_alphanumeric())) {
                    if best.is_none_or(|(b, _)| at > b) {
                        best = Some((at, n));
                    }
                }
            }
        }
        if let Some((_, n)) = best {
            return Some(n.to_string());
        }
    }
    None
}

// ------------------------------------------------------------------------------------------------------ the rules

/// Leftovers that must never be in an answer: reasoning tags, chat-template markers.
const JUNK: &[&str] = &["<think>", "</think>", "<|im_start|>", "<|im_end|>", "<|endoftext|>", "<|eot_id|>", "<|start_header_id|>",
                        "<|end_header_id|>", "<start_of_turn>", "<end_of_turn>", "[INST]", "[/INST]", "<|assistant|>", "<|user|>",
                        "<|system|>", "<|end|>", "</s>", "<|begin_of_text|>"];

/// Template junk in an answer: the marker found, or None.
pub fn junk(answer: &str) -> Option<String> {
    if let Some(j) = JUNK.iter().find(|j| answer.contains(**j)) {
        return Some((*j).to_string());
    }
    let t = answer.trim_start().trim_start_matches("```json").trim_start();
    for k in ["{\"question\"", "[{\"question\"", "{\"answer\"", "{\"translation\"", "[{\"source\"", "{\"source\""] {
        if t.starts_with(k) {
            return Some(format!("a JSON wrapper ({k}...)"));
        }
    }
    let low = answer.trim_start().to_lowercase();
    if low.starts_with("assistant:") || low.starts_with("assistant\n") {
        return Some("a role label".into());
    }
    None
}

/// The answer only repeats the prompt (or the text the prompt quotes, for a translation that is a copy).
pub fn echoes(prompt: &str, answer: &str) -> bool {
    let (p, a) = (normalise(prompt), normalise(answer));
    if a.is_empty() {
        return false;
    }
    if p == a {
        return true;
    }
    // the prompt is the answer plus a short instruction ("Translate into Hindi:" and the text, copied back)
    let (pc, ac) = (p.chars().count(), a.chars().count());
    ac >= 20 && p.contains(&a) && pc - ac <= 64.max(pc / 5)
}

/// How much an answer repeats itself, 0 to 1: the share of its word 3-grams that occur more than once, or 1 when the
/// whole answer is one phrase said two or more times ("x y x y").
pub fn repetition(answer: &str) -> f64 {
    let w = words(answer);
    if w.len() >= 2 {
        for period in 1..=w.len() / 2 {
            if w.len() % period == 0 && (period..w.len()).all(|i| w[i] == w[i % period]) {
                return 1.0;
            }
        }
    }
    const N: usize = 3;
    if w.len() < N + 1 {
        return 0.0;
    }
    let grams: Vec<&[String]> = w.windows(N).collect();
    let mut seen: HashMap<&[String], usize> = HashMap::new();
    for g in &grams {
        *seen.entry(g).or_default() += 1;
    }
    grams.iter().filter(|g| seen[*g] > 1).count() as f64 / grams.len() as f64
}

/// The scripts an answer may be in, and why; empty: not checked.
fn expected_scripts(prompt: &str, meta: &Value) -> (Vec<Script>, String) {
    if let Some(to) = meta["to"].as_str().and_then(script_of_language) {
        return (vec![to], format!("meta.to {}", meta["to"].as_str().unwrap_or("")));
    }
    if let Some(t) = translation_target(prompt) {
        if let Some(s) = script_of_language(&t) {
            return (vec![s], format!("a translation into {t}"));
        }
    }
    let mut v = vec![];
    let mut why = vec![];
    if let (Some(s), n) = dominant_script(prompt) {
        if n >= 3 {
            v.push(s);
            why.push(format!("the prompt's {}", s.as_str()));
        }
    }
    if let Some(l) = meta["language"].as_str() {
        if let Some(s) = script_of_language(l) {
            if !v.contains(&s) {
                v.push(s);
            }
            why.push(format!("meta.language {l}"));
        }
    }
    (v, why.join(" or "))
}

/// With `--lang-match`: why the answer is in the wrong script, or None.
pub fn script_mismatch(prompt: &str, answer: &str, meta: &Value) -> Option<String> {
    let (want, why) = expected_scripts(prompt, meta);
    if want.is_empty() {
        return None;
    }
    let (got, letters) = dominant_script(answer);
    // numbers, code, a symbol: nothing to judge by
    if letters < 8 {
        return None;
    }
    let got = got?;
    (!want.contains(&got)).then(|| format!("the answer is in {} script, expected {why}", got.as_str()))
}

// ------------------------------------------------------------------------------------------------------- minhash

const K: usize = 64;
const BANDS: usize = 16;
const ROWS: usize = K / BANDS;

fn splitmix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn fnv(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in bytes {
        h ^= *b as u64;
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    h
}

/// A MinHash signature of a text's word 3-grams (the whole text when it has fewer words).
pub fn signature(text: &str) -> [u64; K] {
    let w = words(text);
    let mut sig = [u64::MAX; K];
    let shingles: Vec<u64> = if w.len() < 3 {
        if w.is_empty() { vec![] } else { vec![fnv(w.join(" ").as_bytes())] }
    } else {
        w.windows(3).map(|g| fnv(g.join(" ").as_bytes())).collect()
    };
    for s in shingles {
        let (h1, h2) = (splitmix(s), splitmix(s ^ 0x5bd1_e995) | 1);
        for (k, slot) in sig.iter_mut().enumerate() {
            let v = h1.wrapping_add((k as u64).wrapping_mul(h2));
            if v < *slot {
                *slot = v;
            }
        }
    }
    sig
}

pub fn jaccard(a: &[u64; K], b: &[u64; K]) -> f64 {
    a.iter().zip(b).filter(|(x, y)| x == y).count() as f64 / K as f64
}

/// Prompt and answer signatures, banded on the prompt's: a row's near-copies are found among the rows that share a
/// band with it.
#[derive(Default)]
struct NearIndex {
    sigs: Vec<([u64; K], [u64; K])>,
    bands: HashMap<(usize, u64), Vec<usize>>,
}

impl NearIndex {
    fn band_keys(sig: &[u64; K]) -> impl Iterator<Item = (usize, u64)> + '_ {
        (0..BANDS).map(move |b| (b, fnv(&sig[b * ROWS..(b + 1) * ROWS].iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>())))
    }
    /// The first indexed row whose prompt and answer are both at least `t` alike.
    fn find(&self, p: &[u64; K], a: &[u64; K], t: f64) -> Option<usize> {
        let mut seen = HashSet::new();
        for key in Self::band_keys(p) {
            for &i in self.bands.get(&key).map(Vec::as_slice).unwrap_or(&[]) {
                if seen.insert(i) && jaccard(&self.sigs[i].0, p) >= t && jaccard(&self.sigs[i].1, a) >= t {
                    return Some(i);
                }
            }
        }
        None
    }
    fn add(&mut self, p: [u64; K], a: [u64; K]) -> usize {
        let i = self.sigs.len();
        for key in Self::band_keys(&p) {
            self.bands.entry(key).or_default().push(i);
        }
        self.sigs.push((p, a));
        i
    }
}

fn exact_key(prompt: &str, answer: &str) -> [u8; 16] {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(normalise(prompt).as_bytes());
    h.update([0u8]);
    h.update(normalise(answer).as_bytes());
    h.finalize()[..16].try_into().unwrap()
}

fn prompt_key(prompt: &str) -> [u8; 16] {
    use sha2::{Digest, Sha256};
    Sha256::digest(normalise(prompt).as_bytes())[..16].try_into().unwrap()
}

// ------------------------------------------------------------------------------------------------------- the rows

/// One conversation of the inputs.
#[derive(Debug, Clone)]
pub struct Row {
    pub source: usize,
    pub line: usize,
    /// the line as it was read (written out unchanged)
    pub raw: String,
    pub meta: Value,
    pub record: Record,
}

impl Row {
    /// Everything said before the first answer (the system prompt and the user's turns).
    pub fn prompt(&self) -> String {
        let first = self.record.messages.iter().position(|(r, _)| r == "assistant").unwrap_or(self.record.messages.len());
        self.record.messages[..first].iter().map(|(_, c)| c.as_str()).collect::<Vec<_>>().join("\n")
    }
    /// The user's last turn before the first answer (what the answer answers).
    pub fn question(&self) -> String {
        let first = self.record.messages.iter().position(|(r, _)| r == "assistant").unwrap_or(self.record.messages.len());
        self.record.messages[..first].iter().rev().find(|(r, _)| r == "user").map(|(_, c)| c.clone()).unwrap_or_default()
    }
    /// The answers, one after another.
    pub fn answer(&self) -> String {
        self.record.messages.iter().filter(|(r, _)| r == "assistant").map(|(_, c)| c.as_str()).collect::<Vec<_>>().join("\n")
    }
    /// The row's language for capping and counting: `meta.language`, else the non-English side of a translation.
    pub fn language(&self) -> Option<String> {
        if let Some(l) = self.meta["language"].as_str() {
            return Some(iso2(l));
        }
        let (f, t) = (self.meta["from"].as_str().map(iso2), self.meta["to"].as_str().map(iso2));
        match (f, t) {
            (Some(f), Some(t)) if t == "en" => Some(f),
            (_, Some(t)) => Some(t),
            (Some(f), None) => Some(f),
            _ => None,
        }
    }
}

/// Read the rows of a JSONL file; bad lines come back with their number and why.
pub fn read_rows(path: &Path, source: usize) -> Result<(Vec<Row>, Vec<(usize, String)>)> {
    let text = std::fs::read_to_string(path).with_context(|| format!("{}", path.display()))?;
    let (mut rows, mut bad) = (vec![], vec![]);
    for (i, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        match serde_json::from_str::<Value>(line).map_err(|e| anyhow!("invalid JSON: {e}"))
            .and_then(|v| chatdata::record_of(&v, DataFormat::Auto).map(|r| (v, r))) {
            Ok((v, record)) => rows.push(Row { source, line: i + 1, raw: line.to_string(), meta: v["meta"].clone(), record }),
            Err(e) => bad.push((i + 1, e.to_string())),
        }
    }
    Ok((rows, bad))
}

/// One row's length: in tokens with a tokenizer, else in words.
#[derive(Debug, Clone, Copy, Default)]
pub struct Length {
    pub prompt: usize,
    pub answer: usize,
    pub total: usize,
    /// with a tokenizer: the trainer would find an answer token within --max-len
    pub fits: bool,
}

fn length_of(row: &Row, tok: Option<&ChatTokenizer>, max_len: usize) -> Result<Length> {
    match tok {
        Some(t) => {
            let m = t.measure(&row.record)?;
            // the answers' own tokens: what the template adds around them (an empty think block, the end-of-turn
            // marker) is not the answer
            let answer = row.record.messages.iter().filter(|(r, _)| r == "assistant").map(|(_, c)| t.count_text(c)).sum::<Result<usize>>()?;
            Ok(Length { prompt: m.first_answer.unwrap_or(m.total), answer, total: m.total, fits: m.fits(max_len) })
        }
        None => {
            let (p, a) = (row.prompt().split_whitespace().count(), row.answer().split_whitespace().count());
            Ok(Length { prompt: p, answer: a, total: p + a, fits: true })
        }
    }
}

/// The per-row rules (not the duplicate and cap rules, which need the other rows): the rule broken and a detail.
pub fn check_row(s: &Settings, row: &Row, len: &Length) -> Option<(&'static str, String)> {
    let (q, a) = (row.question(), row.answer());
    if !len.fits {
        return Some(("prompt_too_long", format!("{} prompt tokens (--max-len {})", len.prompt, s.max_len)));
    }
    if s.drop_cut && len.total > s.max_len {
        return Some(("too_long", format!("{} tokens (--max-len {})", len.total, s.max_len)));
    }
    if row.record.messages.iter().any(|(r, c)| r == "assistant" && c.trim().is_empty()) {
        return Some(("empty_answer", String::new()));
    }
    if let Some(j) = junk(&a) {
        return Some(("template_junk", j));
    }
    if echoes(&q, &a) {
        return Some(("echo", String::new()));
    }
    let unit = if s.model.is_some() { "tokens" } else { "words" };
    if len.answer < s.min_answer_tokens {
        return Some(("too_short", format!("{} answer {unit} (at least {})", len.answer, s.min_answer_tokens)));
    }
    if s.min_answer_ratio > 0.0 && (len.answer as f64) < s.min_answer_ratio * len.prompt as f64 {
        return Some(("too_short", format!("{} answer {unit} for a {}-{unit} prompt (at least {:.2} of it)", len.answer, len.prompt, s.min_answer_ratio)));
    }
    let r = repetition(&a);
    if r > s.max_repeat {
        return Some(("repetition", format!("{:.0}% of its 3-grams repeat", r * 100.0)));
    }
    if s.lang_match {
        if let Some(why) = script_mismatch(&q, &a, &row.meta) {
            return Some(("script_mismatch", why));
        }
    }
    None
}

// ------------------------------------------------------------------------------------------------------- the run

#[derive(Debug, Clone)]
pub struct Dropped {
    pub source: String,
    pub line: usize,
    pub prompt: String,
    pub answer: String,
    pub detail: String,
}

#[derive(Debug, Clone, Default)]
pub struct SourceCount {
    pub path: String,
    pub read: usize,
    pub kept: usize,
    pub licences: Vec<String>,
    pub models: Vec<String>,
}

#[derive(Debug, Clone, Default)]
pub struct Report {
    pub read: usize,
    pub kept: usize,
    pub dropped: BTreeMap<&'static str, usize>,
    pub examples: BTreeMap<&'static str, Vec<Dropped>>,
    /// kept rows longer than --max-len (the trainer cuts their tail)
    pub cut: usize,
    pub sources: Vec<SourceCount>,
    pub languages: BTreeMap<String, usize>,
    pub excluded_rows: usize,
    pub tokens: bool,
}

fn clip(s: &str, n: usize) -> String {
    let t: String = s.chars().take(n).collect();
    let t = t.replace('\n', " ⏎ ");
    if s.chars().count() > n { format!("{t}…") } else { t }
}

/// Clean the inputs into `s.out` and write the card. `event` gets the JSON events (progress, one per rule, ...).
pub fn run(s: &Settings, mut event: impl FnMut(Value)) -> Result<Report> {
    if s.inputs.is_empty() {
        bail!("no input files");
    }
    if !(0.0..=1.0).contains(&s.max_repeat) || !(0.0..=1.0).contains(&s.near) || s.min_answer_ratio < 0.0 {
        bail!("--max-repeat and --near are 0 to 1, --min-answer-ratio at least 0");
    }
    let tok = match &s.model {
        Some(m) => Some(ChatTokenizer::load(m, s.chat_template.as_deref()).with_context(|| format!("the tokenizer of {}", m.display()))?),
        None => None,
    };
    let mut rep = Report { tokens: tok.is_some(), ..Default::default() };
    let name = |p: &Path| p.file_name().and_then(|x| x.to_str()).unwrap_or("?").to_string();

    // the held-out rows: their prompts, conversations and signatures
    let (mut held_prompts, mut held_exact, mut held_near) = (HashSet::new(), HashSet::new(), NearIndex::default());
    for p in &s.exclude {
        let (rows, _) = read_rows(p, usize::MAX)?;
        for r in &rows {
            let (q, a) = (r.prompt(), r.answer());
            held_prompts.insert(prompt_key(&q));
            held_exact.insert(exact_key(&q, &a));
            held_near.add(signature(&q), signature(&a));
        }
        rep.excluded_rows += rows.len();
        event(json!({"event": "log", "message": format!("{}: {} held-out rows to keep out", p.display(), rows.len())}));
    }

    let mut rows = vec![];
    for (i, p) in s.inputs.iter().enumerate() {
        let (r, bad) = read_rows(p, i)?;
        event(json!({"event": "log", "message": format!("{}: {} rows{}", p.display(), r.len() + bad.len(),
                                                       if bad.is_empty() { String::new() } else { format!(" ({} unreadable)", bad.len()) })}));
        let mut sc = SourceCount { path: p.display().to_string(), read: r.len() + bad.len(), ..Default::default() };
        for row in &r {
            for (k, list) in [("licence", &mut sc.licences), ("model", &mut sc.models)] {
                if let Some(v) = row.meta[k].as_str().or_else(|| if k == "licence" { row.meta["license"].as_str() } else { None }) {
                    if !list.iter().any(|x| x == v) {
                        list.push(v.to_string());
                    }
                }
            }
        }
        rep.sources.push(sc);
        *rep.dropped.entry("bad_row").or_default() += bad.len();
        for (line, why) in bad.into_iter().take(3) {
            rep.examples.entry("bad_row").or_default().push(Dropped { source: name(p), line, prompt: String::new(), answer: String::new(), detail: why });
        }
        rows.extend(r);
    }
    rep.read = rows.len() + rep.dropped.get("bad_row").copied().unwrap_or(0);
    let total = rows.len();
    event(json!({"event": "progress", "done": 0, "total": total, "message": "measuring the rows"}));

    // lengths (the tokenizer is the slow part): in parallel, in order
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).min(16);
    let chunk = total.div_ceil(threads).max(1);
    let lengths: Vec<Result<Length, String>> = std::thread::scope(|sc| {
        let handles: Vec<_> = rows.chunks(chunk).map(|part| {
            let tok = tok.as_ref();
            sc.spawn(move || part.iter().map(|r| length_of(r, tok, s.max_len).map_err(|e| format!("{e:#}"))).collect::<Vec<_>>())
        }).collect();
        handles.into_iter().flat_map(|h| h.join().expect("a measuring thread panicked")).collect()
    });
    event(json!({"event": "progress", "done": total, "total": total, "message": "checking the rules"}));

    let drop_row = |rep: &mut Report, rule: &'static str, row: &Row, detail: String| {
        *rep.dropped.entry(rule).or_default() += 1;
        let ex = rep.examples.entry(rule).or_default();
        if ex.len() < 3 {
            ex.push(Dropped { source: name(&s.inputs[row.source]), line: row.line, prompt: clip(&row.question(), 160), answer: clip(&row.answer(), 240), detail });
        }
    };
    let (mut exact, mut near) = (HashSet::new(), NearIndex::default());
    let mut kept: Vec<(usize, Length)> = vec![];
    for (i, row) in rows.iter().enumerate() {
        let (q, a) = (row.prompt(), row.answer());
        // held out: the same prompt, the same conversation, or a near-copy of one
        if !s.exclude.is_empty() {
            let (qk, ek) = (prompt_key(&q), exact_key(&q, &a));
            if held_prompts.contains(&qk) || held_exact.contains(&ek) {
                drop_row(&mut rep, "held_out", row, "the same prompt as a held-out row".into());
                continue;
            }
            if let Some(j) = held_near.find(&signature(&q), &signature(&a), s.near) {
                drop_row(&mut rep, "held_out", row, format!("a near-copy of held-out row {}", j + 1));
                continue;
            }
        }
        let len = match &lengths[i] {
            Ok(l) => *l,
            Err(e) => {
                drop_row(&mut rep, "bad_row", row, e.clone());
                continue;
            }
        };
        if let Some((rule, detail)) = check_row(s, row, &len) {
            drop_row(&mut rep, rule, row, detail);
            continue;
        }
        if s.dedupe != Dedupe::Off && !exact.insert(exact_key(&q, &a)) {
            drop_row(&mut rep, "exact_duplicate", row, String::new());
            continue;
        }
        if s.dedupe == Dedupe::Near {
            let (ps, as_) = (signature(&q), signature(&a));
            if let Some(j) = near.find(&ps, &as_, s.near) {
                let other = &rows[kept[j].0];
                drop_row(&mut rep, "near_duplicate", row, format!("like {} line {}", name(&s.inputs[other.source]), other.line));
                continue;
            }
            near.add(ps, as_);
        }
        kept.push((i, len));
        if (i + 1) % 5000 == 0 {
            event(json!({"event": "progress", "done": i + 1, "total": total, "message": format!("{} kept so far", kept.len())}));
        }
    }

    // the cap per language: a seeded draw within each language
    let mut draw = Draw::new(s.seed);
    if let Some(cap) = s.cap_per_language {
        let mut by: BTreeMap<String, Vec<usize>> = BTreeMap::new();
        for (k, (i, _)) in kept.iter().enumerate() {
            if let Some(l) = rows[*i].language() {
                by.entry(l).or_default().push(k);
            }
        }
        // the inputs the cap may draw from
        let same = |a: &Path, b: &Path| a == b || a.canonicalize().ok().is_some_and(|x| Some(x) == b.canonicalize().ok());
        let capped: Vec<bool> = s.inputs.iter().map(|p| s.cap_only.is_empty() || s.cap_only.iter().any(|c| same(c, p))).collect();
        let mut off = HashSet::new();
        for (_, ks) in by {
            let (mut pool, whole): (Vec<usize>, Vec<usize>) = ks.into_iter().partition(|k| capped[rows[kept[*k].0].source]);
            shuffle(&mut pool, &mut draw);
            off.extend(pool.into_iter().skip(cap.saturating_sub(whole.len())));
        }
        let mut still = vec![];
        for (k, x) in kept.into_iter().enumerate() {
            if off.contains(&k) {
                let row = &rows[x.0];
                drop_row(&mut rep, "language_cap", row, format!("{} past {cap}", row.language().unwrap_or_default()));
            } else {
                still.push(x);
            }
        }
        kept = still;
    }

    shuffle(&mut kept, &mut draw);
    let mut out = String::new();
    for (i, len) in &kept {
        let row = &rows[*i];
        out.push_str(&row.raw);
        out.push('\n');
        rep.sources[row.source].kept += 1;
        *rep.languages.entry(row.language().unwrap_or_else(|| "unknown".into())).or_default() += 1;
        if rep.tokens && len.total > s.max_len {
            rep.cut += 1;
        }
    }
    rep.kept = kept.len();
    if let Some(parent) = s.out.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent).ok();
    }
    std::fs::write(&s.out, out).with_context(|| format!("writing {}", s.out.display()))?;
    for (rule, _) in RULES {
        let n = rep.dropped.get(rule).copied().unwrap_or(0);
        event(json!({"event": "rule", "rule": rule, "dropped": n, "message": format!("{rule}: {n} dropped"),
                     "examples": rep.examples.get(rule).map(|v| v.iter().map(|d| json!({"source": d.source, "line": d.line, "detail": d.detail})).collect::<Vec<_>>())}));
    }
    if rep.tokens {
        event(json!({"event": "log", "message": format!("{} kept rows are longer than {} tokens: the trainer cuts their tail", rep.cut, s.max_len)}));
    }
    std::fs::write(s.card_path(), card(s, &rep)).with_context(|| format!("writing {}", s.card_path().display()))?;
    Ok(rep)
}

fn shuffle<T>(v: &mut [T], d: &mut Draw) {
    for i in (1..v.len()).rev() {
        let j = (d.next() % (i as u64 + 1)) as usize;
        v.swap(i, j);
    }
}

/// The card: sources, settings, the count per rule, examples of each.
pub fn card(s: &Settings, r: &Report) -> String {
    let name = s.out.file_name().and_then(|x| x.to_str()).unwrap_or("dataset");
    let mut o = format!("# Dataset card: {name}\n\nCleaned by Prastara's Workshop (`am2strata-workshop clean`): {} rows read, **{} kept**.\n\n", r.read, r.kept);
    o.push_str("## Sources\n\n| File | Rows | Kept | Licence | Written by |\n|---|---:|---:|---|---|\n");
    for sc in &r.sources {
        let f = Path::new(&sc.path).file_name().and_then(|x| x.to_str()).unwrap_or(&sc.path);
        o.push_str(&format!("| {f} | {} | {} | {} | {} |\n", sc.read, sc.kept,
                            if sc.licences.is_empty() { "-".into() } else { sc.licences.join(", ") },
                            if sc.models.is_empty() { "-".into() } else { sc.models.join(", ") }));
    }
    o.push_str("\n## Settings\n\n");
    let tokenizer = s.model.as_ref().map_or("none: lengths in words, the length rule not checked".to_string(), |m| m.display().to_string());
    o.push_str(&format!("- tokenizer and chat template: {tokenizer}\n- max length: {} tokens\n- answer at least {} {}{}\n- repetition: at most {:.0}% of the answer's word 3-grams repeated\n- duplicates: {}\n- script check: {}\n- cap per language: {}\n- seed: {}\n",
        s.max_len, s.min_answer_tokens, if r.tokens { "tokens" } else { "words" },
        if s.min_answer_ratio > 0.0 { format!(" and at least {:.2} of the prompt's", s.min_answer_ratio) } else { String::new() },
        s.max_repeat * 100.0, match s.dedupe { Dedupe::Off => "off".to_string(), Dedupe::Exact => "exact".into(), Dedupe::Near => format!("exact and near (Jaccard {:.2})", s.near) },
        if s.lang_match { "on" } else { "off" }, s.cap_per_language.map_or("none".to_string(), |c| if s.cap_only.is_empty() { c.to_string() } else {
            format!("{c}, drawn from {} only", s.cap_only.iter().filter_map(|p| p.file_name()?.to_str()).collect::<Vec<_>>().join(", ")) }), s.seed));
    if !s.exclude.is_empty() {
        o.push_str(&format!("- held out ({} rows kept out): {}\n", r.excluded_rows, s.exclude.iter().map(|p| format!("`{}`", p.display())).collect::<Vec<_>>().join(", ")));
    }
    o.push_str("\n## Dropped, by rule\n\n| Rule | Dropped | What it means |\n|---|---:|---|\n");
    for (rule, what) in RULES {
        o.push_str(&format!("| `{rule}` | {} | {what} |\n", r.dropped.get(rule).copied().unwrap_or(0)));
    }
    if r.tokens {
        o.push_str(&format!("\n{} kept rows are longer than {} tokens: they train on their first {} tokens only.\n", r.cut, s.max_len, s.max_len));
    }
    o.push_str("\n## Kept, by language\n\n| Language | Rows |\n|---|---:|\n");
    for (l, n) in &r.languages {
        o.push_str(&format!("| {l} | {n} |\n"));
    }
    o.push_str("\n## Examples of dropped rows\n");
    for (rule, _) in RULES {
        let Some(ex) = r.examples.get(rule).filter(|e| !e.is_empty()) else { continue };
        o.push_str(&format!("\n### `{rule}`\n\n"));
        for d in ex {
            o.push_str(&format!("- {} line {}{}\n", d.source, d.line, if d.detail.is_empty() { String::new() } else { format!(" ({})", d.detail) }));
            if !d.prompt.is_empty() {
                o.push_str(&format!("  - prompt: {}\n", d.prompt.replace('|', "\\|")));
            }
            if !d.answer.is_empty() {
                o.push_str(&format!("  - answer: {}\n", d.answer.replace('|', "\\|")));
            }
        }
    }
    o.push_str("\n## Notes\n\n- Rows are written unchanged (their `meta` included), in a shuffle drawn from the seed.\n- Each row is dropped for the first rule it breaks, in the order of the table.\n");
    o
}

// -------------------------------------------------------------------------------------------------------- tests

#[cfg(test)]
mod tests {
    use super::*;

    fn row(q: &str, a: &str, meta: Value) -> Row {
        Row { source: 0, line: 1, raw: String::new(), meta, record: Record { messages: vec![("user".into(), q.into()), ("assistant".into(), a.into())] } }
    }

    fn settings() -> Settings {
        Settings::new(vec![], PathBuf::from("/tmp/x.jsonl"))
    }

    fn words_len(r: &Row) -> Length {
        length_of(r, None, 1024).unwrap()
    }

    #[test]
    fn drop_cut_drops_rows_the_trainer_would_cut() {
        let mut s = settings();
        let long = Length { prompt: 100, answer: 1000, total: 1100, fits: true };
        let r = row("q", "a fine answer of several words", json!({}));
        assert_eq!(check_row(&s, &r, &long), None, "kept and counted as cut by default");
        s.drop_cut = true;
        assert_eq!(check_row(&s, &r, &long).map(|x| x.0), Some("too_long"));
    }

    #[test]
    fn words_keep_indic_vowel_signs_and_drop_punctuation() {
        assert_eq!(words("नमस्ते, दुनिया। Hello!"), vec!["नमस्ते", "दुनिया", "hello"]);
        assert_eq!(normalise("  A  b,\n C. "), "a b c");
    }

    #[test]
    fn repetition_catches_a_phrase_said_twice_and_a_repeated_clause_but_not_ordinary_text() {
        // the made-up phrase said twice (the Bengali answer of the first fine-tune)
        assert_eq!(repetition("আমার সোনার বাংলা আমার সোনার বাংলা"), 1.0);
        assert_eq!(repetition("দবঙ্গ দবঙ্গ দবঙ্গ দবঙ্গ"), 1.0);
        // tips that repeat the same clause
        let tips = "रोज़ पानी पिएं और समय पर सोएं। रोज़ पानी पिएं और समय पर सोएं। रोज़ पानी पिएं और समय पर सोएं। रोज़ व्यायाम करें।";
        assert!(repetition(tips) > 0.5, "{}", repetition(tips));
        // a paragraph written out twice
        let p = "Earth is the third planet from the Sun and the only place known to hold life. ";
        assert!(repetition(&format!("{p}{p}")) > 0.9);
        // ordinary text
        let ok = "ঘুমের সমস্যা সমাধানে ধাপে ধাপে এই রুটিনটি অনুসরণ করতে পারেন। প্রতিদিন একই সময়ে বিছানায় যান, যেন শরীরের জৈবিক ঘড়ি অভ্যস্ত হয়। ঘুমানোর আগে অন্তত এক ঘণ্টা স্ক্রিন এড়িয়ে চলুন।";
        assert!(repetition(ok) < 0.1, "{}", repetition(ok));
        assert_eq!(repetition("one two"), 0.0);
        assert_eq!(repetition(""), 0.0);
    }

    #[test]
    fn scripts_are_told_apart_and_languages_map_to_them() {
        assert_eq!(dominant_script("नमस्ते दुनिया hi").0, Some(Script::Devanagari));
        assert_eq!(dominant_script("வணக்கம்").0, Some(Script::Tamil));
        assert_eq!(dominant_script("سلام دنیا").0, Some(Script::Arabic));
        assert_eq!(dominant_script("`let x = 1;` 12 + 3").1, 0);
        assert_eq!(script_of_language("ben"), Some(Script::Bengali));
        assert_eq!(script_of_language("npi"), Some(Script::Devanagari));
        assert_eq!(script_of_language("Tamil"), Some(Script::Tamil));
        assert_eq!(script_of_language("en"), Some(Script::Latin));
        assert_eq!(iso2("pan"), "pa");
    }

    #[test]
    fn translation_targets_are_read_from_the_recipes_wordings() {
        let t = |p: &str| translation_target(p);
        assert_eq!(t("Translate into Hindi:\n\nGood morning.").as_deref(), Some("Hindi"));
        assert_eq!(t("Please translate this Tamil text into English.\n\nவணக்கம்").as_deref(), Some("English"));
        assert_eq!(t("नमस्ते\n\nTranslate the above into English.").as_deref(), Some("English"));
        assert_eq!(t("What is this in Bengali?\n\nHello").as_deref(), Some("Bengali"));
        assert_eq!(t("Translate from Urdu to English, keeping the meaning and tone:\n\nسلام").as_deref(), Some("English"));
        assert_eq!(t("Tell me about Hindi films").as_deref(), None);
    }

    #[test]
    fn script_rule_uses_the_translation_target_then_the_prompt_and_meta() {
        let into_hi = "Translate into Hindi:\n\nThe train leaves at nine in the morning.";
        assert!(script_mismatch(into_hi, "The train leaves at nine in the morning, yes.", &json!({})).is_some());
        assert!(script_mismatch(into_hi, "ट्रेन सुबह नौ बजे निकलती है।", &json!({})).is_none());
        // meta.to wins over the wording
        assert!(script_mismatch("x", "ট্রেন সকাল নটায় ছাড়ে, ঠিক আছে।", &json!({"to": "bn"})).is_none());
        // a Tamil question answered in English
        assert!(script_mismatch("இந்தியாவின் தலைநகரம் எது?", "The capital of India is New Delhi.", &json!({"language": "tam"})).is_some());
        // an English prompt about a Telugu village, answered in Telugu, Aya-style: meta.language allows it
        assert!(script_mismatch("Tell me about this village", "కొత్తవాడ విశాఖపట్నం జిల్లా గ్రామము", &json!({"language": "tel"})).is_none());
        // a number answer is not judged
        assert!(script_mismatch("இரண்டும் மூன்றும் எத்தனை?", "5", &json!({})).is_none());
    }

    #[test]
    fn junk_echo_empty_short_and_long_prompts_are_caught() {
        let s = settings();
        let check = |q: &str, a: &str| { let r = row(q, a, json!({})); check_row(&s, &r, &words_len(&r)).map(|x| x.0) };
        assert_eq!(check("q?", "<think>hmm</think> The answer is four, as two and two make four."), Some("template_junk"));
        assert_eq!(check("q?", "Four, since two plus two is four.<|im_end|>"), Some("template_junk"));
        assert_eq!(check("q?", "[{\"question\": \"a\", \"answer\": \"b\"}]"), Some("template_junk"));
        assert_eq!(check("q?", "   "), Some("empty_answer"));
        assert_eq!(check("What is the capital of France, please?", "What is the capital of France please"), Some("echo"));
        let src = "The train leaves at nine in the morning from platform two.";
        assert_eq!(check(&format!("Translate into Hindi:\n\n{src}"), src), Some("echo"));
        assert_eq!(check("Name a colour.", "Red"), Some("too_short"));
        assert_eq!(check("Name a colour.", "Red is a colour, and so is blue."), None);
        let mut s2 = settings();
        s2.min_answer_ratio = 0.2;
        let long = "word ".repeat(100);
        let r = row(&long, "a short answer of six words", json!({}));
        assert_eq!(check_row(&s2, &r, &words_len(&r)).map(|x| x.0), Some("too_short"));
        // the trainer's rule, as measured with a tokenizer
        let fits = Length { prompt: 1200, answer: 10, total: 1210, fits: false };
        assert_eq!(check_row(&s, &row("q", "a fine answer of several words", json!({})), &fits).map(|x| x.0), Some("prompt_too_long"));
        // the script rule only with --lang-match
        let r = row("இந்தியாவின் தலைநகரம் எது?", "The capital of India is New Delhi.", json!({}));
        assert_eq!(check_row(&s, &r, &words_len(&r)), None);
        s2.lang_match = true;
        s2.min_answer_ratio = 0.0;
        assert_eq!(check_row(&s2, &r, &words_len(&r)).map(|x| x.0), Some("script_mismatch"));
    }

    #[test]
    fn near_copies_need_a_like_prompt_and_a_like_answer() {
        let p = "Read the passage and answer. The river Ganga rises in the Himalayas at Gangotri and flows east across the plains of north India into the Bay of Bengal.";
        let (a1, a2) = ("It rises at Gangotri in the Himalayas.", "It flows into the Bay of Bengal.");
        let mut ix = NearIndex::default();
        ix.add(signature(p), signature(a1));
        assert_eq!(ix.find(&signature(p), &signature(a1), 0.8), Some(0));
        // the same passage, another answer: not a copy
        assert_eq!(ix.find(&signature(p), &signature(a2), 0.8), None);
        let p2 = p.replace("answer.", "answer:");
        assert_eq!(ix.find(&signature(&p2), &signature(a1), 0.8), Some(0));
        assert!(jaccard(&signature("a b c d e f g"), &signature("x y z w v u t")) < 0.2);
    }

    fn write(dir: &Path, name: &str, rows: &[Value]) -> PathBuf {
        let p = dir.join(name);
        std::fs::write(&p, rows.iter().map(|r| r.to_string() + "\n").collect::<String>()).unwrap();
        p
    }

    fn chat(q: &str, a: &str, meta: Value) -> Value {
        json!({"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": a}], "meta": meta})
    }

    #[test]
    fn a_run_drops_by_rule_keeps_out_the_held_out_caps_languages_and_is_deterministic() {
        let dir = std::env::temp_dir().join(format!("clean-unit-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mut rows = vec![];
        for i in 0..30 {
            rows.push(chat(&format!("प्रश्न संख्या {i}: भारत के बारे में बताइए"), &format!("भारत एक बड़ा देश है जिसमें {i} राज्य और अनेक भाषाएँ हैं।"), json!({"language": "hin", "licence": "Apache-2.0"})));
        }
        rows.push(chat("प्रश्न संख्या 3: भारत के बारे में बताइए", "भारत एक बड़ा देश है जिसमें 3 राज्य और अनेक भाषाएँ हैं।", json!({"language": "hin"}))); // exact dup
        rows.push(chat("q one", "<think>x</think> an answer with junk in it", json!({})));
        let a = write(&dir, "a.jsonl", &rows);
        let b = dir.join("b.jsonl");
        std::fs::write(&b, "not json\n".to_string() + &chat("Name the planets near the Sun, please.", "Mercury, Venus, Earth and Mars are the four nearest planets.", json!({"model": "m"})).to_string() + "\n").unwrap();
        let held = write(&dir, "held.jsonl", &[chat("प्रश्न संख्या 7: भारत के बारे में बताइए", "anything", json!({}))]);
        let mut s = Settings::new(vec![a.clone(), b.clone()], dir.join("out.jsonl"));
        s.exclude = vec![held];
        s.cap_per_language = Some(20);
        let mut events = vec![];
        let r = run(&s, |e| events.push(e)).unwrap();
        assert_eq!(r.dropped.get("held_out"), Some(&1));
        assert_eq!(r.dropped.get("exact_duplicate"), Some(&1));
        assert_eq!(r.dropped.get("template_junk"), Some(&1));
        assert_eq!(r.dropped.get("bad_row"), Some(&1));
        assert_eq!(r.dropped.get("language_cap"), Some(&9), "{:?}", r.dropped);
        assert_eq!(r.kept, 21);
        assert_eq!(r.languages.get("hi"), Some(&20));
        assert_eq!(r.sources[1].models, vec!["m".to_string()]);
        assert!(events.iter().any(|e| e["event"] == "rule" && e["rule"] == "held_out" && e["dropped"] == 1));
        let first = std::fs::read_to_string(&s.out).unwrap();
        assert_eq!(first.lines().count(), 21);
        assert!(!first.contains("प्रश्न संख्या 7:"), "the held-out row leaked");
        let card = std::fs::read_to_string(s.card_path()).unwrap();
        assert!(card.contains("| `language_cap` | 9 |") && card.contains("a.jsonl") && card.contains("Apache-2.0"), "{card}");
        // the cap drawn from one input only: the other's rows are all kept, and count towards it
        let other = write(&dir, "gen.jsonl", &(0..5).map(|i| chat(&format!("भारत के राज्य {i} कौन से हैं?"), &format!("उत्तर {i}: भारत में अट्ठाईस राज्य और आठ केंद्रशासित प्रदेश हैं।"), json!({"language": "hi"}))).collect::<Vec<_>>());
        let mut s3 = s.clone();
        s3.inputs.push(other);
        s3.cap_only = vec![a.clone()];
        s3.out = dir.join("out3.jsonl");
        let r3 = run(&s3, |_| {}).unwrap();
        assert_eq!(r3.languages.get("hi"), Some(&20));
        assert_eq!(r3.sources[2].kept, 5);
        assert_eq!(r3.dropped.get("language_cap"), Some(&14));
        // the same inputs and seed: the same file; another seed: another order
        run(&s, |_| {}).unwrap();
        assert_eq!(std::fs::read_to_string(&s.out).unwrap(), first);
        s.seed = 8;
        run(&s, |_| {}).unwrap();
        assert_ne!(std::fs::read_to_string(&s.out).unwrap(), first);
        std::fs::remove_dir_all(&dir).ok();
    }
}
