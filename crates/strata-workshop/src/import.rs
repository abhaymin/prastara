//! `import`: a downloaded dataset (Hugging Face or any other) turned into the Workshop's chat JSONL, in Rust.
//!
//! The files are read as they were published — **Parquet**, JSON, JSONL (also `.gz`), CSV and TSV, plain lines (with
//! a regex that names the columns), text split into documents at blank lines, CoNLL-U treebanks and OPUS Moses zips —
//! also from inside a zip (`archive.zip!inner/*.jsonl`). A small per-dataset **mapping** (JSON) says how a record's
//! columns become a conversation:
//!
//! - `columns`: `system`/`user`/`assistant` templates over the record (`{col}`, `{a.b.0}`, `{col@table}` through a
//!   lookup table); a list of templates is drawn from by the seed, so instructions vary.
//! - `messages`: a column that already holds a conversation (`role_key`/`content_key`, roles renamed by `roles`).
//! - `transcript`: a column of one text with role markers (`USER:` … `ASSISTANT:` …) split into turns.
//! - `continue`: raw text (a book, a news document) cut at sentence ends into "here is a passage, continue it" with the
//!   real continuation as the answer — rule-based, no model.
//! - `entities`: tokens and BIO tags turned into "list the named entities" with the entities as the answer.
//!
//! Every output row is `{"messages": [...], "meta": {"source", "licence", "domain", ...}}`. With `limit`, a seeded
//! reservoir keeps that many rows of the whole stream (memory stays at `limit` rows). `joins` add the columns of
//! another table (by key), `group` folds the rows of one key into one record, `where` filters, `replace` rewrites the
//! text of every turn (for example a tool-call syntax into `<tool_call>` tags).

use std::collections::HashMap;
use std::io::{BufRead as _, Read as _};
use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context as _, Result};
use serde::Deserialize;
use serde_json::{json, Map, Value};

use crate::recipes::Draw;

// ------------------------------------------------------------------------------------------------------- mapping

/// One or more templates: a list is drawn from by the seed.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum Tpl {
    One(String),
    Many(Vec<String>),
}

impl Tpl {
    fn pick(&self, d: &mut Draw) -> &str {
        match self {
            Tpl::One(s) => s,
            Tpl::Many(v) if v.is_empty() => "",
            Tpl::Many(v) => &v[(d.next() % v.len() as u64) as usize],
        }
    }
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct Cond {
    pub col: String,
    #[serde(default)]
    pub eq: Option<Value>,
    #[serde(default)]
    pub ne: Option<Value>,
    #[serde(default, rename = "in")]
    pub in_: Option<Vec<Value>>,
    #[serde(default)]
    pub re: Option<String>,
    #[serde(default)]
    pub not_re: Option<String>,
    /// the text has at least / at most this many characters
    #[serde(default)]
    pub min_chars: Option<usize>,
    #[serde(default)]
    pub max_chars: Option<usize>,
    #[serde(default)]
    pub max_words: Option<usize>,
    #[serde(default)]
    pub min_num: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Join {
    /// the table (any format this reads; a glob or `zip!inner`)
    pub file: String,
    #[serde(default)]
    pub format: Option<String>,
    /// its key column
    pub key: String,
    /// this record's column that holds the key
    pub on: String,
    /// its columns appear as `<prefix>.<col>`
    pub prefix: String,
    /// keep the record when the key is not in the table (default: drop it)
    #[serde(default)]
    pub optional: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Group {
    pub by: Vec<String>,
    /// these columns become the group's values joined with `sep`; the others are the first record's
    pub collect: Vec<String>,
    #[serde(default = "default_sep")]
    pub sep: String,
    /// at most this many values per group (the rest are left out)
    #[serde(default)]
    pub max_items: Option<usize>,
}

fn default_sep() -> String {
    ", ".into()
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Kind {
    Columns,
    Messages,
    Transcript,
    Continue,
    Entities,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Mapping {
    /// the output's name (`<name>.jsonl`); default: the source's last part
    #[serde(default)]
    pub name: Option<String>,
    /// where the data is from (a Hugging Face id or a URL): `meta.source`
    pub source: String,
    pub licence: String,
    pub domain: String,
    /// globs (`~` is the home folder); `archive.zip!inner-glob` reads members of a zip
    pub files: Vec<String>,
    /// parquet, jsonl, json, csv, tsv, lines, docs, conllu, moses (default: from each file's extension)
    #[serde(default)]
    pub format: Option<String>,
    /// json: the dotted path to the list of records (default: the document itself, or its first list)
    #[serde(default)]
    pub json_path: Option<String>,
    /// lines: a regex whose named groups are the columns (default: the line is `text`)
    #[serde(default)]
    pub regex: Option<String>,
    /// csv/tsv: the column names when the file has no header row
    #[serde(default)]
    pub columns: Option<Vec<String>>,
    #[serde(default)]
    pub joins: Vec<Join>,
    #[serde(default)]
    pub group: Option<Group>,
    #[serde(default, rename = "where")]
    pub where_: Vec<Cond>,
    /// lookup tables for `{col@table}` in templates
    #[serde(default)]
    pub lookups: HashMap<String, HashMap<String, String>>,
    pub kind: Kind,

    // columns
    #[serde(default)]
    pub system: Option<Tpl>,
    #[serde(default)]
    pub user: Option<Tpl>,
    #[serde(default)]
    pub assistant: Option<Tpl>,

    // messages / transcript
    #[serde(default)]
    pub column: Option<String>,
    #[serde(default)]
    pub role_key: Option<String>,
    #[serde(default)]
    pub content_key: Option<String>,
    /// roles renamed; a role mapped to "" is dropped, a role not listed is kept as it is
    #[serde(default)]
    pub roles: HashMap<String, String>,
    /// transcript: the markers that start a turn, and the role each starts
    #[serde(default)]
    pub markers: Vec<(String, String)>,
    /// messages/transcript: a separate column that holds the system prompt
    #[serde(default)]
    pub system_column: Option<String>,
    /// messages/transcript: end the conversation at its last assistant turn (no dangling user turn)
    #[serde(default = "yes")]
    pub end_on_assistant: bool,

    // continue
    #[serde(default)]
    pub text: Option<String>,
    #[serde(default = "default_prompt_words")]
    pub prompt_words: usize,
    #[serde(default = "default_answer_words")]
    pub answer_words: usize,
    #[serde(default = "one")]
    pub per_doc: usize,
    /// skip this many words at the start of each document (a book's front matter)
    #[serde(default)]
    pub skip_words: usize,

    // entities
    #[serde(default)]
    pub tokens: Option<String>,
    #[serde(default)]
    pub tags: Option<String>,
    /// tag ids to names (`O`, `B-PER`, ...) when the tags are numbers
    #[serde(default)]
    pub tag_names: Vec<String>,
    /// entity type names in the answer (`PER` → `Person`)
    #[serde(default)]
    pub entity_names: HashMap<String, String>,

    /// templates for extra `meta` fields (`{"language": "{lang}"}`)
    #[serde(default)]
    pub meta: Map<String, Value>,
    /// regex → replacement on the text of every turn, in order
    #[serde(default)]
    pub replace: Vec<(String, String)>,
    /// a turn longer than this many characters drops the row
    #[serde(default)]
    pub max_turn_chars: Option<usize>,
    #[serde(default)]
    pub limit: Option<usize>,
    #[serde(default = "default_seed")]
    pub seed: u64,
}

fn yes() -> bool {
    true
}
fn one() -> usize {
    1
}
fn default_seed() -> u64 {
    7
}
fn default_prompt_words() -> usize {
    160
}
fn default_answer_words() -> usize {
    120
}

impl Mapping {
    pub fn out_name(&self) -> String {
        self.name.clone().unwrap_or_else(|| self.source.rsplit('/').next().unwrap_or("data").to_lowercase())
    }

    pub fn load(path: &Path) -> Result<Vec<Mapping>> {
        let text = std::fs::read_to_string(path).with_context(|| format!("{}", path.display()))?;
        let v: Value = serde_json::from_str(&text).with_context(|| format!("{}: not JSON", path.display()))?;
        let list = match v {
            Value::Array(a) => a,
            o => vec![o],
        };
        list.into_iter().enumerate()
            .map(|(i, m)| serde_json::from_value(m).with_context(|| format!("{}: mapping {}", path.display(), i + 1)))
            .collect()
    }
}

// ------------------------------------------------------------------------------------------------------- reading

fn home(p: &str) -> PathBuf {
    match p.strip_prefix("~/") {
        Some(rest) => std::env::var_os("HOME").map(PathBuf::from).unwrap_or_default().join(rest),
        None => PathBuf::from(p),
    }
}

/// A file to read: on disk, or a zip member.
#[derive(Debug, Clone)]
pub enum Source {
    File(PathBuf),
    Member(PathBuf, String),
}

impl Source {
    pub fn name(&self) -> String {
        match self {
            Source::File(p) => p.display().to_string(),
            Source::Member(z, m) => format!("{}!{m}", z.display()),
        }
    }
    fn inner_name(&self) -> String {
        match self {
            Source::File(p) => p.file_name().and_then(|x| x.to_str()).unwrap_or("").to_string(),
            Source::Member(_, m) => m.rsplit('/').next().unwrap_or(m).to_string(),
        }
    }
    fn bytes(&self) -> Result<Vec<u8>> {
        match self {
            Source::File(p) => std::fs::read(p).with_context(|| format!("{}", p.display())),
            Source::Member(z, m) => {
                let f = std::fs::File::open(z).with_context(|| format!("{}", z.display()))?;
                let mut a = zip::ZipArchive::new(f).with_context(|| format!("{}: not a zip", z.display()))?;
                let mut e = a.by_name(m).with_context(|| format!("{}: no member {m}", z.display()))?;
                let mut out = Vec::with_capacity(e.size() as usize);
                e.read_to_end(&mut out)?;
                Ok(out)
            }
        }
    }
}

/// The files a mapping's globs name, sorted (zip members in archive order).
pub fn expand(specs: &[String]) -> Result<Vec<Source>> {
    let mut out = vec![];
    for spec in specs {
        let (outer, inner) = match spec.split_once('!') {
            Some((a, b)) => (a, Some(b)),
            None => (spec.as_str(), None),
        };
        let pat = home(outer);
        let mut paths: Vec<PathBuf> = glob::glob(&pat.to_string_lossy()).with_context(|| format!("glob {spec}"))?.filter_map(|p| p.ok()).filter(|p| p.is_file()).collect();
        paths.sort();
        if paths.is_empty() {
            bail!("{spec}: no such file");
        }
        for p in paths {
            match inner {
                None => out.push(Source::File(p)),
                Some(ig) => {
                    let ip = glob::Pattern::new(ig).with_context(|| format!("glob {ig}"))?;
                    let f = std::fs::File::open(&p)?;
                    let a = zip::ZipArchive::new(f).with_context(|| format!("{}: not a zip", p.display()))?;
                    let names: Vec<String> = a.file_names().filter(|n| !n.ends_with('/') && ip.matches(n)).map(String::from).collect();
                    let mut names = names;
                    names.sort();
                    out.extend(names.into_iter().map(|n| Source::Member(p.clone(), n)));
                }
            }
        }
    }
    Ok(out)
}

fn format_of(src: &Source, forced: Option<&str>) -> String {
    if let Some(f) = forced {
        return f.to_string();
    }
    let n = src.inner_name().to_lowercase();
    let n = n.strip_suffix(".gz").unwrap_or(&n).to_string();
    for (ext, f) in [(".parquet", "parquet"), (".jsonl", "jsonl"), (".ndjson", "jsonl"), (".json", "json"), (".csv", "csv"), (".tsv", "tsv"),
                     (".conllu", "conllu"), (".txt.zip", "moses")] {
        if n.ends_with(ext) {
            return f.to_string();
        }
    }
    "lines".into()
}

fn gunzip_if(name: &str, bytes: Vec<u8>) -> Result<Vec<u8>> {
    if name.ends_with(".gz") {
        let mut out = vec![];
        flate2::read::MultiGzDecoder::new(&bytes[..]).read_to_end(&mut out)?;
        Ok(out)
    } else {
        Ok(bytes)
    }
}

/// A dotted path into a value (`a.b.0`).
pub fn get<'a>(v: &'a Value, path: &str) -> &'a Value {
    let mut cur = v;
    if let Some(x) = v.get(path) {
        return x;
    }
    for part in path.split('.') {
        cur = match cur {
            Value::Object(m) => m.get(part).unwrap_or(&Value::Null),
            Value::Array(a) => part.parse::<usize>().ok().and_then(|i| a.get(i)).unwrap_or(&Value::Null),
            _ => &Value::Null,
        };
    }
    cur
}

/// A value as template text: strings as they are, numbers and booleans printed, lists joined with ", ".
pub fn as_text(v: &Value) -> String {
    match v {
        Value::Null => String::new(),
        Value::String(s) => s.clone(),
        Value::Array(a) if a.iter().all(|x| !x.is_object() && !x.is_array()) => a.iter().map(as_text).collect::<Vec<_>>().join(", "),
        Value::Number(_) | Value::Bool(_) => v.to_string(),
        other => other.to_string(),
    }
}

/// Every record of a file, given to `f` one by one (`f` returns false to stop).
pub fn each_record(src: &Source, m_format: Option<&str>, json_path: Option<&str>, regex: Option<&regex::Regex>, columns: Option<&[String]>,
                   f: &mut dyn FnMut(Value) -> bool) -> Result<()> {
    let fmt = format_of(src, m_format);
    match fmt.as_str() {
        "parquet" => read_parquet(src, f),
        "jsonl" => {
            let bytes = gunzip_if(&src.inner_name(), src.bytes()?)?;
            for (i, line) in bytes.split(|b| *b == b'\n').enumerate() {
                let line = String::from_utf8_lossy(line);
                if line.trim().is_empty() {
                    continue;
                }
                let v: Value = serde_json::from_str(&line).with_context(|| format!("{} line {}", src.name(), i + 1))?;
                if !f(v) {
                    break;
                }
            }
            Ok(())
        }
        "json" => {
            let bytes = gunzip_if(&src.inner_name(), src.bytes()?)?;
            let v: Value = serde_json::from_slice(&bytes).with_context(|| format!("{}: not JSON", src.name()))?;
            let list = match json_path {
                Some(p) => get(&v, p).clone(),
                None => match v {
                    Value::Object(ref o) if !o.values().any(|x| x.is_array()) => Value::Array(vec![v.clone()]),
                    Value::Object(o) => o.into_iter().map(|(_, x)| x).find(|x| x.is_array()).unwrap_or(Value::Null),
                    other => other,
                },
            };
            match list {
                Value::Array(a) => {
                    for x in a {
                        if !f(x) {
                            break;
                        }
                    }
                    Ok(())
                }
                // a dict of records keyed by id
                Value::Object(o) => {
                    for (k, mut x) in o {
                        if let Value::Object(ref mut m) = x {
                            m.entry("_key").or_insert(Value::String(k));
                        }
                        if !f(x) {
                            break;
                        }
                    }
                    Ok(())
                }
                _ => bail!("{}: no list of records (json_path?)", src.name()),
            }
        }
        "csv" | "tsv" => {
            let bytes = gunzip_if(&src.inner_name(), src.bytes()?)?;
            let mut rb = csv::ReaderBuilder::new();
            rb.delimiter(if fmt == "tsv" { b'\t' } else { b',' }).flexible(true).has_headers(columns.is_none());
            if fmt == "tsv" {
                rb.quoting(false);
            }
            let mut r = rb.from_reader(&bytes[..]);
            let names: Vec<String> = match columns {
                Some(c) => c.to_vec(),
                None => r.headers()?.iter().map(|h| h.trim_start_matches('\u{feff}').to_string()).collect(),
            };
            for rec in r.records() {
                let rec = rec.with_context(|| src.name())?;
                let mut o = Map::new();
                for (k, v) in names.iter().zip(rec.iter()) {
                    o.insert(k.clone(), Value::String(v.to_string()));
                }
                if !f(Value::Object(o)) {
                    break;
                }
            }
            Ok(())
        }
        "lines" => {
            let bytes = gunzip_if(&src.inner_name(), src.bytes()?)?;
            let text = String::from_utf8_lossy(&bytes);
            let mut lines: Vec<&str> = text.lines().collect();
            // a file cut by a byte range: its last line is incomplete
            if !text.ends_with('\n') {
                lines.pop();
            }
            for line in lines {
                let v = match regex {
                    Some(re) => match re.captures(line) {
                        Some(c) => Value::Object(re.capture_names().flatten().filter_map(|n| c.name(n).map(|x| (n.to_string(), Value::String(x.as_str().to_string())))).collect()),
                        None => continue,
                    },
                    None => json!({"text": line}),
                };
                if !f(v) {
                    break;
                }
            }
            Ok(())
        }
        "docs" => {
            let bytes = gunzip_if(&src.inner_name(), src.bytes()?)?;
            let text = String::from_utf8_lossy(&bytes);
            let mut body: &str = &text;
            if !text.ends_with('\n') {
                body = text.rfind('\n').map_or("", |i| &text[..i]);
            }
            let mut doc = String::new();
            let mut n = 0usize;
            for line in body.lines().chain(std::iter::once("")) {
                if line.trim().is_empty() {
                    if !doc.trim().is_empty() {
                        n += 1;
                        if !f(json!({"text": doc.trim(), "doc": n})) {
                            break;
                        }
                    }
                    doc.clear();
                } else {
                    if !doc.is_empty() {
                        doc.push('\n');
                    }
                    doc.push_str(line.trim());
                }
            }
            Ok(())
        }
        "conllu" => read_conllu(src, f),
        "moses" => read_moses(src, f),
        other => bail!("{}: format {other:?} is not read (parquet, jsonl, json, csv, tsv, lines, docs, conllu, moses)", src.name()),
    }
}

fn read_parquet(src: &Source, f: &mut dyn FnMut(Value) -> bool) -> Result<()> {
    use parquet::file::reader::{FileReader, SerializedFileReader};
    // parquet needs random access: a file on disk is read in place, a zip member from memory
    let reader: Box<dyn FileReader> = match src {
        Source::File(p) => Box::new(SerializedFileReader::new(std::fs::File::open(p).with_context(|| format!("{}", p.display()))?)?),
        Source::Member(..) => Box::new(SerializedFileReader::new(bytes::Bytes::from(src.bytes()?))?),
    };
    for row in reader.get_row_iter(None)? {
        if !f(row?.to_json_value()) {
            break;
        }
    }
    Ok(())
}

fn read_conllu(src: &Source, f: &mut dyn FnMut(Value) -> bool) -> Result<()> {
    let bytes = src.bytes()?;
    let mut sent: Vec<Vec<String>> = vec![];
    let (mut text, mut id) = (String::new(), String::new());
    let mut emit = |sent: &mut Vec<Vec<String>>, text: &mut String, id: &mut String| -> bool {
        if sent.is_empty() {
            return true;
        }
        let forms: Vec<&str> = sent.iter().map(|t| t[1].as_str()).collect();
        let upos: Vec<String> = sent.iter().map(|t| format!("{}/{}", t[1], t[3])).collect();
        let lemmas: Vec<String> = sent.iter().map(|t| format!("{} → {}", t[1], t[2])).collect();
        let t = if text.is_empty() { forms.join(" ") } else { text.clone() };
        let v = json!({"text": t, "sent_id": id.clone(), "tokens": forms, "n_tokens": sent.len(), "upos_pairs": upos.join(" "),
                       "lemma_pairs": lemmas.join("\n"), "upos": sent.iter().map(|t| t[3].clone()).collect::<Vec<_>>()});
        sent.clear();
        text.clear();
        id.clear();
        f(v)
    };
    for line in bytes.lines() {
        let line = line?;
        if line.trim().is_empty() {
            if !emit(&mut sent, &mut text, &mut id) {
                return Ok(());
            }
        } else if let Some(t) = line.strip_prefix("# text = ") {
            text = t.trim().to_string();
        } else if let Some(t) = line.strip_prefix("# sent_id = ") {
            id = t.trim().to_string();
        } else if !line.starts_with('#') {
            let cols: Vec<String> = line.split('\t').map(String::from).collect();
            // multi-word token ranges (1-2) and empty nodes (1.1) are not words of the sentence's tag list
            if cols.len() >= 4 && !cols[0].contains('-') && !cols[0].contains('.') {
                sent.push(cols);
            }
        }
    }
    emit(&mut sent, &mut text, &mut id);
    Ok(())
}

/// An OPUS Moses zip (`corpus.l1-l2.txt.zip` with `corpus.l1-l2.l1` and `corpus.l1-l2.l2`, aligned by line).
fn read_moses(src: &Source, f: &mut dyn FnMut(Value) -> bool) -> Result<()> {
    let path = match src {
        Source::File(p) => p.clone(),
        Source::Member(..) => bail!("{}: a Moses zip inside a zip is not read", src.name()),
    };
    let file = std::fs::File::open(&path)?;
    let mut a = zip::ZipArchive::new(file).with_context(|| format!("{}: not a zip", path.display()))?;
    let names: Vec<String> = a.file_names().map(String::from).collect();
    // the pair from a member's name: corpus.l1-l2.l1
    let (stem, l1, l2) = names.iter().find_map(|n| {
        let parts: Vec<&str> = n.rsplitn(3, '.').collect();
        if parts.len() == 3 {
            let (last, pair) = (parts[0], parts[1]);
            if let Some((x, y)) = pair.split_once('-') {
                if last == x {
                    return Some((format!("{}.{pair}", parts[2]), x.to_string(), y.to_string()));
                }
            }
        }
        None
    }).ok_or_else(|| anyhow!("{}: no corpus.l1-l2.l1 member", path.display()))?;
    let mut read = |m: &str| -> Result<String> {
        let mut s = String::new();
        a.by_name(m).with_context(|| format!("{}: no {m}", path.display()))?.read_to_string(&mut s)?;
        Ok(s)
    };
    let (x, y) = (read(&format!("{stem}.{l1}"))?, read(&format!("{stem}.{l2}"))?);
    for (i, (s, t)) in x.lines().zip(y.lines()).enumerate() {
        if !f(json!({"src": s, "tgt": t, "src_lang": l1, "tgt_lang": l2, "line": i + 1})) {
            break;
        }
    }
    Ok(())
}

// ------------------------------------------------------------------------------------------------------ templates

/// `{path}`, `{path@table}` (a lookup), `{{` and `}}` for braces.
pub fn render(tpl: &str, rec: &Value, lookups: &HashMap<String, HashMap<String, String>>) -> Result<String> {
    let mut out = String::with_capacity(tpl.len() + 64);
    let mut it = tpl.chars().peekable();
    while let Some(c) = it.next() {
        match c {
            '{' if it.peek() == Some(&'{') => {
                it.next();
                out.push('{');
            }
            '}' if it.peek() == Some(&'}') => {
                it.next();
                out.push('}');
            }
            '{' => {
                let mut key = String::new();
                for c in it.by_ref() {
                    if c == '}' {
                        break;
                    }
                    key.push(c);
                }
                let (path, table) = match key.split_once('@') {
                    Some((p, t)) => (p, Some(t)),
                    None => (key.as_str(), None),
                };
                let v = as_text(get(rec, path));
                match table {
                    None => out.push_str(&v),
                    Some(t) => {
                        let tab = lookups.get(t).ok_or_else(|| anyhow!("no lookup table {t:?}"))?;
                        out.push_str(tab.get(&v).or_else(|| tab.get("*")).map(String::as_str).unwrap_or(&v));
                    }
                }
            }
            c => out.push(c),
        }
    }
    Ok(out)
}

struct Compiled {
    conds: Vec<(Cond, Option<regex::Regex>, Option<regex::Regex>)>,
    replace: Vec<(regex::Regex, String)>,
}

fn passes(c: &Compiled, rec: &Value) -> bool {
    c.conds.iter().all(|(cond, re, not_re)| {
        let v = get(rec, &cond.col);
        let t = as_text(v);
        let loose_eq = |want: &Value| v == want || as_text(want) == t;
        cond.eq.as_ref().is_none_or(loose_eq)
            && cond.ne.as_ref().is_none_or(|w| !loose_eq(w))
            && cond.in_.as_ref().is_none_or(|l| l.iter().any(loose_eq))
            && re.as_ref().is_none_or(|r| r.is_match(&t))
            && not_re.as_ref().is_none_or(|r| !r.is_match(&t))
            && cond.min_chars.is_none_or(|n| t.chars().count() >= n)
            && cond.max_chars.is_none_or(|n| t.chars().count() <= n)
            && cond.max_words.is_none_or(|n| t.split_whitespace().count() <= n)
            && cond.min_num.is_none_or(|n| t.trim().parse::<f64>().is_ok_and(|x| x >= n))
    })
}

// --------------------------------------------------------------------------------------------- record → messages

/// A sentence end: the text up to here can stop.
fn sentence_end(w: &str) -> bool {
    let w = w.trim_end_matches(['"', '\'', '”', '’', ')', ']', '»', '\u{201d}']);
    w.ends_with(['.', '!', '?', '।', '॥', '。', '！', '？', '؟', '۔'])
        && !matches!(w, "Mr." | "Mrs." | "Dr." | "St." | "Ms." | "Jr." | "Sr." | "No." | "vs." | "etc.")
}

/// Passages of a document cut at sentence ends: (prompt, continuation) pairs, up to `n` of them, seeded.
fn passages(text: &str, m: &Mapping, d: &mut Draw) -> Vec<(String, String)> {
    let words: Vec<&str> = text.split_whitespace().skip(m.skip_words).collect();
    let need = m.prompt_words + m.answer_words;
    if words.len() < need / 2 {
        return vec![];
    }
    let ends: Vec<usize> = (0..words.len()).filter(|i| sentence_end(words[*i])).map(|i| i + 1).collect();
    let mut out = vec![];
    let mut tried = 0;
    while out.len() < m.per_doc && tried < m.per_doc * 6 && ends.len() >= 3 {
        tried += 1;
        // a start right after a sentence end (or the document's start)
        let starts: Vec<usize> = std::iter::once(0).chain(ends.iter().copied()).filter(|s| *s + need / 2 <= words.len()).collect();
        if starts.is_empty() {
            break;
        }
        let s = starts[(d.next() % starts.len() as u64) as usize];
        // the prompt ends at the sentence end nearest prompt_words, the answer at the one nearest answer_words after it
        let near = |from: usize, want: usize| -> Option<usize> {
            ends.iter().copied().filter(|e| *e > from + want / 2 && *e <= from + want * 3 / 2).min_by_key(|e| e.abs_diff(from + want))
        };
        let Some(p) = near(s, m.prompt_words) else { continue };
        let Some(a) = near(p, m.answer_words) else { continue };
        let pr = words[s..p].join(" ");
        let an = words[p..a].join(" ");
        if out.iter().any(|(x, _): &(String, String)| x == &pr) {
            continue;
        }
        out.push((pr, an));
    }
    out
}

/// Tokens and BIO tags → the entities, in order: (type, text).
fn entities(tokens: &[String], tags: &[String]) -> Vec<(String, String)> {
    let mut out: Vec<(String, String)> = vec![];
    let mut cur: Option<(String, Vec<&str>)> = None;
    for (t, g) in tokens.iter().zip(tags) {
        let (bio, ty) = g.split_once('-').map_or((g.as_str(), ""), |(a, b)| (a, b));
        match bio {
            "B" | "S" => {
                if let Some((ty0, w)) = cur.take() {
                    out.push((ty0, w.join(" ")));
                }
                cur = Some((ty.to_string(), vec![t]));
            }
            "I" | "E" if cur.as_ref().is_some_and(|(t0, _)| t0 == ty) => cur.as_mut().unwrap().1.push(t),
            "I" | "E" => {
                if let Some((ty0, w)) = cur.take() {
                    out.push((ty0, w.join(" ")));
                }
                cur = Some((ty.to_string(), vec![t]));
            }
            _ => {
                if let Some((ty0, w)) = cur.take() {
                    out.push((ty0, w.join(" ")));
                }
            }
        }
    }
    if let Some((ty0, w)) = cur {
        out.push((ty0, w.join(" ")));
    }
    out
}

fn list_of_text(v: &Value) -> Vec<String> {
    match v {
        Value::Array(a) => a.iter().map(as_text).collect(),
        Value::String(s) => s.split_whitespace().map(String::from).collect(),
        _ => vec![],
    }
}

/// The conversations one record gives (several for `continue`).
fn conversations(m: &Mapping, rec: &Value, d: &mut Draw) -> Result<Vec<Vec<(String, String)>>> {
    let lk = &m.lookups;
    let mut convs = vec![];
    match m.kind {
        Kind::Columns => {
            let mut msgs = vec![];
            if let Some(s) = &m.system {
                let t = render(s.pick(d), rec, lk)?;
                if !t.trim().is_empty() {
                    msgs.push(("system".to_string(), t));
                }
            }
            let u = render(m.user.as_ref().ok_or_else(|| anyhow!("columns: no user template"))?.pick(d), rec, lk)?;
            let a = render(m.assistant.as_ref().ok_or_else(|| anyhow!("columns: no assistant template"))?.pick(d), rec, lk)?;
            msgs.push(("user".to_string(), u));
            msgs.push(("assistant".to_string(), a));
            convs.push(msgs);
        }
        Kind::Messages => {
            let col = m.column.as_deref().unwrap_or("messages");
            let (rk, ck) = (m.role_key.as_deref().unwrap_or("role"), m.content_key.as_deref().unwrap_or("content"));
            let list = match get(rec, col) {
                Value::Array(a) => a.clone(),
                // a conversation stored as a JSON string
                Value::String(s) => serde_json::from_str::<Vec<Value>>(s).unwrap_or_default(),
                _ => vec![],
            };
            let mut msgs = vec![];
            if let Some(sc) = &m.system_column {
                let s = as_text(get(rec, sc));
                if !s.trim().is_empty() {
                    msgs.push(("system".to_string(), s));
                }
            } else if let Some(s) = &m.system {
                msgs.push(("system".to_string(), render(s.pick(d), rec, lk)?));
            }
            for x in list {
                let role = as_text(get(&x, rk));
                let role = m.roles.get(&role).cloned().unwrap_or(role);
                if role.is_empty() {
                    continue;
                }
                let content = match get(&x, ck) {
                    Value::String(s) => s.clone(),
                    Value::Null => continue,
                    other => other.to_string(),
                };
                msgs.push((role, content));
            }
            convs.push(msgs);
        }
        Kind::Transcript => {
            let col = m.column.as_deref().ok_or_else(|| anyhow!("transcript: no column"))?;
            let text = as_text(get(rec, col));
            let mut msgs = vec![];
            if let Some(sc) = &m.system_column {
                let mut s = as_text(get(rec, sc));
                for (mk, _) in &m.markers {
                    if let Some(rest) = s.trim_start().strip_prefix(mk.as_str()) {
                        s = rest.to_string();
                    }
                }
                if !s.trim().is_empty() {
                    msgs.push(("system".to_string(), s.trim().to_string()));
                }
            }
            // split at the earliest marker each time
            let mut rest = text.as_str();
            let mut role: Option<String> = None;
            loop {
                let next = m.markers.iter().filter_map(|(mk, r)| rest.find(mk.as_str()).map(|i| (i, mk.len(), r))).min_by_key(|x| x.0);
                match next {
                    Some((i, l, r)) => {
                        if let Some(ro) = role.take() {
                            msgs.push((ro, rest[..i].trim().to_string()));
                        }
                        role = Some(r.clone());
                        rest = &rest[i + l..];
                    }
                    None => {
                        if let Some(ro) = role.take() {
                            msgs.push((ro, rest.trim().to_string()));
                        }
                        break;
                    }
                }
            }
            convs.push(msgs.into_iter().filter(|(r, c)| !r.is_empty() && !c.is_empty()).collect());
        }
        Kind::Continue => {
            let text = render(m.text.as_deref().unwrap_or("{text}"), rec, lk)?;
            for (p, a) in passages(&text, m, d) {
                let mut r = rec.clone();
                if let Value::Object(o) = &mut r {
                    o.insert("passage".into(), Value::String(p));
                    o.insert("continuation".into(), Value::String(a));
                }
                let u = render(m.user.as_ref().map(|t| t.pick(d)).unwrap_or("Continue this text:\n\n{passage}"), &r, lk)?;
                let an = render(m.assistant.as_ref().map(|t| t.pick(d)).unwrap_or("{continuation}"), &r, lk)?;
                let mut msgs = vec![];
                if let Some(s) = &m.system {
                    msgs.push(("system".to_string(), render(s.pick(d), &r, lk)?));
                }
                msgs.push(("user".to_string(), u));
                msgs.push(("assistant".to_string(), an));
                convs.push(msgs);
            }
        }
        Kind::Entities => {
            let toks = list_of_text(get(rec, m.tokens.as_deref().unwrap_or("tokens")));
            let raw = get(rec, m.tags.as_deref().unwrap_or("ner_tags"));
            let tags: Vec<String> = match raw {
                Value::Array(a) => a.iter().map(|t| match t.as_u64() {
                    Some(i) => m.tag_names.get(i as usize).cloned().unwrap_or_else(|| "O".into()),
                    None => as_text(t),
                }).collect(),
                other => list_of_text(other),
            };
            if toks.is_empty() || toks.len() != tags.len() {
                return Ok(vec![]);
            }
            let ents = entities(&toks, &tags);
            let mut r = rec.clone();
            let listed: Vec<String> = ents.iter().map(|(t, w)| format!("{w} ({})", m.entity_names.get(t).cloned().unwrap_or_else(|| t.clone()))).collect();
            if let Value::Object(o) = &mut r {
                o.insert("sentence".into(), Value::String(toks.join(" ")));
                o.insert("entities".into(), Value::String(if listed.is_empty() { "None.".into() } else { listed.join("; ") }));
                o.insert("n_entities".into(), json!(ents.len()));
            }
            let u = render(m.user.as_ref().map(|t| t.pick(d)).unwrap_or("List the named entities in this sentence, each with its type:\n\n{sentence}"), &r, lk)?;
            let a = render(m.assistant.as_ref().map(|t| t.pick(d)).unwrap_or("{entities}"), &r, lk)?;
            convs.push(vec![("user".into(), u), ("assistant".into(), a)]);
        }
    }
    if m.end_on_assistant {
        for c in convs.iter_mut() {
            while c.last().is_some_and(|(r, _)| r != "assistant") {
                c.pop();
            }
        }
    }
    Ok(convs)
}

// ------------------------------------------------------------------------------------------------------- the run

#[derive(Debug, Clone, Default)]
pub struct ImportReport {
    pub source: String,
    pub files: usize,
    pub records: usize,
    pub filtered: usize,
    pub unusable: usize,
    pub rows: usize,
    pub kept: usize,
}

/// A table read whole (for a join).
fn table(spec: &str, format: Option<&str>) -> Result<Vec<Value>> {
    let mut rows = vec![];
    for src in expand(&[spec.to_string()])? {
        each_record(&src, format, None, None, None, &mut |v| {
            rows.push(v);
            true
        })?;
    }
    Ok(rows)
}

fn finish(m: &Mapping, c: &Compiled, msgs: Vec<(String, String)>, rec: &Value) -> Result<Option<Value>> {
    let mut out = vec![];
    for (role, mut content) in msgs {
        for (re, rep) in &c.replace {
            content = re.replace_all(&content, rep.as_str()).into_owned();
        }
        let content = content.trim().to_string();
        if m.max_turn_chars.is_some_and(|n| content.chars().count() > n) {
            return Ok(None);
        }
        out.push(json!({"role": role, "content": content}));
    }
    let has_answer = out.iter().any(|x| x["role"] == "assistant" && !x["content"].as_str().unwrap_or("").is_empty());
    let has_user = out.iter().any(|x| x["role"] == "user" && !x["content"].as_str().unwrap_or("").is_empty());
    if !has_answer || !has_user {
        return Ok(None);
    }
    let mut meta = Map::new();
    meta.insert("source".into(), json!(m.source));
    meta.insert("licence".into(), json!(m.licence));
    meta.insert("domain".into(), json!(m.domain));
    for (k, v) in &m.meta {
        let t = match v {
            Value::String(s) => render(s, rec, &m.lookups)?,
            other => as_text(other),
        };
        if !t.is_empty() {
            meta.insert(k.clone(), Value::String(t));
        }
    }
    Ok(Some(json!({"messages": out, "meta": meta})))
}

/// Import one mapping: every row it gives (or a seeded `limit` of them) as JSON values, in a seeded order.
pub fn import(m: &Mapping, mut event: impl FnMut(Value)) -> Result<(Vec<Value>, ImportReport)> {
    let compile = |r: &Option<String>| -> Result<Option<regex::Regex>> { r.as_deref().map(regex::Regex::new).transpose().map_err(Into::into) };
    let c = Compiled {
        conds: m.where_.iter().map(|w| Ok((w.clone(), compile(&w.re)?, compile(&w.not_re)?))).collect::<Result<_>>()?,
        replace: m.replace.iter().map(|(a, b)| Ok((regex::Regex::new(a)?, b.clone()))).collect::<Result<_>>()?,
    };
    let line_re = compile(&m.regex)?;
    let mut joins = vec![];
    for j in &m.joins {
        let mut idx: HashMap<String, Value> = HashMap::new();
        for r in table(&j.file, j.format.as_deref())? {
            idx.entry(as_text(get(&r, &j.key))).or_insert(r);
        }
        event(json!({"event": "log", "message": format!("join {}: {} keys", j.file, idx.len())}));
        joins.push((j, idx));
    }
    let srcs = expand(&m.files)?;
    let mut rep = ImportReport { source: m.source.clone(), files: srcs.len(), ..Default::default() };
    let mut draw = Draw::new(m.seed);
    let mut pick = Draw::new(m.seed ^ 0xA5A5);
    let mut kept: Vec<Value> = vec![];
    let mut seen_rows = 0usize;
    let mut groups: Option<(Vec<String>, HashMap<String, (Value, Vec<Vec<String>>)>)> = m.group.as_ref().map(|_| (vec![], HashMap::new()));

    let mut handle = |rec: Value, rep: &mut ImportReport, kept: &mut Vec<Value>, seen_rows: &mut usize| -> Result<()> {
        for msgs in conversations(m, &rec, &mut draw)? {
            let Some(row) = finish(m, &c, msgs, &rec)? else {
                rep.unusable += 1;
                continue;
            };
            rep.rows += 1;
            *seen_rows += 1;
            // a seeded reservoir of `limit`
            match m.limit {
                Some(n) if kept.len() >= n => {
                    let j = (pick.next() % *seen_rows as u64) as usize;
                    if j < n {
                        kept[j] = row;
                    }
                }
                _ => kept.push(row),
            }
        }
        Ok(())
    };

    let mut err: Option<anyhow::Error> = None;
    for (fi, src) in srcs.iter().enumerate() {
        event(json!({"event": "progress", "done": fi, "total": srcs.len(), "message": format!("reading {}", src.name())}));
        each_record(src, m.format.as_deref(), m.json_path.as_deref(), line_re.as_ref(), m.columns.as_deref(), &mut |mut rec| {
            rep.records += 1;
            for (j, idx) in &joins {
                match idx.get(&as_text(get(&rec, &j.on))) {
                    Some(other) => {
                        if let Value::Object(o) = &mut rec {
                            o.insert(j.prefix.clone(), other.clone());
                        }
                    }
                    None if j.optional => {}
                    None => {
                        rep.filtered += 1;
                        return true;
                    }
                }
            }
            if !passes(&c, &rec) {
                rep.filtered += 1;
                return true;
            }
            if let (Some(g), Some((order, map))) = (m.group.as_ref(), groups.as_mut()) {
                let key = g.by.iter().map(|k| as_text(get(&rec, k))).collect::<Vec<_>>().join("\u{1f}");
                let vals: Vec<String> = g.collect.iter().map(|k| as_text(get(&rec, k))).collect();
                match map.get_mut(&key) {
                    Some((_, items)) => {
                        if g.max_items.is_none_or(|n| items.len() < n) {
                            items.push(vals);
                        }
                    }
                    None => {
                        order.push(key.clone());
                        map.insert(key, (rec, vec![vals]));
                    }
                }
                return true;
            }
            if let Err(e) = handle(rec, &mut rep, &mut kept, &mut seen_rows) {
                err = Some(e);
                return false;
            }
            true
        }).with_context(|| src.name())?;
        if let Some(e) = err.take() {
            return Err(e);
        }
    }
    if let (Some(g), Some((order, mut map))) = (m.group.as_ref(), groups) {
        for key in order {
            let (mut rec, items) = map.remove(&key).unwrap();
            if let Value::Object(o) = &mut rec {
                for (ci, col) in g.collect.iter().enumerate() {
                    let mut vs: Vec<String> = items.iter().map(|v| v[ci].clone()).filter(|s| !s.is_empty()).collect();
                    vs.dedup();
                    o.insert(col.clone(), Value::String(vs.join(&g.sep)));
                }
                o.insert("_count".into(), json!(items.len()));
            }
            handle(rec, &mut rep, &mut kept, &mut seen_rows)?;
        }
    }
    // a seeded order (the reservoir's order is not uniform)
    for i in (1..kept.len()).rev() {
        let j = (pick.next() % (i as u64 + 1)) as usize;
        kept.swap(i, j);
    }
    rep.kept = kept.len();
    Ok((kept, rep))
}

/// The card of one import.
pub fn card(m: &Mapping, r: &ImportReport, out: &Path) -> String {
    let name = out.file_name().and_then(|x| x.to_str()).unwrap_or("data");
    format!("# Dataset card: {name}\n\nImported by Prastara's Workshop (`am2strata-workshop import`), rule-based (no model wrote it).\n\n\
             - source: {}\n- licence: {}\n- domain: {}\n- kind: {:?}\n- files: {} ({})\n- records read: {}, filtered out: {}, unusable: {}\n\
             - rows made: {}, **kept: {}**{}\n- seed: {}\n",
            m.source, m.licence, m.domain, m.kind, r.files, m.files.join(", "), r.records, r.filtered, r.unusable, r.rows, r.kept,
            m.limit.map_or(String::new(), |n| format!(" (a seeded sample of at most {n})")), m.seed)
}

/// Import a mapping into `<dir>/<name>.jsonl` and its card.
pub fn import_to(m: &Mapping, dir: &Path, event: impl FnMut(Value)) -> Result<(PathBuf, ImportReport)> {
    let (rows, rep) = import(m, event)?;
    std::fs::create_dir_all(dir).with_context(|| format!("{}", dir.display()))?;
    let out = dir.join(format!("{}.jsonl", m.out_name()));
    let mut text = String::new();
    for r in &rows {
        text.push_str(&serde_json::to_string(r)?);
        text.push('\n');
    }
    std::fs::write(&out, text).with_context(|| format!("{}", out.display()))?;
    std::fs::write(crate::datagen::card_path(&out), card(m, &rep, &out))?;
    Ok((out, rep))
}

/// The first `n` records of each file a mapping names, as they are read (to write a mapping).
pub fn peek(specs: &[String], format: Option<&str>, json_path: Option<&str>, n: usize) -> Result<Vec<(String, Vec<Value>)>> {
    let mut out = vec![];
    for src in expand(specs)? {
        let mut v = vec![];
        each_record(&src, format, json_path, None, None, &mut |r| {
            v.push(r);
            v.len() < n
        })?;
        out.push((src.name(), v));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mapping(kind: &str, extra: Value) -> Mapping {
        let mut v = json!({"source": "t", "licence": "CC0-1.0", "domain": "test", "files": [], "kind": kind});
        for (k, x) in extra.as_object().unwrap() {
            v[k] = x.clone();
        }
        serde_json::from_value(v).unwrap()
    }

    #[test]
    fn templates_paths_and_lookups() {
        let rec = json!({"a": {"b": ["x", "y"]}, "lang": "hin", "n": 3});
        let lk: HashMap<String, HashMap<String, String>> = [("name".to_string(), [("hin".to_string(), "Hindi".to_string())].into())].into();
        assert_eq!(render("{a.b.1} {lang@name} {n} {{x}}", &rec, &lk).unwrap(), "y Hindi 3 {x}");
    }

    #[test]
    fn transcript_splits_at_markers() {
        let m = mapping("transcript", json!({"column": "chat", "markers": [["USER:", "user"], ["ASSISTANT:", "assistant"]]}));
        let rec = json!({"chat": "USER: hi there ASSISTANT: hello! USER: bye"});
        let c = conversations(&m, &rec, &mut Draw::new(1)).unwrap();
        assert_eq!(c[0], vec![("user".to_string(), "hi there".to_string()), ("assistant".to_string(), "hello!".to_string())]);
    }

    #[test]
    fn entities_from_bio() {
        let t: Vec<String> = "Ram went to New Delhi".split(' ').map(String::from).collect();
        let g: Vec<String> = ["B-PER", "O", "O", "B-LOC", "I-LOC"].iter().map(|s| s.to_string()).collect();
        assert_eq!(entities(&t, &g), vec![("PER".into(), "Ram".into()), ("LOC".into(), "New Delhi".into())]);
    }

    #[test]
    fn continuation_ends_at_a_sentence() {
        let m = mapping("continue", json!({"prompt_words": 8, "answer_words": 6, "per_doc": 1}));
        let text = (0..40).map(|i| format!("Word{i} is here and it goes on. ")).collect::<String>();
        let p = passages(&text, &m, &mut Draw::new(3));
        assert_eq!(p.len(), 1);
        assert!(p[0].0.ends_with('.') && p[0].1.ends_with('.'));
    }

    #[test]
    fn reads_csv_jsonl_lines_and_conllu_and_parquet() {
        let dir = std::env::temp_dir().join(format!("import-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("a.csv"), "q,a\n\"hi, you\",yo\n").unwrap();
        std::fs::write(dir.join("b.jsonl"), "{\"q\":\"x\",\"a\":\"y\"}\n").unwrap();
        std::fs::write(dir.join("c.conllu"), "# text = Hola mundo\n1\tHola\thola\tINTJ\t_\n2\tmundo\tmundo\tNOUN\t_\n\n").unwrap();
        std::fs::write(dir.join("d.dict"), "hello HH AH0 L OW1\ncut").unwrap();
        let m = mapping("columns", json!({"files": [dir.join("a.csv"), dir.join("b.jsonl")], "user": "{q}", "assistant": "{a}"}));
        let (rows, r) = import(&m, |_| {}).unwrap();
        assert_eq!((r.records, rows.len()), (2, 2));
        assert!(rows.iter().any(|x| x["messages"][0]["content"] == "hi, you"));
        assert_eq!(rows[0]["meta"]["domain"], "test");
        let m = mapping("columns", json!({"files": [dir.join("c.conllu")], "user": "Tag: {text}", "assistant": "{upos_pairs}"}));
        let (rows, _) = import(&m, |_| {}).unwrap();
        assert_eq!(rows[0]["messages"][1]["content"], "Hola/INTJ mundo/NOUN");
        let m = mapping("columns", json!({"files": [dir.join("d.dict")], "regex": "^(?P<w>\\S+) (?P<p>.+)$", "user": "{w}", "assistant": "{p}"}));
        let (rows, _) = import(&m, |_| {}).unwrap();
        assert_eq!(rows.len(), 1, "the cut last line is dropped");

        // parquet: write a small file and read it back
        use parquet::data_type::ByteArrayType;
        use parquet::file::writer::SerializedFileWriter;
        let schema = std::sync::Arc::new(parquet::schema::parser::parse_message_type("message s { REQUIRED BYTE_ARRAY q (UTF8); REQUIRED BYTE_ARRAY a (UTF8); }").unwrap());
        let f = std::fs::File::create(dir.join("e.parquet")).unwrap();
        let mut w = SerializedFileWriter::new(f, schema, Default::default()).unwrap();
        let mut rg = w.next_row_group().unwrap();
        for vals in [["p1", "p2"], ["a1", "a2"]] {
            let mut col = rg.next_column().unwrap().unwrap();
            col.typed::<ByteArrayType>().write_batch(&vals.map(|s| s.into()), None, None).unwrap();
            col.close().unwrap();
        }
        rg.close().unwrap();
        w.close().unwrap();
        let m = mapping("columns", json!({"files": [dir.join("e.parquet")], "user": "{q}", "assistant": "{a}", "limit": 1}));
        let (rows, r) = import(&m, |_| {}).unwrap();
        assert_eq!((r.rows, rows.len()), (2, 1));
        std::fs::remove_dir_all(&dir).ok();
    }
}
