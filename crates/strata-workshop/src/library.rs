//! The Workshop's model library: ready-made recipes (a base model, a task and a Soup-format `soup.yaml`) for text,
//! code, tools, reasoning, vision, OCR, image, speech, music and video models.
//!
//! Two parts, both compiled into the binary:
//! - **Soup's recipe catalogue**, ported: Soup (<https://github.com/MakazhanAlpamys/Soup>, Apache-2.0) ships ready-made
//!   recipes for popular models. [`import_catalog`] reads its `recipes/catalog.py` (as text: nothing of Soup runs)
//!   and its table of each base model's architecture, and writes `data/soup-recipes.json`; `am2strata-workshop library
//!   import` does it again when Soup changes. The attribution is in `data/NOTICE-soup.md`.
//! - **Prastara's own recipes** (`data/prastara-recipes.json`): the Mula series and the media models Prastara runs
//!   (speech, music, image and video generation; speech and image understanding).
//!
//! Each recipe says what it can do ([`Ability`]) and whether this Workshop trains it now ([`Entry::status`]):
//! that is decided by the same Soup importer `train --soup` uses, so a recipe marked as training here does, and one
//! that is not names every reason (a task, a modality, an architecture or a setting not done yet).

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{anyhow, bail, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// One recipe.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Entry {
    pub name: String,
    /// the base model (a Hugging Face id)
    pub model: String,
    /// sft, dpo, grpo, tts, asr, …
    pub task: String,
    pub size: String,
    pub tags: Vec<String>,
    pub description: String,
    /// the Soup-format config
    pub yaml: String,
    /// soup or prastara
    pub origin: String,
    /// why the recipe's own source does not offer it (Soup marks a few)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unavailable: Option<String>,
    /// the base model's `model_type`, when known
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_type: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parameters: Option<u64>,
    /// Prastara: where the model runs in Prastara (a tab, an engine)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub serves: Option<String>,
    /// Prastara: abilities stated outright (otherwise they follow from the task, modality and tags)
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub abilities: Vec<Ability>,
}

/// What a model made with a recipe does.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Ability {
    Text,
    Code,
    Tools,
    Reasoning,
    Translation,
    Embedding,
    ImageUnderstanding,
    Ocr,
    ImageGeneration,
    AudioUnderstanding,
    SpeechRecognition,
    SpeechGeneration,
    MusicGeneration,
    VideoGeneration,
}

impl Ability {
    pub const ALL: [Ability; 14] = [Ability::Text, Ability::Code, Ability::Tools, Ability::Reasoning, Ability::Translation,
        Ability::Embedding, Ability::ImageUnderstanding, Ability::Ocr, Ability::ImageGeneration, Ability::AudioUnderstanding,
        Ability::SpeechRecognition, Ability::SpeechGeneration, Ability::MusicGeneration, Ability::VideoGeneration];
    pub fn as_str(self) -> &'static str {
        match self {
            Ability::Text => "text", Ability::Code => "code", Ability::Tools => "tools", Ability::Reasoning => "reasoning",
            Ability::Translation => "translation", Ability::Embedding => "embedding",
            Ability::ImageUnderstanding => "image-understanding", Ability::Ocr => "ocr",
            Ability::ImageGeneration => "image-generation", Ability::AudioUnderstanding => "audio-understanding",
            Ability::SpeechRecognition => "speech-recognition", Ability::SpeechGeneration => "speech-generation",
            Ability::MusicGeneration => "music-generation", Ability::VideoGeneration => "video-generation",
        }
    }
    pub fn from_s(s: &str) -> Result<Ability> {
        Ability::ALL.into_iter().find(|a| a.as_str() == s)
            .ok_or_else(|| anyhow!("ability {s:?}: one of {}", Ability::ALL.map(|a| a.as_str()).join(", ")))
    }
}

/// Whether this Workshop trains a recipe now.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "status", content = "why", rename_all = "kebab-case")]
pub enum Status {
    Trains,
    NotYet(Vec<String>),
}

/// The architectures the Workshop's trainer runs (as `model::LlamaModel` loads them).
pub const TRAINED_ARCHS: [&str; 4] = ["llama", "mistral", "qwen2", "qwen3"];

impl Entry {
    /// The config's `modality:` (text when it has none; Soup's TTS recipes say audio_out).
    pub fn modality(&self) -> String {
        serde_norway::from_str::<Value>(&self.yaml).ok().and_then(|v| v["modality"].as_str().map(str::to_string))
            .unwrap_or_else(|| if matches!(self.task.as_str(), "tts") { "audio_out".into() }
                               else if matches!(self.task.as_str(), "asr" | "transcribe") { "audio".into() }
                               else { "text".into() })
    }

    pub fn abilities(&self) -> Vec<Ability> {
        if !self.abilities.is_empty() {
            return self.abilities.clone();
        }
        let tag = |t: &str| self.tags.iter().any(|x| x.eq_ignore_ascii_case(t));
        let mut a = vec![];
        match (self.task.as_str(), self.modality().as_str()) {
            ("asr" | "transcribe", _) => a.push(Ability::SpeechRecognition),
            ("tts", _) | (_, "audio_out") => a.push(Ability::SpeechGeneration),
            ("embedding", _) => a.push(Ability::Embedding),
            (_, "vision") if tag("image-output") => a.push(Ability::ImageGeneration),
            (_, "vision") => a.push(Ability::ImageUnderstanding),
            (_, "audio") => a.push(Ability::AudioUnderstanding),
            _ => a.push(Ability::Text),
        }
        if tag("ocr") { a.push(Ability::Ocr); }
        if tag("code") || tag("coder") || tag("coding") { a.push(Ability::Code); }
        if tag("tools") || tag("tool-calling") || tag("function-calling") || tag("agent") { a.push(Ability::Tools); }
        if tag("reasoning") || tag("math") || self.task == "grpo" { a.push(Ability::Reasoning); }
        if tag("translation") || tag("speech-translation") { a.push(Ability::Translation); }
        a.sort();
        a.dedup();
        a
    }

    /// Whether this Workshop trains the recipe now, and if not, every reason.
    pub fn status(&self) -> Status {
        let mut why = vec![];
        if let Some(u) = &self.unavailable { why.push(format!("its source does not offer it: {u}")); }
        if self.task != "sft" {
            why.push(format!("task {}: the Workshop trains sft (supervised fine-tuning) only so far", self.task));
        }
        let m = self.modality();
        if m != "text" {
            why.push(format!("modality {m}: the Workshop trains text models only so far (vision is Mula M7)"));
        }
        if self.model.starts_with("mlx-community/") {
            why.push("an MLX (Apple) 4-bit checkpoint: the trainer needs the full-precision Hugging Face model".into());
        }
        if let Some(t) = self.model_type.clone().or_else(|| family_from_id(&self.model).map(str::to_string)) {
            if !TRAINED_ARCHS.contains(&t.as_str()) {
                why.push(format!("architecture {t}: the trainer runs {}", TRAINED_ARCHS.join(", ")));
            }
        }
        if why.is_empty() {
            // the same reader `train --soup` uses (with its stand-ins for 4-bit bases and other optimizers)
            match crate::soup::import(&self.yaml, Path::new("."), true) {
                Ok(_) => {}
                Err(crate::soup::SoupError::Unsupported(keys)) => why.extend(keys.into_iter().map(|k| format!("setting {k}"))),
                Err(e) => why.push(e.to_string()),
            }
        }
        if why.is_empty() { Status::Trains } else { Status::NotYet(why) }
    }

    /// For listings and the server: the entry with its modality, abilities and status.
    pub fn summary(&self) -> Value {
        let mut v = serde_json::to_value(self).unwrap_or_default();
        v["modality"] = Value::from(self.modality());
        v["abilities"] = Value::from(self.abilities().iter().map(|a| a.as_str()).collect::<Vec<_>>());
        v["workshop"] = serde_json::to_value(self.status()).unwrap_or_default();
        v
    }

    /// What `library init` writes beside the config: the data the recipe trains on.
    pub fn data_guide(&self) -> String {
        let fmt = serde_norway::from_str::<Value>(&self.yaml).ok().and_then(|v| v["data"]["format"].as_str().map(str::to_string))
            .unwrap_or_else(|| "auto".into());
        let m = self.modality();
        let fmt_key = match (fmt.as_str(), m.as_str()) {
            ("auto" | "llava" | "sharegpt4v", "vision") => "llava",
            ("auto", "audio") => "asr",
            ("auto", "audio_out") => "audio",
            (f, _) => f,
        };
        let example = match fmt_key {
            "llava" => r#"{"image": "cat.jpg", "messages": [{"role": "user", "content": "<image>\nWhat is in the picture?"}, {"role": "assistant", "content": "A cat asleep on a chair."}]}"#,
            "dpo" | "orpo" | "simpo" | "ipo" => r#"{"prompt": "…", "chosen": "the better answer", "rejected": "the worse answer"}"#,
            "kto" => r#"{"prompt": "…", "completion": "…", "label": true}"#,
            "asr" => r#"{"audio": "clip-001.wav", "text": "what is said in the clip"}"#,
            "audio" => r#"{"audio": "line-001.wav", "text": "the words spoken", "speaker": "narrator"}"#,
            "song" => r#"{"audio": "song-001.wav", "lyrics": "[verse]\n…", "tags": "calm, bansuri, 84 bpm"}"#,
            "image" => r#"{"image": "img-001.png", "caption": "a watercolour of a temple at dawn"}"#,
            "video" => r#"{"video": "clip-001.mp4", "caption": "a slow pan over paddy fields in rain"}"#,
            _ => r#"{"messages": [{"role": "user", "content": "…"}, {"role": "assistant", "content": "…"}]}"#,
        };
        let status = match self.status() {
            Status::Trains => format!("This Workshop trains it: `am2strata-workshop train --soup soup.yaml --model <folder of {}>`.", self.model),
            Status::NotYet(why) => format!("This Workshop does not train it yet:\n{}\n\nThe config runs in Soup itself.",
                                           why.iter().map(|w| format!("- {w}")).collect::<Vec<_>>().join("\n")),
        };
        format!("# {name}\n\n{desc}\n\n- Base model: `{model}`\n- Task: {task}; modality: {modality}\n- Abilities: {ab}\n{serves}\n\
                 ## Data\n\nOne JSON object per line in `data.train` (format `{fmt}`), for example:\n\n```json\n{example}\n```\n\n\
                 Media files are relative to the config's `image_dir`, `audio_dir` or `media_dir`. Training data can be \
                 made with `am2strata-workshop datagen` (qa, chat, instruct, code, indic, translate).\n\n## Training\n\n{status}\n",
                name = self.name, desc = self.description, model = self.model, task = self.task, modality = self.modality(),
                ab = self.abilities().iter().map(|a| a.as_str()).collect::<Vec<_>>().join(", "),
                serves = self.serves.as_ref().map(|s| format!("- Runs in Prastara: {s}\n")).unwrap_or_default())
    }
}

/// The architecture of a base model Soup's table could not look up (gated on the Hub), from its id.
pub fn family_from_id(id: &str) -> Option<&'static str> {
    let l = id.to_lowercase();
    Some(if l.contains("llama-4") || l.contains("llama4") { "llama4" }
         else if l.contains("embeddinggemma") { "gemma3_text" }
         else if l.contains("gemma-3") || l.contains("medgemma") { "gemma3" }
         else if l.contains("gemma-2") { "gemma2" }
         else if l.contains("gemma") { "gemma" }
         else if l.contains("meta-llama/") || l.contains("meditron") || l.contains("codellama") { "llama" }
         else { return None })
}

// --------------------------------------------------------------------------------------------------- the library

/// A recipe file (`data/*.json`): where it came from and its recipes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Catalogue {
    pub source: Value,
    pub recipes: Vec<Entry>,
}

const SOUP: &str = include_str!("../data/soup-recipes.json");
const PRASTARA: &str = include_str!("../data/prastara-recipes.json");

/// Every recipe: Prastara's first, then Soup's.
pub fn all() -> Vec<Entry> {
    let mut out = vec![];
    for text in [PRASTARA, SOUP] {
        let c: Catalogue = serde_json::from_str(text).expect("the built-in recipe files are valid");
        out.extend(c.recipes);
    }
    out
}

/// The built-in catalogues' sources (Soup's repository and commit; Prastara's).
pub fn sources() -> Vec<Value> {
    [PRASTARA, SOUP].iter().map(|t| serde_json::from_str::<Catalogue>(t).expect("valid").source).collect()
}

pub fn get(name: &str) -> Option<Entry> {
    all().into_iter().find(|e| e.name == name)
}

/// A filter for `library list`.
#[derive(Debug, Default, Clone)]
pub struct Query {
    pub text: Option<String>,
    pub task: Option<String>,
    pub ability: Option<Ability>,
    pub origin: Option<String>,
    pub trains_only: bool,
}

pub fn search(q: &Query) -> Vec<Entry> {
    all().into_iter().filter(|e| {
        q.task.as_deref().is_none_or(|t| e.task == t)
            && q.origin.as_deref().is_none_or(|o| e.origin == o)
            && q.ability.is_none_or(|a| e.abilities().contains(&a))
            && q.text.as_deref().is_none_or(|t| {
                let t = t.to_lowercase();
                e.name.to_lowercase().contains(&t) || e.model.to_lowercase().contains(&t)
                    || e.description.to_lowercase().contains(&t) || e.tags.iter().any(|x| x.to_lowercase() == t)
            })
            && (!q.trains_only || e.status() == Status::Trains)
    }).collect()
}

/// `library init`: the recipe's `soup.yaml` and `DATA.md` in `dir` (never over an existing file).
pub fn init(e: &Entry, dir: &Path) -> Result<Vec<std::path::PathBuf>> {
    std::fs::create_dir_all(dir)?;
    let files = [(dir.join("soup.yaml"), e.yaml.clone()), (dir.join("DATA.md"), e.data_guide())];
    for (p, _) in &files {
        if p.exists() { bail!("{} is already there: pick another folder", p.display()); }
    }
    for (p, text) in &files { std::fs::write(p, text)?; }
    Ok(files.into_iter().map(|f| f.0).collect())
}

// ------------------------------------------------------------------------------------------- importing Soup's

/// A Python literal as the catalogue writes them.
#[derive(Debug, Clone, PartialEq)]
enum Py {
    Str(String),
    Tuple(Vec<String>),
    /// anything else (a name, a call), as written
    Expr(String),
}

/// Soup's recipes from its `recipes/catalog.py` (read as text) and, when given, its table of base architectures
/// (`tests/fixtures/recipe_base_architectures.json`: model id → `model_type`, `parameters`).
pub fn import_catalog(py: &str, archs: Option<&Value>) -> Result<Vec<Entry>> {
    let start = py.find("RECIPES: Dict").or_else(|| py.find("RECIPES = {")).ok_or_else(|| anyhow!("no RECIPES table"))?;
    let body = &py[start..];
    let head = regex::Regex::new(r#"(?m)^    "([^"]+)": RecipeMeta\(\s*$"#)?;
    let mut out = vec![];
    for m in head.captures_iter(body) {
        let name = m[1].to_string();
        let from = m.get(0).unwrap().end();
        let end = body[from..].find("\n    ),").ok_or_else(|| anyhow!("{name}: no end of its RecipeMeta"))?;
        let fields = parse_fields(&body[from..from + end]).map_err(|e| anyhow!("{name}: {e}"))?;
        let s = |k: &str| match fields.get(k) {
            Some(Py::Str(s)) => Ok(s.clone()),
            other => Err(anyhow!("{name}: {k} is {other:?}, not a string")),
        };
        let tags = match fields.get("tags") {
            Some(Py::Tuple(t)) => t.clone(),
            Some(Py::Str(s)) => vec![s.clone()],
            _ => vec![],
        };
        let unavailable = match fields.get("unavailable_reason") {
            None => None,
            Some(Py::Str(s)) => Some(s.clone()),
            Some(Py::Tuple(t)) => Some(t.join(" ")),
            Some(Py::Expr(x)) => Some(format!("Soup refuses it on this kind of setup ({x})")),
        };
        let model = s("model")?;
        let arch = archs.and_then(|a| a.get(&model));
        out.push(Entry {
            task: s("task")?, size: s("size")?, tags, description: s("description")?, yaml: s("yaml_str")?,
            origin: "soup".into(), unavailable,
            model_type: arch.and_then(|a| a["model_type"].as_str()).map(str::to_string),
            parameters: arch.and_then(|a| a["parameters"].as_u64()),
            serves: None, abilities: vec![], model, name: name.clone(),
        });
    }
    if out.is_empty() { bail!("no recipes found in the catalogue"); }
    Ok(out)
}

fn parse_fields(src: &str) -> Result<BTreeMap<String, Py>> {
    let c: Vec<char> = src.chars().collect();
    let mut i = 0;
    let mut out = BTreeMap::new();
    let skip = |i: &mut usize| while *i < c.len() && (c[*i].is_whitespace() || c[*i] == ',') { *i += 1; };
    loop {
        skip(&mut i);
        if i >= c.len() { break; }
        let k0 = i;
        while i < c.len() && (c[i].is_alphanumeric() || c[i] == '_') { i += 1; }
        let key: String = c[k0..i].iter().collect();
        if key.is_empty() || i >= c.len() || c[i] != '=' { bail!("expected name= at {:?}", c[k0..(k0 + 30).min(c.len())].iter().collect::<String>()); }
        i += 1;
        while i < c.len() && c[i].is_whitespace() { i += 1; }
        let v = parse_value(&c, &mut i)?;
        out.insert(key, v);
    }
    Ok(out)
}

fn parse_value(c: &[char], i: &mut usize) -> Result<Py> {
    if c[*i] == '"' || c[*i] == '\'' {
        // adjacent literals are one string (the catalogue splits long ones across lines)
        let mut s = parse_str(c, i)?;
        loop {
            let mut j = *i;
            while j < c.len() && c[j].is_whitespace() { j += 1; }
            if j < c.len() && (c[j] == '"' || c[j] == '\'') { *i = j; s.push_str(&parse_str(c, i)?); } else { break; }
        }
        return Ok(Py::Str(s));
    }
    if c[*i] == '(' {
        // a tuple of strings, or adjacent strings in parentheses (one string)
        let save = *i;
        *i += 1;
        let (mut items, mut cur, mut comma) = (vec![], None::<String>, false);
        loop {
            while *i < c.len() && c[*i].is_whitespace() { *i += 1; }
            if *i >= c.len() { bail!("unclosed parenthesis"); }
            match c[*i] {
                ')' => { *i += 1; break; }
                ',' => { comma = true; if let Some(s) = cur.take() { items.push(s); } *i += 1; }
                '"' | '\'' => { let s = parse_str(c, i)?; cur = Some(cur.unwrap_or_default() + &s); }
                _ => { *i = save; return Ok(Py::Expr(parse_expr(c, i))); }
            }
        }
        if let Some(s) = cur { items.push(s); }
        return Ok(if comma || items.is_empty() { Py::Tuple(items) } else { Py::Str(items.concat()) });
    }
    Ok(Py::Expr(parse_expr(c, i)))
}

/// An expression up to the next comma outside brackets and strings.
fn parse_expr(c: &[char], i: &mut usize) -> String {
    let (s, mut depth) = (*i, 0i32);
    while *i < c.len() {
        match c[*i] {
            '(' | '[' | '{' => depth += 1,
            ')' | ']' | '}' => { if depth == 0 { break; } depth -= 1; }
            ',' if depth == 0 => break,
            '"' | '\'' => { let _ = parse_str(c, i); continue; }
            _ => {}
        }
        *i += 1;
    }
    c[s..*i].iter().collect::<String>().trim().to_string()
}

/// A Python string literal (single, double or triple quoted), with its escapes.
fn parse_str(c: &[char], i: &mut usize) -> Result<String> {
    let q = c[*i];
    let triple = *i + 2 < c.len() && c[*i + 1] == q && c[*i + 2] == q;
    *i += if triple { 3 } else { 1 };
    let mut out = String::new();
    loop {
        if *i >= c.len() { bail!("unclosed string"); }
        let ch = c[*i];
        if triple && ch == q && *i + 2 < c.len() && c[*i + 1] == q && c[*i + 2] == q { *i += 3; break; }
        if !triple && ch == q { *i += 1; break; }
        if ch == '\\' && *i + 1 < c.len() {
            let n = c[*i + 1];
            *i += 2;
            match n {
                '\n' => {}
                'n' => out.push('\n'),
                't' => out.push('\t'),
                '\\' => out.push('\\'),
                '"' => out.push('"'),
                '\'' => out.push('\''),
                other => { out.push('\\'); out.push(other); }
            }
            continue;
        }
        out.push(ch);
        *i += 1;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r#"
REASON = "x"
RECIPES: Dict[str, RecipeMeta] = {
    "tiny-sft": RecipeMeta(
        model="org/Tiny",
        task="sft",
        size="1B",
        tags=("llama", "sft", "code"),
        description=(
            "Tiny with a "
            "long description"
        ),
        yaml_str="""\
base: org/Tiny
task: sft

data:
  train: ./data/train.jsonl
  format: auto
  max_length: 2048

training:
  epochs: 3
  lr: 2e-4
  lora:
    r: 16
    alpha: 32
  quantization: 4bit

output: ./output
""",
    ),
    "eye-sft": RecipeMeta(
        model="org/Eye",
        task="sft",
        size="7B",
        tags=("ocr",),
        description="Eye, "
        "\"quoted\"",
        yaml_str="""\
base: org/Eye
task: sft
modality: vision
data:
  train: ./d.jsonl
  format: llava
""",
        unavailable_reason=refusal("4bit"),
    ),
}
"#;

    #[test]
    fn the_catalogue_is_read_as_text() {
        let archs = serde_json::json!({"org/Tiny": {"model_type": "llama", "parameters": 1000}});
        let r = import_catalog(SAMPLE, Some(&archs)).unwrap();
        assert_eq!(r.len(), 2);
        assert_eq!(r[0].description, "Tiny with a long description");
        assert_eq!(r[0].tags, ["llama", "sft", "code"]);
        assert!(r[0].yaml.starts_with("base: org/Tiny\ntask: sft\n"), "the backslash-newline is a continuation: {:?}", r[0].yaml);
        assert_eq!((r[0].model_type.as_deref(), r[0].parameters), (Some("llama"), Some(1000)));
        assert_eq!(r[1].tags, ["ocr"], "a one-item tuple");
        assert_eq!(r[1].description, "Eye, \"quoted\"");
        assert!(r[1].unavailable.as_deref().unwrap().contains("refusal(\"4bit\")"));
        assert_eq!(r[0].abilities(), [Ability::Text, Ability::Code]);
        assert_eq!(r[1].abilities(), [Ability::ImageUnderstanding, Ability::Ocr]);
        assert_eq!(r[0].status(), Status::Trains, "4-bit becomes a bf16 base, as train --soup-relax does");
        let Status::NotYet(why) = r[1].status() else { panic!() };
        assert!(why.iter().any(|w| w.contains("modality vision")) && why.iter().any(|w| w.contains("does not offer")), "{why:?}");
    }

    #[test]
    fn the_built_in_library_reads_and_answers() {
        let all = all();
        assert!(all.len() > 150, "{}", all.len());
        let mut names: Vec<&str> = all.iter().map(|e| e.name.as_str()).collect();
        names.sort();
        let n = names.len();
        names.dedup();
        assert_eq!(names.len(), n, "names are unique");
        for e in &all {
            assert!(serde_norway::from_str::<Value>(&e.yaml).is_ok(), "{}: its yaml parses", e.name);
            assert!(!e.abilities().is_empty(), "{}", e.name);
        }
        for a in [Ability::Code, Ability::Tools, Ability::ImageUnderstanding, Ability::Ocr, Ability::ImageGeneration,
                  Ability::SpeechRecognition, Ability::SpeechGeneration, Ability::MusicGeneration, Ability::VideoGeneration,
                  Ability::AudioUnderstanding, Ability::Translation] {
            assert!(!search(&Query { ability: Some(a), ..Default::default() }).is_empty(), "something for {a:?}");
        }
        let trains = search(&Query { trains_only: true, ..Default::default() });
        assert!(trains.iter().any(|e| e.name == "qwen3-8b-sft"), "Soup's Qwen3 8B SFT trains here");
        assert!(trains.iter().any(|e| e.name.starts_with("mula-code")), "Mula Code trains here");
        assert!(trains.iter().all(|e| e.modality() == "text" && e.task == "sft"));
        for not in ["gemma3-27b-sft", "llama4-scout-17b-sft", "qwen3-8b-sft-mlx"] {
            assert!(matches!(get(not).unwrap().status(), Status::NotYet(_)), "{not}");
        }
        assert_eq!(get("llama3.1-8b-sft").unwrap().status(), Status::Trains, "gated Llama 3 is a llama");
        let mula = search(&Query { origin: Some("prastara".into()), text: Some("mula".into()), ..Default::default() });
        assert!(mula.len() >= 7, "{}", mula.len());
        assert!(sources().iter().any(|s| s["licence"] == "Apache-2.0"));
    }

    #[test]
    fn init_writes_the_config_and_its_data_guide_once() {
        let d = std::env::temp_dir().join(format!("am2-library-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        let e = get("qwen3-8b-sft").unwrap();
        let files = init(&e, &d).unwrap();
        assert_eq!(std::fs::read_to_string(&files[0]).unwrap(), e.yaml);
        let guide = std::fs::read_to_string(&files[1]).unwrap();
        assert!(guide.contains("This Workshop trains it") && guide.contains("\"messages\""), "{guide}");
        assert!(init(&e, &d).unwrap_err().to_string().contains("already there"));
        let v = get("qwen2-vl-7b-sft").unwrap();
        let g = v.data_guide();
        assert!(g.contains("\"image\"") && g.contains("does not train it yet") && g.contains("modality vision"), "{g}");
        let _ = std::fs::remove_dir_all(&d);
    }
}
