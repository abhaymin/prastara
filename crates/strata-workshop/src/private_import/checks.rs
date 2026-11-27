//! The checks an imported item goes through before release, behind the [`Classifier`] trait, and the fixed rules that
//! turn a check model's scores into a decision (`strata_safety::import`).
//!
//! The check models are local folders the owner installs (nothing is downloaded): `<private>/check-models/<check>/`
//! with the Hugging Face `config.json`, `model.safetensors` (or shards), `prastara_check.json` (which labels flag an
//! item), and for text `tokenizer.json`; for images `preprocessor_config.json` is read when present. They run on
//! candle, on the CPU ([`super::nets`]).
//!
//! **Fail closed**: an error, a timeout, an unreadable or undecodable input, scores that are not numbers or a label
//! set that does not match the sidecar is a rejection, never a pass. A check that timed out is not asked again (its
//! thread may still be busy): the import stops, and nothing is released.

use std::path::{Path, PathBuf};
use std::sync::{mpsc, Arc, Mutex};
use std::time::Duration;

use anyhow::{anyhow, bail, Context, Result};
use serde_json::Value;
use strata_safety::import as rules;

use super::nets;

/// What a check reads.
#[derive(Clone)]
pub enum Input {
    Image(Arc<image::RgbImage>),
    Text(Arc<str>),
}

/// Label scores in the model's order: probabilities, or for a one-output model the value itself (an age in years).
pub type Scores = Vec<(String, f32)>;

/// A check model. `classify` may be slow; it runs on its own thread under a timeout ([`Guarded`]).
pub trait Classifier: Send {
    fn classify(&mut self, input: &Input) -> Result<Scores>;
}

/// The checks, each with its own model folder.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CheckKind {
    /// minor safety of an image: does it show, or sexualise, a minor
    ImageMinor,
    /// the estimated age of the people in an image
    Age,
    /// does an image show a real, identifiable person
    ImageRealPerson,
    /// minor safety of a text (beside the fixed word rules, `strata_safety::check`)
    TextMinor,
    /// does a text name or describe a real, identifiable person
    TextRealPerson,
}

impl CheckKind {
    pub const ALL: [CheckKind; 5] = [CheckKind::ImageMinor, CheckKind::Age, CheckKind::ImageRealPerson, CheckKind::TextMinor, CheckKind::TextRealPerson];

    /// The folder under `check-models/` and the name in `prastara_check.json`'s `"check"`.
    pub fn id(self) -> &'static str {
        match self {
            CheckKind::ImageMinor => "image-minor",
            CheckKind::Age => "age",
            CheckKind::ImageRealPerson => "image-real-person",
            CheckKind::TextMinor => "text-minor",
            CheckKind::TextRealPerson => "text-real-person",
        }
    }

    /// How people say it ("the <x> check model is not installed").
    pub fn label(self) -> &'static str {
        match self {
            CheckKind::ImageMinor => "image minor-safety",
            CheckKind::Age => "age estimate",
            CheckKind::ImageRealPerson => "image real-person",
            CheckKind::TextMinor => "text minor-safety",
            CheckKind::TextRealPerson => "text real-person",
        }
    }

    pub fn is_image(self) -> bool {
        matches!(self, CheckKind::ImageMinor | CheckKind::Age | CheckKind::ImageRealPerson)
    }

    /// A flag of this check means a minor (the item is deleted and its hash blocked).
    pub fn is_minor(self) -> bool {
        matches!(self, CheckKind::ImageMinor | CheckKind::Age | CheckKind::TextMinor)
    }

    pub fn of_id(id: &str) -> Option<CheckKind> {
        CheckKind::ALL.into_iter().find(|k| k.id() == id)
    }
}

/// What a check model's scores mean, from its `prastara_check.json`.
#[derive(Debug, Clone, PartialEq)]
pub struct Rule {
    /// labels that flag an item when together they score at least `threshold` (exact names, any case)
    pub flag_labels: Vec<String>,
    pub threshold: f32,
    /// age only: labels that say no person is shown (the age check then has nothing to judge)
    pub no_person_labels: Vec<String>,
    /// age only, a one-output model: an age under 18 plus this margin flags the item
    pub age_margin: f32,
}

impl Rule {
    /// The rule from the sidecar, checked against the model's own labels: a flag label it does not have is an error
    /// (the model would never flag anything). For the age check, without `flag_labels`, every label whose age range
    /// reaches under 18 flags; a label that names no age must then be listed in `no_person_labels` or `adult_labels`.
    pub fn from_sidecar(kind: CheckKind, side: &Value, labels: &[String]) -> Result<Rule> {
        if side["check"].as_str() != Some(kind.id()) {
            bail!("prastara_check.json says \"check\": {}, not \"{}\"", side["check"], kind.id());
        }
        let list = |k: &str| -> Vec<String> { side[k].as_array().into_iter().flatten().filter_map(|x| x.as_str()).map(str::to_string).collect() };
        let has = |l: &str| labels.iter().any(|x| x.eq_ignore_ascii_case(l));
        let (default, max) = if kind.is_minor() { (rules::MINOR_THRESHOLD_DEFAULT, rules::MINOR_THRESHOLD_MAX) }
                             else { (rules::REAL_THRESHOLD_DEFAULT, rules::REAL_THRESHOLD_MAX) };
        let threshold = rules::threshold(side["threshold"].as_f64().map(|t| t as f32), default, max);
        let no_person = list("no_person_labels");
        let margin = side["age_margin"].as_f64().map(|m| m as f32).filter(|m| m.is_finite()).unwrap_or(rules::AGE_MARGIN_MIN).max(rules::AGE_MARGIN_MIN);
        let mut flag = list("flag_labels");
        for l in flag.iter().chain(no_person.iter()) {
            if !has(l) {
                bail!("the model has no label {l:?} (its labels: {})", labels.join(", "));
            }
        }
        if kind == CheckKind::Age && labels.len() == 1 {
            // a model that answers in years
            return Ok(Rule { flag_labels: vec![], threshold, no_person_labels: vec![], age_margin: margin });
        }
        if kind == CheckKind::Age && flag.is_empty() {
            let adult = list("adult_labels");
            for l in labels {
                let named = no_person.iter().chain(adult.iter()).any(|x| x.eq_ignore_ascii_case(l));
                match rules::age_label_minor(l) {
                    Some(true) => flag.push(l.clone()),
                    Some(false) => {}
                    None if named => {}
                    None => bail!("the age model's label {l:?} names no age: list it in flag_labels, adult_labels or no_person_labels"),
                }
            }
        }
        if flag.is_empty() {
            bail!("prastara_check.json lists no flag_labels: the labels that flag an item");
        }
        Ok(Rule { flag_labels: flag, threshold, no_person_labels: no_person, age_margin: margin })
    }
}

/// What a check said of one item.
#[derive(Debug, Clone, PartialEq)]
pub enum Decision {
    Pass,
    /// flagged by the check (the reason names the check, never the content)
    Flag,
    /// age only: no person is shown
    NoPerson,
}

/// The decision from scores (fail closed: no scores, a score that is not a number, or a label it does not know is an error).
pub fn decide(kind: CheckKind, rule: &Rule, scores: &Scores) -> Result<Decision> {
    if scores.is_empty() || scores.iter().any(|(_, s)| !s.is_finite()) {
        bail!("the {} check gave no usable scores", kind.label());
    }
    if kind == CheckKind::Age && scores.len() == 1 {
        let age = scores[0].1;
        return Ok(if age < rules::ADULT_AGE + rule.age_margin { Decision::Flag } else { Decision::Pass });
    }
    let sum = |names: &[String]| -> f32 {
        scores.iter().filter(|(l, _)| names.iter().any(|n| n.eq_ignore_ascii_case(l))).map(|(_, s)| *s).sum()
    };
    for l in &rule.flag_labels {
        if !scores.iter().any(|(x, _)| x.eq_ignore_ascii_case(l)) {
            bail!("the {} check gave no score for {l:?}", kind.label());
        }
    }
    if sum(&rule.flag_labels) >= rule.threshold {
        return Ok(Decision::Flag);
    }
    if kind == CheckKind::Age && !rule.no_person_labels.is_empty() && sum(&rule.no_person_labels) >= 0.5 {
        return Ok(Decision::NoPerson);
    }
    Ok(Decision::Pass)
}

/// A classifier on its own thread, under a timeout. After a timeout it is broken: every later call fails at once.
pub struct Guarded {
    inner: Arc<Mutex<Box<dyn Classifier>>>,
    timeout: Duration,
    broken: bool,
}

impl Guarded {
    pub fn new(c: Box<dyn Classifier>, timeout: Duration) -> Guarded {
        Guarded { inner: Arc::new(Mutex::new(c)), timeout, broken: false }
    }

    pub fn broken(&self) -> bool {
        self.broken
    }

    /// The scores, or an error (the classifier's own, a panic, or the timeout).
    pub fn call(&mut self, input: &Input) -> Result<Scores> {
        if self.broken {
            bail!("the check model timed out earlier");
        }
        let (tx, rx) = mpsc::channel();
        let inner = self.inner.clone();
        let input = input.clone();
        std::thread::Builder::new().name("private-check".into()).spawn(move || {
            let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| match inner.lock() {
                Ok(mut c) => c.classify(&input),
                Err(_) => Err(anyhow!("the check model failed earlier")),
            }));
            let _ = tx.send(r.unwrap_or_else(|_| Err(anyhow!("the check model crashed"))));
        })?;
        match rx.recv_timeout(self.timeout) {
            Ok(r) => r,
            Err(_) => {
                self.broken = true;
                bail!("the check model did not answer within {} s", self.timeout.as_secs_f32())
            }
        }
    }
}

/// One check: its kind, its rule and its model.
pub struct Check {
    pub kind: CheckKind,
    pub rule: Rule,
    pub model: Guarded,
}

impl Check {
    /// The decision for one input; any failure is an error (the caller rejects the item).
    pub fn run(&mut self, input: &Input) -> Result<Decision> {
        let s = self.model.call(input)?;
        decide(self.kind, &self.rule, &s)
    }
}

/// Why a check could not be had: not installed (its folder or weights missing) or broken (it would not load).
#[derive(Debug)]
pub enum Unavailable {
    NotInstalled(CheckKind),
    Unusable(CheckKind, String),
}

impl Unavailable {
    pub fn message(&self) -> String {
        match self {
            Unavailable::NotInstalled(k) => format!("the {} check model is not installed; nothing is released", k.label()),
            Unavailable::Unusable(k, why) => format!("the {} check model could not be loaded ({why}); nothing is released", k.label()),
        }
    }
}

// ------------------------------------------------------------------------------------------------- the real models

/// A ViT image classifier from a model folder.
struct VitClassifier(nets::Vit);

impl Classifier for VitClassifier {
    fn classify(&mut self, input: &Input) -> Result<Scores> {
        match input {
            Input::Image(img) => self.0.classify(img),
            Input::Text(_) => bail!("an image check was given text"),
        }
    }
}

/// A BERT-family text classifier: the text in windows of at most the model's length; the item's score for each label
/// is its highest over the windows (one window that flags flags the item).
struct TextClassifier {
    net: nets::TextNet,
    tok: tokenizers::Tokenizer,
}

/// At most this many windows of one text (a longer text is refused: it is not checked in part).
pub const MAX_WINDOWS: usize = 4096;

impl TextClassifier {
    fn windows(&self, text: &str) -> Result<Vec<Vec<u32>>> {
        let mut out = vec![];
        let mut stack: Vec<String> = chunk_chars(text, 1500);
        stack.reverse();
        while let Some(c) = stack.pop() {
            let enc = self.tok.encode(c.as_str(), true).map_err(|e| anyhow!("tokenizing: {e}"))?;
            let ids = enc.get_ids().to_vec();
            if ids.len() <= self.net.max_len {
                if !ids.is_empty() {
                    out.push(ids);
                }
            } else {
                let n = c.chars().count();
                if n < 2 {
                    bail!("a piece of the text is longer than the model's window");
                }
                let (a, b): (String, String) = (c.chars().take(n / 2).collect(), c.chars().skip(n / 2).collect());
                stack.push(b);
                stack.push(a);
            }
            if out.len() > MAX_WINDOWS {
                bail!("the text is longer than {MAX_WINDOWS} windows of the model; it is not checked in part");
            }
        }
        if out.is_empty() {
            bail!("the text has nothing to check");
        }
        Ok(out)
    }
}

/// `text` in pieces of about `n` characters, cut at whitespace when there is some near the end.
pub fn chunk_chars(text: &str, n: usize) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let mut out = vec![];
    let mut i = 0;
    while i < chars.len() {
        let mut j = (i + n).min(chars.len());
        if j < chars.len() {
            if let Some(k) = (i + n / 2..j).rev().find(|k| chars[*k].is_whitespace()) {
                j = k + 1;
            }
        }
        out.push(chars[i..j].iter().collect());
        i = j;
    }
    out
}

impl Classifier for TextClassifier {
    fn classify(&mut self, input: &Input) -> Result<Scores> {
        let Input::Text(t) = input else { bail!("a text check was given an image") };
        let mut best: Option<Scores> = None;
        for w in self.windows(t)? {
            let s = self.net.classify(&w)?;
            best = Some(match best {
                None => s,
                Some(b) => b.into_iter().zip(s).map(|((l, x), (_, y))| (l, x.max(y))).collect(),
            });
        }
        best.ok_or_else(|| anyhow!("no scores"))
    }
}

/// The installed check model of `kind` under `models_dir` (`<models_dir>/<kind.id()>/`).
pub fn load(kind: CheckKind, models_dir: &Path, timeout: Duration) -> std::result::Result<Check, Unavailable> {
    let dir: PathBuf = models_dir.join(kind.id());
    if !dir.join("config.json").is_file() {
        return Err(Unavailable::NotInstalled(kind));
    }
    let bad = |e: anyhow::Error| Unavailable::Unusable(kind, format!("{e:#}"));
    if nets::safetensors_files(&dir).is_empty() {
        if dir.join("model.onnx").is_file() {
            return Err(Unavailable::Unusable(kind, "it is an ONNX model; check models are read as safetensors (candle, on the CPU)".into()));
        }
        return Err(Unavailable::NotInstalled(kind));
    }
    let read = |n: &str| -> Result<Value> {
        let p = dir.join(n);
        serde_json::from_str(&std::fs::read_to_string(&p).with_context(|| format!("{}", p.display()))?).with_context(|| format!("{}", p.display()))
    };
    let made = (|| -> Result<Check> {
        let cfg = read("config.json")?;
        let side = read("prastara_check.json").context("prastara_check.json says which of the model's labels flag an item")?;
        let arch = cfg["architectures"][0].as_str().unwrap_or("").to_string();
        let w = nets::Weights::load(&dir)?;
        let (model, labels): (Box<dyn Classifier>, Vec<String>) = if kind.is_image() {
            if arch != "ViTForImageClassification" {
                bail!("{arch:?}: an image check model is a ViTForImageClassification");
            }
            let pre = read("preprocessor_config.json").unwrap_or(Value::Null);
            let v = nets::Vit::new(&cfg, &pre, w)?;
            let l = v.labels.clone();
            (Box::new(VitClassifier(v)), l)
        } else {
            let a = nets::TextArch::of(&arch).ok_or_else(|| anyhow!("{arch:?}: a text check model is a Bert, (XLM-)Roberta or DistilBert ...ForSequenceClassification"))?;
            let net = nets::TextNet::new(&cfg, a, w)?;
            let tok = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| anyhow!("tokenizer.json: {e}"))?;
            let l = net.labels.clone();
            (Box::new(TextClassifier { net, tok }), l)
        };
        let rule = Rule::from_sidecar(kind, &side, &labels)?;
        Ok(Check { kind, rule, model: Guarded::new(model, timeout) })
    })();
    made.map_err(bad)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use serde_json::json;

    /// A fake check model: scores by the input, an error, a panic, or a wait longer than any timeout.
    pub enum Fake {
        /// scores by the input: the image's red channel or the text decide which
        By(Box<dyn Fn(&Input) -> Scores + Send>),
        Fail,
        Panic,
        Hang,
    }

    impl Classifier for Fake {
        fn classify(&mut self, input: &Input) -> Result<Scores> {
            match self {
                Fake::By(f) => Ok(f(input)),
                Fake::Fail => bail!("the fake model failed"),
                Fake::Panic => panic!("the fake model panicked"),
                Fake::Hang => { std::thread::sleep(Duration::from_secs(30)); Ok(vec![]) }
            }
        }
    }

    pub fn rule(flag: &[&str]) -> Rule {
        Rule { flag_labels: flag.iter().map(|s| s.to_string()).collect(), threshold: 0.2, no_person_labels: vec![], age_margin: 3.0 }
    }

    pub fn fake(kind: CheckKind, f: Fake) -> Check {
        let r = match kind {
            CheckKind::Age => Rule { no_person_labels: vec!["none".into()], ..rule(&["0-17"]) },
            _ => rule(&["flag"]),
        };
        Check { kind, rule: r, model: Guarded::new(Box::new(f), Duration::from_millis(500)) }
    }

    fn sc(v: &[(&str, f32)]) -> Scores {
        v.iter().map(|(l, s)| (l.to_string(), *s)).collect()
    }

    #[test]
    fn scores_become_decisions_and_bad_scores_are_errors() {
        let r = rule(&["flag"]);
        assert_eq!(decide(CheckKind::ImageMinor, &r, &sc(&[("ok", 0.85), ("flag", 0.15)])).unwrap(), Decision::Pass);
        assert_eq!(decide(CheckKind::ImageMinor, &r, &sc(&[("ok", 0.75), ("flag", 0.25)])).unwrap(), Decision::Flag, "0.2 is the minor threshold");
        assert!(decide(CheckKind::ImageMinor, &r, &vec![]).is_err());
        assert!(decide(CheckKind::ImageMinor, &r, &sc(&[("ok", f32::NAN), ("flag", 0.0)])).is_err());
        assert!(decide(CheckKind::ImageMinor, &r, &sc(&[("ok", 1.0)])).is_err(), "no score for the flag label");
        // age in years: under 18 + 3 flags
        assert_eq!(decide(CheckKind::Age, &r, &sc(&[("age", 20.0)])).unwrap(), Decision::Flag);
        assert_eq!(decide(CheckKind::Age, &r, &sc(&[("age", 30.0)])).unwrap(), Decision::Pass);
        let ar = Rule { no_person_labels: vec!["none".into()], ..rule(&["0-17"]) };
        assert_eq!(decide(CheckKind::Age, &ar, &sc(&[("0-17", 0.05), ("18+", 0.15), ("none", 0.8)])).unwrap(), Decision::NoPerson);
        assert_eq!(decide(CheckKind::Age, &ar, &sc(&[("0-17", 0.3), ("18+", 0.0), ("none", 0.7)])).unwrap(), Decision::Flag, "a flag beats no person");
    }

    #[test]
    fn sidecars_are_checked_against_the_labels() {
        let labels: Vec<String> = ["0-2", "3-9", "10-19", "20-29", "30-39", "more than 70"].iter().map(|s| s.to_string()).collect();
        let r = Rule::from_sidecar(CheckKind::Age, &json!({"check": "age"}), &labels).unwrap();
        assert_eq!(r.flag_labels, ["0-2", "3-9", "10-19"]);
        assert!(Rule::from_sidecar(CheckKind::Age, &json!({"check": "image-minor"}), &labels).is_err(), "the wrong check");
        let l2: Vec<String> = ["safe", "unsafe"].iter().map(|s| s.to_string()).collect();
        assert!(Rule::from_sidecar(CheckKind::ImageMinor, &json!({"check": "image-minor", "flag_labels": ["minor"]}), &l2).is_err(), "no such label");
        assert!(Rule::from_sidecar(CheckKind::ImageMinor, &json!({"check": "image-minor"}), &l2).is_err(), "no flag labels");
        let r = Rule::from_sidecar(CheckKind::ImageMinor, &json!({"check": "image-minor", "flag_labels": ["UNSAFE"], "threshold": 0.99}), &l2).unwrap();
        assert_eq!(r.threshold, rules::MINOR_THRESHOLD_MAX, "a threshold is never raised past the fixed maximum");
        let l3: Vec<String> = ["child", "adult", "nobody"].iter().map(|s| s.to_string()).collect();
        assert!(Rule::from_sidecar(CheckKind::Age, &json!({"check": "age"}), &l3).is_err(), "labels that name no age must be listed");
        let r = Rule::from_sidecar(CheckKind::Age, &json!({"check": "age", "flag_labels": ["child"], "no_person_labels": ["nobody"]}), &l3).unwrap();
        assert_eq!(r.no_person_labels, ["nobody"]);
        let margin = Rule::from_sidecar(CheckKind::Age, &json!({"check": "age", "age_margin": 0}), &["age".to_string()]).unwrap();
        assert_eq!(margin.age_margin, rules::AGE_MARGIN_MIN, "the margin is never lowered");
    }

    #[test]
    fn errors_panics_and_timeouts_fail_closed() {
        let img = Input::Image(Arc::new(image::RgbImage::new(2, 2)));
        assert!(fake(CheckKind::ImageMinor, Fake::Fail).run(&img).is_err());
        assert!(fake(CheckKind::ImageMinor, Fake::Panic).run(&img).is_err());
        let mut h = fake(CheckKind::ImageMinor, Fake::Hang);
        let t0 = std::time::Instant::now();
        assert!(h.run(&img).is_err());
        assert!(t0.elapsed() < Duration::from_secs(5));
        assert!(h.model.broken());
        assert!(h.run(&img).is_err(), "a check that timed out is not asked again");
    }

    #[test]
    fn a_missing_or_onnx_only_model_is_reported_by_name() {
        let d = std::env::temp_dir().join(format!("pi-checks-{}", std::process::id()));
        std::fs::create_dir_all(d.join("age")).unwrap();
        let e = load(CheckKind::ImageMinor, &d, Duration::from_secs(1)).err().unwrap();
        assert_eq!(e.message(), "the image minor-safety check model is not installed; nothing is released");
        std::fs::write(d.join("age/config.json"), "{}").unwrap();
        std::fs::write(d.join("age/model.onnx"), "x").unwrap();
        assert!(load(CheckKind::Age, &d, Duration::from_secs(1)).err().unwrap().message().contains("ONNX"));
        std::fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn an_installed_tiny_vit_and_bert_load_and_answer() {
        let d = std::env::temp_dir().join(format!("pi-checks-real-{}", std::process::id()));
        let img_dir = d.join("image-minor");
        std::fs::create_dir_all(&img_dir).unwrap();
        std::fs::write(img_dir.join("config.json"), nets::tests::tiny_vit_config(&["ok", "flag"]).to_string()).unwrap();
        candle_core::safetensors::save(&nets::tests::tiny_vit(2), img_dir.join("model.safetensors")).unwrap();
        std::fs::write(img_dir.join("prastara_check.json"), r#"{"check": "image-minor", "flag_labels": ["flag"]}"#).unwrap();
        let mut c = load(CheckKind::ImageMinor, &d, Duration::from_secs(30)).map_err(|e| e.message()).unwrap();
        assert!(c.run(&Input::Image(Arc::new(image::RgbImage::from_pixel(16, 16, image::Rgb([10, 200, 10]))))).is_ok());
        assert!(c.run(&Input::Text("x".into())).is_err(), "the wrong input is an error");

        let txt_dir = d.join("text-minor");
        std::fs::create_dir_all(&txt_dir).unwrap();
        std::fs::write(txt_dir.join("config.json"), nets::tests::tiny_bert_config(&["ok", "flag"]).to_string()).unwrap();
        candle_core::safetensors::save(&nets::tests::tiny_bert(2), txt_dir.join("model.safetensors")).unwrap();
        std::fs::write(txt_dir.join("prastara_check.json"), r#"{"check": "text-minor", "flag_labels": ["flag"], "threshold": 0.1}"#).unwrap();
        let vocab = json!({"[PAD]": 0, "[CLS]": 1, "[SEP]": 2, "[UNK]": 3, "lorem": 4, "ipsum": 5, "dolor": 6, "sit": 7, "amet": 8});
        let tok = json!({"version": "1.0", "truncation": null, "padding": null, "added_tokens": [], "normalizer": {"type": "Lowercase"},
            "pre_tokenizer": {"type": "Whitespace"},
            "post_processor": {"type": "TemplateProcessing", "single": [{"SpecialToken": {"id": "[CLS]", "type_id": 0}}, {"Sequence": {"id": "A", "type_id": 0}}, {"SpecialToken": {"id": "[SEP]", "type_id": 0}}],
                               "pair": [{"Sequence": {"id": "A", "type_id": 0}}], "special_tokens": {"[CLS]": {"id": "[CLS]", "ids": [1], "tokens": ["[CLS]"]}, "[SEP]": {"id": "[SEP]", "ids": [2], "tokens": ["[SEP]"]}}},
            "decoder": null, "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "[UNK]"}});
        std::fs::write(txt_dir.join("tokenizer.json"), tok.to_string()).unwrap();
        let mut t = load(CheckKind::TextMinor, &d, Duration::from_secs(30)).map_err(|e| e.message()).unwrap();
        // longer than one window (16 positions): split into windows, each scored
        let lorem = "lorem ipsum dolor sit amet ".repeat(20);
        assert!(t.run(&Input::Text(lorem.into())).is_ok());
        std::fs::remove_dir_all(&d).ok();
    }
}
