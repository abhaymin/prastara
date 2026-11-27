//! Private datasets: Prastara builds the owner's adult datasets itself, from the owner's own local folders, for the
//! owner's private, local-only models (internal: docs/internal/PRIVATE-DATA.md).
//!
//! **Sources** are local folders only (no crawler, no downloader). Each one holds a `SOURCE.toml` the owner writes:
//! `owner_attests_adults_only = true`, `owner_attests_consent = true` and `origin = "..."`. A folder without it, or
//! with either attestation not `true`, is refused, and the refusal names the file to write. In a folder:
//! - images (png, jpg, webp) are described by a local vision model through the server's own chat API;
//! - videos (mp4 with H.264) give frames through the media engine's Rust decoder (`am2strata-media`'s
//!   `video_frames`), each frame handled as an image; without that engine, videos are skipped and counted;
//! - text (txt, md) is chunked and the model writes chat rows from it; a JSONL of chats is used as seed examples.
//!
//! **The model** must be a local one whose run config is marked private, uncensored or abliterated (the server's
//! `/api/models` says so). A `<provider>/<model>` name (anything with a `/`), a remote model or a mix, or an unmarked
//! model is refused, and the server's address must be this machine: adult data never leaves it. Each request carries
//! `"strata_local_only": true`, which keeps the server from handing it to a cluster worker or a remote overflow.
//!
//! **Gates that cannot be switched off**, each counted on the card (counts and hashes only, never what was dropped):
//! - every description, tag list, generation prompt and text row goes through the minor-safety rules
//!   ([`strata_safety::check`]); a match drops the item and puts its file's SHA-256 in `private/blocklist.sha256`;
//! - the model's `age_assessment` must be `adult`: `uncertain`, `minor` or a missing answer drops the item and blocks
//!   its hash;
//! - a description or prompt the model says names a real, identifiable person (`real_person_named`) is kept out of
//!   the generation-prompt data;
//! - a blocked file is skipped by every later run, in any dataset.
//!
//! **Output**: `<data>/private/<name>/` (mode 0700): `vision.jsonl` (`{image, description, tags}`), `prompts.jsonl`
//! (`{prompt, image}`), `text.jsonl` (chat rows), `frames/` (the kept video frames) and `dataset.card.md`. Every row
//! carries `"private": true`, and training or cleaning such a dataset marks its outputs private too
//! ([`is_private_dataset`]).

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context, Result};
use base64::Engine as _;
use serde_json::{json, Value};
use sha2::Digest;

use crate::dataset::{self, Record};
use crate::datagen;

/// The sidecar every source folder needs.
pub const SOURCE_FILE: &str = "SOURCE.toml";
/// The blocklist, in the private root: one SHA-256 (hex) per line, then the gate that blocked it.
pub const BLOCKLIST: &str = "blocklist.sha256";
pub const CARD: &str = "dataset.card.md";
/// The first line of a private dataset's card (how [`is_private_dataset`] knows a card).
pub const CARD_MARK: &str = "<!-- prastara: private -->";

const IMAGE_EXT: [&str; 4] = ["png", "jpg", "jpeg", "webp"];
const VIDEO_EXT: [&str; 5] = ["mp4", "m4v", "mov", "mkv", "webm"];
const TEXT_EXT: [&str; 3] = ["txt", "md", "markdown"];

// ------------------------------------------------------------------------------------------------- attestation

/// What the owner wrote in a source folder's `SOURCE.toml`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Source {
    pub dir: PathBuf,
    pub origin: String,
}

/// A small TOML reader for `key = value` lines (booleans and quoted strings); sections and comments are skipped.
fn toml_values(text: &str) -> BTreeMap<String, Value> {
    let mut out = BTreeMap::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with('[') {
            continue;
        }
        let Some((k, v)) = line.split_once('=') else { continue };
        let (k, v) = (k.trim().trim_matches('"').to_string(), v.trim());
        let val = if v == "true" || v.starts_with("true ") || v.starts_with("true#") {
            Value::Bool(true)
        } else if v == "false" || v.starts_with("false ") || v.starts_with("false#") {
            Value::Bool(false)
        } else if let Some(q) = v.chars().next().filter(|c| *c == '"' || *c == '\'') {
            let inner = &v[1..];
            match inner.find(q) {
                Some(end) => Value::String(inner[..end].replace("\\\"", "\"")),
                None => continue,
            }
        } else {
            continue;
        };
        out.insert(k, val);
    }
    out
}

/// A source folder's attestation, or the refusal (naming the file to write and what it must say).
pub fn read_source(dir: &Path) -> Result<Source> {
    if !dir.is_dir() {
        bail!("{}: a private-data source is a folder (with its own {SOURCE_FILE})", dir.display());
    }
    let file = dir.join(SOURCE_FILE);
    let need = format!("write {} with owner_attests_adults_only = true, owner_attests_consent = true (everyone shown \
                        consented, or the media is fully synthetic or drawn) and origin = \"where it came from\"", file.display());
    let text = std::fs::read_to_string(&file).map_err(|_| anyhow!("{}: refused, no {SOURCE_FILE}: {need}", dir.display()))?;
    let v = toml_values(&text);
    for k in ["owner_attests_adults_only", "owner_attests_consent"] {
        if v.get(k) != Some(&Value::Bool(true)) {
            bail!("{}: refused, {k} is not true: {need}", file.display());
        }
    }
    let origin = v.get("origin").and_then(Value::as_str).map(str::trim).filter(|o| !o.is_empty())
        .ok_or_else(|| anyhow!("{}: refused, no origin: {need}", file.display()))?;
    Ok(Source { dir: dir.to_path_buf(), origin: origin.to_string() })
}

// ------------------------------------------------------------------------------------------------- the model

/// A `<provider>/<model>` name is a remote model: refused, whatever it is.
pub fn refuse_remote_name(name: &str) -> Result<()> {
    if name.contains('/') {
        bail!("{name:?}: refused, a <provider>/<model> name is a remote model, and private data never leaves this machine; \
               use a local model whose run config is marked private, uncensored or abliterated");
    }
    Ok(())
}

/// The model to use from the server's `/api/models` list: `want` (exactly), else the loaded one. It must be a local
/// model marked private, uncensored or abliterated in its run config; `vision` also needs one that reads images.
pub fn pick_model(list: &Value, want: Option<&str>, vision: bool) -> Result<String> {
    if let Some(w) = want {
        refuse_remote_name(w)?;
    }
    let models = list["models"].as_array().cloned().unwrap_or_default();
    let m = match want {
        Some(w) => models.iter().find(|m| m["id"] == w).ok_or_else(|| anyhow!("the server has no local model {w:?}"))?,
        None => models.iter().find(|m| m["current"] == true)
            .ok_or_else(|| anyhow!("no model is loaded: name one with --model (a local private, uncensored or abliterated model)"))?,
    };
    let id = m["id"].as_str().unwrap_or("").to_string();
    refuse_remote_name(&id)?;
    if m["remote"] == true || m["mix"] == true || m["status"] == "remote" {
        bail!("{id:?}: refused, a remote model or a mix; private data never leaves this machine");
    }
    // the run config's own marks ("marks", from this server); an older server lists only "private"
    let marked = match m["marks"].as_array() {
        Some(a) => a.iter().any(|x| matches!(x.as_str(), Some("private" | "uncensored" | "abliterated"))),
        None => m["private"] == true,
    };
    if !marked {
        bail!("{id:?}: refused, its run config is not marked private, uncensored or abliterated (set \"private\": true, or install an abliterated model)");
    }
    if vision && m["multimodal"] != true {
        bail!("{id:?} does not read images: describing images needs a local vision model marked private, uncensored or abliterated");
    }
    Ok(id)
}

/// Whether `url` is this machine: localhost, a loopback address, or an address of one of its own interfaces.
pub fn is_local_url(url: &str) -> bool {
    let Some(rest) = url.strip_prefix("http://").or_else(|| url.strip_prefix("https://")) else { return false };
    let hostport = rest.split(['/', '?', '#']).next().unwrap_or("");
    let host = if let Some(h) = hostport.strip_prefix('[') {
        h.split(']').next().unwrap_or("")
    } else {
        hostport.rsplit_once(':').map_or(hostport, |(h, _)| h)
    };
    if host.eq_ignore_ascii_case("localhost") {
        return true;
    }
    match host.parse::<std::net::IpAddr>() {
        Ok(ip) if ip.is_loopback() => true,
        // an address this machine can bind to is one of its own
        Ok(ip) if !ip.is_unspecified() => std::net::UdpSocket::bind((ip, 0)).is_ok(),
        _ => false,
    }
}

/// The server's chat API on this machine (`AM2STRATA_URL`, `AM2STRATA_API_KEY`, as datagen reaches it).
pub struct Local {
    url: String,
    key: Option<String>,
    client: reqwest::blocking::Client,
    pub model: String,
}

impl Local {
    pub fn from_env() -> Result<Local> {
        let url = std::env::var("AM2STRATA_URL").unwrap_or_else(|_| "http://127.0.0.1:8000".into());
        Local::new(&url, std::env::var("AM2STRATA_API_KEY").ok().filter(|k| !k.is_empty()))
    }

    pub fn new(url: &str, key: Option<String>) -> Result<Local> {
        let url = url.trim_end_matches('/').to_string();
        if !is_local_url(&url) {
            bail!("{url}: refused, the server for private data must be this machine (localhost or one of its own addresses)");
        }
        let client = reqwest::blocking::Client::builder().timeout(std::time::Duration::from_secs(900)).build()?;
        Ok(Local { url, key, client, model: String::new() })
    }

    pub fn models(&self) -> Result<Value> {
        let mut rb = self.client.get(format!("{}/api/models", self.url));
        if let Some(k) = &self.key { rb = rb.bearer_auth(k); }
        let r = rb.send().with_context(|| format!("{} does not answer", self.url))?;
        anyhow::ensure!(r.status().is_success(), "{}/api/models: HTTP {}", self.url, r.status());
        Ok(r.json()?)
    }

    /// Choose the model ([`pick_model`]).
    pub fn choose(&mut self, want: Option<&str>, vision: bool) -> Result<()> {
        if let Some(w) = want {
            refuse_remote_name(w)?; // before anything is asked of the server
        }
        self.model = pick_model(&self.models()?, want, vision)?;
        Ok(())
    }

    /// One answer; `image` is (mime, bytes), sent as a data: URL in the message (never a path or a link).
    pub fn ask(&self, system: &str, user: &str, image: Option<(&str, &[u8])>, temperature: f64, max_tokens: u32) -> Result<String> {
        refuse_remote_name(&self.model)?;
        let content = match image {
            Some((mime, bytes)) => json!([
                {"type": "image_url", "image_url": {"url": format!("data:{mime};base64,{}", base64::engine::general_purpose::STANDARD.encode(bytes))}},
                {"type": "text", "text": user}]),
            None => json!(user),
        };
        let body = json!({"model": self.model, "temperature": temperature, "max_tokens": max_tokens,
                          "strata_local_only": true, "chat_template_kwargs": {"enable_thinking": false},
                          "messages": [{"role": "system", "content": system}, {"role": "user", "content": content}]});
        let mut rb = self.client.post(format!("{}/v1/chat/completions", self.url)).json(&body);
        if let Some(k) = &self.key { rb = rb.bearer_auth(k); }
        let r = rb.send().with_context(|| format!("{} does not answer", self.url))?;
        let ok = r.status().is_success();
        let v: Value = r.json().unwrap_or_default();
        if !ok {
            bail!("the model's answer failed: {}", v["error"]["message"].as_str().unwrap_or("unknown error"));
        }
        Ok(v["choices"][0]["message"]["content"].as_str().unwrap_or("").to_string())
    }
}

// ------------------------------------------------------------------------------------------------- the prompts

const SYSTEM: &str = "You label media for a private, local training dataset that only its owner uses. You answer with \
JSON only: no prose before or after it, no Markdown fences.";

/// The describing request (vision): the description, tags, the age assessment and whether a real person is named.
pub const DESCRIBE: &str = "Describe this image fully, literally and objectively, including any nudity or sexual \
content in plain, explicit terms: the people, their bodies, poses, clothing or its absence, actions, setting, lighting, \
camera angle and style. Do not name or guess the identity of any real person.\n\nAlso judge the age of every person \
shown: \"adult\" only when every person is clearly an adult; \"uncertain\" when anyone's age is unclear; \"minor\" when \
anyone may be under 18. Say whether your description names a real, identifiable person (a celebrity, a public figure \
or a named private person).\n\nAnswer as JSON: {\"description\": \"...\", \"tags\": [\"...\"], \"age_assessment\": \
\"adult\" | \"uncertain\" | \"minor\", \"real_person_named\": true | false}";

/// The generation-prompt request, written from a description.
pub fn prompt_request(description: &str) -> String {
    format!("Write one prompt for a text-to-image model that would produce an image like the one described below: \
             subject, action, setting, composition, lighting and style, as a single paragraph. Every person in it is \
             an adult. Never name a real person.\n\nAnswer as JSON: {{\"prompt\": \"...\", \"real_person_named\": true | \
             false}} (real_person_named: whether your prompt names or depicts a real, identifiable person).\n\n\
             DESCRIPTION:\n{description}")
}

/// What the model said about one image.
#[derive(Debug, Clone, PartialEq)]
pub struct Interp {
    pub description: String,
    pub tags: Vec<String>,
    /// adult, uncertain, minor, or what else it said ("missing" when it said nothing)
    pub age: String,
    pub real_person: bool,
}

/// The describing reply, or None when it holds no usable JSON with a description.
pub fn parse_interp(reply: &str) -> Option<Interp> {
    let v = datagen::extract_json(reply)?;
    let description = v["description"].as_str()?.trim().to_string();
    if description.is_empty() {
        return None;
    }
    let tags = match &v["tags"] {
        Value::Array(a) => a.iter().filter_map(|t| t.as_str()).map(|t| t.trim().to_string()).filter(|t| !t.is_empty()).collect(),
        Value::String(s) => s.split(',').map(|t| t.trim().to_string()).filter(|t| !t.is_empty()).collect(),
        _ => vec![],
    };
    let age = v["age_assessment"].as_str().map(|a| a.trim().to_lowercase()).filter(|a| !a.is_empty()).unwrap_or_else(|| "missing".into());
    // anything but a plain false counts as a real person named (an unclear answer keeps the row out)
    let real_person = v["real_person_named"] != Value::Bool(false);
    Some(Interp { description, tags, age, real_person })
}

// ------------------------------------------------------------------------------------------------- the files

pub fn sha256_hex(bytes: &[u8]) -> String {
    sha2::Sha256::digest(bytes).iter().map(|b| format!("{b:02x}")).collect()
}

/// The blocklist: hashes no run reads again.
#[derive(Debug, Default)]
pub struct Blocklist {
    path: PathBuf,
    hashes: BTreeSet<String>,
}

impl Blocklist {
    pub fn open(root: &Path) -> Result<Blocklist> {
        let path = root.join(BLOCKLIST);
        let hashes = std::fs::read_to_string(&path).unwrap_or_default().lines()
            .filter_map(|l| l.split_whitespace().next()).filter(|h| h.len() == 64).map(str::to_lowercase).collect();
        Ok(Blocklist { path, hashes })
    }

    pub fn contains(&self, hash: &str) -> bool {
        self.hashes.contains(hash)
    }

    pub fn len(&self) -> usize {
        self.hashes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.hashes.is_empty()
    }

    /// Add a hash (with the gate's name, never the content); written at once, so a crash later keeps it.
    pub fn block(&mut self, hash: &str, gate: &str) -> Result<bool> {
        if !self.hashes.insert(hash.to_string()) {
            return Ok(false);
        }
        use std::io::Write as _;
        let mut f = open_private_append(&self.path)?;
        writeln!(f, "{hash} {gate}")?;
        Ok(true)
    }
}

fn open_private_append(path: &Path) -> Result<std::fs::File> {
    let mut o = std::fs::OpenOptions::new();
    o.create(true).append(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt as _;
        o.mode(0o600);
    }
    o.open(path).with_context(|| format!("{}", path.display()))
}

/// A folder only its owner may enter (0700), made if it is missing and tightened if it is not.
pub fn private_dir(p: &Path) -> Result<()> {
    std::fs::create_dir_all(p).with_context(|| format!("{}", p.display()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;
        std::fs::set_permissions(p, std::fs::Permissions::from_mode(0o700)).with_context(|| format!("{}", p.display()))?;
    }
    Ok(())
}

/// A file only its owner may read (0600).
fn write_private(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut o = std::fs::OpenOptions::new();
    o.create(true).write(true).truncate(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt as _;
        o.mode(0o600);
    }
    use std::io::Write as _;
    o.open(path).and_then(|mut f| f.write_all(bytes)).with_context(|| format!("writing {}", path.display()))
}

/// A dataset name: letters, digits, `.`, `_` and `-` (a folder under the private root, nothing else).
pub fn check_name(name: &str) -> Result<()> {
    let ok = !name.is_empty() && name.len() <= 64 && !name.starts_with('.')
        && name.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'));
    if !ok {
        bail!("{name:?}: a private dataset's name is 1-64 letters, digits, '.', '_' or '-' (not starting with '.')");
    }
    Ok(())
}

/// The private root of a data folder: `<data>/private`.
pub fn private_root(data_dir: &Path) -> PathBuf {
    data_dir.join("private")
}

/// Whether a dataset is private: a row carries `"private": true`, or its card is a private one. Training on it or
/// cleaning it makes private outputs.
pub fn is_private_dataset(path: &Path) -> bool {
    let card_private = |c: PathBuf| std::fs::read_to_string(c).is_ok_and(|t| t.starts_with(CARD_MARK));
    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("dataset");
    if card_private(path.with_file_name(format!("{stem}.card.md"))) || card_private(path.with_file_name(CARD)) {
        return true;
    }
    let Ok(f) = std::fs::File::open(path) else { return false };
    use std::io::BufRead as _;
    std::io::BufReader::new(f).lines().map_while(|l| l.ok())
        .any(|l| l.contains("\"private\"") && serde_json::from_str::<Value>(&l).is_ok_and(|v| v["private"] == true))
}

/// A JSONL row marked private.
pub fn mark_row(line: &str) -> String {
    match serde_json::from_str::<Value>(line) {
        Ok(Value::Object(mut o)) => {
            o.insert("private".into(), Value::Bool(true));
            Value::Object(o).to_string()
        }
        _ => line.to_string(),
    }
}

/// A model or adapter folder's config (`config.json` or `adapter_config.json`) marked `"prastara_private": true`:
/// the GGUF it converts to then carries `prastara.private`, and its install is marked private.
pub fn mark_config_private(path: &Path) -> Result<()> {
    let mut v: Value = serde_json::from_str(&std::fs::read_to_string(path).with_context(|| format!("{}", path.display()))?)?;
    v["prastara_private"] = json!(true);
    std::fs::write(path, serde_json::to_string_pretty(&v)?)?;
    Ok(())
}

// ------------------------------------------------------------------------------------------------- the media

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    Image,
    Video,
    Text,
    Seeds,
}

fn kind_of(p: &Path) -> Option<Kind> {
    let e = p.extension().and_then(|e| e.to_str())?.to_lowercase();
    let e = e.as_str();
    if IMAGE_EXT.contains(&e) {
        Some(Kind::Image)
    } else if VIDEO_EXT.contains(&e) {
        Some(Kind::Video)
    } else if TEXT_EXT.contains(&e) {
        Some(Kind::Text)
    } else if e == "jsonl" {
        Some(Kind::Seeds)
    } else {
        None
    }
}

fn mime_of(p: &Path) -> &'static str {
    match p.extension().and_then(|e| e.to_str()).unwrap_or("").to_lowercase().as_str() {
        "jpg" | "jpeg" => "image/jpeg",
        "webp" => "image/webp",
        _ => "image/png",
    }
}

/// The files of a source folder, three levels deep, hidden ones and the sidecar left out.
fn files_of(dir: &Path) -> Vec<(PathBuf, Kind)> {
    fn walk(p: &Path, depth: usize, out: &mut Vec<(PathBuf, Kind)>) {
        let mut entries: Vec<PathBuf> = std::fs::read_dir(p).into_iter().flatten().flatten().map(|e| e.path())
            .filter(|q| !q.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.starts_with('.') || n == SOURCE_FILE)).collect();
        entries.sort();
        for e in entries {
            // a symlink is not followed: the folder the owner attested is the folder read
            let Ok(meta) = std::fs::symlink_metadata(&e) else { continue };
            if meta.is_file() {
                if let Some(k) = kind_of(&e) {
                    out.push((e, k));
                }
            } else if meta.is_dir() && depth > 0 {
                walk(&e, depth - 1, out);
            }
        }
    }
    let mut out = vec![];
    walk(dir, 3, &mut out);
    out
}

/// The media engine (`am2strata-media`) for video frames: `AM2STRATA_MEDIA_EXE`, else beside this command.
pub fn find_media_exe() -> Option<PathBuf> {
    if let Some(p) = std::env::var_os("AM2STRATA_MEDIA_EXE").map(PathBuf::from) {
        return p.is_file().then_some(p);
    }
    let me = std::env::current_exe().ok()?;
    let real = std::fs::canonicalize(&me).unwrap_or(me);
    let p = real.with_file_name(format!("am2strata-media{}", std::env::consts::EXE_SUFFIX));
    p.is_file().then_some(p)
}

/// A video's frames (time in seconds, PNG bytes) through the media engine's `video_frames` (MP4/H.264 in Rust, no
/// ffmpeg). The engine runs on the CPU only for this (no GPU is visible to it).
pub fn video_frames(exe: &Path, video: &Path, every_s: f64, max: usize, width: usize) -> Result<Vec<(f64, Vec<u8>)>> {
    use std::io::{BufRead as _, Write as _};
    let mut child = std::process::Command::new(exe)
        .env("CUDA_VISIBLE_DEVICES", "").env("HIP_VISIBLE_DEVICES", "")
        .stdin(std::process::Stdio::piped()).stdout(std::process::Stdio::piped()).stderr(std::process::Stdio::null())
        .spawn().with_context(|| format!("starting {}", exe.display()))?;
    let result = (|| -> Result<Vec<(f64, Vec<u8>)>> {
        let mut stdin = child.stdin.take().ok_or_else(|| anyhow!("the media engine has no stdin"))?;
        let stdout = child.stdout.take().ok_or_else(|| anyhow!("the media engine has no stdout"))?;
        let cmd = json!({"op": "video_frames", "file": video.to_string_lossy(), "every_s": every_s, "max": max, "width": width, "fit": true});
        writeln!(stdin, "{cmd}")?;
        stdin.flush()?;
        for line in std::io::BufReader::new(stdout).lines() {
            let v: Value = match serde_json::from_str(&line?) { Ok(v) => v, Err(_) => continue };
            match v["event"].as_str() {
                Some("frames") => {
                    let _ = writeln!(stdin, "{}", json!({"op": "quit"}));
                    let mut out = vec![];
                    for f in v["frames"].as_array().into_iter().flatten() {
                        let png = base64::engine::general_purpose::STANDARD.decode(f["png"].as_str().unwrap_or(""))
                            .map_err(|e| anyhow!("a frame: {e}"))?;
                        out.push((f["t"].as_f64().unwrap_or(0.0), png));
                    }
                    return Ok(out);
                }
                Some("error") => bail!("{}", v["message"].as_str().unwrap_or("the media engine failed")),
                _ => {}
            }
        }
        bail!("the media engine ended without frames")
    })();
    let _ = child.kill();
    let _ = child.wait();
    result
}

// ------------------------------------------------------------------------------------------------- the run

#[derive(Debug, Clone)]
pub struct Settings {
    pub name: String,
    /// the owner's source folders (each with its SOURCE.toml)
    pub sources: Vec<PathBuf>,
    /// the private root (`<data>/private`): the dataset goes in `<root>/<name>`, the blocklist in the root
    pub root: PathBuf,
    pub temperature: f64,
    /// text: how many chat rows to aim for
    pub text_count: usize,
    pub per_chunk: usize,
    pub chunk_chars: usize,
    /// video: one frame every this many seconds, at most `max_frames` per video
    pub every_s: f64,
    pub max_frames: usize,
    /// the media engine for video frames (None: videos are skipped and counted)
    pub media_exe: Option<PathBuf>,
}

impl Settings {
    pub fn new(name: &str, sources: Vec<PathBuf>, root: PathBuf) -> Settings {
        Settings { name: name.into(), sources, root, temperature: 0.4, text_count: 100, per_chunk: 3, chunk_chars: 3000,
                   every_s: 5.0, max_frames: 8, media_exe: None }
    }
    pub fn out_dir(&self) -> PathBuf {
        self.root.join(&self.name)
    }
}

/// The gates' names, as the card and the blocklist say them.
pub mod gate {
    pub const MINOR_SAFETY: &str = "minor_safety";
    pub const AGE_UNCERTAIN: &str = "age_uncertain";
    pub const AGE_MINOR: &str = "age_minor";
    pub const AGE_NOT_ADULT: &str = "age_not_adult";
    pub const REAL_PERSON: &str = "real_person";
    pub const BLOCKED: &str = "blocked_before";
}

/// What a run did: counts and hashes only.
#[derive(Debug, Default, Clone)]
pub struct Report {
    pub sources: Vec<Source>,
    pub images: usize,
    pub videos: usize,
    pub frames: usize,
    pub texts: usize,
    /// videos not read: no frame decoder, or one it cannot read
    pub videos_unread: usize,
    pub duplicates: usize,
    pub asked: usize,
    pub unparsable: usize,
    pub failed: usize,
    pub vision_rows: usize,
    pub prompt_rows: usize,
    pub text_rows: usize,
    /// per gate: items dropped (prompt rows only for real_person)
    pub dropped: BTreeMap<&'static str, usize>,
    /// hashes this run added to the blocklist
    pub blocked: Vec<String>,
    pub notes: Vec<String>,
    pub stopped: bool,
}

impl Report {
    fn drop(&mut self, gate: &'static str) {
        *self.dropped.entry(gate).or_default() += 1;
    }
}

/// The gate an interpretation fails, if any (age first, then the minor-safety rules over everything it said).
pub fn gate_of(i: &Interp) -> Option<&'static str> {
    match i.age.as_str() {
        "adult" => {}
        "uncertain" => return Some(gate::AGE_UNCERTAIN),
        "minor" => return Some(gate::AGE_MINOR),
        _ => return Some(gate::AGE_NOT_ADULT),
    }
    let all = format!("{}\n{}", i.description, i.tags.join(", "));
    strata_safety::check(&all).map(|_| gate::MINOR_SAFETY)
}

fn record_text(r: &Record) -> String {
    r.messages.iter().map(|(_, c)| c.as_str()).collect::<Vec<_>>().join("\n")
}

/// One image (or frame) through the model and the gates: kept (with its generation prompt, unless that was kept out),
/// dropped by a gate (the caller blocks its hash), or unusable (no answer, or no usable JSON: neither kept nor blocked).
enum Outcome {
    Kept { interp: Interp, prompt: Option<String> },
    Dropped(&'static str),
    Unusable,
}

fn interpret(model: &Local, s: &Settings, rep: &mut Report, bytes: &[u8], mime: &str) -> Outcome {
    rep.asked += 1;
    let reply = match model.ask(SYSTEM, DESCRIBE, Some((mime, bytes)), s.temperature, 2048)
        .or_else(|_| model.ask(SYSTEM, DESCRIBE, Some((mime, bytes)), s.temperature, 2048)) {
        Ok(r) => r,
        Err(_) => { rep.failed += 1; return Outcome::Unusable; }
    };
    let Some(interp) = parse_interp(&reply) else { rep.unparsable += 1; return Outcome::Unusable };
    if let Some(g) = gate_of(&interp) {
        return Outcome::Dropped(g);
    }
    if interp.real_person {
        rep.drop(gate::REAL_PERSON);
        return Outcome::Kept { interp, prompt: None };
    }
    rep.asked += 1;
    let prompt = match model.ask(SYSTEM, &prompt_request(&interp.description), None, s.temperature, 1024) {
        Ok(r) => r,
        Err(_) => { rep.failed += 1; return Outcome::Kept { interp, prompt: None }; }
    };
    let Some(v) = datagen::extract_json(&prompt) else { rep.unparsable += 1; return Outcome::Kept { interp, prompt: None } };
    let Some(p) = v["prompt"].as_str().map(str::trim).filter(|p| !p.is_empty()) else {
        rep.unparsable += 1;
        return Outcome::Kept { interp, prompt: None };
    };
    // a prompt that trips the rules drops the whole item (its description is suspect too)
    if strata_safety::check(p).is_some() {
        return Outcome::Dropped(gate::MINOR_SAFETY);
    }
    if v["real_person_named"] != Value::Bool(false) {
        rep.drop(gate::REAL_PERSON);
        return Outcome::Kept { interp, prompt: None };
    }
    Outcome::Kept { interp, prompt: Some(p.to_string()) }
}

/// Rows of one item, kept until the item has passed every gate.
struct Rows {
    vision: Vec<String>,
    prompts: Vec<String>,
    text: Vec<String>,
}

fn vision_row(image: &str, i: &Interp, meta: Value) -> String {
    json!({"image": image, "description": i.description, "tags": i.tags, "private": true, "meta": meta}).to_string()
}

/// Build the private dataset. `progress(done, total, message)`; `should_stop()` ends it early (what was made is kept).
pub fn run(s: &Settings, model: &Local, mut progress: impl FnMut(usize, usize, &str), should_stop: impl Fn() -> bool) -> Result<Report> {
    check_name(&s.name)?;
    refuse_remote_name(&model.model)?;
    anyhow::ensure!(!s.sources.is_empty(), "no source folders");
    let mut rep = Report::default();
    // every folder's attestation first: one refusal stops the run before anything is read
    for d in &s.sources {
        rep.sources.push(read_source(d)?);
    }
    let out = s.out_dir();
    if out.join(CARD).exists() {
        bail!("{}: a private dataset of this name exists; pick another name", out.display());
    }
    private_dir(&s.root)?;
    private_dir(&out)?;
    let mut block = Blocklist::open(&s.root)?;
    let mut files = vec![];
    for src in &rep.sources {
        files.extend(files_of(&src.dir));
    }
    let total = files.len();
    let mut rows = Rows { vision: vec![], prompts: vec![], text: vec![] };
    let mut seen = BTreeSet::new();
    let mut text_left = s.text_count;
    for (n, (path, kind)) in files.iter().enumerate() {
        if should_stop() {
            rep.stopped = true;
            break;
        }
        let name = path.display().to_string();
        progress(n, total, &format!("{} of {total} files", n + 1));
        let bytes = match std::fs::read(path) {
            Ok(b) => b,
            Err(e) => { rep.notes.push(format!("a file could not be read: {e}")); continue; }
        };
        let hash = sha256_hex(&bytes);
        if block.contains(&hash) {
            rep.drop(gate::BLOCKED);
            continue;
        }
        if !seen.insert(hash.clone()) {
            rep.duplicates += 1;
            continue;
        }
        match kind {
            Kind::Image => {
                rep.images += 1;
                match interpret(model, s, &mut rep, &bytes, mime_of(path)) {
                    Outcome::Kept { interp, prompt } => {
                        rows.vision.push(vision_row(&name, &interp, json!({"model": model.model, "sha256": hash})));
                        if let Some(p) = prompt {
                            rows.prompts.push(json!({"prompt": p, "image": name, "private": true}).to_string());
                        }
                    }
                    Outcome::Dropped(g) => {
                        rep.drop(g);
                        if block.block(&hash, g)? { rep.blocked.push(hash); }
                    }
                    Outcome::Unusable => {}
                }
            }
            Kind::Video => {
                rep.videos += 1;
                let Some(exe) = &s.media_exe else {
                    rep.videos_unread += 1;
                    continue;
                };
                let frames = match video_frames(exe, path, s.every_s, s.max_frames.max(1), 768) {
                    Ok(f) => f,
                    Err(e) => {
                        rep.videos_unread += 1;
                        rep.notes.push(format!("a video was not read: {e:#}"));
                        continue;
                    }
                };
                // a video is kept only when every frame passes: one frame that fails drops the whole video
                let mut kept: Vec<(f64, String, Vec<u8>, Interp, Option<String>)> = vec![];
                let mut failed_gate = None;
                for (t, png) in frames {
                    rep.frames += 1;
                    let fh = sha256_hex(&png);
                    if block.contains(&fh) {
                        failed_gate = Some((gate::BLOCKED, fh));
                        break;
                    }
                    match interpret(model, s, &mut rep, &png, "image/png") {
                        Outcome::Kept { interp, prompt } => kept.push((t, fh, png, interp, prompt)),
                        Outcome::Dropped(g) => { failed_gate = Some((g, fh)); break; }
                        Outcome::Unusable => {}
                    }
                }
                if let Some((g, fh)) = failed_gate {
                    rep.drop(g);
                    if g != gate::BLOCKED && block.block(&fh, g)? { rep.blocked.push(fh); }
                    if block.block(&hash, g)? { rep.blocked.push(hash); }
                    continue;
                }
                let fdir = out.join("frames");
                if !kept.is_empty() {
                    private_dir(&fdir)?;
                }
                for (t, fh, png, interp, prompt) in kept {
                    let fp = fdir.join(format!("{fh}.png"));
                    write_private(&fp, &png)?;
                    let image = fp.display().to_string();
                    rows.vision.push(vision_row(&image, &interp, json!({"model": model.model, "sha256": fh, "video": name, "video_sha256": hash, "t": t})));
                    if let Some(p) = prompt {
                        rows.prompts.push(json!({"prompt": p, "image": image, "private": true}).to_string());
                    }
                }
            }
            Kind::Text | Kind::Seeds => {
                rep.texts += 1;
                if text_left == 0 {
                    continue;
                }
                match text_rows(model, s, &mut rep, path, *kind, &bytes, text_left)? {
                    Ok(made) => {
                        text_left = text_left.saturating_sub(made.len());
                        rows.text.extend(made);
                    }
                    Err(g) => {
                        rep.drop(g);
                        if block.block(&hash, g)? { rep.blocked.push(hash); }
                    }
                }
            }
        }
    }
    progress(total, total, "writing the dataset");
    rep.vision_rows = rows.vision.len();
    rep.prompt_rows = rows.prompts.len();
    rep.text_rows = rows.text.len();
    for (file, lines) in [("vision.jsonl", &rows.vision), ("prompts.jsonl", &rows.prompts), ("text.jsonl", &rows.text)] {
        let p = out.join(file);
        if lines.is_empty() {
            std::fs::remove_file(&p).ok();
            continue;
        }
        write_private(&p, lines.iter().map(|l| format!("{l}\n")).collect::<String>().as_bytes())?;
    }
    if rep.videos > 0 && s.media_exe.is_none() {
        rep.notes.push("videos were skipped: no media engine (am2strata-media) to decode their frames; set AM2STRATA_MEDIA_EXE".into());
    }
    write_private(&out.join(CARD), card(s, &model.model, &rep).as_bytes())?;
    Ok(rep)
}

/// One text file's chat rows (Ok), or the gate that drops and blocks the whole file (Err).
fn text_rows(model: &Local, s: &Settings, rep: &mut Report, path: &Path, kind: Kind, bytes: &[u8], want: usize)
             -> Result<std::result::Result<Vec<String>, &'static str>> {
    let text = String::from_utf8_lossy(bytes).into_owned();
    // the source itself is read by the rules before the model sees it
    let ds = datagen::Settings {
        recipe: if kind == Kind::Seeds { datagen::Recipe::Instruct } else { datagen::Recipe::Chat },
        sources: vec![], seeds: None, count: want, per_chunk: s.per_chunk.clamp(1, 20), chunk_chars: s.chunk_chars.clamp(500, 20_000),
        overlap: 0, temperature: s.temperature, min_score: 0, topic: None, language: None, out: PathBuf::new(), seed: 0,
        check_code: false, think: false,
    };
    let mut requests: Vec<String> = vec![];
    if kind == Kind::Seeds {
        let (seeds, _) = dataset::read_jsonl(path)?;
        if seeds.is_empty() {
            return Ok(Ok(vec![]));
        }
        if seeds.iter().any(|r| strata_safety::check(&record_text(r)).is_some()) {
            return Ok(Err(gate::MINOR_SAFETY));
        }
        for _ in 0..want.div_ceil(ds.per_chunk) {
            requests.push(datagen::prompt(&ds, None, &seeds));
        }
    } else {
        let chunks = dataset::chunk(&text, ds.chunk_chars, 0);
        if chunks.iter().any(|c| strata_safety::check(&c.text).is_some()) {
            return Ok(Err(gate::MINOR_SAFETY));
        }
        for c in chunks.into_iter().filter(|c| c.text.trim().chars().count() >= 200).take(want.div_ceil(ds.per_chunk)) {
            requests.push(datagen::prompt(&ds, Some(&c.text), &[]));
        }
    }
    let mut out = vec![];
    for req in requests {
        rep.asked += 1;
        let reply = match model.ask(SYSTEM, &req, None, s.temperature, 6144) {
            Ok(r) => r,
            Err(_) => { rep.failed += 1; continue; }
        };
        let Some(v) = datagen::extract_json(&reply) else { rep.unparsable += 1; continue };
        let recs = datagen::records_from(ds.recipe, &v);
        if recs.is_empty() {
            rep.unparsable += 1;
        }
        for r in recs {
            if strata_safety::check(&record_text(&r)).is_some() {
                // one row that trips the rules drops what this file made and blocks the file
                return Ok(Err(gate::MINOR_SAFETY));
            }
            if dataset::filter(&r, &dataset::Rules::default()).is_some() {
                continue;
            }
            out.push(json!({"messages": r.messages.iter().map(|(role, c)| json!({"role": role, "content": c})).collect::<Vec<_>>(),
                            "meta": {"recipe": "private-text", "model": model.model, "source": path.display().to_string()},
                            "private": true}).to_string());
        }
        if out.len() >= want {
            out.truncate(want);
            break;
        }
    }
    Ok(Ok(out))
}

/// The card: private, the sources and their attestations, and counts per gate. Never any dropped content.
pub fn card(s: &Settings, model: &str, r: &Report) -> String {
    let gates = [(gate::MINOR_SAFETY, "matched the minor-safety rules (dropped, hash blocked)"),
                 (gate::AGE_UNCERTAIN, "age assessed as uncertain (dropped, hash blocked)"),
                 (gate::AGE_MINOR, "age assessed as minor (dropped, hash blocked)"),
                 (gate::AGE_NOT_ADULT, "no age assessment, or not \"adult\" (dropped, hash blocked)"),
                 (gate::REAL_PERSON, "named a real person (kept out of the generation prompts)"),
                 (gate::BLOCKED, "on the blocklist from an earlier run (skipped)")];
    let mut o = format!("{CARD_MARK}\n# Private dataset card: {}\n\n**Private.** Built by Prastara's Workshop on this PC from the \
                         owner's own folders, with the local model **{model}**. It stays on this PC: never upload, share or \
                         publish it, or a model trained on it. Every row carries `\"private\": true`, and models and datasets \
                         made from it are marked private too.\n\n", s.name);
    o.push_str("## Sources\n\n| Folder | Origin (the owner's words) | Adults only | Consent |\n|---|---|---|---|\n");
    for src in &r.sources {
        o.push_str(&format!("| `{}` | {} | attested | attested |\n", src.dir.display(), src.origin.replace('|', "/").replace('\n', " ")));
    }
    o.push_str(&format!("\n## Rows\n\n| | |\n|---|---:|\n| Images read | {} |\n| Videos read | {} ({} frames; {} not read) |\n\
                         | Text files read | {} |\n| Duplicates skipped | {} |\n| Requests to the model | {} ({} with no usable JSON, {} failed) |\n\
                         | Vision rows (`vision.jsonl`) | {} |\n| Generation prompts (`prompts.jsonl`) | {} |\n| Text rows (`text.jsonl`) | {} |\n",
                        r.images, r.videos, r.frames, r.videos_unread, r.texts, r.duplicates, r.asked, r.unparsable, r.failed,
                        r.vision_rows, r.prompt_rows, r.text_rows));
    o.push_str("\n## Safety gates (fixed; they cannot be switched off)\n\n| Gate | Dropped |\n|---|---:|\n");
    for (g, what) in gates {
        o.push_str(&format!("| {what} | {} |\n", r.dropped.get(g).copied().unwrap_or(0)));
    }
    o.push_str(&format!("\n## Blocked by this run ({}; SHA-256, in `{}`)\n\n", r.blocked.len(), s.root.join(BLOCKLIST).display()));
    if r.blocked.is_empty() {
        o.push_str("none\n");
    }
    for h in &r.blocked {
        o.push_str(&format!("- `{h}`\n"));
    }
    if r.stopped {
        o.push_str("\nThe run was stopped early; the rows so far were kept.\n");
    }
    if !r.notes.is_empty() {
        o.push_str("\n## Notes\n\n");
        for n in &r.notes {
            o.push_str(&format!("- {n}\n"));
        }
    }
    o
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_toml_is_read_and_refused_when_incomplete() {
        let d = std::env::temp_dir().join(format!("pd-src-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        let e = read_source(&d).unwrap_err().to_string();
        assert!(e.contains("SOURCE.toml") && e.contains(&d.join(SOURCE_FILE).display().to_string()), "{e}");
        std::fs::write(d.join(SOURCE_FILE), "owner_attests_adults_only = true\nowner_attests_consent = false\norigin = \"x\"\n").unwrap();
        assert!(read_source(&d).unwrap_err().to_string().contains("owner_attests_consent"));
        std::fs::write(d.join(SOURCE_FILE), "# mine\nowner_attests_adults_only = true\nowner_attests_consent = true # yes\n").unwrap();
        assert!(read_source(&d).unwrap_err().to_string().contains("origin"));
        std::fs::write(d.join(SOURCE_FILE), "owner_attests_adults_only = true\nowner_attests_consent = true\norigin = 'my own camera'\n").unwrap();
        assert_eq!(read_source(&d).unwrap().origin, "my own camera");
        std::fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn the_model_must_be_local_and_marked() {
        let list = json!({"models": [
            {"id": "mula-abliterated", "marks": ["private", "abliterated"], "multimodal": true, "current": true},
            {"id": "tiel", "marks": [], "multimodal": true, "private": false},
            {"id": "oldserver", "private": true, "multimodal": false},
            {"id": "mix", "mix": true, "status": "remote"},
            {"id": "openai/gpt-5.5", "remote": true, "status": "remote"}]});
        assert_eq!(pick_model(&list, None, true).unwrap(), "mula-abliterated");
        assert!(pick_model(&list, Some("openai/gpt-5.5"), false).unwrap_err().to_string().contains("remote"));
        assert!(pick_model(&list, Some("tiel"), false).unwrap_err().to_string().contains("not marked"));
        assert!(pick_model(&list, Some("mix"), false).is_err());
        assert_eq!(pick_model(&list, Some("oldserver"), false).unwrap(), "oldserver");
        assert!(pick_model(&list, Some("oldserver"), true).unwrap_err().to_string().contains("images"));
        assert!(pick_model(&list, Some("mula"), false).is_err(), "a part of a name is not enough");
    }

    #[test]
    fn only_this_machine_is_a_server_for_private_data() {
        assert!(is_local_url("http://127.0.0.1:8000"));
        assert!(is_local_url("http://localhost:8000/"));
        assert!(is_local_url("https://[::1]:8001"));
        assert!(!is_local_url("https://api.example.com/v1"));
        assert!(!is_local_url("http://203.0.113.9:8000"));
        assert!(!is_local_url("http://0.0.0.0:8000"));
        assert!(Local::new("https://api.example.com", None).is_err());
    }

    #[test]
    fn interpretations_are_parsed_and_gated() {
        let i = parse_interp(r#"{"description": "a red chair by a window", "tags": ["chair", "red"], "age_assessment": "adult", "real_person_named": false}"#).unwrap();
        assert_eq!(gate_of(&i), None);
        assert!(!i.real_person);
        let u = parse_interp(r#"{"description": "a chair", "tags": "a, b", "age_assessment": "Uncertain"}"#).unwrap();
        assert_eq!((gate_of(&u), u.tags.len(), u.real_person), (Some(gate::AGE_UNCERTAIN), 2, true), "no plain false counts as a real person");
        let m = parse_interp(r#"{"description": "a chair"}"#).unwrap();
        assert_eq!(gate_of(&m), Some(gate::AGE_NOT_ADULT));
        // a plainly non-sexual pair of trigger words, as the rules' own tests use
        let t = parse_interp(r#"{"description": "a juvenile court file", "tags": ["explicit"], "age_assessment": "adult", "real_person_named": false}"#).unwrap();
        assert_eq!(gate_of(&t), Some(gate::MINOR_SAFETY));
        assert!(parse_interp("no json").is_none());
    }

    #[test]
    fn names_rows_and_configs() {
        assert!(check_name("my-set_1.v2").is_ok());
        for bad in ["", "../x", "a/b", ".hidden", "with space"] {
            assert!(check_name(bad).is_err(), "{bad}");
        }
        assert_eq!(serde_json::from_str::<Value>(&mark_row(r#"{"messages": []}"#)).unwrap()["private"], true);
        assert_eq!(mark_row("not json"), "not json");
    }
}
