//! Private-data import: the owner's adult dataset brought in from a local folder (a read-only NAS mount too), a
//! direct https URL, or a Hugging Face dataset, into a locked quarantine, and released into a private dataset only
//! after every item passed the fixed checks (internal: docs/internal/PRIVATE-DATA.md).
//!
//! 1. **Attestation.** Each source needs a `SOURCE.toml` (the private-data format: `owner_attests_adults_only`,
//!    `owner_attests_consent`, `origin`; for a URL or Hub source also `licence` and `url` / `hf` naming it). Without
//!    it nothing is read or fetched.
//! 2. **Quarantine.** Everything lands in `<private>/quarantine/<import-id>/` (0700), each item stored by its SHA-256
//!    (no source names kept). Nothing in quarantine can be read by clean, train, datagen, merge, export or regress
//!    ([`in_quarantine`], checked by the server's path validation and by this command's own arguments). Archives are
//!    unpacked with zip-slip and bomb guards ([`archive`]).
//! 3. **Checks** ([`checks`]), all of them before anything is released: the SHA-256 blocklist; for an image the minor-
//!    safety check, the age estimate and the real-person check; for a text the fixed word rules
//!    ([`strata_safety::check`]), the minor-safety check and the real-person check. A check model that is not
//!    installed stops the import in quarantine ("the <x> check model is not installed; nothing is released"). Fail
//!    closed: an error, a timeout, an unreadable or undecodable file is a rejection.
//! 4. **Minors.** An item flagged as involving a minor (or already on the blocklist) is deleted from quarantine at once
//!    (unlinked), its hash blocked, and the event logged with the hash and the reason only. Hard-coded. When the
//!    flagged share of the source passes the limit (1 %, `strata_safety::import::MAX_MINOR_SHARE`; a setting can only
//!    lower it), the whole import is rejected and its quarantine deleted.
//! 5. **Release.** The items that passed are copied to `<private>/datasets/<name>/` with the private mark and a card;
//!    then the quarantine is deleted.
//! 6. **Audit.** `<private>/audit.jsonl` (0600, append only): every fetch, walk, rejection, deletion and release, with
//!    hashes, counts, reasons and times; never content, names or thumbnails.

pub mod archive;
pub mod checks;
pub mod fetch;
pub mod nets;

use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{anyhow, bail, Context, Result};
use serde_json::{json, Value};

use crate::private_data::{self as pd, Blocklist};
use archive::{Caps, Totals};
use checks::{Check, CheckKind, Decision, Input, Unavailable};
pub use fetch::{Net, Probe, Spec};

pub const QUARANTINE: &str = "quarantine";
pub const DATASETS: &str = "datasets";
pub const AUDIT: &str = "audit.jsonl";
pub const CHECK_MODELS: &str = "check-models";
const STATE: &str = "import.json";

/// Whether `p` is inside a private quarantine (`.../private/quarantine/...`), by its components (resolved when it
/// exists, so a link into the quarantine counts too).
pub fn in_quarantine(p: &Path) -> bool {
    let check = |q: &Path| {
        let parts: Vec<String> = q.components().filter_map(|c| match c {
            std::path::Component::Normal(n) => Some(n.to_string_lossy().to_lowercase()),
            _ => None,
        }).collect();
        parts.windows(2).any(|w| w[0] == "private" && w[1] == QUARANTINE)
    };
    check(p) || p.canonicalize().is_ok_and(|c| check(&c))
}

/// The refusal for a path in quarantine (None: it is not).
pub fn refuse_quarantine(p: &Path) -> Option<String> {
    in_quarantine(p).then(|| format!("{}: refused, it is in the private quarantine; nothing in quarantine is used until it is checked and released", p.display()))
}

/// The first argument of a command line that is a path in quarantine (every Workshop command but the import itself
/// refuses it): plain paths and `--flag=path` alike.
pub fn quarantine_args<S: AsRef<str>>(args: &[S]) -> Option<String> {
    args.iter().map(|a| a.as_ref()).map(|a| a.split_once('=').filter(|_| a.starts_with("--")).map_or(a, |(_, v)| v))
        .filter(|a| a.contains('/') || a.contains('\\')).find_map(|a| refuse_quarantine(Path::new(a)))
}

/// An import id: `<name>-<unix seconds>` (letters, digits, `.`, `_`, `-`).
pub fn check_id(id: &str) -> Result<()> {
    pd::check_name(id).map_err(|_| anyhow!("{id:?}: an import id is 1-64 letters, digits, '.', '_' or '-'"))
}

pub(crate) fn create_private(p: &Path) -> Result<std::fs::File> {
    let mut o = std::fs::OpenOptions::new();
    o.create(true).write(true).truncate(true);
    #[cfg(unix)]
    { use std::os::unix::fs::OpenOptionsExt as _; o.mode(0o600); }
    o.open(p).with_context(|| format!("{}", p.display()))
}

fn now() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs())
}

// ------------------------------------------------------------------------------------------------- attestation

/// What the owner attested of a source.
#[derive(Debug, Clone, PartialEq)]
pub struct Attestation {
    pub file: PathBuf,
    pub origin: String,
    pub licence: Option<String>,
}

/// The source's `SOURCE.toml`: `file`, else (a folder) `<folder>/SOURCE.toml`. A file outside the folder must name
/// the source (`folder`, `url` or `hf`), so one attestation cannot be used for another source; a URL or Hub source
/// also needs `licence`.
pub fn read_attestation(spec: &Spec, file: Option<&Path>) -> Result<Attestation> {
    let file = match (file, spec) {
        (Some(f), _) => f.to_path_buf(),
        (None, Spec::Folder(d)) => d.join(pd::SOURCE_FILE),
        (None, _) => bail!("a URL or Hugging Face source needs its {}: write one and give its path (attestation)", pd::SOURCE_FILE),
    };
    let need = format!("write {} with owner_attests_adults_only = true, owner_attests_consent = true, origin = \"...\"{}",
                       file.display(), if spec.is_network() { ", licence = \"...\" and the source it is for (url = \"...\" or hf = \"owner/name\")" } else { "" });
    let text = std::fs::read_to_string(&file).map_err(|_| anyhow!("refused, no {}: {need}", file.display()))?;
    let v = pd::toml_values(&text);
    for k in ["owner_attests_adults_only", "owner_attests_consent"] {
        if v.get(k) != Some(&Value::Bool(true)) {
            bail!("{}: refused, {k} is not true: {need}", file.display());
        }
    }
    let s = |k: &str| v.get(k).and_then(Value::as_str).map(str::trim).filter(|x| !x.is_empty()).map(str::to_string);
    let origin = s("origin").ok_or_else(|| anyhow!("{}: refused, no origin: {need}", file.display()))?;
    let licence = s("licence").or_else(|| s("license"));
    let in_folder = matches!(spec, Spec::Folder(d) if file.parent() == Some(d.as_path()));
    if !in_folder {
        let names = match spec {
            Spec::Folder(d) => s("folder").is_some_and(|f| Path::new(&f) == d.as_path()
                || Path::new(&f).canonicalize().ok().zip(d.canonicalize().ok()).is_some_and(|(a, b)| a == b)),
            Spec::Url(u) => s("url").as_deref() == Some(u.as_str()),
            Spec::Hf { id, .. } => s("hf").as_deref() == Some(id.as_str()),
        };
        if !names {
            let key = match spec { Spec::Folder(_) => "folder", Spec::Url(_) => "url", Spec::Hf { .. } => "hf" };
            bail!("{}: refused, it does not name this source ({key} = \"{}\"): {need}", file.display(), match spec {
                Spec::Folder(d) => d.display().to_string(), Spec::Url(u) => u.clone(), Spec::Hf { id, .. } => id.clone() });
        }
    }
    if spec.is_network() && licence.is_none() {
        bail!("{}: refused, no licence: {need}", file.display());
    }
    Ok(Attestation { file, origin, licence })
}

// ------------------------------------------------------------------------------------------------- the audit log

/// `<private>/audit.jsonl`: one JSON line per event, appended (0600). Never content, names or thumbnails.
pub struct Audit {
    path: PathBuf,
}

impl Audit {
    pub fn open(root: &Path) -> Result<Audit> {
        pd::private_dir(root)?;
        let path = root.join(AUDIT);
        #[cfg(unix)]
        if path.exists() {
            use std::os::unix::fs::PermissionsExt as _;
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600))?;
        }
        Ok(Audit { path })
    }

    pub fn log(&self, import: &str, event: &str, mut extra: Value) -> Result<()> {
        let mut o = std::fs::OpenOptions::new();
        o.create(true).append(true);
        #[cfg(unix)]
        { use std::os::unix::fs::OpenOptionsExt as _; o.mode(0o600); }
        let mut f = o.open(&self.path).with_context(|| format!("{}", self.path.display()))?;
        if !extra.is_object() {
            extra = json!({});
        }
        extra["t"] = json!(now());
        extra["import"] = json!(import);
        extra["event"] = json!(event);
        writeln!(f, "{extra}")?;
        Ok(())
    }
}

// ------------------------------------------------------------------------------------------------- the state

/// A source as JSON (`{"folder"}`, `{"url"}`, `{"hf", "revision", "pattern"}`), as the state keeps it.
pub fn spec_value(s: &Spec) -> Value {
    match s {
        Spec::Folder(p) => json!({"folder": p.display().to_string()}),
        Spec::Url(u) => json!({"url": u}),
        Spec::Hf { id, revision, pattern } => json!({"hf": id, "revision": revision, "pattern": pattern}),
    }
}

pub fn spec_from(v: &Value) -> Result<Spec> {
    let s = |k: &str| v[k].as_str().filter(|x| !x.is_empty()).map(str::to_string);
    match (s("folder"), s("url"), s("hf")) {
        (Some(f), None, None) => Ok(Spec::Folder(PathBuf::from(f))),
        (None, Some(u), None) => Ok(Spec::Url(u)),
        (None, None, Some(h)) => Ok(Spec::Hf { id: h, revision: s("revision").unwrap_or_else(|| "main".into()), pattern: s("pattern") }),
        _ => bail!("not a source: {v}"),
    }
}

/// Counts of one import (what the progress events and the card say).
#[derive(Debug, Default, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Counts {
    /// files fetched or copied from the source (archives counted once)
    pub fetched: usize,
    /// bytes of them
    pub bytes: u64,
    /// unique items in quarantine at the start of the checks (deleted ones included)
    pub items: usize,
    pub checked: usize,
    /// items not released, for any reason (deleted ones included)
    pub rejected: usize,
    /// items deleted at once: flagged as involving a minor, or on the blocklist
    pub deleted: usize,
    /// of those, flagged as involving a minor (or on the blocklist): what the source's share is counted on
    pub minor: usize,
    pub released: usize,
    pub duplicates: usize,
    /// links, devices, hidden files and files that disappeared during the walk: not read
    pub skipped: usize,
    /// archives rejected whole by a guard
    pub archives_rejected: usize,
    /// rejections per reason
    pub reasons: BTreeMap<String, usize>,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Item {
    pub sha256: String,
    pub ext: String,
    pub bytes: u64,
    /// pending, passed, rejected, deleted
    pub status: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct State {
    pub id: String,
    pub name: String,
    pub source: String,
    pub origin: String,
    #[serde(default)]
    pub licence: Option<String>,
    #[serde(default)]
    pub commit: Option<String>,
    pub created: u64,
    /// fetching, checking, stopped, released, rejected
    pub status: String,
    #[serde(default)]
    pub reason: Option<String>,
    pub counts: Counts,
    pub items: Vec<Item>,
    #[serde(default)]
    pub fetched_done: bool,
    /// network files already fetched and taken in (a resumed fetch skips them)
    #[serde(default)]
    pub done_files: Vec<String>,
    /// the source and the SOURCE.toml it started with (a resume uses them)
    #[serde(default)]
    pub spec: Option<Value>,
    #[serde(default)]
    pub attestation: Option<String>,
    /// the size the owner confirmed (a resumed fetch holds to it)
    #[serde(default)]
    pub confirm_bytes: Option<u64>,
}

impl State {
    fn save(&self, dir: &Path) -> Result<()> {
        let tmp = dir.join(format!("{STATE}.tmp"));
        create_private(&tmp)?.write_all(serde_json::to_string_pretty(self)?.as_bytes())?;
        std::fs::rename(&tmp, dir.join(STATE))?;
        Ok(())
    }
}

/// The imports in quarantine (for the server's list): their state, without the item list.
pub fn list(root: &Path) -> Vec<Value> {
    let mut v: Vec<Value> = std::fs::read_dir(root.join(QUARANTINE)).into_iter().flatten().flatten().filter_map(|e| {
        let s: State = serde_json::from_str(&std::fs::read_to_string(e.path().join(STATE)).ok()?).ok()?;
        Some(json!({"id": s.id, "name": s.name, "source": s.source, "status": s.status, "reason": s.reason, "counts": s.counts, "created": s.created}))
    }).collect();
    v.sort_by_key(|x| std::cmp::Reverse(x["created"].as_u64().unwrap_or(0)));
    v
}

/// Delete an import's quarantine (the owner's discard), logged.
pub fn discard(root: &Path, id: &str) -> Result<()> {
    check_id(id)?;
    let dir = root.join(QUARANTINE).join(id);
    if !dir.join(STATE).is_file() {
        bail!("no import {id} in quarantine");
    }
    let items = std::fs::read_dir(dir.join("items")).map(|r| r.count()).unwrap_or(0);
    std::fs::remove_dir_all(&dir).with_context(|| format!("{}", dir.display()))?;
    Audit::open(root)?.log(id, "discard", json!({"items_deleted": items}))
}

// ------------------------------------------------------------------------------------------------- the run

#[derive(Debug, Clone)]
pub struct Settings {
    /// the private root (`<data>/private`)
    pub root: PathBuf,
    pub id: String,
    /// the dataset's name: `<root>/datasets/<name>`
    pub name: String,
    pub spec: Spec,
    /// the SOURCE.toml (None: the folder's own)
    pub attestation: Option<PathBuf>,
    /// the size the owner confirmed (needed for a network source; it must still be the size)
    pub confirm_bytes: Option<u64>,
    /// the flagged share that rejects the source (None: the fixed 1 %; only a lower one is taken)
    pub minor_share: Option<f64>,
    pub caps: Caps,
    /// one check of one item at most this long
    pub timeout: Duration,
}

impl Settings {
    pub fn new(root: PathBuf, name: &str, spec: Spec) -> Settings {
        let id = format!("{name}-{}", now());
        Settings { root, id, name: name.into(), spec, attestation: None, confirm_bytes: None, minor_share: None, caps: Caps::default(),
                   timeout: Duration::from_secs(120) }
    }
    pub fn quarantine(&self) -> PathBuf {
        self.root.join(QUARANTINE).join(&self.id)
    }
    pub fn dataset(&self) -> PathBuf {
        self.root.join(DATASETS).join(&self.name)
    }
}

/// How an import ended.
#[derive(Debug, Clone, PartialEq)]
pub enum Outcome {
    Released { dir: PathBuf, counts: Counts },
    /// the whole source rejected (its quarantine deleted)
    Rejected { reason: String, counts: Counts },
    /// stopped in quarantine (nothing released): a check model missing or broken, or the owner stopped it
    Stopped { reason: String, counts: Counts },
}

/// What a file in quarantine is to the checks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ItemKind {
    Image,
    Text,
    Other,
}

pub const TEXT_EXT: [&str; 8] = ["txt", "md", "markdown", "jsonl", "json", "csv", "tsv", "text"];

/// By the first bytes for images (never the name alone), by the extension for text.
pub fn item_kind(path: &Path, ext: &str) -> ItemKind {
    let mut head = [0u8; 16];
    let n = std::fs::File::open(path).and_then(|mut f| f.read(&mut head)).unwrap_or(0);
    let h = &head[..n];
    if h.starts_with(b"\x89PNG\r\n\x1a\n") || h.starts_with(&[0xff, 0xd8, 0xff]) || (h.len() >= 12 && &h[..4] == b"RIFF" && &h[8..12] == b"WEBP") {
        return ItemKind::Image;
    }
    if TEXT_EXT.contains(&ext) {
        return ItemKind::Text;
    }
    ItemKind::Other
}

/// The checks an item of `kind` needs.
pub fn checks_for(kind: ItemKind) -> &'static [CheckKind] {
    match kind {
        ItemKind::Image => &[CheckKind::ImageMinor, CheckKind::Age, CheckKind::ImageRealPerson],
        ItemKind::Text => &[CheckKind::TextMinor, CheckKind::TextRealPerson],
        ItemKind::Other => &[],
    }
}

/// An image decoded under limits (a huge image is an error, not a memory spike).
fn decode_image(bytes: &[u8]) -> Result<image::RgbImage> {
    let mut r = image::ImageReader::new(std::io::Cursor::new(bytes)).with_guessed_format()?;
    let mut l = image::Limits::default();
    l.max_image_width = Some(16_384);
    l.max_image_height = Some(16_384);
    l.max_alloc = Some(1 << 30);
    r.limits(l);
    Ok(r.decode()?.to_rgb8())
}

/// The reasons, as the card and the audit log say them.
pub mod reason {
    pub const BLOCKLISTED: &str = "blocklisted";
    pub const MINOR_RULES: &str = "minor_text_rules";
    pub const UNREADABLE: &str = "unreadable";
    pub const UNDECODABLE: &str = "undecodable";
    pub const UNSUPPORTED: &str = "unsupported_type";
    pub const CHECK_FAILED: &str = "check_failed";
}

/// Rejected for a check: the check's id (`image-minor`, `age`, ...).
fn flag_reason(k: CheckKind) -> String {
    format!("flagged_{}", k.id().replace('-', "_"))
}

struct Run<'a> {
    s: &'a Settings,
    spec: Spec,
    dir: PathBuf,
    audit: Audit,
    state: State,
    progress: &'a mut dyn FnMut(&Counts, &str),
}

impl Run<'_> {
    fn note(&mut self, msg: &str) {
        (self.progress)(&self.state.counts, msg);
    }

    fn reject(&mut self, i: usize, why: &str) -> Result<()> {
        let it = &mut self.state.items[i];
        it.status = "rejected".into();
        it.reason = Some(why.into());
        self.state.counts.rejected += 1;
        *self.state.counts.reasons.entry(why.into()).or_default() += 1;
        let (h, id) = (it.sha256.clone(), self.state.id.clone());
        self.audit.log(&id, "reject", json!({"sha256": h, "reason": why}))
    }

    /// Deleted at once (a minor, or on the blocklist): unlinked, its hash blocked, logged with the hash and reason.
    fn delete_minor(&mut self, i: usize, why: &str, block: &mut Blocklist) -> Result<()> {
        let (sha, ext) = (self.state.items[i].sha256.clone(), self.state.items[i].ext.clone());
        let p = self.dir.join("items").join(format!("{sha}.{ext}"));
        match std::fs::remove_file(&p) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(e).with_context(|| format!("deleting a flagged item {sha}")),
        }
        block.block(&sha, &format!("import_{why}"))?;
        let it = &mut self.state.items[i];
        it.status = "deleted".into();
        it.reason = Some(why.into());
        let c = &mut self.state.counts;
        c.rejected += 1;
        c.deleted += 1;
        c.minor += 1;
        *c.reasons.entry(why.into()).or_default() += 1;
        let id = self.state.id.clone();
        self.audit.log(&id, "delete", json!({"sha256": sha, "reason": why}))?;
        self.state.save(&self.dir)
    }

    /// One file into the quarantine's items: an archive unpacked (then removed), anything else stored by its hash.
    fn admit(&mut self, path: &Path, ext: &str, totals: &mut Totals) -> Result<()> {
        let items = self.dir.join("items");
        if let Some(kind) = archive::archive_kind(path) {
            let hash = fetch::hash_file(path)?;
            match archive::unpack(path, kind, &items, &self.s.caps, totals)? {
                Ok(u) => {
                    let id = self.state.id.clone();
                    self.audit.log(&id, "unpack", json!({"archive_sha256": hash, "items": u.items.len(), "skipped_entries": u.skipped}))?;
                    self.state.counts.skipped += u.skipped;
                    for (h, e, n) in u.items {
                        self.add_item(h, e, n);
                    }
                }
                Err(r) => {
                    self.state.counts.archives_rejected += 1;
                    let id = self.state.id.clone();
                    self.audit.log(&id, "archive_rejected", json!({"archive_sha256": hash, "reason": r.0}))?;
                    self.note(&format!("an archive was rejected whole: {}", r.0));
                }
            }
            std::fs::remove_file(path).ok();
            return Ok(());
        }
        let mut f = std::fs::File::open(path)?;
        match archive::store(&mut f, &items, ext, 0, &self.s.caps, totals)? {
            Ok((h, n)) => self.add_item(h, ext.into(), n),
            Err(r) => self.note(&format!("a file was not taken: {}", r.0)),
        }
        drop(f);
        std::fs::remove_file(path).ok();
        Ok(())
    }

    fn add_item(&mut self, sha256: String, ext: String, bytes: u64) {
        if self.state.items.iter().any(|i| i.sha256 == sha256 && i.ext == ext) {
            self.state.counts.duplicates += 1;
            return;
        }
        self.state.items.push(Item { sha256, ext, bytes, status: "pending".into(), reason: None });
    }

    /// The owner's folder, read only: links not followed, hidden files and the sidecar left out, a file that
    /// disappears skipped. Nothing is ever written in it.
    fn walk(&mut self, src: &Path, stop: &dyn Fn() -> bool) -> Result<()> {
        let mut files = vec![];
        let mut stack = vec![(src.to_path_buf(), 0usize)];
        while let Some((d, depth)) = stack.pop() {
            let rd = match std::fs::read_dir(&d) {
                Ok(r) => r,
                Err(_) => { self.state.counts.skipped += 1; continue; }
            };
            let mut entries: Vec<PathBuf> = rd.flatten().map(|e| e.path()).collect();
            entries.sort();
            for e in entries {
                let name = e.file_name().and_then(|n| n.to_str()).unwrap_or("");
                if name.starts_with('.') || (depth == 0 && name == pd::SOURCE_FILE) {
                    self.state.counts.skipped += 1;
                    continue;
                }
                match std::fs::symlink_metadata(&e) {
                    Ok(m) if m.is_file() => files.push(e),
                    Ok(m) if m.is_dir() && depth < 16 => stack.push((e, depth + 1)),
                    // links, devices, sockets, too deep, or gone meanwhile
                    _ => self.state.counts.skipped += 1,
                }
            }
        }
        let id = self.state.id.clone();
        self.audit.log(&id, "walk", json!({"source": self.state.source, "files": files.len(), "skipped": self.state.counts.skipped}))?;
        self.copy_listed(&files, stop)
    }

    /// Copy listed files of the source into quarantine (read only; O_NOFOLLOW, so a file swapped for a link is not read).
    fn copy_listed(&mut self, files: &[PathBuf], stop: &dyn Fn() -> bool) -> Result<()> {
        let mut totals = Totals { bytes: self.state.counts.bytes, files: self.state.items.len() };
        let tmp_dir = self.dir.join("incoming");
        for (n, f) in files.iter().enumerate() {
            if stop() {
                bail!("stopped by the owner while copying; nothing is released");
            }
            let mut o = std::fs::OpenOptions::new();
            o.read(true);
            #[cfg(unix)]
            { use std::os::unix::fs::OpenOptionsExt as _; o.custom_flags(libc::O_NOFOLLOW); }
            let mut src = match o.open(f) {
                Ok(x) => x,
                Err(_) => { self.state.counts.skipped += 1; continue; } // gone, a link now, or unreadable
            };
            if !src.metadata().is_ok_and(|m| m.is_file()) {
                self.state.counts.skipped += 1;
                continue;
            }
            let ext = archive::ext_of(&f.to_string_lossy());
            // archives are kept whole first (a named temporary in incoming/), then unpacked
            let tmp = tmp_dir.join(format!("copy-{n}.{ext}"));
            let mut out = create_private(&tmp)?;
            let copied = std::io::copy(&mut (&mut src).take(self.s.caps.max_file_bytes + 1), &mut out);
            drop(out);
            let k = match copied {
                Ok(k) => k,
                Err(_) => { std::fs::remove_file(&tmp).ok(); self.state.counts.skipped += 1; continue; }
            };
            if k > self.s.caps.max_file_bytes {
                std::fs::remove_file(&tmp).ok();
                self.note("a file larger than the cap was not taken");
                self.state.counts.skipped += 1;
                continue;
            }
            self.state.counts.fetched += 1;
            self.state.counts.bytes += k;
            let name = if tmp.extension().is_some_and(|e| e == "gz") && f.to_string_lossy().to_lowercase().ends_with(".tar.gz") {
                let t = tmp_dir.join(format!("copy-{n}.tar.gz"));
                std::fs::rename(&tmp, &t)?;
                t
            } else { tmp };
            self.admit(&name, &ext, &mut totals)?;
            if n % 64 == 0 {
                let msg = format!("{} of {} files copied", n + 1, files.len());
                self.note(&msg);
            }
        }
        Ok(())
    }

    fn fetch_network(&mut self, net: &Net, stop: &dyn Fn() -> bool) -> Result<()> {
        let probe = net.probe(&self.spec)?;
        match self.s.confirm_bytes.or(self.state.confirm_bytes) {
            Some(c) if c == probe.bytes => {}
            Some(c) => bail!("the source is now {} bytes, not the {c} you confirmed; nothing is fetched", probe.bytes),
            None => bail!("a network source is fetched only after you confirm its size ({} bytes)", probe.bytes),
        }
        if probe.files.len() > self.s.caps.max_files || probe.bytes > self.s.caps.max_total_bytes {
            bail!("the source is larger than the import's caps ({} files, {} bytes)", self.s.caps.max_files, self.s.caps.max_total_bytes);
        }
        self.state.commit = probe.commit.clone();
        let id = self.state.id.clone();
        let incoming = self.dir.join("incoming");
        let mut totals = Totals { bytes: 0, files: self.state.items.len() };
        for f in &probe.files {
            if self.state.done_files.contains(&f.name) {
                continue;
            }
            let (path, hash) = net.download(f, &incoming, |_| {}, stop)?;
            self.state.counts.fetched += 1;
            self.state.counts.bytes += f.bytes;
            self.audit.log(&id, "fetch", json!({"source": self.state.source, "commit": probe.commit, "sha256": hash, "bytes": f.bytes,
                                                "sha256_checked": f.sha256.is_some()}))?;
            let ext = archive::ext_of(&f.name);
            self.admit(&path, &ext, &mut totals)?;
            self.state.done_files.push(f.name.clone());
            let msg = format!("{} of {} files fetched", self.state.counts.fetched, probe.files.len());
            self.note(&msg);
            self.state.save(&self.dir)?;
        }
        Ok(())
    }

    /// One item through the checks. Err: a check broke (timed out) and the import must stop.
    fn check_item(&mut self, i: usize, checks: &mut BTreeMap<CheckKind, Check>, block: &mut Blocklist) -> Result<std::result::Result<(), String>> {
        let (sha, ext) = (self.state.items[i].sha256.clone(), self.state.items[i].ext.clone());
        if block.contains(&sha) {
            self.delete_minor(i, reason::BLOCKLISTED, block)?;
            return Ok(Ok(()));
        }
        let path = self.dir.join("items").join(format!("{sha}.{ext}"));
        let kind = item_kind(&path, &ext);
        let bytes = match std::fs::read(&path) {
            Ok(b) if pd::sha256_hex(&b) == sha => b,
            _ => { self.reject(i, reason::UNREADABLE)?; return Ok(Ok(())); }
        };
        let input = match kind {
            ItemKind::Other => { self.reject(i, reason::UNSUPPORTED)?; return Ok(Ok(())); }
            ItemKind::Image => match decode_image(&bytes) {
                Ok(img) => Input::Image(Arc::new(img)),
                Err(_) => { self.reject(i, reason::UNDECODABLE)?; return Ok(Ok(())); }
            },
            ItemKind::Text => match String::from_utf8(bytes) {
                Ok(t) => {
                    if strata_safety::check(&t).is_some() {
                        self.delete_minor(i, reason::MINOR_RULES, block)?;
                        return Ok(Ok(()));
                    }
                    Input::Text(t.into())
                }
                Err(_) => { self.reject(i, reason::UNDECODABLE)?; return Ok(Ok(())); }
            },
        };
        // the minor checks first: a flag deletes the item at once
        let mut order: Vec<CheckKind> = checks_for(kind).to_vec();
        order.sort_by_key(|k| !k.is_minor());
        for k in order {
            let c = checks.get_mut(&k).ok_or_else(|| anyhow!("the {} check is not loaded", k.label()))?;
            match c.run(&input) {
                Ok(Decision::Pass) | Ok(Decision::NoPerson) => {}
                Ok(Decision::Flag) if k.is_minor() => {
                    self.delete_minor(i, &flag_reason(k), block)?;
                    return Ok(Ok(()));
                }
                Ok(Decision::Flag) => { self.reject(i, &flag_reason(k))?; return Ok(Ok(())); }
                Err(_) => {
                    self.reject(i, reason::CHECK_FAILED)?;
                    if c.model.broken() {
                        return Ok(Err(format!("the {} check model did not answer in time; nothing is released", k.label())));
                    }
                    return Ok(Ok(()));
                }
            }
        }
        self.state.items[i].status = "passed".into();
        Ok(Ok(()))
    }

    fn stopped(&mut self, reason: String) -> Result<Outcome> {
        self.state.status = "stopped".into();
        self.state.reason = Some(reason.clone());
        self.state.save(&self.dir)?;
        let id = self.state.id.clone();
        self.audit.log(&id, "stopped", json!({"reason": reason, "counts": self.state.counts}))?;
        Ok(Outcome::Stopped { reason, counts: self.state.counts.clone() })
    }
}

/// Run an import: attest, fetch or walk into quarantine (unless resumed past that), check every item, then reject
/// the whole source or release what passed. `loader` gives each check (the installed models, or fakes in tests).
pub fn run(s: &Settings, net: Option<&Net>, loader: &mut dyn FnMut(CheckKind) -> std::result::Result<Check, Unavailable>,
           progress: &mut dyn FnMut(&Counts, &str), stop: &dyn Fn() -> bool) -> Result<Outcome> {
    pd::check_name(&s.name)?;
    check_id(&s.id)?;
    let dir = s.quarantine();
    // a resumed import keeps the source and the attestation it started with (read again: it must still hold)
    let stored: Option<State> = std::fs::read_to_string(dir.join(STATE)).ok().map(|t| serde_json::from_str(&t)).transpose()?;
    let (spec, att_file) = match stored.as_ref().and_then(|st| st.spec.clone().map(|v| (v, st.attestation.clone()))) {
        Some((v, a)) => (spec_from(&v)?, a.map(PathBuf::from)),
        None => (s.spec.clone(), s.attestation.clone()),
    };
    let att = read_attestation(&spec, att_file.as_deref())?;
    if s.dataset().join(pd::CARD).exists() {
        bail!("{}: a private dataset of this name exists; pick another name", s.dataset().display());
    }
    if let Spec::Folder(d) = &spec {
        let (dc, rc) = (d.canonicalize().with_context(|| format!("{}", d.display()))?, s.root.canonicalize().unwrap_or(s.root.clone()));
        if dc.starts_with(&rc) || rc.starts_with(&dc) {
            bail!("{}: the source may not be the private folder or contain it", d.display());
        }
    }
    pd::private_dir(&s.root)?;
    pd::private_dir(&s.root.join(QUARANTINE))?;
    let audit = Audit::open(&s.root)?;
    let resumed = stored.is_some();
    let state = if let Some(st) = stored {
        if st.name != s.name {
            bail!("import {} is for the dataset {:?}, not {:?}", s.id, st.name, s.name);
        }
        st
    } else {
        State { id: s.id.clone(), name: s.name.clone(), source: spec.describe(), origin: att.origin.clone(), licence: att.licence.clone(),
                commit: None, created: now(), status: "fetching".into(), reason: None, counts: Counts::default(), items: vec![], fetched_done: false,
                done_files: vec![], spec: Some(spec_value(&spec)), attestation: att_file.as_ref().map(|p| p.display().to_string()), confirm_bytes: s.confirm_bytes }
    };
    for d in [dir.clone(), dir.join("items"), dir.join("incoming")] {
        pd::private_dir(&d)?;
    }
    let mut r = Run { s, spec: spec.clone(), dir: dir.clone(), audit, state, progress };
    r.audit.log(&s.id, if resumed { "resume" } else { "start" }, json!({"source": r.state.source, "name": s.name,
        "attestation": {"adults_only": true, "consent": true, "licence": att.licence}}))?;
    r.state.save(&dir)?;

    // 1. into quarantine
    if !r.state.fetched_done {
        let fetched = match &spec {
            Spec::Folder(src) => r.walk(src, stop),
            _ => match net {
                Some(n) => r.fetch_network(n, stop),
                None => Err(anyhow!("no network client for a network source")),
            },
        };
        if let Err(e) = fetched {
            return r.stopped(format!("{e:#}"));
        }
        r.state.fetched_done = true;
        r.state.status = "checking".into();
        r.state.counts.items = r.state.items.len();
        std::fs::remove_dir_all(dir.join("incoming")).ok();
        r.state.save(&dir)?;
        let c = r.state.counts.clone();
        r.audit.log(&s.id, "quarantined", json!({"items": c.items, "fetched": c.fetched, "bytes": c.bytes, "duplicates": c.duplicates,
                                                 "skipped": c.skipped, "archives_rejected": c.archives_rejected}))?;
    }

    // 2. the check models every kind of item here needs: one missing stops the import
    let mut need: Vec<CheckKind> = vec![];
    for it in r.state.items.iter().filter(|i| i.status == "pending") {
        let k = item_kind(&dir.join("items").join(format!("{}.{}", it.sha256, it.ext)), &it.ext);
        for c in checks_for(k) {
            if !need.contains(c) {
                need.push(*c);
            }
        }
    }
    need.sort();
    let mut checks = BTreeMap::new();
    let mut missing = vec![];
    for k in need {
        match loader(k) {
            Ok(c) => { checks.insert(k, c); }
            Err(e) => missing.push(e.message()),
        }
    }
    if !missing.is_empty() {
        return r.stopped(missing.join("; "));
    }

    // 3. every item through its checks
    let mut block = Blocklist::open(&s.root)?;
    let share = strata_safety::import::minor_share(s.minor_share);
    let total = r.state.counts.items;
    let pending: Vec<usize> = (0..r.state.items.len()).filter(|i| r.state.items[*i].status == "pending").collect();
    for (n, i) in pending.iter().enumerate() {
        if stop() {
            return r.stopped("stopped by the owner during the checks; nothing is released (resume it to go on)".into());
        }
        if let Err(why) = r.check_item(*i, &mut checks, &mut block)? {
            // the items not checked yet stay pending for a resume
            return r.stopped(why);
        }
        r.state.counts.checked += 1;
        // the share is passed already: no need to check the rest
        if strata_safety::import::source_rejected(r.state.counts.minor, total, share) {
            break;
        }
        if n % 16 == 0 || n + 1 == pending.len() {
            r.state.save(&dir)?;
            let msg = format!("{} of {} items checked", r.state.counts.checked, pending.len());
            r.note(&msg);
        }
    }

    // 4. the whole source rejected when too much of it involved a minor
    if strata_safety::import::source_rejected(r.state.counts.minor, total, share) {
        let reason = format!("{} of {total} items were flagged as involving a minor (more than {:.1} %): the whole import is rejected and its quarantine deleted",
                             r.state.counts.minor, share * 100.0);
        let counts = r.state.counts.clone();
        std::fs::remove_dir_all(&dir).with_context(|| format!("deleting {}", dir.display()))?;
        r.audit.log(&s.id, "import_rejected", json!({"reason": "minor_share", "minor": counts.minor, "items": total, "share_limit": share, "counts": counts}))?;
        return Ok(Outcome::Rejected { reason, counts });
    }

    // 5. release what passed
    let out = s.dataset();
    if out.exists() {
        return r.stopped(format!("{}: a dataset of this name appeared meanwhile; nothing is released", out.display()));
    }
    pd::private_dir(&s.root.join(DATASETS))?;
    pd::private_dir(&out)?;
    pd::private_dir(&out.join("files"))?;
    let mut manifest = String::new();
    let mut text_rows = String::new();
    let passed: Vec<Item> = r.state.items.iter().filter(|i| i.status == "passed").cloned().collect();
    for it in &passed {
        let src = dir.join("items").join(format!("{}.{}", it.sha256, it.ext));
        let rel = format!("files/{}.{}", it.sha256, it.ext);
        let bytes = std::fs::read(&src)?;
        create_private(&out.join(&rel))?.write_all(&bytes)?;
        let kind = match item_kind(&src, &it.ext) { ItemKind::Image => "image", ItemKind::Text => "text", ItemKind::Other => "other" };
        manifest.push_str(&format!("{}\n", json!({"file": rel, "sha256": it.sha256, "kind": kind, "bytes": it.bytes, "import": s.id, "private": true})));
        if kind == "text" {
            let t = String::from_utf8_lossy(&bytes);
            if it.ext == "jsonl" {
                for l in t.lines().filter(|l| !l.trim().is_empty()) {
                    // chat rows kept as they are, marked private; anything else as text
                    match serde_json::from_str::<Value>(l) {
                        Ok(Value::Object(_)) => text_rows.push_str(&format!("{}\n", pd::mark_row(l))),
                        _ => text_rows.push_str(&format!("{}\n", json!({"text": l, "private": true, "meta": {"sha256": it.sha256}}))),
                    }
                }
            } else {
                text_rows.push_str(&format!("{}\n", json!({"text": t, "private": true, "meta": {"sha256": it.sha256}})));
            }
        }
    }
    create_private(&out.join("manifest.jsonl"))?.write_all(manifest.as_bytes())?;
    if !text_rows.is_empty() {
        create_private(&out.join("text.jsonl"))?.write_all(text_rows.as_bytes())?;
    }
    r.state.counts.released = passed.len();
    let counts = r.state.counts.clone();
    create_private(&out.join(pd::CARD))?.write_all(card(s, &r.state, &checks).as_bytes())?;
    std::fs::remove_dir_all(&dir).with_context(|| format!("deleting {}", dir.display()))?;
    r.audit.log(&s.id, "release", json!({"dataset": out.display().to_string(), "released": counts.released, "counts": counts,
                                         "sha256": passed.iter().map(|i| i.sha256.clone()).collect::<Vec<_>>()}))?;
    Ok(Outcome::Released { dir: out, counts })
}

/// The released dataset's card: private, the source and its attestation, the counts and the checks. No content.
fn card(s: &Settings, st: &State, checks: &BTreeMap<CheckKind, Check>) -> String {
    let c = &st.counts;
    let mut o = format!("{}\n# Private dataset card: {}\n\n**Private.** Imported by Prastara's Workshop on this PC and released from \
        quarantine after every item passed the fixed checks. It stays on this PC: never upload, share or publish it, or a model \
        trained on it. Every row carries `\"private\": true`, and models and datasets made from it are marked private too.\n\n\
        ## Source\n\n- {}\n- origin (the owner's words): {}\n- adults only: attested\n- consent: attested\n", pd::CARD_MARK, s.name,
        st.source, st.origin.replace('\n', " "));
    if let Some(l) = &st.licence { o.push_str(&format!("- licence: {}\n", l.replace('\n', " "))); }
    if let Some(c) = &st.commit { o.push_str(&format!("- commit: `{c}`\n")); }
    o.push_str(&format!("- import: `{}`\n\n## Counts\n\n| | |\n|---|---:|\n| Files fetched or copied | {} ({} bytes) |\n| Items checked | {} of {} |\n\
        | Duplicates | {} |\n| Not read (links, hidden, gone) | {} |\n| Archives rejected whole | {} |\n| Rejected | {} |\n\
        | Deleted at once (minor or blocklisted; hashes blocked) | {} |\n| Released | {} |\n",
        st.id, c.fetched, c.bytes, c.checked, c.items, c.duplicates, c.skipped, c.archives_rejected, c.rejected, c.deleted, c.released));
    o.push_str("\n## Rejections by reason\n\n");
    if c.reasons.is_empty() { o.push_str("none\n"); }
    for (k, n) in &c.reasons { o.push_str(&format!("- {k}: {n}\n")); }
    o.push_str(&format!("\n## Checks (fixed; they cannot be switched off)\n\nThe SHA-256 blocklist; the fixed minor-safety word rules for text; \
        the minor share of the source at most {:.1} %; and these check models:\n\n", strata_safety::import::minor_share(s.minor_share) * 100.0));
    for (k, ch) in checks {
        o.push_str(&format!("- {}: flags {} at {:.2}\n", k.label(), ch.rule.flag_labels.join(", "), ch.rule.threshold));
    }
    o
}

#[cfg(test)]
mod tests;
