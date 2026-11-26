//! The Workshop on the server: convert, quantize and merge jobs (strata-workshop, all Rust), queued one at a time and
//! run as the `am2strata-workshop` command (a job that runs out of memory cannot take the server down), with their
//! progress for the web app's Workshop tab and `am2strata-run workshop`.
//!
//! - `POST /v1/workshop/jobs` `{"kind": "convert" | "quantize" | "merge", ...}` queues a job:
//!   - convert: `"input"` (a Hugging Face model folder), `"output"`? (a .gguf), `"outtype"`? (f16 | f32), `"pre_tokenizer"`?
//!   - quantize: `"input"` (a .gguf), `"output"`?, `"type"` (q8_0 | q6_k | q5_k | q4_k | q4_0 | f16)
//!   - merge: `"base"` (an HF model folder), `"adapter"` (a PEFT/Unsloth adapter folder), `"output"`? (a folder)
//!   - loop: `"input"` (an HF model folder), `"start"`, `"end"` (its blocks start..end), `"passes"`? (2-16, default 2),
//!     `"output"`? (a folder) — a looped copy (Avarta): the core runs again behind gates that start closed, so the copy
//!     is exactly its base until trained; convert carries the loop into the GGUF and the Rust engine runs it
//!   - upcycle: `"input"` (a dense HF model folder, looped or not), `"experts"`? (2-256, default 10), `"top_k"`? (1 to
//!     experts, default 1), `"output"`? (a folder; default `<stem>-moe<E>x<K>`) — a linked copy marked as a mixture of
//!     experts (LoRA upcycling): `train` on it trains a router and each expert's LoRA, and it starts as its base
//!   - export-moe: `"input"` (an upcycled folder), `"adapter"`? (the adapter trained on it), `"type"`? (q4_k | q8_0 |
//!     q6_k | q5_k | q4_0 | f16 | f32, default q4_k), `"pre_tokenizer"`?, `"output"`? (a .gguf) — the real qwen3moe
//!     model, each expert's LoRA folded into its own copy of the feed-forward, quantized as it is written
//!   - abliterate: `"input"` (an HF model folder), `"harmful"` and `"harmless"` (text files in Workshop folders, one
//!     prompt per line), `"layer"`? (auto | N | "N:P"), `"scale"`? (0-2, default 1), `"check"`? (held-out prompts to
//!     answer before and after), `"devices"`?, `"output"`? (a folder; default `<stem>-abliterated`) — the refusal
//!     direction removed from every matrix that writes the residual stream; the copy is marked `prastara_private`
//!   - strata: `"input"` (a .gguf), `"name"`?, `"display_name"`?, `"overwrite"`? — make it a model the server runs: the
//!     same run config and tokenizer folder a downloaded model gets (strata-hub's config writer, which picks the engine
//!     for the architecture and refuses one no engine runs), so it appears in the model list
//!   - datagen: `"recipe"` (qa | chat | instruct | code | indic | translate), `"sources"` (documents: files or
//!     folders; needed for qa and chat, optional for code and translate), `"seeds"`? (a JSONL of example chats;
//!     instruct), `"seed"`? and `"check_code"`? (the built-in recipes' draw, and compiling Rust answers),
//!     `"output"`? (a .jsonl), `"count"`?, `"per_chunk"`?, `"chunk_chars"`?, `"temperature"`?, `"min_score"`? (1-5: a judge pass), `"topic"`?, `"language"`?, `"model"`? (the one to write with; default the loaded one) — training data written by the
//!     loaded model, through this server's own chat API (its address and key reach the command in its environment,
//!     never on its command line)
//!   - regress: `"model"` (an HF model folder), `"adapter"`? (an adapter folder), `"suite"` (a suite .json whose data
//!     files must be inside the Workshop's folders too), `"passes"`? ([1, 2] or "1,2": a looped model at each),
//!     `"against"`? (earlier reports, paths or a list), `"devices"`?, `"max_prompt_tokens"`?, `"judge_model"`?,
//!     `"output"`? (a folder; default `reports/<model>[-<adapter>]`) — the regression suite: report.json, report.md and
//!     a regressions list (a GPU run takes the GPUs as training does). `"engine_gguf"`? (a .gguf of the model, its
//!     adapter merged in) answers the generation tests through the Rust engine, with `"engine_args"`? (only
//!     `--devices`, `--max-context`, `--threads`, `--kv`, `--expert-ram-gib`, `--no-graph`, `--graph`,
//!     `--resident-experts`; `--devices cuda:…` makes it a GPU run). A GPU run whose judge is a local model on the GPUs
//!     is refused at once: the judge's chats would wait for the run and the run for the judge; judge with a remote
//!     (assist) model or a CPU model
//!   - grade-train: `"model"`, `"adapter"`?, `"data"` (prompts, .jsonl), `"grader"`? (code | judge), `"samples"`?,
//!     `"rounds"`?, `"suite"`, `"temperature"`?, `"max_tokens"`?, `"min_score"`?, `"lr"`?, `"epochs"`?, `"max_steps"`?,
//!     `"rank"`?, `"passes"`?, `"devices"`?, `"engine_gguf"`?, `"engine_args"`? (as for regress: the starting model's
//!     samples), `"output"`? (a folder; default `grade-train/<model>-<data>`) — rounds of
//!     sampling, grading, best-of-N fine-tuning and the suite, promoting an adapter only when nothing regressed
//! - `POST /v1/workshop/uploads?name=notes.pdf` (the file as the body) saves a document into `<data>/workshop/uploads`.
//! - `GET /v1/workshop/dataset?path=…&offset=&limit=` pages through a dataset's records (with its card);
//!   `POST /v1/workshop/dataset` `{"path", "remove": [i, …], "replace": {"i": record}}` edits it in place.
//! - `GET /v1/workshop/jobs` lists jobs, `GET /v1/workshop/jobs/{id}` one (with its log), `DELETE` cancels.
//! - `GET /v1/workshop/files` lists the models and adapters the Workshop may use, for its pickers.
//! - `GET /v1/workshop/library` lists the model library's recipes (abilities, and whether the Workshop trains each);
//! - `PUT /v1/workshop/upload?name=…` streams a file of any size into the uploads folder; `POST upload-folder?name=…`
//!   then `PUT upload?folder=…&rel=…` per file uploads a whole folder. `GET /v1/workshop/browse?path=…` lists a folder.
//!   `POST /v1/workshop/library` `{"name"}` writes one's soup.yaml and DATA.md into `<data>/workshop/recipes/`.
//!
//! **Paths** must be inside the models folder, `<data>/workshop`, or a folder listed in the run config's
//! `"workshop": {"dirs": [...]}` — the API cannot read or write anywhere else on the PC. Only the server's own key may
//! use the Workshop (not client profiles).

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use axum::extract::{Path as UrlPath, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::{json, Value};

use crate::{authorized, error, AppState};

const KINDS: [&str; 12] = ["convert", "quantize", "merge", "loop", "upcycle", "export-moe", "abliterate", "train", "strata", "datagen", "regress", "grade-train"];

/// What `export-moe` writes as: the converter's two float types and the quantizer's.
const EXPORT_TYPES: [&str; 7] = ["q4_k", "q8_0", "q6_k", "q5_k", "q4_0", "f16", "f32"];

#[derive(Clone, Debug)]
pub struct Job {
    pub id: u64,
    pub kind: String,
    pub args: Vec<String>,
    pub request: Value,
    pub output: PathBuf,
    pub state: &'static str,
    pub queued_at: u64,
    pub started_at: Option<u64>,
    pub finished_at: Option<u64>,
    pub log: Vec<String>,
    pub error: Option<String>,
    pub result: Option<Value>,
    pub pid: Option<u32>,
    /// the last progress event (datagen): {"done", "total", "message"}
    pub progress: Option<Value>,
    /// a training run's curve: {"step", "loss", "lr"} per step and {"step", "eval_loss"} per held-out measure
    pub points: Vec<Value>,
}

impl Job {
    pub fn view(&self, with_log: bool) -> Value {
        let mut v = json!({"id": self.id, "kind": self.kind, "state": self.state, "request": self.request,
                           "output": self.output.display().to_string(), "queued_at": self.queued_at,
                           "started_at": self.started_at, "finished_at": self.finished_at, "error": self.error,
                           "result": self.result, "progress": self.progress});
        if with_log {
            v["log"] = json!(self.log);
            v["points"] = json!(self.points);
        } else {
            v["last"] = json!(self.log.last());
        }
        v
    }
}

pub struct Workshop {
    roots: Vec<PathBuf>,
    out_dir: PathBuf,
    models_dir: PathBuf,
    packs_dir: PathBuf,
    jobs: Mutex<Vec<Job>>,
    next: Mutex<u64>,
    /// jobs run one at a time, in order
    gate: tokio::sync::Mutex<()>,
    /// this server's own address and key, for jobs that call its API (datagen)
    server: Mutex<Option<(String, Option<String>)>>,
    /// the model manager, for a training run on the GPUs: it takes them from the chat model for the run
    mgr: Mutex<Option<Arc<crate::manager::Manager>>>,
}

static WORKSHOP: std::sync::OnceLock<Arc<Workshop>> = std::sync::OnceLock::new();

pub fn get() -> Option<Arc<Workshop>> {
    WORKSHOP.get().cloned()
}

/// A file or folder's name without its extension, for default output names.
pub fn stem_of(p: &str) -> String {
    let p = std::path::Path::new(p.trim_end_matches('/'));
    p.file_stem().and_then(|x| x.to_str()).unwrap_or("model").to_string()
}

fn now() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

/// Whether a model's run config puts it on a GPU: its `gpu` list, `--devices` naming a GPU (the Rust engine), or GPU
/// layers (`--n-gpu-layers` / `-ngl` above 0).
pub fn model_uses_gpu(cfg: &strata_engine::RunConfig) -> bool {
    let val = |k: &str| cfg.args.iter().position(|a| a == k).and_then(|i| cfg.args.get(i + 1));
    !cfg.gpus().is_empty()
        || val("--devices").is_some_and(|d| d.split(',').any(|x| !x.trim().starts_with("cpu")))
        || ["--n-gpu-layers", "-ngl", "--gpu-layers"].iter().any(|k| val(k).and_then(|v| v.parse::<i64>().ok()).is_some_and(|n| n != 0))
}

/// Whether `--engine-args` put the engine on a GPU.
pub fn engine_args_on_gpu(args: &str) -> bool {
    let a: Vec<&str> = args.split_whitespace().collect();
    a.iter().position(|x| *x == "--devices").and_then(|i| a.get(i + 1)).is_some_and(|d| d.split(',').any(|x| !x.trim().starts_with("cpu")))
}

/// `engine_args` for a regression run's Rust engine: only the flags that place and size it (`--devices`,
/// `--max-context`, `--threads`, `--kv`, `--expert-ram-gib`, `--no-graph`, `--graph`, `--resident-experts`), each value
/// checked; nothing that names a file or another machine.
pub fn check_engine_args(args: &str) -> Result<Vec<String>, String> {
    let a: Vec<&str> = args.split_whitespace().collect();
    if args.len() > 400 || a.len() > 24 {
        return Err("\"engine_args\" is too long".into());
    }
    let mut out = Vec::new();
    let mut i = 0;
    while i < a.len() {
        let flag = a[i];
        let val = a.get(i + 1).copied();
        let need = |ok: bool, what: &str| -> Result<String, String> {
            match val {
                Some(v) if ok => Ok(v.to_string()),
                _ => Err(format!("\"engine_args\": {flag} takes {what}")),
            }
        };
        let num = |lo: f64, hi: f64| val.and_then(|v| v.parse::<f64>().ok()).is_some_and(|n| (lo..=hi).contains(&n));
        let v = match flag {
            "--no-graph" | "--graph" | "--resident-experts" => None,
            "--devices" => Some(need(val.is_some_and(|d| d.split(',').all(|x| {
                let (dev, gib) = x.split_once('=').map_or((x, None), |(d, g)| (d, Some(g)));
                let dev_ok = dev == "cpu" || dev.split_once(':').is_some_and(|(k, n)| ["cuda", "hip", "wgpu", "vulkan", "metal"].contains(&k)
                    && !n.is_empty() && n.chars().all(|c| c.is_ascii_digit()));
                dev_ok && gib.is_none_or(|g| g.parse::<f64>().is_ok_and(|g| g > 0.0 && g <= 1024.0))
            })), "devices such as cuda:0=14,cuda:1=14")?),
            "--max-context" => Some(need(num(256.0, 1_048_576.0), "a context of 256 to 1048576 tokens")?),
            "--threads" => Some(need(num(1.0, 1024.0), "a thread count")?),
            "--expert-ram-gib" => Some(need(num(0.0, 4096.0), "a size in GiB")?),
            "--kv" => Some(need(val.is_some_and(|k| !k.is_empty() && k.len() <= 16 && k.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')), "a cache type such as f16 or q8")?),
            other => return Err(format!("\"engine_args\": {other} is not taken here (only --devices, --max-context, --threads, --kv, --expert-ram-gib, \
                                         --no-graph, --graph, --resident-experts)")),
        };
        out.push(flag.to_string());
        if let Some(v) = v {
            out.push(v);
            i += 1;
        }
        i += 1;
    }
    Ok(out)
}

/// The `am2strata-workshop` command: AM2STRATA_WORKSHOP_EXE, else beside this server's binary.
pub fn exe() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("AM2STRATA_WORKSHOP_EXE") {
        return Some(PathBuf::from(p));
    }
    let me = std::env::current_exe().ok()?;
    let name = if cfg!(windows) { "am2strata-workshop.exe" } else { "am2strata-workshop" };
    let p = me.with_file_name(name);
    p.exists().then_some(p)
}

/// Paths the Workshop never reads, writes, lists or takes uploads into, wherever they are, even inside a Workshop
/// folder or under a folder added to the config: the system's own areas, credential folders and secret files.
/// Returns why, for the refusal.
pub fn sensitive(p: &Path) -> Option<&'static str> {
    const SYSTEM: &[&str] = &["/etc", "/proc", "/sys", "/dev", "/boot", "/root", "/run", "/var/lib", "/var/log", "/usr/bin",
                              "/usr/sbin", "/bin", "/sbin", "/lib", "/lib64", "/snap", "/private/etc", "/System", "/Library/Keychains"];
    if SYSTEM.iter().any(|s| p.starts_with(s)) {
        return Some("a system folder");
    }
    const DIRS: &[&str] = &[".ssh", ".gnupg", ".aws", ".azure", ".kube", ".docker", ".password-store", ".pki", "keyrings",
                            ".mozilla", ".thunderbird", ".local/share/keyrings", ".config/gh", ".config/gcloud", ".vault-token"];
    let parts: Vec<String> = p.components().filter_map(|c| match c {
        std::path::Component::Normal(n) => Some(n.to_string_lossy().to_lowercase()),
        _ => None,
    }).collect();
    let joined = parts.join("/");
    if parts.iter().any(|x| DIRS.contains(&x.as_str())) || DIRS.iter().any(|d| d.contains('/') && joined.contains(d)) {
        return Some("a folder of keys or credentials");
    }
    let name = parts.last().map(String::as_str).unwrap_or("");
    let secret = matches!(name, "passwd" | "shadow" | "gshadow" | "sudoers" | ".env" | ".netrc" | ".git-credentials" | ".pgpass"
                                | "node-tls.json" | "authorized_keys" | "known_hosts" | "credentials" | "credentials.json")
        || ["id_rsa", "id_dsa", "id_ecdsa", "id_ed25519"].iter().any(|k| name.starts_with(k))
        || [".pem", ".key", ".p12", ".pfx", ".keystore", ".jks", ".kdbx", ".asc", ".gpg"].iter().any(|e| name.ends_with(e))
        || name.starts_with(".env.");
    secret.then_some("a file of keys, passwords or credentials")
}

impl Workshop {
    pub fn init(models_dir: &Path, data_dir: &Path, rest: &std::collections::BTreeMap<String, Value>) -> Arc<Workshop> {
        let out_dir = data_dir.join("workshop");
        let mut roots = vec![models_dir.to_path_buf(), out_dir.clone()];
        if let Some(dirs) = rest.get("workshop").and_then(|w| w.get("dirs")).and_then(|d| d.as_array()) {
            roots.extend(dirs.iter().filter_map(|d| d.as_str()).map(|d| crate::expand_path(d)));
        }
        // every root resolved (symlinks followed) once, here; one that cannot be is left out, never compared unresolved
        std::fs::create_dir_all(&out_dir).ok();
        let home = std::env::var_os("HOME").map(PathBuf::from).and_then(|h| h.canonicalize().ok());
        let roots = roots.into_iter().filter_map(|r| match r.canonicalize() {
            Ok(c) if c.parent().is_none() || Some(&c) == home.as_ref() || sensitive(&c).is_some() => {
                tracing::warn!("workshop: {} is too broad or sensitive to be a Workshop folder (the whole disk, a home folder, or system or key folders); left out", c.display());
                None
            }
            Ok(c) => Some(c),
            Err(e) => {
                tracing::warn!("workshop: {} is not usable as a Workshop folder ({e}); left out", r.display());
                None
            }
        }).collect();
        let (models_dir, packs_dir) = (models_dir.to_path_buf(), data_dir.join("packs"));
        WORKSHOP.get_or_init(|| Arc::new(Workshop { roots, out_dir, models_dir, packs_dir, jobs: Mutex::new(vec![]), next: Mutex::new(1),
                                                    gate: tokio::sync::Mutex::new(()), server: Mutex::new(None),
                                                    mgr: Mutex::new(None) })).clone()
    }

    /// Where this server answers (bound to `host`:`port`), so a datagen job can use its chat API.
    pub fn set_manager(&self, m: Arc<crate::manager::Manager>) {
        *self.mgr.lock().unwrap() = Some(m);
    }

    pub fn set_server(&self, host: &str, port: u16, key: Option<String>) {
        let h = match host {
            "" | "0.0.0.0" | "::" | "[::]" => "127.0.0.1".to_string(),
            h if h.contains(':') && !h.starts_with('[') => format!("[{h}]"),
            h => h.to_string(),
        };
        *self.server.lock().unwrap() = Some((format!("http://{h}:{port}"), key));
    }

    pub fn uploads_dir(&self) -> PathBuf {
        self.out_dir.join("uploads")
    }

    /// A new file in the Workshop folder: `dir/name`, or `name-2`, `name-3`, … when it is taken.
    fn fresh(dir: &Path, stem: &str, ext: &str) -> PathBuf {
        std::fs::create_dir_all(dir).ok();
        let mut p = dir.join(format!("{stem}{ext}"));
        let mut n = 2;
        while p.exists() {
            p = dir.join(format!("{stem}-{n}{ext}"));
            n += 1;
        }
        p
    }

    pub fn roots(&self) -> &[PathBuf] {
        &self.roots
    }

    /// `p` resolved, if it is inside one of the allowed folders (an output need not exist yet: its parent is checked).
    pub fn allowed(&self, p: &str, must_exist: bool) -> Result<PathBuf, String> {
        let path = crate::expand_path(p);
        if !path.is_absolute() {
            return Err(format!("{p}: give the full path (it is resolved on the server)"));
        }
        let resolved = if path.exists() {
            // symlinks followed: a link inside a Workshop folder that points outside it is outside
            path.canonicalize().map_err(|e| format!("{p}: {e}"))?
        } else if must_exist {
            return Err(format!("{p} does not exist"));
        } else {
            // a new output: the nearest folder that exists, resolved, plus the plain names after it (no "..")
            let mut base = path.as_path();
            let mut rest: Vec<&std::ffi::OsStr> = vec![];
            while !base.exists() {
                match (base.parent(), base.file_name()) {
                    (Some(parent), Some(name)) => { rest.push(name); base = parent; }
                    _ => return Err(format!("{p}: not a usable path")),
                }
            }
            if path.components().any(|c| matches!(c, std::path::Component::ParentDir | std::path::Component::CurDir)) {
                return Err(format!("{p}: \"..\" and \".\" are not allowed in a new path"));
            }
            let mut r = base.canonicalize().map_err(|e| format!("{p}: {e}"))?;
            for name in rest.iter().rev() {
                r.push(name);
            }
            r
        };
        if let Some(why) = sensitive(&resolved).or_else(|| sensitive(&path)) {
            return Err(format!("{p}: {why}; the Workshop never touches these"));
        }
        if !self.roots.iter().any(|r| resolved.starts_with(r)) {
            return Err(format!("{p} is outside the Workshop's folders ({}); add its folder to \"workshop\": {{\"dirs\": [...]}} in the run config",
                               self.roots.iter().map(|r| r.display().to_string()).collect::<Vec<_>>().join(", ")));
        }
        // only now, inside a Workshop folder, are missing folders for a new output made
        if !must_exist {
            if let Some(parent) = resolved.parent() {
                std::fs::create_dir_all(parent).map_err(|e| format!("{}: {e}", parent.display()))?;
            }
        }
        Ok(resolved)
    }

    fn default_output(&self, input: &Path, suffix: &str) -> PathBuf {
        let stem = input.file_stem().and_then(|s| s.to_str()).unwrap_or("model");
        self.out_dir.join(format!("{stem}{suffix}"))
    }

    /// A training request's settings as `am2strata-workshop` run arguments (shared by `train` and the estimate):
    /// `model` (an HF folder), `data` (a dataset), and optional `rank`, `alpha`, `epochs`, `lr`, `max_len`,
    /// `train_layers`, `targets` (comma-separated projections), `devices` ("cpu", "cuda:0" or "cuda:0,cuda:1"),
    /// `gradient_checkpointing`, `base_dtype` (f32, f16, bf16; bf16 by default on GPUs), `val_split`, `max_steps`.
    pub fn train_args(&self, req: &Value) -> Result<Vec<String>, String> {
        let s = |k: &str| req[k].as_str().filter(|v| !v.is_empty());
        let model = self.allowed(s("model").ok_or("training needs \"model\": a Hugging Face model folder")?, true)?;
        if !model.join("config.json").exists() {
            return Err(format!("{} has no config.json: training takes a Hugging Face model folder", model.display()));
        }
        let data = self.dataset_path(s("data").ok_or("training needs \"data\": a .jsonl dataset")?)?;
        if !data.exists() {
            return Err(format!("{} does not exist", data.display()));
        }
        let mut a: Vec<String> = vec!["--model".into(), model.display().to_string(), "--data".into(), data.display().to_string()];
        let devices = s("devices").unwrap_or("cpu");
        let ok = devices == "cpu" || devices.split(',').all(|d| d.strip_prefix("cuda:").is_some_and(|n| !n.is_empty() && n.chars().all(|c| c.is_ascii_digit())));
        if !ok {
            return Err(format!("\"devices\" is cpu, cuda:0 or cuda:0,cuda:1 (got {devices:?})"));
        }
        if devices != "cpu" {
            a.extend(["--devices".into(), devices.into()]);
            if s("base_dtype").is_none() {
                a.extend(["--base-dtype".into(), "bf16".into()]);
            }
        }
        for (k, flag, lo, hi) in [("rank", "--rank", 1.0, 512.0), ("alpha", "--alpha", 0.0, 4096.0), ("epochs", "--epochs", 0.01, 100.0),
                                  ("lr", "--lr", 1e-8, 1.0), ("max_len", "--max-len", 16.0, 1_048_576.0), ("train_layers", "--train-layers", 1.0, 1024.0),
                                  ("val_split", "--val-split", 0.0, 0.5), ("max_steps", "--max-steps", 1.0, 10_000_000.0)] {
            match &req[k] {
                Value::Null => {}
                v => {
                    let n = v.as_f64().filter(|n| (lo..=hi).contains(n)).ok_or_else(|| format!("\"{k}\" is a number from {lo} to {hi}"))?;
                    a.extend([flag.into(), n.to_string()]);
                }
            }
        }
        if let Some(t) = s("targets") {
            const ALL: [&str; 7] = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"];
            if !t.split(',').all(|x| ALL.contains(&x.trim())) {
                return Err(format!("\"targets\" are some of {}", ALL.join(",")));
            }
            a.extend(["--targets".into(), t.replace(' ', "")]);
        }
        if let Some(b) = s("base_dtype") {
            if !["f32", "f16", "bf16"].contains(&b) {
                return Err("\"base_dtype\" is f32, f16 or bf16".into());
            }
            a.extend(["--base-dtype".into(), b.into()]);
        }
        if req["gradient_checkpointing"] == true {
            a.push("--gradient-checkpointing".into());
        }
        Ok(a)
    }

    /// Whether a training request runs on GPUs (it then takes them from the chat model for the run).
    pub fn train_on_gpu(req: &Value) -> bool {
        req["devices"].as_str().is_some_and(|d| d != "cpu" && !d.is_empty())
            // a regression run's answers through the Rust engine on a GPU take the GPUs too
            || req["engine_args"].as_str().is_some_and(|a| engine_args_on_gpu(a))
    }

    /// The Rust engine for a job's answers (regress, grade-train): AM2STRATA_HYBRID_EXE (the job inherits it), else
    /// beside this server, else the engine a run config of this server names; None leaves it to the command (beside it
    /// or on the PATH).
    fn engine_exe(&self) -> Option<PathBuf> {
        if std::env::var_os("AM2STRATA_HYBRID_EXE").is_some() {
            return None;
        }
        let name = if cfg!(windows) { "am2strata-hybrid.exe" } else { "am2strata-hybrid" };
        if let Some(p) = std::env::current_exe().ok().map(|e| e.with_file_name(name)).filter(|p| p.is_file()) {
            return Some(p);
        }
        let mgr = self.mgr.lock().unwrap().clone()?;
        mgr.models().into_iter().map(|m| m.cfg.exe).find(|e| e.file_name().and_then(|n| n.to_str()) == Some(name) && e.is_file())
    }

    /// Why a GPU run's judge cannot be served (None: it can): the judge is a local model on the GPUs the run takes, so
    /// its chat would wait for the run and the run for the judge. A remote (assist) model or a CPU model can judge.
    pub async fn judge_conflict(&self, req: &Value) -> Option<String> {
        let kind = req["kind"].as_str()?;
        if !matches!(kind, "regress" | "grade-train") || !Self::train_on_gpu(req) {
            return None;
        }
        let suite: Value = req["suite"].as_str().and_then(|p| std::fs::read_to_string(crate::expand_path(p)).ok()).and_then(|t| serde_json::from_str(&t).ok())?;
        let judged = suite["tests"].as_array().is_some_and(|t| t.iter().any(|x| x["kind"] == "judge")) || (kind == "grade-train" && req["grader"] == "judge");
        if !judged {
            return None;
        }
        let mgr = self.mgr.lock().unwrap().clone()?;
        let named = req["judge_model"].as_str().or(suite["judge_model"].as_str()).filter(|n| !n.is_empty()).map(str::to_string);
        let id = match named {
            Some(n) => n,
            None => match mgr.current().await {
                Some(l) => l.entry.id.clone(),
                None => mgr.default_id.clone(),
            },
        };
        let entry = mgr.entry(&id)?; // not a local model: a remote one (or unknown, which the command reports)
        model_uses_gpu(&entry.cfg).then(|| format!("the judge {id} runs on the GPUs this run takes: it could not answer until the run ends, \
            and the run would wait for it. Judge with a remote (assist) model or a model that runs on the CPU (\"judge_model\"), \
            or run the model under test on the CPU"))
    }

    /// Check a request and turn it into the command's arguments and its output path.
    pub fn plan(&self, req: &Value) -> Result<(String, Vec<String>, PathBuf), String> {
        let kind = req["kind"].as_str().ok_or_else(|| format!("\"kind\" is one of {}", KINDS.join(", ")))?;
        let s = |k: &str| req[k].as_str().filter(|v| !v.is_empty());
        std::fs::create_dir_all(&self.out_dir).ok();
        match kind {
            "convert" => {
                let input = self.allowed(s("input").ok_or("convert needs \"input\": a Hugging Face model folder")?, true)?;
                if !input.join("config.json").exists() {
                    return Err(format!("{} has no config.json: convert takes a Hugging Face model folder", input.display()));
                }
                let outtype = s("outtype").unwrap_or("f16");
                if !["f16", "f32"].contains(&outtype) {
                    return Err("\"outtype\" is f16 or f32".into());
                }
                let output = match s("output") {
                    Some(o) => self.allowed(o, false)?,
                    None => self.default_output(&input, &format!("-{outtype}.gguf")),
                };
                let mut args = vec!["convert".into(), input.display().to_string(), "--out".into(), output.display().to_string(),
                                    "--outtype".into(), outtype.into()];
                if let Some(p) = s("pre_tokenizer") {
                    args.extend(["--pre-tokenizer".into(), p.into()]);
                }
                Ok((kind.into(), args, output))
            }
            "quantize" => {
                let input = self.allowed(s("input").ok_or("quantize needs \"input\": a .gguf")?, true)?;
                let ty = s("type").ok_or("quantize needs \"type\": q8_0, q6_k, q5_k, q4_k, q4_0 or f16")?.to_lowercase();
                if !["q8_0", "q6_k", "q5_k", "q4_k", "q4_0", "f16"].contains(&ty.as_str()) {
                    return Err(format!("\"type\" {ty:?} is not one of q8_0, q6_k, q5_k, q4_k, q4_0, f16"));
                }
                let output = match s("output") {
                    Some(o) => self.allowed(o, false)?,
                    None => {
                        let stem = input.file_stem().and_then(|x| x.to_str()).unwrap_or("model");
                        let stem = stem.trim_end_matches("-f16").trim_end_matches("-f32");
                        self.out_dir.join(format!("{stem}-{}.gguf", ty.to_uppercase()))
                    }
                };
                Ok((kind.into(), vec!["quantize".into(), input.display().to_string(), "--out".into(), output.display().to_string(),
                                      "--type".into(), ty], output))
            }
            "merge" => {
                let base = self.allowed(s("base").ok_or("merge needs \"base\": the base model's Hugging Face folder")?, true)?;
                let adapter = self.allowed(s("adapter").ok_or("merge needs \"adapter\": the adapter folder")?, true)?;
                if !adapter.join("adapter_config.json").exists() {
                    return Err(format!("{} has no adapter_config.json: not a LoRA adapter folder", adapter.display()));
                }
                let output = match s("output") {
                    Some(o) => self.allowed(o, false)?,
                    None => self.default_output(&adapter, "-merged"),
                };
                Ok((kind.into(), vec!["merge".into(), base.display().to_string(), adapter.display().to_string(),
                                      "--out".into(), output.display().to_string()], output))
            }
            "loop" => {
                let input = self.allowed(s("input").ok_or("loop needs \"input\": a Hugging Face model folder")?, true)?;
                let cfg: Value = std::fs::read_to_string(input.join("config.json")).ok().and_then(|t| serde_json::from_str(&t).ok())
                    .ok_or_else(|| format!("{} has no config.json: loop takes a Hugging Face model folder", input.display()))?;
                let layers = cfg["num_hidden_layers"].as_u64().ok_or("the model's config.json has no num_hidden_layers")?;
                if !cfg["prastara_loop"].is_null() {
                    return Err(format!("{} is already looped", input.display()));
                }
                let n = |k: &str| req[k].as_u64();
                let (start, end) = (n("start").ok_or("loop needs \"start\": the loop's first block")?, n("end").ok_or("loop needs \"end\": the block after its last")?);
                let passes = n("passes").unwrap_or(2);
                if start >= end || end > layers {
                    return Err(format!("the loop's blocks {start}..{end} must be a non-empty range inside the model's {layers}"));
                }
                if !(2..=16).contains(&passes) {
                    return Err(format!("\"passes\" is 2 to 16 (got {passes})"));
                }
                let output = match s("output") {
                    Some(o) => self.allowed(o, false)?,
                    None => self.default_output(&input, &format!("-loop{start}-{end}x{passes}")),
                };
                if output.exists() {
                    return Err(format!("{} already exists", output.display()));
                }
                Ok((kind.into(), vec!["loop".into(), input.display().to_string(), "--out".into(), output.display().to_string(),
                                      "--start".into(), start.to_string(), "--end".into(), end.to_string(), "--passes".into(), passes.to_string()], output))
            }
            "upcycle" => {
                let input = self.allowed(s("input").ok_or("upcycle needs \"input\": a dense Hugging Face model folder")?, true)?;
                let cfg: Value = std::fs::read_to_string(input.join("config.json")).ok().and_then(|t| serde_json::from_str(&t).ok())
                    .ok_or_else(|| format!("{} has no config.json: upcycle takes a Hugging Face model folder", input.display()))?;
                if !cfg["prastara_moe"].is_null() {
                    return Err(format!("{} is already upcycled", input.display()));
                }
                if let Some(k) = ["num_experts", "num_local_experts", "num_experts_per_tok", "moe_intermediate_size"].iter()
                    .find(|k| cfg.get(**k).is_some_and(|v| !v.is_null() && v.as_u64() != Some(0))) {
                    return Err(format!("{} is a mixture of experts already ({k} in its config.json): upcycle starts from a dense model", input.display()));
                }
                let num = |k: &str, default: u64| -> Result<u64, String> {
                    match &req[k] { Value::Null => Ok(default), v => v.as_u64().ok_or_else(|| format!("\"{k}\" is a whole number")) }
                };
                let (experts, top_k) = (num("experts", 10)?, num("top_k", 1)?);
                if !(2..=256).contains(&experts) {
                    return Err(format!("\"experts\" is 2 to 256 (got {experts})"));
                }
                if top_k == 0 || top_k > experts {
                    return Err(format!("\"top_k\" is 1 to the number of experts, {experts} (got {top_k})"));
                }
                let output = match s("output") {
                    Some(o) => self.allowed(o, false)?,
                    None => self.default_output(&input, &format!("-moe{experts}x{top_k}")),
                };
                if output.exists() {
                    return Err(format!("{} already exists", output.display()));
                }
                Ok((kind.into(), vec!["upcycle".into(), input.display().to_string(), "--out".into(), output.display().to_string(),
                                      "--experts".into(), experts.to_string(), "--top-k".into(), top_k.to_string()], output))
            }
            "export-moe" => {
                let input = self.allowed(s("input").ok_or("export-moe needs \"input\": an upcycled model folder")?, true)?;
                let cfg: Value = std::fs::read_to_string(input.join("config.json")).ok().and_then(|t| serde_json::from_str(&t).ok())
                    .ok_or_else(|| format!("{} has no config.json: export-moe takes an upcycled model folder", input.display()))?;
                let moe = &cfg["prastara_moe"];
                if moe.is_null() {
                    return Err(format!("{} is not upcycled (no prastara_moe in its config.json): run an upcycle job on it first", input.display()));
                }
                let ty = s("type").unwrap_or("q4_k").to_ascii_lowercase();
                if !EXPORT_TYPES.contains(&ty.as_str()) {
                    return Err(format!("\"type\" is one of {} (got {ty:?})", EXPORT_TYPES.join(", ")));
                }
                let mut args = vec!["export-moe".to_string(), input.display().to_string()];
                if let Some(a) = s("adapter") {
                    let a = self.allowed(a, true)?;
                    let ac: Value = std::fs::read_to_string(a.join("adapter_config.json")).ok().and_then(|t| serde_json::from_str(&t).ok())
                        .ok_or_else(|| format!("{} has no adapter_config.json: \"adapter\" is the folder train wrote", a.display()))?;
                    if ac["prastara_moe"]["experts"] != moe["experts"] || ac["prastara_moe"]["top_k"] != moe["top_k"] {
                        return Err(format!("{} was not trained on this upcycled model (its prastara_moe is {}, the model's {moe})", a.display(), ac["prastara_moe"]));
                    }
                    args.extend(["--adapter".into(), a.display().to_string()]);
                }
                if let Some(p) = s("pre_tokenizer") {
                    if !p.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.') {
                        return Err("\"pre_tokenizer\" is a name like qwen2 or llama-bpe".into());
                    }
                    args.extend(["--pre-tokenizer".into(), p.into()]);
                }
                let output = match s("output") {
                    Some(o) => self.allowed(o, false)?,
                    None => self.default_output(&input, &format!("-{ty}.gguf")),
                };
                if output.exists() {
                    return Err(format!("{} already exists", output.display()));
                }
                args.extend(["--out".into(), output.display().to_string(), "--type".into(), ty]);
                Ok((kind.into(), args, output))
            }
            "abliterate" => {
                let input = self.allowed(s("input").ok_or("abliterate needs \"input\": a Hugging Face model folder")?, true)?;
                if !input.join("config.json").exists() || !input.join("tokenizer.json").exists() {
                    return Err(format!("{} needs config.json and tokenizer.json: abliterate takes a Hugging Face model folder", input.display()));
                }
                let list = |k: &str, what: &str| -> Result<PathBuf, String> {
                    let p = self.allowed(s(k).ok_or_else(|| format!("abliterate needs \"{k}\": a text file of {what} prompts, one per line"))?, true)?;
                    if !p.is_file() {
                        return Err(format!("{}: \"{k}\" is a text file, one prompt per line", p.display()));
                    }
                    Ok(p)
                };
                let (harmful, harmless) = (list("harmful", "refused")?, list("harmless", "ordinary")?);
                let layer = match &req["layer"] {
                    Value::Null => "auto".to_string(),
                    Value::Number(n) => n.as_u64().ok_or("\"layer\" is auto, a block number, or N:P")?.to_string(),
                    Value::String(t) if t.is_empty() || t == "auto" => "auto".to_string(),
                    Value::String(t) => {
                        let ok = t.split(':').count() <= 2 && t.split(':').all(|x| !x.is_empty() && x.chars().all(|c| c.is_ascii_digit()));
                        if !ok { return Err("\"layer\" is auto, a block number, or N:P (block N in loop pass P)".into()); }
                        t.clone()
                    }
                    _ => return Err("\"layer\" is auto, a block number, or N:P".into()),
                };
                let scale = match &req["scale"] {
                    Value::Null => 1.0,
                    v => v.as_f64().filter(|x| *x > 0.0 && *x <= 2.0).ok_or("\"scale\" is a number above 0, at most 2 (1 removes the direction)")?,
                };
                let check = match &req["check"] {
                    Value::Null => 0,
                    v => v.as_u64().filter(|n| *n <= 256).ok_or("\"check\" is a whole number of held-out prompts, 0 to 256")?,
                };
                let devices = s("devices").unwrap_or("cpu");
                let ok = devices == "cpu" || devices.split(',').all(|d| d.strip_prefix("cuda:").is_some_and(|n| !n.is_empty() && n.chars().all(|c| c.is_ascii_digit())));
                if !ok {
                    return Err(format!("\"devices\" is cpu, cuda:0 or cuda:0,cuda:1 (got {devices:?})"));
                }
                let output = match s("output") {
                    Some(o) => self.allowed(o, false)?,
                    None => Self::fresh(&self.out_dir, &format!("{}-abliterated", input.file_name().and_then(|x| x.to_str()).unwrap_or("model")), ""),
                };
                if output.exists() {
                    return Err(format!("{} already exists", output.display()));
                }
                let mut args = vec!["abliterate".into(), input.display().to_string(), "--harmful".into(), harmful.display().to_string(),
                                    "--harmless".into(), harmless.display().to_string(), "--layer".into(), layer, "--scale".into(), scale.to_string(),
                                    "--check".into(), check.to_string(), "--out".into(), output.display().to_string()];
                if devices != "cpu" {
                    args.extend(["--devices".into(), devices.into()]);
                }
                Ok((kind.into(), args, output))
            }
            "train" => {
                let mut args = vec!["train".to_string()];
                args.extend(self.train_args(req)?);
                let output = match s("output") {
                    Some(o) => self.allowed(o, false)?,
                    None => {
                        let stem = |k: &str| s(k).map(|p| crate::workshop::stem_of(p)).unwrap_or_default();
                        Self::fresh(&self.out_dir.join("adapters"), &format!("{}-{}", stem("model"), stem("data")), "")
                    }
                };
                if output.exists() {
                    return Err(format!("{} already exists", output.display()));
                }
                args.extend(["--out".into(), output.display().to_string()]);
                Ok((kind.into(), args, output))
            }
            "strata" => {
                let input = self.allowed(s("input").ok_or("strata needs \"input\": a .gguf")?, true)?;
                let file = input.file_name().and_then(|n| n.to_str()).unwrap_or("model.gguf");
                let id = s("name").map(strata_hub::config::id_of).unwrap_or_else(|| strata_hub::config::id_of(file));
                if id.is_empty() {
                    return Err("\"name\" has no letters or digits".into());
                }
                let cfg = self.models_dir.join(format!("strata-{id}.json"));
                if cfg.exists() && req["overwrite"] != true {
                    return Err(format!("a model named {id} is already installed ({}); pick another \"name\" or set \"overwrite\"", cfg.display()));
                }
                Ok((kind.into(), vec![input.display().to_string(), id, s("display_name").unwrap_or("").to_string()], cfg))
            }
            "datagen" => {
                let recipe = s("recipe").unwrap_or("qa");
                const RECIPES: [&str; 6] = ["qa", "chat", "instruct", "code", "indic", "translate"];
                if !RECIPES.contains(&recipe) {
                    return Err(format!("\"recipe\" is one of {}", RECIPES.join(", ")));
                }
                let mut sources = vec![];
                for v in req["sources"].as_array().map(|a| a.as_slice()).unwrap_or(&[]) {
                    let p = v.as_str().filter(|p| !p.is_empty()).ok_or("\"sources\" is a list of paths")?;
                    sources.push(self.allowed(p, true)?);
                }
                let seeds = s("seeds").map(|p| self.allowed(p, true)).transpose()?;
                if matches!(recipe, "qa" | "chat") && sources.is_empty() {
                    return Err(format!("{recipe} needs \"sources\": documents (files or folders) to write the data from"));
                }
                if recipe == "instruct" && seeds.is_none() {
                    return Err("instruct needs \"seeds\": a JSONL of example chats to write more like".into());
                }
                let num = |k: &str, lo: f64, hi: f64| -> Result<Option<String>, String> {
                    match &req[k] {
                        Value::Null => Ok(None),
                        v => v.as_f64().filter(|n| (lo..=hi).contains(n)).map(|n| Some(n.to_string()))
                            .ok_or_else(|| format!("\"{k}\" is a number from {lo} to {hi}")),
                    }
                };
                let output = match s("output") {
                    Some(o) if !o.ends_with(".jsonl") => return Err("\"output\" is a .jsonl file".into()),
                    Some(o) => self.allowed(o, false)?,
                    None => {
                        let from = sources.first().or(seeds.as_ref()).and_then(|p| p.file_stem()).and_then(|x| x.to_str()).unwrap_or("data");
                        Self::fresh(&self.out_dir.join("datasets"), &format!("{from}-{recipe}"), ".jsonl")
                    }
                };
                let mut args = vec!["datagen".into(), "--recipe".into(), recipe.into(), "--out".into(), output.display().to_string()];
                for p in &sources {
                    args.extend(["--source".into(), p.display().to_string()]);
                }
                if let Some(p) = &seeds {
                    args.extend(["--seeds".into(), p.display().to_string()]);
                }
                if recipe == "indic" && !sources.is_empty() {
                    return Err("indic writes from its built-in catalogue: leave out \"sources\"".into());
                }
                if req["check_code"] == true {
                    if recipe != "code" { return Err("\"check_code\" is for the code recipe".into()); }
                    args.push("--check-code".into());
                }
                for (k, flag, lo, hi) in [("seed", "--seed", 0.0, 1e15), ("count", "--count", 1.0, 100_000.0), ("per_chunk", "--per-chunk", 1.0, 20.0),
                                          ("chunk_chars", "--chunk-chars", 500.0, 20_000.0), ("temperature", "--temperature", 0.0, 2.0),
                                          ("min_score", "--min-score", 0.0, 5.0)] {
                    if let Some(v) = num(k, lo, hi)? {
                        // integers for the integer flags (a JSON 100.0 is 100)
                        let v = if k == "temperature" { v } else { (v.parse::<f64>().unwrap_or(0.0) as u64).to_string() };
                        args.extend([flag.into(), v]);
                    }
                }
                for k in ["topic", "language", "model"] {
                    if let Some(v) = s(k) {
                        args.extend([format!("--{k}"), v.chars().take(200).collect()]);
                    }
                }
                Ok((kind.into(), args, output))
            }
            "regress" | "grade-train" => {
                let model = self.allowed(s("model").ok_or_else(|| format!("{kind} needs \"model\": a Hugging Face model folder"))?, true)?;
                if !model.join("config.json").exists() || !model.join("tokenizer.json").exists() {
                    return Err(format!("{} needs config.json and tokenizer.json: {kind} takes a Hugging Face model folder", model.display()));
                }
                let suite = self.suite_path(s("suite").ok_or_else(|| format!("{kind} needs \"suite\": a suite .json listing the tests"))?)?;
                let mut args = vec![kind.to_string(), "--model".into(), model.display().to_string(), "--suite".into(), suite.display().to_string()];
                let adapter = match &req["adapter"] {
                    Value::Null => None,
                    Value::String(a) if a.is_empty() => None,
                    Value::String(a) => {
                        let a = self.allowed(a, true)?;
                        if !a.join("adapter_config.json").exists() {
                            return Err(format!("{} has no adapter_config.json: \"adapter\" is a LoRA adapter folder", a.display()));
                        }
                        args.extend(["--adapter".into(), a.display().to_string()]);
                        Some(a)
                    }
                    _ => return Err("\"adapter\" is the path of an adapter folder".into()),
                };
                let passes: Vec<u64> = match &req["passes"] {
                    Value::Null => vec![],
                    Value::String(t) if t.trim().is_empty() => vec![],
                    Value::String(t) => t.split(',').map(|x| x.trim().parse::<u64>().map_err(|_| "\"passes\" is a list of pass counts, e.g. [1, 2] or \"1,2\"".to_string())).collect::<Result<_, _>>()?,
                    Value::Array(a) => a.iter().map(|x| x.as_u64().ok_or("\"passes\" is a list of pass counts, e.g. [1, 2]".to_string())).collect::<Result<_, _>>()?,
                    _ => return Err("\"passes\" is a list of pass counts, e.g. [1, 2]".into()),
                };
                if passes.iter().any(|p| !(1..=16).contains(p)) || passes.len() > 16 {
                    return Err("\"passes\": each pass count is 1 to 16".into());
                }
                if !passes.is_empty() {
                    args.extend(["--passes".into(), passes.iter().map(|p| p.to_string()).collect::<Vec<_>>().join(",")]);
                }
                let devices = s("devices").unwrap_or("cpu");
                let ok = devices == "cpu" || devices.split(',').all(|d| d.strip_prefix("cuda:").is_some_and(|n| !n.is_empty() && n.chars().all(|c| c.is_ascii_digit())));
                if !ok {
                    return Err(format!("\"devices\" is cpu, cuda:0 or cuda:0,cuda:1 (got {devices:?})"));
                }
                if devices != "cpu" {
                    args.extend(["--devices".into(), devices.into()]);
                }
                let num = |k: &str, lo: f64, hi: f64| -> Result<Option<f64>, String> {
                    match &req[k] {
                        Value::Null => Ok(None),
                        v => v.as_f64().filter(|n| (lo..=hi).contains(n)).map(Some).ok_or_else(|| format!("\"{k}\" is a number from {lo} to {hi}")),
                    }
                };
                if let Some(n) = num("max_prompt_tokens", 16.0, 1_048_576.0)? {
                    args.extend(["--max-prompt-tokens".into(), (n as u64).to_string()]);
                }
                if let Some(m) = s("judge_model") {
                    if m.len() > 200 || m.chars().any(|c| c.is_control()) {
                        return Err("\"judge_model\" is a model name".into());
                    }
                    args.extend(["--judge-model".into(), m.into()]);
                } else if !req["judge_model"].is_null() && !req["judge_model"].is_string() {
                    return Err("\"judge_model\" is a model name".into());
                }
                // the answers through the Rust engine on a GGUF of the model (its adapter merged in)
                match (&req["engine_gguf"], &req["engine_args"]) {
                    (Value::Null, Value::Null) => {}
                    (Value::String(g), a) if !g.is_empty() => {
                        if !g.ends_with(".gguf") {
                            return Err("\"engine_gguf\" is a .gguf file of the model".into());
                        }
                        let g = self.allowed(g, true)?;
                        if !g.is_file() {
                            return Err(format!("{}: \"engine_gguf\" is a .gguf file", g.display()));
                        }
                        args.extend(["--engine-gguf".into(), g.display().to_string()]);
                        match a {
                            Value::Null => {}
                            Value::String(t) if t.trim().is_empty() => {}
                            Value::String(t) => args.extend(["--engine-args".into(), check_engine_args(t)?.join(" ")]),
                            _ => return Err("\"engine_args\" is a string of engine flags, e.g. \"--devices cuda:0=14 --max-context 8192\"".into()),
                        }
                        if let Some(e) = self.engine_exe() {
                            args.extend(["--engine-exe".into(), e.display().to_string()]);
                        }
                    }
                    (Value::String(_) | Value::Null, _) if req["engine_gguf"].as_str().is_none_or(str::is_empty) && !req["engine_args"].is_null() => {
                        return Err("\"engine_args\" goes with \"engine_gguf\"".into());
                    }
                    (Value::String(_), _) => {}
                    _ => return Err("\"engine_gguf\" is the path of a .gguf file".into()),
                }
                let stem = |p: &Path| p.file_name().and_then(|x| x.to_str()).unwrap_or("model").to_string();
                if kind == "regress" {
                    let against: Vec<&str> = match &req["against"] {
                        Value::Null => vec![],
                        Value::String(a) if a.is_empty() => vec![],
                        Value::String(a) => vec![a.as_str()],
                        Value::Array(a) => a.iter().map(|x| x.as_str().ok_or("\"against\" is a report path or a list of them")).collect::<Result<_, _>>()?,
                        _ => return Err("\"against\" is a report path or a list of them".into()),
                    };
                    for a in against {
                        let p = self.allowed(a, true)?;
                        let f = if p.is_dir() { p.join("report.json") } else { p.clone() };
                        if !f.is_file() {
                            return Err(format!("{}: \"against\" is an earlier report (report.json or its folder)", p.display()));
                        }
                        args.extend(["--against".into(), f.display().to_string()]);
                    }
                    let output = match s("output") {
                        Some(o) => self.allowed(o, false)?,
                        None => Self::fresh(&self.out_dir.join("reports"), &format!("{}{}", stem(&model), adapter.as_ref().map_or(String::new(), |a| format!("-{}", stem(a)))), ""),
                    };
                    if output.join("report.json").exists() {
                        return Err(format!("{} already holds a report", output.display()));
                    }
                    args.extend(["--out".into(), output.display().to_string()]);
                    return Ok((kind.into(), args, output));
                }
                let data = self.dataset_path(s("data").ok_or("grade-train needs \"data\": the prompts, a .jsonl")?)?;
                args.extend(["--data".into(), data.display().to_string()]);
                let grader = s("grader").unwrap_or("code");
                if !["code", "judge"].contains(&grader) {
                    return Err("\"grader\" is code or judge".into());
                }
                args.extend(["--grader".into(), grader.into()]);
                for (k, flag, lo, hi, int) in [("samples", "--samples", 1.0, 64.0, true), ("rounds", "--rounds", 1.0, 20.0, true), ("temperature", "--temperature", 0.0, 2.0, false),
                                               ("max_tokens", "--max-tokens", 1.0, 32768.0, true), ("min_score", "--min-score", 1.0, 5.0, true), ("lr", "--lr", 1e-8, 1.0, false),
                                               ("epochs", "--epochs", 0.01, 100.0, false), ("max_steps", "--max-steps", 1.0, 10_000_000.0, true), ("rank", "--rank", 1.0, 512.0, true),
                                               ("train_layers", "--train-layers", 1.0, 1024.0, true), ("max_len", "--max-len", 16.0, 1_048_576.0, true)] {
                    if let Some(n) = num(k, lo, hi)? {
                        args.extend([flag.into(), if int { (n as u64).to_string() } else { n.to_string() }]);
                    }
                }
                let output = match s("output") {
                    Some(o) => self.allowed(o, false)?,
                    None => Self::fresh(&self.out_dir.join("grade-train"), &format!("{}-{}", stem(&model), data.file_stem().and_then(|x| x.to_str()).unwrap_or("prompts")), ""),
                };
                if output.join("rounds.jsonl").exists() {
                    return Err(format!("{} already holds a grade-and-train run", output.display()));
                }
                args.extend(["--out".into(), output.display().to_string()]);
                Ok((kind.into(), args, output))
            }
            other => Err(format!("unknown kind {other:?}: {}", KINDS.join(", "))),
        }
    }

    pub fn submit(self: &Arc<Self>, req: Value) -> Result<Job, String> {
        let (kind, args, output) = self.plan(&req)?;
        let exe = if kind == "strata" { PathBuf::new() } else {
            exe().ok_or("the am2strata-workshop command is not beside the server: build it (cargo build --release -p strata-workshop) or set AM2STRATA_WORKSHOP_EXE")?
        };
        let id = { let mut n = self.next.lock().unwrap(); let id = *n; *n += 1; id };
        let job = Job { id, kind, args, request: req, output, state: "queued", queued_at: now(), started_at: None,
                        finished_at: None, log: vec![], error: None, result: None, pid: None, progress: None, points: vec![] };
        self.jobs.lock().unwrap().push(job.clone());
        let me = self.clone();
        tokio::spawn(async move { me.run(id, exe).await });
        Ok(job)
    }

    fn update(&self, id: u64, f: impl FnOnce(&mut Job)) {
        if let Some(j) = self.jobs.lock().unwrap().iter_mut().find(|j| j.id == id) {
            f(j);
        }
        crate::live_channel::poke();
    }

    async fn run(self: Arc<Self>, id: u64, exe: PathBuf) {
        let _turn = self.gate.lock().await;
        let Some(job) = self.job(id) else { return };
        if job.state == "cancelled" {
            return;
        }
        if job.kind == "strata" {
            self.update(id, |j| { j.state = "running"; j.started_at = Some(now()); });
            let (models, packs, args) = (self.models_dir.clone(), self.packs_dir.clone(), job.args.clone());
            let r = tokio::task::spawn_blocking(move || -> Result<Value, String> {
                let gguf = PathBuf::from(&args[0]);
                let g = strata_hub::config::inspect(&gguf)?;
                let arch = g.architecture().unwrap_or("").to_string();
                let display = if args[2].is_empty() {
                    g.metadata.get("general.name").and_then(|v| v.as_str()).map(str::to_string).unwrap_or_else(|| args[1].clone())
                } else { args[2].clone() };
                let w = strata_hub::config::write(&models, &packs, &gguf, &g, None, None, &args[1], &display)?;
                Ok(json!({"output": w.config.display().to_string(), "kind": "run-config", "model": args[1], "architecture": arch,
                          "tokenizer": w.tokenizer.display().to_string(), "display_name": display}))
            }).await.unwrap_or_else(|e| Err(e.to_string()));
            self.update(id, |j| {
                j.finished_at = Some(now());
                match r {
                    Ok(v) => {
                        j.log.push(format!("{} is installed ({}): pick it in the model list to load it",
                                           v["model"].as_str().unwrap_or(""), v["architecture"].as_str().unwrap_or("")));
                        j.result = Some(v);
                        j.state = "done";
                    }
                    Err(e) => { j.error = Some(e); j.state = "failed"; }
                }
            });
            return;
        }
        // a training run on the GPUs takes them from the chat model, the way a media job does: it waits its turn in the
        // queue, the model is unloaded, chats wait until the run ends, and the model that was loaded comes back after
        // a judge on the GPUs this run takes would wait for the run while the run waits for it: refused at once
        if let Some(why) = self.judge_conflict(&job.request).await {
            self.update(id, |j| { j.state = "failed"; j.error = Some(why); j.finished_at = Some(now()); });
            return;
        }
        let mut lease = None;
        if matches!(job.kind.as_str(), "train" | "abliterate" | "regress" | "grade-train") && Self::train_on_gpu(&job.request) {
            let mgr = self.mgr.lock().unwrap().clone();
            if let Some(m) = mgr {
                let before = m.queue.resident();
                self.update(id, |j| j.log.push("waiting for the GPUs: the chat model is unloaded for the run and loaded again after".into()));
                match m.gpu_exclusive_needing(true, "train", &format!("{} #{id}", job.kind), 4.0).await {
                    Ok(pass) => lease = Some((m, pass, before)),
                    Err(e) => {
                        self.update(id, |j| { j.state = "failed"; j.error = Some(format!("{e:#}")); j.finished_at = Some(now()); });
                        return;
                    }
                }
                if self.job(id).is_some_and(|j| j.state == "cancelled") {
                    Self::give_back(lease.take());
                    return;
                }
            }
        }
        let mut cmd = tokio::process::Command::new(&exe);
        cmd.arg("--json").args(&job.args).stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::piped()).stderr(std::process::Stdio::piped()).kill_on_drop(true);
        // the server's address and key for a job that uses its API: in the environment, never on the command line
        cmd.env_remove("AM2STRATA_URL").env_remove("AM2STRATA_API_KEY");
        // datagen writes through the server's model; regress and grade-train ask it to judge
        if matches!(job.kind.as_str(), "datagen" | "regress" | "grade-train") {
            if let Some((url, key)) = self.server.lock().unwrap().clone() {
                cmd.env("AM2STRATA_URL", url);
                if let Some(k) = key {
                    cmd.env("AM2STRATA_API_KEY", k);
                }
            }
        }
        // its own process group, so cancelling stops everything the job started (not only the command)
        #[cfg(unix)]
        cmd.process_group(0);
        let mut child = match cmd.spawn() {
            Ok(c) => c,
            Err(e) => {
                self.update(id, |j| { j.state = "failed"; j.error = Some(format!("{}: {e}", exe.display())); j.finished_at = Some(now()); });
                Self::give_back(lease);
                return;
            }
        };
        self.update(id, |j| { j.state = "running"; j.started_at = Some(now()); j.pid = child.id(); });
        tracing::info!("workshop: job {id} ({}) started", job.kind);
        let stdout = child.stdout.take().unwrap();
        let stderr = child.stderr.take().unwrap();
        let me = self.clone();
        let errs = tokio::spawn(async move {
            use tokio::io::AsyncBufReadExt;
            let mut lines = tokio::io::BufReader::new(stderr).lines();
            let mut tail = vec![];
            while let Ok(Some(l)) = lines.next_line().await {
                tail.push(l);
                if tail.len() > 20 { tail.remove(0); }
            }
            tail
        });
        {
            use tokio::io::AsyncBufReadExt;
            let mut lines = tokio::io::BufReader::new(stdout).lines();
            while let Ok(Some(l)) = lines.next_line().await {
                let ev: Value = serde_json::from_str(&l).unwrap_or(json!({"event": "log", "message": l}));
                me.update(id, |j| match ev["event"].as_str() {
                    Some("start") => j.log.push(ev["what"].as_str().unwrap_or("").to_string()),
                    Some("log") => {
                        j.log.push(ev["message"].as_str().unwrap_or("").to_string());
                        if ev["eval_loss"].is_number() {
                            j.points.push(json!({"step": ev["step"], "eval_loss": ev["eval_loss"]}));
                        }
                    }
                    Some("done") => j.result = Some(ev.clone()),
                    // abliterate: the layer score table (kept in full for the form) and the refusal checks
                    Some("layers") => {
                        j.log.push(ev["message"].as_str().unwrap_or("").to_string());
                        j.points.push(json!({"layers": ev["layers"], "chosen": ev["chosen"]}));
                    }
                    Some("check") => {
                        let r = &ev["result"];
                        let pct = |k: &str| r[k].as_f64().map_or("?".to_string(), |x| format!("{:.0}%", x * 100.0));
                        j.log.push(format!("{} model: refused {} of the held-out harmful prompts, {} of the harmless{}", ev["which"].as_str().unwrap_or(""),
                                           pct("harmful_refusal_rate"), pct("harmless_refusal_rate"),
                                           r["harmless_loss"].as_f64().map(|l| format!(" (answer loss {l:.3})")).unwrap_or_default()));
                        j.points.push(json!({"check": ev["which"], "result": r}));
                    }
                    Some("progress") => {
                        if ev["loss"].is_number() {
                            j.points.push(json!({"step": ev["step"], "loss": ev["loss"], "lr": ev["lr"]}));
                            // a long run keeps every other point of its first half once it has many
                            if j.points.len() > 4000 {
                                let half = j.points.len() / 2;
                                let mut k = 0;
                                j.points.retain(|_| { k += 1; k > half || k % 2 == 0 });
                            }
                        }
                        if let Some(m) = ev["message"].as_str().filter(|m| !m.is_empty()) {
                            j.log.push(format!("[{}/{}] {m}", ev["done"], ev["total"]));
                        }
                        j.progress = Some(json!({"done": ev["done"], "total": ev["total"], "message": ev["message"]}));
                    }
                    Some("error") => j.error = ev["message"].as_str().map(str::to_string),
                    _ => j.log.push(l.clone()),
                });
            }
        }
        let status = child.wait().await;
        let tail = errs.await.unwrap_or_default();
        self.update(id, |j| {
            j.finished_at = Some(now());
            j.pid = None;
            if j.state == "cancelled" {
                return;
            }
            match status {
                Ok(s) if s.success() && j.error.is_none() => j.state = "done",
                Ok(s) => {
                    j.state = "failed";
                    if j.error.is_none() {
                        // a Rust panic: its message, not the "run with RUST_BACKTRACE" note after it
                        let why = tail.iter().position(|l| l.contains("panicked at"))
                            .map(|i| format!("the job crashed: {}", tail.get(i + 1).map(|m| m.trim()).filter(|m| !m.is_empty()).unwrap_or(&tail[i])))
                            .or_else(|| tail.iter().rev().find(|l| !l.trim().is_empty() && !l.starts_with("note:")).cloned())
                            .unwrap_or_else(|| format!("the job ended with {s}"));
                        j.error = Some(if s.code().is_none() { format!("stopped by the system ({s}; out of memory?): {why}") } else { why });
                    }
                }
                Err(e) => { j.state = "failed"; j.error = Some(e.to_string()); }
            }
        });
        tracing::info!("workshop: job {id} finished: {}", self.job(id).map(|j| j.state).unwrap_or("?"));
        Self::give_back(lease);
    }

    /// The GPUs back to chats after a training run: the turn ends, and the model that was loaded before is loaded again.
    fn give_back(lease: Option<(Arc<crate::manager::Manager>, crate::queue::Pass, crate::queue::Resident)>) {
        let Some((m, pass, before)) = lease else { return };
        drop(pass);
        if let crate::queue::Resident::Text(model) = before {
            tokio::spawn(async move {
                if let Err(e) = m.load(&model).await {
                    tracing::warn!("workshop: {model} could not be loaded again after the training run: {e:#}");
                }
            });
        }
    }

    pub fn job(&self, id: u64) -> Option<Job> {
        self.jobs.lock().unwrap().iter().find(|j| j.id == id).cloned()
    }

    pub fn list(&self) -> Vec<Job> {
        self.jobs.lock().unwrap().iter().rev().cloned().collect()
    }

    /// Cancel a queued or running job (the running command is stopped).
    pub fn cancel(&self, id: u64) -> Result<Job, String> {
        let mut jobs = self.jobs.lock().unwrap();
        let j = jobs.iter_mut().find(|j| j.id == id).ok_or("no such job")?;
        match j.state {
            "queued" | "running" => {
                if let Some(pid) = j.pid {
                    // the whole group: the command and whatever it started; SIGKILL if it is still there in 3 s
                    #[cfg(unix)]
                    {
                        unsafe { libc::kill(-(pid as i32), libc::SIGTERM); }
                        std::thread::spawn(move || {
                            std::thread::sleep(std::time::Duration::from_secs(3));
                            unsafe { libc::kill(-(pid as i32), libc::SIGKILL); }
                        });
                    }
                    #[cfg(not(unix))]
                    let _ = std::process::Command::new("taskkill").args(["/PID", &pid.to_string(), "/T", "/F"]).status();
                }
                j.state = "cancelled";
                j.finished_at = Some(now());
                Ok(j.clone())
            }
            s => Err(format!("the job is already {s}")),
        }
    }

    /// The models and adapters inside the Workshop's folders (two levels deep), for its pickers.
    pub fn files(&self) -> Value {
        let mut gguf = vec![];
        let mut hf = vec![];
        let mut adapters = vec![];
        fn walk(d: &Path, depth: usize, gguf: &mut Vec<Value>, hf: &mut Vec<Value>, adapters: &mut Vec<Value>) {
            let Ok(rd) = std::fs::read_dir(d) else { return };
            for e in rd.flatten() {
                let p = e.path();
                if p.is_dir() {
                    let modified = |p: &Path| p.metadata().ok().and_then(|m| m.modified().ok())
                        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok()).map_or(0, |d| d.as_secs());
                    if p.join("adapter_config.json").exists() {
                        adapters.push(json!({"path": p.display().to_string(), "modified": modified(&p.join("adapter_config.json"))}));
                    } else if p.join("config.json").exists() && std::fs::read_dir(&p).into_iter().flatten().flatten()
                        .any(|f| f.path().extension().and_then(|x| x.to_str()) == Some("safetensors")) {
                        // what the overview shows of a model folder: its architecture, a loop, a private mark
                        let c: Value = std::fs::read_to_string(p.join("config.json")).ok().and_then(|t| serde_json::from_str(&t).ok()).unwrap_or_default();
                        // the shape the Upcycle form counts parameters from
                        let dims = json!({"hidden": c["hidden_size"], "intermediate": c["intermediate_size"], "layers": c["num_hidden_layers"],
                                          "vocab": c["vocab_size"], "heads": c["num_attention_heads"], "kv_heads": c["num_key_value_heads"],
                                          "head_dim": c["head_dim"], "tied": c["tie_word_embeddings"] == true, "family": c["model_type"]});
                        hf.push(json!({"path": p.display().to_string(), "modified": modified(&p.join("config.json")),
                                       "arch": c["model_type"], "loop": c["prastara_loop"], "moe": c["prastara_moe"], "dims": dims,
                                       "private": c["prastara_private"] == true}));
                    } else if depth > 0 {
                        walk(&p, depth - 1, gguf, hf, adapters);
                    }
                } else if p.extension().and_then(|x| x.to_str()) == Some("gguf") {
                    let n = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
                    // shard 2+ of a split model is not a model of its own
                    if !n.contains("-of-") || n.contains("-00001-of-") {
                        let m = e.metadata().ok();
                        gguf.push(json!({"path": p.display().to_string(), "bytes": m.as_ref().map_or(0, |m| m.len()),
                                         "modified": m.and_then(|m| m.modified().ok()).and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok()).map_or(0, |d| d.as_secs())}));
                    }
                }
            }
        }
        for r in &self.roots {
            walk(r, 2, &mut gguf, &mut hf, &mut adapters);
        }
        let listing = |d: &Path, want: &dyn Fn(&Path) -> bool| -> Vec<Value> {
            let mut v: Vec<(u64, Value)> = std::fs::read_dir(d).into_iter().flatten().flatten().filter(|e| e.path().is_file() && want(&e.path()))
                .map(|e| {
                    let m = e.metadata().ok();
                    let t = m.as_ref().and_then(|m| m.modified().ok()).and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok()).map(|d| d.as_secs()).unwrap_or(0);
                    (t, json!({"path": e.path().display().to_string(), "bytes": m.map(|m| m.len()).unwrap_or(0), "modified": t}))
                }).collect();
            v.sort_by(|a, b| b.0.cmp(&a.0));
            v.into_iter().map(|x| x.1).collect()
        };
        let datasets = listing(&self.out_dir.join("datasets"), &|p| p.extension().and_then(|x| x.to_str()) == Some("jsonl"));
        let uploads = listing(&self.uploads_dir(), &|_| true);
        // prompt lists for abliteration: text files, one prompt per line (the Workshop's abliterate folder first)
        let txt = |p: &Path| p.extension().and_then(|x| x.to_str()) == Some("txt");
        let mut prompts = listing(&self.out_dir.join("abliterate"), &txt);
        prompts.extend(listing(&self.out_dir.join("datasets"), &txt));
        prompts.extend(listing(&self.uploads_dir(), &txt));
        // regression suites (the Workshop's suites folder, then datasets and uploads) and the reports written
        let js = |p: &Path| p.extension().and_then(|x| x.to_str()) == Some("json");
        let mut suites = listing(&self.out_dir.join("suites"), &js);
        suites.extend(listing(&self.out_dir.join("datasets"), &js));
        suites.extend(listing(&self.uploads_dir(), &js));
        let mut reports: Vec<(u64, Value)> = std::fs::read_dir(self.out_dir.join("reports")).into_iter().flatten().flatten()
            .filter(|e| e.path().join("report.json").is_file())
            .map(|e| {
                let t = e.path().join("report.json").metadata().ok().and_then(|m| m.modified().ok()).and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok()).map_or(0, |d| d.as_secs());
                (t, json!({"path": e.path().display().to_string(), "modified": t}))
            }).collect();
        reports.sort_by(|a, b| b.0.cmp(&a.0));
        let reports: Vec<Value> = reports.into_iter().map(|x| x.1).collect();
        json!({"gguf": gguf, "hf_models": hf, "adapters": adapters, "datasets": datasets, "prompts": prompts, "suites": suites, "reports": reports, "uploads": uploads, "folders": self.roots.iter().map(|r| r.display().to_string()).collect::<Vec<_>>(),
               "output_folder": self.out_dir.display().to_string(), "installed": exe().is_some()})
    }
}

/// Only the server's own key: the Workshop reads and writes model files.
fn admin(s: &AppState, headers: &HeaderMap) -> Result<Arc<Workshop>, Response> {
    // a client profile's key also passes `authorized`, so it is turned away here by name
    let valid = authorized(s, headers);
    let client_profile = crate::clients::given_key(headers).is_some_and(|k| s.clients.by_key(k).is_some());
    if !valid || client_profile {
        return Err(error(StatusCode::UNAUTHORIZED, "authentication_error", "the Workshop needs the server's own API key"));
    }
    get().ok_or_else(|| error(StatusCode::SERVICE_UNAVAILABLE, "server_error", "the server is still starting"))
}

pub async fn jobs_post(State(s): State<AppState>, headers: HeaderMap, Json(req): Json<Value>) -> Response {
    let w = match admin(&s, &headers) { Ok(w) => w, Err(r) => return r };
    if crate::foreign_origin(&s, &headers) {
        return crate::not_own_page("Workshop jobs can be started");
    }
    match w.submit(req) {
        Ok(j) => (StatusCode::ACCEPTED, Json(j.view(false))).into_response(),
        Err(e) => error(StatusCode::BAD_REQUEST, "invalid_request_error", e),
    }
}

pub async fn jobs_get(State(s): State<AppState>, headers: HeaderMap) -> Response {
    let w = match admin(&s, &headers) { Ok(w) => w, Err(r) => return r };
    Json(json!({"jobs": w.list().iter().map(|j| j.view(false)).collect::<Vec<_>>()})).into_response()
}

/// A job id from the URL, read only after the key was checked (so a stranger learns nothing from the route).
fn job_id(id: &str) -> Result<u64, Response> {
    id.parse().map_err(|_| error(StatusCode::NOT_FOUND, "not_found", "no such job"))
}

pub async fn job_get(State(s): State<AppState>, headers: HeaderMap, UrlPath(id): UrlPath<String>) -> Response {
    let w = match admin(&s, &headers) { Ok(w) => w, Err(r) => return r };
    let id = match job_id(&id) { Ok(i) => i, Err(r) => return r };
    match w.job(id) {
        Some(j) => Json(j.view(true)).into_response(),
        None => error(StatusCode::NOT_FOUND, "not_found", "no such job"),
    }
}

pub async fn job_delete(State(s): State<AppState>, headers: HeaderMap, UrlPath(id): UrlPath<String>) -> Response {
    let w = match admin(&s, &headers) { Ok(w) => w, Err(r) => return r };
    if crate::foreign_origin(&s, &headers) {
        return crate::not_own_page("Workshop jobs can be cancelled");
    }
    let id = match job_id(&id) { Ok(i) => i, Err(r) => return r };
    match w.cancel(id) {
        Ok(j) => Json(j.view(false)).into_response(),
        Err(e) => error(StatusCode::CONFLICT, "invalid_request_error", e),
    }
}

pub async fn files_get(State(s): State<AppState>, headers: HeaderMap) -> Response {
    let w = match admin(&s, &headers) { Ok(w) => w, Err(r) => return r };
    Json(w.files()).into_response()
}

/// Runs `am2strata-workshop --json library <args>` and returns its event (the library is compiled into that command).
async fn library_call(args: &[&str]) -> Result<Value, String> {
    let exe = exe().ok_or("the am2strata-workshop command is not installed beside the server")?;
    let o = tokio::process::Command::new(exe).arg("--json").arg("library").args(args).output().await
        .map_err(|e| format!("am2strata-workshop: {e}"))?;
    let text = String::from_utf8_lossy(&o.stdout);
    let events: Vec<Value> = text.lines().filter_map(|l| serde_json::from_str(l).ok()).collect();
    if let Some(e) = events.iter().find(|e| e["event"] == "error") {
        return Err(e["message"].as_str().unwrap_or("failed").to_string());
    }
    if !o.status.success() {
        return Err(String::from_utf8_lossy(&o.stderr).trim().to_string());
    }
    events.into_iter().find(|e| e["event"] == "library" || e["event"] == "done").ok_or_else(|| "no answer from the library".into())
}

/// `POST /v1/workshop/estimate` with a training request's settings: what the run needs against this machine
/// (`am2strata-workshop profile`): memory part by part and per GPU, whether it fits, a rough time, and its lines.
pub async fn estimate_post(State(s): State<AppState>, headers: HeaderMap, Json(req): Json<Value>) -> Response {
    let w = match admin(&s, &headers) { Ok(w) => w, Err(r) => return r };
    if crate::foreign_origin(&s, &headers) {
        return crate::not_own_page("training can be estimated");
    }
    if let Some(r) = library_missing() { return r; }
    let args = match w.train_args(&req) {
        Ok(a) => a,
        Err(e) => return error(StatusCode::BAD_REQUEST, "invalid_request_error", e),
    };
    let exe = exe().unwrap_or_default();
    let o = match tokio::process::Command::new(exe).arg("--json").arg("profile").args(&args).output().await {
        Ok(o) => o,
        Err(e) => return error(StatusCode::INTERNAL_SERVER_ERROR, "server_error", format!("am2strata-workshop: {e}")),
    };
    let text = String::from_utf8_lossy(&o.stdout);
    let events: Vec<Value> = text.lines().filter_map(|l| serde_json::from_str(l).ok()).collect();
    if let Some(e) = events.iter().find(|e| e["event"] == "error") {
        return error(StatusCode::BAD_REQUEST, "invalid_request_error", e["message"].as_str().unwrap_or("the estimate failed").to_string());
    }
    match events.into_iter().find(|e| e["event"] == "profile") {
        Some(v) => Json(v).into_response(),
        None => error(StatusCode::BAD_REQUEST, "invalid_request_error", String::from_utf8_lossy(&o.stderr).trim().to_string()),
    }
}

/// `GET /v1/workshop/library`: every recipe (Soup's catalogue, ported, and Prastara's own) with its abilities and
/// whether this Workshop trains it.
/// The Workshop command, when it is there to run: else the library answers 501 not_available (a feature this
/// install has not got, not a server error).
fn library_missing() -> Option<Response> {
    exe().filter(|p| p.exists()).is_none().then(|| error(StatusCode::NOT_IMPLEMENTED, "not_available",
        "the am2strata-workshop command is not installed beside the server (or AM2STRATA_WORKSHOP_EXE names no file)"))
}

pub async fn library_get(State(s): State<AppState>, headers: HeaderMap) -> Response {
    if let Err(r) = admin(&s, &headers) { return r; }
    if let Some(r) = library_missing() { return r; }
    match library_call(&["list"]).await {
        Ok(v) => Json(json!({"recipes": v["recipes"], "sources": v["sources"]})).into_response(),
        Err(e) => error(StatusCode::INTERNAL_SERVER_ERROR, "server_error", e),
    }
}

/// `POST /v1/workshop/library` `{"name"}`: the recipe's soup.yaml and DATA.md in `<data>/workshop/recipes/<name>`
/// (a new folder each time).
pub async fn library_post(State(s): State<AppState>, headers: HeaderMap, Json(req): Json<Value>) -> Response {
    let w = match admin(&s, &headers) { Ok(w) => w, Err(r) => return r };
    if crate::foreign_origin(&s, &headers) {
        return crate::not_own_page("recipes can be set up in the Workshop");
    }
    let Some(name) = req["name"].as_str().and_then(safe_name) else {
        return error(StatusCode::BAD_REQUEST, "invalid_request_error", "give the recipe's \"name\"");
    };
    if let Some(r) = library_missing() { return r; }
    let dir = Workshop::fresh(&w.out_dir.join("recipes"), &name, "");
    match library_call(&["init", &name, "--out", &dir.display().to_string()]).await {
        Ok(_) => Json(json!({"folder": dir.display().to_string(), "config": dir.join("soup.yaml").display().to_string(),
                             "guide": dir.join("DATA.md").display().to_string()})).into_response(),
        Err(e) => error(StatusCode::BAD_REQUEST, "invalid_request_error", e),
    }
}

/// A file name made safe: letters, digits, `.`, `-` and `_` only, no leading dot, at most 120 characters.
fn safe_name(n: &str) -> Option<String> {
    let base = n.rsplit(['/', '\\']).next().unwrap_or("");
    let s: String = base.chars().map(|c| if c.is_ascii_alphanumeric() || "._-".contains(c) { c } else { '_' }).collect();
    let s = s.trim_start_matches('.').chars().take(120).collect::<String>();
    (!s.is_empty() && s.chars().any(|c| c.is_ascii_alphanumeric())).then_some(s)
}

/// `POST /v1/workshop/uploads?name=…`: the body is the file; it is saved into the uploads folder (a new name when taken).
pub async fn upload_post(State(s): State<AppState>, headers: HeaderMap, axum::extract::Query(q): axum::extract::Query<std::collections::HashMap<String, String>>,
                         body: axum::body::Bytes) -> Response {
    let w = match admin(&s, &headers) { Ok(w) => w, Err(r) => return r };
    if crate::foreign_origin(&s, &headers) {
        return crate::not_own_page("files can be uploaded to the Workshop");
    }
    let Some(name) = q.get("name").and_then(|n| safe_name(n)) else {
        return error(StatusCode::BAD_REQUEST, "invalid_request_error", "give the file's name: ?name=notes.pdf");
    };
    if body.is_empty() {
        return error(StatusCode::BAD_REQUEST, "invalid_request_error", "the file is empty");
    }
    if let Some(why) = q.get("name").and_then(|raw| sensitive(std::path::Path::new(raw))).or_else(|| sensitive(std::path::Path::new(&name))) {
        return error(StatusCode::BAD_REQUEST, "invalid_request_error", format!("{name}: {why}; not accepted as an upload"));
    }
    let (stem, ext) = match name.rfind('.') {
        Some(i) if i > 0 => (name[..i].to_string(), name[i..].to_string()),
        _ => (name.clone(), String::new()),
    };
    let path = Workshop::fresh(&w.uploads_dir(), &stem, &ext);
    match tokio::fs::write(&path, &body).await {
        Ok(()) => Json(json!({"path": path.display().to_string(), "bytes": body.len()})).into_response(),
        Err(e) => error(StatusCode::INTERNAL_SERVER_ERROR, "server_error", format!("{}: {e}", path.display())),
    }
}

/// `POST /v1/workshop/upload-folder?name=…`: a new, empty folder in the uploads folder for a folder upload (a model
/// or an adapter folder from another device); its files then come one by one to `PUT /v1/workshop/upload`.
pub async fn upload_folder_post(State(s): State<AppState>, headers: HeaderMap,
                                axum::extract::Query(q): axum::extract::Query<std::collections::HashMap<String, String>>) -> Response {
    let w = match admin(&s, &headers) { Ok(w) => w, Err(r) => return r };
    if crate::foreign_origin(&s, &headers) {
        return crate::not_own_page("folders can be uploaded to the Workshop");
    }
    let Some(name) = q.get("name").and_then(|n| safe_name(n)) else {
        return error(StatusCode::BAD_REQUEST, "invalid_request_error", "give the folder's name: ?name=my-model");
    };
    if let Some(why) = q.get("name").and_then(|raw| sensitive(std::path::Path::new(raw))).or_else(|| sensitive(std::path::Path::new(&name))) {
        return error(StatusCode::BAD_REQUEST, "invalid_request_error", format!("{name}: {why}; not accepted as an upload"));
    }
    let dir = Workshop::fresh(&w.uploads_dir(), &name, "");
    match std::fs::create_dir_all(&dir) {
        Ok(()) => Json(json!({"folder": dir.display().to_string()})).into_response(),
        Err(e) => error(StatusCode::INTERNAL_SERVER_ERROR, "server_error", format!("{}: {e}", dir.display())),
    }
}

/// Where a streamed upload goes: a new file in the uploads folder (`?name=`), or `?folder=` (one made by
/// `upload-folder`) plus `?rel=` (the file's path inside the folder being uploaded: plain names, no "..").
fn upload_target(w: &Workshop, q: &std::collections::HashMap<String, String>) -> Result<PathBuf, String> {
    let uploads = w.uploads_dir();
    std::fs::create_dir_all(&uploads).map_err(|e| e.to_string())?;
    let uploads = uploads.canonicalize().map_err(|e| e.to_string())?;
    match (q.get("folder"), q.get("rel")) {
        (Some(folder), Some(rel)) => {
            let folder = std::path::Path::new(folder).canonicalize().map_err(|_| format!("{folder}: no such upload folder"))?;
            if !folder.starts_with(&uploads) || folder == uploads {
                return Err("an upload folder is one made by upload-folder, inside the uploads folder".into());
            }
            if rel.starts_with(['/', '\\']) || rel.contains(':') {
                return Err(format!("{rel}: a path inside the folder, not a full path"));
            }
            let mut p = folder;
            let parts: Vec<&str> = rel.split(['/', '\\']).filter(|x| !x.is_empty()).collect();
            if parts.is_empty() || parts.len() > 16 {
                return Err("give the file's path inside the folder: ?rel=sub/file.safetensors".into());
            }
            if let Some(why) = sensitive(std::path::Path::new(rel)) {
                return Err(format!("{rel}: {why}; not accepted as an upload"));
            }
            for part in parts {
                let safe = safe_name(part).ok_or_else(|| format!("{rel}: not a usable file name"))?;
                if safe != part {
                    return Err(format!("{rel}: use plain names (letters, digits, \".\", \"_\", \"-\")"));
                }
                p.push(part);
            }
            if p.exists() {
                return Err(format!("{} is already there", p.display()));
            }
            Ok(p)
        }
        (None, None) => {
            let raw = q.get("name").map(String::as_str).unwrap_or("");
            let name = safe_name(raw).ok_or("give the file's name: ?name=model.gguf")?;
            // the name as sent: the cleaning above drops a leading dot (".env" would pass as "env")
            if let Some(why) = sensitive(std::path::Path::new(raw)).or_else(|| sensitive(std::path::Path::new(&name))) {
                return Err(format!("{name}: {why}; not accepted as an upload"));
            }
            let (stem, ext) = match name.rfind('.') {
                Some(i) if i > 0 => (name[..i].to_string(), name[i..].to_string()),
                _ => (name.clone(), String::new()),
            };
            Ok(Workshop::fresh(&uploads, &stem, &ext))
        }
        _ => Err("a folder upload gives both ?folder= and ?rel=".into()),
    }
}

/// `PUT /v1/workshop/upload?name=…` (or `?folder=…&rel=…`): the body is the file, streamed to disk as it arrives, so
/// a model file of any size never sits in memory. It is written beside its name as `.part` and renamed when whole.
pub async fn upload_put(State(s): State<AppState>, headers: HeaderMap,
                        axum::extract::Query(q): axum::extract::Query<std::collections::HashMap<String, String>>,
                        body: axum::body::Body) -> Response {
    use futures::StreamExt;
    use tokio::io::AsyncWriteExt;
    let w = match admin(&s, &headers) { Ok(w) => w, Err(r) => return r };
    if crate::foreign_origin(&s, &headers) {
        return crate::not_own_page("files can be uploaded to the Workshop");
    }
    let path = match upload_target(&w, &q) {
        Ok(p) => p,
        Err(e) => return error(StatusCode::BAD_REQUEST, "invalid_request_error", e),
    };
    if let Some(d) = path.parent() {
        if let Err(e) = std::fs::create_dir_all(d) {
            return error(StatusCode::INTERNAL_SERVER_ERROR, "server_error", format!("{}: {e}", d.display()));
        }
    }
    let part = path.with_file_name(format!("{}.part", path.file_name().and_then(|n| n.to_str()).unwrap_or("upload")));
    let mut f = match tokio::fs::File::create(&part).await {
        Ok(f) => f,
        Err(e) => return error(StatusCode::INTERNAL_SERVER_ERROR, "server_error", format!("{}: {e}", part.display())),
    };
    let mut stream = body.into_data_stream();
    let mut bytes = 0u64;
    let fail = |m: String| { let _ = std::fs::remove_file(&part); error(StatusCode::INTERNAL_SERVER_ERROR, "server_error", m) };
    while let Some(chunk) = stream.next().await {
        let chunk = match chunk {
            Ok(c) => c,
            Err(e) => return fail(format!("the upload stopped: {e}")),
        };
        bytes += chunk.len() as u64;
        if let Err(e) = f.write_all(&chunk).await {
            return fail(format!("{}: {e}", part.display()));
        }
    }
    if let Err(e) = f.flush().await {
        return fail(format!("{}: {e}", part.display()));
    }
    drop(f);
    if bytes == 0 {
        let _ = std::fs::remove_file(&part);
        return error(StatusCode::BAD_REQUEST, "invalid_request_error", "the file is empty");
    }
    match std::fs::rename(&part, &path) {
        Ok(()) => Json(json!({"path": path.display().to_string(), "bytes": bytes})).into_response(),
        Err(e) => fail(format!("{}: {e}", path.display())),
    }
}

/// `GET /v1/workshop/browse[?path=…]`: the Workshop's folders, or one folder's entries (folders first), each marked
/// with what it is to the Workshop: a Hugging Face model folder, an adapter, a GGUF or a dataset.
pub async fn browse_get(State(s): State<AppState>, headers: HeaderMap,
                        axum::extract::Query(q): axum::extract::Query<std::collections::HashMap<String, String>>) -> Response {
    let w = match admin(&s, &headers) { Ok(w) => w, Err(r) => return r };
    let Some(p) = q.get("path").filter(|p| !p.is_empty()) else {
        // each root by its own name, with where it is
        let root = |r: &Path| json!({"name": r.file_name().map_or(r.display().to_string(), |n| n.to_string_lossy().to_string()),
                                     "path": r.display().to_string(), "dir": true, "where": r.parent().map(|p| p.display().to_string())});
        let roots: Vec<Value> = w.roots.iter().map(|r| root(r)).chain(std::iter::once(root(&w.uploads_dir()))).collect();
        return Json(json!({"path": null, "parent": null, "entries": roots})).into_response();
    };
    let dir = match w.allowed(p, true) {
        Ok(d) if d.is_dir() => d,
        Ok(d) => return error(StatusCode::BAD_REQUEST, "invalid_request_error", format!("{} is not a folder", d.display())),
        Err(e) => return error(StatusCode::BAD_REQUEST, "invalid_request_error", e),
    };
    let kind = |p: &Path, is_dir: bool| -> &'static str {
        if is_dir {
            if p.join("adapter_config.json").exists() { return "adapter"; }
            if p.join("config.json").exists() { return "hf-model"; }
            return "";
        }
        match p.extension().and_then(|x| x.to_str()) {
            Some("gguf") => "gguf",
            Some("jsonl") => "dataset",
            Some("safetensors") => "weights",
            _ => "",
        }
    };
    let mut entries: Vec<(bool, String, Value)> = std::fs::read_dir(&dir).into_iter().flatten().flatten().take(5000).filter_map(|e| {
        let name = e.file_name().to_string_lossy().to_string();
        if name.starts_with('.') || name.ends_with(".part") { return None; }
        let p = e.path();
        if sensitive(&p).is_some() { return None; }
        let m = e.metadata().ok();
        let is_dir = p.is_dir();
        let modified = m.as_ref().and_then(|m| m.modified().ok()).and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok()).map_or(0, |d| d.as_secs());
        Some((is_dir, name.to_lowercase(), json!({"name": name, "path": p.display().to_string(), "dir": is_dir, "kind": kind(&p, is_dir),
            "bytes": if is_dir { 0 } else { m.map_or(0, |m| m.len()) }, "modified": modified})))
    }).collect();
    entries.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
    // the parent, while it is still one of the Workshop's folders
    let parent = dir.parent().filter(|pp| w.allowed(&pp.display().to_string(), true).is_ok()).map(|pp| pp.display().to_string());
    Json(json!({"path": dir.display().to_string(), "parent": parent, "entries": entries.into_iter().map(|e| e.2).collect::<Vec<_>>()})).into_response()
}

impl Workshop {
    /// A dataset the Workshop may read: a .jsonl inside its folders.
    /// A regression suite: a .json inside the Workshop's folders whose tests' data files are inside them too (the
    /// command reads them; the API must not let a suite reach anywhere else).
    fn suite_path(&self, p: &str) -> Result<PathBuf, String> {
        if !p.ends_with(".json") {
            return Err("a suite is a .json file".into());
        }
        let path = self.allowed(p, true)?;
        let v: Value = std::fs::read_to_string(&path).ok().and_then(|t| serde_json::from_str(&t).ok())
            .ok_or_else(|| format!("{}: not a JSON suite", path.display()))?;
        let tests = v["tests"].as_array().filter(|t| !t.is_empty()).ok_or_else(|| format!("{}: the suite lists no \"tests\"", path.display()))?;
        let base = path.parent().unwrap_or(Path::new("/"));
        for t in tests {
            let d = t["data"].as_str().filter(|d| !d.is_empty()).ok_or_else(|| format!("{}: a test without a \"data\" file", path.display()))?;
            let full = if Path::new(d).is_absolute() { PathBuf::from(d) } else { base.join(d) };
            self.allowed(&full.display().to_string(), true).map_err(|e| format!("the suite's test {}: {e}", t["name"].as_str().unwrap_or("?")))?;
        }
        Ok(path)
    }

    fn dataset_path(&self, p: &str) -> Result<PathBuf, String> {
        if !p.ends_with(".jsonl") {
            return Err("a dataset is a .jsonl file".into());
        }
        self.allowed(p, true)
    }

    /// True while a job is writing `p`.
    fn busy(&self, p: &Path) -> bool {
        self.jobs.lock().unwrap().iter().any(|j| matches!(j.state, "queued" | "running") && j.output == p)
    }
}

/// `GET /v1/workshop/dataset?path=…&offset=0&limit=50`: a page of records, the total, and the dataset's card.
pub async fn dataset_get(State(s): State<AppState>, headers: HeaderMap,
                         axum::extract::Query(q): axum::extract::Query<std::collections::HashMap<String, String>>) -> Response {
    let w = match admin(&s, &headers) { Ok(w) => w, Err(r) => return r };
    let path = match w.dataset_path(q.get("path").map(String::as_str).unwrap_or("")) {
        Ok(p) => p,
        Err(e) => return error(StatusCode::BAD_REQUEST, "invalid_request_error", e),
    };
    let offset: usize = q.get("offset").and_then(|v| v.parse().ok()).unwrap_or(0);
    let limit: usize = q.get("limit").and_then(|v| v.parse().ok()).unwrap_or(50).clamp(1, 500);
    let text = match tokio::fs::read_to_string(&path).await {
        Ok(t) => t,
        Err(e) => return error(StatusCode::BAD_REQUEST, "invalid_request_error", format!("{}: {e}", path.display())),
    };
    let lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
    let records: Vec<Value> = lines.iter().enumerate().skip(offset).take(limit)
        .map(|(i, l)| json!({"i": i, "record": serde_json::from_str::<Value>(l).unwrap_or_else(|_| json!({"unparsable": l}))}))
        .collect();
    // the card beside it, as strata-workshop's datagen names it (`x.jsonl` → `x.card.md`)
    let card_path = path.with_file_name(format!("{}.card.md", path.file_stem().and_then(|x| x.to_str()).unwrap_or("dataset")));
    let card = tokio::fs::read_to_string(&card_path).await.ok();
    Json(json!({"path": path.display().to_string(), "total": lines.len(), "offset": offset, "records": records, "card": card,
                "writing": w.busy(&path)})).into_response()
}

/// `POST /v1/workshop/dataset` `{"path", "remove": [i, …], "replace": {"i": record, …}}`: edits the dataset in place
/// (written to a new file, then moved over the old one).
pub async fn dataset_post(State(s): State<AppState>, headers: HeaderMap, Json(req): Json<Value>) -> Response {
    let w = match admin(&s, &headers) { Ok(w) => w, Err(r) => return r };
    if crate::foreign_origin(&s, &headers) {
        return crate::not_own_page("datasets can be edited");
    }
    let bad = |m: String| error(StatusCode::BAD_REQUEST, "invalid_request_error", m);
    let path = match w.dataset_path(req["path"].as_str().unwrap_or("")) { Ok(p) => p, Err(e) => return bad(e) };
    if w.busy(&path) {
        return error(StatusCode::CONFLICT, "invalid_request_error", "a job is still writing this dataset");
    }
    let text = match tokio::fs::read_to_string(&path).await { Ok(t) => t, Err(e) => return bad(format!("{}: {e}", path.display())) };
    let mut lines: Vec<Option<String>> = text.lines().filter(|l| !l.trim().is_empty()).map(|l| Some(l.to_string())).collect();
    if let Some(m) = req["replace"].as_object() {
        for (k, rec) in m {
            let Some(i) = k.parse::<usize>().ok().filter(|i| *i < lines.len()) else { return bad(format!("no record {k}")) };
            let ok = rec["messages"].as_array().is_some_and(|a| !a.is_empty() && a.iter().all(|m| m["role"].is_string() && m["content"].is_string()));
            if !ok {
                return bad(format!("record {k}: a record is {{\"messages\": [{{\"role\", \"content\"}}, …]}}"));
            }
            lines[i] = Some(rec.to_string());
        }
    }
    for v in req["remove"].as_array().map(|a| a.as_slice()).unwrap_or(&[]) {
        let Some(i) = v.as_u64().map(|i| i as usize).filter(|i| *i < lines.len()) else { return bad(format!("no record {v}")) };
        lines[i] = None;
    }
    let kept: Vec<String> = lines.into_iter().flatten().collect();
    let tmp = path.with_extension("jsonl.tmp");
    let body = if kept.is_empty() { String::new() } else { kept.join("\n") + "\n" };
    if let Err(e) = tokio::fs::write(&tmp, body).await {
        return error(StatusCode::INTERNAL_SERVER_ERROR, "server_error", e.to_string());
    }
    if let Err(e) = tokio::fs::rename(&tmp, &path).await {
        tokio::fs::remove_file(&tmp).await.ok();
        return error(StatusCode::INTERNAL_SERVER_ERROR, "server_error", e.to_string());
    }
    Json(json!({"path": path.display().to_string(), "total": kept.len()})).into_response()
}

#[cfg(test)]
mod tests {
    use super::safe_name;

    #[test]
    fn upload_names_are_made_safe() {
        assert_eq!(safe_name("notes.pdf").as_deref(), Some("notes.pdf"));
        assert_eq!(safe_name("../../etc/passwd").as_deref(), Some("passwd"));
        assert_eq!(safe_name("C:\\x\\my file (1).md").as_deref(), Some("my_file__1_.md"));
        assert_eq!(safe_name(".bashrc").as_deref(), Some("bashrc"));
        assert_eq!(safe_name("..."), None);
        assert_eq!(safe_name(""), None);
    }
}
