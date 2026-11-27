//! Workshop jobs across the cluster (docs/CLUSTER-PLAN.md, part F "minimal" and "Workshop jobs across the cluster").
//!
//! - **The request.** `POST /v1/workshop/jobs` takes an optional `"node"`: a paired worker's name, `"any"` (the idlest
//!   machine that can run that kind: this one or a worker) or nothing (this machine, as before). `"keep_on_node": true`
//!   leaves the outputs on the worker.
//! - **The master** checks the request against its own Workshop first (the same `plan`, so the same refusals and the
//!   same default output), then lists a *mirror* job in its own jobs (the same id space, `"node"` on its view), and in
//!   the background:
//!   1. hashes every input (SHA-256, streamed; kept per path, size and time so a file is hashed once) and copies each
//!      to the worker over the pinned link, content-addressed: a hash the worker has is not sent again, a transfer
//!      cut off resumes at the byte the worker has, and the worker keeps a file only when its hash matches;
//!   2. sends the job with each input path replaced by a placeholder the worker turns into its own path (inside its
//!      Workshop folder) and each output by a fresh folder of the worker's;
//!   3. follows the worker's job every second (its state, log, progress points and result), and forwards a cancel;
//!   4. copies the outputs back into the path the master planned (resumable, checked by hash), unless
//!      `keep_on_node`.
//! - **The worker** answers these only for a master's key on its pinned link (`crate::is_master_request`): never the
//!   owner's key, never a client's. Its blobs live in `<workshop>/cluster/blobs` (read-only), the inputs of a job are
//!   hard links of them in `<workshop>/cluster/in`, its outputs go to `<workshop>/cluster/out/<tag>`.
//! - **Privacy.** A job whose inputs or outputs are private (`prastara_private` / `private: true` in a config.json or
//!   a dataset row, the `private` / `uncensored` / `abliterated` flags of a model config, a GGUF marked private, or
//!   anything under `Strata-data/private`) is routed only to a node the request names; `"any"` refuses it. Workshop
//!   jobs only ever go to the paired machines of this cluster, never to a remote provider.
//! - Files are streamed in 1 MiB chunks both ways: nothing is ever held whole in memory.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use axum::extract::{Path as UrlPath, Query, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::{json, Value};

use crate::cluster::Worker;
use crate::workshop::Workshop;
use crate::{error, AppState};

const CHUNK: usize = 1 << 20;
const TRIES: u32 = 6;

/// The kinds that run well on any machine's CPU; the rest want a GPU unless the request says `"devices": "cpu"`.
pub const CPU_KINDS: [&str; 8] = ["clean", "convert", "quantize", "datagen", "regress", "strata", "merge", "loop"];
/// The kinds a node can be sent at all (every Workshop kind).
pub const ROUTABLE: [&str; 13] = ["convert", "quantize", "merge", "loop", "upcycle", "export-moe", "abliterate", "train", "strata",
                                  "datagen", "clean", "regress", "grade-train"];

/// What `"node"` may say for this machine.
const HERE: [&str; 4] = ["", "local", "here", "this"];

#[derive(Clone, Copy, PartialEq, Debug)]
enum Role {
    /// one path (a file or a folder)
    In,
    /// a list of paths (or one)
    InList,
    /// a suite .json: its tests' data files go with it
    Suite,
    /// an output path (a file or a folder)
    Out,
}

/// The fields of each kind that name files: what is copied there, and what comes back.
fn fields(kind: &str) -> &'static [(&'static str, Role)] {
    use Role::*;
    match kind {
        "convert" | "quantize" | "loop" => &[("input", In), ("output", Out)],
        "merge" => &[("base", In), ("adapter", In), ("output", Out)],
        "upcycle" => &[("input", In), ("calib", In), ("output", Out)],
        "export-moe" => &[("input", In), ("adapter", In), ("output", Out)],
        "abliterate" => &[("input", In), ("harmful", In), ("harmless", In), ("output", Out)],
        "train" => &[("model", In), ("data", In), ("output", Out)],
        "strata" => &[("input", In)],
        "datagen" => &[("sources", InList), ("seeds", In), ("output", Out)],
        "clean" => &[("inputs", InList), ("exclude", InList), ("cap_only", InList), ("model", In), ("output", Out), ("report", Out)],
        "regress" => &[("model", In), ("adapter", In), ("suite", Suite), ("against", InList), ("engine_gguf", In), ("output", Out)],
        "grade-train" => &[("model", In), ("adapter", In), ("data", In), ("suite", Suite), ("engine_gguf", In), ("output", Out)],
        _ => &[],
    }
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

fn valid_sha(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase())
}

/// A plain file or folder name: no separators, not `.` or `..`, no control characters.
fn plain_name(n: &str) -> bool {
    !n.is_empty() && n.len() <= 255 && n != "." && n != ".." && !n.contains('/') && !n.contains('\\') && !n.chars().any(char::is_control)
}

/// A relative path of plain names only (a file inside a copied folder).
fn plain_rel(r: &str) -> bool {
    // each segment as written (Path::components would drop a "." in the middle)
    !r.is_empty() && r.len() <= 4096 && !r.contains('\\') && r.split('/').all(plain_name)
}

/// The SHA-256 of a file, read in chunks (blocking).
pub fn sha_file(p: &Path) -> std::io::Result<String> {
    use sha2::Digest;
    use std::io::Read;
    let mut f = std::fs::File::open(p)?;
    let mut h = sha2::Sha256::new();
    let mut buf = vec![0u8; CHUNK];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(hex(&h.finalize()))
}

/// Hashes already made: (path, size, modified ns) -> SHA-256. A file is hashed again only when it changed.
fn hashes() -> &'static Mutex<HashMap<(PathBuf, u64, i128), String>> {
    static H: OnceLock<Mutex<HashMap<(PathBuf, u64, i128), String>>> = OnceLock::new();
    H.get_or_init(Default::default)
}

fn hash_cached(p: &Path) -> Result<(String, u64), String> {
    let m = std::fs::metadata(p).map_err(|e| format!("{}: {e}", p.display()))?;
    let t = m.modified().ok().and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok()).map_or(0, |d| d.as_nanos() as i128);
    let key = (p.to_path_buf(), m.len(), t);
    if let Some(h) = hashes().lock().unwrap().get(&key) {
        return Ok((h.clone(), m.len()));
    }
    let h = sha_file(p).map_err(|e| format!("{}: {e}", p.display()))?;
    hashes().lock().unwrap().insert(key, h.clone());
    Ok((h, m.len()))
}

// ---------------------------------------------------------------------------------------------------------------
// Privacy

/// Why a path is private (None: it is not): under Strata-data/private, a config or run config marked so, a dataset
/// with a private row, a GGUF marked private. A folder is private when anything in it is.
pub fn private_reason(p: &Path) -> Option<String> {
    let s = p.display().to_string().replace('\\', "/").to_lowercase();
    if s.contains("strata-data/private") {
        return Some(format!("{} is under Strata-data/private", p.display()));
    }
    if p.is_dir() {
        let mut stack = vec![p.to_path_buf()];
        let mut seen = 0;
        while let Some(d) = stack.pop() {
            for e in std::fs::read_dir(&d).into_iter().flatten().flatten() {
                seen += 1;
                if seen > 200_000 {
                    return None;
                }
                let q = e.path();
                if q.is_dir() {
                    stack.push(q);
                } else if let Some(r) = file_private(&q) {
                    return Some(r);
                }
            }
        }
        return None;
    }
    file_private(p)
}

fn flags_private(v: &Value) -> Option<&'static str> {
    for k in ["prastara_private", "private", "uncensored", "abliterated", "prastara_abliterated"] {
        if v[k] == true {
            return Some(k);
        }
    }
    None
}

fn file_private(p: &Path) -> Option<String> {
    let ext = p.extension().and_then(|x| x.to_str()).unwrap_or("").to_ascii_lowercase();
    match ext.as_str() {
        "json" => {
            let m = std::fs::metadata(p).ok()?;
            if m.len() > 64 << 20 {
                return None;
            }
            let v: Value = serde_json::from_str(&std::fs::read_to_string(p).ok()?).ok()?;
            flags_private(&v).map(|k| format!("{} is marked \"{k}\"", p.display()))
        }
        "jsonl" => {
            use std::io::BufRead;
            let f = std::fs::File::open(p).ok()?;
            for (i, line) in std::io::BufReader::new(f).lines().map_while(Result::ok).enumerate() {
                // only a row that names the mark is parsed
                if !line.contains("private") {
                    continue;
                }
                if let Ok(v) = serde_json::from_str::<Value>(&line) {
                    if v["private"] == true || v["prastara_private"] == true {
                        return Some(format!("{} row {} is marked private", p.display(), i + 1));
                    }
                }
            }
            None
        }
        "gguf" => crate::workshop::gguf_marks(p).private.then(|| format!("{} is a GGUF marked private", p.display())),
        _ => None,
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Capabilities

fn gpu_setting() -> &'static Mutex<Option<String>> {
    static G: OnceLock<Mutex<Option<String>>> = OnceLock::new();
    G.get_or_init(Default::default)
}

/// The run config's `"cluster": {"workshop_gpu": "cuda" | "metal" | "none"}` (at start). Without it: cuda when
/// `nvidia-smi -L` lists a GPU (asked once), none on a Mac (its Workshop uses Metal only when built with
/// `--features metal`: say so with "metal").
pub fn init(rest: &std::collections::BTreeMap<String, Value>) {
    let v = rest.get("cluster").and_then(|c| c.get("workshop_gpu")).and_then(|g| g.as_str()).map(str::to_string);
    *gpu_setting().lock().unwrap() = v;
}

fn gpu() -> (bool, bool) {
    static AUTO: OnceLock<bool> = OnceLock::new();
    match gpu_setting().lock().unwrap().as_deref() {
        Some("cuda") => (true, false),
        Some("metal") => (false, true),
        Some(_) => (false, false),
        None if cfg!(target_os = "macos") => (false, false),
        None => (*AUTO.get_or_init(|| std::process::Command::new("nvidia-smi").arg("-L").output()
            .is_ok_and(|o| o.status.success() && String::from_utf8_lossy(&o.stdout).contains("GPU"))), false),
    }
}

/// Free bytes on the disk that holds `p`.
fn disk_free(p: &Path) -> Option<u64> {
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStrExt;
        let c = std::ffi::CString::new(p.as_os_str().as_bytes()).ok()?;
        let mut s: libc::statvfs = unsafe { std::mem::zeroed() };
        if unsafe { libc::statvfs(c.as_ptr(), &mut s) } != 0 {
            return None;
        }
        Some(s.f_bavail as u64 * s.f_frsize as u64)
    }
    #[cfg(not(unix))]
    {
        let _ = p;
        None
    }
}

/// Jobs this machine's own Workshop runs or waits on (not the mirrors of jobs on other machines).
fn active_here(ws: &Workshop) -> usize {
    ws.list().iter().filter(|j| matches!(j.state, "queued" | "running") && j.request["node"].is_null()).count()
}

/// This machine's Workshop as a node: what it can run, its GPU, free memory and disk, and how busy it is.
pub fn caps(s: &AppState, ws: &Workshop) -> Value {
    let installed = crate::workshop::exe().is_some_and(|p| p.is_file());
    let (cuda, metal) = gpu();
    json!({"node": s.cluster.node_name(), "installed": installed,
           "kinds": if installed { ROUTABLE.to_vec() } else { vec![] }, "cpu_kinds": CPU_KINDS,
           "gpu": {"cuda": cuda, "metal": metal},
           "ram_free_gb": (crate::mem::available_gb() * 10.0).round() / 10.0, "ram_total_gb": (crate::mem::total_gb() * 10.0).round() / 10.0,
           "disk_free_bytes": disk_free(&ws.out_dir()), "active": active_here(ws)})
}

/// Whether a node with these capabilities can take a job of `kind` (`gpu`: it asks for a GPU).
fn capable(c: &Value, kind: &str, gpu: bool) -> bool {
    c["installed"] == true && c["kinds"].as_array().is_some_and(|k| k.iter().any(|x| x == kind)) && (!gpu || c["gpu"]["cuda"] == true)
}

async fn worker_caps(w: &Worker) -> Result<Value, String> {
    let r = w.client.get(format!("{}/api/cluster/workshop/caps", w.base())).bearer_auth(w.auth_key())
        .timeout(Duration::from_secs(5)).send().await.map_err(|e| e.to_string())?;
    if !r.status().is_success() {
        return Err(format!("HTTP {} (a worker older than this does not take Workshop jobs)", r.status().as_u16()));
    }
    r.json().await.map_err(|e| e.to_string())
}

/// `GET /api/cluster/workshop` (the main key): this machine and each worker as Workshop nodes, for the "Run on" select.
pub async fn nodes_get(State(s): State<AppState>, headers: HeaderMap) -> Response {
    if !crate::authorized(&s, &headers) {
        return error(StatusCode::UNAUTHORIZED, "authentication_error", "a valid API key is required");
    }
    let Some(ws) = crate::workshop::get() else { return error(StatusCode::SERVICE_UNAVAILABLE, "server_error", "the server is still starting") };
    let reads = s.cluster.workers().into_iter().map(|w| async move {
        let (caps, err) = if w.is_ready() { match worker_caps(&w).await { Ok(c) => (Some(c), None), Err(e) => (None, Some(e)) } }
                          else { (None, Some("not answering".to_string())) };
        json!({"name": w.name, "ready": w.is_ready(), "encrypted": w.tls_url.is_some(), "caps": caps, "error": err})
    });
    let workers = futures::future::join_all(reads).await;
    Json(json!({"this": caps(&s, &ws), "workers": workers,
                "privacy": "a job with private inputs or outputs runs on another machine only when you pick that machine; \"any\" never sends it, and Workshop jobs never go to a remote provider"}))
        .into_response()
}

// ---------------------------------------------------------------------------------------------------------------
// The master

/// A request that asks for another machine (or "any").
pub fn routed(req: &Value) -> bool {
    match &req["node"] {
        Value::Null => false,
        Value::String(n) => !HERE.contains(&n.trim()),
        _ => true,
    }
}

/// A file to send: where it is here, its hash and size.
#[derive(Clone)]
struct Blob {
    path: PathBuf,
    sha: String,
    bytes: u64,
}

/// One input as the placeholder the worker turns into its own path, its files added to `blobs` (blocking).
fn describe(ws: &Workshop, p: &str, blobs: &mut Vec<Blob>) -> Result<Value, String> {
    let path = ws.allowed(p, true)?;
    let name = path.file_name().and_then(|n| n.to_str()).filter(|n| plain_name(n)).ok_or_else(|| format!("{p}: not a plain file or folder name"))?
        .to_string();
    if path.is_file() {
        let (sha, bytes) = hash_cached(&path)?;
        blobs.push(Blob { path, sha: sha.clone(), bytes });
        return Ok(json!({"$in": {"kind": "file", "name": name, "sha": sha, "bytes": bytes}}));
    }
    if !path.is_dir() {
        return Err(format!("{p}: not a file or folder"));
    }
    let mut files = vec![];
    let mut stack = vec![path.clone()];
    while let Some(d) = stack.pop() {
        let mut entries: Vec<PathBuf> = std::fs::read_dir(&d).map_err(|e| format!("{}: {e}", d.display()))?.flatten().map(|e| e.path()).collect();
        entries.sort();
        for e in entries {
            let rel = e.strip_prefix(&path).map_err(|_| format!("{}: outside {}", e.display(), path.display()))?.to_string_lossy().replace('\\', "/");
            if rel.ends_with(".cluster-part") || rel.ends_with(".part") {
                continue;
            }
            if !plain_rel(&rel) {
                return Err(format!("{}: a name that cannot be copied", e.display()));
            }
            // each file (a link followed) must be inside the Workshop's folders too, and never a key file
            let real = ws.allowed(&e.display().to_string(), true).map_err(|why| format!("in {}: {why}", path.display()))?;
            if real.is_dir() {
                stack.push(e);
                continue;
            }
            let (sha, bytes) = hash_cached(&real)?;
            blobs.push(Blob { path: real, sha: sha.clone(), bytes });
            files.push(json!({"rel": rel, "sha": sha, "bytes": bytes}));
            if files.len() > 100_000 {
                return Err(format!("{p}: more than 100,000 files"));
            }
        }
    }
    Ok(json!({"$in": {"kind": "dir", "name": name, "files": files}}))
}

/// A suite and its tests' data files: the suite travels as its JSON with each data path a placeholder.
fn describe_suite(ws: &Workshop, p: &str, blobs: &mut Vec<Blob>) -> Result<Value, String> {
    let path = ws.allowed(p, true)?;
    let mut v: Value = serde_json::from_str(&std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?)
        .map_err(|e| format!("{}: {e}", path.display()))?;
    let base = path.parent().unwrap_or(Path::new("/")).to_path_buf();
    for t in v["tests"].as_array_mut().into_iter().flatten() {
        if let Some(d) = t["data"].as_str().filter(|d| !d.is_empty()).map(str::to_string) {
            let full = if Path::new(&d).is_absolute() { PathBuf::from(&d) } else { base.join(&d) };
            t["data"] = describe(ws, &full.display().to_string(), blobs)?;
        }
    }
    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("suite.json").to_string();
    Ok(json!({"$in": {"kind": "json", "name": name, "value": v}}))
}

/// The paths a request names (inputs and outputs), resolved here, for the privacy check.
fn named_paths(ws: &Workshop, kind: &str, req: &Value, output: &Path) -> Vec<PathBuf> {
    let mut out = vec![output.to_path_buf()];
    for (k, role) in fields(kind) {
        let vals: Vec<&str> = match &req[*k] {
            Value::String(s) if !s.is_empty() => vec![s.as_str()],
            Value::Array(a) => a.iter().filter_map(|x| x.as_str()).collect(),
            _ => vec![],
        };
        for v in vals {
            let Ok(p) = ws.allowed(v, *role != Role::Out) else { continue };
            if *role == Role::Suite {
                // the suite's data files too
                if let Ok(s) = std::fs::read_to_string(&p).map(|t| serde_json::from_str::<Value>(&t).unwrap_or_default()) {
                    let base = p.parent().unwrap_or(Path::new("/")).to_path_buf();
                    for t in s["tests"].as_array().into_iter().flatten() {
                        if let Some(d) = t["data"].as_str() {
                            out.push(if Path::new(d).is_absolute() { PathBuf::from(d) } else { base.join(d) });
                        }
                    }
                }
            }
            out.push(p);
        }
    }
    out
}

/// Where each output of the job goes here: (field, the master's path).
fn outputs_here(ws: &Workshop, kind: &str, req: &Value, output: &Path) -> Result<Vec<(String, PathBuf)>, String> {
    let mut v = vec![];
    for (k, role) in fields(kind) {
        if *role != Role::Out {
            continue;
        }
        if *k == "output" {
            v.push(("output".to_string(), output.to_path_buf()));
        } else if let Some(p) = req[*k].as_str().filter(|p| !p.is_empty()) {
            v.push((k.to_string(), ws.allowed(p, false)?));
        }
    }
    Ok(v)
}

/// The request as the worker gets it: inputs as placeholders, outputs as names of fresh outputs there (blocking).
fn for_worker(ws: &Workshop, kind: &str, req: &Value, outs: &[(String, PathBuf)]) -> Result<(Value, Vec<Blob>), String> {
    let mut r = req.clone();
    if let Some(o) = r.as_object_mut() {
        o.remove("node");
        o.remove("keep_on_node");
    }
    let mut blobs = vec![];
    for (k, role) in fields(kind) {
        let v = r[*k].clone();
        let empty = v.is_null() || v.as_str() == Some("") || v.as_array().is_some_and(|a| a.is_empty());
        if empty || *role == Role::Out {
            continue;
        }
        r[*k] = match (role, v) {
            (Role::In, Value::String(p)) => describe(ws, &p, &mut blobs)?,
            (Role::Suite, Value::String(p)) => describe_suite(ws, &p, &mut blobs)?,
            (Role::InList, Value::String(p)) => describe(ws, &p, &mut blobs)?,
            (Role::InList, Value::Array(a)) => Value::Array(a.iter().map(|x| x.as_str().ok_or_else(|| format!("\"{k}\" is a list of paths"))
                .and_then(|p| describe(ws, p, &mut blobs))).collect::<Result<_, _>>()?),
            _ => return Err(format!("\"{k}\" is a path")),
        };
    }
    for (k, p) in outs {
        let name = p.file_name().and_then(|n| n.to_str()).filter(|n| plain_name(n)).ok_or_else(|| format!("{}: not a plain name", p.display()))?;
        r[k.as_str()] = json!({"$out": name});
    }
    Ok((r, blobs))
}

/// `POST /v1/workshop/jobs` with a `"node"` (workshop.rs hands it here): the job checked here, then sent to the node.
pub async fn submit(s: &AppState, ws: &Arc<Workshop>, mut req: Value) -> Response {
    let bad = |m: String| error(StatusCode::BAD_REQUEST, "invalid_request_error", m);
    let Some(asked) = req["node"].as_str().map(|n| n.trim().to_string()) else {
        return bad("\"node\" is a worker's name, \"any\", or left out for this machine".into());
    };
    if !matches!(req["keep_on_node"], Value::Null | Value::Bool(_)) {
        return bad("\"keep_on_node\" is true or false".into());
    }
    let keep = req["keep_on_node"] == true;
    let mut local = req.clone();
    if let Some(o) = local.as_object_mut() {
        o.remove("node");
        o.remove("keep_on_node");
    }
    // the same checks as a job here: the same refusals, and the output this machine would write
    let (kind, _, output) = match ws.plan(&local) {
        Ok(p) => p,
        Err(e) => return bad(e),
    };
    if !ROUTABLE.contains(&kind.as_str()) {
        return bad(format!("a {kind} job runs on this machine only"));
    }
    let gpu = Workshop::train_on_gpu(&local);
    let ws2 = ws.clone();
    let (k2, l2, o2) = (kind.clone(), local.clone(), output.clone());
    let private = tokio::task::spawn_blocking(move || named_paths(&ws2, &k2, &l2, &o2).iter().find_map(|p| private_reason(p)))
        .await.unwrap_or(None);
    // a datagen job writes through the chat API of the machine it runs on, which may send chats on to a remote
    // model: with private data it runs here only, even when a worker is named
    if kind == "datagen" {
        if let Some(why) = &private {
            return error(StatusCode::BAD_REQUEST, "private_data",
                         format!("{why}: a datagen job with private data runs on this machine only (its chats would go through the worker's own chat API)"));
        }
    }
    let worker: Arc<Worker> = if asked == "any" {
        if let Some(why) = private {
            return error(StatusCode::BAD_REQUEST, "private_data",
                         format!("{why}: a job with private data is sent to another machine only when you name it (\"node\": \"<worker>\"); \
                                  \"any\" never sends it, so leave out \"node\" to run it here"));
        }
        match pick_any(s, ws, &kind, &local, gpu).await {
            Ok(Some(w)) => w,
            Ok(None) => {
                // this machine is the idlest that can: an ordinary job here
                return match ws.submit(local) {
                    Ok(j) => {
                        let mut v = j.view(false);
                        v["picked"] = json!(format!("this machine ({})", s.cluster.node_name()));
                        (StatusCode::ACCEPTED, Json(v)).into_response()
                    }
                    Err(e) => bad(e),
                };
            }
            Err(e) => return error(StatusCode::SERVICE_UNAVAILABLE, "no_node", e),
        }
    } else {
        match s.cluster.worker_by_name(&asked) {
            Some(w) if w.is_ready() => w,
            Some(w) => return error(StatusCode::SERVICE_UNAVAILABLE, "node_unavailable",
                                    format!("worker {} has not answered for 15 s; the job was not sent", w.name)),
            None => return error(StatusCode::NOT_FOUND, "not_found",
                                 format!("no paired worker is named {asked:?} (the workers: {})",
                                         s.cluster.workers().iter().map(|w| w.name.clone()).collect::<Vec<_>>().join(", "))),
        }
    };
    let keep = keep || kind == "strata";
    let outs = match outputs_here(ws, &kind, &local, &output) {
        Ok(o) => o,
        Err(e) => return bad(e),
    };
    req["node"] = json!(worker.name);
    if asked == "any" {
        req["node_asked"] = json!("any");
    }
    let shown = if keep { PathBuf::from(format!("{}:", worker.name)) } else { output.clone() };
    let job = ws.add_mirror(&kind, req, shown);
    let id = job.id;
    tracing::info!("workshop: job {id} ({kind}) goes to worker {}", worker.name);
    let (ws2, w2) = (ws.clone(), worker.clone());
    tokio::spawn(async move {
        let r = run_mirror(&ws2, &w2, id, &kind, &local, outs, keep).await;
        if let Err(e) = r {
            mirror(&ws2, id, |j| {
                j.state = "failed";
                j.error = Some(e);
                j.finished_at = Some(crate::now());
                j.pid = None;
            });
        }
    });
    (StatusCode::ACCEPTED, Json(job.view(false))).into_response()
}

/// The idlest node that can run this kind: None for this machine, else the worker. Err: none can.
async fn pick_any(s: &AppState, ws: &Workshop, kind: &str, req: &Value, gpu: bool) -> Result<Option<Arc<Worker>>, String> {
    let score = |c: &Value| (c["active"].as_u64().unwrap_or(0), -(c["ram_free_gb"].as_f64().unwrap_or(0.0) * 10.0) as i64,
                             -((c["disk_free_bytes"].as_u64().unwrap_or(0) >> 30) as i64));
    let here = caps(s, ws);
    let mut best: Option<((u64, i64, i64), Option<Arc<Worker>>)> = capable(&here, kind, gpu).then(|| (score(&here), None));
    let named_model = (kind == "datagen").then(|| req["model"].as_str().filter(|m| !m.is_empty()).map(str::to_string)).flatten();
    let ready: Vec<Arc<Worker>> = s.cluster.workers().into_iter().filter(|w| w.is_ready()).collect();
    let caps_all = futures::future::join_all(ready.iter().map(|w| worker_caps(w))).await;
    let mut why = vec![];
    for (w, c) in ready.into_iter().zip(caps_all) {
        let c = match c {
            Ok(c) => c,
            Err(e) => { why.push(format!("{}: {e}", w.name)); continue; }
        };
        if !capable(&c, kind, gpu) {
            why.push(format!("{}: {}", w.name, if c["installed"] != true { "no am2strata-workshop there" } else if gpu { "no CUDA GPU" } else { "not this kind" }));
            continue;
        }
        if named_model.as_deref().is_some_and(|m| !w.has_model(m)) {
            why.push(format!("{}: does not have {}", w.name, named_model.as_deref().unwrap_or("")));
            continue;
        }
        let sc = score(&c);
        if best.as_ref().is_none_or(|(b, _)| sc < *b) {
            best = Some((sc, Some(w)));
        }
    }
    match best {
        Some((_, w)) => Ok(w),
        None => Err(format!("no machine in the cluster can run a {kind} job{} now (this machine: {}{}{})",
                            if gpu { " on a GPU" } else { "" },
                            if here["installed"] != true { "no am2strata-workshop" } else if gpu { "no CUDA GPU" } else { "cannot" },
                            if why.is_empty() { "" } else { "; " }, why.join("; "))),
    }
}

/// Change a mirror job; a cancelled one keeps its state (only log lines are added to it).
fn mirror(ws: &Workshop, id: u64, f: impl FnOnce(&mut crate::workshop::Job)) {
    ws.update(id, |j| {
        if j.state == "cancelled" {
            let (state, fin, err) = (j.state, j.finished_at, j.error.clone());
            f(j);
            j.state = state;
            j.finished_at = fin;
            j.error = err;
        } else {
            f(j);
        }
    });
}

fn cancelled(ws: &Workshop, id: u64) -> bool {
    ws.job(id).is_none_or(|j| j.state == "cancelled")
}

fn static_state(s: &str) -> &'static str {
    match s {
        "queued" => "queued",
        "running" => "running",
        "done" => "done",
        "cancelled" => "cancelled",
        _ => "failed",
    }
}

fn gb(b: u64) -> String {
    if b >= 1 << 30 { format!("{:.2} GB", b as f64 / (1u64 << 30) as f64) } else if b >= 1 << 20 { format!("{:.1} MB", b as f64 / (1u64 << 20) as f64) } else { format!("{b} bytes") }
}

/// The mirror's work: copy the inputs, send the job, follow it, copy the outputs back.
async fn run_mirror(ws: &Arc<Workshop>, w: &Arc<Worker>, id: u64, kind: &str, req: &Value, outs: Vec<(String, PathBuf)>, keep: bool)
                    -> Result<(), String> {
    let started = Instant::now();
    mirror(ws, id, |j| {
        j.state = "running";
        j.started_at = Some(crate::now());
        j.log.push(format!("on {} (over {}): hashing the inputs", w.name, if w.tls_url.is_some() { "the pinned TLS link" } else { "plain HTTP" }));
    });
    let (ws2, k2, r2, o2) = (ws.clone(), kind.to_string(), req.clone(), outs.clone());
    let (wreq, blobs) = tokio::task::spawn_blocking(move || for_worker(&ws2, &k2, &r2, &o2)).await.map_err(|e| e.to_string())??;
    // each hash once
    let mut seen = HashSet::new();
    let blobs: Vec<Blob> = blobs.into_iter().filter(|b| seen.insert(b.sha.clone())).collect();
    let total: u64 = blobs.iter().map(|b| b.bytes).sum();
    mirror(ws, id, |j| j.log.push(format!("{} input file{} ({}) to check on {}", blobs.len(), if blobs.len() == 1 { "" } else { "s" }, gb(total), w.name)));
    let done = Arc::new(AtomicU64::new(0));
    let mut sent_files = vec![];
    for b in &blobs {
        if cancelled(ws, id) {
            return Ok(());
        }
        let base = done.load(Ordering::Relaxed);
        let r = send_blob(ws, id, w, b, &done, total).await?;
        done.store(base + b.bytes, Ordering::Relaxed);
        let name = b.path.file_name().and_then(|n| n.to_str()).unwrap_or("").to_string();
        let line = if r.cached { format!("{name}: already on {} (sha256 {}), not sent", w.name, &b.sha[..12]) }
                   else if r.resumed_at > 0 { format!("{name}: {} sent, resumed at {} of {}", gb(r.sent), gb(r.resumed_at), gb(b.bytes)) }
                   else { format!("{name}: {} sent, checked by hash", gb(r.sent)) };
        mirror(ws, id, |j| j.log.push(line));
        sent_files.push(json!({"name": name, "sha256": b.sha, "bytes": b.bytes, "cached": r.cached, "resumed_at": r.resumed_at, "sent": r.sent}));
    }
    if cancelled(ws, id) {
        return Ok(());
    }
    let r = w.client.post(format!("{}/api/cluster/workshop/jobs", w.base())).bearer_auth(w.auth_key())
        .json(&json!({"request": wreq})).timeout(Duration::from_secs(60)).send().await.map_err(|e| format!("{} did not take the job: {e}", w.name))?;
    let status = r.status();
    let v: Value = r.json().await.unwrap_or(Value::Null);
    if !status.is_success() {
        return Err(format!("{} refused the job: {}", w.name, v["error"]["message"].as_str().unwrap_or("no reason given")));
    }
    let rid = v["id"].as_u64().ok_or_else(|| format!("{}'s answer had no job id", w.name))?;
    mirror(ws, id, |j| j.log.push(format!("{} took it as its job #{rid}", w.name)));
    // follow it
    let mut lines = 0usize;
    let mut misses = 0;
    let fin = loop {
        tokio::time::sleep(Duration::from_secs(1)).await;
        if cancelled(ws, id) {
            let r = w.client.delete(format!("{}/api/cluster/workshop/jobs/{rid}", w.base())).bearer_auth(w.auth_key())
                .timeout(Duration::from_secs(10)).send().await;
            let told = r.as_ref().is_ok_and(|r| r.status().is_success() || r.status() == StatusCode::CONFLICT);
            mirror(ws, id, |j| j.log.push(if told { format!("cancelled on {} too (its job #{rid})", w.name) }
                                          else { format!("the cancel did not reach {}: its job #{rid} may still run", w.name) }));
            return Ok(());
        }
        let r = w.client.get(format!("{}/api/cluster/workshop/jobs/{rid}", w.base())).bearer_auth(w.auth_key())
            .timeout(Duration::from_secs(10)).send().await;
        let v: Value = match r {
            Ok(r) if r.status().is_success() => r.json().await.unwrap_or(Value::Null),
            Ok(r) if r.status() == StatusCode::NOT_FOUND => return Err(format!("{} no longer knows its job #{rid} (restarted?)", w.name)),
            Ok(r) => return Err(format!("{} answered HTTP {} about its job #{rid}", w.name, r.status().as_u16())),
            Err(e) => {
                misses += 1;
                if misses > 120 {
                    return Err(format!("{} stopped answering about its job #{rid}: {e}", w.name));
                }
                continue;
            }
        };
        misses = 0;
        let log: Vec<String> = v["log"].as_array().map(|a| a.iter().map(|l| l.as_str().unwrap_or("").to_string()).collect()).unwrap_or_default();
        let fresh: Vec<String> = log.iter().skip(lines).map(|l| format!("{}: {l}", w.name)).collect();
        lines = lines.max(log.len());
        let st = static_state(v["state"].as_str().unwrap_or("failed"));
        let ended = !matches!(st, "queued" | "running");
        mirror(ws, id, |j| {
            j.log.extend(fresh);
            if !ended {
                j.state = st;
            }
            if v["progress"].is_object() {
                j.progress = Some(v["progress"].clone());
            }
            if let Some(p) = v["points"].as_array() {
                j.points = p.clone();
            }
        });
        if ended {
            break v;
        }
    };
    let st = static_state(fin["state"].as_str().unwrap_or("failed"));
    if st != "done" {
        let e = fin["error"].as_str().map(|e| format!("on {}: {e}", w.name)).unwrap_or_else(|| format!("its job on {} ended {st}", w.name));
        mirror(ws, id, |j| {
            j.state = st;
            j.error = Some(e);
            j.result = fin["result"].as_object().map(|_| fin["result"].clone());
            j.finished_at = Some(crate::now());
        });
        return Ok(());
    }
    let mut result = if fin["result"].is_object() { fin["result"].clone() } else { json!({}) };
    result["node"] = json!(w.name);
    result["remote_output"] = fin["output"].clone();
    let mut copied = vec![];
    if keep {
        mirror(ws, id, |j| j.log.push(format!("the output stays on {} (keep_on_node): {}", w.name, fin["output"].as_str().unwrap_or(""))));
    } else {
        let m = w.client.get(format!("{}/api/cluster/workshop/jobs/{rid}/outputs", w.base())).bearer_auth(w.auth_key())
            .timeout(Duration::from_secs(1800)).send().await.map_err(|e| format!("{}: its outputs could not be listed: {e}", w.name))?;
        if !m.status().is_success() {
            return Err(format!("{}: its outputs could not be listed (HTTP {})", w.name, m.status().as_u16()));
        }
        let m: Value = m.json().await.map_err(|e| e.to_string())?;
        let list = m["outputs"].as_array().cloned().unwrap_or_default();
        let total: u64 = list.iter().flat_map(|o| o["files"].as_array().cloned().unwrap_or_default()).map(|f| f["bytes"].as_u64().unwrap_or(0)).sum();
        mirror(ws, id, |j| j.log.push(format!("copying the output back from {} ({})", w.name, gb(total))));
        let done = Arc::new(AtomicU64::new(0));
        for o in &list {
            let field = o["field"].as_str().unwrap_or("");
            let Some((_, here)) = outs.iter().find(|(k, _)| k == field) else { continue };
            for f in o["files"].as_array().into_iter().flatten() {
                if cancelled(ws, id) {
                    return Ok(());
                }
                let rel = f["rel"].as_str().unwrap_or("");
                let (Some(sha), Some(bytes)) = (f["sha256"].as_str().filter(|s| valid_sha(s)), f["bytes"].as_u64()) else {
                    return Err(format!("{}: an output file without its hash", w.name));
                };
                if !rel.is_empty() && !plain_rel(rel) {
                    return Err(format!("{}: an output named {rel:?} is not taken", w.name));
                }
                let target = if rel.is_empty() { here.clone() } else { here.join(rel) };
                // inside this machine's Workshop folders, as every output here
                let target = ws.allowed(&target.display().to_string(), false)?;
                let base = done.load(Ordering::Relaxed);
                let p = pull_file(ws, id, w, rid, field, rel, sha, bytes, &target, &done, total).await?;
                done.store(base + bytes, Ordering::Relaxed);
                copied.push(json!({"path": target.display().to_string(), "sha256": sha, "bytes": bytes, "cached": p.cached, "resumed_at": p.resumed_at}));
            }
        }
        if let Some((_, o)) = outs.iter().find(|(k, _)| k == "output") {
            result["output"] = json!(o.display().to_string());
        }
        mirror(ws, id, |j| j.log.push(format!("{} file{} back here, each checked by its SHA-256", copied.len(), if copied.len() == 1 { "" } else { "s" })));
    }
    result["transfer"] = json!({"inputs": sent_files, "outputs": copied, "seconds": started.elapsed().as_secs()});
    mirror(ws, id, |j| {
        j.result = Some(result);
        j.state = "done";
        j.progress = None;
        j.finished_at = Some(crate::now());
    });
    Ok(())
}

struct Sent {
    cached: bool,
    resumed_at: u64,
    sent: u64,
}

/// Progress of a copy on the mirror, at most twice a second.
fn progress(ws: &Workshop, id: u64, done: u64, total: u64, what: &str, last: &Mutex<Instant>) {
    let mut l = last.lock().unwrap();
    if l.elapsed() < Duration::from_millis(500) && done < total {
        return;
    }
    *l = Instant::now();
    drop(l);
    let msg = format!("{what} ({} of {})", gb(done), gb(total));
    mirror(ws, id, |j| j.progress = Some(json!({"done": done, "total": total.max(1), "message": msg})));
}

/// Send one file to the worker unless it has that hash, resuming where its copy stopped.
async fn send_blob(ws: &Arc<Workshop>, id: u64, w: &Worker, b: &Blob, done: &Arc<AtomicU64>, total: u64) -> Result<Sent, String> {
    let mut first = None;
    let mut sent = 0u64;
    let base = done.load(Ordering::Relaxed);
    for attempt in 0..TRIES {
        if attempt > 0 {
            tokio::time::sleep(Duration::from_secs(attempt as u64)).await;
        }
        let st = w.client.get(format!("{}/api/cluster/files/{}", w.base(), b.sha)).bearer_auth(w.auth_key())
            .timeout(Duration::from_secs(10)).send().await;
        let st: Value = match st {
            Ok(r) if r.status().is_success() => r.json().await.unwrap_or(Value::Null),
            Ok(r) => return Err(format!("{} answered HTTP {} for a file's state", w.name, r.status().as_u16())),
            Err(_) => continue,
        };
        if st["have"] == true {
            return Ok(Sent { cached: first.is_none() && sent == 0, resumed_at: first.unwrap_or(0), sent });
        }
        let offset = st["offset"].as_u64().unwrap_or(0).min(b.bytes);
        first.get_or_insert(offset);
        let mut f = tokio::fs::File::open(&b.path).await.map_err(|e| format!("{}: {e}", b.path.display()))?;
        use tokio::io::AsyncSeekExt;
        f.seek(std::io::SeekFrom::Start(offset)).await.map_err(|e| e.to_string())?;
        let counter = Arc::new(AtomicU64::new(0));
        let (c2, ws2, d2) = (counter.clone(), ws.clone(), done.clone());
        let last = Arc::new(Mutex::new(Instant::now()));
        let what = format!("copying the inputs to {}", w.name);
        let stream = futures::stream::unfold(f, move |mut f| {
            let (c2, ws2, d2, last, what) = (c2.clone(), ws2.clone(), d2.clone(), last.clone(), what.clone());
            async move {
                use tokio::io::AsyncReadExt;
                let mut buf = vec![0u8; CHUNK];
                match f.read(&mut buf).await {
                    Ok(0) => None,
                    Ok(n) => {
                        buf.truncate(n);
                        let c = c2.fetch_add(n as u64, Ordering::Relaxed) + n as u64;
                        progress(&ws2, id, (base + offset + c).min(total), total, &what, &last);
                        let _ = &d2;
                        Some((Ok::<_, std::io::Error>(axum::body::Bytes::from(buf)), f))
                    }
                    Err(e) => Some((Err(e), f)),
                }
            }
        });
        let r = w.client.put(format!("{}/api/cluster/files/{}", w.base(), b.sha)).bearer_auth(w.auth_key())
            .query(&[("offset", offset.to_string()), ("size", b.bytes.to_string())])
            .header("content-type", "application/octet-stream").body(reqwest::Body::wrap_stream(stream)).send().await;
        sent += counter.load(Ordering::Relaxed);
        match r {
            Ok(r) if r.status().is_success() => {
                let v: Value = r.json().await.unwrap_or(Value::Null);
                if v["have"] == true {
                    return Ok(Sent { cached: false, resumed_at: first.unwrap_or(0), sent });
                }
            }
            Ok(r) if r.status() == StatusCode::CONFLICT => {}
            Ok(r) => {
                let code = r.status().as_u16();
                let v: Value = r.json().await.unwrap_or(Value::Null);
                return Err(format!("{} did not keep {}: {}", w.name, b.path.display(), v["error"]["message"].as_str().unwrap_or(&format!("HTTP {code}"))));
            }
            Err(e) => tracing::warn!("workshop: copying {} to {} stopped ({e}); resuming", b.path.display(), w.name),
        }
    }
    Err(format!("{} could not be copied to {} after {TRIES} tries", b.path.display(), w.name))
}

struct Pulled {
    cached: bool,
    resumed_at: u64,
}

/// Copy one output file back from the worker into `target`, resuming a cut copy, kept only when its hash matches.
#[allow(clippy::too_many_arguments)]
async fn pull_file(ws: &Arc<Workshop>, id: u64, w: &Worker, rid: u64, field: &str, rel: &str, sha: &str, bytes: u64, target: &Path,
                   done: &Arc<AtomicU64>, total: u64) -> Result<Pulled, String> {
    use tokio::io::AsyncWriteExt;
    let t2 = target.to_path_buf();
    if target.is_file() && std::fs::metadata(target).is_ok_and(|m| m.len() == bytes)
        && tokio::task::spawn_blocking(move || hash_cached(&t2)).await.ok().and_then(Result::ok).is_some_and(|(h, _)| h == sha) {
        return Ok(Pulled { cached: true, resumed_at: 0 });
    }
    if let Some(d) = target.parent() {
        std::fs::create_dir_all(d).map_err(|e| format!("{}: {e}", d.display()))?;
    }
    let part = target.with_file_name(format!("{}.cluster-part", target.file_name().and_then(|n| n.to_str()).unwrap_or("out")));
    let base = done.load(Ordering::Relaxed);
    let last = Mutex::new(Instant::now());
    let mut first = None;
    for attempt in 0..TRIES {
        if attempt > 0 {
            tokio::time::sleep(Duration::from_secs(attempt as u64)).await;
        }
        let mut have = std::fs::metadata(&part).map(|m| m.len()).unwrap_or(0);
        if have > bytes {
            let _ = std::fs::remove_file(&part);
            have = 0;
        }
        first.get_or_insert(have);
        if have < bytes {
            let r = w.client.get(format!("{}/api/cluster/workshop/jobs/{rid}/file", w.base())).bearer_auth(w.auth_key())
                .query(&[("field", field.to_string()), ("rel", rel.to_string()), ("offset", have.to_string())]).send().await;
            let r = match r {
                Ok(r) if r.status().is_success() => r,
                Ok(r) => return Err(format!("{} answered HTTP {} for an output file", w.name, r.status().as_u16())),
                Err(_) => continue,
            };
            let mut f = tokio::fs::OpenOptions::new().create(true).append(true).open(&part).await.map_err(|e| format!("{}: {e}", part.display()))?;
            let mut stream = r.bytes_stream();
            use futures::StreamExt;
            let mut cut = false;
            while let Some(c) = stream.next().await {
                match c {
                    Ok(c) => {
                        f.write_all(&c).await.map_err(|e| format!("{}: {e}", part.display()))?;
                        have += c.len() as u64;
                        progress(ws, id, (base + have).min(total), total, &format!("copying the output back from {}", w.name), &last);
                        if cancelled(ws, id) {
                            return Ok(Pulled { cached: false, resumed_at: 0 });
                        }
                    }
                    Err(_) => { cut = true; break; }
                }
            }
            f.flush().await.map_err(|e| e.to_string())?;
            drop(f);
            if cut || have < bytes {
                continue;
            }
        }
        let p2 = part.clone();
        let got = tokio::task::spawn_blocking(move || sha_file(&p2)).await.map_err(|e| e.to_string())?.map_err(|e| e.to_string())?;
        if got != sha {
            let _ = std::fs::remove_file(&part);
            return Err(format!("{}: the copy from {} does not match its hash ({} not {})", target.display(), w.name, &got[..12], &sha[..12]));
        }
        std::fs::rename(&part, target).map_err(|e| format!("{}: {e}", target.display()))?;
        return Ok(Pulled { cached: false, resumed_at: first.unwrap_or(0) });
    }
    Err(format!("{} could not be copied back from {} after {TRIES} tries", target.display(), w.name))
}

// ---------------------------------------------------------------------------------------------------------------
// The worker

/// The Workshop, for a request from a master (its key on its pinned link) only.
fn from_master(s: &AppState, headers: &HeaderMap) -> Result<Arc<Workshop>, Response> {
    if !crate::is_master_request(s, headers) {
        return Err(error(StatusCode::UNAUTHORIZED, "authentication_error", "only a paired cluster master's key, on its own link"));
    }
    crate::workshop::get().ok_or_else(|| error(StatusCode::SERVICE_UNAVAILABLE, "server_error", "the server is still starting"))
}

fn store(ws: &Workshop) -> PathBuf {
    ws.out_dir().join("cluster")
}

fn blob_path(ws: &Workshop, sha: &str) -> PathBuf {
    store(ws).join("blobs").join(sha)
}

fn part_path(ws: &Workshop, sha: &str) -> PathBuf {
    store(ws).join("partial").join(format!("{sha}.part"))
}

/// The jobs masters sent this worker: its job id -> the outputs (field, path).
fn routed_jobs() -> &'static Mutex<HashMap<u64, Vec<(String, PathBuf)>>> {
    static R: OnceLock<Mutex<HashMap<u64, Vec<(String, PathBuf)>>>> = OnceLock::new();
    R.get_or_init(Default::default)
}

/// `GET /api/cluster/workshop/caps` (a master): this worker as a Workshop node.
pub async fn caps_get(State(s): State<AppState>, headers: HeaderMap) -> Response {
    let ws = match from_master(&s, &headers) { Ok(w) => w, Err(r) => return r };
    Json(caps(&s, &ws)).into_response()
}

/// `GET /api/cluster/files/{sha}` (a master): whether this worker has the file of that hash, else how much of it.
pub async fn blob_get(State(s): State<AppState>, headers: HeaderMap, UrlPath(sha): UrlPath<String>) -> Response {
    let ws = match from_master(&s, &headers) { Ok(w) => w, Err(r) => return r };
    if !valid_sha(&sha) {
        return error(StatusCode::BAD_REQUEST, "invalid_request_error", "a SHA-256 in lower-case hex");
    }
    let have = blob_path(&ws, &sha).is_file();
    let offset = if have { 0 } else { std::fs::metadata(part_path(&ws, &sha)).map(|m| m.len()).unwrap_or(0) };
    Json(json!({"have": have, "offset": offset})).into_response()
}

/// Files being received now (one sender per hash).
fn receiving() -> &'static Mutex<HashSet<String>> {
    static R: OnceLock<Mutex<HashSet<String>>> = OnceLock::new();
    R.get_or_init(Default::default)
}

struct Receiving(String);

impl Drop for Receiving {
    fn drop(&mut self) {
        receiving().lock().unwrap().remove(&self.0);
    }
}

/// `PUT /api/cluster/files/{sha}?offset=N&size=S` (a master): the file's bytes from `offset`, streamed to disk. A cut
/// stream keeps what came (the next PUT resumes there); the file is kept only when all of it hashes to `sha`.
pub async fn blob_put(State(s): State<AppState>, headers: HeaderMap, UrlPath(sha): UrlPath<String>,
                      Query(q): Query<HashMap<String, String>>, body: axum::body::Body) -> Response {
    use futures::StreamExt;
    use tokio::io::AsyncWriteExt;
    let ws = match from_master(&s, &headers) { Ok(w) => w, Err(r) => return r };
    let bad = |m: &str| error(StatusCode::BAD_REQUEST, "invalid_request_error", m.to_string());
    if !valid_sha(&sha) {
        return bad("a SHA-256 in lower-case hex");
    }
    let (Some(offset), Some(size)) = (q.get("offset").and_then(|v| v.parse::<u64>().ok()), q.get("size").and_then(|v| v.parse::<u64>().ok())) else {
        return bad("?offset=N&size=S");
    };
    let blob = blob_path(&ws, &sha);
    if blob.is_file() {
        return Json(json!({"have": true, "offset": 0})).into_response();
    }
    let part = part_path(&ws, &sha);
    for d in [blob.parent(), part.parent()].into_iter().flatten() {
        if let Err(e) = std::fs::create_dir_all(d) {
            return error(StatusCode::INTERNAL_SERVER_ERROR, "server_error", format!("{}: {e}", d.display()));
        }
    }
    if !receiving().lock().unwrap().insert(sha.clone()) {
        return (StatusCode::CONFLICT, Json(json!({"error": {"type": "busy", "message": "this file is being received already"}, "have": false}))).into_response();
    }
    let _held = Receiving(sha.clone());
    let cur = std::fs::metadata(&part).map(|m| m.len()).unwrap_or(0);
    if offset != cur {
        return (StatusCode::CONFLICT, Json(json!({"error": {"type": "offset", "message": format!("this worker has {cur} bytes of it")},
                                                  "have": false, "offset": cur}))).into_response();
    }
    if size < cur {
        let _ = std::fs::remove_file(&part);
        return bad("size is less than what was sent");
    }
    if disk_free(&part).is_some_and(|f| f < size - cur + (256 << 20)) {
        return error(StatusCode::INSUFFICIENT_STORAGE, "no_space", format!("this worker has too little free disk for {} more", gb(size - cur)));
    }
    let mut f = match tokio::fs::OpenOptions::new().create(true).append(true).open(&part).await {
        Ok(f) => f,
        Err(e) => return error(StatusCode::INTERNAL_SERVER_ERROR, "server_error", format!("{}: {e}", part.display())),
    };
    let mut got = cur;
    let mut stream = body.into_data_stream();
    while let Some(c) = stream.next().await {
        let Ok(c) = c else { break }; // cut off: what came is kept
        got += c.len() as u64;
        if got > size {
            drop(f);
            let _ = std::fs::remove_file(&part);
            return bad("more bytes than its size");
        }
        if let Err(e) = f.write_all(&c).await {
            return error(StatusCode::INTERNAL_SERVER_ERROR, "server_error", format!("{}: {e}", part.display()));
        }
    }
    let _ = f.flush().await;
    drop(f);
    if got < size {
        return Json(json!({"have": false, "offset": got})).into_response();
    }
    let p2 = part.clone();
    let h = tokio::task::spawn_blocking(move || sha_file(&p2)).await.ok().and_then(Result::ok);
    if h.as_deref() != Some(sha.as_str()) {
        let _ = std::fs::remove_file(&part);
        return error(StatusCode::UNPROCESSABLE_ENTITY, "hash_mismatch", "what came does not hash to its SHA-256: not kept (send it again)");
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(&part, std::fs::Permissions::from_mode(0o444));
    }
    if let Err(e) = std::fs::rename(&part, &blob) {
        return error(StatusCode::INTERNAL_SERVER_ERROR, "server_error", format!("{}: {e}", blob.display()));
    }
    Json(json!({"have": true, "offset": 0})).into_response()
}

/// A blob placed at `dest` (a hard link, else a copy).
fn place(ws: &Workshop, sha: &str, dest: &Path) -> Result<(), String> {
    let blob = blob_path(ws, sha);
    if !blob.is_file() {
        return Err(format!("missing:{sha}"));
    }
    if dest.exists() {
        return Ok(());
    }
    if let Some(d) = dest.parent() {
        std::fs::create_dir_all(d).map_err(|e| format!("{}: {e}", d.display()))?;
    }
    if std::fs::hard_link(&blob, dest).is_err() {
        std::fs::copy(&blob, dest).map_err(|e| format!("{}: {e}", dest.display()))?;
    }
    Ok(())
}

/// The placeholders of a master's request made into this worker's paths (blocking).
fn materialize(ws: &Workshop, v: &mut Value) -> Result<(), String> {
    match v {
        Value::Array(a) => a.iter_mut().try_for_each(|x| materialize(ws, x)),
        Value::Object(o) if o.contains_key("$in") => {
            let d = o["$in"].clone();
            let name = d["name"].as_str().filter(|n| plain_name(n)).ok_or("an input without a plain name")?.to_string();
            let path = match d["kind"].as_str() {
                Some("file") => {
                    let sha = d["sha"].as_str().filter(|s| valid_sha(s)).ok_or("an input without its hash")?;
                    let p = store(ws).join("in").join(&sha[..16]).join(&name);
                    place(ws, sha, &p)?;
                    p
                }
                Some("dir") => {
                    use sha2::Digest;
                    let files = d["files"].as_array().cloned().unwrap_or_default();
                    let mut h = sha2::Sha256::new();
                    for f in &files {
                        h.update(format!("{}\0{}\n", f["rel"].as_str().unwrap_or(""), f["sha"].as_str().unwrap_or("")).as_bytes());
                    }
                    let root = store(ws).join("in").join(&hex(&h.finalize())[..16]).join(&name);
                    std::fs::create_dir_all(&root).map_err(|e| format!("{}: {e}", root.display()))?;
                    for f in &files {
                        let rel = f["rel"].as_str().filter(|r| plain_rel(r)).ok_or("a file in a folder without a plain name")?;
                        let sha = f["sha"].as_str().filter(|s| valid_sha(s)).ok_or("a file without its hash")?;
                        place(ws, sha, &root.join(rel))?;
                    }
                    root
                }
                Some("json") => {
                    let mut inner = d["value"].clone();
                    materialize(ws, &mut inner)?;
                    let text = serde_json::to_vec_pretty(&inner).unwrap_or_default();
                    use sha2::Digest;
                    let p = store(ws).join("in").join(&hex(&sha2::Sha256::digest(&text))[..16]).join(&name);
                    std::fs::create_dir_all(p.parent().unwrap()).map_err(|e| e.to_string())?;
                    std::fs::write(&p, text).map_err(|e| format!("{}: {e}", p.display()))?;
                    p
                }
                _ => return Err("an input of an unknown kind".into()),
            };
            *v = json!(path.display().to_string());
            Ok(())
        }
        Value::Object(o) => o.values_mut().try_for_each(|x| materialize(ws, x)),
        _ => Ok(()),
    }
}

/// `POST /api/cluster/workshop/jobs {"request"}` (a master): the job, its inputs already here by hash.
pub async fn jobs_post(State(s): State<AppState>, headers: HeaderMap, body: axum::body::Bytes) -> Response {
    let ws = match from_master(&s, &headers) { Ok(w) => w, Err(r) => return r };
    let v: Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(e) => return error(StatusCode::BAD_REQUEST, "invalid_request_error", format!("the body is not JSON: {e}")),
    };
    let mut req = v["request"].clone();
    let Some(o) = req.as_object_mut() else { return error(StatusCode::BAD_REQUEST, "invalid_request_error", "send {\"request\": {...}}") };
    // a routed job is never routed on
    o.remove("node");
    o.remove("keep_on_node");
    let tag = format!("{}-{}", crate::now(), hex(&rand::random::<[u8; 4]>()));
    let mut outs = vec![];
    for (k, x) in o.iter_mut() {
        if let Some(n) = x.get("$out").and_then(|n| n.as_str()) {
            if !plain_name(n) {
                return error(StatusCode::BAD_REQUEST, "invalid_request_error", format!("\"{k}\": not a plain name"));
            }
            let p = store(&ws).join("out").join(&tag).join(n);
            *x = json!(p.display().to_string());
            outs.push((k.clone(), p));
        }
    }
    o.insert("from_master".into(), json!(true));
    let ws2 = ws.clone();
    let r = tokio::task::spawn_blocking(move || materialize(&ws2, &mut req).map(|_| req)).await;
    let req = match r {
        Ok(Ok(r)) => r,
        Ok(Err(e)) if e.starts_with("missing:") => return error(StatusCode::CONFLICT, "missing_input", format!("an input ({}) is not here yet", &e[8..])),
        Ok(Err(e)) => return error(StatusCode::BAD_REQUEST, "invalid_request_error", e),
        Err(e) => return error(StatusCode::INTERNAL_SERVER_ERROR, "server_error", e.to_string()),
    };
    if let Some(d) = outs.first().and_then(|(_, p)| p.parent()) {
        let _ = std::fs::create_dir_all(d);
    }
    match ws.submit(req) {
        Ok(j) => {
            routed_jobs().lock().unwrap().insert(j.id, outs);
            tracing::info!("workshop: job {} ({}) from a cluster master", j.id, j.kind);
            (StatusCode::ACCEPTED, Json(j.view(false))).into_response()
        }
        Err(e) => error(StatusCode::BAD_REQUEST, "invalid_request_error", e),
    }
}

fn routed_id(id: &str) -> Result<u64, Response> {
    id.parse().ok().filter(|i| routed_jobs().lock().unwrap().contains_key(i))
        .ok_or_else(|| error(StatusCode::NOT_FOUND, "not_found", "no such job from a master"))
}

/// `GET /api/cluster/workshop/jobs/{id}` (a master): a job it sent, with its log and points.
pub async fn job_get(State(s): State<AppState>, headers: HeaderMap, UrlPath(id): UrlPath<String>) -> Response {
    let ws = match from_master(&s, &headers) { Ok(w) => w, Err(r) => return r };
    let id = match routed_id(&id) { Ok(i) => i, Err(r) => return r };
    match ws.job(id) {
        Some(j) => Json(j.view(true)).into_response(),
        None => error(StatusCode::NOT_FOUND, "not_found", "no such job"),
    }
}

/// `DELETE /api/cluster/workshop/jobs/{id}` (a master): cancel a job it sent.
pub async fn job_delete(State(s): State<AppState>, headers: HeaderMap, UrlPath(id): UrlPath<String>) -> Response {
    let ws = match from_master(&s, &headers) { Ok(w) => w, Err(r) => return r };
    let id = match routed_id(&id) { Ok(i) => i, Err(r) => return r };
    match ws.cancel(id) {
        Ok(j) => Json(j.view(false)).into_response(),
        Err(e) => error(StatusCode::CONFLICT, "invalid_request_error", e),
    }
}

/// The files of an output (a file: one with an empty rel; a folder: each file under it), with their hashes (blocking).
fn manifest(p: &Path) -> Result<Vec<Value>, String> {
    if p.is_file() {
        let (h, b) = hash_cached(p)?;
        return Ok(vec![json!({"rel": "", "bytes": b, "sha256": h})]);
    }
    let mut out = vec![];
    let mut stack = vec![p.to_path_buf()];
    while let Some(d) = stack.pop() {
        for e in std::fs::read_dir(&d).into_iter().flatten().flatten() {
            let q = e.path();
            let Ok(ft) = e.file_type() else { continue };
            if ft.is_symlink() {
                continue;
            }
            if ft.is_dir() {
                stack.push(q);
            } else {
                let rel = q.strip_prefix(p).map_err(|e| e.to_string())?.to_string_lossy().replace('\\', "/");
                if !plain_rel(&rel) {
                    continue;
                }
                let (h, b) = hash_cached(&q)?;
                out.push(json!({"rel": rel, "bytes": b, "sha256": h}));
            }
        }
    }
    Ok(out)
}

/// `GET /api/cluster/workshop/jobs/{id}/outputs` (a master): what a finished job wrote, each file with its hash.
pub async fn job_outputs(State(s): State<AppState>, headers: HeaderMap, UrlPath(id): UrlPath<String>) -> Response {
    let ws = match from_master(&s, &headers) { Ok(w) => w, Err(r) => return r };
    let id = match routed_id(&id) { Ok(i) => i, Err(r) => return r };
    if ws.job(id).is_none_or(|j| j.state != "done") {
        return error(StatusCode::CONFLICT, "invalid_request_error", "the job has not finished");
    }
    let outs = routed_jobs().lock().unwrap().get(&id).cloned().unwrap_or_default();
    let r = tokio::task::spawn_blocking(move || -> Result<Vec<Value>, String> {
        outs.iter().filter(|(_, p)| p.exists()).map(|(k, p)| Ok(json!({"field": k, "path": p.display().to_string(),
                                                                     "kind": if p.is_dir() { "dir" } else { "file" }, "files": manifest(p)?}))).collect()
    }).await;
    match r {
        Ok(Ok(v)) => Json(json!({"outputs": v})).into_response(),
        Ok(Err(e)) => error(StatusCode::INTERNAL_SERVER_ERROR, "server_error", e),
        Err(e) => error(StatusCode::INTERNAL_SERVER_ERROR, "server_error", e.to_string()),
    }
}

/// `GET /api/cluster/workshop/jobs/{id}/file?field=&rel=&offset=` (a master): one output file, from `offset`.
pub async fn job_file(State(s): State<AppState>, headers: HeaderMap, UrlPath(id): UrlPath<String>,
                      Query(q): Query<HashMap<String, String>>) -> Response {
    use tokio::io::AsyncSeekExt;
    let ws = match from_master(&s, &headers) { Ok(w) => w, Err(r) => return r };
    let id = match routed_id(&id) { Ok(i) => i, Err(r) => return r };
    let field = q.get("field").cloned().unwrap_or_default();
    let rel = q.get("rel").cloned().unwrap_or_default();
    let offset: u64 = q.get("offset").and_then(|v| v.parse().ok()).unwrap_or(0);
    let Some(base) = routed_jobs().lock().unwrap().get(&id).and_then(|o| o.iter().find(|(k, _)| *k == field).map(|(_, p)| p.clone())) else {
        return error(StatusCode::NOT_FOUND, "not_found", "no such output");
    };
    if !rel.is_empty() && !plain_rel(&rel) {
        return error(StatusCode::BAD_REQUEST, "invalid_request_error", "rel: a plain relative path");
    }
    let p = if rel.is_empty() { base.clone() } else { base.join(&rel) };
    // inside this job's output, inside the Workshop's folders, never a key file
    let ok = p.canonicalize().ok().zip(base.canonicalize().ok()).is_some_and(|(c, b)| c.starts_with(&b) && c.is_file())
        && ws.allowed(&p.display().to_string(), true).is_ok();
    if !ok {
        return error(StatusCode::NOT_FOUND, "not_found", "no such output file");
    }
    let mut f = match tokio::fs::File::open(&p).await {
        Ok(f) => f,
        Err(e) => return error(StatusCode::NOT_FOUND, "not_found", e.to_string()),
    };
    let len = f.metadata().await.map(|m| m.len()).unwrap_or(0);
    if offset > len || f.seek(std::io::SeekFrom::Start(offset)).await.is_err() {
        return error(StatusCode::RANGE_NOT_SATISFIABLE, "invalid_request_error", "offset past the end");
    }
    let stream = futures::stream::unfold(f, |mut f| async move {
        use tokio::io::AsyncReadExt;
        let mut buf = vec![0u8; CHUNK];
        match f.read(&mut buf).await {
            Ok(0) => None,
            Ok(n) => { buf.truncate(n); Some((Ok::<_, std::io::Error>(axum::body::Bytes::from(buf)), f)) }
            Err(e) => Some((Err(e), f)),
        }
    });
    let mut r = Response::new(axum::body::Body::from_stream(stream));
    r.headers_mut().insert("content-type", axum::http::HeaderValue::from_static("application/octet-stream"));
    if let Ok(v) = axum::http::HeaderValue::from_str(&(len - offset).to_string()) {
        r.headers_mut().insert("content-length", v);
    }
    r
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_and_paths_from_another_machine_are_plain() {
        assert!(plain_name("a.jsonl") && !plain_name("..") && !plain_name("a/b") && !plain_name(""));
        assert!(plain_rel("sub/model.safetensors") && !plain_rel("../x") && !plain_rel("/etc/passwd") && !plain_rel("a/./b"));
        assert!(valid_sha(&"a".repeat(64)) && !valid_sha(&"A".repeat(64)) && !valid_sha("abc"));
    }

    #[test]
    fn a_request_is_routed_only_when_it_names_another_node() {
        assert!(!routed(&json!({"kind": "clean"})));
        assert!(!routed(&json!({"node": "local"})) && !routed(&json!({"node": ""})));
        assert!(routed(&json!({"node": "macm4"})) && routed(&json!({"node": "any"})) && routed(&json!({"node": 3})));
    }

    #[test]
    fn private_marks_are_found_in_configs_rows_and_folders() {
        let d = std::env::temp_dir().join(format!("cw-private-{}", std::process::id()));
        let m = d.join("model");
        std::fs::create_dir_all(&m).unwrap();
        std::fs::write(d.join("open.jsonl"), "{\"messages\":[]}\n{\"messages\":[], \"note\": \"not private at all\"}\n").unwrap();
        assert!(private_reason(&d.join("open.jsonl")).is_none(), "a word in a row is not a mark");
        std::fs::write(d.join("rows.jsonl"), "{\"messages\":[]}\n{\"messages\":[],\"private\":true}\n").unwrap();
        assert!(private_reason(&d.join("rows.jsonl")).unwrap().contains("row 2"));
        std::fs::write(m.join("config.json"), "{\"model_type\":\"qwen3\"}").unwrap();
        assert!(private_reason(&m).is_none());
        std::fs::write(m.join("config.json"), "{\"model_type\":\"qwen3\",\"prastara_private\":true}").unwrap();
        assert!(private_reason(&m).unwrap().contains("prastara_private"));
        std::fs::write(d.join("strata-x.json"), "{\"uncensored\":true}").unwrap();
        assert!(private_reason(&d.join("strata-x.json")).is_some());
        assert!(private_reason(Path::new("/data/Strata-data/private/a.jsonl")).is_some());
        std::fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn a_capable_node_has_the_kind_and_the_gpu_asked_for() {
        let c = json!({"installed": true, "kinds": ROUTABLE, "gpu": {"cuda": false, "metal": true}});
        assert!(capable(&c, "clean", false));
        assert!(!capable(&c, "train", true), "metal is not what devices: cuda asks for");
        assert!(!capable(&json!({"installed": false, "kinds": []}), "clean", false));
    }
}
