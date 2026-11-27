//! The server side of a private-data import (docs/internal/PRIVATE-DATA.md, "Import"): the `private-import` job's
//! checks and arguments, the probe the confirm dialog shows (source and size), the list of imports in quarantine and
//! their discard. The work itself is `am2strata-workshop private-import` (strata-workshop's `private_import`).
//!
//! Nothing here fetches anything on its own: a probe and a job each start from the owner's own request (the CLI, or
//! the UI after its confirm dialog), and a network job needs `"confirm_bytes"`, the size the owner saw.

use std::path::{Path, PathBuf};

use axum::extract::{Path as UrlPath, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::{json, Value};

use super::{admin, exe, private_root, Workshop};
use crate::{error, AppState};

/// Whether `p` is inside a private quarantine (`.../private/quarantine/...`), by its components; resolved too when it
/// exists (a link into the quarantine counts). The same rule as strata-workshop's `private_import::in_quarantine`.
pub fn in_quarantine(p: &Path) -> bool {
    let check = |q: &Path| {
        let parts: Vec<String> = q.components().filter_map(|c| match c {
            std::path::Component::Normal(n) => Some(n.to_string_lossy().to_lowercase()),
            _ => None,
        }).collect();
        parts.windows(2).any(|w| w[0] == "private" && w[1] == "quarantine")
    };
    check(p) || p.canonicalize().is_ok_and(|c| check(&c))
}

fn plain(s: &str, max: usize) -> bool {
    !s.is_empty() && s.len() <= max && !s.starts_with('.') && s.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'))
}

/// https, or http to this machine's loopback (a local test server).
fn url_ok(u: &str) -> bool {
    if u.len() > 2048 || u.chars().any(|c| c.is_whitespace() || c.is_control()) {
        return false;
    }
    if u.starts_with("https://") && u.len() > 8 {
        return true;
    }
    let Some(rest) = u.strip_prefix("http://") else { return false };
    let host = rest.split(['/', '?', '#']).next().unwrap_or("");
    let host = host.strip_prefix('[').and_then(|h| h.split(']').next()).unwrap_or_else(|| host.rsplit_once(':').map_or(host, |(h, _)| h));
    host.eq_ignore_ascii_case("localhost") || host.parse::<std::net::IpAddr>().is_ok_and(|ip| ip.is_loopback())
}

impl Workshop {
    pub(crate) fn data_dir(&self) -> PathBuf {
        self.out_dir.parent().map(Path::to_path_buf).unwrap_or_else(|| self.out_dir.clone())
    }

    /// `"source"`: `{"folder"}` (inside the Workshop's folders), `{"url"}` or `{"hf", "revision"?, "pattern"?}`, as
    /// the command's arguments, and whether it goes over the network.
    fn import_source(&self, req: &Value) -> Result<(Vec<String>, bool), String> {
        let src = &req["source"];
        let s = |k: &str| src[k].as_str().map(str::trim).filter(|v| !v.is_empty());
        match (s("folder"), s("url"), s("hf")) {
            (Some(f), None, None) => {
                let p = self.allowed(f, true)?;
                if !p.is_dir() {
                    return Err(format!("{}: a folder source is a folder", p.display()));
                }
                Ok((vec!["--folder".into(), p.display().to_string()], false))
            }
            (None, Some(u), None) => {
                if !url_ok(u) {
                    return Err(format!("{u}: a URL source is a direct https URL"));
                }
                Ok((vec!["--url".into(), u.into()], true))
            }
            (None, None, Some(h)) => {
                let ok = matches!(h.split('/').collect::<Vec<_>>().as_slice(), [a, b] if plain(a, 96) && plain(b, 96));
                if !ok {
                    return Err(format!("{h:?}: a Hugging Face dataset id is owner/name"));
                }
                let mut a = vec!["--hf".to_string(), h.into()];
                if let Some(r) = s("revision") {
                    if r.len() > 128 || r.contains("..") || r.starts_with('/') || !r.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.' | '/')) {
                        return Err(format!("{r:?}: a revision is a branch, tag or commit"));
                    }
                    a.extend(["--revision".into(), r.into()]);
                }
                if let Some(p) = s("pattern") {
                    if p.len() > 512 || p.contains("..") || p.chars().any(|c| c.is_control()) {
                        return Err("\"pattern\" is a file pattern such as data/*.zip".into());
                    }
                    a.extend(["--pattern".into(), p.into()]);
                }
                Ok((a, true))
            }
            _ => Err("\"source\" is one of {\"folder\": ...}, {\"url\": \"https://...\"} or {\"hf\": \"owner/name\"}".into()),
        }
    }

    /// A private-import job: `"name"`, `"source"`, `"attestation"`? (a SOURCE.toml inside the Workshop's folders;
    /// needed for a URL or Hub source), `"confirm_bytes"` (a network source: the size the owner confirmed),
    /// `"minor_share"`? (at most 0.01), `"resume"`? (an import id in quarantine) -> `<data>/private/datasets/<name>`.
    pub(super) fn plan_import(&self, req: &Value) -> Result<(String, Vec<String>, PathBuf), String> {
        let name = req["name"].as_str().filter(|n| !n.is_empty()).ok_or("private-import needs \"name\": the dataset's name")?;
        if !plain(name, 64) {
            return Err("\"name\" is 1-64 letters, digits, '.', '_' or '-' (not starting with '.')".into());
        }
        let data_dir = self.data_dir();
        let root = private_root(&data_dir);
        let output = root.join("datasets").join(name);
        if output.join("dataset.card.md").exists() {
            return Err(format!("{}: a private dataset of this name exists; pick another \"name\"", output.display()));
        }
        let mut args = vec!["private-import".to_string(), "--name".into(), name.into(), "--data-dir".into(), data_dir.display().to_string()];
        // a resume: the import in quarantine keeps its own source and SOURCE.toml (and fetches only what a cut
        // download left, of the size confirmed then)
        if let Some(id) = req["resume"].as_str().filter(|i| !i.is_empty()) {
            if !plain(id, 64) || !root.join("quarantine").join(id).join("import.json").is_file() {
                return Err(format!("{id:?}: no import of that id in quarantine"));
            }
            let st: Value = std::fs::read_to_string(root.join("quarantine").join(id).join("import.json")).ok()
                .and_then(|t| serde_json::from_str(&t).ok()).unwrap_or_default();
            if st["name"].as_str() != Some(name) {
                return Err(format!("import {id} is for the dataset {}, not {name:?}", st["name"]));
            }
            args.extend(["--id".into(), id.into()]);
            if req["source"].is_null() {
                if let Some(n) = st["confirm_bytes"].as_u64() {
                    args.extend(["--confirm-bytes".into(), n.to_string()]);
                }
                return Ok(("private-import".into(), args, output));
            }
        }
        let (src, network) = self.import_source(req)?;
        args.extend(src);
        match req["attestation"].as_str().filter(|a| !a.is_empty()) {
            Some(a) => {
                let p = self.allowed(a, true)?;
                if !p.is_file() {
                    return Err(format!("{}: \"attestation\" is a SOURCE.toml file", p.display()));
                }
                args.extend(["--attestation".into(), p.display().to_string()]);
            }
            None if network => return Err("a URL or Hugging Face source needs \"attestation\": the SOURCE.toml you wrote for it (with licence and the source it is for)".into()),
            None => {}
        }
        match &req["confirm_bytes"] {
            Value::Null if network => return Err("a network source is fetched only after you confirm its size: probe it first, then send \"confirm_bytes\"".into()),
            Value::Null => {}
            v => {
                let n = v.as_u64().ok_or("\"confirm_bytes\" is the size in bytes the probe gave")?;
                args.extend(["--confirm-bytes".into(), n.to_string()]);
            }
        }
        match &req["minor_share"] {
            Value::Null => {}
            v => {
                let x = v.as_f64().filter(|x| (0.0..=0.01).contains(x)).ok_or("\"minor_share\" is from 0 to 0.01 (a stricter limit; the fixed one is 0.01)")?;
                args.extend(["--minor-share".into(), x.to_string()]);
            }
        }
        Ok(("private-import".into(), args, output))
    }

    /// Whether a model name is this server's own local model (a `<provider>/<model>` name, a mix, or a name the
    /// server does not have is not).
    pub(super) fn local_model(&self, name: &str) -> bool {
        if name.contains('/') || name.starts_with("mix") {
            return false;
        }
        match self.mgr.lock().unwrap().clone() {
            Some(m) => m.entry(name).is_some_and(|e| e.id == name),
            None => true,
        }
    }

    /// Why a job over these paths must keep its chats on this machine (None: no private input).
    pub(super) fn private_input(paths: &[PathBuf]) -> Option<String> {
        paths.iter().find_map(|p| crate::cluster_workshop::private_reason(p))
    }
}

async fn run_command(args: Vec<String>) -> Result<Vec<Value>, String> {
    let exe = exe().ok_or("the am2strata-workshop command is not beside the server")?;
    let fut = tokio::process::Command::new(exe).arg("--json").args(&args).stdin(std::process::Stdio::null()).kill_on_drop(true).output();
    let o = tokio::time::timeout(std::time::Duration::from_secs(120), fut).await
        .map_err(|_| "the probe took longer than 2 minutes".to_string())?.map_err(|e| format!("am2strata-workshop: {e}"))?;
    let events: Vec<Value> = String::from_utf8_lossy(&o.stdout).lines().filter_map(|l| serde_json::from_str(l).ok()).collect();
    if let Some(e) = events.iter().find(|e| e["event"] == "error") {
        return Err(e["message"].as_str().unwrap_or("failed").to_string());
    }
    Ok(events)
}

/// `POST /v1/workshop/private-import/probe` `{"source"}`: what the source holds (files, bytes, the Hub commit), for
/// the confirm dialog. Only this request (the owner's) asks the network; nothing is fetched.
pub async fn probe_post(State(s): State<AppState>, headers: HeaderMap, Json(req): Json<Value>) -> Response {
    let w = match admin(&s, &headers) { Ok(w) => w, Err(r) => return r };
    if crate::foreign_origin(&s, &headers) {
        return crate::not_own_page("a private import can be probed");
    }
    let (src, _) = match w.import_source(&req) {
        Ok(a) => a,
        Err(e) => return error(StatusCode::BAD_REQUEST, "invalid_request_error", e),
    };
    let mut args = vec!["private-import-probe".to_string()];
    args.extend(src);
    match run_command(args).await {
        Ok(ev) => match ev.into_iter().find(|e| e["event"] == "probe") {
            Some(p) => Json(json!({"what": p["what"], "bytes": p["bytes"], "files": p["files"], "commit": p["commit"], "network": p["network"]})).into_response(),
            None => error(StatusCode::BAD_GATEWAY, "server_error", "the probe gave no answer"),
        },
        Err(e) => error(StatusCode::BAD_REQUEST, "invalid_request_error", e),
    }
}

/// `GET /v1/workshop/private-imports`: the imports in quarantine (id, name, source, status, why it stopped, counts)
/// and the released private datasets (names). Never an item, a file name or a preview.
pub async fn list_get(State(s): State<AppState>, headers: HeaderMap) -> Response {
    let w = match admin(&s, &headers) { Ok(w) => w, Err(r) => return r };
    let root = private_root(&w.data_dir());
    let mut imports: Vec<Value> = std::fs::read_dir(root.join("quarantine")).into_iter().flatten().flatten().filter_map(|e| {
        let v: Value = serde_json::from_str(&std::fs::read_to_string(e.path().join("import.json")).ok()?).ok()?;
        Some(json!({"id": v["id"], "name": v["name"], "source": v["source"], "status": v["status"], "reason": v["reason"],
                    "counts": v["counts"], "created": v["created"]}))
    }).collect();
    imports.sort_by_key(|x| std::cmp::Reverse(x["created"].as_u64().unwrap_or(0)));
    let mut datasets: Vec<Value> = std::fs::read_dir(root.join("datasets")).into_iter().flatten().flatten()
        .filter(|e| e.path().join("dataset.card.md").is_file())
        .map(|e| json!({"name": e.file_name().to_string_lossy(), "path": e.path().display().to_string(),
                        "manifest": e.path().join("manifest.jsonl").display().to_string(),
                        "text": e.path().join("text.jsonl").is_file().then(|| e.path().join("text.jsonl").display().to_string())}))
        .collect();
    datasets.sort_by(|a, b| a["name"].as_str().cmp(&b["name"].as_str()));
    Json(json!({"imports": imports, "datasets": datasets, "check_models": root.join("check-models").display().to_string()})).into_response()
}

/// `POST /v1/workshop/private-imports/{id}/discard`: delete that import's quarantine (logged in the audit log).
pub async fn discard_post(State(s): State<AppState>, headers: HeaderMap, UrlPath(id): UrlPath<String>) -> Response {
    let w = match admin(&s, &headers) { Ok(w) => w, Err(r) => return r };
    if crate::foreign_origin(&s, &headers) {
        return crate::not_own_page("a private import can be discarded");
    }
    if !plain(&id, 64) {
        return error(StatusCode::BAD_REQUEST, "invalid_request_error", "not an import id");
    }
    let busy = w.jobs.lock().unwrap().iter().any(|j| j.kind == "private-import" && matches!(j.state, "queued" | "running")
        && (j.args.windows(2).any(|a| a[0] == "--id" && a[1] == id) || j.args.windows(2).any(|a| a[0] == "--name" && id.starts_with(&format!("{}-", a[1])))));
    if busy {
        return error(StatusCode::CONFLICT, "invalid_request_error", "this import's job is still running: cancel it first");
    }
    let args = vec!["private-import-discard".to_string(), "--id".into(), id.clone(), "--data-dir".into(), w.data_dir().display().to_string()];
    match run_command(args).await {
        Ok(_) => Json(json!({"discarded": id})).into_response(),
        Err(e) => error(StatusCode::BAD_REQUEST, "invalid_request_error", e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quarantine_paths_and_urls_are_told_apart() {
        assert!(in_quarantine(Path::new("/d/Strata-data/private/quarantine/x-1/items/a.png")));
        assert!(in_quarantine(Path::new("/d/Strata-data/Private/Quarantine/x")));
        assert!(!in_quarantine(Path::new("/d/Strata-data/private/datasets/x/text.jsonl")));
        assert!(url_ok("https://example.com/a.zip") && url_ok("http://127.0.0.1:8/a") && !url_ok("http://example.com/a") && !url_ok("https://a b"));
    }
}
