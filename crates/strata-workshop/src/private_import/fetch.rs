//! Fetching a private-import source over the network, in Rust (reqwest, rustls): a direct https URL, or a Hugging
//! Face dataset through the Hub's HTTP API (`/api/datasets/<id>/revision/<rev>` pins the commit,
//! `/api/datasets/<id>/tree/<commit>?recursive=true` lists the files, `/datasets/<id>/resolve/<commit>/<path>` gives
//! each). Downloads are resumable (a `.part` file continued with a Range request), checked against the size the
//! probe saw and, for the Hub's LFS files, against the SHA-256 the Hub lists; every file's SHA-256 is recorded.
//!
//! Nothing here runs on its own: the owner starts each fetch (the CLI, or the UI after a confirm dialog that names the
//! source and its size), and the size the owner confirmed must still be the size, or nothing is fetched.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{anyhow, bail, Context, Result};
use serde_json::Value;
use sha2::Digest;

/// Where a source is.
#[derive(Debug, Clone, PartialEq)]
pub enum Spec {
    Folder(PathBuf),
    Url(String),
    Hf { id: String, revision: String, pattern: Option<String> },
}

impl Spec {
    /// How the audit log and the card name it.
    pub fn describe(&self) -> String {
        match self {
            Spec::Folder(p) => format!("folder {}", p.display()),
            Spec::Url(u) => format!("url {u}"),
            Spec::Hf { id, revision, pattern } => format!("hf dataset {id}@{revision}{}", pattern.as_ref().map(|p| format!(" ({p})")).unwrap_or_default()),
        }
    }

    pub fn is_network(&self) -> bool {
        !matches!(self, Spec::Folder(_))
    }
}

/// One file to fetch.
#[derive(Debug, Clone, PartialEq)]
pub struct Remote {
    pub url: String,
    /// the name it is saved under in the quarantine's downloads (a plain name, made from the path)
    pub name: String,
    pub bytes: u64,
    /// the SHA-256 the source lists for it (the Hub's LFS files), checked after the download
    pub sha256: Option<String>,
}

/// What a probe found: the files, their size together, and for the Hub the commit they are pinned to.
#[derive(Debug, Clone, PartialEq)]
pub struct Probe {
    pub files: Vec<Remote>,
    pub bytes: u64,
    pub commit: Option<String>,
}

/// The HTTP side of the network sources.
pub struct Net {
    client: reqwest::blocking::Client,
    /// the Hub's address (`HF_ENDPOINT`, else https://huggingface.co)
    pub hf_endpoint: String,
    hf_token: Option<String>,
}

/// Whether a URL may be fetched: https, or http to this machine's loopback (a local test server).
pub fn url_allowed(url: &str) -> bool {
    if url.starts_with("https://") {
        return true;
    }
    let Some(rest) = url.strip_prefix("http://") else { return false };
    let host = rest.split(['/', '?', '#']).next().unwrap_or("");
    let host = host.strip_prefix('[').and_then(|h| h.split(']').next()).unwrap_or_else(|| host.rsplit_once(':').map_or(host, |(h, _)| h));
    host.eq_ignore_ascii_case("localhost") || host.parse::<std::net::IpAddr>().is_ok_and(|ip| ip.is_loopback())
}

/// A Hub dataset id: `owner/name`, letters, digits, `-`, `_`, `.`.
pub fn hf_id_ok(id: &str) -> bool {
    let ok = |s: &str| !s.is_empty() && s.len() <= 96 && !s.starts_with('.') && s.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'));
    matches!(id.split('/').collect::<Vec<_>>().as_slice(), [a, b] if ok(a) && ok(b))
}

/// A revision: a branch, a tag or a commit (no `..`, no spaces).
pub fn revision_ok(r: &str) -> bool {
    !r.is_empty() && r.len() <= 128 && !r.contains("..") && !r.starts_with('/')
        && r.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.' | '/'))
}

/// A file pattern: `*` (within a folder), `**` (any folders), `?`; `|` between alternatives.
pub fn glob(pattern: &str, path: &str) -> bool {
    fn m(p: &[u8], s: &[u8]) -> bool {
        match (p.first(), s.first()) {
            (None, None) => true,
            (None, Some(_)) => false,
            (Some(b'*'), _) if p.get(1) == Some(&b'*') => {
                let rest = if p.get(2) == Some(&b'/') { &p[3..] } else { &p[2..] };
                (0..=s.len()).any(|i| m(rest, &s[i..]))
            }
            (Some(b'*'), _) => (0..=s.len()).take_while(|i| *i == 0 || s[i - 1] != b'/').any(|i| m(&p[1..], &s[i..])),
            (Some(b'?'), Some(c)) if *c != b'/' => m(&p[1..], &s[1..]),
            (Some(a), Some(b)) if a == b => m(&p[1..], &s[1..]),
            _ => false,
        }
    }
    pattern.split('|').map(str::trim).filter(|p| !p.is_empty()).any(|p| m(p.as_bytes(), path.as_bytes()))
}

/// A plain file name from a URL path or a Hub path (`a/b/c.zip` → `a_b_c.zip`).
pub fn plain_name(path: &str) -> String {
    let n: String = path.trim_matches('/').chars().map(|c| if c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | '_') { c } else { '_' }).collect();
    let n = n.trim_start_matches('.').to_string();
    let n = if n.len() > 120 { n[n.len() - 120..].to_string() } else { n };
    if n.is_empty() { "download".into() } else { n }
}

impl Net {
    pub fn new(hf_endpoint: Option<String>, hf_token: Option<String>) -> Result<Net> {
        // redirects only to https (or this machine): a link that sends the download elsewhere in plain http is refused
        let policy = reqwest::redirect::Policy::custom(|a| {
            if a.previous().len() > 8 { a.error("too many redirects") }
            else if url_allowed(a.url().as_str()) { a.follow() }
            else { a.error("a redirect away from https") }
        });
        let client = reqwest::blocking::Client::builder().redirect(policy).connect_timeout(Duration::from_secs(30))
            .timeout(None).user_agent("prastara-workshop").build()?;
        let ep = hf_endpoint.filter(|e| !e.is_empty()).unwrap_or_else(|| "https://huggingface.co".into()).trim_end_matches('/').to_string();
        if !url_allowed(&ep) {
            bail!("{ep}: the Hub's address must be https");
        }
        Ok(Net { client, hf_endpoint: ep, hf_token: hf_token.filter(|t| !t.is_empty()) })
    }

    pub fn from_env() -> Result<Net> {
        Net::new(std::env::var("HF_ENDPOINT").ok(), std::env::var("HF_TOKEN").ok())
    }

    fn get(&self, url: &str) -> reqwest::blocking::RequestBuilder {
        let rb = self.client.get(url);
        match (&self.hf_token, url.starts_with(&self.hf_endpoint)) {
            (Some(t), true) => rb.bearer_auth(t),
            _ => rb,
        }
    }

    /// The size of a URL (HEAD, else a one-byte range); None when the server does not say.
    fn size_of(&self, url: &str) -> Result<u64> {
        let r = self.client.head(url).send().with_context(|| format!("{url} does not answer"))?;
        if r.status().is_success() {
            if let Some(n) = r.headers().get(reqwest::header::CONTENT_LENGTH).and_then(|v| v.to_str().ok()).and_then(|v| v.parse().ok()) {
                return Ok(n);
            }
        }
        let r = self.client.get(url).header(reqwest::header::RANGE, "bytes=0-0").send().with_context(|| format!("{url} does not answer"))?;
        let total = r.headers().get(reqwest::header::CONTENT_RANGE).and_then(|v| v.to_str().ok())
            .and_then(|v| v.rsplit('/').next()).and_then(|v| v.parse().ok());
        total.ok_or_else(|| anyhow!("{url} does not say its size (HTTP {}); nothing is fetched without a size to confirm", r.status()))
    }

    /// What a source holds, without fetching it.
    pub fn probe(&self, spec: &Spec) -> Result<Probe> {
        match spec {
            Spec::Folder(_) => bail!("a folder is walked, not fetched"),
            Spec::Url(u) => {
                if !url_allowed(u) {
                    bail!("{u}: only https URLs are fetched");
                }
                let bytes = self.size_of(u)?;
                let path = u.split(['?', '#']).next().unwrap_or("").rsplit('/').next().unwrap_or("");
                Ok(Probe { files: vec![Remote { url: u.clone(), name: plain_name(path), bytes, sha256: None }], bytes, commit: None })
            }
            Spec::Hf { id, revision, pattern } => {
                if !hf_id_ok(id) || !revision_ok(revision) {
                    bail!("{id}@{revision}: not a Hugging Face dataset id (owner/name) and revision");
                }
                let rev_url = format!("{}/api/datasets/{id}/revision/{revision}", self.hf_endpoint);
                let r = self.get(&rev_url).send().with_context(|| format!("{} does not answer", self.hf_endpoint))?;
                if !r.status().is_success() {
                    bail!("{id}@{revision}: the Hub says HTTP {} (a gated dataset needs its terms accepted on the Hub and HF_TOKEN)", r.status());
                }
                let info: Value = r.json()?;
                let commit = info["sha"].as_str().filter(|s| s.len() >= 7 && s.chars().all(|c| c.is_ascii_hexdigit()))
                    .ok_or_else(|| anyhow!("{id}@{revision}: the Hub gave no commit"))?.to_string();
                let mut url = format!("{}/api/datasets/{id}/tree/{commit}?recursive=true", self.hf_endpoint);
                let mut files = vec![];
                for _ in 0..10_000 {
                    let r = self.get(&url).send()?;
                    if !r.status().is_success() {
                        bail!("{id}: the file list says HTTP {}", r.status());
                    }
                    let next = r.headers().get(reqwest::header::LINK).and_then(|v| v.to_str().ok()).and_then(next_link);
                    let page: Value = r.json()?;
                    for f in page.as_array().into_iter().flatten().filter(|f| f["type"] == "file") {
                        let path = f["path"].as_str().unwrap_or("");
                        if path.is_empty() || path.split('/').any(|p| p == ".." || p.is_empty()) {
                            continue;
                        }
                        // the Hub's own files, never data
                        if path == ".gitattributes" || path.ends_with("/.gitattributes") {
                            continue;
                        }
                        if pattern.as_ref().is_some_and(|p| !glob(p, path)) {
                            continue;
                        }
                        let bytes = f["lfs"]["size"].as_u64().or(f["size"].as_u64()).unwrap_or(0);
                        let sha256 = f["lfs"]["oid"].as_str().filter(|s| s.len() == 64).map(str::to_lowercase);
                        let enc: String = path.split('/').map(|s| percent(s)).collect::<Vec<_>>().join("/");
                        files.push(Remote { url: format!("{}/datasets/{id}/resolve/{commit}/{enc}", self.hf_endpoint), name: plain_name(path), bytes, sha256 });
                    }
                    match next {
                        Some(n) => url = n,
                        None => break,
                    }
                }
                if files.is_empty() {
                    bail!("{id}@{revision}: no files{}", pattern.as_ref().map(|p| format!(" match {p:?}")).unwrap_or_default());
                }
                let bytes = files.iter().map(|f| f.bytes).sum();
                Ok(Probe { files, bytes, commit: Some(commit) })
            }
        }
    }

    /// One file into `dir/<name>`, resumed from `dir/<name>.part`; its SHA-256 (checked when the source lists one).
    /// `stop()` ends it early (the `.part` stays for a resume).
    pub fn download(&self, f: &Remote, dir: &Path, mut progress: impl FnMut(u64), stop: &dyn Fn() -> bool) -> Result<(PathBuf, String)> {
        let dst = dir.join(&f.name);
        let part = dir.join(format!("{}.part", f.name));
        if !dst.exists() {
            let mut have = std::fs::metadata(&part).map(|m| m.len()).unwrap_or(0);
            if have > f.bytes {
                std::fs::remove_file(&part).ok();
                have = 0;
            }
            if have < f.bytes || f.bytes == 0 {
                let mut rb = self.get(&f.url);
                if have > 0 {
                    rb = rb.header(reqwest::header::RANGE, format!("bytes={have}-"));
                }
                let mut r = rb.send().with_context(|| format!("{} does not answer", f.url))?;
                let status = r.status();
                let append = match status.as_u16() {
                    206 => true,
                    200 => false, // the server sends it whole: start again
                    416 if have == f.bytes => true,
                    _ => bail!("{}: HTTP {status}", f.url),
                };
                if !append {
                    have = 0;
                }
                let mut o = std::fs::OpenOptions::new();
                o.create(true).write(true);
                if append { o.append(true); } else { o.truncate(true); }
                #[cfg(unix)]
                { use std::os::unix::fs::OpenOptionsExt as _; o.mode(0o600); }
                let mut out = o.open(&part).with_context(|| format!("{}", part.display()))?;
                if status.as_u16() != 416 {
                    let mut buf = vec![0u8; 1 << 20];
                    loop {
                        if stop() {
                            out.flush()?;
                            bail!("stopped; the download resumes from {have} bytes");
                        }
                        let k = r.read(&mut buf).with_context(|| format!("reading {}", f.url))?;
                        if k == 0 { break; }
                        have += k as u64;
                        if have > f.bytes {
                            bail!("{}: more than the {} bytes confirmed", f.url, f.bytes);
                        }
                        out.write_all(&buf[..k])?;
                        progress(have);
                    }
                }
                out.flush()?;
            }
            if have != f.bytes {
                bail!("{}: {have} of {} bytes; it resumes on the next run", f.url, f.bytes);
            }
            std::fs::rename(&part, &dst)?;
        }
        let hash = hash_file(&dst)?;
        if let Some(want) = &f.sha256 {
            if &hash != want {
                std::fs::remove_file(&dst).ok();
                bail!("{}: its SHA-256 is not the one the source lists; removed", f.url);
            }
        }
        Ok((dst, hash))
    }
}

/// The SHA-256 of a file, streamed.
pub fn hash_file(p: &Path) -> Result<String> {
    let mut f = std::fs::File::open(p).with_context(|| format!("{}", p.display()))?;
    let mut h = sha2::Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let k = f.read(&mut buf)?;
        if k == 0 { break; }
        h.update(&buf[..k]);
    }
    Ok(h.finalize().iter().map(|b| format!("{b:02x}")).collect())
}

fn percent(s: &str) -> String {
    s.bytes().map(|b| if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.' | b'~') { (b as char).to_string() } else { format!("%{b:02X}") }).collect()
}

/// The `rel="next"` target of a Link header.
fn next_link(h: &str) -> Option<String> {
    h.split(',').find(|p| p.contains("rel=\"next\"")).and_then(|p| p.split(['<', '>']).nth(1)).map(str::to_string)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex};

    /// A local test HTTP server (std only): GET with Range, HEAD, fixed JSON. Counts the requests per path.
    pub struct TestServer {
        pub base: String,
        pub hits: Arc<Mutex<HashMap<String, usize>>>,
    }

    pub fn serve(routes: HashMap<String, (String, Vec<u8>)>, cut_first: Option<(String, usize)>) -> TestServer {
        let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let base = format!("http://{}", l.local_addr().unwrap());
        let hits: Arc<Mutex<HashMap<String, usize>>> = Arc::default();
        let h2 = hits.clone();
        let routes = Arc::new(routes);
        let cut = Arc::new(Mutex::new(cut_first));
        std::thread::spawn(move || {
            for s in l.incoming().flatten() {
                let (routes, hits, cut) = (routes.clone(), h2.clone(), cut.clone());
                std::thread::spawn(move || {
                    let mut s = s;
                    let mut buf = vec![0u8; 8192];
                    let n = s.read(&mut buf).unwrap_or(0);
                    let req = String::from_utf8_lossy(&buf[..n]).to_string();
                    let mut lines = req.lines();
                    let first = lines.next().unwrap_or("");
                    let mut parts = first.split(' ');
                    let (method, path) = (parts.next().unwrap_or(""), parts.next().unwrap_or("").to_string());
                    let range = lines.find_map(|l| l.to_lowercase().strip_prefix("range: bytes=").map(|r| r.trim().to_string()));
                    *hits.lock().unwrap().entry(path.clone()).or_default() += 1;
                    let Some((ctype, body)) = routes.get(&path) else {
                        let _ = s.write_all(b"HTTP/1.1 404 Not Found\r\ncontent-length: 0\r\nconnection: close\r\n\r\n");
                        return;
                    };
                    let (status, slice, extra) = match range {
                        Some(r) => {
                            let (a, b) = r.split_once('-').unwrap_or((&r, ""));
                            let a: usize = a.parse().unwrap_or(0);
                            let b: usize = b.parse().ok().unwrap_or(body.len().saturating_sub(1)).min(body.len().saturating_sub(1));
                            if a >= body.len() {
                                ("416 Range Not Satisfiable", vec![], format!("content-range: bytes */{}\r\n", body.len()))
                            } else {
                                ("206 Partial Content", body[a..=b].to_vec(), format!("content-range: bytes {a}-{b}/{}\r\n", body.len()))
                            }
                        }
                        None => ("200 OK", body.clone(), String::new()),
                    };
                    // the first GET of `cut` path is cut off after some bytes (a dropped connection)
                    let mut send = slice.clone();
                    let mut len = slice.len();
                    if method == "GET" {
                        let mut c = cut.lock().unwrap();
                        if c.as_ref().is_some_and(|(p, _)| *p == path) {
                            let k = c.take().unwrap().1;
                            send.truncate(k);
                            len = slice.len();
                        }
                    }
                    let head = format!("HTTP/1.1 {status}\r\ncontent-type: {ctype}\r\ncontent-length: {len}\r\naccept-ranges: bytes\r\n{extra}connection: close\r\n\r\n");
                    let _ = s.write_all(head.as_bytes());
                    if method != "HEAD" {
                        let _ = s.write_all(&send);
                    }
                });
            }
        });
        TestServer { base, hits }
    }

    #[test]
    fn urls_ids_and_patterns_are_checked() {
        assert!(url_allowed("https://example.com/a.zip") && url_allowed("http://127.0.0.1:9/a") && url_allowed("http://localhost/x"));
        assert!(!url_allowed("http://example.com/a.zip") && !url_allowed("ftp://x") && !url_allowed("file:///etc/passwd"));
        assert!(hf_id_ok("someone/some-set_1.0") && !hf_id_ok("a/b/c") && !hf_id_ok("../x") && !hf_id_ok("x"));
        assert!(revision_ok("main") && revision_ok("refs/convert/parquet") && !revision_ok("../x") && !revision_ok("a b"));
        assert!(glob("*.zip", "a.zip") && !glob("*.zip", "d/a.zip") && glob("**/*.zip", "d/e/a.zip") && glob("**/*.zip", "a.zip"));
        assert!(glob("data/*.txt|*.md", "x.md") && !glob("data/?.txt", "data/ab.txt"));
        assert_eq!(plain_name("a/b c/../d.zip"), "a_b_c_.._d.zip".trim_start_matches('.'));
        assert_eq!(next_link("<https://h/api?cursor=2>; rel=\"next\""), Some("https://h/api?cursor=2".into()));
    }

    #[test]
    fn a_url_is_probed_and_a_cut_download_resumes() {
        let body: Vec<u8> = b"lorem ipsum dolor sit amet ".repeat(4000);
        let routes = HashMap::from([("/files/set.txt".to_string(), ("text/plain".to_string(), body.clone()))]);
        let srv = serve(routes, Some(("/files/set.txt".into(), 1000)));
        let net = Net::new(None, None).unwrap();
        let p = net.probe(&Spec::Url(format!("{}/files/set.txt", srv.base))).unwrap();
        assert_eq!((p.bytes, p.files.len(), p.files[0].name.as_str()), (body.len() as u64, 1, "set.txt"));
        let d = std::env::temp_dir().join(format!("pi-fetch-{}", std::process::id()));
        std::fs::remove_dir_all(&d).ok();
        std::fs::create_dir_all(&d).unwrap();
        // the first try is cut off: a .part stays
        assert!(net.download(&p.files[0], &d, |_| {}, &|| false).is_err());
        let had = std::fs::metadata(d.join("set.txt.part")).unwrap().len();
        assert_eq!(had, 1000);
        let (path, hash) = net.download(&p.files[0], &d, |_| {}, &|| false).unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), body);
        assert_eq!(hash, crate::private_data::sha256_hex(&body));
        // a wrong listed hash removes the file
        let mut wrong = p.files[0].clone();
        wrong.sha256 = Some("0".repeat(64));
        wrong.name = "again.txt".into();
        assert!(net.download(&wrong, &d, |_| {}, &|| false).unwrap_err().to_string().contains("SHA-256"));
        assert!(!d.join("again.txt").exists());
        std::fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn a_hub_dataset_is_listed_pinned_and_filtered() {
        let a = b"lorem ipsum".to_vec();
        let commit = "0123456789abcdef0123456789abcdef01234567";
        let tree = serde_json::json!([
            {"type": "file", "path": ".gitattributes", "size": 10},
            {"type": "directory", "path": "data"},
            {"type": "file", "path": "data/a.txt", "size": a.len(), "lfs": {"oid": crate::private_data::sha256_hex(&a), "size": a.len()}},
            {"type": "file", "path": "README.md", "size": 5}]);
        let routes = HashMap::from([
            ("/api/datasets/someone/set/revision/main".to_string(), ("application/json".to_string(), serde_json::json!({"sha": commit}).to_string().into_bytes())),
            (format!("/api/datasets/someone/set/tree/{commit}?recursive=true"), ("application/json".to_string(), tree.to_string().into_bytes())),
            (format!("/datasets/someone/set/resolve/{commit}/data/a.txt"), ("text/plain".to_string(), a.clone())),
        ]);
        let srv = serve(routes, None);
        let net = Net::new(Some(srv.base.clone()), Some("hf_test_token".into())).unwrap();
        let spec = Spec::Hf { id: "someone/set".into(), revision: "main".into(), pattern: Some("data/*.txt".into()) };
        let p = net.probe(&spec).unwrap();
        assert_eq!((p.files.len(), p.bytes, p.commit.as_deref()), (1, a.len() as u64, Some(commit)));
        let d = std::env::temp_dir().join(format!("pi-hf-{}", std::process::id()));
        std::fs::remove_dir_all(&d).ok();
        std::fs::create_dir_all(&d).unwrap();
        let (path, hash) = net.download(&p.files[0], &d, |_| {}, &|| false).unwrap();
        assert_eq!((std::fs::read(path).unwrap(), hash), (a.clone(), crate::private_data::sha256_hex(&a)));
        assert!(net.probe(&Spec::Hf { id: "someone/set".into(), revision: "main".into(), pattern: Some("*.parquet".into()) }).is_err());
        std::fs::remove_dir_all(&d).ok();
    }
}
