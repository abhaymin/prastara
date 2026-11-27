//! Workshop jobs across the cluster (docs/CLUSTER-PLAN.md, "Workshop jobs across the cluster"; cluster_workshop.rs):
//! two servers on this machine, paired over pinned TLS as the cluster tests pair them. A clean job named for the
//! worker runs there and its output comes back matching by hash; a transfer cut off resumes where the worker's copy
//! stopped; a hash the worker has is not sent again; a private input is refused with "any" and taken when the worker
//! is named; a cancel on the master reaches the worker's job. The worker runs a stand-in for am2strata-workshop (no GPU).

#![cfg(unix)]

mod common;

use common::*;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

/// A stand-in for am2strata-workshop's clean: keeps the rows without "drop" in them. An input named *slow* waits 60 s
/// first (for cancel).
fn fake_workshop(dir: &Path) -> PathBuf {
    std::fs::create_dir_all(dir).unwrap();
    let p = dir.join("fake-workshop.sh");
    std::fs::write(&p, r#"#!/bin/sh
shift; kind="$1"; input="$2"; out=""
while [ $# -gt 0 ]; do [ "$1" = "--out" ] && out="$2"; shift; done
echo "{\"event\":\"start\",\"what\":\"$kind $input\"}"
case "$input" in
  *slow*) echo '{"event":"log","message":"a slow one"}'; sleep 60 ;;
esac
mkdir -p "$(dirname "$out")"
grep -v '"drop"' "$input" > "$out"
echo '{"event":"progress","done":1,"total":1,"message":"rows checked"}'
echo "{\"event\":\"log\",\"message\":\"cleaned on the worker\"}"
echo "{\"event\":\"done\",\"output\":\"$out\",\"kind\":\"dataset\"}"
"#).unwrap();
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o755)).unwrap();
    p
}

fn sha(p: &Path) -> String {
    use sha2::Digest;
    sha2::Sha256::digest(std::fs::read(p).unwrap()).iter().map(|b| format!("{b:02x}")).collect()
}

/// A worker (named macm4, the stand-in command, plain HTTP still allowed for its master's key so the test can cut a
/// transfer itself) and a master with no Workshop command of its own, paired over TLS: (worker, master, the
/// worker's name on the master, the key the worker gave the master).
fn pair(tools: &Path) -> (Server, Server, String, String) {
    let exe = fake_workshop(tools);
    let worker = Server::start(&[Model { id: "mock-b", extra: json!({"cluster": {"role": "worker", "name": "macm4", "allow_plain": true,
                                                                                 "tls_port": free_port(), "workshop_gpu": "none"}}),
                                         ..Model::default() }],
                               &[ids("hi")], json!({"env": {"AM2STRATA_WORKSHOP_EXE": exe}}));
    let master = Server::start(&[Model::default()], &[ids("hi")],
                               json!({"env": {"AM2STRATA_WORKSHOP_EXE": "/nonexistent/am2strata-workshop"}}));
    let code = worker.get("/api/cluster", true).1["pairing_code"].as_str().unwrap().to_string();
    let (c, p) = master.post("/api/cluster/pair", &json!({"url": worker.base, "code": code}));
    assert_eq!((c, p["encrypted"].as_bool()), (200, Some(true)), "{p}");
    let name = p["paired"].as_str().unwrap().to_string();
    let saved: Value = serde_json::from_str(&std::fs::read_to_string(master.dir.join("cluster-workers.json")).unwrap()).unwrap();
    let key = saved["workers"][0]["key"].as_str().unwrap().to_string();
    let t0 = Instant::now();
    while master.get("/api/cluster", true).1["workers"][0]["ready"] != true {
        assert!(t0.elapsed() < Duration::from_secs(45), "{}", master.get("/api/cluster", true).1);
        std::thread::sleep(Duration::from_millis(200));
    }
    (worker, master, name, key)
}

fn wait_job(s: &Server, id: u64, want: &[&str]) -> Value {
    let t0 = Instant::now();
    loop {
        let (_, j) = s.get(&format!("/v1/workshop/jobs/{id}"), true);
        if want.iter().any(|w| j["state"] == *w) {
            return j;
        }
        assert!(t0.elapsed() < Duration::from_secs(60), "job {id} is {} not {want:?}: {j}", j["state"]);
        std::thread::sleep(Duration::from_millis(200));
    }
}

fn rows(path: &Path, rows: &[&str]) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, rows.iter().map(|r| format!("{r}\n")).collect::<String>()).unwrap();
}

#[test]
fn a_clean_job_runs_on_the_worker_and_its_files_travel_by_hash() {
    let tools = std::env::temp_dir().join(format!("am2-cw-{}-a", std::process::id()));
    let (worker, master, name, key) = pair(&tools);
    assert_eq!(name, "macm4");
    let ds = master.dir.join("workshop/datasets");

    // the nodes the "Run on" select offers: this machine (no Workshop command here) and the worker, with what it can run
    let (c, nodes) = master.get("/api/cluster/workshop", true);
    assert_eq!(c, 200, "{nodes}");
    assert_eq!(nodes["this"]["installed"], false, "{nodes}");
    let w0 = &nodes["workers"][0];
    assert_eq!((w0["name"].as_str(), w0["caps"]["installed"].as_bool(), w0["caps"]["gpu"]["cuda"].as_bool()), (Some("macm4"), Some(true), Some(false)), "{nodes}");
    assert!(w0["caps"]["kinds"].as_array().unwrap().iter().any(|k| k == "clean"));
    assert!(nodes["privacy"].as_str().unwrap().contains("never"), "{nodes}");
    assert_eq!(master.get("/api/cluster/workshop", false).0, 401);
    // the worker's side opens to the master's key only: not to its own owner's key, not to no key
    assert_eq!(worker.get("/api/cluster/workshop/caps", true).0, 401);
    assert_eq!(worker.get("/api/cluster/workshop/caps", false).0, 401);

    // 1. a clean job for the worker: its input copied there, run there, its output copied back matching by hash
    let input = ds.join("mula-a.jsonl");
    rows(&input, &[r#"{"messages":[{"role":"user","content":"q1"}]}"#, r#"{"messages":[],"drop":1}"#, r#"{"messages":[{"role":"user","content":"q2"}]}"#]);
    let (c, j) = master.post("/v1/workshop/jobs", &json!({"kind": "clean", "inputs": [input], "node": "macm4"}));
    assert_eq!(c, 202, "{j}");
    assert_eq!(j["node"], "macm4", "the card's node badge: {j}");
    let id = j["id"].as_u64().unwrap();
    let done = wait_job(&master, id, &["done", "failed"]);
    assert_eq!(done["state"], "done", "{done}");
    let out = done["result"]["output"].as_str().unwrap().to_string();
    assert!(out.starts_with(&master.dir.join("workshop/datasets").display().to_string()) && out.ends_with("mula-a-clean.jsonl"), "{out}");
    let remote = done["result"]["remote_output"].as_str().unwrap().to_string();
    assert!(remote.starts_with(&worker.dir.join("workshop/cluster/out").display().to_string()), "{remote}");
    assert_eq!(sha(Path::new(&out)), sha(Path::new(&remote)), "the copy back matches the worker's file");
    let text = std::fs::read_to_string(&out).unwrap();
    assert!(text.contains("q1") && text.contains("q2") && !text.contains("drop"), "cleaned there: {text}");
    assert_eq!(done["result"]["transfer"]["outputs"][0]["sha256"], sha(Path::new(&out)).as_str());
    let tin = &done["result"]["transfer"]["inputs"][0];
    assert_eq!((tin["cached"].as_bool(), tin["sent"].as_u64()), (Some(false), Some(std::fs::metadata(&input).unwrap().len())), "{tin}");
    let log: Vec<&str> = done["log"].as_array().unwrap().iter().filter_map(|l| l.as_str()).collect();
    assert!(log.iter().any(|l| *l == "macm4: cleaned on the worker"), "the worker's log, mirrored: {log:?}");
    // the job is listed with the master's own, labelled with its node
    let (_, list) = master.get("/v1/workshop/jobs", true);
    assert!(list["jobs"].as_array().unwrap().iter().any(|x| x["id"] == id && x["node"] == "macm4"), "{list}");
    // the worker kept the input by its hash, read-only
    let blob = worker.dir.join("workshop/cluster/blobs").join(sha(&input));
    assert!(blob.is_file());
    {
        use std::os::unix::fs::PermissionsExt;
        assert_eq!(std::fs::metadata(&blob).unwrap().permissions().mode() & 0o222, 0, "a blob is read-only");
    }

    // 2. the same input again: the worker has its hash, so nothing is sent
    let (c, j) = master.post("/v1/workshop/jobs", &json!({"kind": "clean", "inputs": [input], "node": "macm4"}));
    assert_eq!(c, 202, "{j}");
    let again = wait_job(&master, j["id"].as_u64().unwrap(), &["done", "failed"]);
    assert_eq!(again["state"], "done", "{again}");
    let tin = &again["result"]["transfer"]["inputs"][0];
    assert_eq!((tin["cached"].as_bool(), tin["sent"].as_u64()), (Some(true), Some(0)), "a cached hash is not re-sent: {tin}");
    assert!(again["log"].as_array().unwrap().iter().any(|l| l.as_str().unwrap_or("").contains("already on macm4")), "{again}");

    // 3. a transfer cut off half way: the next one resumes at the byte the worker has
    let big = ds.join("big.jsonl");
    let row = r#"{"messages":[{"role":"user","content":"a longer row, so the file is a few megabytes and the cut lands mid-way"}]}"#;
    let body: String = (0..40_000).map(|i| format!("{}\n", row.replace("a longer", &format!("row {i}, a longer")))).collect();
    std::fs::write(&big, &body).unwrap();
    let size = body.len() as u64;
    let half = size / 2;
    let h = sha(&big);
    // the first half sent with the master's key (on plain HTTP, which this worker still allows), then the stream ends
    let r = reqwest::blocking::Client::new().put(format!("{}/api/cluster/files/{h}?offset=0&size={size}", worker.base)).bearer_auth(&key)
        .body(body.as_bytes()[..half as usize].to_vec()).send().unwrap();
    assert_eq!(r.status().as_u16(), 200);
    let v: Value = r.json().unwrap();
    assert_eq!((v["have"].as_bool(), v["offset"].as_u64()), (Some(false), Some(half)), "{v}");
    // a wrong offset is refused, with what the worker has
    let r = reqwest::blocking::Client::new().put(format!("{}/api/cluster/files/{h}?offset=7&size={size}", worker.base)).bearer_auth(&key)
        .body("x").send().unwrap();
    assert_eq!(r.status().as_u16(), 409);
    // the owner's key of the worker is not a master's
    let r = reqwest::blocking::Client::new().get(format!("{}/api/cluster/files/{h}", worker.base)).bearer_auth(KEY).send().unwrap();
    assert_eq!(r.status().as_u16(), 401);
    let (c, j) = master.post("/v1/workshop/jobs", &json!({"kind": "clean", "inputs": [big], "node": "macm4"}));
    assert_eq!(c, 202, "{j}");
    let resumed = wait_job(&master, j["id"].as_u64().unwrap(), &["done", "failed"]);
    assert_eq!(resumed["state"], "done", "{resumed}");
    let tin = &resumed["result"]["transfer"]["inputs"][0];
    assert_eq!((tin["cached"].as_bool(), tin["resumed_at"].as_u64(), tin["sent"].as_u64()), (Some(false), Some(half), Some(size - half)),
               "resumed where the worker's copy stopped: {tin}");
    assert_eq!(sha(&worker.dir.join("workshop/cluster/blobs").join(&h)), h, "the worker's copy hashes right");
    let out = resumed["result"]["output"].as_str().unwrap();
    assert_eq!(std::fs::read_to_string(out).unwrap(), body, "nothing dropped, and the copy back is whole");

    // 4. a private input: never sent by "any", taken when the worker is named
    let private = ds.join("secret.jsonl");
    rows(&private, &[r#"{"messages":[{"role":"user","content":"x"}],"private":true}"#]);
    let (c, e) = master.post("/v1/workshop/jobs", &json!({"kind": "clean", "inputs": [private], "node": "any"}));
    assert_eq!((c, e["error"]["type"].as_str()), (400, Some("private_data")), "{e}");
    assert!(e["error"]["message"].as_str().unwrap().contains("row 1 is marked private"), "{e}");
    let (c, j) = master.post("/v1/workshop/jobs", &json!({"kind": "clean", "inputs": [private], "node": "macm4"}));
    assert_eq!(c, 202, "named explicitly, it goes: {j}");
    assert_eq!(wait_job(&master, j["id"].as_u64().unwrap(), &["done", "failed"])["state"], "done");
    // and a plain one with "any" goes to the idlest machine that can: the worker (this machine has no Workshop command)
    let (c, j) = master.post("/v1/workshop/jobs", &json!({"kind": "clean", "inputs": [input], "node": "any"}));
    assert_eq!((c, j["node"].as_str()), (202, Some("macm4")), "{j}");
    assert_eq!(wait_job(&master, j["id"].as_u64().unwrap(), &["done", "failed"])["state"], "done");
    // a worker that is not paired, and a refused request, before anything is sent
    let (c, e) = master.post("/v1/workshop/jobs", &json!({"kind": "clean", "inputs": [input], "node": "pc9"}));
    assert_eq!(c, 404, "{e}");
    let (c, _) = master.post("/v1/workshop/jobs", &json!({"kind": "clean", "inputs": ["/etc/passwd.jsonl"], "node": "macm4"}));
    assert_eq!(c, 400, "the master's own checks first");
    std::fs::remove_dir_all(&tools).ok();
}

#[test]
fn a_cancel_on_the_master_reaches_the_workers_job() {
    let tools = std::env::temp_dir().join(format!("am2-cw-{}-b", std::process::id()));
    let (worker, master, _, _) = pair(&tools);
    let slow = master.dir.join("workshop/datasets/slow.jsonl");
    rows(&slow, &[r#"{"messages":[]}"#]);
    let (c, j) = master.post("/v1/workshop/jobs", &json!({"kind": "clean", "inputs": [slow], "node": "macm4"}));
    assert_eq!(c, 202, "{j}");
    let id = j["id"].as_u64().unwrap();
    // running there
    let t0 = Instant::now();
    let remote = loop {
        let (_, l) = worker.get("/v1/workshop/jobs", true);
        if let Some(x) = l["jobs"].as_array().unwrap().iter().find(|x| x["request"]["from_master"] == true && x["state"] == "running") {
            break x["id"].as_u64().unwrap();
        }
        assert!(t0.elapsed() < Duration::from_secs(30), "{l}");
        std::thread::sleep(Duration::from_millis(200));
    };
    let (c, v) = master.delete(&format!("/v1/workshop/jobs/{id}"));
    assert_eq!((c, v["state"].as_str()), (200, Some("cancelled")), "{v}");
    let t0 = Instant::now();
    loop {
        let (_, r) = worker.get(&format!("/v1/workshop/jobs/{remote}"), true);
        if r["state"] == "cancelled" {
            break;
        }
        assert!(t0.elapsed() < Duration::from_secs(15), "the cancel did not reach the worker: {r}");
        std::thread::sleep(Duration::from_millis(200));
    }
    let t0 = Instant::now();
    while !master.get(&format!("/v1/workshop/jobs/{id}"), true).1["log"].as_array().unwrap().iter()
        .any(|l| l.as_str().unwrap_or("").starts_with("cancelled on macm4 too")) {
        assert!(t0.elapsed() < Duration::from_secs(10));
        std::thread::sleep(Duration::from_millis(200));
    }
    assert_eq!(master.get(&format!("/v1/workshop/jobs/{id}"), true).1["state"], "cancelled");
    std::fs::remove_dir_all(&tools).ok();
}
