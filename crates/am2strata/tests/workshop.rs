//! The Workshop's server side end to end against the mock engine, with a stand-in for the `am2strata-workshop`
//! command (the real conversions are tested in strata-workshop against llama.cpp's own output): jobs queue and run one
//! at a time, report their output or their error, can be cancelled, and cannot touch files outside the Workshop's
//! folders. Only the server's own key may use it.

// (off Unix the two tests that need the `#!/bin/sh` stand-in are not built, and their helpers go unused)
#![cfg_attr(not(unix), allow(dead_code, unused_imports))]

mod common;

use common::*;
use serde_json::{json, Value};
use std::time::{Duration, Instant};

/// A stand-in for am2strata-workshop: prints the same JSON events. An input whose name holds "slow" takes 30 s (for
/// cancel), "fail" fails, anything else "converts" by writing the output file.
fn fake_workshop() -> std::path::PathBuf {
    let p = std::env::temp_dir().join(format!("fake-workshop-{}.sh", std::process::id()));
    std::fs::write(&p, r#"#!/bin/sh
# --json <kind> <input> ... --out <output> ...
shift; kind="$1"; input="$2"; out=""
while [ $# -gt 0 ]; do [ "$1" = "--out" ] && out="$2"; shift; done
echo "{\"event\":\"start\",\"what\":\"$kind $input\"}"
case "$input" in
  *slow*) sleep 30 ;;
  *fail*) echo '{"event":"error","message":"this model is not a Llama"}'; exit 1 ;;
esac
mkdir -p "$(dirname "$out")" 2>/dev/null; echo data > "$out"
echo "{\"event\":\"log\",\"message\":\"12 tensors\"}"
echo "{\"event\":\"done\",\"output\":\"$out\",\"kind\":\"gguf\",\"bytes\":5,\"seconds\":0.1}"
"#).unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o755)).unwrap();
    }
    p
}

fn wait_state(s: &Server, id: u64, want: &str) -> Value {
    let t0 = Instant::now();
    loop {
        let (_, j) = s.get(&format!("/v1/workshop/jobs/{id}"), true);
        if j["state"] == want {
            return j;
        }
        assert!(t0.elapsed() < Duration::from_secs(20), "job {id} is {} not {want}: {j}", j["state"]);
        std::thread::sleep(Duration::from_millis(100));
    }
}

// Unix only: the stand-in for am2strata-workshop is a `#!/bin/sh` script, which Windows cannot start as a program.
#[cfg(unix)]
#[test]
fn jobs_run_one_at_a_time_report_cancel_and_stay_inside_their_folders() {
    // the server is given this (the stand-in command) in its own environment; this process's is never changed
    let s = Server::start(&[Model::default()], &[ids("hi")], json!({"env": {"AM2STRATA_WORKSHOP_EXE": fake_workshop()}}));
    // an HF model folder inside the models folder
    let model = s.dir.join("hf-model");
    std::fs::create_dir_all(&model).unwrap();
    std::fs::write(model.join("config.json"), "{}").unwrap();
    std::fs::write(model.join("model.safetensors"), "x").unwrap();

    assert_eq!(s.get("/v1/workshop/jobs", false).0, 401, "needs the key");
    let (code, f) = s.get("/v1/workshop/files", true);
    assert_eq!(code, 200);
    assert!(f["hf_models"].as_array().unwrap().iter().any(|p| p["path"].as_str().unwrap().ends_with("hf-model")), "{f}");

    // a convert job runs and reports its output, in the Workshop folder by default
    let (code, j) = s.post("/v1/workshop/jobs", &json!({"kind": "convert", "input": model}));
    assert_eq!(code, 202, "{j}");
    let done = wait_state(&s, j["id"].as_u64().unwrap(), "done");
    let out = done["result"]["output"].as_str().unwrap();
    assert!(out.contains("/workshop/") && out.ends_with("hf-model-f16.gguf"), "{out}");
    assert!(std::path::Path::new(out).exists());
    assert!(done["log"].as_array().unwrap().iter().any(|l| l == "12 tensors"));

    // a failing job says why
    let bad = s.dir.join("fail-model");
    std::fs::create_dir_all(&bad).unwrap();
    std::fs::write(bad.join("config.json"), "{}").unwrap();
    let (_, j) = s.post("/v1/workshop/jobs", &json!({"kind": "convert", "input": bad}));
    let failed = wait_state(&s, j["id"].as_u64().unwrap(), "failed");
    assert_eq!(failed["error"], "this model is not a Llama");

    // one at a time: a slow job runs, the next waits queued; cancelling the slow one lets the next run
    let slow = s.dir.join("slow-model");
    std::fs::create_dir_all(&slow).unwrap();
    std::fs::write(slow.join("config.json"), "{}").unwrap();
    let (_, a) = s.post("/v1/workshop/jobs", &json!({"kind": "convert", "input": slow}));
    let (_, b) = s.post("/v1/workshop/jobs", &json!({"kind": "convert", "input": model, "outtype": "f32"}));
    let (a, b) = (a["id"].as_u64().unwrap(), b["id"].as_u64().unwrap());
    wait_state(&s, a, "running");
    assert_eq!(s.get(&format!("/v1/workshop/jobs/{b}"), true).1["state"], "queued");
    let (code, c) = s.delete(&format!("/v1/workshop/jobs/{a}"));
    assert_eq!((code, c["state"].as_str()), (200, Some("cancelled")));
    wait_state(&s, b, "done");
    assert_eq!(s.delete(&format!("/v1/workshop/jobs/{b}")).0, 409, "a finished job cannot be cancelled");

    // nothing outside the Workshop's folders, in or out
    let (code, e) = s.post("/v1/workshop/jobs", &json!({"kind": "convert", "input": "/etc"}));
    assert_eq!(code, 400);
    let m = e["error"]["message"].as_str().unwrap();
    assert!(m.contains("outside the Workshop") || m.contains("never touches"), "{e}");
    let (code, _) = s.post("/v1/workshop/jobs", &json!({"kind": "convert", "input": model, "output": "/tmp/x.gguf"}));
    assert_eq!(code, 400);
    let (code, _) = s.post("/v1/workshop/jobs", &json!({"kind": "convert", "input": model.join("../../..")}));
    assert_eq!(code, 400, "no way out with ..");
    // a symlink inside a Workshop folder that points outside it is outside
    #[cfg(unix)]
    {
        let outside = std::env::temp_dir().join(format!("am2-ws-outside-{}", std::process::id()));
        std::fs::create_dir_all(&outside).unwrap();
        std::fs::write(outside.join("config.json"), "{}").unwrap();
        std::os::unix::fs::symlink(&outside, s.dir.join("sneaky")).unwrap();
        let (code, e) = s.post("/v1/workshop/jobs", &json!({"kind": "convert", "input": s.dir.join("sneaky")}));
        assert_eq!(code, 400, "{e}");
        let (code, _) = s.post("/v1/workshop/jobs", &json!({"kind": "convert", "input": model, "output": s.dir.join("sneaky/out.gguf")}));
        assert_eq!(code, 400, "an output through the link is outside too");
        let _ = std::fs::remove_dir_all(&outside);
    }
    // a refused output makes no folders, and a relative path is refused
    let nowhere = std::env::temp_dir().join(format!("am2-ws-nowhere-{}", std::process::id()));
    let (code, _) = s.post("/v1/workshop/jobs", &json!({"kind": "convert", "input": model, "output": nowhere.join("a/b/out.gguf")}));
    assert_eq!(code, 400);
    assert!(!nowhere.exists(), "no folder made outside the Workshop's folders");
    assert_eq!(s.post("/v1/workshop/jobs", &json!({"kind": "convert", "input": "hf-model"})).0, 400, "relative path");
    // and requests that make no sense are refused before anything runs
    assert_eq!(s.post("/v1/workshop/jobs", &json!({"kind": "quantize", "input": out, "type": "q3_x"})).0, 400);
    assert_eq!(s.post("/v1/workshop/jobs", &json!({"kind": "merge", "base": model, "adapter": model})).0, 400, "not an adapter");
    assert_eq!(s.post("/v1/workshop/jobs", &json!({"kind": "train"})).0, 400);
    // a loop: blocks inside the model, 2-16 passes, not a model that is looped already
    let deep = s.dir.join("hf-deep");
    std::fs::create_dir_all(&deep).unwrap();
    std::fs::write(deep.join("config.json"), r#"{"num_hidden_layers": 4}"#).unwrap();
    for (bad, why) in [(json!({"start": 1, "end": 5}), "past the last block"), (json!({"start": 2, "end": 2}), "empty"),
                       (json!({"start": 0, "end": 2, "passes": 17}), "too many passes"), (json!({"end": 2}), "no start")] {
        let mut req = json!({"kind": "loop", "input": deep});
        req.as_object_mut().unwrap().extend(bad.as_object().unwrap().clone());
        assert_eq!(s.post("/v1/workshop/jobs", &req).0, 400, "{why}");
    }
    let looped = s.dir.join("hf-looped");
    std::fs::create_dir_all(&looped).unwrap();
    std::fs::write(looped.join("config.json"), r#"{"num_hidden_layers": 4, "prastara_loop": {"start": 1, "end": 3, "passes": 2}}"#).unwrap();
    let (code, e) = s.post("/v1/workshop/jobs", &json!({"kind": "loop", "input": looped, "start": 0, "end": 2}));
    assert_eq!(code, 400);
    assert!(e["error"]["message"].as_str().unwrap().contains("already looped"), "{e}");
    let (_, list) = s.get("/v1/workshop/jobs", true);
    assert_eq!(list["jobs"].as_array().unwrap().len(), 4);
}

/// A small but complete llama GGUF: the metadata and BPE tokenizer an engine needs (tokens, types, merges, bos/eos).
fn small_llama_gguf(path: &std::path::Path) {
    small_llama_gguf_with(path, &[])
}

/// [`small_llama_gguf`] with boolean keys added (the Workshop's prastara.* marks).
fn small_llama_gguf_with(path: &std::path::Path, marks: &[(&str, bool)]) {
    use strata_pack::gguf_write::{Kv, TensorSpec, Writer};
    let tokens: Vec<String> = ["<s>", "</s>", "t", "o", "to", "a"].iter().map(|s| s.to_string()).collect();
    let kv: Vec<(&str, Kv)> = vec![
        ("general.architecture", Kv::Str("llama".into())), ("general.name", Kv::Str("Small Llama".into())),
        ("llama.context_length", Kv::U32(2048)), ("llama.embedding_length", Kv::U32(4)), ("llama.block_count", Kv::U32(1)),
        ("llama.feed_forward_length", Kv::U32(8)), ("llama.attention.head_count", Kv::U32(1)),
        ("llama.attention.head_count_kv", Kv::U32(1)), ("llama.rope.dimension_count", Kv::U32(4)),
        ("llama.attention.layer_norm_rms_epsilon", Kv::F32(1e-5)), ("llama.vocab_size", Kv::U32(6)),
        ("tokenizer.ggml.model", Kv::Str("gpt2".into())), ("tokenizer.ggml.pre", Kv::Str("gpt-2".into())),
        ("tokenizer.ggml.tokens", Kv::Strings(tokens)), ("tokenizer.ggml.token_type", Kv::Ints(vec![3, 3, 1, 1, 1, 1])),
        ("tokenizer.ggml.merges", Kv::Strings(vec!["t o".into()])),
        ("tokenizer.ggml.bos_token_id", Kv::U32(0)), ("tokenizer.ggml.eos_token_id", Kv::U32(1)),
    ];
    let mut kv = kv;
    kv.extend(marks.iter().map(|(k, v)| (*k, Kv::Bool(*v))));
    let raw: Vec<u8> = (0..24u32).flat_map(|i| (i as f32).to_le_bytes()).collect();
    let mut w = Writer::create_typed(path, &kv, vec![TensorSpec { name: "token_embd.weight".into(), shape: vec![6, 4], ggml_type: 0, nbytes: 96 }]).unwrap();
    w.tensor_data(&raw).unwrap();
    w.end_tensor().unwrap();
    w.finish().unwrap();
}

#[test]
fn a_gguf_made_strata_ready_appears_in_the_model_list_and_is_not_overwritten() {
    // a stand-in for the Rust engine (a llama model runs on it), so the model is listed on any machine, CI included
    let home = std::env::temp_dir().join(format!("am2-ws-home-{}", std::process::id()));
    let engine = home.join("crates/strata-hybrid/target/release/am2strata-hybrid");
    std::fs::create_dir_all(engine.parent().unwrap()).unwrap();
    std::fs::write(&engine, "#!/bin/sh\n").unwrap();
    let s = Server::start(&[Model::default()], &[ids("hi")], json!({"env": {"AM2STRATA_HOME": home}}));
    // a GGUF llama.cpp itself would not load (a BPE vocabulary without merges) is refused, saying why
    let src = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../strata-workshop/tests/fixtures/tiny-llama.f32.gguf");
    let nomerges = s.dir.join("workshop/tiny-llama.f32.gguf");
    std::fs::create_dir_all(nomerges.parent().unwrap()).unwrap();
    std::fs::copy(&src, &nomerges).unwrap();
    let (_, j) = s.post("/v1/workshop/jobs", &json!({"kind": "strata", "input": nomerges}));
    let failed = wait_state(&s, j["id"].as_u64().unwrap(), "failed");
    assert!(failed["error"].as_str().unwrap().contains("merges"), "{failed}");

    let gguf = s.dir.join("workshop/small-llama.gguf");
    small_llama_gguf(&gguf);
    let (code, j) = s.post("/v1/workshop/jobs", &json!({"kind": "strata", "input": gguf, "name": "my-tiny", "display_name": "My Tiny Llama"}));
    assert_eq!(code, 202, "{j}");
    let done = wait_state(&s, j["id"].as_u64().unwrap(), "done");
    assert_eq!(done["result"]["architecture"], "llama");
    let cfg_path = s.dir.join("strata-my-tiny.json");
    let cfg: Value = serde_json::from_slice(&std::fs::read(&cfg_path).unwrap()).unwrap();
    assert_eq!((cfg["model_name"].as_str(), cfg["display_name"].as_str()), (Some("my-tiny"), Some("My Tiny Llama")));
    assert!(std::path::Path::new(done["result"]["tokenizer"].as_str().unwrap()).join("tokenizer.json").exists(), "tokenizer folder");
    let (_, l) = s.get("/api/models", true);
    assert!(l["models"].as_array().unwrap().iter().any(|m| m["id"] == "my-tiny"), "listed: {l}");
    // the same name again is refused before anything runs; with overwrite it is allowed
    let (code, e) = s.post("/v1/workshop/jobs", &json!({"kind": "strata", "input": gguf, "name": "my-tiny"}));
    assert_eq!(code, 400);
    assert!(e["error"]["message"].as_str().unwrap().contains("already installed"), "{e}");
    let (code, _) = s.post("/v1/workshop/jobs", &json!({"kind": "strata", "input": gguf, "name": "my-tiny", "overwrite": true}));
    assert_eq!(code, 202);
    // an abliterated GGUF (the Workshop's prastara.* marks): named -abliterated, marked private and uncensored
    let marked = s.dir.join("workshop/small-llama-marked.gguf");
    small_llama_gguf_with(&marked, &[("prastara.private", true), ("prastara.abliterated", true)]);
    let (code, j) = s.post("/v1/workshop/jobs", &json!({"kind": "strata", "input": marked, "name": "mula"}));
    assert_eq!(code, 202, "{j}");
    let done = wait_state(&s, j["id"].as_u64().unwrap(), "done");
    assert_eq!((done["result"]["model"].as_str(), done["result"]["private"].as_bool(), done["result"]["uncensored"].as_bool()),
               (Some("mula-abliterated"), Some(true), Some(true)), "{done}");
    let cfg: Value = serde_json::from_slice(&std::fs::read(s.dir.join("strata-mula-abliterated.json")).unwrap()).unwrap();
    assert_eq!((&cfg["private"], &cfg["uncensored"], &cfg["abliterated"]), (&json!(true), &json!(true), &json!(true)), "{cfg}");
    assert!(cfg["display_name"].as_str().unwrap().contains("abliterated"), "{cfg}");
    let (_, l) = s.get("/api/models", true);
    let m = l["models"].as_array().unwrap().iter().find(|m| m["id"] == "mula-abliterated").cloned().expect("listed");
    assert_eq!((&m["private"], &m["uncensored"]), (&json!(true), &json!(true)), "{m}");
    // a plain one is neither
    let plain = l["models"].as_array().unwrap().iter().find(|m| m["id"] == "my-tiny").cloned().unwrap();
    assert_eq!((&plain["private"], &plain["uncensored"]), (&json!(false), &json!(false)));
}


/// A stand-in for `am2strata-workshop datagen`: notes its arguments and the server address and key it was given (in
/// its environment), reports progress, and writes three records.
fn fake_datagen(dir: &std::path::Path) -> std::path::PathBuf {
    let p = dir.join("fake-datagen.sh");
    std::fs::create_dir_all(dir).unwrap();
    std::fs::write(&p, r#"#!/bin/sh
here="$(dirname "$0")"
case " $* " in *" --estimate "*)
  echo "$@" > "$here/estimate-args.txt"
  echo "{\"event\":\"estimate\",\"kind\":\"$2\",\"lines\":[\"needs about 11 GB a card\"],\"estimate\":{\"card_bytes\":11000000000,\"disk_bytes\":16000000000}}"
  exit 0;;
esac
if [ "$1" = "--json" ] && [ "$2" = "library" ]; then
  echo "$@" > "$here/library-args.txt"
  [ "$3" = "init" ] && { mkdir -p "$6"; echo 'base: x' > "$6/soup.yaml"; echo '# x' > "$6/DATA.md"; }
  echo '{"event":"library","recipes":[{"name":"qwen3-8b-sft","abilities":["text"],"workshop":{"status":"trains"}}],"sources":[{"name":"Soup"}]}'
  exit 0
fi
if [ "$1" = "--json" ] && [ "$2" = "profile" ]; then
  echo "$@" > "$here/profile-args.txt"
  echo '{"event":"profile","fits_ram":true,"fits_gpu":[{"index":0,"fits":true}],"lines":["fits"]}'
  exit 0
fi
if [ "$1" = "--json" ] && [ "$2" = "train" ]; then
  echo "$@" > "$here/train-args.txt"
  echo '{"event":"start","what":"train"}'
  echo '{"event":"progress","done":1,"total":2,"step":1,"loss":3.5,"lr":0.0002,"eta":1,"message":"loss 3.5"}'
  echo '{"event":"log","message":"held-out loss 2.8 at step 1","step":1,"eval_loss":2.8}'
  echo '{"event":"progress","done":2,"total":2,"step":2,"loss":1.9,"lr":0.0001,"eta":0,"message":"loss 1.9"}'
  echo '{"event":"done","output":"x","kind":"adapter","bytes":10,"seconds":0.1}'
  exit 0
fi
if [ "$1" = "--json" ] && { [ "$2" = "upcycle" ] || [ "$2" = "export-moe" ]; }; then
  echo "$@" > "$here/$2-args.txt"
  out=""; prev=""
  for a in "$@"; do [ "$prev" = "--out" ] && out="$a"; prev="$a"; done
  echo "{\"event\":\"start\",\"what\":\"$2\"}"
  if [ "$2" = "upcycle" ]; then
    mkdir -p "$out"; echo '{"model_type": "qwen3", "prastara_moe": {"experts": 10, "top_k": 1}}' > "$out/config.json"; echo x > "$out/model.safetensors"
    echo "{\"event\":\"done\",\"output\":\"$out\",\"kind\":\"hf-model\",\"experts\":10,\"top_k\":1,\"total_parameters\":57200000000,\"active_parameters\":8200000000,\"bytes\":10,\"seconds\":0.1}"
  else
    mkdir -p "$(dirname "$out")"; echo gguf > "$out"
    echo "{\"event\":\"done\",\"output\":\"$out\",\"kind\":\"gguf\",\"experts\":10,\"top_k\":1,\"trained_blocks\":36,\"bytes\":5,\"seconds\":0.1}"
  fi
  exit 0
fi
if [ "$1" = "--json" ] && { [ "$2" = "regress" ] || [ "$2" = "grade-train" ]; }; then
  echo "$@" > "$here/$2-args.txt"
  printf '%s\n%s\n' "$AM2STRATA_URL" "$AM2STRATA_API_KEY" > "$here/$2-env.txt"
  out=""; prev=""
  for a in "$@"; do [ "$prev" = "--out" ] && out="$a"; prev="$a"; done
  mkdir -p "$out"
  echo "{\"event\":\"start\",\"what\":\"$2\"}"
  if [ "$2" = "regress" ]; then
    echo '{"results":[],"regressions":[]}' > "$out/report.json"
    echo '{"event":"progress","done":1,"total":2,"message":"humaneval at 1 pass"}'
    echo "{\"event\":\"done\",\"output\":\"$out/report.json\",\"kind\":\"report\",\"regressions\":[{\"test\":\"humaneval\",\"passes\":2,\"delta\":-0.2}],\"report\":{\"results\":[{\"test\":\"humaneval\",\"passes\":2,\"metric\":\"pass@1\",\"value\":0.3}],\"regressions\":[{\"test\":\"humaneval\",\"passes\":2}]},\"bytes\":10,\"seconds\":0.1}"
  else
    echo '{"round":1,"promoted":true}' > "$out/rounds.jsonl"
    echo '{"event":"progress","done":1,"total":2,"step":1,"loss":2.5,"message":"round 1: training, loss 2.5"}'
    echo "{\"event\":\"done\",\"output\":\"$out\",\"kind\":\"grade-train\",\"promoted\":\"$out/round-1/adapter\",\"rounds\":[{\"round\":1,\"kept\":3,\"regressions\":0,\"promoted\":true}],\"bytes\":10,\"seconds\":0.1}"
  fi
  exit 0
fi
if [ "$1" = "--json" ] && [ "$2" = "private-data" ]; then
  echo "$@" > "$here/private-data-args.txt"
  printf '%s\n%s\n' "$AM2STRATA_URL" "$AM2STRATA_API_KEY" > "$here/private-data-env.txt"
  name=""; data=""; prev=""
  for a in "$@"; do [ "$prev" = "--name" ] && name="$a"; [ "$prev" = "--data-dir" ] && data="$a"; prev="$a"; done
  mkdir -p "$data/private/$name"
  echo '{"event":"start","what":"private-data"}'
  echo "{\"event\":\"done\",\"output\":\"$data/private/$name\",\"kind\":\"private-dataset\",\"private\":true,\"bytes\":1,\"seconds\":0.1}"
  exit 0
fi
if [ "$1" = "--json" ] && [ "$2" = "abliterate" ]; then
  echo "$@" > "$here/abliterate-args.txt"
  out=""; prev=""
  for a in "$@"; do [ "$prev" = "--out" ] && out="$a"; prev="$a"; done
  mkdir -p "$out"; echo '{"prastara_private": true}' > "$out/config.json"; echo x > "$out/model.safetensors"
  echo '{"event":"start","what":"abliterate"}'
  echo '{"event":"estimate","kind":"abliterate","lines":["memory: about 11 GB on the busiest of 2 GPU(s)"],"estimate":{"card_bytes":11000000000}}'
  echo '{"event":"layers","chosen":1,"layers":[{"block":0,"pass":0,"separation":0.5},{"block":1,"pass":0,"separation":2.5}],"message":"direction from block 1 pass 0"}'
  echo '{"event":"check","which":"before","result":{"n":4,"harmful_refusal_rate":0.75,"harmless_refusal_rate":0.0,"harmless_loss":1.2}}'
  echo '{"event":"check","which":"after","result":{"n":4,"harmful_refusal_rate":0.0,"harmless_refusal_rate":0.0,"harmless_loss":1.25}}'
  echo "{\"event\":\"done\",\"output\":\"$out\",\"kind\":\"hf-model\",\"private\":true,\"bytes\":10,\"seconds\":0.1}"
  exit 0
fi
echo "$@" > "$here/args.txt"
printf '%s\n%s\n' "$AM2STRATA_URL" "$AM2STRATA_API_KEY" > "$here/env.txt"
out=""; prev=""
for a in "$@"; do [ "$prev" = "--out" ] && out="$a"; prev="$a"; done
echo '{"event":"start","what":"qa data"}'
echo '{"event":"progress","done":1,"total":2,"message":"2 records so far"}'
mkdir -p "$(dirname "$out")"
for q in one two three; do
  echo "{\"messages\":[{\"role\":\"user\",\"content\":\"question $q\"},{\"role\":\"assistant\",\"content\":\"answer $q\"}],\"meta\":{\"score\":5}}" >> "$out"
done
echo '# card' > "${out%.jsonl}.card.md"
echo '{"event":"progress","done":2,"total":2,"message":"3 records so far"}'
echo "{\"event\":\"done\",\"output\":\"$out\",\"kind\":\"dataset\",\"bytes\":300,\"seconds\":0.1}"
"#).unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o755)).unwrap();
    }
    p
}

// Unix only: the stand-in for am2strata-workshop is a `#!/bin/sh` script, which Windows cannot start as a program.
#[cfg(unix)]
#[test]
fn training_data_jobs_get_the_server_by_environment_and_their_datasets_can_be_reviewed_and_edited() {
    let tools = std::env::temp_dir().join(format!("am2-ws-datagen-{}", std::process::id()));
    let exe = fake_datagen(&tools);
    let s = Server::start(&[Model::default()], &[ids("hi")], json!({"env": {"AM2STRATA_WORKSHOP_EXE": exe}}));

    // a document comes in through the upload endpoint, under a safe name, never over another
    let up = |name: &str, body: &str| {
        let r = reqwest::blocking::Client::new().post(format!("{}/v1/workshop/uploads", s.base)).query(&[("name", name)])
            .bearer_auth(KEY).body(body.to_string()).send().unwrap();
        (r.status().as_u16(), r.json::<Value>().unwrap_or_default())
    };
    let (code, a) = up("../../notes.md", "# Notes\nsome text");
    assert_eq!(code, 200, "{a}");
    let doc = a["path"].as_str().unwrap().to_string();
    assert!(doc.ends_with("/workshop/uploads/notes.md"), "{doc}");
    let (_, b) = up("notes.md", "other");
    assert!(b["path"].as_str().unwrap().ends_with("/workshop/uploads/notes-2.md"), "{b}");
    assert_eq!(std::fs::read_to_string(&doc).unwrap(), "# Notes\nsome text");
    let r = reqwest::blocking::Client::new().post(format!("{}/v1/workshop/uploads?name=x.md", s.base)).body("x").send().unwrap();
    assert_eq!(r.status().as_u16(), 401, "uploads need the key");

    // checked before anything runs: documents are needed, inside the Workshop's folders, numbers in range
    for (bad, why) in [(json!({"kind": "datagen", "recipe": "qa"}), "needs \"sources\""),
                       (json!({"kind": "datagen", "sources": ["/etc/hostname"]}), "/etc/hostname"),
                       (json!({"kind": "datagen", "sources": [doc], "count": 0}), "\"count\""),
                       (json!({"kind": "datagen", "recipe": "instruct"}), "\"seeds\""),
                       (json!({"kind": "datagen", "recipe": "poems", "sources": [doc]}), "\"recipe\""),
                       (json!({"kind": "datagen", "recipe": "indic", "sources": [doc]}), "built-in catalogue"),
                       (json!({"kind": "datagen", "recipe": "translate", "check_code": true}), "code recipe")] {
        let (code, e) = s.post("/v1/workshop/jobs", &bad);
        assert_eq!(code, 400, "{bad}: {e}");
        assert!(e["error"]["message"].as_str().unwrap().contains(why), "{bad}: {e}");
    }

    let (code, j) = s.post("/v1/workshop/jobs", &json!({"kind": "datagen", "sources": [doc], "count": 3, "min_score": 4, "topic": "notes"}));
    assert_eq!(code, 202, "{j}");
    let done = wait_state(&s, j["id"].as_u64().unwrap(), "done");
    let out = done["output"].as_str().unwrap().to_string();
    assert!(out.ends_with("/workshop/datasets/notes-qa.jsonl"), "{out}");
    assert_eq!(done["progress"]["done"], 2, "{done}");
    assert!(done["log"].as_array().unwrap().iter().any(|l| l == "[2/2] 3 records so far"), "{done}");
    // the server's address and key reached it in its environment, and the key is nowhere on its command line
    let env = std::fs::read_to_string(tools.join("env.txt")).unwrap();
    assert_eq!(env, format!("http://127.0.0.1:{}\n{KEY}\n", s.port));
    let args = std::fs::read_to_string(tools.join("args.txt")).unwrap();
    assert!(!args.contains(KEY), "{args}");
    assert!(args.contains("--min-score 4") && args.contains("--count 3") && args.contains("--topic notes"), "{args}");
    let (_, f) = s.get("/v1/workshop/files", true);
    assert!(f["datasets"].as_array().unwrap().iter().any(|d| d["path"] == out.as_str()), "{f}");
    assert!(f["uploads"].as_array().unwrap().len() >= 2, "{f}");
    // a second run does not write over the first
    let (_, j2) = s.post("/v1/workshop/jobs", &json!({"kind": "datagen", "sources": [doc]}));
    assert!(j2["output"].as_str().unwrap().ends_with("notes-qa-2.jsonl"), "{j2}");
    wait_state(&s, j2["id"].as_u64().unwrap(), "done");

    // the built-in recipes need no documents; their draw, languages and compile check reach the command
    let (code, j3) = s.post("/v1/workshop/jobs", &json!({"kind": "datagen", "recipe": "code", "seed": 7, "check_code": true,
                                                         "language": "Rust", "count": 2}));
    assert_eq!(code, 202, "{j3}");
    assert!(j3["output"].as_str().unwrap().ends_with("/workshop/datasets/data-code.jsonl"), "{j3}");
    wait_state(&s, j3["id"].as_u64().unwrap(), "done");
    let args = std::fs::read_to_string(tools.join("args.txt")).unwrap();
    assert!(args.contains("--recipe code") && args.contains("--seed 7") && args.contains("--check-code")
            && args.contains("--language Rust") && !args.contains("--source"), "{args}");
    for recipe in ["indic", "translate"] {
        let (code, j) = s.post("/v1/workshop/jobs", &json!({"kind": "datagen", "recipe": recipe, "language": "hi,ta"}));
        assert_eq!(code, 202, "{recipe}: {j}");
        wait_state(&s, j["id"].as_u64().unwrap(), "done");
    }
    let args = std::fs::read_to_string(tools.join("args.txt")).unwrap();
    assert!(args.contains("--recipe translate") && args.contains("--language hi,ta") && !args.contains("--check-code"), "{args}");

    // the model library: listed through the workshop command, and a recipe set up in the Workshop's recipes folder
    let (code, lib) = s.get("/v1/workshop/library", true);
    assert_eq!(code, 200, "{lib}");
    assert_eq!(lib["recipes"][0]["name"], "qwen3-8b-sft");
    assert_eq!(s.get("/v1/workshop/library", false).0, 401);
    let (code, made) = s.post("/v1/workshop/library", &json!({"name": "qwen3-8b-sft"}));
    assert_eq!(code, 200, "{made}");
    assert!(made["config"].as_str().unwrap().ends_with("/workshop/recipes/qwen3-8b-sft/soup.yaml"), "{made}");
    assert!(std::path::Path::new(made["guide"].as_str().unwrap()).exists());
    let (code, _) = s.post("/v1/workshop/library", &json!({"name": "../../etc"}));
    assert!(code == 200 || code == 400);
    let largs = std::fs::read_to_string(tools.join("library-args.txt")).unwrap();
    assert!(!largs.contains(".."), "a name is made safe before it reaches the command: {largs}");

    // review: a page of records with the card
    let q = |p: &str| format!("/v1/workshop/dataset?path={}&{p}", urlencode(&out));
    let (code, page) = s.get(&q("offset=1&limit=1"), true);
    assert_eq!(code, 200, "{page}");
    assert_eq!((page["total"].as_u64(), page["records"].as_array().unwrap().len()), (Some(3), 1));
    assert_eq!(page["records"][0]["i"], 1);
    assert_eq!(page["records"][0]["record"]["messages"][0]["content"], "question two");
    assert_eq!(page["card"], "# card\n");
    // edit one, delete another; a bad record is refused
    let (code, e) = s.post("/v1/workshop/dataset", &json!({"path": out, "replace": {"0": {"messages": "x"}}}));
    assert_eq!(code, 400, "{e}");
    let fixed = json!({"messages": [{"role": "user", "content": "question one?"}, {"role": "assistant", "content": "a better answer"}], "meta": {"score": 5}});
    let (code, r) = s.post("/v1/workshop/dataset", &json!({"path": out, "replace": {"0": fixed}, "remove": [1]}));
    assert_eq!((code, r["total"].as_u64()), (200, Some(2)), "{r}");
    let lines: Vec<Value> = std::fs::read_to_string(&out).unwrap().lines().map(|l| serde_json::from_str(l).unwrap()).collect();
    assert_eq!(lines, vec![fixed, serde_json::from_str::<Value>(r#"{"messages":[{"role":"user","content":"question three"},{"role":"assistant","content":"answer three"}],"meta":{"score":5}}"#).unwrap()]);
    // only datasets inside the Workshop's folders
    let (code, _) = s.get("/v1/workshop/dataset?path=/etc/passwd", true);
    assert_eq!(code, 400);
    assert_eq!(s.get(&q(""), false).0, 401);
}

fn urlencode(s: &str) -> String {
    s.bytes().map(|b| if b.is_ascii_alphanumeric() || b"-_.~/".contains(&b) { (b as char).to_string() } else { format!("%{b:02X}") }).collect()
}

/// From another device: a model file streamed to disk whatever its size, a whole folder file by file, and the
/// server's folders browsed, all inside the Workshop's folders.
#[test]
fn uploads_stream_files_and_folders_and_the_workshop_folders_can_be_browsed() {
    let s = Server::start(&[Model::default()], &[ids("hi")], json!({}));
    let c = reqwest::blocking::Client::new();
    let put = |q: &[(&str, &str)], body: Vec<u8>| {
        let r = c.put(format!("{}/v1/workshop/upload", s.base)).query(q).bearer_auth(common::KEY).body(body).send().unwrap();
        (r.status().as_u16(), r.json::<Value>().unwrap_or(Value::Null))
    };
    // a file larger than the server's 256 MB body limit for other requests: streamed, never held whole
    let big = vec![7u8; (257 << 20) as usize];
    let (code, j) = put(&[("name", "model.gguf")], big);
    assert_eq!(code, 200, "{j}");
    let p = std::path::PathBuf::from(j["path"].as_str().unwrap());
    assert_eq!(std::fs::metadata(&p).unwrap().len(), 257 << 20);
    assert!(p.ends_with("workshop/uploads/model.gguf"), "{}", p.display());
    assert!(!p.with_file_name("model.gguf.part").exists(), "the .part file is renamed when whole");
    std::fs::remove_file(&p).unwrap();
    // a folder: made empty, then its files by their paths inside it
    let (code, f) = s.post("/v1/workshop/upload-folder?name=my-model", &json!({}));
    assert_eq!(code, 200, "{f}");
    let folder = f["folder"].as_str().unwrap().to_string();
    assert_eq!(put(&[("folder", &folder), ("rel", "config.json")], b"{}".to_vec()).0, 200);
    assert_eq!(put(&[("folder", &folder), ("rel", "sub/model.safetensors")], b"x".to_vec()).0, 200);
    assert!(std::path::Path::new(&folder).join("sub/model.safetensors").exists());
    for bad in ["../escape.txt", "a/../../b", "/etc/passwd", ""] {
        assert_eq!(put(&[("folder", &folder), ("rel", bad)], b"x".to_vec()).0, 400, "{bad:?}");
    }
    assert_eq!(put(&[("folder", &folder), ("rel", "config.json")], b"{}".to_vec()).0, 400, "never over a file already there");
    assert_eq!(put(&[("folder", "/tmp"), ("rel", "x.txt")], b"x".to_vec()).0, 400, "only into an upload folder");
    assert_eq!(c.put(format!("{}/v1/workshop/upload?name=x.txt", s.base)).body("x").send().unwrap().status().as_u16(), 401, "needs the key");
    // browsing: the roots, then a folder, its entries marked
    let (code, roots) = s.get("/v1/workshop/browse", true);
    assert_eq!(code, 200, "{roots}");
    assert!(roots["entries"].as_array().unwrap().iter().any(|e| e["name"] == "uploads" && e["where"].as_str().unwrap().ends_with("workshop")), "{roots}");
    let (code, up) = s.get(&format!("/v1/workshop/browse?path={}", urlencode(std::path::Path::new(&folder).parent().unwrap().to_str().unwrap())), true);
    assert_eq!(code, 200, "{up}");
    let e = up["entries"].as_array().unwrap().iter().find(|e| e["name"] == "my-model").expect("the uploaded folder");
    assert_eq!((e["dir"].as_bool(), e["kind"].as_str()), (Some(true), Some("hf-model")));
    assert_eq!(s.get("/v1/workshop/browse?path=/etc", true).0, 400, "nothing outside the Workshop's folders");
}

/// Sensitive paths are refused wherever they are, even inside a Workshop folder: the system's areas, key and
/// credential folders, and secret files; as job inputs, as uploads (names and paths inside a folder) and in listings.
#[test]
fn sensitive_paths_are_never_read_written_uploaded_or_listed() {
    let s = Server::start(&[Model::default()], &[ids("hi")], json!({}));
    let c = reqwest::blocking::Client::new();
    let put = |q: &[(&str, &str)]| c.put(format!("{}/v1/workshop/upload", s.base)).query(q).bearer_auth(common::KEY).body("x").send().unwrap().status().as_u16();
    let ws = s.dir.join("workshop");
    // inside the Workshop folder, still refused: a key folder and a secret file
    std::fs::create_dir_all(ws.join(".ssh")).unwrap();
    std::fs::write(ws.join(".ssh/id_rsa"), "k").unwrap();
    std::fs::write(ws.join("server.pem"), "k").unwrap();
    std::fs::create_dir_all(ws.join("model")).unwrap();
    std::fs::write(ws.join("model/config.json"), "{}").unwrap();
    for p in [ws.join(".ssh/id_rsa"), ws.join("server.pem"), ws.join(".ssh"), "/etc/passwd".into(), "/proc/self/environ".into()] {
        let (code, e) = s.post("/v1/workshop/jobs", &json!({"kind": "quantize", "input": p, "type": "q8_0"}));
        assert_eq!(code, 400, "{} {e}", p.display());
    }
    let (_, e) = s.post("/v1/workshop/jobs", &json!({"kind": "quantize", "input": ws.join("server.pem"), "type": "q8_0"}));
    assert!(e["error"]["message"].as_str().unwrap().contains("never touches"), "{e}");
    // uploads: secret names and paths are refused, whether a file or inside a folder
    for name in ["id_ed25519", ".env", "passwd", "cert.key", "node-tls.json"] {
        assert_eq!(put(&[("name", name)]), 400, "{name}");
    }
    let (_, f) = s.post("/v1/workshop/upload-folder?name=m", &json!({}));
    let folder = f["folder"].as_str().unwrap().to_string();
    for rel in ["etc/passwd", "ssh_dir/id_rsa", "keys/server.pem", "a/.env"] {
        assert_eq!(put(&[("folder", &folder), ("rel", rel)]), 400, "{rel}");
    }
    assert_eq!(put(&[("folder", &folder), ("rel", "config.json")]), 200, "an ordinary file still goes in");
    assert_eq!(s.post("/v1/workshop/upload-folder?name=.ssh", &json!({})).0, 400);
    // listings leave them out
    let (_, b) = s.get(&format!("/v1/workshop/browse?path={}", urlencode(ws.to_str().unwrap())), true);
    let names: Vec<&str> = b["entries"].as_array().unwrap().iter().filter_map(|e| e["name"].as_str()).collect();
    assert!(names.contains(&"model") && !names.contains(&"server.pem") && !names.contains(&".ssh"), "{names:?}");
    assert_eq!(s.get(&format!("/v1/workshop/browse?path={}", urlencode(ws.join(".ssh").to_str().unwrap())), true).0, 400);
}

/// Training from the Workshop page: the request's settings become the train command's arguments (a GPU run in
/// bf16 by default), the estimate runs `profile` with the same ones, the job keeps the loss curve, and bad
/// requests are refused before anything runs.
#[cfg(unix)]
#[test]
fn training_runs_from_the_workshop_with_an_estimate_and_its_loss_curve() {
    let tools = std::env::temp_dir().join(format!("am2-ws-train-{}", std::process::id()));
    let exe = fake_datagen(&tools);
    let s = Server::start(&[Model::default()], &[ids("hi")], json!({"env": {"AM2STRATA_WORKSHOP_EXE": exe}}));
    let model = s.dir.join("workshop/base-model");
    std::fs::create_dir_all(&model).unwrap();
    std::fs::write(model.join("config.json"), "{}").unwrap();
    let data = s.dir.join("workshop/datasets/d.jsonl");
    std::fs::create_dir_all(data.parent().unwrap()).unwrap();
    std::fs::write(&data, "{}\n").unwrap();
    let req = json!({"kind": "train", "model": model, "data": data, "devices": "cpu", "rank": 16, "epochs": 1, "targets": "q_proj,v_proj",
                     "gradient_checkpointing": true});
    // the estimate
    let (code, est) = s.post("/v1/workshop/estimate", &req);
    assert_eq!(code, 200, "{est}");
    assert_eq!(est["fits_ram"], true);
    let pa = std::fs::read_to_string(tools.join("profile-args.txt")).unwrap();
    assert!(pa.contains("--rank 16") && pa.contains("--targets q_proj,v_proj") && pa.contains("--gradient-checkpointing"), "{pa}");
    // refused before anything runs
    for (bad, why) in [(json!({"kind": "train", "data": data}), "no model"), (json!({"kind": "train", "model": model, "data": data, "devices": "cuda:x"}), "bad devices"),
                       (json!({"kind": "train", "model": model, "data": data, "rank": 0}), "rank 0"),
                       (json!({"kind": "train", "model": model, "data": data, "targets": "q_proj,evil"}), "unknown target")] {
        assert_eq!(s.post("/v1/workshop/jobs", &bad).0, 400, "{why}");
    }
    // the run: its curve kept, its adapter in the Workshop's adapters folder
    let (code, j) = s.post("/v1/workshop/jobs", &req);
    assert_eq!(code, 202, "{j}");
    let done = wait_state(&s, j["id"].as_u64().unwrap(), "done");
    let (_, d) = s.get(&format!("/v1/workshop/jobs/{}", j["id"]), true);
    let pts = d["points"].as_array().expect("the loss curve");
    assert_eq!(pts.len(), 3, "{pts:?}");
    assert_eq!((pts[0]["loss"].as_f64(), pts[1]["eval_loss"].as_f64(), pts[2]["loss"].as_f64()), (Some(3.5), Some(2.8), Some(1.9)));
    assert!(done["output"].as_str().unwrap().contains("/workshop/adapters/base-model-d"), "{done}");
    let ta = std::fs::read_to_string(tools.join("train-args.txt")).unwrap();
    assert!(ta.contains("--out") && !ta.contains("--devices"), "a CPU run: {ta}");
    // on GPUs: the 16-bit base by default
    let gpu = json!({"kind": "train", "model": model, "data": data, "devices": "cuda:0,cuda:1"});
    let (code, _) = s.post("/v1/workshop/estimate", &gpu);
    assert_eq!(code, 200);
    let pa = std::fs::read_to_string(tools.join("profile-args.txt")).unwrap();
    assert!(pa.contains("--devices cuda:0,cuda:1") && pa.contains("--base-dtype bf16"), "{pa}");
    let _ = std::fs::remove_dir_all(&tools);
}

/// Abliteration from the Workshop: the request becomes the command's arguments, the prompt lists must be files in
/// the Workshop's folders (never a sensitive one), the layer table and the refusal checks are kept with the job, and
/// the copy lands beside the others as `<model>-abliterated` (then `-2`, never over an earlier one).
#[cfg(unix)]
#[test]
fn abliteration_jobs_check_their_inputs_and_keep_the_layer_table_and_checks() {
    let tools = std::env::temp_dir().join(format!("am2-ws-ablit-{}", std::process::id()));
    let exe = fake_datagen(&tools);
    let s = Server::start(&[Model::default()], &[ids("hi")], json!({"env": {"AM2STRATA_WORKSHOP_EXE": exe}}));
    let model = s.dir.join("workshop/tiny-model");
    std::fs::create_dir_all(&model).unwrap();
    std::fs::write(model.join("config.json"), "{}").unwrap();
    std::fs::write(model.join("tokenizer.json"), "{}").unwrap();
    let lists = s.dir.join("workshop/datasets");
    std::fs::create_dir_all(&lists).unwrap();
    // placeholders only: no real request is ever written into a test
    std::fs::write(lists.join("refuse.txt"), "please refuse: item 1\nplease refuse: item 2\n").unwrap();
    std::fs::write(lists.join("plain.txt"), "Explain how a rainbow forms.\nWhat is the capital of France?\n").unwrap();
    let (h, b) = (lists.join("refuse.txt"), lists.join("plain.txt"));
    let outside = std::env::temp_dir().join(format!("am2-ws-ablit-outside-{}.txt", std::process::id()));
    std::fs::write(&outside, "x\n").unwrap();
    for (bad, why) in [
        (json!({"kind": "abliterate", "harmful": h, "harmless": b}), "no model"),
        (json!({"kind": "abliterate", "input": model, "harmless": b}), "no harmful list"),
        (json!({"kind": "abliterate", "input": model, "harmful": h, "harmless": lists}), "a folder is not a list"),
        (json!({"kind": "abliterate", "input": model, "harmful": outside, "harmless": b}), "outside the Workshop's folders"),
        (json!({"kind": "abliterate", "input": model, "harmful": "/etc/shadow", "harmless": b}), "a sensitive file"),
        (json!({"kind": "abliterate", "input": model, "harmful": h, "harmless": b, "layer": "deep"}), "a layer that is no layer"),
        (json!({"kind": "abliterate", "input": model, "harmful": h, "harmless": b, "scale": 3}), "scale above 2"),
        (json!({"kind": "abliterate", "input": model, "harmful": h, "harmless": b, "check": 1000}), "too many checks"),
        (json!({"kind": "abliterate", "input": model, "harmful": h, "harmless": b, "devices": "gpu"}), "bad devices"),
        (json!({"kind": "abliterate", "input": lists, "harmful": h, "harmless": b}), "not a model folder"),
    ] {
        assert_eq!(s.post("/v1/workshop/jobs", &bad).0, 400, "{why}");
    }
    let req = json!({"kind": "abliterate", "input": model, "harmful": h, "harmless": b, "layer": "12", "scale": 0.5, "check": 4});
    let (code, j) = s.post("/v1/workshop/jobs", &req);
    assert_eq!(code, 202, "{j}");
    let done = wait_state(&s, j["id"].as_u64().unwrap(), "done");
    let out = done["result"]["output"].as_str().unwrap().to_string();
    assert!(out.ends_with("/workshop/tiny-model-abliterated"), "{out}");
    assert_eq!(done["result"]["private"], true);
    let args = std::fs::read_to_string(tools.join("abliterate-args.txt")).unwrap();
    assert!(args.contains("--layer 12") && args.contains("--scale 0.5") && args.contains("--check 4") && !args.contains("--devices"), "{args}");
    let (_, d) = s.get(&format!("/v1/workshop/jobs/{}", j["id"]), true);
    let log = d["log"].as_array().unwrap();
    assert!(log.iter().any(|l| l.as_str().unwrap().contains("after model: refused 0%")), "{log:?}");
    let pts = d["points"].as_array().unwrap();
    assert_eq!(pts[0]["layers"].as_array().map(|a| a.len()), Some(2), "the layer table: {pts:?}");
    assert_eq!(pts.iter().filter(|p| p["check"].is_string()).count(), 2);
    // a second run never writes over the first
    let (_, j2) = s.post("/v1/workshop/jobs", &json!({"kind": "abliterate", "input": model, "harmful": h, "harmless": b, "layer": "2:1"}));
    let done2 = wait_state(&s, j2["id"].as_u64().unwrap(), "done");
    assert!(done2["result"]["output"].as_str().unwrap().ends_with("tiny-model-abliterated-2"), "{done2}");
    assert!(std::fs::read_to_string(tools.join("abliterate-args.txt")).unwrap().contains("--layer 2:1 --scale 1 --check 0"));
    // and the Workshop's model list marks it private
    let (_, f) = s.get("/v1/workshop/files", true);
    let it = f["hf_models"].as_array().unwrap().iter().find(|m| m["path"].as_str().unwrap().ends_with("tiny-model-abliterated")).cloned();
    assert_eq!(it.expect("the copy is listed")["private"], true);
    // the estimate first in the log
    assert!(log.iter().any(|l| l.as_str().unwrap().contains("busiest of 2 GPU")), "{log:?}");
    // the form's estimate: the job's own command with --estimate, nothing run
    let (code, e) = s.post("/v1/workshop/estimate", &json!({"kind": "abliterate", "input": model, "harmful": h, "harmless": b, "check": 4, "devices": "cuda:0,cuda:1"}));
    assert_eq!(code, 200, "{e}");
    assert_eq!((e["event"].as_str(), e["kind"].as_str(), e["gpu"].as_bool()), (Some("estimate"), Some("abliterate"), Some(true)), "{e}");
    let ea = std::fs::read_to_string(tools.join("estimate-args.txt")).unwrap();
    assert!(ea.starts_with("--json abliterate") && ea.contains("--devices cuda:0,cuda:1") && ea.contains("--check 4") && ea.ends_with("--estimate\n"), "{ea}");
    let (code, _) = s.post("/v1/workshop/estimate", &json!({"kind": "abliterate", "input": model, "harmless": b}));
    assert_eq!(code, 400, "an estimate checks the request as the job does");
    // a harmless list may be a chat .jsonl (its first user messages)
    std::fs::write(lists.join("plain.jsonl"), "{\"messages\":[{\"role\":\"user\",\"content\":\"Explain how a rainbow forms.\"}]}\n").unwrap();
    let (code, _) = s.post("/v1/workshop/estimate", &json!({"kind": "abliterate", "input": model, "harmful": h, "harmless": lists.join("plain.jsonl")}));
    assert_eq!(code, 200);
    // on the GPUs: the run takes them from the chat model, as training does (unloaded, then loaded again)
    let (code, j3) = s.post("/v1/workshop/jobs", &json!({"kind": "abliterate", "input": model, "harmful": h, "harmless": b, "devices": "cuda:0,cuda:1"}));
    assert_eq!(code, 202, "{j3}");
    wait_state(&s, j3["id"].as_u64().unwrap(), "done");
    let (_, d3) = s.get(&format!("/v1/workshop/jobs/{}", j3["id"]), true);
    assert!(d3["log"].as_array().unwrap().iter().any(|l| l.as_str().unwrap().contains("waiting for the GPUs")), "{d3}");
    assert!(std::fs::read_to_string(tools.join("abliterate-args.txt")).unwrap().contains("--devices cuda:0,cuda:1"));
    let _ = std::fs::remove_file(&outside);
    let _ = std::fs::remove_dir_all(&tools);
}

/// LoRA upcycling from the Workshop (plan Part C): `upcycle` takes a dense folder (never an MoE or an upcycled one) and
/// an experts × top-k shape; `export-moe` takes an upcycled folder, the adapter trained on it (the same shape) and an
/// export type; both become the command's arguments and land beside the others. The model list shows the shape.
#[cfg(unix)]
#[test]
fn upcycle_and_export_moe_jobs_check_their_inputs() {
    let tools = std::env::temp_dir().join(format!("am2-ws-moe-{}", std::process::id()));
    let exe = fake_datagen(&tools);
    let s = Server::start(&[Model::default()], &[ids("hi")], json!({"env": {"AM2STRATA_WORKSHOP_EXE": exe}}));
    let ws = s.dir.join("workshop");
    let dense = ws.join("tiny-dense");
    std::fs::create_dir_all(&dense).unwrap();
    std::fs::write(dense.join("config.json"), r#"{"model_type": "qwen3", "num_hidden_layers": 4, "hidden_size": 32, "intermediate_size": 64}"#).unwrap();
    std::fs::write(dense.join("model.safetensors"), "x").unwrap();
    let mixtral = ws.join("tiny-mixtral");
    std::fs::create_dir_all(&mixtral).unwrap();
    std::fs::write(mixtral.join("config.json"), r#"{"model_type": "mixtral", "num_local_experts": 8}"#).unwrap();
    let up = ws.join("tiny-up");
    std::fs::create_dir_all(&up).unwrap();
    std::fs::write(up.join("config.json"), r#"{"model_type": "qwen3", "num_hidden_layers": 4, "prastara_moe": {"experts": 4, "top_k": 2}}"#).unwrap();
    std::fs::write(up.join("model.safetensors"), "x").unwrap();
    let ad = ws.join("tiny-up-adapter");
    std::fs::create_dir_all(&ad).unwrap();
    std::fs::write(ad.join("adapter_config.json"), r#"{"r": 16, "prastara_moe": {"experts": 4, "top_k": 2, "rank": 16}}"#).unwrap();
    let wrong = ws.join("other-adapter");
    std::fs::create_dir_all(&wrong).unwrap();
    std::fs::write(wrong.join("adapter_config.json"), r#"{"r": 16}"#).unwrap();
    let calib = ws.join("calib.jsonl");
    std::fs::write(&calib, "{\"messages\": []}\n").unwrap();
    // a fine-grained upcycled folder and adapters: the expert width and rank are part of the shape
    let fine_up = ws.join("tiny-fine");
    std::fs::create_dir_all(&fine_up).unwrap();
    std::fs::write(fine_up.join("config.json"), r#"{"model_type": "qwen3", "prastara_moe": {"experts": 8, "top_k": 2, "expert_ff": 16, "split": "cluster", "expert_rank": 4, "shared_a": true}}"#).unwrap();
    std::fs::write(fine_up.join("model.safetensors"), "x").unwrap();
    let fine_ad = ws.join("tiny-fine-adapter");
    std::fs::create_dir_all(&fine_ad).unwrap();
    std::fs::write(fine_ad.join("adapter_config.json"), r#"{"r": 16, "prastara_moe": {"experts": 8, "top_k": 2, "expert_ff": 16, "split": "cluster", "expert_rank": 4, "shared_a": true, "rank": 16}}"#).unwrap();
    let full_ad = ws.join("tiny-full-adapter");
    std::fs::create_dir_all(&full_ad).unwrap();
    std::fs::write(full_ad.join("adapter_config.json"), r#"{"r": 16, "prastara_moe": {"experts": 8, "top_k": 2, "rank": 16}}"#).unwrap();
    let fine = |extra: serde_json::Value| { let mut v = json!({"kind": "upcycle", "input": dense, "experts": 8, "top_k": 2, "expert_ff": 16}); for (k, x) in extra.as_object().unwrap() { v[k] = x.clone(); } v };
    for (bad, why) in [
        (fine(json!({"expert_ff": 24})), "an expert width that does not divide the feed-forward"),
        (fine(json!({"expert_ff": 100})), "an expert width above the feed-forward"),
        (fine(json!({"expert_ff": "wide"})), "an expert width not a number"),
        (fine(json!({"experts": 10})), "experts not whole copies of the 4 groups"),
        (fine(json!({"top_k": 5, "experts": 8})), "top-k above the groups"),
        (fine(json!({})), "a cluster split without calibration text"),
        (fine(json!({"split": "random", "calib": calib})), "an unknown split"),
        (fine(json!({"calib": "/etc/passwd"})), "calibration text outside the Workshop"),
        (fine(json!({"calib": ws})), "calibration text that is a folder"),
        (fine(json!({"split": "contiguous", "dense_start_check": true})), "the dense-start check without calibration text"),
        (fine(json!({"split": "contiguous", "expert_rank": 100})), "an expert rank too large"),
        (fine(json!({"split": "contiguous", "shared_a": true})), "a shared A without an expert rank"),
        (fine(json!({"split": "contiguous", "shared_a": "yes", "expert_rank": 4})), "shared_a not a bool"),
        (fine(json!({"calib": calib, "calib_tokens": 10})), "too few calibration tokens"),
        (json!({"kind": "upcycle", "input": dense, "split": "contiguous"}), "a fine-grained field without an expert width"),
        (json!({"kind": "upcycle", "input": dense, "expert_rank": 4}), "an expert rank at full width"),
        (json!({"kind": "export-moe", "input": fine_up, "adapter": full_ad}), "a full-width adapter on a fine-grained model"),
        (json!({"kind": "export-moe", "input": up, "adapter": fine_ad}), "a fine-grained adapter on another shape"),
    ] {
        let (code, e) = s.post("/v1/workshop/jobs", &bad);
        assert_eq!(code, 400, "{why}: {e}");
    }
    for (bad, why) in [
        (json!({"kind": "upcycle"}), "no model"),
        (json!({"kind": "upcycle", "input": mixtral}), "a mixture of experts already"),
        (json!({"kind": "upcycle", "input": up}), "upcycled already"),
        (json!({"kind": "upcycle", "input": dense, "experts": 1}), "one expert"),
        (json!({"kind": "upcycle", "input": dense, "experts": 300}), "too many experts"),
        (json!({"kind": "upcycle", "input": dense, "experts": 4, "top_k": 5}), "top-k above the experts"),
        (json!({"kind": "upcycle", "input": dense, "top_k": 0}), "top-0"),
        (json!({"kind": "upcycle", "input": dense, "experts": "ten"}), "experts not a number"),
        (json!({"kind": "upcycle", "input": "/etc"}), "outside the Workshop"),
        (json!({"kind": "export-moe", "input": dense}), "not upcycled"),
        (json!({"kind": "export-moe", "input": up, "type": "q3_x"}), "an unknown type"),
        (json!({"kind": "export-moe", "input": up, "adapter": wrong}), "an adapter for another model"),
        (json!({"kind": "export-moe", "input": up, "adapter": dense}), "not an adapter"),
        (json!({"kind": "export-moe", "input": up, "pre_tokenizer": "a b; rm"}), "a pre-tokenizer that is no name"),
        (json!({"kind": "export-moe", "input": up, "output": "/tmp/x.gguf"}), "an output outside"),
    ] {
        let (code, e) = s.post("/v1/workshop/jobs", &bad);
        assert_eq!(code, 400, "{why}: {e}");
    }
    let (code, j) = s.post("/v1/workshop/jobs", &json!({"kind": "upcycle", "input": dense}));
    assert_eq!(code, 202, "{j}");
    let done = wait_state(&s, j["id"].as_u64().unwrap(), "done");
    let out = done["result"]["output"].as_str().unwrap().to_string();
    assert!(out.ends_with("/workshop/tiny-dense-moe10x1"), "the default shape is 10x1: {out}");
    assert_eq!(done["result"]["active_parameters"], 8_200_000_000u64);
    let args = std::fs::read_to_string(tools.join("upcycle-args.txt")).unwrap();
    assert!(args.contains("--experts 10 --top-k 1") && args.contains(&format!("--out {out}")), "{args}");
    let (code, _) = s.post("/v1/workshop/jobs", &json!({"kind": "upcycle", "input": dense}));
    assert_eq!(code, 400, "the copy exists: never written over");
    let (_, j) = s.post("/v1/workshop/jobs", &json!({"kind": "upcycle", "input": dense, "experts": 8, "top_k": 2}));
    let done = wait_state(&s, j["id"].as_u64().unwrap(), "done");
    assert!(done["result"]["output"].as_str().unwrap().ends_with("tiny-dense-moe8x2"));
    // fine-grained (the 56B-A5B design in miniature): 8 experts of 16 neurons, top-2, clustered on the calibration text
    let (code, j) = s.post("/v1/workshop/jobs", &fine(json!({"calib": calib, "calib_tokens": 512, "expert_rank": 4, "shared_a": true, "dense_start_check": true})));
    assert_eq!(code, 202, "{j}");
    let done = wait_state(&s, j["id"].as_u64().unwrap(), "done");
    assert!(done["result"]["output"].as_str().unwrap().ends_with("tiny-dense-moe8f16x2"), "{done}");
    let args = std::fs::read_to_string(tools.join("upcycle-args.txt")).unwrap();
    assert!(args.contains(&format!("--experts 8 --top-k 2 --expert-ff 16 --split cluster --calib {} --calib-tokens 512 --expert-rank 4 --shared-a --dense-start-check", calib.display())), "{args}");
    let (_, j) = s.post("/v1/workshop/jobs", &fine(json!({"split": "contiguous", "output": ws.join("tiny-contig").display().to_string()})));
    wait_state(&s, j["id"].as_u64().unwrap(), "done");
    let args = std::fs::read_to_string(tools.join("upcycle-args.txt")).unwrap();
    assert!(args.contains("--expert-ff 16 --split contiguous") && !args.contains("--calib") && !args.contains("--expert-rank"), "{args}");
    // an expert width equal to the feed-forward is the full-width design
    let (_, j) = s.post("/v1/workshop/jobs", &json!({"kind": "upcycle", "input": dense, "experts": 6, "top_k": 1, "expert_ff": 64}));
    wait_state(&s, j["id"].as_u64().unwrap(), "done");
    let args = std::fs::read_to_string(tools.join("upcycle-args.txt")).unwrap();
    assert!(args.contains("--experts 6 --top-k 1") && !args.contains("--expert-ff"), "{args}");
    let (code, j) = s.post("/v1/workshop/jobs", &json!({"kind": "export-moe", "input": fine_up, "adapter": fine_ad}));
    assert_eq!(code, 202, "a fine-grained adapter on its own model: {j}");
    wait_state(&s, j["id"].as_u64().unwrap(), "done");

    let (code, j) = s.post("/v1/workshop/jobs", &json!({"kind": "export-moe", "input": up, "adapter": ad, "pre_tokenizer": "qwen2"}));
    assert_eq!(code, 202, "{j}");
    let done = wait_state(&s, j["id"].as_u64().unwrap(), "done");
    assert!(done["result"]["output"].as_str().unwrap().ends_with("/workshop/tiny-up-q4_k.gguf"), "{done}");
    let args = std::fs::read_to_string(tools.join("export-moe-args.txt")).unwrap();
    assert!(args.contains(&format!("--adapter {}", ad.display())) && args.contains("--type q4_k") && args.contains("--pre-tokenizer qwen2"), "{args}");
    let (_, j) = s.post("/v1/workshop/jobs", &json!({"kind": "export-moe", "input": up, "type": "Q8_0"}));
    wait_state(&s, j["id"].as_u64().unwrap(), "done");
    let args = std::fs::read_to_string(tools.join("export-moe-args.txt")).unwrap();
    assert!(args.contains("--type q8_0") && !args.contains("--adapter"), "{args}");
    // the model list carries the shape and the dimensions the form counts parameters from
    let (_, f) = s.get("/v1/workshop/files", true);
    let hf = f["hf_models"].as_array().unwrap();
    let u = hf.iter().find(|x| x["path"].as_str().unwrap().ends_with("tiny-up")).unwrap();
    assert_eq!(u["moe"], json!({"experts": 4, "top_k": 2}));
    let d = hf.iter().find(|x| x["path"].as_str().unwrap().ends_with("tiny-dense")).unwrap();
    assert_eq!((d["dims"]["hidden"].as_u64(), d["dims"]["layers"].as_u64()), (Some(32), Some(4)));
    // the forms' estimates: the command with --estimate (for upcycle: the sizes and the training it takes on the cards)
    let (code, e) = s.post("/v1/workshop/estimate", &json!({"kind": "upcycle", "input": dense, "experts": 10, "top_k": 1}));
    assert_eq!((code, e["kind"].as_str()), (200, Some("upcycle")), "{e}");
    let ea = std::fs::read_to_string(tools.join("estimate-args.txt")).unwrap();
    assert!(ea.contains("upcycle") && ea.contains("--experts 10 --top-k 1") && ea.contains("--estimate"), "{ea}");
    let (code, e) = s.post("/v1/workshop/estimate", &fine(json!({"calib": calib, "dense_start_check": true})));
    assert_eq!((code, e["kind"].as_str()), (200, Some("upcycle")), "{e}");
    let ea = std::fs::read_to_string(tools.join("estimate-args.txt")).unwrap();
    assert!(ea.contains("--expert-ff 16 --split cluster") && ea.contains("--dense-start-check") && ea.contains("--estimate"), "{ea}");
    let (code, e) = s.post("/v1/workshop/estimate", &json!({"kind": "export-moe", "input": up, "type": "q4_k"}));
    assert_eq!((code, e["kind"].as_str(), e["gpu"].as_bool()), (200, Some("export-moe"), Some(false)), "{e}");
    let (code, _) = s.post("/v1/workshop/estimate", &json!({"kind": "export-moe", "input": dense}));
    assert_eq!(code, 400, "not upcycled");
    let _ = std::fs::remove_dir_all(&tools);
}

/// Regression runs and grade-and-train from the Workshop (plan Part G): the model, adapter, suite (and every data file
/// it names), earlier reports and prompts must be inside the Workshop's folders; passes, devices and the numbers are
/// checked; the command gets the server's address and key in its environment (for the judge); the report card and the
/// rounds come back as the job's result.
#[cfg(unix)]
#[test]
fn regress_and_grade_train_jobs_check_their_inputs_and_carry_their_report() {
    let tools = std::env::temp_dir().join(format!("am2-ws-regress-{}", std::process::id()));
    let exe = fake_datagen(&tools);
    let s = Server::start(&[Model::default()], &[ids("hi")], json!({"env": {"AM2STRATA_WORKSHOP_EXE": exe}}));
    let ws = s.dir.join("workshop");
    let model = ws.join("tiny-model");
    std::fs::create_dir_all(&model).unwrap();
    for f in ["config.json", "tokenizer.json"] {
        std::fs::write(model.join(f), "{}").unwrap();
    }
    std::fs::write(model.join("model.safetensors"), "x").unwrap();
    let ad = ws.join("tiny-adapter");
    std::fs::create_dir_all(&ad).unwrap();
    std::fs::write(ad.join("adapter_config.json"), r#"{"r": 8}"#).unwrap();
    let suites = ws.join("suites");
    std::fs::create_dir_all(&suites).unwrap();
    std::fs::write(suites.join("heldout.jsonl"), "{}\n").unwrap();
    let suite = suites.join("core.json");
    std::fs::write(&suite, r#"{"tests": [{"name": "held-out", "kind": "loss", "data": "heldout.jsonl"}]}"#).unwrap();
    let leaky = suites.join("leaky.json");
    std::fs::write(&leaky, r#"{"tests": [{"name": "x", "kind": "loss", "data": "/etc/passwd"}]}"#).unwrap();
    let missing = suites.join("missing.json");
    std::fs::write(&missing, r#"{"tests": [{"name": "x", "kind": "loss", "data": "nope.jsonl"}]}"#).unwrap();
    let empty = suites.join("empty.json");
    std::fs::write(&empty, r#"{"tests": []}"#).unwrap();
    let prompts = ws.join("datasets").join("prompts.jsonl");
    std::fs::create_dir_all(prompts.parent().unwrap()).unwrap();
    std::fs::write(&prompts, "{\"prompt\": \"x\"}\n").unwrap();
    for (bad, why) in [
        (json!({"kind": "regress", "suite": suite}), "no model"),
        (json!({"kind": "regress", "model": model}), "no suite"),
        (json!({"kind": "regress", "model": "/etc", "suite": suite}), "a model outside"),
        (json!({"kind": "regress", "model": model, "suite": leaky}), "a suite that reads outside the Workshop"),
        (json!({"kind": "regress", "model": model, "suite": missing}), "a suite whose data is missing"),
        (json!({"kind": "regress", "model": model, "suite": empty}), "a suite without tests"),
        (json!({"kind": "regress", "model": model, "suite": prompts}), "a suite that is not .json"),
        (json!({"kind": "regress", "model": model, "suite": suite, "adapter": model}), "an adapter that is no adapter"),
        (json!({"kind": "regress", "model": model, "suite": suite, "passes": [0]}), "0 passes"),
        (json!({"kind": "regress", "model": model, "suite": suite, "passes": "1,x"}), "passes that are no numbers"),
        (json!({"kind": "regress", "model": model, "suite": suite, "passes": {}}), "passes of the wrong type"),
        (json!({"kind": "regress", "model": model, "suite": suite, "devices": "gpu"}), "bad devices"),
        (json!({"kind": "regress", "model": model, "suite": suite, "against": "/etc/passwd"}), "a report outside"),
        (json!({"kind": "regress", "model": model, "suite": suite, "against": [5]}), "a report that is no path"),
        (json!({"kind": "regress", "model": model, "suite": suite, "output": "/tmp/r"}), "an output outside"),
        (json!({"kind": "grade-train", "model": model, "suite": suite}), "no prompts"),
        (json!({"kind": "grade-train", "model": model, "suite": suite, "data": prompts, "grader": "vibes"}), "an unknown grader"),
        (json!({"kind": "grade-train", "model": model, "suite": suite, "data": prompts, "samples": 0}), "no samples"),
        (json!({"kind": "grade-train", "model": model, "suite": suite, "data": prompts, "rounds": 100}), "too many rounds"),
        (json!({"kind": "grade-train", "model": model, "suite": suite, "data": prompts, "temperature": "hot"}), "a temperature that is no number"),
    ] {
        let (code, e) = s.post("/v1/workshop/jobs", &bad);
        assert_eq!(code, 400, "{why}: {e}");
        assert!(e["error"]["message"].is_string(), "{why}: {e}");
    }
    let (code, j) = s.post("/v1/workshop/jobs", &json!({"kind": "regress", "model": model, "adapter": ad, "suite": suite, "passes": [1, 2]}));
    assert_eq!(code, 202, "{j}");
    let done = wait_state(&s, j["id"].as_u64().unwrap(), "done");
    let out = done["result"]["output"].as_str().unwrap().to_string();
    assert!(out.ends_with("/workshop/reports/tiny-model-tiny-adapter/report.json"), "{out}");
    assert_eq!(done["result"]["regressions"][0]["test"], "humaneval", "the regressions come back: {done}");
    assert_eq!(done["result"]["report"]["results"][0]["metric"], "pass@1");
    let args = std::fs::read_to_string(tools.join("regress-args.txt")).unwrap();
    assert!(args.contains(&format!("--adapter {}", ad.display())) && args.contains("--passes 1,2") && args.contains(&format!("--suite {}", suite.display())), "{args}");
    let env = std::fs::read_to_string(tools.join("regress-env.txt")).unwrap();
    assert!(env.starts_with("http://127.0.0.1:"), "the judge's server comes in the environment: {env}");
    // a second run against the first, from its folder
    let first = std::path::Path::new(&out).parent().unwrap().to_path_buf();
    let (code, j) = s.post("/v1/workshop/jobs", &json!({"kind": "regress", "model": model, "suite": suite, "against": first, "passes": "2"}));
    assert_eq!(code, 202, "{j}");
    wait_state(&s, j["id"].as_u64().unwrap(), "done");
    let args = std::fs::read_to_string(tools.join("regress-args.txt")).unwrap();
    assert!(args.contains(&format!("--against {}", first.join("report.json").display())) && args.contains("--passes 2"), "{args}");
    let (_, f) = s.get("/v1/workshop/files", true);
    assert!(f["suites"].as_array().unwrap().iter().any(|x| x["path"].as_str().unwrap().ends_with("core.json")), "{f}");
    assert!(f["reports"].as_array().unwrap().iter().any(|x| x["path"].as_str().unwrap().ends_with("tiny-model-tiny-adapter")), "{f}");

    let (code, j) = s.post("/v1/workshop/jobs", &json!({"kind": "grade-train", "model": model, "suite": suite, "data": prompts, "samples": 4, "rounds": 2,
                                                         "grader": "judge", "temperature": 0.7, "max_steps": 10}));
    assert_eq!(code, 202, "{j}");
    let done = wait_state(&s, j["id"].as_u64().unwrap(), "done");
    assert!(done["result"]["output"].as_str().unwrap().ends_with("/workshop/grade-train/tiny-model-prompts"), "{done}");
    assert_eq!(done["result"]["rounds"][0]["promoted"], true);
    assert!(done["points"].as_array().unwrap().iter().any(|p| p["loss"] == 2.5), "the training loss is kept for the curve");
    let args = std::fs::read_to_string(tools.join("grade-train-args.txt")).unwrap();
    assert!(args.contains("--samples 4") && args.contains("--rounds 2") && args.contains("--grader judge") && args.contains("--temperature 0.7")
            && args.contains("--max-steps 10") && args.contains(&format!("--data {}", prompts.display())), "{args}");
    let _ = std::fs::remove_dir_all(&tools);
}

/// Regress through the Rust engine: `engine_gguf` is a .gguf inside the Workshop's folders, `engine_args` takes only the
/// flags that place and size the engine (each value checked), and both reach the command. A judge on the GPUs the run
/// takes is refused at once (it would wait for the run, and the run for it); on a CPU run it judges.
#[cfg(unix)]
#[test]
fn regress_through_the_engine_checks_its_settings_and_a_gpu_judge_is_refused_on_a_gpu_run() {
    let tools = std::env::temp_dir().join(format!("am2-ws-engine-{}", std::process::id()));
    let exe = fake_datagen(&tools);
    // the chat model runs on GPU 0
    let gpu_model = Model { id: "gpu-judge", extra: json!({"gpu": [0]}), ..Model::default() };
    let s = Server::start(&[gpu_model], &[ids("hi")], json!({"env": {"AM2STRATA_WORKSHOP_EXE": exe}}));
    let ws = s.dir.join("workshop");
    let model = ws.join("tiny-model");
    std::fs::create_dir_all(&model).unwrap();
    for f in ["config.json", "tokenizer.json"] {
        std::fs::write(model.join(f), "{}").unwrap();
    }
    let gguf = ws.join("tiny-model.Q8_0.gguf");
    std::fs::write(&gguf, "GGUF").unwrap();
    let suites = ws.join("suites");
    std::fs::create_dir_all(&suites).unwrap();
    std::fs::write(suites.join("p.jsonl"), "{\"prompt\": \"hi\"}\n").unwrap();
    let suite = suites.join("gen.json");
    std::fs::write(&suite, r#"{"tests": [{"name": "rep", "kind": "repeat", "data": "p.jsonl"}, {"name": "s", "kind": "samples", "data": "p.jsonl"}]}"#).unwrap();
    let judged = suites.join("judged.json");
    std::fs::write(&judged, r#"{"tests": [{"name": "j", "kind": "judge", "data": "p.jsonl"}]}"#).unwrap();
    let base = json!({"kind": "regress", "model": model, "suite": suite});
    let with = |extra: Value| { let mut b = base.clone(); for (k, v) in extra.as_object().unwrap() { b[k] = v.clone(); } b };
    for (bad, why) in [
        (with(json!({"engine_gguf": "/etc/x.gguf"})), "a GGUF outside the Workshop"),
        (with(json!({"engine_gguf": model.join("config.json")})), "not a .gguf"),
        (with(json!({"engine_gguf": ws.join("none.gguf")})), "a GGUF that does not exist"),
        (with(json!({"engine_gguf": 5})), "a GGUF that is no path"),
        (with(json!({"engine_args": "--devices cuda:0"})), "engine settings without a GGUF"),
        (with(json!({"engine_gguf": gguf, "engine_args": "--pack /etc"})), "a flag that names a file"),
        (with(json!({"engine_gguf": gguf, "engine_args": "--remote-experts 1.2.3.4:9"})), "a flag that names another machine"),
        (with(json!({"engine_gguf": gguf, "engine_args": "--devices gpu0"})), "bad devices"),
        (with(json!({"engine_gguf": gguf, "engine_args": "--max-context lots"})), "a context that is no number"),
        (with(json!({"engine_gguf": gguf, "engine_args": "--max-context"})), "a flag without its value"),
        (with(json!({"engine_gguf": gguf, "engine_args": ["--threads", "2"]})), "settings that are no string"),
        (with(json!({"engine_gguf": gguf, "judge_model": 3})), "a judge that is no name"),
    ] {
        let (code, e) = s.post("/v1/workshop/jobs", &bad);
        assert_eq!(code, 400, "{why}: {e}");
        assert!(e["error"]["message"].is_string(), "{why}: {e}");
    }
    let (code, j) = s.post("/v1/workshop/jobs", &with(json!({"engine_gguf": gguf, "engine_args": "--threads 2 --max-context 4096 --no-graph"})));
    assert_eq!(code, 202, "{j}");
    wait_state(&s, j["id"].as_u64().unwrap(), "done");
    let args = std::fs::read_to_string(tools.join("regress-args.txt")).unwrap();
    assert!(args.contains(&format!("--engine-gguf {}", gguf.display())) && args.contains("--engine-args --threads 2 --max-context 4096 --no-graph"), "{args}");
    // a judge test on a GPU run whose judge (the loaded chat model) is on the GPUs: refused before the GPUs are taken
    let (code, j) = s.post("/v1/workshop/jobs", &json!({"kind": "regress", "model": model, "suite": judged, "devices": "cuda:0"}));
    assert_eq!(code, 202, "{j}");
    let failed = wait_state(&s, j["id"].as_u64().unwrap(), "failed");
    let err = failed["error"].as_str().unwrap_or("");
    assert!(err.contains("gpu-judge") && err.contains("remote") && err.contains("CPU"), "{failed}");
    // the same through the engine on a GPU (the Workshop's model on the CPU): refused the same way
    let (code, j) = s.post("/v1/workshop/jobs", &json!({"kind": "regress", "model": model, "suite": judged, "engine_gguf": gguf, "engine_args": "--devices cuda:0=14"}));
    assert_eq!(code, 202, "{j}");
    assert!(wait_state(&s, j["id"].as_u64().unwrap(), "failed")["error"].as_str().unwrap().contains("GPUs this run takes"));
    // on the CPU the GPU model can judge: the run goes ahead
    let (code, j) = s.post("/v1/workshop/jobs", &json!({"kind": "regress", "model": model, "suite": judged}));
    assert_eq!(code, 202, "{j}");
    wait_state(&s, j["id"].as_u64().unwrap(), "done");
    let _ = std::fs::remove_dir_all(&tools);
}

/// Cleaning a dataset from the Workshop: its inputs, held-out files and tokenizer folder must be inside the Workshop's
/// folders (and never a key file), its settings become the clean command's flags, the output never overwrites an
/// input, and wrong values are refused before anything runs.
#[cfg(unix)]
#[test]
fn clean_jobs_check_their_inputs_and_pass_their_settings() {
    let tools = std::env::temp_dir().join(format!("am2-ws-clean-{}", std::process::id()));
    let exe = fake_datagen(&tools);
    let s = Server::start(&[Model::default()], &[ids("hi")], json!({"env": {"AM2STRATA_WORKSHOP_EXE": exe}}));
    let ws = s.dir.join("workshop");
    let ds = ws.join("datasets");
    std::fs::create_dir_all(&ds).unwrap();
    let (a, b, held) = (ds.join("mula-a.jsonl"), ds.join("aya-b.jsonl"), ds.join("heldout.jsonl"));
    for p in [&a, &b, &held] {
        std::fs::write(p, "{\"messages\":[{\"role\":\"user\",\"content\":\"q\"},{\"role\":\"assistant\",\"content\":\"a\"}]}\n").unwrap();
    }
    let model = ws.join("tok-model");
    std::fs::create_dir_all(&model).unwrap();
    std::fs::write(model.join("tokenizer.json"), "{}").unwrap();
    let no_tok = ws.join("no-tok");
    std::fs::create_dir_all(&no_tok).unwrap();
    std::fs::write(ws.join("server.pem"), "k").unwrap();
    for (bad, why) in [
        (json!({"kind": "clean"}), "needs \"inputs\""),
        (json!({"kind": "clean", "inputs": []}), "needs \"inputs\""),
        (json!({"kind": "clean", "inputs": ["/etc/passwd"]}), ".jsonl"),
        (json!({"kind": "clean", "inputs": ["/nope/x.jsonl"]}), "does not exist"),
        (json!({"kind": "clean", "inputs": [a, 3]}), "list of .jsonl"),
        (json!({"kind": "clean", "inputs": {"a": 1}}), "list of .jsonl"),
        (json!({"kind": "clean", "inputs": [a], "exclude": ["/etc/shadow.jsonl"]}), "exist"),
        (json!({"kind": "clean", "inputs": [a], "model": no_tok}), "tokenizer.json"),
        (json!({"kind": "clean", "inputs": [a], "model": ws.join("server.pem")}), "never touches"),
        (json!({"kind": "clean", "inputs": [a], "model": 5}), "\"model\""),
        (json!({"kind": "clean", "inputs": [a], "max_repeat": 2}), "\"max_repeat\""),
        (json!({"kind": "clean", "inputs": [a], "near": "high"}), "\"near\""),
        (json!({"kind": "clean", "inputs": [a], "max_len": 4}), "\"max_len\""),
        (json!({"kind": "clean", "inputs": [a], "cap_per_language": 0}), "\"cap_per_language\""),
        (json!({"kind": "clean", "inputs": [a], "dedupe": "fuzzy"}), "\"dedupe\""),
        (json!({"kind": "clean", "inputs": [a], "lang_match": "yes"}), "\"lang_match\""),
        (json!({"kind": "clean", "inputs": [a], "output": "/tmp/x.txt"}), ".jsonl"),
        (json!({"kind": "clean", "inputs": [a], "output": "/etc/out.jsonl"}), "/etc/out.jsonl"),
        (json!({"kind": "clean", "inputs": [a, b], "output": b}), "one of the inputs"),
        (json!({"kind": "clean", "inputs": [a], "report": ds.join("card.txt")}), ".md"),
        (json!({"kind": "clean", "inputs": [a], "cap_only": [b]}), "not one of the inputs"),
    ] {
        let (code, e) = s.post("/v1/workshop/jobs", &bad);
        assert_eq!(code, 400, "{bad}: {e}");
        assert!(e["error"]["message"].as_str().unwrap_or("").contains(why), "{bad}: {e}");
    }
    let (code, j) = s.post("/v1/workshop/jobs", &json!({"kind": "clean", "inputs": [a, b], "exclude": [held], "model": model, "max_len": 1024,
        "min_answer_tokens": 8, "min_answer_ratio": 0.05, "max_repeat": 0.5, "dedupe": "near", "near": 0.8, "lang_match": true,
        "cap_per_language": 1500, "cap_only": [b], "seed": 7}));
    assert_eq!(code, 202, "{j}");
    let out = j["output"].as_str().unwrap().to_string();
    assert!(out.ends_with("/workshop/datasets/mula-a-clean.jsonl"), "{out}");
    wait_state(&s, j["id"].as_u64().unwrap(), "done");
    let args = std::fs::read_to_string(tools.join("args.txt")).unwrap();
    for want in [format!("clean {} {}", a.display(), b.display()), format!("--exclude {}", held.display()), format!("--model {}", model.display()),
                 "--max-len 1024".into(), "--min-answer-tokens 8".into(), "--min-answer-ratio 0.05".into(), "--max-repeat 0.5".into(),
                 "--dedupe near".into(), "--near 0.8".into(), "--lang-match".into(), "--cap-per-language 1500".into(), format!("--cap-only {}", b.display()), "--seed 7".into(), format!("--out {out}")] {
        assert!(args.contains(&want), "{want:?} not in {args}");
    }
    // the cleaned file is a dataset like any other: listed, and not written over by the next run
    let (_, f) = s.get("/v1/workshop/files", true);
    assert!(f["datasets"].as_array().unwrap().iter().any(|d| d["path"] == out.as_str()), "{f}");
    let (_, j2) = s.post("/v1/workshop/jobs", &json!({"kind": "clean", "inputs": [a]}));
    assert!(j2["output"].as_str().unwrap().ends_with("mula-a-clean-2.jsonl"), "{j2}");
    wait_state(&s, j2["id"].as_u64().unwrap(), "done");
    let args = std::fs::read_to_string(tools.join("args.txt")).unwrap();
    assert!(!args.contains("--model") && !args.contains("--lang-match") && !args.contains("--exclude"), "{args}");
    std::fs::remove_dir_all(&tools).ok();
}

// Unix only: the stand-in for am2strata-workshop is a `#!/bin/sh` script, which Windows cannot start as a program.
#[cfg(unix)]
#[test]
fn private_data_jobs_check_their_sources_and_model_and_get_the_server_by_environment() {
    use std::os::unix::fs::PermissionsExt as _;
    let tools = std::env::temp_dir().join(format!("am2-ws-private-{}", std::process::id()));
    let exe = fake_datagen(&tools);
    let s = Server::start(&[Model::default()], &[ids("hi")], json!({"env": {"AM2STRATA_WORKSHOP_EXE": exe}}));
    // the owner's folder, inside the Workshop's folders, at first without its SOURCE.toml
    let mine = s.dir.join("mine");
    std::fs::create_dir_all(&mine).unwrap();
    std::fs::write(mine.join("note.txt"), "a plain sentence about a garden").unwrap();
    for (bad, why) in [(json!({"kind": "private-data", "sources": [mine]}), "\"name\""),
                       (json!({"kind": "private-data", "name": "../up", "sources": [mine]}), "\"name\""),
                       (json!({"kind": "private-data", "name": "set"}), "\"sources\""),
                       (json!({"kind": "private-data", "name": "set", "sources": ["/etc"]}), "/etc"),
                       (json!({"kind": "private-data", "name": "set", "sources": [mine]}), "SOURCE.toml")] {
        let (code, e) = s.post("/v1/workshop/jobs", &bad);
        assert_eq!(code, 400, "{bad}: {e}");
        assert!(e["error"]["message"].as_str().unwrap().contains(why), "{bad}: {e}");
    }
    std::fs::write(mine.join("SOURCE.toml"), "owner_attests_adults_only = true\nowner_attests_consent = true\norigin = \"test\"\n").unwrap();
    let (code, e) = s.post("/v1/workshop/jobs", &json!({"kind": "private-data", "name": "set", "sources": [mine], "model": "openai/gpt-5.5"}));
    assert_eq!(code, 400, "{e}");
    assert!(e["error"]["message"].as_str().unwrap().contains("remote model"), "{e}");
    let (code, e) = s.post("/v1/workshop/jobs", &json!({"kind": "private-data", "name": "set", "sources": [mine], "max_frames": 1000}));
    assert_eq!(code, 400, "{e}");

    let (code, j) = s.post("/v1/workshop/jobs", &json!({"kind": "private-data", "name": "set", "sources": [mine], "model": "m1",
                                                        "text_count": 5, "every_s": 2.5}));
    assert_eq!(code, 202, "{j}");
    let done = wait_state(&s, j["id"].as_u64().unwrap(), "done");
    let out = std::path::PathBuf::from(done["output"].as_str().unwrap());
    assert!(out.ends_with("private/set"), "{}", out.display());
    let root = out.parent().unwrap();
    assert_eq!(std::fs::metadata(root).unwrap().permissions().mode() & 0o777, 0o700, "the private root is the owner's only");
    // the server's address and key in its environment, never on its command line
    let env = std::fs::read_to_string(tools.join("private-data-env.txt")).unwrap();
    assert_eq!(env, format!("http://127.0.0.1:{}\n{KEY}\n", s.port));
    let args = std::fs::read_to_string(tools.join("private-data-args.txt")).unwrap();
    assert!(!args.contains(KEY), "{args}");
    assert!(args.contains(&format!("--source {}", mine.canonicalize().unwrap().display())) && args.contains("--model m1")
            && args.contains("--text-count 5") && args.contains("--every-s 2.5") && args.contains("--data-dir"), "{args}");
    // the model list says each model's run config marks, for the job's model check
    let (_, m) = s.get("/api/models", true);
    assert!(m["models"].as_array().unwrap().iter().all(|x| x["marks"].is_array()), "{m}");
}
