//! Regression runs and training by grading (plan Part G) on tiny fixtures, on the CPU: the sandbox stops an endless
//! loop, a memory hog, a sleeper and a network call; answers are graded by really running them (Python and Rust); the
//! paired bootstrap flags a real drop and not noise; a looped model runs at each pass count (an untrained loop gives
//! its base's numbers exactly); and one round of grade-and-train runs end to end with a stand-in grader.

mod common;

use std::path::Path;

use common::*;
use serde_json::json;
use strata_workshop::gradetrain::{self, Grader, Prompt};
use strata_workshop::regress::{self, compare, CodeCheck, CodeLang, CodeRunner, CodeTask, TestResult};
use strata_workshop::sandbox::{self, Ending, Limits, Scratch};

fn sh() -> &'static Path {
    Path::new("/bin/sh")
}

fn quick() -> Limits {
    Limits { cpu_seconds: 1, wall_seconds: 5.0, memory_bytes: 256 << 20, limit_address_space: true, file_bytes: 1 << 20, no_network: true, output_bytes: 1 << 16 }
}

#[cfg(unix)]
#[test]
fn the_sandbox_stops_an_endless_loop_a_memory_hog_a_sleeper_and_the_network() {
    let dir = Scratch::new("t").unwrap();
    let t0 = std::time::Instant::now();
    let o = sandbox::run(sh(), &["-c", "while :; do :; done"], &dir.0, b"", &quick());
    assert_eq!(o.ending, Ending::CpuTime, "{o:?}");
    assert!(t0.elapsed().as_secs_f64() < 4.0, "stopped by the CPU limit, not the clock: {:?}", t0.elapsed());

    let o = sandbox::run(sh(), &["-c", "sleep 30"], &dir.0, b"", &Limits { wall_seconds: 0.5, ..quick() });
    assert_eq!(o.ending, Ending::WallTime, "{o:?}");
    assert!(o.seconds < 3.0);

    let o = sandbox::run(sh(), &["-c", "read x; echo got $x; exit 3"], &dir.0, b"7\n", &quick());
    assert_eq!((o.ending.clone(), o.stdout.trim()), (Ending::Exited(3), "got 7"), "{o:?}");

    let Some(py) = sandbox::find_program("python3") else {
        eprintln!("no python3 here: the memory and network checks are skipped");
        return;
    };
    let o = sandbox::run(&py, &["-I", "-c", "x = bytearray(2 * 1024**3); print(len(x))"], &dir.0, b"", &quick());
    assert_eq!(o.ending, Ending::Memory, "{o:?}");
    // an allocation in a loop that keeps going: stopped by the memory limit too
    let o = sandbox::run(&py, &["-I", "-c", "a = []\nwhile True: a.append(bytearray(10**7))"], &dir.0, b"", &quick());
    assert_eq!(o.ending, Ending::Memory, "{o:?}");
    if sandbox::network_isolation_available() {
        let o = sandbox::run(&py, &["-I", "-c", "import socket; socket.create_connection(('1.1.1.1', 53), timeout=2)"], &dir.0, b"", &quick());
        assert!(!o.ok() && o.network_isolated, "no network inside: {o:?}");
        assert!(o.stderr.contains("unreachable") || o.stderr.contains("Errno"), "{}", o.stderr);
    }
}

#[test]
fn answers_are_graded_by_running_them_with_their_tests() {
    let r = CodeRunner::detect();
    if r.can(CodeLang::Python) {
        let task = CodeTask { id: "t".into(), prompt: String::new(), lang: CodeLang::Python, check: CodeCheck::Program {
            header: String::new(), stub: Some(("def add(a, b):\n    \"\"\"a plus b\"\"\"\n".into(), "add".into())),
            suffix: "def check(f):\n    assert f(2, 3) == 5\n\ncheck(add)\n".into() } };
        assert!(r.grade(&task, "```python\ndef add(a, b):\n    return a + b\n```").passed, "a whole function");
        assert!(r.grade(&task, "```python\n    return a + b\n```").passed, "a body after the stub");
        assert!(!r.grade(&task, "```python\ndef add(a, b):\n    return a - b\n```").passed);
        let g = r.grade(&task, "```python\ndef add(a, b):\n    while True: pass\n```");
        assert!(!g.passed && g.ending == "cpu time limit", "{g:?}");
        let stdio = CodeTask { id: "s".into(), prompt: String::new(), lang: CodeLang::Python,
                               check: CodeCheck::Stdio(vec![("2 3\n".into(), "5\n".into()), ("1 1\n".into(), "2".into())]) };
        let g = r.grade(&stdio, "```python\na, b = map(int, input().split())\nprint(a + b)\n```");
        assert!(g.passed && g.fraction == 1.0, "{g:?}");
        let g = r.grade(&stdio, "```python\nprint(5)\n```");
        assert!(!g.passed && g.fraction == 0.5, "{g:?}");
    }
    if r.can(CodeLang::Rust) {
        let task = CodeTask { id: "r".into(), prompt: String::new(), lang: CodeLang::Rust, check: CodeCheck::Program {
            header: String::new(), stub: None, suffix: "#[cfg(test)]\nmod t { use super::*; #[test] fn a() { assert_eq!(double(4), 8); } }".into() } };
        assert!(r.grade(&task, "```rust\npub fn double(x: i32) -> i32 { x * 2 }\n```").passed);
        assert!(!r.grade(&task, "```rust\npub fn double(x: i32) -> i32 { x + 2 }\n```").passed, "a failing test");
        assert!(r.grade(&task, "```rust\npub fn double(x: i32) -> i32 { x * \"2\" }\n```").ending.starts_with("does not compile"));
    }
}

fn result(name: &str, scores: &[f64]) -> TestResult {
    let v = json!({"test": name, "kind": "code", "passes": 1, "metric": "pass@1", "higher_is_better": true,
                   "items": scores.iter().enumerate().map(|(i, s)| json!({"id": i.to_string(), "score": s})).collect::<Vec<_>>()});
    TestResult::from_json(&v).unwrap()
}

#[test]
fn the_bootstrap_flags_a_real_drop_and_not_noise() {
    // 60 tasks: 48 passed before; after, 18 (a real drop)
    let before: Vec<f64> = (0..60).map(|i| if i % 5 == 4 { 0.0 } else { 1.0 }).collect();
    let dropped: Vec<f64> = (0..60).map(|i| if i % 10 < 3 { 1.0 } else { 0.0 }).collect();
    let c = compare(&result("x", &dropped), &result("x", &before)).unwrap();
    assert!(c.regressed && c.delta < -0.4 && c.ci.1 < 0.0, "{c:?}");
    // the same pass rate with three tasks flipped each way: noise
    let mut noisy = before.clone();
    for (a, b) in [(4, 0), (9, 1), (14, 2)] {
        noisy[a] = 1.0;
        noisy[b] = 0.0;
    }
    let c = compare(&result("x", &noisy), &result("x", &before)).unwrap();
    assert!(!c.regressed && !c.improved && c.ci.0 < 0.0 && c.ci.1 > 0.0, "{c:?}");
    // a small real drop (6 of 60 lost, none gained) is still seen when it is consistent
    let mut small = before.clone();
    for i in [0, 1, 2, 3, 5, 6] {
        small[i] = 0.0;
    }
    assert!(compare(&result("x", &small), &result("x", &before)).unwrap().regressed);
    // a loss (lower is better) that went up
    let mut worse = result("l", &[1.0, 1.1, 0.9, 1.2, 1.0, 0.95]);
    worse.higher_is_better = false;
    let mut base = result("l", &[0.5, 0.6, 0.4, 0.7, 0.5, 0.45]);
    base.higher_is_better = false;
    assert!(compare(&worse, &base).unwrap().regressed);
    assert!(compare(&base, &worse).unwrap().improved);
}

/// A suite of every local kind on the tiny chat model: loss, Python code (jsonl), QA (jsonl).
fn tiny_suite(root: &Path) -> std::path::PathBuf {
    write_jsonl(&root.join("heldout.jsonl"), &chat_rows(6));
    write_jsonl(&root.join("code.jsonl"), &[
        json!({"id": "add", "prompt": "Write add(a, b).", "test": "assert add(1, 2) == 3"}),
        json!({"id": "echo", "prompt": "Echo the input.", "cases": [{"input": "x\n", "output": "x"}]}),
    ]);
    write_jsonl(&root.join("qa.jsonl"), &[
        json!({"id": "q1", "context": "The sky is blue.", "question": "What colour is the sky?", "answers": ["blue"]}),
        json!({"id": "q2", "context": "Snow is white.", "question": "What colour is snow?", "answers": ["white"]}),
        json!({"id": "q3", "context": "No answer.", "question": "Why?", "answers": [""]}),
    ]);
    let suite = root.join("suite.json");
    std::fs::write(&suite, serde_json::to_string(&json!({"name": "tiny", "tests": [
        {"name": "held-out", "kind": "loss", "data": "heldout.jsonl"},
        {"name": "code", "kind": "code", "format": "jsonl", "data": "code.jsonl", "max_tokens": 6},
        {"name": "qa", "kind": "qa", "format": "jsonl", "data": "qa.jsonl", "max_tokens": 4},
    ]})).unwrap()).unwrap();
    suite
}

#[test]
fn a_looped_model_runs_the_suite_at_each_pass_count_and_its_report_compares() {
    let root = temp_dir("regress");
    let (base, looped) = (root.join("model"), root.join("looped"));
    chat_model(&base);
    strata_workshop::model::make_looped(&base, &looped, strata_workshop::model::LoopSpec { start: 1, end: 2, passes: 2 }).unwrap();
    let suite = tiny_suite(&root);
    let mut o = regress::Options::new(&looped, &suite, root.join("r1"));
    o.passes = vec![1, 2];
    let mut events = vec![];
    let r = regress::run(&o, None, &mut |v| events.push(v), &|| false).unwrap();
    assert_eq!(r.results.len(), 6, "three tests at two pass counts");
    assert!(r.results.iter().all(|t| t.error.is_none()), "{:?}", r.results.iter().map(|t| &t.error).collect::<Vec<_>>());
    let qa = r.results.iter().find(|t| t.test == "qa").unwrap();
    assert_eq!((qa.items.len(), qa.skipped), (2, 1), "the unanswerable question is left out");
    // the gate check: an untrained loop gives its base's numbers exactly
    let loss = |p: usize| r.results.iter().find(|t| t.test == "held-out" && t.passes == p).unwrap().items.clone();
    assert_eq!(loss(1), loss(2), "closed gates: pass 2 is pass 1");
    for f in ["report.json", "report.md", "answers.jsonl"] {
        assert!(root.join("r1").join(f).exists(), "{f}");
    }
    assert!(events.iter().any(|e| e["event"] == "progress"));
    // the same model again, against the first report: nothing regressed
    let mut again = o.clone();
    again.out = root.join("r2");
    again.against = vec![root.join("r1")];
    let r2 = regress::run(&again, None, &mut |_| {}, &|| false).unwrap();
    assert!(!r2.comparisons.is_empty() && r2.regressions().is_empty(), "{:?}", r2.json["regressions"]);
    assert!(std::fs::read_to_string(root.join("r2/report.md")).unwrap().contains("No regressions"));
    // a pass count the model does not have
    let mut bad = o.clone();
    bad.passes = vec![3];
    assert!(regress::run(&bad, None, &mut |_| {}, &|| false).is_err());
    // a judge test without a judge reports why, and the rest still runs
    let js = root.join("judge-suite.json");
    write_jsonl(&root.join("j.jsonl"), &[json!({"prompt": "Say hi."})]);
    std::fs::write(&js, r#"{"tests": [{"name": "j", "kind": "judge", "data": "j.jsonl"}, {"name": "l", "kind": "loss", "data": "heldout.jsonl", "limit": 2}]}"#).unwrap();
    let rj = regress::run(&regress::Options::new(&base, &js, root.join("rj")), None, &mut |_| {}, &|| false).unwrap();
    assert!(rj.results[0].error.as_deref().unwrap().contains("no judge") && rj.results[1].items.len() == 2);
}

/// Prefers longer answers (a stand-in for running code or asking a judge): any answer of 3 or more characters passes.
struct LengthGrader;

impl Grader for LengthGrader {
    fn grade(&self, _p: &Prompt, a: &str) -> anyhow::Result<f64> {
        Ok(a.chars().count() as f64)
    }
    fn passing(&self) -> f64 {
        3.0
    }
    fn name(&self) -> String {
        "length".into()
    }
}

#[test]
fn one_round_of_grade_and_train_runs_end_to_end() {
    let root = temp_dir("grade");
    let model = root.join("model");
    chat_model(&model);
    let suite = tiny_suite(&root);
    let prompts: Vec<Prompt> = ["What colour is the sky?", "What colour is snow?", "Name a fruit."].iter().enumerate()
        .map(|(i, t)| Prompt { id: format!("p{i}"), text: t.to_string(), code: None, judge: None }).collect();
    let mut o = gradetrain::Options::new(&model, prompts, &suite, root.join("gt"));
    o.samples = 2;
    o.rounds = 1;
    o.max_tokens = 6;
    o.train.max_steps = Some(2);
    o.train.accumulation = 1;
    o.train.max_len = 128;
    let mut events = vec![];
    let r = gradetrain::run(&o, &LengthGrader, None, &mut |v| events.push(v), &|| false).unwrap();
    assert_eq!(r.rounds.len(), 1);
    let round = &r.rounds[0];
    assert_eq!(round.sampled, 6, "two answers to each of three prompts");
    let gt = root.join("gt");
    let samples = std::fs::read_to_string(gt.join("round-1/samples.jsonl")).unwrap();
    assert_eq!(samples.lines().count(), 6);
    assert!(round.kept > 0, "the stand-in grader passes the answers of 3 characters or more: {round:?}");
    {
        assert!(gt.join("round-1/adapter/adapter_model.safetensors").exists(), "trained on the best");
        assert!(gt.join("round-1/report/report.json").exists(), "the suite ran on it");
        assert_eq!(round.promoted, round.regressions == 0, "promoted exactly when nothing regressed");
        assert_eq!(r.promoted.is_some(), round.promoted);
        let best = std::fs::read_to_string(gt.join("round-1/best.jsonl")).unwrap();
        assert_eq!(best.lines().count(), round.kept, "one best answer per prompt");
    }
    assert!(gt.join("round-0/report/report.json").exists(), "the start was measured first");
    let log = std::fs::read_to_string(gt.join("rounds.jsonl")).unwrap();
    let line: serde_json::Value = serde_json::from_str(log.lines().next().unwrap()).unwrap();
    assert_eq!(line["round"], 1);
    assert!(std::fs::read_to_string(gt.join("rounds.md")).unwrap().contains("| 1 |"));
    assert!(gt.join("promoted.json").exists());
    assert!(events.iter().any(|e| e["event"] == "progress" && e["loss"].is_number()), "the training steps are reported");
}

/// The downloaded benchmark files parse (they are data, never in the repository): set `AM2_EVAL_DIR` to the folder
/// with humaneval.jsonl, mbpp-sanitized.jsonl, livecodebench-v6.jsonl, codeforces-test.jsonl, strandset-rust-test.jsonl
/// and, optionally, `AM2_INDICQA` to an IndicQA file. Without them this test checks nothing.
#[test]
fn the_benchmark_files_load_when_present() {
    let Some(dir) = std::env::var_os("AM2_EVAL_DIR").map(std::path::PathBuf::from) else { return };
    for (file, format, at_least) in [("humaneval.jsonl", "humaneval", 164), ("mbpp-sanitized.jsonl", "mbpp", 250), ("livecodebench-v6.jsonl", "livecodebench", 100),
                                     ("codeforces-test.jsonl", "codeforces", 300), ("strandset-rust-test.jsonl", "strandset", 20)] {
        let (tasks, skipped, why) = regress::load_code(&dir.join(file), format, CodeLang::Python).unwrap();
        eprintln!("{file}: {} tasks, {skipped} left out ({why:?})", tasks.len());
        assert!(tasks.len() >= at_least, "{file}: {} tasks", tasks.len());
        // the reference solutions pass where the file has them
        let r = CodeRunner::detect();
        if format == "humaneval" && r.can(CodeLang::Python) {
            let rows: Vec<serde_json::Value> = std::fs::read_to_string(dir.join(file)).unwrap().lines().take(5).map(|l| serde_json::from_str(l).unwrap()).collect();
            for (t, row) in tasks.iter().zip(&rows) {
                let g = r.grade(t, &format!("```python\n{}```", row["canonical_solution"].as_str().unwrap()));
                assert!(g.passed, "{}: {g:?}", t.id);
            }
        }
        if format == "mbpp" && r.can(CodeLang::Python) {
            let rows: Vec<serde_json::Value> = std::fs::read_to_string(dir.join(file)).unwrap().lines().take(5).map(|l| serde_json::from_str(l).unwrap()).collect();
            for (t, row) in tasks.iter().zip(&rows) {
                assert!(r.grade(t, &format!("```python\n{}\n```", row["code"].as_str().unwrap())).passed, "{}", t.id);
            }
        }
        if format == "strandset" && r.can(CodeLang::Rust) {
            // the dataset's own answers: many are fragments that do not drop into their test file (a method without
            // its impl), so this only reports how many do
            let rows: Vec<serde_json::Value> = std::fs::read_to_string(dir.join(file)).unwrap().lines().map(|l| serde_json::from_str(l).unwrap()).collect();
            let mut passed = 0;
            for t in tasks.iter().take(12) {
                let n: usize = t.id.rsplit('#').next().unwrap().parse().unwrap();
                let out: serde_json::Value = serde_json::from_str(rows[n]["output_data"].as_str().unwrap()).unwrap();
                let code = ["code_after", "fixed_code", "test_cases", "code"].iter().find_map(|k| out[*k].as_str().map(str::to_string)
                    .or_else(|| out[*k].as_array().map(|a| a.iter().filter_map(|x| x.as_str()).collect::<Vec<_>>().join("\n")))).unwrap_or_default();
                let g = r.grade(t, &format!("```rust\n{code}\n```"));
                if g.passed { passed += 1; } else { eprintln!("strandset {}: {}", t.id, g.ending); }
            }
            eprintln!("strandset: {passed} of 12 reference answers pass");
        }
    }
    if let Some(q) = std::env::var_os("AM2_INDICQA") {
        let (items, unanswerable) = regress::load_qa(Path::new(&q), "squad").unwrap();
        assert!(!items.is_empty() && unanswerable > 0, "{} items, {unanswerable} unanswerable", items.len());
    }
}

/// The command: `regress` and `grade-train` with `--json` end in a done event that carries the regressions and the
/// rounds (what the server's jobs show).
#[test]
fn the_commands_report_as_json_events() {
    let root = temp_dir("regress-cli");
    let model = root.join("model");
    chat_model(&model);
    let suite = tiny_suite(&root);
    let exe = env!("CARGO_BIN_EXE_am2strata-workshop");
    let events = |args: &[&str]| -> Vec<serde_json::Value> {
        let o = std::process::Command::new(exe).arg("--json").args(args).env_remove("AM2STRATA_URL").output().unwrap();
        assert!(o.status.success(), "{}\n{}", String::from_utf8_lossy(&o.stdout), String::from_utf8_lossy(&o.stderr));
        String::from_utf8_lossy(&o.stdout).lines().map(|l| serde_json::from_str(l).unwrap()).collect()
    };
    let (m, s, r1) = (model.display().to_string(), suite.display().to_string(), root.join("r1").display().to_string());
    let ev = events(&["regress", "--model", &m, "--suite", &s, "--out", &r1]);
    let done = ev.last().unwrap();
    assert_eq!((done["event"].as_str(), done["kind"].as_str()), (Some("done"), Some("report")), "{done}");
    assert_eq!(done["summary"].as_array().unwrap().len(), 3);
    let r2 = root.join("r2").display().to_string();
    let ev = events(&["regress", "--model", &m, "--suite", &s, "--out", &r2, "--against", &r1]);
    assert_eq!(ev.last().unwrap()["regressions"], json!([]));
    write_jsonl(&root.join("prompts.jsonl"), &[json!({"id": "a", "prompt": "Write add(a, b).", "test": "assert add(1, 2) == 3"})]);
    let (p, gt) = (root.join("prompts.jsonl").display().to_string(), root.join("gt").display().to_string());
    let ev = events(&["grade-train", "--model", &m, "--data", &p, "--samples", "2", "--grader", "code", "--rounds", "1", "--suite", &s, "--out", &gt,
                      "--max-tokens", "4", "--max-steps", "1"]);
    let done = ev.last().unwrap();
    assert_eq!(done["kind"], "grade-train", "{done}");
    // a random tiny model writes no working code: nothing to train on, nothing promoted, and the log says so
    assert_eq!(done["rounds"][0]["kept"], 0, "{done}");
    assert_eq!(done["promoted"], serde_json::Value::Null);
    assert!(std::fs::read_to_string(root.join("gt/rounds.md")).unwrap().contains("no answer was good enough"));
}
