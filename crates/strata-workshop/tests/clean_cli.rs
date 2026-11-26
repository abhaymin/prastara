//! `am2strata-workshop clean` as a command: the `--json` events (one per rule), the tokenizer's lengths (the
//! trainer's rule), the card, and the refusals.

mod common;

use std::process::Command;

use serde_json::{json, Value};

fn run(args: &[&str]) -> (Vec<Value>, bool) {
    let out = Command::new(env!("CARGO_BIN_EXE_am2strata-workshop")).arg("--json").args(args).output().unwrap();
    let events = String::from_utf8_lossy(&out.stdout).lines().map(|l| serde_json::from_str(l).unwrap_or_else(|_| panic!("not a JSON event: {l}"))).collect();
    (events, out.status.success())
}

fn dropped(events: &[Value], rule: &str) -> u64 {
    events.iter().find(|e| e["event"] == "rule" && e["rule"] == rule).and_then(|e| e["dropped"].as_u64()).unwrap_or_else(|| panic!("no {rule} event: {events:?}"))
}

#[test]
fn clean_speaks_the_event_protocol_measures_in_tokens_and_writes_a_card() {
    let (root, model, data) = common::model_and_data("cli-clean");
    let mut rows = common::chat_rows(24);
    rows.push(rows[0].clone()); // a duplicate
    rows.push(json!({"messages": [{"role": "user", "content": "x".repeat(300)}, {"role": "assistant", "content": "It is long."}]})); // fills --max-len 64
    rows.push(json!({"messages": [{"role": "user", "content": "Say it."}, {"role": "assistant", "content": "yes indeed yes indeed yes indeed"}]}));
    rows.push(json!({"messages": [{"role": "user", "content": "Think."}, {"role": "assistant", "content": "<think>hm</think>It is so."}]}));
    let extra = root.join("extra.jsonl");
    common::write_jsonl(&extra, &rows[24..]);
    let held = root.join("held.jsonl");
    common::write_jsonl(&held, &rows[1..2]);
    let out = root.join("clean.jsonl");
    let p = |x: &std::path::Path| x.to_str().unwrap().to_string();
    let (events, ok) = run(&["clean", &p(&data), &p(&extra), "--out", &p(&out), "--model", &p(&model), "--max-len", "64", "--exclude", &p(&held)]);
    assert!(ok, "{events:?}");
    assert_eq!(events[0]["event"], "start");
    for (rule, n) in [("held_out", 1), ("exact_duplicate", 1), ("prompt_too_long", 1), ("repetition", 1), ("template_junk", 1), ("too_short", 0), ("bad_row", 0)] {
        assert_eq!(dropped(&events, rule), n, "{rule}: {events:?}");
    }
    let done = events.last().unwrap();
    assert_eq!((done["event"].as_str(), done["kind"].as_str()), (Some("done"), Some("dataset")));
    let kept: Vec<Value> = std::fs::read_to_string(&out).unwrap().lines().map(|l| serde_json::from_str(l).unwrap()).collect();
    assert_eq!(kept.len(), 23);
    assert!(!kept.contains(&rows[1]), "the held-out row leaked");
    let card = std::fs::read_to_string(root.join("clean.card.md")).unwrap();
    assert!(card.contains("**23 kept**") && card.contains("| `prompt_too_long` | 1 |") && card.contains("train.jsonl") && card.contains("extra.jsonl"), "{card}");
    assert!(card.contains("### `repetition`") && card.contains("yes indeed"), "examples of each rule: {card}");

    // without a tokenizer the lengths are words and the length rule is not checked
    let (events, ok) = run(&["clean", &p(&extra), "--out", &p(&root.join("words.jsonl")), "--min-answer-tokens", "1", "--report", &p(&root.join("words.md"))]);
    assert!(ok, "{events:?}");
    assert_eq!(dropped(&events, "prompt_too_long"), 0);
    assert!(std::fs::read_to_string(root.join("words.md")).unwrap().contains("lengths in words"));

    // refused: the output is an input; every row dropped; a bad setting
    let (events, ok) = run(&["clean", &p(&data), "--out", &p(&data)]);
    assert!(!ok && events.last().unwrap()["message"].as_str().unwrap().contains("cannot be one of the inputs"), "{events:?}");
    let (events, ok) = run(&["clean", &p(&data), "--out", &p(&root.join("none.jsonl")), "--min-answer-tokens", "1000"]);
    assert!(!ok && events.last().unwrap()["message"].as_str().unwrap().contains("no rows kept"), "{events:?}");
    assert!(!root.join("none.jsonl").exists() && !root.join("none.card.md").exists(), "no empty dataset is left behind");
    let (events, ok) = run(&["clean", &p(&data), "--out", &p(&root.join("x.jsonl")), "--dedupe", "fuzzy"]);
    assert!(!ok && events.last().unwrap()["message"].as_str().unwrap().contains("--dedupe"), "{events:?}");
}
