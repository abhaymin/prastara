//! Inkling 2 output parsing (I2): feed the model's tokens one at a time and all at once, and check that reasoning,
//! content and tool calls come out as llama.cpp's server put them in `golden/outputs.json`. The golden has, per case,
//! the generated token ids (`tokens`) and what the reference made of it (`parsed`: reasoning_content, content,
//! tool_calls with the arguments as a JSON string, plus finish_reason). Only these token ids are needed, so they are
//! decoded from `inkling-tokens.json` (one id per string), not by loading the model.

use serde_json::Value;
use strata_frontend::output::InklingParser;

/// The token ids -> their text, exactly as the tokenizer stores them (the reference decodes them the same way).
fn decoder() -> impl Fn(u32) -> Vec<u8> + Send + Sync + 'static {
    let path = format!("{}/tests/data/inkling-tokens.json", env!("CARGO_MANIFEST_DIR"));
    let v: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).expect("load token-id strings");
    move |id: u32| match v.get(id.to_string().as_str()) {
        Some(Value::String(s)) => s.clone().into_bytes(),
        _ => Vec::new(),
    }
}

/// One case fed three ways, each compared to the golden; returns the name and whether it matched.
fn check_case(name: &str, tokens: &[u32], parsed: &Value) -> bool {
    let want_reasoning = parsed["reasoning_content"].as_str().unwrap_or("");
    let want_content = parsed["content"].as_str().unwrap_or("");
    let wants_calls: Vec<(&str, &str)> = parsed["tool_calls"]
        .as_array()
        .into_iter()
        .flatten()
        .map(|c| (c["function"]["name"].as_str().unwrap(), c["function"]["arguments"].as_str().unwrap()))
        .collect();

    // one token at a time
    let mut p1 = InklingParser::new(decoder());
    for &t in tokens {
        p1.feed(t);
    }
    let _ = p1.finish();
    let (r1, c1, calls1) = assemble(&p1);

    // all the tokens at once
    let mut p2 = InklingParser::new(decoder());
    p2.feed_tokens(tokens);
    let _ = p2.finish();
    let (r2, c2, calls2) = assemble(&p2);

    let (ok_r, ok_c) = (r1 == want_reasoning.to_string() && r1 == r2, c1 == want_content.to_string() && c1 == c2);
    let ok_calls = same_calls(&calls1, &calls2, &wants_calls).0;

    if !ok_r || !ok_c || !ok_calls {
        println!("FAIL {name}:");
        if !ok_r { println!("  reasoning: got {:?} want {:?}", r1, want_reasoning); }
        if !ok_c { println!("  content:   got {:?} want {:?}", c1, want_content); }
        if !ok_calls {
            let g: Vec<String> = calls1.iter().map(|(n, a)| format!("{n}={a}")).collect();
            let a2: Vec<String> = calls2.iter().map(|(n, a)| format!("{n}={a}")).collect();
            println!("  tool_calls mismatch (want {}): got {:?}; all-at-once {:?}", wants_calls.len(), g, a2);
        }
        return false;
    }
    println!("ok  {name}: reasoning={:?} content={:?} tool_calls={}", r1, c1, calls1.len());
    true
}

/// Assemble the parser's output into (reasoning, content, vec of (name, arguments)) and compare it to the golden.
fn assemble(p: &InklingParser) -> (String, String, Vec<(String, String)>) {
    let calls = p.tool_calls();
    (p.reasoning_content(), p.content(), calls.into_iter().map(|c| (c.name, c.arguments)).collect())
}

/// The one-at-a-time result and the all-at-once result must also agree with each other; returns (matched golden, a display hint).
fn same_calls(got: &[(String, String)], all: &[(String, String)], want: &[( &str, &str )]) -> (bool, String) {
    if got.len() != want.len() || !got.iter().zip(want).all(|((n, a), (wn, wa))| n == wn && a == wa) {
        return (false, format!("{} calls", got.len()));
    }
    (got == all, format!("{} calls", got.len()))
}

#[test]
fn inkling_output_matches_reference() {
    let path = format!("{}/tests/data/inkling-outputs.json", env!("CARGO_MANIFEST_DIR"));
    let g: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).expect("load golden");

    let mut failures = Vec::new();
    for (name, c) in g.as_object().unwrap() {
        let tokens: Vec<u32> = c["tokens"].as_array().unwrap().iter().map(|t| t.as_u64().unwrap() as u32).collect();
        if !check_case(name, &tokens, &c["parsed"]) {
            failures.push(name.clone());
        }
    }
    assert!(failures.is_empty(), "inkling golden cases differ: {}", failures.join(", "));
}

/// A streamed reply gets its text as it arrives, not one block at the end: the Hindi answer (101 tokens) gives several
/// reasoning and several content events, and joined they are the golden text.
#[test]
fn inkling_streams_text_as_it_arrives() {
    use strata_frontend::output::Event;
    let path = format!("{}/tests/data/inkling-outputs.json", env!("CARGO_MANIFEST_DIR"));
    let g: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let c = &g["hindi"];
    let tokens: Vec<u32> = c["tokens"].as_array().unwrap().iter().map(|t| t.as_u64().unwrap() as u32).collect();
    let mut p = InklingParser::new(decoder());
    let (mut r, mut t) = (Vec::new(), Vec::new());
    for id in tokens {
        for e in p.feed(id).into_iter().chain(std::iter::empty()) {
            match e { Event::Reasoning(s) => r.push(s), Event::Content(s) => t.push(s), _ => {} }
        }
    }
    for e in p.finish() { match e { Event::Reasoning(s) => r.push(s), Event::Content(s) => t.push(s), _ => {} } }
    assert!(r.len() > 3 && t.len() > 3, "several deltas each: {} reasoning, {} content", r.len(), t.len());
    assert_eq!(t.concat(), c["parsed"]["content"].as_str().unwrap());
}

/// A reply cut by max_tokens keeps what it wrote: cut inside the answer, the events carry the answer so far (a prefix
/// of the golden one), and cut inside the reasoning, the reasoning so far. Before, an unclosed block was dropped.
#[test]
fn inkling_reply_cut_by_max_tokens_keeps_its_text() {
    use strata_frontend::output::Event;
    let path = format!("{}/tests/data/inkling-outputs.json", env!("CARGO_MANIFEST_DIR"));
    let g: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let c = &g["hindi"];
    let tokens: Vec<u32> = c["tokens"].as_array().unwrap().iter().map(|t| t.as_u64().unwrap() as u32).collect();
    let answer_at = tokens.iter().rposition(|&t| t == 200004).expect("the answer marker");
    for (cut, in_answer) in [(answer_at + 6, true), (answer_at / 2, false)] {
        let mut p = InklingParser::new(decoder());
        let (mut r, mut t) = (String::new(), String::new());
        for &id in &tokens[..cut] {
            for e in p.feed(id) { match e { Event::Reasoning(s) => r.push_str(&s), Event::Content(s) => t.push_str(&s), _ => {} } }
        }
        for e in p.finish() { match e { Event::Reasoning(s) => r.push_str(&s), Event::Content(s) => t.push_str(&s), _ => {} } }
        if in_answer {
            assert!(!t.is_empty() && c["parsed"]["content"].as_str().unwrap().starts_with(&t), "cut in the answer keeps a prefix: {t:?}");
        } else {
            assert!(!r.is_empty() && t.is_empty(), "cut in the reasoning keeps the reasoning so far");
        }
    }
}
