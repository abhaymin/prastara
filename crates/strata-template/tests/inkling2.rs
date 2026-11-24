//! The Inkling 2 Mini chat prompt against prompts recorded from llama.cpp / PolyStrata on `golden/templates.json`.
//! Renders every request with the model's own template (from the GGUF header) and compares the prompt exactly.
//!
//! The golden file is a map of case name -> {request, prompt}; each request is rendered and compared to its prompt.

use serde_json::Value;
use strata_template::ChatTemplate;

const TPL: &str = "/home/abhaym/Development/am2strata-inkling/crates/strata-template/testdata/chat_template-inkling2.jinja";
const CASES: &str = "/home/abhaym/Development/am2strata-inkling/crates/strata-template/testdata/templates-inkling2.json";

/// PolyStrata / Prastara canonicalize a tool call whose arguments came in as a JSON string into the parsed object the
/// template's `raise_exception("...not a JSON string; canonicalize upstream")` guard expects, so do the same here.
fn canonicalize(messages: &[Value]) -> Vec<Value> {
    messages
        .iter()
        .map(|m| {
            if m.get("role").and_then(Value::as_str) != Some("assistant") {
                return m.clone();
            }
            let mut m = m.clone();
            if let Some(tools) = m.get_mut("tool_calls").and_then(|t| t.as_array_mut()) {
                for tc in tools.iter_mut() {
                    if let Some(fn_) = tc.get_mut("function").and_then(|f| f.as_object_mut()) {
                        if let Some(Value::String(s)) = fn_.get_mut("arguments") {
                            if let Ok(parsed) = serde_json::from_str::<Value>(s) {
                                fn_.insert("arguments".into(), parsed);
                            }
                        }
                    }
                }
            }
            m
        })
        .collect()
}

fn render_case(c: &Value) -> Result<String, String> {
    let req = &c["request"];
    let owned: Vec<Value> = req["messages"].as_array().unwrap().iter().map(Value::clone).collect();
    let messages = canonicalize(&owned);
    let tools = if req["tools"].is_null() { None } else { Some(&req["tools"]) };
    let mut extra: serde_json::Map<String, Value> = serde_json::Map::new();
    if let Some(kwargs) = req.get("chat_template_kwargs") {
        if let Some(obj) = kwargs.as_object() {
            for (k, v) in obj {
                extra.insert(k.clone(), v.clone());
            }
        }
    }
    let tpl = ChatTemplate::from_file(TPL).map_err(|e| format!("load template: {e}"))?;
    tpl.render(&messages, tools, false, &extra)
        .map_err(|e| format!("{e}"))
}

#[test]
fn inkling2_prompts_match_reference() {
    let g: Value = serde_json::from_str(&std::fs::read_to_string(CASES).unwrap()).unwrap();
    assert_eq!(g.as_object().unwrap().len(), 8, "expected 8 golden template cases");
    let mut failures = Vec::new();
    for (name, c) in g.as_object().unwrap() {
        let want = c["prompt"].as_str().unwrap();
        match render_case(c) {
            Ok(got) if got == want => {}
            Ok(got) => {
                let at = got.chars().zip(want.chars()).position(|(a, b)| a != b).unwrap_or(got.len().min(want.len()));
                let ctx = |s: &str| s.chars().skip(at.saturating_sub(30)).take(80).collect::<String>();
                failures.push(format!("{name}: differs at char {at}\n   got:  {:?}\n   want: {:?}", ctx(&got), ctx(want)));
            }
            Err(e) => failures.push(format!("{name}: error {e}")),
        }
    }
    assert!(failures.is_empty(), "inkling2: {} mismatches:\n{}", failures.len(), failures.join("\n"));
}
