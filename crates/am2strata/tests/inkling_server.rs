//! The Inkling 2 Mini stream (I4): a model whose call format is `Inkling` is read by `InklingParser`, fed the engine's
//! token ids, and its reasoning, answer and tool calls become the same deltas every other format does - checked here
//! over both `/v1/chat/completions` and `/v1/messages`, streamed and not, against the golden replies in the frontend
//! test data (one plain, `math_high`, one with two kinds of tool call, `tool_two`). No GPU: a mock engine replays the
//! golden token ids as the reply.

use serde_json::{json, Map, Value};

mod common;
use common::*;

/// The golden file that feeds every Inkling case.
fn golden() -> Value {
    let path = format!("{}/../strata-frontend/tests/data/inkling-outputs.json", env!("CARGO_MANIFEST_DIR"));
    serde_json::from_str(&std::fs::read_to_string(path).expect("the inkling golden")).expect("a valid golden")
}

/// The marker ids Inkling marks its parts with (output.rs), recognised by id regardless of the model's tokenizer.
const INK_HEADER: u32 = 200001; // <|message_model|>
const INK_THINK: u32 = 200008;  // <|content_thinking|>
const INK_TEXT: u32 = 200004;   // <|content_text|>
const INK_END: u32 = 200010;    // <|end_message|>
const INK_TOOL_JSON: u32 = 200049; // <|content_invoke_tool_json|>
const INK_STOP: u32 = 200006;   // <|content_model_end_sampling|> (the Inkling stop token)

/// A reply stream for one case, exactly as an Inkling model writes it: the marker ids above and every piece of text
/// encoded with the mock's tokenizer (qwen35). The parser reads the markers by id and decodes the text with the
/// same bytes llama.cpp did, so it streams reasoning, content and tool calls identical to the golden.
fn ids_of(case: &Value) -> Vec<u32> {
    let p = &case["parsed"];
    let mut v: Vec<u32> = Vec::new();
    if let Some(reasoning) = p["reasoning_content"].as_str() {
        if !reasoning.is_empty() { v.push(INK_HEADER); v.push(INK_THINK); v.extend(ids(reasoning)); v.push(INK_END); }
    }
    if let Some(content) = p["content"].as_str() {
        if !content.is_empty() { v.push(INK_HEADER); v.push(INK_TEXT); v.extend(ids(content)); v.push(INK_END); }
    }
    for c in p["tool_calls"].as_array().into_iter().flatten() {
        let name = c["function"]["name"].as_str().unwrap_or("");
        let args = c["function"]["arguments"].as_str().unwrap_or("");
        v.push(INK_HEADER);
        v.extend(ids(name));
        v.push(INK_TOOL_JSON);
        v.extend(ids(args));
        v.push(INK_END);
    }
    if v.is_empty() { v.push(INK_HEADER); v.push(INK_TEXT); v.extend(ids("")); }
    v.push(INK_STOP);
    v
}

/// A `/v1/chat/completions` answer merged from its chunks (streamed) or read whole (not streamed): content, reasoning,
/// the calls' (name, arguments) and the finish reason.
struct Chat {
    content: Value,
    reasoning: Option<String>,
    calls: Vec<(String, String)>,
    finish: Value,
}

fn chat_non_streamed(s: &Server, req: &Value) -> Chat {
    let (_, body) = s.post("/v1/chat/completions", req);
    let content_val = body["choices"][0]["message"]["content"].clone();
    Chat {
        content: if content_val.is_null() { Value::Null }
                  else if content_val.is_string() && !content_val.as_str().unwrap().is_empty() { json!(content_val.as_str().unwrap()) }
                  else { content_val },
        reasoning: body["choices"][0]["message"]["reasoning_content"].as_str().map(str::to_string),
        calls: body["choices"][0]["message"]["tool_calls"].as_array().unwrap_or(&vec![Value::Null]).iter().filter(|t| t.is_object())
            .map(|t| (t["function"]["name"].as_str().unwrap_or("").to_string(), t["function"]["arguments"].as_str().unwrap_or("").to_string()))
            .collect(),
        finish: body["choices"][0]["finish_reason"].clone(),
    }
}

fn chat_streamed(s: &Server, req: &Value) -> Chat {
    let mut sreq = req.clone();
    sreq["stream"] = json!(true);
    let ev = s.stream("/v1/chat/completions", &sreq);
    let (mut content, mut reasoning) = (String::new(), String::new());
    let mut by_index: std::collections::BTreeMap<u64, (String, String)> = Default::default();
    let mut finish = json!("stop");
    for (_name, v) in &ev {
        if let Some(f) = v["choices"][0].get("finish_reason") {
            if !f.is_null() { finish = f.clone(); }
        }
        let d = &v["choices"][0]["delta"];
        content.push_str(d["content"].as_str().unwrap_or(""));
        reasoning.push_str(d["reasoning_content"].as_str().unwrap_or(""));
        for tc in d["tool_calls"].as_array().unwrap_or(&vec![Value::Null]).iter().filter(|t| t.is_object()) {
            let idx = tc.get("index").and_then(Value::as_u64).unwrap_or(0);
            let f = &tc["function"];
            let cur = by_index.entry(idx).or_insert_with(|| (String::new(), String::new()));
            cur.0.push_str(f.get("name").and_then(Value::as_str).unwrap_or(""));
            cur.1.push_str(f.get("arguments").and_then(Value::as_str).unwrap_or(""));
        }
    }
    Chat {
        content: if content.is_empty() { Value::Null } else { json!(content) },
        reasoning: if reasoning.is_empty() { None } else { Some(reasoning) },
        calls: by_index.into_values().collect(),
        finish,
    }
}

/// A `/v1/messages` answer merged from its streamed events: reasoning, content, the tool uses' (name, input-object) and
/// the stop reason. Mirrors anthropic.rs collect() - blocks are held by index and a tool_use's partial input is parsed
/// only when its block stops.
fn anthropic_merged(items: &[(String, Value)]) -> (Option<String>, String, Vec<(String, Map<String, Value>)>, Value) {
    let mut blocks: Vec<Value> = Vec::new();
    let mut json_buf: Option<String> = None;
    let mut stop = Value::Null;
    for (name, v) in items {
        match name.as_str() {
            "content_block_start" => { blocks.push(v["content_block"].clone()); json_buf = None; }
            "content_block_delta" => {
                let d = &v["delta"];
                match d["type"].as_str() {
                    Some("thinking_delta") | Some("text_delta") => {
                        let key = if d["type"].as_str() == Some("thinking_delta") { "thinking" } else { "text" };
                        if let Some(b) = blocks.last_mut() {
                            b[key] = json!(format!("{}{}", b.get(key).and_then(Value::as_str).unwrap_or(""), d[key].as_str().unwrap_or("")));
                        }
                    }
                    Some("input_json_delta") => json_buf = Some(json_buf.unwrap_or_default() + d["partial_json"].as_str().unwrap_or("")),
                    _ => {}
                }
            }
            "content_block_stop" => {
                if let (Some(j), Some(b)) = (json_buf.take(), blocks.last_mut()) {
                    b["input"] = serde_json::from_str(&j).unwrap_or(json!({}));
                }
            }
            "message_delta" => {
                if let Some(s) = v["delta"]["stop_reason"].as_str() { stop = json!(s); }
            }
            _ => {}
        }
    }
    let (mut thinking, mut text) = (String::new(), String::new());
    let mut calls: Vec<(String, Map<String, Value>)> = Vec::new();
    for b in &blocks {
        match b["type"].as_str() {
            Some("thinking") => thinking.push_str(b.get("thinking").and_then(Value::as_str).unwrap_or("")),
            Some("text") => text.push_str(b.get("text").and_then(Value::as_str).unwrap_or("")),
            Some("tool_use") => calls.push((b["name"].as_str().unwrap_or("").to_string(), b["input"].as_object().cloned().unwrap_or_default())),
            _ => {}
        }
    }
    (if thinking.is_empty() { None } else { Some(thinking) }, text, calls, stop)
}

fn anthropic_non_streamed(s: &Server, req: &Value) -> (Option<String>, String, Vec<(String, Map<String, Value>)>, Value) {
    let (_, body) = s.post("/v1/messages", req);
    let mut thinking = String::new();
    let mut text = String::new();
    let mut calls: Vec<(String, Map<String, Value>)> = Vec::new();
    for b in body["content"].as_array().unwrap_or(&vec![Value::Null]) {
        match b["type"].as_str() {
            Some("thinking") => thinking.push_str(b.get("thinking").and_then(Value::as_str).unwrap_or("")),
            Some("text") => text.push_str(b.get("text").and_then(Value::as_str).unwrap_or("")),
            Some("tool_use") => calls.push((b["name"].as_str().unwrap_or("").to_string(), b["input"].as_object().cloned().unwrap_or_default())),
            _ => {}
        }
    }
    (if thinking.is_empty() { None } else { Some(thinking) }, text, calls, body["stop_reason"].clone())
}

fn requests_for(case: &Value) -> (Value, Value) {
    // an OpenAI request, and the matching Anthropic one (the tools added when the case declares any). Anthropic names a
    // tool's schema `input_schema` (OpenAI's `parameters`) and its name at the top level, so convert each tool.
    let openai = case["request"].clone();
    let mut anthropic = json!({"model": "x", "max_tokens": 512, "thinking": {"type": "disabled"}, "messages": case["request"]["messages"].clone()});
    if let Some(tools) = case["request"].get("tools") {
        let converted: Vec<Value> = tools.as_array().unwrap_or(&vec![]).iter().map(|t| {
            let f = &t["function"];
            json!({"type":"function","name":f.get("name").cloned().unwrap_or(json!("")),"input_schema":f.get("parameters").cloned().unwrap_or(json!({"type":"object"}))})
        }).collect();
        anthropic["tools"] = json!(converted);
    }
    (openai, anthropic)
}

/// A server's `/v1/chat/completions` tool calls compared against golden by name and parsed argument object. The
/// arguments arrive as JSON that the server spaces differently from the golden (`json.dumps` vs compact), so the
/// values are matched as objects rather than as text; the order is the model's own call order.
fn cmp_openai_calls(got: &[(String, String)], want: &[(String, Value)], label: &str) {
    assert_eq!(got.len(), want.len(), "{label}: {} calls", got.len());
    for (i, ((name, arg), (wname, wobj))) in got.iter().zip(want.iter()).enumerate() {
        assert_eq!(name, wname, "{label}: call {i} name");
        let got_obj = serde_json::from_str::<Value>(arg).unwrap_or_default();
        assert_eq!(&got_obj, wobj, "{label}: call {i} input (got {:?}, want {:?}", got_obj, wobj);
    }
}

fn check_case(s: &Server, case: &Value) {
    let parsed = &case["parsed"];
    let want_reasoning = parsed["reasoning_content"].as_str().unwrap_or("");
    let want_content = parsed["content"].as_str().unwrap_or("");
    let want_calls: Vec<(&str, &str)> = parsed["tool_calls"].as_array().into_iter().flatten()
        .map(|t| (t["function"]["name"].as_str().unwrap(), t["function"]["arguments"].as_str().unwrap())).collect();
    let finished_calls = !want_calls.is_empty();
    // golden calls by name and their parsed argument objects: compared against the server's, arguments matched as JSON
    // values (the server serialises them with json.dumps spacing, not the golden's compact form) so text is not comparable.
    let want_objs: Vec<(String, Value)> = parsed["tool_calls"].as_array().into_iter().flatten()
        .map(|t| (t["function"]["name"].as_str().unwrap().to_string(), serde_json::from_str::<Value>(t["function"]["arguments"].as_str().unwrap_or("{}")).unwrap_or_default()))
        .collect();

    let (openai_req, anth_req) = requests_for(case);

    // --- /v1/chat/completions ---
    let non = chat_non_streamed(s, &openai_req);
    let stream = chat_streamed(s, &openai_req);
    assert_eq!(non.reasoning.as_deref().unwrap_or(""), want_reasoning, "chat/non-streamed reasoning of {}", case["parsed"]["content"].as_str().unwrap_or("<calls>"));
    assert_eq!(non.content, if want_content.is_empty() { Value::Null } else { json!(want_content) }, "chat/non-streamed content of {}", case_name(case));
    cmp_openai_calls(&non.calls, &want_objs, &format!("chat/non-streamed calls of {}", case_name(case)));
    assert_eq!(non.finish.as_str().unwrap_or(""), if finished_calls { "tool_calls" } else { "stop" }, "chat/non-streamed finish of {}", case_name(case));
    assert_eq!(stream.reasoning.as_deref().unwrap_or(""), want_reasoning, "chat/streamed reasoning of {}", case_name(case));
    assert_eq!(stream.content, if want_content.is_empty() { Value::Null } else { json!(want_content) }, "chat/streamed content of {}", case_name(case));
    cmp_openai_calls(&stream.calls, &want_objs, &format!("chat/streamed calls of {}", case_name(case)));
    assert_eq!(stream.finish.as_str().unwrap_or(""), if finished_calls { "tool_calls" } else { "stop" }, "chat/streamed finish of {}", case_name(case));

    // --- /v1/messages ---
    let (non_thought, non_text, non_calls, non_stop) = anthropic_non_streamed(s, &anth_req);
    assert_eq!(non_thought.as_deref().unwrap_or(""), want_reasoning, "messages/non-streamed reasoning of {}", case_name(case));
    assert_eq!(non_text, want_content, "messages/non-streamed content of {}", case_name(case));
    assert_eq!(non_stop.as_str().unwrap_or(""), if finished_calls { "tool_use" } else { "end_turn" }, "messages/non-streamed stop_reason of {}", case_name(case));
    check_calls(&want_objs, &non_calls, &format!("messages/non-streamed calls of {}", case_name(case)));

    let mut areq = anth_req.clone();
    areq["stream"] = json!(true);
    let ev = s.stream("/v1/messages", &areq);
    let (s_thought, s_text, s_calls, s_stop) = anthropic_merged(&ev);
    assert_eq!(s_thought.as_deref().unwrap_or(""), want_reasoning, "messages/streamed reasoning of {}", case_name(case));
    assert_eq!(s_text, want_content, "messages/streamed content of {}", case_name(case));
    assert_eq!(s_stop.as_str().unwrap_or(""), if finished_calls { "tool_use" } else { "end_turn" }, "messages/streamed stop_reason of {}", case_name(case));
    check_calls(&want_objs, &s_calls, &format!("messages/streamed calls of {}", case_name(case)));

    println!("ok   {} : reasoning {} chars, content {:?}, {} call(s)", case_name(case), want_reasoning.len(), if want_content.is_empty() { Value::Null } else { json!(want_content) }, want_calls.len());
}

/// The golden argument objects (parsed) compared against the server's input objects.
fn check_calls(want: &Vec<(String, Value)>, got: &[(String, Map<String, Value>)], label: &str) {
    assert_eq!(got.len(), want.len(), "{label}: {} calls", got.len());
    for (i, (name, want_obj)) in want.iter().enumerate() {
        let g = got.get(i).expect("a call at index {i}");
        assert_eq!(&g.0, name, "{label}: call {i} name");
        assert_eq!(g.1.clone(), want_obj.as_object().cloned().unwrap_or_default(), "{label}: call {i} input (got {:?}, want {:?})", g.1, want_obj);
    }
}

fn case_name(case: &Value) -> String {
    case["parsed"]["reasoning_content"].as_str().unwrap_or("<>").chars().take(24).collect::<String>()
}

#[test]
fn inkling_streams_plain_and_tool_calls_over_both_apis() {
    let g = golden();
    // a fresh server per case (the mock replays its one scripted reply for every turn), each an Inkling model
    for (name, case) in g.as_object().unwrap() {
        let s = Server::start(&[Model { format: Some("inkling"), ..Model::default() }], &[ids_of(case)], json!({}));
        check_case(&s, case);
        println!("ok   {} : {}", name, match case.get("request").and_then(Value::as_object).map(|o| o.len().to_string()) { Some(n) => format!("{n} message(s), {} tool call type(s)", case["parsed"]["tool_calls"].as_array().map(|a| a.len()).unwrap_or(0)), None => "no request".into() });
    }
}
