//! The minor-safety check (src/minor_safety.rs) end to end against the mock engine: an uncensored model's request
//! that matches is refused before anything is generated, a reply that matches stops with an error, ordinary chats are
//! untouched, and a model without the marks is not checked. The texts are abstract placeholders that trip the rules,
//! never real requests.

mod common;

use common::*;
use serde_json::{json, Value};

fn user(text: &str) -> Value {
    json!({"role": "user", "content": text})
}

fn uncensored() -> Model<'static> {
    Model { id: "mock-u", extra: json!({"uncensored": true, "private": true}), ..Model::default() }
}

#[test]
fn an_uncensored_models_request_that_matches_is_refused_before_anything_runs() {
    let s = Server::start(&[uncensored()], &[ids("Fine.")], json!({}));
    let (c, r) = s.post("/v1/chat/completions", &json!({"messages": [user("kids. explicit.")], "max_tokens": 20}));
    assert_eq!(c, 400, "{r}");
    assert!(r["error"]["message"].as_str().unwrap().contains("minor-safety"), "{r}");
    assert!(s.gen_lines().is_empty(), "nothing was generated");
    // the Anthropic API goes through the same start
    let (c, r) = s.post("/v1/messages", &json!({"model": "x", "max_tokens": 20, "messages": [user("kids. explicit.")]}));
    assert_eq!(c, 400, "{r}");
    // an ordinary request is answered
    let (c, r) = s.post("/v1/chat/completions", &json!({"messages": [user("What is 2+2?")], "max_tokens": 20}));
    assert_eq!(c, 200, "{r}");
    // and the model list says the model is checked
    let (_, l) = s.get("/api/models", true);
    assert_eq!((&l["models"][0]["private"], &l["models"][0]["uncensored"]), (&json!(true), &json!(true)), "{l}");
}

#[test]
fn a_reply_that_matches_is_stopped() {
    let s = Server::start(&[uncensored()], &[ids("</think>\n\nThe kids. Then explicit things follow and more.")], json!({}));
    let (c, r) = s.post("/v1/chat/completions", &json!({"messages": [user("Tell a story.")], "max_tokens": 50}));
    assert_ne!(c, 200, "{r}");
    let text = r.to_string();
    assert!(text.contains("minor-safety"), "{text}");
    assert!(!text.contains("more."), "the reply stops where it matched: {text}");
}

#[test]
fn a_model_without_the_marks_is_not_checked() {
    let s = Server::start(&[Model::default()], &[ids("</think>\n\nThe kids. Then explicit things follow.")], json!({}));
    let (c, r) = s.post("/v1/chat/completions", &json!({"messages": [user("kids. explicit.")], "max_tokens": 50}));
    assert_eq!(c, 200, "{r}");
    let (_, l) = s.get("/api/models", true);
    assert_eq!(l["models"][0]["uncensored"], false);
}
