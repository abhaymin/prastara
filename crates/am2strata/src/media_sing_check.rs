//! Score comparison of a sung take (`media sing-check`): per syllable the pitch error in cents, the vowel's onset against the
//! beat and how voiced it was, plus the faults that break the word-to-word link (SINGING-PLAN). This is not the check itself:
//! the endpoint only forwards the audio and score to the media engine's `sing_check` op (a CPU job, no model, exactly like
//! `/v1/media/analyze` forwards `analyze`) and returns what the engine reports (`events`, `faults`, `summary`). It reads an
//! audio file in any format the engine decodes (MP3, WAV, FLAC, AAC, OGG) from a multipart `file` or a base64 / data URL in
//! JSON `"audio"`, and takes the song's score (inline `"score": {...}` or a `"score_path"`), the tonic (Sa) hertz (`tonic_hz`)
//! and where to start (`start_s`).

use crate::{api_authorized, error, AppState};
use axum::extract::{FromRequest, Multipart, Request, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use base64::Engine as _;
use serde_json::{json, Value};

/// The largest audio this endpoint forwards to the engine, in bytes (200 MiB).
const MAX_BYTES: usize = 200 * 1024 * 1024;

fn bad(msg: impl Into<String>) -> Response { error(StatusCode::BAD_REQUEST, "invalid_request_error", msg) }

/// POST /v1/media/sing-check: the score comparison of a sung take (a multipart `file`, or a base64 / data URL in JSON
/// `"audio"`) against its song's score (`score` inline, or `score_path`), with an optional tonic `tonic_hz` (Sa, default 130 Hz)
/// and start `start_s`; it forwards them to the media engine's `sing_check` op and returns the engine's event without its
/// `event` field. At most 200 MiB are forwarded.
pub async fn sing_check_endpoint(State(s): State<AppState>, headers: HeaderMap, req: Request) -> Response {
    if !api_authorized(&s, &headers) {
        return error(StatusCode::UNAUTHORIZED, "authentication_error", "a valid API key is required");
    }
    if !s.media.available() {
        return error(StatusCode::NOT_IMPLEMENTED, "not_available", "the media engine is not built");
    }
    let ctype = req.headers().get("content-type").and_then(|v| v.to_str().ok()).unwrap_or("").to_string();
    let (audio, score, score_path, tonic_hz, start_s) = if ctype.starts_with("multipart/form-data") {
        // multipart: the audio plus optional `score` (JSON text), `score_path`, `tonic_hz`, `start_s` as form fields
        let Ok(mut mp) = Multipart::from_request(req, &State(s.clone())).await else { return bad("multipart/form-data could not be read") };
        let mut bytes: Option<Vec<u8>> = None;
        let (mut score_val, mut sp, mut tonic, mut start) = (Value::Null, String::new(), 0.0f64, 0.0f64);
        while let Ok(Some(f)) = mp.next_field().await {
            match f.name() {
                Some("file") => match f.bytes().await {
                    Ok(b) => bytes = Some(b.to_vec()),
                    Err(e) => return bad(&format!("the file could not be read: {e}")),
                },
                Some("score") => if let Some(v) = f.text().await.ok() { score_val = serde_json::from_str(v.trim()).unwrap_or(Value::Null); },
                Some("score_path") => sp = f.text().await.unwrap_or_default().trim().to_string(),
                Some("tonic_hz") => tonic = f.text().await.unwrap_or_default().parse::<f64>().unwrap_or(0.0),
                Some("start_s") => start = f.text().await.unwrap_or_default().parse::<f64>().unwrap_or(0.0),
                _ => {}
            }
        }
        let bytes = match bytes { Some(b) => b, None => return bad("send the audio as the multipart field `file`") };
        (bytes, score_val, sp, tonic, start)
    } else {
        // JSON: {"audio": <base64|data URL>, "score"?:{...}, "score_path"?:<path>, "tonic_hz"?,"start_s"?} — same base64 transport /v1/media/analyze uses.
        let Json(body) = match Json::<Value>::from_request(req, &State(s.clone())).await {
            Ok(b) => b,
            Err(_) => return bad("send JSON with a base64 \"audio\" and a score (\"score\" or \"score_path\")"),
        };
        let src = body["audio"].as_str().filter(|s| !s.is_empty()).unwrap_or("");
        let raw = src.split_once(";base64,").map(|(_, b)| b).unwrap_or(src);
        let bytes = match base64::engine::general_purpose::STANDARD.decode(raw.trim()) {
            Ok(b) => b,
            Err(_) => return bad("{\"audio\": is required (base64 or a data: URL, or multipart `file`)"),
        };
        (bytes, body.get("score").cloned().unwrap_or(Value::Null), body["score_path"].as_str().unwrap_or("").to_string(),
            body["tonic_hz"].as_f64().unwrap_or(0.0), body["start_s"].as_f64().unwrap_or(0.0))
    };

    if audio.is_empty() { return bad("send the audio as \"file\""); }
    if audio.len() > MAX_BYTES { return bad("an audio of at most 200 MiB"); }
    let has_score = !score.is_null();
    if !has_score && score_path.is_empty() { return bad("send a \"score\" (JSON) or a \"score_path\""); }

    // the engine op: the base64 audio, the score inline or as a path (both parse to the same Score on the engine), and the
    // tonic + start the caller asked for (tonic defaults to 130 Hz where there is no drone to read).
    let mut cmd = json!({"op": "sing_check", "audio": base64::engine::general_purpose::STANDARD.encode(&audio)});
    if has_score { cmd["score"] = score; } else { cmd["score_path"] = json!(score_path); }
    if tonic_hz > 0.0 { cmd["tonic_hz"] = json!(tonic_hz); }
    if start_s > 0.0 { cmd["start_s"] = json!(start_s); }

    match s.media.call(cmd, |_| {}).await {
        Ok(mut v) => {
            // the engine's `sing_check` event, returned without its `event` field (events[], faults[], summary{...})
            if let Some(o) = v.as_object_mut() { o.remove("event"); }
            Json(v).into_response()
        }
        Err(e) => error(StatusCode::BAD_GATEWAY, "media_error", format!("{e:#}")),
    }
}

/// POST /v1/media/syllables: a spoken word split into its syllables (`docs/SINGING-PLAN.md`, step 2). JSON
/// `{"audio": <base64|data URL>, "text": <word>}` (split with the syllabifier) or `"texts": [syllables]`; the media engine's
/// `syllables` op answers with each syllable's start, vowel and end, the hum score and how far the split can be trusted.
pub async fn syllables_endpoint(State(s): State<AppState>, headers: HeaderMap, Json(body): Json<Value>) -> Response {
    if !api_authorized(&s, &headers) {
        return error(StatusCode::UNAUTHORIZED, "authentication_error", "a valid API key is required");
    }
    if !s.media.available() {
        return error(StatusCode::NOT_IMPLEMENTED, "not_available", "the media engine is not built");
    }
    let src = body["audio"].as_str().unwrap_or("");
    let raw = src.split_once(";base64,").map(|(_, b)| b).unwrap_or(src);
    let Ok(audio) = base64::engine::general_purpose::STANDARD.decode(raw.trim()) else { return bad("\"audio\": base64 or a data: URL") };
    if audio.is_empty() { return bad("send the word's audio as \"audio\""); }
    if audio.len() > 20 * 1024 * 1024 { return bad("one word's audio: at most 20 MiB"); }
    let mut cmd = json!({"op": "syllables", "audio": base64::engine::general_purpose::STANDARD.encode(&audio)});
    match (body["texts"].as_array(), body["text"].as_str().filter(|t| !t.trim().is_empty())) {
        (Some(t), _) if !t.is_empty() => cmd["texts"] = json!(t),
        (_, Some(w)) => cmd["text"] = json!(w.trim()),
        _ => return bad("send the word as \"text\", or its syllables as \"texts\""),
    }
    match s.media.call(cmd, |_| {}).await {
        Ok(mut v) => { if let Some(o) = v.as_object_mut() { o.remove("event"); } Json(v).into_response() }
        Err(e) => error(StatusCode::BAD_GATEWAY, "media_error", format!("{e:#}")),
    }
}
