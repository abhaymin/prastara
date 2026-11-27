//! Private datasets (strata-workshop's private_data), on the CPU with placeholder data only: tiny generated PNGs, plain
//! sentences, and a mock of the server's chat API on localhost. The gates, the refusals, the 0700 folders, and the
//! private mark carried into what is trained, merged and cleaned from such a dataset.

mod common;

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use base64::Engine as _;
use serde_json::{json, Value};
use strata_workshop::private_data::{self as pd, gate, Local, Settings};

/// A tiny PNG of one flat colour (each test image differs by its colour, so the mock can tell them apart).
fn png(rgb: [u8; 3]) -> Vec<u8> {
    let img = image::RgbImage::from_pixel(4, 4, image::Rgb(rgb));
    let mut out = std::io::Cursor::new(Vec::new());
    img.write_to(&mut out, image::ImageFormat::Png).unwrap();
    out.into_inner()
}

/// A mock of the server on localhost: `/api/models` lists `models`; `/v1/chat/completions` answers `reply(body)`.
/// Every chat body is kept in `seen`.
fn mock(models: Value, reply: impl Fn(&Value) -> String + Send + 'static) -> (String, Arc<Mutex<Vec<Value>>>) {
    use std::io::{BufRead, BufReader, Read, Write};
    let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}", l.local_addr().unwrap());
    let seen = Arc::new(Mutex::new(vec![]));
    let seen2 = seen.clone();
    std::thread::spawn(move || {
        for c in l.incoming() {
            let Ok(mut c) = c else { continue };
            let mut r = BufReader::new(c.try_clone().unwrap());
            let mut first = String::new();
            r.read_line(&mut first).ok();
            let mut len = 0;
            loop {
                let mut line = String::new();
                if r.read_line(&mut line).unwrap_or(0) == 0 || line == "\r\n" { break; }
                if let Some(v) = line.to_lowercase().strip_prefix("content-length:") { len = v.trim().parse().unwrap_or(0); }
            }
            let mut body = vec![0; len];
            let _ = r.read_exact(&mut body);
            let out = if first.starts_with("GET /api/models") {
                models.to_string()
            } else {
                let b: Value = serde_json::from_slice(&body).unwrap_or_default();
                let text = reply(&b);
                seen2.lock().unwrap().push(b);
                json!({"choices": [{"message": {"content": text}}]}).to_string()
            };
            let _ = write!(c, "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{out}", out.len());
        }
    });
    (url, seen)
}

fn models() -> Value {
    json!({"models": [{"id": "mula-abliterated", "marks": ["private", "abliterated"], "multimodal": true, "current": true},
                      {"id": "tiel", "marks": [], "multimodal": true}]})
}

const ATTEST: &str = "owner_attests_adults_only = true\nowner_attests_consent = true\norigin = \"test placeholders\"\n";

/// A source folder with these images (name, colour) and its SOURCE.toml.
fn source(root: &Path, images: &[(&str, [u8; 3])]) -> PathBuf {
    let d = root.join("src");
    std::fs::create_dir_all(&d).unwrap();
    std::fs::write(d.join(pd::SOURCE_FILE), ATTEST).unwrap();
    for (n, c) in images {
        std::fs::write(d.join(n), png(*c)).unwrap();
    }
    d
}

/// The image a describing request carries (its base64), as one of `images`' names.
fn which(body: &Value, images: &[(&str, [u8; 3])]) -> Option<String> {
    let url = body["messages"][1]["content"][0]["image_url"]["url"].as_str()?;
    images.iter().find(|(_, c)| url.ends_with(&base64::engine::general_purpose::STANDARD.encode(png(*c)))).map(|(n, _)| n.to_string())
}

fn describe(description: &str, tags: &[&str], age: &str, real: bool) -> String {
    json!({"description": description, "tags": tags, "age_assessment": age, "real_person_named": real}).to_string()
}

fn lines(p: &Path) -> Vec<Value> {
    std::fs::read_to_string(p).unwrap_or_default().lines().map(|l| serde_json::from_str(l).unwrap()).collect()
}

fn local(url: &str) -> Local {
    let mut m = Local::new(url, None).unwrap();
    m.choose(Some("mula-abliterated"), true).unwrap();
    m
}

#[test]
fn a_folder_without_source_toml_is_refused() {
    let t = common::temp_dir("pd-nosource");
    let d = t.join("src");
    std::fs::create_dir_all(&d).unwrap();
    std::fs::write(d.join("a.png"), png([1, 2, 3])).unwrap();
    let (url, seen) = mock(models(), |_| "{}".into());
    let s = Settings::new("set", vec![d.clone()], t.join("private"));
    let e = pd::run(&s, &local(&url), |_, _, _| {}, || false).unwrap_err().to_string();
    assert!(e.contains(&d.join("SOURCE.toml").display().to_string()) && e.contains("owner_attests_adults_only"), "{e}");
    assert!(seen.lock().unwrap().is_empty(), "nothing was sent to the model");
    assert!(!t.join("private").exists(), "nothing was written");
    // the command refuses it too, before it looks for the server
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_am2strata-workshop"))
        .args(["--json", "private-data", "--name", "set", "--source"]).arg(&d).arg("--data-dir").arg(t.join("data"))
        .env("AM2STRATA_URL", "http://127.0.0.1:1").output().unwrap();
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stdout).contains("SOURCE.toml"));
}

#[test]
fn a_remote_model_name_is_refused() {
    // refused before the server is asked anything (nothing answers on port 1)
    let mut m = Local::new("http://127.0.0.1:1", None).unwrap();
    let e = m.choose(Some("openai/gpt-5.5"), false).unwrap_err().to_string();
    assert!(e.contains("remote"), "{e}");
    // an unmarked local model, or a server that is not this machine
    let (url, _) = mock(models(), |_| "{}".into());
    let mut m = Local::new(&url, None).unwrap();
    assert!(m.choose(Some("tiel"), true).unwrap_err().to_string().contains("not marked"));
    assert!(Local::new("https://api.example.com/v1", None).is_err());
    // and the command
    let t = common::temp_dir("pd-remote");
    let d = source(&t, &[]);
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_am2strata-workshop"))
        .args(["--json", "private-data", "--name", "set", "--model", "someprovider/model", "--source"]).arg(&d)
        .arg("--data-dir").arg(t.join("data")).env("AM2STRATA_URL", "http://127.0.0.1:1").output().unwrap();
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stdout).contains("remote model"));
}

#[test]
fn the_gates_drop_and_block_and_the_rest_is_written_private() {
    let t = common::temp_dir("pd-gates");
    let imgs: [(&str, [u8; 3]); 4] = [("chair.png", [200, 10, 10]), ("court.png", [10, 200, 10]), ("unclear.png", [10, 10, 200]),
                                       ("statue.png", [90, 90, 90])];
    let d = source(&t, &imgs);
    let imgs2 = imgs;
    let (url, seen) = mock(models(), move |b| match which(b, &imgs2).as_deref() {
        Some("chair.png") => describe("a red chair by a window", &["chair", "red", "window"], "adult", false),
        // a plainly non-sexual pair of trigger words (a minor term and a sexual term together), as minor_safety's tests use
        Some("court.png") => describe("a stack of juvenile court files", &["explicit", "paper"], "adult", false),
        Some("unclear.png") => describe("a blurred figure far away", &["blur"], "uncertain", false),
        Some("statue.png") => describe("a bronze statue of a named public figure", &["statue"], "adult", true),
        Some(other) => panic!("{other}"),
        // the prompt request (text only)
        None => json!({"prompt": "a red wooden chair beside a bright window, soft morning light", "real_person_named": false}).to_string(),
    });
    let root = t.join("private");
    let s = Settings::new("set", vec![d.clone()], root.clone());
    let r = pd::run(&s, &local(&url), |_, _, _| {}, || false).unwrap();
    assert_eq!(r.images, 4);
    assert_eq!(r.dropped.get(gate::MINOR_SAFETY), Some(&1), "{r:?}");
    assert_eq!(r.dropped.get(gate::AGE_UNCERTAIN), Some(&1), "{r:?}");
    assert_eq!(r.dropped.get(gate::REAL_PERSON), Some(&1), "{r:?}");
    assert_eq!((r.vision_rows, r.prompt_rows), (2, 1), "chair and statue described; only the chair gets a prompt");
    // the blocked hashes: the two that failed a gate, in the blocklist
    let court = pd::sha256_hex(&png([10, 200, 10]));
    let unclear = pd::sha256_hex(&png([10, 10, 200]));
    let bl = std::fs::read_to_string(root.join(pd::BLOCKLIST)).unwrap();
    assert!(bl.contains(&format!("{court} minor_safety")) && bl.contains(&format!("{unclear} age_uncertain")), "{bl}");
    assert_eq!(bl.lines().count(), 2);
    // the rows: private, and none from the dropped images
    let out = root.join("set");
    let vision = lines(&out.join("vision.jsonl"));
    assert!(vision.iter().all(|v| v["private"] == true && v["description"].is_string() && v["tags"].is_array()));
    assert!(vision.iter().all(|v| !v["image"].as_str().unwrap().contains("court") && !v["image"].as_str().unwrap().contains("unclear")));
    let prompts = lines(&out.join("prompts.jsonl"));
    assert_eq!(prompts.len(), 1);
    assert!(prompts[0]["private"] == true && prompts[0]["image"].as_str().unwrap().ends_with("chair.png"));
    // the card: private, counts and hashes, never the dropped content
    let card = std::fs::read_to_string(out.join(pd::CARD)).unwrap();
    assert!(card.starts_with(pd::CARD_MARK) && card.contains("**Private.**") && card.contains(&court) && card.contains("test placeholders"));
    assert!(!card.contains("juvenile") && !card.contains("blurred") && !card.contains("bronze"), "no dropped content on the card");
    // every request asked for the age and stayed local
    let bodies = seen.lock().unwrap().clone();
    assert!(bodies.iter().all(|b| b["strata_local_only"] == true && b["model"] == "mula-abliterated"));
    assert!(bodies.iter().any(|b| b["messages"][1]["content"][1]["text"].as_str().is_some_and(|t| t.contains("age_assessment"))));
    let asked = bodies.len();

    // a later run skips the blocked files without asking the model about them
    let s2 = Settings::new("set2", vec![d], root.clone());
    let r2 = pd::run(&s2, &local(&url), |_, _, _| {}, || false).unwrap();
    assert_eq!(r2.dropped.get(gate::BLOCKED), Some(&2), "{r2:?}");
    let again: Vec<Value> = seen.lock().unwrap()[asked..].to_vec();
    assert!(again.iter().all(|b| which(b, &imgs).is_none_or(|n| n != "court.png" && n != "unclear.png")));
}

#[test]
fn an_uncertain_age_drops_the_item() {
    let t = common::temp_dir("pd-age");
    let imgs: [(&str, [u8; 3]); 1] = [("a.png", [5, 6, 7])];
    let d = source(&t, &imgs);
    let (url, seen) = mock(models(), |_| describe("a person standing in a field", &["field"], "uncertain", false));
    let s = Settings::new("set", vec![d], t.join("private"));
    let r = pd::run(&s, &local(&url), |_, _, _| {}, || false).unwrap();
    assert_eq!((r.vision_rows, r.prompt_rows), (0, 0));
    assert_eq!(r.dropped.get(gate::AGE_UNCERTAIN), Some(&1));
    assert_eq!(seen.lock().unwrap().len(), 1, "no generation prompt is asked for a dropped item");
    assert!(std::fs::read_to_string(t.join("private").join(pd::BLOCKLIST)).unwrap().contains(&pd::sha256_hex(&png([5, 6, 7]))));
    assert!(!t.join("private/set/vision.jsonl").exists());
}

#[test]
fn text_rows_go_through_the_rules_too() {
    let t = common::temp_dir("pd-text");
    let d = source(&t, &[]);
    let sentence = "The garden has a small pond with three goldfish and a wooden bench under an old apple tree. ";
    std::fs::write(d.join("notes.txt"), sentence.repeat(10)).unwrap();
    std::fs::write(d.join("other.txt"), sentence.repeat(10).replace("goldfish", "carp")).unwrap();
    let (url, _) = mock(models(), |b| {
        let ask = b["messages"][1]["content"].as_str().unwrap_or("");
        let answer = if ask.contains("carp") { "The juvenile court sealed the explicit records long ago." }
                     else { "The pond in the garden holds three goldfish, near a wooden bench." };
        json!([[{"role": "user", "content": "What is in the garden?"}, {"role": "assistant", "content": answer}]]).to_string()
    });
    let mut s = Settings::new("set", vec![d], t.join("private"));
    s.text_count = 4;
    s.chunk_chars = 500;
    let r = pd::run(&s, &local(&url), |_, _, _| {}, || false).unwrap();
    assert_eq!(r.dropped.get(gate::MINOR_SAFETY), Some(&1), "{r:?}");
    let rows = lines(&t.join("private/set/text.jsonl"));
    assert!(!rows.is_empty() && rows.iter().all(|v| v["private"] == true && v["messages"].is_array()));
    assert!(rows.iter().all(|v| !v.to_string().contains("juvenile")));
}

#[cfg(unix)]
#[test]
fn the_private_folders_are_0700_and_the_files_0600() {
    use std::os::unix::fs::PermissionsExt as _;
    let t = common::temp_dir("pd-perm");
    let imgs: [(&str, [u8; 3]); 1] = [("a.png", [1, 1, 1])];
    let d = source(&t, &imgs);
    let (url, _) = mock(models(), |b| if b["messages"][1]["content"].is_array() {
        describe("a grey square", &["grey"], "adult", false)
    } else {
        json!({"prompt": "a flat grey square", "real_person_named": false}).to_string()
    });
    let root = t.join("private");
    // a root that already exists with looser permissions is tightened
    std::fs::create_dir_all(&root).unwrap();
    std::fs::set_permissions(&root, std::fs::Permissions::from_mode(0o755)).unwrap();
    pd::run(&Settings::new("set", vec![d], root.clone()), &local(&url), |_, _, _| {}, || false).unwrap();
    let mode = |p: &Path| std::fs::metadata(p).unwrap().permissions().mode() & 0o777;
    assert_eq!(mode(&root), 0o700);
    assert_eq!(mode(&root.join("set")), 0o700);
    for f in ["vision.jsonl", "prompts.jsonl", pd::CARD] {
        assert_eq!(mode(&root.join("set").join(f)), 0o600, "{f}");
    }
}

#[test]
fn training_on_a_private_dataset_makes_private_outputs() {
    let (root, model, data) = common::model_and_data("pd-train");
    // the same rows, marked private as private_data writes them
    let private = root.join("private.jsonl");
    common::write_jsonl(&private, &common::chat_rows(24).into_iter().map(|mut r| { r["private"] = json!(true); r }).collect::<Vec<_>>());
    assert!(pd::is_private_dataset(&private) && !pd::is_private_dataset(&data));
    let train = |data: &Path, out: &Path| {
        let o = std::process::Command::new(env!("CARGO_BIN_EXE_am2strata-workshop"))
            .args(["--json", "train", "--model"]).arg(&model).arg("--data").arg(data).arg("--out").arg(out)
            .args(["--epochs", "1", "--max-steps", "2", "--rank", "4", "--targets", "q_proj,v_proj", "--max-len", "64", "--threads", "2"])
            .output().unwrap();
        assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stdout));
        let cfg: Value = serde_json::from_str(&std::fs::read_to_string(out.join("adapter_config.json")).unwrap()).unwrap();
        cfg
    };
    let (a, b) = (root.join("adapter-private"), root.join("adapter-plain"));
    assert_eq!(train(&private, &a)["prastara_private"], true, "an adapter trained on private data is private");
    assert!(train(&data, &b).get("prastara_private").is_none(), "an ordinary one is not");
    // merged: the model's config.json is private, which convert carries into the GGUF (prastara.private)
    let merged = root.join("merged");
    strata_workshop::merge::merge(&model, &a, &merged).unwrap();
    let cfg: Value = serde_json::from_str(&std::fs::read_to_string(merged.join("config.json")).unwrap()).unwrap();
    assert_eq!(cfg["prastara_private"], true);
    let plain = root.join("merged-plain");
    strata_workshop::merge::merge(&model, &b, &plain).unwrap();
    let cfg: Value = serde_json::from_str(&std::fs::read_to_string(plain.join("config.json")).unwrap()).unwrap();
    assert!(cfg.get("prastara_private").is_none());
}

#[test]
fn cleaning_a_private_dataset_marks_every_row_and_the_card() {
    let t = common::temp_dir("pd-clean");
    let private = t.join("private.jsonl");
    common::write_jsonl(&private, &common::chat_rows(8).into_iter().map(|mut r| { r["private"] = json!(true); r }).collect::<Vec<_>>());
    let plain = t.join("plain.jsonl");
    common::write_jsonl(&plain, &common::chat_rows(16)[8..]);
    let mut s = strata_workshop::clean::Settings::new(vec![private, plain.clone()], t.join("out.jsonl"));
    s.min_answer_tokens = 1;
    s.dedupe = strata_workshop::clean::Dedupe::Off;
    let r = strata_workshop::clean::run(&s, |_| {}).unwrap();
    assert!(r.private && r.kept > 0);
    assert!(lines(&t.join("out.jsonl")).iter().all(|v| v["private"] == true), "the plain input's rows are private too once mixed in");
    assert!(std::fs::read_to_string(s.card_path()).unwrap().starts_with(pd::CARD_MARK));
    assert!(pd::is_private_dataset(&t.join("out.jsonl")));
    // a clean of only ordinary data stays ordinary
    let s2 = strata_workshop::clean::Settings::new(vec![plain], t.join("out2.jsonl"));
    let r2 = strata_workshop::clean::run(&s2, |_| {}).unwrap();
    assert!(!r2.private && !pd::is_private_dataset(&t.join("out2.jsonl")));
}

#[test]
fn a_video_without_a_frame_decoder_is_counted_not_read() {
    let t = common::temp_dir("pd-video");
    let d = source(&t, &[]);
    std::fs::write(d.join("clip.mp4"), b"not really a video").unwrap();
    let (url, seen) = mock(models(), |_| "{}".into());
    let r = pd::run(&Settings::new("set", vec![d], t.join("private")), &local(&url), |_, _, _| {}, || false).unwrap();
    assert_eq!((r.videos, r.videos_unread), (1, 1));
    assert!(seen.lock().unwrap().is_empty());
    assert!(std::fs::read_to_string(t.join("private/set").join(pd::CARD)).unwrap().contains("videos were skipped"));
}

/// The media engine's line protocol for frames, with a stand-in engine (a shell script that prints a ready line and
/// one frames event): every frame is described, and one frame that fails a gate drops the whole video.
#[cfg(unix)]
#[test]
fn video_frames_come_through_the_media_engine() {
    use std::os::unix::fs::PermissionsExt as _;
    let t = common::temp_dir("pd-frames");
    let d = source(&t, &[]);
    std::fs::write(d.join("clip.mp4"), b"a stand-in video").unwrap();
    std::fs::write(d.join("other.mp4"), b"another stand-in video").unwrap();
    let b64 = |c: [u8; 3]| base64::engine::general_purpose::STANDARD.encode(png(c));
    let exe = t.join("fake-media");
    let frames = json!({"event": "frames", "fps": 1.0, "duration_s": 2.0, "frames": [{"t": 0.0, "png": b64([1, 2, 3])}, {"t": 1.0, "png": b64([4, 5, 6])}]});
    let other = json!({"event": "frames", "fps": 1.0, "duration_s": 1.0, "frames": [{"t": 0.0, "png": b64([7, 8, 9])}]});
    std::fs::write(&exe, format!("#!/bin/sh\necho '{{\"event\":\"ready\"}}'\nread line\ncase \"$line\" in *other.mp4*) echo '{other}';; *) echo '{frames}';; esac\nread line\n")).unwrap();
    std::fs::set_permissions(&exe, std::fs::Permissions::from_mode(0o755)).unwrap();
    let bad = b64([7, 8, 9]);
    let (url, _) = mock(models(), move |b| {
        let u = b["messages"][1]["content"][0]["image_url"]["url"].as_str();
        match u {
            Some(u) if u.ends_with(&bad) => describe("a figure", &["figure"], "minor", false),
            Some(_) => describe("an empty room", &["room"], "adult", false),
            None => json!({"prompt": "an empty room, wide shot", "real_person_named": false}).to_string(),
        }
    });
    let mut s = Settings::new("set", vec![d], t.join("private"));
    s.media_exe = Some(exe);
    let r = pd::run(&s, &local(&url), |_, _, _| {}, || false).unwrap();
    assert_eq!((r.videos, r.frames, r.vision_rows, r.prompt_rows), (2, 3, 2, 2), "{r:?}");
    assert_eq!(r.dropped.get(gate::AGE_MINOR), Some(&1));
    let bl = std::fs::read_to_string(t.join("private").join(pd::BLOCKLIST)).unwrap();
    assert!(bl.contains(&pd::sha256_hex(b"another stand-in video")) && bl.contains(&pd::sha256_hex(&png([7, 8, 9]))), "the video and its frame");
    let vision = lines(&t.join("private/set/vision.jsonl"));
    assert!(vision.iter().all(|v| v["meta"]["video"].as_str().unwrap().ends_with("clip.mp4")));
    assert!(vision.iter().all(|v| Path::new(v["image"].as_str().unwrap()).starts_with(t.join("private/set/frames"))));
}
