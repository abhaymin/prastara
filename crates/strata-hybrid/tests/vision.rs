//! Inkling 2's pictures on the Rust engine (step I6v), against llama.cpp's mtmd; ignored by default (they need the
//! model files and the oracle encoder):
//!   INKLING_MMPROJ=mmproj.gguf VISION_ORACLE=am2strata-llama-vision INKLING_MODEL=model.gguf \
//!   VISION_IMAGES=a.png:b.png cargo test --release --no-default-features --manifest-path crates/strata-hybrid/Cargo.toml \
//!       --test vision -- --ignored --nocapture
//! - rows: am2strata-vision's rows against the oracle's (the same ENC protocol), cosine > 0.999 per row;
//! - answer: GENI with those rows and a question, greedy, on the CPU (INKLING_TOKENIZER: the model's tokenizer folder).

use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

fn env(k: &str) -> String {
    std::env::var(k).unwrap_or_else(|_| panic!("set {k}"))
}

/// Run an encoder over the pictures (ENC lines), in `dir`; the rows of each, and the token counts.
fn encode(exe: &str, extra: &[&str], images: &[PathBuf], dir: &Path, tag: &str) -> Vec<(Vec<f32>, usize)> {
    let mut c = Command::new(exe);
    c.arg("--mmproj").arg(env("INKLING_MMPROJ")).args(extra).arg("--threads").arg("8").current_dir(dir)
        .stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::inherit());
    if let Ok(m) = std::env::var("INKLING_MODEL") {
        c.arg("--model").arg(m);
    }
    let mut p = c.spawn().expect("start the encoder");
    let mut out = BufReader::new(p.stdout.take().unwrap()).lines();
    assert!(out.next().unwrap().unwrap().starts_with("READY"));
    let mut stdin = p.stdin.take().unwrap();
    let mut res = Vec::new();
    for (i, img) in images.iter().enumerate() {
        let dst = dir.join(format!("{tag}-{i}.f32"));
        writeln!(stdin, "ENC {} {}", img.display(), dst.display()).unwrap();
        let reply = out.next().unwrap().unwrap();
        let n: usize = reply.strip_prefix("OK ").unwrap_or_else(|| panic!("{exe}: {reply}")).trim().parse().unwrap();
        let rows = std::fs::read(&dst).unwrap().chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        res.push((rows, n));
    }
    writeln!(stdin, "QUIT").unwrap();
    let _ = p.wait();
    res
}

fn images() -> Vec<PathBuf> {
    // the synthetic pictures in tests/fixtures/vision (synthetic, made for this test) unless VISION_IMAGES names others
    let given = std::env::var("VISION_IMAGES").unwrap_or_default();
    if given.is_empty() {
        let d = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/vision");
        return ["pagoda_320x192.png", "grad_333x217.png", "pagoda_640x640.png"].iter().map(|n| d.join(n)).collect();
    }
    given.split(':').filter(|s| !s.is_empty()).map(|s| std::path::absolute(s).unwrap()).collect()
}

#[test]
#[ignore]
fn rows_match_mtmd() {
    let dir = std::env::temp_dir().join(format!("am2-vision-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let imgs = images();
    let ours = encode(env!("CARGO_BIN_EXE_am2strata-vision"), &[], &imgs, &dir, "ours");
    let refs = encode(&env("VISION_ORACLE"), &[], &imgs, &dir, "ref");
    for ((img, (a, na)), (b, nb)) in imgs.iter().zip(&ours).zip(&refs) {
        assert_eq!(na, nb, "{}: token counts", img.display());
        assert_eq!(a.len(), b.len());
        let w = a.len() / na;
        let (mut worst, mut maxabs) = (1f64, 0f32);
        for (x, y) in a.chunks(w).zip(b.chunks(w)) {
            let dot: f64 = x.iter().zip(y).map(|(p, q)| *p as f64 * *q as f64).sum();
            let nx: f64 = x.iter().map(|p| (*p as f64).powi(2)).sum::<f64>().sqrt();
            let ny: f64 = y.iter().map(|q| (*q as f64).powi(2)).sum::<f64>().sqrt();
            worst = worst.min(dot / (nx * ny));
            maxabs = x.iter().zip(y).map(|(p, q)| (p - q).abs()).fold(maxabs, f32::max);
        }
        eprintln!("{}: {na} rows of {w}, worst cosine {worst:.6}, max abs diff {maxabs:.3e}", img.display());
        assert!(worst > 0.999, "{}", img.display());
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
#[ignore]
fn geni_answers_a_picture_question() {
    let dir = std::env::temp_dir().join(format!("am2-geni-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let img = images().remove(0);
    let (rows, n) = encode(env!("CARGO_BIN_EXE_am2strata-vision"), &[], std::slice::from_ref(&img), &dir, "geni").remove(0);
    let file = dir.join("geni-0.f32");
    let tok = strata_tokenizer::Tokenizer::from_dir(env("INKLING_TOKENIZER")).unwrap();
    let pad = tok.token_id("<|unused_200054|>").unwrap();
    let question = std::env::var("VISION_QUESTION").unwrap_or_else(|_| "What is in this picture, and what time of day is it? Answer in one sentence.".into());
    let text = format!("<|message_system|><|content_text|>Thinking effort level: 0<|end_message|><|message_user|><|content_image|><|unused_200054|><|end_message|><|message_user|><|content_text|>{question}<|end_message|>");
    let mut ids = Vec::new();
    for t in tok.encode(&text, true).unwrap() {
        if t == pad {
            ids.extend(std::iter::repeat(pad).take(n));
        } else {
            ids.push(t);
        }
    }
    assert_eq!(rows.len() % n, 0);
    let mut p = Command::new(env!("CARGO_BIN_EXE_am2strata-hybrid")).arg("--serve").arg("--model").arg(env("INKLING_MODEL"))
        .arg("--max-context").arg("4096").stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::inherit()).spawn().unwrap();
    let mut out = BufReader::new(p.stdout.take().unwrap()).lines();
    let mut stdin = p.stdin.take().unwrap();
    let max_new: usize = std::env::var("VISION_MAX_NEW").ok().and_then(|v| v.parse().ok()).unwrap_or(24);
    let ids_s: Vec<String> = ids.iter().map(|i| i.to_string()).collect();
    writeln!(stdin, "GENI {max_new} temperature=0 image_pad={pad} {} {}", file.display(), ids_s.join(",")).unwrap();
    let mut got = Vec::new();
    for l in out.by_ref() {
        let l = l.unwrap();
        if let Some(t) = l.strip_prefix("T ") {
            got.push(t.trim().parse::<u32>().unwrap());
        } else if l.starts_with("DONE") || l.starts_with("ERR") {
            eprintln!("{l}");
            assert!(l.starts_with("DONE"), "{l}");
            break;
        }
    }
    // a malformed request: one row too few
    std::fs::write(dir.join("short.f32"), &std::fs::read(&file).unwrap()[4 * rows.len() / n..]).unwrap();
    writeln!(stdin, "GENI 4 image_pad={pad} {} {}", dir.join("short.f32").display(), ids_s.join(",")).unwrap();
    let l = out.next().unwrap().unwrap();
    assert!(l.starts_with("ERR GENI:"), "{l}");
    writeln!(stdin, "QUIT").unwrap();
    let _ = p.wait();
    eprintln!("prompt: {} ids ({n} picture rows)\nanswer ids: {got:?}\nanswer: {:?}", ids.len(), tok.decode(&got).unwrap());
    assert!(!got.is_empty());
    let _ = std::fs::remove_dir_all(&dir);
}

/// A synthetic clip, frame by frame (Part F2): a square that moves left to right and turns from red to blue, `n`
/// frames of 320x192 (40 picture rows each), as PNG files in `dir`.
fn square_clip(dir: &Path, n: usize) -> Vec<PathBuf> {
    (0..n)
        .map(|k| {
            let mut img = image::RgbImage::from_pixel(320, 192, image::Rgb([235, 235, 235]));
            let f = k as f32 / (n - 1).max(1) as f32;
            let (x0, y0) = ((20.0 + f * 220.0) as u32, 56u32);
            let colour = image::Rgb([(220.0 * (1.0 - f)) as u8 + 20, 30, (220.0 * f) as u8 + 20]);
            for y in y0..y0 + 80 {
                for x in x0..x0 + 80 {
                    img.put_pixel(x, y, colour);
                }
            }
            let p = dir.join(format!("frame-{k}.png"));
            img.save(&p).unwrap();
            p
        })
        .collect()
}

/// Part F2 on the real engine: GENI with rows=stream, the frames encoded one by one while the engine reads the prompt
/// (each after its "[00:0k]" timestamp), so the attention cache grows frame by frame; then the answer.
///   INKLING_MMPROJ=... INKLING_MODEL=... INKLING_TOKENIZER=... [VIDEO_FRAMES=4] cargo test --release \
///       --no-default-features --manifest-path crates/strata-hybrid/Cargo.toml --test vision geni_streams_a_video -- --ignored --nocapture
#[test]
#[ignore]
fn geni_streams_a_video() {
    let dir = std::env::temp_dir().join(format!("am2-video-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let n: usize = std::env::var("VIDEO_FRAMES").ok().and_then(|v| v.parse().ok()).unwrap_or(4);
    let frames = square_clip(&dir, n);
    let tok = strata_tokenizer::Tokenizer::from_dir(env("INKLING_TOKENIZER")).unwrap();
    let pad = tok.token_id("<|unused_200054|>").unwrap();
    let rows_per = 40; // 320x192: 8 x 5 merged patches
    let question = std::env::var("VISION_QUESTION")
        .unwrap_or_else(|_| "Describe what the square does over the video: where it moves and how its colour changes. Answer in one sentence.".into());
    let mut text = format!("<|message_system|><|content_text|>Thinking effort level: 0<|end_message|><|message_user|><|content_text|>[A video of {n}.0 s, as {n} frames in order, one a second, each after its time:]\n");
    for k in 0..n {
        text.push_str(&format!("{}[00:{k:02}]<|end_message|><|message_user|><|content_image|><|unused_200054|><|end_message|><|message_user|><|content_text|>", if k == 0 { "" } else { "\n" }));
    }
    text.push_str(&format!("\n{question}<|end_message|>"));
    let mut ids = Vec::new();
    for t in tok.encode(&text, true).unwrap() {
        if t == pad { ids.extend(std::iter::repeat_n(pad, rows_per)); } else { ids.push(t); }
    }
    let file = dir.join("video.f32");
    std::fs::write(&file, b"").unwrap();
    let mut p = Command::new(env!("CARGO_BIN_EXE_am2strata-hybrid")).arg("--serve").arg("--model").arg(env("INKLING_MODEL"))
        .arg("--max-context").arg("4096").stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::inherit()).spawn().unwrap();
    let mut out = BufReader::new(p.stdout.take().unwrap()).lines();
    let mut stdin = p.stdin.take().unwrap();
    let mut info = String::new();
    for l in out.by_ref() {
        let l = l.unwrap();
        if l.starts_with("INFO") { info = l.clone(); }
        if l.starts_with("READY") { break; }
    }
    assert!(info.contains("geni_stream=1"), "{info}");
    let max_new: usize = std::env::var("VISION_MAX_NEW").ok().and_then(|v| v.parse().ok()).unwrap_or(40);
    let ids_s: Vec<String> = ids.iter().map(|i| i.to_string()).collect();
    let t0 = std::time::Instant::now();
    writeln!(stdin, "GENI {max_new} temperature=0 image_pad={pad} rows=stream {} {}", file.display(), ids_s.join(",")).unwrap();
    // the engine's lines on their own thread, with their times
    let (tx, rx) = std::sync::mpsc::channel::<(f64, String)>();
    std::thread::spawn(move || {
        for l in out {
            let Ok(l) = l else { break };
            let end = l.starts_with("DONE") || l.starts_with("ERR");
            let _ = tx.send((t0.elapsed().as_secs_f64(), l));
            if end { break; }
        }
    });
    // the frames, encoded one at a time and appended: the engine waits at each picture for its rows
    let mut enc = Command::new(env!("CARGO_BIN_EXE_am2strata-vision")).arg("--mmproj").arg(env("INKLING_MMPROJ")).arg("--threads").arg("8")
        .current_dir(&dir).stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::inherit()).spawn().unwrap();
    let mut enc_out = BufReader::new(enc.stdout.take().unwrap()).lines();
    let mut enc_in = enc.stdin.take().unwrap();
    assert!(enc_out.next().unwrap().unwrap().starts_with("READY"));
    for (k, f) in frames.iter().enumerate() {
        let dst = dir.join(format!("v{k}.f32"));
        writeln!(enc_in, "ENC {} {}", f.display(), dst.display()).unwrap();
        let reply = enc_out.next().unwrap().unwrap();
        assert_eq!(reply.trim(), format!("OK {rows_per}"));
        let bytes = std::fs::read(&dst).unwrap();
        std::fs::OpenOptions::new().append(true).open(&file).unwrap().write_all(&bytes).unwrap();
        eprintln!("frame {k} appended at {:.1} s", t0.elapsed().as_secs_f64());
    }
    writeln!(enc_in, "QUIT").unwrap();
    let _ = enc.wait();
    let mut got = Vec::new();
    let mut last_pp = String::new();
    for (t, l) in rx {
        if let Some(tk) = l.strip_prefix("T ") {
            got.push(tk.trim().parse::<u32>().unwrap());
        } else if l.starts_with("PP ") {
            if l.split_whitespace().nth(1) != last_pp.split_whitespace().nth(1) { eprintln!("{t:.1} s: {l}"); }
            last_pp = l;
        } else if l.starts_with("DONE") || l.starts_with("ERR") {
            eprintln!("{t:.1} s: {l}");
            assert!(l.starts_with("DONE"), "{l}");
            break;
        }
    }
    writeln!(stdin, "QUIT").unwrap();
    let _ = p.wait();
    eprintln!("prompt: {} ids ({n} frames x {rows_per} rows)\nanswer: {:?}", ids.len(), tok.decode(&got).unwrap());
    assert!(!got.is_empty());
    let _ = std::fs::remove_dir_all(&dir);
}
