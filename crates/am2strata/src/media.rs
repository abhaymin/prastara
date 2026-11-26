//! Pictures and video for the image encoder.
//!
//! A picture's source is a data: URL, an http(s) URL or a local file path (never a network share), as PolyStrata
//! reads them. A video is turned into frames before the model sees it: MiniCPM-V and the other models read a video
//! as an ordered series of pictures, each after its timestamp ("[00:42]"), one a second by default. Frames come from
//! the media engine's own MP4/H.264 decoder (Rust); ffmpeg, when installed, only reads what that cannot. On an engine
//! with `geni_stream=1` the server encodes the frames while the engine reads the prompt (run.rs). A client can always
//! send the frames as images instead.

use anyhow::{anyhow, bail, Context, Result};
use base64::Engine as _;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::process::Command;

const URL_MAX: usize = 64 << 20;
/// A chat video's frames: one a second by default (`video_fps`), at most 64 (`video_max_frames`, up to 256), each
/// scaled to 640 pixels wide (`video_width`); a video longer than the cap is sampled more thinly over its whole length.
pub const DEFAULT_FPS: f64 = 1.0;
pub const DEFAULT_MAX_FRAMES: usize = 64;
pub const MAX_FRAMES: usize = 256;
pub const DEFAULT_FRAME_WIDTH: usize = 640;

/// What a request's media sources may reach (main.rs sets it from the config at start). Local paths: only for a
/// server that listens on this PC alone, or with "allow_local_files": true. URLs: public addresses only, unless
/// "allow_private_urls": true (else an API client could read the LAN, the router or this PC's own services).
#[derive(Clone, Copy, Debug, Default)]
pub struct SourcePolicy {
    pub local_files: bool,
    pub private_urls: bool,
}

pub static POLICY: std::sync::OnceLock<SourcePolicy> = std::sync::OnceLock::new();

fn policy() -> SourcePolicy {
    POLICY.get().copied().unwrap_or_default()
}

/// Loopback, private, link-local, CGNAT, unique-local, multicast, unspecified or documentation addresses.
pub fn non_public(ip: std::net::IpAddr) -> bool {
    use std::net::IpAddr;
    match ip {
        IpAddr::V4(a) => {
            let o = a.octets();
            a.is_loopback() || a.is_private() || a.is_link_local() || a.is_unspecified() || a.is_broadcast() || a.is_multicast()
                || o[0] == 0 || (o[0] == 100 && (64..128).contains(&o[1])) || (o[0] == 192 && o[1] == 0 && o[2] == 0)
                || (o[0] == 198 && (18..20).contains(&o[1])) || o[0] >= 240
        }
        IpAddr::V6(a) => {
            if let Some(v4) = a.to_ipv4_mapped() {
                return non_public(IpAddr::V4(v4));
            }
            let s = a.segments();
            a.is_loopback() || a.is_unspecified() || a.is_multicast() || (s[0] & 0xfe00) == 0xfc00 || (s[0] & 0xffc0) == 0xfe80
        }
    }
}

/// GET of a public URL: each hop's address checked and pinned (no DNS rebinding between the check and the
/// connection), up to 5 redirects, each checked again.
pub async fn fetch_public(src: &str, allow_private: bool) -> Result<reqwest::Response> {
    let mut url = reqwest::Url::parse(src).map_err(|e| anyhow!("the URL could not be read: {e}"))?;
    for _ in 0..6 {
        let host = url.host_str().ok_or_else(|| anyhow!("the URL has no host"))?.to_string();
        let port = url.port_or_known_default().unwrap_or(443);
        let addrs: Vec<std::net::SocketAddr> = tokio::net::lookup_host((host.trim_start_matches('[').trim_end_matches(']'), port)).await
            .map_err(|e| anyhow!("the URL's host could not be found: {e}"))?.collect();
        let Some(addr) = addrs.first().copied() else { bail!("the URL's host could not be found") };
        if !allow_private && addrs.iter().any(|a| non_public(a.ip())) {
            bail!("the URL points into a private network or this PC ({host}); such URLs are not fetched (\"allow_private_urls\" in the run config)");
        }
        let resp = reqwest::Client::builder()
            .user_agent("strata")
            .timeout(std::time::Duration::from_secs(60))
            .redirect(reqwest::redirect::Policy::none())
            .resolve(&host, addr)
            .build()?
            .get(url.clone())
            .send()
            .await
            .map_err(|e| anyhow!("the URL could not be read: {e}"))?;
        if resp.status().is_redirection() {
            let to = resp.headers().get(reqwest::header::LOCATION).and_then(|v| v.to_str().ok())
                .ok_or_else(|| anyhow!("the URL redirects nowhere"))?;
            url = url.join(to).map_err(|e| anyhow!("the URL redirects to a bad address: {e}"))?;
            if !matches!(url.scheme(), "http" | "https") {
                bail!("the URL redirects to a {} address", url.scheme());
            }
            continue;
        }
        return Ok(resp);
    }
    bail!("the URL redirects too many times")
}

pub async fn load_source(src: &str) -> Result<Vec<u8>> {
    // a media library id ("mf_..."): a file the server made or was sent before. The id is the file's content hash, so
    // holding it is holding the file; listing the library stays per key (media_files.rs)
    if let Some(f) = crate::media_files::resolve(src, crate::media_files::MAIN) {
        return f.map_err(|e| anyhow!(e));
    }
    if let Some(rest) = src.strip_prefix("data:") {
        let (_, b64) = rest.split_once(',').ok_or_else(|| anyhow!("the data: URL could not be read (no comma)"))?;
        return base64::engine::general_purpose::STANDARD
            .decode(b64.trim())
            .map_err(|e| anyhow!("the data: URL could not be read ({e})"));
    }
    if src.starts_with("http://") || src.starts_with("https://") {
        let resp = fetch_public(src, policy().private_urls).await?;
        if resp.content_length().is_some_and(|n| n as usize > URL_MAX) {
            bail!("the URL's file is over {} MiB", URL_MAX >> 20);
        }
        let bytes = resp.bytes().await.map_err(|e| anyhow!("the URL could not be read: {e}"))?;
        if bytes.len() > URL_MAX {
            bail!("the URL's file is over {} MiB", URL_MAX >> 20);
        }
        return Ok(bytes.to_vec());
    }
    let path = src.strip_prefix("file://").unwrap_or(src);
    if path.starts_with("//") || path.starts_with("\\\\") {
        bail!("a file must be on this computer: network paths are not read");
    }
    if !path.is_empty() && !policy().local_files {
        bail!("local file paths are not read by a server that listens beyond this PC: send the file as a data: URL \
               (or set \"allow_local_files\": true in the run config)");
    }
    if !path.is_empty() && Path::new(path).is_file() {
        return std::fs::read(path).with_context(|| format!("the file could not be read from {path}"));
    }
    bail!("an image must be a data: URL, an http(s) URL or a local file path")
}

/// The media engine, for reading MP4/H.264 frames in Rust when ffmpeg is not installed (main.rs sets it at start).
pub static ENGINE: std::sync::OnceLock<std::sync::Arc<crate::image_gen::Media>> = std::sync::OnceLock::new();

/// Frames from the media engine's own MP4/H.264 decoder (`video_frames`), no ffmpeg: `max` frames spread evenly
/// (every_s 0) or one every `every_s` seconds, each scaled to `width`. -> (PNG bytes per frame, fps, duration in s).
/// Other formats come back as the engine's error, which names them and says they need ffmpeg.
pub async fn rust_frames(video: &[u8], every_s: f64, max: usize, width: usize) -> Result<(Vec<Vec<u8>>, f64, f64, Option<String>)> {
    let s = rust_frames_at(video, every_s, max, width, false).await?;
    Ok((s.frames.into_iter().map(|(_, png)| png).collect(), s.fps, s.duration_s, s.note))
}

/// Frames of a video with their times (seconds from its start), as PNG (Rust) or JPEG (ffmpeg) bytes.
pub struct Sampled {
    pub frames: Vec<(f64, Vec<u8>)>,
    pub fps: f64,
    pub duration_s: f64,
    pub note: Option<String>,
    pub mime: &'static str,
    /// only the keyframes could be read (a B-frame MP4 in the Rust decoder)
    pub keyframes_only: bool,
}

/// `rust_frames` with each frame's time; `fit`: one frame each `every_s`, spread wider so that `max` cover the
/// whole video when it is longer.
pub async fn rust_frames_at(video: &[u8], every_s: f64, max: usize, width: usize, fit: bool) -> Result<Sampled> {
    let Some(engine) = ENGINE.get() else { bail!("this server cannot decode video: no media engine and no ffmpeg") };
    let dir = tempdir()?;
    let input = dir.join("in.mp4");
    std::fs::write(&input, video)?;
    let r = engine.call(json!({"op": "video_frames", "file": input.to_string_lossy(), "every_s": every_s, "max": max, "width": width, "fit": fit}),
                        |_| {}).await;
    let _ = std::fs::remove_dir_all(&dir);
    let v = r?;
    let mut out = Vec::new();
    for f in v["frames"].as_array().into_iter().flatten() {
        let png = base64::engine::general_purpose::STANDARD.decode(f["png"].as_str().unwrap_or("")).map_err(|e| anyhow::anyhow!("a frame: {e}"))?;
        out.push((f["t"].as_f64().unwrap_or(0.0), png));
    }
    if out.is_empty() {
        bail!("the video gave no frames");
    }
    Ok(Sampled { frames: out, fps: v["fps"].as_f64().unwrap_or(0.0), duration_s: v["duration_s"].as_f64().unwrap_or(0.0),
                 note: v["note"].as_str().map(str::to_owned), mime: "image/png", keyframes_only: v["keyframes_only"] == true })
}

pub fn ffmpeg_available() -> bool {
    // tests on a machine that has ffmpeg check the path without it (set only through the test server's environment)
    if std::env::var_os("AM2_TEST_NO_FFMPEG").is_some() {
        return false;
    }
    Command::new(crate::server_tls::program("ffmpeg")).arg("-version").output().is_ok_and(|o| o.status.success())
}

/// `ffmpeg_available`, asked at most once a minute (a cluster master reads it with each worker's capabilities).
pub fn ffmpeg_cached() -> bool {
    static AT: std::sync::Mutex<Option<(std::time::Instant, bool)>> = std::sync::Mutex::new(None);
    if let Some((t, v)) = *AT.lock().unwrap() {
        if t.elapsed() < std::time::Duration::from_secs(60) {
            return v;
        }
    }
    let v = ffmpeg_available();
    *AT.lock().unwrap() = Some((std::time::Instant::now(), v));
    v
}

/// What compose needs of ffmpeg, missing from this one: the filters (xfade, zoompan, overlay, fade, afade, tpad; the captions are
/// drawn by Prastara, so libass is not needed) and
/// the H.264 and AAC encoders. Empty when it can compose. Homebrew's ffmpeg 9, for one, is built without libass, so it
/// has no `subtitles` filter. Asked at most once a minute.
pub fn compose_missing() -> Vec<&'static str> {
    static AT: std::sync::Mutex<Option<(std::time::Instant, Vec<&'static str>)>> = std::sync::Mutex::new(None);
    if let Some((t, v)) = AT.lock().unwrap().clone() {
        if t.elapsed() < std::time::Duration::from_secs(60) {
            return v;
        }
    }
    let v = if !ffmpeg_available() { vec!["ffmpeg"] } else {
        let run = |a: &str| Command::new(crate::server_tls::program("ffmpeg")).args(["-hide_banner", a]).output()
            .map(|o| String::from_utf8_lossy(&o.stdout).to_string()).unwrap_or_default();
        missing_for_compose(&run("-filters"), &run("-encoders"))
    };
    *AT.lock().unwrap() = Some((std::time::Instant::now(), v.clone()));
    v
}

/// The compose needs not in ffmpeg's `-filters` and `-encoders` listings.
pub fn missing_for_compose(filters: &str, encoders: &str) -> Vec<&'static str> {
    let has = |list: &str, name: &str| list.lines().any(|l| l.split_whitespace().nth(1) == Some(name));
    let mut out: Vec<&'static str> = ["xfade", "zoompan", "overlay", "afade", "tpad", "fade", "deflicker", "framerate"].into_iter().filter(|f| !has(filters, f)).collect();
    out.extend(["libx264", "aac"].into_iter().filter(|e| !has(encoders, e)));
    out
}

fn video_source(part: &Value) -> Option<String> {
    let kind = part.get("type").and_then(Value::as_str)?;
    if !matches!(kind, "video_url" | "video" | "input_video") {
        return None;
    }
    let v = part.get("video_url").or_else(|| part.get("video")).or_else(|| part.get("url")).or_else(|| part.get("source"));
    match v {
        Some(Value::String(s)) => Some(s.clone()),
        Some(Value::Object(o)) => o.get("url").or_else(|| o.get("path")).and_then(Value::as_str).map(str::to_owned).or_else(|| {
            // Anthropic-style {"type": "base64", "media_type": ..., "data": ...}
            o.get("data").and_then(Value::as_str).map(|d| format!("data:video/mp4;base64,{d}"))
        }),
        _ => None,
    }
}

/// What a request asks of its videos' frames.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VideoAsk {
    /// one frame each this many seconds; None: `max` frames evenly spread (the older `video_frames: N`)
    pub every_s: Option<f64>,
    pub max: usize,
    pub width: usize,
}

/// `video_fps` (default 1, 0.05 to 8), `video_max_frames` (default 64, at most 256), `video_width` (default 640, 64
/// to 1280); `video_frames: N` alone keeps the older meaning, N frames evenly spread.
pub fn video_ask(req: &Value) -> VideoAsk {
    let fps = req.get("video_fps").and_then(Value::as_f64).filter(|f| f.is_finite() && *f > 0.0).map(|f| f.clamp(0.05, 8.0));
    let even = req.get("video_frames").and_then(Value::as_u64).map(|n| (n as usize).clamp(1, MAX_FRAMES));
    let max = req.get("video_max_frames").and_then(Value::as_u64).map(|n| (n as usize).clamp(1, MAX_FRAMES))
        .or(even).unwrap_or(DEFAULT_MAX_FRAMES);
    let width = req.get("video_width").and_then(Value::as_u64).map(|w| (w as usize).clamp(64, 1280)).unwrap_or(DEFAULT_FRAME_WIDTH);
    let every_s = match (fps, even) {
        (Some(f), _) => Some(1.0 / f),
        (None, Some(_)) => None,
        (None, None) => Some(1.0 / DEFAULT_FPS),
    };
    VideoAsk { every_s, max, width }
}

/// The time a frame is shown at, as the model reads it before the frame: "[00:42]", "[1:02:03]"; tenths when the
/// frames are less than a second apart ("[00:02.5]").
pub fn timestamp(t: f64, tenths: bool) -> String {
    let t = t.max(0.0);
    let whole = if tenths { (t * 10.0).round() / 10.0 } else { t.floor() };
    let secs = whole.floor() as u64;
    let (h, m, s) = (secs / 3600, secs / 60 % 60, secs % 60);
    let frac = if tenths { format!(".{}", ((whole - whole.floor()) * 10.0).round() as u64 % 10) } else { String::new() };
    if h > 0 { format!("[{h}:{m:02}:{s:02}{frac}]") } else { format!("[{m:02}:{s:02}{frac}]") }
}

/// The note before a video's frames: how long it is, how many frames, how far apart.
pub fn video_note(duration_s: f64, frames: usize, every_s: Option<f64>, lowered: Option<&str>, how: Option<&str>) -> String {
    let rate = match every_s {
        Some(e) if (e - 1.0).abs() < 0.01 => ", one a second".to_string(),
        Some(e) if e < 1.0 => format!(", {:.1} a second", 1.0 / e),
        Some(e) => format!(", one every {e:.1} s"),
        None => String::new(),
    };
    let lowered = lowered.map(|w| format!(" ({w})")).unwrap_or_default();
    let len = if duration_s > 0.0 { format!(" of {duration_s:.1} s") } else { String::new() };
    let how = how.map(|n| format!(" ({n})")).unwrap_or_default();
    format!("[A video{len}, as {frames} frames in order{rate}{lowered}{how}, each after its time:]\n")
}

/// Frames with ffmpeg (any format), as JPEG bytes: one each `every_s` (fewer when over `max`: spread over the whole
/// video), else `max` evenly spread.
async fn ffmpeg_frames(video: &[u8], ask: VideoAsk) -> Result<Sampled> {
    let dir = tempdir()?;
    let input = dir.join("in.video");
    std::fs::write(&input, video)?;
    let probe = Command::new(crate::server_tls::program("ffprobe"))
        .args(["-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1"])
        .arg(&input)
        .output()?;
    let duration: f64 = String::from_utf8_lossy(&probe.stdout).trim().parse().unwrap_or(0.0);
    let n = ask.max;
    let mut fps = match ask.every_s {
        Some(e) => 1.0 / e,
        None if duration > 0.0 => n as f64 / duration,
        None => 1.0,
    };
    if duration > 0.0 && duration * fps > n as f64 {
        fps = n as f64 / duration;
    }
    let status = Command::new(crate::server_tls::program("ffmpeg"))
        .args(["-v", "error", "-i"])
        .arg(&input)
        .args(["-vf", &format!("fps={fps:.6},scale='min({},iw)':-2", ask.width), "-frames:v", &n.to_string(), "-q:v", "3"])
        .arg(dir.join("f%04d.jpg"))
        .status()?;
    if !status.success() {
        bail!("ffmpeg could not read the video");
    }
    let mut files: Vec<PathBuf> =
        std::fs::read_dir(&dir)?.flatten().map(|e| e.path()).filter(|p| p.extension().is_some_and(|x| x == "jpg")).collect();
    files.sort();
    let out = files.iter().map(std::fs::read).collect::<std::io::Result<Vec<_>>>()?;
    let _ = std::fs::remove_dir_all(&dir);
    if out.is_empty() {
        bail!("the video gave no frames");
    }
    Ok(Sampled { frames: out.into_iter().enumerate().map(|(k, b)| (k as f64 / fps, b)).collect(), fps: 0.0, duration_s: duration,
                 note: None, mime: "image/jpeg", keyframes_only: false })
}

pub fn tempdir() -> Result<PathBuf> {
    let d = std::env::temp_dir().join(format!("strata-video-{}-{}", std::process::id(), uuid::Uuid::new_v4().simple()));
    std::fs::create_dir_all(&d)?;
    Ok(d)
}

/// A video's frames: the media engine's own MP4/H.264 decoder (Rust, no ffmpeg) first; ffmpeg only for what it cannot
/// read (another format, or a B-frame MP4 it reads only the keyframes of) when ffmpeg is installed.
async fn sample_video(data: &[u8], ask: VideoAsk) -> Result<Sampled> {
    let rust = rust_frames_at(data, ask.every_s.unwrap_or(0.0), ask.max, ask.width, ask.every_s.is_some()).await;
    match rust {
        Ok(s) if !(s.keyframes_only && ffmpeg_available()) => Ok(s),
        _ if ffmpeg_available() => ffmpeg_frames(data, ask).await,
        Ok(s) => Ok(s),
        Err(e) => Err(anyhow!("this server cannot decode this video without ffmpeg: {e:#}. Install ffmpeg ({}), \
                               or send the video's frames as images in one message",
                              crate::doctor::install_hint(std::env::consts::OS, "ffmpeg", "ffmpeg", "Gyan.FFmpeg"))),
    }
}

/// The parts that stand for one video: the note, then each frame after its timestamp text, all marked
/// `"video_frame": {"v", "i", "t"}` (the note: `"note": true`; not "video", which Qwen's template reads as a video) so
/// the server encodes the frames as the engine reads them and
/// can thin them to fit the context (`thin_video`).
pub fn video_parts(v: usize, s: &Sampled, ask: VideoAsk) -> Vec<Value> {
    // tenths when the frames are not on whole seconds (two a second, or spread to the cap)
    let tenths = s.frames.iter().any(|(t, _)| (t - t.round()).abs() >= 0.05);
    // the rate the frames have: wider than asked when the cap spread them over a long video
    let n = s.frames.len();
    let actual = (n > 1).then(|| (s.frames[n - 1].0 - s.frames[0].0) / (n - 1) as f64);
    let (every_s, spread) = match (ask.every_s, actual) {
        (Some(e), Some(a)) if a > e * 1.2 => (Some(a), Some(format!("spread to the {}-frame cap", ask.max))),
        (e, _) => (e, None),
    };
    let mut out = vec![json!({"type": "text", "text": video_note(s.duration_s, n, every_s, spread.as_deref(), s.note.as_deref()),
                              "video_frame": {"v": v, "note": true, "duration_s": s.duration_s, "every_s": every_s, "how": s.note,
                                        "frames": n}})];
    for (i, (t, bytes)) in s.frames.iter().enumerate() {
        let tag = json!({"v": v, "i": i, "t": t});
        let lead = if i == 0 { "" } else { "\n" };
        out.push(json!({"type": "text", "text": format!("{lead}{}", timestamp(*t, tenths)), "video_frame": tag}));
        let url = format!("data:{};base64,{}", s.mime, base64::engine::general_purpose::STANDARD.encode(bytes));
        out.push(json!({"type": "image_url", "image_url": {"url": url}, "video_frame": tag}));
    }
    out.push(json!({"type": "text", "text": "\n"}));
    out
}

/// Keep `keep` frames of video `v` (evenly spread, its first always) in the rendered messages, dropping the others with
/// their timestamps, and say so in its note. -> the frames it had.
pub fn thin_video(messages: &mut [Value], v: usize, keep: usize) -> usize {
    let mut frames: Vec<usize> = Vec::new();
    for m in messages.iter() {
        for item in m.get("content").and_then(Value::as_array).into_iter().flatten() {
            if let Some(tag) = strata_frontend::images::frame_tag(item).filter(|t| t.video == v) {
                if item.get("type").and_then(Value::as_str) == Some("image") {
                    frames.push(tag.frame);
                }
            }
        }
    }
    let had = frames.len();
    if keep >= had || keep == 0 {
        return had;
    }
    let kept: std::collections::HashSet<usize> = if keep == 1 {
        [frames[0]].into()
    } else {
        (0..keep).map(|j| frames[((j as f64) * (had - 1) as f64 / (keep - 1) as f64).round() as usize]).collect()
    };
    for m in messages.iter_mut() {
        let Some(items) = m.get_mut("content").and_then(Value::as_array_mut) else { continue };
        items.retain(|item| strata_frontend::images::frame_tag(item).is_none_or(|t| t.video != v || kept.contains(&t.frame)));
        for item in items.iter_mut() {
            let note = item.get("video_frame").filter(|t| t.get("v").and_then(Value::as_u64) == Some(v as u64) && t.get("note") == Some(&json!(true))).cloned();
            if let Some(tag) = note {
                let d = tag["duration_s"].as_f64().unwrap_or(0.0);
                let every = tag["every_s"].as_f64().map(|e| e * had as f64 / keep as f64);
                item["text"] = json!(video_note(d, keep, every, Some("fewer, to fit the context"), tag["how"].as_str()));
            }
        }
    }
    had
}

/// Replace every video part in an OpenAI request's messages with its frames as image parts, each after a timestamp
/// ("[00:42]"), preceded by a note so the model reads them as one video in order. The request's `video_fps`,
/// `video_max_frames`, `video_width` (or the older `video_frames`) say how many (`video_ask`).
pub async fn expand_videos(req: &mut Value) -> Result<usize> {
    let ask = video_ask(req);
    let Some(messages) = req.get_mut("messages").and_then(Value::as_array_mut) else { return Ok(0) };
    let mut videos = 0;
    for m in messages.iter_mut() {
        let Some(parts) = m.get_mut("content").and_then(Value::as_array_mut) else { continue };
        if !parts.iter().any(|p| video_source(p).is_some()) {
            continue;
        }
        let mut out = Vec::with_capacity(parts.len() + 2 * ask.max);
        for p in parts.drain(..) {
            match video_source(&p) {
                Some(src) => {
                    let data = load_source(&src).await?;
                    let s = sample_video(&data, ask).await?;
                    out.extend(video_parts(videos, &s, ask));
                    videos += 1;
                }
                None => out.push(p),
            }
        }
        *parts = out;
    }
    Ok(videos)
}

#[cfg(test)]
mod video_tests {
    use super::*;

    #[test]
    fn timestamps_read_as_minutes_and_seconds() {
        assert_eq!(timestamp(0.0, false), "[00:00]");
        assert_eq!(timestamp(42.7, false), "[00:42]");
        assert_eq!(timestamp(3723.0, false), "[1:02:03]");
        assert_eq!(timestamp(2.5, true), "[00:02.5]");
        assert_eq!(timestamp(61.0, true), "[01:01.0]");
    }

    #[test]
    fn a_request_asks_for_a_rate_and_a_cap() {
        assert_eq!(video_ask(&json!({})), VideoAsk { every_s: Some(1.0), max: 64, width: 640 });
        assert_eq!(video_ask(&json!({"video_fps": 2, "video_max_frames": 1000, "video_width": 320})),
                   VideoAsk { every_s: Some(0.5), max: 256, width: 320 });
        // the older field: N frames evenly spread
        assert_eq!(video_ask(&json!({"video_frames": 8})), VideoAsk { every_s: None, max: 8, width: 640 });
    }

    #[test]
    fn frames_come_after_their_times_and_thin_evenly() {
        let s = Sampled { frames: (0..5).map(|k| (k as f64, vec![k as u8])).collect(), fps: 4.0, duration_s: 5.0, note: None,
                          mime: "image/png", keyframes_only: false };
        let parts = video_parts(0, &s, video_ask(&json!({})));
        let texts: Vec<&str> = parts.iter().filter(|p| p["type"] == "text").map(|p| p["text"].as_str().unwrap()).collect();
        assert!(texts[0].starts_with("[A video of 5.0 s, as 5 frames in order, one a second"), "{}", texts[0]);
        assert_eq!(&texts[1..6], &["[00:00]", "\n[00:01]", "\n[00:02]", "\n[00:03]", "\n[00:04]"]);
        // as the template sees them (strata-frontend's parts_of keeps "video_frame")
        let mut msgs = vec![json!({"role": "user", "content": parts.iter().map(|p| {
            let mut q = p.clone();
            if q["type"] == "image_url" { q = json!({"type": "image", "source": "x", "video_frame": p["video_frame"]}); }
            q
        }).collect::<Vec<_>>()})];
        assert_eq!(thin_video(&mut msgs, 0, 3), 5);
        let tags: Vec<usize> = strata_frontend::images::image_items(&msgs).iter().map(|(_, t)| t.unwrap().frame).collect();
        assert_eq!(tags, vec![0, 2, 4]);
        let note = msgs[0]["content"][0]["text"].as_str().unwrap();
        assert!(note.contains("as 3 frames") && note.contains("one every 1.7 s") && note.contains("fewer, to fit the context"), "{note}");
        let ts: Vec<&str> = msgs[0]["content"].as_array().unwrap().iter().filter(|p| p["type"] == "text").map(|p| p["text"].as_str().unwrap()).collect();
        assert_eq!(&ts[1..4], &["[00:00]", "\n[00:02]", "\n[00:04]"]);
    }
}

// ------------------------------------------------------------------------------------------------ documents (PDF)

/// Page images sent with a document's text (the rest of the pages go as text only).
pub const MAX_PAGE_IMAGES: usize = 20;

/// Poppler's `name`: the copy a release bundles (poppler/bin beside or above this binary's folder: the Windows
/// download carries one), else the one on PATH.
fn pdf_tool(name: &str) -> PathBuf {
    let file = format!("{name}{}", std::env::consts::EXE_SUFFIX);
    std::env::current_exe().ok().and_then(|exe| {
        let dir = exe.parent()?.to_path_buf();
        [dir.join("poppler/bin"), dir.join("../poppler/bin")].into_iter().map(|d| d.join(&file)).find(|p| p.is_file())
    }).unwrap_or_else(|| PathBuf::from(name))
}

pub fn pdftools_available() -> bool {
    Command::new(pdf_tool("pdftotext")).arg("-v").output().is_ok() && Command::new(pdf_tool("pdftoppm")).arg("-v").output().is_ok()
}

/// A PDF -> (its text, page by page, and up to `max_images` page images as JPEG; none when `images` is false).
pub async fn pdf_pages(pdf: &[u8], images: bool, max_images: usize) -> Result<(Vec<String>, Vec<Vec<u8>>)> {
    if !pdftools_available() {
        bail!("this server cannot read PDF files: {}", crate::doctor::install_hint(std::env::consts::OS, "poppler-utils", "poppler", "oschwartz10612.Poppler"));
    }
    if !pdf.starts_with(b"%PDF") {
        bail!("the document is not a PDF file");
    }
    let dir = tempdir()?;
    let input = dir.join("in.pdf");
    std::fs::write(&input, pdf)?;
    let out = Command::new(pdf_tool("pdftotext")).args(["-layout", "-enc", "UTF-8"]).arg(&input).arg("-").output()?;
    if !out.status.success() {
        let _ = std::fs::remove_dir_all(&dir);
        bail!("the PDF could not be read ({})", String::from_utf8_lossy(&out.stderr).trim());
    }
    // pdftotext ends each page with a form feed
    let text = String::from_utf8_lossy(&out.stdout).into_owned();
    let mut pages: Vec<String> = text.split('\u{c}').map(|p| p.trim_end().to_string()).collect();
    while pages.last().is_some_and(|p| p.is_empty()) {
        pages.pop();
    }
    let mut imgs = Vec::new();
    if images && max_images > 0 {
        let status = Command::new(pdf_tool("pdftoppm"))
            .args(["-jpeg", "-jpegopt", "quality=80", "-r", "110", "-l", &max_images.to_string()])
            .arg(&input)
            .arg(dir.join("p"))
            .status()?;
        if status.success() {
            let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)?.flatten().map(|e| e.path())
                .filter(|p| p.extension().is_some_and(|x| x == "jpg")).collect();
            files.sort();
            imgs = files.iter().take(max_images).map(std::fs::read).collect::<std::io::Result<Vec<_>>>()?;
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
    Ok((pages, imgs))
}

/// The text the model reads for a document: its name, then each page's text under a page heading.
fn document_text(name: &str, pages: &[String], images: usize) -> String {
    let mut s = format!("[Document: {name}, {} page{}{}]\n", pages.len(), if pages.len() == 1 { "" } else { "s" },
                        if images > 0 { format!("; the first {images} page{} also as pictures after the text", if images == 1 { "" } else { "s" }) } else { String::new() });
    for (i, p) in pages.iter().enumerate() {
        s.push_str(&format!("\n--- page {} ---\n{}\n", i + 1, if p.trim().is_empty() { "(no text on this page: see its picture)" } else { p }));
    }
    s
}

/// (data, file name) of a document part in any of the three APIs' shapes, or None when the part is not one.
fn document_source(part: &Value) -> Option<(String, String)> {
    let kind = part.get("type").and_then(Value::as_str)?;
    match kind {
        // OpenAI Chat: {"type": "file", "file": {"file_data": "data:application/pdf;base64,...", "filename": ...}}
        "file" => {
            let f = part.get("file")?;
            Some((f.get("file_data")?.as_str()?.to_string(), f.get("filename").and_then(Value::as_str).unwrap_or("document.pdf").to_string()))
        }
        // OpenAI Responses: {"type": "input_file", "file_data": ..., "filename": ...}
        "input_file" => Some((part.get("file_data")?.as_str()?.to_string(),
                              part.get("filename").and_then(Value::as_str).unwrap_or("document.pdf").to_string())),
        // Anthropic: {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": ...}}
        "document" => {
            let s = part.get("source")?;
            let name = part.get("title").and_then(Value::as_str).unwrap_or("document.pdf").to_string();
            match s.get("type").and_then(Value::as_str) {
                Some("base64") => Some((format!("data:application/pdf;base64,{}", s.get("data")?.as_str()?), name)),
                Some("url") => Some((s.get("url")?.as_str()?.to_string(), name)),
                Some("text") => Some((format!("text:{}", s.get("data")?.as_str()?), name)),
                _ => None,
            }
        }
        _ => None,
    }
}

/// Replace every document part (PDF) in a request's messages with its text and, for a model that reads pictures,
/// its first pages as images, in the request's own API shape (`anthropic`: image blocks, else image_url parts).
pub async fn expand_documents(req: &mut Value, images: bool, anthropic: bool) -> Result<usize> {
    let key = if req.get("input").is_some_and(Value::is_array) { "input" } else { "messages" };
    let Some(messages) = req.get_mut(key).and_then(Value::as_array_mut) else { return Ok(0) };
    let mut docs = 0;
    let mut budget = MAX_PAGE_IMAGES;
    for m in messages.iter_mut() {
        let Some(parts) = m.get_mut("content").and_then(Value::as_array_mut) else { continue };
        if !parts.iter().any(|p| document_source(p).is_some()) {
            continue;
        }
        let mut out = Vec::with_capacity(parts.len() + 4);
        for p in parts.drain(..) {
            let Some((src, name)) = document_source(&p) else {
                out.push(p);
                continue;
            };
            docs += 1;
            let text_kind = if key == "input" { "input_text" } else { "text" };
            if let Some(plain) = src.strip_prefix("text:") {
                out.push(json!({"type": text_kind, "text": format!("[Document: {name}]\n{plain}")}));
                continue;
            }
            let data = load_source(&src).await?;
            let (pages, imgs) = pdf_pages(&data, images, budget).await?;
            budget -= imgs.len();
            out.push(json!({"type": text_kind, "text": document_text(&name, &pages, imgs.len())}));
            for img in imgs {
                let b64 = base64::engine::general_purpose::STANDARD.encode(img);
                out.push(if anthropic {
                    json!({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}})
                } else if key == "input" {
                    json!({"type": "input_image", "image_url": format!("data:image/jpeg;base64,{b64}")})
                } else {
                    json!({"type": "image_url", "image_url": {"url": format!("data:image/jpeg;base64,{b64}")}})
                });
            }
        }
        *parts = out;
    }
    Ok(docs)
}
