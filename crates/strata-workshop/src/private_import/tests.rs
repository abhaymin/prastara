//! The import end to end with fake check models and synthetic stand-in files only: solid-colour PNGs and
//! lorem-ipsum text. A fake flags an image by its colour (red is the stand-in for "flagged"), never by content.

use super::checks::tests::{fake, Fake};
use super::checks::{CheckKind, Input};
use super::*;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};

const LOREM: &str = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.";
const SOURCE: &str = "owner_attests_adults_only = true\nowner_attests_consent = true\norigin = \"synthetic test files\"\n";

struct Scratch {
    base: PathBuf,
}

impl Scratch {
    fn new(tag: &str) -> Scratch {
        let base = std::env::temp_dir().join(format!("pi-run-{tag}-{}", std::process::id()));
        std::fs::remove_dir_all(&base).ok();
        std::fs::create_dir_all(base.join("src")).unwrap();
        Scratch { base }
    }
    fn src(&self) -> PathBuf {
        self.base.join("src")
    }
    fn root(&self) -> PathBuf {
        self.base.join("data/private")
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        // a read-only test tree is made writable again first
        let _ = std::process::Command::new("chmod").args(["-R", "u+w"]).arg(&self.base).status();
        std::fs::remove_dir_all(&self.base).ok();
    }
}

fn png(rgb: [u8; 3], side: u32) -> Vec<u8> {
    let img = image::RgbImage::from_pixel(side, side, image::Rgb(rgb));
    let mut out = std::io::Cursor::new(Vec::new());
    img.write_to(&mut out, image::ImageFormat::Png).unwrap();
    out.into_inner()
}

/// The fake checks: an image is flagged (by whichever check `flag_by` names) when it is pure red; texts pass.
fn loader(flag_by: CheckKind) -> impl FnMut(CheckKind) -> std::result::Result<Check, Unavailable> {
    move |k| {
        let flagged = if k == CheckKind::Age { "0-17" } else { "flag" };
        let ok = if k == CheckKind::Age { "18+" } else { "ok" };
        let f = Fake::By(Box::new(move |i: &Input| {
            let red = matches!(i, Input::Image(img) if img.get_pixel(0, 0).0 == [255, 0, 0]);
            let hit = red && k == flag_by;
            let mut s = vec![(ok.to_string(), if hit { 0.1 } else { 0.95 }), (flagged.to_string(), if hit { 0.9 } else { 0.05 })];
            if k == CheckKind::Age {
                s.push(("none".into(), 0.0));
            }
            s
        }));
        Ok(fake(k, f))
    }
}

fn settings(sc: &Scratch, name: &str) -> Settings {
    let mut s = Settings::new(sc.root(), name, Spec::Folder(sc.src()));
    s.id = format!("{name}-1");
    s.timeout = Duration::from_millis(500);
    s
}

fn audit(sc: &Scratch) -> Vec<Value> {
    std::fs::read_to_string(sc.root().join(AUDIT)).unwrap_or_default().lines().map(|l| serde_json::from_str(l).unwrap()).collect()
}

fn mode(p: &Path) -> u32 {
    use std::os::unix::fs::PermissionsExt as _;
    std::fs::metadata(p).unwrap().permissions().mode() & 0o777
}

fn go(s: &Settings, flag_by: CheckKind) -> Outcome {
    run(s, None, &mut loader(flag_by), &mut |_, _| {}, &|| false).unwrap()
}

fn write_set(sc: &Scratch, n_plain: usize, n_red: usize) {
    std::fs::write(sc.src().join(pd::SOURCE_FILE), SOURCE).unwrap();
    for i in 0..n_red {
        std::fs::write(sc.src().join(format!("a-{i:03}.png")), png([255, 0, 0], 8 + i as u32)).unwrap();
    }
    for i in 0..n_plain {
        std::fs::write(sc.src().join(format!("b-{i:03}.png")), png([0, (i % 250) as u8, 200], 8)).unwrap();
    }
}

#[test]
fn a_folder_is_quarantined_checked_and_released_private() {
    let sc = Scratch::new("ok");
    write_set(&sc, 3, 0);
    std::fs::create_dir_all(sc.src().join("notes")).unwrap();
    std::fs::write(sc.src().join("notes/lorem.txt"), LOREM).unwrap();
    std::fs::write(sc.src().join("notes/chats.jsonl"), format!("{}\n", json!({"messages": [{"role": "user", "content": LOREM}, {"role": "assistant", "content": LOREM}]}))).unwrap();
    let s = settings(&sc, "set");
    let Outcome::Released { dir, counts } = go(&s, CheckKind::ImageMinor) else { panic!("not released") };
    assert_eq!((counts.items, counts.released, counts.rejected, counts.deleted), (5, 5, 0, 0), "{counts:?}");
    assert_eq!(dir, sc.root().join("datasets/set"));
    assert!(!s.quarantine().exists(), "the quarantine goes after the release");
    let card = std::fs::read_to_string(dir.join(pd::CARD)).unwrap();
    assert!(card.starts_with(pd::CARD_MARK) && card.contains("synthetic test files"), "{card}");
    assert!(pd::is_private_dataset(&dir.join("manifest.jsonl")) && pd::is_private_dataset(&dir.join("text.jsonl")));
    for p in [dir.clone(), dir.join("files"), sc.root(), sc.root().join(QUARANTINE)] {
        assert_eq!(mode(&p), 0o700, "{}", p.display());
    }
    for p in [dir.join("manifest.jsonl"), dir.join("text.jsonl"), dir.join(pd::CARD), sc.root().join(AUDIT)] {
        assert_eq!(mode(&p), 0o600, "{}", p.display());
    }
    assert_eq!(std::fs::read_dir(dir.join("files")).unwrap().count(), 5);
    // the audit log: the events, hashes and counts; never content, never the source's file names
    let a = audit(&sc);
    let events: Vec<&str> = a.iter().map(|e| e["event"].as_str().unwrap()).collect();
    assert_eq!(events, ["start", "walk", "quarantined", "release"]);
    assert!(a.iter().all(|e| e["t"].as_u64().is_some() && e["import"] == "set-1"));
    let text = std::fs::read_to_string(sc.root().join(AUDIT)).unwrap();
    assert!(!text.contains("Lorem") && !text.contains("lorem.txt") && !text.contains("b-000"), "{text}");
    assert_eq!(a[3]["sha256"].as_array().unwrap().len(), 5);
    // a second import of the same name is refused
    let mut again = settings(&sc, "set");
    again.id = "set-2".into();
    assert!(run(&again, None, &mut loader(CheckKind::ImageMinor), &mut |_, _| {}, &|| false).unwrap_err().to_string().contains("exists"));
}

#[test]
fn the_private_mark_is_carried_through_clean() {
    let sc = Scratch::new("carry");
    std::fs::write(sc.src().join(pd::SOURCE_FILE), SOURCE).unwrap();
    std::fs::write(sc.src().join("chats.jsonl"), (0..3).map(|i| format!("{}\n", json!({"messages": [{"role": "user", "content": format!("{LOREM} {i}")},
        {"role": "assistant", "content": format!("Answer number {i}: sed ut perspiciatis unde omnis iste natus error sit voluptatem accusantium doloremque laudantium")}]}))).collect::<String>()).unwrap();
    let Outcome::Released { dir, .. } = go(&settings(&sc, "carry"), CheckKind::ImageMinor) else { panic!() };
    let rows = dir.join("text.jsonl");
    // what train reads to mark its adapter private (finetune: run.private) is the same function
    assert!(pd::is_private_dataset(&rows));
    let out = sc.base.join("clean/out.jsonl");
    std::fs::create_dir_all(out.parent().unwrap()).unwrap();
    let mut cs = crate::clean::Settings::new(vec![rows], out.clone());
    cs.min_answer_tokens = 1;
    crate::clean::run(&cs, |_| {}).unwrap();
    let cleaned = std::fs::read_to_string(&out).unwrap();
    assert!(!cleaned.is_empty() && cleaned.lines().all(|l| serde_json::from_str::<Value>(l).unwrap()["private"] == true), "{cleaned}");
}

#[test]
fn a_read_only_source_is_never_written_and_links_are_not_followed() {
    use std::os::unix::fs::PermissionsExt as _;
    let sc = Scratch::new("ro");
    write_set(&sc, 2, 0);
    std::fs::write(sc.src().join(".hidden.png"), png([1, 2, 3], 8)).unwrap();
    let outside = sc.base.join("outside.png");
    std::fs::write(&outside, png([9, 9, 9], 8)).unwrap();
    std::os::unix::fs::symlink(&outside, sc.src().join("link.png")).unwrap();
    std::os::unix::fs::symlink(&sc.base, sc.src().join("loop")).unwrap();
    // read only, as the NAS mount is
    for e in std::fs::read_dir(sc.src()).unwrap().flatten() {
        if !e.file_type().unwrap().is_symlink() {
            std::fs::set_permissions(e.path(), std::fs::Permissions::from_mode(0o444)).unwrap();
        }
    }
    std::fs::set_permissions(sc.src(), std::fs::Permissions::from_mode(0o555)).unwrap();
    let snapshot = |d: &Path| -> Vec<(String, u64, std::time::SystemTime, String)> {
        let mut v: Vec<_> = std::fs::read_dir(d).unwrap().flatten().map(|e| {
            let m = std::fs::symlink_metadata(e.path()).unwrap();
            let h = if m.is_file() { pd::sha256_hex(&std::fs::read(e.path()).unwrap()) } else { String::new() };
            (e.file_name().to_string_lossy().to_string(), m.len(), m.modified().unwrap(), h)
        }).collect();
        v.sort();
        v
    };
    let before = (snapshot(&sc.src()), mode(&sc.src()));
    let Outcome::Released { counts, dir } = go(&settings(&sc, "ro"), CheckKind::ImageMinor) else { panic!() };
    assert_eq!((snapshot(&sc.src()), mode(&sc.src())), before, "the source is exactly as it was");
    assert_eq!(counts.released, 2, "{counts:?}");
    assert!(counts.skipped >= 3, "the hidden file and both links are skipped: {counts:?}");
    let released: Vec<String> = std::fs::read_dir(dir.join("files")).unwrap().flatten().map(|e| e.file_name().to_string_lossy().to_string()).collect();
    assert!(!released.iter().any(|n| n.starts_with(&pd::sha256_hex(&png([9, 9, 9], 8)))), "a link out of the source is never read");
}

#[test]
fn a_file_that_disappears_mid_walk_is_skipped() {
    let sc = Scratch::new("gone");
    write_set(&sc, 2, 0);
    let s = settings(&sc, "gone");
    pd::private_dir(&s.quarantine().join("items")).unwrap();
    pd::private_dir(&s.quarantine().join("incoming")).unwrap();
    let st = State { id: s.id.clone(), name: s.name.clone(), source: s.spec.describe(), origin: "x".into(), licence: None, commit: None, created: 0,
                     status: "fetching".into(), reason: None, counts: Counts::default(), items: vec![], fetched_done: false, done_files: vec![] };
    let mut noop = |_: &Counts, _: &str| {};
    let mut r = Run { s: &s, dir: s.quarantine(), audit: Audit::open(&s.root).unwrap(), state: st, progress: &mut noop };
    let listed = vec![sc.src().join("b-000.png"), sc.src().join("b-001.png")];
    std::fs::remove_file(&listed[0]).unwrap(); // gone between the listing and the copy
    r.copy_listed(&listed, &|| false).unwrap();
    assert_eq!((r.state.items.len(), r.state.counts.skipped), (1, 1));
}

#[test]
fn a_minor_flag_deletes_the_item_at_once_and_blocks_its_hash() {
    for by in [CheckKind::ImageMinor, CheckKind::Age] {
        let sc = Scratch::new(&format!("minor-{}", by.id()));
        write_set(&sc, 199, 1); // 1 of 200: under the 1 % limit
        let red = pd::sha256_hex(&png([255, 0, 0], 8));
        let s = settings(&sc, "m");
        let red_file = s.quarantine().join("items").join(format!("{red}.png"));
        // the checks of every later item see that the flagged one is already gone from quarantine
        let gone_seen = Arc::new(AtomicBool::new(false));
        let (g2, rf) = (gone_seen.clone(), red_file.clone());
        let mut base = loader(by);
        let mut l = move |k: CheckKind| -> std::result::Result<Check, Unavailable> {
            if k == CheckKind::ImageRealPerson {
                let (g3, rf2) = (g2.clone(), rf.clone());
                return Ok(fake(k, Fake::By(Box::new(move |_| {
                    if !rf2.exists() { g3.store(true, Ordering::SeqCst); }
                    vec![("ok".into(), 1.0), ("flag".into(), 0.0)]
                }))));
            }
            base(k)
        };
        let Outcome::Released { counts, dir } = run(&s, None, &mut l, &mut |_, _| {}, &|| false).unwrap() else { panic!() };
        assert!(gone_seen.load(Ordering::SeqCst), "deleted before the next item was checked");
        assert_eq!((counts.deleted, counts.minor, counts.released), (1, 1, 199), "{counts:?}");
        assert!(!dir.join("files").join(format!("{red}.png")).exists());
        let bl = std::fs::read_to_string(sc.root().join(pd::BLOCKLIST)).unwrap();
        assert!(bl.contains(&red), "{bl}");
        let del: Vec<Value> = audit(&sc).into_iter().filter(|e| e["event"] == "delete").collect();
        assert_eq!(del.len(), 1);
        let keys: Vec<&String> = del[0].as_object().unwrap().keys().collect();
        assert_eq!(keys.len(), 5, "only the hash, the reason, the time, the import and the event: {keys:?}");
        assert_eq!((del[0]["sha256"].as_str(), del[0]["reason"].as_str()), (Some(red.as_str()), Some(flag_reason(by).as_str())));
        // a later import meets the blocked hash and deletes it before any check
        let sc2 = Scratch::new(&format!("minor2-{}", by.id()));
        write_set(&sc2, 200, 0);
        std::fs::write(sc2.src().join("again.png"), png([255, 0, 0], 8)).unwrap();
        let mut s2 = settings(&sc2, "m2");
        s2.root = sc.root();
        let Outcome::Released { counts, .. } = go(&s2, CheckKind::TextMinor) else { panic!() };
        assert_eq!((counts.deleted, counts.reasons.get(reason::BLOCKLISTED)), (1, Some(&1)), "{counts:?}");
    }
}

#[test]
fn a_source_over_the_minor_share_is_rejected_whole_and_its_quarantine_deleted() {
    let sc = Scratch::new("share");
    write_set(&sc, 8, 2);
    let s = settings(&sc, "share");
    let Outcome::Rejected { reason, counts } = go(&s, CheckKind::ImageMinor) else { panic!("not rejected") };
    assert!(reason.contains("whole import is rejected"), "{reason}");
    assert!(counts.minor >= 1 && counts.checked < 10, "it stops once the share is passed: {counts:?}");
    assert!(!s.quarantine().exists() && !s.dataset().exists(), "nothing is kept and nothing released");
    let a = audit(&sc);
    assert_eq!(a.last().unwrap()["event"], "import_rejected");
    // a stricter limit asked: 0 rejects on one flagged item; a looser one is not taken
    let sc2 = Scratch::new("share0");
    write_set(&sc2, 300, 1);
    let mut s2 = settings(&sc2, "s0");
    s2.minor_share = Some(0.0);
    assert!(matches!(go(&s2, CheckKind::ImageMinor), Outcome::Rejected { .. }));
    let sc3 = Scratch::new("share5");
    write_set(&sc3, 18, 2);
    let mut s3 = settings(&sc3, "s5");
    s3.minor_share = Some(0.5);
    assert!(matches!(go(&s3, CheckKind::ImageMinor), Outcome::Rejected { .. }), "a looser limit is not taken");
}

#[test]
fn failures_are_rejections_never_passes() {
    let sc = Scratch::new("fail");
    write_set(&sc, 1, 0);
    let mut broken_png = b"\x89PNG\r\n\x1a\n".to_vec();
    broken_png.extend_from_slice(b"not an image at all");
    std::fs::write(sc.src().join("c-broken.png"), broken_png).unwrap();
    std::fs::write(sc.src().join("c-binary.bin"), [0u8, 1, 2, 3]).unwrap();
    std::fs::write(sc.src().join("c-latin1.txt"), [0x4c, 0x6f, 0x72, 0x65, 0x6d, 0xff, 0xfe]).unwrap();
    std::fs::write(sc.src().join("d-err.png"), png([7, 7, 7], 8)).unwrap();
    let mut base = loader(CheckKind::ImageMinor);
    // the real-person check fails on one image (its colour), answers the others
    let mut l = move |k: CheckKind| -> std::result::Result<Check, Unavailable> {
        if k == CheckKind::ImageRealPerson {
            return Ok(fake(k, Fake::By(Box::new(|i| match i {
                Input::Image(img) if img.get_pixel(0, 0).0 == [7, 7, 7] => vec![],
                _ => vec![("ok".into(), 1.0), ("flag".into(), 0.0)],
            }))));
        }
        base(k)
    };
    let Outcome::Released { counts, .. } = run(&settings(&sc, "f"), None, &mut l, &mut |_, _| {}, &|| false).unwrap() else { panic!() };
    assert_eq!(counts.released, 1, "{counts:?}");
    assert_eq!(counts.reasons.get(reason::UNDECODABLE), Some(&2), "a broken PNG and text that is not UTF-8: {counts:?}");
    assert_eq!(counts.reasons.get(reason::UNSUPPORTED), Some(&1));
    assert_eq!(counts.reasons.get(reason::CHECK_FAILED), Some(&1), "no scores is a rejection");
    assert_eq!(counts.deleted, 0, "a failure is not a minor flag");
}

#[test]
fn a_timeout_or_a_missing_model_stops_the_import_in_quarantine() {
    let sc = Scratch::new("stop");
    write_set(&sc, 3, 0);
    std::fs::write(sc.src().join("lorem.txt"), LOREM).unwrap();
    let s = settings(&sc, "stop");
    // the text checks are not installed
    let mut l = |k: CheckKind| -> std::result::Result<Check, Unavailable> {
        if k.is_image() { loader(CheckKind::ImageMinor)(k) } else { Err(Unavailable::NotInstalled(k)) }
    };
    let Outcome::Stopped { reason, .. } = run(&s, None, &mut l, &mut |_, _| {}, &|| false).unwrap() else { panic!() };
    assert_eq!(reason, "the text minor-safety check model is not installed; nothing is released; \
                        the text real-person check model is not installed; nothing is released");
    assert!(s.quarantine().join("items").is_dir() && !s.dataset().exists(), "kept in quarantine, nothing released");
    assert_eq!(list(&s.root)[0]["status"], "stopped");
    // a check that hangs: stopped again, nothing released
    let mut hang = |k: CheckKind| -> std::result::Result<Check, Unavailable> {
        if k == CheckKind::Age { Ok(fake(k, Fake::Hang)) } else { loader(CheckKind::ImageMinor)(k) }
    };
    let Outcome::Stopped { reason, .. } = run(&s, None, &mut hang, &mut |_, _| {}, &|| false).unwrap() else { panic!() };
    assert!(reason.contains("age estimate check model did not answer in time; nothing is released"), "{reason}");
    assert!(!s.dataset().exists());
    // with every model there, the resumed import is checked and released (nothing fetched again)
    let Outcome::Released { counts, .. } = go(&s, CheckKind::ImageMinor) else { panic!() };
    assert_eq!(counts.fetched, 4);
    assert!(counts.released >= 3, "{counts:?}");
    let events: Vec<String> = audit(&sc).iter().map(|e| e["event"].as_str().unwrap().to_string()).collect();
    assert_eq!(events.iter().filter(|e| *e == "stopped").count(), 2);
    assert_eq!(events.iter().filter(|e| *e == "walk").count(), 1, "the resume does not walk again: {events:?}");
    // the owner's stop: nothing released
    let sc2 = Scratch::new("stop2");
    write_set(&sc2, 2, 0);
    let s2 = settings(&sc2, "stop2");
    let r = run(&s2, None, &mut loader(CheckKind::ImageMinor), &mut |_, _| {}, &|| true).unwrap();
    assert!(matches!(r, Outcome::Stopped { .. }) && !s2.dataset().exists());
    discard(&s2.root, &s2.id).unwrap();
    assert!(!s2.quarantine().exists());
    assert_eq!(audit(&sc2).last().unwrap()["event"], "discard");
}

#[test]
fn the_word_rules_delete_a_text_before_any_model() {
    let sc = Scratch::new("rules");
    write_set(&sc, 300, 0);
    // a plainly innocent pair of trigger words, as the rules' own tests use: still a match, by design
    std::fs::write(sc.src().join("court.txt"), "the juvenile court kept the explicit details private").unwrap();
    let Outcome::Released { counts, .. } = go(&settings(&sc, "r"), CheckKind::ImageMinor) else { panic!() };
    assert_eq!(counts.reasons.get(reason::MINOR_RULES), Some(&1), "{counts:?}");
    assert_eq!(counts.deleted, 1);
}

#[test]
fn a_source_needs_its_attestation() {
    let sc = Scratch::new("attest");
    std::fs::write(sc.src().join("a.png"), png([0, 0, 255], 8)).unwrap();
    let s = settings(&sc, "a");
    let e = run(&s, None, &mut loader(CheckKind::ImageMinor), &mut |_, _| {}, &|| false).unwrap_err().to_string();
    assert!(e.contains("SOURCE.toml"), "{e}");
    assert!(!s.root.join(QUARANTINE).exists(), "nothing is read without it");
    // a file outside the folder must name it
    let side = sc.base.join("attest.toml");
    std::fs::write(&side, SOURCE).unwrap();
    assert!(read_attestation(&Spec::Folder(sc.src()), Some(&side)).unwrap_err().to_string().contains("does not name"));
    std::fs::write(&side, format!("{SOURCE}folder = \"{}\"\n", sc.src().display())).unwrap();
    assert!(read_attestation(&Spec::Folder(sc.src()), Some(&side)).is_ok());
    // a URL needs its url and a licence
    let url = Spec::Url("https://example.com/set.zip".into());
    std::fs::write(&side, format!("{SOURCE}url = \"https://example.com/set.zip\"\n")).unwrap();
    assert!(read_attestation(&url, Some(&side)).unwrap_err().to_string().contains("licence"));
    std::fs::write(&side, format!("{SOURCE}url = \"https://example.com/other.zip\"\nlicence = \"cc-by-4.0\"\n")).unwrap();
    assert!(read_attestation(&url, Some(&side)).is_err());
    std::fs::write(&side, format!("{SOURCE}url = \"https://example.com/set.zip\"\nlicence = \"cc-by-4.0\"\n")).unwrap();
    assert_eq!(read_attestation(&url, Some(&side)).unwrap().licence.as_deref(), Some("cc-by-4.0"));
    assert!(read_attestation(&url, None).is_err(), "no folder to hold it");
}

#[test]
fn a_url_archive_is_fetched_after_its_size_is_confirmed() {
    let sc = Scratch::new("url");
    let zip = super::archive::tests::zip_of(&[("p/one.png", &png([0, 0, 255], 8)), ("p/two.png", &png([0, 255, 0], 8)), ("notes.txt", LOREM.as_bytes())]);
    let srv = super::fetch::tests::serve(HashMap::from([("/set.zip".to_string(), ("application/zip".to_string(), zip.clone()))]), None);
    let url = format!("{}/set.zip", srv.base);
    let side = sc.base.join("SOURCE.toml");
    std::fs::write(&side, format!("{SOURCE}url = \"{url}\"\nlicence = \"test-only\"\n")).unwrap();
    let net = Net::new(None, None).unwrap();
    let mut s = Settings::new(sc.root(), "web", Spec::Url(url.clone()));
    s.id = "web-1".into();
    s.attestation = Some(side.clone());
    // no confirmation, or a wrong one: nothing fetched
    let r = run(&s, Some(&net), &mut loader(CheckKind::ImageMinor), &mut |_, _| {}, &|| false).unwrap();
    assert!(matches!(&r, Outcome::Stopped { reason, .. } if reason.contains("confirm")), "{r:?}");
    assert_eq!(srv.hits.lock().unwrap().get("/set.zip").copied(), Some(1), "the probe only (HEAD)");
    discard(&s.root, &s.id).unwrap();
    s.confirm_bytes = Some(zip.len() as u64 + 1);
    let r = run(&s, Some(&net), &mut loader(CheckKind::ImageMinor), &mut |_, _| {}, &|| false).unwrap();
    assert!(matches!(&r, Outcome::Stopped { reason, .. } if reason.contains("not the")), "{r:?}");
    discard(&s.root, &s.id).unwrap();
    s.confirm_bytes = Some(zip.len() as u64);
    let Outcome::Released { counts, dir } = run(&s, Some(&net), &mut loader(CheckKind::ImageMinor), &mut |_, _| {}, &|| false).unwrap() else { panic!() };
    assert_eq!((counts.fetched, counts.items, counts.released), (1, 3, 3), "{counts:?}");
    assert!(std::fs::read_to_string(dir.join(pd::CARD)).unwrap().contains("test-only"));
    let a = audit(&sc);
    let fetch = a.iter().find(|e| e["event"] == "fetch").unwrap();
    assert_eq!(fetch["sha256"].as_str(), Some(pd::sha256_hex(&zip).as_str()));
    assert!(a.iter().any(|e| e["event"] == "unpack" && e["items"] == 3));
}

#[test]
fn a_hub_dataset_is_fetched_pinned_to_its_commit() {
    let sc = Scratch::new("hf");
    let commit = "89abcdef0123456789abcdef0123456789abcdef";
    let img = png([0, 0, 255], 8);
    let tree = json!([{"type": "file", "path": "images/a.png", "size": img.len(), "lfs": {"oid": pd::sha256_hex(&img), "size": img.len()}},
                      {"type": "file", "path": "text/b.txt", "size": LOREM.len()}]);
    let srv = super::fetch::tests::serve(HashMap::from([
        ("/api/datasets/someone/set/revision/main".to_string(), ("application/json".to_string(), json!({"sha": commit}).to_string().into_bytes())),
        (format!("/api/datasets/someone/set/tree/{commit}?recursive=true"), ("application/json".to_string(), tree.to_string().into_bytes())),
        (format!("/datasets/someone/set/resolve/{commit}/images/a.png"), ("image/png".to_string(), img.clone())),
        (format!("/datasets/someone/set/resolve/{commit}/text/b.txt"), ("text/plain".to_string(), LOREM.as_bytes().to_vec())),
    ]), None);
    let side = sc.base.join("SOURCE.toml");
    std::fs::write(&side, format!("{SOURCE}hf = \"someone/set\"\nlicence = \"test-only\"\n")).unwrap();
    let net = Net::new(Some(srv.base.clone()), None).unwrap();
    let spec = Spec::Hf { id: "someone/set".into(), revision: "main".into(), pattern: None };
    let probe = net.probe(&spec).unwrap();
    let mut s = Settings::new(sc.root(), "hub", spec);
    s.id = "hub-1".into();
    s.attestation = Some(side);
    s.confirm_bytes = Some(probe.bytes);
    let Outcome::Released { counts, dir } = run(&s, Some(&net), &mut loader(CheckKind::ImageMinor), &mut |_, _| {}, &|| false).unwrap() else { panic!() };
    assert_eq!((counts.fetched, counts.released), (2, 2), "{counts:?}");
    assert!(std::fs::read_to_string(dir.join(pd::CARD)).unwrap().contains(commit));
    let fetches: Vec<Value> = audit(&sc).into_iter().filter(|e| e["event"] == "fetch").collect();
    assert_eq!(fetches.len(), 2);
    assert!(fetches.iter().all(|f| f["commit"] == commit) && fetches.iter().any(|f| f["sha256_checked"] == true));
}

#[test]
fn quarantine_paths_are_refused_for_every_job_kind() {
    let q = "/home/x/Development/Strata-data/private/quarantine/set-1/items/ab.png";
    assert!(in_quarantine(Path::new(q)));
    assert!(!in_quarantine(Path::new("/home/x/Development/Strata-data/private/datasets/set/manifest.jsonl")));
    assert!(!in_quarantine(Path::new("/home/x/quarantine/private/a")));
    // a link into the quarantine counts too
    let sc = Scratch::new("qlink");
    let qd = sc.root().join("quarantine/x-1");
    std::fs::create_dir_all(&qd).unwrap();
    std::fs::write(qd.join("a.jsonl"), "{}\n").unwrap();
    std::os::unix::fs::symlink(qd.join("a.jsonl"), sc.base.join("innocent.jsonl")).unwrap();
    assert!(in_quarantine(&sc.base.join("innocent.jsonl")));
    // each Workshop command, as the server starts it, with a quarantined input
    let cases: [&[&str]; 9] = [
        &["clean", q, "--out", "/tmp/o.jsonl"],
        &["train", "--model", "/m", "--data", q, "--out", "/tmp/a"],
        &["datagen", "--recipe", "qa", "--source", q, "--out", "/tmp/d.jsonl"],
        &["merge", "/m", q, "--out", "/tmp/m"],
        &["export-moe", q, "--out", "/tmp/e.gguf"],
        &["regress", "--model", "/m", "--suite", q, "--out", "/tmp/r"],
        &["grade-train", "--model", "/m", "--data", q, "--suite", "/s.json", "--out", "/tmp/g"],
        &["private-data", "--name", "x", "--source", q],
        &["convert", &format!("--input={q}"), "--out", "/tmp/c.gguf"],
    ];
    for args in cases {
        assert!(quarantine_args(args).is_some_and(|e| e.contains("quarantine")), "{args:?}");
    }
    assert!(quarantine_args(&["clean", "/home/x/data/a.jsonl", "--out", "/tmp/o.jsonl"]).is_none());
}
