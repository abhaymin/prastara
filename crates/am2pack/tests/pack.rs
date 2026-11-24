// SPDX-License-Identifier: AGPL-3.0-only
//! P0 and P1's gates on a small made-up model: a pack builds and reads back, `unpack` returns the sources byte for
//! byte, and a flipped byte anywhere in the file is caught.

use am2pack::{build, gguf, kind, unpack, BuildOptions, Pack};
use std::path::{Path, PathBuf};

fn s(out: &mut Vec<u8>, v: &str) {
    out.extend_from_slice(&(v.len() as u64).to_le_bytes());
    out.extend_from_slice(v.as_bytes());
}

/// A GGUF with the given tensors (name, dims, type id, data); `junk` puts non-zero bytes into the padding after
/// tensor 2 and after the last one.
fn gguf_file(arch: &str, n_expert: u32, tensors: &[(&str, Vec<u64>, u32, Vec<u8>)], junk: bool) -> Vec<u8> {
    let mut h = Vec::new();
    h.extend_from_slice(b"GGUF");
    h.extend_from_slice(&3u32.to_le_bytes());
    h.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    h.extend_from_slice(&4u64.to_le_bytes());
    s(&mut h, "general.architecture"); h.extend_from_slice(&8u32.to_le_bytes()); s(&mut h, arch);
    s(&mut h, &format!("{arch}.expert_count")); h.extend_from_slice(&4u32.to_le_bytes()); h.extend_from_slice(&n_expert.to_le_bytes());
    // an array of strings and one of numbers, to be stepped over
    s(&mut h, "tokenizer.ggml.tokens"); h.extend_from_slice(&9u32.to_le_bytes()); h.extend_from_slice(&8u32.to_le_bytes());
    h.extend_from_slice(&3u64.to_le_bytes()); for t in ["a", "bc", "def"] { s(&mut h, t); }
    s(&mut h, "tokenizer.ggml.scores"); h.extend_from_slice(&9u32.to_le_bytes()); h.extend_from_slice(&6u32.to_le_bytes());
    h.extend_from_slice(&3u64.to_le_bytes()); h.extend_from_slice(&[0u8; 12]);
    let mut data = Vec::new();
    for (i, (name, dims, ty, bytes)) in tensors.iter().enumerate() {
        while data.len() % 32 != 0 { data.push(if junk && i == 3 { 0xA5 } else { 0 }); }
        s(&mut h, name);
        h.extend_from_slice(&(dims.len() as u32).to_le_bytes());
        for d in dims { h.extend_from_slice(&d.to_le_bytes()); }
        h.extend_from_slice(&ty.to_le_bytes());
        h.extend_from_slice(&(data.len() as u64).to_le_bytes());
        data.extend_from_slice(bytes);
    }
    while h.len() % 32 != 0 { h.push(0); }
    h.extend_from_slice(&data);
    if junk { h.extend_from_slice(b"tail"); }
    h
}

fn noise(n: usize, seed: u32) -> Vec<u8> {
    let mut x = seed.wrapping_mul(2654435761) | 1;
    (0..n).map(|_| { x ^= x << 13; x ^= x >> 17; x ^= x << 5; x as u8 }).collect()
}

/// 2 layers x 3 experts; F16 experts of 4x6 (gate_up) and 6x4 (down), an odd-sized I8 tensor (so there is padding),
/// and a tensor of an unknown type.
fn model(junk: bool) -> Vec<u8> {
    let mut t: Vec<(&str, Vec<u64>, u32, Vec<u8>)> = vec![
        ("output.weight", vec![4, 8], 0, noise(128, 1)),
        ("token_embd.weight", vec![4, 8], 0, noise(128, 2)),
        ("odd.bias", vec![5], 24, noise(5, 3)),
        ("mystery.weight", vec![7], 99, noise(40, 4)),
    ];
    for (l, names) in [["blk.0.attn_q.weight", "blk.0.ffn_gate_up_exps.weight", "blk.0.ffn_down_exps.weight"],
                       ["blk.1.attn_q.weight", "blk.1.ffn_gate_up_exps.weight", "blk.1.ffn_down_exps.weight"]].iter().enumerate() {
        let l = l as u32 * 10;
        t.push((names[0], vec![4, 4], 1, noise(32, 10 + l)));
        t.push((names[1], vec![4, 6, 3], 1, noise(4 * 6 * 3 * 2, 11 + l)));
        t.push((names[2], vec![6, 4, 3], 1, noise(6 * 4 * 3 * 2, 12 + l)));
    }
    gguf_file("toy", 3, &t, junk)
}

struct Dir(PathBuf);
impl Dir {
    fn new(name: &str) -> Dir {
        let d = std::env::temp_dir().join(format!("am2pack-test-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        Dir(d)
    }
    fn put(&self, name: &str, b: &[u8]) -> PathBuf {
        let p = self.0.join(name);
        std::fs::write(&p, b).unwrap();
        p
    }
}
impl Drop for Dir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn pack(d: &Dir, o: BuildOptions) -> PathBuf {
    let out = d.0.join("toy.am2aipac");
    build(&BuildOptions { out: out.clone(), ..o }, &mut |_| {}).unwrap();
    out
}

#[test]
fn builds_reads_and_unpacks_byte_for_byte() {
    let d = Dir::new("roundtrip");
    let (m, proj) = (model(true), gguf_file("clip", 0, &[("v.patch_embd.weight", vec![3, 3], 0, noise(36, 50)), ("mm.0.weight", vec![6], 1, noise(12, 51))], false));
    let profile = { // STRP v1 ranking (1,2) then (0,1)
        let mut p = b"STRP".to_vec();
        for v in [1u32, 2, 3, 2, 2] { p.extend_from_slice(&v.to_le_bytes()); }
        for (l, e) in [(1u16, 2u16), (0, 1)] { p.extend_from_slice(&l.to_le_bytes()); p.extend_from_slice(&e.to_le_bytes()); }
        p.extend_from_slice(&[0xFF; 24]);
        p
    };
    let out = pack(&d, BuildOptions {
        model: d.put("toy.gguf", &m), mmproj: Some(d.put("mmproj-toy.gguf", &proj)), imatrix: Some(d.put("toy.imatrix", b"importance")),
        profile: Some(d.put("toy.profile", &profile)), defaults: Some(d.put("defaults.json", br#"{"context":4096}"#)), ..Default::default() });
    let p = Pack::open(&out).unwrap();
    p.verify().unwrap();
    assert_eq!((p.index.manifest.architecture.as_str(), p.index.manifest.expert_count, p.index.manifest.expert_order.as_str()), ("toy", 3, "profile"));

    // the header section is the GGUF's header, and parses
    let msrc = p.source("model").unwrap();
    let h = gguf::parse(p.gguf_header(msrc).unwrap()).unwrap();
    assert_eq!(h.tensors.len(), 10);
    assert_eq!(&m[..h.data_start as usize], p.gguf_header(msrc).unwrap());

    // core holds the unsplit tensors, layers in order and the output last, each on a 64-byte boundary
    let core = p.index.sections.iter().find(|s| s.kind == kind::CORE).unwrap();
    let mut in_core: Vec<(u64, &str)> = p.index.tensors.iter().filter(|t| t.src == msrc).filter_map(|t| t.off.map(|o| (o, t.name.as_str()))).collect();
    in_core.sort();
    assert!(in_core.iter().all(|(o, _)| o % 64 == 0 && *o >= core.off && *o < core.off + core.len));
    assert_eq!(in_core.iter().map(|x| x.1).collect::<Vec<_>>(),
        ["token_embd.weight", "odd.bias", "mystery.weight", "blk.0.attn_q.weight", "blk.1.attn_q.weight", "output.weight"]);
    // the unknown type is copied by its extent: its 40 bytes and the padding up to the next tensor
    assert_eq!(p.tensor(msrc, "mystery.weight").unwrap().1.unwrap(), &[noise(40, 4), vec![0; 24]].concat()[..]);

    // each expert record holds that expert's slice of both stacked tensors; the profile's two come first
    assert_eq!(p.index.experts.iter().map(|r| (r.layer, r.expert)).collect::<Vec<_>>(), [(1, 2), (0, 1), (0, 0), (0, 2), (1, 0), (1, 1)]);
    let (gu, dn) = (noise(144, 21), noise(144, 22));
    let (r, parts) = p.expert(1, 2).unwrap();
    assert_eq!(r.off % 4096, 0);
    assert_eq!(parts.iter().map(|(t, b)| (t.name.as_str(), *b)).collect::<Vec<_>>(),
        [("blk.1.ffn_gate_up_exps.weight", &gu[96..144]), ("blk.1.ffn_down_exps.weight", &dn[96..144])]);
    assert!(p.tensor(msrc, "blk.1.ffn_down_exps.weight").unwrap().1.is_none());

    // the files kept whole
    assert_eq!(p.whole("imatrix").unwrap(), b"importance");
    assert_eq!(p.whole("profile").unwrap(), &profile[..]);
    // the non-zero padding is kept, and only that
    assert_eq!(p.index.sources[msrc].gaps.iter().map(|g| g.len).collect::<Vec<_>>(), [27, 4]);

    // the start of a source, without rebuilding all of it
    assert_eq!(p.source_head(msrc, 700).unwrap(), &m[..700]);
    assert_eq!(p.source_head(msrc, 1 << 20).unwrap(), m);

    // unpack: every source, byte for byte
    let back = d.0.join("back");
    let files = unpack(&p, &back).unwrap();
    assert_eq!(files.len(), 5);
    for (name, want) in [("toy.gguf", &m[..]), ("mmproj-toy.gguf", &proj[..]), ("toy.imatrix", b"importance"), ("toy.profile", &profile[..])] {
        assert_eq!(std::fs::read(back.join(name)).unwrap(), want, "{name}");
    }
    // and it does not overwrite
    assert!(unpack(&p, &back).is_err());
}

#[test]
fn a_flipped_byte_anywhere_is_caught() {
    let d = Dir::new("flip");
    let out = pack(&d, BuildOptions { model: d.put("toy.gguf", &model(false)), ..Default::default() });
    let good = std::fs::read(&out).unwrap();
    let p = Pack::open(&out).unwrap();
    p.verify().unwrap();
    let sec = |k: &str| p.index.sections.iter().find(|s| s.kind == k).unwrap().clone();
    let (core, experts) = (sec(kind::CORE), sec(kind::EXPERTS));
    let index_off = u64::from_le_bytes(good[16..24].try_into().unwrap()) as usize;
    let spots = [
        ("the magic", 3usize), ("the header's index offset", 17), ("the header's own checksum", 80), ("the header's padding", 2000),
        ("the padding before core", core.off as usize - 1), ("the middle of core", (core.off + core.len / 2) as usize),
        ("an expert record", (experts.off + 10) as usize), ("the padding inside the experts section", (experts.off + experts.len - 4096 + 200) as usize),
        ("the padding before the index", index_off - 1), ("the index", index_off + 20), ("the last byte", good.len() - 1),
    ];
    drop(p);
    for (what, at) in spots {
        let mut bad = good.clone();
        bad[at] ^= 0x40;
        std::fs::write(&out, &bad).unwrap();
        let caught = Pack::open(&out).and_then(|p| p.verify());
        assert!(caught.is_err(), "a flipped byte in {what} (offset {at}) was not caught");
    }
    // a truncated file, and one with bytes added
    for bad in [&good[..good.len() - 1], &[&good[..], b"x"].concat()[..]] {
        std::fs::write(&out, bad).unwrap();
        assert!(Pack::open(&out).is_err());
    }
}

#[test]
fn refuses_what_it_cannot_pack_faithfully() {
    let d = Dir::new("refuse");
    let run = |o: BuildOptions| build(&o, &mut |_| {});
    let m = d.put("toy.gguf", &model(false));
    // the wrong extension, and an existing file
    assert!(run(BuildOptions { model: m.clone(), out: d.0.join("toy.pack"), ..Default::default() }).is_err());
    let out = d.put("there.am2aipac", b"");
    assert!(run(BuildOptions { model: m.clone(), out, ..Default::default() }).unwrap_err().contains("exists"));
    // two tensors on the same bytes
    let mut over = gguf_file("toy", 0, &[("a.weight", vec![8], 0, noise(32, 1)), ("b.weight", vec![8], 0, noise(32, 2))], false);
    let at = over.windows(8).rposition(|w| w == 32u64.to_le_bytes()).unwrap();
    over[at..at + 8].copy_from_slice(&16u64.to_le_bytes());
    let e = run(BuildOptions { model: d.put("over.gguf", &over), out: d.0.join("over.am2aipac"), ..Default::default() }).unwrap_err();
    assert!(e.contains("overlaps"), "{e}");
    assert!(!Path::new(&d.0.join("over.am2aipac.part")).exists(), "a failed build leaves no partial file");
    // a tensor past the end of the file
    let cut = &model(false)[..model(false).len() - 40];
    assert!(run(BuildOptions { model: d.put("cut.gguf", cut), out: d.0.join("cut.am2aipac"), ..Default::default() }).is_err());
    // a profile for another model
    let mut prof = b"STRP".to_vec();
    for v in [1u32, 9, 3, 1, 1] { prof.extend_from_slice(&v.to_le_bytes()); }
    prof.extend_from_slice(&[7, 0, 0, 0]);
    let e = run(BuildOptions { model: m, profile: Some(d.put("p", &prof)), out: d.0.join("p.am2aipac"), ..Default::default() }).unwrap_err();
    assert!(e.contains("different model"), "{e}");
}

/// An index that is intact by its checksum but points outside the file is an error at open, not a panic later.
#[test]
fn a_crafted_index_is_refused() {
    use sha2::{Digest, Sha256};
    let d = Dir::new("craft");
    let out = pack(&d, BuildOptions { model: d.put("toy.gguf", &model(true)), ..Default::default() });
    let good = std::fs::read(&out).unwrap();
    let index_off = u64::from_le_bytes(good[16..24].try_into().unwrap()) as usize;
    let index: serde_json::Value = serde_json::from_slice(&good[index_off..]).unwrap();
    let edits: [(&str, fn(&mut serde_json::Value)); 5] = [
        ("a gap outside the file", |v| v["sources"][0]["gaps"][0]["off"] = (1u64 << 60).into()),
        ("a header section that does not exist", |v| v["sources"][0]["header"] = 99.into()),
        ("a part whose length overflows", |v| v["experts"][0]["parts"][0]["len"] = u64::MAX.into()),
        ("a tensor outside the file", |v| v["tensors"][0]["off"] = (1u64 << 60).into()),
        ("a source that is neither a GGUF nor whole", |v| { v["sources"][0].as_object_mut().unwrap().remove("header"); }),
    ];
    for (what, edit) in edits {
        let mut v = index.clone();
        edit(&mut v);
        let json = serde_json::to_vec(&v).unwrap();
        let mut bad = good[..index_off].to_vec();
        bad.extend_from_slice(&json);
        // a header that is right for the new index
        bad[24..32].copy_from_slice(&(json.len() as u64).to_le_bytes());
        let len = bad.len() as u64;
        bad[32..40].copy_from_slice(&len.to_le_bytes());
        bad[40..72].copy_from_slice(&Sha256::digest(&json));
        let own = Sha256::digest(&bad[..72]);
        bad[72..104].copy_from_slice(&own);
        std::fs::write(&out, &bad).unwrap();
        assert!(Pack::open(&out).is_err(), "{what} was accepted");
    }
}
