// SPDX-License-Identifier: AGPL-3.0-only
//! A pack from GGUF files: every tensor's bytes are copied unchanged, only moved. The model's expert tensors are cut
//! into one record per (layer, expert); everything else goes into `core` in the order the layers run.

use crate::gguf::{self, Header};
use crate::{hex, kind, ExpertRecord, Gap, Index, Manifest, Part, Result, Source, Tensor, Writer};
use crate::{ALIGN_LARGE, ALIGN_RECORD, ALIGN_SECTION, ALIGN_TENSOR};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Default)]
pub struct BuildOptions {
    /// the model's GGUF (one file; a split GGUF is refused)
    pub model: PathBuf,
    /// the image projector beside it
    pub mmproj: Option<PathBuf>,
    /// an importance matrix, kept whole for re-quantizing
    pub imatrix: Option<PathBuf>,
    /// an expert profile (STRP version 1): the expert records are written in its order, hottest first
    pub profile: Option<PathBuf>,
    /// run defaults (a JSON object), kept whole
    pub defaults: Option<PathBuf>,
    pub out: PathBuf,
    /// the pack's name; the output file's stem when absent
    pub name: Option<String>,
}

#[derive(Debug, Clone)]
pub struct Summary {
    pub bytes: u64,
    pub tensors: usize,
    pub core_tensors: usize,
    pub split_tensors: usize,
    pub expert_records: usize,
    pub image_tensors: usize,
    pub expert_order: String,
}

struct Mapped {
    path: PathBuf,
    mm: memmap2::Mmap,
}

fn map(path: &Path) -> Result<Mapped> {
    let f = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mm = unsafe { memmap2::Mmap::map(&f) }.map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(Mapped { path: path.to_path_buf(), mm })
}

fn file_name(p: &Path) -> String {
    p.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default()
}

fn sha(b: &[u8]) -> String {
    let mut h = Sha256::new();
    for c in b.chunks(8 << 20) {
        h.update(c);
    }
    hex(&h.finalize())
}

/// Each tensor's (file offset, bytes), and the runs of the file that are neither header nor tensor. The pieces must
/// tile the file: a tensor past the end, or two that overlap, is refused.
fn layout(path: &Path, h: &Header, size: u64) -> Result<(Vec<(u64, u64)>, Vec<(u64, u64)>)> {
    let p = path.display();
    let mut by_off: Vec<usize> = (0..h.tensors.len()).collect();
    by_off.sort_by_key(|&i| h.tensors[i].offset);
    let mut spans = vec![(0u64, 0u64); h.tensors.len()];
    let mut gaps = Vec::new();
    let mut at = h.data_start;
    if at > size {
        return Err(format!("{p}: the header runs past the end of the file"));
    }
    for (k, &i) in by_off.iter().enumerate() {
        let t = &h.tensors[i];
        let start = h.data_start.checked_add(t.offset).ok_or_else(|| format!("{p}: {}: a bad offset", t.name))?;
        let next = by_off.get(k + 1).map(|&j| h.data_start + h.tensors[j].offset).unwrap_or(size);
        // an encoding this crate does not know is copied by its extent: up to the next tensor
        let len = t.expected_bytes().unwrap_or(next.saturating_sub(start));
        if start < at || start + len > size {
            return Err(format!("{p}: {} overlaps another tensor or runs past the end of the file", t.name));
        }
        if start > at {
            gaps.push((at, start - at));
        }
        spans[i] = (start, len);
        at = start + len;
    }
    if at < size {
        gaps.push((at, size - at));
    }
    Ok((spans, gaps))
}

/// `blk.<N>.` -> N
fn layer_of(name: &str) -> Option<u32> {
    name.strip_prefix("blk.")?.split('.').next()?.parse().ok()
}

/// (layer, expert) pairs of an expert profile, hottest first: an STRP version 1 file, or routing counts as text
/// ("layer expert count" on each line, as strata-poly keeps them), ranked by count and then by (layer, expert).
fn read_profile(b: &[u8], path: &Path) -> Result<Vec<(u32, u32)>> {
    if b.len() >= 4 && &b[..4] != b"STRP" {
        return read_counts(b, path);
    }
    let u = |a: usize| u32::from_le_bytes(b[a..a + 4].try_into().unwrap()) as usize;
    if b.len() < 24 || &b[..4] != b"STRP" || u(4) != 1 || b.len() < 24 + 4 * u(16) {
        return Err(format!("{}: not a version 1 expert profile", path.display()));
    }
    Ok((0..u(16)).map(|i| {
        let a = 24 + 4 * i;
        (u16::from_le_bytes([b[a], b[a + 1]]) as u32, u16::from_le_bytes([b[a + 2], b[a + 3]]) as u32)
    }).collect())
}

fn read_counts(b: &[u8], path: &Path) -> Result<Vec<(u32, u32)>> {
    let bad = || format!("{}: neither a version 1 expert profile nor \"layer expert count\" lines", path.display());
    let text = std::str::from_utf8(b).map_err(|_| bad())?;
    let mut rows: Vec<(u64, u32, u32)> = Vec::new();
    for line in text.lines().filter(|l| !l.starts_with('#') && !l.trim().is_empty()) {
        let f: Vec<&str> = line.split_whitespace().collect();
        match f.as_slice() {
            [l, e, c] => rows.push((c.parse().map_err(|_| bad())?, l.parse().map_err(|_| bad())?, e.parse().map_err(|_| bad())?)),
            _ => return Err(bad()),
        }
    }
    if rows.is_empty() {
        return Err(bad());
    }
    rows.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
    Ok(rows.into_iter().map(|(_, l, e)| (l, e)).collect())
}

/// Adds a GGUF source: its header section, its `Source` and its tensors (not yet placed). Returns the source's index.
fn add_gguf(w: &mut Writer, ix: &mut Index, role: &str, m: &Mapped, h: &Header, say: &mut dyn FnMut(&str)) -> Result<(usize, Vec<(u64, u64)>)> {
    let size = m.mm.len() as u64;
    let (spans, gaps) = layout(&m.path, h, size)?;
    say(&format!("{role}: hashing {} ({size} bytes)", file_name(&m.path)));
    let header = w.section(kind::GGUF_HEADER, role, ALIGN_SECTION, &m.mm[..h.data_start as usize])?;
    let src = ix.sources.len();
    ix.sources.push(Source { role: role.into(), file: file_name(&m.path), size, sha256: sha(&m.mm), header: Some(header), whole: None, gaps: Vec::new() });
    for (t, &(src_off, len)) in h.tensors.iter().zip(&spans) {
        ix.tensors.push(Tensor { src, name: t.name.clone(), ty: t.type_name(), dims: t.dims.clone(), src_off, len, off: None, split: None });
    }
    Ok((src, gaps))
}

/// Writes tensors `which` (indices into ix.tensors) one after another into the open section.
fn copy_tensors(w: &mut Writer, ix: &mut Index, m: &Mapped, which: &[usize]) -> Result<()> {
    for &i in which {
        let off = w.align(ALIGN_TENSOR)?;
        let t = &mut ix.tensors[i];
        w.write(&m.mm[t.src_off as usize..(t.src_off + t.len) as usize])?;
        t.off = Some(off);
    }
    Ok(())
}

pub fn build(o: &BuildOptions, say: &mut dyn FnMut(&str)) -> Result<Summary> {
    if o.out.extension().and_then(|e| e.to_str()) != Some("am2aipac") {
        return Err(format!("{}: a pack's name ends in .am2aipac", o.out.display()));
    }
    if o.out.exists() {
        return Err(format!("{} exists already", o.out.display()));
    }
    let model = map(&o.model)?;
    let mh = gguf::parse(&model.mm).map_err(|e| format!("{}: {e}", o.model.display()))?;
    if mh.u64("split.count").is_some_and(|n| n > 1) {
        return Err(format!("{} is one shard of a split GGUF: merge the shards into one file first", o.model.display()));
    }
    let proj = o.mmproj.as_deref().map(|p| -> Result<(Mapped, Header)> {
        let m = map(p)?;
        let h = gguf::parse(&m.mm).map_err(|e| format!("{}: {e}", p.display()))?;
        Ok((m, h))
    }).transpose()?;
    let arch = mh.architecture().unwrap_or("").to_string();
    let expert_count = mh.u64(&format!("{arch}.expert_count")).unwrap_or(0);

    let mut w = Writer::create(&o.out)?;
    let mut ix = Index::default();
    let (msrc, mgaps) = add_gguf(&mut w, &mut ix, "model", &model, &mh, say)?;
    let n_model = ix.tensors.len();

    // which tensors are stacked experts: 3-D, named blk.N.*_exps*, the last dimension is the expert count, and one
    // expert's slice is a whole number of bytes and of quantization blocks; in a layer where they disagree on the
    // count, none is split
    let mut layers: std::collections::BTreeMap<u32, Vec<usize>> = Default::default();
    for (i, t) in ix.tensors.iter().enumerate() {
        let n = *t.dims.last().unwrap_or(&0);
        let per = t.dims.iter().rev().skip(1).product::<u64>();
        let whole_blocks = gguf::block_geometry(&t.ty).is_some_and(|(be, _)| per % be == 0);
        if let (Some(l), true) = (layer_of(&t.name), t.dims.len() == 3 && t.name.contains("_exps") && n > 1 && n == expert_count && whole_blocks && t.len % n == 0) {
            layers.entry(l).or_default().push(i);
        }
    }
    for (_, v) in layers.iter() {
        for &i in v {
            ix.tensors[i].split = Some(expert_count as u32);
        }
    }

    // core: the tensors before the layers, the layers in order, the output last
    let mut core: Vec<usize> = (0..n_model).filter(|&i| ix.tensors[i].split.is_none()).collect();
    core.sort_by_key(|&i| {
        let n = &ix.tensors[i].name;
        match layer_of(n) { Some(l) => 1 + l as u64, None if n.starts_with("output") => u64::MAX, None => 0 }
    });
    say(&format!("core: {} tensors", core.len()));
    w.begin(kind::CORE, "model", ALIGN_LARGE)?;
    copy_tensors(&mut w, &mut ix, &model, &core)?;
    w.end()?;

    // experts: the profile's order first, then every other (layer, expert) by layer
    let mut order: Vec<(u32, u32)> = Vec::new();
    let mut expert_order = "layer";
    let profile = o.profile.as_deref().map(|p| std::fs::read(p).map(|b| (p, b)).map_err(|e| format!("{}: {e}", p.display()))).transpose()?;
    if let Some((p, b)) = &profile {
        let mut seen = std::collections::HashSet::new();
        for (l, e) in read_profile(b, p)? {
            if !layers.contains_key(&l) || e as u64 >= expert_count {
                return Err(format!("{}: it ranks layer {l} expert {e}, which this model does not have: a profile for a different model", p.display()));
            }
            if seen.insert((l, e)) {
                order.push((l, e));
            }
        }
        expert_order = "profile";
        let rest: Vec<(u32, u32)> = layers.keys().flat_map(|&l| (0..expert_count as u32).map(move |e| (l, e))).filter(|k| !seen.contains(k)).collect();
        order.extend(rest);
    } else {
        order.extend(layers.keys().flat_map(|&l| (0..expert_count as u32).map(move |e| (l, e))));
    }
    if !order.is_empty() {
        say(&format!("experts: {} records ({} layers x {expert_count}), in {expert_order} order", order.len(), layers.len()));
        w.begin(kind::EXPERTS, "model", ALIGN_LARGE)?;
        for (l, e) in order {
            let off = w.align(ALIGN_RECORD)?;
            let mut parts = Vec::new();
            for &ti in &layers[&l] {
                let poff = w.align(ALIGN_TENSOR)?;
                let t = &ix.tensors[ti];
                let plen = t.len / expert_count;
                let at = (t.src_off + e as u64 * plen) as usize;
                w.write(&model.mm[at..at + plen as usize])?;
                parts.push(Part { tensor: ti, off: poff, len: plen });
            }
            ix.experts.push(ExpertRecord { layer: l, expert: e, off, len: w.pos() - off, parts });
        }
        w.end()?;
    }

    // the image projector
    let mut gaps = vec![(msrc, &model, mgaps)];
    let mut image_tensors = 0;
    if let Some((pm, ph)) = &proj {
        let (psrc, pgaps) = add_gguf(&mut w, &mut ix, "mmproj", pm, ph, say)?;
        let which: Vec<usize> = (n_model..ix.tensors.len()).collect();
        image_tensors = which.len();
        say(&format!("image: {image_tensors} tensors"));
        w.begin(kind::IMAGE, "mmproj", ALIGN_LARGE)?;
        copy_tensors(&mut w, &mut ix, pm, &which)?;
        w.end()?;
        gaps.push((psrc, pm, pgaps));
    }

    // padding that is not zero is kept, so that unpack gives the same bytes back
    let nonzero: Vec<(usize, &Mapped, u64, u64)> = gaps.iter().flat_map(|(s, m, g)| g.iter().map(move |&(a, n)| (*s, *m, a, n)))
        .filter(|(_, m, a, n)| m.mm[*a as usize..(*a + *n) as usize].iter().any(|b| *b != 0)).collect();
    if !nonzero.is_empty() {
        w.begin(kind::GAPS, "", ALIGN_SECTION)?;
        for (s, m, a, n) in nonzero {
            ix.sources[s].gaps.push(Gap { src_off: a, len: n, off: w.pos() });
            w.write(&m.mm[a as usize..(a + n) as usize])?;
        }
        w.end()?;
    }

    // files kept whole
    for (role, k, path) in [("imatrix", kind::IMATRIX, &o.imatrix), ("profile", kind::PROFILE, &o.profile), ("defaults", kind::DEFAULTS, &o.defaults)] {
        let Some(path) = path else { continue };
        let b = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        if role == "defaults" && !serde_json::from_slice::<serde_json::Value>(&b).is_ok_and(|v| v.is_object()) {
            return Err(format!("{}: the run defaults must be a JSON object", path.display()));
        }
        let sec = w.section(k, &file_name(path), ALIGN_SECTION, &b)?;
        ix.sources.push(Source { role: role.into(), file: file_name(path), size: b.len() as u64, sha256: sha(&b), header: None, whole: Some(sec), gaps: Vec::new() });
    }

    let mut extra = serde_json::Map::new();
    for (key, k) in [("model_name", "general.name"), ("model_license", "general.license"), ("file_type", "general.file_type")] {
        match mh.kv.get(k) {
            Some(gguf::Value::S(s)) => { extra.insert(key.into(), s.clone().into()); }
            Some(gguf::Value::U(u)) => { extra.insert(key.into(), (*u).into()); }
            _ => {}
        }
    }
    ix.manifest = Manifest {
        name: o.name.clone().unwrap_or_else(|| o.out.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default()),
        architecture: arch,
        created_by: format!("am2pack {}", env!("CARGO_PKG_VERSION")),
        expert_order: expert_order.into(),
        expert_count: if ix.experts.is_empty() { 0 } else { expert_count as u32 },
        extra,
    };
    let (tensors, core_tensors, split_tensors, expert_records) = (ix.tensors.len(), core.len(), n_model - core.len(), ix.experts.len());
    say("writing the index");
    let bytes = w.finish(&mut ix)?;
    Ok(Summary { bytes, tensors, core_tensors, split_tensors, expert_records, image_tensors, expert_order: expert_order.into() })
}

#[cfg(test)]
mod profile_tests {
    use super::*;

    #[test]
    fn routing_counts_rank_by_count_then_place() {
        let p = Path::new("counts.txt");
        let got = read_profile(b"# layer expert count\n0 0 5\n0 1 9\n1 0 9\n1 1 0\n", p).unwrap();
        assert_eq!(got, vec![(0, 1), (1, 0), (0, 0), (1, 1)]);
        assert!(read_profile(b"0 1\n", p).is_err(), "a short line is refused");
        assert!(read_profile(b"not a profile", p).is_err());
    }
}
