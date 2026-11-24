// SPDX-License-Identifier: AGPL-3.0-only
//! The reader: a pack memory-mapped, its tensors and expert records as slices, `verify`, and `unpack`.

use crate::{hex, writer, Index, Result, Tensor, ExpertRecord, HEADER_LEN, MAGIC, VERSION};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};

pub struct Pack {
    pub path: PathBuf,
    pub index: Index,
    mm: memmap2::Mmap,
    by_name: HashMap<(usize, String), usize>,
    by_expert: HashMap<(u32, u32), usize>,
}

/// A run of a source file and where the pack keeps it.
struct Piece {
    src_off: u64,
    off: u64,
    len: u64,
}

impl Pack {
    /// Opens a pack and checks its header and index (not the sections: that is [`Pack::verify`]).
    pub fn open(path: &Path) -> Result<Pack> {
        let p = path.display();
        let f = std::fs::File::open(path).map_err(|e| format!("{p}: {e}"))?;
        let mm = unsafe { memmap2::Mmap::map(&f) }.map_err(|e| format!("{p}: {e}"))?;
        if mm.len() < HEADER_LEN as usize || &mm[..8] != MAGIC {
            return Err(format!("{p}: not an am2aipac pack"));
        }
        let u32_at = |a: usize| u32::from_le_bytes(mm[a..a + 4].try_into().unwrap());
        let u64_at = |a: usize| u64::from_le_bytes(mm[a..a + 8].try_into().unwrap());
        let version = u32_at(8);
        if version > VERSION {
            return Err(format!("{p}: pack format version {version}; this reader knows up to {VERSION}"));
        }
        let (index_off, index_len, file_len) = (u64_at(16), u64_at(24), u64_at(32));
        if mm[..104] != writer::header(index_off, index_len, &mm[40..72], file_len)[..] || u32_at(12) as u64 != HEADER_LEN {
            return Err(format!("{p}: the pack's header is damaged"));
        }
        if file_len != mm.len() as u64 || index_off.checked_add(index_len) != Some(file_len) || index_off < HEADER_LEN {
            return Err(format!("{p}: the pack is {} bytes but its header says {file_len}: truncated or damaged", mm.len()));
        }
        let json = &mm[index_off as usize..];
        if Sha256::digest(json)[..] != mm[40..72] {
            return Err(format!("{p}: the pack's index is damaged"));
        }
        let index: Index = serde_json::from_slice(json).map_err(|e| format!("{p}: the pack's index does not parse: {e}"))?;
        let inside = |off: u64, len: u64| off >= HEADER_LEN && off.checked_add(len).is_some_and(|e| e <= index_off);
        for s in &index.sections {
            if !inside(s.off, s.len) {
                return Err(format!("{p}: section {} {} lies outside the file", s.kind, s.name));
            }
        }
        let sec_ok = |i: Option<usize>| i.is_none_or(|i| i < index.sections.len());
        for s in &index.sources {
            if !sec_ok(s.header) || !sec_ok(s.whole) || s.header.is_some() == s.whole.is_some() || s.gaps.iter().any(|g| !inside(g.off, g.len)) {
                return Err(format!("{p}: source {} has a bad entry", s.file));
            }
        }
        for (i, t) in index.tensors.iter().enumerate() {
            if t.src >= index.sources.len() || t.off.is_some_and(|o| !inside(o, t.len)) || t.off.is_none() == t.split.is_none() {
                return Err(format!("{p}: tensor {i} ({}) has a bad entry", t.name));
            }
        }
        for r in &index.experts {
            for part in &r.parts {
                let ok = index.tensors.get(part.tensor).is_some_and(|t| t.split.is_some_and(|n| r.expert < n && part.len.checked_mul(n as u64) == Some(t.len)));
                if !ok || !inside(part.off, part.len) || !inside(r.off, r.len) || part.off < r.off || part.off + part.len > r.off + r.len {
                    return Err(format!("{p}: the record of layer {} expert {} has a bad entry", r.layer, r.expert));
                }
            }
        }
        let by_name = index.tensors.iter().enumerate().map(|(i, t)| ((t.src, t.name.clone()), i)).collect();
        let by_expert = index.experts.iter().enumerate().map(|(i, r)| ((r.layer, r.expert), i)).collect();
        Ok(Pack { path: path.to_path_buf(), index, mm, by_name, by_expert })
    }

    /// The file's length in bytes.
    pub fn len(&self) -> u64 {
        self.mm.len() as u64
    }

    pub fn bytes(&self, off: u64, len: u64) -> &[u8] {
        &self.mm[off as usize..(off + len) as usize]
    }

    /// The index of the source with this role ("model", "mmproj", ...).
    pub fn source(&self, role: &str) -> Option<usize> {
        self.index.sources.iter().position(|s| s.role == role)
    }

    /// A GGUF source's header bytes (parse with [`crate::gguf::parse`]): settings, tokenizer, chat format.
    pub fn gguf_header(&self, src: usize) -> Option<&[u8]> {
        let s = self.index.sections.get(self.index.sources.get(src)?.header?)?;
        Some(self.bytes(s.off, s.len))
    }

    /// A file kept whole (the imatrix, the profile, the defaults).
    pub fn whole(&self, role: &str) -> Option<&[u8]> {
        let s = self.index.sections.get(self.index.sources.get(self.source(role)?)?.whole?)?;
        Some(self.bytes(s.off, s.len))
    }

    /// A tensor by source and name; its bytes are None when it is split over the expert records.
    pub fn tensor(&self, src: usize, name: &str) -> Option<(&Tensor, Option<&[u8]>)> {
        let t = &self.index.tensors[*self.by_name.get(&(src, name.to_string()))?];
        Some((t, t.off.map(|o| self.bytes(o, t.len))))
    }

    /// One expert's record: each stacked tensor of its layer, and this expert's slice of it.
    pub fn expert(&self, layer: u32, expert: u32) -> Option<(&ExpertRecord, Vec<(&Tensor, &[u8])>)> {
        let r = &self.index.experts[*self.by_expert.get(&(layer, expert))?];
        Some((r, r.parts.iter().map(|p| (&self.index.tensors[p.tensor], self.bytes(p.off, p.len))).collect()))
    }

    /// The runs a GGUF source is made of, in the source's order; what lies between them is zeros.
    fn pieces(&self, src: usize) -> Result<Vec<Piece>> {
        let s = &self.index.sources[src];
        let h = &self.index.sections[s.header.ok_or("not a GGUF source")?];
        let mut v = vec![Piece { src_off: 0, off: h.off, len: h.len }];
        v.extend(self.index.tensors.iter().filter(|t| t.src == src).filter_map(|t| t.off.map(|off| Piece { src_off: t.src_off, off, len: t.len })));
        for r in &self.index.experts {
            for p in &r.parts {
                let t = &self.index.tensors[p.tensor];
                if t.src == src {
                    v.push(Piece { src_off: t.src_off.saturating_add(r.expert as u64 * p.len), off: p.off, len: p.len });
                }
            }
        }
        v.extend(s.gaps.iter().map(|g| Piece { src_off: g.src_off, off: g.off, len: g.len }));
        v.sort_by_key(|p| p.src_off);
        let mut at = 0;
        for p in &v {
            if p.src_off < at || p.src_off.checked_add(p.len).is_none_or(|e| e > s.size) {
                return Err(format!("{}: the pack's pieces of {} overlap or run past its end", self.path.display(), s.file));
            }
            at = p.src_off + p.len;
        }
        Ok(v)
    }

    /// Feeds a source file's bytes, in order, to `sink`: the file as it was before packing.
    pub fn rebuild(&self, src: usize, sink: &mut dyn FnMut(&[u8]) -> Result<()>) -> Result<()> {
        let s = &self.index.sources[src];
        if let Some(w) = s.whole {
            let sec = &self.index.sections[w];
            return sink(self.bytes(sec.off, sec.len));
        }
        const Z: [u8; 4096] = [0; 4096];
        let zeros = |mut n: u64, sink: &mut dyn FnMut(&[u8]) -> Result<()>| -> Result<()> {
            while n > 0 {
                let k = n.min(4096);
                sink(&Z[..k as usize])?;
                n -= k;
            }
            Ok(())
        };
        let mut at = 0;
        for p in self.pieces(src)? {
            zeros(p.src_off - at, sink)?;
            for c in self.bytes(p.off, p.len).chunks(8 << 20) {
                sink(c)?;
            }
            at = p.src_off + p.len;
        }
        zeros(s.size - at, sink)
    }

    /// The first `n` bytes of a source file as it was before packing (fewer when the file is shorter).
    pub fn source_head(&self, src: usize, n: usize) -> Result<Vec<u8>> {
        let mut out = Vec::with_capacity(n);
        let r = self.rebuild(src, &mut |b| {
            out.extend_from_slice(&b[..b.len().min(n - out.len())]);
            if out.len() == n { Err(String::new()) } else { Ok(()) }
        });
        match r {
            Err(e) if !e.is_empty() => Err(e),
            _ => Ok(out),
        }
    }

    /// Every check: each section's checksum, zeros everywhere outside the sections, and each source rebuilt in
    /// memory to its recorded SHA-256. Returns one line per thing checked.
    pub fn verify(&self) -> Result<Vec<String>> {
        let p = self.path.display();
        let mut lines = Vec::new();
        if self.mm[104..HEADER_LEN as usize].iter().any(|b| *b != 0) {
            return Err(format!("{p}: the header's padding is not zero"));
        }
        let mut order: Vec<&crate::Section> = self.index.sections.iter().collect();
        order.sort_by_key(|s| s.off);
        let mut at = HEADER_LEN;
        let index_off = u64::from_le_bytes(self.mm[16..24].try_into().unwrap());
        for s in &order {
            if s.off < at {
                return Err(format!("{p}: sections overlap at {} {}", s.kind, s.name));
            }
            if self.bytes(at, s.off - at).iter().any(|b| *b != 0) {
                return Err(format!("{p}: the padding before {} {} is not zero", s.kind, s.name));
            }
            if hash(self.bytes(s.off, s.len)) != s.sha256 {
                return Err(format!("{p}: section {} {} is damaged (its checksum does not match)", s.kind, s.name));
            }
            lines.push(format!("section {} {}: {} bytes ok", s.kind, s.name, s.len));
            at = s.off + s.len;
        }
        if self.bytes(at, index_off - at).iter().any(|b| *b != 0) {
            return Err(format!("{p}: the padding before the index is not zero"));
        }
        for (i, s) in self.index.sources.iter().enumerate() {
            let mut h = Sha256::new();
            let mut n = 0u64;
            self.rebuild(i, &mut |b| { h.update(b); n += b.len() as u64; Ok(()) })?;
            if n != s.size || hex(&h.finalize()) != s.sha256 {
                return Err(format!("{p}: {} does not rebuild to the file it was packed from", s.file));
            }
            lines.push(format!("source {} {}: rebuilds to its {} bytes, sha256 {}", s.role, s.file, s.size, s.sha256));
        }
        Ok(lines)
    }
}

fn hash(b: &[u8]) -> String {
    let mut h = Sha256::new();
    for c in b.chunks(8 << 20) {
        h.update(c);
    }
    hex(&h.finalize())
}

/// Writes every source file back into `dir`, under its own name, and checks each against its recorded SHA-256.
/// Existing files are not overwritten.
pub fn unpack(pack: &Pack, dir: &Path) -> Result<Vec<PathBuf>> {
    std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let mut out = Vec::new();
    for (i, s) in pack.index.sources.iter().enumerate() {
        let name = Path::new(&s.file).file_name().ok_or_else(|| format!("the pack names a source with no file name: {:?}", s.file))?;
        let path = dir.join(name);
        let io = |e: std::io::Error| format!("{}: {e}", path.display());
        let f = std::fs::OpenOptions::new().write(true).create_new(true).open(&path).map_err(io)?;
        let mut w = std::io::BufWriter::with_capacity(8 << 20, f);
        let mut h = Sha256::new();
        let r = pack.rebuild(i, &mut |b| { h.update(b); w.write_all(b).map_err(io) }).and_then(|_| w.flush().map_err(io));
        let ok = r.is_ok() && hex(&h.finalize()) == s.sha256;
        if !ok {
            drop(w);
            let _ = std::fs::remove_file(&path);
            return Err(r.err().unwrap_or_else(|| format!("{}: does not match the file it was packed from", path.display())));
        }
        out.push(path);
    }
    Ok(out)
}
