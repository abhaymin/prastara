//! Strata's placement inputs for this engine: a pack (`--pack`, written by `am2strata-run iq-pack` / PolyStrata's
//! tools/iq_pack.py), the expert loadings (`--expert-profile`, PolyStrata's STRP v1 ranked list of (layer, expert)),
//! and the residency they drive (`--resident-experts`).
//!
//! - The pack's dense.bin holds the float tensors (norms, routers, SSM parameters) in the forms strata's engines read;
//!   they replace the GGUF's copies, so this engine computes with exactly what the CUDA engine uploads. Its
//!   native_experts.txt names, per MoE layer, the GGUF offsets of the gate/up/down expert tensors: a pack built from a
//!   different file is refused instead of read.
//! - The profile ranks experts hottest first. The engine counts how often each expert is routed to and writes a
//!   learned profile (`--expert-profile-save`) in the same format, ranked as PolyStrata's rank_learned_profile does:
//!   by heat, then by the prior profile's rank.
//! - Residency: the hottest experts within `--expert-ram-gib` are paged in at load and kept (mlock when the memlock limit
//!   allows it, else MADV_WILLNEED), the others are marked MADV_COLD (Linux) so that under memory pressure the kernel evicts them
//!   first: a model larger than the free RAM then reads only its cold experts from the SSD.

use crate::model::{Block, Model, Moe, QMat};
use std::collections::HashMap;
use std::io::Write;
use std::path::Path;
use std::sync::atomic::Ordering;

/// A ranked expert profile: (layer, expert) pairs, hottest first.
#[derive(Debug)]
pub struct Profile {
    pub n_layers: usize,
    pub n_expert: usize,
    pub ranked: Vec<(u16, u16)>,
}

impl Profile {
    /// Reads a version 1 STRP file (expert_cache.cpp read_expert_profile), checked against the model's shape.
    pub fn read(path: &Path, n_layers: usize, n_expert: usize) -> Result<Profile, String> {
        let b = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        if b.len() < 24 || &b[..4] != b"STRP" {
            return Err(format!("{} is not an expert profile (no STRP header)", path.display()));
        }
        let h = |i: usize| u32::from_le_bytes(b[4 + 4 * i..8 + 4 * i].try_into().unwrap()) as usize;
        let (version, nl, ne, slots, n) = (h(0), h(1), h(2), h(3), h(4));
        if version != 1 {
            return Err(format!("{} is a version {version} profile; this engine reads version 1", path.display()));
        }
        if nl != n_layers || ne != n_expert {
            return Err(format!("{} is {nl}x{ne} but this model is {n_layers}x{n_expert}: a profile for a different model", path.display()));
        }
        if n > slots || b.len() < 24 + 4 * n {
            return Err(format!("{}: the ranked list is truncated or longer than its slots", path.display()));
        }
        let mut ranked = Vec::with_capacity(n);
        for i in 0..n {
            let at = 24 + 4 * i;
            let (l, e) = (u16::from_le_bytes([b[at], b[at + 1]]), u16::from_le_bytes([b[at + 2], b[at + 3]]));
            if l as usize >= nl || e as usize >= ne {
                return Err(format!("{}: pair {i} is (layer {l}, expert {e}), out of range", path.display()));
            }
            ranked.push((l, e));
        }
        Ok(Profile { n_layers, n_expert, ranked })
    }

    /// Writes it as write_expert_profile does: the header, the pairs, then the rank of every (layer, expert) (-1 when
    /// unranked); through a temporary file, so a reader never sees half of one.
    pub fn write(&self, path: &Path) -> Result<(), String> {
        if self.n_layers > 65535 || self.n_expert > 65535 {
            return Err("the model's layout does not fit the profile format".into());
        }
        let mut table = vec![-1i32; self.n_layers * self.n_expert];
        let mut out = Vec::with_capacity(24 + self.ranked.len() * 4 + table.len() * 4);
        out.extend_from_slice(b"STRP");
        for v in [1, self.n_layers, self.n_expert, self.ranked.len(), self.ranked.len()] {
            out.extend_from_slice(&(v as u32).to_le_bytes());
        }
        for (r, &(l, e)) in self.ranked.iter().enumerate() {
            table[l as usize * self.n_expert + e as usize] = r as i32;
            out.extend_from_slice(&l.to_le_bytes());
            out.extend_from_slice(&e.to_le_bytes());
        }
        for v in table {
            out.extend_from_slice(&v.to_le_bytes());
        }
        let tmp = path.with_extension("tmp");
        let mut f = std::fs::File::create(&tmp).map_err(|e| format!("{}: {e}", tmp.display()))?;
        f.write_all(&out).and_then(|_| f.sync_all()).map_err(|e| format!("{}: {e}", tmp.display()))?;
        std::fs::rename(&tmp, path).map_err(|e| format!("{}: {e}", path.display()))
    }

    /// The learned ranking from routing counts (rank_learned_profile without the resident tier): by heat, then the
    /// prior profile's rank, then (layer, expert). Only experts that exist in their layer are listed.
    pub fn learned(model: &Model, prior: Option<&Profile>) -> Profile {
        let ne = model.profile_experts;
        let nl = model.profile_layers();
        let mut prior_rank = vec![usize::MAX; nl * ne];
        if let Some(p) = prior {
            for (r, &(l, e)) in p.ranked.iter().enumerate() {
                let i = l as usize * ne + e as usize;
                if prior_rank[i] == usize::MAX {
                    prior_rank[i] = r;
                }
            }
        }
        let mut all: Vec<usize> = model.expert_layers().0.into_iter().flat_map(|(l, n)| (0..n).map(move |e| l * ne + e)).collect();
        let heat = |i: usize| model.heat[i].load(Ordering::Relaxed);
        all.sort_by(|&a, &b| heat(b).cmp(&heat(a)).then(prior_rank[a].cmp(&prior_rank[b])).then(a.cmp(&b)));
        Profile { n_layers: nl, n_expert: ne, ranked: all.into_iter().map(|i| ((i / ne) as u16, (i % ne) as u16)).collect() }
    }
}

/// The pack's index (index.txt v3) and its native experts (native_experts.txt v3 / v4, or am2strata's p1).
pub struct Pack {
    pub dir: std::path::PathBuf,
    /// float tensors stored in dense.bin: name -> (kind, offset, bytes)
    floats: HashMap<String, (u32, u64, u64)>,
    /// per MoE layer: (gate, up, down) absolute offsets in the GGUF named by the header (0 when the role is absent)
    experts: HashMap<usize, [u64; 3]>,
    n_native: usize,
}

impl Pack {
    pub fn open(dir: &Path) -> Result<Pack, String> {
        let read = |n: &str| std::fs::read_to_string(dir.join(n)).map_err(|e| format!("pack {}: {n}: {e}", dir.display()));
        let index = read("index.txt")?;
        if !index.starts_with("# strata pack index v3") {
            return Err(format!("pack {}: index.txt is not a v3 strata pack index", dir.display()));
        }
        let mut floats = HashMap::new();
        let mut n_native = 0;
        for line in index.lines().filter(|l| !l.starts_with('#') && !l.trim().is_empty()) {
            let f: Vec<&str> = line.split_whitespace().collect();
            if f.len() < 12 {
                return Err(format!("pack {}: a short index row: {line}", dir.display()));
            }
            let kind: u32 = f[2].parse().map_err(|_| format!("pack: bad kind in {line}"))?;
            if kind == 0 {
                n_native += 1;
                continue;
            }
            let num = |i: usize| f[i].parse::<u64>().map_err(|_| format!("pack: bad number in {line}"));
            floats.insert(f[0].to_string(), (kind, num(3)?, num(4)?));
        }
        let mut experts = HashMap::new();
        if let Ok(text) = read("native_experts.txt") {
            // by its header: strata's v3 / v4 and am2strata's p1 (per-layer pruned, an n_expert column); Strata 0.2's v5
            // is Hadamard-folded experts, which only Strata 0.2's engine computes - refused, never misread
            let head = text.lines().next().unwrap_or("");
            if head.starts_with("# strata native experts v5:") {
                return Err(format!("pack {}: native_experts.txt v5 holds Hadamard-folded experts (Strata 0.2), which this \
                                    engine does not compute", dir.display()));
            }
            if !(head.starts_with("# strata native experts v3:") || head.starts_with("# strata native experts v4:")
                 || head.starts_with("# am2strata native experts p1:")) {
                return Err(format!("pack {}: native_experts.txt has a header this engine does not know: {:?}", dir.display(),
                                   head.chars().take(48).collect::<String>()));
            }
            for line in text.lines().filter(|l| !l.starts_with('#') && !l.trim().is_empty()) {
                let f: Vec<&str> = line.split_whitespace().collect();
                if f.len() >= 8 {
                    let n = |i: usize| f[i].parse::<u64>().unwrap_or(0);
                    experts.insert(n(0) as usize, [n(5), n(6), n(7)]);
                }
            }
        }
        Ok(Pack { dir: dir.to_path_buf(), floats, experts, n_native })
    }

    /// Checks the pack against the GGUF's tensors (its expert offsets must be this file's), then swaps in the float
    /// tensors from dense.bin in the pack's forms. Returns how many tensors now come from the pack.
    pub fn apply(&self, tensors: &mut HashMap<String, QMat>, offsets: &HashMap<String, u64>) -> Result<usize, String> {
        for (&l, want) in &self.experts {
            for (i, role) in ["gate", "up", "down"].iter().enumerate() {
                let name = format!("blk.{l}.ffn_{role}_exps.weight");
                match offsets.get(&name) {
                    Some(&off) if want[i] != 0 && want[i] != off => {
                        return Err(format!("pack {}: {name} is at {} in the pack but at {off} in this GGUF: the pack was built \
                                            from a different file", self.dir.display(), want[i]));
                    }
                    None if want[i] != 0 && i > 0 => {
                        return Err(format!("pack {}: {name} is in the pack but not in this GGUF: a pack for a different model",
                                           self.dir.display()));
                    }
                    _ => {}
                }
            }
        }
        let f = std::fs::File::open(self.dir.join("dense.bin")).map_err(|e| format!("pack: dense.bin: {e}"))?;
        let mm: &'static memmap2::Mmap = Box::leak(Box::new(unsafe { memmap2::Mmap::map(&f) }.map_err(|e| e.to_string())?));
        let mut n = 0;
        for (name, &(kind, off, bytes)) in &self.floats {
            let Some(q) = tensors.get_mut(name) else {
                return Err(format!("pack {}: {name} is not in this GGUF: a pack for a different model", self.dir.display()));
            };
            let end = (off + bytes) as usize;
            if end > mm.len() {
                return Err(format!("pack: {name} lies beyond dense.bin"));
            }
            let raw: &'static [u8] = &mm[off as usize..end];
            // kinds: 2 F32, 3 F32 that the loader rounds to F16 (done here, as the CUDA engine's loader does), 4 BF16, 5 F16
            *q = match kind {
                2 => q.with_data(raw, "F32")?,
                3 => {
                    let h: Vec<u8> = raw.chunks_exact(4)
                        .flat_map(|c| strata_pack::num::f32_to_f16(f32::from_le_bytes(c.try_into().unwrap())).to_le_bytes()).collect();
                    q.with_data(Box::leak(h.into_boxed_slice()), "F16")?
                }
                4 => q.with_data(raw, "BF16")?,
                5 => q.with_data(raw, "F16")?,
                k => return Err(format!("pack: {name} has an unknown kind {k}")),
            };
            n += 1;
        }
        Ok(n)
    }

    pub fn n_native(&self) -> usize {
        self.n_native
    }
}

/// What residency did: experts kept in RAM (and how), and the rest marked cold.
pub struct Residency {
    pub resident: Vec<bool>,
    pub n_resident: usize,
    pub bytes: u64,
    pub locked: bool,
}

fn expert_ranges(m: &Moe, e: usize) -> Vec<&'static [u8]> {
    let mut v = vec![];
    if let Some(g) = &m.gate_exps {
        v.push(g.0[e].data);
    }
    v.push(m.up_exps.0[e].data);
    v.push(m.down_exps.0[e].data);
    v
}

#[cfg(unix)]
fn advise(b: &[u8], advice: libc::c_int) -> bool {
    // madvise wants page-aligned starts: round the start down, the length up
    let page = 4096usize;
    let start = b.as_ptr() as usize & !(page - 1);
    let len = (b.as_ptr() as usize + b.len() - start).div_ceil(page) * page;
    unsafe { libc::madvise(start as *mut libc::c_void, len, advice) == 0 }
}

// Residency is a hint: without these, the operating system's page cache decides and the model still runs.
#[cfg(unix)]
pub(crate) fn pin(b: &[u8]) -> bool {
    unsafe { libc::mlock(b.as_ptr() as *const libc::c_void, b.len()) == 0 }
}

#[cfg(not(unix))]
pub(crate) fn pin(_: &[u8]) -> bool {
    false
}

/// A run of the mapped file asked from the disk ahead of its use (the reads start and this returns).
pub fn read_ahead(b: &[u8]) {
    wanted(b);
}

/// A run of the mapped file read from the disk ahead of its use by several threads at once: an NVMe SSD answers many
/// reads in parallel, and one thread faulting pages in as it copies them reads far below the disk's speed (measured
/// cold on Inkling 2 Mini: 0.9 GiB/s of the WD SN580's 2.8). The caller stores in the returned counter the address it
/// has consumed up to (the uploads go through the run in order, to its end): the readers stay within a window ahead of
/// it and skip what is behind it. Returns at once.
#[cfg_attr(not(feature = "gpu"), allow(dead_code))]
pub fn read_ahead_run(b: &'static [u8]) -> std::sync::Arc<std::sync::atomic::AtomicUsize> {
    use std::sync::atomic::AtomicUsize;
    const CHUNK: usize = 16 << 20;
    const THREADS: usize = 4;
    // how far ahead of the consumer the readers go: enough to keep the disk busy, without mapping the whole run at
    // once when it is already in the page cache (that showed as 16 GiB of resident memory)
    const WINDOW: usize = 2 << 30;
    let progress = std::sync::Arc::new(AtomicUsize::new(b.as_ptr() as usize));
    wanted(b);
    let page = 4096usize;
    let lo = (b.as_ptr() as usize) & !(page - 1);
    let hi = b.as_ptr() as usize + b.len();
    let next = std::sync::Arc::new(AtomicUsize::new(0));
    let n = (hi - lo).div_ceil(CHUNK);
    for _ in 0..THREADS.min(n) {
        let (next, progress) = (next.clone(), progress.clone());
        std::thread::spawn(move || loop {
            let i = next.fetch_add(1, Ordering::Relaxed);
            if i >= n {
                break;
            }
            let (a, z) = (lo + i * CHUNK, (lo + (i + 1) * CHUNK).min(hi));
            while a > progress.load(Ordering::Relaxed) + WINDOW {
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            if z <= progress.load(Ordering::Relaxed) {
                continue;
            }
            // read into the page cache and map: one call per chunk where the kernel has it (Linux 5.14), else a
            // byte of every page
            #[cfg(target_os = "linux")]
            let done = unsafe { libc::madvise(a as *mut libc::c_void, z - a, libc::MADV_POPULATE_READ) == 0 };
            #[cfg(not(target_os = "linux"))]
            let done = false;
            if !done {
                let mut sum = 0u8;
                for p in (a..z).step_by(page) {
                    sum = sum.wrapping_add(unsafe { std::ptr::read_volatile(p as *const u8) });
                }
                std::hint::black_box(sum);
            }
        });
    }
    progress
}

/// The model files this process has mapped: (start, end) addresses. `released` acts only inside them.
static FILE_MAPS: std::sync::Mutex<Vec<(usize, usize)>> = std::sync::Mutex::new(Vec::new());

/// Note a mapped model file (the loaders call this), so `released` knows its pages are the file's.
pub fn mapped_file(b: &[u8]) {
    FILE_MAPS.lock().unwrap().push((b.as_ptr() as usize, b.as_ptr() as usize + b.len()));
}

/// A run of a mapped model file this process no longer reads (a device holds the matrix): its pages leave this
/// process's memory. They stay in the page cache while there is room, so the next start is still warm (asking them
/// paged out instead made the next load read the disk again: 9.6 s became 15 s); read again, they come back.
/// Memory that is not a mapped file - a matrix the RAM tier copied - is left alone: this advice would zero it.
#[cfg_attr(not(feature = "gpu"), allow(dead_code))]
pub fn released(_b: &[u8]) {
    #[cfg(unix)]
    {
        // whole pages inside the run only: a page shared with a neighbour stays
        let page = 4096usize;
        let start = (_b.as_ptr() as usize).div_ceil(page) * page;
        let end = (_b.as_ptr() as usize + _b.len()) / page * page;
        if end > start && FILE_MAPS.lock().unwrap().iter().any(|&(lo, hi)| start >= lo && end <= hi) {
            unsafe { libc::madvise(start as *mut libc::c_void, end - start, libc::MADV_DONTNEED) };
        }
    }
}

/// One expert's matrices asked for in the background (MADV_WILLNEED starts the reads and returns).
pub fn prefetch_expert(m: &Moe, e: usize) {
    expert_ranges(m, e).iter().for_each(|r| wanted(r));
}

/// The same for one matrix, unless a device holds it (its file pages are not needed then).
pub fn wanted_off_device(q: &crate::model::QMat) {
    #[cfg(feature = "gpu")]
    if q.gpu.is_some() {
        return;
    }
    wanted(q.data);
}

fn wanted(_b: &[u8]) {
    #[cfg(unix)]
    advise(_b, libc::MADV_WILLNEED);
}

// Not MADV_RANDOM: an expert is megabytes in one piece, and without readahead it is read 4 KiB per fault (measured
// on GLM-4.7-Flash in 10 GiB: 5.8 M major faults instead of 0.2 M, generation 2.4 tok/s instead of 4.4).
// macOS has no MADV_COLD equivalent: there the cold experts get no hint, and the page cache decides alone.
fn cold(_b: &[u8]) {
    #[cfg(target_os = "linux")]
    advise(_b, libc::MADV_COLD);
}

/// Keeps the hottest experts (profile order, else layer order) within `budget` bytes in RAM; marks the rest cold.
pub fn place(model: &Model, profile: Option<&Profile>, budget: u64) -> Residency {
    let ne = model.profile_experts;
    let moes: HashMap<usize, &Moe> = model.moes().map(|m| (m.layer, m)).collect();
    let mut order: Vec<(usize, usize)> = match profile {
        Some(p) => p.ranked.iter().map(|&(l, e)| (l as usize, e as usize)).collect(),
        None => vec![],
    };
    // experts the profile does not rank follow, layer by layer
    let mut seen = vec![false; model.layers.len() * ne];
    order.iter().for_each(|&(l, e)| seen[l * ne + e] = true);
    for m in model.moes() {
        for e in 0..m.n_expert {
            if !seen[m.layer * ne + e] {
                order.push((m.layer, e));
            }
        }
    }
    let mut resident = vec![false; model.layers.len() * ne];
    let (mut bytes, mut n_resident, mut locked) = (0u64, 0usize, true);
    for &(l, e) in &order {
        let Some(m) = moes.get(&l) else { continue };
        if e >= m.n_expert {
            continue;
        }
        let ranges = expert_ranges(m, e);
        let size: u64 = ranges.iter().map(|r| r.len() as u64).sum();
        if bytes + size > budget {
            break;
        }
        for r in &ranges {
            locked = locked && pin(r);
            if !locked {
                wanted(r);
            }
        }
        resident[l * ne + e] = true;
        bytes += size;
        n_resident += 1;
    }
    // the cold experts: evicted first under memory pressure, and read without readahead
    for m in model.moes() {
        for e in 0..m.n_expert {
            if !resident[m.layer * ne + e] {
                expert_ranges(m, e).iter().for_each(|r| cold(r));
            }
        }
    }
    Residency { resident, n_resident, bytes, locked: locked && n_resident > 0 }
}

/// The experts' placement order and what it came from: the profile's ranking, else a model pack's own record order
/// ("file": each tier is then one run of the file), then every expert not listed, layer by layer.
pub fn expert_order(model: &Model, profile: Option<&Profile>) -> (Vec<(usize, usize)>, &'static str) {
    let (mut order, from): (Vec<(usize, usize)>, _) = match profile {
        Some(p) => (p.ranked.iter().map(|&(l, e)| (l as usize, e as usize)).collect(), "profile"),
        None if !model.file_order.is_empty() => (model.file_order.clone(), "file"),
        None => (vec![], "layer"),
    };
    let ne = model.profile_experts.max(1);
    let (n_exp, n_layers) = model.expert_layers();
    let mut seen = vec![false; n_layers * ne];
    order.retain(|&(l, e)| l * ne + e < seen.len() && !std::mem::replace(&mut seen[l * ne + e], true));
    for (l, n) in n_exp {
        for e in 0..n {
            if !seen[l * ne + e] {
                order.push((l, e));
            }
        }
    }
    (order, from)
}

/// Which experts the RAM tier takes: walking `sizes` (each expert's bytes that are not on a device, in placement
/// order), every expert that still fits `budget`, stopping at the first that does not, so the tier is one run of the
/// order. An expert wholly on a device (0 bytes) is passed over.
pub fn ram_plan(sizes: &[u64], budget: u64) -> Vec<bool> {
    let mut left = budget;
    let mut open = true;
    sizes.iter().map(|&b| {
        if b == 0 || !open {
            return false;
        }
        if b > left {
            open = false;
            return false;
        }
        left -= b;
        true
    }).collect()
}

/// The RAM tier, after the devices took theirs (any model; `place` above is the older path for a GGUF's MoE blocks):
/// the matrices every token uses that are on no device, then the experts on no device in placement order within
/// `budget`, are kept in RAM - locked where the memlock limit allows, else copied into memory of our own, because the
/// kernel drops file pages under pressure whatever the advice. What follows stays on the SSD, marked cold; the file
/// pages of what a device holds are marked cold too. From a pack the tier is one run of the file, asked from the
/// disk in one go before the copies. -> (what was placed, a summary for INFO).
pub fn ram_tier(model: &mut Model, profile: Option<&Profile>, budget: u64) -> (Residency, String) {
    let t0 = std::time::Instant::now();
    let ne = model.profile_experts.max(1);
    let n_slots = model.heat.len();
    let (order, from) = expert_order(model, profile);
    let file_map = model.file_map.filter(|_| from == "file");
    let (every, mut experts) = model.matrices_mut();
    let off_device = |q: &QMat| !on_device(q);
    let mut keep: Vec<&mut QMat> = Vec::new();
    let mut always = 0u64;
    for q in every {
        if off_device(q) {
            always += q.data.len() as u64;
            keep.push(q);
        } else {
            cold(q.data);
        }
    }
    let n_always = keep.len();
    let groups: Vec<((usize, usize), Vec<&mut QMat>)> = order.iter().filter_map(|k| experts.remove(k).map(|g| (*k, g))).collect();
    let sizes: Vec<u64> = groups.iter().map(|(_, g)| g.iter().filter(|q| off_device(q)).map(|q| q.data.len() as u64).sum()).collect();
    let take = ram_plan(&sizes, budget.saturating_sub(always));
    let mut resident = vec![false; n_slots];
    let (mut bytes, mut n_resident, mut n_cold, mut cold_bytes) = (0u64, 0usize, 0usize, 0u64);
    for (((l, e), group), (&size, &t)) in groups.into_iter().zip(sizes.iter().zip(&take)) {
        if t {
            if let Some(r) = resident.get_mut(l * ne + e) {
                *r = true;
            }
            n_resident += 1;
            bytes += size;
        } else if size > 0 {
            n_cold += 1;
            cold_bytes += size;
        }
        for q in group {
            if t && off_device(q) {
                keep.push(q);
            } else {
                cold(q.data);
            }
        }
    }
    // one read-ahead over the tier's run of the file, then lock or copy matrix by matrix
    if let Some(map) = file_map {
        let base = map.as_ptr() as usize;
        let spans: Vec<(usize, usize)> = keep[n_always..].iter().map(|q| (q.data.as_ptr() as usize, q.data.as_ptr() as usize + q.data.len()))
            .filter(|&(a, z)| a >= base && z <= base + map.len()).collect();
        let (lo, hi) = (spans.iter().map(|s| s.0).min(), spans.iter().map(|s| s.1).max());
        if let (Some(lo), Some(hi)) = (lo, hi) {
            read_ahead(&map[lo - base..hi - base]);
        }
    }
    // locked where the limit allows; from the first refusal on, copied - on every core, since a copy is mostly page
    // faults (first the file's, then our own memory's)
    let mut locked = true;
    let mut to_copy: Vec<&mut QMat> = Vec::new();
    for q in keep {
        if !(locked && pin(q.data)) {
            locked = false;
            to_copy.push(q);
        }
    }
    {
        use rayon::prelude::*;
        to_copy.par_iter_mut().for_each(|q| {
            let old = q.data;
            q.data = owned(old);
            cold(old);
        });
    }
    let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
    let summary = format!("ram_experts={n_resident} ram_gib={:.2} ram_always_gib={:.2} ssd_experts={n_cold} ssd_gib={:.2} ram={} ram_ms={}",
                          gib(bytes), gib(always), gib(cold_bytes), if locked { "mlock" } else { "copy" }, t0.elapsed().as_millis());
    (Residency { resident, n_resident, bytes, locked }, summary)
}

fn on_device(_q: &QMat) -> bool {
    #[cfg(feature = "gpu")]
    return _q.gpu.is_some();
    #[cfg(not(feature = "gpu"))]
    false
}

/// Moves the matrices every token uses, then the resident experts, into the engine's own memory, for when mlock was refused (the usual 8 MiB memlock limit).
/// File pages are the kernel's to evict under pressure, so advice alone did not keep them: on GLM-4.7-Flash in 10 GiB a
/// profile routing 95% to "resident" experts still read 25 GB a request. Anonymous memory stays (macOS compresses it
/// instead of dropping it). The mapping's copies are then marked cold. -> bytes copied.
pub fn copy_resident(model: &mut Model, res: &Residency) -> u64 {
    let ne = model.profile_experts;
    // first the matrices every token uses (attention, dense and shared feed-forwards, the output): when they are
    // evicted, every token reads them again (round 3: 21 GB a request with 95% of the routing resident)
    let mut copied = copy_always(model);
    for layer in model.layers.iter_mut() {
        for (_, block) in layer.parts.iter_mut() {
            let Block::Moe(m) = block else { continue };
            for e in 0..m.n_expert {
                if !res.resident.get(m.layer * ne + e).copied().unwrap_or(false) {
                    continue;
                }
                let mut qs: Vec<&mut QMat> = vec![&mut m.up_exps.0[e], &mut m.down_exps.0[e]];
                if let Some(g) = &mut m.gate_exps {
                    qs.push(&mut g.0[e]);
                }
                for q in qs {
                    let old = q.data;
                    q.data = owned(old);
                    cold(old);
                    copied += old.len() as u64;
                }
            }
        }
    }
    copied
}

/// Keeps the matrices every token uses in RAM when the experts were locked: mlock where the limit allows, else a copy.
/// Without it they are file pages like the cold experts and are evicted with them (Tiel on the 16 GB Mac: the GDN and
/// attention took 3x longer while the cold experts paged in). -> "mlock" or "copy".
pub fn keep_always(model: &mut Model) -> &'static str {
    if model.matrices_mut().0.iter().all(|q| pin(q.data)) {
        "mlock"
    } else {
        copy_always(model);
        "copy"
    }
}

/// Moves the matrices every token uses into the engine's own memory. -> bytes copied.
fn copy_always(model: &mut Model) -> u64 {
    let mut copied = 0u64;
    for q in model.matrices_mut().0 {
        let old = q.data;
        q.data = owned(old);
        cold(old);
        copied += old.len() as u64;
    }
    copied
}

/// A copy in memory of our own, 64-byte aligned like the GGUF's tensor data, kept for the life of the process: it is
/// never freed (deliberately; the engine loads one model per process), so calling `copy_resident` again would leak.
pub(crate) fn owned(b: &[u8]) -> &'static [u8] {
    if b.is_empty() {
        return &[];
    }
    let layout = std::alloc::Layout::from_size_align(b.len(), 64).expect("layout");
    unsafe {
        let p = std::alloc::alloc(layout);
        if p.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        std::ptr::copy_nonoverlapping(b.as_ptr(), p, b.len());
        std::slice::from_raw_parts(p, b.len())
    }
}

/// Routing since the last call that went to resident experts: (resident hits, all routed).
pub fn hit_rate(model: &Model, res: &Residency, last: &mut Vec<u32>) -> (u64, u64) {
    if last.len() != model.heat.len() {
        *last = vec![0; model.heat.len()];
    }
    let (mut hit, mut all) = (0u64, 0u64);
    for (i, h) in model.heat.iter().enumerate() {
        let now = h.load(Ordering::Relaxed);
        let d = (now - last[i]) as u64;
        last[i] = now;
        all += d;
        if res.resident.get(i).copied().unwrap_or(false) {
            hit += d;
        }
    }
    (hit, all)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_ram_tier_is_one_run_after_the_devices() {
        // experts 0 and 1 are on a device (0 bytes left for RAM); 2 and 3 fit; 4 does not, and 5 - though it would -
        // is not taken, so the tier stays one run of the order
        assert_eq!(ram_plan(&[0, 0, 4, 4, 9, 1], 10), vec![false, false, true, true, false, false]);
        assert_eq!(ram_plan(&[4, 4], 0), vec![false, false]);
        assert_eq!(ram_plan(&[], 10), Vec::<bool>::new());
    }

    #[test]
    fn profile_round_trip() {
        let p = Profile { n_layers: 3, n_expert: 4, ranked: vec![(1, 2), (0, 3), (2, 0)] };
        let path = std::env::temp_dir().join(format!("am2-prof-{}.bin", std::process::id()));
        p.write(&path).unwrap();
        let b = std::fs::read(&path).unwrap();
        // header, 3 pairs, then the rank table of 3 x 4 (write_expert_profile's layout)
        assert_eq!(b.len(), 24 + 3 * 4 + 12 * 4);
        assert_eq!(&b[..4], b"STRP");
        let rank_of = |l: usize, e: usize| i32::from_le_bytes(b[36 + 4 * (l * 4 + e)..40 + 4 * (l * 4 + e)].try_into().unwrap());
        assert_eq!((rank_of(1, 2), rank_of(0, 3), rank_of(2, 0), rank_of(0, 0)), (0, 1, 2, -1));
        let q = Profile::read(&path, 3, 4).unwrap();
        assert_eq!(q.ranked, p.ranked);
        assert!(Profile::read(&path, 3, 5).unwrap_err().contains("different model"));
        let _ = std::fs::remove_file(&path);
    }
}
