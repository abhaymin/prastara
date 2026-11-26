//! Weights on GPUs (docs/GPU-PLAN.md phases 3 and 4): `--devices cuda:0=12,hip:0=6` puts matrices on those devices
//! within those budgets (GiB), in their GGUF encoding, and the products of those matrices are computed there by
//! strata-gpu's kernels (CubeCL: CUDA, ROCm/HIP, Vulkan). The rest stays on the CPU, memory-mapped as before.
//!
//! The order is the strata placement's: first what every token uses (the output projection, then each layer's
//! attention, SSM, dense, shared-expert and latent matrices), then the routed experts as the expert profile ranks
//! them (the hottest first; experts it does not rank follow, layer by layer). Devices fill in the order given, so the
//! fastest one should come first; an integrated GPU (it shares the system RAM) comes after a discrete one.

use crate::model::{Model, QMat};
use crate::pack::Profile;
use std::sync::{Arc, OnceLock};

static DEVICES: OnceLock<Vec<strata_gpu::Gpu>> = OnceLock::new();

/// A matrix's copy on a device.
#[derive(Clone)]
pub struct OnDevice {
    dev: usize,
    w: Arc<strata_gpu::Weights>,
}

impl OnDevice {
    pub fn matvec(&self, x: &[f32]) -> Vec<f32> {
        let t0 = std::time::Instant::now();
        let y = DEVICES.get().expect("devices opened")[self.dev].matvec(&self.w, x);
        crate::model::times::device(t0);
        y
    }

    /// Several matrices on the same device with the same input, in one submission.
    pub fn matvec_many(group: &[&OnDevice], x: &[f32]) -> Vec<Vec<f32>> {
        let ws: Vec<&strata_gpu::Weights> = group.iter().map(|g| g.w.as_ref()).collect();
        let t0 = std::time::Instant::now();
        let ys = DEVICES.get().expect("devices opened")[group[0].dev].matvec_many(&ws, x);
        crate::model::times::device(t0);
        ys
    }

    pub fn device(&self) -> usize {
        self.dev
    }

    pub(crate) fn weights(&self) -> &strata_gpu::Weights {
        &self.w
    }
}

/// An open device, for code that chains kernels on it (graph.rs).
pub(crate) fn device(d: usize) -> &'static strata_gpu::Gpu {
    &DEVICES.get().expect("devices opened")[d]
}

/// What each device's --devices budget has left after placement (bytes), in device order: what a graph's KV cache
/// may still take (Kilo on #76: the cache must count against the budget, not come on top of it).
pub static ROOM: std::sync::OnceLock<Vec<u64>> = std::sync::OnceLock::new();

/// The most one device holds. Past 11-12 GiB on a 16 GB card, CubeCL's memory manager panics on its stream thread
/// ("Memory page 0 doesn't exist"), measured with Cyber-Tiel UD-Q8_K_XL.
const MAX_PER_DEVICE: u64 = 11 << 30;

/// `cuda:0=12,hip:0=6` -> [("cuda:0", 11 GiB), ("hip:0", 6 GiB)]; a device without `=` gets the most it can hold.
pub fn parse(spec: &str) -> Result<Vec<(String, u64)>, String> {
    spec.split(',').map(str::trim).filter(|s| !s.is_empty()).map(|s| {
        let (name, gib) = s.split_once('=').unwrap_or((s, "1024"));
        let g: f64 = gib.parse().map_err(|_| format!("--devices: {s:?}: the budget is GiB, e.g. cuda:0=12"))?;
        Ok((name.to_string(), ((g * (1u64 << 30) as f64) as u64).min(MAX_PER_DEVICE)))
    }).collect()
}

/// What placement needs to know of a matrix: its bytes, and whether a kernel takes it on a device (its type has one
/// and its width is whole blocks).
#[derive(Clone, Copy)]
pub struct Item {
    pub bytes: u64,
    /// the type has a kernel
    pub kernel: bool,
    /// and the width is a multiple of its block
    pub whole: bool,
}

impl Item {
    fn of(q: &QMat) -> Item {
        let g = strata_gpu::kernels::geometry(q.ty());
        Item { bytes: q.data.len() as u64, kernel: g.is_some(), whole: g.is_some_and(|(per_block, _, _)| q.cols % per_block as usize == 0) }
    }
}

/// Where each matrix goes, in the order given: `Some(device)` or `None` (the CPU), and what each budget has left.
/// Devices fill in order and are never returned to: a matrix that does not fit the current one moves on to the next,
/// and so does everything after it, so each device holds one run of the order (in a model pack, one run of the file).
/// A type with no kernel stays on the CPU without moving on.
pub fn plan(items: &[Item], budgets: &[u64]) -> (Vec<Option<usize>>, Vec<u64>) {
    let mut left = budgets.to_vec();
    let mut d = 0usize;
    let to = items.iter().map(|it| {
        while d < left.len() {
            if it.kernel && it.whole && it.bytes <= left[d] {
                left[d] -= it.bytes;
                return Some(d);
            }
            if !it.kernel {
                break;
            }
            d += 1;
        }
        None
    }).collect();
    (to, left)
}

/// How much a device may have queued before the loader has it flushed: the runtime keeps a host copy of every upload
/// until then, so without a bound the whole model sits in RAM a second time (measured: 17 GiB for Inkling 2 Mini).
const QUEUE_BYTES: u64 = 512 << 20;

/// Given what placement did, whether no matrix (dense/attention or expert) landed on any of the requested devices.
pub fn nothing_placed(dense_on_device: usize, experts_on_device: usize) -> bool {
    dense_on_device == 0 && experts_on_device == 0
}

/// Place the model's matrices on the devices; -> (a summary for the engine's INFO line, how many dense/attention
/// matrices were placed on a device, how many experts). Findings item 2: the caller uses the counts to know whether
/// `--devices` actually put anything where it was asked.
///
/// The plan is made first (sizes are known), then the uploads are fed to all devices at once: each device's thread
/// copies to its card while the loader reads the next matrix for another, and the loader waits for a device only
/// when it has QUEUE_BYTES queued. From a model pack the experts go in the file's order unless a profile is given,
/// so each device's share is one run of the file, asked from the disk ahead of the uploads.
///
/// `reserve` bytes of the first device's budget are kept out of the plan and left in ROOM for the device graph's KV
/// cache (graph::kv_need), so a model's experts do not take the room the graph needs.
pub fn place(m: &mut Model, devices: &[(String, u64)], profile: Option<&Profile>, reserve: u64) -> Result<(String, usize, usize), String> {
    let t0 = std::time::Instant::now();
    let gpus: Vec<strata_gpu::Gpu> = devices.iter().map(|(n, _)| strata_gpu::Gpu::open(n)).collect::<Result<_, _>>()?;
    let open_ms = t0.elapsed().as_millis();
    let (order, from) = crate::pack::expert_order(m, profile);
    let from_file = from == "file";
    let file_map = m.file_map;
    let (every, mut experts) = m.matrices_mut();
    // every matrix in placement order, and which expert (by its place in the order) each belongs to
    let n_every = every.len();
    let mut qs: Vec<&mut QMat> = every;
    let mut owner: Vec<usize> = vec![usize::MAX; n_every];
    let mut n_groups = 0usize;
    for key in order {
        if let Some(group) = experts.remove(&key) {
            owner.extend(std::iter::repeat_n(n_groups, group.len()));
            qs.extend(group);
            n_groups += 1;
        }
    }
    let items: Vec<Item> = qs.iter().map(|q| Item::of(q)).collect();
    let mut budgets: Vec<u64> = devices.iter().map(|(_, b)| *b).collect();
    let reserve = reserve.min(budgets.first().copied().unwrap_or(0));
    if let Some(b) = budgets.first_mut() {
        *b -= reserve;
    }
    let (to, mut left) = plan(&items, &budgets);
    if let Some(l) = left.first_mut() {
        *l += reserve;
    }

    // read-ahead: from a pack each device's matrices are one run of the file (the core, then a run of the experts),
    // read from the disk by several threads ahead of the uploads
    let mut ahead: Vec<Option<std::sync::Arc<std::sync::atomic::AtomicUsize>>> = vec![None; gpus.len()];
    if let Some(map) = file_map.filter(|_| from_file) {
        let base = map.as_ptr() as usize;
        for (d, slot) in ahead.iter_mut().enumerate() {
            let mut span: Option<(usize, usize)> = None;
            for (q, _) in qs.iter().zip(&to).filter(|(_, t)| **t == Some(d)) {
                let a = q.data.as_ptr() as usize;
                if a >= base && a + q.data.len() <= base + map.len() {
                    span = Some(span.map_or((a, a + q.data.len()), |(lo, hi)| (lo.min(a), hi.max(a + q.data.len()))));
                }
            }
            if let Some((lo, hi)) = span {
                *slot = Some(crate::pack::read_ahead_run(&map[lo - base..hi - base]));
            }
        }
    }

    // the uploads, each device fed in turn: always the device with the least sent so far that still has work
    let mut queues: Vec<std::collections::VecDeque<usize>> = vec![Default::default(); gpus.len()];
    to.iter().enumerate().for_each(|(i, t)| if let Some(d) = t { queues[*d].push_back(i) });
    let mut slots: Vec<Option<&mut QMat>> = qs.into_iter().map(Some).collect();
    let mut sent = vec![0u64; gpus.len()];
    let mut queued = vec![0u64; gpus.len()];
    let mut waited = vec![0u128; gpus.len()];
    let t_up = std::time::Instant::now();
    while let Some(d) = (0..gpus.len()).filter(|&d| !queues[d].is_empty()).min_by_key(|&d| sent[d]) {
        let i = queues[d].pop_front().expect("a queue with work");
        let q = slots[i].take().expect("each matrix is uploaded once");
        let w = gpus[d].upload(q.ty(), q.data, q.rows, q.cols)?;
        q.gpu = Some(OnDevice { dev: d, w: Arc::new(w) });
        // the device has its copy now (the client copied it already): the file's pages need not stay mapped
        crate::pack::released(q.data);
        if let Some(p) = &ahead[d] {
            p.fetch_max(q.data.as_ptr() as usize + q.data.len(), std::sync::atomic::Ordering::Relaxed);
        }
        sent[d] += items[i].bytes;
        queued[d] += items[i].bytes;
        if queued[d] >= QUEUE_BYTES {
            let t = std::time::Instant::now();
            // a flush waits for the uploads before the previous one and frees their host copies: at most two
            // queues' worth of copies are alive per device
            gpus[d].flush()?;
            waited[d] += t.elapsed().as_millis();
            queued[d] = 0;
        }
    }
    let feed_ms = t_up.elapsed().as_millis();
    for (d, g) in gpus.iter().enumerate() {
        let t = std::time::Instant::now();
        g.flush()?;
        g.flush()?;
        g.sync()?;
        waited[d] += t.elapsed().as_millis();
    }
    let upload_ms = t_up.elapsed().as_millis();

    let mut placed = vec![(0usize, 0u64); gpus.len()];
    let (mut cpu_n, mut cpu_bytes) = (0usize, 0u64);
    let mut group_on = vec![true; n_groups];
    for (i, t) in to.iter().enumerate() {
        match t {
            Some(d) => { placed[*d].0 += 1; placed[*d].1 += items[i].bytes; }
            None => {
                cpu_n += 1;
                cpu_bytes += items[i].bytes;
                if owner[i] != usize::MAX {
                    group_on[owner[i]] = false;
                }
            }
        }
    }
    let n_experts_on = group_on.iter().filter(|&&on| on).count();
    let dense_on_device = to[..n_every].iter().filter(|t| t.is_some()).count();
    let _ = DEVICES.set(gpus);
    let _ = ROOM.set(left);
    let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
    let per: Vec<String> = devices.iter().zip(&placed).map(|((n, _), (k, b))| format!("{n}:{k}:{:.2}GiB", gib(*b))).collect();
    let waits: Vec<String> = waited.iter().map(|w| w.to_string()).collect();
    Ok((format!("devices={} experts_on_devices={n_experts_on} cpu_matrices={cpu_n} cpu_gib={:.2} order={} open_ms={open_ms} upload_ms={upload_ms} feed_ms={feed_ms} device_wait_ms={}",
                per.join(","), gib(cpu_bytes), from, waits.join("/")),
        dense_on_device, n_experts_on))
}

/// findings item 2: the decision the startup path makes after placement. `--devices` asked the GPU to take weights but
/// nothing landed on any requested device (dense and experts), so it falls back to the CPU. Pure, so a test covers it
/// without hardware; the default strata-hybrid features build this module and run it.
#[cfg(feature = "gpu")]
#[test]
fn nothing_placed_on_any_device_is_a_cpu_run() {
    assert!(nothing_placed(0, 0), "--devices requested a GPU but placed no dense matrix and no expert on it");
    assert!(!nothing_placed(1, 0), "dense or attention weights on a device are a real GPU placement");
    assert!(!nothing_placed(0, 2), "experts on a device count as placed too");
}

/// The plan fills the devices in order, each with one run of the order, and never returns to an earlier device.
#[cfg(feature = "gpu")]
#[test]
fn the_plan_gives_each_device_one_run() {
    let it = |bytes| Item { bytes, kernel: true, whole: true };
    let items = [it(4), it(4), it(4), Item { bytes: 1, kernel: false, whole: false }, it(4), it(2), it(4), it(9)];
    let (to, left) = plan(&items, &[10, 9]);
    // 4+4 on the first; the third does not fit and opens the second; no kernel: the CPU, without moving on;
    // 4 fits the second (4+4); 2 does not (1 left) and nothing returns to the first, so the rest is the CPU's
    assert_eq!(to, vec![Some(0), Some(0), Some(1), None, Some(1), None, None, None]);
    assert_eq!(left, vec![2, 1]);
    // a width that is not whole blocks moves on like one that does not fit (as the upload always did)
    let (to, _) = plan(&[Item { bytes: 1, kernel: true, whole: false }, it(1)], &[10, 10]);
    assert_eq!(to, vec![None, None]);
}
