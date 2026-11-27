//! Layer streaming: train a model whose frozen base does not fit on the device, one decoder block at a time.
//!
//! The idea is Soup's `training.stream_layers` (<https://github.com/MakazhanAlpamys/Soup>, Apache-2.0, Copyright 2026
//! Makazhan Alpamys; its `utils/layer_stream.py`, `layer_stream_runtime.py` and `layer_shard.py` were read as text):
//! the frozen base lives in host memory (the RAM tier) or in a per-model shard cache on disk (the disk tier), and each
//! block's frozen weights are copied to the device just before the block runs — in the forward pass, and again in the
//! checkpoint replay of the backward pass (`dL/dx = Wᵀ·dL/dy` needs W again to reach the blocks below, so each block is
//! read twice a step). Only the adapters, the MoE routers, their gradients and AdamW's moments, the final norm and the
//! head stay on the device. This is written from scratch in Rust on candle; no code of Soup is used.
//!
//! **Buffers.** Each device keeps at most `buffers` (2 to 8) blocks' frozen weights at once ([`Pool`]): the block that
//! runs and the one prefetched next. Soup pre-allocates the buffers and copies into them in place; candle's tensors
//! cannot be written in place from the host, so each fill is a fresh device tensor of the same size, which candle's
//! stream-ordered CUDA memory pool hands back from the block the evicted one freed. A slot that is evicted while a
//! graph still holds its tensors therefore keeps them alive instead of being overwritten — a wrong gradient is not
//! possible, only memory above the bound until that graph is dropped.
//!
//! **Prefetch and pinning (CUDA).** In the RAM tier each block's matrices are copied once into page-locked host memory
//! (cudarc's `alloc_pinned`, in the block's own device's context) as long as the pinned total stays under a limit
//! (0.7 of the free RAM at load, or `AM2_STREAM_PIN_LIMIT` bytes); beyond it they stay pageable, said in the log. A
//! second CUDA stream per device (cudarc `new_stream`) carries the copies: the next block's upload is queued there
//! while the current block's kernels run on candle's stream, and cudarc's per-allocation events make the compute
//! stream wait for the copy (and the copy's buffer outlive the kernels reading it). A pageable or disk-tier block is
//! uploaded synchronously on the compute stream (no overlap). On the CPU (tests) a fill is a plain copy.
//!
//! **Order.** [`LayerStream::begin`] is told the order the blocks will run in (with a loop, its blocks again for each
//! pass); a block's forward prefetches the next one in that order, and its replay in the backward pass the block that
//! was checkpointed before it. A wrong guess costs a synchronous load, never a wrong result: every block is fetched on
//! demand and prefetching is only a hint.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
#[cfg(feature = "cuda")]
use std::sync::Arc;
use std::sync::Mutex;

use anyhow::{anyhow, bail, Context as _, Result};
use candle_core::{DType, Device, Tensor};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use crate::model::{BaseDtype, ModelConfig, Target};

/// The fewest buffers: one block running and one loading (a single buffer cannot overlap the two — Soup calls it a
/// scheduler bug, not a configuration).
pub const MIN_BUFFERS: usize = 2;
pub const MAX_BUFFERS: usize = 8;
pub const DEFAULT_BUFFERS: usize = 2;
/// The RAM tier is chosen by `auto` when the streamed base is under this share of the free RAM (Soup's headroom).
pub const RAM_TIER_HEADROOM: f64 = 0.7;
/// A USB4 / Thunderbolt tunnel's practical PCIe throughput (40 Gb/s links carry about 3 GB/s of PCIe data): what a
/// card behind one is assumed to reach, whatever link its sysfs entry reports.
pub const USB4_TUNNEL_BYTES_S: f64 = 3.0e9;
/// The share of a PCIe link's raw rate a pinned host-to-device copy reaches.
pub const PCIE_EFFICIENCY: f64 = 0.8;
/// When nothing says (no sysfs, not Linux): a PCIe 4.0 x4 link, the slowest internal slot a GPU sits in here.
pub const DEFAULT_LINK_BYTES_S: f64 = 6.3e9;
/// The shard cache's format.
pub const SHARD_FORMAT: u64 = 1;

/// Where the frozen base is streamed from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tier {
    /// host memory (page-locked on CUDA where possible)
    Ram,
    /// a per-model shard cache on disk, one file per block, read on demand
    Disk,
}

impl Tier {
    pub fn as_str(self) -> &'static str {
        match self {
            Tier::Ram => "ram",
            Tier::Disk => "disk",
        }
    }
}

/// `--stream-layers ram|disk|auto`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamMode {
    Ram,
    Disk,
    /// the RAM tier when the base fits [`RAM_TIER_HEADROOM`] of the free RAM, else the disk tier
    Auto,
}

impl StreamMode {
    pub fn from_s(s: &str) -> Result<Self> {
        match s.trim() {
            "ram" => Ok(StreamMode::Ram),
            "disk" => Ok(StreamMode::Disk),
            "auto" | "" => Ok(StreamMode::Auto),
            other => bail!("--stream-layers {other:?}: ram, disk or auto"),
        }
    }
    pub fn as_str(self) -> &'static str {
        match self {
            StreamMode::Ram => "ram",
            StreamMode::Disk => "disk",
            StreamMode::Auto => "auto",
        }
    }
}

/// `--stream-buffers`: 2 to 8.
pub fn validate_buffers(n: usize) -> Result<usize> {
    if !(MIN_BUFFERS..=MAX_BUFFERS).contains(&n) {
        bail!("--stream-buffers {n}: choose {MIN_BUFFERS} to {MAX_BUFFERS} (one buffer cannot load the next block while the current one runs)");
    }
    Ok(n)
}

/// The tier for `mode`: `auto` takes the RAM tier when the streamed base (`store_bytes`) is under
/// [`RAM_TIER_HEADROOM`] of `ram_available` (0: unknown, then RAM), else the disk tier, and says so.
pub fn choose_tier(mode: StreamMode, store_bytes: u64, ram_available: u64) -> (Tier, Option<String>) {
    match mode {
        StreamMode::Ram => (Tier::Ram, None),
        StreamMode::Disk => (Tier::Disk, None),
        StreamMode::Auto if ram_available == 0 || (store_bytes as f64) < ram_available as f64 * RAM_TIER_HEADROOM => (Tier::Ram, None),
        StreamMode::Auto => (Tier::Disk, Some(format!(
            "the streamed base is {:.1} GB and only {:.1} GB of RAM is free: streaming from the disk tier (a shard cache read on demand, slower than RAM; \
             --stream-layers ram refuses instead)", store_bytes as f64 / 1e9, ram_available as f64 / 1e9))),
    }
}

/// The settings a streamed load takes.
#[derive(Debug, Clone)]
pub struct StreamOptions {
    pub mode: StreamMode,
    pub buffers: usize,
    /// the disk tier's cache root (default [`default_cache_root`])
    pub cache_root: Option<PathBuf>,
}

/// The streamed bytes of one block: its seven matrices in the base dtype (norms and biases stay on the device).
pub fn block_stream_bytes(cfg: &ModelConfig, dtype: BaseDtype) -> u64 {
    (cfg.block_params() - cfg.block_small_params()) * dtype.bytes()
}

/// `MemAvailable` (bytes), 0 when the OS does not say.
pub fn ram_available() -> u64 {
    std::fs::read_to_string("/proc/meminfo").map(|t| crate::profile::parse_meminfo(&t).1).unwrap_or(0)
}

// ---------------------------------------------------------------------------------------------------------------------
// the disk tier's shard cache (after Soup's layer_shard: one file per block, an index written last, keyed by what the
// weights depend on so a changed checkpoint is never streamed from stale shards)
// ---------------------------------------------------------------------------------------------------------------------

/// `$AM2_STREAM_CACHE`, else `$XDG_CACHE_HOME/prastara/layer-stream`, else `~/.cache/prastara/layer-stream`.
pub fn default_cache_root() -> PathBuf {
    if let Some(p) = std::env::var_os("AM2_STREAM_CACHE").filter(|p| !p.is_empty()) {
        return PathBuf::from(p);
    }
    let base = std::env::var_os("XDG_CACHE_HOME").filter(|p| !p.is_empty()).map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))
        .unwrap_or_else(std::env::temp_dir);
    base.join("prastara").join("layer-stream")
}

/// SHA-256 of a file's first `limit` bytes (the whole file when it is shorter), read a MiB at a time.
fn sha_file_head(path: &Path, limit: u64) -> Result<String> {
    use std::io::Read as _;
    let f = std::fs::File::open(path).with_context(|| format!("{}", path.display()))?;
    let mut r = f.take(limit);
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = r.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(format!("{:x}", h.finalize()))
}

/// What a model's shards depend on: the config, each weight file's name, size, modification time and first MiB, the
/// fine-grained permutation file, and the base dtype.
pub fn shard_fingerprint(model_dir: &Path, dtype: BaseDtype) -> Result<Value> {
    let mut weights = Vec::new();
    for f in crate::model::weight_files(model_dir)? {
        let meta = std::fs::metadata(&f)?;
        let mtime = meta.modified().ok().and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok()).map_or(0, |d| d.as_nanos() as u64);
        weights.push(json!({"file": f.file_name().and_then(|n| n.to_str()), "bytes": meta.len(), "mtime_ns": mtime, "head_sha256": sha_file_head(&f, 1 << 20)?}));
    }
    let init = model_dir.join(crate::model::MOE_INIT_FILE);
    Ok(json!({"format": SHARD_FORMAT, "config_sha256": sha_file_head(&model_dir.join("config.json"), u64::MAX)?, "weights": weights,
              "moe_init_sha256": if init.exists() { Some(sha_file_head(&init, u64::MAX)?) } else { None }, "base_dtype": dtype.as_str()}))
}

/// The cache folder of a model: `<root>/<folder name>-<dtype>-<12 hex of the fingerprint>`.
pub fn shard_dir(root: &Path, model_dir: &Path, fingerprint: &Value, dtype: BaseDtype) -> PathBuf {
    let name: String = model_dir.file_name().and_then(|n| n.to_str()).unwrap_or("model").chars()
        .map(|c| if c.is_ascii_alphanumeric() || "._-".contains(c) { c } else { '_' }).take(96).collect();
    let hash = format!("{:x}", Sha256::digest(fingerprint.to_string().as_bytes()));
    root.join(format!("{}-{}-{}", name.trim_matches('.'), dtype.as_str(), &hash[..12]))
}

pub fn shard_path(dir: &Path, block: usize) -> PathBuf {
    dir.join(format!("block-{block:03}.safetensors"))
}

/// Whether `dir` holds a complete cache for `fingerprint` and `blocks` blocks.
pub fn shard_cache_ready(dir: &Path, fingerprint: &Value, blocks: usize) -> bool {
    let Ok(text) = std::fs::read_to_string(dir.join("index.json")) else { return false };
    let Ok(idx) = serde_json::from_str::<Value>(&text) else { return false };
    idx["fingerprint"] == *fingerprint && idx["blocks"].as_u64() == Some(blocks as u64) && (0..blocks).all(|b| shard_path(dir, b).exists())
}

fn free_disk_bytes(p: &Path) -> Option<u64> {
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStrExt as _;
        let mut dir = p.to_path_buf();
        while !dir.exists() {
            dir = dir.parent()?.to_path_buf();
        }
        let c = std::ffi::CString::new(dir.as_os_str().as_bytes()).ok()?;
        let mut s: libc::statvfs = unsafe { std::mem::zeroed() };
        if unsafe { libc::statvfs(c.as_ptr(), &mut s) } != 0 {
            return None;
        }
        Some(s.f_bavail as u64 * s.f_frsize as u64)
    }
    #[cfg(not(unix))]
    {
        let _ = p;
        None
    }
}

/// Make (or find) the shard cache of `model_dir` under `root`: each block's seven matrices in `dtype`, as the trainer
/// holds them (a fine-grained upcycle's feed-forward already in group order), one file per block, written one block at
/// a time (the RAM it takes is one block). Returns the folder.
pub fn build_shard_cache(model_dir: &Path, dtype: BaseDtype, root: &Path, on_log: &mut dyn FnMut(String)) -> Result<PathBuf> {
    let cfg = ModelConfig::load(model_dir)?;
    let fp = shard_fingerprint(model_dir, dtype)?;
    let dir = shard_dir(root, model_dir, &fp, dtype);
    let n = cfg.num_hidden_layers;
    if shard_cache_ready(&dir, &fp, n) {
        on_log(format!("layer streaming: the shard cache {} is complete ({n} blocks)", dir.display()));
        return Ok(dir);
    }
    let need = block_stream_bytes(&cfg, dtype) * n as u64;
    if let Some(free) = free_disk_bytes(root) {
        if free < need + (need / 20) {
            bail!("the layer-stream shard cache needs {:.1} GB and {} has {:.1} GB free: choose another --stream-cache folder, or --stream-layers ram",
                  need as f64 / 1e9, root.display(), free as f64 / 1e9);
        }
    }
    std::fs::remove_dir_all(&dir).ok();
    std::fs::create_dir_all(&dir).with_context(|| format!("{}", dir.display()))?;
    on_log(format!("layer streaming: writing the shard cache ({:.1} GB, {n} blocks) to {}", need as f64 / 1e9, dir.display()));
    for b in 0..n {
        let one = crate::model::LlamaModel::load_parts(model_dir, dtype, &Device::Cpu, b..b + 1, false, false)?;
        let mut map: HashMap<String, Tensor> = HashMap::new();
        for t in Target::ALL {
            let w = one.block_weight(b, t).ok_or_else(|| anyhow!("block {b}: no {}", t.as_str()))?;
            map.insert(t.as_str().to_string(), w.contiguous()?);
        }
        let tmp = shard_path(&dir, b).with_extension("tmp");
        candle_core::safetensors::save(&map, &tmp)?;
        std::fs::rename(&tmp, shard_path(&dir, b))?;
    }
    std::fs::write(dir.join("index.json"), serde_json::to_string_pretty(&json!({"fingerprint": fp, "blocks": n, "dtype": dtype.as_str(),
        "tensors": Target::ALL.iter().map(|t| t.as_str()).collect::<Vec<_>>(), "credit": "after Soup's layer_shard (Apache-2.0)"}))?)?;
    Ok(dir)
}

// ---------------------------------------------------------------------------------------------------------------------
// the host store and the device buffers
// ---------------------------------------------------------------------------------------------------------------------

/// One frozen matrix in host memory.
enum HostTensor {
    /// a CPU tensor (pageable on CUDA; on the CPU it is the store itself)
    Cpu(Tensor),
    /// page-locked host memory in the block's device's context, with its shape and dtype
    #[cfg(feature = "cuda")]
    Pinned(Pinned, Vec<usize>),
}

#[cfg(feature = "cuda")]
enum Pinned {
    BF16(candle_core::cuda_backend::cudarc::driver::PinnedHostSlice<half::bf16>),
    F16(candle_core::cuda_backend::cudarc::driver::PinnedHostSlice<half::f16>),
    F32(candle_core::cuda_backend::cudarc::driver::PinnedHostSlice<f32>),
}

/// Where a block's matrices come from.
enum HostBlock {
    Ram(Vec<HostTensor>),
    Disk(PathBuf),
}

/// One device's buffers: at most `n` blocks' weights, least recently used out first.
struct Pool {
    /// (block, its seven matrices on the device, last use)
    slots: Vec<Option<(usize, Vec<Tensor>, u64)>>,
    clock: u64,
    /// the slot the last acquire used (a prefetch never evicts it)
    current: Option<usize>,
    /// slots filled by a prefetch and not yet used
    prefetched: Vec<bool>,
}

impl Pool {
    fn find(&self, block: usize) -> Option<usize> {
        self.slots.iter().position(|s| s.as_ref().is_some_and(|(b, _, _)| *b == block))
    }
    fn victim(&self, keep: Option<usize>) -> Option<usize> {
        if let Some(e) = (0..self.slots.len()).find(|i| self.slots[*i].is_none() && Some(*i) != keep) {
            return Some(e);
        }
        (0..self.slots.len()).filter(|i| Some(*i) != keep).min_by_key(|i| self.slots[*i].as_ref().map_or(0, |(_, _, u)| *u))
    }
}

/// What streaming did, for the log and the report.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct StreamStats {
    /// blocks copied to a device
    pub loads: u64,
    pub bytes: u64,
    /// loads a block's run had to wait for (not prefetched)
    pub demand_loads: u64,
    /// runs that found their block already prefetched
    pub prefetch_hits: u64,
}

#[derive(Default)]
struct Schedule {
    order: Vec<usize>,
    /// positions run through a checkpoint node (they are replayed in the backward pass)
    replayed: Vec<bool>,
    generation: u64,
}

/// The streamed base of one model: its host store, each device's buffers and copy stream, and the run order.
pub struct LayerStream {
    pub tier: Tier,
    pub buffers: usize,
    blocks: Vec<Option<HostBlock>>,
    shapes: Vec<(usize, usize)>,
    dtype: DType,
    devs: Vec<Device>,
    block_dev: Vec<usize>,
    pools: Vec<Mutex<Pool>>,
    #[cfg(feature = "cuda")]
    copy_streams: Vec<Option<Arc<candle_core::cuda_backend::cudarc::driver::CudaStream>>>,
    schedule: Mutex<Schedule>,
    stats: Mutex<StreamStats>,
    /// bytes held page-locked, and in pageable host memory
    pub pinned_bytes: u64,
    pub pageable_bytes: u64,
    /// the disk tier's folder
    pub shard_dir: Option<PathBuf>,
    /// what the log should say about how the copies run
    pub notes: Vec<String>,
}

impl std::fmt::Debug for LayerStream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "LayerStream({}, {} buffers, {} blocks)", self.tier.as_str(), self.buffers, self.blocks.len())
    }
}

/// Builds a [`LayerStream`] while the model loads.
pub struct StreamBuilder {
    tier: Tier,
    buffers: usize,
    blocks: Vec<Option<HostBlock>>,
    shapes: Vec<(usize, usize)>,
    dtype: DType,
    devs: Vec<Device>,
    block_dev: Vec<usize>,
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pin_limit: u64,
    pinned_bytes: u64,
    pageable_bytes: u64,
    pin_failed: Option<String>,
    shard_dir: Option<PathBuf>,
}

impl StreamBuilder {
    pub fn new(cfg: &ModelConfig, dtype: DType, tier: Tier, buffers: usize, devs: &[Device], block_dev: &[usize], shard_dir: Option<PathBuf>) -> Result<Self> {
        validate_buffers(buffers)?;
        let pin_limit = std::env::var("AM2_STREAM_PIN_LIMIT").ok().and_then(|v| v.trim().parse::<u64>().ok())
            .unwrap_or_else(|| { let a = ram_available(); if a == 0 { u64::MAX } else { (a as f64 * RAM_TIER_HEADROOM) as u64 } });
        Ok(Self { tier, buffers, blocks: (0..cfg.num_hidden_layers).map(|_| None).collect(), shapes: Target::ALL.iter().map(|t| cfg.target_shape(*t)).collect(),
                  dtype, devs: devs.to_vec(), block_dev: block_dev.to_vec(), pin_limit, pinned_bytes: 0, pageable_bytes: 0, pin_failed: None, shard_dir })
    }

    pub fn tier(&self) -> Tier {
        self.tier
    }

    /// Block `i`'s seven matrices (CPU tensors, in [`Target::ALL`] order, as the model holds them) into the RAM store.
    pub fn put_ram(&mut self, i: usize, ws: Vec<Tensor>) -> Result<()> {
        let dev = self.devs[self.block_dev[i]].clone();
        let mut held = Vec::with_capacity(ws.len());
        for w in ws {
            let bytes = (w.elem_count() * w.dtype().size_in_bytes()) as u64;
            held.push(self.hold(w, &dev, bytes)?);
        }
        self.blocks[i] = Some(HostBlock::Ram(held));
        Ok(())
    }

    /// Block `i` is in the disk tier's cache.
    pub fn put_disk(&mut self, i: usize) -> Result<()> {
        let dir = self.shard_dir.as_ref().ok_or_else(|| anyhow!("the disk tier without a shard cache"))?;
        self.blocks[i] = Some(HostBlock::Disk(shard_path(dir, i)));
        Ok(())
    }

    #[allow(unused_variables)]
    fn hold(&mut self, w: Tensor, dev: &Device, bytes: u64) -> Result<HostTensor> {
        let w = w.contiguous()?;
        #[cfg(feature = "cuda")]
        if let Device::Cuda(cd) = dev {
            if self.pin_failed.is_none() && self.pinned_bytes + bytes <= self.pin_limit {
                match pin(&w, cd) {
                    Ok(p) => {
                        self.pinned_bytes += bytes;
                        return Ok(HostTensor::Pinned(p, w.dims().to_vec()));
                    }
                    Err(e) => self.pin_failed = Some(format!("{e}")),
                }
            } else if self.pin_failed.is_none() {
                self.pin_failed = Some(format!("the pinned store reached its limit of {:.1} GB (AM2_STREAM_PIN_LIMIT, default 0.7 of the free RAM)", self.pin_limit as f64 / 1e9));
            }
        }
        self.pageable_bytes += bytes;
        Ok(HostTensor::Cpu(w))
    }

    pub fn build(self) -> Result<LayerStream> {
        let mut notes = Vec::new();
        #[cfg(feature = "cuda")]
        let copy_streams: Vec<Option<Arc<candle_core::cuda_backend::cudarc::driver::CudaStream>>> = self.devs.iter().map(|d| match d {
            Device::Cuda(cd) => match cd.cuda_stream().context().new_stream() {
                Ok(s) => Some(s),
                Err(e) => {
                    notes.push(format!("no second CUDA stream ({e}): uploads run on the compute stream, without overlap"));
                    None
                }
            },
            _ => None,
        }).collect();
        let any_cuda = self.devs.iter().any(|d| d.is_cuda());
        if any_cuda {
            match self.tier {
                Tier::Ram => {
                    notes.push(format!("{:.2} GB of the base page-locked, {:.2} GB pageable", self.pinned_bytes as f64 / 1e9, self.pageable_bytes as f64 / 1e9));
                    if let Some(why) = &self.pin_failed {
                        notes.push(format!("not all of the base could be page-locked ({why}): those blocks are uploaded synchronously (no overlap with the compute)"));
                    }
                    if self.pinned_bytes > 0 {
                        notes.push("the next block's upload is queued on a second CUDA stream while the current block computes".into());
                    }
                }
                Tier::Disk => notes.push("disk tier: each block is read from the shard cache and uploaded synchronously (no overlap with the compute)".into()),
            }
        } else if self.devs.iter().all(|d| d.is_cpu()) {
            notes.push("on the CPU there is no device memory to save: streaming copies each block in host memory (it runs the same path, for tests)".into());
        } else {
            notes.push("uploads are synchronous on this backend".into());
        }
        let pools = self.devs.iter().map(|_| Mutex::new(Pool { slots: vec![None; self.buffers], clock: 0, current: None, prefetched: vec![false; self.buffers] })).collect();
        Ok(LayerStream { tier: self.tier, buffers: self.buffers, blocks: self.blocks, shapes: self.shapes, dtype: self.dtype, devs: self.devs, block_dev: self.block_dev,
                         pools, #[cfg(feature = "cuda")] copy_streams, schedule: Mutex::new(Schedule::default()), stats: Mutex::new(StreamStats::default()),
                         pinned_bytes: self.pinned_bytes, pageable_bytes: self.pageable_bytes, shard_dir: self.shard_dir, notes })
    }
}

#[cfg(feature = "cuda")]
fn pin(w: &Tensor, cd: &candle_core::CudaDevice) -> Result<Pinned> {
    use candle_core::{CpuStorage, Storage};
    let ctx = cd.cuda_stream().context().clone();
    let (st, layout) = w.storage_and_layout();
    let (start, n) = (layout.start_offset(), w.elem_count());
    macro_rules! fill {
        ($v:expr, $ty:ty, $variant:ident) => {{
            let mut p = unsafe { ctx.alloc_pinned::<$ty>(n) }.map_err(|e| anyhow!("cuMemHostAlloc of {} bytes: {e}", n * std::mem::size_of::<$ty>()))?;
            p.as_mut_slice().map_err(|e| anyhow!("{e}"))?.copy_from_slice(&$v[start..start + n]);
            Pinned::$variant(p)
        }};
    }
    Ok(match &*st {
        Storage::Cpu(CpuStorage::BF16(v)) => fill!(v, half::bf16, BF16),
        Storage::Cpu(CpuStorage::F16(v)) => fill!(v, half::f16, F16),
        Storage::Cpu(CpuStorage::F32(v)) => fill!(v, f32, F32),
        _ => bail!("only CPU f32, f16 and bf16 matrices are page-locked"),
    })
}

impl LayerStream {
    pub fn device_of(&self, block: usize) -> &Device {
        &self.devs[self.block_dev[block]]
    }

    pub fn stats(&self) -> StreamStats {
        self.stats.lock().unwrap().clone()
    }

    /// The bytes one block streams.
    pub fn block_bytes(&self) -> u64 {
        self.shapes.iter().map(|(o, i)| (o * i) as u64).sum::<u64>() * self.dtype.size_in_bytes() as u64
    }

    pub fn has_block(&self, i: usize) -> bool {
        self.blocks.get(i).is_some_and(Option::is_some)
    }

    /// Copy block `i`'s matrices to its device now.
    fn load(&self, i: usize) -> Result<Vec<Tensor>> {
        let d = self.block_dev[i];
        let dev = &self.devs[d];
        let out = match self.blocks.get(i).and_then(Option::as_ref).ok_or_else(|| anyhow!("block {i} is not in the layer stream"))? {
            HostBlock::Ram(ts) => ts.iter().map(|h| self.upload(h, d)).collect::<Result<Vec<_>>>()?,
            HostBlock::Disk(path) => {
                let mut map = candle_core::safetensors::load(path, &Device::Cpu).with_context(|| format!("layer-stream shard {}", path.display()))?;
                Target::ALL.iter().zip(&self.shapes).map(|(t, (o, inp))| {
                    let w = map.remove(t.as_str()).ok_or_else(|| anyhow!("{}: no {}", path.display(), t.as_str()))?;
                    if w.dims() != [*o, *inp] || w.dtype() != self.dtype {
                        bail!("{}: {} is {:?} {:?}, the model needs [{o}, {inp}] {:?} (delete the cache folder)", path.display(), t.as_str(), w.dims(), w.dtype(), self.dtype);
                    }
                    self.upload(&HostTensor::Cpu(w), d)
                }).collect::<Result<Vec<_>>>()?
            }
        };
        let _ = dev;
        let mut s = self.stats.lock().unwrap();
        s.loads += 1;
        s.bytes += self.block_bytes();
        Ok(out)
    }

    fn upload(&self, h: &HostTensor, d: usize) -> Result<Tensor> {
        let dev = &self.devs[d];
        match h {
            // a fresh copy on the CPU (the same path a GPU takes, minus the transfer), a plain upload elsewhere
            HostTensor::Cpu(t) => Ok(if dev.is_cpu() { t.copy()? } else { t.to_device(dev)? }),
            #[cfg(feature = "cuda")]
            HostTensor::Pinned(p, dims) => {
                use candle_core::{op::BackpropOp, CudaStorage, Storage};
                let Device::Cuda(cd) = dev else { bail!("a page-locked block on a device that is not CUDA") };
                let stream = self.copy_streams[d].clone().unwrap_or_else(|| cd.cuda_stream());
                let storage = match p {
                    Pinned::BF16(p) => CudaStorage::wrap_cuda_slice(stream.clone_htod(p).map_err(|e| anyhow!("upload: {e}"))?, cd.clone()),
                    Pinned::F16(p) => CudaStorage::wrap_cuda_slice(stream.clone_htod(p).map_err(|e| anyhow!("upload: {e}"))?, cd.clone()),
                    Pinned::F32(p) => CudaStorage::wrap_cuda_slice(stream.clone_htod(p).map_err(|e| anyhow!("upload: {e}"))?, cd.clone()),
                };
                Ok(Tensor::from_storage(Storage::Cuda(storage), dims.clone(), BackpropOp::none(), false))
            }
        }
    }

    /// Block `i`'s frozen matrices on its device, for a run of it: from its buffer when it is resident (prefetched or
    /// still there), else loaded now into the least recently used one.
    pub fn acquire(&self, i: usize) -> Result<Vec<Tensor>> {
        let d = self.block_dev[i];
        let mut pool = self.pools[d].lock().unwrap();
        pool.clock += 1;
        let clock = pool.clock;
        if let Some(s) = pool.find(i) {
            if std::mem::take(&mut pool.prefetched[s]) {
                self.stats.lock().unwrap().prefetch_hits += 1;
            }
            let slot = pool.slots[s].as_mut().unwrap();
            slot.2 = clock;
            let w = slot.1.clone();
            pool.current = Some(s);
            return Ok(w);
        }
        let s = pool.victim(None).ok_or_else(|| anyhow!("no stream buffer"))?;
        pool.slots[s] = None;
        let w = self.load(i)?;
        self.stats.lock().unwrap().demand_loads += 1;
        pool.slots[s] = Some((i, w.clone(), clock));
        pool.prefetched[s] = false;
        pool.current = Some(s);
        Ok(w)
    }

    /// Start copying block `i` to its device into a buffer other than the one in use, unless it is resident.
    pub fn prefetch(&self, i: usize) -> Result<()> {
        if !self.has_block(i) {
            return Ok(());
        }
        let d = self.block_dev[i];
        let mut pool = self.pools[d].lock().unwrap();
        if pool.find(i).is_some() {
            return Ok(());
        }
        let keep = pool.current;
        let Some(s) = pool.victim(keep) else { return Ok(()) };
        pool.slots[s] = None;
        let w = self.load(i)?;
        pool.clock += 1;
        let clock = pool.clock;
        pool.slots[s] = Some((i, w, clock));
        pool.prefetched[s] = true;
        Ok(())
    }

    /// A forward pass is about to run the blocks in `order`; returns its generation (a replay from an older pass does
    /// not prefetch by this one's order).
    pub fn begin(&self, order: Vec<usize>) -> u64 {
        let mut s = self.schedule.lock().unwrap();
        s.replayed = vec![false; order.len()];
        s.order = order;
        s.generation += 1;
        s.generation
    }

    /// The forward pass ran position `pos`: prefetch the next block in the order.
    pub fn after_forward(&self, pos: usize) -> Result<()> {
        let next = self.schedule.lock().unwrap().order.get(pos + 1).copied();
        match next {
            Some(n) => self.prefetch(n),
            None => Ok(()),
        }
    }

    /// Position `pos` was run through a checkpoint node: its replay comes in the backward pass.
    pub fn mark_replayed(&self, pos: usize) {
        if let Some(r) = self.schedule.lock().unwrap().replayed.get_mut(pos) {
            *r = true;
        }
    }

    /// The replay of position `pos` (of pass `generation`) started: prefetch the block replayed after it, the last
    /// checkpointed position before it.
    pub fn after_replay(&self, pos: usize, generation: u64) -> Result<()> {
        let next = {
            let s = self.schedule.lock().unwrap();
            if s.generation != generation { None } else { (0..pos.min(s.order.len())).rev().find(|q| s.replayed[*q]).map(|q| s.order[q]) }
        };
        match next {
            Some(n) => self.prefetch(n),
            None => Ok(()),
        }
    }

    /// Drop every buffer (between phases, or before the model is dropped).
    pub fn clear(&self) {
        for p in &self.pools {
            let mut p = p.lock().unwrap();
            for s in p.slots.iter_mut() {
                *s = None;
            }
            p.current = None;
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// the transfer estimate (Soup's arithmetic: each streamed block is read twice per step, forward and replay; the time is
// the bytes over the link's bandwidth)
// ---------------------------------------------------------------------------------------------------------------------

/// A PCIe link's usable host-to-device rate in bytes/s from its sysfs speed (`16.0 GT/s PCIe`) and width, at
/// [`PCIE_EFFICIENCY`]: 128b/130b coding from 8 GT/s, 8b/10b below.
pub fn link_bytes_s(gt_s: f64, width: u32) -> f64 {
    let coding = if gt_s >= 8.0 { 128.0 / 130.0 } else { 0.8 };
    gt_s * 1e9 * coding * width as f64 / 8.0 * PCIE_EFFICIENCY
}

/// Each NVIDIA GPU's host-to-device rate as this machine's sysfs describes its link (in the order of their PCI
/// addresses, which is nvidia-smi's default order): the link's current speed (its highest when the card idles at
/// 2.5 GT/s) times its current width, and [`USB4_TUNNEL_BYTES_S`] at most for a card behind a USB4 / Thunderbolt tunnel (its
/// `removable` attribute). Empty off Linux.
pub fn gpu_link_rates() -> Vec<(String, f64)> {
    let mut out = Vec::new();
    let Ok(rd) = std::fs::read_dir("/sys/bus/pci/devices") else { return out };
    let mut devs: Vec<PathBuf> = rd.flatten().map(|e| e.path()).collect();
    devs.sort();
    for d in devs {
        let read = |f: &str| std::fs::read_to_string(d.join(f)).map(|s| s.trim().to_string()).unwrap_or_default();
        if read("vendor") != "0x10de" || !read("class").starts_with("0x03") {
            continue;
        }
        let gt = |f: &str| read(f).split_whitespace().next().and_then(|v| v.parse::<f64>().ok()).unwrap_or(0.0);
        // the current speed, unless it is 2.5 GT/s: an idle card trains its link down to that, so then its highest
        let (cur, max) = (gt("current_link_speed"), gt("max_link_speed"));
        let speed = if cur > 2.5 { cur } else { max.max(cur) };
        let width: u32 = read("current_link_width").parse().unwrap_or(0);
        let mut rate = if speed > 0.0 && width > 0 { link_bytes_s(speed, width) } else { DEFAULT_LINK_BYTES_S };
        if read("removable") == "removable" {
            rate = rate.min(USB4_TUNNEL_BYTES_S);
        }
        out.push((d.file_name().and_then(|n| n.to_str()).unwrap_or("").to_string(), rate));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiers_buffers_and_links() {
        assert!(validate_buffers(1).is_err() && validate_buffers(9).is_err() && validate_buffers(2).is_ok() && validate_buffers(8).is_ok());
        assert_eq!(choose_tier(StreamMode::Auto, 10, 100).0, Tier::Ram);
        let (t, note) = choose_tier(StreamMode::Auto, 80, 100);
        assert!(t == Tier::Disk && note.is_some());
        assert_eq!(choose_tier(StreamMode::Ram, 1000, 1).0, Tier::Ram);
        assert_eq!(StreamMode::from_s("disk").unwrap(), StreamMode::Disk);
        assert!(StreamMode::from_s("nvme").is_err());
        // PCIe 4.0 x4: 16 GT/s x 4 lanes x 128/130 / 8 = 7.88 GB/s raw, 6.3 GB/s at 80 %
        assert!((link_bytes_s(16.0, 4) - 6.30e9).abs() < 0.01e9);
        assert!((link_bytes_s(2.5, 2) - 0.4e9).abs() < 0.01e9);
    }
}
