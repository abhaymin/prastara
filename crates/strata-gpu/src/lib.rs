//! strata-gpu: Prastara's GPU layer, in Rust (CubeCL) - see docs/GPU-PLAN.md.
//!
//! - [`list`] the GPUs every runtime here can use: CUDA (NVIDIA), HIP (AMD ROCm) and wgpu (Vulkan: NVIDIA, AMD and
//!   Intel, integrated ones included), and the CPU runtime;
//! - [`Gpu::open`] one of them by name ("cuda:0", "hip:0", "vulkan:discrete", "vulkan:integrated", "cpu");
//! - weights uploaded in their GGUF encoding ([`Gpu::upload`]) and multiplied there ([`Gpu::matvec`]).
//!
//! One source of kernels (kernels.rs) for every backend; the C/C++ under it is only the vendors' runtimes.

pub mod kernels;
#[cfg(target_os = "macos")]
pub mod native;

/// What the devices were asked to do, counted for every device since the process started (AM2_PROFILE in the engine
/// reads it per token): kernel launches, host syncs (a read that waits for the device), bytes each way, and the time
/// the host spent waiting in those reads. Relaxed atomics: a few nanoseconds against a kernel launch's microseconds.
pub mod stats {
    use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
    static LAUNCHES: AtomicU64 = AtomicU64::new(0);
    static SYNCS: AtomicU64 = AtomicU64::new(0);
    static D2H: AtomicU64 = AtomicU64::new(0);
    static H2D: AtomicU64 = AtomicU64::new(0);
    static H2D_COPIES: AtomicU64 = AtomicU64::new(0);
    static WAIT_NS: AtomicU64 = AtomicU64::new(0);

    #[derive(Clone, Copy, Default, Debug, PartialEq)]
    pub struct Stats {
        pub launches: u64,
        pub syncs: u64,
        pub d2h_bytes: u64,
        pub h2d_bytes: u64,
        pub h2d_copies: u64,
        pub wait_ns: u64,
    }

    impl std::ops::Sub for Stats {
        type Output = Stats;
        fn sub(self, o: Stats) -> Stats {
            Stats { launches: self.launches - o.launches, syncs: self.syncs - o.syncs, d2h_bytes: self.d2h_bytes - o.d2h_bytes,
                    h2d_bytes: self.h2d_bytes - o.h2d_bytes, h2d_copies: self.h2d_copies - o.h2d_copies, wait_ns: self.wait_ns - o.wait_ns }
        }
    }

    pub fn snapshot() -> Stats {
        Stats { launches: LAUNCHES.load(Relaxed), syncs: SYNCS.load(Relaxed), d2h_bytes: D2H.load(Relaxed), h2d_bytes: H2D.load(Relaxed),
                h2d_copies: H2D_COPIES.load(Relaxed), wait_ns: WAIT_NS.load(Relaxed) }
    }

    pub(crate) fn launch(n: u64) {
        LAUNCHES.fetch_add(n, Relaxed);
    }

    pub(crate) fn h2d(bytes: usize) {
        H2D.fetch_add(bytes as u64, Relaxed);
        H2D_COPIES.fetch_add(1, Relaxed);
    }

    /// A read back: it waited `t0.elapsed()` (the device's queue, then the copy).
    pub(crate) fn d2h(bytes: usize, t0: std::time::Instant) {
        SYNCS.fetch_add(1, Relaxed);
        D2H.fetch_add(bytes as u64, Relaxed);
        WAIT_NS.fetch_add(t0.elapsed().as_nanos() as u64, Relaxed);
    }
}

use cubecl::prelude::*;
use cubecl::server::Handle;
use std::panic::{catch_unwind, AssertUnwindSafe};

/// The runtimes, as `Gpu` holds them.
enum Client {
    #[cfg(feature = "cuda")]
    Cuda(ComputeClient<cubecl::cuda::CudaRuntime>),
    #[cfg(feature = "hip")]
    Hip(ComputeClient<cubecl::hip::HipRuntime>),
    #[cfg(feature = "wgpu")]
    Wgpu(ComputeClient<cubecl::wgpu::WgpuRuntime>),
    Cpu(ComputeClient<cubecl::cpu::CpuRuntime>),
    /// the native Metal backend (native.rs), for `metal:*` when AM2_METAL_NATIVE=1
    #[cfg(target_os = "macos")]
    Metal(native::Native),
}

/// Where a buffer or a matrix lives: a CubeCL handle, or a native Metal buffer.
#[derive(Clone)]
enum Mem {
    Cube(Handle),
    #[cfg(target_os = "macos")]
    Mtl(std::sync::Arc<native::MBuf>),
}

impl Mem {
    fn cube(&self) -> Handle {
        match self {
            Mem::Cube(h) => h.clone(),
            #[cfg(target_os = "macos")]
            Mem::Mtl(_) => panic!("a native Metal buffer given to a CubeCL kernel"),
        }
    }

    #[cfg(target_os = "macos")]
    fn mtl(&self) -> &native::MBuf {
        match self {
            Mem::Mtl(b) => b,
            Mem::Cube(_) => panic!("a CubeCL buffer given to a native Metal kernel"),
        }
    }
}

/// A device: its name as `Gpu::open` takes it, and what it is.
#[derive(Clone, Debug)]
pub struct DeviceInfo {
    pub name: String,
    pub backend: &'static str,
    pub detail: String,
}

/// One weight matrix on the device, in its GGUF encoding.
pub struct Weights {
    handle: Mem,
    pub rows: u32,
    pub cols: u32,
    row_bytes: u32,
    geometry: (u32, u32, u32),
    pub bytes: usize,
    /// whether matvec_q4k's shared memory (x, its group sums and the cube's partial sums) fits this device: Metal
    /// allows 32 KiB, so a 9216-column Q4_K matrix (a 3B model's ffn_down) must take the plain kernel there
    wide_fits: bool,
    /// a Q8_0 matrix on CubeCL is repacked (kernels.rs, the fast products): `handle` holds its values, rows * cols
    /// bytes, and this its f16 scales
    scales: Option<Mem>,
    /// whether the device's planes are 32 units wide, so the fast kernels sum a row's warp with a plane reduction
    plane: bool,
}

impl Weights {
    /// Whether the fast kernels take this matrix (kernels.rs, the fast products): Q8_0 (repacked) and Q4_K on CubeCL.
    pub fn fast(&self) -> bool {
        self.scales.is_some() || (self.geometry.2 == kernels::kind::Q4_K && matches!(self.handle, Mem::Cube(_))
                                  && FAST_Q4K.load(std::sync::atomic::Ordering::Relaxed))
    }

    /// The values as 16-byte loads, for the fast kernels.
    fn v4<R: Runtime>(&self) -> ArrayArg<R> {
        unsafe { ArrayArg::from_raw_parts(self.handle.cube(), self.value_words() / 4) }
    }

    /// A repacked Q8_0 matrix's scales (anything for the other kinds: the kernels do not read it).
    fn sc<R: Runtime>(&self) -> ArrayArg<R> {
        unsafe { ArrayArg::from_raw_parts(self.scales.as_ref().unwrap_or(&self.handle).cube(), 1) }
    }

    /// The words of `handle`: a repacked Q8_0 matrix's values, or the whole matrix.
    fn value_words(&self) -> usize {
        if self.scales.is_some() { self.rows as usize * self.cols as usize / 4 } else { self.bytes.div_ceil(4) }
    }
}

/// Values on the device, so that a chain of kernels needs no trip through the CPU.
#[derive(Clone)]
pub struct Buf {
    handle: Mem,
    pub len: usize,
}

impl Buf {
    fn arg<R: Runtime>(&self) -> ArrayArg<R> {
        unsafe { ArrayArg::from_raw_parts(self.handle.cube(), self.len) }
    }
}

pub struct Gpu {
    client: Client,
    pub name: String,
}

macro_rules! with_client {
    ($self:expr, $c:ident, $R:ident, $body:expr) => {
        match &$self.client {
            #[cfg(feature = "cuda")]
            Client::Cuda($c) => { #[allow(dead_code)] type $R = cubecl::cuda::CudaRuntime; $body }
            #[cfg(feature = "hip")]
            Client::Hip($c) => { #[allow(dead_code)] type $R = cubecl::hip::HipRuntime; $body }
            #[cfg(feature = "wgpu")]
            Client::Wgpu($c) => { #[allow(dead_code)] type $R = cubecl::wgpu::WgpuRuntime; $body }
            Client::Cpu($c) => { #[allow(dead_code)] type $R = cubecl::cpu::CpuRuntime; $body }
            #[cfg(target_os = "macos")]
            Client::Metal(_) => unreachable!("native Metal is handled before with_client!"),
        }
    };
}

/// Runs `$body` with `$n` the native Metal backend when that is this device's client, and returns its value.
macro_rules! native {
    ($self:expr, $n:ident, $body:expr) => {
        #[cfg(target_os = "macos")]
        if let Client::Metal($n) = &$self.client {
            return $body;
        }
    };
}

/// From this many rows a Q4_K or Q6_K product takes `matvec_q4k` / `matvec_q6k` (a variable so a test can force either kernel).
pub static WIDE_ROWS: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(kernels::WIDE_ROWS);

/// From this many tokens a Q8_0 or Q4_K product of a chunk takes the tiled kernel (`kernels::mm`), below it the decode
/// kernel once per token (a variable so a test can force either kernel).
pub static MM_TOKENS: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(8);
/// Whether Q4_K products take the fast decode kernel (`kernels::mv_q4_k`); off, the older kernels (for comparisons).
pub static FAST_Q4K: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

/// Queues y[y_off..] (+)= W x for `tokens` vectors with a fast kernel (Weights::fast): the tiled one for a chunk, else
/// the decode one (once per vector). `ylen`: the values in y's buffer.
macro_rules! fast {
    ($c:expr, $R:ident, $w:expr, $hx:expr, $n:expr, $hy:expr, $ylen:expr, $tokens:expr, $y_off:expr, $add:expr) => {{
        let w: &Weights = $w;
        let k = w.geometry.2;
        stats::launch(1);
        let tiled = $tokens >= MM_TOKENS.load(std::sync::atomic::Ordering::Relaxed) && w.cols % kernels::MM_K == 0;
        let words = w.value_words();
        unsafe {
            let y = ArrayArg::from_raw_parts($hy, $ylen);
            let x4 = ArrayArg::from_raw_parts($hx, $n / 4);
            let wd = ArrayArg::from_raw_parts(w.scales.as_ref().unwrap_or(&w.handle).cube(), 1);
            if tiled {
                let count = CubeCount::Static(w.rows.div_ceil(kernels::MM_M), ($tokens as u32).div_ceil(kernels::MM_N), 1);
                kernels::mm::launch_unchecked::<$R>($c, count, CubeDim::new_1d(kernels::MM_CUBE), ArrayArg::from_raw_parts(w.handle.cube(), words),
                    ArrayArg::from_raw_parts(w.handle.cube(), words / 4), wd, x4, y, w.rows, w.cols, $tokens as u32, $y_off, k, $add);
            } else {
                kernels::mv_fast::launch_unchecked::<$R>($c, mv_count(w.rows, 0, 0, $tokens as u32), CubeDim::new_1d(kernels::MV_ROWS * 32),
                    ArrayArg::from_raw_parts(w.handle.cube(), words / 4), wd, x4, y, w.rows, w.cols, $y_off, k, w.plane, $add);
            }
        }
    }};
}

/// The grid of the fast decode kernels for these rows (of up to three matrices) and tokens: x and y (each axis has a
/// limit of 65535 on some backends), z the tokens.
fn mv_count(r0: u32, r1: u32, r2: u32, tokens: u32) -> CubeCount {
    let cubes = r0.div_ceil(kernels::MV_ROWS) + r1.div_ceil(kernels::MV_ROWS) + r2.div_ceil(kernels::MV_ROWS);
    let (cx, cy) = (cubes.min(65535), cubes.div_ceil(cubes.min(65535)));
    CubeCount::Static(cx, cy, tokens)
}

/// Queues y = W x for `tokens` vectors with the kernel that suits the matrix.
macro_rules! product {
    ($c:expr, $R:ident, $w:expr, $hx:expr, $n:expr, $hy:expr, $tokens:expr) => {{
        let w = $w;
        let (be, bb, k) = w.geometry;
        if w.fast() {
            fast!($c, $R, w, $hx, $n, $hy, $tokens as usize * w.rows as usize, $tokens, 0u32, false);
        } else {
        let wide = (k == kernels::kind::Q4_K || k == kernels::kind::Q6_K) && w.wide_fits
            && w.rows >= WIDE_ROWS.load(std::sync::atomic::Ordering::Relaxed);
        // cubes over x and y of the grid (each axis has a limit of 65535 on some backends)
        let cubes = w.rows.div_ceil(if wide { 32 } else { kernels::ROWS_PER_CUBE });
        let (cx, cy) = (cubes.min(65535), cubes.div_ceil(cubes.min(65535)));
        let count = CubeCount::Static(cx, cy, $tokens);
        stats::launch(1);
        unsafe {
            let y = ArrayArg::from_raw_parts($hy, $tokens as usize * w.rows as usize);
            let (wa, xa) = (ArrayArg::from_raw_parts(w.handle.cube(), w.bytes.div_ceil(4)), ArrayArg::from_raw_parts($hx, $n));
            if wide && k == kernels::kind::Q6_K {
                kernels::matvec_q6k::launch_unchecked::<$R>($c, count, CubeDim::new_1d(kernels::WIDE_CUBE), wa, xa, y, w.rows, w.row_bytes, w.cols);
            } else if wide {
                kernels::matvec_q4k::launch_unchecked::<$R>($c, count, CubeDim::new_1d(kernels::WIDE_CUBE), wa, xa, y, w.rows, w.row_bytes, w.cols);
            } else {
                kernels::matvec::launch_unchecked::<$R>($c, count, CubeDim::new_1d(kernels::CUBE), wa, xa, y, w.rows, w.cols, w.row_bytes, be, bb, k);
            }
        }
        }
    }};
}

/// A runtime that works here: a tiny kernel runs and gives the right answer (a missing driver or runtime library
/// panics inside CubeCL; that is caught and the device left out).
fn works<R: Runtime>(dev: R::Device) -> Option<ComputeClient<R>> {
    let prev = std::panic::take_hook();
    let loud = std::env::var_os("STRATA_GPU_DEBUG").is_some();
    std::panic::set_hook(Box::new(move |i| if loud { eprintln!("strata-gpu: {i}") }));
    let r = catch_unwind(AssertUnwindSafe(|| {
        let client = R::client(&dev);
        let x: Vec<f32> = (0..32).map(|i| i as f32).collect();
        let w: Vec<u32> = (0..32).map(|_| 1.0f32.to_bits()).collect();
        let (hw, hx, hy) = (client.create_from_slice(u32::as_bytes(&w)), client.create_from_slice(f32::as_bytes(&x)), client.empty(4));
        unsafe {
            kernels::matvec::launch_unchecked::<R>(&client, CubeCount::Static(1, 1, 1), CubeDim::new_1d(kernels::CUBE),
                ArrayArg::from_raw_parts(hw, 32), ArrayArg::from_raw_parts(hx, 32), ArrayArg::from_raw_parts(hy.clone(), 1),
                1u32, 32u32, 128u32, 32, 128, kernels::kind::F32);
        }
        let y = f32::from_bytes(&client.read_one_unchecked(hy))[0];
        if loud { eprintln!("strata-gpu: self-check gave {y}") }
        (y == 496.0).then_some(client)
    }));
    std::panic::set_hook(prev);
    r.ok().flatten()
}

/// What the wgpu runtime's name is on this platform (findings item 2): Metal on macOS/iOS, where wgpu uses it, else
/// Vulkan. Device names are "{backend}:{subtype}", so `--list-devices` suggests exactly what "--devices" should take.
#[cfg(feature = "wgpu")]
fn wgpu_backend() -> &'static str {
    match std::env::consts::OS { "macos" | "ios" => "metal", _ => "vulkan" }
}

/// Every device a runtime here can use.
pub fn list() -> Vec<DeviceInfo> {
    let mut out = Vec::new();
    #[cfg(feature = "cuda")]
    for i in 0..8 {
        if works::<cubecl::cuda::CudaRuntime>(cubecl::cuda::CudaDevice { index: i }).is_none() {
            break;
        }
        out.push(DeviceInfo { name: format!("cuda:{i}"), backend: "cuda", detail: "NVIDIA (CUDA)".into() });
    }
    #[cfg(feature = "hip")]
    for i in 0..8 {
        if works::<cubecl::hip::HipRuntime>(cubecl::hip::AmdDevice { index: i }).is_none() {
            break;
        }
        out.push(DeviceInfo { name: format!("hip:{i}"), backend: "hip", detail: "AMD (ROCm/HIP)".into() });
    }
    #[cfg(feature = "wgpu")]
    {
        use cubecl::wgpu::WgpuDevice;
        let backend = wgpu_backend();
        let kind = |s: &str| format!("{backend}:{s}");
        for (n, d, what) in [(kind("discrete"), WgpuDevice::DiscreteGpu(0), "a discrete GPU"),
                             (kind("integrated"), WgpuDevice::IntegratedGpu(0), "an integrated GPU")] {
            if works::<cubecl::wgpu::WgpuRuntime>(d).is_some() {
                out.push(DeviceInfo { name: n, backend, detail: what.into() });
            }
        }
    }
    out.push(DeviceInfo { name: "cpu".into(), backend: "cpu", detail: "the CPU (CubeCL's CPU runtime)".into() });
    out
}

impl Gpu {
    /// Open a device by its name from [`list`].
    pub fn open(name: &str) -> Result<Gpu, String> {
        let (kind, idx) = name.split_once(':').unwrap_or((name, "0"));
        let i: usize = idx.parse().unwrap_or(0);
        let client = match kind {
            #[cfg(feature = "cuda")]
            "cuda" => works::<cubecl::cuda::CudaRuntime>(cubecl::cuda::CudaDevice { index: i }).map(Client::Cuda),
            #[cfg(feature = "hip")]
            "hip" | "rocm" => works::<cubecl::hip::HipRuntime>(cubecl::hip::AmdDevice { index: i }).map(Client::Hip),
            #[cfg(feature = "wgpu")]
            // "metal" is what macOS names it (see wgpu_backend); "vulkan"/"wgpu" stay accepted for older configs.
            #[cfg(target_os = "macos")]
            "metal" if native::wanted() => {
                return native::Native::open().map(|n| Gpu { client: Client::Metal(n), name: name.to_string() });
            }
            #[cfg(feature = "wgpu")]
            "metal" | "vulkan" | "wgpu" => {
                use cubecl::wgpu::WgpuDevice;
                let d = match idx {
                    "integrated" | "igpu" => WgpuDevice::IntegratedGpu(0),
                    "discrete" | "dgpu" => WgpuDevice::DiscreteGpu(0),
                    _ => WgpuDevice::DiscreteGpu(i),
                };
                works::<cubecl::wgpu::WgpuRuntime>(d).map(Client::Wgpu)
            }
            "cpu" => works::<cubecl::cpu::CpuRuntime>(cubecl::cpu::CpuDevice).map(Client::Cpu),
            _ => return Err(format!("unknown device {name:?}: cuda:N, hip:N, metal:discrete, metal:integrated (vulkan still accepted) or cpu")),
        };
        client.map(|client| Gpu { client, name: name.to_string() }).ok_or_else(|| format!("{name} is not usable here"))
    }

    /// Upload a `rows` x `cols` matrix stored as GGUF type `ty` (its bytes as they are in the file).
    pub fn upload(&self, ty: &str, data: &[u8], rows: usize, cols: usize) -> Result<Weights, String> {
        let geometry = kernels::geometry(ty).ok_or_else(|| format!("no GPU kernel for {ty} weights yet"))?;
        let (be, bb, _) = geometry;
        if cols % be as usize != 0 {
            return Err(format!("{cols} columns are not whole {ty} blocks"));
        }
        let row_bytes = cols / be as usize * bb as usize;
        if data.len() != rows * row_bytes {
            return Err(format!("{} bytes for a {rows}x{cols} {ty} matrix (want {})", data.len(), rows * row_bytes));
        }
        stats::h2d(data.len());
        native!(self, n, {
            let t = kernels::table_for(ty);
            let all: Vec<u8> = t.iter().chain(data).copied().collect();
            Ok(Weights { handle: Mem::Mtl(n.upload(&all)), rows: rows as u32, cols: cols as u32, row_bytes: row_bytes as u32,
                         geometry, bytes: all.len(), wide_fits: false, scales: None, plane: false })
        });
        // whole u32 words: the kind's lookup table first (IQ3_XXS's grid), then the rows, the tail padded with zeros.
        // A matrix with no table and whole words (every K-quant and float one) goes as it is: the client makes the
        // one copy the device thread uploads from, so the mapped file is read once and nothing is rebuilt here.
        let table = kernels::table_for(ty);
        let (plane, shared) = with_client!(self, c, R, {
            let p = c.properties();
            (p.hardware.plane_size_min == 32 && p.hardware.plane_size_max == 32 && p.features.plane.contains(cubecl::ir::features::Plane::Ops),
             p.hardware.max_shared_memory_size)
        });
        if geometry.2 == kernels::kind::Q8_0 {
            // the values first, then the scales (kernels.rs): 16-byte rows the fast kernels read in 16-byte loads
            let blocks = data.len() / 34;
            let mut qs = Vec::with_capacity(blocks * 32);
            let mut ds = Vec::with_capacity(blocks * 2 + 2);
            for b in data.chunks_exact(34) {
                ds.extend_from_slice(&b[..2]);
                qs.extend_from_slice(&b[2..]);
            }
            ds.resize(ds.len().div_ceil(4) * 4, 0);
            let (hq, hd) = with_client!(self, c, R, (c.create_from_slice(&qs), c.create_from_slice(&ds)));
            return Ok(Weights { handle: Mem::Cube(hq), rows: rows as u32, cols: cols as u32, row_bytes: row_bytes as u32, geometry,
                                bytes: data.len(), wide_fits: false, scales: Some(Mem::Cube(hd)), plane });
        }
        let handle = if table.is_empty() && data.len() % 4 == 0 {
            Mem::Cube(with_client!(self, c, R, c.create_from_slice(data)))
        } else {
            let padded = (table.len() + data.len()).div_ceil(4) * 4;
            let mut all = Vec::with_capacity(padded);
            all.extend_from_slice(&table);
            all.extend_from_slice(data);
            all.resize(padded, 0);
            Mem::Cube(with_client!(self, c, R, c.create_from_slice(&all)))
        };
        let data_len = table.len() + data.len();
        let need = match geometry.2 {
            kernels::kind::Q6_K => kernels::wide_q6k_shared_bytes(cols as u32),
            _ => kernels::wide_shared_bytes(cols as u32),
        };
        let wide_fits = need <= shared;
        Ok(Weights { handle, rows: rows as u32, cols: cols as u32, row_bytes: row_bytes as u32, geometry, bytes: data_len, wide_fits, scales: None, plane })
    }

    /// Let go of the host copies of uploads the device has finished reading. The runtime keeps each upload's bytes
    /// until its queue is flushed, and frees them one flush later (it waits for the device first), so without this
    /// a model's whole upload stays in RAM beside its mapped file until the first kernels run.
    pub fn flush(&self) -> Result<(), String> {
        native!(self, _n, Ok(()));
        with_client!(self, c, R, c.flush().map_err(|e| format!("{}: {e:?}", self.name)))
    }

    /// Wait until the device has done everything queued so far (uploads included).
    pub fn sync(&self) -> Result<(), String> {
        native!(self, _n, Ok(()));
        with_client!(self, c, R, cubecl::future::block_on(c.sync()).map_err(|e| format!("{}: {e:?}", self.name)))
    }

    /// The products of several matrices with the same input (q, k, v; gate and up): one upload of x, the kernels
    /// launched back to back, then the results read (only the first read waits for the device).
    pub fn matvec_many(&self, ws: &[&Weights], x: &[f32]) -> Vec<Vec<f32>> {
        if ws.is_empty() {
            return Vec::new();
        }
        let cols = ws[0].cols;
        debug_assert!(ws.iter().all(|w| w.cols == cols));
        let tokens = (x.len() / cols as usize).max(1) as u32;
        native!(self, n, {
            let hx = self.buffer(x);
            let outs: Vec<Buf> = ws.iter().map(|w| self.mv_tokens(w, &hx, tokens)).collect();
            let _ = n;
            outs.iter().map(|b| self.read(b)).collect()
        });
        with_client!(self, c, R, {
            stats::h2d(x.len() * 4);
            let hx = c.create_from_slice(f32::as_bytes(x));
            let outs: Vec<_> = ws.iter().map(|w| {
                let hy = c.empty(tokens as usize * w.rows as usize * 4);
                product!(c, R, *w, hx.clone(), x.len(), hy.clone(), tokens);
                hy
            }).collect();
            outs.into_iter().map(|hy| {
                let t0 = std::time::Instant::now();
                let v = f32::from_bytes(&c.read_one_unchecked(hy)).to_vec();
                stats::d2h(v.len() * 4, t0);
                v
            }).collect()
        })
    }

    /// A buffer of f32 on the device.
    pub fn buffer(&self, v: &[f32]) -> Buf {
        stats::h2d(v.len() * 4);
        native!(self, n, Buf { handle: Mem::Mtl(n.upload(f32::as_bytes(v))), len: v.len() });
        Buf { handle: Mem::Cube(with_client!(self, c, R, c.create_from_slice(f32::as_bytes(v)))), len: v.len() }
    }

    /// A K or V cache of `len` values: half on the native Metal backend (its attention kernels read half), f32 on CubeCL.
    pub fn kv_cache(&self, len: usize) -> Buf {
        // 1 MiB to spare: the matrix-unit attention reads whole blocks of 8 positions, past the context's end (masked)
        native!(self, n, Buf { handle: Mem::Mtl(n.empty(len * 2 + (1 << 20))), len });
        self.scratch(len)
    }

    /// Bytes per value of [`Gpu::kv_cache`].
    pub fn kv_bytes(&self) -> u64 {
        if self.is_native() { 2 } else { 4 }
    }

    /// An uninitialized buffer of `len` f32 on the device.
    pub fn scratch(&self, len: usize) -> Buf {
        native!(self, n, Buf { handle: Mem::Mtl(n.empty(len * 4)), len });
        Buf { handle: Mem::Cube(with_client!(self, c, R, c.empty(len * 4))), len }
    }

    /// Waits for the device and reads a buffer back. Only this waits: every call below just queues its kernel.
    pub fn read(&self, b: &Buf) -> Vec<f32> {
        let t0 = std::time::Instant::now();
        let v = (|| {
            native!(self, n, n.read(b.handle.mtl(), b.len));
            with_client!(self, c, R, f32::from_bytes(&c.read_one_unchecked(b.handle.cube())).to_vec())
        })();
        stats::d2h(b.len * 4, t0);
        v
    }

    /// y = W x, queued; y stays on the device.
    pub fn mv(&self, w: &Weights, x: &Buf) -> Buf {
        self.mv_tokens(w, x, 1)
    }

    /// y = W x for `tokens` vectors of `w.cols` values in x, queued; y (tokens * rows) stays on the device.
    pub fn mv_tokens(&self, w: &Weights, x: &Buf, tokens: u32) -> Buf {
        #[cfg(target_os = "macos")]
        if self.is_native() { stats::launch(1); }
        native!(self, n, Buf { handle: Mem::Mtl(n.mv(w.handle.mtl(), w.rows, w.cols, w.row_bytes, w.geometry.2, x.handle.mtl(), tokens)),
                               len: tokens as usize * w.rows as usize });
        with_client!(self, c, R, {
            let hy = c.empty(tokens as usize * w.rows as usize * 4);
            product!(c, R, w, x.handle.cube(), x.len, hy.clone(), tokens);
            Buf { handle: Mem::Cube(hy), len: tokens as usize * w.rows as usize }
        })
    }

    /// Whether a decode kernel (not the tiled one) takes `tokens` vectors of these fast matrices of one kind and width:
    /// the fused kernels below are decode kernels.
    fn fused(&self, ws: &[&Weights], tokens: u32) -> bool {
        tokens < MM_TOKENS.load(std::sync::atomic::Ordering::Relaxed)
            && ws.iter().all(|w| w.fast() && w.geometry.2 == ws[0].geometry.2 && w.cols == ws[0].cols)
    }

    /// y += W x for `tokens` vectors, queued: the residual added in the product's own kernel when it is a fast one.
    pub fn mv_add(&self, w: &Weights, x: &Buf, y: &Buf, tokens: u32) {
        if w.fast() {
            with_client!(self, c, R, fast!(c, R, w, x.handle.cube(), x.len, y.handle.cube(), y.len, tokens, 0u32, true));
            return;
        }
        self.add_into(&self.mv_tokens(w, x, tokens), y);
    }

    /// W x for `tokens` vectors written at y[off..], queued (V straight into its cache).
    pub fn mv_into(&self, w: &Weights, x: &Buf, y: &Buf, off: usize, tokens: u32) {
        if w.fast() {
            with_client!(self, c, R, fast!(c, R, w, x.handle.cube(), x.len, y.handle.cube(), y.len, tokens, off as u32, false));
            return;
        }
        self.kv_append(&self.mv_tokens(w, x, tokens), y, off);
    }

    /// silu(G x) * (U x) for `tokens` vectors, queued: one kernel for both products and the gate when it can.
    pub fn glu(&self, gate: &Weights, up: &Weights, x: &Buf, tokens: u32) -> Buf {
        if !self.fused(&[gate, up], tokens) || gate.rows != up.rows {
            let (u, g) = (self.mv_tokens(up, x, tokens), self.mv_tokens(gate, x, tokens));
            self.silu_gate(&g, &u);
            return u;
        }
        stats::launch(1);
        let y = self.scratch(tokens as usize * up.rows as usize);
        with_client!(self, c, R, unsafe {
            kernels::mv_glu::launch_unchecked::<R>(c, mv_count(up.rows, 0, 0, tokens), CubeDim::new_1d(kernels::MV_ROWS * 32), gate.v4(), gate.sc(),
                up.v4(), up.sc(), ArrayArg::from_raw_parts(x.handle.cube(), x.len / 4), y.arg(), up.rows, up.cols, up.geometry.2, up.plane);
        });
        y
    }

    /// Attention's three products of x for `tokens` vectors, queued: -> (q, k); v is written into `vc` at `v_off`. One
    /// kernel for the three when it can.
    #[allow(clippy::too_many_arguments)]
    pub fn qkv(&self, q: &Weights, k: &Weights, v: &Weights, x: &Buf, tokens: u32, vc: &Buf, v_off: usize) -> (Buf, Buf) {
        if !self.fused(&[q, k, v], tokens) {
            let (qy, ky) = (self.mv_tokens(q, x, tokens), self.mv_tokens(k, x, tokens));
            self.mv_into(v, x, vc, v_off, tokens);
            return (qy, ky);
        }
        stats::launch(1);
        let (qy, ky) = (self.scratch(tokens as usize * q.rows as usize), self.scratch(tokens as usize * k.rows as usize));
        with_client!(self, c, R, unsafe {
            kernels::mv3::launch_unchecked::<R>(c, mv_count(q.rows, k.rows, v.rows, tokens), CubeDim::new_1d(kernels::MV_ROWS * 32), q.v4(), q.sc(),
                k.v4(), k.sc(), v.v4(), v.sc(), ArrayArg::from_raw_parts(x.handle.cube(), x.len / 4), qy.arg(), ky.arg(), vc.arg(), q.rows, k.rows,
                v.rows, v_off as u32, q.cols, q.geometry.2, q.plane);
        });
        (qy, ky)
    }

    /// Attention's q and k after their products, for `tokens` tokens, queued: each head RMS-normed by `qn` / `kn` when
    /// given, rotated by `tab` (each token's (cos, sin) pairs), and k written into its cache `kc` at `k_off`. -> q. One
    /// kernel on CubeCL (heads of at most 512 values); the norm, rope and append kernels in turn elsewhere.
    #[allow(clippy::too_many_arguments)]
    pub fn qk_rope(&self, q: &Buf, k: &Buf, qn: Option<&Buf>, kn: Option<&Buf>, tab: &Buf, kc: &Buf, k_off: usize, hd: u32, n_rot: u32,
                   neox: bool, tokens: u32, eps: f32) -> Buf {
        if self.is_native() || hd > 512 || qn.is_some() != kn.is_some() {
            let (mut q, mut k) = (q.clone(), k.clone());
            if let Some(w) = qn {
                q = self.rms_norm(&q, w, eps);
            }
            if let Some(w) = kn {
                k = self.rms_norm(&k, w, eps);
            }
            if tokens == 1 {
                self.rope(tab, &q, hd, n_rot, neox);
                self.rope(tab, &k, hd, n_rot, neox);
            } else {
                self.rope_tokens(tab, &q, hd, n_rot, neox, tokens);
                self.rope_tokens(tab, &k, hd, n_rot, neox, tokens);
            }
            self.kv_append(&k, kc, k_off);
            return q;
        }
        stats::launch(1);
        let (n_head, n_kv) = (q.len as u32 / hd / tokens, k.len as u32 / hd / tokens);
        let norm = qn.is_some();
        let (qn, kn) = (qn.unwrap_or(tab), kn.unwrap_or(tab));
        with_client!(self, c, R, unsafe {
            kernels::qk_rope::launch_unchecked::<R>(c, CubeCount::Static(n_head + n_kv, tokens, 1), CubeDim::new_1d(kernels::CUBE), q.arg(), k.arg(),
                qn.arg(), kn.arg(), tab.arg(), kc.arg(), hd, n_rot, neox as u32, n_head, n_kv, k_off as u32, eps, norm);
        });
        q.clone()
    }

    /// y = x * w / sqrt(mean(x^2) + eps), queued.
    /// The norm of each row of `w.len` values: one vector, a chunk's tokens, or each head (q/k norms).
    pub fn rms_norm(&self, x: &Buf, w: &Buf, eps: f32) -> Buf {
        stats::launch(1);
        native!(self, n, Buf { handle: Mem::Mtl(n.rms_norm(x.handle.mtl(), w.handle.mtl(), w.len as u32, (x.len / w.len) as u32, eps)), len: x.len });
        let y = self.scratch(x.len);
        let rows = (x.len / w.len) as u32;
        with_client!(self, c, R, unsafe {
            kernels::rms_norm::launch_unchecked::<R>(c, CubeCount::Static(rows, 1, 1), CubeDim::new_1d(kernels::CUBE), x.arg(), w.arg(), y.arg(), w.len as u32, eps);
        });
        y
    }

    /// y += x, queued.
    pub fn add_into(&self, x: &Buf, y: &Buf) {
        stats::launch(1);
        native!(self, m, m.add_into(x.handle.mtl(), y.handle.mtl(), x.len as u32));
        let n = x.len as u32;
        with_client!(self, c, R, unsafe {
            kernels::add_into::launch_unchecked::<R>(c, CubeCount::Static(n.div_ceil(kernels::CUBE), 1, 1), CubeDim::new_1d(kernels::CUBE), x.arg(), y.arg(), n);
        });
    }

    /// up *= silu(gate), queued.
    pub fn silu_gate(&self, gate: &Buf, up: &Buf) {
        stats::launch(1);
        native!(self, m, m.silu_gate(gate.handle.mtl(), up.handle.mtl(), up.len as u32));
        let n = up.len as u32;
        with_client!(self, c, R, unsafe {
            kernels::silu_gate::launch_unchecked::<R>(c, CubeCount::Static(n.div_ceil(kernels::CUBE), 1, 1), CubeDim::new_1d(kernels::CUBE), gate.arg(), up.arg(), n);
        });
    }

    /// Rotates x, `hd` values per head, by a (cos, sin) table of `n_rot / 2` pairs, queued.
    pub fn rope(&self, tab: &Buf, x: &Buf, hd: u32, n_rot: u32, neox: bool) {
        stats::launch(1);
        let heads = x.len as u32 / hd;
        native!(self, m, m.rope(tab.handle.mtl(), x.handle.mtl(), heads, hd, n_rot, neox));
        self.rope_cube(tab, x, hd, n_rot, neox, heads, 1);
    }

    fn rope_cube(&self, tab: &Buf, x: &Buf, hd: u32, n_rot: u32, neox: bool, heads: u32, tokens: u32) {
        with_client!(self, c, R, unsafe {
            kernels::rope::launch_unchecked::<R>(c, CubeCount::Static(heads, tokens, 1), CubeDim::new_1d(kernels::CUBE), tab.arg(), x.arg(), hd, n_rot,
                                                 neox as u32, heads);
        });
    }

    /// cache[at..] = x, queued.
    pub fn kv_append(&self, x: &Buf, cache: &Buf, at: usize) {
        stats::launch(1);
        native!(self, m, m.kv_append(x.handle.mtl(), cache.handle.mtl(), x.len as u32, at as u32));
        let n = x.len as u32;
        with_client!(self, c, R, unsafe {
            kernels::kv_append::launch_unchecked::<R>(c, CubeCount::Static(n.div_ceil(kernels::CUBE), 1, 1), CubeDim::new_1d(kernels::CUBE), x.arg(), cache.arg(), at as u32, n);
        });
    }

    /// Attention of q (`n_head` heads of `hd`) over `n` positions of K and V (`n_kv` heads each), queued; -> n_head * hd
    /// values. `s` is scratch for n_head * n scores.
    #[allow(clippy::too_many_arguments)]
    pub fn attend(&self, q: &Buf, k: &Buf, v: &Buf, s: &Buf, n: u32, n_head: u32, n_kv: u32, hd: u32, scale: f32) -> Buf {
        if let Some(per) = self.flash(hd) {
            // flash decoding: runs of positions per KV head, then their merge
            stats::launch(2);
            // about 32 runs (at least 32 positions each), so that a short context and a long one both fill the card
            let len = n.div_ceil(32).div_ceil(32).max(1) * 32;
            let splits = n.div_ceil(len);
            let part = self.scratch((n_head * splits * (hd + 2)) as usize);
            let o = self.scratch((n_head * hd) as usize);
            let warps = (n_head / n_kv).clamp(1, 8);
            with_client!(self, c, R, unsafe {
                kernels::attend_split::launch_unchecked::<R>(c, CubeCount::Static(n_kv, splits, 1), CubeDim::new_1d(32 * warps), q.arg(), k.arg(), v.arg(),
                    part.arg(), n, n_kv, n_head / n_kv, scale, len, per);
                kernels::attend_merge::launch_unchecked::<R>(c, CubeCount::Static(n_head, 1, 1), CubeDim::new_1d(kernels::CUBE), part.arg(), o.arg(),
                    splits, hd);
            });
            return o;
        }
        stats::launch(3);
        native!(self, m, Buf { handle: Mem::Mtl(m.attend(q.handle.mtl(), k.handle.mtl(), v.handle.mtl(), s.handle.mtl(), n, n_head, n_kv, hd, scale)),
                               len: (n_head * hd) as usize });
        let o = self.scratch((n_head * hd) as usize);
        let group = n_head / n_kv;
        with_client!(self, c, R, unsafe {
            kernels::attn_scores::launch_unchecked::<R>(c, CubeCount::Static(n, n_head, 1), CubeDim::new_1d(kernels::CUBE), q.arg(), k.arg(), s.arg(), n, n_kv, group, hd, scale);
            kernels::attn_softmax::launch_unchecked::<R>(c, CubeCount::Static(n_head, 1, 1), CubeDim::new_1d(kernels::CUBE), s.arg(), n);
            kernels::attn_blend::launch_unchecked::<R>(c, CubeCount::Static(n_head, 1, 1), CubeDim::new_1d(kernels::CUBE), s.arg(), v.arg(), o.arg(), n, n_kv, group, hd, n_head);
        });
        o
    }

    /// Whether attention takes kernels::attend_rows (heads of 32, 64, 128 or 256 values, a device with 32-wide planes,
    /// AM2_FLASH not 0): -> the values per unit.
    fn flash(&self, hd: u32) -> Option<u32> {
        static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if *OFF.get_or_init(|| std::env::var("AM2_FLASH").is_ok_and(|v| v == "0")) || self.is_native() || !matches!(hd, 32 | 64 | 128 | 256) {
            return None;
        }
        let plane = with_client!(self, c, R, {
            let p = c.properties();
            p.hardware.plane_size_min == 32 && p.hardware.plane_size_max == 32 && p.features.plane.contains(cubecl::ir::features::Plane::Ops)
        });
        plane.then_some(hd / 32)
    }

    #[allow(clippy::too_many_arguments)]
    fn attend_rows(&self, q: &Buf, k: &Buf, v: &Buf, scales: &Buf, base: u32, tokens: u32, n_head: u32, n_kv: u32, per: u32, scale: f32,
                   chunk: bool) -> Buf {
        stats::launch(1);
        let o = self.scratch((tokens * n_head * per * 32) as usize);
        with_client!(self, c, R, unsafe {
            kernels::attend_rows::launch_unchecked::<R>(c, CubeCount::Static(n_head, tokens, 1), CubeDim::new_1d(128), q.arg(), k.arg(), v.arg(), o.arg(),
                scales.arg(), base, n_kv, n_head / n_kv, n_head, scale, per, chunk);
        });
        o
    }

    /// Whether this device runs the native Metal backend (which has the chunk kernels below).
    pub fn is_native(&self) -> bool {
        #[cfg(target_os = "macos")]
        if let Client::Metal(_) = &self.client {
            return true;
        }
        false
    }

    /// RoPE over a chunk of `tokens` tokens (native Metal only): `tab` holds each token's (cos, sin) pairs in turn.
    #[cfg_attr(not(target_os = "macos"), allow(unused_variables))]
    pub fn rope_tokens(&self, tab: &Buf, x: &Buf, hd: u32, n_rot: u32, neox: bool, tokens: u32) {
        stats::launch(1);
        native!(self, m, m.rope_tokens(tab.handle.mtl(), x.handle.mtl(), x.len as u32 / hd / tokens, tokens, hd, n_rot, neox));
        self.rope_cube(tab, x, hd, n_rot, neox, x.len as u32 / hd / tokens, tokens);
    }

    /// x[at .. at + n] as a buffer of its own (native Metal only).
    #[cfg_attr(not(target_os = "macos"), allow(unused_variables))]
    pub fn row(&self, x: &Buf, at: usize, n: usize) -> Buf {
        native!(self, m, { stats::launch(1); Buf { handle: Mem::Mtl(m.copy_from(x.handle.mtl(), at as u32, n as u32)), len: n } });
        let y = self.scratch(n);
        self.copy(x, &y, at, 0, n);
        y
    }

    /// Causal attention for a chunk (native Metal only): `tokens` queries at positions base.., K and V already appended.
    #[allow(clippy::too_many_arguments)]
    #[cfg_attr(not(target_os = "macos"), allow(unused_variables))]
    pub fn attend_tokens(&self, q: &Buf, k: &Buf, v: &Buf, scales: &Buf, base: u32, tokens: u32, n_head: u32, n_kv: u32, hd: u32) -> Buf {
        native!(self, m, { stats::launch(1); Buf { handle: Mem::Mtl(m.attend_tokens(q.handle.mtl(), k.handle.mtl(), v.handle.mtl(), scales.handle.mtl(), base, tokens,
                                                                n_head, n_kv, hd)), len: (tokens * n_head * hd) as usize } });
        if let Some(per) = self.flash(hd) {
            return self.attend_rows(q, k, v, scales, base, tokens, n_head, n_kv, per, 0.0, true);
        }
        stats::launch(3);
        let n = base + tokens;
        let rows = tokens * n_head;
        let s = self.scratch((rows * n) as usize);
        let o = self.scratch((rows * hd) as usize);
        let group = n_head / n_kv;
        with_client!(self, c, R, unsafe {
            kernels::attn_scores_tokens::launch_unchecked::<R>(c, CubeCount::Static(n, n_head, tokens), CubeDim::new_1d(kernels::CUBE), q.arg(), k.arg(), s.arg(),
                                                               scales.arg(), n, n_kv, group, hd, n_head, base);
            kernels::attn_softmax::launch_unchecked::<R>(c, CubeCount::Static(rows, 1, 1), CubeDim::new_1d(kernels::CUBE), s.arg(), n);
            kernels::attn_blend::launch_unchecked::<R>(c, CubeCount::Static(rows, 1, 1), CubeDim::new_1d(kernels::CUBE), s.arg(), v.arg(), o.arg(), n, n_kv, group, hd, n_head);
        });
        o
    }

    /// Whether this device has the kernels of the loop and of the experts below (CubeCL; not yet the native Metal backend).
    pub fn has_moe_kernels(&self) -> bool {
        !self.is_native()
    }

    /// y[dst .. dst + n] = x[src .. src + n], queued.
    pub fn copy(&self, x: &Buf, y: &Buf, src: usize, dst: usize, n: usize) {
        stats::launch(1);
        let n = n as u32;
        with_client!(self, c, R, unsafe {
            kernels::copy_range::launch_unchecked::<R>(c, CubeCount::Static(n.div_ceil(kernels::CUBE), 1, 1), CubeDim::new_1d(kernels::CUBE), x.arg(), y.arg(),
                                                       src as u32, dst as u32, n);
        });
    }

    /// x = saved + g * (x - saved), `g` one value per channel, queued (the loop's gate mix).
    pub fn loop_mix(&self, saved: &Buf, x: &Buf, g: &Buf) {
        stats::launch(1);
        let n = x.len as u32;
        with_client!(self, c, R, unsafe {
            kernels::loop_mix::launch_unchecked::<R>(c, CubeCount::Static(n.div_ceil(kernels::CUBE), 1, 1), CubeDim::new_1d(kernels::CUBE), saved.arg(), x.arg(),
                                                     g.arg(), n, g.len as u32);
        });
    }

    /// The router's top `k` of `n_e` experts per token from its logits (`tokens` rows of n_e), softmax gating, queued.
    /// -> per token k indices (f32), then k weights (kernels::route_topk).
    pub fn route_topk(&self, logits: &Buf, n_e: u32, k: u32, norm: bool, scale: f32) -> Buf {
        stats::launch(1);
        let tokens = logits.len as u32 / n_e;
        let out = self.scratch((tokens * 2 * k) as usize);
        with_client!(self, c, R, unsafe {
            kernels::route_topk::launch_unchecked::<R>(c, CubeCount::Static(tokens, 1, 1), CubeDim::new_1d(32), logits.arg(), out.arg(), n_e, k, norm as u32, scale);
        });
        out
    }

    /// The rows `idx` (f32 row numbers) of x, `dim` wide, queued.
    pub fn gather_rows(&self, x: &Buf, idx: &Buf, dim: usize) -> Buf {
        stats::launch(1);
        let y = self.scratch(idx.len * dim);
        with_client!(self, c, R, unsafe {
            kernels::gather_rows::launch_unchecked::<R>(c, CubeCount::Static((dim as u32).div_ceil(kernels::CUBE), idx.len as u32, 1), CubeDim::new_1d(kernels::CUBE),
                                                        x.arg(), idx.arg(), y.arg(), dim as u32);
        });
        y
    }

    /// Row r of x to row idx[r] of y, `dim` wide, queued.
    pub fn scatter_rows(&self, x: &Buf, idx: &Buf, y: &Buf, dim: usize) {
        stats::launch(1);
        with_client!(self, c, R, unsafe {
            kernels::scatter_rows::launch_unchecked::<R>(c, CubeCount::Static((dim as u32).div_ceil(kernels::CUBE), idx.len as u32, 1), CubeDim::new_1d(kernels::CUBE),
                                                         x.arg(), idx.arg(), y.arg(), dim as u32);
        });
    }

    /// x += the chosen experts' outputs (`slots`: tokens * k rows of x's width), weighted by `routes` and added in
    /// routing order, queued.
    pub fn moe_combine(&self, slots: &Buf, routes: &Buf, x: &Buf, k: u32) {
        stats::launch(1);
        let n = x.len as u32;
        let dim = (slots.len / (routes.len / 2)) as u32;
        with_client!(self, c, R, unsafe {
            kernels::moe_combine::launch_unchecked::<R>(c, CubeCount::Static(n.div_ceil(kernels::CUBE), 1, 1), CubeDim::new_1d(kernels::CUBE), slots.arg(),
                                                        routes.arg(), x.arg(), k, dim, n);
        });
    }

    /// y = W x for `x.len() / cols` vectors at once (rows of y in the same order).
    pub fn matvec(&self, w: &Weights, x: &[f32]) -> Vec<f32> {
        let tokens = (x.len() / w.cols as usize).max(1) as u32;
        native!(self, m, { let _ = m; let hx = self.buffer(x); let y = self.mv_tokens(w, &hx, tokens); self.read(&y) });
        with_client!(self, c, R, {
            stats::h2d(x.len() * 4);
            let hx = c.create_from_slice(f32::as_bytes(x));
            let hy = c.empty(tokens as usize * w.rows as usize * 4);
            product!(c, R, w, hx, x.len(), hy.clone(), tokens);
            let t0 = std::time::Instant::now();
            let v = f32::from_bytes(&c.read_one_unchecked(hy)).to_vec();
            stats::d2h(v.len() * 4, t0);
            v
        })
    }
}
