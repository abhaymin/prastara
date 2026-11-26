//! Running untrusted code (a model's answer to a coding test) in a child process with limits: CPU time, memory, file
//! size, a wall clock, no core dumps, and — where the kernel lets an unprivileged process have its own network
//! namespace — no network at all. Its own process group, so whatever it starts is killed with it. The environment is
//! cleared (a fixed PATH, a scratch HOME), and it runs in its own scratch folder.
//!
//! This is a guard against mistakes (an endless loop, a runaway allocation, a stray request), not a jail: the code
//! still runs as this user and can read what this user can read. Regression tests run benchmark answers, which are
//! written by our own model, not an adversary.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::OnceLock;
use std::time::{Duration, Instant};

/// What a run may use.
#[derive(Debug, Clone, Copy)]
pub struct Limits {
    /// CPU seconds (the kernel stops the process past this)
    pub cpu_seconds: u64,
    /// wall-clock seconds (a sleeping or blocked process is killed past this)
    pub wall_seconds: f64,
    /// memory in bytes: the address space (`RLIMIT_AS`) for an interpreter, the data segment (`RLIMIT_DATA`) for a
    /// compiler, which reserves far more address space than it uses
    pub memory_bytes: u64,
    pub limit_address_space: bool,
    /// the largest file it may write
    pub file_bytes: u64,
    /// a network namespace of its own (no interfaces but a down loopback), when the kernel allows one
    pub no_network: bool,
    /// at most this much of stdout and of stderr is kept
    pub output_bytes: usize,
}

impl Limits {
    /// For running a test program: 10 s of CPU, 20 s on the clock, 1 GB of address space, 16 MB files, no network.
    pub fn run() -> Limits {
        Limits { cpu_seconds: 10, wall_seconds: 20.0, memory_bytes: 1 << 30, limit_address_space: true, file_bytes: 16 << 20, no_network: true, output_bytes: 1 << 20 }
    }

    /// For compiling (rustc): 120 s of CPU, 180 s on the clock, 4 GB of data, 512 MB files, no network.
    pub fn compile() -> Limits {
        Limits { cpu_seconds: 120, wall_seconds: 180.0, memory_bytes: 4 << 30, limit_address_space: false, file_bytes: 512 << 20, no_network: true, output_bytes: 1 << 20 }
    }
}

/// How a run ended.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Ending {
    /// exited by itself with this code
    Exited(i32),
    /// stopped by the CPU-time limit
    CpuTime,
    /// killed by the wall clock
    WallTime,
    /// ran out of the memory it was allowed (an allocation failed)
    Memory,
    /// another signal
    Signal(i32),
    /// it could not be started
    NotStarted(String),
}

impl Ending {
    pub fn label(&self) -> String {
        match self {
            Ending::Exited(0) => "ok".into(),
            Ending::Exited(c) => format!("exit {c}"),
            Ending::CpuTime => "cpu time limit".into(),
            Ending::WallTime => "wall time limit".into(),
            Ending::Memory => "memory limit".into(),
            Ending::Signal(s) => format!("signal {s}"),
            Ending::NotStarted(e) => format!("not started: {e}"),
        }
    }
}

/// What a run did.
#[derive(Debug, Clone)]
pub struct Outcome {
    pub ending: Ending,
    pub stdout: String,
    pub stderr: String,
    pub seconds: f64,
    /// it ran in a network namespace of its own
    pub network_isolated: bool,
}

impl Outcome {
    pub fn ok(&self) -> bool {
        self.ending == Ending::Exited(0)
    }
}

/// Whether this kernel lets an unprivileged process take a network namespace of its own (a user namespace with it),
/// found once by trying.
pub fn network_isolation_available() -> bool {
    static AVAILABLE: OnceLock<bool> = OnceLock::new();
    *AVAILABLE.get_or_init(|| {
        #[cfg(target_os = "linux")]
        {
            let Some(t) = ["/bin/true", "/usr/bin/true"].into_iter().map(Path::new).find(|p| p.exists()) else { return false };
            let mut c = Command::new(t);
            c.stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::null());
            use std::os::unix::process::CommandExt as _;
            unsafe {
                c.pre_exec(|| {
                    if libc::unshare(libc::CLONE_NEWUSER | libc::CLONE_NEWNET) != 0 {
                        return Err(std::io::Error::last_os_error());
                    }
                    Ok(())
                });
            }
            c.status().map(|s| s.success()).unwrap_or(false)
        }
        #[cfg(not(target_os = "linux"))]
        {
            false
        }
    })
}

/// A scratch folder of its own for one run, removed when dropped.
pub struct Scratch(pub PathBuf);

impl Scratch {
    pub fn new(tag: &str) -> std::io::Result<Scratch> {
        static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0);
        let d = std::env::temp_dir().join(format!("am2-sandbox-{tag}-{}-{nanos}-{}", std::process::id(), N.fetch_add(1, std::sync::atomic::Ordering::SeqCst)));
        std::fs::create_dir_all(&d)?;
        Ok(Scratch(d))
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn read_capped(mut r: impl Read + Send + 'static, cap: usize) -> std::thread::JoinHandle<Vec<u8>> {
    std::thread::spawn(move || {
        let mut out = Vec::new();
        let mut buf = [0u8; 8192];
        loop {
            match r.read(&mut buf) {
                Ok(0) | Err(_) => break,
                Ok(n) => {
                    if out.len() < cap {
                        out.extend_from_slice(&buf[..n.min(cap - out.len())]);
                    }
                }
            }
        }
        out
    })
}

/// Run `program args` in `dir` with `stdin` under `limits`.
pub fn run(program: &Path, args: &[&str], dir: &Path, stdin: &[u8], limits: &Limits) -> Outcome {
    let isolate = limits.no_network && network_isolation_available();
    let mut c = Command::new(program);
    c.args(args).current_dir(dir).env_clear()
        .env("PATH", "/usr/local/bin:/usr/bin:/bin").env("HOME", dir).env("TMPDIR", dir).env("LANG", "C.UTF-8")
        .env("PYTHONDONTWRITEBYTECODE", "1").env("PYTHONHASHSEED", "0")
        .stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped());
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt as _;
        let l = *limits;
        unsafe {
            c.pre_exec(move || {
                // its own group, so a kill reaches whatever it starts
                libc::setpgid(0, 0);
                let set = |what, soft: u64, hard: u64| -> std::io::Result<()> {
                    let r = libc::rlimit { rlim_cur: soft as libc::rlim_t, rlim_max: hard as libc::rlim_t };
                    if libc::setrlimit(what, &r) != 0 { Err(std::io::Error::last_os_error()) } else { Ok(()) }
                };
                set(libc::RLIMIT_CPU, l.cpu_seconds, l.cpu_seconds + 1)?;
                set(if l.limit_address_space { libc::RLIMIT_AS } else { libc::RLIMIT_DATA }, l.memory_bytes, l.memory_bytes)?;
                set(libc::RLIMIT_FSIZE, l.file_bytes, l.file_bytes)?;
                set(libc::RLIMIT_CORE, 0, 0)?;
                #[cfg(target_os = "linux")]
                if isolate && libc::unshare(libc::CLONE_NEWUSER | libc::CLONE_NEWNET) != 0 {
                    return Err(std::io::Error::last_os_error());
                }
                Ok(())
            });
        }
    }
    let t0 = Instant::now();
    let mut child = match c.spawn() {
        Ok(ch) => ch,
        Err(e) => return Outcome { ending: Ending::NotStarted(e.to_string()), stdout: String::new(), stderr: String::new(), seconds: 0.0, network_isolated: isolate },
    };
    let pid = child.id() as i32;
    let input = stdin.to_vec();
    let mut sin = child.stdin.take();
    let writer = std::thread::spawn(move || {
        if let Some(s) = sin.as_mut() {
            let _ = s.write_all(&input);
        }
        drop(sin);
    });
    let out = read_capped(child.stdout.take().unwrap(), limits.output_bytes);
    let err = read_capped(child.stderr.take().unwrap(), limits.output_bytes);
    let deadline = Duration::from_secs_f64(limits.wall_seconds.max(0.1));
    let mut timed_out = false;
    let status = loop {
        match child.try_wait() {
            Ok(Some(s)) => break Some(s),
            Ok(None) if t0.elapsed() >= deadline => {
                timed_out = true;
                #[cfg(unix)]
                unsafe { libc::kill(-pid, libc::SIGKILL); }
                let _ = child.kill();
                break child.wait().ok();
            }
            Ok(None) => std::thread::sleep(Duration::from_millis(5)),
            Err(_) => break None,
        }
    };
    // whatever it left running in its group goes too (the group is the child's own: its id is the child's PID)
    #[cfg(unix)]
    unsafe { libc::kill(-pid, libc::SIGKILL); }
    let _ = writer.join();
    let stdout = String::from_utf8_lossy(&out.join().unwrap_or_default()).into_owned();
    let stderr = String::from_utf8_lossy(&err.join().unwrap_or_default()).into_owned();
    let seconds = t0.elapsed().as_secs_f64();
    let oom = |e: &str| e.contains("MemoryError") || e.contains("memory allocation of") || e.contains("Cannot allocate memory") || e.contains("std::bad_alloc");
    let ending = if timed_out {
        Ending::WallTime
    } else {
        match status {
            None => Ending::Signal(0),
            Some(s) => {
                #[cfg(unix)]
                {
                    use std::os::unix::process::ExitStatusExt as _;
                    match (s.code(), s.signal()) {
                        (_, Some(libc::SIGXCPU)) => Ending::CpuTime,
                        // the hard CPU limit is SIGKILL, a second after the soft one
                        (_, Some(libc::SIGKILL)) if seconds + 0.5 >= limits.cpu_seconds as f64 => Ending::CpuTime,
                        (_, Some(libc::SIGABRT)) | (_, Some(libc::SIGSEGV)) if oom(&stderr) => Ending::Memory,
                        (_, Some(sig)) => Ending::Signal(sig),
                        (Some(c), _) if c != 0 && oom(&stderr) => Ending::Memory,
                        (Some(c), _) => Ending::Exited(c),
                        (None, None) => Ending::Signal(0),
                    }
                }
                #[cfg(not(unix))]
                {
                    match s.code() {
                        Some(c) if c != 0 && oom(&stderr) => Ending::Memory,
                        Some(c) => Ending::Exited(c),
                        None => Ending::Signal(0),
                    }
                }
            }
        }
    };
    Outcome { ending, stdout, stderr, seconds, network_isolated: isolate }
}

/// `name` on the PATH, if it is there.
pub fn find_program(name: &str) -> Option<PathBuf> {
    std::env::var_os("PATH").and_then(|p| std::env::split_paths(&p).map(|d| d.join(name)).find(|f| f.is_file()))
}
