//! Answers through Prastara's Rust engine (`am2strata-hybrid --serve`, the engine the server runs): a GGUF of the model
//! under test, its attention cache and, with `--devices`, its fused GPU graph — the speed the Workshop's own model
//! (no attention cache: each token runs the whole sequence again) cannot reach on code tests and open answers.
//!
//! The process speaks the engine line protocol ([`strata_protocol`], docs/PROTOCOL.md): `GEN <max_new> [keys] <ids>`
//! in, `T <id>` per token and `DONE …` out, `STOP` to end a request early, `QUIT` to exit. Prompts are tokenized here
//! with the model's own chat template ([`ChatTokenizer`], the renderer training used), so both answer paths read the
//! very same ids; the engine only continues them. Synchronous, one request at a time, the engine's stderr in a log.

use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{Receiver, RecvTimeoutError};
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Context, Result};
use strata_protocol::{Event, Gen};

use crate::chatdata::ChatTokenizer;

/// The engine's binary name.
pub const ENGINE_NAME: &str = if cfg!(windows) { "am2strata-hybrid.exe" } else { "am2strata-hybrid" };

/// How an engine is started.
#[derive(Debug, Clone)]
pub struct EngineSpec {
    pub exe: PathBuf,
    pub gguf: PathBuf,
    /// more engine arguments, e.g. `--devices cuda:0=14 --max-context 8192`
    pub args: Vec<String>,
    /// where the engine's stderr goes (None: discarded)
    pub log: Option<PathBuf>,
    /// how long the model may take to load (a GPU placement of an 8B model takes a minute or two)
    pub ready_timeout: Duration,
    /// how long a request may make no token before the engine is taken to be stuck
    pub stall_timeout: Duration,
}

impl EngineSpec {
    pub fn new(exe: PathBuf, gguf: PathBuf, args: Vec<String>) -> Self {
        EngineSpec { exe, gguf, args, log: None, ready_timeout: Duration::from_secs(900), stall_timeout: Duration::from_secs(300) }
    }

    /// Whether the engine arguments put the model on a GPU (`--devices` naming anything but the CPU).
    pub fn on_gpu(&self) -> bool {
        self.args.iter().position(|a| a == "--devices").and_then(|i| self.args.get(i + 1)).is_some_and(|d| d.split(',').any(|x| !x.trim().starts_with("cpu")))
    }
}

/// The engine binary: `explicit`, else `AM2STRATA_HYBRID_EXE`, else beside this program, else on the PATH.
pub fn find_exe(explicit: Option<&Path>) -> Result<PathBuf> {
    if let Some(p) = explicit {
        if !p.is_file() {
            bail!("--engine-exe {}: no such file", p.display());
        }
        return Ok(p.to_path_buf());
    }
    if let Some(p) = std::env::var_os("AM2STRATA_HYBRID_EXE").map(PathBuf::from).filter(|p| p.is_file()) {
        return Ok(p);
    }
    if let Some(p) = std::env::current_exe().ok().and_then(|e| e.parent().map(|d| d.join(ENGINE_NAME))).filter(|p| p.is_file()) {
        return Ok(p);
    }
    crate::sandbox::find_program(ENGINE_NAME)
        .ok_or_else(|| anyhow!("no {ENGINE_NAME} beside this program or on the PATH: build it (cargo build --release --manifest-path \
                                crates/strata-hybrid/Cargo.toml) or pass --engine-exe"))
}

/// What one request did.
#[derive(Debug, Clone, Default)]
pub struct Generated {
    pub ids: Vec<u32>,
    pub prompt_tokens: u64,
    pub prompt_ms: f64,
    pub decode_ms: f64,
    /// `eos`, `length`, `stop` (ended here at an end-of-turn token)
    pub finish: String,
}

/// A running engine.
pub struct Engine {
    child: Child,
    stdin: ChildStdin,
    events: Receiver<Option<Event>>,
    stall: Duration,
    /// the `INFO` lines it sent at start-up
    pub info: Vec<String>,
    /// its context (`INFO … context=N`)
    pub context: usize,
    pub load_seconds: f64,
}

fn read_events(stdout: std::process::ChildStdout) -> Receiver<Option<Event>> {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        for line in BufReader::new(stdout).lines() {
            let Ok(line) = line else { break };
            if tx.send(Some(Event::parse(&line))).is_err() {
                return;
            }
        }
        let _ = tx.send(None);
    });
    rx
}

impl Engine {
    /// Start `spec`'s engine (a looped model at `loop_passes` when given) and wait for READY.
    pub fn start(spec: &EngineSpec, loop_passes: Option<usize>) -> Result<Engine> {
        let mut cmd = Command::new(&spec.exe);
        cmd.arg("--serve").arg("--model").arg(&spec.gguf).args(&spec.args);
        if let Some(p) = loop_passes {
            cmd.args(["--loop-passes", &p.to_string()]);
        }
        cmd.stdin(Stdio::piped()).stdout(Stdio::piped());
        match &spec.log {
            Some(l) => {
                let f = std::fs::OpenOptions::new().create(true).append(true).open(l).with_context(|| format!("{}", l.display()))?;
                cmd.stderr(Stdio::from(f));
            }
            None => {
                cmd.stderr(Stdio::null());
            }
        }
        let t0 = Instant::now();
        let mut child = cmd.spawn().with_context(|| format!("starting the engine {}", spec.exe.display()))?;
        let stdin = child.stdin.take().ok_or_else(|| anyhow!("the engine's stdin"))?;
        let events = read_events(child.stdout.take().ok_or_else(|| anyhow!("the engine's stdout"))?);
        let mut e = Engine { child, stdin, events, stall: spec.stall_timeout, info: vec![], context: 32768, load_seconds: 0.0 };
        let log_hint = spec.log.as_ref().map_or(String::new(), |l| format!(" (its log: {})", l.display()));
        loop {
            let left = spec.ready_timeout.saturating_sub(t0.elapsed());
            match e.events.recv_timeout(left) {
                Ok(Some(Event::Ready(_))) => break,
                Ok(Some(Event::Info(s))) => {
                    if let Some(c) = s.split_whitespace().find_map(|w| w.strip_prefix("context=")).and_then(|v| v.parse().ok()) {
                        e.context = c;
                    }
                    e.info.push(s);
                }
                Ok(Some(_)) => {}
                Ok(None) | Err(RecvTimeoutError::Disconnected) => {
                    let status = e.child.wait().ok();
                    bail!("the engine stopped while loading {}{}{log_hint}", spec.gguf.display(), status.map_or(String::new(), |s| format!(": {s}")));
                }
                Err(RecvTimeoutError::Timeout) => bail!("the engine did not load {} within {} s{log_hint}", spec.gguf.display(), spec.ready_timeout.as_secs()),
            }
        }
        e.load_seconds = t0.elapsed().as_secs_f64();
        Ok(e)
    }

    /// Up to `max_new` tokens after `ids`; ends at the engine's own end of sequence or at any of `stop` (the chat's
    /// end-of-turn tokens, asked to STOP here). Neither end token is part of the answer.
    pub fn generate(&mut self, ids: &[u32], max_new: usize, sampling: &strata_protocol::Sampling, stop: &[u32]) -> Result<Generated> {
        let room = self.context.saturating_sub(ids.len() + 8);
        if ids.is_empty() || room == 0 {
            bail!("the prompt ({} tokens) does not fit the engine's context of {}", ids.len(), self.context);
        }
        let line = Gen { max_new: max_new.min(room) as u32, sampling, turn_at: None, ids, images: None }.line();
        self.stdin.write_all(line.as_bytes()).and_then(|_| self.stdin.flush()).context("the engine is gone")?;
        let mut g = Generated::default();
        let mut stopped = false;
        loop {
            match self.events.recv_timeout(self.stall) {
                Ok(Some(Event::Token(t))) => {
                    if stopped {
                        continue;
                    }
                    if stop.contains(&t) {
                        stopped = true;
                        self.stdin.write_all(strata_protocol::STOP.as_bytes()).and_then(|_| self.stdin.flush()).context("the engine is gone")?;
                        continue;
                    }
                    g.ids.push(t);
                }
                Ok(Some(Event::Done(d))) => {
                    g.prompt_tokens = d.prompt_tokens;
                    g.prompt_ms = d.prompt_ms;
                    g.decode_ms = d.decode_ms;
                    g.finish = if stopped { "stop".into() } else { d.finish.clone() };
                    // the engine's own end token is sent before DONE: it ends the answer, it is not part of it
                    if d.finish == "eos" && !stopped {
                        g.ids.pop();
                    }
                    return Ok(g);
                }
                Ok(Some(Event::Err(m))) => bail!("the engine refused the request: {m}"),
                Ok(Some(_)) => {}
                Ok(None) | Err(RecvTimeoutError::Disconnected) => bail!("the engine stopped during a request (out of memory, or a crash)"),
                Err(RecvTimeoutError::Timeout) => bail!("the engine made no token for {} s", self.stall.as_secs()),
            }
        }
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        let _ = self.stdin.write_all(strata_protocol::QUIT.as_bytes());
        let _ = self.stdin.flush();
        let t0 = Instant::now();
        while t0.elapsed() < Duration::from_secs(20) {
            if let Ok(Some(_)) = self.child.try_wait() {
                return;
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// The GGUF must be the same model as the Hugging Face folder the loss tests and the tokenizer come from: the same
/// block count and vocabulary, and the same loop. Returns the GGUF's architecture.
pub fn check_gguf_matches(gguf: &Path, cfg: &crate::model::ModelConfig) -> Result<String> {
    let g = strata_gguf::Gguf::open(gguf).map_err(|e| anyhow!("{}: {e}", gguf.display()))?;
    let arch = g.architecture().ok_or_else(|| anyhow!("{}: no general.architecture", gguf.display()))?.to_string();
    let blocks = g.u64(&format!("{arch}.block_count"));
    if blocks.is_some_and(|b| b as usize != cfg.num_hidden_layers) {
        bail!("{}: {} blocks, the model folder has {}: --engine-gguf must be this model converted", gguf.display(), blocks.unwrap_or(0), cfg.num_hidden_layers);
    }
    let vocab = g.tensors.iter().find(|t| t.name == "token_embd.weight").and_then(|t| t.dims.get(1).copied());
    if vocab.is_some_and(|v| v as usize != cfg.vocab_size) {
        bail!("{}: a vocabulary of {}, the model folder has {}", gguf.display(), vocab.unwrap_or(0), cfg.vocab_size);
    }
    let loop_count = g.u64(&format!("{arch}.loop.count")).map(|c| c as usize);
    let own = cfg.loop_.map(|l| l.passes);
    if loop_count.unwrap_or(1) != own.unwrap_or(1) {
        bail!("{}: the loop runs {} passes, the model folder's {}", gguf.display(), loop_count.unwrap_or(1), own.unwrap_or(1));
    }
    Ok(arch)
}

/// Totals over the requests an answerer made, for the report.
#[derive(Debug, Clone, Default)]
pub struct Speed {
    pub requests: usize,
    pub prompt_tokens: u64,
    pub generated: u64,
    pub prompt_ms: f64,
    pub decode_ms: f64,
    pub load_seconds: f64,
}

impl Speed {
    pub fn tokens_per_second(&self) -> f64 {
        if self.decode_ms > 0.0 { self.generated as f64 / (self.decode_ms / 1000.0) } else { 0.0 }
    }

    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({"requests": self.requests, "prompt_tokens": self.prompt_tokens, "generated": self.generated, "prompt_ms": self.prompt_ms,
                           "decode_ms": self.decode_ms, "tokens_per_second": self.tokens_per_second(), "load_seconds": self.load_seconds})
    }
}

/// Answers from a running engine, with the model's chat template.
pub struct EngineGenerator<'a> {
    engine: std::cell::RefCell<Engine>,
    pub tok: &'a ChatTokenizer,
    pub stop: Vec<u32>,
    pub max_prompt: usize,
    pub speed: std::cell::RefCell<Speed>,
}

impl<'a> EngineGenerator<'a> {
    pub fn new(engine: Engine, tok: &'a ChatTokenizer, max_prompt: usize) -> Self {
        let speed = Speed { load_seconds: engine.load_seconds, ..Speed::default() };
        EngineGenerator { engine: std::cell::RefCell::new(engine), tok, stop: tok.stop_ids(), max_prompt, speed: std::cell::RefCell::new(speed) }
    }

    pub fn into_speed(self) -> Speed {
        self.speed.into_inner()
    }
}

impl crate::regress::Answer for EngineGenerator<'_> {
    fn tok(&self) -> &ChatTokenizer {
        self.tok
    }

    fn max_prompt(&self) -> usize {
        self.max_prompt.min(self.engine.borrow().context.saturating_sub(16))
    }

    fn generate(&self, prompt: &[u32], max_new: usize, sampling: Option<crate::regress::Sampling>) -> Result<Vec<u32>> {
        let s = match sampling {
            Some(s) if s.temperature > 0.0 => strata_protocol::Sampling { temperature: Some(s.temperature), top_p: Some(s.top_p),
                                                                        seed: Some((s.seed % (i64::MAX as u64)).max(1) as i64), ..Default::default() },
            _ => strata_protocol::Sampling::default(),
        };
        let g = self.engine.borrow_mut().generate(prompt, max_new, &s, &self.stop)?;
        let mut sp = self.speed.borrow_mut();
        sp.requests += 1;
        sp.prompt_tokens += g.prompt_tokens;
        sp.generated += g.ids.len() as u64;
        sp.prompt_ms += g.prompt_ms;
        sp.decode_ms += g.decode_ms;
        Ok(g.ids)
    }

    fn label(&self) -> String {
        "engine".into()
    }
}
