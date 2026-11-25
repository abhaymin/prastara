//! am2strata-vision: Inkling 2's image encoder in Rust (vision.rs), with the protocol of Prastara's image encoder
//! (crates/strata-engine/src/vision.rs): one `READY` line, then for each `ENC <image> <out>` line it writes the
//! picture's rows to <out> (raw little-endian f32, n_tokens x the text model's width) and answers `OK <tokens>`, or
//! `ERR <why>`. Paths are relative to the working directory the server gives it.
//!   am2strata-vision --mmproj <file> [--model <text gguf>] [--threads N] [--max-tokens N] [--min-tokens N] [--gpu]
//! --model only checks that the rows are as wide as the text model's input; --gpu is accepted and runs on the CPU.

mod vision;

use std::io::{BufRead, Write};
use std::path::Path;

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (mut mmproj, mut model, mut threads, mut max_t) = (None, None, 0usize, 0usize);
    let mut it = args.iter();
    while let Some(a) = it.next() {
        let mut val = |name: &str| it.next().cloned().ok_or_else(|| format!("{name} needs a value"));
        match a.as_str() {
            "--mmproj" => mmproj = Some(val("--mmproj")?),
            "--model" => model = Some(val("--model")?),
            "--gpu" => {}
            "--threads" => threads = val("--threads")?.parse().map_err(|_| "--threads: not a number")?,
            "--max-tokens" => max_t = val("--max-tokens")?.parse().map_err(|_| "--max-tokens: not a number")?,
            "--min-tokens" => { val("--min-tokens")?.parse::<usize>().map_err(|_| "--min-tokens: not a number")?; }
            other => return Err(format!("unknown argument: {other}")),
        }
    }
    if threads > 0 {
        let _ = rayon::ThreadPoolBuilder::new().num_threads(threads).build_global();
    }
    let t0 = std::time::Instant::now();
    let mut enc = vision::Encoder::load(Path::new(&mmproj.ok_or("missing --mmproj")?))?;
    if max_t > 0 {
        enc.max_tokens = max_t;
    }
    if let Some(m) = model {
        let g = strata_gguf::Gguf::open(Path::new(&m)).map_err(|e| e.to_string())?;
        let width = g.arch_u64("embedding_length").ok_or_else(|| format!("{m} names no embedding_length"))? as usize;
        if width != enc.out_dim {
            return Err(format!("the projector gives {}-wide rows and the text model takes {width}", enc.out_dim));
        }
    }
    eprintln!("am2strata-vision: loaded in {} ms ({} wide rows)", t0.elapsed().as_millis(), enc.out_dim);
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    writeln!(out, "READY am2strata-vision").and_then(|_| out.flush()).map_err(|e| e.to_string())?;
    for line in std::io::stdin().lock().lines() {
        let line = line.map_err(|e| e.to_string())?;
        let f: Vec<&str> = line.split_whitespace().collect();
        let reply = match f.as_slice() {
            ["QUIT"] => break,
            ["ENC", img, dst] => {
                let t = std::time::Instant::now();
                match enc.encode_file(Path::new(img)) {
                    Ok((rows, n)) => {
                        let bytes: Vec<u8> = rows.iter().flat_map(|v| v.to_le_bytes()).collect();
                        match std::fs::write(dst, &bytes) {
                            Ok(()) => {
                                eprintln!("am2strata-vision: {img}: {n} tokens in {} ms", t.elapsed().as_millis());
                                format!("OK {n}")
                            }
                            Err(e) => format!("ERR cannot write {dst} ({e})"),
                        }
                    }
                    Err(e) => format!("ERR {e}"),
                }
            }
            [] => continue,
            _ => "ERR unknown command".into(),
        };
        writeln!(out, "{reply}").and_then(|_| out.flush()).map_err(|e| e.to_string())?;
    }
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        println!("ERR {e}");
        eprintln!("am2strata-vision: {e}");
        std::process::exit(1);
    }
}
