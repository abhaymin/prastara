//! The product kernels' achieved bandwidth on one device, at a dense 8B model's shapes (Qwen3-8B: 4096 wide, 12288 in
//! the feed-forward, a 151936-token vocabulary):
//!   cargo run --release --example kbench -- cuda:1 [Q8_0,Q4_K | attn] [reps]
//! Decode: y = W x for one vector (GB/s of weights read). Prompt chunks: y = W X for 32/128/512 vectors at once; GB/s
//! there is the weights' bytes over the time (above the card's bandwidth means the weights were read fewer times than
//! once per token), and GFLOP/s the multiply-adds.
use strata_gpu::Gpu;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let dev = a.get(1).map(String::as_str).unwrap_or("cuda:1");
    let types: Vec<String> = a.get(2).map(|s| s.split(',').map(str::to_string).collect()).unwrap_or(vec!["Q8_0".into(), "Q4_K".into()]);
    let reps: usize = a.get(3).and_then(|v| v.parse().ok()).unwrap_or(50);
    let g = Gpu::open(dev).unwrap();
    let shapes = [(4096usize, 4096usize), (12288, 4096), (4096, 12288), (151936, 4096)];
    // decode attention over n positions (32 query heads of 128, 8 KV heads; AM2_FLASH=0 for the three-kernel path)
    if types.iter().any(|t| t == "attn") {
        let (nh, nkv, hd) = (32u32, 8u32, 128u32);
        let ctx = 4096usize;
        let kc = g.buffer(&vec![0.01f32; ctx * (nkv * hd) as usize]);
        let vc = g.buffer(&vec![0.02f32; ctx * (nkv * hd) as usize]);
        let s = g.scratch(nh as usize * ctx);
        let q = g.buffer(&vec![0.03f32; (nh * hd) as usize]);
        for n in [128u32, 1024, 4096] {
            g.read(&g.attend(&q, &kc, &vc, &s, n, nh, nkv, hd, 0.1));
            let t = std::time::Instant::now();
            let mut y = g.attend(&q, &kc, &vc, &s, n, nh, nkv, hd, 0.1);
            for _ in 1..reps {
                y = g.attend(&q, &kc, &vc, &s, n, nh, nkv, hd, 0.1);
            }
            g.read(&y);
            let us = t.elapsed().as_secs_f64() * 1e6 / reps as f64;
            let gb = 2.0 * n as f64 * (nkv * hd) as f64 * 4.0 / 1e9;
            println!("attend n={n:<5} {us:8.1} us {:6.1} GB/s of K and V", gb / (us / 1e6));
        }
        return;
    }
    for ty in &types {
        let (be, bb, _) = strata_gpu::kernels::geometry(ty).unwrap();
        for &(rows, cols) in &shapes {
            let mut data = vec![0u8; rows * cols / be as usize * bb as usize];
            // pseudo-random bytes with small f16 scales: the values do not matter for the speed
            for (i, b) in data.iter_mut().enumerate() {
                *b = ((i as u64).wrapping_mul(2654435761) >> 13) as u8 & 0x3f;
            }
            let w = g.upload(ty, &data, rows, cols).unwrap();
            let gb = data.len() as f64 / 1e9;
            let mut line = format!("{ty:5} {rows:>6}x{cols:<6} {:7.1} MB |", data.len() as f64 / 1e6);
            for tokens in [1usize, 32, 128, 512] {
                if rows == 151936 && tokens > 1 {
                    continue; // the output matrix runs for the last token of a chunk only
                }
                let x = g.buffer(&vec![0.01f32; tokens * cols]);
                g.read(&g.mv_tokens(&w, &x, tokens as u32)); // compiled and warm
                let n = if tokens == 1 { reps } else { (reps / 5).max(3) };
                let t = std::time::Instant::now();
                let mut y = g.mv_tokens(&w, &x, tokens as u32);
                for _ in 1..n {
                    y = g.mv_tokens(&w, &x, tokens as u32);
                }
                g.read(&y);
                let s = t.elapsed().as_secs_f64() / n as f64;
                let flops = 2.0 * rows as f64 * cols as f64 * tokens as f64;
                if tokens == 1 {
                    line += &format!(" decode {:7.1} us {:6.1} GB/s |", s * 1e6, gb / s);
                } else {
                    line += &format!(" t={tokens:<3} {:8.0} us {:6.0} GB/s {:6.0} GFLOP/s |", s * 1e6, gb / s, flops / s / 1e9);
                }
                drop(y);
            }
            println!("{line}");
            drop(w);
            g.sync().unwrap();
        }
    }
}
