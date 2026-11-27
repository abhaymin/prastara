//! A packed training pass holds no more memory than its conversations run alone side by side: the peak bytes
//! allocated during a pass's forward and backward (checkpointed, as a streamed run is), counted by this test's own
//! allocator, for a packed pass against the sum of each conversation's own pass. The block-diagonal table the first
//! packed run used (`PACK_TABLE`) is measured too: it holds `(Σ len)²` scores per head where its conversations hold
//! `Σ len²`.

mod common;

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use candle_core::Device;
use strata_workshop::model::{make_upcycled, BaseDtype, Ctx, LlamaModel, LoraSpec, MoeSpec, PassLayout, Target, PACK_TABLE};

struct Counting;

static NOW: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        let p = unsafe { System.alloc(l) };
        if !p.is_null() {
            let n = NOW.fetch_add(l.size(), Ordering::Relaxed) + l.size();
            PEAK.fetch_max(n, Ordering::Relaxed);
        }
        p
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        unsafe { System.dealloc(p, l) };
        NOW.fetch_sub(l.size(), Ordering::Relaxed);
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, new: usize) -> *mut u8 {
        let q = unsafe { System.realloc(p, l, new) };
        if !q.is_null() {
            if new > l.size() {
                let n = NOW.fetch_add(new - l.size(), Ordering::Relaxed) + new - l.size();
                PEAK.fetch_max(n, Ordering::Relaxed);
            } else {
                NOW.fetch_sub(l.size() - new, Ordering::Relaxed);
            }
        }
        q
    }
}

#[global_allocator]
static A: Counting = Counting;

/// The peak bytes above the start while `f` runs.
fn peak_of(f: impl FnOnce()) -> usize {
    let start = NOW.load(Ordering::Relaxed);
    PEAK.store(start, Ordering::Relaxed);
    f();
    PEAK.load(Ordering::Relaxed) - start
}

fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn model(dir: &std::path::Path) -> LlamaModel {
    let mut m = LlamaModel::load(dir, BaseDtype::F32, &Device::Cpu).unwrap();
    m.add_lora(&LoraSpec { r: 4, alpha: 8.0, dropout: 0.0, use_rslora: false, targets: Target::ALL.to_vec(), train_layers: None, seed: 3 }).unwrap();
    m.set_checkpointing(true);
    m
}

/// One training pass (forward, backward, the replays' gradient parts taken) over `convs` laid as one sequence.
fn pass(m: &LlamaModel, convs: &[(Vec<u32>, Vec<bool>)]) {
    let layout = PassLayout { seqs: vec![convs.iter().map(|c| c.0.len()).collect()] };
    let tokens: Vec<u32> = convs.iter().flat_map(|c| c.0.clone()).collect();
    let sup: Vec<bool> = convs.iter().flat_map(|c| c.1.clone()).collect();
    let ctx = Ctx::train(1);
    let h = if convs.len() == 1 { m.lower(&tokens).unwrap() } else { m.lower_packed(&tokens, &layout).unwrap() };
    let loss = m.loss_packed(&h, &layout, &tokens, &sup, &ctx).unwrap();
    let g = loss.backward().unwrap();
    let extra = m.take_checkpoint_grads();
    drop((g, extra, loss, h));
}

#[test]
fn a_packed_pass_holds_no_more_than_its_conversations_alone() {
    // one thread: the matrix products' per-thread buffers would otherwise blur the counts
    std::env::set_var("RAYON_NUM_THREADS", "1");
    let root = common::temp_dir("pack-memory");
    let dense = root.join("dense");
    // 8 heads, so attention is a real share of a block's memory, as on the 56B-A5B (32 heads, hidden 4096)
    let cfg = strata_workshop::testmodel::TinyLlama { seed: 0xC4A7, num_layers: 2, vocab_size: strata_workshop::testmodel::CHAT_VOCAB, hidden: 64, intermediate: 128,
                                                      num_attention_heads: 8, num_key_value_heads: 2, shard_count: 1 };
    strata_workshop::testmodel::write_chat_model(&dense, &cfg).unwrap();
    let moe = root.join("moe");
    make_upcycled(&dense, &moe, MoeSpec { expert_ff: Some(32), shared_a: true, expert_rank: Some(2), ..MoeSpec::full(8, 2) }).unwrap();
    let lens = [150usize, 90, 210, 60, 120];
    let mut st = 7u64;
    for dir in [&dense, &moe] {
        let m = model(dir);
        let convs: Vec<(Vec<u32>, Vec<bool>)> = lens.iter().map(|&n| {
            ((0..n).map(|_| (splitmix(&mut st) % m.cfg.vocab_size as u64) as u32).collect(), (0..n).map(|p| p >= n / 2).collect())
        }).collect();
        // warm up (the first pass makes the thread pool and caches)
        pass(&m, &convs[..1]);
        let alone: Vec<usize> = convs.iter().map(|c| peak_of(|| pass(&m, std::slice::from_ref(c)))).collect();
        let sum: usize = alone.iter().sum();
        PACK_TABLE.store(false, Ordering::Relaxed);
        let packed = peak_of(|| pass(&m, &convs));
        PACK_TABLE.store(true, Ordering::Relaxed);
        let table = peak_of(|| pass(&m, &convs));
        PACK_TABLE.store(false, Ordering::Relaxed);
        let mib = |b: usize| b as f64 / (1 << 20) as f64;
        println!("{}: {} tokens in one packed sequence: peak {:.2} MiB a conversation at a time ({:.2} with one block-diagonal table); \
                  alone {:?} MiB, summed {:.2} MiB, the largest {:.2}",
                 dir.file_name().unwrap().to_string_lossy(), lens.iter().sum::<usize>(), mib(packed), mib(table),
                 alone.iter().map(|b| (mib(*b) * 100.0).round() / 100.0).collect::<Vec<_>>(), mib(sum), mib(*alone.iter().max().unwrap()));
        assert!(packed as f64 <= sum as f64 * 1.05 + 256.0 * 1024.0, "packed {packed} bytes against {sum} alone");
        assert!(table > packed, "the table costs more: {table} vs {packed}");
    }
}
