//! The start of a fine-grained upcycle (the 56B-A5B design, [`crate::model::MoeSpec`]): which dense neurons form
//! each expert's group, and the router's first values. Written to `moe_init.safetensors` beside the linked weights
//! (`perm.<b>`, `router.<b>`), read by the Workshop when it loads the folder and by `export-moe`.
//!
//! **The split** (`--split`):
//! - `contiguous`: neurons `g·ff_e .. (g+1)·ff_e` are group g;
//! - `cluster` (MoEfication, Zhang et al. 2022): the dense model runs over calibration text, and each neuron's
//!   activation magnitude `|silu(gate)·up|` over the calibration tokens is its feature; balanced spherical k-means
//!   (a seeded farthest-point start, then assignment by sorted similarity with every group capped at exactly ff_e
//!   neurons) puts neurons that fire together in one group. Deterministic for a seed.
//!
//! **The router's start.** Expert e is copy `c = e div G` of group `g = e mod G`; its row is
//! `ROUTER_COPY · d_c + ROUTER_GROUP · r_g`:
//! - `r_g` predicts how much group g matters for a token: a ridge regression (conjugate gradients over the
//!   calibration states) of the group's standardised importance `Σ_{n∈g} |act_n|·‖down_n‖` on the feed-forward's
//!   input, scaled so `r_g·h` has unit spread. A token's best groups by `r_g·h` are those the router picks first.
//! - `d_c` is a seeded random direction per copy (the input's mean direction projected out, unit spread). All of a
//!   token's top-k come from the copy whose `d_c·h` is highest, as long as its lead over the next copy is larger
//!   than the small group term's range: then a token's k experts are k DIFFERENT groups, its best k by `r_g`, and
//!   their renormalised weights are nearly equal (about 1/k: the out-scale k makes them the plain sum of the groups).
//!   Near a boundary between two copies a token can pick the same group twice; the share of tokens whose k picks are
//!   k distinct groups is measured and reported (`distinct`). Copies of a group are identical at the start, so
//!   which copy serves a token changes nothing until training makes them differ.
//!
//! Without calibration text the split is contiguous and `r_g`, `d_c` are seeded random directions.
//!
//! **Honest limits.** With `top_k · ff_e < ff` a token uses only part of the dense feed-forward, so the upcycled
//! model at step 0 is NOT the dense model: the router's linear guess at the important groups and the dropped groups
//! cost loss, which training then has to win back. `--dense-start-check` measures it on held-out conversations
//! (dense vs upcycled-at-start cross-entropy, block by block on the same machine). Only E = G with top_k = G (every
//! group, one copy each) is the dense model exactly.
//!
//! **Memory.** The sweep runs one block at a time (the loop's blocks together, since it repeats them): the block's
//! weights, the calibration tokens' hidden states (`tokens × hidden` f32), and per block waiting to be grouped its
//! inputs and activations (`tokens × (hidden + ff)` f32, kept across the loop's passes until the loop is done).

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{anyhow, bail, Context as _, Result};
use candle_core::{DType, Device, Tensor};
use serde_json::{json, Value};

use crate::chatdata::{read_jsonl, tokenize, ChatTokenizer, DataFormat};
use crate::model::{start_upcycled, BaseDtype, Ctx, LlamaModel, ModelConfig, MoeSpec, MoeSplit, Target, MOE_INIT_FILE};

/// The weight of a copy's random direction in its router row (the term that picks the copy).
pub const ROUTER_COPY: f32 = 1.0;
/// The weight of a group's importance direction (the term that orders the groups within the chosen copy): small, so
/// it rarely beats a copy's lead (distinct groups) and the chosen weights stay nearly equal.
pub const ROUTER_GROUP: f32 = 0.01;

/// How an upcycle is calibrated.
#[derive(Debug, Clone)]
pub struct InitOptions {
    /// calibration conversations (JSONL); None: contiguous split, seeded router
    pub calib: Option<PathBuf>,
    /// calibration tokens in all (conversations are taken in a seeded order until this many)
    pub calib_tokens: usize,
    /// each conversation cut at this many tokens
    pub max_len: usize,
    pub seed: u64,
    pub base_dtype: BaseDtype,
    pub device: Device,
    /// held-out tokens for the dense-vs-upcycled loss (0: no check)
    pub check_tokens: usize,
    /// k-means rounds
    pub iterations: usize,
}

impl Default for InitOptions {
    fn default() -> Self {
        Self { calib: None, calib_tokens: 4096, max_len: 512, seed: 42, base_dtype: BaseDtype::BF16, device: Device::Cpu, check_tokens: 0, iterations: 20 }
    }
}

/// One block's start, as measured on the calibration tokens.
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct BlockStat {
    pub block: usize,
    /// mean cosine similarity of a neuron's activation pattern to its group's centre: the clustered groups, and the
    /// contiguous ones for comparison
    pub cluster_similarity: f64,
    pub contiguous_similarity: f64,
    /// share of tokens whose top-k experts are k different groups
    pub distinct: f64,
    /// share of the feed-forward's importance (Σ_g importance) the chosen groups hold: the router's start, the best k
    /// groups per token (the most any k groups can), and k/G (any k groups on average)
    pub coverage_router: f64,
    pub coverage_best: f64,
    pub coverage_random: f64,
    /// the busiest expert's share of the assignments over the even share (1: even)
    pub busiest: f64,
}

/// What an upcycle's start came to.
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct InitReport {
    pub split: String,
    pub calibrated: bool,
    pub calib_conversations: usize,
    pub calib_tokens: usize,
    pub seconds: f64,
    pub blocks: Vec<BlockStat>,
    /// dense loss, upcycled-at-start loss, scored tokens (`--dense-start-check`)
    pub check: Option<(f64, f64, usize)>,
    pub peak_rss_bytes: Option<u64>,
}

impl InitReport {
    pub fn mean(&self, f: impl Fn(&BlockStat) -> f64) -> f64 {
        if self.blocks.is_empty() { 0.0 } else { self.blocks.iter().map(f).sum::<f64>() / self.blocks.len() as f64 }
    }

    /// Human lines for the log.
    pub fn lines(&self) -> Vec<String> {
        let mut out = vec![format!("split {} ({}); {} calibration tokens from {} conversations; {:.0} s",
                                   self.split, if self.calibrated { "calibrated" } else { "no calibration text: seeded router" },
                                   self.calib_tokens, self.calib_conversations, self.seconds)];
        if self.calibrated {
            out.push(format!("groups: neuron-to-centre similarity {:.3} clustered vs {:.3} contiguous (mean over blocks)",
                             self.mean(|b| b.cluster_similarity), self.mean(|b| b.contiguous_similarity)));
            out.push(format!("router start: {:.1}% of tokens pick k different groups; the chosen groups hold {:.1}% of the feed-forward's importance (best possible {:.1}%, any k groups {:.1}%); busiest expert {:.2}x the even share",
                             100.0 * self.mean(|b| b.distinct), 100.0 * self.mean(|b| b.coverage_router), 100.0 * self.mean(|b| b.coverage_best),
                             100.0 * self.mean(|b| b.coverage_random), self.mean(|b| b.busiest)));
        }
        if let Some((d, u, n)) = self.check {
            out.push(format!("dense-start check on {n} held-out tokens: dense loss {d:.4}, upcycled at the start {u:.4} ({:+.4}, perplexity x{:.2}); training has this to win back",
                             u - d, (u - d).exp()));
        }
        if let Some(b) = self.peak_rss_bytes {
            out.push(format!("peak memory {:.2} GB", b as f64 / 1e9));
        }
        out
    }
}

fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Standard normal values from a seed (Box–Muller).
fn gaussian(seed: u64, n: usize) -> Vec<f32> {
    let mut state = seed;
    let mut out = Vec::with_capacity(n + 1);
    while out.len() < n {
        let u1 = ((splitmix(&mut state) >> 11) as f64 + 1.0) / (1u64 << 53) as f64;
        let u2 = (splitmix(&mut state) >> 11) as f64 / (1u64 << 53) as f64;
        let r = (-2.0 * u1.ln()).sqrt();
        out.push((r * (2.0 * std::f64::consts::PI * u2).cos()) as f32);
        out.push((r * (2.0 * std::f64::consts::PI * u2).sin()) as f32);
    }
    out.truncate(n);
    out
}

/// The router rows `[E, h]`: row `c·G + g` = ROUTER_COPY · d_c + ROUTER_GROUP · r_g.
pub fn router_rows(groups: usize, copies: usize, h: usize, group_dirs: &[Vec<f32>], copy_dirs: &[Vec<f32>]) -> Vec<f32> {
    let mut out = Vec::with_capacity(groups * copies * h);
    for c in 0..copies {
        for g in 0..groups {
            out.extend((0..h).map(|j| ROUTER_COPY * copy_dirs[c][j] + ROUTER_GROUP * group_dirs[g][j]));
        }
    }
    out
}

/// Random unit-spread directions: `n` vectors of `h`, each N(0, 1/h).
fn random_dirs(seed: u64, n: usize, h: usize) -> Vec<Vec<f32>> {
    let s = 1.0 / (h as f32).sqrt();
    (0..n).map(|i| gaussian(seed ^ (i as u64 + 1).wrapping_mul(0xA24B_AED4_963E_E407), h).into_iter().map(|x| x * s).collect()).collect()
}

/// Write `moe_init.safetensors` for an upcycled folder without calibration: contiguous groups, a seeded router.
pub fn write_uncalibrated(out: &Path, cfg: &ModelConfig, m: MoeSpec, seed: u64) -> Result<()> {
    let (groups, h, ff) = (m.groups(cfg), cfg.hidden_size, cfg.intermediate_size);
    let copies = m.experts / groups;
    let mut t: HashMap<String, Tensor> = HashMap::new();
    for b in 0..cfg.num_hidden_layers {
        let bseed = seed ^ ((b as u64 + 1) << 40);
        let rows = router_rows(groups, copies, h, &random_dirs(bseed ^ 0x6C, groups, h), &random_dirs(bseed ^ 0xC0, copies, h));
        t.insert(format!("perm.{b}"), Tensor::from_vec((0..ff as u32).collect::<Vec<u32>>(), (ff,), &Device::Cpu)?);
        t.insert(format!("router.{b}"), Tensor::from_vec(rows, (m.experts, h), &Device::Cpu)?);
    }
    candle_core::safetensors::save(&t, out.join(MOE_INIT_FILE))?;
    Ok(())
}

/// Rows of `x` `[n, d]` scaled to unit length (zero rows stay zero).
fn unit_rows(x: &Tensor) -> Result<Tensor> {
    let n = (x.sqr()?.sum_keepdim(1)?.sqrt()? + 1e-12)?;
    Ok(x.broadcast_div(&n)?)
}

/// Balanced spherical k-means: `f` `[n, d]` unit rows into `groups` groups of exactly `n / groups`. Returns each
/// row's group.
pub fn balanced_kmeans(f: &Tensor, groups: usize, seed: u64, iterations: usize) -> Result<Vec<usize>> {
    let n = f.dim(0)?;
    let size = n / groups;
    if size * groups != n {
        bail!("{n} rows do not split into {groups} equal groups");
    }
    // farthest-point start: a seeded first centre, then each next the row least like the centres so far
    let mut state = seed ^ 0x6B6D_6561_6E73;
    let mut centres: Vec<usize> = vec![(splitmix(&mut state) % n as u64) as usize];
    let mut closest = vec![f32::NEG_INFINITY; n];
    while centres.len() < groups {
        let c = f.get(*centres.last().unwrap())?;
        let sim = f.matmul(&c.unsqueeze(1)?)?.squeeze(1)?.to_vec1::<f32>()?;
        for (x, s) in closest.iter_mut().zip(sim) {
            *x = x.max(s);
        }
        let next = (0..n).filter(|i| !centres.contains(i)).min_by(|&a, &b| closest[a].total_cmp(&closest[b]).then(a.cmp(&b))).unwrap();
        centres.push(next);
    }
    let idx = Tensor::from_vec(centres.iter().map(|&i| i as u32).collect::<Vec<u32>>(), (groups,), f.device())?;
    let mut c = f.index_select(&idx, 0)?;
    let mut assign = vec![usize::MAX; n];
    for _ in 0..iterations.max(1) {
        let sim = f.matmul(&c.t()?)?.to_vec2::<f32>()?;
        let next = balanced_assign(&sim, groups, size);
        let changed = next != assign;
        assign = next;
        // the centres: each group's mean direction
        let mut onehot = vec![0f32; groups * n];
        for (i, g) in assign.iter().enumerate() {
            onehot[g * n + i] = 1.0 / size as f32;
        }
        c = unit_rows(&Tensor::from_vec(onehot, (groups, n), f.device())?.matmul(f)?)?;
        if !changed {
            break;
        }
    }
    Ok(assign)
}

/// Each row to a group, best similarities first, no group above `size`.
fn balanced_assign(sim: &[Vec<f32>], groups: usize, size: usize) -> Vec<usize> {
    let mut pairs: Vec<(f32, u32, u32)> = sim.iter().enumerate().flat_map(|(i, row)| row.iter().enumerate().map(move |(g, s)| (*s, i as u32, g as u32))).collect();
    pairs.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
    let mut assign = vec![usize::MAX; sim.len()];
    let mut fill = vec![0usize; groups];
    for (_, i, g) in pairs {
        let (i, g) = (i as usize, g as usize);
        if assign[i] == usize::MAX && fill[g] < size {
            assign[i] = g;
            fill[g] += 1;
        }
    }
    assign
}

/// Mean cosine similarity of the rows of `f` (unit) to their group's mean direction.
fn group_similarity(f: &Tensor, assign: &[usize], groups: usize) -> Result<f64> {
    let n = assign.len();
    let mut onehot = vec![0f32; groups * n];
    for (i, g) in assign.iter().enumerate() {
        onehot[g * n + i] = 1.0;
    }
    let c = unit_rows(&Tensor::from_vec(onehot, (groups, n), f.device())?.matmul(f)?)?;
    let sim = f.matmul(&c.t()?)?.to_vec2::<f32>()?;
    Ok(assign.iter().enumerate().map(|(i, g)| sim[i][*g] as f64).sum::<f64>() / n as f64)
}

/// Ridge regression `(HᵀH + λI) R = HᵀZ` by conjugate gradients, each column of Z its own system. `h` `[T, d]`,
/// `z` `[T, G]` -> `[d, G]`.
fn ridge(h: &Tensor, z: &Tensor, lambda: f64, iterations: usize) -> Result<Tensor> {
    let apply = |p: &Tensor| -> Result<Tensor> { Ok((h.t()?.matmul(&h.matmul(p)?)? + (p * lambda)?)?) };
    let b = h.t()?.matmul(z)?;
    let bn = b.sqr()?.sum_keepdim(0)?;
    let mut x = b.zeros_like()?;
    let mut r = b.clone();
    let mut p = r.clone();
    let mut rs = bn.clone();
    for _ in 0..iterations {
        let ap = apply(&p)?;
        let alpha = (&rs / ((&p * &ap)?.sum_keepdim(0)? + 1e-30)?)?;
        x = (x + p.broadcast_mul(&alpha)?)?;
        r = (r - ap.broadcast_mul(&alpha)?)?;
        let rs_new = r.sqr()?.sum_keepdim(0)?;
        let worst = (&rs_new / (&bn + 1e-30)?)?.max_all()?.to_scalar::<f32>()?;
        if worst < 1e-10 {
            break;
        }
        p = (&r + p.broadcast_mul(&(&rs_new / (&rs + 1e-30)?)?)?)?;
        rs = rs_new;
    }
    Ok(x)
}

/// Each column of `x` `[d, n]` scaled so that `h · column` has unit standard deviation over the rows of `h`.
fn unit_spread(h: &Tensor, x: &Tensor) -> Result<Tensor> {
    let y = h.matmul(x)?;
    let mean = y.mean_keepdim(0)?;
    let sd = (y.broadcast_sub(&mean)?.sqr()?.mean_keepdim(0)?.sqrt()? + 1e-12)?;
    Ok(x.broadcast_div(&sd)?)
}

/// The chosen experts per row of `logits` `[T, E]` (top-k, ties to the lower index).
fn top_k(logits: &[Vec<f32>], k: usize) -> Vec<Vec<usize>> {
    logits.iter().map(|p| {
        let mut order: Vec<usize> = (0..p.len()).collect();
        order.sort_by(|&a, &b| p[b].total_cmp(&p[a]).then(a.cmp(&b)));
        order.truncate(k);
        order
    }).collect()
}

/// One block's calibrated start from its inputs `h` `[T, hidden]`, activations `act` `[T, ff]` and down projection
/// `down` `[hidden, ff]` (as stored, the dense order): the permutation, the router `[E, hidden]` and the measures.
pub fn calibrate_block(m: MoeSpec, cfg: &ModelConfig, block: usize, h: &Tensor, act: &Tensor, down: &Tensor, opts: &InitOptions) -> Result<(Vec<u32>, Vec<f32>, BlockStat)> {
    let (groups, fe, hid, ff) = (m.groups(cfg), m.ff_e(cfg), cfg.hidden_size, cfg.intermediate_size);
    let copies = m.experts / groups;
    let bseed = opts.seed ^ ((block as u64 + 1) << 40);
    let mag = act.abs()?;
    let feat = unit_rows(&mag.t()?.contiguous()?)?;
    let contiguous: Vec<usize> = (0..ff).map(|n| n / fe).collect();
    let assign = match m.split {
        MoeSplit::Cluster => balanced_kmeans(&feat, groups, bseed, opts.iterations)?,
        MoeSplit::Contiguous => contiguous.clone(),
    };
    let mut perm: Vec<u32> = Vec::with_capacity(ff);
    for g in 0..groups {
        perm.extend((0..ff).filter(|n| assign[*n] == g).map(|n| n as u32));
    }
    let stat_sim = (group_similarity(&feat, &assign, groups)?, group_similarity(&feat, &contiguous, groups)?);
    drop(feat);
    // importance per token and group: Σ_{n∈g} |act_n| · ‖down column n‖
    let col = down.to_dtype(DType::F32)?.sqr()?.sum(0)?.sqrt()?;
    let weighted = mag.broadcast_mul(&col.unsqueeze(0)?)?;
    let pidx = Tensor::from_vec(perm.clone(), (ff,), h.device())?;
    let t = h.dim(0)?;
    let s = weighted.index_select(&pidx, 1)?.reshape((t, groups, fe))?.sum(2)?;
    drop((weighted, mag));
    let z = {
        let mean = s.mean_keepdim(1)?;
        let c = s.broadcast_sub(&mean)?;
        let sd = (c.sqr()?.mean_keepdim(1)?.sqrt()? + 1e-12)?;
        c.broadcast_div(&sd)?
    };
    let lambda = 0.1 * h.sqr()?.sum_all()?.to_scalar::<f32>()? as f64 / hid as f64;
    let r = unit_spread(h, &ridge(h, &z, lambda, 60)?)?;
    let group_dirs: Vec<Vec<f32>> = r.t()?.contiguous()?.to_vec2::<f32>()?;
    // the copies' directions: random, the inputs' mean direction taken out, unit spread
    let mean = h.mean(0)?;
    let mean_sq = mean.sqr()?.sum_all()?.to_scalar::<f32>()?.max(1e-12);
    let d = Tensor::from_vec(gaussian(bseed ^ 0xC0, hid * copies), (hid, copies), h.device())?;
    let along = mean.unsqueeze(0)?.matmul(&d)?;
    let d = (d - mean.unsqueeze(1)?.matmul(&along)?.affine(1.0 / mean_sq as f64, 0.0)?)?;
    let d = unit_spread(h, &d)?;
    let copy_dirs: Vec<Vec<f32>> = d.t()?.contiguous()?.to_vec2::<f32>()?;
    let rows = router_rows(groups, copies, hid, &group_dirs, &copy_dirs);
    // the measures, on the calibration tokens
    let router = Tensor::from_vec(rows.clone(), (m.experts, hid), h.device())?;
    let chosen = top_k(&h.matmul(&router.t()?)?.to_vec2::<f32>()?, m.top_k);
    let s = s.to_vec2::<f32>()?;
    let (mut distinct, mut cov_r, mut cov_b) = (0usize, 0.0f64, 0.0f64);
    let mut load = vec![0usize; m.experts];
    for (tok, ch) in chosen.iter().enumerate() {
        let gs: Vec<usize> = ch.iter().map(|e| e % groups).collect();
        let mut uniq = gs.clone();
        uniq.sort();
        uniq.dedup();
        if uniq.len() == gs.len() {
            distinct += 1;
        }
        ch.iter().for_each(|e| load[*e] += 1);
        let total: f64 = s[tok].iter().map(|x| *x as f64).sum::<f64>().max(1e-30);
        cov_r += uniq.iter().map(|g| s[tok][*g] as f64).sum::<f64>() / total;
        let mut best = s[tok].clone();
        best.sort_by(|a, b| b.total_cmp(a));
        cov_b += best.iter().take(m.top_k).map(|x| *x as f64).sum::<f64>() / total;
    }
    let even = (t * m.top_k) as f64 / m.experts as f64;
    let stat = BlockStat { block, cluster_similarity: stat_sim.0, contiguous_similarity: stat_sim.1, distinct: distinct as f64 / t as f64,
                           coverage_router: cov_r / t as f64, coverage_best: cov_b / t as f64, coverage_random: m.top_k as f64 / groups as f64,
                           busiest: *load.iter().max().unwrap_or(&0) as f64 / even.max(1e-9) };
    Ok((perm, rows, stat))
}

/// The calibration and held-out conversations: seeded order, the first until `calib_tokens`, the next until
/// `check_tokens`. Returns (calibration tokens, held-out (tokens, supervised), conversations used for calibration).
type Held = Vec<(Vec<u32>, Vec<bool>)>;
fn conversations(model: &Path, data: &Path, opts: &InitOptions) -> Result<(Vec<Vec<u32>>, Held, usize)> {
    let tok = ChatTokenizer::load(model, None)?;
    let (records, _) = read_jsonl(data, DataFormat::Auto)?;
    if records.is_empty() {
        bail!("{}: no usable conversations", data.display());
    }
    let order = crate::finetune::permutation(opts.seed, 0xCA11B, records.len());
    let (mut calib, mut held) = (Vec::new(), Vec::new());
    let (mut n_calib, mut n_held) = (0usize, 0usize);
    for i in order {
        let t = tokenize(&tok, std::slice::from_ref(&records[i]), opts.max_len, false);
        let Some(ex) = t.examples.into_iter().next() else { continue };
        if n_calib < opts.calib_tokens {
            n_calib += ex.tokens.len();
            calib.push(ex.tokens);
        } else if n_held < opts.check_tokens {
            n_held += ex.supervised_count();
            held.push((ex.tokens, ex.supervised));
        } else {
            break;
        }
    }
    let n = calib.len();
    Ok((calib, held, n))
}

/// What a sweep tells its caller.
enum SweepEvent<'a> {
    /// a block ran on a stream: what its feed-forward saw (a probing stream only)
    Probe { block: usize, seen: Vec<(Tensor, Tensor)> },
    /// these blocks are done (all their passes) and still in memory
    Done { blocks: std::ops::Range<usize>, model: &'a LlamaModel },
}

/// Run `inputs` (per stream, per conversation, the hidden state entering block 0) through `dir`'s blocks one at a
/// time (the loop's together), each stream with its own context. Returns the states after the last block.
fn sweep(dir: &Path, cfg: &ModelConfig, dtype: BaseDtype, dev: &Device, mut states: Vec<Vec<Tensor>>, ctxs: &[fn() -> Ctx],
         on: &mut dyn FnMut(SweepEvent) -> Result<()>) -> Result<Vec<Vec<Tensor>>> {
    let n = cfg.num_hidden_layers;
    let mut segments: Vec<(std::ops::Range<usize>, usize)> = Vec::new();
    match cfg.loop_ {
        Some(l) => {
            segments.extend((0..l.start).map(|i| (i..i + 1, 1)));
            segments.push((l.start..l.end, l.passes));
            segments.extend((l.end..n).map(|i| (i..i + 1, 1)));
        }
        None => segments.extend((0..n).map(|i| (i..i + 1, 1))),
    }
    for (range, passes) in segments {
        let model = LlamaModel::load_parts(dir, dtype, dev, range.clone(), false, false)?;
        let gates: Vec<Tensor> = (1..passes).map(|p| model.loop_gate_tensors().remove(&format!("loop.gate.{p}")).ok_or_else(|| anyhow!("loop gate {p}")))
            .collect::<Result<_>>()?;
        for (s, stream) in states.iter_mut().enumerate() {
            let ctx = ctxs[s]();
            for x in stream.iter_mut() {
                let mut run = |x: &Tensor| -> Result<Tensor> {
                    let mut y = x.clone();
                    for b in range.clone() {
                        y = model.run_one_block(b, &y, &ctx)?;
                        let seen = ctx.take_probe();
                        if !seen.is_empty() {
                            let seen = seen.into_iter().map(|(a, b)| Ok((a.to_device(&Device::Cpu)?, b.to_device(&Device::Cpu)?))).collect::<Result<Vec<_>>>()?;
                            on(SweepEvent::Probe { block: b, seen })?;
                        }
                    }
                    Ok(y)
                };
                let mut y = run(x)?;
                for g in &gates {
                    let z = run(&y)?;
                    y = (&y + (z - &y)?.broadcast_mul(&g.to_device(y.device())?)?)?;
                }
                *x = y;
            }
        }
        on(SweepEvent::Done { blocks: range, model: &model })?;
    }
    Ok(states)
}

/// The fine-grained upcycle with calibration: `out` becomes the linked, marked copy of `model` with its
/// `moe_init.safetensors` (and `moe_init.json`, the report). With `opts.calib` None this is [`write_uncalibrated`];
/// with `opts.check_tokens` > 0 the dense-start check runs on held-out conversations.
pub fn upcycle(model: &Path, out: &Path, m: MoeSpec, opts: &InitOptions, log: &mut dyn FnMut(String)) -> Result<InitReport> {
    let t0 = Instant::now();
    let dense = ModelConfig::load(model)?;
    m.check_with(&dense)?;
    if !m.fine(&dense) {
        bail!("not a fine-grained shape (--expert-ff below the intermediate size {}): use make_upcycled", dense.intermediate_size);
    }
    let mut report = InitReport { split: m.split.as_str().into(), ..Default::default() };
    let Some(data) = &opts.calib else {
        if m.split == MoeSplit::Cluster {
            bail!("--split cluster needs calibration conversations (--calib FILE.jsonl); --split contiguous works without");
        }
        start_upcycled(model, out, m)?;
        write_uncalibrated(out, &dense, m, opts.seed)?;
        if opts.check_tokens > 0 {
            bail!("--dense-start-check needs --calib (its held-out conversations come from the same file)");
        }
        report.seconds = t0.elapsed().as_secs_f64();
        std::fs::write(out.join("moe_init.json"), serde_json::to_string_pretty(&report)?)?;
        return Ok(report);
    };
    let (calib, held, n_conv) = conversations(model, data, opts).with_context(|| format!("calibration text {}", data.display()))?;
    report.calibrated = true;
    report.calib_conversations = n_conv;
    report.calib_tokens = calib.iter().map(Vec::len).sum();
    log(format!("calibrating on {} tokens of {n_conv} conversations, block by block ({})", report.calib_tokens, if opts.device.is_cpu() { "CPU" } else { "GPU" }));
    start_upcycled(model, out, m)?;
    let dev = &opts.device;
    let emb = LlamaModel::load_parts(model, opts.base_dtype, dev, 0..0, true, false)?;
    let start: Vec<Tensor> = calib.iter().map(|t| emb.embed_tokens(t)).collect::<Result<_>>()?;
    drop(emb);
    let mut pending: HashMap<usize, Vec<(Tensor, Tensor)>> = HashMap::new();
    let mut init: HashMap<String, Tensor> = HashMap::new();
    let mut stats = Vec::new();
    sweep(model, &dense, opts.base_dtype, dev, vec![start], &[Ctx::probing], &mut |ev| {
        match ev {
            SweepEvent::Probe { block, seen } => pending.entry(block).or_default().extend(seen),
            SweepEvent::Done { blocks, model: mm } => {
                for b in blocks {
                    let seen = pending.remove(&b).ok_or_else(|| anyhow!("block {b}: nothing seen"))?;
                    let h = Tensor::cat(&seen.iter().map(|(h, _)| h).collect::<Vec<_>>(), 0)?;
                    let act = Tensor::cat(&seen.iter().map(|(_, a)| a).collect::<Vec<_>>(), 0)?;
                    drop(seen);
                    let down = mm.block_weight(b, Target::Down).ok_or_else(|| anyhow!("block {b}: no down projection"))?.to_device(&Device::Cpu)?;
                    let (perm, rows, st) = calibrate_block(m, &dense, b, &h, &act, &down, opts)?;
                    log(format!("block {b}: similarity {:.3} (contiguous {:.3}), distinct {:.1}%, coverage {:.1}% (best {:.1}%)", st.cluster_similarity,
                                st.contiguous_similarity, 100.0 * st.distinct, 100.0 * st.coverage_router, 100.0 * st.coverage_best));
                    let ff = perm.len();
                    init.insert(format!("perm.{b}"), Tensor::from_vec(perm, (ff,), &Device::Cpu)?);
                    init.insert(format!("router.{b}"), Tensor::from_vec(rows, (m.experts, dense.hidden_size), &Device::Cpu)?);
                    stats.push(st);
                }
            }
        }
        Ok(())
    })?;
    stats.sort_by_key(|s| s.block);
    report.blocks = stats;
    candle_core::safetensors::save(&init, out.join(MOE_INIT_FILE))?;
    drop(init);
    if opts.check_tokens > 0 {
        if held.is_empty() {
            bail!("--dense-start-check: no held-out conversation is left after the calibration ones");
        }
        report.check = Some(dense_start_check(out, &held, opts.base_dtype, dev, log)?);
    }
    report.seconds = t0.elapsed().as_secs_f64();
    report.peak_rss_bytes = crate::finetune::peak_rss_bytes();
    std::fs::write(out.join("moe_init.json"), serde_json::to_string_pretty(&report)?)?;
    Ok(report)
}

/// The held-out loss of the upcycled folder `dir` run dense (every feed-forward whole) and as the mixture at its
/// start, block by block: (dense, upcycled, scored tokens).
pub fn dense_start_check(dir: &Path, held: &[(Vec<u32>, Vec<bool>)], dtype: BaseDtype, dev: &Device, log: &mut dyn FnMut(String)) -> Result<(f64, f64, usize)> {
    let cfg = ModelConfig::load(dir)?;
    log(format!("dense-start check: {} held-out conversations, dense and upcycled, block by block", held.len()));
    let emb = LlamaModel::load_parts(dir, dtype, dev, 0..0, true, false)?;
    let start: Vec<Tensor> = held.iter().map(|(t, _)| emb.embed_tokens(t)).collect::<Result<_>>()?;
    drop(emb);
    let out = sweep(dir, &cfg, dtype, dev, vec![start.clone(), start], &[Ctx::dense_inference, Ctx::inference], &mut |_| Ok(()))?;
    let head = LlamaModel::load_parts(dir, dtype, dev, 0..0, false, true)?;
    let mut sums = [0.0f64; 2];
    let mut count = 0usize;
    for (i, (t, s)) in held.iter().enumerate() {
        for (k, stream) in out.iter().enumerate() {
            let (l, n) = head.loss_of_state(&stream[i], t, s)?;
            sums[k] += l * n as f64;
            if k == 0 {
                count += n;
            }
        }
    }
    let c = count.max(1) as f64;
    Ok((sums[0] / c, sums[1] / c, count))
}

/// The calibration's cost before it runs: (seconds at `gflops`, peak bytes). The sweep's FLOPs are the dense forward
/// on the calibration tokens (and the check's two passes on the held-out ones); the memory is one block's weights
/// (the loop's blocks together), the embeddings, the tokens' states, and the blocks waiting to be grouped.
pub fn estimate(cfg: &ModelConfig, m: MoeSpec, opts: &InitOptions, gflops: f64) -> (f64, u64) {
    let (h, ff, n) = (cfg.hidden_size as f64, cfg.intermediate_size as f64, cfg.num_hidden_layers);
    let loop_blocks = cfg.loop_.map_or(0, |l| l.end - l.start);
    let passes = cfg.loop_.map_or(1, |l| l.passes);
    let block_runs = (n + loop_blocks * (passes - 1)) as f64;
    let block_flops = 2.0 * (cfg.block_params() - cfg.block_small_params()) as f64;
    let attn = 4.0 * (cfg.num_attention_heads * cfg.head_dim) as f64 * opts.max_len as f64 / 2.0;
    let per_token = block_runs * (block_flops + attn);
    let tokens = opts.calib_tokens as f64 + 2.0 * opts.check_tokens as f64;
    // grouping: k-means (ff x G x T per round) and the ridge (T x h x G per iteration, twice)
    let t = opts.calib_tokens as f64 * passes as f64;
    let groups = m.groups(cfg) as f64;
    let grouping = n as f64 * (opts.iterations as f64 * 2.0 * ff * groups * t + 60.0 * 4.0 * t * h * groups);
    let secs = (per_token * tokens + grouping) / (gflops * 1e9);
    let dt = opts.base_dtype.bytes() as f64;
    let block_bytes = (cfg.block_params() - cfg.block_small_params()) as f64 * dt;
    let weights = block_bytes * loop_blocks.max(1) as f64 + (cfg.vocab_size as f64 * h * dt);
    let states = opts.calib_tokens.max(2 * opts.check_tokens) as f64 * h * 4.0;
    let waiting = loop_blocks.max(1) as f64 * t * (h + ff) * 4.0;
    // one block's grouping: |act|, its transposed unit copy, the weighted one (3 x T x ff) and the sweep's working set
    let work = 3.0 * t * ff * 4.0 + opts.max_len as f64 * (4.0 * h + 3.0 * ff) * 4.0 * 4.0 + (512u64 << 20) as f64;
    (secs, (weights + states + waiting + work) as u64)
}

/// The JSON of a report.
pub fn report_json(r: &InitReport) -> Value {
    json!({"split": r.split, "calibrated": r.calibrated, "calib_tokens": r.calib_tokens, "calib_conversations": r.calib_conversations,
           "seconds": r.seconds, "distinct": r.mean(|b| b.distinct), "coverage_router": r.mean(|b| b.coverage_router),
           "coverage_best": r.mean(|b| b.coverage_best), "coverage_random": r.mean(|b| b.coverage_random),
           "cluster_similarity": r.mean(|b| b.cluster_similarity), "contiguous_similarity": r.mean(|b| b.contiguous_similarity),
           "dense_start_check": r.check.map(|(d, u, n)| json!({"dense_loss": d, "upcycled_loss": u, "gap": u - d, "tokens": n})),
           "peak_rss_bytes": r.peak_rss_bytes})
}
