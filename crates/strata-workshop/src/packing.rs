//! Sequence packing and the order of a run's training units.
//!
//! A streamed base costs the same PCIe traffic for a pass whatever its length: every block's weights cross once
//! forward and once in the checkpoint replay. Packing fills each pass: whole conversations are laid end to end into
//! sequences of up to `max_len` tokens ([`pack`]), and the model keeps them apart (a block-diagonal causal mask,
//! positions from 0 for each conversation: [`crate::model::PassLayout`]), so each conversation's hidden states,
//! loss and gradients are those it has alone.
//!
//! [`Schedule`] is the run's order of units, a pure function of the data's lengths, the seed, the epochs and the cap:
//! each epoch's conversations are shuffled by `(seed, epoch)` exactly as an unpacked run orders them, then packed in
//! that order. A resumed run rebuilds it and lands on the same units.

use crate::finetune::permutation;

/// How many sequences stay open for a conversation to fit into while packing (first fit over these). More fill the
/// sequences better and move a conversation further from its shuffled place.
pub const OPEN_SEQUENCES: usize = 8;

/// Pack the conversations `order` (indices into `lens`, in the order they are to be trained) into sequences of at
/// most `max_len` tokens. A conversation is never split: each goes into the first of the last [`OPEN_SEQUENCES`]
/// opened sequences it fits in, or opens a new one; one of `max_len` tokens or more stays alone. The sequences come
/// out in the order they were opened, their conversations in the order they were added.
pub fn pack(lens: &[usize], order: &[usize], max_len: usize) -> Vec<Vec<usize>> {
    let mut seqs: Vec<(Vec<usize>, usize)> = Vec::new();
    let mut open: std::collections::VecDeque<usize> = std::collections::VecDeque::new();
    for &i in order {
        let n = lens[i];
        if n >= max_len {
            seqs.push((vec![i], n));
            continue;
        }
        match open.iter().copied().find(|s| seqs[*s].1 + n <= max_len) {
            Some(s) => {
                seqs[s].0.push(i);
                seqs[s].1 += n;
                if seqs[s].1 == max_len {
                    open.retain(|o| *o != s);
                }
            }
            None => {
                seqs.push((vec![i], n));
                open.push_back(seqs.len() - 1);
                if open.len() > OPEN_SEQUENCES {
                    open.pop_front();
                }
            }
        }
    }
    seqs.into_iter().map(|(s, _)| s).collect()
}

/// The order of a run's training units (micro-batch `m` of the run is [`Schedule::unit`]`(m)`): one conversation each,
/// or (packed) a sequence of conversations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Schedule {
    /// per epoch, its units (each a list of conversation indices)
    epochs: Vec<Vec<Vec<usize>>>,
    /// units the whole run takes: every whole epoch's, and the started share of the last
    total: usize,
    pub packed: bool,
}

impl Schedule {
    /// The units of `epochs` epochs over conversations of `lens` tokens. Unpacked, epoch e is
    /// `permutation(seed, e + 1, n)`, one conversation a unit, and the run takes `ceil(n · epochs)` units; packed, that
    /// same order is packed ([`pack`]) and the run takes each whole epoch's sequences and `ceil` of the share of the
    /// last one's.
    pub fn new(lens: &[usize], seed: u64, epochs: f64, max_len: usize, packed: bool) -> Self {
        let n = lens.len();
        let whole = epochs.ceil().max(1.0) as usize;
        let mut out = Vec::with_capacity(whole);
        for e in 0..whole {
            let order = permutation(seed, e as u64 + 1, n);
            out.push(if packed { pack(lens, &order, max_len) } else { order.into_iter().map(|i| vec![i]).collect() });
        }
        let total = if packed {
            let full = epochs.floor() as usize;
            let mut t: usize = out.iter().take(full).map(Vec::len).sum();
            let frac = epochs - full as f64;
            if frac > 0.0 {
                t += (out[full].len() as f64 * frac).ceil() as usize;
            }
            t.max(1)
        } else {
            (n as f64 * epochs).ceil() as usize
        };
        Self { epochs: out, total, packed }
    }

    /// The units the run takes in all.
    pub fn total(&self) -> usize {
        self.total
    }

    /// The units of the first epoch (the conversations, unpacked; the sequences, packed).
    pub fn per_epoch(&self) -> usize {
        self.epochs.first().map_or(0, Vec::len)
    }

    /// Unit `m` of the run (past the last epoch built, the epochs repeat; a run never asks for those).
    pub fn unit(&self, m: usize) -> &[usize] {
        let mut m = m;
        loop {
            for e in &self.epochs {
                if m < e.len() {
                    return &e[m];
                }
                m -= e.len();
            }
        }
    }

    /// The units of optimizer step `step` with `accumulation` units a step.
    pub fn step_units(&self, step: usize, accumulation: usize) -> Vec<&[usize]> {
        let start = step * accumulation;
        (start..(start + accumulation).min(self.total)).map(|m| self.unit(m)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packing_keeps_conversations_whole_and_within_the_cap() {
        let lens: Vec<usize> = (0..200).map(|i| 20 + (i * 37) % 300).chain([1024, 1500]).collect();
        let order = permutation(3, 1, lens.len());
        let seqs = pack(&lens, &order, 1024);
        let mut seen: Vec<usize> = seqs.iter().flatten().copied().collect();
        seen.sort_unstable();
        assert_eq!(seen, (0..lens.len()).collect::<Vec<_>>(), "every conversation exactly once");
        for s in &seqs {
            let n: usize = s.iter().map(|i| lens[*i]).sum();
            assert!(n <= 1024 || s.len() == 1, "{s:?}: {n} tokens");
        }
        let used: usize = lens.iter().filter(|n| **n < 1024).sum();
        let packed = seqs.iter().filter(|s| s.len() > 1 || lens[s[0]] < 1024).count();
        assert!(used as f64 / (packed * 1024) as f64 > 0.9, "packs well: {used} tokens in {packed} sequences");
    }
}
