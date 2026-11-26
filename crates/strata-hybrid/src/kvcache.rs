//! The attention cache's storage on the CPU path (docs/MULA-PLAN.md L5): keys or values of one layer, one row of
//! `hd` values per KV head per token, kept as f16 (the default: llama.cpp's cache type, the values the engine always
//! rounded to), f32, or 8 bits with one f32 scale per row (`--kv q8`: about a quarter of f32). A sliding-window layer
//! keeps only the last `window` tokens (and some slack, so a short rollback stays exact), not the whole text.
//!
//! The device graph keeps the same three encodings (strata-gpu `KvMode`), with the same rounding: f16 round to nearest
//! even, q8 `round(x * 127 / amax)` per row.

use std::sync::LazyLock;
use strata_pack::num::{f16_to_f32, f32_to_f16};

/// How the attention cache stores a value (`--kv f16|f32|q8`).
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum KvType {
    F32,
    #[default]
    F16,
    /// 8-bit values, one f32 scale (amax / 127) per row of a head
    Q8,
}

impl KvType {
    pub fn parse(s: &str) -> Result<KvType, String> {
        match s.to_ascii_lowercase().as_str() {
            "f32" => Ok(KvType::F32),
            "f16" => Ok(KvType::F16),
            "q8" | "q8_0" | "int8" => Ok(KvType::Q8),
            _ => Err(format!("--kv {s}: f16, f32 or q8")),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            KvType::F32 => "f32",
            KvType::F16 => "f16",
            KvType::Q8 => "q8",
        }
    }

    /// The bytes a row of `hd` values takes.
    pub fn row_bytes(self, hd: usize) -> usize {
        match self {
            KvType::F32 => 4 * hd,
            KvType::F16 => 2 * hd,
            KvType::Q8 => hd + 4,
        }
    }
}

/// Every f16's value: a row's values are looked up, not converted bit by bit.
static HALF: LazyLock<Vec<f32>> = LazyLock::new(|| (0..=u16::MAX).map(f16_to_f32).collect());

/// A row quantized to 8 bits: -> its scale. The same arithmetic as strata-gpu's kv_store kernel.
pub fn q8_row(x: &[f32], out: &mut [i8]) -> f32 {
    let amax = x.iter().fold(0f32, |m, v| m.max(v.abs()));
    let d = amax / 127.0;
    let id = if amax > 0.0 { 127.0 / amax } else { 0.0 };
    for (o, v) in out.iter_mut().zip(x) {
        *o = (v * id).round().clamp(-127.0, 127.0) as i8;
    }
    d
}

enum Data {
    F32(Vec<f32>),
    F16(Vec<u16>),
    Q8(Vec<i8>, Vec<f32>),
}

/// Tokens past the window kept so a rollback (prompt lookup's rejected guesses) can still see a whole window.
const SLACK: usize = 256;

/// One layer's keys or values (one pass of the loop's).
pub struct KvStore {
    data: Data,
    hd: usize,
    heads: usize,
    /// the first token kept
    start: usize,
    /// tokens kept
    n: usize,
    window: Option<usize>,
}

impl KvStore {
    pub fn new(ty: KvType, hd: usize, heads: usize, window: Option<usize>) -> KvStore {
        let data = match ty {
            KvType::F32 => Data::F32(Vec::new()),
            KvType::F16 => Data::F16(Vec::new()),
            KvType::Q8 => Data::Q8(Vec::new(), Vec::new()),
        };
        KvStore { data, hd, heads, start: 0, n: 0, window }
    }

    /// One token's rows (heads * hd values).
    pub fn push(&mut self, x: &[f32]) {
        debug_assert_eq!(x.len(), self.hd * self.heads);
        match &mut self.data {
            Data::F32(v) => v.extend_from_slice(x),
            Data::F16(v) => v.extend(x.iter().map(|&f| f32_to_f16(f))),
            Data::Q8(v, s) => {
                let at = v.len();
                v.resize(at + x.len(), 0);
                for (r, row) in x.chunks_exact(self.hd).enumerate() {
                    s.push(q8_row(row, &mut v[at + r * self.hd..at + (r + 1) * self.hd]));
                }
            }
        }
        self.n += 1;
        if let Some(w) = self.window {
            if self.n > w + 2 * SLACK {
                self.drop_front(self.n - w - SLACK);
            }
        }
    }

    fn drop_front(&mut self, tokens: usize) {
        let vals = tokens * self.heads * self.hd;
        match &mut self.data {
            Data::F32(v) => { v.drain(..vals); }
            Data::F16(v) => { v.drain(..vals); }
            Data::Q8(v, s) => {
                v.drain(..vals);
                s.drain(..tokens * self.heads);
            }
        }
        self.start += tokens;
        self.n -= tokens;
    }

    /// One past the last token.
    pub fn end(&self) -> usize {
        self.start + self.n
    }

    /// The first token attention may read at the next query (the window's start), given `n` = end().
    pub fn first_visible(&self) -> usize {
        self.window.map_or(0, |w| self.end().saturating_sub(w))
    }

    /// Back to tokens ..end. A window must still be whole for the token after it.
    pub fn cut(&mut self, end: usize) {
        if end >= self.end() {
            return;
        }
        if let Some(w) = self.window {
            assert!(end + 1 >= self.start + w || self.start == 0, "rolled back past the sliding window's slack ({end} < {} + {w})", self.start);
        }
        let keep = end.saturating_sub(self.start);
        let vals = keep * self.heads * self.hd;
        match &mut self.data {
            Data::F32(v) => v.truncate(vals),
            Data::F16(v) => v.truncate(vals),
            Data::Q8(v, s) => {
                v.truncate(vals);
                s.truncate(keep * self.heads);
            }
        }
        self.n = keep;
    }

    fn row_at(&self, t: usize, h: usize) -> usize {
        debug_assert!(t >= self.start && t < self.end(), "token {t} not kept ({}..{})", self.start, self.end());
        (t - self.start) * self.heads + h
    }

    /// q . the row of token t, head h.
    pub fn dot(&self, t: usize, h: usize, q: &[f32]) -> f32 {
        let r = self.row_at(t, h);
        let hd = self.hd;
        match &self.data {
            Data::F32(v) => crate::model::dot(q, &v[r * hd..(r + 1) * hd]),
            Data::F16(v) => {
                let mut buf = [0f32; 512];
                let row = &mut buf[..hd];
                row.iter_mut().zip(&v[r * hd..(r + 1) * hd]).for_each(|(o, &b)| *o = HALF[b as usize]);
                crate::model::dot(q, row)
            }
            Data::Q8(v, s) => {
                let mut buf = [0f32; 512];
                let row = &mut buf[..hd];
                row.iter_mut().zip(&v[r * hd..(r + 1) * hd]).for_each(|(o, &b)| *o = b as f32);
                crate::model::dot(q, row) * s[r]
            }
        }
    }

    /// out += p * the row of token t, head h.
    pub fn axpy(&self, t: usize, h: usize, p: f32, out: &mut [f32]) {
        let r = self.row_at(t, h);
        let hd = self.hd;
        match &self.data {
            Data::F32(v) => out.iter_mut().zip(&v[r * hd..(r + 1) * hd]).for_each(|(o, x)| *o += p * x),
            Data::F16(v) => out.iter_mut().zip(&v[r * hd..(r + 1) * hd]).for_each(|(o, &b)| *o += p * HALF[b as usize]),
            Data::Q8(v, s) => {
                let ps = p * s[r];
                out.iter_mut().zip(&v[r * hd..(r + 1) * hd]).for_each(|(o, &b)| *o += ps * b as f32)
            }
        }
    }

    /// The bytes held.
    pub fn bytes(&self) -> usize {
        match &self.data {
            Data::F32(v) => v.len() * 4,
            Data::F16(v) => v.len() * 2,
            Data::Q8(v, s) => v.len() + s.len() * 4,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rows(n: usize, hd: usize, heads: usize) -> Vec<Vec<f32>> {
        let mut s = 0x1234_5678u64;
        (0..n).map(|_| (0..hd * heads).map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 24) as f32) * 4.0 - 2.0
        }).collect()).collect()
    }

    #[test]
    fn f16_is_the_rounding_the_engine_always_did_and_q8_is_close() {
        let (hd, heads) = (32, 2);
        let xs = rows(10, hd, heads);
        let q: Vec<f32> = (0..hd).map(|i| (i as f32 * 0.37).sin()).collect();
        let mut st = [KvType::F32, KvType::F16, KvType::Q8].map(|t| KvStore::new(t, hd, heads, None));
        for x in &xs {
            st.iter_mut().for_each(|s| s.push(x));
        }
        for (t, x) in xs.iter().enumerate() {
            for h in 0..heads {
                let row = &x[h * hd..(h + 1) * hd];
                let r16: Vec<f32> = row.iter().map(|&v| f16_to_f32(f32_to_f16(v))).collect();
                assert_eq!(st[1].dot(t, h, &q), crate::model::dot(&q, &r16));
                assert_eq!(st[0].dot(t, h, &q), crate::model::dot(&q, row));
                let amax = row.iter().fold(0f32, |m, v| m.max(v.abs()));
                let bound = q.iter().map(|v| v.abs()).sum::<f32>() * amax / 254.0 * 1.01;
                assert!((st[2].dot(t, h, &q) - st[0].dot(t, h, &q)).abs() <= bound);
            }
        }
        assert_eq!(st[0].bytes(), 10 * 64 * 4);
        assert_eq!(st[1].bytes(), 10 * 64 * 2);
        assert_eq!(st[2].bytes(), 10 * 64 + 10 * 2 * 4);
    }

    #[test]
    fn a_window_keeps_its_tokens_and_cuts_back() {
        let (hd, heads, w) = (4, 1, 8);
        let xs = rows(2000, hd, heads);
        let mut s = KvStore::new(KvType::F32, hd, heads, Some(w));
        for x in &xs {
            s.push(x);
        }
        assert_eq!(s.end(), 2000);
        assert!(s.n <= w + 2 * SLACK && s.n >= w + SLACK);
        assert_eq!(s.first_visible(), 2000 - w);
        let q = [1.0, 0.0, 0.0, 0.0];
        for t in s.first_visible()..2000 {
            assert_eq!(s.dot(t, 0, &q), xs[t][0]);
        }
        s.cut(1990);
        assert_eq!(s.end(), 1990);
        assert_eq!(s.dot(1989 - w, 0, &q), xs[1989 - w][0]);
    }
}
