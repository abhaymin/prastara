//! The kernels, in CubeCL: written once, compiled at run time for CUDA, HIP, Vulkan (wgpu) or the CPU.
//!
//! Weights stay in their GGUF encoding on the device, uploaded as bytes in a `u32` array (k-quant blocks are not
//! 4-byte aligned: Q6_K's are 210 bytes), and are dequantized as they are read, exactly as strata_pack::quants does on
//! the CPU. The activations arrive already rounded the way ggml rounds them (the caller's Vdt), so the products
//! are the CPU engine's; only the order of the sums differs.

#![allow(unused_assignments)] // a `let mut` given a value in every branch: how CubeCL kernels declare it

use cubecl::prelude::*;

/// Units per cube for the row kernels (a power of two, at most every backend's limit).
pub const CUBE: u32 = 128;

/// Weight encodings with a kernel.
pub mod kind {
    pub const F32: u32 = 0;
    pub const F16: u32 = 1;
    pub const BF16: u32 = 2;
    pub const Q8_0: u32 = 3;
    pub const Q4_0: u32 = 4;
    pub const Q4_K: u32 = 5;
    pub const Q5_K: u32 = 6;
    pub const Q6_K: u32 = 7;
    /// IQ3_XXS: 256 values in 98 bytes (an f16 scale, 64 grid indices, 8 words of signs and scales). Its 256-entry
    /// grid travels at the front of the matrix's own buffer ([`TABLE_BYTES`]), so every backend reads it the same way.
    pub const IQ3_XXS: u32 = 8;
    /// Q5_0: 32 values in 22 bytes (an f16 scale, 4 bytes of fifth bits, 16 bytes of nibbles), values minus 16.
    pub const Q5_0: u32 = 9;
}

/// (values per block, bytes per block, kind) of a GGUF type name this crate has a kernel for.
pub fn geometry(ty: &str) -> Option<(u32, u32, u32)> {
    Some(match ty {
        "F32" => (32, 128, kind::F32),
        "F16" => (32, 64, kind::F16),
        "BF16" => (32, 64, kind::BF16),
        "Q8_0" => (32, 34, kind::Q8_0),
        "Q4_0" => (32, 18, kind::Q4_0),
        "Q5_0" => (32, 22, kind::Q5_0),
        "Q4_K" => (256, 144, kind::Q4_K),
        "Q5_K" => (256, 176, kind::Q5_K),
        "Q6_K" => (256, 210, kind::Q6_K),
        "IQ3_XXS" => (256, 98, kind::IQ3_XXS),
        _ => return None,
    })
}

/// Bytes of lookup table at the front of a matrix of kind `k` (its grid), before its first row.
pub const fn table_bytes(k: u32) -> u32 {
    if k == kind::IQ3_XXS { 1024 } else { 0 }
}

/// The table a matrix of kind `ty` carries at its front, as little-endian bytes (empty for most kinds).
pub fn table_for(ty: &str) -> Vec<u8> {
    match ty {
        "IQ3_XXS" => strata_pack::grids::IQ3XXS_GRID.iter().flat_map(|g| g.to_le_bytes()).collect(),
        _ => Vec::new(),
    }
}

/// IQ2_XS's sign table, which IQ3_XXS uses too: the 7 bits given, and an eighth that makes the count of minus signs even.
#[cube]
fn ksigns(i: u32) -> u32 {
    let mut p = i ^ (i >> 4);
    p ^= p >> 2;
    p ^= p >> 1;
    i | ((p & 1) << 7)
}

/// Byte `off` of the weights.
#[cube]
fn byte(w: &Array<u32>, off: u32) -> u32 {
    (w[(off / 4) as usize] >> ((off % 4) * 8)) & 0xff
}

/// The f16 at byte `off` (little-endian), as f32.
#[cube]
fn half(w: &Array<u32>, off: u32) -> f32 {
    half_bits(byte(w, off) | (byte(w, off + 1) << 8))
}

/// An f16's bits as f32.
#[cube]
fn half_bits(h: u32) -> f32 {
    let e = (h >> 10) & 31;
    let m = h & 1023;
    let mut v = f32::new(0.0);
    if e == 0 {
        v = f32::cast_from(m) * 5.9604645e-8f32; // subnormal: m * 2^-24
    } else if e == 31 {
        v = f32::reinterpret(0x7f80_0000u32 | (m << 13));
    } else {
        v = f32::reinterpret(((e + 112) << 23) | (m << 13));
    }
    if (h >> 15) == 1 {
        v = -v;
    }
    v
}

/// A byte as a signed 8-bit value.
#[cube]
fn signed(b: u32) -> f32 {
    f32::cast_from(b) - f32::cast_from((b >> 7) * 256)
}

/// The dot product of one block (at byte `o`) with x[xo..].
#[cube]
fn block_dot(w: &Array<u32>, o: u32, x: &Array<f32>, xo: u32, #[comptime] k: u32) -> f32 {
    let mut acc = f32::new(0.0);
    if comptime![k == kind::F32] {
        for j in 0..32u32 {
            let bits = byte(w, o + 4 * j) | (byte(w, o + 4 * j + 1) << 8) | (byte(w, o + 4 * j + 2) << 16) | (byte(w, o + 4 * j + 3) << 24);
            acc += f32::reinterpret(bits) * x[(xo + j) as usize];
        }
    } else if comptime![k == kind::F16] {
        for j in 0..32u32 {
            acc += half(w, o + 2 * j) * x[(xo + j) as usize];
        }
    } else if comptime![k == kind::BF16] {
        for j in 0..32u32 {
            let bits = (byte(w, o + 2 * j) | (byte(w, o + 2 * j + 1) << 8)) << 16;
            acc += f32::reinterpret(bits) * x[(xo + j) as usize];
        }
    } else if comptime![k == kind::Q8_0] {
        let d = half(w, o);
        for j in 0..32u32 {
            acc += signed(byte(w, o + 2 + j)) * x[(xo + j) as usize];
        }
        acc *= d;
    } else if comptime![k == kind::Q4_0] {
        let d = half(w, o);
        for j in 0..16u32 {
            let b = byte(w, o + 2 + j);
            acc += (f32::cast_from(b & 0x0f) - 8.0) * x[(xo + j) as usize];
            acc += (f32::cast_from(b >> 4) - 8.0) * x[(xo + 16 + j) as usize];
        }
        acc *= d;
    } else if comptime![k == kind::Q5_0] {
        let d = half(w, o);
        let qh = byte(w, o + 2) | (byte(w, o + 3) << 8) | (byte(w, o + 4) << 16) | (byte(w, o + 5) << 24);
        for j in 0..16u32 {
            let b = byte(w, o + 6 + j);
            let lo = (b & 0x0f) | (((qh >> j) & 1) << 4);
            let hi = (b >> 4) | (((qh >> (j + 16)) & 1) << 4);
            acc += (f32::cast_from(lo) - 16.0) * x[(xo + j) as usize];
            acc += (f32::cast_from(hi) - 16.0) * x[(xo + 16 + j) as usize];
        }
        acc *= d;
    } else if comptime![k == kind::Q4_K || k == kind::Q5_K] {
        // 8 groups of 32: a 6-bit scale and min each (packed in 12 bytes), then 4-bit values (+ a 5th bit for Q5_K)
        let d = half(w, o);
        let dmin = half(w, o + 2);
        let s = o + 4;
        let qs = if comptime![k == kind::Q5_K] { o + 48 } else { o + 16 };
        let qh = o + 16;
        for g in 0..8u32 {
            let j = g % 4;
            let mut sc = u32::new(0);
            let mut mn = u32::new(0);
            if g < 4 {
                sc = byte(w, s + j) & 0x3f;
                mn = byte(w, s + 4 + j) & 0x3f;
            } else {
                sc = (byte(w, s + 8 + j) & 0x0f) | ((byte(w, s + j) >> 2) & 0x30);
                mn = (byte(w, s + 8 + j) >> 4) | ((byte(w, s + 4 + j) >> 2) & 0x30);
            }
            let dd = d * f32::cast_from(sc);
            let dm = dmin * f32::cast_from(mn);
            let c = g / 2;
            let sh = (g % 2) * 4;
            let mut part = f32::new(0.0);
            let mut xs = f32::new(0.0);
            for l in 0..32u32 {
                let mut q = (byte(w, qs + c * 32 + l) >> sh) & 0x0f;
                if comptime![k == kind::Q5_K] {
                    q |= ((byte(w, qh + l) >> g) & 1) << 4;
                }
                let xv = x[(xo + g * 32 + l) as usize];
                part += f32::cast_from(q) * xv;
                xs += xv;
            }
            acc += dd * part - dm * xs;
        }
    } else if comptime![k == kind::Q6_K] {
        // 16 groups of 16 with signed 8-bit scales; 4 low bits in ql, 2 high bits in qh
        let d = half(w, o + 208);
        for hh in 0..2u32 {
            for kk in 0..4u32 {
                for l in 0..32u32 {
                    let lo = (byte(w, o + hh * 64 + (kk % 2) * 32 + l) >> (4 * (kk / 2))) & 0x0f;
                    let hi = (byte(w, o + 128 + hh * 32 + l) >> (2 * kk)) & 0x03;
                    let q = f32::cast_from(lo | (hi << 4)) - 32.0;
                    let e = (hh * 4 + kk) * 32 + l;
                    acc += d * signed(byte(w, o + 192 + e / 16)) * q * x[(xo + e) as usize];
                }
            }
        }
    }
    acc
}

/// The dot product of 32 values - group `g` of the block at byte `o` - with x[xo..xo + 32]: the unit of work of
/// [`matvec`] (a row's groups are spread over 32 units, so even a short row keeps them busy).
#[cube]
fn group_dot(w: &Array<u32>, o: u32, g: u32, x: &Array<f32>, xo: u32, #[comptime] k: u32) -> f32 {
    let mut acc = f32::new(0.0);
    if comptime![k == kind::F32 || k == kind::F16 || k == kind::BF16 || k == kind::Q8_0 || k == kind::Q4_0 || k == kind::Q5_0] {
        // these blocks are 32 values: the group is the block
        acc = block_dot(w, o, x, xo, k);
    } else if comptime![k == kind::Q4_K || k == kind::Q5_K] {
        let d = half(w, o);
        let dmin = half(w, o + 2);
        let s = o + 4;
        let j = g % 4;
        let mut sc = u32::new(0);
        let mut mn = u32::new(0);
        if g < 4 {
            sc = byte(w, s + j) & 0x3f;
            mn = byte(w, s + 4 + j) & 0x3f;
        } else {
            sc = (byte(w, s + 8 + j) & 0x0f) | ((byte(w, s + j) >> 2) & 0x30);
            mn = (byte(w, s + 8 + j) >> 4) | ((byte(w, s + 4 + j) >> 2) & 0x30);
        }
        let qs = if comptime![k == kind::Q5_K] { o + 48 } else { o + 16 } + (g / 2) * 32;
        let sh = (g % 2) * 4;
        let mut part = f32::new(0.0);
        let mut xs = f32::new(0.0);
        // the blocks are 144 / 176 bytes: whole words, four values per load
        for i in 0..8u32 {
            let word = w[(qs / 4 + i) as usize];
            let mut hb = u32::new(0);
            if comptime![k == kind::Q5_K] {
                hb = w[((o + 16) / 4 + i) as usize];
            }
            for b in 0..4u32 {
                let l = i * 4 + b;
                let mut q = (word >> (b * 8 + sh)) & 0x0f;
                if comptime![k == kind::Q5_K] {
                    q |= ((hb >> (b * 8 + g)) & 1) << 4;
                }
                let xv = x[(xo + l) as usize];
                part += f32::cast_from(q) * xv;
                xs += xv;
            }
        }
        acc = d * f32::cast_from(sc) * part - dmin * f32::cast_from(mn) * xs;
    } else if comptime![k == kind::IQ3_XXS] {
        // group g of 8: its 8 grid indices (each 4 values) at 2 + 8g, its signs and scale in the word at 66 + 4g
        let d = half(w, o);
        let sw = byte(w, o + 66 + 4 * g) | (byte(w, o + 67 + 4 * g) << 8) | (byte(w, o + 68 + 4 * g) << 16) | (byte(w, o + 69 + 4 * g) << 24);
        let db = d * (f32::new(0.5) + f32::cast_from(sw >> 28)) * f32::new(0.5);
        for l in 0..4u32 {
            let ks = ksigns((sw >> (7 * l)) & 0x7f);
            for j in 0..8u32 {
                let gi = byte(w, o + 2 + 8 * g + 2 * l + j / 4);
                let gv = (w[gi as usize] >> (8 * (j % 4))) & 0xff; // the grid is the buffer's first 256 words
                let s = f32::new(1.0) - f32::new(2.0) * f32::cast_from((ks >> j) & 1);
                acc += db * f32::cast_from(gv) * s * x[(xo + 8 * l + j) as usize];
            }
        }
    } else if comptime![k == kind::Q6_K] {
        let d = half(w, o + 208);
        let hh = g / 4;
        let kk = g % 4;
        let lo_at = o + hh * 64 + (kk % 2) * 32;
        let lo_sh = 4 * (kk / 2);
        let hi_at = o + 128 + hh * 32;
        let s0 = d * signed(byte(w, o + 192 + g * 2));
        let s1 = d * signed(byte(w, o + 192 + g * 2 + 1));
        for l in 0..32u32 {
            let lo = (byte(w, lo_at + l) >> lo_sh) & 0x0f;
            let hi = (byte(w, hi_at + l) >> (2 * kk)) & 0x03;
            let q = f32::cast_from(lo | (hi << 4)) - 32.0;
            let sc = if l < 16 { s0 } else { s1 };
            acc += sc * q * x[(xo + l) as usize];
        }
    }
    acc
}

/// Rows per cube in [`matvec`], 32 units each.
pub const ROWS_PER_CUBE: u32 = CUBE / 32;

/// y[t * rows + r] = W[r] . x[t * cols ..], for `tokens` vectors at once (1 for decoding, the prompt's for prefill).
/// A cube takes ROWS_PER_CUBE rows; each row's 32-value groups are spread over 32 units, summed in shared memory.
#[cube(launch_unchecked)]
pub fn matvec(w: &Array<u32>, x: &Array<f32>, y: &mut Array<f32>, rows: u32, cols: u32, row_bytes: u32,
              #[comptime] values_per_block: u32, #[comptime] bytes_per_block: u32, #[comptime] k: u32) {
    let cube = CUBE_POS_X + CUBE_POS_Y * CUBE_COUNT_X;
    let lane = UNIT_POS % 32;
    let row = cube * ROWS_PER_CUBE + UNIT_POS / 32;
    let t = CUBE_POS_Z;
    let groups_per_block = values_per_block / 32;
    let mut acc = f32::new(0.0);
    if row < rows {
        let ng = cols / 32;
        let mut gi = lane;
        while gi < ng {
            let b = gi / groups_per_block;
            let g = gi % groups_per_block;
            acc += group_dot(w, comptime![table_bytes(k)] + row * row_bytes + b * bytes_per_block, g, x, t * cols + gi * 32, k);
            gi += 32;
        }
    }
    let mut sum = SharedMemory::<f32>::new(CUBE as usize);
    sum[UNIT_POS as usize] = acc;
    sync_cube();
    let mut stride = u32::new(16);
    while stride > 0 {
        if lane < stride {
            sum[UNIT_POS as usize] = sum[UNIT_POS as usize] + sum[(UNIT_POS + stride) as usize];
        }
        sync_cube();
        stride /= 2;
    }
    if lane == 0 && row < rows {
        y[(t * rows + row) as usize] = sum[UNIT_POS as usize];
    }
}

/// Units in a cube of [`matvec_q4k`]: 32 rows of 32 units.
pub const WIDE_CUBE: u32 = 1024;
/// The shared memory [`matvec_q4k`] takes for `cols` columns: x and the cube's partial sums.
/// A device whose limit is smaller (Metal: 32 KiB) gives wrong results with it, so the plain kernel is used there.
pub fn wide_shared_bytes(cols: u32) -> usize {
    (cols as usize + WIDE_CUBE as usize) * 4
}

/// Rows from which [`matvec_q4k`] takes a Q4_K product.
pub const WIDE_ROWS: u32 = 256;

/// The 6-bit scale (`min == false`) or min of group `g` of a Q4_K/Q5_K block, from its three scale words.
#[cube]
fn k4_scale(s0: u32, s1: u32, s2: u32, g: u32, #[comptime] min: bool) -> u32 {
    let at = (g % 4) * 8;
    let lo = (s0 >> at) & 0xff;
    let hi = (s1 >> at) & 0xff;
    let mut v = u32::new(0);
    if g < 4 {
        v = if comptime![min] { hi & 0x3f } else { lo & 0x3f };
    } else {
        let more = (s2 >> at) & 0xff;
        v = if comptime![min] { (more >> 4) | ((hi >> 2) & 0x30) } else { (more & 0x0f) | ((lo >> 2) & 0x30) };
    }
    v
}

/// The dot products of up to 32 rows of a Q4_K matrix, with x in shared memory (read once per cube, not once per
/// weight). A row's 32 units take one block at a time, unit l reading word l of the block's 32 words of nibbles: a
/// warp's loads are 128 contiguous bytes. Word l holds 4 values of groups 2(l/8) (low nibbles) and 2(l/8)+1 (high).
#[cube(launch_unchecked)]
pub fn matvec_q4k(w: &Array<u32>, x: &Array<f32>, y: &mut Array<f32>, rows: u32, row_bytes: u32, #[comptime] cols: u32) {
    let row = (CUBE_POS_X + CUBE_POS_Y * CUBE_COUNT_X) * 32 + UNIT_POS / 32;
    let lane = UNIT_POS % 32;
    let t = CUBE_POS_Z;
    let mut sx = SharedMemory::<f32>::new(comptime![cols as usize]);
    let mut sum = SharedMemory::<f32>::new(WIDE_CUBE as usize);
    let mut at = UNIT_POS;
    while at < comptime![cols] {
        sx[at as usize] = x[(t * comptime![cols] + at) as usize];
        at += WIDE_CUBE;
    }
    sync_cube();
    let mut acc = f32::new(0.0);
    if row < rows {
        let ga = (lane / 8) * 2;
        let gb = ga + 1;
        let p = (lane % 8) * 4;
        let nb = row_bytes / 144;
        let mut b = u32::new(0);
        while b < nb {
            let o = (row * row_bytes + b * 144) / 4;
            let head = w[o as usize];
            let d = half_bits(head & 0xffff);
            let dmin = half_bits(head >> 16);
            let (s0, s1, s2) = (w[(o + 1) as usize], w[(o + 2) as usize], w[(o + 3) as usize]);
            let word = w[(o + 4 + lane) as usize];
            let xa = b * 256 + ga * 32 + p;
            let xb = xa + 32;
            let mut pa = f32::new(0.0);
            let mut sa = f32::new(0.0);
            let mut pb = f32::new(0.0);
            let mut sb = f32::new(0.0);
            for k in 0..4u32 {
                let va = sx[(xa + k) as usize];
                let vb = sx[(xb + k) as usize];
                pa += f32::cast_from((word >> (k * 8)) & 0x0f) * va;
                sa += va;
                pb += f32::cast_from((word >> (k * 8 + 4)) & 0x0f) * vb;
                sb += vb;
            }
            acc += d * f32::cast_from(k4_scale(s0, s1, s2, ga, false)) * pa - dmin * f32::cast_from(k4_scale(s0, s1, s2, ga, true)) * sa
                + d * f32::cast_from(k4_scale(s0, s1, s2, gb, false)) * pb - dmin * f32::cast_from(k4_scale(s0, s1, s2, gb, true)) * sb;
            b += 1;
        }
    }
    sum[UNIT_POS as usize] = acc;
    sync_cube();
    let mut stride = u32::new(16);
    while stride > 0 {
        if lane < stride {
            sum[UNIT_POS as usize] = sum[UNIT_POS as usize] + sum[(UNIT_POS + stride) as usize];
        }
        sync_cube();
        stride /= 2;
    }
    if lane == 0 && row < rows {
        y[(t * rows + row) as usize] = sum[UNIT_POS as usize];
    }
}

/// The 4 bytes at an even byte offset `off` as one word: Q6_K blocks are 210 bytes, so a block (and every 2-byte field
/// in it) starts on a 2-byte boundary but only every other one on a word. Two loads at most.
#[cube]
fn word_at_even(w: &Array<u32>, off: u32) -> u32 {
    let i = off / 4;
    let mut v = w[i as usize];
    if off % 4 != 0 {
        v = (v >> 16) | (w[(i + 1) as usize] << 16);
    }
    v
}

/// The shared memory [`matvec_q6k`] takes for `cols` columns: x and the cube's partial sums.
pub fn wide_q6k_shared_bytes(cols: u32) -> usize {
    (cols as usize + WIDE_CUBE as usize) * 4
}

/// [`matvec_q4k`]'s design for Q6_K: up to 32 rows per cube of 1024 units, x in shared memory, a row's 32 units on one
/// block at a time. Unit l reads word l of the block's 128 bytes of low bits (contiguous across the warp) and the word
/// of high bits that goes with it: 4 values at low nibbles, 4 at high, as ggml's dequantize_row_q6_K lays them out.
/// Q6_K holds the output matrix and ffn_down of most k-quant models.
#[cube(launch_unchecked)]
pub fn matvec_q6k(w: &Array<u32>, x: &Array<f32>, y: &mut Array<f32>, rows: u32, row_bytes: u32, #[comptime] cols: u32) {
    let row = (CUBE_POS_X + CUBE_POS_Y * CUBE_COUNT_X) * 32 + UNIT_POS / 32;
    let lane = UNIT_POS % 32;
    let t = CUBE_POS_Z;
    let mut sx = SharedMemory::<f32>::new(comptime![cols as usize]);
    let mut sum = SharedMemory::<f32>::new(WIDE_CUBE as usize);
    let mut at = UNIT_POS;
    while at < comptime![cols] {
        sx[at as usize] = x[(t * comptime![cols] + at) as usize];
        at += WIDE_CUBE;
    }
    sync_cube();
    let mut acc = f32::new(0.0);
    if row < rows {
        // this unit's place in a block: half hh, low-bits chunk c (ql[l] or ql[l + 32]), values l0..l0 + 4
        let hh = lane / 16;
        let c = (lane / 8) % 2;
        let l0 = (lane % 8) * 4;
        let v_lo = hh * 128 + c * 32 + l0;
        let v_hi = v_lo + 64;
        let sh_lo = c * 2;
        let sh_hi = c * 2 + 4;
        let nb = row_bytes / 210;
        let mut b = u32::new(0);
        while b < nb {
            let o = row * row_bytes + b * 210;
            let d = half(w, o + 208);
            let sc_lo = d * signed(byte(w, o + 192 + v_lo / 16));
            let sc_hi = d * signed(byte(w, o + 192 + v_hi / 16));
            let ql = word_at_even(w, o + lane * 4);
            let qh = word_at_even(w, o + 128 + hh * 32 + l0);
            let xo = b * 256;
            let mut p_lo = f32::new(0.0);
            let mut p_hi = f32::new(0.0);
            for k in 0..4u32 {
                let h = (qh >> (k * 8)) & 0xff;
                let q_lo = ((ql >> (k * 8)) & 0x0f) | (((h >> sh_lo) & 0x03) << 4);
                let q_hi = ((ql >> (k * 8 + 4)) & 0x0f) | (((h >> sh_hi) & 0x03) << 4);
                p_lo += (f32::cast_from(q_lo) - 32.0) * sx[(xo + v_lo + k) as usize];
                p_hi += (f32::cast_from(q_hi) - 32.0) * sx[(xo + v_hi + k) as usize];
            }
            acc += sc_lo * p_lo + sc_hi * p_hi;
            b += 1;
        }
    }
    sum[UNIT_POS as usize] = acc;
    sync_cube();
    let mut stride = u32::new(16);
    while stride > 0 {
        if lane < stride {
            sum[UNIT_POS as usize] = sum[UNIT_POS as usize] + sum[(UNIT_POS + stride) as usize];
        }
        sync_cube();
        stride /= 2;
    }
    if lane == 0 && row < rows {
        y[(t * rows + row) as usize] = sum[UNIT_POS as usize];
    }
}

// ------------------------------------------------------------------ a whole token on the device (docs/ENGINE-PLAN.md phase 1)
// These run between the matrix products, so the activations stay on the device and only the logits come back.

/// y = x * w / sqrt(mean(x^2) + eps): the RMS norm of rows of `n` values, a cube per row (CUBE_POS_X): one vector, a
/// chunk's tokens, or a vector's heads (Qwen3's q/k norms, `w` the head's `n` weights).
#[cube(launch_unchecked)]
pub fn rms_norm(x: &Array<f32>, w: &Array<f32>, y: &mut Array<f32>, n: u32, eps: f32) {
    let base = CUBE_POS_X * n;
    let mut sum = SharedMemory::<f32>::new(CUBE as usize);
    let mut acc = f32::new(0.0);
    let mut j = UNIT_POS;
    while j < n {
        let v = x[(base + j) as usize];
        acc += v * v;
        j += CUBE;
    }
    sum[UNIT_POS as usize] = acc;
    sync_cube();
    let mut stride = u32::new(64);
    while stride > 0 {
        if UNIT_POS < stride {
            sum[UNIT_POS as usize] = sum[UNIT_POS as usize] + sum[(UNIT_POS + stride) as usize];
        }
        sync_cube();
        stride /= 2;
    }
    let k = f32::new(1.0) / f32::sqrt(sum[0usize] / f32::cast_from(n) + eps);
    let mut j = UNIT_POS;
    while j < n {
        y[(base + j) as usize] = x[(base + j) as usize] * k * w[j as usize];
        j += CUBE;
    }
}

/// y += x, over `n` values.
#[cube(launch_unchecked)]
pub fn add_into(x: &Array<f32>, y: &mut Array<f32>, n: u32) {
    let j = CUBE_POS_X * CUBE + UNIT_POS;
    if j < n {
        y[j as usize] = y[j as usize] + x[j as usize];
    }
}

/// up *= silu(gate), over `n` values (the gated feed-forward).
#[cube(launch_unchecked)]
pub fn silu_gate(gate: &Array<f32>, up: &mut Array<f32>, n: u32) {
    let j = CUBE_POS_X * CUBE + UNIT_POS;
    if j < n {
        let g = gate[j as usize];
        up[j as usize] = up[j as usize] * (g / (f32::new(1.0) + f32::exp(-g)));
    }
}

/// Rotates each of the head vectors of x (`hd` values each) by the (cos, sin) pairs of `tab`: `n_rot / 2` pairs per
/// token, adjacent values (`neox` 0) or the two halves (`neox` 1). A cube per head (CUBE_POS_X) and token (CUBE_POS_Y):
/// x holds `heads` heads per token, `tab` each token's pairs in turn.
#[cube(launch_unchecked)]
pub fn rope(tab: &Array<f32>, x: &mut Array<f32>, hd: u32, n_rot: u32, neox: u32, heads: u32) {
    let at = (CUBE_POS_Y * heads + CUBE_POS_X) * hd;
    let tb = CUBE_POS_Y * n_rot;
    let mut p = UNIT_POS;
    while p < n_rot / 2 {
        let c = tab[(tb + p * 2) as usize];
        let s = tab[(tb + p * 2 + 1) as usize];
        let mut a = at + p * 2;
        let mut b = a + 1;
        if neox == 1 {
            a = at + p;
            b = a + n_rot / 2;
        }
        let x0 = x[a as usize];
        let x1 = x[b as usize];
        x[a as usize] = x0 * c - x1 * s;
        x[b as usize] = x0 * s + x1 * c;
        p += CUBE;
    }
}

/// cache[at + i] = x[i], over `n` values: one position of a layer's K or V.
#[cube(launch_unchecked)]
pub fn kv_append(x: &Array<f32>, cache: &mut Array<f32>, at: u32, n: u32) {
    let j = CUBE_POS_X * CUBE + UNIT_POS;
    if j < n {
        cache[(at + j) as usize] = x[j as usize];
    }
}

/// Attention scores: s[hh * n + t] = (q_hh . k_t) * scale * ts for every query head and position (a cube per pair).
/// K and V hold `n_kv` heads per position, and query head hh reads KV head hh / group.
#[cube(launch_unchecked)]
pub fn attn_scores(q: &Array<f32>, k: &Array<f32>, s: &mut Array<f32>, n: u32, n_kv: u32, group: u32, hd: u32, scale: f32) {
    let t = CUBE_POS_X;
    let hh = CUBE_POS_Y;
    let kh = hh / group;
    let qo = hh * hd;
    let ko = (t * n_kv + kh) * hd;
    let mut acc = f32::new(0.0);
    let mut j = UNIT_POS;
    while j < hd {
        acc += q[(qo + j) as usize] * k[(ko + j) as usize];
        j += CUBE;
    }
    let mut sum = SharedMemory::<f32>::new(CUBE as usize);
    sum[UNIT_POS as usize] = acc;
    sync_cube();
    let mut stride = u32::new(64);
    while stride > 0 {
        if UNIT_POS < stride {
            sum[UNIT_POS as usize] = sum[UNIT_POS as usize] + sum[(UNIT_POS + stride) as usize];
        }
        sync_cube();
        stride /= 2;
    }
    if UNIT_POS == 0 {
        s[(hh * n + t) as usize] = sum[0usize] * scale;
    }
}

/// s[hh * n..] -> softmax in place, for one query head per cube.
#[cube(launch_unchecked)]
pub fn attn_softmax(s: &mut Array<f32>, n: u32) {
    let hh = CUBE_POS_X;
    let base = hh * n;
    let mut m = SharedMemory::<f32>::new(CUBE as usize);
    let mut top = f32::new(-1.0e30);
    let mut t = UNIT_POS;
    while t < n {
        top = f32::max(top, s[(base + t) as usize]);
        t += CUBE;
    }
    m[UNIT_POS as usize] = top;
    sync_cube();
    let mut stride = u32::new(64);
    while stride > 0 {
        if UNIT_POS < stride {
            m[UNIT_POS as usize] = f32::max(m[UNIT_POS as usize], m[(UNIT_POS + stride) as usize]);
        }
        sync_cube();
        stride /= 2;
    }
    let top = m[0usize];
    sync_cube();
    let mut acc = f32::new(0.0);
    let mut t = UNIT_POS;
    while t < n {
        let e = f32::exp(s[(base + t) as usize] - top);
        s[(base + t) as usize] = e;
        acc += e;
        t += CUBE;
    }
    m[UNIT_POS as usize] = acc;
    sync_cube();
    let mut stride = u32::new(64);
    while stride > 0 {
        if UNIT_POS < stride {
            m[UNIT_POS as usize] = m[UNIT_POS as usize] + m[(UNIT_POS + stride) as usize];
        }
        sync_cube();
        stride /= 2;
    }
    let total = m[0usize];
    let mut t = UNIT_POS;
    while t < n {
        s[(base + t) as usize] = s[(base + t) as usize] / total;
        t += CUBE;
    }
}

/// o[r * hd + j] = sum over t of p[r, t] * v[t, (r % n_head) / group][j]: a cube per row r of scores (a query head of
/// a token: r = token * n_head + head), a unit per `j % CUBE`.
#[cube(launch_unchecked)]
pub fn attn_blend(p: &Array<f32>, v: &Array<f32>, o: &mut Array<f32>, n: u32, n_kv: u32, group: u32, hd: u32, n_head: u32) {
    let r = CUBE_POS_X;
    let kh = (r % n_head) / group;
    let mut j = UNIT_POS;
    while j < hd {
        let mut acc = f32::new(0.0);
        let mut t = u32::new(0);
        while t < n {
            acc += p[(r * n + t) as usize] * v[((t * n_kv + kh) * hd + j) as usize];
            t += 1;
        }
        o[(r * hd + j) as usize] = acc;
        j += CUBE;
    }
}

/// Causal attention scores for a chunk of tokens at positions base..: s[(tok * n_head + hh) * n + t] for t < n, the
/// positions after the token's own masked to -1e30 (softmax gives them 0). A cube per (position, head, token).
#[cube(launch_unchecked)]
pub fn attn_scores_tokens(q: &Array<f32>, k: &Array<f32>, s: &mut Array<f32>, scales: &Array<f32>, n: u32, n_kv: u32, group: u32,
                          hd: u32, n_head: u32, base: u32) {
    let t = CUBE_POS_X;
    let hh = CUBE_POS_Y;
    let tok = CUBE_POS_Z;
    let kh = hh / group;
    let qo = (tok * n_head + hh) * hd;
    let ko = (t * n_kv + kh) * hd;
    let live = t <= base + tok;
    let mut acc = f32::new(0.0);
    if live {
        let mut j = UNIT_POS;
        while j < hd {
            acc += q[(qo + j) as usize] * k[(ko + j) as usize];
            j += CUBE;
        }
    }
    let mut sum = SharedMemory::<f32>::new(CUBE as usize);
    sum[UNIT_POS as usize] = acc;
    sync_cube();
    let mut stride = u32::new(64);
    while stride > 0 {
        if UNIT_POS < stride {
            sum[UNIT_POS as usize] = sum[UNIT_POS as usize] + sum[(UNIT_POS + stride) as usize];
        }
        sync_cube();
        stride /= 2;
    }
    if UNIT_POS == 0 {
        let mut v = f32::new(-1.0e30);
        if live {
            v = sum[0usize] * scales[tok as usize];
        }
        s[((tok * n_head + hh) * n + t) as usize] = v;
    }
}

// ------------------------------------------------------------------ the loop and the experts on the device

/// y[dst + j] = x[src + j], over `n` values.
#[cube(launch_unchecked)]
pub fn copy_range(x: &Array<f32>, y: &mut Array<f32>, src: u32, dst: u32, n: u32) {
    let j = CUBE_POS_X * CUBE + UNIT_POS;
    if j < n {
        y[(dst + j) as usize] = x[(src + j) as usize];
    }
}

/// The loop's mix after an extra pass: x = s + g * (x - s), `g` one value per channel of `dim`, x and s `n` values
/// (whole tokens).
#[cube(launch_unchecked)]
pub fn loop_mix(s: &Array<f32>, x: &mut Array<f32>, g: &Array<f32>, n: u32, dim: u32) {
    let j = CUBE_POS_X * CUBE + UNIT_POS;
    if j < n {
        let sv = s[j as usize];
        x[j as usize] = sv + g[(j % dim) as usize] * (x[j as usize] - sv);
    }
}

/// The router's choice per token (a cube per token, CUBE_POS_X), as model.rs `route` makes it with softmax gating: the
/// probabilities of `n_e` logits, the `k` best (ties to the lower index, as a stable sort), their weights renormalised
/// to sum 1 when `norm` is 1 (the sum at least 2^-14), times `scale` when it is not 0 or 1. -> out[tok * 2k ..]: the
/// experts' indices (as f32, exact below 2^24), then their weights.
#[cube(launch_unchecked)]
pub fn route_topk(logits: &Array<f32>, out: &mut Array<f32>, n_e: u32, k: u32, norm: u32, scale: f32) {
    if UNIT_POS == 0 {
        let tok = CUBE_POS_X;
        let base = tok * n_e;
        let ob = tok * 2 * k;
        let mut m = f32::new(-1.0e30);
        for e in 0..n_e {
            m = f32::max(m, logits[(base + e) as usize]);
        }
        let mut total = f32::new(0.0);
        for e in 0..n_e {
            total += f32::exp(logits[(base + e) as usize] - m);
        }
        // each pick is the best below the previous one in the order (p descending, then index ascending)
        let mut prev_p = f32::new(1.0e30);
        let mut prev_e = u32::new(0);
        let mut first = true;
        let mut wsum = f32::new(0.0);
        for s in 0..k {
            let mut best_p = f32::new(-1.0);
            let mut best_e = u32::new(0);
            for e in 0..n_e {
                let p = f32::exp(logits[(base + e) as usize] - m) / total;
                let below = first || p < prev_p || (p == prev_p && e > prev_e);
                if below && p > best_p {
                    best_p = p;
                    best_e = e;
                }
            }
            out[(ob + s) as usize] = f32::cast_from(best_e);
            out[(ob + k + s) as usize] = best_p;
            wsum += best_p;
            prev_p = best_p;
            prev_e = best_e;
            first = false;
        }
        if norm == 1 {
            let d = f32::max(wsum, f32::new(6.103515625e-5));
            for s in 0..k {
                out[(ob + k + s) as usize] = out[(ob + k + s) as usize] / d;
            }
        }
        if scale != f32::new(0.0) && scale != f32::new(1.0) {
            for s in 0..k {
                out[(ob + k + s) as usize] = out[(ob + k + s) as usize] * scale;
            }
        }
    }
}

/// y[r * dim + j] = x[idx[r] * dim + j]: the rows of the tokens an expert takes, `idx` as f32.
#[cube(launch_unchecked)]
pub fn gather_rows(x: &Array<f32>, idx: &Array<f32>, y: &mut Array<f32>, dim: u32) {
    let j = CUBE_POS_X * CUBE + UNIT_POS;
    let r = CUBE_POS_Y;
    if j < dim {
        let from = u32::cast_from(idx[r as usize]);
        y[(r * dim + j) as usize] = x[(from * dim + j) as usize];
    }
}

/// y[idx[r] * dim + j] = x[r * dim + j]: an expert's outputs to their (token, slot) rows, `idx` as f32.
#[cube(launch_unchecked)]
pub fn scatter_rows(x: &Array<f32>, idx: &Array<f32>, y: &mut Array<f32>, dim: u32) {
    let j = CUBE_POS_X * CUBE + UNIT_POS;
    let r = CUBE_POS_Y;
    if j < dim {
        let to = u32::cast_from(idx[r as usize]);
        y[(to * dim + j) as usize] = x[(r * dim + j) as usize];
    }
}

/// x[tok * dim + j] += sum over slots s (in order) of w[tok, s] * slots[(tok * k + s) * dim + j]: the chosen experts'
/// outputs weighted and added in routing order, as model.rs `moe` sums them; `routes` as route_topk writes it.
#[cube(launch_unchecked)]
pub fn moe_combine(slots: &Array<f32>, routes: &Array<f32>, x: &mut Array<f32>, k: u32, dim: u32, n: u32) {
    let i = CUBE_POS_X * CUBE + UNIT_POS;
    if i < n {
        let tok = i / dim;
        let j = i % dim;
        let mut acc = f32::new(0.0);
        for s in 0..k {
            acc += routes[(tok * 2 * k + k + s) as usize] * slots[((tok * k + s) * dim + j) as usize];
        }
        x[i as usize] = x[i as usize] + acc;
    }
}

// ------------------------------------------------------------------ the fast products (docs/MULA-PLAN.md M8: bandwidth)
// Decode (one vector, or a few) reads every weight once per token, so its speed is the card's bandwidth: these load 16
// bytes per unit per load (coalesced across the warp), keep x in L1 (16-byte loads too), and sum a row's warp with one
// reduction at the end. Prompt chunks (many vectors) go through `mm`: a tile of weights dequantized once into shared
// memory and reused for a tile of tokens, so the weights are read once per 64 tokens, not once per token.
//
// Q8_0 is repacked when it is uploaded (Gpu::upload): its 34-byte blocks are not 4-byte aligned, so the values go first
// (rows * cols bytes, 16-byte aligned rows) and the f16 scales after them in a buffer of their own.

/// Rows per cube of the fast decode kernels: a warp of 32 units per row.
pub const MV_ROWS: u32 = 4;
/// Units per cube of [`mm`]: 16 x 16, each a 4 x 4 tile of the output.
pub const MM_CUBE: u32 = 256;
/// The output tile of [`mm`]: rows x tokens, and the columns per step.
pub const MM_M: u32 = 64;
pub const MM_N: u32 = 64;
pub const MM_K: u32 = 64;

type V4 = Vector<u32, Const<4>>;
type F4 = Vector<f32, Const<4>>;

/// The sum of `v` over each warp of 32 units, valid in the warp's unit 0: a plane reduction where the device's planes
/// are 32 wide (CUDA), else through shared memory (the whole cube must call it).
#[cube]
fn warp_sum(v: f32, #[comptime] plane: bool, #[comptime] cube: u32) -> f32 {
    let mut out = v;
    if comptime![plane] {
        out = plane_sum(v);
    } else {
        let mut sum = SharedMemory::<f32>::new(comptime![cube as usize]);
        sum[UNIT_POS as usize] = v;
        sync_cube();
        let lane = UNIT_POS % 32;
        let mut stride = u32::new(16);
        while stride > 0 {
            if lane < stride {
                sum[UNIT_POS as usize] = sum[UNIT_POS as usize] + sum[(UNIT_POS + stride) as usize];
            }
            sync_cube();
            stride /= 2;
        }
        out = sum[UNIT_POS as usize];
    }
    out
}

/// The signed byte `k` of `w` as f32, without an integer conversion: 2^23 + 128 + b has b in its low mantissa bits.
#[cube]
fn i8f(w: u32, #[comptime] k: u32) -> f32 {
    f32::reinterpret(0x4b00_0000u32 | (((w >> comptime![8 * k]) & 0xff) ^ 0x80)) - f32::new(8388736.0)
}

/// The nibble at bit `sh` of `w` as f32 (the same trick).
#[cube]
fn u4f(w: u32, #[comptime] sh: u32) -> f32 {
    f32::reinterpret(0x4b00_0000u32 | ((w >> comptime![sh]) & 0x0f)) - f32::new(8388608.0)
}

/// The 4 signed bytes of `w` against the 4 values of `x`.
#[cube]
fn dot_i8x4(w: u32, x: F4) -> f32 {
    i8f(w, 0u32) * x[0] + i8f(w, 1u32) * x[1] + i8f(w, 2u32) * x[2] + i8f(w, 3u32) * x[3]
}

/// The low (`hi` false) or high nibbles of the 4 bytes of `w` against x; and the sum of x.
#[cube]
fn dot_u4x4(w: u32, x: F4, #[comptime] hi: bool) -> f32 {
    let s = if comptime![hi] { 4u32 } else { 0u32 };
    u4f(w, comptime![s]) * x[0] + u4f(w, comptime![s + 8]) * x[1] + u4f(w, comptime![s + 16]) * x[2] + u4f(w, comptime![s + 24]) * x[3]
}

/// The f16 scale of block `b` of a repacked Q8_0 matrix (two per word).
#[cube]
fn q8_scale(wd: &Array<u32>, b: u32) -> f32 {
    half_bits((wd[(b / 2) as usize] >> ((b % 2) * 16)) & 0xffff)
}

/// This unit's part of row `row`'s product with x[t] (a warp per row; the warp's sum is the product). Q8_0 (repacked:
/// `w` its values, `wd` its scales): each unit 16 values (one 16-byte load) at a time. Q4_K (as in the file, 144-byte
/// blocks: 9 loads of 16 bytes; `wd` unused): 8 units per block, each one 16-byte load of nibbles (16 values of two
/// groups), 4 blocks at a time.
#[cube]
fn row_part(w: &Array<V4>, wd: &Array<u32>, x: &Array<F4>, row: u32, cols: u32, t: u32, lane: u32, #[comptime] k: u32) -> f32 {
    let mut acc = f32::new(0.0);
    if comptime![k == kind::Q8_0] {
        let nc = cols / 16;
        let qb = row * nc;
        let db = row * (cols / 32);
        let xb = t * (cols / 4);
        let mut c = lane;
        while c < nc {
            let q = w[(qb + c) as usize];
            let d = q8_scale(wd, db + c / 2);
            let xo = xb + c * 4;
            let s = dot_i8x4(q[0], x[xo as usize]) + dot_i8x4(q[1], x[(xo + 1) as usize])
                + dot_i8x4(q[2], x[(xo + 2) as usize]) + dot_i8x4(q[3], x[(xo + 3) as usize]);
            acc += d * s;
            c += 32;
        }
    } else {
        let nb = cols / 256;
        let s = lane % 8;
        let ga = (s / 2) * 2;
        let p0 = (s % 2) * 16;
        let mut b = lane / 8;
        while b < nb {
            let blk = (row * nb + b) * 9;
            let h = w[blk as usize];
            let q = w[(blk + 1 + s) as usize];
            let d = half_bits(h[0] & 0xffff);
            let dmin = half_bits(h[0] >> 16);
            // x of group ga at p0.., and of group ga + 1, in 16-byte loads
            let xa = (t * cols + b * 256 + ga * 32 + p0) / 4;
            let xb = xa + 8;
            let (x0, x1, x2, x3) = (x[xa as usize], x[(xa + 1) as usize], x[(xa + 2) as usize], x[(xa + 3) as usize]);
            let (z0, z1, z2, z3) = (x[xb as usize], x[(xb + 1) as usize], x[(xb + 2) as usize], x[(xb + 3) as usize]);
            let pa = dot_u4x4(q[0], x0, false) + dot_u4x4(q[1], x1, false) + dot_u4x4(q[2], x2, false) + dot_u4x4(q[3], x3, false);
            let pb = dot_u4x4(q[0], z0, true) + dot_u4x4(q[1], z1, true) + dot_u4x4(q[2], z2, true) + dot_u4x4(q[3], z3, true);
            let sa = (x0[0] + x0[1] + x0[2] + x0[3]) + (x1[0] + x1[1] + x1[2] + x1[3]) + (x2[0] + x2[1] + x2[2] + x2[3]) + (x3[0] + x3[1] + x3[2] + x3[3]);
            let sb = (z0[0] + z0[1] + z0[2] + z0[3]) + (z1[0] + z1[1] + z1[2] + z1[3]) + (z2[0] + z2[1] + z2[2] + z2[3]) + (z3[0] + z3[1] + z3[2] + z3[3]);
            acc += d * f32::cast_from(k4_scale(h[1], h[2], h[3], ga, false)) * pa - dmin * f32::cast_from(k4_scale(h[1], h[2], h[3], ga, true)) * sa
                + d * f32::cast_from(k4_scale(h[1], h[2], h[3], ga + 1, false)) * pb - dmin * f32::cast_from(k4_scale(h[1], h[2], h[3], ga + 1, true)) * sb;
            b += 4;
        }
    }
    acc
}

/// y[at] = v, or y[at] += v when `add` (the residual added in the product's own kernel).
#[cube]
fn put(y: &mut Array<f32>, at: u32, v: f32, #[comptime] add: bool) {
    if comptime![add] {
        y[at as usize] = y[at as usize] + v;
    } else {
        y[at as usize] = v;
    }
}

/// y[y_off + t * rows + r] (+)= W[r] . x[t], W Q8_0 (repacked) or Q4_K: a warp per row (row_part), t = CUBE_POS_Z.
/// `y_off` puts the result in a larger buffer (V straight into its cache), `add` adds it there (the residual).
#[cube(launch_unchecked)]
pub fn mv_fast(w: &Array<V4>, wd: &Array<u32>, x: &Array<F4>, y: &mut Array<f32>, rows: u32, cols: u32, y_off: u32,
               #[comptime] k: u32, #[comptime] plane: bool, #[comptime] add: bool) {
    let lane = UNIT_POS % 32;
    let row = (CUBE_POS_X + CUBE_POS_Y * CUBE_COUNT_X) * MV_ROWS + UNIT_POS / 32;
    let t = CUBE_POS_Z;
    let mut acc = f32::new(0.0);
    if row < rows {
        acc = row_part(w, wd, x, row, cols, t, lane, k);
    }
    let total = warp_sum(acc, plane, comptime![MV_ROWS * 32]);
    if lane == 0 && row < rows {
        put(y, y_off + t * rows + row, total, add);
    }
}

/// y[t * rows + r] = silu(G[r] . x[t]) * (U[r] . x[t]): the gated feed-forward's two products and its gate in one
/// kernel (the same arithmetic as `silu_gate`), a warp per row of both.
#[cube(launch_unchecked)]
pub fn mv_glu(wg: &Array<V4>, dg: &Array<u32>, wu: &Array<V4>, du: &Array<u32>, x: &Array<F4>, y: &mut Array<f32>, rows: u32, cols: u32,
              #[comptime] k: u32, #[comptime] plane: bool) {
    let lane = UNIT_POS % 32;
    let row = (CUBE_POS_X + CUBE_POS_Y * CUBE_COUNT_X) * MV_ROWS + UNIT_POS / 32;
    let t = CUBE_POS_Z;
    let mut pg = f32::new(0.0);
    let mut pu = f32::new(0.0);
    if row < rows {
        pg = row_part(wg, dg, x, row, cols, t, lane, k);
        pu = row_part(wu, du, x, row, cols, t, lane, k);
    }
    let g = warp_sum(pg, plane, comptime![MV_ROWS * 32]);
    let u = warp_sum(pu, plane, comptime![MV_ROWS * 32]);
    if lane == 0 && row < rows {
        y[(t * rows + row) as usize] = u * (g / (f32::new(1.0) + f32::exp(-g)));
    }
}

/// Three products of the same x in one launch (attention's q, k and v): the first cubes take W0's rows, the next W1's,
/// the last W2's; y2 at `off2` (V straight into its cache). t = CUBE_POS_Z.
#[cube(launch_unchecked)]
pub fn mv3(w0: &Array<V4>, d0: &Array<u32>, w1: &Array<V4>, d1: &Array<u32>, w2: &Array<V4>, d2: &Array<u32>, x: &Array<F4>,
           y0: &mut Array<f32>, y1: &mut Array<f32>, y2: &mut Array<f32>, rows0: u32, rows1: u32, rows2: u32, off2: u32, cols: u32,
           #[comptime] k: u32, #[comptime] plane: bool) {
    let lane = UNIT_POS % 32;
    let cube = CUBE_POS_X + CUBE_POS_Y * CUBE_COUNT_X;
    let t = CUBE_POS_Z;
    let c0 = rows0.div_ceil(MV_ROWS);
    let c1 = rows1.div_ceil(MV_ROWS);
    // the branch is the same for the whole cube, so the reductions in it are reached by all its units
    if cube < c0 {
        let row = cube * MV_ROWS + UNIT_POS / 32;
        let mut acc = f32::new(0.0);
        if row < rows0 {
            acc = row_part(w0, d0, x, row, cols, t, lane, k);
        }
        let total = warp_sum(acc, plane, comptime![MV_ROWS * 32]);
        if lane == 0 && row < rows0 {
            y0[(t * rows0 + row) as usize] = total;
        }
    } else if cube < c0 + c1 {
        let row = (cube - c0) * MV_ROWS + UNIT_POS / 32;
        let mut acc = f32::new(0.0);
        if row < rows1 {
            acc = row_part(w1, d1, x, row, cols, t, lane, k);
        }
        let total = warp_sum(acc, plane, comptime![MV_ROWS * 32]);
        if lane == 0 && row < rows1 {
            y1[(t * rows1 + row) as usize] = total;
        }
    } else {
        let row = (cube - c0 - c1) * MV_ROWS + UNIT_POS / 32;
        let mut acc = f32::new(0.0);
        if row < rows2 {
            acc = row_part(w2, d2, x, row, cols, t, lane, k);
        }
        let total = warp_sum(acc, plane, comptime![MV_ROWS * 32]);
        if lane == 0 && row < rows2 {
            y2[(off2 + t * rows2 + row) as usize] = total;
        }
    }
}

/// Attention's q and k after their products, in one launch: each head (CUBE_POS_X: the n_head heads of q, then the n_kv
/// of k) of each token (CUBE_POS_Y) RMS-normed by its weights when `norm` (Qwen3's q/k norms; the same arithmetic as
/// `rms_norm`), then rotated by the token's (cos, sin) pairs as `rope` rotates; q in place, k into its cache at
/// k_off + (token * n_kv + head) * hd (what kv_append did).
#[cube(launch_unchecked)]
pub fn qk_rope(q: &mut Array<f32>, k: &Array<f32>, qn: &Array<f32>, kn: &Array<f32>, tab: &Array<f32>, cache: &mut Array<f32>,
               hd: u32, n_rot: u32, neox: u32, n_head: u32, n_kv: u32, k_off: u32, eps: f32, #[comptime] norm: bool) {
    let head = CUBE_POS_X;
    let tok = CUBE_POS_Y;
    let is_q = head < n_head;
    let mut src = u32::new(0);
    if is_q {
        src = (tok * n_head + head) * hd;
    } else {
        src = (tok * n_kv + head - n_head) * hd;
    }
    let mut v = SharedMemory::<f32>::new(512usize);
    let mut j = UNIT_POS;
    while j < hd {
        let mut a = f32::new(0.0);
        if is_q {
            a = q[(src + j) as usize];
        } else {
            a = k[(src + j) as usize];
        }
        v[j as usize] = a;
        j += CUBE;
    }
    if comptime![norm] {
        let mut sum = SharedMemory::<f32>::new(CUBE as usize);
        let mut acc = f32::new(0.0);
        let mut j = UNIT_POS;
        while j < hd {
            acc += v[j as usize] * v[j as usize];
            j += CUBE;
        }
        sum[UNIT_POS as usize] = acc;
        sync_cube();
        let mut stride = u32::new(64);
        while stride > 0 {
            if UNIT_POS < stride {
                sum[UNIT_POS as usize] = sum[UNIT_POS as usize] + sum[(UNIT_POS + stride) as usize];
            }
            sync_cube();
            stride /= 2;
        }
        let kk = f32::new(1.0) / f32::sqrt(sum[0usize] / f32::cast_from(hd) + eps);
        let mut j = UNIT_POS;
        while j < hd {
            let mut wv = f32::new(0.0);
            if is_q {
                wv = qn[j as usize];
            } else {
                wv = kn[j as usize];
            }
            v[j as usize] = v[j as usize] * kk * wv;
            j += CUBE;
        }
    }
    sync_cube();
    // the values past n_rot stay as they are; the pairs below it rotate
    let mut j = UNIT_POS;
    while j < hd {
        let mut out = v[j as usize];
        if j < n_rot {
            let mut p = j / 2;
            let mut a = j - j % 2;
            let mut b = a + 1;
            let mut second = j % 2 == 1;
            if neox == 1 {
                let half = n_rot / 2;
                p = j % half;
                a = p;
                b = p + half;
                second = j >= half;
            }
            let c = tab[(tok * n_rot + p * 2) as usize];
            let s = tab[(tok * n_rot + p * 2 + 1) as usize];
            let x0 = v[a as usize];
            let x1 = v[b as usize];
            if second {
                out = x0 * s + x1 * c;
            } else {
                out = x0 * c - x1 * s;
            }
        }
        if is_q {
            q[(src + j) as usize] = out;
        } else {
            cache[(k_off + src + j) as usize] = out;
        }
        j += CUBE;
    }
}

/// Y = W X for a chunk of tokens: y[t * rows + r] = W[r] . x[t]. A cube takes MM_M rows x MM_N tokens; per step of
/// MM_K columns it dequantizes the weights' tile once into shared memory (k-major, 4 rows per cell) and loads the tokens'
/// tile beside it (k-major, 4 tokens per cell), then each unit adds a 4 x 4 tile of products (4 rows, 4 tokens). The
/// weights are read once per MM_N tokens. `k` is Q8_0 (repacked: `w` the values as words, `wd` the scales) or Q4_K
/// (`w` the blocks as words, `wh` the same as 16-byte loads for the headers). cols % MM_K == 0.
#[cube(launch_unchecked)]
pub fn mm(w: &Array<u32>, wh: &Array<V4>, wd: &Array<u32>, x: &Array<F4>, y: &mut Array<f32>, rows: u32, cols: u32, tokens: u32,
          y_off: u32, #[comptime] k: u32, #[comptime] add: bool) {
    let tid = UNIT_POS;
    let r0 = CUBE_POS_X * MM_M;
    let t0 = CUBE_POS_Y * MM_N;
    let mut ws = SharedMemory::<F4>::new(comptime![(MM_K * MM_M / 4) as usize]);
    let mut xs = SharedMemory::<F4>::new(comptime![(MM_K * MM_N / 4) as usize]);
    // this unit's loads: rows 4 lq.. (and tokens 4 lq..) at columns 4 lk..
    let lq = tid % 16;
    let lk = tid / 16;
    // its output tile: rows 4 tm.., tokens 4 tn..
    let tm = tid % 16;
    let tn = tid / 16;
    let mut a0 = F4::empty().fill(0.0);
    let mut a1 = F4::empty().fill(0.0);
    let mut a2 = F4::empty().fill(0.0);
    let mut a3 = F4::empty().fill(0.0);
    let steps = cols / MM_K;
    let c4 = cols / 4;
    for st in 0..steps {
        let k0 = st * MM_K;
        // the weights' tile: 4 values (columns k0 + 4 lk..) of each of 4 rows, stored as 4 cells of 4 rows
        let mut c0 = F4::empty().fill(0.0);
        let mut c1 = F4::empty().fill(0.0);
        let mut c2 = F4::empty().fill(0.0);
        let mut c3 = F4::empty().fill(0.0);
        #[unroll]
        for i in 0..4u32 {
            let row = r0 + 4 * lq + i;
            let mut v0 = f32::new(0.0);
            let mut v1 = f32::new(0.0);
            let mut v2 = f32::new(0.0);
            let mut v3 = f32::new(0.0);
            if row < rows {
                if comptime![k == kind::Q8_0] {
                    let wv = w[((row * cols + k0) / 4 + lk) as usize];
                    let d = q8_scale(wd, row * (cols / 32) + (k0 + 4 * lk) / 32);
                    v0 = d * i8f(wv, 0u32);
                    v1 = d * i8f(wv, 1u32);
                    v2 = d * i8f(wv, 2u32);
                    v3 = d * i8f(wv, 3u32);
                } else {
                    // Q4_K: chunk c (32 bytes) of block bk: low nibbles are group 2c, high ones 2c + 1; columns 4 lk..
                    // are word lk % 8 of the chunk, low nibbles for lk < 8
                    let blk = row * (cols / 256) + k0 / 256;
                    let c = (k0 % 256) / 64;
                    let g = 2 * c + lk / 8;
                    let h = wh[(blk * 9) as usize];
                    let wv = w[(blk * 36 + 4 + c * 8 + lk % 8) as usize];
                    let dd = half_bits(h[0] & 0xffff) * f32::cast_from(k4_scale(h[1], h[2], h[3], g, false));
                    let dm = half_bits(h[0] >> 16) * f32::cast_from(k4_scale(h[1], h[2], h[3], g, true));
                    let sh = (lk / 8) * 4;
                    v0 = dd * f32::cast_from((wv >> sh) & 0x0f) - dm;
                    v1 = dd * f32::cast_from((wv >> (sh + 8)) & 0x0f) - dm;
                    v2 = dd * f32::cast_from((wv >> (sh + 16)) & 0x0f) - dm;
                    v3 = dd * f32::cast_from((wv >> (sh + 24)) & 0x0f) - dm;
                }
            }
            c0[i as usize] = v0;
            c1[i as usize] = v1;
            c2[i as usize] = v2;
            c3[i as usize] = v3;
        }
        let at = (4 * lk) * (MM_M / 4) + lq;
        ws[at as usize] = c0;
        ws[(at + MM_M / 4) as usize] = c1;
        ws[(at + 2 * MM_M / 4) as usize] = c2;
        ws[(at + 3 * MM_M / 4) as usize] = c3;
        // the tokens' tile: 4 values (columns k0 + 4 lk..) of each of 4 tokens, transposed into 4 cells of 4 tokens
        let mut e0 = F4::empty().fill(0.0);
        let mut e1 = F4::empty().fill(0.0);
        let mut e2 = F4::empty().fill(0.0);
        let mut e3 = F4::empty().fill(0.0);
        #[unroll]
        for i in 0..4u32 {
            let tok = t0 + 4 * lq + i;
            let mut xv = F4::empty().fill(0.0);
            if tok < tokens {
                xv = x[(tok * c4 + k0 / 4 + lk) as usize];
            }
            e0[i as usize] = xv[0];
            e1[i as usize] = xv[1];
            e2[i as usize] = xv[2];
            e3[i as usize] = xv[3];
        }
        let at = (4 * lk) * (MM_N / 4) + lq;
        xs[at as usize] = e0;
        xs[(at + MM_N / 4) as usize] = e1;
        xs[(at + 2 * MM_N / 4) as usize] = e2;
        xs[(at + 3 * MM_N / 4) as usize] = e3;
        sync_cube();
        #[unroll]
        for kk in 0..MM_K {
            let av = ws[(kk * (MM_M / 4) + tm) as usize];
            let bv = xs[(kk * (MM_N / 4) + tn) as usize];
            a0 += av * F4::empty().fill(bv[0]);
            a1 += av * F4::empty().fill(bv[1]);
            a2 += av * F4::empty().fill(bv[2]);
            a3 += av * F4::empty().fill(bv[3]);
        }
        sync_cube();
    }
    let r = r0 + 4 * tm;
    let tb = t0 + 4 * tn;
    #[unroll]
    for i in 0..4u32 {
        if r + i < rows {
            if tb < tokens {
                put(y, y_off + tb * rows + r + i, a0[i as usize], add);
            }
            if tb + 1 < tokens {
                put(y, y_off + (tb + 1) * rows + r + i, a1[i as usize], add);
            }
            if tb + 2 < tokens {
                put(y, y_off + (tb + 2) * rows + r + i, a2[i as usize], add);
            }
            if tb + 3 < tokens {
                put(y, y_off + (tb + 3) * rows + r + i, a3[i as usize], add);
            }
        }
    }
}

/// Attention in one kernel (scores, softmax and the blend of V, with a running maximum as flash attention keeps it), for
/// each query head (CUBE_POS_X) of each token (CUBE_POS_Y) of a chunk at positions base..: token `tok` sees positions
/// 0..=base + tok. Four warps split the positions; a warp's 32 units split the head's values (`per` each, hd = 32 per),
/// one plane sum per position. The warps' partial results are merged at the end. The scale is scales[tok] for a chunk
/// (`chunk`), else `scale`. Needs 32-wide planes (CUDA).
#[cube(launch_unchecked)]
pub fn attend_rows(q: &Array<f32>, k: &Array<f32>, v: &Array<f32>, o: &mut Array<f32>, scales: &Array<f32>, base: u32, n_kv: u32,
                   group: u32, n_head: u32, scale: f32, #[comptime] per: u32, #[comptime] chunk: bool) {
    let head = CUBE_POS_X;
    let tok = CUBE_POS_Y;
    let kh = head / group;
    let hd = comptime![per * 32];
    let lane = UNIT_POS % 32;
    let warp = UNIT_POS / 32;
    let n = base + tok + 1;
    let mut sc = scale;
    if comptime![chunk] {
        sc = scales[tok as usize];
    }
    let qo = (tok * n_head + head) * hd;
    let mut qv = Array::<f32>::new(comptime![per as usize]);
    let mut acc = Array::<f32>::new(comptime![per as usize]);
    #[unroll]
    for i in 0..per {
        qv[i as usize] = q[(qo + lane + 32 * i) as usize];
        acc[i as usize] = f32::new(0.0);
    }
    let mut m = f32::new(-1.0e30);
    let mut l = f32::new(0.0);
    let mut t = warp;
    while t < n {
        let ko = (t * n_kv + kh) * hd;
        let mut d = f32::new(0.0);
        #[unroll]
        for i in 0..per {
            d += qv[i as usize] * k[(ko + lane + 32 * i) as usize];
        }
        let s = plane_sum(d) * sc;
        let m2 = f32::max(m, s);
        let c = f32::exp(m - m2);
        let p = f32::exp(s - m2);
        l = l * c + p;
        #[unroll]
        for i in 0..per {
            acc[i as usize] = acc[i as usize] * c + p * v[(ko + lane + 32 * i) as usize];
        }
        m = m2;
        t += 4;
    }
    let mut sm = SharedMemory::<f32>::new(4usize);
    let mut sl = SharedMemory::<f32>::new(4usize);
    let mut sa = SharedMemory::<f32>::new(comptime![(4 * per * 32) as usize]);
    if lane == 0 {
        sm[warp as usize] = m;
        sl[warp as usize] = l;
    }
    #[unroll]
    for i in 0..per {
        sa[(warp * hd + lane + 32 * i) as usize] = acc[i as usize];
    }
    sync_cube();
    let top = f32::max(f32::max(sm[0usize], sm[1usize]), f32::max(sm[2usize], sm[3usize]));
    let e0 = f32::exp(sm[0usize] - top);
    let e1 = f32::exp(sm[1usize] - top);
    let e2 = f32::exp(sm[2usize] - top);
    let e3 = f32::exp(sm[3usize] - top);
    let total = sl[0usize] * e0 + sl[1usize] * e1 + sl[2usize] * e2 + sl[3usize] * e3;
    let mut j = UNIT_POS;
    while j < hd {
        let s = sa[j as usize] * e0 + sa[(hd + j) as usize] * e1 + sa[(2 * hd + j) as usize] * e2 + sa[(3 * hd + j) as usize] * e3;
        o[(qo + j) as usize] = s / total;
        j += 128;
    }
}
