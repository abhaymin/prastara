//! Inkling 2's image encoder (mmproj `projector_type inkling2v`) on the CPU, in Rust: a ViT with a CLS token, learned
//! positions on a 32x32 grid resized bilinearly to the picture's grid, 2D RoPE (x on the first half of each head, y on
//! the second), layer scale, then the patches merged 2x2 and a two-layer GELU projector to the text model's width.
//! The math follows llama.cpp's tools/mtmd/models/inkling2v.cpp and the helpers it calls (the parity oracle):
//! - preprocessing (mtmd-image.cpp, inkling2v): no resize unless over the token limit, the picture padded on the
//!   right and bottom to multiples of 40 with raw -1 (normalised with the rest, so -1.0078), (x - 127.5) / 127.5;
//! - every product with an F16 weight takes its input rounded to f16, as ggml's CPU backend does (its vec_dot_type),
//!   and accumulates in f32 (AM2_VIS_F32_ACT=1 keeps the inputs in f32).

use rayon::prelude::*;
use std::path::Path;
use strata_pack::num::{f16_to_f32, f32_to_f16};

pub struct Linear {
    /// [n_out][n_in], row-major
    w: Vec<f32>,
    b: Option<Vec<f32>>,
    n_in: usize,
    n_out: usize,
    /// the weight was F16: its input is rounded to f16 first
    f16: bool,
}

struct Block {
    ln1: (Vec<f32>, Vec<f32>),
    ln2: (Vec<f32>, Vec<f32>),
    ls1: Vec<f32>,
    ls2: Vec<f32>,
    q: Linear,
    k: Linear,
    v: Linear,
    out: Linear,
    up: Linear,
    down: Linear,
}

pub struct Encoder {
    pub patch: usize,
    pub merge: usize,
    pub n_embd: usize,
    pub n_head: usize,
    pub out_dim: usize,
    eps: f32,
    mean: [f32; 3],
    std: [f32; 3],
    /// the image-token limit (mtmd's set_limit_image_tokens(1, 4096), or --max-tokens)
    pub max_tokens: usize,
    patch_embd: Linear,
    /// [side * side][n_embd]
    pos: Vec<f32>,
    side: usize,
    cls: Vec<f32>,
    pre_ln: (Vec<f32>, Vec<f32>),
    blocks: Vec<Block>,
    mm1: Linear,
    mm2: Linear,
    f32_act: bool,
}

/// A preprocessed picture: [height][width][3] normalised floats, both sides multiples of patch * merge.
pub struct Picture {
    pub width: usize,
    pub height: usize,
    pub data: Vec<f32>,
}

fn round16(v: &mut [f32]) {
    v.iter_mut().for_each(|x| *x = f16_to_f32(f32_to_f16(*x)));
}

impl Linear {
    /// y = W x + b for `n` inputs ([n][n_in] -> [n][n_out]).
    pub fn forward(&self, x: &[f32], n: usize, f32_act: bool) -> Vec<f32> {
        debug_assert_eq!(x.len(), n * self.n_in);
        let mut xr;
        let x = if self.f16 && !f32_act {
            xr = x.to_vec();
            round16(&mut xr);
            &xr[..]
        } else {
            x
        };
        // Yt = W X^T in blocks of output rows (contiguous), then transposed with the bias
        let cb = 32usize;
        let mut yt = vec![0f32; self.n_out * n];
        yt.par_chunks_mut(cb * n).enumerate().for_each(|(bi, c)| {
            let rows = c.len() / n;
            let w = &self.w[bi * cb * self.n_in..(bi * cb + rows) * self.n_in];
            unsafe {
                matrixmultiply::sgemm(rows, self.n_in, n, 1.0, w.as_ptr(), self.n_in as isize, 1, x.as_ptr(), 1, self.n_in as isize,
                                      0.0, c.as_mut_ptr(), n as isize, 1);
            }
        });
        let mut y = vec![0f32; n * self.n_out];
        y.par_chunks_mut(self.n_out).enumerate().for_each(|(t, row)| {
            for (o, v) in row.iter_mut().enumerate() {
                *v = yt[o * n + t] + self.b.as_ref().map_or(0.0, |b| b[o]);
            }
        });
        y
    }
}

/// ggml_norm then weight and bias, row by row (the sums in f64, as ggml's ggml_float).
fn layer_norm(x: &[f32], d: usize, w: &[f32], b: &[f32], eps: f32) -> Vec<f32> {
    let mut out = vec![0f32; x.len()];
    out.par_chunks_mut(d).zip(x.par_chunks(d)).for_each(|(o, r)| {
        let mean = (r.iter().map(|v| *v as f64).sum::<f64>() / d as f64) as f32;
        let var = (r.iter().map(|v| ((*v - mean) * (*v - mean)) as f64).sum::<f64>() / d as f64) as f32;
        let k = 1.0 / (var + eps).sqrt();
        for i in 0..d {
            o[i] = (r[i] - mean) * k * w[i] + b[i];
        }
    });
    out
}

fn gelu_erf(v: &mut [f32]) {
    const SQRT_2_INV: f32 = 0.707_106_77;
    v.par_iter_mut().for_each(|x| *x = 0.5 * *x * (1.0 + libm::erff(*x * SQRT_2_INV)));
}

/// ggml_interpolate, GGML_SCALE_MODE_BILINEAR without align-corners: the [side x side] grid of `d`-wide rows (row index
/// y * side + x) resized to [w x h] (row index y * w + x), half-pixel centres, edges clamped.
pub fn resize_bilinear(pos: &[f32], d: usize, side: usize, w: usize, h: usize) -> Vec<f32> {
    let (sf0, sf1) = (w as f32 / side as f32, h as f32 / side as f32);
    let last = side as i64 - 1;
    let mut out = vec![0f32; w * h * d];
    out.par_chunks_mut(d).enumerate().for_each(|(i, o)| {
        let (i0, i1) = (i % w, i / w);
        let y = (i1 as f32 + 0.5) / sf1 - 0.5;
        let y0f = y.floor() as i64;
        let (y0, y1) = (y0f.clamp(0, last), (y0f + 1).clamp(0, last));
        let dy = (y - y0 as f32).clamp(0.0, 1.0);
        let x = (i0 as f32 + 0.5) / sf0 - 0.5;
        let x0f = x.floor() as i64;
        let (x0, x1) = (x0f.clamp(0, last), (x0f + 1).clamp(0, last));
        let dx = (x - x0 as f32).clamp(0.0, 1.0);
        let at = |xx: i64, yy: i64| &pos[(yy as usize * side + xx as usize) * d..][..d];
        let (a, b, c, e) = (at(x0, y0), at(x1, y0), at(x0, y1), at(x1, y1));
        for j in 0..d {
            o[j] = a[j] * (1.0 - dx) * (1.0 - dy) + b[j] * dx * (1.0 - dy) + c[j] * (1.0 - dx) * dy + e[j] * dx * dy;
        }
    });
    out
}

/// build_patch_merge_permute: [hp][wp] rows of `d` -> [(hp/s) * (wp/s)] rows of `d*s*s`, row-major over the merged
/// grid; each merged row is its s x s patches with dy outer and dx inner.
pub fn merge_patches(x: &[f32], d: usize, wp: usize, hp: usize, s: usize) -> Vec<f32> {
    let (wm, hm) = (wp / s, hp / s);
    let mut out = vec![0f32; wm * hm * d * s * s];
    out.par_chunks_mut(d * s * s).enumerate().for_each(|(t, o)| {
        let (xb, yb) = (t % wm, t / wm);
        for dy in 0..s {
            for dx in 0..s {
                let p = (yb * s + dy) * wp + xb * s + dx;
                o[(dy * s + dx) * d..][..d].copy_from_slice(&x[p * d..(p + 1) * d]);
            }
        }
    });
    out
}

/// ggml_rope_ext in normal mode over `n_dims` channels starting at `offs` of a `hd`-wide head, position `p`.
fn rope_part(head: &mut [f32], offs: usize, n_dims: usize, p: f32, theta_scale: f32) {
    let mut theta = p;
    for i in (0..n_dims).step_by(2) {
        let (c, s) = (theta.cos(), theta.sin());
        let (x0, x1) = (head[offs + i], head[offs + i + 1]);
        head[offs + i] = x0 * c - x1 * s;
        head[offs + i + 1] = x0 * s + x1 * c;
        theta *= theta_scale;
    }
}

impl Encoder {
    pub fn load(path: &Path) -> Result<Encoder, String> {
        let g = strata_gguf::Gguf::open(path).map_err(|e| e.to_string())?;
        if g.architecture() != Some("clip") {
            return Err(format!("{}: not an mmproj (architecture {:?})", path.display(), g.architecture()));
        }
        let pt = g.str("clip.vision.projector_type").unwrap_or("");
        if pt != "inkling2v" {
            return Err(format!("{}: projector type {pt:?}; this encoder reads inkling2v", path.display()));
        }
        let f = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let mm = unsafe { memmap2::Mmap::map(&f) }.map_err(|e| e.to_string())?;
        let u = |k: &str| g.u64(&format!("clip.vision.{k}")).map(|v| v as usize).ok_or_else(|| format!("clip.vision.{k} is missing"));
        let fl = |k: &str| g.metadata.get(&format!("clip.vision.{k}")).and_then(|v| v.as_f64()).map(|v| v as f32);
        let arr3 = |k: &str| -> [f32; 3] {
            match g.metadata.get(&format!("clip.vision.{k}")).and_then(|v| v.as_array()) {
                Some(a) if a.len() == 3 => [0, 1, 2].map(|i| a[i].as_f64().unwrap_or(0.5) as f32),
                _ => [0.5; 3],
            }
        };
        let tensor = |name: &str| -> Result<(Vec<f32>, Vec<u64>, bool), String> {
            let t = g.tensors.iter().find(|t| t.name == name).ok_or_else(|| format!("tensor {name} is missing"))?;
            let n = t.expected_bytes().ok_or_else(|| format!("{name}: unsupported type {}", t.type_name()))? as usize;
            let start = (g.data_start + t.offset) as usize;
            let mut out = vec![0f32; t.elements() as usize];
            strata_pack::quants::dequantize_into(&t.type_name(), &mm[start..start + n], &mut out)?;
            Ok((out, t.dims.clone(), t.type_name() == "F16"))
        };
        let vecf = |name: &str| tensor(name).map(|t| t.0);
        let linear = |name: &str, bias: bool| -> Result<Linear, String> {
            let (w, dims, f16) = tensor(&format!("{name}.weight"))?;
            let n_in = dims[0] as usize;
            let n_out = w.len() / n_in;
            let b = if bias || g.tensors.iter().any(|t| t.name == format!("{name}.bias")) {
                Some(vecf(&format!("{name}.bias"))?)
            } else {
                None
            };
            Ok(Linear { w, b, n_in, n_out, f16 })
        };
        let n_embd = u("embedding_length")?;
        let n_layer = u("block_count")?;
        let patch = u("patch_size")?;
        let mut blocks = Vec::with_capacity(n_layer);
        for l in 0..n_layer {
            let p = |n: &str| format!("v.blk.{l}.{n}");
            blocks.push(Block {
                ln1: (vecf(&p("ln1.weight"))?, vecf(&p("ln1.bias"))?),
                ln2: (vecf(&p("ln2.weight"))?, vecf(&p("ln2.bias"))?),
                ls1: vecf(&p("ls1.weight"))?,
                ls2: vecf(&p("ls2.weight"))?,
                q: linear(&p("attn_q"), true)?,
                k: linear(&p("attn_k"), true)?,
                v: linear(&p("attn_v"), true)?,
                out: linear(&p("attn_out"), true)?,
                up: linear(&p("ffn_up"), true)?,
                down: linear(&p("ffn_down"), true)?,
            });
        }
        // the patch embedding as a matrix: [n_embd][3 * patch * patch], channel then row then column (ggml's im2col)
        let (pw, pdims, pf16) = tensor("v.patch_embd.weight")?;
        if pdims != [patch as u64, patch as u64, 3, n_embd as u64] {
            return Err(format!("v.patch_embd.weight: shape {pdims:?}"));
        }
        let patch_embd = Linear { w: pw, b: None, n_in: 3 * patch * patch, n_out: n_embd, f16: pf16 };
        let pos = vecf("v.position_embd.weight")?;
        let side = ((pos.len() / n_embd) as f64).sqrt() as usize;
        if side * side * n_embd != pos.len() {
            return Err("v.position_embd.weight is not a square grid".into());
        }
        let cls: Vec<f32> = vecf("v.class_embd")?.iter().zip(vecf("v.class_position_embd.weight")?).map(|(a, b)| a + b).collect();
        let mm1 = linear("mm.1", false)?;
        let mm2 = linear("mm.2", false)?;
        let merge = u("spatial_merge_size")?;
        if mm1.n_in != n_embd * merge * merge {
            return Err(format!("mm.1 takes {} values, the merged patches have {}", mm1.n_in, n_embd * merge * merge));
        }
        let mean = arr3("image_mean");
        let std = arr3("image_std");
        Ok(Encoder {
            patch, merge, n_embd, n_head: u("attention.head_count")?, out_dim: mm2.n_out,
            eps: fl("attention.layer_norm_epsilon").unwrap_or(1e-5),
            mean, std, max_tokens: 4096,
            patch_embd, pos, side, cls,
            pre_ln: (vecf("v.pre_ln.weight")?, vecf("v.pre_ln.bias")?),
            blocks, mm1, mm2,
            f32_act: std::env::var("AM2_VIS_F32_ACT").ok().as_deref() == Some("1"),
        })
    }

    pub fn align(&self) -> usize {
        self.patch * self.merge
    }

    /// RGB8 pixels -> the encoder's input: resized only when over the token limit, padded to the alignment with raw -1,
    /// all normalised.
    pub fn preprocess(&self, rgb: &[u8], width: usize, height: usize) -> Picture {
        let align = self.align();
        let up = |v: usize| v.div_ceil(align) * align;
        let max_pixels = self.max_tokens * align * align;
        let resized;
        let (src, w, h) = if up(width) * up(height) > max_pixels {
            // calc_size_preserved_ratio: round to the alignment, then shrink to fit
            let beta = ((height as f32 * width as f32) / max_pixels as f32).sqrt();
            let fl = |x: f32| ((x / align as f32).floor() as usize * align).max(align);
            let (nw, nh) = (fl(width as f32 / beta), fl(height as f32 / beta));
            let img = image::RgbImage::from_raw(width as u32, height as u32, rgb.to_vec()).expect("rgb buffer");
            resized = image::imageops::resize(&img, nw as u32, nh as u32, image::imageops::FilterType::CatmullRom).into_raw();
            (&resized[..], nw, nh)
        } else {
            (rgb, width, height)
        };
        let (pw, ph) = (up(w), up(h));
        let mut data = vec![-1.0f32; pw * ph * 3];
        for y in 0..h {
            for x in 0..w {
                for c in 0..3 {
                    data[(y * pw + x) * 3 + c] = src[(y * w + x) * 3 + c] as f32;
                }
            }
        }
        let (mean, std) = (self.mean.map(|m| m * 255.0), self.std.map(|s| s * 255.0));
        for (i, v) in data.iter_mut().enumerate() {
            *v = (*v - mean[i % 3]) / std[i % 3];
        }
        Picture { width: pw, height: ph, data }
    }

    pub fn n_tokens(&self, pic: &Picture) -> usize {
        (pic.width / self.align()) * (pic.height / self.align())
    }

    /// The picture's rows for the text model: [n_tokens][out_dim].
    pub fn encode(&self, pic: &Picture) -> Vec<f32> {
        let (ps, d) = (self.patch, self.n_embd);
        let (wp, hp) = (pic.width / ps, pic.height / ps);
        let n_patches = wp * hp;
        let n = n_patches + 1;
        let act = self.f32_act;
        // im2col: each patch's values, channel, then row, then column
        let k = 3 * ps * ps;
        let mut cols = vec![0f32; n_patches * k];
        cols.par_chunks_mut(k).enumerate().for_each(|(p, o)| {
            let (px, py) = (p % wp, p / wp);
            for c in 0..3 {
                for ky in 0..ps {
                    for kx in 0..ps {
                        o[(c * ps + ky) * ps + kx] = pic.data[((py * ps + ky) * pic.width + px * ps + kx) * 3 + c];
                    }
                }
            }
        });
        let patches = self.patch_embd.forward(&cols, n_patches, act);
        let pos = if wp == self.side && hp == self.side { self.pos.clone() } else { resize_bilinear(&self.pos, d, self.side, wp, hp) };
        let mut x = Vec::with_capacity(n * d);
        x.extend_from_slice(&self.cls);
        x.extend(patches.iter().zip(&pos).map(|(a, b)| a + b));
        let mut x = layer_norm(&x, d, &self.pre_ln.0, &self.pre_ln.1, self.eps);
        // 2D RoPE positions: CLS at (0, 0), the patches 1-indexed
        let posw: Vec<f32> = std::iter::once(0.0).chain((0..n_patches).map(|p| (p % wp + 1) as f32)).collect();
        let posh: Vec<f32> = std::iter::once(0.0).chain((0..n_patches).map(|p| (p / wp + 1) as f32)).collect();
        let hd = d / self.n_head;
        let half = hd / 2;
        let theta_scale = 10000f32.powf(-2.0 / half as f32);
        for blk in &self.blocks {
            let h = layer_norm(&x, d, &blk.ln1.0, &blk.ln1.1, self.eps);
            let mut q = blk.q.forward(&h, n, act);
            let mut kk = blk.k.forward(&h, n, act);
            let v = blk.v.forward(&h, n, act);
            for m in [&mut q, &mut kk] {
                m.par_chunks_mut(d).enumerate().for_each(|(t, row)| {
                    for head in row.chunks_mut(hd) {
                        rope_part(head, 0, half, posw[t], theta_scale);
                        rope_part(head, half, half, posh[t], theta_scale);
                    }
                });
            }
            let att = attention(&q, &kk, &v, n, self.n_head, hd);
            let mut o = blk.out.forward(&att, n, act);
            o.par_chunks_mut(d).zip(x.par_chunks(d)).for_each(|(o, r)| {
                for i in 0..d {
                    o[i] = o[i] * blk.ls1[i] + r[i];
                }
            });
            x = o;
            let h = layer_norm(&x, d, &blk.ln2.0, &blk.ln2.1, self.eps);
            let mut u = blk.up.forward(&h, n, act);
            gelu_erf(&mut u);
            let f = blk.down.forward(&u, n, act);
            x.par_chunks_mut(d).zip(f.par_chunks(d)).for_each(|(r, f)| {
                for i in 0..d {
                    r[i] += f[i] * blk.ls2[i];
                }
            });
        }
        // without the CLS token, merged, projected
        let merged = merge_patches(&x[d..], d, wp, hp, self.merge);
        let n_tok = (wp / self.merge) * (hp / self.merge);
        let mut y = self.mm1.forward(&merged, n_tok, act);
        gelu_erf(&mut y);
        self.mm2.forward(&y, n_tok, act)
    }

    /// Decode an image file (PNG, JPEG, WebP, GIF's first frame, BMP) and encode it.
    pub fn encode_file(&self, path: &Path) -> Result<(Vec<f32>, usize), String> {
        let img = image::ImageReader::open(path).map_err(|e| e.to_string())?.with_guessed_format().map_err(|e| e.to_string())?
            .decode().map_err(|e| format!("the image could not be decoded ({e})"))?;
        let rgb = img.to_rgb8();
        let pic = self.preprocess(rgb.as_raw(), rgb.width() as usize, rgb.height() as usize);
        let n = self.n_tokens(&pic);
        Ok((self.encode(&pic), n))
    }
}

/// Full (non-causal) multi-head attention: q, k, v are [n][n_head * hd]; the result is the same shape.
fn attention(q: &[f32], k: &[f32], v: &[f32], n: usize, n_head: usize, hd: usize) -> Vec<f32> {
    let d = n_head * hd;
    let split = |m: &[f32]| -> Vec<f32> {
        let mut o = vec![0f32; n * d];
        o.par_chunks_mut(n * hd).enumerate().for_each(|(h, oh)| {
            for t in 0..n {
                oh[t * hd..(t + 1) * hd].copy_from_slice(&m[t * d + h * hd..t * d + (h + 1) * hd]);
            }
        });
        o
    };
    let (qh, kh, vh) = (split(q), split(k), split(v));
    let scale = 1.0 / (hd as f32).sqrt();
    let bq = 64usize;
    let mut heads = vec![0f32; n * d];
    heads.par_chunks_mut(n * hd).enumerate().for_each(|(h, oh)| {
        let (qh, kh, vh) = (&qh[h * n * hd..(h + 1) * n * hd], &kh[h * n * hd..(h + 1) * n * hd], &vh[h * n * hd..(h + 1) * n * hd]);
        oh.par_chunks_mut(bq * hd).enumerate().for_each(|(bi, ob)| {
            let rows = ob.len() / hd;
            let qb = &qh[bi * bq * hd..(bi * bq + rows) * hd];
            let mut s = vec![0f32; rows * n];
            unsafe {
                matrixmultiply::sgemm(rows, hd, n, scale, qb.as_ptr(), hd as isize, 1, kh.as_ptr(), 1, hd as isize, 0.0, s.as_mut_ptr(), n as isize, 1);
            }
            for r in s.chunks_mut(n) {
                let mx = r.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0f64;
                for v in r.iter_mut() {
                    *v = (*v - mx).exp();
                    sum += *v as f64;
                }
                let inv = (1.0 / sum) as f32;
                r.iter_mut().for_each(|v| *v *= inv);
            }
            unsafe {
                matrixmultiply::sgemm(rows, n, hd, 1.0, s.as_ptr(), n as isize, 1, vh.as_ptr(), hd as isize, 1, 0.0, ob.as_mut_ptr(), hd as isize, 1);
            }
        });
    });
    let mut out = vec![0f32; n * d];
    out.par_chunks_mut(d).enumerate().for_each(|(t, row)| {
        for h in 0..n_head {
            row[h * hd..(h + 1) * hd].copy_from_slice(&heads[(h * n + t) * hd..(h * n + t + 1) * hd]);
        }
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny() -> Encoder {
        let lin = |i: usize, o: usize| Linear { w: vec![0.0; i * o], b: None, n_in: i, n_out: o, f16: false };
        Encoder {
            patch: 20, merge: 2, n_embd: 4, n_head: 1, out_dim: 4, eps: 1e-5, mean: [0.5; 3], std: [0.5; 3], max_tokens: 4096,
            patch_embd: lin(1200, 4), pos: vec![0.0; 4], side: 1, cls: vec![0.0; 4], pre_ln: (vec![1.0; 4], vec![0.0; 4]),
            blocks: vec![], mm1: lin(16, 4), mm2: lin(4, 4), f32_act: true,
        }
    }

    /// Padding to multiples of 40 on the right and bottom with raw -1, everything normalised as (x - 127.5) / 127.5,
    /// and one token per 40 x 40 square.
    #[test]
    fn preprocess_pads_normalises_and_counts_tokens() {
        let e = tiny();
        let (w, h) = (333usize, 217usize);
        let rgb: Vec<u8> = (0..w * h * 3).map(|i| (i * 7 % 256) as u8).collect();
        let pic = e.preprocess(&rgb, w, h);
        assert_eq!((pic.width, pic.height), (360, 240));
        assert_eq!(e.n_tokens(&pic), 9 * 6);
        for (y, x, c) in [(0usize, 0usize, 0usize), (5, 100, 1), (216, 332, 2)] {
            let want = (rgb[(y * w + x) * 3 + c] as f32 - 127.5) / 127.5;
            assert_eq!(pic.data[(y * 360 + x) * 3 + c], want);
        }
        let pad = (-1.0f32 - 127.5) / 127.5;
        assert_eq!(pic.data[(0 * 360 + 333) * 3], pad);
        assert_eq!(pic.data[(217 * 360 + 0) * 3 + 2], pad);
        assert_eq!(pic.data[(239 * 360 + 359) * 3 + 1], pad);
        // an aligned picture keeps its size: 320 x 192 -> 8 x 5 squares (192 pads to 200)
        let pic = e.preprocess(&vec![255u8; 320 * 192 * 3], 320, 192);
        assert_eq!((pic.width, pic.height, e.n_tokens(&pic)), (320, 200, 40));
        assert_eq!(pic.data[0], 1.0);
    }

    /// Over the token limit the picture shrinks to fit, keeping its ratio.
    #[test]
    fn preprocess_shrinks_a_picture_over_the_limit() {
        let mut e = tiny();
        e.max_tokens = 16;
        let pic = e.preprocess(&vec![0u8; 400 * 400 * 3], 400, 400);
        assert_eq!((pic.width, pic.height), (160, 160));
        assert!(e.n_tokens(&pic) <= 16);
    }

    /// At the grid's own size the bilinear resize gives the positions back unchanged.
    #[test]
    fn bilinear_resize_at_the_native_size_is_the_identity() {
        let (d, side) = (3usize, 32usize);
        let pos: Vec<f32> = (0..side * side * d).map(|i| (i as f32 * 0.37).sin()).collect();
        assert_eq!(resize_bilinear(&pos, d, side, side, side), pos);
        // and a 2x upscale of a 2x2 grid: half-pixel centres, clamped edges
        let g = [0.0f32, 1.0, 2.0, 3.0];
        let r = resize_bilinear(&g, 1, 2, 4, 4);
        assert_eq!(&r[..4], &[0.0, 0.25, 0.75, 1.0]);
        assert_eq!(&r[12..], &[2.0, 2.25, 2.75, 3.0]);
    }

    /// A 2x2 merge takes, for each merged square in row-major order, its top-left, top-right, bottom-left and
    /// bottom-right patches.
    #[test]
    fn merge_order_is_row_major_with_dy_outer() {
        // a 4 x 2 patch grid, one value per patch: its index
        let x: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let m = merge_patches(&x, 1, 4, 2, 2);
        assert_eq!(m, vec![0.0, 1.0, 4.0, 5.0, 2.0, 3.0, 6.0, 7.0]);
        // two values per patch stay together
        let x: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let m = merge_patches(&x, 2, 4, 2, 2);
        assert_eq!(&m[..8], &[0.0, 1.0, 2.0, 3.0, 8.0, 9.0, 10.0, 11.0]);
    }

    /// The products with matrixmultiply equal plain dot products.
    #[test]
    fn linear_matches_dot_products() {
        let (i, o, n) = (37usize, 70usize, 5usize);
        let w: Vec<f32> = (0..i * o).map(|k| ((k * 13 % 17) as f32 - 8.0) / 8.0).collect();
        let b: Vec<f32> = (0..o).map(|k| k as f32 * 0.1).collect();
        let x: Vec<f32> = (0..i * n).map(|k| ((k * 5 % 11) as f32 - 5.0) / 4.0).collect();
        let lin = Linear { w: w.clone(), b: Some(b.clone()), n_in: i, n_out: o, f16: false };
        let y = lin.forward(&x, n, true);
        for t in 0..n {
            for r in 0..o {
                let want: f32 = (0..i).map(|c| w[r * i + c] * x[t * i + c]).sum::<f32>() + b[r];
                assert!((y[t * o + r] - want).abs() < 1e-4);
            }
        }
    }
}
