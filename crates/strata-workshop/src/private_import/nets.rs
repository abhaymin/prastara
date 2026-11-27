//! The check models' networks, on candle (the Workshop's own Rust inference), CPU only, f32: Hugging Face
//! `ViTForImageClassification` for images, and `BertForSequenceClassification`, `RobertaForSequenceClassification`
//! (XLM-RoBERTa too) and `DistilBertForSequenceClassification` for text. Weights come from the model folder's
//! safetensors (any float type, read as f32); nothing is downloaded. Each gives the label scores (softmax), or for a
//! one-output (regression) model the raw value.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context, Result};
use candle_core::{DType, Device, Tensor, D};
use serde_json::Value;

/// The tensors of a model folder (`model.safetensors`, or the shards `model-*.safetensors`).
pub struct Weights(HashMap<String, Tensor>);

impl Weights {
    pub fn load(dir: &Path) -> Result<Weights> {
        let files = safetensors_files(dir);
        if files.is_empty() {
            bail!("{}: no model.safetensors", dir.display());
        }
        let mut all = HashMap::new();
        for f in files {
            let t = candle_core::safetensors::load(&f, &Device::Cpu).with_context(|| format!("{}", f.display()))?;
            all.extend(t);
        }
        Ok(Weights(all))
    }

    pub fn from_map(m: HashMap<String, Tensor>) -> Weights {
        Weights(m)
    }

    fn get(&self, name: &str) -> Result<Tensor> {
        let t = self.0.get(name).ok_or_else(|| anyhow!("the weights have no {name}"))?;
        Ok(t.to_dtype(DType::F32)?)
    }

    /// The first of `names` that exists (checkpoints differ: `LayerNorm.weight` or `LayerNorm.gamma`).
    fn any(&self, names: &[String]) -> Result<Tensor> {
        for n in names {
            if self.0.contains_key(n) {
                return self.get(n);
            }
        }
        bail!("the weights have none of {}", names.join(", "))
    }

    fn has(&self, name: &str) -> bool {
        self.0.contains_key(name)
    }
}

/// `model.safetensors`, else the shards `model-*-of-*.safetensors`, sorted.
pub fn safetensors_files(dir: &Path) -> Vec<PathBuf> {
    let one = dir.join("model.safetensors");
    if one.is_file() {
        return vec![one];
    }
    let mut v: Vec<PathBuf> = std::fs::read_dir(dir).into_iter().flatten().flatten().map(|e| e.path())
        .filter(|p| p.extension().and_then(|x| x.to_str()) == Some("safetensors")
                && p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.starts_with("model-")))
        .collect();
    v.sort();
    v
}

/// The labels in output order, from `id2label` (else LABEL_0, LABEL_1, ...).
pub fn labels(cfg: &Value, n: usize) -> Vec<String> {
    (0..n).map(|i| cfg["id2label"][i.to_string()].as_str().map(str::to_string).unwrap_or_else(|| format!("LABEL_{i}"))).collect()
}

fn usize_of(cfg: &Value, keys: &[&str], default: Option<usize>) -> Result<usize> {
    for k in keys {
        if let Some(n) = cfg[*k].as_u64() {
            return Ok(n as usize);
        }
    }
    default.ok_or_else(|| anyhow!("config.json has no {}", keys.join(" or ")))
}

fn linear(x: &Tensor, w: &Tensor, b: Option<&Tensor>) -> Result<Tensor> {
    // x [.., in] · wᵀ [in, out] + b
    let y = x.broadcast_matmul(&w.t()?)?;
    Ok(match b { Some(b) => y.broadcast_add(b)?, None => y })
}

fn layer_norm(x: &Tensor, g: &Tensor, b: &Tensor, eps: f64) -> Result<Tensor> {
    Ok(candle_nn::ops::layer_norm(x, g, b, eps as f32)?)
}

/// Multi-head self-attention over `x` [1, n, h] (one window, no padding).
fn attention(x: &Tensor, q: (&Tensor, &Tensor), k: (&Tensor, &Tensor), v: (&Tensor, &Tensor), heads: usize) -> Result<Tensor> {
    let (b, n, h) = x.dims3()?;
    if heads == 0 || h % heads != 0 {
        bail!("{h} is not a whole number of {heads} heads");
    }
    let d = h / heads;
    let split = |t: Tensor| -> Result<Tensor> { Ok(t.reshape((b, n, heads, d))?.transpose(1, 2)?.contiguous()?) };
    let qh = split(linear(x, q.0, Some(q.1))?)?;
    let kh = split(linear(x, k.0, Some(k.1))?)?;
    let vh = split(linear(x, v.0, Some(v.1))?)?;
    let s = (qh.matmul(&kh.t()?)? / (d as f64).sqrt())?;
    let p = candle_nn::ops::softmax_last_dim(&s)?;
    Ok(p.matmul(&vh)?.transpose(1, 2)?.contiguous()?.reshape((b, n, h))?)
}

fn act(x: &Tensor, name: &str) -> Result<Tensor> {
    Ok(match name {
        "relu" => x.relu()?,
        "gelu_new" | "gelu_pytorch_tanh" | "gelu_fast" => x.gelu()?,
        _ => x.gelu_erf()?,
    })
}

/// The scores of the last layer's output: softmax over the labels, or the value itself for one output.
fn scores(logits: &Tensor, labels: &[String]) -> Result<Vec<(String, f32)>> {
    let l = logits.flatten_all()?;
    let v: Vec<f32> = if labels.len() == 1 { l.to_vec1()? } else { candle_nn::ops::softmax(&l, D::Minus1)?.to_vec1()? };
    if v.len() != labels.len() || v.iter().any(|x| !x.is_finite()) {
        bail!("the model gave {} scores for {} labels, or a score that is not a number", v.len(), labels.len());
    }
    Ok(labels.iter().cloned().zip(v).collect())
}

// ------------------------------------------------------------------------------------------------- ViT

/// `ViTForImageClassification`: patches of a resized, normalised RGB image, a [CLS] token, pre-norm blocks.
pub struct Vit {
    w: Weights,
    hidden: usize,
    layers: usize,
    heads: usize,
    patch: usize,
    pub image_size: usize,
    eps: f64,
    act: String,
    pub mean: [f32; 3],
    pub std: [f32; 3],
    pub rescale: f32,
    pub labels: Vec<String>,
}

impl Vit {
    pub fn new(cfg: &Value, pre: &Value, w: Weights) -> Result<Vit> {
        let hidden = usize_of(cfg, &["hidden_size"], None)?;
        let image_size = match &pre["size"] {
            Value::Number(n) => n.as_u64().map(|n| n as usize),
            Value::Object(o) => o.get("height").or(o.get("shortest_edge")).and_then(Value::as_u64).map(|n| n as usize),
            _ => None,
        }.or(cfg["image_size"].as_u64().map(|n| n as usize)).unwrap_or(224);
        let trio = |k: &str, d: f32| -> [f32; 3] {
            match pre[k].as_array() {
                Some(a) if a.len() == 3 => [0, 1, 2].map(|i| a[i].as_f64().unwrap_or(d as f64) as f32),
                _ => [d; 3],
            }
        };
        let n_labels = w.get("classifier.weight")?.dim(0)?;
        let v = Vit {
            hidden, layers: usize_of(cfg, &["num_hidden_layers"], None)?, heads: usize_of(cfg, &["num_attention_heads"], None)?,
            patch: usize_of(cfg, &["patch_size"], Some(16))?, image_size, eps: cfg["layer_norm_eps"].as_f64().unwrap_or(1e-12),
            act: cfg["hidden_act"].as_str().unwrap_or("gelu").to_string(),
            mean: trio("image_mean", 0.5), std: trio("image_std", 0.5),
            rescale: if pre["do_rescale"] == false { 1.0 } else { pre["rescale_factor"].as_f64().unwrap_or(1.0 / 255.0) as f32 },
            labels: labels(cfg, n_labels), w,
        };
        if v.patch == 0 || v.image_size % v.patch != 0 {
            bail!("an image of {} is not a whole number of {}-pixel patches", v.image_size, v.patch);
        }
        Ok(v)
    }

    /// The scores of an RGB image (any size; resized to the model's).
    pub fn classify(&self, img: &image::RgbImage) -> Result<Vec<(String, f32)>> {
        let s = self.image_size as u32;
        let r = image::imageops::resize(img, s, s, image::imageops::FilterType::Triangle);
        let (p, side) = (self.patch, self.image_size / self.patch);
        let n = side * side;
        // patches as rows of (channel, y, x), the order of the projection's flattened kernel
        let mut rows = vec![0f32; n * 3 * p * p];
        for py in 0..side {
            for px in 0..side {
                let base = (py * side + px) * 3 * p * p;
                for c in 0..3 {
                    for y in 0..p {
                        for x in 0..p {
                            let v = r.get_pixel((px * p + x) as u32, (py * p + y) as u32)[c] as f32 * self.rescale;
                            rows[base + c * p * p + y * p + x] = (v - self.mean[c]) / self.std[c];
                        }
                    }
                }
            }
        }
        let w = &self.w;
        let pre = |k: &str| format!("vit.{k}");
        let patches = Tensor::from_vec(rows, (1, n, 3 * p * p), &Device::Cpu)?;
        let proj = w.get(&pre("embeddings.patch_embeddings.projection.weight"))?.reshape((self.hidden, 3 * p * p))?;
        let mut x = linear(&patches, &proj, Some(&w.get(&pre("embeddings.patch_embeddings.projection.bias"))?))?;
        let cls = w.get(&pre("embeddings.cls_token"))?.reshape((1, 1, self.hidden))?;
        x = Tensor::cat(&[&cls, &x], 1)?;
        let pos = w.get(&pre("embeddings.position_embeddings"))?;
        if pos.dim(1)? != n + 1 {
            bail!("the position embeddings are for {} patches, the image gives {n}", pos.dim(1)? - 1);
        }
        x = x.broadcast_add(&pos)?;
        for i in 0..self.layers {
            let l = |k: &str| pre(&format!("encoder.layer.{i}.{k}"));
            let h = layer_norm(&x, &w.get(&l("layernorm_before.weight"))?, &w.get(&l("layernorm_before.bias"))?, self.eps)?;
            let g = |k: &str| -> Result<(Tensor, Tensor)> { Ok((w.get(&l(&format!("{k}.weight")))?, w.get(&l(&format!("{k}.bias")))?)) };
            let (q, k, v, o) = (g("attention.attention.query")?, g("attention.attention.key")?, g("attention.attention.value")?, g("attention.output.dense")?);
            let a = attention(&h, (&q.0, &q.1), (&k.0, &k.1), (&v.0, &v.1), self.heads)?;
            x = (x + linear(&a, &o.0, Some(&o.1))?)?;
            let h = layer_norm(&x, &w.get(&l("layernorm_after.weight"))?, &w.get(&l("layernorm_after.bias"))?, self.eps)?;
            let (i1, o2) = (g("intermediate.dense")?, g("output.dense")?);
            let f = linear(&act(&linear(&h, &i1.0, Some(&i1.1))?, &self.act)?, &o2.0, Some(&o2.1))?;
            x = (x + f)?;
        }
        let x = layer_norm(&x, &w.get(&pre("layernorm.weight"))?, &w.get(&pre("layernorm.bias"))?, self.eps)?;
        let cls = x.narrow(1, 0, 1)?;
        let logits = linear(&cls, &w.get("classifier.weight")?, Some(&w.get("classifier.bias")?))?;
        scores(&logits, &self.labels)
    }
}

// ------------------------------------------------------------------------------------------------- BERT family

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextArch {
    Bert,
    Roberta,
    DistilBert,
}

impl TextArch {
    pub fn of(architecture: &str) -> Option<TextArch> {
        match architecture {
            "BertForSequenceClassification" => Some(TextArch::Bert),
            "RobertaForSequenceClassification" | "XLMRobertaForSequenceClassification" => Some(TextArch::Roberta),
            "DistilBertForSequenceClassification" => Some(TextArch::DistilBert),
            _ => None,
        }
    }
}

/// A BERT-family sequence classifier: post-norm blocks, the first token's state through the head.
pub struct TextNet {
    w: Weights,
    arch: TextArch,
    layers: usize,
    heads: usize,
    eps: f64,
    act: String,
    pad: u32,
    pub max_len: usize,
    pub labels: Vec<String>,
}

impl TextNet {
    pub fn new(cfg: &Value, arch: TextArch, w: Weights) -> Result<TextNet> {
        let n_labels = match arch {
            TextArch::Roberta => w.get("classifier.out_proj.weight")?.dim(0)?,
            _ => w.get("classifier.weight")?.dim(0)?,
        };
        let pad = cfg["pad_token_id"].as_u64().unwrap_or(if arch == TextArch::Roberta { 1 } else { 0 }) as u32;
        let positions = usize_of(cfg, &["max_position_embeddings"], Some(512))?;
        // RoBERTa's positions start after the padding id
        let max_len = if arch == TextArch::Roberta { positions.saturating_sub(pad as usize + 1) } else { positions };
        Ok(TextNet {
            layers: usize_of(cfg, &["num_hidden_layers", "n_layers"], None)?, heads: usize_of(cfg, &["num_attention_heads", "n_heads"], None)?,
            eps: cfg["layer_norm_eps"].as_f64().unwrap_or(1e-12),
            act: cfg["hidden_act"].as_str().or(cfg["activation"].as_str()).unwrap_or("gelu").to_string(),
            pad, max_len: max_len.max(8), labels: labels(cfg, n_labels), arch, w,
        })
    }

    /// The scores of one window of token ids (with the special tokens; at most `max_len`).
    pub fn classify(&self, ids: &[u32]) -> Result<Vec<(String, f32)>> {
        if ids.is_empty() || ids.len() > self.max_len {
            bail!("{} tokens: a window is 1 to {}", ids.len(), self.max_len);
        }
        let w = &self.w;
        let n = ids.len();
        let dev = Device::Cpu;
        let ids_t = Tensor::from_vec(ids.to_vec(), (1, n), &dev)?;
        let base = match self.arch { TextArch::Bert => "bert", TextArch::Roberta => "roberta", TextArch::DistilBert => "distilbert" };
        let p = |k: &str| format!("{base}.{k}");
        let ln = |pfx: &str| -> Result<(Tensor, Tensor)> {
            Ok((w.any(&[format!("{pfx}.weight"), format!("{pfx}.gamma")])?, w.any(&[format!("{pfx}.bias"), format!("{pfx}.beta")])?))
        };
        let word = w.get(&p("embeddings.word_embeddings.weight"))?;
        let pos_ids: Vec<u32> = match self.arch {
            TextArch::Roberta => (0..n as u32).map(|i| i + self.pad + 1).collect(),
            _ => (0..n as u32).collect(),
        };
        let pos = w.get(&p("embeddings.position_embeddings.weight"))?.index_select(&Tensor::from_vec(pos_ids, n, &dev)?, 0)?;
        let mut x = word.index_select(&ids_t.flatten_all()?, 0)?.add(&pos)?;
        if self.arch != TextArch::DistilBert && w.has(&p("embeddings.token_type_embeddings.weight")) {
            let tt = w.get(&p("embeddings.token_type_embeddings.weight"))?.narrow(0, 0, 1)?;
            x = x.broadcast_add(&tt)?;
        }
        let (g, b) = ln(&p("embeddings.LayerNorm"))?;
        let mut x = layer_norm(&x.unsqueeze(0)?, &g, &b, self.eps)?;
        for i in 0..self.layers {
            let lw = |k: &str| -> Result<(Tensor, Tensor)> {
                let k = match self.arch {
                    TextArch::DistilBert => p(&format!("transformer.layer.{i}.{k}")),
                    _ => p(&format!("encoder.layer.{i}.{k}")),
                };
                Ok((w.get(&format!("{k}.weight"))?, w.get(&format!("{k}.bias"))?))
            };
            let names = match self.arch {
                TextArch::DistilBert => ["attention.q_lin", "attention.k_lin", "attention.v_lin", "attention.out_lin", "sa_layer_norm", "ffn.lin1", "ffn.lin2", "output_layer_norm"],
                _ => ["attention.self.query", "attention.self.key", "attention.self.value", "attention.output.dense", "attention.output.LayerNorm",
                      "intermediate.dense", "output.dense", "output.LayerNorm"],
            };
            let (q, k, v, o) = (lw(names[0])?, lw(names[1])?, lw(names[2])?, lw(names[3])?);
            let a = attention(&x, (&q.0, &q.1), (&k.0, &k.1), (&v.0, &v.1), self.heads)?;
            let n1 = match self.arch {
                TextArch::DistilBert => lw(names[4])?,
                _ => {
                    let k = p(&format!("encoder.layer.{i}.{}", names[4]));
                    ln(&k)?
                }
            };
            x = layer_norm(&(x + linear(&a, &o.0, Some(&o.1))?)?, &n1.0, &n1.1, self.eps)?;
            let (f1, f2) = (lw(names[5])?, lw(names[6])?);
            let f = linear(&act(&linear(&x, &f1.0, Some(&f1.1))?, &self.act)?, &f2.0, Some(&f2.1))?;
            let n2 = match self.arch {
                TextArch::DistilBert => lw(names[7])?,
                _ => ln(&p(&format!("encoder.layer.{i}.{}", names[7])))?,
            };
            x = layer_norm(&(x + f)?, &n2.0, &n2.1, self.eps)?;
        }
        let first = x.narrow(1, 0, 1)?;
        let logits = match self.arch {
            TextArch::Bert => {
                let pooled = linear(&first, &w.get(&p("pooler.dense.weight"))?, Some(&w.get(&p("pooler.dense.bias"))?))?.tanh()?;
                linear(&pooled, &w.get("classifier.weight")?, Some(&w.get("classifier.bias")?))?
            }
            TextArch::Roberta => {
                let h = linear(&first, &w.get("classifier.dense.weight")?, Some(&w.get("classifier.dense.bias")?))?.tanh()?;
                linear(&h, &w.get("classifier.out_proj.weight")?, Some(&w.get("classifier.out_proj.bias")?))?
            }
            TextArch::DistilBert => {
                let h = linear(&first, &w.get("pre_classifier.weight")?, Some(&w.get("pre_classifier.bias")?))?.relu()?;
                linear(&h, &w.get("classifier.weight")?, Some(&w.get("classifier.bias")?))?
            }
        };
        scores(&logits, &self.labels)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use serde_json::json;

    /// A tensor of small deterministic numbers (no randomness: the same every run).
    pub fn t(shape: &[usize], seed: u32) -> Tensor {
        let n: usize = shape.iter().product();
        let v: Vec<f32> = (0..n).map(|i| (((i as u32).wrapping_mul(2654435761).wrapping_add(seed * 97)) % 1000) as f32 / 1000.0 - 0.5).collect();
        Tensor::from_vec(v, shape, &Device::Cpu).unwrap().to_dtype(DType::F32).unwrap()
    }

    /// A tiny random ViT classifier's tensors (hidden 8, one block, 4 px patches of a 8 px image).
    pub fn tiny_vit(labels: usize) -> HashMap<String, Tensor> {
        let (h, f, p) = (8, 16, 4);
        let mut m = HashMap::new();
        let mut put = |k: &str, s: &[usize]| { let seed = m.len() as u32; m.insert(k.to_string(), t(s, seed)); };
        put("vit.embeddings.patch_embeddings.projection.weight", &[h, 3, p, p]);
        put("vit.embeddings.patch_embeddings.projection.bias", &[h]);
        put("vit.embeddings.cls_token", &[1, 1, h]);
        put("vit.embeddings.position_embeddings", &[1, 5, h]);
        for k in ["layernorm_before", "layernorm_after"] {
            put(&format!("vit.encoder.layer.0.{k}.weight"), &[h]);
            put(&format!("vit.encoder.layer.0.{k}.bias"), &[h]);
        }
        for k in ["attention.attention.query", "attention.attention.key", "attention.attention.value", "attention.output.dense"] {
            put(&format!("vit.encoder.layer.0.{k}.weight"), &[h, h]);
            put(&format!("vit.encoder.layer.0.{k}.bias"), &[h]);
        }
        put("vit.encoder.layer.0.intermediate.dense.weight", &[f, h]);
        put("vit.encoder.layer.0.intermediate.dense.bias", &[f]);
        put("vit.encoder.layer.0.output.dense.weight", &[h, f]);
        put("vit.encoder.layer.0.output.dense.bias", &[h]);
        put("vit.layernorm.weight", &[h]);
        put("vit.layernorm.bias", &[h]);
        put("classifier.weight", &[labels, h]);
        put("classifier.bias", &[labels]);
        m
    }

    pub fn tiny_vit_config(labels: &[&str]) -> Value {
        let id2label: serde_json::Map<String, Value> = labels.iter().enumerate().map(|(i, l)| (i.to_string(), json!(l))).collect();
        json!({"architectures": ["ViTForImageClassification"], "hidden_size": 8, "num_hidden_layers": 1, "num_attention_heads": 2,
               "intermediate_size": 16, "patch_size": 4, "image_size": 8, "id2label": id2label})
    }

    /// A tiny BERT classifier's tensors (vocabulary 12, hidden 8, one block).
    pub fn tiny_bert(labels: usize) -> HashMap<String, Tensor> {
        let (h, f, v) = (8, 16, 12);
        let mut m = HashMap::new();
        let mut put = |k: &str, s: &[usize]| { let seed = m.len() as u32; m.insert(k.to_string(), t(s, seed)); };
        put("bert.embeddings.word_embeddings.weight", &[v, h]);
        put("bert.embeddings.position_embeddings.weight", &[16, h]);
        put("bert.embeddings.token_type_embeddings.weight", &[2, h]);
        put("bert.embeddings.LayerNorm.weight", &[h]);
        put("bert.embeddings.LayerNorm.bias", &[h]);
        for k in ["attention.self.query", "attention.self.key", "attention.self.value", "attention.output.dense"] {
            put(&format!("bert.encoder.layer.0.{k}.weight"), &[h, h]);
            put(&format!("bert.encoder.layer.0.{k}.bias"), &[h]);
        }
        for k in ["attention.output.LayerNorm", "output.LayerNorm"] {
            put(&format!("bert.encoder.layer.0.{k}.weight"), &[h]);
            put(&format!("bert.encoder.layer.0.{k}.bias"), &[h]);
        }
        put("bert.encoder.layer.0.intermediate.dense.weight", &[f, h]);
        put("bert.encoder.layer.0.intermediate.dense.bias", &[f]);
        put("bert.encoder.layer.0.output.dense.weight", &[h, f]);
        put("bert.encoder.layer.0.output.dense.bias", &[h]);
        put("bert.pooler.dense.weight", &[h, h]);
        put("bert.pooler.dense.bias", &[h]);
        put("classifier.weight", &[labels, h]);
        put("classifier.bias", &[labels]);
        m
    }

    pub fn tiny_bert_config(labels: &[&str]) -> Value {
        let id2label: serde_json::Map<String, Value> = labels.iter().enumerate().map(|(i, l)| (i.to_string(), json!(l))).collect();
        json!({"architectures": ["BertForSequenceClassification"], "hidden_size": 8, "num_hidden_layers": 1, "num_attention_heads": 2,
               "intermediate_size": 16, "max_position_embeddings": 16, "vocab_size": 12, "id2label": id2label})
    }

    #[test]
    fn a_tiny_vit_scores_an_image_and_the_scores_sum_to_one() {
        let v = Vit::new(&tiny_vit_config(&["safe", "flag"]), &json!({}), Weights::from_map(tiny_vit(2))).unwrap();
        let img = image::RgbImage::from_pixel(20, 12, image::Rgb([200, 30, 30]));
        let s = v.classify(&img).unwrap();
        assert_eq!(s.iter().map(|x| x.0.as_str()).collect::<Vec<_>>(), ["safe", "flag"]);
        assert!((s.iter().map(|x| x.1).sum::<f32>() - 1.0).abs() < 1e-4, "{s:?}");
        // the same image gives the same scores; another colour gives others
        assert_eq!(v.classify(&img).unwrap(), s);
        assert_ne!(v.classify(&image::RgbImage::from_pixel(8, 8, image::Rgb([0, 0, 255]))).unwrap(), s);
    }

    #[test]
    fn a_tiny_bert_scores_a_window_and_refuses_one_too_long() {
        let n = TextNet::new(&tiny_bert_config(&["ok", "minor"]), TextArch::Bert, Weights::from_map(tiny_bert(2))).unwrap();
        let s = n.classify(&[1, 5, 6, 2]).unwrap();
        assert!((s.iter().map(|x| x.1).sum::<f32>() - 1.0).abs() < 1e-4, "{s:?}");
        assert!(n.classify(&[1; 17]).is_err());
        assert!(n.classify(&[]).is_err());
    }

    #[test]
    fn missing_weights_are_an_error_not_a_score() {
        let mut m = tiny_vit(2);
        m.remove("vit.layernorm.weight");
        let v = Vit::new(&tiny_vit_config(&["a", "b"]), &json!({}), Weights::from_map(m)).unwrap();
        assert!(v.classify(&image::RgbImage::new(8, 8)).is_err());
    }
}
