//! Tiny synthetic Llama-shaped safetensors fixture + a `tokenizer.json`, built deterministically from a fixed seed — no
//! downloads and no Python. Every workshop step's test reuses this to get a small, well-shaped model on disk, so the
//! round-trips exercise real code paths. The golden references that do need Python or llama.cpp (the
//! `convert_hf_to_gguf` output of this tiny model, the `llama-quantize` bytes) are made once by the reviewer and
//! committed under `tests/fixtures/` with a README; tests only read them.

use std::path::{Path, PathBuf};

use candle_core::{Device, Tensor};
use serde_json::Value as Json;
use serde_json::json;
use safetensors::tensor as st;

/// A tiny random Llama (2 layers by default) laid out the way a real checkpoint's tensors are named. All weights are
/// f32 and generated from `seed` so a round-trip read compares against identical values. Sizes scale the fixture up or
/// down but never download anything.
#[derive(Debug, Clone)]
pub struct TinyLlama {
    pub seed: u64,
    /// how many layers; the plan's "2-layer" proof uses 2
    pub num_layers: usize,
    pub vocab_size: usize,
    pub hidden: usize,
    pub intermediate: usize,
    /// attention heads (hidden is a multiple of these), written to config.json and used for the rotary q/k permute
    pub num_attention_heads: usize,
    /// GQA key/value heads (defaults to full attention here)
    pub num_key_value_heads: usize,
    /// how many shard files to write (1 = one file; >=2 lets step 1 exercise shard enumeration)
    pub shard_count: usize,
}

impl Default for TinyLlama {
    fn default() -> Self {
        Self { seed: 0x5EED, num_layers: 2, vocab_size: 32, hidden: 16, intermediate: 32, num_attention_heads: 2, num_key_value_heads: 2, shard_count: 1 }
    }
}

/// One generated tensor: its name, shape and raw f32 contents (row-major).
#[derive(Debug, Clone)]
pub struct TensorData {
    pub name: String,
    pub shape: Vec<usize>,
    pub values: Vec<f32>,
}

impl TinyLlama {
    /// The canonical tensor list, in a fixed order. `write` and the round-trip test's "expected" values come from here,
    /// so a read always matches what was written regardless of on-disk ordering.
    pub fn generate(&self) -> Vec<TensorData> {
        let mut gen = SplitMix64::new(self.seed);
        let mut out = Vec::new();
        // realistic scales, so the model's outputs can move: norm gains 1.0 (as Llama initialises them), embeddings
        // about ±1, projections about ±1/sqrt(fan-in). Everything at ±0.02, norm gains included, made the tied head's
        // logits ~0.006, i.e. a uniform output no adapter could change.
        let w = 1.0 / (self.hidden as f32).sqrt();
        // head_dim drives the projection output dims: q projects to num_attention_heads*head_dim rows, k/v to
        // num_key_value_heads*head_dim (GQA); when the two are equal this is exactly `hidden` for every projection.
        let head_dim = self.hidden / self.num_attention_heads;
        let q_out = self.num_attention_heads * head_dim;
        let kv_out = self.num_key_value_heads * head_dim;
        // embeddings + output norm are shared across layers
        out.push(TensorData { name: "model.embed_tokens.weight".into(), shape: vec![self.vocab_size, self.hidden], values: mat(&mut gen, self.vocab_size, self.hidden, 1.0) });
        for layer in 0..self.num_layers {
            let p = |key: &str| format!("model.layers.{layer}.{key}");
            out.push(TensorData { name: p("input_layernorm.weight"), shape: vec![self.hidden], values: vec![1.0; self.hidden] });
            out.push(TensorData { name: p("self_attn.q_proj.weight"), shape: vec![q_out, self.hidden], values: mat(&mut gen, q_out, self.hidden, w) });
            out.push(TensorData { name: p("self_attn.k_proj.weight"), shape: vec![kv_out, self.hidden], values: mat(&mut gen, kv_out, self.hidden, w) });
            out.push(TensorData { name: p("self_attn.v_proj.weight"), shape: vec![kv_out, self.hidden], values: mat(&mut gen, kv_out, self.hidden, w) });
            out.push(TensorData { name: p("self_attn.o_proj.weight"), shape: vec![self.hidden, self.hidden], values: mat(&mut gen, self.hidden, self.hidden, w) });
            out.push(TensorData { name: p("post_attention_layernorm.weight"), shape: vec![self.hidden], values: vec![1.0; self.hidden] });
            out.push(TensorData { name: p("mlp.gate_proj.weight"), shape: vec![self.intermediate, self.hidden], values: mat(&mut gen, self.intermediate, self.hidden, w) });
            out.push(TensorData { name: p("mlp.up_proj.weight"), shape: vec![self.intermediate, self.hidden], values: mat(&mut gen, self.intermediate, self.hidden, w) });
            out.push(TensorData { name: p("mlp.down_proj.weight"), shape: vec![self.hidden, self.intermediate], values: mat(&mut gen, self.hidden, self.intermediate, 1.0 / (self.intermediate as f32).sqrt()) });
        }
        out.push(TensorData { name: "model.norm.weight".into(), shape: vec![self.hidden], values: vec![1.0; self.hidden] });
        // a distinct output head (tie_word_embeddings = false in config.json), so the converter writes an `output` tensor
        out.push(TensorData { name: "lm_head.weight".into(), shape: vec![self.vocab_size, self.hidden], values: mat(&mut gen, self.vocab_size, self.hidden, 1.0 / (self.hidden as f32).sqrt()) });
        out
    }

    /// Write the fixture into `dir`: `<count>` safetensors shards named like an HF checkpoint, a real `config.json` and
    /// `tokenizer_config.json`, plus a real `tokenizer.json`. Returns the shard files (in order). The config and
    /// tokenizer are exactly what the step-6 converter reads to build the GGUF and what llama.cpp's converter produced
    /// the reference from; they are byte-identical to `tests/fixtures/tiny-llama-hf/`.
    pub fn write(&self, dir: &Path) -> anyhow::Result<ShardFiles> {
        let tensors = self.generate();
        std::fs::create_dir_all(dir)?;
        let shards = partition(&tensors, self.shard_count);
        let mut shard_files = Vec::new();
        for (i, chunk) in shards.iter().enumerate() {
            let name = format!("model-{:05}-of-{:05}.safetensors", i + 1, shards.len());
            write_shard(dir.join(&name), chunk)?;
            shard_files.push(dir.join(name));
        }
        write_config_json(dir, &self.hf_config())?;
        write_tokenizer_config_json(dir)?;
        write_tokenizer_json(dir, self.vocab_size)?;
        Ok(ShardFiles { dir: dir.to_path_buf(), shard_files })
    }

    /// The config.json contents this fixture writes, as a serde_json value — so the converter can be tested against it.
    pub fn hf_config(&self) -> serde_json::Value {
        json!({
            "architectures": ["LlamaForCausalLM"],
            "hidden_size": self.hidden,
            "num_hidden_layers": self.num_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "intermediate_size": self.intermediate,
            "rms_norm_eps": 1e-5,
            "vocab_size": self.vocab_size,
            "max_position_embeddings": 512,
            "rope_theta": 10000.0,
            "tie_word_embeddings": false,
        })
    }
}

/// The chat fixture's vocabulary: three special tokens, then the 256 byte-level symbols.
pub const CHAT_VOCAB: usize = 259;

/// ChatML, as Qwen and many fine-tunes use it.
pub const CHATML_TEMPLATE: &str = "{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}";

/// A tiny random model that can read real text, for the trainer's tests: [`TinyLlama`] with a vocabulary of
/// [`CHAT_VOCAB`] and, in place of the `tok_N` tokenizer, a byte-level one (every byte is a token, so any text
/// tokenises) with `<|endoftext|>`, `<|im_start|>`, `<|im_end|>` as ids 0 to 2 and a ChatML chat template.
pub fn write_chat_model(dir: &Path, model: &TinyLlama) -> anyhow::Result<()> {
    anyhow::ensure!(model.vocab_size == CHAT_VOCAB, "the chat fixture needs vocab_size {CHAT_VOCAB}");
    model.write(dir)?;
    let specials = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"];
    let mut symbols: Vec<char> = tokenizers::pre_tokenizers::byte_level::ByteLevel::alphabet().into_iter().collect();
    symbols.sort();
    let mut vocab = serde_json::Map::new();
    for (i, s) in specials.iter().enumerate() {
        vocab.insert(s.to_string(), json!(i));
    }
    for (i, c) in symbols.iter().enumerate() {
        vocab.insert(c.to_string(), json!(i + specials.len()));
    }
    anyhow::ensure!(vocab.len() == CHAT_VOCAB, "byte-level alphabet of {} symbols", symbols.len());
    let mut tok = tokenizer_json(0);
    tok["added_tokens"] = Json::Array(specials.iter().enumerate().map(|(i, s)| added_token(i as u64, s, true)).collect());
    tok["model"]["vocab"] = Json::Object(vocab);
    std::fs::write(dir.join("tokenizer.json"), serde_json::to_string_pretty(&tok)?)?;
    std::fs::write(dir.join("tokenizer_config.json"), serde_json::to_string_pretty(&json!({
        "tokenizer_class": "PreTrainedTokenizerFast",
        "eos_token": "<|im_end|>",
        "pad_token": "<|endoftext|>",
        "chat_template": CHATML_TEMPLATE,
    }))?)?;
    Ok(())
}

/// Where a written fixture landed.
pub struct ShardFiles {
    pub dir: PathBuf,
    /// all shards in order (1 entry when `shard_count == 1`)
    pub shard_files: Vec<PathBuf>,
}

/// Partition `tensors` into `n` contiguous chunks (the last may hold the remainder). One safetensors file holds one
/// chunk; reading the shards in order reproduces the full tensor list.
fn partition(tensors: &[TensorData], n: usize) -> Vec<Vec<TensorData>> {
    if n <= 1 {
        return vec![tensors.to_vec()];
    }
    let per = tensors.len().div_ceil(n);
    tensors.chunks(per).map(|c| c.to_vec()).collect()
}

/// Serialize a chunk to `path` with the safetensors crate (row-major f32; dtype "F32").
fn write_shard(path: PathBuf, chunk: &[TensorData]) -> anyhow::Result<()> {
    let mut owned: Vec<Tensor> = Vec::with_capacity(chunk.len());
    for t in chunk {
        owned.push(Tensor::from_vec(t.values.clone(), t.shape.clone(), &Device::Cpu)?);
    }
    // views borrow the tensors above; they are all built before this immutable borrow, so serialize can read them.
    let views: Vec<(&str, &Tensor)> = (0..chunk.len()).map(|i| (chunk[i].name.as_str(), &owned[i])).collect();
    st::serialize_to_file(views, None, &path)?;
    Ok(())
}

/// A minimal but valid HuggingFace `tokenizer.json` (ByteLevel BPE, no merges) so later steps that read a tokenizer
/// from the fixture have something loadable. Vocab tokens are derived from the seed.
fn write_config_json(dir: &Path, cfg: &serde_json::Value) -> anyhow::Result<()> {
    std::fs::write(dir.join("config.json"), serde_json::to_string_pretty(cfg)?)?;
    Ok(())
}

/// Write the HF `tokenizer_config.json` (chat template, bos/eos, add flags) so the converter and llama.cpp agree on the
/// tokenizer's special tokens. bos is tok_0 / eos is tok_1 (plain strings, as a real `PreTrainedTokenizerFast` writes),
/// matching `tests/fixtures/tiny-llama-hf/tokenizer_config.json` byte for byte.
fn write_tokenizer_config_json(dir: &Path) -> anyhow::Result<()> {
    std::fs::write(
        dir.join("tokenizer_config.json"),
        serde_json::to_string_pretty(&tokenizer_config_json())?,
    )?;
    Ok(())
}

/// Write a real Hugging Face `tokenizer.json` (version, truncation/padding, `added_tokens` for tok_0/tok_1 as special,
/// normalizer, ByteLevel pre_tokenizer and decoder, and a full BPE model). Identical to the one committed in
/// `tests/fixtures/tiny-llama-hf/tokenizer.json`.
fn write_tokenizer_json(dir: &Path, vocab_size: usize) -> anyhow::Result<()> {
    std::fs::write(
        dir.join("tokenizer.json"),
        serde_json::to_string_pretty(&tokenizer_json(vocab_size))?,
    )?;
    Ok(())
}

/// One entry of `tokenizer.json`'s `added_tokens`: tok_0 / tok_1, special and otherwise inert. Kept as a builder so both
/// the writer and the reference-equality test construct it identically.
fn added_token(id: u64, content: &str, special: bool) -> Json {
    json!({
        "id": id,
        "content": content,
        "single_word": false,
        "lstrip": false,
        "rstrip": false,
        "normalized": false,
        "special": special,
    })
}

/// The `tokenizer.json` this fixture writes — the exact structure committed alongside the reference GGUF. Built as a
/// value so a test can assert (with `serde_json::Value` equality) that the writer never drifts back to an old form and
/// destroys the fixture on disk.
fn tokenizer_json(vocab_size: usize) -> Json {
    let mut vocab = serde_json::Map::new();
    for i in 0..vocab_size {
        vocab.insert(format!("tok_{i}"), json!(i));
    }
    let mut added_tokens = Vec::new();
    if vocab_size > 0 {
        added_tokens.push(added_token(0, "tok_0", true));
    }
    if vocab_size > 1 {
        added_tokens.push(added_token(1, "tok_1", true));
    }
    json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": added_tokens,
        "normalizer": null,
        "pre_tokenizer": { "type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": true },
        "post_processor": null,
        "decoder": { "type": "ByteLevel", "add_prefix_space": true, "trim_offsets": true, "use_regex": true },
        "model": {
            "type": "BPE",
            "dropout": null,
            "unk_token": null,
            "continuing_subword_prefix": null,
            "end_of_word_suffix": null,
            "fuse_unk": false,
            "byte_fallback": false,
            "ignore_merges": false,
            "vocab": Json::Object(vocab),
            "merges": [],
        },
    })
}

/// The `tokenizer_config.json` this fixture writes (bos/eos as plain strings). Matches the committed reference exactly.
fn tokenizer_config_json() -> Json {
    json!({
        "add_bos_token": true,
        "add_eos_token": false,
        "tokenizer_class": "PreTrainedTokenizerFast",
        "bos_token": "tok_0",
        "eos_token": "tok_1",
        "chat_template": "{% for message in messages %}{% if loop.first and message['role'] != 'system' %}{{ system_prompt }}{% endif %}{% if message['role'] == 'system' %}{{ message['content'] }}{% else %}{{ message['content'] }}{% endif %}{% endfor %}",
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// The committed fixture tokenizer files are the reference; re-derive the expected value in the test and compare
    /// with `serde_json::Value` equality so a drift in [`tokenizer_json`] / [`tokenizer_config_json`] is caught here
    /// (and, crucially, before [`save_tiny_checkpoint`](crate::tests::convert_reference::save_tiny_checkpoint) overwrites
    /// the folder and hides it).
    #[test]
    fn tokenizer_files_match_reference() {
        let fx = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny-llama-hf");
        let got_tok: Json = serde_json::from_str(
            &std::fs::read_to_string(fx.join("tokenizer.json")).expect("committed tokenizer.json parses"),
        )
        .expect("committed tokenizer.json parses");
        let got_cfg: Json = serde_json::from_str(
            &std::fs::read_to_string(fx.join("tokenizer_config.json")).expect("committed tokenizer_config.json parses"),
        )
        .expect("committed tokenizer_config parses");
        assert_eq!(got_tok, tokenizer_json(32), "committed tokenizer.json differs from what TinyLlama writes");
        assert_eq!(got_cfg, tokenizer_config_json(), "committed tokenizer_config.json differs from what TinyLlama writes");
    }

    /// A fresh temp dir (under `TMPDIR`, so the harness's scratch is used), removed at the end of each test.
    fn fixture_dir(tag: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
        let d = std::env::temp_dir().join(format!("am2-workshop-inkling-{tag}-{nanos}"));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn tiny_inkling_round_trips_through_strata_gguf_and_is_deterministic() {
        use strata_gguf::Gguf;

        let d = fixture_dir("inkling-roundtrip");
        let model = TinyInkling::default();
        let gg = d.join("tiny.inkling.gguf");
        model.write_gguf(&gg).expect("write the inkling fixture");
        let g = Gguf::open(&gg).expect("read it back with strata_gguf");

        assert_eq!(g.architecture(), Some("inkling2_loop"), "architecture key");

        // every tensor, name and ggml ne (dims) in order — the fixture's own spec, so a drift in names/shapes fails here
        let expect: Vec<(String, Vec<u64>)> = model
            .tensor_list(false)
            .iter()
            .map(|(n, ne, _)| (n.clone(), ne.iter().map(|&x| x as u64).collect()))
            .collect();
        let got: Vec<(String, Vec<u64>)> = g.tensors.iter().map(|t| (t.name.clone(), t.dims.clone())).collect();
        assert_eq!(got, expect, "read-back tensors must match the fixture's names and ggml shapes");
        assert_eq!(expect.len(), 170, "default writes make_fixture.py's 170 tensors");

        // metadata keys with the values the loop model needs: start<end over layers 2-3, three passes, block 2
        assert_eq!(g.u64("inkling2_loop.block_count"), Some(6));
        assert_eq!(g.arch_u64("embedding_length"), Some(128));
        assert_eq!(g.arch_u64("feed_forward_length"), Some(64));
        assert_eq!(g.arch_u64("expert_count"), Some(8));
        assert_eq!(g.arch_u64("expert_used_count"), Some(2));
        assert_eq!(g.arch_u64("loop.start"), Some(2));
        assert_eq!(g.arch_u64("loop.end"), Some(4));
        assert_eq!(g.arch_u64("loop.count"), Some(3));
        assert_eq!(g.arch_u64("attn_res.block_size"), Some(2));

        // arrays stay typed: head counts per layer, and the full-attention pattern as BOOLs (not ints)
        let hc = g.metadata.get("inkling2_loop.attention.head_count").and_then(|v| v.as_array()).unwrap();
        assert_eq!(hc.len(), 6);
        assert!(hc.iter().all(|h| h.as_u64() == Some(4)), "head_count is 4 for every layer");
        let hkv = g.metadata.get("inkling2_loop.attention.head_count_kv").and_then(|v| v.as_array()).unwrap();
        assert!(hkv.iter().all(|h| h.as_u64() == Some(2)), "kv heads is 2 for every layer");
        let swa = g.metadata.get("inkling2_loop.attention.sliding_window_pattern").cloned().unwrap();
        assert_eq!(swa, serde_json::json!([true, true, true, false, true, false]), "sliding-window pattern stays BOOL");

        // the small tokenizer: one gpt2 BPE file, vocab padded above the token count, specials at ids in the file
        assert_eq!(g.str("tokenizer.ggml.model"), Some("gpt2"));
        assert_eq!(g.str("tokenizer.ggml.pre"), Some("inkling2"));
        let toks = g.metadata.get("tokenizer.ggml.tokens").and_then(|v| v.as_array()).unwrap();
        assert_eq!(toks.len(), 258); // the token count: specials + byte symbols
        assert_eq!(g.u64("inkling2_loop.vocab_size_unpadded"), Some(258));
        assert_eq!(g.u64("inkling2_loop.vocab_size"), Some(320), "padded above the token count");
        // the specials I2 parses live at ids written into the file, so a tool can point at them
        assert_eq!(g.u64("tokenizer.ggml.eos_token_id"), Some(1));
        assert_eq!(g.u64("tokenizer.ggml.eot_token_id"), Some(1));
        assert_eq!(g.u64("tokenizer.ggml.padding_token_id"), Some(0));

        // the two vocabulary matrices are Q8_0 (type 8) with shape [E, padded]; a norm stays F32 (type 0)
        for nm in ["token_embd.weight", "output.weight"] {
            let t = g.tensors.iter().find(|t| t.name == nm).expect("tensor present");
            assert_eq!(t.ggml_type, 8, "{nm} is Q8_0");
            assert_eq!(t.dims, vec![128u64, 320], "{nm} shape [embedding, padded vocab]");
            assert!(t.expected_bytes().is_some(), "its byte size decodes from the encoding");
        }
        let norm = g.tensors.iter().find(|t| t.name == "blk.0.attn_norm.weight").unwrap();
        assert_eq!(norm.ggml_type, 0, "a norm is F32");

        std::fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn same_seed_writes_identical_bytes() {
        let d = fixture_dir("inkling-determinism");
        // default and wide are different models; each must be byte-identical to a fresh instance of itself with the same seed
        let (a, aname) = (TinyInkling::default(), "default");
        let (w, wname) = (TinyInkling::wide(), "wide");
        let ga = d.join("a.gguf"); let gw = d.join("b.gguf");
        a.write_gguf(&ga).unwrap();
        TinyInkling::default().write_gguf(&d.join("a2.gguf")).unwrap();
        w.write_gguf(&gw).unwrap();
        TinyInkling::wide().write_gguf(&d.join("b2.gguf")).unwrap();
        assert_eq!(std::fs::read(&ga).unwrap(), std::fs::read(d.join("a2.gguf")).unwrap(), "default: same seed, same bytes");
        assert_eq!(std::fs::read(&gw).unwrap(), std::fs::read(d.join("b2.gguf")).unwrap(), "wide: same seed, same bytes");
        assert_ne!(std::fs::read(&ga).unwrap(), std::fs::read(&gw).unwrap(), "different presets differ");

        // and a different seed moves the weights, so the determinism is real rather than a constant file
        let d2 = fixture_dir("inkling-different-seed");
        let p1 = d2.join("s1.gguf"); let p2 = d2.join("s2.gguf");
        TinyInkling { seed: 1, ..TinyInkling::default() }.write_gguf(&p1).unwrap();
        TinyInkling { seed: 2, ..TinyInkling::default() }.write_gguf(&p2).unwrap();
        assert_ne!(std::fs::read(&p1).unwrap(), std::fs::read(&p2).unwrap());

        std::fs::remove_dir_all(&d).ok();
        std::fs::remove_dir_all(&d2).ok();
        let _ = (aname, wname);
    }

    #[test]
    fn the_q8_option_quantizes_only_the_expert_tensors() {
        use strata_gguf::Gguf;
        let d = fixture_dir("inkling-q8experts");
        // wide() has an expert width (192) that is not a multiple of its embedding (256), like the real model's 1,152 is not
        let model = TinyInkling::wide().with_q8_experts(true);
        let gg = d.join("wide.q8.gguf");
        model.write_gguf(&gg).expect("write the quantized expert preset");
        let g = Gguf::open(&gg).unwrap();
        for nm in ["blk.0.ffn_gate_up_exps.weight", "blk.0.ffn_down_exps.weight"] {
            let t = g.tensors.iter().find(|t| t.name == nm).expect("expert tensor present");
            assert_eq!(t.ggml_type, 8, "{nm} is Q8_0 when the option is on");
        }
        // the vocabulary matrices were always Q8_0; an ordinary layer tensor stays F32 (type 0)
        let norm = g.tensors.iter().find(|t| t.name == "blk.0.attn_norm.weight").unwrap();
        assert_eq!(norm.ggml_type, 0);
        // wide()'s bigger embedding and expert width are in the metadata
        assert_eq!(g.arch_u64("embedding_length"), Some(256));
        assert_eq!(g.arch_u64("feed_forward_length"), Some(192));
        std::fs::remove_dir_all(&d).ok();
    }

    /// The reference oracle (a CPU llama.cpp, `oracle-inkling`). Task gates the three tests below on `AM2_ORACLE` pointing
    /// at it; when unset they skip so the normal `cargo test` needs no shared library or binary.
    fn oracle_bin() -> std::path::PathBuf {
        std::env::var("AM2_ORACLE").map(std::path::PathBuf::from).unwrap_or_else(|_| PathBuf::from("oracle-inkling"))
    }

    /// The `name -> ne` map a run with `ORACLE_DUMP=<dir>` writes to `<dir>/index.tsv`; values keep every trailing column
    /// too, which the dump assertions below ignore (they only read the leading shape columns).
    fn oracle_index(path: &std::path::Path) -> std::collections::HashMap<String, Vec<u64>> {
        let text = std::fs::read_to_string(path).unwrap();
        let mut m = std::collections::HashMap::new();
        for line in text.lines() {
            if line.trim().is_empty() { continue; }
            let c: Vec<&str> = line.split_whitespace().collect();
            if c.len() < 2 { continue; }
            let ne: Vec<u64> = c[1..].iter().filter_map(|s| s.parse().ok()).collect();
            if !ne.is_empty() { m.insert(c[0].to_string(), ne); }
        }
        m
    }

    /// Greedy top-1 ids (first of each step's `top`) from the oracle JSON stdout, at most `max` of them.
    fn oracle_generated_ids(bytes: &[u8], max: usize) -> Vec<u64> {
        let mut got = Vec::new();
        for b in bytes.split(|&c| c == b'\n') {
            let s = String::from_utf8_lossy(b);
            if !s.starts_with('{') || !s.contains("\"step\"") { continue; }
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(&s) {
                if let Some(id) = v["top"][0][0].as_u64() { got.push(id); if got.len() >= max { break; } }
            }
        }
        got
    }

    #[test]
    #[ignore = "runs the reference oracle (llama.cpp) on both presets; set AM2_ORACLE to its path"]
    fn oracle_loads_both_presets_and_generates_eight_tokens() {
        if std::env::var("AM2_ORACLE").is_err() { return; }
        use std::process::{Command, Stdio};
        let bin = oracle_bin();
        let d = fixture_dir("inkling-oracle-gen");
        // both presets: llama.cpp loads our GGUF and generates 8 tokens from a short id prompt on each
        for (nm, model) in [("default", TinyInkling::default()), ("wide", TinyInkling::wide())] {
            let p = d.join(format!("tiny-{nm}.gguf"));
            model.write_gguf(&p).unwrap();
            let out = Command::new(&bin)
                .args([p.to_str().expect("ascii path"), "8", "1,2,3,4"])
                .stdout(Stdio::piped()).stderr(Stdio::piped()).output().expect("run oracle");
            assert!(out.status.success(), "oracle exited non-zero on {nm}: {}", String::from_utf8_lossy(&out.stderr));
            let ids = oracle_generated_ids(&out.stdout, 8);
            assert_eq!(ids.len(), 8, "{nm}: oracle produced 8 generated tokens");
            assert!(ids.iter().all(|id| *id < model.vocab_size_padded() as u64), "{nm}: every generated id is a valid token (< padded vocab)");
        }
        std::fs::remove_dir_all(&d).ok();
    }

    #[test]
    #[ignore = "needs AM2_ORACLE set to the oracle; dumps tensors with ORACLE_DUMP and checks their shapes"]
    fn oracle_dump_writes_loop_and_attention_tensors_with_the_architectural_shapes() {
        if std::env::var("AM2_ORACLE").is_err() { return; }
        use std::process::Command;
        // a 150-token prompt (sliding window is 64, the context extent 128 -> this runs past both) plus new tokens
        let dump = fixture_dir("inkling-oracle-dump");
        let p = dump.join("tiny.inkling.gguf");
        TinyInkling::default().write_gguf(&p).unwrap();
        let prompt: String = (0..150).map(|i| i.to_string()).collect::<Vec<_>>().join(",");
        let out = Command::new(oracle_bin())
            .args([p.to_str().expect("ascii path"), "4", &prompt])
            .env("ORACLE_DUMP", &dump)
            .output().expect("run oracle with ORACLE_DUMP");
        assert!(out.status.success(), "oracle dump run failed: {}", String::from_utf8_lossy(&out.stderr));

        let dumped = oracle_index(&dump.join("index.tsv"));
        // Qcur-0 .. Qcur-9, the per-pass queries: ne = [D=32, H=4, seq] (the leading columns are D then H)
        for i in 0..10 {
            let ne = dumped.get(&format!("Qcur-{i}")).expect("Qcur present");
            assert_eq!(ne.first(), Some(&32u64), "Qcur-{i} head dim is D=32");
            assert_eq!(ne.get(1), Some(&4u64), "Qcur-{i} heads is H=4");
        }
        // l_out-0 .. l_out-9, the per-layer outputs: ne[0] = E (embedding width)
        for i in 0..10 {
            let ne = dumped.get(&format!("l_out-{i}")).expect("l_out present");
            assert_eq!(ne.first(), Some(&128u64), "l_out-{i} is embedding-sized");
        }
        // attn_out-N is dumped only for the full-attention layers (sliding-window ones are not); each is embedding-sized
        let attn: Vec<(&String, &Vec<u64>)> = dumped.iter().filter(|(k, _)| k.starts_with("attn_out-")).collect();
        assert!(!attn.is_empty(), "full-attention layers dump attn_out");
        for (k, ne) in attn { assert_eq!(ne.first(), Some(&128u64), "{k} is embedding-sized"); }

        std::fs::remove_dir_all(&dump).ok();
    }

    #[test]
    #[ignore = "runs the reference oracle; set AM2_ORACLE to its path"]
    fn oracle_runs_a_150_token_prompt_past_the_window_and_extent() {
        if std::env::var("AM2_ORACLE").is_err() { return; }
        use std::process::Command;
        let p = fixture_dir("inkling-oracle-150").join("tiny.inkling.gguf");
        TinyInkling::default().write_gguf(&p).unwrap();
        // 64-token sliding window and the 128-token extent are both shorter than this prompt
        let prompt: String = (0..150).map(|i| i.to_string()).collect::<Vec<_>>().join(",");
        let out = Command::new(oracle_bin()).args([p.to_str().expect("ascii path"), "4", &prompt]).output().expect("run oracle");
        assert!(out.status.success(), "oracle exited non-zero: {}", String::from_utf8_lossy(&out.stderr));
        let ids = oracle_generated_ids(&out.stdout, 4);
        assert_eq!(ids.len(), 4, "oracle decoded past the sliding window and the extent from a 150-token prompt");
        std::fs::remove_dir_all(p.parent().unwrap()).ok();
    }
}

use std::f64::consts::PI;

use strata_pack::gguf_write::{Kv, TensorSpec, Writer};
use strata_pack::quantize as quant;

/// A random tensor's shape: `One` is written as all 1.0 (a norm gain), `Zero` as all 0.0, and `Scale(s)` as standard-normal
/// samples times `s`. These are exactly the three kinds `make_fixture.py` uses for its `T(...)` calls.
#[derive(Debug, Clone, Copy)]
enum WeightKind { One, Zero, Scale(f32) }

impl WeightKind {
    /// The default a bare `T(name, ne)` in make_fixture.py gets: standard_normal times 0.08.
    const DEFAULT: Self = WeightKind::Scale(0.08);
    fn sample(&self, gen: &mut SplitMix64, n: usize) -> Vec<f32> {
        match self {
            WeightKind::One => vec![1.0; n],
            WeightKind::Zero => vec![0.0; n],
            WeightKind::Scale(s) => (0..n).map(|_| (sample_normal(gen) * *s) as f32).collect(),
        }
    }
}

/// One tensor the writer streams: its gguf name, its numpy shape reversed into the file's `ne` order, the kind that
/// fills it and whether the vocabulary/expert matrices are written Q8_0. A single source for both the writer and the
/// round-trip test's "expected" list, so a drift between them fails loudly.
#[derive(Debug, Clone)]
struct Def {
    name: String,
    ne: Vec<usize>,
    kind: WeightKind,
    q8: bool,
}

/// A tiny random `inkling2_loop` model for the Workshop's loop-model tests (W1–W4). It writes a GGUF with the same tensors,
/// names, shapes and metadata keys as [`make_fixture.py`] (`promo/documentary/code/inkling-w0/make_fixture.py`) — readable by
/// `strata_gguf` and loadable by the llama.cpp oracle — but with a small built-in gpt2 BPE tokenizer (no 16 GB vocabulary)
/// and the loop core, block size and sliding-window pattern as fields. The seed weights the deterministic randoms; the two
/// presets [`TinyInkling::default`] and [`TinyInkling::wide`] are what W1–W4's tests build.
#[derive(Debug, Clone)]
pub struct TinyInkling {
    pub seed: u64,
    // dimensions (make_fixture.py's: E, H/kv heads, D=head_dim, NE/NU experts, FF expert width)
    pub embedding: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub num_experts: usize,
    pub used_experts: usize,
    pub ffn: usize,
    // the loop over `loop_start..loop_end`, run `loop_passes` times; each pass keeps its own KV cache and conv state with
    // small controller weights passing messages between passes
    pub num_layers: usize,
    pub loop_start: usize,
    pub loop_end: usize,
    pub loop_passes: usize,
    /// per-layer relative-position bias width (`REL`)
    pub rel_len: usize,
    /// short-convolution taps on K/V/attention/expert outputs
    pub n_conv: usize,
    /// rank of the loop controllers' down/up projections
    pub gate_rank: usize,
    /// depth-mixing block size for the residual streams (`attn_res.block_size`)
    pub block_size: usize,
    /// which layers attend to the whole context (a full-attention layer inside the loop core and one after it)
    pub full_attention: Vec<bool>,
    /// `tokenizer.ggml.pre`; a gpt2 BPE tokenizer llama.cpp accepts for this model
    pub pre: String,
    /// also write the expert tensors as Q8_0 (the two vocab matrices are always Q8_0, as make_fixture.py does). The option
    /// exists so the pack tests can quantize an expert tensor whose width is not a multiple of the embedding.
    pub quantize_experts: bool,
}

impl Default for TinyInkling {
    fn default() -> Self {
        Self {
            seed: 0x5EED,
            embedding: 128, num_heads: 4, num_kv_heads: 2, head_dim: 32,
            num_experts: 8, used_experts: 2, ffn: 64,
            num_layers: 6, loop_start: 2, loop_end: 4, loop_passes: 3,
            rel_len: 16, n_conv: 4, gate_rank: 8, block_size: 2,
            full_attention: vec![true, true, true, false, true, false],
            pre: "inkling2".into(),
            quantize_experts: false,
        }
    }
}

impl TinyInkling {
    /// A wider preset (embedding 256, expert width 192 — not a multiple of the embedding, like the real model's 1,152 is
    /// not) so the pack tests can quantize an expert tensor whose axis is odd-sized.
    pub fn wide() -> Self {
        let mut m = Self::default();
        m.embedding = 256;
        m.ffn = 192;
        m
    }

    /// Point the option at the experts: write `ffn_gate_up_exps` / `ffn_down_exps` as Q8_0 alongside the vocab matrices.
    pub fn with_q8_experts(mut self, q8: bool) -> Self {
        self.quantize_experts = q8;
        self
    }

    /// The small tokenizer's token count: two specials (pad, end-of-turn at ids 0 and 1, the ones I2 parses) plus the 256
    /// byte-level symbols. `tokenizer.ggml.pre = inkling2` makes llama.cpp treat it as a gpt2 BPE vocabulary.
    fn token_count(&self) -> usize { 2 + 256 }

    /// The padded vocab size written as `inkling2_loop.vocab_size`, above the token count (as the real model pads its
    /// 200,058 to 201,024 so the embedding/output matrices are a clean width).
    fn vocab_size_padded(&self) -> usize {
        let n = self.token_count();
        (n + 63) / 64 * 64
    }

    /// The gguf metadata: every key `make_fixture.py` writes, in the same order, with this model's values — tokenizer
    /// arrays here and the loop/attention hyper-parameters as fields.
    pub fn kv(&self) -> Vec<(&'static str, Kv)> {
        let nl = self.num_layers;
        let nv = self.vocab_size_padded() as u32;
        let nn = self.token_count() as u32;
        // the same constants make_fixture.py writes for the tiny file (from tiny-fixture-header.txt), recomputed where the
        // script derived them from D: qk_clip is sqrt(head_dim) * 4
        let rms_eps = 9.999999974752427e-07_f32;
        let qk_clip = (self.head_dim as f32).sqrt() * 4.0f32;
        // small gpt2 tokenizer: pad and end-of-turn specials first, then the byte symbols; SPECIALS are CONTROL (3)
        let tokens: Vec<String> = {
            let mut t = vec!["<|pad|>".to_string(), "<|eot|>".to_string()];
            for c in 0..=255u32 { t.push(char::from_u32(c).unwrap().to_string()); }
            t
        };
        let token_type: Vec<i32> = (0..self.token_count()).map(|i| if i < 2 { 3 } else { 1 }).collect();
        // specials I2 parses: pad at id 0, end-of-turn at id 1, written into the file so a tool can point at them
        let pad_id = 0u32;
        let eot_id = 1u32;
        vec![
            ("general.architecture", Kv::Str("inkling2_loop".into())),
            ("general.type", Kv::Str("model".into())),
            ("general.sampling.temp", Kv::F32(1.0)),
            ("general.name", Kv::Str("Inkling-2-Mini".into())),
            ("general.size_label", Kv::Str("128x1.8B".into())),
            ("general.license", Kv::Str("apache-2.0".into())),
            ("general.license.link", Kv::Str("https://www.apache.org/licenses/LICENSE-2.0".into())),
            ("general.tags", Kv::Strings(
                vec!["conversational", "image-text-to-text", "moe", "image-text-to-text"].into_iter().map(str::to_string).collect())),
            ("inkling2_loop.block_count", Kv::U32(nl as u32)),
            ("inkling2_loop.context_length", Kv::U32(1_048_576)),
            ("inkling2_loop.embedding_length", Kv::U32(self.embedding as u32)),
            ("inkling2_loop.feed_forward_length", Kv::U32(self.ffn as u32)),
             ("inkling2_loop.attention.head_count", Kv::Ints(vec![self.num_heads as i32; nl])),
             ("inkling2_loop.attention.head_count_kv", Kv::Ints(vec![self.num_kv_heads as i32; nl])),
            ("inkling2_loop.attention.layer_norm_rms_epsilon", Kv::F32(rms_eps)),
            ("inkling2_loop.expert_count", Kv::U32(self.num_experts as u32)),
            ("inkling2_loop.expert_used_count", Kv::U32(self.used_experts as u32)),
            ("inkling2_loop.attention.key_length", Kv::U32(self.head_dim as u32)),
            ("inkling2_loop.attention.value_length", Kv::U32(self.head_dim as u32)),
            ("inkling2_loop.loop.count", Kv::U32(self.loop_passes as u32)),
            ("inkling2_loop.loop.start", Kv::U32(self.loop_start as u32)),
            ("inkling2_loop.loop.end", Kv::U32(self.loop_end as u32)),
            ("inkling2_loop.loop.gate_rank", Kv::U32(self.gate_rank as u32)),
            ("inkling2_loop.vocab_size", Kv::U32(nv)),
            ("inkling2_loop.vocab_size_unpadded", Kv::U32(nn)),
            ("inkling2_loop.expert_feed_forward_length", Kv::U32(self.ffn as u32)),
            ("inkling2_loop.attention.key_length_swa", Kv::U32(self.head_dim as u32)),
            ("inkling2_loop.attention.value_length_swa", Kv::U32(self.head_dim as u32)),
            ("inkling2_loop.rope.dimension_count", Kv::U32(0)),
            ("inkling2_loop.attention.sliding_window", Kv::U32(64)),
            ("inkling2_loop.attention.sliding_window_pattern", Kv::Bools(self.full_attention.clone())),
            ("inkling2_loop.shortconv.l_cache", Kv::U32(4)),
            ("inkling2_loop.expert_weights_scale", Kv::F32(8.0)),
            ("inkling2_loop.activation.situ_beta", Kv::F32(4.0)),
            ("inkling2_loop.activation.situ_linear_beta", Kv::F32(25.0)),
            ("inkling2_loop.attn_res.block_size", Kv::U32(self.block_size as u32)),
            ("inkling2_loop.final_logit_softcapping", Kv::F32(30.0)),
            ("inkling2_loop.attention.relative_length", Kv::U32(16)),
            ("inkling2_loop.attention.relative_extent", Kv::U32(128)),
            ("inkling2_loop.attention.qk_clip", Kv::F32(qk_clip)),
            ("inkling2_loop.attention.relative_clip", Kv::F32(8.0)),
            ("inkling2_loop.attention.log_scale_floor", Kv::U32(64)),
            ("inkling2_loop.attention.log_scale_alpha", Kv::F32(0.1)),
            ("tokenizer.ggml.model", Kv::Str("gpt2".into())),
            ("tokenizer.ggml.pre", Kv::Str(self.pre.clone().into())),
            ("tokenizer.ggml.tokens", Kv::Strings(tokens)),
            ("tokenizer.ggml.token_type", Kv::Ints(token_type)),
            ("tokenizer.ggml.merges", Kv::Strings(Vec::new())),
            ("tokenizer.ggml.eos_token_id", Kv::U32(eot_id)),
            ("tokenizer.ggml.padding_token_id", Kv::U32(pad_id)),
            ("tokenizer.ggml.eot_token_id", Kv::U32(eot_id)),
            ("tokenizer.ggml.add_bos_token", Kv::Bool(false)),
            ("tokenizer.ggml.add_eos_token", Kv::Bool(false)),
            ("tokenizer.chat_template", Kv::Str(CHAT_TEMPLATE.into())),
            ("tokenizer.ggml.clean_spaces", Kv::Bool(false)),
            ("inkling2_loop.loop.varying", Kv::Bool(false)),
        ]
    }

    /// Every tensor in writing order (shared embed/output, then one layer at a time), the make_fixture.py `T(...)` shapes,
    /// the fill and whether it is Q8_0 — the single list the writer streams and the round-trip test checks.
    fn tensor_defs(&self, quantize_experts: bool) -> Vec<Def> {
        let E = self.embedding; let H = self.num_heads; let HKV = self.num_kv_heads; let D = self.head_dim;
        let NE = self.num_experts; let FF = self.ffn; let REL = self.rel_len; let NCONV = self.n_conv; let RANK = self.gate_rank;
        let NV = self.vocab_size_padded();
        let looped = |i: usize| self.loop_start <= i && i < self.loop_end;
        let mut out: Vec<Def> = vec![
            Def { name: "token_embd.weight".into(), ne: vec![E, NV], kind: WeightKind::Scale(0.5), q8: true },   // the two vocab matrices are always Q8_0
            Def { name: "token_embd_norm.weight".into(), ne: vec![E], kind: WeightKind::One, q8: false },
            Def { name: "output.weight".into(), ne: vec![E, NV], kind: WeightKind::Scale(0.08), q8: true },
            Def { name: "output_norm.weight".into(), ne: vec![E], kind: WeightKind::One, q8: false },
            Def { name: "output_res_query.weight".into(), ne: vec![E], kind: WeightKind::Scale(0.1), q8: false },
        ];
        for i in 0..self.num_layers {
            let swa = self.full_attention[i];
            let b = format!("blk.{i}.");
            // attn_q/k/v/output/gate/sink/rel, default scale (a bare T(name, ne) in make_fixture.py)
            for (n, ne) in [
                ("attn_q", vec![E, D * H]), ("attn_k", vec![E, D * HKV]), ("attn_v", vec![E, D * HKV]),
                ("attn_output", vec![D * H, E]), ("attn_gate", vec![E, D * H]), ("attn_sink_proj", vec![E, H]),
                ("attn_rel", vec![E, REL * H]),
            ] {
                out.push(Def { name: format!("{b}{n}.weight"), ne, kind: WeightKind::DEFAULT, q8: false });
            }
            out.push(Def { name: format!("{b}attn_q_clip.weight"), ne: vec![D, H], kind: WeightKind::One, q8: false });
            out.push(Def { name: format!("{b}attn_rel_clip.weight"), ne: vec![REL, H], kind: WeightKind::One, q8: false });
            // the sink's relative bias is narrow for a full-attention layer and wide for a sliding-window one
            let rel_bias_second = if swa { 64 } else { 128 };
            out.push(Def { name: format!("{b}attn_rel_bias.weight"), ne: vec![REL, rel_bias_second], kind: WeightKind::Scale(0.3), q8: false });
            for (n, ne) in [
                ("attn_k_conv", vec![NCONV, D * HKV]), ("attn_v_conv", vec![NCONV, D * HKV]),
                ("attn_post_conv", vec![NCONV, E]), ("ffn_post_conv", vec![NCONV, E]),
            ] {
                out.push(Def { name: format!("{b}{n}.weight"), ne, kind: WeightKind::Scale(0.2), q8: false });
            }
            if swa {
                // a full-attention layer normalizes Q/K/V together (one gain); the sliding-window ones keep per-input norms
                out.push(Def { name: format!("{b}attn_norm.weight"), ne: vec![E], kind: WeightKind::One, q8: false });
            } else {
                for n in ["attn_q_inp_norm", "attn_k_inp_norm", "attn_v_inp_norm"] {
                    out.push(Def { name: format!("{b}{n}.weight"), ne: vec![E], kind: WeightKind::One, q8: false });
                }
                for n in ["attn_k_conv_proj", "attn_v_conv_proj"] {
                    out.push(Def { name: format!("{b}{n}.weight"), ne: vec![E, NCONV * HKV], kind: WeightKind::DEFAULT, q8: false });
                }
            }
            // a new residual stream starts every block (block_size); each layer mixes the earlier streams with queries.
            if i > 0 {
                let res_channels = if swa { 1 } else { 3 };
                out.push(Def { name: format!("{b}attn_res_query.weight"), ne: vec![E, res_channels], kind: WeightKind::Scale(0.1), q8: false });
            }
            out.push(Def { name: format!("{b}ffn_res_query.weight"), ne: vec![E], kind: WeightKind::Scale(0.1), q8: false });
            out.push(Def { name: format!("{b}ffn_norm.weight"), ne: vec![E], kind: WeightKind::One, q8: false });
            out.push(Def { name: format!("{b}ffn_gate_inp.weight"), ne: vec![E, NE], kind: WeightKind::Scale(0.2), q8: false });
            out.push(Def { name: format!("{b}exp_probs_b.bias"), ne: vec![NE], kind: WeightKind::Scale(0.05), q8: false });
            // the expert weights, Q8_0 when the option is on (the vocabulary matrices stay Q8_0 regardless)
            let e = quantize_experts;
            out.push(Def { name: format!("{b}ffn_gate_up_exps.weight"), ne: vec![E, 2 * FF, NE], kind: WeightKind::DEFAULT, q8: e });
            out.push(Def { name: format!("{b}ffn_down_exps.weight"), ne: vec![FF, E, NE], kind: WeightKind::DEFAULT, q8: e });
            if looped(i) {
                // the loop core: a message into each pass and down/up/forget controllers for Q, K, V and FFN passes
                out.push(Def { name: format!("{b}loop_msg_query.weight"), ne: vec![E, 4], kind: WeightKind::Scale(0.1), q8: false });
                for c in ["q", "k", "v", "ffn"] {
                    out.push(Def { name: format!("{b}loop_{c}_ctrl_down.weight"), ne: vec![2 * E, RANK], kind: WeightKind::DEFAULT, q8: false });
                    out.push(Def { name: format!("{b}loop_{c}_ctrl_gain.weight"), ne: vec![RANK, E], kind: WeightKind::DEFAULT, q8: false });
                    out.push(Def { name: format!("{b}loop_{c}_ctrl_forget.weight"), ne: vec![RANK, E], kind: WeightKind::DEFAULT, q8: false });
                }
            }
        }
        out
    }

    /// The tensors as (name, `ne`, q8) — what the round-trip test compares against a read-back file.
    pub fn tensor_list(&self, quantize_experts: bool) -> Vec<(String, Vec<usize>, bool)> {
        self.tensor_defs(quantize_experts).into_iter().map(|d| (d.name, d.ne, d.q8)).collect()
    }

    /// Write the fixture to `out` as a GGUF with `strata_pack::gguf_write`, exactly the writer the converter uses. The two
    /// vocabulary matrices are Q8_0; when [`TinyInkling::quantize_experts`] is set the expert tensors are too. Byte order is
    /// fixed (KV in insertion order, tensors in [`Self::tensor_defs`] order, weights from `seed`), so the same seed gives the
    /// same bytes — the determinism the round-trip test relies on.
    pub fn write_gguf(&self, out: &Path) -> anyhow::Result<()> {
        let defs = self.tensor_defs(self.quantize_experts);
        // specs need the numpy order (the reverse of `ne`) so the writer reverses it back into this tensor's `ne` on disk
        let mut specs: Vec<TensorSpec> = Vec::with_capacity(defs.len());
        for d in &defs {
            let n = nbytes(d);
            let mut shape: Vec<u64> = d.ne.iter().map(|&x| x as u64).collect();
            shape.reverse();
            specs.push(TensorSpec { name: d.name.clone(), shape, ggml_type: if d.q8 { 8 } else { 0 }, nbytes: n });
        }
        let mut w = Writer::create_typed(out, &self.kv(), specs)?;
        let mut gen = SplitMix64::new(self.seed);
        for d in &defs {
            let n = nbytes(d);
            if d.q8 {
                // quantize the F32 values to Q8_0 (ggml's reference, as strata_pack's packer does) and stream those bytes
                let vals = d.kind.sample(&mut gen, d.ne.iter().product());
                let mut q = Vec::with_capacity((n / 34) as usize);
                quant::q8_0(&vals, &mut q);
                assert_eq!(q.len(), n as usize, "{}: Q8_0 byte length", d.name);
                w.tensor_data(&q)?;
            } else {
                let vals = d.kind.sample(&mut gen, d.ne.iter().product());
                let bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
                assert_eq!(bytes.len(), n as usize, "{}: F32 byte length", d.name);
                w.tensor_data(&bytes)?;
            }
            w.end_tensor()?;
        }
        w.finish()?;
        Ok(())
    }
}

/// Bytes a tensor takes on disk: `n/32 * 34` for Q8_0 (a 32-element block is a fp16 scale plus 32 int8 codes), `n*4` else.
fn nbytes(d: &Def) -> u64 {
    let n = d.ne.iter().product::<usize>() as u64;
    if d.q8 { n / 32 * 34 } else { n * 4 }
}

/// One standard-normal sample (seeded, via Box-Muller), so the weights a fixture writes are reproducible without `rand`.
fn sample_normal(gen: &mut SplitMix64) -> f32 {
    let u1 = (((gen.next_u64() >> 11) as f64) + 1.0) / (1u64 << 53) as f64;      // in (0, 1]
    let u2 = (gen.next_u64() as f64) / (1u64 << 53) as f64;                       // in [0, 1)
    let r = (-2.0 * u1.ln()).sqrt();
    (r * (2.0 * PI * u2).cos()) as f32
}

/// The tiny fixture's chat template — metadata only, so its exact text does not matter for the GGUF; it renders real prompts
/// the way Prastara's server renders them. A fixed string keeps the file deterministic.
const CHAT_TEMPLATE: &str = "sys:<eff> user:<text>\nassistant::<answer><|end_message|>";

/// Deterministic generator so `write` and a test's expected values agree exactly.
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Uniform f32 in [0, 1).
    fn next_f32(&mut self) -> f32 {
        let bits = (self.next_u64() >> 40) as u32; // top 24 bits
        bits as f32 / 16_777_216.0
    }
}

fn mat(gen: &mut SplitMix64, rows: usize, cols: usize, scale: f32) -> Vec<f32> {
    (0..rows * cols).map(|_| scale * (gen.next_f32() * 2.0 - 1.0)).collect()
}
