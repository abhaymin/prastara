//! `am2strata-workshop`: the Workshop's jobs as a command, in Rust (no Python).
//!
//! ```text
//! am2strata-workshop info <model.gguf | hf-folder>
//! am2strata-workshop convert <hf-folder> --out <model.gguf> [--outtype f16|f32] [--pre-tokenizer NAME] [--yarn 4 [--yarn-original 32768]]
//! am2strata-workshop quantize <in.gguf> --out <out.gguf> --type q8_0|q6_k|q5_k|q4_k|q4_0|f16
//! am2strata-workshop merge <base-hf-folder> <adapter-folder> --out <merged-hf-folder>
//! am2strata-workshop upcycle <hf-folder> --experts 10 --top-k 1 --out <dir>
//! am2strata-workshop export-moe <upcycled-folder> [--adapter <dir>] --out <model.gguf> [--type q4_k|q8_0|f16|f32]
//! am2strata-workshop abliterate <hf-folder> --harmful a.txt --harmless b.txt [--layer auto|N|N:P] [--scale 1.0] [--check N] --out <dir>
//! am2strata-workshop train --model <hf-folder> --data <chat.jsonl> --out <adapter-folder> [--soup soup.yaml] [...]
//! am2strata-workshop find-lr --model <hf-folder> --data <chat.jsonl> [--steps 40]
//! am2strata-workshop profile (--model <hf-folder> | --preset llama-2-7b) [--data <chat.jsonl>] [...]
//! am2strata-workshop autopilot --model <hf-folder> --data <chat.jsonl> [--time-budget 2h] [--ram-budget 12GB]
//! am2strata-workshop regress --model <hf-folder> [--adapter <dir>] [--passes 1,2] --suite <suite.json> --out <report-dir> [--against <report>]
//! am2strata-workshop clean <in.jsonl>... --out <out.jsonl> [--model <hf-folder>] [--max-len 1024] [--min-answer-tokens N] [--max-repeat R] [--dedupe exact|near|off] [--lang-match] [--exclude held-out.jsonl] [--cap-per-language N [--cap-only in.jsonl]] [--report card.md]
//! am2strata-workshop grade-train --model <hf-folder> --data <prompts.jsonl> --samples N --grader code|judge --rounds R --suite <suite.json> --out <dir>
//! ```
//!
//! `train` fine-tunes a LoRA adapter on the CPU (after Soup, in Rust: docs/MODEL-WORKSHOP-PLAN.md). Its progress
//! events carry `step`, `total`, `loss`, `lr` and `eta` beside `done`/`total`/`message`; SIGTERM writes a checkpoint
//! and stops, and `--resume auto` continues from it.
//!
//! With `--json` every line on stdout is one event: `{"event":"start",...}`, `{"event":"log","message":...}`,
//! `{"event":"done","output":...,"bytes":...,"seconds":...}` or `{"event":"error","message":...}`. The server's Workshop
//! runs jobs this way (a job that runs out of memory cannot take the server down); people get readable lines.

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use clap::{Args, Parser, Subcommand};
use serde_json::{json, Value};
use strata_workshop::chatdata::{self, ChatTokenizer, DataFormat};
use strata_workshop::finetune::{self, Event, FindLr, Resume, TrainSettings};
use strata_workshop::model::{BaseDtype, ModelConfig, Target};
use strata_workshop::profile::{self, AutopilotInput, Machine, ProfileInput};
use strata_workshop::{clean, convert, datagen, merge, quantize_gguf, soup};

#[derive(Parser)]
#[command(name = "am2strata-workshop", about = "Convert, quantize and merge models in Rust (Prastara's Workshop)")]
struct Cli {
    /// one JSON event per line on stdout (for the server and scripts)
    #[arg(long, global = true)]
    json: bool,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// what a GGUF or a Hugging Face model folder holds
    Info { path: PathBuf },
    /// Avarta: a looped copy of a Hugging Face model folder; blocks start..end run `passes` times with the same
    /// weights, each extra pass behind a gate that starts closed (the copy gives the base's logits until trained)
    Loop {
        model: PathBuf,
        #[arg(long)]
        out: PathBuf,
        /// the first block of the loop
        #[arg(long)]
        start: usize,
        /// the block after the loop's last
        #[arg(long)]
        end: usize,
        /// how many times the loop runs (2 to 16)
        #[arg(long, default_value_t = 2)]
        passes: usize,
    },
    /// LoRA upcycling: a copy of a dense Llama or Qwen3 folder (looped or not) marked as a mixture of `experts`
    /// experts, `top-k` per token; weights linked. `train` on it trains a router and every expert's LoRA over the shared
    /// feed-forward (it starts exactly as the base); `export-moe` writes the real qwen3moe model
    Upcycle {
        model: PathBuf,
        #[arg(long)]
        out: PathBuf,
        /// experts per block (2 to 256)
        #[arg(long, default_value_t = 10)]
        experts: usize,
        /// experts each token is routed to (1: Switch-style, the active size of the dense base)
        #[arg(long, default_value_t = 1)]
        top_k: usize,
    },
    /// an upcycled folder and the adapter trained on it to a qwen3moe GGUF: each expert's LoRA folded into its own
    /// copy of the feed-forward, the router from the adapter; streamed, and quantized as it is written with --type
    ExportMoe {
        model: PathBuf,
        /// the adapter `train` wrote on the upcycled folder (without one every expert is the base feed-forward)
        #[arg(long)]
        adapter: Option<PathBuf>,
        #[arg(long)]
        out: PathBuf,
        /// f32, f16, q8_0, q6_k, q5_k, q4_k or q4_0
        #[arg(long = "type", default_value = "q4_k")]
        ty: String,
        /// name the BPE pre-tokenizer when its hash is not a known one (e.g. "qwen2", "llama-bpe", "gpt-2")
        #[arg(long)]
        pre_tokenizer: Option<String>,
    },
    /// abliteration: remove the direction the model writes into its residual stream before refusing (from the mean
    /// residuals of two prompt lists, one prompt per line) from every o_proj, down_proj and the embeddings; the copy
    /// is marked private and stays on this PC
    Abliterate {
        model: PathBuf,
        /// prompts the model refuses, one per line
        #[arg(long)]
        harmful: PathBuf,
        /// ordinary prompts, one per line
        #[arg(long)]
        harmless: PathBuf,
        /// the block the direction is read after: auto (the most separating one between 30 and 70 % deep), N, or N:P
        /// for pass P of a loop
        #[arg(long, default_value = "auto")]
        layer: String,
        /// how much of the direction to remove (1.0: all of it)
        #[arg(long, default_value_t = 1.0)]
        scale: f32,
        /// answer N held-out prompts of each list before and after and count refusals (slow on big models: no cache)
        #[arg(long, default_value_t = 0)]
        check: usize,
        /// cpu (the default), cuda:N or metal[:N]; several split the blocks
        #[arg(long, value_delimiter = ',')]
        devices: Vec<String>,
        /// hold the model as f32, f16 or bf16 while reading (default: f32 for an f32 checkpoint, else bf16)
        #[arg(long)]
        base_dtype: Option<String>,
        #[arg(long)]
        out: PathBuf,
    },
    /// a long-context copy of a model folder: YaRN positions to --length, every --full-every-th layer full
    /// attention, the rest a --window of the last tokens (weights linked; it still needs long-context training)
    Context {
        model: PathBuf,
        #[arg(long)]
        out: PathBuf,
        /// the context it is set up for, in tokens (e.g. 1048576)
        #[arg(long)]
        length: usize,
        /// the length the model was trained to (default: its max_position_embeddings)
        #[arg(long)]
        original: Option<usize>,
        #[arg(long, default_value_t = 4096)]
        window: usize,
        #[arg(long, default_value_t = 4)]
        full_every: usize,
    },
    /// a Hugging Face model folder (safetensors + config + tokenizer) to one GGUF, as llama.cpp's converter writes it
    Convert {
        dir: PathBuf,
        #[arg(long)]
        out: PathBuf,
        /// f16 (half the size) or f32
        #[arg(long, default_value = "f16")]
        outtype: String,
        /// name the BPE pre-tokenizer when its hash is not a known one (e.g. "llama-bpe", "qwen2", "gpt-2")
        #[arg(long)]
        pre_tokenizer: Option<String>,
        /// add YaRN positions (Qwen3 folders without rope_scaling): this factor over --yarn-original tokens; Qwen's
        /// recommendation is 4 over 32768 (131,072 tokens)
        #[arg(long)]
        yarn: Option<f64>,
        /// the length --yarn stretches (default: the config's max_position_embeddings, at most 32768)
        #[arg(long)]
        yarn_original: Option<usize>,
    },
    /// a GGUF to another type, as `llama-quantize --pure` does
    Quantize {
        input: PathBuf,
        #[arg(long)]
        out: PathBuf,
        /// q8_0, q6_k, q5_k, q4_k, q4_0 or f16
        #[arg(long = "type")]
        ty: String,
    },
    /// training data written by the server's loaded model (AM2STRATA_URL, AM2STRATA_API_KEY): from documents (qa,
    /// chat), from seed examples (instruct), or from the built-in catalogues (code, indic, translate), filtered,
    /// de-duplicated and optionally judged
    Datagen {
        /// qa, chat, instruct, code, indic or translate
        #[arg(long, default_value = "qa")]
        recipe: String,
        /// documents: files or folders (text, Markdown, code, HTML, PDF)
        #[arg(long = "source")]
        sources: Vec<PathBuf>,
        /// instruct: a JSONL of example chat records
        #[arg(long)]
        seeds: Option<PathBuf>,
        #[arg(long)]
        out: PathBuf,
        #[arg(long, default_value_t = 100)]
        count: usize,
        #[arg(long, default_value_t = 4)]
        per_chunk: usize,
        #[arg(long, default_value_t = 3000)]
        chunk_chars: usize,
        #[arg(long, default_value_t = 200)]
        overlap: usize,
        #[arg(long, default_value_t = 0.7)]
        temperature: f64,
        /// judge each answer 1-5 against its source and keep those at least this (0: no judge)
        #[arg(long, default_value_t = 0)]
        min_score: u8,
        #[arg(long)]
        topic: Option<String>,
        /// qa/chat/instruct: the language to write in; indic/translate: Indian languages (codes or names,
        /// comma-separated; all 15 when not given); code: programming languages (Rust, Python, ...)
        #[arg(long)]
        language: Option<String>,
        /// code, indic, translate: the draw from the built-in catalogue (the same seed gives the same requests)
        #[arg(long, default_value_t = 1)]
        seed: u64,
        /// code: compile each self-contained Rust answer with rustc and drop those that fail
        #[arg(long)]
        check_code: bool,
        /// let a thinking model think before it writes (slower)
        #[arg(long)]
        think: bool,
        /// the model to write with (any the server lists, remote ones included); default: the loaded one
        #[arg(long)]
        model: Option<String>,
    },
    /// a chat dataset cleaned for training: duplicates and near-copies, answers that repeat themselves, are too short,
    /// empty, echo the prompt, are in the wrong script or carry template junk, prompts too long for --max-len, rows of
    /// a held-out set; seeded order, a card with the count per rule and examples
    Clean {
        /// the JSONL files to read (any shape the trainer reads)
        #[arg(required = true)]
        inputs: Vec<PathBuf>,
        #[arg(long)]
        out: PathBuf,
        /// a Hugging Face model folder: its tokenizer and chat template measure the rows in tokens (else in words, and
        /// the length rule is not checked)
        #[arg(long)]
        model: Option<PathBuf>,
        #[arg(long)]
        chat_template: Option<PathBuf>,
        /// the trainer's sequence length: a row whose prompt alone fills it is dropped
        #[arg(long, default_value_t = 1024)]
        max_len: usize,
        /// an answer needs at least this many tokens (words without --model)
        #[arg(long, default_value_t = 4)]
        min_answer_tokens: usize,
        /// and at least this share of the prompt's tokens (0: no such rule)
        #[arg(long, default_value_t = 0.0)]
        min_answer_ratio: f64,
        /// the most of an answer's word 3-grams that may repeat (0 to 1)
        #[arg(long, default_value_t = 0.5)]
        max_repeat: f64,
        /// exact, near (exact too) or off
        #[arg(long, default_value = "near")]
        dedupe: String,
        /// near-duplicates: prompt and answer at least this alike (MinHash Jaccard, 0 to 1)
        #[arg(long, default_value_t = 0.8)]
        near: f64,
        /// the answer must be in the script of the prompt, its meta.language, or the language a translation asks for
        #[arg(long)]
        lang_match: bool,
        /// rows to keep out (a regression test's held-out set; repeat it)
        #[arg(long)]
        exclude: Vec<PathBuf>,
        /// at most this many rows per language (meta.language), drawn by the seed
        #[arg(long)]
        cap_per_language: Option<usize>,
        /// the cap drops rows only from these inputs (repeat it); the others are kept whole and count towards it
        #[arg(long)]
        cap_only: Vec<PathBuf>,
        #[arg(long, default_value_t = 7)]
        seed: u64,
        /// the card (default: <out>.card.md)
        #[arg(long)]
        report: Option<PathBuf>,
    },
    /// the model library: ready-made recipes (Soup's catalogue, ported, and Prastara's own) for text, code, tools,
    /// vision, OCR, image, speech, music and video models:
    /// list [--ability A] [--task T] [--origin soup|prastara] [--query Q] [--trains] | show <name> |
    /// init <name> --out <folder> (soup.yaml and DATA.md) | import --catalog catalog.py [--archs archs.json] --commit SHA --out soup-recipes.json
    Library {
        action: String,
        name: Option<String>,
        /// text, code, tools, reasoning, translation, embedding, image-understanding, ocr, image-generation,
        /// audio-understanding, speech-recognition, speech-generation, music-generation, video-generation
        #[arg(long)]
        ability: Option<String>,
        #[arg(long)]
        task: Option<String>,
        #[arg(long)]
        origin: Option<String>,
        #[arg(long)]
        query: Option<String>,
        /// only the recipes this Workshop trains now
        #[arg(long)]
        trains: bool,
        #[arg(long)]
        out: Option<PathBuf>,
        /// import: Soup's src/soup_cli/recipes/catalog.py
        #[arg(long)]
        catalog: Option<PathBuf>,
        /// import: Soup's tests/fixtures/recipe_base_architectures.json
        #[arg(long)]
        archs: Option<PathBuf>,
        /// import: the Soup commit the files are from
        #[arg(long)]
        commit: Option<String>,
    },
    /// a LoRA adapter (PEFT or Unsloth: adapter_model.safetensors + adapter_config.json) into its base model
    Merge {
        base: PathBuf,
        adapter: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
    /// the regression suite on a model (and an adapter), at each pass count of a loop: held-out loss, code run in a
    /// sandbox (pass@1), QA exact match and F1, judge scores (through the server's API, AM2STRATA_URL/KEY); writes
    /// report.json, report.md and answers.jsonl, and with --against lists what regressed beyond noise (a paired bootstrap)
    Regress {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        adapter: Option<PathBuf>,
        /// pass counts to run a looped model at, e.g. 1,2 (default: its own)
        #[arg(long, value_delimiter = ',')]
        passes: Vec<usize>,
        #[arg(long)]
        suite: PathBuf,
        #[arg(long)]
        out: PathBuf,
        /// an earlier report (report.json or its folder); several: each test against the best of them
        #[arg(long)]
        against: Vec<PathBuf>,
        /// cpu (the default), cuda:N or metal[:N]; several split the blocks
        #[arg(long, value_delimiter = ',')]
        devices: Vec<String>,
        #[arg(long)]
        base_dtype: Option<String>,
        /// prompts longer than this many tokens are left out
        #[arg(long, default_value_t = 4096)]
        max_prompt_tokens: usize,
        /// the model that judges (any the server lists); default the suite's judge_model, else the loaded one
        #[arg(long)]
        judge_model: Option<String>,
    },
    /// training by grading, in rounds: N sampled answers per prompt, graded (code: run with its tests; judge: 1-5
    /// through the server's API), the best of each prompt fine-tuned on, the suite run, and the adapter promoted only
    /// when nothing regressed; a round log in --out (rounds.jsonl, rounds.md, promoted.json)
    GradeTrain {
        #[arg(long)]
        model: PathBuf,
        /// an adapter to start from
        #[arg(long)]
        adapter: Option<PathBuf>,
        /// prompts: code {"id","prompt","language","test"|"cases"}, judge {"id","prompt","reference"?}
        #[arg(long)]
        data: PathBuf,
        #[arg(long, default_value_t = 4)]
        samples: usize,
        /// code or judge
        #[arg(long, default_value = "code")]
        grader: String,
        #[arg(long, default_value_t = 1)]
        rounds: usize,
        #[arg(long)]
        suite: PathBuf,
        #[arg(long)]
        out: PathBuf,
        #[arg(long, default_value_t = 0.8)]
        temperature: f64,
        #[arg(long, default_value_t = 0.95)]
        top_p: f64,
        /// new tokens per answer
        #[arg(long, default_value_t = 512)]
        max_tokens: usize,
        /// judge: the least score (1-5) an answer needs to be trained on
        #[arg(long, default_value_t = 4)]
        min_score: u8,
        #[arg(long, default_value_t = 1)]
        seed: u64,
        #[arg(long, value_delimiter = ',')]
        passes: Vec<usize>,
        #[arg(long, value_delimiter = ',')]
        devices: Vec<String>,
        #[arg(long)]
        base_dtype: Option<String>,
        #[arg(long, default_value_t = 2e-4)]
        lr: f64,
        #[arg(long, default_value_t = 2.0)]
        epochs: f64,
        #[arg(long)]
        max_steps: Option<usize>,
        #[arg(long, default_value_t = 16)]
        rank: usize,
        #[arg(long)]
        train_layers: Option<usize>,
        #[arg(long, default_value_t = 2048)]
        max_len: usize,
        #[arg(long, default_value_t = 1)]
        accumulation: usize,
        #[arg(long, default_value_t = 4096)]
        max_prompt_tokens: usize,
        #[arg(long)]
        judge_model: Option<String>,
    },
    /// fine-tune a LoRA adapter on chat data (CPU; Llama, Mistral, Qwen2, Qwen3 decoders; a 4-bit base is not done)
    Train {
        #[command(flatten)]
        run: RunArgs,
        /// run although the estimate says it will not fit in the free memory
        #[arg(long)]
        force: bool,
    },
    /// the learning-rate range test: a short run with a rising rate, and the rate it suggests (writes nothing)
    FindLr {
        #[command(flatten)]
        run: RunArgs,
        #[arg(long, default_value_t = 1e-7)]
        start_lr: f64,
        #[arg(long, default_value_t = 1.0)]
        end_lr: f64,
        #[arg(long, default_value_t = 40)]
        steps: usize,
        /// write the curve and the suggestion here as JSON
        #[arg(long)]
        report: Option<PathBuf>,
    },
    /// what a training run needs (parameters, memory part by part, a rough time) against this machine, before it runs
    Profile {
        #[command(flatten)]
        run: RunArgs,
        /// a known model's shape instead of --model (no files needed): llama-2-7b, mistral-7b, llama-3.1-8b, qwen2.5-7b, qwen2.5-0.5b
        #[arg(long)]
        preset: Option<String>,
        /// conversations, when no --data is given to count them
        #[arg(long)]
        examples: Option<usize>,
        /// mean tokens per conversation, when no --data is given
        #[arg(long)]
        avg_tokens: Option<usize>,
        /// this CPU's matrix-product speed, instead of measuring it
        #[arg(long)]
        cpu_gflops: Option<f64>,
    },
    /// settings that fit this machine, the data and a time budget, each with its reason, and the train command for them
    Autopilot {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        data: PathBuf,
        #[arg(long, default_value = "auto")]
        format: String,
        /// where the adapter would go (for the printed command)
        #[arg(long)]
        out: Option<PathBuf>,
        /// how long the run may take: 90m, 2h, 1d, or seconds
        #[arg(long, default_value = "8h")]
        time_budget: String,
        /// memory the run may use: 12GB, 800MB; the free RAM when not given
        #[arg(long)]
        ram_budget: Option<String>,
        #[arg(long)]
        cpu_gflops: Option<f64>,
        #[arg(long)]
        chat_template: Option<PathBuf>,
        #[arg(long)]
        threads: Option<usize>,
    },
}

/// The settings of a run, shared by train, find-lr and profile. A `--soup` file gives them all; a flag given beside
/// it wins. Without a Soup file, what is not given takes Soup's default (see `TrainSettings::new`).
#[derive(Args, Clone, Default)]
struct RunArgs {
    /// the base model: a Hugging Face folder (config.json, safetensors, tokenizer.json)
    #[arg(long)]
    model: Option<PathBuf>,
    /// chat JSONL: {"messages": [...]} rows (also alpaca, sharegpt, prompt/completion)
    #[arg(long)]
    data: Option<PathBuf>,
    /// the adapter's folder
    #[arg(long)]
    out: Option<PathBuf>,
    /// a Soup config (soup.yaml) to take the settings from; what it sets that is not supported is refused by name
    #[arg(long)]
    soup: Option<PathBuf>,
    /// with --soup: hold a `quantization: 4bit|8bit` base in bf16 and use AdamW for another optimizer, saying so
    #[arg(long)]
    soup_relax: bool,
    /// auto, chatml, alpaca or sharegpt
    #[arg(long)]
    format: Option<String>,
    #[arg(long)]
    eval_data: Option<PathBuf>,
    /// share of the data held out to measure on, when no --eval-data
    #[arg(long)]
    val_split: Option<f64>,
    #[arg(long)]
    rank: Option<usize>,
    #[arg(long)]
    alpha: Option<f64>,
    #[arg(long)]
    dropout: Option<f64>,
    /// rsLoRA scaling: alpha / sqrt(rank)
    #[arg(long)]
    rslora: bool,
    /// of q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
    #[arg(long)]
    targets: Option<String>,
    /// LoRA in the top N blocks only: the blocks below run without a backward pass
    #[arg(long)]
    train_layers: Option<usize>,
    /// with --train-layers: store the hidden state entering the first trained block here, once per conversation
    #[arg(long)]
    cache_activations: Option<PathBuf>,
    /// hold the frozen base as f32, f16 or bf16 (half the memory; LoRA and the loss stay f32)
    #[arg(long)]
    base_dtype: Option<String>,
    /// where to train: cpu (default) or GPUs, e.g. cuda:0,cuda:1 (the blocks split between them; needs a cuda build) or metal (Apple Silicon builds)
    #[arg(long, value_delimiter = ',')]
    devices: Vec<String>,
    /// recompute each trained block's activations in the backward pass (less memory, about a third more time)
    #[arg(long)]
    gradient_checkpointing: bool,
    #[arg(long)]
    lr: Option<f64>,
    #[arg(long)]
    epochs: Option<f64>,
    #[arg(long)]
    max_steps: Option<usize>,
    /// conversations per optimizer step (the batch is always 1)
    #[arg(long, alias = "grad-accum")]
    accumulation: Option<usize>,
    /// longer conversations are cut to this many tokens
    #[arg(long)]
    max_len: Option<usize>,
    #[arg(long)]
    warmup_ratio: Option<f64>,
    #[arg(long)]
    weight_decay: Option<f64>,
    /// 0 turns clipping off
    #[arg(long)]
    max_grad_norm: Option<f64>,
    #[arg(long)]
    seed: Option<u64>,
    #[arg(long)]
    save_steps: Option<usize>,
    #[arg(long)]
    eval_steps: Option<usize>,
    #[arg(long)]
    logging_steps: Option<usize>,
    /// continue from a checkpoint folder, or `auto` for the newest in --out
    #[arg(long)]
    resume: Option<String>,
    /// a chat template file, for a model without one
    #[arg(long)]
    chat_template: Option<PathBuf>,
    /// score every token, not only the assistant's
    #[arg(long)]
    score_everything: bool,
    /// CPU threads (all cores when not given)
    #[arg(long)]
    threads: Option<usize>,
    /// one device: cpu (default), cuda:N or metal; for several, --devices
    #[arg(long, default_value = "cpu")]
    device: String,
}

impl RunArgs {
    /// The settings these flags (and the Soup file) describe, and what to say about them.
    fn settings(&self, need_data: bool) -> Result<(TrainSettings, Vec<String>), String> {
        let (mut s, mut notes) = match &self.soup {
            Some(path) => {
                let imp = soup::import_file(path, self.soup_relax).map_err(|e| e.to_string())?;
                (imp.settings, imp.notes.into_iter().map(|n| format!("soup: {n}")).collect())
            }
            None => (TrainSettings::new(PathBuf::new(), PathBuf::new(), PathBuf::new()), Vec::new()),
        };
        let soup_given = self.soup.is_some();
        if let Some(m) = &self.model { s.model = m.clone(); }
        if let Some(d) = &self.data { s.data = d.clone(); }
        if let Some(o) = &self.out { s.out = o.clone(); }
        if s.model.as_os_str().is_empty() { return Err("--model: the base model's Hugging Face folder is needed".into()); }
        if need_data && s.data.as_os_str().is_empty() { return Err("--data: the training conversations (JSONL) are needed".into()); }
        if s.out.as_os_str().is_empty() { s.out = PathBuf::from("./output"); }
        if let Some(f) = &self.format { s.data_format = DataFormat::from_s(f).map_err(|e| e.to_string())?; }
        if let Some(e) = &self.eval_data { s.eval_data = Some(e.clone()); }
        if let Some(v) = self.val_split { s.val_split = v; }
        if !soup_given {
            // LoRA's usual settings when nothing says otherwise (Soup's own default, r 64 at 2e-5, suits its 4-bit GPU runs)
            s.lora.r = 16;
            s.lora.alpha = 32.0;
            s.lora.dropout = 0.05;
        }
        if let Some(r) = self.rank { s.lora.r = r; if self.alpha.is_none() && !soup_given { s.lora.alpha = 2.0 * r as f64; } }
        if let Some(a) = self.alpha { s.lora.alpha = a; }
        if let Some(d) = self.dropout { s.lora.dropout = d; }
        if self.rslora { s.lora.use_rslora = true; }
        if let Some(t) = &self.targets { s.lora.targets = Target::parse_list(t).map_err(|e| e.to_string())?; }
        if let Some(n) = self.train_layers { s.lora.train_layers = Some(n); }
        if let Some(c) = &self.cache_activations { s.cache_dir = Some(c.clone()); }
        if let Some(d) = &self.base_dtype { s.base_dtype = BaseDtype::from_s(d).map_err(|e| e.to_string())?; }
        s.devices = if self.devices.is_empty() && self.device != "cpu" { vec![self.device.clone()] } else { self.devices.clone() };
        s.gradient_checkpointing = self.gradient_checkpointing;
        if let Some(v) = self.lr { s.lr = v; }
        if let Some(v) = self.epochs { s.epochs = v; }
        if let Some(v) = self.max_steps { s.max_steps = Some(v); }
        if let Some(v) = self.accumulation { s.accumulation = v; }
        if let Some(v) = self.max_len { s.max_len = v; }
        if let Some(v) = self.warmup_ratio { s.warmup_ratio = v; }
        if let Some(v) = self.weight_decay { s.weight_decay = v; }
        if let Some(v) = self.max_grad_norm { s.max_grad_norm = v; }
        if let Some(v) = self.seed { s.seed = v; s.lora.seed = v; }
        if let Some(v) = self.save_steps { s.save_steps = v; }
        if let Some(v) = self.eval_steps { s.eval_steps = v; }
        if let Some(v) = self.logging_steps { s.logging_steps = v; }
        if let Some(r) = &self.resume { s.resume = Some(if r == "auto" { Resume::Auto } else { Resume::Dir(PathBuf::from(r)) }); }
        if let Some(c) = &self.chat_template { s.chat_template = Some(c.clone()); }
        if self.score_everything { s.score_everything = true; }
        if s.cache_dir.is_some() && s.lora.train_layers.is_none() {
            notes.push("--cache-activations needs --train-layers N: with every block training there is no frozen part to cache".into());
        }
        s.validate().map_err(|e| e.to_string())?;
        Ok((s, notes))
    }
}

/// `90m`, `2h`, `1d`, `45s` or plain seconds.
fn parse_duration(s: &str) -> Result<f64, String> {
    let t = s.trim().to_ascii_lowercase();
    let (num, unit) = match t.char_indices().find(|(_, c)| c.is_ascii_alphabetic()) {
        Some((i, _)) => (&t[..i], &t[i..]),
        None => (t.as_str(), "s"),
    };
    let n: f64 = num.trim().parse().map_err(|_| format!("{s:?} is not a duration (90m, 2h, 1d, or seconds)"))?;
    Ok(n * match unit { "s" | "sec" => 1.0, "m" | "min" => 60.0, "h" => 3600.0, "d" => 86400.0, _ => return Err(format!("{s:?}: use s, m, h or d")) })
}

/// `12GB`, `800MB`, or bytes.
fn parse_bytes(s: &str) -> Result<u64, String> {
    let t = s.trim().to_ascii_lowercase();
    let (num, unit) = match t.char_indices().find(|(_, c)| c.is_ascii_alphabetic()) {
        Some((i, _)) => (&t[..i], &t[i..]),
        None => (t.as_str(), "b"),
    };
    let n: f64 = num.trim().parse().map_err(|_| format!("{s:?} is not a size (12GB, 800MB)"))?;
    Ok((n * match unit { "b" => 1.0, "kb" | "k" => 1e3, "mb" | "m" => 1e6, "gb" | "g" => 1e9, "gib" => (1u64 << 30) as f64, "mib" => (1u64 << 20) as f64, _ => return Err(format!("{s:?}: use MB or GB")) }) as u64)
}

/// Conversations, mean and 95th-percentile token counts of a data file under the model's tokenizer and template.
fn data_stats(s: &TrainSettings) -> Result<(usize, usize, usize), String> {
    let tok = ChatTokenizer::load(&s.model, s.chat_template.as_deref()).map_err(|e| e.to_string())?;
    let (records, _) = chatdata::read_jsonl(&s.data, s.data_format).map_err(|e| e.to_string())?;
    let mut counts: Vec<usize> = records.iter().filter_map(|r| tok.count(r).ok()).collect();
    if counts.is_empty() {
        return Err(format!("{}: no usable conversation", s.data.display()));
    }
    counts.sort_unstable();
    let p95 = counts[((counts.len() as f64 * 0.95).ceil() as usize).clamp(1, counts.len()) - 1];
    Ok((counts.len(), counts.iter().sum::<usize>() / counts.len(), p95))
}

/// The estimate's input for these settings: the data's size from the file when there is one.
fn profile_input(s: &TrainSettings, cfg: ModelConfig, examples: Option<usize>, avg_tokens: Option<usize>) -> Result<ProfileInput, String> {
    let (n, avg) = if s.data.as_os_str().is_empty() {
        (examples.unwrap_or(1000), avg_tokens.unwrap_or(s.max_len / 2))
    } else {
        let (n, avg, _) = data_stats(s)?;
        // what is held out is not trained on
        let held = if s.eval_data.is_none() && s.val_split > 0.0 && n >= 2 { ((n as f64 * s.val_split).round() as usize).clamp(1, n - 1) } else { 0 };
        (examples.unwrap_or(n - held), avg_tokens.unwrap_or(avg))
    };
    let max_len = s.max_len.min(cfg.max_position_embeddings.max(8));
    Ok(ProfileInput { checkpointing: s.gradient_checkpointing, devices: s.devices.len().max(1), lora: s.lora.clone(), base_dtype: s.base_dtype, max_len, accumulation: s.accumulation,
                      cache: s.cache_dir.is_some() && s.lora.first_trained(cfg.num_hidden_layers) > 0, examples: n, epochs: s.epochs, avg_tokens: avg.min(max_len), cfg })
}

fn os_threads() -> Option<u64> {
    std::fs::read_to_string("/proc/self/status").ok()?.lines().find(|l| l.starts_with("Threads:"))?.split_whitespace().nth(1)?.parse().ok()
}

/// set by SIGTERM (a cancel from the server): datagen stops after the current request and writes what it made
static TERM: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

#[cfg(unix)]
extern "C" fn on_term(_: libc::c_int) {
    TERM.store(true, std::sync::atomic::Ordering::SeqCst);
}

struct Out {
    json: bool,
}

impl Out {
    fn event(&self, v: Value) {
        if self.json {
            println!("{v}");
            return;
        }
        match v["event"].as_str() {
            Some("start") => println!("{} …", v["what"].as_str().unwrap_or("")),
            Some("log") => println!("  {}", v["message"].as_str().unwrap_or("")),
            Some("progress") => println!("  [{}/{}] {}", v["done"], v["total"], v["message"].as_str().unwrap_or("")),
            Some("done") => println!("done in {:.1} s: {} ({})", v["seconds"].as_f64().unwrap_or(0.0),
                                     v["output"].as_str().unwrap_or(""), human(v["bytes"].as_u64().unwrap_or(0))),
            Some("error") => eprintln!("error: {}", v["message"].as_str().unwrap_or("")),
            Some("info") => println!("{}", serde_json::to_string_pretty(&v["info"]).unwrap_or_default()),
            Some("layers") => {
                println!("  {}", v["message"].as_str().unwrap_or(""));
                println!("{}", serde_json::to_string(&v["layers"]).unwrap_or_default());
            }
            Some("check") => {
                let r = &v["result"];
                let pct = |k: &str| format!("{:.0}%", r[k].as_f64().unwrap_or(0.0) * 100.0);
                println!("  {}: refused {} of {} held-out harmful prompts and {} of the harmless; answer loss {}", v["which"].as_str().unwrap_or(""),
                         pct("harmful_refusal_rate"), r["n"], pct("harmless_refusal_rate"), r["harmless_loss"].as_f64().map_or("-".into(), |l| format!("{l:.3}")));
            }
            Some("rule") => println!("  {}", v["message"].as_str().unwrap_or("")),
            Some("profile") | Some("plan") | Some("find-lr") | Some("library") => {
                for l in v["lines"].as_array().into_iter().flatten() {
                    println!("{}", l.as_str().unwrap_or(""));
                }
            }
            _ => println!("{v}"),
        }
    }
}

fn human(b: u64) -> String {
    let b = b as f64;
    if b >= 1e9 { format!("{:.2} GB", b / 1e9) } else if b >= 1e6 { format!("{:.1} MB", b / 1e6) } else { format!("{:.0} KB", b / 1e3) }
}

/// 57.2 B, 8.19 B, 412 M: a parameter count as people say it.
fn params_label(n: u64) -> String {
    if n >= 1_000_000_000 { format!("{:.2} B", n as f64 / 1e9) } else if n >= 1_000_000 { format!("{:.1} M", n as f64 / 1e6) } else { format!("{n}") }
}

/// Free bytes on the file system that `p` (or its folder) is on.
fn disk_free(p: &Path) -> Option<u64> {
    #[cfg(unix)]
    {
        let dir = p.parent().filter(|d| !d.as_os_str().is_empty()).unwrap_or(Path::new("."));
        let dir = std::iter::successors(Some(dir), |d| d.parent()).find(|d| d.exists())?;
        let c = std::ffi::CString::new(dir.as_os_str().as_encoded_bytes()).ok()?;
        let mut s: libc::statvfs = unsafe { std::mem::zeroed() };
        if unsafe { libc::statvfs(c.as_ptr(), &mut s) } != 0 {
            return None;
        }
        Some(s.f_bavail as u64 * s.f_frsize as u64)
    }
    #[cfg(not(unix))]
    {
        let _ = p;
        None
    }
}

fn size_of(p: &Path) -> u64 {
    if p.is_dir() {
        std::fs::read_dir(p).into_iter().flatten().flatten().map(|e| size_of(&e.path())).sum()
    } else {
        std::fs::metadata(p).map(|m| m.len()).unwrap_or(0)
    }
}

/// Free memory (MemAvailable) in bytes, when the OS says.
fn mem_available() -> Option<u64> {
    let s = std::fs::read_to_string("/proc/meminfo").ok()?;
    let kb: u64 = s.lines().find(|l| l.starts_with("MemAvailable:"))?.split_whitespace().nth(1)?.parse().ok()?;
    Some(kb * 1024)
}

/// The f32 working size of a model folder's safetensors (the converter holds them as f32 today).
fn f32_bytes_of(dir: &Path) -> u64 {
    let mut n = 0u64;
    for e in std::fs::read_dir(dir).into_iter().flatten().flatten() {
        let p = e.path();
        if p.extension().and_then(|x| x.to_str()) == Some("safetensors") {
            if let Ok(s) = strata_workshop::safetensors_io::Shard::open(&p) {
                for t in s.tensors() {
                    n += t.shape().iter().product::<usize>() as u64 * 4;
                }
            }
        }
    }
    n
}

/// Refuse a job whose working set will not fit in free memory, before it starts (not halfway, with the PC swapping).
fn fits(need: u64, what: &str) -> Result<(), String> {
    match mem_available() {
        Some(free) if need + (2 << 30) > free => Err(format!(
            "{what} needs about {} of memory and {} is free; free memory first (unload the text model from Settings › Models) or use a smaller model",
            human(need), human(free))),
        _ => Ok(()),
    }
}

fn info(path: &Path) -> Result<Value, String> {
    if path.is_dir() {
        let cfg: Value = std::fs::read_to_string(path.join("config.json")).ok().and_then(|t| serde_json::from_str(&t).ok())
            .ok_or_else(|| format!("{}: no config.json (not a Hugging Face model folder)", path.display()))?;
        let adapter = path.join("adapter_config.json").exists();
        return Ok(json!({
            "kind": if adapter { "lora-adapter" } else { "hf-model" },
            "architectures": cfg["architectures"], "model_type": cfg["model_type"],
            "hidden_size": cfg["hidden_size"], "layers": cfg["num_hidden_layers"], "heads": cfg["num_attention_heads"],
            "kv_heads": cfg["num_key_value_heads"], "vocab_size": cfg["vocab_size"],
            "f32_bytes": f32_bytes_of(path), "bytes": size_of(path),
        }));
    }
    let g = strata_gguf::Gguf::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut types: std::collections::BTreeMap<String, usize> = Default::default();
    for t in &g.tensors {
        *types.entry(strata_gguf::type_name(t.ggml_type)).or_default() += 1;
    }
    let m = &g.metadata;
    let arch = m.get("general.architecture").and_then(|v| v.as_str()).unwrap_or("").to_string();
    Ok(json!({
        "kind": "gguf", "architecture": arch, "name": m.get("general.name"), "file_type": m.get("general.file_type"),
        "tensors": g.tensors.len(), "tensor_types": types, "bytes": size_of(path),
        "context_length": m.get(&format!("{arch}.context_length")), "block_count": m.get(&format!("{arch}.block_count")),
        "tokenizer": m.get("tokenizer.ggml.model"),
    }))
}

/// What a training event looks like on the wire: progress with step/total/loss/lr/eta, the rest as log lines.
fn training_event(out: &Out, e: Event) {
    match e {
        Event::Log(m) => out.event(json!({"event": "log", "message": m})),
        Event::Step { step, total, loss, lr, eta_seconds, aux } => out.event(json!({
            "event": "progress", "done": step, "total": total, "step": step, "loss": loss, "lr": lr, "eta": eta_seconds, "aux_loss": aux,
            "message": format!("loss {loss:.4}{} · rate {lr:.3e} · {} left", aux.map_or(String::new(), |a| format!(" (router balance {a:.3})")), profile::human_seconds(eta_seconds))})),
        Event::Eval { step, loss } => out.event(json!({"event": "log", "message": format!("held-out loss {loss:.4} at step {step}"), "step": step, "eval_loss": loss})),
        Event::Checkpoint { step, dir } => out.event(json!({"event": "log", "message": format!("checkpoint at step {step}: {}", dir.display()), "step": step, "checkpoint": dir.display().to_string()})),
        Event::Cache { done, total } => out.event(json!({"event": "progress", "done": done, "total": total, "message": "filling the activation cache"})),
    }
}

/// The judge for a suite's judge tests: the server's model (`--judge-model`, the suite's `judge_model`, or the loaded
/// one), when the suite has such tests and the server answers; otherwise None (those tests then report why).
fn judge_for(s: &strata_workshop::regress::Suite, want: Option<&str>, out: &Out) -> Option<datagen::Model> {
    if !s.tests.iter().any(|t| t.kind == "judge") && want.is_none() {
        return None;
    }
    match datagen::Model::from_env_with(want.or(s.judge_model.as_deref())) {
        Ok(m) => {
            out.event(json!({"event": "log", "message": format!("judging with {} through the server's API", m.name)}));
            Some(m)
        }
        Err(e) => {
            out.event(json!({"event": "log", "message": format!("no judge ({e:#}): the judge tests will report an error")}));
            None
        }
    }
}

fn install_term_handler() {
    // a cancel (SIGTERM from the server) is noticed after the current step: a checkpoint is written, then the run ends
    #[cfg(unix)]
    unsafe { libc::signal(libc::SIGTERM, on_term as extern "C" fn(libc::c_int) as libc::sighandler_t); }
}

fn run(cmd: &Cmd, out: &Out) -> Result<(PathBuf, String, Value), String> {
    let plain = |r: Result<(PathBuf, String), String>| r.map(|(p, k)| (p, k, json!({})));
    match cmd {
        Cmd::Train { run, force } => {
            let (s, notes) = run.settings(true)?;
            let cfg = ModelConfig::load(&s.model).map_err(|e| format!("{e:#}"))?;
            out.event(json!({"event": "start", "what": format!("train a LoRA adapter on {} with {} into {}", s.model.display(), s.data.display(), s.out.display()),
                             "threads": run.threads, "device": if s.devices.is_empty() { "cpu".to_string() } else { s.devices.join(",") }, "base_dtype": s.base_dtype.as_str()}));
            for n in notes {
                out.event(json!({"event": "log", "message": n}));
            }
            out.event(json!({"event": "log", "message": format!("the base is held in {}; a 4-bit base (QLoRA) is not done in this trainer", s.base_dtype.as_str())}));
            // the estimate first: a run that will not fit is refused before it loads anything
            let p = profile_input(&s, cfg, None, None)?;
            let est = profile::estimate(&p);
            let machine = Machine::detect();
            for l in profile::report_lines(&p, &est, &machine).into_iter().skip(1).take(4) {
                out.event(json!({"event": "log", "message": l}));
            }
            let v = profile::verdict(&est, &machine);
            if v.fits_ram == Some(false) {
                let msg = format!("this run needs about {} of memory and {} is free", profile::human_bytes(v.need_bytes), profile::human_bytes(machine.ram_available));
                if !*force {
                    return Err(format!("{msg}: `am2strata-workshop autopilot` picks settings that fit (a 16-bit base, fewer trained blocks, a shorter sequence); --force runs it anyway"));
                }
                out.event(json!({"event": "log", "message": format!("{msg}; running anyway (--force)")}));
            }
            install_term_handler();
            let report = finetune::train(&s, &mut |e| training_event(out, e), &|| TERM.load(std::sync::atomic::Ordering::SeqCst)).map_err(|e| format!("{e:#}"))?;
            let extra = json!({
                "steps": report.steps_done, "total_steps": report.total_steps, "stopped": report.stopped, "final_loss": report.losses.last(),
                "first_loss": report.losses.first(), "eval_loss": report.eval_loss, "trainable_params": report.trainable_params,
                "tokens_seen": report.tokens_seen, "peak_rss_bytes": report.peak_rss_bytes, "os_threads": os_threads(),
                "estimated_bytes": est.total_bytes,
            });
            if report.stopped {
                out.event(json!({"event": "log", "message": format!("stopped at step {} of {}: continue with --resume auto", report.steps_done, report.total_steps)}));
                return Ok((s.out.join(format!("checkpoint-{}", report.steps_done)), "checkpoint".into(), extra));
            }
            Ok((s.out, "lora-adapter".into(), extra))
        }
        Cmd::FindLr { run, start_lr, end_lr, steps, report } => {
            let (s, notes) = run.settings(true)?;
            out.event(json!({"event": "start", "what": format!("learning-rate range test on {} with {} ({steps} steps, {start_lr:e} to {end_lr:e})", s.model.display(), s.data.display())}));
            for n in notes {
                out.event(json!({"event": "log", "message": n}));
            }
            let r = finetune::find_lr(&s, FindLr { start: *start_lr, end: *end_lr, steps: *steps }, &mut |e| training_event(out, e)).map_err(|e| format!("{e:#}"))?;
            let sg = &r.suggestion;
            let mut lines = vec![match sg.suggested {
                Some(lr) => format!("suggested learning rate: {lr:.2e} (where the loss was falling fastest)"),
                None => "no suggestion: the loss never fell in this range — try a lower --start-lr or more --steps".to_string(),
            }];
            lines.push(format!("lowest loss at {:.2e} (too high to train at: the loss turns there){}", sg.min_loss_lr,
                               sg.diverged_at.map(|d| format!("; the loss blew up from {d:.2e}")).unwrap_or_default()));
            if r.diverged {
                lines.push(format!("stopped after {} of {steps} steps: the loss diverged, which ends a range test", r.lrs.len()));
            }
            let v = json!({"event": "find-lr", "suggested_lr": sg.suggested, "min_loss_lr": sg.min_loss_lr, "diverged_at": sg.diverged_at, "diverged": r.diverged,
                           "lrs": r.lrs, "losses": r.losses, "smoothed": sg.smoothed, "lines": lines});
            if let Some(p) = report {
                std::fs::write(p, serde_json::to_string_pretty(&v).unwrap_or_default()).map_err(|e| format!("{}: {e}", p.display()))?;
            }
            out.event(v);
            Ok((s.model, "info".into(), json!({})))
        }
        Cmd::Profile { run, preset, examples, avg_tokens, cpu_gflops } => {
            let mut run = run.clone();
            let cfg = match preset {
                Some(name) => {
                    run.model.get_or_insert_with(|| PathBuf::from(name));
                    profile::preset(name).ok_or_else(|| format!("no preset {name:?}: {}", profile::PRESETS.join(", ")))?
                }
                None => ModelConfig::load(run.model.as_ref().ok_or("--model DIR or --preset NAME is needed")?).map_err(|e| format!("{e:#}"))?,
            };
            if preset.is_some() && run.data.is_some() {
                return Err("--preset has no tokenizer to count --data with: give --examples and --avg-tokens".into());
            }
            let (s, notes) = run.settings(false)?;
            let p = profile_input(&s, cfg, *examples, *avg_tokens)?;
            let est = profile::estimate(&p);
            let mut machine = Machine::detect();
            if let Some(g) = cpu_gflops {
                machine.cpu_gflops = *g;
            }
            let mut lines = profile::report_lines(&p, &est, &machine);
            lines.extend(notes);
            lines.push("a 4-bit base (QLoRA), expert adapters of mixture-of-experts models, and vision towers are not done; several GPUs need a cuda build (--devices)".into());
            let mut v = profile::estimate_json(&p, &est, &machine);
            v["event"] = json!("profile");
            v["lines"] = json!(lines);
            out.event(v);
            Ok((s.model, "info".into(), json!({})))
        }
        Cmd::Autopilot { model, data, format, out: dst, time_budget, ram_budget, cpu_gflops, chat_template, threads: _ } => {
            let cfg = ModelConfig::load(model).map_err(|e| format!("{e:#}"))?;
            let mut s = TrainSettings::new(model, data, dst.clone().unwrap_or_else(|| PathBuf::from("./output")));
            s.data_format = DataFormat::from_s(format).map_err(|e| e.to_string())?;
            s.chat_template = chat_template.clone();
            let (n, avg, p95) = data_stats(&s)?;
            let machine = Machine::detect();
            let budget = match ram_budget {
                Some(b) => parse_bytes(b)?,
                None if machine.ram_available > 0 => machine.ram_available,
                None => return Err("this system does not say how much RAM is free: pass --ram-budget (for example 12GB)".into()),
            };
            let held = if n >= 2 { ((n as f64 * s.val_split).round() as usize).clamp(1, n - 1) } else { 0 };
            let plan = profile::autopilot(&AutopilotInput {
                cfg, source_dtype: profile::source_dtype(model).map_err(|e| format!("{e:#}"))?, examples: n - held, p95_tokens: p95, avg_tokens: avg, budget_bytes: budget,
                time_budget_seconds: parse_duration(time_budget)?, cpu_gflops: cpu_gflops.unwrap_or(machine.cpu_gflops),
            }).map_err(|e| e.to_string())?;
            let mut command = format!("am2strata-workshop train --model {} --data {} --out {} --rank {} --alpha {} --dropout {} --targets {} --base-dtype {} --max-len {} --accumulation {} --epochs {} --lr {:e}",
                model.display(), data.display(), s.out.display(), plan.lora.r, plan.lora.alpha, plan.lora.dropout,
                plan.lora.targets.iter().map(|t| t.as_str()).collect::<Vec<_>>().join(","), plan.base_dtype.as_str(), plan.max_len, plan.accumulation, plan.epochs, plan.lr);
            if let Some(nl) = plan.lora.train_layers {
                command.push_str(&format!(" --train-layers {nl}"));
                if plan.cache {
                    command.push_str(&format!(" --cache-activations {}", s.out.join("activations").display()));
                }
            }
            let mut lines = vec![format!("{} conversations to train on ({held} held out), {avg} tokens on average, 95 % within {p95}; {} of memory and {} to work with",
                                         n - held, profile::human_bytes(budget), profile::human_seconds(parse_duration(time_budget)?))];
            lines.extend(plan.choices.iter().map(|c| format!("{:<17}{:<16}{}", c.setting, c.value, c.why)));
            lines.push(format!("estimate: {} of memory with headroom, about {} on this CPU, {} steps",
                               profile::human_bytes(profile::need_bytes(&plan.estimate)), profile::human_seconds(plan.seconds), plan.estimate.steps));
            if !plan.fits_memory {
                lines.push(format!("THIS DOES NOT FIT the memory: even the smallest plan needs {}; a smaller model is the way (a 4-bit base, QLoRA, is not done)", profile::human_bytes(profile::need_bytes(&plan.estimate))));
            }
            if !plan.fits_time {
                lines.push(format!("THIS DOES NOT FIT the time: the shortest plan takes about {}; fewer conversations, or a GPU run when that lands", profile::human_seconds(plan.seconds)));
            }
            lines.push(command.clone());
            out.event(json!({"event": "plan", "fits_memory": plan.fits_memory, "fits_time": plan.fits_time, "command": command, "lines": lines,
                "settings": {"rank": plan.lora.r, "alpha": plan.lora.alpha, "dropout": plan.lora.dropout, "targets": plan.lora.targets.iter().map(|t| t.as_str()).collect::<Vec<_>>(),
                             "train_layers": plan.lora.train_layers, "base_dtype": plan.base_dtype.as_str(), "max_len": plan.max_len, "accumulation": plan.accumulation,
                             "epochs": plan.epochs, "lr": plan.lr, "cache_activations": plan.cache},
                "choices": plan.choices.iter().map(|c| json!({"setting": c.setting, "value": c.value, "why": c.why})).collect::<Vec<_>>(),
                "estimate": {"bytes": plan.estimate.total_bytes, "needed_with_headroom": profile::need_bytes(&plan.estimate), "seconds": plan.seconds, "steps": plan.estimate.steps,
                             "trainable_parameters": plan.estimate.trainable_params, "cache_disk_bytes": plan.estimate.cache_disk_bytes}}));
            Ok((model.clone(), "info".into(), json!({})))
        }
        Cmd::Abliterate { model, harmful, harmless, layer, scale, check, devices, base_dtype, out: dst } => {
            use strata_workshop::abliterate;
            out.event(json!({"event": "start", "what": format!("abliterate {} into {}", model.display(), dst.display())}));
            let read = |p: &Path| abliterate::read_prompts(p).map_err(|e| format!("{e:#}"));
            let mut o = abliterate::Options::new(model, read(harmful)?, read(harmless)?, dst);
            o.layer = abliterate::LayerChoice::parse(layer).map_err(|e| e.to_string())?;
            o.scale = *scale;
            o.check = *check;
            o.devices = devices.clone();
            o.base_dtype = base_dtype.as_deref().map(BaseDtype::from_s).transpose().map_err(|e| e.to_string())?;
            if let Ok(cfg) = ModelConfig::load(model) {
                // the model while reading, and the largest tensor while writing
                let held = cfg.total_params() * if o.base_dtype == Some(BaseDtype::F32) { 4 } else { 2 };
                fits(held + (cfg.vocab_size * cfg.hidden_size) as u64 * 4, "abliteration")?;
            }
            let r = abliterate::run(&o, &mut |v| out.event(v)).map_err(|e| format!("{e:#}"))?;
            let extra = json!({"layer": r.chosen.site.block, "pass": r.chosen.site.pass, "separation": r.chosen.separation,
                               "changed": r.ortho.changed.len(), "untied": r.ortho.untied, "private": true,
                               "before": r.before.as_ref().map(|c| c.to_json()), "after": r.after.as_ref().map(|c| c.to_json())});
            Ok((dst.clone(), "hf-model".into(), extra))
        }
        Cmd::Upcycle { model, out: dst, experts, top_k } => {
            use strata_workshop::model::{make_upcycled, MoeSpec};
            out.event(json!({"event": "start", "what": format!("upcycle {} to {experts} experts, top-{top_k}, as {}", model.display(), dst.display())}));
            let m = MoeSpec { experts: *experts, top_k: *top_k };
            let (total, active) = make_upcycled(model, dst, m).map_err(|e| format!("{e:#}"))?;
            let cfg = ModelConfig::load(dst).map_err(|e| format!("{e:#}"))?;
            out.event(json!({"event": "log", "message": format!("{experts} experts x top-{top_k}: {} parameters in all, {} active per token (the dense base has {})",
                                                                 params_label(total), params_label(active), params_label(cfg.total_params()))}));
            let sizes: serde_json::Map<String, Value> = ["q4_k", "q8_0", "f16"].iter().filter_map(|t| {
                let ty = strata_workshop::export_moe::ExportType::from_s(t).ok()?;
                Some((t.to_string(), json!(strata_workshop::export_moe::export_bytes(&cfg, ty).ok()?)))
            }).collect();
            out.event(json!({"event": "log", "message": format!("export sizes: q4_k {}, q8_0 {}, f16 {}",
                human(sizes["q4_k"].as_u64().unwrap_or(0)), human(sizes["q8_0"].as_u64().unwrap_or(0)), human(sizes["f16"].as_u64().unwrap_or(0)))}));
            Ok((dst.clone(), "hf-model".into(), json!({"experts": experts, "top_k": top_k, "total_parameters": total, "active_parameters": active,
                                                         "dense_parameters": cfg.total_params(), "export_bytes": sizes})))
        }
        Cmd::ExportMoe { model, adapter, out: dst, ty, pre_tokenizer } => {
            use strata_workshop::export_moe::{export_bytes, export_moe, ExportType};
            let t = ExportType::from_s(ty).map_err(|e| e.to_string())?;
            let cfg = ModelConfig::load(model).map_err(|e| format!("{e:#}"))?;
            let need = export_bytes(&cfg, t).map_err(|e| format!("{}: {e}", model.display()))?;
            if let Some(free) = disk_free(dst) {
                if free < need + (1 << 30) {
                    return Err(format!("the export needs about {} on disk and {} is free there", human(need), human(free)));
                }
            }
            // one feed-forward matrix as f32, a few copies, and the adapter
            fits(4 * 4 * (cfg.hidden_size * cfg.intermediate_size.max(cfg.vocab_size)) as u64 + adapter.as_ref().map_or(0, |a| size_of(a)), "exporting this model")?;
            out.event(json!({"event": "start", "what": format!("export {}{} as a qwen3moe GGUF ({ty}, about {}) to {}", model.display(),
                adapter.as_ref().map_or(String::new(), |a| format!(" with {}", a.display())), human(need), dst.display())}));
            let r = export_moe(model, adapter.as_deref(), dst, t, pre_tokenizer.as_deref()).map_err(|e| format!("{e:#}"))?;
            out.event(json!({"event": "log", "message": format!("{} experts x top-{}: {} parameters, {} active per token; {} of {} blocks carry trained experts",
                r.moe.experts, r.moe.top_k, params_label(r.total_params), params_label(r.active_params), r.trained_blocks, cfg.num_hidden_layers)}));
            Ok((r.out, "gguf".into(), json!({"experts": r.moe.experts, "top_k": r.moe.top_k, "total_parameters": r.total_params,
                                              "active_parameters": r.active_params, "trained_blocks": r.trained_blocks})))
        }
        Cmd::Regress { model, adapter, passes, suite, out: dst, against, devices, base_dtype, max_prompt_tokens, judge_model } => {
            use strata_workshop::regress;
            out.event(json!({"event": "start", "what": format!("the suite {} on {}{} into {}", suite.display(), model.display(),
                adapter.as_ref().map_or(String::new(), |a| format!(" with {}", a.display())), dst.display())}));
            let mut o = regress::Options::new(model, suite, dst);
            o.adapter = adapter.clone();
            o.passes = passes.clone();
            o.against = against.clone();
            o.devices = devices.clone();
            o.base_dtype = base_dtype.as_deref().map(BaseDtype::from_s).transpose().map_err(|e| e.to_string())?;
            o.max_prompt_tokens = *max_prompt_tokens;
            let s = regress::Suite::load(suite).map_err(|e| format!("{e:#}"))?;
            let judge = judge_for(&s, judge_model.as_deref(), out);
            install_term_handler();
            let r = regress::run(&o, judge.as_ref().map(|j| j as &dyn regress::Judge), &mut |v| out.event(v), &|| TERM.load(std::sync::atomic::Ordering::SeqCst))
                .map_err(|e| format!("{e:#}"))?;
            if !out.json {
                print!("{}", regress::report_markdown(&r.json));
            }
            // the report card without its per-item scores (those stay in report.json), for the job's result
            let mut card = r.json.clone();
            for t in card["results"].as_array_mut().into_iter().flatten() {
                if let Some(o) = t.as_object_mut() { o.remove("items"); }
            }
            Ok((dst.join("report.json"), "report".into(), json!({"regressions": r.json["regressions"], "results": r.results.len(), "report": card,
                "summary": r.results.iter().map(|t| json!({"test": t.test, "passes": t.passes, "metric": t.metric, "value": t.value, "n": t.items.len(), "error": t.error})).collect::<Vec<_>>()})))
        }
        Cmd::GradeTrain { model, adapter, data, samples, grader, rounds, suite, out: dst, temperature, top_p, max_tokens, min_score, seed, passes, devices, base_dtype,
                          lr, epochs, max_steps, rank, train_layers, max_len, accumulation, max_prompt_tokens, judge_model } => {
            use strata_workshop::{gradetrain, regress};
            let code = match grader.as_str() { "code" => true, "judge" => false, g => return Err(format!("--grader {g:?}: code or judge")) };
            out.event(json!({"event": "start", "what": format!("grade and train {} on {} ({rounds} round(s), {samples} samples each, graded by {grader}) into {}",
                                                               model.display(), data.display(), dst.display())}));
            let prompts = gradetrain::load_prompts(data, code).map_err(|e| format!("{e:#}"))?;
            let mut o = gradetrain::Options::new(model, prompts, suite, dst);
            o.adapter = adapter.clone();
            o.samples = *samples;
            o.rounds = *rounds;
            o.temperature = *temperature;
            o.top_p = *top_p;
            o.max_tokens = *max_tokens;
            o.seed = *seed;
            o.passes = passes.clone();
            o.devices = devices.clone();
            o.base_dtype = base_dtype.as_deref().map(BaseDtype::from_s).transpose().map_err(|e| e.to_string())?;
            o.max_prompt_tokens = *max_prompt_tokens;
            o.train.lr = *lr;
            o.train.epochs = *epochs;
            o.train.max_steps = *max_steps;
            o.train.lora.r = *rank;
            o.train.lora.alpha = 2.0 * *rank as f64;
            o.train.lora.train_layers = *train_layers;
            o.train.max_len = *max_len;
            o.train.accumulation = *accumulation;
            o.train.validate().map_err(|e| e.to_string())?;
            let s = regress::Suite::load(suite).map_err(|e| format!("{e:#}"))?;
            let judge = judge_for(&s, judge_model.as_deref(), out);
            let judge_ref = judge.as_ref().map(|j| j as &dyn regress::Judge);
            install_term_handler();
            let stop = || TERM.load(std::sync::atomic::Ordering::SeqCst);
            let r = if code {
                gradetrain::run(&o, &gradetrain::CodeGrader(regress::CodeRunner::detect()), judge_ref, &mut |v| out.event(v), &stop)
            } else {
                let j = judge_ref.ok_or("--grader judge: the server's API is not reachable (AM2STRATA_URL, AM2STRATA_API_KEY)")?;
                gradetrain::run(&o, &gradetrain::JudgeGrader { judge: j, min: *min_score as f64 }, judge_ref, &mut |v| out.event(v), &stop)
            }.map_err(|e| format!("{e:#}"))?;
            Ok((dst.clone(), "grade-train".into(), json!({"promoted": r.promoted.as_ref().map(|p| p.display().to_string()),
                "rounds": r.rounds.iter().map(|x| json!({"round": x.round, "sampled": x.sampled, "passing": x.passing, "kept": x.kept,
                    "regressions": x.regressions, "promoted": x.promoted, "note": x.note})).collect::<Vec<_>>()})))
        }
        other => plain(run_tools(other, out)),
    }
}

fn run_tools(cmd: &Cmd, out: &Out) -> Result<(PathBuf, String), String> {
    match cmd {
        Cmd::Library { action, name, ability, task, origin, query, trains, out: dst, catalog, archs, commit } => {
            use strata_workshop::library::{self, Ability, Query, Status};
            let e = |s: String| s;
            match action.as_str() {
                "list" => {
                    let q = Query { text: query.clone(), task: task.clone(), origin: origin.clone(), trains_only: *trains,
                                    ability: ability.as_deref().map(Ability::from_s).transpose().map_err(|x| x.to_string())? };
                    let found = library::search(&q);
                    let mut lines: Vec<String> = found.iter().map(|r| format!("{:<34} {:<5} {:<8} {:<9} {:<38} {}",
                        r.name, r.size, r.task, if r.status() == Status::Trains { "trains" } else { "not yet" },
                        r.abilities().iter().map(|a| a.as_str()).collect::<Vec<_>>().join(","), r.model)).collect();
                    let trains_n = found.iter().filter(|r| r.status() == Status::Trains).count();
                    lines.insert(0, format!("{:<34} {:<5} {:<8} {:<9} {:<38} {}", "RECIPE", "SIZE", "TASK", "WORKSHOP", "ABILITIES", "BASE MODEL"));
                    lines.push(format!("{} recipes ({trains_n} train in this Workshop now); `library show <name>` for one",
                                       found.len()));
                    out.event(json!({"event": "library", "lines": lines, "recipes": found.iter().map(|r| r.summary()).collect::<Vec<_>>(),
                                     "sources": library::sources()}));
                    Ok((PathBuf::new(), "info".into()))
                }
                "show" => {
                    let n = name.as_deref().ok_or("library show <name>")?;
                    let r = library::get(n).ok_or_else(|| format!("no recipe {n:?}: `library list --query ...` finds one"))?;
                    let mut lines: Vec<String> = r.data_guide().lines().map(str::to_string).collect();
                    lines.push(String::new());
                    lines.push("## soup.yaml".into());
                    lines.push(String::new());
                    lines.extend(r.yaml.lines().map(str::to_string));
                    out.event(json!({"event": "library", "lines": lines, "recipe": r.summary()}));
                    Ok((PathBuf::new(), "info".into()))
                }
                "init" => {
                    let n = name.as_deref().ok_or("library init <name> --out <folder>")?;
                    let dir = dst.as_ref().ok_or("library init <name> --out <folder>")?;
                    let r = library::get(n).ok_or_else(|| format!("no recipe {n:?}"))?;
                    let files = library::init(&r, dir).map_err(|x| e(x.to_string()))?;
                    out.event(json!({"event": "log", "message": format!("wrote {}", files.iter().map(|f| f.display().to_string()).collect::<Vec<_>>().join(" and "))}));
                    Ok((dir.clone(), "recipe".into()))
                }
                "import" => {
                    let cat = catalog.as_ref().ok_or("library import --catalog catalog.py")?;
                    let dst = dst.as_ref().ok_or("library import ... --out soup-recipes.json")?;
                    let py = std::fs::read_to_string(cat).map_err(|x| format!("{}: {x}", cat.display()))?;
                    let a: Option<Value> = match archs {
                        Some(p) => Some(serde_json::from_str(&std::fs::read_to_string(p).map_err(|x| format!("{}: {x}", p.display()))?)
                            .map_err(|x| format!("{}: {x}", p.display()))?),
                        None => None,
                    };
                    let recipes = library::import_catalog(&py, a.as_ref()).map_err(|x| x.to_string())?;
                    let c = library::Catalogue { source: json!({"name": "Soup", "repository": "https://github.com/MakazhanAlpamys/Soup",
                        "licence": "Apache-2.0", "commit": commit, "files": ["src/soup_cli/recipes/catalog.py", "tests/fixtures/recipe_base_architectures.json"],
                        "note": "Soup's recipe catalogue, read as text and ported to this library; see NOTICE-soup.md"}), recipes };
                    out.event(json!({"event": "log", "message": format!("{} recipes from {}", c.recipes.len(), cat.display())}));
                    std::fs::write(dst, serde_json::to_string_pretty(&c).unwrap() + "\n").map_err(|x| format!("{}: {x}", dst.display()))?;
                    Ok((dst.clone(), "recipes".into()))
                }
                other => Err(format!("library {other:?}: list, show, init or import")),
            }
        }
        Cmd::Train { .. } | Cmd::FindLr { .. } | Cmd::Profile { .. } | Cmd::Autopilot { .. } | Cmd::Abliterate { .. } | Cmd::Upcycle { .. } | Cmd::ExportMoe { .. } | Cmd::Regress { .. } | Cmd::GradeTrain { .. } => unreachable!("handled by run"),
        Cmd::Info { path } => {
            out.event(json!({"event": "info", "info": info(path)?}));
            Ok((path.clone(), "info".into()))
        }
        Cmd::Convert { dir, out: dst, outtype, pre_tokenizer, yarn, yarn_original } => {
            let ot = convert::OutType::from_s(outtype).map_err(|e| e.to_string())?;
            // the Llama converter holds the model in f32; the Qwen3 one streams a tensor at a time
            if !convert::is_qwen3(dir) {
                fits(f32_bytes_of(dir), "converting this model")?;
            }
            out.event(json!({"event": "start", "what": format!("convert {} to {} ({outtype})", dir.display(), dst.display())}));
            let add = match yarn {
                Some(f) => {
                    let original = match yarn_original {
                        Some(o) => *o,
                        None => {
                            let c: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).map_err(|e| e.to_string())?)
                                .map_err(|e| e.to_string())?;
                            c["max_position_embeddings"].as_u64().map_or(32768, |m| m.min(32768) as usize)
                        }
                    };
                    Some(convert::AddYarn { factor: *f, original })
                }
                None => None,
            };
            convert::convert_with(dir, dst, ot, pre_tokenizer.as_deref(), add).map_err(|e| e.to_string())?;
            Ok((dst.clone(), "gguf".into()))
        }
        Cmd::Quantize { input, out: dst, ty } => {
            let t = quantize_gguf::PureType::from_s(ty).map_err(|e| e.to_string())?;
            fits(size_of(input) * 2, "quantizing this model")?;
            out.event(json!({"event": "start", "what": format!("quantize {} to {} ({ty})", input.display(), dst.display())}));
            quantize_gguf::quantize(input, dst, t).map_err(|e| e.to_string())?;
            Ok((dst.clone(), "gguf".into()))
        }
        Cmd::Datagen { recipe, sources, seeds, out: dst, count, per_chunk, chunk_chars, overlap, temperature, min_score, topic, language, seed, check_code, think, model: writer } => {
            let settings = datagen::Settings {
                recipe: datagen::Recipe::from_s(recipe).map_err(|e| e.to_string())?, sources: sources.clone(), seeds: seeds.clone(),
                count: *count, per_chunk: (*per_chunk).clamp(1, 20), chunk_chars: (*chunk_chars).clamp(500, 20_000),
                overlap: *overlap, temperature: *temperature, min_score: (*min_score).min(5), topic: topic.clone(),
                language: language.clone(), out: dst.clone(), seed: *seed, check_code: *check_code, think: *think,
            };
            let model = datagen::Model::from_env_with(writer.as_deref()).map_err(|e| e.to_string())?;
            out.event(json!({"event": "start", "what": format!("{} data with {} into {}", recipe, model.name, dst.display())}));
            // a cancel (SIGTERM from the server) stops after the current request and still writes what was made
            #[cfg(unix)]
            unsafe { libc::signal(libc::SIGTERM, on_term as extern "C" fn(libc::c_int) as libc::sighandler_t); }
            let r = datagen::run(&settings, &model,
                |done, total, msg| out.event(json!({"event": "progress", "done": done, "total": total, "message": msg})),
                || TERM.load(std::sync::atomic::Ordering::SeqCst)).map_err(|e| e.to_string())?;
            out.event(json!({"event": "log", "message": format!("kept {} of {} written ({} filtered, {} duplicates, {} judged out, {} requests failed)",
                r.kept, r.generated, r.filtered.values().sum::<usize>(), r.duplicates, r.judged_out, r.failed)}));
            if let Some(e) = &r.stopped {
                out.event(json!({"event": "log", "message": format!("the model stopped answering ({e}); the records so far were kept")}));
            }
            if r.kept == 0 && !TERM.load(std::sync::atomic::Ordering::SeqCst) {
                let why = if let Some(e) = &r.stopped {
                    format!("the model stopped answering before any record was kept: {e}")
                } else if r.generated == 0 {
                    format!("{} of {} replies from {} held no records in the asked-for JSON: try a larger or instruction-tuned model",
                            r.unparsable, r.asked, model.name)
                } else {
                    "every record was dropped by the filters, as duplicates or by the judge: see the dataset card".to_string()
                };
                // nothing to train on: no empty dataset is left behind (its card would only repeat this)
                std::fs::remove_file(dst).ok();
                std::fs::remove_file(datagen::card_path(dst)).ok();
                return Err(format!("no records kept: {why}"));
            }
            Ok((dst.clone(), "dataset".into()))
        }
        Cmd::Clean { inputs, out: dst, model, chat_template, max_len, min_answer_tokens, min_answer_ratio, max_repeat, dedupe, near,
                     lang_match, exclude, cap_per_language, cap_only, seed, report } => {
            let mut s = clean::Settings::new(inputs.clone(), dst.clone());
            s.model = model.clone();
            s.chat_template = chat_template.clone();
            s.max_len = (*max_len).max(2);
            s.min_answer_tokens = *min_answer_tokens;
            s.min_answer_ratio = *min_answer_ratio;
            s.max_repeat = *max_repeat;
            s.dedupe = clean::Dedupe::from_s(dedupe).map_err(|e| e.to_string())?;
            s.near = *near;
            s.lang_match = *lang_match;
            s.exclude = exclude.clone();
            s.cap_per_language = *cap_per_language;
            s.cap_only = cap_only.clone();
            if let Some(c) = cap_only.iter().find(|c| !inputs.contains(c)) {
                return Err(format!("--cap-only {}: not one of the inputs", c.display()));
            }
            s.seed = *seed;
            s.report = report.clone();
            if inputs.iter().chain(exclude.iter()).any(|p| p == dst) {
                return Err(format!("{}: the output cannot be one of the inputs", dst.display()));
            }
            out.event(json!({"event": "start", "what": format!("clean {} file(s) into {}", inputs.len(), dst.display())}));
            let r = clean::run(&s, |v| out.event(v)).map_err(|e| format!("{e:#}"))?;
            out.event(json!({"event": "log", "message": format!("kept {} of {} rows; card: {}", r.kept, r.read, s.card_path().display())}));
            if r.kept == 0 {
                std::fs::remove_file(dst).ok();
                std::fs::remove_file(s.card_path()).ok();
                let why: Vec<String> = r.dropped.iter().filter(|(_, n)| **n > 0).map(|(k, n)| format!("{n} {k}")).collect();
                return Err(format!("no rows kept: every row was dropped ({})", why.join(", ")));
            }
            Ok((dst.clone(), "dataset".into()))
        }
        Cmd::Loop { model, out: dst, start, end, passes } => {
            out.event(json!({"event": "start", "what": format!("loop blocks {start}..{end} of {} x{passes} as {}", model.display(), dst.display())}));
            strata_workshop::model::make_looped(model, dst, strata_workshop::model::LoopSpec { start: *start, end: *end, passes: *passes })
                .map_err(|e| e.to_string())?;
            Ok((dst.clone(), "hf-model".into()))
        }
        Cmd::Context { model, out: dst, length, original, window, full_every } => {
            out.event(json!({"event": "start", "what": format!("{} set up for {length} tokens as {}", model.display(), dst.display())}));
            strata_workshop::model::make_long_context(model, dst, *length, *original, *window, *full_every).map_err(|e| e.to_string())?;
            Ok((dst.clone(), "hf-model".into()))
        }
        Cmd::Merge { base, adapter, out: dst } => {
            out.event(json!({"event": "start", "what": format!("merge {} into {} as {}", adapter.display(), base.display(), dst.display())}));
            let r = merge::merge(base, adapter, dst).map_err(|e| e.to_string())?;
            let touched: Vec<&merge::TensorDelta> = r.per_tensor.iter().filter(|t| t.delta_rel_norm > 0.0).collect();
            out.event(json!({"event": "log", "message": format!("{} of {} tensors changed by the adapter", touched.len(), r.per_tensor.len())}));
            for t in touched.iter().take(8) {
                out.event(json!({"event": "log", "message": format!("{}: |ΔW|/|W| = {:.4}", t.name, t.delta_rel_norm)}));
            }
            Ok((r.out_dir, "hf-model".into()))
        }
    }
}

/// `--threads N`: candle and its matrix products size their thread pools from `RAYON_NUM_THREADS` and
/// `CANDLE_NUM_THREADS` the first time they compute. This process never changes its own environment (other threads
/// may read it, which makes `set_var` unsound); when the two variables do not say N yet, it starts itself again with
/// them set for the new process only. On Unix that is `exec`: the same process (the same PID, so the server's SIGTERM
/// still reaches the run). Elsewhere a child runs and its exit code is passed on. `None`: they say N, carry on.
fn with_threads(n: usize) -> Option<ExitCode> {
    let want = n.to_string();
    let set = |k: &str| std::env::var(k).ok().as_deref() == Some(want.as_str());
    if set("RAYON_NUM_THREADS") && set("CANDLE_NUM_THREADS") {
        return None;
    }
    let exe = match std::env::current_exe() {
        Ok(e) => e,
        Err(_) => return None, // no way to start again: all cores, said by the start event's thread count being absent
    };
    let mut cmd = std::process::Command::new(exe);
    cmd.args(std::env::args_os().skip(1)).env("RAYON_NUM_THREADS", &want).env("CANDLE_NUM_THREADS", &want);
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt as _;
        let e = cmd.exec(); // returns only on failure
        eprintln!("--threads {n}: could not restart with the thread count set ({e}); running on all cores");
        None
    }
    #[cfg(not(unix))]
    {
        match cmd.status() {
            Ok(st) => Some(ExitCode::from(st.code().unwrap_or(1).clamp(0, 255) as u8)),
            Err(e) => {
                eprintln!("--threads {n}: could not restart with the thread count set ({e}); running on all cores");
                None
            }
        }
    }
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let threads = match &cli.cmd {
        Cmd::Train { run, .. } | Cmd::FindLr { run, .. } | Cmd::Profile { run, .. } => run.threads,
        Cmd::Autopilot { threads, .. } => *threads,
        _ => None,
    };
    if let Some(n) = threads.filter(|n| *n > 0) {
        if let Some(code) = with_threads(n) {
            return code;
        }
    }
    let out = Out { json: cli.json };
    let t0 = Instant::now();
    match run(&cli.cmd, &out) {
        Ok((path, kind, extra)) => {
            if kind != "info" {
                let mut done = json!({"event": "done", "output": path.display().to_string(), "kind": kind,
                                      "bytes": size_of(&path), "seconds": t0.elapsed().as_secs_f64()});
                for (k, v) in extra.as_object().into_iter().flatten() {
                    done[k.as_str()] = v.clone();
                }
                out.event(done);
            }
            ExitCode::SUCCESS
        }
        Err(e) => {
            out.event(json!({"event": "error", "message": e}));
            ExitCode::from(1)
        }
    }
}
