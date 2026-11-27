//! `am2strata-workshop`: the Workshop's jobs as a command, in Rust (no Python).
//!
//! ```text
//! am2strata-workshop info <model.gguf | hf-folder>
//! am2strata-workshop convert <hf-folder> --out <model.gguf> [--outtype f16|f32] [--pre-tokenizer NAME] [--yarn 4 [--yarn-original 32768]]
//! am2strata-workshop quantize <in.gguf> --out <out.gguf> --type q8_0|q6_k|q5_k|q4_k|q4_0|f16
//! am2strata-workshop merge <base-hf-folder> <adapter-folder> --out <merged-hf-folder> [--no-abliterate-carry | --allow-unprojected] [--direction <file>]
//! am2strata-workshop upcycle <hf-folder> --experts 10 --top-k 1 --out <dir>
//! am2strata-workshop upcycle <hf-folder> --experts 120 --expert-ff 1024 --top-k 5 [--split cluster|contiguous] [--calib chat.jsonl] [--dense-start-check] --out <dir>
//! am2strata-workshop export-moe <upcycled-folder> [--adapter <dir>] --out <model.gguf> [--type q4_k|q8_0|f16|f32] [--no-abliterate-carry | --allow-unprojected] [--direction <file>]
//! am2strata-workshop abliterate <hf-folder> --harmful a.txt --harmless b.txt [--layer auto|N|N:P] [--scale 1.0] [--check N] --out <dir>
//! am2strata-workshop abliterate --direction-only <original-hf-folder> --harmful a.txt --harmless b.txt --layer N [--pass P] [--tolerance 0.05] --out <abliterated-folder>
//! am2strata-workshop refusal-check --engine-gguf <model.gguf|model.kosha> --tokenizer <hf-folder> --harmful a.txt --harmless b.txt [--n 32] [--max-tokens 48] [--engine-args "…"] [--before <other.gguf>] [--report out.json]
//! am2strata-workshop train --model <hf-folder> --data <chat.jsonl> --out <adapter-folder> [--soup soup.yaml] [...]
//! am2strata-workshop find-lr --model <hf-folder> --data <chat.jsonl> [--steps 40]
//! am2strata-workshop profile (--model <hf-folder> | --preset llama-2-7b) [--data <chat.jsonl>] [...]
//! am2strata-workshop autopilot --model <hf-folder> --data <chat.jsonl> [--time-budget 2h] [--ram-budget 12GB]
//! am2strata-workshop regress --model <hf-folder> [--adapter <dir>] [--passes 1,2] --suite <suite.json> --out <report-dir> [--against <report>]
//! am2strata-workshop clean <in.jsonl>... --out <out.jsonl> [--model <hf-folder>] [--max-len 1024] [--min-answer-tokens N] [--max-repeat R] [--dedupe exact|near|off] [--lang-match] [--drop-cut] [--exclude held-out.jsonl] [--cap-per-language N [--cap-only in.jsonl]] [--report card.md]
//! am2strata-workshop private-data --name <name> --source <folder>... [--model <local-model>] [--data-dir ~/Development/Strata-data]
//! am2strata-workshop private-import (--folder <dir> | --url <https-url> | --hf <owner/name> [--revision main] [--pattern "*.zip"]) --name <name> [--attestation SOURCE.toml] [--confirm-bytes N] [--id <import-id>]
//! am2strata-workshop private-import-probe (--folder <dir> | --url <https-url> | --hf <owner/name>)
//! am2strata-workshop private-import-discard --id <import-id>
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
use strata_workshop::finetune::{self, Event, FindLr, Lisa, Resume, TrainSettings};
use strata_workshop::layer_stream::StreamMode;
use strata_workshop::model::{BaseDtype, ModelConfig, Target};
use strata_workshop::profile::{self, AutopilotInput, Machine, ProfileInput, StreamProfile};
use strata_workshop::{clean, convert, datagen, merge, quantize_gguf, soup};

#[derive(Parser)]
#[command(name = "am2strata-workshop", about = "Convert, quantize and merge models in Rust (Prastara's Workshop)")]
struct Cli {
    /// one JSON event per line on stdout (for the server and scripts)
    #[arg(long, global = true)]
    json: bool,
    /// a job over private data (the server adds it): every chat with the server stays on this machine
    #[arg(long, global = true, hide = true)]
    local_only: bool,
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
        /// fine-grained experts of this many of the dense feed-forward's neurons (e.g. 1024 of Qwen3-8B's 12,288: 12
        /// groups, --experts a multiple of 12); without it every expert is the whole feed-forward
        #[arg(long)]
        expert_ff: Option<usize>,
        /// how the neurons are grouped: cluster (by co-activation on --calib, MoEfication) or contiguous
        #[arg(long, default_value = "cluster")]
        split: String,
        /// the experts' LoRA rank (0: train the router only); fine-grained default: the largest that fits the cards
        #[arg(long)]
        expert_rank: Option<usize>,
        /// one LoRA A per neuron group shared by its copies (each copy its own B); with --expert-rank given
        #[arg(long)]
        shared_a: bool,
        /// calibration conversations (JSONL) for the split and the router's start
        #[arg(long)]
        calib: Option<PathBuf>,
        /// calibration tokens in all
        #[arg(long, default_value_t = 4096)]
        calib_tokens: usize,
        /// each calibration conversation cut at this many tokens
        #[arg(long, default_value_t = 512)]
        calib_max_len: usize,
        /// measure the held-out loss of the dense model and of the upcycled one at its start (needs --calib)
        #[arg(long)]
        dense_start_check: bool,
        /// held-out answer tokens for --dense-start-check
        #[arg(long, default_value_t = 2048)]
        check_tokens: usize,
        /// the calibration's device: cpu (default) or cuda:N
        #[arg(long, default_value = "cpu")]
        device: String,
        /// how the base is held while calibrating: bf16 (default), f16 or f32
        #[arg(long, default_value = "bf16")]
        base_dtype: String,
        #[arg(long, default_value_t = 42)]
        seed: u64,
        /// print the sizes, the export sizes and what training it takes on this machine's GPUs, and stop
        #[arg(long)]
        estimate: bool,
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
        /// print the disk and memory it needs, and stop
        #[arg(long)]
        estimate: bool,
        #[command(flatten)]
        carry: CarryArgs,
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
        /// with --direction-only: the abliterated folder the direction is written into; else the new folder
        #[arg(long)]
        out: PathBuf,
        /// print what the run needs (memory, disk, time) and stop
        #[arg(long)]
        estimate: bool,
        /// only recompute the refusal direction of an abliterated folder (--out) that has no refusal_direction.safetensors,
        /// on its original model (the positional folder) at the recorded block and pass; writes only that file
        #[arg(long)]
        direction_only: bool,
        /// the loop pass of --layer (the same as --layer N:P)
        #[arg(long)]
        pass: Option<usize>,
        /// --direction-only: how far the recomputed separation may be from the recorded one (relative; 0.05 is 5 %)
        #[arg(long, default_value_t = 0.05)]
        tolerance: f64,
    },
    /// how often a GGUF or Kosha pack refuses, answered greedily through the Rust engine: the last --n prompts of each
    /// list, a refusal phrase list (abliterate's); --before answers the same prompts with another file
    RefusalCheck {
        #[command(flatten)]
        engine: EngineOpts,
        /// the model's Hugging Face folder: its tokenizer and chat template (the folder the file was made from)
        #[arg(long)]
        tokenizer: PathBuf,
        #[arg(long)]
        harmful: PathBuf,
        #[arg(long)]
        harmless: PathBuf,
        /// prompts of each list (the last ones)
        #[arg(long, default_value_t = 32)]
        n: usize,
        /// new tokens per answer
        #[arg(long, default_value_t = 48)]
        max_tokens: usize,
        /// another GGUF or pack to answer the same prompts with (e.g. the model before abliteration)
        #[arg(long)]
        before: Option<PathBuf>,
        /// write the result here as JSON
        #[arg(long)]
        report: Option<PathBuf>,
    },
    /// ordinary prompts from chat JSONL files or folders of them (the first user message of each record), one per line:
    /// the harmless list for `abliterate`, from our own data
    Prompts {
        /// .jsonl files or folders holding them
        #[arg(required = true)]
        inputs: Vec<PathBuf>,
        #[arg(long)]
        out: PathBuf,
        /// at most this many prompts
        #[arg(long, default_value_t = 520)]
        max: usize,
        /// skip prompts longer than this many characters
        #[arg(long, default_value_t = 300)]
        max_chars: usize,
        /// only prompts that are almost all ASCII (to match an English harmful list)
        #[arg(long)]
        latin: bool,
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
    /// a private dataset built from the owner's own folders (each with its SOURCE.toml), described and written by a
    /// local model marked private, uncensored or abliterated, through the server's chat API on this machine
    /// (AM2STRATA_URL, AM2STRATA_API_KEY); into <data-dir>/private/<name>/ (0700). The safety gates cannot be turned off.
    PrivateData {
        /// the dataset's name: its folder under <data-dir>/private
        #[arg(long)]
        name: String,
        /// a source folder (repeat it); each needs its own SOURCE.toml
        #[arg(long = "source", required = true)]
        sources: Vec<PathBuf>,
        /// the local model (default: the loaded one); a <provider>/<model> name is refused
        #[arg(long)]
        model: Option<String>,
        /// Prastara's data folder
        #[arg(long, default_value = "~/Development/Strata-data")]
        data_dir: String,
        /// text sources: how many chat rows to aim for
        #[arg(long, default_value_t = 100)]
        text_count: usize,
        #[arg(long, default_value_t = 3)]
        per_chunk: usize,
        #[arg(long, default_value_t = 3000)]
        chunk_chars: usize,
        #[arg(long, default_value_t = 0.4)]
        temperature: f64,
        /// videos: one frame every this many seconds
        #[arg(long, default_value_t = 5.0)]
        every_s: f64,
        /// videos: at most this many frames each
        #[arg(long, default_value_t = 8)]
        max_frames: usize,
    },
    /// import the owner's adult dataset (a folder, a read-only NAS mount too, an https URL or a Hugging Face dataset;
    /// each with its SOURCE.toml) into <data-dir>/private/quarantine/<id>/, check every item with the local check models
    /// (<data-dir>/private/check-models/), and release what passed to <data-dir>/private/datasets/<name>/. A network
    /// source needs --confirm-bytes, the size its probe gave. The checks cannot be turned off.
    PrivateImport {
        #[command(flatten)]
        src: ImportSource,
        /// the dataset's name: its folder under <data-dir>/private/datasets
        #[arg(long)]
        name: String,
        /// the SOURCE.toml (default: the folder's own); outside the folder it names the source (folder, url or hf)
        #[arg(long)]
        attestation: Option<PathBuf>,
        /// a network source: the size in bytes you confirmed (it must still be the size)
        #[arg(long)]
        confirm_bytes: Option<u64>,
        /// the flagged share that rejects the whole source (at most and by default 0.01)
        #[arg(long)]
        minor_share: Option<f64>,
        /// resume this import (in quarantine) instead of starting one
        #[arg(long)]
        id: Option<String>,
        /// one check of one item at most this many seconds
        #[arg(long, default_value_t = 120)]
        check_timeout_s: u64,
        #[arg(long, default_value = "~/Development/Strata-data")]
        data_dir: String,
    },
    /// what a private-import source holds (files, bytes, the Hub commit) without fetching it
    PrivateImportProbe {
        #[command(flatten)]
        src: ImportSource,
    },
    /// delete an import's quarantine
    PrivateImportDiscard {
        #[arg(long)]
        id: String,
        #[arg(long, default_value = "~/Development/Strata-data")]
        data_dir: String,
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
        /// drop rows longer than --max-len instead of keeping them for the trainer to cut
        #[arg(long)]
        drop_cut: bool,
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
        #[command(flatten)]
        carry: CarryArgs,
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
        /// run even when the memory estimate says a device is too small
        #[arg(long)]
        force: bool,
        #[command(flatten)]
        engine: EngineOpts,
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
        #[command(flatten)]
        engine: EngineOpts,
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
    /// stream each block's frozen weights to the device just before it runs (after Soup's stream_layers): from host RAM
    /// (page-locked on CUDA), from a shard cache on disk, or auto (RAM when the base fits 70 % of the free RAM).
    /// Turns gradient checkpointing on.
    #[arg(long, num_args = 0..=1, default_missing_value = "auto", value_name = "ram|disk|auto")]
    stream_layers: Option<String>,
    /// with --stream-layers: blocks held on each device at once, 2 to 8 (2: the one running and the next)
    #[arg(long)]
    stream_buffers: Option<usize>,
    /// with --stream-layers disk: where the shard cache goes (default ~/.cache/prastara/layer-stream)
    #[arg(long)]
    stream_cache: Option<PathBuf>,
    /// with --stream-layers: each GPU's host-to-device rate in GB/s for the estimate (one value, or one per card);
    /// default from the PCIe link in sysfs
    #[arg(long, value_delimiter = ',')]
    stream_bandwidth: Vec<f64>,
    /// LISA (after Soup): only K sampled blocks' adapters train at a time
    #[arg(long, value_name = "K")]
    lisa_layers: Option<usize>,
    /// with --lisa-layers: sample the blocks anew every N optimizer steps (default 20, Soup's)
    #[arg(long, value_name = "N")]
    lisa_interval: Option<usize>,
    /// with --lisa-layers: the sampling seed (default: --seed)
    #[arg(long)]
    lisa_seed: Option<u64>,
    /// Soup's freeze_layers: no adapters in the bottom N blocks (the same as --train-layers <blocks - N>)
    #[arg(long, value_name = "N")]
    freeze_bottom: Option<usize>,
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
        if let Some(m) = &self.stream_layers {
            s.stream_layers = Some(StreamMode::from_s(m).map_err(|e| e.to_string())?);
        }
        if let Some(b) = self.stream_buffers {
            if s.stream_layers.is_none() {
                return Err("--stream-buffers needs --stream-layers".into());
            }
            s.stream_buffers = b;
        }
        if let Some(c) = &self.stream_cache { s.stream_cache = Some(c.clone()); }
        if s.stream_layers.is_some() && !s.gradient_checkpointing {
            notes.push("--stream-layers turns gradient checkpointing on (each block is replayed from its streamed weights in the backward pass)".into());
            s.gradient_checkpointing = true;
        }
        match (self.lisa_layers, self.lisa_interval) {
            (Some(k), n) => s.lisa = Some(Lisa { layers: k, interval: n.unwrap_or(20), seed: self.lisa_seed.unwrap_or(s.seed) }),
            (None, Some(_)) => return Err("--lisa-interval needs --lisa-layers K".into()),
            (None, None) if self.lisa_seed.is_some() => return Err("--lisa-seed needs --lisa-layers K".into()),
            (None, None) => {}
        }
        if let Some(n) = self.freeze_bottom {
            s.freeze_bottom = Some(n);
        }
        if s.freeze_bottom.is_some() {
            // resolved now when the model's config is there (a --preset has none: profile resolves it)
            if let Ok(cfg) = ModelConfig::load(&s.model) {
                s.apply_freeze_bottom(cfg.num_hidden_layers).map_err(|e| e.to_string())?;
            }
        }
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
fn profile_input(s: &TrainSettings, cfg: ModelConfig, examples: Option<usize>, avg_tokens: Option<usize>, bandwidth: &[f64]) -> Result<ProfileInput, String> {
    let (n, avg) = if s.data.as_os_str().is_empty() {
        (examples.unwrap_or(1000), avg_tokens.unwrap_or(s.max_len / 2))
    } else {
        let (n, avg, _) = data_stats(s)?;
        // what is held out is not trained on
        let held = if s.eval_data.is_none() && s.val_split > 0.0 && n >= 2 { ((n as f64 * s.val_split).round() as usize).clamp(1, n - 1) } else { 0 };
        (examples.unwrap_or(n - held), avg_tokens.unwrap_or(avg))
    };
    let max_len = s.max_len.min(cfg.max_position_embeddings.max(8));
    let mut s = s.clone();
    s.apply_freeze_bottom(cfg.num_hidden_layers).map_err(|e| e.to_string())?;
    let devices = s.devices.len().max(1);
    let stream = s.stream_layers.map(|mode| {
        let on_gpu = s.devices.iter().any(|d| d.starts_with("cuda"));
        let skip = if s.cache_dir.is_some() { s.lora.first_trained(cfg.num_hidden_layers) } else { 0 };
        let store = strata_workshop::layer_stream::block_stream_bytes(&cfg, s.base_dtype) * (cfg.num_hidden_layers - skip) as u64;
        let (tier, _) = strata_workshop::layer_stream::choose_tier(mode, store, strata_workshop::layer_stream::ram_available());
        let rates: Vec<f64> = if !bandwidth.is_empty() {
            bandwidth.iter().map(|g| g * 1e9).collect()
        } else {
            // the cards the run names, by index (cuda:N), from the links in sysfs
            let links = strata_workshop::layer_stream::gpu_link_rates();
            s.devices.iter().filter_map(|d| d.strip_prefix("cuda:")?.parse::<usize>().ok()).map(|i| links.get(i).map_or(strata_workshop::layer_stream::DEFAULT_LINK_BYTES_S, |l| l.1)).collect()
        };
        StreamProfile { mode, tier, buffers: s.stream_buffers, rates, on_gpu }
    });
    Ok(ProfileInput { checkpointing: s.gradient_checkpointing || stream.is_some(), devices, lora: s.lora.clone(), base_dtype: s.base_dtype, max_len, accumulation: s.accumulation,
                      cache: s.cache_dir.is_some() && s.lora.first_trained(cfg.num_hidden_layers) > 0, examples: n, epochs: s.epochs, avg_tokens: avg.min(max_len), cfg, stream })
}

fn os_threads() -> Option<u64> {
    std::fs::read_to_string("/proc/self/status").ok()?.lines().find(|l| l.starts_with("Threads:"))?.split_whitespace().nth(1)?.parse().ok()
}

/// A private-import source: one of a folder, an https URL, or a Hugging Face dataset.
#[derive(Args, Clone)]
struct ImportSource {
    /// a local folder (a read-only NAS mount too); never written to
    #[arg(long)]
    folder: Option<PathBuf>,
    /// a direct https URL to an archive or a file
    #[arg(long)]
    url: Option<String>,
    /// a Hugging Face dataset id (owner/name), fetched over the Hub's HTTP API (HF_ENDPOINT, HF_TOKEN)
    #[arg(long)]
    hf: Option<String>,
    /// --hf: the branch, tag or commit
    #[arg(long, default_value = "main")]
    revision: String,
    /// --hf: only the files that match (`*`, `**`, `?`; `|` between patterns)
    #[arg(long)]
    pattern: Option<String>,
}

impl ImportSource {
    fn spec(&self) -> Result<strata_workshop::private_import::Spec, String> {
        use strata_workshop::private_import::Spec;
        match (&self.folder, &self.url, &self.hf) {
            (Some(f), None, None) => Ok(Spec::Folder(f.clone())),
            (None, Some(u), None) => Ok(Spec::Url(u.clone())),
            (None, None, Some(h)) => Ok(Spec::Hf { id: h.clone(), revision: self.revision.clone(), pattern: self.pattern.clone() }),
            _ => Err("give one source: --folder, --url or --hf".into()),
        }
    }
}

fn home_path(p: &str) -> Result<PathBuf, String> {
    Ok(match p.strip_prefix("~/") {
        Some(rest) => std::env::var_os("HOME").map(PathBuf::from).ok_or("no HOME for ~")?.join(rest),
        None => PathBuf::from(p),
    })
}

/// A folder's plain files (links not followed) and their bytes, for the confirm dialog.
fn folder_size(d: &Path) -> (u64, usize, Option<String>) {
    let (mut bytes, mut files) = (0u64, 0usize);
    let mut stack = vec![(d.to_path_buf(), 0)];
    while let Some((p, depth)) = stack.pop() {
        for e in std::fs::read_dir(&p).into_iter().flatten().flatten() {
            match std::fs::symlink_metadata(e.path()) {
                Ok(m) if m.is_file() => { bytes += m.len(); files += 1; }
                Ok(m) if m.is_dir() && depth < 16 => stack.push((e.path(), depth + 1)),
                _ => {}
            }
        }
    }
    (bytes, files, None)
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
            Some("profile") | Some("plan") | Some("find-lr") | Some("library") | Some("estimate") => {
                for l in v["lines"].as_array().into_iter().flatten() {
                    println!("{}", l.as_str().unwrap_or(""));
                }
            }
            _ => println!("{v}"),
        }
    }
}

/// What training an upcycled `cfg` takes on this machine's GPUs (all of them, the blocks split), at the settings a
/// real run starts from (rank 16, all seven targets, 1,024 tokens, gradient checkpointing, a bf16 base), and the
/// largest settings that fit when those do not: lines, and the same as JSON.
fn moe_training_plan(cfg: &ModelConfig) -> (Vec<String>, Value) {
    let machine = Machine { cpu_gflops: 0.0, ..Machine::detect_memory() };
    let cards = machine.gpus.len().max(1);
    let p = ProfileInput { cfg: cfg.clone(), lora: strata_workshop::model::LoraSpec { r: 16, alpha: 32.0, targets: Target::ALL.to_vec(), ..Default::default() },
                           base_dtype: BaseDtype::BF16, max_len: 1024, accumulation: 4, cache: false, examples: 1000, epochs: 1.0, avg_tokens: 512,
                           checkpointing: true, devices: cards, stream: None };
    let e = profile::estimate(&p);
    let card = profile::card_bytes(&e);
    let mut lines = vec![format!("training (rank 16, all targets, 1,024 tokens, checkpointing, bf16 base, {cards} GPU(s)): about {} on the busiest card, {} trainable",
                                 human(card), params_label(e.trainable_params))];
    let fit = profile::suggest_gpu_fit(&p, &machine);
    if machine.gpus.is_empty() {
        lines.push("no NVIDIA GPU found here: the training estimate is for one card".into());
    } else if let Some(l) = profile::gpu_fit_line(&p, &machine) {
        lines.push(l);
    } else {
        lines.push("fits the cards (the chat model is unloaded for a GPU run)".into());
    }
    (lines, json!({"card_bytes": card, "devices": cards, "trainable_parameters": e.trainable_params,
                   "fits": fit.is_none() && !machine.gpus.is_empty() && profile::gpu_fit_line(&p, &machine).is_none(),
                   "suggested": fit.map(|(r, len, need)| json!({"rank": r, "max_len": len, "card_bytes": need}))}))
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

/// Whether a source folder holds images or videos (then the model must read images).
fn walk_has_media(dir: &Path) -> bool {
    fn walk(p: &Path, depth: usize) -> bool {
        std::fs::read_dir(p).into_iter().flatten().flatten().any(|e| {
            let q = e.path();
            let media = q.extension().and_then(|x| x.to_str()).is_some_and(|x| {
                ["png", "jpg", "jpeg", "webp", "mp4", "m4v", "mov", "mkv", "webm"].contains(&x.to_lowercase().as_str())
            });
            (q.is_file() && media) || (depth > 0 && q.is_dir() && !q.is_symlink() && walk(&q, depth - 1))
        })
    }
    walk(dir, 3)
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

/// The Rust engine for regress and grade-train answers.
#[derive(clap::Args, Debug, Clone)]
struct EngineOpts {
    /// answer the generation tests (code, qa, judge, repeat, samples) and samples through Prastara's Rust engine on this
    /// GGUF of the model (an adapter merged in); loss tests keep the Workshop's model
    #[arg(long)]
    engine_gguf: Option<PathBuf>,
    /// the engine binary (default: AM2STRATA_HYBRID_EXE, am2strata-hybrid beside this command, or on the PATH)
    #[arg(long)]
    engine_exe: Option<PathBuf>,
    /// more engine arguments as one string, e.g. "--devices cuda:0=14 --max-context 8192"
    #[arg(long, allow_hyphen_values = true)]
    engine_args: Option<String>,
}

impl EngineOpts {
    fn spec(&self) -> Result<Option<strata_workshop::engine::EngineSpec>, String> {
        let Some(gguf) = &self.engine_gguf else {
            if self.engine_exe.is_some() || self.engine_args.is_some() {
                return Err("--engine-exe and --engine-args go with --engine-gguf".into());
            }
            return Ok(None);
        };
        if !gguf.is_file() {
            return Err(format!("--engine-gguf {}: no such file", gguf.display()));
        }
        let exe = strata_workshop::engine::find_exe(self.engine_exe.as_deref()).map_err(|e| format!("{e:#}"))?;
        let args = self.engine_args.as_deref().unwrap_or("").split_whitespace().map(str::to_string).collect();
        Ok(Some(strata_workshop::engine::EngineSpec::new(exe, gguf.clone(), args)))
    }
}

/// What merge and export-moe do with an abliterated base (strata_workshop::carry).
#[derive(clap::Args, Debug, Clone, Default)]
struct CarryArgs {
    /// do not keep the adapter's changes off an abliterated base's refusal direction: the result may refuse again and
    /// loses its abliterated mark (it stays private)
    #[arg(long)]
    no_abliterate_carry: bool,
    /// an abliterated base without its refusal_direction.safetensors: go on without the projection (marked "carried": false)
    #[arg(long)]
    allow_unprojected: bool,
    /// the refusal direction file to use instead of the one beside the base
    #[arg(long)]
    direction: Option<PathBuf>,
}

impl CarryArgs {
    fn options(&self) -> Result<strata_workshop::carry::Options, String> {
        use strata_workshop::carry::Mode;
        let mode = match (self.no_abliterate_carry, self.allow_unprojected) {
            (true, true) => return Err("--no-abliterate-carry and --allow-unprojected: one or the other".into()),
            (true, false) => Mode::Skip,
            (false, true) => Mode::AllowUnprojected,
            (false, false) => Mode::Carry,
        };
        if self.direction.is_some() && mode == Mode::Skip {
            return Err("--direction goes with the carry, not with --no-abliterate-carry".into());
        }
        Ok(strata_workshop::carry::Options { mode, direction: self.direction.clone() })
    }

    /// The plan for `base`, decided before the job starts (a missing direction refused at once), and its lines logged.
    fn plan(&self, base: &Path, out: &Out) -> Result<strata_workshop::carry::Options, String> {
        let o = self.options()?;
        let cfg: Value = std::fs::read_to_string(base.join("config.json")).ok().and_then(|t| serde_json::from_str(&t).ok()).unwrap_or_default();
        let p = strata_workshop::carry::plan(base, &cfg, &o).map_err(|e| format!("{e:#}"))?;
        for l in p.lines() {
            if !out.json {
                eprintln!("{l}");
            }
            out.event(json!({"event": "log", "message": l, "carry": p.state()}));
        }
        Ok(o)
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
            let p = profile_input(&s, cfg, None, None, &run.stream_bandwidth)?;
            let est = profile::estimate(&p);
            let machine = Machine::detect();
            let lines = profile::report_lines(&p, &est, &machine);
            let streamed: Vec<String> = lines.iter().filter(|l| l.starts_with("layer streaming") || l.starts_with("PCIe") || l.starts_with("host RAM for")
                                                              || l.starts_with("disk for the shard")).cloned().collect();
            let first: Vec<String> = lines.into_iter().skip(1).take(4).filter(|l| !streamed.contains(l)).collect();
            for l in first.into_iter().chain(streamed) {
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
            let p = profile_input(&s, cfg, *examples, *avg_tokens, &run.stream_bandwidth)?;
            let est = profile::estimate(&p);
            let mut machine = Machine::detect();
            if let Some(g) = cpu_gflops {
                machine.cpu_gflops = *g;
            }
            let mut lines = profile::report_lines(&p, &est, &machine);
            if s.devices.iter().any(|d| d.starts_with("cuda")) {
                lines.extend(profile::gpu_fit_line(&p, &machine));
            }
            lines.extend(notes);
            lines.push("a 4-bit base (QLoRA) and vision towers are not done; several GPUs need a cuda build (--devices)".into());
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
        Cmd::Abliterate { model, harmful, harmless, layer, scale, check, devices, base_dtype, out: dst, estimate, direction_only, pass, tolerance } => {
            use strata_workshop::abliterate;
            let read = |p: &Path| abliterate::read_prompt_list(p).map_err(|e| format!("{}: {e:#}", p.display()));
            let mut choice = abliterate::LayerChoice::parse(layer).map_err(|e| e.to_string())?;
            if let Some(p) = pass {
                match &mut choice {
                    abliterate::LayerChoice::Site(s) if !layer.contains(':') => s.pass = *p,
                    abliterate::LayerChoice::Site(_) => return Err("--pass and --layer N:P: say the pass once".into()),
                    abliterate::LayerChoice::Auto => return Err("--pass goes with --layer N".into()),
                }
            }
            if *direction_only {
                if *estimate || *check > 0 || *scale != 1.0 {
                    return Err("--direction-only writes the direction of an existing abliterated folder: --estimate, --check and --scale do not apply".into());
                }
                let mut o = abliterate::DirectionOnly::new(model, read(harmful)?, read(harmless)?, dst);
                o.layer = choice;
                o.tolerance = *tolerance;
                o.devices = devices.clone();
                o.base_dtype = base_dtype.as_deref().map(BaseDtype::from_s).transpose().map_err(|e| e.to_string())?;
                out.event(json!({"event": "start", "what": format!("recompute the refusal direction of {} from {}", dst.display(), model.display())}));
                let r = abliterate::direction_only(&o, &mut |v| out.event(v)).map_err(|e| format!("{e:#}"))?;
                return Ok((r.file.clone(), "refusal-direction".into(), json!({"layer": r.chosen.site.block, "pass": r.chosen.site.pass,
                    "separation": r.chosen.separation, "recorded_separation": r.recorded_separation, "separation_error": r.separation_error,
                    "writers": r.writers.iter().map(|(n, b, a)| json!({"tensor": n, "original": b, "abliterated": a})).collect::<Vec<_>>()})));
            }
            let mut o = abliterate::Options::new(model, read(harmful)?, read(harmless)?, dst);
            o.layer = choice;
            o.scale = *scale;
            o.check = *check;
            o.devices = devices.clone();
            o.base_dtype = base_dtype.as_deref().map(BaseDtype::from_s).transpose().map_err(|e| e.to_string())?;
            let cfg = ModelConfig::load(model).map_err(|e| format!("{e:#}"))?;
            // what it needs, before anything is loaded: the prompts' length through the chat template
            let gpus = devices.iter().filter(|d| d.as_str() != "cpu").count();
            let dtype = o.base_dtype.unwrap_or(match profile::source_dtype(model) { Ok(BaseDtype::F32) => BaseDtype::F32, _ => BaseDtype::BF16 });
            let tokens = strata_workshop::chatdata::ChatTokenizer::load(model, None).ok().map(|t| {
                let all: Vec<usize> = o.harmful.iter().chain(&o.harmless).take(64).filter_map(|p| t.prompt_ids_answering(p).ok().map(|i| i.len())).collect();
                all.iter().sum::<usize>() / all.len().max(1)
            }).unwrap_or(48);
            let source = strata_workshop::model::weight_files(model).map(|f| f.iter().map(|p| size_of(p)).sum()).unwrap_or(0);
            let est = abliterate::estimate(&cfg, dtype, gpus, o.harmful.len() + o.harmless.len() - 2 * o.check, tokens, o.check, o.check_tokens, source);
            let mut lines = vec![format!("{} harmful and {} harmless prompts ({} of each held out for the check), about {tokens} tokens each with the template",
                                         o.harmful.len(), o.harmless.len(), o.check)];
            lines.extend(est.lines(gpus, if *estimate || gpus == 0 { Machine::detect().cpu_gflops } else { 0.0 }));
            if let Some(free) = disk_free(dst) {
                if free < est.disk_bytes + (1 << 30) {
                    lines.push(format!("NOT ENOUGH DISK: the output needs about {} and {} is free there", human(est.disk_bytes), human(free)));
                }
            }
            if *estimate {
                out.event(json!({"event": "estimate", "kind": "abliterate", "estimate": est.to_json(), "lines": lines, "prompt_tokens": tokens}));
                return Ok((dst.clone(), "info".into(), json!({})));
            }
            out.event(json!({"event": "start", "what": format!("abliterate {} into {}", model.display(), dst.display())}));
            out.event(json!({"event": "estimate", "kind": "abliterate", "estimate": est.to_json(), "lines": lines}));
            fits(est.ram_bytes, "abliteration")?;
            if let Some(free) = disk_free(dst) {
                if free < est.disk_bytes + (1 << 30) {
                    return Err(format!("the abliterated copy needs about {} on disk and {} is free there", human(est.disk_bytes), human(free)));
                }
            }
            let r = abliterate::run(&o, &mut |v| out.event(v)).map_err(|e| format!("{e:#}"))?;
            let extra = json!({"layer": r.chosen.site.block, "pass": r.chosen.site.pass, "separation": r.chosen.separation,
                               "changed": r.ortho.changed.len(), "untied": r.ortho.untied, "private": true,
                               "before": r.before.as_ref().map(|c| c.to_json()), "after": r.after.as_ref().map(|c| c.to_json())});
            Ok((dst.clone(), "hf-model".into(), extra))
        }
        Cmd::Prompts { inputs, out: dst, max, max_chars, latin } => {
            let got = strata_workshop::abliterate::harvest_prompts(inputs, *max, *max_chars, *latin).map_err(|e| format!("{e:#}"))?;
            if got.is_empty() {
                return Err("no prompts matched (first user messages of {\"messages\": [...]} records)".into());
            }
            if let Some(p) = dst.parent().filter(|p| !p.as_os_str().is_empty()) {
                std::fs::create_dir_all(p).map_err(|e| format!("{}: {e}", p.display()))?;
            }
            std::fs::write(dst, got.join("\n") + "\n").map_err(|e| format!("{}: {e}", dst.display()))?;
            out.event(json!({"event": "log", "message": format!("{} prompts written to {}", got.len(), dst.display())}));
            Ok((dst.clone(), "prompts".into(), json!({"prompts": got.len()})))
        }
        Cmd::Upcycle { model, out: dst, experts, top_k, expert_ff, split, expert_rank, shared_a, calib, calib_tokens, calib_max_len, dense_start_check,
                       check_tokens, device, base_dtype, seed, estimate } => {
            use strata_workshop::model::{make_upcycled, MoeSpec, MoeSplit};
            use strata_workshop::moe_init;
            let dense = ModelConfig::load(model).map_err(|e| format!("{e:#}"))?;
            if dense.moe.is_some() {
                return Err(format!("{} is already upcycled", model.display()));
            }
            let mut m = MoeSpec { expert_ff: expert_ff.filter(|f| *f < dense.intermediate_size), split: MoeSplit::from_s(split).map_err(|e| e.to_string())?,
                                  expert_rank: *expert_rank, shared_a: *shared_a, ..MoeSpec::full(*experts, *top_k) };
            m.check_with(&dense).map_err(|e| format!("{e:#}"))?;
            let fine = m.fine(&dense);
            let mut cfg = dense.clone();
            cfg.moe = Some(m);
            let machine = Machine::detect_memory();
            let cards = machine.gpus.len().max(1);
            let card_total = machine.gpus.iter().map(|g| g.total_bytes).min().unwrap_or(profile::DEFAULT_CARD_BYTES);
            // fine-grained without --expert-rank: the largest scheme that fits the cards with the margin
            if fine && expert_rank.is_none() {
                if let Some(s) = profile::moe_default_scheme(&cfg, cards, card_total) {
                    m.expert_rank = Some(s.expert_rank);
                    m.shared_a = s.shared_a;
                    cfg.moe = Some(m);
                }
            }
            let (total, active) = m.params(&dense);
            let sizes: serde_json::Map<String, Value> = ["q4_k", "q8_0", "f16"].iter().filter_map(|t| {
                let ty = strata_workshop::export_moe::ExportType::from_s(t).ok()?;
                Some((t.to_string(), json!(strata_workshop::export_moe::export_bytes(&cfg, ty).ok()?)))
            }).collect();
            let shape = match m.expert_ff.filter(|_| fine) {
                Some(fe) => format!("{experts} experts of {fe} neurons ({} groups x {} copies) x top-{top_k}", m.groups(&dense), experts / m.groups(&dense)),
                None => format!("{experts} experts x top-{top_k}"),
            };
            let mut lines = vec![format!("{shape}: {} parameters in all, {} active per token (the dense base has {})",
                                         params_label(total), params_label(active), params_label(dense.total_params())),
                                 format!("export sizes: q4_k {}, q8_0 {}, f16 {}", human(sizes["q4_k"].as_u64().unwrap_or(0)),
                                         human(sizes["q8_0"].as_u64().unwrap_or(0)), human(sizes["f16"].as_u64().unwrap_or(0)))];
            let (train_lines, mut train) = moe_training_plan(&cfg);
            lines.extend(train_lines);
            let mut calib_json = Value::Null;
            let mut opts = moe_init::InitOptions { calib: calib.clone(), calib_tokens: *calib_tokens, max_len: *calib_max_len, seed: *seed,
                                               base_dtype: BaseDtype::from_s(base_dtype).map_err(|e| e.to_string())?,
                                               // opened only when the calibration runs (an estimate touches no card)
                                               device: candle_core::Device::Cpu,
                                               check_tokens: if *dense_start_check { *check_tokens } else { 0 }, ..Default::default() };
            if fine {
                lines.push(format!("not exact at the start: a token uses {} of the {} feed-forward neurons (top-{top_k} x {}), so the upcycled model starts as an approximation of the dense one{}",
                                   top_k * m.ff_e(&dense), dense.intermediate_size, m.ff_e(&dense),
                                   if *dense_start_check { "; --dense-start-check measures how far" } else { " (--dense-start-check measures how far)" }));
                let schemes = profile::moe_schemes(&cfg, cards, card_total);
                lines.push(format!("expert training schemes on {cards} card(s) of {} (fits = with {} to spare):", human(card_total), human(profile::MOE_CARD_MARGIN_BYTES)));
                for sc in &schemes {
                    lines.push(format!("  expert rank {:>2}{} at {:>4} tokens: {} trainable, {} on the busiest card{}", sc.expert_rank,
                                       if sc.shared_a { ", A shared per group" } else { "                    " }, sc.max_len, params_label(sc.trainable_params),
                                       human(sc.card_bytes), if sc.fits { "  fits" } else { "" }));
                }
                lines.push(format!("this upcycle: expert rank {}{}", m.rank(16), if m.shared_a { ", A shared per group" } else { "" }));
                train["schemes"] = json!(schemes.iter().map(|s| json!({"expert_rank": s.expert_rank, "shared_a": s.shared_a, "max_len": s.max_len,
                    "trainable_parameters": s.trainable_params, "card_bytes": s.card_bytes, "fits": s.fits})).collect::<Vec<_>>());
                if calib.is_some() {
                    let gflops = profile::measure_cpu_gflops() * 0.5;
                    let (secs, ram) = moe_init::estimate(&dense, m, &opts, if device == "cpu" { gflops } else { 2000.0 });
                    lines.push(format!("calibration ({} split, {} tokens{}): about {} and {} of memory", m.split.as_str(), calib_tokens,
                                       if *dense_start_check { format!(", check on {check_tokens} held-out tokens") } else { String::new() },
                                       profile::human_seconds(secs), human(ram)));
                    calib_json = json!({"seconds": secs, "ram_bytes": ram});
                } else if m.split == MoeSplit::Cluster {
                    lines.push("--split cluster needs --calib FILE.jsonl (calibration conversations); --split contiguous works without".into());
                }
            }
            if *estimate {
                out.event(json!({"event": "estimate", "kind": "upcycle", "lines": lines, "total_parameters": total, "active_parameters": active,
                                 "dense_parameters": dense.total_params(), "export_bytes": sizes, "training": train, "moe": m.to_json(), "calibration": calib_json}));
                return Ok((dst.clone(), "info".into(), json!({})));
            }
            out.event(json!({"event": "start", "what": format!("upcycle {} to {shape}, as {}", model.display(), dst.display())}));
            let mut init = Value::Null;
            if fine && (calib.is_some() || *dense_start_check) {
                if let Some(json) = calib_json.as_object() {
                    fits(json["ram_bytes"].as_u64().unwrap_or(0), "calibrating the upcycle")?;
                }
                opts.device = finetune::open_devices(std::slice::from_ref(device)).map_err(|e| format!("{e:#}"))?.remove(0);
                let r = moe_init::upcycle(model, dst, m, &opts, &mut |l| out.event(json!({"event": "log", "message": l}))).map_err(|e| format!("{e:#}"))?;
                lines.extend(r.lines());
                init = moe_init::report_json(&r);
            } else {
                make_upcycled(model, dst, m).map_err(|e| format!("{e:#}"))?;
            }
            for l in &lines {
                out.event(json!({"event": "log", "message": l}));
            }
            Ok((dst.clone(), "hf-model".into(), json!({"experts": experts, "top_k": top_k, "moe": m.to_json(), "total_parameters": total, "active_parameters": active,
                                                         "dense_parameters": dense.total_params(), "export_bytes": sizes, "training": train, "init": init})))
        }
        Cmd::ExportMoe { model, adapter, out: dst, ty, pre_tokenizer, estimate, carry } => {
            use strata_workshop::export_moe::{export_moe_carry, export_needs, ExportType};
            let carry_opts = carry.plan(model, out)?;
            let t = ExportType::from_s(ty).map_err(|e| e.to_string())?;
            let cfg = ModelConfig::load(model).map_err(|e| format!("{e:#}"))?;
            let (need, ram) = export_needs(&cfg, t, adapter.as_ref().map_or(0, |a| size_of(a))).map_err(|e| format!("{}: {e}", model.display()))?;
            let free = disk_free(dst);
            let mut lines = vec![format!("the file: about {} ({ty}), written one tensor at a time (no larger file first)", human(need)),
                                 format!("memory: about {} (the embeddings as f32 and encoded, or a feed-forward matrix, an expert's copy and its LoRA product; the adapter)", human(ram))];
            if let Some(f) = free {
                lines.push(format!("{} is free on the disk it goes to{}", human(f), if f < need + (1 << 30) { ": NOT ENOUGH" } else { "" }));
            }
            if *estimate {
                out.event(json!({"event": "estimate", "kind": "export-moe", "lines": lines, "disk_bytes": need, "ram_bytes": ram, "disk_free": free}));
                return Ok((dst.clone(), "info".into(), json!({})));
            }
            if let Some(free) = free {
                if free < need + (1 << 30) {
                    return Err(format!("the export needs about {} on disk and {} is free there", human(need), human(free)));
                }
            }
            fits(ram, "exporting this model")?;
            out.event(json!({"event": "start", "what": format!("export {}{} as a qwen3moe GGUF ({ty}, about {}) to {}", model.display(),
                adapter.as_ref().map_or(String::new(), |a| format!(" with {}", a.display())), human(need), dst.display())}));
            for l in &lines {
                out.event(json!({"event": "log", "message": l}));
            }
            let mut last = Instant::now();
            let r = export_moe_carry(model, adapter.as_deref(), dst, t, pre_tokenizer.as_deref(), &carry_opts, &mut |done, all, name| {
                if last.elapsed().as_secs_f64() >= 2.0 || done == all {
                    last = Instant::now();
                    out.event(json!({"event": "progress", "done": done, "total": all, "message": format!("writing {name}")}));
                }
            }).map_err(|e| format!("{e:#}"))?;
            out.event(json!({"event": "log", "message": format!("{} experts x top-{}: {} parameters, {} active per token; {} of {} blocks carry trained experts",
                r.moe.experts, r.moe.top_k, params_label(r.total_params), params_label(r.active_params), r.trained_blocks, cfg.num_hidden_layers)}));
            if r.carry.state == "carried" {
                out.event(json!({"event": "log", "message": format!("abliteration carried: {} writer tensors had the adapter's change projected off the refusal direction", r.carry.projected.len())}));
            }
            Ok((r.out, "gguf".into(), json!({"experts": r.moe.experts, "top_k": r.moe.top_k, "total_parameters": r.total_params,
                                              "active_parameters": r.active_params, "trained_blocks": r.trained_blocks, "abliteration": r.carry.to_json()})))
        }
        Cmd::Regress { model, adapter, passes, suite, out: dst, against, devices, base_dtype, max_prompt_tokens, judge_model, force, engine } => {
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
            o.force = *force;
            o.engine = engine.spec()?;
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
                          lr, epochs, max_steps, rank, train_layers, max_len, accumulation, max_prompt_tokens, judge_model, engine } => {
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
            o.engine = engine.spec()?;
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
        Cmd::PrivateData { name, sources, model, data_dir, text_count, per_chunk, chunk_chars, temperature, every_s, max_frames } => {
            use strata_workshop::private_data as pd;
            pd::check_name(name).map_err(|e| e.to_string())?;
            if let Some(m) = model {
                pd::refuse_remote_name(m).map_err(|e| e.to_string())?;
            }
            // every folder's attestation before the server is asked anything
            for d in sources {
                pd::read_source(d).map_err(|e| e.to_string())?;
            }
            let data = match data_dir.strip_prefix("~/") {
                Some(rest) => std::env::var_os("HOME").map(PathBuf::from).ok_or("no HOME for ~")?.join(rest),
                None => PathBuf::from(data_dir),
            };
            let mut s = pd::Settings::new(name, sources.clone(), pd::private_root(&data));
            s.text_count = *text_count;
            s.per_chunk = *per_chunk;
            s.chunk_chars = *chunk_chars;
            s.temperature = *temperature;
            s.every_s = every_s.max(0.1);
            s.max_frames = (*max_frames).clamp(1, 64);
            s.media_exe = pd::find_media_exe();
            let has_media = sources.iter().any(|d| walk_has_media(d));
            let mut local = pd::Local::from_env().map_err(|e| e.to_string())?;
            local.choose(model.as_deref(), has_media).map_err(|e| e.to_string())?;
            out.event(json!({"event": "start", "what": format!("private dataset {name} with {} into {}", local.model, s.out_dir().display())}));
            if s.media_exe.is_none() {
                out.event(json!({"event": "log", "message": "no media engine (am2strata-media) found: videos will be skipped and counted"}));
            }
            #[cfg(unix)]
            unsafe { libc::signal(libc::SIGTERM, on_term as extern "C" fn(libc::c_int) as libc::sighandler_t); }
            let r = pd::run(&s, &local,
                |done, total, msg| out.event(json!({"event": "progress", "done": done, "total": total, "message": msg})),
                || TERM.load(std::sync::atomic::Ordering::SeqCst)).map_err(|e| format!("{e:#}"))?;
            let dropped: usize = r.dropped.iter().filter(|(g, _)| **g != pd::gate::REAL_PERSON).map(|(_, n)| n).sum();
            out.event(json!({"event": "log", "message": format!("{} vision rows, {} generation prompts, {} text rows; {} items dropped by the gates, {} hashes blocked; card: {}",
                r.vision_rows, r.prompt_rows, r.text_rows, dropped, r.blocked.len(), s.out_dir().join(pd::CARD).display())}));
            Ok((s.out_dir(), "private-dataset".into(), json!({"private": true, "vision_rows": r.vision_rows, "prompt_rows": r.prompt_rows,
                "text_rows": r.text_rows, "dropped": r.dropped, "blocked": r.blocked.len()})))
        }
        Cmd::PrivateImport { src, name, attestation, confirm_bytes, minor_share, id, check_timeout_s, data_dir } => {
            use strata_workshop::private_import as pi;
            let root = strata_workshop::private_data::private_root(&home_path(data_dir)?);
            // a resume takes the source its import started with (kept in quarantine)
            let spec = match (src.spec(), id) {
                (Ok(sp), _) => sp,
                (Err(_), Some(_)) => pi::Spec::Folder(PathBuf::new()),
                (Err(e), None) => return Err(e),
            };
            let mut s = pi::Settings::new(root.clone(), name, spec.clone());
            if let Some(i) = id { s.id = i.clone(); }
            s.attestation = attestation.clone();
            s.confirm_bytes = *confirm_bytes;
            s.minor_share = *minor_share;
            s.timeout = std::time::Duration::from_secs((*check_timeout_s).clamp(1, 3600));
            // a folder never needs it (a bad HF_ENDPOINT then only stops a network source)
            let net = pi::Net::from_env().ok();
            out.event(json!({"event": "start", "what": if id.is_some() { format!("private import {} resumed", s.id) } else { format!("private import {} of {} into quarantine", s.id, spec.describe()) }}));
            install_term_handler();
            let models = root.join(pi::CHECK_MODELS);
            let timeout = s.timeout;
            let mut loader = |k: pi::checks::CheckKind| pi::checks::load(k, &models, timeout);
            let mut progress = |c: &pi::Counts, msg: &str| out.event(json!({"event": "progress", "done": c.checked, "total": c.items.max(c.checked),
                "message": msg, "counts": c, "import": s.id}));
            let stop = || TERM.load(std::sync::atomic::Ordering::SeqCst);
            match pi::run(&s, net.as_ref(), &mut loader, &mut progress, &stop).map_err(|e| format!("{e:#}"))? {
                pi::Outcome::Released { dir, counts } => {
                    out.event(json!({"event": "progress", "done": counts.checked, "total": counts.items, "message": "released", "counts": counts, "import": s.id}));
                    Ok((dir, "private-dataset".into(), json!({"private": true, "import": s.id, "status": "released", "counts": counts})))
                }
                pi::Outcome::Rejected { reason, counts } | pi::Outcome::Stopped { reason, counts } => {
                    out.event(json!({"event": "progress", "done": counts.checked, "total": counts.items, "message": "nothing released", "counts": counts, "import": s.id}));
                    Err(reason)
                }
            }
        }
        Cmd::PrivateImportProbe { src } => {
            use strata_workshop::private_import as pi;
            let spec = src.spec()?;
            let (bytes, files, commit) = match &spec {
                pi::Spec::Folder(d) => folder_size(d),
                _ => {
                    let p = pi::Net::from_env().and_then(|n| n.probe(&spec)).map_err(|e| format!("{e:#}"))?;
                    (p.bytes, p.files.len(), p.commit)
                }
            };
            out.event(json!({"event": "probe", "what": spec.describe(), "bytes": bytes, "files": files, "commit": commit, "network": spec.is_network()}));
            Ok((PathBuf::new(), "info".into(), json!({})))
        }
        Cmd::PrivateImportDiscard { id, data_dir } => {
            let root = strata_workshop::private_data::private_root(&home_path(data_dir)?);
            strata_workshop::private_import::discard(&root, id).map_err(|e| format!("{e:#}"))?;
            Ok((root.join("quarantine"), "discarded".into(), json!({"import": id})))
        }
        Cmd::RefusalCheck { engine, tokenizer, harmful, harmless, n, max_tokens, before, report } => {
            use strata_workshop::{abliterate, refusal_check};
            let spec = engine.spec()?.ok_or("refusal-check needs --engine-gguf: the GGUF or Kosha pack to check")?;
            let read = |p: &Path| abliterate::read_prompt_list(p).map_err(|e| format!("{}: {e:#}", p.display()));
            let mut spec = spec;
            spec.log = report.as_ref().map(|r| r.with_extension("engine.log"));
            let o = refusal_check::Options { engine: spec, tokenizer: tokenizer.clone(), harmful: read(harmful)?, harmless: read(harmless)?,
                                             n: *n, max_tokens: *max_tokens, before: before.clone() };
            out.event(json!({"event": "start", "what": format!("refusal check of {} through the Rust engine", o.engine.gguf.display())}));
            let r = refusal_check::run(&o, &mut |v| out.event(v)).map_err(|e| format!("{e:#}"))?;
            let rate = |c: &abliterate::CheckResult, k: usize| if c.n == 0 { 0.0 } else { 100.0 * k as f64 / c.n as f64 };
            let line = |label: &str, f: &refusal_check::FileResult| format!("{label}: {:.0} % of harmful and {:.0} % of harmless prompts refused ({} each, {:.1} tokens/s)",
                rate(&f.check, f.check.harmful_refused), rate(&f.check, f.check.harmless_refused), f.check.n, f.speed.tokens_per_second());
            if let Some(b) = &r.before {
                out.event(json!({"event": "log", "message": line("before", b)}));
            }
            out.event(json!({"event": "log", "message": line("this file", &r.after)}));
            let v = r.to_json();
            let path = match report {
                Some(p) => {
                    if let Some(d) = p.parent().filter(|d| !d.as_os_str().is_empty()) {
                        std::fs::create_dir_all(d).map_err(|e| format!("{}: {e}", d.display()))?;
                    }
                    std::fs::write(p, serde_json::to_string_pretty(&v).unwrap_or_default()).map_err(|e| format!("{}: {e}", p.display()))?;
                    p.clone()
                }
                None => o.engine.gguf.clone(),
            };
            Ok((path, "report".into(), json!({"refusal": v})))
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
        Cmd::Train { .. } | Cmd::FindLr { .. } | Cmd::Profile { .. } | Cmd::Autopilot { .. } | Cmd::Abliterate { .. } | Cmd::Upcycle { .. } | Cmd::ExportMoe { .. } | Cmd::Prompts { .. } | Cmd::Regress { .. } | Cmd::GradeTrain { .. } | Cmd::PrivateData { .. } | Cmd::PrivateImport { .. } | Cmd::PrivateImportProbe { .. } | Cmd::PrivateImportDiscard { .. } | Cmd::RefusalCheck { .. } => unreachable!("handled by run"),
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
                     lang_match, drop_cut, exclude, cap_per_language, cap_only, seed, report } => {
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
            s.drop_cut = *drop_cut;
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
        Cmd::Merge { base, adapter, out: dst, carry } => {
            out.event(json!({"event": "start", "what": format!("merge {} into {} as {}", adapter.display(), base.display(), dst.display())}));
            let opts = carry.plan(base, out)?;
            let mut m = merge::Merger::new(base, adapter, dst);
            m.set_carry(opts);
            let r = m.run().map_err(|e| format!("{e:#}"))?;
            if r.carry.state == "carried" {
                out.event(json!({"event": "log", "message": format!("abliteration carried: {} writer tensors had the adapter's change projected off the refusal direction", r.carry.projected.len())}));
            }
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
    if cli.local_only {
        datagen::set_local_only();
    }
    let out = Out { json: cli.json };
    let t0 = Instant::now();
    // nothing in a private quarantine is read by any command but the import itself (it is not checked yet)
    if !matches!(cli.cmd, Cmd::PrivateImport { .. } | Cmd::PrivateImportProbe { .. } | Cmd::PrivateImportDiscard { .. }) {
        let args: Vec<String> = std::env::args().skip(1).collect();
        if let Some(why) = strata_workshop::private_import::quarantine_args(&args) {
            out.event(json!({"event": "error", "message": why}));
            return ExitCode::from(1);
        }
    }
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
