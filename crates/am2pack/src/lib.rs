// SPDX-License-Identifier: AGPL-3.0-only
//! The `.am2aipac` model pack (format version 1; the format is written down in docs/PACK-FORMAT.md).
//!
//! One file holds a model, its image projector and optional extras, laid out for a mixture-of-experts engine:
//! the tensors every token uses are one sequential run (`core`), and each (layer, expert) is one record holding all
//! of that expert's matrices together (`experts`), hottest first when a profile is given.
//!
//! - [`Writer`]: the container. Sections are byte streams; it knows nothing about where the bytes come from.
//! - [`build`]: a pack from a GGUF model (and its `mmproj`), tensor bytes copied unchanged.
//! - [`Pack`]: the memory-mapped reader, [`Pack::verify`] and [`unpack`], which gives the source files back
//!   byte for byte.

mod build;
pub mod gguf;
mod reader;
mod writer;

pub use build::{build, BuildOptions, Summary};
pub use reader::{unpack, Pack};
pub use writer::Writer;

use serde::{Deserialize, Serialize};

/// The first 8 bytes of a pack.
pub const MAGIC: &[u8; 8] = b"AM2AIPAC";
/// The format version this crate writes and reads.
pub const VERSION: u32 = 1;
/// The fixed header's length; the first section starts at or after it.
pub const HEADER_LEN: u64 = 4096;
/// Small sections start on a multiple of this.
pub const ALIGN_SECTION: u64 = 4096;
/// The tensor sections (`core`, `experts`, `image`) start on a multiple of this (one huge page).
pub const ALIGN_LARGE: u64 = 2 << 20;
/// Every tensor, and every part of an expert record, starts on a multiple of this.
pub const ALIGN_TENSOR: u64 = 64;
/// Every expert record starts on a multiple of this (one page: a cold expert is read with whole-page I/O).
pub const ALIGN_RECORD: u64 = 4096;

pub type Result<T> = std::result::Result<T, String>;

/// Section kinds this version writes. A reader skips kinds it does not know.
pub mod kind {
    /// a GGUF source's header, verbatim: its settings, tokenizer, chat format and tensor table
    pub const GGUF_HEADER: &str = "gguf-header";
    /// the tensors every token uses, in the order the layers run
    pub const CORE: &str = "core";
    /// one record per (layer, expert)
    pub const EXPERTS: &str = "experts";
    /// the image encoder and projector's tensors
    pub const IMAGE: &str = "image";
    /// bytes of a source that are neither header nor tensor and are not zero
    pub const GAPS: &str = "gaps";
    /// a file kept whole: the importance matrix, the expert profile, the run defaults
    pub const IMATRIX: &str = "imatrix";
    pub const PROFILE: &str = "profile";
    pub const DEFAULTS: &str = "defaults";
}

/// The index: everything a reader needs to find a byte. Stored as JSON at the end of the file.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Index {
    pub format: String,
    pub version: u32,
    pub manifest: Manifest,
    pub sections: Vec<Section>,
    pub sources: Vec<Source>,
    pub tensors: Vec<Tensor>,
    pub experts: Vec<ExpertRecord>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Manifest {
    pub name: String,
    /// `general.architecture` of the model
    pub architecture: String,
    /// the program and version that wrote the pack
    pub created_by: String,
    /// "profile" when the expert records follow a profile's ranking, else "layer"
    pub expert_order: String,
    /// experts per layer (0 when the model has none)
    pub expert_count: u32,
    /// anything else: the model's licence name, notes. Readers ignore keys they do not know.
    #[serde(default, skip_serializing_if = "serde_json::Map::is_empty")]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

/// A contiguous run of the file with its own checksum. Offsets are from the start of the file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Section {
    pub kind: String,
    /// which source it belongs to ("model", "mmproj"), or a file name
    pub name: String,
    pub off: u64,
    pub len: u64,
    /// SHA-256 of the section's bytes, lower-case hex
    pub sha256: String,
}

/// A file the pack was built from, and what is needed to write it back.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Source {
    /// "model", "mmproj", "imatrix", "profile", "defaults"
    pub role: String,
    /// the file's name (no directory)
    pub file: String,
    pub size: u64,
    pub sha256: String,
    /// a GGUF source: the section holding its header (bytes 0..data_start of the file)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub header: Option<usize>,
    /// a file kept whole: the section holding it
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub whole: Option<usize>,
    /// bytes outside the header and the tensors that are not zero (padding that is zero is not stored)
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub gaps: Vec<Gap>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Gap {
    /// where the bytes sit in the source file
    pub src_off: u64,
    pub len: u64,
    /// where they sit in the pack
    pub off: u64,
}

/// One tensor of a GGUF source.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tensor {
    /// index into `sources`
    pub src: usize,
    pub name: String,
    /// the GGML type's name ("Q4_K", "F16", ...)
    #[serde(rename = "type")]
    pub ty: String,
    /// as in the GGUF: innermost dimension first
    pub dims: Vec<u64>,
    /// where its data began in the source file
    pub src_off: u64,
    /// bytes of data (the whole tensor, also when it is split)
    pub len: u64,
    /// where it sits in the pack; absent when the tensor is split over the expert records
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub off: Option<u64>,
    /// a stacked expert tensor split into this many equal slices, one in each of its layer's expert records
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub split: Option<u32>,
}

/// One (layer, expert): all of that expert's matrices, next to each other.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpertRecord {
    pub layer: u32,
    pub expert: u32,
    /// the record's start and length in the pack (its parts lie inside)
    pub off: u64,
    pub len: u64,
    pub parts: Vec<Part>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Part {
    /// index into `tensors`: the stacked tensor this is slice `expert` of
    pub tensor: usize,
    pub off: u64,
    pub len: u64,
}

pub(crate) fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

pub(crate) fn up(x: u64, a: u64) -> u64 {
    x.div_ceil(a) * a
}
