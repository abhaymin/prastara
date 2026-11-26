//! A streaming safetensors writer: a whole checkpoint goes out one tensor at a time, never held in memory.
//!
//! The caller says up front which tensors will be written (name, dtype, shape, in order): from that the writer plans
//! the shards (about [`SHARD_BYTES`] each, a tensor never split), so every shard's header can be written before its
//! first tensor, and then each tensor's bytes are appended as they come. One shard is `model.safetensors`; several are
//! `model-0000i-of-0000n.safetensors` with a `model.safetensors.index.json` whose `weight_map` maps each tensor name to
//! its file name, as transformers and the Workshop's own reader expect. Files are written as `*.part` and renamed when
//! the last tensor is in, so a stopped run leaves no file that looks finished.

use std::collections::BTreeMap;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context as _, Result};
use serde_json::{json, Map, Value};

use crate::safetensors_io::Dtype;

/// The shard size transformers writes by default (`max_shard_size="5GB"`).
pub const SHARD_BYTES: u64 = 5_000_000_000;

/// The index file of a sharded checkpoint.
pub const INDEX_FILE: &str = "model.safetensors.index.json";

/// One tensor to be written.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorSpec {
    pub name: String,
    pub dtype: Dtype,
    pub shape: Vec<usize>,
}

impl TensorSpec {
    pub fn new(name: impl Into<String>, dtype: Dtype, shape: &[usize]) -> Self {
        Self { name: name.into(), dtype, shape: shape.to_vec() }
    }

    pub fn nbytes(&self) -> u64 {
        self.shape.iter().map(|d| *d as u64).product::<u64>() * self.dtype.size_bytes() as u64
    }
}

/// The file names for `n` shards.
pub fn shard_name(i: usize, n: usize) -> String {
    if n == 1 { "model.safetensors".to_string() } else { format!("model-{:05}-of-{:05}.safetensors", i + 1, n) }
}

/// The plan: which tensors go in which shard (contiguous runs of the spec list, in order).
pub fn plan(specs: &[TensorSpec], max_shard_bytes: u64) -> Vec<std::ops::Range<usize>> {
    let mut out = Vec::new();
    let (mut start, mut size) = (0usize, 0u64);
    for (i, s) in specs.iter().enumerate() {
        let n = s.nbytes();
        if i > start && size + n > max_shard_bytes {
            out.push(start..i);
            start = i;
            size = 0;
        }
        size += n;
    }
    if start < specs.len() || specs.is_empty() {
        out.push(start..specs.len());
    }
    out
}

/// The header of one shard: `__metadata__` {format: pt}, and per tensor its dtype, shape and byte span, padded with
/// spaces to a multiple of 8 (as the safetensors crate writes it).
fn header(specs: &[TensorSpec]) -> Vec<u8> {
    let mut m = Map::new();
    m.insert("__metadata__".into(), json!({"format": "pt"}));
    let mut at = 0u64;
    for s in specs {
        let n = s.nbytes();
        m.insert(s.name.clone(), json!({"dtype": s.dtype.as_str(), "shape": s.shape, "data_offsets": [at, at + n]}));
        at += n;
    }
    let mut h = Value::Object(m).to_string().into_bytes();
    while h.len() % 8 != 0 {
        h.push(b' ');
    }
    h
}

/// Writes the planned tensors, in the planned order. See the module docs.
pub struct StWriter {
    dir: PathBuf,
    specs: Vec<TensorSpec>,
    shards: Vec<std::ops::Range<usize>>,
    /// the next tensor to be written (an index into `specs`)
    next: usize,
    /// the shard being written and its file
    current: Option<(usize, BufWriter<std::fs::File>)>,
}

impl StWriter {
    /// Plan the shards of `specs` (about `max_shard_bytes` each) for the folder `dir` (made if missing). Names must be
    /// unique.
    pub fn create(dir: &Path, specs: Vec<TensorSpec>, max_shard_bytes: u64) -> Result<Self> {
        let mut seen = std::collections::HashSet::new();
        for s in &specs {
            if !seen.insert(s.name.as_str()) {
                bail!("tensor {} is listed twice", s.name);
            }
        }
        std::fs::create_dir_all(dir).with_context(|| format!("{}", dir.display()))?;
        let shards = plan(&specs, max_shard_bytes.max(1));
        Ok(Self { dir: dir.to_path_buf(), specs, shards, next: 0, current: None })
    }

    /// The final file names, in order.
    pub fn file_names(&self) -> Vec<String> {
        (0..self.shards.len()).map(|i| shard_name(i, self.shards.len())).collect()
    }

    /// The next tensor the writer expects, if any.
    pub fn expects(&self) -> Option<&TensorSpec> {
        self.specs.get(self.next)
    }

    fn part(&self, i: usize) -> PathBuf {
        self.dir.join(format!("{}.part", shard_name(i, self.shards.len())))
    }

    /// Append the next tensor: `name` must be the next planned one and `bytes` its exact size (little-endian, as the
    /// dtype stores it).
    pub fn write(&mut self, name: &str, bytes: &[u8]) -> Result<()> {
        let spec = self.specs.get(self.next).ok_or_else(|| anyhow!("{name}: every planned tensor is already written"))?;
        if spec.name != name {
            bail!("{name} came where {} was planned (tensors are written in the planned order)", spec.name);
        }
        if bytes.len() as u64 != spec.nbytes() {
            bail!("{name}: {} bytes, its dtype and shape make {}", bytes.len(), spec.nbytes());
        }
        let shard = self.shards.iter().position(|r| r.contains(&self.next)).ok_or_else(|| anyhow!("{name}: in no shard"))?;
        if self.current.as_ref().map(|c| c.0) != Some(shard) {
            self.close_current()?;
            let path = self.part(shard);
            let mut f = BufWriter::with_capacity(8 << 20, std::fs::File::create(&path).with_context(|| format!("{}", path.display()))?);
            let h = header(&self.specs[self.shards[shard].clone()]);
            f.write_all(&(h.len() as u64).to_le_bytes())?;
            f.write_all(&h)?;
            self.current = Some((shard, f));
        }
        let (_, f) = self.current.as_mut().expect("opened above");
        f.write_all(bytes)?;
        self.next += 1;
        Ok(())
    }

    fn close_current(&mut self) -> Result<()> {
        if let Some((_, mut f)) = self.current.take() {
            f.flush()?;
            f.into_inner().map_err(|e| anyhow!("{e}"))?.sync_all()?;
        }
        Ok(())
    }

    /// After the last tensor: the `.part` files take their names and, for several shards, the index is written.
    /// Returns the shard files in order.
    pub fn finish(mut self) -> Result<Vec<PathBuf>> {
        if self.next != self.specs.len() {
            bail!("{} of {} tensors were written (the next was {})", self.next, self.specs.len(), self.specs[self.next].name);
        }
        self.close_current()?;
        let n = self.shards.len();
        if self.specs.is_empty() {
            // nothing planned: an empty, valid single file
            let h = header(&[]);
            let mut b = (h.len() as u64).to_le_bytes().to_vec();
            b.extend_from_slice(&h);
            std::fs::write(self.part(0), b)?;
        }
        let mut files = Vec::with_capacity(n);
        for i in 0..n {
            let dst = self.dir.join(shard_name(i, n));
            std::fs::rename(self.part(i), &dst).with_context(|| format!("{}", dst.display()))?;
            files.push(dst);
        }
        if n > 1 {
            let mut map = BTreeMap::new();
            for (i, r) in self.shards.iter().enumerate() {
                for s in &self.specs[r.clone()] {
                    map.insert(s.name.clone(), shard_name(i, n));
                }
            }
            let total: u64 = self.specs.iter().map(|s| s.nbytes()).sum();
            let index = json!({"metadata": {"total_size": total}, "weight_map": map});
            std::fs::write(self.dir.join(INDEX_FILE), serde_json::to_string_pretty(&index)?)?;
        } else {
            std::fs::remove_file(self.dir.join(INDEX_FILE)).ok();
        }
        Ok(files)
    }
}

impl Drop for StWriter {
    fn drop(&mut self) {
        // an unfinished writer leaves no part files behind
        if self.next != self.specs.len() || self.current.is_some() {
            self.current.take();
            for i in 0..self.shards.len() {
                std::fs::remove_file(self.part(i)).ok();
            }
        }
    }
}

/// Make `out` ready for a checkpoint written from `src`: refused when it is `src` itself; weights an earlier run left
/// there (`model.safetensors`, `model-*.safetensors`, the index) are removed, since they would be read beside the new ones.
pub fn prepare_out(src: &Path, out: &Path) -> Result<()> {
    std::fs::create_dir_all(out).with_context(|| format!("{}", out.display()))?;
    if std::fs::canonicalize(src)? == std::fs::canonicalize(out)? {
        bail!("{}: the output folder must not be the input folder", out.display());
    }
    for e in std::fs::read_dir(out)?.flatten() {
        let name = e.file_name().to_string_lossy().to_string();
        if name == "model.safetensors" || name == INDEX_FILE || (name.starts_with("model-") && name.ends_with(".safetensors")) {
            std::fs::remove_file(e.path())?;
        }
    }
    Ok(())
}

/// A folder's `model.safetensors.index.json` as tensor name -> file name (`None` without an index); an index whose
/// values are not file names is refused.
pub fn read_index(dir: &Path) -> Result<Option<BTreeMap<String, String>>> {
    let p = dir.join(INDEX_FILE);
    if !p.exists() {
        return Ok(None);
    }
    let v: Value = serde_json::from_str(&std::fs::read_to_string(&p)?).with_context(|| format!("{}", p.display()))?;
    let map = v["weight_map"].as_object().ok_or_else(|| anyhow!("{}: no weight_map", p.display()))?;
    map.iter().map(|(k, f)| Ok((k.clone(), f.as_str().ok_or_else(|| anyhow!("{}: weight_map[{k}] is not a file name", p.display()))?.to_string()))).collect::<Result<_>>().map(Some)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_plan_cuts_between_tensors_never_inside_one() {
        let s = |n: &str, k: usize| TensorSpec::new(n, Dtype::F32, &[k]);
        let specs = vec![s("a", 10), s("b", 10), s("c", 30), s("d", 1)];
        // 40, 40, 120, 4 bytes against 80 a shard
        assert_eq!(plan(&specs, 80), vec![0..2, 2..3, 3..4]);
        assert_eq!(plan(&specs, 1000), vec![0..4]);
        assert_eq!(plan(&[], 10), vec![0..0]);
    }

    #[test]
    fn the_header_is_padded_to_eight() {
        let h = header(&[TensorSpec::new("x", Dtype::BF16, &[3])]);
        assert_eq!(h.len() % 8, 0);
        let v: Value = serde_json::from_slice(&h).unwrap();
        assert_eq!(v["x"]["data_offsets"], json!([0, 6]));
        assert_eq!(v["__metadata__"]["format"], "pt");
    }
}
