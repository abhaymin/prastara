//! Abliteration that survives training: the refusal direction r is saved beside an abliterated model
//! ([`DIRECTION_FILE`]), and every later step that writes new weights from it — `merge` (a LoRA folded into the base)
//! and `export-moe` (each expert's LoRA folded into its own copy of the feed-forward) — removes r again from what the
//! adapter added to a matrix that writes the residual stream.
//!
//! A LoRA trained on an abliterated base can write r back in through any writer it targets (o_proj, a down_proj, an
//! expert's down LoRA). The carry projects the adapter's change Δ of each such matrix: `W' = W + Δ − s·r(rᵀΔ)` along the
//! hidden axis ([`crate::abliterate::role`]: o_proj, down_proj, fused or per-expert down tensors, the embeddings). The
//! base's own weights are not touched again, so a partial abliteration (`scale` < 1) stays as it was made, and at
//! scale 1 the merged writer is orthogonal to r exactly as the abliterated base was. Readers (q, k, v, gate, up, the
//! router, the output head) are left alone, as abliterate leaves them.
//!
//! An Avarta loop's gates are not weights of this kind: they mix two passes channel by channel, which no weight edit
//! keeps orthogonal (see [`crate::abliterate`]); trained gates are carried as they are.
//!
//! The marks: the output's config keeps `"prastara_abliterated"` with `"carried": true` (and `"prastara_private"`), and
//! a GGUF made from it carries `prastara.abliterated.carried`. `--no-abliterate-carry` skips the projection and drops
//! the abliterated mark (the private mark stays); `--allow-unprojected` proceeds without a direction file and marks it
//! `"carried": false`.

use std::io::Write as _;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context as _, Result};
use serde_json::{json, Value};

use crate::abliterate::{project_out, role, Role};
use crate::safetensors_io::{Dtype, Shard};

/// The direction's file in an abliterated model folder (linked into an upcycled or looped copy of it).
pub const DIRECTION_FILE: &str = "refusal_direction.safetensors";

/// A saved refusal direction.
#[derive(Debug, Clone, PartialEq)]
pub struct Direction {
    /// r, unit length, `hidden` values
    pub r: Vec<f32>,
    /// how much of it was removed (abliterate's `--scale`)
    pub scale: f32,
    /// the site it was read at
    pub layer: usize,
    pub pass: usize,
    /// its separation on the prompts it was computed from
    pub separation: f64,
    /// `abliterate` (saved by the run that removed it) or `direction-only` (recomputed afterwards)
    pub source: String,
}

/// Write `d` as `dir/`[`DIRECTION_FILE`]: tensors `direction` [hidden] and `scale` [1], both F32, and the site in the
/// header's `__metadata__`.
pub fn save(dir: &Path, d: &Direction) -> Result<PathBuf> {
    let path = dir.join(DIRECTION_FILE);
    let n = d.r.len();
    let meta = json!({"format": "pt", "layer": d.layer.to_string(), "pass": d.pass.to_string(), "scale": d.scale.to_string(),
                      "separation": d.separation.to_string(), "source": d.source, "hidden": n.to_string()});
    let header = json!({"__metadata__": meta,
                        "direction": {"dtype": "F32", "shape": [n], "data_offsets": [0, 4 * n]},
                        "scale": {"dtype": "F32", "shape": [1], "data_offsets": [4 * n, 4 * n + 4]}});
    let mut h = serde_json::to_vec(&header)?;
    while h.len() % 8 != 0 {
        h.push(b' ');
    }
    let tmp = dir.join(format!("{DIRECTION_FILE}.part"));
    {
        let mut f = std::fs::File::create(&tmp).with_context(|| format!("{}", tmp.display()))?;
        f.write_all(&(h.len() as u64).to_le_bytes())?;
        f.write_all(&h)?;
        for x in &d.r {
            f.write_all(&x.to_le_bytes())?;
        }
        f.write_all(&d.scale.to_le_bytes())?;
        f.sync_all()?;
    }
    // over a link (an upcycled copy's file points at its base's), never through it
    if path.symlink_metadata().is_ok() {
        std::fs::remove_file(&path)?;
    }
    std::fs::rename(&tmp, &path)?;
    Ok(path)
}

/// Read a direction file back.
pub fn load(path: &Path) -> Result<Direction> {
    let p = path.display();
    let s = Shard::open(path).with_context(|| format!("{p}: not a direction file"))?;
    let f32s = |name: &str| -> Result<Vec<f32>> {
        let t = s.get(name).ok_or_else(|| anyhow!("{p}: no tensor {name:?}"))?;
        if t.dtype() != Dtype::F32 {
            bail!("{p}: {name} is {}, not F32", t.dtype().as_str());
        }
        Ok(s.bytes(t).chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
    };
    let r = f32s("direction")?;
    let scale = *f32s("scale")?.first().ok_or_else(|| anyhow!("{p}: an empty scale"))?;
    // the site, from the header's metadata
    let bytes = std::fs::read(path)?;
    let len = u64::from_le_bytes(bytes.get(..8).ok_or_else(|| anyhow!("{p}: too short"))?.try_into()?) as usize;
    let header: Value = serde_json::from_slice(bytes.get(8..8 + len).ok_or_else(|| anyhow!("{p}: a cut header"))?)?;
    let m = &header["__metadata__"];
    let num = |k: &str| m[k].as_str().and_then(|v| v.parse::<f64>().ok());
    let norm = r.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>().sqrt();
    if r.is_empty() || (norm - 1.0).abs() > 1e-3 {
        bail!("{p}: the direction is not of unit length (|r| = {norm})");
    }
    Ok(Direction { r, scale, layer: num("layer").ok_or_else(|| anyhow!("{p}: no layer in its metadata"))? as usize,
                   pass: num("pass").unwrap_or(0.0) as usize, separation: num("separation").unwrap_or(f64::NAN),
                   source: m["source"].as_str().unwrap_or("").to_string() })
}

/// The direction file of a model folder: its own, or — for a copy whose weights are links into an abliterated folder
/// (upcycle, loop, context) — the one beside the weights it links to.
pub fn find(dir: &Path) -> Option<PathBuf> {
    let own = dir.join(DIRECTION_FILE);
    if own.is_file() {
        return Some(own);
    }
    let weights = crate::model::weight_files(dir).ok()?;
    for w in weights {
        if !w.symlink_metadata().is_ok_and(|m| m.file_type().is_symlink()) {
            continue;
        }
        if let Some(f) = std::fs::canonicalize(&w).ok().and_then(|t| t.parent().map(|p| p.join(DIRECTION_FILE))).filter(|f| f.is_file()) {
            return Some(f);
        }
    }
    None
}

/// What a merge or an export is asked to do about an abliterated base.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Mode {
    /// project the adapter's changes (the default); refuse when the direction file is missing
    #[default]
    Carry,
    /// `--allow-unprojected`: carry when the file is there, else go on and mark the result `"carried": false`
    AllowUnprojected,
    /// `--no-abliterate-carry`: no projection; the result loses its abliterated mark
    Skip,
}

#[derive(Debug, Clone, Default)]
pub struct Options {
    pub mode: Mode,
    /// a direction file named outright (`--direction`), instead of the one found beside the base
    pub direction: Option<PathBuf>,
}

/// What happens, decided from the base's config before anything is written.
#[derive(Debug, Clone)]
pub enum Plan {
    /// the base is not abliterated: nothing to carry
    None,
    /// project every writer's change against this direction (from this file)
    Project(Direction, PathBuf),
    /// abliterated, no direction file, `--allow-unprojected`
    Unprojected,
    /// `--no-abliterate-carry`
    Dropped,
}

impl Plan {
    pub fn state(&self) -> &'static str {
        match self {
            Plan::None => "none",
            Plan::Project(..) => "carried",
            Plan::Unprojected => "unprojected",
            Plan::Dropped => "dropped",
        }
    }

    pub fn direction(&self) -> Option<&Direction> {
        match self {
            Plan::Project(d, _) => Some(d),
            _ => None,
        }
    }

    /// The lines a command prints about it (the loud warning for a dropped or unprojected carry).
    pub fn lines(&self) -> Vec<String> {
        match self {
            Plan::None => vec![],
            Plan::Project(d, f) => vec![format!("the base is abliterated (block {} pass {}, scale {}): the adapter's changes to every matrix that writes the residual stream are projected off the refusal direction ({})",
                                                d.layer, d.pass, d.scale, f.display())],
            Plan::Unprojected => vec!["WARNING: the base is abliterated but has no refusal direction file, and --allow-unprojected was given: the adapter's changes are NOT projected, so training may have written the refusal direction back in. The result is marked \"carried\": false.".into()],
            Plan::Dropped => vec!["WARNING: --no-abliterate-carry: the adapter's changes are NOT projected off the refusal direction, so the result may refuse again. Its abliterated mark is dropped (it stays private).".into()],
        }
    }
}

/// The command that recomputes a missing direction, for messages.
pub fn direction_only_hint(abliterated: &Path, meta: &Value) -> String {
    let layer = meta["layer"].as_u64().map_or("N".to_string(), |l| l.to_string());
    let pass = meta["pass"].as_u64().unwrap_or(0);
    format!("am2strata-workshop abliterate --direction-only <the original (un-abliterated) model folder> --harmful <the harmful list> --harmless <the harmless list> \
             --layer {layer} --pass {pass} --out {}", abliterated.display())
}

/// Decide the plan for a base folder `dir` whose config is `cfg`.
pub fn plan(dir: &Path, cfg: &Value, o: &Options) -> Result<Plan> {
    let meta = &cfg["prastara_abliterated"];
    if !meta.is_object() {
        if o.direction.is_some() {
            bail!("--direction was given but {} is not marked abliterated (no prastara_abliterated in its config.json)", dir.display());
        }
        return Ok(Plan::None);
    }
    if o.mode == Mode::Skip {
        return Ok(Plan::Dropped);
    }
    let file = o.direction.clone().or_else(|| find(dir));
    let Some(file) = file else {
        if o.mode == Mode::AllowUnprojected {
            return Ok(Plan::Unprojected);
        }
        // where the weights really are: the folder the direction belongs in
        let home = crate::model::weight_files(dir).ok().and_then(|w| w.first().and_then(|f| std::fs::canonicalize(f).ok()))
            .and_then(|f| f.parent().map(Path::to_path_buf)).unwrap_or_else(|| dir.to_path_buf());
        bail!("{} is marked abliterated (block {} pass {}) but has no {DIRECTION_FILE}, so what the adapter adds cannot be kept off the refusal \
               direction. Recompute it with\n  {}\nthen run this again; or pass --allow-unprojected (the result is marked \"carried\": false) or \
               --no-abliterate-carry (the abliterated mark is dropped)",
              dir.display(), meta["layer"], meta["pass"].as_u64().unwrap_or(0), direction_only_hint(&home, meta));
    };
    let d = load(&file)?;
    let hidden = cfg["hidden_size"].as_u64().ok_or_else(|| anyhow!("{}: config.json has no hidden_size", dir.display()))? as usize;
    if d.r.len() != hidden {
        bail!("{}: the direction has {} values, the model is {hidden} wide", file.display(), d.r.len());
    }
    let (layer, pass) = (meta["layer"].as_u64(), meta["pass"].as_u64().unwrap_or(0));
    if layer.is_some_and(|l| l as usize != d.layer) || pass as usize != d.pass {
        bail!("{}: the direction is from block {} pass {}, the model was abliterated at block {} pass {pass}", file.display(), d.layer, d.pass, meta["layer"]);
    }
    if let Some(s) = meta["scale"].as_f64() {
        if (s - d.scale as f64).abs() > 1e-4 {
            bail!("{}: the direction's scale is {}, the model was abliterated at {s}", file.display(), d.scale);
        }
    }
    Ok(Plan::Project(d, file))
}

/// The output config's marks for `plan` (on a copy of the base's config).
pub fn mark(cfg: &mut Value, plan: &Plan) {
    match plan {
        Plan::None => {}
        Plan::Project(..) | Plan::Unprojected => {
            cfg["prastara_abliterated"]["carried"] = json!(matches!(plan, Plan::Project(..)));
            cfg["prastara_private"] = json!(true);
        }
        Plan::Dropped => {
            if let Some(o) = cfg.as_object_mut() {
                o.remove("prastara_abliterated");
            }
            cfg["prastara_private"] = json!(true);
        }
    }
}

/// Rewrite `out/config.json` with [`mark`], and put the direction beside the result (a further merge carries it on).
pub fn finish_folder(out: &Path, plan: &Plan) -> Result<()> {
    if matches!(plan, Plan::None) {
        return Ok(());
    }
    let p = out.join("config.json");
    let mut cfg: Value = serde_json::from_str(&std::fs::read_to_string(&p).with_context(|| format!("{}", p.display()))?)?;
    mark(&mut cfg, plan);
    std::fs::write(&p, serde_json::to_string_pretty(&cfg)?)?;
    let dst = out.join(DIRECTION_FILE);
    if dst.symlink_metadata().is_ok() {
        std::fs::remove_file(&dst)?;
    }
    if let Plan::Project(_, f) = plan {
        std::fs::copy(f, &dst).with_context(|| format!("copying {}", f.display()))?;
    }
    Ok(())
}

/// `delta` (the adapter's change of the tensor `name` of `shape`) projected off r along the hidden axis, when the tensor
/// writes the residual stream. Returns whether it did.
pub fn project_delta(name: &str, shape: &[usize], delta: &mut [f32], d: &Direction) -> Result<bool> {
    let hidden = d.r.len();
    match role(name, shape, hidden)? {
        Role::Writer { axis } => project_out(delta, shape, axis, &d.r, d.scale),
        Role::Embed => project_out(delta, shape, 1, &d.r, d.scale),
        Role::Untouched => return Ok(false),
    }
    Ok(true)
}

/// `after` = `before` + the change between them projected off r, for a writer `name` of `shape` (see
/// [`project_delta`]). Returns whether it was a writer.
pub fn project_change(name: &str, shape: &[usize], before: &[f32], after: &mut [f32], d: &Direction) -> Result<bool> {
    let mut delta: Vec<f32> = after.iter().zip(before).map(|(a, b)| a - b).collect();
    if !project_delta(name, shape, &mut delta, d)? {
        return Ok(false);
    }
    after.iter_mut().zip(before.iter().zip(&delta)).for_each(|(a, (b, x))| *a = b + x);
    Ok(true)
}

/// What a carry did, for the report.
#[derive(Debug, Clone, Default)]
pub struct Report {
    /// none, carried, unprojected or dropped
    pub state: String,
    /// the writers whose change was projected
    pub projected: Vec<String>,
}

impl Report {
    pub fn to_json(&self) -> Value {
        json!({"state": self.state, "projected": self.projected.len()})
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_direction_saves_and_reads_back() {
        let d = std::env::temp_dir().join(format!("carry-unit-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        let r: Vec<f32> = { let v = [0.3f32, -0.5, 0.2, 0.7, 0.1]; let n = v.iter().map(|x| x * x).sum::<f32>().sqrt(); v.iter().map(|x| x / n).collect() };
        let dir = Direction { r, scale: 0.75, layer: 22, pass: 1, separation: 9.25, source: "abliterate".into() };
        save(&d, &dir).unwrap();
        assert_eq!(load(&d.join(DIRECTION_FILE)).unwrap(), dir);
        let _ = std::fs::remove_dir_all(&d);
    }
}
