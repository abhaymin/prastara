//! Prastara v2 media generation: see src/bin/am2strata-media.rs (the engine process) and docs/V2-PLAN.md.
#[allow(dead_code)]
/// Music facts over mono samples (analyze.rs): tempo, repeat period, tonic, range and phrases. The library exposes it so
/// `sing::check` can reuse its pitch picker; the media engine also runs it as the `analyze` op. Compiled separately from the
/// engine's own inline copy, which is why the rest of its helpers stay quiet here.
pub(crate) mod analyze;
pub mod sing;
