//! `cargo run --release -p strata-workshop --example write-inkling-fixtures -- <dir>`: writes the default and the wide
//! TinyInkling GGUF (tiny-inkling.gguf, tiny-inkling-wide.gguf) into <dir>, for llama.cpp (`oracle-inkling`) and the
//! engine tests. Both presets have the same architecture; only the sizes differ.
use std::path::PathBuf;

use strata_workshop::testmodel::TinyInkling;

fn main() -> anyhow::Result<()> {
    let out = PathBuf::from(std::env::args().nth(1).ok_or_else(|| anyhow::anyhow!("usage: write-inkling-fixtures <dir>"))?);
    std::fs::create_dir_all(&out)?;
    let def = out.join("tiny-inkling.gguf");
    let wide = out.join("tiny-inkling-wide.gguf");
    TinyInkling::default().write_gguf(&def)?;
    TinyInkling::wide().write_gguf(&wide)?;
    println!("wrote {} and {}", def.display(), wide.display());
    Ok(())
}
