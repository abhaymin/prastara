//! Dumps TinyInkling's tensor names + ggml shapes and the metadata key count, so they can be diffed against
//! `tiny-fixture-header.txt` (make_fixture.py's spec). Tensor `ne` here is what strata_gguf reports back as dims.
use strata_workshop::testmodel::TinyInkling;

fn main() -> anyhow::Result<()> {
    let m = TinyInkling::default();
    let list = m.tensor_list(false);
    println!("tensor_count {}", list.len());
    println!("kv_count {}", m.kv().len());
    for (name, ne, q8) in &list {
        let enc = if *q8 { " Q8_0" } else { "" };
        println!("{} {:?}{}", name, ne, enc);
    }
    Ok(())
}
