//! Generate a tiny id->token-string map for exactly the ids used in the golden outputs, so the frontend test needs no 16 GB file.

use serde_json::Value;
use strata_gguf::Gguf;
use strata_tokenizer::Tokenizer;
use std::collections::HashSet;

const MODEL: &str = "/home/abhaym/Development/models/inkling2-mini-uncensored/Inkling-2-Mini-Uncensored-AD-Q4_K.gguf";
const GOLDEN: &str = "/home/abhaym/Development/am2strata-promo/documentary/code/inkling-2/golden/outputs.json";
const OUT: &str = "/home/abhaym/Development/am2strata-inkling/crates/strata-frontend/tests/data/inkling-tokens.json";
const SCRATCH: &str = "/home/abhaym/Development/am2strata-promo/documentary/code/inkling-2/scratch/inkling-tokenizer-for-fixture";

fn ids(v: &Value) -> Vec<u32> { v.as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect() }

#[test]
#[ignore = "regenerates crates/strata-frontend/tests/data/inkling-tokens.json from the 16 GB model file; run by hand"]
fn make_fixture() {
    let gguf = Gguf::open(std::path::Path::new(MODEL)).expect("open gguf");
    std::fs::create_dir_all(SCRATCH).expect("make scratch");
    let _ = strata_gguf::extract_tokenizer(&gguf, std::path::Path::new(SCRATCH)).expect("extract tokenizer");
    let tok = Tokenizer::from_dir(format!("{SCRATCH}/tokenizer")).expect("load tokenizer");

    let g: Value = serde_json::from_str(&std::fs::read_to_string(GOLDEN).unwrap()).unwrap();
    let mut used = HashSet::new();
    for c in g.as_object().unwrap().values() {
        for &id in ids(&c["tokens"]).iter() { used.insert(id); }
    }
    let mut map = serde_json::Map::new();
    let mut n = 0;
    for id in used.iter().copied() {
        match tok.token_bytes(id) {
            Ok(b) if (id as usize) < tok.vocab_size() => {
                map.insert(id.to_string(), serde_json::Value::String(String::from_utf8_lossy(&b).into_owned()));
                n += 1;
            }
            _ => {}
        }
    }
    std::fs::write(OUT, serde_json::to_string_pretty(&Value::Object(map)).unwrap()).expect("write fixture");
    println!("wrote {n} entries to {OUT}");
}
