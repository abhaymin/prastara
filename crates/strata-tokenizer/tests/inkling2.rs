//! The Inkling 2 Mini tokenizer against token ids recorded from llama.cpp on `golden/tokenize.json`. Loads the
//! vocabulary straight from the model's GGUF header (as the production path does) into a scratch folder, so it needs
//! the 16 GB model file: run with `-- --ignored` to exercise it.

use serde_json::Value;
use strata_tokenizer::Tokenizer;

const MODEL: &str = "/home/abhaym/Development/models/inkling2-mini-uncensored/Inkling-2-Mini-Uncensored-AD-Q4_K.gguf";
const SCRATCH: &str = "/home/abhaym/Development/am2strata-promo/documentary/code/inkling-1/scratch/inkling2-tokenizer";

fn ids(v: &Value) -> Vec<u32> {
    v.as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect()
}

#[test]
#[ignore] // needs the 16 GB model file at MODEL
fn inkling2_matches_llamacpp() {
    let path = std::path::Path::new(MODEL);
    if !path.exists() {
        eprintln!("skipped: no model file at {MODEL}");
        return;
    }
    let gguf = strata_gguf::Gguf::open(path).expect("open gguf");
    // The tokenizer's own text (tokenizer.chat_template) is in the GGUF header; the golden prompt test uses its copy.
    eprintln!("inkling2: tokenizer.ggml.model={:?} pre={:?}", gguf.str("tokenizer.ggml.model"), gguf.str("tokenizer.ggml.pre"));
    std::fs::create_dir_all(SCRATCH).expect("make scratch");
    strata_gguf::extract_tokenizer(&gguf, std::path::Path::new(SCRATCH)).expect("extract tokenizer");
    let tok = Tokenizer::from_dir(format!("{SCRATCH}/tokenizer")).expect("load tokenizer");

    let g: Value = serde_json::from_str(&std::fs::read_to_string(format!("{}/testdata/tokenize-inkling2.json", env!("CARGO_MANIFEST_DIR"))).unwrap()).unwrap();
    let Value::Array(cases) = &g else { panic!("golden/tokenize-inkling2.json is not an array") };
    let mut failures = Vec::new();
    for (i, c) in cases.iter().enumerate() {
        let text = c["text"].as_str().unwrap();
        let ps = c.get("parse_special").and_then(Value::as_bool).unwrap_or(true);
        match tok.encode(text, ps) {
            Ok(got) => {
                if got != ids(&c["ids"]) {
                    let at = got.iter().zip(ids(&c["ids"])).position(|(a, b)| a != &b).unwrap_or(got.len().min(ids(&c["ids"]).len()));
                    failures.push(format!("case {i} {text:?}: first difference at token {at} (got {} ids, want {})", got.len(), ids(&c["ids"]).len()));
                }
            }
            Err(e) => failures.push(format!("case {i} {text:?}: {e}")),
        }
    }
    assert!(failures.is_empty(), "inkling2: {} mismatches:\n{}", failures.len(), failures.join("\n"));
}
