# Mula: Prastara's own model series

**Request (owner, 2026-11-25):** a model series named **Mula**. It expands on looping models and on qwen4-based
models, and covers top-tier coding, agentic work, tool calls and MCP, songs, narrative, vision (with OCR and the other
key image tasks) and actions. The models are built with Prastara's own Workshop. Once refined, Mula should be able to
code Prastara itself and grow the app with a stable set of features.

The models are versioned **v0.1** onward. Prastara becomes **v1.0** when Mula v0.1 ships.

**Mula** (मूल, *mūla*) is Sanskrit for root, origin, foundation: the models Prastara grows from.

## Decisions (owner, 2026-11-25)

| Question | Answer |
|---|---|
| Base | A looped **Qwen3-8B** (Apache-2.0); the qwen4 model (Qwen3.8 Flash Next) teaches |
| First models | tool calls and MCP; code and agent; song and narrative; action; vision (with OCR and image features) |
| Prastara v1.0 | ships with Mula v0.1 (v0.2.0 is the current release) |
| Downloads | the base model; open datasets, each listed with its licence for the owner's yes before it is fetched; local models may write training data |

## Why this shape

- **The qwen4 model cannot be trained here.** Qwen3.8 Flash Next (`qwen4exp`) has 177B parameters and is 80 GB even
  at IQ3, more than this PC's GPU and system memory together. It runs live at about 72 tokens/s, so it is the
  **teacher**: it writes training examples, and Mula learns from them (distillation).
- **A looped student.** Mula takes Qwen3-8B's middle layers and runs them more than once with the same weights
  (Avarta). The model thinks deeper for the same size on disk, and the loop's pass count can be lowered per request
  for speed. A loop added this way starts as an exact copy of the base: each extra pass enters through a gate that
  starts at zero, so the first check is that the converted model gives the base's logits exactly.
- **One base, one model per skill.** Each Mula model is the same looped base with its own trained adapter, merged and
  packed (Kosha). The family is stored as one base plus variants (Kula and Rupa in the README).
- **Teachers and licences.** Qwen3.8 Flash Next is Apache-2.0 and Cyber-Tiel Coder is MIT; both allow training on
  their outputs. Every generated example passes the local content gate before it is kept. Public Mula models keep
  the base's safety behaviour; any uncensored variant stays private, as decided for all models.

## The series, v0.1

| Model | Skill | Teacher and data | Gate (measured before release) |
|---|---|---|---|
| Mula Setu (सेतु, bridge) | tool calls and MCP: picking the tool, exact arguments, multi-call turns | Qwen3.8 Flash Next on Prastara's own tool and MCP schemas; open function-calling sets | schema-valid calls and the right tool on a held-out set; Prastara's tool suites |
| Mula Code | coding: Rust first (Prastara's own code), then Python, TS, C++ | Cyber-Tiel Coder and Qwen3.8 Flash Next; open code sets; Prastara's history (commit → diff pairs) | `am2strata-run eval` jobs with locked checkers; tests pass on the repo's own tasks |
| Mula Agent | agentic work: plan, act, check, recover | agent transcripts from Prastara's coding agents and eval jobs | eval job pass rate against the base |
| Mula Gita (गीत, song) | lyrics with structure for the song engine, in English, Hindi and Sanskrit | Qwen3.8 Flash Next; public-domain verse | structure tags valid; the song engine's word check on the takes |
| Mula Katha (कथा, story) | narrative, scripts and narration | Qwen3.8 Flash Next; public-domain stories | judged scripts; length and format kept |
| Mula Karma (कर्म, action) | computer and browser use: read a screen, pick the click, key or tool | Prastara's browser and Computer tab sessions and site tests | the browser test suites and VM tasks |
| Mula Drishti (दृष्टि, sight) | vision: pictures in, OCR, charts, screenshots, documents | an open vision encoder and projector trained in the Workshop; open OCR and captioning sets | OCR accuracy and image questions on a held-out set |

The skill names are proposals for the owner; "Mula" is decided.

## What has to be built first

| Step | What | Gate |
|---|---|---|
| M0 | Qwen3-8B downloaded and checked (licence, files, revision); runs in the Workshop's forward pass | logits equal the reference within the Workshop's parity gate |
| M1 | **Loop conversion** in the Workshop: choose the core layers and the passes; each extra pass enters through a gate that starts at zero | the converted model's logits equal the base's exactly; the gate trains |
| M2 | **GPU training** in the Workshop: the CUDA side of its two custom nodes, the base in BF16 across the two cards, LoRA on the looped model | the GPU run's loss and adapter match the CPU run's on a small model; 8B fits in 2×16 GB with measured peak memory |
| M3 | **Data:** datagen from the teachers through the server queue (only when it is idle), the content gate, de-duplication, a licence log per source, held-out splits | every kept example has a source, a licence and a gate verdict |
| M4 | **Evals** for each skill (the gates in the table above), run on the base first as the baseline | a report per skill |
| M5 | Mula Setu, Code, Agent v0.1: train, merge, export to GGUF and Kosha | each beats the base on its gate; no regressions on the others |
| M6 | Mula Gita, Katha, Karma v0.1 | as above |
| M7 | **Vision** in the Workshop: an open encoder, projector training, OCR; then Mula Drishti v0.1 | as above |
| M8 | **The engine runs Mula fast:** the looped Qwen3 on the GPU graph path, on one card | the same tokens as the Workshop's forward pass; speed reported |
| M9 | Prastara v1.0: Mula v0.1 in the catalog, the Workshop's new steps in the UI and the CLI | the release gates |

Later (v0.2 and on): Mula on a mixture of experts made from the looped base (qwen4-style experts, upcycled), longer
contexts, and Mula refining itself on Prastara's own tasks.

## Status

- 2026-11-25: plan written. Qwen3-8B (revision b968826d, Apache-2.0) is in `Strata-data/models/Qwen3-8B`; every
  shard's size checked against its header.
- **M0 met:** the Workshop's forward pass runs Qwen3-8B in BF16 on the CPU (25 s for a 5-token prompt, 17 GB of RAM)
  and answers "The capital of France is" with " Paris". A logit-level comparison against a reference engine is still
  to do.
- **M1 met:** `am2strata-workshop loop <model> --out <dir> --start S --end E --passes N` writes a looped copy (the
  weights linked, `config.json` with `"prastara_loop"`, the gates in `loop.safetensors`, all zero). On the tiny Qwen3
  fixture and on Qwen3-8B (blocks 12..24, two passes: `Strata-data/mula/mula-base-loop-v0.1`) the copy's logits equal
  the base's bit for bit. The gates get a gradient while closed, train with a LoRA adapter, save in it
  (`base_model.model.model.loop.gate.<p>`), load back, and survive `merge` (`tests/avarta.rs`). Training from a block
  inside the loop is refused.
- **M2, the code:** `am2strata-workshop train --devices cuda:0,cuda:1` (a `cuda` build) splits the blocks over the
  cards. The two custom nodes have their GPU side, and hidden states and their gradients move between cards. Checked
  on a tiny looped model: split over both cards, the logits and every gradient equal the CPU's to 1e-6, and a 6-step
  training run gives the same losses (2.4e-7) and the same adapter (1.5e-8) as on the CPU.
  - Found on the way: candle's direct card-to-card copy is not ordered against the next kernel in the backward pass,
    so gradients came out a few percent wrong while the forward values were right. Moves between two GPUs now go
    through host memory (`model::move_to`).
- **M2, open:** the 8B on the cards. Training runs need the owner's yes for each run (owner, 2026-11-25: dedicated
  runs, approved one at a time), since the cards hold the live model.
  - Estimate, not measured: the BF16 weights are 8.2 GB per card. The activations kept for the backward pass, from
    block 12 up with the loop's second pass, come to about 0.5 GB per block pass at 1,024 tokens, about 18 GB.
    So 1,024-token examples need gradient checkpointing (recompute a block's activations in the backward pass),
    which is the next piece.
