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

## 1M-token context (owner, 2026-11-25: "increase context limit of the model to 1M and actually make the model accept 1M context")

Accepting 1M tokens means the model still finds and uses what is in them, measured. Saying so in a config is not
enough. Qwen3-8B is trained to 32K (its config allows 40,960 positions).

**Why it cannot be done by a setting alone (Qwen3-8B, 36 layers, 8 key/value heads of 128):**
- **Keys and values:** every layer keeps them for every token, 144 KB per token. At 1M that is 137 GiB, and 183 GiB
  with the loop's second pass. This PC has 32 GB of VRAM and 59 GB of RAM.
- **Training attention:** it keeps a seq x seq table per head. At 64K tokens that is 512 GB for one block.
- **Positions:** beyond 32K the model has never seen them; stretching them (YaRN) needs training at long lengths
  to hold up.

**The design:**

| Piece | What | Effect at 1M tokens |
|---|---|---|
| L1 YaRN positions | `rope_scaling: yarn`, factor 32 over the original 32,768 (the method Qwen documents for 128K), in the Workshop and the engine | positions up to 1,048,576 |
| L2 Hybrid attention | one layer in four keeps full attention (9 of 36); the other 27 attend to a 4,096-token window (`layer_types`, `sliding_window`), then trained to recover | keys and values for 12 full layers (with the loop): 46 GiB in BF16, 23 GiB at 8 bits, about 12 GiB at 4 bits |
| L3 Chunked attention in training | attention computed a block of queries at a time, storing only the output and one number per row; the backward pass recomputes each block (the flash-attention method), with exact gradients | memory grows with the length, not its square: 64K-token examples train |
| L4 Long training | continued training at 16K, then 32K, then 64K, on long documents, code repositories and books, with the hybrid layers and YaRN | the model learns to use the far context |
| L5 The engine at 1M | YaRN, sliding windows, the loop, 8-bit keys and values tiered across VRAM and RAM (Stara), prompts read in chunks | the cache fits (about 23 GiB); a 1M prompt reads in about an hour on these cards (estimate), decoding at a few tokens/s |
| L6 The gates | passkey and needle-in-a-haystack retrieval at 32K, 128K, 256K, 512K and 1M, at several depths, and a multi-needle test; short-context quality kept | a model is called 1M only when the 1M needles are found |

## What has to be built first

| Step | What | Gate |
|---|---|---|
| M0 | Qwen3-8B downloaded and checked (licence, files, revision); runs in the Workshop's forward pass | logits equal the reference within the Workshop's parity gate |
| M1 | **Loop conversion** in the Workshop: choose the core layers and the passes; each extra pass enters through a gate that starts at zero | the converted model's logits equal the base's exactly; the gate trains |
| M2 | **GPU training** in the Workshop: the CUDA side of its two custom nodes, the base in BF16 across the two cards, LoRA on the looped model | the GPU run's loss and adapter match the CPU run's on a small model; 8B fits in 2×16 GB with measured peak memory |
| M3 | **Data:** datagen from the teachers through the server queue (only when it is idle; the Workshop's built-in `code`, `indic` and `translate` recipes, with compile and script checks, plus `code` over Prastara's own source), the content gate, de-duplication, a licence log per source, held-out splits | every kept example has a source, a licence and a gate verdict |
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
    So 1,024-token examples need gradient checkpointing.
- **M2, gradient checkpointing:** `--gradient-checkpointing` keeps only each trained block's input and recomputes
  its activations in the backward pass, with the same dropout masks. Its gradients equal the plain ones to 6e-8 on
  a looped model with dropout, and a checkpointed 6-step run on two GPUs gives the CPU's losses and adapter.
- **M2, memory:** a 16-bit matrix larger than 64 M values (the output head: 152k x 4096) is widened to f32 a slice
  of rows at a time, each row's product as before (2.5 GB to 256 MB). With several devices the last one, which holds
  the head, takes three blocks fewer. `profile` now counts the loop's extra block passes, checkpointing and
  the split. Its estimate for the looped 8B over two cards (rank 16 on all seven projections, top 24 blocks) is
  12.2 GB on the busier card at 512 tokens, 15.2 GB at 1,024 and 21.8 GB at 2,048. These are estimates; the run
  below measures.
- **First run on the cards (owner's yes, 2026-11-25 06:41):**
  - **Setup:** the looped 8B over both GPUs, BF16 base, rank 16 on all seven projections in the top 24 blocks, the
    loop gate, checkpointing.
  - **Result:** 4 steps trained. Peak 9.5 GB on GPU 0 and 8.1 GB on GPU 1. 4 steps plus loading took 14 s. Loss 3.53
    to 1.89, held-out 2.84.
  - **Live model:** Cyber-Tiel was off the server for about 1.5 minutes and reloaded in 30 s.
  - **Not measured yet:** long examples. The smoke set's conversations are about 40 tokens, so the 512 and 1,024
    limits were never reached. Long examples need a second run.
- **1M context, the Workshop's part (2026-11-25):**
  - **L1, YaRN:** `rope_scaling` / `rope_parameters` with `rope_type: yarn` are read. The frequencies equal
    transformers' formula within 1e-12 on Qwen3's head (factor 32, 32,768 original).
  - **L2, sliding windows:** `layer_types` and `sliding_window` are read; a sliding layer's query i sees the keys
    i - window < j <= i.
  - **The `context` command:** `am2strata-workshop context <model> --out <dir> --length 1048576 --original 32768
    --window 4096 --full-every 4` writes the long-context copy with linked weights, keeping a loop. Every fourth
    layer and the last stay full.
    - `Strata-data/mula/mula-base-1m-loop-v0.1`: full attention at layers 3, 7, ..., 35, the loop 12..24 kept.
    - Untrained, it still answers "The capital of France is" with " Paris", at 0.38 probability against the
      looped base's 0.53. That is the cost of the stretched positions before L4's training.
  - **L3, chunked attention:** sequences over 2,048 tokens never build a seq x seq table. A chunk of 512 queries is
    computed at a time against the keys it may see, and the backward pass recomputes each chunk with exact
    gradients.
    - Against the table on a looped model: logits within 8e-7 and gradients within 9e-8, full and sliding layers,
      chunks of 1 to 16, with and without checkpointing.
    - On two GPUs the same training run matches the CPU (losses 3.6e-7).
  - **Next:**
    - L4 needs long data and GPU runs (each run approved).
    - L5 is the engine: YaRN, sliding windows, the loop and an 8-bit tiered cache in `am2strata-hybrid` for
      qwen3.
    - L6, the needle tests, runs on the engine.
- **Part E stage 1, the engine's long context (2026-11-26, branch `engine/long-context`):** 131K for the looped
  Mula, and what L5 needs from the engine.
  - **Convert:** `convert_qwen3` writes a config's YaRN as llama.cpp's keys (`rope.scaling.type` yarn, `.factor`,
    `.original_context_length`, `.yarn_beta_fast` / `_slow`, `.attn_factor` only when the config's attention factor
    is not YaRN's own), `context_length` the stretched length. `am2strata-workshop convert <dir> --yarn 4
    [--yarn-original 32768]` adds YaRN to a folder that has none (Qwen's recommendation: 131,072 tokens; the default
    original is the config's `max_position_embeddings`, at most 32,768). The `context` command's sliding layers convert
    too: `attention.sliding_window` and the per-layer `attention.sliding_window_pattern` (true: the layer slides).
  - **The cache, `--kv f16|f32|q8`** (f16 the default), on the CPU path (`kvcache.rs`) and the device graph
    (strata-gpu `KvCache`): f16 is the rounding the CPU always did, now stored in half the memory; q8 is 8 bits with
    one f32 scale per head row (amax / 127). On the graph K and V are written as f32 rows, then stored by one kernel;
    the flash kernels and the scores/softmax/blend kernels read every encoding (`--kv f32` keeps the old cache and
    its kernels' arithmetic).
  - **Sliding windows** in the engine: a sliding qwen3 layer's query i sees the keys i - W < j <= i. The CPU keeps the
    window and 256-512 tokens of slack (a rollback of prompt lookup's guesses stays exact); the graph keeps a ring of
    W + 512 tokens, in any encoding, and reads prompts in chunks of at most 512 then.
  - **Gates** (all green: the CPU tests, the graph tests on cuda:1, the Workshop's convert and long-context tests):
    - The loop and MoE fixtures: the CPU's f16 cache within 6.1e-4 of an f32 cache, q8 within 1.6e-2, against the
      Workshop's logits the same; a chunk equals token by token, bit for bit, in every encoding.
    - The graph against the CPU in the same encoding: within 3e-4 (heads of 16, 32 and 64; f32 1e-6, q8 3e-4).
    - A new Workshop fixture, `tiny-qwen3-long` (YaRN 4 over 32 added by `convert --yarn`; a `context` copy with YaRN 16
      and two of its three layers over 5 tokens; 24 tokens): the engine equals the Workshop's forward pass within
      1.9e-6 with an f32 cache (f16 9.7e-4, q8 2.7e-2), on the CPU and on the graph, token by token and in chunks.
    - A windowed synthetic model over 1,300 tokens (the graph's ring wraps twice): the graph within 3.4e-4 of the
      CPU (q8 2.9e-3).
  - **Memory per token, the looped 8B** (36 layers + the loop's 12 again = 48 caches, 8 KV heads of 128):
    | `--kv` | Per layer, K and V | Per token (48 caches) | 32K | 131K |
    |---|---|---|---|---|
    | f32 | 8,192 B | 384 KiB | 12 GiB | 48 GiB |
    | f16 | 4,096 B | 192 KiB | 6 GiB | 24 GiB |
    | q8 | 2,112 B (16 scales) | 99 KiB | 3.1 GiB | 12.4 GiB |

    One 16 GB card (15.9 GiB) beside 8.1 GiB of Q8_0 weights, about 1 GiB kept for CUDA and the chunk's scratch,
    leaves ~6.8 GiB: **~18K tokens in f32, ~37K in f16, ~72K in q8.** Without the loop's pass (36 caches) q8 fits ~96K.
    131K in q8 needs 12.4 GiB of cache: in RAM on the CPU path it fits today; on the graph it needs the cache over two
    cards or tiered into RAM (L5's Stara), or L2's hybrid layers (12 full caches at 131K in q8: 3.1 GiB, the windowed
    ones 0.3 GiB, which fits one card).
  - **A 32K prompt on the graph** (`graph.rs long_prompt_on_the_graph`, ignored; cuda:1 beside the live model): a
    synthetic model of Qwen3-8B's shapes with 8 layers in Q8_0 (2 GB), 32,768 tokens in chunks of 128, then 32
    decoded tokens:

    | `--kv` | Prefill tok/s (8K / 16K / 24K / 32K reached at) | Decode at 32K | Caches | Device memory |
    |---|---|---|---|---|
    | f32 | 394 (8 / 25 / 50 / 83 s) | 83.7 tok/s | 2.04 GiB | 5.6 GB |
    | f16 | 318 (8 / 26 / 65 / 103 s) | 62.4 tok/s | 1.00 GiB | 3.8 GB |
    | q8 | 353 (9 / 27 / 55 / 93 s) | 65.2 tok/s | 0.52 GiB | 4.6 GB* |

    *the CubeCL pool kept the f16 run's buffers. At 32K attention dominates (the speed falls from ~670 tok/s at 8K):
    the chunk kernel reads K and V once per query token, so prompts cost N² reads. Decoding the compact caches costs
    their unpacking (f16 first decoded with branches: 150 / 31.8 tok/s; branch-free now). For the 48-cache looped 8B
    this predicts ~6x the time: a 32K prompt in about 10 minutes, 131K (16x the attention) in hours, until the chunk
    kernel shares each K/V load across a tile of queries.
  - **Not yet:** the graph's caches on more than one device or in RAM; q8 per 32-value block (llama.cpp's q8_0, more
    accurate than a scale per 128-value row) or 4 bits; flash attention for chunks reads K and V once per query token
    (a tile of queries sharing each K/V load, and tensor cores, would make long prompts several times faster); a
    position-scaled engine run of the real 8B at 131K (the owner's, with the live model unloaded); export_moe still
    refuses YaRN and sliding layers.
- **What long training still needs (profile of the 1M copy over two cards, checkpointing on):**
  - **Estimates:** 31 GB on the busier card at 4K tokens, 98 GB at 16K, 186 GB at 32K.
  - **Where it goes:** the attention table is gone. The output head's logits for every position remain, 15 GB
    at 4K. So does one block's working set while it is recomputed, and the saved block inputs (19 GB at 32K).
  - **Three pieces before L4 can run at 16K to 64K:**
    - the loss computed a chunk of rows at a time, with the head's recomputation in the backward pass;
    - token-chunked recomputation inside a block (everything but attention works token by token);
    - the checkpointed block inputs kept in RAM, not on the cards.
- **The loop in the engine (2026-11-25), M8's first half:**
  - **Convert:** Qwen3 folders now convert to a `qwen3` GGUF as llama.cpp's converter writes it, a tensor at a time
    (the looped 8B: 56 s, 16.4 GB in F16, 8 GB of memory, mostly the mapped shards). The loop goes with it.
  - **Engine:** `am2strata-hybrid` runs the loop on the CPU path, token by token and a chunk at a time, each pass
    with its own attention cache. On the tiny looped Qwen3 with open gates, its logits equal the Workshop's within
    2e-3 (the base alone is 6e-4 off, from the F16 cache; the loop moves the logits by 1.7).
  - **The looped 8B** (`Strata-data/mula/gguf/mula-base-loop-v0.1-f16.gguf`, gates still closed) answers "The capital
    of France is" with " Paris", the same 8 tokens with one pass or two. On the CPU in F16: 18.0 s with one pass,
    23.5 s with two.
  - **Open:** the GPU graph path for qwen3 and the loop (M8's second half); a per-request pass count.
- **M8's second half, the GPU graph (2026-11-26, branch `engine/moe-graph`, the plan's Part D):**
  - **Profile first:** `AM2_PROFILE=1` prints, per request, the prompt's and the decode's cost per token: time, time
    waiting on the device, host syncs, kernel launches, bytes each way. Matrix by matrix, a small qwen3moe made 257
    syncs per decoded token and waited on the device 6.4 of its 11 ms; the dense qwen3 113, its looped copy 169.
  - **The graph runs qwen3** (per-head q/k norms, NEOX RoPE, `key_length`), **the loop** (each extra pass with its
    own KV cache on the card, the gate mix on the card) **and softmax-routed experts** (`qwen3moe`, Mixtral-style
    `llama`; the loop allowed on `qwen3moe`). The router and its top-k run on the card; one wait per MoE layer reads the
    choice. Experts on the card run there; the others on the CPU, with only the activations crossing. Prompts go
    through it a chunk at a time on CUDA/HIP/Vulkan too, each chosen expert once per chunk.
  - **Gates:** the tiny looped Qwen3 on the graph equals the Workshop's own logits within 9.5e-7 (one token at a time
    and as a chunk). Against the CPU path (f16 KV there): 5.9e-4 to 1.5e-3 on qwen3, llama, the loop at 1, 2 and 3
    passes, and qwen3moe with all, half or none of its experts on the card. The best token is the same at every step.
    A chunk equals token by token, exactly on the graph and on the CPU path.
  - **Speed** on cuda:1 (synthetic models from `testgguf.rs`, 128-token prompt, 32 new tokens, the median of 3 runs;
    the machine was under load, load average 4-9). "Before" is the commit before this work, matrix by matrix:

    | Model | Before: prefill / decode tok/s | After: prefill / decode tok/s | CPU alone |
    |---|---|---|---|
    | small qwen3, 16 layers, 1024 wide, Q4_K | 146 / 116 | 1402 / 561 | 148 / 62 |
    | the same looped (blocks 4..12, 2 passes) | 82 / 55 | 928 / 381 | 87 / 37 |
    | small qwen3moe, 32 experts, top-4, Q4_K | 97 / 75 | 1303 / 248 | 151 / 70 |
    | the same in Q8_0 | 71 / 55 | 396 / 176 | 143 / 62 |
    | Q4_K, 0.55 GiB budget (276 of 512 experts on the card) | 94 / 84 | 592 / 140 | |
    | tiny looped qwen3 (64 wide) | 768 / 744 | 25800 / 1600 | 2867 / 842 |

    Syncs per decoded token went from 113-257 to 1 (dense, loop) and 17 (MoE: one per layer and the logits).
    On these small models, decode is now limited by the host queueing kernels: 340-630 launches per token, at
    about 2-4 µs each through CubeCL. The next step is fewer, larger kernels: fused norm and RoPE, and one launch for
    all experts per layer.
  - **For Mula on one card:** the looped 8B must be a Q8_0 or Q4_K GGUF to fit one 16 GB card with its two passes' KV
    cache. The F16 file (16.4 GB) does not fit. The owner runs the 8B benchmarks.
  - **Not yet on the graph:** Inkling 2 (`inkling2_loop`, model/inkling.rs). It needs attention without RoPE, with a
    learned relative-position bias, a sink and clipped q/k; short causal convolutions with recurrent state; depth-mixed
    residual streams; fused gate/up SiTU experts; and the loop controllers. That is a separate port, not an extension
    of this graph. Also not yet: sigmoid-routed and shared or latent experts (deepseek2, qwen35moe, nemotron), the
    loop and experts on the native Metal backend (refused there with a message), and a device-side expert dispatch
    (an indexed product) that would drop the one wait per MoE layer when all experts are on the card.

- **M8, the kernels (2026-11-26, branch `engine/kernels`):** the looped 8B in Q8_0 on one card decoded at 8.7 tok/s,
  about 110 ms of device time per token where its ~11.6 GB of weight reads need ~26 ms at the card's 448 GB/s, and read
  prompts no faster than it decoded. The Q8_0 product read its 34-byte blocks a byte at a time (95 GB/s), and a chunk
  ran the decode product once per token.
  - **Measured on cuda:1** (RTX 5060 Ti 16 GB, ~448 GB/s; the live server's model on the other card; load average 8-10).
    `cargo run --release --example kbench -- cuda:1` (strata-gpu) gives GB/s of weights for one vector and GFLOP/s for
    chunks of 32/128/512 tokens; `kbench -- cuda:1 attn` times decode attention.

    | Product | Decode GB/s before -> after | 128-token chunk GFLOP/s before -> after | 512 tokens after |
    |---|---|---|---|
    | Q8_0 4096 x 4096 | 95 -> 400-658 (L2) | 184 -> 5240-5420 | 7280-7440 |
    | Q8_0 12288 x 4096 | 95 -> 400 | 183 -> 6010-6400 | 5640-5910 |
    | Q8_0 4096 x 12288 | 95 -> 390-397 | 183 -> 5190-5300 | 7660-7740 |
    | Q8_0 151936 x 4096 | 97 -> 424 | | |
    | Q4_K 4096 x 4096 | 245 -> 310-358 | 1000 -> 5960-6270 | 6750-6820 |
    | Q4_K 12288 x 4096 | 267 -> 377-384 | 995 -> 6700-6880 | 7100-7150 |
    | Q4_K 4096 x 12288 | 271 -> 368-377 | 1080 -> 5460-5860 | 7530-7550 |
    | Q4_K 151936 x 4096 | 228 -> 403 | | |

    Decode attention (32 heads of 128, 8 KV heads), µs per layer: 128 positions 18 -> 12, 1024: 112 -> 18, 4096:
    457 -> 63.
  - **What changed:** Q8_0 is repacked on upload (values first, f16 scales apart; the same bytes on the card). Decode
    products load 16 bytes per unit, a warp per row, one warp reduction per row (`mv_fast`), 90-95% of the card's
    bandwidth on the large shapes. Chunks of 8+ tokens take a tiled matmul (`mm`: a 64-row x 64-token tile, the
    weights dequantized once into shared memory). Per layer now: norm, q/k/v in one kernel (V straight into its
    cache), q/k norms + RoPE + K into its cache in one, attention (flash decoding: runs of positions per KV head, then
    a merge; one kernel with a running maximum for chunks), o + residual, norm, gate/up + SiLU in one, down + residual.
  - **Synthetic models** (`testgguf.rs write_kernel_bench_models`: Qwen3-8B's shapes, 8 layers, 151936 vocabulary;
    512-token prompt in 128-token chunks, 64 new tokens, after a warm-up request):

    | Model | Prompt tok/s before -> after | Decode tok/s before -> after | Launches per decoded token |
    |---|---|---|---|
    | Q8_0, 2.3 GB on the card | 58 -> 1790 | 38.8 -> 153 | 170 -> 74 |
    | Q4_K, 1.2 GB | 306 -> 2016 | 146 -> 250 | 170 -> 74 |
    | Q8_0 looped (blocks 2..6, 2 passes: 12 layer passes) | 39 -> 1208 | 28.9 -> 116 | 256 -> 112 |

    On the Q8_0 one, nsys shows 5.8 ms of kernels per decoded token (~395 GB/s of weights over the whole token) and
    0.5 ms between tokens (the logits' read, sampling, the next token's uploads). Prompt reading is 11.7x decode.
  - **For the looped 8B** (48 layer passes, ~11.6 GB read per token) this predicts about 30 ms per token, ~33 tok/s
    decode, and ~300 tok/s prompt reading (the owner measures it with the live model unloaded).
  - **Parity:** every kernel test green on cuda:1 (the tiled kernel within 2.6e-7 relative; each fused kernel equal to
    the kernels it replaces); a new graph test runs tiny Q8_0 and Q4_K models against the CPU path on the same model
    dequantized to F32 (9.1e-4 / 9.4e-4, the same best token at every step); the other graph tests unchanged.
  - **Not yet:** the tiled matmul runs on the FP32 units at 5-7.7 TFLOP/s (about a third of their peak); tensor cores
    (CubeCL's cmma) would make prompt reading several times faster again. Q6_K, Q5_K, F16 and the IQ kinds keep the
    older kernels (Q4_K_M models store some matrices as Q6_K). Activations and the KV cache stay f32 (an f16 cache
    would halve attention's reads). One host sync per token remains, with the full logits read back.

## Mula-MoE 56B-A5B: fine-grained upcycling (owner 2026-11-26: "56B-A5B"; branch `workshop/moe-a5b`)
- **Shape:** each block's dense feed-forward (12,288 neurons, 151M parameters) is split into 12 groups of 1,024
  neurons; 120 experts = 12 groups x 10 copies; top-5. Total 57.13B (2.76B shared + 36 x 120 experts of 12.6M),
  active 5.04B per token. `upcycle --experts 120 --expert-ff 1024 --top-k 5`; without `--expert-ff` the full-width
  design (10 experts top-1, 57B-A8B) is unchanged.
- **The split** (`--split cluster`, the default; `contiguous` the fallback): MoEfication. The dense model runs over
  calibration conversations block by block, and each neuron's `|silu(gate)·up|` over the calibration tokens is its
  feature. Balanced spherical k-means (seeded) makes groups of exactly 1,024. The permutation is stored in
  `moe_init.safetensors` and applied to the weights once at load: nothing is copied.
- **The router's start:** expert (copy c, group g) = `d_c + 0.01 · r_g`. `r_g` is a ridge regression of the group's
  importance (`Σ |act|·‖down column‖`) on the block's input. `d_c` is a random direction per copy. A token takes all
  its k experts from one copy, so it gets k different groups: its best k by `r_g`, with nearly equal weights. The
  chosen experts' outputs are multiplied by k (folded into `ffn_down_exps` at export), so k equal weights sum k groups,
  as the dense model sums all 12.
- **Not exact at the start:** a token uses 5 x 1,024 of 12,288 neurons. `--dense-start-check` measures the held-out
  loss of the dense model and of the upcycled one at the start. Only E = G = top-k (every group once) equals the dense
  model, and that is the slicing's parity gate.
- **Training memory** (two 16 GB cards, the calibration of the first real run: 2.8 GB CUDA overhead a card):
  - At 1,024 tokens nothing fits with a 1.5 GB margin. Even the router alone needs 16.1 GB a card.
  - At 512 tokens, the default is expert rank 4 with A shared per group (each copy its own B): 155M trainable,
    15.4 GB on the busiest card.
  - Rank 8 shared needs 16.6 GB; rank 4 unshared 16.9 GB.
  - `upcycle --estimate` prints the whole table.
  - Router losses: Switch's load balance over 120 experts (0.01) and ST-MoE's z-loss (0.001).
- **Export:** `ffn_{gate,up}_exps [4096, 1024, 120]`, `ffn_down_exps [1024, 4096, 120]`, `expert_feed_forward_length`
  1,024; Q4_K about 32.2 GB.
- **Engine:** the qwen3moe path reads the experts' width from the tensors. The Workshop's fine-grained fixtures (a
  shuffled split, an untrained block on its start router, a loop with a shared A) match its logits within 5e-6 with an
  f32 KV cache. With the default f16 cache the difference is 1e-3, and 7e-3 for the looped top-3 fixture, because
  routing near the k-th place is sensitive. A synthetic 120-expert top-5 model with a loop runs on the CPU, and the
  graph tests cover it on a card.
- **The real start** (2026-11-27, CPU, 22 min, peak 13.8 GB RSS):
  - The run: `Strata-data/mula/mula-base-loop-v0.1-moe120f1024x5`, a cluster split on 4,291 tokens of 16
    mula-train-v2 conversations (512-token cap).
  - Neuron-to-centre similarity is 0.655 clustered against 0.635 contiguous.
  - 95.9% of tokens pick 5 different groups. The chosen groups hold 54.5% of the feed-forward's importance: the best
    possible is 55.1%, and any 5 groups hold 41.7%. The busiest expert gets 2.63x the even share.
  - **Dense-start check on 2,122 held-out answer tokens:** the dense model's loss is 0.914 and the upcycled model's at
    the start is 2.208 (+1.29 nats, perplexity x3.65). The first epoch has to win that back.
  - `profile` at 512 tokens on two cards (rank 16 on the attention, experts rank 4 with A shared): 3,158 steps of 4
    conversations, about 15.1 h an epoch, 15.47 GB on the busiest card.
