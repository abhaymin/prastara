# Inkling 2 Mini: run it, and pack it

**Request (2026-11-21):** Prastara supports Inkling 2 Mini, and can pack and consume models of this kind.

## The model

`inkling2_loop`: 24 layers, 128 experts with 8 used per token, text and image input, a 200,058-token vocabulary (padded
to 201,024 in the weights).

- **The loop:** layers 6-17 run 3 times per token. Each pass keeps its own KV cache and its own convolution state, and
  small controller weights pass messages between the passes.
- **Attention:** no RoPE. A learned relative-position bias, an attention sink, and clipping of Q, K and the bias. 4 of
  the 24 layers attend to the whole context; the other 20 see a 1,024-token window.
- **Short convolutions** (4 taps) on K, V, the attention output and the expert output, with a recurrent state.
- **Depth mixing:** every 6 layers a new residual stream starts, and each layer mixes the earlier streams with learned
  queries.
- **Experts:** the SiTU activation, a router kept in F32, and expert weights scaled by 8.
- **Tensor types in the test file** (AD-Q4_K): F32, Q8_0, Q5_0 (the 1,152-wide expert down projection, which k-quants
  cannot take) and Q4_K.
- **Tokenizer:** BPE with llama.cpp's GPT-4o pre-tokenizer (`tokenizer.ggml.pre = inkling2`), no BOS.
- **Chat format:** `<|message_user|><|content_text|>…<|end_message|>`, a system message "Thinking effort level: N" set
  by `reasoning_effort` (none 0, minimal 0.1, low 0.2, medium 0.7, high 0.9, xhigh and max 0.99), and tools declared in
  a `tool_declare` system message.

## The reference

llama.cpp master at `19bc41dbb` (b12608 or later), built in its own folder. It loads the file and passes chat,
reasoning, a tool call and an image question: about 23 tokens per second on two 16 GB cards, 18 GB used at 8K context
with a small batch. It is the oracle for every gate below.

## Steps

| Step | What | Gate |
|---|---|---|
| I1 | The tokenizer (`inkling2`) and the chat template | the reference's token ids for the golden texts; the reference's prompt for each golden request |
| I2 | Output parsing: reasoning, answer and tool calls from the model's own tokens | the golden raw outputs give the same reasoning, content and tool calls as the reference server |
| I3 | A bridge engine: llama.cpp's library behind Prastara's engine protocol, for any architecture llama.cpp runs | the same tokens as the reference at temperature 0; STOP ends with DONE; the same tokens with and without the conversation cache |
| I4 | The server: the catalog entry, `reasoning_effort`, the run config, `pull` | a chat, a tool call and a streamed answer through `/v1/chat/completions` and `/v1/messages` |
| I5 | Vision through the bridge | an image question answered as the reference answers it |
| I6 | The Rust engine: `inkling2_loop` in `am2strata-hybrid`, CPU first | teacher-forced parity with the reference on a tiny random model and on the real file |
| I7 | Packing: quantize, LoRA merge (the GGUF adapter kind too), expert packs and profiles, each tested on this architecture or refused by name | byte for byte against `llama-quantize` where it applies; a merged file gives the adapter's tokens |
| I8 | The Rust engine on the GPU, with expert tiers | no slower than the bridge, the same tokens |

I3 makes the model usable while I6 and I8 are built; it is not the end state.

## Direction (owner, 2026-11-23): native only

**"I don't want llama.cpp as part of Prastara. I want the full feature in Prastara, not an extension."**

The bridge (I3, I5: `crates/strata-llama`, the `LLAMA_ARCHS` routing in `strata-hub`, the repo config's bridge and
image-encoder entries) is scaffolding. It stays local and unpublished, keeps the model usable meanwhile, and is deleted
when the native engine passes I8's gate. llama.cpp remains only what it is for every other architecture: an external
test oracle, never linked into or shipped with Prastara.

**`main` is Rust only (owner, 2026-11-24): "not to leave main system in any other language".** So on `main` the
bridge is already gone: no `crates/strata-llama`, no `LLAMA_ARCHS` routing, and Inkling's repo config and `pull`
run it on the Rust engine (`am2strata-hybrid`, with pictures since I6v, generation about 6 to 7 tokens/s on two cards
until I8). The bridge lives only on a local branch, and the sections below that describe it (I3, I5) are its
history, not something `main` builds.

Order of work, each with its gate:

| Step | What | Gate |
|---|---|---|
| I6a | `inkling2` blocks in `strata-hybrid` on the CPU: clip, depth mix, short convolutions with their state, relative bias with the sink, windowed and full attention, the SiTU experts with the fused gate/up matrix | the oracle's per-layer dumps on the tiny model, layers 0-1 |
| I6b | The loop: 3 passes over the core, one KV cache and one convolution state per slot, the controllers' messages | the dumps for every slot, then teacher-forced logits on the tiny model |
| I6c | The real file, memory capped | teacher-forced parity with the oracle |
| I6v | **Done 2026-11-25 (below).** The image encoder in Rust: the ViT with a CLS token, learned positions resized bilinearly, 2D RoPE, patch merge, the two-layer projector; `GENI` in `strata-hybrid` (embedding rows skip the token norm) | the rows against mtmd's rows for the golden image; the golden image question |
| I8 | The GPU path: Q5_0 kernels, the new blocks on the device, two cards | the same tokens, and no slower than the bridge's 24 tok/s |
| I7 | Packing for the native engine | as above |
| I9 | Remove the bridge: the crate, the routing, the config entries, the docs | the server suites pass with no `strata-llama` in the workspace; the model loads from a fresh install |

## Training looped models in the Workshop (owner, 2026-11-21)

The Workshop's fine-tuning (candle, in Rust) learns this architecture too, so Prastara can train and tune models of
this kind itself.

| Step | What | Gate |
|---|---|---|
| W1 | The forward pass in candle, with the loop: the 12 core layers run 3 times with the same weights, each pass with its own convolution state and the controllers' messages between passes | the same logits as the reference on the tiny random model |
| W2 | LoRA training on it: adapters on attention, the experts and the loop controllers; a shared core layer gets one adapter used by all 3 passes, its gradient summed over the passes | the loss falls on a small set; a merged adapter gives the same tokens as base plus adapter in the reference |
| W3 | Memory: gradient checkpointing per pass and expert-sparse gradients, so a tune fits beside this PC's other work | measured peak memory and speed on the real model, reported before any long run |
| W4 | Export and reuse: the adapter as a GGUF LoRA (the kind llama.cpp loads), merge, quantize, and the run config | the exported adapter loads in the reference and in Prastara's engine and gives the same tokens |

W1 needs I6's model code; W2 to W4 follow the Workshop's existing LoRA path (`strata-workshop`).

## Status (updated by the reviewer)

**2026-11-21**
- **Reference:** llama.cpp `19bc41dbb`, built twice in `~/Development/llama.cpp-inkling`: CUDA in `build/`, and CPU with
  the architecture test in `build-cpu/`. The CPU oracle is `Strata-data/oracle/oracle-inkling`
  (`tools/reference-llamacpp/oracle.cpp`).
- **The tiny model:** `Strata-data/oracle/fixtures/inkling2/inkling2-tiny.gguf` (73 MB), made by
  `promo/documentary/code/inkling-1/make_fixture.py`, because llama.cpp's saver refuses this architecture. It has:
  - 6 layers, with the loop over layers 2-3, 3 passes and a block size of 2;
  - full attention at layers 3 and 5;
  - an embedding width of 128, 8 experts with 2 used, and the real tokenizer.

  The oracle runs it on the CPU.
- **Types:** Prastara's reader and `strata-pack` know every type in the real file: F32, Q8_0, Q5_0 and Q4_K. The
  engine's GPU kernels for Q5_0 are not checked yet (I8).
- **I2 golden:** `golden/raw_outputs.json` keeps the token ids. Reasoning sits between special tokens (200008 …
  200010), then the answer (200004 … 200010), then 200006 ends the turn. The answer text alone loses that structure, so
  I2 must be built from the ids.
- **I1:** the agent's round 1 stopped (a refused read of `~/.cargo`); round 2 is running.
- **Next:** per-layer tensor dumps from the oracle (`Qcur`, `attn_out`, `ffn_inp`), so that I6 and W1 can be checked
  layer by layer: layers 0-1 first, then the loop.
- **Per-layer dumps (2026-11-21):** with `ORACLE_DUMP=<dir>`, the oracle writes every named tensor of the prompt pass
  (`Qcur-N`, `Kcur-N`, `Vcur-N`, `Rcur-N`, `kq_b-N`, `attn_out-N`, `ffn_inp-N`, `ffn_moe_*-N`, `l_out-N`) as raw f32,
  with `index.tsv`. N is llama.cpp's cache slot, not the physical layer. On the tiny model, slots 0-5 are the 6 layers
  in their first pass, and 6-9 are passes 2 and 3 of layers 2-3: slot `6 + 2*(pass-2) + (layer-2)`.

## I3: the llama.cpp engine (`crates/strata-llama`, 2026-11-22)

- **`am2strata-llama`** is all Rust: hand-written `extern "C"` declarations for llama.cpp's C API, linked at build time
  from `AM2_LLAMA_LIB`, with an rpath. It speaks docs/PROTOCOL.md: `INFO`, `READY`, `GEN` (greedy, or temperature,
  top_p, top_k, min_p and seed), `T`, `PP`, `DONE` (5 fields), `STOP`, `ERR` and `QUIT`.
- **STOP is ordered against the GENs around it.** It stops every request up to the latest GEN read, so a STOP sent while
  the prompt is still being read is kept, and a late STOP never stops the next request.
- **No prompt reuse:** each GEN clears the memory. The per-pass convolution state makes cutting back to a shared
  prefix unsafe.
- **Build:** `AM2_LLAMA_LIB=~/Development/llama.cpp-inkling/build/bin cargo build --release -p strata-llama` (CUDA), or
  `build-cpu/bin` for the CPU. Without the variable, the binary says it was built without llama.cpp and exits.
- **Tests** (`AM2_LLAMA_LIB=<cpu build> cargo test --release -p strata-llama -- --test-threads=1`), 4 of 4:
  - greedy parity with the oracle on 3 prompts (5, 15 and 40 ids, 16 tokens each);
  - two requests in a row;
  - STOP mid-request, then a full request;
  - bad GENs answer ERR, then a good one is served.
- **Run config:** `configs/strata-inkling-2-mini-uncensored-ad-q4_k.json`. The tokenizer folder must be extracted
  from the GGUF first (as `pull` does). It is not tested in the server yet: that is I4.

## I4: the server serves it (2026-11-22)

- **The stream:** a model with `CallFormat::Inkling` is read by `InklingParser` in `run.rs`, fed the token ids, on
  `/v1/chat/completions` (streamed and not) and `/v1/messages`. A reply that ends in tool calls ends with `tool_calls`
  (`tool_use`). Test: `tests/inkling_server.rs`, two golden replies over both APIs, streamed and not. The whole
  `cargo test --release -p am2strata` passes (23 suites).
- **A run config may name its format** with `"call_format": "inkling"`, which overrides detection. The tests use it,
  because the mock engine's tokenizer is not Inkling's.
- **The real model, once, by hand:** a second server on port 8093, the CPU engine and the AD-Q4_K file. The server
  writes the tokenizer folder itself from the GGUF (`ensure_tokenizer`) when the config names a `tokenizer` folder
  that does not exist.

  | Request | Result |
  |---|---|
  | `tool_one` | `get_weather {"city":"Kochi"}` and `finish_reason` `tool_calls`, as the reference; on `/v1/messages` a thinking block, then `tool_use` with `stop_reason` `tool_use` |
  | `math_high` | "391" |
  | `hindi` | a correct Hindi answer |

  Each took 14-19 s on the CPU.
- **The prompt difference, found:** the reasoning text and the Hindi wording differed from the reference because the
  server adds its three memory tools (`remember`, `recall` and `forget`) as a `tool_declare` message to every request. That
  made 336 tokens against the reference's 27; the template renders them as it renders any tools.
  - **With `"strata_memory": false`:** for all 7 golden requests, the prompt ids the server sent (recorded by the mock
    engine, with Inkling's real tokenizer folder and template, and no `call_format` override) equal the reference's
    exactly: 14, 28, 29, 105, 188, 161 and 26 tokens.
  - **Still to check:** that a streamed reply gives several content deltas (the parser holds a block until
    `<|end_message|>`), and what a reply cut by max_tokens returns.
- **Two engine bugs found by this run, fixed:**
  - the model path went to C without its NUL (a `&str` pointer);
  - `build.rs` watched `AM2_LLAMA_LIB` only when it was set, so a build that did not set it never rebuilt once it was.

## W0: tiny looped models from Rust (2026-11-22)

- **`strata_workshop::testmodel::TinyInkling`** writes an `inkling2_loop` GGUF with `strata_pack::gguf_write`, which
  gained a BOOL array (`sliding_window_pattern`). It has its own small vocabulary, with the I2 parser's special tokens
  at fixed ids. There are two presets:
  - `default()`: width 128, experts 64 wide, 6 layers, the loop over 2-3 for 3 passes;
  - `wide()`: width 256, experts 192 wide (not a multiple of 256, as the real 1,152 is not), for the pack tests.

  F32 throughout, or Q8_0 for the vocabulary matrices and the experts.
- **Make them:** `cargo run --release -p strata-workshop --example write-inkling-fixtures -- <dir>`.
- **Tests:**
  - always: the file reads back with every tensor and key, and the same seed gives the same bytes;
  - with `AM2_ORACLE=<oracle-inkling>` (`--ignored`):
    - llama.cpp loads both presets and generates;
    - the dump lists the loop and attention tensors with the right shapes;
    - a 150-token prompt runs past the window and the extent.

  All pass, and the Python generator is no longer needed.
- The agent wrote this over two runs; the second timed out before committing, and the reviewer ran the gates and
  committed.

## On the GPUs, and a load test (2026-11-22)

- **Two RTX 5060 Ti:** the live server's model was unloaded (`/api/models/unload`, no restart). `am2strata-llama` was built
  against the CUDA library (`AM2_LLAMA_LIB=…/build/bin`).
- **`--tensor-split a,b`:** llama.cpp splits by cache slot, and this model has 49 slots for 24 physical layers. The
  default therefore put 14.8 GB on GPU 0 and 0.5 GB on GPU 1, and the context failed. With `--tensor-split 0.25,0.75`
  the weights are 8.0 / 7.3 GB.
- **Settings:** 32K context, batch 512, ubatch 128. 64K needs a 7.4 GB KV buffer on GPU 1 that does not fit, and a
  larger ubatch asks for a 27 GB compute buffer. In use: 11.4 GB and 13.3 GB.
- **Load test** (`promo/demos/inkling-loadtest/load.py`):

  | Test | Result |
  |---|---|
  | `math_high`, `tool_one`, `tool_two`, `hindi` at temperature 0 | content and tool calls identical to the reference's; 21-26 tokens/s |
  | streamed 1,318-token story | 22.8 tokens/s, first token at 0.26 s, 1,314 deltas |
  | a 26,533-token prompt | right answer, but 361 s (about 74 prompt tokens/s): the slow part |
  | 8 requests at once | all 8 answered and correct, served in turn, 13.3 s in all |

- **Next for speed:** a faster prompt read (a larger ubatch within memory, a quantized KV cache) and prompt reuse.

- **The long-context slowdown, found and fixed (2026-11-22):** llama.cpp's library default is `swa_full = true`, a
  full-size cache for the sliding layers, so each of Inkling's 20 sliding layers (1,024 wide, several run 3 times)
  attended over the whole context. llama-server sets it false. With `cp.swa_full = 0` at a 15K context, matching stock
  llama-server on the same GPUs and settings:

  | | Before | After | llama-server |
  |---|---|---|---|
  | Prompt | 117 tokens/s | 210 tokens/s | 208 tokens/s |
  | Decode | 5.0 tokens/s | 14.0 tokens/s | 14.2 tokens/s |

  The sliding-layer KV fell from 5.1 GB to 0.2 GB.
- **Ruled out on the way:** CPU threads (4 against 8), the sampler (greedy and sampled are the same), weight repacking
  (`--repack`, now an option) and the context-parameter struct, checked field by field against llama.h.
- **Found by the gist run:** Inkling, as an opencode agent, wrote a 16,426-token file in one tool call and filled the 32K
  context before the call ended, so the call was lost. opencode asks for 32,000 tokens a request, so the run config
  needs `"fit_max_tokens": true`.

## I5: pictures through the bridge (2026-11-22)

- **The encoder:** `am2strata-llama-vision` (crates/strata-llama/src/vision_main.rs) speaks the server's existing
  `ENC <image> <out>` / `OK <tokens>` protocol (crates/strata-engine/src/vision.rs) over llama.cpp's mtmd and the
  model's mmproj. It writes the picture's rows as raw f32, as wide as the text model's input.
  - The text model is opened vocab-only. That load leaves its width at 0, so the width comes from the GGUF's
    `<arch>.embedding_length`. A `no_alloc` load aborted in `load_tensors` and still used 16 GB.
  - The embeddings are byte-identical to the ones from a full model load (`AM2_VIS_FULL_MODEL=1`).
  - It refuses a projector that needs M-RoPE positions or non-causal image attention. Inkling's needs neither.
  - On the CPU, the 320×192 test picture took 0.7 s and 0.76 GB of RAM, and gave 40 tokens.
- **The engine:** GENI replaces each `image_pad` id with the next row of the file. A run of pads is decoded with
  `llama_batch_ext` and `llama_process`, as mtmd's helper builds it: normal positions and no outputs. A file whose size
  doesn't match the pads is an ERR that names the row width. `--vision` is accepted.
- **The C shim:** `src/shim.c` holds the embedding batch and the mtmd calls. It is compiled by build.rs against the
  headers of the llama.cpp checkout it links, so no struct of these calls is mirrored in Rust. Link order matters:
  mtmd, llama and ggml-base come after the shim's archive.
- **The server:** for Inkling the picture's token is `<|unused_200054|>` after `<|content_image|>`, which is the GGUF
  template's `<|content_image|><|unused_200054|><|end_message|>`. llama.cpp's mtmd puts the picture right after
  `<|content_image|>`.
- **The install path:** `LLAMA_ARCHS` (`inkling2_loop`) in strata-hub's config.rs. `pull` and the Workshop's `strata`
  step now write the bridge's run config instead of refusing the file. The config has:
  - 64K context, batch 512 and ubatch 128;
  - a 0.25,0.75 split on two GPUs, or `--gpu-layers 0` with none;
  - `fit_max_tokens`;
  - the image encoder (on the CPU), from the given mmproj or an `mmproj*.gguf` beside the model.

  A native Rust-engine pack still waits on I6.
- **Gate:** the same picture and question, greedy, through Prastara (CPU, port 8093) and stock llama-server with
  `--mmproj` (CPU, port 8094).
  - **The prompt:** identical. The 78 ids match, with `<|content_image|>` (200005) followed by the 40 picture tokens.
  - **Case 1** (what building, and the time of day): the replies split at the 4th token. The reference's own top two
    there are " to" (−0.963) and " me" (−0.978), a 0.015 near-tie. Both describe "a multi-tiered pagoda/tower
    silhouetted against a sunset/sunrise sky".
  - **Case 2** (how many roof tiers): identical for 622 of 708 characters (about 175 tokens), with "5 roof tiers" on
    both.
  - **Ruled out for case 1:** the micro-batch (128 and 512), image-row outputs, the old `llama_batch` against
    `llama_batch_ext`, a vocab-only against a full model load, mtmd's settings, and llama-server's image warm-up
    (`--no-warmup` still gives " to").
  - **Unexplained:** text-only prompts match exactly, so something in the image input still differs slightly.
    llama-server's own embeddings were not compared. The gate (identical replies) is met on one case of two; case 1
    splits only at a 0.015 tie.
- **Live (port 8000, GPU, 2026-11-22):**
  - one picture: 62 prompt tokens against 19 for the text alone (40 picture tokens and 3 markers);
  - the 960×192 strip of three frames: 120 picture tokens;
  - two pictures in one request were both read;
  - streaming gave 29 deltas;
  - the Responses API (`input_image`) works.
- **The install path, run for real:** the Workshop `strata` step on the real GGUF (hard-linked into the Workshop's
  folder) wrote a config that matches the hand-made live one, apart from names, paths and `--threads`. Its tokenizer
  folder is identical to the hand-made one, and the encoder was found beside the model. It loaded and answered the
  same image question with the same 67 tokens and the same sentence.
- **Tests:** `geni_reads_image_rows_and_refuses_a_mismatch` (engine, tiny model) and
  `inkling_gets_the_llama_engine_config_with_its_image_encoder` (strata-hub).

## I6: the Rust engine runs it (2026-11-23)

`crates/strata-hybrid/src/model/inkling.rs`: `inkling2` and `inkling2_loop`, one token at a time, with its own cache
slots (`InkSlot`: K, V, and the convolutions' last inputs) in `State`, checkpoint and rollback included.

- **Tiny models, CPU, teacher-forced against the oracle:** `inkling2-tiny` 63/64 steps identical with 1 near-tie,
  `tiny-inkling` 64/64. Per-slot dumps (`AM2_INK_DUMP=<dir>`, the oracle's names) agree to 1e-5 on all 10 slots, every
  pass of the loop.
- **`tiny-inkling-wide`: 56/64.** At the one position examined (slot 4, token 173) the router's biased probabilities of
  two experts are 0.8649 and 0.8650 and the engines choose differently. The other 7 steps (45-56) are not examined yet.
- **The real file (AD-Q4_K), CPU:** 12/12 steps identical, worst logit difference 0.36 (0.008 on the tiny models: not
  explained yet). 2.2 tokens/s on 8 threads.
- **The real file on both GPUs** (`--devices cuda:0=11,cuda:1=11 --no-graph`): every matrix placed (14.9 GiB, 3,072
  experts), 24/24 steps identical with the oracle, worst logit difference 0.74. **8.3 tokens/s**, load 22 s. The bridge
  does 24 tokens/s. This is the matrix-by-matrix path: every product crosses to the device and back.
- **Found on the way:** the CPU dot kernels dropped the columns past a row's last block of 32 (Inkling's bias table is
  16 wide); fixed in `cpu_dot.rs` with a test. A Q5_0 kernel in `strata-gpu` (the experts' down projection), tested
  on CUDA, HIP and Vulkan; its Metal version is written and not compiled yet.
- **Matched to the oracle:** the attention bias less the sink is rounded to f16 (llama.cpp's mask type), K and V to f16.

Open before I6 is called passed:
- a golden chat prompt for 64+ steps, and a prompt longer than 1,024 tokens (the real window), on the real file;
- per-slot dumps on the real file, to place the 0.36 to 0.74 logit gap;
- the windowed slots keep every token's K and V in f32: trim them to the window before a long context;
- `--expert-profile`, `--expert-profile-save` and `--resident-experts` still read `layers`, which Inkling leaves
  empty: route them through `expert_layers()`;
- the prompt goes token by token (`forward_chunk`).

I8 is a device graph: the whole token on the devices (the core's passes on one card, the prelude, the last layers and
the output on the other is the first split to measure), instead of about a thousand round trips per token.

## I6v: pictures on the Rust engine (2026-11-25)

- **The encoder:** `crates/strata-hybrid/src/vision.rs`, binary `am2strata-vision` (the bridge encoder's `ENC <image>
  <out>` / `OK <tokens>` protocol; `--mmproj`, `--threads`, `--max-tokens`; `--model` only checks the row width,
  `--gpu` and `--min-tokens` are accepted and ignored). The ViT on the CPU with rayon and `matrixmultiply`, the F16
  weights widened to f32 (1.9 GB RSS). Decoding with the `image` crate (PNG, JPEG, WebP, GIF, BMP).
  - As mtmd: the padding is raw -1 before the normalisation (so -1.0078), not -1 after it; CLS at RoPE position (0, 0),
    the patches 1-indexed; x on the first half of each head, y on the second; positions resized with ggml's bilinear
    (half-pixel centres, clamped); the merge's 2x2 order is dy outer, dx inner; no post-norm; GELU with erf.
  - Every product with an F16 weight takes its input rounded to f16, as ggml's CPU backend does: worst cosine 0.999998
    against 0.999988 with f32 inputs (`AM2_VIS_F32_ACT=1`).
- **Gate A (rows), against `am2strata-llama-vision` on the CPU** (`tests/vision.rs` `rows_match_mtmd`), worst row:
  - synthetic 320x192 pagoda PNG (the I5 picture is not on disk): 40 rows, cosine 0.999998, max abs diff 1.1e-3;
  - 333x217 (padding on both sides): 54 rows, 0.999996, 1.7e-3;
  - 640x640 (the native 32x32 grid): 256 rows, 0.999990, 1.4e-3;
  - a real 512x512 photo (PNG): 169 rows, 0.999995, 1.3e-3.
  - **The same photo as JPEG: 0.99718 worst, 0.99992 mean.** The JPEG decoders differ (zune-jpeg here, stb_image in
    mtmd; ffmpeg's decode gives 0.99674 against stb too): the encoder is sensitive to 1-level pixel differences.
- **Encode time, CPU, 8 threads:** 320x192 0.6-0.8 s, 333x217 1.0 s, 512x512 6.8 s (4 threads, beside another job),
  640x640 5.2 s (4.5 s on 16 threads). The bits are the same for any thread count.
- **GENI in `am2strata-hybrid`:** `GENI <max_new> [keys] image_pad=<id> <file> <ids>`; each pad id is the next row,
  through `Model::forward_embd` (no token norm, as llama.cpp's `if (ubatch.token)`), text runs in chunks as before. A
  file whose size does not match the pads is an `ERR GENI:` naming the width; a model without rows or the device graph
  is refused. A GENI request starts a fresh state and leaves no history to resume from.
- **Gate B (answer)**, CPU, greedy, effort 0, the 320x192 picture and "What is in this picture, and what time of day
  is it?" (75 ids, 40 picture rows): the model starts thinking with "stylized black pagoda/temple silhouette against
  gradient sunset sky". Prompt 57 s (token by token, under a 16 GB cap), 32 tokens in 22 s.
- **Tests:** preprocessing (padding, normalisation, tokens = (W/40)(H/40), the shrink over the limit), the bilinear
  resize at 32x32 is the identity, the merge order, the products; the two gates above as ignored tests.
- **Open:** JPEG bit-parity with stb_image; the shrink over 4,096 tokens uses Catmull-Rom, not mtmd's bicubic (not
  compared). (The config switch to `am2strata-vision` is done: F2 below.)

## F2: video, frame by frame, on the Rust engine (2026-11-26)

- **Frames:** a chat video (`video_url`, `video`, `input_video` parts) is cut by the media engine's own MP4/H.264
  decoder (`video_frames`, Rust; ffmpeg only for what it cannot read, when installed): **one a second** by default.
  Request options: `video_fps` (0.05 to 8), `video_max_frames` (default 64, at most 256), `video_width` (default 640,
  64 to 1280), `video_fit` (`"error"`: refuse rather than thin); the older `video_frames: N` still means N frames
  evenly spread. Over the cap, the frames are spread over the whole video (one every duration/cap s: `"fit"` in the
  op, `pick_every` in am2strata-media), not cut off after the first minute.
- **In the prompt:** a note (`[A video of 42.0 s, as 42 frames in order, one a second, each after its time:]`), then
  each frame after its timestamp, `[00:42]` (`[1:02:03]` past an hour; tenths, `[00:02.5]`, when the frames are not on
  whole seconds). The parts are marked `"video_frame": {"v", "i", "t"}` (strata-frontend keeps the mark; not `"video"`,
  which Qwen's template renders as `<|video_pad|>`).
- **Fed incrementally:** the server encodes each video's first frame (its token count is every frame's: one decoder,
  one size) and the other pictures, builds the prompt, and sends `GENI ... rows=stream <file> <ids>` to an engine whose
  `INFO` has `geni_stream=1` (am2strata-hybrid). The engine reads the text, and at each picture waits for its rows in
  the file (re-sending its last `PP` each second, so the stall watchdog sees it alive; `STOP` ends the wait;
  `AM2_ROWS_WAIT_S`, default 600, is the limit). The server encodes frame 2, 3, ... in order and appends their rows
  (run.rs `FrameStream`): the attention cache grows frame by frame while the encoder works on the next. An engine
  without `geni_stream` (strata-poly, strata) gets all frames encoded first and one file, as before.
- **Progress:** `/metrics` (and the queue) show `frames_encoded`, `frames_read` (from the engine's `PP` against where
  each frame's rows end) and `frames_total` beside `prompt_read`/`prompt_total`.
- **Token budget:** frames x (rows + about 8 for the timestamp and markers) + the text must fit the context with
  `max_tokens` (else 2,048, at most a quarter of the context) left for the answer. Over it, each video is thinned
  evenly (its first frame kept; the note says `one every 2.5 s (fewer, to fit the context)`), rebuilt and checked
  again; with `"video_fit": "error"`, or when one frame per video is still too many, a 400 says what to lower.
- **Config:** `strata-hub` writes a `vision` section for `inkling2`/`inkling2_loop` with an mmproj on the Rust engine:
  `am2strata-vision` beside the engine (or `${AM2STRATA_HOME}/crates/strata-hybrid/target/release/`), on the CPU;
  the manager no longer adds `--vision` to am2strata-hybrid. Nothing on this branch names `am2strata-llama-vision`.
- **Tests:** server (mock engine): `a_video_is_read_frame_by_frame_with_timestamps` (6 frames, 6 GENI picture runs
  after `[00:00]`..`[00:05]` in order, `rows=stream`, the GENI line sent before the last frame is encoded, every row
  arrived; the cap spreads 4 frames over 10 s; 2 fps gives tenths), `a_video_without_streaming_is_encoded_first`,
  `video_frames_are_thinned_to_fit_the_context`; units: `pick_every` (sampling times, the 29.97 fps drift, the cap),
  timestamps, request options, thinning; strata-hub `inkling_pictures_use_the_rust_encoder`; the real engine:
  `tests/vision.rs` `geni_streams_a_video` (ignored; a synthetic square moving and turning red to blue).
- **Open:** pictures on the device graph (`--devices`): GENI is refused there, so a GPU-placed Inkling reads pictures
  only with `--no-graph`; F3 (the same path for Mula Drishti).
