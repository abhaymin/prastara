# Model packs: one file, laid out for a fast load and fast experts

**Request (owner, 2026-11-23):** a pack that puts the model's GGUF and its image files into a single package, with the
expert layers arranged so that the model loads quickly and runs fast. It must work from a quantized model, from the full
unquantized model, and from safetensors. It is Prastara's own format, built so that it can be released separately as
open source.

**"Imatrix for image" (owner, confirmed 2026-11-23):** for GGUF models it is the image projector, the `mmproj` file
beside a vision model's GGUF. From safetensors the image encoder comes out of the checkpoint itself. An importance
matrix (the `imatrix` used when quantizing) is also accepted as an optional input, and kept in the pack so the model can
be re-quantized later.

## Why a new file

A GGUF is laid out for a converter, not for a mixture-of-experts engine:

- **An expert is scattered.** All experts' gate/up matrices sit in one tensor and all their down matrices in another, so
  using one expert means two reads far apart, per layer.
- **Nothing is ordered by use.** The matrices every token needs are mixed with experts that are rarely chosen, so a load
  either reads everything or seeks.
- **The pieces are separate files.** The model, the image projector, the tokenizer files and the expert profile each
  live somewhere else, and a run config has to name them all.
- **The engine rearranges at load.** Weights are copied into the layout the kernels want every time the model starts.

## The pack: one file, `<name>.am2aipac`

Sections, each aligned to 4 KiB (2 MiB for the large ones), each with its own checksum, found through an index that
the file's header points to (the index sits at the end of the file, so a pack is written in one pass). The exact
layout is in `PACK-FORMAT.md`.

| Section | Holds |
|---|---|
| Manifest | architecture, where the pack came from (source file names and hashes), the model's licence name, format version |
| Model header | the model's settings, vocabulary, merges, special tokens and chat format. From a GGUF this is its header kept verbatim, so nothing is copied twice and `unpack` is exact |
| Core | every matrix a token always uses (attention, routers, norms, convolutions, loop controllers, output), in the order the engine walks them |
| Experts | one record per (layer, expert): its gate/up and down matrices together, hottest experts first |
| Image | the image encoder and projector's tensors, and its preprocessing settings |
| Profile | the expert ranking the order was built from, and the counts behind it |
| Imatrix (optional) | the importance matrix, for re-quantizing |
| Run defaults | context length, device split, sampling defaults |

What this buys:

- **Fast load.** The core is one sequential read and one upload per device. The experts are hottest-first, so the GPU
  tier is the start of the expert section, the RAM tier is what follows, and the cold tail stays on the SSD. No seeking
  and no rearranging: the bytes are already in the kernels' layout.
- **Fast experts.** A cold expert is one read of one record instead of two reads. The same order is what
  `--resident-experts` and the GPU placement already want, so they stop building it at start-up.
- **One thing to install, copy and share.** The model answers image questions with no second file to find.
- **Looped models.** The core layers' weights are stored once and shared by every pass; the profile counts a looped
  layer's experts across its passes.

## Sources

| From | How | Check |
|---|---|---|
| A quantized GGUF (and its `mmproj`) | each tensor's bytes are copied unchanged, only reordered | the same tokens as the GGUF; `unpack` gives back the source files byte for byte |
| A full-precision GGUF (F32, F16, BF16) | kept as it is, or quantized while packing by `strata-pack`'s quantizers, with the imatrix when given | kept: as above; quantized: the same bytes as quantizing first and packing after |
| Safetensors (a Hugging Face checkpoint) | through the Workshop's Rust converter (`strata-workshop/src/convert.rs`), then as above; each architecture needs its tensor-name map | the same tensors as the reference converter's GGUF |

Everything is Rust; no Python step.

## Parts

- **`am2pack` (its own crate, no dependency on the server or the engines):** the format, a reader (memory-mapped), a
  writer, `verify` and `unpack`. This crate and `docs/PACK-FORMAT.md` are what can be released on their own.
- **Engine:** `am2strata-hybrid --model <file>.am2aipac` (`--pack` is already the older directory pack); the image
  encoder reads the same file.
- **CLI:** `am2strata-run pack build | inspect | verify | unpack`.
- **UI:** a Pack step in the Workshop (source, quantization, profile, progress, the result's size and load time), and
  packs listed with the models.
- **`pull`:** builds the pack after a download when the architecture is one the Rust engine runs.

The older directory packs (`iq-pack`, for the `strata` engine) stay as they are.

## Steps and gates

| Step | What | Gate |
|---|---|---|
| P0 | The format written down (`docs/PACK-FORMAT.md`), and the `am2pack` crate with writer, reader, `verify` | a tiny model packs and reads back; a flipped byte is caught |
| P1 | Build from a quantized GGUF plus `mmproj`; `unpack` | `unpack` is byte-identical to the sources, on the tiny models and on Inkling 2 Mini |
| P2 | The Rust engine loads a pack (text) | the same tokens as from the GGUF; load time measured cold and warm, against the GGUF |
| P3 | Expert order from a profile; GPU, RAM and SSD tiers taken straight from the file | tokens/s and time to first token against the GGUF, reported |
| P4 | The image section, read by the Rust image encoder | the golden image question, from the pack alone |
| P5 | Build from full precision with quantization and imatrix; build from safetensors | as in the sources table |
| P6 | CLI, the Workshop step, `pull`, docs | a pack built and loaded from the UI and from the CLI |

P2 onward needs the native engine for the architecture (for Inkling: I6 and I8 in `INKLING-PLAN.md`); P0 and P1 do not.

## Status (2026-11-23)

Owner: start the code inside Prastara first; it is split into its own release later.

- **P0 and P1 are done** (CLI only; the Workshop step is P6).
  - `crates/am2pack` (AGPL-3.0, no dependency on the rest of the repository) and `docs/PACK-FORMAT.md`.
  - `am2strata-run pack build | inspect | verify | unpack`, behind the `pack` cargo feature (on by default).
  - Tests: a made-up model packs, reads back and unpacks byte for byte; a flipped byte in the header, the padding,
    a section or the index is caught; overlapping tensors, a cut file and a wrong profile are refused.
  - Inkling 2 Mini (AD-Q4_K, 15.39 GiB) with its `mmproj` (0.60 GiB): one 16.00 GiB pack in 22 s, 656 tensors in
    core, 3,072 expert records, 422 image tensors. `verify` 16 s. `unpack` 38 s, both files identical to the
    originals by `cmp`. The three tiny Inkling fixtures: identical too.
- **P2: the token gate is met** (text only); the timing is below.
  - `am2strata-hybrid --model <file>.am2aipac` loads a pack: the header is the GGUF's, the tensors are mapped in
    place, and each expert's matrices come from its record (the `aipac` cargo feature, on by default).
  - Same tokens and the same logits as from the GGUF:
    - the three tiny Inkling models, 32 steps each, CPU;
    - Inkling 2 Mini, 12 steps on the CPU and 24 steps with every matrix on the two GPUs;
    - GLM-4.7-Flash (the general expert path: gate, up and down per expert, 2,944 records), 8 steps, CPU.
  - A cluster node running the pack gives the same model fingerprint as one running the GGUF.
  - Both features compile off (`am2strata --no-default-features`, `strata-hybrid --no-default-features`).
- **P2's timing gate and P3's loading are measured** (Inkling 2 Mini AD-Q4_K, 14.86 GiB on two RTX 5060 Ti 16 GB,
  `--devices cuda:0=11,cuda:1=11 --no-graph`, WD SN580 NVMe; ms to READY, then to the first of 24 tokens; "before" is
  the engine as it was at P2). Every run gave the same 24 tokens, and with everything on the cards the top-5 logits
  of all 24 steps are identical between the old engine on the GGUF and the new one on the pack.

  | Run | Before, GGUF | Now, GGUF | Now, pack |
  |---|---|---|---|
  | Warm (file in the page cache) | 21.0 to 23.2 s | 5.7 s | 5.1 s |
  | Cold (file evicted first) | 31.5 to 33.6 s | 17.1 s | 6.9 to 7.0 s |
  | Peak resident memory while loading | 33.5 GiB | 2.7 GiB | 6.9 GiB |

  Resident memory at READY, measured on the pack only: 2.2 GiB (17.4 GiB of it was the runtime's host copies before
  the flush below, plus the 15 GiB mapped file).

  What changed in the engine (`strata-gpu`, `strata-hybrid/src/gpu.rs`, `pack.rs`):
  - a matrix goes to the GPU runtime as it is mapped (two host copies and a per-word loop are gone);
  - the runtime kept a host copy of every upload until its queue was flushed, which nothing did during a load: the
    whole model sat in RAM a second time. It is flushed every 512 MiB per device now;
  - the mapped file's pages are let go as soon as a device holds the matrix (they stay in the page cache);
  - placement is planned first, then both devices are fed in turn;
  - from a pack the experts are placed in the file's order, so each device's share is one run of the file, read
    ahead by four threads within a 2 GiB window. Cold, that is 6.9 s against the GGUF's 17.1 s (the NVMe reads
    2.8 GB/s raw). The read-ahead runs for a pack only so far, so this is not yet a like-for-like comparison of the
    two layouts: a GGUF placed whole is also one run, and has not been given the same readers.
- **The tiers are exercised** with small budgets (`--devices cuda:0=4,cuda:1=4 --resident-experts --expert-ram-gib 4`):
  1,540 experts on the two cards, the next 4 GiB copied into RAM (0.5 s warm, on every core), the rest on the SSD.
  Same tokens as the GGUF under the same device budgets; the top-5 logits differ by up to 0.52, because the two
  runs keep different experts on the cards (file order against layer order) and the CPU and GPU kernels round
  differently. 5.5 tokens/s against the GGUF's 3.3 without a RAM tier: two things differ there (the order and the
  tier), so it is not a measure of either. `pack::ram_tier` serves Inkling and any model loaded
  from a pack; a GGUF's MoE blocks keep the older `--resident-experts` path.
- **Inkling's expert profile saves and loads** (it read an empty layer list). The test pack is built with one, but
  from a 12-token run: it shows the mechanics, not real use.
- **Open** (the first two are answered in the next section):
  - a real profile for Inkling (from ordinary use), and tokens/s against the GGUF with it under small budgets;
  - Inkling does not ask for its cold experts ahead (the prefetch is the MoE blocks'); they page in one at a time;
  - the read-ahead passes over most of the core section (the output matrix, at its end, is uploaded first and the
    readers skip what lies before it), and is not given to a GGUF;
  - the padded branch of the upload (IQ3_XXS) passes strata-gpu's kernel tests, not a model run;
  - the device share is uneven (11 GiB and 3.86 GiB): the first card fills before the second gets any;
  - generation speed on the GPUs is I8's work (`INKLING-PLAN.md`), not changed here: 6.8 tokens/s in these runs;
  - the server: a pack in the catalog and the tokenizer read from the pack's header.

## Status, later on 2026-11-23: packs on the live server, Cyber-Tiel, and what blocks a switch

Owner: is the pack in the running service, pack Qwen Flash Next and Cyber-Tiel, use the packed model with a way back
to the GGUF.

- **Not in the running service's code.** The service is built from the main checkout; this work is on
  `engine/inkling2`, unmerged and unpushed. But a pack needs no server change to run: the server reads its catalog
  again on every listing, so a run config naming `am2strata-hybrid` and a `.am2aipac` is a new entry at once.
- **The way back is a second entry.** A pack gets its own entry (`<id>-pack`); the GGUF's entry, its files and the
  service's default are not touched. Going back is choosing the other entry. The configs written here are kept in
  `Strata-data/aipacs/configs/`.
- **Inkling 2 Mini's pack runs through the live server** as `inkling-2-mini-uncensored-ad-q4_k-pack`: loaded by
  `/api/models/load` (43 s with the unload of the model before it), then eight chats, text only, each cut at 96
  tokens: three gave the start of an answer, five spent the 96 tokens on reasoning that reads sensibly (17 x 23
  worked out as 391). Generation 5.9 to 6.8 tokens/s. A chat's prompt arrives with the server's tools, about 1,000 tokens, and the engine takes it
  token by token: 1 to 3 minutes before the first word. Its expert profile is saved by the engine after each
  request (`--expert-profile-save`), so this entry's profile is from real chats: eight (code, Hindi, a story,
  arithmetic with JSON, a recipe, an explanation, a summary), and the pack is rebuilt in that order (16.00 GiB in
  41 s, `verify` ok, the same 12 tokens on the CPU as before). What that profile is worth:
  - about nine tenths of the tokens counted are the server's own tools prompt, the same in every chat;
  - the file keeps ranks, not counts: a later run ranks by its own counts and uses the file only to break ties,
    so the profile does not add up over runs. The eight-chat file is kept as `aipacs/inkling2-mini.8chats.strp`.
  Running it took the usual model off the server for 21 minutes.
- **Cyber-Tiel UD-Q8_K_XL is packed**, with its image projector, its importance matrix and the routing counts of
  its real use (`pack build --profile` takes strata-poly's "layer expert count" text now): 36.84 GiB, 10,240 expert
  records, built in 81 s, `verify` ok in 74 s. On the CPU the pack and the GGUF give the same 12 tokens.
  In `am2strata-hybrid` (`--devices cuda:0=11,cuda:1=11 --resident-experts`: 6,132 experts on the cards, 13.2 GiB in
  the RAM tier):

  | | Pack, Rust engine | GGUF, Rust engine | GGUF, live engine (strata-poly) |
  |---|---|---|---|
  | Load to ready | 13.6 s warm, 11.5 s cold | 33.1 s | 33 s (the server's learnt figure) |
  | Generation | 7.3 tokens/s | 5.7 tokens/s | 63 to 66 tokens/s |

  Each figure is one run. The pack's "warm" run came after a build and a verify under a 6 GB memory cap, so the
  file was probably not all in the page cache (the "cold" run was the faster one); and the two Rust-engine runs
  take their RAM tier by different code. So this shows the pack loading in 11 to 14 s here, not a ratio that
  belongs to the layout. The two runs do not give the same tokens on the cards (different experts sit on the
  cards, and the prompt is 16 arbitrary ids). The CPU run is the like-for-like check, and a thin one: 9 of its
  12 tokens only continue the prompt's count.
- **Not switched, and why.** A pack is read by the Rust engine only, and on a mixture-of-experts model that engine
  is the matrix-by-matrix path: about a thousand products per token each cross to a card and back. That is 7
  tokens/s against the live engine's 65: a faster load and a generation 9 times slower, so Cyber-Tiel's pack is
  built and verified but is not an entry on the server (its config is written, in `aipacs/configs/`).
- **Tried for speed, no gain:** all of a layer's chosen experts queued on the cards before any is read (one wait
  instead of sixteen). Same tokens, 6.5 tokens/s against 6.8, so it is not kept. Why it did not help is not
  measured: the one timed run (`AM2_TIMES`, which now counts device products for Inkling too) ran while a video
  render held every core, and even there the device products were 16.0 s of 37.5 s, 1,073 of them per token. A
  clean timing run is the next step; it decides between the device graph (I8: the token stays on the cards, only
  the router's choice and the logits come back; today it exists for dense models on one card) and the CPU-side
  work between the products.
- **Qwen 3.8 Flash Next is not packed.** Three things stand in the way: its GGUF is two shards (the builder takes
  one file), a pack beside the GGUF needs 86 GB and 87 GB is free, and its architecture runs in the Rust engine
  far slower than live.
- **The GPU test that failed once is explained.** Beside the loaded model (954 MiB free on each card) the kernel
  tests fail every time, serial or parallel: CubeCL cannot allocate a 990 MiB pool page, its stream thread panics
  ("Memory page 0 doesn't exist"), and the products read back are wrong (47 of 48 rows). On empty cards the same
  binary passes in parallel, three runs of three. So it was the full cards. It also shows a hazard: when a card
  runs out, the engine does not stop, it returns wrong numbers. The 11 GiB cap per card is what keeps a model
  away from that today; an explicit check is open.
- **Inkling asks for its cold experts ahead** (`wanted_off_device`, on with `--resident-experts`): the chosen
  experts that no card holds are requested together before the first is multiplied. Same tokens on the CPU;
  its effect on tokens/s under small budgets is not measured yet.
- **The server writes a tokenizer folder from a pack's header** when the folder is missing; the files equal the
  ones written from the GGUF (an ignored test in `manager.rs`, run on Inkling 2 Mini's pack).
- **Open, in order:** the device graph for the MoE path (I8), which decides whether any pack can replace a live
  GGUF entry; prompts in chunks for Inkling; a stop instead of wrong numbers when a card is full; shards as a
  build source; the image section read by the Rust encoder (P4), until then a pack entry is text only.

## Decided (owner, 2026-11-23)

1. The image part of a GGUF model is its `mmproj` projector.
2. The extension is `.am2aipac`.
3. The separate release is under the AGPL (version 3). The `am2pack` crate carries that licence in its own
   `Cargo.toml` and `LICENSE` file from P0.
4. Licences (owner, 2026-11-23: "pick best on AGPL 3"; the choice below is Claude's, made on that word):
   - `am2pack` is `AGPL-3.0-only` (version 3 as asked; the owner holds the whole copyright and can add a later
     version or another licence at any time, which "or later" would have given away to whatever version comes).
   - Prastara's own code stays MIT. Reading and writing packs is a cargo feature (`pack` in `am2strata`, `aipac`
     in `strata-hybrid`), on by default.
   - A binary built with the feature contains `am2pack`, so it is distributed under the AGPL's terms (its source
     offered, also to users over a network). Built with `--no-default-features` it is MIT alone and reads no packs.
