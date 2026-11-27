# Private data (internal)

Prastara builds the owner's private adult datasets itself, from the owner's own local folders, for the owner's
private, abliterated, local-only models (the Mula line): image and video description, including explicit content,
and writing prompts for image generation. This page is internal: the feature stays out of the public README, release
notes and docs.

Code: `crates/strata-workshop/src/private_data.rs` (the library), `am2strata-workshop private-data` (the command), the
server's `private-data` job (`crates/am2strata/src/workshop.rs`, `plan_private`), `am2strata-run workshop
private-data`, and Workshop > Data > Private data in the web app. The minor-safety rules are in
`crates/strata-safety` (shared with the server's check on uncensored models' chats).

## Sources

Only the owner's own local folders, inside the Workshop's allowed folders (the run config's `workshop.dirs`). There is
no crawler and no downloader. Each folder needs a `SOURCE.toml` the owner writes:

```toml
owner_attests_adults_only = true   # everyone shown is an adult (18 or over)
owner_attests_consent = true       # everyone shown consented, or the media is fully synthetic or drawn
origin = "my own photos, 2026"     # free text; it goes on the dataset card
```

A folder without it, or with either attestation not `true`, or without `origin`, is refused, and the refusal names
the file to write. Subfolders (three levels) are covered by their top folder's `SOURCE.toml`; symlinks are not
followed.

What is read in a folder:

| Files | What happens |
|---|---|
| png, jpg, jpeg, webp | described by the vision model; a generation prompt is written from the description |
| mp4 (H.264) | frames through the media engine's Rust decoder (`am2strata-media`'s `video_frames`, on the CPU), each frame as an image; kept frames are saved in `frames/` |
| m4v, mov, mkv, webm | tried the same way; the Rust decoder reads only MP4/H.264, so others are counted as not read |
| txt, md | chunked; the model writes chat rows grounded in each chunk (datagen's `chat` prompt) |
| jsonl | chat records used as seed examples; the model writes new ones like them (datagen's `instruct` prompt) |

## The model

A local model whose run config is marked `"private": true`, `"uncensored": true` or `"abliterated": true` (the
server's `/api/models` lists these as `marks`; the Workshop's install marks an abliterated GGUF this way). Refused:
any name with a `/` (a `<provider>/<model>` remote name), a remote model or a mix, an unmarked model, a partial name,
and, for images and videos, a model that does not read images. The server address (`AM2STRATA_URL`) must be this
machine: localhost, a loopback address, or one of its own interface addresses. The address and key reach the command
in its environment (as for `datagen`), never on its command line.

Every request carries `"strata_local_only": true`: the cluster layer then never forwards it to a worker, and the remote
layer never overflows it to a remote model (and refuses it if it names one).

## Gates (fixed; no setting turns them off)

1. **Minor safety.** Every description, tag list, generation prompt, text chunk (before the model sees it) and text
   row goes through `strata_safety::check` (the server's minor-safety rules). A match drops the item (for text, what
   that file made) and blocks its file's SHA-256.
2. **Age.** The describing prompt asks for `age_assessment`: `adult`, `uncertain` or `minor`. Anything but `adult`,
   including no answer, drops the item and blocks its hash. For a video, one frame that fails drops the whole video
   and blocks the video and that frame.
3. **Real people.** The describing and the prompt requests ask for `real_person_named`. Anything but a plain `false`
   keeps that item out of `prompts.jsonl` (no generation prompts of real people's likenesses); its description row
   stays.
4. **Blocklist.** `private/blocklist.sha256` (one `sha256 gate` per line, mode 0600). Every later run, of any private
   dataset, skips a blocked file without sending it to the model.

The card counts what each gate dropped and lists the hashes this run blocked. It never holds any dropped content.

## Output

`<data>/private/<name>/` (by default `~/Development/Strata-data/private/<name>/`), the folders mode 0700, the files
0600:

- `vision.jsonl`: `{"image", "description", "tags", "private": true, "meta": {"model", "sha256", ...}}` (a frame's
  meta also has `video`, `video_sha256` and `t`)
- `prompts.jsonl`: `{"prompt", "image", "private": true}`
- `text.jsonl`: `{"messages": [...], "meta": {"recipe": "private-text", "model", "source"}, "private": true}`
- `frames/`: the kept video frames, named by their SHA-256
- `dataset.card.md`: starts with `<!-- prastara: private -->`, says private, the sources with their origin and
  attestations, the counts and the gates

A name that exists already is refused. The server's Workshop adds `<data>/private` to its folders, so train, clean and
review jobs can read the datasets.

## Private propagation

A dataset is private when a row carries `"private": true` or its card starts with the private mark
(`private_data::is_private_dataset`).

- **clean**: an input that is private makes every kept row `"private": true` (rows from ordinary inputs mixed in too),
  the card private, and the output files 0600.
- **train**: data or eval data that is private writes `"prastara_private": true` into the adapter's
  `adapter_config.json` (checkpoints too), and `training_report.json` says `"private": true`.
- **merge**: an adapter marked private marks the merged folder's `config.json` `"prastara_private": true`; convert
  then writes `prastara.private` into the GGUF, and the install marks the run config `"private": true`.

## Running it

```sh
# through the server (the job shows under Workshop > Jobs)
am2strata-run workshop private-data ~/Development/Strata-data/mula/my-photos --name my-set --writer mula-abliterated

# on its own, against the server on this PC
AM2STRATA_URL=http://127.0.0.1:8000 AM2STRATA_API_KEY=... am2strata-workshop private-data --name my-set \
  --source ~/Development/Strata-data/mula/my-photos --model mula-abliterated
```

Server job: `{"kind": "private-data", "name", "sources": [...], "model"?, "text_count"?, "per_chunk"?,
"chunk_chars"?, "temperature"?, "every_s"?, "max_frames"?}`.

## Known gaps

- Video frames need the media engine binary (`AM2STRATA_MEDIA_EXE`, or `am2strata-media` beside
  `am2strata-workshop`); without it videos are skipped and counted on the card. Its decoder reads MP4/H.264 only, and
  only keyframes of a B-frame MP4.
- The age, minor-safety and real-person gates rest on the describing model's answers and fixed word rules; they are
  strict by design and can drop innocent items.
- Chats with a private model from other clients (not this job) can still be routed by the cluster or overflowed by
  the remote layer, unless they send `"strata_local_only": true`.

# Import (a dataset brought in, quarantined, checked, then released)

The owner's own adult dataset from somewhere else, imported into a locked quarantine and released only after every
item passed fixed checks on this PC. Code: `crates/strata-workshop/src/private_import/` (`mod.rs` the run,
`fetch.rs` the URL and Hub fetchers, `archive.rs` the unpacking guards, `checks.rs` the `Classifier` trait and the
rules, `nets.rs` the check models on candle), the fixed thresholds in `strata_safety::import`, the command
`am2strata-workshop private-import`, the server's `private-import` job (`crates/am2strata/src/workshop/private_import.rs`),
`am2strata-run workshop private-import`, and Workshop > Private data > Import a dataset in the web app.

## Sources

One per import:

| Source | Request | Notes |
|---|---|---|
| a folder | `{"folder": "/mnt/nas-download/<sub>"}` | inside the Workshop's folders (add a NAS mount to `workshop.dirs`); opened read only, links never followed (`O_NOFOLLOW`), hidden files and `SOURCE.toml` left out, a file that disappears during the walk skipped; nothing is ever written in it |
| an https URL | `{"url": "https://..."}` | an archive or one file; http only to this machine's loopback (tests); redirects only to https |
| a Hugging Face dataset | `{"hf": "owner/name", "revision"?: "main", "pattern"?: "data/*.zip"}` | the Hub's HTTP API (`HF_ENDPOINT`, `HF_TOKEN` in the server's environment): the revision pinned to its commit, the files listed, each downloaded resumable and checked against the SHA-256 the Hub lists for LFS files |

Each needs a `SOURCE.toml` (the format above). A folder's own file is used when none is given (a read-only NAS
folder cannot hold one: write it elsewhere and give `attestation`). A `SOURCE.toml` outside the folder must name the
source it is for, so one attestation is never reused for another source:

```toml
owner_attests_adults_only = true
owner_attests_consent = true
origin = "where it came from"
licence = "its licence"            # needed for a URL or Hub source
url = "https://example.com/set.zip"  # or: hf = "owner/name", or: folder = "/mnt/nas-download/set"
```

**Nothing is fetched on its own.** The owner starts every fetch: `POST /v1/workshop/private-import/probe` gives the
source's files, bytes and Hub commit (the UI's confirm dialog names the source and its size; the CLI asks
`[y/N]`, or takes `--yes`), and the job then needs `"confirm_bytes"`: if the source's size changed since, nothing is
fetched. There is no schedule and no automatic retry.

## Quarantine

`<data>/private/quarantine/<import-id>/` (0700): `items/` (every item by its SHA-256 and extension: no source names
kept), `incoming/` (downloads, resumable `.part` files), `import.json` (0600: the state, counts and item hashes).
Archives (zip, tar, tar.gz, told by their first bytes) are unpacked once; entry names are never used as paths; an
entry with an absolute name or `..` (zip-slip) rejects the whole archive, links and devices are skipped, and the bytes
are counted as they come out: a file over 8 GiB, more than 256 GiB in all, more than 200,000 files or an entry that
expands more than 200 times past its first MiB (a bomb) rejects the whole archive (what it gave is removed).

Nothing in quarantine is used: the server's path check (`Workshop::allowed`) refuses any path under
`private/quarantine` (a link into it too) for every job kind (convert, quantize, merge, loop, upcycle, export-moe,
abliterate, train, strata, pack, datagen, clean, regress, grade-train, private-data, private-import's own folder
source), and the browse, dataset and files listings never show it; `am2strata-workshop` itself refuses a quarantine
path in the arguments of every command but the import.

## Checks (fixed; no setting turns them off)

Every item, in this order; one flag or failure ends it:

1. **Blocklist**: a hash on `private/blocklist.sha256` is deleted at once (counted as a minor flag).
2. **Type**: images by their first bytes (PNG, JPEG, WebP), text by extension (txt, md, jsonl, json, csv, tsv) and
   valid UTF-8; anything else is rejected (`unsupported_type`), an image that does not decode (`undecodable`;
   decoded under size limits) too.
3. **Images**: the image minor-safety check, then the age estimate (skipped only when the age model's own no-person
   label wins), then the real-person check.
4. **Text**: the fixed word rules (`strata_safety::check`), then the text minor-safety check (the text in windows of
   the model's length; one window that flags flags the item; more than 4,096 windows is a rejection), then the text
   real-person check.

**Fail closed**: an error, a panic, no scores or scores that are not numbers is a rejection (`check_failed`); a check
that does not answer within the timeout (120 s, `--check-timeout-s`) stops the import ("the <x> check model did not
answer in time; nothing is released"), the items not yet checked staying in quarantine for a resume.

**Minors**: an item flagged by a minor check (image minor-safety, age, text minor-safety, the word rules) or on the
blocklist is unlinked from quarantine at once, its hash goes on the blocklist (`import_<reason>`), and the audit log
gets the hash and the reason only. When more than 1 % of the source's items are so flagged
(`strata_safety::import::MAX_MINOR_SHARE`; `"minor_share"` can only lower it), the whole import is rejected and its
quarantine deleted, and nothing is released. Real-person flags are rejected (not released) but not blocked.

**Release**: only after every item is checked and the share is under the limit: the items that passed are copied to
`<data>/private/datasets/<name>/` (0700; files 0600): `files/<sha256>.<ext>`, `manifest.jsonl` (`{"file", "sha256",
"kind", "bytes", "import", "private": true}`), `text.jsonl` (chat rows of released JSONL files kept and marked
`"private": true`, other text as `{"text", "private": true}`) and `dataset.card.md` (the private mark, the source,
origin, licence, commit, counts, rejections by reason and the check models' rules; never content). Then the
quarantine is deleted. Clean and train carry the private mark from these files as from any private dataset.

## The check models

None is downloaded by Prastara. The owner installs each as a Hugging Face model folder in
`<data>/private/check-models/<check>/`, with the model's `config.json`, `model.safetensors` (or `model-*.safetensors`
shards; any float type), for text its `tokenizer.json`, for images its `preprocessor_config.json` when it has one,
and a `prastara_check.json` the owner writes. They run on candle on the CPU (f32). A folder with only `model.onnx`
is refused (check models are read as safetensors). A missing one stops the import in quarantine: "the <x> check model
is not installed; nothing is released". Only the checks the items need are loaded (images need the three image
checks, text the two text checks).

| Folder | Check | Architecture | Input | Output read |
|---|---|---|---|---|
| `image-minor` | image minor-safety | `ViTForImageClassification` | the image resized to the model's size (`preprocessor_config.json` size, mean, std; else `image_size`, 0.5/0.5), RGB | softmax over `id2label`; flagged when the `flag_labels` together score at least the threshold |
| `age` | age estimate | `ViTForImageClassification` | as above | softmax over age-range labels ("0-2", "10-19", "20-29", "more than 70", ...): every range reaching under 18 flags unless `flag_labels` are given; `no_person_labels` skip the check; or one output (regression): the age in years, flagged under 18 + `age_margin` (at least 3) |
| `image-real-person` | real, identifiable person | `ViTForImageClassification` | as above | softmax; `flag_labels` |
| `text-minor` | text minor-safety | `BertForSequenceClassification`, `RobertaForSequenceClassification` / `XLMRobertaForSequenceClassification`, `DistilBertForSequenceClassification` | token windows of at most `max_position_embeddings` (with the tokenizer's special tokens) | softmax; the highest score of each label over the windows; `flag_labels` |
| `text-real-person` | text names or describes a real person | as text-minor | as text-minor | as text-minor |

`prastara_check.json`:

```json
{"check": "image-minor", "flag_labels": ["<the model's label for a minor>"], "threshold": 0.2}
{"check": "age", "no_person_labels": ["<its no-person label, if it has one>"], "adult_labels": [], "age_margin": 3}
```

`check` must be the folder's check; every label named must be one of the model's own (or it does not load). The
threshold is the share of probability that flags: by default 0.2 for the minor checks (at most 0.5) and 0.5 for the
real-person checks (at most 0.9); a higher one is cut to the maximum.

## Audit log

`<data>/private/audit.jsonl` (0600, appended, never rewritten): one JSON line per event, each with `t` (unix
seconds), `import` and `event`: `start` / `resume` (the source, the attestation's yes and licence), `walk` (files,
skipped), `fetch` (the source, commit, SHA-256, bytes, whether the SHA-256 was checked against the source's),
`unpack` / `archive_rejected` (the archive's SHA-256, counts or the guard), `quarantined` (counts), `reject` (SHA-256,
reason), `delete` (SHA-256, reason: nothing else), `stopped` (why, counts), `import_rejected` (the share), `release`
(the dataset folder, counts, the released hashes) and `discard`. It never holds content, file names or thumbnails.

## Never leaves the machine

- A `private-import` or `private-data` job runs on this machine only (the cluster refuses it, `"any"` or a named
  worker alike), and anything in `<data>/private` (the released datasets and the quarantine) is never sent to a
  cluster worker, even a named one (`private_data`). Other private-marked inputs keep the rule above: a named worker
  only, never `"any"`.
- A datagen, regress or grade-train job over private data refuses a remote model or a mix as its writer or judge (a
  `<provider>/<model>` name, or a name this server does not have), and runs with `--local-only`: every chat it sends
  carries `"strata_local_only": true`, so no worker and no remote overflow answers it.
- The check models are local folders; the import sends nothing to any model server.

## Running it

```sh
am2strata-run workshop private-import /mnt/nas-download/set --name my-set --attestation ~/Development/Strata-data/mula/set.SOURCE.toml
am2strata-run workshop private-import https://example.com/set.zip --name web-set --attestation ./SOURCE.toml
am2strata-run workshop private-import hf:owner/name --revision main --pattern "data/*.zip" --name hub-set --attestation ./SOURCE.toml
am2strata-run workshop private-imports                    # the imports in quarantine (counts, why stopped) and the released sets
am2strata-run workshop private-import --resume my-set-1795789257 --name my-set   # check again after installing a model
am2strata-run workshop private-import-discard my-set-1795789257
```

Server job: `{"kind": "private-import", "name", "source": {...}, "attestation"?, "confirm_bytes", "minor_share"?,
"resume"?}`; `GET /v1/workshop/private-imports`; `POST /v1/workshop/private-imports/{id}/discard`. The job's progress
carries `counts` (`fetched`, `checked`, `rejected`, `deleted`, `released`, ...); a stopped or rejected import is a
failed job whose error says why, and nothing is released.

## Known gaps (import)

- Videos, Parquet files and nested archives are not checked, so they are rejected (`unsupported_type`), never
  released. A Hub dataset that keeps its images inside Parquet gives nothing until a Parquet reader exists.
- Image check models: ViT only; text: the BERT family only. An ONNX-only model needs converting to safetensors first.
- There is no person detector: the age check runs on every image unless its own model says no person is shown.
