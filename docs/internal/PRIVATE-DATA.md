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
