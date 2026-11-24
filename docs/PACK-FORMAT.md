# The `.am2aipac` pack format, version 1

One file that holds a model, its image projector and optional extras, laid out for a mixture-of-experts engine. The
reference reader and writer are the `am2pack` crate (`crates/am2pack`, AGPL-3.0). All integers are little-endian.

## Layout

```
0            header, 4096 bytes
4096 ...     sections, each starting on an aligned offset; zeros between them
...          the index (JSON), starting on a 4096-byte boundary and ending at the end of the file
```

### Header (4096 bytes)

| Offset | Size | Field |
|---|---|---|
| 0 | 8 | magic, the ASCII bytes `AM2AIPAC` |
| 8 | 4 | format version (1) |
| 12 | 4 | header length (4096) |
| 16 | 8 | index offset |
| 24 | 8 | index length |
| 32 | 8 | file length |
| 40 | 32 | SHA-256 of the index |
| 72 | 32 | SHA-256 of bytes 0 to 71 |
| 104 | 3992 | zeros |

The index must end exactly at the file length. A reader refuses a file whose version is higher than it knows.

### Sections

A section is a contiguous run of bytes with a kind, a name and its own SHA-256. The bytes between sections, and
between the last section and the index, are zeros; `verify` checks that. Padding inside a section belongs to the
section and is covered by its checksum.

| Kind | Holds | Starts on |
|---|---|---|
| `gguf-header` | A GGUF source's header, verbatim (bytes 0 to its data start): the model's settings, tokenizer, chat format and tensor table. Name: the source's role. | 4 KiB |
| `core` | Every model tensor that is not split into expert records. Tensors before the layers first, then `blk.0`, `blk.1`, ... in order, then the tensors named `output*`. | 2 MiB |
| `experts` | One record per (layer, expert). | 2 MiB |
| `image` | The image projector's tensors, in its own order. | 2 MiB |
| `gaps` | Bytes of a source that are neither header nor tensor and are not zero. | 4 KiB |
| `imatrix`, `profile`, `defaults` | A file kept whole: the importance matrix, the expert profile, the run defaults (a JSON object). | 4 KiB |

A reader skips section kinds it does not know, so later versions can add kinds without breaking older readers.

Every tensor, and every part of an expert record, starts on a 64-byte boundary. Tensor bytes are the source's bytes,
unchanged: the same GGML encodings, the same row order.

### Expert records

A model tensor is split when all of these hold: its name starts with `blk.<N>.` and contains `_exps`, it has three
dimensions, its last dimension equals `<architecture>.expert_count`, and one expert's slice is a whole number of
bytes and of quantization blocks. Slice `e` of a split tensor is bytes `e * len / count` to `(e + 1) * len / count`
of its data.

The record of (layer `N`, expert `e`) starts on a 4 KiB boundary and holds slice `e` of each split tensor of layer `N`,
in the source's tensor order. So a cold expert is one read of whole pages, and the engine never has to gather an
expert from several places.

Record order: with a profile (STRP version 1, a ranked list of (layer, expert) pairs), the ranked experts come first
in that order, hottest first, then every other expert by layer and number. Without one, by layer and number. The
manifest's `expert_order` says which (`profile` or `layer`).

### The order is the tiers

The engine places experts in record order: the first run of the expert section goes to the first GPU, the next run to
the second, the run after that into RAM, and the tail stays on the SSD. A pack built with `--profile` therefore has
its hottest experts on the fastest tier, and each tier is one sequential read. No tier boundary is stored: the same
pack serves any machine, and the budgets given at load decide where each run ends.

### Index (JSON)

```
{
  "format": "am2aipac", "version": 1,
  "manifest": { "name", "architecture", "created_by", "expert_order", "expert_count", "extra": {...} },
  "sections": [ { "kind", "name", "off", "len", "sha256" } ],
  "sources":  [ { "role", "file", "size", "sha256",
                  "header": <section index>            (a GGUF source)
                  "whole":  <section index>            (a file kept whole)
                  "gaps": [ { "src_off", "len", "off" } ] } ],
  "tensors":  [ { "src": <source index>, "name", "type", "dims", "src_off", "len",
                  "off": <file offset>                 (a whole tensor)
                  "split": <expert count> } ],         (a split tensor: exactly one of off and split)
  "experts":  [ { "layer", "expert", "off", "len",
                  "parts": [ { "tensor": <tensor index>, "off", "len" } ] } ]
}
```

Every `off` is from the start of the file. `src_off` is where the bytes sat in the source file. `dims` are as in GGUF,
innermost first. `type` is the GGML type's name. Roles: `model`, `mmproj`, `imatrix`, `profile`, `defaults`. A reader
ignores keys it does not know.

## Getting the sources back

A GGUF source is its header section at offset 0, each whole tensor at its `src_off`, slice `e` of each split tensor at
`src_off + e * part length`, each stored gap at its `src_off`, and zeros everywhere else, up to `size`. The result's
SHA-256 equals the source's recorded `sha256`. A file kept whole is its section.

The writer refuses a source whose tensors overlap or run past the end of the file, and a GGUF that is one shard of a
split model.

## What `verify` checks

1. The header's own checksum, the index's checksum, and that the file is exactly as long as the header says.
2. Every section lies inside the file, sections do not overlap, and each one's SHA-256 matches.
3. The header's padding and all bytes between sections are zero.
4. Every source, rebuilt in memory, has its recorded length and SHA-256.

## Not in version 1

- Building from a full-precision model with quantization, or from safetensors (plan steps P5). The container already
  takes sections as byte streams, so these add sources, not format changes.
- Split GGUF sources.
