# Index format (`primolix`)

> This page comes from an experiment you can rerun: build a small index, list every file, then delete
> each part in turn and try to query. Environment: this machine, Python 3.13, a 4-document corpus.
> `fmt` = `primolix-bm25-v3`.
> Chinese version: [index_format.md](index_format.md)

## 1. What the directory looks like

```
<index dir>/
├─ bm25.npz                       <- the primitives (the only real data)
├─ bm25.npz.mm/                   <- mmap materialisation of the postings (derived)
│   ├─ T_data.npy  T_idx.npy  T_ptr.npy  T_shape.npy
│   ├─ dl.npy  idf.npy  dead.npy
│   ├─ mm_key.txt                 <- **generation key**: the npz's "name|mtime|size" (a mismatch refuses the whole `.mm/`)
│   ├─ w_0.npy … w_k.npy          <- materialised layer (only with `--w-cache` / `save(w_cache=True)`)
│   └─ w_key.txt                  <- its content key (compared on load; a mismatch is refused)
├─ .segments/                      <- derived artefacts of staged publish (only when `stage_*` was used; the whole directory is safe to delete)
│   ├─ manifest.jsonl              <- append-only event log (`publish` / `folded_all` / `reset`)
│   └─ <tag>/                      <- the segment triple plus `dl.npy` (see the last section)
├─ bm25.npz.vocab/                <- vocabulary carrier (mmap friendly, derived)
│   ├─ vocab.blob  vocab_offs.u32  vocab_sorted.u32
│   ├─ vocab_hot.u32              <- hot terms (optional)
│   └─ vocab_key.txt              <- **generation key**: the npz's "name|mtime|size" (a mismatch, a missing key or a length mismatch refuses the whole carrier)
├─ meta.jsonl                     <- per-chunk metadata (file / line / text / row)
└─ hashes.json                    <- source file md5 (lets `update` detect changes)
```

Both `.mm/` and `.vocab/` are **derived**: deleting them only makes the next load rebuild or degrade,
never changes correctness (`bm25.npz` is the one thing you must keep). The materialised layer is **not
written by default**; turn it on and you get `w_*.npy` plus `w_key.txt`, paying `+nnz x 4 B` on disk to
skip the rebuild on a cold start.

**What the generation key means**: `mm_key.txt` pins `.mm/` to one specific `bm25.npz` (name / mtime /
size). A mismatch, a missing key, or a truncated key makes the loader **refuse the whole `.mm/`** and read
`T_data` from the npz instead (equivalent results, just no mmap). So every residue of an interrupted save
is **fail-safe**: you never get "new npz with old postings" silently.
Upgrade note: indexes written **before** the key existed have no `mm_key.txt`, so their first load
**refuses** `.mm/` (correct, slightly slower, one WARN line); **saving once more** restores the mmap path.

**What the vocabulary carrier's generation key means**: `vocab_key.txt` pins `.vocab/` to the same
`bm25.npz` (name / mtime / size, with the vocabulary length appended for readability). When the carrier
belongs to a **different generation**, the **key is missing**, or its **length does not match the npz's
own vocabulary**, the loader **refuses the carrier** and falls back to the **vocabulary stored inside
the npz** (the `vocab` key is always present; results are equivalent, only the carrier's mmap is
skipped). So residue such as "the sidecar write failed during an in-place re-save" can no longer make
new terms silently disappear.

**Practical result for multi-process deployments (measured)**: once a reader process has mmap'd `.mm/`,
a writer's **in-place re-save** can only replace `bm25.npz` -- `.mm/` and `.vocab/` fail to be written
(Windows raises `OSError(22)` and the library logs a `[WARN]`). So to publish a new generation, **write to a
new directory** (or re-save in a window with no readers). A "half-published" directory is still **safe**:
a fresh reader finds the generation key mismatched, **refuses the stale `.mm/`**, and reads a **complete**
generation straight from the npz (just without mmap).

Keys inside `bm25.npz` (measured):

| Key | dtype / shape | Meaning |
|---|---|---|
| `fmt` | string | `primolix-bm25-v3` (format version) |
| `T_data` / `T_idx` / `T_ptr` | float32 / int32 / int32 | CSR triple of the **`tf` primitive** |
| `T_shape` | int64 x2 | `(rows = documents, cols = vocabulary)` |
| `df` | int32 x V | document frequency |
| `dl` | float32 x N | document length |
| `_idf` | float32 x V | idf at save time (recomputable at query time) |
| `dead` | bool x N | **tombstone** bitmap (rows retired by incremental writes) |
| `N_live` | scalar | live document count |
| `params` | float32 x3 | `k1` / `b` / `avgdl` |
| `vocab` | object x V | vocabulary (term -> column) |
| `W_shape` | int64 x2 | compatibility (only old v1/v2 stored the `W` score matrix) |

In one line: **`bm25.npz` holds invertible primitives; everything else is a materialisation or a
manifest that can be rebuilt from it.**

## 2. Which parts are required (measured: delete one, then query)

| Part removed | Result (one query, this run) |
|---|---|
| `bm25.npz.mm/T_idx.npy`, `T_ptr.npy` | fails: `FileNotFoundError` |
| `bm25.npz.mm/idf.npy` | fails: **`TypeError: 'NoneType' object is not subscriptable`** (unhelpful) |
| `bm25.npz.vocab/` (all three files) | fails: `FileNotFoundError` |
| `bm25.npz.vocab/vocab_key.txt` | not tested in isolation: by the rule the carrier is **refused** and the npz's own vocabulary is used (equivalent results, only the carrier mmap is skipped) |
| `meta.jsonl` | fails: `FileNotFoundError` |
| `bm25.npz.mm/dl.npy` | **passes this run** |
| `bm25.npz.mm/T_data.npy` | **passes this run** |
| `bm25.npz.mm/T_shape.npy` | **passes this run** |
| `hashes.json` | **passes this run** (`update` loses its basis for "which files changed") |

> **"Passes this run" is weak evidence**: a single query against a 4-document index only shows that
> *this path* does not read it — it does **not** mean the part is disposable. The safe rule:
> **do not delete individual files**; to clean up, rebuild.

## 3. Portability

- **Copying the whole directory elsewhere was queryable in our run** (relative references internally,
  no hard-coded absolute paths).
- A read-only mount or read-only share is safe: queries only read; only `update` writes.
- Copy **the whole directory** (missing parts fail as in section 2).

## 4. Versions and compatibility

- The version lives in the `fmt` key of `bm25.npz`: `primolix-bm25-v3` (current).
- **Indexes written before the rename (`fmt = zelix-bm25-v3`) still load**: the reader normalises the
  old format string to the new name (the on-disk format did not change, only the package/class name),
  so **upgrading does not require rebuilding**.
- Older formats (`v1` / `v2`, carrying the precomputed `W_data` score matrix) **still load** too: they
  restore the old fast path. New indexes no longer store `W` (scores are derived).
- There is **no** cross-major-version migration tool: to change major version, rebuild the index
  (rebuilding is deterministic and the readings are reconcilable).

## 5. Rebuild and cleanup

- **Rebuild**: `python -m primolix index <dir> --out <index>` (the old directory is overwritten).
- **To save space**: delete the **whole index directory** and rebuild — safer than deleting individual
  files (see the weak-evidence note in section 2).
- **Incremental**: `python -m primolix update <dir> --out <index>` (uses `hashes.json`; editing one
  document does the work of that one document).

## 6. Known unhelpful errors (to fix)

- Missing `idf.npy` reports `TypeError: 'NoneType' object is not subscriptable`;
- Missing other parts report a bare `FileNotFoundError` with no context.

Desired behaviour: an integrity pre-check on load that reports "index incomplete: missing `X`; rebuild
with `index` or restore that file".

## Appendix: `.segments/`, the derived artefacts of staged publish (safe to delete)

    <idx>/.segments/manifest.jsonl      append-only event log
    <idx>/.segments/<tag>/<tag>.data.npy   the segment triple (CSC)
    <idx>/.segments/<tag>/<tag>.idx.npy
    <idx>/.segments/<tag>/<tag>.ptr.npy
    <idx>/.segments/<tag>/dl.npy           per-row document lengths

`manifest.jsonl` is **append-only** and holds three event kinds: `publish` (the commit point),
`folded_all` (segments have been folded into the main table) and `reset` (a full rebuild invalidates
earlier events). On load the events are replayed in order and segments that are still mounted are
re-attached; segments that were folded, or that belong to a **different vocabulary generation**, are
**skipped and reported** -- never silently mixed. Saving is refused while segments are mounted (the main
table cannot represent their rows): `save(fold_first=True)` folds first and then writes. The whole
`.segments/` directory can be deleted; the only cost is re-publishing those batches.

## 7. Reproducing this page

1. Build a small index from a few documents: `python -m primolix index <corpus dir> --out <index dir>`.
2. List every file under the index directory.
3. Back the directory up, then delete one file at a time and run
   `python -m primolix query "..." --out <index dir>`; note whether it passed or failed.
4. Collect "which file was deleted, what happened" into the table in section 2.
