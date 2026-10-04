# API reference (primolix)

> Public interfaces and **contracts** only; private methods are omitted. Items marked as contracts
> were measured in this repo and are pinned by tests. Pitfalls we actually hit are listed too.
> Index files and compatibility: [index_format.en.md](index_format.en.md).
> Chinese version: [api.md](api.md)

## 1. Top level

```python
from primolix import Primolix, SparseBM25, zh_tokenize, simple_tokenize, log
```

| Name | What it is |
|---|---|
| `Primolix` | **index object**: scans a directory of documents, chunks, persists, owns views |
| `SparseBM25` | **kernel**: BM25 sparse matrix + true incremental writes + blocked-CSC queries (`primolix.kernel`) |
| `zh_tokenize` / `simple_tokenize` | Chinese tokeniser (jieba; degrades to char + adjacent bigram when missing) / English |
| `log` | logging hook (writes to **stderr**; on the library path it honours `core.VERBOSE`) |

## 2. `Primolix` (index object)

```python
z = Primolix.build("docs/", "docs/.idx")      # build; classmethod, returns an instance
z = Primolix("docs/.idx")                      # open an existing index
```

| Method | Notes |
|---|---|
| `Primolix.build(root, out='.primolix_index', dense=False, model_path='', workers=0, skip_dirs=None)` | build an index (classmethod); `dense=True` also writes `dense.npy` |
| `Primolix(index_dir, dense=False, model_path='')` | open an index (reads from disk) |
| `query(q, k=10, dense=False)` | search -> `[{'rank','score','file','line','text'}]`, **score descending**, truncated at `score <= 0` |
| `update(root, workers=0)` | incremental update driven by source-file md5 -> **returns `{'kind', 'detail'}`** (see contracts in section 5) |
| `rebuild_in_place(root, workers=0)` | full rebuild into the same directory (out-of-vocabulary fallback / explicit rebuild) |
| `info` | attribute: `N` (chunks), `V`, `size_mb`, ... |
| `browse(file=None, page=0, n=20)` | browse chunks (used by the CLI `browse` command) |
| `list_files()` | list indexed files |

**Views** (multi-version / A-B / dense reranking candidates):

| Method | Notes |
|---|---|
| `attach_view(name, vectors, meta_positions=None, *, ...)` | attach a view (vectors aligned to global docids) |
| `detach_view(name)` / `view_report()` / `covered_mask(view_name)` | detach / report / coverage mask |
| `save_view(name, docids, vectors, meta=None)` / `load_views(extra_dirs=None)` | persist / load views |
| `fused_scores(qtoks, qvec, view_name, alpha=0.35)` | fused sparse + dense scoring |

Boundary: the view family ships with the package, but **the released CI does not cover the view
path** (end-to-end and kernel unit tests cover build, query, incremental write, formula change,
delete, reload, plus segment mount/fold/unmount U10-U12) -- verify it yourself before relying on it.

## 3. `SparseBM25` (kernel)

```python
from primolix.kernel import SparseBM25

bm = SparseBM25(texts, lambda s: s.split(), k1=1.5, b=0.75)   # from texts
bm = SparseBM25.load("idx/bm25.npz")                          # from disk (classmethod)
```

| Member | Notes |
|---|---|
| `score_all(qtoks)` | **takes a token list only** (not a string) -> returns `float32[N]` scores |
| `update_docs(rows, toks_list)` | true-incremental upsert: **tombstone old rows + append new rows**, returns new row ids |
| `append_docs(toks_list)` | append only (no tombstoning) |
| `oov_terms(toks_list)` | **out-of-vocabulary pre-check** (empty set means the incremental path is safe) |
| `live_rows()` | currently live row ids (`range(N)` when there are no tombstones) |
| `save(path, w_cache=False)` / `load(path, N=0, V=0)` | persist / load (`load` also accepts the old `zelix-bm25-v2/v3` format strings). With `w_cache=True` the materialised layer `w` is **additionally** written into `.mm/` (`w_*.npy` plus the content key `w_key.txt`; a key mismatch on load is **refused**) so a cold start need not rebuild it (measured at 1M: first query 238.3 -> 3.1 ms; costs `+nnz x 4 B` on disk) |
| `SparseBM25(docs, tok, k1=1.5, b=0.75, workers=0, stream=False, vocab=None)` | constructor. **`vocab=`** gives the **column-ordered vocabulary** (shard builds need one consistent column space; out-of-vocabulary terms are **refused, never silently added**) · **`stream=True`** tokenises twice without holding `doc_toks` (memory `O(N x len)` -> `O(triples)`) · **`stream="ids"`** builds from token ids (paired with `vocab=` for per-shard builds) |
| **`mount_segment(seg_dir, tag, new_terms, dl, rows=None, *, generation=None)`** | **mount one read-only segment** of postings (three `.npy` files, CSC, columns in the "widened main" space): attach it as a block on the query path + **merge `df`** (add for old terms, extend for new ones) + extend the vocabulary/`dl`/`N` + recompute `_idf` -> **bit-identical to a full rebuild** (`max\|delta\| = 0` measured). Never writes to disk, never touches the main `T`. Cost: the vocabulary is overlaid with a plain dict (the carrier is read-only) |
| `unmount_segment(tag)` | **withdraw a mounted segment**: undo everything `mount_segment` did (detach the block, subtract its `df` contribution, shrink the vocabulary/`dl`/`dead`/`N`, recompute the statistics), returning scores **bit-exactly to the pre-mount state** (test U12). Two preconditions, both raised explicitly: (1) the segment has **not been folded** (its content is in the main table by then, so only a rebuild can undo it); (2) **last-in, first-out** (it must be the most recently mounted one, otherwise the row/column accounting would be wrong) |
| **`segment_report()` / `folded_report()`** | mounted-segment list (with `gen` vocabulary generation, `row_base`, `m_new`, `df_added`, `bytes`) / archive of **folded** segments |
| **`fold_segment(tag)`** | **fold a mounted segment into the main table** (widen columns + append rows, one `O(nnz)` pass) -> after folding, queries touch the main table only. `df`/`dl`/vocabulary were already merged by `mount_segment`, so this **does not touch them again** (that would double-count). Returns `{rows, ms, nnz_before, nnz_after, ...}` and moves the tag into `folded_report()` |
| **`fold_all()`** | **fold every mounted segment into the main table in one pass** (ordered by base row, blocks padded to a common column count, then an assertion that the main table's row count equals `N`). The single-segment fold does not hold with several segments mounted (it asserts "rows after folding equals `N`", while `N` counts **all** mounted segments), so use this one for multiple segments; the facade's `compact()` picks it automatically. Returns `{tags, rows, ms, nnz_before, nnz_after, N, V}` |
| **`topk(qtoks, k=10, *, sort=True)`** | **score + take top-k in one call**, returning `(indices, scores)` (score-descending by default). Saves the full-vector negation copy that "`score_all` then `np.argpartition(-s, k-1)`" performs (measured **0.47 ms / 2 MB** saved at 500k; that copy grows ×49 once it leaves cache). Takes a **token list**, like `score_all`. On ties at the k boundary the choice is unspecified (score values are identical) |
| `k1`, `b`, `avgdl`, `dl`, `df`, `N`, `V`, `nnz` | parameters and statistics (`V`/`nnz` are properties) |

**Class-level switches**:

| Switch | Default | Effect |
|---|---|---|
| `SparseBM25.T_SLACK` | `1.5` | row-direction capacity slack (amortises append to `O(new nnz)`) |
| **`SparseBM25.FAST_SCORE`** | **`True`** (since 2026-10-03) | **query fast path** (avoids `getcol` objects/copies; drops intermediate variables). **Bit-exact** (only changes that preserve float operation order). The three switches below only take effect when this is `True` (leaving it `False` disables all three) |
| `SparseBM25.TC_SHARED` | `False` | put the blocked CSC in shared memory (several processes share one copy) |
| `SparseBM25.WRITE_DICT` | **`False`** | use a dict on the write path (measured **no gain, 0.99x**; not recommended) |
| **`SparseBM25.DN_PRECOMP`** | **`True`** (since 2026-10-03) | precompute the per-document normalisation term `dn` (**N-level**, 4 MB at 1M) -> **+8-12%**, **bit-exact** (U14). Only takes effect when `FAST_SCORE=True` |
| **`SparseBM25.ADD_AT`** | **`True`** (since 2026-10-03) | accumulate with `np.add.at` (**unbuffered**) -> **+40-65%** (500k hit path 1.87 -> 1.02 ms), **bit-exact** (U17). The *gain* depends on numpy internals (**may regress, will not miscompute**); also only with `FAST_SCORE=True` |
| **`SparseBM25.W_CACHE`** | `False` (**opt-in**) | **materialised layer `w`** (`nnz x 4 B`, 165 MB at 1M): one add per posting on hit -> **+60%** (2.4-3.2x total), **bit-exact** (U16), invalidated automatically on parameter change (key includes `k1/b/avgdl/generation/block count`). First hit pays an **O(nnz) build** (~0.6 s at 1M); **optionally persisted** (`save(w_cache=True)` writes `.mm/w_*.npy` plus a content key `w_key.txt`, refused on mismatch at load time); **not used while a segment is mounted** (queries inside a batch take the computed path, and the first query after a fold rebuilds it once) |
| `SparseBM25.MATVEC` | `False` (not recommended) | M1 hand-written matrix form -- **measured regression** (0.728x at 500k, 27% slower) -> kept as a recorded negative result |
| `SparseBM25.SWAP_LOCK` | **`True`** | **concurrency safety: segment swaps and queries exclude each other** through a reentrant reader/writer lock (concurrent readers, exclusive writer, writer-preferring). Basis: with no lock about **3%** (2.3%-3.7% across repeated measurements) of concurrent reads were torn, **96% of them silently wrong**; with the lock, **0 torn** measured. Cost **+3.43 us/query** (17.6% on a tiny index, about 7% at 44,757 chunks) -- set it to `False` for single-threaded use with no concurrent swaps |

**How to choose a switch (cost and payback)**:

- `FAST_SCORE` / `DN_PRECOMP` / `ADD_AT`: **leave them on**. All three are bit-exact (pinned by tests
  U14/U17), add no resident memory (`dn` is N-level, ~4 MB at 1M), and their first cost is `O(N)`
  (~4 ms); turning them off only makes queries slower.
- `W_CACHE`: **trade memory for latency**. Resident `nnz x 4 B` (~165 MB at 1M; ~1.465 GB at 8.84M by
  extrapolation); the first hit builds it in `O(nnz)` (~0.5-0.6 s at 1M); changing `k1`/`b`/`avgdl`, or
  any write/tombstone that changes the block structure, **changes the key and invalidates it**, falling
  back to on-the-fly scoring. Suits read-only corpora, spare resident memory, and many queries per
  process; with few queries it is a net loss (about 200 queries to pay back, estimated from the 500k
  readings). For long-term use, persist it at index time (`save(..., w_cache=True)`, CLI `--w-cache`)
  and let load reuse it after checking the content key `_w_persist_key()` (measured at 1M: first query
  238.3 ms down to 3.1 ms).
- `TC_SHARED`: **trade deployment shape for resident memory**. It only pays off with "one index plus
  several long-lived processes"; **do not enable it for a single process** (you only pay an extra
  publish). A write invalidates `_tc_key()` (`path|mtime|shape|nnz`, i.e. the **whole index**), so other
  processes must republish; **cross-OS semantics are untested**; report memory in **both calibers**
  (`private` drops while `RSS` rises, since shared pages count towards RSS). If it is unavailable it
  falls back to a private CSC with identical scores.
- `WRITE_DICT` / `MATVEC`: **not recommended** (the former shows no gain, the latter is 27% slower).

## 4. Module-level tools (`primolix.kernel`)

`tokenizer_status()` (is the tokeniser degraded?) · `zh_tokenize` / `simple_tokenize` ·
`auto_workers()` (memory-aware parallelism) · `parallel_tokenize` / `parallel_tokenize_stream` · `log` ·
`full_hash` · `ViewError` · `PreallocCSR` (capacity-preallocated CSR).

`primolix.core` also exports tunable constants: `MAX_CHUNK` (1200) · `CODE_BLOCK_LINES` (80) ·
`MAX_FILE_BYTES` (2 MB) · `EXCLUDE_DIRS` / `EXCLUDE_EXT` / `CODE_EXT` / `PROSE_EXT` (scan/chunk rules).

## 5. Contracts and pitfalls (measured)

| # | Contract / pitfall |
|---|---|
| 1 | **`update()`'s return value is a contract**: `{'kind': 'noop'\|'incremental'\|'rebuild', 'detail': str}`. **An out-of-vocabulary term forces `kind='rebuild'`** (a full fallback rebuild, **never a silent drop**); in-vocabulary edits give `'incremental'` (`detail` looks like `tomb=1 append=1`). Judge "which path ran" from `kind`, not from "no exception was raised". |
| 2 | **Argument shape differs across layers**: `SparseBM25.score_all(qtoks)` takes a **token list**; `Primolix.query(text)` takes **text and tokenises for you**. Passing a string to the kernel treats it as a **set of characters** and **silently returns all zeros**. |
| 3 | **The vocabulary is closed**: new terms cannot be added incrementally, so by default they trigger a **structural rebuild** (`O(V+nnz)`, roughly 4-6 s locally); pre-check with `oov_terms()`. Another route is **wired into the kernel (minimal tier, all three stages are API)**: `mount_segment(...)` catches new terms in a **read-only segment** (primitives in the segment, scoring with the **merged `df`**, `segment_report()` gives the vocabulary generation) -> **bit-identical to a full rebuild** (`max\|delta\| = 0` measured), **without touching the main `T`** and without writing to disk; `fold_segment(tag)` merges the segment into the main table (one `O(nnz)` pass; scores unchanged across the fold); `unmount_segment(tag)` **withdraws** a mounted segment (scores bit-exactly back to the pre-mount state; requires **last-in, first-out** and only while it has not been folded). **An atomic segment swap under concurrency is now handled in the kernel by default** (see "fixed in the current version" below); what is **still missing** is **cross-process segment sharing and publishing** (`tc_shared` shares the main table only; publishing segments across processes is not wired up yet). Measured before that fix, it was **not atomic**: with no lock about **3%** (2.3%-3.7% across repeated measurements) of concurrent reads are torn, **96% of them silently wrong** (no exception, a score that is neither the pre- nor the post-mount snapshot); **external serialisation gives 0 torn**, and **the reader-side lazy block build is not a problem**. Fixed in the current version: the **kernel locks by default** (`SWAP_LOCK=True`, a reentrant reader/writer lock -- concurrent readers, exclusive writer; **0 torn** measured, about **+3.4 us/query**, switchable off for single-threaded use), and the facade (section 8) adds a coarser lock so composite sequences are atomic too. The cost is query time proportional to segment count (sublinear: about 5x at 30 segments). Provenance: `research/seg_atomic_probe.py` (this probe is not shipped with this release). |
| 4 | **Library output goes to stderr**; `import primolix.core as core; core.VERBOSE = False` silences it completely (the CLI is unaffected). |
| 5 | **`FAST_SCORE` has been `True` by default since 2026-10-03**; turning it off also disables `DN_PRECOMP` / `ADD_AT` / `W_CACHE`, and queries get slower. |
| 6 | **`PreallocCSR` only supports row growth**: after replacing `T` by hand (column width changed) you must clear the block cache (`_tc_blocks=[]; _tc_rows=0`) so it is rebuilt. |
| 7 | **The vocabulary carrier is a read-only mmap** (`VocabMmap`): `term2idx[t] = i` raises `TypeError`, so adding terms requires re-publishing the carrier or an overlay. |
| 8 | **Do not delete files inside an index by hand** (necessity table in `index_format.en.md`); to clean up, rebuild the whole directory. |
| 9 | **Two id spaces** (when you bring your own corpus): a "vocabulary" file such as `vocab.json` is **not necessarily** the index's column space — **map by string through `term2idx`** and never treat external numbers as column ids. |
| 10 | **Both the materialised layer and the shared CSC have invalidation rules.** `W_CACHE`'s key includes parameters and block structure (a parameter change, a write or a tombstone invalidates it); **`.mm/` has a generation key `mm_key.txt`** (the npz's name/mtime/size; a mismatch refuses the whole `.mm/` and falls back to the `T_data` inside the npz); `TC_SHARED`'s key covers **the main table only** (`path\|mtime\|shape\|nnz`), so only **re-saving the index, a format change, or a change of the main table's shape/nnz** stops it matching -- **mounting or unmounting a segment does not** (the shared CSC is built from the main `T`, and segment mount/unmount never touch `T`). Note that `_seg_key()` exists for a future "segments take part in sharing" design but **has no call site today** (do not assume it is active). Neither is "set once and forget" — decide up front who pays the rebuild. |
| 11 | **Resident memory has two calibers**: `private` and `RSS` (which counts shared pages). `mmap` postings and the shared CSC make the two point in **opposite** directions — reporting either alone misleads. Say which caliber you mean; the table in the README is the **private** caliber (excluding shared pages). |

## 6. Common tasks (shortest path)

```python
# build / query
z = Primolix.build("docs/", "docs/.idx")
for h in z.query("sparse retrieval", k=5):
    print(h["rank"], round(h["score"], 3), h["file"], h["line"])

# incremental (read `kind` to decide whether a rebuild was needed)
rep = z.update("docs/")
print(rep)                       # {'kind': 'incremental', 'detail': 'tomb=1 append=1'}

# change the formula (kernel level, no rebuild)
bm = SparseBM25.load("docs/.idx/bm25.npz")
bm.k1, bm.b = 2.0, 0.3

# attribution (decompose a score per term; public attributes + the idf formula from the docs)
# see explain() in examples/run_examples.py

# views (multi-version / A-B)
z.attach_view("v2", vectors); print(z.view_report())
```

## 7. Hierarchical lineage index (`primolix.lineage`)

This release does **not** include this layer (`primolix.lineage` stays in the source tree; it has no regression tests and is not in CI, so it is not shipped).

## Appendix: staged publish

Batch online additions and publish them **once** per batch -- the intended mode when writes are frequent:

    st = idx.stage_begin("batch7")
    st.add("doc-001", "the primitive layer of sparse retrieval ...")
    st.add_many([("doc-002", "..."), ("doc-003", "...")])
    st.stats()                         # {pending, new_terms, base_V, base_T_rows}
    pub = st.publish()                 # one publish; returns {tag, rows, V0, V1, N, V}
    st.abort()                         # drop the batch (leaves nothing behind)

- Accumulation **does not modify** the index: new documents sit in a read-only segment, attached to the
  query path at publish time (bit-identical to a full rebuild).
- A publish is **one segment mount** under the write lock, so it is atomic for concurrent queries.
- `idx.compact()` folds segments into the main table (queries return to a single block);
  `idx.save(fold_first=True)` folds first and then persists. The segments and their sidecar directory
  `<index dir>/.segments/` can be deleted safely; the only cost is re-publishing those batches.
- Why not `idx.add()`: `add()` is **in place** and is refused while a segment is mounted (the appended
  rows would overlap the segment's logical rows). Use `stage_begin()`, or `compact()` first.
  **Pure deletions are unaffected.**
- Segments do not take part in the materialised layer (`W_CACHE`): queries inside a batch use the
  computed path, and the first query after a fold rebuilds the layer once. Shorter batches, less recompute.

## 8. One-line facade (`primolix.api`, **recommended entry point**)

> The kernel is convenient for research and verbose as a library (writing three segment `.npy` files by
> hand, aligning views to `meta` positions, switches as class attributes, reaching for `bm.k1` to change
> the formula). This layer **does not change the kernel**; it collapses the common actions into object
> methods and adds two defaults the kernel does not have.

```python
import primolix

idx = primolix.build("docs/", "docs/.idx")   # build (returns a facade object)
idx = primolix.open("docs/.idx")             # open an existing index (missing -> raises, never guesses)

idx.search("sparse retrieval", k=5)          # text in, hits out (tokenised for you)
idx.add("docs/")                             # incremental (out-of-vocabulary -> honest rebuild, kind='rebuild')
idx.tune(k1=2.0, b=0.3)                      # change the formula: no rebuild, material untouched
idx.explain("sparse retrieval")              # attribution: split the score per term
idx.stats()                                  # size / switches / segment count / view count
idx.compact()                                # compaction: fold mounted segments in -> single block again
idx.save(w_cache=True)                       # persist (optionally with the materialised layer)
```

| Member | Notes |
|---|---|
| `open(path)` · `open_index(...)` · `build(root, out, ...)` · `build_index(...)` | entry points; `build(..., w_cache=True)` also persists the materialised layer `w` at index time |
| **Concurrency-safe by default** | the facade holds a **reader/writer lock** (`RWLock`): `search`/`stats`/`explain`/`fuse` take the **shared** side, while `add`/`rebuild`/`tune`/the three segment stages/`compact`/view writes/`save` take the **exclusive** side -> readers run concurrently, a writer excludes readers -> **sharing one object across threads cannot return wrong scores** (basis: with no lock about **3%** (2.3%-3.7% across repeated measurements) of concurrent reads are torn, **96% of them silently wrong**) |
| Segments | `mount(seg_dir, tag, new_terms, dl, rows=None, generation=None)` · `unmount(tag)` · `fold(tag)` · `segments()` |
| Views | `views()` · `attach_view(name, vectors, positions=None, ...)` · `detach_view(name)` · `fuse(q, qvec, name, alpha=0.35)` · `save_view` / `load_views` |
| Parameters and switches | `tune(k1=, b=, persist=)` · `stats()` (with current switch values) · `set_switches(**kw)` (class attributes, so **process-wide**) |
| Persistence | `save(path=None, w_cache=False)`: by default it writes back into the index directory; with `path` it saves just the kernel index elsewhere |

**Boundaries, stated plainly**:

1. The facade **does not hide the kernel's boundaries**: the view family and segments keep their own
   "not in CI / verify it yourself" notes. The kernel now carries **its own** concurrency lock
   (`SWAP_LOCK`, see section 4); what the facade adds is atomicity for **composite sequences**
   (`add()` = tombstone + append + persist, `compact()` = fold + clear + rebuild), so calling the kernel
   directly still cannot return wrong scores, but multi-step combinations need your own outer lock.
2. **Switches are class attributes, so they are process-wide**: `set_switches(W_CACHE=True)` affects
   **every** index in this process, and is **not persisted**.
3. Facade writes **wait for all readers to leave** (reader/writer semantics), so heavy swapping on one
   object makes readers and the writer wait for each other; that is the price of correctness. The policy
   is **writer preference** (readers yield: they may be delayed and may lag, but **never read a torn
   state**). Making the **writer stop waiting for readers** needs read-side snapshots or an atomic
   segment-list pointer swap (see section 3; the related probe `research/seg_atomic_probe.py` is not
   shipped with this release).
