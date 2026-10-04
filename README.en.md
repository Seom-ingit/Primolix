# primolix

A sparse retrieval kernel: the index stores only invertible primitives (`tf`, `df`, `dl` plus a
tombstone bitmap), and scores are computed at query time.

*Chinese docs: [README.md](README.md)*

## Install

```bash
pip install -e .          # for now: from a clone (use the next line once published)
pip install primolix      # after the PyPI release
```

Python >= 3.9. Dependencies: `numpy`, `scipy` (**a missing scipy raises, it does not degrade**).

**Chinese tokenisation uses `jieba`**: without it the tokenizer degrades to "character + adjacent
bigram", still usable but less precise -- the degradation is visible through
`primolix.kernel.tokenizer_status()`. Optional: `pip install primolix[dense]` (dense reranking,
falls back to BM25-only), `psutil` (memory-aware worker count for parallel tokenising), and `rich`
(coloured CLI output; without it the CLI prints plain text).

## Ten-second demo

```bash
python -m primolix index  examples/corpus/ --out examples/corpus/.idx   # build
python -m primolix query  "sparse kernel" --out examples/corpus/.idx --k 3
python -m primolix update examples/corpus/ --out examples/corpus/.idx   # incremental (one edit does that one edit's work)
```

Real output (this machine, over the ten documents in `examples/corpus/`):

```
$ python -m primolix index examples/corpus/ --out examples/corpus/.idx
✔ 建索引完成

$ python -m primolix query "真增量 原语" --out examples/corpus/.idx --k 3
───────────────────────────  top-3 · BM25 · 569ms  ────────────────────────────
  # 1 3.653 01_incremental.md:1
     真增量写入：改一篇文档只做那一篇的工作，不重建整库。墓碑标记旧行，新内容追加到尾部。
  # 2 1.962 02_primitive.md:1
     原语层：索引里只存 tf、df、dl 与墓碑。分数（BM25 权重）是这些原语的函数，查询时算出来。

# edit a document, reusing existing terms -> incremental path
$ python -m primolix update examples/corpus/ --out examples/corpus/.idx
✔ 真增量更新完成 (tomb=1 append=0)

# edit a document, introducing new terms -> documented fallback to a rebuild (never a silent drop)
$ python -m primolix update examples/corpus/ --out examples/corpus/.idx
✔ 整库回退重建完成 (原因: oov=6)
```

The difference between the two `update` runs is exactly "in-vocabulary / out-of-vocabulary" -- see the
contract table in [docs/api.en.md](docs/api.en.md). (CLI output is Chinese; the API is the same either way.)

## Should you use it

The trade-offs and the honest boundaries are in **[docs/when_to_use.en.md](docs/when_to_use.en.md)**
(English) or [docs/when_to_use.md](docs/when_to_use.md) (Chinese), including when *not* to use it.
One line: **frequent edits, repeated tuning, explainability and auditability** fit well; **read-only,
query-dense, hard latency budgets** favour a precomputed scheme.
**API reference:** [docs/api.en.md](docs/api.en.md) (English) / [docs/api.md](docs/api.md) (Chinese) ·
**index format and compatibility:** [docs/index_format.en.md](docs/index_format.en.md) (English) /
[docs/index_format.md](docs/index_format.md) (Chinese).

On distributed deployment, stated precisely:

- **Across machines**: this package has **no** shard routing and **no** coordination layer. That is
  **"not implemented", not a mechanism weakness**.
- **Within one machine, the package already ships these** (each usable on its own):
  - **segments as pieces** (a **segment** is a read-only block mounted beside the main table; the main
    table is never touched): `mount_segment` / `fold_segment` / `unmount_segment` attach read-only
    segments to one index (including withdrawal), bit-identical to a full rebuild (tests U10-U12);
  - **cross-process shared CSC**: `TC_SHARED` lets several processes share one blocked CSC (off by
    default; see the "Performance switches" section below for when to turn it on);
  - **two key pieces for shard-parallel builds**: `SparseBM25(..., vocab=...)` locks the column space
    (**out-of-vocabulary terms are refused, never silently added**) and `stream="ids"` tokenises once
    and then builds from token ids. The **orchestration** of per-shard appends lives in the probes
    under `research/` in the source research tree (`shard_stream_build_probe.py` / `shard_sim_probe.py`;
    neither probe is shipped with this release) and **is not a package API yet**.
- Low private memory **is** an advantage (a node holds more shards or replicas; segments are read-only
  and trivially replicable), but **"low memory => higher per-core throughput" does not follow**: queries
  recompute scores on the CPU, so low memory buys *deployment density*, not per-core throughput.
- And the real difficulty in sharding -- the global `idf` -- is exactly what storing `df` as a primitive
  solves exactly: aggregate the global `df`, get the exact idf, **without rebuilding each shard**.
- The hierarchical lineage index (`primolix.lineage`) is not shipped with this release (it has no
  regression tests yet).

## Name

`primolix` = `primo` / `primitive` (the primitives) + `LIX`.

`LIX` comes from the previous generation, `ZELIX` — *Zero-Encoding Lazy Indexing & eXecution*.
The two generations relate like this:

- **ZELIX (previous generation, sibling project)**: zero-encoding index plus lazy encoding of the
  candidate pool at query time; its claim is DPR-class quality at BM25-class indexing cost on 8.84M
  passages.
- **primolix (this generation)**: the primitives layer extracted from the ZELIX kernel, focused on
  the index lifecycle — incremental writes, formula changes without rebuilds, attribution,
  reconciliation, multiple views.

`ZELIX` is the previous generation's internal codename: this package names no module, class or export
after it. It is mentioned above only to explain where `LIX` comes from.

The name only promises what this package actually does: primitives, and lazy indexing and
execution. It is **not** a multimodal system — it only handles text tokens. It is unrelated to
Zelix Pty Ltd (the Java obfuscator KlassMaster); the two only share the three letters `LIX`.

## What this project is

`primolix` is a BM25 sparse retrieval kernel written in Python, running on CPU only.

What sets it apart from most retrieval libraries is how it treats scores. The common approach
persists the computed weighted scores as part of the index, which means: editing one document
requires rebuilding the whole index, changing one parameter requires a rebuild, asking "why did
this document rank first" yields only a single total, and confirming "is it still correct after the
edit" requires a rebuild plus a diff.

`primolix` stores the primitives (`tf`, `df`, `dl`) and treats scores as a function of them. The
same index can therefore:

- edit documents in place, doing work proportional to the edit only;
- change `k1` / `b` / `idf` without rebuilding and without touching the material;
- decompose one document's score per term, and answer "if I change the parameters, what would this
  document score";
- verify a change by reconciling primitives only, at a cost proportional to the change.

The default path computes scores at query time: **0.049 ms/query** on the small 44,757-chunk index
(see the numbers below). **With `W_CACHE` enabled and a read-only index, queries can overtake a
precomputed scheme; otherwise they are slower.** Since 2026-10-03 the default path carries two
optional layers, both **bit-exact**:

**Query performance (on by default)**: at 500k chunks, same batch of 100 queries, same-batch readings --
old default **5.38 ms/query** -> **new default 2.70 ms** (1.99x) -> with the optional **materialised
layer `W_CACHE`** **1.16 ms** (4.65x); same-batch reference (precomputed weights) **1.39 ms**.
All three are **`max|delta| = 0`** against the old path. The three switches `DN_PRECOMP`, `ADD_AT`,
`W_CACHE` sit behind the `FAST_SCORE` master switch; see [docs/api.en.md](docs/api.en.md).

Boundaries, stated plainly: the materialised layer lives **in memory by default** (`nnz x 4 B`, 165 MB at
1M chunks); to keep it long term, write it to disk at index time (`--w-cache`) and let load reuse it after
checking the content key. It **depends on being hit** (changing `k1`/`b`/`avgdl` or adding tombstones
invalidates it, and it then **falls back to on-the-fly scoring**); **cross-batch absolute values on this
machine drift 20% to 40% (up to 2x in individual batches)**, so the multipliers above hold **within a
batch only**.

Good fit: documents that change often, repeated parameter tuning, explaining ranked results, and
auditing retrieval. Poor fit: read-only corpora that never change **with the materialised layer off**,
and pushing single-query latency to the limit.

## Features

1. **True incremental writes (in-vocabulary).** Editing 100 documents does the work of those 100,
   and does not touch the other 44,657.
   Warning, the boundary: if an edit introduces **terms outside the vocabulary**, the default is a
   **structural rebuild** (`O(V+nnz)`, roughly 4-6 s locally) and `update()` reports
   `kind='rebuild'` (**it never silently drops terms**). A second route is **now wired into the
   kernel (minimal tier; all three stages are API)**: `SparseBM25.mount_segment(...)` catches new
   terms in a **read-only segment** (primitives in the segment, scoring with the **merged `df`**,
   vocabulary generation reported) — **bit-identical to a full rebuild** (`max|delta| = 0` measured),
   **without touching the main `T`** and without writing to disk; `fold_segment(tag)` then merges it
   into the main table (scores unchanged across the fold), and `unmount_segment(tag)` withdraws it
   (**last-in, first-out**, and only while it has not been folded). Concurrency note: concurrent segment
   swaps are **measured to be non-atomic** -- with no lock about **3%** (2.3%-3.7% across repeated
   measurements) of concurrent reads are torn and **96% of those are silently wrong** -- so the
   **kernel locks by default** (`SWAP_LOCK=True`, a
   reentrant reader/writer lock: concurrent readers, exclusive writer, **0 torn** measured, costing about
   **+3.4 us/query** and switchable off for single-threaded use), and the facade adds a coarser lock so
   that composite sequences (tombstone + append + persist) are atomic too. See the contract table and
   section 8 of [docs/api.en.md](docs/api.en.md).
2. **Formula changes without rebuilds.** Changing `k1` / `b` is two attribute assignments; changing
   `idf` recomputes a vector of length `V`. The material is not touched.
3. **Rollback and multiple views.** A view is the tuple `(segment list, tombstones, parameters,
   vocabulary generation)`. Switching views swaps the list only, so several views can be open at
   once (version comparison, A/B tuning). Boundary: the view API ships with the package (see the
   view family in [docs/api.en.md](docs/api.en.md)), but **the released CI does not cover the view
   path** -- end-to-end and kernel unit tests cover build, query, incremental write, formula change,
   delete, reload, and segment mount/fold/unmount (U10-U12).
4. **Attribution.** Any (query, document) score decomposes per term into
   `idf x saturation(tf) x document length`, and the score under different parameters can be
   computed on the spot.
5. **Incremental reconciliation.** After a change, checking the primitives is enough to confirm the
   result; a full rebuild is not required.
6. **Compaction.** Append-only growth increases the number of index blocks and slows queries down;
   compaction merges small blocks back into one large block and restores query speed.

## Numbers

Environment: one Windows machine, one index (44,757 documents, vocabulary 73,849, nnz 2,618,712,
26.9 MB on disk), 30 queries. Each number was measured in separate processes over several rounds,
taking the steady-state median.

| Item | Value | Conditions |
|---|---|---|
| Incremental write | 0.22 us / token | batches of 100, steady state |
| Change `k1` / `b` | 1.2 us | two attribute assignments |
| Change `idf` | 5.3 ms | recompute a length-`V` vector |
| Full rebuild (reference) | 5.7 s | same corpus |
| Switch view | 10.9 ms | 7 segments, excluding first load |
| Attribution counterfactual | 0.9 ms + 1.8 ms | recompute idf + 30 queries |
| Incremental reconciliation | 8.7 ms | 100 edited + 100 appended |
| Query | 0.049 ms / query | fast path **on by default since 2026-10-03**; 0.132 when off |
| Resident memory (private) | 2.80 MB | excluding shared pages |
| Cold start | 36 ms | loading the index |
| Compaction | 21 ms | 17 blocks into 1; query 0.217 to 0.065 ms/query |

Correctness (each item is a reconciliation against a full rebuild or a from-scratch build):

- scores after incremental writes match a rebuild, bit for bit;
- scores after changing `k1` / `b` or `idf` match a build with those parameters, bit for bit;
- the per-term contributions sum to the scoring function's output, bit for bit;
- corrupting one `df` entry is detected during reconciliation.

Caliber: comparisons inside one batch are trustworthy; absolute values drift 20% to 40% between
batches (up to 2x in individual batches), so the table above is a local, single-batch reading, not a
cross-environment guarantee. Without `W_CACHE`, queries are slower than a precomputed scheme (with
`W_CACHE` on and a read-only index they can overtake it), and resident memory looks larger when shared
pages are counted — both are visible in the data and are not argued away.

## Performance switches: three on by default, two left to you

The default path already carries every "free" optimisation, and they are **bit-exact** — you do not
have to do anything for them:

| Switch | Default | What it does | Cost |
|---|---|---|---|
| `FAST_SCORE` | on | query fast path (master switch) | none (bit-exact; off is simply slower) |
| `DN_PRECOMP` | on | precompute the per-document normalisation term | 4 MB at 1M (N-level), plus ~4 ms once |
| `ADD_AT` | on | unbuffered accumulation | none (bit-exact, zero memory) |
| `SWAP_LOCK` | on | **concurrency safety**: swaps and queries exclude each other (concurrent readers, exclusive writer) | about **+3.4 us/query** (17.6% on a tiny index, ~7% at 44.7k chunks) |

The two below are **off by default** — not an oversight: each trades a different axis for latency, so
the choice belongs to your workload.

**`W_CACHE` (materialised layer) — trade memory for query latency.** On a hit, each posting costs one add.
- Worth enabling when the corpus is essentially read-only, parameters are settled, resident memory is
  spare, and each process runs many queries (about **200 queries** to pay back, estimated from the 500k
  readings; that figure is an estimate, not a direct measurement).
- Cost: resident `nnz x 4 B` (~**165 MB** at 1M; ~**1.465 GB** at 8.84M by extrapolation); the first hit
  builds it (~**0.5-0.6 s** at 1M); changing `k1`/`b`/`avgdl`, or any write or tombstone that changes the
  block structure, **invalidates it automatically** and scoring falls back to on-the-fly.
- For long-term use, persist it at index time: `--w-cache` (`save(..., w_cache=True)` in Python), then
  load and use it directly — measured at 1M, the first query goes from **238.3 ms to 3.1 ms**, at the
  price of `nnz x 4 B` on disk. Re-saving in place hits the Windows mmap lock, so write to a new directory.
- Leaving it off is never wrong: a miss falls back to on-the-fly scoring.

**`TC_SHARED` (sharing that CSC across processes) — trade deployment shape for resident memory.**
- Worth enabling when **one index is shared by two or more long-lived processes** (worker pool, multi-tenant).
- Do not enable it for a single process: you only pay an extra publish.
- Prerequisites and cost: **a write invalidates the whole shared key** (the key is `path|mtime|shape|nnz`),
  so other processes must republish; **cross-OS semantics are untested**; report memory in **both calibers**
  (`private` drops noticeably while `RSS` rises, because shared pages count towards RSS).
- If it cannot be published it **falls back safely** to a private CSC: slower, never wrong.

**Not recommended**: `WRITE_DICT` (measured no gain), `MATVEC` (measured 27% slower).

For finer switch semantics, invalidation rules and payback calibers see the "How to choose a switch"
block and section 5 of **[docs/api.en.md](docs/api.en.md)**.

## Usage

To see it run first: **`python examples/run_examples.py`** (ships 10 corpus files and
demonstrates build, query, incremental write, formula change and attribution; repeatable).

**Facade (recommended entry point)**: one line per action, and **concurrency-safe by default**
(concurrent readers, exclusive writer):

```python
import primolix

idx = primolix.build("my_docs", "my_docs/.idx")   # build (returns a facade object)
idx = primolix.open("my_docs/.idx")               # open an existing index (missing -> raises)
idx.search("sparse kernel", k=10)                 # text in, hits out (tokenised for you)
idx.add("my_docs")                                # incremental (OOV -> honest rebuild)
idx.tune(k1=2.0, b=0.3)                           # change the formula: no rebuild
idx.explain("sparse kernel")                      # attribution: split the score per term
idx.stats(); idx.segments(); idx.views(); idx.compact()   # overview / segments / views / compaction
```

Command line (`python -m primolix --help` lists everything):

```bash
python -m primolix info    --out my_docs/.idx                 # overview: size / switches / segments / views
python -m primolix params  --out my_docs/.idx --k1 2.0        # change the formula (--save persists)
python -m primolix config  --set W_CACHE=true                 # switches (process-wide)
python -m primolix explain "sparse kernel" --out my_docs/.idx  # attribution per term
python -m primolix segment list --out my_docs/.idx            # segments: list / mount / unmount / fold
python -m primolix view    list --out my_docs/.idx            # views: list / attach / detach / fuse
python -m primolix compact --out my_docs/.idx                 # compaction: fold mounted segments in
```

Command line:

```bash
python -m primolix index  my_docs/  --out my_docs/.idx   # build
python -m primolix query  "sparse kernel"  --out my_docs/.idx --k 10
python -m primolix update my_docs/  --out my_docs/.idx   # true incremental: only changed files
python -m primolix browse --out my_docs/.idx --files
```

Python:

```python
from primolix import Primolix

z = Primolix.build("my_docs", "my_docs/.idx")
z = Primolix("my_docs/.idx")
scores = z.query("sparse kernel", k=10)
```

Using the kernel directly:

```python
from primolix.kernel import SparseBM25

bm = SparseBM25(texts, lambda s: s.split(), k1=1.5, b=0.75)
bm.update_docs([3, 7], [["new", "content"], ["another", "document"]])   # in place
bm.k1, bm.b = 2.0, 0.3                                                  # no rebuild
```

About output: library progress and diagnostics go to **stderr**, so the caller's stdout stays
clean. Set `primolix.core.VERBOSE = False` to silence them completely. The command line entry point
is unaffected and keeps printing results to stdout.

## Tests

```bash
python tests/test_primolix_e2e.py     # end to end: build, query, update, delete, reload
python tests/test_kernel_unit.py      # kernel invariants: roundtrip, tombstones, append, formulas,
                                      # segment stages, and bit-exact switches
python tests/test_staged_publish.py   # staged publish: mount / fold / unmount, bit-exact rollback
                                      # (T1-T7)
```

All three print `PASS` / `FAIL` per check and exit non-zero on failure. No external data is needed.
Coverage: kernel **U1-U22** (including segment mount / fold / unmount, U10-U12; U20 pins "load then
update again" and "a `.mm/` generation-key mismatch is refused"), plus 10 end-to-end checks and
7 staged-publish checks (T1-T7) at the index layer. **Not covered**: the view family.

## Layout

```
primolix/    package: CLI, index object, BM25 kernel, one-line facade, vocabulary carrier,
             cross-process sharing
             (cli.py, core.py, kernel.py, api.py, tc_shared.py, vocab_mmap.py)
tests/       end-to-end and unit tests (English output)
examples/    runnable examples: run_examples.py (kernel and index object)
docs/        API and index-format docs, when-to-use guidance, and research records
research/    probes and readings: an index of which probe produced each number above (including
             shard-parallel build and multi-process probes). The probe scripts and their readings
             are not shipped with this release.
```

## License

MIT, see LICENSE. Contributing: [CONTRIBUTING.md](CONTRIBUTING.md) (Chinese). Versioning and
compatibility promises: [CHANGELOG.md](CHANGELOG.md).
