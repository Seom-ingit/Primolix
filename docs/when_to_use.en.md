# When to use it, and when not to

> This page is only about trade-offs. The numbers are local, single-batch readings (absolute values
> drift 20%-40% between batches, up to 2x in individual batches); full context lives in the research
> records under `docs/`. Head-to-head comparisons
> with other implementations are in `docs/` too — this page makes **no multiplier claims**.
> Chinese version: [when_to_use.md](when_to_use.md)

## What kind of thing it is

`primolix` is a sparse retrieval kernel oriented towards the **index lifecycle**: writes, parameter
changes, attribution and reconciliation are cheap, and the price is that **every query recomputes
scores**. So it is the mirror image of "precompute the weighted scores and store them":

| | Precomputed weights | primolix |
|---|---|---|
| Query | fast (reads stored numbers) | on-the-fly by default (**0.049 ms/query** on 44,757 chunks; the fast path is now the default); with the optional **materialised layer** it overtakes: **1.16 ms vs 1.39 ms** for a precomputed scheme at 500k chunks (same batch) |
| Editing documents | expensive (usually a full rebuild) | **0.22 us/token**; editing 100 documents does the work of those 100 |
| Changing parameters | expensive (rebuild) | `k1`/`b` are **1.2 us**; changing idf is **5.3 ms** |
| Explaining a ranking | a single total | decomposable per term, bit-consistent |
| Verifying a change | rebuild, then diff | reconcile the primitives only: **8.7 ms**, proportional to the change |

## Good fit (any of these makes it worth considering)

- **Documents change often**: writes are truly incremental (tombstone + tail append); other documents
  are not touched.
- **Repeated tuning / A-B work**: changing `k1`/`b` touches no material and needs no rebuild; two views
  can be open at once.
- **You need to explain "why did this rank first"**: a score decomposes per term into
  `idf x saturation(tf) x document length`.
- **You need to audit / reconcile**: checking the primitives (`df`/`dl`/tombstones) confirms a change,
  at a cost proportional to the change.
- **You need rollback / multiple versions**: a view is "segment list + tombstones + parameters +
  vocabulary generation"; switching views swaps the list (~10.9 ms locally).
- **CPU only, light dependencies**: the core needs numpy + scipy, no GPU; resident private memory
  **2.80 MB** (small index).
- **Squeezing latency further on one index**: the default path already carries every "free"
  optimisation; whether to add the materialised layer `W_CACHE` (memory for latency) or cross-process
  sharing `TC_SHARED` (deployment shape for resident memory) is covered in the "Performance switches"
  section of the README and the "How to choose a switch" block in [`api.en.md`](api.en.md).
- **Append-only growth**: block count grows and queries slow down; compaction is **21 ms** and pays
  back after roughly 130 queries.

## Bad fit (any one of these: look elsewhere first)

- **Read-only, never edited, and with the materialised layer off**: there is no rebuild cost to save, so
  a precomputed scheme (or an off-the-shelf inverted index) gives you lower query latency. This is the
  least favourable case -- with `W_CACHE` on, the materialised layer overtakes it (**1.16 ms vs
  1.39 ms**, same batch).
- **Single-query latency is a hard budget** (e.g. a tight P99): every query recomputes, so it is slower
  than reading stored numbers.
- **Multimodal** (image / audio / video): this package handles text tokens only.
- **Distributed / multi-machine sharding, out of the box**: this package is single-machine,
  single-index — no shard routing, no coordination layer. That is **"not implemented", not a mechanism
  weakness**. Three separate things: (1) **low private memory is an advantage** (a node holds more
  shards/replicas; segments are read-only and trivially replicable); (2) **but "low memory => higher
  per-core throughput" does not follow** — queries recompute scores on the CPU, so low memory buys
  *deployment density*, not per-core throughput; (3) the real difficulty in sharding is the **global
  `idf`** (it depends on the global `df`) — and that is exactly what storing `df` as a primitive solves
  exactly: aggregate the global `df`, get the exact idf, **without rebuilding each shard**.
- **GPU-accelerated dense retrieval**: the core is CPU sparse; `dense` is only an optional candidate
  reranking extra.
- **New terms appear frequently**: the vocabulary is closed. Out-of-vocabulary terms are **not**
  silently dropped: `update()` falls back to a **full rebuild** and reports `kind='rebuild'` (in-vocabulary
  edits take the incremental path and report `kind='incremental'`). So this is not "unusable" — it is
  "**with new terms in every edit, you keep paying for a full rebuild**". If that is your normal case,
  fix the vocabulary upstream (fold new terms in at build time) or pick an open-vocabulary scheme.
  There is also a route that is **now wired into the kernel (minimal tier)**: `SparseBM25.mount_segment(...)`
  catches new terms in a **read-only segment** (segment carries primitives, scoring uses the **merged `df`**,
  `segment_report()` gives the vocabulary generation) — **bit-identical to a full rebuild** (measured
  `max|delta| = 0`) **without touching the main `T`** and without writing to disk; the cost is **query time
  proportional to segment count** (measured **sublinear**: about 5x at 30 segments). `fold_segment(tag)`
  merges a segment into the main table (scores unchanged across the fold), and `unmount_segment(tag)`
  withdraws a mounted segment (**last-in, first-out**, and only while it has not been folded).
  The only thing **not atomic in the kernel** is a segment swap under concurrency: measured, it is
  **not atomic** -- with no lock about **3%** (2.3%-3.7% across repeated measurements) of concurrent
  reads are torn, **96% of them silently
  wrong** (no exception, a score that is neither the pre- nor the post-mount snapshot). The **kernel
  now locks by default** (`SWAP_LOCK=True`, a reentrant reader/writer lock: concurrent readers,
  exclusive writer, **0 torn** measured, about **+3.4 us/query**), the facade adds a coarser lock for
  composite sequences, and single-threaded users can switch `SWAP_LOCK` off.
  (Provenance: `research/seg_atomic_probe.py` for the concurrency measurement and
  `research/fold_demo_probe.py` for the three stages; both probes are not shipped with this release.)
- **Scale far beyond what is verified**: the absolute numbers published here are at the 45k-document
  scale; the 500k and 1M readings are comparable **within one batch only**. The previous generation has
  8.84M-passage measurements — those are not this package's readings.

## Positioning (category level)

- **Versus precomputed-weight schemes**: choose primolix when **edits are frequent**; choose them when
  the corpus is **read-only and query-dense**.
- **Versus mature inverted-index libraries**: choose them for ecosystem, plugins, distribution and
  years of operational polish; choose primolix when you need **invertible primitives + true incremental
  writes + attribution**. The two can coexist.
- **Versus vector databases / dense retrieval**: complementary semantics, not substitutes; primolix can
  serve as their first-stage candidate generator (that is what the `dense` extra is for).

## Three questions decide it

1. **Will this corpus change?** No -> prefer a precomputed scheme. Yes -> continue.
2. **Do you want "edit and query immediately, change parameters without rebuilding, explain the
   ranking"?** No -> a precomputed scheme is cheaper. Yes -> primolix.
3. **Can your latency budget absorb query-time scoring?** No (tight P99) -> use primolix for writes and
   reconciliation, and pair it with a precomputed/vector index for online queries.

## Reproducing the numbers on this page

The three commands below are mechanism demos and correctness checks (they carry their own corpus and
need no external data). The measured readings on this page (torn-read rate, segment-count cost, rebuild
time) come from probes under `research/`, and those probes are not shipped with this release.

```bash
python examples/run_examples.py     # build / query / incremental / parameter change / attribution
python tests/test_primolix_e2e.py   # end-to-end correctness
python tests/test_kernel_unit.py    # kernel invariants
```
