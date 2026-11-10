#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""L2 增量持久化（delta 日志）回归。无外部数据。

Run:  python tests/test_delta_persist.py

判据（对应 05_L2 的 L2-1 / L2-2 / L2-3 / L2-4）：
  D1 npz 模式（默认）：一次增量后 npz **被重写**、`.delta/` 不存在
  D2 delta 模式：一次增量后 npz **一个字节不动**、`.delta/` 多一个文件（L2-2 的形态）
  D3 回放逐位：重载后的分数与内存逐位相同、`_meta` 条数相同（L2-1）
  D4 两个 delta 链式回放（L2-1）
  D5 未提交的 `.tmp` 不被回放（L2-3 的第一态）
  D6 全量落盘会清空 `.delta/`，且重载仍逐位相同（L2-4 的压实语义）
  D7 delta 模式下折叠段也写成 delta（链不断、重载逐位）
  D8 自动压实：delta 超阈值时自动全量落盘并清空（S3）
  D9 崩在「delta 已提交、meta 未写」之间：文档仍可见（身份在 delta 里）
  D10 该文档之后被删，重载不许从 delta 的 meta 里复活
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import primolix                                    # noqa: E402
from primolix import core, Primolix                # noqa: E402

FAILS: list[str] = []
Q = ["kernel", "sparse index", "bm25 postings"]


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  |  {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


def scores(z) -> list[np.ndarray]:
    return [np.asarray(z._bm.score_all(primolix.zh_tokenize(q)), dtype=np.float32) for q in Q]


def maxdiff(a, b) -> float:
    return max((float(np.abs(x - y).max()) if x.size else 0.0) for x, y in zip(a, b))


def main() -> int:
    work = HERE / "_delta_run"
    shutil.rmtree(work, ignore_errors=True)
    root = work / "docs"
    root.mkdir(parents=True)
    (root / "a.txt").write_text("kernel incremental retrieval engine\n" * 3, encoding="utf-8")
    (root / "b.txt").write_text("bm25 sparse index and postings\n" * 3, encoding="utf-8")
    idx = work / "idx"
    z = Primolix.build(str(root), str(idx))
    npz = idx / "bm25.npz"

    try:
        # ---- D1：npz 模式（默认）----
        (root / "c.txt").write_text("kernel retrieval kernel retrieval\n" * 3, encoding="utf-8")
        m0 = (npz.stat().st_size, npz.stat().st_mtime_ns)
        rep1 = z.update(str(root))
        m1 = (npz.stat().st_size, npz.stat().st_mtime_ns)
        check("D1 npz mode rewrites the npz and leaves no .delta/",
              rep1.get("kind") == "incremental" and m0 != m1
              and not (idx / core.DELTA_SUBDIR).exists(),
              f"kind={rep1.get('kind')!r} · npz 变={m0 != m1} · delta 目录="
              f"{(idx / core.DELTA_SUBDIR).exists()}")

        # ---- 切到 delta 模式 ----
        core.set_persist_mode("delta")
        (root / "d.txt").write_text("kernel index kernel index\n" * 3, encoding="utf-8")
        m2 = (npz.stat().st_size, npz.stat().st_mtime_ns)
        rep2 = z.update(str(root))
        m3 = (npz.stat().st_size, npz.stat().st_mtime_ns)
        dfiles = core.delta_files(idx)
        check("D2 delta mode leaves npz untouched and writes one delta",
              rep2.get("kind") == "incremental" and m2 == m3 and len(dfiles) == 1
              and "delta#" in str(rep2.get("detail")),
              f"kind={rep2.get('kind')!r} · npz 未变={m2 == m3} · deltas={len(dfiles)}"
              f" · detail={rep2.get('detail')!r}")

        # ---- D3：回放逐位 ----
        mem = scores(z)
        z2 = Primolix(str(idx)); z2._load()
        rel = scores(z2)
        d3 = maxdiff(mem, rel)
        check("D3 replay is bit-exact (scores) and meta count matches",
              d3 == 0.0 and len(z2._meta) == len(z._meta),
              f"max|d|={d3:.3e} · meta {len(z._meta)} → {len(z2._meta)}")

        # ---- D4：两个 delta 链式（临时抬高压实阈值，否则小索引会当场压实）----
        _old_r = core.DELTA_COMPACT_RATIO
        core.DELTA_COMPACT_RATIO = 1e9
        (root / "e.txt").write_text("postings postings bm25\n" * 3, encoding="utf-8")
        rep4 = z.update(str(root))
        core.DELTA_COMPACT_RATIO = _old_r
        dfiles = core.delta_files(idx)
        mem4 = scores(z)
        z3 = Primolix(str(idx)); z3._load()
        d4 = maxdiff(mem4, scores(z3))
        check("D4 two deltas replay in order (bit-exact)",
              len(dfiles) == 2 and d4 == 0.0 and len(z3._meta) == len(z._meta),
              f"deltas={len(dfiles)} · max|d|={d4:.3e} · meta={len(z3._meta)}")

        # ---- D5：未提交的 .tmp 不被回放 ----
        tmp = idx / core.DELTA_SUBDIR / "000999.npz.tmp"
        tmp.write_bytes(b"not a real delta")
        z4 = Primolix(str(idx)); z4._load()
        d5 = maxdiff(mem4, scores(z4))
        check("D5 an uncommitted .tmp is ignored (rename is the commit point)",
              d5 == 0.0 and len(z4._meta) == len(z._meta),
              f"max|d|={d5:.3e} · delta 文件数={len(core.delta_files(idx))}")

        # ---- D6：全量落盘清空 delta ----
        z._persist()
        left = len(core.delta_files(idx))
        z5 = Primolix(str(idx)); z5._load()
        d6 = maxdiff(scores(z), scores(z5))
        check("D6 a full save clears .delta/ and stays bit-exact",
              left == 0 and d6 == 0.0,
              f"剩余 delta={left} · max|d|={d6:.3e}")
        # ---- D7：delta 模式下的折叠 —— 折出来的行写成一条 delta（临时抬高压实阈值）----
        core.set_persist_mode("delta")
        _old_r7 = core.DELTA_COMPACT_RATIO
        core.DELTA_COMPACT_RATIO = 1e9
        (root / "f.txt").write_text("zznovelword 只有这里出现\n" * 3, encoding="utf-8")
        rep7 = z.update(str(root))                 # 带新词 → staged（挂段）
        (root / "g.txt").write_text("kernel retrieval engine incremental\n" * 3, encoding="utf-8")
        rep8 = z.update(str(root))                 # 段在场 + 词表内 → fold(delta) + incremental
        core.DELTA_COMPACT_RATIO = _old_r7
        mem7 = scores(z)
        z7 = Primolix(str(idx))
        z7._load()
        d7 = maxdiff(mem7, scores(z7))
        check("D7 delta mode folds a mounted segment into a delta (chain intact)",
              rep7.get("kind") == "staged" and rep8.get("kind") == "incremental"
              and d7 == 0.0 and len(z7._meta) == len(z._meta)
              and len(core.delta_files(idx)) >= 1,
              f"stage={rep7.get('kind')!r} · fold+inc={rep8.get('kind')!r} · max|d|={d7:.3e}"
              f" · deltas={len(core.delta_files(idx))} · meta={len(z7._meta)}")

        # ---- D8：自动压实（把阈值压到 0 让它必触发）----
        old_ratio = core.DELTA_COMPACT_RATIO
        core.DELTA_COMPACT_RATIO = 0.0
        try:
            (root / "h.txt").write_text("kernel kernel kernel\n" * 3, encoding="utf-8")
            rep9 = z.update(str(root))
            left = len(core.delta_files(idx))
        finally:
            core.DELTA_COMPACT_RATIO = old_ratio
        z8 = Primolix(str(idx))
        z8._load()
        d8 = maxdiff(scores(z), scores(z8))
        check("D8 auto-compaction fires when deltas exceed the ratio",
              left == 0 and "+压实" in str(rep9.get("detail")) and d8 == 0.0,
              f"remaining={left} · detail={rep9.get('detail')!r} · max|d|={d8:.3e}")

        # ---- D9：崩在「delta 已提交、meta 未写」之间 ----
        core.set_persist_mode("delta")
        _old_r9 = core.DELTA_COMPACT_RATIO
        core.DELTA_COMPACT_RATIO = 1e9
        meta_bak = (idx / "meta.jsonl").read_bytes()
        (root / "i.txt").write_text("kernel index postings kernel\n" * 3, encoding="utf-8")
        rep9b = z.update(str(root))
        (idx / "meta.jsonl").write_bytes(meta_bak)      # 回滚 ⇒ 模拟「meta 还没写就崩了」
        z9 = Primolix(str(idx))
        z9._load()
        d9 = maxdiff(scores(z), scores(z9))
        check("D9 a committed delta survives a lost meta.jsonl (identity travels in the delta)",
              rep9b.get("kind") == "incremental" and len(z9._meta) == len(z._meta) and d9 == 0.0,
              f"kind={rep9b.get('kind')!r} · meta {len(z._meta)} → {len(z9._meta)}"
              f" · max|d|={d9:.3e}")

        # ---- D10：之后把它删掉，重载不许「复活」----
        (root / "i.txt").unlink()
        rep10 = z.update(str(root))
        core.DELTA_COMPACT_RATIO = _old_r9
        z10 = Primolix(str(idx))
        z10._load()
        d10 = maxdiff(scores(z), scores(z10))
        check("D10 a deleted doc is not resurrected from a delta's meta",
              len(z10._meta) == len(z._meta) and d10 == 0.0,
              f"tomb={rep10.get('detail')!r} · meta {len(z._meta)} → {len(z10._meta)}"
              f" · max|d|={d10:.3e}")
    finally:
        core.set_persist_mode("npz")
        shutil.rmtree(work, ignore_errors=True)

    print("-" * 60)
    print("ALL PASS" if not FAILS else "FAILURES: " + ", ".join(FAILS))
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
