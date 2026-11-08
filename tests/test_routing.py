#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Routing-table checks for primolix. Pure function, no storage, no index files.

Run:  python tests/test_routing.py

0.2.1 (04c A+C): `refuse` 出口退役 —— 段在场时不再抛"先 compact()"，而是 `fold_first`
（当场折回主表，再按 `then` 走）。折叠的代价由调用方从返回里如实读走。

Checks (each prints PASS/FAIL; process exits 1 if anything fails):
  R1  noop: nothing changed
  R2  noop wins over an OOV count
  R3  incremental: in-vocab, no segment
  R4  A: in-vocab while a segment is mounted -> fold_first(then=incremental)
  R5  fold flag is set exactly on the fold_first exit
  R6  staged: OOV, no segment
  R7  staged: OOV + segment, blocks under the threshold (no fold)
  R8  C: OOV + segment + blocks at the threshold -> fold_first(then=staged)
  R9  C boundary is inclusive at compact_blocks
  R10 rebuild when dense is on and no segment is mounted
  R11 dense + segment -> fold_first(then=rebuild)
  R12 OOV + segment rows to tombstone -> fold_first(then=staged)
  R13 in-vocab + segment rows to tombstone -> fold_first(then=incremental)
  R14 priority: tomb-in-segment outranks the block threshold
  R15 priority: in-vocab path outranks the block threshold
  R16 batch over the crossover -> rebuild (no segments)
  R17 batch exactly at the crossover -> staged (strict '>')
  R18 'then' is None on every non-fold route
  R19 negative counters raise ValueError
  R20 compact_blocks < 2 raises ValueError
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from primolix.routing import (  # noqa: E402
    DEFAULT_COMPACT_BLOCKS, FOLD_FIRST, INCREMENTAL, NOOP, REBUILD, STAGED,
    R_FOLD_BLOCKS, R_FOLD_DENSE, R_FOLD_INPLACE, R_FOLD_TOMB_SEGMENT,
    R_REBUILD_BATCH_TOO_BIG, R_REBUILD_DENSE, R_STAGED, route_update,
)

_fails: list[str] = []
_n: list[int] = [0]


def check(name: str, cond: bool, why: str = "") -> None:
    _n[0] += 1
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"  — {why}" if not cond and why else ""))
    if not cond:
        _fails.append(name)


def expect(name: str, kw: dict, route: str, code: str, then: str | None = "?") -> dict:
    r = route_update(**kw)
    ok = r["route"] == route and r["code"] == code
    if then != "?":
        ok = ok and r["then"] == then
    check(name, ok, f"got route={r['route']!r} then={r['then']!r} code={r['code']!r}"
                    f"（期望 {route!r}/{then!r}/{code!r}）")
    return r


def main() -> int:
    # R1 / R2
    expect("R1 noop", dict(), NOOP, "noop", None)
    expect("R2 noop wins over OOV count", dict(n_oov=7), NOOP, "noop", None)

    # R3–R5 —— 词表内 ＋ A
    expect("R3 incremental", dict(n_added=1, n_oov=0), INCREMENTAL, "incremental", None)
    r4 = expect("R4 A: in-vocab under a segment folds first",
                dict(n_added=1, n_segments=2, n_blocks=3),
                FOLD_FIRST, R_FOLD_INPLACE, INCREMENTAL)
    check("R5 folds flag set on fold_first", r4["folds"] is True)
    check("R5b folds flag clear elsewhere",
          route_update(n_added=1)["folds"] is False
          and route_update(n_added=1, n_oov=3)["folds"] is False)

    # R6–R9 —— staged 与 C
    expect("R6 staged", dict(n_added=3, n_oov=12, n_blocks=1), STAGED, R_STAGED, None)
    expect("R7 OOV + segment, blocks under threshold -> staged",
           dict(n_added=1, n_oov=1, n_segments=1, n_blocks=4, compact_blocks=8),
           STAGED, R_STAGED, None)
    expect("R8 C: blocks at threshold -> fold then stage",
           dict(n_added=1, n_oov=1, n_segments=2, n_blocks=8, compact_blocks=8),
           FOLD_FIRST, R_FOLD_BLOCKS, STAGED)
    expect("R9 threshold boundary inclusive (7 < 8 -> no fold)",
           dict(n_added=1, n_oov=1, n_segments=2, n_blocks=7, compact_blocks=8),
           STAGED, R_STAGED, None)

    # R10–R11 —— dense
    expect("R10 dense -> rebuild", dict(n_added=1, n_oov=5, dense_on=True),
           REBUILD, R_REBUILD_DENSE, None)
    expect("R11 dense + segment -> fold then rebuild",
           dict(n_added=1, n_oov=5, dense_on=True, n_segments=1, n_blocks=2),
           FOLD_FIRST, R_FOLD_DENSE, REBUILD)

    # R12–R15 —— 段内墓碑与优先级
    expect("R12 OOV + tomb in segment -> fold then stage",
           dict(n_added=1, n_oov=5, n_tomb_seg_rows=2, n_segments=1, n_blocks=2),
           FOLD_FIRST, R_FOLD_TOMB_SEGMENT, STAGED)
    expect("R13 in-vocab + tomb in segment -> fold then incremental",
           dict(n_removed=1, n_tomb_seg_rows=1, n_segments=1, n_blocks=2),
           FOLD_FIRST, R_FOLD_TOMB_SEGMENT, INCREMENTAL)
    expect("R14 tomb-in-segment outranks the block threshold",
           dict(n_added=1, n_oov=5, n_tomb_seg_rows=1, n_segments=3,
                n_blocks=99, compact_blocks=8),
           FOLD_FIRST, R_FOLD_TOMB_SEGMENT, STAGED)
    expect("R15 in-vocab path outranks the block threshold",
           dict(n_added=1, n_segments=3, n_blocks=99, compact_blocks=8),
           FOLD_FIRST, R_FOLD_INPLACE, INCREMENTAL)

    # R16–R18 —— 跨界点与 then 的语义
    expect("R16 batch over crossover -> rebuild",
           dict(n_added=1, n_oov=5, batch_nnz=10_000, rebuild_batch_nnz=9_999),
           REBUILD, R_REBUILD_BATCH_TOO_BIG, None)
    expect("R17 crossover is strict",
           dict(n_added=1, n_oov=5, batch_nnz=9_999, rebuild_batch_nnz=9_999),
           STAGED, R_STAGED, None)
    check("R18 then is None on non-fold routes",
          all(route_update(**kw)["then"] is None for kw in (
              dict(), dict(n_added=1), dict(n_added=1, n_oov=1),
              dict(n_added=1, n_oov=1, dense_on=True))))

    # R19–R20 —— 入参守卫
    try:
        route_update(n_added=-1)
        check("R19 negative counter raises", False, "没有抛错")
    except ValueError:
        check("R19 negative counter raises", True)
    try:
        route_update(n_added=1, compact_blocks=1)
        check("R20 compact_blocks < 2 raises", False, "没有抛错")
    except ValueError:
        check("R20 compact_blocks < 2 raises", True)

    # 默认阈值只作为常量存在，别让它在测试里悄悄漂
    check("R21 default compact_blocks is 8", DEFAULT_COMPACT_BLOCKS == 8)

    print()
    if _fails:
        print(f"FAILED: {len(_fails)} / {_n[0]} → {_fails}")
        return 1
    print(f"ALL PASS ({_n[0]} checks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
