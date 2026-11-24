#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression tests for source-audit fixes #1-#5 (2026-11-22).

Run:  python tests/test_audit_fixes.py

Checks:
  A1 save_view + load_views + Index.fuse roundtrip (fixes _say(..., file=sys.stderr) TypeError)
  A2 STAGED (with tombstones) -> delta fold -> Tier-2 compaction preserves folded segment meta
  A3 rebuild_in_place syncs persistence cursors so subsequent delta updates do not duplicate vocab or drop meta
  A4 mount_segment extends views (no dead/cov broadcast ValueError), unmount shrinks views, and _write_dict invalidates
  A5 explain() and diagnose() use live-doc idf/df_ratio matching search() after deletions
"""
from __future__ import annotations

import atexit
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import primolix                                    # noqa: E402
from primolix import core, kernel as K, Primolix   # noqa: E402

core.VERBOSE = False

FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  |  {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


def pad(text: str) -> str:
    return (text + " ") * 4 + ("  " * 8)


def main() -> int:
    work = HERE / "_audit_fixes_run"
    shutil.rmtree(work, ignore_errors=True)
    atexit.register(shutil.rmtree, work, ignore_errors=True)

    # ---- A1: save_view + load_views + fuse ----
    d1 = work / "a1_docs"
    i1 = work / "a1_idx"
    d1.mkdir(parents=True)
    (d1 / "a.txt").write_text(pad("alpha beta gamma delta"), encoding="utf-8")
    (d1 / "b.txt").write_text(pad("alpha epsilon zeta"), encoding="utf-8")
    idx1 = primolix.build(str(d1), str(i1))
    docids = list(idx1._z._meta.docids)
    vecs = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    saved_path = idx1.save_view("v1", docids, vecs, meta={"model_id": "unit-test"})
    idx1_re = primolix.open(str(i1))
    loaded = idx1_re.load_views()
    qvec = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
    bm_hits = idx1_re.search("alpha", k=2)
    fu_hits = idx1_re.fuse("alpha", qvec, "v1", alpha=0.5, k=2)
    idx1_re.detach_view("v1")
    fu_detached = idx1_re.fuse("alpha", qvec, "v1", alpha=0.5, k=2)
    bm_map = {h["file"]: h["score"] for h in bm_hits}
    fu_map = {h["file"]: h["score"] for h in fu_hits}
    de_map = {h["file"]: h["score"] for h in fu_detached}
    ok_a1 = (
        Path(saved_path).exists()
        and len(loaded) == 1
        and loaded[0].get("attached") is True
        and loaded[0].get("unmatched") == 0
        and fu_map["b.txt"] > bm_map["b.txt"]
        and fu_map["a.txt"] == bm_map["a.txt"]
        and de_map == bm_map
    )
    check("A1 save_view / load_views / fuse roundtrip", ok_a1,
          f"loaded={loaded} · b_bm={bm_map.get('b.txt'):.4f} b_fu={fu_map.get('b.txt'):.4f}")

    # ---- A2: STAGED (with tombstones) -> delta fold -> Tier-2 preserves meta ----
    d2 = work / "a2_docs"
    i2 = work / "a2_idx"
    d2.mkdir(parents=True)
    (d2 / "f0.txt").write_text(pad("alpha common topic one"), encoding="utf-8")
    (d2 / "f1.txt").write_text(pad("beta common topic two"), encoding="utf-8")
    (d2 / "f2.txt").write_text(pad("gamma common topic three"), encoding="utf-8")
    (d2 / "f3.txt").write_text(pad("delta common topic four"), encoding="utf-8")
    old_mode = core.PERSIST_MODE
    old_ratio = core.DELTA_COMPACT_RATIO
    try:
        core.set_persist_mode("delta")
        z2 = Primolix.build(str(d2), str(i2))
        n_before_stage = int(z2._bm.T.shape[0])
        # Update 1: modify f0 with OOV term -> STAGED with tomb_rows non-empty
        (d2 / "f0.txt").write_text(pad("alpha oovstagedtoken one"), encoding="utf-8")
        r_stage = z2.update(str(d2))
        persisted_after_stage = int(z2._persisted_N)
        # Update 2: in-vocab modify f1 -> FOLD_FIRST (delta) + incremental
        (d2 / "f1.txt").write_text(pad("beta alpha common two"), encoding="utf-8")
        r_fold = z2.update(str(d2))
        # Update 3: trigger Tier-2 compaction (meta=False + _clear_deltas)
        core.DELTA_COMPACT_RATIO = 0.001
        (d2 / "f2.txt").write_text(pad("gamma beta common three"), encoding="utf-8")
        r_tier2 = z2.update(str(d2))
        z2_re = Primolix(str(i2))
        z2_re._load()
        hits_oov = z2_re.query("oovstagedtoken", k=5)
        ok_a2 = (
            r_stage["kind"] == "staged"
            and persisted_after_stage <= n_before_stage
            and r_fold["kind"] == "incremental"
            and "压实2" in r_tier2.get("detail", "")
            and len(hits_oov) == 1
            and hits_oov[0]["file"] == "f0.txt"
        )
        check("A2 STAGED(tomb) -> delta fold -> Tier-2 preserves segment meta", ok_a2,
              f"persisted_N={persisted_after_stage}/{n_before_stage} · tier2={r_tier2.get('detail')!r} · hits={len(hits_oov)}")
    finally:
        core.DELTA_COMPACT_RATIO = old_ratio
        core.set_persist_mode(old_mode)

    # ---- A3: rebuild_in_place syncs cursors before subsequent delta update ----
    d3 = work / "a3_docs"
    i3 = work / "a3_idx"
    d3.mkdir(parents=True)
    for k in range(5):
        (d3 / f"g{k}.txt").write_text(pad(f"sharedterm item{k} base"), encoding="utf-8")
    try:
        core.set_persist_mode("delta")
        z3 = Primolix.build(str(d3), str(i3))
        # Delete 3 files -> old _persisted_N=5; then rebuild_in_place with 2 files + OOV terms -> N=2 < 5, V1 > V0
        for k in (2, 3, 4):
            (d3 / f"g{k}.txt").unlink()
        (d3 / "g0.txt").write_text(pad("sharedterm brandnewoovzero"), encoding="utf-8")
        z3.rebuild_in_place(str(d3))
        v_rebuilt, n_rebuilt = int(z3._bm.V), int(z3._bm.N)
        cursors_ok = (z3._persisted_V == v_rebuilt and z3._persisted_N == n_rebuilt
                      and z3._delta_count == 0 and z3._delta_nnz == 0)
        # Now append an in-vocab file in delta mode, then reload
        (d3 / "g_added.txt").write_text(pad("sharedterm brandnewoovzero"), encoding="utf-8")
        rep_inc = z3.update(str(d3))
        s_mem = z3._bm.score_all(K.zh_tokenize("brandnewoovzero"))
        z3_re = Primolix(str(i3))
        z3_re._load()
        s_disk = z3_re._bm.score_all(K.zh_tokenize("brandnewoovzero"))
        max_d = float(np.abs(s_mem - s_disk).max())
        ok_a3 = (
            cursors_ok
            and rep_inc["kind"] == "incremental"
            and int(z3_re._bm.V) == int(z3._bm.V)
            and len(z3_re._meta) == len(z3._meta)
            and max_d == 0.0
        )
        check("A3 rebuild_in_place + delta update keeps V/meta/scores exact", ok_a3,
              f"cursors_ok={cursors_ok} · V={z3_re._bm.V}/{z3._bm.V} · meta={len(z3_re._meta)}/{len(z3._meta)} · max|d|={max_d:.3e}")
    finally:
        core.set_persist_mode(old_mode)

    # ---- A4: mount_segment / unmount_segment sync _views and invalidate _write_dict ----
    seg_dir = work / "a4_seg"
    seg_dir.mkdir(parents=True)
    bm4 = K.SparseBM25(["alpha beta", "alpha gamma", "delta epsilon", "alpha zeta"], lambda s: s.split())
    v4 = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    bm4.attach_view("v", v4, rows=[0, 1])
    bm4._tombstone_rows([2])  # _dead is now non-None (size 4)
    # Build a 2-row segment with 1 new term ("segonly")
    import scipy.sparse as sp
    V0 = bm4.V
    csc = sp.csc_matrix(
        (np.asarray([1.0, 1.0], np.float32),
         (np.asarray([0, 1], np.int32), np.asarray([0, V0], np.int32))),
        shape=(2, V0 + 1),
    )
    # Note: col 0 is term idx2term[0]
    np.save(seg_dir / "s1.data.npy", np.asarray(csc.data, np.float32))
    np.save(seg_dir / "s1.idx.npy", np.asarray(csc.indices, np.int32))
    np.save(seg_dir / "s1.ptr.npy", np.asarray(csc.indptr, np.int32))
    bm4.mount_segment(str(seg_dir), "s1", ["segonly"], np.asarray([1.0, 1.0], np.float32), rows=2)
    fused_ok = False
    try:
        sc_fused = bm4.fused_scores(["alpha"], np.asarray([1.0, 0.0], np.float32), "v", alpha=0.4)
        fused_ok = (sc_fused.shape == (6,))
    except Exception:
        fused_ok = False
    # Attach a view covering segment row 4, unmount s1, then append new rows -> row 4 must NOT stay covered
    bm4.attach_view("vseg", np.asarray([[1.0, 0.0], [0.0, 1.0]], np.float32), rows=[0, 4])
    bm4.unmount_segment("s1")
    bm4.append_docs([["alpha"], ["alpha"]])
    cov_after = bm4.view_report(["vseg"])[0]["covered_total"]

    # Also verify _write_dict invalidation across mount -> fold -> append
    old_wd = K.SparseBM25.WRITE_DICT
    try:
        K.SparseBM25.WRITE_DICT = True
        bm4.append_docs([["alpha"]])  # populates _write_dict before mount
        bm4.mount_segment(str(seg_dir), "s1", ["segonly"], np.asarray([1.0, 1.0], np.float32), rows=2)
        bm4.fold_segment("s1")
        new_rows = bm4.append_docs([["segonly"]])
        col_seg = bm4.term2idx["segonly"]
        tf_written = float(bm4.T[int(new_rows[0]), col_seg])
    finally:
        K.SparseBM25.WRITE_DICT = old_wd
    check("A4 mount/unmount sync _views & invalidate _write_dict",
          fused_ok and cov_after == 1 and tf_written == 1.0,
          f"fused_ok={fused_ok} · cov_after_unmount={cov_after} · tf_after_fold={tf_written}")

    # ---- A5: explain() and diagnose() match search() n_live after deletions ----
    d5 = work / "a5_docs"
    i5 = work / "a5_idx"
    d5.mkdir(parents=True)
    for k in range(6):
        w = "targetword" if k < 2 else "otherword"
        (d5 / f"h{k}.txt").write_text(pad(f"commontoken {w} doc{k}"), encoding="utf-8")
    idx5 = primolix.build(str(d5), str(i5))
    for k in (3, 4, 5):
        (d5 / f"h{k}.txt").unlink()
    idx5.add(str(d5))  # 3 dead tombstones, n_active=3, bm.N=6
    hits5 = idx5.search("commontoken targetword", k=1)
    exp5 = idx5.explain("commontoken targetword", k=1)
    diag5 = idx5.diagnose("commontoken targetword")
    sum_contrib = sum(x["contribution"] for x in exp5 if x["status"] == "ok")
    score_top = hits5[0]["score"]
    df_common = diag5["terms"][0]["df"]
    ratio_common = diag5["terms"][0]["df_ratio"]
    n_act = diag5["vocab"]["n_active"]
    ok_a5 = (
        n_act == 3
        and int(idx5.bm.N) == 6
        and abs(sum_contrib - score_top) < 1e-5
        and abs(ratio_common - round(df_common / n_act, 6)) < 1e-9
    )
    check("A5 explain & diagnose match live-doc idf/df_ratio after deletes", ok_a5,
          f"score={score_top:.6f} sum_contrib={sum_contrib:.6f} · df_ratio={ratio_common} (n_active={n_act}, N={idx5.bm.N})")

    shutil.rmtree(work, ignore_errors=True)
    print("-" * 60)
    print(f"{'ALL PASS' if not FAILS else 'FAILURES: ' + ', '.join(FAILS)}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
