#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""0.4 P1 回归测试：OverlayVocab 增量保真、Windows 双缓冲安全落盘、统一 compact(mode='auto')。

Run:  python tests/test_v04_p1_overlay_compact.py
"""
from __future__ import annotations

import atexit
import hashlib
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import primolix                                    # noqa: E402
from primolix import Reranker, core, kernel        # noqa: E402
from primolix.vocab_mmap import OverlayVocab, VocabMmap  # noqa: E402

core.VERBOSE = False
kernel.VERBOSE = False

FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  |  {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


def _dummy_encode(texts):
    out = np.zeros((len(texts), 3), dtype=np.float32)
    for i, t in enumerate(texts):
        v0 = 1.0 if "语义" in t else 0.1
        v1 = 1.0 if "压实" in t else 0.1
        h = int(hashlib.md5(t.encode("utf-8")).hexdigest()[:4], 16) % 50
        out[i] = [v0, v1, 0.01 * (h + 1)]
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)


def main() -> int:
    work = HERE / "_v04_p1_run"
    shutil.rmtree(work, ignore_errors=True)
    atexit.register(shutil.rmtree, work, ignore_errors=True)

    docs = work / "docs"
    idx_dir = work / "idx"
    docs.mkdir(parents=True)
    for i in range(6):
        (docs / f"d{i}.md").write_text(
            f"基础语料文档编号{i}：检索 系统 词表 映射 测试，包含常见关键词 语义 与 压实 验证。\n",
            encoding="utf-8",
        )

    core.set_persist_mode("delta")
    with primolix.build(str(docs), str(idx_dir)) as idx0:
        pass

    # ---- P1-1: OverlayVocab 保持 VocabMmap 不被 OOV 增量炸成全量 dict ----
    with primolix.open(str(idx_dir)) as idx1:
        base_carrier = idx1.bm._vocab_carrier
        base_is_mmap = isinstance(idx1.bm.idx2term, VocabMmap)
        st = idx1.stage_begin("seg_ov1")
        st.add("ov_doc_1", "这是一段带有 全新词表外标识符OverlayToken 的发布段落，测试分层词表代理。")
        st.publish()
        after_stage_overlay = (
            isinstance(idx1.bm.idx2term, OverlayVocab)
            and idx1.bm.idx2term is idx1.bm.term2idx
            and idx1.bm.idx2term.base is base_carrier
        )
        ov_hits = idx1.search("OverlayToken", k=2)
        um = idx1.unmount("seg_ov1")
        after_unmount_mmap = (
            isinstance(idx1.bm.idx2term, VocabMmap)
            and idx1.bm.idx2term is base_carrier
            and idx1.bm.term2idx is base_carrier
        )
    check(
        "P1-1 OverlayVocab wraps VocabMmap on staged OOV & restores on unmount",
        base_is_mmap and after_stage_overlay and len(ov_hits) == 1 and after_unmount_mmap and um["N"] == 6,
        f"base_mmap={base_is_mmap} overlay={after_stage_overlay} unmount_mmap={after_unmount_mmap}",
    )

    # ---- P1-2: compact(mode='delta') Tier-1 合并多 delta 且不触碰基线 bm25.npz ----
    old_max = core.DELTA_MAX_FILES
    old_ratio = core.DELTA_COMPACT_RATIO
    core.DELTA_MAX_FILES = 999
    core.DELTA_COMPACT_RATIO = 999.0
    try:
        with primolix.open(str(idx_dir)) as idx2:
            for j in range(3):
                (docs / f"inc_{j}.md").write_text(
                    f"增量追加批次 {j}：检索 系统 词表 测试 验证 文档内容，补充 语义 信号。\n",
                    encoding="utf-8",
                )
                idx2.add(str(docs))
            h_before_d = idx2.health()
            npz_mtime_before = (idx_dir / "bm25.npz").stat().st_mtime_ns
            ref_scores_d = np.asarray(idx2.bm.score_all(["检索", "系统", "语义"]), dtype=np.float32)

            rep_d = idx2.compact()  # auto -> 3 deltas, dead_ratio=0, segs=0 => mode="delta"
            h_after_d = idx2.health()
            npz_mtime_after = (idx_dir / "bm25.npz").stat().st_mtime_ns
        with primolix.open(str(idx_dir)) as idx2_re:
            re_scores_d = np.asarray(idx2_re.bm.score_all(["检索", "系统", "语义"]), dtype=np.float32)
    finally:
        core.DELTA_MAX_FILES = old_max
        core.DELTA_COMPACT_RATIO = old_ratio

    diff_d = float(np.max(np.abs(ref_scores_d - re_scores_d)))
    check(
        "P1-2 compact(mode='auto') runs Tier-1 delta merge without touching bm25.npz",
        h_before_d["delta_files"] == 3
        and rep_d["mode"] == "delta"
        and rep_d["merged_deltas"] == 3
        and h_after_d["delta_files"] == 1
        and npz_mtime_before == npz_mtime_after
        and diff_d == 0.0,
        f"before={h_before_d['delta_files']} after={h_after_d['delta_files']} same_npz={npz_mtime_before == npz_mtime_after} max|d|={diff_d:.2e}",
    )

    # ---- P1-3: staged -> compact(mode='delta') 折段进 delta 后重载仍走 OverlayVocab ----
    with primolix.open(str(idx_dir)) as idx3:
        st3 = idx3.stage_begin("seg_d1")
        st3.add("ov_doc_2", "这是第二段带有 DeltaOverlay词 的文档，折为 delta 后重载仍应维持 VocabMmap 基座。")
        st3.publish()
        rep_sd = idx3.compact(mode="delta")
    with primolix.open(str(idx_dir)) as idx3_re:
        delta_overlay_ok = (
            isinstance(idx3_re.bm.idx2term, OverlayVocab)
            and idx3_re.bm._vocab_carrier is not None
            and "deltaoverlay" in idx3_re.bm.term2idx
        )
        sd_hits = idx3_re.search("DeltaOverlay词", k=2)
    check(
        "P1-3 _replay_deltas keeps VocabMmap base via OverlayVocab",
        rep_sd["mode"] == "delta" and delta_overlay_ok and len(sd_hits) == 1,
        f"overlay={delta_overlay_ok} hits={len(sd_hits)}",
    )

    # ---- P1-4: dead_ratio >= 0.2 时 compact(mode='auto') 执行 Tier-2 full 回收墓碑 + 同步侧车 ----
    rr = Reranker(pool_k=20, sub_batch=4, cache_view=True, encoder=_dummy_encode)
    with primolix.open(str(idx_dir)) as idx4:
        # 挂载 dense 视图并制造 >= 30% 死行
        _ = idx4.search("检索 语义", k=5, rerank=rr)
        for i in range(4):
            p = docs / f"d{i}.md"
            if p.exists():
                p.unlink()
        idx4.add(str(docs))
        h_sick = idx4.health()
        live_ref = {h["docid"]: h["score"] for h in idx4.search("压实 验证", k=20)}

        rep_full = idx4.compact()  # auto -> dead_ratio >= 0.2 => full
        h_clean = idx4.health()
        # 验证就地双缓冲切换后 .mm 与 .vocab 世代键严格同步，且当前实例仍可继续检索/重排
        after_hits = idx4.search("压实 验证", k=20)
        rr_hits = idx4.search("压实 验证", k=5, rerank=rr)
        npzp = idx_dir / "bm25.npz"
        stp = npzp.stat()
        want_gen = f"v1|{npzp.name}|{stp.st_mtime_ns}|{stp.st_size}"
        mm_key_ok = (idx_dir / "bm25.npz.mm" / "mm_key.txt").read_text(encoding="utf-8").startswith(want_gen)
        v_key_ok = (idx_dir / "bm25.npz.vocab" / "vocab_key.txt").read_text(encoding="utf-8").startswith(want_gen)

    with primolix.open(str(idx_dir)) as idx4_re:
        re_hits = idx4_re.search("压实 验证", k=20)
        re_mmap_ok = isinstance(idx4_re.bm.idx2term, VocabMmap) and idx4_re.bm._mm_postings is not None

    after_map = {h["docid"]: h["score"] for h in after_hits}
    re_map = {h["docid"]: h["score"] for h in re_hits}
    score_diff_full = max(
        [abs(live_ref[d] - after_map[d]) for d in live_ref]
        + [abs(live_ref[d] - re_map[d]) for d in live_ref],
        default=0.0,
    )
    ok_p1_4 = (
        h_sick["compact_recommended"]
        and rep_full["mode"] == "full"
        and rep_full["reclaimed_dead"] >= 4
        and not h_clean["compact_recommended"]
        and h_clean["dead"] == 0
        and h_clean["delta_files"] == 0
        and h_clean["segments"] == 0
        and mm_key_ok and v_key_ok
        and re_mmap_ok
        and set(live_ref) == set(after_map) == set(re_map)
        and score_diff_full < 1e-6
        and len(rr_hits) > 0
    )
    check(
        "P1-4 compact(mode='auto') reclaims dead rows, converges health(), & syncs .mm/.vocab on Windows",
        ok_p1_4,
        f"sick={h_sick['compact_reasons']} reclaimed={rep_full.get('reclaimed_dead')} "
        f"clean_dead={h_clean['dead']} keys=({mm_key_ok},{v_key_ok}) max|d|={score_diff_full:.2e}",
    )

    # ---- P1-5: Windows 句柄清理后可整目录 rmtree ----
    rm_err = None
    try:
        shutil.rmtree(work)
    except Exception as e:
        rm_err = repr(e)
    check("P1-5 clean Windows rmtree after OverlayVocab & full compact", rm_err is None, f"rm_err={rm_err}")

    print("-" * 60)
    print(f"{'ALL PASS' if not FAILS else 'FAILURES: ' + ', '.join(FAILS)}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
