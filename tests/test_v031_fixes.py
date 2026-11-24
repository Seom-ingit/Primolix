#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""0.3.1 回归测试：三层 close() 句柄释放、unmount 同步、tc_shared 校验/抢占、启动预检、rank_delta/health。

Run:  python tests/test_v031_fixes.py
"""
from __future__ import annotations

import atexit
import hashlib
import shutil
import subprocess
import sys
import time
from multiprocessing import shared_memory
from pathlib import Path

import numpy as np
from scipy import sparse as sp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import primolix                                    # noqa: E402
from primolix import Reranker, core, kernel        # noqa: E402
from primolix import tc_shared                     # noqa: E402

core.VERBOSE = False

FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  |  {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


def _dummy_encode(texts):
    out = np.zeros((len(texts), 3), dtype=np.float32)
    for i, t in enumerate(texts):
        v0 = 1.0 if "语义" in t else 0.1
        v1 = 1.0 if "增量" in t else 0.1
        h = int(hashlib.md5(t.encode("utf-8")).hexdigest()[:4], 16) % 50
        out[i] = [v0, v1, 0.01 * (h + 1)]
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)


def main() -> int:
    work = HERE / "_v031_run"
    shutil.rmtree(work, ignore_errors=True)
    atexit.register(shutil.rmtree, work, ignore_errors=True)

    docs = work / "docs"
    idx_dir = work / "idx"
    docs.mkdir(parents=True)
    (docs / "a.md").write_text("检索 检索 检索 普通背景介绍，没有额外线索。\n\n另一段普通内容用于测试。\n", encoding="utf-8")
    (docs / "b.md").write_text("这里讨论 检索 与 语义 重排机制，语义 信号强。\n", encoding="utf-8")

    # ---- C1: 三层 close() + 上下文管理释放 Windows mmap / 文件句柄 ----
    idx0 = primolix.build(str(docs), str(idx_dir), w_cache=True)
    idx0.close()
    idx0.close()  # 幂等

    with primolix.open(str(idx_dir)) as idx1:
        hits1 = idx1.search("语义 检索", k=2)
        _ = idx1.bm.term2idx.get("语义")

    # 若未释放 .mm/.vocab/meta.jsonl 句柄，Windows 上 rmtree 会报 PermissionError
    copy_dir = work / "idx_copy"
    shutil.copytree(idx_dir, copy_dir)
    with primolix.open(str(copy_dir)) as idx_tmp:
        _ = idx_tmp.search("检索", k=2)
    rm_err = None
    try:
        shutil.rmtree(copy_dir)
    except Exception as e:
        rm_err = repr(e)
    check("C1 3-layer close() releases mmap/file handles for clean rmtree on Windows",
          rm_err is None and not copy_dir.exists() and len(hits1) > 0,
          f"rm_err={rm_err}")

    # ---- C2: Index.unmount() 同步清理 manifest 与 _meta ----
    with primolix.open(str(idx_dir)) as idx2:
        base_n = int(idx2.bm.N)
        _, base_total = idx2.browse()
        base_hits = idx2.search("跨段新词", k=3)
        st = idx2.stage_begin("seg_u1")
        st.add("seg_doc_1", "这是一篇包含 跨段新词 的旁路发布文档用于验证卸载同步。")
        st.publish()
        mounted_hits = idx2.search("跨段新词", k=3)
        um = idx2.unmount("seg_u1")
        after_hits = idx2.search("跨段新词", k=3)
        _, after_total = idx2.browse()

    with primolix.open(str(idx_dir)) as idx2_re:
        re_segs = idx2_re.segments()["mounted"]
        re_n = int(idx2_re.bm.N)
        re_hits = idx2_re.search("跨段新词", k=3)
        _, re_total = idx2_re.browse()

    ok_c2 = (
        len(mounted_hits) == 1
        and um.get("meta_removed") == 1
        and after_hits == base_hits == []
        and after_total == base_total
        and int(re_n) == base_n
        and re_segs == {}
        and re_hits == []
        and re_total == base_total
    )
    check("C2 Index.unmount() syncs manifest and _meta across reload",
          ok_c2, f"um={um} re_segs={re_segs} re_n={re_n}/{base_n}")

    # ---- C3: tc_shared CRC32 校验与死 PID _STATE_BUILDING 抢占 ----
    T_test = sp.csr_matrix([[1.0, 2.0, 0.0], [0.0, 1.0, 3.0]], dtype=np.float32)
    key_crc = f"v031_crc_{time.time_ns()}"
    key_stale = f"v031_stale_{time.time_ns()}"
    keep_handles = []
    try:
        tc_shared.cleanup(key_crc)
        tc_shared.cleanup(key_stale)
        csc1, info1 = tc_shared.publish_or_attach(T_test, key_crc)
        keep_handles.append(info1["shm"])  # Windows 上保留一个句柄以维持命名段存活

        # 破坏 1 字节载荷，应触发 checksum mismatch 自愈重建
        shm_bad = shared_memory.SharedMemory(name=tc_shared.name_of(key_crc))
        try:
            shm_bad.buf[tc_shared.HDR] ^= 0xFF
        finally:
            shm_bad.close()

        kind_bad, _ = tc_shared._try_attach(key_crc, T_test.shape, tc_shared.csc_sizes(T_test)[0], verify_crc=True)
        csc_healed, info_healed = tc_shared.publish_or_attach(T_test, key_crc)
        if info_healed.get("shm") is not None:
            keep_handles.append(info_healed["shm"])

        # 写入已退出子进程的 PID 作为陈旧 _STATE_BUILDING 锁，应在 < 1s 内抢占而非干等 3s
        dead_proc = subprocess.Popen([sys.executable, "-c", "pass"])
        dead_proc.wait()
        want_s, size_s = tc_shared.csc_sizes(T_test)
        shm_stale = shared_memory.SharedMemory(name=tc_shared.name_of(key_stale), create=True, size=size_s)
        keep_handles.append(shm_stale)
        shm_stale.buf[:tc_shared.HDR] = tc_shared._pack(
            key_stale, want_s, tc_shared._STATE_BUILDING, 0.0,
            owner_pid=dead_proc.pid, build_ts=time.time(),
        )
        t_pre0 = time.perf_counter()
        csc_pre, info_pre = tc_shared.publish_or_attach(T_test, key_stale)
        pre_dt = time.perf_counter() - t_pre0
        if info_pre.get("shm") is not None:
            keep_handles.append(info_pre["shm"])

        ok_c3 = (
            csc1 is not None
            and kind_bad == "corrupt"
            and csc_healed is not None and bool(info_healed.get("built_here"))
            and csc_pre is not None and bool(info_pre.get("shared")) and pre_dt < 1.5
        )
        check("C3 tc_shared CRC32 corrupt recovery & dead-PID stale lock preemption",
              ok_c3, f"bad={kind_bad} healed={info_healed.get('built_here')} pre_dt={pre_dt:.3f}s")
    finally:
        for h in keep_handles:
            try:
                h.close()
            except Exception:
                pass
        tc_shared.cleanup(key_crc)
        tc_shared.cleanup(key_stale)

    # ---- C4: 启动完整性预检 (whats_missing #7b) ----
    chk_dir = work / "idx_precheck"
    shutil.copytree(idx_dir, chk_dir)
    # 删掉 .mm/idf.npy 与 .vocab/vocab.blob，不应再报 TypeError: 'NoneType'
    (chk_dir / "bm25.npz.mm" / "idf.npy").unlink()
    (chk_dir / "bm25.npz.vocab" / "vocab.blob").unlink()
    with primolix.open(str(chk_dir)) as idx_chk:
        hits_chk = idx_chk.search("语义 检索", k=2)
    # 删掉 bm25.npz 必须给出明确含文件名的 FileNotFoundError
    (chk_dir / "bm25.npz").unlink()
    missing_msg = ""
    try:
        primolix.open(str(chk_dir))
    except FileNotFoundError as e:
        missing_msg = str(e)
    check("C4 startup integrity pre-check handles missing .mm/.vocab & missing bm25.npz",
          len(hits_chk) > 0 and "bm25.npz" in missing_msg,
          f"hits={len(hits_chk)} msg={missing_msg[:60]}")

    # ---- C5: rerank rank_delta & Index.health() ----
    rr = Reranker(pool_k=10, sub_batch=2, cache_view=True, encoder=_dummy_encode)
    with primolix.open(str(idx_dir)) as idx_h:
        h_hits = idx_h.search("检索 语义", k=2, rerank=rr)
        h0 = idx_h.health()
        idx_h.bm._tombstone_rows([0])
        h1 = idx_h.health()
    r0 = h_hits[0]
    ok_c5 = (
        "rank_delta" in r0
        and r0["rank_delta"] == r0["pool_rank"] - r0["rank"]
        and not h0["compact_recommended"]
        and h1["dead"] == 1 and h1["dead_ratio"] > 0.2 and h1["compact_recommended"]
        and h1["rerank_status"] is not None
    )
    check("C5 rerank rank_delta & Index.health() self-inspection",
          ok_c5, f"rank_delta={r0.get('rank_delta')} dead_ratio={h1['dead_ratio']:.2f}")

    shutil.rmtree(work, ignore_errors=True)
    print("-" * 60)
    print(f"{'ALL PASS' if not FAILS else 'FAILURES: ' + ', '.join(FAILS)}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
