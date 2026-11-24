#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""0.4 P0 回归测试：SparseBM25.load 零解压旁路与 _MetaSeq .meta.cache。

Run:  python tests/test_v04_p0_coldstart.py
"""
from __future__ import annotations

import atexit
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import primolix                                    # noqa: E402
from primolix import core, kernel                  # noqa: E402
from primolix.core import _MetaSeq                 # noqa: E402
from primolix.kernel import SparseBM25             # noqa: E402

core.VERBOSE = False
kernel.VERBOSE = False

FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  |  {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


def main() -> int:
    work = HERE / "_v04_p0_run"
    shutil.rmtree(work, ignore_errors=True)
    atexit.register(shutil.rmtree, work, ignore_errors=True)

    docs = work / "docs"
    idx_dir = work / "idx"
    docs.mkdir(parents=True)
    (docs / "a.md").write_text("零解压 快速 加载 测试 文档一：冷启动 不 解压 矩阵 与 词表 缓存 验证。\n\n增量 词表 缓存 验证 保持 稳定。\n", encoding="utf-8")
    (docs / "b.md").write_text("冷启动 旁路 映射 语义 检索 文档二：测试 索引 快速 读取 与 零 解压 路径。\n", encoding="utf-8")
    (docs / "c.py").write_text("def cold_start():\n    return 'primolix zero decompression test function'\n", encoding="utf-8")

    with primolix.build(str(docs), str(idx_dir)) as idx0:
        ref_hits = idx0.search("冷启动 检索", k=3)
        ref_scores = np.asarray(idx0.bm.score_all(["冷启动", "检索"]), dtype=np.float32)

    # ---- P0-1: SparseBM25.load 不得触发 zip 内大数组解压 ----
    opened_zip_entries: list[str] = []
    orig_zip_open = zipfile.ZipFile.open

    def tracked_zip_open(self, name, *args, **kwargs):
        member = name.filename if hasattr(name, "filename") else str(name)
        opened_zip_entries.append(member)
        return orig_zip_open(self, name, *args, **kwargs)

    zipfile.ZipFile.open = tracked_zip_open
    try:
        with SparseBM25.load(str(idx_dir / "bm25.npz")) as bm_fast:
            fast_scores = np.asarray(bm_fast.score_all(["冷启动", "检索"]), dtype=np.float32)
            v_fast = len(bm_fast.idx2term)
            is_mmap_vocab = bm_fast._vocab_carrier is not None
    finally:
        zipfile.ZipFile.open = orig_zip_open

    forbidden = {"T_data.npy", "T_idx.npy", "T_ptr.npy", "vocab.npy", "dl.npy", "_idf.npy", "dead.npy"}
    touched_forbidden = sorted(forbidden.intersection(opened_zip_entries))
    diff_fast = float(np.max(np.abs(ref_scores - fast_scores)))
    check(
        "P0-1 SparseBM25.load skips decompressing T_*/vocab/dl/_idf/dead",
        len(touched_forbidden) == 0 and is_mmap_vocab and diff_fast == 0.0 and v_fast > 0,
        f"opened={opened_zip_entries} touched_forbidden={touched_forbidden} max|d|={diff_fast:.2e}",
    )

    # ---- P0-2: .meta.cache 生成、命中等价性与惰性 text 读取 ----
    meta_path = idx_dir / "meta.jsonl"
    cache_path = _MetaSeq._cache_path(meta_path)
    cache_exists_after_build = cache_path.exists()

    seq_raw, top_raw, ndel_raw = _MetaSeq.from_jsonl(meta_path, use_cache=False)
    seq_hit, top_hit, ndel_hit = _MetaSeq.from_jsonl(meta_path, use_cache=True)
    # 校验数组字段与偏移逐元素一致，且惰性 text 仍可精确取回
    meta_eq = (
        top_raw == top_hit
        and ndel_raw == ndel_hit
        and np.array_equal(seq_raw.rows, seq_hit.rows)
        and np.array_equal(seq_raw.lines, seq_hit.lines)
        and np.array_equal(seq_raw.char_starts, seq_hit.char_starts)
        and np.array_equal(seq_raw.char_ends, seq_hit.char_ends)
        and np.array_equal(seq_raw._toff, seq_hit._toff)
        and list(seq_raw.docids) == list(seq_hit.docids)
        and list(seq_raw.files) == list(seq_hit.files)
        and [seq_raw[i]["text"] for i in range(len(seq_raw))]
        == [seq_hit[i]["text"] for i in range(len(seq_hit))]
    )
    seq_raw.close()
    seq_hit.close()
    check(
        "P0-2 .meta.cache roundtrip matches from_jsonl(use_cache=False) bit-for-bit",
        cache_exists_after_build and meta_eq,
        f"cache={cache_path.name} rows={len(seq_hit)} max_row={top_hit}",
    )

    # ---- P0-3: .meta.cache 损坏时自动回退 meta.jsonl 并自愈 ----
    cache_path.write_bytes(b"CORRUPT_HEADER_DATA")
    seq_rec, top_rec, ndel_rec = _MetaSeq.from_jsonl(meta_path, use_cache=True)
    rec_texts = [seq_rec[i]["text"] for i in range(len(seq_rec))]
    seq_rec.close()
    # 损坏的缓存应已被重写自愈，下一次可命中
    seq_healed = _MetaSeq._load_cache(meta_path, meta_path.stat())
    healed_ok = seq_healed is not None and len(seq_healed[0]) == len(rec_texts)
    if seq_healed is not None:
        seq_healed[0].close()
    check(
        "P0-3 corrupt .meta.cache falls back cleanly and self-heals",
        top_rec == top_raw and ndel_rec == ndel_raw and len(rec_texts) == len(seq_raw) and healed_ok,
        f"healed={healed_ok} count={len(rec_texts)}",
    )

    # ---- P0-4: delta 追加修改 meta.jsonl 后缓存自动失效且重载一致 ----
    core.set_persist_mode("delta")
    with primolix.open(str(idx_dir)) as idx1:
        (docs / "a.md").write_text("旁路 映射 语义 检索 文档一：测试 索引 快速 读取 与 零 解压 路径。\n", encoding="utf-8")
        rep = idx1._z.update(str(docs))
        cache_cleared_or_fresh = not cache_path.exists()
        expected_hits = idx1.search("旁路", k=3)
    with primolix.open(str(idx_dir)) as idx2:
        reloaded_hits = idx2.search("旁路", k=3)
    check(
        "P0-4 meta append invalidates stale .meta.cache across delta reload",
        rep.get("kind") == "incremental"
        and cache_cleared_or_fresh
        and len(expected_hits) == 2
        and len(reloaded_hits) == 2
        and [h["docid"] for h in expected_hits] == [h["docid"] for h in reloaded_hits]
        and len(ref_hits) > 0,
        f"kind={rep.get('kind')} cleared_before_reload={cache_cleared_or_fresh} hits={len(reloaded_hits)}",
    )

    # ---- P0-5: Windows 句柄可彻底释放并删除目录 ----
    rm_err = None
    try:
        shutil.rmtree(work)
    except Exception as e:
        rm_err = repr(e)
    check("P0-5 clean Windows rmtree with .meta.cache present", rm_err is None, f"rm_err={rm_err}")

    print("-" * 60)
    print(f"{'ALL PASS' if not FAILS else 'FAILURES: ' + ', '.join(FAILS)}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
