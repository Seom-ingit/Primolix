#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""0.3 质量产品化回归测试：批式重排 + 渐进视图缓存 + dense.npy 退役 + span 接入 + tune 预设。

Run:  python tests/test_v03_rerank_views_span.py
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
from primolix import Reranker, cli, core, rerank   # noqa: E402

core.VERBOSE = False

FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  |  {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


def _dummy_encode_factory(calls: list[int]):
    """确定性轻量编码器：按关键词映射到正交基向量，记录编码条数。"""

    def encode(texts):
        calls.append(len(texts))
        out = np.zeros((len(texts), 4), dtype=np.float32)
        for i, t in enumerate(texts):
            tl = t.lower()
            v0 = 1.0 if ("语义" in tl or "semantic" in tl) else 0.1
            v1 = 1.0 if ("增量" in tl or "incremental" in tl) else 0.1
            v2 = 1.0 if ("视图" in tl or "view" in tl) else 0.1
            h = int(hashlib.md5(tl.encode("utf-8")).hexdigest()[:4], 16) % 100
            out[i] = [v0, v1, v2, 0.01 * (h + 1)]
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(norms, 1e-12)

    return encode


def main() -> int:
    work = HERE / "_v03_run"
    shutil.rmtree(work, ignore_errors=True)
    atexit.register(shutil.rmtree, work, ignore_errors=True)

    docs = work / "docs"
    idx_dir = work / "idx"
    docs.mkdir(parents=True)

    # 多段落语料 + 代码文件，覆盖 span 与重排排序反转
    (docs / "a.md").write_text(
        "第一段关于普通系统背景介绍，出现关键词 检索 检索 检索。\n\n"
        "第二段讨论 检索 与 增量 设计细节，但没有语义线索。\n",
        encoding="utf-8",
    )
    (docs / "b.md").write_text(
        "本段讨论 检索 里的 语义 匹配与重排机制，语义 信号强。\n\n"
        "另一段补充 视图 缓存 的工程收益。\n",
        encoding="utf-8",
    )
    (docs / "c.py").write_text(
        "def search_view():\n    # 检索 视图 增量 融合 实现代码\n    return 'semantic view kernel'\n",
        encoding="utf-8",
    )

    idx = primolix.build(str(docs), str(idx_dir))

    # ---- R1: 批式去重 + 渐进视图缓存命中 + 单纯 BM25 零退化 ----
    encode_counts: list[int] = []
    enc = _dummy_encode_factory(encode_counts)
    rr = Reranker(pool_k=10, sub_batch=2, cache_view=True, encoder=enc)

    bm_plain = idx.search("检索 语义", k=3)
    res_batch = idx.search_batch(["检索 语义", "检索 语义"], k=3, rerank=rr)
    st1 = idx.rerank_status()
    encoded_first = st1.get("encoded", -1)
    hits_first = st1.get("cache_hits", -1)

    # 第二次查询命中渐进视图缓存：encoded 必须降为 0
    res_single = idx.search("检索 语义", k=3, rerank=rr)
    st2 = idx.rerank_status()
    encoded_second = st2.get("encoded", -1)
    hits_second = st2.get("cache_hits", -1)

    # 卸掉缓存视图后，纯 BM25 逐位一致
    idx.detach_view("rerank_cache")
    bm_after = idx.search("检索 语义", k=3)

    ok_r1 = (
        len(res_batch) == 2
        and res_batch[0] == res_batch[1]
        and res_batch[0] == res_single
        and "语义" in res_single[0]["text"]
        and "bm25_score" in res_single[0]
        and "rerank_score" in res_single[0]
        and "pool_rank" in res_single[0]
        and encoded_first > 0 and hits_first == 0
        and encoded_second == 0 and hits_second == encoded_first
        and bm_plain == bm_after
    )
    check("R1 batch dedup rerank + progressive view cache + BM25 non-degradation", ok_r1,
          f"1st(enc={encoded_first},hit={hits_first}) · 2nd(enc={encoded_second},hit={hits_second})")

    # ---- R2: 模型缺席诚实降级（不崩溃、与 BM25-only 逐位一致、status 可观测） ----
    fb_single = idx.search("检索 语义", k=3, rerank="/nonexistent/model/dir_xyz")
    st_fb = idx.rerank_status()
    fb_batch = idx.search_batch(["检索 语义"], k=3, rerank="/nonexistent/model/dir_xyz")
    ok_r2 = (
        fb_single == bm_plain
        and fb_batch == [bm_plain]
        and st_fb is not None
        and st_fb.get("mode") == "bm25_fallback"
        and st_fb.get("available") is False
        and bool(st_fb.get("ceiling_note"))
    )
    check("R2 missing model degrades honestly to BM25-only", ok_r2,
          f"mode={st_fb.get('mode')!r} available={st_fb.get('available')}")

    # ---- R3: 废除 dense.npy -> views/dense.npz + 增量/挂段不再整库 REBUILD ----
    d3 = work / "d3_docs"
    i3 = work / "d3_idx"
    d3.mkdir(parents=True)
    (d3 / "f1.txt").write_text("检索 内核 与 增量 索引 基础文档内容。\n" * 2, encoding="utf-8")
    (d3 / "f2.txt").write_text("普通 文本 段落，不包含 语义 特征词。\n" * 2, encoding="utf-8")

    enc_calls_3: list[int] = []
    dummy_model = _dummy_encode_factory(enc_calls_3)
    orig_get_rr = rerank.get_rerank_model
    try:
        rerank.get_rerank_model = lambda mp="", device="auto": (
            dummy_model, {"available": True, "model": "dummy-bge", "model_path": "dummy", "device": "cpu", "reason": None}
        )
        idx3 = primolix.build(str(d3), str(i3), dense=True)
        no_dense_npy = not (i3 / "dense.npy").exists()
        has_view_npz = (i3 / "views" / "dense.npz").exists()

        # 重载自动重挂视图
        idx3_re = primolix.open(str(i3))
        reloaded_has_dense = idx3_re.stats().get("dense") is True

        # 词表内增量：不走 rebuild，仅编码新增 1 条（不重编码已有 f1/f2）
        enc_calls_3.clear()
        (d3 / "f3.txt").write_text("检索 内核 与 普通 文本 段落，语义 增量 索引。\n" * 2, encoding="utf-8")
        rep_inc = idx3_re.add(str(d3))
        inc_encoded_n = sum(enc_calls_3)

        # 含 OOV 挂段：不走 rebuild，走 staged，视图同步覆盖新增段内文档
        enc_calls_3.clear()
        (d3 / "f4.txt").write_text("全新 表外新词oov03 文档，带 语义 与 检索 信号。\n" * 2, encoding="utf-8")
        rep_stg = idx3_re.add(str(d3))
        stg_encoded_n = sum(enc_calls_3)

        hits_d3 = idx3_re.search("oov03 语义", k=3, dense=True)
        idx3_re2 = primolix.open(str(i3))
        hits_d3_re = idx3_re2.search("oov03 语义", k=3, dense=True)

        ok_r3 = (
            no_dense_npy and has_view_npz and reloaded_has_dense
            and rep_inc.get("kind") == "incremental" and inc_encoded_n == 1
            and rep_stg.get("kind") == "staged" and stg_encoded_n == 1
            and any(h["file"] == "f4.txt" for h in hits_d3)
            and hits_d3 == hits_d3_re
        )
        check("R3 retire dense.npy -> views/dense.npz & lift REBUILD curse", ok_r3,
              f"no_npy={no_dense_npy} view_npz={has_view_npz} reloaded={reloaded_has_dense} "
              f"inc=({rep_inc.get('kind')},{inc_encoded_n}) stg=({rep_stg.get('kind')},{stg_encoded_n}) eq={hits_d3 == hits_d3_re}")
    finally:
        rerank.get_rerank_model = orig_get_rr

    # ---- R4: span=True (hit_span + char_start/char_end) & delta reload ----
    sp_hits = idx.search("语义 匹配", k=3, span=True)
    raw_b = (docs / "b.md").read_text(encoding="utf-8")
    b_hit = next((h for h in sp_hits if h["file"] == "b.md"), None)
    span_ok = False
    if b_hit is not None and b_hit.get("hit_span") is not None:
        s0, e0 = b_hit["hit_span"]
        cs, ce = b_hit.get("char_start", -1), b_hit.get("char_end", -1)
        span_ok = (
            "语义" in b_hit["text"][s0:e0]
            and bool(b_hit.get("hit_ranges"))
            and cs >= 0 and raw_b[cs:ce] == b_hit["text"]
        )
    # 词表内增量 delta 重载 + staged 重载 + compact 重写后验证 char_start/char_end 与 text 不丢
    (docs / "d_delta.md").write_text("检索 语义 增量 融合 匹配 重排 视图。\n", encoding="utf-8")
    rep_r4_inc = idx.add(str(docs))
    (docs / "d.md").write_text("补充 语义 段落，验证增量写入后跨度坐标持久化。\n", encoding="utf-8")
    rep_r4_stg = idx.add(str(docs))
    idx_re = primolix.open(str(idx_dir))
    dh_inc = next((h for h in idx_re.search("重排 视图", k=4, span=True) if h["file"] == "d_delta.md"), None)
    dh_stg = next((h for h in idx_re.search("跨度坐标", k=2, span=True) if h["file"] == "d.md"), None)
    raw_dd = (docs / "d_delta.md").read_text(encoding="utf-8")
    raw_d = (docs / "d.md").read_text(encoding="utf-8")
    delta_span_ok = (
        rep_r4_inc.get("kind") == "incremental"
        and dh_inc is not None
        and dh_inc.get("char_start", -1) >= 0
        and raw_dd[dh_inc["char_start"]:dh_inc["char_end"]] == dh_inc["text"]
    )
    staged_span_ok = (
        rep_r4_stg.get("kind") == "staged"
        and dh_stg is not None
        and dh_stg.get("char_start", -1) >= 0
        and raw_d[dh_stg["char_start"]:dh_stg["char_end"]] == dh_stg["text"]
    )
    idx_re.compact()
    idx_cmp = primolix.open(str(idx_dir))
    bh_cmp = next((h for h in idx_cmp.search("语义 匹配", k=4, span=True) if h["file"] == "b.md"), None)
    compact_text_ok = (
        bh_cmp is not None
        and bool(bh_cmp.get("text"))
        and bh_cmp.get("char_start", -1) >= 0
        and raw_b[bh_cmp["char_start"]:bh_cmp["char_end"]] == bh_cmp["text"]
    )
    check("R4 span=True hit_span & char_start/char_end survive delta/staged/compact reload",
          span_ok and delta_span_ok and staged_span_ok and compact_text_ok,
          f"span={span_ok} delta={delta_span_ok} staged={staged_span_ok} compact={compact_text_ok}")

    # ---- R5: tune preset="zh_short" + CLI smoke ----
    t_res = idx.tune(preset="zh_short")
    preset_ok = (t_res["after"] == {"k1": 0.6, "b": 0.75} and t_res["preset"] == "zh_short")
    idx.tune(preset="default")
    bad_preset_ok = False
    try:
        idx.tune(preset="no_such_preset")
    except ValueError:
        bad_preset_ok = True
    rc_params = cli.main(["params", "--out", str(idx_dir), "--preset", "zh_short"])
    rc_query = cli.main(["query", "检索", "--out", str(idx_dir), "--rerank", "--span", "--model", "/nonexistent"])
    check("R5 tune preset='zh_short' & CLI --rerank/--span/--preset",
          preset_ok and bad_preset_ok and rc_params == 0 and rc_query == 0,
          f"preset_ok={preset_ok} bad_raised={bad_preset_ok} cli=({rc_params},{rc_query})")

    shutil.rmtree(work, ignore_errors=True)
    print("-" * 60)
    print(f"{'ALL PASS' if not FAILS else 'FAILURES: ' + ', '.join(FAILS)}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
