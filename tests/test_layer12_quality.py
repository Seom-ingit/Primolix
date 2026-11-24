#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Layer 1 & Layer 2 底层质量优化回归测试：Markdown 短标题合并、Python AST 边界切块、span_compactness 与 z2 融合。

Run:  python tests/test_layer12_quality.py
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
from primolix.span import span_compactness         # noqa: E402

core.VERBOSE = False
kernel.VERBOSE = False

FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  |  {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


def _flat_encoder(texts):
    """给故意拉平的语义向量（近似相等），用来验证 z2 / z2_pos 是否能靠 bm25_z 与紧凑度打破平局。"""
    out = np.zeros((len(texts), 4), dtype=np.float32)
    for i, t in enumerate(texts):
        h = int(hashlib.md5(t.encode("utf-8")).hexdigest()[:4], 16) % 10
        out[i] = [1.0, 0.2, 0.1, 0.001 * h]
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)


def main() -> int:
    work = HERE / "_l12_run"
    shutil.rmtree(work, ignore_errors=True)
    atexit.register(shutil.rmtree, work, ignore_errors=True)

    # ---- L1-1: chunk_prose 保留 <20 字短标题并保持 raw[cs:ce] == text ----
    md_text = (
        "# 主标题\n\n"
        "## 并发写锁\n\n"
        "这里只写正文细节，故意不重复写前面小节标题里的两个核心名词。\n\n"
        "## 其他章节名称\n\n"
        "另一段无关说明文字，用于验证标题不会跨节乱粘连。\n"
    )
    chunks_md = core.chunk_prose(md_text, "guide.md", with_spans=True)
    ok_l1_md = (
        len(chunks_md) == 2
        and "并发写锁" in chunks_md[0][0]
        and "正文细节" in chunks_md[0][0]
        and "其他章节名称" in chunks_md[1][0]
        and all(md_text[cs:ce] == txt for txt, _r, _ln, cs, ce in chunks_md)
    )
    check("L1-1 chunk_prose retains short Markdown headings & preserves exact spans",
          ok_l1_md, f"chunks={len(chunks_md)}")

    # ---- L1-2: chunk_code 贴合 AST 边界、跳过前导空行且 raw[cs:ce] == text ----
    py_lines = [f"PAD_{i} = {i}" for i in range(1, 76)]
    py_lines += [
        "",
        "",
        "def cross_boundary_func(alpha, beta):",
        "    \"\"\"Spans across legacy 80-line cut without being severed.\"\"\"",
        "    step_one = alpha + beta",
        "    step_two = step_one * 2",
        "    step_three = step_two + 42",
        "    step_four = step_three - 1",
        "    return step_four",
    ]
    py_text = "\n".join(py_lines) + "\n"
    chunks_py = core.chunk_code(py_text, "mod.py", with_spans=True)
    fn_chunks = [c for c in chunks_py if "cross_boundary_func" in c[0]]
    ok_l1_py = (
        len(fn_chunks) == 1
        and fn_chunks[0][0].startswith("def cross_boundary_func")
        and fn_chunks[0][2] == 78
        and "return step_four" in fn_chunks[0][0]
        and all(py_text[cs:ce] == txt for txt, _r, _ln, cs, ce in chunks_py)
    )
    check("L1-2 chunk_code keeps Python functions whole at AST boundary with exact line & span",
          ok_l1_py, f"fn_chunks={len(fn_chunks)} line={fn_chunks[0][2] if fn_chunks else None}")

    # ---- L2-1: span_compactness 区分紧密共现与远距离凑数 ----
    tight = span_compactness("SparseBM25 mmap handle release avoids OSError.", ["sparsebm25", "oserror"])
    far = span_compactness(
        "SparseBM25 init starts here." + (" padding token" * 40) + " later sees OSError.",
        ["sparsebm25", "oserror"],
    )
    ok_l2_pos = (
        tight["proximity"] > far["proximity"]
        and tight["compactness"] > far["compactness"]
        and tight["compactness"] >= 0.65
        and far["compactness"] < 0.35
    )
    check("L2-1 span_compactness separates tight co-occurrence from distant spread",
          ok_l2_pos, f"tight={tight['compactness']} far={far['compactness']}")

    # ---- L2-2 & L2-3: confidence 标点过滤/半OOV保护 + Reranker(fusion='z2'/'z2_pos') ----
    repo = work / "repo"
    idx_dir = work / "idx"
    repo.mkdir(parents=True)
    (repo / "tight.md").write_text(
        "# 设计要点\n\nWindows mmap handle 紧密相邻出现，同时包含全部关键词。\n", encoding="utf-8"
    )
    (repo / "loose.md").write_text(
        "# 松散对照\n\nWindows 开头提到这里。" + (" 无关背景填充词语。" * 25) + " 末尾才提 mmap。\n",
        encoding="utf-8",
    )
    with primolix.build(str(repo), str(idx_dir), code_aware=True) as idx:
        hits_bm = idx.search("_tight.md Windows mmap", k=2, confidence=True)
        c0 = hits_bm[0]["confidence"]
        # 纯符号 `_`、`.` 不应进 matched_terms / oov
        no_punct_noise = ("_" not in c0["matched_terms"] and "_" not in c0["oov"]
                          and "span_compactness" in c0 and "proximity" in c0)

        rr_z2 = Reranker(pool_k=5, fusion="z2", encoder=_flat_encoder)
        rr_pos = Reranker(pool_k=5, fusion="z2_pos", encoder=_flat_encoder)
        hz2 = idx.search("Windows mmap", k=2, rerank=rr_z2, confidence=True)
        hpos = idx.search("Windows mmap", k=2, rerank=rr_pos, confidence=True)
        # 半 OOV 在平语义向量下仍被压在 low（cos ~ 0.99? 注意 _flat_encoder 同方向 cos~1.0 会 >=0.60, 只验证 z2_pos > tight first）
        ok_rr = (
            bool(hz2) and bool(hpos)
            and "rerank_score" in hz2[0] and "bm25_score" in hz2[0]
            and hpos[0]["file"] == "tight.md"
            and rr_pos.last_status.get("fusion") == "z2_pos"
        )
    check("L2-2/L2-3 confidence filters punctuation, exports compactness, & z2/z2_pos reranks cleanly",
          no_punct_noise and ok_rr,
          f"no_punct={no_punct_noise} top={hpos[0]['file'] if hpos else None} sc={c0.get('span_compactness')}")

    # ---- L2-4: z2_safe 严格锁死在 cos Top-32 内，绝不让 33+ 位高词频噪声跨槛反超 ----
    import numpy as np

    class _MockMeta:
        def __init__(self, n: int):
            self.rows = np.arange(n, dtype=np.int64)
            self._items = [
                {"docid": str(i), "file": f"f{i}.md", "line": 1, "text": f"doc {i}"}
                for i in range(n)
            ]

        def __len__(self) -> int:
            return len(self._items)

        def __getitem__(self, idx: int) -> dict:
            return self._items[int(idx)]

    def _mock_enc(texts):
        # q="query" -> [1, 0]; d0..d31 -> high cos [0.95..0.64, ...]; d32..d35 -> low cos [0.10, ...]
        out = []
        for t in texts:
            if t == "query":
                out.append([1.0, 0.0])
            else:
                i = int(t.split()[-1])
                cx = max(0.55, 0.95 - i * 0.01) if i < 32 else 0.10
                out.append([cx, (1.0 - cx * cx) ** 0.5])
        return np.asarray(out, dtype=np.float32)

    mock_meta = _MockMeta(36)
    pool = np.arange(36, dtype=np.int64)
    # 故意让 rank 32..35 具备极高 BM25，而 d0..d31 内靠后 doc 的 BM25 略高
    bscores = np.asarray([1.0 if i >= 32 else (0.1 + 0.02 * i) for i in range(36)], dtype=np.float32)
    rr_safe = Reranker(pool_k=36, fusion="z2_safe", encoder=_mock_enc)
    safe_hits, st_safe = rr_safe.rerank_pools(["query"], [pool], [bscores], mock_meta, k=36, with_text=False)
    safe_ids = [int(h["docid"]) for h in safe_hits[0]]
    ok_safe = (
        len(safe_ids) == 36
        and st_safe.get("fusion") == "z2_safe"
        and set(safe_ids[:32]) == set(range(32))
        and safe_ids[32:] == [32, 33, 34, 35]
        and all(safe_hits[0][j]["score"] >= safe_hits[0][j + 1]["score"] for j in range(35))
    )
    check("L2-4 z2_safe confines fusion strictly inside cos Top-32 & keeps monotonic scores",
          ok_safe, f"top32={safe_ids[:32]} tail={safe_ids[32:]}")

    # ---- L1-3 & L1-4: 续块上下文面包屑仅增强索引分词，不破坏原文切片与精确行号 ----
    bc_repo = work / "bc_repo"
    bc_idx = work / "bc_idx"
    bc_repo.mkdir(parents=True)
    long_sec_p1 = "第一段详细阐述架构边界背景知识。" * 50
    long_sec_p2 = "后续第二段只写孤立实现细节，专门讨论 phantom_tail_latch 如何收敛。" + ("补充背景。" * 10)
    md_bc = (
        "# 总架构指南\n\n"
        "## §7 幽灵锁协议\n\n"
        + long_sec_p1 + "\n\n"
        + long_sec_p2 + "\n\n"
        "```python\n# fake_heading_inside_fence\nx = 1\n```\n"
    )
    (bc_repo / "spec.md").write_text(md_bc, encoding="utf-8", newline="\n")

    # 100+ 行 Python 方法，强制切出不含 def 头的续块
    body_lines = [f"        step_{i} = val + {i}" for i in range(95)]
    py_bc = (
        "class GhostStorageEngine:\n"
        "    \"\"\"引擎容器\"\"\"\n"
        "    def commit_segmented_wal(self, val: int) -> int:\n"
        + "\n".join(body_lines)
        + "\n        sentinel_tail_checksum = val + 999\n        return sentinel_tail_checksum\n"
    )
    (bc_repo / "engine.py").write_text(py_bc, encoding="utf-8", newline="\n")

    with primolix.build(str(bc_repo), str(bc_idx), code_aware=True) as bidx:
        # 搜索 "幽灵锁协议 phantom_tail_latch" 应直击不含标题字样的第 2 个 prose 块，且 text 严格为原文切片
        h_md = bidx.search("幽灵锁协议 phantom_tail_latch", k=3, span=True)
        ok_md_bc = (
            bool(h_md)
            and h_md[0]["file"] == "spec.md"
            and "phantom_tail_latch" in h_md[0]["text"]
            and "幽灵锁协议" not in h_md[0]["text"]
            and md_bc[h_md[0]["char_start"]:h_md[0]["char_end"]] == h_md[0]["text"]
        )
        check("L1-3 Markdown continuation chunk recalls ancestor heading without altering verbatim text",
              ok_md_bc, f"top_line={h_md[0]['line'] if h_md else None}")

        # 搜索 "GhostStorageEngine commit_segmented_wal sentinel_tail_checksum" 应直击函数下半段续块
        h_py = bidx.search("GhostStorageEngine commit_segmented_wal sentinel_tail_checksum", k=3, span=True)
        ok_py_bc = (
            bool(h_py)
            and h_py[0]["file"] == "engine.py"
            and "sentinel_tail_checksum" in h_py[0]["text"]
            and "def commit_segmented_wal" not in h_py[0]["text"]
            and py_bc[h_py[0]["char_start"]:h_py[0]["char_end"]] == h_py[0]["text"]
        )
        check("L1-4 Python continuation chunk recalls enclosing class/def without altering span",
              ok_py_bc, f"top_line={h_py[0]['line'] if h_py else None}")

    shutil.rmtree(work, ignore_errors=True)
    print("-" * 60)
    print(f"{'ALL PASS' if not FAILS else 'FAILURES: ' + ', '.join(FAILS)}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
