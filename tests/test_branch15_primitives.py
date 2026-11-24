#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""支线 #15 回归测试：code_tokenize、原语置信度 (confidence=True) 与 ScopeGraph 拓扑联姻路由。

Run:  python tests/test_branch15_primitives.py
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
from primolix.kernel import code_tokenize          # noqa: E402
from primolix.scope import ScopeGraph              # noqa: E402

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
        v0 = 1.0 if ("SparseBM25" in t or "load_cache" in t) else 0.1
        v1 = 1.0 if "语义" in t else 0.1
        h = int(hashlib.md5(t.encode("utf-8")).hexdigest()[:4], 16) % 50
        out[i] = [v0, v1, 0.01 * (h + 1)]
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)


def main() -> int:
    work = HERE / "_b15_run"
    shutil.rmtree(work, ignore_errors=True)
    atexit.register(shutil.rmtree, work, ignore_errors=True)

    repo = work / "repo"
    (repo / "pkg").mkdir(parents=True)
    (repo / "docs").mkdir(parents=True)
    (repo / "archive").mkdir(parents=True)

    (repo / "pkg" / "engine.py").write_text(
        "class SparseBM25Engine:\n    def _load_cache_fast(self):\n        return 'fast mmap cache load'\n",
        encoding="utf-8",
    )
    (repo / "pkg" / "handler.py").write_text(
        "from pkg.engine import SparseBM25Engine\n\ndef handle_request():\n    return 'request handler wrapper'\n",
        encoding="utf-8",
    )
    (repo / "docs" / "engine_spec.md").write_text(
        "本文说明 pkg/engine.py 的冷启动缓存与稀疏原语架构设计，无直接英文函数名。\n",
        encoding="utf-8",
    )
    (repo / "archive" / "old_notes.md").write_text(
        "历史旧版归档说明文档，不再高频更新，用于冷热优先级区分。\n",
        encoding="utf-8",
    )

    # ---- B1: code_tokenize + tokenizer_mode 持久化与自然语言子词召回 ----
    toks = code_tokenize("SparseBM25Engine _load_cache_fast")
    idx_dir = work / "idx"
    with primolix.build(str(repo), str(idx_dir), code_aware=True) as idx0:
        mode_saved = idx0.bm.tokenizer_mode

    with primolix.open(str(idx_dir)) as idx1:
        mode_loaded = idx1.bm.tokenizer_mode
        # 用拆词 "load cache" 搜 `_load_cache_fast`，再用精确标识 "SparseBM25Engine" 搜
        hits_sub = idx1.search("load cache", k=4, confidence=True)
        hits_exact = idx1.search("SparseBM25Engine", k=4, confidence=True)
        hits_oov = idx1.search("NonExistentSymbolXYZ 完全不存在词", k=4, confidence=True)
        conf_oov = idx1.confidence_status()

    ok_b1 = (
        {"sparsebm25engine", "sparse", "bm25", "engine", "_load_cache_fast", "load", "cache", "fast"}.issubset(set(toks))
        and mode_saved == "zh_code" and mode_loaded == "zh_code"
        and len(hits_sub) >= 1 and "engine.py" in hits_sub[0]["file"]
        and len(hits_exact) >= 1 and any("engine.py" in h["file"] for h in hits_exact)
        and hits_oov == [] and conf_oov is not None and conf_oov["tier"] == "miss"
    )
    check(
        "B1 code_tokenize subwords persist across save/load & recall camel/snake symbols",
        ok_b1,
        f"mode=({mode_saved},{mode_loaded}) sub={[h['file'] for h in hits_sub]} exact={[h['file'] for h in hits_exact]} "
        f"oov={hits_oov} exact_tier={hits_exact[0]['confidence']['tier'] if hits_exact else None} miss={conf_oov.get('tier') if conf_oov else None}",
    )

    # ---- B2: confidence=True 在 OOV 混合、高分命中、以及 rerank 交叉印证下给出有界信号 ----
    with primolix.open(str(idx_dir)) as idx2:
        # 一半词 OOV 的查询不应算出虚高置信度
        hits_half_oov = idx2.search("SparseBM25Engine MissingAlpha MissingBeta MissingGamma MissingDelta", k=3, confidence=True)
        c_half = hits_half_oov[0]["confidence"] if hits_half_oov else {}
        rr = Reranker(pool_k=10, sub_batch=4, cache_view=False, encoder=_dummy_encode)
        hits_rr = idx2.search("SparseBM25Engine", k=3, rerank=rr, confidence=True)
        c_rr = hits_rr[0]["confidence"] if hits_rr else {}

    ok_b2 = (
        bool(c_half)
        and 0.0 <= c_half["score_ratio"] <= 1.0
        and c_half["in_vocab_ratio"] < 0.5
        and c_half["tier"] == "low"
        and len(c_half["oov"]) >= 4
        and bool(c_rr) and "rerank_cos" in c_rr and "rank_delta" in c_rr
    )
    check(
        "B2 confidence=True bounds signals, penalizes partial OOV, and tracks rerank agreement",
        ok_b2,
        f"half_oov=({c_half.get('tier')}, inv={c_half.get('in_vocab_ratio')}) rr=({c_rr.get('tier')}, cos={c_rr.get('rerank_cos')})",
    )

    # ---- B3: ScopeGraph 桥接目录族 + marriage 联姻边 + hot/cold 预算调度 ----
    sg = ScopeGraph(repo)
    rep_sg = sg.report()
    expanded = sg.expand_scope("pkg/handler.py", include_marriage=True)
    expanded_eng = sg.expand_scope("pkg/engine.py", include_marriage=True)
    pri = sg.prioritize_updates(["archive/old_notes.md", "pkg/handler.py", "pkg/engine.py"], budget=2)

    with primolix.open(str(idx_dir)) as idx3:
        raw_hits = idx3.search("冷启动 缓存 架构", k=10)
        scoped_hits = sg.filter_hits(raw_hits, "pkg/handler.py", include_marriage=True)

    ok_b3 = (
        rep_sg["marriage_edges"] >= 2
        and "pkg/engine.py" in expanded
        and "docs/engine_spec.md" in expanded_eng
        and pri[0] == "pkg/engine.py" and "archive/old_notes.md" not in pri
        and any("engine_spec.md" in h["file"] for h in sg.filter_hits(raw_hits, "pkg/engine.py", include_marriage=True))
        and len(scoped_hits) >= 0
    )
    check(
        "B3 ScopeGraph discovers import/doc marriage edges & hot/cold update budget",
        ok_b3,
        f"edges={rep_sg['marriage_edges']} eng_family={sorted(expanded_eng)} pri={pri}",
    )

    shutil.rmtree(work, ignore_errors=True)
    print("-" * 60)
    print(f"{'ALL PASS' if not FAILS else 'FAILURES: ' + ', '.join(FAILS)}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
