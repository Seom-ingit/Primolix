#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""可选 INT8 紧凑向量侧车与优雅降级回归测试（纯 NumPy/stdlib 可跑，不强依赖 torch/模型）。"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import primolix                                                 # noqa: E402
from primolix import Reranker                                   # noqa: E402
from primolix.dense_int8 import quantize_int8                   # noqa: E402

FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))
    if not ok:
        FAILS.append(name)


def _semantic_stub_encoder(texts):
    """确定性语义原型：把“汽车/vehicle”概念映射同一方向，让同义表达 0 BM25 重叠也能被 INT8 捞回。"""
    vecs = []
    for t in texts:
        low = t.lower()
        if "automobile" in low or "汽车" in t or "car engine" in low:
            v = [0.96, 0.28, 0.0, 0.0]
        elif "database" in low or "btree" in low or "索引" in t:
            v = [0.0, 0.15, 0.98, 0.0]
        else:
            v = [0.1, 0.1, 0.1, 0.98]
        vecs.append(v)
    return np.asarray(vecs, dtype=np.float32)


def main() -> int:
    work = Path(tempfile.mkdtemp(prefix="pl_dense_int8_"))
    repo = work / "repo"
    idx_dir = work / "idx"
    repo.mkdir(parents=True)
    try:
        # a.txt 与查询“automobile”零词面重叠，但语义同一方向；b.txt 含无关关键词
        (repo / "vehicle_cn.md").write_text(
            "# 动力系统说明\n\n这台现代汽车装配涡轮增压缸内直喷引擎与热管理系统。\n",
            encoding="utf-8",
        )
        (repo / "db.md").write_text(
            "# 存储结构说明\n\n数据库使用 BTree 与 LSM 完成键值范围扫描。\n",
            encoding="utf-8",
        )
        (repo / "misc.md").write_text(
            "# 杂项笔记\n\n今天记录一些常规实验日志与配置说明。\n",
            encoding="utf-8",
        )

        # ---- DI-1: 默认建库/开库绝不强推 dense（零编码、无侧车） ----
        with primolix.build(str(repo), str(idx_dir), code_aware=True) as idx:
            st0 = idx.dense_int8_status()
            has_side_dir = (idx_dir / "views" / "dense_int8").exists()
            check("DI-1 default build never forces dense_int8 sidecar",
                  (not st0["available"]) and (not has_side_dir),
                  f"status={st0}")

            # ---- DI-2: 缺侧车时 dense_union=True 优雅降级为与纯 BM25 一致，不报错 ----
            bm_only = idx.search("动力", k=3)
            u_missing = idx.search("动力", k=3, dense_union=True)
            st_fb = idx.rerank_status() or {}
            check("DI-2 missing sidecar gracefully falls back to pure BM25",
                  [h["file"] for h in bm_only] == [h["file"] for h in u_missing]
                  and st_fb.get("mode") == "bm25_fallback",
                  f"mode={st_fb.get('mode')} reason={st_fb.get('reason')}")

            # ---- DI-3: INT8 量化精度 + 显式构建 + 零词面重叠语义并池召回 + z2_pool ----
            rng = np.random.default_rng(7)
            raw_v = rng.normal(size=(16, 64)).astype(np.float32)
            raw_v /= np.linalg.norm(raw_v, axis=1, keepdims=True)
            q_i8, sc_f16 = quantize_int8(raw_v)
            deq = q_i8.astype(np.float32) * sc_f16[:, None].astype(np.float32)
            mae = float(np.mean(np.abs((raw_v @ raw_v.T) - (deq @ deq.T))))
            check("DI-3a quantize_int8 preserves cosine similarity cleanly (MAE < 2e-3)",
                  q_i8.dtype == np.int8 and sc_f16.dtype == np.float16 and mae < 2e-3,
                  f"mae={mae:.6f}")

            b_res = idx.build_dense_int8(encoder=_semantic_stub_encoder)
            st1 = idx.dense_int8_status()
            check("DI-3b build_dense_int8 builds views/dense_int8 cleanly",
                  b_res.get("available") is True and b_res.get("dim") == 4 and st1.get("covered") == st1.get("total_active"),
                  f"b_res={b_res}")

            # “automobile” 在 bm25 里词表完全不在，pure BM25 命中 0 条；dense_union=True 捞回 vehicle_cn.md
            bm_miss = idx.search("automobile", k=3)
            rr_stub = Reranker(pool_k=10, fusion="z2_safe", encoder=_semantic_stub_encoder)
            u_hits = idx.search("automobile", k=3, rerank=rr_stub, dense_union=True, dense_k=5, confidence=True)
            st_u = idx.rerank_status() or {}
            check("DI-3c dense_union rescues zero-BM25-overlap doc via INT8 semantic sidecar",
                  len(bm_miss) == 0 and bool(u_hits) and u_hits[0]["file"] == "vehicle_cn.md"
                  and st_u.get("mode") == "dense_int8_union" and "confidence" in u_hits[0],
                  f"bm_len={len(bm_miss)} top={u_hits[0]['file'] if u_hits else None} mode={st_u.get('mode')}")

            # ---- DI-4: 增量加新文档后未入侧车行仍可用，侧车缺损/模型缺失继续优雅降级 ----
            (repo / "new_electric.md").write_text(
                "# 纯电驱说明\n\n这台新车使用碳化硅电驱逆变器。\n", encoding="utf-8"
            )
            idx.add(str(repo), verify="full")
            st_lag = idx.dense_int8_status()
            u_lag = idx.search("碳化硅", k=3, rerank=rr_stub, dense_union=True)
            st_ulag = idx.rerank_status() or {}
            check("DI-4 incremental uncovered doc is safely encoded on demand without crashing",
                  st_lag.get("total_active") == st_lag.get("covered") + 1
                  and any(h["file"] == "new_electric.md" for h in u_lag)
                  and st_ulag.get("uncovered_encoded", 0) >= 1,
                  f"covered={st_lag.get('covered')}/{st_lag.get('total_active')} uncov={st_ulag.get('uncovered_encoded')}")

            # 故意损坏 meta.json → 优雅退回 BM25，不抛异常
            (idx_dir / "views" / "dense_int8" / "meta.json").write_text("{corrupt", encoding="utf-8")
            idx._z._dense_int8_sidecar().clear()
            u_corrupt = idx.search("动力", k=3, dense_union=True)
            st_corrupt = idx.rerank_status() or {}
            check("DI-5 corrupt dense_int8 meta gracefully degrades to BM25",
                  [h["file"] for h in bm_only] == [h["file"] for h in u_corrupt]
                  and st_corrupt.get("mode") == "bm25_fallback",
                  f"mode={st_corrupt.get('mode')} reason={st_corrupt.get('reason')}")

            # 缺模型时 build_dense_int8 诚实降级不抛错
            b_no_model = idx.build_dense_int8(model_path=str(work / "missing_model_dir"))
            check("DI-6 build_dense_int8 degrades gracefully when model path is invalid",
                  b_no_model.get("available") is False and b_no_model.get("mode") == "bm25_fallback",
                  f"res={b_no_model}")

        # Windows rmtree 验证无文件锁残留
        shutil.rmtree(work)
        check("DI-7 Windows rmtree after dense_int8 sidecar has no file-lock errors", not work.exists())
        print("-" * 60)
        print("ALL PASS" if not FAILS else "FAILURES: " + ", ".join(FAILS))
        return 1 if FAILS else 0
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
