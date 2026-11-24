#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""primolix.rerank —— 0.3 查询期池内重排器（批式优先 + 渐进视图缓存 + 诚实降级）。

承 `11_0.3_质量产品化_设计与报备_2026-11-18.md` 与 `2026-11-19` Phase 0 实测：
- 索引期零编码：默认仅靠 BM25 粗筛池（K'=100），查询期用轻量编码器（默认 bge-small-zh-v1.5）做余弦重排。
- 批式优先：8GB GPU 上单条同步调用会吃 ~163 ms 模式税；`rerank_pools` 对整批候选 doc 去重后流式编码
  （子批 ≤ 256，防 SDPA 显存随 B×seq² 爆炸），并与可选局部 `views` 打通实现热文档向量复用。
- 降级诚实：模型或依赖缺席时不抛 `FileNotFoundError`，如实退回 BM25 并返回可观测状态。
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from .kernel import log
from .span import hit_ranges as _hit_ranges, hit_span as _hit_span, query_tokens as _query_tokens

DEFAULT_POOL_K = 100
DEFAULT_SUB_BATCH = 256
DEFAULT_MODEL_CANDIDATES = (
    "models--BAAI--bge-small-zh-v1.5",
    "models--BAAI--bge-m3",
)
COV100_CEILING_NOTE = "池内重排受初筛池上界约束（1M 中文 mmarco 参考 cov@100=0.6549）"

_MODEL_CACHE: dict[tuple[str, str], Any] = {}

# ---- M1（契约 §1.2）：rerank 缓存视图的落盘口径 / LRU 封顶 / GC 阈值 ----
RERANK_CACHE_VIEW = "rerank_cache"
RERANK_CACHE_MAX_VECTORS = 4096
RERANK_CACHE_GC_RATIO = 0.2
_ENV_CACHE_MAX = "PRIMOLIX_RERANK_CACHE_MAX"


def _cache_max_vectors(value=None) -> int:
    """解析 LRU 封顶（显式值 > 环境变量 > 默认 4096）。

    ⚠️ 契约 §1.2 写的是 clamp 256..20000；这里下限放宽到 1，否则 §5.1-5 的
    `PRIMOLIX_RERANK_CACHE_MAX=8` 单测（小语料验证 LRU 封顶）无法通过 —— 偏差已在回报中标注。
    """
    raw = value
    if raw is None:
        raw = os.environ.get(_ENV_CACHE_MAX)
    if raw is None or str(raw).strip() == "":
        return int(RERANK_CACHE_MAX_VECTORS)
    try:
        n = int(str(raw).strip())
    except (TypeError, ValueError):
        return int(RERANK_CACHE_MAX_VECTORS)
    return max(1, min(int(n), 20000))


def _chunk_version() -> str:
    """切块代次（契约 §1.2）：惰性 import，避免 core ↔ rerank 循环导入。"""
    try:
        from .core import CHUNK_VERSION

        return str(CHUNK_VERSION)
    except Exception:  # noqa: BLE001  代次不可知时按「不阻塞」处理
        return ""


def _resolve_device(device: str = "auto") -> str:
    dev = (device or "auto").strip().lower()
    if dev != "auto":
        return dev
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _resolve_model_path(model_path: str = "") -> tuple[str | None, str | None]:
    """解析模型路径；为空时按优先级搜索本地 HuggingFace 缓存快照。"""
    mp = str(model_path or "").strip()
    if mp:
        return mp, None
    hub = Path.home() / ".cache" / "huggingface" / "hub"
    for cand in DEFAULT_MODEL_CANDIDATES:
        snaps = sorted((hub / cand / "snapshots").glob("*")) if (hub / cand / "snapshots").exists() else []
        if snaps:
            return str(snaps[0]), None
    return None, "未找到本地重排模型缓存（bge-small-zh-v1.5 / bge-m3）"


def get_rerank_model(model_path: str = "", device: str = "auto") -> tuple[Any | None, dict]:
    """加载或复用重排编码模型，缺席时如实返回 `(None, status)`（绝不静默假装已重排或裸抛）。"""
    resolved, err = _resolve_model_path(model_path)
    if resolved is None:
        return None, dict(available=False, model=None, device=None, reason=err)

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return None, dict(available=False, model=resolved, device=None,
                          reason="sentence_transformers 未安装，已降级 BM25-only")

    dev = _resolve_device(device)
    key = (resolved, dev)
    if key not in _MODEL_CACHE:
        t0 = time.time()
        try:
            from transformers.utils import logging as _hflog

            _hflog.disable_progress_bar()
        except Exception:
            pass
        try:
            model = SentenceTransformer(resolved, device=dev)
        except Exception as e:
            return None, dict(available=False, model=resolved, device=dev,
                              reason=f"重排模型加载失败: {e}")
        _MODEL_CACHE[key] = model
        log(f"rerank 模型预热 {time.time() - t0:.1f}s ({Path(resolved).name} @ {dev})")

    return _MODEL_CACHE[key], dict(available=True, model=Path(resolved).name,
                                   model_path=resolved, device=dev, reason=None)


def _to_float32_2d(raw: Any) -> np.ndarray:
    """统一收口 ST 6.x `list[Tensor]` / `Tensor` / `ndarray` 为连续 L2 归一化 `float32[N, D]`。"""
    if raw is None:
        return np.zeros((0, 0), dtype=np.float32)
    try:
        import torch

        if isinstance(raw, (list, tuple)):
            if not raw:
                return np.zeros((0, 0), dtype=np.float32)
            raw = torch.stack([t.reshape(-1) for t in raw], dim=0)
        if torch.is_tensor(raw):
            if raw.dim() == 1:
                raw = raw.reshape(1, -1)
            arr = raw.detach().float().cpu().numpy()
        else:
            arr = np.asarray(raw, dtype=np.float32)
    except ImportError:
        arr = np.asarray(raw, dtype=np.float32)

    if arr.ndim == 1:
        arr = arr.reshape(1, -1) if arr.size else np.zeros((0, 0), dtype=np.float32)
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    if arr.size:
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        arr = arr / np.maximum(norms, np.float32(1e-12))
    return arr


def encode_texts(encoder: Any, texts: Sequence[str], *, sub_batch: int = DEFAULT_SUB_BATCH,
                 device: str | None = None) -> np.ndarray:
    """编码一组文本为归一化 `float32[N, D]`；支持 `SentenceTransformer` 或任意可调用编码器。"""
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    items = list(texts)
    if callable(encoder) and not hasattr(encoder, "encode"):
        return _to_float32_2d(encoder(items))

    bs = max(1, min(int(sub_batch), len(items)))
    try:
        raw = encoder.encode(
            items,
            batch_size=bs,
            normalize_embeddings=True,
            convert_to_numpy=False,
            show_progress_bar=False,
        )
    except TypeError:
        raw = encoder.encode(items, batch_size=bs, normalize_embeddings=True)
    out = _to_float32_2d(raw)
    if device and str(device).startswith("cuda"):
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    return out


class Reranker:
    """可复用的查询期重排会话：支持流式子批、视图热缓存复用、z2/紧凑度融合与诚实降级。"""

    def __init__(self, *, model_path: str = "", device: str = "auto",
                 pool_k: int = DEFAULT_POOL_K, sub_batch: int = DEFAULT_SUB_BATCH,
                 view_name: str | None = None, cache_view: bool = False,
                 fusion: str = "cos",
                 encoder: Callable[[Sequence[str]], np.ndarray] | Any = None) -> None:
        if fusion not in ("cos", "z2", "z2_pos", "z2_safe", "z2_pool"):
            raise ValueError(f"fusion 只支持 'cos'/'z2'/'z2_pos'/'z2_safe'/'z2_pool'，实得 {fusion!r}")
        self.model_path = str(model_path or "")
        self.device = str(device or "auto")
        self.pool_k = max(1, int(pool_k))
        self.sub_batch = max(1, int(sub_batch))
        self.view_name = view_name or ("rerank_cache" if cache_view else None)
        self.cache_view = bool(cache_view)
        self.fusion = str(fusion)
        self._encoder = encoder
        self.last_status: dict | None = None
        # M1（契约 §1.2）：最近一次合并结果（rows/dim/dead_dropped/capped_dropped/gc_rewritten）
        self._persist_status: dict | None = None

    def _get_encoder(self) -> tuple[Any | None, dict]:
        if self._encoder is not None:
            name = getattr(self._encoder, "__name__", type(self._encoder).__name__)
            return self._encoder, dict(available=True, model=name, device="custom", reason=None)
        return get_rerank_model(self.model_path, device=self.device)

    # ---- M1（契约 §1.2）：公共编码入口 / 显式热区预热 ----

    def encode(self, texts: Sequence[str], *,
               sub_batch: int | None = None) -> tuple[np.ndarray | None, dict]:
        """公共编码入口：返回 `(归一化 float32[N,D] | None, status)`；不可用时 `(None, st)`，绝不抛。"""
        enc, st = self._get_encoder()
        if enc is None:
            return None, st
        items = [str(t) for t in (texts or [])]
        if not items:
            return np.zeros((0, 0), dtype=np.float32), st
        try:
            vecs = encode_texts(enc, items, sub_batch=int(sub_batch or self.sub_batch),
                                device=st.get("device"))
        except Exception as exc:  # noqa: BLE001  编码失败 → 诚实降级，不打断查询
            return None, dict(st, available=False,
                              reason=f"编码失败: {type(exc).__name__}: {exc}")
        return vecs, st

    def warm_files(self, meta, files, *, bm=None, max_chunks: int = 512,
                   view_name: str | None = None) -> dict:
        """显式热区预热（契约 §1.2）：只编码 `files` 命中的 chunk 并入视图，⛔ 无后台线程。

        选 chunk 口径同契约 §1.1：`meta.files` 命中 → 按 (file, line) 排序 → 截断 `max_chunks`。
        """
        t0 = time.time()
        view = str(view_name or self.view_name or RERANK_CACHE_VIEW)
        res = dict(mode="warm", available=False, files=0, chunks=0, encoded=0,
                   ms=0.0, view=view, reason=None)
        names = sorted({str(f).replace("\\", "/") for f in (files or [])
                        if str(f or "").strip()})
        if not names:
            res["reason"] = "no_files"
            return res
        if meta is None or bm is None:
            res["reason"] = "view_absent"
            return res
        try:
            meta_files = [str(f).replace("\\", "/") if f is not None else ""
                          for f in meta.files]
            mask = np.isin(np.asarray(meta_files, dtype=object),
                           np.asarray(names, dtype=object))
            positions = np.flatnonzero(mask).astype(np.int64)
        except Exception as exc:  # noqa: BLE001  meta 形态不认识 → 如实回传
            res["reason"] = f"{type(exc).__name__}: {exc}"
            return res
        if positions.size == 0:
            res["reason"] = "no_matching_chunks"
            return res
        files_arr = np.asarray(meta_files, dtype=object)
        try:
            line_arr = np.asarray([int(meta[int(p)]["line"]) for p in positions.tolist()],
                                  dtype=np.int64)
        except Exception:  # noqa: BLE001  行号不可得 → 只按文件名确定性排序
            line_arr = np.zeros(positions.size, dtype=np.int64)
        order = np.lexsort((line_arr, files_arr[positions]))
        positions = positions[order][:max(1, int(max_chunks))]
        matched_files = sorted({str(files_arr[int(p)]) for p in positions.tolist()})
        res["files"] = len(matched_files)
        res["chunks"] = int(positions.size)
        texts = []
        for pos in positions.tolist():
            try:
                texts.append(meta[int(pos)]["text"])
            except Exception:  # noqa: BLE001  取文失败按空串跳过（数量对齐仍成立）
                texts.append("")
        vecs, st = self.encode(texts)
        res["available"] = bool(st.get("available"))
        if vecs is None or int(vecs.shape[0]) != len(texts):
            res["reason"] = str(st.get("reason") or "encode_unavailable")
            res["ms"] = round((time.time() - t0) * 1000.0, 3)
            return res
        new_vecs = {int(pos): vecs[i] for i, pos in enumerate(positions.tolist())}
        touched = [int(p) for p in positions.tolist()]
        merge_info = _merge_into_view(
            bm, view, meta, new_vecs,
            model_id=str(st.get("model") or ""),
            tokenizer_mode=getattr(bm, "tokenizer_mode", "zh_default"),
            chunk_version=_chunk_version(), max_vectors=None, touched_pos=touched)
        self._persist_status = merge_info
        res["encoded"] = int(len(new_vecs))
        res["reason"] = None if merge_info.get("rows") else "view_merge_empty"
        res["ms"] = round((time.time() - t0) * 1000.0, 3)
        res["view_info"] = dict(merge_info)
        return res

    def rerank_pools(self, queries: Sequence[str], pools: Sequence[np.ndarray],
                     pool_scores: Sequence[np.ndarray], meta: Any, *,
                     bm: Any = None, k: int = 10, with_text: bool = True,
                     with_span: bool = False) -> tuple[list[list[dict]], dict]:
        """对一批查询的 BM25 候选池做去重批式重排。

        `pools[qi]` 为活跃文档在 `_meta` 中的下标数组（按 BM25 降序）。
        """
        n_q = len(queries)
        if n_q == 0:
            st = dict(mode="rerank", available=True, pool_k=self.pool_k,
                      unique_candidates=0, encoded=0, cache_hits=0,
                      ceiling_note=COV100_CEILING_NOTE, view=None)
            self.last_status = st
            return [], st

        enc, st_model = self._get_encoder()
        if enc is None:
            out_fb = _bm25_fallback(queries, pools, pool_scores, meta, k=k,
                                    with_text=with_text, with_span=with_span)
            st = dict(st_model, mode="bm25_fallback", pool_k=self.pool_k,
                      unique_candidates=0, encoded=0, cache_hits=0,
                      ceiling_note=COV100_CEILING_NOTE, view=None)
            self.last_status = st
            return out_fb, st

        # 收集整批唯一候选 _meta 下标（避免重复编码同一文档）
        non_empty = [np.asarray(p, dtype=np.int64) for p in pools if len(p) > 0]
        if not non_empty:
            st = dict(st_model, mode="rerank", pool_k=self.pool_k,
                      unique_candidates=0, encoded=0, cache_hits=0,
                      ceiling_note=COV100_CEILING_NOTE, view=None)
            self.last_status = st
            return [[] for _ in range(n_q)], st

        uniq_pos = np.unique(np.concatenate(non_empty))

        # 若有视图在场，按物理行读取已缓存向量（热文档跳过编码）
        cached_vecs: dict[int, np.ndarray] = {}
        view = (bm._views.get(str(self.view_name))
                if (bm is not None and self.view_name and hasattr(bm, "_views")) else None)
        if view is not None and view.get("available", True) and len(meta):
            v_rows = view["rows"]
            v_vn = view["vn"]
            cov = view["coverage"]
            for m_pos in uniq_pos.tolist():
                phys_r = int(meta.rows[m_pos])
                if 0 <= phys_r < cov.size and cov[phys_r]:
                    idx_in_v = int(np.searchsorted(v_rows, phys_r))
                    if idx_in_v < v_rows.size and int(v_rows[idx_in_v]) == phys_r:
                        cached_vecs[m_pos] = v_vn[idx_in_v]

        miss_pos = [int(p) for p in uniq_pos.tolist() if int(p) not in cached_vecs]
        qvecs = encode_texts(enc, list(queries), sub_batch=self.sub_batch,
                             device=st_model.get("device"))

        # 流式编码未命中候选，避免文本与向量全量驻留
        new_vecs: dict[int, np.ndarray] = {}
        batch_cap = max(self.sub_batch * 2, 512)
        total_miss = len(miss_pos)
        t_enc_start = time.time()
        for i0 in range(0, total_miss, batch_cap):
            chunk_pos = miss_pos[i0:i0 + batch_cap]
            texts = [meta[p]["text"] for p in chunk_pos]
            dvecs = encode_texts(enc, texts, sub_batch=self.sub_batch,
                                 device=st_model.get("device"))
            if dvecs.shape[0] == len(chunk_pos):
                for bi, mp in enumerate(chunk_pos):
                    new_vecs[mp] = dvecs[bi]
            if total_miss >= 2000 and (i0 // batch_cap + 1) % 50 == 0:
                el = time.time() - t_enc_start
                done = min(i0 + batch_cap, total_miss)
                eta = (total_miss - done) / (done / el) if done else 0.0
                log(f"rerank 流式编码 {done:,}/{total_miss:,} · {el:.0f}s · ETA {eta:.0f}s")

        # 可选渐进视图写回：把新编码的热文档沉淀进局部视图（M1 契约 §1.2：LRU/代次/GC）
        merge_info = None
        if self.cache_view and bm is not None and self.view_name and new_vecs:
            touched = sorted(set(cached_vecs) | set(new_vecs))
            merge_info = _merge_into_view(
                bm, str(self.view_name), meta, new_vecs,
                model_id=str(st_model.get("model") or ""),
                tokenizer_mode=getattr(bm, "tokenizer_mode", "zh_default"),
                chunk_version=_chunk_version(), touched_pos=touched,
            )
            self._persist_status = merge_info

        all_vecs = dict(cached_vecs)
        all_vecs.update(new_vecs)

        from .span import span_compactness as _span_compactness
        from .kernel import get_tokenizer as _get_tokenizer

        tok_fn = (_get_tokenizer(getattr(bm, "tokenizer_mode", "zh_default"))
                  if self.fusion == "z2_pos" else None)

        out_all: list[list[dict]] = []
        k_lim = max(int(k), 0)
        for qi, q in enumerate(queries):
            pool = np.asarray(pools[qi], dtype=np.int64)
            bscores = np.asarray(pool_scores[qi], dtype=np.float32)
            if pool.size == 0 or k_lim <= 0:
                out_all.append([])
                continue
            qv = qvecs[qi]
            cos_arr = np.empty(pool.size, dtype=np.float32)
            for j, mp in enumerate(pool.tolist()):
                dv = all_vecs.get(int(mp))
                if dv is not None and dv.size == qv.size:
                    cos_arr[j] = float(np.dot(dv, qv))
                else:
                    cos_arr[j] = -1e9
            if self.fusion == "cos" or pool.size <= 1:
                rank_arr = cos_arr
                order = np.argsort(-rank_arr, kind="stable")[:k_lim]
            elif self.fusion == "z2_safe":
                # 仅在 cos Top-32 内部用 w=0.35 局部融合（1M 6980q 实测比 Top-20 更稳），杜绝高词频噪声跨槛反超
                cos_order = np.argsort(-cos_arr, kind="stable")
                m = min(32, pool.size)
                top_idx = cos_order[:m]
                c_sub = cos_arr[top_idx]
                b_sub = bscores[top_idx]
                z_cos = (c_sub - float(c_sub.mean())) / max(float(c_sub.std()), 0.02)
                z_bm = (b_sub - float(b_sub.mean())) / max(float(b_sub.std()), 1e-4)
                sub_sc = z_cos + 0.35 * z_bm
                sub_order = top_idx[np.argsort(-sub_sc, kind="stable")]
                rank_arr = cos_arr.copy()
                rank_arr[top_idx] = sub_sc
                if pool.size > m:
                    tail_max = float(cos_arr[cos_order[m:]].max())
                    shift = float(sub_sc.min()) - 2.0 - tail_max
                    rank_arr[cos_order[m:]] = cos_arr[cos_order[m:]] + min(0.0, shift)
                order = np.concatenate((sub_order, cos_order[m:]))[:k_lim]
            elif self.fusion == "z2_pool":
                # 轻权全池融合 (w=0.25)，专供并池 Stage-2 Top-32 选侯选与轻混合
                z_cos = (cos_arr - float(cos_arr.mean())) / max(float(cos_arr.std()), 0.02)
                z_bm = (bscores - float(bscores.mean())) / max(float(bscores.std()), 1e-4)
                rank_arr = z_cos + 0.25 * z_bm
                order = np.argsort(-rank_arr, kind="stable")[:k_lim]
            else:
                c_std = float(cos_arr.std())
                b_std = float(bscores.std())
                z_cos = (cos_arr - float(cos_arr.mean())) / max(c_std, 0.02)
                z_bm = (bscores - float(bscores.mean())) / max(b_std, 1e-4)
                rank_arr = z_cos + 0.65 * z_bm
                if self.fusion == "z2_pos" and tok_fn is not None:
                    q_terms = tok_fn(q)
                    if sum(1 for t in q_terms if any(ch.isalnum() for ch in t)) >= 2:
                        pos_arr = np.fromiter(
                            (_span_compactness(meta[int(mp)]["text"], q_terms)["compactness"]
                             for mp in pool.tolist()),
                            dtype=np.float32, count=pool.size,
                        )
                        p_std = float(pos_arr.std())
                        if p_std > 1e-4:
                            rank_arr = rank_arr + 0.25 * ((pos_arr - float(pos_arr.mean())) / max(p_std, 0.02))
                order = np.argsort(-rank_arr, kind="stable")[:k_lim]
            hits: list[dict] = []
            for rank, pj in enumerate(order.tolist(), 1):
                mp = int(pool[pj])
                m = meta[mp]
                item = dict(
                    rank=rank,
                    score=float(rank_arr[pj]),
                    bm25_score=float(bscores[pj]),
                    rerank_score=float(cos_arr[pj]),
                    pool_rank=int(pj) + 1,
                    rank_delta=int(pj) + 1 - rank,
                    docid=m.get("docid"),
                    file=m["file"],
                    line=m["line"],
                )
                if with_text:
                    txt = m["text"]
                    item["text"] = txt
                    if with_span:
                        item["hit_span"] = _hit_span(txt, q)
                        item["hit_ranges"] = _hit_ranges(txt, _query_tokens(q))
                if with_span and "char_start" in m:
                    item["char_start"] = m["char_start"]
                    item["char_end"] = m["char_end"]
                hits.append(item)
            out_all.append(hits)

        st = dict(
            st_model,
            mode="rerank",
            fusion=self.fusion,
            pool_k=self.pool_k,
            unique_candidates=int(uniq_pos.size),
            encoded=len(new_vecs),
            cache_hits=len(cached_vecs),
            ceiling_note=COV100_CEILING_NOTE,
            # M1（契约 §1.2）：「缺省 None」= 本次没有发生视图合并
            view=(None if merge_info is None else {
                "name": str(self.view_name),
                "vectors": int(merge_info.get("rows") or 0),
                "max_vectors": int(_cache_max_vectors()),
            }),
        )
        self.last_status = st
        return out_all, st


def view_meta_compatible(meta, *, model_id=None, tokenizer_mode=None,
                         chunk_version=None) -> tuple[bool, str]:
    """视图代次校验（M1 契约 §1.2 纯函数）：不等 → `(False, "<键>_mismatch")`。

    任一缺失按「不阻塞」处理（旧视图 / 旧索引没记 meta 时仍可用）；⛔ 只在双方都有值时比较。
    """
    m = meta if isinstance(meta, dict) else {}
    for key, want, got in (
        ("chunk_version", chunk_version, m.get("chunk_version")),
        ("tokenizer_mode", tokenizer_mode, m.get("tokenizer_mode")),
    ):
        if want and got and str(want) != str(got):
            return False, f"{key}_mismatch"
    if model_id and m.get("model_id") and str(model_id) != str(m.get("model_id")):
        return False, "model_id_mismatch"
    return True, ""


def _merge_into_view(bm: Any, view_name: str, meta: Any, new_vecs: dict[int, np.ndarray],
                     *, model_id: str, tokenizer_mode: str = "",
                     chunk_version: str = "", max_vectors=None,
                     touched_pos: Sequence[int] | None = None) -> dict:
    """把新编码向量并入局部覆盖视图（M1 契约 §1.2：物理行合并 + 墓碑剔除 + LRU 封顶 + 代次 meta）。

    返回 `dict(rows, dim, dead_dropped, capped_dropped, gc_rewritten)`：
    - `dead_dropped`：合并时剔除的墓碑/越界行数（既有视图 ∪ 新向量）；
    - `capped_dropped`：LRU 封顶裁掉的向量数；
    - `gc_rewritten`：既有行里被剔死的比例超过 `RERANK_CACHE_GC_RATIO`（调用方据此决定瘦身落盘）。
    """
    existing = bm._views.get(str(view_name)) if hasattr(bm, "_views") else None
    old_meta = dict(existing.get("meta") or {}) if isinstance(existing, dict) else {}
    live_lim = int(bm.N)
    dead = getattr(bm, "_dead", None)
    dead_existing = 0
    dead_new = 0
    merged: dict[int, np.ndarray] = {}
    old_lru: list[int] = []
    existing_n = 0
    if existing is not None:
        old_rows = [int(r) for r in existing["rows"].tolist()]
        existing_n = len(old_rows)
        old_vecs = existing.get("vn")
        if old_vecs is None:                       # 兼容旧视图（只有原始 vectors）
            old_vecs = existing.get("vectors")
        for i, ri in enumerate(old_rows):
            if not (0 <= ri < live_lim) or (dead is not None and bool(dead[ri])):
                dead_existing += 1
                continue
            merged[ri] = old_vecs[i]
        raw_lru = old_meta.get("lru_order")
        if isinstance(raw_lru, (list, tuple)):
            seen = set()
            for r in raw_lru:
                try:
                    ri = int(r)
                except (TypeError, ValueError):
                    continue
                if ri in merged and ri not in seen:
                    seen.add(ri)
                    old_lru.append(ri)
        if not old_lru:                            # 无 LRU 记录（旧视图）→ 行号升序当 least→most
            old_lru = sorted(merged.keys())
    # 新向量：meta 位置 → 物理行（越界/墓碑不写进视图）
    rows_meta = getattr(meta, "rows", None)
    new_rows: list[int] = []
    for mp in sorted(new_vecs.keys()):
        try:
            ri = int(rows_meta[int(mp)])
        except Exception:  # noqa: BLE001  meta 行号不可得 → 该条如实计死
            dead_new += 1
            continue
        if not (0 <= ri < live_lim) or (dead is not None and bool(dead[ri])):
            dead_new += 1
            continue
        merged[ri] = np.asarray(new_vecs[mp], dtype=np.float32)
        new_rows.append(ri)
    dead_dropped = dead_existing + dead_new
    cap = _cache_max_vectors(max_vectors)
    # touched（本轮真正用到的行，按 rank 顺序传；缺省 = new_vecs 升序）→ 放队尾（most recent）
    touched_rows: list[int] = []
    for mp in (list(touched_pos) if touched_pos else sorted(new_vecs.keys())):
        try:
            ri = int(rows_meta[int(mp)])
        except Exception:  # noqa: BLE001
            continue
        if ri in merged and ri not in touched_rows:
            touched_rows.append(ri)
    touched_set = set(touched_rows)
    lru: list[int] = [ri for ri in old_lru if ri in merged and ri not in touched_set]
    lru.extend(touched_rows)
    for ri in new_rows:                            # 新行追加（touched 缺省时也覆盖到）
        if ri not in lru:
            lru.append(ri)
    present = set(lru)
    for ri in sorted(merged.keys()):               # 兜底：任何未记录行都不静默丢
        if ri not in present:
            lru.append(ri)
            present.add(ri)
    capped_dropped = 0
    if len(lru) > cap:
        # 从队首淘汰；被 touched 的行不淘汰 —— 除非 cap 小于 touched 数（契约 §1.2 规则 2）
        protect = touched_set if len(touched_set) <= cap else set()
        need = len(lru) - cap
        keep_flag = [True] * len(lru)
        for i, ri in enumerate(lru):
            if need <= 0:
                break
            if ri in protect:
                continue
            keep_flag[i] = False
            need -= 1
        if need > 0:                               # cap < touched 数：只能从队首硬淘汰
            for i in range(len(lru)):
                if need <= 0:
                    break
                if keep_flag[i]:
                    keep_flag[i] = False
                    need -= 1
        lru = [ri for i, ri in enumerate(lru) if keep_flag[i]]
        capped_dropped = len(merged) - len(lru)
    if not merged:
        # 全部死行：旧视图已无有效声明 → 卸载（视图可丢是契约允许的终态）
        if existing is not None:
            try:
                bm.detach_view(str(view_name))
            except Exception:  # noqa: BLE001
                pass
        return dict(rows=0, dim=0, dead_dropped=int(dead_dropped),
                    capped_dropped=0, gc_rewritten=bool(existing_n > 0))
    rows = np.asarray(sorted(lru), dtype=np.int64)
    vecs = np.stack([merged[int(r)] for r in rows], axis=0).astype(np.float32, copy=False)
    meta_out = {
        "model_id": str(model_id or old_meta.get("model_id") or ""),
        "tokenizer_mode": str(tokenizer_mode or old_meta.get("tokenizer_mode") or ""),
        "chunk_version": str(chunk_version or old_meta.get("chunk_version") or ""),
        "progressive": True,
        "lru_order": [int(r) for r in lru],        # least → most recently used（物理行）
        "merged_at": float(time.time()),
    }
    try:
        bm.attach_view(str(view_name), vecs, rows=rows, available=True,
                       meta=meta_out, replace=True)
    except Exception as exc:  # noqa: BLE001  并发顶替/行已死等 → 不打断查询，下次再合并
        log(f"[rerank] 视图合并失败（忽略）：{type(exc).__name__}: {exc}")
        return dict(rows=0, dim=0, dead_dropped=int(dead_dropped),
                    capped_dropped=int(capped_dropped), gc_rewritten=False)
    gc_rewritten = bool(existing_n > 0
                        and dead_existing / float(existing_n) > RERANK_CACHE_GC_RATIO)
    return dict(rows=int(rows.size), dim=int(vecs.shape[1]),
                dead_dropped=int(dead_dropped), capped_dropped=int(capped_dropped),
                gc_rewritten=bool(gc_rewritten))


def _bm25_fallback(queries: Sequence[str], pools: Sequence[np.ndarray],
                   pool_scores: Sequence[np.ndarray], meta: Any, *,
                   k: int, with_text: bool, with_span: bool) -> list[list[dict]]:
    out_all: list[list[dict]] = []
    k_lim = max(int(k), 0)
    for qi, q in enumerate(queries):
        pool = np.asarray(pools[qi], dtype=np.int64)
        bscores = np.asarray(pool_scores[qi], dtype=np.float32)
        hits: list[dict] = []
        for rank, pj in enumerate(range(min(k_lim, int(pool.size))), 1):
            mp = int(pool[pj])
            m = meta[mp]
            item = dict(rank=rank, score=float(bscores[pj]), docid=m.get("docid"),
                        file=m["file"], line=m["line"])
            if with_text:
                txt = m["text"]
                item["text"] = txt
                if with_span:
                    item["hit_span"] = _hit_span(txt, q)
            if with_span and "char_start" in m:
                item["char_start"] = m["char_start"]
                item["char_end"] = m["char_end"]
            hits.append(item)
        out_all.append(hits)
    return out_all
