#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""primolix.dense_int8 —— 可选 INT8 紧凑语义向量侧车（默认不启用；缺模型/缺侧车/损坏时诚实优雅降级 BM25-only）。

设计约束：
- 始终严格可选：默认建库/开库/检索零编码开销；
- 物理隔离：仅写 `<index_dir>/views/dense_int8/`，删掉即逐位回到纯 BM25；
- Windows 文件锁免疫：盘面读取直接加载为内存 ndarray（不保留悬空 mmap 句柄），close/rmtree 零阻塞；
- 纯 NumPy 兜底：无 torch/CUDA 时分块走 BLAS float32；有 CUDA 时分块 dequant fp16 搜索；
- 滞后容忍：增量 `add()` 后按 `docid` 对齐活跃行，未覆盖新行记作 uncovered，不报错也不错行。
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from .kernel import log
from .rerank import DEFAULT_SUB_BATCH, encode_texts, get_rerank_model

SIDE_SUBDIR = Path("views") / "dense_int8"
FORMAT_TAG = "primolix-dense-int8-v1"


def quantize_int8(vecs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """L2 归一化后按行对称量化为 `(int8[N, D], float16[N])`。"""
    v = np.asarray(vecs, dtype=np.float32)
    if v.ndim != 2 or v.shape[0] == 0:
        dim = int(v.shape[1]) if v.ndim == 2 else 0
        return np.zeros((0, dim), dtype=np.int8), np.zeros(0, dtype=np.float16)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    v = v / np.maximum(norms, np.float32(1e-12))
    max_abs = np.max(np.abs(v), axis=1)
    scales = np.maximum(max_abs / np.float32(127.0), np.float32(1e-8)).astype(np.float16)
    q = np.rint(v / scales[:, None].astype(np.float32)).clip(-127, 127).astype(np.int8)
    return np.ascontiguousarray(q), np.ascontiguousarray(scales)


def union_fuse_scores(cos_arr: np.ndarray, bm_arr: np.ndarray, *, fusion: str = "z2_safe",
                      w_bm: float = 0.35, top_m: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """对并集池内的 `(cos_arr, bm_arr)` 执行排序（`cos` / `z2` / `z2_safe` / `z2_pool`），返回 `(order, scores)`。"""
    n = int(cos_arr.size)
    if n <= 1:
        return np.arange(n, dtype=np.int64), cos_arr.astype(np.float32, copy=True)

    c = np.asarray(cos_arr, dtype=np.float32)
    b = np.asarray(bm_arr, dtype=np.float32)
    if fusion == "cos":
        order = np.argsort(-c, kind="stable")
        return order, c

    if fusion == "z2_safe":
        cos_ord = np.argsort(-c, kind="stable")
        m = min(int(top_m), n)
        top_idx = cos_ord[:m]
        cs, bs = c[top_idx], b[top_idx]
        zc = (cs - float(cs.mean())) / max(float(cs.std()), 0.02)
        zb = (bs - float(bs.mean())) / max(float(bs.std()), 1e-4)
        sub_sc = zc + float(w_bm) * zb
        sub_ord = top_idx[np.argsort(-sub_sc, kind="stable")]
        out_sc = c.copy()
        out_sc[top_idx] = sub_sc
        if n > m:
            tail_max = float(c[cos_ord[m:]].max())
            shift = float(sub_sc.min()) - 2.0 - tail_max
            out_sc[cos_ord[m:]] = c[cos_ord[m:]] + min(0.0, shift)
        return np.concatenate((sub_ord, cos_ord[m:])), out_sc

    w = 0.25 if fusion == "z2_pool" else 0.65
    zc = (c - float(c.mean())) / max(float(c.std()), 0.02)
    zb = (b - float(b.mean())) / max(float(b.std()), 1e-4)
    sc = zc + w * zb
    return np.argsort(-sc, kind="stable"), sc


class DenseInt8Sidecar:
    """可选持久化 `INT8` 侧车管理器；加载/读取失败绝不抛异常阻断主库。"""

    def __init__(self, index_dir: str | Path) -> None:
        self.index_dir = Path(index_dir)
        self.dir = self.index_dir / SIDE_SUBDIR
        self.vecs_int8: np.ndarray | None = None
        self.scales_f16: np.ndarray | None = None
        self.docids: np.ndarray | None = None
        self.meta: dict[str, Any] = {}
        self.did_to_vrow: dict[str, int] = {}
        self.loaded: bool = False
        self.load_error: str | None = None

    def exists_on_disk(self) -> bool:
        return (
            (self.dir / "vecs_int8.npy").is_file()
            and (self.dir / "scales_f16.npy").is_file()
            and (self.dir / "docids.npy").is_file()
            and (self.dir / "meta.json").is_file()
        )

    def clear(self) -> None:
        self.vecs_int8 = None
        self.scales_f16 = None
        self.docids = None
        self.meta = {}
        self.did_to_vrow = {}
        self.loaded = False

    def load(self) -> bool:
        """懒加载侧车到内存（绝不留 Windows 文件锁；损坏返回 False 并记录 `load_error`）。"""
        if self.loaded and self.vecs_int8 is not None:
            return True
        self.clear()
        if not self.exists_on_disk():
            self.load_error = "views/dense_int8 侧车不存在（未构建或已被清理）"
            return False
        try:
            meta = json.loads((self.dir / "meta.json").read_text(encoding="utf-8"))
            if meta.get("fmt") != FORMAT_TAG:
                self.load_error = f"dense_int8 格式不兼容: {meta.get('fmt')!r}"
                return False
            with np.load(self.dir / "vecs_int8.npy", allow_pickle=False) as v_arr:
                vecs = np.array(v_arr, dtype=np.int8, copy=True)
            with np.load(self.dir / "scales_f16.npy", allow_pickle=False) as s_arr:
                scales = np.array(s_arr, dtype=np.float16, copy=True)
            with np.load(self.dir / "docids.npy", allow_pickle=False) as d_arr:
                dids = np.array([str(x) for x in d_arr.tolist()])
            if (
                vecs.ndim != 2
                or scales.ndim != 1
                or vecs.shape[0] != scales.shape[0]
                or vecs.shape[0] != dids.shape[0]
            ):
                self.load_error = "dense_int8 形状不一致，已安全降级"
                return False
            self.vecs_int8 = vecs
            self.scales_f16 = scales
            self.docids = dids
            self.meta = meta
            self.did_to_vrow = {str(d): i for i, d in enumerate(dids.tolist())}
            self.loaded = True
            self.load_error = None
            return True
        except Exception as exc:  # noqa: BLE001  损坏/IO 失败一律降级 BM25-only
            self.clear()
            self.load_error = f"dense_int8 加载失败，已降级: {type(exc).__name__}: {exc}"
            return False

    def build_and_save(
        self,
        docids: Sequence[str],
        texts: Sequence[str],
        *,
        model_path: str = "",
        device: str = "auto",
        sub_batch: int = DEFAULT_SUB_BATCH,
        fp16: bool = True,
        encoder: Callable[[Sequence[str]], np.ndarray] | Any = None,
        chunk_encode: int = 8192,
    ) -> dict[str, Any]:
        """显式构建并原子写入 `views/dense_int8/`；缺编码器时诚实返回 `available=False`（不报错）。"""
        n_docs = len(docids)
        if n_docs != len(texts):
            raise ValueError(f"docids({n_docs}) 与 texts({len(texts)}) 数量不一致")
        if n_docs == 0:
            return dict(available=False, mode="bm25_fallback", reason="空索引无可编码文档")

        if encoder is not None:
            enc = encoder
            model_name = getattr(encoder, "__name__", type(encoder).__name__)
            used_dev = "custom"
            used_mp = ""
            st_info = dict(available=True, model=model_name, model_path="", device="custom")
        else:
            enc, st_info = get_rerank_model(model_path, device=device)
            if enc is None:
                return dict(st_info, mode="bm25_fallback", built=False)
            used_dev = str(st_info.get("device") or "cpu")
            used_mp = str(st_info.get("model_path") or "")
            model_name = str(st_info.get("model") or "dense")

        t0 = time.time()
        # 可选 half()：CUDA 上速度 ~3×、实测 MAE 7e-5
        if fp16 and used_dev.startswith("cuda") and hasattr(enc, "half"):
            try:
                enc.half()
            except Exception:  # noqa: BLE001
                pass

        vecs_chunks: list[np.ndarray] = []
        scales_chunks: list[np.ndarray] = []
        step = max(int(chunk_encode), int(sub_batch), 1)
        for s in range(0, n_docs, step):
            blk = list(texts[s:s + step])
            embs = encode_texts(enc, blk, sub_batch=sub_batch, device=used_dev)
            if embs.ndim != 2 or embs.shape[0] != len(blk):
                return dict(available=False, mode="bm25_fallback",
                            reason="编码输出维度异常，已放弃写入 dense_int8")
            q_blk, sc_blk = quantize_int8(embs)
            vecs_chunks.append(q_blk)
            scales_chunks.append(sc_blk)

        vecs_all = np.concatenate(vecs_chunks, axis=0)
        scales_all = np.concatenate(scales_chunks, axis=0)
        dids_arr = np.asarray([str(d) for d in docids])
        dim = int(vecs_all.shape[1])
        dt = round(time.time() - t0, 3)

        self.dir.mkdir(parents=True, exist_ok=True)
        meta_out = {
            "fmt": FORMAT_TAG,
            "N": int(n_docs),
            "dim": dim,
            "model": model_name,
            "model_path": used_mp,
            "fp16_encode": bool(fp16),
            "build_s": dt,
        }
        tmp_v = self.dir / "vecs_int8.tmp.npy"
        tmp_s = self.dir / "scales_f16.tmp.npy"
        tmp_d = self.dir / "docids.tmp.npy"
        tmp_m = self.dir / "meta.tmp.json"
        np.save(str(tmp_v), vecs_all)
        np.save(str(tmp_s), scales_all)
        np.save(str(tmp_d), dids_arr)
        tmp_m.write_text(json.dumps(meta_out, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp_v, self.dir / "vecs_int8.npy")
        os.replace(tmp_s, self.dir / "scales_f16.npy")
        os.replace(tmp_d, self.dir / "docids.npy")
        os.replace(tmp_m, self.dir / "meta.json")

        self.vecs_int8 = vecs_all
        self.scales_f16 = scales_all
        self.docids = dids_arr
        self.meta = meta_out
        self.did_to_vrow = {str(d): i for i, d in enumerate(dids_arr.tolist())}
        self.loaded = True
        self.load_error = None

        size_mb = round((vecs_all.nbytes + scales_all.nbytes) / 1048576.0, 2)
        log(f"dense_int8 侧车落盘 {self.dir} ({n_docs:,}×{dim}, {size_mb:.1f} MiB, {dt:.1f}s)")
        return dict(
            available=True,
            built=True,
            mode="dense_int8_ready",
            N=int(n_docs),
            dim=dim,
            size_mb=size_mb,
            model=model_name,
            model_path=used_mp,
            device=used_dev,
            build_s=dt,
        )

    def align_active(self, active_docids: Sequence[str]) -> tuple[np.ndarray, np.ndarray, int]:
        """对齐侧车到当前活跃 `_meta` 行位置，返回 `(covered_meta_pos, covered_vrows, n_unmatched)`。"""
        if not self.load() or self.docids is None:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), 0
        m_pos: list[int] = []
        v_rows: list[int] = []
        active_set: set[str] = set()
        for mp, did in enumerate(active_docids):
            sd = str(did)
            active_set.add(sd)
            vr = self.did_to_vrow.get(sd)
            if vr is not None:
                m_pos.append(mp)
                v_rows.append(vr)
        unmatched = sum(1 for sd in self.did_to_vrow if sd not in active_set)
        return np.asarray(m_pos, dtype=np.int64), np.asarray(v_rows, dtype=np.int64), int(unmatched)

    def search_aligned(
        self,
        qvec: np.ndarray,
        covered_meta_pos: np.ndarray,
        covered_vrows: np.ndarray,
        *,
        top_k: int = 100,
    ) -> tuple[np.ndarray, np.ndarray]:
        """对单个 query 向量精确检索对齐的 INT8 子集，返回 `(meta_positions, cos_scores)`。"""
        if (
            not self.loaded
            or self.vecs_int8 is None
            or self.scales_f16 is None
            or covered_meta_pos.size == 0
        ):
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32)

        qv = np.asarray(qvec, dtype=np.float32).reshape(-1)
        if qv.size != self.vecs_int8.shape[1]:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32)

        k_lim = min(max(int(top_k), 1), int(covered_meta_pos.size))
        n_cov = int(covered_vrows.size)
        is_full_identity = (
            n_cov == self.vecs_int8.shape[0]
            and covered_vrows[0] == 0
            and covered_vrows[-1] == n_cov - 1
            and bool(np.array_equal(covered_vrows, np.arange(n_cov, dtype=np.int64)))
        )

        # 大库自动走 CUDA 分块 fp16，小库或无 GPU 走纯 NumPy BLAS 分块
        try:
            import torch

            if torch.cuda.is_available() and n_cov >= 8192:
                return self._search_cuda(qv, covered_meta_pos, covered_vrows, is_full_identity, k_lim, torch)
        except Exception:  # noqa: BLE001  CUDA 不可用无缝退回 CPU NumPy
            pass

        scores = np.empty(n_cov, dtype=np.float32)
        blk = 65536
        for s in range(0, n_cov, blk):
            e = min(s + blk, n_cov)
            rows_blk = slice(s, e) if is_full_identity else covered_vrows[s:e]
            dv = self.vecs_int8[rows_blk].astype(np.float32)
            dv *= self.scales_f16[rows_blk][:, None].astype(np.float32)
            scores[s:e] = dv @ qv
        order = np.argpartition(-scores, k_lim - 1)[:k_lim] if k_lim < n_cov else np.arange(n_cov)
        order = order[np.argsort(-scores[order], kind="stable")]
        return covered_meta_pos[order], scores[order]

    def _search_cuda(self, qv: np.ndarray, covered_meta_pos: np.ndarray, covered_vrows: np.ndarray,
                     is_full_identity: bool, k_lim: int, torch_mod: Any) -> tuple[np.ndarray, np.ndarray]:
        qt = torch_mod.from_numpy(qv.astype(np.float16)).to("cuda")
        n_cov = int(covered_vrows.size)
        blk = 250_000
        best_v = torch_mod.full((k_lim,), -1e9, dtype=torch_mod.float32, device="cuda")
        best_idx = torch_mod.full((k_lim,), -1, dtype=torch_mod.int64, device="cuda")
        for s in range(0, n_cov, blk):
            e = min(s + blk, n_cov)
            rows_blk = slice(s, e) if is_full_identity else covered_vrows[s:e]
            d_blk = torch_mod.from_numpy(np.array(self.vecs_int8[rows_blk], copy=True)).to("cuda", dtype=torch_mod.float16)
            sc_blk = torch_mod.from_numpy(np.array(self.scales_f16[rows_blk], copy=True)).to("cuda", dtype=torch_mod.float16)
            d_blk.mul_(sc_blk.unsqueeze(1))
            sims = torch_mod.mv(d_blk, qt).to(torch_mod.float32)
            kk = min(k_lim, e - s)
            vals, locs = torch_mod.topk(sims, k=kk)
            c_idx = locs.to(torch_mod.int64) + s
            comb_v = torch_mod.cat((best_v, vals))
            comb_i = torch_mod.cat((best_idx, c_idx))
            tv, ti = torch_mod.topk(comb_v, k=k_lim)
            best_v = tv
            best_idx = comb_i[ti]
            del d_blk, sc_blk, sims, vals, locs, c_idx, comb_v, comb_i, tv, ti
        chosen = best_idx.cpu().numpy()
        out_v = best_v.cpu().numpy()
        del qt, best_v, best_idx
        return covered_meta_pos[chosen], out_v

    def cos_for_docids(self, qvec: np.ndarray, docids: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
        """按 docid 查侧车精确余弦；未命中（未覆盖）标为 `mask[i]=False`。"""
        n = len(docids)
        cos_out = np.zeros(n, dtype=np.float32)
        mask = np.zeros(n, dtype=bool)
        if not self.loaded or self.vecs_int8 is None or self.scales_f16 is None or n == 0:
            return cos_out, mask
        qv = np.asarray(qvec, dtype=np.float32).reshape(-1)
        if qv.size != self.vecs_int8.shape[1]:
            return cos_out, mask
        hit_pos, hit_vr = [], []
        for i, did in enumerate(docids):
            vr = self.did_to_vrow.get(str(did))
            if vr is not None:
                hit_pos.append(i)
                hit_vr.append(vr)
        if not hit_pos:
            return cos_out, mask
        vr_arr = np.asarray(hit_vr, dtype=np.int64)
        dv = self.vecs_int8[vr_arr].astype(np.float32) * self.scales_f16[vr_arr][:, None].astype(np.float32)
        vals = dv @ qv
        hp_arr = np.asarray(hit_pos, dtype=np.int64)
        cos_out[hp_arr] = vals
        mask[hp_arr] = True
        return cos_out, mask

    def status(self, active_docids: Sequence[str] | None = None) -> dict[str, Any]:
        if not self.exists_on_disk():
            return dict(available=False, loaded=False, reason="views/dense_int8 未构建（纯 BM25 零编码）")
        ok = self.load()
        if not ok:
            return dict(available=False, loaded=False, reason=self.load_error)
        total_v = int(self.vecs_int8.shape[0]) if self.vecs_int8 is not None else 0
        dim = int(self.vecs_int8.shape[1]) if self.vecs_int8 is not None else 0
        size_mb = round(((self.vecs_int8.nbytes + self.scales_f16.nbytes) / 1048576.0)
                        if (self.vecs_int8 is not None and self.scales_f16 is not None) else 0.0, 2)
        covered, unmatched, total_active = total_v, 0, total_v
        if active_docids is not None:
            m_pos, _, unmatched = self.align_active(active_docids)
            covered = int(m_pos.size)
            total_active = len(active_docids)
        return dict(
            available=True,
            loaded=True,
            dim=dim,
            vectors=total_v,
            covered=covered,
            total_active=total_active,
            unmatched=unmatched,
            size_mb=size_mb,
            model=self.meta.get("model"),
            model_path=self.meta.get("model_path"),
        )
