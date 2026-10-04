#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""primolix.api —— 一层薄门面：把繁琐的工程 API 收成一句话。

内核把每个机制都摊开（段要自己写三个 `.npy`、开关是类属性、归因要自己拼 idf 公式），当库用太啰嗦。
这层不改内核，只把常用动作收成对象方法，并给出两个内核没有的默认：并发安全默认开（读持共享侧、
写持独占侧，写操作会等所有读者退出，内核裸调用仍不安全）、不许静默（可能悄悄换算法处一律如实返回或抛出）。
用法：`idx = primolix.open("docs/.idx")` → `search` / `add` / `tune` / `explain` / `mount` / `compact` / `stats`。
开关（`FAST_SCORE` 等）是类属性 → 进程级，`config()` 会如实说明。
"""
from __future__ import annotations

import os
import threading
import time
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from . import kernel
from .core import Primolix, get_dense_model
from .kernel import RWLock, SparseBM25, zh_tokenize     # RWLock 复用内核的可重入实现

__all__ = ["RWLock", "Index", "open_index", "build_index", "set_switches"]

# 门面可切换的开关（都是类属性 → 影响本进程内所有索引）
SWITCHES = ("FAST_SCORE", "DN_PRECOMP", "ADD_AT", "W_CACHE", "MATVEC", "WRITE_DICT",
            "TC_SHARED", "EAGER_TC", "SWAP_LOCK", "T_SLACK")


# 门面自用的锁：复用内核的可重入读写锁。内核只保证单次调用的原子性；门面有跨调用组合的操作
# （`add()` = 墓碑＋追加＋落盘；`compact()` = 折段＋清块＋重建）→ 需要一层更粗的临界区。


class Index:
    """索引的一句话门面。构造用 `primolix.open()` / `primolix.build()`。"""

    def __init__(self, path, *, dense: bool = False, model_path: str = "") -> None:
        self.dir = Path(path)
        if not self.dir.exists():
            raise FileNotFoundError(f"索引目录不存在：{self.dir}（要新建请用 primolix.build(...)）")
        self._lock = RWLock()
        self._stages: set = set()
        self._z = Primolix(str(self.dir), dense=dense, model_path=model_path)
        self._z._load()

    # ---------------------------------------------------------------- 内部
    @property
    def bm(self) -> SparseBM25:
        """内核对象（摸它就等于绕过门面的并发保护 —— 只在单线程里用）。"""
        return self._z._bm

    def _need_save(self) -> None:
        self._z._persist()

    # ---------------------------------------------------------------- 读
    def search(self, q: str, k: int = 10, *, dense: bool = False) -> list[dict]:
        """检索：文本进、结果出（自动分词）。返回 `[{rank, score, file, line, text}]`。"""
        with self._lock.read():
            return self._z.query(q, k=k, dense=dense)

    def stats(self) -> dict:
        """索引概览：规模 ＋ 落盘大小 ＋ 开关 ＋ 段/视图计数。"""
        with self._lock.read():
            info = dict(self._z.info)          # `Primolix.info` 是 property（不加括号）
            bm = self._z._bm
            info.update(
                k1=float(bm.k1), b=float(bm.b), avgdl=float(getattr(bm, "avgdl", 0.0)),
                segments=len(bm.segment_report()), folded=len(bm.folded_report()),
                views=len(getattr(bm, "_views", {}) or {}),
                blocks=len(getattr(bm, "_tc_blocks", []) or []),
                switches={s: getattr(SparseBM25, s) for s in SWITCHES},
                tokenizer=kernel.tokenizer_status(),
            )
            return info

    def files(self) -> list[str]:
        with self._lock.read():
            return self._z.list_files()

    def browse(self, file: str | None = None, page: int = 0, n: int = 20):
        with self._lock.read():
            return self._z.browse(file=file, page=page, n=n)

    def explain(self, q: str, *, docid: str | None = None, k: int = 1, top: int = 6) -> list[dict]:
        """归因：把第 `k` 名的分数按词拆成 `idf × tf 饱和 × 文档长度` 的贡献。

        `docid` 给定时拆那一篇（按 `file:line` 或文件名匹配），否则拆当前第 `k` 名。
        """
        from collections import Counter
        with self._lock.read():
            hits = self._z.query(q, k=max(k, 1))
            if not hits:
                return []
            hit = hits[min(k, len(hits)) - 1]
            if docid:
                cand = [h for h in hits if docid in (h["file"] or "")]
                if not cand:
                    raise KeyError(f"本次 top-{len(hits)} 里没有匹配 {docid!r} 的文档")
                hit = cand[0]
            bm = self._z._bm
            row = int(next(m["row"] for m in self._z._meta
                           if m["file"] == hit["file"] and m["line"] == hit["line"]))
            dl, avg, n = float(bm.dl[row]), float(bm.avgdl), int(bm.N)
            T = bm.T.tocsr() if bm.T is not None else None
            out = []
            for t, cnt in Counter(zh_tokenize(q)).items():
                c = bm.term2idx.get(t)
                if c is None or T is None:
                    continue
                tf = float(T[row, c])
                if tf <= 0:
                    continue
                df = float(bm.df[c])
                idf = float(np.log((n - df + 0.5) / (df + 0.5) + 1.0))
                sat = tf * (bm.k1 + 1.0) / (tf + bm.k1 * (1.0 - bm.b + bm.b * dl / avg))
                out.append(dict(term=t, q_count=int(cnt), tf=tf, df=df, idf=idf,
                                contribution=idf * sat))
            out.sort(key=lambda d: -d["contribution"])
            return out[:top]

    # ---------------------------------------------------------------- 写
    def add(self, root: str, *, workers: int = 0) -> dict:
        """增量：只处理改动过的文件；出现词表外新词 → 整库回退重建（`kind='rebuild'`）。

        已挂段时就地更新会被拒（段的逻辑行会与就地追加的行重叠 → 静默错分）→ 先 `compact()`
        折叠，或改走 `stage_begin()`；纯删除不受限。
        """
        with self._lock.write():
            if self._z._bm.segment_report():
                raise RuntimeError(
                    f"已挂段 {sorted(self._z._bm.segment_report())} → 先 `compact()`（折叠进主表）"
                    f"再走就地更新，或改用 `stage_begin()` 走挂段路径")
            return self._z.update(root, workers=workers)

    def rebuild(self, root: str, *, workers: int = 0, force: bool = False) -> dict:
        """整库重建到同一目录（词表外回退 / 显式重建）。

        已发布的未折叠段不在源语料里 → 重建会丢掉它们：默认拒绝，要丢就显式 `force=True`。
        """
        with self._lock.write():
            self._z.rebuild_in_place(root, workers=workers, force=force)
            return dict(kind="rebuild", force=bool(force), files=len(self._z._hashes))

    def tune(self, *, k1: float | None = None, b: float | None = None,
             persist: bool = False) -> dict:
        """换公式：`k1` / `b` 是两次属性赋值，不重建、不动物料。

        `persist=True` 时把新参数写回索引（否则只在本进程生效 —— 与内核语义一致）。
        """
        with self._lock.write():
            bm = self._z._bm
            before = dict(k1=float(bm.k1), b=float(bm.b))
            if k1 is not None:
                bm.k1 = float(k1)
            if b is not None:
                bm.b = float(b)
            if persist:
                self._need_save()
            return dict(before=before, after=dict(k1=float(bm.k1), b=float(bm.b)),
                        persisted=bool(persist))

    def config(self, **kw) -> dict:
        """读/写开关。开关是类属性 → 进程级（改了会影响本进程里所有索引）。"""
        with self._lock.write():
            return set_switches(**kw)

    # ---------------------------------------------------------------- 段
    def mount(self, seg_dir: str, tag: str, new_terms, dl, rows=None, *,
              generation: str | None = None) -> dict:
        """挂一段只读 postings（三个 `.npy`，CSC，列号落在"主表加宽后"的空间里）。

        与全库重建逐位一致；不写盘、不动主表 `T`。后进先出：只有最后挂的段能卸。
        """
        with self._lock.write():
            return self._z._bm.mount_segment(seg_dir, tag, new_terms, dl, rows=rows,
                                             generation=generation)

    def unmount(self, tag: str) -> dict:
        """撤回一个已挂的段（分数逐位回到挂载前）。已折叠的段不能撤。"""
        with self._lock.write():
            return self._z._bm.unmount_segment(tag)

    def fold(self, tag: str) -> dict:
        """把已挂的段并进主表（一次 `O(nnz)`；折叠前后分数不变 → 段不再存在）。"""
        with self._lock.write():
            return self._z._bm.fold_segment(tag)

    def segments(self) -> dict:
        """`{'mounted': {...}, 'folded': {...}}`。"""
        with self._lock.read():
            return dict(mounted=self._z._bm.segment_report(),
                        folded=self._z._bm.folded_report())

    def compact(self) -> dict:
        """压实：把已挂段全部折进主表 ＋ 清分块缓存 → 查询恢复单块。

        查询耗时随分块数上升，长期只追加会把块数堆起来；折叠是一次 `O(nnz)`，折完只走主表。
        内核不变量 `fold_segment` 断言"折完 T 行数 == N"（`N` 含所有已挂段）→ 多段时用不了
        `fold_segment`，这里改走 `fold_all()`。
        """
        with self._lock.write():
            from .core import manifest_append
            bm = self._z._bm
            segs = list(bm.segment_report())
            if len(segs) > 1:
                # 多段时走内核的 `fold_all()`（`fold_segment` 的不变量在多段下不成立）
                info = bm.fold_all()
                bm._tc_blocks, bm._tc_rows = [], 0
                bm._invalidate_w()
                bm._recompute_stats()
                bm._ensure_tc()
                # 折叠 = 一次提交：必须往旁车追加 `folded_all`，否则重载时旁车仍把已折的段当"在挂"。
                manifest_append(self.dir, {"event": "folded_all", "tags": list(info["tags"]),
                                           "via": "compact/fold_all"})
                return dict(folded=[dict(tag=t, **{k: v for k, v in info.items()
                                                   if k not in ("tags",)}) for t in info["tags"]],
                            blocks=len(bm._tc_blocks), N=int(bm.N), V=int(bm.V),
                            via="fold_all", ms=info.get("ms"))
            folded = []
            for tag in segs:
                folded.append(dict(tag=tag, **bm.fold_segment(tag)))
            if folded:
                manifest_append(self.dir, {"event": "folded_all",
                                           "tags": [f["tag"] for f in folded],
                                           "via": "compact/fold_segment"})
            bm._tc_blocks, bm._tc_rows = [], 0
            bm._invalidate_w()
            bm._recompute_stats()
            bm._ensure_tc()                       # 立刻按新 T 建回单块（否则下一次查询才建）
            return dict(folded=folded, blocks=len(bm._tc_blocks), N=int(bm.N), V=int(bm.V))

    # ---------------------------------------------------------------- 临时段发布
    def stage_begin(self, tag: str | None = None, *, generation: str | None = None) -> "Stage":
        """起一批旁路累积（累积期不碰任何共享状态）；一批完只做一次发布。

        与 `add()`（就地真增量）互斥：段在场时就地追加会与段的逻辑行重叠（静默错分），
        门面在这里提前给出更清楚的报错。
        """
        with self._lock.read():
            st = Stage(self, tag=tag, generation=generation)
            self._stages.add(st)
            return st

    def stages(self) -> list[dict]:
        """当前在飞的批次（通常 ≤1）。"""
        return [s.report() for s in list(self._stages)]

    def remount_staged(self) -> list[dict]:
        """按旁车 manifest 重挂已发布的段（`load` 时也会自动做一次）。"""
        with self._lock.write():
            from .core import remount_staged as _rm
            return _rm(self._z._bm, self.dir)

    # ---------------------------------------------------------------- 视图
    def views(self) -> list[dict]:
        with self._lock.read():
            return self._z.view_report()

    def attach_view(self, name: str, vectors, meta_positions=None, **kw) -> dict:
        """挂一个派生视图（`vectors` 与 `meta_positions` 对齐；缺省 = 全部活跃 chunk）。

        视图是可丢的派生层：删掉全部视图 → 查询逐位回到纯 BM25。
        """
        with self._lock.write():
            return self._z.attach_view(name, vectors, meta_positions, **kw)

    def detach_view(self, name: str) -> dict:
        with self._lock.write():
            return self._z.detach_view(name)

    def fuse(self, q: str, qvec, view_name: str, *, alpha: float = 0.35,
             k: int = 10) -> list[dict]:
        """稀疏 ＋ 稠密融合打分（覆盖行分数单调不降），返回 top-k。"""
        with self._lock.read():
            z = self._z
            s = z._bm.fused_scores(zh_tokenize(q), qvec, view_name, alpha=alpha)
            rows = np.asarray([c["row"] for c in z._meta], dtype=np.int64)
            ranked = np.argsort(-s[rows])[:k]
            out = []
            for i, j in enumerate(ranked, 1):
                m = z._meta[int(j)]
                out.append(dict(rank=i, score=float(s[rows[j]]), file=m["file"],
                                line=m["line"], text=m["text"]))
            return out

    def save_view(self, name: str, docids, vectors, meta=None) -> dict:
        with self._lock.write():
            return self._z.save_view(name, docids, vectors, meta=meta)

    def load_views(self, extra_dirs=None) -> list[dict]:
        with self._lock.write():
            return self._z.load_views(extra_dirs)

    # ---------------------------------------------------------------- 落盘
    def save(self, path: str | None = None, *, w_cache: bool = False,
             fold_first: bool = False) -> dict:
        """落盘。`w_cache=True` 时额外把物化层 `w` 写进 `.mm/`（冷启动免重建）。

        段在场时内核会拒（挂段是内存视图 → 写出的 npz 会 `dl`/`T` 不一致 → 重载崩）→
        显式 `fold_first=True` 就先 `compact()` 再落盘。
        `path` 缺省 = 写回本索引目录（此时把 `w_cache` 记在索引对象上）；给 `path` = 只把内核
        索引另存一份（不动本目录的 meta/hashes）。
        """
        with self._lock.write():
            if fold_first and self._z._bm.segment_report():
                self.compact()                      # 折叠 ＋ 追加 `folded_all` 事件
            if path is None:
                self._z._w_cache = bool(w_cache)
                self._need_save()
                return dict(path=str(self.dir / "bm25.npz"), w_cache=bool(w_cache),
                            scope="index_dir")
            self._z._bm.save(str(path), w_cache=w_cache)
            return dict(path=str(path), w_cache=bool(w_cache), scope="kernel_only")


# -------------------------------------------------------------------- 入口
def set_switches(**kw) -> dict:
    """设置内核开关（不依赖任何索引对象 —— 它们是类属性 → 进程级）。

    影响本进程内所有索引；不改盘上任何东西（进程重启即回到默认）。
    """
    unknown = [k for k in kw if k not in SWITCHES]
    if unknown:
        raise KeyError(f"未知开关 {unknown}；可选 {list(SWITCHES)}")
    before = {k: getattr(SparseBM25, k) for k in kw}
    for k, v in kw.items():
        setattr(SparseBM25, k, v)
    return dict(before=before, after={k: getattr(SparseBM25, k) for k in kw},
                scope="process（类属性 → 影响本进程所有索引）")


def open_index(path, *, dense: bool = False, model_path: str = "") -> Index:
    """打开已有索引（与 `open_index(...)` 同义的是 `primolix.open(...)`）。"""
    return Index(path, dense=dense, model_path=model_path)


def build_index(root, out: str = ".primolix_index", *, dense: bool = False,
                w_cache: bool = False, workers: int = 0,
                model_path: str = "") -> Index:
    """建索引并返回门面对象。`w_cache=True` → 建库时把物化层一并落盘。"""
    z = Primolix.build(root, out=out, dense=dense, model_path=model_path,
                       workers=workers, w_cache=w_cache)
    idx = Index.__new__(Index)                 # 复用刚建好的对象，避免二次装载
    idx.dir = Path(out)
    idx._lock = RWLock()
    idx._stages = set()
    idx._z = z
    return idx


class Stage:
    """一个旁路累积中的批次（`Index.stage_begin()` 造；累积期不碰任何共享状态）。

    提交点 = 向旁车 manifest 追加 `publish` 事件（追加式，从不改写历史行）：段文件先写好 →
    追加 manifest（提交）→ 再挂段（进写锁 → 对查询原子）。
    崩溃语义：文件写好但没追加 → 未承诺（孤儿文件）；已追加但没挂上 → 重载时会挂上。
    与 `add()`（就地真增量）互斥：段在场时就地追加会与段的逻辑行重叠（静默错分）。
    """

    def __init__(self, idx: "Index", tag: str | None = None, generation: str | None = None):
        self.idx = idx
        self.bm = idx.bm
        self.tag = tag or (time.strftime("%Y%m%d-%H%M%S") + f"-{os.getpid()}")
        self.generation = generation or self.tag
        self.base_V = len(self.bm.idx2term)
        self.base_T_rows = int(self.bm.T.shape[0])
        _t2i = self.bm.term2idx                      # 可能是 dict，也可能是 `VocabMmap`（只读载体）
        self._t2i = None if hasattr(_t2i, "id_of") else dict(_t2i)
        self._rows: list = []
        self._new_terms: list = []
        self._new_pos: dict = {}

    # ---------------------------------------------------------------- 累积（只读词表）
    def _has(self, t: str) -> bool:
        return (t in self._t2i) if self._t2i is not None else (t in self.bm.term2idx)

    def add(self, docid, text_or_tokens) -> "Stage":
        """加入一篇：文本（分词一次）或已分好的 token 列表。返回 `self`（可链式）。"""
        toks = (zh_tokenize(text_or_tokens) if isinstance(text_or_tokens, str)
                else list(text_or_tokens))
        for t in toks:
            if self._has(t) or t in self._new_pos:
                continue
            self._new_pos[t] = self.base_V + len(self._new_terms)
            self._new_terms.append(t)
        self._rows.append((docid, toks))
        return self

    def add_many(self, pairs) -> "Stage":
        for docid, x in pairs:
            self.add(docid, x)
        return self

    @property
    def pending(self) -> int:
        return len(self._rows)

    def report(self) -> dict:
        return dict(tag=self.tag, pending=self.pending, new_terms=len(self._new_terms),
                    base_V=self.base_V, base_T_rows=self.base_T_rows)

    def stats(self) -> dict:
        r = self.report()
        r["tokens_pending"] = int(sum(len(t) for _, t in self._rows))
        return r

    def abort(self) -> dict:
        """丢弃本批（段文件不入总线，共享状态从未被碰过）。"""
        n = self.pending
        self._rows, self._new_terms, self._new_pos = [], [], {}
        self.idx._stages.discard(self)
        return dict(tag=self.tag, aborted=n)

    # ---------------------------------------------------------------- 发布（一次）
    def publish(self, *, fold: bool = False) -> dict:
        """一次性发布：写段文件 → 追加旁车（提交点）→ 挂段（写锁内，对查询原子）。

        `fold=True`：再折进主表并落盘（代价 `O(总 nnz)` 写盘），并追加 `folded_all` 事件。
        """
        if not self._rows:
            raise RuntimeError("空批次：没有任何文档可发布（先 `add()`）")
        from .core import SEG_SUBDIR, manifest_append
        idx, bm = self.idx, self.bm
        with idx._lock.write():
            if len(bm.idx2term) != self.base_V:
                raise RuntimeError(f"基世代已变（base_V={self.base_V} → {len(bm.idx2term)}）"
                                   f" → 这批必须重新分段（不静默重映射）")
            if int(bm.T.shape[0]) != self.base_T_rows:
                raise RuntimeError(f"主表行数已变（{self.base_T_rows} → {int(bm.T.shape[0])}）"
                                   f" → 有人就地追加过；先 `compact()` 再重来")
            V0, M = self.base_V, len(self._new_terms)
            seg_dir = Path(idx.dir) / SEG_SUBDIR / str(self.tag)
            seg_dir.mkdir(parents=True, exist_ok=True)
            ri, ci, da = [], [], []
            for i, (_docid, toks) in enumerate(self._rows):
                for t, c in Counter(toks).items():
                    col = self._t2i.get(t) if self._t2i is not None else bm.term2idx.get(t)
                    if col is None:
                        col = self._new_pos[t]
                    ri.append(i)
                    ci.append(col)
                    da.append(float(c))
            csc = sp.csc_matrix((np.asarray(da, np.float32),
                                 (np.asarray(ri, np.int32), np.asarray(ci, np.int32))),
                                shape=(len(self._rows), V0 + M))
            np.save(seg_dir / f"{self.tag}.data.npy", np.asarray(csc.data, np.float32))
            np.save(seg_dir / f"{self.tag}.idx.npy", np.asarray(csc.indices, np.int32))
            np.save(seg_dir / f"{self.tag}.ptr.npy", np.asarray(csc.indptr, np.int32))
            dl = np.asarray([len(t) for _, t in self._rows], np.float32)
            np.save(seg_dir / "dl.npy", dl)
            rec = dict(event="publish", tag=self.tag, generation=self.generation,
                       rows=len(self._rows), row_base=int(bm.N), base_V=V0, m_new=M,
                       new_terms=list(self._new_terms), created=time.time(),
                       seg_dir=f"{SEG_SUBDIR}/{self.tag}")
            manifest_append(idx.dir, rec)            # 提交点（append-only）
            info = bm.mount_segment(str(seg_dir), str(self.tag), list(self._new_terms), dl,
                                    rows=len(self._rows), generation=self.generation)
            folded = None
            if fold:
                folded = bm.fold_segment(str(self.tag))
                idx._z._persist()                    # 折叠必须落盘才算持久（O(总 nnz)）
                manifest_append(idx.dir, {"event": "folded_all", "tag": self.tag})
            idx._stages.discard(self)
            self._rows, self._new_terms, self._new_pos = [], [], {}
            return dict(tag=self.tag, rows=rec["rows"], V0=V0, V1=V0 + M,
                        N=int(bm.N), V=int(bm.V), mounted=info, folded=folded)
