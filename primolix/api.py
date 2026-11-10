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

import json
import os
import threading
import time
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from . import kernel
from .core import Primolix, get_dense_model, publish_segment, _topk_idx
from .kernel import RWLock, SparseBM25, zh_tokenize     # RWLock 复用内核的可重入实现

__all__ = ["RWLock", "Index", "open_index", "build_index", "set_switches"]

# 门面可切换的开关（都是类属性 → 影响本进程内所有索引）
SWITCHES = ("FAST_SCORE", "DN_PRECOMP", "ADD_AT", "W_CACHE", "MATVEC", "WRITE_DICT",
            "WRITE_VEC", "TC_SHARED", "EAGER_TC", "SWAP_LOCK", "T_SLACK")


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
        """检索：文本进、结果出（自动分词）。返回 `[{rank, score, docid, file, line, text}]`。

        `docid` 是 additive 字段（2026-10-18 乙档）：段内文档常常没有真实 `file/line`，
        只靠 `file/line` 认不出来。
        """
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

    def explain(self, q: str, *, docid: str | None = None, k: int = 1, top: int = 6,
                diagnose: bool = False):
        """归因：把第 `k` 名的分数按词拆成 `idf × tf 饱和 × 文档长度` 的贡献。

        `docid` 给定时拆那一篇（按文件名片段匹配，全库找、取该文件分最高的 chunk），
        否则拆当前第 `k` 名。

        2026-10-19 补两处（此前这两问答不了）：
        1. 词表外（OOV）不再被静默跳过：显式出现在返回列表里（`in_vocab=False`、
           `contribution=0.0`），且不受 `top` 截断 —— 产品上「这个词你的库里没有」
           是最要紧的一句，不能被 `top` 挤掉。
        2. 没搜到时不再只返回空 list：`diagnose=True` ⇒ 返回
           `dict(contributions=[...], diagnosis={...})`，`diagnosis` 说清为什么没搜到。
           默认仍返回 `list`（向后兼容现存调用方）；只想看诊断可直接调 `diagnose()`。

        `top` 只限制有贡献的词条数；`status` 为 `oov` / `no_tf` / `unavailable`
        的项必出（它们解释的是"为什么这个词没帮上忙"）。
        """
        with self._lock.read():
            qtoks = zh_tokenize(q)
            s = self._z._score_active(qtoks)
            if docid:
                rows = [i for i, m in enumerate(self._z._meta)
                        if docid in (m["file"] or "")]
                if not rows:
                    raise KeyError(f"库里没有文件名匹配 {docid!r} 的文档")
                row = int(max(rows, key=lambda i: float(s[i])))
            else:
                # 无正分 ⇒ 没有"第 k 名"可拆。旧实现在这里 `return []`，
                if s.size == 0 or not bool((s > 0).any()):
                    return (dict(contributions=[], diagnosis=self._diagnose(q, k, s, qtoks))
                            if diagnose else [])
                order = _topk_idx(s, max(int(k), 1))
                row = int(order[min(max(int(k), 1), len(order)) - 1])
            bm = self._z._bm
            dl, avg, n = float(bm.dl[row]), float(bm.avgdl), int(bm.N)
            avg = avg if avg > 0 else 1.0          # 空索引/单文档时 avgdl 可能为 0
            T = bm.T.tocsr() if bm.T is not None else None
            nT = int(T.shape[0]) if T is not None else 0
            # 段内文档（未折叠）的物理行不在主表 `T` 里 ⇒ 去分块里取 tf（否则 `T[row, c]` 越界）
            blk = None
            for _c, _rb in (getattr(bm, "_tc_blocks", []) or []):
                if int(_rb) <= row < int(_rb) + int(_c.shape[0]):
                    blk = (_c, int(_rb))
                    break
            contrib, extra = [], []
            for t, cnt in Counter(qtoks).items():
                c = bm.term2idx.get(t)
                if c is None:
                    extra.append(dict(term=t, q_count=int(cnt), in_vocab=False,
                                      status="oov", tf=0.0, df=0.0, idf=0.0,
                                      contribution=0.0))
                    continue
                df = float(bm.df[c])
                idf = float(np.log((n - df + 0.5) / (df + 0.5) + 1.0))
                if T is not None and row < nT:
                    tf = float(T[row, c])
                elif blk is not None:
                    tf = float(blk[0][row - blk[1], c])
                else:
                    extra.append(dict(term=t, q_count=int(cnt), in_vocab=True,
                                      status="unavailable", tf=None, df=df, idf=idf,
                                      contribution=None))
                    continue
                if tf <= 0:
                    extra.append(dict(term=t, q_count=int(cnt), in_vocab=True,
                                      status="no_tf", tf=0.0, df=df, idf=idf,
                                      contribution=0.0))
                    continue
                sat = tf * (bm.k1 + 1.0) / (tf + bm.k1 * (1.0 - bm.b + bm.b * dl / avg))
                contrib.append(dict(term=t, q_count=int(cnt), in_vocab=True,
                                    status="ok", tf=tf, df=df, idf=idf,
                                    contribution=idf * sat))
            contrib.sort(key=lambda d: -d["contribution"])
            out = contrib[:max(int(top), 0)] + extra
            if diagnose:
                return dict(contributions=out, diagnosis=self._diagnose(q, k, s, qtoks))
            return out

    def _diagnose(self, q: str, k: int = 10, s=None, qtoks=None) -> dict:
        """（内部）**为什么没搜到** / 为什么只搜到这些 —— 只输出有把握的判断。

        每条 `messages[*]` 都带 `evidence`（可点开验证的实数）与 `action`（人话动作），
        符合本层「不许静默、解释必须可举证」的约定。不猜：拿不到的数一律不编。
        """
        qtoks = list(qtoks if qtoks is not None else zh_tokenize(q))
        s = self._z._score_active(qtoks) if s is None else s
        bm = self._z._bm
        n = int(bm.N) or 1
        n_active = int(self._z._active_rows().size)
        terms, oov = [], []
        for t, c0 in Counter(qtoks).items():
            ci = bm.term2idx.get(t)
            if ci is None:
                oov.append(t)
                terms.append(dict(term=t, q_count=int(c0), in_vocab=False,
                                  df=0, df_ratio=None, idf=None))
                continue
            df = int(bm.df[ci])
            idf = float(np.log((n - df + 0.5) / (df + 0.5) + 1.0))
            terms.append(dict(term=t, q_count=int(c0), in_vocab=True, df=df,
                              df_ratio=round(df / n, 6), idf=round(idf, 6)))
        n_pos = int((s > 0).sum()) if s.size else 0
        if not qtoks:
            status = "no_tokens"
        elif not any(t["in_vocab"] for t in terms):
            status = "all_oov"
        elif n_pos == 0:
            status = "no_positive"
        elif oov:
            status = "partial_oov"
        else:
            status = "ok"
        msgs = []
        if status == "no_tokens":
            msgs.append(dict(level="warn", what="这句话里没有能用来检索的内容",
                             evidence=dict(tokens=qtoks),
                             action="换一句更具体的话"))
        if n_active == 0:
            msgs.append(dict(level="warn", what="索引里没有活跃文档",
                             evidence=dict(N=int(bm.N), n_active=0),
                             action="先入库再查"))
        if oov:
            msgs.append(dict(level="warn" if status == "all_oov" else "info",
                             what=f"这 {len(oov)} 个词你的资料库里没有：" + "、".join(oov),
                             evidence=dict(oov=oov, V=int(bm.V)),
                             action="补一份含这些词的资料，或换成库里已有的说法"))
        if status == "no_positive":
            msgs.append(dict(level="warn", what="库里有这些词，但没有任何一条文档同时命中",
                             evidence=dict(n_positive=0, terms=[t["term"] for t in terms
                                                              if t["in_vocab"]]),
                             action="删掉一两个限定词再试"))
        # 低区分度词：df_ratio 是实数，直接可举证；0.5 是经验阈值（不是统计结论）
        low = [t for t in terms if t["df_ratio"] is not None and t["df_ratio"] >= 0.5]
        if low:
            msgs.append(dict(level="info",
                             what="这些词库里多数文档都有，帮你筛不出东西："
                                  + "、".join(x["term"] for x in low),
                             evidence=dict(df_ratio={x["term"]: x["df_ratio"] for x in low},
                                           threshold=0.5),
                             action="换成更具体的词"))
        if n_pos > max(int(k), 1):
            msgs.append(dict(level="info",
                             what=f"库里有 {n_pos} 条沾边，只细看了前 {max(int(k), 1)} 条",
                             evidence=dict(n_positive=n_pos, k=max(int(k), 1)),
                             action="要更多结果就加大 k"))
        return dict(query=q, tokens=qtoks, n_tokens=len(qtoks), terms=terms,
                    vocab=dict(V=int(bm.V), N=int(bm.N), n_active=n_active,
                               avgdl=float(bm.avgdl)),
                    scores=dict(k=max(int(k), 1),
                                max=float(s.max()) if s.size else 0.0,
                                n_positive=n_pos),
                    status=status, messages=msgs)

    def diagnose(self, q: str, k: int = 10) -> dict:
        """诊断：**为什么没搜到** / 为什么只搜到这些 —— 产品侧入口（人话 + 可举证）。

        与 `explain()` 分工：`explain` 拆「某一篇为什么得分高」，`diagnose` 看「整次查询的
        处境」（词在不在库里、沾边有多少、哪些词没区分度）。两者用同一份分数数组口径
        （`_score_active`），不另算一遍公式。
        """
        with self._lock.read():
            return self._diagnose(q, k=k)

    # ---------------------------------------------------------------- 写
    def add(self, root: str, *, workers: int = 0, verify: str = "auto") -> dict:
        """增量：只处理改动过的文件，走 `Primolix.update()` 的路由（见 `04c` 判据件）。

        四个出口（`kind`）：`noop` / `incremental` / `staged` / `rebuild`。
        `verify`：`"auto"`（默认，有 size/mtime 印记走 stat 快扫）/ `"stat"` / `"full"`；
        返回里带 `verify`，如实说明这一批校验到了什么程度。
        段在场时：含新词 → 还能再挂一段（`staged`）；词表内 → 先自动折叠再增量。
        """
        with self._lock.write():
            return self._z.update(root, workers=workers, verify=verify)

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
        """压实：把已挂段全部折进主表 ＋ **落盘** ＋ 清分块缓存 → 查询恢复单块。

        查询耗时随分块数上升，长期只追加会把块数堆起来；折叠是一次 `O(nnz)`，折完只走主表。
        内核不变量 `fold_segment` 断言"折完 T 行数 == N"（`N` 含所有已挂段）→ 多段时用不了
        `fold_segment`，这里改走 `fold_all()`。

        必须先 `_persist()` 再写 `folded_all` 事件（2026-10-18 修）：折叠改的是内存里的 `T`，
        段的内容不在源语料里 ⇒ 不落盘就写提交事件，重载时 npz 里没有那些行、旁车却说"已折叠"
        ⇒ 段内容静默丢失（`_meta` 也一起丢）。顺序落盘→提交：崩溃只留"已落盘、旁车仍说在挂"，
        重载时 `remount_staged` 会因 `T.shape[0] > row_base` 判为已折叠而跳过 ⇒ 安全。
        """
        with self._lock.write():
            out = self._z.compact_segments()      # 与 update() 的 `fold_first` 出口共用
            return dict(folded=out["folded"], blocks=out["blocks"], N=out["N"],
                        V=out["V"], ms=out["ms"], via=out["via"],
                        tags=out["tags"])

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
        """稀疏 ＋ 稠密融合打分（覆盖行分数单调不降），返回 top-k。

        01m（P-B）：三处换成缓存/单趟 —— ① `_active_rows()`（不再每次用 Python
        列表推导重建行号数组：@1M 那是 70.28 ms/次 的写法）；② 行分只 gather 一次
        （改前每返回一行都再 `s[rows[j]]` 一次）；③ `_topk_idx`（P-O3：采样定阈 → 候选内
        `argpartition`；`TOPK_APPROX=False` 时退回整段 `argpartition`）替代
        全量 `argsort`。并列时取哪几行本就不保证（`_topk_idx` 契约），与 `core.query`
        口径一致；判据盯的是选出的 k 个分数值多重集逐位相同。
        """
        with self._lock.read():
            z = self._z
            s = z._bm.fused_scores(zh_tokenize(q), qvec, view_name, alpha=alpha)
            rows = z._active_rows()
            sv = s[rows]                              # ⭐ 只取一次
            ranked = _topk_idx(sv, k)
            out = []
            for i, j in enumerate(ranked, 1):
                m = z._meta[int(j)]
                out.append(dict(rank=i, score=float(sv[j]), file=m["file"],
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
                stream=False,
                model_path: str = "") -> Index:
    r"""建索引并返回门面对象。`w_cache=True` → 建库时把物化层一并落盘。

    `stream` 只改建库峰值内存、不改分数：`False`（默认）· `True` · `"ids"`（最低档、不重分词）。
    实测 100k 档 net 峰值 607 / 216 / 234 MB ⇒ 差 2.6–2.8×。
    """
    z = Primolix.build(root, out=out, dense=dense, model_path=model_path,
                       workers=workers, w_cache=w_cache, stream=stream)
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

    def add(self, docid, text_or_tokens, *, file=None, line=None) -> "Stage":
        """加入一篇：文本（分词一次）或已分好的 token 列表。返回 `self`（可链式）。

        `file` / `line` 可选：段发布时它们会进段级文档身份侧车（`.segments/<tag>/docs.jsonl`），
        从而在产品层 `search`/`browse` 可见。缺省时用 `docid` 当 `file`、`line=1`（不编造路径）。
        `text` 记原文（传 token 列表时退化为空格连接，会丢失原文间距）。
        """
        raw = text_or_tokens
        toks = (zh_tokenize(raw) if isinstance(raw, str) else list(raw))
        for t in toks:
            if self._has(t) or t in self._new_pos:
                continue
            self._new_pos[t] = self.base_V + len(self._new_terms)
            self._new_terms.append(t)
        self._rows.append((docid, toks, file, line,
                           raw if isinstance(raw, str) else " ".join(toks)))
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
        r["tokens_pending"] = int(sum(len(t) for _, t, _f, _l, _x in self._rows))
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
        idx = self.idx
        with idx._lock.write():
            out = publish_segment(idx._z, list(self._rows), tag=self.tag,
                                  generation=self.generation,
                                  base_V=self.base_V, base_T_rows=self.base_T_rows,
                                  new_terms=list(self._new_terms),
                                  new_pos=self._new_pos, t2i=self._t2i, fold=fold)
            idx._stages.discard(self)
            self._rows, self._new_terms, self._new_pos = [], [], {}
            return out
