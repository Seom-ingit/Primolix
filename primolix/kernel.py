#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import hashlib
import math  # noqa: F401
import multiprocessing
import os
import re
import sys
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from functools import wraps

import numpy as np

try:
    from scipy.sparse import csr_matrix, csc_matrix, vstack as _sp_vstack
except ImportError:  # pragma: no cover - 仅当 scipy 缺失时
    csr_matrix = None
    csc_matrix = None
    _sp_vstack = None

LOG_PATH = None

__all__ = [
    "SparseBM25", "zh_tokenize", "simple_tokenize", "tokenizer_status",
    "log", "auto_workers", "ViewError",
]


def auto_workers(cap=16, per_worker_mb=200):
    """内存感知的并行 worker 数: min(CPU 核数, 可用内存/每 worker 开销, cap)。

    spawn 场景每个 worker ~200MB (Python 运行时 + numpy + jieba dict + chunk
    序列化缓冲)。无 psutil 时回退 CPU-only。
    """
    cores = os.cpu_count() or 1
    try:
        import psutil
        avail_mb = psutil.virtual_memory().available / 2 ** 20
        by_mem = max(int(avail_mb / per_worker_mb), 1)
    except Exception:
        by_mem = cores
    return max(min(cores, by_mem, cap), 1)


def warn(msg):
    r"""附加件失败必须留痕：与 `log` 不同，它不受 `core.VERBOSE` 静音。

    附加件（`.mm/` / 词表载体 / `w`）失败至少要留一行，且与 verbose 无关。
    """
    try:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, file=sys.stderr, flush=True)
    except Exception:
        pass


def log(msg):
    # 作为库使用时保持安静：core.VERBOSE 为 False（库的默认）就不往 stderr 打进度。
    # 命令行入口会把 core.VERBOSE 置 True，所以 CLI 照旧有输出。
    try:
        from . import core as _core
        if not getattr(_core, "VERBOSE", True):
            return
    except Exception:
        pass
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, file=sys.stderr, flush=True)
    if LOG_PATH:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")


# ======================== 工具 ========================

# CJK 统一表意文字 + 扩展 A + 兼容表意 + 日文假名 + 谚文。
_CJK = (r'\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff'
        r'\u3040-\u30ff\uac00-\ud7af')
# 交替的左侧优先是关键: 每个 CJK 字符单独成一个 unit, 其余走 \w+。
_UNIT_RE = re.compile(f'[{_CJK}]|\\w+')
_IS_CJK_RE = re.compile(f'[{_CJK}]')


def simple_tokenize(text):
    units = _UNIT_RE.findall(text.lower())
    out = []
    prev = None
    for u in units:
        if len(u) == 1 and _IS_CJK_RE.fullmatch(u):
            out.append(u)                 # CJK 单字
            if prev is not None:
                out.append(prev + u)      # 相邻二元（提供判别力）
            prev = u
        else:
            out.append(u)
            prev = None
    return out


_JIEBA = None
_TOKENIZER = {'name': 'unknown', 'degraded': False}


def tokenizer_status():
    if _JIEBA is None:
        zh_tokenize('')          # 触发一次探测（空串成本可忽略）
    return dict(_TOKENIZER)


def zh_tokenize(text):
    global _JIEBA
    if _JIEBA is None:
        try:
            import jieba
            jieba.setLogLevel(60)
            _JIEBA = jieba
            _TOKENIZER.update(name='jieba', degraded=False)
        except ImportError:
            log("[WARN] jieba 未安装 → 中文分词降级为 simple_cjk_bigram"
                "(字+二元, 可用但精度低于 jieba)。"
                " 该降级必须向上层可见, 见 primolix.kernel.tokenizer_status()")
            _JIEBA = False
            _TOKENIZER.update(name='simple_cjk_bigram', degraded=True)
    if _JIEBA:
        return [w.lower() for w in _JIEBA.cut(text) if w.strip()]
    return simple_tokenize(text)


def full_hash(text):
    return hashlib.md5(text.encode('utf-8')).hexdigest()


# ======================== S1: scipy 稀疏 BM25 ========================

_TOK = None  # 并行分词 worker 共享


class ViewError(ValueError):
    """派生视图（接口 C）的用法错误 —— 名字冲突 / 行号非法 / 维度不匹配。"""


def _mp_ctx():
    """跨平台 multiprocessing context: fork(继承内存) / Windows spawn。"""
    try:
        return multiprocessing.get_context('fork')
    except ValueError:
        return multiprocessing.get_context('spawn')


def _tok_init(tok):
    global _TOK
    _TOK = tok


def _tok_chunk(docs):
    return [_TOK(d) for d in docs]


def _chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def parallel_tokenize(docs, tok, workers, chunksize=50000):
    """多进程并行分词（8.84M 段内存安全: 按 5 万段分 task 流式回传）"""
    ctx = _mp_ctx()
    out = []
    with ctx.Pool(workers, initializer=_tok_init, initargs=(tok,)) as pool:
        for part in pool.map(_tok_chunk, _chunks(docs, chunksize)):
            out.extend(part)
    return out


def _chunks_with_idx(lst, n):
    for i in range(0, len(lst), n):
        yield i, lst[i:i + n]


def _tok_chunk_idx(arg):
    start, docs = arg
    return start, [_TOK(d) for d in docs]


def parallel_tokenize_stream(docs, tok, workers, chunksize=50000):
    """多进程流式分词: imap 按块回传 (start, toks), 边分边处理。"""
    ctx = _mp_ctx()
    with ctx.Pool(workers, initializer=_tok_init, initargs=(tok,)) as pool:
        for start, part in pool.imap(_tok_chunk_idx,
                                     _chunks_with_idx(docs, chunksize),
                                     chunksize=1):
            yield start, part


def _stream_scan(docs, tok, workers, on_doc):
    """遍历全部 docs 分词, 回调 on_doc(i, toks); workers>1 时并行 (失败回退单线程)。"""
    if workers > 1:
        try:
            for start, part in parallel_tokenize_stream(docs, tok, workers):
                for k, toks in enumerate(part):
                    on_doc(start + k, toks)
            return
        except Exception as e:
            log(f"[WARN] S1 流式并行分词失败({e}), 回退单线程")
    for i, d in enumerate(docs):
        on_doc(i, tok(d))


class PreallocCSR:

    def __init__(self, V, n_rows_cap, nnz_cap):
        self.V = V
        self.nnz_cap, self.n_rows_cap = int(nnz_cap), int(n_rows_cap)
        self.data_cap = np.zeros(self.nnz_cap, dtype=np.float32)
        self.idx_cap = np.zeros(self.nnz_cap, dtype=np.int32)
        self.ptr_cap = np.zeros(self.n_rows_cap + 1, dtype=np.int32)
        self.nnz = self.n_rows = 0
        self.grow_count = 0

    @classmethod
    def from_csr(cls, T, slack=1.5):
        """一次性换后端: 拷贝三元组进预留容量 (O(nnz), 只在换后端时付一次)。

        - 排序不变量在这里一次性钉死 (已排序则只置 flag, 否则原地排一次)
        - int32 预检: nnz ≥ 2^31 直接拒绝 (静默 wrap 比报错可怕, 见 dtype 审计)
        """
        if T.data.dtype != np.float32:
            raise TypeError(f"PreallocCSR 只接受 float32 的 tf (got {T.data.dtype})")
        if T.nnz and int(T.indptr[-1]) > 2 ** 31 - 1:
            raise OverflowError("nnz ≥ 2^31: int32 ptr_cap 预检失败, 拒绝预分配后端")
        m = cls(T.shape[1], max(int(T.shape[0] * slack) + 8, 8),
                max(int(T.nnz * slack) + 8, 8))
        m.nnz, m.n_rows = int(T.nnz), int(T.shape[0])
        m.data_cap[:m.nnz] = T.data
        m.idx_cap[:m.nnz] = T.indices
        m.ptr_cap[:m.n_rows + 1] = T.indptr
        chk = csr_matrix((m.data_cap[:m.nnz], m.idx_cap[:m.nnz],
                          m.ptr_cap[:m.n_rows + 1]), shape=(m.n_rows, m.V))
        chk.sort_indices()          # 已排序时是 no-op (只置 flag); 否则原地排 (写回 cap 视图)
        if not chk.has_sorted_indices:
            raise ValueError("from_csr: T 行内 indices 无序且排序失败 —— 拒绝换后端")
        return m

    def _grow(self, need_nnz=0, need_rows=0):
        r"""扩容。支持"目标制"：`need_nnz/need_rows` 给出本次要装下的量，一次到位。

        只按「当前 `nnz` × 2」扩会让"一次追加一整片"装不下；不传目标时行为与原来一致（×2）。
        """
        self.grow_count += 1
        nnz_cap = max(int(self.nnz * 2) + 8, self.nnz + 8, int(need_nnz) + 8)
        rows_cap = max(int(self.n_rows * 2) + 8, self.n_rows + 8, int(need_rows) + 8)
        d = np.zeros(nnz_cap, dtype=np.float32)
        i = np.zeros(nnz_cap, dtype=np.int32)
        p = np.zeros(rows_cap + 1, dtype=np.int32)
        d[:self.nnz] = self.data_cap[:self.nnz]
        i[:self.nnz] = self.idx_cap[:self.nnz]
        p[:self.n_rows + 1] = self.ptr_cap[:self.n_rows + 1]
        self.data_cap, self.idx_cap, self.ptr_cap = d, i, p
        self.nnz_cap, self.n_rows_cap = nnz_cap, rows_cap

    def _sort_new_rows(self, base, n_new):
        """新写入块内逐行局部排序 —— O(新行 nnz)。

         必须传 ``base`` (旧行数), 不能读 self.n_rows: 调用点若已自增,
        会读到写区之外, ``hi-lo>1`` 恒假, 静默不排序, 而"结构等价"断言照样通过。
        """
        p, idx, dat = self.ptr_cap, self.idx_cap, self.data_cap
        for i in range(n_new):
            lo, hi = int(p[base + i]), int(p[base + i + 1])
            if hi - lo > 1:
                order = np.argsort(idx[lo:hi], kind="stable")
                idx[lo:hi] = idx[lo:hi][order]
                dat[lo:hi] = dat[lo:hi][order]

    def wrap(self):
        """交视图给 csr_matrix (O(1))。排序不变量由 from_csr/_sort_new_rows 保证。"""
        m = csr_matrix((self.data_cap[:self.nnz],
                        self.idx_cap[:self.nnz],
                        self.ptr_cap[:self.n_rows + 1]),
                       shape=(self.n_rows, self.V))
        m.has_sorted_indices = True   # 重包默认 False —— 不置位则首个消费方会重排
        return m

    def append(self, rows, cols, tfs, counts):
        """追加 n_new=len(counts) 行 (行可为空: 对应 count=0)。

        ``rows`` 只取长度 (行号由 indptr 隐含); ``counts[i]`` = 第 i 行的 posting 数。
        """
        n_new = len(counts)
        need = self.nnz + len(rows)
        if need > self.nnz_cap or self.n_rows + n_new > self.n_rows_cap:
            self._grow(need, self.n_rows + n_new)     # 目标制：一次到位（大块追加, 见 `_grow` docstring）
        start = self.nnz
        if len(rows):
            self.data_cap[start:need] = tfs
            self.idx_cap[start:need] = cols
        p, base, off = self.ptr_cap, self.n_rows, start
        for i, c in enumerate(counts):
            p[base + 1 + i] = off + int(c)
            off += int(c)
        p[base + 1 + n_new] = start + len(rows)
        self._sort_new_rows(base, n_new)   # 必须早于/独立于 n_rows 自增
        self.nnz, self.n_rows = need, base + n_new


class RWLock:
    r"""可重入读写锁：读者并发 · 写者独占 · 写者优先 · 同线程可重入。

    内核里存在"读里套读"（`fused_scores` → `score_all`）与"写里套写"
    （`update_docs` → `_tombstone_rows`/`append_docs`），普通读写锁会自死锁。
    不支持"读侧里取写侧"（必然自死锁）：显式抛错，不静默挂住。
    """

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._readers = 0                     # 当前持读侧的线程数（非重入次数）
        self._writer = False
        self._waiting_writers = 0
        self._reader_depth: dict[int, int] = {}
        self._writer_owner = None
        self._writer_depth = 0

    @contextmanager
    def read(self):
        me = threading.get_ident()
        with self._cond:
            if self._reader_depth.get(me, 0) > 0:            # 同线程重入：直接进
                self._reader_depth[me] += 1
            else:
                while self._writer or self._waiting_writers > 0:
                    self._cond.wait()
                self._reader_depth[me] = 1
                self._readers += 1
        try:
            yield
        finally:
            with self._cond:
                d = self._reader_depth.get(me, 0) - 1
                if d <= 0:
                    self._reader_depth.pop(me, None)
                    self._readers -= 1
                    if self._readers == 0:
                        self._cond.notify_all()
                else:
                    self._reader_depth[me] = d

    @contextmanager
    def write(self):
        me = threading.get_ident()
        with self._cond:
            if self._writer_owner == me:                     # 同线程重入
                self._writer_depth += 1
            else:
                if self._reader_depth.get(me, 0) > 0:
                    raise RuntimeError("RWLock：不支持在读侧里取写侧（必然自死锁）")
                self._waiting_writers += 1
                try:
                    while self._writer or self._readers:
                        self._cond.wait()
                    self._writer = True
                    self._writer_owner = me
                    self._writer_depth = 1
                finally:
                    self._waiting_writers -= 1
        try:
            yield
        finally:
            with self._cond:
                self._writer_depth -= 1
                if self._writer_depth <= 0:
                    self._writer = False
                    self._writer_owner = None
                    self._cond.notify_all()

    def report(self) -> dict:
        with self._cond:
            return dict(readers=int(self._readers), writer=bool(self._writer),
                        waiting_writers=int(self._waiting_writers))


def _with_read(fn):
    """给查询侧方法套上读临界区（读者并发；`SWAP_LOCK=False` 时零开销直通）。"""
    @wraps(fn)
    def wrapper(self, *a, **k):
        with self._read():
            return fn(self, *a, **k)
    return wrapper


def _with_write(fn):
    """给换段/写入侧方法套上写临界区（独占；同线程可重入）。"""
    @wraps(fn)
    def wrapper(self, *a, **k):
        with self._write():
            return fn(self, *a, **k)
    return wrapper


class SparseBM25:
    """两遍扫描构建 scipy CSR BM25 权重矩阵（8.84M 段内存友好）:

    并行分词一次缓存（workers>1 时 fork 并行），pass1 建词表 / pass2 收集
    三元组（不再重分词）。``stream=True`` 时两遍各自分词、不驻留 doc_toks，
    内存从 O(N×len) 降到 O(三元组)，代价是分词两次。
    W[d,t] = idf[t] * tf*(k1+1) / (tf + k1*(1-b + b*dl/avgdl))
    """

    T_SLACK = 1.5   # 01e 容量 slack: data/indices 预留倍率 (内存↔摊还对价见 NOTES_01e 第 2.3 节)
    #    True = 建块0 时就走 `T.tocsc()`；False = 推到首查由 `_ensure_tc()` 建（省一份私有 postings）。
    EAGER_TC = True
    #    True 时经 `primolix/tc_shared.py` 让多进程共享同一份 CSC（默认 False，行为不变）。
    TC_SHARED = False
    FAST_SCORE = True
    #    `FAST_SCORE` 是 `DN_PRECOMP`/`ADD_AT`/`W_CACHE` 的总闸：不开它那三个开关在默认路径上全部失效。
    #    逐位：只做"不改变浮点运算顺序"的改动；`WRITE_DICT` 的收益：写路径逐词查找 5.163µs → 0.214µs。
    WRITE_DICT = False
    # 01f：预计算"每行规范化项" `dn[d] = k1(1-b+b·dl/avgdl)`（N 级，1M 段 ≈4 MB），
    #      查询期省掉每 posting 一次 `dl[rows]` gather ＋ 3 次逐元素运算。
    #      表达式与现算逐字同形, 逐位等价；只在 `FAST_SCORE=True` 的路径里生效。
    DN_PRECOMP = True
    # 01h：物化层 `w`（缓存，opt-in，默认关，不改默认行为）——
    #      命中时每 posting 一次 gather ＋ 一次 scatter-add（≈bm25s 形态）；
    #      键不符即重建（O(nnz)，1M 段 ≈0.5 s）；不做也不亏（未命中走现算 ≈4.4 ms/查询）。
    W_CACHE = False
    # 01i：累加写法（opt-in，默认关，不改默认行为）——
    #      `np.add.at(scores, rows, x)`（无缓冲）vs `scores[rows] += x`（要缓冲）。
    #      同一列内行号不重复, 无缓冲反而省掉缓冲与两遍读写；逐位相同。
    ADD_AT = True
    # 01g：M1 手写矩阵化（opt-in，默认关，不改默认行为）——
    #      一次拼块 ＋ 一次 φ ＋ 一次 bincount，遍历 8–10 次 → 3–4 次；
    #      `bincount` 在 float64 里累加且顺序不同, 可能与快路径差 ~1 ulp。
    MATVEC = False
    # 并发安全（默认开）：换段与查询之间用可重入读写锁互斥。
    #      无锁并发换段会撕裂读数（静默错分）；代价是每查询多两次轻量加解锁（~1–2 µs）。
    #      单线程可设 `SWAP_LOCK=False` 拿回开销（那时并发换段不安全）。
    SWAP_LOCK = True

    def __init__(self, docs, tok, k1=1.5, b=0.75, workers=0, stream=False, vocab=None):
        if csr_matrix is None:
            raise RuntimeError("scipy 未安装: pip install scipy")
        # 并发原语（`SWAP_LOCK` 的载体）：换段取写侧、查询取读侧；派生缓存另有互斥。
        self._rw = RWLock()
        self._cache_mu = threading.RLock()
        self.k1 = k1
        self.b = b
        self.N = len(docs)
        self.dl = np.zeros(self.N, dtype=np.float32)
        self.T = None          # CSR tf 矩阵 (fp32); 首次 append 起由 _Tbuf 提供视图
        self._Tbuf = None      # PreallocCSR 后端 (None = 还是普通 csr, 首 append 时换)
        self._tc_blocks = []   # 01e-C 分块 CSC: [(csc_matrix, row_base), ...]
        self._tc_rows = 0      # 已覆盖行数 (append 后 < T 行数, 首查补一块)
        self.W = self.Wc = None  # 01d: 新索引不构建/不落盘权重矩阵 (导出物)
        self.df = None         # np.int32[V] 存活文档计数
        self._idf = None       # np.float32[V] 现算缓存 (df/N_live 驱动)
        self._dead = None      # bool[N] 墓碑 (None=全活); append 时扩展
        # 视图 = BM25 权威层之上可丢可滞后可分版的派生层（覆盖位图 + 视图可用性）。
        # 删掉全部视图时查询逐位回 BM25-only；视图不持久化（save/load 都不写它），
        # 派生层可随时由 sidecar 重挂。
        self._views = {}
        t0 = time.time()
        term2idx, idx2term = {}, []
        # 01k：给了 `vocab`（列序的 term 列表）就预置列空间，各片共用同一套列号。
        #    不许悄悄扩表：构建完若 `len(idx2term) != len(vocab)` 就抛错（不静默丢/不静默加）。
        if vocab is not None:
            idx2term = [str(t) for t in vocab]
            term2idx = {t: i for i, t in enumerate(idx2term)}
        _df_by_id = (stream == "ids")
        if stream == "ids":
            # 01j：分词一次 ＋ 立即转 id ＋ 只留 int32（不缓存 Python 字符串）——
            #    目标：时间 ≈ 非流式（分词只做一遍）而内存 ≈ 流式
            #    （`41M token × 4 B = 165 MB` vs 缓存字符串 ≈ 2 GB）。
            doc_ids = []
            for _i, _d in enumerate(docs):
                _toks = tok(_d)
                self.dl[_i] = len(_toks)
                _arr = np.empty(len(_toks), dtype=np.int32)
                for _j, _t in enumerate(_toks):
                    _c = term2idx.get(_t)
                    if _c is None:
                        _c = len(idx2term)
                        term2idx[_t] = _c
                        idx2term.append(_t)
                    _arr[_j] = _c
                doc_ids.append(_arr)
            V = len(idx2term)
            log(f"S1 词表完成(ids 模式) N={self.N} V={V} {time.time()-t0:.1f}s")
            rows, cols, data = [], [], []
            df = defaultdict(int)
            for _i, _arr in enumerate(doc_ids):      # 本趟不再分词
                _tf = {}
                for _c in _arr.tolist():
                    _tf[_c] = _tf.get(_c, 0) + 1
                for _c, _n in _tf.items():
                    rows.append(_i); cols.append(_c); data.append(_n); df[_c] += 1
            del doc_ids
        elif stream:
            # ===== 流式: 两遍各自分词, 不驻留 doc_toks (并行可选, 用满多核) =====
            log(f"S1 流式分词 (stream): 两遍扫描省内存, {self.N} 篇"
                f"{f' ×{workers} 并行' if workers > 1 else ''}...")

            def _idx_doc(i, toks):
                self.dl[i] = len(toks)
                for t in set(toks):
                    if t not in term2idx:
                        term2idx[t] = len(idx2term)
                        idx2term.append(t)
            _stream_scan(docs, tok, workers, _idx_doc)
            V = len(idx2term)
            log(f"S1 词表完成 N={self.N} V={V} {time.time()-t0:.1f}s")
            rows, cols, data = [], [], []
            df = defaultdict(int)

            def _tri_doc(i, toks):
                tf = {}
                for t in toks:
                    tf[t] = tf.get(t, 0) + 1
                for t, c in tf.items():
                    rows.append(i)
                    cols.append(term2idx[t])
                    data.append(c)
                    df[t] += 1
            _stream_scan(docs, tok, workers, _tri_doc)
        else:
            # ===== 常规: 分词一次缓存 (省时间, 内存 O(N×len)) =====
            if workers > 1:
                try:
                    doc_toks = parallel_tokenize(docs, tok, workers)
                    log(f"S1 并行分词完成 {self.N} 篇 ({workers} workers) {time.time()-t0:.1f}s")
                except Exception as e:
                    log(f"[WARN] 并行分词失败({e}), 回退单线程")
                    doc_toks = [tok(d) for d in docs]
            else:
                doc_toks = [tok(d) for d in docs]
            # pass1: 词表（复用 doc_toks）
            for i, toks in enumerate(doc_toks):
                self.dl[i] = len(toks)
                for t in set(toks):
                    if t not in term2idx:
                        term2idx[t] = len(idx2term)
                        idx2term.append(t)
            V = len(idx2term)
            log(f"S1 词表完成 N={self.N} V={V} {time.time()-t0:.1f}s")
            # pass2: 收集 tf 三元组（复用 doc_toks）
            rows, cols, data = [], [], []
            df = defaultdict(int)
            for i, toks in enumerate(doc_toks):
                tf = {}
                for t in toks:
                    tf[t] = tf.get(t, 0) + 1
                for t, c in tf.items():
                    rows.append(i)
                    cols.append(term2idx[t])
                    data.append(c)
                    df[t] += 1
            del doc_toks
        if vocab is not None and len(idx2term) != len(vocab):
            raise ValueError(
                f"给了词表（{len(vocab)} 词），但语料里出现了 {len(idx2term) - len(vocab)} 个表外词 "
                f"→ 拒绝静默扩表（分片建库要求列空间一致）；请先把表外词并进词表，或显式不要传 `vocab`")
        self.term2idx = term2idx
        self.idx2term = idx2term
        t1 = time.time()
        log(f"S1 三元组收集完成 {len(data)} 项 {t1-t0:.1f}s (nnz≈{len(data)//1000000}M)")
        rows = np.asarray(rows, dtype=np.int32)
        cols = np.asarray(cols, dtype=np.int32)
        data = np.asarray(data, dtype=np.float32)   # 注意: data = tf (词频计数)
        self.avgdl = float(self.dl.sum()) / max(self.N, 1)
        # ---- INCR: 权威原语 (tf/df/dl), 供真增量 + 现算路径 ----
        self.df = (np.array([df.get(_c, 0) for _c in range(V)], dtype=np.int32) if _df_by_id
                   else np.array([df.get(t, 0) for t in idx2term], dtype=np.int32))
        self._idf = np.log((self.N - self.df + 0.5) / (self.df + 0.5) + 1).astype(np.float32)
        self.T = csr_matrix((data, (rows, cols)), shape=(self.N, V)).tocsr()
        # ---- 01e-A: 换容量预分配后端 (一次性 O(nnz) 拷贝, 计入建索引成本) ----
        self._Tbuf = PreallocCSR.from_csr(self.T, slack=self.T_SLACK)
        self.T = self._Tbuf.wrap()
        # ---- 01e-C: 分块 CSC —— 块0 覆盖全部初始行 (与旧 T.tocsc() 等价) ----
        self._tc_blocks = [(self.T.tocsc(), 0)]
        self._tc_rows = int(self.T.shape[0])
        # ---- 01d: 不再构建 W/Wc —— 权重是导出物, 查询走 _score_incr (分块 CSC) ----
        #      省 2/4 份常驻 (各 21MB 量级) + save 不再付 _rebuild_w 的 O(nnz)
        self.W = self.Wc = None
        log(f"S1 BM25 稀疏矩阵就绪 nnz={self.T.nnz} 总耗时 {time.time()-t0:.1f}s")

    # ---------------- 并发：读侧 / 写侧（`SWAP_LOCK=False` 时零开销直通） ----------------

    @contextmanager
    def _read(self):
        """查询侧临界区（读者并发）。`SWAP_LOCK=False` 时不加锁（那时并发换段不安全）。"""
        if type(self).SWAP_LOCK:
            with self._rw.read():
                yield
        else:
            yield

    @contextmanager
    def _write(self):
        """换段/写入侧临界区（独占）。同线程可重入（`update_docs` 会套 `append_docs`）。"""
        if type(self).SWAP_LOCK:
            with self._rw.write():
                yield
        else:
            yield

    def _score_incr(self, idxs):
        """现算路径 (01d 后新索引的唯一路径; 旧格式含 W 时走 score_all 的兼容分支)。

        norm = tf(k1+1)/(tf + k1(1-b + b·dl/avgdl)) 只依赖 posting 自身 → 严格等于全量公式。
        01e-C: 查询用分块 CSC —— append 后只补一块 O(新 nnz), 旧块从不失效;
        分块不改每文档的累加顺序 (外层按 term, 文档只属一块), 故与单体 Tc 路径逐位一致。
        """
        self._ensure_tc()
        scores = np.zeros(self.N, dtype=np.float32)
        dl, avg, k1, b = self.dl, self.avgdl, self.k1, self.b
        dead = self._dead
        idf = self._idf
        for c in idxs:
            for csc, rb in self._tc_blocks:
                if c >= csc.shape[1]:      # 段挂载后各块列宽不同：该块没有这一列, 跳过
                    continue
                col = csc.getcol(c)
                tf = col.data
                if tf.size == 0:
                    continue
                rows = col.indices
                if rb:
                    rows = rows + rb
                if dead is not None:
                    m = ~dead[rows]
                    if not m.any():
                        continue
                    rows = rows[m]
                    tf = tf[m]
                norm = tf * (k1 + 1.0) / (tf + k1 * (1.0 - b + b * dl[rows] / avg))
                scores[rows] += idf[c] * norm
        return scores

    def _ensure_dn(self):
        r"""预计算"每行规范化项"（01f，opt-in）：

            dn[d] = k1 * (1.0 - b + b * dl[d] / avgdl)          # 与现算逐字同形（逐位的前提）

        省掉查询期对每个 posting 都要做的 `dl[rows]` gather ＋ 三次逐元素运算;
        空间落在 N 级（1M 段 ≈ 4 MB）。只在 `k1/b/avgdl/dl` 不变时有效，故用 key 兜住（变了就重建）。
        """
        key = (float(self.k1), float(self.b), float(self.avgdl), id(self.dl), int(self.dl.size))
        if getattr(self, "_dn", None) is not None and getattr(self, "_dn_key", None) == key:
            return self._dn
        with self._cache_mu:                     # 并发：`dn` 也是派生缓存（双检）
            if getattr(self, "_dn", None) is not None and getattr(self, "_dn_key", None) == key:
                return self._dn
            dl, avg, k1, b = self.dl, self.avgdl, self.k1, self.b
            self._dn = k1 * (1.0 - b + b * dl / avg)     # 表达式与 `_score_incr_fast` 里逐字相同
            self._dn_key = key
            return self._dn

    def _score_matvec(self, idxs):
        r"""M1 手写矩阵化（opt-in）：一次拼出 K 列的 `rows`/`tf`/`w`，在整块上算一次
        `norm = tf(k1+1)/(tf + dn[rows])`，再一次 `np.bincount(rows, weights=w*norm, minlength=N)`
        完成累加（遍历 8–10 次 → 3–4 次）。

        `np.bincount` 在 float64 里累加、且顺序与"按词按块"不同，结果可能与快路径差 ~1 ulp
        —— 这是显式的口径取舍。
        """
        self._ensure_tc()
        N = int(self.N)
        dl, avg, k1, b = self.dl, self.avgdl, self.k1, self.b
        dead, idf = self._dead, self._idf
        rp, tp, wp = [], [], []
        for c in idxs:
            for csc, rb in self._tc_blocks:
                if c >= csc.shape[1]:
                    continue
                s = int(csc.indptr[c]); e = int(csc.indptr[c + 1])
                if e <= s:
                    continue
                rows = csc.indices[s:e]
                tf = csc.data[s:e]
                if rb:
                    rows = rows + rb
                if dead is not None:
                    keep = ~dead[rows]
                    if not keep.any():
                        continue
                    rows = rows[keep]; tf = tf[keep]
                rp.append(rows); tp.append(tf)
                wp.append(np.full(rows.size, idf[c], dtype=np.float32))   # 每词的 idf 广播到它的 postings 上
        if not rp:
            return np.zeros(N, dtype=np.float32)
        rows = np.concatenate(rp); tf = np.concatenate(tp); w = np.concatenate(wp)
        del rp, tp, wp
        if type(self).DN_PRECOMP:
            dn = self._ensure_dn()
            norm = tf * (k1 + 1.0) / (tf + dn[rows])
        else:
            norm = tf * (k1 + 1.0) / (tf + k1 * (1.0 - b + b * dl[rows] / avg))
        out = np.bincount(rows, weights=(w * norm), minlength=N)          # 一次累加（float64）
        return out[:N].astype(np.float32)

    def _w_key(self):
        """`w` 缓存键：参数/统计/块结构任一变化都必须改键（否则会静默用旧 `w`, 是最危险的错）。"""
        return (float(self.k1), float(self.b), float(self.avgdl), int(getattr(self, "_gen", 0)),
                int(self.N), len(self._tc_blocks))

    def _w_persist_key(self):
        r"""内容型缓存键（01h-持久化）：跨会话可比, 用它判断盘上的 `w` 还能不能用。

        与 `_w_key()`（含 `_gen`/块数，进程内用）不同：这里只用可从盘上重算/读出的量。
        浮点用 `%.17g`（float64 往返无损）, 不许用 `%.6g` 之类（会假命中）。
        """
        nd = int(self._dead.sum()) if self._dead is not None else 0
        nnz = int(self.T.nnz) if self.T is not None else 0
        V = len(self.idx2term)
        return (f"v1|k1={self.k1:.17g}|b={self.b:.17g}|avg={self.avgdl:.17g}"
                f"|N={int(self.N)}|nnz={nnz}|V={V}|live={int(self.N) - nd}")

    def _ensure_w(self):
        r"""物化层 `w`（01h，opt-in）：按 `_tc_blocks` 逐块预计算"每 posting 的加权值"

            w = idf(t) · tf·(k1+1)/(tf + k1(1−b + b·dl/avgdl))

        查询命中时每个 posting 只做一次 gather ＋ 一次 scatter-add（＝ bm25s 的形态）。
        键不符就重建（`O(nnz)`）；不做也不亏（未命中走现算 ≈4.4 ms/查询）。
        段在场时返回 `None`（批次内不参与）：挂段改全局统计 `idf`/`avgdl`，已建的 `w` 必属旧世代，
        正解是批次内一律现算、批次末折回单块后重建一次。`_w_blocks` 是派生缓存, 用 `_cache_mu` 双检。
        """
        if getattr(self, "_segments", None):
            return None
        key = self._w_key()
        if getattr(self, "_w_key_val", None) == key and getattr(self, "_w_blocks", None) is not None:
            return self._w_blocks
        with self._cache_mu:
            if (getattr(self, "_w_key_val", None) == key
                    and getattr(self, "_w_blocks", None) is not None):
                return self._w_blocks
            # 计数：一次批次应只让这个计数 +1（批次内必须为 0）
            self._w_misses = int(getattr(self, "_w_misses", 0)) + 1
            return self._ensure_w_locked(key)

    def _ensure_w_locked(self, key):
        """`_ensure_w` 的变异部分（调用方已持有 `_cache_mu`）。"""
        # 01h：优先用盘上落盘的 `w`（索引期写好，冷启动即命中；mmap 按需缺页）
        wdir = getattr(self, "_w_persist_dir", None)
        if wdir is not None and getattr(self, "_w_persist_ok", False):
            try:
                from pathlib import Path as _Pw
                blks = sorted(_Pw(wdir).glob("w_*.npy"), key=lambda p: int(p.stem.split("_")[1]))
                self._ensure_tc()
                if blks and len(blks) == len(self._tc_blocks):
                    out = []
                    for p, (csc, _rb) in zip(blks, self._tc_blocks):
                        arr = np.load(p, mmap_mode="r")           # 不整体读入
                        if int(arr.size) != int(csc.data.size):
                            raise ValueError(f"{p.name} 长度 {arr.size} ≠ 块 nnz {csc.data.size}")
                        out.append(arr)
                    self._w_blocks = out
                    self._w_key_val = key
                    self._w_from_disk = True
                    return out
            except Exception as _e:                                # noqa: BLE001
                log(f"[WARN] 盘上 `w` 不可用（{_e!r}）→ 回退内存重建/现算")
                self._w_persist_ok = False
        self._ensure_tc()
        dn = self._ensure_dn()
        dl, idf = self.dl, self._idf
        k1, b, avg = self.k1, self.b, self.avgdl
        out = []
        for csc, _rb in self._tc_blocks:
            tf = np.asarray(csc.data, dtype=np.float32)
            rows = np.asarray(csc.indices, dtype=np.int32)
            norm = tf * (k1 + 1.0) / (tf + dn[rows])                  # 与现算逐字同形
            cnt = np.diff(np.asarray(csc.indptr, dtype=np.int64))     # 每列 posting 数, 把 idf 广播到 posting 上
            out.append((norm * np.repeat(idf[:len(cnt)], cnt)).astype(np.float32))
        self._w_blocks = out
        self._w_key_val = key
        return out

    def _score_incr_fast(self, idxs):
        """ 与 `_score_incr` 逐位等价的快路径。

         只做"不改变浮点运算顺序"的改动，否则拿不到逐位一致（本仓要求逐位）：
        ·  直接切 `indptr/data/indices`，不用 `csc.getcol(c)`（它每次新建对象并拷贝列数据）。
        ·  去掉中间变量（`m = ~dead[rows]` 的两次索引），一次布尔索引。
        ·  不做常数提取（`k1*(1-b)`、`k1*b/avg` 等）—— 那会重结合浮点，逐位必挂。
        ·  只对 CSC 成立（`indptr[c]:indptr[c+1]` 才是列），断言 `format == "csc"`（写错成 CSR 会静默算错）。
        """
        self._ensure_tc()
        if type(self).MATVEC:                      # 01g：M1 手写矩阵化（opt-in）
            return self._score_matvec(idxs)
        # 批次内不建/不用 `w`：挂段会改全局统计（`idf`/`avgdl`），已建的 `w` 都属旧世代，
        #    故批次内一律现算（跳过本分支，落到下面的现算路径，逐位等价），
        #    批次末折回单块后再惰性重建一次。
        if type(self).W_CACHE and not getattr(self, "_segments", None):
            wb = self._ensure_w()
            scores = np.zeros(self.N, dtype=np.float32)
            for c in idxs:
                for (csc, rb), wblk in zip(self._tc_blocks, wb):
                    if c >= csc.shape[1]:
                        continue
                    s = int(csc.indptr[c]); e = int(csc.indptr[c + 1])
                    if e <= s:
                        continue
                    rows = csc.indices[s:e]
                    if rb:
                        rows = rows + rb
                    if type(self).ADD_AT:                     # 01i：无缓冲累加（逐位）
                        np.add.at(scores, rows, wblk[s:e])
                    else:
                        scores[rows] += wblk[s:e]             # 每个 posting 只有一次 gather ＋ 一次累加
            return scores
        scores = np.zeros(self.N, dtype=np.float32)
        dl, avg, k1, b = self.dl, self.avgdl, self.k1, self.b
        dead, idf = self._dead, self._idf
        dn = self._ensure_dn() if type(self).DN_PRECOMP else None    # 01f：opt-in 预计算（默认 None, 零成本）
        for c in idxs:
            for csc, rb in self._tc_blocks:
                assert csc.format == "csc", f"分块必须是 CSC，实得 {csc.format}"
                if c >= csc.shape[1]:      # 段挂载后各块列宽不同：该块没有这一列, 跳过
                    continue
                s = int(csc.indptr[c])
                e = int(csc.indptr[c + 1])
                if e <= s:
                    continue
                tf = csc.data[s:e]
                rows = csc.indices[s:e]
                if rb:
                    rows = rows + rb
                if dead is not None:
                    keep = ~dead[rows]
                    if not keep.any():
                        continue
                    rows = rows[keep]
                    tf = tf[keep]
                if dn is not None:
                    norm = tf * (k1 + 1.0) / (tf + dn[rows])                     # 01f：预计算版（逐位等价）
                else:
                    norm = tf * (k1 + 1.0) / (tf + k1 * (1.0 - b + b * dl[rows] / avg))
                if type(self).ADD_AT:                    # 01i：无缓冲累加（逐位）
                    np.add.at(scores, rows, idf[c] * norm)
                else:
                    scores[rows] += idf[c] * norm
        return scores

    def _tc_key(self):
        """ 共享 CSC 的失效代数 key：`路径|mtime_ns|shape|nnz`
        （索引重建/换版, mtime 变则 key 变, 旧段自动不命中, 不会被误用）。"""
        from pathlib import Path as _Pk
        p = getattr(self, "_index_path", None)
        try:
            mt = _Pk(p).stat().st_mtime_ns if p else 0
        except OSError:
            mt = 0
        return f"{p}|{mt}|{tuple(self.T.shape)}|{int(self.T.nnz)}"

    def _ensure_tc(self):
        """01e-C: 补建分块 CSC 缺口 —— 只建 [covered, T行数) 这一块, O(新 nnz)。

        块只存 tf 结构 (idf/dead 查询时应用), 统计变化不失效旧块; 唯一的
        重建诱因是 append 增行 (旧实现 append 后首查要付整表 `T.tocsc()`)。

        并发：它会改 `_tc_blocks`/`_tc_rows`（读者也会调它）, 变异部分用
        `_cache_mu` 兜住（双检：快路径不加锁，只有真要补块时才进临界区）。
        """
        if self.T is None:
            raise RuntimeError("v1 索引(无 tf 原语)无分块缓存 —— 但 v1 必有 W 快路径, "
                               "走到这里说明 W 被外置置空且无 T, 属非法状态")
        t_rows = int(self.T.shape[0])
        if self._tc_rows >= t_rows:
            return
        with self._cache_mu:
            if self._tc_rows >= t_rows:          # 双检：别的线程可能刚补完
                return
            self._ensure_tc_locked(t_rows)

    def _ensure_tc_locked(self, t_rows: int):
        """`_ensure_tc` 的变异部分（调用方已持有 `_cache_mu`）。"""
        if not self._tc_blocks:
            if type(self).TC_SHARED:
                from .tc_shared import publish_or_attach
                csc, info = publish_or_attach(self.T, self._tc_key())
                self._tc_shared_info = info
                if csc is not None:
                    self._tc_blocks = [(csc, 0)]
                    self._tc_rows = t_rows
                    log(f"S1 CSC 走共享内存（owner={info.get('owner')} name={info.get('name')} "
                        f"{info.get('bytes', 0):,}B shared={info.get('shared')}）")
                    return
                log(f"[WARN] 共享 CSC 不可用（{info.get('reason')}）→ 回落私有 tocsc()")
            self._tc_blocks = [(self.T.tocsc(), 0)]
            self._tc_rows = t_rows
            return
        base = self._tc_rows
        seg = self.T[base:t_rows].tocsc()   # 行切片 = O(新行 nnz)
        self._tc_blocks.append((seg, base))
        self._tc_rows = t_rows

    @_with_read
    def topk(self, qtoks, k=10, *, sort=True):
        r"""打分 ＋ 取 top-k 一次做完（调用方不必自己拼 `argpartition`）。

        不要改成 `np.argpartition(s, n-k)[n-k:]`（`kth` 靠近末尾）：numpy 对 `kth` 位置的
        代价不对称，那样大约慢 1.9×；这里保留 `argpartition(-s, k-1)[:k]`（`kth` 靠近开头）。

        返回 `(idx, scores)`；`idx` 为行号（默认按分数降序；`sort=False` 时集合相同、顺序不定）。
        与 `score_all` 一样：收 token 列表，不是文本。
        """
        idxs = [self.term2idx[t] for t in set(qtoks) if t in self.term2idx]
        if not idxs:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32)
        s = self._score_incr_fast(idxs) if type(self).FAST_SCORE else self._score_incr(idxs)
        n = int(s.size)
        k = int(min(max(k, 0), n))
        if k <= 0:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32)
        if k >= n:
            idx = np.arange(n)
        else:
            idx = np.argpartition(-s, k - 1)[:k]      # kth 靠近开头（比"末尾"快）
        if sort:
            idx = idx[np.argsort(-s[idx])]
        return idx, s[idx]

    # ---------------- 段挂载：只读 postings ＋ 合并 df 查询（最小档） ----------------

    @_with_write
    def mount_segment(self, seg_dir, tag, new_terms, dl, rows=None, *, generation=None):
        """把一段只读 postings 挂到主表上 —— 表外新词走增量路径的最小档。

        段是三个 `.npy`（`<tag>.data.npy` / `.idx.npy` / `.ptr.npy`，CSC；不用 npz：zip 不能 mmap），
        列号已经落在"主表加宽后"的空间里（老词用主表列号，新词用 `V .. V+M-1`）。

        只做四件事（不写盘、不改主表的 `T`）：挂成查询路径的一个分块；把段的去重
        (文档,词) 计数并进 `df`；扩词表与 `dl`/`N`；重算 `_idf`（全局量，老文档分数也会变）。
        于是「增量 ≡ 全库重建」在折叠之前就成立。

        两条边界：词表用普通 dict 覆盖（载体只读 mmap，不能就地加词）；
        段可撤回或并入（`unmount_segment` 后进先出且仅对未折叠段 / `fold_segment`），两者都清块缓存并重算统计。
        """
        from pathlib import Path as _PathS          # 本文件按惯例在方法内导入 Path
        seg_dir = _PathS(seg_dir)
        base = seg_dir / tag
        pd, pi, pp = (base.with_suffix(".data.npy"), base.with_suffix(".idx.npy"),
                      base.with_suffix(".ptr.npy"))
        for p in (pd, pi, pp):
            if not p.exists():
                raise FileNotFoundError(f"段缺失：{p.name} → 该段不可挂（须重建，或压实前保留该段）")
        d = np.asarray(np.load(pd, mmap_mode="r"))
        i = np.asarray(np.load(pi, mmap_mode="r"))
        p = np.asarray(np.load(pp, mmap_mode="r"))
        V0 = len(self.idx2term)
        M = len(new_terms)
        V1 = V0 + M
        n_col = int(len(p) - 1)
        # 列宽可以比当前词表窄（追加式词表：早先发布的段用它当时的列号），这里加空列补齐
        #    （CSC 加空列 = `indptr` 尾部填 `p[-1]`）；但不能比当前词表宽。
        if n_col > V1:
            raise ValueError(f"段的列数 {n_col} > 主表加宽后的 {V1}"
                             f"（段的列号必须落在已存在的列里）")
        if n_col < V1:
            _p = np.asarray(p)
            p = np.concatenate([_p, np.full(V1 - n_col, int(_p[-1]), dtype=_p.dtype)])
            log(f"S1 段 {tag} 列宽 {n_col} < 当前 {V1} → 补 {V1 - n_col} 个空列（追加式词表）")
        n_seg = int(rows) if rows is not None else int(len(dl))
        if len(dl) != n_seg:
            raise ValueError(f"`dl` 长度 {len(dl)} ≠ 段行数 {n_seg}")
        seg = csc_matrix((np.asarray(d, dtype=np.float32), np.asarray(i, dtype=np.int32),
                          np.asarray(p, dtype=np.int64)), shape=(n_seg, V1))

        row_base = int(self.N)
        # ① 挂块（查询侧原样复用分块 CSC 路径）
        self._ensure_tc()
        self._tc_blocks.append((seg, row_base))
        self._tc_rows = row_base + n_seg
        # ② 合并 df：段对每列的"去重 (文档,词) 对"计数 = CSC 的 indptr 差分
        contrib = np.diff(seg.indptr).astype(np.int32)
        self.df = np.concatenate([np.asarray(self.df, dtype=np.int32),
                                  np.zeros(M, dtype=np.int32)]) + contrib
        # ③ 扩词表（载体只读, 用普通 dict 覆盖）＋ 扩 dl/N
        self.idx2term = list(self.idx2term) + list(new_terms)
        self.term2idx = {t: k for k, t in enumerate(self.idx2term)}
        self.dl = np.concatenate([np.asarray(self.dl, dtype=np.float32),
                                  np.asarray(dl, dtype=np.float32)])
        if self._dead is not None:
            self._dead = np.concatenate([self._dead, np.zeros(n_seg, dtype=bool)])
        self.N = row_base + n_seg
        # ④ 重算统计（idf 是全局量, 老文档分数也会变）
        self._invalidate_w()
        self._recompute_stats()
        segs = getattr(self, "_segments", None)
        if segs is None:
            segs = self._segments = {}
        segs[tag] = dict(rows=n_seg, row_base=row_base, m_new=M, V0=V0, V1=V1,
                         terms=list(new_terms), contrib=contrib,
                         gen=(str(generation) if generation is not None else None),
                         df_added=int(contrib.sum()), bytes=int(d.nbytes + i.nbytes + p.nbytes))
        return segs[tag]

    @_with_write
    def unmount_segment(self, tag):
        """卸载一个已挂的段：把 `mount_segment` 做过的事逐项撤回（回收 / 换段用）。

        两条前提（都会显式报错，不静默）：
          · 该段没有被 `fold_segment` 折叠过（折叠过的在 `folded_report()` 里，内容已在主表）
          · 后进先出：它必须是最后挂的那个（否则撤回会把别人的列/行算错）
        """
        segs = getattr(self, "_segments", None)
        if not segs or tag not in segs:
            raise KeyError(f"未挂载的段：{tag}（已挂：{sorted((segs or {}).keys())}）")
        if tag in getattr(self, "_folded", {}):
            raise RuntimeError(f"段 {tag} 已被折叠（内容在主表里），不能用 `unmount` 撤 → 需要的话请重建")
        info = segs[tag]
        rb, V0, M = int(info["row_base"]), int(info["V0"]), int(info["m_new"])
        if int(self.N) != rb + int(info["rows"]):
            raise RuntimeError(f"段 {tag} 不是最后挂的（后进先出）：当前 N={int(self.N)}，"
                               f"该段覆盖 {rb}..{rb + int(info['rows'])}")
        if int(self.V) != V0 + M:
            raise RuntimeError(f"段 {tag} 之后词表又变过（V={int(self.V)} ≠ {V0 + M}）：先卸载别的段")
        # ① 摘块 ＋ 清分块缓存
        self._tc_blocks = [(c, b) for c, b in self._tc_blocks if int(b) != rb]
        self._tc_blocks, self._tc_rows = [], 0
        # ② 撤 df（减掉该段的贡献）＋ 缩词表
        self.df = (np.asarray(self.df, dtype=np.int32) - np.asarray(info["contrib"], dtype=np.int32))[:V0]
        self.idx2term = list(self.idx2term)[:V0]
        self.term2idx = {t: k for k, t in enumerate(self.idx2term)}
        # ③ 缩 dl / dead / N
        self.dl = np.asarray(self.dl, dtype=np.float32)[:rb]
        if self._dead is not None:
            self._dead = np.asarray(self._dead)[:rb]
        self.N = rb
        # ④ 统计重算 ＋ 删除留档
        self._invalidate_w()
        self._recompute_stats()
        del segs[tag]
        return dict(tag=tag, rows=int(info["rows"]), V0=V0, M=M, N=int(self.N), V=int(self.V))

    @_with_write
    def fold_segment(self, tag):
        """把已挂的段折叠进主表：扩列 ＋ 追加行，段的内容成为主表的一部分。

        与 `mount_segment` 的分工：
          · 挂段：段的 postings 留在段里（主表 `T` 一字不动），查询走两个块
          · 折段：段的 postings 并进主表 `T`，之后查询只走主表（代价：一次性 `O(nnz)` 拷贝）

        `df`/`dl`/词表在 `mount_segment` 时已经合并过，本方法不再动它们（否则双重计数）。
        折叠完成后：段从 `_segments` 移入 `_folded`（留档），分块缓存被清空以便按新 `T` 重建。
        """
        segs = getattr(self, "_segments", None)
        if not segs or tag not in segs:
            raise KeyError(f"未挂载的段：{tag}（已挂：{sorted((segs or {}).keys())}）")
        info = segs[tag]
        rb = int(info["row_base"])
        blk = next(((c, int(b)) for c, b in self._tc_blocks if int(b) == rb), None)
        if blk is None:
            raise RuntimeError(f"段 {tag} 的分块不在查询路径里（row_base={rb}）")
        csc, _ = blk
        V1 = int(csc.shape[1])
        n_before = int(self.N)
        nnz_before = int(self.T.nnz) if self.T is not None else 0
        t0 = time.perf_counter()

        T = self.T.tocsr()
        if int(T.shape[1]) < V1:                       # 扩列：只换形状，不拷数据
            T = csr_matrix((T.data, T.indices, T.indptr), shape=(int(T.shape[0]), V1))
        self.T = _sp_vstack([T, csc.tocsr()], format="csr").astype(np.float32)   # 追加行：O(nnz) 一次性
        if int(self.T.shape[0]) != n_before:
            raise RuntimeError(f"折叠后行数 {int(self.T.shape[0])} ≠ N {n_before}（挂段时已算过行数）")
        if int(self.T.shape[1]) != int(len(self.idx2term)):
            raise RuntimeError(f"折叠后列数 {int(self.T.shape[1])} ≠ 词表 {len(self.idx2term)}")
        # 不动 df/dl/词表（挂段时已合并）
        self._tc_blocks, self._tc_rows = [], 0         # 清块缓存, 下次查询按新 T 重建
        self._invalidate_w()
        self._recompute_stats()
        fold = dict(info)
        fold.update(rows=int(csc.shape[0]), ms=(time.perf_counter() - t0) * 1e3,
                    nnz_before=nnz_before, nnz_after=int(self.T.nnz), V1=V1)
        del segs[tag]
        folded = getattr(self, "_folded", None)
        if folded is None:
            folded = self._folded = {}
        folded[tag] = fold
        return fold

    @_with_write
    def fold_all(self):
        """把全部已挂段一次性并进主表，折完 `T 行数 == N`。

        为什么需要它：`fold_segment` 断言"折完 `T` 行数 == `N`"，而 `N` 含所有已挂段，
        多段时单段折叠在数学上不可能。
        做法：按 `row_base` 排序取块，各块补到同一列宽，`vstack([T] + 各段)`，再断言行数。
        """
        segs = dict(getattr(self, "_segments", None) or {})
        if not segs:
            return dict(tags=[], rows=0, ms=0.0, N=int(self.N), V=int(self.V))
        if self.T is None:
            raise RuntimeError("v1 索引(无 tf 原语)不支持折叠")
        t0 = time.perf_counter()
        order = sorted(segs.items(), key=lambda kv: int(kv[1]["row_base"]))
        blk_by_rb = {int(b): c for c, b in self._tc_blocks}
        mats, widths = [], [int(self.T.shape[1])]
        for _tag, info in order:
            rb = int(info["row_base"])
            blk = blk_by_rb.get(rb)
            if blk is None:
                raise RuntimeError(f"段 {_tag} 的分块不在查询路径里（row_base={rb}）")
            mats.append(blk.tocsr())
            widths.append(int(blk.shape[1]))
        V1 = max(widths)                                  # 各段列宽可能不同, 统一补到最宽
        T = self.T.tocsr()
        if int(T.shape[1]) < V1:
            T = csr_matrix((T.data, T.indices, T.indptr), shape=(int(T.shape[0]), V1))
        rs = []
        for m in mats:
            m = m.tocsr()
            if int(m.shape[1]) < V1:
                m = csr_matrix((m.data, m.indices, m.indptr), shape=(int(m.shape[0]), V1))
            rs.append(m)
        nnz_before = int(T.nnz)
        self.T = _sp_vstack([T] + rs, format="csr").astype(np.float32)
        if int(self.T.shape[0]) != int(self.N):
            raise RuntimeError(f"折叠后行数 {int(self.T.shape[0])} ≠ N {int(self.N)}"
                               f"（挂段时已算过行数）")
        self._tc_blocks, self._tc_rows = [], 0             # 清块缓存, 下次查询按新 T 重建
        self._invalidate_w()
        self._recompute_stats()
        folded = {}
        for _tag, info in order:
            rec = dict(info)
            rec.update(nnz_before=nnz_before, nnz_after=int(self.T.nnz), V1=V1,
                       ms=(time.perf_counter() - t0) * 1e3)
            folded[_tag] = rec
            del self._segments[_tag]
        self._folded = {**getattr(self, "_folded", {}), **folded}
        return dict(tags=list(folded), rows=int(len(rs)), ms=(time.perf_counter() - t0) * 1e3,
                    nnz_before=nnz_before, nnz_after=int(self.T.nnz),
                    N=int(self.N), V=int(self.V))

    @_with_read
    def segment_report(self):
        """已挂段的清单（含词表代次）：`{tag: {...}}`。段可被 `unmount_segment(tag)` 撤回，或 `fold_segment(tag)` 折叠。"""
        return dict(getattr(self, "_segments", {}))

    @_with_read
    def folded_report(self):
        """已折叠的段（留档：折叠是"段内容并进主表"的动作，之后段本身不再存在）。"""
        return dict(getattr(self, "_folded", {}))

    def _seg_key(self):
        """词表代次键：段一变，旧视图/旧共享段就不该命中（`tc_shared` 用它做失效判据）。"""
        segs = getattr(self, "_segments", {})
        return "|".join(f"{t}:{v['rows']}:{v['m_new']}:{v.get('gen')}" for t, v in sorted(segs.items()))

    @_with_read
    def score_all(self, qtoks):
        idxs = [self.term2idx[t] for t in set(qtoks) if t in self.term2idx]
        if not idxs:
            return np.zeros(self.N, dtype=np.float32)
        if self.W is not None:
            # 旧格式 (v1/v2 含 W_data) 加载后的兼容快路径: CSC 列切片 → 按行求和 (O(Σdf))
            # 01d 新索引 W=None, 走下面的分块 CSC 现算 (权重不常驻)
            return np.asarray(self.Wc[:, idxs].sum(axis=1)).ravel().astype(np.float32)
        return self._score_incr_fast(idxs) if type(self).FAST_SCORE else self._score_incr(idxs)

    # ---------------- 形状/规模访问 (01d: W 不常驻时的替代口径) ----------------

    @property
    def V(self):
        """词表大小 (不依赖 W)。"""
        return len(self.idx2term)

    @property
    def nnz(self):
        """非零元数: 权威 T 优先; 仅旧格式 (W-only) 时回退 W。"""
        if self.T is not None:
            return int(self.T.nnz)
        if self.W is not None:
            return int(self.W.nnz)
        return 0

    # 正确性根基: BM25 = idf[t] × sat(tf_dt, dl_d) 可分离。权威存 (T= tf, df, dl);
    # 删除/修改 = 墓碑 (读原 toks 修正 df) + 尾部追加; 统计 (df/N_live/avgdl/idf) 向量化重算 O(V),
    # 零 CSR 重建。词表封闭约束: upsert 文本中的词必须已存在于词表 (词表扩张 = 重建级事件)。

    def _invalidate_w(self):
        """快路径失效: W/Wc 依赖冻结的全局 idf, 任何统计变化后必须作废。

         不动分块 CSC: _tc_blocks 只存 tf 结构, idf/dead 在查询时应用,
         故统计变化与块无关, append 后旧块继续有效 (只在首查补新块)。"""
        self.W = None
        self.Wc = None
        self._gen = int(getattr(self, "_gen", 0)) + 1     # 01h：物化层 `w` 的缓存代次（失效兜底）

    def _recompute_stats(self):
        """df 修正后: N_live / avgdl / idf 一次性向量化重算 O(V)。"""
        if self._dead is not None:
            n_live = int(self._dead.size - int(self._dead.sum()))
            live_dl_sum = float(self.dl[:self.N][~self._dead].sum())
        else:
            n_live = self.N
            live_dl_sum = float(self.dl[:self.N].sum())
        self.avgdl = live_dl_sum / max(n_live, 1)
        self._idf = np.log((n_live - self.df + 0.5) / (self.df + 0.5) + 1).astype(np.float32)
        self._gen = int(getattr(self, "_gen", 0)) + 1     # 01h：统计变了, `w` 缓存必须换键
        return n_live

    @_with_write
    def _tombstone_rows(self, rows):
        """墓碑 rows: 读原行 toks → df 修正 (每词 −1) → dead 标记。

        物理行保留 (score 过滤)。成本 O(变更行 nnz); 不做 CSR 物理删除。
        """
        if self.T is None:
            raise RuntimeError("v1 索引(无 tf 原语)不支持增量, 请重建 (--force)")
        if self._dead is None:
            self._dead = np.zeros(self.N, dtype=bool)
        T = self.T.tocsr()
        changed = False
        for r in rows:
            if r >= self.N or self._dead[r]:
                continue
            seg = slice(int(T.indptr[r]), int(T.indptr[r + 1]))
            cols = T.indices[seg]
            if cols.size:
                # cols 无重复 (每词一格) → 每格 df −1 即该文档对该词的贡献移除
                self.df[cols] -= 1
                np.maximum(self.df, 0, out=self.df)
            self._dead[r] = True
            changed = True
        if changed:
            self._invalidate_w()

    @_with_write
    def append_docs(self, toks_list):
        """追加 toks_list 为新物理行, 返回新行号数组 (docid 映射由调用方迁移)。

        成本 O(新增 nnz); 词表外 token 跳过 (词表封闭约束)。空/全词表外文档
        仍追加为合法行 (dl=0)。
        """
        if self.T is None:
            raise RuntimeError("v1 索引(无 tf 原语)不支持增量, 请重建 (--force)")
        # 禁止段在场时就地追加：此时 `N = row_base + n_seg` 而 `T` 只有 `row_base` 行,
        #    就地追加会让 `T` 的新行与段的逻辑行重叠（行空间重叠）、`dl`/`dead` 拼接次序也不一致,
        #    静默错分。先 `fold_segment()` 把段并进主表（或改走挂段路径）再追加。
        if getattr(self, "_segments", None) and len(toks_list):
            raise RuntimeError(
                f"已挂段 {sorted(self._segments)} 时就地追加会与段的逻辑行重叠（静默错分）"
                f" → 请先 `fold_segment(tag)` 把段并进主表，或改走 `mount_segment` 路径"
                f"（纯删除走 `_tombstone_rows` 不受此限）")
        k = len(toks_list)
        base = self.N
        #  写路径的词表查找：dict 优先（`WRITE_DICT=True` 时按需建一次并缓存到实例）；
        t2i = getattr(self, "_write_dict", None)
        if t2i is None and type(self).WRITE_DICT:
            t2i = self._write_dict = {t: i for i, t in enumerate(self.idx2term)}
        if t2i is None:
            t2i = self.term2idx
        new_rows, new_cols, new_data = [], [], []
        counts = np.zeros(k, dtype=np.int64)   # 每新行的 posting 数 (0 = 空/全词表外文档)
        for i, toks in enumerate(toks_list):
            tf = defaultdict(int)
            for w in toks:
                tf[w] += 1
            for w, c in tf.items():
                ci = t2i.get(w)
                if ci is None:
                    continue  # 词表封闭约束
                new_rows.append(base + i)
                new_cols.append(ci)
                new_data.append(float(c))
                counts[i] += 1
                self.df[ci] += 1
        # ---- 01e-A: 容量预分配追加 —— O(新 nnz) (容量耗尽那次倍增 = 摊还 O(总 nnz)) ----
        # 新行按列号局部排序, 与旧 vstack→tocsr 路径逐位一致 (prealloc_csr.py 断言)
        if self._Tbuf is None:
            self._Tbuf = PreallocCSR.from_csr(self.T.tocsr(), slack=self.T_SLACK)
        self._Tbuf.append(np.asarray(new_rows, dtype=np.int32) - base,
                          np.asarray(new_cols, dtype=np.int32),
                          np.asarray(new_data, dtype=np.float32),
                          counts)
        self.T = self._Tbuf.wrap()
        self.dl = np.concatenate([self.dl, np.asarray([len(t) for t in toks_list], dtype=np.float32)])
        if self._dead is not None:
            self._dead = np.concatenate([self._dead, np.zeros(k, dtype=bool)])
        self.N = base + k
        self._invalidate_w()
        self._recompute_stats()
        self._views_extend(k)
        return np.arange(base, base + k)

    @_with_write
    def update_docs(self, rows, toks_list):
        """真增量 upsert: 墓碑 rows (旧内容) + 尾部追加新内容。

        返回新行号 (与 rows 一一对应) —— 文档身份不变, 行号迁移, 调用方更新
        docid→row。成本 = O(变更行 nnz + V); 对比旧实现 (hash diff → 全量重建)
        零重建。
        """
        rows = list(rows)
        assert len(rows) == len(toks_list), "rows/toks_list 长度必须一致"
        self._tombstone_rows(rows)
        return self.append_docs(toks_list)

    def _rebuild_w(self):
        """从 tf 原语重建权重视图 (O(nnz) 一次)。
        """
        T = self.T.tocsr()
        n = T.shape[0]
        row_of = np.repeat(np.arange(n, dtype=np.int32), np.diff(T.indptr))
        tf = T.data
        dl_r = self.dl[row_of]
        w = tf * (self.k1 + 1.0) / (tf + self.k1 * (1.0 - self.b + self.b * dl_r / self.avgdl))
        w *= self._idf[T.indices]
        if self._dead is not None:
            w = w * (~self._dead[row_of])  # 死行 posting 归零 → 分数天然排除
        self.W = csr_matrix((w.astype(np.float32, copy=False), (row_of, T.indices)),
                            shape=T.shape).tocsr()
        self.Wc = self.W.tocsc()

    # ---------------- 行管理辅助 (供上层增量编排) ----------------

    def live_rows(self):
        """当前存活的物理行号 (非墓碑)。"""
        if self._dead is None:
            return list(range(self.N))
        return [r for r in range(self.N) if not self._dead[r]]

    def oov_terms(self, toks_list):
        """检测 toks_list 中是否有词表外词（真增量会静默跳过 → 调用方须回退重建）。

        返回 OOV 词集合; 空集 = 可安全走真增量 (词表封闭满足)。
        """
        oov = set()
        for toks in toks_list:
            for w in toks:
                if w not in self.term2idx:
                    oov.add(w)
        return oov

    # 接口 C（覆盖位图 + 视图可用性）：视图 = 密集向量（按物理行） + 覆盖位图 + 可用性 + meta。
    #  查询融合 `fused_scores` 只在视图在场且 available 时偏离 BM25，否则逐位返回 BM25-only（解耦）；
    #  覆盖行融合分 = base + α·s01·span（delta ≥ 0, 覆盖行分数不降）；append 扩 N 时覆盖位图补 False,
    #  墓碑行由 live 门排除。视图不落盘（save/load 不含它），派生层可丢、由 sidecar 重挂；
    #  `meta.model_id` 只记录不校验。

    def attach_view(self, name, vectors, rows=None, *, available=True,
                    meta=None, replace=False):
        """挂一个派生视图（覆盖位图 + 可用性由内核持有）。

        `vectors`: array [n, dim]（float32，与 `rows` 一一对齐；行内范数在挂载时
        预归一化一次）。`rows`: 物理行号（缺省 = 当前全部存活行）；墓碑行
        拒绝（覆盖必须是对"活行"的诚实声明）。重复名字默认拒绝（`replace=True`
        才允许顶替 —— 顶替即"分版"语义，旧版即刻释放）。
        """
        V = np.asarray(vectors, dtype=np.float32)
        if V.ndim != 2 or V.shape[0] == 0:
            raise ViewError(f"vectors 必须是 [n>0, dim] 的二维数组，收到 shape={V.shape}")
        rows = (np.arange(self.N, dtype=np.int64) if rows is None
                else np.asarray(rows, dtype=np.int64))
        if rows.ndim != 1 or rows.shape[0] != V.shape[0]:
            raise ViewError(f"rows({rows.shape}) 与 vectors({V.shape[0]} 行) 不对齐")
        if name in self._views and not replace:
            raise ViewError(f"视图 {name!r} 已在场（replace=True 才允许顶替）")
        if self._dead is not None:
            dead_hit = [int(r) for r in rows if r < self._dead.size and self._dead[r]]
            if dead_hit:
                raise ViewError(f"rows 含墓碑行 {dead_hit[:3]}… —— 覆盖只许声明活行")
        out_of = [int(r) for r in rows if not (0 <= r < self.N)]
        if out_of:
            raise ViewError(f"rows 越界 {out_of[:3]}…（N={self.N}）")
        if len(set(rows.tolist())) != rows.shape[0]:
            raise ViewError("rows 有重复行号 —— 一个物理行在同一个视图里至多一个向量")
        order = np.argsort(rows, kind="stable")
        rows = rows[order]
        V = np.ascontiguousarray(V[order], dtype=np.float32)
        norms = np.linalg.norm(V, axis=1, keepdims=True)
        vn = V / np.maximum(norms, np.float32(1e-12))
        cov = np.zeros(self.N, dtype=bool)
        cov[rows] = True
        self._views[str(name)] = dict(vectors=V, vn=vn, rows=rows,
                                      coverage=cov, dim=int(V.shape[1]),
                                      available=bool(available),
                                      meta=dict(meta or {}))
        return self.view_report([str(name)])[0]

    def detach_view(self, name):
        """卸载视图（释放向量）。返回是否存在。

        解耦判据的机械保证：本方法只 `del self._views[name]`，不碰 T/df/dl/idf
        任何评分状态, 故删后 `fused_scores` 逐位等于从未挂载。
        """
        return self._views.pop(str(name), None) is not None

    def set_view_available(self, name, available):
        """翻视图可用性（不释放向量）—— 接口 C 的"可用性"那一半。"""
        v = self._views.get(str(name))
        if v is None:
            raise ViewError(f"视图 {name!r} 不在场")
        v["available"] = bool(available)

    def _views_extend(self, k):
        """append_docs 扩 N 后：每个视图的覆盖位图补 k 个 False（新行 = 未覆盖）。

        旧物化对象（load 前的旧脚本）可能没有 `_views`, 惰性补空, 不报错。
        """
        views = getattr(self, "_views", None)
        if not views or k <= 0:
            return
        for v in views.values():
            v["coverage"] = np.concatenate(
                [v["coverage"], np.zeros(k, dtype=bool)])

    def view_report(self, names=None):
        """接口 C 消费侧：视图清单（覆盖位图统计 + 可用性 + meta）。

        `covered_live` 是给检索用的那一格（墓碑行不算覆盖——融合时会被
        live 门排除，报出来就必须与行为一致）；`covered_total` 留作对账。
        """
        live = None if self._dead is None else ~self._dead
        if isinstance(names, str):
            names = [names]                            # 单视图查询别按字符迭代
        out = []
        for name in sorted(self._views if names is None else
                           [n for n in names if n in self._views]):
            v = self._views[name]
            cov = v["coverage"][: self.N]
            n_live = int((cov & live).sum()) if live is not None else int(cov.sum())
            row = dict(name=name, available=bool(v["available"]),
                       covered_live=n_live, covered_total=int(cov.sum()),
                       dim=int(v["dim"]))
            row.update(v["meta"])
            out.append(row)
        return out

    @_with_read
    def fused_scores(self, qtoks, qvec, view_name, alpha=0.35):
        """BM25 分数 + 指定视图的密集加分（覆盖行单调不降）。

        - 视图不在场 / `available=False` 时逐位返回 BM25-only（解耦）；
        - 覆盖的活行：`fused = base + α·s01·span`，其中 `s01 = clip(cos,0,1)`
          （零/负相似不加分，"覆盖即命中"会把词法池平白放大）、
          `span = max(base)`（base 全 0 时取 1.0）——delta ≥ 0, 覆盖行分数不降；
          IEEE 加法对 `y≥0` 单调, 该性质是构造保证, 不靠阈值；
        - `qvec` 维度必须与视图一致（不一致抛 `ViewError`，绝不静默降维）；
        - float32 全程：同一输入同一结果（无随机、无并行归约）。
        """
        base = self.score_all(qtoks)
        v = self._views.get(str(view_name))
        if v is None or not v["available"]:
            return base
        q = np.asarray(qvec, dtype=np.float32)
        if q.ndim != 1 or q.shape[0] != v["dim"]:
            raise ViewError(
                f"qvec 维度 {q.shape} 与视图 {view_name!r} 的 dim={v['dim']} 不匹配")
        live = None if self._dead is None else ~self._dead
        cov = v["coverage"][: self.N]
        mask = cov if live is None else (cov & live)
        sel = np.flatnonzero(mask)
        if sel.size == 0:
            return base
        pos = np.searchsorted(v["rows"], sel)      # rows 挂载时已排序
        # s01 = clip(cos, 0, 1)：零/负相似不加分 —— 若用 (cos+1)/2，cos=0 也拿
        # 一半加分，"覆盖即命中"会把词法池平白放大（所有覆盖行恒 >0）。
        s01 = np.clip(v["vn"][pos] @ (q / max(float(np.linalg.norm(q)), 1e-12)),
                      np.float32(0.0), np.float32(1.0)).astype(np.float32)
        span = float(base.max())
        delta = (np.float32(alpha) * s01) * np.float32(span if span > 0.0 else 1.0)
        fused = base.copy()
        fused[sel] = base[sel] + delta
        return fused

    @_with_read
    def save(self, path, w_cache=False):
        """落盘索引。`w_cache=True` 时额外把物化层 `w` 写进 `.mm/`（opt-in，默认不写）。

        `w` 是派生缓存：盘上同时留 `w_key.txt`（内容型键）；加载时比对，不一致就拒用。
        代价：盘 +`nnz×4 B`（1M 段 ≈ +165 MB）· 建库时间 +≈1%；写密集时 `avgdl` 会让它失效。
        """
        if self.T is None and self.W is None:
            raise RuntimeError("索引既无权重视图也无 tf 原语, 无法持久化")
        # 禁止段在场时落盘：挂段是内存视图, `T` 里没有段的行, 写出的 npz 会内部不一致
        #    （`dl` 长度 = `N`（含段行）而 `T_shape[0]` = 主表行数）, 重载时会崩。
        #    盘上状态必须对应某一个完整世代：要么先 `fold_all()`, 要么先 `unmount_segment()`。
        _segs = dict(getattr(self, "_segments", None) or {})
        if _segs:
            raise RuntimeError(
                f"已挂段 {sorted(_segs)}：`save()` 会写出 `dl`(len={int(self.N)}) 与 "
                f"`T`(rows={int(self.T.shape[0]) if self.T is not None else 0}) 不一致的 npz "
                f"（重载会崩）→ 请先 `fold_all()` 把它们并进主表，或先 `unmount_segment(tag)`）")
        z = dict(vocab=np.asarray(self.idx2term, dtype=object),
                 dl=self.dl,  # dl 必须落盘, 否则 load 后 BM25 分数失真
                 params=np.asarray([self.k1, self.b, self.avgdl], dtype=np.float32))
        if self.T is not None:
            z.update(fmt='primolix-bm25-v3',
                     W_shape=np.asarray(self.T.shape, dtype=np.int64),
                     T_data=self.T.data, T_idx=self.T.indices, T_ptr=self.T.indptr,
                     T_shape=np.asarray(self.T.shape, dtype=np.int64),
                     df=self.df, _idf=self._idf,
                     dead=self._dead if self._dead is not None else np.zeros(0, bool),
                     N_live=np.int64(self.N - (self._dead.sum() if self._dead is not None else 0)))
        else:
            # v1 遗留 (无 tf 原语): 权重视图是唯一内容, 必须在场
            z.update(fmt='primolix-bm25-v2',
                     W_data=self.W.data, W_idx=self.W.indices,
                     W_ptr=self.W.indptr, W_shape=np.asarray(self.W.shape, dtype=np.int64))
        np.savez_compressed(path, **z)
        #    附加落盘（不动上方 npz, 向后兼容不变）。词表常驻 ~9MB(私有) → ~0.14MB(文件后备)。
        #     附加件：失败不许拖垮落盘。
        try:
            from .vocab_mmap import write as _vm_write
            _acct = _vm_write(self.idx2term, str(path) + ".vocab")
            # 载体的世代键：载体被 `load` 优先采用, 若没有键, 侧车就地重存失败残留的旧载体
            #    会让重载的新词静默消失。键口径与 `.mm/mm_key.txt` 一致：
            #    `v1|名字|mtime_ns|size`（尾部带 `V` 便于人读）。
            from pathlib import Path as _Pv
            _npzv = _Pv(str(path))
            if not _npzv.exists():
                _npzv = _Pv(str(path) + ".npz")
            if _npzv.exists():
                _stv = _npzv.stat()
                (_Pv(str(path) + ".vocab") / "vocab_key.txt").write_text(
                    f"v1|{_npzv.name}|{_stv.st_mtime_ns}|{_stv.st_size}|V={len(self.idx2term)}",
                    encoding="utf-8")
            log(f"S1 词表载体落盘: {str(path)}.vocab（{_acct['total']:,} B / "
                f"{_acct['total'] / max(len(self.idx2term), 1):.1f} B/词；世代键已写）")
        except Exception as _e:
            warn(f"[WARN] 词表载体落盘跳过（附加件失败，功能不受影响但会退化）: {_e!r}")
        #    落成独立 `.npy`, `load` 可 `mmap_mode='r'`（文件后备、可共享、可丢）。
        #     `df` 故意不落：`_tombstone_rows` 会原地写 `df`（只读 mmap 会当场报错）。
        try:
            from pathlib import Path as _P2
            _d2 = _P2(str(path) + ".mm")
            _d2.mkdir(parents=True, exist_ok=True)
            if self.T is not None:
                np.save(_d2 / "T_data.npy", self.T.data)
                np.save(_d2 / "T_idx.npy", self.T.indices)
                np.save(_d2 / "T_ptr.npy", self.T.indptr)
                np.save(_d2 / "T_shape.npy", np.asarray(self.T.shape, dtype=np.int64))
            for _nm, _fn in (("_idf", "idf.npy"), ("dl", "dl.npy")):
                _a = getattr(self, _nm, None)
                if _a is not None:
                    np.save(_d2 / _fn, _a)
            if self._dead is not None:
                np.save(_d2 / "dead.npy", self._dead)
            # 世代键：`.mm/` 与 npz 必须同代, 否则 `load` 会跨源取字段（`T/dl/idf/dead` ← `.mm/`,
            #    `df/params/N` ← npz）而同形状静默给错分数。键 = npz 的 `名字|mtime_ns|size`,
            #    放在 npz 之后写；键不符则 `load` 拒用 `.mm/`（npz 自带 `T_data`, 只退化成"不 mmap"）。
            _npz = _P2(str(path)) if _P2(str(path)).exists() else _P2(str(path) + ".npz")
            if _npz.exists():
                _st = _npz.stat()
                (_d2 / "mm_key.txt").write_text(
                    f"v1|{_npz.name}|{_st.st_mtime_ns}|{_st.st_size}"
                    f"|N={int(self.N)}|nnz={int(self.T.nnz) if self.T is not None else 0}",
                    encoding="utf-8")
            _b = 0 if self.T is None else (int(self.T.data.nbytes) + int(self.T.indices.nbytes)
                                           + int(self.T.indptr.nbytes))
            log(f"S1 postings 上总线: {_d2}（T {_b:,} B，世代键已写 mm_key.txt）")
        except Exception as _e:
            warn(f"[WARN] postings 落盘跳过（附加件失败）: {type(_e).__name__}: {_e!r}"
                 f"  注意：Windows 上「就地重存」会撞 mmap 锁（既有 .mm/ 已被 mmap）→ 建议存到新目录")
        # 01h：物化层 `w` 单独一个 try —— 不许被上面 `T_*` 的覆盖失败连坐
        if w_cache and self.T is not None:
            try:
                from pathlib import Path as _P3
                _d3 = _P3(str(path) + ".mm")
                _d3.mkdir(parents=True, exist_ok=True)
                _wb = self._ensure_w()
                for _i, _arr in enumerate(_wb):
                    np.save(_d3 / f"w_{_i}.npy", np.asarray(_arr))
                (_d3 / "w_key.txt").write_text(self._w_persist_key(), encoding="utf-8")
                log(f"S1 物化层 w 落盘: {_d3}（{len(_wb)} 块 → "
                    f"{sum(int(np.asarray(a).nbytes) for a in _wb):,} B；键已写入 w_key.txt）")
            except Exception as _e:
                warn(f"[WARN] 物化层 w 落盘失败（附加件；功能不受影响，只是冷启动要重建）: "
                     f"{type(_e).__name__}: {_e!r}")
        log(f"S1 索引落盘: {path} (含词表 {len(self.idx2term)})")
    @classmethod
    def load(cls, path, N=0, V=0):
        """从 npz 加载, 三格式兼容:

        - ``fmt=primolix-bm25-v3`` (01d): 仅原语 (T/df/dl/dead), 无 W —— 查询走分块 CSC 现算
        - ``fmt=primolix-bm25-v2`` 且含 ``W_data``: 恢复 W/Wc 快路径 (旧行为不变) + tf 原语 (增量可用)
        - v1 (无 ``T_data``): 仅 W 快路径, 增量不可用

        ``N``/``V`` 参数仅为兼容旧调用而保留, 实际以文件形状为准
        (增量后物理行 > 原语料, 禁用调用方猜测的 N)。
        """
        z = np.load(path, allow_pickle=True)
        fmt = str(z.get('fmt', b'primolix-bm25-v1')) if 'fmt' in z else 'primolix-bm25-v1'
        # 兼容改名前的索引：复名前落盘写的是 `zelix-bm25-v2/v3`, 归一化后走同一套逻辑。
        #    （改名只换了包名与类名, 盘上格式一字未变, 旧索引必须继续可加载）
        fmt = fmt.replace('zelix-bm25-', 'primolix-bm25-')
        has_w = 'W_data' in z
        has_t = (fmt in ('primolix-bm25-v2', 'primolix-bm25-v3')
                 and 'T_data' in z and z['T_data'].size)
        if not has_w and not has_t:
            raise RuntimeError(f"{path}: 既无 W_data 也无 T_data, 无法加载"
                               f"（fmt={fmt!r}；若为旧版格式请确认 fmt 是否被改名影响）")
        #    且多进程可共享。 载体缺席则回落老路径（list+dict），行为不变。
        from pathlib import Path as _Path
        _carrier = _Path(str(path) + ".vocab")
        _npz_vocab = z['vocab'].tolist() if 'vocab' in z else []
        vocab = _npz_vocab                              # 默认就用 npz 自带的词表（永远在场）
        if _carrier.exists():
            # 世代键校验：载体被优先采用, 故键不符 / 缺键 / 长度与 npz 的 `vocab` 不一致
            #    时一律拒用载体, 回落 npz 词表（否则会沿用残留旧载体, 新词静默消失）。
            _npzp = _Path(str(path)) if _Path(str(path)).exists() else _Path(str(path) + ".npz")
            _want = ""
            if _npzp.exists():
                _stp = _npzp.stat()
                _want = f"v1|{_npzp.name}|{_stp.st_mtime_ns}|{_stp.st_size}"
            _keyf = _carrier / "vocab_key.txt"
            _got = _keyf.read_text(encoding="utf-8").strip() if _keyf.exists() else ""
            _use = bool(_want) and _got.startswith(_want)
            if _use:
                from .vocab_mmap import VocabMmap
                _cand = VocabMmap(_carrier)
                if len(_cand) == len(_npz_vocab):
                    vocab = _cand
                else:
                    _use = False
                    _cand.close()
            if not _use:
                log(f"[WARN] 词表载体与 npz 不同代/长度不符 → 拒用，回落 npz 自带词表"
                    f"（盘上={_got[:36] or '(缺键)'} · 期望={_want[:36]}）")
        bm = cls.__new__(cls)
        bm._rw = RWLock()                    # `load` 走 `__new__`, 并发原语要在这里补建
        bm._cache_mu = threading.RLock()
        bm.W = bm.Wc = None
        if has_w:
            bm.W = csr_matrix((z['W_data'], z['W_idx'], z['W_ptr']),
                              shape=(int(z['W_shape'][0]), int(z['W_shape'][1])))
            bm.Wc = bm.W.tocsc()   # 旧格式: 重建 CSC 缓存 (一次 O(nnz), 与旧 load 相同)
            n_rows = int(z['W_shape'][0])  # 文件行数权威
        else:
            n_rows = int(z['T_shape'][0])  # v3: T 形状即行数权威
        bm.N = n_rows
        bm.idx2term = vocab
        bm.term2idx = vocab if hasattr(vocab, "id_of") else {t: i for i, t in enumerate(vocab)}
        bm._vocab_carrier = vocab if hasattr(vocab, "id_of") else None   #  两阶段 attach_hot 用
        bm._index_path = str(path)          #  共享 CSC 的 key 用（`_tc_key`）
        bm._tc_shared_info = None
        bm.T = bm.df = bm._idf = None
        bm._dead = None
        bm._views = {}             # 接口 C: 视图不持久化（派生层，可由 sidecar 重挂）
        bm._Tbuf = None            # 01e: 首次 append 时一次性换预分配后端
        bm._tc_blocks = []         # 01e-C: 分块 CSC (有 T 时在下面建块0)
        bm._tc_rows = 0
        if 'dl' in z and len(z['dl']) == n_rows:
            bm.dl = z['dl'].astype(np.float32)
            bm.avgdl = float(z['params'][2]) if 'params' in z else float(bm.dl.sum()) / max(n_rows, 1)
            bm.k1, bm.b = (float(z['params'][0]), float(z['params'][1])
                           if 'params' in z else (1.5, 0.75))
        else:
            # 旧版索引无 dl: 从 CSR 按行还原 (O(nnz), 比伪 dl=0/avgdl=1.0 诚实)
            log("[WARN] 索引缺 dl, 从 nnz 按行还原 (旧版格式)")
            bm.dl = np.diff(bm.W.indptr).astype(np.float32)
            bm.avgdl = float(bm.dl.sum()) / max(n_rows, 1)
            bm.k1, bm.b = 1.5, 0.75
        if has_t:
            # v2/v3: 恢复 tf/df 原语 → 增量可用; 分块 CSC 块0 = 旧 `Tc = T.tocsc()` 等价物
            if 'T_shape' in z:
                Tshape = (int(z['T_shape'][0]), int(z['T_shape'][1]))
            else:
                Tshape = (int(z['W_shape'][0]), int(z['W_shape'][1]))
            _mmd = _Path(str(path) + ".mm")
            _mm_ok = _mmd.exists() and (_mmd / "T_data.npy").exists()
            # 世代校验：`.mm/` 与 npz 不同代时必须拒用 —— `load` 跨源取字段
            #    （`T/dl/idf/dead` ← `.mm/`, `df/params/N` ← npz）, 同形状下不报错却给陈旧分数。
            if _mm_ok:
                _mk = _mmd / "mm_key.txt"
                if not _mk.exists():
                    _mm_ok = False
                    log("[WARN] `.mm/` 缺世代键 `mm_key.txt`（旧版落盘）→ 拒用，"
                        "改从 npz 读 postings（判据同 `w` 层：不一致一律拒用）")
                else:
                    try:
                        _st = _Path(str(path)).stat() if _Path(str(path)).exists() else None
                        _cur = (f"v1|{_Path(str(path)).name}|{_st.st_mtime_ns}|{_st.st_size}"
                                if _st is not None else "")
                        _got = _mk.read_text(encoding="utf-8").strip()
                        if not _cur or not _got.startswith(_cur):
                            _mm_ok = False
                            log(f"[WARN] `.mm/` 与 npz 不同代 → 拒用（不回退静默！）·"
                                f" 盘上={_got[:60]} · 当前={_cur[:60]}")
                    except Exception as _e:                      # noqa: BLE001
                        _mm_ok = False
                        log(f"[WARN] `.mm/` 世代键不可读（{_e!r}）→ 拒用")
            if _mm_ok:
                _td = np.load(_mmd / "T_data.npy", mmap_mode="r")
                _ti = np.load(_mmd / "T_idx.npy", mmap_mode="r")
                _tp = np.load(_mmd / "T_ptr.npy", mmap_mode="r")
                bm.T = csr_matrix((_td, _ti, _tp), shape=Tshape, copy=False)
                bm._mm_postings = (_td, _ti, _tp)
                _pi, _pl, _pd = _mmd / "idf.npy", _mmd / "dl.npy", _mmd / "dead.npy"
                if _pi.exists():
                    bm._idf = np.load(_pi, mmap_mode="r")
                if _pl.exists():
                    bm.dl = np.load(_pl, mmap_mode="r")
                if _pd.exists():
                    # 必须拷成可写：`_tombstone_rows` 要就地写 `_dead[r] = True`, 而 mmap 只读,
                    #    否则"落盘→重载→再更新"会报 `ValueError: assignment destination is read-only`。
                    #    （`dl`/`_idf` 只被重新绑定, 可留在 mmap 上）
                    bm._dead = np.array(np.load(_pd, mmap_mode="r"), dtype=bool, copy=True)
                #  `df` 仍来自 npz：`_tombstone_rows` 会原地写它（只读 mmap 会当场报错）
                bm.df = z['df'].astype(np.int32) if 'df' in z else None
                if cls.EAGER_TC and not cls.TC_SHARED:
                    bm._tc_blocks = [(bm.T.tocsc(), 0)]
                    bm._tc_rows = int(Tshape[0])
                    log("S1 postings 走 mmap · CSC eager")
                else:
                    log("S1 postings 走 mmap · CSC 推到首查"
                        + ("（TC_SHARED → 首查时发布/attach 共享段）" if cls.TC_SHARED
                           else "（EAGER_TC=False）"))
            else:
                bm.T = csr_matrix((z['T_data'], z['T_idx'], z['T_ptr']), shape=Tshape)
                bm._tc_blocks = [(bm.T.tocsc(), 0)]
                bm._tc_rows = int(Tshape[0])
                bm.df = z['df'].astype(np.int32) if 'df' in z else None
                bm._idf = z['_idf'].astype(np.float32) if '_idf' in z else None
                d = z['dead'] if 'dead' in z and z['dead'].size else None
                bm._dead = d.astype(bool) if d is not None else None
        nnz = (bm.T.nnz if bm.T is not None
               else (bm.W.nnz if bm.W is not None else 0))
        # 01h：盘上有没有可用的物化层 `w`？—— 内容型键比对（不一致一律拒用）
        bm._w_persist_dir = None
        bm._w_persist_ok = False
        bm._w_from_disk = False
        try:
            _wd = _Path(str(path) + ".mm")
            _wk = _wd / "w_key.txt"
            if bm.T is not None and _wk.exists() and list(_wd.glob("w_*.npy")):
                _want = bm._w_persist_key()
                _got = _wk.read_text(encoding="utf-8").strip()
                if _got == _want:
                    bm._w_persist_dir = str(_wd)
                    bm._w_persist_ok = True
                    log(f"S1 物化层 w 可用（键相符 → 首查直接 mmap，不必重建）")
                else:
                    log(f"[WARN] 盘上 w 的键不符 → 拒用（不回退静默！）·"
                        f" 盘上={_got} · 当前={_want}")
        except Exception as _e:                                    # noqa: BLE001
            log(f"[WARN] 物化层 w 探测跳过: {_e!r}")
        log(f"S1 从 {path} 加载索引 (fmt={fmt}, nnz={nnz}, 词表 {len(vocab)}, "
            f"avgdl={bm.avgdl:.1f}, k1={bm.k1}, b={bm.b}"
            + (", tf原语就绪" if bm.T is not None else ", tf原语缺失(v1, 增量不可用)"))
        return bm
