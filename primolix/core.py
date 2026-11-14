#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import hashlib
import json
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np

from . import kernel
from .kernel import SparseBM25, zh_tokenize
from .kernel import set_mm_sidecar, set_npz_level      # noqa: F401  （落盘档位，从 core 也可调）
from .routing import FOLD_FIRST, REBUILD, STAGED, route_update

__all__ = ["Primolix"]

# ---- 建索引默认排除项（面向通用文件树, 非研究仓库专用） ----
EXCLUDE_DIRS = {'.git', '.workbuddy', 'node_modules', '__pycache__', '.venv',
                'venv', 'env', '.env', 'dist', 'build', '.next',
                '.pytest_cache', '.mypy_cache', '.idea', '.vscode', '.idx',
                '.primolix_index', '.zelix_index', '.idx'}
EXCLUDE_EXT = {'.png', '.jpg', '.jpeg', '.gif', '.svg', '.ico', '.pdf', '.zip',
               '.npz', '.npy', '.log', '.out', '.pyc', '.bin', '.woff',
               '.woff2', '.exe', '.dll'}
CODE_EXT = {'.py', '.js', '.ts', '.tsx', '.jsx', '.java', '.go', '.rs', '.c',
            '.cpp', '.h', '.hpp', '.sh', '.bat', '.ps1', '.yaml', '.yml',
            '.toml'}
PROSE_EXT = {'.md', '.txt', '.rst', '.json', '.html', '.css', '.htm', '.csv',
             '.tsv'}
MAX_CHUNK = 1200
CODE_BLOCK_LINES = 80
MAX_FILE_BYTES = 2_000_000

# 模型缓存: 进程内只加载一次 dense 模型
_MODEL_CACHE = {}

# 01m（P-D）：`_topk_idx` 的取负缓冲 —— 线程私有（共享会在并发查询下互相污染）
_NEG_BUF = threading.local()

#   `TOPK_APPROX=False` ⇒ 逐位回到改动前的精确 `argpartition`（回退用，须与旧行为逐位一致）。
#   `TOPK_FRAC` 是采样步长（1/32）；`TOPK_SHRINK` 是阈值余量（采样第 k 大 × 0.98）。
TOPK_APPROX = True
TOPK_FRAC = 32
TOPK_SHRINK = 0.98
_TOPK_STAT = threading.local()


def topk_stats(reset=False):
    """本线程 `_topk_idx` 的路径计数（用于"两臂都真的动过"的准入自证）。"""
    d = getattr(_TOPK_STAT, "d", None)
    if d is None:
        d = _TOPK_STAT.d = dict(calls=0, approx=0, exact=0, fallback=0, cand=0)
    out = dict(d)
    if reset:
        for key in d:
            d[key] = 0
    return out


# ------------------------------------------------------------------ 输出
# Progress/diagnostic messages go to stderr, never to stdout (the caller may pipe the output).
# Set primolix.core.VERBOSE = False to silence them entirely.
VERBOSE = True


def _say(msg) -> None:
    if VERBOSE:
        from .kernel import log as _log
        _log(str(msg))


def get_dense_model(model_path=''):
    """加载（或复用）dense 编码模型。model_path 为空自动找 bge-m3 缓存。

    sentence_transformers / 模型缺省时返回 None（调用方降级 BM25-only）。
    """
    if not model_path:
        snaps = sorted(Path.home().glob(
            '.cache/huggingface/hub/models--BAAI--bge-m3/snapshots/*'))
        if not snaps:
            raise FileNotFoundError("未找到 bge-m3 模型缓存 (或指定 --model snapshot)")
        model_path = str(snaps[0])
    if model_path not in _MODEL_CACHE:
        t0 = time.time()
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            _say("[提示] sentence_transformers 未安装, 无法加载 dense 模型")
            return None
        try:  # 关闭 transformers 的 "Loading weights" 进度条
            from transformers.utils import logging as _hflog
            _hflog.disable_progress_bar()
        except Exception:
            pass
        try:
            model = SentenceTransformer(model_path, device='cpu')
        except Exception as e:
            _say(f"[提示] dense 模型加载失败: {e}")
            return None
        _MODEL_CACHE[model_path] = model
        _say(f"dense 模型预热 {time.time() - t0:.1f}s ({model_path.split('/')[-1]})")
    return _MODEL_CACHE[model_path]


# ---------------- 文件扫描与分块（纯 stdlib） ----------------

def iter_files(root: Path, skip_dirs=None):
    """遍历 root 下文件（**os.scandir 递归**：用 dirent 自带的类型/大小，避免每文件 `resolve` ＋ 重复 `stat`）。

    · `skip_dirs`：绝对路径集合，其下所有文件跳过（如索引目录自身）。按相对前缀跳过，
      不再对每个文件调 `resolve()` —— 那是原实现里最贵的一笔（每文件一次路径解析）。
    · 返回 `[(Path, rel_path)]`，按 `rel` 字符串排序（与原实现同序，保住建库/切段的确定性）。
    · `rel` 用平台原生分隔符（与原实现一致）—— 哈希字典的键靠它，别改成 `/`。
    """
    root = Path(root)
    try:
        root_abs = root.resolve()
    except OSError:                                    # 极少数：路径不可解析 → 退回原样
        root_abs = root
    skip_rel = set()
    for d in (skip_dirs or []):
        try:
            rel_d = Path(d).resolve().relative_to(root_abs)
            skip_rel.add(str(rel_d).replace('\\', '/').lower())
        except (ValueError, OSError):
            continue                                   # 不在 root 下 ⇒ 不影响遍历
    out = []
    stack = [("", root_abs)]
    while stack:
        rel_base, cur = stack.pop()
        try:
            with os.scandir(cur) as it:
                for e in it:
                    name = e.name
                    relp = Path(rel_base) / name if rel_base else Path(name)
                    rel_posix = str(relp).replace('\\', '/').lower()
                    try:
                        if e.is_dir(follow_symlinks=False):
                            if name in EXCLUDE_DIRS or rel_posix in skip_rel:
                                continue
                            stack.append((str(relp), Path(e.path)))
                            continue
                        if not e.is_file(follow_symlinks=False):
                            continue
                    except OSError:
                        continue
                    if name in {'.env', '.env.local'} or Path(name).suffix.lower() in EXCLUDE_EXT:
                        continue
                    try:
                        if e.stat(follow_symlinks=False).st_size > MAX_FILE_BYTES:
                            continue
                    except OSError:
                        continue
                    out.append((Path(e.path), relp))
        except OSError:
            continue
    out.sort(key=lambda pr: str(pr[1]))
    return out


def file_hash(p: Path):
    h = hashlib.md5()
    with open(p, 'rb') as f:
        for blk in iter(lambda: f.read(65536), b''):
            h.update(blk)
    return h.hexdigest()


def chunk_prose(text, rel):
    """段落级切分(检索甜区 200-800 字), 过长再按字符窗断。"""
    paras, buf, start = [], [], 1
    line_no = 1
    for line in text.splitlines(keepends=True):
        if line.strip() == '':
            if buf:
                paras.append((''.join(buf), start))
                buf, start = [], line_no + 1
        else:
            if not buf:
                start = line_no
            buf.append(line)
        line_no += 1
    if buf:
        paras.append((''.join(buf), start))
    out = []
    for t, ln in paras:
        t = t.strip()
        if len(t) < 20:
            continue
        while len(t) > MAX_CHUNK:                 # 过长按字符窗口断(甜区内)
            out.append((t[:MAX_CHUNK], rel, ln))
            t = t[MAX_CHUNK:]
        out.append((t, rel, ln))
    return out


def chunk_code(text, rel):
    lines = text.splitlines()
    out = []
    for i in range(0, len(lines), CODE_BLOCK_LINES):
        blk = '\n'.join(lines[i:i + CODE_BLOCK_LINES]).strip()
        if len(blk) >= 20:
            out.append((blk, rel, i + 1))
    return out


def scan_root_ex(root: Path, skip_dirs=None, known=None, stamps=None, verify="full"):
    """扫描语料目录。返回 `(files, stamps_now)`。

    `files`   = `[(rel, md5, chunks|None)]`（`known` 命中 ⇒ `chunks=None`）
    `stamps`  = `{rel: (size, mtime_ns)}`，只含进入了 `files` 的文件

    `verify`：
        `"full"`（默认）—— 每个文件读内容算 md5；
        `"stat"`        —— 先比 `(size, mtime_ns)`，元数据没变就不读内容（实测 ~1/10 成本）；
                           元数据变了才读，内容 hash 没变时仍算"未变"（只刷新 stamp）。

    `"stat"` 的前提是写入方会更新 mtime；保留 mtime 的还原类工具会漏报 ⇒ 用 `"full"` 兜底。
    缺 `known`/`stamps` 时 `"stat"` 如实退回 `"full"`（不假装快扫）。
    """
    if verify not in ("full", "stat"):
        raise ValueError(f"verify 只接受 'full' / 'stat'，实得 {verify!r}")
    if verify == "stat" and (known is None or stamps is None or not stamps):
        verify = "full"
    files, stamps_now = [], {}
    for p, rel in iter_files(root, skip_dirs):
        rel_s = str(rel)
        try:
            st = p.stat()
        except OSError:
            continue
        stamp = (int(st.st_size), int(st.st_mtime_ns))
        if verify == "stat" and rel_s in known and stamps.get(rel_s) == stamp:
            files.append((rel_s, known[rel_s], None))     # 元数据未变：不读内容
            stamps_now[rel_s] = stamp
            continue
        try:
            data = p.read_bytes()
        except OSError:
            continue
        h = hashlib.md5(data).hexdigest()
        if known is not None and known.get(rel_s) == h:
            files.append((rel_s, h, None))                # 内容未变（可能只是 mtime 变了）
            stamps_now[rel_s] = stamp
            continue
        text = data.decode('utf-8', errors='ignore')
        ext = p.suffix.lower()
        chunks = chunk_code(text, rel_s) if ext in CODE_EXT else chunk_prose(text, rel_s)
        if chunks:
            files.append((rel_s, h, chunks))
            stamps_now[rel_s] = stamp
    return files, stamps_now


def scan_root(root: Path, skip_dirs=None, known=None):
    """返回 [(rel_path_str, hash, chunks)]；chunks=[(text, rel, line)]。

    `known`（`{rel: md5}`，只有 `update()` 会传）：命中 ⇒ 该文件不 decode、不切段，第三项为 `None`
    （调用方必须跳过 `None`；`build` 不传 `known`，所以永远拿到真 chunks）。
    每个文件只 读一次字节：同一次读既算 md5 又（必要时）decode。
    需要 `stat` 快扫或拿 `(size, mtime_ns)` ⇒ 用 `scan_root_ex()`。
    """
    return scan_root_ex(root, skip_dirs, known=known)[0]


# ---------------- Primolix 索引对象 ----------------

def _norm_stream(v):
    r"""建库分词档归一：`False` / `True` / `"ids"` (也收 `off`/`on`/`ids` 字串)。

    只改建库期的分词驻留方式, 不改分数; 非法值报错, 不静默回落。
    """
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("ids", "id"):
            return "ids"
        if s in ("1", "true", "on", "yes"):
            return True
        if s in ("0", "false", "off", "no", ""):
            return False
        raise ValueError(f"stream 只接受 off/on/ids，实得 {v!r}")
    return "ids" if v == "ids" else bool(v)


def _topk_exact(x, k):
    """精确路径（改动前的原实现）：N 长取负 + 一遍 `argpartition`。

    调用前必须已保证 `0 < k < x.size`。返回 k 个下标（分数降序、并列按下标升序）。
    """
    n = int(x.size)
    buf = getattr(_NEG_BUF, "arr", None)
    if buf is None or buf.size < n or buf.dtype != x.dtype:
        buf = np.empty(n, dtype=x.dtype)
        _NEG_BUF.arr = buf
    neg = np.negative(x, out=buf[:n])                     # ⭐ 复用 ⇒ 无 N 长临时
    idx = np.argpartition(neg, k - 1)[:k].astype(np.int64, copy=False)
    return idx[np.lexsort((idx, -x[idx]))]


def _topk_sample(x, n, k, st):
    """⭐ P-O3 近似路径：1/`TOPK_FRAC` 采样定阈 → `flatnonzero` → 只对候选选 k。

    返回 `None` 表示"本形态不适合近似"（交回精确路径，退化有界）：
      · 非浮点 / 非正分布（全零、全负）—— `_topk_idx` 的通用契约；
      · 正分文档数 < k（命中文档极少的查询）。
    这是概率性保证：阈值 = 采样第 k 大 × `TOPK_SHRINK`，理论上可能高于真第 k 大
       ⇒ 漏掉边界文档。经验面（1M · 200 查询 ·
       k∈{10,100,1000} · frac∈{16,32}：分数多重集 0 违规、候选覆盖 100%）。
    """
    if x.dtype.kind != "f":
        return None
    if not (float(x.max()) > 0.0):
        return None
    f = int(TOPK_FRAC)
    lim = n // (4 * k + 8)                 # 采样点至少 ~4k+8 ⇒ 阈值估计不至于退化
    if lim < f:
        f = max(1, lim)
    sub = x[::f]
    m = int(sub.size)
    kk = min(k, m - 1)
    thr = np.partition(sub, m - kk)[m - kk] * x.dtype.type(TOPK_SHRINK)
    if not (thr > 0):
        # 采样第 k 大 = 0（命中文档极少）：阈值退化为 0 会把全部零分捞进候选。
        # 改成"只要正分" ⇒ 候选 = 真正命中的文档数（天然有界）。
        thr = x.dtype.type(np.nextafter(np.float32(0.0), np.float32(1.0)))
    cand = np.flatnonzero(x >= thr)
    if cand.size < k:                      # 有界退化 1：降阈到"只要正分"
        cand = np.flatnonzero(x > 0)
        if cand.size < k:                  # 有界退化 2：回精确（最坏 ≈ 基线）
            st["fallback"] += 1
            return None
    st["cand"] += int(cand.size)
    sub2 = x[cand]
    j = np.argpartition(-sub2, k - 1)[:k]
    idx = cand[j].astype(np.int64, copy=False)
    st["approx"] += 1
    return idx[np.lexsort((idx, -x[idx]))]


def _topk_idx(x, k):
    r"""取前 k 大的下标 (分数降序; 并列按下标升序)。

    P-O3（2026-10-19）：默认走「采样定阈」（`TOPK_APPROX=True`）——
    选择段 @1M 由 4.93 ms → 0.57 ms（配对 0.138，600/600 全胜）。做法：
    1/32 抽样估第 k 大 × 0.98 当阈值 → `flatnonzero` 取候选 → 只对候选 `argpartition`。
    不改返回契约：仍是"k 个下标、分数降序、值等于全量排序的前 k 个"。

    非正分布 / 正分不足 k / 非浮点 ⇒ 自动回精确路径（`TOPK_APPROX=False` 亦然）。
    近似性是概率保证而非证明；实测口径见 docs/api.md 第 5 节 #12。
    κ 很大（如 dense 分支的 `k*10`）时采样点相对 k 变少 ⇒ 收益收窄、退化档变多。

    01m（P-D）：精确路径的取负那一步写进线程私有常驻缓冲，免掉每次调用分配的
    N 长临时（@1M ≈4 MB/查询、@8.84M ≈35 MB/查询）。必须线程私有：共享缓冲在
    两个并发查询下会互相污染（本层的锁只保护索引结构，不保护这条快路径）。
    """
    n = int(x.size)
    k = int(min(max(k, 0), n))
    if k <= 0:
        return np.zeros(0, dtype=np.int64)
    if k >= n:
        #    分数降序。`query()` 的循环以 `s[i] <= 0: break` 截断 ⇒ 升序时一旦第一个是
        #    零分就整条查询返回空（k ≥ 活跃 chunk 数时必然触发：小语料 / 大 k）。
        #    不报错、不崩，只给一个"看起来干净"的空结果 —— 正是要防的那类错。
        idx = np.arange(n, dtype=np.int64)
        return idx[np.lexsort((idx, -x[idx]))]
    d = getattr(_TOPK_STAT, "d", None)
    if d is None:
        d = _TOPK_STAT.d = dict(calls=0, approx=0, exact=0, fallback=0, cand=0)
    d["calls"] += 1
    if TOPK_APPROX:
        idx = _topk_sample(x, n, k, d)
        if idx is not None:
            return idx
    d["exact"] += 1
    return _topk_exact(x, k)


class Primolix:
    """零编码检索索引的 Python API：`build` / `query` / `browse` / `update` / `info`。

    `update` = hash diff，词表内走真增量、词表外回退整库重建。
    属性约定：`_meta` 活跃 chunk 列表（每条含 `row` = SparseBM25 物理行）、`_bm` 内核实例、
    `_vecs` dense 向量（与 `_meta` 顺序对齐，None = 无 dense）、`_hashes` = {rel_path: md5}。
    """

    def __init__(self, index_dir, dense=False, model_path=''):
        self.dir = Path(index_dir)
        self._bm = None
        self._meta = None
        self._rows = None          # 活跃 chunk 的物理行号缓存 (每个 `self._meta =` 点置 None)
        self._rows_ident = None    # ⭐ 01m：`(rows 对象, 是否恒等)` —— 见 `_active_rows`
        self._hashes = None        # {rel: md5}
        self._stamps = {}          # {rel: (size, mtime_ns)} —— `verify="stat"` 快扫的物理前提
        self._delta_nnz = 0        # delta 累计 nnz（L2：用来判压实）
        self._delta_count = 0      # delta 文件数
        self._persisted_V = 0      # 盘上（npz ＋ 已提交 delta）能重建出的词表宽度
        self._persisted_N = 0      # `meta.jsonl` 里已经写到的行号（追加式的游标）
        self._meta_dels = 0        # `meta.jsonl` 里的删除记录条数（超阈值就整理一次）
        self._vecs = None
        self._dense = dense
        self._model_path = model_path

    # ---------- 建索引 ----------

    @classmethod
    def build(cls, root, out='.primolix_index', dense=False, model_path='',
              workers=0, skip_dirs=None, w_cache=False, stream=False):
        r"""建索引。`w_cache=True` → 同时把物化层 `w` 落盘（代价：盘 +`nnz×4 B`，默认关）。

        `stream` 控制分词期是否驻留 token（不改分数，只改建库内存峰值）：
          `False`（默认）＝ 分词一次并缓存，峰值最高；
          `True` ＝ 两遍各自分词、不驻留，代价是 jieba 跑两遍；
          `"ids"` ＝ 分词一次、立刻转 id、只留 `int32`，峰值最低且不重分词。
        该选择随索引落盘，重载后由 OOV 触发的重建继续用它。
        """
        self = cls(out, dense=dense, model_path=model_path)
        self._w_cache = bool(w_cache)
        self._stream = _norm_stream(stream)
        t0 = time.time()
        _say("扫描文件...")
        files, _stamps = scan_root_ex(Path(root), skip_dirs)
        chunks = []
        for rel, h, cs in files:
            for t, r, ln in cs:
                # 初始 build: chunk i ↔ SparseBM25 物理行 i (一一对应)
                chunks.append(dict(docid=f'{h[:8]}:{len(chunks)}', row=len(chunks),
                                   file=r, line=ln, text=t))
        self.dir.mkdir(parents=True, exist_ok=True)
        _say("构建 BM25 词表...")
        self._bm = SparseBM25([c['text'] for c in chunks], zh_tokenize,
                              workers=workers, stream=self._stream)
        self._meta = chunks
        self._rows = None
        self._hashes = {rel: h for rel, h, _ in files}
        self._stamps = _stamps          # size/mtime 印记：下一次 update 可走 stat 快扫
        self._persist()
        if dense:
            self._encode_dense()
        _say(f"建索引 {len(files)} 文件 → {len(chunks)} chunks, "
              f"{time.time() - t0:.0f}s ({self.info['size_mb']:.1f}MB)")
        return self

    # ---------- 持久化 ----------

    def _persist(self, meta: bool = True):
        """落盘 bm25.npz + meta.jsonl + hashes.json（build/update 共用）。

        `meta=False`（0.2.3）：不重写 `meta.jsonl`。1M 上整份重写要 5.45 s，
        而 delta 模式下它可以靠追加式维护（`_meta_append_pending`）＋ delta 自带的身份兜底。
        用 `meta=False` 前必须先调一次 `_meta_append_pending()` 让文件追上，否则会留下"有行没身份"。

        `hashes.json` 自 0.2.2 起是 `{"v":2, "hashes": {...}, "stamps": {rel: [size, mtime_ns]}}`；
        `stamps` 是 `verify="stat"` 快扫的物理前提（缺它就退化成每次读内容）。

        全量落盘会盖掉之前所有 delta（npz 里已经有那些行）⇒ 写完就清 `.delta/`。
        """
        self._bm.save(str(self.dir / 'bm25.npz'), w_cache=bool(getattr(self, "_w_cache", False)))
        if meta:
            self._write_meta_full()
        self._write_hashes()
        self._clear_deltas()
        self._delta_nnz = 0
        self._delta_count = 0
        self._persisted_V = int(self._bm.V)     # 全量落盘后，盘上的词表 = 当前词表

    def _maybe_compact_delta(self) -> dict | None:
        """两层压实（2026-11-10 起）。

        第一层（便宜，常用）：`delta 文件数 ≥ DELTA_MAX_FILES` ⇒ `_merge_deltas()`
        —— 只把已有的几个 delta 合成一个，代价 O(Σ delta nnz)，不动基线。
        第二层（贵，罕见）：`Σdelta.nnz > DELTA_COMPACT_RATIO × base.nnz` ⇒ 全量 `_persist()`
        —— 把 delta 折进 npz，代价 O(总 nnz)。

        为什么要分层：1M 上全量落盘 34 s，而 delta 自己只有几十 KB。原来只要 delta 攒够就付 34 s，
        等于把"写变便宜"的收益在压实点全还回去。
        只在 `PERSIST_MODE == "delta"` 且当前没有挂段时才会真的写盘。
        """
        if PERSIST_MODE != "delta" or self._delta_count <= 0:
            return None
        if self._bm.segment_report():
            return None                       # 段在场 ⇒ `save()` 会抛错，等折完再说
        base = int(self._bm.T.nnz) - int(self._delta_nnz)
        big = int(self._delta_nnz) > DELTA_COMPACT_RATIO * max(base, 1)
        many = int(self._delta_count) >= DELTA_MAX_FILES
        if not (big or many):
            return None
        if big:
            t0 = time.time()
            cnt, nnz = self._delta_count, self._delta_nnz
            self._meta_append_pending()          # 先让 meta.jsonl 追上，才能用 meta=False
            self._persist(meta=False)            # ⭐ 省掉 5.45 s（1M）的整份 meta 重写
            out = dict(compacted=True, tier=2, why="nnz", deltas=cnt, delta_nnz=nnz,
                       ms=(time.time() - t0) * 1000.0)
            _say(f"[压实2] {cnt} 个 delta（{nnz:,} nnz）→ 全量落盘 {out['ms']:.0f} ms")
            return out
        return self._merge_deltas()

    def _merge_deltas(self) -> dict | None:
        """第一层压实：把生效中的多个 delta 合成一个（O(Σ delta nnz)，⛔ 不碰基线 npz）。

        提交顺序：写 `tmp` → `rename` 成新 seq（新文件自带 `supersedes`）→ 删旧文件。
        崩在中间任一时刻都安全：没 rename ⇒ 全是旧文件；rename 后 ⇒ 新文件声明取代旧的，
        旧文件即使还在也不会被回放。合并后链不断（base_N 取第一个、N_after 取最后一个）。
        """
        files = delta_files(self.dir)
        if len(files) < 2:
            return None
        t0 = time.time()
        seqs, ptr_parts, idx_parts, dat_parts = [], [], [], []
        dl_parts, dead_parts, meta_parts, vocab_parts = [], [], [], []
        base_N, N_after, V, nnz_off = None, None, 0, 0
        for p in files:
            with np.load(p, allow_pickle=False) as z:
                s = int(z["seq"]) if "seq" in z.files else int(p.stem)
                seqs.append(s)
                if base_N is None:
                    base_N = int(z["base_N"])
                N_after = int(z["N_after"])
                V = max(V, int(z["V"]))
                ptr = np.asarray(z["ptr"], np.int64)
                ptr_parts.append(ptr[1:] + nnz_off)
                nnz_off += int(ptr[-1])
                idx_parts.append(np.asarray(z["idx"], np.int32))
                dat_parts.append(np.asarray(z["dat"], np.float32))
                dl_parts.append(np.asarray(z["dl"], np.float32))
                if int(np.asarray(z["dead"]).size):
                    dead_parts.append(np.asarray(z["dead"], np.int64))
                if "meta_json" in z.files:
                    meta_parts.extend(json.loads(str(z["meta_json"])))
                if "vocab_json" in z.files:
                    vocab_parts.extend(json.loads(str(z["vocab_json"])))
        new_seq = max(seqs) + 1
        ptr = np.concatenate([np.zeros(1, np.int64)] + ptr_parts)
        rec = dict(
            seq=np.int64(new_seq),
            base_N=np.int64(base_N), N_after=np.int64(N_after),
            ptr=ptr,
            idx=(np.concatenate(idx_parts) if idx_parts else np.zeros(0, np.int32)),
            dat=(np.concatenate(dat_parts) if dat_parts else np.zeros(0, np.float32)),
            dl=(np.concatenate(dl_parts) if dl_parts else np.zeros(0, np.float32)),
            dead=(np.concatenate(dead_parts) if dead_parts else np.zeros(0, np.int64)),
            V=np.int64(V),
            vocab_json=np.asarray(json.dumps(vocab_parts, ensure_ascii=False)),
            meta_json=np.asarray(json.dumps(meta_parts, ensure_ascii=False)),
            supersedes=np.asarray(seqs, np.int64),      # ⭐ 声明取代：旧文件不必先删
        )
        d = Path(self.dir) / DELTA_SUBDIR
        tmp = d / f"{new_seq:06d}.npz.tmp"
        with open(tmp, "wb") as fh:
            np.savez_compressed(fh, **rec)
        os.replace(tmp, d / f"{new_seq:06d}.npz")       # ⭐ 提交点
        for p in files:                                  # 尽力删；删不掉也不影响正确性
            try:
                p.unlink()
            except OSError:
                pass
        self._delta_count = 1
        self._delta_nnz = int(nnz_off)
        ms = (time.time() - t0) * 1000.0
        _say(f"[压实1] {len(seqs)} 个 delta → 1 个（{nnz_off:,} nnz）· {ms:.0f} ms · ⛔ 未动基线")
        return dict(compacted=True, tier=1, why="files", merged=len(seqs),
                    delta_nnz=int(nnz_off), ms=ms)

    def _clear_deltas(self):
        """删除 `.delta/*`（全量落盘之后调用；未 rename 的 `.tmp` 一并清）。"""
        d = Path(self.dir) / DELTA_SUBDIR
        if not d.exists():
            return 0
        n = 0
        for p in list(d.glob("*.npz")) + list(d.glob("*.tmp")):
            try:
                p.unlink()
                n += 1
            except OSError:
                pass
        return n

    def _write_hashes(self):
        """只写 `hashes.json`（v2 格式，含 size/mtime 快扫印记）。"""
        doc = {"v": 2,
               "hashes": self._hashes or {},
               "stamps": {k: [int(v[0]), int(v[1])]
                          for k, v in (getattr(self, "_stamps", None) or {}).items()}}
        with open(self.dir / 'hashes.json', 'w', encoding='utf-8') as f:
            json.dump(doc, f, ensure_ascii=False, indent=1)

    def _persist_under_segments(self, del_rows=None):
        """段在场时的落盘：`meta.jsonl`（**只写主表行**）＋ `hashes.json`，⛔ 不动 `bm25.npz`。

        为什么不能直接 `_persist()`：它内含 `SparseBM25.save()`，而 `save()` 在段在场时
        直接抛错（挂段是内存视图，写出的 npz 会与 `dl`/`T` 不一致）。
        为什么 `meta.jsonl` 只写 `row < T.shape[0]` 的行：段内身份走段级侧车
        （`docs.jsonl`），重载由 `_absorb_staged` 补回；两边都写会引入双重计数，
        虽然按行去重挡得住，但不该靠它。
        0.2.3：改成追加式（`ceiling=seg_floor`），不再整份重写。
        """
        seg_floor = int(self._bm.T.shape[0])
        self._meta_append_pending(del_rows=del_rows, ceiling=seg_floor)
        self._write_hashes()

    # ---------- L2：增量持久化（delta 日志） ----------

    def _write_meta_full(self):
        """整份重写 `meta.jsonl`（全部活跃 chunk）。⛔ 不写 npz —— 构建/压实用它。"""
        with open(self.dir / 'meta.jsonl', 'w', encoding='utf-8') as f:
            for c in self._meta:
                f.write(json.dumps(c, ensure_ascii=False) + '\n')
        self._persisted_N = int(self._bm.N) if self._bm is not None else 0
        self._meta_dels = 0

    def _meta_append_pending(self, del_rows=None, ceiling=None) -> dict:
        """**追加式**维护 `meta.jsonl`（2026-11-10）：只写新增行 ＋ 一条删除记录。

        为什么能省：`meta.jsonl` 是 99.9 万条 chunk 的 JSON，整份重写在 1M 上要 5.45 s；
        而一次增量只多几行。追加的代价是 O(新增行数)，删除用 `{"del": [row…]}` 记一笔。
        读取时按 `row` 建表、遇 `del` 剔除，再按 `row` 排序。

        只在 delta 模式下安全：那里 delta 自带身份，即使 `meta.jsonl` 落后也不会"有行没身份"；
        npz 模式没有第二份身份，仍走整份重写（见 `_persist`）。
        """
        bm = self._bm
        floor = int(getattr(self, '_persisted_N', 0) or 0)
        top = int(ceiling) if ceiling is not None else None
        added = [c for c in self._meta
                 if int(c.get('row', -1)) >= floor
                 and (top is None or int(c.get('row', -1)) < top)]
        dels = [int(r) for r in (del_rows if del_rows is not None else [])]
        if not added and not dels:
            return dict(added=0, dels=0, bytes=0)
        path = self.dir / 'meta.jsonl'
        nbytes = 0
        with open(path, 'a', encoding='utf-8') as f:
            for c in added:
                line = json.dumps(c, ensure_ascii=False) + '\n'
                f.write(line)
                nbytes += len(line.encode('utf-8'))
            if dels:
                line = json.dumps({"del": dels}, ensure_ascii=False) + '\n'
                f.write(line)
                nbytes += len(line.encode('utf-8'))
        if added:
            self._persisted_N = max(int(c['row']) for c in added) + 1
        else:
            self._persisted_N = max(floor, int(bm.N) if bm is not None else floor)
        self._meta_dels = int(getattr(self, '_meta_dels', 0) or 0) + len(dels)
        if self._meta_dels > 4096:           # 删除记录攒多了就整理一次（回到纯追加的干净状态）
            self._write_meta_full()
        return dict(added=len(added), dels=len(dels), bytes=nbytes)

    def _persist_delta(self, new_rows, dead_rows, meta_rows=None) -> dict:
        """把这一批的追加行与墓碑写成 `delta/<seq>.npz`（**只写 O(新 nnz)**）。

        提交语义与段一致：先写 `.tmp`，再 `os.replace` —— rename 才是提交点。
        `base_N` 记下"这一批写之前主表多少行"，回放时按它做链式校验（对不上就停，不猜）。

        `meta_rows`（2026-11-10 补）：这批新 chunk 的身份（docid/file/line/text）也写进 delta。
        为什么必须带着走：写入顺序是 `delta → meta.jsonl → hashes.json`，若崩在 delta 与 meta 之间，
        盘上就会"有行、没身份" —— 重载后这些行占着 N、算进 df，却永远搜不到，而且下次
        `update()` 看到 hash 未变就 noop ⇒ 不会自愈。带上身份之后，rename 就是唯一提交点，
        `meta.jsonl` 降级为快照。
        """
        d = Path(self.dir) / DELTA_SUBDIR
        d.mkdir(parents=True, exist_ok=True)
        seq = 1 + max([int(p.stem) for p in d.glob("*.npz")], default=0)
        bm = self._bm
        T = bm.T.tocsr()
        rows = [int(r) for r in (new_rows if new_rows is not None else [])]
        ptr = [0]
        idx_parts, dat_parts = [], []
        for r in rows:
            lo, hi = int(T.indptr[r]), int(T.indptr[r + 1])
            if hi > lo:
                idx_parts.append(np.asarray(T.indices[lo:hi], dtype=np.int32))
                dat_parts.append(np.asarray(T.data[lo:hi], dtype=np.float32))
            ptr.append(ptr[-1] + max(hi - lo, 0))
        pv = int(getattr(self, "_persisted_V", 0) or 0)
        new_terms = list(bm.idx2term)[pv:] if int(bm.V) > pv else []
        meta_payload = [{k: c.get(k) for k in ("docid", "file", "line", "text", "row")}
                        for c in (meta_rows or [])]
        rec = dict(
            seq=np.int64(seq),
            base_N=np.int64(int(bm.N) - len(rows)),
            N_after=np.int64(int(bm.N)),
            ptr=np.asarray(ptr, dtype=np.int64),
            idx=(np.concatenate(idx_parts) if idx_parts else np.zeros(0, np.int32)),
            dat=(np.concatenate(dat_parts) if dat_parts else np.zeros(0, np.float32)),
            dl=(np.asarray([float(bm.dl[r]) for r in rows], dtype=np.float32)
                if rows else np.zeros(0, np.float32)),
            dead=np.asarray([int(r) for r in (dead_rows if dead_rows is not None else [])],
                            dtype=np.int64),
            V=np.int64(int(bm.V)),
            vocab_json=np.asarray(json.dumps(new_terms, ensure_ascii=False)),
            meta_json=np.asarray(json.dumps(meta_payload, ensure_ascii=False)),
        )
        tmp = d / f"{seq:06d}.npz.tmp"
        with open(tmp, "wb") as fh:                    # ⚠️ 传文件对象：savez 不会再补 .npz
            np.savez_compressed(fh, **rec)
        final = d / f"{seq:06d}.npz"
        os.replace(tmp, final)                          # ⭐ 提交点
        self._delta_nnz += int(rec["idx"].size)
        self._delta_count += 1
        self._persisted_V = int(bm.V)          # 已提交的 delta 已把新词带进盘上的链
        return dict(seq=seq, path=str(final), rows=len(rows), dead=len(rec["dead"]),
                    nnz=int(rec["idx"].size), bytes=int(final.stat().st_size))

    def _merge_meta(self, parts) -> int:
        """把 delta 带来的文档身份并进 `_meta`（按 `row` 去重；⛔ 墓碑行不复活）。"""
        have = {int(c.get('row', -1)) for c in self._meta}
        dead = getattr(self._bm, '_dead', None)
        n_lim = int(self._bm.N)
        add = []
        for c in parts:
            try:
                r = int(c.get('row', -1))
            except (TypeError, ValueError):
                continue
            if r < 0 or r >= n_lim or r in have:
                continue
            if dead is not None and r < int(dead.size) and bool(dead[r]):
                continue                                    # 已删：身份不补，否则"幽灵文档"复活
            have.add(r)
            add.append(c)
        if add:
            self._meta = sorted(self._meta + add, key=lambda c: int(c.get('row', -1)))
            self._rows = None
        return len(add)

    def _replay_deltas(self) -> dict:
        """按 seq 回放 `.delta/*.npz`（链式校验 `base_N`；对不上就停并如实报告）。"""
        files = delta_files(self.dir)
        if not files:
            self._delta_nnz = 0
            self._delta_count = 0
            self._persisted_V = int(self._bm.V)
            return dict(applied=0, stopped=None)
        from .kernel import _sp_vstack
        from scipy import sparse as sp
        bm = self._bm
        blocks, dls, deads = [], [], []
        meta_parts = []
        N, applied, stopped = int(bm.T.shape[0]), 0, None
        total_nnz = 0
        for p in files:
            with np.load(p, allow_pickle=False) as z:
                if int(z["base_N"]) != N:
                    stopped = (p.name, int(z["base_N"]), N)
                    _say(f"[delta] {p.name} base_N={int(z['base_N'])} ≠ 当前 {N}"
                         f" → 停止回放（⛔ 不猜，后续 delta 全部忽略）")
                    break
                k = int(z["dl"].size)
                # 词表先扩到 delta 记的宽度：段带来的新词不在 npz 里，必须由 delta 带着
                vt = json.loads(str(z["vocab_json"])) if "vocab_json" in z.files else []
                if vt:
                    bm.idx2term = list(bm.idx2term) + list(vt)
                    bm.term2idx = {t: i for i, t in enumerate(bm.idx2term)}
                    bm.df = np.concatenate([np.asarray(bm.df, np.int32),
                                            np.zeros(len(vt), np.int32)]).astype(np.int32)
                V = max(int(bm.V), int(z["V"]))
                blk = sp.csr_matrix((np.asarray(z["dat"], np.float32),
                                     np.asarray(z["idx"], np.int32),
                                     np.asarray(z["ptr"], np.int64)), shape=(k, V))
                blocks.append(blk)
                dls.append(np.asarray(z["dl"], np.float32))
                if int(z["dead"].size):
                    deads.append(np.asarray(z["dead"], np.int64))
                N = int(z["N_after"])
                total_nnz += int(np.asarray(z["idx"]).size)
                if "meta_json" in z.files:            # ⭐ 身份跟着 delta 走（见 _persist_delta）
                    meta_parts.extend(json.loads(str(z["meta_json"])))
                applied += 1
        if not blocks:
            self._delta_nnz = 0
            self._delta_count = 0
            return dict(applied=0, stopped=stopped)
        V = int(bm.V)
        if any(b.shape[1] != V for b in blocks):       # 各 delta 的列宽可能不同（词表在长）
            blocks = [b if b.shape[1] == V else
                      sp.csr_matrix((b.data, b.indices, b.indptr), shape=(b.shape[0], V))
                      for b in blocks]
        T0 = bm.T.tocsr()
        if int(T0.shape[1]) != V:                      # ⚠️ 主表也要补到同一列宽，否则 vstack 直接报错
            T0 = sp.csr_matrix((T0.data, T0.indices, T0.indptr),
                               shape=(int(T0.shape[0]), V))
        bm.T = _sp_vstack([T0] + blocks, format="csr").astype(np.float32)
        bm.dl = np.concatenate([np.asarray(bm.dl, np.float32)] + dls)
        # df 必须跟着补：npz 里的 df 不含 delta 行的贡献；每行内列号唯一
        #    ⇒ 每列出现几次就是"几篇文档含该词"（与 `mount_segment` 的 `diff(indptr)` 同义）。
        df_add = np.zeros(int(bm.V), dtype=np.int32)
        for b in blocks:
            if b.nnz:
                df_add += np.bincount(np.asarray(b.indices, np.int64),
                                      minlength=int(bm.V)).astype(np.int32)
        bm.df = (np.asarray(bm.df, np.int32) + df_add).astype(np.int32)
        if int(bm.T.shape[0]) != int(N):
            raise RuntimeError(f"delta 回放后行数 {int(bm.T.shape[0])} ≠ N_after {N}")
        bm.N = int(N)
        if deads:
            dd = np.concatenate(deads)
            dead = (np.zeros(int(bm.N), bool) if bm._dead is None
                    else np.asarray(bm._dead, bool))
            if dead.size < int(bm.N):
                dead = np.concatenate([dead, np.zeros(int(bm.N) - dead.size, bool)])
            for r in dd:
                r = int(r)
                if r < 0 or r >= dead.size or dead[r]:
                    continue
                lo, hi = int(bm.T.indptr[r]), int(bm.T.indptr[r + 1])
                if hi > lo:
                    bm.df[bm.T.indices[lo:hi]] -= 1
                dead[r] = True
            np.maximum(bm.df, 0, out=bm.df)
            bm._dead = dead
        bm._Tbuf = None
        bm._invalidate_w()
        bm._recompute_stats()
        if meta_parts:                       # ⭐ 身份以 delta 为准补齐（meta.jsonl 只是快照）
            added = self._merge_meta(meta_parts)
            if added:
                _say(f"[delta] 从 delta 补回 {added} 条文档身份（meta.jsonl 落后于 delta）")
        self._delta_nnz = int(total_nnz)
        self._delta_count = int(applied)
        self._persisted_V = int(bm.V)
        _say(f"[delta] 回放 {applied} 个 delta → N={int(bm.N):,}（停在 {stopped}）"
             if stopped else f"[delta] 回放 {applied} 个 delta → N={int(bm.N):,}")
        return dict(applied=applied, stopped=stopped)

    def _read_meta(self):
        """读 `meta.jsonl`（支持追加式的 `{"del": [row…]}` 记录）。

        返回 `(entries, 文件里最大 row, 删除标记数)`。
        2026-11-11：二进制 v2（`meta.bin` ＋ 定长记录）实现过并回退了 ——
        10 万条只省 9.5%、加载不变快（瓶颈是构造 dict，不是解析），不值得多维护一套格式。
        见 `06_meta格式_设计与判据_2026-11-11.md §2.5`。
        """
        path = self.dir / 'meta.jsonl'
        if not path.exists():
            return [], -1, 0
        by_row, dels = {}, set()
        with open(path, encoding='utf-8') as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                rec = json.loads(ln)
                if 'del' in rec:
                    for r in rec['del']:
                        dels.add(int(r))
                    continue
                by_row[int(rec.get('row', -1))] = rec
        out = [c for r, c in sorted(by_row.items()) if r not in dels]
        return out, (max(by_row) if by_row else -1), len(dels)

    def _load(self):
        # 索引目录含 dense.npy → 视为 dense 索引 (update/reload 保持 dense 一致)
        self._dense = self._dense or (self.dir / 'dense.npy').exists()
        self._bm = SparseBM25.load(str(self.dir / 'bm25.npz'))
        self._meta, _meta_top, _meta_ndel = self._read_meta()
        self._persisted_N = int(_meta_top) + 1      # 文件里已经有的行；之后的要追加
        self._meta_dels = int(_meta_ndel)
        self._rows = None          # 失效行号缓存
        # 建库分词档随索引落盘（`kernel.stream_mode`），重载后重建仍用它；
        #    缺字段 = 旧版索引 → 按默认档。
        self._stream = _norm_stream(getattr(self._bm, "stream_mode", "off"))
        hf = self.dir / 'hashes.json'
        doc = json.load(open(hf, encoding='utf-8')) if hf.exists() else {}
        if isinstance(doc, dict) and doc.get("v") == 2:      # 0.2.2+：hashes ＋ stamps
            self._hashes = doc.get("hashes") or {}
            self._stamps = {k: (int(v[0]), int(v[1]))
                            for k, v in (doc.get("stamps") or {}).items()}
        else:                                                # 旧格式：rel → md5，无快扫印记
            self._hashes = doc or {}
            self._stamps = {}
        # L2：回放 delta 日志（若有）。必须在挂段之前 —— 段的 row_base 依赖 `bm.N`。
        try:
            self._replay_deltas()
        except Exception as _e:                              # noqa: BLE001
            _say(f"[警告] delta 回放失败（{_e!r}）→ 按 npz 状态可用，但最近的增量可能缺失")
        self._vecs = None
        if self._dense:
            f = self.dir / 'dense.npy'
            if f.exists():
                v = np.load(f)
                if len(v) == len(self._meta):
                    self._vecs = v
                else:
                    _say(f"[警告] dense 向量数({len(v)}) != 活跃 chunk({len(self._meta)}), 忽略 dense")
                    self._vecs = None
        # 临时段发布：按旁车 manifest 重挂已发布的段（无旁车 → 空操作）。
        try:
            _rm = remount_staged(self._bm, self.dir)
            if _rm:
                _ok = [r for r in _rm if r.get("mounted")]
                _skip = [r for r in _rm if not r.get("mounted")]
                _say(f"[staged] 按旁车重挂 {len(_ok)} 个段"
                     + (f" · 跳过 {len(_skip)}（{_skip[0].get('reason', '')[:60]}）" if _skip else ""))
                # 乙档：把已挂段里的文档身份补回 `_meta`（否则段内文档在产品层不可见）
                _ab = self._absorb_staged(_ok)
                if _ab["added"]:
                    _say(f"[staged] 已把 {_ab['added']} 篇段内文档补回 `_meta`"
                         f"（产品层可见；跳过 {_ab['skipped']}）")
        except Exception as _e:                                      # noqa: BLE001
            _say(f"[警告] 旁车重挂失败（{_e!r}）→ 该索引按 npz 状态可用，但已发布的段未挂上")

    def _absorb_staged(self, mounted) -> dict:
        r"""把已挂段里的文档身份补回 `self._meta`（乙档：产品层可见性的唯一写入方）。

        行号以实际挂载位置为准（`bm._segments[tag]['row_base']`），不信侧车里记的
        `row` —— `reset`/多次重挂之后同一段的行号可能与发布时不同。
        幂等（按物理行去重）：`save()` 在段在场时直接抛错（`kernel.py` 的 `save`），
        所以"`_meta` 已有、段又挂上"的结构性来源不存在；去重是为将来放宽留的保险。
        `_vecs` 与 `_meta` 逐位对齐 ⇒ 追加后长度不符就必须弃用 dense并如实告警。
        """
        added = skipped = 0
        have = {int(m.get("row", -1)) for m in (self._meta or [])}
        segs = getattr(self._bm, "_segments", None) or {}
        for r in mounted or []:
            if not r.get("mounted"):
                continue
            tag = r.get("tag")
            info = segs.get(tag) or {}
            rb = int(info.get("row_base", -1))
            docs = load_seg_docs(self.dir, tag)
            if rb < 0 or not docs:
                skipped += len(docs)
                _say(f"[staged] 段 {tag} 没有文档身份侧车（{SEG_DOCS}）"
                     f"→ 该段 {info.get('rows', '?')} 行在产品层**不可见**（如实降级，不猜身份）")
                continue
            for i, d in enumerate(docs):
                row = rb + i
                if row in have:
                    continue
                docid = str(d.get("docid", f"seg:{tag}:{i}"))
                self._meta.append(dict(docid=docid, row=row,
                                       file=d.get("file") or docid,
                                       line=int(d.get("line") or 1),
                                       text=d.get("text") or ""))
                have.add(row)
                added += 1
        if added:
            self._rows = None            # ⭐ 失效行号缓存（新数组对象 ⇒ 恒等判定自动作废）
            if self._vecs is not None and len(self._vecs) != len(self._meta):
                self._vecs = None
                _say("[警告] 段内文档已并入 `_meta`，但 dense 向量数不匹配 → 本次**忽略 dense**"
                     "（重编码后可恢复）")
        return dict(added=added, skipped=skipped)

    def _encode_dense(self):
        try:
            model = get_dense_model(self._model_path)
        except FileNotFoundError as e:
            _say(f"[提示] {e}")
            return
        if model is None:
            return
        t0 = time.time()
        vecs = model.encode([c['text'] for c in self._meta], batch_size=16,
                            normalize_embeddings=True, convert_to_numpy=True,
                            show_progress_bar=True).astype('float32')
        np.save(self.dir / 'dense.npy', vecs)
        _say(f"dense 向量 {vecs.shape} 落盘 "
              f"{time.time() - t0:.0f}s ({len(self._meta) / (time.time() - t0):.1f} 段/s CPU)")
        self._vecs = vecs

    # ---------- 统计 ----------

    @property
    def info(self):
        if self._bm is None:
            self._load()
        bm_path = self.dir / 'bm25.npz'
        # 新格式无 W：形状/规模走 SparseBM25.V/.nnz（权威 T 优先，旧格式回退 W）
        return dict(N=len(self._meta),
                    V=int(self._bm.V), nnz=int(self._bm.nnz),
                    size_mb=round(bm_path.stat().st_size / 1e6, 2) if bm_path.exists() else 0.0,
                    dense=(self._vecs is not None),
                    files=len(self._hashes))

    # ---------- 检索 ----------

    def _active_rows(self):
        r"""活跃 chunk 的物理行号数组 (缓存)。

        只在 `_meta` 变化时变, 故缓存; 每个 `self._meta =` 赋值点都要把 `self._rows`
        置 `None` (与 `_w_persist_key` 同一条纪律: 键不符还用 = 静默错分)。
        """
        if self._rows is None:
            n = len(self._meta)
            self._rows = (np.fromiter((c['row'] for c in self._meta), dtype=np.int64, count=n)
                          if n else np.zeros(0, dtype=np.int64))
            # 01m：顺手判「恒等」（`_meta` 的行号恰是 `0..N-1` 且与物理行一一对应）。
            #    判一次 O(n)（只在这条缓存重建时付）；此后 `_score_active` 每查询免掉
            #    用对象身份当失效键：六处 `self._rows = None` 的失效点都会让本函数
            #    重建出一个新数组对象 ⇒ 这条判定自动作废，无须（也不会）漏改失效点。
            #    失效是惰性的：`_rows = None` 之后、下一次 `_active_rows()` 之前，
            #    生产路径满足这一点（`_score_active` 先调 `_active_rows()`，另有
            #    `ident[0] is rows` 兜底）；外部读它前必须先自己调一次 `_active_rows()`。
            bm = self._bm
            ok = False
            if n and bm is not None and n == int(bm.N):
                ok = bool((self._rows == np.arange(n, dtype=np.int64)).all())
            self._rows_ident = (self._rows, ok)
        return self._rows

    def _score_active(self, qtoks):
        """返回与 `self._meta` 对齐的 BM25 分数数组（按每条 row 取物理行分）。

        01m：`_meta` 行号恒等时（新建索引未删未迁的常见形态）直接返回
        `score_all` 的数组 —— 省掉一整趟 `gather` ＋ `astype` 拷贝。
        返回的是 `score_all` 当次新建的数组（不与任何常驻缓冲共享）⇒ 调用方可安全
        持有；但调用方 不得就地改写它（本包内唯一调用方 `query` 只读它）。
        """
        scores = self._bm.score_all(qtoks)
        rows = self._active_rows()
        if rows.size == 0:
            return np.zeros(0, dtype=np.float32)
        ident = self._rows_ident
        if (ident is not None and ident[0] is rows and ident[1]
                and scores.size == rows.size and scores.dtype == np.float32):
            return scores
        return scores[rows].astype(np.float32)

    def query(self, q, k=10, dense=False):
        """检索 → [{'rank','score','file','line','text'}]

        dense=True 且索引有 dense.npy → BM25 粗筛后做 dense 精排（懒编码）。
        """
        if self._bm is None:
            self._load()
        s = self._score_active(zh_tokenize(q))     # 与 self._meta 对齐
        if dense and self._vecs is not None:
            cand = np.unique(_topk_idx(s, max(k * 10, 100)))
            cand = cand[s[cand] > 0] if (s[cand] > 0).any() else cand
            model = get_dense_model(self._model_path)
            if model is not None:
                qv = model.encode([q], normalize_embeddings=True).astype(np.float32)[0]
                dsims = self._vecs[cand] @ qv
                s_d = np.zeros(len(self._meta), dtype=np.float32)
                s_d[cand] = dsims
                b = s / max(s.max(), 1e-9)
                d = s_d / max(s_d.max(), 1e-9)
                fused = 0.6 * d + 0.4 * b
                order = _topk_idx(fused, k)
            else:
                order = _topk_idx(s, k)
        else:
            order = _topk_idx(s, k)
        out = []
        for rank, i in enumerate(order, 1):
            if s[i] <= 0:
                break
            m = self._meta[int(i)]
            out.append(dict(rank=rank, score=float(s[i]), docid=m.get('docid'),
                            file=m['file'], line=m['line'], text=m['text']))
        return out

    # ---------- 浏览 ----------

    def browse(self, file=None, page=0, n=20):
        if self._meta is None:
            self._load()
        if file is not None:
            file_n = file.replace('\\', '/')
            idxs = [i for i, m in enumerate(self._meta)
                    if m['file'].replace('\\', '/') == file_n]
        else:
            idxs = list(range(len(self._meta)))
        start = page * n
        return [dict(docid=self._meta[i].get('docid'), file=self._meta[i]['file'],
                     line=self._meta[i]['line'], text=self._meta[i]['text'])
                for i in idxs[start:start + n]], len(idxs)

    def list_files(self):
        if self._meta is None:
            self._load()
        return sorted({m['file'] for m in self._meta})

    # ---------- 真增量 update（方案 A） ----------

    def compact_segments(self, *, persist=True) -> dict:
        """把已挂段全部折进主表 ＋（可选）落盘 ＋ 清分块缓存。

        `api.Index.compact()` 与 `update()` 的 `fold_first` 出口共用这一个实现。
        `persist` 三态：

        · `True`：折叠 → `_persist()`（先落盘）→ 追加 `folded_all`（提交点）。
          反过来就是"先提交后落盘"：崩溃时旁车说已折叠、npz 里没有那些行 ⇒ 段内容静默丢失。
        · `False`（L1）：只折内存，npz 与 `folded_all` 都不写 —— 盘上保持
          「旧主表 ＋ 段清单」的等价状态，重载按旁车重挂即可复原。
        · `"delta"`（L2/S4）：把折出来的行写成一条 delta，再追加 `folded_all`。
          顺序是 delta → meta/hashes → 事件：若在 delta 与事件之间崩，重载会重挂段，
          但此时 T 已因 delta 变长、`T.shape[0] > row_base` 会把重挂跳过 ⇒ 不重复计数。
        `False` 时两件事必须同时省：只省 npz 却写事件 ⇒ 重载以为已折、盘上却没有那些行。
        """
        bm = self._bm
        segs = list(bm.segment_report())
        if not segs:
            return dict(folded=[], tags=[], blocks=len(bm._tc_blocks or []),
                        N=int(bm.N), V=int(bm.V), ms=0.0, via=None, persisted=False,
                        persist_mode=persist)
        t0 = time.time()
        n0 = int(bm.T.shape[0])                # ⚠️ 折叠前的**主表**行数（`bm.N` 已含段行，不能用它）
                                               #    折出来的行是 [n0, N)；delta 的 base_N 也必须是 n0
        if len(segs) > 1:
            # 多段时走内核的 `fold_all()`（`fold_segment` 的不变量在多段下不成立）
            info = bm.fold_all()
            bm._tc_blocks, bm._tc_rows = [], 0
            bm._invalidate_w()
            bm._recompute_stats()
            bm._ensure_tc()
            tags = list(info["tags"])
            folded = [dict(tag=t, **{k: v for k, v in info.items() if k != "tags"})
                      for t in tags]
            via = "fold_all"
        else:
            folded = [dict(tag=t, **bm.fold_segment(t)) for t in segs]
            tags = [f["tag"] for f in folded]
            bm._tc_blocks, bm._tc_rows = [], 0
            bm._invalidate_w()
            bm._recompute_stats()
            bm._ensure_tc()                       # 立刻建回单块（否则下一次查询才建）
            via = "fold_segment"
        if persist is True or persist == "delta":
            if persist == "delta":
                _n1 = int(bm.N)
                self._persist_delta(range(n0, _n1), [],
                                    meta_rows=[c for c in self._meta
                                               if n0 <= int(c.get('row', -1)) < _n1])
                self._meta_append_pending()        # ⭐ 追加式：折出来的行是新的尾部
                self._write_hashes()
            else:
                self._persist()                   # ⭐ 先落盘（折叠改了 T 与 meta）
            manifest_append(self.dir, {"event": "folded_all", "tags": tags,
                                       "via": f"compact/{via}"})
        self._rows = None
        return dict(folded=folded, tags=tags, blocks=len(bm._tc_blocks or []),
                    N=int(bm.N), V=int(bm.V), ms=(time.time() - t0) * 1000.0, via=via,
                    persisted=bool(persist), persist_mode=persist)

    def update(self, root, workers=0, *, verify="auto"):
        """hash diff ＋ 路由（`primolix.routing.route_update`）。

        四个出口，返回里的 `kind` 与之一一对应：
            noop         无改动
            incremental  词表内 → 就地真增量
            staged       含新词 → 挂只读段（`O(新 nnz)`，不重建）
            rebuild      兜底：某条 precondition 不满足（dense / 段数上限 / 批太大）
        判不了或安全执行不了（例如段在场却只能重建）→ 抛 `RuntimeError`，
        不静默选一条。判据见 `04c_丙档OOV路由_设计与判据_2026-11-08.md`。

        `verify`（0.2.2）：
            `"auto"`（默认）—— 有 `(size, mtime_ns)` 印记就走 `"stat"`，否则退回 `"full"`；
            `"stat"`        —— 只比元数据，没变就不读内容（实测 ~1/10 成本）；
            `"full"`        —— 每个文件读内容算 md5（保留 mtime 的还原类工具必须用这档）。
        返回里带 `verify`，如实说明这一批到底校验到了什么程度。
        """
        if self._bm is None:
            self._load()
        self._dense = self._dense or (self.dir / 'dense.npy').exists()
        t0 = time.time()
        mode = verify
        if mode == "auto":
            mode = "stat" if getattr(self, "_stamps", None) else "full"
        if mode not in ("full", "stat"):
            raise ValueError(f"verify 只接受 'auto' / 'full' / 'stat'，实得 {verify!r}")
        # 排除索引输出目录自身 (可能在 root 内部, 如 <root>/.idx)
        files, stamps_now = scan_root_ex(Path(root), [self.dir], known=(self._hashes or {}),
                                         stamps=getattr(self, "_stamps", None) or {},
                                         verify=mode)
        new = {rel: h for rel, h, _ in files}
        old = self._hashes
        added = [r for r in new if r not in old]
        removed = [r for r in old if r not in new]
        changed = [r for r in new if r in old and new[r] != old[r]]
        unchanged = len(new) - len(added) - len(changed)
        _say(f"S4 hash diff（verify={mode}）: 新增 {len(added)} / 删除 {len(removed)} / "
              f"变更 {len(changed)} / 跳过 {unchanged} ({unchanged / max(len(new), 1):.0%})")
        if not (added or removed or changed):
            return {'kind': 'noop', 'detail': '库已同步, 无需重建', 'verify': mode}

        # ---- 收集新 chunk (added 全文件 + changed 文件的新内容)，分词一次复用 ----
        rel2scan = {x[0]: x for x in files}
        new_chunk_texts = []
        for rel in added + changed:
            f = rel2scan.get(rel)
            if f is not None:
                for t, _, _ in f[2]:
                    new_chunk_texts.append(t)
        chunk_tokens = [zh_tokenize(t) for t in new_chunk_texts]
        append_chunks = []
        for rel in added + changed:
            f = rel2scan.get(rel)
            if f is None:
                continue
            h = f[1]
            for t, _, ln in f[2]:
                append_chunks.append(dict(docid=f'{h[:8]}:{len(append_chunks)}',
                                          file=rel, line=ln, text=t))

        # ---- 墓碑/保留：整文件级替换（removed + changed 的旧 chunk 全剔除） ----
        drop_rels = set(removed) | set(changed)
        tomb_rows = [c['row'] for c in self._meta if c['file'] in drop_rels]
        keep_meta = [c for c in self._meta if c['file'] not in drop_rels]
        # 行号 ≥ T 行数 ⇒ 该行落在段里（段内不支持删除）—— 与"这批带不带新词"无关
        seg_floor = int(self._bm.T.shape[0])
        n_tomb_seg = sum(1 for r in tomb_rows if int(r) >= seg_floor)

        # ---- 路由 ----
        oov = self._bm.oov_terms(chunk_tokens) if chunk_tokens else set()
        segs_n = len(self._bm.segment_report())
        blocks_n = len(getattr(self._bm, "_tc_blocks", []) or []) or (segs_n + 1)
        route = route_update(
            n_added=len(added), n_changed=len(changed), n_removed=len(removed),
            n_oov=len(oov), n_tomb_rows=len(tomb_rows), n_tomb_seg_rows=n_tomb_seg,
            n_segments=segs_n, n_blocks=blocks_n,
            dense_on=getattr(self, "_vecs", None) is not None,
            batch_nnz=int(sum(len(t) for t in chunk_tokens)),
            total_nnz=int(getattr(self._bm, "nnz", 0)),
        )
        _say(f"[路由] {route['route']}（{route['code']}）：{route['detail']}")
        fold_note = ""
        cinfo = None
        if route['route'] == FOLD_FIRST:
            # A / C：当场折回主表（不再让调用方自己去 compact）。
            #    `rebuild` 直接 `rebuild_in_place` —— 所以折叠不必再写一次盘（原先会连写两次）。
            # 例外：`staged` 分支只写 meta/hashes、不写 npz；若折叠也不落盘，
            #    meta.jsonl 里的行数会超过盘上主表行数 ⇒ 重载时 `scores[rows]` 越界。
            #    所以 `then == staged` 时折叠照旧落盘。
            # 折叠的落盘方式按持久化模式分档（L2/S4）：
            #    delta 模式 ⇒ 折出来的行写成一条 delta（O(段 nnz)），盘上链不断；
            #    npz 模式 ＋ then == staged ⇒ 必须落盘（staged 只写 meta/hashes，折叠不落盘会让 meta 账不平）；
            #    npz 模式 ＋ then ∈ {incremental, rebuild} ⇒ 后续分支自己会全量落盘，折叠不必再写。
            if PERSIST_MODE == "delta":
                fold_persist = "delta"
            else:
                fold_persist = True if route['then'] == STAGED else False
            cinfo = self.compact_segments(persist=fold_persist)
            fold_note = (f"folded={len(cinfo['tags'])} fold_ms={cinfo['ms']:.1f}"
                         + ("" if cinfo['persisted'] else " (deferred)"))
            _say(f"[折] {cinfo['tags']} → 主表 {cinfo['N']} 行（{cinfo['ms']:.1f} ms"
                 f"{'' if cinfo['persisted'] else '，暂不落盘'}）")
            seg_floor = int(self._bm.T.shape[0])
            n_tomb_seg = 0
            route = dict(route, route=route['then'], detail=route['detail'])
        if route['route'] == REBUILD:
            self.rebuild_in_place(root, workers=workers)
            _say(f"S4 回退重建完成 {time.time() - t0:.0f}s")
            return {'kind': 'rebuild',
                    'detail': f"oov={len(oov)} why={route['code']} {fold_note}".strip(),
                    'verify': mode}

        # ---- staged：不重建，把这批挂成只读段 ----
        if route['route'] == STAGED:
            base_V = int(len(self._bm.idx2term))
            base_T_rows = int(self._bm.T.shape[0])
            new_pos, new_terms = {}, []
            for toks in chunk_tokens:
                for t in toks:
                    if t in self._bm.term2idx or t in new_pos:
                        continue
                    new_pos[t] = base_V + len(new_terms)
                    new_terms.append(t)
            tag = time.strftime("%Y%m%d-%H%M%S") + f"-{os.getpid()}"
            if tomb_rows:
                self._bm._tombstone_rows(tomb_rows)
            self._meta = keep_meta
            rows = [(c['docid'], toks, c['file'], c['line'], c['text'])
                    for c, toks in zip(append_chunks, chunk_tokens)]
            out = publish_segment(self, rows, tag=tag, generation=tag,
                                  base_V=base_V, base_T_rows=base_T_rows,
                                  new_terms=new_terms, new_pos=new_pos)
            self._rows = None          # ⭐ 失效行号缓存（段内文档已并入 _meta）
            self._hashes = new
            self._stamps = {rel: stamps_now[rel] for rel in new if rel in stamps_now}
            # 段在场 ⇒ `_persist()` 里的 `bm.save()` 会抛错：只落 meta(主表行)＋hashes。
            self._persist_under_segments(del_rows=tomb_rows)
            _say(f"S4 挂段完成 {time.time() - t0:.0f}s (新词 {len(oov)}, 段 {out['tag']}, "
                  f"墓碑 {len(tomb_rows)} 行, 活跃 {len(self._meta)})")
            return {'kind': 'staged',
                    'detail': f"oov={len(oov)} tag={out['tag']} rows={out['rows']} "
                              f"tomb={len(tomb_rows)} {fold_note}".strip(),
                    'folded_tags': len(cinfo['tags']) if cinfo else 0,
                    'verify': mode}

        # ---- incremental：词表内就地真增量 ----
        if tomb_rows:
            self._bm._tombstone_rows(tomb_rows)
        new_rows = self._bm.append_docs(chunk_tokens)
        for c, row in zip(append_chunks, new_rows):
            c['row'] = int(row)
        self._meta = keep_meta + append_chunks
        self._rows = None          # ⭐ 失效行号缓存（行号已迁移/追加）
        self._hashes = new
        self._stamps = {rel: stamps_now[rel] for rel in new if rel in stamps_now}
        # 持久化分档（L2）：delta 模式只写 O(新 nnz)，npz 模式维持原行为
        if PERSIST_MODE == "delta":
            dinfo = self._persist_delta(new_rows, tomb_rows, meta_rows=append_chunks)
            self._meta_append_pending(del_rows=tomb_rows)   # ⭐ 追加式：O(新增行)，不重写整份
            self._write_hashes()
            persist_note = f"delta#{dinfo['seq']}({dinfo['nnz']} nnz)"
            cauto = self._maybe_compact_delta()          # L2/S3：两层压实
            if cauto:
                persist_note += f"+压实{cauto.get('tier', '?')}"
        else:
            self._persist()      # 走到这里 ⇒ 段已不在场（要折的已在上面折过）
            persist_note = "npz"
        # dense: update 后整库重对齐 (rows 迁移, 旧向量错位; 小语料可负担)
        if self._dense:
            self._encode_dense()
        _say(f"S4 真增量完成 {time.time() - t0:.0f}s "
              f"(墓碑 {len(tomb_rows)} 行, 追加 {len(append_chunks)} chunk, "
              f"活跃 {len(self._meta)})")
        return {'kind': 'incremental',
                'detail': f'tomb={len(tomb_rows)} append={len(append_chunks)} '
                          f'{fold_note} {persist_note}'.strip(),
                'folded_tags': len(cinfo['tags']) if cinfo else 0,
                'verify': mode}

    def rebuild_in_place(self, root, workers=0, *, force=False):
        """整库重建到同一目录（词表外回退 / 显式重建）。保持 dense 开关。

        重建写出全新的词表/主表 → 旁车里那些段（基世代不同）不再适用，故追加一条 `reset`
        事件（旁车追加式，不改写历史行）。段在场时默认拒绝：已发布但未折叠的段其内容不在
        源语料里、重建必然丢掉它们；要丢就显式 `force=True`。
        """
        if not force:
            _segs = (self._bm.segment_report() if self._bm is not None else {}) or {}
            if _segs:
                raise RuntimeError(
                    f"已挂段 {sorted(_segs)} 且尚未折叠：它们的内容不在源语料里，"
                    f"整库重建会丢掉这些已发布的数据 → 先 `compact()`（折进主表并保存），"
                    f"或显式 `rebuild(..., force=True)` 承认丢弃")
        z = Primolix.build(root, str(self.dir), dense=self._dense,
                        model_path=self._model_path, workers=workers,
                        skip_dirs=[self.dir], stream=getattr(self, "_stream", False))
        manifest_append(self.dir, {"event": "reset", "reason": "rebuild_in_place"})
        self._bm = z._bm
        self._meta = z._meta
        self._rows = None          # ⭐ 失效行号缓存
        self._hashes = z._hashes
        self._stamps = z._stamps        # 重建后 size/mtime 印记一并换新，下一次可走 stat 快扫
        self._vecs = z._vecs

    # 视图按 meta 位置（chunk）对齐挂载，内核侧换算成物理行 + 覆盖位图：update 真增量后物理行号
    # 不迁移、meta 只含活跃 chunk → 旧视图对死行自动失效、对新行自动"未覆盖"（滞后是合法终态）。
    # sidecar `<dir>/views/<name>.npz` 记 docid ↔ 向量（不记物理行），重启后按 docid 重匹配，
    # 对不上的如实计入 `unmatched`（那是"视图滞后于库"，不是错误）；视图可丢（npz 是唯一持久化物）。

    def attach_view(self, name, vectors, meta_positions=None, *,
                    available=True, meta=None, replace=False):
        """把 [n, dim] 向量按 meta 位置挂成派生视图（换算成物理行）。

        `meta_positions` 缺省 = 全部活跃 chunk。返回挂载后的视图报告行。
        """
        if self._bm is None:
            self._load()
        pos = (np.arange(len(self._meta), dtype=np.int64) if meta_positions is None
               else np.asarray(meta_positions, dtype=np.int64))
        if pos.ndim != 1 or (pos.size and (pos.min() < 0 or pos.max() >= len(self._meta))):
            raise ValueError(f"meta_positions 越界（活跃 chunk 数={len(self._meta)}）")
        if len(set(pos.tolist())) != pos.size:
            raise ValueError("meta_positions 有重复 —— 一个 chunk 至多一个向量")
        rows = np.asarray([int(self._meta[i]["row"]) for i in pos], dtype=np.int64)
        m = dict(meta or {})
        m.setdefault("n_active", len(self._meta))
        m.setdefault("covered", int(pos.size))
        return self._bm.attach_view(name, vectors, rows=rows, available=available,
                                    meta=m, replace=replace)

    def detach_view(self, name):
        if self._bm is None:
            self._load()
        return self._bm.detach_view(name)

    def view_report(self):
        """视图清单（接口 C 消费侧；VIEW_STATUS / 降级矩阵的读数来源）。"""
        if self._bm is None:
            self._load()
        return self._bm.view_report()

    def covered_mask(self, view_name):
        """视图对活跃 chunk 的覆盖位图（bool[len(_meta)]；无视图 → None）。

        未覆盖的活跃 chunk = 检索语义里的"冷"（无向量但仍在 S1 池）。
        """
        if self._bm is None:
            self._load()
        v = self._bm._views.get(str(view_name))
        if v is None:
            return None
        rows = v["rows"]
        cov = np.zeros(len(self._meta), dtype=bool)
        if len(self._meta) == 0 or rows.size == 0:
            return cov
        meta_rows = np.asarray([int(m["row"]) for m in self._meta], dtype=np.int64)
        hit = np.isin(meta_rows, rows)
        cov[hit] = True
        return cov

    def fused_scores(self, qtoks, qvec, view_name, alpha=0.35):
        """与 `_meta` 对齐的融合分数（视图不在场 → 逐位 `_score_active`）。"""
        if self._bm is None:
            self._load()
        base = self._bm.fused_scores(qtoks, qvec, view_name, alpha=alpha)
        if len(self._meta) == 0:
            return base
        rows = np.asarray([m["row"] for m in self._meta], dtype=np.int64)
        return base[rows].astype(np.float32)

    def save_view(self, name, docids, vectors, meta=None):
        """视图 sidecar 落盘：`<dir>/views/<name>.npz`（docid ↔ 向量，不记物理行）。

        `docids[i] ↔ vectors[i]`；`meta` 至少要有 `model_id`（码本/模型代次冻结记录位）
        与 `space`/`dim`。返回 sidecar 路径。
        """
        V = np.asarray(vectors, dtype=np.float32)
        if V.ndim != 2 or len(docids) != V.shape[0]:
            raise ValueError(f"docids({len(docids)}) 与 vectors({V.shape}) 不对齐")
        vdir = self.dir / "views"
        vdir.mkdir(parents=True, exist_ok=True)
        m = dict(meta or {})
        m.update(name=str(name), dim=int(V.shape[1]), n_vectors=int(V.shape[0]))
        path = vdir / f"{str(name)}.npz"
        np.savez_compressed(path, vecs=V,
                            docids=np.asarray([str(d) for d in docids]),
                            meta=np.asarray(json.dumps(m, ensure_ascii=False)))
        _say(f"[views] 视图 {name} 落盘 {path}（{V.shape[0]} 条 × {V.shape[1]} 维）",
              file=sys.stderr)
        return path

    def load_views(self, extra_dirs=None):
        """扫视图 sidecar 并按 docid ↔ meta 重挂。

        扫描面 = `<dir>/views/*.npz`（索引自带）＋ `extra_dirs` 里每个目录的 `views/*.npz`；
        本方法不搬文件。对不上的 docid 如实计入 `unmatched`（"视图滞后于库"，合法终态，
        不许静默丢弃也不许当成错误）。返回逐视图的装载报告（含 unmatched）。
        """
        if self._bm is None:
            self._load()
        vdirs = [self.dir / "views"]
        for d in (extra_dirs or []):
            p = Path(d)
            vdirs.append(p / "views" if p.name != "views" else p)
        out = []
        seen = set()
        sidecars = []
        for vd in vdirs:
            if vd.is_dir():
                for f in sorted(vd.glob("*.npz")):
                    key = str(f.resolve())
                    if key not in seen:
                        seen.add(key)
                        sidecars.append(f)
        pos_of = {}
        for i, m in enumerate(self._meta):
            key = m.get("docid") or m.get("chunk_id")
            if key is not None and key not in pos_of:
                pos_of[key] = i
        for f in sidecars:
            try:
                z = np.load(f, allow_pickle=False)
                docids = [str(x) for x in z["docids"]]
                vecs = np.asarray(z["vecs"], dtype=np.float32)
                meta_info = json.loads(str(z["meta"])) if "meta" in z else {}
            except Exception as e:                         # noqa: BLE001
                out.append(dict(name=f.stem, attached=False,
                                unmatched=0, reason=f"sidecar 读不出来: {e}"))
                continue
            hit_pos = [pos_of[d] for d in docids if d in pos_of]
            hit_vec = [i for i, d in enumerate(docids) if d in pos_of]
            unmatched = len(docids) - len(hit_pos)
            name = str(meta_info.get("name") or f.stem)
            if not hit_pos:
                out.append(dict(name=name, attached=False, unmatched=unmatched,
                                reason="sidecar 的 docid 与当前库零交集（库已重建/漂移）"))
                continue
            v = vecs[np.asarray(hit_vec, dtype=np.int64)]
            info = dict(meta_info)
            info["unmatched"] = unmatched
            self.attach_view(name, v, np.asarray(hit_pos, dtype=np.int64),
                             meta=info, replace=True)
            out.append(dict(name=name, attached=True, unmatched=unmatched,
                            covered=int(len(hit_pos))))
        return out


# ================================================================ 临时段发布：旁车

SEG_SUBDIR = ".segments"
SEG_MANIFEST = "manifest.jsonl"
# 段级文档身份侧车（乙档）：一行一篇 `{docid,row,file,line,text}`，与段同生共死。
#    （写到一半崩 ⇒ 历史身份全丢）。侧车是 append-only 的，且先于 manifest 提交写好。
SEG_DOCS = "docs.jsonl"

# ================================================================ L2：增量持久化开关
# "npz"（默认）＝每次 `_persist()` 重写整份 npz（0.2.1 及以前的行为）；
# "delta"     ＝平时只写 `delta/<seq>.npz`（O(新 nnz)），`_persist()` 只在压实时发生。
# 默认仍是 "npz"：S1 只加分支，现网行为一字不变。切换默认要等 L2-1…L2-5 全过。
PERSIST_MODE = "npz"
DELTA_SUBDIR = ".delta"
DELTA_COMPACT_RATIO = 0.3        # `[提案]`：Σdelta.nnz > 0.3 × base.nnz 就该压实
DELTA_MAX_FILES = 32             # `[提案]`：delta 文件数上限（加载要逐个回放）


def set_persist_mode(mode: str) -> str:
    """切换持久化模式（进程级）。返回切换后的值。"""
    global PERSIST_MODE
    if mode not in ("npz", "delta"):
        raise ValueError(f"PERSIST_MODE 只接受 'npz' / 'delta'，实得 {mode!r}")
    PERSIST_MODE = mode
    return PERSIST_MODE


def delta_files(index_dir) -> list:
    """按 seq 升序返回**当前生效**的 `<index>/.delta/*.npz`。

    · 忽略 `.tmp`：未 rename ＝ 未提交。
    · 合并（2026-11-10）：一个 delta 可以在自己的 `supersedes` 里声明"我取代了哪些 seq"。
      被声明的文件即使还在盘上（删除失败或崩在删除之前）也不进生效集 ⇒
      合并的提交点仍然是那一次 `rename`，不需要第二份清单。
    """
    d = Path(index_dir) / DELTA_SUBDIR
    if not d.exists():
        return []
    out = [p for p in d.glob("*.npz") if p.stem.isdigit()]
    out.sort(key=lambda p: int(p.stem))
    if not out:
        return []
    sup = set()
    for p in out:
        try:
            with np.load(p, allow_pickle=False) as z:
                if "supersedes" in z.files:
                    sup.update(int(x) for x in np.asarray(z["supersedes"]).ravel().tolist())
        except (OSError, ValueError, KeyError):
            continue
    return [p for p in out if int(p.stem) not in sup]


def seg_docs_path(index_dir, tag) -> Path:
    """段 `tag` 的文档身份侧车路径（与段的三个 `.npy` 同目录）。"""
    return Path(index_dir) / SEG_SUBDIR / str(tag) / SEG_DOCS


def load_seg_docs(index_dir, tag) -> list:
    """读段 `tag` 的文档身份侧车；不存在/空 ⇒ `[]`（老段没有侧车，如实降级）。"""
    p = seg_docs_path(index_dir, tag)
    if not p.exists():
        return []
    out = []
    for ln in p.read_text(encoding="utf-8").splitlines():
        if ln.strip():
            try:
                out.append(json.loads(ln))
            except ValueError:
                break                     # 半行（写崩）⇒ 只认前缀，⛔ 不猜
    return out


def manifest_append(index_dir, rec: dict):
    """向旁车 manifest 追加一条事件（追加式：从不改写历史行）。

    事件三种：`publish`（提交点）· `folded_all`（段已折进主表）· `reset`（整库重建）。
    """
    d = Path(index_dir) / SEG_SUBDIR
    d.mkdir(parents=True, exist_ok=True)
    p = d / SEG_MANIFEST
    rec = dict(rec)
    rec.setdefault("ts", time.time())
    with open(p, "a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return p


def _manifest_events(index_dir) -> list:
    p = Path(index_dir) / SEG_SUBDIR / SEG_MANIFEST
    if not p.exists():
        return []
    out = []
    for ln in p.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        try:
            out.append(json.loads(ln))
        except ValueError:
            out.append({"event": "corrupt", "raw": ln[:80]})
    return out


def remount_staged(bm, index_dir) -> list:
    """按旁车 manifest 重挂已发布的段（`Primolix._load` 会自动调一次）。

    回放规则（追加式）：`reset` / `folded_all` → 清空在挂清单；`publish` → 加入。
    防重复计数：主表行数已超过该段基行号 → 该段内容已在主表里（被折叠且已保存）→ 跳过并如实报。
    基世代不符（词表变过）→ `mount_segment` 抛错 → 如实记 `stale`，不静默重映射。
    """
    active = []
    for ev in _manifest_events(index_dir):
        kind = ev.get("event", "publish")
        if kind in ("reset", "folded_all"):
            active = []
        elif kind == "publish":
            active.append(ev)
    out = []
    base = Path(index_dir) / SEG_SUBDIR
    for rec in active:
        tag = rec.get("tag")
        rows = int(rec.get("rows", 0))
        try:
            # 防重复计数：主表行数长过该段基行号 → 内容已在主表里 → 跳过。
            # 不能用 `!=` 判：多段时 T 比后面每个段的基行号都短（挂段不动 T）→ 会把它们误判成"已折叠"而静默丢段。
            if int(bm.T.shape[0]) > int(rec.get("row_base", -1)):
                out.append(dict(tag=tag, mounted=False, reason=(
                    f"主表行数 {int(bm.T.shape[0])} 已超过该段基行号 {rec.get('row_base')}"
                    f" → 判定已折叠进主表（跳过，避免重复计数）")))
                continue
            dl = np.load(base / str(tag) / "dl.npy")
            # 只补"还不在当前词表里"的新词：旁车里 new_terms 若已被 save() 写进 npz，
            # 再加一遍会造出重复列 → 只传缺的那些，段的列宽由 mount_segment 加空列补齐。
            _missing = [t for t in (rec.get("new_terms") or []) if t not in bm.term2idx]
            bm.mount_segment(str(base / str(tag)), str(tag), _missing,
                             dl, rows=rows, generation=rec.get("generation"))
            out.append(dict(tag=tag, mounted=True, rows=rows, base_V=rec.get("base_V"),
                            new_terms=_missing))
        except Exception as e:                                        # noqa: BLE001
            out.append(dict(tag=tag, mounted=False,
                            reason=f"stale/不可挂：{type(e).__name__}: {str(e)[:90]}"))
    return out


def publish_segment(z, rows, *, tag, generation=None, base_V=None, base_T_rows=None,
                    new_terms=(), new_pos=None, t2i=None, fold=False) -> dict:
    r"""把一批 `(docid, toks, file, line, text)` 写成只读段并挂上。

    `Stage.publish()`（门面的旁路累积）与 `Primolix.update()` 的 `staged` 分支
    共用这一个实现 —— 段的提交顺序只有一处，不许各写一份。

    顺序（既有不变量，不许调换）：
        三个 `.npy` → `dl.npy` → 文档身份侧车 → `manifest` 追加（提交点）
        → `mount_segment` → `_absorb_staged`（段内文档对产品层可见）。

    调用方须已持写锁（`Index.add` / `Stage.publish` 都在锁内）。
    段在场时 `SparseBM25.save()` 会抛错 ⇒ 调用方不能顺手 `_persist()`。
    """
    from collections import Counter

    from scipy import sparse as sp

    bm = z._bm
    V0 = int(len(bm.idx2term) if base_V is None else base_V)
    T0 = int(bm.T.shape[0] if base_T_rows is None else base_T_rows)
    if len(bm.idx2term) != V0:
        raise RuntimeError(f"基世代已变（base_V={V0} → {len(bm.idx2term)}）"
                           f" → 这批必须重新分段（不静默重映射）")
    if int(bm.T.shape[0]) != T0:
        raise RuntimeError(f"主表行数已变（{T0} → {int(bm.T.shape[0])}）"
                           f" → 有人就地追加过；先 `compact()` 再重来")
    new_pos = dict(new_pos or {})
    tag = str(tag)
    gen = str(generation or tag)
    M = len(new_terms)
    seg_dir = Path(z.dir) / SEG_SUBDIR / tag
    seg_dir.mkdir(parents=True, exist_ok=True)
    ri, ci, da = [], [], []
    for i, (_docid, toks, _f, _l, _x) in enumerate(rows):
        for t, c in Counter(toks).items():
            col = t2i.get(t) if t2i is not None else bm.term2idx.get(t)
            if col is None:
                col = new_pos[t]
            ri.append(i)
            ci.append(col)
            da.append(float(c))
    csc = sp.csc_matrix((np.asarray(da, np.float32),
                         (np.asarray(ri, np.int32), np.asarray(ci, np.int32))),
                        shape=(len(rows), V0 + M))
    np.save(seg_dir / f"{tag}.data.npy", np.asarray(csc.data, np.float32))
    np.save(seg_dir / f"{tag}.idx.npy", np.asarray(csc.indices, np.int32))
    np.save(seg_dir / f"{tag}.ptr.npy", np.asarray(csc.indptr, np.int32))
    dl = np.asarray([len(toks) for _d, toks, _f, _l, _x in rows], np.float32)
    np.save(seg_dir / "dl.npy", dl)
    # 乙档：文档身份侧车先写好（沿用「段文件 → 侧车 → manifest 提交 → 挂段」）
    row_base = int(bm.N)
    with open(seg_dir / SEG_DOCS, "w", encoding="utf-8", newline="\n") as _fh:
        for i, (_docid, _t, _f, _l, _x) in enumerate(rows):
            _d = str(_docid)
            _fh.write(json.dumps(dict(docid=_d, row=row_base + i,
                                      file=(_f or _d), line=int(_l or 1),
                                      text=_x), ensure_ascii=False) + "\n")
    rec = dict(event="publish", tag=tag, generation=gen,
               rows=len(rows), row_base=row_base, base_V=V0, m_new=M,
               new_terms=list(new_terms), created=time.time(),
               seg_dir=f"{SEG_SUBDIR}/{tag}", docs=SEG_DOCS)
    manifest_append(z.dir, rec)                       # 提交点（append-only）
    info = bm.mount_segment(str(seg_dir), tag, list(new_terms), dl,
                            rows=len(rows), generation=gen)
    # 乙档：立刻把段内文档补进产品层 `_meta`（否则本进程内 `search` 看不到它们）
    absorbed = z._absorb_staged([dict(tag=tag, mounted=True)])
    folded = None
    if fold:
        folded = bm.fold_segment(tag)
        z._persist()                                  # 折叠必须落盘才算持久（O(总 nnz)）
        manifest_append(z.dir, {"event": "folded_all", "tag": tag})
    return dict(tag=tag, rows=len(rows), V0=V0, V1=V0 + M,
                N=int(bm.N), V=int(bm.V), mounted=info, folded=folded,
                meta_added=int(absorbed.get("added", 0)))

