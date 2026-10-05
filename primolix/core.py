#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

from . import kernel
from .kernel import SparseBM25, zh_tokenize

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
    """遍历 root 下文件。``skip_dirs``: 绝对路径集合（跳过其下所有文件, 如索引目录）。"""
    skip_prefix = tuple(
        str(Path(d).resolve()).replace('\\', '/').rstrip('/').lower() + '/'
        for d in (skip_dirs or []))
    for p in sorted(root.rglob('*')):
        if not p.is_file():
            continue
        rel = p.relative_to(root)
        if any(part in EXCLUDE_DIRS for part in rel.parts):
            continue
        if str(p.resolve()).replace('\\', '/').lower().startswith(skip_prefix):
            continue
        if p.name in {'.env', '.env.local'} or p.suffix.lower() in EXCLUDE_EXT:
            continue
        try:
            if p.stat().st_size > MAX_FILE_BYTES:
                continue
        except OSError:
            continue
        yield p, rel


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


def scan_root(root: Path, skip_dirs=None):
    """返回 [(rel_path_str, hash, [chunks...])]; chunks=[(text, rel, line)]"""
    files = []
    for p, rel in iter_files(root, skip_dirs):
        try:
            text = p.read_text(encoding='utf-8', errors='ignore')
        except Exception:
            continue
        h = file_hash(p)
        rel_s = str(rel)
        ext = p.suffix.lower()
        chunks = chunk_code(text, rel_s) if ext in CODE_EXT else chunk_prose(text, rel_s)
        if chunks:
            files.append((rel_s, h, chunks))
    return files


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


def _topk_idx(x, k):
    r"""取前 k 大的下标 (分数降序; 并列按下标升序)。

    用 `argpartition` 而非 `argsort`: 只要前 k 个, 不必全排序。
    返回的分数值与原全量排序相同 (并列时取哪些行号本就不保证)。
    """
    n = int(x.size)
    k = int(min(max(k, 0), n))
    if k <= 0:
        return np.zeros(0, dtype=np.int64)
    if k >= n:
        idx = np.arange(n, dtype=np.int64)
    else:
        idx = np.argpartition(-x, k - 1)[:k].astype(np.int64, copy=False)
    return idx[np.lexsort((idx, -x[idx]))]


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
        self._hashes = None
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
        files = scan_root(Path(root), skip_dirs)
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
        self._persist()
        if dense:
            self._encode_dense()
        _say(f"建索引 {len(files)} 文件 → {len(chunks)} chunks, "
              f"{time.time() - t0:.0f}s ({self.info['size_mb']:.1f}MB)")
        return self

    # ---------- 持久化 ----------

    def _persist(self):
        """落盘 bm25.npz + meta.jsonl + hashes.json（build/update 共用）。

        `w_cache=True` 时额外把物化层 `w` 写进 `.mm/`。
        """
        self._bm.save(str(self.dir / 'bm25.npz'), w_cache=bool(getattr(self, "_w_cache", False)))
        with open(self.dir / 'meta.jsonl', 'w', encoding='utf-8') as f:
            for c in self._meta:
                f.write(json.dumps(c, ensure_ascii=False) + '\n')
        with open(self.dir / 'hashes.json', 'w', encoding='utf-8') as f:
            json.dump(self._hashes, f, ensure_ascii=False, indent=1)

    def _load(self):
        # 索引目录含 dense.npy → 视为 dense 索引 (update/reload 保持 dense 一致)
        self._dense = self._dense or (self.dir / 'dense.npy').exists()
        self._bm = SparseBM25.load(str(self.dir / 'bm25.npz'))
        self._meta = [json.loads(l) for l in open(self.dir / 'meta.jsonl', encoding='utf-8')]
        self._rows = None          # 失效行号缓存
        # 建库分词档随索引落盘（`kernel.stream_mode`），重载后重建仍用它；
        #    缺字段 = 旧版索引 → 按默认档。
        self._stream = _norm_stream(getattr(self._bm, "stream_mode", "off"))
        hf = self.dir / 'hashes.json'
        self._hashes = json.load(open(hf, encoding='utf-8')) if hf.exists() else {}
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
        except Exception as _e:                                      # noqa: BLE001
            _say(f"[警告] 旁车重挂失败（{_e!r}）→ 该索引按 npz 状态可用，但已发布的段未挂上")

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
        return self._rows

    def _score_active(self, qtoks):
        """返回与 self._meta 对齐的 BM25 分数数组（按每条 row 取物理行分）。"""
        scores = self._bm.score_all(qtoks)
        rows = self._active_rows()
        if rows.size == 0:
            return np.zeros(0, dtype=np.float32)
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
            out.append(dict(rank=rank, score=float(s[i]), file=m['file'],
                            line=m['line'], text=m['text']))
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
        return [dict(file=self._meta[i]['file'], line=self._meta[i]['line'],
                     text=self._meta[i]['text'])
                for i in idxs[start:start + n]], len(idxs)

    def list_files(self):
        if self._meta is None:
            self._load()
        return sorted({m['file'] for m in self._meta})

    # ---------- 真增量 update（方案 A） ----------

    def update(self, root, workers=0):
        """hash diff + 方案 A: 词表内走真增量, 词表外整库回退重建。

        返回 {'kind': 'noop'|'incremental'|'rebuild', 'detail': str}。
        """
        if self._bm is None:
            self._load()
        self._dense = self._dense or (self.dir / 'dense.npy').exists()
        t0 = time.time()
        # 排除索引输出目录自身 (可能在 root 内部, 如 <root>/.idx)
        files = scan_root(Path(root), [self.dir])
        new = {rel: h for rel, h, _ in files}
        old = self._hashes
        added = [r for r in new if r not in old]
        removed = [r for r in old if r not in new]
        changed = [r for r in new if r in old and new[r] != old[r]]
        unchanged = len(new) - len(added) - len(changed)
        _say(f"S4 hash diff: 新增 {len(added)} / 删除 {len(removed)} / "
              f"变更 {len(changed)} / 跳过 {unchanged} ({unchanged / max(len(new), 1):.0%})")
        if not (added or removed or changed):
            return {'kind': 'noop', 'detail': '库已同步, 无需重建'}

        # ---- 收集新 chunk 文本 (added 全文件 + changed 文件的新内容) ----
        new_chunk_texts = []
        rel2scan = {x[0]: x for x in files}
        for rel in added + changed:
            f = rel2scan.get(rel)
            if f is not None:
                for t, _, _ in f[2]:
                    new_chunk_texts.append(t)

        # ---- OOV 预检: 词表外词 → 整库回退重建 (拒绝静默丢词) ----
        if new_chunk_texts:
            oov = self._bm.oov_terms([zh_tokenize(t) for t in new_chunk_texts])
            if oov:
                n_oov = len(oov)
                sample = sorted(oov)[:5]
                _say(f"[方案A] 检测到 {n_oov} 个词表外词 {sample}… "
                      f"(词表封闭, 真增量会静默丢词) → 整库回退重建")
                self.rebuild_in_place(root, workers=workers)
                _say(f"S4 回退重建完成 {time.time() - t0:.0f}s")
                return {'kind': 'rebuild', 'detail': f'oov={n_oov}'}

        # ---- 词表内安全: 走真增量 ----
        # 需要墓碑的行 = removed 文件旧 chunk 行 + changed 文件旧 chunk 行
        drop_rels = set(removed) | set(changed)
        tomb_rows = [c['row'] for c in self._meta if c['file'] in drop_rels]
        # 剩余活跃 chunk (保留): 整文件级替换 → removed/changed 的旧 chunk 全剔除
        keep_meta = [c for c in self._meta if c['file'] not in drop_rels]
        # 追加新 chunk (added 全文件 + changed 的新内容, 均来自本次 scan)
        append_chunks = []
        for rel in added + changed:
            f = rel2scan.get(rel)
            if f is None:
                continue
            h = f[1]
            for t, _, ln in f[2]:
                append_chunks.append(dict(docid=f'{h[:8]}:{len(append_chunks)}',
                                          file=rel, line=ln, text=t))

        if tomb_rows:
            self._bm._tombstone_rows(tomb_rows)
        new_rows = self._bm.append_docs([zh_tokenize(c['text']) for c in append_chunks])
        for c, row in zip(append_chunks, new_rows):
            c['row'] = int(row)
        self._meta = keep_meta + append_chunks
        self._rows = None          # 失效行号缓存
        self._hashes = new
        self._persist()
        # dense: update 后整库重对齐 (rows 迁移, 旧向量错位; 小语料可负担)
        if self._dense:
            self._encode_dense()
        _say(f"S4 真增量完成 {time.time() - t0:.0f}s "
              f"(墓碑 {len(tomb_rows)} 行, 追加 {len(append_chunks)} chunk, "
              f"活跃 {len(self._meta)})")
        return {'kind': 'incremental',
                'detail': f'tomb={len(tomb_rows)} append={len(append_chunks)}'}

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
        self._rows = None          # 失效行号缓存
        self._hashes = z._hashes
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
