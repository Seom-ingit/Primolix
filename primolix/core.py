#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ⚠️ Python 3.9 兼容：本文件有 `dict | None` 这类 PEP 604 注解。
#    没有这行 future import 时，3.9 会在**定义函数时**求值注解并抛 TypeError
#    （`unsupported operand type(s) for |`），导致整个包 import 失败。CI 的 3.9 job 就是这么红的。
from __future__ import annotations

import ast
from bisect import bisect_right
import hashlib
import json
import os
import re
import struct
import sys
import threading
import time
from pathlib import Path

import numpy as np

from . import kernel
from .kernel import SparseBM25, get_tokenizer, zh_tokenize
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
CODE_OVERLAP_LINES = 8
# M1（契约 §1.1）：切块规则代次 —— 切块规则变了就 bump 此串；旧 rerank 缓存视图会因代次不符作废
CHUNK_VERSION = "ast_v2:block=80:overlap=8:prose=heading_merge"
PROSE_HEADING_COALESCE = 700
MAX_FILE_BYTES = 2_000_000
_MD_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+\S")

# 模型缓存: 进程内只加载一次 dense 模型
_MODEL_CACHE = {}

# ⭐ 01m（P-D）：`_topk_idx` 的取负缓冲 —— **线程私有**（共享会在并发查询下互相污染）
_NEG_BUF = threading.local()

# ⭐ P-O3（2026-10-19）：选择段「采样定阈」近似路径的开关与参数。
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


def get_dense_model(model_path='', device='auto'):
    """加载（或复用）dense/rerank 编码模型。

    0.3 起委托 `rerank.get_rerank_model`：优先查本地 bge-small-zh-v1.5，其次 bge-m3；
    未安装或无缓存时如实返回 None（降级 BM25-only，绝不抛 FileNotFoundError）。
    """
    from .rerank import get_rerank_model

    model, st = get_rerank_model(model_path, device=device)
    if model is None and st.get("reason"):
        _say(f"[提示] {st['reason']}")
    return model


# ---------------- 文件扫描与分块（纯 stdlib） ----------------

def iter_files(root: Path, skip_dirs=None):
    """遍历 root 下文件（**os.scandir 递归**：用 dirent 自带的类型/大小，避免每文件 `resolve` ＋ 重复 `stat`）。

    · `skip_dirs`：绝对路径集合，其下所有文件跳过（如索引目录自身）。按**相对前缀**跳过，
      ⛔ 不再对每个文件调 `resolve()` —— 那是原实现里最贵的一笔（每文件一次路径解析）。
    · 返回 `[(Path, rel_path)]`，按 `rel` 字符串排序（与原实现同序，保住建库/切段的确定性）。
    · `rel` 用**平台原生分隔符**（与原实现一致）—— 哈希字典的键靠它，⛔ 别改成 `/`。
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
                            if (name in EXCLUDE_DIRS or rel_posix in skip_rel
                                    or name.endswith((".npz.mm", ".npz.vocab"))
                                    or os.path.exists(os.path.join(e.path, "bm25.npz"))):
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


def chunk_prose(text, rel, *, with_spans=False):
    """段落级切分（检索甜区 200-800 字）：保留短 Markdown 标题并与所属正文连续拼接。

    `with_spans=True` 时返回 `(chunk_text, rel, line_no, char_start, char_end)`，
    且保证 `text[char_start:char_end] == chunk_text` 逐字符成立。
    """
    raw_paras = []  # (cs, ce, start_line, is_heading)
    buf, start = [], 1
    line_no = 1
    pos = 0
    para_char = 0

    def _push_para(lines_buf, st_ln, st_pos):
        raw_p = ''.join(lines_buf)
        lead = len(raw_p) - len(raw_p.lstrip())
        tail = len(raw_p) - len(raw_p.rstrip())
        cs = st_pos + lead
        ce = st_pos + len(raw_p) - tail
        if ce <= cs:
            return
        stripped = text[cs:ce]
        first_ln = stripped.splitlines()[0] if stripped else ""
        is_h = (stripped.count('\n') <= 1) and bool(_MD_HEADING_RE.match(first_ln))
        raw_paras.append((cs, ce, st_ln, is_h))

    for line in text.splitlines(keepends=True):
        nxt = pos + len(line)
        if line.strip() == '':
            if buf:
                _push_para(buf, start, para_char)
                buf, start = [], line_no + 1
        else:
            if not buf:
                start = line_no
                para_char = pos
            buf.append(line)
        pos = nxt
        line_no += 1
    if buf:
        _push_para(buf, start, para_char)

    # 连续归并：不丢短 `#` 标题，并在标题节区内把短碎段落合并至 ~700 字甜区（不跨新标题）
    groups = []  # (cs, ce, start_line)
    has_headings = any(is_h for _, _, _, is_h in raw_paras)
    i = 0
    while i < len(raw_paras):
        cs0, ce0, ln0, is_h0 = raw_paras[i]
        cur_cs, cur_ce, cur_ln = cs0, ce0, ln0
        in_headed_section = is_h0
        seen_body = not is_h0
        i += 1
        while i < len(raw_paras):
            cs_n, ce_n, _ln_n, is_hn = raw_paras[i]
            if is_hn:
                # 已进入正文后遇到新标题必须断块；仅“连续短标题（如主标题接副标题）”允许顺延
                if seen_body or not in_headed_section or (ce_n - cur_cs) > 220:
                    break
            elif not in_headed_section and not (has_headings and (cur_ce - cur_cs) < 80):
                break
            limit = PROSE_HEADING_COALESCE if in_headed_section else MAX_CHUNK
            if seen_body and (ce_n - cur_cs) > limit and (cur_ce - cur_cs) >= 20:
                break
            if (ce_n - cur_cs) > MAX_CHUNK:
                break
            cur_ce = ce_n
            if not is_hn:
                seen_body = True
                in_headed_section = has_headings
            i += 1
        groups.append((cur_cs, cur_ce, cur_ln))

    out = []
    for cs, ce, ln in groups:
        t = text[cs:ce]
        if len(t) < 20:
            continue
        while len(t) > MAX_CHUNK:
            sl = t[:MAX_CHUNK]
            if with_spans:
                out.append((sl, rel, ln, cs, cs + len(sl)))
            else:
                out.append((sl, rel, ln))
            cs += MAX_CHUNK
            t = t[MAX_CHUNK:]
        if with_spans:
            out.append((t, rel, ln, cs, cs + len(t)))
        else:
            out.append((t, rel, ln))
    return out


def _py_ast_segments(text: str, n_lines: int) -> list[tuple[int, int]] | None:
    """用 stdlib ast 把 Python 源码切成贴合 class/def 的连续行区间 `[lo, hi)`（0-based）。"""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return None
    spans: list[tuple[int, int]] = []
    for node in tree.body:
        lo = max(0, int(node.lineno) - 1)
        hi = min(n_lines, int(getattr(node, "end_lineno", node.lineno)))
        if isinstance(node, ast.ClassDef) and (hi - lo) > CODE_BLOCK_LINES and node.body:
            methods = [m for m in node.body if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))]
            if methods:
                m0_lo = max(lo, int(methods[0].lineno) - 1)
                if m0_lo > lo:
                    spans.append((lo, m0_lo))
                cur = m0_lo
                for m in methods:
                    mlo = max(cur, int(m.lineno) - 1)
                    mhi = min(hi, int(getattr(m, "end_lineno", m.lineno)))
                    if mlo > cur:
                        spans.append((cur, mlo))
                    spans.append((mlo, max(mlo + 1, mhi)))
                    cur = max(cur, mhi)
                if cur < hi:
                    spans.append((cur, hi))
                continue
        spans.append((lo, max(lo + 1, hi)))

    if not spans:
        return None
    full: list[tuple[int, int]] = []
    pos = 0
    for lo, hi in sorted(spans):
        if lo > pos:
            full.append((pos, lo))
        if hi > pos:
            full.append((max(pos, lo), hi))
            pos = hi
    if pos < n_lines:
        full.append((pos, n_lines))
    return full


def chunk_code(text, rel, *, with_spans=False):
    raw_lines = text.splitlines(keepends=True)
    if not raw_lines:
        return []
    lines = [ln.rstrip('\r\n') for ln in raw_lines]
    offs = []
    cur = 0
    for ln in raw_lines:
        offs.append(cur)
        cur += len(ln)

    out = []
    seen_windows: set[tuple[int, int]] = set()

    def _emit_win(lo: int, hi: int) -> None:
        while lo < hi and not lines[lo].strip():
            lo += 1
        if hi <= lo or (lo, hi) in seen_windows:
            return
        seen_windows.add((lo, hi))
        blk = '\n'.join(lines[lo:hi]).strip()
        if len(blk) < 20:
            return
        if with_spans:
            raw_blk = ''.join(raw_lines[lo:hi])
            lead = len(raw_blk) - len(raw_blk.lstrip())
            cs = offs[lo] + lead
            ce = offs[lo] + len(raw_blk.rstrip())
            out.append((blk, rel, lo + 1, cs, ce))
        else:
            out.append((blk, rel, lo + 1))

    n_lines = len(lines)
    stride = max(1, CODE_BLOCK_LINES - CODE_OVERLAP_LINES)
    segs = _py_ast_segments(text, n_lines) if str(rel).lower().endswith(".py") else None
    if not segs:
        if n_lines <= CODE_BLOCK_LINES:
            _emit_win(0, n_lines)
        else:
            for i in range(0, n_lines, stride):
                _emit_win(i, min(n_lines, i + CODE_BLOCK_LINES))
                if i + CODE_BLOCK_LINES >= n_lines:
                    break
        return out

    # 按 AST 边界贪心打包：小语句尽量合并到 ≤ CODE_BLOCK_LINES；超大函数内部走滑窗带重叠
    pack_lo, pack_hi = segs[0]
    for slo, shi in segs[1:]:
        if (shi - pack_lo) <= CODE_BLOCK_LINES:
            pack_hi = shi
        else:
            if (pack_hi - pack_lo) <= CODE_BLOCK_LINES:
                _emit_win(pack_lo, pack_hi)
            else:
                for i in range(pack_lo, pack_hi, stride):
                    _emit_win(i, min(pack_hi, i + CODE_BLOCK_LINES))
                    if i + CODE_BLOCK_LINES >= pack_hi:
                        break
            pack_lo, pack_hi = slo, shi
    if (pack_hi - pack_lo) <= CODE_BLOCK_LINES:
        _emit_win(pack_lo, pack_hi)
    else:
        for i in range(pack_lo, pack_hi, stride):
            _emit_win(i, min(pack_hi, i + CODE_BLOCK_LINES))
            if i + CODE_BLOCK_LINES >= pack_hi:
                break
    return out


_FENCE_RE = re.compile(r"^\s{0,3}(```+|~~~+)")
_CODE_DEF_RE = re.compile(
    r"^(\s*)(?:export\s+(?:default\s+)?)?(?:pub\s+)?(?:async\s+)?"
    r"(?:class|struct|interface|impl|trait|enum|fn|func|def|function)\s+[A-Za-z_][A-Za-z0-9_<>:]*"
)


def _prose_chunk_contexts(text: str, spans: list[tuple[int, int]]) -> list[str]:
    """为正文章节后续块提取祖先 `# / ##` 标题面包屑（跳过代码块内 `#` 注释；不修改原文）。"""
    if not spans or "#" not in text:
        return [""] * len(spans)
    headings: list[tuple[int, int, str]] = []  # (char_pos, level, clean_heading)
    in_fence = False
    fence_char = ""
    pos = 0
    for line in text.splitlines(keepends=True):
        s_line = line.lstrip()
        fm = _FENCE_RE.match(s_line)
        if fm:
            marker = fm.group(1)[0]
            if not in_fence:
                in_fence, fence_char = True, marker
            elif marker == fence_char:
                in_fence = False
        elif not in_fence and _MD_HEADING_RE.match(line):
            stripped = line.strip()
            lvl = len(stripped) - len(stripped.lstrip("#"))
            title = stripped.rstrip("#").strip()[:90]
            if title:
                headings.append((pos, lvl, title))
        pos += len(line)

    if not headings:
        return [""] * len(spans)

    out: list[str] = []
    stack: list[tuple[int, int, str]] = []  # (level, char_pos, heading)
    hi = 0
    nh = len(headings)
    for cs, ce in spans:
        while hi < nh and headings[hi][0] < cs:
            h_pos, lvl, h_txt = headings[hi]
            while stack and stack[-1][0] >= lvl:
                stack.pop()
            stack.append((lvl, h_pos, h_txt))
            hi += 1
        if not stack:
            out.append("")
            continue
        head_win = text[cs:min(ce, cs + 200)]
        crumbs = [h_txt for _lvl, _hp, h_txt in stack[-2:] if h_txt not in head_win]
        out.append(" > ".join(crumbs) if crumbs else "")
    return out


def _code_chunk_contexts(text: str, rel: str, spans: list[tuple[int, int]]) -> list[str]:
    """为代码续块提取外层 `class / def` 签名上下文（不修改原文切片）。"""
    if not spans:
        return []
    raw_lines = text.splitlines(keepends=True)
    if not raw_lines:
        return [""] * len(spans)
    offs: list[int] = []
    cur = 0
    for ln in raw_lines:
        offs.append(cur)
        cur += len(ln)
    n_lines = len(raw_lines)
    clean_lines = [ln.rstrip("\r\n") for ln in raw_lines]

    scopes: list[tuple[int, int, str]] = []  # (lo_line, hi_line, sig) 0-based inclusive
    if str(rel).lower().endswith(".py"):
        try:
            tree = ast.parse(text)

            def _walk(node) -> None:
                for child in ast.iter_child_nodes(node):
                    if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                        lo = max(0, int(child.lineno) - 1)
                        hi = min(n_lines - 1, int(getattr(child, "end_lineno", child.lineno)) - 1)
                        sig = clean_lines[lo].strip()[:90]
                        if sig:
                            scopes.append((lo, hi, sig))
                    _walk(child)

            _walk(tree)
        except SyntaxError:
            scopes = []

    if not scopes:
        # 轻量正则兜底（JS/TS/Go/Rust/C++ 等或残缺 Python）
        for idx_ln, ln in enumerate(clean_lines):
            if len(ln) - len(ln.lstrip()) <= 4 and _CODE_DEF_RE.match(ln):
                sig = ln.strip().rstrip("{").strip()[:90]
                if sig:
                    scopes.append((idx_ln, n_lines - 1, sig))

    if not scopes:
        return [""] * len(spans)

    scopes.sort(key=lambda x: (x[0], -x[1]))
    is_py = str(rel).lower().endswith(".py")
    out: list[str] = []
    for cs, _ce in spans:
        lo_line = max(0, bisect_right(offs, cs) - 1)
        if is_py:
            anc = [sig for s_lo, s_hi, sig in scopes if s_lo < lo_line <= s_hi][-2:]
        else:
            # 正则无精确 end_lineno，只取最近一条外层声明且非本行起头
            prev = [sig for s_lo, _s_hi, sig in scopes if s_lo < lo_line]
            anc = prev[-1:] if prev else []
        out.append(" > ".join(anc) if anc else "")
    return out


def extract_chunk_contexts(text: str, rel: str, spans: list[tuple[int, int]], *, is_code: bool | None = None) -> list[str]:
    """给定原文与切块 `(char_start, char_end)` 列表，返回每块的上下文面包屑（仅作用于索引分词）。"""
    if is_code is None:
        is_code = Path(str(rel)).suffix.lower() in CODE_EXT
    return _code_chunk_contexts(text, str(rel), spans) if is_code else _prose_chunk_contexts(text, spans)


def scan_root_ex(root: Path, skip_dirs=None, known=None, stamps=None, verify="full",
                 *, with_spans=False, with_ctx=False):
    """扫描语料目录。返回 `(files, stamps_now)`。

    `files`   = `[(rel, md5, chunks|None)]`（`known` 命中 ⇒ `chunks=None`）
    `stamps`  = `{rel: (size, mtime_ns)}`，只含进入了 `files` 的文件

    `verify`：
        `"full"`（默认）—— 每个文件读内容算 md5；
        `"stat"`        —— 先比 `(size, mtime_ns)`，**元数据没变就不读内容**（实测 ~1/10 成本）；
                           元数据变了才读，内容 hash 没变时仍算"未变"（只刷新 stamp）。

    ⚠️ `"stat"` 的前提是写入方会更新 mtime；保留 mtime 的还原类工具会漏报 ⇒ 用 `"full"` 兜底。
    ⚠️ 缺 `known`/`stamps` 时 `"stat"` 如实退回 `"full"`（不假装快扫）。
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
        is_c = ext in CODE_EXT
        chunks = (chunk_code(text, rel_s, with_spans=with_spans)
                  if is_c else chunk_prose(text, rel_s, with_spans=with_spans))
        if chunks:
            if with_ctx and with_spans:
                ctxs = extract_chunk_contexts(
                    text, rel_s, [(c[3], c[4]) for c in chunks], is_code=is_c,
                )
                chunks = [(*c, ctx) for c, ctx in zip(chunks, ctxs)]
            files.append((rel_s, h, chunks))
            stamps_now[rel_s] = stamp
    return files, stamps_now


def scan_root(root: Path, skip_dirs=None, known=None):
    """返回 [(rel_path_str, hash, chunks)]；chunks=[(text, rel, line)]。

    `known`（`{rel: md5}`，只有 `update()` 会传）：命中 ⇒ 该文件**不 decode、不切段**，第三项为 `None`
    （调用方必须跳过 `None`；`build` 不传 `known`，所以永远拿到真 chunks）。
    ⭐ 每个文件只 **读一次字节**：同一次读既算 md5 又（必要时）decode。
    ⭐ 需要 `stat` 快扫或拿 `(size, mtime_ns)` ⇒ 用 `scan_root_ex()`。
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

    ⚠️ 调用前必须已保证 `0 < k < x.size`。返回 k 个下标（分数降序、并列按下标升序）。
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

    返回 `None` 表示"本形态不适合近似"（交回精确路径，**退化有界**）：
      · 非浮点 / 非正分布（全零、全负）—— `_topk_idx` 的通用契约；
      · 正分文档数 < k（命中文档极少的查询）。
    ⚠️ 这是**概率性**保证：阈值 = 采样第 k 大 × `TOPK_SHRINK`，理论上可能高于真第 k 大
       ⇒ 漏掉边界文档。经验面见 `_raw/topk_equiv_2026-10-19.json`（1M · 200 查询 ·
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

    ⭐ **P-O3（2026-10-19）**：默认走「采样定阈」（`TOPK_APPROX=True`）——
    选择段 @1M 由 **4.93 ms → 0.57 ms**（配对 0.138，600/600 全胜）。做法：
    1/32 抽样估第 k 大 × 0.98 当阈值 → `flatnonzero` 取候选 → 只对候选 `argpartition`。
    ⛔ **不改返回契约**：仍是"k 个下标、分数降序、值等于全量排序的前 k 个"。

    非正分布 / 正分不足 k / 非浮点 ⇒ 自动回精确路径（`TOPK_APPROX=False` 亦然）。
    ⚠️ 近似性是**概率**保证而非证明；判据与读数见 `_raw/topk_equiv_2026-10-19.json`。
    ⚠️ κ 很大（如 dense 分支的 `k*10`）时采样点相对 k 变少 ⇒ 收益收窄、退化档变多。

    ⭐ **01m（P-D）**：精确路径的取负那一步写进**线程私有常驻缓冲**，免掉每次调用分配的
    N 长临时（@1M ≈4 MB/查询、@8.84M ≈35 MB/查询）。⚠️ **必须线程私有**：共享缓冲在
    两个并发查询下会互相污染（本层的锁只保护索引结构，不保护这条快路径）。
    """
    n = int(x.size)
    k = int(min(max(k, 0), n))
    if k <= 0:
        return np.zeros(0, dtype=np.int64)
    if k >= n:
        # ⚠️ 2026-11-08 修：这里原来直接 `return arange(n)`（升序），而本函数的契约是
        #    **分数降序**。`query()` 的循环以 `s[i] <= 0: break` 截断 ⇒ 升序时一旦第一个是
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


class _MetaEntry(dict):
    """`_meta` 的单条：docid/row/file/line 是真键；`text` 惰性（首次访问才从盘上取）。

    ⭐ X13（2026-11-18）：[实测] text 占 meta 记录字节的 ~78%，而查询期只有最终 k 条命中
    才需要它 —— 常驻它 = 白付。`__missing__` 只在 `m['text']` 触发一次并缓存回本条；
    `get`/`in` 一并覆盖（包内唯一 `.get("text")` 消费点读的是段侧车，不是这里 —— 审计过）。
    """

    __slots__ = ('_seq', '_i')

    def __init__(self, seq, i, base):
        super().__init__(base)
        self._seq = seq
        self._i = i

    def __missing__(self, key):
        if key == 'text':
            t = self._seq._text_at(self._i)
            self['text'] = t
            return t
        raise KeyError(key)

    def get(self, key, default=None):
        if key == 'text' and not dict.__contains__(self, 'text'):
            return self.__missing__('text')
        return dict.get(self, key, default)

    def __contains__(self, key):
        if key == 'text':
            return True                      # 惰性语义：每条必有 text（取的时候才付）
        return dict.__contains__(self, key)


class _MetaSeq:
    r"""`_meta` 的数组化载体（X13）：docid/row/file/line 常驻，`text` 惰性。

    为什么：[实测] meta 加载的瓶颈是**构造 N 个 dict**（二进制 v2 只省 9.5% 且不提速，
    已回退 —— 见 `06_meta格式 §2.5`）；常驻里 text 占 ~78%。本类两件一起解决：
      · 加载只建数组与紧凑列表（row/line 进 numpy；docid/file 是本来就常驻的身份串）；
      · `__getitem__` 才按需构造 `_MetaEntry`（查询期只有 k 条命中需要）；
      · `text` 按行偏移从 `meta.jsonl` 本身取 —— **零格式改动**，不碰追加式协议
        （追加不动旧行 ⇒ 旧偏移恒有效；整份重写后由 `rebind_from_file` 重算）。
    兼容面：`len/迭代/下标/append` 与旧 list-of-dict 等价；消费方 71 处已逐一核对，
    只读 row/file 的全表扫全部改走数组直达（`rows` / `files`），点取走 `__getitem__`。
    """

    __slots__ = ('rows', 'lines', 'char_starts', 'char_ends', 'docids', 'files', 'texts',
                 '_toff', '_tpath', '_tfh')

    _CACHE_MAGIC = b"PMXMC1\x00\x00"
    _CACHE_HDR = struct.Struct("<8sqqqqqQQ")

    @staticmethod
    def _cache_path(path) -> Path:
        p = Path(path)
        return p.parent / (".meta.cache" if p.name == "meta.jsonl" else f".{p.name}.cache")

    @classmethod
    def invalidate_cache(cls, path) -> None:
        """删除 `meta.jsonl` 对应的二进制旁路缓存（幂等）。"""
        try:
            cls._cache_path(path).unlink()
        except OSError:
            pass

    @classmethod
    def _load_cache(cls, path, st):
        """校验 `(mtime_ns, size)` 一致时直读 `.meta.cache`；任何不一致/损坏返回 None。"""
        cp = cls._cache_path(path)
        try:
            data = cp.read_bytes()
        except OSError:
            return None
        hdr_sz = cls._CACHE_HDR.size
        if len(data) < hdr_sz:
            return None
        try:
            magic, mtime_ns, fsize, n, max_row, ndel, d_len, f_len = cls._CACHE_HDR.unpack_from(data, 0)
            if magic != cls._CACHE_MAGIC or mtime_ns != st.st_mtime_ns or fsize != st.st_size or n < 0:
                return None
            if len(data) != hdr_sz + n * 36 + d_len + f_len:
                return None
            off = hdr_sz
            s = cls.__new__(cls)
            s.rows = np.frombuffer(data, dtype=np.int64, count=n, offset=off).copy()
            off += n * 8
            s.lines = np.frombuffer(data, dtype=np.int32, count=n, offset=off).copy()
            off += n * 4
            s.char_starts = np.frombuffer(data, dtype=np.int32, count=n, offset=off).copy()
            off += n * 4
            s.char_ends = np.frombuffer(data, dtype=np.int32, count=n, offset=off).copy()
            off += n * 4
            s._toff = np.frombuffer(data, dtype=np.int64, count=n * 2, offset=off).reshape(n, 2).copy()
            off += n * 16
            docids_list = json.loads(data[off:off + d_len].decode('utf-8'))
            off += d_len
            files_list = json.loads(data[off:off + f_len].decode('utf-8'))
            if (not isinstance(docids_list, list) or not isinstance(files_list, list)
                    or len(docids_list) != n or len(files_list) != n):
                return None
            s.docids = np.asarray(docids_list, dtype=object) if n else np.zeros(0, dtype=object)
            s.files = np.asarray(files_list, dtype=object) if n else np.zeros(0, dtype=object)
            s.texts = np.full(n, None, dtype=object)
            s._tpath = Path(path)
            s._tfh = None
            return s, int(max_row), int(ndel)
        except Exception:                                    # noqa: BLE001
            return None

    def write_cache(self, path, max_row: int, ndel: int, st=None) -> None:
        """原子写 `.meta.cache`（只含结构配列与行偏移，text 保持惰性；失败静默忽略）。"""
        if self._toff is None:
            return
        path = Path(path)
        cp = self._cache_path(path)
        tmp = cp.with_name(f"{cp.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        try:
            if st is None:
                st = path.stat()
            n = int(self.rows.size)
            docids_b = json.dumps(self.docids.tolist(), ensure_ascii=False).encode('utf-8')
            files_b = json.dumps(self.files.tolist(), ensure_ascii=False).encode('utf-8')
            hdr = self._CACHE_HDR.pack(
                self._CACHE_MAGIC, int(st.st_mtime_ns), int(st.st_size),
                n, int(max_row), int(ndel), len(docids_b), len(files_b),
            )
            with open(tmp, 'wb') as f:
                f.write(hdr)
                f.write(np.ascontiguousarray(self.rows, dtype=np.int64).tobytes())
                f.write(np.ascontiguousarray(self.lines, dtype=np.int32).tobytes())
                f.write(np.ascontiguousarray(self.char_starts, dtype=np.int32).tobytes())
                f.write(np.ascontiguousarray(self.char_ends, dtype=np.int32).tobytes())
                f.write(np.ascontiguousarray(self._toff, dtype=np.int64).tobytes())
                f.write(docids_b)
                f.write(files_b)
            os.replace(tmp, cp)
        except OSError:
            try:
                tmp.unlink()
            except OSError:
                pass

    # ---------------- 构造 ----------------
    @classmethod
    def from_dicts(cls, dicts):
        """从已物化的 dict 列表建（build 路径）：text 直接常驻（写盘后可 `rebind` 转惰性）。"""
        s = cls.__new__(cls)
        n = len(dicts)
        s.rows = np.fromiter((int(d.get('row', -1)) for d in dicts),
                             dtype=np.int64, count=n)
        s.lines = np.fromiter((int(d.get('line', 0)) for d in dicts),
                              dtype=np.int32, count=n)
        s.char_starts = np.fromiter((int(d.get('char_start', -1)) for d in dicts),
                                    dtype=np.int32, count=n)
        s.char_ends = np.fromiter((int(d.get('char_end', -1)) for d in dicts),
                                  dtype=np.int32, count=n)
        # ⭐ X14：docids/files/texts 一律 **object 数组** —— `_take` 的花式索引是 C 级拷贝
        #    （@1M ≈20 ms），Python 列表逐元素重建是 0.25 s/趟（[实测] cProfile 抓到的主犯）
        s.docids = np.asarray([d.get('docid') for d in dicts], dtype=object)
        s.files = np.asarray([d.get('file') for d in dicts], dtype=object)
        s.texts = np.asarray([d.get('text') or '' for d in dicts], dtype=object)
        s._toff = None
        s._tpath = None
        s._tfh = None
        return s

    @classmethod
    def from_jsonl(cls, path, *, use_cache: bool = True):
        """读 `meta.jsonl` → `(seq, 文件里最大 row, 删除记录数)`。

        语义与旧 `_read_meta` 逐条对齐：按 row 建表（**后写覆盖先写**）、`del` 剔除、
        按 row 升序；`最大 row` 计**所有**解析过的行（含被删的 —— 游标语义）。
        ⚠️ 一遍二进制扫描同时拿字段与行偏移；text 字符串**不驻留**（解析后即弃）。
        0.4 P0: `use_cache=True` 且 `.meta.cache` 的 `(mtime_ns, size)` 匹配时直读紧凑缓存。
        """
        path = Path(path)
        if use_cache:
            try:
                st0 = path.stat()
                cached = cls._load_cache(path, st0)
                if cached is not None:
                    return cached
            except OSError:
                pass
        rows, lines_, cstarts, cends, docids, files = [], [], [], [], [], []
        starts, lens = [], []
        row_to_pos, dels = {}, set()
        max_row = -1
        pos = 0
        with open(path, 'rb') as f:
            st_read = os.fstat(f.fileno())
            for raw in f:
                _start, pos = pos, pos + len(raw)
                ln = raw.strip()
                if not ln:
                    continue
                rec = json.loads(ln.decode('utf-8'))
                if 'del' in rec:
                    for r in rec['del']:
                        dels.add(int(r))
                    continue
                r = int(rec.get('row', -1))
                if r > max_row:
                    max_row = r
                rows.append(r)
                lines_.append(int(rec.get('line', 0)))
                cstarts.append(int(rec.get('char_start', -1)))
                cends.append(int(rec.get('char_end', -1)))
                docids.append(rec.get('docid'))
                files.append(rec.get('file'))
                starts.append(_start)
                lens.append(len(raw))
                row_to_pos[r] = len(rows) - 1          # 后写覆盖先写
        order = sorted((r, p) for r, p in row_to_pos.items() if r not in dels)
        sel = np.fromiter((p for _, p in order), dtype=np.int64,
                          count=len(order)) if order else np.zeros(0, dtype=np.int64)
        s = cls.__new__(cls)
        s.rows = np.asarray(rows, dtype=np.int64)[sel] if len(rows) else np.zeros(0, np.int64)
        s.lines = np.asarray(lines_, dtype=np.int32)[sel] if len(lines_) else np.zeros(0, np.int32)
        s.char_starts = np.asarray(cstarts, dtype=np.int32)[sel] if len(cstarts) else np.zeros(0, np.int32)
        s.char_ends = np.asarray(cends, dtype=np.int32)[sel] if len(cends) else np.zeros(0, np.int32)
        s.docids = np.asarray(docids, dtype=object)[sel] if rows else np.zeros(0, dtype=object)
        s.files = np.asarray(files, dtype=object)[sel] if rows else np.zeros(0, dtype=object)
        s.texts = np.full(len(order), None, dtype=object)     # ⭐ X14：object 数组（惰性占位）
        s._toff = (np.stack([np.asarray(starts, dtype=np.int64)[sel],
                             np.asarray(lens, dtype=np.int64)[sel]], axis=1)
                   if order else np.zeros((0, 2), dtype=np.int64))
        s._tpath = path
        s._tfh = None
        if use_cache and st_read.st_size > 0:
            s.write_cache(path, max_row, len(dels), st=st_read)
        return s, max_row, len(dels)

    # ---------------- 读面 ----------------
    def __len__(self):
        return int(self.rows.size)

    def __bool__(self):                        # `(self._meta or [])` 这类写法要它
        return self.rows.size > 0

    def __getitem__(self, i):
        i = int(i)
        base = dict(docid=self.docids[i], row=int(self.rows[i]),
                    file=self.files[i], line=int(self.lines[i]))
        if int(self.char_starts[i]) >= 0:
            base['char_start'] = int(self.char_starts[i])
            base['char_end'] = int(self.char_ends[i])
        if self.texts[i] is not None:
            base['text'] = self.texts[i]
        return _MetaEntry(self, i, base)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def _text_at(self, i):
        t = self.texts[i]
        if t is not None:
            return t
        if self._toff is None:
            raise KeyError("text 无来源：载体既无常驻 text 也无文件偏移（from_dicts 后未 rebind？）")
        s, n = int(self._toff[i, 0]), int(self._toff[i, 1])
        if s < 0:
            raise KeyError("该条目无文件偏移且无常驻 text —— 结构性不一致（垫行被走到 = bug）")
        if self._tfh is None or getattr(self._tfh, "closed", False):
            self._tfh = open(self._tpath, 'rb')
        self._tfh.seek(s)
        rec = json.loads(self._tfh.read(n).decode('utf-8'))
        t = rec.get('text') or ''
        self.texts[i] = t
        return t

    def bulk_text_load(self):
        r"""把所有惰性 text **按文件序**一次读出并缓存（整份重写 / dense 编码等全量消费者用）。

        按偏移排序后 seek 单调 ⇒ 实际顺序读；随机逐条 fetch 在 1M 上会慢好几倍。
        """
        need = [i for i, t in enumerate(self.texts)
                if t is None and self._toff is not None and int(self._toff[i, 0]) >= 0]
        if not need:
            return
        need.sort(key=lambda i: int(self._toff[i, 0]))
        with open(self._tpath, 'rb') as f:
            for i in need:
                s, n = int(self._toff[i, 0]), int(self._toff[i, 1])
                f.seek(s)
                rec = json.loads(f.read(n).decode('utf-8'))
                self.texts[i] = rec.get('text') or ''

    def rebind_from_file(self, path):
        r"""整份重写 `meta.jsonl` 之后必须重绑：从**新文件**重算行偏移，text 恢复惰性。

        ⚠️ 不重绑 = 旧偏移指向新文件里的错行 —— 静默取错 text，属最危险的一类错。
        新文件由 `_write_meta_full` 逐条写出 ⇒ 第 i 行 ⇔ 第 i 条，映射天然 1:1。
        """
        path = Path(path)
        buf = path.read_bytes()                    # ⚠️ 瞬时一份全文（重写路径本来就重）
        if buf:
            nl = np.frombuffer(buf, dtype=np.uint8) == 10
            ends = np.flatnonzero(nl)
            if ends.size and ends[-1] != len(buf) - 1:
                ends = np.concatenate([ends, [len(buf) - 1]])
            elif not ends.size:
                ends = np.asarray([len(buf) - 1])
            starts = np.concatenate([[0], (ends[:-1] + 1)]).astype(np.int64)
            lens = (ends - starts + 1).astype(np.int64)
            self._toff = np.stack([starts, lens], axis=1)
        else:
            self._toff = np.zeros((0, 2), dtype=np.int64)
        self._tpath = path
        self.texts = np.full(len(self), None, dtype=object)
        if self._tfh is not None:                  # 旧句柄指向旧 inode 内容，重开
            self._tfh.close()
            self._tfh = None

    # ---------------- 写面（列表手术的等价原语） ----------------
    def append(self, d):
        self.rows = np.concatenate([self.rows, [int(d.get('row', -1))]])
        self.lines = np.concatenate([self.lines, [int(d.get('line', 0))]])
        self.char_starts = np.concatenate([self.char_starts,
                                           np.asarray([int(d.get('char_start', -1))], dtype=np.int32)])
        self.char_ends = np.concatenate([self.char_ends,
                                         np.asarray([int(d.get('char_end', -1))], dtype=np.int32)])
        # ⭐ X14：object 数组上的追加也是 numpy 拼接（小量 O(1) 级）
        self.docids = np.concatenate([self.docids, np.asarray([d.get('docid')], dtype=object)])
        self.files = np.concatenate([self.files, np.asarray([d.get('file')], dtype=object)])
        self.texts = np.concatenate([self.texts, np.asarray([d.get('text') or ''], dtype=object)])
        # ⭐ 不变量：`_toff` 与条目**等长**（无文件来源的行垫 -1）。
        #    追加不动旧行 ⇒ 旧偏移天然有效；新条目 text 已就位 ⇒ -1 行永远不会被读。
        if self._toff is not None:
            self._toff = np.concatenate(
                [self._toff, np.full((1, 2), -1, dtype=np.int64)])

    def filter_by_file(self, drop_rels):
        r"""一次 `np.isin` 同时给出 `(保留载体, 墓碑位置)`。

        ⭐ X14：旧实现在调用方再做一趟 Python 循环取墓碑（两趟 O(N)）；isin 在
        object 数组上是 C 循环，@1M 一趟 ≈40 ms。顺序保持；`_toff` 同步切片 ⇒ 偏移仍对齐。
        """
        drop = list(drop_rels) if drop_rels else []
        if drop:
            mask = np.isin(self.files, np.asarray(drop, dtype=object))
        else:
            mask = np.zeros(len(self), dtype=bool)
        tomb = np.flatnonzero(mask)
        keep = np.flatnonzero(~mask)
        return self._take(keep), tomb

    def with_appended(self, dicts):
        r"""**就地**追加并返回自身（X14）。

        ⚠️ 语义变化：不再复制整个载体（旧版 `_take(全量)` 在 @1M 是 0.5 s 的纯浪费——
        cProfile 抓到的主犯）。调用方**不得**再持有旧引用（update 的用法正是如此）。
        需要保留旧载体时用 `merged_sorted_by_row`（它显式走 numpy 廉价的 `_take` 拷贝）。
        """
        for d in dicts:
            self.append(d)
        return self

    def merged_sorted_by_row(self, add_dicts):
        """`sorted(本载体 + add, key=row)` 的等价（add 的行号不在本载体里 —— 调用方保证）。

        ⭐ 本方法**不就地**：先 `_take(全量)`（numpy 花式索引，@1M ≈20 ms）再追加排序。
        """
        n = len(self)
        s = self._take(np.arange(n, dtype=np.int64))
        for d in add_dicts:
            s.append(d)
        order = np.argsort(s.rows, kind='stable')
        return s._take_order(order)

    def entries_in_row_range(self, lo, hi):
        """`lo <= row < hi` 的条目（物化成 dict；折叠写 delta 的 meta_rows 用）。"""
        sel = np.flatnonzero((self.rows >= lo) & (self.rows < hi))
        return [dict(c, text=c['text']) for c in (self[int(i)] for i in sel)]

    def rows_where_row_ge(self, floor, top=None):
        """行号 ∈ [floor, top) 的下标（追加游标判定用；只碰数组，不物化）。"""
        m = self.rows >= floor
        if top is not None:
            m &= self.rows < top
        return np.flatnonzero(m).tolist()

    # ---------------- 内部 ----------------
    def close(self):
        """释放 `meta.jsonl` 惰性读取文件句柄（幂等）。"""
        fh = getattr(self, "_tfh", None)
        if fh is not None:
            try:
                fh.close()
            except Exception:
                pass
            self._tfh = None

    def truncate_rows_lt(self, limit: int):
        """保留 `row < limit` 的条目（用于 `unmount` 后同步回退 `_meta`）。"""
        keep = np.flatnonzero(self.rows < int(limit))
        return self._take(keep)

    def _take(self, keep):
        keep = np.asarray(keep, dtype=np.int64)
        s = _MetaSeq.__new__(_MetaSeq)
        # ⭐ X14：全 numpy 花式索引 —— @1M 的 C 级拷贝，替代旧的逐元素列表重建
        s.rows = self.rows[keep]
        s.lines = self.lines[keep]
        s.char_starts = self.char_starts[keep]
        s.char_ends = self.char_ends[keep]
        s.docids = self.docids[keep]
        s.files = self.files[keep]
        s.texts = self.texts[keep]
        s._toff = (self._toff[keep] if self._toff is not None else None)
        s._tpath = self._tpath
        s._tfh = None                          # 派生载体按需重开，不与旧载体共享句柄
        return s

    def _take_order(self, order):
        order = np.asarray(order, dtype=np.int64)
        s = _MetaSeq.__new__(_MetaSeq)
        s.rows = self.rows[order]
        s.lines = self.lines[order]
        s.char_starts = self.char_starts[order]
        s.char_ends = self.char_ends[order]
        s.docids = self.docids[order]
        s.files = self.files[order]
        s.texts = self.texts[order]
        s._toff = (self._toff[order] if self._toff is not None else None)
        s._tpath = self._tpath
        s._tfh = None
        return s


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
        self.load_report = None            # ⭐ X13：`_load` 后填（delta 回放 applied/stopped/error）
        self.last_rerank_status = None     # ⭐ 0.3：最近一次 rerank/dense 状态（含降级原因）
        self.last_confidence = None        # ⭐ 支线 #15：最近一次 confidence=True 查询摘要
        self._hash_events_n = 0            # ⭐ X14：hashes.jsonl 未压实的事件数（加载时重数）

    def _tok_fn(self):
        """返回与当前索引持久化的 `tokenizer_mode` 一致的分词函数。"""
        return get_tokenizer(getattr(self._bm, "tokenizer_mode", "zh_default") if self._bm is not None else "zh_default")

    # ---------- 建索引 ----------

    @classmethod
    def build(cls, root, out='.primolix_index', dense=False, model_path='',
              workers=0, skip_dirs=None, w_cache=False, stream=False, code_aware=False):
        r"""建索引。`w_cache=True` → 同时把物化层 `w` 落盘（代价：盘 +`nnz×4 B`，默认关）。

        `stream` 控制分词期是否驻留 token（不改分数，只改建库内存峰值）：
          `False`（默认）＝ 分词一次并缓存，峰值最高；
          `True` ＝ 两遍各自分词、不驻留，代价是 jieba 跑两遍；
          `"ids"` ＝ 分词一次、立刻转 id、只留 `int32`，峰值最低且不重分词。
        `code_aware=True`：启用 `zh_code` 代码标识符分词并随索引落盘（默认 `False` 不改旧行为）。
        该选择随索引落盘，重载后由 OOV 触发的重建继续用它。
        """
        self = cls(out, dense=dense, model_path=model_path)
        self._w_cache = bool(w_cache)
        self._stream = _norm_stream(stream)
        tok_fn = get_tokenizer("zh_code" if code_aware else "zh_default")
        t0 = time.time()
        _say("扫描文件...")
        eff_skips = [self.dir, *(skip_dirs or [])]
        files, _stamps = scan_root_ex(Path(root), eff_skips, with_spans=True, with_ctx=True)
        chunks = []
        index_texts = []
        for rel, h, cs in files:
            for item in cs:
                t, r, ln, c0, c1 = item[:5]
                ctx = item[5] if len(item) > 5 else ""
                # 初始 build: chunk i ↔ SparseBM25 物理行 i (一一对应)；text 严格保持原文切片
                chunks.append(dict(docid=f'{h[:8]}:{len(chunks)}', row=len(chunks),
                                   file=r, line=ln, char_start=c0, char_end=c1, text=t))
                index_texts.append(f"{ctx}\n{t}" if ctx else t)
        self.dir.mkdir(parents=True, exist_ok=True)
        _say("构建 BM25 词表...")
        self._bm = SparseBM25(index_texts, tok_fn,
                              workers=workers, stream=self._stream)
        self._meta = _MetaSeq.from_dicts(chunks)      # ⭐ X13：数组化载体（text 暂常驻，落盘后转惰性）
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

        `meta=False`（0.2.3）：**不重写 `meta.jsonl`**。1M 上整份重写要 **5.45 s**，
        而 delta 模式下它可以靠追加式维护（`_meta_append_pending`）＋ delta 自带的身份兜底。
        ⚠️ 用 `meta=False` 前必须先调一次 `_meta_append_pending()` 让文件追上，否则会留下"有行没身份"。

        `hashes.json` 自 0.2.2 起是 `{"v":2, "hashes": {...}, "stamps": {rel: [size, mtime_ns]}}`；
        `stamps` 是 `verify="stat"` 快扫的物理前提（缺它就退化成每次读内容）。

        ⭐ 全量落盘会**盖掉之前所有 delta**（npz 里已经有那些行）⇒ 写完就清 `.delta/`。
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

        **第一层（便宜，常用）**：`delta 文件数 ≥ DELTA_MAX_FILES` ⇒ `_merge_deltas()`
        —— 只把已有的几个 delta 合成一个，**代价 O(Σ delta nnz)**，⛔ 不动基线。
        **第二层（贵，罕见）**：`Σdelta.nnz > DELTA_COMPACT_RATIO × base.nnz` ⇒ 全量 `_persist()`
        —— 把 delta 折进 npz，代价 O(总 nnz)。

        为什么要分层：1M 上全量落盘 34 s，而 delta 自己只有几十 KB。原来只要 delta 攒够就付 34 s，
        等于把"写变便宜"的收益在压实点全还回去。
        ⚠️ 只在 `PERSIST_MODE == "delta"` 且当前没有挂段时才会真的写盘。
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
        旧文件即使还在也不会被回放。**合并后链不断**（base_N 取第一个、N_after 取最后一个）。
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

    def _write_hashes(self, events=None):
        r"""写哈希账本：默认**追加事件**（`hashes.jsonl`，O(改动)）；`events=None` ⇒ 整份快照。

        ⭐ X14（2026-11-18）：增量收尾不再整份重写 `hashes.json`（@20k 文件 ≈140 ms/次，
        delta 落盘之后残存的最后一个整份重写）。设计照抄 `meta.jsonl` 追加式的先例：
          · 事件行：`{"rel":..., "md5":..., "stamp":[size, mtime_ns]}` / `{"del": rel}`；
          · 加载 = 快照（v2 不变，旧读方照常）＋ 事件回放（`_load`）；
          · 事件攒过 `HASHES_COMPACT_N` 条 → 快照整理一次并清日志。
        ⚠️ `stamps` 是 `verify="stat"` 快扫的物理前提 ⇒ 事件必须同步带 stamp。
        """
        if events is None:
            doc = {"v": 2,
                   "hashes": self._hashes or {},
                   "stamps": {k: [int(v[0]), int(v[1])]
                              for k, v in (getattr(self, "_stamps", None) or {}).items()}}
            with open(self.dir / 'hashes.json', 'w', encoding='utf-8') as f:
                json.dump(doc, f, ensure_ascii=False, indent=1)
            (self.dir / 'hashes.jsonl').unlink(missing_ok=True)   # 快照即全量 ⇒ 清事件日志
            self._hash_events_n = 0
            return
        n_ev = 0
        with open(self.dir / 'hashes.jsonl', 'a', encoding='utf-8') as f:
            for ev in events:
                if ev[0] == 'del':
                    f.write(json.dumps({"del": ev[1]}, ensure_ascii=False) + '\n')
                else:
                    _, rel, md5, stamp = ev
                    sz, mt = (int(stamp[0]), int(stamp[1])) if stamp else (0, 0)
                    f.write(json.dumps({"rel": rel, "md5": md5, "stamp": [sz, mt]},
                                       ensure_ascii=False) + '\n')
                n_ev += 1
        self._hash_events_n = int(getattr(self, "_hash_events_n", 0) or 0) + n_ev
        if self._hash_events_n > HASHES_COMPACT_N:
            self._write_hashes()                                  # 攒多了 → 快照整理

    def _persist_under_segments(self, del_rows=None, events=None):
        """段在场时的落盘：`meta.jsonl`（**只写主表行**）＋ 哈希账本，⛔ 不动 `bm25.npz`。

        为什么不能直接 `_persist()`：它内含 `SparseBM25.save()`，而 `save()` 在段在场时
        **直接抛错**（挂段是内存视图，写出的 npz 会与 `dl`/`T` 不一致）。
        为什么 `meta.jsonl` 只写 `row < T.shape[0]` 的行：段内身份走段级侧车
        （`docs.jsonl`），重载由 `_absorb_staged` 补回；两边都写会引入双重计数，
        虽然按行去重挡得住，但⛔ 不该靠它。
        ⭐ 0.2.3：改成**追加式**（`ceiling=seg_floor`），不再整份重写。
        """
        seg_floor = int(self._bm.T.shape[0])
        self._meta_append_pending(del_rows=del_rows, ceiling=seg_floor)
        self._write_hashes(events=events)    # ⭐ X14：None ⇒ 整份（旧路径），events ⇒ 追加

    # ---------- L2：增量持久化（delta 日志） ----------

    def _write_meta_full(self):
        """整份重写 `meta.jsonl`（全部活跃 chunk）。⛔ 不写 npz —— 构建/压实用它。

        ⭐ X13：text 惰性载体下重写两步走 —— 先 `bulk_text_load()`（按文件序一次取回，
        免掉 1M 次随机小读），写完 `rebind_from_file()` 从**新文件**重算行偏移、
        text 恢复惰性（不重绑 = 旧偏移指向新文件的错行，静默取错 text —— 最危险的一类错）。
        """
        meta_path = self.dir / 'meta.jsonl'
        _MetaSeq.invalidate_cache(meta_path)
        self._meta.bulk_text_load()
        with open(meta_path, 'w', encoding='utf-8') as f:
            for c in self._meta:
                f.write(json.dumps(dict(c, text=c['text']), ensure_ascii=False) + '\n')
        self._meta.rebind_from_file(meta_path)
        max_r = int(self._meta.rows.max()) if len(self._meta) else -1
        self._meta.write_cache(meta_path, max_r, 0)
        self._persisted_N = int(self._bm.N) if self._bm is not None else 0
        self._meta_dels = 0

    def _meta_append_pending(self, del_rows=None, ceiling=None) -> dict:
        """**追加式**维护 `meta.jsonl`（2026-11-10）：只写新增行 ＋ 一条删除记录。

        为什么能省：`meta.jsonl` 是 99.9 万条 chunk 的 JSON，整份重写在 1M 上要 **5.45 s**；
        而一次增量只多几行。追加的代价是 O(新增行数)，删除用 `{"del": [row…]}` 记一笔。
        读取时按 `row` 建表、遇 `del` 剔除，再按 `row` 排序。

        ⚠️ 只在 **delta 模式**下安全：那里 delta 自带身份，即使 `meta.jsonl` 落后也不会"有行没身份"；
        npz 模式没有第二份身份，仍走整份重写（见 `_persist`）。
        """
        bm = self._bm
        floor = int(getattr(self, '_persisted_N', 0) or 0)
        top = int(ceiling) if ceiling is not None else None
        # ⭐ X13：只读 row 的全表扫改走数组直达（旧实现逐条取 dict['row']，
        #    惰性载体下那会物化 1M 个 dict —— 白付）
        added = [self._meta[int(i)] for i in self._meta.rows_where_row_ge(floor, top)]
        dels = [int(r) for r in (del_rows if del_rows is not None else [])]
        if not added and not dels:
            return dict(added=0, dels=0, bytes=0)
        path = self.dir / 'meta.jsonl'
        _MetaSeq.invalidate_cache(path)
        nbytes = 0
        with open(path, 'a', encoding='utf-8') as f:
            for c in added:
                line = json.dumps(dict(c, text=c['text']), ensure_ascii=False) + '\n'
                f.write(line)
                nbytes += len(line.encode('utf-8'))
            if dels:
                line = json.dumps({"del": dels}, ensure_ascii=False) + '\n'
                f.write(line)
                nbytes += len(line.encode('utf-8'))
        if added:
            self._persisted_N = max(int(c['row']) for c in added) + 1
        else:
            # ceiling 在场（段已挂）时绝不能把 _persisted_N 越过 top，否则后续折叠写成 delta 会跳过段内行
            cap = top if top is not None else (int(bm.N) if bm is not None else floor)
            if bm is not None and top is not None:
                cap = min(cap, int(bm.N))
            self._persisted_N = max(floor, cap)
        self._meta_dels = int(getattr(self, '_meta_dels', 0) or 0) + len(dels)
        if self._meta_dels > 4096 and top is None:  # 段在场时不触发全写，免得把未折叠段行写入 meta.jsonl
            self._write_meta_full()
        return dict(added=len(added), dels=len(dels), bytes=nbytes)

    def _persist_delta(self, new_rows, dead_rows, meta_rows=None) -> dict:
        """把这一批的追加行与墓碑写成 `delta/<seq>.npz`（**只写 O(新 nnz)**）。

        提交语义与段一致：先写 `.tmp`，再 `os.replace` —— **rename 才是提交点**。
        `base_N` 记下"这一批写之前主表多少行"，回放时按它做链式校验（对不上就停，⛔ 不猜）。

        ⭐ `meta_rows`（2026-11-10 补）：这批新 chunk 的**身份**（docid/file/line/text）也写进 delta。
        为什么必须带着走：写入顺序是 `delta → meta.jsonl → hashes.json`，若崩在 delta 与 meta 之间，
        盘上就会"有行、没身份" —— 重载后这些行**占着 N、算进 df，却永远搜不到**，而且下次
        `update()` 看到 hash 未变就 noop ⇒ **不会自愈**。带上身份之后，rename 就是唯一提交点，
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
        meta_payload = [{k: c.get(k) for k in ("docid", "file", "line", "text", "row", "char_start", "char_end")
                         if c.get(k) is not None}
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
        have = {int(r) for r in self._meta.rows.tolist()}        # ⭐ X13：数组直达
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
            self._meta = self._meta.merged_sorted_by_row(add)
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
                # ⭐ 词表先扩到 delta 记的宽度：段带来的新词不在 npz 里，必须由 delta 带着
                vt = json.loads(str(z["vocab_json"])) if "vocab_json" in z.files else []
                if vt:
                    bm._extend_vocab(vt)
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
        # ⭐ df 必须跟着补：npz 里的 df 不含 delta 行的贡献；每行内列号唯一
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
        r"""读 `meta.jsonl`（支持追加式的 `{"del": [row…]}` 记录）→ `(_MetaSeq, 最大row, 删除数)`。

        ⭐ X13（2026-11-18）：返回**数组化载体**，不再构造 N 个 dict —— [实测] 那是加载瓶颈
        （二进制 v2 的教训：解析不是大头，dict 构造才是；见 `06_meta格式 §2.5`）。
        text 惰性：按行偏移从本文件取（`_MetaSeq._text_at`），**零格式改动**。
        ⚠️ 遗留字段 `chunk_id` 不再进载体（包内无人写它；唯一读点 `load_views` 的兜底
        同步改为只认 `docid` —— 0.1 之前的古索引才可能只有 chunk_id）。
        """
        path = self.dir / 'meta.jsonl'
        if not path.exists():
            return _MetaSeq.from_dicts([]), -1, 0
        return _MetaSeq.from_jsonl(path)

    def _load(self):
        # 0.3 / 0.3.1：启动预检核心文件 + 加载内核 + 挂载视图侧车
        npz_path = self.dir / 'bm25.npz'
        if not npz_path.exists():
            raise FileNotFoundError(
                f"索引不完整：目录 {self.dir} 缺少 bm25.npz（可用 `primolix index` 重建或补回该文件）"
            )
        self._bm = SparseBM25.load(str(npz_path))
        self._meta, _meta_top, _meta_ndel = self._read_meta()
        self._persisted_N = int(_meta_top) + 1      # 文件里已经有的行；之后的要追加
        self._meta_dels = int(_meta_ndel)
        self._rows = None          # 失效行号缓存
        # 建库分词档随索引落盘（`kernel.stream_mode`），重载后重建仍用它；
        #    缺字段 = 旧版索引 → 按默认档。
        self._stream = _norm_stream(getattr(self._bm, "stream_mode", "off"))
        hf = self.dir / 'hashes.json'
        if hf.exists():
            with open(hf, encoding='utf-8') as _hf:
                doc = json.load(_hf)
        else:
            doc = {}
        if isinstance(doc, dict) and doc.get("v") == 2:      # 0.2.2+：hashes ＋ stamps
            self._hashes = doc.get("hashes") or {}
            self._stamps = {k: (int(v[0]), int(v[1]))
                            for k, v in (doc.get("stamps") or {}).items()}
        else:                                                # 旧格式：rel → md5，无快扫印记
            self._hashes = doc or {}
            self._stamps = {}
        # ⭐ X14：回放哈希事件日志（若有）—— 追加式的对账点在加载侧，快照 v2 不动。
        self._hash_events_n = 0
        evp = self.dir / 'hashes.jsonl'
        if evp.exists():
            with evp.open(encoding='utf-8') as f:
                for ln in f:
                    ln = ln.strip()
                    if not ln:
                        continue
                    rec = json.loads(ln)
                    if 'del' in rec:
                        self._hashes.pop(rec['del'], None)
                        self._stamps.pop(rec['del'], None)
                    else:
                        self._hashes[rec['rel']] = rec['md5']
                        self._stamps[rec['rel']] = (int(rec['stamp'][0]), int(rec['stamp'][1]))
                    self._hash_events_n += 1
        # L2：回放 delta 日志（若有）。⛔ 必须在挂段之前 —— 段的 row_base 依赖 `bm.N`。
        rep = dict(applied=0, stopped=0, error=None)
        try:
            rep.update(self._replay_deltas() or {})
        except Exception as _e:                              # noqa: BLE001
            rep['error'] = repr(_e)
            _say(f"[警告] delta 回放失败（{_e!r}）→ 按 npz 状态可用，但最近的增量可能缺失")
        # ⭐ X13：回放报告常驻（`info['load_report']`）——「最近增量缺失」必须可观测，
        #    这是 `PERSIST_MODE` 翻默认前的安全前提（吞成 stderr 警告 = 静默偏差）。
        self.load_report = rep
        self._vecs = None
        # 临时段发布：按旁车 manifest 重挂已发布的段（无旁车 → 空操作）。
        try:
            _rm = remount_staged(self._bm, self.dir)
            if _rm:
                _ok = [r for r in _rm if r.get("mounted")]
                _skip = [r for r in _rm if not r.get("mounted")]
                _say(f"[staged] 按旁车重挂 {len(_ok)} 个段"
                     + (f" · 跳过 {len(_skip)}（{_skip[0].get('reason', '')[:60]}）" if _skip else ""))
                # ⭐ 乙档：把已挂段里的**文档身份**补回 `_meta`（否则段内文档在产品层不可见）
                _ab = self._absorb_staged(_ok)
                if _ab["added"]:
                    _say(f"[staged] 已把 {_ab['added']} 篇段内文档补回 `_meta`"
                         f"（产品层可见；跳过 {_ab['skipped']}）")
        except Exception as _e:                                      # noqa: BLE001
            _say(f"[警告] 旁车重挂失败（{_e!r}）→ 该索引按 npz 状态可用，但已发布的段未挂上")
        # ⭐ 0.3：在段身份吸纳之后挂载 views/*.npz（按 docid 对齐，含段内文档）
        if (self.dir / "views").is_dir():
            try:
                self.load_views()
            except Exception as _e:                                  # noqa: BLE001
                _say(f"[警告] 视图侧车加载失败（{_e!r}）→ 退回 BM25-only")
        # 旧版 dense.npy 只读兼容：若无 `dense` 视图但盘上有 dense.npy，则挂成只读视图（不再继续产出 dense.npy）
        if "dense" not in (getattr(self._bm, "_views", None) or {}):
            legacy_dense = self.dir / 'dense.npy'
            if legacy_dense.exists():
                v = np.load(legacy_dense)
                if len(v) == len(self._meta) and len(v) > 0:
                    self.attach_view("dense", v, meta={"legacy": True}, replace=True)
                    _say("[提示] 检测到旧版 dense.npy（0.3 已退役），已转挂为可丢视图 `dense`")
                else:
                    _say(f"[警告] 旧 dense.npy 向量数({len(v)}) != 活跃 chunk({len(self._meta)}), 忽略")
        self._dense = self._dense or ("dense" in (getattr(self._bm, "_views", None) or {}))

    def _absorb_staged(self, mounted) -> dict:
        r"""把已挂段里的文档身份补回 `self._meta`（乙档：产品层可见性的唯一写入方）。

        ⭐ 行号以**实际挂载位置**为准（`bm._segments[tag]['row_base']`），⛔ 不信侧车里记的
        `row` —— `reset`/多次重挂之后同一段的行号可能与发布时不同。
        ⭐ **幂等（按物理行去重）**：`save()` 在段在场时直接抛错（`kernel.py` 的 `save`），
        所以"`_meta` 已有、段又挂上"的结构性来源不存在；去重是为将来放宽留的保险。
        ⚠️ `_vecs` 与 `_meta` 逐位对齐 ⇒ 追加后长度不符就必须**弃用 dense**并如实告警。
        """
        added = skipped = 0
        have = ({int(r) for r in self._meta.rows.tolist()}
                if self._meta is not None else set())            # ⭐ X13：数组直达
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
                entry = dict(docid=docid, row=row,
                             file=d.get("file") or docid,
                             line=int(d.get("line") or 1),
                             text=d.get("text") or "")
                if d.get("char_start") is not None and int(d["char_start"]) >= 0:
                    entry["char_start"] = int(d["char_start"])
                    entry["char_end"] = int(d["char_end"])
                self._meta.append(entry)
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
        """0.3：向量全量编码并写入可丢视图 `views/dense.npz`（废除 `dense.npy`）。"""
        from .rerank import DEFAULT_SUB_BATCH, encode_texts, get_rerank_model

        model, st = get_rerank_model(self._model_path, device="auto")
        if model is None:
            self.last_rerank_status = dict(st, mode="bm25_fallback")
            if st.get("reason"):
                _say(f"[提示] {st['reason']}")
            return
        if len(self._meta) == 0:
            return
        t0 = time.time()
        self._meta.bulk_text_load()          # 按文件序一次取回全部正文
        vecs = encode_texts(model, [c['text'] for c in self._meta],
                            sub_batch=DEFAULT_SUB_BATCH, device=st.get("device"))
        docids = [str(d) for d in self._meta.docids]
        meta_info = {"model_id": str(st.get("model") or "dense"),
                     "model_path": str(st.get("model_path") or "")}
        self.save_view("dense", docids, vecs, meta=meta_info)
        self.attach_view("dense", vecs, meta=meta_info, replace=True)
        self._vecs = None
        _say(f"dense 视图 {vecs.shape} 落盘 views/dense.npz "
             f"{time.time() - t0:.1f}s ({len(self._meta) / max(time.time() - t0, 1e-6):.1f} 段/s)")

    def _sync_dense_view_incremental(self, append_chunks):
        """0.3 增量视图维护：仅编码本批新增/修改的 `append_chunks`，其余复用既有 `dense` 视图。"""
        from .rerank import DEFAULT_SUB_BATCH, encode_texts, get_rerank_model

        model, st = get_rerank_model(self._model_path, device="auto")
        if model is None:
            self.last_rerank_status = dict(st, mode="bm25_fallback")
            if st.get("reason"):
                _say(f"[提示] {st['reason']}")
            return
        old_map: dict[str, np.ndarray] = {}
        vpath = self.dir / "views" / "dense.npz"
        if vpath.exists():
            try:
                z = np.load(vpath, allow_pickle=False)
                for did, vec in zip(z["docids"], z["vecs"]):
                    old_map[str(did)] = np.asarray(vec, dtype=np.float32)
            except Exception:                                        # noqa: BLE001
                old_map = {}
        elif "dense" in (getattr(self._bm, "_views", None) or {}):
            v = self._bm._views["dense"]
            row_to_did = {int(r): str(d) for r, d in zip(self._meta.rows, self._meta.docids)}
            for r, vec in zip(v["rows"].tolist(), v["vectors"]):
                did = row_to_did.get(int(r))
                if did is not None:
                    old_map[did] = np.asarray(vec, dtype=np.float32)

        new_texts = [c["text"] for c in (append_chunks or [])]
        new_dids = [str(c["docid"]) for c in (append_chunks or [])]
        if new_texts:
            new_vecs = encode_texts(model, new_texts, sub_batch=DEFAULT_SUB_BATCH,
                                    device=st.get("device"))
            for did, vec in zip(new_dids, new_vecs):
                old_map[did] = vec

        active_dids: list[str] = []
        active_pos: list[int] = []
        active_vecs: list[np.ndarray] = []
        for idx_m, did in enumerate(self._meta.docids):
            sd = str(did)
            if sd in old_map:
                active_dids.append(sd)
                active_pos.append(idx_m)
                active_vecs.append(old_map[sd])
        if not active_vecs:
            if "dense" in (getattr(self._bm, "_views", None) or {}):
                self.detach_view("dense")
            return
        mat = np.stack(active_vecs, axis=0).astype(np.float32, copy=False)
        meta_info = {"model_id": str(st.get("model") or "dense"),
                     "model_path": str(st.get("model_path") or "")}
        self.save_view("dense", active_dids, mat, meta=meta_info)
        self.attach_view("dense", mat, np.asarray(active_pos, dtype=np.int64),
                         meta=meta_info, replace=True)

    # ---------- 统计 ----------

    @property
    def info(self):
        if self._bm is None:
            self._load()
        bm_path = self.dir / 'bm25.npz'
        views_d = getattr(self._bm, "_views", None) or {}
        # 新格式无 W：形状/规模走 SparseBM25.V/.nnz（权威 T 优先，旧格式回退 W）
        return dict(N=len(self._meta),
                    V=int(self._bm.V), nnz=int(self._bm.nnz),
                    size_mb=round(bm_path.stat().st_size / 1e6, 2) if bm_path.exists() else 0.0,
                    dense=bool(self._vecs is not None or "dense" in views_d),
                    views=len(views_d),
                    files=len(self._hashes),
                    load_report=getattr(self, 'load_report', None),
                    rerank_status=getattr(self, 'last_rerank_status', None))

    # ---------- 检索 ----------

    def _active_rows(self):
        r"""活跃 chunk 的物理行号数组 (缓存)。

        只在 `_meta` 变化时变, 故缓存; 每个 `self._meta =` 赋值点都要把 `self._rows`
        置 `None` (与 `_w_persist_key` 同一条纪律: 键不符还用 = 静默错分)。
        """
        if self._rows is None:
            n = len(self._meta)
            # ⭐ X13：`rows` 就是数组 —— `.copy()` 保住「重建必出新对象」的失效纪律
            self._rows = (self._meta.rows.copy()
                          if n else np.zeros(0, dtype=np.int64))
            # ⭐ 01m：顺手判「恒等」（`_meta` 的行号恰是 `0..N-1` 且与物理行一一对应）。
            #    判一次 O(n)（只在这条缓存重建时付）；此后 `_score_active` 每查询免掉
            #    `gather` ＋ `astype`（[实测] @1M ≈3.2–3.6 ms/查询）。
            #    ⚠️ 用**对象身份**当失效键：六处 `self._rows = None` 的失效点都会让本函数
            #    重建出一个**新数组对象** ⇒ 这条判定自动作废，⛔ 无须（也不会）漏改失效点。
            #    ⚠️⚠️ **失效是惰性的**：`_rows = None` 之后、下一次 `_active_rows()` 之前，
            #    `_rows_ident` 仍是上一次的判决 ⇒ **只在 `_active_rows()` 之后读它才有意义**。
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

        ⭐ **01m**：`_meta` 行号**恒等**时（新建索引未删未迁的常见形态）直接返回
        `score_all` 的数组 —— 省掉一整趟 `gather` ＋ `astype` 拷贝。
        ⚠️ 返回的是 `score_all` 当次**新建**的数组（不与任何常驻缓冲共享）⇒ 调用方可安全
        持有；但调用方 ⛔ 不得就地改写它（本包内唯一调用方 `query` 只读它）。
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

    def _bm25_pool(self, q, pool_k):
        """返回一条查询的 `(pool_meta_indices, pool_bm25_scores)`（均为正分、降序）。"""
        s = self._score_active(self._tok_fn()(q))
        if s.size == 0 or not bool((s > 0).any()):
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32)
        order = _topk_idx(s, min(int(pool_k), int(s.size)))
        pos = np.asarray([int(i) for i in order if s[i] > 0], dtype=np.int64)
        return pos, s[pos].astype(np.float32, copy=False)

    def _attach_span_fields(self, item, q, m):
        from .span import hit_ranges, hit_span, query_tokens

        txt = item.get('text')
        if txt is not None:
            item['hit_span'] = hit_span(txt, q)
            item['hit_ranges'] = hit_ranges(txt, query_tokens(q))
        if 'char_start' in m:
            item['char_start'] = m['char_start']
            item['char_end'] = m['char_end']

    def _resolve_reranker(self, rerank, pool_k):
        from .rerank import Reranker

        if isinstance(rerank, Reranker):
            return rerank
        mp = rerank if isinstance(rerank, str) else self._model_path
        return Reranker(model_path=mp, pool_k=pool_k)

    def _attach_confidence(self, qtoks, hits, hit_rows):
        from .confidence import evaluate_confidence

        n_live = int(self._active_rows().size) or 1
        summary = evaluate_confidence(self._bm, qtoks, hits, hit_rows=hit_rows, n_live=n_live)
        self.last_confidence = summary
        return hits

    def _dense_int8_sidecar(self):
        sc = getattr(self, "_d8_sidecar", None)
        if sc is None:
            from .dense_int8 import DenseInt8Sidecar

            sc = DenseInt8Sidecar(self.dir)
            self._d8_sidecar = sc
        return sc

    def build_dense_int8(self, *, model_path="", device="auto", sub_batch=256,
                         fp16=True, encoder=None):
        """显式按需构建可选 `views/dense_int8` 紧凑侧车；缺模型时诚实降级不抛异常。"""
        if self._bm is None:
            self._load()
        self._meta.bulk_text_load()
        docids = [str(d) for d in self._meta.docids]
        texts = [self._meta[i]["text"] for i in range(len(self._meta))]
        sc = self._dense_int8_sidecar()
        st = sc.build_and_save(
            docids, texts,
            model_path=str(model_path or self._model_path or ""),
            device=device, sub_batch=sub_batch, fp16=fp16, encoder=encoder,
        )
        self.last_rerank_status = st
        return st

    def dense_int8_status(self):
        """返回可选 `views/dense_int8` 侧车的可用性与对齐覆盖率（从不抛错）。"""
        if self._bm is None:
            self._load()
        docids = [str(d) for d in self._meta.docids]
        return self._dense_int8_sidecar().status(docids)

    def _query_dense_int8_union(self, q, qtoks, *, k=10, rerank=False,
                                pool_k=100, dense_k=100, span=False, confidence=False):
        """可选 `BM25 ∪ INT8_Dense` 并池检索；缺侧车/缺编码器/维不符时返回 None 并记录诚实降级状态。"""
        from .dense_int8 import union_fuse_scores
        from .rerank import Reranker, encode_texts, get_rerank_model

        sc = self._dense_int8_sidecar()
        if not sc.load():
            self.last_rerank_status = {
                "available": False, "mode": "bm25_fallback", "dense_union": False,
                "reason": sc.load_error or "dense_int8 侧车不可用",
            }
            return None

        rr_obj = rerank if isinstance(rerank, Reranker) else None
        if rr_obj is not None and rr_obj._encoder is not None:
            enc = rr_obj._encoder
            used_dev = "custom"
            st_enc = {"available": True, "model": getattr(enc, "__name__", type(enc).__name__),
                      "device": "custom"}
        else:
            mp = ""
            if rr_obj is not None and rr_obj.model_path:
                mp = rr_obj.model_path
            elif isinstance(rerank, str):
                mp = rerank
            elif self._model_path:
                mp = self._model_path
            elif sc.meta.get("model_path"):
                mp = str(sc.meta["model_path"])
            dev = rr_obj.device if rr_obj is not None else "auto"
            enc, st_enc = get_rerank_model(mp, device=dev)
            if enc is None:
                self.last_rerank_status = dict(st_enc, mode="bm25_fallback", dense_union=False)
                return None
            used_dev = str(st_enc.get("device") or "cpu")

        qvecs = encode_texts(enc, [q], sub_batch=1, device=used_dev)
        if qvecs.ndim != 2 or qvecs.shape[0] == 0 or (
            sc.vecs_int8 is not None and qvecs.shape[1] != sc.vecs_int8.shape[1]
        ):
            self.last_rerank_status = {
                "available": False, "mode": "bm25_fallback", "dense_union": False,
                "reason": "查询编码维度与 dense_int8 侧车不匹配",
            }
            return None
        qv = qvecs[0]

        active_docids = [str(d) for d in self._meta.docids]
        m_pos, v_rows, unmatched = sc.align_active(active_docids)
        if m_pos.size == 0:
            self.last_rerank_status = {
                "available": False, "mode": "bm25_fallback", "dense_union": False,
                "reason": "dense_int8 侧车与当前活跃文档无交集",
            }
            return None

        bm_all = self._score_active(qtoks)
        pk = max(int(rr_obj.pool_k) if rr_obj is not None else int(pool_k), int(k))
        dk = max(int(dense_k), int(k))
        bm_pos: list[int] = []
        if bm_all.size > 0 and bool((bm_all > 0).any()):
            bm_ord = _topk_idx(bm_all, min(pk, int(bm_all.size)))
            bm_pos = [int(x) for x in bm_ord if bm_all[x] > 0]
        d_pos_arr, _ = sc.search_aligned(qv, m_pos, v_rows, top_k=dk)

        seen: set[int] = set()
        union_pos: list[int] = []
        for p in bm_pos:
            if p not in seen:
                seen.add(p)
                union_pos.append(p)
        for p in d_pos_arr.tolist():
            pi = int(p)
            if pi not in seen:
                seen.add(pi)
                union_pos.append(pi)

        if not union_pos:
            self.last_rerank_status = dict(st_enc, mode="dense_int8_union", dense_union=True,
                                           bm_pool=0, dense_pool=0, union_pool=0)
            return []

        u_dids = [active_docids[p] for p in union_pos]
        cos_arr, cov_mask = sc.cos_for_docids(qv, u_dids)
        miss_idx = [idx_u for idx_u, ok in enumerate(cov_mask.tolist()) if not ok]
        if miss_idx:
            miss_texts = [self._meta[union_pos[idx_u]]["text"] for idx_u in miss_idx]
            miss_vecs = encode_texts(enc, miss_texts, sub_batch=max(len(miss_texts), 1), device=used_dev)
            if miss_vecs.ndim == 2 and miss_vecs.shape[0] == len(miss_idx) and miss_vecs.shape[1] == qv.size:
                for local_i, idx_u in enumerate(miss_idx):
                    cos_arr[idx_u] = float(np.dot(miss_vecs[local_i], qv))
            else:
                for idx_u in miss_idx:
                    cos_arr[idx_u] = -1.0

        bscores = bm_all[np.asarray(union_pos, dtype=np.int64)].astype(np.float32, copy=False)
        fusion = rr_obj.fusion if rr_obj is not None else "z2_safe"
        if fusion == "z2_pos":
            fusion = "z2"
        order, rank_sc = union_fuse_scores(cos_arr, bscores, fusion=fusion, w_bm=0.35, top_m=32)

        k_lim = max(int(k), 0)
        hits: list[dict] = []
        h_rows: list[int] = []
        for rank, uj in enumerate(order[:k_lim].tolist(), 1):
            mp = union_pos[uj]
            m = self._meta[mp]
            item = dict(
                rank=rank,
                score=float(rank_sc[uj]),
                bm25_score=float(bscores[uj]),
                rerank_score=float(cos_arr[uj]),
                docid=m.get("docid"),
                file=m["file"],
                line=m["line"],
                text=m["text"],
            )
            if span:
                self._attach_span_fields(item, q, m)
            hits.append(item)
            h_rows.append(int(m["row"]))

        self.last_rerank_status = dict(
            st_enc,
            mode="dense_int8_union",
            dense_union=True,
            fusion=fusion,
            bm_pool=len(bm_pos),
            dense_pool=int(d_pos_arr.size),
            union_pool=len(union_pos),
            uncovered_encoded=len(miss_idx),
            unmatched=unmatched,
        )
        if confidence:
            self._attach_confidence(qtoks, hits, h_rows)
        return hits

    def query(self, q, k=10, dense=False, *, rerank=False, pool_k=100,
              dense_union=False, dense_k=100, span=False, confidence=False):
        """检索 → [{'rank','score','docid','file','line','text', ...}]

        - `dense_union=True`（严格可选）：若 `views/dense_int8` 与编码器就绪，走 `BM25 ∪ INT8` 并池检索；
          任一缺席时自动优雅降级为纯 BM25（或既有 `rerank`），不报错并记录 `self.last_rerank_status`。
        - `rerank`（0.3）：`True` / 模型路径 / `Reranker` → BM25 粗筛池 `pool_k` 后做查询期语义重排；
          模型缺席时诚实降级 BM25-only 并记录 `self.last_rerank_status`。
        - `dense=True`：若已挂 `dense` 视图（`views/dense.npz`）则走单调融合；否则走查询期重排。
        - `span=True`：附加 `hit_span` / `hit_ranges` 及源文档 `char_start` / `char_end`。
        - `confidence=True`（支线 #15）：附加有界原语置信度 `hit['confidence']` 并更新 `last_confidence`。
        """
        if self._bm is None:
            self._load()
        qtoks = self._tok_fn()(q)
        if dense_union:
            u_hits = self._query_dense_int8_union(
                q, qtoks, k=k, rerank=rerank, pool_k=pool_k,
                dense_k=dense_k, span=span, confidence=confidence,
            )
            if u_hits is not None:
                return u_hits
            fb_status = dict(self.last_rerank_status or {})
            if rerank:
                rr = self._resolve_reranker(rerank, pool_k)
                pk = max(int(rr.pool_k), int(k))
                p_pos, p_sc = self._bm25_pool(q, pk)
                out_all, st = rr.rerank_pools(
                    [q], [p_pos], [p_sc], self._meta, bm=self._bm,
                    k=k, with_text=True, with_span=span,
                )
                if st.get("mode") == "rerank":
                    st = dict(st, dense_union_fallback=fb_status.get("reason"))
                self.last_rerank_status = st
                hits = out_all[0]
                if confidence:
                    did_to_row = {str(self._meta.docids[int(pos)]): int(self._meta.rows[int(pos)])
                                  for pos in p_pos}
                    h_rows = [did_to_row.get(str(h.get("docid")), -1) for h in hits]
                    self._attach_confidence(qtoks, hits, h_rows)
                return hits
            self.last_rerank_status = fb_status
        if rerank:
            rr = self._resolve_reranker(rerank, pool_k)
            pk = max(int(rr.pool_k), int(k))
            p_pos, p_sc = self._bm25_pool(q, pk)
            out_all, st = rr.rerank_pools(
                [q], [p_pos], [p_sc], self._meta, bm=self._bm,
                k=k, with_text=True, with_span=span,
            )
            self.last_rerank_status = st
            hits = out_all[0]
            if confidence:
                did_to_row = {str(self._meta.docids[int(pos)]): int(self._meta.rows[int(pos)])
                              for pos in p_pos}
                h_rows = [did_to_row.get(str(h.get("docid")), -1) for h in hits]
                self._attach_confidence(qtoks, hits, h_rows)
            return hits
        if dense:
            views = getattr(self._bm, "_views", None) or {}
            vname = "dense" if "dense" in views else (next(iter(views), None))
            if vname is not None:
                from .rerank import encode_texts, get_rerank_model

                v_meta = views[vname].get("meta") or {}
                mp = str(self._model_path or v_meta.get("model_path") or "")
                model, st = get_rerank_model(mp, device="auto")
                if model is not None:
                    qv = encode_texts(model, [q], sub_batch=1, device=st.get("device"))[0]
                    fused = self.fused_scores(qtoks, qv, vname, alpha=0.6)
                    order = _topk_idx(fused, k)
                    self.last_rerank_status = dict(st, mode="view_fuse", view=vname)
                    out, h_rows = [], []
                    for rank, i in enumerate(order, 1):
                        if fused[i] <= 0:
                            break
                        m = self._meta[int(i)]
                        item = dict(rank=rank, score=float(fused[i]), docid=m.get('docid'),
                                    file=m['file'], line=m['line'], text=m['text'])
                        if span:
                            self._attach_span_fields(item, q, m)
                        out.append(item)
                        h_rows.append(int(m['row']))
                    if confidence:
                        self._attach_confidence(qtoks, out, h_rows)
                    return out
                self.last_rerank_status = dict(st, mode="bm25_fallback")
            else:
                # 零编码模式（无预存视图）：dense=True 直接委托查询期池内重排
                return self.query(q, k=k, dense=False, rerank=True,
                                  pool_k=max(int(pool_k), int(k) * 10, 100),
                                  span=span, confidence=confidence)

        s = self._score_active(qtoks)     # 与 self._meta 对齐
        order = _topk_idx(s, k)
        out, h_rows = [], []
        for rank, i in enumerate(order, 1):
            if s[i] <= 0:
                break
            m = self._meta[int(i)]
            item = dict(rank=rank, score=float(s[i]), docid=m.get('docid'),
                        file=m['file'], line=m['line'], text=m['text'])
            if span:
                self._attach_span_fields(item, q, m)
            out.append(item)
            h_rows.append(int(m['row']))
        if confidence:
            self._attach_confidence(qtoks, out, h_rows)
        return out

    def query_many(self, qs, k=10, dense=False, with_text=True, *,
                   rerank=False, pool_k=100, span=False):
        r"""批量查询（0.3 升级：重排臂对整批候选去重并流式编码）。

        `with_text=False` 时命中项不带 `text`（省掉每命中一条拷一段全文）。
        """
        if self._bm is None:
            self._load()
        qs_list = list(qs)
        if not qs_list:
            return []
        if rerank:
            rr = self._resolve_reranker(rerank, pool_k)
            pk = max(int(rr.pool_k), int(k))
            pools, pscores = [], []
            for q in qs_list:
                p, ps = self._bm25_pool(q, pk)
                pools.append(p)
                pscores.append(ps)
            out_all, st = rr.rerank_pools(
                qs_list, pools, pscores, self._meta, bm=self._bm,
                k=k, with_text=with_text, with_span=span,
            )
            self.last_rerank_status = st
            return out_all
        if dense:
            views = getattr(self._bm, "_views", None) or {}
            if not views:
                return self.query_many(
                    qs_list, k=k, dense=False, with_text=with_text,
                    rerank=True, pool_k=max(int(pool_k), int(k) * 10, 100), span=span,
                )
            out_all = []
            for q in qs_list:
                rows_q = self.query(q, k=k, dense=True, span=span)
                if not with_text:
                    for item in rows_q:
                        item.pop('text', None)
                out_all.append(rows_q)
            return out_all
        out_all = []
        for q in qs_list:
            s = self._score_active(self._tok_fn()(q))
            order = _topk_idx(s, k)
            out = []
            for rank, i in enumerate(order, 1):
                if s[i] <= 0:
                    break
                m = self._meta[int(i)]
                d = dict(rank=rank, score=float(s[i]), docid=m.get('docid'),
                         file=m['file'], line=m['line'])
                if with_text:
                    d['text'] = m['text']
                if span:
                    self._attach_span_fields(d, q, m)
                out.append(d)
            out_all.append(out)
        return out_all

    # ---------- 浏览 ----------

    def browse(self, file=None, page=0, n=20):
        if self._meta is None:
            self._load()
        if file is not None:
            file_n = file.replace('\\', '/')
            # ⭐ X13：只读 file 的全表扫走紧凑列表，不物化 dict
            idxs = [i for i, f in enumerate(self._meta.files)
                    if f is not None and f.replace('\\', '/') == file_n]
        else:
            idxs = list(range(len(self._meta)))
        start = page * n
        return [dict(docid=self._meta[i].get('docid'), file=self._meta[i]['file'],
                     line=self._meta[i]['line'], text=self._meta[i]['text'])
                for i in idxs[start:start + n]], len(idxs)

    def list_files(self):
        if self._meta is None:
            self._load()
        return sorted(set(self._meta.files))          # ⭐ X13：files 是紧凑 str 列表

    # ---------- 真增量 update（方案 A） ----------

    def compact_segments(self, *, persist=True) -> dict:
        """把已挂段全部折进主表 ＋（可选）落盘 ＋ 清分块缓存。

        `api.Index.compact()` 与 `update()` 的 `fold_first` 出口**共用这一个实现**。
        `persist` 三态：

        · `True`：折叠 → `_persist()`（**先落盘**）→ 追加 `folded_all`（提交点）。
          反过来就是"先提交后落盘"：崩溃时旁车说已折叠、npz 里没有那些行 ⇒ 段内容静默丢失。
        · `False`（L1）：只折内存，**npz 与 `folded_all` 都不写** —— 盘上保持
          「旧主表 ＋ 段清单」的等价状态，重载按旁车重挂即可复原。
        · `"delta"`（L2/S4）：把折出来的行写成**一条 delta**，再追加 `folded_all`。
          顺序是 delta → meta/hashes → 事件：若在 delta 与事件之间崩，重载会重挂段，
          但此时 T 已因 delta 变长、`T.shape[0] > row_base` 会把重挂跳过 ⇒ 不重复计数。
        ⚠️ `False` 时两件事必须**同时**省：只省 npz 却写事件 ⇒ 重载以为已折、盘上却没有那些行。
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
                                    meta_rows=self._meta.entries_in_row_range(n0, _n1))
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

    def _reclaim_dead_rows(self) -> int:
        """物理清理 `_dead` 墓碑行并同步重映射 `_meta.rows` 与内存 `_views`（用于 Tier-2 full compact）。"""
        bm = self._bm
        if bm is None or bm.T is None or getattr(bm, "_dead", None) is None:
            return 0
        dead_arr = np.asarray(bm._dead[:bm.N], dtype=bool)
        dead_n = int(dead_arr.sum())
        if dead_n <= 0:
            return 0
        live_rows = np.flatnonzero(~dead_arr)
        old_n = int(bm.N)
        old_to_new = np.full(old_n, -1, dtype=np.int64)
        old_to_new[live_rows] = np.arange(live_rows.size, dtype=np.int64)

        T_csr = bm.T.tocsr()
        bm.T = T_csr[live_rows].tocsr()
        bm.dl = np.asarray(bm.dl[:old_n], dtype=np.float32)[live_rows]
        bm._dead = None
        bm.N = int(live_rows.size)
        bm._Tbuf = None
        bm._tc_blocks, bm._tc_rows = [], 0
        bm._invalidate_w()
        bm._recompute_stats()

        for v in (getattr(bm, "_views", None) or {}).values():
            v_rows = np.asarray(v["rows"], dtype=np.int64)
            in_bounds = (v_rows >= 0) & (v_rows < old_to_new.size)
            safe_idx = np.clip(v_rows, 0, max(old_to_new.size - 1, 0))
            mapped = np.where(in_bounds, old_to_new[safe_idx], -1) if old_to_new.size else np.full(v_rows.shape, -1, dtype=np.int64)
            keep = mapped >= 0
            new_r = mapped[keep]
            v["rows"] = new_r
            v["vectors"] = v["vectors"][keep]
            v["vn"] = v["vn"][keep]
            cov = np.zeros(bm.N, dtype=bool)
            if new_r.size:
                cov[new_r] = True
            v["coverage"] = cov

        if self._meta is not None and len(self._meta):
            mr = np.asarray(self._meta.rows, dtype=np.int64)
            ok = (mr >= 0) & (mr < old_to_new.size)
            mapped = np.full(mr.shape, -1, dtype=np.int64)
            mapped[ok] = old_to_new[mr[ok]]
            keep_pos = np.flatnonzero(mapped >= 0)
            self._meta.bulk_text_load()
            if keep_pos.size != len(self._meta):
                self._meta = self._meta._take(keep_pos)
            self._meta.rows = mapped[keep_pos]
            self._rows = None
            self._rows_ident = None
        return dead_n

    def compact(self, mode: str = "auto") -> dict:
        """统一自治压实（0.4 P1）：支持 `'auto'` / `'delta'` / `'full'` / `'segments'`。

        - `'segments'`: 仅折叠已挂段并全量落盘（兼容旧行为）；
        - `'delta'` (Tier-1): 把已挂段折为 delta（若有）并合并多个 `.delta/*.npz` 为单个，不重写基线 `bm25.npz`；
        - `'full'` (Tier-2): 折叠已挂段 ＋ 物理回收 `_dead` 墓碑 ＋ 双缓冲落盘重置（`dead=0`、`blocks=1`、清 `.delta/`）；
        - `'auto'`: 根据 `dead_ratio >= 0.2` / `big_delta` / 已挂段 / `delta_files >= 2` 自动选 Tier-2 / 段折叠 / Tier-1。
        """
        if mode not in ("auto", "delta", "full", "segments"):
            raise ValueError(f"compact mode 只接受 'auto'/'delta'/'full'/'segments'，实得 {mode!r}")
        if self._bm is None:
            self._load()
        bm = self._bm
        segs = list(bm.segment_report())
        dead_n = int(bm._dead[:bm.N].sum()) if getattr(bm, "_dead", None) is not None else 0
        n_total = int(getattr(bm, "N", 0) or 0)
        dead_ratio = (dead_n / float(n_total)) if n_total > 0 else 0.0
        d_files = delta_files(self.dir)
        delta_cnt = len(d_files)
        base_nnz = max((int(bm.T.nnz) if bm.T is not None else 0) - int(self._delta_nnz), 1)
        big_delta = int(self._delta_nnz) > DELTA_COMPACT_RATIO * base_nnz

        if mode == "segments":
            out = self.compact_segments(persist=True)
            out.update(mode="segments", reclaimed_dead=0, merged_deltas=0,
                       delta_files=len(delta_files(self.dir)))
            return out

        if mode == "auto":
            if dead_ratio >= 0.2 or big_delta:
                mode = "full"
            elif segs:
                out = self.compact_segments(persist=True)
                out.update(mode="full", reclaimed_dead=0, merged_deltas=delta_cnt,
                           delta_files=len(delta_files(self.dir)))
                return out
            elif delta_cnt >= 2:
                mode = "delta"
            else:
                return dict(
                    mode="noop", folded=[], tags=[],
                    blocks=len(getattr(bm, "_tc_blocks", []) or []),
                    N=int(bm.N), V=int(bm.V), ms=0.0, via=None,
                    reclaimed_dead=0, merged_deltas=0, delta_files=delta_cnt,
                )

        t0 = time.time()
        if mode == "delta":
            seg_out = self.compact_segments(persist="delta") if segs else None
            m_out = self._merge_deltas()
            merged_n = int(m_out["merged"]) if m_out else 0
            return dict(
                mode="delta",
                folded=(seg_out["folded"] if seg_out else []),
                tags=(seg_out["tags"] if seg_out else []),
                blocks=len(getattr(bm, "_tc_blocks", []) or []),
                N=int(bm.N), V=int(bm.V),
                ms=(time.time() - t0) * 1000.0,
                via=(seg_out["via"] if seg_out else ("merge_deltas" if merged_n else "noop")),
                reclaimed_dead=0,
                merged_deltas=merged_n,
                delta_files=len(delta_files(self.dir)),
            )

        # mode == "full"
        seg_out = self.compact_segments(persist=False) if segs else None
        reclaimed = self._reclaim_dead_rows()
        self._persist(meta=True)
        if seg_out and seg_out.get("tags"):
            manifest_append(self.dir, {
                "event": "folded_all",
                "tags": seg_out["tags"],
                "via": f"compact/{seg_out['via']}",
            })
        if bm.T is not None:
            bm._ensure_tc()
        return dict(
            mode="full",
            folded=(seg_out["folded"] if seg_out else []),
            tags=(seg_out["tags"] if seg_out else []),
            blocks=len(getattr(bm, "_tc_blocks", []) or []),
            N=int(bm.N), V=int(bm.V),
            ms=(time.time() - t0) * 1000.0,
            via=(seg_out["via"] if seg_out else ("reclaim_dead" if reclaimed else "persist")),
            reclaimed_dead=reclaimed,
            merged_deltas=delta_cnt,
            delta_files=len(delta_files(self.dir)),
        )

    def update(self, root, workers=0, *, verify="auto", skip_dirs=None):
        """hash diff ＋ 路由（`primolix.routing.route_update`）。

        四个出口，返回里的 `kind` 与之一一对应：
            noop         无改动
            incremental  词表内 → 就地真增量
            staged       含新词 → 挂只读段（`O(新 nnz)`，⛔ 不重建）
            rebuild      兜底：某条 precondition 不满足（dense / 段数上限 / 批太大）
        ⛔ 判不了或**安全执行不了**（例如段在场却只能重建）→ 抛 `RuntimeError`，
        ⛔ 不静默选一条。判据见 `04c_丙档OOV路由_设计与判据_2026-11-08.md`。

        `verify`（0.2.2）：
            `"auto"`（默认）—— 有 `(size, mtime_ns)` 印记就走 `"stat"`，否则退回 `"full"`；
            `"stat"`        —— 只比元数据，没变就不读内容（实测 ~1/10 成本）；
            `"full"`        —— 每个文件读内容算 md5（保留 mtime 的还原类工具必须用这档）。
        返回里带 `verify`，**如实说明这一批到底校验到了什么程度**。
        """
        if self._bm is None:
            self._load()
        self._dense = self._dense or ("dense" in (getattr(self._bm, "_views", None) or {}))
        t0 = time.time()
        mode = verify
        if mode == "auto":
            mode = "stat" if getattr(self, "_stamps", None) else "full"
        if mode not in ("full", "stat"):
            raise ValueError(f"verify 只接受 'auto' / 'full' / 'stat'，实得 {verify!r}")
        # 排除索引输出目录自身 (可能在 root 内部, 如 <root>/.idx) 及调用方指定的 skip_dirs
        skips = [self.dir, *(skip_dirs or [])]
        files, stamps_now = scan_root_ex(Path(root), skips, known=(self._hashes or {}),
                                         stamps=getattr(self, "_stamps", None) or {},
                                         verify=mode, with_spans=True, with_ctx=True)
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
        new_index_texts = []
        append_chunks = []
        for rel in added + changed:
            f = rel2scan.get(rel)
            if f is None:
                continue
            h = f[1]
            for item in f[2]:
                t, _, ln, c0, c1 = item[:5]
                ctx = item[5] if len(item) > 5 else ""
                new_index_texts.append(f"{ctx}\n{t}" if ctx else t)
                append_chunks.append(dict(docid=f'{h[:8]}:{len(append_chunks)}',
                                          file=rel, line=ln,
                                          char_start=c0, char_end=c1, text=t))
        tok_fn = self._tok_fn()
        chunk_tokens = [tok_fn(t) for t in new_index_texts]

        # ---- 墓碑/保留：整文件级替换（removed + changed 的旧 chunk 全剔除） ----
        # ⭐ X14：`filter_by_file` 一次 isin 同时给保留集与墓碑位置（旧版两趟 O(N) Python 循环）
        drop_rels = set(removed) | set(changed)
        keep_meta, _tomb_pos = self._meta.filter_by_file(drop_rels)
        tomb_rows = self._meta.rows[_tomb_pos].tolist()
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
            dense_on=False,
            batch_nnz=int(sum(len(t) for t in chunk_tokens)),
            total_nnz=int(getattr(self._bm, "nnz", 0)),
        )
        _say(f"[路由] {route['route']}（{route['code']}）：{route['detail']}")
        fold_note = ""
        cinfo = None
        if route['route'] == FOLD_FIRST:
            # A / C：当场折回主表（⛔ 不再让调用方自己去 compact）。
            # ⭐ L1（2026-11-09）：后续分支自己会全量落盘 —— `incremental` 收尾调 `_persist()`、
            #    `rebuild` 直接 `rebuild_in_place` —— 所以折叠**不必再写一次盘**（原先会连写两次）。
            # ⚠️ 例外：`staged` 分支只写 meta/hashes、**不写 npz**；若折叠也不落盘，
            #    meta.jsonl 里的行数会超过盘上主表行数 ⇒ 重载时 `scores[rows]` 越界。
            #    所以 `then == staged` 时折叠照旧落盘。
            # ⭐ 折叠的落盘方式按持久化模式分档（L2/S4）：
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
            self.rebuild_in_place(root, workers=workers, skip_dirs=skip_dirs)
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
            tag = _unique_seg_tag(self.dir)   # ⭐ X14：秒级 tag 在同秒二次发布时会撞已 mmap 的段文件
            if tomb_rows:
                self._bm._tombstone_rows(tomb_rows)
            self._meta = keep_meta
            rows = [(c['docid'], toks, c['file'], c['line'], c['text'],
                     c.get('char_start', -1), c.get('char_end', -1))
                    for c, toks in zip(append_chunks, chunk_tokens)]
            out = publish_segment(self, rows, tag=tag, generation=tag,
                                  base_V=base_V, base_T_rows=base_T_rows,
                                  new_terms=new_terms, new_pos=new_pos)
            self._rows = None          # ⭐ 失效行号缓存（段内文档已并入 _meta）
            self._hashes = new
            self._stamps = {rel: stamps_now[rel] for rel in new if rel in stamps_now}
            # ⚠️ 段在场 ⇒ `_persist()` 里的 `bm.save()` 会抛错：只落 meta(主表行)＋hashes。
            _hev = [("del", rel) for rel in removed]           # ⭐ X14：哈希账本同样追加化
            _hev += [("set", rel, new[rel], stamps_now.get(rel))
                     for rel in list(changed) + list(added)]
            self._persist_under_segments(del_rows=tomb_rows, events=_hev)
            if self._dense:
                self._sync_dense_view_incremental(append_chunks)
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
        self._meta = keep_meta.with_appended(append_chunks)   # ⭐ X13：数组化载体的列表手术等价
        self._rows = None          # ⭐ 失效行号缓存（行号已迁移/追加）
        self._hashes = new
        self._stamps = {rel: stamps_now[rel] for rel in new if rel in stamps_now}
        # ⭐ X14：哈希账本追加化 —— 只记 touched（removed/changed/added），不再整份重写
        _hev = [("del", rel) for rel in removed]
        _hev += [("set", rel, new[rel], stamps_now.get(rel))
                 for rel in list(changed) + list(added)]
        # ⭐ 持久化分档（L2）：delta 模式只写 O(新 nnz)，npz 模式维持原行为
        if PERSIST_MODE == "delta":
            dinfo = self._persist_delta(new_rows, tomb_rows, meta_rows=append_chunks)
            self._meta_append_pending(del_rows=tomb_rows)   # ⭐ 追加式：O(新增行)，不重写整份
            self._write_hashes(events=_hev)
            persist_note = f"delta#{dinfo['seq']}({dinfo['nnz']} nnz)"
            cauto = self._maybe_compact_delta()          # L2/S3：两层压实
            if cauto:
                persist_note += f"+压实{cauto.get('tier', '?')}"
        else:
            self._persist()      # 走到这里 ⇒ 段已不在场（要折的已在上面折过）
            persist_note = "npz"
        # 0.3：dense 已视图化（docid 键控），增量只需编码新 chunk，不重编码全库
        if self._dense:
            self._sync_dense_view_incremental(append_chunks)
        _say(f"S4 真增量完成 {time.time() - t0:.0f}s "
              f"(墓碑 {len(tomb_rows)} 行, 追加 {len(append_chunks)} chunk, "
              f"活跃 {len(self._meta)})")
        return {'kind': 'incremental',
                'detail': f'tomb={len(tomb_rows)} append={len(append_chunks)} '
                          f'{fold_note} {persist_note}'.strip(),
                'folded_tags': len(cinfo['tags']) if cinfo else 0,
                'verify': mode}

    def rebuild_in_place(self, root, workers=0, *, force=False, skip_dirs=None):
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
        stream_mode = getattr(self, "_stream", False)
        code_aware = (getattr(self._bm, "tokenizer_mode", "zh_default") == "zh_code")
        self.close()
        z = Primolix.build(root, str(self.dir), dense=self._dense,
                        model_path=self._model_path, workers=workers,
                        skip_dirs=[self.dir, *(skip_dirs or [])], stream=stream_mode, code_aware=code_aware)
        manifest_append(self.dir, {"event": "reset", "reason": "rebuild_in_place"})
        self._bm = z._bm
        self._meta = z._meta
        self._rows = None          # ⭐ 失效行号缓存
        self._rows_ident = None
        self._hashes = z._hashes
        self._stamps = z._stamps        # 重建后 size/mtime 印记一并换新，下一次可走 stat 快扫
        self._vecs = z._vecs
        # 同步持久化游标：否则同进程重建后走 delta 增量会重复写词表或漏写 meta
        self._persisted_V = int(z._persisted_V)
        self._persisted_N = int(z._persisted_N)
        self._meta_dels = int(z._meta_dels)
        self._delta_nnz = int(z._delta_nnz)
        self._delta_count = int(z._delta_count)
        self._hash_events_n = int(z._hash_events_n)
        self.last_rerank_status = z.last_rerank_status

    def unmount_segment(self, tag: str) -> dict:
        """撤回已挂段并同步清理旁车 manifest 事件与 `_meta` 段文档身份。"""
        if self._bm is None:
            self._load()
        res = self._bm.unmount_segment(tag)
        old_meta_n = len(self._meta) if self._meta is not None else 0
        if self._meta is not None:
            self._meta.close()
            self._meta = self._meta.truncate_rows_lt(int(res["N"]))
            self._rows = None
            self._rows_ident = None
        manifest_append(self.dir, {
            "event": "unmount",
            "tag": str(tag),
            "N": int(res["N"]),
            "V": int(res["V"]),
        })
        out = dict(res)
        out["meta_removed"] = max(0, old_meta_n - (len(self._meta) if self._meta is not None else 0))
        return out

    def close(self) -> None:
        """显式释放 `_meta`、`_bm` 及可选 `dense_int8` 侧车内存。"""
        if getattr(self, "_d8_sidecar", None) is not None:
            self._d8_sidecar.clear()
        if getattr(self, "_meta", None) is not None:
            self._meta.close()
        if getattr(self, "_bm", None) is not None:
            self._bm.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

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
        rows = self._meta.rows[pos]           # ⭐ X13：数组直达（物化 1M 个 dict 的旧写法已废）
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
        hit = np.isin(self._meta.rows, rows)      # ⭐ X13：数组直达
        cov[hit] = True
        return cov

    def fused_scores(self, qtoks, qvec, view_name, alpha=0.35):
        """与 `_meta` 对齐的融合分数（视图不在场 → 逐位 `_score_active`）。"""
        if self._bm is None:
            self._load()
        base = self._bm.fused_scores(qtoks, qvec, view_name, alpha=alpha)
        if len(self._meta) == 0:
            return base
        rows = self._meta.rows                     # ⭐ X13：数组直达
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
        _say(f"[views] 视图 {name} 落盘 {path}（{V.shape[0]} 条 × {V.shape[1]} 维）")
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
        for i, did in enumerate(self._meta.docids):     # ⭐ X13：数组直达（chunk_id 兜底废止，见 _read_meta）
            if did is not None and did not in pos_of:
                pos_of[did] = i
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

    # ---- M1（契约 §1.1）：rerank 缓存视图的落盘（活行 + LRU 封顶）与显式热区预热 ----

    def persist_view(self, name="rerank_cache", *, max_vectors=None, extra_meta=None) -> dict:
        """把内存视图按 docid ↔ 向量写回 `<dir>/views/<name>.npz`（GC：只写活行 + LRU 封顶）。

        返回 `{"persisted", "path", "vectors", "dim", "dead_dropped", "capped_dropped",
        "ms", "reason"}`。视图不在场 → `persisted=False, reason="view_absent"`；
        墓碑行 / 已不在 `_meta` 里的滞后行一律不写（计入 `dead_dropped`）。
        """
        t0 = time.time()
        name = str(name)
        base = dict(persisted=False, path=None, vectors=0, dim=0, dead_dropped=0,
                    capped_dropped=0, ms=0.0, reason=None)
        if self._bm is None:
            self._load()
        views = getattr(self._bm, "_views", None) or {}
        view = views.get(name)
        if not isinstance(view, dict):
            base.update(reason="view_absent", ms=round((time.time() - t0) * 1000.0, 3))
            return base
        old_meta = dict(view.get("meta") or {})
        rows = np.asarray(view.get("rows"), dtype=np.int64)
        vecs_raw = view.get("vn")
        if vecs_raw is None:
            vecs_raw = view.get("vectors")
        vecs = np.asarray(vecs_raw, dtype=np.float32)
        if rows.ndim != 1 or vecs.ndim != 2 or rows.shape[0] != vecs.shape[0]:
            base.update(reason="view_malformed", ms=round((time.time() - t0) * 1000.0, 3))
            return base
        dim0 = int(vecs.shape[1]) if vecs.ndim == 2 else int(view.get("dim") or 0)
        n_live = int(getattr(self._bm, "N", 0) or 0)
        dead = getattr(self._bm, "_dead", None)
        valid = (rows >= 0) & (rows < n_live)
        if dead is not None and rows.size:
            in_range = valid.copy()
            if in_range.any():
                valid[in_range] = ~np.asarray(dead[rows[in_range]], dtype=bool)
        # 物理行 → meta 位置（_meta.rows 不保证有序，故排序后 searchsorted）
        meta_rows = np.asarray(self._meta.rows, dtype=np.int64)
        pos = np.full(rows.size, -1, dtype=np.int64)
        if rows.size and meta_rows.size:
            order = np.argsort(meta_rows, kind="stable")
            sorted_rows = meta_rows[order]
            idx = np.searchsorted(sorted_rows, rows)
            idx_c = np.minimum(idx, sorted_rows.size - 1)
            hit = sorted_rows[idx_c] == rows
            pos[hit] = order[idx_c[hit]]
        matched = valid & (pos >= 0)
        dead_dropped = int((~matched).sum())        # 墓碑 / 越界 / 已不在 meta（视图滞后）
        live_idx = np.flatnonzero(matched)
        cap = self._view_lru_cap(max_vectors)
        # LRU：读到 lru_order 就按「least → most」裁剪；否则按 docid 确定性排序截断
        raw_lru = old_meta.get("lru_order")
        row_to_idx = {int(rows[i]): int(i) for i in live_idx.tolist()}
        kept_idx: list[int] = []
        if isinstance(raw_lru, (list, tuple)) and row_to_idx:
            lru_rows: list[int] = []
            seen_rows = set()
            for r in raw_lru:
                try:
                    ri = int(r)
                except (TypeError, ValueError):
                    continue
                if ri in row_to_idx and ri not in seen_rows:
                    seen_rows.add(ri)
                    lru_rows.append(ri)
            for ri in sorted(row_to_idx.keys()):    # 未记录行按「更旧」处理，避免确定性丢失
                if ri not in seen_rows:
                    seen_rows.add(ri)
                    lru_rows.append(ri)
            over = max(0, len(lru_rows) - cap)
            capped_dropped = int(over)
            kept_rows = lru_rows[over:]             # 保留尾部 = 最近活跃 max_vectors 条
            kept_idx = [row_to_idx[ri] for ri in kept_rows]
        else:
            order_docid = sorted(live_idx.tolist(),
                                 key=lambda i: str(self._meta.docids[int(pos[i])]))
            capped_dropped = max(0, len(order_docid) - cap)
            kept_idx = order_docid[:cap]
        docids_out = [str(self._meta.docids[int(pos[i])]) for i in kept_idx]
        vecs_out = (vecs[np.asarray(kept_idx, dtype=np.int64)]
                    if kept_idx else np.zeros((0, dim0), dtype=np.float32))
        meta_out = {
            "model_id": str(old_meta.get("model_id") or ""),
            "tokenizer_mode": str(old_meta.get("tokenizer_mode")
                                  or getattr(self._bm, "tokenizer_mode", "zh_default")),
            "chunk_version": str(old_meta.get("chunk_version") or CHUNK_VERSION),
            "progressive": bool(old_meta.get("progressive", True)),
            "persisted_at": float(time.time()),
            # docid 口径的 LRU（物理行重启后失效 → 不落物理 lru_order，避免误当有效代次）
            "lru_docids": docids_out,
        }
        if isinstance(extra_meta, dict):
            meta_out.update(extra_meta)
        try:
            path = self.save_view(name, docids_out, vecs_out, meta=meta_out)
        except Exception as exc:  # noqa: BLE001  落盘失败如实回传（⛔ 不假装已持久化）
            base.update(reason=f"save_failed: {type(exc).__name__}: {exc}",
                        dead_dropped=int(dead_dropped),
                        capped_dropped=int(capped_dropped),
                        ms=round((time.time() - t0) * 1000.0, 3))
            return base
        return dict(persisted=True, path=str(path), vectors=int(len(docids_out)),
                    dim=int(vecs_out.shape[1]) if vecs_out.size else dim0,
                    dead_dropped=int(dead_dropped), capped_dropped=int(capped_dropped),
                    ms=round((time.time() - t0) * 1000.0, 3), reason=None)

    @staticmethod
    def _view_lru_cap(max_vectors=None) -> int:
        """LRU 封顶：复用 rerank 的环境变量解析（两处默认值不许漂移）。"""
        try:
            from .rerank import _cache_max_vectors

            return int(_cache_max_vectors(max_vectors))
        except Exception:  # noqa: BLE001
            try:
                return max(1, int(max_vectors)) if max_vectors else 4096
            except (TypeError, ValueError):
                return 4096

    def warm_view(self, files, reranker, *, view_name="rerank_cache",
                  max_chunks=512) -> dict:
        """显式热区预热：只对 `files` 命中的 chunk 编码并入视图（⛔ 不扫全仓、无后台线程）。

        编码器不可用 → `available=False / encoded=0`，`reason` 如实回传（绝不抛）。
        """
        t0 = time.time()
        view = str(view_name)
        out = dict(mode="warm", available=False, files=0, chunks=0, encoded=0,
                   ms=0.0, view=view, reason=None)
        if self._bm is None:
            self._load()
        try:
            if reranker is None:
                out.update(reason="reranker_absent",
                           ms=round((time.time() - t0) * 1000.0, 3))
                return out
            rep = reranker.warm_files(self._meta, files, bm=self._bm,
                                      max_chunks=int(max_chunks), view_name=view)
            if isinstance(rep, dict):
                for key in ("available", "files", "chunks", "encoded", "ms", "reason"):
                    if key in rep:
                        out[key] = rep[key]
                out["view"] = str(rep.get("view") or view)
        except Exception as exc:  # noqa: BLE001  预热失败只如实回传，不打乱调用方
            out.update(reason=f"{type(exc).__name__}: {exc}",
                       ms=round((time.time() - t0) * 1000.0, 3))
        return out


# ================================================================ 临时段发布：旁车

SEG_SUBDIR = ".segments"
SEG_MANIFEST = "manifest.jsonl"


def _unique_seg_tag(index_dir, base=None) -> str:
    r"""段 tag 唯一化（X14，2026-11-18 修）。

    ⛔ 事故形态：`%Y%m%d-%H%M%S` 是**秒级**的 —— 同一进程在同一秒内做两次 staged
    发布（小语料/脚本化灌入完全做得到）⇒ 两次拿到**同一个 tag** ⇒ 第二次去
    `np.save` **已被 `mount_segment` mmap 的段文件** ⇒ Windows 拒绝改写活动映射
    ⇒ `OSError(22)`，且看起来像"路径非法"（本测试第一版就是被它咬的）。
    修法：目录已存在就顺序加 `-2/-3/…` 后缀；manifest 的唯一性语义不变。
    跨进程本就带 PID ⇒ 不会撞；同进程的 TOCTOU 窗口在此形态下不存在（串行发布）。
    """
    base = base or (time.strftime("%Y%m%d-%H%M%S") + f"-{os.getpid()}")
    tag, k = base, 1
    while (Path(index_dir) / SEG_SUBDIR / tag).exists():
        k += 1
        tag = f"{base}-{k}"
    return tag
# ⭐ 段级文档身份侧车（乙档）：一行一篇 `{docid,row,file,line,text}`，与段同生共死。
#    为什么不塞 `meta.jsonl`：那份是 `"w"` 整份重写，与「提交点＝追旁车」的崩溃语义打架
#    （写到一半崩 ⇒ 历史身份全丢）。侧车是 append-only 的，且**先于** manifest 提交写好。
SEG_DOCS = "docs.jsonl"

# ================================================================ L2：增量持久化开关
# "npz"     ＝每次 `_persist()` 重写整份 npz（0.2.1 及以前的行为）；
# "delta"   ＝平时只写 `delta/<seq>.npz`（O(新 nnz)），`_persist()` 只在压实时发生。
# ⭐ 0.2.4 起默认 "delta"（用户裁定 2026-11-18）。翻默认的组件表（全部 [实测] @1M）：
#   一次增量 update ≈0.5–0.7s（大头 stat 扫）· tier-1 压实 197ms · tier-2 全量 2.51s
#   回放 0.31s/delta 文件（⇒ MAX_FILES=16 最坏冷启罚 ≈5s）· `info['load_report']` 可观测。
PERSIST_MODE = "delta"
DELTA_SUBDIR = ".delta"
DELTA_COMPACT_RATIO = 0.3        # `[提案]`：Σdelta.nnz > 0.3 × base.nnz 就该压实
DELTA_MAX_FILES = 16             # ⭐ 0.2.4：32→16（回放罚 0.31s/文件 ⇒ 16 最坏 ≈5s 冷启）
HASHES_COMPACT_N = 4096          # ⭐ X14：hashes 事件攒过这条数就快照整理（同 meta 的 4096 先例）


def set_persist_mode(mode: str) -> str:
    """切换持久化模式（进程级）。返回切换后的值。"""
    global PERSIST_MODE
    if mode not in ("npz", "delta"):
        raise ValueError(f"PERSIST_MODE 只接受 'npz' / 'delta'，实得 {mode!r}")
    PERSIST_MODE = mode
    return PERSIST_MODE


def delta_files(index_dir) -> list:
    """按 seq 升序返回**当前生效**的 `<index>/.delta/*.npz`。

    · ⛔ 忽略 `.tmp`：未 rename ＝ 未提交。
    · ⭐ **合并**（2026-11-10）：一个 delta 可以在自己的 `supersedes` 里声明"我取代了哪些 seq"。
      被声明的文件即使还在盘上（删除失败或崩在删除之前）也**不进生效集** ⇒
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
        elif kind == "unmount":
            utag = ev.get("tag")
            active = [a for a in active if a.get("tag") != utag]
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

    ⭐ `Stage.publish()`（门面的旁路累积）与 `Primolix.update()` 的 `staged` 分支
    **共用这一个实现** —— 段的提交顺序只有一处，⛔ 不许各写一份。

    顺序（既有不变量，⛔ 不许调换）：
        三个 `.npy` → `dl.npy` → 文档身份侧车 → `manifest` 追加（**提交点**）
        → `mount_segment` → `_absorb_staged`（段内文档对产品层可见）。

    ⚠️ **调用方须已持写锁**（`Index.add` / `Stage.publish` 都在锁内）。
    ⚠️ 段在场时 `SparseBM25.save()` 会抛错 ⇒ 调用方**不能**顺手 `_persist()`。
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
    for i, row in enumerate(rows):
        toks = row[1]
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
    dl = np.asarray([len(row[1]) for row in rows], np.float32)
    np.save(seg_dir / "dl.npy", dl)
    # 乙档：文档身份侧车先写好（沿用「段文件 → 侧车 → manifest 提交 → 挂段」）
    row_base = int(bm.N)
    with open(seg_dir / SEG_DOCS, "w", encoding="utf-8", newline="\n") as _fh:
        for i, row in enumerate(rows):
            _docid, _t, _f, _l, _x = row[:5]
            _d = str(_docid)
            rec_doc = dict(docid=_d, row=row_base + i,
                           file=(_f or _d), line=int(_l or 1), text=_x)
            if len(row) >= 7 and int(row[5]) >= 0:
                rec_doc["char_start"] = int(row[5])
                rec_doc["char_end"] = int(row[6])
            _fh.write(json.dumps(rec_doc, ensure_ascii=False) + "\n")
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

