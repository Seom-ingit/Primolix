#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""primolix.scope —— 轻量化 Lineage 拓扑与范围路由器（支线 #15）。

把 `lineage.py` 的核心结构思想（目录族/文件系拓扑、`marriage` 跨模块联姻边、
`hot/cold` 策略权重、`update_prioritized(budget)` 削峰）桥接到 `Primolix` 的
通用代码/文档文件树上，不引入另一套只认 `.jsonl` 的 BM25 存储。
"""
from __future__ import annotations

import os
import re
from collections import defaultdict
from pathlib import Path

from .core import CODE_EXT, PROSE_EXT, iter_files

_PY_IMPORT_RE = re.compile(
    r"^\s*(?:from\s+([A-Za-z0-9_.]+)\s+import|import\s+([A-Za-z0-9_.]+))",
    re.MULTILINE,
)
_DOC_REF_RE = re.compile(r"([A-Za-z0-9_./-]+\.(?:py|md|ts|tsx|js|jsx|go|rs))")
_COLD_DIRS = {"vendor", "third_party", "archive", "archives", "legacy", "_raw", "backup"}


class ScopeGraph:
    """面向代码/文档仓库的轻量血缘拓扑图（家族树 + `marriage` 联姻边 + `hot/cold` 调度）。"""

    def __init__(self, root: str | os.PathLike[str], files: list[str] | None = None,
                 cold_dirs: set[str] | None = None):
        self.root = Path(root)
        self.cold_dirs = set(cold_dirs) if cold_dirs is not None else set(_COLD_DIRS)
        self.files: list[str] = []
        self.dir_to_files: dict[str, list[str]] = defaultdict(list)
        self.file_to_dir: dict[str, str] = {}
        self.marriage: dict[str, set[str]] = defaultdict(set)  # node/file <-> linked files/dirs
        self.strategy: dict[str, str] = {}
        self.refresh(files=files)

    @staticmethod
    def _norm(p: str) -> str:
        return str(p).replace("\\", "/").lstrip("./")

    def refresh(self, files: list[str] | None = None) -> None:
        if files is None:
            scanned = iter_files(self.root)
            rel_files = [self._norm(rel) for _, rel in scanned
                         if Path(rel).suffix.lower() in (CODE_EXT | PROSE_EXT)]
        else:
            rel_files = [self._norm(f) for f in files]
        self.files = sorted(set(rel_files))
        self.dir_to_files.clear()
        self.file_to_dir.clear()
        self.marriage.clear()
        self.strategy.clear()

        by_stem: dict[str, list[str]] = defaultdict(list)
        for rel in self.files:
            parent = str(Path(rel).parent).replace("\\", "/")
            if parent == ".":
                parent = "<root>"
            self.dir_to_files[parent].append(rel)
            self.file_to_dir[rel] = parent
            parts = set(rel.split("/"))
            self.strategy[rel] = "cold" if (parts & self.cold_dirs) else "hot"
            by_stem[Path(rel).stem.lower()].append(rel)

        # 自动发现轻量 `marriage` 联姻边：Python import 与文档对源码的路径引用
        file_set = set(self.files)
        for rel in self.files:
            fp = self.root / rel
            if not fp.is_file():
                continue
            try:
                text = fp.read_text(encoding="utf-8", errors="ignore")[:32768]
            except OSError:
                continue
            ext = fp.suffix.lower()
            if ext == ".py":
                pkg_dir = self.file_to_dir[rel]
                for m in _PY_IMPORT_RE.finditer(text):
                    mod = (m.group(1) or m.group(2) or "").strip()
                    if not mod:
                        continue
                    if mod.startswith("."):
                        mod_name = mod.lstrip(".").split(".")[-1].lower()
                        cand_dir = pkg_dir
                    else:
                        mod_name = mod.split(".")[-1].lower()
                        cand_dir = None
                    for target in by_stem.get(mod_name, ()):
                        if target == rel:
                            continue
                        if cand_dir is None or self.file_to_dir.get(target) == cand_dir:
                            self.marriage[rel].add(target)
                            self.marriage[target].add(rel)
            elif ext in PROSE_EXT:
                for m in _DOC_REF_RE.finditer(text):
                    ref = self._norm(m.group(1))
                    if ref in file_set and ref != rel:
                        self.marriage[rel].add(ref)
                        self.marriage[ref].add(rel)
                    else:
                        stem = Path(ref).stem.lower()
                        for target in by_stem.get(stem, ()):
                            if target != rel and target.endswith(ref):
                                self.marriage[rel].add(target)
                                self.marriage[target].add(rel)

    def expand_scope(self, scope: str | None, *, include_marriage: bool = True) -> set[str]:
        """将目录前缀或文件精确展开为允许命中的相对路径集合；`scope=None` 返回全集。"""
        if not scope:
            return set(self.files)
        sc = self._norm(scope).rstrip("/")
        if sc == "<root>":
            direct = set(self.dir_to_files.get("<root>", ()))
        elif sc in set(self.files):
            direct = {sc}
        else:
            prefix = sc + "/"
            direct = {f for f in self.files if f == sc or f.startswith(prefix)}
        if not include_marriage or not direct:
            return direct
        out = set(direct)
        for f in list(direct):
            out.update(self.marriage.get(f, ()))
        return out

    def filter_hits(self, hits: list[dict], scope: str | None, *,
                    include_marriage: bool = True, limit: int | None = None) -> list[dict]:
        """按 Scope 血缘子图过滤检索结果并保留原始排序（重排后也可做范围裁剪）。"""
        if not scope:
            return hits[:limit] if limit is not None else hits
        allowed = self.expand_scope(scope, include_marriage=include_marriage)
        filtered = []
        for h in hits:
            rel = self._norm(h.get("file", ""))
            if rel in allowed:
                item = dict(h)
                item["rank"] = len(filtered) + 1
                filtered.append(item)
                if limit is not None and len(filtered) >= limit:
                    break
        return filtered

    def prioritize_updates(self, dirty_files: list[str], *,
                           w_hot: float = 1.0, w_cold: float = 0.3,
                           w_marriage: float = 0.1, budget: int | None = None) -> list[str]:
        """按 `lineage` 的 hot/cold 策略 ＋ 联姻度数降序排定后台更新顺序（低预算削峰）。"""
        scored = []
        for rel in dirty_files:
            nr = self._norm(rel)
            base = w_hot if self.strategy.get(nr, "hot") == "hot" else w_cold
            deg = len(self.marriage.get(nr, ()))
            scored.append((-(base + w_marriage * deg), nr))
        scored.sort()
        out = [nr for _w, nr in scored]
        return out[:budget] if budget is not None else out

    def report(self) -> dict:
        n_edges = sum(len(v) for v in self.marriage.values()) // 2
        n_hot = sum(1 for v in self.strategy.values() if v == "hot")
        return dict(
            root=str(self.root),
            files=len(self.files),
            dirs=len(self.dir_to_files),
            hot_files=n_hot,
            cold_files=len(self.files) - n_hot,
            marriage_edges=n_edges,
        )
