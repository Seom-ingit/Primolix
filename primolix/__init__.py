#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""primolix —— 零编码稀疏检索内核 + 文件夹检索工具（独立可发布包）。

门面 `primolix.build` / `primolix.open` → `api.Index`（search/add/tune/explain/compact/stats，读者并发·写者独占）。
内核 `kernel.SparseBM25`（裸用无并发保护）、`core.Primolix`；CLI `python -m primolix <index|query|update|browse|info|repl>`。
零编码：默认无 GPU/torch（BM25 + 段落 chunk），dense 精排可选、未装自动降级 BM25-only。
用法：`idx = primolix.build('demo_docs', 'demo_docs/.idx')` → `idx.search('什么是检索', k=10)` / `idx.add(...)` / `idx.tune(k1=2.0, b=0.3)`。
"""
from .kernel import (SparseBM25, zh_tokenize, simple_tokenize,
                     tokenizer_status, log)
from .core import Primolix
from .api import Index, RWLock, build_index, open_index

__version__ = "0.1.0"

# `open` 会遮蔽内置 `open` → 保留显式 `primolix.open(...)`，但不进 `__all__`。
open = open_index
build = build_index

__all__ = ["SparseBM25", "Primolix", "Index", "RWLock",
           "build", "build_index", "open_index",
           "zh_tokenize", "simple_tokenize", "tokenizer_status", "log"]
