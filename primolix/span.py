"""span 级定位（meta 层，不进 BM25 打分链路）。

设计裁决见 span级返回_发散与施工方案_2026-10-11.md：
- 路线 C（meta 层折中）：位置是展示层需求，tokenize/打分契约不动。
- 纯子串匹配口径，与明理前端 lib/span.ts 完全一致（无分词歧义、两形态可对拍）。
- 供 SparseBM25 直载形态与 core.Primolix 门面形态共同复用（独立模块，不依赖内核类）。

开放接口（后续接入点，今天不做）：
- chunk meta 记 char_start/char_end（chunk_prose 段落切分处）
- core.search 返回后调 hit_span 标注 hits: [(s, e, tok)]
"""

from __future__ import annotations

import re

# 与前端 queryTokens 同款切分：空白 + 中英标点（UNICODE 口径）
_SPLIT_RE = re.compile(
    r"[\s，。；：、！？（）「」『』《》〈〉【】〔〕(){}\[\]'\"“”‘’·…—\-~,.;:!?\\|/*+]+"
)


def query_tokens(query: str) -> list[str]:
    """查询切词：按空白与中英标点切（与前端 lib/span.ts 同款，不做真分词）。"""
    return [w for w in _SPLIT_RE.split(query) if w]


def hit_ranges(text: str, tokens: list[str]) -> list[tuple[int, int, str]]:
    """每个 token 在 text 内的全部出现区间 [(start, end, tok)]，按起点排序。"""
    out: list[tuple[int, int, str]] = []
    for tok in tokens:
        if not tok:
            continue
        start = 0
        while (i := text.find(tok, start)) >= 0:
            out.append((i, i + len(tok), tok))
            start = i + 1
    out.sort(key=lambda r: (r[0], -r[1]))
    return out


def hit_span(text: str, query: str) -> tuple[int, int] | None:
    """最小覆盖跨度 [start, end)：整串优先（更准）；无词法命中返回 None。

    与前端 hitSpan 逐行为对拍关系（同输入必同输出）。
    """
    q = query.strip()
    if not q or not text:
        return None
    whole = text.find(q)
    if whole >= 0:
        return (whole, whole + len(q))
    words = query_tokens(q)
    if not words:
        return None
    ranges = hit_ranges(text, words)
    if not ranges:
        return None
    return (min(r[0] for r in ranges), max(r[1] for r in ranges))
