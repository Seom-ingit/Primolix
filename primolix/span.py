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


def span_compactness(text: str, q_terms) -> dict:
    """原语级位置紧凑度 P_pos：计算查询有效词项在文档内的覆盖与最小聚集跨度 ∈ [0, 1]。"""
    terms: list[str] = []
    seen: set[str] = set()
    for t in q_terms:
        tl = str(t).strip().lower()
        if not tl or tl in seen or not any(ch.isalnum() for ch in tl):
            continue
        seen.add(tl)
        terms.append(tl)
    n_q = len(terms)
    if n_q == 0 or not text:
        return dict(matched_terms_n=0, q_terms_n=n_q, coverage=0.0,
                    min_window_chars=0, proximity=0.0, compactness=0.0)

    tl_doc = text.lower()
    occs: list[tuple[int, int, int]] = []
    matched_ids: set[int] = set()
    matched_len_sum = 0
    for tid, tok in enumerate(terms):
        pos = tl_doc.find(tok)
        if pos < 0:
            continue
        matched_ids.add(tid)
        matched_len_sum += len(tok)
        tlen = len(tok)
        while pos >= 0:
            occs.append((pos, pos + tlen, tid))
            pos = tl_doc.find(tok, pos + 1)

    n_hit = len(matched_ids)
    if n_hit == 0:
        return dict(matched_terms_n=0, q_terms_n=n_q, coverage=0.0,
                    min_window_chars=0, proximity=0.0, compactness=0.0)

    cov = n_hit / float(n_q)
    if n_hit == 1:
        prox = 1.0 if n_q == 1 else 0.25
        win = matched_len_sum
    else:
        occs.sort(key=lambda x: (x[0], x[1]))
        counts: dict[int, int] = {}
        have = 0
        left = 0
        win = len(tl_doc) + 1
        for _s, e, tid in occs:
            if counts.get(tid, 0) == 0:
                have += 1
            counts[tid] = counts.get(tid, 0) + 1
            while have == n_hit:
                ls, _le, ltid = occs[left]
                cur_win = e - ls
                if cur_win < win:
                    win = cur_win
                counts[ltid] -= 1
                if counts[ltid] == 0:
                    have -= 1
                left += 1
        slack = max(0, win - matched_len_sum)
        prox = 1.0 / (1.0 + slack / 80.0)
    return dict(
        matched_terms_n=n_hit,
        q_terms_n=n_q,
        coverage=round(cov, 4),
        min_window_chars=int(win),
        proximity=round(prox, 4),
        compactness=round(cov * prox, 4),
    )
