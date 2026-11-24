#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""primolix.confidence —— 原语驱动的检索置信度校准层（支线 #15）。

基于 Primolix 保留的 BM25 原语（`tf`/`df`/`idf`/`dl`/`k1`/`_dead`）及可选语义重排分数，
为 AI Agent 导出有界、可举证的置信度信号（不修改底层检索分数与排序）：
  - `score_ratio`: 相对全查询理论 BM25 上界比例 ∈ [0, 1]（含 OOV 惩罚）
  - `idf_cov`: 命中词 IDF 占查询总信息量的比例 ∈ [0, 1]
  - `in_vocab_ratio`: 查询词入表率 ∈ [0, 1]
  - `margin`: 相对下一候选的分数区分度 ∈ [0, 1]
  - `tier`: 'high' / 'medium' / 'low' / 'miss' + 可执行 `hint`
"""
from __future__ import annotations

from collections import Counter
import math
import re

import numpy as np

from .span import span_compactness

# 纯标点 / 纯符号 token（如 `_`、`.`、`：`、`()`）不携带检索信息：既不该进 `in_vocab_ratio`
# 的分母，也不该出现在 `matched_terms` / `high_df` 里。Exp D 实测：含下划线/点号的查询会被这类
# token 稀释而低估置信度。⚠️ 只过滤置信度的统计口径，⛔ 不改底层 BM25 打分与排序。
_CONTENT_TOKEN_RE = re.compile(r"[0-9A-Za-z\u3400-\u4dbf\u4e00-\u9fff]")


def _lookup_matched_cols(bm, row: int, cols: list[int]) -> set[int]:
    """查询单条物理行 `row` 上命中的查询词列集合（成本 O(该行 nnz)，不物化全行）。"""
    if not cols or row < 0:
        return set()
    col_set = set(cols)
    T = getattr(bm, "T", None)
    if T is not None:
        T_csr = T.tocsr()
        if row < int(T_csr.shape[0]):
            s = int(T_csr.indptr[row])
            e = int(T_csr.indptr[row + 1])
            row_cols = T_csr.indices[s:e]
            return {int(c) for c in row_cols.tolist() if int(c) in col_set}
    for blk, rb in (getattr(bm, "_tc_blocks", None) or []):
        if int(rb) <= row < int(rb) + int(blk.shape[0]):
            local_r = row - int(rb)
            matched = set()
            for c in cols:
                if c < int(blk.shape[1]) and float(blk[local_r, c]) > 0.0:
                    matched.add(int(c))
            return matched
    return set()


def evaluate_confidence(
    bm,
    qtoks: list[str],
    hits: list[dict],
    *,
    hit_rows: list[int] | None = None,
    n_live: int | None = None,
    high_df_threshold: float = 0.35,
) -> dict:
    """为单次查询的命中列表计算原语置信度并原地附加 `hit['confidence']`，返回查询级摘要。"""
    qtoks = list(qtoks)
    # ⭐ 过滤纯标点 token 再建词项分母（见模块顶部 `_CONTENT_TOKEN_RE`）：不影响 hits/排序，
    #    只让 in_vocab_ratio / matched_terms / high_df 的统计口径不再被 `_`、`.` 稀释。
    clean_toks = [t for t in qtoks if _CONTENT_TOKEN_RE.search(t)] or list(qtoks)
    q_counts = Counter(clean_toks)
    if n_live is None:
        dead_n = int(bm._dead[:bm.N].sum()) if getattr(bm, "_dead", None) is not None else 0
        n_live = max(int(getattr(bm, "N", 0) or 0) - dead_n, 1)

    t2i = bm.term2idx
    idf_arr = getattr(bm, "_idf", None)
    df_arr = getattr(bm, "df", None)
    k1 = float(getattr(bm, "k1", 1.5))

    in_vocab: list[tuple[str, int, int, float, float]] = []
    oov_terms: list[str] = []
    oov_count = 0
    high_df_terms: list[str] = []

    for term, cnt in q_counts.items():
        ci = t2i.get(term)
        if ci is None:
            oov_terms.append(term)
            oov_count += int(cnt)
            continue
        df_val = float(df_arr[ci]) if df_arr is not None and ci < len(df_arr) else 0.0
        df_ratio = df_val / max(float(n_live), 1.0)
        if idf_arr is not None and ci < len(idf_arr):
            idf_val = float(idf_arr[ci])
        else:
            idf_val = math.log((n_live - df_val + 0.5) / (df_val + 0.5) + 1.0)
        in_vocab.append((term, int(ci), int(cnt), idf_val, df_ratio))
        if df_ratio >= high_df_threshold:
            high_df_terms.append(term)

    n_uniq = len(q_counts)
    in_vocab_ratio = len(in_vocab) / max(n_uniq, 1)
    # OOV 惩罚按罕见词（df=1）的参考 IDF 估算，防止「9 个 OOV + 1 个常用词」算出虚高饱和比
    oov_ref_idf = math.log((max(n_live, 2) - 0.5) / 1.5 + 1.0)
    idf_sum_invocab = sum(cnt * idf_v for _t, _c, cnt, idf_v, _dr in in_vocab)
    idf_sum_full = idf_sum_invocab + oov_count * oov_ref_idf
    s_max_invocab = idf_sum_invocab * (k1 + 1.0)
    s_max_full = idf_sum_full * (k1 + 1.0)

    if not hits:
        hint = "all_oov_rewrite_or_grep" if not in_vocab else "no_positive_hits_check_oov_or_scope"
        return dict(
            tier="miss",
            hint=hint,
            score_ratio=0.0,
            score_ratio_invocab=0.0,
            idf_cov=0.0,
            in_vocab_ratio=in_vocab_ratio,
            margin=0.0,
            oov=oov_terms,
            high_df=high_df_terms,
            s_max=s_max_full,
            n_hits=0,
        )

    cols = [ci for _t, ci, _cnt, _idf_v, _dr in in_vocab]
    col_info = {ci: (t, cnt, idf_v) for t, ci, cnt, idf_v, _dr in in_vocab}

    bm25_scores = [float(h.get("bm25_score", h.get("score", 0.0))) for h in hits]
    top_bm25 = max(bm25_scores[0], 0.0)
    # 排序间距始终对齐当前命中实际排序分（重排融合或纯余弦为 score，未重排时同 score）
    has_rerank = any("rerank_score" in h for h in hits)
    rank_scores = [float(h.get("score", h.get("rerank_score", 0.0))) for h in hits]

    summary = None
    for idx, h in enumerate(hits):
        bm25_s = bm25_scores[idx]
        sr_full = min(max(bm25_s / s_max_full, 0.0), 1.0) if s_max_full > 0 else 0.0
        sr_in = min(max(bm25_s / s_max_invocab, 0.0), 1.0) if s_max_invocab > 0 else 0.0

        row = hit_rows[idx] if (hit_rows is not None and idx < len(hit_rows)) else -1
        matched_cols = _lookup_matched_cols(bm, row, cols) if row >= 0 else set()
        if matched_cols and idf_sum_full > 0:
            matched_idf = sum(col_info[c][1] * col_info[c][2] for c in matched_cols)
            idf_cov_full = min(max(matched_idf / idf_sum_full, 0.0), 1.0)
            idf_cov_in = min(max(matched_idf / max(idf_sum_invocab, 1e-9), 0.0), 1.0)
            matched_terms = [col_info[c][0] for c in cols if c in matched_cols]
        else:
            idf_cov_full = sr_full
            idf_cov_in = sr_in
            matched_terms = []

        if idx + 1 < len(rank_scores) and abs(rank_scores[idx]) > 1e-9:
            margin = max((rank_scores[idx] - rank_scores[idx + 1]) / abs(rank_scores[idx]), 0.0)
        else:
            margin = 1.0 if len(hits) == 1 else 0.0

        rerank_cos = float(h["rerank_score"]) if "rerank_score" in h else None
        rank_delta = int(h["rank_delta"]) if "rank_delta" in h else None
        sc_pos = span_compactness(h.get("text") or "", clean_toks) if h.get("text") else {
            "proximity": 0.0, "compactness": 0.0, "min_window_chars": 0
        }

        # 判级（半 OOV 超过 50% 时需语义余弦 >= 0.60 才允许脱离 low，防假符号陷阱）
        if oov_terms and in_vocab_ratio < 0.5 and (rerank_cos is None or rerank_cos < 0.60):
            tier = "low"
            hint = "high_oov_rewrite_or_grep"
        elif rerank_cos is not None:
            if rerank_cos >= 0.60 and (idf_cov_full >= 0.45 or sr_full >= 0.25):
                tier = "high"
                hint = "lexical_and_semantic_agreement"
            elif rerank_cos >= 0.60 and idf_cov_full < 0.45:
                tier = "medium"
                hint = "semantic_hit_lexical_partial"
            elif rerank_cos < 0.40 and sr_in >= 0.35:
                tier = "low"
                hint = "keyword_trap_low_semantic"
            elif rerank_cos >= 0.45:
                tier = "medium"
                hint = "moderate_semantic_verify_span"
            else:
                tier = "low"
                hint = "weak_semantic_and_lexical"
        else:
            if sr_full >= 0.25 and idf_cov_full >= 0.60 and in_vocab_ratio >= 0.67:
                tier = "high"
                hint = "strong_lexical_match"
            elif sr_full >= 0.12 and idf_cov_full >= 0.30:
                tier = "medium"
                hint = "partial_idf_coverage_verify_span"
            elif high_df_terms and len(high_df_terms) == len(in_vocab):
                tier = "low"
                hint = "all_high_df_terms_narrow_query"
            else:
                tier = "low"
                hint = "low_lexical_confidence"

        conf = dict(
            tier=tier,
            hint=hint,
            score_ratio=round(sr_full, 4),
            score_ratio_invocab=round(sr_in, 4),
            idf_cov=round(idf_cov_full, 4),
            idf_cov_invocab=round(idf_cov_in, 4),
            in_vocab_ratio=round(in_vocab_ratio, 4),
            margin=round(margin, 4),
            proximity=float(sc_pos["proximity"]),
            span_compactness=float(sc_pos["compactness"]),
            matched_terms=matched_terms,
            oov=oov_terms,
            high_df=high_df_terms,
        )
        if rerank_cos is not None:
            conf["rerank_cos"] = round(rerank_cos, 4)
        if rank_delta is not None:
            conf["rank_delta"] = rank_delta
        h["confidence"] = conf
        if idx == 0:
            summary = dict(conf, s_max=round(s_max_full, 4), n_hits=len(hits), top_bm25=round(top_bm25, 4),
                           has_rerank=has_rerank)
    return summary or {}
