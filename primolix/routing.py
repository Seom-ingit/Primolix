"""路由判定：`update()` 这一批该怎么走。

纯函数、无 I/O、无全局可变状态：只判定，不执行。
判据件：`04c_丙档OOV路由_设计与判据_2026-11-08.md`。

五个出口：
    noop         没有改动
    incremental  词表内、就地真增量
    staged       含新词 → 挂只读段（`O(新 nnz)`，不重建）
    rebuild      整库重建（dense / 批太大）
    fold_first   先把已挂段折回主表，再按 `then` 走（`then` ∈ incremental/staged/rebuild）

两条硬约束（来自代码事实，不是口味）：
  ① 段在场时就地追加会与段的逻辑行重叠（静默错分）⇒ 必须先折回主表；
  ② 段在场时 `save()` 拒写 ⇒ 墓碑改的 `df`/`avgdl` 落不了盘（重载会复活删掉的文档）
     ⇒ 连纯删除也要先折。

`fold_first` 就是这两条的统一出口（0.2.1 起）：不再对调用方抛"先 compact()"，
而是当场折、把折叠耗时如实报账。
"""
from __future__ import annotations

# ---- 出口 ----------------------------------------------------------------
NOOP = "noop"
INCREMENTAL = "incremental"
STAGED = "staged"
REBUILD = "rebuild"
FOLD_FIRST = "fold_first"

EXITS = (NOOP, INCREMENTAL, STAGED, REBUILD, FOLD_FIRST)

# ---- 原因码（单测与日志都按它断言，不要拿 detail 文案当契约） ----------
R_NOOP = "noop"
R_INCREMENTAL = "incremental"
R_STAGED = "staged"
R_REBUILD_DENSE = "rebuild_dense"
R_REBUILD_BATCH_TOO_BIG = "rebuild_batch_too_big"
R_FOLD_INPLACE = "fold_then_incremental"
R_FOLD_BLOCKS = "fold_then_staged_blocks"
R_FOLD_TOMB_SEGMENT = "fold_then_tomb_segment"
R_FOLD_DENSE = "fold_then_rebuild_dense"

# 下面这个默认值没有读数支撑：块数-查询曲线（D3）只在 4.5 万篇 / 17 块上有一点
DEFAULT_COMPACT_BLOCKS = 8


def route_update(*, n_added: int = 0, n_changed: int = 0, n_removed: int = 0,
                 n_oov: int = 0, n_tomb_rows: int = 0, n_tomb_seg_rows: int = 0,
                 n_segments: int = 0, n_blocks: int = 0, dense_on: bool = False,
                 batch_nnz: int = 0, total_nnz: int = 0,
                 compact_blocks: int = DEFAULT_COMPACT_BLOCKS,
                 rebuild_batch_nnz: int | None = None) -> dict:
    """判定这一批的走法。返回 `{route, then, code, detail, folds}`。

    参数（全部是"已数好的量"，函数不做任何扫描）：
        n_added / n_changed / n_removed   hash diff 的三类文件数
        n_oov                             新增/变更内容里词表外的 token 数
        n_tomb_rows                       本批要墓碑的行数（含段内行）
        n_tomb_seg_rows                   其中落在段内的行数（段内不支持删除）
        n_segments                        当前已挂段数
        n_blocks                          当前查询块数（含主表块）
        dense_on                          索引是否带 dense 向量
        batch_nnz / total_nnz             批 / 全库的非零数（只给跨界点判据用）
        compact_blocks                    块数阈值：到了就先折（C）
        rebuild_batch_nnz                 批 nnz 超过它就直接重建；None = 不设限
    """
    _check_non_negative(locals())
    if int(compact_blocks) < 2:
        raise ValueError(f"compact_blocks 必须 ≥ 2（当前 {compact_blocks}）")

    n_segments = int(n_segments)
    n_blocks = int(n_blocks)
    if int(n_added) + int(n_changed) + int(n_removed) == 0:
        return _out(NOOP, R_NOOP, "无改动")

    wants_stage = int(n_oov) > 0

    # ---- 先折后走：三条触发（A / C / 墓碑），外加 dense 与段不共存 ----
    if n_segments > 0:
        if int(n_tomb_seg_rows) > 0:
            # 段内行不可墓碑；折回主表之后就变成普通行了。这条先于 in-vocab 判定，
            # 否则"纯删除 + 段内行"会被报成 fold_inplace（原因码失真）。
            return _fold(n_segments, STAGED if wants_stage else INCREMENTAL,
                         R_FOLD_TOMB_SEGMENT,
                         f"本批要墓碑 {int(n_tomb_seg_rows)} 行**段内**文档（段内不支持删除）"
                         f" → 先折回主表")
        if not wants_stage:
            return _fold(n_segments, INCREMENTAL, R_FOLD_INPLACE,
                         f"已挂 {n_segments} 段：就地更新会与段的逻辑行重叠（静默错分）、"
                         f"且墓碑改的 df/avgdl 在段在场时落不了盘 → 先折回主表再走增量")
        if dense_on:
            return _fold(n_segments, REBUILD, R_FOLD_DENSE,
                         "dense 在场：段内文档并入会让 _vecs 失配 → 先折回主表再重建")
        if n_blocks >= int(compact_blocks):
            return _fold(n_segments, STAGED, R_FOLD_BLOCKS,
                         f"查询块数 {n_blocks} ≥ 阈值 {int(compact_blocks)}"
                         f"（[提案] 默认值，无 1M 读数）→ 先折回主表再挂新段")

    # ---- 段不在场（或已判为不用折） ----
    if not wants_stage:
        return _out(INCREMENTAL, R_INCREMENTAL, "词表内 → 就地真增量")
    if dense_on:
        return _out(REBUILD, R_REBUILD_DENSE,
                    "dense 在场：段内文档并入会让 _vecs 长度失配 → 弃用 dense；"
                    "本批不接受这个降级，改走重建")
    if rebuild_batch_nnz is not None and int(batch_nnz) > int(rebuild_batch_nnz):
        return _out(REBUILD, R_REBUILD_BATCH_TOO_BIG,
                    f"批 nnz {int(batch_nnz)} > 跨界点 {int(rebuild_batch_nnz)}：这一批重建反而更便宜")
    return _out(STAGED, R_STAGED,
                f"含 {int(n_oov)} 个新词 → 走挂段路径（O(新 nnz)，不重建）")


# ---- 内部 ----------------------------------------------------------------
def _out(route: str, code: str, detail: str, *, then: str | None = None) -> dict:
    return dict(route=route, then=then, code=code, detail=detail,
                folds=bool(route == FOLD_FIRST))


def _fold(n_segments: int, then: str, code: str, detail: str) -> dict:
    return _out(FOLD_FIRST, code, detail + f"；折叠 {n_segments} 段", then=then)


def _check_non_negative(scope: dict) -> None:
    for k in ("n_added", "n_changed", "n_removed", "n_oov", "n_tomb_rows",
              "n_tomb_seg_rows", "n_segments", "n_blocks", "batch_nnz", "total_nnz"):
        v = scope.get(k, 0)
        if int(v) < 0:
            raise ValueError(f"{k} 不能为负（当前 {v}）")
