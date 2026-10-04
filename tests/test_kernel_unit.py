#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Kernel-level unit checks for primolix. Fast, no external data.

Run:  python tests/test_kernel_unit.py

Checks (each prints PASS/FAIL; process exits 1 if anything fails):
  U1  score_all is deterministic (same call twice, identical array)
  U2  save / load roundtrip keeps scores bit-exact
  U3  update_docs tombstones the old row and appends the new one
  U4  oov_terms reports exactly the out-of-vocabulary tokens
  U5  append into the preallocated CSR equals a scipy vstack build (bit-exact arrays)
  U6  score increases with term frequency (formula sanity)
  U7  changing k1/b does not modify the material (T, df, dl, avgdl unchanged)
  U8  no public legacy alias ('ZELIX' is an internal codename)
  U9  a legacy fmt string ('zelix-bm25-v3') still loads
  U10 mount_segment == full rebuild (bit-exact)
  U11 fold_segment: scores unchanged across the fold, and == full rebuild
  U12 unmount_segment: scores return to the pre-mount state; N / V / report restored
  U13 topk == score_all (compared by score values; ties may be ordered differently)
  U14 DN_PRECOMP is bit-exact (including a rebuild after a parameter change)
  U15 MATVEC == fast path (within a documented tolerance)
  U16 W_CACHE hit == on-the-fly, and a parameter change forces invalidation
  U17 ADD_AT (unbuffered accumulation) is bit-exact on both paths
  U18 persisting w at index time: read back bit-exact; a key mismatch is refused
  U19 large-block append (targeted growth) does not blow up and stays bit-exact
  U20 load-then-update still works (dead is writable); a `.mm/` generation-key mismatch is refused (never a silent mixed-generation read)
  U21 appending in place while a segment is mounted is refused loudly (it overlaps the segment's rows)
  U22 saving while a segment is mounted is refused loudly; `fold_all()` makes it durable, reload stays bit-exact (no double counting)

  Not covered here: the view family (see docs/api.md).
"""
from __future__ import annotations

import atexit
import os
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from primolix.kernel import PreallocCSR, SparseBM25  # noqa: E402

FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  |  {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


def tok(s: str) -> list[str]:
    return s.split()


DOCS = [
    "sparse retrieval kernel with incremental updates",
    "bm25 weights from term frequency and document length",
    "cooking bread soup onion carrot celery",
    "lazy execution and candidate pool reranking",
]
# NOTE: the kernel takes token lists (Primolix.query takes text and tokenizes for you).
QUERIES = [tok(q) for q in ("sparse kernel", "bm25 weights", "cooking soup")]

# ---------------------------------------------------------------- U1 determinism
bm = SparseBM25(list(DOCS), tok, k1=1.5, b=0.75)
a1 = np.asarray(bm.score_all(QUERIES[0]), dtype=np.float32)
a2 = np.asarray(bm.score_all(QUERIES[0]), dtype=np.float32)
check("U1 score_all is deterministic", np.array_equal(a1, a2) and float(a1.max()) > 0.0,
      f"len={a1.size}, max={float(a1.max()):.3f}")

# ---------------------------------------------------------------- U2 save / load roundtrip
run = Path(os.environ.get("PRIMOLIX_UNIT_DIR") or (HERE / "_unit_run"))
shutil.rmtree(run, ignore_errors=True)
atexit.register(shutil.rmtree, run, ignore_errors=True)
run.mkdir(parents=True)
idx = run / "bm25.npz"
bm.save(str(idx))
bm2 = SparseBM25.load(str(idx))
d2 = 0.0
for q in QUERIES:
    d2 = max(d2, float(np.abs(np.asarray(bm.score_all(q), dtype=np.float32)
                              - np.asarray(bm2.score_all(q), dtype=np.float32)).max()))
check("U2 save / load roundtrip is bit-exact", d2 == 0.0,
      f"max|d|={d2:.3e}; files={sorted(p.name for p in run.iterdir())[:4]}")

# ---------------------------------------------------------------- U3 tombstone + append
n0 = len(DOCS)
s_old = np.asarray(bm.score_all(tok("sparse")), dtype=np.float32)
bm.update_docs([0], [["lazy", "execution", "lazy", "execution"]])
s_new = np.asarray(bm.score_all(tok("lazy")), dtype=np.float32)
check("U3 update tombstones old row and appends new one",
      s_new.size == n0 + 1 and float(s_new[0]) == 0.0 and float(s_new[n0]) > 0.0,
      f"N {n0} -> {s_new.size}; old row score={float(s_new[0]):.3f}; appended={float(s_new[n0]):.3f}")

# ---------------------------------------------------------------- U4 oov_terms
oov = bm.oov_terms([["sparse", "zznovelterm", "anothernovelterm"]])
check("U4 oov_terms reports exactly the unknown tokens",
      set(oov) == {"zznovelterm", "anothernovelterm"},
      f"reported={sorted(oov)}; in-vocab query reports={bm.oov_terms([['sparse']])}")

# ---------------------------------------------------------------- U5 PreallocCSR append == vstack
import scipy.sparse as sp  # noqa: E402
base = sp.csr_matrix(np.asarray([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]], dtype=np.float32))
tail = sp.csr_matrix(np.asarray([[0.0, 4.0, 0.0]], dtype=np.float32))
buf = PreallocCSR.from_csr(base, slack=1.5)
buf.append(np.asarray([0], dtype=np.int32), np.asarray([1], dtype=np.int32),
           np.asarray([4.0], dtype=np.float32), np.asarray([1], dtype=np.int64))
got = buf.wrap().tocsr()
want = sp.vstack([base, tail]).tocsr()
ok5 = (np.array_equal(got.indptr, want.indptr) and np.array_equal(got.indices, want.indices)
       and np.array_equal(got.data, want.data))
check("U5 preallocated append equals vstack (bit-exact arrays)", ok5,
      f"shape {got.shape} vs {want.shape}")

# ---------------------------------------------------------------- U6 frequency monotonicity
low = SparseBM25(["kernel", "kernel kernel", "other"], tok)
s = np.asarray(low.score_all(tok("kernel")), dtype=np.float32)
check("U6 score increases with term frequency", float(s[1]) > float(s[0]) > 0.0,
      f"scores={[round(float(x), 3) for x in s]}")

# ---------------------------------------------------------------- U7 formula change touches nothing
def fingerprint(m):
    return (bytes(np.ascontiguousarray(m.T.data[:64]).tobytes()),
            bytes(np.ascontiguousarray(np.asarray(m.df)).tobytes()),
            bytes(np.ascontiguousarray(np.asarray(m.dl)).tobytes()),
            float(m.avgdl), int(m.N), int(m.T.nnz))

before = fingerprint(bm)
bm.k1, bm.b = 2.5, 0.2
after = fingerprint(bm)
check("U7 changing k1/b leaves the material untouched", before == after,
      f"avgdl={after[3]:.3f}, N={after[4]}, nnz={after[5]}")

# ---------------------------------------------------------------- U8 no public legacy alias
# "ZELIX" is the previous generation's internal codename: it must not be exported and must not be
# reachable from the top level.
import primolix as _primolix  # noqa: E402
check("U8 no public legacy alias ('ZELIX' is an internal codename)",
      "ZELIX" not in getattr(_primolix, "__all__", []) and not hasattr(_primolix, "ZELIX"),
      f"__all__={getattr(_primolix, '__all__', None)} · hasattr(primolix, 'ZELIX')="
      f"{hasattr(_primolix, 'ZELIX')}")

# ---------------------------------------------------------------- U9 legacy fmt string
# Indexes written before the rename carry fmt="zelix-bm25-v3". The on-disk format did not change
# (only the package/class name did), so load() must keep accepting them.
import shutil as _shutil  # noqa: E402
import os as _os  # noqa: E402

legacy = run / "legacy.npz"
_shutil.copy2(idx, legacy)
for suffix in (".mm", ".vocab"):                        # 同伴目录一起改名
    src, dst = Path(str(idx) + suffix), Path(str(legacy) + suffix)
    if src.exists():
        _shutil.rmtree(dst, ignore_errors=True)
        _shutil.copytree(src, dst, dirs_exist_ok=True)
with np.load(legacy, allow_pickle=True) as zf:
    kw = {k: zf[k] for k in zf.files}
    new_fmt = str(kw.get("fmt", ""))
    kw["fmt"] = np.array(new_fmt.replace("primolix-bm25-", "zelix-bm25-"))
tmp = legacy.with_name(legacy.name + ".tmp.npz")
np.savez(tmp, **kw)
_os.replace(tmp, legacy)
bm9 = SparseBM25.load(str(legacy))
bm_ref9 = SparseBM25.load(str(idx))          # 与 legacy 同一份磁盘内容（区别只剩 fmt）
d9 = float(np.abs(np.asarray(bm9.score_all(QUERIES[0]), dtype=np.float32)
                  - np.asarray(bm_ref9.score_all(QUERIES[0]), dtype=np.float32)).max())
check("U9 legacy fmt string ('zelix-bm25-v3') still loads",
      new_fmt.startswith("primolix-bm25-") and d9 == 0.0,
      f"把 `fmt` 改回 `{kw['fmt'].item()}` 后仍可加载 · 分数 max|d|={d9:.3e} · N={int(bm9.N)}"
      f"（注意：对照组必须是同一份磁盘内容，不能用被 U3 改过的内存对象）")

# ---------------------------------------------------------------- U10 mount_segment
# 只读段 + 合并 df → 与"全库重建"逐位一致（两条路都走内核自己的打分器 → 要求 max|Δ| == 0）
bm10 = SparseBM25(list(DOCS), tok, k1=1.5, b=0.75)
V0_10 = int(bm10.V)
NEWT = ["zznewterm", "zzother"]
seg_docs = ["lazy execution candidate pool zznewterm",
            "sparse kernel kernel zznewterm zzother"]      # 含重复词 → 顺便验 tf/df 的区分
V1_10 = V0_10 + len(NEWT)
t2i10 = {w: int(bm10.term2idx[w]) for d in DOCS for w in tok(d)}   # DOCS 是句子，要按词建映射
rows10, cols10, vals10, dl10 = [], [], [], []
for r, doc in enumerate(seg_docs):
    n = 0
    for w in tok(doc):
        if w in t2i10:
            c = t2i10[w]
        elif w in NEWT:
            c = V0_10 + NEWT.index(w)
        else:
            continue
        rows10.append(r); cols10.append(c); vals10.append(1.0); n += 1
    dl10.append(float(n))
seg_csc10 = sp.csc_matrix((np.asarray(vals10, dtype=np.float32),
                           (np.asarray(rows10, dtype=np.int32), np.asarray(cols10, dtype=np.int32))),
                          shape=(len(seg_docs), V1_10))
segdir10 = run / "seg0"
segdir10.mkdir(parents=True, exist_ok=True)
np.save(segdir10 / "seg0.data.npy", np.asarray(seg_csc10.data, dtype=np.float32))
np.save(segdir10 / "seg0.idx.npy", np.asarray(seg_csc10.indices, dtype=np.int32))
np.save(segdir10 / "seg0.ptr.npy", np.asarray(seg_csc10.indptr, dtype=np.int32))
rep10 = bm10.mount_segment(segdir10, "seg0", NEWT, dl10)
ref10 = SparseBM25(list(DOCS) + seg_docs, tok, k1=1.5, b=0.75)
d10 = 0.0
for q in QUERIES + [tok("zznewterm")]:
    a = np.asarray(bm10.score_all(q), dtype=np.float32)
    b = np.asarray(ref10.score_all(q), dtype=np.float32)
    if a.shape == b.shape:
        d10 = max(d10, float(np.abs(a - b).max()))
    else:
        d10 = float("inf")
check("U10 mounted segment == full rebuild (bit exact, mount_segment)", d10 == 0.0,
      f"max|d|={d10:.3e} · 段 {rep10['rows']} 行 / 新词 {rep10['m_new']} 个 · "
      f"N {len(DOCS)}→{int(bm10.N)} · V {V0_10}→{int(bm10.V)} · 报告={sorted(bm10.segment_report())}")

# ---------------------------------------------------------------- U11 fold_segment
# 折叠（扩列 + 追加行）之后：分数不变，且仍与全库重建逐位一致；段从"已挂"移入"已折叠"
s_before11 = np.stack([np.asarray(bm10.score_all(q), dtype=np.float32) for q in QUERIES])
rep11 = bm10.fold_segment("seg0")
s_after11 = np.stack([np.asarray(bm10.score_all(q), dtype=np.float32) for q in QUERIES])
ref_s11 = np.stack([np.asarray(ref10.score_all(q), dtype=np.float32) for q in QUERIES])
d_fold = float(np.abs(s_before11 - s_after11).max())
d_ref = float(np.abs(s_after11 - ref_s11).max())
check("U11 fold_segment: 折叠前后分数不变，且 ≡ 全库重建",
      d_fold == 0.0 and d_ref == 0.0 and not bm10.segment_report() and "seg0" in bm10.folded_report(),
      f"折叠前↔后 {d_fold:.3e} · 折叠后↔全库重建 {d_ref:.3e} · 折叠耗时 {rep11['ms']:.2f} ms · "
      f"nnz {rep11['nnz_before']}→{rep11['nnz_after']} · 已挂={sorted(bm10.segment_report())} 已折叠={sorted(bm10.folded_report())}")

# ---------------------------------------------------------------- U12 unmount_segment
# 卸载 = 把挂载逐项撤回：分数回到挂载前、N/V 缩回、报告清空
bm12 = SparseBM25(list(DOCS), tok, k1=1.5, b=0.75)
base_s12 = np.stack([np.asarray(bm12.score_all(q), dtype=np.float32) for q in QUERIES])
N0_12, V0_12 = int(bm12.N), int(bm12.V)
segdir12, dls12 = segdir10, dl10                      # 复用 U10 写的段（同语料、同形状）
rep12 = bm12.mount_segment(segdir12, "seg0", NEWT, dls12, rows=len(seg_docs), generation="u12")
mid12 = np.stack([np.asarray(bm12.score_all(q), dtype=np.float32) for q in QUERIES])
un12 = bm12.unmount_segment("seg0")
after_s12 = np.stack([np.asarray(bm12.score_all(q), dtype=np.float32) for q in QUERIES])
d_back = float(np.abs(after_s12 - base_s12).max())
# 挂段期间 N=6、挂载前 N=4 → 只比原文档那几行（活跃集对齐；否则 ValueError (3,6)(3,4)）
d_mid = float(np.abs(mid12[:, :N0_12] - base_s12).max())
check("U12 unmount_segment: 卸载后分数回到挂载前，且 N/V/报告复原",
      d_back == 0.0 and int(bm12.N) == N0_12 and int(bm12.V) == V0_12
      and not bm12.segment_report() and un12["rows"] == len(seg_docs),
      f"卸载前↔后 {d_back:.3e}（要求 0）· 挂段期间相对挂载前 {d_mid:.3e}（>0 → 说明段真的生效过）· "
      f"N {N0_12}→{int(bm12.N)} · V {V0_12}→{int(bm12.V)} · 报告={sorted(bm12.segment_report())}")

# ---------------------------------------------------------------- U13 topk
# 判据用"分数值"而不是"行号集合"：k 边界上并列（tie）时两种取法都合法
#    （查 `zznewterm` 时 4 篇 0 分并列 → `argpartition` 与 `argsort` 各挑一个 → 行号集合必然可不同）
bm13 = SparseBM25(list(DOCS) + seg_docs, tok, k1=1.5, b=0.75)
ok13 = True
d13 = 0.0
last13 = None
for q in QUERIES + [tok("zznewterm")]:
    s = np.asarray(bm13.score_all(q), dtype=np.float32)
    k = min(5, s.size)
    want_vals = np.sort(s)[::-1][:k]                       # 前 k 大的"分数值"
    got, gs = bm13.topk(q, k=5)
    desc = all(float(gs[i]) >= float(gs[i + 1]) for i in range(len(gs) - 1))
    uniq = len({int(x) for x in got}) == len(got)          # 无重复行号
    ok13 = ok13 and bool(len(got) == k and desc and uniq and np.array_equal(gs, want_vals.astype(np.float32)))
    last13 = [int(x) for x in got]
    if len(got):
        d13 = max(d13, float(np.abs(gs - s[np.asarray(got)]).max()))
check("U13 topk ≡ score_all（按分数值判，允许并列取法不同）", ok13 and d13 == 0.0,
      f"分数自洽 max|d|={d13:.3e} · 最后一例 top-5 行号 {last13}（与参考前 5 大的分数值逐位一致）")

# ---------------------------------------------------------------- U14 DN_PRECOMP
# 预计算"每行规范化项"必须逐位等价（同一表达式、同一 dtype 规则）＋ 参数变了必须重建
bm14 = SparseBM25(list(DOCS) + seg_docs, tok, k1=1.5, b=0.75)
q14 = QUERIES + [tok("zznewterm")]
_DEF_FAST = SparseBM25.FAST_SCORE      # 快照"当前默认"并在 finally 里恢复（硬编码 False 会在改默认时把后面测试带偏）
SparseBM25.FAST_SCORE = True
try:
    SparseBM25.DN_PRECOMP = False
    base14 = np.stack([np.asarray(bm14.score_all(q), dtype=np.float32) for q in q14])
    SparseBM25.DN_PRECOMP = True
    pre14 = np.stack([np.asarray(bm14.score_all(q), dtype=np.float32) for q in q14])
    d14 = float(np.abs(pre14 - base14).max())
    # 换参数后必须重建（否则 dn 会带着旧参数 → 静默算错）
    bm14.k1 = 2.0
    a14 = np.asarray(bm14.score_all(q14[0]), dtype=np.float32)
    SparseBM25.DN_PRECOMP = False
    b14 = np.asarray(bm14.score_all(q14[0]), dtype=np.float32)
    d14b = float(np.abs(a14 - b14).max())
    dn14 = bm14._ensure_dn()
finally:
    SparseBM25.DN_PRECOMP = False
    SparseBM25.FAST_SCORE = _DEF_FAST
check("U14 DN_PRECOMP 逐位等价（含换参数后重建）", d14 == 0.0 and d14b == 0.0,
      f"开关前↔后 {d14:.3e}（要求 0）· 换 `k1` 后 {d14b:.3e}（要求 0）· "
      f"dn dtype={np.asarray(dn14).dtype} · 长度 {np.asarray(dn14).size} · 常驻 ≈{np.asarray(dn14).nbytes / 1e6:.1f} MB")

# ---------------------------------------------------------------- U15 MATVEC
# 判据不要求逐位：`bincount` 在 float64 里累加、顺序与"按词按块"不同 → 报 max|d|
bm15 = SparseBM25(list(DOCS) + seg_docs, tok, k1=1.5, b=0.75)
SparseBM25.FAST_SCORE = True
SparseBM25.DN_PRECOMP = True
try:
    SparseBM25.MATVEC = False
    ref15 = np.stack([np.asarray(bm15.score_all(q), dtype=np.float32) for q in q14])
    SparseBM25.MATVEC = True
    mv15 = np.stack([np.asarray(bm15.score_all(q), dtype=np.float32) for q in q14])
finally:
    SparseBM25.MATVEC = False
    SparseBM25.DN_PRECOMP = False
    SparseBM25.FAST_SCORE = _DEF_FAST
d15 = float(np.abs(mv15 - ref15).max())
rel15 = d15 / (float(np.abs(ref15).max()) or 1.0)
check("U15 MATVEC ≡ 快路径（按约定容差判；实测并留档 max|d|）", rel15 <= 1e-5,
      f"max|d| = {d15:.3e}（相对 {rel15:.2e} ≤1e-5）· 非逐位是已知口径取舍"
      f"（`bincount` float64 累加、顺序不同）· 若为 0 则更好：{d15 == 0.0}")

# ---------------------------------------------------------------- U16 W_CACHE（物化层 + 缓存键）
# ① 命中时 ≡ 现算 ② 换参数后必须失效（否则静默用旧 w）
bm16 = SparseBM25(list(DOCS) + seg_docs, tok, k1=1.5, b=0.75)
SparseBM25.FAST_SCORE = True
SparseBM25.DN_PRECOMP = True
try:
    SparseBM25.W_CACHE = False
    ref16 = np.stack([np.asarray(bm16.score_all(q), dtype=np.float32) for q in q14])
    SparseBM25.W_CACHE = True
    hit16 = np.stack([np.asarray(bm16.score_all(q), dtype=np.float32) for q in q14])
    d16 = float(np.abs(hit16 - ref16).max())
    # 换参数 → 键变 → 必须重建（否则静默用旧 w）
    bm16.k1 = 2.0
    with_cache = np.asarray(bm16.score_all(q14[0]), dtype=np.float32)
    SparseBM25.W_CACHE = False
    without = np.asarray(bm16.score_all(q14[0]), dtype=np.float32)
    d16b = float(np.abs(with_cache - without).max())
    kb = bm16._w_key()
finally:
    SparseBM25.W_CACHE = False
    SparseBM25.DN_PRECOMP = False
    SparseBM25.FAST_SCORE = _DEF_FAST
check("U16 W_CACHE 命中 ≡ 现算，且换参后强制失效", d16 == 0.0 and d16b == 0.0,
      f"命中↔现算 {d16:.3e}（要求 0）· 换 `k1` 后 {d16b:.3e}（要求 0 → 证明键真的变了）· "
      f"键末项（块数）={kb[-1]} · 代次={kb[3]}")

# ---------------------------------------------------------------- U17 ADD_AT（无缓冲累加）
# ① 现算路径：开/关逐位相同 ② 物化层命中路径：开/关逐位相同
bm17 = SparseBM25(list(DOCS) + seg_docs, tok, k1=1.5, b=0.75)
SparseBM25.FAST_SCORE = True
SparseBM25.DN_PRECOMP = True
try:
    SparseBM25.ADD_AT = False
    a17 = np.stack([np.asarray(bm17.score_all(q), dtype=np.float32) for q in q14])
    SparseBM25.ADD_AT = True
    b17 = np.stack([np.asarray(bm17.score_all(q), dtype=np.float32) for q in q14])
    d17a = float(np.abs(a17 - b17).max())
    SparseBM25.W_CACHE = True
    c17 = np.stack([np.asarray(bm17.score_all(q), dtype=np.float32) for q in q14])
    SparseBM25.W_CACHE = False
    d17b = float(np.abs(a17 - c17).max())
finally:
    SparseBM25.ADD_AT = False
    SparseBM25.W_CACHE = False
    SparseBM25.DN_PRECOMP = False
    SparseBM25.FAST_SCORE = _DEF_FAST
check("U17 ADD_AT 无缓冲累加（现算与命中路径都逐位）", d17a == 0.0 and d17b == 0.0,
      f"现算 开↔关 {d17a:.3e} · 命中(ADD_AT 开)↔现算 {d17b:.3e}（都要求 0）")

# ---------------------------------------------------------------- U18 索引期落盘 w
# ① 存（`w_cache=True`）→ 读 → 命中 ≡ 现算（逐位） ② 键不符 → 必须拒用
import shutil as _sh18
_d18 = run / "u18"                       # 用测试自己的可写目录（沙箱下 tempfile 会被拒）
_sh18.rmtree(_d18, ignore_errors=True)
_d18.mkdir(parents=True, exist_ok=True)
bm18 = SparseBM25(list(DOCS) + seg_docs, tok, k1=1.5, b=0.75)
SparseBM25.FAST_SCORE = True
SparseBM25.DN_PRECOMP = True
SparseBM25.ADD_AT = True
SparseBM25.W_CACHE = False
try:
    ref18 = np.stack([np.asarray(bm18.score_all(q), dtype=np.float32) for q in q14])
    p18 = _d18 / "bm25.npz"
    bm18.save(str(p18), w_cache=True)
    has_w = (p18.parent / "bm25.npz.mm" / "w_key.txt").exists() and list((p18.parent / "bm25.npz.mm").glob("w_*.npy"))
    bm18b = SparseBM25.load(str(p18))
    used_disk = bool(getattr(bm18b, "_w_persist_ok", False))
    SparseBM25.W_CACHE = True
    got18 = np.stack([np.asarray(bm18b.score_all(q), dtype=np.float32) for q in q14])
    d18a = float(np.abs(got18 - ref18).max())
    from_disk = bool(getattr(bm18b, "_w_from_disk", False))
    # 键不符 → 拒用
    (p18.parent / "bm25.npz.mm" / "w_key.txt").write_text("v1|k1=WRONG|b=0|avg=0|N=0|nnz=0|V=0|live=0",
                                                          encoding="utf-8")
    bm18c = SparseBM25.load(str(p18))
    rejected = not bool(getattr(bm18c, "_w_persist_ok", False))
    got18c = np.stack([np.asarray(bm18c.score_all(q), dtype=np.float32) for q in q14])
    d18b = float(np.abs(got18c - ref18).max())
finally:
    SparseBM25.W_CACHE = False
    SparseBM25.ADD_AT = _DEF_ADD_AT if "_DEF_ADD_AT" in dir() else True
    SparseBM25.DN_PRECOMP = _DEF_DN if "_DEF_DN" in dir() else True
    SparseBM25.FAST_SCORE = _DEF_FAST
check("U18 索引期落盘 w：读回逐位等价；键不符则拒用", bool(has_w and used_disk and from_disk)
      and d18a == 0.0 and rejected and d18b == 0.0,
      f"w_key/w_*.npy 落盘={bool(has_w)} · load 后判定可用={used_disk} · 实际走盘={from_disk} · "
      f"读回↔现算 {d18a:.3e} · 键不符被拒={rejected} · 拒用后仍正确 {d18b:.3e}")

# ---------------------------------------------------------------- U19 PreallocCSR 大块 append（目标制扩容）
# `_grow()` 按「当前 nnz × 2」扩 → 一次追加一整"片"（远超已用量）时扩一次仍装不下
#    → 分片建库当场失败（`could not broadcast ...`）
bm19 = SparseBM25(list(DOCS), tok, k1=1.5, b=0.75)
n19_before = int(bm19.T.nnz)
_S19 = "sparse retrieval kernel with incremental updates tombstones appended shard"
big = [tok(_S19) for _ in range(400)]      # append_docs 收"已分词的行"（list of list）
texts19 = [_S19] * 400                     # 参考建库收"原文"（str）—— 两者别混
t0_19 = __import__("time").perf_counter()
bm19.append_docs(big)                      # 一次追加的 nnz 远大于当前已用（≈400 行 × 10+ 词）
grew_ok = True
try:
    _ = bm19.T.shape
except Exception as e:                     # noqa: BLE001
    grew_ok = False
d19 = 0.0
bm19r = SparseBM25(list(DOCS) + texts19, tok, k1=1.5, b=0.75)
s_a = np.asarray(bm19.score_all(QUERIES[0]), dtype=np.float32)
s_b = np.asarray(bm19r.score_all(QUERIES[0]), dtype=np.float32)
if s_a.size == s_b.size:
    d19 = float(np.abs(s_a - s_b).max())
check("U19 大块 append 不崩且逐位（`_grow` 目标制）", grew_ok and d19 == 0.0 and int(bm19.N) > len(DOCS),
      f"追加 400 行：nnz {n19_before} → {int(bm19.T.nnz)} · `grow_count`={getattr(bm19._Tbuf, 'grow_count', '?')}"
      f" · vs 一次建库 max|d|={d19:.3e}（要求 0）· 耗时 {(__import__('time').perf_counter()-t0_19)*1e3:.0f} ms")

# ------------------------------------------- U20 载入后可继续更新 ＋ `.mm/` 世代键不符则拒用
# 两条约束：① `.mm/dead.npy` 以 `mmap_mode="r"` 载入 → 重载后再打墓碑必须可写（否则当场 ValueError）；
#   ② `.mm/` 与 npz 不同代（同形状时长度检查拦不住）→ 靠 `mm_key.txt` 世代键拒用，不许静默混代。
_w20 = run / "u20"
_w20.mkdir(parents=True, exist_ok=True)
_bm20 = SparseBM25(list(DOCS), tok, k1=1.5, b=0.75)
_bm20._tombstone_rows([0])                       # 造出墓碑 → save 会写 dead.npy
_bm20.save(str(_w20 / "bm25.npz"))
_bm20b = SparseBM25.load(str(_w20 / "bm25.npz"))  # 走 `.mm/` 路径（含 dead.npy）
_reloaded_ok = True
try:
    _bm20b._tombstone_rows([1])                  # 修复前：read-only 报错
except Exception as _e:                          # noqa: BLE001
    _reloaded_ok = False
    _d1_err = repr(_e)
_ref20 = np.asarray(_bm20b.score_all(QUERIES[0]), dtype=np.float32)

#   ② 世代键：同形状、不同内容（tf(alpha)=2 → 1；N/V/nnz 全同）→ 必须拒用（否则静默错）
_qa, _qb = run / "u20a", run / "u20b"
_qa.mkdir(exist_ok=True)
_qb.mkdir(exist_ok=True)
_ma = SparseBM25(["alpha alpha beta"], tok)
_ma.save(str(_qa / "bm25.npz"))
_mb = SparseBM25(["alpha beta beta"], tok)
_mb.save(str(_qb / "bm25.npz"))
_ref_a = np.asarray(_ma.score_all(["alpha"]), dtype=np.float32)
_ref_b = np.asarray(_mb.score_all(["alpha"]), dtype=np.float32)
_same_shape = (int(_ma.T.nnz) == int(_mb.T.nnz) and int(_ma.N) == int(_mb.N)
               and int(_ma.V) == int(_mb.V))
import shutil as _sh20                                  # noqa: E402
_sh20.copy2(_qb / "bm25.npz", _qa / "bm25.npz")         # npz ← B，`.mm/` 仍是 A（同形状 → 无长度检查拦得住）
_mx = SparseBM25.load(str(_qa / "bm25.npz"))
_got20 = np.asarray(_mx.score_all(["alpha"]), dtype=np.float32)
_refused = bool(np.array_equal(_got20, _ref_b)) and not bool(np.array_equal(_got20, _ref_a))
_d20 = float(np.abs(_ref20 - np.asarray(_bm20b.score_all(QUERIES[0]), dtype=np.float32)).max())
check("U20 载入后可继续更新（dead 可写）＋ `.mm/` 世代键不符则拒用（不静默混代）",
      _reloaded_ok and _same_shape and _refused and _d20 == 0.0,
      f"重载后再打墓碑={_reloaded_ok}（修复前 `read-only`）· 同形状混代={_same_shape}"
      f" · 拒用后给出新世代分数={_refused}（旧世代 {_ref_a.tolist()} / 新世代 {_ref_b.tolist()}"
      f" / 实得 {_got20.tolist()}）· 复算一致 max|d|={_d20:.3e}")

# ------------------------------------------- U21 段在场时就地追加必须响亮拒绝（不许静默错分）
# 挂段后就地 `append_docs` 会给错分数：`T` 的新行与段的逻辑行重叠（`T` 从 row_base 长，而 `N` 已含段行）
_seg21 = run / "u21seg"
_seg21.mkdir(parents=True, exist_ok=True)
_bm21 = SparseBM25(list(DOCS), tok, k1=1.5, b=0.75)
_V21 = len(_bm21.idx2term)
_t21 = {w: i for i, w in enumerate(_bm21.idx2term)}
_ri21, _ci21, _da21, _n21 = [], [], [], ["zzeta", "zztheta"]
for _i, _toks in enumerate([["zzeta", "zztheta"], ["zztheta"] + [DOCS[0].split()[0]]]):
    for _w, _c in __import__("collections").Counter(_toks).items():
        _ri21.append(_i)
        _col21 = _t21.get(_w)                    # 别写 `dict.get(k, 默认)`：默认值会被急切求值
        if _col21 is None:                       #    → 老词会先撞 `index()` 抛错
            _col21 = _V21 + _n21.index(_w)
        _ci21.append(_col21)
        _da21.append(float(_c))
_c21 = sp.csc_matrix((np.asarray(_da21, np.float32),
                      (np.asarray(_ri21, np.int32), np.asarray(_ci21, np.int32))),
                     shape=(2, _V21 + len(_n21)))
for _nm, _arr in (("s21.data.npy", _c21.data), ("s21.idx.npy", _c21.indices),
                  ("s21.ptr.npy", _c21.indptr)):
    np.save(_seg21 / _nm, _arr)
_bm21.mount_segment(str(_seg21), "s21", _n21, np.asarray([2.0, 2.0], np.float32), rows=2)
_seg_refused, _seg_msg = False, ""
try:
    _bm21.append_docs([tok("alpha beta")])
except RuntimeError as _e:                                    # 必须是响亮拒绝
    _seg_refused = "fold_segment" in str(_e)
    _seg_msg = str(_e)[:90]
check("U21 段在场时就地追加 → 响亮拒绝（不许静默错分）", _seg_refused,
      f"已挂段={sorted(_bm21.segment_report())} · 追加被拒={_seg_refused} · 报错={_seg_msg!r}"
      f"（修复前：不报错且 `max|Δ| = 1.73`）")

# ------------------------- U22 段在场时 save() 必须响亮拒绝；fold_all() 后可落盘且重载逐位一致
# 段在场时落盘的 npz 内部不一致（`dl` 长 `N` 含段行，而 `T` 只有主表行）→ 重载会崩。
# 覆盖 `fold_all()`：多段一次性折叠（`fold_segment` 的不变量"折完 T 行数==N"在多段下不成立）
def _mkseg22(bm, tag, texts_, new_terms_):
    V0 = len(bm.idx2term)
    t2i = {w: i for i, w in enumerate(bm.idx2term)}
    ri, ci, da = [], [], []
    for i, toks in enumerate(texts_):
        for w, c in __import__("collections").Counter(toks).items():
            col = t2i.get(w)                                  # 别用 get(k, 默认)：默认值会被急切求值
            if col is None:
                col = V0 + new_terms_.index(w)
            ri.append(i)
            ci.append(col)
            da.append(float(c))
    m = sp.csc_matrix((np.asarray(da, np.float32),
                       (np.asarray(ri, np.int32), np.asarray(ci, np.int32))),
                      shape=(len(texts_), V0 + len(new_terms_)))
    d = run / f"u22_{tag}"
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / f"{tag}.data.npy", np.asarray(m.data, np.float32))   # 内核约定：`<tag>.data.npy`
    np.save(d / f"{tag}.idx.npy", np.asarray(m.indices, np.int32))
    np.save(d / f"{tag}.ptr.npy", np.asarray(m.indptr, np.int64))
    return d, np.asarray([len(t) for t in texts_], np.float32)


_pa22 = run / "u22"
_pa22.mkdir(parents=True, exist_ok=True)
_bm22 = SparseBM25(list(DOCS), tok, k1=1.5, b=0.75)
_d1, _dl1 = _mkseg22(_bm22, "s1", [["zz22a", "zz22b"]], ["zz22a", "zz22b"])
_bm22.mount_segment(str(_d1), "s1", ["zz22a", "zz22b"], _dl1, rows=1)
_d2, _dl2 = _mkseg22(_bm22, "s2", [["zz22c", DOCS[0].split()[0]]], ["zz22c"])
_bm22.mount_segment(str(_d2), "s2", ["zz22c"], _dl2, rows=1)
_ref22 = np.asarray(_bm22.score_all(QUERIES[0]), dtype=np.float32)
_save_refused, _save_msg = False, ""
try:
    _bm22.save(str(_pa22 / "bm25.npz"))
except RuntimeError as _e:                                    # 必须是响亮拒绝
    _save_refused = "fold_all" in str(_e)
    _save_msg = str(_e)[:70]
_fo22 = _bm22.fold_all()                                      # 多段一次性折叠
_after_fold = np.asarray(_bm22.score_all(QUERIES[0]), dtype=np.float32)
_d22f = (float(np.abs(_after_fold - _ref22).max())
         if _after_fold.shape == _ref22.shape else float("nan"))
_bm22.save(str(_pa22 / "bm25.npz"))                           # 折后落盘应当成功
_bm22b = SparseBM25.load(str(_pa22 / "bm25.npz"))
_after_reload = np.asarray(_bm22b.score_all(QUERIES[0]), dtype=np.float32)
_d22r = (float(np.abs(_after_reload - _ref22).max())
         if _after_reload.shape == _ref22.shape else float("nan"))
check("U22 段在场时 save() 响亮拒绝；fold_all() 后可落盘且重载逐位一致（不重复计数）",
      _save_refused and _fo22.get("rows") == 2 and _d22f == 0.0 and _d22r == 0.0
      and not _bm22b.segment_report() and int(_bm22b.N) == int(_bm22.N),
      f"两段 → save 被拒={_save_refused}（{_save_msg!r}）· fold_all rows={_fo22.get('rows')}"
      f" · 折后↔折前 {_d22f:.3e} · 重载后↔折前 {_d22r:.3e}"
      f" · 重载后已挂段={sorted(_bm22b.segment_report())} · N {int(_bm22.N)}→{int(_bm22b.N)}")

print("-" * 60)
print("ALL PASS" if not FAILS else "FAILURES: " + ", ".join(FAILS))
raise SystemExit(1 if FAILS else 0)
