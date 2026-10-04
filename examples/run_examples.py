#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""primolix 示例：建索引 → 查询 → 真增量 → 换公式 → 归因。

跑法：
    python examples/run_examples.py

它会在本文件旁边建 examples/corpus/（10 个小文本，首次运行时写出）与 examples/_run/（索引与临时文件），
跑完自动清理 _run/。库默认安静（进度走 stderr）；这里显式关掉，让示例输出只剩结果。
"""
from __future__ import annotations

import atexit
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))          # 让示例直接用仓库里的包

import numpy as np
import primolix
from primolix.kernel import SparseBM25, zh_tokenize

primolix.core.VERBOSE = False                 # 库默认就会走 stderr；这里彻底静音

CORPUS = {
    "01_incremental.md": "真增量写入：改一篇文档只做那一篇的工作，不重建整库。墓碑标记旧行，新内容追加到尾部。",
    "02_primitive.md": "原语层：索引里只存 tf、df、dl 与墓碑。分数（BM25 权重）是这些原语的函数，查询时算出来。",
    "03_formula.md": "换公式：k1 与 b 只影响查询时的饱和项，因此改参数不需要重建索引，也不动物料。",
    "04_attribution.md": "归因：一篇文档的分数可以按词拆成 idf 乘饱和项乘文档长度各自的贡献。",
    "05_reconcile.md": "对账：改动之后只需核对原语（df、dl、墓碑）即可确认结果，成本与改动量成正比。",
    "06_views.md": "多视图：视图是段清单、墓碑、参数与词表代次的四元组；切视图只换清单，物料不动。",
    "07_bm25.md": "BM25 sparse retrieval kernel with inverted postings, incremental updates and lazy execution.",
    "08_cooking.md": "面包与汤的做法：面粉、水、盐、酵母，低温长时间发酵；汤用洋葱、胡萝卜和芹菜打底。",
    "09_travel.md": "旅行笔记：淡季的机票便宜，清晨的火车人少，山里的旅馆要提前订。",
    "10_other.md": "无关文本：园艺、木工、修表。这些内容只是为了验证检索能够区分主题。",
}


def materialize() -> Path:
    """写出示例语料。每次运行都重写，保证示例可重复（第 3 步会改写其中一个文件）。"""
    corpus = HERE / "corpus"
    corpus.mkdir(exist_ok=True)
    for name, text in CORPUS.items():
        (corpus / name).write_text(text + "\n", encoding="utf-8")
    return corpus


def explain(bm: SparseBM25, row: int, qtoks: list[str], top: int = 4) -> list[tuple]:
    """把一行的分数按词拆开（只用公开属性与文档里的 idf 公式）。

    分数 = sum_t idf[t] * tf*(k1+1) / (tf + k1*(1-b + b*dl/avgdl))
    """
    n = len(bm.dl)
    dl = float(bm.dl[row])
    avg = float(bm.avgdl)
    out = []
    for t in qtoks:
        c = bm.term2idx.get(t)
        if c is None:
            continue
        df = float(bm.df[c])
        idf = float(np.log((n - df + 0.5) / (df + 0.5) + 1.0))
        # 该词在这一行里的 tf：从 CSR 里取（T 是原语，不是算好的权重）
        T = bm.T.tocsr()
        tf = float(T[row, c])
        if tf <= 0:
            continue
        sat = tf * (bm.k1 + 1.0) / (tf + bm.k1 * (1.0 - bm.b + bm.b * dl / avg))
        out.append((t, idf, tf, idf * sat))
    out.sort(key=lambda x: -x[3])
    return out[:top]


def main() -> int:
    corpus = materialize()
    run = HERE / "_run"
    shutil.rmtree(run, ignore_errors=True)
    atexit.register(shutil.rmtree, run, ignore_errors=True)
    run.mkdir(parents=True)
    idx = run / "idx"
    print(f"语料：{corpus}（{len(CORPUS)} 个文件）\n")

    # 1) 建索引
    z = primolix.Primolix.build(str(corpus), str(idx))
    print(f"[1] 建索引：{z.info['N']} chunks，{z.info['size_mb']:.2f} MB\n")

    # 2) 查询
    for q in ("真增量 原语", "面包 做法"):
        hits = z.query(q, k=3)
        print(f"[2] 查询 “{q}”")
        if not hits:
            print("     (无命中)")
        for h in hits:
            print(f"     #{h['rank']}  {h['score']:6.3f}  {Path(h['file']).name}")
        print()

    # 3) 真增量：改一篇，立刻可查；比的是同一篇文档的分数
    target = corpus / "09_travel.md"
    q = "真增量 原语"
    keep = Path(z.query(q, k=1)[0]["file"]).name
    score_of = lambda name: next((h["score"] for h in z.query(q, k=5) if Path(h["file"]).name == name), 0.0)
    before_target, before_top = score_of("09_travel.md"), keep
    target.write_text("真增量 原语 索引 内核 的 说明，改写之后只处理这一篇。\n", encoding="utf-8")
    z.update(str(corpus))
    after_target = score_of("09_travel.md")
    print(f"[3] 真增量：改写 09_travel.md 后，同一篇文档的分数 {before_target:.3f} → {after_target:.3f}")
    print(f"    top-1 由 {before_top} 变为 {Path(z.query(q, k=1)[0]['file']).name}\n")

    # 4) 换公式：不重建、不落盘
    bm = z._bm                                     # 示例里用内部字段；公开 API 见 docs/api.md
    base = z.query("内核 分数", k=2)
    bm.k1, bm.b = 2.0, 0.1
    tuned = z.query("内核 分数", k=2)
    print("[4] 换公式（k1 1.5→2.0, b 0.75→0.1，仅改两个属性）")
    for a, b in zip(base, tuned):
        print(f"     {Path(a['file']).name}  {a['score']:6.3f} → {b['score']:6.3f}")
    bm.k1, bm.b = 1.5, 0.75
    print()

    # 5) 归因：为什么这篇排前面
    q = "真增量 原语"
    hit = z.query(q, k=1)[0]
    row = int(next(m["row"] for m in z._meta if m["file"] == hit["file"] and m["line"] == hit["line"]))
    print(f"[5] 归因：{Path(hit['file']).name} 的分数 {hit['score']:.3f} 来自")
    for t, idf, tf, contrib in explain(bm, row, zh_tokenize(q)):
        print(f"     {t:<8} idf={idf:5.3f}  tf={tf:4.1f}  贡献={contrib:6.3f}")
    print("\n完成。索引留在 examples/_run/idx（会自动清理）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
