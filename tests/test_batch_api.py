#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""X12：`Index.search_batch` / `Primolix.query_many` 契约测试。

判据：
  B1  batch([q]) 与 search(q) 逐字段相等（含 text）
  B2  with_text=False ⇒ 无 'text' 键，其余字段仍与 search 对应行相等
  B3  空列表 → 空列表；无命中查询 → 空命中列表（与 search 一致）
  B4  tune 之后 batch 仍反映新参数（k1/b 属性路径不被 batch 缓存）
  B5  dense 臂逐条等价 search(q, dense=True)（无模型时两边同为 BM25-only）
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # zelix_research/

import primolix                                        # noqa: E402

CORPUS = {
    "a.md": "真增量 原语 的 说明，第一篇讲内核与索引。",
    "b.md": "面包 的 做法：面粉 酵母 烤箱，真增量 面包。",
    "c.md": "旅行 计划：火车 酒店 行李，预算 与 路线。",
    "d.md": "内核 分数 的 可逆性：不把权重烤进索引。",
}

def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="pl_batch_"))
    n_checks = 0
    ok = True
    try:
        # 建小语料（自包含，不走 fixtures）
        cdir = tmp / "docs"
        cdir.mkdir()
        for name, text in CORPUS.items():
            (cdir / name).write_text(text + "\n", encoding="utf-8")
        idx = primolix.build(str(cdir), str(tmp / "idx"))

        qs = ["真增量 原语", "面包 做法", "不存在的词xyz"]

        # B1
        batch = idx.search_batch(qs)
        single = [idx.search(q) for q in qs]
        assert batch == single, "B1: batch 与逐条 search 不相等"
        n_checks += 1

        # B2
        bt = idx.search_batch(qs, with_text=False)
        for row_b, row_s in zip(bt, single):
            for hb, hs in zip(row_b, row_s):
                assert "text" not in hb, "B2: with_text=False 仍带 text"
                d = dict(hs); d.pop("text")
                assert hb == d, "B2: 除 text 外字段不相等"
                n_checks += 1

        # B3
        assert idx.search_batch([]) == []
        n_checks += 1
        assert idx.search_batch(["不存在的词xyz"]) == [idx.search("不存在的词xyz")], \
               "B3: 单条批量与 search 不一致（注意两边都是列表的列表）"
        n_checks += 1

        # B4：tune 后 batch 反映新参数
        s_old = idx.search_batch(["真增量 原语"])[0]
        idx.tune(k1=2.0, b=0.1)
        s_new = idx.search_batch(["真增量 原语"], with_text=False)[0]
        assert s_new, "B4: tune 后 batch 无命中"
        assert abs(s_new[0]["score"] - idx.search("真增量 原语")[0]["score"]) < 1e-12, \
            "B4: batch 与 search 在 tune 后不一致"
        n_checks += 1
        idx.tune(k1=1.5, b=0.75)

        # B5：dense 臂（无模型 ⇒ 两边同为 BM25-only）
        db = idx.search_batch(["真增量 原语"], dense=True)
        ds = idx.search("真增量 原语", dense=True)
        assert db == [ds], "B5: dense 批量臂与逐条不一致"
        n_checks += 1

        print(f"ALL PASS ({n_checks} checks)")
        return 0
    except AssertionError as e:
        print(f"FAIL: {e}")
        ok = False
        return 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        if ok:
            print("test_batch_api 完成")


if __name__ == "__main__":
    raise SystemExit(main())
