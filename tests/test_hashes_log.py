#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""X14：哈希账本追加式（hashes.jsonl 事件日志）契约测试。

判据：
  H1  build ⇒ 只有快照（hashes.json），无事件日志
  H2  delta 模式改一个文件 ⇒ 事件日志追加 set 行；快照字节不动
  H3  加一个文件 + 删一个文件 ⇒ set + del 事件各就位
  H4  重载回放 ⇒ hashes/stamps 与内存真值逐一相等；且 verify="stat" 生效（noop）
  H5  事件攒过 HASHES_COMPACT_N ⇒ 快照整理一次、日志清空、状态仍对
  H6  npz 模式（默认）行为不变：整份快照、无日志（兼容面）
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import primolix.core as core                    # noqa: E402
from primolix import Primolix                   # noqa: E402
from primolix.core import set_persist_mode      # noqa: E402


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="pl_hashes_"))
    n = 0
    try:
        d = tmp / "docs"
        d.mkdir()
        # ⚠️ 每段必须 ≥ 20 字：chunk_prose 对短段**静默丢弃**（core.py 内联 len<20），
        #    玩具语料太短会让 scan 返回空、update 误判 noop（本测试第一版就栽在这）。
        docs = {
            "f0.md": "真增量 原语 的内核说明，第一篇讲索引与可逆性。",
            "f1.md": "面包 的 做法：面粉、酵母与烤箱的温度细节。",
            "f2.md": "旅行 计划：火车、酒店与行李的预算安排。",
        }
        for k, t in docs.items():
            (d / k).write_text(t + "\n", encoding="utf-8")
        z = Primolix.build(str(d), str(tmp / "idx"))
        assert int(z._bm.N) >= 3, "语料全部被切块丢弃 —— 测试语料需 ≥20 字/段"
        snap = tmp / "idx" / "hashes.json"
        log = tmp / "idx" / "hashes.jsonl"

        # H1
        assert snap.exists() and not log.exists(), "H1: build 后应有快照无日志"
        n += 1

        set_persist_mode("delta")
        (d / "f1.md").write_text("面包 的 做法：面粉、酵母与烤箱的温度细节，改写后的版本。\n",
                                encoding="utf-8")
        snap_bytes = snap.read_bytes()
        z.update(str(d))
        assert log.exists(), "H2: delta 更新后应有事件日志"
        lines = [json.loads(x) for x in log.read_text(encoding="utf-8").splitlines() if x]
        assert any(r.get("rel") == "f1.md" for r in lines), "H2: 缺 f1.md 的 set 事件"
        assert snap.read_bytes() == snap_bytes, "H2: 快照字节不应动"
        n += 1

        # H3：加 f3.md、删 f0.md
        (d / "f3.md").write_text("新增 的 文档：关于内核分词与词表的讨论。\n", encoding="utf-8")
        (d / "f0.md").unlink()
        z.update(str(d))
        lines = [json.loads(x) for x in log.read_text(encoding="utf-8").splitlines() if x]
        assert any(r.get("del") == "f0.md" for r in lines), "H3: 缺 f0.md 的 del 事件"
        assert any(r.get("rel") == "f3.md" for r in lines), "H3: 缺 f3.md 的 set 事件"
        n += 1

        # H4：重载回放对账（真值 = 内存态在重载前留档）
        truth_h, truth_s = dict(z._hashes), dict(z._stamps)
        z2 = Primolix(str(tmp / "idx"))
        z2._load()
        assert z2._hashes == truth_h, "H4: 回放后 hashes 不等"
        assert z2._stamps == truth_s, "H4: 回放后 stamps 不等"
        res = z2.update(str(d))
        assert res["kind"] == "noop" and res.get("verify") in ("stat", "auto"), \
            "H4: stat 快扫应给出 noop"
        n += 1

        # H5：人为攒事件过阈值 ⇒ 快照整理 + 日志清空
        # ⚠️ 场景设计：此时已有段挂载，若走 staged 会先 fold_first ⇒ compact 压实快照并
        #    **重置事件计数**（语义正确）——为确定性越过阈值，这里用**词表内**小改
        #    （incremental 路由，不折叠不挂段）：重载已计 3 条事件，阈值设 2，改一次即 4>2。
        core.HASHES_COMPACT_N = 2
        (d / "f1.md").write_text("面包 的 做法：烤箱 温度 与 面粉 酵母 细节 的 安排。\n",
                                 encoding="utf-8")
        res5 = z2.update(str(d))
        assert res5["kind"] == "incremental", f"H5 前置：应为词表内增量，实得 {res5['kind']}"
        core.HASHES_COMPACT_N = 4096
        assert not (tmp / "idx" / "hashes.jsonl").exists(), \
            "H5: 过阈值应快照整理并清日志"
        z3 = Primolix(str(tmp / "idx"))
        z3._load()
        assert z3._hashes == z2._hashes, "H5: 整理后重载状态不等"
        n += 1

        # H6：npz 模式（默认进程态切回）⇒ 整份快照、无日志
        set_persist_mode("npz")
        (d / "f1.md").write_text("面包 的 做法：温度 细节 与 安排 的 面包 酵母 烤箱。\n",
                                 encoding="utf-8")
        z3.update(str(d))
        assert (tmp / "idx" / "hashes.json").exists()
        assert not (tmp / "idx" / "hashes.jsonl").exists(), "H6: npz 模式不应留事件日志"
        z4 = Primolix(str(tmp / "idx"))
        z4._load()
        assert z4._hashes == z3._hashes, "H6: npz 模式重载状态不等"
        n += 1

        print(f"ALL PASS ({n} checks)")
        return 0
    except AssertionError as e:
        print(f"FAIL: {e}")
        return 1
    finally:
        set_persist_mode("npz")
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
