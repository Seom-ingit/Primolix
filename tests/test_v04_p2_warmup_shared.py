#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""0.4 P2 回归测试：Index.warmup(shared=True) 父进程预热与子进程零拷贝 attach + health() 巡检。

Run:  python tests/test_v04_p2_warmup_shared.py
"""
from __future__ import annotations

import atexit
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import primolix                                    # noqa: E402
from primolix import core, kernel, tc_shared       # noqa: E402

core.VERBOSE = False
kernel.VERBOSE = False

FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  |  {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


def main() -> int:
    work = HERE / "_v04_p2_run"
    shutil.rmtree(work, ignore_errors=True)
    atexit.register(shutil.rmtree, work, ignore_errors=True)

    docs = work / "docs"
    idx_dir = work / "idx"
    docs.mkdir(parents=True)
    for i in range(6):
        (docs / f"doc{i}.md").write_text(
            f"共享内存 预热 测试 文档 {i}：多进程 只读 挂载 与 零拷贝 词表。\n",
            encoding="utf-8",
        )

    with primolix.build(str(docs), str(idx_dir)) as idx0:
        pass

    tc_key_used = None
    try:
        with primolix.open(str(idx_dir)) as parent:
            w = parent.warmup(shared=True)
            tc_key_used = w.get("tc_key")
            h = parent.health()
            parent_scores = [
                round(float(hit["score"]), 6)
                for hit in parent.search("共享内存 多进程 词表", k=6)
            ]

            child_code = (
                "import json, sys\n"
                "sys.path.insert(0, sys.argv[1])\n"
                "import primolix\n"
                "with primolix.open(sys.argv[2]) as ch:\n"
                "    w = ch.warmup(shared=True)\n"
                "    sc = [round(float(x['score']), 6) for x in ch.search('共享内存 多进程 词表', k=6)]\n"
                "    print(json.dumps({'w': w, 'scores': sc}))\n"
            )
            proc = subprocess.run(
                [sys.executable, "-c", child_code, str(HERE.parent), str(idx_dir)],
                capture_output=True, text=True, check=True,
            )
            child_res = json.loads(proc.stdout.strip().splitlines()[-1])

        ok_p2_1 = (
            w["shared"] and w["owner"] and bool(w["name"])
            and child_res["w"]["shared"]
            and not child_res["w"]["owner"]
            and child_res["w"]["name"] == w["name"]
            and child_res["scores"] == parent_scores
        )
        check(
            "P2-1 parent warmup(shared=True) lets child attach same shm segment with identical scores",
            ok_p2_1,
            f"name={w.get('name')} parent_owner={w.get('owner')} child_owner={child_res['w'].get('owner')} eq={child_res['scores'] == parent_scores}",
        )
        ok_p2_2 = (
            h["vocab_carrier"] == "VocabMmap"
            and h["tc_shared"]["shared"] is True
            and h["tc_shared"]["name"] == w["name"]
            and primolix.__version__ == "0.4.0"
        )
        check(
            "P2-2 health() reports vocab_carrier & tc_shared status and version is 0.4.0",
            ok_p2_2,
            f"carrier={h.get('vocab_carrier')} tc_shared={h.get('tc_shared')} ver={primolix.__version__}",
        )
    finally:
        if tc_key_used:
            tc_shared.cleanup(tc_key_used)
        shutil.rmtree(work, ignore_errors=True)

    print("-" * 60)
    print(f"{'ALL PASS' if not FAILS else 'FAILURES: ' + ', '.join(FAILS)}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
