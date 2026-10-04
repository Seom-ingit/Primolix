#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""End-to-end regression for the primolix package. No external data needed.

Run:  python tests/test_primolix_e2e.py

Checks (each prints PASS/FAIL; process exits 1 if anything fails):
  1a every module parses
  1b the package imports
  2  Primolix.build produces an index directory
  3  query returns ranked hits (score descending), expected file first
  4  kernel invariant: update_docs == full rebuild, scores bit-exact
  5  formula change: assigning k1/b == fresh build with those params, bit-exact
  6  index level true-incremental: a file edit is picked up
  7  out-of-vocabulary term: update or rebuild_in_place, then findable
  8  delete: a removed file is no longer listed / findable
  9  reload from disk is deterministic (same files and scores)

  Not covered here: the view family (see docs/api.md).
"""
from __future__ import annotations

import ast
import atexit
import os
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PKG = HERE.parent / "primolix"
sys.path.insert(0, str(HERE.parent))

FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  |  {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


# ---------------------------------------------------------------- 1 parse + import
mods = sorted(PKG.glob("*.py"))
bad = []
for p in mods:
    try:
        ast.parse(p.read_text(encoding="utf-8"))
    except SyntaxError as e:
        bad.append(f"{p.name}: {e}")
check("1a every module parses", not bad, f"{len(mods)} modules" + ("; " + "; ".join(bad) if bad else ""))

try:
    import primolix
    from primolix import Primolix
    from primolix.kernel import SparseBM25
    check("1b package imports", True, f"primolix -> {Path(primolix.__file__).name}")
except Exception as e:                                     # pragma: no cover
    check("1b package imports", False, repr(e))
    print("\n".join(FAILS))
    raise SystemExit(1)

# ---------------------------------------------------------------- corpus
# NOTE: build the scratch corpus next to this file (or under $PRIMOLIX_E2E_DIR).
# The system temp dir may be non-writable under a sandbox.
work = Path(os.environ.get("PRIMOLIX_E2E_DIR") or (HERE / "_e2e_run"))
shutil.rmtree(work, ignore_errors=True)
atexit.register(shutil.rmtree, work, ignore_errors=True)   # clean up even on early exit
root = work / "docs"
root.mkdir(parents=True)
(root / "a.txt").write_text("kernel incremental retrieval engine\n" * 3, encoding="utf-8")
(root / "b.txt").write_text("cooking recipe for bread and soup\n" * 3, encoding="utf-8")
(root / "c.txt").write_text("bm25 sparse index and postings\n" * 3, encoding="utf-8")
idx = work / "idx"

# ---------------------------------------------------------------- 2 build
try:
    z = Primolix.build(str(root), str(idx))
    check("2 Primolix.build", idx.exists() and any(idx.iterdir()),
          "files: " + ", ".join(sorted(p.name for p in idx.iterdir())[:6]))
except Exception as e:
    check("2 Primolix.build", False, repr(e))
    raise SystemExit(1)

# ---------------------------------------------------------------- 3 query
hits = z.query("kernel", k=3)
ok3 = bool(hits) and hits[0]["file"].endswith("a.txt") and \
    all(hits[i]["score"] >= hits[i + 1]["score"] for i in range(len(hits) - 1))
check("3 query ranks expected file first", ok3,
      f"top={hits[0]['file'].split(chr(92))[-1] if hits else None} score={hits[0]['score']:.3f}" if hits else "no hits")

# ---------------------------------------------------------------- 4 kernel: update == rebuild
texts = ["kernel incremental retrieval engine", "cooking recipe for bread",
         "bm25 sparse index and postings", "retrieval engine kernel"]
bm = SparseBM25(list(texts), lambda s: s.split(), k1=1.5, b=0.75)
qm = ["kernel", "retrieval"]
before = np.asarray(bm.score_all(qm), dtype=np.float32)
mod_rows = [0]
new_toks = [["retrieval", "engine", "kernel", "kernel", "sparse"]]
bm.update_docs(mod_rows, new_toks)
after = np.asarray(bm.score_all(qm), dtype=np.float32)
ref_texts = list(texts)
ref_texts[0] = " ".join(new_toks[0])
ref = SparseBM25(list(ref_texts), lambda s: s.split(), k1=1.5, b=0.75)
want = np.asarray(ref.score_all(qm), dtype=np.float32)
# update_docs appends new rows and tombstones the old ones, so align by document identity:
#   updated -> new content at index N-1 (appended), untouched rows 1..N-2; rebuild -> replaced content at index 0
_new = float(after[len(texts)])
_kept = after[1:len(texts)]
_ref_new = float(want[0])
_ref_kept = want[1:len(texts)]
d4 = max(abs(_new - _ref_new), float(np.abs(_kept - _ref_kept).max()))
check("4 update_docs == rebuild (bit exact, aligned by document)",
      d4 == 0.0,
      f"max|d|={d4:.3e}; tombstoned row0 score={float(after[0]):.3f}; "
      f"kept rows changed: {int((_kept != before[1:len(texts)]).sum())}")

# ---------------------------------------------------------------- 5 formula change
bm.k1, bm.b = 2.0, 0.3
got = np.asarray(bm.score_all(qm), dtype=np.float32)
ref2 = SparseBM25(list(ref_texts), lambda s: s.split(), k1=2.0, b=0.3)
want2 = np.asarray(ref2.score_all(qm), dtype=np.float32)
_g_new = float(got[len(texts)])
_g_kept = got[1:len(texts)]
_r_new = float(want2[0])
_r_kept = want2[1:len(texts)]
d5 = max(abs(_g_new - _r_new), float(np.abs(_g_kept - _r_kept).max()))
check("5 k1/b change == fresh build (bit exact, aligned by document)", d5 == 0.0, f"max|d|={d5:.3e}")

# ---------------------------------------------------------------- 6 index-level incremental
# update() documents its contract: returns {'kind': 'noop'|'incremental'|'rebuild', 'detail': str}.
# An in-vocabulary edit must take the *incremental* path (assert the kind, not just "no exception").
s_before = z.query("kernel", k=1)[0]["score"]
(root / "a.txt").write_text("kernel kernel kernel kernel incremental retrieval\n" * 3, encoding="utf-8")
rep6 = z.update(str(root))
s_after = z.query("kernel", k=1)[0]["score"] if z.query("kernel", k=1) else 0.0
check("6 in-vocabulary edit takes the incremental path ('kind' contract)",
      rep6.get("kind") == "incremental" and s_after > s_before,
      f"update() -> kind={rep6.get('kind')!r} detail={rep6.get('detail')!r}; "
      f"score {s_before:.3f} -> {s_after:.3f}")

# ---------------------------------------------------------------- 7 out-of-vocabulary term
# Documented behaviour: the vocabulary is closed, so an out-of-vocabulary term triggers a full
# rebuild -- never a silent drop. Assert the reported kind and that the term is findable.
(root / "d.txt").write_text("zznovelterm appears only here\n" * 3, encoding="utf-8")
rep7 = z.update(str(root))
hits7 = z.query("zznovelterm", k=3)
check("7 out-of-vocabulary term -> documented full rebuild ('kind' contract), term findable",
      rep7.get("kind") == "rebuild" and bool(hits7) and hits7[0]["file"].endswith("d.txt"),
      f"update() -> kind={rep7.get('kind')!r} detail={rep7.get('detail')!r}; hits={len(hits7)}")

# ---------------------------------------------------------------- 8 delete
(root / "b.txt").unlink()
rep8 = z.update(str(root))          # in-vocabulary change -> incremental path
listed = [f.get("file", "") if isinstance(f, dict) else str(f) for f in (z.list_files() or [])]
hits8 = z.query("cooking", k=3)
check("8 removed file gone (and the delete took the incremental path)",
      rep8.get("kind") == "incremental"
      and not any("b.txt" in s for s in listed) and not any("b.txt" in h["file"] for h in hits8),
      f"update() -> kind={rep8.get('kind')!r}; listed={len(listed)}; cooking hits={len(hits8)}")

# ---------------------------------------------------------------- 9 reload determinism
q = "sparse index"
h1 = [(h["file"], round(h["score"], 6)) for h in z.query(q, k=3)]
z2 = Primolix(str(idx))
h2 = [(h["file"], round(h["score"], 6)) for h in z2.query(q, k=3)]
check("9 reload is deterministic", h1 == h2, f"{len(h1)} hits compared")

shutil.rmtree(work, ignore_errors=True)
print("-" * 60)
print(f"{'ALL PASS' if not FAILS else 'FAILURES: ' + ', '.join(FAILS)}")
raise SystemExit(1 if FAILS else 0)

