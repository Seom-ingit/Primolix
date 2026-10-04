#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""primolix.cli —— Primolix 零编码检索命令行入口。

用法：`python -m primolix <index|query|update|browse|repl|info|params|config|segment|view|compact|explain>`
（不带子命令 = repl）。索引目录由 `--out` 指定，缺省按 `.primolix_index`/`.zelix_index`/`.idx` 自动找。
Python API 用 `primolix.core.Primolix` / `primolix.api.Index`；本模块只负责命令行与展示。
"""
import argparse
import json
import re as _re
import sys
import time
from pathlib import Path

import numpy as np

from .core import Primolix, get_dense_model

# ---- 可选 Rich (无 rich 降级纯文本) ----
try:
    from rich.console import Console
    from rich.text import Text
    RICH = True
    _c = Console()
except ImportError:
    RICH = False
    _c = None


def _note(msg):
    if RICH:
        _c.print(Text(f"● {msg}", style="yellow"))
    else:
        print(f"[提示] {msg}")


def _ok(msg):
    if RICH:
        _c.print(Text(f"✔ {msg}", style="green"))
    else:
        print(f"[OK] {msg}")


def _err(msg):
    if RICH:
        _c.print(Text(f"✖ {msg}", style="red"))
    else:
        print(f"[错误] {msg}")


def _warn(msg):
    if RICH:
        _c.print(Text(f" {msg}", style="yellow"))
    else:
        print(f"[警告] {msg}")


def _snip(text, n=150):
    s = text.replace('\n', ' ').strip()
    return s if len(s) <= n else s[:n] + '…'


def _hl_query(text, query):
    """高亮查询词（本函数只做纯文本；rich 路径在 `_show_hits` 里单独做）。"""
    return text


def _show_hits(hits, q, dt, tag, k=0):
    shown = k or len(hits)
    if RICH:
        _c.rule(f" top-{shown} · {tag} · {dt:.0f}ms ", style="cyan")
        if not hits:
            _c.print(Text('   (无结果) 试试换关键词', style='yellow'))
            return
        for h in hits:
            row = Text('  ')
            row.append(f"#{h['rank']:>2} ", style='bold green')
            row.append(f"{h['score']:.3f} ", style='dim')
            row.append(f"{h['file']}:{h['line']}", style='bold cyan')
            _c.print(row)
            _c.print(Text('     ') + Text(_snip(h['text'])))
        _c.print()
    else:
        print(f"[{tag} {dt:.0f}ms] top-{shown}:")
        if not hits:
            print('  (无结果)')
            return
        for h in hits:
            print(f"  #{h['rank']} score={h['score']:.3f} [{h['file']}:{h['line']}]")
            print(f"      {_snip(h['text'])}")


def _show_browse(chunks, total, page, n, files=None):
    if RICH:
        if files is not None:
            _c.print(Text(f"索引 {total} chunks / {len(files)} 文件", style='cyan'))
            for f in files:
                _c.print('  ' + Text(f, style='green'))
            return
        if not chunks:
            _c.print(Text('   空页 / 无内容', style='yellow'))
            return
        from rich.table import Table
        from rich import box as _box
        tbl = Table(box=_box.ROUNDED, title_justify='left')
        tbl.add_column('文件', style='cyan', no_wrap=True)
        tbl.add_column('行', justify='right', style='dim')
        tbl.add_column('内容')
        for c in chunks:
            tbl.add_row(c['file'], str(c['line']), _snip(c['text'], 120))
        _c.print(tbl)
    else:
        if files is not None:
            print(f"索引 {total} chunks / {len(files)} 文件")
            for f in files:
                print(f'  {f}')
            return
        print(f'浏览 {len(chunks)} 条 (共 {total}):')
        for c in chunks:
            print(f"  [{c['file']}:{c['line']}] {_snip(c['text'], 100)}")


def _find_index(out_arg):
    """解析索引目录: 显式 --out > 当前目录 .primolix_index/.idx > 提示。"""
    if out_arg:
        return str(Path(out_arg).resolve())
    for cand in ('.primolix_index', '.zelix_index', '.idx'):
        if (Path(cand) / 'bm25.npz').exists():
            return str(Path(cand).resolve())
    return None


def _load(out):
    if not out:
        _err('未找到索引. 先: python -m primolix index <文件夹> [--out <索引>]')
        raise SystemExit(1)
    if not (Path(out) / 'bm25.npz').exists():
        _err(f'索引不存在: {out}. 先 index 建库。')
        raise SystemExit(1)
    z = Primolix(out)
    z._load()
    return z


def _cmd_index(args):
    Primolix.build(args.root, args.out, dense=args.dense,
                   model_path=args.model, workers=args.workers,
                   w_cache=bool(getattr(args, "w_cache", False)))   # 物化层 w 落盘（opt-in）
    _ok('建索引完成')


def _cmd_query(args):
    if not args.q:
        _err('缺少查询词: python -m primolix query "词"')
        raise SystemExit(1)
    z = _load(_find_index(args.out))
    z._model_path = args.model or ''
    if args.dense and z._vecs is None and (z.dir / 'dense.npy').exists():
        z._vecs = np.load(z.dir / 'dense.npy')
    if args.dense and z._vecs is None:
        _note('索引无 dense.npy, 用 index --dense 重建')
    t0 = time.time()
    hits = z.query(args.q, k=args.k or 10, dense=args.dense)
    dt = (time.time() - t0) * 1000
    tag = 'BM25+dense' if (args.dense and z._vecs is not None) else 'BM25'
    _show_hits(hits, args.q, dt, tag, k=args.k or 10)


def _cmd_update(args):
    out = _find_index(args.out)
    z = _load(out)
    z._dense = args.dense
    root = args.root
    res = z.update(root, workers=args.workers)
    if res['kind'] == 'noop':
        _ok(res['detail'])
    elif res['kind'] == 'incremental':
        _ok(f"真增量更新完成 ({res['detail']})")
    else:
        _ok(f"整库回退重建完成 (原因: {res['detail']})")


def _cmd_browse(args):
    z = _load(_find_index(args.out))
    if args.files:
        files = z.list_files()
        _show_browse(None, z.info['N'], 0, 0, files=files)
        return
    chunks, total = z.browse(file=args.file, page=args.page, n=args.n)
    _show_browse(chunks, total, args.page, args.n)


def _cmd_repl(args):
    z = _load(_find_index(args.out))
    print(f"[primolix] {z.info['N']} chunks 已加载 (BM25). 输入关键词检索, q 退出。")
    k = args.k or 10
    while True:
        try:
            line = input('> ').strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line.lower() in ('q', 'exit', 'quit'):
            break
        t0 = time.time()
        hits = z.query(line, k=k, dense=False)
        dt = (time.time() - t0) * 1000
        _show_hits(hits, line, dt, 'BM25', k=k)


# ================================================================ 门面命令
# 这一组走 `primolix.api.Index`（并发安全默认开）→ 命令行与库调用是同一条路，不必摸私有字段。

def _idx(args):
    """按 `--out` 打开门面对象（找不到索引就明确报错，不猜）。"""
    from .api import Index
    out = _find_index(getattr(args, "out", None))
    if not out:
        _err('未找到索引. 先: python -m primolix index <文件夹> [--out <索引>]')
        raise SystemExit(1)
    return Index(out, dense=bool(getattr(args, "dense", False)))


def _show_json(d) -> None:
    print(json.dumps(d, ensure_ascii=False, indent=1, default=str))


def _load_terms(p: str) -> list:
    """段带来的新词：JSON 数组 或 一行一词。"""
    t = Path(p).read_text(encoding="utf-8").strip()
    if t.startswith('['):
        return [str(x) for x in json.loads(t)]
    return [ln.strip() for ln in t.splitlines() if ln.strip()]


def _load_dl(p: str):
    """段各行的文档长度：`.npy` 或 JSON 数组。"""
    if str(p).endswith('.npy'):
        return np.load(p)
    return np.asarray(json.loads(Path(p).read_text(encoding="utf-8")), dtype=np.float32)


def _cmd_info(args):
    _show_json(_idx(args).stats())


def _cmd_params(args):
    idx = _idx(args)
    if args.k1 is None and args.b is None:
        s = idx.stats()
        _show_json(dict(k1=s["k1"], b=s["b"], avgdl=s["avgdl"],
                        note="改参数：--k1/--b（不重建）；--save 写回索引文件"))
        return
    _show_json(idx.tune(k1=args.k1, b=args.b, persist=bool(args.save)))


def _cmd_config(args):
    from .kernel import SparseBM25
    from .api import SWITCHES, set_switches
    if not args.set:
        _show_json({k: getattr(SparseBM25, k) for k in SWITCHES})
        return
    kw = {}
    for kv in args.set:
        if '=' not in kv:
            _err(f"--set 需要 K=V 形式：{kv}")
            raise SystemExit(1)
        k, v = (x.strip() for x in kv.split('=', 1))
        if k not in SWITCHES:
            _err(f"未知开关 {k}；可选 {list(SWITCHES)}")
            raise SystemExit(1)
        cur = getattr(SparseBM25, k)
        if isinstance(cur, bool):
            kw[k] = v.lower() in ('1', 'true', 'yes', 'on')
        elif isinstance(cur, float):
            kw[k] = float(v)
        elif isinstance(cur, int):
            kw[k] = int(v)
        else:
            kw[k] = v
    _show_json(set_switches(**kw))          # 进程级开关：不需要索引


def _cmd_segment(args):
    idx = _idx(args)
    if args.action == 'list':
        _show_json(idx.segments())
    elif args.action == 'fold':
        _show_json(idx.fold(args.tag))
    elif args.action == 'unmount':
        _show_json(idx.unmount(args.tag))
    else:
        if not (args.tag and args.seg_dir and args.terms_file and args.dl_file):
            _err('mount 需要 --tag --seg-dir --terms-file --dl-file（可选 --rows --generation）')
            raise SystemExit(1)
        _show_json(idx.mount(args.seg_dir, args.tag, _load_terms(args.terms_file),
                             _load_dl(args.dl_file), rows=args.rows,
                             generation=args.generation))


def _cmd_view(args):
    idx = _idx(args)
    if args.action == 'list':
        _show_json(idx.views())
    elif args.action == 'detach':
        _show_json(idx.detach_view(args.name))
    elif args.action == 'attach':
        if not (args.name and args.vectors):
            _err('attach 需要 --name 与 --vectors（可选 --positions --replace）')
            raise SystemExit(1)
        pos = np.load(args.positions) if args.positions else None
        _show_json(idx.attach_view(args.name, np.load(args.vectors), pos,
                                   replace=bool(args.replace)))
    else:
        if not (args.name and args.qvec and args.q):
            _err('fuse 需要 --name --qvec 与查询词')
            raise SystemExit(1)
        for h in idx.fuse(args.q, np.load(args.qvec), args.name,
                          alpha=args.alpha, k=args.k or 10):
            print(f"  #{h['rank']} {h['score']:.4f} [{h['file']}:{h['line']}] "
                  f"{_snip(h['text'])}")


def _cmd_compact(args):
    _show_json(_idx(args).compact())


def _cmd_explain(args):
    rows = _idx(args).explain(args.q, docid=args.docid, k=args.k or 1)
    print(f"查询 {args.q!r} 的逐词贡献（{len(rows)} 项）：")
    for r in rows:
        print(f"  {r['term']:<14} tf={r['tf']:>5.1f}  df={r['df']:>7.0f}  "
              f"idf={r['idf']:6.3f}  贡献={r['contribution']:9.4f}")


def main(argv=None):
    # Windows 控制台编码保险
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    ap = argparse.ArgumentParser(prog='primolix', description='Primolix 零编码检索')
    sub = ap.add_subparsers(dest='cmd')
    sp = sub.add_parser('index', help='建索引')
    sp.add_argument('root')
    sp.add_argument('--out', default='.primolix_index')
    sp.add_argument('--dense', action='store_true')
    sp.add_argument('--model', default='')
    sp.add_argument('--workers', type=int, default=0)
    sp.add_argument('--w-cache', dest='w_cache', action='store_true',
                    help='同时把物化层 w 落盘（冷启动首查快 ~77×；盘 +nnz×4 B，1M 段约 +165MB）')
    sp = sub.add_parser('query', help='检索')
    sp.add_argument('q')
    sp.add_argument('--out', default=None)
    sp.add_argument('--k', type=int, default=10)
    sp.add_argument('--dense', action='store_true')
    sp.add_argument('--model', default='')
    sp = sub.add_parser('update', help='增量同步 (方案A真增量)')
    sp.add_argument('root')
    sp.add_argument('--out', default=None)
    sp.add_argument('--dense', action='store_true')
    sp.add_argument('--workers', type=int, default=0)
    sp = sub.add_parser('browse', help='浏览索引')
    sp.add_argument('--out', default=None)
    sp.add_argument('--file', default=None)
    sp.add_argument('--page', type=int, default=0)
    sp.add_argument('--n', type=int, default=20)
    sp.add_argument('--files', action='store_true')
    sp = sub.add_parser('repl', help='交互式检索')
    sp.add_argument('--out', default=None)
    sp.add_argument('--k', type=int, default=10)
    # ---- 门面命令（把繁琐的工程 API 收进 CLI）----
    sp = sub.add_parser('info', help='索引概览（规模 / 开关 / 段 / 视图）')
    sp.add_argument('--out', default=None)
    sp = sub.add_parser('params', help='查看 / 修改 k1、b（不重建）')
    sp.add_argument('--out', default=None)
    sp.add_argument('--k1', type=float, default=None)
    sp.add_argument('--b', type=float, default=None)
    sp.add_argument('--save', action='store_true', help='把新参数写回索引文件')
    sp = sub.add_parser('config', help='查看 / 修改开关（进程级）')
    sp.add_argument('--out', default=None)
    sp.add_argument('--set', action='append', default=None, metavar='K=V',
                    help='可重复，如 --set W_CACHE=true --set TC_SHARED=true')
    sp = sub.add_parser('segment', help='段：list / mount / unmount / fold')
    sp.add_argument('action', choices=['list', 'mount', 'unmount', 'fold'])
    sp.add_argument('--out', default=None)
    sp.add_argument('--tag', default=None)
    sp.add_argument('--seg-dir', dest='seg_dir', default=None)
    sp.add_argument('--terms-file', dest='terms_file', default=None, help='新词：JSON 数组或一行一词')
    sp.add_argument('--dl-file', dest='dl_file', default=None, help='段各行长度：.npy 或 JSON 数组')
    sp.add_argument('--rows', type=int, default=None)
    sp.add_argument('--generation', default=None)
    sp = sub.add_parser('view', help='视图：list / attach / detach / fuse')
    sp.add_argument('action', choices=['list', 'attach', 'detach', 'fuse'])
    sp.add_argument('--out', default=None)
    sp.add_argument('--name', default=None)
    sp.add_argument('--vectors', default=None, help='attach：.npy，[n, dim]')
    sp.add_argument('--positions', default=None, help='attach：.npy，meta 位置（缺省=全部活跃 chunk）')
    sp.add_argument('--replace', action='store_true')
    sp.add_argument('--qvec', default=None, help='fuse：查询向量 .npy')
    sp.add_argument('--alpha', type=float, default=0.35)
    sp.add_argument('--k', type=int, default=10)
    sp.add_argument('q', nargs='?', default=None)
    sp = sub.add_parser('compact', help='压实：把已挂段并进主表（查询恢复单块）')
    sp.add_argument('--out', default=None)
    sp = sub.add_parser('explain', help='归因：把分数按词拆开')
    sp.add_argument('q')
    sp.add_argument('--out', default=None)
    sp.add_argument('--k', type=int, default=1, help='拆第 k 名（默认第 1 名）')
    sp.add_argument('--docid', default=None, help='按文件名片段指定文档')
    args = ap.parse_args(argv)
    if args.cmd == 'index':
        _cmd_index(args)
    elif args.cmd == 'query':
        _cmd_query(args)
    elif args.cmd == 'update':
        _cmd_update(args)
    elif args.cmd == 'browse':
        _cmd_browse(args)
    elif args.cmd == 'repl':
        _cmd_repl(args)
    elif args.cmd == 'info':
        _cmd_info(args)
    elif args.cmd == 'params':
        _cmd_params(args)
    elif args.cmd == 'config':
        _cmd_config(args)
    elif args.cmd == 'segment':
        _cmd_segment(args)
    elif args.cmd == 'view':
        _cmd_view(args)
    elif args.cmd == 'compact':
        _cmd_compact(args)
    elif args.cmd == 'explain':
        _cmd_explain(args)
    else:
        ap.print_help()
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
