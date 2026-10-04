
from __future__ import annotations

import mmap as _mmap
from array import array
from pathlib import Path

BLOB = "vocab.blob"
OFFS = "vocab_offs.u32"
SORTED = "vocab_sorted.u32"
HOT = "vocab_hot.u32"


def write(terms, outdir, hot_ids=None) -> dict:
    """把 `terms`（id 序）与可选热点 id 落盘；返回字节账。"""
    d = Path(outdir)
    d.mkdir(parents=True, exist_ok=True)
    enc = [t.encode("utf-8") for t in terms]
    offs = array("I", [0])
    buf = bytearray()
    for e in enc:
        buf += e
        offs.append(len(buf))
    assert offs.itemsize == 4, "本平台 uint32 不是 4 字节 → 布局失效"
    sorted_ids = array("I", sorted(range(len(terms)), key=lambda i: enc[i]))
    (d / BLOB).write_bytes(bytes(buf))
    (d / OFFS).write_bytes(offs.tobytes())
    (d / SORTED).write_bytes(sorted_ids.tobytes())
    if hot_ids is not None:
        with open(d / HOT, "wb") as f:
            array("I", list(hot_ids)).tofile(f)
    return dict(V=len(terms), blob=len(buf), offs=4 * len(offs), sorted_ids=4 * len(sorted_ids),
                total=len(buf) + 4 * len(offs) + 4 * len(sorted_ids))


class VocabMmap:
    """只读、文件后备的词表；接口对齐原 `list`（`__getitem__`/`__len__`/`__iter__`）
    与 `dict`（`get`/`__getitem__`/`__contains__`）。"""

    def __init__(self, outdir, hot_ids=None):
        self.dir = Path(outdir)
        self._fh = {}
        self.blob = self._map(BLOB)
        self.offs = memoryview(self._map(OFFS)).cast("I")
        self.sorted_ids = memoryview(self._map(SORTED)).cast("I")
        self.n = len(self.offs) - 1
        self.hot: dict[str, int] = {}
        if hot_ids is None:
            p = self.dir / HOT
            if p.exists() and p.stat().st_size:
                a = array("I")
                a.frombytes(p.read_bytes())
                hot_ids = a
        if hot_ids is not None:
            self.attach_hot(hot_ids)

    def _map(self, name):
        f = open(self.dir / name, "rb")
        self._fh[name] = f
        return _mmap.mmap(f.fileno(), 0, access=_mmap.ACCESS_READ)

    def attach_hot(self, hot_ids):
        """ 两阶段装配：拿到 `df` 之后再决定热点。"""
        for i in hot_ids:
            self.hot[self.term_of(int(i))] = int(i)
        return len(self.hot)

    # ---------------------------------------------------------------- 查找
    def term_of(self, i: int) -> str:
        a = self.offs[i]
        b = self.offs[i + 1]
        return self.blob[a:b].decode("utf-8")

    def id_of(self, term: str):
        h = self.hot.get(term)
        if h is not None:
            return h
        key = term.encode("utf-8")
        lo, hi = 0, self.n
        blob, offs, sid = self.blob, self.offs, self.sorted_ids
        while lo < hi:
            mid = (lo + hi) >> 1
            i = sid[mid]
            v = blob[offs[i]:offs[i + 1]]
            if v < key:
                lo = mid + 1
            elif v > key:
                hi = mid
            else:
                return int(i)
        return None

    # ------------------------------------------------ list / dict 兼容面
    def __len__(self) -> int:
        return self.n

    def __getitem__(self, k):
        if isinstance(k, str):
            i = self.id_of(k)
            if i is None:
                raise KeyError(k)
            return i
        return self.term_of(k)

    def __contains__(self, k) -> bool:
        return self.id_of(k) is not None

    def __iter__(self):
        for i in range(self.n):
            yield self.term_of(i)

    def get(self, term, default=None):
        i = self.id_of(term)
        return default if i is None else i

    def close(self):
        for f in self._fh.values():
            try:
                f.close()
            except Exception:
                pass
        self._fh.clear()

    @property
    def bytes_on_disk(self) -> int:
        return int(sum(p.stat().st_size for p in self.dir.glob("*") if p.is_file()))
