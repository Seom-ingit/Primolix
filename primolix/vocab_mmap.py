
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
        self._mmaps = []
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
        m = _mmap.mmap(f.fileno(), 0, access=_mmap.ACCESS_READ)
        self._mmaps.append(m)
        return m

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
        for mv in (getattr(self, "offs", None), getattr(self, "sorted_ids", None)):
            if mv is not None and hasattr(mv, "release"):
                try:
                    mv.release()
                except Exception:
                    pass
        self.offs = None
        self.sorted_ids = None
        self.blob = None
        for m in getattr(self, "_mmaps", ()):
            try:
                m.close()
            except Exception:
                pass
        if hasattr(self, "_mmaps"):
            self._mmaps.clear()
        for f in self._fh.values():
            try:
                f.close()
            except Exception:
                pass
        self._fh.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    @property
    def bytes_on_disk(self) -> int:
        return int(sum(p.stat().st_size for p in self.dir.glob("*") if p.is_file()))


class OverlayVocab:
    """分层词表代理（0.4 P1）：只读 `VocabMmap` 基座 ＋ 增量小表 `extra`。

    增量写入（`mount_segment` / `_replay_deltas`）与卸载（`unmount_segment`）
    不再把 535k+ `VocabMmap` 炸回 Python `dict`；只为新增 OOV 词分配小量内存。
    接口与 `VocabMmap`、`list`（`__len__`/`__iter__`/`__getitem__[int/slice]`）
    和 `dict`（`id_of`/`get`/`__contains__`/`__getitem__[str]`）一致。
    """

    def __init__(self, base: VocabMmap, extra_terms=None):
        self.base = base
        self.base_len = int(len(base))
        self.extra_terms: list[str] = []
        self.extra_map: dict[str, int] = {}
        if extra_terms:
            self._append_in_place(extra_terms)

    def _append_in_place(self, new_terms) -> None:
        idx = self.base_len + len(self.extra_terms)
        for t in new_terms:
            s = str(t)
            self.extra_terms.append(s)
            self.extra_map[s] = idx
            idx += 1

    def extended(self, new_terms) -> "OverlayVocab":
        """返回追加 `new_terms` 后的新分层视图（共享只读 `base`，不改写旧实例）。"""
        if not new_terms:
            return self
        return OverlayVocab(self.base, self.extra_terms + list(new_terms))

    def truncated(self, new_len: int):
        """按长度截断（用于 `unmount_segment`）；回到 `base_len` 时直接还回 `base`。"""
        n = int(new_len)
        if n == self.base_len:
            return self.base
        if n > self.base_len:
            keep = n - self.base_len
            return OverlayVocab(self.base, self.extra_terms[:keep])
        return [self.base.term_of(i) for i in range(max(n, 0))]

    def term_of(self, i: int) -> str:
        n = self.base_len + len(self.extra_terms)
        if i < 0:
            i += n
        if i < 0 or i >= n:
            raise IndexError(f"vocab index out of range: {i}")
        if i < self.base_len:
            return self.base.term_of(i)
        return self.extra_terms[i - self.base_len]

    def id_of(self, term: str):
        hit = self.extra_map.get(term)
        if hit is not None:
            return hit
        return self.base.id_of(term)

    def __len__(self) -> int:
        return self.base_len + len(self.extra_terms)

    def __getitem__(self, k):
        if isinstance(k, slice):
            return [self.term_of(i) for i in range(*k.indices(len(self)))]
        if isinstance(k, str):
            i = self.id_of(k)
            if i is None:
                raise KeyError(k)
            return i
        return self.term_of(int(k))

    def __contains__(self, k) -> bool:
        return self.id_of(k) is not None

    def __iter__(self):
        for i in range(self.base_len):
            yield self.base.term_of(i)
        for t in self.extra_terms:
            yield t

    def get(self, term, default=None):
        i = self.id_of(term)
        return default if i is None else i

    def close(self):
        """只清增量视图引用；底层 `base` 生命周期由持有者（`_vocab_carrier`）管理。"""
        self.extra_terms = []
        self.extra_map = {}

    @property
    def bytes_on_disk(self) -> int:
        return self.base.bytes_on_disk
