from __future__ import annotations

import gc
import hashlib
import struct
import time
import zlib
from multiprocessing import shared_memory

import numpy as np
from scipy.sparse import csc_matrix

MAGIC = b"ZTCS1"
HDR = 128
_KEY_HEX = 40
_FMT = f"<5s{_KEY_HEX}sQQQQd"        # magic, key_sha1(hex40), n_data, n_ind, n_ptr, state, checksum
assert struct.calcsize(_FMT) <= HDR, "header 放不下"
_NAME_PREFIX = "primolix_tc_"
_DT_D, _DT_I, _DT_P = np.float32, np.int32, np.int32   # 发布用的固定 dtype（尺寸因此可先算）
_STATE_BUILDING, _STATE_READY = 0, 1
_WAIT_S = 3.0                        # 总等待预算（超时 → 回落私有）
_FLOOR_MS = 1.0                      # 周期下限（防热自旋）
# 标定值：换机器/换规模需重标（只用于定周期下限，不参与正确性）。
_MS_PER_NNZ = 9.2e-6


def key_sha1(key: str) -> str:
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


def name_of(key: str) -> str:
    return _NAME_PREFIX + key_sha1(key)[:16]


def _dbg(msg: str) -> None:
    import os
    if os.environ.get("PRIMOLIX_TC_DEBUG") or os.environ.get("ZELIX_TC_DEBUG"):
        import sys as _s
        print(f"[tc_shared:{os.getpid()}] {msg}", file=_s.stderr, flush=True)


class Period:
    """最近 3 次探测耗时的中位 × 1.05，每 3 次刷新；带下限。"""

    def __init__(self, floor_ms: float = _FLOOR_MS):
        self.floor_ms = float(floor_ms)
        self.period_ms = float(floor_ms)        #  起始取下限（高频次）
        self.samples: list[float] = []
        self.n = 0

    def observe(self, dt_ms: float) -> float:
        self.n += 1
        self.samples.append(float(dt_ms))
        if len(self.samples) > 3:
            self.samples.pop(0)
        if len(self.samples) == 3:              # 每 3 次刷新一遍
            self.period_ms = max(float(np.median(self.samples)) * 1.05, self.floor_ms)
        return self.period_ms

    def sleep(self) -> None:
        time.sleep(self.period_ms / 1e3)


def csc_sizes(T) -> tuple[tuple[int, int, int], int]:
    """ 不建 CSC 就算出它的三个数组长度与总字节（`data`/`indices` = nnz、`indptr` = n_col+1）。"""
    nnz, ncol = int(T.nnz), int(T.shape[1])
    want = (nnz, nnz, ncol + 1)
    size = HDR + want[0] * 4 + want[1] * 4 + want[2] * 4     # float32 + int32 + int32
    return want, size


def expected_build_ms(T) -> float:
    """预期建块耗时（标定值，只用来定周期下限；不参与正确性）。"""
    return max(1.0, int(T.nnz) * _MS_PER_NNZ)


# ----------------------------------------------------------------- 头部
def _crc32(d, i, p) -> int:
    """ 零拷贝 ACK：直接对三个数组的 buffer 算 `crc32`。 """
    c = zlib.crc32(memoryview(np.ascontiguousarray(d)))
    c = zlib.crc32(memoryview(np.ascontiguousarray(i)), c)
    c = zlib.crc32(memoryview(np.ascontiguousarray(p)), c)
    return int(c)


def _checksum(d, i, p) -> float:
    """兼容旧名：返回 `float(crc32)`（< 2^32，双精度可精确表示，可放进 header 的 `d` 字段）。"""
    return float(_crc32(d, i, p))


def _pack(key: str, want, state: int, ck: float) -> bytes:
    k = key_sha1(key)
    assert len(k) == _KEY_HEX, f"key 摘要长度 {len(k)} ≠ 字段宽度 {_KEY_HEX} → 会被静默截断"
    b = struct.pack(_FMT, MAGIC, k.encode("ascii"), want[0], want[1], want[2], state, ck)
    return b + b"\0" * (HDR - len(b))


def _header(shm):
    """`(kind, h)`，`kind ∈ {'ready','building','alien'}`。 全零 = 建中（≠ 损坏）。"""
    head = bytes(shm.buf[:struct.calcsize(_FMT)])
    if head[:5] != MAGIC:
        if not head.strip(b"\0"):
            return "building", None
        return "alien", None
    magic, k, nd, ni, np_, state, ck = struct.unpack(_FMT, head)
    h = dict(key=k.decode("ascii", "replace"), n_data=nd, n_ind=ni, n_ptr=np_, state=state, checksum=ck)
    return ("ready" if state == _STATE_READY else "building"), h


def _view(shm, want, shape):
    nd, ni, np_ = want
    o = HDR
    d = np.ndarray((nd,), dtype=_DT_D, buffer=shm.buf, offset=o)
    i = np.ndarray((ni,), dtype=_DT_I, buffer=shm.buf, offset=o + nd * 4)
    p = np.ndarray((np_,), dtype=_DT_P, buffer=shm.buf, offset=o + nd * 4 + ni * 4)
    return csc_matrix((d, i, p), shape=shape)


def _try_attach(key: str, shape, want=None):
    """`('ok', (csc, info))` / `('building', None)` / `('alien', None)` / `('none', None)`。"""
    try:
        shm = shared_memory.SharedMemory(name=name_of(key))
    except FileNotFoundError:
        return "none", None
    kind, h = _header(shm)
    if kind == "alien":
        _dbg(f"→ alien：前 8 字节={bytes(shm.buf[:8])!r}")
        shm.close()
        return "alien", None
    if kind == "building":
        shm.close()
        return "building", None
    if h["key"] != key_sha1(key):
        _dbg(f"→ alien(key 不符)：header={h['key']!r} 期望={key_sha1(key)!r}")
        shm.close()
        return "alien", None
    got = (h["n_data"], h["n_ind"], h["n_ptr"])
    if want is not None and got != want:
        _dbg(f"→ alien(尺寸不符)：header={got} want={want}")
        shm.close()
        return "alien", None
    csc = _view(shm, got, shape)
    return "ok", (csc, dict(shm=shm, owner=False, shared=True, name=name_of(key),
                            bytes=int(shm.size), checksum=h["checksum"],
                            n_data=got[0], n_ind=got[1], n_ptr=got[2]))


def publish_or_attach(T, key: str):
    """`T`(CSR) → 共享的 `csc_matrix`；返回 `(csc, info)`（`csc=None` 时 `info['reason']`）。"""
    shape = (int(T.shape[0]), int(T.shape[1]))
    want, size = csc_sizes(T)
    per = Period(floor_ms=max(_FLOOR_MS, expected_build_ms(T) / 10.0))
    t_end = time.time() + _WAIT_S
    while True:
        t0 = time.perf_counter()
        kind, hit = _try_attach(key, shape, want)
        dt_ms = (time.perf_counter() - t0) * 1e3
        if kind == "ok":
            hit[1].update(n_probes=per.n, period_ms=per.period_ms, built_here=False)
            _dbg(f"命中共享（探测 {per.n} 次，周期 {per.period_ms:.2f}ms）")
            return hit
        if kind == "alien":
            return None, dict(owner=False, shared=False, reason="alien_segment",
                              n_probes=per.n, period_ms=per.period_ms, built_here=False)
        if kind == "none":
            #  先占位、后计算：create 只需尺寸（上面已不建算出），失败者一次都不算。
            _t = time.perf_counter()
            try:
                shm = shared_memory.SharedMemory(name=name_of(key), create=True, size=size)
            except FileExistsError:
                t_claim = 0.0                            # 撞车，没花钱
                pass                                     # 别人先占了 → 转等待
            except OSError as e:
                _dbg(f"create 失败：{e!r}")
                return None, dict(owner=False, shared=False, reason="create_failed",
                                  n_probes=per.n, period_ms=per.period_ms, built_here=False)
            else:
                # 命名映射的建立本身是固定成本，单独计时。
                t_claim = (time.perf_counter() - _t) * 1e3
                shm.buf[:HDR] = _pack(key, want, _STATE_BUILDING, 0.0)     # 先声明"建中"
                _t = time.perf_counter()
                tc = T.tocsc()                                            # ← 高成本在占位之后
                t_tocsc = (time.perf_counter() - _t) * 1e3
                _t = time.perf_counter()
                d = np.ascontiguousarray(tc.data, dtype=_DT_D)
                i = np.ascontiguousarray(tc.indices, dtype=_DT_I)
                p = np.ascontiguousarray(tc.indptr, dtype=_DT_P)
                t_contig = (time.perf_counter() - _t) * 1e3
                got = (int(d.size), int(i.size), int(p.size))
                if got != want:                          # 尺寸预算错了：拆槽位、回落私有（不静默）
                    del tc, d, i, p
                    gc.collect()
                    try:
                        shm.close()
                        shared_memory.SharedMemory(name=name_of(key)).unlink()
                    except Exception:
                        pass
                    _dbg(f"尺寸预算不符：预算 {want} 实得 {got}")
                    return None, dict(owner=False, shared=False, reason="size_mismatch",
                                      n_probes=per.n, period_ms=per.period_ms, built_here=True)
                o = HDR
                _t = time.perf_counter()
                for arr in (d, i, p):
                    shm.buf[o:o + arr.nbytes] = arr.tobytes()
                    o += arr.nbytes
                t_copy = (time.perf_counter() - _t) * 1e3
                _t = time.perf_counter()
                ck = _checksum(d, i, p)
                shm.buf[:HDR] = _pack(key, want, _STATE_READY, ck)          # 数据写好才置"就绪"
                t_ack = (time.perf_counter() - _t) * 1e3
                _t = time.perf_counter()
                csc = _view(shm, want, shape)
                t_view = (time.perf_counter() - _t) * 1e3
                _t = time.perf_counter()                 #  计时赋值必须在构造 info 之前
                #  只 `del`，不做全局 `gc.collect()`：这几个都是局部变量、无循环引用，
                #  CPython 引用计数会立即释放，全局 GC 扫描纯属白花时间。
                del tc, d, i, p
                t_free = (time.perf_counter() - _t) * 1e3
                info = dict(shm=shm, owner=True, shared=True, name=name_of(key), bytes=int(shm.size),
                            checksum=ck, n_data=want[0], n_ind=want[1], n_ptr=want[2],
                            n_probes=per.n, period_ms=per.period_ms, built_here=True,
                            #  发布路径七个子阶段计时
                            t_claim_ms=t_claim, t_tocsc_ms=t_tocsc, t_contig_ms=t_contig,
                            t_copy_ms=t_copy, t_ack_ms=t_ack, t_view_ms=t_view, t_free_ms=t_free,
                            build_ms=t_claim + t_tocsc + t_contig + t_copy + t_ack + t_view + t_free)
                _dbg(f"发布共享（探测 {per.n} 次，周期 {per.period_ms:.2f}ms）")
                return csc, info
        per.observe(dt_ms)
        if time.time() > t_end:
            return None, dict(owner=False, shared=False, reason="build_timeout",
                              n_probes=per.n, period_ms=per.period_ms, built_here=False)
        per.sleep()


def cleanup(key: str) -> bool:
    """显式提前回收（Windows 上最后一个句柄一走段就没了 → 通常不需要）。"""
    try:
        shm = shared_memory.SharedMemory(name=name_of(key))
    except FileNotFoundError:
        return False
    try:
        shm.unlink()
    finally:
        shm.close()
    return True
