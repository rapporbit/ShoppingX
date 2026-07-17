"""进程内语义缓存（E 块）——精确 dict 快路径 + 向量近似命中，复用召回层的 embedding。

**用在哪。** 适合「**弱时效** + 键是自然语言 + 近义应命中同一结果」的场景：典型就是
``category_insight``（品类知识，重复 / 近义品类查询多，命中即省一整条 Hybrid 召回 + 精排）。
**强时效数据（商品价格 / 库存）严禁用**——「近似命中」返回别的 query 的旧结果，在购物场景是硬伤。

**两级。**
- **精确层**（``cachetools.TTLCache``）：key 完全相同直接命中，**免编码、免检索**，O(1) 最快路径。
- **语义层**：精确未命中时，把 query 编码成向量，与缓存里的向量算余弦——近义 query（"luggage"
  ↔ "行李箱"，别名表没收的）也能命中。向量复用召回层的 ``TowerClient.encode_query``（已 L2 归一，
  余弦=点积），不引额外模型。

**为什么进程内而非 Qdrant collection（对方案 §二·五 C 的主动更正）。** category 的取值空间小
（几十个品类词），进程内几百条向量足够覆盖；单机下进程内点积**无网络往返**，严格优于再起一个
Qdrant collection（多一跳 + 一套 collection 生命周期管理）。Qdrant 路线留作大规模 / 多副本毕业线
——那时缓存要跨进程共享才轮到它。这与「进程内原语在单机下严格更优」的判据一致。

**容量与时效。** 精确层与语义层都受 ``max_entries`` 上限与 ``ttl`` 约束（超量淘汰最旧、过期跳过），
不会无界增长。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Generic, TypeVar

import numpy as np
from cachetools import TTLCache

T = TypeVar("T")


@dataclass
class _VecEntry(Generic[T]):
    vec: np.ndarray
    group: str  # 维度隔离（如 depth=quick/deep 不互相命中）
    value: T
    ts: float


class SemanticCache(Generic[T]):
    """精确 + 语义两级的进程内缓存。线程模型：asyncio 单线程，方法均同步、无 await，无需锁。

    **不可变约定。** ``get_*`` 返回的是缓存内对象的**同一引用**（不深拷贝，省开销）。调用方须
    把返回值当只读——若就地改它（如截断列表、改字段），改动会污染缓存、被后续命中读到。
    category_insight 的输出只读不改，满足此约定；将来若有消费方要改，应先 ``model_copy``。
    """

    def __init__(
        self, name: str, *, max_entries: int = 256, ttl: float = 3600.0, threshold: float = 0.92
    ) -> None:
        self.name = name
        self.ttl = ttl
        self.threshold = threshold  # 余弦 ≥ 此值才算语义命中；越高越保守（更像精确）
        self._max = max_entries
        self._exact: TTLCache[str, T] = TTLCache(maxsize=max_entries, ttl=ttl)
        self._vecs: list[_VecEntry[T]] = []

    def get_exact(self, key: str) -> T | None:
        """精确命中（key 完全相同）；未命中 / 已过期返回 None。"""
        try:
            return self._exact[key]
        except KeyError:
            return None

    def get_semantic(self, vec: np.ndarray, group: str = "") -> T | None:
        """语义命中：与同 ``group`` 的缓存向量算余弦，最相似且 ≥ 阈值则返回其值；否则 None。"""
        now = time.monotonic()
        # 顺手清掉过期项（语义层自管 TTL；精确层由 TTLCache 自管）。
        self._vecs = [e for e in self._vecs if now - e.ts < self.ttl]
        best: _VecEntry[T] | None = None
        best_sim = -1.0
        for e in self._vecs:
            if e.group != group:
                continue
            # 维度守卫：towers 远程↔本地回退可能产出不同维向量，混维会让 np.dot 抛 ValueError
            # 打挂整个工具（比没缓存更糟）。维度不一致直接跳过，当不命中处理。
            if e.vec.shape != vec.shape:
                continue
            sim = float(np.dot(vec, e.vec))  # 两侧都 L2 归一 → 点积即余弦
            if sim > best_sim:
                best_sim = sim
                best = e
        if best is not None and best_sim >= self.threshold:
            return best.value
        return None

    def put(self, key: str, value: T, vec: np.ndarray | None = None, group: str = "") -> None:
        """写入缓存。**精确层总是写**；语义层仅在提供 ``vec`` 时写——这样即便编码不可用（towers
        挂了 / 离线），相同 key 仍能走精确快路径，不至于连精确缓存都失效。语义层超容量淘汰最旧。"""
        self._exact[key] = value
        if vec is not None:
            self._vecs.append(_VecEntry(vec=vec, group=group, value=value, ts=time.monotonic()))
            if len(self._vecs) > self._max:
                self._vecs = self._vecs[-self._max :]

    def clear(self) -> None:
        """清空（测试 / 灌库后失效用）。"""
        self._exact.clear()
        self._vecs.clear()
