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

import hashlib
import os
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

import numpy as np
from cachetools import TTLCache

from app.utils.env import env_bool, env_float, env_int

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


# ══════════════════════════════════════════════════════════════════════════════
# 整轮结果缓存（批2-5）——上面那套是「一次工具调用」的缓存，这里缓存的是**一整轮 Agent**。
# ══════════════════════════════════════════════════════════════════════════════
#
# **它是压测 / 演示用的开关，不是常开特性，所以默认关。** 削峰队列压测时同一条 query 会被重复打
# 几百遍，每遍都真跑一次 AgentLoop 的话，测出来的是模型供应商的限流曲线而不是本系统的吞吐；
# 演示同理（讲解时反复问同一句，不该每次等 40 秒）。
#
# **为什么必须默认关、且评测脚本要主动拒跑。** 缓存一开，Rubric 评测就可能拿到上一次跑的答案——
# 分数变成「上次那份的复读」，改了 prompt 也看不出差别，而且**全程零报错**。这类「测出来的数
# 是假的」比崩溃危险得多，所以除了默认关，``scripts/eval/run_rubric.py`` 开跑前还要显式查一次
# （见那边的 ``_assert_turn_cache_off``）。
#
# **key 的四个成分**（缺一个就会串味）：
# - **buyer**：偏好注入、行为亲和、订单归属都按人不同，跨用户复用等于把别人的结果给你看。
# - **偏好指纹**：同一个人改了偏好（加一条「不要皮革」），旧答案立刻不成立。
# - **prompt 指纹**：用户这句话 **+ prompts.yml 的内容指纹**。后者是为了让「改了提示词」自动
#   失效整片缓存——不然调完 prompt 重跑，看到的还是旧行为，会把人引到完全错误的结论上。
# - **模型**：换模型就是换系统，不能复用。
#
# **两类轮次一律不入缓存**（判据在 :func:`turn_is_cacheable`）：**有历史轮**的（答案依赖上文，
# 而上文不在 key 里）与**写意图 / 交互轮**的（下单、取消、澄清、遗忘偏好——重放一份「已下单」
# 是真实伤害，不是少省一点钱）。

#: 这些工具一旦在本轮出现过，本轮就**不许**进缓存：三个写工具会改真实状态，``ask_user`` 的答案
#: 取决于当时用户怎么回的、``forget_preference`` 改的是长期记忆。重放它们等于伪造一次交互。
UNCACHEABLE_TOOLS = frozenset(
    {"create_order", "cancel_order", "query_order", "ask_user", "forget_preference"}
)


def turn_cache_enabled() -> bool:
    """整轮缓存是否开着。**默认关**，见本节开头。"""
    return env_bool("TURN_CACHE_ENABLED", False)


@dataclass(frozen=True)
class TurnCacheEntry:
    """一轮的可复用产物。只存「给用户看的那两样」——文案与商品卡。

    刻意**不存** ``AgentState`` / 候选池 / 产物文件：那些是给「下一轮续聊」用的，而带历史的轮次
    本就不入缓存，存了也没人读，却要为每条缓存背上几十 KB。
    """

    final_text: str
    items: list[dict[str, Any]]


class TurnCache:
    """进程内的整轮结果缓存（TTL + 容量上限）。

    **进程内而非 Redis**：命中的价值是「省掉一整轮 LLM」，跨副本共享省的只是「另一个副本也各自
    跑一次」——收益二阶，却要给每轮加一次网络往返 + 一份序列化。真要跨副本共享时再说。
    """

    def __init__(self, *, max_entries: int = 128, ttl: float = 900.0) -> None:
        self._c: TTLCache[str, TurnCacheEntry] = TTLCache(maxsize=max_entries, ttl=ttl)

    def get(self, key: str) -> TurnCacheEntry | None:
        try:
            return self._c[key]
        except KeyError:
            return None

    def put(self, key: str, entry: TurnCacheEntry) -> None:
        self._c[key] = entry

    def clear(self) -> None:
        self._c.clear()

    def __len__(self) -> int:
        return len(self._c)


_turn_cache: TurnCache | None = None


def get_turn_cache() -> TurnCache:
    """进程级单例（懒建，容量与 TTL 从环境变量读一次）。"""
    global _turn_cache
    if _turn_cache is None:
        _turn_cache = TurnCache(
            max_entries=env_int("TURN_CACHE_MAX_ENTRIES", 128),
            ttl=env_float("TURN_CACHE_TTL", 900.0),
        )
    return _turn_cache


def reset_turn_cache() -> None:
    """丢掉单例（测试 / 改完配置想重建时用）。"""
    global _turn_cache
    _turn_cache = None


def _prompts_fingerprint() -> str:
    """``prompt/prompts.yml`` 的内容指纹——改了提示词，整片缓存自动失效。

    读不到就返回空串（缓存照常工作，只是少了这层失效）：为一个缓存指纹让主链路起不来不值当。
    """
    try:
        from app.utils.path_utils import PROJECT_ROOT

        return hashlib.sha256((PROJECT_ROOT / "prompt" / "prompts.yml").read_bytes()).hexdigest()[
            :16
        ]
    except Exception:
        return ""


def preference_fingerprint(entries: Sequence[Any]) -> str:
    """把一组偏好压成一个指纹。

    取 ``dedup_key`` + 正文 + 是否硬淘汰，**排序后**再哈希：库里的返回顺序不保证稳定，不排序的话
    同一组偏好会算出不同指纹，缓存永远不命中（症状是「开了没用」而不是报错，最难查）。
    """
    parts = sorted(
        "{}|{}|{:d}".format(
            getattr(e, "dedup_key", ""),
            getattr(e, "content", ""),
            bool(getattr(e, "is_blocking", False)),
        )
        for e in entries
    )
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()[:16]


def turn_cache_key(*, buyer: str, prefs_fp: str, query: str, model: str = "") -> str:
    """buyer + 偏好指纹 + prompt 指纹 + 模型 → 一个键。四个成分的理由见本节开头。"""
    prompt_fp = hashlib.sha256(query.strip().encode("utf-8")).hexdigest()[:24]
    model_name = model or os.environ.get("LLM_MAIN", "")
    return f"{buyer or 'anon'}|{prefs_fp}|{prompt_fp}|{_prompts_fingerprint()}|{model_name}"


def turn_is_cacheable(tool_names: Iterable[str], final_text: str) -> bool:
    """本轮能不能进缓存：非空回复 + 没碰过写 / 交互类工具（:data:`UNCACHEABLE_TOOLS`）。

    「有历史轮不入」不在这里判——那要在**开跑之前**就知道（否则白跑一轮才发现不能存），由调用方
    在查缓存那一步一并决定：不查缓存的轮次也不写缓存。
    """
    if not final_text.strip():
        return False
    return not (set(tool_names) & UNCACHEABLE_TOOLS)


def turn_cache_status() -> dict[str, Any]:
    """给 ``/api/health`` 用的一行状态（评测脚本据此拒跑）。**只在开着时才建单例**，
    免得一次探活就把缓存对象建出来。"""
    enabled = turn_cache_enabled()
    return {
        "enabled": enabled,
        "entries": len(get_turn_cache()) if enabled else 0,
    }
