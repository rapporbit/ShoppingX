"""商品召回的两级缓存：L1 进程内 + L2 Redis（阶段 3）。

**缓存的是哪一段。** 包住「编码 query → Qdrant 召回」这两步的结果（``RecallCandidate`` 列表），
**不缓存 ``item_search`` 的最终产出**。后者还要过会话级 P_t 的硬排除、记忆、槽位盖章、候选登记，
这些每轮都可能变；连它们一起缓存，就会出现「用户刚说不要塑料，下一轮又原样端回来」——缓存把
一个已经生效的约束静默回滚，比不缓存糟得多。

**为什么连 embedding 一起包。** 命中时省掉的不只是 Qdrant 那一跳，还有一次 embedding 往返
（远程编码实测是这条链路上更贵的一段）。所以回源闭包由调用方传进来，编码在闭包里、也在缓存
边界之内；调用方那边做成惰性的，三次召回（主/放宽/探测）只在真回源时编码一次。

**只做精确 key 命中**，不做近似命中——理由见 :mod:`app.recall.semantic_cache` 的 docstring：
商品价格类数据下，「意思差不多」的两条 query 返回同一批货就是错的。

**退化。** L2 不可用（Redis 挂 / 慢）时退成 L1 + 回源，不抛；L1 与 L2 都不命中且 singleflight
锁没抢到时，短轮询 L2 一会儿，还没有就自己回源。任何一层出问题，最差也就是退回没有缓存的现状。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from cachetools import LRUCache

from app.recall.schemas import RecallCandidate
from app.utils.circuit_breaker import CircuitBreaker, CircuitOpenError
from app.utils.env import env_bool, env_float, env_int

logger = logging.getLogger("shoppingx.search_cache")

KEY_PREFIX = "shoppingx:recall:"


def cache_enabled() -> bool:
    """总开关（``RETRIEVAL_CACHE_ENABLED``，默认开）。关掉 = 每次都回源，行为回到改造前。"""
    return env_bool("RETRIEVAL_CACHE_ENABLED", True)


def _ttl() -> float:
    """正常结果的存活时长。商品库是离线建的、一天也不动一次，900s 内的陈旧完全可接受。"""
    return env_float("RETRIEVAL_CACHE_TTL", 900.0)


def _empty_ttl() -> float:
    """空结果的存活时长，远短于正常值。

    空结果要缓存（挡穿透：冷门词一遍遍打到 Qdrant），但不能缓存久——库一扩、索引一重建，
    「这个词没货」就不再成立，而用户对「搜不到」的重试往往就在几十秒内。
    """
    return env_float("RETRIEVAL_CACHE_EMPTY_TTL", 60.0)


def _index_version() -> str:
    """索引版本：换了库 / 重建了索引，旧 key 自然作废。

    默认取 collection 名（重建索引通常连带换名或换实例）；``RETRIEVAL_INDEX_VERSION`` 可显式
    盖过它——原地重建同名 collection 时，只有手工改这个值才能把旧 key 甩掉。
    """
    explicit = os.environ.get("RETRIEVAL_INDEX_VERSION", "").strip()
    if explicit:
        return explicit
    return os.environ.get("QDRANT_COLLECTION", "shoppingx_items")


def make_key(
    query: str,
    top_k: int,
    platforms: Sequence[str],
    *,
    price_usd_max: float | None = None,
    min_rating: float | None = None,
) -> str:
    """把一次召回的**全部入参**压成一个 key。

    归一化的两条：query 去首尾空白 + 小写；平台名同样归一后**排序**——``["ebay","amazon"]`` 与
    ``["amazon","ebay"]`` 搜的是同一批货，不归一就是白白多一份副本。

    top_k 进 key 而不是「取大的那份截断」：召回是 ANN，top_k 不同拿到的不是同一个前缀
    （HNSW 的 ef 随 limit 变），截断出来的东西与真搜 20 条不是一回事。
    """
    payload = {
        "q": query.strip().lower(),
        "k": int(top_k),
        "p": sorted({str(p).strip().lower() for p in platforms if str(p).strip()}),
        "price_max": price_usd_max,
        "min_rating": min_rating,
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:32]  # noqa: S324 - 做 key 不是签名
    return f"{KEY_PREFIX}{_index_version()}:{digest}"


# ── L1：进程内 ────────────────────────────────────────────────────────────────
# 存 ``(过期时刻, 序列化好的 JSON)``：两种 TTL（正常 / 空结果）共用一个结构，比起挂两个 TTLCache
# 少一半状态。存 JSON 而不是对象列表，是为了和 L2 同一份形态——回填、读出走同一条解码路径，
# 不会出现「L1 命中给的是对象、L2 命中给的是 dict」这种下游得分情况处理的坑。
_l1: LRUCache[str, tuple[float, str]] = LRUCache(maxsize=env_int("RETRIEVAL_CACHE_L1_SIZE", 512))

#: 进程内 singleflight：同 key 的并发回源共用一个 Future（同轮 batch 跨平台搜同一个词时就会撞）。
_inflight: dict[str, asyncio.Future[list[RecallCandidate]]] = {}


def _l1_get(key: str) -> str | None:
    hit = _l1.get(key)
    if hit is None:
        return None
    expires_at, payload = hit
    if expires_at <= time.monotonic():
        _l1.pop(key, None)
        return None
    return payload


def _l1_put(key: str, payload: str, ttl: float) -> None:
    _l1[key] = (time.monotonic() + ttl, payload)


def reset_cache() -> None:
    """清空 L1 与在飞表（测试用；L2 不动——那是共享状态，测试不该替别人清）。"""
    _l1.clear()
    _inflight.clear()


# ── L2：Redis ────────────────────────────────────────────────────────────────
_client: Any | None = None
_resolved = False
_breaker = CircuitBreaker(
    "retrieval_cache_l2",
    failure_threshold=env_int("RETRIEVAL_CACHE_CB_THRESHOLD", 3),
    recovery_timeout=env_float("RETRIEVAL_CACHE_CB_RECOVERY", 30.0),
)


def _redis_url() -> str:
    return os.environ.get("RETRIEVAL_CACHE_REDIS_URL") or os.environ.get(
        "QUEUE_REDIS_URL", os.environ.get("EVENT_REDIS_URL", "redis://localhost:6379/2")
    )


def _get_client() -> Any | None:
    """懒建 Redis 客户端；建不起来返回 ``None``（整层退 L1 + 回源，不抛）。

    超时按「1 秒」钉死，与令牌桶同理由：缓存坐在每次检索的必经之路上，Redis 卡住必须立刻退，
    而不是让每次检索先干等——那正是缓存要消除的那种等待。
    """
    global _client, _resolved
    if _resolved:
        return _client
    _resolved = True
    try:
        import redis.asyncio as aredis  # 可选依赖，懒加载（与 queue / control / 令牌桶同源）

        _client = aredis.from_url(
            _redis_url(),
            decode_responses=True,
            socket_connect_timeout=1.0,
            socket_timeout=1.0,
        )
    except Exception as exc:  # pragma: no cover - 缺包 / URL 非法
        logger.warning("检索缓存 L2 初始化失败，只用 L1：%s", exc)
        _client = None
    return _client


def set_client(client: Any | None) -> None:
    """注入 Redis 客户端（测试 / 手工装配）；传 ``None`` = 关掉 L2，只留 L1。"""
    global _client, _resolved
    _resolved = True
    _client = client
    _breaker.reset()


async def _l2_get(key: str) -> str | None:
    client = _get_client()
    if client is None:
        return None
    try:
        value = await _breaker.call(lambda: client.get(key))
    except CircuitOpenError:
        return None
    except Exception as exc:  # noqa: BLE001 - 缓存读失败只是没命中，不能反噬主链路
        logger.warning("检索缓存 L2 读失败，退回源：%s: %s", type(exc).__name__, exc)
        return None
    return value if isinstance(value, str) else None


async def _l2_put(key: str, payload: str, ttl: float) -> None:
    client = _get_client()
    if client is None:
        return
    try:
        await _breaker.call(lambda: client.set(key, payload, ex=max(1, int(ttl))))
    except CircuitOpenError:
        return
    except Exception as exc:  # noqa: BLE001 - 写不进去就是下次再查一遍，不值得让请求失败
        logger.warning("检索缓存 L2 写失败，忽略：%s: %s", type(exc).__name__, exc)


async def _try_lock(key: str) -> bool:
    """跨进程 singleflight：抢到锁的那个副本去回源，没抢到的短轮询 L2。

    锁的 TTL 取「一次回源最多要多久」的量级（默认 5s）：抢锁的副本中途挂了，锁最多卡住这么久，
    之后别的副本照样能回源。抢不到锁**不阻塞到底**——轮询上限之后就自己查，宁可多打一次 Qdrant
    也不让用户等在一个可能永远不会被填上的缓存键上。
    """
    client = _get_client()
    if client is None:
        return True  # 没有 L2 就没有跨进程协调，各查各的
    try:
        ok = await _breaker.call(
            lambda: client.set(
                f"{key}:lock", "1", nx=True, ex=env_int("RETRIEVAL_CACHE_LOCK_TTL", 5)
            )
        )
    except (CircuitOpenError, Exception):  # noqa: B014 - 锁拿不到就当拿到了，退化成各查各的
        return True
    return bool(ok)


async def _unlock(key: str) -> None:
    client = _get_client()
    if client is None:
        return
    try:
        await _breaker.call(lambda: client.delete(f"{key}:lock"))
    except Exception:  # noqa: BLE001 - 删不掉就等 TTL，无伤
        pass


# ── 序列化 ───────────────────────────────────────────────────────────────────
def _dump(candidates: Sequence[RecallCandidate]) -> str:
    return json.dumps([c.model_dump() for c in candidates], ensure_ascii=False, default=str)


def _load(payload: str) -> list[RecallCandidate] | None:
    """解码缓存值；**解不动就当没命中**。

    缓存里躺着的可能是上一个版本的进程写的（``RecallCandidate`` 加了必填字段、改了类型）。
    这里抛出去会让一次本来能正常回源的检索直接失败——而缓存的任何一层出问题，代价都只该是
    「这次没省到」。
    """
    try:
        raw = json.loads(payload)
        return [RecallCandidate(**item) for item in raw]
    except Exception as exc:  # noqa: BLE001
        logger.warning("检索缓存解码失败，当作未命中：%s: %s", type(exc).__name__, exc)
        return None


# ── 主入口 ───────────────────────────────────────────────────────────────────
async def cached_recall(
    query: str,
    top_k: int,
    platforms: Sequence[str],
    *,
    price_usd_max: float | None = None,
    min_rating: float | None = None,
    fetch: Callable[[], Awaitable[list[RecallCandidate]]],
) -> list[RecallCandidate]:
    """读 L1 → L2 → ``fetch()`` 回源，命中即回填两级。

    ``fetch`` 是调用方给的回源闭包（编码 + Qdrant 召回），只在两级都没命中时才会被 await ——
    调用方据此把编码做成惰性的，全命中时一次 embedding 往返都不发。

    回源抛出的异常（如 :class:`~app.utils.dependency.DependencyDown`）**原样向上抛**：依赖挂了
    是要让模型知道的事，缓存不该把它吞成一个空列表——那会让「服务挂了」长得和「库里没货」一样。
    """
    if not cache_enabled():
        return await fetch()

    key = make_key(query, top_k, platforms, price_usd_max=price_usd_max, min_rating=min_rating)
    cached = _l1_get(key)
    if cached is not None:
        candidates = _load(cached)
        if candidates is not None:
            _observe("l1")
            return candidates

    # 进程内 singleflight：同轮 batch 跨平台搜同一个词时，几条调用会在同一瞬间撞上同一个 key。
    # 让它们共用一个 Future，回源只发一次。
    running = _inflight.get(key)
    if running is not None:
        return list(await asyncio.shield(running))

    loop = asyncio.get_running_loop()
    future: asyncio.Future[list[RecallCandidate]] = loop.create_future()
    _inflight[key] = future
    try:
        result = await _resolve(key, fetch)
    except BaseException as exc:
        if not future.done():
            future.set_exception(exc)
        # 没人 await 这个 future 时，异常会在 GC 时报 "never retrieved"——先消费掉。
        future.exception()
        raise
    else:
        if not future.done():
            future.set_result(result)
        return result
    finally:
        _inflight.pop(key, None)


async def _resolve(
    key: str, fetch: Callable[[], Awaitable[list[RecallCandidate]]]
) -> list[RecallCandidate]:
    """L2 → 跨进程 singleflight → 回源，并回填两级。"""
    payload = await _l2_get(key)
    if payload is not None:
        candidates = _load(payload)
        if candidates is not None:
            _l1_put(key, payload, _ttl())  # L2 命中也回填 L1：同一进程的下次不必再走网络
            _observe("l2")
            return candidates

    got_lock = await _try_lock(key)
    if not got_lock:
        waited = await _wait_for_peer(key)
        if waited is not None:
            _observe("l2_wait")
            return waited

    try:
        result = await fetch()
    finally:
        if got_lock:
            await _unlock(key)
    payload = _dump(result)
    ttl = _empty_ttl() if not result else _ttl()
    _l1_put(key, payload, ttl)
    await _l2_put(key, payload, ttl)
    _observe("miss")
    return result


async def _wait_for_peer(key: str) -> list[RecallCandidate] | None:
    """没抢到锁：短轮询 L2 等别的副本填上；等不到就返回 ``None``（自己回源）。"""
    step = env_float("RETRIEVAL_CACHE_WAIT_STEP", 0.05)
    rounds = env_int("RETRIEVAL_CACHE_WAIT_ROUNDS", 10)
    for _ in range(rounds):
        await asyncio.sleep(step)
        payload = await _l2_get(key)
        if payload is None:
            continue
        candidates = _load(payload)
        if candidates is not None:
            _l1_put(key, payload, _ttl())
            return candidates
    return None


def _observe(layer: str) -> None:
    """命中分层计数（l1 / l2 / l2_wait / miss）。命中率是这层唯一值得看的指标。

    复用既有的 ``shoppingx_cache_events_total``（TurnCache 用的同一个），只是 ``cache`` 标签取
    ``retrieval``——同一类事实放两个指标名，看板上迟早只画其中一个。
    """
    try:
        from app.observability.metrics import record_cache

        record_cache("retrieval", layer)
    except Exception:  # noqa: BLE001 - 指标不该反噬主链路
        pass
