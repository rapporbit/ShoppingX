"""事件回放日志（D 块）——每个 thread 一条 Redis Stream，断线重连时补发缺口事件。

**解决什么。** 现状 connect-first 协议只保证「任务起步早于首次连接」不丢早期事件；但**中途
断线重连**（用户刷新页面 / 短暂断网）期间产生的事件就丢了——任务在后台照跑（与 WS 解耦），
用户看到的事件流却有个洞。这里给每个 thread 把 AGUI 事件**持久化**进一条 Redis Stream，重连时
带上「我收到的最后一个事件 id」，服务端从该 id 之后补发，再转直播。

**为什么是 Redis Stream。** Stream 本就是为「带位置追踪的事件日志」设计的：``XADD`` 的返回值
（stream id，单调递增）天然就是 ``last_event_id``，``XRANGE (last_id +`` 天然就是「补发缺口」。
``MAXLEN`` 自动裁剪老事件、不会无界增长。且同一套 Stream 正是将来多副本事件总线的底座（毕业线）。

**优雅降级（复用 B 块断路器）。** Redis 没配 / 没装 / 挂了都不该拖垮主链路：用
:class:`CircuitBreaker` 包 Redis 操作——连续失败即熔断、之后**快速失败**（不再每次等连接
超时），Redis 恢复后半开探测自动恢复。降级时 ``append`` 返回 ``None``、``replay_after``
返回 ``[]``，退回现状（只直播、无回放）。断路器同时自动进 A 块 metrics
（``shoppingx_circuit_breaker_state{name="event_log"}``）——三块联动。
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from app.utils.circuit_breaker import CircuitBreaker
from app.utils.env import env_bool, env_int

logger = logging.getLogger("shoppingx.event_log")

# 每条 thread Stream 保留的最近事件数（约 200 条够覆盖一次任务的全部事件，断线重连补得回）。
_MAXLEN = env_int("EVENT_REPLAY_MAXLEN", 200)
# 每条 Stream 的存活时长（秒，默认 6h）。MAXLEN 只限单条流的长度、不限流的**数量**——每个
# thread_id 一条流，不设 TTL 就会随会话无界累积。每次 append 刷新 TTL（滑动过期）：活跃任务期间
# 不过期、任务结束几小时后自动回收。回放是「短时间内断线重连」的事，过期了也不需要再回放。
_TTL = env_int("EVENT_REPLAY_TTL", 6 * 3600)
# 事件流是本项目**唯一**还用 Redis 的地方（长期偏好 / 历史 / 收藏已全部搬进关系库，见
# app/db/models.py）。故直接读 EVENT_REDIS_URL，不再回退早年的 STORE_REDIS_URL。
_URL = os.environ.get("EVENT_REDIS_URL", "redis://localhost:6379/2")

# 每轮 run_agent 一上来必发的事件名（见 main_agent：report_session_created 是开局第一条）。
# 同一 thread 的 Stream 里多轮事件首尾相接，「最后一个 session_created 起」即「当前这轮」的边界。
# 不 import monitor 的常量（monitor 反过来 import 本模块，会成环），就地写字面量并在此注明同源。
_SESSION_CREATED = "session_created"

# Redis 操作的断路器：连续失败即熔断，避免 Redis 没启时每个事件都干等连接超时。
_breaker = CircuitBreaker(
    "event_log",
    failure_threshold=env_int("EVENT_REPLAY_CB_THRESHOLD", 3),
    recovery_timeout=env_int("EVENT_REPLAY_CB_RECOVERY", 30),
)

_client: Any = None
_client_init_failed = False


def _enabled() -> bool:
    return env_bool("EVENT_REPLAY", True)


def _key(thread_id: str) -> str:
    return f"shoppingx:events:{thread_id}"


def _get_client() -> Any:
    """懒加载 redis.asyncio 客户端；未启用 / 未装 redis 时返回 None（降级到无回放）。"""
    global _client, _client_init_failed
    if not _enabled() or _client_init_failed:
        return None
    if _client is None:
        try:
            import redis.asyncio as aredis  # 可选依赖，懒加载（与 store.py 同源）

            # 连接 + 操作都设短超时：Redis 没启（连不上）或连上但卡住（hang）时单次操作都快速失败，
            # 抛异常喂给断路器，不让 _emit（每个事件都过这里）干等——光设 connect_timeout 挡不住
            # 「连上了但 XADD 卡住」那种 hang。
            _client = aredis.from_url(
                _URL, decode_responses=True, socket_connect_timeout=0.5, socket_timeout=2.0
            )
        except Exception as exc:  # 没装 redis 包等：标记失败、永不再试，静默降级
            logger.warning("event_log 初始化 redis 失败，事件回放降级关闭：%s", exc)
            _client_init_failed = True
            return None
    return _client


def set_client(client: Any) -> None:
    """注入 redis 客户端（测试用）；传 None 复位为懒加载。"""
    global _client, _client_init_failed
    _client = client
    _client_init_failed = False
    _breaker.reset()


async def append(thread_id: str, payload: dict[str, Any]) -> str | None:
    """持久化一条事件进该 thread 的 Stream，返回 stream id（即 last_event_id）；降级时返回 None。"""
    client = _get_client()
    if client is None:
        return None
    key = _key(thread_id)

    async def _xadd_and_expire() -> str:
        sid: str = await client.xadd(
            key,
            {"json": json.dumps(payload, ensure_ascii=False)},
            maxlen=_MAXLEN,
            approximate=True,  # 近似裁剪：Redis 更高效，长度在 MAXLEN 附近浮动即可
        )
        await client.expire(key, _TTL)  # 滑动刷新 TTL：活跃期间不过期，任务结束后自动回收
        return sid

    try:
        return await _breaker.call(_xadd_and_expire)
    except Exception as exc:  # 含 CircuitOpenError 与 redis 错误：持久化失败不拖垮主链路
        logger.debug("event_log append 降级（thread=%s）：%s", thread_id, exc)
        return None


async def replay_after(thread_id: str, last_event_id: str) -> list[dict[str, Any]]:
    """补发 ``last_event_id`` **之后**的事件（不含它本身）；降级 / 无缺口时返回空列表。

    每条补发事件回填其 stream id 到 ``payload["id"]``，与直播事件同构，前端可按 id 去重。
    """
    client = _get_client()
    if client is None or not last_event_id:
        return []
    try:
        # "(" 前缀 = 排他下界（Redis 6.2+）：从 last_event_id 之后开始，不重发它本身。
        entries = await _breaker.call(
            lambda: client.xrange(_key(thread_id), min=f"({last_event_id}", max="+")
        )
    except Exception as exc:
        logger.debug("event_log replay 降级（thread=%s）：%s", thread_id, exc)
        return []
    return _parse_entries(entries)


def _parse_entries(entries: list[Any]) -> list[dict[str, Any]]:
    """把 XRANGE 返回的 ``(stream_id, fields)`` 列表解析为带 ``id`` 的事件 payload 列表。

    与 :func:`replay_after` 同款解析（单条坏数据跳过），抽出来给 :func:`replay_current_run` 复用。
    """
    out: list[dict[str, Any]] = []
    for stream_id, fields in entries:
        try:
            payload = json.loads(fields["json"])
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
        payload["id"] = stream_id
        out.append(payload)
    return out


async def replay_current_run(thread_id: str) -> list[dict[str, Any]]:
    """返回该 thread「**当前这一轮 run**」的全部事件——即最后一个 ``session_created`` 起的事件。

    用途：刷新页面 / 切回某对话时，若该 thread 仍有任务在后台跑，前端要重建「正在跑的那一轮」。
    历史轮已落 turns.json（由 ``/api/history`` 回看），故这里只截取**尚未收尾**的最新一轮，避免与
    历史轮重复。降级（Redis 没配 / 挂了）或流里没有 ``session_created`` 时返回空列表，前端退回
    「无回放、纯靠后续直播续看」（live 推送不依赖 Redis，新事件照样到）。

    截取边界靠「最后一个 session_created」而非时间：MAXLEN 近似裁剪下，极长的一轮可能把开头的
    session_created 挤掉——那时退回从流首开始（已是裁剪后的最近窗口），少量开头事件丢失可接受。
    """
    client = _get_client()
    if client is None:
        return []
    try:
        entries = await _breaker.call(lambda: client.xrange(_key(thread_id), min="-", max="+"))
    except Exception as exc:  # 含 CircuitOpenError 与 redis 错误：降级为空，不拖垮主链路
        logger.debug("event_log replay_current_run 降级（thread=%s）：%s", thread_id, exc)
        return []
    parsed = _parse_entries(entries)
    start = 0
    for i in range(len(parsed) - 1, -1, -1):
        if parsed[i].get("event") == _SESSION_CREATED:
            start = i
            break
    return parsed[start:]
