"""请求指纹去重（refdocs 16-5 §3.2 第三层）——用户手抖连点两下，不该跑两遍 Agent。

**为什么光靠 ``active_tasks`` 不够。** 第 1 层按 ``thread_id`` 索引，只能挡住「同一会话里重复
提交」。不管 thread_id 的调用方（脚本、裸 API、压测器）每次都拿一个新 thread_id，同一条 query
连发十遍就真跑十遍——第 1 层完全看不见。指纹去重守的就是这个缺口：key 里刻意**不含 thread_id**。

**只对「不自带 thread_id」的调用方生效**（判定在 ``server.create_task`` 里）。本项目前端是
connect-first：它自己生成 thread_id、先连 WS、再 POST。把它并进别人的 thread_id 会让它那条 WS
一个事件都收不到、界面永远转圈；而且它的 thread_id 存在 localStorage、刷新不换，refdocs 设想的
「刷新 → 换 thread_id → 重复提交」在本前端根本不会发生。详见 ``server.create_task`` 的注释。

**窗口为什么这么短（默认 5 秒）。** 这层要区分的是「手抖 / 刷新」和「用户真的想再问一次」。
5 秒足够覆盖前者（双击、刷新、网络重试都在这个量级），又不会误伤后者——没有哪个用户会精确地在
5 秒内故意重发完全相同的一句话，就算真发了，代价也不过是拿到上一次的 thread_id 继续看结果。

**窗口在 Redis，没有进程内回退（阶段 1-2）。** 原先是一个进程内 dict，多副本下每台各有一份表，
跨副本的重复提交挡不住——压测器拿一个新 thread_id 连发十遍，打到两台就跑两遍。搬进 Redis 之后
``SET NX EX`` 一条命令同时完成「查」和「登记」，**判定的原子性由 Redis 保证，不再依赖调用方在
两步之间不出现 ``await``**（那是老实现最脆的地方：中间一让出事件循环，两个并发请求双双查到
「不重复」）。Redis 不可达时**直接 503，不退回进程内**：退回等于「看起来还在去重，其实每台各
去各的」，比明说不可用更糟；而多副本形态下 Redis 本就是队列与事件的硬依赖，它挂了请求也跑不成。

**先登记、被拒再撤（与 run_holds 的预扣同一手法）。** ``SET NX`` 是原子的一步，没法只查不登记，
所以进门就登记；后面被 429 / ``already_running`` 拒掉的路径各自调 :func:`forget` 撤销——**不撤的
话用户退避重试会被自己刚才那次失败的请求挡住**，陷入「越重试越被判重复」的死循环。

**refdocs 的第二层（Checkpoint 防重跑）本项目不做。** 它依赖 LangGraph checkpointer 把中断的图
状态存进 Redis、重启后从断点恢复。本项目是轮级快照（session.json），没有 step 级恢复需求
（见 `docs/plans/后端优化执行计划-2026-09-16.md` §0）。所以幂等只有第一层（threads 条件更新）
和第三层（本模块）。
"""

from __future__ import annotations

import hashlib
import logging
import os
from typing import Any, Protocol, cast

from app.utils.env import env_bool, env_int

logger = logging.getLogger("shoppingx.dedup")

# 同一 (user_id, query) 在这个窗口内再次提交 → 判为重复。
DEDUP_WINDOW_SEC = env_int("TASK_DEDUP_WINDOW_SEC", 5)

# 键前缀：与队列 / 事件 / 控制面共用一个 Redis 实例，靠前缀分账。
KEY_PREFIX = "globex:dedup:"


class DedupUnavailable(RuntimeError):
    """Redis 不可达。调用方转 503——本层没有降级形态，理由见模块 docstring。"""


class _Client(Protocol):
    """只用到三条命令，照这个形状注入假客户端即可（测试无需 fakeredis 包）。"""

    async def set(self, key: str, value: str, *, nx: bool = ..., ex: int = ...) -> Any: ...

    async def get(self, key: str) -> Any: ...

    async def delete(self, *keys: str) -> Any: ...


_client: _Client | None = None
_resolved = False


def dedup_enabled() -> bool:
    """关掉即整层不生效（``check_duplicate`` 恒返回 None）。默认开。"""
    return env_bool("TASK_DEDUP_ENABLED", True)


def _redis_url() -> str:
    """复用队列 / 事件那个实例（键名带前缀，同一个 db 不会撞）。与 control.py 同一套回退链。"""
    return (
        os.environ.get("DEDUP_REDIS_URL")
        or os.environ.get("QUEUE_REDIS_URL")
        or os.environ.get("EVENT_REDIS_URL", "redis://localhost:6379/2")
    )


def get_client() -> _Client:
    """进程级单例。建不起来直接抛 :class:`DedupUnavailable`——本层没有降级形态。"""
    global _client, _resolved
    if _resolved and _client is not None:
        return _client
    try:
        import redis.asyncio as aredis  # 可选依赖，懒加载（与 queue / control / event_log 同源）

        client = aredis.from_url(
            _redis_url(),
            decode_responses=True,
            socket_connect_timeout=2.0,
            socket_timeout=2.0,  # 去重是请求关键路径上的一次往返，宁可快失败也不吊着用户
        )
    except Exception as exc:  # 包没装 / URL 不合法
        raise DedupUnavailable(str(exc)) from exc
    # cast：redis-py 的 set/get/delete 签名比这里用到的宽（多一堆可选参数），结构上兼容但
    # mypy 认不出——Protocol 只是给假客户端立的形状，不值得为它把真客户端的签名抄一遍。
    _client = cast(_Client, client)
    _resolved = True
    return _client


def set_client(client: _Client | None) -> None:
    """注入实例（测试 / 手工装配用）。传 ``None`` 即复位为「按环境变量重新决定」。"""
    global _client, _resolved
    _client = client
    _resolved = client is not None


def reset() -> None:
    """复位进程级单例（测试收尾用）。"""
    set_client(None)


def _key(user_id: str | None, query: str) -> str:
    """(user_id, query) 的稳定指纹。**不含 thread_id**——本层要挡的正是换 thread_id 的重复提交。"""
    raw = f"{user_id or ''}:{query.strip()}"
    return KEY_PREFIX + hashlib.sha256(raw.encode("utf-8")).hexdigest()


async def check_duplicate(user_id: str | None, query: str, thread_id: str) -> str | None:
    """窗口内重复提交则返回**上一次的 thread_id**，否则登记本次并返回 ``None``。

    返回原 thread_id 而不是布尔值，是为了让调用方能把用户引导回那个正在跑的任务——重复提交的
    正确处理是「你要的东西已经在做了，看这里」，不是「请求被拒绝」。

    登记与查询是同一条 ``SET NX EX``（理由见模块 docstring）。被拒的路径记得 :func:`forget`。
    """
    if not dedup_enabled():
        return None
    client = get_client()
    key = _key(user_id, query)
    try:
        if await client.set(key, thread_id, nx=True, ex=DEDUP_WINDOW_SEC):
            return None
        return await client.get(key) or None
    except DedupUnavailable:
        raise
    except Exception as exc:
        logger.warning("去重窗口不可用：%s", exc)
        raise DedupUnavailable(str(exc)) from exc


async def forget(user_id: str | None, query: str) -> None:
    """撤销一次登记（任务最终没起来时调）。失败只记日志——最坏是这条 query 被挡 5 秒。"""
    if not dedup_enabled():
        return
    try:
        await get_client().delete(_key(user_id, query))
    except Exception as exc:
        logger.warning("去重窗口撤销失败（%s 秒后自然过期）：%s", DEDUP_WINDOW_SEC, exc)
