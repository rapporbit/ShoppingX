"""削峰任务队列。端口见 :mod:`app.queue.ports`，两份实现在 ``redis_stream`` / ``inprocess``。

对外只暴露一个工厂 :func:`get_task_queue`：进程里全局一份，**恒是 Redis Stream**（阶段 1 条 7 起
删掉 ``QUEUE_ENABLED``，队列不再是可选项——多副本是唯一形态，API 与 worker 是不同进程，进程内
deque 在两边各有一份、谁也收不到谁的任务）。``server.py`` 与 ``worker.py`` 都从它拿队列，谁也不该
自己 ``RedisStreamQueue(...)``——不然两边各拿一个客户端、各建一次组，连的还可能不是同一个 URL。

``InProcessQueue`` 只留给测试：用 :func:`set_task_queue` 显式注入，不再有「连不上就悄悄回落」的
路径——那条路会把「重启不丢」和跨进程投递一起悄悄取消掉，还没人知道。
"""

from __future__ import annotations

import logging
import os

from app.queue.inprocess import InProcessQueue
from app.queue.ports import (
    TERMINAL_STATES,
    IntentTask,
    TaskHandler,
    TaskQueue,
    TaskState,
    TaskStatus,
)
from app.queue.redis_stream import RedisStreamQueue

logger = logging.getLogger("shoppingx.queue")

__all__ = [
    "TERMINAL_STATES",
    "IntentTask",
    "InProcessQueue",
    "RedisStreamQueue",
    "TaskHandler",
    "TaskQueue",
    "TaskState",
    "TaskStatus",
    "get_task_queue",
    "queue_redis_url",
    "set_task_queue",
]

_queue: TaskQueue | None = None


def queue_redis_url() -> str:
    """队列 Redis 地址。未单独配置时复用事件回放那个实例——键名都带 ``globex:`` 前缀，同一个 db
    里放两套数据不会撞；为它单开一个实例是运维成本，不是隔离收益。"""
    return os.environ.get("QUEUE_REDIS_URL") or os.environ.get(
        "EVENT_REDIS_URL", "redis://localhost:6379/2"
    )


def get_task_queue() -> TaskQueue:
    """返回进程级单例队列（Redis Stream）。客户端建不起来就抛，**不回落进程内**。

    回落曾经是「这个部署没开队列」的兜底，现在没有那种部署了：悄悄改用进程内 deque，等于让 API
    把任务扔进一个没有消费方的队列，用户对着 running 等到天荒地老。真正的 Redis 可达性在启动时
    由 ``server.lifespan`` 拒绝启动那道闸守（阶段 1 条 7）；入队失败由调用方显式处理（503）。
    """
    global _queue
    if _queue is not None:
        return _queue
    try:
        import redis.asyncio as aredis

        client = aredis.from_url(
            queue_redis_url(),
            decode_responses=True,
            socket_connect_timeout=2.0,
            socket_timeout=5.0,  # 比事件回放宽：XREADGROUP 是长轮询，短超时会把 block 打断
        )
    except Exception as exc:  # pragma: no cover - 缺 redis 包 / URL 非法
        url = queue_redis_url()
        raise RuntimeError(f"任务队列的 Redis 客户端建不起来（{url}）：{exc}") from exc
    logger.info("任务队列启用 Redis Stream：%s", queue_redis_url())
    _queue = RedisStreamQueue(client)
    return _queue


def set_task_queue(queue: TaskQueue | None) -> None:
    """注入队列实现（测试 / 手工装配用）；传 ``None`` 复位为按环境变量懒加载。"""
    global _queue
    _queue = queue
