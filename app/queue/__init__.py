"""削峰任务队列。端口见 :mod:`app.queue.ports`，两份实现在 ``redis_stream`` / ``inprocess``。

对外只暴露一个工厂 :func:`get_task_queue`：进程里全局一份，按 ``QUEUE_ENABLED`` 选实现。
``server.py`` 与 ``worker.py`` 都从它拿队列，谁也不该自己 ``RedisStreamQueue(...)``——不然两边各拿
一个客户端、各建一次组，连的还可能不是同一个 URL。
"""

from __future__ import annotations

import logging
import os

from app.queue.inprocess import InProcessQueue
from app.queue.ports import IntentTask, TaskHandler, TaskQueue, TaskState, TaskStatus
from app.queue.redis_stream import RedisStreamQueue
from app.utils.env import env_bool

logger = logging.getLogger("shoppingx.queue")

__all__ = [
    "IntentTask",
    "InProcessQueue",
    "RedisStreamQueue",
    "TaskHandler",
    "TaskQueue",
    "TaskState",
    "TaskStatus",
    "get_task_queue",
    "queue_enabled",
    "set_task_queue",
]

_queue: TaskQueue | None = None


def queue_enabled() -> bool:
    """默认关。开队列意味着多起一个 worker 进程，那是部署形态的变化，不该由默认值替用户决定。"""
    return env_bool("QUEUE_ENABLED", False)


def _redis_url() -> str:
    """队列 Redis 地址。未单独配置时复用事件回放那个实例——键名都带 ``globex:`` 前缀，同一个 db
    里放两套数据不会撞；为它单开一个实例是运维成本，不是隔离收益。"""
    return os.environ.get("QUEUE_REDIS_URL") or os.environ.get(
        "EVENT_REDIS_URL", "redis://localhost:6379/2"
    )


def get_task_queue() -> TaskQueue:
    """返回进程级单例队列。``QUEUE_ENABLED=0``（默认）或 Redis 客户端建不起来时回落进程内。

    注意**建客户端失败才回落，入队失败不回落**：前者是「这个部署没开队列」，后者是「队列开着但
    这一刻挂了」——那时候悄悄改用进程内跑，等于把削峰上限和「重启不丢」一起悄悄取消掉，还没人
    知道。入队失败该由调用方显式处理。
    """
    global _queue
    if _queue is not None:
        return _queue
    if not queue_enabled():
        _queue = InProcessQueue()
        return _queue
    try:
        import redis.asyncio as aredis

        client = aredis.from_url(
            _redis_url(),
            decode_responses=True,
            socket_connect_timeout=2.0,
            socket_timeout=5.0,  # 比事件回放宽：XREADGROUP 是长轮询，短超时会把 block 打断
        )
    except Exception as exc:
        logger.error("QUEUE_ENABLED=1 但 Redis 客户端建不起来，回落进程内队列：%s", exc)
        _queue = InProcessQueue()
        return _queue
    logger.info("任务队列启用 Redis Stream：%s", _redis_url())
    _queue = RedisStreamQueue(client)
    return _queue


def set_task_queue(queue: TaskQueue | None) -> None:
    """注入队列实现（测试 / 手工装配用）；传 ``None`` 复位为按环境变量懒加载。"""
    global _queue
    _queue = queue
