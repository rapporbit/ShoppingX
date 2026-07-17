"""用户澄清的阻塞/恢复桥梁。

``ask_user`` 工具在 Agent loop 里 await 一个 Future；WebSocket handler 收到用户回复后
resolve 该 Future——二者在同一事件循环但不同协程栈中（不共享 ContextVar），用模块级
dict 按 ``thread_id`` 做 key 桥接。

同一 thread 同一时刻最多一个 pending clarification（Agent loop 是串行的）。
单线程 asyncio、无 await 在 dict 操作之间，不需加锁。
"""

from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger("shoppingx.clarification")

_pending: dict[str, asyncio.Future[str]] = {}


def create_pending(thread_id: str) -> asyncio.Future[str]:
    """为该 thread 创建一个 pending Future。已有则先 cancel 再替换（防泄漏）。"""
    old = _pending.get(thread_id)
    if old is not None and not old.done():
        old.cancel()
        logger.debug("replaced existing pending clarification: thread_id=%s", thread_id)
    loop = asyncio.get_running_loop()
    fut: asyncio.Future[str] = loop.create_future()
    _pending[thread_id] = fut
    return fut


def resolve_pending(thread_id: str, text: str) -> bool:
    """从 WS handler 调用：resolve Future 并移除。返回是否成功（False=无 pending 或已超时）。"""
    fut = _pending.pop(thread_id, None)
    if fut is None or fut.done():
        return False
    fut.set_result(text)
    return True


def cancel_pending(thread_id: str) -> None:
    """任务取消/结束时清理：cancel Future（若还在 await 会抛 CancelledError）并移除。"""
    fut = _pending.pop(thread_id, None)
    if fut is not None and not fut.done():
        fut.cancel()
        logger.debug("cancelled pending clarification: thread_id=%s", thread_id)


def has_pending(thread_id: str) -> bool:
    """该 thread 是否有正在等待的澄清（调试 / health 用）。"""
    fut = _pending.get(thread_id)
    return fut is not None and not fut.done()
