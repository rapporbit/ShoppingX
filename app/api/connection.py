"""ConnectionManager —— ``thread_id → WebSocket`` 的路由表。

一个浏览器页面对应一个 ``thread_id``，发起任务后用它建一条 WebSocket 长连接订阅过程
事件。后台 AgentLoop 在任意深处上报事件时，:mod:`app.api.monitor` 只知道当前 ContextVar
里的 ``thread_id``，并不知道连接对象在哪——由本路由表负责「按 thread_id 找到那条连接并
推过去」。这样工具/子 Agent 上报事件时完全不必关心连接细节。

两个容易踩的坑，这里都处理掉：

1. **重连误删**：用户刷新页面会建一条**新** WebSocket，**旧**连接稍后才触发断开回调。
   若断开时按 ``thread_id`` 盲删，会把刚建好的新连接一起删掉（串台）。所以 :meth:`disconnect`
   只在「当前登记的就是要断开的这个对象」时才删（``is`` 身份比较）。
2. **并发改表**：asyncio 单线程里多个协程会交替读写连接表，用 ``asyncio.Lock`` 串行化
   增删，避免迭代中被改。``send_json`` 的网络等待放在锁外，避免一条慢连接卡住所有上报。
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger("shoppingx.connection")


@runtime_checkable
class WebSocketLike(Protocol):
    """只用到 WebSocket 的这两个动作；定成 Protocol 便于测试注入假连接。

    FastAPI 的 ``WebSocket`` 结构上满足本协议（``accept`` / ``send_json`` 都有），
    因此无需在模块层硬依赖 FastAPI。
    """

    async def accept(self) -> None: ...

    async def send_json(self, data: Any) -> None: ...


class ConnectionManager:
    """维护 ``thread_id → WebSocket`` 映射，负责注册、按身份注销、按 thread 推送。"""

    def __init__(self) -> None:
        self._connections: dict[str, WebSocketLike] = {}
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocketLike, thread_id: str) -> None:
        """接受握手并登记连接（同 thread 再次连接会覆盖旧登记，对应刷新重连）。"""
        await websocket.accept()
        async with self._lock:
            self._connections[thread_id] = websocket
        logger.debug("WS connected: thread_id=%s", thread_id)

    async def disconnect(self, websocket: WebSocketLike, thread_id: str) -> None:
        """注销连接——仅当登记的就是该对象时才删，避免误删刷新后的新连接。"""
        async with self._lock:
            if self._connections.get(thread_id) is websocket:
                del self._connections[thread_id]
                logger.debug("WS disconnected: thread_id=%s", thread_id)

    async def send_to_thread(self, thread_id: str, payload: dict[str, Any]) -> bool:
        """把事件推给某 thread 的连接；无连接返回 ``False``，推送失败摘除该连接。"""
        async with self._lock:
            websocket = self._connections.get(thread_id)
        if websocket is None:
            return False
        try:
            await websocket.send_json(payload)
            return True
        except Exception as exc:  # 连接已死/写失败：摘除，绝不让上报异常冒泡进 AgentLoop
            logger.debug("WS send failed for thread_id=%s: %s; dropping", thread_id, exc)
            await self.disconnect(websocket, thread_id)
            return False

    def is_connected(self, thread_id: str) -> bool:
        """该 thread 当前是否有活跃连接（前端可据此决定要不要重连）。"""
        return thread_id in self._connections

    @property
    def connection_count(self) -> int:
        """当前活跃连接数（监控/调试用）。"""
        return len(self._connections)
