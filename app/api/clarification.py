"""用户澄清的阻塞/恢复桥梁。

``ask_user`` 工具在 Agent loop 里 await 一个 Future；WebSocket handler 收到用户回复后
resolve 该 Future——二者在同一事件循环但不同协程栈中（不共享 ContextVar），用模块级
dict 按 ``thread_id`` 做 key 桥接。

同一 thread 同一时刻最多一个 pending clarification（Agent loop 是串行的）。
单线程 asyncio、无 await 在 dict 操作之间，不需加锁。

**跨进程那一半（批2-4）。** ``QUEUE_ENABLED=1`` 时 loop 跑在 worker 进程，那个 Future 住在
worker 的内存里；用户的回复却是从 API 进程的 WebSocket / HTTP 进来的，``resolve_pending`` 在那边
找不到任何 pending，回复静默丢掉、Agent 干等到超时。补法是一张**等待令牌**：

- worker 侧 ``ask_user`` 建 Future 的同时，把 ``{thread_id, turn_id, reply_id, expires}`` 写进
  Redis（键按 thread_id，TTL 就是提问超时），本进程另存一份用于比对；
- API 侧收到回复先试本地 ``resolve_pending``（单进程模式到此为止，行为一个字节不变），落空再读
  令牌，读到就经控制面 Pub/Sub 把 ``{reply_id, text}`` 广播出去；
- worker 的控制面订阅收到后比对 ``reply_id`` 再 resolve 本地 Future。

**为什么要 ``reply_id`` 而不是只按 thread 投递。** 同一个 thread 会问很多次，用户也可能在上一问
超时之后才点回来。按 thread 盲投会把**上一问的答案**塞给这一问——模型拿着答非所问的输入继续跑，
既不报错也查不出来。令牌过期（键 TTL 到点 / 本地已撤销）即**按取消处理**：迟到的回复一律拒收，
等待方走它原本的超时兜底路径。
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
import uuid
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from app.api import control

logger = logging.getLogger("shoppingx.clarification")

_pending: dict[str, asyncio.Future[str]] = {}

#: Redis 里等待令牌的键前缀（按 thread_id：同一 thread 同时刻只可能有一个 pending 提问）。
_TOKEN_PREFIX = "globex:clarify:"
#: 控制面上澄清回复的指令类型。
KIND_REPLY = "clarify_reply"

#: 本进程持有的等待令牌（thread_id → 令牌）。远端回复要比对它的 reply_id 才算数。
_local_tokens: dict[str, WaitToken] = {}
#: 撤销令牌的后台任务强引用池（同 backplane：不持引用会被 GC 掉，表现为偶发的键残留）。
_cleanups: set[asyncio.Task[Any]] = set()

#: 当前正在消费的队列任务 id，由 worker 在 ``handle_task`` 里写入、令牌生成时读取。
#: 单进程模式下没人写，令牌里的 ``turn_id`` 为空——那条路根本不写令牌，无所谓。
_turn_id_var: ContextVar[str] = ContextVar("shoppingx_turn_id", default="")


def set_turn_id(turn_id: str) -> None:
    """worker 侧调：把「正在跑的这条队列任务」的 id 绑进当前上下文（工具侧只读）。"""
    _turn_id_var.set(turn_id)


@dataclass(frozen=True)
class WaitToken:
    """一次 ``ask_user`` 的等待凭据。``expires`` 是 epoch 秒，与 Redis 键的 TTL 同一个时刻。"""

    thread_id: str
    turn_id: str
    reply_id: str
    expires: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "thread_id": self.thread_id,
            "turn_id": self.turn_id,
            "reply_id": self.reply_id,
            "expires": self.expires,
        }

    @staticmethod
    def from_dict(raw: dict[str, Any]) -> WaitToken:
        return WaitToken(
            thread_id=str(raw["thread_id"]),
            turn_id=str(raw.get("turn_id") or ""),
            reply_id=str(raw["reply_id"]),
            expires=float(raw.get("expires") or 0.0),
        )


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


# ── 跨进程：等待令牌（worker 侧写）─────────────────────────────────────────────
async def register_waiter(thread_id: str, *, timeout_sec: int) -> WaitToken:
    """登记一次等待：生成令牌、写 Redis（若控制面启用），并在本进程留一份用于比对。

    **必须在 ``clarification_request`` 事件发出之前调**：事件经背板到浏览器只是几毫秒，脚本类
    客户端完全可能在事件送达的下一跳就把回复打回来。令牌后写就会撞上「回复到了、还没人登记等待」
    的窗口，那条回复只能被拒收。
    """
    token = WaitToken(
        thread_id=thread_id,
        turn_id=_turn_id_var.get(""),
        reply_id=uuid.uuid4().hex,
        expires=time.time() + max(1, timeout_sec),
    )
    _local_tokens[thread_id] = token
    bus = control.get_control_bus()
    if bus is not None:
        await bus.store(
            f"{_TOKEN_PREFIX}{thread_id}",
            json.dumps(token.to_dict(), ensure_ascii=False),
            max(1, timeout_sec),
        )
    return token


def clear_waiter(thread_id: str, token: WaitToken | None = None) -> None:
    """撤销等待（``ask_user`` 的 finally 调）。**同步撤本地、异步删远端**。

    本地那一份必须当场撤掉：它是「这条回复算不算数」的判据，晚一步就可能把迟到的回复认成有效。
    Redis 那份走后台任务——本函数最常见的调用时机恰恰是任务被取消的 finally，在那里 await 会立刻
    再吃一个 ``CancelledError``、键就删不成了；删不掉也不要紧，它本来就带着与提问超时等长的 TTL。
    """
    current = _local_tokens.get(thread_id)
    if token is not None and current is not None and current.reply_id != token.reply_id:
        return  # 已经是下一问的令牌了，别把它撤掉
    _local_tokens.pop(thread_id, None)
    bus = control.get_control_bus()
    if bus is None:
        return
    try:
        task = asyncio.create_task(bus.drop(f"{_TOKEN_PREFIX}{thread_id}"))
    except RuntimeError:  # 无运行中的事件循环
        return
    _cleanups.add(task)
    task.add_done_callback(_cleanups.discard)


async def drop_stale_waiter(thread_id: str, *, turn_id: str) -> None:
    """worker 领到新任务时清一次残留令牌：上一轮被 SIGKILL 掐死的话，键会留到 TTL 到点。

    只清 ``turn_id`` 不是本轮的那些——同一轮里正当的等待不能误伤（重投场景下 turn_id 相同）。
    """
    bus = control.get_control_bus()
    if bus is None:
        return
    raw = await bus.load(f"{_TOKEN_PREFIX}{thread_id}")
    if raw is None:
        return
    try:
        token = WaitToken.from_dict(json.loads(raw))
    except (json.JSONDecodeError, ValueError, KeyError, TypeError):
        await bus.drop(f"{_TOKEN_PREFIX}{thread_id}")
        return
    if token.turn_id and token.turn_id == turn_id:
        return
    logger.info("清理上一轮残留的澄清令牌：thread_id=%s（turn=%s）", thread_id, token.turn_id)
    await bus.drop(f"{_TOKEN_PREFIX}{thread_id}")


# ── 跨进程：回复投递（API 侧发、worker 侧收）───────────────────────────────────
#: :func:`deliver_reply` 的结果。``local``=本进程就有人等着（单进程模式的全部）；``forwarded``=
#: 已广播给持有令牌的那个进程；``no_waiter``=没人在等（含令牌已过期，按取消处理）；
#: ``publish_failed``=有人在等但控制面发不出去（Redis 挂了）。
DeliveryResult = str


async def deliver_reply(thread_id: str, text: str) -> DeliveryResult:
    """把用户的澄清回复交给正在等它的那个 ``ask_user``，无论它在哪个进程。

    **顺序是「先本地后远端」，不是并列的两条路**：本进程有 pending 就说明 loop 就在这里跑，
    远端广播纯属多余（还会让别的进程收到一条投不出去的指令）。单进程模式在第一行就返回，与
    批2-4 之前逐字同义。
    """
    if resolve_pending(thread_id, text):
        return "local"
    bus = control.get_control_bus()
    if bus is None:
        return "no_waiter"
    raw = await bus.load(f"{_TOKEN_PREFIX}{thread_id}")
    if raw is None:
        return "no_waiter"  # 键的 TTL 到点即令牌过期 —— 迟到的回复按取消处理，不投递
    try:
        token = WaitToken.from_dict(json.loads(raw))
    except (json.JSONDecodeError, ValueError, KeyError, TypeError) as exc:
        logger.warning("澄清令牌解析失败：thread_id=%s（%s）", thread_id, exc)
        return "no_waiter"
    if token.expires and token.expires <= time.time():
        # 键还在但时刻已过（TTL 与 expires 之间总有秒级误差）——以令牌里的时刻为准，宁可拒收。
        await bus.drop(f"{_TOKEN_PREFIX}{thread_id}")
        return "no_waiter"
    ok = await bus.publish(KIND_REPLY, thread_id=thread_id, reply_id=token.reply_id, text=text)
    return "forwarded" if ok else "publish_failed"


async def on_reply_message(payload: dict[str, Any]) -> None:
    """控制面订阅端的处理器（worker 侧注册）：比对 reply_id 后 resolve 本地 Future。

    比对不上就丢弃——那是**上一问**的答案迟到了。把它塞给当前这一问不会报错，只会让模型拿着
    答非所问的输入继续跑，是最难查的一类脏数据。
    """
    thread_id = str(payload.get("thread_id") or "")
    reply_id = str(payload.get("reply_id") or "")
    if not thread_id or not reply_id:
        return
    token = _local_tokens.get(thread_id)
    if token is None or token.reply_id != reply_id:
        logger.info("丢弃过期 / 不匹配的澄清回复：thread_id=%s reply_id=%s", thread_id, reply_id)
        return
    if not resolve_pending(thread_id, str(payload.get("text") or "")):
        logger.info("澄清回复到达时已无人等待：thread_id=%s", thread_id)


def register_control_handlers() -> None:
    """把回复处理器挂到控制面上（worker 启动时调一次；控制面没启用则是空操作）。"""
    bus = control.get_control_bus()
    if bus is not None:
        bus.on(KIND_REPLY, on_reply_message)


async def _drain_cleanups() -> None:
    """等后台的撤销任务收干净（测试 / 关服用；生产路径不必调）。"""
    for task in list(_cleanups):
        with contextlib.suppress(Exception, asyncio.CancelledError):
            await task
