"""跨进程控制面（批 2）——把「取消」「澄清回复」这两条**反方向**的指令从 API 送到 worker。

**与事件背板的关系是一对镜像。** :mod:`app.api.backplane` 解决的是 worker → 浏览器（AGUI 事件往
外流）；本模块解决的是浏览器 → worker（用户的指令往里流）。队列模式下这两条路都断着：任务在
worker 进程里跑，而 WebSocket 与 HTTP 端点都挂在 API 进程上，于是 ``POST /api/task/{tid}/cancel``
只掐得掉 API 侧那个等结果的影子协程（worker 照样把这一轮跑完、照样烧 token），``ask_user`` 的回复
更是永远送不到那个 await 着 Future 的协程。

**为什么不复用背板那条通道。** 语义相反：事件是「尽力而为、丢了只是少看几条」，指令是「丢了用户
就掐不掉任务 / 永远等不到回答」。两者的降级方向、日志级别、是否需要落一份可查的标记都不一样——
合成一条通道就只能按最松的那套来。所以这里另开一个频道，且**发布失败要看得见**（warning + 调用方
拿得到布尔），不像背板那样只打 debug。

**指令通道的三个必答问题：**

1. **消息可能比任务先到**（用户手快、或 worker 还没领到这条任务）。所以取消**不只发一条广播**，
   还要在 Redis 落一个按 ``task_id`` 的标记；worker 领到任务的第一件事就是查它。只发广播的话，
   还在队列里排着的任务取消不掉——那恰恰是最该取消的那些。
2. **标记必须按 ``task_id`` 而不是 ``thread_id``**。覆盖重发（同一个 thread 换个问题重问）会紧接着
   入一条新任务，按 thread 打的标记会把刚提交的新任务也一并掐掉，且用户完全看不出为什么。
3. **本进程也可能就是执行方**（``QUEUE_ENABLED=1`` 但 Redis 建不起来时 API 兼任 worker）。所以先查
   一遍进程内的在飞登记表（:func:`cancel_local`），命中就地掐掉，不劳 Redis 跑一趟。
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from collections.abc import Awaitable, Callable
from contextlib import suppress
from typing import Any

from app.utils.env import env_bool, env_int, env_str

logger = logging.getLogger("shoppingx.control")

CHANNEL = env_str("CONTROL_CHANNEL", "globex:control")
# 本进程身份，用途只有一个：别把自己发的指令又当远端指令执行一遍（同背板的 origin）。
ORIGIN = uuid.uuid4().hex
_CANCEL_PREFIX = "globex:cancel:"
# 取消标记的存活时长。要长于「任务在队列里最久能躺多久 + 单轮耗时」，否则标记先过期、任务后被领走，
# 取消就丢了；又不能长到无限，那是一堆永不回收的键。
_CANCEL_TTL = env_int("CONTROL_CANCEL_TTL", 1800)
_RETRY_SECONDS = 2.0

# 本进程正在跑的队列任务：task_id → 跑它的 asyncio task。worker 侧登记，取消侧查。
_inflight: dict[str, tuple[str, asyncio.Task[Any]]] = {}
# 已经被「用户取消」掐过的 task_id。handle_task 靠它把两种 CancelledError 分开：用户取消要 ack
# （别重投），优雅退出的取消不能 ack（留在 PEL 里等下一个 worker 捡）。
_cancelled_local: set[str] = set()
# fire-and-forget 的发布任务强引用池（理由同 backplane：不持引用会被 GC 掉，表现为偶发丢指令）。
_pending_publishes: set[asyncio.Task[Any]] = set()

Handler = Callable[[dict[str, Any]], Awaitable[None]]


def control_enabled() -> bool:
    """默认跟随 ``QUEUE_ENABLED``：单进程部署里 API 与 worker 是同一个进程，指令直接走内存。"""
    return env_bool("CONTROL_ENABLED", env_bool("QUEUE_ENABLED", False))


def _redis_url() -> str:
    """复用队列 / 事件那个实例——键名与频道名都带 ``globex:`` 前缀，同一个 db 里不会撞。"""
    return (
        os.environ.get("CONTROL_REDIS_URL")
        or os.environ.get("QUEUE_REDIS_URL")
        or os.environ.get("EVENT_REDIS_URL", "redis://localhost:6379/2")
    )


class ControlBus:
    """一个 Redis Pub/Sub 频道上的指令发布端 + 订阅端，外加取消标记的读写。

    订阅端按 ``kind`` 分发给注册进来的处理器（``cancel`` / ``clarify_reply``），这样 worker 只需起
    **一条**订阅循环、一个 Redis 连接，新增指令类型不必再开一路。
    """

    def __init__(self, client: Any, *, channel: str = CHANNEL, origin: str = ORIGIN) -> None:
        self._client = client
        self._channel = channel
        self._origin = origin
        self._handlers: dict[str, Handler] = {}
        self._reader: asyncio.Task[None] | None = None
        self._stopping = False

    # ── 发布端 ───────────────────────────────────────────────────────────────
    def on(self, kind: str, handler: Handler) -> None:
        """注册一类指令的处理器（重复注册即覆盖，测试里换桩用）。"""
        self._handlers[kind] = handler

    async def publish(self, kind: str, **fields: Any) -> bool:
        """发一条指令。失败返回 ``False`` 并打 **warning**：指令丢了用户有感，不能只记 debug。"""
        envelope = json.dumps({"origin": self._origin, "kind": kind, **fields}, ensure_ascii=False)
        try:
            await self._client.publish(self._channel, envelope)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("控制指令发布失败（kind=%s）：%s", kind, exc)
            return False
        return True

    # ── 带 TTL 的小键（取消标记 / 澄清等待令牌共用）───────────────────────────
    #
    # 指令是广播，广播只能送到「此刻在线且已订阅」的进程；而这两件事都存在「指令先到、执行方后来」
    # 的时序（任务还在队列里排着 / worker 刚重启）。所以广播之外必须再落一份**可查的**小状态，
    # 让后到的那一方自己发现。键都带 TTL：控制面的状态过期即失效，不该留成垃圾。
    async def store(self, key: str, value: str, ttl: int) -> bool:
        try:
            await self._client.set(key, value, ex=ttl)
        except Exception as exc:
            logger.warning("控制面写键失败：%s（%s）", key, exc)
            return False
        return True

    async def load(self, key: str) -> str | None:
        try:
            raw = await self._client.get(key)
        except Exception as exc:
            logger.debug("控制面读键失败：%s（%s）", key, exc)
            return None
        return None if raw is None else (raw.decode() if isinstance(raw, bytes) else str(raw))

    async def drop(self, key: str) -> None:
        try:
            await self._client.delete(key)
        except Exception as exc:
            logger.debug("控制面删键失败（有 TTL 兜底）：%s（%s）", key, exc)

    async def mark_cancel(self, task_id: str) -> bool:
        """落一个「这条任务已被取消」的标记，供**还没开始跑**的任务在被领走时自查。"""
        return await self.store(f"{_CANCEL_PREFIX}{task_id}", "1", _CANCEL_TTL)

    async def was_cancelled(self, task_id: str) -> bool:
        """查标记。读不到（Redis 抖 / 没开）一律当「没取消」——宁可多跑一轮，不能凭一次读失败
        就把用户提交的任务扔掉。"""
        return await self.load(f"{_CANCEL_PREFIX}{task_id}") is not None

    async def clear_cancel(self, task_id: str) -> None:
        await self.drop(f"{_CANCEL_PREFIX}{task_id}")

    # ── 订阅端 ───────────────────────────────────────────────────────────────
    async def start(self) -> None:
        """开始订阅（幂等）。断线自己退避重连——理由同背板：断了不会报错，只是从此再也收不到指令。"""
        if self._reader is not None:
            return
        self._stopping = False
        self._reader = asyncio.create_task(self._read_loop())
        logger.info("控制面已订阅 %s（origin=%s）", self._channel, self._origin[:8])

    async def _read_loop(self) -> None:
        while not self._stopping:
            pubsub = None
            try:
                pubsub = self._client.pubsub()
                await pubsub.subscribe(self._channel)
                async for message in pubsub.listen():
                    if self._stopping:
                        break
                    await self._dispatch(message)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("控制面订阅中断，%.0fs 后重连：%s", _RETRY_SECONDS, exc)
            finally:
                if pubsub is not None:
                    await _quiet_close(pubsub)
            if not self._stopping:
                await asyncio.sleep(_RETRY_SECONDS)

    async def _dispatch(self, message: Any) -> None:
        """一条 Pub/Sub 消息 → 对应 kind 的处理器。不相干 / 自己发的一律跳过。"""
        if not isinstance(message, dict) or message.get("type") != "message":
            return
        raw = message.get("data")
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", "replace")
        if not isinstance(raw, str):
            return
        try:
            payload = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return
        if not isinstance(payload, dict) or payload.get("origin") == self._origin:
            return
        handler = self._handlers.get(str(payload.get("kind") or ""))
        if handler is None:
            return
        try:
            await handler(payload)
        except asyncio.CancelledError:
            raise
        except Exception:
            # 一条指令处理炸了不能把订阅循环带走——那会让这个副本**永久**收不到后续指令。
            logger.exception("控制指令处理失败：%s", payload.get("kind"))

    async def stop(self) -> None:
        self._stopping = True
        if self._reader is not None:
            self._reader.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await self._reader
            self._reader = None
        await _quiet_close(self._client)


async def _quiet_close(obj: Any) -> None:
    """关 redis 客户端 / pubsub（``aclose`` 优先），失败只记 debug。"""
    for name in ("aclose", "close"):
        closer = getattr(obj, name, None)
        if closer is None:
            continue
        try:
            result = closer()
            if asyncio.iscoroutine(result):
                await result
        except Exception as exc:
            logger.debug("控制面关闭 %s 失败：%s", name, exc)
        return


# ── 进程级单例 ───────────────────────────────────────────────────────────────
# 与 app.queue / backplane 同一套取舍：谁也不该自己 ``ControlBus(...)``，否则一个进程里出现两个
# origin，自己发的指令在自己这儿去重不掉；``_resolved`` 记「已经决定过了」，关着时不必反复读环境。
_bus: ControlBus | None = None
_resolved = False


def get_control_bus() -> ControlBus | None:
    """返回进程级控制面；未启用 / Redis 客户端建不起来时返回 ``None``（退化为纯进程内取消）。"""
    global _bus, _resolved
    if _resolved:
        return _bus
    _resolved = True
    if not control_enabled():
        return None
    try:
        import redis.asyncio as aredis  # 可选依赖，懒加载（与 queue / backplane 同源）

        client = aredis.from_url(
            _redis_url(),
            decode_responses=True,
            socket_connect_timeout=2.0,
            # 不设 socket_timeout：``listen()`` 天然长时间没数据，设了会被打断成假故障、无限重连。
            # 这与队列客户端（socket_timeout=5.0）的取舍相反，两处别互相抄（背板同此坑）。
        )
    except Exception as exc:
        logger.warning("控制面初始化失败，跨进程取消 / 澄清关闭：%s", exc)
        return None
    _bus = ControlBus(client)
    logger.info("跨进程控制面启用：%s（频道 %s）", _redis_url(), CHANNEL)
    return _bus


def set_control_bus(bus: ControlBus | None) -> None:
    """注入实例（测试 / 手工装配用）；传 ``None`` 即彻底关掉（不再按环境变量懒加载）。"""
    global _bus, _resolved
    _bus = bus
    _resolved = True


def reset_control_bus() -> None:
    """复位为「按环境变量重新决定」，并清空进程内登记（测试收尾用）。"""
    global _bus, _resolved
    _bus = None
    _resolved = False
    _inflight.clear()
    _cancelled_local.clear()


# ── 在飞任务登记（worker 侧写，取消侧读）───────────────────────────────────────
def register_inflight(task_id: str, thread_id: str, task: asyncio.Task[Any]) -> None:
    """登记「本进程正在跑这条队列任务」。worker 的 handle_task 一进来就调。"""
    _inflight[task_id] = (thread_id, task)


def unregister_inflight(task_id: str) -> None:
    _inflight.pop(task_id, None)
    _cancelled_local.discard(task_id)


def cancel_local(task_id: str | None = None, thread_id: str | None = None) -> bool:
    """就地掐掉本进程在跑的那条任务，命中返回 ``True``。

    **按 task_id 优先、thread_id 兜底**：远端指令一定带 task_id（精确）；单进程 / 兼任 worker 的
    场景下调用方手里可能只有 thread_id。掐之前先把 task_id 记进 :data:`_cancelled_local` ——
    ``handle_task`` 靠它区分「用户取消（要 ack，别重投）」与「优雅退出（不 ack，交还队列）」。
    """
    hit: str | None = None
    if task_id and task_id in _inflight:
        hit = task_id
    elif thread_id:
        hit = next((tid for tid, (th, _t) in _inflight.items() if th == thread_id), None)
    if hit is None:
        return False
    _, task = _inflight[hit]
    _cancelled_local.add(hit)
    if not task.done():
        task.cancel()
    return True


def was_cancelled_locally(task_id: str) -> bool:
    """这条任务是不是被「用户取消」掐的（同步查，不碰 Redis）。"""
    return task_id in _cancelled_local


async def request_cancel(thread_id: str, task_id: str | None) -> bool:
    """请求取消一条任务：先就地试，再落标记 + 广播。返回「是否已把取消送达某处」。

    三件事的顺序不能换：**先本地**（可能就是自己在跑，一步到位）→ **再落标记**（任务还在队列里
    排着，靠它在被领走时自杀）→ **最后广播**（任务已经被某个 worker 领走，靠它立刻掐断）。
    先广播后落标记会有一个窗口：广播时任务还没被领走，没人响应；标记又还没写，等它被领走时也查
    不到——取消就凭空丢了。
    """
    if cancel_local(task_id=task_id, thread_id=thread_id):
        return True
    bus = get_control_bus()
    if bus is None or not task_id:
        return False
    marked = await bus.mark_cancel(task_id)
    published = await bus.publish("cancel", thread_id=thread_id, task_id=task_id)
    return marked or published


def request_cancel_nowait(thread_id: str, task_id: str | None) -> None:
    """:func:`request_cancel` 的不等待版本，给「不允许出现 await」的调用点用（覆盖重发那一段）。

    本地那一半是同步的，会**当场**执行；剩下的 Redis 往返丢进后台任务并持强引用（不持引用会被
    GC 掉，表现为偶发取消不掉且无法复现）。
    """
    if cancel_local(task_id=task_id, thread_id=thread_id):
        return
    if get_control_bus() is None or not task_id:
        return
    try:
        task = asyncio.create_task(request_cancel(thread_id, task_id))
    except RuntimeError:  # 无运行中的事件循环（同步上下文 / 测试）
        return
    _pending_publishes.add(task)
    task.add_done_callback(_pending_publishes.discard)


async def consume_cancel_mark(task_id: str) -> bool:
    """worker 领到任务时自查：这条是不是在排队期间已经被取消了。命中即顺手把标记清掉。"""
    bus = get_control_bus()
    if bus is None:
        return False
    if not await bus.was_cancelled(task_id):
        return False
    await bus.clear_cancel(task_id)
    return True


async def close_control_bus() -> None:
    """关服时收掉控制面（**只关已经建出来的那个**，不会顺手把它创建出来）。"""
    if _bus is not None:
        await _bus.stop()


async def clear_cancel_mark(task_id: str) -> None:
    """兑现完一条取消之后把标记清掉（worker 侧调）。

    不清也不会出错——标记有 TTL，且这条消息已经 ack 过、不会再被投递。清它是为了别在 Redis 里
    留半小时的垃圾键：取消是可能被连点好几次的操作，攒起来就不只是一两个了。
    """
    bus = get_control_bus()
    if bus is not None:
        await bus.clear_cancel(task_id)
