"""事件背板（批 2）——把 AGUI 事件从「产生它的进程」送到「挂着那条 WebSocket 的进程」。

**解决什么。** `QUEUE_ENABLED=1` 之后，跑 AgentLoop 的是独立 worker 进程，而浏览器的 WebSocket
连在 API 进程上。事件由 worker 侧的 :class:`ConnectionManager` 发出，那张路由表里一条连接都没有
——于是每一条 `tool_start` / `summary_delta` / `task_result` 都静默丢掉，用户对着一个转圈的界面等
到收尾（`GET /api/task/{id}` 与落盘历史仍然是对的，但实时性没了）。这正是批2-2 报告里点名的
「队列模式此刻不适合上线给前端用」。背板补的就是这一跳：worker 把送不出去的事件广播到 Redis
Pub/Sub，API 进程订阅同一个频道，收到后查自己的连接表，有就推给前端。

**为什么是 Pub/Sub 而不是 Stream。** 事件回放已经有 Stream 了（:mod:`app.api.event_log`，按
thread 一条流、带 id、可补发）。背板要的是另一件事：**此刻**把消息扇出给所有在线进程，没有订阅者
就该丢掉——Pub/Sub 的「不持久化」在这里不是缺点而是需求，持久化那一半已经由 event_log 承担了。
两者分工清楚：断线重连的缺口找 Stream 补，实时直播走 Pub/Sub。

**降级方向：静默降级（沿用批2-1 的口径）。** 背板属事件侧，与队列相反——Redis 挂了只是少看几条
实时事件（收尾结果照样从状态表拿得到、历史照样落盘），绝不能让它把主链路拖垮。所以发布是
fire-and-forget、订阅循环自己退避重连，任何异常都只打日志。

**两个必须写对的细节：**

1. **`origin` 跳过自己。** 每个进程启动时生成一个随机 origin，发布时带上、收到时比对——不然多副本
   API 会把自己刚推给前端的事件从背板上再收一遍、再推一次，前端看到每条事件出现 N 遍（N = 副本数）。
2. **fire-and-forget 的任务要持强引用。** `asyncio.create_task` 的返回值不留引用时，事件循环只持
   弱引用，任务可能在跑完之前被 GC 掉——表现是「偶尔丢事件」，且完全无法复现。所以放进一个 set，
   `add_done_callback` 里再移除。
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from contextlib import suppress
from typing import Any

from app.api.connection import ConnectionManager
from app.utils.env import env_bool, env_int, env_str

logger = logging.getLogger("shoppingx.backplane")

# 全进程共用一个频道：事件量级是「每轮任务几十条」，按 thread 开频道要在每次 WS 连接/断开时
# subscribe/unsubscribe，多一套生命周期还换不来什么——收端一次字典查表就能滤掉不相干的 thread。
CHANNEL = env_str("BACKPLANE_CHANNEL", "globex:agui")
# 本进程的身份。只在内存里活着，进程重启换一个即可——它唯一的用途是「这条是不是我自己发的」。
ORIGIN = uuid.uuid4().hex
# 订阅循环断开后的重连间隔（秒）。
_RETRY_SECONDS = 2.0
# 未完成的发布任务上限。Redis 卡住（连得上但不响应）时任务会一直堆，不设顶就是慢性内存泄漏；
# 超过就丢弃新的发布——丢事件总好过把跑任务的进程拖垮，这与本模块的降级方向一致。
_MAX_INFLIGHT = env_int("BACKPLANE_MAX_INFLIGHT", 1000)


def backplane_enabled() -> bool:
    """默认跟随 ``QUEUE_ENABLED``：单进程部署下背板没有任何用处，多一个 Redis 依赖反而是风险。

    直接读环境变量而不 import ``app.queue``：本模块被 :mod:`app.api.monitor` 引用，而 monitor 几乎
    被所有工具引用——不把队列包拖进这条 import 链上，省掉一整类循环导入的隐患。默认值与
    ``app.queue.queue_enabled`` 同源（都是 ``QUEUE_ENABLED`` 默认关），此处不引入第二个开关口径。
    """
    return env_bool("BACKPLANE_ENABLED", env_bool("QUEUE_ENABLED", False))


def _redis_url() -> str:
    """背板 Redis 地址：不单配则依次复用队列、事件回放那个实例（键名/频道名都带前缀，不撞）。"""
    return (
        os.environ.get("BACKPLANE_REDIS_URL")
        or os.environ.get("QUEUE_REDIS_URL")
        or os.environ.get("EVENT_REDIS_URL", "redis://localhost:6379/2")
    )


class EventBackplane:
    """一个 Redis Pub/Sub 频道上的发布端 + 订阅端。

    发布端由 :mod:`app.api.monitor` 在每条送不出去的事件上调用（见 :meth:`publish_nowait`）；
    订阅端由 API 进程在 lifespan 里启动（见 :meth:`start`），把远端事件转交给本进程的
    :class:`ConnectionManager`。两端在同一个类里，因为它们共用 ``origin`` —— 发布时写、订阅时比。
    """

    def __init__(
        self, client: Any, *, channel: str = CHANNEL, origin: str = ORIGIN, max_inflight: int = 0
    ) -> None:
        self._client = client
        self._channel = channel
        self._origin = origin
        self._max_inflight = max_inflight or _MAX_INFLIGHT
        # 强引用池：fire-and-forget 的任务放这里，防止跑完之前被 GC 掉（见模块 docstring 第 2 点）。
        self._tasks: set[asyncio.Task[bool]] = set()
        self._reader: asyncio.Task[None] | None = None
        self._manager: ConnectionManager | None = None
        self._stopping = False
        self.dropped = 0  # 因在途上限被丢弃的发布数（调试/观测用）

    # ── 发布端 ───────────────────────────────────────────────────────────────
    def publish_nowait(self, payload: dict[str, Any]) -> None:
        """发布一条 AGUI 事件，**不等它发完**。

        上报点在 AgentLoop 的最深处，多等一次 Redis 往返就是给每条事件加一次延迟；而事件送不送得
        到本就是尽力而为的事。没有运行中的事件循环（同步上下文 / 测试）时直接跳过。
        """
        if self._stopping:
            return
        if len(self._tasks) >= self._max_inflight:
            self.dropped += 1
            logger.warning("背板在途发布已达 %d 条，丢弃本条事件", self._max_inflight)
            return
        try:
            task = asyncio.create_task(self.publish(payload))
        except RuntimeError:  # 无运行中的事件循环：不是错误，只是这条送不出去
            return
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def publish(self, payload: dict[str, Any]) -> bool:
        """真正发一条（带 ``origin`` 信封）。失败只记日志并返回 ``False``——静默降级。"""
        envelope = json.dumps({"origin": self._origin, "payload": payload}, ensure_ascii=False)
        try:
            await self._client.publish(self._channel, envelope)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.debug("背板发布失败（静默降级）：%s", exc)
            return False
        return True

    # ── 订阅端 ───────────────────────────────────────────────────────────────
    async def start(self, manager: ConnectionManager) -> None:
        """开始订阅，把远端进程的事件转发给 ``manager`` 里挂着的 WebSocket（幂等）。"""
        if self._reader is not None:
            return
        self._manager = manager
        self._stopping = False
        self._reader = asyncio.create_task(self._read_loop())
        logger.info("事件背板已订阅 %s（origin=%s）", self._channel, self._origin[:8])

    async def _read_loop(self) -> None:
        """订阅 → 收 → 转发；断了就退避重连。整个循环里没有一条会往外抛的路径。

        为什么要自己重连：Pub/Sub 连接是长连接，Redis 重启 / 网络抖一下就断，而断了之后**不会有
        任何报错**——只是从此再也收不到事件。不重连的话，一次几秒的网络抖动就会让这个 API 副本
        永久性地变成「前端不再有实时事件」，且没人看得出来。
        """
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
                logger.warning("背板订阅中断，%.0fs 后重连：%s", _RETRY_SECONDS, exc)
            finally:
                if pubsub is not None:
                    await _quiet_close(pubsub)
            if not self._stopping:
                await asyncio.sleep(_RETRY_SECONDS)

    async def _dispatch(self, message: Any) -> None:
        """把一条 Pub/Sub 消息还原成 AGUI 事件并推给本进程的连接（不相干的一律静默跳过）。"""
        if not isinstance(message, dict) or message.get("type") != "message":
            return  # subscribe 确认帧等控制消息
        raw = message.get("data")
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", "replace")
        if not isinstance(raw, str):
            return
        try:
            envelope = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return
        if not isinstance(envelope, dict) or envelope.get("origin") == self._origin:
            return  # 自己发的：本进程早就直投过一次了，再推一遍就是重复事件
        payload = envelope.get("payload")
        if not isinstance(payload, dict):
            return
        thread_id = payload.get("thread_id")
        if not thread_id or self._manager is None:
            return
        # 没有这条 thread 的连接时 send_to_thread 返回 False，本进程什么也不做——收端过滤就在这。
        await self._manager.send_to_thread(str(thread_id), payload)

    async def stop(self) -> None:
        """停订阅、收干净在途发布任务、关客户端。关服路径调用，异常一律吞掉。"""
        self._stopping = True
        if self._reader is not None:
            self._reader.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await self._reader
            self._reader = None
        for task in list(self._tasks):
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        await _quiet_close(self._client)


async def _quiet_close(obj: Any) -> None:
    """关掉 redis 客户端 / pubsub（``aclose`` 优先，退回 ``close``），失败只记 debug。"""
    for name in ("aclose", "close"):
        closer = getattr(obj, name, None)
        if closer is None:
            continue
        try:
            result = closer()
            if asyncio.iscoroutine(result):
                await result
        except Exception as exc:
            logger.debug("背板关闭 %s 失败：%s", name, exc)
        return


# ── 进程级单例与入口 ─────────────────────────────────────────────────────────
# 与 app.queue 的工厂同一套取舍：谁也不该自己 ``EventBackplane(...)``，否则一个进程里会出现两个
# origin，自己发的事件在自己这儿也去重不掉。``_resolved`` 记「已经决定过了」——关着的时候不必
# 每条事件都重读一次环境变量。
_backplane: EventBackplane | None = None
_resolved = False


def get_backplane() -> EventBackplane | None:
    """返回进程级背板；未启用 / Redis 客户端建不起来时返回 ``None``（静默降级为单进程行为）。"""
    global _backplane, _resolved
    if _resolved:
        return _backplane
    _resolved = True
    if not backplane_enabled():
        return None
    try:
        import redis.asyncio as aredis  # 可选依赖，懒加载（与 event_log / queue 同源）

        client = aredis.from_url(
            _redis_url(),
            decode_responses=True,
            socket_connect_timeout=2.0,
            # 订阅是长连接、``listen()`` 天然长时间没数据，设 socket_timeout 会把它打断成假故障。
            # 发布侧的失败由 connect_timeout 与断路器之外的 try/except 兜（本模块一律静默降级）。
        )
    except Exception as exc:
        logger.warning("事件背板初始化失败，跨进程事件转发关闭：%s", exc)
        return None
    _backplane = EventBackplane(client)
    logger.info("事件背板启用：%s（频道 %s）", _redis_url(), CHANNEL)
    return _backplane


def set_backplane(backplane: EventBackplane | None) -> None:
    """注入背板实例（测试 / 手工装配用）；传 ``None`` 即彻底关掉（不再回退按环境变量懒加载）。"""
    global _backplane, _resolved
    _backplane = backplane
    _resolved = True


def reset_backplane() -> None:
    """复位为「按环境变量重新决定」（测试收尾用）。"""
    global _backplane, _resolved
    _backplane = None
    _resolved = False


def publish_event(payload: dict[str, Any]) -> None:
    """monitor 的调用点：把一条本进程投递不出去的 AGUI 事件广播给别的进程。未启用时是空操作。"""
    backplane = get_backplane()
    if backplane is not None:
        backplane.publish_nowait(payload)


async def start_forwarding(manager: ConnectionManager) -> EventBackplane | None:
    """API 进程启动时调：订阅背板，把远端事件转发进 ``manager``。未启用返回 ``None``。

    **为什么订阅端装在这里而不是 ConnectionManager 内部**：那个类是一张纯粹的
    ``thread_id → WebSocket`` 路由表，不认识 Redis 也不该认识——它同时被离线脚本、单测和 worker
    进程用着。把「谁来喂它事件」留在装配层（lifespan），路由表本身就还是可以零依赖地单独测。
    """
    backplane = get_backplane()
    if backplane is None:
        return None
    await backplane.start(manager)
    return backplane
