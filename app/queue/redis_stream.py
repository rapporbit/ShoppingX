"""Redis Stream 队列实现：双流分级 + 消费者组 + ack + pending 重投 + 死信。

**为什么是 Stream 不是 List。** List（``LPUSH`` / ``BRPOP``）一弹出消息就从 Redis 消失了——worker
在跑到一半时崩掉，那条任务无人知晓、无从重投。Stream 有消费者组与 pending 列表（PEL）：消息领走
后仍留在 PEL 里直到 ``XACK``，worker 崩了就由 ``XAUTOCLAIM`` 让别的 worker 领回来重跑。削峰队列
的任务动辄跑几十秒到几分钟，「跑一半进程没了」不是罕见情况而是每次部署都会发生的常态。

**双流分级。** ``globex:intents``（normal）与 ``globex:intents:large``（heavy）用**同一个消费者
组名**，``XREADGROUP`` 时 normal 排在前面——Redis 按传入顺序返回，短对话天然优先于长续聊。分流阈值
走 :func:`app.api.concurrency.classify_request`（见 ports 模块的说明，两处必须共用一份）。

**死信。** 同一条消息重投 ``max_deliveries`` 次仍失败就进 ``globex:intents:dead`` 并 ack 掉。不设
死信的后果不是「多试几次」而是队头阻塞：一条必然失败的消息（比如序列化不出来的旧版本 payload）
会被永远重投，把 worker 的并发度一点点吃干净。

**降级口径与 event_log 不同，这是有意的。** 事件回放丢了只是少看几条事件，故它 Redis 一挂就静默
降级；队列丢了就是任务凭空消失，故这里的 :meth:`enqueue` **失败就抛**，由调用方决定回落到进程内
执行还是回 5xx。只有 :meth:`depth` 这种纯观测路径才吞异常。
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable
from dataclasses import replace
from typing import Any

from app.queue.ports import IntentTask, TaskHandler, TaskStatus
from app.utils.env import env_int

logger = logging.getLogger("shoppingx.queue")

STREAM_NORMAL = "globex:intents"
STREAM_LARGE = "globex:intents:large"
STREAM_DEAD = "globex:intents:dead"
# 读取顺序即优先级：dict 保序，redis-py 按序拼进 XREADGROUP 的 STREAMS 参数。
STREAMS = (STREAM_NORMAL, STREAM_LARGE)
GROUP = "globex-workers"
_STATUS_PREFIX = "globex:task:"

# 状态键存活时长：够前端把一次异步提交轮询完即可，不是持久存储（真·历史在 turns.json / 关系库）。
_STATUS_TTL = env_int("QUEUE_STATUS_TTL", 3600)
# 死信流保留条数。死信是给人看的（排查为什么这条跑不动），不是给程序重放的，留最近若干条就够；
# 不设上限则一次线上事故能把 Redis 内存吃光。
_DEAD_MAXLEN = env_int("QUEUE_DEAD_MAXLEN", 1000)
# 一条消息在 PEL 里闲置多久算「上一个 worker 大概是挂了」。要明显长于单轮任务耗时（本仓约 40s，
# 长续聊几分钟），否则会把还在正常跑的任务抢过来重跑一遍——那不是容错是双跑。
_CLAIM_IDLE_MS = env_int("QUEUE_CLAIM_IDLE_MS", 600_000)
_BLOCK_MS = env_int("QUEUE_BLOCK_MS", 2000)
_MAX_DELIVERIES = env_int("QUEUE_MAX_DELIVERIES", 3)


def _s(value: Any) -> str:
    """Redis 客户端可能返回 bytes（``decode_responses=False`` 时），统一成 str。"""
    return value.decode() if isinstance(value, bytes) else str(value)


class RedisStreamQueue:
    """:class:`app.queue.ports.TaskQueue` 的 Redis Stream 实现。"""

    def __init__(self, client: Any, *, group: str = GROUP) -> None:
        self._client = client
        self._group = group

    # ── 生产侧 ───────────────────────────────────────────────────────────────
    async def ensure_group(self) -> None:
        """幂等建组（两条流各一个）。组已存在时 Redis 抛 BUSYGROUP，那是正常路径不是错误。

        ``id="0"`` 而非 ``"$"``：从流头开始消费。用 ``$`` 的话建组之前已入队的消息永远不会被投递，
        而「先起 API 入了几条、再起 worker」在部署顺序上完全正常。``mkstream=True`` 让流不存在时
        顺带建出来，省掉「必须先 XADD 一条才能建组」的鸡生蛋。
        """
        for stream in STREAMS:
            try:
                await self._client.xgroup_create(stream, self._group, id="0", mkstream=True)
            except Exception as exc:
                if "BUSYGROUP" not in str(exc):
                    raise

    async def enqueue(self, task: IntentTask) -> None:
        stream = STREAM_LARGE if task.kind == "heavy" else STREAM_NORMAL
        payload = json.dumps(task.to_dict(), ensure_ascii=False)
        await self._client.xadd(stream, {"payload": payload})
        logger.info("入队 %s（%s，thread=%s）", task.task_id, stream, task.thread_id)

    async def set_status(self, status: TaskStatus) -> None:
        await self._client.set(
            f"{_STATUS_PREFIX}{status.task_id}",
            json.dumps(status.to_dict(), ensure_ascii=False),
            ex=_STATUS_TTL,
        )

    async def get_status(self, task_id: str) -> TaskStatus | None:
        raw = await self._client.get(f"{_STATUS_PREFIX}{task_id}")
        if raw is None:
            return None
        try:
            status = TaskStatus.from_dict(json.loads(_s(raw)))
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.warning("任务状态解析失败：%s（%s）", task_id, exc)
            return None
        if status.state != "queued":
            return status
        # 排队中才附上队列深度：跑起来之后这个数对用户没有意义，白花一次 XINFO。
        return replace(status, queue_depth=await self.depth())

    async def depth(self) -> int:
        total = 0
        for stream in STREAMS:
            total += await self._stream_depth(stream)
        return total

    async def _stream_depth(self, stream: str) -> int:
        """单流的未投递数（lag）。流还没建 / Redis 抖了都返回 0——观测不该拖垮主链路。"""
        try:
            groups = await self._client.xinfo_groups(stream)
        except Exception:
            return 0
        for group in groups or ():
            if _s(group.get("name")) != self._group:
                continue
            lag = group.get("lag")
            # lag 是 Redis 7.0 才有的字段；老版本退回 pending（已领未 ack 数）——它不等于待领数，
            # 但同样是「还没干完的活」的量级，比返回 0 骗人强。
            return int(lag) if lag is not None else int(group.get("pending", 0) or 0)
        return 0

    async def close(self) -> None:
        aclose = getattr(self._client, "aclose", None)
        if aclose is not None:
            await aclose()

    # ── 消费侧 ───────────────────────────────────────────────────────────────
    async def consume(
        self,
        consumer: str,
        handler: TaskHandler,
        should_stop: Callable[[], bool],
        concurrency: int = 1,
        *,
        block_ms: int = _BLOCK_MS,
        max_deliveries: int = _MAX_DELIVERIES,
        claim_idle_ms: int = _CLAIM_IDLE_MS,
    ) -> None:
        """消费循环。跑到 ``should_stop()`` 为真、且在途任务收干净才返回。

        **并发上限用信号量兜而不是只靠「少读几条」**：``XREADGROUP`` 的 COUNT 是**每条流**的上限，
        两条流一次最多能返回 2×count 条。少读那版看着对、峰值却会跑出双倍并发（本仓一个 loop 就
        是一串 LLM 外呼，超卖直接把下游打爆）。读到手的都不退回（退回 = 留在 PEL 里等重投，白白
        多一次投递计数），排不上号的就在信号量上等。

        **优先级的口径是「同一批里 normal 排前面」**，不是「normal 清空前不碰 large」：一次读两条流
        各取 COUNT 条，长续聊仍会被穿插着消费。这与准入池 ``HEAVY_SLOTS_MIN`` 不设 0 是同一个判断
        ——饿死长任务不是背压，是拒绝服务。

        **空转时才去捡 pending**：有活干的时候不该分神，闲下来正好扫一遍上一个 worker 留下的烂摊子。
        """
        await self.ensure_group()
        limit = max(1, concurrency)
        sem = asyncio.Semaphore(limit)
        in_flight: set[asyncio.Task[None]] = set()
        while not should_stop():
            in_flight = {t for t in in_flight if not t.done()}
            free = limit - len(in_flight)
            if free <= 0:
                await asyncio.wait(in_flight, return_when=asyncio.FIRST_COMPLETED)
                continue
            entries = await self._read(consumer, free, block_ms)
            if not entries:
                entries = await self._reclaim(consumer, claim_idle_ms, free)
            for stream, message_id, fields in entries:
                in_flight.add(
                    asyncio.create_task(
                        self._handle_one(stream, message_id, fields, handler, max_deliveries, sem)
                    )
                )
        if in_flight:
            logger.info("停止领新任务，等 %d 个在途任务跑完", len(in_flight))
            await asyncio.gather(*in_flight, return_exceptions=True)

    async def _read(
        self, consumer: str, count: int, block_ms: int
    ) -> list[tuple[str, str, dict[Any, Any]]]:
        try:
            batches = await self._client.xreadgroup(
                self._group,
                consumer,
                dict.fromkeys(STREAMS, ">"),
                count=count,
                block=block_ms,
            )
        except Exception as exc:
            # 连不上 / 组被人删了都走这里。退避 1s 再试，别把日志和 CPU 一起打满。
            logger.warning("队列读取失败，1s 后重试：%s", exc)
            await asyncio.sleep(1)
            return []
        out: list[tuple[str, str, dict[Any, Any]]] = []
        for stream, entries in batches or ():
            # ack 必须回到消息**所属的那条流**，不能写死 normal，否则 large 流的消息永远 ack 不掉。
            name = _s(stream)
            for message_id, fields in entries:
                out.append((name, _s(message_id), fields or {}))
        return out

    async def _reclaim(
        self, consumer: str, idle_ms: int, count: int
    ) -> list[tuple[str, str, dict[Any, Any]]]:
        """把闲置超时的 pending 消息领回本 consumer（上一个 worker 崩了的情况）。两条流都要扫。"""
        out: list[tuple[str, str, dict[Any, Any]]] = []
        for stream in STREAMS:
            try:
                result = await self._client.xautoclaim(
                    stream, self._group, consumer, min_idle_time=idle_ms, count=count
                )
            except Exception as exc:
                logger.debug("XAUTOCLAIM 跳过（%s）：%s", stream, exc)
                continue
            entries = result[1] if result and len(result) > 1 else []
            for message_id, fields in entries or ():
                if fields:
                    out.append((stream, _s(message_id), fields))
                else:
                    # 原消息已被裁剪掉、只剩 PEL 里的空壳：ack 掉，否则它每次都被捡起来一遍。
                    await self._ack(stream, _s(message_id))
        if out:
            logger.info("重投 %d 条超时未 ack 的任务（consumer=%s）", len(out), consumer)
        return out

    async def _handle_one(
        self,
        stream: str,
        message_id: str,
        fields: dict[Any, Any],
        handler: TaskHandler,
        max_deliveries: int,
        sem: asyncio.Semaphore,
    ) -> None:
        raw = fields.get("payload") or fields.get(b"payload")
        if not raw:
            await self._ack(stream, message_id)  # 空消息没什么可重投的，ack 掉别占 PEL
            return
        text = _s(raw)
        try:
            task = IntentTask.from_dict(json.loads(text))
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            # 解不开的消息重投一万次也还是解不开，直接进死信，不能让它卡住队列。
            await self._to_dead(stream, message_id, text, f"payload 解析失败：{exc}")
            return
        async with sem:
            try:
                await handler(task)
            except asyncio.CancelledError:
                # 优雅退出时的取消不算失败：不 ack、不进死信，留在 PEL 里等下一个 worker 捡。
                raise
            except Exception as exc:
                await self._on_failure(stream, message_id, text, task, exc, max_deliveries)
                return
            await self._ack(stream, message_id)

    async def _on_failure(
        self,
        stream: str,
        message_id: str,
        text: str,
        task: IntentTask,
        exc: Exception,
        max_deliveries: int,
    ) -> None:
        deliveries = await self._delivery_count(stream, message_id)
        if deliveries >= max_deliveries:
            logger.error("任务重投超限，进死信：%s（%s）", task.task_id, exc)
            await self._to_dead(stream, message_id, text, f"重投 {deliveries} 次仍失败：{exc}")
            return
        # 不 ack —— 留在 PEL 里，由 XAUTOCLAIM 重投。
        logger.warning("任务失败（第 %d 次投递）：%s（%s）", deliveries, task.task_id, exc)

    async def _delivery_count(self, stream: str, message_id: str) -> int:
        try:
            pending = await self._client.xpending_range(
                stream, self._group, min=message_id, max=message_id, count=1
            )
            return int(pending[0]["times_delivered"]) if pending else 1
        except Exception:
            return 1  # 数不出来就当第一次，宁可多重投一次也别提前扔进死信

    async def _to_dead(self, stream: str, message_id: str, text: str, reason: str) -> None:
        try:
            await self._client.xadd(
                STREAM_DEAD,
                {"payload": text, "reason": reason, "origin": stream},
                maxlen=_DEAD_MAXLEN,
                approximate=True,
            )
        except Exception as exc:
            logger.error("写死信失败（消息仍会被 ack）：%s", exc)
        await self._ack(stream, message_id)

    async def _ack(self, stream: str, message_id: str) -> None:
        try:
            await self._client.xack(stream, self._group, message_id)
        except Exception as exc:
            # ack 失败的后果是「这条任务将来会被重投一次」，比在这里抛掉整个消费循环轻。
            logger.warning("XACK 失败（%s/%s）：%s", stream, message_id, exc)
