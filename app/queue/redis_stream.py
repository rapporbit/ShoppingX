"""Redis Stream 队列实现：双流分级 + 消费者组 + ack + pending 重投 + 死信。

**为什么是 Stream 不是 List。** List（``LPUSH`` / ``BRPOP``）一弹出消息就从 Redis 消失了——worker
在跑到一半时崩掉，那条任务无人知晓、无从重投。Stream 有消费者组与 pending 列表（PEL）：消息领走
后仍留在 PEL 里直到 ``XACK``，worker 崩了（租约过期）就由别的 worker ``XCLAIM`` 领回来重跑。削峰队列
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

from app.queue.ports import IntentTask, TaskHandler, TaskStatus, cancel_in_flight
from app.utils.env import env_float, env_int

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
# **「消息投递」与「任务存活」拆成两样东西**：PEL 管投递（谁领了、投了几次），独立的租约键
# ``globex:lease:<stream>:<message_id>`` 管存活（持有者还活着吗）。领到消息就 ``SET`` 租约，
# 心跳每 ``_HEARTBEAT_SEC`` 秒用 Lua「值是自己才 PEXPIRE」续期——写法同分布式锁续期（Redisson
# 看门狗默认也是 30s 租约 / 10s 续一次）。接管方只认「租约没了」，不认 idle：在跑的消息 idle
# 会一直涨，那不代表持有者死了。
#
# 早年没有心跳时接管判据是纯 idle（600s，worker 崩了要等十分钟）；上一版用 ``XCLAIM JUSTID``
# 把 idle 清零来续租，但「查属主 → XCLAIM」是两条命令，中间被接管的话心跳会把消息抢回来。
_HEARTBEAT_SEC = env_float("QUEUE_HEARTBEAT_SEC", 10.0)
# 租约时长按流分设：长续聊那条流的单步（一次 LLM 外呼）更长、更容易把事件循环卡住一拍，可以单独
# 放宽容忍度。它**不需要**长于任务耗时——任务跑多久都靠心跳续着。
_LEASE_MS = {
    STREAM_NORMAL: env_int("QUEUE_LEASE_MS", 30_000),
    STREAM_LARGE: env_int("QUEUE_LEASE_MS_LARGE", 30_000),
}
# idle 只剩一个作用：盖住「XREADGROUP 领到 → SET 租约」之间那几毫秒（此刻 PEL 有它、租约还没有）。
# 所以只需远大于这段窗口，不再和任务耗时挂钩。
_CLAIM_IDLE_MS = env_int("QUEUE_CLAIM_IDLE_MS", 10_000)
# 接管时一次 XPENDING 扫多少条。在跑的长任务 idle 都超阈值、都会被扫到再因租约在而跳过，扫描窗口
# 必须盖住「全部 worker 的在途总数」，否则排在后面的真孤儿永远轮不到。
_RECLAIM_SCAN = env_int("QUEUE_RECLAIM_SCAN", 200)
_LEASE_PREFIX = "globex:lease:"

# 三段 Lua 都是「先比值再动手」，一条命令原子执行——拆成 GET + PEXPIRE 两步，中间换了持有者
# 就会给别人续期 / 删掉别人的租约。
# 领取：没人持有就占上；已是自己的就续期（接管路径上接管方已经 SET NX 过一次）。
LEASE_ACQUIRE = """
local v = redis.call('GET', KEYS[1])
if not v then
  redis.call('SET', KEYS[1], ARGV[1], 'PX', ARGV[2])
  return 1
elseif v == ARGV[1] then
  redis.call('PEXPIRE', KEYS[1], ARGV[2])
  return 1
end
return 0
"""
# 续期：只认「值是自己」。租约已过期就返回 0，**不重新占**——过期后别人可能正在接管。
LEASE_RENEW = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
  return redis.call('PEXPIRE', KEYS[1], ARGV[2])
end
return 0
"""
LEASE_RELEASE = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
  return redis.call('DEL', KEYS[1])
end
return 0
"""
# 背压闸：「各流未投递数 + 已准入未入队数」判定与登记在一条脚本里做完（理由见 TaskQueue.admit）。
# 登记用 ZSET（score=登记时刻），超过 ARGV[3] 毫秒的视为持有者已崩、先清掉再算——写法同带超时的
# 计数信号量，API 进程被 kill 在「准入 → 入队」之间也不会把名额永久占住。时刻取 Redis 的 TIME，
# 多个 API 进程之间不必对钟。lag 取不到时退回 pending，口径同 _stream_depth。
# KEYS = [登记 ZSET, 流...]；ARGV = [消费组, ticket, 过期毫秒, 上限]。
ADMIT = """
local t = redis.call('TIME')
local now = t[1] * 1000 + math.floor(t[2] / 1000)
redis.call('ZREMRANGEBYSCORE', KEYS[1], '-inf', now - tonumber(ARGV[3]))
local total = redis.call('ZCARD', KEYS[1])
for i = 2, #KEYS do
  local ok, groups = pcall(redis.call, 'XINFO', 'GROUPS', KEYS[i])
  if ok then
    for _, g in ipairs(groups) do
      local f = {}
      for j = 1, #g, 2 do f[g[j]] = g[j + 1] end
      if f['name'] == ARGV[1] then
        total = total + (tonumber(f['lag']) or tonumber(f['pending']) or 0)
      end
    end
  end
end
if total >= tonumber(ARGV[4]) then
  return -1
end
redis.call('ZADD', KEYS[1], now, ARGV[2])
redis.call('PEXPIRE', KEYS[1], tonumber(ARGV[3]) * 2)
return total
"""
_ADMIT_KEY = "globex:admitted"
_ADMIT_TTL_MS = env_int("QUEUE_ADMIT_TTL_MS", 30_000)
# 已有这几种终态的任务再被投递（worker 写完终态、XACK 之前被 kill），直接 ack 跳过，不重跑。
# 不含 ``failed``：失败那条路本来就是「留 PEL 等重投」，写了 failed 再重跑正是设计意图。
_SKIP_ON_REDELIVERY = frozenset({"done", "cancelled", "interrupted"})
_BLOCK_MS = env_int("QUEUE_BLOCK_MS", 2000)
_MAX_DELIVERIES = env_int("QUEUE_MAX_DELIVERIES", 3)


def _lease_key(stream: str, message_id: str) -> str:
    # 按 message_id 而非 task_id：接管方从 XPENDING 只拿得到 message_id，按 task_id 键的话每条候选
    # 都得多一次 XRANGE 读 payload。
    return f"{_LEASE_PREFIX}{stream}:{message_id}"


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

    async def admit(self, ticket: str, limit: int) -> int | None:
        total = await self._client.eval(
            ADMIT, 1 + len(STREAMS), _ADMIT_KEY, *STREAMS, self._group, ticket, _ADMIT_TTL_MS, limit
        )
        return None if int(total) < 0 else int(total)

    async def release_admission(self, ticket: str) -> None:
        await self._client.zrem(_ADMIT_KEY, ticket)

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
        heartbeat_sec: float = _HEARTBEAT_SEC,
        lease_ms: int | None = None,
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

        **被取消时必须显式掐掉在途任务**：它们是 ``create_task`` 出来的独立 task，取消本协程并不会
        连带取消它们（只有 ``gather`` 的取消才会往下传）。不补这一手，worker 优雅退出超时那条路上
        会留下一批孤儿协程——进程都在退出了，它们还在跑 LLM，而消息既没 ack 也没人管。
        """
        await self.ensure_group()
        leases = dict.fromkeys(STREAMS, lease_ms) if lease_ms else dict(_LEASE_MS)
        if heartbeat_sec > 0 and min(leases.values()) < heartbeat_sec * 3000:
            # 租约不到 3 次心跳：一次网络抖动没续上，在跑的任务就会被别的 worker 接管双跑。
            logger.warning("租约 %s ms 小于 3 倍心跳（%.1fs），存在误抢风险", leases, heartbeat_sec)
        limit = max(1, concurrency)
        sem = asyncio.Semaphore(limit)
        in_flight: set[asyncio.Task[None]] = set()
        try:
            while not should_stop():
                in_flight = {t for t in in_flight if not t.done()}
                free = limit - len(in_flight)
                if free <= 0:
                    await asyncio.wait(in_flight, return_when=asyncio.FIRST_COMPLETED)
                    continue
                entries = await self._read(consumer, free, block_ms)
                if not entries:
                    entries = await self._reclaim(consumer, claim_idle_ms, free, leases)
                for stream, message_id, fields in entries:
                    in_flight.add(
                        asyncio.create_task(
                            self._handle_one(
                                stream,
                                message_id,
                                fields,
                                handler,
                                max_deliveries,
                                sem,
                                consumer=consumer,
                                heartbeat_sec=heartbeat_sec,
                                lease_ms=leases[stream],
                            )
                        )
                    )
            if in_flight:
                logger.info("停止领新任务，等 %d 个在途任务跑完", len(in_flight))
                await asyncio.gather(*in_flight, return_exceptions=True)
        except asyncio.CancelledError:
            await cancel_in_flight(in_flight)
            raise

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
        self, consumer: str, idle_ms: int, count: int, leases: dict[str, int]
    ) -> list[tuple[str, str, dict[Any, Any]]]:
        """领回持有者已死的 pending 消息。两条流都要扫。

        判死只认租约：``XPENDING IDLE`` 先筛出「领走有一阵了」的候选（在跑的长任务也在里面），
        再逐条 ``SET NX`` 抢租约——抢得到说明原持有者没在续期，抢不到就是还活着、跳过。
        **不能用 XAUTOCLAIM**：它只看 idle，会把续着租约的在跑任务直接领走。

        抢租约与 ``XCLAIM`` 的先后不能反：两个接管方同时看到租约没了，只有 ``SET NX`` 成功的那个
        往下走；``XCLAIM`` 带 ``min_idle_time`` 再挡一次（第一个领走后 idle 清零，第二个领不到）。
        ``XCLAIM`` 不带 JUSTID，投递计数照常 +1，死信判据不变。
        """
        out: list[tuple[str, str, dict[Any, Any]]] = []
        for stream in STREAMS:
            if len(out) >= count:
                break
            try:
                rows = await self._client.xpending_range(
                    stream, self._group, min="-", max="+", count=_RECLAIM_SCAN, idle=idle_ms
                )
            except Exception as exc:
                logger.debug("XPENDING 跳过（%s）：%s", stream, exc)
                continue
            for row in rows or ():
                if len(out) >= count:
                    break
                message_id = _s(row["message_id"])
                key = _lease_key(stream, message_id)
                try:
                    won = await self._client.set(key, consumer, px=leases[stream], nx=True)
                except Exception as exc:
                    logger.debug("抢租约失败（%s）：%s", message_id, exc)
                    continue
                if not won:
                    continue  # 持有者还在续期
                try:
                    claimed = await self._client.xclaim(
                        stream, self._group, consumer, idle_ms, [message_id]
                    )
                except Exception as exc:
                    logger.debug("XCLAIM 失败（%s）：%s", message_id, exc)
                    claimed = []
                if not claimed:
                    # 别人抢先领走 / 已被 ack / 原消息已裁剪（Redis 7 会顺手删掉 PEL 项）。
                    await self._release(key, consumer)
                    continue
                _mid, fields = claimed[0]
                if fields:
                    out.append((stream, message_id, fields))
                else:
                    # 原消息已被裁剪、只剩 PEL 里的空壳：ack 掉，否则它每次都被捡起来一遍。
                    await self._ack(stream, message_id)
                    await self._release(key, consumer)
        if out:
            logger.info("接管 %d 条租约已过期的任务（consumer=%s）", len(out), consumer)
        return out

    async def _handle_one(
        self,
        stream: str,
        message_id: str,
        fields: dict[Any, Any],
        handler: TaskHandler,
        max_deliveries: int,
        sem: asyncio.Semaphore,
        *,
        consumer: str = "",
        heartbeat_sec: float = 0.0,
        lease_ms: int = 0,
    ) -> None:
        # 关心跳 = 占了租约不续期，跑得比租约久的任务会被接管（chaos 脚本的对照组靠这个）。
        key = _lease_key(stream, message_id) if consumer else ""
        lease_ms = lease_ms or _LEASE_MS.get(stream, _LEASE_MS[STREAM_NORMAL])
        try:
            await self._process(
                stream,
                message_id,
                fields,
                handler,
                max_deliveries,
                sem,
                key,
                consumer,
                heartbeat_sec,
                lease_ms,
            )
        finally:
            # 成功、失败、被取消都释放：失败的那条要尽快被重投，不该干等租约自然过期。
            # 早退路径（空消息 / 解析失败 / 已有终态）上本 worker 可能根本没占租约，比值删是空操作。
            if key:
                await self._release(key, consumer)

    async def _process(
        self,
        stream: str,
        message_id: str,
        fields: dict[Any, Any],
        handler: TaskHandler,
        max_deliveries: int,
        sem: asyncio.Semaphore,
        key: str,
        consumer: str,
        heartbeat_sec: float,
        lease_ms: int,
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
        if await self._already_finished(task.task_id):
            # at-least-once 的另一面：worker 写完终态、XACK 之前被 kill，这条会被接管方再领一次。
            # 按 task_id 查终态去重，别把一轮已经给了用户答复的对话重跑一遍。
            logger.info("任务已有终态，重投跳过：%s", task.task_id)
            await self._ack(stream, message_id)
            return
        beat = None
        if key:
            # 租约从领到手就占上，不是从拿到信号量才占：在信号量上排队的消息同样挂在 PEL 里、idle
            # 同样在涨，没租约就会被别的 worker 当成孤儿领走。
            if not await self._acquire(key, consumer, lease_ms):
                # 领到手到占租约这几毫秒里被别人接管了（idle 阈值设得过小才可能）：让给对方。
                logger.warning("租约已被他人持有，放弃本次投递：%s", message_id)
                return
            if heartbeat_sec > 0:
                beat = asyncio.create_task(self._heartbeat(key, consumer, heartbeat_sec, lease_ms))
        try:
            async with sem:
                try:
                    await handler(task)
                except asyncio.CancelledError:
                    # 取消不算失败：不 ack、不进死信，留在 PEL 里等下一个 worker 捡。优雅退出的
                    # 正常路径走不到这儿——handler（worker.handle_task）自己按 interrupted 收尾后
                    # 正常返回，由下面那行 ack 掉。走到这儿的是收尾也没兜住的意外取消，留 PEL
                    # 是兜底。
                    raise
                except Exception as exc:
                    await self._on_failure(stream, message_id, text, task, exc, max_deliveries)
                    return
                await self._ack(stream, message_id)
        finally:
            if beat is not None:
                beat.cancel()

    async def _already_finished(self, task_id: str) -> bool:
        try:
            status = await self.get_status(task_id)
        except Exception:
            return False  # 查不到就当没跑过：宁可重跑一次（写工具另有 operation_id 幂等），不能丢
        return status is not None and status.state in _SKIP_ON_REDELIVERY

    # ── 租约 ─────────────────────────────────────────────────────────────────
    async def _acquire(self, key: str, owner: str, lease_ms: int) -> bool:
        try:
            return bool(await self._client.eval(LEASE_ACQUIRE, 1, key, owner, lease_ms))
        except Exception as exc:
            # Redis 抖了占不上：照跑。最坏是被别人接管双跑，由终态去重与写工具幂等兜住；
            # 反过来因为占不上就不跑，这条消息会一直卡在 PEL 里等 idle。
            logger.warning("占租约失败，照常执行：%s（%s）", key, exc)
            return True

    async def _release(self, key: str, owner: str) -> None:
        try:
            await self._client.eval(LEASE_RELEASE, 1, key, owner)
        except Exception as exc:
            # 没删掉只是让重投多等一个租约周期，不影响正确性。
            logger.debug("释放租约失败：%s（%s）", key, exc)

    async def _heartbeat(self, key: str, owner: str, interval: float, lease_ms: int) -> None:
        """续期：每 ``interval`` 秒跑一次 :data:`LEASE_RENEW`（值是自己才 PEXPIRE）。

        返回 0 = 租约已不是自己的（过期了，或已被接管方 SET NX 占走），停止续期。**不去 cancel
        handler**：handler 被取消会按 interrupted 写终态，可能盖掉接管方正在写的状态；本地这份
        跑完照常 ack（XACK 不看属主），双跑由终态去重与写工具幂等兜住。
        """
        while True:
            await asyncio.sleep(interval)
            try:
                renewed = await self._client.eval(LEASE_RENEW, 1, key, owner, lease_ms)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                # 一次没续上不致命：租约是心跳的 3 倍，留足了重试余量。
                logger.debug("续租失败（%s）：%s", key, exc)
                continue
            if not renewed:
                logger.warning("租约已丢失，停止续期（任务仍会跑完）：%s", key)
                return

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
        # 不 ack —— 留在 PEL 里；租约在 _handle_one 收尾时释放，空闲 worker 扫到就领回重投。
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
