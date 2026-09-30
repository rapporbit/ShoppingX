"""削峰队列的确定性单测：双流分级 / ack / pending 重投 / 死信 / 进程内回落。

**测试策略沿用 `tests/test_event_replay.py` 的惯例**：一个内存 FakeRedis 实现 Stream 消费者组的最小
子集（xadd / xreadgroup / xack / xpending_range / xclaim / 租约 Lua 等），不依赖真 Redis，也不引
fakeredis 包。真 Redis 的价值在于验协议细节，而这里要钉的是**我们自己的取舍**——normal 先于 large、
ack 回原流、失败留 PEL、超限进死信——这些用假客户端反而断言得更死（能把「投递第几次」直接摆出来）。

FakeRedis 刻意实现了 ``times_delivered`` 与 ``min_idle_time``：死信与重投的判据全压在这两个数上，
把它们写成常量假值的话，这组用例会在真实行为退化时照样绿。
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import replace
from typing import Any

import pytest
import structlog
from prometheus_client import REGISTRY

from app import queue as queue_pkg
from app import worker
from app.queue import InProcessQueue, RedisStreamQueue, get_task_queue, set_task_queue
from app.queue.ports import IntentTask, TaskQueue, TaskStatus
from app.queue.redis_stream import (
    GROUP,
    LEASE_ACQUIRE,
    LEASE_RELEASE,
    LEASE_RENEW,
    STREAM_DEAD,
    STREAM_LARGE,
    STREAM_NORMAL,
    _lease_key,
)


class FakeRedis:
    """Redis Stream 消费者组的最小内存实现（含 PEL、投递计数、idle 时间）。"""

    def __init__(self) -> None:
        self.streams: dict[str, list[tuple[str, dict[str, Any]]]] = {}
        self.groups: dict[tuple[str, str], dict[str, Any]] = {}
        self.kv: dict[str, str] = {}
        self.expiry: dict[str, float] = {}
        self.eval_calls: dict[str, int] = {}
        self.counter = 0
        self.xclaim_calls = 0

    # ── 生产 ────────────────────────────────────────────────────────────────
    async def xadd(
        self,
        stream: str,
        fields: dict[str, Any],
        maxlen: int | None = None,
        approximate: bool = True,
    ) -> str:
        self.counter += 1
        sid = f"{self.counter}-0"
        self.streams.setdefault(stream, []).append((sid, dict(fields)))
        if maxlen:
            self.streams[stream] = self.streams[stream][-maxlen:]
        return sid

    async def xgroup_create(self, stream: str, group: str, id: str = "0", mkstream: bool = False):
        if (stream, group) in self.groups:
            raise RuntimeError("BUSYGROUP Consumer Group name already exists")
        self.streams.setdefault(stream, [])
        self.groups[(stream, group)] = {"cursor": 0, "pending": {}}
        return True

    # ── 消费 ────────────────────────────────────────────────────────────────
    async def xreadgroup(
        self,
        group: str,
        consumer: str,
        streams: dict[str, str],
        count: int = 1,
        block: int = 0,
    ) -> list[tuple[str, list[tuple[str, dict[str, Any]]]]]:
        out = []
        for stream in streams:  # 保序：调用方靠这个顺序表达优先级
            state = self.groups.get((stream, group))
            if state is None:
                continue
            entries = self.streams.get(stream, [])[state["cursor"] : state["cursor"] + count]
            if not entries:
                continue
            state["cursor"] += len(entries)
            for sid, _fields in entries:
                state["pending"][sid] = {
                    "consumer": consumer,
                    "times_delivered": 1,
                    "delivered_at": time.monotonic(),
                }
            out.append((stream, entries))
        if not out:
            await asyncio.sleep(0)  # 模拟 block：把控制权让出去，别把事件循环饿死
        return out

    async def xack(self, stream: str, group: str, message_id: str) -> int:
        state = self.groups.get((stream, group))
        if state is None:
            return 0
        return 1 if state["pending"].pop(message_id, None) is not None else 0

    async def xpending_range(
        self,
        stream: str,
        group: str,
        min: str,
        max: str,
        count: int = 10,
        idle: int | None = None,
    ) -> list[dict[str, Any]]:
        """单条查询（min == max）与 ``- +`` 全扫两种形态；``idle`` 同真 Redis 按闲置毫秒过滤。"""
        state = self.groups.get((stream, group), {"pending": {}})
        now = time.monotonic()
        ids = list(state["pending"]) if min == "-" else [min]
        out = []
        for sid in ids:
            entry = state["pending"].get(sid)
            if entry is None:
                continue
            if idle is not None and (now - entry["delivered_at"]) * 1000 < idle:
                continue
            out.append(
                {
                    "message_id": sid,
                    "consumer": entry["consumer"],
                    "times_delivered": entry["times_delivered"],
                }
            )
        return out[:count]

    async def xclaim(
        self,
        stream: str,
        group: str,
        consumer: str,
        min_idle_time: int,
        message_ids: list[str],
    ) -> list[tuple[str, dict[str, Any]]]:
        """换属主 + idle 清零 + 投递计数 +1（不带 JUSTID 的真 Redis 行为）；未到 idle 的不领。"""
        state = self.groups.get((stream, group), {"pending": {}})
        body = dict(self.streams.get(stream, []))
        now = time.monotonic()
        out = []
        for sid in message_ids:
            entry = state["pending"].get(sid)
            if entry is None or (now - entry["delivered_at"]) * 1000 < min_idle_time:
                continue
            entry["consumer"] = consumer
            entry["times_delivered"] += 1
            entry["delivered_at"] = now
            self.xclaim_calls += 1
            out.append((sid, body.get(sid, {})))
        return out

    async def xinfo_groups(self, stream: str) -> list[dict[str, Any]]:
        if stream not in self.streams:
            raise RuntimeError("ERR no such key")
        out = []
        for (s, g), state in self.groups.items():
            if s != stream:
                continue
            out.append(
                {
                    "name": g,
                    "lag": max(0, len(self.streams.get(stream, [])) - state["cursor"]),
                    "pending": len(state["pending"]),
                }
            )
        return out

    # ── 状态键 / 租约键 ─────────────────────────────────────────────────────
    def _live(self, key: str) -> str | None:
        exp = self.expiry.get(key)
        if exp is not None and time.monotonic() >= exp:
            self.kv.pop(key, None)
            self.expiry.pop(key, None)
        return self.kv.get(key)

    async def set(
        self,
        key: str,
        value: str,
        ex: int | None = None,
        px: int | None = None,
        nx: bool = False,
    ) -> bool | None:
        if nx and self._live(key) is not None:
            return None  # 同真 Redis：NX 未写入回 None
        self.kv[key] = value
        self.expiry.pop(key, None)
        if px is not None:
            self.expiry[key] = time.monotonic() + px / 1000
        return True

    async def get(self, key: str) -> str | None:
        return self._live(key)

    async def eval(self, script: str, numkeys: int, *args: Any) -> int:
        """只认 redis_stream 里那三段租约 Lua，按它们的语义在内存里执行（单线程即原子）。"""
        key, owner = args[0], args[1]
        cur = self._live(key)
        if script == LEASE_RELEASE:
            if cur == owner:
                self.kv.pop(key, None)
                self.expiry.pop(key, None)
                return 1
            return 0
        self.eval_calls[script] = self.eval_calls.get(script, 0) + 1
        ttl = int(args[2]) / 1000
        if script == LEASE_RENEW:
            if cur != owner:
                return 0
        elif script == LEASE_ACQUIRE:
            if cur is not None and cur != owner:
                return 0
            self.kv[key] = owner
        else:
            raise AssertionError("FakeRedis 不认识的 Lua 脚本")
        self.expiry[key] = time.monotonic() + ttl
        return 1

    def expire_now(self, key: str) -> None:
        """测试辅助：让某个键立刻过期（模拟持有者死掉、不再续期）。"""
        self.expiry[key] = time.monotonic() - 1

    # ── 测试辅助 ────────────────────────────────────────────────────────────
    def pending_ids(self, stream: str, group: str = GROUP) -> list[str]:
        return list(self.groups.get((stream, group), {"pending": {}})["pending"])


@pytest.fixture
def fake() -> FakeRedis:
    return FakeRedis()


@pytest.fixture
def rq(fake: FakeRedis) -> RedisStreamQueue:
    return RedisStreamQueue(fake)


@pytest.fixture(autouse=True)
def _reset_singleton() -> Any:
    """工厂是进程级单例，用例之间必须复位，否则先跑的那个把队列固化给后面所有人。"""
    set_task_queue(None)
    yield
    set_task_queue(None)


def _task(query: str = "买个背包", *, turns: int = 0, tid: str = "t1") -> IntentTask:
    return IntentTask.create(task_id=tid, thread_id=tid, query=query, history_turns=turns)


async def _consume_until(
    queue: Any, should_stop: Any, handler: Any, *, wait_s: float = 3.0, **kw: Any
) -> None:
    # 兜底超时：消费循环写错（如 should_stop 永不为真）时让用例超时失败，而不是把整套 pytest 挂住。
    await asyncio.wait_for(queue.consume("worker-1", handler, should_stop, 2, **kw), wait_s)


async def _drain(queue: Any, expected: int, *, fail: bool = False, **kw: Any) -> list[IntentTask]:
    """跑消费循环直到 handler 被调用 ``expected`` 次；``fail=True`` 让 handler 每次都抛。"""
    handled: list[IntentTask] = []
    done = asyncio.Event()

    async def _handler(task: IntentTask) -> None:
        try:
            if fail:
                raise RuntimeError("handler boom")
        finally:
            handled.append(task)
            if len(handled) >= expected:
                done.set()

    await _consume_until(queue, done.is_set, _handler, **kw)
    return handled


# ── 分流与序列化 ────────────────────────────────────────────────────────────
def test_task_kind_follows_heavy_turns_threshold() -> None:
    from app.api.concurrency import HEAVY_TURNS_THRESHOLD

    assert _task(turns=0).kind == "normal"
    assert _task(turns=HEAVY_TURNS_THRESHOLD).kind == "heavy"


def test_task_dict_roundtrip_and_tolerates_missing_fields() -> None:
    task = IntentTask.create(
        task_id="a",
        thread_id="t",
        query="q",
        history_turns=99,
        platforms=["amazon"],
        dest_country="JP",
    )
    assert IntentTask.from_dict(task.to_dict()) == task
    # 滚动更新期间旧进程写的 payload 缺新字段——只有三个必需键在就该能跑起来。
    lean = IntentTask.from_dict({"task_id": "a", "thread_id": "t", "query": "q"})
    assert (lean.kind, lean.platforms, lean.user_id, lean.dest_country) == ("normal", (), None, "")
    assert lean.request_id == "", "老消息没有 request_id，不该炸也不该编一个"


def test_request_id_survives_the_queue() -> None:
    """request_id 要跨序列化活下来——它是 API 与 worker 两个进程唯一的那根线。

    盯的是 ``to_dict``：字段加在 dataclass 上而忘了加进 payload，本地跑全绿（同进程直接传对象），
    上了队列才静默丢——而丢了不会报错，只会让日志再也串不起来。
    """
    task = IntentTask.create(task_id="a", thread_id="t", query="q", request_id="ab12cd34")
    assert task.to_dict()["request_id"] == "ab12cd34"
    assert IntentTask.from_dict(task.to_dict()).request_id == "ab12cd34"


_TP = "00-" + "ab" * 16 + "-" + "cd" * 8 + "-01"


def test_traceparent_survives_the_queue() -> None:
    """traceparent 同 request_id：加在 dataclass 上忘了进 payload，同进程测全绿、过队列静默丢。"""
    task = replace(_task(), traceparent=_TP)
    assert IntentTask.from_dict(json.loads(json.dumps(task.to_dict()))).traceparent == _TP
    lean = IntentTask.from_dict({"task_id": "a", "thread_id": "t", "query": "q"})
    assert lean.traceparent == "", "老消息没有 traceparent，留空由 worker 自成一条 trace"


async def test_enqueue_routes_by_kind(rq: RedisStreamQueue, fake: FakeRedis) -> None:
    await rq.ensure_group()
    await rq.enqueue(_task(turns=0, tid="short"))
    await rq.enqueue(_task(turns=50, tid="long"))
    assert len(fake.streams[STREAM_NORMAL]) == 1
    assert len(fake.streams[STREAM_LARGE]) == 1
    payload = json.loads(fake.streams[STREAM_LARGE][0][1]["payload"])
    assert payload["task_id"] == "long"


async def test_ensure_group_is_idempotent(rq: RedisStreamQueue) -> None:
    await rq.ensure_group()
    await rq.ensure_group()  # BUSYGROUP 是正常路径，不该抛


# ── 消费：ack / 优先级 / 重投 / 死信 ────────────────────────────────────────
async def test_consume_acks_on_success(rq: RedisStreamQueue, fake: FakeRedis) -> None:
    await rq.ensure_group()
    await rq.enqueue(_task(tid="a"))
    handled = await _drain(rq, 1, block_ms=0)
    assert [t.task_id for t in handled] == ["a"]
    assert fake.pending_ids(STREAM_NORMAL) == []  # 成功即 ack，PEL 清空


async def test_normal_stream_served_before_large(rq: RedisStreamQueue) -> None:
    await rq.ensure_group()
    await rq.enqueue(_task(turns=50, tid="long"))  # 先入 heavy
    await rq.enqueue(_task(turns=0, tid="short"))
    handled = await _drain(rq, 2, block_ms=0)
    assert [t.task_id for t in handled] == ["short", "long"]


async def test_failure_keeps_message_pending(rq: RedisStreamQueue, fake: FakeRedis) -> None:
    await rq.ensure_group()
    await rq.enqueue(_task(tid="a"))
    await _drain(rq, 1, fail=True, block_ms=0, claim_idle_ms=10_000, max_deliveries=3)
    # 失败不 ack：消息留在 PEL 里等重投，且还没到死信。
    assert fake.pending_ids(STREAM_NORMAL) != []
    assert STREAM_DEAD not in fake.streams
    assert _leases(fake) == {}  # 失败即释放租约：重投不必干等它自然过期


async def test_redelivery_then_dead_letter(rq: RedisStreamQueue, fake: FakeRedis) -> None:
    await rq.ensure_group()
    await rq.enqueue(_task(tid="a"))
    # claim_idle_ms=0：空转一圈就把 PEL 里那条捡回来重投，第 3 次仍失败即进死信。
    handled = await _drain(rq, 3, fail=True, block_ms=0, claim_idle_ms=0, max_deliveries=3)
    assert [t.task_id for t in handled] == ["a", "a", "a"]
    assert len(fake.streams[STREAM_DEAD]) == 1
    assert "重投 3 次" in fake.streams[STREAM_DEAD][0][1]["reason"]
    assert fake.pending_ids(STREAM_NORMAL) == []  # 进死信要连带 ack，否则永远被捡起来


async def test_reclaim_covers_large_stream(rq: RedisStreamQueue, fake: FakeRedis) -> None:
    """回归：重投必须两条流都扫。只扫 normal 的话，heavy 任务一旦卡在 PEL 就永远回不来。"""
    await rq.ensure_group()
    await rq.enqueue(_task(turns=50, tid="long"))
    # 模拟前一个 worker 领了 large 流的消息后崩掉（读了不 ack）。
    await fake.xreadgroup(GROUP, "dead-worker", {STREAM_LARGE: ">"}, count=10, block=0)
    assert fake.pending_ids(STREAM_LARGE) != []

    before = _reclaimed(STREAM_LARGE)
    handled = await _drain(rq, 1, block_ms=0, claim_idle_ms=0)
    assert [t.task_id for t in handled] == ["long"]
    assert fake.pending_ids(STREAM_LARGE) == []
    # 接管次数按流记：这一次是 large 流被领回来重跑。
    assert _reclaimed(STREAM_LARGE) - before == 1


def _reclaimed(stream: str) -> float:
    value = REGISTRY.get_sample_value("shoppingx_queue_reclaimed_total", {"stream": stream})
    return value or 0.0


# ── 租约与重投去重 ──────────────────────────────────────────────────────────
_SHORT = dict.fromkeys((STREAM_NORMAL, STREAM_LARGE), 150)


def _leases(fake: FakeRedis) -> dict[str, str]:
    return {k: v for k, v in fake.kv.items() if k.startswith("globex:lease:") and fake._live(k)}


async def test_lease_keeps_long_task_from_being_stolen(
    rq: RedisStreamQueue, fake: FakeRedis
) -> None:
    """跑得比租约还久的任务靠续期不被接管——别人不行，自己的空转扫描也不能把它再领一遍。"""
    await rq.ensure_group()
    await rq.enqueue(_task(tid="long"))
    done = asyncio.Event()

    async def _slow(_task: IntentTask) -> None:
        await asyncio.sleep(0.5)
        done.set()

    consumer = asyncio.create_task(
        _consume_until(
            rq, done.is_set, _slow, block_ms=0, claim_idle_ms=0, heartbeat_sec=0.04, lease_ms=150
        )
    )
    await asyncio.sleep(0.3)  # 已是租约的 2 倍：不续期的话早过期了
    assert await rq._reclaim("thief", 0, 10, _SHORT) == []
    assert await rq._reclaim("worker-1", 0, 10, _SHORT) == []
    await consumer
    assert fake.eval_calls[LEASE_RENEW] >= 3
    assert fake.xclaim_calls == 0
    assert fake.pending_ids(STREAM_NORMAL) == []
    assert _leases(fake) == {}  # 跑完即释放


async def test_expired_lease_is_taken_over(rq: RedisStreamQueue, fake: FakeRedis) -> None:
    """对照组：持有者不再续期，租约过期后被接管，投递计数 +1、租约换成接管方。"""
    await rq.ensure_group()
    await rq.enqueue(_task(tid="long"))
    await fake.xreadgroup(GROUP, "dead-worker", {STREAM_NORMAL: ">"}, count=1, block=0)
    [mid] = fake.pending_ids(STREAM_NORMAL)
    key = _lease_key(STREAM_NORMAL, mid)
    assert await rq._acquire(key, "dead-worker", 100)
    assert await rq._reclaim("thief", 0, 10, _SHORT) == []  # 租约还在：idle 再大也不动
    await asyncio.sleep(0.15)
    stolen = await rq._reclaim("thief", 0, 10, _SHORT)
    assert [m for _s, m, _f in stolen] == [mid]
    entry = fake.groups[(STREAM_NORMAL, GROUP)]["pending"][mid]
    assert entry["consumer"] == "thief" and entry["times_delivered"] == 2
    assert fake.kv[key] == "thief"


async def test_idle_threshold_guards_the_pre_lease_window(
    rq: RedisStreamQueue, fake: FakeRedis
) -> None:
    """刚领到、还没来得及占租约的消息 idle 很小，不在接管候选里。"""
    await rq.ensure_group()
    await rq.enqueue(_task(tid="a"))
    await fake.xreadgroup(GROUP, "fresh", {STREAM_NORMAL: ">"}, count=1, block=0)
    assert await rq._reclaim("thief", 10_000, 10, _SHORT) == []
    assert _leases(fake) == {}  # 连租约都没去抢


async def test_two_takers_only_one_wins(rq: RedisStreamQueue, fake: FakeRedis) -> None:
    """两个接管方同时看到租约没了：SET NX 只放一个过去，不会双双 XCLAIM。"""
    await rq.ensure_group()
    await rq.enqueue(_task(tid="a"))
    await fake.xreadgroup(GROUP, "dead-worker", {STREAM_NORMAL: ">"}, count=1, block=0)
    a, b = await asyncio.gather(
        rq._reclaim("w-a", 0, 10, _SHORT), rq._reclaim("w-b", 0, 10, _SHORT)
    )
    assert sorted([len(a), len(b)]) == [0, 1]
    assert fake.xclaim_calls == 1


async def test_renew_never_resurrects_a_lost_lease(rq: RedisStreamQueue, fake: FakeRedis) -> None:
    """持有者卡住、租约过期被别人占走后醒来：续期失败即停，不把租约抢回来。

    这正是上一版（XPENDING 查属主 → XCLAIM JUSTID 续租）两条命令之间的漏洞。
    """
    key = _lease_key(STREAM_NORMAL, "1-0")
    assert await rq._acquire(key, "me", 1000)
    fake.expire_now(key)
    assert await fake.set(key, "thief", px=1000, nx=True)
    await asyncio.wait_for(rq._heartbeat(key, "me", 0.01, 1000), 1.0)
    assert fake.kv[key] == "thief"


async def test_release_only_deletes_own_lease(rq: RedisStreamQueue, fake: FakeRedis) -> None:
    key = _lease_key(STREAM_NORMAL, "1-0")
    await fake.set(key, "thief", px=1000)
    await rq._release(key, "me")
    assert fake.kv[key] == "thief"
    await rq._release(key, "thief")
    assert key not in fake.kv


@pytest.mark.parametrize("state", ["done", "cancelled", "interrupted"])
async def test_redelivery_of_finished_task_is_skipped(
    rq: RedisStreamQueue, fake: FakeRedis, state: str
) -> None:
    """worker 写完终态、XACK 前被 kill：接管方按 task_id 查到终态，直接 ack，不重跑。"""
    await rq.ensure_group()
    await rq.enqueue(_task(tid="a"))
    await rq.set_status(TaskStatus(task_id="a", state=state, thread_id="a"))  # type: ignore[arg-type]
    calls: list[str] = []

    async def _handler(task: IntentTask) -> None:
        calls.append(task.task_id)

    def _acked() -> bool:
        cursor = fake.groups[(STREAM_NORMAL, GROUP)]["cursor"]
        return cursor == 1 and fake.pending_ids(STREAM_NORMAL) == []

    await _consume_until(rq, _acked, _handler, block_ms=0)
    assert calls == []


async def test_failed_status_is_still_retried(rq: RedisStreamQueue, fake: FakeRedis) -> None:
    """``failed`` 不在跳过名单里：失败留 PEL 本来就是为了重跑。"""
    await rq.ensure_group()
    await rq.enqueue(_task(tid="a"))
    await rq.set_status(TaskStatus(task_id="a", state="failed", thread_id="a"))
    handled = await _drain(rq, 1, block_ms=0)
    assert [t.task_id for t in handled] == ["a"]


async def test_unparsable_payload_goes_straight_to_dead(
    rq: RedisStreamQueue, fake: FakeRedis
) -> None:
    await rq.ensure_group()
    await fake.xadd(STREAM_NORMAL, {"payload": "{ 这不是 json"})
    called: list[IntentTask] = []

    async def _never(task: IntentTask) -> None:
        called.append(task)

    await _consume_until(
        rq, lambda: bool(fake.streams.get(STREAM_DEAD)), _never, block_ms=0, claim_idle_ms=10_000
    )
    assert called == []  # 解不开就别喂给 handler
    assert "解析失败" in fake.streams[STREAM_DEAD][0][1]["reason"]
    assert fake.pending_ids(STREAM_NORMAL) == []


# ── 观测：depth / status ────────────────────────────────────────────────────
async def test_depth_sums_both_streams(rq: RedisStreamQueue) -> None:
    assert await rq.depth() == 0  # 流还没建也要给得出数，不能抛
    await rq.ensure_group()
    await rq.enqueue(_task(tid="a"))
    await rq.enqueue(_task(tid="b"))
    await rq.enqueue(_task(turns=50, tid="c"))
    assert await rq.depth() == 3


async def test_status_roundtrip_attaches_depth_only_while_queued(rq: RedisStreamQueue) -> None:
    await rq.ensure_group()
    await rq.enqueue(_task(tid="a"))
    await rq.set_status(TaskStatus(task_id="a", state="queued", thread_id="t1"))
    queued = await rq.get_status("a")
    assert queued is not None and queued.queue_depth == 1

    await rq.set_status(TaskStatus(task_id="a", state="done", final_text="给你挑了 8 件"))
    done = await rq.get_status("a")
    assert done is not None and done.state == "done" and done.queue_depth == 0
    assert await rq.get_status("missing") is None


# ── 进程内回落 ──────────────────────────────────────────────────────────────
async def test_inprocess_prioritizes_normal_and_acks_nothing() -> None:
    queue = InProcessQueue()
    await queue.enqueue(_task(turns=50, tid="long"))
    await queue.enqueue(_task(turns=0, tid="short"))
    assert await queue.depth() == 2
    handled = await _drain(queue, 2, poll_interval=0.01)
    assert [t.task_id for t in handled] == ["short", "long"]
    assert await queue.depth() == 0


async def test_inprocess_failure_marks_status_failed() -> None:
    queue = InProcessQueue()
    await queue.enqueue(_task(tid="a"))
    await _drain(queue, 1, fail=True, poll_interval=0.01)
    status = await queue.get_status("a")
    assert status is not None and status.state == "failed" and "boom" in status.error


async def test_inprocess_status_table_is_bounded() -> None:
    queue = InProcessQueue()
    for i in range(1100):
        await queue.set_status(TaskStatus(task_id=f"t{i}", state="done"))
    assert await queue.get_status("t0") is None  # 最老的被挤掉，不随任务数无界增长
    assert await queue.get_status("t1099") is not None


# ── 工厂 ────────────────────────────────────────────────────────────────────
def test_factory_builds_a_redis_stream_queue(monkeypatch: pytest.MonkeyPatch) -> None:
    """工厂恒给 Redis Stream（已删掉 QUEUE_ENABLED）；单例只建一次。"""
    monkeypatch.setattr(queue_pkg, "_queue", None)
    monkeypatch.setenv("QUEUE_REDIS_URL", "redis://127.0.0.1:6379/15")
    q = get_task_queue()
    assert isinstance(q, RedisStreamQueue)
    assert get_task_queue() is q  # 单例


def test_factory_raises_instead_of_falling_back(monkeypatch: pytest.MonkeyPatch) -> None:
    """客户端建不起来就抛：悄悄回落进程内 deque = 把任务扔进一个没有消费方的队列。"""
    monkeypatch.setattr(queue_pkg, "_queue", None)
    monkeypatch.setenv("QUEUE_REDIS_URL", "notredis://nowhere")
    with pytest.raises(RuntimeError):
        get_task_queue()


def test_both_implementations_satisfy_the_port(fake: FakeRedis) -> None:
    assert isinstance(InProcessQueue(), TaskQueue)
    assert isinstance(RedisStreamQueue(fake), TaskQueue)


# ── worker 进程（app/worker.py）─────────────────────────────────────────────
#
# 这一组钉的是**优雅退出**：SIGTERM 之后停领新任务、等在飞跑完、超时把它们交还队列重投。三步任一
# 步错了都不会报错，只会在滚动更新时静默丢任务或双跑，所以每一步都要有一条能变红的用例。


async def test_worker_writes_running_then_done(monkeypatch: pytest.MonkeyPatch) -> None:
    """状态由 worker 写：跑之前 running（轮询方看得见它已经开工），跑完 done + 结论文本。"""
    queue = InProcessQueue()
    seen: list[str] = []

    async def _fake_run(query: str, thread_id: str, **_kw: Any) -> dict[str, Any]:
        status = await queue.get_status("t1")
        seen.append(status.state if status else "missing")
        return {"final_text": "这三件更耐操"}

    monkeypatch.setattr(worker, "run_agent", _fake_run)
    await worker.handle_task(_task(), queue)

    assert seen == ["running"]
    done = await queue.get_status("t1")
    assert done is not None and done.state == "done"
    assert done.final_text == "这三件更耐操"


async def test_worker_continues_trace_and_binds_trace_id(monkeypatch: pytest.MonkeyPatch) -> None:
    """worker 把 traceparent 原样交给 run_agent（根 span 靠它挂到 API 那段下面），并把 trace_id
    绑进日志上下文——两个进程的日志靠它串成一条线。"""
    seen: dict[str, Any] = {}

    async def _fake_run(query: str, thread_id: str, **kw: Any) -> dict[str, Any]:
        seen["traceparent"] = kw.get("traceparent")
        seen["log_ctx"] = structlog.contextvars.get_contextvars()
        return {"final_text": "ok"}

    monkeypatch.setattr(worker, "run_agent", _fake_run)
    await worker.handle_task(replace(_task(), traceparent=_TP, request_id="rq1"), InProcessQueue())

    assert seen["traceparent"] == _TP
    assert seen["log_ctx"]["trace_id"] == "ab" * 16
    assert seen["log_ctx"]["request_id"] == "rq1"
    assert "trace_id" not in structlog.contextvars.get_contextvars(), "收尾要解绑，不许外溢"


async def test_worker_failure_writes_failed_and_reraises(monkeypatch: pytest.MonkeyPatch) -> None:
    """失败必须往外抛：队列侧靠这个异常决定「留 PEL 重投」还是「进死信」，吞掉就没人管了。"""
    queue = InProcessQueue()

    async def _boom(*_a: Any, **_kw: Any) -> dict[str, Any]:
        raise RuntimeError("模型挂了")

    monkeypatch.setattr(worker, "run_agent", _boom)
    with pytest.raises(RuntimeError):
        await worker.handle_task(_task(), queue)

    status = await queue.get_status("t1")
    assert status is not None and status.state == "failed" and "模型挂了" in status.error


async def test_worker_shutdown_cancel_writes_interrupted(monkeypatch: pytest.MonkeyPatch) -> None:
    """关停掐断写 ``interrupted`` 而不是 failed：任务没毛病，是进程要走了。

    写 failed 会让脚本类调用方按「跑挂了」重试同一条；不写终态（早先的做法）则让轮询方
    一直等下去——而消息现在是被 ack 掉的，那边等的东西永远不会再动。
    """
    queue = InProcessQueue()

    async def _hang(*_a: Any, **_kw: Any) -> dict[str, Any]:
        await asyncio.sleep(10)
        return {}

    monkeypatch.setattr(worker, "run_agent", _hang)
    running = asyncio.create_task(worker.handle_task(_task(), queue))
    await asyncio.sleep(0.05)
    running.cancel()
    await asyncio.wait_for(running, 2.0)  # 不往外抛 = 调用方会 ack 掉这条消息

    status = await queue.get_status("t1")
    assert status is not None and status.state == "interrupted"
    assert status.error  # 带上「请重发」的说明，别让轮询方只看到一个陌生状态词


async def test_worker_stop_finishes_inflight_and_leaves_rest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """停止信号之后：在飞的那条跑完，还没领的那条原封不动留在队列里。"""
    queue = InProcessQueue()
    started = asyncio.Event()
    release = asyncio.Event()

    async def _slow(query: str, thread_id: str, **_kw: Any) -> dict[str, Any]:
        started.set()
        await release.wait()
        return {"final_text": "ok"}

    monkeypatch.setattr(worker, "run_agent", _slow)
    await queue.enqueue(_task(tid="t1"))
    stop = asyncio.Event()
    runner = asyncio.create_task(
        worker.run_worker(queue, concurrency=2, grace_seconds=5, stop=stop, install_signals=False)
    )
    await asyncio.wait_for(started.wait(), 2.0)

    stop.set()
    await asyncio.sleep(0.05)  # 让消费循环先走出 while，确认它之后不再领新的
    await queue.enqueue(_task(tid="t2"))
    release.set()
    await asyncio.wait_for(runner, 3.0)

    finished = await queue.get_status("t1")
    assert finished is not None and finished.state == "done"
    assert await queue.get_status("t2") is None  # 停止之后入的队没被领走
    assert await queue.depth() == 1  # 它还躺在队列里


async def test_worker_grace_timeout_returns_message_to_pending(
    fake: FakeRedis, rq: RedisStreamQueue, monkeypatch: pytest.MonkeyPatch
) -> None:
    """宽限期内跑不完 → 掐掉在飞任务 → 按 interrupted 收尾并**ack 掉**，PEL 清空。

    这条断言的方向：从前是「不 ack、留 PEL 等重投」，现在是「给个定论、ack 掉，
    要不要再来一次由用户决定」（理由见 worker 模块 docstring 第 3 条）。

    把 ports.cancel_in_flight 那一手去掉即红：消费循环被取消时在途 task 会变成孤儿协程（进程都在
    退出还在跑 LLM），没人替它收尾，PEL 也就空不掉——所以额外断言在飞协程真的被取消了。
    """
    cancelled = asyncio.Event()

    async def _hang(*_a: Any, **_kw: Any) -> dict[str, Any]:
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return {}

    monkeypatch.setattr(worker, "run_agent", _hang)
    await rq.ensure_group()
    await rq.enqueue(_task())
    stop = asyncio.Event()
    runner = asyncio.create_task(
        worker.run_worker(rq, concurrency=1, grace_seconds=0, stop=stop, install_signals=False)
    )
    for _ in range(100):  # 等它把消息领进 PEL
        await asyncio.sleep(0.02)
        if fake.pending_ids(STREAM_NORMAL):
            break
    assert fake.pending_ids(STREAM_NORMAL)

    stop.set()
    await asyncio.wait_for(runner, 3.0)
    await asyncio.wait_for(cancelled.wait(), 1.0)
    assert not fake.pending_ids(STREAM_NORMAL)  # 已 ack：不会有下一个 worker 背着用户重跑
    status = await rq.get_status("t1")
    assert status is not None and status.state == "interrupted"
