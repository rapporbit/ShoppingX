"""批 2 · 削峰队列的确定性单测：双流分级 / ack / pending 重投 / 死信 / 进程内回落。

**测试策略沿用 `tests/test_event_replay.py` 的惯例**：一个内存 FakeRedis 实现 Stream 消费者组的最小
子集（xadd / xreadgroup / xack / xpending_range / xautoclaim 等），不依赖真 Redis，也不引
fakeredis 包。真 Redis 的价值在于验协议细节，而这里要钉的是**我们自己的取舍**——normal 先于 large、
ack 回原流、失败留 PEL、超限进死信——这些用假客户端反而断言得更死（能把「投递第几次」直接摆出来）。

FakeRedis 刻意实现了 ``times_delivered`` 与 ``min_idle_time``：死信与重投的判据全压在这两个数上，
把它们写成常量假值的话，这组用例会在真实行为退化时照样绿。
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import pytest

from app import worker
from app.queue import InProcessQueue, RedisStreamQueue, get_task_queue, set_task_queue
from app.queue.ports import IntentTask, TaskQueue, TaskStatus
from app.queue.redis_stream import GROUP, STREAM_DEAD, STREAM_LARGE, STREAM_NORMAL


class FakeRedis:
    """Redis Stream 消费者组的最小内存实现（含 PEL、投递计数、idle 时间）。"""

    def __init__(self) -> None:
        self.streams: dict[str, list[tuple[str, dict[str, Any]]]] = {}
        self.groups: dict[tuple[str, str], dict[str, Any]] = {}
        self.kv: dict[str, str] = {}
        self.counter = 0

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
        self, stream: str, group: str, min: str, max: str, count: int = 10
    ) -> list[dict[str, Any]]:
        state = self.groups.get((stream, group), {"pending": {}})
        entry = state["pending"].get(min)
        if entry is None:
            return []
        return [{"message_id": min, "times_delivered": entry["times_delivered"]}]

    async def xautoclaim(
        self,
        stream: str,
        group: str,
        consumer: str,
        min_idle_time: int = 0,
        count: int = 10,
    ) -> tuple[str, list[tuple[str, dict[str, Any]]], list[str]]:
        state = self.groups.get((stream, group))
        if state is None:
            return "0-0", [], []
        now = time.monotonic()
        claimed: list[tuple[str, dict[str, Any]]] = []
        body = dict(self.streams.get(stream, []))
        for sid, entry in list(state["pending"].items()):
            if len(claimed) >= count:
                break
            if (now - entry["delivered_at"]) * 1000 < min_idle_time:
                continue
            entry["consumer"] = consumer
            entry["times_delivered"] += 1
            entry["delivered_at"] = now
            claimed.append((sid, body.get(sid, {})))
        return "0-0", claimed, []

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

    # ── 状态键 ──────────────────────────────────────────────────────────────
    async def set(self, key: str, value: str, ex: int | None = None) -> bool:
        self.kv[key] = value
        return True

    async def get(self, key: str) -> str | None:
        return self.kv.get(key)

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
        task_id="a", thread_id="t", query="q", history_turns=99, platforms=["amazon"]
    )
    assert IntentTask.from_dict(task.to_dict()) == task
    # 滚动更新期间旧进程写的 payload 缺新字段——只有三个必需键在就该能跑起来。
    lean = IntentTask.from_dict({"task_id": "a", "thread_id": "t", "query": "q"})
    assert (lean.kind, lean.platforms, lean.user_id) == ("normal", (), None)


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

    handled = await _drain(rq, 1, block_ms=0, claim_idle_ms=0)
    assert [t.task_id for t in handled] == ["long"]
    assert fake.pending_ids(STREAM_LARGE) == []


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
def test_factory_defaults_to_inprocess(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("QUEUE_ENABLED", raising=False)
    assert isinstance(get_task_queue(), InProcessQueue)
    assert get_task_queue() is get_task_queue()  # 单例


def test_factory_falls_back_when_redis_client_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """开了队列但 Redis 客户端建不起来时回落进程内——启动期缺依赖不该让整个服务起不来。"""
    monkeypatch.setenv("QUEUE_ENABLED", "1")
    monkeypatch.setenv("QUEUE_REDIS_URL", "notredis://nowhere")
    assert isinstance(get_task_queue(), InProcessQueue)


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


async def test_worker_cancel_leaves_status_non_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    """被取消不写终态：这条消息没 ack，会被下一个 worker 领回重跑，写 failed 是在骗轮询方。"""
    queue = InProcessQueue()

    async def _hang(*_a: Any, **_kw: Any) -> dict[str, Any]:
        await asyncio.sleep(10)
        return {}

    monkeypatch.setattr(worker, "run_agent", _hang)
    running = asyncio.create_task(worker.handle_task(_task(), queue))
    await asyncio.sleep(0.05)
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running

    status = await queue.get_status("t1")
    assert status is not None and status.state == "running"


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
    """宽限期内跑不完 → 掐掉在飞任务 → 消息**不 ack**、留在 PEL 里等下一个 worker 领回重跑。

    这条是「超时转回 pending」的回归。把 ports.cancel_in_flight 那一手去掉即红：消费循环被取消时
    在途 task 会变成孤儿协程（进程都在退出还在跑 LLM），而这里断言的 PEL 反倒照样是满的——所以
    额外断言在飞协程真的被取消了。
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
    assert fake.pending_ids(STREAM_NORMAL)  # 仍未 ack，可被 XAUTOCLAIM 领回


def test_worker_main_refuses_when_queue_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """QUEUE_ENABLED=0 起 worker 是纯误配：它消费的进程内 deque 没有生产方，宁可起不来。"""
    monkeypatch.delenv("QUEUE_ENABLED", raising=False)
    with pytest.raises(SystemExit):
        worker.main()
