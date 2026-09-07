"""批 2 · 事件背板：跨进程 AGUI 事件转发 / origin 去重 / fire-and-forget 强引用 / 静默降级。

**测试策略沿用 `tests/test_queue.py` 的惯例**：一个进程内的 ``FakeBus`` 当「同一个 Redis」，多个
``FakeRedis`` 客户端接在上面当「不同进程」。真 Redis 的价值在验协议（`listen()` 会先吐一条
subscribe 确认帧、data 可能是 bytes），这两件事在 FakeRedis 里**照原样复刻**，其余噪声不引入；
真双进程的冒烟另跑（见 docs/plans/批2-进度.md）。

本文件要钉死的四件事，每一件失手都不会报错、只会静默出错：
1. 远端进程的事件能落到本进程挂着的那条 WS 上（背板存在的全部理由）；
2. **自己发的不再收一遍**（origin 去重失手 = 前端每条事件重复 N 遍）；
3. publish_nowait 的任务**有人持强引用**（失手 = 偶发丢事件，且无法复现）；
4. Redis 抽风时一律静默降级（失手 = 事件侧把主链路拖垮，与批2-1 定的降级方向相反）。
"""

from __future__ import annotations

import asyncio
import gc
import json
from typing import Any

import pytest

from app.api import backplane as bp
from app.api import monitor
from app.api.backplane import EventBackplane
from app.api.connection import ConnectionManager


class FakeBus:
    """一个 Redis 实例：频道 → 订阅者队列。多个 FakeRedis 共享它即「连同一个 Redis 的多个进程」。"""

    def __init__(self) -> None:
        self.subscribers: dict[str, list[asyncio.Queue[str]]] = {}
        self.published: list[tuple[str, str]] = []

    def publish(self, channel: str, data: str) -> None:
        self.published.append((channel, data))
        for queue in self.subscribers.get(channel, []):
            queue.put_nowait(data)


class FakePubSub:
    """复刻两处真实行为：``listen()`` 先吐一条 subscribe 确认帧；``data`` 是 bytes。"""

    def __init__(self, bus: FakeBus) -> None:
        self._bus = bus
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._channel = ""
        self.closed = False

    async def subscribe(self, channel: str) -> None:
        self._channel = channel
        self._bus.subscribers.setdefault(channel, []).append(self._queue)

    async def listen(self) -> Any:
        yield {"type": "subscribe", "channel": self._channel, "data": 1}
        while True:
            data = await self._queue.get()
            yield {"type": "message", "channel": self._channel, "data": data.encode()}

    async def aclose(self) -> None:
        self.closed = True
        subs = self._bus.subscribers.get(self._channel, [])
        if self._queue in subs:
            subs.remove(self._queue)


class FakeRedis:
    """一个「进程」的客户端。``fail_publish`` 用来演 Redis 抽风。"""

    def __init__(self, bus: FakeBus, *, fail_publish: bool = False) -> None:
        self.bus = bus
        self.fail_publish = fail_publish
        self.closed = False

    async def publish(self, channel: str, data: str) -> int:
        if self.fail_publish:
            raise ConnectionError("redis 挂了")
        self.bus.publish(channel, data)
        return 1

    def pubsub(self) -> FakePubSub:
        return FakePubSub(self.bus)

    async def aclose(self) -> None:
        self.closed = True


class FakeWS:
    """ConnectionManager 只要 accept / send_json 两个动作（见 WebSocketLike）。"""

    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []

    async def accept(self) -> None:
        return None

    async def send_json(self, data: Any) -> None:
        self.sent.append(data)


def make_event(thread_id: str = "t-1", event: str = "tool_start") -> dict[str, Any]:
    return {
        "type": "monitor_event",
        "event": event,
        "message": "正在调用 item_search",
        "data": {"tool": "item_search"},
        "thread_id": thread_id,
    }


async def wait_for(predicate: Any, limit: float = 1.0) -> bool:
    """轮询等一个条件成立（转发链路是异步的，断言不能紧贴着 publish 写）。"""
    deadline = asyncio.get_running_loop().time() + limit
    while asyncio.get_running_loop().time() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.005)
    return False


@pytest.fixture(autouse=True)
def _reset_singleton() -> Any:
    """每个用例前后都把进程级背板复位，免得一个用例注入的实例漏给下一个（monitor 直接读它）。"""
    bp.reset_backplane()
    yield
    bp.reset_backplane()


@pytest.fixture
async def wired() -> Any:
    """装配「两个进程」：A 有 WS 并订阅背板，B（worker）只发布。返回 (A, B, manager, ws)。"""
    bus = FakeBus()
    manager = ConnectionManager()
    ws = FakeWS()
    await manager.connect(ws, "t-1")
    api = EventBackplane(FakeRedis(bus), origin="origin-api")
    worker = EventBackplane(FakeRedis(bus), origin="origin-worker")
    await api.start(manager)
    try:
        yield api, worker, manager, ws
    finally:
        await api.stop()
        await worker.stop()


# ── 1. 主干：远端事件落到本进程的 WS ────────────────────────────────────────────
async def test_remote_event_reaches_local_websocket(wired: Any) -> None:
    """worker 进程发的事件，浏览器连着的 API 进程要能推给前端——背板存在的全部理由。"""
    _api, worker, _manager, ws = wired
    payload = make_event()
    assert await worker.publish(payload) is True
    assert await wait_for(lambda: ws.sent), "远端事件没有转发到本地 WS"
    assert ws.sent[0] == payload  # 原样转发，不加壳不改字段（前端按同一套结构解析）


async def test_remote_event_for_unknown_thread_is_dropped(wired: Any) -> None:
    """收端按 thread_id 过滤：不是本进程挂着的会话，收到也不推（多副本时人人都收得到广播）。"""
    _api, worker, _manager, ws = wired
    await worker.publish(make_event(thread_id="t-somewhere-else"))
    await asyncio.sleep(0.05)
    assert ws.sent == []


# ── 2. origin 去重 ──────────────────────────────────────────────────────────────
async def test_own_message_is_skipped_by_origin(wired: Any) -> None:
    """自己发的事件从背板兜回来时必须丢掉，否则前端每条事件出现 N 遍（N = 副本数）。"""
    api, _worker, _manager, ws = wired
    await api.publish(make_event())
    await asyncio.sleep(0.05)
    assert ws.sent == [], "origin 去重失效：自己发的事件又推了一遍"


async def test_origin_travels_in_the_envelope(wired: Any) -> None:
    """信封结构本身也钉一下：origin 在外层、AGUI 事件原样在 payload 里。"""
    _api, worker, _manager, _ws = wired
    await worker.publish(make_event())
    _channel, raw = worker._client.bus.published[-1]
    envelope = json.loads(raw)
    assert envelope["origin"] == "origin-worker"
    assert envelope["payload"]["event"] == "tool_start"


# ── 3. fire-and-forget 的强引用 ────────────────────────────────────────────────
async def test_publish_nowait_holds_strong_reference(wired: Any) -> None:
    """``create_task`` 的返回值不留引用会被 GC 掉（偶发丢事件且无法复现），故必须进 set。"""
    _api, worker, _manager, ws = wired
    worker.publish_nowait(make_event())
    assert worker._tasks, "publish_nowait 没有持住任务引用"
    gc.collect()  # 强引用在，GC 拿不走
    assert worker._tasks
    assert await wait_for(lambda: ws.sent)
    assert await wait_for(lambda: not worker._tasks), "任务完成后没有从强引用池摘除"


async def test_publish_nowait_drops_beyond_max_inflight(wired: Any) -> None:
    """Redis 连得上但不响应时任务会一直堆——到顶丢新事件，宁可少几条也不让它拖垮进程。"""
    bus = FakeBus()

    class SlowRedis(FakeRedis):
        async def publish(self, channel: str, data: str) -> int:
            await asyncio.sleep(5)
            return 1

    backplane = EventBackplane(SlowRedis(bus), origin="o", max_inflight=1)
    backplane.publish_nowait(make_event())
    backplane.publish_nowait(make_event())
    assert len(backplane._tasks) == 1
    assert backplane.dropped == 1
    await backplane.stop()


# ── 4. 静默降级 ────────────────────────────────────────────────────────────────
async def test_publish_failure_is_silent() -> None:
    """Redis 挂了：发布返回 False 而不是抛——事件侧的降级方向与队列相反（见批2-1 口径）。"""
    backplane = EventBackplane(FakeRedis(FakeBus(), fail_publish=True), origin="o")
    assert await backplane.publish(make_event()) is False
    backplane.publish_nowait(make_event())  # 也不许从 fire-and-forget 那条路炸出来
    await asyncio.sleep(0.02)
    await backplane.stop()


async def test_malformed_messages_are_ignored(wired: Any) -> None:
    """频道上混进别的东西（非 JSON / 没有 payload / 没有 thread_id）时跳过，不能把订阅循环搞死。"""
    api, worker, _manager, ws = wired
    worker._client.bus.publish(bp.CHANNEL, "not json at all")
    worker._client.bus.publish(bp.CHANNEL, json.dumps({"origin": "x"}))
    worker._client.bus.publish(bp.CHANNEL, json.dumps({"origin": "x", "payload": {"a": 1}}))
    await worker.publish(make_event())  # 坏消息之后这条好的仍要送到 —— 证明循环还活着
    assert await wait_for(lambda: ws.sent)
    assert len(ws.sent) == 1
    assert api._reader is not None and not api._reader.done()


async def test_subscription_reconnects_after_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pub/Sub 长连接断了不会报错、只会从此收不到事件——所以必须自己退避重连。"""
    monkeypatch.setattr(bp, "_RETRY_SECONDS", 0.01)
    bus = FakeBus()
    attempts = {"n": 0}

    class FlakyRedis(FakeRedis):
        def pubsub(self) -> FakePubSub:
            attempts["n"] += 1
            pubsub = FakePubSub(self.bus)
            if attempts["n"] == 1:

                async def _boom(channel: str) -> None:
                    raise ConnectionError("连接被重置")

                pubsub.subscribe = _boom  # type: ignore[method-assign]
            return pubsub

    manager = ConnectionManager()
    ws = FakeWS()
    await manager.connect(ws, "t-1")
    api = EventBackplane(FlakyRedis(bus), origin="origin-api")
    await api.start(manager)
    try:
        assert await wait_for(lambda: attempts["n"] >= 2), "订阅断开后没有重连"
        assert await wait_for(lambda: bool(bus.subscribers.get(bp.CHANNEL)))
        worker = EventBackplane(FakeRedis(bus), origin="origin-worker")
        await worker.publish(make_event())
        assert await wait_for(lambda: ws.sent), "重连之后仍收不到事件"
    finally:
        await api.stop()


# ── 5. 工厂开关：默认关，单进程部署一个字节都不执行 ──────────────────────────────
def test_disabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BACKPLANE_ENABLED", raising=False)
    monkeypatch.delenv("QUEUE_ENABLED", raising=False)
    assert bp.backplane_enabled() is False
    assert bp.get_backplane() is None


def test_follows_queue_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """默认跟随 QUEUE_ENABLED：开了队列就必须开背板，否则前端一条实时事件都收不到。"""
    monkeypatch.delenv("BACKPLANE_ENABLED", raising=False)
    monkeypatch.setenv("QUEUE_ENABLED", "1")
    assert bp.backplane_enabled() is True
    monkeypatch.setenv("BACKPLANE_ENABLED", "0")  # 显式关掉仍然说了算
    assert bp.backplane_enabled() is False


async def test_start_forwarding_noop_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """关着时 lifespan 那一行什么也不做（不 import redis、不起 task）。"""
    monkeypatch.delenv("BACKPLANE_ENABLED", raising=False)
    monkeypatch.delenv("QUEUE_ENABLED", raising=False)
    assert await bp.start_forwarding(ConnectionManager()) is None


# ── 6. 与 monitor 的接线：只在本地投不到时才广播 ────────────────────────────────
@pytest.fixture
def quiet_event_log(monkeypatch: pytest.MonkeyPatch) -> None:
    """事件回放走 Redis，本文件不验它——钉成 no-op，免得每条事件干等一次连接超时。"""

    async def _append(thread_id: str, payload: dict[str, Any]) -> None:
        return None

    monkeypatch.setattr(monitor.event_log, "append", _append)


async def test_monitor_broadcasts_when_no_local_connection(
    quiet_event_log: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """worker 进程的处境：一条 WS 都没挂着 → 事件必须上背板，否则前端永远收不到。"""
    bus = FakeBus()
    backplane = EventBackplane(FakeRedis(bus), origin="origin-worker")
    bp.set_backplane(backplane)
    monkeypatch.setattr(monitor, "_manager", ConnectionManager())
    await monitor.report_error("boom", "出事了", thread_id="t-1")
    assert await wait_for(lambda: bus.published), "本地投不到却没有广播到背板"
    envelope = json.loads(bus.published[-1][1])
    assert envelope["payload"]["event"] == "error"
    assert envelope["payload"]["thread_id"] == "t-1"
    await backplane.stop()


async def test_monitor_skips_broadcast_when_delivered_locally(
    quiet_event_log: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """API 进程的处境：WS 就在手边，直投成功后再广播一遍纯属白烧带宽（收端也只会丢掉）。"""
    bus = FakeBus()
    backplane = EventBackplane(FakeRedis(bus), origin="origin-api")
    bp.set_backplane(backplane)
    manager = ConnectionManager()
    ws = FakeWS()
    await manager.connect(ws, "t-1")
    monkeypatch.setattr(monitor, "_manager", manager)
    await monitor.report_error("boom", "出事了", thread_id="t-1")
    await asyncio.sleep(0.02)
    assert len(ws.sent) == 1
    assert bus.published == []
    await backplane.stop()


async def test_monitor_is_noop_when_backplane_off(
    quiet_event_log: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """默认（单进程）路径：monitor 里那两行一个字节都不执行，行为与批2-2 末态逐字相同。"""
    monkeypatch.delenv("BACKPLANE_ENABLED", raising=False)
    monkeypatch.delenv("QUEUE_ENABLED", raising=False)
    monkeypatch.setattr(monitor, "_manager", ConnectionManager())
    await monitor.report_error("boom", "出事了", thread_id="t-1")  # 不抛即可
    assert bp.get_backplane() is None
