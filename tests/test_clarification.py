"""app/api/clarification 单元测试：Future 的创建 / 解决 / 取消 / 超时 + 跨进程澄清与取消。"""

import asyncio
import json
import time
from typing import Any

import pytest

from app.api import control
from app.api.clarification import (
    KIND_REPLY,
    cancel_pending,
    clear_waiter,
    create_pending,
    deliver_reply,
    drop_stale_waiter,
    has_pending,
    on_reply_message,
    register_waiter,
    resolve_pending,
    set_turn_id,
)


@pytest.fixture(autouse=True)
def _clean():
    """每个用例后清理残留的 pending Future。"""
    yield
    cancel_pending("test-thread")


async def test_create_and_resolve():
    fut = create_pending("test-thread")
    assert has_pending("test-thread")
    assert not fut.done()
    assert resolve_pending("test-thread", "女款")
    assert await fut == "女款"
    assert not has_pending("test-thread")


async def test_resolve_nonexistent_returns_false():
    assert not resolve_pending("no-such-thread", "hello")


async def test_cancel_pending():
    fut = create_pending("test-thread")
    cancel_pending("test-thread")
    assert not has_pending("test-thread")
    assert fut.cancelled()


async def test_cancel_nonexistent_is_noop():
    cancel_pending("no-such-thread")


async def test_timeout():
    fut = create_pending("test-thread")
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(fut, timeout=0.05)
    assert not has_pending("test-thread") or fut.done()


async def test_replace_existing():
    """create_pending 对同一 thread 再调会 cancel 旧的、返回新的。"""
    fut1 = create_pending("test-thread")
    fut2 = create_pending("test-thread")
    assert fut1.cancelled()
    assert not fut2.done()
    resolve_pending("test-thread", "ok")
    assert await fut2 == "ok"


async def test_resolve_after_cancel_returns_false():
    create_pending("test-thread")
    cancel_pending("test-thread")
    assert not resolve_pending("test-thread", "too late")


# ══════════════════════════════════════════════════════════════════════════════
# 跨进程澄清 / 取消（批2-4）
#
# 单进程那半边（上面那批）是本模块的**回归基线**：跨进程改造之后它们必须一行不改地全绿——
# ``deliver_reply`` 的第一步就是 ``resolve_pending``，队列关着时后面的代码一个字节都不执行。
# 下面这批钉的是新加的那半边：等待令牌、reply_id 比对、令牌过期即拒收、控制面取消。
#
# 用一个内存假 Redis 驱动**真的** ControlBus，而不是把 bus 整个换成桩：这里要验的恰恰是
# ControlBus 自己的取舍（origin 去重、坏消息不带走订阅循环、键的 TTL 语义），换成桩就全测空了。
# ══════════════════════════════════════════════════════════════════════════════


class FakePubSub:
    """``pubsub()`` 的最小替身：``listen()`` 从一条队列里源源不断地吐消息。"""

    def __init__(self, inbox: asyncio.Queue[Any]) -> None:
        self._inbox = inbox
        self.channels: list[str] = []
        self.closed = False

    async def subscribe(self, channel: str) -> None:
        self.channels.append(channel)

    async def listen(self):
        while True:
            yield await self._inbox.get()

    async def aclose(self) -> None:
        self.closed = True


class FakeControlRedis:
    """控制面用到的那几个命令：set(ex) / get / delete / publish / pubsub。

    ``publish`` 直接把消息塞进本进程的订阅队列——单进程里模拟「广播出去又收回来」，正好能验
    origin 去重（自己发的必须被丢掉）。
    """

    def __init__(self) -> None:
        self.kv: dict[str, str] = {}
        self.ttls: dict[str, int] = {}
        self.published: list[tuple[str, str]] = []
        self.inbox: asyncio.Queue[Any] = asyncio.Queue()
        self.fail_publish = False

    async def set(self, key: str, value: str, ex: int | None = None) -> None:
        self.kv[key] = value
        if ex is not None:
            self.ttls[key] = ex

    async def get(self, key: str) -> str | None:
        return self.kv.get(key)

    async def delete(self, key: str) -> None:
        self.kv.pop(key, None)
        self.ttls.pop(key, None)

    async def publish(self, channel: str, data: str) -> None:
        if self.fail_publish:
            raise ConnectionError("redis down")
        self.published.append((channel, data))
        self.inbox.put_nowait({"type": "message", "channel": channel, "data": data})

    def pubsub(self) -> FakePubSub:
        return FakePubSub(self.inbox)

    async def aclose(self) -> None:
        pass


@pytest.fixture
def bus() -> Any:
    """装一条真 ControlBus（假客户端），并在用例后复位进程级单例与本地令牌表。"""
    client = FakeControlRedis()
    instance = control.ControlBus(client, channel="test:control", origin="origin-worker")
    control.set_control_bus(instance)
    yield instance
    control.reset_control_bus()
    clarification_module_state_clear()


def clarification_module_state_clear() -> None:
    from app.api import clarification as _c

    _c._local_tokens.clear()
    _c._pending.clear()


# ── 等待令牌 ─────────────────────────────────────────────────────────────────
async def test_register_waiter_writes_token_before_the_question_goes_out(bus: Any) -> None:
    """令牌落 Redis，且带齐 §8 要求的四个字段。"""
    set_turn_id("task-1")
    token = await register_waiter("t1", timeout_sec=60)
    raw = await bus.load("globex:clarify:t1")
    assert raw is not None
    saved = json.loads(raw)
    assert saved == {
        "thread_id": "t1",
        "turn_id": "task-1",
        "reply_id": token.reply_id,
        "expires": token.expires,
    }


async def test_token_ttl_follows_the_question_timeout(bus: Any) -> None:
    """TTL = 提问超时：令牌自己会过期，不依赖谁记得来删（worker 被 SIGKILL 时没人删得成）。"""
    await register_waiter("t1", timeout_sec=45)
    assert bus._client.ttls["globex:clarify:t1"] == 45


async def test_local_reply_short_circuits_before_touching_redis(bus: Any) -> None:
    """本进程有人等 → 就地 resolve，**不**广播。单进程模式走的就是这一条。"""
    fut = create_pending("t1")
    await register_waiter("t1", timeout_sec=60)
    assert await deliver_reply("t1", "女款") == "local"
    assert await fut == "女款"
    assert bus._client.published == []


async def test_reply_is_forwarded_when_the_waiter_is_in_another_process(bus: Any) -> None:
    """本进程没人等、但令牌在 → 广播出去，带上令牌里的 reply_id。"""
    token = await register_waiter("t1", timeout_sec=60)
    clarification_local_only_drop()  # 模拟「令牌是别的进程写的」
    assert await deliver_reply("t1", "要防水") == "forwarded"
    _, raw = bus._client.published[-1]
    sent = json.loads(raw)
    assert sent["kind"] == KIND_REPLY
    assert sent["reply_id"] == token.reply_id
    assert sent["text"] == "要防水"


def clarification_local_only_drop() -> None:
    """只清本进程的令牌副本，Redis 里那份留着——模拟「等待方在另一个进程」。"""
    from app.api import clarification as _c

    _c._local_tokens.clear()


# ── 令牌过期 = 按取消处理 ────────────────────────────────────────────────────
async def test_expired_token_rejects_the_late_reply(bus: Any) -> None:
    """键还在但时刻已过：以令牌里的 expires 为准拒收，并顺手把键删掉。

    「按取消处理」在本仓的落点就是**不投递**——等待方那边照旧走它的超时兜底（``ask_user`` 的
    except 分支）。把迟到的回复硬塞进去才是真的坏：模型不会报错，只会拿着答非所问的输入继续跑。
    """
    await register_waiter("t1", timeout_sec=60)
    clarification_local_only_drop()
    stale = json.loads(await bus.load("globex:clarify:t1"))
    stale["expires"] = time.time() - 1
    await bus.store("globex:clarify:t1", json.dumps(stale), 60)
    assert await deliver_reply("t1", "太晚了") == "no_waiter"
    assert await bus.load("globex:clarify:t1") is None


async def test_no_token_means_nobody_is_waiting(bus: Any) -> None:
    assert await deliver_reply("t1", "无人问津") == "no_waiter"
    assert bus._client.published == []


async def test_clear_waiter_revokes_the_token_synchronously(bus: Any) -> None:
    """撤销必须**当场**对本地生效：它是「这条回复算不算数」的判据。"""
    token = await register_waiter("t1", timeout_sec=60)
    clear_waiter("t1", token)
    from app.api import clarification as _c

    assert "t1" not in _c._local_tokens
    await _c._drain_cleanups()
    assert await bus.load("globex:clarify:t1") is None


async def test_clear_waiter_does_not_revoke_the_next_question(bus: Any) -> None:
    """拿着旧令牌来撤销时不能误伤下一问（同 thread 连问两次的形态）。"""
    old = await register_waiter("t1", timeout_sec=60)
    new = await register_waiter("t1", timeout_sec=60)
    clear_waiter("t1", old)
    from app.api import clarification as _c

    assert _c._local_tokens["t1"].reply_id == new.reply_id


async def test_publish_failure_is_reported_not_swallowed(bus: Any) -> None:
    """控制面发不出去要让调用方知道（端点据此回 503），不像事件背板那样静默。"""
    await register_waiter("t1", timeout_sec=60)
    clarification_local_only_drop()
    bus._client.fail_publish = True
    assert await deliver_reply("t1", "发不出去") == "publish_failed"


# ── worker 侧收件：reply_id 比对 ─────────────────────────────────────────────
async def test_worker_resolves_the_future_on_matching_reply(bus: Any) -> None:
    fut = create_pending("t1")
    token = await register_waiter("t1", timeout_sec=60)
    await on_reply_message({"thread_id": "t1", "reply_id": token.reply_id, "text": "要黑色"})
    assert await fut == "要黑色"


async def test_worker_drops_a_reply_belonging_to_the_previous_question(bus: Any) -> None:
    """上一问超时后用户才点回来 —— 这条回复必须丢掉。

    按 thread 盲投的写法在这条用例上会绿着通过并把「上一问的答案」交给这一问：不报错、日志正常、
    只有模型的输出莫名其妙。reply_id 就是为了这一条存在的。
    """
    old = await register_waiter("t1", timeout_sec=60)
    fut = create_pending("t1")
    await register_waiter("t1", timeout_sec=60)  # 换成第二问
    await on_reply_message({"thread_id": "t1", "reply_id": old.reply_id, "text": "上一问的答案"})
    assert not fut.done()


async def test_worker_ignores_malformed_reply_messages(bus: Any) -> None:
    await on_reply_message({"text": "缺字段"})
    await on_reply_message({"thread_id": "t1", "reply_id": "", "text": "缺 reply_id"})


# ── 残留令牌清理 ─────────────────────────────────────────────────────────────
async def test_stale_token_from_a_crashed_run_is_dropped(bus: Any) -> None:
    """上一轮被 SIGKILL 掐死，令牌留到 TTL 到点：新一轮领到任务时清掉它。

    不清的话，用户这一轮的回复会被转发给一个早已不存在的等待方——界面上表现为「答了没反应」。
    """
    set_turn_id("task-old")
    await register_waiter("t1", timeout_sec=60)
    await drop_stale_waiter("t1", turn_id="task-new")
    assert await bus.load("globex:clarify:t1") is None


async def test_waiter_of_the_same_turn_survives_a_redelivery(bus: Any) -> None:
    """同一轮被重投时 turn_id 相同，正当的等待不能误伤。"""
    set_turn_id("task-1")
    await register_waiter("t1", timeout_sec=60)
    await drop_stale_waiter("t1", turn_id="task-1")
    assert await bus.load("globex:clarify:t1") is not None


# ══════════════════════════════════════════════════════════════════════════════
# 跨进程取消
# ══════════════════════════════════════════════════════════════════════════════
async def _forever() -> None:
    await asyncio.sleep(3600)


async def test_cancel_local_hits_by_task_id_then_by_thread(bus: Any) -> None:
    """本进程就是执行方时（API 兼任 worker）根本不必走 Redis。"""
    task = asyncio.create_task(_forever())
    control.register_inflight("task-1", "t1", task)
    assert control.cancel_local(thread_id="t1") is True
    assert control.was_cancelled_locally("task-1")
    with pytest.raises(asyncio.CancelledError):
        await task
    control.unregister_inflight("task-1")
    assert control.cancel_local(task_id="task-1") is False


async def test_request_cancel_marks_then_broadcasts(bus: Any) -> None:
    """不在本进程 → 落标记（管排队中的）+ 广播（管在跑的）。少任何一半都有一类取消不掉。"""
    assert await control.request_cancel("t1", "task-9") is True
    assert await bus.was_cancelled("task-9") is True
    _, raw = bus._client.published[-1]
    assert json.loads(raw)["kind"] == "cancel"


async def test_request_cancel_nowait_is_synchronous_for_the_local_half(bus: Any) -> None:
    """覆盖重发那段不允许出现 await，所以本地那一半必须**当场**生效。"""
    task = asyncio.create_task(_forever())
    control.register_inflight("task-2", "t2", task)
    control.request_cancel_nowait("t2", "task-2")
    assert task.cancelling() > 0 or task.cancelled() or control.was_cancelled_locally("task-2")
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    control.unregister_inflight("task-2")


async def test_consume_cancel_mark_clears_it(bus: Any) -> None:
    await bus.mark_cancel("task-3")
    assert await control.consume_cancel_mark("task-3") is True
    assert await control.consume_cancel_mark("task-3") is False  # 只兑现一次


async def test_unreachable_redis_is_treated_as_not_cancelled(bus: Any) -> None:
    """读标记失败一律当「没取消」——凭一次读失败就把用户提交的任务扔掉是更坏的失败方向。"""

    class Broken(FakeControlRedis):
        async def get(self, key: str) -> str | None:
            raise ConnectionError("redis down")

    control.set_control_bus(control.ControlBus(Broken(), channel="c", origin="o"))
    assert await control.consume_cancel_mark("task-4") is False


# ── 控制面订阅循环 ───────────────────────────────────────────────────────────
async def _feed(bus_obj: Any, envelope: dict[str, Any] | str) -> None:
    """把一条消息塞进订阅队列并让循环跑一拍。"""
    data = envelope if isinstance(envelope, str) else json.dumps(envelope)
    bus_obj._client.inbox.put_nowait({"type": "message", "channel": "test:control", "data": data})
    await asyncio.sleep(0.02)


async def test_subscriber_dispatches_by_kind(bus: Any) -> None:
    seen: list[dict[str, Any]] = []

    async def _handler(payload: dict[str, Any]) -> None:
        seen.append(payload)

    bus.on("cancel", _handler)
    await bus.start()
    await _feed(bus, {"origin": "other", "kind": "cancel", "task_id": "x"})
    await bus.stop()
    assert [p["task_id"] for p in seen] == ["x"]


async def test_own_messages_are_skipped_by_origin(bus: Any) -> None:
    """自己发的指令不能自己再执行一遍——多副本 API 会把彼此的取消互相放大。"""
    seen: list[Any] = []
    bus.on("cancel", lambda p: asyncio.sleep(0, result=seen.append(p)))
    await bus.start()
    await bus.publish("cancel", task_id="x")  # 用的是 bus 自己的 origin
    await asyncio.sleep(0.02)
    await bus.stop()
    assert seen == []


async def test_bad_messages_do_not_kill_the_subscriber(bus: Any) -> None:
    """坏 JSON / 未知 kind / 处理器抛异常，三样都不能把订阅循环带走。

    带走了不会报错，只是这个副本从此**永久**收不到任何指令——取消按钮和澄清回复一起静默失效。
    """
    seen: list[Any] = []

    async def _boom(_payload: dict[str, Any]) -> None:
        raise RuntimeError("处理器炸了")

    async def _ok(payload: dict[str, Any]) -> None:
        seen.append(payload)

    bus.on("cancel", _boom)
    bus.on(KIND_REPLY, _ok)
    await bus.start()
    await _feed(bus, "not json at all")
    await _feed(bus, {"origin": "other", "kind": "unknown"})
    await _feed(bus, {"origin": "other", "kind": "cancel"})
    await _feed(bus, {"origin": "other", "kind": KIND_REPLY, "thread_id": "t1"})
    await bus.stop()
    assert len(seen) == 1
