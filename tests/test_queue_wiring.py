"""批 2 · API ↔ 队列 ↔ worker 的接线（`QUEUE_ENABLED=1` 那条路）。

分工上这里只管**接线**，队列自身的语义（双流分级 / ack / 重投 / 死信）在 ``tests/test_queue.py``。
两件事必须分别钉死，因为它们的失败形态完全不同：队列写错是任务跑错地方，接线写错是**契约悄悄变了**
——前端还在按老响应体渲染，用户看到的就是一个永远转圈的界面。

所以本文件的第一等公民是那条「前端零改动」的断言：队列模式下 ``POST /api/task`` 的响应体、
``active_tasks`` 的登记、``queue_status`` 事件三样与单进程模式**逐项同形**，只多一个加法字段
``task_id``（老前端会忽略它）。以及反向的那条：``QUEUE_ENABLED=0``（默认）时这套代码一个字节都不
执行，本仓既有的 ``tests/test_server.py`` 全绿即是它的回归。

用 ``InProcessQueue`` 而不是 FakeRedis：接线关心的是「谁调了谁、传了什么」，换成 Redis 只会把
Stream 协议的噪声引进来，真 Redis 的双进程冒烟另跑（见 docs/plans/批2-进度.md）。
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import AsyncIterator
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

import app.api.server as server
from app import worker
from app.api import control, dedup, monitor
from app.api.concurrency import PriorityRequestQueue
from app.queue import InProcessQueue, set_task_queue


@pytest.fixture
async def client() -> AsyncIterator[AsyncClient]:
    transport = ASGITransport(app=server.app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.fixture
def queue(monkeypatch: pytest.MonkeyPatch) -> InProcessQueue:
    """把工厂单例换成本用例专属的队列，并把 ``QUEUE_ENABLED`` 拨到开。

    ``server.queue_enabled`` 是 import 进 server 命名空间的名字，patch 它比改环境变量更准——环境
    变量还要考虑 ``.env`` 里已有的值与模块级常量的读取时机。
    """
    q = InProcessQueue()
    set_task_queue(q)
    monkeypatch.setattr(server, "queue_enabled", lambda: True)
    monkeypatch.setattr(server, "QUEUE_POLL_SECONDS", 0.02)  # 用例里不等整秒
    return q


@pytest.fixture(autouse=True)
async def _clean(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[None]:
    """惯例同 ``tests/test_server.py``：换新准入池 + 清指纹表，用例后取消遗留任务、复位队列单例。"""
    monkeypatch.setattr(server, "task_queue", PriorityRequestQueue())
    dedup.reset()
    yield
    for handle in list(server.active_tasks.values()):
        if not handle.task.done():
            handle.task.cancel()
    server.active_tasks.clear()
    dedup.reset()
    set_task_queue(None)


def _stub_agent(monkeypatch: pytest.MonkeyPatch, text: str = "选这三件") -> list[str]:
    """替换 worker 侧的 run_agent（真实 LLM 不进单测），返回被跑过的 thread_id 列表。"""
    ran: list[str] = []

    async def _fake(query: str, thread_id: str, **_kw: Any) -> dict[str, Any]:
        ran.append(thread_id)
        return {"final_text": text}

    monkeypatch.setattr(worker, "run_agent", _fake)
    return ran


async def _wait(cond: Any, limit_s: float = 3.0) -> None:
    """轮询等一个条件成立（同步或异步都收）——比固定 sleep 稳，超时即让用例红掉而不是挂住。

    队列模式下入队发生在影子协程里，HTTP 响应返回时它还没被调度过，所以断言前一律先等一等。
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + limit_s
    while loop.time() < deadline:
        result = cond()
        if inspect.isawaitable(result):
            result = await result
        if result:
            return
        await asyncio.sleep(0.01)
    raise AssertionError("等待超时")


# ---------- POST /api/task：契约不变 ----------
async def test_queue_mode_keeps_response_contract(
    client: AsyncClient, queue: InProcessQueue, monkeypatch: pytest.MonkeyPatch
) -> None:
    """队列模式下响应体与单进程模式同形（多一个加法字段 task_id），且 active_tasks 照常登记。"""
    _stub_agent(monkeypatch)
    resp = await client.post("/api/task", json={"query": "买帐篷", "thread_id": "q-a"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "started"  # 队列是空的 → 排第 1 位，语义等同「直接占到槽」
    assert body["thread_id"] == "q-a"
    assert body["queue_position"] == 0
    assert body["task_id"]  # 加法字段：老前端忽略，脚本可拿去 GET /api/task/{id}
    await _wait(queue.depth)
    assert await queue.depth() == 1  # 任务真的进了队列，而不是在 API 进程里跑起来了
    assert "q-a" in server.active_tasks  # /inflight、取消口、幂等第 1 层的账本照常


async def test_queue_mode_runs_via_worker_and_clears_handle(
    client: AsyncClient, queue: InProcessQueue, monkeypatch: pytest.MonkeyPatch
) -> None:
    """端到端：API 入队 → worker 消费 → 状态 done → API 侧的影子协程自己退休。"""
    ran = _stub_agent(monkeypatch)
    stop = asyncio.Event()
    runner = asyncio.create_task(
        worker.run_worker(queue, concurrency=2, stop=stop, install_signals=False)
    )
    resp = await client.post("/api/task", json={"query": "买帐篷", "thread_id": "q-b"})
    task_id = resp.json()["task_id"]

    await _wait(lambda: ran == ["q-b"])
    status = await queue.get_status(task_id)
    assert status is not None and status.state == "done" and status.final_text == "选这三件"
    await _wait(lambda: "q-b" not in server.active_tasks)

    stop.set()
    await asyncio.wait_for(runner, 3.0)


async def test_queue_mode_reports_position_and_wait(
    client: AsyncClient, queue: InProcessQueue, monkeypatch: pytest.MonkeyPatch
) -> None:
    """前面有人排队时推 queue_status，位置与预估等待都带上（预估复用 concurrency 那份公式）。"""
    _stub_agent(monkeypatch)
    seen: list[tuple[str, int, int, str]] = []

    async def _spy(thread_id: str, position: int, wait_s: int, kind: str) -> None:
        seen.append((thread_id, position, wait_s, kind))

    monkeypatch.setattr(monitor, "report_queue_status", _spy)
    for i in range(2):  # 队列里先躺两条，谁也不消费
        await queue.enqueue(
            server.IntentTask.create(task_id=f"x{i}", thread_id=f"x{i}", query="占位")
        )

    resp = await client.post("/api/task", json={"query": "买帐篷", "thread_id": "q-c"})
    assert resp.json() == {
        "status": "queued",
        "thread_id": "q-c",
        "queue_position": 3,
        "task_id": resp.json()["task_id"],
    }
    await _wait(lambda: bool(seen))
    thread_id, position, wait_s, kind = seen[0]
    assert (thread_id, position, kind) == ("q-c", 3, "normal")
    assert wait_s > 0


async def test_queue_mode_429_when_backlog_over_cap(
    client: AsyncClient, queue: InProcessQueue, monkeypatch: pytest.MonkeyPatch
) -> None:
    """队列深度是队列模式下唯一的背压闸（准入池被跳过了），超上限照样 429 + Retry-After。"""
    _stub_agent(monkeypatch)
    monkeypatch.setattr(server, "QUEUE_MAX_DEPTH", 1)
    await queue.enqueue(server.IntentTask.create(task_id="x", thread_id="x", query="占位"))

    resp = await client.post("/api/task", json={"query": "买帐篷", "thread_id": "q-d"})
    assert resp.status_code == 429
    assert resp.headers["Retry-After"]
    assert "q-d" not in server.active_tasks  # 被拒的请求不该在系统里留足迹


async def test_enqueue_failure_surfaces_as_error_event(
    client: AsyncClient, queue: InProcessQueue, monkeypatch: pytest.MonkeyPatch
) -> None:
    """入队失败必须让用户看见：响应早发出去了，只能补一条 error 事件（前端本就按它收尾）。"""
    _stub_agent(monkeypatch)
    seen: list[tuple[str, str | None]] = []

    async def _boom(_task: Any) -> None:
        raise RuntimeError("redis 挂了")

    async def _spy(error_type: str, message: str, thread_id: str | None = None) -> None:
        seen.append((error_type, thread_id))

    monkeypatch.setattr(queue, "enqueue", _boom)
    monkeypatch.setattr(monitor, "report_error", _spy)

    resp = await client.post("/api/task", json={"query": "买帐篷", "thread_id": "q-e"})
    assert resp.status_code == 200  # 契约不变：起任务这一步照样立刻返回
    await _wait(lambda: bool(seen))
    assert seen[0] == ("enqueue_failed", "q-e")  # thread_id 必须显式带上，否则前端收不到
    await _wait(lambda: "q-e" not in server.active_tasks)


# ---------- POST /api/task/async + GET /api/task/{task_id} ----------
async def test_task_async_requires_queue_enabled(client: AsyncClient) -> None:
    """关着队列时 503 而不是退回本进程跑：那样会返回一个永远停在 queued 的 task_id，是骗人。"""
    resp = await client.post("/api/task/async", json={"query": "买帐篷"})
    assert resp.status_code == 503


async def test_task_async_returns_id_then_state_endpoint_tracks_it(
    client: AsyncClient, queue: InProcessQueue, monkeypatch: pytest.MonkeyPatch
) -> None:
    """异步提交：立即拿 task_id → GET 能查到 queued（且带预估等待）→ worker 跑完变 done。"""
    _stub_agent(monkeypatch, text="给你选了三件")
    resp = await client.post("/api/task/async", json={"query": "买帐篷", "thread_id": "q-f"})
    assert resp.status_code == 200
    body = resp.json()
    task_id = body["task_id"]
    assert (body["thread_id"], body["status"], body["queue_position"]) == ("q-f", "queued", 1)
    assert body["estimated_wait_seconds"] == 0  # 前面没人挡着，别凭空吓人说要等 30s
    # 异步口不在 API 侧留影子协程——没有 WS 要伺候的调用方不该占 active_tasks
    assert "q-f" not in server.active_tasks

    queued = await client.get(f"/api/task/{task_id}")
    assert queued.status_code == 200
    assert queued.json()["state"] == "queued"

    stop = asyncio.Event()
    runner = asyncio.create_task(
        worker.run_worker(queue, concurrency=1, stop=stop, install_signals=False)
    )
    try:
        await _wait(lambda: _state_is(client, task_id, "done"))
    finally:
        stop.set()
        await asyncio.wait_for(runner, 3.0)
    final = (await client.get(f"/api/task/{task_id}")).json()
    assert final["final_text"] == "给你选了三件"
    assert final["estimated_wait_seconds"] == 0  # 跑起来之后这个数对用户没意义


async def _state_is(client: AsyncClient, task_id: str, state: str) -> bool:
    resp = await client.get(f"/api/task/{task_id}")
    return resp.status_code == 200 and resp.json()["state"] == state


async def test_task_state_404_when_unknown(client: AsyncClient, queue: InProcessQueue) -> None:
    """状态键有 TTL，过期即 404：它是给一次提交轮询几分钟用的，不是历史存储。"""
    assert (await client.get("/api/task/nope")).status_code == 404


# ---------- QUEUE_ENABLED=0（默认）：这套代码一个字节都不执行 ----------
async def test_default_mode_does_not_touch_the_queue(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """默认路径仍在 API 进程里跑 run_agent、仍占准入槽，队列深度纹丝不动。"""
    q = InProcessQueue()
    set_task_queue(q)
    started = asyncio.Event()

    async def _fake_run(query: str, thread_id: str, **_kw: Any) -> dict[str, Any]:
        started.set()
        return {"thread_id": thread_id}

    monkeypatch.setattr(server, "run_agent", _fake_run)  # 注意是 server 侧那个名字，不是 worker 的
    resp = await client.post("/api/task", json={"query": "买帐篷", "thread_id": "q-g"})

    assert resp.json()["status"] == "started"
    assert "task_id" not in resp.json()  # 默认路径的响应体逐字节不变
    await asyncio.wait_for(started.wait(), 2.0)
    assert await q.depth() == 0


# ---------- 跨进程取消 / 澄清（批2-4）----------
#
# 这里只钉**接线**：取消口有没有把指令送出去、覆盖重发有没有从「多跑一轮」变成「换一轮」、
# handle_task 有没有把两种 CancelledError 分开。控制面自己的语义（令牌 / origin / 重连）在
# tests/test_clarification.py。


class _SpyBus:
    """只记录调用的控制面替身：接线测试关心「谁调了谁、传了什么」，不关心 Redis 协议。"""

    def __init__(self) -> None:
        self.marked: list[str] = []
        self.published: list[dict[str, Any]] = []

    async def mark_cancel(self, task_id: str) -> bool:
        self.marked.append(task_id)
        return True

    async def publish(self, kind: str, **fields: Any) -> bool:
        self.published.append({"kind": kind, **fields})
        return True

    async def was_cancelled(self, task_id: str) -> bool:
        return task_id in self.marked

    async def clear_cancel(self, task_id: str) -> None:
        while task_id in self.marked:
            self.marked.remove(task_id)

    async def load(self, _key: str) -> str | None:
        return None  # 澄清令牌不在本文件的射程内（见 tests/test_clarification.py）

    async def store(self, _key: str, _value: str, _ttl: int) -> bool:
        return True

    async def drop(self, _key: str) -> None:
        return None


@pytest.fixture
def spy_bus() -> Any:
    bus = _SpyBus()
    control.set_control_bus(bus)
    yield bus
    control.reset_control_bus()


async def test_cancel_endpoint_sends_the_cancel_across_processes(
    client: AsyncClient, queue: InProcessQueue, monkeypatch: pytest.MonkeyPatch, spy_bus: Any
) -> None:
    """队列模式下取消口要把指令送到 worker：标记（管排队中的）+ 广播（管在跑的）各一份。"""
    _stub_agent(monkeypatch)
    posted = await client.post("/api/task", json={"query": "买帐篷", "thread_id": "c-a"})
    task_id = posted.json()["task_id"]
    resp = await client.post("/api/task/c-a/cancel")

    assert resp.status_code == 200
    assert spy_bus.marked == [task_id]  # 按 task_id 打标记，不是按 thread（覆盖重发会误伤）
    assert spy_bus.published[-1] == {"kind": "cancel", "thread_id": "c-a", "task_id": task_id}


async def test_replace_cancels_the_old_task_in_the_worker(
    client: AsyncClient, queue: InProcessQueue, monkeypatch: pytest.MonkeyPatch, spy_bus: Any
) -> None:
    """覆盖重发 = 换一轮，不是多跑一轮：旧那条要在 worker 侧真的被掐掉。

    批2-2 报告里标注的缺口就是这条。少了它，用户改主意重问一句，旧问题仍在后台烧 token，
    两轮的事件还会同时往同一条 WS 上推。
    """
    _stub_agent(monkeypatch)
    r1 = await client.post("/api/task", json={"query": "买帐篷", "thread_id": "c-b"})
    r2 = await client.post("/api/task", json={"query": "改买睡袋", "thread_id": "c-b"})
    first, second = r1.json()["task_id"], r2.json()["task_id"]

    assert second != first
    await _wait(lambda: bool(spy_bus.published))
    assert spy_bus.published[-1]["task_id"] == first  # 掐的是**旧**那条
    assert spy_bus.marked == [first]  # 新任务的 task_id 绝不能被打上取消标记


def _slow_agent(monkeypatch: pytest.MonkeyPatch, started: asyncio.Event) -> None:
    """一个跑得足够久、能被中途掐掉的 run_agent 替身。"""

    async def _fake(query: str, thread_id: str, **_kw: Any) -> dict[str, Any]:
        started.set()
        await asyncio.sleep(30)
        return {"final_text": "不该跑到这里"}

    monkeypatch.setattr(worker, "run_agent", _fake)


async def test_task_cancelled_while_queued_is_skipped_on_pickup(
    queue: InProcessQueue, monkeypatch: pytest.MonkeyPatch, spy_bus: Any
) -> None:
    """在队列里排着时被取消：worker 领到手先查标记，一步都不跑。

    广播只送得到「已经领走它的那个 worker」，还没被领走的任务只能靠这张标记——而那恰恰是最该
    取消的一批（用户还没等到它开始就反悔了）。
    """
    ran = _stub_agent(monkeypatch)
    task = server.IntentTask.create(task_id="ct-1", thread_id="ct-1", query="买帐篷")
    await spy_bus.mark_cancel("ct-1")

    await worker.handle_task(task, queue)

    assert ran == []
    status = await queue.get_status("ct-1")
    assert status is not None and status.state == "cancelled"
    assert spy_bus.marked == []  # 标记只兑现一次，别留成垃圾


async def test_user_cancel_acks_the_message_instead_of_requeueing_it(
    queue: InProcessQueue, monkeypatch: pytest.MonkeyPatch, spy_bus: Any
) -> None:
    """用户取消 → handle_task **不往外抛**（于是消息被 ack、不会重投），状态落 cancelled。

    这条是本单最容易写错的地方：用户取消与优雅退出都表现为 CancelledError，可语义相反。往外抛
    会让消息留在 PEL 里被下一个 worker 领回来**重跑一遍**——用户按了取消，任务却又活了过来。
    """
    started = asyncio.Event()
    _slow_agent(monkeypatch, started)
    intent = server.IntentTask.create(task_id="ct-2", thread_id="ct-2", query="买帐篷")
    running = asyncio.create_task(worker.handle_task(intent, queue))
    await asyncio.wait_for(started.wait(), 2.0)

    assert control.cancel_local(task_id="ct-2") is True
    await asyncio.wait_for(running, 2.0)  # 正常返回，没有 CancelledError 漏出去

    status = await queue.get_status("ct-2")
    assert status is not None and status.state == "cancelled"


async def test_graceful_shutdown_cancel_still_returns_the_message_to_pending(
    queue: InProcessQueue, monkeypatch: pytest.MonkeyPatch, spy_bus: Any
) -> None:
    """优雅退出的取消照旧往外抛：消息不 ack，留在 PEL 里等下一个 worker 领回重跑。

    与上一条是同一枚硬币的两面。判据是控制面的进程内标记——没有它就只能二选一，两种取消里
    必然有一种是错的。
    """
    started = asyncio.Event()
    _slow_agent(monkeypatch, started)
    intent = server.IntentTask.create(task_id="ct-3", thread_id="ct-3", query="买帐篷")
    running = asyncio.create_task(worker.handle_task(intent, queue))
    await asyncio.wait_for(started.wait(), 2.0)

    running.cancel()  # 没打过取消标记 = 不是用户取消
    with pytest.raises(asyncio.CancelledError):
        await running

    status = await queue.get_status("ct-3")
    assert status is not None and status.state == "running"  # 不写终态：它待会儿还会活过来


# ---------- POST /api/clarify/{thread_id}：没有 WS 的调用方也能回答提问 ----------
async def test_clarify_endpoint_delivers_to_a_local_waiter(client: AsyncClient) -> None:
    """本进程就有人等 → 就地 resolve。前端不用这个口子（它走 WS），脚本 / 冒烟用它。"""
    from app.api.clarification import create_pending

    fut = create_pending("cl-a")
    resp = await client.post("/api/clarify/cl-a", json={"text": "要女款"})

    assert resp.status_code == 200
    assert resp.json()["route"] == "local"
    assert await asyncio.wait_for(fut, 1.0) == "要女款"


async def test_clarify_endpoint_404_when_nobody_is_waiting(client: AsyncClient) -> None:
    """没人在等（从没提问 / 已超时 / 令牌过期）→ 404，绝不把迟到的回复塞给下一个问题。"""
    resp = await client.post("/api/clarify/cl-b", json={"text": "无人问津"})
    assert resp.status_code == 404
