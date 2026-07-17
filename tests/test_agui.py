"""M8 验收：AGUI 事件协议 + WebSocket 路由的确定性测试（不依赖真实 LLM / 真 WS 服务）。

覆盖 ROADMAP M8 验收点：
- ConnectionManager 按 thread_id 正确路由，未连接 thread 推送是 no-op
- 重连场景按对象身份注销，不误删新连接
- 推送失败的死连接被自动摘除
- monitor.report_* 产出统一结构事件、按序经连接推出
- dispatch fork 时向【父】thread 上报 fork 事件
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from app.agent.dispatch_tool import _run_sub_agent
from app.api import monitor
from app.api.connection import ConnectionManager
from app.utils.thread_ctx import thread_scope


class FakeWebSocket:
    """记录所有 accept/send_json 调用的假连接；可选模拟 send 抛错（死连接）。"""

    def __init__(self, *, fail_send: bool = False) -> None:
        self.accepted = False
        self.sent: list[dict[str, Any]] = []
        self._fail_send = fail_send

    async def accept(self) -> None:
        self.accepted = True

    async def send_json(self, data: Any) -> None:
        if self._fail_send:
            raise ConnectionError("socket closed")
        self.sent.append(data)


# ---------- ConnectionManager 路由 ----------
async def test_connect_accepts_and_routes() -> None:
    mgr = ConnectionManager()
    ws = FakeWebSocket()
    await mgr.connect(ws, "t1")
    assert ws.accepted
    assert mgr.is_connected("t1")
    assert mgr.connection_count == 1

    ok = await mgr.send_to_thread("t1", {"hello": "world"})
    assert ok
    assert ws.sent == [{"hello": "world"}]


async def test_send_to_unknown_thread_is_noop() -> None:
    mgr = ConnectionManager()
    assert await mgr.send_to_thread("nope", {"x": 1}) is False


async def test_routing_isolated_per_thread() -> None:
    mgr = ConnectionManager()
    a, b = FakeWebSocket(), FakeWebSocket()
    await mgr.connect(a, "ta")
    await mgr.connect(b, "tb")
    await mgr.send_to_thread("ta", {"to": "a"})
    assert a.sent == [{"to": "a"}]
    assert b.sent == []  # 不串台


# ---------- 重连身份校验 ----------
async def test_disconnect_only_removes_same_object() -> None:
    mgr = ConnectionManager()
    old, new = FakeWebSocket(), FakeWebSocket()
    await mgr.connect(old, "t1")
    # 用户刷新页面：新连接登记覆盖旧连接。
    await mgr.connect(new, "t1")
    # 旧连接稍后才触发断开——必须按对象身份判断，不能误删刚建好的新连接。
    await mgr.disconnect(old, "t1")
    assert mgr.is_connected("t1")
    assert mgr.send_to_thread is not None
    await mgr.send_to_thread("t1", {"ping": 1})
    assert new.sent == [{"ping": 1}]
    assert old.sent == []


async def test_disconnect_same_object_removes() -> None:
    mgr = ConnectionManager()
    ws = FakeWebSocket()
    await mgr.connect(ws, "t1")
    await mgr.disconnect(ws, "t1")
    assert not mgr.is_connected("t1")


# ---------- 死连接自动摘除 ----------
async def test_failed_send_drops_connection() -> None:
    mgr = ConnectionManager()
    ws = FakeWebSocket(fail_send=True)
    await mgr.connect(ws, "t1")
    ok = await mgr.send_to_thread("t1", {"x": 1})
    assert ok is False
    assert not mgr.is_connected("t1")  # 死连接被摘掉


# ---------- monitor.report_* 结构与路由 ----------
async def test_monitor_emits_structured_event_in_order() -> None:
    mgr = ConnectionManager()
    monitor.set_connection_manager(mgr)
    ws = FakeWebSocket()
    await mgr.connect(ws, "task-1")
    try:
        with thread_scope("task-1", Path("/tmp/task-1")):
            await monitor.report_session_created()
            await monitor.report_tool_start("planner", intent="买帐篷")
            await monitor.report_tool_end("planner", category="tent")
            await monitor.report_task_result("这是最终清单")
    finally:
        monitor.set_connection_manager(ConnectionManager())  # 还原全局，避免污染其他测试

    events = [m["event"] for m in ws.sent]
    assert events == ["session_created", "tool_start", "tool_end", "task_result"]
    # 统一结构校验。
    first = ws.sent[0]
    assert first["type"] == "monitor_event"
    assert first["thread_id"] == "task-1"
    assert "timestamp" in first and "data" in first and "message" in first
    # tool_start 的 data 带 tool 名与入参。
    ts = ws.sent[1]
    assert ts["data"] == {"tool": "planner", "intent": "买帐篷"}


# ---------- 活动流录制：供历史回看还原「思考过程」 ----------
async def test_activity_capture_records_thought_events_only() -> None:
    # begin_activity_capture 攒思考过程事件（session_created / assistant_call / tool_* / fork），
    # 但**不**收终结类（task_result）——那是状态信号、商品卡走 items 另存，回看不画思考行。
    mgr = ConnectionManager()
    monitor.set_connection_manager(mgr)
    try:
        with thread_scope("cap-1", Path("/tmp/cap-1")):
            rec = monitor.begin_activity_capture()
            await monitor.report_session_created()
            await monitor.report_assistant_call(preview="想想看")
            await monitor.report_tool_start("item_search", query="帐篷")
            await monitor.report_tool_end("item_search", count=3)
            await monitor.report_task_result("最终清单")  # 不该进活动流
    finally:
        monitor.set_connection_manager(ConnectionManager())

    captured = [e["event"] for e in rec.events]
    assert captured == ["session_created", "assistant_call", "tool_start", "tool_end"]
    # 录的是完整事件信封，前端可原样喂回 ActivityFeed。
    assert rec.events[2]["data"] == {"tool": "item_search", "query": "帐篷"}


async def test_activity_capture_ignores_foreign_thread_events() -> None:
    # 只录与录制起点同一 thread 的事件：子 thread（fork 内部）的事件被过滤，
    # 与实时前端「父任务页只显示父 thread 事件」一致。
    try:
        with thread_scope("root", Path("/tmp/root")):
            rec = monitor.begin_activity_capture()
            await monitor.report_tool_start("planner")  # 父 thread → 收
            with thread_scope("child", Path("/tmp/root")):
                await monitor.report_tool_start("item_search")  # 子 thread → 弃
            await monitor.report_tool_end("planner")  # 父 thread → 收
    finally:
        monitor.set_connection_manager(ConnectionManager())

    assert [e["data"]["tool"] for e in rec.events] == ["planner", "planner"]


async def test_monitor_noop_without_context() -> None:
    # 无 thread 上下文（离线脚本）：上报不抛、不推送。
    mgr = ConnectionManager()
    monitor.set_connection_manager(mgr)
    try:
        await monitor.report_tool_start("planner", intent="x")  # 不应抛
        assert mgr.connection_count == 0
    finally:
        monitor.set_connection_manager(ConnectionManager())


async def test_long_text_is_clipped() -> None:
    mgr = ConnectionManager()
    monitor.set_connection_manager(mgr)
    ws = FakeWebSocket()
    await mgr.connect(ws, "t1")
    try:
        with thread_scope("t1", Path("/tmp/t1")):
            await monitor.report_task_result("x" * 5000)
    finally:
        monitor.set_connection_manager(ConnectionManager())
    answer = ws.sent[0]["data"]["final_answer"]
    assert answer.endswith("…[truncated]")
    assert len(answer) < 5000


# ---------- fork 事件路由到父 thread ----------
async def test_fork_event_reported_to_parent_thread() -> None:
    mgr = ConnectionManager()
    monitor.set_connection_manager(mgr)
    parent_ws = FakeWebSocket()
    await mgr.connect(parent_ws, "parent")

    # provider 返回空工具集会让 create_agent 在子 Agent 里很快失败——无所谓，
    # 我们只验证「fork 事件在子任务真正跑起来之前已发给父 thread」。
    def empty_provider() -> list[Any]:
        return []

    try:
        with thread_scope("parent", Path("/tmp/parent")):
            result = await _run_sub_agent("去 amazon 搜帐篷", empty_provider, "you are a test")
    finally:
        monitor.set_connection_manager(ConnectionManager())

    # 子任务失败与否都该返回字符串（dispatch 容错），不抛。
    assert isinstance(result, str)
    fork_events = [m for m in parent_ws.sent if m["event"] == "fork"]
    assert len(fork_events) == 1
    data = fork_events[0]["data"]
    assert data["demands"] == "去 amazon 搜帐篷"
    assert data["sub_thread_id"].startswith("sub-")
    # 事件落在父 thread 的连接上（不是子 thread）。
    assert fork_events[0]["thread_id"] == "parent"


@pytest.mark.parametrize(
    "reporter",
    [
        lambda: monitor.report_assistant_call(preview="想想看"),
        lambda: monitor.report_task_cancelled(),
        lambda: monitor.report_error("ValueError", "崩了"),
    ],
)
async def test_all_reporters_route(reporter: Any) -> None:
    mgr = ConnectionManager()
    monitor.set_connection_manager(mgr)
    ws = FakeWebSocket()
    await mgr.connect(ws, "t1")
    try:
        with thread_scope("t1", Path("/tmp/t1")):
            await reporter()
    finally:
        monitor.set_connection_manager(ConnectionManager())
    assert len(ws.sent) == 1
    assert ws.sent[0]["type"] == "monitor_event"
