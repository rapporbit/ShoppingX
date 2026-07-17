"""M10 验收：FastAPI 接口的确定性测试（不依赖真实 LLM / 真 uvicorn 服务）。

覆盖 ROADMAP M10 的「工具 / 连接管理 / 取消任务」接口面：
- ``POST /api/task`` 起后台任务、登记 active_tasks、客户端可指定 thread_id。
- ``POST /api/task/{tid}/cancel`` 取消运行中任务 / 不存在任务 404。
- ``GET /api/files/...`` 下载产物、缺文件 404、路径穿越 400。
- ``POST /api/upload`` 落盘、超大 413、穿越文件名 400。
- ``GET /api/preferences/{uid}`` 读长期偏好、按置信度排序。

真 WebSocket 事件流端到端（connect-first、收 ws_ready 再发任务）见
``examples/10_server_e2e.py``——那条起真 uvicorn，不进单测（httpx ASGITransport 不跑 WS）。
连接管理本身的路由 / 重连 / 死连接摘除在 ``tests/test_agui.py`` 已覆盖。
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

import app.api.server as server
from app.api import dedup
from app.api.concurrency import PriorityRequestQueue
from app.memory.parser import UserPrefDraft
from app.memory.store import PreferenceEntry, get_store


@pytest.fixture
async def client() -> AsyncIterator[AsyncClient]:
    """ASGI 直连后端 app 的 httpx 客户端（不开真实端口）。"""
    transport = ASGITransport(app=server.app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.fixture(autouse=True)
async def _clean_tasks(monkeypatch: Any) -> AsyncIterator[None]:
    """每个用例前换一个全新的任务队列 + 清空指纹表，用例后清 active_tasks 并取消遗留任务。

    队列与指纹表都是模块单例：被取消任务的 ``release`` 是异步的、可能晚于用例结束，不换新的会让
    槽位计数泄漏到下一个用例（看似没满其实占着）；指纹表不清会让下个用例的同名 query 被判重复。
    """
    monkeypatch.setattr(server, "task_queue", PriorityRequestQueue())
    dedup.reset()
    yield
    for handle in list(server.active_tasks.values()):
        if not handle.task.done():
            handle.task.cancel()
    server.active_tasks.clear()
    dedup.reset()


# ---------- POST /api/task ----------
async def test_create_task_registers_and_returns_thread_id(
    client: AsyncClient, monkeypatch: Any
) -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    async def _fake_run(
        query: str, thread_id: str, user_id: str | None = None, **_kw: Any
    ) -> dict[str, Any]:
        started.set()
        await release.wait()  # 卡住任务，让它在 active_tasks 里可被观察 / 取消
        return {"thread_id": thread_id}

    monkeypatch.setattr(server, "run_agent", _fake_run)

    resp = await client.post("/api/task", json={"query": "买帐篷", "thread_id": "t-a"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "started"
    assert body["thread_id"] == "t-a"  # 客户端指定的 thread_id 被沿用（connect-first）

    await asyncio.wait_for(started.wait(), timeout=1.0)
    assert "t-a" in server.active_tasks
    release.set()


async def test_create_task_generates_thread_id_when_absent(
    client: AsyncClient, monkeypatch: Any
) -> None:
    async def _fake_run(
        query: str, thread_id: str, user_id: str | None = None, **_kw: Any
    ) -> dict[str, Any]:
        return {"thread_id": thread_id}

    monkeypatch.setattr(server, "run_agent", _fake_run)
    resp = await client.post("/api/task", json={"query": "买帐篷"})
    assert resp.status_code == 200
    assert resp.json()["thread_id"]  # 服务端兜底生成非空 id


# ---------- cancel ----------
async def test_cancel_running_task(client: AsyncClient, monkeypatch: Any) -> None:
    started = asyncio.Event()

    async def _slow_run(query: str, thread_id: str, user_id: str | None = None, **_kw: Any) -> None:
        started.set()
        await asyncio.sleep(30)  # 长任务，等被取消

    monkeypatch.setattr(server, "run_agent", _slow_run)
    await client.post("/api/task", json={"query": "q", "thread_id": "t-cancel"})
    await asyncio.wait_for(started.wait(), timeout=1.0)

    resp = await client.post("/api/task/t-cancel/cancel")
    assert resp.status_code == 200
    assert resp.json()["status"] == "cancelling"
    # 取消后任务最终从表里摘除。
    for _ in range(50):
        if "t-cancel" not in server.active_tasks:
            break
        await asyncio.sleep(0.01)
    assert "t-cancel" not in server.active_tasks


async def test_cancel_unknown_task_404(client: AsyncClient) -> None:
    resp = await client.post("/api/task/nope/cancel")
    assert resp.status_code == 404


# ---------- inflight（刷新 / 切回对话自动续看的后端支点）----------
async def test_inflight_not_running(client: AsyncClient) -> None:
    """没有活跃任务的 thread：running=false、不回吐 query / events。"""
    resp = await client.get("/api/task/nope/inflight")
    assert resp.status_code == 200
    assert resp.json() == {"running": False, "query": None, "images": [], "events": []}


async def test_inflight_running_returns_query_and_events(
    client: AsyncClient, monkeypatch: Any
) -> None:
    """任务在跑：回吐发起的 query 原文 + 「当前这轮」事件，供前端重建在跑轮 + 续直播。"""
    started = asyncio.Event()
    release = asyncio.Event()

    async def _fake_run(query: str, thread_id: str, user_id: str | None = None, **_kw: Any) -> None:
        started.set()
        await release.wait()

    events = [{"type": "monitor_event", "event": "session_created", "id": "1-0"}]

    async def _fake_replay(thread_id: str) -> list[dict[str, Any]]:
        return events

    monkeypatch.setattr(server, "run_agent", _fake_run)
    monkeypatch.setattr(server.event_log, "replay_current_run", _fake_replay)

    await client.post("/api/task", json={"query": "买帐篷", "thread_id": "t-live"})
    await asyncio.wait_for(started.wait(), timeout=1.0)

    resp = await client.get("/api/task/t-live/inflight")
    assert resp.status_code == 200
    body = resp.json()
    assert body["running"] is True
    assert body["query"] == "买帐篷"  # 后端权威 query（前端刷新后本地已无此上下文）
    assert body["events"] == events
    release.set()


async def test_inflight_terminal_event_treated_as_ended(
    client: AsyncClient, monkeypatch: Any
) -> None:
    """task_result 已进流、但 _runner 还没把任务摘出 active_tasks 的瞬时窗口：以「流末为终结类」
    判其已结束，避免与刚落盘的历史轮重出一份在跑轮。"""
    started = asyncio.Event()
    release = asyncio.Event()

    async def _fake_run(query: str, thread_id: str, user_id: str | None = None, **_kw: Any) -> None:
        started.set()
        await release.wait()

    async def _fake_replay(thread_id: str) -> list[dict[str, Any]]:
        return [
            {"type": "monitor_event", "event": "session_created", "id": "1-0"},
            {"type": "monitor_event", "event": "task_result", "id": "2-0"},
        ]

    monkeypatch.setattr(server, "run_agent", _fake_run)
    monkeypatch.setattr(server.event_log, "replay_current_run", _fake_replay)

    await client.post("/api/task", json={"query": "q", "thread_id": "t-ended"})
    await asyncio.wait_for(started.wait(), timeout=1.0)

    resp = await client.get("/api/task/t-ended/inflight")
    assert resp.status_code == 200
    assert resp.json() == {"running": False, "query": None, "images": [], "events": []}
    release.set()


# ---------- 并发背压 + 优先级队列（refdocs 16-5 §2）----------
def _blocking_run(started: asyncio.Event, release: asyncio.Event) -> Any:
    async def _run(
        query: str, thread_id: str, user_id: str | None = None, **_kw: Any
    ) -> dict[str, Any]:
        started.set()
        await release.wait()  # 卡住任务，占着槽不放
        return {"thread_id": thread_id}

    return _run


async def test_create_task_queues_when_slots_full(client: AsyncClient, monkeypatch: Any) -> None:
    """槽满不再直接 429——先进有界队列，回 ``queued`` + 排队位置。"""
    monkeypatch.setattr(server, "task_queue", PriorityRequestQueue(normal_slots=1, heavy_slots=1))
    started, release = asyncio.Event(), asyncio.Event()
    monkeypatch.setattr(server, "run_agent", _blocking_run(started, release))

    r1 = await client.post("/api/task", json={"query": "q1", "thread_id": "t1"})
    assert r1.json()["status"] == "started"
    await asyncio.wait_for(started.wait(), timeout=1.0)

    r2 = await client.post("/api/task", json={"query": "q2", "thread_id": "t2"})
    assert r2.status_code == 200
    assert r2.json()["status"] == "queued"
    assert r2.json()["queue_position"] == 1
    assert "t2" in server.active_tasks  # 排队中的任务也登记，可被取消
    release.set()


async def test_create_task_429_when_queue_full(client: AsyncClient, monkeypatch: Any) -> None:
    """队列也满才 429——有界排队守住背压，不堆成「看似没满其实全在等」。"""
    monkeypatch.setattr(
        server, "task_queue", PriorityRequestQueue(normal_slots=1, heavy_slots=1, queue_depth=0)
    )
    started, release = asyncio.Event(), asyncio.Event()
    monkeypatch.setattr(server, "run_agent", _blocking_run(started, release))

    r1 = await client.post("/api/task", json={"query": "q1", "thread_id": "t1"})
    assert r1.status_code == 200
    await asyncio.wait_for(started.wait(), timeout=1.0)

    r2 = await client.post("/api/task", json={"query": "q2", "thread_id": "t2"})
    assert r2.status_code == 429
    assert r2.headers["Retry-After"] == str(server.TASK_RETRY_AFTER_SEC)
    assert "t2" not in server.active_tasks  # 被拒任务不登记
    release.set()


async def test_heavy_task_does_not_block_normal_task(
    client: AsyncClient, monkeypatch: Any, tmp_path: Path
) -> None:
    """分池的全部意义：长续聊占满 heavy 槽，短对话照样直接开跑。"""
    monkeypatch.setattr(
        server, "task_queue", PriorityRequestQueue(normal_slots=1, heavy_slots=1, queue_depth=0)
    )
    monkeypatch.setattr(server, "OUTPUT_ROOT", tmp_path)

    # 让 heavy_thread 被判为 heavy：伪造一段很长的历史。
    async def _turns(tid: str) -> int:  # 正文进库后 _history_turns 是 async（一次 DB 查询）
        return 99 if tid == "heavy_thread" else 0

    monkeypatch.setattr(server, "_history_turns", _turns)
    started, release = asyncio.Event(), asyncio.Event()
    monkeypatch.setattr(server, "run_agent", _blocking_run(started, release))

    r1 = await client.post("/api/task", json={"query": "q1", "thread_id": "heavy_thread"})
    assert r1.json()["status"] == "started"
    await asyncio.wait_for(started.wait(), timeout=1.0)
    assert server.task_queue.stats()["heavy"]["active"] == 1

    # heavy 池满了（容量 1，队列 0），但 normal 池毫发无伤
    r2 = await client.post("/api/task", json={"query": "q2", "thread_id": "short_thread"})
    assert r2.json()["status"] == "started"
    release.set()


async def test_queued_task_reports_position_over_websocket(
    client: AsyncClient, monkeypatch: Any
) -> None:
    """排队反馈必须早于 session_created 推出去——用户不该对着空白屏幕猜。"""
    monkeypatch.setattr(server, "task_queue", PriorityRequestQueue(normal_slots=1, heavy_slots=1))
    reported: list[tuple[str, int]] = []

    async def _fake_report(thread_id: str, position: int, eta: int, kind: str) -> None:
        reported.append((thread_id, position))

    monkeypatch.setattr(server.monitor, "report_queue_status", _fake_report)
    started, release = asyncio.Event(), asyncio.Event()
    monkeypatch.setattr(server, "run_agent", _blocking_run(started, release))

    await client.post("/api/task", json={"query": "q1", "thread_id": "t1"})
    await asyncio.wait_for(started.wait(), timeout=1.0)
    await client.post("/api/task", json={"query": "q2", "thread_id": "t2"})
    for _ in range(50):  # 让排队任务的后台协程跑到上报那一步
        if reported:
            break
        await asyncio.sleep(0.01)

    assert reported == [("t2", 1)]
    release.set()


async def test_create_task_replace_bypasses_limit(client: AsyncClient, monkeypatch: Any) -> None:
    monkeypatch.setattr(
        server, "task_queue", PriorityRequestQueue(normal_slots=1, heavy_slots=1, queue_depth=0)
    )
    started, release = asyncio.Event(), asyncio.Event()
    monkeypatch.setattr(server, "run_agent", _blocking_run(started, release))

    r1 = await client.post("/api/task", json={"query": "q1", "thread_id": "same"})
    assert r1.status_code == 200
    await asyncio.wait_for(started.wait(), timeout=1.0)

    # 同 thread_id、**不同 query** = 用户改主意了：即便槽满也放行，不该被自己的旧任务挡在门外。
    r2 = await client.post("/api/task", json={"query": "q2", "thread_id": "same"})
    assert r2.status_code == 200
    assert r2.json()["status"] == "started"
    release.set()


async def test_replacing_a_queued_task_does_not_exceed_concurrency(
    client: AsyncClient, monkeypatch: Any
) -> None:
    """覆盖重发一个**还在排队**的任务不能强占：它没持槽，强占就是凭空多出一个并发。"""
    monkeypatch.setattr(
        server, "task_queue", PriorityRequestQueue(normal_slots=1, heavy_slots=1, queue_depth=4)
    )
    started, release = asyncio.Event(), asyncio.Event()
    monkeypatch.setattr(server, "run_agent", _blocking_run(started, release))

    await client.post("/api/task", json={"query": "q1", "thread_id": "t1"})  # 占住唯一的槽
    await asyncio.wait_for(started.wait(), timeout=1.0)
    r2 = await client.post("/api/task", json={"query": "q2", "thread_id": "t2"})
    assert r2.json()["status"] == "queued"  # t2 排队中，没持槽

    # t2 改问 → 覆盖重发。旧的 t2 还在排队，新的不该强占。
    r3 = await client.post("/api/task", json={"query": "q2-改", "thread_id": "t2"})
    assert r3.json()["status"] == "queued"
    stats = server.task_queue.stats()["normal"]
    assert stats["active"] <= stats["capacity"], "覆盖排队中的任务时并发上限被绕过"
    release.set()


async def test_replacing_a_running_task_may_force_a_slot(
    client: AsyncClient, monkeypatch: Any
) -> None:
    """旧任务持槽时，覆盖重发照旧强占——它马上被 cancel 并还槽，真实并发不变。"""
    monkeypatch.setattr(
        server, "task_queue", PriorityRequestQueue(normal_slots=1, heavy_slots=1, queue_depth=0)
    )
    started, release = asyncio.Event(), asyncio.Event()
    monkeypatch.setattr(server, "run_agent", _blocking_run(started, release))

    await client.post("/api/task", json={"query": "q1", "thread_id": "same"})
    await asyncio.wait_for(started.wait(), timeout=1.0)
    r2 = await client.post("/api/task", json={"query": "q2", "thread_id": "same"})
    assert r2.json()["status"] == "started"  # 不被自己的旧任务挡在门外
    release.set()


async def test_create_task_slot_released_after_done(client: AsyncClient, monkeypatch: Any) -> None:
    monkeypatch.setattr(
        server, "task_queue", PriorityRequestQueue(normal_slots=1, heavy_slots=1, queue_depth=0)
    )

    async def _quick_run(query: str, thread_id: str, user_id: str | None = None) -> dict[str, Any]:
        return {"thread_id": thread_id}  # 立即完成，应释放槽

    monkeypatch.setattr(server, "run_agent", _quick_run)

    r1 = await client.post("/api/task", json={"query": "q1", "thread_id": "a"})
    assert r1.status_code == 200
    # 等第一个任务收尾、释放槽。
    for _ in range(50):
        if server.task_queue.active == 0:
            break
        await asyncio.sleep(0.01)
    assert server.task_queue.active == 0
    # 槽已释放：下一个任务能正常起，不被 429。
    r2 = await client.post("/api/task", json={"query": "q2", "thread_id": "b"})
    assert r2.status_code == 200


# ---------- 幂等三层（refdocs 16-5 §3）----------
async def test_same_thread_same_query_is_idempotent(client: AsyncClient, monkeypatch: Any) -> None:
    """第 1 层：刷新页面 / 双击提交 → 领回原任务，不重跑、不 cancel 旧任务。"""
    started, release = asyncio.Event(), asyncio.Event()
    monkeypatch.setattr(server, "run_agent", _blocking_run(started, release))

    r1 = await client.post("/api/task", json={"query": "买包", "thread_id": "t1"})
    assert r1.json()["status"] == "started"
    await asyncio.wait_for(started.wait(), timeout=1.0)
    original = server.active_tasks["t1"].task

    r2 = await client.post("/api/task", json={"query": "买包", "thread_id": "t1"})
    assert r2.json() == {"status": "already_running", "thread_id": "t1"}
    assert server.active_tasks["t1"].task is original  # 原任务没被换掉，也没被 cancel
    assert not original.cancelled()
    release.set()


async def test_cross_thread_duplicate_for_clients_without_thread_id(
    client: AsyncClient, monkeypatch: Any
) -> None:
    """第 3 层：不自带 thread_id 的调用方（脚本 / 裸 API）连发同一 query → 领回原任务。"""
    started, release = asyncio.Event(), asyncio.Event()
    monkeypatch.setattr(server, "run_agent", _blocking_run(started, release))

    r1 = await client.post("/api/task", json={"query": "买包", "user_id": "u"})
    assert r1.json()["status"] == "started"
    first_tid = r1.json()["thread_id"]
    await asyncio.wait_for(started.wait(), timeout=1.0)

    r2 = await client.post("/api/task", json={"query": "买包", "user_id": "u"})
    assert r2.json() == {"status": "duplicate", "thread_id": first_tid}
    assert len(server.active_tasks) == 1  # 没有起第二个任务
    release.set()


async def test_dedup_never_hijacks_a_client_supplied_thread_id(
    client: AsyncClient, monkeypatch: Any
) -> None:
    """connect-first 的前端自带 thread_id，绝不能被并进别人的 thread——否则它那条 WS 收不到任何
    事件、界面永远转圈。它的重复由第 1 层（同 thread 同 query）管。"""
    started, release = asyncio.Event(), asyncio.Event()
    monkeypatch.setattr(server, "run_agent", _blocking_run(started, release))

    r1 = await client.post("/api/task", json={"query": "买包", "thread_id": "t1", "user_id": "u"})
    assert r1.json()["status"] == "started"
    await asyncio.wait_for(started.wait(), timeout=1.0)

    # 同一句话、不同 thread_id、5s 内——但客户端自带 thread_id，照常起任务、用它自己的 tid。
    r2 = await client.post("/api/task", json={"query": "买包", "thread_id": "t2", "user_id": "u"})
    assert r2.json()["status"] == "started"
    assert r2.json()["thread_id"] == "t2"
    assert "t2" in server.active_tasks
    release.set()


async def test_rejected_task_leaves_no_fingerprint(client: AsyncClient, monkeypatch: Any) -> None:
    """被 429 的请求不留指纹——否则用户退避重试会被当成「重复提交」再拒一次，陷入死循环。"""
    monkeypatch.setattr(
        server, "task_queue", PriorityRequestQueue(normal_slots=1, heavy_slots=1, queue_depth=0)
    )
    started, release = asyncio.Event(), asyncio.Event()
    monkeypatch.setattr(server, "run_agent", _blocking_run(started, release))

    await client.post("/api/task", json={"query": "q1", "thread_id": "t1"})
    await asyncio.wait_for(started.wait(), timeout=1.0)
    r2 = await client.post("/api/task", json={"query": "q2", "thread_id": "t2"})
    assert r2.status_code == 429

    assert dedup.check_duplicate(None, "q2") is None  # 没留下指纹
    release.set()


# ---------- 文件下载 ----------
async def test_download_existing_file(
    client: AsyncClient, monkeypatch: Any, tmp_path: Path
) -> None:
    monkeypatch.setattr(server, "OUTPUT_ROOT", tmp_path)
    session = tmp_path / "t-dl"
    session.mkdir()
    (session / "summary.md").write_text("# 购物清单\n帆布旅行包", encoding="utf-8")

    resp = await client.get("/api/files/t-dl/summary.md")
    assert resp.status_code == 200
    assert "帆布旅行包" in resp.text


async def test_download_missing_session_404(
    client: AsyncClient, monkeypatch: Any, tmp_path: Path
) -> None:
    monkeypatch.setattr(server, "OUTPUT_ROOT", tmp_path)
    resp = await client.get("/api/files/ghost/summary.md")
    assert resp.status_code == 404


async def test_download_path_traversal_blocked(
    client: AsyncClient, monkeypatch: Any, tmp_path: Path
) -> None:
    monkeypatch.setattr(server, "OUTPUT_ROOT", tmp_path)
    (tmp_path / "t-dl").mkdir()
    # 编码的 ../ 想读会话目录外的文件 → safe_join 拦截 → 400。
    resp = await client.get("/api/files/t-dl/..%2f..%2fsecret.txt")
    assert resp.status_code == 400


# ---------- 上传 ----------
# 合法 PNG 的字节头（magic bytes）+ 一点载荷。上传口按字节头认图，假头会被 415 挡下。
_PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"fake body"


async def test_upload_writes_file(client: AsyncClient, monkeypatch: Any, tmp_path: Path) -> None:
    monkeypatch.setattr(server, "UPLOAD_ROOT", tmp_path)
    resp = await client.post(
        "/api/upload",
        data={"thread_id": "t-up"},
        files={"file": ("ref.png", _PNG_BYTES, "image/png")},
    )
    assert resp.status_code == 200
    assert resp.json()["filename"] == "ref.png"
    assert (tmp_path / "t-up" / "ref.png").read_bytes() == _PNG_BYTES


async def test_upload_rejects_non_image(
    client: AsyncClient, monkeypatch: Any, tmp_path: Path
) -> None:
    """非图片一律 415：这些字节会被转 base64 喂进 VL 模型。

    认 magic bytes，不认扩展名、也不认 Content-Type——两者都是客户端随便填的。
    """
    monkeypatch.setattr(server, "UPLOAD_ROOT", tmp_path)
    resp = await client.post(
        "/api/upload",
        data={"thread_id": "t-up"},
        # 一个改名成 .png、Content-Type 也谎报成 image/png 的 PDF——两者都不可信，只信字节头。
        files={"file": ("evil.png", b"%PDF-1.7 not an image", "image/png")},
    )
    assert resp.status_code == 415
    assert not (tmp_path / "t-up" / "evil.png").exists()


async def test_upload_thread_id_traversal_blocked(
    client: AsyncClient, monkeypatch: Any, tmp_path: Path
) -> None:
    monkeypatch.setattr(server, "UPLOAD_ROOT", tmp_path)
    # thread_id 来自表单、可控：用 ../ 想把目录建到 UPLOAD_ROOT 外 → _safe_session_dir 拦下 400。
    resp = await client.post(
        "/api/upload",
        data={"thread_id": "../../evil"},
        files={"file": ("x.png", b"x", "image/png")},
    )
    assert resp.status_code == 400
    assert not (tmp_path.parent / "evil").exists()  # 没在 root 外建出目录


async def test_upload_oversize_413(client: AsyncClient, monkeypatch: Any, tmp_path: Path) -> None:
    monkeypatch.setattr(server, "UPLOAD_ROOT", tmp_path)
    monkeypatch.setattr(server, "MAX_UPLOAD_BYTES", 8)
    resp = await client.post(
        "/api/upload",
        data={"thread_id": "t-up"},
        files={"file": ("big.bin", b"x" * 100, "application/octet-stream")},
    )
    assert resp.status_code == 413


# ---------- 偏好读取 ----------
async def test_get_preferences_sorted(
    client: AsyncClient, monkeypatch: Any, tmp_path: Path
) -> None:
    store = get_store()
    await store.write(
        "u1",
        PreferenceEntry(slug="niche", content="喜欢小众品牌", category="brand", domain="other"),
    )
    await store.write(
        "u1",
        PreferenceEntry(
            slug="plastic",
            content="不要塑料",
            category="material",
            polarity="dislike",
            domain="other",
        ),
    )
    monkeypatch.setattr(server, "get_store", lambda: store)

    resp = await client.get("/api/preferences/u1")
    assert resp.status_code == 200
    prefs = resp.json()["preferences"]
    assert len(prefs) == 2
    # dislike 排在 like 前面（按 polarity 排序）。
    assert prefs[0]["content"] == "不要塑料"
    assert prefs[0]["polarity"] == "dislike"


async def test_get_preferences_empty_user(
    client: AsyncClient, monkeypatch: Any, tmp_path: Path
) -> None:
    store = get_store()
    monkeypatch.setattr(server, "get_store", lambda: store)
    resp = await client.get("/api/preferences/nobody")
    assert resp.status_code == 200
    assert resp.json()["preferences"] == []


async def test_health(client: AsyncClient) -> None:
    resp = await client.get("/api/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    # 背压闸用量也在 health 里（A 块 metrics 的雏形）。
    assert body["task_slots"]["limit"] == server.task_queue.limit
    assert body["task_slots"]["active"] == 0
    assert body["task_slots"]["full"] is False
    # 双池明细：normal / heavy 各自的容量与排队深度。
    assert set(body["pools"]) == {"normal", "heavy"}
    assert body["pools"]["normal"]["pending"] == 0


# ---------- 偏好写入 / 删除 / 我的资料（偏好管理页的三个写口）----------
def _draft(**kw: Any) -> dict[str, Any]:
    """一条结构化偏好草稿的 JSON（POST / PUT 的请求体形状）。"""
    base = {
        "content": "不要塑料的",
        "category": "material",
        "domain": "other",
        "slug": "plastic",
        "polarity": "dislike",
        "blocking": False,
        "keywords": ["塑料", "plastic"],
    }
    return {**base, **kw}


async def test_parse_preference_returns_drafts_without_persisting(
    client: AsyncClient, monkeypatch: Any, tmp_path: Path
) -> None:
    """/parse 只解析、**不落库**——草稿要先摆给用户过目，不让 LLM 猜的字段悄悄进库。"""
    store = get_store()
    monkeypatch.setattr(server, "get_store", lambda: store)

    async def fake_parse(text: str) -> list[UserPrefDraft]:
        assert "塑料" in text
        return [UserPrefDraft(**_draft())]

    monkeypatch.setattr(server, "parse_user_preference", fake_parse)

    resp = await client.post("/api/preferences/u1/parse", json={"text": "不要塑料的"})
    assert resp.status_code == 200
    drafts = resp.json()["drafts"]
    assert drafts[0]["blocking"] is False
    assert drafts[0]["keywords"] == ["塑料", "plastic"]
    assert await store.read("u1") == []  # 关键：还没落库


async def test_add_preference_persists_structured_entries(
    client: AsyncClient, monkeypatch: Any, tmp_path: Path
) -> None:
    """POST 收**结构化**条目（用户确认 / 改过的草稿），落库为 source=user。"""
    store = get_store()
    monkeypatch.setattr(server, "get_store", lambda: store)

    # 用户在草稿卡上勾了「绝不推荐」——这是 blocking 唯一的合法来源，也正是草稿卡存在的意义：
    # LLM 只负责把一句话拆成结构化字段，**要不要给它硬淘汰权，由用户按下那一下决定**。
    resp = await client.post("/api/preferences/u1", json={"entries": [_draft(blocking=True)]})
    assert resp.status_code == 200
    added = resp.json()["added"]
    assert added[0]["dedup_key"] == "dislike:material:other:plastic"
    assert added[0]["source"] == "user"
    assert added[0]["blocking"] is True
    assert added[0]["keywords"] == ["塑料", "plastic"]


async def test_update_preference_rekeys_on_field_change(
    client: AsyncClient, monkeypatch: Any, tmp_path: Path
) -> None:
    """改 polarity/category/domain/slug 会换 dedup_key —— PUT 用旧 key 删、按新字段写，不留孤儿。"""
    store = get_store()
    entry = PreferenceEntry(
        slug="plastic", content="不要塑料", category="material", polarity="dislike", domain="other"
    )
    await store.write("u1", entry)
    monkeypatch.setattr(server, "get_store", lambda: store)

    resp = await client.put(
        f"/api/preferences/u1/entry/{entry.dedup_key}",
        json=_draft(polarity="like", content="塑料也行"),  # 极性反转 → 换钥匙
    )
    assert resp.status_code == 200
    entries = await store.read("u1")
    assert len(entries) == 1  # 旧 key 已删，没留下自相矛盾的两条
    assert entries[0].dedup_key == "like:material:other:plastic"
    assert entries[0].content == "塑料也行"
    assert entries[0].source == "user"  # 用户亲手改过 → 归他管，curator 此后不得覆盖


async def test_add_preference_rejects_unparseable(
    client: AsyncClient, monkeypatch: Any, tmp_path: Path
) -> None:
    """解析不出偏好（用户写了句「你好」）→ 400，让前端提示换个说法，而不是静默写脏数据。"""
    store = get_store()
    monkeypatch.setattr(server, "get_store", lambda: store)

    async def fake_parse(text: str) -> list[UserPrefDraft]:
        return []

    monkeypatch.setattr(server, "parse_user_preference", fake_parse)
    resp = await client.post("/api/preferences/u1/parse", json={"text": "你好呀"})
    assert resp.status_code == 400
    assert await store.read("u1") == []


async def test_delete_preference(client: AsyncClient, monkeypatch: Any, tmp_path: Path) -> None:
    """删一条（页面上的 × / 回复下方「记住了 …」的撤销）；重复删幂等，不报错。"""
    store = get_store()
    entry = PreferenceEntry(
        slug="plastic", content="不要塑料", category="material", polarity="dislike", domain="other"
    )
    await store.write("u1", entry)
    monkeypatch.setattr(server, "get_store", lambda: store)

    resp = await client.delete(f"/api/preferences/u1/{entry.dedup_key}")
    assert resp.status_code == 200
    assert await store.read("u1") == []
    assert (await client.delete(f"/api/preferences/u1/{entry.dedup_key}")).status_code == 200


async def test_update_profile_writes_ship_to_and_budget(
    client: AsyncClient, monkeypatch: Any, tmp_path: Path
) -> None:
    """「我的资料」：固定 slug → dedup_key 恒定 → 改值即覆盖（单值语义，不越攒越多）。

    收货国 content 里必须留大写 ISO 码——planner 的收货国解析正是从 content 正则认国家的。
    """
    store = get_store()
    monkeypatch.setattr(server, "get_store", lambda: store)

    await client.put("/api/preferences/u1/profile", json={"dest_country": "JP"})
    resp = await client.put(
        "/api/preferences/u1/profile", json={"dest_country": "US", "budget_max_usd": 300}
    )
    assert resp.status_code == 200
    keys = {p["dedup_key"]: p for p in resp.json()["preferences"]}
    assert "JP" not in keys["like:location:global:ship_to"]["content"]  # 改值即覆盖，不留旧条目
    assert "US" in keys["like:location:global:ship_to"]["content"]
    assert keys["like:budget:global:budget_max"]["content"] == "预算上限约 300 美元"
    assert len(keys) == 2

    # 传空 / 非正数 = 清除该项
    resp = await client.put(
        "/api/preferences/u1/profile", json={"dest_country": "", "budget_max_usd": 0}
    )
    assert resp.json()["preferences"] == []


async def test_update_profile_rejects_unknown_country(
    client: AsyncClient, monkeypatch: Any, tmp_path: Path
) -> None:
    store = get_store()
    monkeypatch.setattr(server, "get_store", lambda: store)
    resp = await client.put("/api/preferences/u1/profile", json={"dest_country": "ZZ"})
    assert resp.status_code == 400


# ---------- GET /api/similar：blocking 黑名单不豁免 ----------
async def test_similar_filters_blocking_blacklist(client: AsyncClient, monkeypatch: Any) -> None:
    """搜同款不走 AgentLoop，但「绝不推荐」黑名单照样生效——展示通路不豁免用户授权的硬排除。
    匿名用户无黑名单，原样直通。"""
    from app.recall.schemas import RecallCandidate

    hits = [
        RecallCandidate(item_id="L1", platform="amazon", title="leather briefcase", score=0.9),
        RecallCandidate(item_id="C1", platform="amazon", title="canvas briefcase", score=0.8),
    ]

    class _FakeRecall:
        def similar(self, item_id: str, top_k: int = 8) -> list[RecallCandidate]:
            return hits[:top_k]

    monkeypatch.setattr(server, "get_recall_client", lambda: _FakeRecall())

    # 匿名：无黑名单，两件直通。
    resp = await client.get("/api/similar/X1")
    assert [i["item_id"] for i in resp.json()["items"]] == ["L1", "C1"]

    # 登录用户拉黑 leather（用户亲手勾的 blocking，domain=bags——这条通路刻意不设域闸）。
    await get_store().write(
        "user-sim",
        PreferenceEntry(
            slug="leather",
            content="绝不推荐皮革",
            category="material",
            domain="bags",
            polarity="dislike",
            keywords=["leather"],
            source="user",
            blocking=True,
        ),
    )
    server.app.dependency_overrides[server.get_current_user_id] = lambda: "user-sim"
    try:
        resp2 = await client.get("/api/similar/X1")
    finally:
        server.app.dependency_overrides.pop(server.get_current_user_id, None)
    assert [i["item_id"] for i in resp2.json()["items"]] == ["C1"]


# ---------- GET /api/uploads（参考图回显：对话气泡 / 历史回看都靠它取图）----------
async def test_download_upload_returns_image(
    client: AsyncClient, monkeypatch: Any, tmp_path: Path
) -> None:
    """上传的图能按文件名取回——这是气泡里那张缩略图的来源。"""
    monkeypatch.setattr(server, "UPLOAD_ROOT", tmp_path)
    (tmp_path / "t-img").mkdir()
    (tmp_path / "t-img" / "ref.png").write_bytes(_PNG_BYTES)

    resp = await client.get("/api/uploads/t-img/ref.png")
    assert resp.status_code == 200
    assert resp.content == _PNG_BYTES


async def test_download_upload_traversal_blocked(
    client: AsyncClient, monkeypatch: Any, tmp_path: Path
) -> None:
    """``../`` 拼接被 safe_join 拦下：取图口与 /api/files 同构，防线不能只装一半。"""
    monkeypatch.setattr(server, "UPLOAD_ROOT", tmp_path)
    (tmp_path / "t-img").mkdir()
    (tmp_path / "secret.txt").write_text("不该被读到")

    # 用 %2f 而非裸 ../：裸的会先被 URL 规范化掉（307 重定向），压根到不了 safe_join，
    # 那样这条测试就成了「测客户端会不会规范化」，而不是「测服务端拦不拦得住」。
    resp = await client.get("/api/uploads/t-img/..%2f..%2fsecret.txt")
    assert resp.status_code == 400


async def test_download_upload_missing_404(
    client: AsyncClient, monkeypatch: Any, tmp_path: Path
) -> None:
    """图不在（比如老会话的上传目录已清）：404 而非 500。前端据此跳过这一张，不连坐其余张。"""
    monkeypatch.setattr(server, "UPLOAD_ROOT", tmp_path)
    (tmp_path / "t-img").mkdir()

    resp = await client.get("/api/uploads/t-img/gone.png")
    assert resp.status_code == 404
