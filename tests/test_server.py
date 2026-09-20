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
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
from conftest import FakeRedis  # tests/ 的 conftest（pytest 已把它放进 sys.path）
from httpx import ASGITransport, AsyncClient

import app.api.server as server
from app import worker
from app.api import dedup
from app.queue import InProcessQueue, set_task_queue
from app.db.models import User
from app.db.session import init_db, session_factory
from app.memory.fact_store import get_fact_store
from app.memory.facts import MemoryCategory, MemoryFact


@pytest.fixture
async def client() -> AsyncIterator[AsyncClient]:
    """ASGI 直连后端 app 的 httpx 客户端（不开真实端口）。"""
    transport = ASGITransport(app=server.app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.fixture(autouse=True)
async def _clean_tasks(fake_redis: FakeRedis) -> AsyncIterator[None]:
    """每个用例前换一份全新队列**并起一个消费方**，用例后清 active_tasks 并取消遗留任务。

    阶段 1 条 7 起 API 进程不再直接跑 ``run_agent``：任务一律入队，由 worker 消费。所以单测要自己
    扮演那个 worker——否则每条用例的任务都会停在队列里，``started`` 永远等不到。用例里换掉的是
    ``worker.run_agent``（真正跑它的那个名字），不是 ``server.run_agent``。

    指纹窗口 1-2 起在 Redis 上，这里清的是 conftest 注入的那份假客户端——**不能调
    ``dedup.reset()``**，那会把假客户端一起摘掉，于是每条用例都去连真 Redis、连不上就 503。
    """
    queue = InProcessQueue()
    set_task_queue(queue)
    stop = asyncio.Event()
    consumer = asyncio.create_task(queue.consume("test-worker", worker.handle_task, stop.is_set, 4))
    fake_redis.store.clear()
    yield
    stop.set()
    consumer.cancel()
    with suppress(asyncio.CancelledError):
        await consumer
    for handle in list(server.active_tasks.values()):
        if not handle.task.done():
            handle.task.cancel()
    server.active_tasks.clear()
    set_task_queue(None)
    fake_redis.store.clear()


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

    monkeypatch.setattr(worker, "run_agent", _fake_run)

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

    monkeypatch.setattr(worker, "run_agent", _fake_run)
    resp = await client.post("/api/task", json={"query": "买帐篷"})
    assert resp.status_code == 200
    assert resp.json()["thread_id"]  # 服务端兜底生成非空 id


# ---------- cancel ----------
async def test_cancel_running_task(client: AsyncClient, monkeypatch: Any) -> None:
    started = asyncio.Event()

    async def _slow_run(query: str, thread_id: str, user_id: str | None = None, **_kw: Any) -> None:
        started.set()
        await asyncio.sleep(30)  # 长任务，等被取消

    monkeypatch.setattr(worker, "run_agent", _slow_run)
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

    monkeypatch.setattr(worker, "run_agent", _fake_run)
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

    monkeypatch.setattr(worker, "run_agent", _fake_run)
    monkeypatch.setattr(server.event_log, "replay_current_run", _fake_replay)

    await client.post("/api/task", json={"query": "q", "thread_id": "t-ended"})
    await asyncio.wait_for(started.wait(), timeout=1.0)

    resp = await client.get("/api/task/t-ended/inflight")
    assert resp.status_code == 200
    assert resp.json() == {"running": False, "query": None, "images": [], "events": []}
    release.set()


# ---------- 背压（进程内准入池随阶段 1 条 7 删除）----------
#
# 「槽满排队 / 队列满 429 / 覆盖重发不超并发 / 收尾放槽」这四件事现在由队列深度闸 + run_holds 的
# 用户级并发承担，用例分别在 tests/test_queue_wiring.py（429 与排位）与 tests/test_holds.py（用户
# 级并发上限）。这里不再留同名用例，免得读的人以为进程内还有一个池子。


def _blocking_run(started: asyncio.Event, release: asyncio.Event) -> Any:
    async def _run(
        query: str, thread_id: str, user_id: str | None = None, **_kw: Any
    ) -> dict[str, Any]:
        started.set()
        await release.wait()  # 卡住任务，让它在 active_tasks 里可被观察 / 取消
        return {"thread_id": thread_id}

    return _run


# ---------- 幂等三层（refdocs 16-5 §3）----------
async def test_same_thread_same_query_is_idempotent(client: AsyncClient, monkeypatch: Any) -> None:
    """第 1 层：刷新页面 / 双击提交 → 领回原任务，不重跑、不 cancel 旧任务。"""
    started, release = asyncio.Event(), asyncio.Event()
    monkeypatch.setattr(worker, "run_agent", _blocking_run(started, release))

    r1 = await client.post("/api/task", json={"query": "买包", "thread_id": "t1"})
    assert r1.json()["status"] == "started"
    await asyncio.wait_for(started.wait(), timeout=1.0)
    original = server.active_tasks["t1"].task

    r2 = await client.post("/api/task", json={"query": "买包", "thread_id": "t1"})
    assert r2.json() == {"status": "already_running", "thread_id": "t1"}
    assert server.active_tasks["t1"].task is original  # 原任务没被换掉，也没被 cancel
    assert not original.cancelled()
    release.set()


async def test_idempotency_holds_when_this_process_never_saw_the_run(
    client: AsyncClient, monkeypatch: Any
) -> None:
    """**真相在 DB，不在 active_tasks**（阶段 1-2）：清空本进程的字典（= 另一台副本收到请求），
    同 thread 同 query 仍被判 already_running，不会各起一个 run。"""
    started, release = asyncio.Event(), asyncio.Event()
    monkeypatch.setattr(worker, "run_agent", _blocking_run(started, release))

    try:
        await client.post("/api/task", json={"query": "买包", "thread_id": "t1"})
        await asyncio.wait_for(started.wait(), timeout=1.0)

        handle = server.active_tasks.pop("t1")  # 这台副本对那个 run 一无所知
        r2 = await client.post("/api/task", json={"query": "买包", "thread_id": "t1"})
        assert r2.json() == {"status": "already_running", "thread_id": "t1"}
        assert "t1" not in server.active_tasks  # 也没有凭空起第二个
        server.active_tasks["t1"] = handle  # 放回去，让 teardown 照常收尾
    finally:
        release.set()


async def test_cross_thread_duplicate_for_clients_without_thread_id(
    client: AsyncClient, monkeypatch: Any
) -> None:
    """第 3 层：不自带 thread_id 的调用方（脚本 / 裸 API）连发同一 query → 领回原任务。"""
    started, release = asyncio.Event(), asyncio.Event()
    monkeypatch.setattr(worker, "run_agent", _blocking_run(started, release))

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
    monkeypatch.setattr(worker, "run_agent", _blocking_run(started, release))

    r1 = await client.post("/api/task", json={"query": "买包", "thread_id": "t1", "user_id": "u"})
    assert r1.json()["status"] == "started"
    await asyncio.wait_for(started.wait(), timeout=1.0)

    # 同一句话、不同 thread_id、5s 内——但客户端自带 thread_id，照常起任务、用它自己的 tid。
    r2 = await client.post("/api/task", json={"query": "买包", "thread_id": "t2", "user_id": "u"})
    assert r2.json()["status"] == "started"
    assert r2.json()["thread_id"] == "t2"
    assert "t2" in server.active_tasks
    release.set()


async def test_rejected_task_leaves_no_fingerprint(
    client: AsyncClient, monkeypatch: Any, fake_redis: FakeRedis
) -> None:
    """被 429 的请求不留指纹——否则用户退避重试会被当成「重复提交」再拒一次，陷入死循环。

    1-2 起指纹是**进门就登记**的（``SET NX`` 没法只查不登记），所以这条断言守的是**闸的相对顺序**：
    队列深度闸必须排在指纹登记之前，被它拒掉的请求才不会在窗口里留下自己的指纹。
    """
    monkeypatch.setattr(server, "QUEUE_MAX_DEPTH", 0)  # 深度闸当场拒：模拟队列已积压到顶

    r2 = await client.post("/api/task", json={"query": "q2"})  # 不自带 tid → 才会走指纹层
    assert r2.status_code == 429
    assert fake_redis.store == {}  # 窗口里什么都没留下：退避重试不会被自己刚才那次挡住
    assert await dedup.check_duplicate(None, "q2", "t-retry") is None


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


async def test_health(client: AsyncClient) -> None:
    resp = await client.get("/api/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    # 背压用量也在 health 里（A 块 metrics 的雏形）：本副本在等结果的轮数 + 队列积压。
    assert body["active_tasks"] == 0
    assert body["queue"]["max_depth"] == server.QUEUE_MAX_DEPTH
    assert body["queue"]["depth"] == 0
    # 整轮缓存状态（批2-5）：评测脚本靠这一项拒跑，缺了它 run_rubric 就探不到后端开着缓存。
    assert body["turn_cache"] == {"enabled": False, "entries": 0}


async def _with_user(uid: str) -> None:
    """建一个真用户行——``memory_facts.user_id`` 有外键，``memory_purge_gen`` 也挂在这张表上。

    ``created_at`` 设成昨天：``ratelimit.guard_daily_signups`` 只数当天新建的 users 行，而测试库
    是整套共享的，用默认「现在」会把当天注册名额吃光（只在全量跑时才复现的串台）。
    """
    await init_db()
    async with session_factory()() as db:
        db.add(
            User(
                id=uid,
                username=f"t_{uid[:8]}",
                password_hash="x",
                created_at=datetime.now(UTC) - timedelta(days=1),
            )
        )
        await db.commit()


# ---------- 长期记忆：读 / 手填 / 改 / 删 / 清空（偏好管理页的五个口）----------
def _uid() -> str:
    """每个用例一个独立 user_id——测试库是整套共享的，隔离靠身份。"""
    return f"u-{uuid4().hex[:8]}"


def _fact_body(**kw: Any) -> dict[str, Any]:
    """一条事实的 JSON（POST / PUT 的请求体形状）——就是模型看到的那三个字段。"""
    return {"key": "material_avoid", "value": "不要塑料的", "category": "constraint", **kw}


async def test_add_and_get_preferences(
    client: AsyncClient, monkeypatch: Any, tmp_path: Path
) -> None:
    """手填直接落库，无「先解析成草稿」那一步——三个字段用户自己填得出来。"""
    uid = _uid()
    store = get_fact_store()
    monkeypatch.setattr(server, "get_fact_store", lambda: store)

    resp = await client.post(f"/api/preferences/{uid}", json=_fact_body())
    assert resp.status_code == 200
    assert resp.json()["added"][0]["key"] == "material_avoid"

    got = await client.get(f"/api/preferences/{uid}")
    prefs = got.json()["preferences"]
    assert len(prefs) == 1
    assert prefs[0]["value"] == "不要塑料的"
    assert prefs[0]["category"] == "constraint"


async def test_get_preferences_empty_user(
    client: AsyncClient, monkeypatch: Any, tmp_path: Path
) -> None:
    uid = _uid()
    store = get_fact_store()
    monkeypatch.setattr(server, "get_fact_store", lambda: store)
    resp = await client.get("/api/preferences/nobody")
    assert resp.status_code == 200
    assert resp.json()["preferences"] == []


async def test_add_preference_rejects_pii_without_echoing_value(
    client: AsyncClient, monkeypatch: Any, tmp_path: Path
) -> None:
    """页面**不是特权入口**：与 save_memory / 回合后抽取过同一道 validate_fact。

    400 的消息里不回显 value——被拒的多半正是不该扩散的东西。
    """
    uid = _uid()
    store = get_fact_store()
    monkeypatch.setattr(server, "get_fact_store", lambda: store)

    card = "4111 1111 1111 1111"
    resp = await client.post(f"/api/preferences/{uid}", json=_fact_body(value=f"我的卡号 {card}"))
    assert resp.status_code == 400
    assert card not in resp.text
    assert await store.get_facts(uid) == []


async def test_update_preference_rekeys_and_deletes_old(
    client: AsyncClient, monkeypatch: Any, tmp_path: Path
) -> None:
    """改 key = 换一条：PUT 用旧 key 删、按新 key 写，不留孤儿。"""
    uid = _uid()
    store = get_fact_store()
    await store.upsert_facts(uid, [MemoryFact(key="material_avoid", value="不要塑料")])
    monkeypatch.setattr(server, "get_fact_store", lambda: store)

    resp = await client.put(
        f"/api/preferences/{uid}/entry/material_avoid",
        json=_fact_body(key="material_like", value="塑料也行", category="preference"),
    )
    assert resp.status_code == 200
    facts = await store.get_facts(uid)
    assert len(facts) == 1  # 旧 key 已删，没留下自相矛盾的两条
    assert (facts[0].key, facts[0].value) == ("material_like", "塑料也行")


async def test_delete_preference_is_idempotent(
    client: AsyncClient, monkeypatch: Any, tmp_path: Path
) -> None:
    """删一条（页面上的 × / 回复下方「记住了 …」的撤销）；重复删幂等，不报错。"""
    uid = _uid()
    store = get_fact_store()
    await store.upsert_facts(uid, [MemoryFact(key="material_avoid", value="不要塑料")])
    monkeypatch.setattr(server, "get_fact_store", lambda: store)

    resp = await client.delete(f"/api/preferences/{uid}/material_avoid")
    assert resp.status_code == 200
    assert await store.get_facts(uid) == []
    assert (await client.delete(f"/api/preferences/{uid}/material_avoid")).status_code == 200


async def test_clear_preferences_bumps_purge_generation(
    client: AsyncClient, monkeypatch: Any, tmp_path: Path
) -> None:
    """「全部清除」：清空 + 代数加一，让正在跑的抽取整批作废（否则清了个寂寞）。"""
    uid = f"u-purge-{uuid4().hex[:8]}"
    await _with_user(uid)
    store = get_fact_store()
    await store.upsert_facts(uid, [MemoryFact(key="k1", value="v1")])
    monkeypatch.setattr(server, "get_fact_store", lambda: store)
    before = await store.purge_generation(uid)

    resp = await client.delete(f"/api/preferences/{uid}")
    assert resp.status_code == 200
    assert await store.get_facts(uid) == []
    assert await store.purge_generation(uid) == before + 1


# ---------- GET /api/similar：不再按长期记忆过滤（M4）----------
async def test_similar_returns_neighbors_unfiltered(client: AsyncClient, monkeypatch: Any) -> None:
    """搜同款不走 AgentLoop，也**不再**自己按记忆挡一遍。

    记忆现在只有一种生效方式——模型把它写进工具入参——而这条通路没有模型。留一条看不见的
    第二生效通路，正是 M4 要消灭的东西。
    """
    from app.recall.schemas import RecallCandidate

    hits = [
        RecallCandidate(item_id="L1", platform="amazon", title="leather briefcase", score=0.9),
        RecallCandidate(item_id="C1", platform="amazon", title="canvas briefcase", score=0.8),
    ]

    class _FakeRecall:
        def similar(self, item_id: str, top_k: int = 8) -> list[RecallCandidate]:
            return hits[:top_k]

    monkeypatch.setattr(server, "get_recall_client", lambda: _FakeRecall())

    resp = await client.get("/api/similar/X1")
    assert [i["item_id"] for i in resp.json()["items"]] == ["L1", "C1"]

    # 登录用户即便记着「绝不推荐皮革」，同款列表也照常给两件。
    await get_fact_store().upsert_facts(
        "user-sim",
        [MemoryFact(key="material_avoid", value="绝不要皮革", category=MemoryCategory.CONSTRAINT)],
    )
    server.app.dependency_overrides[server.get_current_user_id] = lambda: "user-sim"
    try:
        resp2 = await client.get("/api/similar/X1")
    finally:
        server.app.dependency_overrides.pop(server.get_current_user_id, None)
    assert [i["item_id"] for i in resp2.json()["items"]] == ["L1", "C1"]


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


# ---------- request_id（阶段 4-5：API 与 worker 两个进程的日志关联）----------
async def test_response_carries_request_id(client: AsyncClient) -> None:
    """每个响应都回显 request_id——出问题时用户截图里就带着它。"""
    resp = await client.get("/api/task/nope/inflight")
    rid = resp.headers.get("X-Request-Id", "")
    assert len(rid) == 16, f"没生成 request_id：{rid!r}"


async def test_incoming_request_id_is_reused(client: AsyncClient) -> None:
    """前面挡着 nginx 时它才是真正的入口，自己再生成一个等于在链路中间断一次。"""
    resp = await client.get("/api/task/nope/inflight", headers={"X-Request-Id": "from-edge-1"})
    assert resp.headers["X-Request-Id"] == "from-edge-1"
