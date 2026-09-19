"""阶段 1-1 credit 预授权：并发透支、按 run_id 幂等结算、崩溃后的额度回收。

**这些用例真正在问的是三件 ``UsageLedger`` 答不了的事。** 账本是事后累加的，所以它对「同一个人
同时发 20 条」完全无感（每条进门读到的余额都是满的）；它没有 run_id，所以一条消息被重投、整轮
重跑就是重复计费；它也不知道「这个人此刻在跑几个」。下面按这三条各自的失败形态写：
``test_concurrent_acquires_do_not_overdraw`` 盯透支、``test_settle_twice_charges_once`` 盯重投、
``test_expired_hold_frees_the_slot`` 盯 kill -9 之后额度还回不回得来。
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select, update

import app.api.server as server
from app.agent.session_io import charge_quota
from app.db import holds, quota
from app.db.models import RunHold, User
from app.db.session import session_factory

pytestmark = pytest.mark.anyio


@pytest.fixture(autouse=True)
async def _holds_on(monkeypatch: Any) -> AsyncIterator[None]:
    """预授权跟着配额走，配额又要求开着鉴权 —— 整个文件在「鉴权已开」下跑。"""
    monkeypatch.setenv("AUTH_ENABLED", "true")
    monkeypatch.setenv("JWT_SECRET", "test-secret-not-real")
    monkeypatch.setenv("DAILY_QUOTA_USD", "1.0")  # = 1000 credits
    monkeypatch.setenv("HOLDS_ENABLED", "true")
    yield


@pytest.fixture
async def client() -> AsyncIterator[AsyncClient]:
    transport = ASGITransport(app=server.app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.fixture(autouse=True)
async def _fake_agent(monkeypatch: Any) -> AsyncIterator[None]:
    """API 用例验的是准入，不是 Agent：run_agent 换成立刻返回的假实现。"""

    async def _noop(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return {"final": "ok"}

    monkeypatch.setattr(server, "run_agent", _noop)
    yield
    for handle in list(server.active_tasks.values()):
        if not handle.task.done():
            handle.task.cancel()
    server.active_tasks.clear()


async def _signup(client: AsyncClient, username: str) -> tuple[str, dict[str, str]]:
    """注册一个夹具用户，并把 ``created_at`` 挪到昨天。

    挪日期不是细节：``users`` 表不在 conftest 的清理名单里，每日新增用户上限（默认 50）数的是
    **今天**多出来几行。本文件一口气建 9 个，全记在今天就会把 tests/test_ratelimit.py 的基数顶穿——
    红的是别人的用例，原因还跟限流毫无关系。仓库既有约定就是夹具用户建在昨天，见那边的 docstring。
    """
    resp = await client.post(
        "/api/auth/register", json={"username": username, "password": "sup3r-secret"}
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    async with session_factory()() as db:
        await db.execute(
            update(User)
            .where(User.id == body["user_id"])
            .values(created_at=datetime.now(UTC) - timedelta(days=1))
        )
        await db.commit()
    return body["user_id"], {"Authorization": f"Bearer {body['access_token']}"}


async def _rows(user_id: str) -> list[RunHold]:
    async with session_factory()() as db:
        res = await db.execute(select(RunHold).where(RunHold.user_id == user_id))
        return list(res.scalars())


# ── 预扣本身 ──────────────────────────────────────────────────────────────────


async def test_acquire_writes_a_queued_hold(client: AsyncClient) -> None:
    """进门占一笔：落一行 queued，占住该档位的 credit。"""
    uid, _ = await _signup(client, "h-acquire")
    res = await holds.acquire_hold(run_id="r1", user_id=uid, thread_id="t1", kind="normal")

    assert res.ok and res.credits_held == holds.HOLD_CREDITS["normal"]
    (row,) = await _rows(uid)
    assert (row.run_id, row.state, row.credits_held) == ("r1", "queued", 20)


async def test_concurrent_acquires_do_not_overdraw(client: AsyncClient) -> None:
    """**额度闸对并发不再是瞎的**：余额只够 2 笔预扣时，第 3 笔拿不到。

    这正是老 ``_enforce_quota`` 漏掉的形态——它读事后账本，三条请求进门时读到的余额一模一样，
    三条全放行。预扣让后到的看见前面占住的那些。
    """
    uid, _ = await _signup(client, "h-overdraw")
    await quota.add_usage(uid, 0.96)  # 烧掉 960 credits，只剩 40 = 两个 normal 档

    assert (await holds.acquire_hold(run_id="a", user_id=uid, kind="normal")).ok
    assert (await holds.acquire_hold(run_id="b", user_id=uid, kind="normal")).ok
    third = await holds.acquire_hold(run_id="c", user_id=uid, kind="normal")

    assert not third.ok and third.reason == holds.REASON_QUOTA
    assert len(await _rows(uid)) == 2  # 被拒的那笔不留行


async def test_concurrency_cap_rejects_the_fourth_run(client: AsyncClient) -> None:
    """同一个人最多同时跑 ``MAX_CONCURRENT_RUNS`` 个；第 4 个被拒，且理由与额度耗尽分开。

    两种拒绝必须分得开：并发超限等一会儿就能进（429），额度耗尽要等日切（402）。
    """
    uid, _ = await _signup(client, "h-concurrency")
    for i in range(holds.MAX_CONCURRENT_RUNS):
        assert (await holds.acquire_hold(run_id=f"run-{i}", user_id=uid)).ok

    over = await holds.acquire_hold(run_id="run-over", user_id=uid)
    assert not over.ok
    assert over.reason == holds.REASON_CONCURRENCY
    assert over.active_runs == holds.MAX_CONCURRENT_RUNS


async def test_same_run_id_does_not_hold_twice(client: AsyncClient) -> None:
    """同一个 run_id 再申请一次不重复占（幂等重发 / 重投）——撞主键即认原来那笔有效。"""
    uid, _ = await _signup(client, "h-idempotent")
    assert (await holds.acquire_hold(run_id="same", user_id=uid)).ok
    assert (await holds.acquire_hold(run_id="same", user_id=uid)).ok

    rows = await _rows(uid)
    assert len(rows) == 1 and rows[0].credits_held == 20


# ── 结算 ─────────────────────────────────────────────────────────────────────


async def _used_credits(user_id: str) -> int:
    async with session_factory()() as db:
        return (await quota.get_quota(db, user_id)).used_credits


async def test_settle_charges_real_cost_and_frees_the_hold(client: AsyncClient) -> None:
    """结算 = 把预扣换成真实用量：行转 settled、账本记的是 cost_usd 而不是当初猜的档位。"""
    uid, _ = await _signup(client, "h-settle")
    await holds.acquire_hold(run_id="s1", user_id=uid, kind="heavy")  # 猜 60

    assert await holds.settle("s1", 0.007, input_tokens=100, output_tokens=20) is True

    (row,) = await _rows(uid)
    assert row.state == "settled" and row.settled_at is not None
    assert (row.credits_held, row.credits_charged) == (60, 7)  # 猜 60、真花 7
    assert await _used_credits(uid) == 7


async def test_settle_twice_charges_once(client: AsyncClient) -> None:
    """**整轮重跑不重复计费**：第二次结算被条件更新挡下，返回 True（不许调用方退回老路补记）。

    这是 at-least-once 投递下最容易漏的洞：消息留在 PEL 被别的 worker 领回来从头跑一遍，
    没有 run_id 时就是实打实的双份账单。
    """
    uid, _ = await _signup(client, "h-twice")
    await holds.acquire_hold(run_id="s2", user_id=uid)

    assert await holds.settle("s2", 0.01) is True
    assert await holds.settle("s2", 0.01) is True  # 重投：钱已记过，仍然「归这条路管」
    assert await _used_credits(uid) == 10  # 只记了一次


async def test_settle_without_hold_returns_false(client: AsyncClient) -> None:
    """没有预扣行 → 返回 False，调用方据此退回 ``add_usage`` 老路（否则这轮的账会整个丢掉）。"""
    await _signup(client, "h-nohold")
    assert await holds.settle("never-acquired", 0.01) is False


async def test_release_frees_the_slot_without_charging(client: AsyncClient) -> None:
    """任务没跑成（入队失败 / 幂等命中）→ 零成本释放：并发额度当场还回来，账本不动。"""
    uid, _ = await _signup(client, "h-release")
    await holds.acquire_hold(run_id="r-free", user_id=uid)

    await holds.release("r-free")

    async with session_factory()() as db:
        assert (await holds._active_snapshot(db, uid)) == (0, 0)
    assert await _used_credits(uid) == 0


async def test_expired_hold_frees_the_slot(client: AsyncClient) -> None:
    """**kill -9 的兜底**：进程没了、行永远停在 running，过了 TTL 就不再占并发与额度。

    读侧按 ``expires_at`` 判，没有扫表协程；过期行留在表里正好是「这个 run 没有终态」的线索。
    """
    uid, _ = await _signup(client, "h-expired")
    await holds.acquire_hold(run_id="zombie", user_id=uid)
    async with session_factory()() as db:  # 模拟：它是很久以前占的
        await db.execute(
            update(RunHold)
            .where(RunHold.run_id == "zombie")
            .values(state="running", expires_at=datetime.now(UTC) - timedelta(seconds=1))
        )
        await db.commit()

    async with session_factory()() as db:
        assert (await holds._active_snapshot(db, uid)) == (0, 0)
    assert (await holds.acquire_hold(run_id="fresh", user_id=uid)).ok


async def test_disabled_switch_is_a_full_noop(client: AsyncClient, monkeypatch: Any) -> None:
    """``HOLDS_ENABLED=0`` → 一行都不落、``settle`` 返回 False，整条链退回老账本。"""
    uid, _ = await _signup(client, "h-off")
    monkeypatch.setenv("HOLDS_ENABLED", "false")

    assert (await holds.acquire_hold(run_id="off-1", user_id=uid)).ok
    assert await _rows(uid) == []
    assert await holds.settle("off-1", 0.01) is False


# ── 接线：记账入口与 HTTP 准入 ────────────────────────────────────────────────


async def test_charge_quota_routes_through_settle(client: AsyncClient) -> None:
    """``run_agent`` 的记账出口带上 run_id 后走结算，重跑一遍仍只记一次。"""
    uid, _ = await _signup(client, "h-charge")
    await holds.acquire_hold(run_id="c1", user_id=uid)
    snap: dict[str, float | int] = {"cost_usd": 0.012, "input_tokens": 10, "output_tokens": 5}

    await charge_quota(uid, snap, run_id="c1")
    await charge_quota(uid, snap, run_id="c1")  # 整轮重跑

    assert await _used_credits(uid) == 12


async def test_charge_quota_falls_back_without_hold(client: AsyncClient) -> None:
    """没有 run_id（开关关着 / 没走准入）→ 老账本照记，绝不能因为改造把账丢了。"""
    uid, _ = await _signup(client, "h-charge-old")
    snap: dict[str, float | int] = {"cost_usd": 0.012, "input_tokens": 10, "output_tokens": 5}

    await charge_quota(uid, snap)

    assert await _used_credits(uid) == 12


async def test_api_rejects_the_fourth_concurrent_run(client: AsyncClient) -> None:
    """HTTP 层：并发超限 → 429 + ``Retry-After``（不是 402——过一会儿真的能进）。"""
    uid, headers = await _signup(client, "h-api-429")
    for i in range(holds.MAX_CONCURRENT_RUNS):
        resp = await client.post(
            "/api/task", json={"query": f"买背包{i}", "thread_id": f"t-{i}"}, headers=headers
        )
        assert resp.status_code == 200, resp.text

    over = await client.post(
        "/api/task", json={"query": "再来一个", "thread_id": "t-over"}, headers=headers
    )
    assert over.status_code == 429
    assert over.headers["Retry-After"]
    detail = over.json()["detail"]
    assert detail["error"] == holds.REASON_CONCURRENCY
    assert detail["max_concurrent_runs"] == holds.MAX_CONCURRENT_RUNS


async def test_already_running_returns_the_hold(client: AsyncClient, monkeypatch: Any) -> None:
    """幂等命中的请求**不留下预扣**：它没起新任务，占着额度就是纯泄漏。

    位置上它躲不掉先占一笔——准入到占槽那一段必须无 ``await``，预扣只能排在幂等判定之前。
    所以每条幂等分支都要负责还回去，这条用例就是那个约束的守栏。
    """
    started = asyncio.Event()

    async def _hang(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        started.set()
        await asyncio.sleep(60)
        return {}

    monkeypatch.setattr(server, "run_agent", _hang)
    uid, headers = await _signup(client, "h-api-dup")
    body = {"query": "同一句话", "thread_id": "t-dup"}

    first = await client.post("/api/task", json=body, headers=headers)
    assert first.status_code == 200
    await asyncio.wait_for(started.wait(), timeout=2)
    again = await client.post("/api/task", json=body, headers=headers)

    assert again.json()["status"] == "already_running"
    alive = [row for row in await _rows(uid) if row.state != "settled"]
    assert len(alive) == 1  # 第二笔当场还掉，只剩真正在跑的那一笔
