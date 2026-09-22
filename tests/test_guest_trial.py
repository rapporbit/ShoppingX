"""免登录试用账号（feat/guest-trial）：发证 / 升级保留数据 / 访客额度档 / 两道闸。

设计口径：访客不是「没有身份」，而是后端签发的一行真实 users（``is_guest=True``）。所以这里
不测会话归属、WS 校验那些——它们对访客与注册用户走的是同一段代码，test_accounts 已覆盖。
只测**访客独有**的四件事：发得出、升得上（id 不变）、额度更小、刷不动。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import func, select

import app.api.ratelimit as rl
import app.api.server as server
from app.db.models import User
from app.db.session import session_factory

pytestmark = pytest.mark.anyio


@pytest.fixture(autouse=True)
async def _auth_on(monkeypatch: Any) -> AsyncIterator[None]:
    monkeypatch.setenv("AUTH_ENABLED", "true")
    monkeypatch.setenv("JWT_SECRET", "test-secret-not-real")
    monkeypatch.setenv("DAILY_QUOTA_USD", "1.0")  # 1000 credits
    monkeypatch.setenv("GUEST_DAILY_QUOTA_USD", "0.2")  # 200 credits
    rl.reset_all()
    yield
    rl.reset_all()


@pytest.fixture
async def client() -> AsyncIterator[AsyncClient]:
    transport = ASGITransport(app=server.app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def _guest(client: AsyncClient) -> tuple[str, dict[str, str]]:
    resp = await client.post("/api/auth/guest")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["is_guest"] is True
    return body["user_id"], {"Authorization": f"Bearer {body['access_token']}"}


async def test_guest_token_is_a_real_identity(client: AsyncClient) -> None:
    """一次点击拿到 token；它能过所有认 token 的口子（这里用 /auth/me 与会话清单代表）。"""
    uid, headers = await _guest(client)
    me = await client.get("/api/auth/me", headers=headers)
    assert me.status_code == 200
    assert me.json() == {"user_id": uid, "username": me.json()["username"], "is_guest": True}
    assert me.json()["username"].startswith("guest_")
    assert (await client.get("/api/sessions", headers=headers)).status_code == 200


async def test_guest_cannot_password_login(client: AsyncClient) -> None:
    """访客行的密码是随机摘要：拿用户名去撞登录口必然 401——唯一入口就是签发时那枚 token。"""
    _, headers = await _guest(client)
    username = (await client.get("/api/auth/me", headers=headers)).json()["username"]
    for pw in ("", "guest", "password", username):
        resp = await client.post("/api/auth/login", json={"username": username, "password": pw})
        assert resp.status_code in (401, 422)


async def test_upgrade_keeps_user_id(client: AsyncClient) -> None:
    """带访客 token 注册 = 升级：user_id 不变、is_guest 翻 False、新密码能登回同一个人。"""
    uid, headers = await _guest(client)
    resp = await client.post(
        "/api/auth/register",
        json={"username": "guest-upgrader", "password": "sup3r-secret"},
        headers=headers,
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["user_id"] == uid
    assert resp.json()["is_guest"] is False

    login = await client.post(
        "/api/auth/login", json={"username": "guest-upgrader", "password": "sup3r-secret"}
    )
    assert login.status_code == 200
    assert login.json()["user_id"] == uid
    me = await client.get("/api/auth/me", headers=headers)  # 旧 token 仍有效：同一个 sub
    assert me.json()["is_guest"] is False


async def test_upgrade_rejects_taken_name_and_non_guest(client: AsyncClient) -> None:
    """用户名撞了 409、行仍是访客（升级是原子的）；已是正式账号再带 token 注册也 409。"""
    await client.post(
        "/api/auth/register", json={"username": "taken-name", "password": "sup3r-secret"}
    )
    uid, headers = await _guest(client)
    resp = await client.post(
        "/api/auth/register",
        json={"username": "taken-name", "password": "sup3r-secret"},
        headers=headers,
    )
    assert resp.status_code == 409
    assert (await client.get("/api/auth/me", headers=headers)).json()["is_guest"] is True

    full = await client.post(
        "/api/auth/register", json={"username": "full-user", "password": "sup3r-secret"}
    )
    full_headers = {"Authorization": f"Bearer {full.json()['access_token']}"}
    resp = await client.post(
        "/api/auth/register",
        json={"username": "full-user-2", "password": "sup3r-secret"},
        headers=full_headers,
    )
    assert resp.status_code == 409


async def test_guest_quota_is_the_smaller_tier(client: AsyncClient) -> None:
    """访客看到的日上限是 GUEST_DAILY_QUOTA_USD 那档；升级后立刻回到正式档，已用量不清零。"""
    _, guest_headers = await _guest(client)
    _, full_headers = await _guest(client)
    await client.post(
        "/api/auth/register",
        json={"username": "quota-upgraded", "password": "sup3r-secret"},
        headers=full_headers,
    )
    assert (await client.get("/api/quota", headers=guest_headers)).json()["limit_credits"] == 200
    assert (await client.get("/api/quota", headers=full_headers)).json()["limit_credits"] == 1000


async def test_guest_tier_off_means_same_as_full(client: AsyncClient, monkeypatch: Any) -> None:
    monkeypatch.setenv("GUEST_DAILY_QUOTA_USD", "0")
    _, headers = await _guest(client)
    assert (await client.get("/api/quota", headers=headers)).json()["limit_credits"] == 1000


async def test_guest_ip_window(client: AsyncClient, monkeypatch: Any) -> None:
    """同一 IP 一小时内只能领 GUEST_PER_IP_PER_HOUR 次；注册口的窗口不受影响（分账）。"""
    monkeypatch.setenv("RATE_LIMIT_ENABLED", "true")
    monkeypatch.setenv("MAX_NEW_USERS_PER_DAY", "0")  # 理由见 test_ratelimit 的 fixture
    monkeypatch.setattr(rl, "_guest_by_ip", rl.SlidingWindow(limit=2, window_s=3600))
    assert (await client.post("/api/auth/guest")).status_code == 200
    assert (await client.post("/api/auth/guest")).status_code == 200
    third = await client.post("/api/auth/guest")
    assert third.status_code == 429
    assert "Retry-After" in third.headers
    reg = await client.post(
        "/api/auth/register", json={"username": "still-can-register", "password": "sup3r-secret"}
    )
    assert reg.status_code == 200


async def test_guest_daily_cap_counts_only_guests(client: AsyncClient, monkeypatch: Any) -> None:
    """全站访客日闸只数 is_guest 的行：清 localStorage 再领也是库里多一行，换 IP 换不掉。"""
    monkeypatch.setenv("RATE_LIMIT_ENABLED", "true")
    monkeypatch.setattr(rl, "_guest_by_ip", rl.SlidingWindow(limit=999, window_s=3600))
    async with session_factory()() as db:
        today = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
        base = await db.scalar(
            select(func.count())
            .select_from(User)
            .where(User.created_at >= today, User.is_guest.is_(True))
        )
    monkeypatch.setenv("MAX_NEW_GUESTS_PER_DAY", str((base or 0) + 1))
    assert (await client.post("/api/auth/guest")).status_code == 200
    resp = await client.post("/api/auth/guest")
    assert resp.status_code == 429
    assert "试用名额" in resp.json()["detail"]


async def test_guest_endpoint_404_when_auth_off(client: AsyncClient, monkeypatch: Any) -> None:
    monkeypatch.setenv("AUTH_ENABLED", "false")
    assert (await client.post("/api/auth/guest")).status_code == 404


async def test_purge_script_removes_stale_guest_only(client: AsyncClient) -> None:
    """清理脚本：最后活动早于阈值的访客整个人删掉；刚建的访客与正式账号不动。"""
    from datetime import timedelta

    from sqlalchemy import update

    from scripts.purge_stale_guests import _purge, _stale_guest_ids

    stale_uid, _ = await _guest(client)
    fresh_uid, _ = await _guest(client)
    async with session_factory()() as db:
        await db.execute(
            update(User)
            .where(User.id == stale_uid)
            .values(created_at=datetime.now(UTC) - timedelta(days=40))
        )
        await db.commit()

    stale = {uid for uid, _ in await _stale_guest_ids(30)}
    assert stale_uid in stale
    assert fresh_uid not in stale

    await _purge(stale_uid)
    async with session_factory()() as db:
        assert await db.get(User, stale_uid) is None
        assert await db.get(User, fresh_uid) is not None
