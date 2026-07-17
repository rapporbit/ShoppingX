"""I 块 · JWT 鉴权：token 签发/验签 + 开关语义 + 越权读 403 + 任务身份取自 token。

核心断言两件事：
1. 鉴权**关闭**（默认）：现状不变——任意读他人偏好、user_id 信前端传入（保 demo 连续性）。
2. 鉴权**开启**：身份只认 token 的 sub——越权读他人偏好 403、缺/伪造 token 401、跑任务用 token 身份。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import jwt
import pytest
from httpx import ASGITransport, AsyncClient

import app.api.auth as auth
import app.api.server as server
from app.memory.store import PreferenceEntry, get_store

_SECRET = "test-secret-please-ignore-0123456789abcdef"  # ≥32 字节，避开 PyJWT 弱密钥告警


@pytest.fixture
async def client() -> AsyncIterator[AsyncClient]:
    transport = ASGITransport(app=server.app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.fixture
def auth_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AUTH_ENABLED", "true")
    monkeypatch.setenv("JWT_SECRET", _SECRET)


# ---------- 单元：签发 / 验签 ----------
def test_create_and_decode_roundtrip(auth_on: None) -> None:
    token = auth.create_access_token("alice")
    assert auth.decode_token(token) == "alice"


def test_decode_rejects_wrong_secret(auth_on: None, monkeypatch: pytest.MonkeyPatch) -> None:
    # 用别的密钥签的 token，本服务验签必失败（401）——伪造 token 进不来。
    forged = jwt.encode({"sub": "mallory"}, "attacker-secret-0123456789abcdef", algorithm="HS256")
    with pytest.raises(Exception) as exc:  # HTTPException(401)
        auth.decode_token(forged)
    assert getattr(exc.value, "status_code", None) == 401


def test_expired_token_rejected(auth_on: None) -> None:
    token = auth.create_access_token("bob", expires_in=-1)  # 已过期
    with pytest.raises(Exception) as exc:
        auth.decode_token(token)
    assert getattr(exc.value, "status_code", None) == 401


def test_secret_required_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    # 开启鉴权却没配密钥：拒绝用弱默认密钥，直接抛（致命配置错）。
    monkeypatch.setenv("AUTH_ENABLED", "true")
    monkeypatch.delenv("JWT_SECRET", raising=False)
    with pytest.raises(RuntimeError, match="JWT_SECRET"):
        auth.create_access_token("x")


# ---------- 集成：开关语义 ----------
async def test_auth_disabled_allows_cross_user_read(
    client: AsyncClient, monkeypatch: Any, tmp_path: Path
) -> None:
    # 默认关闭：现状——谁都能读他人偏好（这正是本块开启后要堵的洞）。
    monkeypatch.setenv("AUTH_ENABLED", "false")
    store = get_store()
    await store.write(
        "u1", PreferenceEntry(slug="niche", content="喜欢小众", category="brand", domain="other")
    )
    monkeypatch.setattr(server, "get_store", lambda: store)

    resp = await client.get("/api/preferences/u1")  # 不带 token
    assert resp.status_code == 200
    assert len(resp.json()["preferences"]) == 1


async def test_auth_enabled_blocks_cross_user_read(
    client: AsyncClient, auth_on: None, monkeypatch: Any, tmp_path: Path
) -> None:
    store = get_store()
    await store.write(
        "victim",
        PreferenceEntry(slug="secret", content="机密偏好", category="other", domain="other"),
    )
    monkeypatch.setattr(server, "get_store", lambda: store)

    alice_token = auth.create_access_token("alice")
    headers = {"Authorization": f"Bearer {alice_token}"}

    # alice 读 victim 的偏好 → 403（堵越权读）。
    r_other = await client.get("/api/preferences/victim", headers=headers)
    assert r_other.status_code == 403
    # alice 读自己的 → 放行。
    r_self = await client.get("/api/preferences/alice", headers=headers)
    assert r_self.status_code == 200


async def test_auth_enabled_requires_token(
    client: AsyncClient, auth_on: None, monkeypatch: Any, tmp_path: Path
) -> None:
    store = get_store()
    monkeypatch.setattr(server, "get_store", lambda: store)
    resp = await client.get("/api/preferences/anyone")  # 开启后缺 token
    assert resp.status_code == 401


async def test_task_identity_from_token_not_body(
    client: AsyncClient, auth_on: None, monkeypatch: Any
) -> None:
    # 开启后：跑任务的 user_id 取自 token 的 sub，前端 body 里传的 user_id 被忽略（防冒名写偏好）。
    seen: dict[str, str | None] = {}

    async def _fake_run(
        query: str, thread_id: str, user_id: str | None = None, **_kw: Any
    ) -> dict[str, Any]:
        seen["user_id"] = user_id
        return {"thread_id": thread_id}

    monkeypatch.setattr(server, "run_agent", _fake_run)
    # 得先真注册一个用户：M16 起，thread 会被认领到 token 的 sub 名下，而归属表有外键——
    # 给一个查无此人的 user_id 签的 token 现在会被拒（401，见 accounts.claim_thread）。
    reg = await client.post(
        "/api/auth/register", json={"username": "real-user", "password": "sup3r-secret"}
    )
    token = reg.json()["access_token"]
    real_uid = reg.json()["user_id"]
    resp = await client.post(
        "/api/task",
        json={"query": "买帐篷", "thread_id": "t-auth", "user_id": "FAKE-injected"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200
    import asyncio

    await asyncio.sleep(0.05)  # 让后台 _runner 跑起来
    assert seen["user_id"] == real_uid  # 用 token 身份，不用 body 里的 FAKE-injected


async def test_issue_token_endpoint(client: AsyncClient, auth_on: None, monkeypatch: Any) -> None:
    monkeypatch.setenv("AUTH_DEV_TOKEN", "true")  # 发证口需独立开关
    resp = await client.post("/api/auth/token", json={"user_id": "carol"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["token_type"] == "bearer"
    assert auth.decode_token(body["access_token"]) == "carol"


async def test_issue_token_404_when_auth_disabled(client: AsyncClient, monkeypatch: Any) -> None:
    monkeypatch.setenv("AUTH_ENABLED", "false")
    resp = await client.post("/api/auth/token", json={"user_id": "x"})
    assert resp.status_code == 404


async def test_issue_token_404_when_dev_flag_off(
    client: AsyncClient, auth_on: None, monkeypatch: Any
) -> None:
    # 关键安全断言：只开 AUTH_ENABLED、不开 AUTH_DEV_TOKEN → 发证口关闭（开鉴权≠暴露冒名工厂）。
    monkeypatch.setenv("AUTH_DEV_TOKEN", "false")
    resp = await client.post("/api/auth/token", json={"user_id": "victim"})
    assert resp.status_code == 404


def test_decode_rejects_token_without_exp(auth_on: None) -> None:
    # 无 exp 声明的 token 必拒（防泄漏 token 永久有效）。
    import jwt as _jwt

    no_exp = _jwt.encode({"sub": "x"}, _SECRET, algorithm="HS256")  # 不带 exp
    with pytest.raises(Exception) as exc:
        auth.decode_token(no_exp)
    assert getattr(exc.value, "status_code", None) == 401


def test_validate_auth_config_fails_without_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    # 启动校验：开鉴权但漏配密钥 → fail-fast（而非每请求 500）。
    monkeypatch.setenv("AUTH_ENABLED", "true")
    monkeypatch.delenv("JWT_SECRET", raising=False)
    with pytest.raises(RuntimeError, match="JWT_SECRET"):
        auth.validate_auth_config()
