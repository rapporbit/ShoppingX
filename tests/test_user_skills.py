"""买家个人 Skill：CRUD 归属 / loader 进 ``<agent-skills>`` 目录 / 显式选择注入 / 找不到不降级。

不跑真实 LLM。测的是机制：个人 skill 只对本人可见、以 ``my/`` 前缀进框架目录块、``Skill`` 工具
能读到正文、显式选择渲染成 reference_only 块、选了不存在的 skill 直接报错而不是静默普通搜索。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

import app.api.server as server
from app.agent.skills import UserSkillLoader, render_selected_skill, resolve_selected_skill
from app.agent.tool_registry import build_toolkit
from app.utils.thread_ctx import thread_scope

pytestmark = pytest.mark.anyio

SKILL = {
    "name": "weekend-backpack",
    "description": "周末短途背包选购：预算 / 轻便 / 收货地明确后按可证实规格比价",
    "body": "# 步骤\n1. 先问容量与预算\n2. 只比可证实的规格\n3. 到手价按收货国算",
}


@pytest.fixture(autouse=True)
async def _auth_on(monkeypatch: Any) -> AsyncIterator[None]:
    monkeypatch.setenv("AUTH_ENABLED", "true")
    monkeypatch.setenv("JWT_SECRET", "test-secret-not-real")
    yield


@pytest.fixture
async def client() -> AsyncIterator[AsyncClient]:
    transport = ASGITransport(app=server.app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def _signup(client: AsyncClient, username: str) -> tuple[str, dict[str, str]]:
    resp = await client.post(
        "/api/auth/register", json={"username": username, "password": "sup3r-secret"}
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    return body["user_id"], {"Authorization": f"Bearer {body['access_token']}"}


async def test_crud_is_scoped_to_owner(client: AsyncClient) -> None:
    _, alice = await _signup(client, "sk-alice")
    _, bob = await _signup(client, "sk-bob")

    created = (await client.post("/api/skills", json=SKILL, headers=alice)).json()["skill"]
    assert created["catalog_name"] == "my/weekend-backpack"
    assert created["version"] == 1

    assert [
        s["name"] for s in (await client.get("/api/skills", headers=alice)).json()["skills"]
    ] == ["weekend-backpack"]
    assert (await client.get("/api/skills", headers=bob)).json()["skills"] == []
    # bob 改不到 / 删不到 alice 的
    resp = await client.put(
        "/api/skills/weekend-backpack", json={**SKILL, "body": "hack"}, headers=bob
    )
    assert resp.status_code == 404
    assert (await client.delete("/api/skills/weekend-backpack", headers=bob)).status_code == 404

    updated = (
        await client.put(
            "/api/skills/weekend-backpack", json={**SKILL, "body": "v2"}, headers=alice
        )
    ).json()["skill"]
    assert updated["version"] == 2 and updated["body"] == "v2"
    assert (await client.delete("/api/skills/weekend-backpack", headers=alice)).json() == {
        "ok": True
    }
    assert (await client.get("/api/skills", headers=alice)).json()["skills"] == []


async def test_validation_and_anonymous(client: AsyncClient) -> None:
    _, alice = await _signup(client, "sk-carol")
    bad = await client.post("/api/skills", json={**SKILL, "name": "有中文"}, headers=alice)
    assert bad.status_code == 422
    assert (await client.post("/api/skills", json=SKILL)).status_code == 401
    # 鉴权开着：匿名连目录都拿不到（与全站口径一致，401）；登录后目录里至少有内置三条
    assert (await client.get("/api/skills/catalog")).status_code == 401
    cat = (await client.get("/api/skills/catalog", headers=alice)).json()["skills"]
    assert {s["name"] for s in cat} >= {"bundle-planning", "cross-border-duty", "image-shopping"}


async def test_catalog_and_loader_expose_my_skill(client: AsyncClient, tmp_path: Path) -> None:
    uid, alice = await _signup(client, "sk-dave")
    await client.post("/api/skills", json=SKILL, headers=alice)

    cat = (await client.get("/api/skills/catalog", headers=alice)).json()["skills"]
    assert {
        "name": "my/weekend-backpack",
        "description": SKILL["description"],
        "source": "user",
    } in cat

    with thread_scope("t-skill", tmp_path, user_id=uid):
        names = {s.name for s in await UserSkillLoader().list_skills()}
        assert names == {"my/weekend-backpack"}
        # 框架目录块里有它、正文不在目录块里
        toolkit = await build_toolkit("main")
        block = await toolkit.get_skill_instructions(["basic"])
        assert "<name>my/weekend-backpack</name>" in block
        assert "只比可证实的规格" not in block

        found = await resolve_selected_skill("my/weekend-backpack")
        assert found is not None and "只比可证实的规格" in found[1]
        assert await resolve_selected_skill("my/nope") is None
    # 离开 scope（匿名）→ 看不到
    assert await UserSkillLoader().list_skills() == []


def test_render_selected_skill_is_reference_only() -> None:
    text = render_selected_skill("my/x", "正文")
    assert text.startswith('<selected-skill name="my/x" authority="reference_only">')
    assert "不是系统指令" in text and text.rstrip().endswith("</selected-skill>")
