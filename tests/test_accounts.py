"""M16 账户与会话归属：注册 / 登录 / 会话清单 / thread 维度越权。

**这些用例真正在问的是同一个问题：「别人能不能碰到我的东西」。** 归属表之前，答案是「能」——
拿到 thread_id 就能读他人历史、下他人产物、连他人事件流。所以越权那几条是本文件的主心骨，
注册登录只是它的前置条件。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

import app.api.server as server

pytestmark = pytest.mark.anyio


@pytest.fixture(autouse=True)
async def _auth_on(monkeypatch: Any) -> AsyncIterator[None]:
    """账户功能只在开启鉴权时存在（关闭时一律 404），故整个文件都在「鉴权已开」下跑。

    建表不在这儿——它是全局前置，由 conftest 统一做（见那边的注释）。
    """
    monkeypatch.setenv("AUTH_ENABLED", "true")
    monkeypatch.setenv("JWT_SECRET", "test-secret-not-real")
    yield


@pytest.fixture
async def client() -> AsyncIterator[AsyncClient]:
    transport = ASGITransport(app=server.app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def _signup(client: AsyncClient, username: str) -> tuple[str, dict[str, str]]:
    """注册一个用户，返回 (user_id, 可直接用的 Authorization 头)。"""
    resp = await client.post(
        "/api/auth/register", json={"username": username, "password": "sup3r-secret"}
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    return body["user_id"], {"Authorization": f"Bearer {body['access_token']}"}


async def test_register_then_login(client: AsyncClient) -> None:
    """注册 → 拿到 token；同样的用户名密码能登回同一个 user_id（身份是稳定的）。"""
    uid, _ = await _signup(client, "alice")
    resp = await client.post(
        "/api/auth/login", json={"username": "alice", "password": "sup3r-secret"}
    )
    assert resp.status_code == 200
    assert resp.json()["user_id"] == uid  # 同一个人，不是每次登录换个身份


async def test_duplicate_username_rejected(client: AsyncClient) -> None:
    """用户名唯一——靠数据库唯一索引，不靠应用层「先查再插」（并发下必漏）。"""
    await _signup(client, "bob")
    resp = await client.post(
        "/api/auth/register", json={"username": "bob", "password": "another-pass"}
    )
    assert resp.status_code == 409


async def test_wrong_password_rejected_without_leaking_existence(client: AsyncClient) -> None:
    """密码错与查无此人**回一模一样的 401**：否则登录口就成了用户名探测器。"""
    await _signup(client, "carol")
    wrong = await client.post(
        "/api/auth/login", json={"username": "carol", "password": "wrong-pass-here"}
    )
    ghost = await client.post(
        "/api/auth/login", json={"username": "nobody-here", "password": "wrong-pass-here"}
    )
    assert wrong.status_code == ghost.status_code == 401
    assert wrong.json()["detail"] == ghost.json()["detail"]


async def test_password_never_stored_in_plaintext(client: AsyncClient) -> None:
    """库里存的是 bcrypt 摘要，不是明文——库被拖走也拿不到用户的密码。"""
    from sqlalchemy import select

    from app.db.models import User
    from app.db.session import session_factory

    await _signup(client, "dave")
    async with session_factory()() as db:
        user = (await db.execute(select(User).where(User.username == "dave"))).scalar_one()
    assert "sup3r-secret" not in user.password_hash
    assert user.password_hash.startswith("$2b$")  # bcrypt 摘要的特征前缀


async def test_unauthenticated_request_rejected(client: AsyncClient) -> None:
    """开了鉴权就必须带 token——不带一律 401，不再有「信前端传的 user_id」这条路。"""
    assert (await client.get("/api/sessions")).status_code == 401


# --- 会话归属：本文件的主心骨 ------------------------------------------------


async def _own_thread(client: AsyncClient, headers: dict[str, str], tid: str, query: str) -> None:
    """让某人起一次任务，从而把 tid 认领到他名下（run_agent 已被换成假的，不真跑 LLM）。"""
    resp = await client.post("/api/task", json={"query": query, "thread_id": tid}, headers=headers)
    assert resp.status_code == 200, resp.text


@pytest.fixture(autouse=True)
async def _fake_agent(monkeypatch: Any) -> AsyncIterator[None]:
    """把 run_agent 换成立刻返回的假实现：本文件验的是归属与鉴权，不是 Agent 本身。"""

    async def _noop(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return {"final": "ok"}

    monkeypatch.setattr(server, "run_agent", _noop)
    yield
    for handle in list(server.active_tasks.values()):
        if not handle.task.done():
            handle.task.cancel()
    server.active_tasks.clear()


async def test_task_claims_thread_and_it_shows_up_in_my_sessions(client: AsyncClient) -> None:
    """起一次任务 = 认领一段会话；它随即出现在「我的会话」里。

    这就是「换台设备登录还能看到我的历史」的底座：清单来自后端归属表，不再是浏览器 localStorage。
    """
    _, alice = await _signup(client, "erin")
    await _own_thread(client, alice, "t-erin-1", "买帐篷")

    sessions = (await client.get("/api/sessions", headers=alice)).json()["sessions"]
    assert [s["thread_id"] for s in sessions] == ["t-erin-1"]
    assert sessions[0]["title"] == "买帐篷"  # 标题取首轮提问，省一次 LLM 调用


async def test_sessions_are_scoped_per_user(client: AsyncClient) -> None:
    """我的清单里只有我的会话——别人的不会漏进来。"""
    _, frank = await _signup(client, "frank")
    _, grace = await _signup(client, "grace")
    await _own_thread(client, frank, "t-frank-1", "买登山杖")

    assert (await client.get("/api/sessions", headers=grace)).json()["sessions"] == []


@pytest.mark.parametrize(
    "method,path_tpl,slug",
    [
        ("get", "/api/history/{tid}", "history"),  # 对话正文——最要紧的一个口
        ("get", "/api/files/{tid}/summary.md", "files"),  # 会话产物
        ("get", "/api/task/{tid}/inflight", "inflight"),  # 在跑那轮的提问原文 + 事件流
        ("post", "/api/task/{tid}/cancel", "cancel"),  # 掐断别人正在跑的任务
    ],
)
async def test_other_user_cannot_touch_my_thread(
    client: AsyncClient, method: str, path_tpl: str, slug: str
) -> None:
    """**归属表要堵的洞：** 拿到别人的 thread_id，就能读他的历史 / 下他的产物 / 看他的实时事件 /
    掐断他的任务。逐个口子验 403——这四个口子此前全是敞开的。

    每个参数用独立的用户名与 tid：库是整轮 pytest 共用的一个文件，共用名字会让先跑的用例把
    后面的 tid 先认领走（再认领即 403），红的就成了测试自己而不是被测代码。
    """
    tid = f"t-victim-{slug}"
    _, victim = await _signup(client, f"victim-{slug}")
    _, attacker = await _signup(client, f"attacker-{slug}")
    await _own_thread(client, victim, tid, "买相机")

    resp = await getattr(client, method)(path_tpl.format(tid=tid), headers=attacker)
    assert resp.status_code == 403, f"{method.upper()} {path_tpl} 未拦住越权：{resp.status_code}"


async def test_delete_session_removes_it_from_my_list(client: AsyncClient) -> None:
    """侧栏删除：从我的清单里消失。删的是归属记录（入口），对话正文仍在磁盘上。"""
    _, ivan = await _signup(client, "ivan")
    await _own_thread(client, ivan, "t-ivan-1", "买耳机")

    resp = await client.delete("/api/sessions/t-ivan-1", headers=ivan)
    assert resp.status_code == 200
    assert (await client.get("/api/sessions", headers=ivan)).json()["sessions"] == []


async def test_cannot_delete_someone_elses_session(client: AsyncClient) -> None:
    """删别人的会话 → 403。删除口和读取口一样要校验属主，漏一个就是个洞。"""
    _, judy = await _signup(client, "judy")
    _, karl = await _signup(client, "karl")
    await _own_thread(client, judy, "t-judy-1", "买键盘")

    assert (await client.delete("/api/sessions/t-judy-1", headers=karl)).status_code == 403
    # judy 的会话完好无损
    assert (await client.get("/api/sessions", headers=judy)).json()["sessions"][0][
        "thread_id"
    ] == "t-judy-1"


async def test_token_for_nonexistent_user_is_rejected(client: AsyncClient) -> None:
    """token 验签通过、但它的 sub 查无此人（账号被删 / 换了库 / 开发态发证口签的假身份）——
    该让人重新登录（401），而不是让外键约束以 500 的形式打在脸上。"""
    from app.api import auth

    token = auth.create_access_token("ghost-user-never-registered")
    resp = await client.post(
        "/api/task",
        json={"query": "买帐篷", "thread_id": "t-ghost"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 401


async def test_cannot_hijack_someone_elses_thread_by_posting_to_it(client: AsyncClient) -> None:
    """越权**写**：拿别人的 thread_id 发消息，不能把他的会话过户到自己名下。

    只校验读是不够的——认领若是「谁最后发言算谁的」，攻击者发一句话就把受害者的会话抢走了。
    """
    _, henry = await _signup(client, "henry")
    _, mallory = await _signup(client, "mallory")
    await _own_thread(client, henry, "t-henry-1", "买滑雪板")

    resp = await client.post(
        "/api/task", json={"query": "偷偷续聊", "thread_id": "t-henry-1"}, headers=mallory
    )
    assert resp.status_code == 403
    # 会话仍在 henry 名下，且没跑到 mallory 的清单里去
    assert (await client.get("/api/sessions", headers=mallory)).json()["sessions"] == []
    assert (await client.get("/api/sessions", headers=henry)).json()["sessions"][0][
        "thread_id"
    ] == "t-henry-1"
