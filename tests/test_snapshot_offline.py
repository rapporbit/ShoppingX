"""快照评测的离线三条（A0-2，不调 LLM，随默认 pytest 跑）。

真 LLM 的四条在 ``scripts/eval/snapshot/``。这里放不需要模型就能判的：

1. 卡片路（``POST /api/threads/{id}/confirmations/orders``）商品库里也没有的 id → 400；
2. 卡片路本会话从没展示过、但商品库里有的 id → **计划要求拒，现状放行**（``hydrate`` 登记表未命中
   即按 id 回源 Qdrant），strict xfail 记录缺口，补上来源校验那天它会翻红提醒摘掉；
3. 终结后的 ``ask_user``：等回复形态被硬停闸拦下，同批的 ``closes_turn=True`` 收尾问句放行
   （D2 定的口径，计划 §3-10）；
4. ``ask_user(closes_turn=True)`` 本身不登记 waiter、不阻塞、置位终结。
"""

from __future__ import annotations

import pytest

from app.tools.schemas import ItemCandidate

pytestmark = pytest.mark.anyio

ADDR = {
    "recipient_name": "张三",
    "country": "CN",
    "state": "",
    "city": "上海",
    "address_line": "某路 1 号",
    "postal_code": "",
    "phone": "",
}
CATALOG_ITEM = ItemCandidate(
    item_id="B0C64FNWPH", platform="amazon", title="Casual Daypack", price=26.0, currency="USD"
)


@pytest.fixture
async def _client(monkeypatch: pytest.MonkeyPatch, tmp_path):  # type: ignore[no-untyped-def]
    """开着鉴权的 ASGI 客户端，输出根钉到 tmp。"""
    from httpx import ASGITransport, AsyncClient

    import app.api.server as server

    monkeypatch.setenv("AUTH_ENABLED", "true")
    monkeypatch.setenv("JWT_SECRET", "test-secret-not-real")
    monkeypatch.setattr(server, "OUTPUT_ROOT", tmp_path / "output")
    async with AsyncClient(transport=ASGITransport(app=server.app), base_url="http://test") as c:
        yield c


async def _card_prepare(client, username: str, thread_id: str, item_id: str) -> int:  # type: ignore[no-untyped-def]
    """注册 → 认领会话 → 走卡片路出卡，返回状态码。"""
    from app.db.accounts import claim_thread
    from app.db.session import init_db, session_factory

    await init_db()
    resp = await client.post(
        "/api/auth/register", json={"username": username, "password": "sup3r-secret"}
    )
    assert resp.status_code == 200, resp.text
    uid = resp.json()["user_id"]
    headers = {"Authorization": f"Bearer {resp.json()['access_token']}"}
    async with session_factory()() as db:
        await claim_thread(db, thread_id, uid, "下单")
    body = {"items": [{"item_id": item_id, "quantity": 1}], "shipping_address": ADDR}
    r = await client.post(
        f"/api/threads/{thread_id}/confirmations/orders", headers=headers, json=body
    )
    return r.status_code


async def test_card_route_rejects_id_not_in_catalog(_client, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import app.tools._candidates as candidates

    monkeypatch.setattr(candidates, "_fetch_from_store", lambda ids: [])
    assert await _card_prepare(_client, "snap-card-a", "t-snap-card-a", "B0FAKE00000") == 400


@pytest.mark.xfail(
    strict=True,
    reason="来源校验缺口：hydrate 登记表未命中即回源 Qdrant，本会话没展示过的商品也能出卡",
)
async def test_card_route_rejects_id_never_shown_in_thread(_client, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import app.tools._candidates as candidates

    monkeypatch.setattr(
        candidates, "_fetch_from_store", lambda ids: [CATALOG_ITEM] if "B0C64FNWPH" in ids else []
    )
    assert await _card_prepare(_client, "snap-card-b", "t-snap-card-b", "B0C64FNWPH") == 400


async def test_ask_user_after_summary_same_round_is_blocked() -> None:
    """D2 定的口径：终结工具置位后本轮一切工具照拦，**同批的收尾问句除外**。

    ``shopping_summary`` + ``ask_user(closes_turn=True)`` 是同一次决策（给完清单顺带问下一步想看
    什么），批次原子化本就为这种兄弟调用而设，与同轮双 ``create_order`` 一个道理。而等回复形态的
    ``ask_user`` 照拦——清单都给完了还去阻塞等回复，是收尾后的新动作，正是硬停闸要断的打转。
    """
    from app.harness.adapter import HarnessSession
    from app.harness.hooks.termination import check_terminal_reached, mark_terminal
    from app.harness.middleware import HookRejectSignal

    s = HarnessSession(original_query="通勤背包")
    await mark_terminal({"_guard": s.guard, "tool_name": "shopping_summary"})
    with pytest.raises(HookRejectSignal):
        await check_terminal_reached({"_guard": s.guard, "tool_name": "ask_user"})
    with pytest.raises(HookRejectSignal):  # closes_turn=False 显式给出也照拦
        await check_terminal_reached(
            {"_guard": s.guard, "tool_name": "ask_user", "tool_args": {"closes_turn": False}}
        )
    assert (
        await check_terminal_reached(
            {"_guard": s.guard, "tool_name": "ask_user", "tool_args": {"closes_turn": True}}
        )
        is None
    )

    # 下一批（think_step 前进）即便是收尾问句也拦：那已经不是同一次决策了。
    s.guard.think_step += 1
    with pytest.raises(HookRejectSignal):
        await check_terminal_reached(
            {"_guard": s.guard, "tool_name": "ask_user", "tool_args": {"closes_turn": True}}
        )


async def test_closing_ask_user_terminates_and_does_not_wait() -> None:
    """``ask_user(closes_turn=True)``：不登记 waiter、不阻塞，返回问题原文，并置位终结。

    不登记是关键——任务已经结束，没人再去取那个 Future，令牌却要挂满 ``ASK_USER_TIMEOUT_SEC``；
    用户这时点选项打回来，也没有 waiter 接得住。
    """
    from app.agent.constants import is_terminal_call
    from app.api.clarification import create_pending
    from app.harness.adapter import HarnessSession
    from app.harness.hooks.termination import mark_terminal
    from app.tools import ask_user as ask_mod
    from app.utils.thread_ctx import thread_scope

    registered: list[str] = []
    monkey = pytest.MonkeyPatch()
    monkey.setattr(
        ask_mod, "register_waiter", lambda *a, **k: registered.append("registered") or "tok"
    )
    try:
        with thread_scope("t-closes-turn", None):
            out = await ask_mod.ask_user.ainvoke(
                {"question": "想先看哪一类？", "options": ["背包", "行李箱"], "closes_turn": True}
            )
    finally:
        monkey.undo()
    assert out == "想先看哪一类？"
    assert registered == []
    assert create_pending("t-closes-turn") is not None  # 上一问没占着这个 thread 的 pending 位

    assert is_terminal_call("ask_user", {"closes_turn": True})
    assert not is_terminal_call("ask_user", {"closes_turn": False})
    assert not is_terminal_call("ask_user", None)  # 入参拿不到时按非终结走，宁可多跑一轮
    s = HarnessSession(original_query="通勤背包")
    await mark_terminal(
        {"_guard": s.guard, "tool_name": "ask_user", "tool_args": {"closes_turn": True}}
    )
    assert s.guard.terminal_reached
