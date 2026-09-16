"""快照评测的离线三条（A0-2，不调 LLM，随默认 pytest 跑）。

真 LLM 的四条在 ``scripts/eval/snapshot/``。这里放不需要模型就能判的：

1. 卡片路（``POST /api/threads/{id}/confirmations/orders``）商品库里也没有的 id → 400；
2. 卡片路本会话从没展示过、但商品库里有的 id → **计划要求拒，现状放行**（``hydrate`` 登记表未命中
   即按 id 回源 Qdrant），strict xfail 记录缺口，补上来源校验那天它会翻红提醒摘掉；
3. 同轮 ``shopping_summary`` 之后的 ``ask_user`` 被终结硬停闸拦下——D2 给 ``ask_user`` 加
   ``closes_turn`` 时要改这里的口径（计划 §3-10）。
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
    """现状：终结工具置位后本轮一切工具都被拦，含 ask_user。D2 的 closes_turn 要在这里定新口径。"""
    from app.harness.adapter import HarnessSession
    from app.harness.hooks.termination import check_terminal_reached, mark_terminal
    from app.harness.middleware import HookRejectSignal

    s = HarnessSession(original_query="通勤背包")
    await mark_terminal({"_guard": s.guard, "tool_name": "shopping_summary"})
    with pytest.raises(HookRejectSignal):
        await check_terminal_reached({"_guard": s.guard, "tool_name": "ask_user"})
