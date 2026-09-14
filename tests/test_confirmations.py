"""交易确认记录：准备 / 决议 / 幂等 / 过期 / hash / 归属 / 注入块 / HTTP 接口。"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from app.tools.schemas import ItemCandidate
from app.trade.confirmation import Confirmation, ConfirmationError
from app.trade.confirmations import (
    prepare_cancel_confirmation,
    prepare_order_confirmation,
    resolve_confirmation,
    trade_state,
)
from app.trade.order import Order, OrderStatus
from app.trade.usecases import LineRequest, NoCandidateError

pytestmark = pytest.mark.anyio


class MemConfirmations:
    def __init__(self) -> None:
        self.rows: dict[str, Confirmation] = {}

    async def save(self, c: Confirmation) -> None:
        self.rows[c.confirmation_id] = c

    async def find_by_id(self, cid: str) -> Confirmation | None:
        return self.rows.get(cid)

    async def list_by_thread(self, user_id: str, thread_id: str, limit: int = 20):  # type: ignore[no-untyped-def]
        found = [c for c in self.rows.values() if (c.user_id, c.thread_id) == (user_id, thread_id)]
        return sorted(found, key=lambda c: c.created_at)[-limit:]


class MemOrders:
    def __init__(self) -> None:
        self.rows: dict[str, Order] = {}

    async def save(self, order: Order) -> None:
        self.rows[order.order_id] = order

    async def find_by_id(self, order_id: str) -> Order | None:
        return self.rows.get(order_id)

    async def list_by_user(self, user_id: str, limit: int = 20) -> list[Order]:
        return [o for o in self.rows.values() if o.user_id == user_id][:limit]

    async def next_order_id(self) -> str:
        return f"GBX-{len(self.rows) + 1:06d}"

    async def find_by_idempotency_key(self, key: str) -> Order | None:
        return next((o for o in self.rows.values() if o.idempotency_key == key), None)


POOL = {
    "B01": ItemCandidate(
        item_id="B01", platform="amazon", title="旅行包", price=19.99, currency="USD"
    ),
    "B02": ItemCandidate(
        item_id="B02", platform="amazon", title="洗漱包", price=5.0, currency="USD"
    ),
    "NOPRICE": ItemCandidate(item_id="NOPRICE", platform="amazon", title="没标价", price=None),
}


def _hydrate(ids):  # type: ignore[no-untyped-def]
    return [POOL[i] for i in ids if i in POOL]


ADDR = {
    "recipient_name": "张三",
    "country": "CN",
    "state": "",
    "city": "上海",
    "address_line": "某路 1 号",
    "postal_code": "",
    "phone": "",
}


async def _prepare(crepo: MemConfirmations, thread: str = "t1", items=("B01",)):  # type: ignore[no-untyped-def]
    return await prepare_order_confirmation(
        crepo,
        user_id="u1",
        thread_id=thread,
        lines=[LineRequest(i) for i in items],
        shipping_address=ADDR,
        hydrate=_hydrate,
    )


async def test_prepare_only_records_pending_snapshot() -> None:
    """准备 = 落一条 pending 记录：金额是最小单位整数、hash 非空、30 分钟有效期，**不落订单**。"""
    crepo = MemConfirmations()
    conf = await _prepare(crepo, items=("B01", "B02"))
    assert conf.status == "pending" and conf.action == "create"
    assert conf.payload["total_amount_minor"] == 1999 + 500 and conf.payload["currency"] == "USD"
    assert conf.snapshot_hash and len(conf.snapshot_hash) == 64
    assert timedelta(minutes=29) < conf.expires_at - conf.created_at <= timedelta(minutes=30)
    assert conf.envelope()["buyer_id"] == "u1" and conf.envelope()["expired"] is False


async def test_prepare_rejects_bad_input() -> None:
    crepo = MemConfirmations()
    with pytest.raises(NoCandidateError):
        await _prepare(crepo, items=("NOPE",))
    with pytest.raises(NoCandidateError, match="没有价格"):
        await _prepare(crepo, items=("NOPRICE",))
    with pytest.raises(ConfirmationError, match="缺少 city"):
        await prepare_order_confirmation(
            crepo,
            user_id="u1",
            thread_id="t1",
            lines=[LineRequest("B01")],
            shipping_address={**ADDR, "city": " "},
            hydrate=_hydrate,
        )
    with pytest.raises(ConfirmationError, match="登录"):
        await prepare_order_confirmation(
            crepo,
            user_id="",
            thread_id="t1",
            lines=[LineRequest("B01")],
            shipping_address=ADDR,
            hydrate=_hydrate,
        )


async def test_resolve_approved_places_order_once_and_is_idempotent() -> None:
    """同意 → 按快照落一张 CONFIRMED 订单（不回候选池）；再点一次原样返回、不开第二张单。"""
    crepo, orepo = MemConfirmations(), MemOrders()
    conf = await _prepare(crepo, items=("B01", "B02"))
    done = await resolve_confirmation(
        crepo,
        orepo,
        user_id="u1",
        thread_id="t1",
        confirmation_id=conf.confirmation_id,
        snapshot_hash=conf.snapshot_hash,
        approved=True,
    )
    assert done.status == "approved" and done.result and done.result["status"] == "CONFIRMED"
    order = orepo.rows[done.result["order_id"]]
    assert order.status is OrderStatus.CONFIRMED and order.total().amount_minor == 2499
    assert order.idempotency_key == conf.operation_id
    again = await resolve_confirmation(
        crepo,
        orepo,
        user_id="u1",
        thread_id="t1",
        confirmation_id=conf.confirmation_id,
        snapshot_hash=conf.snapshot_hash,
        approved=True,
    )
    assert again.result == done.result and len(orepo.rows) == 1
    # 已同意再拒绝：不能翻盘
    with pytest.raises(ConfirmationError) as exc:
        await resolve_confirmation(
            crepo,
            orepo,
            user_id="u1",
            thread_id="t1",
            confirmation_id=conf.confirmation_id,
            snapshot_hash=conf.snapshot_hash,
            approved=False,
        )
    assert exc.value.code == "conflict"


async def test_resolve_rejects_hash_mismatch_expired_and_foreign() -> None:
    crepo, orepo = MemConfirmations(), MemOrders()
    conf = await _prepare(crepo)
    kw = dict(user_id="u1", thread_id="t1", confirmation_id=conf.confirmation_id, approved=True)
    with pytest.raises(ConfirmationError) as exc:
        await resolve_confirmation(crepo, orepo, snapshot_hash="deadbeef", **kw)
    assert exc.value.code == "conflict" and not orepo.rows
    # 别人的 / 别的会话 → not_found（同一个码，不泄露存在性）
    with pytest.raises(ConfirmationError) as exc:
        await resolve_confirmation(
            crepo,
            orepo,
            user_id="u2",
            thread_id="t1",
            confirmation_id=conf.confirmation_id,
            snapshot_hash=conf.snapshot_hash,
            approved=True,
        )
    assert exc.value.code == "not_found"
    # 过期：pending 但过了 expires_at → expired，不同意也不拒绝
    conf.expires_at = datetime.now(UTC) - timedelta(seconds=1)
    with pytest.raises(ConfirmationError) as exc:
        await resolve_confirmation(crepo, orepo, snapshot_hash=conf.snapshot_hash, **kw)
    assert exc.value.code == "expired" and conf.envelope()["expired"] is True


async def test_reject_marks_rejected_without_order() -> None:
    crepo, orepo = MemConfirmations(), MemOrders()
    conf = await _prepare(crepo)
    done = await resolve_confirmation(
        crepo,
        orepo,
        user_id="u1",
        thread_id="t1",
        confirmation_id=conf.confirmation_id,
        snapshot_hash=conf.snapshot_hash,
        approved=False,
    )
    assert done.status == "rejected" and done.result is None and not orepo.rows


async def test_cancel_confirmation_flow() -> None:
    """取消：先出卡（订单仍 CONFIRMED），同意后才 CANCELLED 并记原因；已取消的单不能再出卡。"""
    crepo, orepo = MemConfirmations(), MemOrders()
    conf = await _prepare(crepo)
    placed = await resolve_confirmation(
        crepo,
        orepo,
        user_id="u1",
        thread_id="t1",
        confirmation_id=conf.confirmation_id,
        snapshot_hash=conf.snapshot_hash,
        approved=True,
    )
    oid = placed.result["order_id"]  # type: ignore[index]
    with pytest.raises(ConfirmationError, match="原因"):
        await prepare_cancel_confirmation(
            crepo, orepo, user_id="u1", thread_id="t1", order_id=oid, reason=" "
        )
    cancel = await prepare_cancel_confirmation(
        crepo, orepo, user_id="u1", thread_id="t1", order_id=oid, reason="买错了"
    )
    assert cancel.action == "cancel" and orepo.rows[oid].status is OrderStatus.CONFIRMED
    done = await resolve_confirmation(
        crepo,
        orepo,
        user_id="u1",
        thread_id="t1",
        confirmation_id=cancel.confirmation_id,
        snapshot_hash=cancel.snapshot_hash,
        approved=True,
    )
    assert done.result["status"] == "CANCELLED"  # type: ignore[index]
    assert orepo.rows[oid].cancel_reason == "买错了"
    from app.trade.order import OrderStateError

    with pytest.raises(OrderStateError):
        await prepare_cancel_confirmation(
            crepo, orepo, user_id="u1", thread_id="t1", order_id=oid, reason="再取消一次"
        )


async def test_trade_state_block_lists_pending_and_orders_without_address() -> None:
    from app.harness.hooks.context_shaping import render_trade_state_block

    crepo, orepo = MemConfirmations(), MemOrders()
    pending = await _prepare(crepo, items=("B02",))
    conf = await _prepare(crepo)
    await resolve_confirmation(
        crepo,
        orepo,
        user_id="u1",
        thread_id="t1",
        confirmation_id=conf.confirmation_id,
        snapshot_hash=conf.snapshot_hash,
        approved=True,
    )
    state = await trade_state(crepo, orepo, user_id="u1", thread_id="t1")
    assert [c["confirmation_id"] for c in state["pending_confirmations"]] == [
        pending.confirmation_id
    ]
    assert state["orders"][0]["status"] == "CONFIRMED"
    block = render_trade_state_block(state)
    assert "<trade_state>" in block and "洗漱包×1" in block and "GBX-000001" in block
    assert "张三" not in block and "某路" not in block
    assert render_trade_state_block({"pending_confirmations": [], "orders": []}) == ""


def test_ask_user_strips_inline_markdown() -> None:
    from app.tools.ask_user import strip_inline_markdown

    assert (
        strip_inline_markdown("帮你下单 **Casual Daypack**（约 `$26`）？")
        == "帮你下单 Casual Daypack（约 $26）？"
    )
    assert strip_inline_markdown("5 * 3 = 15") == "5 * 3 = 15"
    assert strip_inline_markdown("**__双层__**") == "双层"


@pytest.fixture
async def _client(monkeypatch: pytest.MonkeyPatch, tmp_path):  # type: ignore[no-untyped-def]
    """开着鉴权的 ASGI 客户端，输出根钉到 tmp。"""
    from httpx import ASGITransport, AsyncClient

    import app.api.server as server

    monkeypatch.setenv("AUTH_ENABLED", "true")
    monkeypatch.setenv("JWT_SECRET", "test-secret-not-real")
    monkeypatch.setattr(server, "OUTPUT_ROOT", tmp_path / "output")
    async with AsyncClient(transport=ASGITransport(app=server.app), base_url="http://test") as c:
        yield c, tmp_path / "output"


async def _login(client, username: str):  # type: ignore[no-untyped-def]
    resp = await client.post(
        "/api/auth/register", json={"username": username, "password": "sup3r-secret"}
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["user_id"], {"Authorization": f"Bearer {resp.json()['access_token']}"}


async def test_http_prepare_list_resolve_roundtrip(  # type: ignore[no-untyped-def]
    _client, monkeypatch: pytest.MonkeyPatch
) -> None:
    """表单 → 出卡 → 列表能看到 → 同意落单 → 别人碰不到。

    出卡时任务已结束、登记表为空，候选按 id 回源商品库。"""
    import app.tools._candidates as candidates
    from app.db.accounts import claim_thread
    from app.db.session import init_db, session_factory

    # 候选不落盘：hydrate 登记表未命中时按 id 回源 Qdrant，这里把回源替换成测试商品池。
    monkeypatch.setattr(
        candidates, "_fetch_from_store", lambda ids: [POOL[i] for i in ids if i in POOL]
    )
    client, _out = _client
    await init_db()
    uid, headers = await _login(client, "buyer-a")
    _, other = await _login(client, "buyer-b")
    async with session_factory()() as db:
        await claim_thread(db, "t-http", uid, "下单")
    body = {"items": [{"item_id": "B01", "quantity": 2}], "shipping_address": ADDR}
    r = await client.post("/api/threads/t-http/confirmations/orders", headers=headers, json=body)
    assert r.status_code == 200, r.text
    conf = r.json()
    assert conf["status"] == "pending" and conf["payload"]["total_amount_minor"] == 3998
    # 未登录 401；别人的会话 403
    assert (
        await client.post("/api/threads/t-http/confirmations/orders", json=body)
    ).status_code == 401
    assert (
        await client.post("/api/threads/t-http/confirmations/orders", headers=other, json=body)
    ).status_code == 403
    listed = await client.get("/api/threads/t-http/confirmations", headers=headers)
    assert [c["confirmation_id"] for c in listed.json()["confirmations"]] == [
        conf["confirmation_id"]
    ]
    url = f"/api/threads/t-http/confirmations/{conf['confirmation_id']}/resolve"
    bad = await client.post(url, headers=headers, json={"snapshot_hash": "x", "approved": True})
    assert bad.status_code == 409
    ok = await client.post(
        url, headers=headers, json={"snapshot_hash": conf["snapshot_hash"], "approved": True}
    )
    assert ok.status_code == 200 and ok.json()["result"]["status"] == "CONFIRMED"
    mine = await client.get("/api/orders", headers=headers)
    assert ok.json()["result"]["order_id"] in [o["order_id"] for o in mine.json()["orders"]]
