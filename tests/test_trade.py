"""交易域（批 1 / 7.2）：值对象、状态机、幂等、归属、确认门、顺序闸。

仓储用**内存实现**（`ports.OrderRepository` 的另一实现）：状态机与幂等这些规则与「存在哪」无关，
让它们的测试也与之无关——毫秒级、不建库、不清表。SQL 实现另有一条端到端用例兜住映射。
"""

import pytest

from app.trade.address import Address
from app.trade.money import CurrencyMismatchError, Money, minor_factor
from app.trade.order import Order, OrderLine, OrderStateError, OrderStatus
from app.trade.usecases import (
    LineRequest,
    NoCandidateError,
    OrderNotFoundError,
    cancel_order,
    idempotency_key,
    place_order,
    query_orders,
)


class InMemoryRepo:
    """`OrderRepository` 的内存实现。"""

    def __init__(self) -> None:
        self.rows: dict[str, Order] = {}

    async def save(self, order: Order) -> None:
        self.rows[order.order_id] = order

    async def find_by_id(self, order_id: str) -> Order | None:
        return self.rows.get(order_id)

    async def list_by_user(self, user_id: str, limit: int = 20) -> list[Order]:
        found = [o for o in self.rows.values() if o.user_id == user_id]
        return sorted(found, key=lambda o: o.created_at, reverse=True)[:limit]

    async def next_order_id(self) -> str:
        return f"GBX-{len(self.rows) + 1:06d}"

    async def find_by_idempotency_key(self, key: str) -> Order | None:
        return next((o for o in self.rows.values() if o.idempotency_key == key), None)


def _addr() -> Address:
    return Address(recipient="张三", line="上海市某路 1 号", country="CN")


def _order(**kw: object) -> Order:
    defaults = dict(
        order_id="GBX-000001",
        user_id="u1",
        thread_id="t1",
        lines=[
            OrderLine(
                platform="amazon",
                item_id="B01",
                title="旅行包",
                unit_price=Money.from_major(19.99, "USD"),
            )
        ],
        address=_addr(),
    )
    defaults.update(kw)
    return Order(**defaults)  # type: ignore[arg-type]


# ---------- Money ----------


def test_money_rounds_half_up_not_bankers() -> None:
    """钱上没人期待银行家舍入（`round(2.675, 2)` 给 2.67）。"""
    assert Money.from_major(2.675, "USD").amount_minor == 268
    assert Money.from_major("0.1", "USD").add(Money.from_major("0.2", "USD")).amount_minor == 30


def test_zero_decimal_currency_factor() -> None:
    """日元没有小数位——拿 100 去乘一单就差 100 倍。"""
    assert minor_factor("JPY") == 1
    assert minor_factor("USD") == 100
    assert Money.from_major(1500, "JPY").amount_minor == 1500
    assert float(Money.from_major(1500, "JPY").to_major()) == 1500.0


def test_money_rejects_cross_currency_add() -> None:
    with pytest.raises(CurrencyMismatchError):
        Money.from_major(1, "USD").add(Money.from_major(1, "EUR"))


# ---------- 状态机 ----------


def test_place_then_cancel_transitions() -> None:
    order = _order()
    assert order.status is OrderStatus.DRAFT
    order.place()
    assert order.status is OrderStatus.CONFIRMED and order.confirmed_at is not None
    order.cancel("不想要了")
    assert order.status is OrderStatus.CANCELLED and order.cancel_reason == "不想要了"


def test_double_cancel_raises_not_silently_succeeds() -> None:
    """重复取消要报错——静默成功会让模型永远学不会先查单。"""
    order = _order()
    order.place()
    order.cancel()
    with pytest.raises(OrderStateError):
        order.cancel()


def test_cancel_before_place_raises() -> None:
    with pytest.raises(OrderStateError):
        _order().cancel()


def test_total_sums_lines_and_rejects_mixed_currency() -> None:
    order = _order(
        lines=[
            OrderLine("amazon", "B01", "包", Money.from_major(19.99, "USD"), quantity=2),
            OrderLine("amazon", "B02", "袋", Money.from_major(5.01, "USD")),
        ]
    )
    assert order.total().amount_minor == 19_99 * 2 + 5_01
    with pytest.raises(ValueError, match="混币种"):
        _order(
            lines=[
                OrderLine("amazon", "B01", "包", Money.from_major(1, "USD")),
                OrderLine("ebay", "B02", "袋", Money.from_major(1, "EUR")),
            ]
        )


# ---------- 用例：幂等 / 归属 / 候选 ----------


@pytest.fixture
def _candidates(monkeypatch: pytest.MonkeyPatch) -> None:
    """把候选登记表换成固定两件商品（下单只按 id 取，不收模型重吐的价格）。"""
    from app.tools.schemas import ItemCandidate

    pool = {
        "B01": ItemCandidate(
            item_id="B01", platform="amazon", title="旅行包", price=19.99, currency="USD"
        ),
        "B02": ItemCandidate(
            item_id="B02", platform="amazon", title="洗漱包", price=5.0, currency="USD"
        ),
        "NOPRICE": ItemCandidate(
            item_id="NOPRICE", platform="amazon", title="没标价", price=None, currency="USD"
        ),
    }
    monkeypatch.setattr(
        "app.trade.usecases.hydrate", lambda ids: [pool[i] for i in ids if i in pool]
    )


async def test_idempotency_key_is_content_derived_and_order_insensitive() -> None:
    """同一组商品换个顺序是同一把钥匙；换会话则不是（隔天再买一次是两笔真实订单）。"""
    a = idempotency_key("u1", "t1", [LineRequest("B01"), LineRequest("B02")])
    b = idempotency_key("u1", "t1", [LineRequest("B02"), LineRequest("B01")])
    c = idempotency_key("u1", "t2", [LineRequest("B01"), LineRequest("B02")])
    assert a == b and a != c


@pytest.mark.usefixtures("_candidates")
async def test_place_order_is_idempotent_on_retry() -> None:
    """重试撞同一把钥匙 → 取回上次那张单，不再开一张。模型重试是常态。"""
    repo = InMemoryRepo()
    kw = dict(user_id="u1", thread_id="t1", lines=[LineRequest("B01")], address=_addr())
    first = await place_order(repo, **kw)  # type: ignore[arg-type]
    second = await place_order(repo, **kw)  # type: ignore[arg-type]
    assert first.order_id == second.order_id
    assert len(repo.rows) == 1


@pytest.mark.usefixtures("_candidates")
async def test_place_order_takes_price_from_registry_not_caller() -> None:
    repo = InMemoryRepo()
    order = await place_order(
        repo,
        user_id="u1",
        thread_id="t1",
        lines=[LineRequest("B01", quantity=2), LineRequest("B02")],
        address=_addr(),
    )
    assert order.status is OrderStatus.CONFIRMED
    assert order.total().amount_minor == 19_99 * 2 + 5_00


@pytest.mark.usefixtures("_candidates")
async def test_place_order_rejects_unknown_or_priceless_items() -> None:
    """编出来的 id 与没标价的候选都不许下单——一张 0 元的单比下单失败更糟。"""
    repo = InMemoryRepo()
    with pytest.raises(NoCandidateError):
        await place_order(
            repo, user_id="u1", thread_id="t1", lines=[LineRequest("造的")], address=_addr()
        )
    with pytest.raises(NoCandidateError, match="没有价格"):
        await place_order(
            repo, user_id="u1", thread_id="t1", lines=[LineRequest("NOPRICE")], address=_addr()
        )


@pytest.mark.usefixtures("_candidates")
async def test_other_users_order_is_indistinguishable_from_missing() -> None:
    """越权查单与订单不存在回同一句话——区分开来等于给人一个订单号探测器。"""
    repo = InMemoryRepo()
    order = await place_order(
        repo, user_id="u1", thread_id="t1", lines=[LineRequest("B01")], address=_addr()
    )
    with pytest.raises(OrderNotFoundError) as owned:
        await query_orders(repo, user_id="u2", order_id=order.order_id)
    with pytest.raises(OrderNotFoundError) as missing:
        await query_orders(repo, user_id="u2", order_id="GBX-999999")
    assert str(owned.value).replace(order.order_id, "X") == str(missing.value).replace(
        "GBX-999999", "X"
    )


@pytest.mark.usefixtures("_candidates")
async def test_cancel_requires_ownership_and_confirmed_status() -> None:
    repo = InMemoryRepo()
    order = await place_order(
        repo, user_id="u1", thread_id="t1", lines=[LineRequest("B01")], address=_addr()
    )
    with pytest.raises(OrderNotFoundError):
        await cancel_order(repo, user_id="u2", order_id=order.order_id)
    cancelled = await cancel_order(repo, user_id="u1", order_id=order.order_id, reason="改主意了")
    assert cancelled.status is OrderStatus.CANCELLED
    with pytest.raises(OrderStateError, match="已经是取消状态"):
        await cancel_order(repo, user_id="u1", order_id=order.order_id)


# ---------- 工具层：确认门 / 顺序闸 / SQL 往返 ----------


@pytest.mark.usefixtures("_candidates")
async def test_create_order_first_call_only_previews(monkeypatch: pytest.MonkeyPatch, tmp_path):  # type: ignore[no-untyped-def]
    """confirmed=False 只出确认卡、不落库；且**第一次就传 True 也照样先出卡**。

    后半条是两段式的机制那一半：约定挡不住模型直接传 confirmed=True，会话级确认门挡得住。
    """
    from app.tools import create_order as mod
    from app.tools._order_guard import reset_order_guard
    from app.utils.thread_ctx import thread_scope

    monkeypatch.setattr("app.tools.create_order.hydrate", _pool_hydrate)
    saved: list[object] = []
    monkeypatch.setattr(
        "app.tools.create_order.place_order",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("不该落库")),
    )
    with thread_scope("t-trade", tmp_path):
        reset_order_guard()
        out = await mod.create_order.ainvoke(
            {
                "item_ids": ["B01"],
                "recipient": "张三",
                "address_line": "上海市某路 1 号",
                "confirmed": True,  # 跳过确认卡的企图
            }
        )
    assert out.confirmed is False
    assert out.preview and out.preview[0]["item_id"] == "B01"
    assert not saved


@pytest.mark.usefixtures("_candidates")
async def test_create_order_confirms_after_preview(monkeypatch: pytest.MonkeyPatch, tmp_path):  # type: ignore[no-untyped-def]
    """出过卡之后，confirmed=True 才真落库。"""
    from app.tools import create_order as mod
    from app.tools._order_guard import reset_order_guard
    from app.utils.thread_ctx import thread_scope

    repo = InMemoryRepo()
    monkeypatch.setattr("app.tools.create_order.hydrate", _pool_hydrate)
    monkeypatch.setattr("app.tools.create_order.order_repository", lambda **kw: repo)
    monkeypatch.setattr("app.tools.create_order.get_user_id", lambda: "u1")
    args = {"item_ids": ["B01"], "recipient": "张三", "address_line": "上海市某路 1 号"}
    with thread_scope("t-trade2", tmp_path):
        reset_order_guard()
        await mod.create_order.ainvoke({**args, "confirmed": False})
        out = await mod.create_order.ainvoke({**args, "confirmed": True})
    assert out.confirmed is True and out.order["status"] == "CONFIRMED"
    assert len(repo.rows) == 1


def _pool_hydrate(ids):  # type: ignore[no-untyped-def]
    from app.tools.schemas import ItemCandidate

    pool = {
        "B01": ItemCandidate(
            item_id="B01", platform="amazon", title="旅行包", price=19.99, currency="USD"
        )
    }
    return [pool[i] for i in ids if i in pool]


async def test_cancel_without_query_is_hard_rejected() -> None:
    """顺序闸是**硬拒**：没查就取消，工具压根不执行。"""
    from app.harness.hooks.tool_gates import check_trade_sequence
    from app.harness.middleware import HookRejectSignal

    with pytest.raises(HookRejectSignal):
        await check_trade_sequence({"tool_name": "cancel_order", "called_tools": set()})
    # 查过了就放行——判据是「查过」，不是「查到了什么」
    assert (
        await check_trade_sequence(
            {"tool_name": "cancel_order", "called_tools": {"query_order"}}
        )
        is None
    )


@pytest.mark.usefixtures("_candidates")
async def test_sql_repository_round_trip() -> None:
    """SQL 实现的映射往返：Money 拆成两列再拼回来，状态与地址不失真。"""
    from app.db.session import init_db
    from app.trade.repository_sql import SqlOrderRepository

    await init_db()
    repo = SqlOrderRepository()
    order = await place_order(
        repo,
        user_id="u-sql",
        thread_id="t-sql",
        lines=[LineRequest("B01", quantity=2)],
        address=_addr(),
    )
    got = await repo.find_by_id(order.order_id)
    assert got is not None
    assert got.status is OrderStatus.CONFIRMED
    assert got.total().amount_minor == 19_99 * 2
    assert got.address.country == "CN"
    assert [o.order_id for o in await repo.list_by_user("u-sql")] == [order.order_id]
