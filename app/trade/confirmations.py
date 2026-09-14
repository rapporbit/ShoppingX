"""确认记录用例：准备（下单 / 取消）、决议、列表、给模型的交易状态。

对齐参考项目 ``confirmation_service.py`` 的分工：``prepare_*`` 是模型工具与前端表单共用的入口，
``resolve`` **只供用户动作入口（HTTP）调用、不注册为模型工具**。approved 才真调
:func:`place_order` / :func:`cancel_order`，订单幂等键 = ``operation_id``，同一张卡点两次只落
一张单。

**不做库存与可配送国校验**（参考项目有）：本仓商品是离线快照，没有库存与 ``ships_to`` 字段，
编出来的校验只会把合法地址拒掉。
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from typing import Any

from app.trade.address import Address
from app.trade.confirmation import (
    CONFIRMATION_TTL,
    Confirmation,
    ConfirmationError,
    new_confirmation_id,
    new_operation_id,
    snapshot_hash,
)
from app.trade.money import Money
from app.trade.order import Order, OrderLine, OrderStateError, OrderStatus
from app.trade.ports import ConfirmationRepository, OrderRepository
from app.trade.usecases import LineRequest, NoCandidateError, _load_owned

ADDRESS_REQUIRED = ("recipient_name", "country", "city", "address_line")
ADDRESS_FIELDS = (
    "recipient_name",
    "country",
    "state",
    "city",
    "address_line",
    "postal_code",
    "phone",
)


def _now() -> datetime:
    return datetime.now(UTC)


def normalize_address(raw: dict[str, Any]) -> dict[str, str]:
    """收货地址：七个字段全转成去空白的字符串，四个必填非空。国家解析走 :class:`Address`。"""
    if not isinstance(raw, dict):
        raise ConfirmationError("收货地址格式无效")
    out: dict[str, str] = {}
    for key in ADDRESS_FIELDS:
        value = raw.get(key, "")
        if value is None:
            value = ""
        if not isinstance(value, str):
            raise ConfirmationError(f"收货地址字段 {key} 必须是文字")
        out[key] = value.strip()
    for key in ADDRESS_REQUIRED:
        if not out[key]:
            raise ConfirmationError(f"收货地址缺少 {key}")
    return out


def address_from_shipping(shipping: dict[str, str]) -> Address:
    """结构化地址 → 订单聚合用的 :class:`Address`（地址行由省 / 市 / 街道 / 邮编拼成一行）。"""
    line = " ".join(
        p
        for p in (
            shipping["state"],
            shipping["city"],
            shipping["address_line"],
            shipping["postal_code"],
        )
        if p
    )
    return Address.parse(
        shipping["recipient_name"], line, country_hint=shipping["country"], phone=shipping["phone"]
    )


def _lines_from_candidates(
    cands: Iterable[Any], quantities: dict[str, int]
) -> list[dict[str, Any]]:
    lines: list[dict[str, Any]] = []
    for c in cands:
        if c.price is None:
            raise NoCandidateError(f"这件商品没有价格，不能下单：{c.item_id}")
        unit = Money.from_major(c.price, c.currency)
        lines.append(
            {
                "platform": c.platform,
                "item_id": c.item_id,
                "title": c.title,
                "unit_price_minor": unit.amount_minor,
                "currency": unit.currency,
                "quantity": quantities.get(c.item_id, 1),
                "landed_usd": c.landed_usd,
            }
        )
    return lines


def _total(lines: list[dict[str, Any]]) -> tuple[int, str]:
    currencies = {ln["currency"] for ln in lines}
    if len(currencies) > 1:
        raise ConfirmationError(f"一张订单不能混币种：{sorted(currencies)}")
    return sum(int(ln["unit_price_minor"]) * int(ln["quantity"]) for ln in lines), lines[0][
        "currency"
    ]


async def prepare_order_confirmation(
    repo: ConfirmationRepository,
    *,
    user_id: str,
    thread_id: str,
    lines: list[LineRequest],
    shipping_address: dict[str, Any],
    hydrate: Callable[[list[str]], list[Any]],
) -> Confirmation:
    """下单确认卡：候选 hydrate → 组行 → 落一条 pending 记录。不落订单、不扣任何东西。"""
    if not user_id:
        raise ConfirmationError("匿名会话不能下单，请先登录", code="unauthorized")
    if not lines:
        raise ConfirmationError("下单至少要有一件商品")
    quantities: dict[str, int] = {}
    for ln in lines:
        if type(ln.quantity) is not int or ln.quantity <= 0:
            raise ConfirmationError("订单数量必须为正整数")
        quantities[ln.item_id] = quantities.get(ln.item_id, 0) + ln.quantity
    shipping = normalize_address(shipping_address)
    address_from_shipping(shipping)  # 国家能解析、收件人非空——不通过就在这里炸
    cands = hydrate(list(quantities))
    if len(cands) != len(quantities):
        missing = sorted(set(quantities) - {c.item_id for c in cands})
        raise NoCandidateError(f"这些商品不在本会话的候选里：{missing}")
    items = _lines_from_candidates(cands, quantities)
    total_minor, currency = _total(items)
    payload = {
        "items": items,
        "shipping_address": shipping,
        "total_amount_minor": total_minor,
        "currency": currency,
        "amount_scope": "merchandise_only",
        "order_kind": "purchase_intent",
    }
    return await _save_pending(repo, user_id, thread_id, "create", payload)


async def _save_pending(
    repo: ConfirmationRepository, user_id: str, thread_id: str, action: str, payload: dict[str, Any]
) -> Confirmation:
    now = _now()
    conf = Confirmation(
        confirmation_id=new_confirmation_id(),
        operation_id=new_operation_id(),
        user_id=user_id,
        thread_id=thread_id,
        action=action,  # type: ignore[arg-type]
        payload=payload,
        snapshot_hash=snapshot_hash(action, payload),
        expires_at=now + CONFIRMATION_TTL,
        created_at=now,
    )
    await repo.save(conf)
    return conf


async def prepare_cancel_confirmation(
    repo: ConfirmationRepository,
    orders: OrderRepository,
    *,
    user_id: str,
    thread_id: str,
    order_id: str,
    reason: str,
) -> Confirmation:
    """取消确认卡：先核归属与状态（只有 CONFIRMED 能取消），再落一条 pending 记录。"""
    if not user_id:
        raise ConfirmationError("匿名会话没有订单，无法取消", code="unauthorized")
    reason = (reason or "").strip()
    if not reason:
        raise ConfirmationError("取消需要说明原因")
    order = await _load_owned(orders, order_id, user_id)
    if order.status is not OrderStatus.CONFIRMED:
        raise OrderStateError(f"只有 CONFIRMED 能取消，订单 {order_id} 当前 {order.status.value}")
    payload = {
        "order_id": order.order_id,
        "reason": reason,
        "items": [
            {
                "platform": ln.platform,
                "item_id": ln.item_id,
                "title": ln.title,
                "unit_price_minor": ln.unit_price.amount_minor,
                "currency": ln.unit_price.currency,
                "quantity": ln.quantity,
                "landed_usd": ln.landed_usd,
            }
            for ln in order.lines
        ],
        "shipping_address": _shipping_from_address(order.address),
        "total_amount_minor": order.total().amount_minor,
        "currency": order.currency,
        "amount_scope": "merchandise_only",
        "order_kind": "cancel_intent",
    }
    return await _save_pending(repo, user_id, thread_id, "cancel", payload)


def _shipping_from_address(addr: Address) -> dict[str, str]:
    """订单里的扁平地址 → 卡片用的结构化地址（省市邮编当年没拆，只能落在 address_line）。"""
    return {
        "recipient_name": addr.recipient,
        "country": addr.country,
        "state": "",
        "city": "-",
        "address_line": addr.line,
        "postal_code": "",
        "phone": addr.phone,
    }


async def _load_owned_confirmation(
    repo: ConfirmationRepository, confirmation_id: str, user_id: str, thread_id: str
) -> Confirmation:
    conf = await repo.find_by_id(confirmation_id)
    # 别人的卡与不存在的卡回同一句：不泄露 id 是否存在。
    if conf is None or conf.user_id != user_id or conf.thread_id != thread_id:
        raise ConfirmationError(f"确认记录 {confirmation_id} 不存在", code="not_found")
    return conf


async def resolve_confirmation(
    repo: ConfirmationRepository,
    orders: OrderRepository,
    *,
    user_id: str,
    thread_id: str,
    confirmation_id: str,
    snapshot_hash: str,
    approved: bool,
) -> Confirmation:
    """决议一张卡。**只供用户动作入口调用，不注册为模型工具。**

    幂等：已决议的卡再收到同一决定原样返回，不同决定报错（迟到的第二次点击不能翻盘）。
    过期的 pending 卡不能同意也不能拒绝——它已经不代表任何东西，让前端重新出卡。
    hash 不符 = 页面上那张与库里这张不是同一份快照，拒绝执行。
    """
    if type(approved) is not bool:
        raise ConfirmationError("确认决定必须为布尔值")
    conf = await _load_owned_confirmation(repo, confirmation_id, user_id, thread_id)
    if conf.status != "pending":
        if (conf.status == "approved") == approved:
            return conf
        raise ConfirmationError(
            f"确认记录已{('同意' if conf.status == 'approved' else '拒绝')}，不能改",
            code="conflict",
        )
    if conf.expired():
        raise ConfirmationError("确认卡已过期，请重新生成", code="expired")
    if snapshot_hash != conf.snapshot_hash:
        raise ConfirmationError("确认快照已变化，请刷新后重试", code="conflict")

    if not approved:
        conf.status = "rejected"
        conf.resolved_at = _now()
        await repo.save(conf)
        return conf

    if conf.action == "create":
        order = await _place_from_snapshot(orders, conf)
    else:
        order = await _load_owned(orders, str(conf.payload["order_id"]), user_id)
        if order.status is OrderStatus.CONFIRMED:
            order.cancel(str(conf.payload.get("reason", "")))
            await orders.save(order)
        elif order.status is not OrderStatus.CANCELLED:
            raise OrderStateError(f"只有 CONFIRMED 能取消，当前 {order.status.value}")
    conf.status = "approved"
    conf.resolved_at = _now()
    conf.result = {
        "order_id": order.order_id,
        "status": order.status.value,
        "total_amount_minor": order.total().amount_minor,
        "currency": order.currency,
    }
    await repo.save(conf)
    return conf


async def _place_from_snapshot(orders: OrderRepository, conf: Confirmation) -> Order:
    """按确认卡快照落订单：**不再回候选池 hydrate**——刷新页面后候选池可能已空，而卡上的行
    就是用户看到并同意的那份。幂等键 = operation_id。"""
    existing = await orders.find_by_idempotency_key(conf.operation_id)
    if existing is not None:
        return existing
    payload = conf.payload
    order = Order(
        order_id=await orders.next_order_id(),
        user_id=conf.user_id,
        thread_id=conf.thread_id,
        lines=[
            OrderLine(
                platform=str(ln["platform"]),
                item_id=str(ln["item_id"]),
                title=str(ln["title"]),
                unit_price=Money(int(ln["unit_price_minor"]), str(ln["currency"])),
                quantity=int(ln["quantity"]),
                landed_usd=ln.get("landed_usd"),
            )
            for ln in payload["items"]
        ],
        address=address_from_shipping(payload["shipping_address"]),
        idempotency_key=conf.operation_id,
    )
    order.place()
    await orders.save(order)
    return order


async def list_confirmations(
    repo: ConfirmationRepository, *, user_id: str, thread_id: str, limit: int = 20
) -> list[Confirmation]:
    if type(limit) is not int or not 1 <= limit <= 20:
        raise ConfirmationError("确认列表数量必须为 1 至 20 的整数")
    return await repo.list_by_thread(user_id, thread_id, limit=limit)


async def trade_state(
    repo: ConfirmationRepository, orders: OrderRepository, *, user_id: str, thread_id: str
) -> dict[str, Any]:
    """给模型的精简权威交易状态：待决议的卡 + 本会话产生的订单。**不带地址、不带 hash。**"""
    confs = await repo.list_by_thread(user_id, thread_id, limit=20)
    pending = [
        {
            "confirmation_id": c.confirmation_id,
            "action": c.action,
            "items": [
                {"item_id": ln["item_id"], "title": ln["title"], "quantity": ln["quantity"]}
                for ln in c.payload.get("items", [])
            ],
            "order_id": c.payload.get("order_id"),
        }
        for c in confs
        if c.status == "pending" and not c.expired()
    ]
    order_ids = list(dict.fromkeys(c.result["order_id"] for c in confs if c.result))
    found = [await orders.find_by_id(oid) for oid in order_ids]
    return {
        "pending_confirmations": pending,
        "orders": [
            {
                "order_id": o.order_id,
                "status": o.status.value,
                "total": float(o.total().to_major()),
                "currency": o.currency,
                "items": [
                    {"item_id": ln.item_id, "title": ln.title, "quantity": ln.quantity}
                    for ln in o.lines
                ],
            }
            for o in found
            if o is not None
        ],
    }
