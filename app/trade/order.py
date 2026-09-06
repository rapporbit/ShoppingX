"""订单聚合与状态机。

状态机只有三态：``DRAFT → CONFIRMED → CANCELLED``。没有 PAID / SHIPPED / COMPLETED——本仓不做
支付与物流，编出那些状态却没有任何东西会推动它们流转，属于「看着像真的」的假实现。

**不变量都写在聚合里，不写在用例里**：`cancel()` 只接受 CONFIRMED、金额由订单行汇总而不是外部
传入。用例层可以被绕过（工具直接构造、脚本直接调仓储），聚合的方法绕不过去。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from app.trade.address import Address
from app.trade.money import Money


class OrderStatus(StrEnum):
    DRAFT = "DRAFT"
    CONFIRMED = "CONFIRMED"
    CANCELLED = "CANCELLED"


class OrderStateError(RuntimeError):
    """状态机不允许的迁移（如取消一张已取消的单）。"""


def _now() -> datetime:
    return datetime.now(UTC)


@dataclass(slots=True)
class OrderLine:
    """一条订单行：下单那一刻的**快照**。

    标题、单价、到手价全部按值存下来，不存「指向候选池的引用」：候选池是会话级的（`output/
    <thread_id>/candidates.json`），会话一过就没了；而订单要能在三个月后查出来还显示得出买了
    什么。这也是为什么取消一张老单不需要候选池还在。
    """

    platform: str
    item_id: str
    title: str
    unit_price: Money
    quantity: int = 1
    landed_usd: float | None = None

    def subtotal(self) -> Money:
        return self.unit_price.multiply(self.quantity)


@dataclass(slots=True)
class Order:
    """订单聚合。``order_id`` 形如 ``GBX-000123``。"""

    order_id: str
    user_id: str
    thread_id: str
    lines: list[OrderLine]
    address: Address
    status: OrderStatus = OrderStatus.DRAFT
    idempotency_key: str = ""
    created_at: datetime = field(default_factory=_now)
    confirmed_at: datetime | None = None
    cancelled_at: datetime | None = None
    cancel_reason: str = ""

    def __post_init__(self) -> None:
        if not self.lines:
            raise ValueError("订单至少要有一行")
        currencies = {line.unit_price.currency for line in self.lines}
        if len(currencies) > 1:
            # 跨币种订单要么按某个汇率折算（那就得存汇率与折算时点），要么拆成多单。两条路都不是
            # mock 交易域该背的复杂度，所以在入口挡住——比存一个说不清怎么来的总额诚实。
            raise ValueError(f"一张订单不能混币种：{sorted(currencies)}")

    @property
    def currency(self) -> str:
        return self.lines[0].unit_price.currency

    def total(self) -> Money:
        total = Money(0, self.currency)
        for line in self.lines:
            total = total.add(line.subtotal())
        return total

    def place(self) -> None:
        """下单。本域没有支付环节，所以直接 CONFIRMED。"""
        if self.status is not OrderStatus.DRAFT:
            raise OrderStateError(f"只有 DRAFT 能下单，当前 {self.status.value}")
        self.status = OrderStatus.CONFIRMED
        self.confirmed_at = _now()

    def cancel(self, reason: str = "") -> None:
        """取消。**只有 CONFIRMED 可取消**——重复取消要报错而不是幂等地返回成功。

        理由：模型很擅长「没查就直接取消」。如果重复取消静默成功，它就永远学不会先 query_order；
        而用户看到的是「取消成功」，实际那张单可能早在上一轮就被取消了，或者压根不存在。
        """
        if self.status is not OrderStatus.CONFIRMED:
            raise OrderStateError(f"只有 CONFIRMED 能取消，当前 {self.status.value}")
        self.status = OrderStatus.CANCELLED
        self.cancelled_at = _now()
        self.cancel_reason = reason.strip()

    def snapshot(self) -> dict[str, Any]:
        """给工具结果 / API / 前端商品卡用的一份纯数据视图。"""
        return {
            "order_id": self.order_id,
            "status": self.status.value,
            "currency": self.currency,
            "total": float(self.total().to_major()),
            "address": self.address.masked(),
            "created_at": self.created_at.isoformat(),
            "cancel_reason": self.cancel_reason,
            "lines": [
                {
                    "platform": line.platform,
                    "item_id": line.item_id,
                    "title": line.title,
                    "unit_price": float(line.unit_price.to_major()),
                    "quantity": line.quantity,
                    "landed_usd": line.landed_usd,
                }
                for line in self.lines
            ],
        }
