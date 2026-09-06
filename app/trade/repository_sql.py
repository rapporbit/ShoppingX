"""``OrderRepository`` 的 SQLAlchemy 实现。

映射两侧的形状不同，转换全收在本文件：聚合侧金额是 :class:`Money`（币种 + 最小单位整数）、
地址是值对象；表侧是裸列与 JSON。别把这层转换漏到用例里——那样每加一个用例就要重写一遍
「Money 怎么变成两列」。
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import func, select

from app.db.models import OrderLineRow, OrderRow
from app.db.session import session_factory
from app.trade.address import Address
from app.trade.money import Money
from app.trade.order import Order, OrderLine, OrderStatus

ORDER_ID_PREFIX = "GBX-"


def _to_domain(row: OrderRow) -> Order:
    return Order(
        order_id=row.order_id,
        user_id=row.user_id,
        thread_id=row.thread_id,
        lines=[
            OrderLine(
                platform=line.platform,
                item_id=line.item_id,
                title=line.title,
                unit_price=Money(line.unit_price_minor, line.currency),
                quantity=line.quantity,
                landed_usd=line.landed_usd,
            )
            for line in row.lines
        ],
        address=Address.from_dict(dict(row.address_json or {})),
        status=OrderStatus(row.status),
        idempotency_key=row.idempotency_key,
        created_at=row.created_at,
        confirmed_at=row.confirmed_at,
        cancelled_at=row.cancelled_at,
        cancel_reason=row.cancel_reason,
    )


class SqlOrderRepository:
    """订单仓储（SQLite / 任何 SQLAlchemy 后端）。"""

    async def save(self, order: Order) -> None:
        """按 ``order_id`` upsert，订单行整批重写。

        行整批重写而不是逐行 diff：一张单最多几行，diff 的复杂度换不来任何东西，还容易在
        「取消后又改行」这类路径上留下半新半旧的行。
        """
        async with session_factory()() as db:
            row = await db.get(OrderRow, order.order_id)
            if row is None:
                row = OrderRow(order_id=order.order_id, created_at=order.created_at)
                db.add(row)
            row.user_id = order.user_id
            row.thread_id = order.thread_id
            row.status = order.status.value
            row.currency = order.currency
            row.total_minor = order.total().amount_minor
            row.address_json = order.address.to_dict()
            row.idempotency_key = order.idempotency_key
            row.confirmed_at = order.confirmed_at
            row.cancelled_at = order.cancelled_at
            row.cancel_reason = order.cancel_reason
            row.lines = [
                OrderLineRow(
                    platform=line.platform,
                    item_id=line.item_id,
                    title=line.title,
                    unit_price_minor=line.unit_price.amount_minor,
                    currency=line.unit_price.currency,
                    quantity=line.quantity,
                    landed_usd=line.landed_usd,
                )
                for line in order.lines
            ]
            await db.commit()

    async def find_by_id(self, order_id: str) -> Order | None:
        async with session_factory()() as db:
            row = await db.get(OrderRow, order_id)
            return _to_domain(row) if row else None

    async def list_by_user(self, user_id: str, limit: int = 20) -> list[Order]:
        async with session_factory()() as db:
            rows = await db.execute(
                select(OrderRow)
                .where(OrderRow.user_id == user_id)
                .order_by(OrderRow.created_at.desc())
                .limit(limit)
            )
            return [_to_domain(r) for r in rows.scalars().all()]

    async def next_order_id(self) -> str:
        """``GBX-000123``：按现有行数 + 1 编号。

        单机单 worker（见 db/models 的 SQLite 说明）下够用；真并发写时靠 ``orders`` 主键冲突
        兜底——冲突了就是重试一次拿下一个号，不会串号。
        """
        async with session_factory()() as db:
            total = await db.scalar(select(func.count()).select_from(OrderRow))
            return f"{ORDER_ID_PREFIX}{int(total or 0) + 1:06d}"

    async def find_by_idempotency_key(self, key: str) -> Order | None:
        async with session_factory()() as db:
            row = await db.execute(select(OrderRow).where(OrderRow.idempotency_key == key))
            found = row.scalars().first()
            return _to_domain(found) if found else None


def order_repository(**_kw: Any) -> SqlOrderRepository:
    """默认仓储的取用口（测试里换成内存实现的接缝）。"""
    return SqlOrderRepository()
