"""仓储端口（Protocol）。

用例只依赖这个协议，不依赖 SQLAlchemy。收益不是「以后能换数据库」（那多半不会发生），而是
**测试里能塞一个内存实现**：订单的状态机、幂等、归属校验这些逻辑本来就与「存在哪」无关，让它们
的测试也与之无关，跑起来就是毫秒级、不用建库、不用清表。
"""

from __future__ import annotations

from typing import Protocol

from app.trade.order import Order


class OrderRepository(Protocol):
    """订单仓储。实现见 ``repository_sql.SqlOrderRepository``。"""

    async def save(self, order: Order) -> None:
        """落库（新建或更新，按 ``order_id`` upsert）。"""
        ...

    async def find_by_id(self, order_id: str) -> Order | None: ...

    async def list_by_user(self, user_id: str, limit: int = 20) -> list[Order]:
        """按创建时间倒序列出某用户的订单。"""
        ...

    async def next_order_id(self) -> str:
        """分配一个新订单号（``GBX-000123``）。"""
        ...

    async def find_by_idempotency_key(self, key: str) -> Order | None:
        """幂等查询：同一个 key 已经下过单就把那张单取回来。

        为什么必须有：模型重试是常态（网络抖动、harness 的 retry_nudge、用户连点两次确认），
        而下单不是只读操作。没有这道，一次重试就是两张单、两笔钱。
        """
        ...
