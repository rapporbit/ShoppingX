"""仓储端口（Protocol）。

用例只依赖这个协议，不依赖 SQLAlchemy。收益不是「以后能换数据库」（那多半不会发生），而是
**测试里能塞一个内存实现**：订单的状态机、幂等、归属校验这些逻辑本来就与「存在哪」无关，让它们
的测试也与之无关，跑起来就是毫秒级、不用建库、不用清表。
"""

from __future__ import annotations

from typing import Protocol

from app.trade.confirmation import Confirmation
from app.trade.order import Order


class ConfirmationRepository(Protocol):
    """确认记录仓储。实现见 ``repository_sql.SqlConfirmationRepository``。"""

    async def save(self, confirmation: Confirmation) -> None:
        """按 ``confirmation_id`` upsert。"""
        ...

    async def find_by_id(self, confirmation_id: str) -> Confirmation | None: ...

    async def find_by_request_key(self, key: str) -> Confirmation | None:
        """幂等查询：同一轮（run_id）同一份载荷已经出过卡就把那张取回来。

        为什么必须有：整轮重跑是常态（队列消息被 PEL 重投、worker 崩了重领），而出确认卡是
        写操作。没有这道，一次重投就是两张待决议的卡，用户点哪张都对不上另一张。
        """
        ...

    async def list_by_thread(
        self, user_id: str, thread_id: str, limit: int = 20
    ) -> list[Confirmation]:
        """某用户某会话的确认记录，按创建时间**正序**（前端按顺序画、合并时后者覆盖前者）。"""
        ...


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
