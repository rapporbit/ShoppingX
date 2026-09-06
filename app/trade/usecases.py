"""交易域用例：下单 / 查单 / 取消。工具层只做入参强转，业务规则全在这里与聚合里。

三条贯穿本文件的规矩：

1. **候选按 id 取，不收模型重吐的商品信息**。工具入参只有 ``item_id``，标题与价格从会话候选
   登记表 hydrate。理由与 price_compare / shipping_calc 一致（见 `_candidates.hydrate`）：让
   模型把商品信息当参数重吐一遍，它就有机会把价格「记错」——而这里记错的是要落库的钱。
2. **归属校验在用例里，不在 API 里**。工具走的是 Agent 通路，不经过 HTTP 路由；只在路由上
   校验，等于给「模型被诱导去查别人的单」留了一整条没设防的路。
3. **幂等键由内容派生**，不是随机数：随机数每次重试都不同，等于没设。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from app.tools._candidates import hydrate
from app.trade.address import Address
from app.trade.money import Money
from app.trade.order import Order, OrderLine, OrderStateError, OrderStatus
from app.trade.ports import OrderRepository


class OrderNotFoundError(LookupError):
    """订单不存在，或不属于当前用户（**两种情况回同一句话**，见 `_load_owned`）。"""


class NoCandidateError(LookupError):
    """要下单的 item_id 在本会话候选登记表里找不到。"""


@dataclass(frozen=True, slots=True)
class LineRequest:
    """下单请求里的一行：只有 id 与数量。"""

    item_id: str
    quantity: int = 1


def idempotency_key(user_id: str, thread_id: str, lines: list[LineRequest]) -> str:
    """由「谁 + 哪个会话 + 买了哪些东西各几件」派生。

    含 thread_id：同一个人隔天在新会话里再买一次同样的东西，是**两笔**真实订单，不该被幂等
    掉。不含时间戳：那会让同一轮里的重试各自成单，幂等就失效了。
    """
    payload = f"{user_id}|{thread_id}|" + ",".join(
        f"{ln.item_id}x{ln.quantity}" for ln in sorted(lines, key=lambda x: x.item_id)
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:32]


async def _load_owned(repo: OrderRepository, order_id: str, user_id: str) -> Order:
    """取一张**属于该用户**的订单。

    订单不存在与订单不属于你，对外回同一句话：区分开来等于给人一个探测器——挨个试
    ``GBX-000001..999``，凭「无权访问」与「不存在」的差别就能数出平台一共有多少单。
    """
    order = await repo.find_by_id(order_id)
    if order is None or order.user_id != user_id:
        raise OrderNotFoundError(f"找不到订单 {order_id}")
    return order


async def place_order(
    repo: OrderRepository,
    *,
    user_id: str,
    thread_id: str,
    lines: list[LineRequest],
    address: Address,
) -> Order:
    """下单：候选 hydrate → 组装订单行 → 幂等检查 → place → 落库。

    **无库存检查**（CLAUDE.md 的不覆盖项）：商品数据是离线快照，编一个库存数字只会让人以为
    它对得上真实平台。
    """
    if not lines:
        raise ValueError("下单至少要有一件商品")
    key = idempotency_key(user_id, thread_id, lines)
    existing = await repo.find_by_idempotency_key(key)
    if existing is not None:
        return existing  # 重试撞上同一把钥匙 → 把上次那张单原样交回，不再开一张

    wanted = {ln.item_id: ln.quantity for ln in lines}
    candidates = hydrate(list(wanted))
    if len(candidates) != len(wanted):
        missing = sorted(set(wanted) - {c.item_id for c in candidates})
        raise NoCandidateError(f"这些商品不在本会话的候选里：{missing}")

    order_lines = [
        OrderLine(
            platform=c.platform,
            item_id=c.item_id,
            title=c.title,
            # 价格取**原币种报价**：折算成 USD 存会把汇率时点混进订单金额里，而这个域不做汇率
            # 对账。price 缺失（召回没带价）的候选不许下单——一张 0 元的单比下单失败更糟。
            unit_price=Money.from_major(c.price, c.currency),
            quantity=wanted[c.item_id],
            landed_usd=c.landed_usd,
        )
        for c in candidates
        if c.price is not None
    ]
    if len(order_lines) != len(candidates):
        no_price = sorted(c.item_id for c in candidates if c.price is None)
        raise NoCandidateError(f"这些商品没有价格，不能下单：{no_price}")

    order = Order(
        order_id=await repo.next_order_id(),
        user_id=user_id,
        thread_id=thread_id,
        lines=order_lines,
        address=address,
        idempotency_key=key,
    )
    order.place()
    await repo.save(order)
    return order


async def query_orders(
    repo: OrderRepository, *, user_id: str, order_id: str = "", limit: int = 10
) -> list[Order]:
    """查单：给了 ``order_id`` 就查那一张（带归属校验），否则列出该用户最近的几张。"""
    if order_id:
        return [await _load_owned(repo, order_id, user_id)]
    return await repo.list_by_user(user_id, limit=limit)


async def cancel_order(
    repo: OrderRepository, *, user_id: str, order_id: str, reason: str = ""
) -> Order:
    """取消：只有 CONFIRMED 能取消，状态机不合法时把 :class:`OrderStateError` 抛给工具层。"""
    order = await _load_owned(repo, order_id, user_id)
    if order.status is OrderStatus.CANCELLED:
        # 单独给一句更有用的话：模型最常见的错是没 query 就直接 cancel，笼统的「状态不对」
        # 会让它接着乱试。
        raise OrderStateError(f"订单 {order_id} 已经是取消状态，无需重复取消")
    order.cancel(reason)
    await repo.save(order)
    return order
