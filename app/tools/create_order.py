"""create_order —— 下单（mock 交易域，终结性）。

**两段式**：``confirmed=False`` 只回一张确认卡（买什么、几件、多少钱、寄到哪），不落库；用户
在下一轮明确同意后才 ``confirmed=True`` 真下单。为什么不靠模型自觉：它对「用户说想买」和「用户
确认要买」的区分并不稳，而这里的代价是真落库的一张单。两段式把这件事变成机制——第一段拿不到
任何写库能力，第二段则要求本轮之前确实出现过确认卡。

商品信息按 ``item_id`` 从会话候选登记表取，工具入参不收标题与价格（模型重吐会把价格记错，而这里
记错的是要落库的钱）。
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.api import monitor
from app.api.context import get_thread_id, get_user_id
from app.tools._args import StrListArg
from app.tools._candidates import hydrate
from app.tools._order_guard import mark_preview_shown, preview_shown
from app.tools._shell import tool
from app.trade.address import Address
from app.trade.repository_sql import order_repository
from app.trade.usecases import LineRequest, NoCandidateError, place_order


class CreateOrderOutput(BaseModel):
    """create_order 的结构化返回（前端 OrderCard 按它渲染）。"""

    confirmed: bool = Field(default=False, description="是否已真下单（False = 只是确认卡）")
    order: dict[str, Any] = Field(default_factory=dict, description="订单快照，未确认时为空")
    preview: list[dict[str, Any]] = Field(default_factory=list, description="确认卡里的商品行")
    total_display: str = Field(default="", description="合计（含币种）")
    address: str = Field(default="", description="收货地址摘要")
    note: str = Field(default="", description="给模型与用户的说明")


@tool
async def create_order(
    item_ids: StrListArg,
    recipient: str,
    address_line: str,
    confirmed: bool = False,
    quantities: StrListArg | None = None,
    country: str = "",
    phone: str = "",
) -> CreateOrderOutput:
    """下单（**终结性**：调用后本轮结束）。先出确认卡，用户同意后再真下单。

    何时调用：用户明确表示要买清单里的某几件时。**第一次调用必须 confirmed=False**——先把
    确认卡给用户看；只有用户在随后一轮明确说了「确认 / 下单 / 就买这个」，才用 confirmed=True
    再调一次。
    参数：
      - item_ids：要买的商品 id 列表（必须来自本会话已出现过的候选，不能自己编）。
      - recipient / address_line：收件人与收货地址。用户没给过就先 ask_user 问，别自己编。
      - confirmed：False = 只回确认卡不落库；True = 真下单。
      - quantities：与 item_ids 一一对应的数量，缺省每件 1 个。
      - country / phone：可选。country 不传则从地址文本里解析。
    """
    await monitor.report_tool_start("create_order", item_ids=item_ids, confirmed=confirmed)
    ids = list(item_ids or [])
    qty = [int(q) for q in (quantities or [])]
    lines = [
        LineRequest(item_id=i, quantity=qty[n] if n < len(qty) else 1) for n, i in enumerate(ids)
    ]

    try:
        addr = Address.parse(recipient, address_line, country_hint=country, phone=phone)
    except ValueError as e:
        await monitor.report_tool_end("create_order", error=str(e))
        return CreateOrderOutput(note=f"[error] 收货地址不完整：{e}。请先向用户问清收件人与地址。")

    if not lines:
        return CreateOrderOutput(note="[error] 没有指定要买的商品 id。")

    # 两段式的**机制**那一半：没出过确认卡就当 confirmed=False 处理，退回去先出卡。
    # 退回而不是硬拒——硬拒只会让模型原地再试一次同样的调用，退回则把它推上正确的那条路。
    if confirmed and not preview_shown(ids):
        confirmed = False

    if not confirmed:
        cands = hydrate(ids)
        if not cands:
            return CreateOrderOutput(note="[error] 这些 item_id 不在本会话候选里，无法下单。")
        qty_of = {ln.item_id: ln.quantity for ln in lines}
        preview: list[dict[str, Any]] = [
            {
                "item_id": c.item_id,
                "title": c.title,
                "platform": c.platform,
                "unit_price": c.price,
                "currency": c.currency,
                "quantity": qty_of.get(c.item_id, 1),
            }
            for c in cands
        ]
        total = sum((c.price or 0.0) * qty_of.get(c.item_id, 1) for c in cands)
        currency = cands[0].currency
        mark_preview_shown(ids)
        await monitor.report_order_card(
            "preview",
            {
                "preview": preview,
                "total_display": f"{total:.2f} {currency}",
                "address": addr.masked(),
            },
        )
        await monitor.report_tool_end("create_order", confirmed=False, items=len(preview))
        return CreateOrderOutput(
            confirmed=False,
            preview=preview,
            total_display=f"{total:.2f} {currency}",
            address=addr.masked(),
            note=(
                "这是确认卡，尚未下单。请展示给用户，"
                "等用户明确确认后再以 confirmed=True 调一次。"
            ),
        )

    user_id = get_user_id() or ""
    if not user_id:
        return CreateOrderOutput(note="[error] 匿名会话不能下单，请先登录。")
    try:
        order = await place_order(
            order_repository(),
            user_id=user_id,
            thread_id=get_thread_id() or "",
            lines=lines,
            address=addr,
        )
    except (NoCandidateError, ValueError) as e:
        await monitor.report_tool_end("create_order", error=str(e))
        return CreateOrderOutput(note=f"[error] 下单失败：{e}")

    snap = order.snapshot()
    await monitor.report_order_card("placed", {"order": snap})
    await monitor.report_tool_end("create_order", order_id=order.order_id)
    return CreateOrderOutput(
        confirmed=True,
        order=snap,
        total_display=f"{snap['total']:.2f} {snap['currency']}",
        address=snap["address"],
        note=f"下单成功，订单号 {order.order_id}。",
    )
