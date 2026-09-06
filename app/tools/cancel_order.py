"""cancel_order —— 取消订单（写操作，**终结性**）。

只有 CONFIRMED 能取消，重复取消**报错而不是幂等成功**（理由见 ``trade/order.Order.cancel``）：
静默成功会让模型永远学不会先 query_order，而用户看到的「取消成功」可能对应着一张早就取消掉的、
甚至压根不属于他的单。

调用顺序上要求先 query_order，由 ``hooks/step_validator.py`` 的 sequencing 断言把关——它是
「有证据才硬拒」的那类闸：本会话轨迹里确实没出现过 query_order 才拦。
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.api import monitor
from app.api.context import get_user_id
from app.tools._shell import tool
from app.trade.order import OrderStateError
from app.trade.repository_sql import order_repository
from app.trade.usecases import OrderNotFoundError
from app.trade.usecases import cancel_order as _cancel_order


class CancelOrderOutput(BaseModel):
    """cancel_order 的结构化返回。"""

    cancelled: bool = Field(default=False, description="是否成功取消")
    order: dict[str, Any] = Field(default_factory=dict, description="取消后的订单快照")
    note: str = Field(default="", description="给模型与用户的说明")


@tool
async def cancel_order(order_id: str, reason: str = "") -> CancelOrderOutput:
    """取消一张订单（**终结性**：调用后本轮结束）。

    何时调用：用户明确要求取消某张订单时。**必须先用 query_order 查到那张单**、确认它存在且
    状态是 CONFIRMED，再调本工具——没查就取消，取消错了是真的改了库里的状态。
    参数：
      - order_id：要取消的订单号（形如 GBX-000123），从 query_order 的结果里取，不要自己编。
      - reason：可选。用户说明的取消原因，会记进订单。
    """
    await monitor.report_tool_start("cancel_order", order_id=order_id)
    user_id = get_user_id() or ""
    if not user_id:
        return CancelOrderOutput(note="[error] 匿名会话没有订单，无法取消。")
    try:
        order = await _cancel_order(
            order_repository(), user_id=user_id, order_id=order_id, reason=reason
        )
    except OrderNotFoundError as e:
        await monitor.report_tool_end("cancel_order", error=str(e))
        return CancelOrderOutput(note=f"[error] {e}。请先用 query_order 确认订单号。")
    except OrderStateError as e:
        await monitor.report_tool_end("cancel_order", error=str(e))
        return CancelOrderOutput(note=f"[error] {e}")

    await monitor.report_order_card("cancelled", {"order": order.snapshot()})
    await monitor.report_tool_end("cancel_order", order_id=order.order_id)
    return CancelOrderOutput(
        cancelled=True,
        order=order.snapshot(),
        note=f"订单 {order.order_id} 已取消。",
    )
