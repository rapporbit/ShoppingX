"""cancel_order —— 准备取消确认卡（写操作，**终结性**）。

对齐参考项目：本工具**只准备一张取消确认卡**，真正把订单改成 CANCELLED 的是用户在页面上点
「确认取消」（``resolve`` 只走 HTTP，不注册为模型工具）。只有 CONFIRMED 能出取消卡，重复取消
**报错而不是幂等成功**（理由见 ``trade/order.Order.cancel``）。

调用顺序上要求先 query_order，由 ``hooks/sequencing.py`` 的断言把关——它是「有证据才硬拒」
的那类闸：本会话轨迹里确实没出现过 query_order 才拦。
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.api import monitor
from app.api.context import get_user_id
from app.tools._shell import tool
from app.trade.confirmation import ConfirmationError
from app.trade.confirmations import prepare_cancel_confirmation
from app.trade.order import OrderStateError
from app.trade.repository_sql import confirmation_repository, order_repository
from app.trade.usecases import OrderNotFoundError


class CancelOrderOutput(BaseModel):
    """cancel_order 的结构化返回。"""

    confirmation_required: bool = Field(
        default=False, description="True = 已出取消确认卡，等用户在页面上点按钮；本工具不直接取消"
    )
    confirmation_id: str = Field(default="", description="确认记录 id")
    order_id: str = Field(default="", description="待取消的订单号")
    expires_at: str = Field(default="", description="确认卡失效时刻（UTC ISO）")
    note: str = Field(default="", description="给模型与用户的说明")


@tool
async def cancel_order(order_id: str, reason: str) -> CancelOrderOutput:
    """准备取消确认卡（**终结性**：调用后本轮结束）。**不会直接取消**——只生成一张由用户在
    页面上点按钮决议的确认卡。

    何时调用：用户明确要求取消某张订单、且说明了原因。**必须先用 query_order 查到那张单**、
    确认它存在且状态是 CONFIRMED，再调本工具。调完后告诉用户核对页面上的取消确认卡并点击；
    不得声称已取消。
    参数：
      - order_id：要取消的订单号（形如 GBX-000123），从 query_order 的结果里取，不要自己编。
      - reason：用户说明的取消原因（必填；用户没说就先问一句，别替他编）。
    """
    await monitor.report_tool_start("cancel_order", order_id=order_id)
    thread_id = monitor.root_thread_id() or ""
    try:
        conf = await prepare_cancel_confirmation(
            confirmation_repository(),
            order_repository(),
            user_id=get_user_id() or "",
            thread_id=thread_id,
            order_id=order_id,
            reason=reason,
        )
    except OrderNotFoundError as e:
        await monitor.report_tool_end("cancel_order", error=str(e))
        return CancelOrderOutput(note=f"[error] {e}。请先用 query_order 确认订单号。")
    except (OrderStateError, ConfirmationError) as e:
        await monitor.report_tool_end("cancel_order", error=str(e))
        return CancelOrderOutput(note=f"[error] {e}")

    env = conf.envelope()
    await monitor.report_confirmation("required", env, thread_id=thread_id)
    await monitor.report_tool_end("cancel_order", confirmation_id=conf.confirmation_id)
    return CancelOrderOutput(
        confirmation_required=True,
        confirmation_id=conf.confirmation_id,
        order_id=order_id,
        expires_at=env["expires_at"],
        note=(
            f"已生成订单 {order_id} 的取消确认卡（尚未取消）。请告诉用户核对页面上的确认卡"
            "并点击「确认取消」；对话里说「确认」不算数。"
        ),
    )
