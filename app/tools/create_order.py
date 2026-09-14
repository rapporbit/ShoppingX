"""create_order —— 准备下单确认卡（mock 交易域，终结性）。

对齐参考项目：**本工具只准备一张服务端确认卡，不下单**。决议（同意 / 拒绝）只有用户在页面上
点按钮这一条路（``POST /api/threads/{id}/confirmations/{cid}/resolve``），模型说「用户已确认」
不算数、也没有 ``confirmed=True`` 可调。为什么不靠模型自觉：它对「用户说想买」和「用户确认要买」
的区分并不稳，而这里的代价是真落库的一张单。确认记录持久化在 ``trade_confirmations`` 表
（见 :mod:`app.trade.confirmations`），刷新页面、重启进程都还在。

商品信息按 ``item_id`` 从会话候选登记表取，工具入参不收标题与价格（模型重吐会把价格记错，
而这里记错的是要落库的钱）。
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.api import monitor
from app.api.context import get_user_id
from app.tools._args import StrListArg
from app.tools._candidates import hydrate
from app.tools._shell import tool
from app.trade.confirmation import ConfirmationError
from app.trade.confirmations import prepare_order_confirmation
from app.trade.money import Money
from app.trade.repository_sql import confirmation_repository
from app.trade.usecases import LineRequest, NoCandidateError


class CreateOrderOutput(BaseModel):
    """create_order 的结构化返回。"""

    confirmation_required: bool = Field(
        default=False, description="True = 已出确认卡，等用户在页面上点按钮；本工具永远不直接下单"
    )
    confirmation_id: str = Field(default="", description="确认记录 id")
    items: list[dict[str, Any]] = Field(default_factory=list, description="确认卡里的商品行")
    total_display: str = Field(default="", description="商品合计（含币种，不含税运）")
    address: str = Field(default="", description="收货地址摘要")
    expires_at: str = Field(default="", description="确认卡失效时刻（UTC ISO）")
    note: str = Field(default="", description="给模型与用户的说明")


@tool
async def create_order(
    item_ids: StrListArg,
    recipient_name: str,
    country: str,
    city: str,
    address_line: str,
    quantities: StrListArg | None = None,
    state: str = "",
    postal_code: str = "",
    phone: str = "",
) -> CreateOrderOutput:
    """准备下单确认卡（**终结性**：调用后本轮结束）。**不会下单**——只生成一张由用户在页面上
    点按钮决议的确认卡。

    何时调用：用户明确表示要买清单里的某几件、且收件人 / 国家 / 城市 / 详细地址都齐了。
    缺任何一项都不要调，先回一条消息让用户补（不要编造收货信息）。
    调完后告诉用户「请核对页面上的确认卡并点击确认」；**不得声称已下单**。用户在对话里说
    「确认」不能代替页面上的按钮——此时不要再调本工具，回一句请他点卡片即可。
    参数：
      - item_ids：要买的商品 id 列表（必须来自本会话已出现过的候选，不能自己编）。
      - recipient_name / country / city / address_line：收件人、国家或地区、城市、详细地址（必填）。
      - quantities：与 item_ids 一一对应的数量，缺省每件 1 个。
      - state / postal_code / phone：省或州、邮编、电话（可选）。
    """
    await monitor.report_tool_start("create_order", item_ids=item_ids)
    ids = list(item_ids or [])
    qty = [int(q) for q in (quantities or [])]
    lines = [
        LineRequest(item_id=i, quantity=qty[n] if n < len(qty) else 1) for n, i in enumerate(ids)
    ]
    shipping = {
        "recipient_name": recipient_name,
        "country": country,
        "state": state,
        "city": city,
        "address_line": address_line,
        "postal_code": postal_code,
        "phone": phone,
    }
    thread_id = monitor.root_thread_id() or ""
    try:
        conf = await prepare_order_confirmation(
            confirmation_repository(),
            user_id=get_user_id() or "",
            thread_id=thread_id,
            lines=lines,
            shipping_address=shipping,
            hydrate=hydrate,
        )
    except (ConfirmationError, NoCandidateError, ValueError) as e:
        await monitor.report_tool_end("create_order", error=str(e))
        return CreateOrderOutput(note=f"[error] 无法生成确认卡：{e}")

    env = conf.envelope()
    await monitor.report_confirmation("required", env, thread_id=thread_id)
    await monitor.report_tool_end("create_order", confirmation_id=conf.confirmation_id)
    payload = conf.payload
    total = Money(int(payload["total_amount_minor"]), str(payload["currency"]))
    addr = payload["shipping_address"]
    return CreateOrderOutput(
        confirmation_required=True,
        confirmation_id=conf.confirmation_id,
        items=[
            {"item_id": ln["item_id"], "title": ln["title"], "quantity": ln["quantity"]}
            for ln in payload["items"]
        ],
        total_display=str(total),
        address=f"{addr['recipient_name']}｜{addr['city']}｜{addr['country']}",
        expires_at=env["expires_at"],
        note=(
            "已生成确认卡（尚未下单）。请告诉用户核对页面上的确认卡并点击「确认下单」；"
            "对话里说「确认」不算数。金额只含商品价，不含税运。"
        ),
    )
