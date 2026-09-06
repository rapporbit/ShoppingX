"""query_order —— 查订单（只读，带归属校验，**非终结**）。

非终结是有意的：用户问「我那单怎么样了」往往接着要取消或再买一件，查完就把本轮结束等于逼他
再说一遍。取消前必须先经过它（见 ``hooks/step_validator.py`` 的 sequencing 断言）——「没查就
取消」是模型最容易犯的那种错，而它取消的是真落库的单。

归属校验在用例层（``usecases._load_owned``）而不是这里：工具走的是 Agent 通路、不经 HTTP 路由，
只在路由上校验等于给「模型被诱导去查别人的单」留一条没设防的路。订单不存在与不属于你回同一句
话，理由见那个函数。
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.api import monitor
from app.api.context import get_user_id
from app.tools._shell import tool
from app.trade.repository_sql import order_repository
from app.trade.usecases import OrderNotFoundError, query_orders


class QueryOrderOutput(BaseModel):
    """query_order 的结构化返回。"""

    orders: list[dict[str, Any]] = Field(default_factory=list, description="订单快照列表")
    count: int = Field(default=0, description="命中条数")
    note: str = Field(default="", description="给模型的简短说明")


@tool
async def query_order(order_id: str = "", limit: int = 10) -> QueryOrderOutput:
    """查订单（只读，非终结：查完可以继续做别的）。

    何时调用：用户问「我的订单」「那单到哪了」「我买过什么」时；**以及取消订单之前**——必须
    先查到那张单、确认它确实存在且状态可取消，再调 cancel_order。
    参数：
      - order_id：查某一张（形如 GBX-000123）。不传则列出该用户最近的几张。
      - limit：不传 order_id 时最多列几张，默认 10。
    """
    await monitor.report_tool_start("query_order", order_id=order_id)
    user_id = get_user_id() or ""
    if not user_id:
        return QueryOrderOutput(note="匿名会话没有订单记录（未登录）。")
    try:
        orders = await query_orders(
            order_repository(), user_id=user_id, order_id=order_id, limit=limit
        )
    except OrderNotFoundError as e:
        await monitor.report_tool_end("query_order", error=str(e))
        return QueryOrderOutput(note=f"[error] {e}")

    snaps = [o.snapshot() for o in orders]
    await monitor.report_tool_end("query_order", count=len(snaps))
    note = "该用户还没有任何订单。" if not snaps else f"查到 {len(snaps)} 张订单。"
    return QueryOrderOutput(orders=snaps, count=len(snaps), note=note)
