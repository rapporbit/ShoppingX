"""生产侧 MCP server —— 把本仓的只读三工具开给仓外消费者。

    uv run python -m app.mcp.server --port 8765

暴露 ``item_search`` / ``price_compare`` / ``shipping_calc``，全部 ``readOnlyHint=True``。
**为什么正好是这三个**：它们是本仓 ``_READ_ONLY_TOOLS`` 里唯一自洽成链的一组——搜出候选、
折算比价、算到手价，外部消费者拿这三件就能完成「跨平台找货并按实付排序」这一件完整的事。
其余只读工具要么依赖本仓的会话状态（``item_picker`` 要偏好与槽表）、要么是内部环节
（``planner`` 吐的是给本仓 loop 用的结构化计划），单独开出去只会给出无法解释的返回。

**写工具一个都不开。** 这不是保守：MCP server 没有本仓的两段式确认卡、没有 ``_order_guard``、
没有 harness 的顺序闸，``create_order`` 开出去就是一个无人看管的下单端点。读写切分在这里
仍然是结构性的——server 的工具表里根本没有它们。

**会话续接（``session_id``）**：``price_compare`` / ``shipping_calc`` 吃的是**上一次
``item_search`` 登记进会话的候选**（按 id hydrate，本仓刻意不让模型重吐候选对象）。MCP 是
无状态调用，所以把这条会话线显式交给调用方：第一次调 ``item_search`` 不传 ``session_id``，
返回里带一个；后两个工具把它原样传回来，就落在同一个会话目录、同一份候选登记表上。不传就
各自开一条新会话——那时 ``price_compare`` 会诚实地返回「没有候选」，而不是静默给空结果。
"""

from __future__ import annotations

import argparse
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from app.tools.item_search import item_search as _item_search
from app.tools.price_compare import price_compare as _price_compare
from app.tools.shipping_calc import shipping_calc as _shipping_calc
from app.utils.path_utils import ensure_session_dir
from app.utils.thread_ctx import thread_scope

MCP_SERVER_NAME = "globex-readonly"

#: 开出去的工具名。**新增一项前先回答「它在没有本仓会话状态时还讲得通吗」**。
EXPOSED_TOOL_NAMES: tuple[str, ...] = ("item_search", "price_compare", "shipping_calc")

_READ_ONLY = ToolAnnotations(readOnlyHint=True, destructiveHint=False, openWorldHint=False)

mcp = FastMCP(
    MCP_SERVER_NAME,
    instructions=(
        "globex 跨平台商品检索（只读）。典型链路：item_search 拿候选 → 用返回的 session_id "
        "调 price_compare 得到到手价排序 → 需要换收货国时再调 shipping_calc。"
    ),
)


@contextmanager
def _session(session_id: str) -> Iterator[str]:
    """把一次 MCP 调用绑进一条会话线（thread_id / session_dir）。

    ``thread_id`` 一律加 ``mcp-`` 前缀并只保留安全字符：它会被当成目录名拼进 ``output/``，
    调用方传什么都不该变成路径穿越。空则新开一条。
    """
    raw = "".join(ch for ch in session_id.strip() if ch.isalnum() or ch in "-_")
    thread_id = raw or f"mcp-{uuid.uuid4().hex[:12]}"
    if not thread_id.startswith("mcp-"):
        thread_id = f"mcp-{thread_id}"
    with thread_scope(thread_id, ensure_session_dir(thread_id)):
        yield thread_id


@mcp.tool(annotations=_READ_ONLY)
async def item_search(
    query: str,
    platform: str = "all",
    price_usd_max: float | None = None,
    min_rating: float | None = None,
    session_id: str = "",
) -> dict[str, Any]:
    """在电商平台检索商品（语义召回）。

    何时调用：需要按自然语言意图找货时。query 用品类核心词（如 laptop backpack），
    风格 / 场景词不要堆进来。platform 取 amazon/walmart/shein/lazada/shopee 或 all。
    返回里的 session_id 请在后续 price_compare / shipping_calc 调用中原样传回。
    """
    with _session(session_id) as thread_id:
        out = await _item_search.ainvoke(
            {
                "query": query,
                "platform": platform,
                "price_usd_max": price_usd_max,
                "min_rating": min_rating,
            }
        )
        return {"session_id": thread_id, **out.model_dump()}


@mcp.tool(annotations=_READ_ONLY)
async def price_compare(session_id: str, top_n: int = 12) -> dict[str, Any]:
    """把上一次 item_search 的候选统一折算成 USD 并算出到手价（货价+运费+关税）排序。

    何时调用：拿到候选后要按实付金额横向比较时。**调完本工具无需再调 shipping_calc。**
    参数：session_id 取自 item_search 的返回；top_n 返回前 N 条最便宜的。
    """
    with _session(session_id) as thread_id:
        out = await _price_compare.ainvoke({"top_n": top_n})
        return {"session_id": thread_id, **out.model_dump()}


@mcp.tool(annotations=_READ_ONLY)
async def shipping_calc(
    session_id: str, item_ids: list[str] | None = None, dest_country: str = ""
) -> dict[str, Any]:
    """按指定收货国重算候选的到手价（货价+国际运费+关税）。

    何时调用：只在需要换一个收货国重算、或候选还没经过 price_compare 时才用。
    参数：session_id 取自 item_search；item_ids 留空即本轮全部候选；
    dest_country 收货国 ISO 码（如 JP / DE），留空走默认国。
    """
    with _session(session_id) as thread_id:
        out = await _shipping_calc.ainvoke(
            {"item_ids": list(item_ids or []), "dest_country": dest_country}
        )
        return {"session_id": thread_id, **out.model_dump()}


def main() -> None:
    parser = argparse.ArgumentParser(description="globex 只读工具 MCP server（streamable HTTP）")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    # 只监听回环：这台 server 没有鉴权，而它背后是真实的召回栈（Qdrant / reranker / 外部
    # web_search 额度）。绑 0.0.0.0 等于把这些开给整个网段。
    mcp.settings.host = args.host
    mcp.settings.port = args.port
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
