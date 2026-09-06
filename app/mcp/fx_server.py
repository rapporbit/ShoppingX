"""自建汇率 MCP server —— SearchAgent 消费的那个「外部 MCP」。

    uv run python -m app.mcp.fx_server --port 8766

**为什么自建而不是接 Tavily MCP**：验收要证的是「本仓的 Toolkit 能把一个 MCP server 的工具
列出来、权限判对、真调通」。接第三方就把这条结论绑在它今天在不在线、密钥有没有配上——测试
会因为一个与被测机制无关的原因红。汇率表是纯静态字典（``app/recall/fx.py``），起进程零依赖、
毫秒级，正好当稳定对端。（真要接 Tavily：`.env` 里已有 ``TAVILY_API_KEY`` 的环境把
``MCP_SEARCH_URL`` 指过去即可，接线是同一套，见 ``app/agent/mcp_registry.py``。）

**为什么给 SearchAgent 挑「汇率」这个能力**：它手上只有 ``item_search`` / ``web_search``，
拿回来的候选价格是各平台的原币（SGD / MYR / BRL…），而 ``price_compare`` 是 depth==0 专属
（要跨平台合流后的全局视图才有意义，见 ``tool_registry``）。所以 worker 想在回传摘要里说一句
「这几件折合美元大概多少」时，本来无路可走。这是**真的补了一块**，不是为了摆一个 MCP。

**只读**：两个工具都声明 ``readOnlyHint=True``。这不是文档说明——``agentscope.tool.MCPTool``
读的就是这个 annotation 来定 ``is_read_only``，而 ``is_read_only`` 是 ``PermissionEngine``
的判据。声明错 = SearchAgent 的「只读组」被破。消费侧另有 ``enable_tools`` 白名单兜第二层。
"""

from __future__ import annotations

import argparse

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from app.recall.fx import FX_TO_USD, UnknownCurrencyError, to_base

MCP_SERVER_NAME = "globex-fx"

#: 消费侧 ``enable_tools`` 白名单要与这里逐字对上（见 ``app.agent.mcp_registry``）。
FX_TOOL_NAMES: tuple[str, ...] = ("convert_currency", "list_currencies")

_READ_ONLY = ToolAnnotations(readOnlyHint=True, destructiveHint=False, openWorldHint=False)

mcp = FastMCP(
    MCP_SERVER_NAME,
    instructions=(
        "静态汇率换算（近似中间价，非实时行情）。用于把各平台原币报价折算成统一基准币做粗略比较。"
    ),
)


@mcp.tool(annotations=_READ_ONLY)
def convert_currency(amount: float, from_currency: str, to_currency: str = "USD") -> dict:
    """把金额从一种货币折算到另一种（静态近似汇率，非实时行情）。

    何时调用：手里的商品报价是平台原币（SGD / MYR / BRL…），要折成统一基准币粗略比较时。
    参数：amount 金额；from_currency 原币 ISO 码；to_currency 目标币 ISO 码（默认 USD）。
    """
    try:
        converted = to_base(amount, from_currency, to_currency)
    except UnknownCurrencyError as exc:
        # MCP 工具的错误也要回成模型读得懂的结构，别抛出去变成一条协议级 error——那样模型
        # 只看得到「工具挂了」，看不到「币种不在表里，换一个」。
        return {
            "ok": False,
            "error": str(exc),
            "supported": sorted(FX_TO_USD),
        }
    return {
        "ok": True,
        "amount": round(converted, 2),
        "currency": to_currency.strip().upper(),
        "source": "static-table",
        "note": "近似中间价，非实时行情；仅供粗略比较。",
    }


@mcp.tool(annotations=_READ_ONLY)
def list_currencies() -> dict:
    """列出汇率表里支持的全部币种 ISO 码。

    何时调用：``convert_currency`` 报了未知币种，想知道有哪些能折时。
    """
    return {"currencies": sorted(FX_TO_USD), "base": "USD", "source": "static-table"}


def main() -> None:
    parser = argparse.ArgumentParser(description="globex 汇率 MCP server（streamable HTTP）")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    # 只监听回环：这台 server 没有任何鉴权，绑 0.0.0.0 等于把它开给整个网段。
    mcp.settings.host = args.host
    mcp.settings.port = args.port
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
