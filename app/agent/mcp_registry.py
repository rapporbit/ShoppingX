"""消费侧 MCP 接线（批 4-3）—— 把一个外部 MCP server 的工具挂进 SearchAgent 的只读组。

对端默认是本仓自建的汇率 MCP（``app/mcp/fx_server.py``，纯静态表、零外部依赖）。想换成
Tavily MCP 或别的 HTTP MCP，把 ``MCP_SEARCH_URL`` 指过去、``MCP_SEARCH_NAME`` /
``MCP_SEARCH_TOOLS`` 跟着改即可，接线一行不用动。

**默认关（``MCP_SEARCH_URL`` 空 = 不挂）**。理由不是保守：MCP 工具是**每次模型调用**都要向
server 要一次工具表的（框架在 ``_get_available_tools`` 里现拉，见 ``Toolkit``）。默认指向一个
多半没起进程的 localhost，代价是每一轮都吃一次连接超时 + 一条 warning，而且工具表会在「server
恰好起着」和「没起」之间飘——同一个问题两次跑出不同的工具集，这种不确定性比少一个汇率工具贵
得多。要用就显式配 URL，配了就当它该在。

**读写切分怎么不被 MCP 破坏**（三保证逐条对上）：

1. **发放范围** —— ``MCP_ROLES`` 钉死只有 ``search`` 拿得到；``main`` / ``trade`` 的 Toolkit
   里根本没有这个 client。
2. **``is_read_only`` 标记** —— ``agentscope.tool.MCPTool`` 的 ``is_read_only`` 取自 MCP 工具的
   ``annotations.readOnlyHint``（**取不到就是 False**）。所以对端必须声明；本仓的 fx server
   两个工具都声明了。这一层依赖对端自觉，故有第 3 层。
3. **``enable_tools`` 白名单** —— 客户端侧只放行 ``MCP_SEARCH_TOOLS`` 里逐字列出的工具名，
   其余一律不进工具表。对端将来加了一个写工具、或声明错了 ``readOnlyHint``，这层仍然挡得住：
   它拦的是「有没有资格出现」，不是「它自称是什么」。

**诚实标注：MCP 工具不经过 harness 的工具中间件。** 本仓的 ``HarnessToolAdapter`` 是挂在
``FunctionTool`` **实例**上的（见 ``tool_registry._make_tools``），而 MCP 工具对象由框架在
``Toolkit`` 内部现造，我们够不着。后果：MCP 工具的调用不走 ``pre_tool_call`` / ``post_tool_call``
那一串闸（白名单 / 阶段门 / 截断 / 内容过滤），也不发 AGUI 的 ``tool_start`` / ``tool_end``。
它仍在框架原生的 ``TracingMiddleware`` 里可见（span 在 acting 段下）。这是**本单接受的边界**，
不是遗漏——把控制面强行接到框架内部造的对象上，得靠 monkeypatch ``Toolkit`` 的私有方法，那种
接法会在框架小版本升级时静默失效，比缺一段遥测危险。因此对端只许是**只读且无副作用**的
server；一旦要接有副作用的 MCP，先把这段控制面补上再说。
"""

from __future__ import annotations

from agentscope.mcp import HttpMCPConfig, MCPClient

from app.mcp.fx_server import FX_TOOL_NAMES
from app.utils.env import env_float, env_str

#: 拿得到 MCP 的角色。见模块 docstring「发放范围」。
MCP_ROLES: frozenset[str] = frozenset({"search"})

#: 默认对端 = 自建汇率 MCP。名字进模型看到的工具名（``mcp__{name}__{tool}``），
#: 框架要求它匹配 ``^[a-zA-Z0-9_-]+$``。
DEFAULT_MCP_NAME = "globex-fx"


def _tool_whitelist() -> list[str]:
    raw = env_str("MCP_SEARCH_TOOLS", ",".join(FX_TOOL_NAMES))
    return [name.strip() for name in raw.split(",") if name.strip()]


def mcp_tool_names(role: str = "search") -> list[str]:
    """该角色的 MCP 工具在模型侧的全名（``mcp__{server}__{tool}``）。

    给安全白名单用（``app/security/tool_whitelist.py``）：白名单只回答「这是不是本系统的
    工具」，MCP 工具是本系统主动挂进去的，就该认得。**不查 server** —— 白名单不能依赖一次
    网络往返，否则对端一挂，第一道安全闸自己先不可用。
    """
    if role not in MCP_ROLES or not env_str("MCP_SEARCH_URL"):
        return []
    server = env_str("MCP_SEARCH_NAME", DEFAULT_MCP_NAME)
    return [f"mcp__{server}__{tool}" for tool in _tool_whitelist()]


def mcp_clients(role: str = "search") -> list[MCPClient]:
    """按角色返回要挂进 Toolkit 的 MCP 客户端（未配 URL 则空表）。

    ``is_stateful=False``：无状态 HTTP，每次调用现开一条临时会话。选它而不是长连接，是因为
    长连接要在 Toolkit 构造**之前**完成 ``connect()``（框架会对未连接的 stateful client 直接
    报错），而本仓的 Toolkit 是每个 loop 现建的——那等于给每次装配加一次握手，还得配套一个
    谁都记不住的 ``close()``。汇率这种一问一答的调用不需要会话态。
    """
    if role not in MCP_ROLES:
        return []
    url = env_str("MCP_SEARCH_URL")
    if not url:
        return []
    return [
        MCPClient(
            name=env_str("MCP_SEARCH_NAME", DEFAULT_MCP_NAME),
            is_stateful=False,
            mcp_config=HttpMCPConfig(url=url, timeout=env_float("MCP_SEARCH_TIMEOUT", 10.0)),
            # 只放行白名单里的工具名（第 3 层保证，见模块 docstring）。
            enable_tools=_tool_whitelist(),
            execution_timeout=env_float("MCP_SEARCH_EXEC_TIMEOUT", 20.0),
        )
    ]
