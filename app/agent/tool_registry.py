"""工具注册表：一份工具全集 + 按角色发放。

每个业务工具一个文件（模块名 = 工具名），在这里汇总成 ``TOOLS``，再由 :func:`build_toolkit`
发给主 Agent。A4 删掉 SearchAgent 后角色只剩 ``main``；``is_read_only`` 标记仍是
``PermissionEngine`` 放行判定的依据（非只读工具必须进 ``permissions.DEFAULT_ALLOWED_TOOLS``）。

``TERMINAL_TOOLS`` 里的工具一旦被调用即终结循环（堵「不收尾死循环」这个最常见的 Agent 失败）。
"""

from agentscope.tool import FunctionTool, Toolkit, ToolMiddlewareBase

from app.agent.constants import TERMINAL_TOOLS as _TERMINAL_TOOLS
from app.agent.mcp_registry import mcp_clients
from app.agent.skills import skill_loaders
from app.tools._shell import ToolShell, to_function_tool
from app.tools.ask_user import ask_user
from app.tools.cancel_order import cancel_order
from app.tools.category_insight import category_insight
from app.tools.chat_fallback import chat_fallback
from app.tools.create_order import create_order
from app.tools.forget_preference import forget_preference
from app.tools.image_understand import image_understand
from app.tools.item_picker import item_picker
from app.tools.item_search import item_search
from app.tools.planner import planner
from app.tools.present_comparison import present_comparison
from app.tools.price_compare import price_compare
from app.tools.query_order import query_order
from app.tools.recall_memories import recall_memories
from app.tools.research import research
from app.tools.save_memory import save_memory
from app.tools.shipping_calc import shipping_calc
from app.tools.shopping_summary import shopping_summary
from app.tools.web_search import web_search

# 调用即终结主 loop 的工具。定义在 ``app.agent.constants``（无依赖模块，harness 侧也从那里读，
# 见该文件里「为什么是 4 个」与两份字面量分叉的旧账）；这里 re-export 只为让「工具的事在
# 工具注册表里查得到」。
TERMINAL_TOOLS = _TERMINAL_TOOLS

# 业务工具（每文件一个，模块名 = 工具名）：九大主工具 + ask_user 澄清 + 三个记忆工具。
# 三个记忆工具的分工：recall_memories 读（注入只给 tier-one 那批，用户问「我以前买的那双鞋」
# 时要的恰恰是没进注入的）、save_memory 写（M2 加，用户当场说「记住 X」时即时生效并给回执）、
# forget_preference 删（M4 的删除清单里，遗忘将改为 save_memory 同 key 覆盖）。
# 回合后的 curator（app/memory/curator.py）仍是另一条写路径，两条都过 facts.validate_fact
# 同一道门、按 key 覆盖同一张表，不构成两套语义。
_BUSINESS_TOOLS: list[ToolShell] = [
    planner,
    image_understand,
    item_search,
    price_compare,
    shipping_calc,
    category_insight,
    item_picker,
    web_search,
    research,
    present_comparison,
    chat_fallback,
    shopping_summary,
    ask_user,
    recall_memories,
    forget_preference,
    save_memory,
    create_order,
    query_order,
    cancel_order,
]


# 只读 = 不写任何持久状态、不与用户交互、可安全并发重放。
# 反例说明（别凭感觉标）：ask_user 会挂起等用户回复，forget_preference 删长期偏好，
# shopping_summary / chat_fallback 是终结工具（写会话产物 + 决定 loop 结束），都不是只读。
_READ_ONLY_TOOLS = frozenset(
    {
        "planner",
        "image_understand",
        "item_search",
        "price_compare",
        "shipping_calc",
        "category_insight",
        "item_picker",
        "web_search",
        "research",
        "query_order",
        "recall_memories",
    }
)


def _make_tools(middlewares: list[ToolMiddlewareBase] | None = None) -> list[FunctionTool]:
    """造一批新壳（可带工具中间件）。

    **为什么每次 loop 都要重造**：``ToolMiddlewareBase`` 是挂在**工具实例**上的
    （``ToolBase._middlewares``），而 harness 的工具适配器持有 per-loop 的 ``HarnessSession``。
    共享一份工具实例就等于共享控制面状态——并发的多个会话的断言、循环检测、熔断计数会串成
    一锅。重造的代价只是 15 个闭包 + 15 个 ``FunctionTool`` 对象（schema 与实现函数仍是
    同一份，不重复解析业务逻辑），比起状态串味那种查半天的 bug，这点开销买得值。
    """
    return [
        to_function_tool(t, is_read_only=t.name in _READ_ONLY_TOOLS, middlewares=middlewares)
        for t in _BUSINESS_TOOLS
    ]


# 无中间件的一份（元数据查询 / 白名单 / 测试等不需要控制面的场景用）。真正跑 loop 的工具由
# build_toolkit 现造，见 _make_tools 的说明。
TOOLS: list[FunctionTool] = _make_tools()

TOOLS_BY_NAME: dict[str, FunctionTool] = {t.name: t for t in TOOLS}

# 角色 → 该角色能拿到的工具名（None = 全集）。批 1 起曾有 ``search``（只读 SearchAgent）与
# ``trade`` 两个 worker 角色，A1 / A4 先后删除：440 个会话里派发全是单跳壳，交易工具只出确认卡。
# 发放口径（Skill / MCP 也按 role 切）仍保留这张表，将来真要加受限角色时从这里开口。
#
# 为什么不用 ``ToolGroup``：它是**运行时可激活 / 停用**的分组——模型可以调 meta tool 把组激活
# 回来，工具对象照样住在 Toolkit 里。那是「按需露出」，不是权限边界。
_ROLE_TOOLS: dict[str, frozenset[str] | None] = {
    "main": None,
}


async def build_toolkit(
    role: str = "main",
    tool_middlewares: list[ToolMiddlewareBase] | None = None,
) -> Toolkit:
    """按角色发一份 Toolkit（AgentScope 侧）。

    每次调用新建 Toolkit **与工具实例**：Toolkit 带角色态（激活的 tool group 等）不能跨 Agent
    共享，工具实例则因为要挂 per-loop 的中间件而必须一 loop 一份（见 :func:`_make_tools`）。

    批 4-3 起同一个「发放范围」口径多管两样东西，都走框架原生、都按 role 切：
    **Skill**（``skills_or_loaders``，见 ``app.agent.skills``）与 **MCP**
    （``mcps``，只放只读白名单，见 ``app.agent.mcp_registry``）。它们都进
    Toolkit 的 ``basic`` 组——本仓不用 ToolGroup 表达权限（理由见上方 ``_ROLE_TOOLS`` 注释），
    组只有一个，边界仍然是「这份 Toolkit 里有没有」。
    """
    if role not in _ROLE_TOOLS:
        raise ValueError(f"未知角色 {role!r}，可选：{sorted(_ROLE_TOOLS)}")
    allowed = _ROLE_TOOLS[role]
    toolkit = Toolkit(skills_or_loaders=skill_loaders(role), mcps=mcp_clients(role))
    for tool_obj in _make_tools(tool_middlewares) if tool_middlewares else TOOLS:
        if allowed is None or tool_obj.name in allowed:
            await toolkit.add_tool(tool_obj)
    return toolkit
