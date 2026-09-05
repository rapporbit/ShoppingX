"""主 / 子 Agent 共用的唯一工具集（同质 fork 的硬约束）。

``FULL_TOOL_SET`` 是主 loop 与所有 fork 出的子 loop 共享的同一份列表对象。
dispatch 元工具通过闭包延迟读取这份 live 列表，因此子 Agent 拿到的工具集始终和主
Agent 完全一致（含 dispatch 自身 → 支持递归 fork）。

M4 起集齐九大业务工具 + dispatch 元工具。主 / 子 Agent 必须用这同一份集合（同质 fork）。
``TERMINAL_TOOLS`` 中的工具一旦被调用即终结循环（堵「不收尾死循环」）。
"""

from agentscope.tool import FunctionTool, Toolkit
from langchain_core.tools import BaseTool

from app.agent.dispatch_tool import make_dispatch_tools
from app.tools._as_tools import as_function_tool
from app.tools.ask_user import ask_user
from app.tools.category_insight import category_insight
from app.tools.chat_fallback import chat_fallback
from app.tools.forget_preference import forget_preference
from app.tools.image_understand import image_understand
from app.tools.item_picker import item_picker
from app.tools.item_search import item_search
from app.tools.planner import planner
from app.tools.price_compare import price_compare
from app.tools.shipping_calc import shipping_calc
from app.tools.shopping_summary import shopping_summary
from app.tools.web_search import web_search

# 调用即终结主 loop 的工具。
TERMINAL_TOOLS = {"shopping_summary", "chat_fallback"}

# 业务工具（每文件一个，模块名 = 工具名）：九大主工具 + ask_user 澄清 + forget_preference。
# 注意:**没有** remember_preference——偏好的识别 / 沉淀已剥离给会话结束后独立运行的记忆管家
# （app/memory/curator.py），购物工作流里不再有「随手记长期偏好」的工具。forget_preference 保留:
# 用户明确要撤回某条长期偏好时即时生效，这与记忆判定正交。
_BUSINESS_TOOLS: list[BaseTool] = [
    planner,
    image_understand,
    item_search,
    price_compare,
    shipping_calc,
    category_insight,
    item_picker,
    web_search,
    chat_fallback,
    shopping_summary,
    ask_user,
    forget_preference,
]

# 主 / 子共用的同一份列表对象；先放业务工具，再 append dispatch 元工具。
FULL_TOOL_SET: list[BaseTool] = list(_BUSINESS_TOOLS)

# dispatch 元工具的工具集 provider：返回 live 的 FULL_TOOL_SET，保证同质 + 可递归。
dispatch_tool, parallel_dispatch_tool = make_dispatch_tools(lambda: FULL_TOOL_SET)

FULL_TOOL_SET.extend([dispatch_tool, parallel_dispatch_tool])


# ============================================================================
# AgentScope 侧的工具发放（批 0 / L2，与上面的 FULL_TOOL_SET 并存）
# ----------------------------------------------------------------------------
# 与旧壳指向同一批实现函数（见 app/tools/_as_tools.py 的双包装说明）。批 0 三个 role 都
# 发全集——读写切分是批 1 的事，这里先把「按角色发放」这个入口摆好，免得 L3 装配时又要动
# 一次结构。**只读标记现在就要标准**：批 1 的 SearchAgent 靠它做结构性权限边界。
# ============================================================================

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
    }
)

AS_TOOLS: list[FunctionTool] = [
    as_function_tool(t, is_read_only=t.name in _READ_ONLY_TOOLS) for t in _BUSINESS_TOOLS
]

AS_TOOLS_BY_NAME: dict[str, FunctionTool] = {t.name: t for t in AS_TOOLS}

# 角色 → 该角色能拿到的工具名。批 0 全是全集；批 1 改这张表即完成读写切分（SearchAgent 只拿
# 只读子集、TradeAgent 只拿交易写工具），**切的是发放范围，不是实现**。
_ROLE_TOOLS: dict[str, frozenset[str] | None] = {
    "main": None,  # None = 全集
    "search": None,
    "trade": None,
}


async def build_toolkit(role: str = "main") -> Toolkit:
    """按角色发一份 Toolkit（AgentScope 侧）。

    每次调用**新建** Toolkit 实例但复用同一批 ``FunctionTool`` 对象：工具是无状态的，
    共享省掉重复构造；Toolkit 带角色态（激活的 tool group 等），不能跨 Agent 共享。
    """
    if role not in _ROLE_TOOLS:
        raise ValueError(f"未知角色 {role!r}，可选：{sorted(_ROLE_TOOLS)}")
    allowed = _ROLE_TOOLS[role]
    toolkit = Toolkit()
    for tool_obj in AS_TOOLS:
        if allowed is None or tool_obj.name in allowed:
            await toolkit.add_tool(tool_obj)
    return toolkit
