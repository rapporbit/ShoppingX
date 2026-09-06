"""工具注册表：一份工具全集 + 按角色发放。

每个业务工具一个文件（模块名 = 工具名），在这里汇总成 ``TOOLS``，再由 :func:`build_toolkit`
按角色发给主 Agent / 各类 worker——**切的是发放范围，不是实现**：三种角色拿到的是同一批实现
函数包出来的工具，只是集合不同。这是读写边界的结构性保证之一（另两个是 ``is_read_only`` 标记
与 ``PermissionEngine`` 精准放行），不靠提示词劝退。

``TERMINAL_TOOLS`` 里的工具一旦被调用即终结循环（堵「不收尾死循环」这个最常见的 Agent 失败）。
"""

from agentscope.tool import FunctionTool, Toolkit, ToolMiddlewareBase

from app.agent.dispatch_tool import task_dispatch
from app.tools._shell import ToolShell, to_function_tool
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
_BUSINESS_TOOLS: list[ToolShell] = [
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


def _make_tools(middlewares: list[ToolMiddlewareBase] | None = None) -> list[FunctionTool]:
    """造一批新壳（可带工具中间件）。

    **为什么每次 loop 都要重造**：``ToolMiddlewareBase`` 是挂在**工具实例**上的
    （``ToolBase._middlewares``），而 harness 的工具适配器持有 per-loop 的 ``HarnessSession``。
    共享一份工具实例就等于共享控制面状态——主 loop 与并发 worker 的断言、循环检测、熔断计数
    会串成一锅。重造的代价只是 12 个闭包 + 12 个 ``FunctionTool`` 对象（schema 与实现函数仍是
    同一份，不重复解析业务逻辑），比起状态串味那种查半天的 bug，这点开销买得值。
    """
    tools = [
        to_function_tool(t, is_read_only=t.name in _READ_ONLY_TOOLS, middlewares=middlewares)
        for t in _BUSINESS_TOOLS
    ]
    tools.append(
        FunctionTool(
            task_dispatch,
            # 同轮多个独立子任务靠它并发（框架看到多个 tool_call 就并行执行），这也是不再
            # 需要一个「入参是列表」的并行派发元工具的底气所在。
            is_concurrency_safe=True,
            is_read_only=False,
            middlewares=middlewares,
        )
    )
    return tools


# 无中间件的一份（元数据查询 / 白名单 / 测试等不需要控制面的场景用）。真正跑 loop 的工具由
# build_toolkit 现造，见 _make_tools 的说明。
TOOLS: list[FunctionTool] = _make_tools()

TOOLS_BY_NAME: dict[str, FunctionTool] = {t.name: t for t in TOOLS}

# 角色 → 该角色能拿到的工具名（批 1 的读写切分，**切的是发放范围，不是实现**）。
#
# 为什么不用 ``ToolGroup``：它是**运行时可激活 / 停用**的分组——非 basic 组默认不激活，模型可
# 以调 meta tool 把组激活回来，而且工具对象照样住在 Toolkit 里（``get_tool("create_order")``
# 拿得到）。那是「按需露出」，不是权限边界。这里要的是结构性保证：worker 的 Toolkit 里**根本
# 没有**那个工具对象，模型再怎么想调也调不出来。
#
# ``task_dispatch`` 只在 main 手上：「worker 派不了 worker」是派发安全第①层（深度上限）的
# 结构性保证，比 fork_guard 的计数守卫更硬——计数守卫拦的是次数，这个拦的是可能性。
#
# **为什么只有两个**（对齐手册 §7.1 那张表的刻意偏离）：手册照 refdocs 给的是五个
# （+ category_insight / price_compare / shipping_calc），但本仓早有一道 ``depth_gate``
# 把这三个划为 depth==0 专属，理由至今成立：
#   · price_compare 要的是**跨平台合流后的全局视图**——只搜了一个平台的 worker 拿不出别家数据，
#     它在那里比价，比的是个寂寞；
#   · category_insight 平台无关、主流程跑一次结果就写进 demands，N 个 worker 各跑一遍纯属重复解码；
#   · shipping_calc 同理跟着合流后的候选集算，否则每个 worker 都为自己那批候选算一遍运费。
# 发了工具又被闸硬拒 = 模型每次调都白烧一轮再吃一条拒绝文案。**发放范围与闸的口径必须一致**，
# 取交集后 SearchAgent 就是「搜货 + 查库外事实」这两件事——这也正是它现在实际在做的全部。
_SEARCH_TOOLS = frozenset({"item_search", "web_search"})

# 交易写工具，批 1 的 7.2 落地后填进来（那之前 task_dispatch 对 trade 直接拒派）。
_TRADE_TOOLS: frozenset[str] = frozenset()

_ROLE_TOOLS: dict[str, frozenset[str] | None] = {
    "main": None,  # None = 全集
    "search": _SEARCH_TOOLS,
    "trade": _TRADE_TOOLS,
}

# 自检：search 拿到的必须全是只读工具。读写边界的三根支柱（发放范围 / is_read_only 标记 /
# PermissionEngine）里，前两根在这里对齐——漏标一个只读，或往 search 集合里塞进一个写工具，
# 都在 import 期就炸，而不是等线上某轮 worker 偷偷写了状态。
assert _SEARCH_TOOLS <= _READ_ONLY_TOOLS, sorted(_SEARCH_TOOLS - _READ_ONLY_TOOLS)


def trade_tools_ready() -> bool:
    """交易域是否已就绪（7.2 落地后为真）。派发入口据此决定 ``trade`` 能不能派。

    判据是「TradeAgent 的工具集非空」而不是某个开关变量：工具还没建出来的时候，派过去就是一个
    零工具的 Agent 空转一轮再超时——那种失败模式对用户表现为「卡了 90 秒然后说不知道」。
    """
    return bool(_TRADE_TOOLS)


async def build_toolkit(
    role: str = "main",
    tool_middlewares: list[ToolMiddlewareBase] | None = None,
) -> Toolkit:
    """按角色发一份 Toolkit（AgentScope 侧）。

    每次调用新建 Toolkit **与工具实例**：Toolkit 带角色态（激活的 tool group 等）不能跨 Agent
    共享，工具实例则因为要挂 per-loop 的中间件而必须一 loop 一份（见 :func:`_make_tools`）。
    """
    if role not in _ROLE_TOOLS:
        raise ValueError(f"未知角色 {role!r}，可选：{sorted(_ROLE_TOOLS)}")
    allowed = _ROLE_TOOLS[role]
    toolkit = Toolkit()
    for tool_obj in _make_tools(tool_middlewares) if tool_middlewares else TOOLS:
        if allowed is None or tool_obj.name in allowed:
            await toolkit.add_tool(tool_obj)
    return toolkit
