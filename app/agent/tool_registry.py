"""主 / 子 Agent 共用的唯一工具集（同质 fork 的硬约束）。

``FULL_TOOL_SET`` 是主 loop 与所有 fork 出的子 loop 共享的同一份列表对象。
dispatch 元工具通过闭包延迟读取这份 live 列表，因此子 Agent 拿到的工具集始终和主
Agent 完全一致（含 dispatch 自身 → 支持递归 fork）。

M4 起集齐九大业务工具 + dispatch 元工具。主 / 子 Agent 必须用这同一份集合（同质 fork）。
``TERMINAL_TOOLS`` 中的工具一旦被调用即终结循环（堵「不收尾死循环」）。
"""

from langchain_core.tools import BaseTool

from app.agent.dispatch_tool import make_dispatch_tools
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
