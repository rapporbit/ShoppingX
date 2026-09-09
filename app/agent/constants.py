"""无依赖常量：被 ``app.agent`` 与 ``app.harness`` 两侧同时需要、又不能引入循环导入的那些。

**这个模块只准放字面量，不准 import 本仓任何东西。** 它存在的唯一理由是打破
``tool_registry → dispatch_tool → harness → tool_registry`` 这个环——harness 侧要判「这个工具
是不是终结工具」，而 tool_registry 装配时要 import 到 harness。历史上的解法是「在 harness 那边
再抄一份字面量」，抄完之后两份就分了叉（见下），代价是同一轮里两处「是否终结」判定不一致。
"""

# 调用即终结主 loop 的工具。``query_order`` **不在**其中：用户问完订单往往接着要取消或再买
# 一件，查完就结束等于逼他再说一遍。
#
# **为什么是 4 个而不是教学口径的 2 个。** refdocs 08-1 的评测代码按
# ``last_tool in {"shopping_summary", "chat_fallback"}`` 判收尾，create_order / cancel_order
# 在教学里是「带前置条件的写工具」（17-4 的 PREREQUISITES）。本仓多出的两个是交易域落地时的
# 自有扩展，理由是 ``create_order(confirmed=False)`` 出确认卡后本轮**确实该结束等用户表态**，
# 继续跑没有意义。
#
# 这份字面量此前在 ``harness/budgets.py`` 另有一份 2 个的旧值，后果是交易轮收尾时
# ``terminal_reached`` 不置位、``terminal_enforcer`` 认为「本轮没调过终结工具」而追加一次
# 重发（多一轮往返，且文案只提 shopping_summary / chat_fallback，与 system prompt 的
# ``<termination>`` 段「交易轮的终结工具是 create_order / cancel_order 本身」自相矛盾）。
# 2026-09-10 统一到这一份（审查报告 B1）。
TERMINAL_TOOLS = frozenset(
    {"shopping_summary", "chat_fallback", "create_order", "cancel_order"}
)
