"""无依赖常量：被 ``app.agent`` 与 ``app.harness`` 两侧同时需要、又不能引入循环导入的那些。

**这个模块只准放字面量，不准 import 本仓任何东西。** 它存在的唯一理由是打破
``tool_registry → harness → tool_registry`` 这个环——harness 侧要判「这个工具
是不是终结工具」，而 tool_registry 装配时要 import 到 harness。历史上的解法是「在 harness 那边
再抄一份字面量」，抄完之后两份就分了叉（见下），代价是同一轮里两处「是否终结」判定不一致。
"""

# 调用即终结主 loop 的工具。``query_order`` **不在**其中：用户问完订单往往接着要取消或再买
# 一件，查完就结束等于逼他再说一遍。
#
# **为什么是 4 个而不是教学口径的 2 个。** refdocs 08-1 的评测代码按
# ``last_tool in {"shopping_summary", "chat_fallback"}`` 判收尾，create_order / cancel_order
# 在教学里是「带前置条件的写工具」（17-4 的 PREREQUISITES）。本仓多出的两个是交易域落地时的
# 自有扩展，理由是 ``create_order`` 出确认卡后本轮**确实该结束等用户表态**，
# 继续跑没有意义。
#
# 这份字面量此前在 ``harness/budgets.py`` 另有一份 2 个的旧值，后果是交易轮收尾时
# ``terminal_reached`` 不置位、``terminal_enforcer`` 认为「本轮没调过终结工具」而追加一次
# 重发（多一轮往返，且文案只提 shopping_summary / chat_fallback，与 system prompt 的
# ``<termination>`` 段「交易轮的终结工具是 create_order / cancel_order 本身」自相矛盾）。
# 2026-09-10 统一到这一份（审查报告 B1）。
# ``present_comparison``（C4）也在其中，同一条理由：它产出的就是面向用户的最终结构化答案，
# 调完再让模型去调 shopping_summary，只会把同一份判断用散文重讲一遍（over-loop 的老形态）。
# ``present_guide``（S3）同理：它就是「没有商品卡那一轮」的最终答案本体，调完再让模型去调
# chat_fallback 把标准复述一遍，正是 present_comparison 那条注释说的 over-loop 老形态。
TERMINAL_TOOLS = frozenset(
    {
        "shopping_summary",
        "chat_fallback",
        "create_order",
        "cancel_order",
        "present_comparison",
        "present_guide",
    }
)


def is_terminal_call(tool_name: object, tool_args: object = None) -> bool:
    """这一次调用是不是终结调用——判据是「工具名 + 入参」，不只是名字。

    多出入参这一维是 ``ask_user`` 逼出来的（D2）：同一个工具承载两种形态，``closes_turn=False``
    暂停 loop 等用户回复（非终结），``closes_turn=True`` 发一组 chips 让用户挑、调完即收尾
    （终结，等价于 Anthropic 博客的 ``present_suggestions``）。**不为收尾形态新增第三个工具**
    是有意的：两个职责高度重叠的工具并存，模型会乱选（执行计划 §3-10）。

    入参拿不到时退回名字判据——安全方向是「当成非终结」，宁可多跑一轮也不要把还在等回复的
    那一问判成收尾、把用户晾在半路。
    """
    if tool_name in TERMINAL_TOOLS:
        return True
    if tool_name == "ask_user" and isinstance(tool_args, dict):
        return bool(tool_args.get("closes_turn"))
    return False
