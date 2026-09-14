"""写工具的精准放行（批 0 / L3）。

AgentScope 在 ``PermissionMode.DEFAULT`` 下对**非只读工具**一律发 ``RequireUserConfirmEvent``
把 reply 挂起等人点确认。本仓的写工具（``ask_user`` / ``forget_preference`` / 两个终结工具 /
``task_dispatch``）是 Agent 自己的工作流零件，不是「要不要让 Agent 动你的磁盘」那类高危动作——
每次都弹确认等于让主链路停在半路。

**但不用 ``BYPASS``**：整档关掉权限引擎，等于把批 1 的 TradeAgent（``create_order`` 真该问一句）
一起放行了，那才是本该拦住的东西。这里按工具名逐个 ALLOW，放行范围写死在代码里、看得见：
新加一个写工具时**默认是被拦的**，作者必须显式决定要不要进这张表——失效方向是「多问一次」，
而不是「悄悄替用户下了单」。

规则用 ``rule_content=None``（工具名级放行，见 ``ToolBase.match_rule`` 的说明），
``source="projectSettings"``——它是仓库自带的策略，不是用户当场点的「本次允许」。
"""

from agentscope.permission import (
    PermissionBehavior,
    PermissionEngine,
    PermissionRule,
)
from agentscope.state import AgentState

# 放行清单。每一项都要能回答「为什么它不需要弹确认」。
#
# **交易域为什么也在这里**（L3 时曾计划让它走原生 ``RequireUserConfirmEvent``）：`create_order` /
# `cancel_order` 本身**只生成一张持久化的确认卡**（``trade_confirmations`` 表），真正落订单的
# ``resolve`` 只走 HTTP、由用户在页面上点按钮——模型手里根本没有写订单的能力，工具调用本身
# 不需要再弹一层确认。需要换回原生通路时，把这两项从表里删掉即可（L0 spike 验过它可跨实例恢复）。
DEFAULT_ALLOWED_TOOLS: frozenset[str] = frozenset(
    {
        "ask_user",  # 向用户提问，回复通路是自建 Future 桥（见 app/api/clarification.py）
        "forget_preference",  # 删一条长期偏好，用户明说要忘才会被调
        "shopping_summary",  # 终结工具：产清单 + 落会话产物
        "chat_fallback",  # 终结工具：非购物意图兜底
        "task_dispatch",  # 派 worker，副作用只在本次会话内
        "create_order",  # 只出确认卡，决议走 HTTP（见上）
        "cancel_order",  # 取消前必须先 query_order（sequencing 断言），且只有 CONFIRMED 可取消
    }
)


def allow_tools(state: AgentState, names: frozenset[str] | set[str] | None = None) -> None:
    """给 ``state`` 里的权限上下文加一批工具名级 ALLOW 规则（幂等）。

    直接吃 ``AgentState`` 而不是 ``Agent``：装配时 state 先于 Agent 存在（会话恢复时它是从
    磁盘读回来的那份），先放行再建 Agent，就不必依赖「Agent 建好后还能改它的引擎」这种时序。
    ``PermissionEngine`` 与 Agent 内部那个共享同一个 ``PermissionContext`` 对象，故两边等价。

    幂等很要紧：同一个 state 跨轮复用（续聊 / 恢复），不去重的话规则表会随轮数线性膨胀，
    每次权限检查都要多扫一遍。
    """
    engine = PermissionEngine(state.permission_context)
    existing = state.permission_context.allow_rules
    for name in sorted(names if names is not None else DEFAULT_ALLOWED_TOOLS):
        if any(rule.rule_content is None for rule in existing.get(name, [])):
            continue
        engine.add_rule(
            PermissionRule(
                tool_name=name,
                rule_content=None,
                behavior=PermissionBehavior.ALLOW,
                source="projectSettings",
            )
        )
