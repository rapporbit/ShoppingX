"""安全护栏挂进 Hook Pipeline（refdocs 16-6 §2）——L1 / L3 / L4 三层的接线。

四层里 L2（system prompt 的 ``<security_boundary>`` 边界声明）是纯 prompt，不需要 Hook；另外三层
都是确定性代码，各挂一个 Hook 点：

    pre_tool_call   priority=1   tool_whitelist   非白名单工具名 → 拒绝执行
    post_tool_call  priority=5   content_filter   外部数据源返回 → 洗掉注入指令
    on_session_end  priority=20  output_audit     最终回答 → 脱敏内部信息

**priority 的两处讲究（易碎，勿动）：**

- 白名单是 ``pre_tool_call`` 的**第一道**（1 < 5 的 terminal_reached）。理由：一个根本不存在的
  工具名，没必要先过阶段门 / 预算闸 / 熔断器——那些闸的语义都建立在「这是我们的工具」之上。
  先确认身份，再谈授权。
- 内容过滤是 ``post_tool_call`` 的**第一道**（5 < 10 的 truncate_result）。理由有两条：截断会把
  长结果尾部切掉，注入若藏在尾部就会被截断「顺手清掉」——看起来安全，实则是运气；更要命的是
  ``result_nudges``（priority 20）会往结果尾部追加我们自己的哨兵文案，那些文案里带
  ``[强制收敛]`` 这类方括号标记，晚于它跑的过滤器有误伤自家文案的风险。**先洗外部的，再贴自己的。**
"""

from __future__ import annotations

import logging
from typing import Any

from app.harness.middleware import HookRejectSignal, harness_hook
from app.harness.sentinels import TOOL_NOT_ALLOWED
from app.observability import metrics
from app.security.content_filter import EXTERNAL_SOURCE_TOOLS, sanitize_tool_output
from app.security.output_guard import audit_output
from app.security.tool_whitelist import validate_tool_call

logger = logging.getLogger("shoppingx.harness.security")


@harness_hook("pre_tool_call", name="tool_whitelist", priority=1)
async def check_tool_whitelist(context: dict[str, Any]) -> dict[str, Any] | None:
    """L1：工具名不在 ``FULL_TOOL_SET`` 里 → 直接拒，工具不执行。

    正常情况下这道闸永远不开火（LangChain 只会执行注册过的工具）。它开火意味着两件事之一：
    模型被诱导幻觉出了工具名，或者工具表被动态改过——两者都值得一条 warning + 一个 metric。
    """
    tool_name = context.get("tool_name", "")
    if validate_tool_call(tool_name):
        return None
    metrics.record_security_event("tool_not_allowed")
    logger.warning("L1 工具白名单拦截：tool=%r 不在 FULL_TOOL_SET 内", tool_name)
    raise HookRejectSignal(TOOL_NOT_ALLOWED.format(tool=tool_name), raw=True)


@harness_hook("post_tool_call", name="content_filter", priority=5)
async def filter_tool_output(context: dict[str, Any]) -> dict[str, Any] | None:
    """L3：外部数据源工具的返回，洗掉伪装成指令的文本再回给模型。

    只作用于 ``EXTERNAL_SOURCE_TOOLS``（web_search / item_search / category_insight）——它们的返回
    里有网页正文、卖家写的商品描述、RAG 卡片，是真正的不可信输入。其余工具的返回是我们自己算的。
    """
    tool_name = context.get("tool_name", "")
    if tool_name not in EXTERNAL_SOURCE_TOOLS:
        return None
    result = context.get("tool_result")
    if not isinstance(result, str) or not result:
        return None

    cleaned, hits = sanitize_tool_output(result)
    if not hits:
        return None
    metrics.record_security_event("prompt_injection_filtered")
    logger.warning("L3 内容过滤：%s 的返回命中 %d 处疑似注入，已替换", tool_name, hits)
    context["tool_result"] = cleaned
    return context


@harness_hook("on_session_end", name="output_audit", priority=20)
async def audit_final_answer(context: dict[str, Any]) -> dict[str, Any] | None:
    """L4：最终回答里的密钥 / 内网地址 / 服务器路径 → 脱敏后再推给用户。

    在 ``session_hooks.audit_final_output``（priority 10，洗 Harness 内部控制文案）之后跑：
    先去噪、再脱敏。改写走 ``context["final_answer"]``，由 ``run_agent()`` 消费。
    """
    final = context.get("final_answer")
    if not isinstance(final, str) or not final:
        return None
    is_clean, cleaned, hits = audit_output(final)
    if is_clean:
        return None
    for hit in hits:
        metrics.record_security_event(f"output_{hit}")
    context["final_answer"] = cleaned
    return context
