"""安全底线（硬，无逃生门，无开关）：白名单 / 深度断言 / 内容过滤 / 结果截断 / 输出审核与脱敏。

    pre_tool_call    1  tool_whitelist    L1：不在白名单的工具名一律拒
    pre_tool_call   10  depth_gate        worker 碰主 loop 专属工具 → 只报警不拦
                                          （边界在 Toolkit 发放范围）
    post_tool_call   5  content_filter    L3：外部来源工具返回里的提示词注入
    post_tool_call  10  truncate_result   过长结果按 token 预算截断
                                          （**必须早于任何追加提示的 Hook**）
    on_session_end  10  output_guard      把 Harness 内部控制文案从最终回复里剔掉
    on_session_end  20  output_audit      L4：密钥 / 内网地址 / 服务器路径脱敏（先去噪、再脱敏）

输出审核刻意保守：只清洗模型鹦鹉学舌的哨兵文案，**不做** item_id 之类的正则脱敏——那些常是用户
真正想要的信息，宁可漏放不误杀。
"""

from __future__ import annotations

import logging
import re
from typing import Any

from app.agent.fork_guard import current_fork_depth
from app.harness.budgets import (
    DEPTH0_ONLY_TOOLS,
    MAIN_ONLY_CONTEXT_TOOLS,
)
from app.harness.middleware import harness_hook
from app.harness.sentinels import (
    INTERNAL_MARKERS,
)
from app.harness.state import guard_of
from app.harness.truncation import truncate_tool_result
from app.observability import metrics
from app.security.content_filter import EXTERNAL_SOURCE_TOOLS, sanitize_tool_output
from app.security.output_guard import audit_output
from app.security.tool_whitelist import validate_tool_call

logger = logging.getLogger("shoppingx.harness.safety")


@harness_hook("pre_tool_call", name="tool_whitelist", priority=1)
async def check_tool_whitelist(context: dict[str, Any]) -> dict[str, Any] | None:
    """L1 **断言**：工具名不在 ``FULL_TOOL_SET`` 里就报警——但不再拒。

    **拒绝这件事框架已经做了**：Toolkit 里没有的工具名，AgentScope 根本不会执行。这道闸
    301 会话 0 触发，自述也写着「正常永不开火」。它真正的价值是那条 error + metric：开火
    意味着模型被诱导幻觉出了工具名，或工具表被动态改过——这是**要被看见的异常**，不是
    要被执法的越界。2026-09-10 由 ``raise`` 降为告警（A4）。
    """
    tool_name = context.get("tool_name", "")
    if validate_tool_call(tool_name):
        return None
    metrics.record_security_event("tool_not_allowed")
    logger.error(
        "L1 工具白名单告警：tool=%r 不在 FULL_TOOL_SET 内（框架侧会自行拒绝执行）", tool_name
    )
    return None


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


@harness_hook("pre_tool_call", name="depth_gate", priority=10)
async def check_depth_permission(context: dict[str, Any]) -> dict[str, Any] | None:
    """深度**断言**：worker（depth≥1）碰了主 loop 专属工具就报警——但不再拦。

    **这不是闸，是发放范围的看门狗。** 真正的边界在 ``tool_registry._SEARCH_TOOLS`` /
    ``_TRADE_TOOLS``：读写切分后 worker 的 Toolkit 里根本没有这些工具对象，模型连 schema
    都看不到，本函数在 split 模式下**结构性不可达**（301 会话 0 触发）。它唯一的价值是
    「将来有人往 worker 的发放范围里加错工具时，日志里有一条 error」——那是断言的职责。

    2026-09-10 由 ``raise`` 降为 ``logger.error + metrics``，两点后果如实记在这：

    - 一条从未被走过的异常路径消失了。留着它，等于让「边界靠什么保证」有两个答案。
    - ``WORKER_MODE=clone`` 对照实验里 worker 拿的是全集，此前被本闸拦住；现在放行。
      这反而让 clone 更忠实于它要复现的历史形态（M2/M9 的同质 fork 本来就没有这道闸）。
    """
    if current_fork_depth() < 1:
        return None
    tool_name = context.get("tool_name", "")
    if tool_name in DEPTH0_ONLY_TOOLS or tool_name in MAIN_ONLY_CONTEXT_TOOLS:
        metrics.record_security_event("worker_tool_scope_violation")
        logger.error(
            "发放范围异常：worker（depth=%d）拿到了主 loop 专属工具 %r——"
            "检查 tool_registry 的 _SEARCH_TOOLS / _TRADE_TOOLS 是否加错了工具",
            current_fork_depth(),
            tool_name,
        )
    return None


@harness_hook("post_tool_call", name="truncate_result", priority=10)
async def truncate_result(context: dict[str, Any]) -> dict[str, Any] | None:
    """工具返回过长时按 token 预算截断并留提示。

    必须排在 ``result_nudges`` 之前：先截断、再追加系统提示，否则刚贴上的提示会被截掉。
    """
    guard = guard_of(context)
    result = context.get("tool_result")
    if guard is None or not isinstance(result, str):
        return None
    truncated = truncate_tool_result(result, guard.max_tool_tokens)
    if truncated != result:
        context["tool_result"] = truncated
        return context
    return None


# Harness 内部控制文案的标记：单一事实源在 sentinels.INTERNAL_MARKERS（哨兵与清洗表共用，
# 新增哨兵在那边登记即自动进清洗）。模型偶尔会把整段哨兵抄进面向用户的回复里。
_INTERNAL_MARKERS = INTERNAL_MARKERS

# 一整行以内部标记开头 → 整行删掉（模型通常是整段抄）。
_MARKER_LINE = re.compile(
    r"^[ \t>*_-]*(?:" + "|".join(re.escape(m) for m in _INTERNAL_MARKERS) + r").*$",
    re.MULTILINE,
)


@harness_hook("on_session_end", name="output_guard", priority=10)
async def audit_final_output(context: dict[str, Any]) -> dict[str, Any] | None:
    """输出审核：把内部控制文案从面向用户的最终回复里剔掉。

    改写走 ``context["final_answer"]``，由 ``run_agent()`` 消费——这条回写通路必须留着，否则
    Hook 改了没人读（refdocs 17-2 §4.1）。
    """
    final = context.get("final_answer")
    if not isinstance(final, str) or not final:
        return None
    if not any(marker in final for marker in _INTERNAL_MARKERS):
        return None

    cleaned = _MARKER_LINE.sub("", final)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    logger.warning("输出审核：最终回复含内部控制文案，已清洗 %d 字符", len(final) - len(cleaned))
    context["final_answer"] = cleaned or final  # 全被清空则宁可回原文，不给用户空白
    return context
