"""安全底线（硬，无逃生门，无开关）：内容过滤 / 结果截断 / 输出审核与脱敏。

    post_tool_call   5  content_filter    L3：外部来源工具返回里的提示词注入
    post_tool_call  10  truncate_result   过长结果按 token 预算截断
                                          （**必须早于任何追加提示的 Hook**）
    on_session_end  10  final_answer_audit 先把 Harness 内部控制文案从最终回复里剔掉（去噪），
                                           再做 L4 密钥 / 内网地址 / 服务器路径脱敏
                                           （曾是 output_guard(10) + output_audit(20) 两个 hook）

这里曾有两道 pre_tool_call 断言（tool_whitelist / depth_gate），2026-09-15 删：工具名不在 Toolkit
里框架就不执行，worker 的 Toolkit 里根本没有主 loop 专属工具——两条都是 301 会话零触发的运行时
看门狗。它们守的不变量改由 ``tests/test_toolkit_scope.py`` 在装配层断言，比运行时日志更早暴露。

输出审核刻意保守：只清洗模型鹦鹉学舌的哨兵文案，**不做** item_id 之类的正则脱敏——那些常是用户
真正想要的信息，宁可漏放不误杀。
"""

from __future__ import annotations

import logging
import re
from typing import Any

from app.harness.middleware import harness_hook
from app.harness.sentinels import (
    INTERNAL_MARKERS,
)
from app.harness.state import guard_of
from app.harness.truncation import truncate_tool_result
from app.observability import metrics
from app.security.content_filter import EXTERNAL_SOURCE_TOOLS, sanitize_tool_output
from app.security.output_guard import audit_output

logger = logging.getLogger("shoppingx.harness.safety")


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


async def audit_final_answer(context: dict[str, Any]) -> dict[str, Any] | None:
    """L4：最终回答里的密钥 / 内网地址 / 服务器路径 → 脱敏后再推给用户。

    在 ``audit_final_output``（洗 Harness 内部控制文案）之后跑：先去噪、再脱敏。
    改写走 ``context["final_answer"]``，由 ``run_agent()`` 消费。
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


@harness_hook("on_session_end", name="final_answer_audit", priority=10)
async def audit_final(context: dict[str, Any]) -> dict[str, Any] | None:
    """最终回复审核，顺序固定：先去噪（内部控制文案）、再脱敏（密钥 / 内网地址 / 路径）。"""
    guarded = await audit_final_output(context)
    audited = await audit_final_answer(context)
    return context if (guarded or audited) else None
