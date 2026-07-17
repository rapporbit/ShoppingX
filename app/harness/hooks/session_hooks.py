"""会话级 Hook：on_session_start 初始化 / on_session_end 输出审核。

由 ``run_agent()`` 显式调用（不在中间件内——它们不属于任何一次模型/工具调用）。

**输出审核（L4）刻意保守**：只清洗 Agent 把 Harness 自己的内部控制文案（哨兵 / 系统提示）当成
回答内容鹦鹉学舌给用户的情况——这是真实失败模式（模型常把 ``[系统提示] 精选清单已就绪…`` 原样
抄进收尾文案）。**不做** item_id / 商品编号之类的正则脱敏：那些在购物场景里往往是用户真正想要的
信息，宁可漏放也不误杀（与 ``forget_preference`` 的确定性删除同一取向）。
"""

from __future__ import annotations

import logging
import re
from typing import Any

from app.harness.middleware import harness_hook
from app.harness.phase_machine import Phase, get_phase_machine, set_phase_machine
from app.harness.phase_machine import PhaseStateMachine as _PSM
from app.harness.sentinels import INTERNAL_MARKERS

logger = logging.getLogger("shoppingx.harness.session")

# Harness 内部控制文案的标记：单一事实源在 sentinels.INTERNAL_MARKERS（哨兵与清洗表共用，
# 新增哨兵在那边登记即自动进清洗）。模型偶尔会把整段哨兵抄进面向用户的回复里。
_INTERNAL_MARKERS = INTERNAL_MARKERS

# 一整行以内部标记开头 → 整行删掉（模型通常是整段抄）。
_MARKER_LINE = re.compile(
    r"^[ \t>*_-]*(?:" + "|".join(re.escape(m) for m in _INTERNAL_MARKERS) + r").*$",
    re.MULTILINE,
)


@harness_hook("on_session_start", name="phase_init", priority=10)
async def init_phase_machine(context: dict[str, Any]) -> dict[str, Any] | None:
    """会话开始：把阶段机复位到 PLANNING。

    续聊复用同一 thread 时，ContextVar 里可能还留着上一轮跑到 CONCLUDING 的阶段机——不复位的话
    新一轮开局就只剩 shopping_summary 可用。
    """
    machine = get_phase_machine()
    if machine is None:
        set_phase_machine(_PSM())
    elif machine.phase is not Phase.PLANNING:
        machine.reset()
        logger.info("会话开始：阶段机复位到 planning")
    return None


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
