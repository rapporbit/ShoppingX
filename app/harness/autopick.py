"""检索合流后自动比价 + 精挑（延迟治理 round3 刀 2）。

**为什么**：改前主 loop 固定 5 轮（planner → 搜 → price_compare → item_picker → summary），
其中 price_compare / item_picker 两轮模型只是在「把上一步的结论搬进下一步的入参」——零决策、
纯解码（每轮 2~3s）。到手价是纯本地计算，精挑的条件（预算 / 排除 / 偏好）planner 早已确定性
落进会话 P_t，``item_picker`` 无参调用即按 P_t 执行。故把这两步下沉到工具层：检索类工具
（``item_search`` / ``task_dispatch(search)``）成功返回后**武装**，下一次模型调用前（pre_think，
即同轮多个并发派发全部合流之后）自动跑一遍，结果以 hint 注入，模型下一步即可 ``shopping_summary``。

**不动的东西**：``price_compare`` / ``item_picker`` 工具本身保留（模型仍可显式调，显式调即
解除武装）；套装轮（≥2 槽）不自动——它要经 ``ask_user`` 确认组成，入口不动；worker 不自动。
自动执行走与真实工具调用**同一条** post_tool_call 管线（截断 / 收线通告 / schema 断言 / 偏好
注入），信号（picks 数 / oncat / 阶段机）与模型亲手调完全一致。任何异常都吞掉并解除武装：
失效方向 = 退回改前的「模型自己调」，不会更差。

关 ``AUTOPICK=0`` 即回到改前行为（对照实验用）。
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

from app.harness.fork_guard import current_fork_depth
from app.harness.phase_machine import Phase, get_phase_machine
from app.harness.signals import candidate_count

if TYPE_CHECKING:  # pragma: no cover
    from app.harness.session import HarnessSession

logger = logging.getLogger("shoppingx.harness.autopick")

_SEARCH_TOOLS = frozenset({"item_search", "task_dispatch"})


def autopick_enabled() -> bool:
    return os.getenv("AUTOPICK", "1").strip().lower() not in {"0", "false", "off"}


def _is_bundle_turn() -> bool:
    from app.tools._bundle import get_session_bundle

    return len(get_session_bundle()) >= 2


def autopick_applies() -> bool:
    """本轮是否归自动比价精挑管：开关开 + 主 loop + 非套装轮。收线通告的措辞也看这个。"""
    return autopick_enabled() and current_fork_depth() == 0 and not _is_bundle_turn()


def arm_on_tool(s: HarnessSession, tool_name: str, tool_args: dict[str, Any]) -> None:
    """工具成功返回后的武装 / 解除：检索类武装，模型显式 item_picker 解除。"""
    if tool_name == "item_picker":
        s.autopick_armed = False
        return
    if tool_name not in _SEARCH_TOOLS:
        return
    if tool_name == "task_dispatch" and tool_args.get("subagent_type") != "search":
        return
    s.autopick_armed = True


async def maybe_autopick(s: HarnessSession) -> None:
    """武装态且有候选 → 依次跑 price_compare、item_picker，结果进 inject 通道。"""
    if not s.autopick_armed or not autopick_applies():
        return
    s.autopick_armed = False
    if candidate_count() == 0:
        return
    from app.harness.adapter import after_tool_success
    from app.tools._shell import _to_text
    from app.tools.item_picker import item_picker
    from app.tools.price_compare import price_compare

    # 阶段机先推到 COMPARING：item_picker 的「直接 shopping_summary 收尾」通告只在 COMPARING 发，
    # 而真实链路里这一步转移发生在下一次 post_reflect（晚于本次注入）。
    machine = get_phase_machine()
    if machine is not None and machine.phase is Phase.SEARCHING:
        machine.try_transition("candidates_available")

    texts: list[str] = []
    try:
        pc = await price_compare.ainvoke({})
        await after_tool_success(s, "price_compare", {}, _to_text(pc))
        texts.append(f"到手价已按收货国 {pc.dest_country or 'US'} 折算并回写全部候选。")
        pk = await item_picker.ainvoke({})
        texts.append(await after_tool_success(s, "item_picker", {}, _to_text(pk)))
    except Exception:  # noqa: BLE001 - 自动步骤失败就退回模型自己调，绝不打断主 loop
        logger.warning("autopick 失败，退回模型自行比价精挑", exc_info=True)
        return
    s.pending_inject.append(
        {
            "content": (
                "[系统已自动执行] 检索已合流，系统已按本轮约束（预算 / 排除 / 偏好）完成 "
                "price_compare 与 item_picker，无需再调这两个工具。item_picker 结果：\n"
                + "\n".join(texts)
            )
        }
    )
    logger.info("autopick 完成：picks=%d", s.last_picks)
