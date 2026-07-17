"""同参数重复调用的结果回放（memoization）——「重复搜索」的机制性止损。

实测（trace 1ac9f76 等）模型会对同一意图换措辞重发检索，其中**参数完全相同**的那部分没有任何
新信息可拿：真执行一遍只是把同样的结果再算一次（Qdrant / 汇率表都是确定性的），还多付一次
工具 RT。本 Hook 把「本轮已成功执行过的 (tool, args)」的结果缓存在 GuardState 里，重复调用
直接回放当时的结果 + 一句「别再用相同参数重试」的提示，工具**不执行**。

边界（想清楚才敢回放）：
- **只回放幂等只读工具**（``_MEMO_TOOLS``）。planner 有副作用（写 P_t / 域 / retrieval 判定），
  item_picker 读的候选登记表在补搜 / 回退后会变（同参数 ≠ 同结果），都不回放。
- **per-loop 生命周期**：缓存挂在 GuardState 上，主 Agent 每轮新建实例即失效，子 Agent 各自
  独立——不存在跨轮 / 跨 loop 的陈旧回放。
- **回放先于预算计数**（priority 27 < retrieval_charge 45）：回放不是真实检索，不该消耗检索
  预算；但阶段门（20）仍在它之前——越阶段的调用轮不到回放，照旧被阶段哨兵拦。
- **回放不进 post_tool_call**（适配器对 ``_rejected`` 提前返回），LoopDetector 会漏看这些
  轮次，故在回放时手动把调用喂给 detector，打转照样触发换思路提示。
"""

from __future__ import annotations

import json
import logging
from typing import Any

from app.harness.middleware import HookRejectSignal, harness_hook
from app.harness.state import GuardState

logger = logging.getLogger("shoppingx.harness.tool_memo")

# 幂等只读工具：同参数在一轮任务内重复执行不产生任何新信息。
_MEMO_TOOLS = frozenset(
    {"item_search", "web_search", "category_insight", "price_compare", "shipping_calc"}
)


def _memo_key(tool_name: str, args: Any) -> str | None:
    """(tool, args) 的稳定指纹。args 序列化失败（理论不会，模型产出即 JSON）返回 None 不回放。"""
    try:
        return f"{tool_name}:{json.dumps(args, sort_keys=True, ensure_ascii=False, default=str)}"
    except (TypeError, ValueError):
        return None


def _guard(context: dict[str, Any]) -> GuardState | None:
    guard = context.get("_guard")
    return guard if isinstance(guard, GuardState) else None


@harness_hook("pre_tool_call", name="tool_memo_replay", priority=27)
async def replay_duplicate_call(context: dict[str, Any]) -> dict[str, Any] | None:
    """同参数重复调用 → 回放缓存结果，不真执行。

    priority=27：在阶段门（20）/ 顺序断言（25）之后——治理优先于省钱，越权调用照旧吃阶段
    哨兵；在检索计数（30/45）与熔断（48）之前——回放不占预算、不碰熔断窗。
    """
    guard = _guard(context)
    tool_name = context.get("tool_name", "")
    if guard is None or tool_name not in _MEMO_TOOLS:
        return None
    key = _memo_key(tool_name, context.get("tool_args"))
    if key is None:
        return None
    cached = guard.tool_result_cache.get(key)
    if cached is None:
        return None

    # 回放轮次照样喂 LoopDetector（正常路径由 result_nudges 喂，回放不走 post_tool_call）——
    # 模型盯着同一调用反复刷时，打转提示不能因为「都被回放了」而失明。置 _detector_fed
    # 告知适配器的拒绝路径别再喂一次（其余闸拒绝由适配器统一喂，见 awrap_tool_call）。
    context["_detector_fed"] = True
    looping = guard.detector.record(tool_name)
    note = (
        f"\n\n[Harness 提示] 本次调用与本轮此前一次 {tool_name} 的参数完全相同，"
        "已直接复用当时的结果（工具未重新执行）。相同参数不会带来新信息："
        "如需补充请更换参数，否则请基于已有结果进入下一步。"
    )
    if looping:
        note += f" {guard.detector.nudge_message(tool_name)}"
    logger.info("回放重复调用 %s（参数指纹命中，未执行）", tool_name)
    raise HookRejectSignal(cached + note, raw=True)


@harness_hook("post_tool_call", name="tool_memo_record", priority=15)
async def record_tool_result(context: dict[str, Any]) -> dict[str, Any] | None:
    """把成功执行的幂等工具结果记进回放缓存。

    priority=15：在截断（10）之后——缓存的就是模型实际看到的版本，回放不会把被截掉的
    大结果又灌回上下文；在分级提示（20）之前——收敛 / 打转 nudge 是针对「当时那次调用」的
    附言，不该跟着结果一起被回放。
    """
    guard = _guard(context)
    tool_name = context.get("tool_name", "")
    if guard is None or tool_name not in _MEMO_TOOLS:
        return None
    key = _memo_key(tool_name, context.get("tool_args"))
    result = context.get("tool_result")
    if key is None or not isinstance(result, str) or not result:
        return None
    guard.tool_result_cache[key] = result
    return None
