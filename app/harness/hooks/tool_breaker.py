"""工具级熔断：pre_tool_call 判定 + post_tool_call 计数（refdocs 17-2 §2.2 / §5.1）。

**和已有的依赖级熔断不是一回事**：``app/tools/web_search.py`` / ``app/recall/reranker.py`` 里的
``CircuitBreaker`` 包的是**某一次外呼**（tavily / siliconflow HTTP）。本 Hook 包的是**工具本身**
——某个工具无论因为什么原因（外呼挂了、参数总是错、下游数据缺失）连续失败到阈值，就短路掉它，
让 Agent 立刻收到「这个工具暂时不可用」的哨兵，改走别的路，而不是每轮都去踩同一个坑、把迭代
预算烧光。

失败的判定口径：工具抛异常。返回了「空结果」不算失败——那是业务信号（没搜到货），由漂移检测的
「探索发散」信号负责，不该熔断工具。
"""

from __future__ import annotations

import logging
from typing import Any

from app.harness.middleware import HookRejectSignal, harness_hook
from app.harness.sentinels import tool_breaker_open
from app.utils.circuit_breaker import CircuitBreaker
from app.utils.env import env_bool, env_int

logger = logging.getLogger("shoppingx.harness.tool_breaker")

TOOL_BREAKER_ENABLED = env_bool("HARNESS_TOOL_BREAKER", True)
_FAILURE_THRESHOLD = env_int("HARNESS_TOOL_BREAKER_THRESHOLD", 3)
_RECOVERY_TIMEOUT = float(env_int("HARNESS_TOOL_BREAKER_RECOVERY_SEC", 60))

# 每个工具一个断路器，进程级共享（跨会话累积失败——某平台 API 挂了就是挂了，不该每个会话重新试
# 三次）。``all_breakers()`` 会枚举它们，metrics 里能看到状态。
_breakers: dict[str, CircuitBreaker] = {}


def get_tool_breaker(tool_name: str) -> CircuitBreaker:
    """取（或懒建）某个工具的断路器。"""
    breaker = _breakers.get(tool_name)
    if breaker is None:
        breaker = CircuitBreaker(
            f"tool:{tool_name}",
            failure_threshold=_FAILURE_THRESHOLD,
            recovery_timeout=_RECOVERY_TIMEOUT,
        )
        _breakers[tool_name] = breaker
    return breaker


def reset_tool_breakers() -> None:
    """清空所有工具断路器（测试用）。"""
    for breaker in _breakers.values():
        breaker.reset()
    _breakers.clear()


@harness_hook("pre_tool_call", name="tool_breaker_gate", priority=48)
async def check_tool_breaker(context: dict[str, Any]) -> dict[str, Any] | None:
    """断路器 OPEN 且未到恢复窗口 → 快速失败，工具不执行。

    ``allow()`` **有副作用**（可能把 OPEN 推进到 HALF_OPEN 以放行一次探测），放行后必须成对地
    记一次成败，否则半开探测悬空、断路器再也回不到 CLOSED。

    所以本闸的 priority 必须是 pre_tool_call 里**最后一个**（48 > retrieval_charge 的 45）：排在
    前面的话，深度闸 / 阶段门 / 预算闸任何一道后置拒绝，都会让这次「已放行的探测」没有对应的成败
    记录。放在最后 → 只要它放行，工具就一定执行，成败一定会被记上。

    代价：被本闸拒绝时，前面 retrieval_charge 的检索计数已经自增了一次。这是可接受的——熔断本就是
    异常路径，且少算一次检索额度只会让 Agent 更早收敛，方向是安全的。
    """
    if not TOOL_BREAKER_ENABLED:
        return None
    tool_name = context.get("tool_name", "")
    if not tool_name:
        return None

    breaker = get_tool_breaker(tool_name)
    if not breaker.allow():
        logger.warning("工具 %s 处于熔断态，快速失败", tool_name)
        raise HookRejectSignal(tool_breaker_open(tool_name), raw=True)
    context["_breaker_armed"] = tool_name  # 已放行：适配器/下游必须成对记一次成败
    return context


@harness_hook("post_tool_call", name="tool_breaker_record", priority=5)
async def record_tool_outcome(context: dict[str, Any]) -> dict[str, Any] | None:
    """工具正常返回 → 记一次成功（复位失败计数 / 半开探测成功即恢复 CLOSED）。

    失败路径不在这里：工具抛异常时 post_tool_call 根本不会跑，由适配器捕获异常后
    调 ``record_failure`` 并重新抛出。
    """
    if not TOOL_BREAKER_ENABLED:
        return None
    tool_name = context.get("tool_name", "")
    if tool_name:
        get_tool_breaker(tool_name).record_success()
    return None
