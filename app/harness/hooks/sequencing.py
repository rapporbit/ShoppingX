"""工具前置条件：一张 ``PREREQUISITES`` 表 + 一条写路径硬拒。

    pre_tool_call   12  trade_sequence_gate   cancel_order 前必须 query_order 过——**硬拒**
                                              （写操作代价不对称）
    pre_tool_call   25  sequencing_assertion  其余前置只警告
                                              （记入 assertions_failed，由 validation 汇总注入）

读路径不硬拒：有些场景确实要跳步（用户直接给了候选）；候选消费者以「登记表有候选」为真实前置。
"""

from __future__ import annotations

import logging
from typing import Any

from app.harness.middleware import HookRejectSignal, harness_hook
from app.harness.sentinels import (
    CANCEL_WITHOUT_QUERY,
)
from app.harness.signals import candidate_count

logger = logging.getLogger("shoppingx.harness.sequencing")


@harness_hook("pre_tool_call", name="trade_sequence_gate", priority=12)
async def check_trade_sequence(context: dict[str, Any]) -> dict[str, Any] | None:
    """取消订单前必须先查单——**硬拒**，不是警告。

    与 `step_validator` 的 sequencing 软断言是一对：那条给的是「通常应该先…」的提醒，对读操作
    够用；取消不是读操作。模型最典型的错法是从用户一句「把上次那单取消了」里直接编一个订单号
    调 cancel_order——编出来的号大概率不存在（那还好，会失败），但也可能**恰好命中另一张真单**。

    判据是本轮轨迹里有没有 query_order，不是「查到了什么」：查了发现不存在也算查过，那时模型
    收到的是查询工具的如实结果，它该做的是告诉用户查不到，而不是继续取消。
    """
    if context.get("tool_name") != "cancel_order":
        return None
    called: set[str] = context.get("called_tools", set())
    if "query_order" in called:
        return None
    raise HookRejectSignal(CANCEL_WITHOUT_QUERY, raw=True)


# ---------- Sequencing Assertion ----------

# 工具名 → 前置工具候选列表，**满足其一即可**（不是全部都要）。
#
# 检索的前置必须把 fork 通路算进去：本项目跨平台检索的主路径是 dispatch_tool /
# parallel_dispatch_tool 派子 Agent 去 item_search，主 loop 自己从头到尾可能一次 item_search
# 都没调过。只认 item_search 会让「fork 检索 → item_picker」这条正常链路每次都被误报顺序错误。
PREREQUISITES: dict[str, list[str]] = {
    # 「搜完直接收尾」（round3 刀 2）是合法序列：自动比价精挑（harness.autopick）走
    # after_tool_success 同一条管线，item_picker 照样进 called_tools，这条前置自然满足。
    "shopping_summary": ["item_picker"],
    # 派发工具名 L8 已统一成 task_dispatch。这里曾留着 dispatch_tool / parallel_dispatch_tool
    # 两个死名字——工具名对不上等于那条路径永不满足，「派发过所以有候选」的前置白写。
    "price_compare": ["item_search", "task_dispatch"],
    "shipping_calc": ["price_compare"],
    "item_picker": ["item_search", "task_dispatch"],
    # 取消前先查单。这里是**软**断言（注入一条警告），硬闸在本文件上方的 trade_sequence_gate：
    # 写操作的代价不对称，光警告拦不住一个已经打算取消的模型。
    "cancel_order": ["query_order"],
}

# 这些工具的真实前置是「登记表里有候选」，工具名只是达成它的若干条路径之一。
# 有候选即视为前置已满足——结构化信号比工具名可靠（候选也可能来自续聊的历史轮次）。
_CANDIDATE_CONSUMERS = frozenset({"item_picker", "price_compare"})


@harness_hook("pre_tool_call", name="sequencing_assertion", priority=25)
async def check_sequencing(context: dict[str, Any]) -> dict[str, Any] | None:
    """验证工具调用顺序是否满足前置条件（满足任一前置即通过）。

    不硬拒绝——有些场景确实需要跳步（如用户直接给了候选列表）。
    只注入警告让模型自己判断是否继续。
    """
    tool_name = context.get("tool_name", "")
    prerequisites = PREREQUISITES.get(tool_name)
    if not prerequisites:
        return None

    called: set[str] = context.get("called_tools", set())
    if any(p in called for p in prerequisites):
        return None

    if tool_name in _CANDIDATE_CONSUMERS and candidate_count() > 0:
        return None

    context.setdefault("assertions_failed", []).append(
        {
            "type": "sequencing",
            "tool": tool_name,
            "reason": (
                f"{tool_name} 通常在 {' 或 '.join(prerequisites)} 之后调用，但它们都还没执行过"
            ),
        }
    )
    logger.info("Sequencing warning: %s called before any of %s", tool_name, prerequisites)
    return context
