"""工具调用的信号提取与观测出口（两套运行时适配器共用的纯函数）。

从 ``agent_middleware`` 抽出来的原因很实在：批 0 迁移期 LangChain 与 AgentScope 两个适配器
并存，它们对「一次工具调用产生了什么信号」的读法必须**逐字相同**——否则迁移前后的阶段机会
基于不同的候选数 / picks 数做决策，那种偏差查起来能耗掉一整天。放这里让两边只能共用一份。

这些函数全是纯函数（除两个观测出口），不依赖任何一个运行时的消息类型。
"""

from __future__ import annotations

import json
from typing import Any

from app.agent.tracing import current_trace_id
from app.observability import alerts, metrics

# 单条行为摘要里工具参数的长度上限——摘要要进漂移检测的 LLM prompt，不能让长 query 撑爆。
_ARG_SUMMARY_MAX = 60


def _count_picks(result: str) -> int:
    """从 item_picker 的返回里数出真实 picks 数量。

    不能拿「item_picker 被调用过」当作「picks 已就绪」：候选全超预算 / 全被排除词淘汰时，
    item_picker 返回的是 ``picks: []``。把「调过」当成 picks_count=1 会有两个后果——阶段机在没有
    任何精选结果时就推进到 CONCLUDING（→ shopping_summary 空输出，正是 17-1 §5 列的失败模式），
    且回退闸从此永远不再触发（picks_count 恒 >0）。
    """
    try:
        data = json.loads(result)
        picks = data.get("picks")
        if isinstance(picks, list):
            return len(picks)
    except (json.JSONDecodeError, ValueError, AttributeError):
        pass
    # 走到这里说明不是合法 JSON——多半是被截断 Hook 按 token 预算截断了。截断只发生在长结果上，
    # 而长结果必然意味着 picks 非空；显式的空数组则一定完整出现在头部、不会被截掉。
    if '"picks": []' in result or '"picks":[]' in result:
        return 0
    return 1 if '"picks"' in result else 0


def _as_opt_int(value: object) -> int | None:
    """诊断侧信道字段的宽松取整：None 原样透传（= 本轮不适用），其余尽力转 int。"""
    if value is None:
        return None
    try:
        return int(value)  # type: ignore[call-overload]
    except (TypeError, ValueError):
        return None


# 会往候选池里添货的工具：主流程直搜，以及 fork 子 Agent 回传候选的两个派发口。
_SEARCH_TOOLS = frozenset({"item_search", "dispatch_tool", "parallel_dispatch_tool"})


def _count_candidates(result: str) -> int:
    """从检索类工具的真实返回里数出**本轮新召回**的候选数。

    **为什么不去数候选登记表**：登记表是个累积容器——上一轮的候选会被 ``load_candidates`` 读回来
    （供 item_picker 按 id hydrate），一旦拿它的总数当「本轮搜到了东西」的进展信号，换品类那轮就会
    被旧候选骗过去：planner 一跑完，阶段机看见「已有 12 件候选」直接推进 COMPARING，模型想搜键盘
    却发现 item_search 在 COMPARING 不放行。
    「仓库里有什么」与「这趟活干了什么」是两回事，不该共用一个计数器——和 :func:`_count_picks`
    坚持从 item_picker 的真实返回里数 picks 是同一条原则。
    """
    try:
        data = json.loads(result)
    except (json.JSONDecodeError, ValueError):
        # 子 Agent 回传的是自然语言总结（非 JSON）：数不出来就不计数。宁可少算不可多算——
        # 多算会把阶段机推过头，少算最多让模型多搜一次。
        return 0
    if not isinstance(data, dict):
        return 0
    cands = data.get("candidates")
    if isinstance(cands, list):
        return len(cands)
    total = data.get("total_recall")
    return total if isinstance(total, int) else 0


def _observe_tool(tool_name: str, elapsed_sec: float, status: str) -> None:
    """一次工具执行的可观测出口：喂 Prometheus 指标 + 喂 RT 告警窗。**同一个计时，两个消费者。**

    别把告警窗塞进 ``metrics.record_tool`` 里：``alerts`` 要读 ``metrics.SECURITY_EVENTS`` 做安全
    事件告警，反向再依赖就成了循环 import。打点位置只有这一处，两边取的是同一个 ``elapsed``，
    数据源不会漂移。

    **只有 ok 的调用进 RT 窗口。** 失败调用的耗时（尤其断路器 OPEN 时 ~0ms 的快速失败）会把 P95
    拉低，在故障最严重的时候反而报「已恢复」。错误面由 ``TOOL_CALLS{status="error"}`` 与断路器
    告警规则覆盖。
    """
    metrics.record_tool(tool_name, elapsed_sec, status)
    if status == "ok":
        # trace_id 一并存进窗口：告警触发时能直接给出「最慢那次」的 Langfuse 链接。
        alerts.record_tool_sample(tool_name, elapsed_sec * 1000.0, current_trace_id())


def _summarize_call(tool_name: str, args: Any) -> str:
    """把一次工具调用摘成 ``tool_name(参数文本)``。

    行为摘要必须带上参数，不能只有工具名：漂移检测的「目标遗忘」信号要拿用户 query 的关键词去
    匹配 Agent 最近在做什么，而工具名（item_search / price_compare）里永远不含 query 关键词——
    只喂工具名，命中数恒为 0，信号恒真。参数里的检索词才是模型「Think 的产物」。
    """
    if not isinstance(args, dict) or not args:
        return tool_name
    parts = [v.strip() for v in args.values() if isinstance(v, str) and v.strip()]
    if not parts:
        return tool_name
    return f"{tool_name}({' '.join(parts)[:_ARG_SUMMARY_MAX]})"


def _as_text(value: Any) -> str:
    """ToolMessage.content 的类型是 ``str | list[...]``，统一收敛成 str 再交给 Hook。"""
    return value if isinstance(value, str) else str(value)
