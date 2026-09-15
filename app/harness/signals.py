"""Harness 的信号源：「一次工具调用产生了什么」与「会话此刻处于什么状态」只有一份读法。

- 工具调用信号（纯函数）：候选数 / picks 数 / 参数摘要 / 观测出口。阶段机、补搜闸、漂移检测都按
  这里的口径做决策——多一份读法就多一种口径，那种偏差查起来能耗掉一整天。
- 会话状态信号（同步、零 IO、异常兜底回中性值）：候选登记表条数、会话级 P_t 的黑名单。Hook 判断
  「有没有候选」「有没有踩黑名单」一律走这里，不去 grep 工具返回的文本——返回格式会变，登记表不会。
"""

from __future__ import annotations

import json
import logging
from typing import Any

from app.agent.tracing import current_trace_id
from app.observability import alerts, metrics

logger = logging.getLogger("shoppingx.harness.signals")

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


# 会往候选池里添货的工具。
_SEARCH_TOOLS = frozenset({"item_search"})


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


def candidate_count() -> int:
    """当前会话候选登记表里的候选总数。读不到（无会话作用域等）返回 0。"""
    try:
        from app.api.context import get_session_dir
        from app.tools._candidates import _REGISTRY

        sd = get_session_dir()
        if sd is None:
            return 0
        return len(_REGISTRY.get(str(sd), {}))
    except Exception:
        logger.debug("candidate_count 读取失败", exc_info=True)
        return 0


def blacklist_terms() -> list[str]:
    """会话级 P_t 里的**硬** dislike 原子词（即黑名单）。

    只取 hard：soft dislike 在 item_picker 里是「减分」语义，出现在结果里不算违规。
    长期库的 dislike 需要 async + Store IO，不适合在每次工具返回后同步跑；会话级 P_t 已由
    curator 从长期偏好合流，覆盖绝大多数场景。
    """
    try:
        from app.api.context import get_session_pt

        pt = get_session_pt()
        if pt is None:
            return []
        return pt.dislike_terms()
    except Exception:
        logger.debug("blacklist_terms 读取失败", exc_info=True)
        return []


def blacklist_hits(text: str) -> list[str]:
    """``text`` 中命中的黑名单词（返回 P_t 原词，保序去重）。无黑名单或无命中返回 ``[]``。

    命中口径必须与执行层同一套：先 :func:`normalize_terms`（中↔英扩词）再 :func:`term_hits`
    （词边界 + 否定修饰）。初版是裸 ``t in lowered``，双向失效——P_t 的中文词（「塑料」）对
    英文结果永远匹不中（信号空转），英文词又无词边界与否定过滤（"plastic-free" 被数成违规）。
    检测层一旦弱于执行层（picker 走的是归一口径），这个兜底就永远兜不到执行层漏掉的东西。
    """
    if not text:
        return []
    terms = blacklist_terms()
    if not terms:
        return []
    lowered = text.lower()
    from app.utils.terms import normalize_terms, term_hits

    return [t for t in terms if t and any(term_hits(v, lowered) for v in normalize_terms([t]))]
