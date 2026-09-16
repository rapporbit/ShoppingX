"""预算：把「再找找更好的」这个动机用额度兜死，prompt 只当辅助。

    pre_think       20  budget_router  按全树成本定档：换模型 / 注入 hint / FALLBACK 不调 LLM
    pre_tool_call   30  spend_gate     token 档位：minimal 档收走成本放大器
    pre_tool_call   45  search_gate    web_search 用途门 + research 配额门 → 检索计数自增 /
                                       越线软收敛 / 硬挡

**顺序契约**：spend 在 search 之前——预算拒绝是事实闸，不该先给被拒的调用记一次检索。
（A4 删子 Agent 时，fork 次数闸、子搜上限、postfork 直搜闸随之删除。）

**效率闸 vs 安全闸**（逃生门见 ``middleware._try_escape``）：依据推定的（websearch）声明
``escape_key``，连拒 2 次放行；依据事实的（token / 检索预算）永远硬拒。
**预算的定义住在哪（消费在本文件，定义分两个包，改额度先找对地方）**：
- 检索：全树计数、web_search 任务配额、research 搜索配额（三本账互不透支）在
  ``app/harness/retrieval_budget.py``；上限 / 工具集合在 ``app/harness/budgets.py``。
- token / 成本：全树成本与四档 ``Tier`` 在 ``app/harness/token_budget.py`` / ``model_router.py``。
- 一次失控最多烧多少（超时 / max_iters）：``app/agent/limits.py`` 一页看全。
"""

from __future__ import annotations

import logging
from typing import Any

from app.harness import model_router
from app.harness.budgets import (
    COST_AMPLIFIER_TOOLS,
    RETRIEVAL_TOOLS,
)
from app.harness.middleware import HookRejectSignal, harness_hook
from app.harness.model_router import Tier, current_tier
from app.harness.msgs import system_message
from app.harness.retrieval_budget import (
    RESEARCH_SEARCH_QUOTA,
    charge_research,
    charge_tree_retrieval,
    note_web_search,
    research_remaining,
    web_search_allowed,
)
from app.harness.sentinels import (
    BUDGET_HARD_DENIED,
    WEBSEARCH_DENIED,
    research_quota_denied,
    retrieval_exhausted,
)
from app.harness.state import GuardState, guard_of
from app.observability import metrics

logger = logging.getLogger("shoppingx.harness.budget")


async def check_websearch(context: dict[str, Any]) -> dict[str, Any] | None:
    """web_search 门控：独立知识查询 / 任务口径配额 / 购物流程空召回时放行；其余有候选就拦。

    堵住「想找更好」从 web_search 漏出来——它不是「找更好商品」的渠道。evaluate /
    category_intel 任务另有小配额（``WEB_SEARCH_TASK_QUOTA``，planner 落 session 的确定性
    tasks 判据）：prompt 明确要这些任务补 web_search 口碑，不该罚它们走逃生门白花 2 轮往返。

    **效率闸，接统一逃生门**：「有候选就不需要外部信息」是动机推定不是事实——用户完全可能
    在有候选后要「查查这几款的评测」。模型连拒 2 次还坚持即放行；web_search 属
    ``RETRIEVAL_TOOLS``，逃生后仍被检索预算（45 号闸）兜底。
    """
    if context.get("tool_name") != "web_search":
        return None
    if not web_search_allowed():
        raise HookRejectSignal(WEBSEARCH_DENIED, raw=True, escape_key="web_search")
    return None


async def check_research(context: dict[str, Any]) -> dict[str, Any] | None:
    """research 配额门：按**本次会发几条搜索**预扣会话额度，不够就拦。

    与 web_search 那道门的区别在判据性质：web_search 的门是**动机推定**（「有候选就不需要外部
    信息」不一定成立），所以接逃生门、连拒 2 次放行；这道门是**事实**（额度还剩几条是数出来的），
    照安全闸的规矩硬拒、不给 escape_key。

    额度不够时给的是**可执行的出路**（「只保留最关键的 N 个对象重调一次」）而不是一句禁令——
    research 的入参天然可缩，模型缩了就能过，比换措辞硬撞同一道闸有用。
    """
    if context.get("tool_name") != "research":
        return None
    from app.tools.research import normalize_targets  # 延迟 import：harness 不在顶层依赖 tools

    args = context.get("tool_args")
    planned = len(normalize_targets((args or {}).get("targets")))
    if planned <= 0:
        return None  # 空 targets 交给工具自己回「没有给出研究对象」，别在闸上耗额度
    if not charge_research(planned):
        raise HookRejectSignal(
            research_quota_denied(planned, research_remaining(), RESEARCH_SEARCH_QUOTA),
            raw=True,
        )
    return None


async def check_token_budget(context: dict[str, Any]) -> dict[str, Any] | None:
    """预算档位到 minimal 即收走成本放大器工具，只留收尾链。

    **从 minimal 就开始拦，而不是等撞线（hard）才拦**：撞线意味着预算已经是 0，那时连收尾的
    shopping_summary（它内部还要调一次 LLM 生成文案）都付不起了。minimal 档（剩余 <20%）收权，
    正好把最后那点预算留给收尾链——保留便宜的精挑 / 终结工具，让任务能「花得起地」结束，
    而非硬停丢掉已收敛的候选。

    refdocs 16-4 §3.3 在这一档只往 system prompt 里注入一句「不要再检索了」。本项目照注入
    （见 ``budget_router``），但同时把工具真收走——**机制兜底优于提示词**，弱模型读不懂 hint 的
    时候，闸还在。
    """
    tool_name = context.get("tool_name", "")
    if tool_name in COST_AMPLIFIER_TOOLS and current_tier() >= Tier.MINIMAL:
        raise HookRejectSignal(BUDGET_HARD_DENIED, raw=True)
    return None


async def charge_retrieval(context: dict[str, Any]) -> dict[str, Any] | None:
    """对「商品检索」工具计数，越预算则软收敛 / 硬挡。

    优先用会话级全树计数（``charge_tree_retrieval``，按 session_dir 聚合）；无 session 作用域
    （单测）回退 per-instance。

    - ``count <= cap``：放行。
    - ``count == cap + 1``（刚越线）：**执行**，但在结果尾部追加强制收敛指令（软收敛）——
      经 ``context["converge_count"]`` 传给 post_tool_call 的 nudge Hook。
    - 再越线：硬挡，工具不执行。
    """
    tool_name = context.get("tool_name", "")
    if tool_name not in RETRIEVAL_TOOLS:
        return None
    guard = guard_of(context)
    if guard is None:
        return None

    if tool_name == "item_search":
        guard.item_search_calls += 1
    elif tool_name == "web_search":
        # 同一顺序契约：check_websearch 读自增前值判任务口径配额（已完成 < 配额即放行）。
        note_web_search()

    tree = charge_tree_retrieval()  # None=无 session 作用域
    if tree is None:
        guard.retrieval_count += 1
        count, cap = guard.retrieval_count, guard.retrieval_cap
    else:
        count, cap = tree, guard.tree_retrieval_cap

    if count <= cap:
        return None
    if count == cap + 1:
        context["converge_count"] = count  # 软收敛：执行，但结果尾部追加强制收敛指令
        logger.info("检索预算软越线（%d/%d），追加强制收敛指令", count, cap)
        return context
    raise HookRejectSignal(retrieval_exhausted(count), raw=True)


@harness_hook("pre_tool_call", name="spend_gate", priority=30)
async def check_spend(context: dict[str, Any]) -> dict[str, Any] | None:
    """token 档位：minimal 档收走成本放大器。"""
    return await check_token_budget(context)


@harness_hook("pre_tool_call", name="search_gate", priority=45)
async def check_search(context: dict[str, Any]) -> dict[str, Any] | None:
    """web_search 用途门 + research 配额门 → 检索计数与越线处理。

    三道各判各的工具、互不影响：``check_websearch`` 只看 web_search，``check_research`` 只看
    research（它不在 ``RETRIEVAL_TOOLS`` 里，后面的 ``charge_retrieval`` 对它直接返回 None）。
    """
    await check_websearch(context)
    await check_research(context)
    return await charge_retrieval(context)


@harness_hook("pre_think", name="budget_router", priority=20)
async def route_by_budget(context: dict[str, Any]) -> dict[str, Any] | None:
    """按剩余预算定档，把决策写进 context 交给适配器执行（Hook 决策、适配器落地）。

    三个出口，都不在这里直接操作模型——Hook 拿不到 ``ModelRequest``：

    - ``model_tier``：适配器按档位名解析出本运行时的模型（lite / minimal 档换便宜模型）
    - ``messages`` 追加 hint：minimal 档让模型自己也知道该收了
    - ``fallback_answer``：适配器**跳过模型调用**，直接把这段文本当 AIMessage 返回，loop 自然终止

    档位只降不升（成本单调增），所以每档只上报一次 metric——用 ``GuardState.last_tier`` 去重，
    否则一个 20 轮的任务会把 minimal 档记 15 次，降级率统计直接失真。

    全程走 ``model_router.xxx`` 而不是 ``from ... import xxx``：档位依赖全树成本，每次模型调用后都在
    变，必须现算；模块级引用也让单测能 monkeypatch 掉整条链（import 绑定的名字打不中）。
    """
    guard = context.get("_guard")
    tier = model_router.current_tier()

    entered_new_tier = not isinstance(guard, GuardState) or tier.label != guard.last_tier
    if isinstance(guard, GuardState) and entered_new_tier:
        if tier > Tier.MAIN:  # 只记降级，不记「留在 main」
            metrics.record_tier_change(tier.label)
            logger.info("预算降档：%s → %s", guard.last_tier, tier.label)
        guard.last_tier = tier.label

    if tier is Tier.MAIN:
        return None

    if tier is Tier.FALLBACK:
        # 连一次 LLM 调用都付不起了：用已有候选拼一个诚实的回答，不再进模型。
        context["fallback_answer"] = model_router.build_fallback_answer(
            context.get("original_query", "")
        )
        logger.warning("预算耗尽，走 fallback 规则兜底（不调 LLM）")
        return context

    # Hook 只产**档位名**，由适配器解析成本运行时的模型——「Hook 决策、适配器落地」的分工。
    # 这里曾并存一个 ``model_override`` 键（装模型**对象**），生产代码零处读、只剩每次降档白
    # 构造一个对象外加「这里在换模型」的假象，已删（批 A2）。
    context["model_tier"] = "lite"

    if tier is Tier.MINIMAL and entered_new_tier:
        # 只在**进入** minimal 那一轮注入：hint 经 persist_messages 落 state 后长驻历史，
        # 每轮重复注入只会攒出一摞相同提醒（且每次都斩断一次缓存前缀）。档位只降不升，
        # 「进入过」等价于「此后每轮都看得到」。
        messages = context.get("messages")
        if isinstance(messages, list):
            hint = system_message(model_router.MINIMAL_HINT, context)
            messages.append(hint)
            context.setdefault("persist_messages", []).append(hint)
    return context
