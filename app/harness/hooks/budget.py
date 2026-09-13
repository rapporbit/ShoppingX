"""预算：把「再找找更好的」这个动机用额度兜死，prompt 只当辅助。

    pre_think       20  budget_router         按全树成本定档：换模型 / 注入 hint / FALLBACK 不调 LLM
    pre_tool_call   15  websearch_gate        有候选就不该拿 web_search 找更好（效率闸，带逃生门）
    pre_tool_call   30  search_authority_gate 子搜够 / 主 loop fork 后直搜 → 拦（**读自增前计数**）
    pre_tool_call   33  token_budget_gate     minimal 档收走成本放大器（**必须早于 35**）
    pre_tool_call   35  fork_budget_gate      task_dispatch 次数上限，charge 即扣槽
    pre_tool_call   45  retrieval_charge_gate 检索计数自增（**必须晚于 30**）；越线软收敛 / 硬挡

**顺序契约（易碎，勿动）**：``search_authority``(30) 读的是 ``item_search_calls`` 的自增前值，自增在
``retrieval_charge``(45)；谁把自增挪到 30 之前，子 Agent 的「恰好放行一次」就塌成「放行 0 次」。
``token_budget``(33) 必须早于 ``fork_budget``(35)：fork 闸 charge 即扣槽，反过来 minimal 档下
一次被拒的
fork 会先烧掉唯一的并行额度。

**效率闸 vs 安全闸**（逃生门见 ``middleware._try_escape``）：依据推定的（websearch、postfork 直搜）
声明 ``escape_key``，连拒 2 次放行；依据事实的（子搜上限 / token / fork / 检索预算）永远硬拒。
**预算的定义住在哪（消费在本文件，定义分两个包，改额度先找对地方）**：
- 检索：全树计数与 web_search 任务配额在 ``app/agent/retrieval_budget.py``；上限 / 复用轮小预算 /
  子搜上限 / 工具集合在 ``app/harness/budgets.py``。
- fork：派发次数 ``ForkBudget`` 与并发信号量在 ``app/harness/budgets.py``；深度上限在
  ``app/agent/fork_guard.py``（读 ``app/agent/limits.py``）。
- token / 成本：全树成本与四档 ``Tier`` 在 ``app/agent/token_budget.py`` / ``model_router.py``。
- 一次失控最多烧多少（超时 / max_iters / 深度 / 派发数）：``app/agent/limits.py`` 一页看全。
"""

from __future__ import annotations

import logging
from typing import Any

from app.agent import model_router
from app.agent.fork_guard import current_fork_depth
from app.agent.model_router import Tier, current_tier
from app.agent.retrieval_budget import (
    charge_tree_retrieval,
    note_web_search,
    web_search_allowed,
)
from app.api.context import get_retrieval_mode
from app.harness.budgets import (
    COST_AMPLIFIER_TOOLS,
    FORK_TOOLS,
    RETRIEVAL_TOOLS,
    REUSE_RETRIEVAL_BUDGET,
    SUB_ITEM_SEARCH_CAP,
    get_fork_budget,
)
from app.harness.middleware import HookRejectSignal, harness_hook
from app.harness.msgs import system_message
from app.harness.sentinels import (
    BUDGET_HARD_DENIED,
    MAIN_POSTFORK_SEARCH_DENIED,
    SUB_SEARCH_EXHAUSTED,
    WEBSEARCH_DENIED,
    retrieval_exhausted,
    reuse_backfill_note,
    reuse_retrieval_exhausted,
)
from app.harness.signals import candidate_count
from app.harness.state import GuardState, guard_of
from app.observability import metrics
from app.tools._bundle import resolve_slot

logger = logging.getLogger("shoppingx.harness.budget")


@harness_hook("pre_tool_call", name="websearch_gate", priority=15)
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


@harness_hook("pre_tool_call", name="search_authority_gate", priority=30)
async def check_search_authority(context: dict[str, Any]) -> dict[str, Any] | None:
    """item_search 的「耗尽夺权」硬闸：搜够了就真拦下（不靠模型自觉）。

    - 子（depth≥1）本平台 item_search 已达 ``SUB_ITEM_SEARCH_CAP`` → ``SUB_SEARCH_EXHAUSTED``。
    - 主 loop（depth==0）已跑过并行 fork → ``MAIN_POSTFORK_SEARCH_DENIED``（fork 即检索阶段）。

    读的是 ``item_search_calls`` 的**自增前**值——自增在 ``retrieval_charge``(priority 45)。
    """
    if context.get("tool_name") != "item_search":
        return None
    guard = guard_of(context)
    if guard is None:
        return None

    depth = current_fork_depth()
    if depth >= 1 and guard.item_search_calls >= SUB_ITEM_SEARCH_CAP:
        raise HookRejectSignal(SUB_SEARCH_EXHAUSTED, raw=True)
    if depth == 0:
        budget = get_fork_budget()
        if budget is not None and budget.dispatched:
            # 结果感知的解锁（棘轮闸的三个出口）——否则「fork 跑过」就永久锁死直搜：
            # 1) 候选池是空的：整轮 fork 失败/超时/全空召回时，「候选已汇集」的哨兵是假话，
            #    直调 item_search 是仅剩的补救通路。放行（检索总量预算在 45 照常兜底）。
            # 2) phase_rollback 发过补搜授权：阶段机指路「回 SEARCHING 重搜」，闸必须放行，
            #    否则回退腿撞 postfork 哨兵就是死路。授权一次消费一次。
            # 3) 套装按槽补搜：每个已登记槽 1 次直搜额度。「fork 即检索阶段结束」对 bundle
            #    不成立（某槽召回全是蹭词垃圾时要按槽重搜），且不开正规出口时同批并行补搜
            #    会被逃生门按到达顺序放行一半——谁被拦与需求无关（badcase 4c0ac682）。
            if candidate_count() == 0:
                return None
            if guard.postfork_search_grants > 0:
                guard.postfork_search_grants -= 1
                return None
            # 槽引用走全链路同一个 resolve_slot（id/精确名/漂移名，含懒读回）——闸和
            # item_search 的盖章必须是同一副眼睛，否则漂移名会在闸前被拒、闸后被认。
            # 额度按稳定 id 记账：同一槽换个写法不给第二次。
            slot = resolve_slot(str((context.get("tool_args") or {}).get("slot") or ""))
            if slot is not None and slot.id not in guard.slot_backfill_used:
                guard.slot_backfill_used.add(slot.id)
                return None
            # 效率闸，接统一逃生门：「fork 即检索阶段结束」是语义推定不是事实（fork 可能搜偏
            # 了品类 / 平台）。逃生放行后重新武装收线通告；检索总量预算（45 号闸）照常兜底。
            raise HookRejectSignal(
                MAIN_POSTFORK_SEARCH_DENIED,
                raw=True,
                escape_key="postfork:item_search",
                on_escape=lambda: guard.notified_transitions.discard("search_close"),
            )
    return None


@harness_hook("pre_tool_call", name="token_budget_gate", priority=33)
async def check_token_budget(context: dict[str, Any]) -> dict[str, Any] | None:
    """预算档位到 minimal 即收走成本放大器工具，只留收尾链。

    **priority=33，必须早于 fork_budget_gate(35)**：fork 闸的 charge 即扣槽（parallel 槽只有
    1 个），本闸若排在它之后，minimal 档下模型发起的 fork 会先被扣槽、再被本闸拒绝——被拒的
    尝试烧掉唯一的并行额度，还连带触发 postfork 直搜拦截。预算判定是纯读取（current_tier），
    放到 charge 前零代价。

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


@harness_hook("pre_tool_call", name="fork_budget_gate", priority=35)
async def check_fork_budget(context: dict[str, Any]) -> dict[str, Any] | None:
    """对 fork 元工具计入树级 fork 预算；耗尽即硬挡。无树作用域则放行。"""
    tool_name = context.get("tool_name", "")
    if tool_name not in FORK_TOOLS:
        return None
    budget = get_fork_budget()
    if budget is None:
        return None
    denied = budget.charge(tool_name)
    if denied is not None:
        raise HookRejectSignal(denied, raw=True)
    return None


@harness_hook("pre_tool_call", name="retrieval_charge_gate", priority=45)
async def charge_retrieval(context: dict[str, Any]) -> dict[str, Any] | None:
    """对「商品检索」工具计数，越预算则软收敛 / 硬挡。

    优先用跨 fork 树共享的全树计数（``charge_tree_retrieval``，按 session_dir 聚合，连子里的
    item_search 一起兜）；无 session 作用域（单测）回退 per-instance。

    常规轮（search / augment）：
    - ``count <= cap``：放行。
    - ``count == cap + 1``（刚越线）：**执行**，但在结果尾部追加强制收敛指令（软收敛）——
      经 ``context["converge_count"]`` 传给 post_tool_call 的 nudge Hook。
    - 再越线：硬挡，工具不执行。

    复用轮（planner 判 reuse）：全树 cap 收紧到 ``REUSE_RETRIEVAL_BUDGET``（≥1），预算内执行 +
    缀软线文案（``context["converge_note"]``），越线硬挡。reuse 从「禁止检索」降为「小预算
    检索」的理由见 docs/decisions/0001-阶段白名单降级为遥测.md。
    """
    tool_name = context.get("tool_name", "")
    if tool_name not in RETRIEVAL_TOOLS:
        return None
    guard = guard_of(context)
    if guard is None:
        return None

    if tool_name == "item_search":
        # 顺序契约：此自增必须排在 search_authority_gate(30) 之后。
        guard.item_search_calls += 1
    elif tool_name == "web_search":
        # 同一顺序契约：websearch_gate(15) 读自增前值判任务口径配额（已完成 < 配额即放行）。
        note_web_search()

    tree = charge_tree_retrieval()  # None=无 session 作用域
    if tree is None:
        guard.retrieval_count += 1
        count, cap = guard.retrieval_count, guard.retrieval_cap
    else:
        count, cap = tree, guard.tree_retrieval_cap

    # 复用轮小预算：planner 判 reuse 后本轮检索不被锁死，而是收紧到
    # REUSE_RETRIEVAL_BUDGET（≥1，永不为 0）——reuse 是假设不是承诺，模型确认
    # 旧候选不适用（如换品类）时第一次补搜就直接放行执行，不用攒拒绝换逃生。预算内执行并在结果
    # 尾部缀「搜完即收敛」的软线文案；越线硬挡。refine_backfill / phase_rollback 授权补搜时会把
    # mode 改写为 augment，本分支即不再命中、自动恢复全树预算。
    if get_retrieval_mode() == "reuse":
        if count <= REUSE_RETRIEVAL_BUDGET:
            context["converge_note"] = reuse_backfill_note(count, REUSE_RETRIEVAL_BUDGET)
            logger.info("复用轮补搜（%d/%d），执行并缀收敛提示", count, REUSE_RETRIEVAL_BUDGET)
            return context
        raise HookRejectSignal(reuse_retrieval_exhausted(count, REUSE_RETRIEVAL_BUDGET), raw=True)

    if count <= cap:
        return None
    if count == cap + 1:
        context["converge_count"] = count  # 软收敛：执行，但结果尾部追加强制收敛指令
        logger.info("检索预算软越线（%d/%d），追加强制收敛指令", count, cap)
        return context
    raise HookRejectSignal(retrieval_exhausted(count), raw=True)


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
