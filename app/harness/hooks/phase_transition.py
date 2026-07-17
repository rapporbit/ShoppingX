"""阶段转移（post_reflect）+ 收线通告（post_tool_call）：推进 / 回退对话阶段并当场告知模型。

转移信号由 HarnessAgentMiddleware 从可靠数据源（工具名 + 候选登记表）填入 context：
- planner_output_ready (bool)：planner 已执行 → PLANNING → SEARCHING
- total_candidates (int)：候选登记表条数 > 0 → SEARCHING → COMPARING
- picks_count (int)：item_picker 已执行 → COMPARING → CONCLUDING

特殊回退：COMPARING 连续 2 轮无进展 → 回退 SEARCHING；薄复用（<3 件）→ 补搜。

**状态与通告分离**：转移本体在 post_reflect（``try_phase_transition``），对模型的「收线」
通告在 post_tool_call（``append_transition_notice``，缀在触发转移的工具结果尾部）——
post_reflect 的 inject 通道要到再下一轮才被模型看见，晚一轮（perf-audit-r3 实测撞哨兵）。
"""

from __future__ import annotations

import logging
from typing import Any

from app.agent.fork_guard import current_fork_depth
from app.api.context import get_retrieval_mode, get_session_tasks, set_retrieval_mode
from app.harness.budgets import REUSE_RETRIEVAL_BUDGET
from app.harness.middleware import harness_hook
from app.harness.phase_machine import Phase, get_phase_machine
from app.harness.state import GuardState

logger = logging.getLogger("shoppingx.harness.phase_transition")

_ROLLBACK_THRESHOLD = 2


def _reset_drift_counters(context: dict[str, Any]) -> None:
    """阶段转移成功 → 重置漂移的「连续」类计数器（refdocs 17-4 §9 的跨章耦合）。

    阶段推进是实打实的进展信号：planner 出了结构化字段、候选进了登记表、精挑出了结果。此前
    攒下的「连续空结果」「连续严重漂移」是针对上一阶段的判断，不该带进新阶段继续累积——否则
    「搜了三次空 → 换个方向搜到了 → 进 COMPARING」的正常曲折，会被算成仍在发散。

    ``blacklist_violations`` **不重置**：推荐面出现用户明确排除的属性，不因阶段推进而变得可接受。
    """
    state = context.get("_drift_state")
    if state is None:
        return
    state.consecutive_empty_results = 0
    state.consecutive_severe = 0


@harness_hook("post_reflect", name="phase_transition", priority=40)
async def try_phase_transition(context: dict[str, Any]) -> dict[str, Any] | None:
    """根据当前执行状态判断是否触发阶段转移。仅 depth 0 生效。

    priority=40 排在 drift_detector（20）之后：本轮漂移判定基于「转移前」的计数器，判完再重置。
    """
    if current_fork_depth() >= 1:
        return None

    machine = get_phase_machine()
    if machine is None:
        return None

    current = machine.phase
    moved = False

    if current == Phase.PLANNING:
        if context.get("planner_output_ready"):
            moved = machine.try_transition("planner_output_ready")
            # planner 判定 reuse（用户只是在上一轮结果上收紧条件）→ 本轮不检索，直接跳到 COMPARING
            # 让 item_picker 拿到准入。否则模型为了「挣到」精挑资格，得先把 item_search 白跑一遍
            # （实测整条链重跑要多花近百秒，拿回的还是同一批商品）。
            # 判据是 planner 的**显式意图判定**，不是「候选池里有没有货」——有货 ≠ 本轮是追问。
            # 对模型的「检索已跳过」通告不在这里发：post_reflect 跑在下一次模型调用之后，通告
            # 要等再下一轮才被看见——晚一轮（perf-audit-r3 实测）。它由 transition_notice 在
            # planner 结果尾部当场缀上（post_tool_call），这里只做状态机本体的推进。
            if moved and get_retrieval_mode() == "reuse":
                machine.set_phase(Phase.COMPARING)

    elif current == Phase.SEARCHING:
        # 「检索收线」通告同样不在这里发（晚一轮，理由见上）——由 transition_notice 缀在
        # 首个非空检索结果尾部，模型下一次解码当场看见。这里只推进状态机。
        if context.get("total_candidates", 0) > 0:
            moved = machine.try_transition("candidates_available")

    elif current == Phase.COMPARING:
        if context.get("picks_count", 0) > 0:
            moved = machine.try_transition("picks_ready")

    if moved:
        _reset_drift_counters(context)
        return context

    return None


_REFINE_MIN_PICKS = 3


def _reuse_backfill_due(picks: int, must_hits: int | None) -> bool:
    """复用轮「旧池不够用」：量不够（<3 件）或质全空（本轮新硬条件池内 0 命中）。"""
    return picks < _REFINE_MIN_PICKS or must_hits == 0


def _pollution_backfill_due(mode: str, oncat: int | None, offcat: int | None) -> bool:
    """污染分支（**首搜轮也生效**）：品类门把池子大半判为跨品类混入、剩下相符的不够出清单。

    手表 badcase（thread 6718ed65）：检索词 formal dress watch men business 里的通用正装词把
    西装/皮鞋/马甲一并召回，10 条里只有 2 条真手表——品类一致性门正确沉底了 8 条，但没有任何
    机制补货，径直收尾出了 2 件的清单。「门杀得对」和「池子该补」是两件事，这里管后者。

    三个条件缺一不可：
    - ``mode != "augment"``：已补搜过一次就不再回退（augment 由补搜闸自己写，天然只触发一次）；
    - ``oncat < 3``：池内品类相符的候选不够出清单（picks 件数看不出——沉底垃圾也占 picks 名额）；
    - ``offcat > oncat``：污染占主导。**刻意不给「池子小但干净」触发**（oncat=2、offcat=0 是库存
      稀疏，不是检索词的错，重搜同样的词只会拿回同样的池子）。
    """
    return (
        mode != "augment"
        and oncat is not None
        and oncat < _REFINE_MIN_PICKS
        and (offcat or 0) > oncat
    )


def _hard_cull_backfill_due(
    mode: str, picks: int, excluded: int | None, over_budget: int | None
) -> bool:
    """硬淘汰杀池分支（**首搜轮也生效**）：预算 / 排除词把池子淘汰到不够出清单。

    与污染分支同构的三条件：已补搜过（augment）不再回退；``picks < 3`` 不够出清单；
    ``淘汰数 > picks`` 即硬约束才是池子空的主因——**刻意不给「池子小但干净」触发**
    （picks=2、淘汰 0 是库存稀疏，重搜同样的词只会拿回同一批）。硬淘汰杀池则不同：
    池子是按相关性召回的 top-k，不是按「预算内的相关性」——库里完全可能有预算内 /
    不含排除材质、却排在 k 名开外的商品，带 price_usd_max / 换开检索词补搜捞得回来
    （指路话术见 transition_notice，判据两边共用本函数）。
    诊断缺席（None）不触发——失效方向中性，与侧信道的降级语义一致。
    """
    culled = (excluded or 0) + (over_budget or 0)
    return (
        mode != "augment"
        and (excluded is not None or over_budget is not None)
        and picks < _REFINE_MIN_PICKS
        and culled > picks
    )


@harness_hook("post_reflect", name="refine_backfill", priority=39)
async def check_refine_backfill(context: dict[str, Any]) -> dict[str, Any] | None:
    """复用轮（reuse）精挑后候选太少 → 补搜一次。**治复用带来的召回损失。**

    复用旧候选是有损的：旧候选是「上一轮那句 query」下的 top-k，库里完全可能有更贴合本轮新条件、
    却在旧 query 下排在 k 名开外、从没进过池子的商品——在旧池子里过滤多少遍也捞不出来。
    「旧池不够用」有两种表现，各配一个触发条件：
    - **量不够**：过滤完剩 <3 件。剩得够多时复用的召回损失可以接受（省下近百秒）。
    - **质全空**（``must_have_hits == 0``）：picks 件数正常、但本轮新硬条件在池内**一件都没命中**。
      picker 的 must_have 是加分不淘汰（防误杀空结果），8 件素色候选对「必须刺绣」照样原样返回
      8 件——件数信号对这种假阳性是瞎的（实测 bad case：「三件套→要中式刺绣」reuse 轮 picks=8、
      命中 0，径直推进 CONCLUDING 后模型想补搜已被阶段哨兵拦死，只能收尾承认失败）。
      None（本轮没传 must_have）不触发——没有新硬条件就没有「质」可言。
    触发即退回 SEARCHING 用新条件补搜一次，把新老候选合流。

    另有两条**不限 reuse 的分支**（判据函数与 transition_notice 共用，两边永远同步）：
    - 污染分支（:func:`_pollution_backfill_due`）：检索词被场景词稀释、品类一致性门沉底
      大半后池内相符候选不够——指路「用聚焦品类词重搜」。
    - 硬淘汰杀池分支（:func:`_hard_cull_backfill_due`）：预算/排除词把按相关性召回的
      top-k 杀到不够出清单——指路「带 price_usd_max / 换开检索词补搜」。

    把 retrieval 改写成 ``augment``：既表达「已经在补搜了」的真实语义，也让本闸只触发一次——
    否则补搜回来若仍不足 3 件，会无限回退重搜。

    **priority=39，必须先于 phase_transition(40)**：40 见 picks>0 就把阶段推进 CONCLUDING，
    本钩子的 ``phase == COMPARING`` 前置条件随即失效——薄复用（1~2 件）的补搜腿在集成链路里
    就死了（单测直调钩子暴露不了）。先判补搜、后判转移：补搜火了阶段退回 SEARCHING，40 的
    COMPARING 分支自然不再触发。
    """
    if current_fork_depth() >= 1:
        return None

    machine = get_phase_machine()
    if machine is None or machine.phase != Phase.COMPARING:
        return None
    if not context.get("picker_attempted"):
        return None

    picks = context.get("picks_count", 0)
    must_hits = context.get("must_have_hits")
    mode = get_retrieval_mode()
    reuse_thin = mode == "reuse" and _reuse_backfill_due(picks, must_hits)
    # 污染分支不限 reuse：首搜轮检索词被场景词稀释、品类门沉底大半后池子吃空，同样要补搜
    # （见 _pollution_backfill_due——手表 badcase 就是全新会话的第一搜）。
    polluted = _pollution_backfill_due(
        mode, context.get("oncat_count"), context.get("offcat_count")
    )
    # 硬淘汰杀池分支同样不限 reuse：首搜按相关性召回的 top-k 被预算/排除词杀空，带
    # price_usd_max / 换开检索词补搜捞得回预算内的货（见 _hard_cull_backfill_due）。
    hard_culled = _hard_cull_backfill_due(
        mode, picks, context.get("excluded_count"), context.get("over_budget_count")
    )
    if not reuse_thin and not polluted and not hard_culled:
        return None

    # mode=augment 是本闸的触发闩（只补搜一次），先于回退写。回退本身连同状态回收
    # （同轮闭锁 / 进展计数清零 / 收线通告重武装 / 直搜解锁）全在 regress 事务里——
    # 曾经散在这里手抄、40 号钩子同轮吞回退，见 PhaseStateMachine.regress 的 docstring。
    set_retrieval_mode("augment")
    machine.regress(Phase.SEARCHING, reason="refine_backfill", context=context)
    # picks_close 通告重武装是本闸专属（薄复用那次被刻意压下了，见 transition_notice）；
    # search_close 已由 regress 统一处理。
    guard = context.get("_guard")
    if isinstance(guard, GuardState):
        guard.notified_transitions.discard("picks_close")
    if polluted:
        logger.info(
            "候选池被跨品类污染（品类相符 %s 件 / 沉底 %s 件），退回检索补搜一次",
            context.get("oncat_count"),
            context.get("offcat_count"),
        )
    elif hard_culled:
        logger.info(
            "候选池被硬淘汰杀空（剩 %d 件；排除词杀 %s 件 / 超预算杀 %s 件），退回检索补搜一次",
            picks,
            context.get("excluded_count"),
            context.get("over_budget_count"),
        )
    elif picks < _REFINE_MIN_PICKS:
        logger.info("复用轮精挑仅 %d 件（<%d），退回检索补搜一次", picks, _REFINE_MIN_PICKS)
    else:
        logger.info("复用轮 must_have 池内 0 命中（picks=%d），退回检索补搜一次", picks)
    # 不再走 inject_messages 发「请重新检索」：那条消息要到**再下一轮**才被消费，而模型在
    # transition_notice 缀在 picker 结果上的指路（零时差）驱动下，多半这一轮已经在补搜了——
    # 迟到的重复指令只会诱导它搜第二遍。本钩子只负责状态：退阶段、改 mode、重新武装通告。
    return context


@harness_hook("post_reflect", name="phase_rollback", priority=41)
async def check_phase_rollback(context: dict[str, Any]) -> dict[str, Any] | None:
    """COMPARING 里 item_picker 精挑不出东西时回退到 SEARCHING：扩大搜索范围。

    触发条件严格按 refdocs 17-4 §4.3——**ItemPicker 返回空** + 连续 2 轮无进展。
    「item_picker 还没被调过」不算无进展：模型在 COMPARING 里先 price_compare / shipping_calc /
    向用户澄清都是正常路径，那时回退纯属误伤（实测会打断正常链路，把阶段推回 SEARCHING）。
    真正在 COMPARING 里卡死不动的情形由 TGM 的迭代上限兜底。
    """
    if current_fork_depth() >= 1:
        return None

    machine = get_phase_machine()
    if machine is None or machine.phase != Phase.COMPARING:
        return None

    picks = context.get("picks_count", 0)
    if picks > 0:
        machine.reset_no_progress()
        return None

    if not context.get("picker_attempted"):
        return None  # 还没精挑过，谈不上「精挑不出东西」

    rounds = machine.record_no_progress()
    if rounds >= _ROLLBACK_THRESHOLD:
        # 状态回收（同轮闭锁 / 进展计数清零 / 收线通告重武装 / 直搜解锁）全在 regress 事务里。
        # 旧散装版还漏了进展计数清零：本请求已搜到的候选数下一轮仍算「进展」，回退刚落地就被
        # 40 号钩子凭旧计数推回 COMPARING——与 refine_backfill 曾踩的是同一族坑，事务一并治掉。
        machine.regress(Phase.SEARCHING, reason="phase_rollback", context=context)
        context.setdefault("inject_messages", []).append(
            {
                "role": "system",
                "content": (
                    "当前候选集无法满足用户需求。已回退到搜索阶段。"
                    "请尝试调整搜索条件（放宽预算/换品类/减少约束）。"
                ),
            }
        )
    return context


# ── 阶段收线通告：缀在**触发转移的工具结果**尾部（post_tool_call）─────────────────────
#
# 为什么不在 try_phase_transition（post_reflect）里注入：转移信号（候选入池 / 精挑完成）产生于
# **工具执行**，而 post_reflect 跑在「下一次模型调用」之后——等通告经 inject 通道被消费，模型
# 已经又解码了一轮、下一步早定了。perf-audit-r3 实测：通告晚一轮到场，模型照样连发 item_search
# 撞哨兵，白耗两轮。缀在工具结果尾部则是模型下一次解码的必读内容，零时差。

_SEARCH_NOTICE_TOOLS = frozenset({"item_search", "dispatch_tool", "parallel_dispatch_tool"})


def _price_tasks_hint() -> str:
    """planner 判定本轮无比价 / 到手价诉求 → 提示跳过 price_compare / shipping_calc。

    单平台推荐轮里这两步近乎空转（候选价格已在检索结果里），各省一轮解码。只打动机不打机制：
    工具仍可用。planner 判空（判不出 / 未跑）时不提示——宁可多调，不误伤真比价诉求。
    """
    tasks = get_session_tasks()
    if tasks and "price_compare" not in tasks and "landed_cost" not in tasks:
        return (
            "另外，本轮用户没有比价 / 算到手价的诉求（planner 判定），候选价格已在检索结果中，"
            "无需 price_compare / shipping_calc。"
        )
    return ""


@harness_hook("post_tool_call", name="transition_notice", priority=19)
async def append_transition_notice(context: dict[str, Any]) -> dict[str, Any] | None:
    """把「阶段收线」通告当场缀在触发它的工具结果尾部。仅主 loop（depth 0）。

    priority=19：在截断（10）与回放缓存记录（15）之后——通告不进回放缓存（回放那次自带
    「换参数或进下一步」的提示，不需要旧通告）；在分级提示（20）之前，与 nudge 各说各的。

    三条边、每 loop 各一次（回退 / 补搜会重新武装 search_close / picks_close）：
    - planner 判 reuse → 「检索已跳过，直接精挑」（阶段本体的直跳在 post_reflect）
    - 检索类工具首次带回非空候选 → 「检索收线，别再搜」（+ 无比价诉求时的跳过提示）
    - item_picker 精挑非空 → 「直接 shopping_summary 收尾」；复用轮不够用（<3 件，或
      must_have 池内 0 命中）时改发「请重新检索」——refine_backfill 马上要把阶段退回
      SEARCHING，让模型提前拿到指路。
    """
    if current_fork_depth() >= 1:
        return None
    machine = get_phase_machine()
    guard = context.get("_guard")
    if machine is None or not isinstance(guard, GuardState):
        return None
    result = context.get("tool_result")
    if not isinstance(result, str) or not result:
        return None

    tool = context.get("tool_name", "")
    notice = ""
    if (
        tool == "planner"
        and machine.phase is Phase.PLANNING
        and get_retrieval_mode() == "reuse"
        and "reuse_skip" not in guard.notified_transitions
    ):
        guard.notified_transitions.add("reuse_skip")
        notice = (
            "\n\n[阶段推进] 本轮复用上一轮候选、默认不重新检索：直接用 item_picker 按本轮"
            "条件精挑。仅当你确认旧候选完全不适用（如用户换了品类）时才 item_search 补搜"
            f"——本轮补搜预算只有 {REUSE_RETRIEVAL_BUDGET} 次，超出会被机制拒绝。"
        ) + _price_tasks_hint()
    elif (
        tool in _SEARCH_NOTICE_TOOLS
        and machine.phase is Phase.SEARCHING
        and context.get("call_candidates", 0) > 0
        and "search_close" not in guard.notified_transitions
    ):
        guard.notified_transitions.add("search_close")
        # web_search 一并点名：它的闸在 websearch_gate（有候选即拦），但 perf-audit-r6 实测
        # 通告只点 item_search 时，模型转头连发 4 个 web_search「求证」，白耗一轮撞闸。
        notice = (
            "\n\n[阶段推进] 候选已入池，检索阶段就此收线：不要再调用 item_search / "
            "dispatch_tool / web_search——继续检索只会消耗全树检索预算并很快被机制拒绝。"
            "请基于已入池候选继续（price_compare / shipping_calc / item_picker → "
            "shopping_summary）。"
        ) + _price_tasks_hint()
    elif tool == "item_picker" and machine.phase is Phase.COMPARING:
        picks = context.get("call_picks", 0)
        oncat = context.get("call_oncat")
        offcat = context.get("call_offcat")
        if _pollution_backfill_due(get_retrieval_mode(), oncat, offcat):
            # 污染补搜在即（refine_backfill 将在 post_reflect 退回 SEARCHING）：指路必须点明
            # 「换聚焦品类词」——照原样重搜同一句被场景词稀释的 query，拿回的还是同一池西装皮鞋。
            # 判据与 refine_backfill 共用 _pollution_backfill_due，两边永远同步。
            notice = (
                f"\n\n[阶段回退] 候选池被跨品类结果稀释：{(oncat or 0) + (offcat or 0)} 件里"
                f"只有 {oncat} 件与目标品类相符，不够出清单——检索词里的场景/人群词"
                "（formal / business / men 这类）会把其他正装品类一并召回。请改用**聚焦的"
                "品类核心词**（如 men's wristwatch）item_search 重搜一次（机制已放行），"
                "场景词改放 item_picker 的 prefer_keywords；新老候选合流后再精挑。"
            )
        elif _hard_cull_backfill_due(
            get_retrieval_mode(),
            picks,
            context.get("call_excluded"),
            context.get("call_over_budget"),
        ):
            # 硬淘汰杀池在即（refine_backfill 将退回 SEARCHING）：指路必须点明「换条件搜」——
            # 照原样重搜拿回的还是同一批超预算/踩排除词的货。判据与闸共用，两边永远同步。
            # 排在 picks<=0 之前：0 件恰恰是被杀得最狠的形态，更需要指路而不是沉默。
            over_n = context.get("call_over_budget") or 0
            excl_n = context.get("call_excluded") or 0
            fix = (
                "带 price_usd_max=预算 重新 item_search（召回期就过滤价格）"
                if over_n >= excl_n
                else "改用避开排除材质/属性的检索词重新 item_search"
            )
            notice = (
                f"\n\n[阶段回退] 硬约束把候选池筛得只剩 {picks} 件（超预算 {over_n} 件、"
                f"命中排除词 {excl_n} 件），不够出清单——召回是按相关性排的 top-k，预算内/"
                f"合规的商品可能排在名次外没进池。请{fix}补搜一次（机制已放行），"
                "新老候选合流后再精挑。"
            )
        elif picks <= 0:
            return None
        elif get_retrieval_mode() == "reuse" and _reuse_backfill_due(
            picks, context.get("call_must_hits")
        ):
            # 补搜在即（refine_backfill 将在 post_reflect 把阶段退回 SEARCHING）：
            # 指路「重新检索」而不是「收尾」。触发口径与 refine_backfill 一致（量不够 or
            # must_have 池内 0 命中），两边不同步的话模型会拿着「去收尾」的指路撞上已回退的
            # 阶段闸。不记 notified——补搜回来 mode 已是 augment，不会再进本分支；精挑再
            # 完成时正常发 picks_close。
            why = (
                f"精挑后只剩 {picks} 件，不够出清单"
                if picks < _REFINE_MIN_PICKS
                else f"这 {picks} 件没有一件命中本轮的硬条件（must_have）"
            )
            notice = (
                f"\n\n[阶段回退] 复用旧候选{why}——旧候选"
                "未必含符合本轮新条件的商品。请用**本轮的新条件** item_search 重新检索"
                "一次（机制已放行），新老候选合流后再精挑。"
            )
        elif "picks_close" not in guard.notified_transitions:
            guard.notified_transitions.add("picks_close")
            notice = (
                f"\n\n[阶段推进] 精挑已完成（{picks} 件），比价阶段就此结束：无需再调用 "
                "price_compare / shipping_calc / item_picker，价格与运费信息已在候选数据中。"
                "请直接调 shopping_summary 给出最终清单。"
            )

    if not notice:
        return None
    context["tool_result"] = result + notice
    return context
