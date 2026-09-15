"""检索进度机：PLANNING → SEARCHING → COMPARING → CONCLUDING。**它是进度机，不是权限机**——
写工具的保护在权限引擎 / 确认卡 / 幂等键 / 顺序断言四道防线（见 ``app/agent/permissions.py``），
这里唯一的硬拒是 shopping_summary 的收尾资格。

    pre_tool_call    20  phase_check         收尾资格底线：无候选 / 本轮没精挑 → 不许出清单
    post_tool_call   19  transition_notice   收线通告缀在触发转移的工具结果尾部
                                             （post_reflect 注入晚一轮）
    post_reflect     40  phase_step          三步固定顺序：补搜判定（薄复用 / 污染 / 硬淘汰杀池
                                             → 退回 SEARCHING）→ 按候选 / picks 推进 → COMPARING
                                             连续 2 轮无进展回退（曾是 39/40/41 三个 hook）

阶段机复位到 PLANNING 不在这里：``orchestrator.run_agent`` 开局与其它会话级 ContextVar 一起
``fresh_phase_machine()``（曾是 on_session_start 的 phase_init hook）。

转移信号由适配器从可靠数据源（工具名 + 候选登记表，见 ``signals.py``）填入 context，不 grep 文本。
阶段推进会重置漂移的「连续」类计数器（``_reset_drift_counters``），``blacklist_violations`` 不重置。
"""

from __future__ import annotations

import logging
from typing import Any

from app.api.context import get_session_tasks
from app.harness.autopick import autopick_applies
from app.harness.middleware import HookRejectSignal, harness_hook
from app.harness.phase_machine import Phase, get_phase_machine
from app.harness.retrieval_budget import (
    budget_relax_due,
)
from app.harness.signals import candidate_count
from app.harness.state import GuardState

logger = logging.getLogger("shoppingx.harness.progress")


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


async def try_phase_transition(context: dict[str, Any]) -> dict[str, Any] | None:
    """根据当前执行状态判断是否触发阶段转移。仅 depth 0 生效。

    在 drift_detector（20）之后：本轮漂移判定基于「转移前」的计数器，判完再重置。
    """

    machine = get_phase_machine()
    if machine is None:
        return None

    current = machine.phase
    moved = False

    if current == Phase.PLANNING:
        if context.get("planner_output_ready"):
            moved = machine.try_transition("planner_output_ready")

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

# 补搜闩：一轮只自动补搜一次，否则补搜回来若仍不足 3 件会无限回退重搜。记在
# ``guard.notified_transitions``（regress 事务不清它），与「已发过的通告」同一本账。
BACKFILL_LATCH = "backfill_done"


def _backfilled(guard: GuardState | None) -> bool:
    return guard is not None and BACKFILL_LATCH in guard.notified_transitions


def _pollution_backfill_due(backfilled: bool, oncat: int | None, offcat: int | None) -> bool:
    """污染分支（**首搜轮也生效**）：品类门把池子大半判为跨品类混入、剩下相符的不够出清单。

    手表 badcase（thread 6718ed65）：检索词 formal dress watch men business 里的通用正装词把
    西装/皮鞋/马甲一并召回，10 条里只有 2 条真手表——品类一致性门正确沉底了 8 条，但没有任何
    机制补货，径直收尾出了 2 件的清单。「门杀得对」和「池子该补」是两件事，这里管后者。

    三个条件缺一不可：
    - ``not backfilled``：已补搜过一次就不再回退（闩由补搜闸自己写，天然只触发一次）；
    - ``oncat < 3``：池内品类相符的候选不够出清单（picks 件数看不出——沉底垃圾也占 picks 名额）；
    - ``offcat > oncat``：污染占主导。**刻意不给「池子小但干净」触发**（oncat=2、offcat=0 是库存
      稀疏，不是检索词的错，重搜同样的词只会拿回同样的池子）。
    """
    return (
        not backfilled and oncat is not None and oncat < _REFINE_MIN_PICKS and (offcat or 0) > oncat
    )


def _hard_cull_backfill_due(
    backfilled: bool, picks: int, excluded: int | None, over_budget: int | None
) -> bool:
    """硬淘汰杀池分支（**首搜轮也生效**）：预算 / 排除词把池子淘汰到不够出清单。

    与污染分支同构的三条件：已补搜过不再回退；``picks < 3`` 不够出清单；
    ``淘汰数 > picks`` 即硬约束才是池子空的主因——**刻意不给「池子小但干净」触发**
    （picks=2、淘汰 0 是库存稀疏，重搜同样的词只会拿回同一批）。硬淘汰杀池则不同：
    池子是按相关性召回的 top-k，不是按「预算内的相关性」——库里完全可能有预算内 /
    不含排除材质、却排在 k 名开外的商品，带 price_usd_max / 换开检索词补搜捞得回来
    （指路话术见 transition_notice，判据两边共用本函数）。
    诊断缺席（None）不触发——失效方向中性，与侧信道的降级语义一致。
    """
    culled = (excluded or 0) + (over_budget or 0)
    if budget_relax_due():
        # 补搜已被证伪：item_search 的探测召回显示「不带预算过滤也只捞得到超预算的货」，
        # 再带 price_usd_max 搜一次必然还是空——那一轮解码纯属白烧，还会把模型往「再换个词
        # 试试」的死循环上推。此时正确的动作是**如实告知 + 问用户要不要放宽**，由
        # transition_notice 的放宽分支指路（判据同源，见 _budget_relax_notice_due）。
        return False
    return (
        not backfilled
        and (excluded is not None or over_budget is not None)
        and picks < _REFINE_MIN_PICKS
        and culled > picks
    )


def _budget_relax_notice_due(
    backfilled: bool, picks: int, excluded: int | None, over_budget: int | None
) -> bool:
    """该给「建议用户放宽预算」的指路了吗？＝硬淘汰杀池的形态 + 探测证明预算内确实没货。

    条件与 :func:`_hard_cull_backfill_due` 的形态判据同源（故意重复那三条，而不是靠调用它——
    它已经被 ``budget_relax_due`` 提前否掉了），只是结论相反：一个说「换条件补搜」，
    一个说「补搜没用，去问用户」。
    """
    culled = (excluded or 0) + (over_budget or 0)
    return (
        budget_relax_due()
        and not backfilled
        and (excluded is not None or over_budget is not None)
        and picks < _REFINE_MIN_PICKS
        and culled > picks
    )


async def check_refine_backfill(context: dict[str, Any]) -> dict[str, Any] | None:
    """精挑后候选池被判「该补」→ 退回 SEARCHING 补搜一次（每轮最多一次）。

    两条分支（判据函数与 transition_notice 共用，两边永远同步）：
    - 污染分支（:func:`_pollution_backfill_due`）：检索词被场景词稀释、品类一致性门沉底
      大半后池内相符候选不够——指路「用聚焦品类词重搜」。
    - 硬淘汰杀池分支（:func:`_hard_cull_backfill_due`）：预算/排除词把按相关性召回的
      top-k 杀到不够出清单——指路「带 price_usd_max / 换开检索词补搜」。

    触发即写补搜闩 :data:`BACKFILL_LATCH`：既表达「已经在补搜了」的真实语义，也让本闸只触发
    一次——否则补搜回来若仍不足 3 件，会无限回退重搜。

    **必须先于 try_phase_transition**（phase_step 内的固定顺序）：转移见 picks>0 就把阶段推进
    CONCLUDING，本函数的 ``phase == COMPARING`` 前置条件随即失效。先判补搜、后判转移：补搜火了
    阶段退回 SEARCHING，转移的 COMPARING 分支自然不再触发。
    """
    machine = get_phase_machine()
    if machine is None or machine.phase != Phase.COMPARING:
        return None
    if not context.get("picker_attempted"):
        return None

    picks = context.get("picks_count", 0)
    guard = context.get("_guard")
    guard = guard if isinstance(guard, GuardState) else None
    backfilled = _backfilled(guard)
    # 污染分支：检索词被场景词稀释、品类门沉底大半后池子吃空（见 _pollution_backfill_due——
    # 手表 badcase 就是全新会话的第一搜）。
    polluted = _pollution_backfill_due(
        backfilled, context.get("oncat_count"), context.get("offcat_count")
    )
    # 硬淘汰杀池分支：按相关性召回的 top-k 被预算/排除词杀空，带 price_usd_max / 换开检索词
    # 补搜捞得回预算内的货（见 _hard_cull_backfill_due）。
    hard_culled = _hard_cull_backfill_due(
        backfilled, picks, context.get("excluded_count"), context.get("over_budget_count")
    )
    if not polluted and not hard_culled:
        return None

    # 补搜闩先于回退写。回退本身连同状态回收（同轮闭锁 / 进展计数清零 / 收线通告重武装 /
    # 直搜解锁）全在 regress 事务里——曾经散在这里手抄、转移步同轮吞回退，见
    # PhaseStateMachine.regress 的 docstring。
    if guard is not None:
        guard.notified_transitions.add(BACKFILL_LATCH)
    machine.regress(Phase.SEARCHING, reason="refine_backfill", context=context)
    # picks_close 通告重武装是本闸专属；search_close 已由 regress 统一处理。
    if guard is not None:
        guard.notified_transitions.discard("picks_close")
    if polluted:
        logger.info(
            "候选池被跨品类污染（品类相符 %s 件 / 沉底 %s 件），退回检索补搜一次",
            context.get("oncat_count"),
            context.get("offcat_count"),
        )
    else:
        logger.info(
            "候选池被硬淘汰杀空（剩 %d 件；排除词杀 %s 件 / 超预算杀 %s 件），退回检索补搜一次",
            picks,
            context.get("excluded_count"),
            context.get("over_budget_count"),
        )
    # 不再走 inject_messages 发「请重新检索」：那条消息要到**再下一轮**才被消费，而模型在
    # transition_notice 缀在 picker 结果上的指路（零时差）驱动下，多半这一轮已经在补搜了——
    # 迟到的重复指令只会诱导它搜第二遍。本钩子只负责状态：退阶段、写闩、重新武装通告。
    return context


async def check_phase_rollback(context: dict[str, Any]) -> dict[str, Any] | None:
    """COMPARING 里 item_picker 精挑不出东西时回退到 SEARCHING：扩大搜索范围。

    触发条件严格按 refdocs 17-4 §4.3——**ItemPicker 返回空** + 连续 2 轮无进展。
    「item_picker 还没被调过」不算无进展：模型在 COMPARING 里先 price_compare / shipping_calc /
    向用户澄清都是正常路径，那时回退纯属误伤（实测会打断正常链路，把阶段推回 SEARCHING）。
    真正在 COMPARING 里卡死不动的情形由 TGM 的迭代上限兜底。
    """

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


@harness_hook("post_reflect", name="phase_step", priority=40, main_only=True)
async def step_phase(context: dict[str, Any]) -> dict[str, Any] | None:
    """一轮 post_reflect 的阶段机三步，顺序固定：补搜判定 → 推进 → 无进展回退。"""
    changed = False
    for step in (check_refine_backfill, try_phase_transition, check_phase_rollback):
        if await step(context) is not None:
            changed = True
    return context if changed else None


# ── 阶段收线通告：缀在**触发转移的工具结果**尾部（post_tool_call）─────────────────────
#
# 为什么不在 try_phase_transition（post_reflect）里注入：转移信号（候选入池 / 精挑完成）产生于
# **工具执行**，而 post_reflect 跑在「下一次模型调用」之后——等通告经 inject 通道被消费，模型
# 已经又解码了一轮、下一步早定了。perf-audit-r3 实测：通告晚一轮到场，模型照样连发 item_search
# 撞哨兵，白耗两轮。缀在工具结果尾部则是模型下一次解码的必读内容，零时差。

# 只有直搜：本条件还要求 ``call_candidates > 0``，而派发路径数不出新候选
# （见 signals._SEARCH_TOOLS），加上 task_dispatch 也恒为假。
_SEARCH_NOTICE_TOOLS = frozenset({"item_search"})


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


@harness_hook("post_tool_call", name="transition_notice", priority=19, main_only=True)
async def append_transition_notice(context: dict[str, Any]) -> dict[str, Any] | None:
    """把「阶段收线」通告当场缀在触发它的工具结果尾部。仅主 loop（depth 0）。

    priority=19：在截断（10）与回放缓存记录（15）之后——通告不进回放缓存（回放那次自带
    「换参数或进下一步」的提示，不需要旧通告）；在分级提示（20）之前，与 nudge 各说各的。

    两条边、每 loop 各一次（回退 / 补搜会重新武装 search_close / picks_close）：
    - 检索类工具首次带回非空候选 → 「检索收线，别再搜」（+ 无比价诉求时的跳过提示）
    - item_picker 精挑非空 → 「直接 shopping_summary 收尾」；池子被污染 / 被硬淘汰杀空时
      改发「请重新检索」——refine_backfill 马上要把阶段退回 SEARCHING，让模型提前拿到指路。
    """
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
            "task_dispatch / web_search——继续检索只会消耗全树检索预算并很快被机制拒绝。"
        )
        # round3 刀 2：普通轮由系统在下一次思考前自动比价 + 精挑（harness.autopick），指路
        # 直接改成「等结果、然后收尾」；套装轮 / 关开关时仍指模型自己走 price_compare → picker。
        if autopick_applies():
            notice += (
                "系统将自动完成比价（到手价）与精挑并把结果给你，届时直接 shopping_summary 收尾。"
            )
        else:
            notice += (
                "请基于已入池候选继续（price_compare / shipping_calc / item_picker → "
                "shopping_summary）。"
            ) + _price_tasks_hint()
    elif tool == "item_picker" and machine.phase is Phase.COMPARING:
        picks = context.get("call_picks", 0)
        oncat = context.get("call_oncat")
        offcat = context.get("call_offcat")
        backfilled = _backfilled(guard)
        if _pollution_backfill_due(backfilled, oncat, offcat):
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
        elif _budget_relax_notice_due(
            backfilled,
            picks,
            context.get("call_excluded"),
            context.get("call_over_budget"),
        ):
            # 补搜无解（探测已证明预算内没货）：指路「问用户」而不是「再搜一次」。**必须排在
            # 硬淘汰分支之前**——两者形态判据相同，顺序反了这条永远轮不到。
            over_n = context.get("call_over_budget") or 0
            notice = (
                f"\n\n[阶段回退] 预算把候选池筛得只剩 {picks} 件（超预算 {over_n} 件），"
                "而系统的探测召回显示：**不带预算过滤也只捞得到超预算的货**——库里这个品类"
                "在用户预算内确实没有，再补搜一次拿回的还是同一批。请直接如实告诉用户"
                "「符合的商品都在预算之外，最低约 $X」，并用 ask_user 问是否放宽预算或换方向；"
                "不要自作主张放宽预算，也不要再重复检索。"
            )
        elif _hard_cull_backfill_due(
            backfilled,
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


@harness_hook("pre_tool_call", name="phase_check", priority=20, main_only=True)
async def check_phase_permission(context: dict[str, Any]) -> dict[str, Any] | None:
    """shopping_summary 收尾资格底线。仅 depth 0 生效，其余工具一律放行。"""

    machine = get_phase_machine()
    if machine is None:
        return None

    if context.get("tool_name", "") != "shopping_summary":
        return None

    if not candidate_count():
        raise HookRejectSignal(
            "当前还没有任何候选商品，无法生成购物清单。"
            "请先检索到候选再调 shopping_summary；若本轮并非购物意图，请改调 chat_fallback。"
        )
    if machine.phase is Phase.PLANNING:
        raise HookRejectSignal(
            "还没有为本轮做过精挑，不能直接出清单。手上的候选是上一轮按上一轮条件搜的，"
            "请先调 planner 判断本轮意图，再用 item_picker 按本轮条件精挑，然后收尾。"
        )
    if "item_picker" not in context.get("called_tools", set()):
        raise HookRejectSignal(
            "本轮还没精挑过，不能直接出清单——收尾清单只认本轮 item_picker 的定稿，"
            "跳过精挑会产出与候选自相矛盾的空清单。请先调 item_picker（对已入池候选"
            "就地打分过滤，开销很小），再调 shopping_summary 收尾。"
        )
    return None
