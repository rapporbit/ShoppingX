"""一次会话共享的「商品检索」预算 + 召回信号（按 session_dir 聚合）。

为什么把预算打在「检索总量」而不是某个具体机制上：堵住任何单一渠道，「再找找更好的」这个**动机**
都不会消失，压力只会顶到还开着的那个口（挤气球）。所以 ``item_search`` / ``web_search`` 计进
**同一个计数器**，过阈值由 middleware 注入强制收尾信号。

为什么按 session_dir 聚合（住 :mod:`app.api.run_state` 的 run 状态表）而不是裸 ContextVar：
asyncio 子任务创建时会**拷贝**一份 context，子任务里对 ContextVar 的 ``set`` 不回传父 loop——
同轮 batch 的几个工具各跑在自己的子任务里，用 ContextVar 就会静默漏计。``session_dir`` 由
``thread_scope`` 设好后被子任务继承，按同一 key 自增才数得准。（历史：这套聚合最初是为跨 fork
树共享写的，2026-09-16 删子 Agent 后口径收窄为「一次 run_agent」，机制不变。）

状态表需要收尾清理（防无界增长）：``run_agent`` 结束时调 :func:`reset_run`。

三本账各管各的，互不透支：一次 run 的检索总量（``count``，堵找更好商品的动机）、web_search 任务配额
（``WEB_SEARCH_TASK_QUOTA``）、research 搜索配额（``RESEARCH_SEARCH_QUOTA``，见下方长注释）。
"""

from __future__ import annotations

from dataclasses import dataclass

from app.api.context import get_session_tasks
from app.api.run_state import clear_run_slot, peek_run_slot, run_slot
from app.utils.env import env_int

# 任务口径的 web_search 小配额（窄口径用途门）：planner 判 evaluate / category_intel 时，
# prompt 明确要求「+ web_search 口碑」，但「有候选就拦」的位置门会罚这条路——只能靠连拒 2 次
# 的逃生门走通，每次白花 2 轮往返。判据用 planner 落进 session 的 tasks（确定性信号，不是模型
# 自由意志），配额防挤气球：recommend 主链路照旧拦死，延迟零回退。
WEB_SEARCH_TASK_QUOTA = env_int("WEB_SEARCH_TASK_QUOTA", 2)
_TASKS_WANT_WEB = frozenset({"evaluate", "category_intel"})

# ``research``（C2 的有界研究函数）的**独立**会话配额，单位是**搜索条数**不是调用次数：
# 单次上界 ``RESEARCH_MAX_TARGETS``（3，每 target 一条模板查询、aspects 合进同一条）已在工具侧
# 截断，这里管的是会话累计 —— 6 条 ≈ 两次满载调用。
#
# **为什么与 WEB_SEARCH_TASK_QUOTA 分账（两个计数器互不透支）**：research 内部直调 ``search_web``，
# 绕开 web_search 的工具层门；它单次就发 3 搜，是那道门（配额 2）的 1.5 倍，共用一份额度等于两边
# 互相饿死。两者吃的也不是同一种成本：web_search 每条整页正文原样进主环 messages，research 的正文
# 只进归纳模型、主环只见 schema。
#
# **为什么不塞进本次 run 的检索总额（RETRIEVAL_TOOLS / RETRIEVAL_BUDGET_CAP）**：那份额度堵的是
# 「再找找更好的商品」这个动机，research 不产候选、不是这条路上的渠道，混进去只会挤掉 item_search
# 的额度，且「停止检索立即收尾」的软收敛哨兵对它并不成立。挤气球风险（item_search 撞线后改调
# research 兜圈子）由本配额自己封顶兜住：最多 2 次调用，且拿不到可下单候选。
#
# **为什么是 6 而不是博客口径的 3**：博客那个 3 说的是自由 web_search —— 正文全进主环上下文。
# research 是有界函数，主环单次增量约为裸搜的 1/10，同样的上下文预算能放更多次。
RESEARCH_SEARCH_QUOTA = env_int("RESEARCH_SEARCH_QUOTA", 6)


@dataclass
class _RunRetrieval:
    count: int = 0  # item_search + web_search 本次 run 累计（预算计数）
    item_search_runs: int = 0  # item_search 调用次数（含召回为空的）
    web_search_runs: int = 0  # web_search 已执行次数（任务口径配额用，会话共享）
    research_searches: int = 0  # research 已发出的搜索条数（独立配额，与 web_search 不互通）
    nonempty_item_search: int = 0  # 召回到 ≥1 候选的 item_search 次数（web_search 兜底门用）
    # ── item_search 探测召回（filtered_out）的会话汇总，供「该建议放宽预算还是该补搜」判定 ──
    probe_runs: int = 0  # 跑过探测的 item_search 次数（＝带硬过滤且命中不足的那些）
    probe_price_blocked: int = 0  # 探测差集里「只差预算」的条数
    probe_other_blocked: int = 0  # 探测差集里因排除词 / 品牌 / 评分被挡的条数
    probe_hits: int = 0  # 上述那些 item_search 各自的实际命中数之和


def _state(create: bool = True) -> _RunRetrieval | None:
    """取当前 session 的检索状态；无 session 作用域（单测）返回 None。

    ``create=False`` 时「本轮还没建过这一格」也返回 None——几处门控靠这个区分「还没进入购物
    检索流程」与「搜过但计数是 0」。
    """
    return run_slot(_RunRetrieval) if create else peek_run_slot(_RunRetrieval)


def charge_retrieval_count() -> int | None:
    """item_search / web_search 计一次，返回本次 run 的累计；无 session 作用域返回 None。"""
    st = _state()
    if st is None:
        return None
    st.count += 1
    return st.count



def note_web_search() -> None:
    """web_search 执行时计一次（配额消耗）。挂在 retrieval_charge(45)——门控 websearch_gate(15)
    读的是自增前值（＝之前已完成次数），「已完成 < 配额」即放行，与 item_search_calls 同一套
    顺序契约。无 session 作用域（单测）不计。"""
    st = _state()
    if st is not None:
        st.web_search_runs += 1


def research_remaining() -> int:
    """本会话 ``research`` 还剩几条搜索额度（无 session 作用域＝单测，回满额）。"""
    st = _state(create=False)
    used = st.research_searches if st is not None else 0
    return max(0, RESEARCH_SEARCH_QUOTA - used)


def charge_research(planned: int) -> bool:
    """预扣 ``planned`` 条 research 搜索额度：够则扣掉回 True，不够则**不扣**回 False。

    **判与扣在同一个同步段里**（中间无 await），所以同轮 batch 并发发两次 research 时，第二次
    读到的是第一次扣完后的数——不会两条都看见满额然后各搜 3 条把会话上限撑到 6 以上。

    **预扣（执行前）而不是执行后结算**：与 :func:`note_web_search` 同一顺序契约。代价是全降级
    （没配 TAVILY_API_KEY / 熔断）那次也照扣；这与 web_search 现状一致，且降级本身会带 note 回
    模型，不会诱发重试。
    """
    planned = max(0, planned)
    st = _state()
    if st is None:
        return True  # 无 session 作用域（单测直调）：失效方向中性，不拦
    if st.research_searches + planned > RESEARCH_SEARCH_QUOTA:
        return False
    st.research_searches += planned
    return True


def note_item_search(total_recall: int) -> None:
    """item_search 完成后登记一次召回信号（供 web_search 兜底门判定）。"""
    st = _state()
    if st is None:
        return
    st.item_search_runs += 1
    if total_recall > 0:
        st.nonempty_item_search += 1


def note_filtered_probe(*, hits: int, price_blocked: int, other_blocked: int) -> None:
    """登记一次 item_search 探测召回的结论（见 ``app.tools.item_search`` 的探测段）。

    会话级聚合（同 session_dir）：跨平台是同轮 batch 多条 item_search 各搜一份，「预算内到底
    有没有货」是合流后的结论，不该由某一个平台单独说了算。
    """
    st = _state()
    if st is None:
        return
    st.probe_runs += 1
    st.probe_hits += max(0, hits)
    st.probe_price_blocked += max(0, price_blocked)
    st.probe_other_blocked += max(0, other_blocked)


def budget_relax_due() -> bool:
    """是否已有确凿证据表明「库里有相关货，但在用户预算内一件都没有」。

    判据刻意保守（三条全要）——它要否掉的是补搜闸的「带 price_usd_max 重搜一次」，误判的代价
    是本可捞回的货被放弃：

    1. 探测跑过且**只**被价格挡（``other_blocked == 0``）：混着排除词 / 低评分被挡时，放宽预算
       也未必能买，说「放宽预算就有」是误导；
    2. 带过滤的那几次检索命中数合计为 0：只要捞到过一件预算内的货，就轮不到谈放宽；
    3. 至少有一条被价格挡下（``price_blocked > 0``）——否则没有任何「库里其实有货」的证据。

    无 session 作用域（单测直调）返回 False：失效方向中性，维持既有补搜行为。
    """
    st = _state(create=False)
    if st is None:
        return False
    return st.probe_price_blocked > 0 and st.probe_other_blocked == 0 and st.probe_hits == 0


def web_search_allowed() -> bool:
    """web_search 此刻是否允许调用。三种合法场景：

    1. **独立知识查询**：还没进入购物检索流程（item_search 未跑过），用户可能在问品牌口碑、
       评测、趋势等外部事实，或按 plan 的 intent_grounding=web 做意图翻译——放行。
    2. **任务口径配额**（窄口径用途门）：planner 判了 evaluate / category_intel——prompt 明确
       要求 web_search 口碑佐证的任务，在 ``WEB_SEARCH_TASK_QUOTA`` 内放行，不再罚它走
       连拒 2 次的逃生门。判据是 planner 落 session 的确定性 tasks，不是模型自由意志。
    3. **购物流程内兜底**：item_search 跑过但全部召回为空——放行，作为兜底补线索。

    recommend 主链路已有候选时仍然拦截——web_search 不是「找更好」的渠道。

    点名评价 / 比较具体商品走场景 2（planner 判 evaluate）。原来那套「定点调查按子任务隔离
    信号」已删（2026-09-16，真实会话 0 次使用）。
    """
    st = _state(create=False)
    if st is None:
        return True  # 无 session 作用域（单测），或还没进入购物检索流程 → 允许独立知识查询
    if st.item_search_runs == 0:
        return True  # 同上：session 存在但还没搜过商品
    if _TASKS_WANT_WEB & set(get_session_tasks()) and st.web_search_runs < WEB_SEARCH_TASK_QUOTA:
        return True  # 评价 / 行情任务的口碑配额（配额尽则落回下面的兜底判定）
    return st.nonempty_item_search == 0  # 搜过但全空 → 兜底放行；有候选 → 拦


def reset_run() -> None:
    """清掉本 session 的检索预算条目（任务收尾时调，防 run 状态表无界增长）。"""
    clear_run_slot(_RunRetrieval)
