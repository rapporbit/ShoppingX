"""套装（bundle）机制层 —— 「一套齐」跨品类组合优选的状态与算法。

场景：「新生入学一套，预算 1500」不是单品类清单，而是**跨品类凑一套**：planner 拆出
3~6 个槽位（床品 / 台灯 / 收纳箱…）→ 主环同轮 batch ``item_search(slot=槽名)`` 各槽并发
检索 → 在**总预算**约束下做组合优选（哪槽该花钱、哪槽降级、加起来不超）→ 分组收尾。

本模块只放**机制**（状态 + 纯算法），不放工具入口：
  - 槽位定义登记：planner 判出 ``bundle_slots`` 后写进来（模块 dict 按 session_dir 聚合，
    跨工具可见——同 ``_candidates`` 的既有套路）。**槽表只活一轮**：不落盘，``run_agent``
    收尾清内存；续聊轮由 planner 按用户原话 + P_t 重拆。
  - **槽名即身份**：模型入参、候选盖章、检索记账、分配报告、前端分组讲的都是同一个槽名，
    不另发 id。LLM 措辞漂移只在 :func:`resolve_slot` 这一个入口按名字包含关系归并。
  - 检索侧打标：``item_search(slot=...)`` 经 :func:`register_slot` 解析成规范槽名后盖章；
    没盖上的候选由 picker 用槽 keywords 匹配标题兜底归槽。
  - 组合优选：Multiple-Choice Knapsack——essential 槽必选一件、optional 槽可整槽放弃，
    约束 Σ有效价 ≤ 总预算，目标 max Σ分数。槽 ≤6 × 每槽 top5 → 穷举即可（≤ 数万组合，
    毫秒级），不需要近似算法。**不可行时如实报**：给最省组合 + 超支额，绝不静默超预算。
  - 组合报告：分配表（哪槽花了多少、砍了谁、缺了谁）登记给 shopping_summary 注入文案。

「是不是槽位轮」由机制判（本轮登记的槽 ≥2），不由模型自报——同 planner 的币种确定性回填
一个思路。

**槽位有两种形态**（``SLOT_MODE_*``，planner 判、随槽表一起登记）：``bundle`` 是上面说的
「一套齐」；``parallel`` 是「多品类并列」——用户一次要看几类互不相干的东西（「跑鞋 +
降噪耳机」），各类分头检索、各自给推荐，**不配套、不砍类、预算是每件上限不是总和**。
两者共用槽位登记 / 打标 / 分组精排 / 报告结构，差别只在最后那一步选择规则（组合优选 vs
每类各取 top N）与文案措辞。
"""

from __future__ import annotations

import itertools
import logging
import re
from collections.abc import Iterable
from typing import Any, NamedTuple

from pydantic import BaseModel, Field

from app.api.context import get_session_dir
from app.tools.schemas import ItemCandidate
from app.utils.env import env_int
from app.utils.terms import term_hits

logger = logging.getLogger("shoppingx.bundle")

# 槽位数硬上限：「一套」的粒度是子品类不是 SKU，拆到 6 个以上就是过度拆解（且组合枚举
# 规模按槽数指数涨）。planner 的 validator 与 register_slot 都按它封顶。
MAX_SLOTS = env_int("BUNDLE_MAX_SLOTS", 6)
# 每槽进组合枚举的候选上限：组合规模 = (每槽候选+1)^槽数，5×6 槽 ≈ 4.7 万组合，纯 Python
# 毫秒级。再大收益也小——第 6 名靠分数进组合的概率已经很低。
TOP_PER_SLOT = env_int("BUNDLE_TOP_PER_SLOT", 5)
# 并列模式下每个子需求展示几件。「一套齐」每槽只能要一件（配套），并列需求则是**各给一份
# 推荐**——3 件够用户在每类里做选择，再多会把三类的卡片堆成一屏刷不完。
PARALLEL_PER_SLOT = env_int("PARALLEL_PER_SLOT", 3)

# 槽位的两种形态。**是同一套槽位机制的两种消费方式**，共用登记 / 打标 / 分组 rerank / 报告：
#   bundle   —— 「一套齐」：配套、共享**总预算**、essential 必选 optional 可砍，跨槽做组合
#                优选（MCKP），每槽定稿一件。
#   parallel —— 「多品类并列」：用户一次要调研几类互不相干的东西（「跑鞋 + 降噪耳机」），
#                各类各自给推荐、不配套、不砍类、预算是**每件**上限而非总和。
# 为什么必须分开：MCKP 会为了「凑一套不超总预算」擅自砍掉某一类（optional 槽整槽放弃），
# 这在「一套齐」里是正确行为，在并列需求里就是把用户明确要的一类东西弄丢了。
SLOT_MODE_BUNDLE = "bundle"
SLOT_MODE_PARALLEL = "parallel"


class BundleSlot(BaseModel):
    """套装里的一个槽位（要买的一个子品类）。planner 拆解产出，picker 组合消费。

    ``name`` 就是槽的身份：盖章 / 检索记账 / 拒绝复活 / 报告全按名字走（见模块头）。
    """

    name: str = Field(description="槽位名（中文短名，如「床品」「台灯」）")
    keywords: list[str] = Field(
        default_factory=list, description='该槽的英文检索词（如 ["bedding set", "comforter"]）'
    )
    prefer: list[str] = Field(
        default_factory=list,
        description='槽级软偏好原子词（英文优先，如箱子要 ["spinner wheels"]）',
    )
    essential: bool = Field(
        default=True, description="必备槽（少了这套就不成立）；False=可选槽，预算紧时可整槽放弃"
    )
    evidence: str = Field(
        default="",
        description="用户原话里点名这件的片段（照抄）；是按常识推断补的槽就留空——"
        "系统据此判断「套装组成要不要先跟用户确认」。",
    )


# ── 会话级槽位登记（同 _candidates 的「按 session_dir 聚合的模块级 dict」套路）──────────
# 全部只活一轮：run_agent 收尾 reset_session_bundle() 清掉。
# session_dir -> 槽位定义（planner 写、picker / item_search 读）
_BUNDLE: dict[str, list[BundleSlot]] = {}
# session_dir -> 本轮真正检索过的槽名（item_search 盖章时记）。用来区分「搜了但没货」
# （essential 缺货，要如实报）与「压根没搜」（用户在 ask_user 里删掉的槽，静默不列）。
_SEARCHED: dict[str, set[str]] = {}
# session_dir -> 最近一次组合优选的报告（picker 写、shopping_summary 注入文案时读）。
_REPORT: dict[str, dict[str, Any]] = {}
# session_dir -> 用户在组成确认里明确不要的槽名（reconcile_slots_from_reply 记）。register_slot
# 据此拒绝复活：模型检索时再传这个词不代表用户改了主意。跨轮不保留——下一轮用户怎么说，
# planner 就按原话重拆。
_DECLINED: dict[str, list[str]] = {}
# session_dir -> 槽位形态（SLOT_MODE_*），与槽表同生命周期。
_MODE: dict[str, str] = {}

# 老会话历史里 item_search 返回过「slot: s2」这类槽 id（槽名即身份之前的格式），模型可能照抄
# 回来——这种引用不当成一个叫「s2」的新品类去建槽。
_LEGACY_ID_RE = re.compile(r"s\d+")


def _key() -> str | None:
    sd = get_session_dir()
    return str(sd) if sd is not None else None


def set_session_bundle(slots: Iterable[BundleSlot], mode: str | None = None) -> None:
    """登记本轮的槽位（planner 判出 ``bundle_slots`` ≥2 时调；补槽 / 删槽通路也走这里）。

    名字是身份：同名只留第一个，封顶 MAX_SLOTS。``mode`` 是槽位形态（见 SLOT_MODE_*）：
    ``None`` = 沿用已登记的那个。补槽通路（``register_slot`` / ``reconcile_slots_from_reply``）
    都走这个默认值——它们改的是槽表，不该顺手把形态重置回 bundle，那会让并列轮在用户确认
    组成后突然变成「一套齐」。
    """
    k = _key()
    if k is None:
        return
    seen: set[str] = set()
    cleaned: list[BundleSlot] = []
    for s in slots:
        s.name = s.name.strip()
        if s.name and s.name not in seen:
            seen.add(s.name)
            cleaned.append(s)
    if not cleaned:
        return
    _BUNDLE[k] = cleaned[:MAX_SLOTS]
    if mode is not None:
        _MODE[k] = mode if mode in (SLOT_MODE_BUNDLE, SLOT_MODE_PARALLEL) else SLOT_MODE_BUNDLE


def get_session_bundle() -> list[BundleSlot]:
    """读本轮登记的槽位。无会话作用域（单测直调）或本轮没登记 → 空列表 = 不是槽位轮。"""
    k = _key()
    return list(_BUNDLE.get(k, [])) if k is not None else []


def get_session_mode() -> str:
    """本轮的槽位形态。没登记过返回 bundle——那种轮次槽 <2、下游根本不看形态。"""
    k = _key()
    return _MODE.get(k, SLOT_MODE_BUNDLE) if k is not None else SLOT_MODE_BUNDLE


def _match_name(candidate: str, target: str) -> bool:
    """槽名模糊匹配：精确相等，或互为包含（短名 ≥2 字，防「包」这类单字吸走所有槽）。"""
    if candidate == target:
        return True
    return min(len(candidate), len(target)) >= 2 and (candidate in target or target in candidate)


def resolve_slot(ref: str) -> BundleSlot | None:
    """把模型产出的槽名（精确名 / 漂移名）解析成已登记槽；解析不出 → None。

    这是**全链路唯一的名字模糊点**：item_search 盖章经 register_slot 走这里，盖上的是登记表
    里的规范名，下游记账 / 分组 / 报告只做精确比较。先精确、再互含，免得「收纳袋」抢走
    「旅行收纳袋」。
    """
    ref = (ref or "").strip()
    if not ref:
        return None
    slots = get_session_bundle()
    exact = next((s for s in slots if s.name == ref), None)
    return exact or next((s for s in slots if _match_name(s.name, ref)), None)


def register_slot(ref: str) -> str:
    """item_search 的 ``slot`` 入参 → 该盖的规范槽名；空串 = 不盖章。

    已登记（含漂移名）→ 登记表里的名字。没登记时建新槽，两种情形：
      - 槽表非空：用户在确认组成时新增了一类（「再加个生活用品」），补登为 essential；
      - 槽表为空：planner 本轮没拆槽，但模型检索时明确按类传了 slot——这是并列需求（「一套齐」
        轮 planner 必然已拆槽），按 parallel 登记。planner 判得不稳（实测同一条三品类 query 有
        几次没拆），没有这条通路时候选一个章都盖不上，精挑退化成全池单 query 排序、某一类
        屠版（评测 pl02：三类只剩跑鞋）。
    不建：用户在组成确认里删过的槽（含漂移名——「台灯」删了「护眼台灯」也不复活）、老会话
    历史里的 id 形状引用（``s2``）、已到槽数上限。
    """
    ref = (ref or "").strip()
    hit = resolve_slot(ref)
    if hit is not None:
        return hit.name
    k = _key()
    if k is None or not ref or _LEGACY_ID_RE.fullmatch(ref):
        return ""
    if any(_match_name(d, ref) for d in _DECLINED.get(k, [])):
        return ""
    slots = get_session_bundle()
    if len(slots) >= MAX_SLOTS:
        return ""
    if slots:
        set_session_bundle([*slots, BundleSlot(name=ref, evidence="用户确认新增")])
    else:
        new = BundleSlot(name=ref, evidence="检索时按类传入")
        set_session_bundle([new], mode=SLOT_MODE_PARALLEL)
        logger.info("planner 本轮未拆槽，按 item_search(slot=%s) 登记并列槽", ref)
    return ref


def reconcile_slots_from_reply(reply: str, offered: Iterable[str] | None = None) -> list[str]:
    """按 ask_user 的用户回复核销套装组成：被问及但没被选中的槽**删除**（用户明确不要）。

    register_slot 只会「增」，没有这条「删」的通路时，用户在确认组成里去掉的槽会以
    essential 身份留在槽表，收尾被说成「没找到、建议再搜」（线上 badcase 4c0ac682：
    用户答「书包 + 文具 + 水杯 + 生活用品」，summary 却劝他单独再搜笔记本电脑和台灯）。

    确定性触发，不靠模型自觉：仅当回复**点名 ≥2 个已登记槽**（枚举式回答）才核销——
    「水杯换大点的」这类单槽追问不会误伤。``offered``（ask_user 的 options 标签）非空时
    只删标签里出现过的槽——没上问卷的槽不算被问及，不错杀。返回删掉的槽名（日志用）。
    """
    k = _key()
    reply = (reply or "").strip()
    slots = get_session_bundle()
    if k is None or not reply or not slots:
        return []
    mentioned = {s.name for s in slots if s.name in reply}
    if len(mentioned) < 2:
        return []
    offered_text = " ".join(offered) if offered else ""
    removed = [
        s.name
        for s in slots
        if s.name not in mentioned and (not offered_text or s.name in offered_text)
    ]
    if not removed:
        return []
    _DECLINED.setdefault(k, []).extend(removed)
    set_session_bundle([s for s in slots if s.name not in removed])
    logger.info("组成确认核销：删槽 %s，保留 %s", removed, sorted(mentioned))
    return removed


def note_slot_searched(ref: str) -> None:
    """记下「这个槽本轮真的检索过」（item_search 盖章时调）——essential 缺货判定的依据。

    漂移名先归到规范名；解析不出的原样记（报告层对不上号，等价于「没搜过」，失效方向与
    漏记一致）。
    """
    k = _key()
    ref = (ref or "").strip()
    if k is None or not ref:
        return
    s = resolve_slot(ref)
    _SEARCHED.setdefault(k, set()).add(s.name if s is not None else ref)


def searched_slots() -> set[str]:
    """本轮检索过的槽名集合。"""
    k = _key()
    return set(_SEARCHED.get(k, set())) if k is not None else set()


def set_bundle_report(report: dict[str, Any]) -> None:
    """登记最近一次组合优选的分配报告（picker 写、shopping_summary 注入文案时读）。"""
    k = _key()
    if k is not None:
        _REPORT[k] = report


def get_bundle_report() -> dict[str, Any] | None:
    """读最近一次组合报告；本轮没跑过组合优选返回 None。"""
    k = _key()
    return _REPORT.get(k) if k is not None else None


def reset_session_bundle() -> None:
    """清本会话的槽位状态：``run_agent`` 收尾调（槽只活一轮，也防模块级 dict 无界增长）；
    planner 判换域时调（旧套装与新需求无关）。"""
    k = _key()
    if k is None:
        return
    _BUNDLE.pop(k, None)
    _SEARCHED.pop(k, None)
    _REPORT.pop(k, None)
    _DECLINED.pop(k, None)
    _MODE.pop(k, None)


# ── 组合优选（Multiple-Choice Knapsack，穷举）──────────────────────────────────


class SlotPick(NamedTuple):
    """组合定稿的一件：属于哪个槽（完整槽对象，name 供盖章与文案）+ 候选本体 +
    命中的偏好词（含槽级 prefer，供理由）。"""

    slot: BundleSlot
    cand: ItemCandidate
    matched: list[str]


class BundleOutcome(NamedTuple):
    chosen: list[SlotPick]  # 按槽位定义顺序
    report: dict[str, Any]  # 分配报告（picker 结果文本 + summary 注入共用）


def _price(c: ItemCandidate) -> float | None:
    """组合用的有效价：优先到手价（与 item_picker._effective_price 同口径）。"""
    return c.landed_usd if c.landed_usd is not None else c.price_usd


def _searchable(c: ItemCandidate) -> str:
    return f"{c.title} {c.brand} {c.category}".lower()


def slot_query(s: BundleSlot) -> str:
    """槽的**干净品类 query**（cross-encoder 相关性打分用）：只用检索 keywords，**绝不拼
    prefer 软偏好词**——拼了实测排序反转（背包 badcase：偏好词字面命中把跨品类垃圾抬到
    真品之上）。keywords 为空（用户确认新增的槽）返回空串 = 该槽无法执法，相关性门跳过。
    """
    return " ".join(kw for kw in s.keywords if kw.strip())


def prospective_slot(c: ItemCandidate, slots: list[BundleSlot]) -> str:
    """这件候选将归入哪个槽（返回**槽名**）：盖章优先（机制主通路），没盖章或章不在本轮槽表
    里的用槽 keywords 匹配标题兜底；都不中 → ``""``。``_assign`` 与 picker 的相关性打分共用
    这一份判定，防两处逻辑漂移（否则被门降级的候选会从「盖章路」漏进「keywords 兜底路」
    二次归槽）。
    """
    if c.slot and any(s.name == c.slot for s in slots):
        return c.slot
    text = _searchable(c)
    return next(
        (s.name for s in slots if any(term_hits(kw, text) for kw in s.keywords if kw.strip())),
        "",
    )


def _assign(
    survivors: list[ItemCandidate],
    slots: list[BundleSlot],
    slot_relevance: dict[str, float] | None = None,
    relevance_floor: float = 0.0,
) -> dict[str, list[ItemCandidate]]:
    """把幸存候选归到槽；``slot_relevance``（item_id → cross-encoder 分）非 None 时执行
    **槽位相关性门**：低于 ``relevance_floor`` 的候选逐出槽、落 ``""`` 组——「water bottle
    stickers」这类标题蹭词的跨品类垃圾，keywords 字面匹配拦不住（标题真含 water bottle），
    只有语义门拦得住。逐出后槽内可能一件不剩 → 组合按缺货如实报，绝不硬塞。

    keywords 匹不上的候选**不硬塞**（归错槽比丢一件更糟——组合会拿台灯占床品的名额），
    落进 ``""`` 组，报告里如实计数。
    """
    groups: dict[str, list[ItemCandidate]] = {s.name: [] for s in slots}
    groups[""] = []
    gated = {s.name for s in slots if slot_query(s)} if slot_relevance is not None else set()
    for c in survivors:
        hit = prospective_slot(c, slots)
        if (
            hit
            and hit in gated
            and slot_relevance is not None
            and c.item_id in slot_relevance
            and slot_relevance[c.item_id] < relevance_floor
        ):
            hit = ""
        groups[hit].append(c)
    return groups


def _slot_options(
    stocked: list[BundleSlot],
    groups: dict[str, list[ItemCandidate]],
    base_scores: dict[str, float],
    matched: dict[str, list[str]],
    *,
    top_n: int,
    w_cheap: float,
    w_slot_pref: float,
    slot_relevance: dict[str, float] | None,
    w_relevance: float,
) -> dict[str, list[tuple[float, ItemCandidate, list[str]]]]:
    """每槽的候选按**槽内**打分排序，取 top N。两种形态共用这一份打分。

    便宜度必须在槽内归一：床垫（$200 档）在全局归一里永远垫底、台灯（$20 档）永远满分，
    跨槽比就成了比价格档位而不是比商品优劣。并列形态同理——跑鞋和耳机的价格没有可比性。
    """
    options: dict[str, list[tuple[float, ItemCandidate, list[str]]]] = {}
    for s in stocked:
        cands = groups[s.name]
        priced = [p for p in (_price(c) for c in cands) if p is not None]
        lo, hi = (min(priced), max(priced)) if priced else (0.0, 0.0)
        span = hi - lo
        rows: list[tuple[float, ItemCandidate, list[str]]] = []
        for c in cands:
            p = _price(c)
            cheap = 0.5 if (p is None or span == 0) else (hi - p) / span
            slot_hits = [kw for kw in s.prefer if kw.strip() and term_hits(kw, _searchable(c))]
            score = base_scores.get(c.item_id, 0.0) + w_cheap * cheap + w_slot_pref * len(slot_hits)
            # 槽内品类相关性加分（cross-encoder）：真品在场时把蹭词垃圾压下去（实测真水杯
            # 0.92 vs 贴纸 ≤0.60）。只做**排序信号**不做二值门——绝对分数因 query 措辞剧烈
            # 漂移（真笔袋 vs "stationery pen" 才 0.055），阈值门已被真实数据标定证伪。
            if slot_relevance is not None and c.item_id in slot_relevance:
                score += w_relevance * slot_relevance[c.item_id]
            rows.append((score, c, [*slot_hits, *matched.get(c.item_id, [])]))
        rows.sort(key=lambda r: r[0], reverse=True)
        options[s.name] = rows[:top_n]
    return options


def combine_bundle(
    survivors: list[ItemCandidate],
    base_scores: dict[str, float],
    matched: dict[str, list[str]],
    budget_usd: float | None,
    *,
    w_cheap: float,
    w_slot_pref: float,
    slot_relevance: dict[str, float] | None = None,
    relevance_floor: float = 0.0,
    w_relevance: float = 0.0,
) -> BundleOutcome | None:
    """在总预算约束下做跨槽组合优选；本轮不构成套装（槽 <2 或分组后不足 2 组有货）返回 None。

    入参 ``base_scores`` / ``matched`` 是 item_picker 已算好的**槽无关**部分（偏好命中 + 语义 +
    评分——注意**不含便宜度**）：便宜度必须在**槽内**归一重算，否则床垫（$200 档）在全局归一里
    永远垫底、台灯（$20 档）永远满分，跨槽求和就被价格档位而非商品优劣主导了。

    选择规则：essential 槽必选一件（有货的前提下）、optional 槽可整槽放弃（skip 记 0 分 0 价）；
    可行组合里取 Σ分数最大、同分取更省的；**没有可行组合时取最省的**并如实标 ``feasible=False``
    + 超支额——宁可告诉用户「最省也要超 $x」，绝不静默超预算。
    """
    slots = get_session_bundle()
    if len(slots) < 2:
        return None
    groups = _assign(survivors, slots, slot_relevance, relevance_floor)
    stocked = [s for s in slots if groups.get(s.name)]
    if len(stocked) < 2:
        return None  # 打标全失败 / 只有一个槽有货——组合无意义，退化普通精挑

    options = _slot_options(
        stocked,
        groups,
        base_scores,
        matched,
        top_n=TOP_PER_SLOT,
        w_cheap=w_cheap,
        w_slot_pref=w_slot_pref,
        slot_relevance=slot_relevance,
        w_relevance=w_relevance,
    )

    # 穷举组合：optional 槽多一个「放弃」选项（None，0 分 0 价）。价格未知按 0 计入（组合层
    # 不惩罚它，报告里如实标注件数——比拍一个假价格诚实）。
    choice_lists: list[list[tuple[float, ItemCandidate, list[str]] | None]] = [
        [*options[s.name], *([None] if not s.essential else [])] for s in stocked
    ]
    best: tuple[float, float, tuple] | None = None  # (总分, 总价, 组合)
    best_any: tuple[float, float, tuple] | None = None  # 无视预算的最省组合（不可行时的兜底）
    for combo in itertools.product(*choice_lists):
        total = sum(_price(row[1]) or 0.0 for row in combo if row is not None)
        score = sum(row[0] for row in combo if row is not None)
        if best_any is None or (total, -score) < (best_any[1], -best_any[0]):
            best_any = (score, total, combo)
        if budget_usd is not None and total > budget_usd:
            continue
        if best is None or (score, -total) > (best[0], -best[1]):
            best = (score, total, combo)
    feasible = best is not None
    _score, total, combo = best if best is not None else best_any  # type: ignore[misc]

    chosen = [
        SlotPick(slot=s, cand=row[1], matched=row[2])
        for s, row in zip(stocked, combo, strict=True)
        if row is not None
    ]
    report = _build_report(
        slots, stocked, options, chosen, combo, budget_usd, total, feasible, len(groups[""])
    )
    set_bundle_report(report)
    return BundleOutcome(chosen=chosen, report=report)


def combine_parallel(
    survivors: list[ItemCandidate],
    base_scores: dict[str, float],
    matched: dict[str, list[str]],
    budget_usd: float | None,
    *,
    w_cheap: float,
    w_slot_pref: float,
    slot_relevance: dict[str, float] | None = None,
    relevance_floor: float = 0.0,
    w_relevance: float = 0.0,
) -> BundleOutcome | None:
    """并列形态的分配：**每个子需求各自取 top N**，不做跨槽组合优选。

    与 :func:`combine_bundle` 的区别只有一件事，但它是这个形态存在的全部理由：**不砍类**。
    MCKP 为了「一套不超总预算」可以整槽放弃 optional 槽，用在「跑鞋 + 降噪耳机」这种并列
    需求上，就是把用户明说要看的一类东西悄悄弄丢。这里没有跨槽预算耦合——预算在 picker 上游
    已按**单件**硬筛过（每类各自受同一个上限约束），到这一步只剩「每类挑几件最好的」。

    分组、槽内打分、缺货如实报全部复用 bundle 那条路：形态不同的是**选择规则**，不是机制。
    槽 <2 或分组后不足 2 组有货 → None，退化普通精挑（失效方向安全）。
    """
    slots = get_session_bundle()
    if len(slots) < 2:
        return None
    groups = _assign(survivors, slots, slot_relevance, relevance_floor)
    stocked = [s for s in slots if groups.get(s.name)]
    if len(stocked) < 2:
        return None  # 只有一类有货——分组展示无意义，退化普通精挑
    options = _slot_options(
        stocked,
        groups,
        base_scores,
        matched,
        top_n=PARALLEL_PER_SLOT,
        w_cheap=w_cheap,
        w_slot_pref=w_slot_pref,
        slot_relevance=slot_relevance,
        w_relevance=w_relevance,
    )
    chosen = [
        SlotPick(slot=s, cand=row[1], matched=row[2]) for s in stocked for row in options[s.name]
    ]
    report = _build_parallel_report(slots, stocked, chosen, budget_usd, len(groups[""]))
    set_bundle_report(report)
    return BundleOutcome(chosen=chosen, report=report)


def _build_parallel_report(
    slots: list[BundleSlot],
    stocked: list[BundleSlot],
    chosen: list[SlotPick],
    budget_usd: float | None,
    unslotted: int,
) -> dict[str, Any]:
    """并列形态的报告。键与 bundle 报告**同名同义**（下游 render / 刷新 / 落盘共用一套读法），
    差别只在语义标注：``budget_usd`` 是**每件**上限不是总预算，``total_usd`` 只是各件求和的
    参考数（并列需求没有「一套的总价」这回事），因此 ``feasible`` 恒 True、不判超支。
    """
    searched = searched_slots()
    stocked_names = {s.name for s in stocked}
    total = sum(p or 0.0 for p in (_price(c.cand) for c in chosen))
    return {
        "mode": SLOT_MODE_PARALLEL,
        "budget_usd": budget_usd,
        "total_usd": round(total, 2),
        "feasible": True,
        "over_usd": 0,
        "rows": [
            {
                "slot": p.slot.name,
                "essential": p.slot.essential,
                "item_id": p.cand.item_id,
                "title": p.cand.title[:60],
                "price_usd": _price(p.cand),
            }
            for p in chosen
        ],
        "skipped_optional": [],  # 并列形态不砍类，这一栏永远空
        # 搜了但一件都没有的类：并列需求里 essential 没有意义（用户要的每一类都得如实交代），
        # 但键名沿用 bundle 那套，渲染层按 mode 换措辞即可。
        "missing_essential": sorted(
            s.name for s in slots if s.name not in stocked_names and s.name in searched
        ),
        "missing_optional": [],
        "not_included": sorted(
            s.name for s in slots if s.name not in stocked_names and s.name not in searched
        ),
        "unslotted": unslotted,
        "price_unknown": sum(1 for p in chosen if _price(p.cand) is None),
        # 并列形态每类已经展示了 top N，「升降级备选」没有额外信息量，留空键保结构一致。
        "alternatives": {},
    }


def _build_report(
    slots: list[BundleSlot],
    stocked: list[BundleSlot],
    options: dict[str, list[tuple[float, ItemCandidate, list[str]]]],
    chosen: list[SlotPick],
    combo: tuple,
    budget_usd: float | None,
    total: float,
    feasible: bool,
    unslotted: int,
) -> dict[str, Any]:
    """组合结果 → 分配报告（picker 结果文本 / summary 注入 / ItemPickerOutput 回显共用）。"""
    searched = searched_slots()
    stocked_names = {s.name for s in stocked}
    chosen_ids = {p.cand.item_id for p in chosen}
    rows = [
        {
            "slot": p.slot.name,
            "essential": p.slot.essential,
            "item_id": p.cand.item_id,
            "title": p.cand.title[:60],
            "price_usd": _price(p.cand),
        }
        for p in chosen
    ]
    # 每槽的升/降级备选（组合没选上的前两名）：追问轮「箱子换便宜的」可直接引用。
    alternatives = {
        name: [
            {"item_id": c.item_id, "title": c.title[:50], "price_usd": _price(c)}
            for _sc, c, _m in opts
            if c.item_id not in chosen_ids
        ][:2]
        for name, opts in options.items()
    }
    return {
        "budget_usd": budget_usd,
        "total_usd": round(total, 2),
        "feasible": feasible,
        "over_usd": (
            round(total - budget_usd, 2) if (budget_usd is not None and not feasible) else 0
        ),
        "rows": rows,
        # optional 槽进了枚举但组合放弃了它（预算紧 / 分数为负）。
        "skipped_optional": [s.name for s, row in zip(stocked, combo, strict=True) if row is None],
        # essential 槽检索过但一件候选都没有——如实报缺，绝不拿别的槽的货顶。
        "missing_essential": sorted(
            s.name
            for s in slots
            if s.essential and s.name not in stocked_names and s.name in searched
        ),
        # optional 槽检索过但没货（含被相关性门逐空的，如「水杯」槽召回全是贴纸）——同样
        # 如实列出：不列它就是静默消失，用户以为这件没被考虑过。
        "missing_optional": sorted(
            s.name
            for s in slots
            if not s.essential and s.name not in stocked_names and s.name in searched
        ),
        # 定义了但本轮没检索（用户在确认组成时删掉的槽，或模型没派）——中性列出，不算缺货。
        "not_included": sorted(
            s.name for s in slots if s.name not in stocked_names and s.name not in searched
        ),
        "unslotted": unslotted,
        "price_unknown": sum(1 for p in chosen if _price(p.cand) is None),
        "alternatives": alternatives,
    }


def drop_pick_from_report(item_id: str) -> None:
    """把收尾阶段被摘除的入选商品（slot off-intent：贴纸占了水杯槽）从分配报告里剔掉。

    该槽改报缺货（missing_essential / missing_optional 按行内 essential 标记归类）、总价重算。
    就地改 ``_REPORT`` 里的那份（get_bundle_report 返回引用）——报告是续聊轮与产物落盘的
    事实来源，不同步就会「商品卡没有贴纸、分配表还挂着它」。本轮没有报告 / 行不在则静默跳过。
    """
    report = get_bundle_report()
    if not report:
        return
    row = next((r for r in report.get("rows", []) if r.get("item_id") == item_id), None)
    if row is None:
        return
    report["rows"] = [r for r in report["rows"] if r.get("item_id") != item_id]
    key = "missing_essential" if row.get("essential", True) else "missing_optional"
    slot = str(row.get("slot", ""))
    # 并列形态一类有好几件，摘掉一件不等于这类没货了——该类还剩行就不报缺（bundle 每槽只有
    # 一件，摘掉即空，行为与原来一致）。
    still_there = any(r.get("slot") == slot for r in report["rows"])
    if slot and not still_there and slot not in report.get(key, []):
        report[key] = sorted([*report.get(key, []), slot])
    report["total_usd"] = round(
        sum(r["price_usd"] for r in report["rows"] if r.get("price_usd") is not None), 2
    )


def refresh_report_prices(report: dict[str, Any], picks: list[ItemCandidate]) -> dict[str, Any]:
    """用**当前**有效价（到手价优先）刷新分配表的单价与总价，返回新 report（不改原件）。

    组合是在比价 / 到手价**之前**定稿的（按货价算），收尾时入选件多半已补上 landed_usd——
    不刷新就会出现「分配表总价 $25.48、商品卡却写到手 $28.32」的自相矛盾（e2e 实测）。
    刷新后若超了预算，如实改标 ``feasible`` / ``over_usd``：件已定、组合不重跑，但超支必须
    说出来而不是藏在旧口径里。
    """
    by_id = {c.item_id: c for c in picks}
    rows = [dict(r) for r in report.get("rows", [])]
    total = 0.0
    bare = 0  # 只有裸价、没算出到手价的件数（summary 已尽力补算后仍缺的，如实标口径）
    for r in rows:
        item_id = r.get("item_id")
        c = by_id.get(item_id) if isinstance(item_id, str) else None
        p = _price(c) if c is not None else r.get("price_usd")
        if p is not None:
            r["price_usd"] = round(p, 2)
            total += p
        if c is not None and c.landed_usd is None and c.price_usd is not None:
            bare += 1
    out = {**report, "rows": rows, "total_usd": round(total, 2), "bare_price": bare}
    budget = report.get("budget_usd")
    # 并列形态的 budget 是**每件**上限（单件超预算在 picker 上游就淘汰了），各类价格求和超过它
    # 完全正常——在这儿判超支会凭空报一条「超预算」的假警。
    if report.get("mode") == SLOT_MODE_PARALLEL:
        return out
    if budget is not None and total > budget:
        out["feasible"] = False
        out["over_usd"] = round(total - budget, 2)
    return out


def _render_parallel(report: dict[str, Any]) -> str:
    """并列形态的分配文本。**刻意不出现总价与「一套」字样**：用户要的是几类互不相干的东西，
    把跑鞋和耳机的价格加起来给他看，这个数字没有任何含义，还会诱导收尾文案讲成「这一套」。
    """
    by_slot: dict[str, list[dict[str, Any]]] = {}
    for r in report["rows"]:
        by_slot.setdefault(str(r["slot"]), []).append(r)
    lines = [f"分头调研：{len(by_slot)} 类各给推荐（各类独立、不配套，**禁止把各类价格相加**）"]
    for slot, rows in by_slot.items():
        priced = [r["price_usd"] for r in rows if r["price_usd"] is not None]
        span = f"${min(priced):.2f}–${max(priced):.2f}" if priced else "价格未知"
        lines.append(f"·【{slot}】{len(rows)} 件（{span}）")
        lines += [f"  - {r['title']}" for r in rows]
    budget = report.get("budget_usd")
    if budget is not None:
        lines.append(f"预算口径：每件 ≤ ${budget:.2f}（不是几类加起来的总额）")
    if report.get("bare_price"):
        lines.append(
            f"⚠ 其中 {report['bare_price']} 件只有商品裸价（未含运费关税），文案不得声称"
            "「全部含税到手价」"
        )
    if report["missing_essential"]:
        lines.append("搜了但没找到货的类：" + "、".join(report["missing_essential"]))
    if report["not_included"]:
        lines.append("本轮未检索的类（不是缺货）：" + "、".join(report["not_included"]))
    return "\n".join(lines)


def render_allocation(report: dict[str, Any]) -> str:
    """把分配报告渲染成人读的多行文本（前端思考过程 + summary 注入共用，零 LLM）。"""
    if report.get("mode") == SLOT_MODE_PARALLEL:
        return _render_parallel(report)
    lines: list[str] = []
    budget = report.get("budget_usd")
    head = f"套装组合：总价 ${report['total_usd']:.2f}"
    if budget is not None:
        head += f"（总预算 ${budget:.2f}"
        head += f"，剩余 ${budget - report['total_usd']:.2f}）" if report["feasible"] else "）"
    lines.append(head)
    if not report["feasible"]:
        lines.append(f"⚠ 预算内凑不齐这一套：当前组合超支 ${report['over_usd']:.2f}")
    for r in report["rows"]:
        price = f"${r['price_usd']:.2f}" if r["price_usd"] is not None else "价格未知"
        tag = "必备" if r["essential"] else "可选"
        lines.append(f"·【{r['slot']}】{r['title']}（{price}，{tag}）")
    if report.get("bare_price"):
        lines.append(
            f"⚠ 其中 {report['bare_price']} 件只有商品裸价（未含运费关税）——总价口径是混合的，"
            "文案不得声称「全部含税到手价」"
        )
    if report["skipped_optional"]:
        lines.append("已放弃的可选槽：" + "、".join(report["skipped_optional"]))
    if report["missing_essential"]:
        lines.append("搜了但没找到货的必备槽：" + "、".join(report["missing_essential"]))
    if report.get("missing_optional"):
        lines.append("搜了但没找到合适货的可选槽：" + "、".join(report["missing_optional"]))
    if report["not_included"]:
        lines.append(
            "本轮未检索的槽（组成里定义了但没派检索，不是缺货）："
            + "、".join(report["not_included"])
        )
    return "\n".join(lines)
