"""套装（bundle）机制层 —— 「一套齐」跨品类组合优选的状态与算法。

场景：「新生入学一套，预算 1500」不是单品类清单，而是**跨品类凑一套**：planner 拆出
3~6 个槽位（床品 / 台灯 / 收纳箱…）→ 每槽一个子 Agent 并行检索 → 在**总预算**约束下做
组合优选（哪槽该花钱、哪槽降级、加起来不超）→ 分组收尾。

本模块只放**机制**（状态 + 纯算法），不放工具入口：
  - 槽位定义登记：planner 判出 ``bundle_slots`` 后写进来（模块 dict 按 session_dir 聚合，
    跨工具可见——同 ``_candidates`` 的既有套路），并落 ``bundle.json`` 供续聊轮读回；
    planner 判 ``search``（换品类）时随候选池一起清掉——**槽位生命周期 = 候选池生命周期**。
  - 检索侧打标：候选属于哪个槽，主通路是 dispatch 派发时从 demand 里**确定性**解析
    「套装槽位：X」标记，经 ContextVar 传给子 Agent 内的 item_search 盖章；模型侧另有
    ``item_search(slot=...)`` 参数与关键词归槽兜底。三层兜底，不指望任何单点自觉。
  - 组合优选：Multiple-Choice Knapsack——essential 槽必选一件、optional 槽可整槽放弃，
    约束 Σ有效价 ≤ 总预算，目标 max Σ分数。槽 ≤6 × 每槽 top5 → 穷举即可（≤ 数万组合，
    毫秒级），不需要近似算法。**不可行时如实报**：给最省组合 + 超支额，绝不静默超预算。
  - 组合报告：分配表（哪槽花了多少、砍了谁、缺了谁）登记给 shopping_summary 注入文案。

「是不是套装轮」由机制判（会话里登记的槽 ≥2），不由模型自报——同 planner 的 retrieval
/ 币种确定性回填一个思路。
"""

from __future__ import annotations

import itertools
import json
import logging
import re
from collections.abc import Iterable
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, NamedTuple

from pydantic import BaseModel, Field, ValidationError

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


class BundleSlot(BaseModel):
    """套装里的一个槽位（要买的一个子品类）。planner 拆解产出，picker 组合消费。

    ``id`` 是槽的**稳定身份**（s1、s2…，登记时机制发号，模型无权自造）：盖章 / 检索记账 /
    补搜额度 / 拒绝复活全按 id 走，名字降级为展示属性——LLM 措辞漂移只可能发生在「名字 → id」
    的解析边界（:func:`resolve_slot`，全链路唯一模糊点），进了身份层就再也不会漂。
    """

    id: str = Field(default="", description="稳定槽位 id（s1、s2…），set_session_bundle 发号")
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
# session_dir -> 槽位定义（planner 写、picker/dispatch 读）
_BUNDLE: dict[str, list[BundleSlot]] = {}
# session_dir -> 本轮真正检索过的槽 **id**（item_search 盖章时记）。用来区分「搜了但没货」
# （essential 缺货，要如实报）与「压根没派」（用户在 ask_user 里删掉的槽，静默不включ）。
_SEARCHED: dict[str, set[str]] = {}
# session_dir -> 最近一次组合优选的报告（picker 写、shopping_summary 注入文案时读）。
_REPORT: dict[str, dict[str, Any]] = {}
# session_dir -> 用户在组成确认里明确不要的槽（reconcile_slots_from_reply 记，随 bundle.json
# 落盘——只放内存的话续聊轮清内存后拦不住复活）。register_slot 据此拒绝复活：demand 文本里
# 再飘出这个词不代表用户改了主意。存 {"id","name"}：id 供记账，name 供创建时的漂移匹配。
_DECLINED: dict[str, list[dict[str, str]]] = {}

_BUNDLE_FILE = "bundle.json"


def _key() -> str | None:
    sd = get_session_dir()
    return str(sd) if sd is not None else None


def _next_id(k: str, slots: list[BundleSlot]) -> int:
    """下一个可发的槽号：已用号（含 declined 里退役的）最大值 +1——id 永不复用。"""
    used = [s.id for s in slots] + [d.get("id", "") for d in _DECLINED.get(k, [])]
    nums = [int(i[1:]) for i in used if re.fullmatch(r"s\d+", i or "")]
    return max(nums, default=0) + 1


def set_session_bundle(slots: Iterable[BundleSlot]) -> None:
    """登记本会话的套装槽位（planner 判出 bundle_slots ≥2 时调），并落盘供续聊轮读回。

    机制在此**发槽位 id**（s1、s2…，没带 id 的补发、带了的保留）——id 是身份，模型无权自造。
    落盘失败只记日志——槽位是工作记忆，丢了最多退化成普通单品类清单，不拖垮主链路。
    """
    k = _key()
    if k is None:
        return
    cleaned = [s for s in slots if s.name.strip()][:MAX_SLOTS]
    if not cleaned:
        return
    seen_ids: set[str] = set()
    for s in cleaned:  # 撞号（不管来路）后到者视为没号，重新发——id 唯一性是身份层的地基
        if s.id in seen_ids:
            s.id = ""
        elif re.fullmatch(r"s\d+", s.id or ""):
            seen_ids.add(s.id)
    n = _next_id(k, cleaned)
    for s in cleaned:
        if not re.fullmatch(r"s\d+", s.id or ""):
            s.id = f"s{n}"
            n += 1
    _BUNDLE[k] = cleaned
    sd = get_session_dir()
    if sd is not None:
        try:
            (sd / _BUNDLE_FILE).write_text(
                json.dumps(
                    {
                        "slots": [s.model_dump() for s in cleaned],
                        "declined": _DECLINED.get(k, []),
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
        except OSError as exc:
            logger.warning("套装槽位落盘失败（续聊轮将退化为普通清单）：%s", exc)


def get_session_bundle() -> list[BundleSlot]:
    """读本会话的套装槽位；内存 miss 时懒读 ``bundle.json`` 回灌（续聊轮 / 进程重启后续跑）。

    无会话作用域（单测直调）或从没登记过 → 空列表 = 本轮不是套装，picker 走普通精挑。
    """
    k = _key()
    if k is None:
        return []
    if k in _BUNDLE:
        return list(_BUNDLE[k])
    sd = get_session_dir()
    path = sd / _BUNDLE_FILE if sd is not None else None
    if path is None or not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        # 旧格式（裸列表，无 id/declined）兼容：按序补发 id，declined 视为空。
        rows = data if isinstance(data, list) else data.get("slots", [])
        slots = [BundleSlot.model_validate(r) for r in rows]
        declined = [] if isinstance(data, list) else list(data.get("declined", []))
    except (OSError, ValueError, ValidationError) as exc:
        logger.warning("套装槽位读回失败（本轮退化为普通清单）：%s", exc)
        return []
    _DECLINED[k] = declined
    n = _next_id(k, slots)
    for s in slots:
        if not re.fullmatch(r"s\d+", s.id or ""):
            s.id = f"s{n}"
            n += 1
    _BUNDLE[k] = slots
    return list(slots)


def _match_name(candidate: str, target: str) -> bool:
    """槽名模糊匹配：精确相等，或互为包含（短名 ≥2 字，防「包」这类单字吸走所有槽）。"""
    if candidate == target:
        return True
    return min(len(candidate), len(target)) >= 2 and (candidate in target or target in candidate)


def resolve_slot(ref: str) -> BundleSlot | None:
    """把一个模型/用户产出的槽引用（id / 精确名 / 漂移名）解析成已登记槽。

    这是**全链路唯一的字符串模糊点**：item_search 盖章、补搜闸、demand 打标全经此收口，
    解析成功后拿到的是带稳定 id 的槽对象，身份层（记账/额度/报告）从此只讲 id、不再漂移。
    内置懒读回（get_session_bundle）——续聊轮内存被清后照常工作。解析不出 → None。
    """
    ref = (ref or "").strip()
    if not ref:
        return None
    slots = get_session_bundle()
    for s in slots:
        if s.id == ref:
            return s
    for s in slots:
        if s.name == ref:
            return s
    return next((s for s in slots if _match_name(s.name, ref)), None)


def slot_display(ref: str) -> str:
    """槽引用 → 展示名（卡片 / 预览 / 报告用）。解析不出原样返回——旧会话候选按名字盖的章、
    非套装轮的空串都直接透传。"""
    s = resolve_slot(ref)
    return s.name if s is not None else ref


def register_slot(ref: str) -> str:
    """把一个槽引用解析成**槽 id**；确实是新品类时补登新槽（「用户确认新增」通路），
    返回其 id。解析不出且不该创建 → 空串（调用方不盖章）。

    创建的门槛：套装已激活（≥1 槽，懒读回后判）、不是纯 id 形状的野引用（模型幻觉出 s9
    不代表用户要买叫「s9」的东西）、没撞上用户在组成确认里明确删掉的槽（含漂移匹配——
    「台灯」被删后「护眼台灯」也不复活）、没超槽数上限。
    """
    ref = (ref or "").strip()
    hit = resolve_slot(ref)
    if hit is not None:
        return hit.id
    k = _key()
    slots = get_session_bundle()
    if k is None or not ref or not slots or re.fullmatch(r"s\d+", ref):
        return ""
    if any(_match_name(d.get("name", ""), ref) for d in _DECLINED.get(k, [])):
        return ""  # 用户在组成确认里明确不要过它，不复活
    if len(slots) >= MAX_SLOTS:
        return ""
    new = BundleSlot(name=ref, essential=True, evidence="用户确认新增")
    set_session_bundle([*slots, new])  # 走同一落盘口发 id，保持文件与内存一致
    return new.id


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
    slots = get_session_bundle()  # 懒读回：ask_user 回复可能是续聊轮的第一次槽表访问
    if k is None or not reply or not slots:
        return []
    mentioned = {s.id for s in slots if s.name and s.name in reply}
    if len(mentioned) < 2:
        return []
    offered_text = " ".join(offered) if offered else ""
    removed = [
        s for s in slots if s.id not in mentioned and (not offered_text or s.name in offered_text)
    ]
    if not removed:
        return []
    # 先记 declined 再落盘——set_session_bundle 连带把 declined 写进 bundle.json，
    # 续聊轮清内存后拒绝复活依然生效。
    _DECLINED.setdefault(k, []).extend({"id": s.id, "name": s.name} for s in removed)
    removed_ids = {s.id for s in removed}
    set_session_bundle([s for s in slots if s.id not in removed_ids])
    names = [s.name for s in removed]
    logger.info("组成确认核销：删槽 %s，保留 %s", names, sorted(mentioned))
    return names


def note_slot_searched(ref: str) -> None:
    """记下「这个槽本轮真的检索过」（item_search 盖章时调）——essential 缺货判定的依据。

    接受任意槽引用（id / 名），统一解析成 id 入账；解析不出的原样记（不丢信息，只是
    报告层对不上号——等价于「没搜过」，失效方向与漏记一致）。
    """
    k = _key()
    ref = (ref or "").strip()
    if k is None or not ref:
        return
    s = resolve_slot(ref)
    _SEARCHED.setdefault(k, set()).add(s.id if s is not None else ref)


def searched_slots() -> set[str]:
    """本轮检索过的槽 id 集合。"""
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


def reset_session_bundle(*, clear_file: bool = False) -> None:
    """清本会话的套装状态。

    两种调法：``run_agent`` 收尾清**内存**（模块级 dict 防无界增长，文件留着供续聊轮读回）；
    planner 判 ``search``（换品类）时 ``clear_file=True`` 连盘上的一起清——旧套装与新需求无关。
    """
    k = _key()
    if k is None:
        return
    _BUNDLE.pop(k, None)
    _SEARCHED.pop(k, None)
    _REPORT.pop(k, None)
    _DECLINED.pop(k, None)
    if clear_file:
        sd = get_session_dir()
        if sd is not None:
            (sd / _BUNDLE_FILE).unlink(missing_ok=True)


# ── 检索侧槽位打标 ─────────────────────────────────────────────────────────────
# dispatch 派发子 Agent 前从 demand 解析出槽名，slot_scope 设进 ContextVar；子 Agent 内的
# item_search（asyncio task 继承 ContextVar 快照）读它给候选盖章。父子是「向下继承」，
# 不是工具间横向传递，所以这里用裸 ContextVar 是安全的（对比 context.py 里踩过三次的坑）。
_current_slot: ContextVar[str] = ContextVar("shoppingx_search_slot", default="")

# demand 里的确定性槽位标记（prompt 约定每条套装 demand 开头写「套装槽位：X」）。
_SLOT_MARKER_RE = re.compile(r"套装槽位[:：]\s*([^\s，。;；,、）)]+)")


@contextmanager
def slot_scope(name: str):
    """把当前检索槽名设进 ContextVar（dispatch 派发子 Agent 时包住整个子 loop）。"""
    token = _current_slot.set(name.strip())
    try:
        yield
    finally:
        _current_slot.reset(token)


def current_slot() -> str:
    """当前检索所属的槽名；非套装派发路径为空串。"""
    return _current_slot.get()


def detect_slot(text: str) -> str | None:
    """从一条 demand 文本里确定性解析它属于哪个槽，返回**槽引用**（id 或原始标记文本）。

    优先认「套装槽位：X」标记——原样返回（可能是新加的槽名，交给 item_search 的
    register_slot 统一解析/补登，保持全链路单一解析点）；退而求其次匹已登记的槽名
    （长名优先，避免「床品」抢了「床垫床品套装」的匹配），命中返回其 id。
    都没有 → None（这条不是套装检索）。
    """
    m = _SLOT_MARKER_RE.search(text)
    if m:
        return m.group(1)
    for s in sorted(get_session_bundle(), key=lambda s: -len(s.name)):
        if s.name and s.name in text:
            return s.id
    return None


# ── 组合优选（Multiple-Choice Knapsack，穷举）──────────────────────────────────


class SlotPick(NamedTuple):
    """组合定稿的一件：属于哪个槽（完整槽对象，id 供盖章、name 供文案）+ 候选本体 +
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
    """这件候选将归入哪个槽（返回**槽 id**）：盖章优先（机制主通路，章即 id；旧会话读回的
    候选按名字盖章，兼容认作对应槽），没盖章的用槽 keywords 匹配标题兜底；都不中 → ``""``。
    ``_assign`` 与 picker 的相关性打分共用这一份判定，防两处逻辑漂移（否则被门降级的候选
    会从「盖章路」漏进「keywords 兜底路」二次归槽）。
    """
    if c.slot:
        hit = next((s for s in slots if s.id == c.slot or s.name == c.slot), None)
        if hit is not None:
            return hit.id
    text = _searchable(c)
    return next(
        (s.id for s in slots if any(term_hits(kw, text) for kw in s.keywords if kw.strip())),
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
    groups: dict[str, list[ItemCandidate]] = {s.id: [] for s in slots}
    groups[""] = []
    gated = {s.id for s in slots if slot_query(s)} if slot_relevance is not None else set()
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
    stocked = [s for s in slots if groups.get(s.id)]
    if len(stocked) < 2:
        return None  # 打标全失败 / 只有一个槽有货——组合无意义，退化普通精挑

    # 槽内打分：base + 槽内便宜度 + 槽级 prefer 命中，取每槽 top N 进枚举。
    options: dict[str, list[tuple[float, ItemCandidate, list[str]]]] = {}
    for s in stocked:
        cands = groups[s.id]
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
        options[s.id] = rows[:TOP_PER_SLOT]

    # 穷举组合：optional 槽多一个「放弃」选项（None，0 分 0 价）。价格未知按 0 计入（组合层
    # 不惩罚它，报告里如实标注件数——比拍一个假价格诚实）。
    choice_lists: list[list[tuple[float, ItemCandidate, list[str]] | None]] = [
        [*options[s.id], *([None] if not s.essential else [])] for s in stocked
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
    searched = searched_slots()  # id 集合
    stocked_ids = {s.id for s in stocked}
    chosen_ids = {p.cand.item_id for p in chosen}
    by_id = {s.id: s for s in slots}
    # 报告是展示/文案层的事实来源（summary 注入、前端思考过程、落盘）——槽一律写**名字**；
    # 身份層的 id 只在内存里的比较中使用，不进报告。
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
        by_id[sid].name: [
            {"item_id": c.item_id, "title": c.title[:50], "price_usd": _price(c)}
            for _sc, c, _m in opts
            if c.item_id not in chosen_ids
        ][:2]
        for sid, opts in options.items()
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
            s.name for s in slots if s.essential and s.id not in stocked_ids and s.id in searched
        ),
        # optional 槽检索过但没货（含被相关性门逐空的，如「水杯」槽召回全是贴纸）——同样
        # 如实列出：不列它就是静默消失，用户以为这件没被考虑过。
        "missing_optional": sorted(
            s.name
            for s in slots
            if not s.essential and s.id not in stocked_ids and s.id in searched
        ),
        # 定义了但本轮没检索（用户在确认组成时删掉的槽，或模型没派）——中性列出，不算缺货。
        "not_included": sorted(
            s.name for s in slots if s.id not in stocked_ids and s.id not in searched
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
    if slot and slot not in report.get(key, []):
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
        c = by_id.get(r.get("item_id"))
        p = _price(c) if c is not None else r.get("price_usd")
        if p is not None:
            r["price_usd"] = round(p, 2)
            total += p
        if c is not None and c.landed_usd is None and c.price_usd is not None:
            bare += 1
    out = {**report, "rows": rows, "total_usd": round(total, 2), "bare_price": bare}
    budget = report.get("budget_usd")
    if budget is not None and total > budget:
        out["feasible"] = False
        out["over_usd"] = round(total - budget, 2)
    return out


def render_allocation(report: dict[str, Any]) -> str:
    """把分配报告渲染成人读的多行文本（前端思考过程 + summary 注入共用，零 LLM）。"""
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
