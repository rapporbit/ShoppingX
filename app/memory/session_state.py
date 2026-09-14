"""会话级短期偏好状态 P_t —— 单次选购会话内累积的结构化约束（RecBot 论文原型，arXiv:2509.21317）。

把**一次浏览会话内**用户逐轮下的自然语言命令，映射成一个会话内稳定累积、逐轮 merge 的结构化
偏好状态：品类 / 预算 / 三个词表（硬淘汰 / 软减分 / 加分）。它是「三层记忆」的中间层——既不是
跑完即弃的 messages 上下文，也不是跨会话的长期结论库。

与另两层的分界（务必别混）:
- **长期库**（:mod:`app.memory.store`）：跨会话的一贯取向，按用户聚合、语义去重、带半衰期。
- **本模块 P_t**：**本次选购会话内**的约束（「这次预算 300」「今天不要塑料」），按 thread 隔离、
  随轮累积、会话结束即止。唯一写者是 planner（``_sync_session_pt`` → :func:`merge_pt_lite`），
  curator 只读。
- **行为历史**（``HistoryEntry``）：上次搜 / 买了什么的事实快照，与偏好正交。

**less is more（2026-09-14 重构）**：曾经每条约束带 id / epoch / archived / source_quote / TTL，
撤回靠「LLM 抄 id + 词面核验」——五个机制服务一个「撤回精确到条」的能力，实测用得极少。现在
退回论文原型：**词表就是状态**，撤回按词（``retract_terms``，逐词对原话核验，过了才删，宁紧），
换品类域整表清空、预算保留。无 id、无代际、无 TTL。

**存放：** 住 ``AgentState.middle_context["pt"]``，随会话唯一的跨轮产物 session.json 一起落盘 /
读回（:func:`pt_into_state` / :func:`pt_from_state`），不单独成文件、不进长期库。读坏只记日志
降级为空（等价「本轮从零累积」），不反噬主链路。
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.utils.terms import normalize_terms, term_hits

logger = logging.getLogger("shoppingx.session_state")


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _atoms(words: Iterable[str]) -> list[str]:
    """小写、去空、保序去重——三个词表的统一形态（拿去和商品标题做字符串匹配）。"""
    out: list[str] = []
    seen: set[str] = set()
    for w in words:
        low = (w or "").strip().lower()
        if low and low not in seen:
            seen.add(low)
            out.append(low)
    return out


class SessionPrefState(BaseModel):
    """一次选购会话的短期偏好状态 P_t（按 thread 隔离，随 session.json 落盘）。

    - ``exclude_terms``：硬淘汰（用户说「不要 X」）→ item_picker 命中即淘汰。
    - ``avoid_terms``：软减分（「尽量别」「不太喜欢」）→ 命中减分、不淘汰。硬淘汰匹不准的词
      就是拿误杀去赌——「花哨」这种词一旦匹上（「塑料感」连坐 plastic），杀掉的可能正是用户要的。
    - ``prefer_terms``：加分。正向**不做二值淘汰**（数据没有可靠的材质 / 风格字段，keep-only 会
      误杀一大片），所以即便「必须金属」也只作强加分。
    - ``category`` / ``domains``：本轮主品类与品类域。换域（bags → footwear）时三个词表清空、
      预算保留——「不要塑料」是买鞋时说的，买沙发不该还在生效；预算是人的钱包，跨品类也在。
    - ``dest_country``：本会话明示过的收货国（「寄到日本」），供到手价四层解析的第 2 层。

    消费接口沿用旧名（``dislike_terms`` / ``soft_dislike_terms`` / ``like_terms``），下游
    assemble / signals / drift / item_picker 零改动。
    """

    model_config = ConfigDict(extra="forbid")

    category: str = Field(default="", description="本轮主品类（最近一次明确的）")
    domains: list[str] = Field(default_factory=list, description="本轮品类域（换域判据）")
    budget_usd: float | None = Field(default=None, description="累积的预算上限 USD，无则 None")
    dest_country: str = Field(default="", description="本会话明示过的收货国 ISO 码")
    exclude_terms: list[str] = Field(default_factory=list, description="硬淘汰词")
    avoid_terms: list[str] = Field(default_factory=list, description="软减分词")
    prefer_terms: list[str] = Field(default_factory=list, description="加分词")
    updated_at: str = Field(default="", description="最近更新时间（UTC ISO）")

    def is_empty(self) -> bool:
        return (
            not self.category
            and self.budget_usd is None
            and not self.dest_country
            and not self.exclude_terms
            and not self.avoid_terms
            and not self.prefer_terms
        )

    def dislike_terms(self) -> list[str]:
        """硬 dislike 词 → item_picker 命中即淘汰。"""
        return list(self.exclude_terms)

    def soft_dislike_terms(self) -> list[str]:
        """软 dislike 词 → item_picker 命中减分、不淘汰。"""
        return list(self.avoid_terms)

    def like_terms(self) -> list[str]:
        """like 词 → item_picker 加分。"""
        return list(self.prefer_terms)

    def render(self) -> str:
        """渲染成给模型看的一段文本（planner 结果尾部 / 排障）。空状态返回占位。"""
        if self.is_empty():
            return "（本会话尚无累积约束）"
        lines: list[str] = []
        if self.category:
            lines.append(f"- 品类：{self.category}")
        if self.budget_usd is not None:
            lines.append(f"- 预算：≤ ${self.budget_usd:.0f}")
        if self.dest_country:
            lines.append(f"- 收货国：{self.dest_country}")
        if self.exclude_terms:
            lines.append("- 硬排除（命中即淘汰）：" + "、".join(self.exclude_terms))
        if self.avoid_terms:
            lines.append("- 软避讳（只减分）：" + "、".join(self.avoid_terms))
        if self.prefer_terms:
            lines.append("- 偏好（命中加分）：" + "、".join(self.prefer_terms))
        return "\n".join(lines)


# ---------- 存放：AgentState.middle_context ----------

# P_t 在 ``AgentState.middle_context`` 里的键。会话唯一的跨轮产物是 session.json
# （= ``AgentState.model_dump_json()``），P_t 作为其中一个子字典随它一起落盘 / 读回。
PT_STATE_KEY = "pt"


def pt_from_state(middle_context: dict[str, Any]) -> SessionPrefState:
    """从 ``AgentState.middle_context`` 取回 P_t；缺失 / 损坏（含旧格式）一律返回空状态，不抛。"""
    raw = middle_context.get(PT_STATE_KEY)
    if not raw:
        return SessionPrefState()
    try:
        return SessionPrefState.model_validate(raw)
    except ValidationError as exc:
        logger.warning("middle_context 里的 P_t 无法解析，按空处理：%s", exc)
        return SessionPrefState()


def pt_into_state(middle_context: dict[str, Any], state: SessionPrefState) -> None:
    """把 P_t 写进 ``AgentState.middle_context``（就地改，空状态也写——覆盖上一轮的旧值）。"""
    state.updated_at = _now_iso()
    middle_context[PT_STATE_KEY] = state.model_dump()


# ---------- 偏好面板：按词展示 / 按词删 ----------

_BUCKETS: tuple[tuple[str, str, str, bool], ...] = (
    ("exclude", "exclude_terms", "dislike", True),
    ("avoid", "avoid_terms", "dislike", False),
    ("prefer", "prefer_terms", "like", False),
)


def constraint_rows(pt: SessionPrefState) -> list[dict[str, Any]]:
    """三个词表 → 面板行（``id`` = ``<词表>:<词>``，删除按它打 DELETE）。前端契约不变。"""
    rows: list[dict[str, Any]] = []
    for bucket, attr, polarity, blocking in _BUCKETS:
        for term in getattr(pt, attr):
            rows.append(
                {
                    "id": f"{bucket}:{term}",
                    "content": term,
                    "source_quote": "",
                    "polarity": polarity,
                    "blocking": blocking,
                }
            )
    return rows


def drop_constraint(pt: SessionPrefState, row_id: str) -> bool:
    """按面板行 id 删一个词（用户亲手点的，不走原话核验）。返回是否真删了（幂等）。"""
    bucket, _, term = row_id.partition(":")
    for name, attr, _p, _b in _BUCKETS:
        if name == bucket:
            terms = getattr(pt, attr)
            if term in terms:
                setattr(pt, attr, [t for t in terms if t != term])
                return True
    return False


# ---------- 合并：唯一调用方 planner ----------


def _same_thing(a: str, b: str) -> bool:
    """两个词是不是同一件事（含中→英扩词：塑料 vs plastic）——撤回 / 极性翻转的判据。"""
    na, nb = set(normalize_terms([a])), set(normalize_terms([b]))
    if na & nb:
        return True
    ta, tb = " ".join(na), " ".join(nb)
    return any(term_hits(w, tb) for w in na) or any(term_hits(w, ta) for w in nb)


def _without(terms: list[str], drop: Iterable[str]) -> list[str]:
    drop = list(drop)
    return [t for t in terms if not any(_same_thing(t, d) for d in drop)]


def merge_pt_lite(
    prev: SessionPrefState,
    *,
    exclude: Iterable[str] = (),
    avoid: Iterable[str] = (),
    prefer: Iterable[str] = (),
    retract_terms: Iterable[str] = (),
    user_utterance: str = "",
    category: str = "",
    domains: Iterable[str] = (),
    budget_usd: float | None = None,
    clear_budget: bool = False,
    dest_country: str = "",
) -> SessionPrefState:
    """把本轮增量 merge 进 P_t——顺序固定、全部确定性代码。

    1. **换域**：本轮 ``domains`` 与既有域**无交集**（两边都非空）→ 三个词表清空，预算保留。
       只比域不比品类字符串：planner 对同一件东西每轮措辞会漂（旅行包 / travel bag），按字符串
       比会把追问轮误判成换品类。
    2. **撤回按词、宁紧**：``retract_terms`` 逐词——只有该词**在本轮原话里出现**（:func:`term_hits`）
       才从三个词表里删掉与之同义的词；核验不过记 warning 不删（幻觉词被这道闸挡住）。
    3. **并入**：新词按小写词面去重追加；一个词进了某个表就从另外两个表里移除（「不要蓝色」→
       「还是要蓝色」极性翻转，最新表达为准）。
    4. 预算 / 品类 / 收货国：None / 空 = 本轮未提及保持不变；``clear_budget`` 单列——「明确放开」
       必须可表达，否则 item_picker 的「无预算用 P_t 兜底」会让旧预算每轮暗中卡人。
    """
    new_domains = _atoms(domains)
    if prev.domains and new_domains and not set(prev.domains) & set(new_domains):
        logger.info("P_t 换域 %s → %s，词表清空、预算保留", prev.domains, new_domains)
        ex, av, pf = [], [], []
    else:
        ex, av, pf = list(prev.exclude_terms), list(prev.avoid_terms), list(prev.prefer_terms)

    utt_low = (user_utterance or "").lower()
    verified: list[str] = []
    for w in _atoms(retract_terms):
        if any(term_hits(t, utt_low) for t in normalize_terms([w])):
            verified.append(w)
        else:
            logger.warning("撤回核验失败：「%s」不在本轮原话里，不删", w)
    if verified:
        ex, av, pf = _without(ex, verified), _without(av, verified), _without(pf, verified)

    ex_new, av_new, pf_new = _atoms(exclude), _atoms(avoid), _atoms(prefer)
    ex = _atoms([*_without(ex, av_new + pf_new), *ex_new])
    av = _atoms([*_without(av, ex_new + pf_new), *av_new])
    pf = _atoms([*_without(pf, ex_new + av_new), *pf_new])

    if clear_budget:
        budget = None
    elif budget_usd is not None:
        budget = budget_usd
    else:
        budget = prev.budget_usd
    return SessionPrefState(
        category=category or prev.category,
        domains=new_domains or list(prev.domains),
        budget_usd=budget,
        dest_country=(dest_country or prev.dest_country).strip().upper(),
        exclude_terms=ex,
        avoid_terms=av,
        prefer_terms=pf,
        updated_at=prev.updated_at,
    )
