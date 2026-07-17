"""会话级短期偏好状态 P_t —— 单次选购会话内累积的结构化约束。

对齐 RecBot 论文（arXiv:2509.21317）的 ``P_t``：把**一次浏览会话内**用户逐轮下的自然语言命令,
映射成一个**会话内稳定累积、逐轮 merge** 的结构化偏好状态 ``{P⁺,P⁻}×{hard,soft}``。它是本项目
「三层记忆」里此前缺失的中间层——既不是跑完即弃的 messages 上下文,也不是跨会话的长期结论库。

与另两层的分界（务必别混）:
- **长期库**（:mod:`app.memory.store` 的 ``PreferenceEntry``）:跨会话的**一贯取向**（「我一直不要
  塑料」）,按用户聚合、语义去重、带半衰期。
- **本模块 P_t**:**本次选购会话内**的约束（「这次预算 300」「今天要蓝色」）,按 thread 隔离、随轮
  累积、**会话结束即止**（不进长期库）。唯一写者是 planner（见 ``planner._sync_session_pt``），
  curator 只读（只判长期偏好）。
- **行为历史**（``HistoryEntry``）:上次搜/买了什么的事实快照,与偏好正交。

**单写者 + id 增量**（bundle 槽位 id 化 3541d2a 的同款纪律）：约束的**跨轮身份**由代码发号
（c1/c2/…，:attr:`SessionPrefState.next_id`），LLM 只输出**增量**——本轮新说的约束 + 要撤回的
id（从渲染文本里抄）。合并（换代 / 撤回核验 / 归并 / 追加发号）全部是 :func:`merge_pt` 里的
确定性代码。四条不变量由此获得代码保证而非 prompt 祈祷：
- I1 存续：说过且未撤回的约束一直生效（存续不过 LLM 的手——旧约束从不要求模型重放）。
- I2 撤回：改口能精确删掉那一条（引用代码发的 id + 词面核验双闸）。
- I3 过期：不跨意图（``epoch`` 换代）、不跨会话（TTL）。
- I4 失效方向：任何单点失败宁松勿紧、可见可纠（核验不过=不删、撤回失败=约束仍可见）。

**为什么 P_t 单独落 pt.json（而非只靠 messages 重捞）:** 续聊靠 ``load_prior_turns`` 回喂精简文本对,
第 2 轮要靠 LLM 从对话历史里重新捞出「不要塑料」,依赖每轮重解析、可能漏。给本轮约束一个结构化持久
对象,让它在会话内稳定累积、注入时可靠,又能在会话结束（TTL）自然清理、不污染长期库。

**容错口径**（与 :mod:`app.memory.history` / :mod:`app.memory.store` 同一套「降级不崩」）:pt.json
是会话附带产物,读/写任一环出问题都只记日志降级——P_t 退化为空（等价「本轮从零累积」）,不反噬主链路。
旧格式 pt.json（id 化之前的 slug 版）同走此路：``extra="forbid"`` 使其 ValidationError → 按空开局。
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.memory.store import PrefCategory
from app.utils.env import env_int
from app.utils.terms import normalize_terms, term_hits

logger = logging.getLogger("shoppingx.session_state")

# pt.json 过期阈值（小时）:无真实「会话结束」信号,靠 TTL + thread_id 天然隔离兜底。超过则 load 当空
# 开局（同一 thread 隔很久再来,视为新一段选购,旧约束不该复活）。可经 env 覆盖。
SESSION_PT_TTL_HOURS = env_int("SESSION_PT_TTL_HOURS", 24)

Polarity = Literal["like", "dislike"]


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def constraint_verb(polarity: str, blocking: bool) -> str:
    """约束 content 的动词前缀约定（planner 生成与本模块渲染共用，别两处各写一份漂移开）。"""
    if polarity == "dislike":
        return "不要" if blocking else "尽量避免"
    return "偏好"


class SessionConstraint(BaseModel):
    """P_t 里的一条会话内偏好约束（本轮有效、跨会话不保留）。

    **跨轮身份是 ``id``（代码发号 c1/c2/…），不是语义键。** 曾经用 LLM 产的 slug 派生 dedup_key
    当身份，身份随 LLM 每轮的措辞漂移（blue/蓝色/color_blue），upsert 去重形同虚设。现在 LLM
    只在**提议撤回**时引用 id（从渲染文本里抄），发号、归并、核验全在 :func:`merge_pt`。

    **``blocking`` 区分硬淘汰 / 软减分，但它记的是「用户自己的语气」，不是 LLM 的置信度。**
    「不要塑料」→ ``blocking=True``（命中即淘汰）；「尽量别太花哨」→ ``blocking=False``（只减分，
    仍留在候选里）。这两件事必须分开：
    - **识别**用户说的是「绝对不要」还是「尽量别」—— LLM 该做，那是用户自己的措辞。
    - **授权**一条约束有没有杀掉商品的权力 —— LLM 不该做，由来源定：用户说的可以硬，模型
      **推断**出来的一贯取向只能软。

    软档砍不掉的根因在**数据**：商品没有材质 / 风格字段，约束最终都落成「拿关键词匹标题」。
    对匹配不可靠的词（「花哨」「小众」）做硬淘汰就是拿误杀去赌。默认 ``True``：P_t 装的是用户
    本轮亲口说的话，不明确标软就按硬走。

    ``source_quote`` 是用户原话锚（≤60 字）：撤回核验的词面依据之一、偏好面板给用户看
    「这是你哪句话」、排障时定位来源——一举三得，代价一个短字符串。
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default="", description="跨轮身份，代码发号（c1/c2/…）；进 merge_pt 前为空")
    content: str = Field(description="约束内容，如「要蓝色」「不要塑料、plastic」")
    polarity: Polarity = Field(default="like")
    keywords: list[str] = Field(default_factory=list, description="可用于过滤的原子词")
    category: PrefCategory = Field(default="other", description="归类，如 color / material / style")
    source_quote: str = Field(default="", description="用户原话锚（≤60 字），供撤回核验/面板/排障")
    turn_added: int = Field(default=0, description="首次落进 P_t 的轮号（供排序 / 调试）")
    blocking: bool = Field(
        default=True,
        description=(
            "命中即淘汰（True）还是仅减分（False）。**只对 dislike 有意义**——正向从不做二值淘汰。"
        ),
    )

    def norm_keywords(self) -> list[str]:
        """本条约束的归一化词面（原子词 + 中→英扩词，小写）——归并 / 撤回核验的比对基准。"""
        return normalize_terms(self.keywords or [self.content])


class SessionPrefState(BaseModel):
    """一次选购会话的短期偏好状态 P_t（按 thread 隔离，落 pt.json）。

    ``budget_usd`` / ``category`` 是会话内结构化锚点（数值预算单列,供渲染「预算 ≤ $X」;下游据此收紧
    检索）;``constraints`` 是 like/dislike 约束集——**只存当前代（epoch）的 active 约束**，因此
    ``dislike_terms()`` 等消费接口对换代无感。``turn`` 记已累积轮数,``updated_at`` 供 TTL 判。

    - ``epoch``：意图代际。planner 判 ``retrieval="search"``（换品类 / 换意图）时 +1，上一代约束
      整体挪进 ``archived``——I3「约束不跨意图」的机制落点。``reuse``/``augment`` 是同一意图的
      收紧 / 放宽，不换代。
    - ``next_id``：约束 id 发号器（c1/c2/…，跨 epoch 单调递增，id 永不复用）。
    - ``archived``：上一代约束（**只留一代**，供排障不供消费——没有任何 terms 接口读它）。
    - ``current_intent``：本轮/近期最新的一句话意图摘要，纯 prompt 可读性用途，不被机制消费。
    - ``slots``：单值客观事实（如鞋码 / 收货国），覆盖式 patch——和 ``constraints``（梯度式取向）
      分开存：前者是二值匹配，后者要按 polarity 路由到 item_picker 的淘汰 / 加分。

    **open_questions 已删**：待澄清问题的生命周期本就归 ask_user 的 Future 桥接管（问了没答完
    loop 根本不往下走），P_t 里那份从 curator 退出写权后已无人维护，删掉胜过留一个假装有人管的
    字段。**rejected_options / decisions_made 此前已删**（淘汰流水账是排查日志，不是记忆）。
    """

    model_config = ConfigDict(extra="forbid")

    current_intent: str = Field(default="", description="一句话意图摘要，如「买一双跑步鞋」")
    budget_usd: float | None = Field(default=None, description="本轮累积的预算上限 USD，无则 None")
    category: str = Field(default="", description="本轮主品类（最近一次明确的）")
    slots: dict[str, str] = Field(
        default_factory=dict, description="单值客观事实槽位（如 size），按 key 覆盖式 patch"
    )
    constraints: list[SessionConstraint] = Field(
        default_factory=list, description="当前代 active 约束（上一代在 archived）"
    )
    epoch: int = Field(default=0, description="意图代际；search 换代 +1")
    next_id: int = Field(default=1, description="约束 id 发号器（c{next_id}），跨 epoch 单调")
    archived: list[SessionConstraint] = Field(
        default_factory=list, description="上一代约束（只留一代，排障用，不进任何消费接口）"
    )
    turn: int = Field(default=0, description="已累积轮数")
    updated_at: str = Field(default="", description="最近更新时间（UTC ISO），供 TTL 判过期")

    def is_empty(self) -> bool:
        return (
            not self.current_intent
            and self.budget_usd is None
            and not self.category
            and not self.slots
            and not self.constraints
        )

    def dislike_terms(self) -> list[str]:
        """本会话内**硬** dislike 的原子词（小写、保序去重）→ item_picker 命中即淘汰。

        只收 ``blocking=True``（用户说的是「不要 / 不能」）。「尽量别太花哨」这类弱表达走
        :meth:`soft_dislike_terms` 减分——见 :class:`SessionConstraint` 关于「识别 vs 授权」的说明。
        """
        return self._terms("dislike", blocking=True)

    def soft_dislike_terms(self) -> list[str]:
        """本会话内**软** dislike 的原子词 → item_picker 命中减分、不淘汰。

        用户本轮说的弱表达（「尽量别」「不太喜欢」）。硬淘汰它们会拿误杀去赌——「花哨」这种词
        压根匹不到商品标题，而一旦匹上（如「塑料感」连坐 plastic），杀掉的可能正是用户要的。
        """
        return self._terms("dislike", blocking=False)

    def like_terms(self) -> list[str]:
        """本会话内 like 约束的原子词（小写、保序去重）→ item_picker 加分。

        正向约束**不做二值淘汰**（数据没有可靠的材质 / 风格字段，keep-only 会误杀一大片），
        所以即便是本轮明说的「必须金属」，也只作强加分而非硬性保留——故不分 blocking。
        """
        return self._terms("like")

    def _terms(self, polarity: Polarity, *, blocking: bool | None = None) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for c in self.constraints:
            if c.polarity != polarity:
                continue
            if blocking is not None and c.blocking != blocking:
                continue
            for t in c.keywords or [c.content]:
                low = t.strip().lower()
                if low and low not in seen:
                    seen.add(low)
                    out.append(low)
        return out

    def render(self) -> str:
        """渲染成拼进当轮 human message 尾部的一段文本（供主 loop 把已累积约束折进本轮处理）。

        **不注入 system prompt**：P_t 每轮必变，混进 system prompt 会打断 prompt cache 前缀（见
        ``app.agent.main_agent._inject_runtime_context``）。空状态返回占位，让模型明确「本会话
        尚无累积约束」。**不展示约束 id**——id 是 planner 撤回通道的引用键（见 planner 侧的
        带 id 渲染），主 loop 的可读性渲染混进 c1/c2 只会诱导模型把内部主键说给用户听。
        """
        if self.is_empty():
            return "（本会话尚无累积约束）"
        lines: list[str] = []
        if self.current_intent:
            lines.append(f"- 当前意图：{self.current_intent}")
        if self.budget_usd is not None:
            lines.append(f"- 预算：≤ ${self.budget_usd:.0f}")
        if self.category:
            lines.append(f"- 品类：{self.category}")
        for k, v in self.slots.items():
            lines.append(f"- {k}：{v}")
        for c in self.constraints:
            # **标签必须带上 blocking**，不能只按 polarity 分「排除 / 需要」两档。续聊轮 planner
            # 是照着这段文字重新识别约束的：软避讳一旦被标成「排除」，它就会原样填进
            # exclude_keywords，于是「尽量别太花哨」在第二轮变成硬淘汰——档位在渲染出口丢失，
            # 和 curator 曾经用无 blocking 的 draft 覆盖 planner 是同一个 bug 的两种长法。
            if c.polarity == "dislike":
                tag = "硬排除，命中即淘汰" if c.blocking else "软避讳，只减分不淘汰"
            else:
                tag = "偏好，命中加分"
            lines.append(f"- 本轮约束（{tag}）：{c.content}")
        return "\n".join(lines)


def _pt_path(session_dir: Path) -> Path:
    return session_dir / "pt.json"


def _is_expired(updated_at: str) -> bool:
    """按 ``updated_at`` 判 pt.json 是否过期（超过 TTL 视为过期，load 当空开局）。

    时间戳缺失 / 解析失败时**不判过期**（宁可留着旧约束,也不因坏时间戳误清）。TTL≤0 视为永不过期。
    """
    if SESSION_PT_TTL_HOURS <= 0 or not updated_at:
        return False
    try:
        ref = datetime.fromisoformat(updated_at)
    except (ValueError, TypeError):
        return False
    if ref.tzinfo is None:
        # naive 时间戳（手改/旧数据）：按 UTC 补齐时区，否则 aware-naive 减法抛 TypeError，
        # 而本函数在 load_pt 的 try 之外——「读不出当空」的容错承诺会被炸穿。
        ref = ref.replace(tzinfo=UTC)
    age_hours = (datetime.now(UTC) - ref).total_seconds() / 3600
    return age_hours > SESSION_PT_TTL_HOURS


def load_pt(session_dir: Path) -> SessionPrefState:
    """读出本会话的 P_t;缺失 / 损坏 / 过期一律返回空状态（记日志，不抛）。

    与 :func:`app.memory.history.load_prior_turns` 同一套容错:pt.json 是会话附带产物,读不出就当
    「本轮从零累积」,绝不反噬主链路。过期（超 :data:`SESSION_PT_TTL_HOURS`）也当空——旧约束不复活。
    **旧格式（id 化之前的 slug 版）同走此路**：``extra="forbid"`` 让旧字段触发 ValidationError
    → 按空开局，不写迁移代码（P_t 生命周期只有 24h，迁移代码服务不了任何真实存量）。
    """
    path = _pt_path(session_dir)
    if not path.exists():
        return SessionPrefState()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        state = SessionPrefState.model_validate(raw)
    except (OSError, json.JSONDecodeError, ValidationError) as exc:
        logger.warning("P_t 无法读取/解析，按空处理（path=%s）：%s", path, exc)
        return SessionPrefState()
    if _is_expired(state.updated_at):
        logger.info("P_t 已过期（updated_at=%s），按空开局（path=%s）", state.updated_at, path)
        return SessionPrefState()
    return state


def save_pt(session_dir: Path, state: SessionPrefState) -> None:
    """把 P_t 落到 pt.json（覆盖式，刷新 updated_at）。写失败只记日志降级，不拖垮主链路。"""
    state.updated_at = _now_iso()
    path = _pt_path(session_dir)
    try:
        path.write_text(
            json.dumps(state.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except OSError:
        logger.warning("写 P_t 失败（path=%s），降级跳过", path, exc_info=True)


def _overlaps(new_norm: set[str], old: SessionConstraint) -> bool:
    """新旧约束是否词面重叠（任一方向命中即算）——归并「同一件事的重新表达」的判据。

    双向查：新词面进旧词面（「花哨」升级轮命中旧软避讳），或旧词面进新词面。用
    :func:`normalize_terms` 的中→英扩词把「塑料 vs plastic」这类跨语言同义对上。
    """
    old_norm = set(old.norm_keywords())
    if new_norm & old_norm:
        return True
    old_text = " ".join(old_norm)
    new_text = " ".join(new_norm)
    return any(term_hits(w, old_text) for w in new_norm) or any(
        term_hits(w, new_text) for w in old_norm
    )


def merge_pt(
    prev: SessionPrefState,
    new_constraints: list[SessionConstraint] | None = None,
    *,
    retract_ids: list[str] | None = None,
    user_utterance: str = "",
    retrieval: str = "",
    budget_usd: float | None = None,
    clear_budget: bool = False,
    category: str = "",
    current_intent: str | None = None,
    slots_patch: dict[str, str] | None = None,
    turn: int | None = None,
) -> SessionPrefState:
    """把本轮增量 merge 进 P_t——**单一语义，顺序固定，全部确定性代码**（唯一调用方 planner）。

    1. **换代**：``retrieval == "search"`` 且有既有约束 → ``epoch+1``，旧代整体挪 ``archived``
       （只留一代）。``reuse``/``augment`` 是同一意图的收紧 / 放宽，不换代。
    2. **撤回核验**：``retract_ids`` 逐条——id 必须存在于 active 集，**且**该约束词面与本轮
       原话呼应（:func:`term_hits`，禁精确相等比较）才删；核验不过记 warning 不删（I4：撤回
       失败但约束可见，用户重说可纠；幻觉 id / 抄错 id 都被这道闸挡住）。**整条删**：一条约束
       可能装着同桶多个概念（「不要塑料、尼龙」），只撤其一会连坐另一个——失效方向是「多放出
       商品」（宁松），好过按词局部剪除时英文同义词残留继续暗中过滤（宁紧，不可见）。
    3. **归并**：新约束与既有 active 词面重叠 → 新条**继承旧 id** 顶掉旧条——兜住「改口升级
       只 add 不 retract」的半拉子操作（软升硬后 P_t 只剩一条硬的，极性翻转同理）。
    4. **追加**：真正的新约束发新号（``c{next_id}``，跨 epoch 单调、永不复用）。
    5. 预算/slots/category/current_intent：None/空 = 本轮未提及保持不变。``clear_budget`` 单列
       ——``budget_usd=None`` 的语义已被「没提」占用，「明确放开」必须可表达，否则 item_picker
       的「无预算用 P_t 兜底」会让旧预算每轮暗中卡人。
    """
    if retrieval == "search" and prev.constraints:
        epoch = prev.epoch + 1
        archived = list(prev.constraints)  # 只留一代：上上代直接丢（排障看最近一次换代足够）
        active = []
        logger.info("P_t 换代 epoch=%d，归档 %d 条上代约束", epoch, len(archived))
    else:
        epoch = prev.epoch
        archived = list(prev.archived)
        active = list(prev.constraints)

    utt_low = (user_utterance or "").lower()
    for rid in retract_ids or []:
        target = next((c for c in active if c.id == rid), None)
        if target is None:
            logger.warning("撤回核验失败：id=%s 不在 active 集（幻觉/抄错/已换代），不删", rid)
            continue
        if not any(term_hits(kw, utt_low) for kw in target.norm_keywords()):
            logger.warning(
                "撤回核验失败：id=%s 词面与本轮原话无呼应（%r），不删", rid, target.content
            )
            continue
        active.remove(target)
        logger.info("撤回 id=%s（%s）", rid, target.content)

    next_id = prev.next_id
    for nc in new_constraints or []:
        new_norm = set(nc.norm_keywords())
        hit = next((c for c in active if _overlaps(new_norm, c)), None)
        if hit is not None:
            # 同一件事的重新表达（升级/改口/重复）：继承身份顶掉旧条，最新表达为准。
            active[active.index(hit)] = nc.model_copy(update={"id": hit.id})
        else:
            active.append(nc.model_copy(update={"id": f"c{next_id}"}))
            next_id += 1

    if clear_budget:
        merged_budget = None
    elif budget_usd is not None:
        merged_budget = budget_usd
    else:
        merged_budget = prev.budget_usd

    return SessionPrefState(
        current_intent=current_intent or prev.current_intent,
        budget_usd=merged_budget,
        category=category or prev.category,
        slots={**prev.slots, **(slots_patch or {})},
        constraints=active,
        epoch=epoch,
        next_id=next_id,
        archived=archived,
        turn=turn if turn is not None else prev.turn + 1,
        updated_at=prev.updated_at,
    )
