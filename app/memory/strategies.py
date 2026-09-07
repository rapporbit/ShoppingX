"""成功策略库（18-4）：把「这类局面该怎么打」从高分轨迹里沉下来，再注回 system prompt。

**与 :mod:`app.memory.curator` 并列，互不混用。** 两者都叫「记忆」，学的却是不同的东西：

| | curator | 本模块 |
|---|---|---|
| 学什么 | **用户**的一贯取向（不要皮革） | **Agent** 的打法（预算陷阱要先算到手价再排序） |
| 归谁 | 某个 user_id | 全局，没有 user_id 这一列 |
| 来源 | 每轮对话结束后的 fast 档 LLM | 离线蒸馏（``scripts/eval/distill_strategies.py``） |
| 进哪 | ``<user_long_term_preferences>``（planner 后） | ``<learned_strategies>``（system 段） |
| 怎么退场 | 用户改 / ``forget_preference`` 删 | 连续失败自动淘汰（本模块） |

两条路径共用一个 Store 类是很自然的下意识做法，但那会立刻要求回答「A 用户把这条策略跑挂了，
B 用户那份算不算数」——一个不该存在的问题。所以是两张表、两个 Store、零共享状态。

**策略是假设，不是结论。** 蒸馏的素材是 Rubric 高分轨迹，而 judge 有单样本 0↔100 对翻的抖动
（记忆 rubric-judge-calibration-pitfalls）。所以：① 写入前必须过门禁重放（3 条同类 query 不
退化，见蒸馏脚本）；② 写进来之后仍带血量——命中回血、连续失败淘汰，让不灵的策略自己退场，
而不是靠人定期回来清一张只增不减的表。

**注入代价说清楚**：策略块拼在 system prompt **末尾**，所以它前面那段（role / workflow /
tool_policy / …）逐字不变，隐式前缀缓存照常从头命中，只有尾巴这一小段随命中的策略变。这是
本仓 formatter 里「缓存收益来自前缀字节稳定」那条判断的直接推论（见 harness/formatter.py）：
只要新增内容在最后，就不会把前面的前缀推歪。
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from functools import lru_cache
from uuid import uuid4

from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError

from app.db.models import StrategyRow
from app.db.session import session_factory
from app.utils.env import env_int
from app.utils.terms import normalize_terms, term_hits

logger = logging.getLogger("shoppingx.memory.strategies")

#: 血量上限。3 = 「连挂三轮才退休」，与 :data:`RETIRE_AFTER_FAILURES` 同源：一条刚过门禁的
#: 策略不该被一次偶发失败（超时 / 网关 5xx）判死，也不该在明显不灵之后还赖着不走。
MAX_HEALTH = env_int("STRATEGY_MAX_HEALTH", 3)
#: 连续失败几次淘汰。**连续**是关键——中间只要成功过一次就清零，因为「偶尔挂」是本仓常态
#: （限流退避、judge 抖动），只有连着挂才是策略本身的问题。
RETIRE_AFTER_FAILURES = env_int("STRATEGY_RETIRE_AFTER_FAILURES", 3)
#: 一轮最多注入几条。策略块进的是 system prompt，条数无上限就等于把 prompt 越写越长，
#: 而模型对长 prompt 中段的指令最不敏感——注入 3 条强指令好过注入 12 条弱指令。
MAX_INJECTED = env_int("STRATEGY_MAX_INJECTED", 3)
#: 触发词至少命中几个才算这条策略适用。1 = 宁可多注入不误漏（策略是建议不是硬闸，
#: 多注入一条的代价只是几十 token；漏注入则等于这套机制白做）。
MIN_TRIGGER_HITS = env_int("STRATEGY_MIN_TRIGGER_HITS", 1)

_SLUG_RE = re.compile(r"[^a-z0-9_]+")


def _now() -> datetime:
    return datetime.now(UTC)


def make_slug(text: str) -> str:
    """把一句话压成可做去重身份的原子标识（小写 ASCII + 下划线，截断 48）。

    LLM 给的 slug 常带空格 / 中文 / 驼峰，直接进 ``dedup_key`` 会让「同一条策略」在库里
    出现好几份变体。归一放在**写入口**而不是靠提示词求 LLM 守规矩。
    """
    slug = _SLUG_RE.sub("_", (text or "").strip().lower()).strip("_")
    return slug[:48] or "unnamed"


class Strategy(BaseModel):
    """一条成功策略：``{trigger, actions, category, evidence}`` + 生命周期。

    ``trigger`` 是给模型读的整句（「用户给了预算且品类里有大量低价配件」），``trigger_keywords``
    才是**确定性匹配**的抓手——用 :func:`app.utils.terms.term_hits` 逐词判，不动 LLM。匹配环节
    绝不能再塞一次模型调用：那等于每轮多一次往返、多一处漂移源，而它要做的只是「这句话里有没有
    出现预算」这种查表活。
    """

    category: str = Field(default="", description="场景类别，对齐种子集 bucket（预算陷阱 / 假货…）")
    slug: str = Field(default="", description="原子标识（英文小写），与 category 一起构成去重身份")
    trigger: str = Field(description="什么局面下适用（一句话，给模型读）")
    trigger_keywords: list[str] = Field(default_factory=list, description="确定性匹配用的原子词")
    actions: list[str] = Field(default_factory=list, description="该怎么做（2~4 条祈使句）")
    evidence: list[str] = Field(default_factory=list, description="出处：报告 + query id + 分数")

    status: str = Field(default="active", description="active / retired")
    health: int = Field(default=MAX_HEALTH)
    hits: int = Field(default=0, description="被注入的累计次数")
    consecutive_failures: int = Field(default=0)
    source_report: str = Field(default="")
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)

    @property
    def dedup_key(self) -> str:
        """去重身份：由 ``category`` + ``slug`` **派生**，与 ``PreferenceEntry`` 同口径。"""
        return f"{self.category}:{self.slug or make_slug(self.trigger)}"

    def render(self) -> str:
        """注入 system prompt 的那一条的文本形态（越短越好，模型只需要能照做）。"""
        acts = "\n".join(f"  - {a}" for a in self.actions)
        return f"- 当【{self.trigger}】时：\n{acts}"

    @classmethod
    def from_row(cls, row: StrategyRow) -> Strategy:
        return cls(
            category=row.category,
            slug=row.slug,
            trigger=row.trigger,
            trigger_keywords=list(row.trigger_keywords or []),
            actions=list(row.actions or []),
            evidence=list(row.evidence or []),
            status=row.status,
            health=row.health,
            hits=row.hits,
            consecutive_failures=row.consecutive_failures,
            source_report=row.source_report,
            created_at=row.created_at,
            updated_at=row.updated_at,
        )


class StrategyStore:
    """策略库的读写口。后端与偏好共用 :mod:`app.db` 的 SQLite，但**表和实例都不共用**。

    容错口径与 :class:`app.memory.store.PreferenceStore` 逐条一致：**读不出来就返回空**。
    策略是增强不是依赖——库挂了这一轮就当没有策略跑，绝不让它演变成「这次任务失败」。
    """

    async def read_active(self) -> list[Strategy]:
        """读全部在役策略（``status="active"``）。库故障返回空。"""
        try:
            async with session_factory()() as db:
                rows = (
                    await db.execute(select(StrategyRow).where(StrategyRow.status == "active"))
                ).scalars()
                return [Strategy.from_row(r) for r in rows]
        except SQLAlchemyError as exc:
            logger.warning("读取策略失败，本轮降级为无策略：%s", exc)
            return []

    async def read_all(self) -> list[Strategy]:
        """读全部（含退休的），给 CLI / 审计看。"""
        try:
            async with session_factory()() as db:
                rows = (await db.execute(select(StrategyRow))).scalars()
                return [Strategy.from_row(r) for r in rows]
        except SQLAlchemyError as exc:
            logger.warning("读取策略失败：%s", exc)
            return []

    async def upsert(self, strategy: Strategy) -> None:
        """写一条策略；同 ``dedup_key`` 覆盖内容并**复活**（血量回满、失败计数清零）。

        为什么重新蒸馏出同一条就该复活：它刚刚又过了一次门禁重放。上一轮把它淘汰掉的那串失败
        可能来自别的原因（那阵子的 prompt / 索引 / 网关），拿旧账压住一条刚被证明过的策略，
        等于让这套机制只会单向减少。
        """
        try:
            async with session_factory()() as db:
                row = (
                    await db.execute(
                        select(StrategyRow).where(StrategyRow.dedup_key == strategy.dedup_key)
                    )
                ).scalar_one_or_none()
                if row is None:
                    db.add(
                        StrategyRow(
                            id=uuid4().hex,
                            dedup_key=strategy.dedup_key,
                            category=strategy.category,
                            slug=strategy.slug or make_slug(strategy.trigger),
                            trigger=strategy.trigger,
                            trigger_keywords=list(strategy.trigger_keywords),
                            actions=list(strategy.actions),
                            evidence=list(strategy.evidence),
                            status="active",
                            health=MAX_HEALTH,
                            hits=0,
                            consecutive_failures=0,
                            source_report=strategy.source_report,
                        )
                    )
                else:
                    row.trigger = strategy.trigger
                    row.trigger_keywords = list(strategy.trigger_keywords)
                    row.actions = list(strategy.actions)
                    row.evidence = list(strategy.evidence)
                    row.source_report = strategy.source_report
                    row.status = "active"
                    row.health = MAX_HEALTH
                    row.consecutive_failures = 0
                await db.commit()
        except SQLAlchemyError as exc:
            logger.warning("写入策略失败（key=%s）：%s", strategy.dedup_key, exc)

    async def record_outcome(self, dedup_keys: Sequence[str], *, success: bool) -> list[str]:
        """本轮注入过的策略结账：命中回血 / 连续失败淘汰。返回**本次被淘汰**的 key。

        ``hits`` 记的是「被注入过几次」（成功失败都算），血量只随成败动——两者分开，才答得出
        「这条策略上过 40 次场、掉了 3 次血」这种问题。
        """
        retired: list[str] = []
        if not dedup_keys:
            return retired
        try:
            async with session_factory()() as db:
                rows = (
                    await db.execute(
                        select(StrategyRow).where(StrategyRow.dedup_key.in_(list(dedup_keys)))
                    )
                ).scalars()
                for row in rows:
                    row.hits += 1
                    if success:
                        row.consecutive_failures = 0
                        row.health = min(MAX_HEALTH, row.health + 1)
                        continue
                    row.consecutive_failures += 1
                    row.health -= 1
                    if row.consecutive_failures >= RETIRE_AFTER_FAILURES or row.health <= 0:
                        row.status = "retired"
                        retired.append(row.dedup_key)
                await db.commit()
        except SQLAlchemyError as exc:
            logger.warning("策略结账失败（keys=%s）：%s", list(dedup_keys), exc)
            return []
        if retired:
            logger.warning("策略连续失败达阈值，已淘汰：%s", ", ".join(retired))
        return retired


@lru_cache(maxsize=1)
def get_strategy_store() -> StrategyStore:
    """进程内共享的策略库（无状态，单例只为省对象）。"""
    return StrategyStore()


def match_strategies(
    query: str, strategies: Sequence[Strategy], *, limit: int = MAX_INJECTED
) -> list[Strategy]:
    """按用户原话挑出适用的策略（确定性匹配，零 LLM）。

    命中判定复用 :func:`app.utils.terms.term_hits` + :func:`normalize_terms`——不是裸 ``in``：
    ``bag`` 不该命中 ``baggage``，「不要皮革」里的 leather 不该算成用户想要皮革。这两个函数是
    本仓匹配层的单一事实源，另起一套子串匹配正是记忆 memory-bugs-are-silent-inversions 记的
    那类静默反转的温床。

    排序：命中词数多的优先，同分按血量高的优先（活得好的策略更可信），再同分按 trigger 稳定
    排序——**必须有最后这一档**，否则同分策略的注入顺序随库的返回顺序变，system prompt 的尾巴
    就每轮都在抖，前缀缓存白搭。
    """
    text = (query or "").lower()
    if not text:
        return []
    scored: list[tuple[int, int, str, Strategy]] = []
    for s in strategies:
        if s.status != "active":
            continue
        terms = normalize_terms(s.trigger_keywords)
        hits = sum(1 for t in terms if term_hits(t, text))
        if hits >= MIN_TRIGGER_HITS:
            scored.append((-hits, -s.health, s.trigger, s))
    scored.sort(key=lambda t: (t[0], t[1], t[2]))
    return [s for _, _, _, s in scored[:limit]]


def render_strategy_block(strategies: Sequence[Strategy]) -> str:
    """把选中的策略渲染成 system prompt 尾部那一段。空列表返回空串（不塞空占位）。"""
    if not strategies:
        return ""
    body = "\n".join(s.render() for s in strategies)
    return (
        "<learned_strategies>\n"
        "以下是从**历史高分轨迹**里沉淀下来的打法，已通过重放门禁。它们是经验性建议，优先级\n"
        "**低于** <constraints> 里的硬约束——两者冲突时以硬约束为准，别为了套用策略去绕规则。\n"
        f"{body}\n"
        "</learned_strategies>"
    )


#: 门禁重放专用：强制注入这批策略（还没进库，正在被验证）。**只有蒸馏脚本会设**——线上路径
#: 永远读库。用 ContextVar 而不是模块级变量，是为了与本仓其它会话态一致：并发跑多条重放时
#: 各自的 ContextVar 互不串台（模块级 dict 串台是 demo 踩过的坑，见手册 §12 的「不要照抄」）。
_forced: ContextVar[tuple[Strategy, ...] | None] = ContextVar("forced_strategies", default=None)


@contextmanager
def force_strategies(strategies: Sequence[Strategy]) -> Iterator[None]:
    """在这个作用域内，注入位只看这批策略、不读库（门禁重放用）。"""
    token = _forced.set(tuple(strategies))
    try:
        yield
    finally:
        _forced.reset(token)


async def strategies_for_query(query: str) -> list[Strategy]:
    """注入位唯一入口：拿本轮该注入的策略。强制态优先，否则读库在役集合再匹配。"""
    forced = _forced.get()
    pool = list(forced) if forced is not None else await get_strategy_store().read_active()
    return match_strategies(query, pool)
