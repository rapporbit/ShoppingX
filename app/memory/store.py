"""长期记忆 Store —— 跨会话持久化的用户偏好 / 历史 / 收藏。

对齐 refdocs/06 的核心区分：**长上下文 ≠ 长期记忆**。长上下文（消息历史）按 token 涨钱、只在单
会话有效、随轮数膨胀；长期记忆按条目持久化、跨会话共享、只在「检测到新偏好」时写入。有了 Store
保底（「不要塑料」已落库），上下文才能放心压缩——即使丢掉那条历史消息，下次新会话仍会重新注入。

**Mmem 重构：三件事变了。**

1. **落地介质从「JSON 文件 / Redis 双后端」收成 SQLite**（复用 M16/M17 已有的 ``app.db``）。
   原来那两个后端各有一处并发写隐患：文件后端覆盖式 ``write_text``（多 worker 会丢写）、Redis
   后端 ``hget`` + ``hset`` 非原子（并发写同一条会丢更新——而 Redis 存在的理由恰恰是「可多实例
   共享」，它在自己的目标场景里是不安全的）。关系库的事务一并管掉，还顺手把「收藏超 200 条裁最旧」
   这类操作从「读全量→算 overflow→回写」压成一条 SQL。

2. **``strength`` 字段删除，改由 ``blocking`` 表达杀伤力**——见 :class:`PreferenceEntry`。

3. **时间衰减（``recency_weight``）整个删除**，连带删掉 ``read_relevant`` / ``rank_relevant``
   语义排序，本模块因此**不再依赖 ``app.recall.towers``**。理由见 :func:`build_preference_block`
   一侧的注释：域隔离（``PrefDomain``）已经把「本轮相关的偏好」压到个位数，叠在上面的语义 top-k
   是为「几十上百条偏好」准备的第二层解法，纯属冗余。

**容错口径（降级不崩）：** 记忆是**增强**而非**依赖**。库读不出来只记日志、返回空——「这一轮没有
历史偏好」，绝不让记忆故障演变成「这次任务失败」。
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from functools import lru_cache
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field
from sqlalchemy import delete, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Favorite, HistoryRecord, Preference
from app.db.session import session_factory
from app.memory.domains import DOMAIN_OTHER, PrefDomain
from app.utils.env import env_int

logger = logging.getLogger("shoppingx.memory")

Polarity = Literal["like", "dislike"]
# 谁写的这条偏好：agent=curator 从对话里学到的；user=用户在偏好页面手填的。
# 它同时决定两件事：页面上的来源徽标，以及 curator 不得覆盖用户手填的内容（见 _apply_merge）。
Source = Literal["agent", "user"]
# 偏好的「维度」（材质 / 颜色 / 品牌…），区别于 PrefDomain 的「品类」（鞋 / 沙发…）。两者正交：
# 一条偏好是「在 footwear 这个**品类**下，关于 material 这个**维度**的取向」。
# location = 常用收货地（「我一般寄到日本」）——不是口味偏好，但同样跨会话稳定，且决定关税免征额。
PrefCategory = Literal["material", "style", "brand", "budget", "color", "size", "location", "other"]

HISTORY_TTL_DAYS = env_int("HISTORY_TTL_DAYS", 30)
# 每种 kind 保留最近几条历史（超出的按 created_at 淘汰最旧）。设为 1 即退回 last-write-wins。
HISTORY_MAX_PER_KIND = env_int("HISTORY_MAX_PER_KIND", 3)
# 收藏上限：超过则丢最旧的。收藏是用户手工攒的清单，不会自动膨胀，上限只是防脚本刷爆。
FAVORITES_MAX = env_int("FAVORITES_MAX", 200)


def _now() -> datetime:
    return datetime.now(UTC)


def _new_id() -> str:
    return uuid4().hex


class PreferenceEntry(BaseModel):
    """用户偏好的一条记录（跨会话持久）。领域对象，与 ORM 行 ``app.db.models.Preference`` 互转。

    由 :mod:`app.memory.curator`（从对话学到）或偏好页面（用户手填）写入，两条路共用
    ``injector.persist_new_preferences`` 这个唯一落库口。

    **去重身份由代码派生，不由 LLM 手拼**：``polarity`` / ``category`` / ``domain`` 已是独立字段，
    再让 LLM 拼一个含它们的复合 key 字符串，等于同一份信息存两遍、还可能自相矛盾（key 说 dislike，
    polarity 字段却填 like，没有校验拦得住）。LLM 只提供真正的原子新信息 ``slug``。

    **``blocking``：杀伤力只由用户授予。** 只有 ``blocking=True`` 的条目会在 item_picker 里硬淘汰
    商品；agent 学到的偏好一律只减分（授权分档见 ``memory.assemble``）。这是 Mmem 的核心决策——
    原来的 ``strength: hard|soft`` 是 curator 的 LLM **猜**出来的，而 hard 意味着永久、跨品类、
    静默的硬淘汰。让一个每轮都在猜的模型决定「这件商品用户永远不该看到」，风险和收益完全不匹配。
    硬软的分界不该是 LLM 的置信度，而该是**信息的来源**：用户明说的可以硬执行，agent 推断的只能软。
    """

    polarity: Polarity = Field(default="like")
    category: PrefCategory = Field(default="other", description="偏好维度：material/style/brand/…")
    domain: PrefDomain = Field(
        default=DOMAIN_OTHER,
        description="品类域：决定这条偏好在哪些轮次生效。global=跨品类底线（安全/过敏/伦理）",
    )
    slug: str = Field(
        default="", description="LLM 给的原子标识（规范化英文，如 leather/brand_nike）"
    )
    content: str = Field(description="偏好内容，如「不接受皮革材质」")
    # 可直接用于硬过滤 / 减分的原子词。content 多是整句、无法做子串匹配；keywords 才是能落地用的。
    keywords: list[str] = Field(default_factory=list)
    source: Source = Field(default="agent", description="agent=curator 学到的 / user=用户手填的")
    blocking: bool = Field(
        default=False, description="绝不推荐（硬淘汰）——仅 source=user 可为 True，见类 docstring"
    )
    source_session: str = Field(default="", description="来源会话 thread_id（可追溯出处）")
    created_at: datetime = Field(default_factory=_now)
    last_confirmed_at: datetime = Field(
        default_factory=_now,
        description="上次被确认的时间。**不参与打分**，只供 UI 显示「多久没用」",
    )

    @property
    def dedup_key(self) -> str:
        """去重键：由结构化字段确定性组装，杜绝 LLM 手拼与字段不一致的风险。"""
        return f"{self.polarity}:{self.category}:{self.domain}:{self.slug}"

    @property
    def is_blocking(self) -> bool:
        """这条偏好是否有权硬淘汰商品。**只有用户显式勾选的才算**——agent 学到的一律只减分。

        双重校验（``source == "user"`` **且** ``blocking``）而非只看 ``blocking``：多一道防线，
        免得将来哪条写入路径忘了守「仅 user 可设 blocking」的约定，就把硬淘汰权泄给了 LLM。
        """
        return self.blocking and self.source == "user"

    @classmethod
    def from_row(cls, row: Preference) -> PreferenceEntry:
        return cls(
            polarity=row.polarity,  # type: ignore[arg-type]
            category=row.category,  # type: ignore[arg-type]
            domain=row.domain,  # type: ignore[arg-type]
            slug=row.slug,
            content=row.content,
            keywords=list(row.keywords or []),
            source=row.source,  # type: ignore[arg-type]
            blocking=row.blocking,
            source_session=row.source_session,
            created_at=row.created_at,
            last_confirmed_at=row.last_confirmed_at,
        )


HistoryKind = Literal["purchase", "search"]


class HistoryEntry(BaseModel):
    """一条行为历史（对齐 refdocs/06 §3.2 的 last_purchase / last_search）。

    与偏好分开存、定位也不同：偏好是「一贯的取向」（去重 + 覆盖合并），历史是「做过什么」的事实
    快照——既不去重也不合并，每种 kind 保留最近 :data:`HISTORY_MAX_PER_KIND` 条 + TTL 过期。
    """

    kind: HistoryKind = Field(description="purchase=购买 / search=搜索")
    content: str = Field(description="行为快照，如「搜了旅行收纳袋」")
    source_session: str = Field(default="")
    created_at: datetime = Field(default_factory=_now)

    @classmethod
    def from_row(cls, row: HistoryRecord) -> HistoryEntry:
        return cls(
            kind=row.kind,  # type: ignore[arg-type]
            content=row.content,
            source_session=row.source_session,
            created_at=row.created_at,
        )


class FavoriteItem(BaseModel):
    """用户收藏（♡）的一件商品。**纯展示数据——不进 prompt、不进偏好库、不影响检索与精挑。**

    收藏一件商品并不能可靠地推出任何偏好（可能只是想再比比价），拿它去改 Agent 行为是过度解读。
    """

    item_id: str = Field(description="商品 id（同 id 视为同一件，重复收藏即覆盖）")
    title: str
    platform: str = ""
    price_usd: float | None = None
    landed_usd: float | None = Field(default=None, description="到手价（含税运），没算过则 None")
    image_url: str = ""
    url: str = Field(default="", description="平台商品页，抽屉里点击跳转")
    created_at: datetime = Field(default_factory=_now)

    @classmethod
    def from_row(cls, row: Favorite) -> FavoriteItem:
        return cls(
            item_id=row.item_id,
            title=row.title,
            platform=row.platform,
            price_usd=row.price_usd,
            landed_usd=row.landed_usd,
            image_url=row.image_url,
            url=row.url,
            created_at=row.created_at,
        )


def _apply_merge(row: Preference, incoming: PreferenceEntry) -> None:
    """``dedup_key`` 撞车时的合并（就地改 ORM 行）：取最新表达，保留最早 ``created_at``。

    ``last_confirmed_at`` 刷新到当前时间——重复提及即「确认这条偏好还活着」。

    **例外：用户手填的条目不被 curator 覆盖**（只刷新确认时间）。否则用户在偏好页面亲手写下的
    「预算严格 300 以内」，会被 Agent 下一轮的推断悄悄改掉——「我明明改过了」是记忆功能最伤信任
    的一种失败。用户自己再次手填（incoming 也是 user）时照常覆盖，那本来就是他要改。
    """
    row.last_confirmed_at = _now()
    if row.source == "user" and incoming.source != "user":
        return
    row.content = incoming.content
    row.keywords = list(incoming.keywords)
    row.source = incoming.source
    row.source_session = incoming.source_session
    # blocking 随 incoming——但写入口已保证只有 user 能置 True（见 persist_new_preferences）。
    row.blocking = incoming.blocking


class PreferenceStore:
    """用户级持久数据的读写口（偏好 / 历史 / 收藏），后端是 :mod:`app.db` 的 SQLite。

    **不再有后端抽象基类**：原来 ABC + LocalFileStore + RedisStore 的三层结构，是为了「离线可跑」
    与「可选真后端」——而 SQLite 两样都占（零外部依赖、库文件躺在持久卷上），一个实现就够了。
    少一层抽象，就少一处「写了不读」的接缝。
    """

    async def read(self, user_id: str) -> list[PreferenceEntry]:
        """读某用户的全部偏好。库故障返回空（降级为「本轮没有历史偏好」），不抛。"""
        if not user_id:
            return []
        try:
            async with session_factory()() as db:
                rows = (
                    await db.execute(select(Preference).where(Preference.user_id == user_id))
                ).scalars()
                return [PreferenceEntry.from_row(r) for r in rows]
        except SQLAlchemyError as exc:
            logger.warning("读取偏好失败，本轮降级为空偏好（user=%s）：%s", user_id, exc)
            return []

    async def write(self, user_id: str, entry: PreferenceEntry) -> None:
        """写一条偏好；相同 ``dedup_key`` 走 :func:`_apply_merge` 覆盖合并，不堆重复条目。"""
        if not user_id:
            return
        try:
            async with session_factory()() as db:
                row = (
                    await db.execute(
                        select(Preference).where(
                            Preference.user_id == user_id,
                            Preference.dedup_key == entry.dedup_key,
                        )
                    )
                ).scalar_one_or_none()
                if row is None:
                    db.add(
                        Preference(
                            id=_new_id(),
                            user_id=user_id,
                            dedup_key=entry.dedup_key,
                            polarity=entry.polarity,
                            category=entry.category,
                            domain=entry.domain,
                            slug=entry.slug,
                            content=entry.content,
                            keywords=list(entry.keywords),
                            source=entry.source,
                            blocking=entry.blocking,
                            source_session=entry.source_session,
                            created_at=entry.created_at,
                            last_confirmed_at=entry.last_confirmed_at,
                        )
                    )
                else:
                    _apply_merge(row, entry)
                await db.commit()
        except SQLAlchemyError as exc:
            # 写失败：这条偏好本轮不落库（下次识别到会再写），但不拖垮收尾链路。
            logger.warning(
                "写入偏好失败，本条未持久化（user=%s，dedup_key=%s）：%s",
                user_id,
                entry.dedup_key,
                exc,
            )

    async def delete(self, user_id: str, dedup_key: str) -> None:
        """删一条偏好（用户主动撤回 / curator 矛盾消解）。不存在则静默无操作。"""
        if not user_id:
            return
        try:
            async with session_factory()() as db:
                await db.execute(
                    delete(Preference).where(
                        Preference.user_id == user_id, Preference.dedup_key == dedup_key
                    )
                )
                await db.commit()
        except SQLAlchemyError as exc:
            logger.warning("删除偏好失败（user=%s，dedup_key=%s）：%s", user_id, dedup_key, exc)

    # ── 行为历史 ────────────────────────────────────────────────────────────────────

    async def read_history(self, user_id: str) -> list[HistoryEntry]:
        """读行为历史：每种 kind 最近 :data:`HISTORY_MAX_PER_KIND` 条，顺带惰性清掉过期的。"""
        if not user_id:
            return []
        cutoff = _now() - timedelta(days=HISTORY_TTL_DAYS)
        try:
            async with session_factory()() as db:
                # 惰性清理：过期条目在读时删掉。**先删再查**，免得刚删的又被查出来。
                await db.execute(
                    delete(HistoryRecord).where(
                        HistoryRecord.user_id == user_id, HistoryRecord.created_at < cutoff
                    )
                )
                await db.commit()
                rows = (
                    await db.execute(
                        select(HistoryRecord)
                        .where(HistoryRecord.user_id == user_id)
                        .order_by(HistoryRecord.created_at.desc())
                    )
                ).scalars()
                # 每种 kind 只留最近 N 条（条数很少，Python 侧分组比写窗口函数简单，且 SQLite
                # 的窗口函数支持要看版本，不值得为这点数据量赌）。
                kept: dict[str, list[HistoryEntry]] = {}
                for r in rows:
                    bucket = kept.setdefault(r.kind, [])
                    if len(bucket) < HISTORY_MAX_PER_KIND:
                        bucket.append(HistoryEntry.from_row(r))
                return [e for bucket in kept.values() for e in bucket]
        except SQLAlchemyError as exc:
            logger.warning("读取历史失败，本轮降级为空（user=%s）：%s", user_id, exc)
            return []

    async def write_history(self, user_id: str, entry: HistoryEntry) -> None:
        """记一条行为历史（纯 append，不去重不合并）。超额条目由 :meth:`read_history` 惰性裁。"""
        if not user_id:
            return
        try:
            async with session_factory()() as db:
                db.add(
                    HistoryRecord(
                        id=_new_id(),
                        user_id=user_id,
                        kind=entry.kind,
                        content=entry.content,
                        source_session=entry.source_session,
                        created_at=entry.created_at,
                    )
                )
                await db.commit()
        except SQLAlchemyError as exc:
            logger.warning("写入历史失败，本条未持久化（user=%s）：%s", user_id, exc)

    # ── 收藏（不注入 prompt；只经 memory.affinity 一条窄路进 item_picker 的弱加分）─────────

    async def read_favorites(self, user_id: str) -> list[FavoriteItem]:
        """读收藏（新→旧）。读失败只记日志、返回空。

        **收藏已不只是展示数据**：:mod:`app.memory.affinity` 把它当隐式行为信号，聚合成 item_picker
        的弱加分项。降级口径仍然是「返回空」——读不出收藏，就是这一轮没有行为亲和加分，排序略钝一点，
        绝不让它演变成任务失败（记忆是增强，不是依赖）。
        """
        if not user_id:
            return []
        try:
            async with session_factory()() as db:
                rows = (
                    await db.execute(
                        select(Favorite)
                        .where(Favorite.user_id == user_id)
                        .order_by(Favorite.created_at.desc())
                    )
                ).scalars()
                return [FavoriteItem.from_row(r) for r in rows]
        except SQLAlchemyError as exc:
            logger.warning("读取收藏失败，本次降级为空（user=%s）：%s", user_id, exc)
            return []

    async def write_favorite(self, user_id: str, item: FavoriteItem) -> None:
        """收藏一件商品；同 ``item_id`` 覆盖（重复点 ♡ 幂等）。超上限则丢最旧的。"""
        if not user_id:
            return
        try:
            async with session_factory()() as db:
                row = (
                    await db.execute(
                        select(Favorite).where(
                            Favorite.user_id == user_id, Favorite.item_id == item.item_id
                        )
                    )
                ).scalar_one_or_none()
                if row is None:
                    db.add(
                        Favorite(
                            id=_new_id(),
                            user_id=user_id,
                            item_id=item.item_id,
                            title=item.title,
                            platform=item.platform,
                            price_usd=item.price_usd,
                            landed_usd=item.landed_usd,
                            image_url=item.image_url,
                            url=item.url,
                            created_at=item.created_at,
                        )
                    )
                else:  # 重复收藏：刷新快照（价格 / 到手价可能已变），created_at 保持不动
                    row.title = item.title
                    row.platform = item.platform
                    row.price_usd = item.price_usd
                    row.landed_usd = item.landed_usd
                    row.image_url = item.image_url
                    row.url = item.url
                await db.commit()
                await self._trim_favorites(db, user_id)
        except SQLAlchemyError as exc:
            logger.warning("写入收藏失败，本条未持久化（user=%s）：%s", user_id, exc)

    async def _trim_favorites(self, db: AsyncSession, user_id: str) -> None:
        """超出上限就丢最旧的。收藏**无 TTL**（用户攒的清单该长期留着），只裁条数、不判过期。"""
        stale = (
            await db.execute(
                select(Favorite.id)
                .where(Favorite.user_id == user_id)
                .order_by(Favorite.created_at.desc())
                .offset(FAVORITES_MAX)
            )
        ).scalars()
        ids = list(stale)
        if ids:
            await db.execute(delete(Favorite).where(Favorite.id.in_(ids)))
            await db.commit()

    async def delete_favorite(self, user_id: str, item_id: str) -> None:
        """取消收藏。``item_id`` 不存在则静默无操作。"""
        if not user_id:
            return
        try:
            async with session_factory()() as db:
                await db.execute(
                    delete(Favorite).where(Favorite.user_id == user_id, Favorite.item_id == item_id)
                )
                await db.commit()
        except SQLAlchemyError as exc:
            logger.warning("取消收藏失败（user=%s，item_id=%s）：%s", user_id, item_id, exc)


@lru_cache(maxsize=1)
def get_store() -> PreferenceStore:
    """进程内共享的 Store（主 / 子 Agent 共用）。无状态，单例只为省对象。"""
    return PreferenceStore()
