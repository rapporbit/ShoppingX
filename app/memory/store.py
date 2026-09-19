"""用户级持久数据 —— 跨会话的**行为历史 / 收藏**。

对齐 refdocs/06 的核心区分：**长上下文 ≠ 长期记忆**。长上下文（消息历史）按 token 涨钱、只在单
会话有效、随轮数膨胀；长期数据按条目持久化、跨会话共享、只在有新事实时写入。

**长期记忆（偏好 / 约束 / 背景）不在本模块**，见 :mod:`app.memory.fact_store`：M1~M4 把它换成了
``key / value / category`` 的事实模型，同 key 覆盖写。本模块只剩两样不属于那套建模的东西：

- **行为历史**：「做过什么」的事实快照，既不去重也不合并，每种 kind 留最近几条 + TTL 过期。
- **收藏**：用户手工攒的商品清单，经 :mod:`app.memory.affinity` 一条窄路进 item_picker 的弱加分。

落地介质是 SQLite（复用 M16/M17 已有的 ``app.db``），不再有后端抽象基类——原来 ABC +
LocalFileStore + RedisStore 那三层各有一处并发写隐患（文件覆盖式 ``write_text`` 丢写、Redis
``hget`` + ``hset`` 非原子），而关系库的事务一并管掉，还顺手把「收藏超 200 条裁最旧」从
「读全量→算 overflow→回写」压成一条 SQL。

**容错口径（降级不崩）：** 这些数据是**增强**而非**依赖**。库读不出来只记日志、返回空——「这一轮
没有历史」，绝不让它演变成「这次任务失败」。
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

from app.db.models import Favorite, HistoryRecord
from app.db.session import session_factory
from app.utils.env import env_int

logger = logging.getLogger("shoppingx.memory")

HISTORY_TTL_DAYS = env_int("HISTORY_TTL_DAYS", 30)
# 每种 kind 保留最近几条历史（超出的按 created_at 淘汰最旧）。设为 1 即退回 last-write-wins。
HISTORY_MAX_PER_KIND = env_int("HISTORY_MAX_PER_KIND", 3)
# 收藏上限：超过则丢最旧的。收藏是用户手工攒的清单，不会自动膨胀，上限只是防脚本刷爆。
FAVORITES_MAX = env_int("FAVORITES_MAX", 200)


def _now() -> datetime:
    return datetime.now(UTC)


def _new_id() -> str:
    return uuid4().hex


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


class PreferenceStore:
    """用户级持久数据的读写口（**行为历史 / 收藏**），后端是 :mod:`app.db` 的 SQLite。

    **偏好那一腿已经不在这里了**：长期记忆改由 :class:`app.memory.fact_store.MemoryFactStore`
    按 ``key / value / category`` 存 ``memory_facts``（M1~M4）。旧的 ``preferences`` 表与
    ``app.db.models.Preference`` 行保留着不 drop，作为一版回滚依据，但**没有任何代码再读写它**。

    类名沿用 ``PreferenceStore`` 只是为了不动收藏 / 历史那十几处调用点；它现在名实不副，
    等收藏与历史也重构时一并改名。

    **不再有后端抽象基类**：原来 ABC + LocalFileStore + RedisStore 的三层结构，是为了「离线可跑」
    与「可选真后端」——而 SQLite 两样都占（零外部依赖、库文件躺在持久卷上），一个实现就够了。
    少一层抽象，就少一处「写了不读」的接缝。
    """

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
