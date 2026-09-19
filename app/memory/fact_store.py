"""长期记忆事实的存取口：六个方法，后端是 `app.db` 的 SQLite（阶段 1 后是 MySQL）。

与 `PreferenceStore` 并存一段时间：`preferences` 表不 drop，旧读写路径在 M4 的删除清单里一起摘，
这中间两张表都在库里，但**只有这一个 store 被新代码调用**（见计划 §4.1 C4，留旧表作回滚依据）。

容错口径与 `PreferenceStore` 一致：**记忆是增强不是依赖**。读失败返回空 = 「这轮没有长期记忆」，
写失败只记日志 = 「这条没记住，下次再说」，都不许把一次购物任务拖失败。
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime

from sqlalchemy import delete, select, update
from sqlalchemy.exc import SQLAlchemyError

from app.db.models import MemoryFactRow, User
from app.db.session import session_factory
from app.memory.facts import MemoryCategory, MemoryFact, match_facts

logger = logging.getLogger(__name__)


def _new_id() -> str:
    return uuid.uuid4().hex


def _to_fact(row: MemoryFactRow) -> MemoryFact:
    return MemoryFact(
        key=row.fact_key,
        value=row.fact_value,
        category=MemoryCategory(row.category)
        if row.category in {c.value for c in MemoryCategory}
        else MemoryCategory.PREFERENCE,
        updated_at=row.updated_at or datetime.now(UTC),
        source_session=row.source_session or "",
    )


class MemoryFactStore:
    """六个方法就是全部契约。任何新的读写需求先问「能不能用这六个拼出来」。"""

    async def get_facts(self, user_id: str) -> list[MemoryFact]:
        """某用户的全部事实，按 ``updated_at`` 倒序。库故障返回空，不抛。"""
        if not user_id:
            return []
        try:
            async with session_factory()() as db:
                rows = (
                    await db.execute(
                        select(MemoryFactRow)
                        .where(MemoryFactRow.user_id == user_id)
                        .order_by(MemoryFactRow.updated_at.desc())
                    )
                ).scalars()
                return [_to_fact(r) for r in rows]
        except SQLAlchemyError as exc:
            logger.warning("读取长期记忆失败，本轮按空处理（user=%s）：%s", user_id, exc)
            return []

    async def upsert_facts(self, user_id: str, facts: list[MemoryFact]) -> bool:
        """按 ``key`` 覆盖写：存在则替换 value / category / 时间，不存在则插入。返回是否真的落库。

        **覆盖而不是叠加**是这套模型的要点——「我不要塑料」后来变成「塑料也行」时，库里不该同时留着
        两条互相矛盾的事实等注入时再打架。

        返回 bool 而不是 None：`save_memory` 要当场给用户一句「已记住」，库挂了还说记住了就是
        撒谎——用户以为不用再说第二遍。容错口径不变（异常仍只记日志、不往上抛）。
        """
        if not user_id or not facts:
            return False
        try:
            async with session_factory()() as db:
                for fact in facts:
                    row = (
                        await db.execute(
                            select(MemoryFactRow).where(
                                MemoryFactRow.user_id == user_id,
                                MemoryFactRow.fact_key == fact.key,
                            )
                        )
                    ).scalar_one_or_none()
                    now = fact.updated_at or datetime.now(UTC)
                    if row is None:
                        db.add(
                            MemoryFactRow(
                                id=_new_id(),
                                user_id=user_id,
                                fact_key=fact.key,
                                fact_value=fact.value,
                                category=fact.category.value,
                                source_session=fact.source_session,
                                created_at=now,
                                updated_at=now,
                            )
                        )
                    else:
                        row.fact_value = fact.value
                        row.category = fact.category.value
                        row.source_session = fact.source_session or row.source_session
                        row.updated_at = now
                await db.commit()
                return True
        except SQLAlchemyError as exc:
            logger.warning(
                "写入长期记忆失败，%d 条未持久化（user=%s）：%s", len(facts), user_id, exc
            )
            return False

    async def search_facts(self, user_id: str, query: str) -> list[MemoryFact]:
        """按主题召回。**在 Python 侧用 :func:`match_facts` 过滤，不写成 SQL LIKE**：

        一个用户的事实是几条到几十条，全读回来过一遍的成本可以忽略；换来的是匹配口径只有一份
        （中文子串那套），不会出现「SQL 里一种、内存里另一种」的两套语义。
        """
        return match_facts(await self.get_facts(user_id), query)

    async def delete_fact(self, user_id: str, key: str) -> bool:
        """按 key 删一条，返回是否真的删掉了。**只由偏好页的 HTTP 接口调用**——

        模型侧的「忘掉 X」走 `save_memory` 覆盖写，不给模型直接删除的能力（计划 §3.2 第 2 条）。
        """
        if not user_id or not key:
            return False
        try:
            async with session_factory()() as db:
                result = await db.execute(
                    delete(MemoryFactRow).where(
                        MemoryFactRow.user_id == user_id,
                        MemoryFactRow.fact_key == key,
                    )
                )
                await db.commit()
                # DELETE 拿到的是 CursorResult，运行时确有 rowcount；
                # ``AsyncSession.execute`` 的返回标注是笼统的 ``Result``，故 getattr 取。
                return bool(getattr(result, "rowcount", 0))
        except SQLAlchemyError as exc:
            logger.warning("删除记忆失败（user=%s，key=%s）：%s", user_id, key, exc)
            return False

    async def clear(self, user_id: str) -> None:
        """清空该用户全部事实，**并把 ``memory_purge_gen`` 加一**。

        两件事必须在同一个事务里：代数是「这次清空之前开始的抽取，结果作废」的唯一判据，
        先删后加之间崩掉的话，一批本该作废的抽取会落回刚清空的库里。
        """
        if not user_id:
            return
        try:
            async with session_factory()() as db:
                await db.execute(
                    delete(MemoryFactRow).where(MemoryFactRow.user_id == user_id)
                )
                await db.execute(
                    update(User)
                    .where(User.id == user_id)
                    .values(memory_purge_gen=User.memory_purge_gen + 1)
                )
                await db.commit()
        except SQLAlchemyError as exc:
            logger.warning("清空记忆失败（user=%s）：%s", user_id, exc)

    async def purge_generation(self, user_id: str) -> int:
        """该用户被清空过几次；没有这个用户或读不出来都返回 0。

        读失败返回 0 的代价是**抽取照常落库**（代数比对不出差异）。反过来——读失败当作「变了」
        而丢弃——会让一次库抖动静默吃掉用户刚说的偏好，那个更难被发现。
        """
        if not user_id:
            return 0
        try:
            async with session_factory()() as db:
                gen = (
                    await db.execute(
                        select(User.memory_purge_gen).where(User.id == user_id)
                    )
                ).scalar_one_or_none()
                return int(gen or 0)
        except SQLAlchemyError as exc:
            logger.warning("读取清空代数失败（user=%s）：%s", user_id, exc)
            return 0


_store = MemoryFactStore()


def get_fact_store() -> MemoryFactStore:
    """进程内单例。无状态，复用只是省掉重复构造。"""
    return _store
