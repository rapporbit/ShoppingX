"""参数覆盖的持久化：``config_overrides`` 表 ↔ 覆盖层内存。

内存那份（``overrides._OVERRIDES``）是**权威运行态**，库是它的持久影子：启动时库 → 内存，
改参时先落库再改内存。分成两个模块而非塞进 overrides.py，是为了让覆盖层本身不依赖数据库——
测试与脚本可以只 import overrides 做纯内存的参数改动，不用起库。
"""

from __future__ import annotations

import logging

from sqlalchemy import delete, select

from app.config import overrides
from app.config.registry import BY_KEY
from app.db.models import ConfigOverride
from app.db.session import session_factory

logger = logging.getLogger(__name__)


async def load_into_memory() -> int:
    """启动时把库里的覆盖读进内存并生效。返回成功应用的条数。

    **绝不因为一条脏数据让服务起不来**——库里可能留着已从注册表删掉的 key，或范围收紧后变得
    非法的旧值。这类行逐条跳过并记 warning（值继续跟随 .env / 代码默认），而不是让 lifespan 抛异常。
    """
    async with session_factory()() as session:
        rows = (await session.execute(select(ConfigOverride))).scalars().all()

    applied: dict[str, str] = {}
    for row in rows:
        if row.key not in BY_KEY:
            logger.warning("跳过库里的未知参数 %s（注册表已删？），其值不再生效", row.key)
            continue
        try:
            overrides.normalize(row.key, row.value)
        except overrides.ParamValidationError as e:
            logger.warning("跳过库里的非法参数值 %s=%r：%s", row.key, row.value, e)
            continue
        applied[row.key] = row.value

    if applied:
        overrides.apply(applied)
        logger.info("已从库载入 %d 条参数覆盖：%s", len(applied), ", ".join(sorted(applied)))
    return len(applied)


async def save(values: dict[str, str], updated_by: str) -> None:
    """把已规范化的覆盖 upsert 进库（调用方负责先经 :func:`overrides.normalize` 校验）。"""
    async with session_factory()() as session:
        for key, value in values.items():
            row = await session.get(ConfigOverride, key)
            if row is None:
                session.add(ConfigOverride(key=key, value=value, updated_by=updated_by))
            else:
                row.value = value
                row.updated_by = updated_by
        await session.commit()


async def remove(keys: list[str]) -> None:
    """删除覆盖行 = 恢复默认：该参数之后重新跟随 ``.env`` / 代码默认值。"""
    if not keys:
        return
    async with session_factory()() as session:
        await session.execute(delete(ConfigOverride).where(ConfigOverride.key.in_(keys)))
        await session.commit()
