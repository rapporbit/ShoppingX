"""买家个人 Skill 的读写（``user_skills`` 表）。

只做四件事：列自己的、建、改、软删。命名 / 长度校验放这里而不放 API 层，loader 与 API 两个
入口共用同一套规则。``name`` 是模型调 ``Skill(skill="my/<name>")`` 时的定位键，所以限成
``[a-z0-9][a-z0-9_-]{0,63}``：中文名放 ``description`` 里给模型看，name 只当 id 用。
"""

from __future__ import annotations

import re
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import UserSkill

#: 个人 skill 在 ``<agent-skills>`` 目录里的命名空间前缀，与 ``skills/`` 下内置 skill 隔开。
USER_SKILL_PREFIX = "my/"

MAX_SKILLS_PER_USER = 20
MAX_DESCRIPTION_LEN = 400
MAX_BODY_LEN = 8000
_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


class UserSkillError(ValueError):
    """入参不合法 / 数量超限 / 重名。API 层转 4xx。"""


def validate_fields(name: str, description: str, body: str) -> tuple[str, str, str]:
    name = (name or "").strip().lower()
    description = (description or "").strip()
    body = (body or "").strip()
    if not _NAME_RE.match(name):
        raise UserSkillError("name 只能是小写字母 / 数字 / - / _，且以字母或数字开头，最长 64")
    if not description or len(description) > MAX_DESCRIPTION_LEN:
        raise UserSkillError(f"description 不能为空且不超过 {MAX_DESCRIPTION_LEN} 字")
    if not body or len(body) > MAX_BODY_LEN:
        raise UserSkillError(f"正文不能为空且不超过 {MAX_BODY_LEN} 字")
    return name, description, body


async def list_user_skills(db: AsyncSession, user_id: str) -> list[UserSkill]:
    rows = await db.execute(
        select(UserSkill)
        .where(UserSkill.user_id == user_id, UserSkill.active.is_(True))
        .order_by(UserSkill.updated_at.desc())
    )
    return list(rows.scalars())


async def get_user_skill(db: AsyncSession, user_id: str, name: str) -> UserSkill | None:
    row = await db.execute(
        select(UserSkill).where(
            UserSkill.user_id == user_id,
            UserSkill.name == name,
            UserSkill.active.is_(True),
        )
    )
    return row.scalar_one_or_none()


async def create_user_skill(
    db: AsyncSession, user_id: str, name: str, description: str, body: str
) -> UserSkill:
    name, description, body = validate_fields(name, description, body)
    existing = await list_user_skills(db, user_id)
    if len(existing) >= MAX_SKILLS_PER_USER:
        raise UserSkillError(f"最多保存 {MAX_SKILLS_PER_USER} 个个人 Skill")
    if any(s.name == name for s in existing):
        raise UserSkillError(f"已有同名 Skill：{name}")
    # 同名但已软删的旧行：复活并换正文，避免撞 (user_id, name) 唯一约束。
    dead = await db.execute(
        select(UserSkill).where(UserSkill.user_id == user_id, UserSkill.name == name)
    )
    skill = dead.scalar_one_or_none()
    if skill is None:
        skill = UserSkill(id=uuid.uuid4().hex, user_id=user_id, name=name)
        db.add(skill)
        skill.version = 1
    else:
        skill.version += 1
    skill.description, skill.body, skill.active = description, body, True
    await db.commit()
    await db.refresh(skill)
    return skill


async def update_user_skill(
    db: AsyncSession, user_id: str, name: str, description: str, body: str
) -> UserSkill | None:
    skill = await get_user_skill(db, user_id, name)
    if skill is None:
        return None
    _, description, body = validate_fields(name, description, body)
    if (description, body) != (skill.description, skill.body):
        skill.description, skill.body = description, body
        skill.version += 1
        await db.commit()
        await db.refresh(skill)
    return skill


async def delete_user_skill(db: AsyncSession, user_id: str, name: str) -> bool:
    skill = await get_user_skill(db, user_id, name)
    if skill is None:
        return False
    skill.active = False
    await db.commit()
    return True
