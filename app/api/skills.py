"""买家个人 Skill 的 HTTP 接口 + 目录读口。

- ``GET  /api/skills/catalog``        内置 + 我的 skill 目录（name / description / source），
  喂输入框 ``/`` 菜单
- ``GET  /api/skills``                我的 skill 全量（含正文，编辑面板用）
- ``POST /api/skills``                新建
- ``PUT  /api/skills/{name}``         改 description / body（name 是定位键，改名 = 删了重建）
- ``DELETE /api/skills/{name}``       软删

归属：开启 ``AUTH_ENABLED`` 后只认 token 里的 sub；未开启时退回 ``user_id`` 查询参数（与偏好接口
同口径）。没有身份就没有个人 skill——匿名只能看内置目录。
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.skills import list_catalog
from app.api.auth import auth_enabled, get_current_user_id
from app.db.session import get_db
from app.db.user_skills import (
    USER_SKILL_PREFIX,
    UserSkillError,
    create_user_skill,
    delete_user_skill,
    list_user_skills,
    update_user_skill,
)

router = APIRouter(prefix="/api/skills", tags=["skills"])


class SkillWrite(BaseModel):
    name: str = ""
    description: str
    body: str


def _skill_json(s: Any) -> dict[str, Any]:
    return {
        "name": s.name,
        "catalog_name": f"{USER_SKILL_PREFIX}{s.name}",
        "description": s.description,
        "body": s.body,
        "version": s.version,
        "updated_at": s.updated_at.isoformat() if s.updated_at else None,
    }


def _uid(auth_uid: str | None, user_id: str | None) -> str:
    """有效身份：鉴权开着只认 token；关着退回 query 的 user_id。都没有 → 401。"""
    uid = auth_uid if auth_enabled() else (auth_uid or user_id)
    if not uid:
        raise HTTPException(401, "个人 Skill 需要登录")
    return uid


@router.get("/catalog")
async def catalog(
    auth_uid: str | None = Depends(get_current_user_id),
    user_id: str | None = Query(default=None),
) -> dict[str, Any]:
    uid = auth_uid if auth_enabled() else (auth_uid or user_id)
    return {"skills": await list_catalog(uid)}


@router.get("")
async def my_skills(
    auth_uid: str | None = Depends(get_current_user_id),
    user_id: str | None = Query(default=None),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    uid = _uid(auth_uid, user_id)
    return {"skills": [_skill_json(s) for s in await list_user_skills(db, uid)]}


@router.post("")
async def create_skill(
    req: SkillWrite,
    auth_uid: str | None = Depends(get_current_user_id),
    user_id: str | None = Query(default=None),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    uid = _uid(auth_uid, user_id)
    try:
        s = await create_user_skill(db, uid, req.name, req.description, req.body)
    except UserSkillError as e:
        raise HTTPException(422, str(e)) from None
    return {"skill": _skill_json(s)}


@router.put("/{name}")
async def update_skill(
    name: str,
    req: SkillWrite,
    auth_uid: str | None = Depends(get_current_user_id),
    user_id: str | None = Query(default=None),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    uid = _uid(auth_uid, user_id)
    try:
        s = await update_user_skill(db, uid, name, req.description, req.body)
    except UserSkillError as e:
        raise HTTPException(422, str(e)) from None
    if s is None:
        raise HTTPException(404, "Skill 不存在")
    return {"skill": _skill_json(s)}


@router.delete("/{name}")
async def delete_skill(
    name: str,
    auth_uid: str | None = Depends(get_current_user_id),
    user_id: str | None = Query(default=None),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    uid = _uid(auth_uid, user_id)
    if not await delete_user_skill(db, uid, name):
        raise HTTPException(404, "Skill 不存在")
    return {"ok": True}
