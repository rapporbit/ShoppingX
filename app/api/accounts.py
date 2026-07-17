"""账户路由（M16）：注册 / 登录 / 我是谁 / 我的会话清单。

单独成一个 router 而不是继续堆进 ``server.py``——后者已经是任务、WS、取消、文件、偏好五摊事，
账户是**独立的一摊**（它不碰 AgentLoop），分开放读起来清楚。

**为什么这套口子会「取代」原来的 ``POST /api/auth/token``。** 那个是开发态发证口：不验密码、
给任意 user_id 签 token，本质是个「冒名工厂」，只为把鉴权链路端到端跑通。现在有了真实密码登录，
它在生产必须关（``AUTH_DEV_TOKEN=false``，默认就是关的），否则刚建好的账户体系等于没有门。
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.auth import auth_enabled, create_access_token, get_current_user_id
from app.api.ratelimit import guard_daily_signups, guard_login_ip, guard_register_ip
from app.db.accounts import (
    MAX_PASSWORD_BYTES,
    MIN_PASSWORD_LEN,
    authenticate,
    create_user,
    delete_thread,
    list_threads,
)
from app.db.session import get_db

router = APIRouter(prefix="/api", tags=["accounts"])

# 登录失败一律回这一句：不区分「查无此人」与「密码错」，否则登录口就成了用户名探测器
# （先枚举出哪些账号真实存在，再对着它们集中撞库）。
_BAD_CREDENTIALS = "用户名或密码错误"


class Credentials(BaseModel):
    """注册 / 登录共用的请求体。长度上限不是洁癖：bcrypt 只认前 72 字节，超出部分被静默忽略——
    不拦住的话，用户以为自己设了个 100 字符的强密码，实际生效的只有前 72 字节。"""

    username: str = Field(min_length=3, max_length=64)
    password: str = Field(min_length=MIN_PASSWORD_LEN)


def _require_auth_on() -> None:
    """账户体系只在开启鉴权时有意义。``AUTH_ENABLED=false`` 时签出的 token 没人验，
    注册登录只会给人「我已经登录了」的错觉——与其发一枚没人看的 token，不如明确 404。"""
    if not auth_enabled():
        raise HTTPException(404, "账户功能未开启（需 AUTH_ENABLED=true）")


def _issue(user_id: str, username: str) -> dict[str, Any]:
    return {
        "access_token": create_access_token(user_id),
        "token_type": "bearer",
        "user_id": user_id,
        "username": username,
    }


@router.post("/auth/register")
async def register(
    req: Credentials, request: Request, db: AsyncSession = Depends(get_db)
) -> dict[str, Any]:
    """注册并直接发 token（注册完不用再登录一次——少一次往返，用户少一步）。

    两道限流闸在**建号之前**跑：注册一旦成功，这个号今天就有了一份 credit 额度，事后再拦没有意义。
    """
    _require_auth_on()
    guard_register_ip(request)
    await guard_daily_signups(db)
    if len(req.password.encode()) > MAX_PASSWORD_BYTES:
        raise HTTPException(400, f"密码过长（上限 {MAX_PASSWORD_BYTES} 字节）")
    try:
        user = await create_user(db, req.username.strip(), req.password)
    except ValueError as exc:  # 用户名已被占用（数据库唯一索引拦下的）
        raise HTTPException(409, str(exc)) from exc
    return _issue(user.id, user.username)


@router.post("/auth/login")
async def login(
    req: Credentials, request: Request, db: AsyncSession = Depends(get_db)
) -> dict[str, Any]:
    """用户名 + 密码换 token。这里限流防的是**撞库**（拿密码字典对着一个账号猛试），不是防刷号。"""
    _require_auth_on()
    guard_login_ip(request)
    user = await authenticate(db, req.username.strip(), req.password)
    if user is None:
        raise HTTPException(401, _BAD_CREDENTIALS)
    return _issue(user.id, user.username)


@router.get("/auth/me")
async def whoami(uid: str | None = Depends(get_current_user_id)) -> dict[str, Any]:
    """前端刷新后拿本地 token 问一句「我还是我吗」——token 过期 / 被改则 401，前端据此跳登录页。"""
    _require_auth_on()
    return {"user_id": uid}


@router.delete("/sessions/{thread_id}")
async def delete_session(
    thread_id: str,
    uid: str | None = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
) -> dict[str, str]:
    """把一段会话从我的清单里删掉（侧栏那个删除按钮）。别人的会话删不动（403）。"""
    _require_auth_on()
    if uid is None:
        raise HTTPException(401, "未登录")
    try:
        await delete_thread(db, thread_id, uid)
    except PermissionError as exc:
        raise HTTPException(403, "无权访问该会话") from exc
    return {"status": "deleted", "thread_id": thread_id}


@router.get("/sessions")
async def my_sessions(
    uid: str | None = Depends(get_current_user_id), db: AsyncSession = Depends(get_db)
) -> dict[str, Any]:
    """当前用户的会话清单（最近聊过的在前）——侧栏历史的后端真源。

    在此之前侧栏只存在于浏览器的 localStorage 里：换台设备、清个缓存，后端数据明明还在，用户却
    再也找不回自己的对话。有了这张归属表，「换个浏览器登录还能看到我的历史」才真正成立。
    """
    _require_auth_on()
    if uid is None:
        raise HTTPException(401, "未登录")
    rows = await list_threads(db, uid)
    return {
        "sessions": [
            {
                "thread_id": t.id,
                "title": t.title,
                "created_at": t.created_at.isoformat(),
                "updated_at": t.updated_at.isoformat(),
            }
            for t in rows
        ]
    }
