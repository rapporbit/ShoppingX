"""后台管理 API：热更新模型档位与检索 / 展示参数（见 :mod:`app.config.registry`）。

**管理员认定走 env 白名单**（``ADMIN_USERNAMES=zjl,someone``），不在库里加 ``is_admin`` 列：
个人 demo 只有一个管理员，一条 env 就够，零迁移；且「谁是管理员」由部署方掌握，不会因为库被写坏
而多出一个管理员。将来真要在页面上授权他人，再加列不迟。

**鉴权未开时整个后台不可用**（403，而不是放行）。这是本模块最要紧的一条：这些接口能换模型、能
改召回阈值，是高权限面；而 ``AUTH_ENABLED=false`` 下身份来自前端自报的 user_id，可随意伪造——
放行等于把「改模型」开放给公网任何人。本地想调参直接改 ``.env`` 重启即可，不需要这个页面，故
不为本地便利开逃生门。
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.auth import auth_enabled, get_current_user_id
from app.config import overrides, store
from app.config.registry import BY_KEY, GROUPS, PARAMS
from app.db.models import User
from app.db.session import get_db
from app.utils.env import env_str

logger = logging.getLogger("shoppingx.admin")

router = APIRouter(prefix="/api/admin", tags=["admin"])


def admin_usernames() -> set[str]:
    """白名单（``ADMIN_USERNAMES``，逗号分隔）。空 = 没有任何管理员，后台整体关闭。"""
    raw = env_str("ADMIN_USERNAMES", "")
    return {name.strip() for name in raw.split(",") if name.strip()}


async def require_admin(
    uid: str | None = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
) -> str:
    """FastAPI 依赖：确认调用者是管理员，返回其用户名。否则 403。

    token 的 sub 是 user_id（随机 hex），白名单配的是用户名，故要查一次库换名。
    """
    if not auth_enabled():
        raise HTTPException(403, "后台管理需要开启鉴权（AUTH_ENABLED=true）后才可用")
    allowed = admin_usernames()
    if not allowed:
        raise HTTPException(403, "未配置任何管理员（ADMIN_USERNAMES 为空）")
    if not uid:
        raise HTTPException(401, "缺少凭证")

    user = await db.get(User, uid)
    if user is None or user.username not in allowed:
        # 记下来：有人拿着合法 token 试后台，值得知道。
        logger.warning("非管理员访问后台被拒：uid=%s", uid)
        raise HTTPException(403, "无后台管理权限")
    return user.username


class ParamView(BaseModel):
    """一个参数在前端表单里的完整描述 + 当前状态。"""

    key: str
    group: str
    label: str
    kind: str
    value: object  # 当前生效值；密钥类恒为空串（永不回显）
    default: object  # 代码默认值（「恢复默认」的目标之一，另一是 .env 基线）
    source: str  # override / env / default
    help: str
    warning: str
    minimum: float | None
    maximum: float | None
    allow_empty: bool
    secret: bool
    # 密钥的 sk-…a1b2 掩码，只用于核对「配没配 / 是不是那一把」。非密钥恒为空串。
    masked: str


class ConfigView(BaseModel):
    groups: dict[str, dict[str, str]]
    params: list[ParamView]


class UpdateRequest(BaseModel):
    values: dict[str, object]


class ResetRequest(BaseModel):
    # None = 全部恢复默认。
    keys: list[str] | None = None


def _snapshot() -> ConfigView:
    return ConfigView(
        groups=GROUPS,
        params=[
            ParamView(
                key=p.key,
                group=p.group,
                label=p.label,
                kind=p.kind,
                value=overrides.current_value(p),
                default=p.default,
                source=overrides.source_of(p.key),
                help=p.help,
                warning=p.warning,
                minimum=p.minimum,
                maximum=p.maximum,
                allow_empty=p.allow_empty,
                secret=p.secret,
                masked=overrides.masked_value(p),
            )
            for p in PARAMS
        ],
    )


@router.get("/config", response_model=ConfigView)
async def get_config(_: str = Depends(require_admin)) -> ConfigView:
    """注册表 + 每个参数的当前值与来源。前端表单完全由这份响应驱动。"""
    return _snapshot()


@router.put("/config", response_model=ConfigView)
async def update_config(req: UpdateRequest, admin: str = Depends(require_admin)) -> ConfigView:
    """改参数：校验 → 落库 → 写 env → 重载模块。对**新任务**生效。

    先落库再改内存：反过来的话，进程在两步之间挂掉就会出现「页面显示改了、重启后又变回去」的
    幽灵状态。落库失败则整个请求失败，内存不动，页面上看到的仍是真实生效值。
    """
    if not req.values:
        raise HTTPException(400, "没有要改的参数")

    # 密钥留空 = 不改（见 registry.Param.secret 规则 2）。**这道剔除必须在后端**：密钥在页面上天生
    # 是空的（不回显），把空当成新值提交会有两种后果，取决于该密钥的 allow_empty——
    #   - False（如 OPENAI_API_KEY）：normalize 拒绝空串 → **整批 400**。管理员改任何别的参数都被
    #     这个空密钥连累，保存永远失败，页面根本没法用。
    #   - True：空串被当真值写进 env → **key 真被抹掉**，全站 LLM 调用当场失效且毫无提示。
    # 剔除后「留空」才真正等于「不改」，两种后果都不会发生。前端「只提交改动过的字段」是第一道，
    # 但靠它兜等于把这件事押在客户端上。
    values = {
        k: v
        for k, v in req.values.items()
        if not (k in BY_KEY and BY_KEY[k].secret and not str(v).strip())
    }
    if not values:
        return _snapshot()  # 全是「密钥留空」→ 什么也没改，不是错误

    try:
        normalized = {k: overrides.normalize(k, v) for k, v in values.items()}
    except overrides.ParamValidationError as e:
        raise HTTPException(400, str(e)) from e

    await store.save(normalized, updated_by=admin)
    overrides.apply(normalized)
    # 日志里绝不能出现密钥明文——它会进 structlog、进磁盘、进任何日志收集管道，
    # 那比页面回显传得更远、活得更久。
    logged = {k: ("***" if BY_KEY[k].secret else v) for k, v in normalized.items()}
    logger.info("管理员 %s 改了 %d 个参数：%s", admin, len(normalized), logged)
    return _snapshot()


@router.post("/config/reset", response_model=ConfigView)
async def reset_config(req: ResetRequest, admin: str = Depends(require_admin)) -> ConfigView:
    """恢复默认：删覆盖行，值退回 ``.env`` 基线或代码默认值。"""
    keys = req.keys if req.keys else [p.key for p in PARAMS]
    try:
        overrides.reset(keys)
    except overrides.ParamValidationError as e:
        raise HTTPException(400, str(e)) from e
    await store.remove(keys)
    logger.info("管理员 %s 恢复默认：%s", admin, "全部" if not req.keys else keys)
    return _snapshot()


@router.get("/whoami")
async def whoami(admin: str = Depends(require_admin)) -> dict[str, str]:
    """给前端判断「要不要显示后台入口」用——非管理员拿到 403，入口就不渲染。"""
    return {"username": admin}
