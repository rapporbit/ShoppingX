"""JWT 鉴权（I 块 P1）—— ``user_id`` 从 token 解析，堵伪造 user_id 越权读他人 Store 的真实漏洞。

**修的是什么洞。** 现状 ``user_id`` 由前端经请求体 / URL 段传入、**无任何校验**：任意改 user_id
即可读他人长期偏好 Store（``GET /api/preferences/{user_id}``）或冒名跑任务、把偏好写进他人名下。
这是**已存在的越权读 / 越权写漏洞**，不是「完整度打磨」。修法是标准做法：身份只认 token 里签名过的
``sub``，**绝不信前端传来的 user_id**。

**为什么 P1 单做鉴权、不带限流 / 版本 / 幂等。** 越权读是安全洞（拿到别人数据），其余是生产化打磨
（防滥用 / 平滑演进）——前者优先级严格高于后者。本块只堵洞，限流 / 版本 / 幂等维持 P2 收尾。

**graceful 开关（与项目 demo 连续性的取舍）。** 现有前端不带 token，硬性强制 JWT 会让 demo 打不开。
故用 env ``AUTH_ENABLED`` 控制：

- ``AUTH_ENABLED=false``（默认）：保持现状——user_id 由前端传入，记一条 warning 提示「鉴权未开、
  存在越权读风险」。让课程 demo 照常跑。
- ``AUTH_ENABLED=true``：强制 ``Authorization: Bearer <jwt>``，**身份一律取 token 的 sub**，前端传的
  user_id 被忽略 / 校验；越权访问他人资源 403。

**token 怎么来（demo 边界）。** 配套一个 ``POST /api/auth/token`` 的**开发态发证口**（只认 user_id、
不验密码）——真实 OAuth / 密码登录 / 刷新令牌留作业（与 refdocs「真实平台 OAuth 不覆盖」一致）。它只
为「让鉴权链路能端到端被验证」，绝非生产登录，且单独由 ``AUTH_DEV_TOKEN`` 把守（见
:func:`dev_token_enabled`）。

**本块的范围边界（诚实标注，勿误读为「全锁了」）。** 本块只堵 **user_id 维度的越权读 Store**——这是
计划里点名的真实漏洞（``/api/preferences/{user_id}`` + 跑任务写偏好的身份）。**thread 维度的资源
（``/api/history``、``/api/files``、``/ws``、``/api/task/{tid}/cancel``、``/api/upload``）目前仍只按
thread_id 寻址、无属主校验**：谁知道某个 thread_id（connect-first 下它由前端生成、会出现在 URL /
事件流里）就能读其历史 / 产物 / 实时事件。要堵这半边，需要一张 **thread_id → owner(user_id) 归属表**
（任务创建时落、各 thread 接口校验属主），属下一增量。本块不做半成品归属表，先把点名的 Store
洞封死。
"""

from __future__ import annotations

import os
import time

import jwt
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.utils.env import env_bool, env_int

# 签名算法固定 HS256（对称密钥，单服务足够；多服务 / 第三方验签再上 RS256 非对称，留毕业线）。
_ALG = "HS256"
# auto_error=False：鉴权关闭时不强制要 header，由各依赖自行决定放行/拒绝（见 get_current_user_id）。
_bearer = HTTPBearer(auto_error=False)


def auth_enabled() -> bool:
    """是否开启 JWT 鉴权（env ``AUTH_ENABLED``，默认关——保 demo 连续性，见模块 docstring）。"""
    return env_bool("AUTH_ENABLED", False)


def dev_token_enabled() -> bool:
    """是否开启**开发态发证口**（env ``AUTH_DEV_TOKEN``，默认关）。

    **独立于 ``AUTH_ENABLED`` 的关键安全取舍**：发证口无密码、给任意 user_id 签 token——它本身就是
    一个「冒名工厂」。若与 ``AUTH_ENABLED`` 共用一个开关，那「开启鉴权」反而会同时暴露这个工厂、
    把刚堵上的越权洞原样捅开。故单列开关：``AUTH_ENABLED=true`` 单开即真正安全（无发证口），本地 /
    测试要发 token 才额外开 ``AUTH_DEV_TOKEN=true``。生产必须关它、换真实 IdP（OAuth / 密码登录）。
    """
    return env_bool("AUTH_DEV_TOKEN", False)


def _secret() -> str:
    """签名密钥（env ``JWT_SECRET``）。开启鉴权但没配密钥是致命配置错——直接抛，绝不用弱默认值
    （弱默认密钥 = 谁都能伪造 token，比不鉴权还危险）。"""
    secret = os.environ.get("JWT_SECRET", "").strip()
    if not secret:
        raise RuntimeError("AUTH_ENABLED=true 但未配置 JWT_SECRET——拒绝用弱默认密钥签发/验签 token")
    return secret


def validate_auth_config() -> None:
    """启动时校验鉴权配置：开了鉴权就必须有密钥，缺了当场 fail-fast，不拖到每请求 500。"""
    if auth_enabled():
        _secret()  # 缺密钥即抛 RuntimeError，让服务起不来（比上线后每请求 500 强）


def create_access_token(user_id: str, expires_in: int | None = None) -> str:
    """签发一个 ``sub=user_id`` 的 JWT（开发态发证 / 测试用）。

    ``exp`` 默认取 env ``JWT_EXP_SECONDS``（默认 24h）。用 ``time.time()`` 取签发时刻——PyJWT
    会据此写 ``iat`` / ``exp`` 并在验签时校验过期。
    """
    now = int(time.time())
    ttl = expires_in if expires_in is not None else env_int("JWT_EXP_SECONDS", 86400)
    payload = {"sub": user_id, "iat": now, "exp": now + ttl}
    return jwt.encode(payload, _secret(), algorithm=_ALG)


def decode_token(token: str) -> str:
    """验签并解出 ``sub``（= user_id）。验签失败 / 过期 / 缺 sub 一律抛 401（不泄漏具体原因）。"""
    try:
        # algorithms 锁死 HS256（防 alg=none / HS-RS 混淆）；require exp 拒绝无过期声明的 token
        # （否则一枚泄漏的无 exp token 永久有效）。签名 + 过期由 PyJWT 默认校验。
        payload = jwt.decode(token, _secret(), algorithms=[_ALG], options={"require": ["exp"]})
    except jwt.PyJWTError as exc:
        raise HTTPException(401, "无效或过期的凭证") from exc
    sub = payload.get("sub")
    if not sub or not isinstance(sub, str):
        raise HTTPException(401, "凭证缺少有效身份(sub)")
    return sub


def get_current_user_id(
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> str | None:
    """FastAPI 依赖：解析出当前请求的**可信** user_id。

    - 鉴权关闭：返回 ``None``——调用方退回「信前端传入的 user_id」的现状行为（带 warning）。
    - 鉴权开启：必须带 ``Authorization: Bearer <jwt>``，缺失 401；验签通过返回 token 里的 sub。

    **关键**：开启后身份只来自这里（token），调用方不得再用请求体 / URL 里的 user_id 当身份。
    """
    if not auth_enabled():
        return None
    if creds is None or not creds.credentials:
        raise HTTPException(401, "缺少 Authorization: Bearer 凭证")
    return decode_token(creds.credentials)


def resolve_identity(authenticated: str | None, requested: str | None) -> str | None:
    """合流「token 身份」与「前端传入身份」，返回本次任务**应当生效**的 user_id。

    - 鉴权开启：一律用 ``authenticated``（token 的 sub），忽略前端传入——杜绝冒名写偏好。
    - 鉴权关闭：退回 ``requested``（现状），匿名为 None。
    """
    return authenticated if auth_enabled() else requested
