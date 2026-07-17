"""用户级 **credit 配额**：每个用户每天能烧掉多少 LLM 成本，到顶就不给发新任务。

**它和 token_budget 是两把不同的闸，必须同时存在。** ``app/agent/token_budget.py`` 管的是**一次
任务**（一棵 fork 树）不许烧过 ``TOKEN_BUDGET_USD``——它防的是「单条 query 把 Agent 带进无底洞」。
但那把闸对「同一个人连发一百条 query」完全无感：每条都合规，加起来照样把账单打穿。本模块管的正是
后者——**跨会话、跨进程重启的累计**，故必须落库（:class:`app.db.models.UsageLedger`），不能像
token_budget 那样住在进程内 dict 里。

**判定口径是 cost_usd，不是 token 数。** 复用 token_budget 已有的三档计价（input / output /
cache_read 各自单价），cache 命中的那部分 input 享折扣——按 token 裸数计费会把「靠 prompt cache
省下来的钱」当成用户真花的钱，对追问多轮的用户平白苛刻，而追问恰恰是本产品希望用户做的事。

**但用户看到的不是美元，是 credit。** 美元是我们的成本口径，不该泄给用户（它暴露供应商费率、也
让人误以为在充值）。对外一律换算成整数 credit（:data:`CREDITS_PER_USD`），前端只认这个数。

**周期靠「换一行」自然重置，没有定时任务。** period_key 是 UTC 自然日；新的一天第一次记账 upsert
出一行新的、从 0 起算，老行留着当历史账单。用 UTC 而非本地时区：服务器时区一改，用户的额度会凭空
多出或蒸发一段。

**配额只在鉴权开启时生效**（``AUTH_ENABLED=false`` 的 demo / 本地开发不设闸）：关掉鉴权时所有人
共用一个假身份 ``demo-user``，对它记账等于「一个人烧完全体停用」，既不公平也拦不住真正想薅的人
（不登录就没有身份可限，那道门本来就该由鉴权来关）。
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, cast

from sqlalchemy import CursorResult, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.auth import auth_enabled
from app.db.models import UsageLedger
from app.db.session import session_factory
from app.utils.env import env_float

logger = logging.getLogger("shoppingx.quota")

# 1 credit = $0.001。取这个刻度是为了让日常数字落在「几百到几千」这个人类好读的区间：一次典型
# 购物任务约 $0.01~0.05，即 10~50 credit；默认日额 $2 = 2000 credit，够几十次。
CREDITS_PER_USD = 1000


def daily_quota_usd() -> float:
    """每人每日成本上限（美元）。``<=0`` 表示不设闸。"""
    return env_float("DAILY_QUOTA_USD", 2.0)


def quota_enabled() -> bool:
    """配额是否生效：**必须开着鉴权**（否则没有可信身份可限，见模块 docstring）且上限 > 0。"""
    return auth_enabled() and daily_quota_usd() > 0


def current_period() -> str:
    """当前计费周期键：UTC 自然日，形如 ``"2026-07-14"``。"""
    return datetime.now(UTC).strftime("%Y-%m-%d")


def next_reset_at() -> datetime:
    """本周期的重置时刻（下一个 UTC 零点）——前端拿它显示「几点回满」。"""
    now = datetime.now(UTC)
    return (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)


def to_credits(usd: float) -> int:
    """美元 → credit（向上取整，宁可多算一点也不让用户白嫖出零头）。"""
    return int(usd * CREDITS_PER_USD + 0.999)


@dataclass(frozen=True)
class QuotaStatus:
    """某用户当前周期的配额状态。``enabled=False`` 时其余字段无意义（前端据此隐藏余额条）。"""

    enabled: bool
    period: str
    used_credits: int
    limit_credits: int
    remaining_credits: int
    task_count: int
    reset_at: str  # ISO8601（UTC）

    @property
    def exhausted(self) -> bool:
        return self.enabled and self.remaining_credits <= 0

    def as_dict(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "period": self.period,
            "used_credits": self.used_credits,
            "limit_credits": self.limit_credits,
            "remaining_credits": self.remaining_credits,
            "task_count": self.task_count,
            "reset_at": self.reset_at,
            "exhausted": self.exhausted,
        }


def disabled_status() -> QuotaStatus:
    """「不设闸」时的空状态：前端见 enabled=false 即整块隐藏余额条。"""
    return QuotaStatus(
        enabled=False,
        period=current_period(),
        used_credits=0,
        limit_credits=0,
        remaining_credits=0,
        task_count=0,
        reset_at=next_reset_at().isoformat(),
    )


async def get_quota(db: AsyncSession, user_id: str | None) -> QuotaStatus:
    """读某用户当前周期的配额状态。**不写库**——没记过账的人就是「一分没用」，不必先插一行占位。"""
    if not quota_enabled() or not user_id:
        return disabled_status()
    period = current_period()
    row = (
        await db.execute(
            select(UsageLedger).where(
                UsageLedger.user_id == user_id, UsageLedger.period_key == period
            )
        )
    ).scalar_one_or_none()

    limit_usd = daily_quota_usd()
    used_usd = row.cost_usd if row else 0.0
    limit_credits = to_credits(limit_usd)
    used_credits = to_credits(used_usd)
    return QuotaStatus(
        enabled=True,
        period=period,
        used_credits=used_credits,
        limit_credits=limit_credits,
        remaining_credits=max(0, limit_credits - used_credits),
        task_count=row.task_count if row else 0,
        reset_at=next_reset_at().isoformat(),
    )


async def remaining_usd(user_id: str | None) -> float | None:
    """当前周期剩余额度（美元）；不设闸 / 无身份返回 ``None``（调用方据此不加任何限制）。

    给任务入口与 :mod:`app.agent.token_budget` 用：本次任务的成本上限会被压到「不超过今日剩余」，
    这样即便入口放行了最后一次任务，它也只能透支到额度用尽为止，不会整整多烧一个单任务预算。
    """
    if not quota_enabled() or not user_id:
        return None
    async with session_factory()() as db:
        status = await get_quota(db, user_id)
    return max(0.0, (status.limit_credits - status.used_credits) / CREDITS_PER_USD)


async def add_usage(
    user_id: str | None, cost_usd: float, input_tokens: int = 0, output_tokens: int = 0
) -> None:
    """把一次任务的全树用量累加进当前周期的账本。**记账失败绝不反噬主链路**（吞异常记日志）。

    先 ``UPDATE ... SET cost = cost + ?``（数据库内做加法，不是「读出来加完写回去」——后者在并发下
    会丢更新），影响 0 行说明本周期还没有这一行，再 ``INSERT``；两个请求同时插入时靠唯一索引挡下
    后到的那个，捕获 :class:`IntegrityError` 后重跑一次 UPDATE 即可。方言无关，换 Postgres 照跑。

    **鉴权关闭时不记账**（``quota_enabled()`` 为假）：那时所有人共用假身份，记出来的账没有意义。
    """
    if not quota_enabled() or not user_id or cost_usd <= 0:
        return
    period = current_period()
    try:
        async with session_factory()() as db:
            for attempt in range(2):
                stmt = (
                    update(UsageLedger)
                    .where(UsageLedger.user_id == user_id, UsageLedger.period_key == period)
                    .values(
                        cost_usd=UsageLedger.cost_usd + cost_usd,
                        input_tokens=UsageLedger.input_tokens + input_tokens,
                        output_tokens=UsageLedger.output_tokens + output_tokens,
                        task_count=UsageLedger.task_count + 1,
                    )
                )
                res = cast(CursorResult[Any], await db.execute(stmt))
                if res.rowcount:
                    await db.commit()
                    return
                if attempt:  # UPDATE 落空两次 → 行既插不进也更新不到，不该发生，别死循环
                    logger.warning("配额记账落空：user=%s period=%s", user_id, period)
                    return
                db.add(
                    UsageLedger(
                        id=uuid.uuid4().hex,
                        user_id=user_id,
                        period_key=period,
                        cost_usd=cost_usd,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        task_count=1,
                    )
                )
                try:
                    await db.commit()
                    return
                except IntegrityError:  # 并发下别人先插了这一行 → 回去走 UPDATE 累加
                    await db.rollback()
    except Exception:
        logger.exception("配额记账失败（不影响本次任务结果）：user=%s", user_id)
