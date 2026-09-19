"""run 级 **credit 预授权与结算**（阶段 1-1）—— 把额度闸从「事后记账」改成「进门先占」。

**为什么非要预扣。** :mod:`app.db.quota` 的账本是跑完才累加的，于是同一个人并发发 20 条 query 时，
每条进门读到的余额都还是满的，20 条全放行；跑完一起记账，超支已成事实。额度闸对并发是瞎的。预扣
把「打算花」也算进占用：后到的请求看见的余额已经扣过前面那些还在跑的，透支窗口就关上了。

**为什么顺带管并发上限。** 「这个人同时在跑几个」此前唯一的答案是 API 进程内的 ``active_tasks``
字典，多副本下一人打两台就各算各的。既然预扣行本来就要按 (user_id, state) 查一次，并发数就是同一
次查询的行数——不必再为它单开一套状态。

**三个口径说清楚：**

1. **预扣是猜，结算是真。** 进门按档位（normal 20 / heavy 60 credits）先占一笔，跑完按 ``cost_usd``
   算出真实量、多退少补。档位值该由压测实测均值定；``credits_held`` 与 ``credits_charged`` 两列都
   留着，就是为了事后能查「猜得准不准」。
2. **结算按 ``run_id`` 幂等。** 条件更新 ``WHERE state IN ('queued','running')``，影响 0 行 = 已经
   结算过（PEL 重投、整轮重跑），直接跳过。标 settled 与记账在**同一个事务**里（
   :func:`app.db.quota.accumulate_usage` 不自己 commit 就是为这个），否则两者之间崩一下会出现
   「额度释放了但钱没记」。
3. **崩溃的兜底是过期，不是扫表。** 进程被 kill -9 时行会停在 running 占着额度。读侧一律只数
   ``expires_at > now`` 的行，过期行自然失效、留在表里当排障线索（「这个 run 没有终态」）。

**开关与生效条件。** ``HOLDS_ENABLED=0`` 整个关掉，退回 ``add_usage`` 老路。另外它跟着
:func:`app.db.quota.quota_enabled` 走——鉴权关闭时所有人共用假身份 ``demo-user``，对它限并发等于
「一个人在跑，全体排队」，比不限更糟（理由与 quota 模块 docstring 同源）。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, cast

from sqlalchemy import CursorResult, func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import RunHold, User
from app.db.quota import accumulate_usage, get_quota, quota_enabled, to_credits
from app.db.session import session_factory
from app.utils.env import env_bool, env_int

logger = logging.getLogger("shoppingx.holds")

# 档位预扣量（credit）。normal ≈ 一次普通购物轮的实测均值向上取整，heavy 给多轮追问留余量。
# 这两个数是**可以猜错的**——结算会多退少补，猜大只是让同一人的并发额度提前用完。
HOLD_CREDITS: dict[str, int] = {"normal": 20, "heavy": 60}

# 一个人同时能跑几个 run。3 = 前端一个标签页顶多一个在飞，留两个给多开标签页 / 脚本。
MAX_CONCURRENT_RUNS = env_int("MAX_CONCURRENT_RUNS", 3)

# 预扣多久后自动失效。必须 > 单轮最长耗时（MAIN_AGENT_TIMEOUT_SEC=300），否则还在跑的 run 会被
# 当成过期、额度被别的请求重复占用。900s 留了三倍余量，够覆盖排队 + 重投一次。
HOLD_TTL_SEC = env_int("HOLD_TTL_SEC", 900)

# 拒绝原因（进 HTTP 响应体的 error 字段，前端按它分文案）。
REASON_CONCURRENCY = "user_concurrency"
REASON_QUOTA = "quota_exhausted"


def holds_enabled() -> bool:
    """预授权是否生效：开关打开 **且** 配额本身生效（见模块 docstring 最后一段）。"""
    return env_bool("HOLDS_ENABLED", True) and quota_enabled()


def hold_credits(kind: str) -> int:
    """某档位预扣多少 credit；未知档位按 normal 算（宁可少占也别为拼错的字符串炸掉准入）。"""
    return HOLD_CREDITS.get(kind, HOLD_CREDITS["normal"])


@dataclass(frozen=True)
class HoldResult:
    """一次准入判定的结论。``ok=False`` 时 ``reason`` 决定 API 回 429 还是 402。"""

    ok: bool
    reason: str = ""
    run_id: str = ""
    credits_held: int = 0
    active_runs: int = 0
    remaining_credits: int = 0

    def as_dict(self) -> dict[str, object]:
        return {
            "error": self.reason,
            "active_runs": self.active_runs,
            "max_concurrent_runs": MAX_CONCURRENT_RUNS,
            "remaining_credits": self.remaining_credits,
        }


# 「还活着」的两个状态。settled 是终态；过期与否另看 expires_at，不写成状态（见模块 docstring 3.）。
ACTIVE_STATES = ("queued", "running")


def _now() -> datetime:
    return datetime.now(UTC)


async def _active_snapshot(db: AsyncSession, user_id: str) -> tuple[int, int]:
    """该用户此刻 **(在飞 run 数, 已占 credit 数)**。一次聚合查完，两个判据共用同一份事实。"""
    row = (
        await db.execute(
            select(
                func.count(RunHold.run_id),
                func.coalesce(func.sum(RunHold.credits_held), 0),
            ).where(
                RunHold.user_id == user_id,
                RunHold.state.in_(ACTIVE_STATES),
                RunHold.expires_at > _now(),
            )
        )
    ).one()
    return int(row[0]), int(row[1])


async def acquire_hold(
    *, run_id: str, user_id: str | None, thread_id: str = "", kind: str = "normal"
) -> HoldResult:
    """准入：数在飞数、扣额度、落一行 ``queued``。被拒时 ``ok=False`` 且不留任何行。

    **余额判据与老 ``_enforce_quota`` 保持同一口径**：只有「可用额度 ≤ 0」才拒，剩一点点时照样放行、
    但只占住剩下的那点（``min(档位, 可用)``）。改成「不够一个档位就拒」会让用户在额度末尾完全发不
    出任务，而本轮真实花费大概率远小于档位——那是拿猜出来的数去拒真实的请求。透支上限仍被
    :func:`app.db.quota.remaining_usd` 压在 token_budget 上，最多超出最后那一轮。

    **同一个 ``run_id`` 重复申请不重复占**（幂等重发 / 重投）：撞主键即认原来那笔仍然有效。
    """
    if not holds_enabled() or not user_id:
        return HoldResult(ok=True, run_id=run_id)
    async with session_factory()() as db:
        # 锁住这个人的 users 行：并发的两个请求在这里排队，后到的那个一定看得见先到的写下的行。
        # SQLite 方言不支持行锁、会静默忽略 FOR UPDATE——它的整库写锁本就把并发串行化了。
        await db.execute(select(User.id).where(User.id == user_id).with_for_update())
        active, held = await _active_snapshot(db, user_id)
        quota = await get_quota(db, user_id)
        available = quota.remaining_credits - held
        if active >= MAX_CONCURRENT_RUNS:
            return HoldResult(
                ok=False,
                reason=REASON_CONCURRENCY,
                active_runs=active,
                remaining_credits=max(0, available),
            )
        if available <= 0:
            return HoldResult(
                ok=False, reason=REASON_QUOTA, active_runs=active, remaining_credits=0
            )
        credits = min(hold_credits(kind), available)
        db.add(
            RunHold(
                run_id=run_id,
                user_id=user_id,
                thread_id=thread_id,
                kind=kind,
                credits_held=credits,
                state="queued",
                expires_at=_now() + timedelta(seconds=HOLD_TTL_SEC),
            )
        )
        try:
            await db.commit()
        except IntegrityError:
            await db.rollback()
            logger.info("预扣已存在，按幂等放行：run_id=%s", run_id)
        return HoldResult(
            ok=True,
            run_id=run_id,
            credits_held=credits,
            active_runs=active + 1,
            remaining_credits=available - credits,
        )


async def mark_running(run_id: str) -> None:
    """``queued`` → ``running``。**纯粹是排障标签**，两个状态在并发与额度计算里等价（都算在飞）。

    有它才能在库里看出「这条卡在队列里」还是「worker 真的在跑它」——多副本下这是唯一能区分的地方。
    """
    if not holds_enabled():
        return
    async with session_factory()() as db:
        await db.execute(
            update(RunHold)
            .where(RunHold.run_id == run_id, RunHold.state == "queued")
            .values(state="running")
        )
        await db.commit()


async def settle(
    run_id: str,
    cost_usd: float,
    input_tokens: int = 0,
    output_tokens: int = 0,
    prompt_version: str = "",
) -> bool:
    """结算一次 run：释放预扣、按真实用量记账。**返回「这笔账已由预扣这条路负责」**。

    返回 ``False`` 只有一种含义：这条 run 没有预扣行（开关关着，或它根本没走准入），调用方该退回
    ``add_usage`` 老路自己记。**已经结算过的返回 ``True``**——条件更新影响 0 行就是幂等判据，钱早
    记过了，此时退回老路才是真的重复计费。PEL 重投把整轮重跑一遍时，靠的就是这条。

    标 settled 与记账在同一个事务里提交，中间崩一下不会留下「额度放了钱没记」的洞。
    """
    if not holds_enabled():
        return False
    async with session_factory()() as db:
        hold = await db.get(RunHold, run_id)
        if hold is None:
            return False
        user_id = hold.user_id
        res = cast(
            CursorResult[Any],
            await db.execute(
                update(RunHold)
                .where(RunHold.run_id == run_id, RunHold.state.in_(ACTIVE_STATES))
                .values(state="settled", credits_charged=to_credits(cost_usd), settled_at=_now())
            ),
        )
        if not res.rowcount:  # 已是 settled：重投 / 重复调用，钱早记过了
            logger.info("结算已完成，跳过重复计费：run_id=%s", run_id)
            return True
        if cost_usd > 0:
            await accumulate_usage(
                db,
                user_id,
                cost_usd,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                prompt_version=prompt_version,
            )
        await db.commit()
    return True


async def release(run_id: str) -> None:
    """零成本释放（入队失败、任务还没跑就没了）。结算成 0 credit，不记账。"""
    await settle(run_id, 0.0)


async def hold_state(run_id: str) -> str | None:
    """这条 run 的状态；没有 hold 行返回 ``None``。给消费侧去重（B4）与排障用。"""
    if not holds_enabled():
        return None
    async with session_factory()() as db:
        hold = await db.get(RunHold, run_id)
        return hold.state if hold else None
