"""清理过期的免登录试用账号（feat/guest-trial）。

访客没有密码，token 过期（``GUEST_JWT_EXP_SECONDS``，默认 7 天）后那行 users 和它挂着的会话就
再也没人能打开——留着只是占库、占 ``output/<thread_id>/`` 磁盘。本脚本把「最后活动早于 N 天」的
访客整个人删掉：users 行 + 所有按 user_id 挂的表 + 会话消息 + 会话目录。

**默认 dry-run**：只打印会删掉谁；加 ``--apply`` 才真删。跳过还有活跃 run_holds 的人（正在跑）。
「最后活动」= 该用户最近一段会话的 ``updated_at``；一段会话都没有的取 users.created_at。

用法：
    uv run python scripts/purge_stale_guests.py            # 看看会删谁（默认 30 天）
    uv run python scripts/purge_stale_guests.py --days 14 --apply
"""

from __future__ import annotations

import argparse
import asyncio
import shutil
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import delete, func, select  # noqa: E402

from app.db.holds import ACTIVE_STATES  # noqa: E402
from app.db.models import (  # noqa: E402
    ConfirmationRow,
    Favorite,
    HistoryRecord,
    MemoryFactRow,
    Message,
    OrderLineRow,
    OrderRow,
    RunHold,
    Thread,
    UsageLedger,
    User,
    UserSkill,
)
from app.db.session import init_db, session_factory  # noqa: E402
from app.utils.path_utils import OUTPUT_ROOT  # noqa: E402

# 按 user_id 直挂的表（Thread 单独处理：它带出消息与目录；OrderLineRow 经 order_id 挂）。
_USER_TABLES = (
    MemoryFactRow,
    HistoryRecord,
    Favorite,
    UsageLedger,
    RunHold,
    ConfirmationRow,
    UserSkill,
)


async def _stale_guest_ids(days: int) -> list[tuple[str, datetime]]:
    cutoff = datetime.now(UTC) - timedelta(days=days)
    async with session_factory()() as db:
        last_seen = (
            select(Thread.user_id, func.max(Thread.updated_at).label("last"))
            .group_by(Thread.user_id)
            .subquery()
        )
        rows = (
            await db.execute(
                select(User.id, func.coalesce(last_seen.c.last, User.created_at))
                .outerjoin(last_seen, last_seen.c.user_id == User.id)
                .where(User.is_guest.is_(True))
            )
        ).all()
        busy = set(
            (
                await db.execute(select(RunHold.user_id).where(RunHold.state.in_(ACTIVE_STATES)))
            ).scalars()
        )
    out = []
    for uid, last in rows:
        if last.tzinfo is None:  # SQLite 回来的 naive 时间按 UTC 读
            last = last.replace(tzinfo=UTC)
        if last < cutoff and uid not in busy:
            out.append((uid, last))
    return out


async def _purge(uid: str) -> int:
    """删掉一个访客的一切，返回删掉的会话数。"""
    async with session_factory()() as db:
        tids = list((await db.execute(select(Thread.id).where(Thread.user_id == uid))).scalars())
        if tids:
            await db.execute(delete(Message).where(Message.thread_id.in_(tids)))
            await db.execute(delete(Thread).where(Thread.id.in_(tids)))
        oids = list(
            (await db.execute(select(OrderRow.order_id).where(OrderRow.user_id == uid))).scalars()
        )
        if oids:
            await db.execute(delete(OrderLineRow).where(OrderLineRow.order_id.in_(oids)))
            await db.execute(delete(OrderRow).where(OrderRow.order_id.in_(oids)))
        for table in _USER_TABLES:
            await db.execute(delete(table).where(table.user_id == uid))
        await db.execute(delete(User).where(User.id == uid))
        await db.commit()
    for tid in tids:
        shutil.rmtree(OUTPUT_ROOT / tid, ignore_errors=True)
    return len(tids)


async def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--days", type=int, default=30, help="最后活动早于这么多天的访客才删（默认 30）"
    )
    ap.add_argument("--apply", action="store_true", help="真删；不加只打印")
    args = ap.parse_args()
    await init_db()  # 与应用启动同一条路：库没升到 0017 就先升，脚本不依赖服务起过
    stale = await _stale_guest_ids(args.days)
    mode = "执行删除" if args.apply else "dry-run"
    print(f"过期访客 {len(stale)} 人（阈值 {args.days} 天，{mode}）")
    for uid, last in stale:
        n = await _purge(uid) if args.apply else 0
        print(f"  {uid}  最后活动 {last:%Y-%m-%d}" + (f"  已删 {n} 段会话" if args.apply else ""))


if __name__ == "__main__":
    asyncio.run(main())
