"""SQLite → MySQL 一次性搬数（后端优化阶段 1 · 条 9）。

gcjp 那份 ``var/globex.db`` 里有真实注册用户、会话、收藏与账单流水，切到 MySQL 不能重来一遍。
本脚本把它整库搬过去，**停写窗口内跑一次**，跑完逐表核对行数，对不上就退出码 1。

**为什么按 SQLAlchemy 的表元数据搬，而不是 ``.dump`` 出 SQL 再灌。** SQLite 的 SQL 方言与 MySQL
对不上（``AUTOINCREMENT``、布尔存 0/1、JSON 存 TEXT、时间存字符串），转义与类型全得手工纠。
走 Core 的话，读出来是 Python 对象（``list`` / ``datetime`` / ``bool``），再由 MySQL 方言按列类型
编码回去，两头的类型转换都是库自己做的，不用我们猜。

**时间列的口径。** 模型全是 ``DateTime(timezone=True)``，值来自 ``datetime.now(UTC)``；SQLite 的
DATETIME 存的时候**把 tzinfo 丢了**（存 "2026-09-14 14:44:43.812380"），读回来是 naive UTC。
MySQL 的 DATETIME 同样不带时区。所以两头都是「naive，值即 UTC」——脚本仍显式归一一次
（aware → 转 UTC 再去掉 tzinfo），免得将来源库换成别的、悄悄按本地时间搬过去。

用法::

    # 先看一眼两边各有多少行，不写任何数据
    uv run python scripts/migrate_sqlite_to_mysql.py --dry-run \\
        --target "mysql+aiomysql://user:pass@host:3306/globex?charset=utf8mb4"

    # 真搬（目标表必须是空的；要重来一次加 --truncate）
    uv run python scripts/migrate_sqlite_to_mysql.py --target "mysql+aiomysql://..."

回滚：目标库是新建的，出问题就 ``DROP DATABASE`` 重来；源 SQLite 全程只读，一个字节都不改。
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import Table, func, select, text  # noqa: E402
from sqlalchemy.ext.asyncio import AsyncConnection  # noqa: E402

from app.db.models import Base  # noqa: E402
from app.utils.path_utils import PROJECT_ROOT  # noqa: E402

#: 一次 executemany 的行数。表都不大（最大的 messages 千行量级），分批只是防某天历史表涨起来
#: 之后一条 INSERT 撑爆 ``max_allowed_packet``（MySQL 默认 64MB，messages 带 JSON 列很能吃）。
BATCH = 500

#: 版本号表不搬：目标库的版本由它自己跑迁移决定，源库停在哪一版与它无关（源可能还落后几版）。
SKIP_TABLES = {"alembic_version"}


def _normalize(value: Any) -> Any:
    """aware datetime 一律折成 UTC naive——库内只存 UTC，见模块 docstring。"""
    if isinstance(value, datetime) and value.tzinfo is not None:
        return value.astimezone(UTC).replace(tzinfo=None)
    return value


async def _count(conn: AsyncConnection, table: Table) -> int:
    return (await conn.execute(select(func.count()).select_from(table))).scalar_one()


async def _copy(src: AsyncConnection, dst: AsyncConnection, table: Table) -> int:
    """整表读出来再分批灌进去。整表读是刻意的：表都不大，而流式读 SQLite 换不来任何东西，
    反倒要处理游标与事务边界。真到了读不下的那天，这里换 ``stream()`` 即可。"""
    rows = (await src.execute(select(table))).fetchall()
    payload = [{k: _normalize(v) for k, v in row._mapping.items()} for row in rows]
    for i in range(0, len(payload), BATCH):
        await dst.execute(table.insert(), payload[i : i + BATCH])
    return len(payload)


async def _preflight(
    src: AsyncConnection, dst: AsyncConnection, tables: list[Table], truncate: bool
) -> tuple[list[Table], dict[str, int]]:
    """搬之前先把两边都点一遍：源有哪些表、目标是不是空的。

    **全部检查完再动手**，不边查边搬——搬到第七张表才发现目标不空、前六张已经进去了，这时候
    人得手工判断哪些是新灌的、哪些是原有的，比一开始就拒绝难收拾得多。
    """
    from app.db.session import _head_revision, _table_names  # noqa: PLC0415 —— 见 run()

    src_tables = await src.run_sync(_table_names)
    # **源库必须已经升到 head**：脚本按当前模型的列 SELECT，源库落后一版就会去查一个它没有的列，
    # 报的还是 "no such column" 这种看不出根因的错。线上库天天跟着应用升，落后的通常是手里某份
    # 旧备份——那就先用旧库起一次应用（或 alembic upgrade head）再搬。
    head = _head_revision()
    if "alembic_version" not in src_tables:
        raise SystemExit("源库没有 alembic_version 表，不像是本应用管理的库；确认 --source 指对了")
    src_rev = (await src.execute(text("SELECT version_num FROM alembic_version"))).scalar()
    if src_rev != head:
        raise SystemExit(
            f"源库版本 {src_rev} 不是 head（{head}）。先把源库升到 head 再搬："
            f"DATABASE_URL=<源 DSN> uv run alembic upgrade head"
        )
    # 目标侧也先取一次表清单：--dry-run 常在目标库还没建表时跑，直接 count 会撞「表不存在」。
    dst_tables = await dst.run_sync(_table_names)
    todo: list[Table] = []
    src_counts: dict[str, int] = {}
    occupied: list[str] = []
    for table in tables:
        if table.name not in src_tables:
            continue  # 源库落后几版、还没这张表：目标由迁移建成空表即可，不算错
        src_counts[table.name] = await _count(src, table)
        if table.name in dst_tables and await _count(dst, table) and not truncate:
            occupied.append(table.name)
        todo.append(table)
    if occupied:
        raise SystemExit(
            f"目标库这些表已有数据：{', '.join(occupied)}。"
            "确认要覆盖就加 --truncate（会先清空这些表），否则换一个干净的库。"
        )
    return todo, src_counts


async def run(source: str, target: str, *, dry_run: bool, truncate: bool, migrate: bool) -> int:
    # app.db.session 在 **import 那一刻**就按 DATABASE_URL 造了共享引擎，所以环境变量必须先设、
    # 再 import；顺序反过来的话 init_db() 会去升源 SQLite，而不是目标 MySQL。
    os.environ["DATABASE_URL"] = target
    from app.db.session import init_db, make_engine

    if migrate and not dry_run:
        print("· 目标库跑 alembic upgrade head …")
        await init_db()

    tables = [t for t in Base.metadata.sorted_tables if t.name not in SKIP_TABLES]
    src_engine, dst_engine = make_engine(source), make_engine(target)
    failures = 0
    try:
        async with src_engine.connect() as src, dst_engine.begin() as dst:
            todo, src_counts = await _preflight(src, dst, tables, truncate)
            if dry_run:
                for table in todo:
                    print(f"  {table.name:24} 源 {src_counts[table.name]:>7}")
                print("\n(--dry-run：未写入任何数据)")
                return 0
            if truncate:
                # 反序删：子表先清，否则外键把父表的 DELETE 挡住。
                for table in reversed(todo):
                    await dst.execute(table.delete())
            print(f"{'表':24} {'源':>7} {'目标':>7}  结果")
            for table in todo:
                moved = await _copy(src, dst, table)
                got = await _count(dst, table)
                want = src_counts[table.name]
                ok = moved == want == got
                failures += 0 if ok else 1
                print(f"{table.name:24} {want:>7} {got:>7}  {'OK' if ok else '✗ 行数对不上'}")
    finally:
        await src_engine.dispose()
        await dst_engine.dispose()
    if failures:
        print(f"\n{failures} 张表行数对不上，事务已回滚（目标库未改变）")
        return 1
    print("\n全部表行数一致，搬运完成")
    return 0


def main() -> int:
    default_src = f"sqlite+aiosqlite:///{PROJECT_ROOT / 'var' / 'globex.db'}"
    parser = argparse.ArgumentParser(description="SQLite → MySQL 一次性搬数（停写窗口内跑）")
    parser.add_argument("--source", default=default_src, help=f"源 DSN，默认 {default_src}")
    parser.add_argument("--target", default=os.environ.get("DATABASE_URL", ""), help="目标 DSN")
    parser.add_argument("--dry-run", action="store_true", help="只报两边行数，不写")
    parser.add_argument("--truncate", action="store_true", help="目标表非空时先清空（危险）")
    parser.add_argument("--skip-migrate", action="store_true", help="目标已建好表，跳过 upgrade")
    args = parser.parse_args()
    if not args.source.startswith("sqlite"):
        parser.error("--source 必须是 SQLite DSN")
    if not args.target or args.target.startswith("sqlite"):
        parser.error("--target 必须给一个非 SQLite 的 DSN（或设好 DATABASE_URL）")
    return asyncio.run(
        run(
            args.source,
            args.target,
            dry_run=args.dry_run,
            truncate=args.truncate,
            migrate=not args.skip_migrate,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
