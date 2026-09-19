"""threads 加 active_run_id / run_status / active_query，并去掉 user_id 外键

阶段 1-2：「同 thread 还有没有任务在跑」从 API 进程内的 ``active_tasks`` 字典搬进 DB，靠一条
条件更新认定（见 :class:`app.db.models.Thread`）。外键必须去掉——鉴权关闭的 demo 模式下
``user_id`` 是不在 users 表里的假身份，带外键就登记不进来，唯一真相又会退回进程内。

**SQLite 去外键只能重建表**（它没有 ``ALTER TABLE … DROP CONSTRAINT``），所以走
``batch_alter_table(recreate="always", copy_from=…)``。**坑在 ``copy_from`` 的语义**：新表的
DDL 就是照它生成的（数据按列名拷），所以它描述的不是「旧结构」而是「重建出来的结构」——传一份
带外键的定义进去，重建完外键原样还在（实测过）。因此 upgrade 传**不含外键**的那份，downgrade
反过来传含外键的那份。MySQL 侧简单：反射出外键名逐个 drop（名字是自动生成的 ``threads_ibfk_1``，
写死会在别的库上对不上）。

Revision ID: 0014_thread_active_run
Revises: 0013_run_holds
Create Date: 2026-09-19
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0014_thread_active_run"
down_revision: str | None = "0013_run_holds"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

#: 新列名（只用于 drop 与断言，建列走 :func:`_new_columns`）。
_NEW_COLUMN_NAMES = ("active_run_id", "run_status", "active_query")


def _new_columns() -> list[sa.Column]:
    """每次**新建**一组 Column 对象——一个 Column 只能属于一张 Table，复用会在第二次绑定时炸
    （SQLAlchemy 2.0 起 ``Column.copy()`` 已移除，没有「复制一份」这条路）。"""
    return [
        sa.Column("active_run_id", sa.String(length=64), nullable=True),
        sa.Column("run_status", sa.String(length=16), nullable=False, server_default="idle"),
        sa.Column("active_query", sa.String(length=500), nullable=False, server_default=""),
    ]


def _threads(*, with_fk: bool, with_new_columns: bool) -> sa.Table:
    """一份手写的 threads 结构，给 SQLite 的 batch 重建当 ``copy_from``。

    手写而不让 batch 去反射，是因为反射会把外键一起带进新表——那正是这次要去掉的东西。
    索引也必须写在这里：重建走的是「建新表 → 拷数据 → 换名」，``copy_from`` 里没有的索引重建后
    就没了（侧栏按 user_id 查会话是本表最热的路径，丢了它等于每次全表扫）。
    """
    md = sa.MetaData()
    cols: list[sa.Column] = [
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("user_id", sa.String(length=32), nullable=False),
        sa.Column("title", sa.String(length=200), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    ]
    if with_new_columns:
        cols.extend(_new_columns())
    constraints: list[sa.schema.SchemaItem] = [sa.PrimaryKeyConstraint("id")]
    if with_fk:
        constraints.append(sa.ForeignKeyConstraint(["user_id"], ["users.id"]))
    table = sa.Table("threads", md, *cols, *constraints)
    sa.Index("ix_threads_user_id", table.c.user_id)
    return table


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "sqlite":
        with op.batch_alter_table(
            "threads",
            schema=None,
            recreate="always",
            # 不含外键 = 重建后的样子（见模块 docstring 里那个坑）。
            copy_from=_threads(with_fk=False, with_new_columns=False),
        ) as batch_op:
            for col in _new_columns():
                batch_op.add_column(col)
        return

    for fk in sa.inspect(bind).get_foreign_keys("threads"):
        if fk.get("name"):
            op.drop_constraint(fk["name"], "threads", type_="foreignkey")
    for col in _new_columns():
        op.add_column("threads", col)


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "sqlite":
        # 回到「带外键」的旧样子：copy_from 带 fk + 带新列（新列由下面的 drop_column 去掉）。
        with op.batch_alter_table(
            "threads",
            schema=None,
            recreate="always",
            copy_from=_threads(with_fk=True, with_new_columns=True),
        ) as batch_op:
            for name in _NEW_COLUMN_NAMES:
                batch_op.drop_column(name)
        return

    for name in _NEW_COLUMN_NAMES:
        op.drop_column("threads", name)
    op.create_foreign_key(None, "threads", "users", ["user_id"], ["id"])
