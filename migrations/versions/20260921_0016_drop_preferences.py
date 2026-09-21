"""drop preferences —— 旧长期偏好表退场

长期记忆在 0012 换成了 ``memory_facts``（user_id + fact_key 唯一，同 key 覆盖），旧
``preferences`` 表当时留了一版作回滚依据。到今天它已经**没有任何代码读写**（搬数据的
``scripts/migrate_preferences_to_facts.py`` 随本次减法一并删掉），留着只是让人误以为
还有第二条长期记忆路径。这条迁移把它摘掉。

**downgrade 只恢复结构，恢复不了数据。** 建表定义原样照抄 0002，回滚后是一张空表——真要
找回旧偏好，得从库备份里捞。

**两边都成立：** 表没有外键（0002 起三张用户级表的 user_id 就只建 index 不加 FK），所以
不存在「先删子表还是先删父表」的顺序问题。索引随 ``DROP TABLE`` 一起消失，SQLite 与 MySQL
都是；这里仍先显式 ``DROP INDEX``，是为了让「表没了索引还在」这种半拉状态不可能出现（部分
MySQL online DDL 工具会把两步拆开跑）。删之前先问 inspector，表 / 索引不在就跳过，重复执行
不炸。

Revision ID: 0016_drop_preferences
Revises: 0015_confirmation_request_key
Create Date: 2026-09-21
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0016_drop_preferences"
down_revision: str | None = "0015_confirmation_request_key"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TABLE = "preferences"
_INDEX = "ix_preferences_user_id"


def upgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    if _TABLE not in inspector.get_table_names():
        return
    if _INDEX in {ix["name"] for ix in inspector.get_indexes(_TABLE)}:
        op.drop_index(_INDEX, table_name=_TABLE)
    op.drop_table(_TABLE)


def downgrade() -> None:
    """按 0002 的定义原样建回结构（**数据回不来**，回滚后是空表）。"""
    op.create_table(
        _TABLE,
        sa.Column("id", sa.String(length=32), nullable=False),
        sa.Column("user_id", sa.String(length=64), nullable=False),
        sa.Column("dedup_key", sa.String(length=200), nullable=False),
        sa.Column("polarity", sa.String(length=16), nullable=False),
        sa.Column("category", sa.String(length=32), nullable=False),
        sa.Column("domain", sa.String(length=32), nullable=False),
        sa.Column("slug", sa.String(length=64), nullable=False),
        sa.Column("content", sa.String(length=500), nullable=False),
        sa.Column("keywords", sa.JSON(), nullable=False),
        sa.Column("source", sa.String(length=16), nullable=False),
        sa.Column("blocking", sa.Boolean(), nullable=False),
        sa.Column("source_session", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_confirmed_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("user_id", "dedup_key", name="uq_pref_user_key"),
    )
    op.create_index(_INDEX, _TABLE, ["user_id"], unique=False)
