"""strategies 表（批 4 / 18-4 记忆成功策略沉淀）

Revision ID: 0009_strategies
Revises: 0008_prompt_version
Create Date: 2026-09-07 06:19:54.555255

autogenerate 的产物**原样保留**（只改了 revision id 与本段说明）。0008 那处「必须加
server_default」的坑在这里不存在：这是**建新表**不是给老表加列，老库里没有需要回填的行。
建表语句里所有 ``nullable=False`` 都由 ORM 侧的 ``default=`` 在写入时兜住。
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0009_strategies"
down_revision: str | None = "0008_prompt_version"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "strategies",
        sa.Column("id", sa.String(length=32), nullable=False),
        sa.Column("dedup_key", sa.String(length=200), nullable=False),
        sa.Column("category", sa.String(length=64), nullable=False),
        sa.Column("slug", sa.String(length=64), nullable=False),
        sa.Column("trigger", sa.String(length=300), nullable=False),
        sa.Column("trigger_keywords", sa.JSON(), nullable=False),
        sa.Column("actions", sa.JSON(), nullable=False),
        sa.Column("evidence", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("health", sa.Integer(), nullable=False),
        sa.Column("hits", sa.Integer(), nullable=False),
        sa.Column("consecutive_failures", sa.Integer(), nullable=False),
        sa.Column("source_report", sa.String(length=200), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("dedup_key", name="uq_strategy_key"),
    )
    with op.batch_alter_table("strategies", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_strategies_category"), ["category"], unique=False)
        batch_op.create_index(batch_op.f("ix_strategies_status"), ["status"], unique=False)


def downgrade() -> None:
    with op.batch_alter_table("strategies", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_strategies_status"))
        batch_op.drop_index(batch_op.f("ix_strategies_category"))
    op.drop_table("strategies")
