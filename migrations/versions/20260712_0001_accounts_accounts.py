"""accounts：users + threads（基线）

**这条是「基线」，语义特殊。** 它建的表结构与 M16 用 ``create_all`` 建出来的**完全等价**。因此线上
那份老库（有表、无 ``alembic_version``）不需要执行它，只要 ``stamp`` 到这个 revision 即可——
``app/db/session.py`` 的 ``init_db()`` 自动完成这件事，见 ``BASELINE_REVISION``。
**改这条迁移 = 改老库的历史，别改**；结构要演进就往后加新的 revision。

Revision ID: 0001_accounts
Revises:
Create Date: 2026-07-12
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0001_accounts"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.String(length=32), nullable=False),
        sa.Column("username", sa.String(length=64), nullable=False),
        sa.Column("password_hash", sa.String(length=128), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    with op.batch_alter_table("users", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_users_username"), ["username"], unique=True)

    op.create_table(
        "threads",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("user_id", sa.String(length=32), nullable=False),
        sa.Column("title", sa.String(length=200), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    with op.batch_alter_table("threads", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_threads_user_id"), ["user_id"], unique=False)


def downgrade() -> None:
    with op.batch_alter_table("threads", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_threads_user_id"))

    op.drop_table("threads")
    with op.batch_alter_table("users", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_users_username"))

    op.drop_table("users")
