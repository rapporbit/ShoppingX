"""user_skills 表（买家个人 Skill）

Revision ID: 0010_user_skills
Revises: 0009_strategies
Create Date: 2026-09-14

建新表，老库无需回填；nullable=False 的列由 ORM 侧 default 兜住。
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0010_user_skills"
down_revision: str | None = "0009_strategies"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "user_skills",
        sa.Column("id", sa.String(length=32), nullable=False),
        sa.Column("user_id", sa.String(length=64), nullable=False),
        sa.Column("name", sa.String(length=64), nullable=False),
        sa.Column("description", sa.String(length=400), nullable=False),
        sa.Column("body", sa.Text(), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("active", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("user_id", "name", name="uq_user_skill_name"),
    )
    with op.batch_alter_table("user_skills", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_user_skills_user_id"), ["user_id"], unique=False)
        batch_op.create_index(batch_op.f("ix_user_skills_active"), ["active"], unique=False)


def downgrade() -> None:
    with op.batch_alter_table("user_skills", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_user_skills_active"))
        batch_op.drop_index(batch_op.f("ix_user_skills_user_id"))
    op.drop_table("user_skills")
