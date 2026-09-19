"""run_holds

credit 预授权表（阶段 1-1）：进门按档位预扣、跑完按真实用量结算，``run_id`` 作幂等键。
纯新增一张表，不动任何既有表 —— 回滚就是 drop。

Revision ID: 0013_run_holds
Revises: 0012_memory_facts
Create Date: 2026-09-19
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = '0013_run_holds'
down_revision: str | None = '0012_memory_facts'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        'run_holds',
        sa.Column('run_id', sa.String(length=64), nullable=False),
        sa.Column('user_id', sa.String(length=64), nullable=False),
        sa.Column('thread_id', sa.String(length=64), nullable=False),
        sa.Column('kind', sa.String(length=16), nullable=False),
        sa.Column('credits_held', sa.Integer(), nullable=False),
        sa.Column('credits_charged', sa.Integer(), nullable=False),
        sa.Column('state', sa.String(length=16), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('settled_at', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('run_id'),
    )
    with op.batch_alter_table('run_holds', schema=None) as batch_op:
        # 热查询只有一条：「这个人有几行还活着」。两列各建一个索引即可覆盖
        # （MySQL 的索引合并 / SQLite 的单索引 + 回表都够用，行数本就极小）。
        batch_op.create_index(batch_op.f('ix_run_holds_user_id'), ['user_id'], unique=False)
        batch_op.create_index(batch_op.f('ix_run_holds_state'), ['state'], unique=False)


def downgrade() -> None:
    with op.batch_alter_table('run_holds', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_run_holds_state'))
        batch_op.drop_index(batch_op.f('ix_run_holds_user_id'))

    op.drop_table('run_holds')
