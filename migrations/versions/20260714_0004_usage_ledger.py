"""usage ledger

Revision ID: 27dff157193c
Revises: 0003_messages
Create Date: 2026-07-14 12:02:42.902817
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = '0004_usage_ledger'
down_revision: str | None = '0003_messages'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table('usage_ledger',
    sa.Column('id', sa.String(length=32), nullable=False),
    sa.Column('user_id', sa.String(length=64), nullable=False),
    sa.Column('period_key', sa.String(length=16), nullable=False),
    sa.Column('cost_usd', sa.Float(), nullable=False),
    sa.Column('input_tokens', sa.Integer(), nullable=False),
    sa.Column('output_tokens', sa.Integer(), nullable=False),
    sa.Column('task_count', sa.Integer(), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('user_id', 'period_key', name='uq_usage_user_period')
    )
    with op.batch_alter_table('usage_ledger', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_usage_ledger_user_id'), ['user_id'], unique=False)



def downgrade() -> None:
    with op.batch_alter_table('usage_ledger', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_usage_ledger_user_id'))

    op.drop_table('usage_ledger')
