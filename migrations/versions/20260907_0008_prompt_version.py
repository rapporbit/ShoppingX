"""usage_ledger 加 prompt_version（批 4 / 18-3 提示词 A/B）

Revision ID: 0008_prompt_version
Revises: 0007_orders
Create Date: 2026-09-07 06:04:05.762316

autogenerate 的产物**改了一处**：加 ``server_default=""``。原样跑会在有数据的老库上炸——
批处理模式重建表、把老行拷进来，新列拷进去是 NULL，正撞 NOT NULL（SQLite 报
"Cannot add a NOT NULL column with default value NULL"）。本地测试库每次重建、行数为 0，
所以「不加也绿」，线上那份老库才现原形。默认值取空串而非某个版本号：老账目本就产生在
分版本之前，硬安一个版本号是**编造归因**，空串诚实地表示「不知道」。
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0008_prompt_version"
down_revision: str | None = "0007_orders"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("usage_ledger", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column("prompt_version", sa.String(length=16), nullable=False, server_default="")
        )


def downgrade() -> None:
    with op.batch_alter_table("usage_ledger", schema=None) as batch_op:
        batch_op.drop_column("prompt_version")
