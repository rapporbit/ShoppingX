"""trade_confirmations 加 request_key（写工具的请求级幂等键）

阶段 3：整轮重跑（队列消息被 PEL 重投、worker 崩了重领）时 run_id 不变，同一轮再调一次
``create_order`` 该复用原来那张确认卡，而不是出第二张让用户不知道点哪个。键的形状是
``run_id:action:快照 hash``，见 :func:`app.trade.confirmation.request_key`。

**可空 + 唯一**：HTTP 表单入口与离线脚本没有 run 作用域，写 NULL；SQLite 与 MySQL 的唯一索引
都允许多行 NULL，所以这两条路的行为和加这道之前完全一致。老数据同理全是 NULL，不用回填。

加列 + 建唯一索引在 SQLite 上不必重建表（``ALTER TABLE ADD COLUMN`` 与 ``CREATE UNIQUE INDEX``
它都支持），两边走同一条路。

Revision ID: 0015_confirmation_request_key
Revises: 0014_thread_active_run
Create Date: 2026-09-21
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0015_confirmation_request_key"
down_revision: str | None = "0014_thread_active_run"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TABLE = "trade_confirmations"
_COLUMN = "request_key"
_INDEX = "ix_trade_confirmations_request_key"


def upgrade() -> None:
    op.add_column(_TABLE, sa.Column(_COLUMN, sa.String(length=128), nullable=True))
    op.create_index(_INDEX, _TABLE, [_COLUMN], unique=True)


def downgrade() -> None:
    op.drop_index(_INDEX, table_name=_TABLE)
    # 走 batch：SQLite 直到 3.35 才有 DROP COLUMN，batch 会在老版本上退回「重建表」那条路。
    with op.batch_alter_table(_TABLE) as batch_op:
        batch_op.drop_column(_COLUMN)
