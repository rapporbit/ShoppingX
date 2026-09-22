"""user is_guest —— 免登录试用账号标记

``users`` 表加一位 ``is_guest``（默认 false）。访客账号由 ``POST /api/auth/guest`` 签发，
是一行真实 users 记录：会话归属 / credit 配额 / WS 校验全部按普通 user_id 走，本列只决定
「日额度取哪一档」与「注册升级时原地翻回 false」。

**为什么是加列而不是靠用户名前缀判断。** ``guest_<hex>`` 的前缀在升级成正式账号后就没了，
而升级恰恰要求 id 不变（试用期数据全留）；用一列标记，翻一下就完，不用改任何外部引用。

存量行全部落到 ``false``（都是注册用户），故 ``server_default`` 必须给：MySQL 对 NOT NULL
无默认值的加列会拒绝（表非空时）。downgrade 直接删列，访客标记丢失即所有访客变成「正式
账号」——只影响额度档位，不影响数据归属，可接受。

Revision ID: 0017_user_is_guest
Revises: 0016_drop_preferences
Create Date: 2026-09-22
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0017_user_is_guest"
down_revision: str | None = "0016_drop_preferences"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    cols = {c["name"] for c in sa.inspect(op.get_bind()).get_columns("users")}
    if "is_guest" in cols:
        return
    with op.batch_alter_table("users") as batch:
        batch.add_column(
            sa.Column("is_guest", sa.Boolean(), nullable=False, server_default=sa.false())
        )


def downgrade() -> None:
    cols = {c["name"] for c in sa.inspect(op.get_bind()).get_columns("users")}
    if "is_guest" not in cols:
        return
    with op.batch_alter_table("users") as batch:
        batch.drop_column("is_guest")
