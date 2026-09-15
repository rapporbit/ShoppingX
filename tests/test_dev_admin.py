"""本地调试默认管理员：启动时按 DEV_ADMIN_* 建号（幂等、不重置密码），并算进管理员白名单。"""

from __future__ import annotations

from typing import Any

import pytest

from app.api.admin import admin_usernames, dev_admin_username
from app.db.accounts import authenticate, ensure_dev_admin
from app.db.session import session_factory

pytestmark = pytest.mark.anyio


async def test_ensure_dev_admin_creates_once_and_keeps_password() -> None:
    """首次建号；再调一次（哪怕换了密码）不新建、也不改掉原密码——重启不该把人踢出去。"""
    async with session_factory()() as db:
        assert await ensure_dev_admin(db, "dev-admin-t1", "first-pass-123") is True
    async with session_factory()() as db:
        assert await ensure_dev_admin(db, "dev-admin-t1", "other-pass-456") is False
    async with session_factory()() as db:
        assert await authenticate(db, "dev-admin-t1", "first-pass-123") is not None
        assert await authenticate(db, "dev-admin-t1", "other-pass-456") is None


async def test_dev_admin_counts_as_admin_only_with_password(monkeypatch: Any) -> None:
    """只配用户名不配密码不算数（不会建号，也就不该凭空多出一个管理员）。"""
    monkeypatch.setenv("ADMIN_USERNAMES", "zjl")
    monkeypatch.setenv("DEV_ADMIN_USERNAME", "admin")
    monkeypatch.delenv("DEV_ADMIN_PASSWORD", raising=False)
    assert dev_admin_username() == ""
    assert admin_usernames() == {"zjl"}

    monkeypatch.setenv("DEV_ADMIN_PASSWORD", "admin1234")
    assert admin_usernames() == {"zjl", "admin"}
