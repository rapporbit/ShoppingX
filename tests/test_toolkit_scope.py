"""Toolkit 发放范围的结构性不变量。

A4 起只剩主 Agent 一个角色（SearchAgent / task_dispatch 已删）。这里钉死两件事：
发放的每个工具都在白名单里；已删的角色与派发工具不会被悄悄加回来。
"""

from __future__ import annotations

import pytest

from app.agent.tool_registry import build_toolkit
from app.security.tool_whitelist import allowed_tools


async def _names(role: str) -> frozenset[str]:
    toolkit = await build_toolkit(role)
    return frozenset(s["function"]["name"] for s in await toolkit.get_tool_schemas())


@pytest.mark.asyncio
async def test_all_issued_tools_are_whitelisted() -> None:
    missing = await _names("main") - allowed_tools()
    assert not missing, f"main 发放了白名单外的工具：{sorted(missing)}"


@pytest.mark.asyncio
async def test_no_dispatch_tool_and_no_worker_role() -> None:
    assert "task_dispatch" not in await _names("main")
    assert "task_dispatch" not in allowed_tools()
    with pytest.raises(ValueError):
        await build_toolkit("search")
