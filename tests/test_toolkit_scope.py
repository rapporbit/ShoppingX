"""Toolkit 发放范围的结构性不变量。

替代 2026-09-15 删掉的 depth_gate / tool_whitelist 两道运行时断言。

worker 的边界靠「Toolkit 里根本没有那个工具」保证，不靠运行时看门狗。这里在装配层把不变量钉死：
将来有人往 ``_SEARCH_TOOLS`` / ``_TRADE_TOOLS`` 加错工具，跑测试即红，而不是等线上日志。
"""

from __future__ import annotations

import pytest

from app.agent.tool_registry import TOOLS, build_toolkit
from app.harness.budgets import DEPTH0_ONLY_TOOLS, FORK_TOOLS, MAIN_ONLY_CONTEXT_TOOLS
from app.security.tool_whitelist import allowed_tools

_MAIN_ONLY = DEPTH0_ONLY_TOOLS | MAIN_ONLY_CONTEXT_TOOLS | FORK_TOOLS
_BY_NAME = {t.name: t for t in TOOLS}


async def _names(role: str) -> frozenset[str]:
    toolkit = await build_toolkit(role)
    return frozenset(s["function"]["name"] for s in await toolkit.get_tool_schemas())


@pytest.mark.asyncio
@pytest.mark.parametrize("role", ["search", "trade"])
async def test_worker_toolkit_has_no_main_only_tools(role: str) -> None:
    leaked = await _names(role) & _MAIN_ONLY
    assert not leaked, f"{role} worker 拿到了主 loop 专属工具：{sorted(leaked)}"


@pytest.mark.asyncio
async def test_search_worker_is_read_only() -> None:
    names = await _names("search")
    writable = [n for n in names if n in _BY_NAME and not _BY_NAME[n].is_read_only]
    assert not writable, f"SearchAgent 拿到了非只读工具：{writable}"


@pytest.mark.asyncio
async def test_all_issued_tools_are_whitelisted() -> None:
    for role in ("main", "search", "trade"):
        missing = await _names(role) - allowed_tools()
        assert not missing, f"{role} 发放了白名单外的工具：{sorted(missing)}"
