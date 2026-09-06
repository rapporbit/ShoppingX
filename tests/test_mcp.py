"""MCP 两侧（批 4-3）：生产侧 server 的只读契约 + 消费侧接线与读写切分三保证。

集成用例**真起一个本地 MCP server 进程**（自建汇率 server，纯静态表、零外部依赖），验的是
「Toolkit 真列得出、真调得通」。不打桩：打桩验的是我们自己写的假对象，而这条链路上最容易坏
的恰恰是框架与协议的接缝（工具名拼接、readOnlyHint 有没有传过来、无状态 HTTP 会不会要
connect）。整条链路不碰 LLM。
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import socket
import subprocess
import sys
import time
from collections.abc import Iterator

import pytest
from agentscope.mcp import HttpMCPConfig, MCPClient
from agentscope.message import ToolCallBlock
from agentscope.state import AgentState

from app.agent.mcp_registry import MCP_ROLES, mcp_clients, mcp_tool_names
from app.agent.tool_registry import _READ_ONLY_TOOLS, build_toolkit
from app.mcp.fx_server import FX_TOOL_NAMES
from app.mcp.server import EXPOSED_TOOL_NAMES

# ── 生产侧：开出去的必须全是只读，且写工具一个都不在 ────────────────────────────


async def test_exposed_tools_are_exactly_three_read_only() -> None:
    from app.mcp.server import mcp as produce_server

    tools = await produce_server.list_tools()
    assert tuple(sorted(t.name for t in tools)) == tuple(sorted(EXPOSED_TOOL_NAMES))
    for tool in tools:
        assert tool.annotations is not None, tool.name
        assert tool.annotations.readOnlyHint is True, tool.name


def test_exposed_names_are_a_subset_of_repo_read_only_set() -> None:
    """开出去的名字必须在本仓的只读集合里——server 与 Toolkit 用的是同一份只读口径。

    漏了这条，某天有人把 ``create_order`` 加进 ``EXPOSED_TOOL_NAMES`` 只会多一个 MCP 端点，
    没有任何一处会红：那台 server 没有两段式确认卡、没有 ``_order_guard``、没有顺序闸。
    """
    assert set(EXPOSED_TOOL_NAMES) <= _READ_ONLY_TOOLS


# ── 消费侧：发放范围与白名单（不需要起 server）──────────────────────────────────


def test_clients_are_empty_without_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MCP_SEARCH_URL", "")
    assert mcp_clients("search") == []
    assert mcp_tool_names("search") == []


def test_only_search_gets_mcp(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MCP_SEARCH_URL", "http://127.0.0.1:1/mcp")
    assert MCP_ROLES == frozenset({"search"})
    assert len(mcp_clients("search")) == 1
    for role in ("main", "trade"):
        assert mcp_clients(role) == []
        assert mcp_tool_names(role) == []


def test_enable_tools_whitelist_is_declarative(monkeypatch: pytest.MonkeyPatch) -> None:
    """白名单是**客户端侧**的：对端多开的工具进不来，与它自称什么无关。"""
    monkeypatch.setenv("MCP_SEARCH_URL", "http://127.0.0.1:1/mcp")
    monkeypatch.setenv("MCP_SEARCH_TOOLS", "convert_currency")
    (client,) = mcp_clients("search")
    assert client.enable_tools == ["convert_currency"]
    assert mcp_tool_names("search") == ["mcp__globex-fx__convert_currency"]


def test_mcp_tool_names_are_whitelisted(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.security.tool_whitelist import allowed_tools, validate_tool_call

    monkeypatch.setenv("MCP_SEARCH_URL", "http://127.0.0.1:1/mcp")
    allowed_tools.cache_clear()
    try:
        for name in mcp_tool_names("search"):
            assert validate_tool_call(name)
    finally:
        allowed_tools.cache_clear()


def test_client_is_stateless(monkeypatch: pytest.MonkeyPatch) -> None:
    """有状态 client 必须在 Toolkit 构造前 connect，而本仓的 Toolkit 是每个 loop 现建的。"""
    monkeypatch.setenv("MCP_SEARCH_URL", "http://127.0.0.1:1/mcp")
    (client,) = mcp_clients("search")
    assert client.is_stateful is False


# ── 集成：真起进程 ──────────────────────────────────────────────────────────────


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.fixture(scope="module")
def fx_server() -> Iterator[str]:
    """起一个本地汇率 MCP server，返回它的 ``/mcp`` URL。"""
    port = _free_port()
    proc = subprocess.Popen(
        [sys.executable, "-m", "app.mcp.fx_server", "--port", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    url = f"http://127.0.0.1:{port}/mcp"
    try:
        # 就绪判据必须是「MCP 握手成功」，不是「端口能连上」：uvicorn 先绑端口、再启
        # StreamableHTTP session manager，中间那段窗口对 /mcp 回的是 503，拿它当就绪会让
        # 后续用例在 Toolkit 里静默丢掉这个 server（框架对列表失败只 warning，不抛）。
        probe = MCPClient(
            name="probe", is_stateful=False, mcp_config=HttpMCPConfig(url=url, timeout=2.0)
        )
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                pytest.fail(f"fx MCP server 起不来，退出码 {proc.returncode}")
            with contextlib.suppress(Exception):
                if asyncio.run(probe.list_raw_tools()):
                    break
            time.sleep(0.3)
        else:
            pytest.fail("fx MCP server 30 秒内没握上手")
        yield url
    finally:
        proc.terminate()
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=10)


async def test_search_toolkit_lists_mcp_tools(
    fx_server: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MCP_SEARCH_URL", fx_server)
    toolkit = await build_toolkit("search")
    names = {s["function"]["name"] for s in await toolkit.get_tool_schemas()}
    for tool in FX_TOOL_NAMES:
        assert f"mcp__globex-fx__{tool}" in names


async def test_every_tool_in_search_group_is_read_only(
    fx_server: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """读写切分第②根支柱：MCP 进来之后，search 组仍然一个非只读工具都没有。

    ``MCPTool.is_read_only`` 取自对端的 ``annotations.readOnlyHint``，**取不到就是 False**——
    所以这条断言真的会因为 server 少写一个 annotation 而红，不是走过场。
    """
    monkeypatch.setenv("MCP_SEARCH_URL", fx_server)
    toolkit = await build_toolkit("search")
    available = await toolkit._get_available_tools(["basic"])
    assert any(rt.tool.is_mcp for rt in available.values())
    not_read_only = [name for name, rt in available.items() if not rt.tool.is_read_only]
    assert not_read_only == []


async def test_mcp_tool_is_callable_through_toolkit(
    fx_server: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MCP_SEARCH_URL", fx_server)
    toolkit = await build_toolkit("search")
    call = ToolCallBlock(
        type="tool_call",
        id="t1",
        name="mcp__globex-fx__convert_currency",
        input=json.dumps({"amount": 100, "from_currency": "EUR", "to_currency": "USD"}),
    )
    chunks = [chunk async for chunk in toolkit.call_tool(call, AgentState())]
    payload = json.loads(chunks[-1].content[0].text)
    assert payload["ok"] is True
    assert payload["amount"] == pytest.approx(108.0)


async def test_unknown_currency_comes_back_as_data_not_protocol_error(
    fx_server: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """工具内部的业务错误要回成模型读得懂的结构，不是一条「工具挂了」。"""
    monkeypatch.setenv("MCP_SEARCH_URL", fx_server)
    toolkit = await build_toolkit("search")
    call = ToolCallBlock(
        type="tool_call",
        id="t2",
        name="mcp__globex-fx__convert_currency",
        input=json.dumps({"amount": 10, "from_currency": "XYZ"}),
    )
    chunks = [chunk async for chunk in toolkit.call_tool(call, AgentState())]
    payload = json.loads(chunks[-1].content[0].text)
    assert payload["ok"] is False
    assert "USD" in payload["supported"]


async def test_main_and_trade_toolkits_have_no_mcp_tools(
    fx_server: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """发放范围（第①根支柱）：URL 配着也一样，main / trade 的 Toolkit 里根本没有这个 client。"""
    monkeypatch.setenv("MCP_SEARCH_URL", fx_server)
    for role in ("main", "trade"):
        toolkit = await build_toolkit(role)
        available = await toolkit._get_available_tools(["basic"])
        assert not any(rt.tool.is_mcp for rt in available.values()), role
