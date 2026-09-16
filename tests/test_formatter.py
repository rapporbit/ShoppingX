"""formatter 的 cache_control 标记（上下文压缩已交框架，本仓只剩这一层缓存治理）。

三个容易失效的点：

- 标记只落在 system 那一条：位置与形态都不随轮次动，前缀才是逐字稳定的。
- ``role="tool"`` 的条目不许被改成 block 列表（OpenAI 兼容端点要求它的 content 是字符串）。
- system 段不足最小写入阈值不打标记——写了也不会被缓存，白占一个额度。
"""

import json

import pytest
from agentscope.message import Msg, TextBlock, ToolCallBlock, ToolResultBlock

from app.harness.formatter import MIN_CACHE_PREFIX_TOKENS, CacheAwareOpenAIFormatter
from app.utils.tokens import count_tokens

_UNIT = "想买便宜又抗造的旅行三件套，预算300，不要塑料的，喜欢小众品牌。"


def _payload(min_tokens: int) -> str:
    s = _UNIT
    while count_tokens(s) < min_tokens:
        s += _UNIT
    return s


def _turn(n_tools: int, payload: str) -> list[Msg]:
    """造一轮完整交互：一条 user Msg + 一条把 n_tools 次工具调用全塞进去的 assistant Msg。"""
    content: list = []
    for i in range(n_tools):
        content.extend(
            [
                ToolCallBlock(type="tool_call", id=f"c{i}", name="item_search", input="{}"),
                ToolResultBlock(
                    type="tool_result",
                    id=f"c{i}",
                    name="item_search",
                    output=[TextBlock(type="text", text=payload)],
                ),
            ]
        )
    return [
        Msg(name="user", role="user", content=[TextBlock(type="text", text="需求")]),
        Msg(name="shoppingx", role="assistant", content=content),
    ]


def _system(text: str) -> Msg:
    return Msg(name="system", role="system", content=[TextBlock(type="text", text=text)])


@pytest.mark.asyncio
async def test_formatter_marks_only_system() -> None:
    big = _payload(MIN_CACHE_PREFIX_TOKENS + 500)
    formatted = await CacheAwareOpenAIFormatter().format([_system(big), *_turn(4, _UNIT)])
    marked = [
        i
        for i, e in enumerate(formatted)
        if isinstance(e.get("content"), list)
        and any(isinstance(b, dict) and "cache_control" in b for b in e["content"])
    ]
    assert marked == [0]
    assert all(isinstance(e["content"], str) for e in formatted if e.get("role") == "tool")


@pytest.mark.asyncio
async def test_formatter_marker_position_stable_across_turns() -> None:
    """再来几轮工具调用，标记仍钉在同一条上，且 system 之后的条目字节不受影响。"""
    system = _system(_payload(MIN_CACHE_PREFIX_TOKENS + 500))
    fmt = CacheAwareOpenAIFormatter()
    short = await fmt.format([system, *_turn(2, _UNIT)])
    long = await fmt.format([system, *_turn(6, _UNIT)])
    assert short[0] == long[0]
    assert all("cache_control" not in json.dumps(e, default=str) for e in long[1:])


@pytest.mark.asyncio
async def test_formatter_skips_when_prefix_too_short() -> None:
    msgs = [
        _system("短"),
        Msg(name="user", role="user", content=[TextBlock(type="text", text="短")]),
    ]
    formatted = await CacheAwareOpenAIFormatter().format(msgs)
    assert all("cache_control" not in json.dumps(e, default=str) for e in formatted)
