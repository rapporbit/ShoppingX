"""S0 spike：钉死 12 个工具的返回形态与 ToolMiddleware 的写法。

手册 §6-L0 的 S0：同一业务函数分别写成「async return ToolChunk」与
「async generator yield ToolChunk」，各挂一个 ToolMiddlewareBase，
断言中间件前后钩子各执行一次、累积出的 ToolResponse 内容一致。

跑法：uv run python scripts/spikes/agentscope_s0_tool_shape.py
"""

import asyncio
import json
from collections.abc import AsyncGenerator, Callable
from typing import Any

from agentscope.message import TextBlock, ToolCallBlock
from agentscope.state import AgentState
from agentscope.tool import (
    FunctionTool,
    Toolkit,
    ToolMiddlewareBase,
    ToolResponse,
)
from agentscope.tool._base import ToolBase
from agentscope.tool._response import ToolChunk, ToolResultState

PROBE_LOG: list[str] = []


async def _impl(query: str) -> dict:
    """被两种壳共用的业务实现（对应本仓 _xxx_impl）。"""
    return {"query": query, "hits": 2}


async def tool_return(query: str) -> ToolChunk:
    """写法 A：async 函数直接 return 一个 ToolChunk。"""
    out = await _impl(query)
    return ToolChunk(
        content=[TextBlock(type="text", text=json.dumps(out))],
        state=ToolResultState.SUCCESS,
        metadata={"schema": "SpikeOutput"},
    )


async def tool_yield(query: str) -> AsyncGenerator[ToolChunk, None]:
    """写法 B：async generator 分两片 yield ToolChunk。"""
    out = await _impl(query)
    yield ToolChunk(
        content=[TextBlock(type="text", text=json.dumps(out)[:10])],
        state=ToolResultState.SUCCESS,
        is_last=False,
    )
    yield ToolChunk(
        content=[TextBlock(type="text", text=json.dumps(out))],
        state=ToolResultState.SUCCESS,
        metadata={"schema": "SpikeOutput"},
    )


class ProbeMiddleware(ToolMiddlewareBase):
    """洋葱式探针：next_handler 前后各记一笔，逐 chunk 转发。"""

    async def on_tool_call(
        self,
        tool: ToolBase,
        input_kwargs: dict[str, Any],
        next_handler: Callable[..., AsyncGenerator[ToolChunk, None]],
    ) -> AsyncGenerator[ToolChunk, None]:
        PROBE_LOG.append(f"pre:{tool.name}")
        async for chunk in next_handler(**input_kwargs):
            yield chunk
        PROBE_LOG.append(f"post:{tool.name}")


async def _run(toolkit: Toolkit, name: str) -> tuple[int, ToolResponse]:
    """走 Toolkit.call_tool（Agent 的真实路径），返回 chunk 数与终态。"""
    call = ToolCallBlock(
        type="tool_call",
        id=f"call_{name}",
        name=name,
        input=json.dumps({"query": "travel set"}),
    )
    chunks, final = 0, None
    async for item in toolkit.call_tool(call, AgentState()):
        if isinstance(item, ToolResponse):
            final = item
        else:
            chunks += 1
    return chunks, final


async def main() -> None:
    results = {}
    for name, func in (("tool_return", tool_return), ("tool_yield", tool_yield)):
        PROBE_LOG.clear()
        toolkit = Toolkit()
        await toolkit.add_tool(
            FunctionTool(
                func,
                name=name,
                is_read_only=True,
                middlewares=[ProbeMiddleware()],
            ),
        )
        chunks, final = await _run(toolkit, name)
        text = "".join(b.text for b in final.content if b.type == "text")
        results[name] = {
            "chunks": chunks,
            "probe": list(PROBE_LOG),
            "text": text,
            "state": final.state.value,
            "metadata": final.metadata,
        }
        assert PROBE_LOG == [f"pre:{name}", f"post:{name}"], PROBE_LOG

    # 无中间件时 __call__ 的返回类型差异（L4 适配器要知道）
    bare_ret = await FunctionTool(tool_return, name="bare_ret")(query="x")
    bare_gen = await FunctionTool(tool_yield, name="bare_gen")(query="x")
    results["_bare_call_types"] = {
        "return_shape": type(bare_ret).__name__,
        "yield_shape": type(bare_gen).__name__,
    }
    print(json.dumps(results, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(main())
