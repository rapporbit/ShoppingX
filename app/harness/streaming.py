"""一次模型调用的流式侧：转发 chunk、推 summary 增量、流结束时计费。

从 ``HarnessAgentAdapter`` 抽出来——它们是「模型调用期间的旁路动作」，不是 hook_point 到
AgentScope 中间件的桥接；留在 adapter 里会让 ``on_model_call`` 同时管档位 / 注入 / 计费 / 推流。
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING

from agentscope.model import ChatResponse

from app.api import monitor
from app.harness.token_budget import charge_usage

if TYPE_CHECKING:  # pragma: no cover
    from app.harness.session import HarnessSession


async def stream_summary_delta(chunk: ChatResponse, emitted: int) -> int:
    """主模型流式吐 ``shopping_summary`` 入参时，把 summary 的累计文本推给前端（round3 刀 1）。

    收尾文案改由主模型在工具入参里写之后，原来收尾工具内部那条 summary_delta 流没了；chunk 是
    累积快照、``ToolCallBlock.input`` 在流式期间是累积的原始 JSON 串，从里面抠 summary 即可。
    解不出就跳过本 tick，最终产物不受影响。
    """
    for block in getattr(chunk, "content", None) or []:
        if getattr(block, "type", None) != "tool_call" or getattr(block, "name", "") != (
            "shopping_summary"
        ):
            continue
        raw = getattr(block, "input", None)
        if not isinstance(raw, str):
            continue
        from app.tools.shopping_summary import _DELTA_MIN_CHARS, _partial_summary

        text = _partial_summary(raw)
        if len(text) >= emitted + _DELTA_MIN_CHARS:
            await monitor.report_summary_delta(text)
            return len(text)
    return emitted


async def charge_stream(
    stream: AsyncGenerator[ChatResponse, None], model_name: str, session: HarnessSession
) -> AsyncGenerator[ChatResponse, None]:
    """转发流式响应，并在流结束时把这次调用的用量计进全树。

    **只认最后一个 chunk 的 usage**：chunk 是累积快照（基类把增量攒好再吐），逐个入账
    会把同一次调用重复计上十几遍。放 finally 是因为半路取消时 token 也已真实花掉——
    少算的账会让预算闸和用户 credit 配额一起失真。AgentScope 没有「一次模型调用结束」的
    钩子，入账只能接在这里。
    """
    last: ChatResponse | None = None
    emitted = 0
    try:
        async for chunk in stream:
            last = chunk
            emitted = await stream_summary_delta(chunk, emitted)
            yield chunk
    finally:
        charge_usage(model_name, getattr(last, "usage", None))
        session.track_token_delta()
