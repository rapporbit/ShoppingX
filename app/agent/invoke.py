"""对 AgentScope 模型做**一次性调用**的薄助手（批 0 / L6 起）。

不是所有 LLM 调用都跑在 Agent loop 里：评测判官、离线标注投票、记忆管家、偏好解析……这些是
「给一段 prompt、要一段回答」的单次调用。LangChain 侧有 ``llm.ainvoke(prompt)`` 这种一行写法，
AgentScope 侧则要自己处理三件事：

1. 入参得包成 ``list[Msg]``（模型层不吃裸字符串）；
2. 本仓的模型一律 ``stream=True``（主 loop 要流式），于是 ``__call__`` 返回的是异步生成器——
   基类会**累积**增量并在最后吐一个 ``is_last=True`` 的完整 ``ChatResponse``，所以取结果 =
   迭代到最后一个；半路 break 拿到的是残缺前缀（且会把网关闸门的 slot 留在生成器里，见
   ``gateway._stream_holding_slot``）；
3. 文本在 ``response.content`` 的 text block 里，不是 ``.text`` 属性。

三件事各处重写一遍迟早漂，收在这里。L7 迁 ``memory/curator`` / ``parser`` 那批结构化调用时，
在本模块加一个 ``call_structured``（走 ``model.generate_structured_output``）即可，不要再各写各的。
"""

from typing import Any

from agentscope.message import Msg, TextBlock


def _text_of(response: Any) -> str:
    """从 ``ChatResponse`` 取纯文本：拼所有 text block，丢 thinking / tool_use / 多媒体块。"""
    content = getattr(response, "content", None)
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content or []:
        btype = block.get("type") if isinstance(block, dict) else getattr(block, "type", None)
        if btype != "text":
            continue
        text = block.get("text") if isinstance(block, dict) else getattr(block, "text", None)
        if isinstance(text, str):
            parts.append(text)
    return "".join(parts)


async def call_text(model: Any, prompt: str) -> str:
    """给模型一段 prompt，拿回完整文本。流式 / 非流式两种模型都吃。

    **超时由调用方包**（``async with asyncio.timeout(...)``）：本函数只管把一次调用跑完整，
    「等多久算挂了」是各场景自己的事——离线批量标注等得起 150 秒，线上一次判分等不起。
    挂起的请求会占死并发槽让 ``gather`` 永不返回，批量脚本务必包上（M21 实测教训）。
    """
    result = await model(
        [Msg(name="user", role="user", content=[TextBlock(type="text", text=prompt)])]
    )
    if not hasattr(result, "__aiter__"):
        return _text_of(result)
    last: Any = None
    async for chunk in result:
        last = chunk
    return _text_of(last) if last is not None else ""
