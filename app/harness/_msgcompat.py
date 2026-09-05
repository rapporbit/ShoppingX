"""消息类型的双运行时兼容层（批 0 迁移期）。

Hook 是控制面，本不该知道底下跑的是 LangChain 还是 AgentScope。可有三个 Hook 绕不开消息
类型——``watchdog`` / ``context_compress`` 要往消息流里追加一条 system 提示，
``terminal_enforce`` 要判断「本轮是否已调过终结工具」。迁移期两套消息类型并存，与其让这三个
文件各写一遍 if-else，不如把差异**收进这一个文件**：L8 摘掉 LangChain 时，删的是这里的
一个分支，三个 Hook 一行不动。

运行时由适配器写进 context 的 ``_runtime`` 键标明（缺省按 LangChain，因为迁移期它仍是主链路）。
"""

from __future__ import annotations

from typing import Any

RUNTIME_KEY = "_runtime"
RUNTIME_AGENTSCOPE = "agentscope"
RUNTIME_LANGCHAIN = "langchain"


def runtime_of(context: dict[str, Any]) -> str:
    """本次 Hook 跑在哪个运行时上。"""
    value = context.get(RUNTIME_KEY)
    return value if isinstance(value, str) else RUNTIME_LANGCHAIN


def system_message(text: str, context: dict[str, Any]) -> Any:
    """造一条「系统提示」消息，类型随当前运行时。"""
    if runtime_of(context) == RUNTIME_AGENTSCOPE:
        from agentscope.message import Msg, TextBlock

        return Msg(name="system", role="system", content=[TextBlock(type="text", text=text)])
    from langchain_core.messages import SystemMessage

    return SystemMessage(content=text)


def has_tool_result_from(messages: Any, names: frozenset[str] | set[str]) -> bool:
    """消息历史里是否出现过 ``names`` 中某个工具的**执行结果**。

    两套运行时的形态不同：LangChain 是一条 ``ToolMessage``（``.name`` 就是工具名）；
    AgentScope 是 ``Msg.content`` 里的 ``ToolResultBlock``（工具名在 block 的 ``name`` 上）。
    判据都是「结果回来了」而不是「模型说要调」——模型说了没做不算数。
    """
    if not messages:
        return False
    for msg in messages:
        # LangChain：ToolMessage 顶层就带 name
        if type(msg).__name__ == "ToolMessage" and getattr(msg, "name", None) in names:
            return True
        content = getattr(msg, "content", None)
        if not isinstance(content, list):
            continue
        for block in content:
            block_type = getattr(block, "type", None) or (
                block.get("type") if isinstance(block, dict) else None
            )
            if block_type != "tool_result":
                continue
            block_name = getattr(block, "name", None) or (
                block.get("name") if isinstance(block, dict) else None
            )
            if block_name in names:
                return True
    return False
