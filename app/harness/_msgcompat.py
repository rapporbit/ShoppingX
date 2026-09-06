"""Hook 侧要用的两个消息小工具。

Hook 是控制面，本不该操心消息的具体类型；可有三个 Hook 绕不开——``watchdog`` /
``context_compress`` 要往消息流里追加一条系统提示，``terminal_enforce`` 要判断「本轮是否已调过
终结工具」。把这点类型知识收在这一个文件里，Hook 那边就只调函数。
"""

from __future__ import annotations

from typing import Any

from agentscope.message import Msg, TextBlock


def system_message(text: str, context: dict[str, Any] | None = None) -> Msg:
    """造一条「系统提示」消息。

    ``context`` 保留在签名里只为调用点写法统一（Hook 手上永远有它），本身不参与判断。
    """
    return Msg(name="system", role="system", content=[TextBlock(type="text", text=text)])


def has_tool_result_from(messages: Any, names: frozenset[str] | set[str]) -> bool:
    """消息历史里是否出现过 ``names`` 中某个工具的**执行结果**。

    一整轮的 tool_call / tool_result 都在同一条 assistant ``Msg`` 的 content 里，所以要下钻到
    block 看 ``type == "tool_result"``。判据是「结果回来了」而不是「模型说要调」——模型说了没做
    不算数（终结纪律靠这一点区分「真收尾」与「嘴上收尾」）。
    """
    if not messages:
        return False
    for msg in messages:
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
