"""消息形态的小工具：**「怎么从一条 Msg 里把东西掏出来」的知识只住在这一个文件**。

原本是给 Hook 用的两个函数（Hook 是控制面，本不该操心消息的具体类型），现在也供适配器与
编排层用——因为同一份掏法此前散在四处各写一遍（审查报告 P1-1 / P1-2）：``adapter`` 与
``orchestrator`` 各有一份「倒着抠最后一次 shopping_summary」，连「解析失败要继续往前找」的
坑注释都各抄一遍；``_text_of`` / ``_block_text`` 有四份。一处修 bug 另一处必漏。

**刻意没有并进来的一份**：``compress/blocks._block_text``。它长得像，语义不同——返回 ``None``
表示「这块不是纯文本，压缩层一律不碰」，而这里的 ``block_text`` 返回空串表示「没文字」。
把「不碰」和「没文字」合并成同一个返回值，压缩层就会把图片块 ``str()`` 化后截断。**看着重复
不等于是重复。**
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

from agentscope.message import Msg, TextBlock, ToolCallBlock, ToolResultBlock

from app.utils.tokens import count_tokens


def system_message(text: str, context: dict[str, Any] | None = None) -> Msg:
    """造一条「系统提示」消息。

    ``context`` 保留在签名里只为调用点写法统一（Hook 手上永远有它），本身不参与判断。
    """
    return Msg(name="system", role="system", content=[TextBlock(type="text", text=text)])


def _attr(block: Any, key: str) -> Any:
    """block 可能是 pydantic 对象也可能是 dict（不同运行时 / 不同来源），统一取字段。"""
    if isinstance(block, dict):
        return block.get(key)
    return getattr(block, key, None)


def block_text(block: Any) -> str:
    """content block → 文本。非文本块给空串（不是 None，见模块 docstring 那段「刻意没并」）。"""
    text = _attr(block, "text")
    return text if isinstance(text, str) else ""


def text_of(obj: Any) -> str:
    """``Msg`` / ``ChatResponse`` / ``None`` → 纯文本：拼所有 text block。

    丢掉 thinking / tool_use / 多媒体块——调用方要的都是「给人看的那段字」。``content`` 已是
    字符串时原样返回（有的供应商壳会这么给）。
    """
    if obj is None:
        return ""
    content = getattr(obj, "content", None)
    if isinstance(content, str):
        return content
    return "".join(block_text(b) for b in content or [] if _attr(b, "type") == "text")


def iter_tool_results(messages: Any, name: str) -> Iterator[str]:
    """**倒着**遍历某个工具的每一次执行结果文本（最近的先出）。

    两处方向都必须是倒着来（消息倒着、消息内的 block 也倒着），因为 AgentScope 把一整轮的
    tool_call / tool_result 全塞进同一条 assistant 消息的 content 里。

    做成生成器而不是「返回最后一次」，是为了让调用方能**解析失败就接着往前找**：
    ``shopping_summary`` 在一轮里被调好几次是常态——前几次撞上 harness 的阶段闸（「还没精挑
    就想出清单」）拿回的是哨兵文案，最后一次才真出清单。取第一个、解析失败就认输，等于永远
    只看得到被拒绝的那次。这个坑此前在两个调用点各踩各的。
    """
    for msg in reversed(list(messages or [])):
        for block in reversed(list(getattr(msg, "content", None) or [])):
            if _attr(block, "type") != "tool_result" or _attr(block, "name") != name:
                continue
            output = _attr(block, "output")
            if isinstance(output, str):
                yield output
            elif output:
                yield "".join(block_text(b) for b in output)


# 这里曾有 ``has_tool_result_from(messages, names)``：扫消息历史判「这些工具里有没有哪个真执行
# 过」。唯一的消费者是 ``terminal_enforce``，而它问的其实是「**本轮**调过没有」——扫 messages
# 答不了这个问题，因为续聊时 messages 里还有恢复回来的上一轮历史。改由 ``called_tools`` 回答
# （每轮新建、只记真执行成功的工具）后本函数无人使用，已删（审查报告 B4）。


def tool_blocks(call_id: str, name: str, args: dict[str, Any], result: str) -> list[Any]:
    """造一对「调用 + 结果」的 block（形状与框架自己产生的逐字同构）。

    ``ToolCallBlock.input`` 是 **JSON 字符串**不是 dict（流式解析时一段段拼出来的），当 dict
    用不会报错，只会让轨迹渲染成 ``planner()``、评测侧看不到入参。
    """
    return [
        ToolCallBlock(
            type="tool_call",
            id=call_id,
            name=name,
            input=json.dumps(args, ensure_ascii=False),
        ),
        ToolResultBlock(
            type="tool_result",
            id=call_id,
            name=name,
            output=result,
        ),
    ]


def context_tokens(messages: list[Msg]) -> int:
    """粗估整段历史的 token（只为「该不该兜底压缩」这一个是非题服务）。

    刻意不用框架的 ``model.count_tokens``——它要先跑 ``_prepare_model_input``（私有、还要拉一遍
    工具 schema），而这里只需要判个数量级。本仓的 ``count_tokens`` 对中文更准（框架默认实现是
    bytes/4，中文低估约 3 倍），偏保守正合适：宁可早压一轮，不可撑爆上下文。
    """
    total = 0
    for msg in messages:
        content = msg.content
        if isinstance(content, str):
            total += count_tokens(content)
            continue
        for block in content:
            for field in ("text", "thinking"):
                value = getattr(block, field, None)
                if isinstance(value, str):
                    total += count_tokens(value)
            output = getattr(block, "output", None)
            if isinstance(output, str):
                total += count_tokens(output)
            elif isinstance(output, list):
                total += sum(count_tokens(getattr(x, "text", "") or "") for x in output)
    return total


def terminal_summary(messages: list[Msg]) -> str:
    """从历史里取 shopping_summary 的完整清单原文（终结直出用）。

    从 ``ToolResultBlock`` 的 output 取而不是从截断后的文本取：截断 Hook 只改模型视野里的
    副本，这里要的是完整原文。chat_fallback 不走此路——闲聊收尾本就该由模型口吻说。

    「倒着找 + 解析不出继续往前」的遍历由 ``iter_tool_results`` 负责（那个坑的说明也在那里），
    本函数只管解析：撞阶段闸那几次拿回的是哨兵文案，不是 JSON，跳过即可。
    """
    for text in iter_tool_results(messages, "shopping_summary"):
        try:
            summary = json.loads(text).get("summary")
        except (json.JSONDecodeError, ValueError, AttributeError):
            continue
        if isinstance(summary, str) and summary:
            return summary
    return ""
