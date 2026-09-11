"""M1 示例②：流式事件 —— 用 reply_stream 实时观察循环过程。

目的：直接 ``await agent(msg)`` 要等整条循环跑完才返回；真实任务可能十几秒，前端会以为卡死。
``reply_stream`` 在执行过程中逐步吐事件，让前端能实时显示「正在思考 / 正在调用 X」。
这些框架原生事件就是 M8 AGUI 事件协议的**数据源之一**。

说「之一」是因为本仓的 AGUI 事件并不全从这里来：``tool_start`` / ``tool_end`` 由**各工具自己**
在函数体内上报（只有工具知道该报召回条数、是否降级这些字段，框架事件手上只有工具名），框架
事件流真正独占的是「迭代超限」「最终 Msg」这类只有框架知道的事。详见 app/agent/events.py 顶部。

复用示例①的玩具工具，只把执行方式换成流式。
运行：uv run python examples/02_stream.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentscope.agent import Agent, ReActConfig  # noqa: E402
from agentscope.event import (  # noqa: E402
    ModelCallEndEvent,
    ModelCallStartEvent,
    ReplyEndEvent,
    ToolCallStartEvent,
    ToolResultEndEvent,
)
from agentscope.message import Msg, TextBlock  # noqa: E402

from app.agent.llm import get_llm  # noqa: E402
from examples.tools_toy import SYSTEM_PROMPT, toy_toolkit  # noqa: E402

# 关心的原生事件 → 对应 AgentLoop 阶段（教学用）。
EVENT_LABELS = {
    ModelCallStartEvent: "Think  · 模型开始推理",
    ToolCallStartEvent: "Act    · 工具开始执行",
    ToolResultEndEvent: "Observe· 工具返回结果",
    ModelCallEndEvent: "Reflect· 本轮模型推理完成",
}


def _describe(event: object) -> str:
    """把事件里最值得看的那个字段挑出来。"""
    if isinstance(event, ToolCallStartEvent):
        return event.tool_call_name
    if isinstance(event, ToolResultEndEvent):
        # state 是 SUCCESS / ERROR —— 工具内部报错不外抛而是回 ERROR，这里看得见（见 _shell.py）。
        return f"state={event.state}"
    if isinstance(event, ModelCallEndEvent):
        # 用量在事件上直接挂着，本仓的 token 账（预算闸 / credit 配额）就从这取。
        return f"in={event.input_tokens} out={event.output_tokens}"
    if isinstance(event, ModelCallStartEvent):
        return event.model_name
    return ""


async def main() -> None:
    agent = Agent(
        name="stream_demo",
        system_prompt=SYSTEM_PROMPT,
        model=get_llm(),
        toolkit=await toy_toolkit(),
        react_config=ReActConfig(max_iters=8),
    )

    query = "帮我搜旅行收纳袋，预算300、不要塑料，然后跨平台比个价给我推荐"
    print(f"[用户] {query}\n--- 事件流 ---")

    inputs = Msg(name="user", role="user", content=[TextBlock(type="text", text=query)])
    # yield_final_msg=True：最终回复 Msg 也会从流里出来（否则只能事后翻 state.context，
    # 在「模型最后一轮只调工具没说话」时会摸到错的那条 —— 见 app/agent/events.py）。
    async for event in agent.reply_stream(inputs, yield_final_msg=True):
        if isinstance(event, Msg):
            print(f"  [最终回复] {str(event.get_text_content() or '')[:120]}")
            continue
        if isinstance(event, ReplyEndEvent):
            print(f"  [循环结束] finished_reason={event.finished_reason}")
            continue
        label = EVENT_LABELS.get(type(event))
        if label is None:
            continue  # 只展示关心的几类，过滤掉 delta 等高频噪声事件
        print(f"  [{label}] {_describe(event)}".rstrip())

    print("\n--- 事件流结束 ---")
    print("提示：这些原生事件是 M8 AGUI 协议的数据源之一")
    print("  ModelCallStart → assistant_call；ReplyEnd(EXCEED_MAX_ITERS) → error")
    print("  而 tool_start/tool_end 本仓由工具自己发，字段更全（见 app/agent/events.py）")


if __name__ == "__main__":
    asyncio.run(main())
