"""M1 示例①：最小 AgentLoop —— 跑通 Think → Act → Observe → Reflect。

目的：用最轻的玩具工具，演示「循环次数由模型自判，而非外部写死」。
- 工具按主线规范写（async + Pydantic 输出 + 给模型看的 docstring），见 examples/tools_toy.py。
- 用 AgentScope 的 ``Agent`` 承载循环；模型自己决定先 planner、再 item_search、
  再 price_compare，信息够了就用自然语言收尾、不再调工具。

``ReActConfig(max_iters=...)`` 是**安全网不是流程控制**：它只保证跑飞时能停下来，正常路径上
模型该在远小于上限的轮数里自己收尾。真判「够了没」的是 system prompt 里的 ``<termination>`` 段。

运行：uv run python examples/01_min_loop.py
"""

import asyncio
import sys
from pathlib import Path
from typing import Any

# 直接以脚本方式运行时，sys.path[0] 是 examples/ 而非项目根，需手动把根加入以 import app。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentscope.agent import Agent, ReActConfig  # noqa: E402
from agentscope.message import Msg, TextBlock  # noqa: E402

from app.agent.llm import get_llm  # noqa: E402
from examples.tools_toy import SYSTEM_PROMPT, toy_toolkit  # noqa: E402


def _field(block: Any, key: str) -> Any:
    """blocks 可能是 dict 也可能是带属性的对象，统一取字段。"""
    if isinstance(block, dict):
        return block.get(key)
    return getattr(block, key, None)


def print_trajectory(context: list[Msg]) -> int:
    """打印消息轨迹，返回「工具调用轮数」（= tool_call block 总数）。

    AgentScope 的消息形态要点就在这里：一轮的 tool_call 与 tool_result 都装在**同一条 Msg 的
    content blocks 列表**里，而不是每次工具调用各占一条消息。所以「轮数」要数 block，不是数消息。
    """
    rounds = 0
    for msg in context:
        content = msg.content if isinstance(msg.content, list) else []
        if msg.role == "user":
            text = "".join(str(_field(b, "text") or "") for b in content)
            print(f"  [用户] {text[:120]}")
            continue
        for block in content:
            btype = _field(block, "type")
            if btype == "tool_call":
                rounds += 1
                print(f"  [AI · Think→Act] 调用 {_field(block, 'name')}({_field(block, 'input')})")
            elif btype == "tool_result":
                # output 是 [TextBlock]（本仓工具都走 ToolChunk(content=[TextBlock])），
                # 直接 str() 会打出一串对象 repr，取里面的文本才是模型真正看到的东西。
                out = _field(block, "output")
                if isinstance(out, list):
                    out = "".join(str(_field(b, "text") or "") for b in out)
                print(f"  [Tool · Observe] {_field(block, 'name')} -> {str(out)[:90]}")
            elif btype == "text" and str(_field(block, "text") or "").strip():
                # 模型在轮内说的话——大多是「我打算干什么」的思考，不是收尾。真正的收尾只有
                # 最后那条（判据见 main 里的 terminated_cleanly），这里不预判、只按序打印。
                print(f"  [AI · 说] {str(_field(block, 'text')).strip()[:200]}")
    return rounds


async def main() -> None:
    agent = Agent(
        name="min_loop",
        system_prompt=SYSTEM_PROMPT,
        model=get_llm(),
        toolkit=await toy_toolkit(),
        # 安全网：模型不收尾时兜底停住，正常路径用不到（见模块 docstring）。
        react_config=ReActConfig(max_iters=8),
    )

    query = "帮我搜旅行收纳袋，预算300、不要塑料，然后跨平台比个价给我推荐"
    print("--- 轨迹 ---")

    # 入口是 ``agent.reply(...)``——``Agent`` 没有 ``__call__``，写成 ``agent(msg)`` 会
    # 报 'Agent' object is not callable。要流式看过程用 ``reply_stream``（见示例②）。
    inputs = Msg(name="user", role="user", content=[TextBlock(type="text", text=query)])
    reply = await agent.reply(inputs)
    rounds = print_trajectory(list(agent.state.context))

    # 干净收尾 = 最后交回的是一条**有话说**的回复，而不是又一次工具调用。
    blocks = reply.content if reply is not None and isinstance(reply.content, list) else []
    terminated_cleanly = any(
        _field(b, "type") == "text" and str(_field(b, "text") or "").strip() for b in blocks
    ) and not any(_field(b, "type") == "tool_call" for b in blocks)
    print(f"\n循环轮数（模型自判）：{rounds} 次工具调用")
    print(f"是否干净收尾（最后一条是无工具调用的 AI 回复）：{terminated_cleanly}")


if __name__ == "__main__":
    asyncio.run(main())
