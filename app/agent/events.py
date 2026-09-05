"""AgentScope 事件流 → AGUI 事件（批 0 / L3）。

**这一层是补位的，不是总闸**——本仓 AGUI 事件的发送权早就分散在真正知情的地方：

- ``tool_start`` / ``tool_end`` → **各工具自己发**（12 个工具里 11 个在函数体内
  ``monitor.report_tool_*``）。只有工具自己知道该报哪些字段（召回条数、是否降级、有没有
  截断）；框架的 ``ToolCallStartEvent`` 手上只有工具名和入参。
- ``assistant_call`` → ``HarnessAgentAdapter.on_model_call``（L4）。步数口径是 harness 的
  ``guard.think_step``，与终结纪律共用一份计数。
- ``items_preview`` → ``item_picker``：精挑定稿即出货，不等收尾。
- ``summary_delta`` → ``shopping_summary``。本仓语义是「收尾清单文案的**累计全文**」，工具
  内流式产出时就在发。把主 loop 的 ``TextBlockDeltaEvent`` 也灌进去，等于用模型对清单的复述
  覆盖掉用户正在看的那份清单——**这是对手册映射表的一处刻意偏离**，理由就在这。
- ``fork`` → ``task_dispatch``：要在进入子 ``thread_scope`` **之前**发，事件才路由得到父
  thread 的前端连接。
- ``task_result`` / ``task_cancelled`` / ``error``（异常）→ ``orchestrator``：会话级收尾，
  不属于某一次 reply。
- ``model_fallback`` → ``ThrottledChatModel``（L1）：换模型这件事只有网关侧知道。

于是真正落在本模块的只剩「框架知道、而其它人都不知道」的那几件事：迭代超限、需要外部交互
（批 0 不该出现）、以及把最终 ``Msg`` 交回给 orchestrator。宁可薄，也不为了凑满映射表去发
重复或不准的事件——AGUI 事件流是用户唯一能看见 Agent 在干什么的窗口，重复即噪声。
"""

import logging
from collections.abc import AsyncGenerator
from typing import Any

from agentscope.event import (
    ReplyEndEvent,
    ReplyFinishedReason,
    RequireExternalExecutionEvent,
    RequireUserConfirmEvent,
)
from agentscope.message import Msg

from app.api import monitor

logger = logging.getLogger("shoppingx.agent.events")


async def pump_events(stream: AsyncGenerator[Any, None]) -> Msg | None:
    """消费一次 ``reply_stream``，转发该转的事件，返回最终回复 ``Msg``。

    调用方须用 ``yield_final_msg=True`` 开流——否则拿不到最终消息，只能从
    ``agent.state.context`` 里倒着找，那在「模型最后一轮只调工具没说话」时会摸到错的那条。

    返回 ``None`` 表示这次 reply 没有产出最终消息（被打断 / 挂起等外部交互）。
    """
    final: Msg | None = None
    async for event in stream:
        if isinstance(event, Msg):
            final = event
            continue
        if isinstance(event, ReplyEndEvent):
            # 收尾原因从 ``ReplyEndEvent.finished_reason`` 读，**不认 ``ExceedMaxItersEvent``**：
            # 2.0.7 起后者已 deprecated（框架两个都发），认两处就会把同一次超限报两遍。
            if event.finished_reason == ReplyFinishedReason.EXCEED_MAX_ITERS:
                # 模型在上限内没收尾。**当错误上报**（前端画红条），但不抛异常——此刻上下文里
                # 往往已有可用的中间结果，orchestrator 照常收尾比整轮作废对用户好。
                await monitor.report_error("max_iters", "Agent 达到迭代上限仍未调用终结工具")
                logger.warning("达到 max_iters 仍未收尾")
            elif event.finished_reason == ReplyFinishedReason.ERROR:
                detail = str(getattr(event, "error", "") or "reply 内部错误")
                await monitor.report_error("reply_error", detail)
                logger.warning("reply 以错误结束：%s", detail)
            continue
        if isinstance(event, RequireUserConfirmEvent | RequireExternalExecutionEvent):
            # 批 0 不该走到这里：非只读工具已由 app.agent.permissions 精准放行，权限引擎不会
            # 挂起。真出现说明有工具漏进放行表——只记日志、不发 clarification_request：发了
            # 前端会弹一个**没人接得住**的确认框（回复通路要到批 1 接原生 UserConfirmResultEvent
            # 才通），用户点了也没用，比不发更糟。
            names = [getattr(tc, "name", "?") for tc in getattr(event, "tool_calls", []) or []]
            logger.warning(
                "reply 因需要外部交互而挂起（%s，工具=%s）——检查放行表 permissions.py",
                type(event).__name__,
                names,
            )
            continue
    return final
