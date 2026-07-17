"""ask_user —— 向用户提出澄清问题，阻塞等待回复后返回。

Agent 在 Think 阶段判定关键信息缺失（且无法用合理默认推进）时调用本工具，暂停循环、
通过 WebSocket 把问题推给前端，等用户在输入框回复后拿到回复文本，继续 Observe→Reflect。

限制：
- 仅主 Agent（fork depth 0）可用——子 Agent 应根据 demands 中已有信息工作。
- 超时 120s 无回复则返回兜底文案，Agent 自行决定继续或收尾。
- 不是终结性工具：回复后循环继续。
"""

from __future__ import annotations

import asyncio
import logging

from langchain_core.tools import tool

from app.agent.fork_guard import current_fork_depth
from app.api import monitor
from app.api.clarification import create_pending
from app.api.context import get_thread_id
from app.tools._args import StrListArg
from app.tools._bundle import reconcile_slots_from_reply
from app.utils.env import env_int

logger = logging.getLogger("shoppingx.tools.ask_user")

ASK_USER_TIMEOUT_SEC = env_int("ASK_USER_TIMEOUT_SEC", 120)


@tool
async def ask_user(
    question: str,
    options: StrListArg | None = None,
    multi_select: bool = False,
    preselected: StrListArg | None = None,
) -> str:
    """向用户提出一个澄清问题并等待回复。

    何时调用：当用户意图中有**关键信息确实缺失**、且无法用合理默认值推进时
    （如用户说「同款」但从未提过具体型号，或说了预算但完全无法判断币种）。
    **不要**为了缩小范围而频繁反问——品类偏宽时直接搜、精挑时覆盖多方向。

    参数：
      - question：要问用户的问题（一句话，清晰具体）。
      - options：可选。**当答案是从若干具体项里挑**时（要/不要防水、套装含哪些件），
        把候选项一条条列进来，前端会渲染成**可点选**的按钮/清单，用户点鼠标即可，不必打字。
        每项写成简短标签（如「书包/双肩包」「笔记本电脑」）。答案是开放式（型号、预算数字）时**留空**。
      - multi_select：options 非空时生效。True=用户可多选（如套装组成：勾选要包含的每一件）；
        False=单选（点一项即回复）。
      - preselected：multi_select=True 时的默认勾选项（须是 options 的子集）。套装组成确认里
        **只把必备槽放进来**，可选/锦上添花的槽留给用户自己勾。
    """
    if current_fork_depth() > 0:
        return (
            "[子任务无权向用户提问] 你是被派发的单平台子任务，无法直接与用户交互。"
            "请根据 demands 中已有的信息继续。"
        )

    thread_id = get_thread_id()
    if thread_id is None:
        return "（无活跃会话，跳过澄清）用户未回复，请基于已有信息继续。"

    await monitor.report_tool_start("ask_user", question=question)

    # 只把真在 options 里的项当默认勾选（模型偶尔会把 preselected 写成 options 外的词）。
    opts = [o for o in (options or []) if o and o.strip()]
    pre = [p for p in (preselected or []) if p in opts] if opts else []
    await monitor.report_clarification_request(
        question,
        options=opts or None,
        multi_select=bool(multi_select) if opts else False,
        preselected=pre or None,
    )

    fut = create_pending(thread_id)
    responded = True
    try:
        response = await asyncio.wait_for(fut, timeout=ASK_USER_TIMEOUT_SEC)
    except (TimeoutError, asyncio.CancelledError):
        # 区分两种取消：用户点「取消任务」时 cancel 端点先 cancel_pending 再 task.cancel()，
        # 此处若把任务级 CancelledError 也当"未回复"吞掉，Agent 会拿着兜底文案继续跑、
        # 任务永远掐不死。cancelling()>0 说明取消是冲着任务来的，必须向上传播。
        cur = asyncio.current_task()
        if cur is not None and cur.cancelling() > 0:
            raise
        response = "（用户未在规定时间内回复，请基于已有信息继续）"
        responded = False

    if responded:
        # 套装组成确认的「删」通路（机制判，模型只负责问）：用户点名了要哪些槽，没点名的
        # 被问及槽从套装里核销——否则它们以 essential 留表，收尾被说成「没找到、建议再搜」。
        # options（可点选标签）就是「被问及」的确定性依据：没上问卷的槽不算被拒。
        reconcile_slots_from_reply(response, offered=opts or None)

    await monitor.report_tool_end("ask_user", responded=responded)
    return response
