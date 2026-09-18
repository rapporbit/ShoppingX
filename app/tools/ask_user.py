"""ask_user —— 向用户提出澄清问题，阻塞等待回复后返回。

Agent 在 Think 阶段判定关键信息缺失（且无法用合理默认推进）时调用本工具，暂停循环、
通过 WebSocket 把问题推给前端，等用户在输入框回复后拿到回复文本，继续 Observe→Reflect。

**两种形态，由 ``closes_turn`` 区分**（D2；不为收尾形态另开一个工具，两个职责重叠的工具并存
模型会乱选，见执行计划 §3-10）：

- ``closes_turn=False``（默认）：暂停 loop 等回复，拿到回复继续 Observe→Reflect。**非终结**。
- ``closes_turn=True``：问题连同 1~4 个选项作为本轮收尾发出，**不等回复**。等价于 Anthropic 博客
  的 ``present_suggestions``。**终结**——判定见 ``app/agent/constants.is_terminal_call``，它连入参
  一起判，这是全仓唯一一个「终不终结取决于入参」的工具。

限制：
- 超时 120s 无回复则返回兜底文案，Agent 自行决定继续或收尾（只对等回复的那种形态有意义）。
"""

from __future__ import annotations

import asyncio
import logging
import re

from app.api import monitor
from app.api.clarification import clear_waiter, create_pending, register_waiter
from app.api.context import get_thread_id
from app.tools._args import StrListArg
from app.tools._bundle import reconcile_slots_from_reply
from app.tools._shell import tool
from app.utils.env import env_int

logger = logging.getLogger("shoppingx.tools.ask_user")

ASK_USER_TIMEOUT_SEC = env_int("ASK_USER_TIMEOUT_SEC", 120)


_INLINE_MD = re.compile(r"(\*\*|__|`)(.+?)\1")


def strip_inline_markdown(text: str) -> str:
    """剥掉行内加粗 / 下划线 / 反引号，保留内容。只处理成对标记，落单的星号原样留着。"""
    prev = None
    while prev != text:
        prev, text = text, _INLINE_MD.sub(r"\2", text)
    return text


@tool
async def ask_user(
    question: str,
    options: StrListArg | None = None,
    multi_select: bool = False,
    preselected: StrListArg | None = None,
    closes_turn: bool = False,
) -> str:
    """向用户提一个澄清问题；仅关键信息确缺且无法默认时用，别为缩小范围频繁反问。
    参数 question；options 答案是有限项时列出（前端可点选）、开放式留空；multi_select
    是否多选；preselected 多选时的默认勾选（只放必备项）。
    closes_turn=False（默认）暂停等用户回复，拿到回复继续本轮；closes_turn=True 则把问题连同
    1~4 个选项作为**本轮的收尾**发出去、不等回复——已经给过清单/答案，只是顺带问下一步想看什么
    时用它，这种情况下继续空等只会把用户晾住。
    """
    thread_id = get_thread_id()
    if thread_id is None:
        return "（无活跃会话，跳过澄清）用户未回复，请基于已有信息继续。"

    # 问句渲染成一条普通 assistant 消息，不走 Markdown：模型爱在商品名上套 **粗体**，星号会原样
    # 露出来。这里用规则剥掉行内标记，比在 prompt 里叮嘱「别写 Markdown」可靠。
    question = strip_inline_markdown(question)
    await monitor.report_tool_start("ask_user", question=question)

    # 只把真在 options 里的项当默认勾选（模型偶尔会把 preselected 写成 options 外的词）。
    opts = [o for o in (options or []) if o and o.strip()]
    pre = [p for p in (preselected or []) if p in opts] if opts else []

    # 收尾形态（closes_turn=True，即 Anthropic 博客的 present_suggestions）：问题连同选项发出去
    # 就结束本轮，**不登记等待**。这里若照常注册 waiter，任务已经终结、没人再去取那个 Future，
    # 令牌要挂满 120s 才过期；用户这时点选项打回来的回复也无处可投。前端据事件里的 closes_turn
    # 把选项渲染成「下一步」chips，点一下发起新一轮任务，而不是回填到这一轮。
    if closes_turn:
        await monitor.report_clarification_request(
            question,
            options=opts or None,
            multi_select=bool(multi_select) if opts else False,
            preselected=pre or None,
            closes_turn=True,
        )
        await monitor.report_tool_end("ask_user", responded=False, closes_turn=True)
        # 返回问题原文而不是「已发出」：终结工具的产出就是最终答案。这里**不需要**像
        # chat_fallback 那样并回 final_text（adapter._merge_terminal_body）——问题与选项走
        # clarification_request 事件独立推给前端并进回放，屏幕上和历史里都看得到；chat_fallback
        # 的答案则只有 final_text 一条路，被模型补的复述顶掉就彻底没了。
        return question

    # **先登记等待、再把问题发出去**：队列模式下问题经背板到浏览器只是几毫秒，而登记要写一次
    # Redis。反过来写就有一个「回复已经打回来、还没人登记在等」的窗口，那条回复只能被拒收。
    # 单进程模式下 register_waiter 只动一个内存 dict，与原先的顺序无可观测差异。
    fut = create_pending(thread_id)
    token = await register_waiter(thread_id, timeout_sec=ASK_USER_TIMEOUT_SEC)

    await monitor.report_clarification_request(
        question,
        options=opts or None,
        multi_select=bool(multi_select) if opts else False,
        preselected=pre or None,
    )

    responded = True
    try:
        response = await asyncio.wait_for(fut, timeout=ASK_USER_TIMEOUT_SEC)
    except (TimeoutError, asyncio.CancelledError):
        # 区分两种取消：用户点「取消任务」时 cancel 端点先 cancel_pending 再 task.cancel()，
        # 此处若把任务级 CancelledError 也当"未回复"吞掉，Agent 会拿着兜底文案继续跑、
        # 任务永远掐不死。cancelling()>0 说明取消是冲着任务来的，必须向上传播。
        cur = asyncio.current_task()
        if cur is not None and cur.cancelling() > 0:
            raise  # 令牌由下面的 finally 撤（同步操作，取消路径上照样跑得完）
        response = "（用户未在规定时间内回复，请基于已有信息继续）"
        responded = False
    finally:
        # 令牌与这一问同生共死：撤了它，迟到的回复就会被 deliver_reply 拒收（= 按取消处理），
        # 而不是塞给下一问。同步撤本地 + 异步删远端，故取消路径上也跑得完（见 clear_waiter）。
        clear_waiter(thread_id, token)

    if responded:
        # 套装组成确认的「删」通路（机制判，模型只负责问）：用户点名了要哪些槽，没点名的
        # 被问及槽从套装里核销——否则它们以 essential 留表，收尾被说成「没找到、建议再搜」。
        # options（可点选标签）就是「被问及」的确定性依据：没上问卷的槽不算被拒。
        reconcile_slots_from_reply(response, offered=opts or None)

    await monitor.report_tool_end("ask_user", responded=responded)
    return response
