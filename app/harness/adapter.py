"""把 Harness 的 hook_point 接到 AgentScope 的中间件面。

落点选择（用探针脚本逐个 hook 实测后钉死；探针脚本未入库）：

| hook_point | 落点 | 为什么是这里 |
|---|---|---|
| ``pre_think`` | ``on_model_call`` 前 | **只有这里拿得到 messages**（``on_reasoning`` 的
  input_kwargs 只有 tool_choice），而 pre_think 的四个 hook 全都要改 messages / 换模型 |
| ``post_reflect`` | ``on_reasoning`` 后 | 此刻本轮 assistant 消息已落进 ``agent.state.context``，
  能如实读出「这轮到底调没调工具」 |
| ``pre_tool_call`` | ``HarnessToolAdapter`` 内，next_handler 之前 | 洋葱外层，拒绝时不调它 |
| ``post_tool_call`` | 同上，next_handler 之后 | 对聚合后的结果文本跑 |
| ``on_session_end`` | ``on_reply`` 结束前 | 一次 reply 的收尾 |

**retry_nudge 走「吞事件」而不是手工重发**：AgentScope 原生支持「吞掉 ``ReplyEndEvent`` 即强制
再来一轮」，所以在 ``on_reply`` 里把提示追加进 ``agent.state.context`` 后吞掉结束事件。省掉一次
手工重发，且这轮纠正会跟着 state 走（下一轮模型仍看得见自己被纠正过什么）。
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncGenerator, Callable
from typing import TYPE_CHECKING, Any

from agentscope.message import (
    HintBlock,
    Msg,
    TextBlock,
)
from agentscope.middleware import MiddlewareBase
from agentscope.model import ChatResponse
from agentscope.tool import ToolMiddlewareBase
from agentscope.tool._response import ToolChunk, ToolResultState
from pydantic import ValidationError

from app.api import monitor
from app.harness.autopick import maybe_autopick
from app.harness.middleware import harness
from app.harness.msgs import block_text, terminal_summary, text_of
from app.harness.phase_machine import get_phase_machine
from app.harness.prefill import prefill
from app.harness.session import HarnessSession, collect_call_signals
from app.harness.signals import (
    _observe_tool,
    _summarize_call,
)
from app.harness.streaming import charge_stream
from app.harness.tiering import first_round_tier, resolve_model_tier
from app.harness.token_budget import charge_usage

if TYPE_CHECKING:  # pragma: no cover
    from agentscope.agent import Agent
    from agentscope.tool import ToolBase

logger = logging.getLogger("shoppingx.harness.adapter")


# ``_text_of`` / ``_block_text`` 的实现在 ``harness.msgs``（消息形态的知识只住那一个文件）。
# 这里保留本名的别名：调用点密集，改名收益不抵 diff 噪声。
_block_text = block_text
_text_of = text_of


def _has_tool_calls(msg: Msg | None) -> bool:
    if msg is None:
        return False
    return any(getattr(b, "type", None) == "tool_call" for b in msg.content)


def _persist_injections(agent: Agent, injected: list[Msg] | None) -> None:
    """把本轮新增的注入（漂移纠正 / 断言纠正 / 预算 hint）落进 ``state.context``。

    **不落 state 的代价是缓存塌方**（L5 实测抓到）：``_prepare_model_input`` 每轮都从
    ``state.context`` 重建 messages，注入若只加在这一次的 ``input_kwargs`` 里，下一轮就从历史
    里蒸发了——上一轮 payload 的第 n 条是「[漂移纠正]…」，这一轮第 n 条变成模型的回复，前缀从
    注入点起全部失配。实测一条 9 次模型调用的链，注入那一对的前缀稳定率掉到 0.9，且注入越多掉越狠。

    落地形态选 ``HintBlock`` 而不是独立的 system ``Msg``，两个理由：

    1. 它是框架给「循环中塞外部提示」准备的原生块（框架自己的 runtime-state 注入就用它），
       formatter 会渲染成一条 ``role="user"`` 消息——语义上这确实是外部对模型说的话，不是模型
       自己说的，塞进 assistant 的 content 会让模型把纠正读成自己的发言。
    2. 它跟着 ``append_context`` 进当前 assistant 消息的 blocks，不新建消息，因而不会出现两条
       ``id`` 都等于 ``reply_id`` 的消息（``append_context`` 遇到非 assistant 结尾会新建一条，
       那会给 ``get_awaiting_tool_calls`` / 落盘恢复埋下同 id 的坑）。

    调用点在**构造本轮视图之前**（见 ``on_model_call``），所以本轮与下一轮看到的是同一份字节，
    连「注入当轮断一次」都不会发生——比「注入只断一轮」的旧口径再进一步。
    唯一的例外是 pre_think 的 hook 自己往 ``messages`` 里 append 的那种（预算 MINIMAL hint），
    它这一轮是 system 消息、下一轮是 hint，会断一次；档位只降不升，一个 loop 至多一次。
    """
    if not injected:
        return
    blocks: list[Any] = [
        HintBlock(source="harness", hint=[TextBlock(type="text", text=text)])
        for text in (_text_of(m) for m in injected)
        if text
    ]
    if blocks:
        agent.state.append_context(agent.name, blocks)


def _last_assistant(agent: Agent) -> Msg | None:
    for msg in reversed(agent.state.context):
        if msg.role == "assistant":
            return msg
    return None


class HarnessAgentAdapter(MiddlewareBase):
    """Agent 侧的三个落点：pre_think / post_reflect / 会话收尾与 retry_nudge。"""

    def __init__(self, session: HarnessSession) -> None:
        self._s = session

    # ── pre_think（含终结直出与预算 fallback 两条早退）──

    async def on_model_call(
        self,
        agent: Agent,
        input_kwargs: dict,
        next_handler: Callable[..., Any],
    ) -> ChatResponse | AsyncGenerator[ChatResponse, None]:
        s = self._s
        messages: list[Msg] = list(input_kwargs.get("messages") or [])

        # 终结直出（延迟治理 round2 刀 2）：shopping_summary 已产出面向用户的完整清单，此处再
        # 唤起模型只会把同一份清单复述一遍（实测 729 tok / 7.5s，还多一道转录出错风险）。
        # **只认本轮真调过 shopping_summary 的情况**：``messages`` 是多轮上下文，续聊时里面还躺着
        # 上一轮的清单——第二轮用 chat_fallback 收尾（比如「帮我下单第一款」缺地址）若也走这里，
        # 会把上一轮的清单当成本轮回复原样复述（e2e 实测踩到）。called_tools 每轮新建，答得了
        # 「本轮」。
        if s.guard.terminal_reached and "shopping_summary" in s.called_tools:
            summary = terminal_summary(messages)
            if summary:
                return ChatResponse(
                    content=[TextBlock(type="text", text=summary)],
                    is_last=True,
                )

        s.round_counter += 1
        s.guard.think_step += 1
        # 终结提醒配额是「每次模型调用一次」，不是整个 loop 一次——第 3 轮催过，第 12 轮还想
        # 蒙混时照样得催。
        s.guard.terminal_nudge_retries = 0
        await monitor.report_assistant_call(step=str(s.guard.think_step))

        # 检索合流后自动比价 + 精挑（round3 刀 2）：放在消费 inject 通道之前，它的结果与它触发的
        # 收线通告 / 偏好注入一并随本轮 hint 落 state。
        await maybe_autopick(s)

        # 注入**先落 state 再构造视图**：``messages`` 里的 Msg 与 ``state.context`` 是同一批
        # 对象，``append_context`` 原地把 hint 挂进末尾那条 assistant 消息，本轮视图因此自动
        # 含它、且与下一轮从 state 重建出来的形态逐字一致——注入连一次前缀断裂都不会造成。
        # （末尾不是 assistant 时 append_context 会新建一条 Msg，它不在 messages 快照里，
        #   所以下面把新增部分补进视图。）
        before = len(agent.state.context)
        _persist_injections(agent, s.consume_inject())
        ctx = s.base_context()
        ctx["messages"] = [*messages, *agent.state.context[before:]]
        # 留给 pre_think 的 hook 追加自己的注入（如预算 MINIMAL hint），hook 跑完后一并落 state。
        ctx["persist_messages"] = []
        ctx["system_message"] = next((m for m in messages if m.role == "system"), None)
        ctx["recent_actions_summary"] = s.recent_actions_summary()
        ctx = await harness.run("pre_think", ctx)

        # 预算 fallback 档：连一次 LLM 调用都付不起了，直接把规则兜底的回答当模型输出返回。
        # 无 tool_call → loop 自然终止。post_reflect **仍会跑**（on_model_call 在 on_reasoning
        # 内层），靠这里置的 terminal_reached 让终结纪律 Hook 放行——否则它会因「没调终结工具
        # 就想收尾」要求重发，on_reply 吞 ReplyEnd 再合成同一段，空转到 max_iters。
        fallback = ctx.get("fallback_answer")
        if isinstance(fallback, str) and fallback:
            s.guard.terminal_reached = True
            return ChatResponse(content=[TextBlock(type="text", text=fallback)], is_last=True)

        input_kwargs["messages"] = ctx["messages"]
        _persist_injections(agent, ctx.get("persist_messages"))
        # 换档（预算降 lite / 第一轮加档）：Hook 只给**档位名**，模型对象在这里解析——
        # 这是「Hook 决策、适配器落地」的落点，也是档位→模型解析的**唯一**一处。
        # 顺序即优先级：Hook（budget_router）写过档就照它的来，没写才轮到第一轮加档。
        tier = ctx.get("model_tier") or first_round_tier(ctx)
        model = resolve_model_tier(tier)
        if model is not None:
            input_kwargs["current_model"] = model

        model_name = getattr(input_kwargs.get("current_model") or agent.model, "model", "")
        res = await next_handler(**input_kwargs)
        if hasattr(res, "__aiter__"):
            return charge_stream(res, str(model_name), s)
        charge_usage(str(model_name), getattr(res, "usage", None))
        s.track_token_delta()
        return res

    # 框架自带的摘要压缩（``on_compress_context``）**不接管**：超阈值时由框架写 continuation
    # summary 进 ``state.summary``，随 session.json 一起持久化。本仓不再做 block 级视图压缩。

    # ── post_reflect ──

    async def on_reasoning(
        self,
        agent: Agent,
        input_kwargs: dict,
        next_handler: Callable[..., AsyncGenerator],
    ) -> AsyncGenerator:
        async for event in next_handler(**input_kwargs):
            yield event
        await self._run_post_reflect(agent)

    async def _run_post_reflect(self, agent: Agent) -> None:
        s = self._s
        # 「轮」的边界：解除上一轮的回退闭锁（回退后同轮不得再前进）。
        machine = get_phase_machine()
        if machine is not None:
            machine.begin_round()
        ai_msg = _last_assistant(agent)
        ctx = s.base_context()
        ctx["recent_actions_summary"] = s.recent_actions_summary()
        ctx["messages"] = list(agent.state.context)
        ctx["response_ai_message"] = ai_msg
        ctx["response_has_tool_calls"] = _has_tool_calls(ai_msg)
        ctx["planner_output_ready"] = s.planner_done
        ctx["total_candidates"] = s.fresh_candidates
        ctx["picks_count"] = s.last_picks
        ctx["picker_attempted"] = s.picker_attempted
        ctx["must_have_hits"] = s.last_must_hits
        ctx["oncat_count"] = s.last_oncat
        ctx["offcat_count"] = s.last_offcat
        ctx["excluded_count"] = s.last_excluded
        ctx["over_budget_count"] = s.last_over_budget

        ctx = await harness.run("post_reflect", ctx)
        # 补搜闸宣判「这池子不够用」后，污染批不再算「本轮已搜到货」。
        if ctx.pop("reset_fresh_candidates", False):
            s.fresh_candidates = 0
        s.collect(ctx)
        nudge = ctx.get("retry_nudge")
        if isinstance(nudge, str) and nudge:
            s.retry_nudge = nudge

    # ── 会话收尾 + retry_nudge 的兑现 ──

    async def on_reply(
        self,
        agent: Agent,
        input_kwargs: dict,
        next_handler: Callable[..., AsyncGenerator],
    ) -> AsyncGenerator:
        s = self._s
        await self._prefill(agent)
        async for event in next_handler(**input_kwargs):
            if type(event).__name__ == "ReplyEndEvent" and s.retry_nudge:
                if self._force_another_round(agent, s.retry_nudge):
                    s.retry_nudge = None
                    continue  # 吞掉结束事件 = 强制再来一轮（框架原生语义）
                s.retry_nudge = None
            if isinstance(event, Msg):
                event = await self._finalize(event)
            yield event

    async def _prefill(self, agent: Agent) -> None:
        """开局预置（planner / 参考图），实现见 :mod:`app.harness.prefill`。"""
        await prefill(self._s, agent)

    @staticmethod
    def _force_another_round(agent: Agent, nudge: str) -> bool:
        """把催收尾的提示追加进上下文，并确认还有轮次可用。

        没轮次了就**不吞**结束事件：硬吞会让框架在 max_iters 上原地打转，比「这次没催成」
        坏得多——催收尾是纪律，不是不惜代价。
        """
        reply_ctx = agent.state.reply_context
        cur = getattr(reply_ctx, "cur_iter", None)
        max_iters = getattr(reply_ctx, "max_iters", None)
        if isinstance(cur, int) and isinstance(max_iters, int) and cur >= max_iters:
            logger.info("已达 max_iters，放弃 retry_nudge（不硬吞 ReplyEnd）")
            return False
        agent.state.context.append(
            Msg(name="user", role="user", content=[TextBlock(type="text", text=nudge)]),
        )
        logger.info("模型未调终结工具就想收尾，追加提示强制再来一轮")
        return True

    async def _finalize(self, msg: Msg) -> Msg:
        """on_session_end：output_guard / output_audit 对最终答案的加工。

        改写走 ``context["final_answer"]``——这条回写通路必须留着，否则脱敏与审计就成了
        只会记日志的摆设。
        """
        ctx = self._s.base_context()
        ctx["final_answer"] = _text_of(msg)
        ctx = await harness.run("on_session_end", ctx)
        final = ctx.get("final_answer")
        if not isinstance(final, str) or final == _text_of(msg):
            return msg
        return msg.model_copy(update={"content": [TextBlock(type="text", text=final)]})


async def after_tool_success(
    s: HarnessSession,
    tool_name: str,
    tool_args: dict[str, Any],
    result_text: str,
    *,
    pre_ctx: dict[str, Any] | None = None,
) -> str:
    """一次**成功**工具调用之后的全部控制面动作，返回模型最终看到的文本。

    从 ``HarnessToolAdapter.on_tool_call`` 提出来成独立函数，是为了让自动比价精挑
    （``harness.autopick``）走**同一条**管线：进展续命、called_tools、阶段信号、post_tool_call
    的截断 / 收线通告 / schema 断言 / 偏好注入——自动跑出来的结果与模型亲手调的在控制面上
    不可区分。``pre_ctx`` 是 pre_tool_call 的产出（熔断武装 / 收敛计数），自动路径没有就留空。
    """
    from app.harness.autopick import arm_on_tool

    s.guard.last_progress_at = time.monotonic()
    s.guard.watchdog_nudged_at = 0.0
    s.called_tools.add(tool_name)
    s.recent_actions.append(_summarize_call(tool_name, tool_args))
    if len(s.recent_actions) > 30:
        s.recent_actions = s.recent_actions[-20:]
    arm_on_tool(s, tool_name, tool_args)

    signals = collect_call_signals(s, tool_name, result_text)

    # post_tool_call：截断 / 提示 / 终结标记 / 熔断计数 / 断言 / 漂移信号
    pre = pre_ctx or {}
    post_ctx = s.base_context()
    post_ctx["tool_name"] = tool_name
    post_ctx["tool_args"] = tool_args
    post_ctx["tool_result"] = result_text
    post_ctx.update(signals)
    post_ctx["converge_count"] = pre.get("converge_count")
    post_ctx["converge_note"] = pre.get("converge_note")
    post_ctx["_breaker_armed"] = pre.get("_breaker_armed")
    post_ctx = await harness.run("post_tool_call", post_ctx)
    s.collect(post_ctx)

    guarded = post_ctx.get("tool_result")
    return guarded if isinstance(guarded, str) else result_text


class HarnessToolAdapter(ToolMiddlewareBase):
    """工具侧的两个落点：pre_tool_call（含拒绝）与 post_tool_call（含结果改写）。

    洋葱顺序是 **Harness 在外、工具自身的中间件在内**：闸门要在任何执行发生前就能拦下，
    拒绝时直接 yield 一条 ERROR chunk、根本不调 ``next_handler``。
    """

    def __init__(self, session: HarnessSession) -> None:
        self._s = session

    async def on_tool_call(  # type: ignore[override]  # 基类签名标的是 Coroutine，实际契约是 async generator（见基类 docstring 的示例）
        self,
        tool: ToolBase,
        input_kwargs: dict[str, Any],
        next_handler: Callable[..., AsyncGenerator[ToolChunk, None]],
    ) -> AsyncGenerator[ToolChunk, None]:
        s = self._s
        tool_name = tool.name
        tool_args = dict(input_kwargs)

        # 1. pre_tool_call：硬闸 + 阶段门 + 顺序断言 + 熔断判定
        ctx = s.base_context()
        ctx["tool_name"] = tool_name
        ctx["tool_args"] = tool_args
        ctx["tool_call_id"] = ""
        ctx = await harness.run("pre_tool_call", ctx)

        if ctx.get("_rejected"):
            reason = ctx.get("_reject_reason", "Hook 拒绝")
            # raw=True 的哨兵原样回模型（本就是写给模型的完整指令）；否则加前缀标明来源。
            content = reason if ctx.get("_reject_raw") else f"[Harness 拒绝] {reason}"
            # 被闸拦下的调用同样喂 LoopDetector：模型换着参数硬撞同一道闸时，拒绝路径不计数
            # 就是循环检测的盲区。
            if s.guard.detector.record(tool_name):
                content += f"\n\n[系统提示] {s.guard.detector.nudge_message(tool_name)}"
            yield ToolChunk(
                content=[TextBlock(type="text", text=content)],
                state=ToolResultState.ERROR,
                metadata={"harness_rejected": True, "tool": tool_name},
            )
            return

        s.collect(ctx)

        # 2. 真实执行。计时与熔断计数只覆盖真实执行——被闸拦下的哨兵不算。
        start = time.monotonic()
        chunks: list[ToolChunk] = []
        try:
            async for chunk in next_handler(**input_kwargs):
                chunks.append(chunk)
        except Exception as exc:
            _observe_tool(tool_name, time.monotonic() - start, "error")
            # 参数校验类失败（ValidationError）**不计入熔断**：那是调用方（模型）的锅，不是工具
            # 基础设施故障。断路器是进程级共享的，计入会让一个会话连发 3 次畸形参数就把该工具对
            # 全进程所有会话熔断 60s。不记也不会卡死断路器：HALF_OPEN 下一次调用照常放行探测。
            if ctx.get("_breaker_armed") == tool_name and not isinstance(exc, ValidationError):
                from app.harness.hooks.repetition import get_tool_breaker
                from app.utils import shared_breaker

                # 这条 if 就是「谁算失败」的唯一判据（含 ValidationError 豁免）；shared_breaker
                # 只负责把判定结果多写一份到 Redis，不做二次判断。
                await shared_breaker.record_failure(get_tool_breaker(tool_name))
            raise

        last = chunks[-1] if chunks else None
        result_text = "".join(
            _block_text(b) for c in chunks for b in c.content if getattr(b, "type", None) == "text"
        )
        # 工具内部报错走 state=ERROR（见 app/tools/_as_tools.py）：
        # 工具没真跑过，不记 called_tools、不推阶段信号、不给看门狗续命、
        # 不跑 post_tool_call。只喂 LoopDetector——硬撞同一个错误正是打转。
        if last is not None and last.state == ToolResultState.ERROR:
            _observe_tool(tool_name, time.monotonic() - start, "error")
            if s.guard.detector.record(tool_name):
                nudge = f"\n\n[系统提示] {s.guard.detector.nudge_message(tool_name)}"
                yield ToolChunk(
                    content=[TextBlock(type="text", text=result_text + nudge)],
                    state=ToolResultState.ERROR,
                    metadata=dict(last.metadata or {}),
                )
                return
            for chunk in chunks:
                yield chunk
            return

        _observe_tool(tool_name, time.monotonic() - start, "ok")
        guarded = await after_tool_success(s, tool_name, tool_args, result_text, pre_ctx=ctx)
        if guarded != result_text:
            yield ToolChunk(
                content=[TextBlock(type="text", text=guarded)],
                state=ToolResultState.SUCCESS,
                metadata=dict(last.metadata or {}) if last is not None else {},
            )
            return
        for chunk in chunks:
            yield chunk
