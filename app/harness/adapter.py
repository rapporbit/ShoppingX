"""把 Harness 的六个 hook_point 接到 AgentScope 的中间件面（批 0 / L4）。

对照 :mod:`app.harness.agent_middleware`（LangChain 版）逐条重建，**注册表、hook 名称、
priority、顺序契约一个都不动**——控制面的语义是整仓最易碎的部分（``tool_gates.py`` 头注释
里那串顺序契约尤其），迁移只换「钩子挂在哪」，不换「钩子做什么」。

落点选择（探针实测钉死，见 scratchpad probe_hooks）：

| hook_point | 落点 | 为什么是这里 |
|---|---|---|
| ``on_session_start`` | orchestrator 进入时手动跑 | 会话级，不属于任何一次 reply |
| ``pre_think`` | ``on_model_call`` 前 | **只有这里拿得到 messages**（``on_reasoning`` 的
  input_kwargs 只有 tool_choice），而 pre_think 的四个 hook 全都要改 messages / 换模型 |
| ``post_reflect`` | ``on_reasoning`` 后 | 此刻本轮 assistant 消息已落进 ``agent.state.context``，
  能如实读出「这轮到底调没调工具」 |
| ``pre_tool_call`` | ``HarnessToolAdapter`` 内，next_handler 之前 | 洋葱外层，拒绝时不调它 |
| ``post_tool_call`` | 同上，next_handler 之后 | 对聚合后的结果文本跑 |
| ``on_session_end`` | ``on_reply`` 结束前 | 一次 reply 的收尾 |

**retry_nudge 的落地方式变了，语义没变**：LangChain 版是在同一次 ``awrap_model_call`` 里手动
再发一次模型；AgentScope 原生支持「吞掉 ``ReplyEndEvent`` 即强制再来一轮」，所以改成在
``on_reply`` 里把提示追加进 ``agent.state.context`` 后吞掉结束事件。省掉一次手工重发，且这轮
纠正会跟着 state 走（下一轮模型仍看得见自己被纠正过什么）。
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncGenerator, Callable
from typing import TYPE_CHECKING, Any

from agentscope.message import HintBlock, Msg, TextBlock
from agentscope.middleware import MiddlewareBase
from agentscope.model import ChatResponse
from agentscope.tool import ToolMiddlewareBase
from agentscope.tool._response import ToolChunk, ToolResultState

from app.agent.fork_guard import current_fork_depth
from app.agent.token_budget import tree_snapshot
from app.api import monitor
from app.compress.as_blocks import as_post_step_compress
from app.harness._msgcompat import RUNTIME_AGENTSCOPE, RUNTIME_KEY
from app.harness._tool_signals import (
    _SEARCH_TOOLS,
    _as_opt_int,
    _count_candidates,
    _count_picks,
    _observe_tool,
    _summarize_call,
)
from app.harness.hooks.context_compress import _compress_opts
from app.harness.hooks.drift_detector import DriftState
from app.harness.middleware import harness
from app.harness.phase_machine import get_phase_machine
from app.harness.state import GuardState
from app.tools._diagnostics import consume_diagnostics
from app.utils.tokens import count_tokens

if TYPE_CHECKING:  # pragma: no cover
    from agentscope.agent import Agent
    from agentscope.tool import ToolBase

logger = logging.getLogger("shoppingx.harness.adapter")


class HarnessSession:
    """一次 AgentLoop 的控制面状态，被 Agent 适配器与 Tool 适配器**共享**。

    LangChain 版把这些字段挂在中间件实例上，因为那边模型钩子与工具钩子同属一个类。AgentScope
    把两者拆成了 ``MiddlewareBase`` 与 ``ToolMiddlewareBase``，于是状态必须外提——否则
    pre_tool_call 攒的断言就流不到 post_reflect 的 ``assertion_handler`` 手里（LangChain 版
    docstring 里的「三条接力通道」在这里一条都不能少）。
    """

    def __init__(
        self,
        *,
        original_query: str = "",
        guard: GuardState | None = None,
    ) -> None:
        self.original_query = original_query
        self.guard = guard if guard is not None else GuardState()
        self.round_counter = 0
        self.called_tools: set[str] = set()
        self.drift_state = DriftState()
        self.recent_actions: list[str] = []
        # 接力通道 1：Hook 产出的注入消息 → 下一次 pre_think 消费
        self.pending_inject: list[dict[str, str]] = []
        # 接力通道 2：pre/post_tool_call 攒的断言失败 → 本轮 post_reflect 消费
        self.pending_assertions: list[dict[str, Any]] = []
        # 接力通道 3：GuardState（上面的 self.guard）
        self.last_total_tokens = 0
        self.planner_done = False
        self.picker_attempted = False
        self.last_picks = 0
        self.last_must_hits: int | None = None
        self.last_oncat: int | None = None
        self.last_offcat: int | None = None
        self.last_excluded: int | None = None
        self.last_over_budget: int | None = None
        self.fresh_candidates = 0
        # post_reflect 判定「该催收尾」后挂在这里，由 on_reply 在 ReplyEnd 时兑现
        self.retry_nudge: str | None = None

    def base_context(self) -> dict[str, Any]:
        return {
            "original_query": self.original_query,
            "round_number": self.round_counter,
            "called_tools": set(self.called_tools),
            "_drift_state": self.drift_state,
            "_guard": self.guard,
            RUNTIME_KEY: RUNTIME_AGENTSCOPE,
        }

    def recent_actions_summary(self) -> str:
        return "; ".join(self.recent_actions[-9:]) if self.recent_actions else ""

    def collect(self, ctx: dict[str, Any]) -> None:
        """把 Hook 产出的注入与断言收进接力通道。"""
        inject = ctx.get("inject_messages")
        if inject:
            self.pending_inject.extend(inject)
        failed = ctx.get("assertions_failed")
        if failed:
            self.pending_assertions.extend(failed)

    def consume_inject(self) -> list[Msg]:
        if not self.pending_inject:
            return []
        msgs = [
            Msg(name="system", role="system", content=[TextBlock(type="text", text=m["content"])])
            for m in self.pending_inject
            if m.get("content")
        ]
        self.pending_inject.clear()
        return msgs

    def track_token_delta(self) -> None:
        """本次模型调用的 token 增量喂给漂移检测（信号 4：成本失控）。"""
        snap = tree_snapshot()
        if not snap:
            return
        total = int(snap.get("input_tokens", 0)) + int(snap.get("output_tokens", 0))
        delta = total - self.last_total_tokens
        self.last_total_tokens = total
        if delta > 0:
            self.drift_state.token_history.append(delta)


def _resolve_model_tier(tier: Any) -> Any | None:
    """档位名 → 本运行时的模型对象。认不出的档位不换模型（失效方向是「照常跑」）。

    延迟导入 ``llm``：本模块在 Agent 装配前就被 import，模块级拉模型工厂会把 ``.env`` 的读取
    时机提前到 import 期，测试里 monkeypatch 环境变量就来不及了。
    """
    if not tier:
        return None
    from app.agent.llm import get_as_lite_llm, get_as_llm

    if tier == "reasoning":
        return get_as_llm()
    if tier == "lite":
        return get_as_lite_llm()
    logger.warning("未知模型档位 %r，本轮不换模型", tier)
    return None


def _block_text(block: Any) -> str:
    """content block → 文本（非文本块给空串）。"""
    return getattr(block, "text", "") or ""


def _text_of(msg: Msg | None) -> str:
    if msg is None:
        return ""
    return "".join(_block_text(b) for b in msg.content if getattr(b, "type", None) == "text")


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
    连「注入当轮断一次」都不会发生——比 LangChain 侧 M6.1 的口径（注入只断一轮）再进一步。
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


def _context_tokens(messages: list[Msg]) -> int:
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
        if s.guard.terminal_reached:
            summary = _terminal_summary(messages)
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
        # 无 tool_call → loop 自然终止。**刻意绕过 post_reflect**：终结纪律 Hook 会因为
        # 「没调终结工具就想收尾」要求重发，可预算正是为此耗尽的，再发一次纯属把最后的钱烧掉。
        fallback = ctx.get("fallback_answer")
        if isinstance(fallback, str) and fallback:
            s.guard.terminal_reached = True
            return ChatResponse(content=[TextBlock(type="text", text=fallback)], is_last=True)

        input_kwargs["messages"] = ctx["messages"]
        _persist_injections(agent, ctx.get("persist_messages"))
        # 换档（第一轮开 reasoning / 预算降 lite）：Hook 只给**档位名**，模型对象在这里解析。
        # 刻意不读 ``model_override``——那个键装的是 LangChain 模型对象，塞进 current_model
        # 会在调用时炸「'ChatOpenAI' object is not callable」（L3 的冒烟测试抓到过）。
        model = _resolve_model_tier(ctx.get("model_tier"))
        if model is not None:
            input_kwargs["current_model"] = model

        res = await next_handler(**input_kwargs)
        s.track_token_delta()
        return res

    # ── 框架自带的摘要压缩：接管，不放行 ──

    async def on_compress_context(
        self,
        agent: Agent,
        input_kwargs: dict,
        next_handler: Callable[..., Any],
    ) -> None:
        """把框架的「LLM 摘要 + 替换 context」换成本仓的 block 级压缩。**刻意不调 next_handler。**

        框架的实现会让模型写一份 continuation summary，然后用它**替换掉** ``state.context`` 里
        被压的那几条。两处与本仓冲突：

        1. 本仓的第一性原则是「压缩只改这一次送给模型的视图，不改历史原文」（见 as_blocks）。
           原文一旦被摘要替换，``agent_state.json`` 落盘的就是摘要，跨进程续聊读回来的历史
           从此是二手的——L3 验收过的那条「模型接住了上文」的链路会悄悄降级。
        2. 它要额外烧一次 LLM 调用，而触发它的场景（上下文逼近 102k）恰恰是预算最紧的时候。

        所以这里就地对 ``state.context`` 跑一次同一套 block 级压缩：把较旧的工具结果截到上限，
        一条消息不删、一个 block 不丢。这是**兜底**——正常路径上 pre_think 的视图压缩早已把体积
        压住，能走到这里说明历史真的异常长，此时保结构比保细节重要。

        压缩后仍不达标不会死循环：本函数幂等（已截断的不再压），框架下一轮照常调模型，真超上限
        由网关报错——比静默把历史换成摘要更诚实。

        **阈值判断必须自己做**（实测踩到的坑）：本钩子是 ``_reply_impl`` 在**每次** Reasoning
        前无条件调的，「超没超阈值」判在 ``_compress_context_impl`` 里、也就是 next_handler
        那一侧。不判就直接压 = 每轮都把 ``state.context`` 的历史原文截一遍，比框架的摘要还狠。
        """
        threshold = agent.context_config.trigger_ratio * agent.model.context_size
        if _context_tokens(agent.state.context) < threshold:
            return

        keep_recent, max_tool_tokens, _ = _compress_opts()
        before = len(agent.state.context)
        agent.state.context = as_post_step_compress(
            list(agent.state.context),
            keep_recent=keep_recent,
            max_tool_tokens=max_tool_tokens,
        )
        logger.warning(
            "上下文逼近上限，已就地做 block 级压缩兜底（%d 条消息，未启用框架摘要）", before
        )

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
        # 「轮」的边界：解除上一轮的回退闭锁（回退后同轮不得再前进）。阶段机是主 loop 独有。
        if current_fork_depth() == 0:
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
        if s.pending_assertions:
            ctx["assertions_failed"] = list(s.pending_assertions)
            s.pending_assertions.clear()

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
        async for event in next_handler(**input_kwargs):
            if type(event).__name__ == "ReplyEndEvent" and s.retry_nudge:
                if self._force_another_round(agent, s.retry_nudge):
                    s.retry_nudge = None
                    continue  # 吞掉结束事件 = 强制再来一轮（框架原生语义）
                s.retry_nudge = None
            if isinstance(event, Msg):
                event = await self._finalize(event)
            yield event

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


def _terminal_summary(messages: list[Msg]) -> str:
    """从历史里取 shopping_summary 的完整清单原文（终结直出用）。

    从 ``ToolResultBlock`` 的 output 取而不是从截断后的文本取：截断 Hook 只改模型视野里的
    副本，这里要的是完整原文。chat_fallback 不走此路——闲聊收尾本就该由模型口吻说。

    倒着找、且解析不出就继续往前找：同一轮里 ``shopping_summary`` 常被调好几次，前几次撞
    阶段闸拿回的是哨兵文案（不是 JSON）。取到那次就等于把一段哨兵直出给用户。
    """
    import json

    for msg in reversed(messages):
        for block in reversed(list(getattr(msg, "content", []) or [])):
            if getattr(block, "type", None) != "tool_result":
                continue
            if getattr(block, "name", None) != "shopping_summary":
                continue
            output = getattr(block, "output", "")
            text = output if isinstance(output, str) else "".join(_block_text(b) for b in output)
            try:
                summary = json.loads(text).get("summary")
            except (json.JSONDecodeError, ValueError, AttributeError):
                continue
            if isinstance(summary, str) and summary:
                return summary
    return ""


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
            # 就是循环检测的盲区。回放路径（tool_memo）已自己喂过，跳过防双记。
            if not ctx.get("_detector_fed") and s.guard.detector.record(tool_name):
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
        except Exception:
            _observe_tool(tool_name, time.monotonic() - start, "error")
            if ctx.get("_breaker_armed") == tool_name:
                from app.harness.hooks.tool_breaker import get_tool_breaker

                get_tool_breaker(tool_name).record_failure()
            raise

        last = chunks[-1] if chunks else None
        result_text = "".join(
            _block_text(b) for c in chunks for b in c.content if getattr(b, "type", None) == "text"
        )
        # 工具内部报错走 state=ERROR（见 app/tools/_as_tools.py），语义等同 LangChain 版的
        # status="error"：工具没真跑过，不记 called_tools、不推阶段信号、不给看门狗续命、
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
        s.guard.last_progress_at = time.monotonic()
        s.guard.watchdog_nudged_at = 0.0
        s.called_tools.add(tool_name)
        s.recent_actions.append(_summarize_call(tool_name, tool_args))
        if len(s.recent_actions) > 30:
            s.recent_actions = s.recent_actions[-20:]

        signals = _collect_call_signals(s, tool_name, result_text)

        # 3. post_tool_call：截断 / 提示 / 终结标记 / 熔断计数 / 断言 / 漂移信号
        post_ctx = s.base_context()
        post_ctx["tool_name"] = tool_name
        post_ctx["tool_args"] = tool_args
        post_ctx["tool_result"] = result_text
        post_ctx.update(signals)
        post_ctx["converge_count"] = ctx.get("converge_count")
        post_ctx["converge_note"] = ctx.get("converge_note")
        post_ctx["_breaker_armed"] = ctx.get("_breaker_armed")
        post_ctx = await harness.run("post_tool_call", post_ctx)
        s.collect(post_ctx)

        guarded = post_ctx.get("tool_result")
        if isinstance(guarded, str) and guarded != result_text:
            yield ToolChunk(
                content=[TextBlock(type="text", text=guarded)],
                state=ToolResultState.SUCCESS,
                metadata=dict(last.metadata or {}) if last is not None else {},
            )
            return
        for chunk in chunks:
            yield chunk


def _collect_call_signals(s: HarnessSession, tool_name: str, result_text: str) -> dict[str, Any]:
    """从工具的**真实返回**里数阶段信号（不从全局状态反推，口径与 LangChain 版逐字相同）。"""
    call_candidates = 0
    call_picks = 0
    call_must_hits: int | None = None
    call_oncat: int | None = None
    call_offcat: int | None = None
    call_excluded: int | None = None
    call_over_budget: int | None = None
    if tool_name == "planner":
        s.planner_done = True
    elif tool_name == "item_picker":
        s.picker_attempted = True
        # 诊断走结构化侧信道（picker 返回前登记），不从模型可见文本里正则抠——
        # 文本截断 / 格式变化都伤不到信号。
        diag = consume_diagnostics("item_picker")
        if diag is not None:
            s.last_picks = call_picks = _as_opt_int(diag.get("picks")) or 0
            s.last_must_hits = call_must_hits = _as_opt_int(diag.get("must_have_hits"))
            s.last_oncat = call_oncat = _as_opt_int(diag.get("oncat_count"))
            s.last_offcat = call_offcat = _as_opt_int(diag.get("offcat_count"))
            s.last_excluded = call_excluded = _as_opt_int(diag.get("excluded_count"))
            s.last_over_budget = call_over_budget = _as_opt_int(diag.get("over_budget_count"))
        else:
            # 侧信道意外空：picks 退回文本兜底，三个诊断字段退化为 None =「不适用」，
            # 补搜闸 fail-open 不误触发（失效方向中性）。
            logger.warning("item_picker 诊断侧信道为空，picks 退回文本解析兜底")
            s.last_picks = call_picks = _count_picks(result_text)
            s.last_must_hits = s.last_oncat = s.last_offcat = None
            s.last_excluded = s.last_over_budget = None
    elif tool_name in _SEARCH_TOOLS:
        call_candidates = _count_candidates(result_text)
        s.fresh_candidates += call_candidates
    return {
        "call_candidates": call_candidates,
        "call_picks": call_picks,
        "call_must_hits": call_must_hits,
        "call_oncat": call_oncat,
        "call_offcat": call_offcat,
        "call_excluded": call_excluded,
        "call_over_budget": call_over_budget,
    }
