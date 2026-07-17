"""HarnessAgentMiddleware：Hook Pipeline 与 LangChain Agent 生命周期之间**唯一**的适配器。

控制面全部住在 Hook 里（``app/harness/hooks/``），本文件不做任何控制决策——它只负责：

1. 把 LangChain 的两个挂载点翻译成 Harness 的 6 个 Hook 点；
2. 提供 Hook 之间接力数据的通道（``_pending_inject`` / ``_pending_assertions`` / ``GuardState``）；
3. 把 Hook 对 context 的改写**写回**真实对象（messages / system_message / ToolMessage.content）；
4. 承担纯观测职责（AGUI 事件、工具 RT metrics、token 记账）——按 refdocs 17-2 §1.3，
   「只需要看不需要改」的逻辑不进 Hook Pipeline。

生命周期映射：

    awrap_model_call →  pre_think（模型调用前）
                        [模型]
                        post_reflect（模型调用后；可置 retry_nudge 要求当场重发）
    awrap_tool_call  →  pre_tool_call（工具执行前，可拒绝）
                        [工具]
                        post_tool_call（工具执行后，可改写结果）

``on_session_start`` / ``on_session_end`` 由 ``run_agent()`` 显式调用，不在中间件内。

**每个 Agent 实例独享一个本中间件**：内部状态（GuardState 的 LoopDetector 窗口 / 检索计数 /
终结标记、DriftState、阶段信号）都是 per-loop 语义，跨实例复用会让会话之间计数串台。
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from langchain.agents.middleware import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
    ToolCallRequest,
)
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from pydantic import ValidationError

from app.agent.fork_guard import current_fork_depth
from app.agent.token_budget import charge_tree_usage, tree_snapshot
from app.agent.tracing import current_trace_id
from app.api import monitor
from app.harness.hooks.drift_detector import DriftState
from app.harness.middleware import harness
from app.harness.phase_machine import get_phase_machine
from app.harness.state import GuardState
from app.observability import alerts, metrics
from app.tools._diagnostics import consume_diagnostics

logger = logging.getLogger("shoppingx.harness.adapter")

_SyncHandler = Callable[[ModelRequest], ModelResponse]
_AsyncHandler = Callable[[ModelRequest], Awaitable[ModelResponse]]
_SyncToolHandler = Callable[[ToolCallRequest], Any]
_AsyncToolHandler = Callable[[ToolCallRequest], Awaitable[Any]]
_ToolResult = ToolMessage | Any

# 单条行为摘要里工具参数的长度上限——摘要要进漂移检测的 LLM prompt，不能让长 query 撑爆。
_ARG_SUMMARY_MAX = 60


def _count_picks(result: str) -> int:
    """从 item_picker 的返回里数出真实 picks 数量。

    不能拿「item_picker 被调用过」当作「picks 已就绪」：候选全超预算 / 全被排除词淘汰时，
    item_picker 返回的是 ``picks: []``。把「调过」当成 picks_count=1 会有两个后果——阶段机在没有
    任何精选结果时就推进到 CONCLUDING（→ shopping_summary 空输出，正是 17-1 §5 列的失败模式），
    且回退闸从此永远不再触发（picks_count 恒 >0）。
    """
    try:
        data = json.loads(result)
        picks = data.get("picks")
        if isinstance(picks, list):
            return len(picks)
    except (json.JSONDecodeError, ValueError, AttributeError):
        pass
    # 走到这里说明不是合法 JSON——多半是被截断 Hook 按 token 预算截断了。截断只发生在长结果上，
    # 而长结果必然意味着 picks 非空；显式的空数组则一定完整出现在头部、不会被截掉。
    if '"picks": []' in result or '"picks":[]' in result:
        return 0
    return 1 if '"picks"' in result else 0


def _as_opt_int(value: object) -> int | None:
    """诊断侧信道字段的宽松取整：None 原样透传（= 本轮不适用），其余尽力转 int。"""
    if value is None:
        return None
    try:
        return int(value)  # type: ignore[call-overload]
    except (TypeError, ValueError):
        return None


# 会往候选池里添货的工具：主流程直搜，以及 fork 子 Agent 回传候选的两个派发口。
_SEARCH_TOOLS = frozenset({"item_search", "dispatch_tool", "parallel_dispatch_tool"})


def _count_candidates(result: str) -> int:
    """从检索类工具的真实返回里数出**本轮新召回**的候选数。

    **为什么不去数候选登记表**：登记表是个累积容器——上一轮的候选会被 ``load_candidates`` 读回来
    （供 item_picker 按 id hydrate），一旦拿它的总数当「本轮搜到了东西」的进展信号，换品类那轮就会
    被旧候选骗过去：planner 一跑完，阶段机看见「已有 12 件候选」直接推进 COMPARING，模型想搜键盘
    却发现 item_search 在 COMPARING 不放行。
    「仓库里有什么」与「这趟活干了什么」是两回事，不该共用一个计数器——和 :func:`_count_picks`
    坚持从 item_picker 的真实返回里数 picks 是同一条原则。
    """
    try:
        data = json.loads(result)
    except (json.JSONDecodeError, ValueError):
        # 子 Agent 回传的是自然语言总结（非 JSON）：数不出来就不计数。宁可少算不可多算——
        # 多算会把阶段机推过头，少算最多让模型多搜一次。
        return 0
    if not isinstance(data, dict):
        return 0
    cands = data.get("candidates")
    if isinstance(cands, list):
        return len(cands)
    total = data.get("total_recall")
    return total if isinstance(total, int) else 0


def _observe_tool(tool_name: str, elapsed_sec: float, status: str) -> None:
    """一次工具执行的可观测出口：喂 Prometheus 指标 + 喂 RT 告警窗。**同一个计时，两个消费者。**

    别把告警窗塞进 ``metrics.record_tool`` 里：``alerts`` 要读 ``metrics.SECURITY_EVENTS`` 做安全
    事件告警，反向再依赖就成了循环 import。打点位置只有这一处，两边取的是同一个 ``elapsed``，
    数据源不会漂移。

    **只有 ok 的调用进 RT 窗口。** 失败调用的耗时（尤其断路器 OPEN 时 ~0ms 的快速失败）会把 P95
    拉低，在故障最严重的时候反而报「已恢复」。错误面由 ``TOOL_CALLS{status="error"}`` 与断路器
    告警规则覆盖。
    """
    metrics.record_tool(tool_name, elapsed_sec, status)
    if status == "ok":
        # trace_id 一并存进窗口：告警触发时能直接给出「最慢那次」的 Langfuse 链接。
        alerts.record_tool_sample(tool_name, elapsed_sec * 1000.0, current_trace_id())


def _summarize_call(tool_name: str, args: Any) -> str:
    """把一次工具调用摘成 ``tool_name(参数文本)``。

    行为摘要必须带上参数，不能只有工具名：漂移检测的「目标遗忘」信号要拿用户 query 的关键词去
    匹配 Agent 最近在做什么，而工具名（item_search / price_compare）里永远不含 query 关键词——
    只喂工具名，命中数恒为 0，信号恒真。参数里的检索词才是模型「Think 的产物」。
    """
    if not isinstance(args, dict) or not args:
        return tool_name
    parts = [v.strip() for v in args.values() if isinstance(v, str) and v.strip()]
    if not parts:
        return tool_name
    return f"{tool_name}({' '.join(parts)[:_ARG_SUMMARY_MAX]})"


def _as_text(value: Any) -> str:
    """ToolMessage.content 的类型是 ``str | list[...]``，统一收敛成 str 再交给 Hook。"""
    return value if isinstance(value, str) else str(value)


class HarnessAgentMiddleware(AgentMiddleware):
    """把 Harness Hook Pipeline 挂到 LangChain Agent 生命周期。

    **接力通道 1 —— ``_pending_inject``**：Hook 往模型消息流里注入内容的唯一通道。
    post_tool_call / post_reflect 的 Hook 把 inject_messages 存进这里，下一次 awrap_model_call
    开头消费并追加到 messages——这样纠正提示、漂移警告、强制收尾指令才能真正进入模型视野。
    消费后经 ``persist_messages`` 随 ModelResponse.result 落 state（不是一次性视图）：注入若
    下一轮就从 prompt 里消失，第 N+1 轮请求便不再是第 N 轮的字节延伸，隐式前缀缓存链每轮被
    斩断（实测塌到只剩 system 段命中）；持久化同时让模型后续轮次仍看得到自己被纠正过什么。

    **接力通道 2 —— ``_pending_assertions``**：三类断言分别在 pre_tool_call（sequencing）和
    post_tool_call（schema / semantic）产生，而汇总它们的 ``assertion_handler`` 挂在 post_reflect。
    三个 Hook 点各自拿到的是**不同的 context dict**，断言结果不会自己流过去，必须在这里接力。

    **接力通道 3 —— ``GuardState``**：Hook 是模块级函数，无处安放 per-loop 计数，统一经
    ``context["_guard"]`` 传递。
    """

    def __init__(
        self,
        *,
        original_query: str = "",
        image_paths: Sequence[str] = (),
        guard: GuardState | None = None,
    ) -> None:
        super().__init__()
        self._original_query = original_query
        self._image_paths = tuple(image_paths)
        self._guard = guard if guard is not None else GuardState()
        self._round_counter = 0
        self._called_tools: set[str] = set()
        self._drift_state = DriftState()
        self._recent_actions: list[str] = []
        # 挂起的注入消息：Hook 产出 → 下一轮 pre_think 消费
        self._pending_inject: list[dict[str, str]] = []
        # 挂起的断言失败：pre/post_tool_call 产出 → 本轮 post_reflect 消费
        self._pending_assertions: list[dict[str, Any]] = []
        # 上一次读到的全树 token 总量，用于给漂移检测的「成本失控」信号算每轮增量
        self._last_total_tokens = 0
        # 阶段信号累积器
        self._planner_done = False
        self._picker_attempted = False  # item_picker 是否真的跑过（区分「没调」与「调了返回空」）
        self._last_picks = 0  # 最近一次 item_picker 返回的 picks 数量
        # 最近一次 item_picker 的 must_have 池内命中件数；None = 那次没传 must_have（不适用）
        self._last_must_hits: int | None = None
        # 最近一次 item_picker 的品类一致性计数（oncat/offcat）；None = 那次没跑相关性门
        self._last_oncat: int | None = None
        self._last_offcat: int | None = None
        # 最近一次 item_picker 的硬淘汰归因计数（命中排除词 / 超预算）；None = 无诊断
        self._last_excluded: int | None = None
        self._last_over_budget: int | None = None
        self._fresh_candidates = 0  # 本轮检索**新召回**的候选数（跨轮读回的旧候选不算进展）
        # 阶段状态机**不在这里创建**：它归 on_session_start 的 phase_init Hook 管（会话生命周期
        # 拥有它，而不是某个中间件实例）。没开会话就跑 Agent（单测 / examples）时 ContextVar 为
        # None，阶段门自然失效——这是对的：没有会话，就没有「对话阶段」。

    # ── context 构造与接力 ──

    def _build_base_context(self) -> dict[str, Any]:
        return {
            "original_query": self._original_query,
            "round_number": self._round_counter,
            "called_tools": set(self._called_tools),
            "_drift_state": self._drift_state,
            "_guard": self._guard,
        }

    def _build_recent_actions_summary(self) -> str:
        if not self._recent_actions:
            return ""
        return "; ".join(self._recent_actions[-9:])

    def _consume_pending_inject(self) -> list[SystemMessage]:
        if not self._pending_inject:
            return []
        msgs = [
            SystemMessage(content=m["content"]) for m in self._pending_inject if m.get("content")
        ]
        self._pending_inject.clear()
        return msgs

    def _collect_inject(self, ctx: dict[str, Any]) -> None:
        inject = ctx.get("inject_messages")
        if inject:
            self._pending_inject.extend(inject)

    def _collect_assertions(self, ctx: dict[str, Any]) -> None:
        failed = ctx.get("assertions_failed")
        if failed:
            self._pending_assertions.extend(failed)

    # ── 纯观测（不是控制面，故不进 Hook；refdocs 17-2 §1.3 的 callback 定位）──

    def _charge_usage(self, response: Any) -> None:
        """把本次模型调用的用量计进全树成本。

        计费绝不反噬主链路：拿不到 result 或计费途中任何异常（畸形 usage_metadata / 非常规 result
        形状）都吞掉，绝不让记账拖垮模型调用。
        """
        result = getattr(response, "result", None)
        if result is None:
            return
        try:
            charge_tree_usage(result)
        except Exception:
            logger.debug("token 计费失败，跳过本次（不反噬主链路）", exc_info=True)

    def _track_token_delta(self) -> None:
        """把本次模型调用消耗的 token 追加到漂移检测的 token_history（信号 4：成本失控）。"""
        snap = tree_snapshot()
        if not snap:
            return
        total = int(snap.get("input_tokens", 0)) + int(snap.get("output_tokens", 0))
        delta = total - self._last_total_tokens
        self._last_total_tokens = total
        if delta > 0:
            self._drift_state.token_history.append(delta)

    # ── 开局：确定性预置 planner ──

    async def abefore_agent(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        """本轮 loop 开跑**之前**先把 planner 跑掉，把「调用 + 结果」两条消息写进 state。

        **省掉的是什么**：改前，第 1 轮模型调用（混合档下全程唯一开 reasoning、也最贵的一轮）
        花一整次往返，产出只有一个工具名——``planner``。可这个决定压根不需要模型做：planner 判
        ``retrieval``（reuse / augment / search）靠的是系统确定性注入的会话状态 + 候选登记表
        （见 :func:`app.tools.planner._render_prior_context`），它**不消费主 loop 模型的任何输出**，
        入参只有用户原话。既然如此，就不该为它付一次模型往返。

        **省掉的不是编排决策**：第 1 轮仍在、reasoning 仍开（``reasoning_boost`` 看
        ``round_number == 1``，预置不占轮次），只是它面对的问题从「我该调什么工具」变成「plan
        已在手，我该怎么检索」——单平台直搜还是 fork 并行、reuse 时直接精挑既有候选。那才是真正
        需要推理的一步。

        **为什么在 loop 内（middleware）而不是在 run_agent 里手工预跑**：域内长期偏好注入
        （``hooks/preference_inject``）、阶段机 PLANNING→SEARCHING（``hooks/phase_transition``
        读 ``planner_output_ready``）都挂在「planner 在 loop 内被调过」这个事实上。走这里、照常
        触发 ``post_tool_call`` 与阶段信号，它们原样生效，一行都不用复刻。消息形状也与改前逐字
        同构（AIMessage(tool_call) + ToolMessage），前缀缓存不受扰动。

        **跳过 pre_tool_call 是有意的**：那一层的闸（工具白名单 / 阶段门 / 熔断 / 循环检测 /
        检索预算）管的是「模型的自由发挥」，而这次调用是机制自己决定的、不是模型要的——让它去
        过一道为约束模型而设的闸，只会平添「机制被自己的护栏拦下」这种荒诞失败。

        降级：planner 抛错 / 拿不到本轮 query → 返回 None，回到改前的老路（模型自己决定调
        planner，prompt 里那条规则仍在）。预置是快路径，不是唯一路径。
        """
        # 子 Agent 不预置：它的活是「按 demands 检索」，demands 里已带着主流程拆好的结构化字段，
        # prompt 本就规定它不要再调 planner（同质 fork 只隔离上下文，不裁剪工具集）。
        if current_fork_depth() >= 1 or not self._original_query:
            return None
        from app.tools.planner import planner as planner_tool  # 懒 import：防注册期导入环

        # 有参考图 → 看图**必须先于 planner**（M20）。同上一段的论证：这个决定不需要模型做（有图
        # 就得看，没图就不看），所以不为它付一次模型往返。更要紧的是**顺序**：planner 是拿用户原话
        # 拆结构化字段的，若图的结论晚于它产出，「只发一张图 + 想买这个」这类 query 会让 planner
        # 拆出一片空白（没品类、没预算），后面全链路跟着空转。故先看图、把结论并进 intent 再拆。
        prefix: list[Any] = []
        intent = self._original_query
        if self._image_paths:
            vision_msgs, vision_hint = await self._prefill_vision()
            prefix.extend(vision_msgs)
            if vision_hint:
                intent = f"{intent}\n\n[用户上传的参考图，已识别] {vision_hint}"

        args = {"intent": intent}
        call_id = "prefill_planner"
        try:
            # 传 tool_call 形态 → BaseTool 直接返回 ToolMessage（与 loop 内 ToolNode 的产物同构）。
            # planner 自己会报 AGUI 事件、记 token，故观测与记账两条通路都不必在这里另接。
            tool_msg = await planner_tool.ainvoke(
                {"name": "planner", "args": args, "id": call_id, "type": "tool_call"}
            )
        except Exception:
            logger.warning("planner 预置失败，回退为模型自行调用（老路径）", exc_info=True)
            return None
        if not isinstance(tool_msg, ToolMessage):
            return None

        # 阶段信号与行为摘要：与 awrap_tool_call 里真调一次 planner 记的东西完全一致——第 1 轮
        # post_reflect 据 planner_output_ready 把阶段从 PLANNING 推到 SEARCHING。
        self._planner_done = True
        self._called_tools.add("planner")
        self._recent_actions.append(_summarize_call("planner", args))

        post_ctx = self._build_base_context()
        post_ctx["tool_name"] = "planner"
        post_ctx["tool_args"] = args
        post_ctx["tool_result"] = _as_text(tool_msg.content)
        post_ctx = await harness.run("post_tool_call", post_ctx)
        self._collect_assertions(post_ctx)
        # 偏好注入落在 _pending_inject，由下一次 awrap_model_call 开头消费——那正是第 1 轮。
        self._collect_inject(post_ctx)
        guarded = post_ctx.get("tool_result")
        if isinstance(guarded, str) and guarded != tool_msg.content:
            tool_msg = tool_msg.model_copy(update={"content": guarded})

        ai_msg = AIMessage(
            content="",
            tool_calls=[{"name": "planner", "args": args, "id": call_id}],
        )
        return {"messages": [*prefix, ai_msg, tool_msg]}

    # 一次任务最多看几张图：每张都是一次 VL 往返 + 一段上下文，传一堆图既烧预算又稀释意图。
    MAX_PREFILL_IMAGES = 3

    async def _prefill_vision(self) -> tuple[list[Any], str]:
        """开局把参考图逐张看掉，返回（要写进 state 的消息对, 给 planner 的一句话线索）。

        消息形状与真调一次工具逐字同构（AIMessage(tool_call) + ToolMessage），主 loop 因此能像
        读任何工具结果一样读到图的结论；AGUI 事件由 image_understand 内部照常上报，前端能看见
        「正在看图」这一步。看图失败（未配 LLM_VISION / 图读不到 / 模型抽风）不阻断——工具自身
        已降级返回 note，主 loop 照常按文字意图往下走。
        """
        from app.tools.image_understand import image_understand  # 懒 import：防注册期导入环

        msgs: list[Any] = []
        hints: list[str] = []
        for idx, name in enumerate(self._image_paths[: self.MAX_PREFILL_IMAGES]):
            call_id = f"prefill_vision_{idx}"
            args = {"filename": name}
            try:
                out = await image_understand.ainvoke(args)
            except Exception:
                logger.warning("参考图预读失败：%s", name, exc_info=True)
                continue
            content = out.model_dump_json(exclude_none=True)
            msgs.append(
                AIMessage(
                    content="",
                    tool_calls=[{"name": "image_understand", "args": args, "id": call_id}],
                )
            )
            msgs.append(ToolMessage(content=content, tool_call_id=call_id, name="image_understand"))
            self._called_tools.add("image_understand")
            self._recent_actions.append(_summarize_call("image_understand", args))
            if not out.degraded and out.search_query:
                hints.append(f"{out.subject or out.category}（检索词：{out.search_query}）")
        return msgs, "；".join(hints)

    # ── model call ──

    @staticmethod
    def _ai_message(response: Any) -> Any:
        # result 可能带持久化注入前缀（system/human），AI 回复恒在其后 → 从尾部找。
        result = getattr(response, "result", None) or []
        for msg in reversed(result):
            if isinstance(msg, AIMessage):
                return msg
        return result[0] if result else None

    async def awrap_model_call(
        self, request: ModelRequest, handler: _AsyncHandler
    ) -> ModelResponse:
        # 终结直出（延迟归因 round2 刀2）：shopping_summary 已产出面向用户的完整清单
        # （ToolMessage.content 就是 out.summary），此处若再唤起模型，它只会把同一份清单
        # 复述一遍（实测 729 tok / 7.5s，还引入转录出错风险）。直接用终结产物合成收尾消息，
        # 无 tool_calls → loop 自然终止。与预算 fallback 档同一先例（见下），同样不过 post_reflect。
        # 从 artifact 取而非 tool_result 字符串：artifact 不经截断 Hook，拿到的是完整原文。
        # chat_fallback 不走此路（其返回非 content_and_artifact，且闲聊收尾本就该由模型口吻说）。
        if self._guard.terminal_reached:
            from app.tools.shopping_summary import ShoppingSummaryOutput  # 懒 import 防注册期导入环

            for m in reversed(request.messages):
                art = getattr(m, "artifact", None) if isinstance(m, ToolMessage) else None
                if isinstance(art, ShoppingSummaryOutput) and art.summary:
                    return ModelResponse(result=[AIMessage(content=art.summary)])
            # 没有可直出的终结产物（如 chat_fallback）→ 照常唤起模型收尾

        self._round_counter += 1
        self._guard.think_step += 1
        # 终结提醒的配额是**每次模型调用**一次，不是整个 loop 一次：模型第 3 轮想用纯文字收尾
        # 被催过，第 12 轮再想蒙混时同样得被催。这个计数器只防「同一次调用里无限重发」，故每次进
        # awrap_model_call 都清零（对齐迁移前 for _ in range(MAX_TERMINAL_NUDGE_RETRIES) 的语义）。
        self._guard.terminal_nudge_retries = 0
        # 进入 Think 即上报 assistant_call，让前端看到「Agent 思考中」中间态。
        # 子 Agent 的 thread 无前端连接 → 事件静默丢弃（上下文隔离，与 fork 设计一致）。
        await monitor.report_assistant_call(step=str(self._guard.think_step))

        # 1. pre_think：消费挂起注入 → Hook 可再改 messages / system_message（预算 hint、压缩）
        #
        # persist_messages：本次请求视图里**新增**的消息（纠正注入 / 预算 hint），随本次
        # ModelResponse.result 一并落 state。曾经它们只进视图、下一轮就消失——于是第 N+1 轮的
        # prompt 不再是第 N 轮的字节延伸，隐式前缀缓存链每轮被斩断（eval q05/q03/q16 命中率
        # 卡死在 system 段 2048，未命中 7.5 万 token/条）。落 state 后下一轮 prompt 严格延伸
        # 上一轮，缓存接链；副作用是纠正文本此后一直在历史里——这本就更对：模型不该下一轮就
        # 忘了自己被纠正过。视图专属的改动（压缩截断）不在此列——它们幂等重放，不破前缀。
        inject = self._consume_pending_inject()
        messages: list[AnyMessage] = [*request.messages, *inject]
        ctx = self._build_base_context()
        ctx["messages"] = messages
        ctx["persist_messages"] = list(inject)
        ctx["system_message"] = request.system_message
        ctx["recent_actions_summary"] = self._build_recent_actions_summary()
        ctx = await harness.run("pre_think", ctx)

        # 1.5 预算 fallback 档（refdocs 16-4 §6）：连一次 LLM 调用都付不起了，直接把规则兜底的
        # 回答当作模型输出返回。无 tool_calls → AgentLoop 自然终止。
        #
        # **刻意绕过 post_reflect**：那里的终结纪律 Hook 会因为「没调终结工具就想收尾」而要求当场
        # 重发模型——可预算正是为此耗尽的，再重发一次纯属把最后的钱也烧掉。fallback 不是模型的
        # 失误，是系统的决定，不该被纠正回路拉回去。
        fallback = ctx.get("fallback_answer")
        if isinstance(fallback, str) and fallback:
            self._guard.terminal_reached = True  # 后续任何工具调用都会被 terminal_reached_gate 拦下
            persisted: list[AnyMessage] = ctx.get("persist_messages") or []
            return ModelResponse(result=[*persisted, AIMessage(content=fallback)])

        overrides: dict[str, Any] = {"messages": ctx["messages"]}
        if ctx.get("system_message") is not request.system_message:
            overrides["system_message"] = ctx["system_message"]
        # 降档换模型（lite / minimal）：Hook 只做决策，真正的 override 在这里落地。
        model_override = ctx.get("model_override")
        if model_override is not None:
            overrides["model"] = model_override
        request = request.override(**overrides)

        # 2. 调模型。视图新增消息前置进 result 落 state（顺序=模型实际所见：注入在前、回复在后；
        # 路由与 _ai_message 都按「最后一条 AIMessage」取，前缀混入 system/human 不影响循环）。
        response = await handler(request)
        self._charge_usage(response)
        self._track_token_delta()
        persist: list[AnyMessage] = ctx.get("persist_messages") or []
        if persist:
            response = ModelResponse(
                result=[*persist, *response.result],
                structured_response=getattr(response, "structured_response", None),
            )

        # 3. post_reflect
        reflect_ctx = await self._run_post_reflect(request, response)

        # 4. terminal_enforcer 要求当场重发（模型没调工具就想收尾 → loop 会直接结束，等不到下一轮）
        retry_nudge = reflect_ctx.get("retry_nudge")
        if retry_nudge:
            ai_msg = self._ai_message(response)
            nudge_msg = HumanMessage(content=retry_nudge)
            request = request.override(messages=[*request.messages, ai_msg, nudge_msg])
            retried = await handler(request)
            self._charge_usage(retried)
            self._track_token_delta()
            # 首答 + nudge 一并落 state（曾被丢弃 → 下一轮 prompt 缺这两条，同样斩断缓存链）。
            # response.result 已含 persist 前缀与首答；路由只看最后一条 AIMessage，仍是重发的回复。
            response = ModelResponse(
                result=[*response.result, nudge_msg, *retried.result],
                structured_response=getattr(retried, "structured_response", None),
            )

        return response

    async def _run_post_reflect(self, request: ModelRequest, response: Any) -> dict[str, Any]:
        # 「轮」的边界：解除上一轮回退闭锁（见 PhaseStateMachine.regress——回退后同轮不得
        # 再前进，前进资格从下一轮凭新证据重新挣）。阶段机是主 loop 独有，子 loop 不 tick。
        if current_fork_depth() == 0:
            machine = get_phase_machine()
            if machine is not None:
                machine.begin_round()
        ai_msg = self._ai_message(response)
        ctx = self._build_base_context()
        ctx["recent_actions_summary"] = self._build_recent_actions_summary()
        ctx["messages"] = request.messages
        ctx["response_ai_message"] = ai_msg
        ctx["response_has_tool_calls"] = bool(getattr(ai_msg, "tool_calls", None))
        # 阶段信号——从工具名 + 结构化数据源判断，不做字符串匹配
        ctx["planner_output_ready"] = self._planner_done
        # 本轮**新召回**的候选数（不含跨轮读回的旧候选，见 _count_candidates 的 docstring）
        ctx["total_candidates"] = self._fresh_candidates
        ctx["picks_count"] = self._last_picks
        ctx["picker_attempted"] = self._picker_attempted
        ctx["must_have_hits"] = self._last_must_hits
        ctx["oncat_count"] = self._last_oncat
        ctx["offcat_count"] = self._last_offcat
        ctx["excluded_count"] = self._last_excluded
        ctx["over_budget_count"] = self._last_over_budget
        # 接力本轮攒下的断言失败——assertion_handler 在 post_reflect 上等着消费它们
        if self._pending_assertions:
            ctx["assertions_failed"] = list(self._pending_assertions)
            self._pending_assertions.clear()

        ctx = await harness.run("post_reflect", ctx)
        # 补搜闸宣判「这池子不够用」后，污染批不再算「本轮已搜到货」——不清掉的话，下一次
        # post_reflect 里 SEARCHING 会凭旧计数被立刻推回 COMPARING（见 refine_backfill）。
        if ctx.pop("reset_fresh_candidates", False):
            self._fresh_candidates = 0
        self._collect_inject(ctx)
        return ctx

    def wrap_model_call(self, request: ModelRequest, handler: _SyncHandler) -> ModelResponse:
        """同步路径不接 Hook Pipeline——全链路 async 是本项目硬约束，此路仅为接口完整性存在。"""
        raise NotImplementedError(
            "ShoppingX 全链路 async：请用 ainvoke/astream 驱动 Agent。"
            "同步路径会绕过整个 Harness 控制面（工具闸 / 截断 / 熔断 / 压缩），故显式禁用。"
        )

    # ── tool call ──

    async def awrap_tool_call(
        self, request: ToolCallRequest, handler: _AsyncToolHandler
    ) -> _ToolResult:
        tool_name = request.tool_call.get("name", "")
        tool_args = request.tool_call.get("args", {})
        tool_call_id = request.tool_call.get("id", "")

        # 1. pre_tool_call：全部硬闸 + 阶段门 + 顺序断言 + 熔断判定
        ctx = self._build_base_context()
        ctx["tool_name"] = tool_name
        ctx["tool_args"] = tool_args
        ctx["tool_call_id"] = tool_call_id
        ctx = await harness.run("pre_tool_call", ctx)

        if ctx.get("_rejected"):
            reason = ctx.get("_reject_reason", "Hook 拒绝")
            # raw=True 的哨兵原样回模型（它们本就是写给模型的完整指令）；否则加前缀标明来源。
            content = reason if ctx.get("_reject_raw") else f"[Harness 拒绝] {reason}"
            # 被闸拦下的调用同样要喂 LoopDetector：模型换着参数硬撞同一道闸时（哨兵文案每次
            # 相同、无升级），拒绝路径不计数就是循环检测的盲区，只剩 recursion_limit 硬兜底。
            # 攒到阈值就在哨兵尾部追加打转升级提示。回放路径（tool_memo）已自己喂过并附了
            # 提示（置 _detector_fed），这里跳过防双记。
            if not ctx.get("_detector_fed") and self._guard.detector.record(tool_name):
                content += f"\n\n[系统提示] {self._guard.detector.nudge_message(tool_name)}"
            return ToolMessage(content=content, tool_call_id=tool_call_id, name=tool_name)

        self._collect_assertions(ctx)
        self._collect_inject(ctx)

        # 2. 执行工具。计时与熔断计数只覆盖**真实执行**——被闸拦下的哨兵不算。
        start = time.monotonic()
        try:
            result = await handler(request)
        except Exception as exc:
            _observe_tool(tool_name, time.monotonic() - start, "error")
            # 参数校验类失败（ValidationError）不计入熔断：那是调用方（模型）的锅，不是工具
            # 基础设施故障。断路器进程级共享，计入会让一个会话连发 3 次畸形参数就把该工具对
            # 全进程所有会话熔断 60s。不记成败也不会卡死断路器：HALF_OPEN 下一次调用照常放行探测。
            if ctx.get("_breaker_armed") == tool_name and not isinstance(exc, ValidationError):
                from app.harness.hooks.tool_breaker import get_tool_breaker

                get_tool_breaker(tool_name).record_failure()
            raise
        # 参数校验失败不算执行成功：ToolNode 把 ToolInvocationError 转成 status="error" 的
        # ToolMessage **正常返回**（langgraph _default_handle_tool_errors 只吞这一类，其余异常
        # 照常 raise 进上面的 except），handler 层面「没抛异常」骗不了这里——工具没真正跑过。
        # 线上实锤（gcjp 会话 d0724e95）：模型把 list 参数吐成 JSON 字符串连挂 4 次，全被当成
        # 「已精挑」记入 called_tools，phase_check 底线 3 判据被污染放行收尾；阶段信号照推，
        # 收线通告还缀在错误消息尾部教唆模型跳 shopping_summary。
        # 于是与被闸拦下的哨兵同罪同罚：不记 called_tools / recent_actions、不推阶段信号、
        # 不跑 post_tool_call（收线通告 / memo / 断言全以「工具真执行了」为前提）、不给看门狗
        # 续命。只喂 LoopDetector——换着（或不换）参数硬撞同一个校验错误正是打转，攒到阈值在
        # 错误尾部追加升级提示。熔断不记 failure：口径同 except 里的 ValidationError 豁免。
        if isinstance(result, ToolMessage) and result.status == "error":
            _observe_tool(tool_name, time.monotonic() - start, "error")
            if self._guard.detector.record(tool_name):
                nudge = f"\n\n[系统提示] {self._guard.detector.nudge_message(tool_name)}"
                return result.model_copy(update={"content": _as_text(result.content) + nudge})
            return result
        _observe_tool(tool_name, time.monotonic() - start, "ok")
        # 看门狗口径的「实质进展」：工具真实执行成功。被闸拦下的哨兵（上方 early-return）与
        # tool_memo 回放（拒绝通道）都到不了这里——模型空转不给看门狗续命。
        self._guard.last_progress_at = time.monotonic()
        self._guard.watchdog_nudged_at = 0.0

        # 3. 记录已调用工具。行为摘要带上参数文本——漂移检测靠它匹配 query 关键词
        self._called_tools.add(tool_name)
        self._recent_actions.append(_summarize_call(tool_name, tool_args))
        if len(self._recent_actions) > 30:
            self._recent_actions = self._recent_actions[-20:]

        is_tool_message = isinstance(result, ToolMessage)
        result_text = _as_text(result.content) if is_tool_message else str(result)

        # 阶段信号一律从**工具的真实返回**里数，不从全局状态反推（picks 如此，候选也如此）
        call_candidates = 0
        call_picks = 0
        call_must_hits: int | None = None
        call_oncat: int | None = None
        call_offcat: int | None = None
        call_excluded: int | None = None
        call_over_budget: int | None = None
        if tool_name == "planner":
            self._planner_done = True
        elif tool_name == "item_picker":
            self._picker_attempted = True
            # 诊断走结构化侧信道（picker 返回前登记，见 app/tools/_diagnostics.py），
            # 不再从模型可见文本里正则抠——文本截断 / 格式变化都伤不到信号。
            diag = consume_diagnostics("item_picker")
            if diag is not None:
                self._last_picks = call_picks = int(diag.get("picks") or 0)
                self._last_must_hits = call_must_hits = _as_opt_int(diag.get("must_have_hits"))
                self._last_oncat = call_oncat = _as_opt_int(diag.get("oncat_count"))
                self._last_offcat = call_offcat = _as_opt_int(diag.get("offcat_count"))
                self._last_excluded = call_excluded = _as_opt_int(diag.get("excluded_count"))
                self._last_over_budget = call_over_budget = _as_opt_int(
                    diag.get("over_budget_count")
                )
            else:
                # 侧信道意外空（picker 未走到登记点的旁路）：picks 退回文本兜底；三个诊断
                # 字段退化为 None = 「不适用」，补搜闸 fail-open 不误触发（失效方向中性）。
                logger.warning("item_picker 诊断侧信道为空，picks 退回文本解析兜底")
                self._last_picks = call_picks = _count_picks(result_text)
                self._last_must_hits = call_must_hits = None
                self._last_oncat = call_oncat = None
                self._last_offcat = call_offcat = None
                self._last_excluded = call_excluded = None
                self._last_over_budget = call_over_budget = None
        elif tool_name in _SEARCH_TOOLS:
            call_candidates = _count_candidates(result_text)
            self._fresh_candidates += call_candidates

        # 4. post_tool_call：截断 / 提示 / 终结标记 / 熔断计数 / 断言 / 漂移信号
        post_ctx = self._build_base_context()
        post_ctx["tool_name"] = tool_name
        post_ctx["tool_args"] = tool_args
        post_ctx["tool_result"] = result_text
        # 本次调用的阶段信号（transition_notice 靠它把「阶段收线」通告缀在触发它的结果尾部——
        # 转移本体在 post_reflect，但那晚一轮：模型在下一次 post_reflect 之前就已决定了下一步）。
        post_ctx["call_candidates"] = call_candidates
        post_ctx["call_picks"] = call_picks
        post_ctx["call_must_hits"] = call_must_hits
        post_ctx["call_oncat"] = call_oncat
        post_ctx["call_offcat"] = call_offcat
        post_ctx["call_excluded"] = call_excluded
        post_ctx["call_over_budget"] = call_over_budget
        post_ctx["converge_count"] = ctx.get("converge_count")
        post_ctx["converge_note"] = ctx.get("converge_note")
        post_ctx["_breaker_armed"] = ctx.get("_breaker_armed")
        post_ctx = await harness.run("post_tool_call", post_ctx)
        self._collect_assertions(post_ctx)
        self._collect_inject(post_ctx)

        # 5. 把 Hook 改写过的结果写回。不原地改 ToolMessage（可能被别处引用），返回副本。
        guarded = post_ctx.get("tool_result")
        if is_tool_message and isinstance(guarded, str) and guarded != result_text:
            return result.model_copy(update={"content": guarded})
        return result

    def wrap_tool_call(self, request: ToolCallRequest, handler: _SyncToolHandler) -> _ToolResult:
        """同步路径不接 Hook Pipeline——理由同 :meth:`wrap_model_call`。"""
        raise NotImplementedError(
            "ShoppingX 全链路 async：请用 ainvoke/astream 驱动 Agent。"
            "同步路径会绕过整个 Harness 控制面（工具闸 / 截断 / 熔断 / 压缩），故显式禁用。"
        )


def build_agent_middleware(
    *,
    original_query: str = "",
    image_paths: Sequence[str] = (),
    guard: GuardState | None = None,
) -> list[AgentMiddleware]:
    """主 / 子 AgentLoop 共用的中间件栈（每次新建）。

    **栈里只有一个中间件**——控制面全在 Hook Pipeline 里。截断 / 循环检测 / 熔断 / 各类硬闸 /
    压缩 / 终结纪律都是 Hook，按 priority 排序执行，新增一道检查只要写个函数 + 一行
    ``@harness_hook``，不动这里、也不动主循环（refdocs 17-2 §7）。

    每次新建而非复用单例：``GuardState`` 有状态（LoopDetector 窗口、检索计数、终结标记），不同
    Agent 实例必须各自独立计数。主 loop 在 ``main_agent``、子 loop 在 ``dispatch_tool`` 都用这个
    工厂，保证两者挂的是**同一套**控制面——同质 fork 的硬约束在控制面层的落实。

    ``original_query`` 供漂移检测 / 语义断言做对齐基准。**子 Agent 不传**（``dispatch_tool`` 里
    就是空调用），于是漂移检测在 fork 里自动跳过——这是有意的：子 loop 的越界由机制兜（深度闸 /
    迭代上限 / 检索预算），不靠再加一层 LLM 判定；而且「强制收尾 → 调 shopping_summary」这种纠正
    语义只对主 loop 成立。阶段机同理只在 depth 0 生效。Hook 集合本身与主 loop 完全同构。
    """
    from app.harness.setup import setup_harness

    setup_harness()  # 幂等：首次调用注册全部 Hook
    return [
        HarnessAgentMiddleware(original_query=original_query, image_paths=image_paths, guard=guard)
    ]
