"""主链路执行入口：一条 query 从入口跑到收尾（``run_agent`` 的唯一实现）。

编排顺序是建会话目录 → 读历史 / P_t → 跑 loop → 收尾落产物 → 记忆判定。三处值得单独记住的
接线（都是 AgentScope 运行时的特性，迁移时逐条验过）：

1. **loop 怎么跑**：``Agent.reply_stream`` + 事件泵（见 ``app/agent/events.py``），
   而不是「跑完拿返回值」——AGUI 事件要在过程中实时推给前端。
2. **on_session_end 不在这里调**：由 ``HarnessAgentAdapter.on_reply`` 在框架内部改写最终
   ``Msg``，所以这里拿到的 ``final_text`` **已经是审核后的**。别再补一次——重复审核会把哨兵
   文案二次剥离，且 ``output_audit`` 的计数会翻倍。
3. **会话恢复两条腿**：``agent_state.json``（``AgentState`` 全量落盘）优先，缺失时退回精简的
   (q,a) 回放。任一条失效另一条还能把会话续上，见 :func:`_load_state`。

与运行时无关的那几件事（当轮上下文拼装、产物落盘、配额记账）住在 ``app/agent/session_io.py``。
"""

import asyncio
import json
import logging
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

from agentscope.message import Msg, TextBlock
from agentscope.state import AgentState

from app.agent.agents import build_main_agent
from app.agent.events import pump_events
from app.agent.platform_scope import platform_scope
from app.agent.retrieval_budget import reset_tree as reset_retrieval_tree
from app.agent.session_io import (
    MAIN_AGENT_TIMEOUT_SEC,
    charge_quota,
    inject_runtime_context,
    write_session_artifacts,
)
from app.agent.token_budget import budget_status, set_task_cap, tree_snapshot
from app.agent.token_budget import reset_tree as reset_token_tree
from app.agent.tracing import current_trace_id, turn_span
from app.agent.usage import summarize_usage
from app.api import monitor
from app.api.context import (
    begin_learned_prefs,
    get_learned_pref_items,
    get_learned_prefs,
    get_session_domains,
    get_session_pt,
    reset_dest_country,
    reset_original_query,
    reset_retrieval_mode,
    reset_session_domains,
    reset_session_pt,
    reset_session_tasks,
    set_original_query,
    set_session_pt,
)
from app.db.quota import remaining_usd
from app.harness.budgets import fork_budget_scope, fork_concurrency_scope
from app.harness.middleware import harness
from app.harness.phase_machine import reset_phase_machine
from app.harness.setup import setup_harness
from app.memory.curator import curate_turn
from app.memory.history import append_turn, load_prior_turns
from app.memory.injector import build_history_block, record_search_history
from app.memory.session_state import load_pt
from app.memory.store import get_store
from app.observability import metrics
from app.recall.semantic_cache import (
    TurnCacheEntry,
    get_turn_cache,
    preference_fingerprint,
    turn_cache_enabled,
    turn_cache_key,
    turn_is_cacheable,
)
from app.tools._bundle import reset_session_bundle
from app.tools._candidates import (
    load_candidates,
    persist_candidates,
    render_prior_candidates,
    reset_candidates,
)
from app.tools._diagnostics import reset_diagnostics
from app.tools.shopping_summary import ShoppingSummaryOutput
from app.utils.path_utils import ensure_session_dir
from app.utils.thread_ctx import thread_scope

logger = logging.getLogger("shoppingx.orchestrator")

# AgentState 落盘文件名（每会话一份，与 turns.json / history.json 并列在 session_dir 下）。
STATE_FILE = "agent_state.json"


def _state_path(session_dir: Path) -> Path:
    return session_dir / STATE_FILE


def _load_state(session_dir: Path) -> AgentState | None:
    """读回上一轮落盘的 ``AgentState``（续聊恢复的第一条腿）。

    读不到 / 读坏了都返回 ``None``——调用方会退回第二条腿（精简 (q,a) 回放）。恢复是**加成**，
    不是前提：一份坏掉的 state 让整轮聊天起不来，比少一段上下文糟得多。
    """
    path = _state_path(session_dir)
    if not path.exists():
        return None
    try:
        return AgentState.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("agent_state.json 解析失败，退回历史回放（%s）", path, exc_info=True)
        return None


def _save_state(session_dir: Path, state: AgentState) -> None:
    """把本轮结束时的 ``AgentState`` 落盘，供下一轮 / 换进程恢复。

    与 ``append_turn`` 写库是**双做**而非二选一：state 保住的是「模型看得见的完整上下文」
    （含工具调用与结果，恢复后模型不必重新推一遍），turns 表保住的是「人看得懂的对话」
    （前端回看、跨会话行为历史、评测取样都吃它）。任一条腿失效，另一条还能把会话续上。
    """
    try:
        _state_path(session_dir).write_text(state.model_dump_json(), encoding="utf-8")
    except Exception:
        logger.warning("写 agent_state.json 失败，下轮退回历史回放", exc_info=True)


def _extract_summary(messages: Sequence[Msg]) -> ShoppingSummaryOutput | None:
    """从消息流里取最后一次 ``shopping_summary`` 的结构化产物。

    LangChain 版认的是 ``ToolMessage.artifact`` 的**类型**；AgentScope 的工具结果里没有
    artifact 这条侧信道，结构化结果就是 ``ToolResultBlock`` 里那段 JSON 文本（见
    ``app/tools/_as_tools.py`` 的 ``_to_text``）。所以这里按**工具名 + 能否验成
    ShoppingSummaryOutput** 双条件认。

    **两处方向都必须是倒着来**（消息倒着、消息内的 block 也倒着），且解析失败要**接着往前
    找**而不是就此认输：AgentScope 把一整轮的 tool_call / tool_result 全塞进同一条 assistant
    消息的 content 里，而 ``shopping_summary`` 在一轮里被调好几次是常态——前几次撞上 harness
    的阶段闸（「还没精挑就想出清单」）拿回哨兵文案，最后一次才真出清单。正着找第一个、解析
    失败就 return None，等于永远只看得到被拒绝的那次，items 恒为空。
    """
    for msg in reversed(list(messages)):
        for block in reversed(list(getattr(msg, "content", []) or [])):
            if getattr(block, "type", None) != "tool_result":
                continue
            if getattr(block, "name", None) != "shopping_summary":
                continue
            output = getattr(block, "output", "")
            text = (
                output
                if isinstance(output, str)
                else "".join(getattr(b, "text", "") or "" for b in output)
            )
            try:
                return ShoppingSummaryOutput.model_validate_json(text)
            except Exception:
                continue  # 哨兵文案 / 报错文本：不是清单，继续往前找真正出货的那次
    return None


def _replay_msgs(prior_turns: Sequence[tuple[str, str]]) -> list[Msg]:
    """精简 (role, content) 历史 → ``list[Msg]``（续聊恢复的第二条腿）。

    历史前缀逐字稳定才命中 prompt cache，所以这里只做类型转换，不加任何装饰。
    """
    out: list[Msg] = []
    for role, content in prior_turns:
        # 库里存的 role 只会是 user / assistant（append_turn 那两行写死的）；真混进别的值就
        # 跳过，不猜也不硬塞——一条 role 不合法的历史消息会让整轮请求被网关打回。
        if not content or role not in ("user", "assistant"):
            continue
        speaker: Literal["user", "assistant"] = "user" if role == "user" else "assistant"
        out.append(Msg(name=speaker, role=speaker, content=[TextBlock(type="text", text=content)]))
    return out


def _save_trace(session_dir: Path, messages: Sequence[Msg]) -> None:
    """完整消息轨迹落 history.json（覆盖式，供审计 / 排障）。

    与 ``memory.history.save_full_trace`` 同一个文件、同一个用途，只是序列化换成
    ``Msg.model_dump()``——那边吃的是 LangChain 消息，两个运行时的轨迹格式本就不同。
    """
    try:
        data = [m.model_dump() for m in messages]
        (session_dir / "history.json").write_text(
            json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
    except Exception:
        logger.warning("写完整对话轨迹失败（session_dir=%s），降级跳过", session_dir, exc_info=True)


async def _turn_cache_key(query: str, user_id: str | None, *, first_turn: bool) -> str | None:
    """算这一轮的整轮缓存键；不参与缓存时返回 ``None``（关着 / 不是干净的第一轮 / 算不出来）。

    返回 ``None`` 同时意味着**本轮结束也不写缓存**——查与写用同一个判据，不会出现「查的时候说
    不能复用、跑完又把它存下来」这种自相矛盾。
    """
    if not first_turn or not turn_cache_enabled():
        return None
    try:
        entries = await get_store().read(user_id or "")
        return turn_cache_key(
            buyer=user_id or "", prefs_fp=preference_fingerprint(entries), query=query
        )
    except Exception:
        # 偏好读不到就宁可不缓存：拿一个「假装没有偏好」的指纹去命中，等于把别人的偏好结果给你。
        logger.warning("整轮缓存键计算失败，本轮不走缓存", exc_info=True)
        return None


async def _replay_cached_turn(
    cached: TurnCacheEntry,
    query: str,
    thread_id: str,
    session_dir: Path,
    user_id: str | None,
    started_at: float,
    image_paths: Sequence[str] | None,
) -> dict[str, Any]:
    """整轮缓存命中：把上次那轮的文案与商品卡原样发出去，一次模型调用都不发起。

    仍然**照常落一轮历史**（``append_turn``）——命中与否对用户是透明的，聊天记录不能因为走了
    缓存就缺一轮。不落的是 ``agent_state.json`` / 候选池：那两样是给续聊用的，而带上文的轮次
    本就不进缓存，这一轮之后的追问会退回「有历史但无 state」那条腿（精简 (q,a) 回放）。
    """
    elapsed_ms = int((time.monotonic() - started_at) * 1000)
    logger.info("整轮缓存命中 thread=%s（%d 件商品，未调用模型）", thread_id, len(cached.items))
    await append_turn(
        thread_id,
        query,
        cached.final_text,
        items=cached.items,
        activity=[],
        elapsed_ms=elapsed_ms,
        tokens=None,
        session_dir=session_dir,
        images=list(image_paths or ()),
    )
    await monitor.report_task_result(
        cached.final_text, items=cached.items, elapsed_ms=elapsed_ms, tokens=None
    )
    return {
        "thread_id": thread_id,
        "trace_id": current_trace_id(),
        "final_text": cached.final_text,
        "messages": [],
        "items": cached.items,
        "learned_preferences": get_learned_prefs(),
        "cached": True,
    }


def _called_tool_names(messages: Sequence[Msg]) -> set[str]:
    """本轮出现过的工具名（判「能不能入缓存」用）。

    认的是 ``tool_call`` 块而不是 ``tool_result``：被 harness 闸拦下的调用没有结果块，但它**表达
    了写意图**（模型确实想下单），这种轮次同样不该被复用。
    """
    names: set[str] = set()
    for msg in messages:
        for block in getattr(msg, "content", []) or []:
            if getattr(block, "type", None) == "tool_call":
                name = getattr(block, "name", "")
                if name:
                    names.add(name)
    return names


async def run_agent(
    query: str,
    thread_id: str,
    user_id: str | None = None,
    platforms: Sequence[str] | None = None,
    image_paths: Sequence[str] | None = None,
) -> dict[str, Any]:
    """主 AgentLoop 的入口（AgentScope 运行时）。参数与返回值同 ``main_agent.run_agent``。

    返回 ``{thread_id, trace_id, final_text, messages, items, learned_preferences}``。
    异常都先上报（task_cancelled / error）再向上抛，让 API 层决定怎么响应。
    """
    started_at = time.monotonic()
    session_dir = ensure_session_dir(thread_id)
    with (
        thread_scope(thread_id, session_dir, user_id=user_id),
        platform_scope(platforms) as enabled_platforms,
        # 一轮 = 一条 trace 的根 span。主 loop 与 worker 的 span 靠 OTEL 上下文自动挂进来
        # （不像 LangChain 侧要手工传 trace_id），多轮再靠 session_id=thread_id 聚成 Session。
        # 未启用观测时它是个空壳。
        turn_span(session_id=thread_id, user_id=user_id),
    ):
        activity_rec = monitor.begin_activity_capture()
        await monitor.report_session_created(session_dir)

        # 配额压成本次任务的成本上限（未开鉴权 / 未设配额时为 None，一切照旧）。
        quota_left = await remaining_usd(user_id)
        if quota_left is not None:
            set_task_cap(quota_left)

        # 上一轮候选池读回内存登记表；「这轮要不要重搜」仍由 planner 判，不由「有没有候选」猜。
        prior_cands = load_candidates(session_dir)
        # 清掉上一轮残留的 ContextVar / 模块级状态（retrieval 判定、收货国、品类域、任务清单）：
        # 同 thread 续聊时它们会让本轮 planner 还没跑就先按上轮结论走。
        reset_retrieval_mode()
        reset_dest_country()
        reset_session_domains()
        reset_session_tasks()
        set_original_query(query)

        # on_session_start 是**会话级**的，不属于任何一次 reply，所以由 orchestrator 手动跑
        # （L4 的落点表里唯一没挂进框架钩子的那个）。
        setup_harness()  # 幂等
        await harness.run(
            "on_session_start",
            {"query": query, "thread_id": thread_id, "user_id": user_id},
        )

        begin_learned_prefs()

        # 入口只读近期行为历史；长期偏好等 planner 判出品类域之后由 preference_inject 注入
        # （在这里读等于跨域全量注入，见 session_io.inject_runtime_context 的说明）。
        history_block = await build_history_block(user_id or "")
        pt = load_pt(session_dir)
        set_session_pt(pt)

        # 续聊恢复两条腿：优先 AgentState（模型视野的完整上下文），缺失退回精简 (q,a) 回放。
        prior_state = _load_state(session_dir)
        prior_turns = (
            [] if prior_state is not None else await load_prior_turns(thread_id, session_dir)
        )

        # 整轮结果缓存（默认关，压测 / 演示用）。**只有干净的第一轮才参与**：带上文的轮次，
        # 答案依赖的上文根本不在 key 里，命中就是串味。查得到就直接回放，一轮 LLM 都不跑。
        # 「干净的第一轮」= 两条恢复腿都空。**只看 prior_turns 是不够的**：有 agent_state.json 时
        # 那条腿根本不会去读历史（恒为空列表），于是第二轮会被误判成第一轮、直接命中上一轮的答案。
        cache_key = await _turn_cache_key(
            query, user_id, first_turn=prior_state is None and not prior_turns
        )
        if cache_key is not None:
            cached = get_turn_cache().get(cache_key)
            if cached is not None:
                return await _replay_cached_turn(
                    cached, query, thread_id, session_dir, user_id, started_at, image_paths
                )

        agent, _session = await build_main_agent(
            original_query=query,
            image_paths=tuple(image_paths or ()),
            state=prior_state,
        )
        replay: list[Msg] = _replay_msgs(prior_turns)

        turn_query = inject_runtime_context(
            query,
            history_block,
            pt,
            enabled_platforms,
            prior_candidates=render_prior_candidates(prior_cands),
            image_paths=tuple(image_paths or ()),
        )
        inputs: list[Msg] = [
            *replay,
            Msg(name="user", role="user", content=[TextBlock(type="text", text=turn_query)]),
        ]

        try:
            # fork 预算（拦主 loop 多轮 re-dispatch）+ fork 并发闸（限同时在跑的 worker 数）。
            with fork_budget_scope(), fork_concurrency_scope():
                async with asyncio.timeout(MAIN_AGENT_TIMEOUT_SEC):
                    final_msg = await pump_events(agent.reply_stream(inputs, yield_final_msg=True))
            # 必须在下面 finally 清理之前快照：curator 跑在收尾之后，而 finally 会把品类域清掉。
            session_domains = get_session_domains()
            pt = get_session_pt() or pt
        except asyncio.CancelledError:
            await monitor.report_task_cancelled()
            raise
        except TimeoutError:
            await monitor.report_error(
                "TimeoutError", f"主 loop 超过 {MAIN_AGENT_TIMEOUT_SEC}s 未完成"
            )
            raise
        except Exception as e:
            await monitor.report_error(type(e).__name__, str(e))
            raise
        finally:
            # 成本归集 + 全部按 session_dir / thread_id 聚合的模块级状态清理。放 finally：
            # 取消 / 超时也照样记账 + 清理，绝不漏账或泄漏模块级 dict。逐条的理由见
            # main_agent.run_agent 的同名段落（这里刻意保持逐条一致，别在迁移里悄悄改语义）。
            snap = tree_snapshot()
            if snap is not None:
                status = budget_status()
                metrics.record_cost(float(snap["cost_usd"]), status)
                logger.info(
                    "cost thread=%s usd=%.6f in=%d out=%d calls=%d budget=%s",
                    thread_id,
                    snap["cost_usd"],
                    snap["input_tokens"],
                    snap["output_tokens"],
                    snap["model_calls"],
                    status,
                )
            reset_token_tree()
            reset_retrieval_tree()
            persist_candidates(session_dir)  # 先落盘再清内存，顺序不能反
            reset_candidates()
            reset_diagnostics(thread_id)
            reset_session_bundle()
            reset_retrieval_mode()
            reset_dest_country()
            reset_session_domains()
            reset_session_tasks()
            reset_original_query()
            reset_session_pt()
            if snap is not None:
                await charge_quota(user_id, snap)

        messages: list[Msg] = list(agent.state.context)
        # **final_text 取事件泵拿到的那条 Msg，不从 context 尾部取**：on_session_end 的输出审核
        # 由 HarnessAgentAdapter 在 on_reply 里改写**流出去的**消息（L4），state 里留的是原文。
        # 从 context 取等于把未审核的文本发给用户、落进产物和历史——审核就白做了。
        final_text = (final_msg.get_text_content() or "") if final_msg is not None else ""
        # 本轮结束态落盘，供下一轮 / 换进程恢复（与 append_turn 双做，见 _save_state）。
        _save_state(session_dir, agent.state)

        # 用量以**记账树**为准（snap 在 finally 里取，那时树还没 reset）：一次 reply 只落一条
        # assistant 消息，光数消息会得到 model_calls 恒为 1 的废指标。
        usage = summarize_usage(messages, tree=snap)
        logger.info(
            "usage thread=%s calls=%d carried=%d peak=%d out=%d cache_read=%d hit=%.1f%%",
            thread_id,
            usage.model_calls,
            usage.carried_input_tokens,
            usage.peak_input_tokens,
            usage.output_tokens,
            usage.cache_read_tokens,
            usage.cache_hit_rate * 100,
        )

        # 终结产物三处复用：写回偏好 / 落产物文件 / task_result 带商品卡。
        summary = _extract_summary(messages)
        items = [it.model_dump() for it in summary.items] if summary else []
        # 零候选时用 shopping_summary 自己那份干净文案覆盖：收尾那一轮不受机制约束，曾复现
        # 混入 category_insight 的品类数据给没找到的商品背书。
        if summary is not None and not items:
            final_text = summary.summary

        write_session_artifacts(session_dir, final_text, summary)

        # 写整轮缓存：查得到键（= 关着 / 非第一轮时压根不写）且这轮不含写意图 / 交互工具。
        if cache_key is not None and turn_is_cacheable(_called_tool_names(messages), final_text):
            get_turn_cache().put(cache_key, TurnCacheEntry(final_text=final_text, items=items))

        elapsed_ms = int((time.monotonic() - started_at) * 1000)
        tokens: dict[str, Any] | None = None
        if snap is not None:
            inp = snap["input_tokens"]
            cache_read = snap.get("cache_read_tokens", 0)
            tokens = {
                "input": inp,
                "output": snap["output_tokens"],
                "total": inp + snap["output_tokens"],
                "cost_usd": snap["cost_usd"],
                "cache_read": cache_read,
                "cache_hit_rate": round(cache_read / inp, 4) if inp else 0.0,
            }

        await append_turn(
            thread_id,
            query,
            final_text,
            items=items,
            activity=activity_rec.events,
            elapsed_ms=elapsed_ms,
            tokens=tokens,
            session_dir=session_dir,
            images=list(image_paths or ()),
        )
        _save_trace(session_dir, messages)

        # 只记 query 不记结果：items[0] 是系统排序第一名，用户从未表过态，记成「你上次选的」
        # 会让烂召回反过来污染下一轮上下文。
        if summary is not None:
            await record_search_history(user_id or "", f"搜了「{query[:60]}」")

        await monitor.report_task_result(
            final_text, items=items, elapsed_ms=elapsed_ms, tokens=tokens
        )

        # 记忆判定（后处理）：主回复已下发，用户零感知延迟。curator 只判长期库，P_t 归 planner。
        await curate_turn(
            user_id or "",
            query,
            final_text,
            prev_pt=pt,
            session_domains=session_domains,
        )
        await monitor.report_memory_updated(get_learned_pref_items())

        reset_phase_machine()

        return {
            "thread_id": thread_id,
            "trace_id": current_trace_id(),
            "final_text": final_text,
            "messages": messages,
            "items": items,
            "learned_preferences": get_learned_prefs(),
        }
