"""主链路执行入口：一条 query 从入口跑到收尾（``run_agent`` 的唯一实现）。

编排顺序是建会话目录 → 读历史 / P_t → 跑 loop → 收尾落产物 → 记忆判定。三处值得单独记住的
接线（都是 AgentScope 运行时的特性，迁移时逐条验过）：

1. **loop 怎么跑**：``Agent.reply_stream`` + 事件泵（见 ``app/agent/events.py``），
   而不是「跑完拿返回值」——AGUI 事件要在过程中实时推给前端。
2. **on_session_end 不在这里调**：由 ``HarnessAgentAdapter.on_reply`` 在框架内部改写最终
   ``Msg``，所以这里拿到的 ``final_text`` **已经是审核后的**。别再补一次——重复审核会把哨兵
   文案二次剥离，且 ``output_audit`` 的计数会翻倍。
3. **会话恢复只有一条腿**：``session.json``（``AgentState.model_dump_json()``，P_t 住
   ``middle_context``）。读不到 / 读坏 → 空开局，不回放 messages 表——那张表只给前端回看，
   Agent 不读。见 :func:`load_session_state` / :func:`save_session_state`。

与运行时无关的那几件事（当轮上下文拼装、产物落盘、配额记账）住在 ``app/agent/session_io.py``。
"""

import asyncio
import json
import logging
import os
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from agentscope.message import Msg, TextBlock
from agentscope.state import AgentState

from app.agent.ab import assign as assign_prompt_version
from app.agent.agents import build_main_agent
from app.agent.events import pump_events
from app.agent.limits import MAIN_AGENT_TIMEOUT_SEC
from app.agent.platform_scope import platform_scope
from app.agent.session_io import (
    charge_quota,
    inject_runtime_context,
    write_session_artifacts,
)
from app.agent.skills import render_selected_skill, resolve_selected_skill
from app.agent.tracing import current_trace_id, turn_span
from app.agent.usage import summarize_usage
from app.api import monitor
from app.api.context import (
    begin_learned_prefs,
    get_learned_pref_items,
    get_learned_prefs,
    get_session_pt,
    reset_dest_country,
    reset_original_query,
    reset_session_pt,
    reset_session_tasks,
    set_original_query,
    set_session_pt,
)
from app.db.quota import remaining_usd
from app.harness.msgs import iter_tool_results
from app.harness.phase_machine import fresh_phase_machine, reset_phase_machine
from app.harness.retrieval_budget import reset_tree as reset_retrieval_tree
from app.harness.setup import setup_harness
from app.harness.token_budget import budget_status, set_task_cap, tree_snapshot
from app.harness.token_budget import reset_tree as reset_token_tree
from app.memory.curator import curate_turn
from app.memory.fact_store import get_fact_store
from app.memory.facts import select_tier_one_facts
from app.memory.history import append_turn
from app.memory.injector import build_history_block, record_search_history
from app.memory.session_state import pt_from_state, pt_into_state
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
from app.tools._candidates import reset_candidates
from app.tools._diagnostics import reset_diagnostics
from app.tools.shopping_summary import ShoppingSummaryOutput
from app.utils.path_utils import ensure_session_dir
from app.utils.thread_ctx import thread_scope

logger = logging.getLogger("shoppingx.orchestrator")

# 会话唯一的跨轮产物：AgentState 全量（含 messages 上下文、框架摘要、middle_context 里的 P_t）。
STATE_FILE = "session.json"


def _state_path(session_dir: Path) -> Path:
    return session_dir / STATE_FILE


def load_session_state(session_dir: Path) -> AgentState | None:
    """读回上一轮落盘的 ``AgentState``（续聊恢复的唯一一条腿）。

    读不到 / 读坏了都返回 ``None`` → 调用方按空开局。恢复是**加成**，不是前提：一份坏掉的
    state 让整轮聊天起不来，比少一段上下文糟得多。
    """
    path = _state_path(session_dir)
    if not path.exists():
        return None
    try:
        return AgentState.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("session.json 解析失败，按空开局（%s）", path, exc_info=True)
        return None


def save_session_state(session_dir: Path, state: AgentState) -> None:
    """把本轮结束时的 ``AgentState`` 原子落盘（临时文件 + rename），供下一轮 / 换进程恢复。

    这是会话状态的**唯一写点**，只在成功收尾时调；取消 / 超时不写，上一轮那份原样留着。
    原子写是为了同一个保证：任何时刻磁盘上要么是上一轮的完整 state，要么是这一轮的，
    绝不会是写到一半的残片。写失败只记日志——下轮空开局，比拖垮本轮的收尾好。
    """
    path = _state_path(session_dir)
    tmp = path.with_suffix(".json.tmp")
    try:
        tmp.write_text(state.model_dump_json(), encoding="utf-8")
        os.replace(tmp, path)
    except Exception:
        logger.warning("写 session.json 失败，下轮按空开局", exc_info=True)
        tmp.unlink(missing_ok=True)


def _extract_summary(messages: Sequence[Msg]) -> ShoppingSummaryOutput | None:
    """从消息流里取最后一次 ``shopping_summary`` 的结构化产物。

    AgentScope 的工具结果里没有 artifact 这类侧信道，结构化结果就是 ``ToolResultBlock`` 里那段
    JSON 文本（见 ``app/tools/_as_tools.py`` 的 ``_to_text``）。所以这里按**工具名 + 能否验成
    ShoppingSummaryOutput** 双条件认，而不是认类型。

    「倒着找 + 解析失败接着往前」的遍历由 ``harness.msgs.iter_tool_results`` 负责，那个坑的
    完整说明也在那里——它此前在这里和 ``adapter._terminal_summary`` 各写一遍，连坑注释都各抄
    一份。同一件事的两份实现，一处修 bug 另一处必漏。
    """
    for text in iter_tool_results(messages, "shopping_summary"):
        try:
            return ShoppingSummaryOutput.model_validate_json(text)
        except Exception:
            continue  # 哨兵文案 / 报错文本：不是清单，继续往前找真正出货的那次
    return None


async def _turn_cache_key(
    query: str, user_id: str | None, *, first_turn: bool, prompt_version: str = ""
) -> str | None:
    """算这一轮的整轮缓存键；不参与缓存时返回 ``None``（关着 / 不是干净的第一轮 / 算不出来）。

    返回 ``None`` 同时意味着**本轮结束也不写缓存**——查与写用同一个判据，不会出现「查的时候说
    不能复用、跑完又把它存下来」这种自相矛盾。
    """
    if not first_turn or not turn_cache_enabled():
        return None
    try:
        facts = select_tier_one_facts(await get_fact_store().get_facts(user_id or ""))
        return turn_cache_key(
            buyer=user_id or "",
            prefs_fp=preference_fingerprint(facts),
            query=query,
            prompt_version=prompt_version,
        )
    except Exception:
        # 记忆读不到就宁可不缓存：拿一个「假装没有记忆」的指纹去命中，等于把别人的记忆结果给你。
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
    缓存就缺一轮。不落的是 ``session.json``：这一轮之后的追问会按空开局（缓存只在干净的
    第一轮参与，命中那轮本就没有可恢复的 state）。
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


def _skills_read(messages: Sequence[Msg]) -> list[str]:
    """本轮主 Agent 读过的 skill 名（``Skill(skill=…)`` 的入参），按首次出现顺序去重。

    ``Skill`` 是框架内置工具、不过 harness 的工具中间件，所以没有 tool_start / tool_end 事件；
    要让产品面看见「这轮读了哪个 skill」，只能从消息里的 tool_call 块回收。入参可能是 dict
    也可能是 JSON 串（取决于框架版本怎么存），两种都认。
    """
    out: list[str] = []
    for msg in messages:
        for block in getattr(msg, "content", []) or []:
            if getattr(block, "type", None) != "tool_call" or getattr(block, "name", "") != "Skill":
                continue
            raw = getattr(block, "input", None)
            if isinstance(raw, str):
                try:
                    raw = json.loads(raw)
                except ValueError:
                    raw = None
            name = str((raw or {}).get("skill") or "") if isinstance(raw, dict) else ""
            if name and name not in out:
                out.append(name)
    return out


def _experiment_summary(ab_assign: Any, messages: Sequence[Msg]) -> dict[str, Any]:
    """本轮「实验与自进化」归属：提示词版本 / A/B 桶 / 注入的策略 / 读过的 skill。

    MCP 工具**不在这里**：它们不过 harness（见 mcp_registry 的诚实标注）——要看去 Langfuse 的
    acting span。
    """
    from app.harness.hooks.context_shaping import injected_strategy_keys

    return {
        "prompt_version": ab_assign.version,
        "ab_bucket": ab_assign.bucket,
        "in_experiment": bool(ab_assign.in_experiment),
        "strategies": list(injected_strategy_keys()),
        "skills": _skills_read(messages),
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
    skill: str | None = None,
) -> dict[str, Any]:
    """主 AgentLoop 的入口：一轮任务从这里进、从这里出。

    ``skill``：用户在输入框 ``/`` 显式选中的 skill 目录名。服务端在首次模型调用前校验归属并把
    正文拼进本轮用户消息（``authority=reference_only``）；找不到就报错结束本轮，**不静默降级
    成普通搜索**——用户点了方案却被无视，比明说「方案已失效」更糟。

    返回 ``{thread_id, trace_id, final_text, messages, items, learned_preferences}``。
    异常都先上报（task_cancelled / error）再向上抛，让 API 层决定怎么响应。
    """
    started_at = time.monotonic()
    session_dir = ensure_session_dir(thread_id)
    # 提示词 A/B：桶号与版本本轮**只算一次**，trace / 账本 / 返回值共用同一份结论。分开各算各的
    # 会在「刚好跨过热更新」的那一轮记出互相矛盾的归属——A/B 报告最怕的就是这种错行。
    ab_assign = assign_prompt_version(user_id)
    with (
        thread_scope(thread_id, session_dir, user_id=user_id),
        platform_scope(platforms) as enabled_platforms,
        # 一轮 = 一条 trace 的根 span。主 loop 与 worker 的 span 靠 OTEL 上下文自动挂进来
        # （不必手工传 trace_id），多轮再靠 session_id=thread_id 聚成 Session。
        # 未启用观测时它是个空壳。
        turn_span(
            session_id=thread_id,
            user_id=user_id,
            prompt_version=ab_assign.version,
            ab_bucket=ab_assign.bucket,
        ),
    ):
        activity_rec = monitor.begin_activity_capture()
        await monitor.report_session_created(session_dir)

        # 配额压成本次任务的成本上限（未开鉴权 / 未设配额时为 None，一切照旧）。
        quota_left = await remaining_usd(user_id)
        if quota_left is not None:
            set_task_cap(quota_left)

        # 清掉上一轮残留的 ContextVar / 模块级状态（收货国、任务清单）：
        # 同 thread 续聊时它们会让本轮 planner 还没跑就先按上轮结论走。
        reset_dest_country()
        reset_session_tasks()
        set_original_query(query)
        fresh_phase_machine()  # 会话级复位，与上面几个 reset 同列（曾是 on_session_start hook）
        setup_harness()  # 幂等

        begin_learned_prefs()

        selected_skill: tuple[str, str] | None = None
        if skill:
            selected_skill = await resolve_selected_skill(skill)
            if selected_skill is None:
                await monitor.report_error("SkillNotFound", f"所选方案 {skill!r} 不存在或已删除")
                raise LookupError(f"所选方案 {skill!r} 不存在或已删除，请刷新后重选")

        # 入口只读近期行为历史；长期偏好等 planner 判出品类域之后由 preference_inject 注入
        # （在这里读等于跨域全量注入，见 session_io.inject_runtime_context 的说明）。
        history_block = await build_history_block(user_id or "")

        # 续聊恢复唯一一条腿：session.json → AgentState（模型视野的完整上下文 + middle_context
        # 里的 P_t）。缺失 / 读坏 → 空开局。候选池**不跨轮**：追问轮照常重搜，跨轮引用
        # （「买第 2 个」）按 item_id 回源 Qdrant（见 _candidates.hydrate）。
        prior_state = load_session_state(session_dir)
        pt = pt_from_state(prior_state.middle_context) if prior_state else pt_from_state({})
        set_session_pt(pt)

        # 整轮结果缓存（默认关，压测 / 演示用）。**只有干净的第一轮才参与**：带上文的轮次，
        # 答案依赖的上文根本不在 key 里，命中就是串味。查得到就直接回放，一轮 LLM 都不跑。
        # 「干净的第一轮」= 没有可恢复的 session.json。
        cache_key = await _turn_cache_key(
            query,
            user_id,
            # 显式选了 skill 的轮次不参与整轮缓存：答案依赖方案正文，而正文不在 key 里。
            first_turn=prior_state is None and selected_skill is None,
            prompt_version=ab_assign.version,
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
        turn_query = inject_runtime_context(
            query,
            history_block,
            enabled_platforms,
            image_paths=tuple(image_paths or ()),
        )
        if selected_skill is not None:
            turn_query = f"{turn_query}\n\n{render_selected_skill(*selected_skill)}"
        inputs: list[Msg] = [
            Msg(name="user", role="user", content=[TextBlock(type="text", text=turn_query)]),
        ]
        # 本轮消息的起点：恢复回来的 state 里还躺着前几轮的上下文，收尾产物（清单 / 商品卡）只能
        # 从这个下标往后找，否则第二轮用 chat_fallback 收尾时会把上一轮的清单当成本轮产物。
        turn_start = len(agent.state.context)

        try:
            async with asyncio.timeout(MAIN_AGENT_TIMEOUT_SEC):
                final_msg = await pump_events(agent.reply_stream(inputs, yield_final_msg=True))
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
            # 取消 / 超时也照样记账 + 清理，绝不漏账或泄漏模块级 dict。
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
            reset_candidates()  # 候选登记表只活一轮；跨轮引用按 item_id 回源 Qdrant
            reset_diagnostics(thread_id)
            reset_session_bundle()
            reset_dest_country()
            reset_session_tasks()
            reset_original_query()
            reset_session_pt()
            if snap is not None:
                await charge_quota(user_id, snap, prompt_version=ab_assign.version)

        messages: list[Msg] = list(agent.state.context)
        # **final_text 取事件泵拿到的那条 Msg，不从 context 尾部取**：on_session_end 的输出审核
        # 由 HarnessAgentAdapter 在 on_reply 里改写**流出去的**消息（L4），state 里留的是原文。
        # 从 context 取等于把未审核的文本发给用户、落进产物和历史——审核就白做了。
        final_text = (final_msg.get_text_content() or "") if final_msg is not None else ""
        # 成功收尾的唯一写点：P_t 填进 middle_context，随 AgentState 一起原子落盘。取消 / 超时
        # 走不到这里，session.json 保持上一轮那份。append_turn 只喂前端回看，Agent 不读它。
        pt_into_state(agent.state.middle_context, pt)
        save_session_state(session_dir, agent.state)

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
        summary = _extract_summary(messages[turn_start:])
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

        experiment = _experiment_summary(ab_assign, messages)
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
            experiment=experiment,
        )
        # 只记 query 不记结果：items[0] 是系统排序第一名，用户从未表过态，记成「你上次选的」
        # 会让烂召回反过来污染下一轮上下文。
        if summary is not None:
            await record_search_history(user_id or "", f"搜了「{query[:60]}」")

        await monitor.report_task_result(
            final_text, items=items, elapsed_ms=elapsed_ms, tokens=tokens, experiment=experiment
        )

        # 记忆抽取（后处理）：主回复已下发，用户零感知延迟。只读本轮对话文本、只写长期事实库。
        await curate_turn(user_id or "", query, final_text)
        await monitor.report_memory_updated(get_learned_pref_items())

        reset_phase_machine()

        return {
            "thread_id": thread_id,
            "trace_id": current_trace_id(),
            "final_text": final_text,
            "messages": messages,
            "items": items,
            "learned_preferences": get_learned_prefs(),
            # 以下四项给离线 A/B 报告按桶聚合用（`scripts/eval/ab_report.py`）：光有版本号还
            # 判不了优劣，「轮数 / token」是版本变化最先反映出来的两处代价。
            "prompt_version": ab_assign.version,
            "ab_bucket": ab_assign.bucket,
            "tokens": tokens,
            "model_calls": usage.model_calls,
        }
