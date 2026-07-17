"""主 AgentLoop 组装 —— M9 合龙节点。

把前面 8 个里程碑铺好的零件一次性串成可运行的主链路：

    入口：建会话目录 + 绑定 thread 上下文（M0/M8 监控的 session_created）
      └─ 读 Store 注入长期偏好到 system prompt（M7）；会话级 P_t 拼进当轮 human message
         （不进 system prompt——见 §prompt cache 说明，session_state）
         └─ create_agent(FULL_TOOL_SET, 同质中间件栈)（M4 工具集 + M2 fork + M6 压缩 + ③④ 防御）
            └─ 跑主 loop：Think→Act→Observe→Reflect，到模型自判够了调终结性工具收尾
               └─ 收尾：取 shopping_summary artifact 落产物 + 上报 task_result（M8）
                  └─ 记忆判定（后处理异步，report 之后）：curator 扫本轮 → 更 P_t + 提升长期偏好

设计要点（对 refdocs/14 的主动更正）：

- refdocs 用的是已废弃的 ``create_react_agent`` + ``post_model_hook`` + 全局 ``store`` 单例；
  本仓库按真实依赖改为 LangChain 1.x 的 ``create_agent`` + ``middleware=[...]``（与 M1/M6 一致），
  Store 走 M7 的 ``get_store()`` 选后端，压缩/截断/循环检测走 M6/M2 的中间件。
- **同质 fork 的硬约束**：主 loop 与 fork 出的子 loop 用同一份 ``FULL_TOOL_SET``、同一份
  system prompt、同一套中间件（``build_agent_middleware``）。子 loop 在 ``dispatch_tool`` 内组装。
- **<termination> 先于功能**：主 loop 不写死步数，靠四重护栏防失控——① 终结工具 + 收尾提示词
  （模型自判收敛）② ``recursion_limit`` 硬上限 ③ 整体超时 ④ ``LoopDetector`` 刷屏提示。
  不强插「调完终结工具就跳 END」：留最后一轮让模型把结构化结果转成面向用户的收尾文案，
  自然终止比硬跳更稳，也不丢这段文案。
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import AnyMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

from app.agent.llm import get_fast_llm
from app.agent.platform_scope import platform_scope
from app.agent.prompts import get_system_prompt
from app.agent.retrieval_budget import reset_tree as reset_retrieval_tree
from app.agent.token_budget import budget_status, set_task_cap, tree_snapshot
from app.agent.token_budget import reset_tree as reset_token_tree
from app.agent.tool_registry import FULL_TOOL_SET
from app.agent.tracing import apply_tracing, current_trace_id, record_trace_scores
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
from app.db.quota import add_usage, remaining_usd
from app.harness.agent_middleware import build_agent_middleware
from app.harness.budgets import fork_budget_scope, fork_concurrency_scope
from app.harness.middleware import harness
from app.harness.phase_machine import reset_phase_machine
from app.harness.setup import setup_harness
from app.memory.curator import curate_turn
from app.memory.history import append_turn, load_prior_turns, save_full_trace
from app.memory.injector import (
    HISTORY_EMPTY,
    build_history_block,
    record_search_history,
)
from app.memory.session_state import SessionPrefState, load_pt
from app.observability import metrics
from app.tools._bundle import reset_session_bundle
from app.tools._candidates import (
    load_candidates,
    persist_candidates,
    render_prior_candidates,
    reset_candidates,
)
from app.tools._diagnostics import reset_diagnostics
from app.tools.shopping_summary import ShoppingSummaryOutput
from app.utils.env import env_int
from app.utils.path_utils import ensure_session_dir
from app.utils.thread_ctx import thread_scope

logger = logging.getLogger("shoppingx.main_agent")

# 主 loop 防失控参数（比子 loop 宽：主任务允许更长的链）。可经 env 覆盖以适配不同模型时延。
MAIN_AGENT_MAX_ITERATIONS = env_int("MAIN_AGENT_MAX_ITERATIONS", 30)
# langgraph 按「超步」计数，一轮 Think→Act 约 2 步，故上限 ≈ 2×迭代 + 1（与 dispatch_tool 同口径）。
MAIN_AGENT_RECURSION_LIMIT = MAIN_AGENT_MAX_ITERATIONS * 2 + 1
MAIN_AGENT_TIMEOUT_SEC = env_int("MAIN_AGENT_TIMEOUT_SEC", 300)


def _build_main_agent(
    system_prompt: str, *, original_query: str = "", image_paths: Sequence[str] = ()
) -> Any:
    """组装主 AgentLoop：同一份 FULL_TOOL_SET + 同质中间件栈。

    与 ``dispatch_tool`` 里子 Agent 的组装保持一致（同质 fork）——区别只在 thread/上下文隔离，
    工具集、提示词、中间件三者都相同。``original_query`` 透传给 Harness 中间件层（漂移检测 +
    语义断言需要原始 query 作为对齐基准）。

    **基座模型是快档**（``get_fast_llm``：同一个模型名、只关 thinking）。主 loop 唯一值得开思考的
    是第 1 轮——那是全链路唯一没被机制锁死的决策（购物还是闲聊、先拆解还是先查品类、单平台还是
    fork）；第 2 轮起决策空间已被阶段机白名单 + 候选 id 化夹死，thinking 买不到东西却要花 7~10s。
    故第 1 轮由 ``harness.hooks.reasoning_boost`` 在 pre_think 里 override 成 reasoning 模型，
    这里不必区分——两者模型名相同、工具表不变，切档不打断 prompt cache 前缀。
    """
    return create_agent(
        model=get_fast_llm(),
        tools=FULL_TOOL_SET,
        system_prompt=system_prompt,
        middleware=build_agent_middleware(original_query=original_query, image_paths=image_paths),
    )


def _extract_summary(messages: list[AnyMessage]) -> ShoppingSummaryOutput | None:
    """从消息流里取最后一次 ``shopping_summary`` 的结构化 artifact。

    shopping_summary 走 ``content_and_artifact``：结构化输出挂在 ToolMessage.artifact 上，
    这里**可靠**取回（不解析 ``str(pydantic)`` 这种不稳 repr）。判别直接认 **artifact 的类型**
    （只有 shopping_summary 产 ``ShoppingSummaryOutput``），而非靠 ``m.name``——name 是 Optional、
    序列化往返可能丢，artifact 类型本身才是确定信号。没有终结工具结果（如闲聊走 chat_fallback）
    则返回 ``None``——上游据此跳过写回 / 商品卡 / 产物文件。
    """
    for m in reversed(messages):
        art = getattr(m, "artifact", None) if isinstance(m, ToolMessage) else None
        if isinstance(art, ShoppingSummaryOutput):
            return art
    return None


def _render_platform_block(enabled: tuple[str, ...]) -> str:
    """渲染 ``<enabled_platforms>``——本轮允许检索的平台（用户在前端设置里勾的，默认只 amazon）。

    单平台时**显式告诉主 loop 不要 fork**：跨平台 fork 的唯一理由是「多个平台能并行」，只有一个
    平台时 fork 只剩开销（一个子 Agent 的完整上下文 + 一轮往返）而无并行收益，主 loop 自己一次
    item_search 就够。多平台时才列平台清单让它按 <fork_protocol> 一平台一条派发。

    这只是给模型的**动机**；真正的硬保证在机制层（dispatch_tool 丢弃未启用平台的 demand、
    item_search 的 Qdrant filter 收口到启用集合）——prompt 打动机、机制打保证。
    """
    names = " / ".join(enabled)
    if len(enabled) == 1:
        return (
            f"<enabled_platforms>\n本次只启用 **{names}** 一个平台（用户未开启多平台比价）。\n"
            f"- **不要**跨平台 fork：只有一个平台，parallel_dispatch_tool 没有并行收益。"
            f'直接在主流程 item_search(platform="{names}") 检索、精挑、收尾。\n'
            f"- 比价 / 到手价照常算，但只在该平台内部的候选之间比。\n"
            f"- 收尾时如实说明「本次只搜了 {names}」，不要暗示比过其它平台。\n"
            "</enabled_platforms>"
        )
    return (
        f"<enabled_platforms>\n本次启用 {len(enabled)} 个平台：{names}。\n"
        f"- 跨平台泛搜按 <fork_protocol>：一次 parallel_dispatch_tool，**一平台一条、只列这些平台**"
        f"（共 {len(enabled)} 条），不要派未启用的平台。\n"
        "</enabled_platforms>"
    )


def _inject_runtime_context(
    query: str,
    history_block: str,
    pt: SessionPrefState,
    enabled_platforms: tuple[str, ...] = (),
    prior_candidates: str = "",
    image_paths: Sequence[str] = (),
) -> str:
    """把运行时用户上下文（启用平台 + 近期行为历史 + 会话级 P_t）拼进本轮 query 前，组成当轮
    human message——而不是塞进 system prompt。

    它们都**每轮必变**：历史每轮收尾覆盖、P_t 每轮更新。system prompt 在请求里排在 messages 之前，
    把任何每轮变的东西混进去，都会连累它自己 + 它后面「本该跨轮稳定」的全部历史一起打断 prompt
    cache 前缀。这条 human message 排在**干净的** ``prior_turns``(q,a) 之后、是缓存断点之后永不
    缓存的部分（对齐 refdocs/05 §4.4「按易变性分层，越易变越靠后」）。空的块跳过（不塞「暂无」
    占位，省 token 也不给模型噪声）；全空则原样返回 query。

    **长期偏好不在这里注入**（这是本次重构改掉的）。它曾经拼在这条 human 的最前面，而那时 planner
    还没跑、``session_domains`` 还是空的——``injector._in_scope`` 对空域一律放行，于是模型看到的
    偏好块**必然是跨域全量**的：「买跑鞋时不要皮革」会出现在买旅行包的这一轮，模型很自觉地把
    leather 转述进 ``item_picker(exclude_keywords=...)``，硬淘汰就这么绕过域闸生效了。
    改由 ``harness.hooks.preference_inject`` 在 planner **之后**注入域内偏好——那时域才存在。
    """
    parts: list[str] = []
    # 启用平台随用户设置而变（默认单平台 amazon），同属「每轮可变」——与历史/P_t 一样走
    # human message，不进 system prompt（否则打断跨轮稳定的 cache 前缀）。
    if enabled_platforms:
        parts.append(_render_platform_block(enabled_platforms))
    if history_block and history_block != HISTORY_EMPTY:
        parts.append(f"<user_recent_history>\n{history_block}\n</user_recent_history>")
    if not pt.is_empty():
        parts.append(f"<session_constraints>\n{pt.render()}\n</session_constraints>")
    # 上一轮已检索、已登记的候选：让「只要防水的」这类追问能直接在既有候选上过滤（item_picker），
    # 而不是把 planner → item_search → price_compare 整条链重跑一遍。候选体本身仍在工具内 hydrate，
    # 这里只给模型看 item_id + 决策字段（compact 投影）。
    if prior_candidates:
        parts.append(
            "<prior_candidates>\n"
            "上一轮已检索并登记的候选（本会话内可直接按 item_id 复用，无需重新检索）：\n"
            f"{prior_candidates}\n"
            "</prior_candidates>"
        )
    # 参考图（M20）：只报**文件名**，图本身不进 messages——主模型是纯文本的，多模态消息塞进来只会
    # 报错或被静默忽略。图关在 image_understand 工具里，它的识别结果已由 Harness 在开局预跑写进上文
    # （见 agent_middleware.abefore_agent，先于 planner）。这条块只交代「用户是拿图来买东西的」这个
    # 意图，免得模型把上文那条 image_understand 结果当成无主的噪声。
    if image_paths:
        names = "\n".join(f"- {name}" for name in image_paths)
        parts.append(
            "<reference_images>\n"
            "用户本轮上传了参考图，想买「和图里这个类似的商品」。图已由系统识别，结论见上文的 "
            "image_understand 工具结果——按它的 search_query / keywords 检索即可，不必重复调用。\n"
            "若识别结果为降级（degraded=true），说明图没看成：如实告诉用户并请他用文字描述，别硬编。\n"
            "**多主体消歧**：识别结果的 multi_subject=true 时，图里有多件可买的商品（见 objects），"
            "而 subject / search_query 只描述了其中一件——直接拿去搜就是在替用户瞎猜。此时：\n"
            "- 用户消息里已指明要哪件（如「找图里那个包」「这双鞋多少钱」，或「黑色那个」而 "
            "objects 里只有一件是黑的）→ 按用户指的那件重写检索词直接搜，**不要多问**。\n"
            "- 用户消息没提是哪件（只说了预算、平台等与选件无关的信息，或什么都没说）→ 先调 "
            "ask_user 列出 objects 问清楚要找哪件，拿到回复再搜。别默认挑最大的那件。\n"
            f"{names}\n"
            "</reference_images>"
        )
    if not parts:
        return query
    return "\n\n".join(parts) + f"\n\n用户本轮消息：\n{query}"


def _write_session_artifacts(
    session_dir: Path, final_text: str, summary: ShoppingSummaryOutput | None
) -> None:
    """把本次任务产物落到会话目录，供 ``GET /api/files/<thread_id>/<name>`` 下载（M10）。

    - ``summary.md``：购物清单文案。**优先用 shopping_summary 的结构化 ``summary`` 字段**
      （那才是精挑清单本体），只有没终结产物时（闲聊兜底）才退回 ``final_text``——否则模型
      收尾若多说一句「希望对你有帮助」当作 final_text，下载到的 md 就只剩那句废话。
    - ``result.json``：完整结构化结果（机器读，前端商品卡 / 二次处理用），仅在有终结结果时写。

    产物落 ``output/<thread_id>/``（已 gitignore）。写文件失败不该拖垮主任务——产物是附带
    交付物，记日志降级即可，主链路的偏好写回与 task_result 上报照常进行。故捕获面放宽到
    ``Exception``：除磁盘 OSError，``model_dump``/``json.dumps`` 万一抛也得照样降级，不反噬主链路。
    """
    md = summary.summary if summary is not None else final_text
    try:
        (session_dir / "summary.md").write_text(md or "", encoding="utf-8")
        if summary is not None:
            (session_dir / "result.json").write_text(
                json.dumps(summary.model_dump(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
    except Exception:
        logger.warning("写会话产物失败（session_dir=%s），降级跳过", session_dir, exc_info=True)


# 在途的配额记账 task：只为保一个强引用——detached task 若无人引用会被 GC 掉，账就丢了。
_pending_charges: set[asyncio.Task[None]] = set()


async def _charge_quota(user_id: str | None, snap: dict[str, float | int]) -> None:
    """把本轮全树成本记进用户配额账本，**取消路径下也要记完**。

    为什么绕这一圈而不是直接 ``await add_usage(...)``：本函数跑在 run_agent 的 finally 里，而这条
    路径最常见的触发者恰恰是「用户点了取消」——此时本 task 已被 cancel，直接 await 会在第一个挂起
    点就抛 CancelledError，记账协程连库都摸不到，用户就凭「烧完 token 再取消」白嫖了一整轮。
    ``shield`` 让记账在独立 task 里跑到完，外层的取消信号照常传播（suppress 只吞掉 shield 这个
    await 点二次抛出的 CancelledError，不影响 run_agent 里原本那条 raise）。
    """
    task = asyncio.create_task(
        add_usage(
            user_id,
            float(snap["cost_usd"]),
            int(snap["input_tokens"]),
            int(snap["output_tokens"]),
        )
    )
    _pending_charges.add(task)
    task.add_done_callback(_pending_charges.discard)
    with contextlib.suppress(asyncio.CancelledError):
        await asyncio.shield(task)


async def run_agent(
    query: str,
    thread_id: str,
    user_id: str | None = None,
    platforms: Sequence[str] | None = None,
    image_paths: Sequence[str] | None = None,
) -> dict[str, Any]:
    """主 AgentLoop 的入口：一条 query 从入口跑到收尾，串起记忆 / 压缩 / 监控全链路。

    参数：
      - query：用户本轮的购物意图原文。
      - thread_id：本次任务的会话标识（output/<thread_id>/ 为产物目录，AGUI 事件按它路由）。
        **同一 thread_id 复用即续聊**：开局会回喂该 thread 此前的历史轮次（M-history），
        收尾再把本轮追加进 turns.json，下次接着聊。
      - user_id：登录用户标识；用于读 / 写长期偏好。匿名（None）则不碰 Store。
      - platforms：本轮启用的平台（前端设置面板勾选，随请求带）。缺省 / 空 → 进程级默认（amazon
        单平台）。只勾一个平台时主 loop 不跨平台 fork（见 ``agent.platform_scope``）。
      - image_paths：本轮参考图的文件名（已由 POST /api/upload 落在 uploaded/<thread_id>/）。
        只传文件名，图不进 messages——主模型是纯文本的，看图这件事关在 image_understand 工具里。

    返回：``{thread_id, trace_id, final_text, messages, items, learned_preferences}``。异常都先上报
    （task_cancelled / error）再向上抛，让 API 层（M10）决定怎么响应。
    """
    # 整轮墙钟起点：用 monotonic（不受系统时钟回拨影响）。覆盖「发起→出结果」全程——偏好读取、
    # user 塔、续聊回喂、ainvoke、落盘——即用户感知的「这一轮等了多久」。收尾算差值，存历史 + 随
    # task_result 下发，让实时显示与历史回看用同一权威口径（前端不必各自计时，免得两处对不上）。
    started_at = time.monotonic()
    session_dir = ensure_session_dir(thread_id)
    with (
        thread_scope(thread_id, session_dir, user_id=user_id),
        platform_scope(platforms) as enabled_platforms,
    ):
        # 开一段活动流录制：把本轮思考过程（assistant_call / tool_start|end / fork）攒下来，
        # 收尾随 turns.json 落盘，供前端「回看历史轮」时原样还原「思考过程」折叠区。
        # 在 report_session_created 之前开，连首条 session_created 一并收录。
        activity_rec = monitor.begin_activity_capture()
        await monitor.report_session_created(session_dir)

        # 配额与单任务预算联动（M18）：把「用户今日剩余额度」压成本次任务的成本上限。入口的配额闸
        # 只判「还有没有余额」，判完即放行——若剩余只够 $0.01 而单任务预算是 $0.50，这一趟就能透支
        # 近半个额度。压低后透支最多到「额度刚好用尽」为止，再往下由 hard 闸夺权收尾。
        # 未开鉴权 / 未设配额时 remaining_usd 返回 None，一切照旧（set_task_cap 不调）。
        quota_left = await remaining_usd(user_id)
        if quota_left is not None:
            set_task_cap(quota_left)

        # 上一轮候选池读回内存登记表（首轮读空）：本轮的下游工具因此能按 item_id hydrate 到上一轮
        # 搜过的商品。**只是让候选可用，不代表本轮就是追问轮**——「这轮要不要重搜」由 planner 判
        # （retrieval: reuse/augment/search），不由「有没有候选」这个信号猜。
        prior_cands = load_candidates(session_dir)
        # 清掉上一轮的 retrieval 判定：同 thread 续聊时 ContextVar 可能还留着上轮的 reuse，
        # 会让本轮 planner 还没跑，阶段机就先按复用走。收货国同理（上轮寄日本、这轮没提，
        # 该走 P_t slots / 默认值重新判，而不是让上轮的 JP 赖着）。
        reset_retrieval_mode()
        reset_dest_country()
        # 品类域同理，且更要紧：上轮买鞋残留 [footwear]，这轮改口买沙发而 planner 还没跑完的话，
        # 「买鞋时不要皮革」会被误判为与本轮相关，把皮沙发全杀掉——域隔离要防的事，反倒由陈旧
        # 状态自己制造出来。任务清单同理（上轮比价残留，这轮转移通告会漏发「跳过比价」提示）。
        reset_session_domains()
        reset_session_tasks()
        # 本轮原始用户 query（未经 LLM 转述）——planner 域反证 / picker 品类门锚核验的
        # 独立信号源，每轮覆盖写。
        set_original_query(query)

        # Harness on_session_start：初始化阶段状态机、偏好加载等 session 级 Hook。
        setup_harness()  # 幂等
        await harness.run(
            "on_session_start",
            {
                "query": query,
                "thread_id": thread_id,
                "user_id": user_id,
            },
        )

        # 开本轮「已沉淀偏好」累加器（须在任何 fork 之前）：curator 写库时会记进这里，收尾汇总。
        begin_learned_prefs()

        # 入口只读近期行为历史。**长期偏好不在这里读**——它要等 planner 判出品类域之后才知道哪些
        # 本轮相关，改由 harness.hooks.preference_inject 在 planner 后注入域内那部分（这里读等于
        # 跨域全量注入，正是本次重构修掉的 bug，见 _inject_runtime_context 的 docstring）。
        history_block = await build_history_block(user_id or "")
        # 会话级短期偏好状态 P_t：本次选购会话逐轮累积的约束（预算/材质/颜色…），跨续聊轮次稳定
        # 保留、会话结束（TTL）自然清理。收尾后由记忆管家（curator）在其上 merge 本轮新约束并存回。
        # 匿名用户同样有 P_t——本轮约束是会话态，与是否登录无关。无 session_dir 时
        # （理论不会）退化空状态。
        pt = load_pt(session_dir)
        # P_t 双通道消费：① render 拼进当轮 human（连同长期偏好 + 行为历史，见下方 turn_query），
        # 让主 loop 把累积约束折进 planner intent——**不**塞进 system prompt：P_t 每轮必变，混进
        # system prompt 会连累它后面「本该跨轮稳定」的全部历史一起打断 prompt cache 前缀。② 塞
        # ContextVar 供 item_picker 机制性强制执行（硬 dislike 并入 exclude、软并入 attenuate、预算
        # 兜底）——把「续聊约束」从 prompt 建议升为硬保证，不靠模型每轮转述。fork 子继承快照。
        set_session_pt(pt)
        # system prompt 纯静态（无运行时注入）→ 跨轮 / 跨会话字节稳定、可命中 prompt cache；主与子
        # Agent 的 system 段字节相同，子 Agent 也能命中主 Agent 的缓存。
        agent = _build_main_agent(
            get_system_prompt(), original_query=query, image_paths=tuple(image_paths or ())
        )

        # 续聊：把本 thread 此前的历史轮次（精简 user→assistant 对）回喂进开局 messages，
        # 让同一 thread 的第二次 /api/task 能接住上下文。无历史则为空，等价全新开局。
        prior_turns = await load_prior_turns(thread_id, session_dir)

        config: RunnableConfig = {"recursion_limit": MAIN_AGENT_RECURSION_LIMIT}
        # 挂 Langfuse 观测（仅主对话链路），session=thread_id、user=user_id。root=True：本轮在此
        # 生成 trace_id 写进 ContextVar，fork 子 loop 复用它归并到同一条 trace（一轮=一条 trace）。
        # 未装包 / 未配 key / ENABLED 非真时 apply_tracing 原样返回，主链路无感。
        apply_tracing(config, session_id=thread_id, user_id=user_id, root=True)
        # 当轮 human = 运行时上下文（偏好 + 历史 + P_t）+ 本轮 query，全打包进最后一条消息；
        # prior_turns 是**干净原始** (q,a) 对（append_turn 存原始 query、不含注入），
        # 故历史前缀逐字稳定、可跨轮命中。
        turn_query = _inject_runtime_context(
            query,
            history_block,
            pt,
            enabled_platforms,
            prior_candidates=render_prior_candidates(prior_cands),
            image_paths=tuple(image_paths or ()),
        )
        payload: Any = {"messages": [*prior_turns, ("user", turn_query)]}
        try:
            # 跨整棵 fork 树共享一份 fork 预算：拦住主 loop 多轮 re-fork（软提示拦不住）。检索总量
            # 预算（按 session_dir 聚合，连子里的 item_search 一起兜）则拦住「再找找更好的」从主
            # loop 直调 item_search/web_search 漏出来——两道都在 middleware，这里只开/收作用域。
            # fork_concurrency_scope 与之正交：限同一时刻并发子 Agent 数（资源闸），给下游背压。
            with fork_budget_scope(), fork_concurrency_scope():
                result: dict[str, Any] = await asyncio.wait_for(
                    agent.ainvoke(payload, config=config),
                    timeout=MAIN_AGENT_TIMEOUT_SEC,
                )
            # **必须在 finally 清理之前快照**：curator 跑在下面（收尾后处理），而下面那个 finally
            # 会把 planner 判定的品类域从模块级 dict 里清掉。不快照的话，curator 读到的永远是空
            # ——「用本轮域给漏填 domain 的偏好兜底」就静默失效了。收货国快照已随 curator 的
            # slots 写入权一起删（planner._sync_session_pt 当轮已按「本轮明示」门槛写过 slots）。
            session_domains = get_session_domains()
            # P_t 给 curator 当**只读参考**（帮它把本轮约束挡在长期库外）：取 planner 更新后的
            # 那份（含本轮约束）比开局那份信息更全。curator 不写 P_t，无时序可协调。
            pt = get_session_pt() or pt
        except asyncio.CancelledError:
            # 用户取消（API 层 task.cancel()）：上报后重抛，让事件循环正常结束这条任务。
            await monitor.report_task_cancelled()
            raise
        except TimeoutError:
            await monitor.report_error(
                "TimeoutError", f"主 loop 超过 {MAIN_AGENT_TIMEOUT_SEC}s 未完成"
            )
            raise
        except Exception as e:  # 其余异常：上报后重抛，不在这里吞
            await monitor.report_error(type(e).__name__, str(e))
            raise
        finally:
            # F 块 FinOps：收尾把本任务全树成本归集进 metrics（含撞没撞预算闸），再清条目。
            # 放 finally：取消 / 超时也照样记账 + 清理，绝不漏账或泄漏模块级 dict。
            snap = tree_snapshot()
            if snap is not None:
                status = budget_status()  # 一次读取，metric 标签与日志共用，避免两次读不一致
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
            # 清掉本 session 的全树检索计数条目（模块级 dict 按 session_dir 为键，防无界增长）。
            reset_retrieval_tree()
            # 候选登记表：先落盘（供下一轮追问直接复用 item_id，见 _candidates.persist_candidates）
            # 再清内存（模块级 dict 按 session_dir 为键，不清会无界增长）。顺序不能反。
            persist_candidates(session_dir)
            reset_candidates()
            # 诊断侧信道稳态下消费即空，这里只兜「middleware 异常路径漏消费」的残留
            # （按 thread_id 为键，防无界增长）。
            reset_diagnostics(thread_id)
            # 套装槽位同理：只清内存（bundle.json 留着供续聊轮 get_session_bundle 懒读回；
            # 换品类时由 planner 判 search 连文件一起清，见 planner）。
            reset_session_bundle()
            # planner 的 retrieval 判定 / 收货国 / 品类域 / 任务清单同理（模块级 dict，
            # 按 session_dir 为键）。
            reset_retrieval_mode()
            reset_dest_country()
            reset_session_domains()
            reset_session_tasks()
            reset_original_query()
            # P_t 同样是按 session_dir 聚合的模块级 dict（planner 要跨 context 写它，裸 ContextVar
            # 传不出工具边界）。**上面已把它快照进 pt**，curator 用的是那份快照，这里清干净不影响。
            reset_session_pt()

            # 用户级配额记账（M18）：把本轮**全树**成本累加进 usage_ledger（跨会话、跨重启的余额
            # 真源，与上面进程内的 metrics 记账互补）。放 finally 的**最末**：取消 / 超时同样要
            # 记——token 已经烧掉了，不记等于让用户白嫖一次超时任务；而放最后可保证它即便抛异常
            # 也不会打断上面那一串状态清理。await 见 _charge_quota，取消路径下账照记完。
            if snap is not None:
                await _charge_quota(user_id, snap)

        messages: list[AnyMessage] = result["messages"]
        # 用 .text 而非 str(content)：模型最终回复的 content 可能是多模态 block 列表，
        # str() 会把它变成 Python repr（"[{'type':'text',...}]"）灌给前端；.text 取纯文本。
        final_text = messages[-1].text if messages else ""

        # 测量驱动 L3 决策：聚合本轮 token 用量（携带量峰值 + 缓存命中率），进日志 + Langfuse。
        # 1M 窗口下不盯「溢出」，盯的是「上下文变长→变笨」与成本；这些数是日后是否上 L3 的判据闸门。
        usage = summarize_usage(messages)
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
        record_trace_scores(
            {
                "carried_input_tokens": usage.carried_input_tokens,
                "peak_input_tokens": usage.peak_input_tokens,
                "cache_hit_rate": usage.cache_hit_rate,
                "output_tokens": usage.output_tokens,
            }
        )

        # 取一次终结产物，三处复用：写回偏好 / 落产物文件（下载）/ task_result 带商品卡。
        summary = _extract_summary(messages)
        items = [it.model_dump() for it in summary.items] if summary else []

        # shopping_summary 判定「零候选」（如实道歉 + 建议）后，主 loop 通常还会被追问一轮
        # 自然收尾文字——那一轮不受机制约束，曾复现过混入 category_insight 的品类聚合数据
        # 给没找到的具体商品背书（如实标注：这是 prompt 已禁止但没有机制兜的口子）。零候选时
        # 直接用 shopping_summary 自己产出的干净文案覆盖，不依赖模型这轮听不听话，用户在
        # 对话 / 历史里看到的与落盘产物保持一致。
        if summary is not None and not items:
            final_text = summary.summary

        # Harness on_session_end：输出审核（把内部控制文案剔出面向用户的回复）等 session 级 Hook。
        # **必须排在所有 final_text 消费者之前**——落产物 / 写历史 / report_task_result / curator
        # 都吃它。审核放在它们后面等于没审：用户在前端看到的、下载到的 md、落盘的历史全是原文，
        # 只有 run_agent 的返回值是干净的。（这条通路必须消费返回 ctx，见 refdocs 17-2 §4.1。）
        end_ctx = await harness.run(
            "on_session_end",
            {
                "final_answer": final_text,
                "thread_id": thread_id,
                "trajectory": messages,
            },
        )
        final_text = end_ctx.get("final_answer", final_text)

        # 落产物到会话目录，供下载接口取（M10 闭环的「可下载文件」一环）。
        _write_session_artifacts(session_dir, final_text, summary)

        # 本轮总耗时（毫秒）：随历史落盘 + 随 task_result 下发，前端在该轮右下角显示「用时」。
        elapsed_ms = int((time.monotonic() - started_at) * 1000)

        # 本轮全树 token 用量：snap 在上面 finally 里取自 tree_snapshot()（主 + 各 fork 子 Agent
        # 累计，比只看主 loop messages 的 summarize_usage 更全），同随历史落盘 + 随 task_result
        # 下发，前端在该轮右下角与「用时」并排显示「token 消耗」。未记账（snap=None）即不带。
        tokens: dict[str, Any] | None = None
        if snap is not None:
            inp = snap["input_tokens"]
            cache_read = snap.get("cache_read_tokens", 0)
            tokens = {
                "input": inp,
                "output": snap["output_tokens"],
                "total": inp + snap["output_tokens"],
                "cost_usd": snap["cost_usd"],
                # cache 命中口径一并透传：cost 早已 cache-aware（token_budget 三档计价），但只报
                # input/output/cost 会把「缓存到底有没有真生效」这个健康指标藏起来。cache_read =
                # 命中折扣档的 input，hit_rate = cache_read/input（input 为 0 时置 0，绝不除零）。
                "cache_read": cache_read,
                "cache_hit_rate": round(cache_read / inp, 4) if inp else 0.0,
            }

        # 落对话历史：追加本轮到 messages 表（下次续聊 / 前端回看用）+ 覆盖写完整轨迹供审计。
        # items（商品卡）+ activity（思考过程）+ elapsed_ms（耗时）一并写进 assistant 轮，让回看
        # 能还原卡片、思考行与用时。放在 load_prior_turns 之后，故本轮只计一次；写失败已降级。
        await append_turn(
            thread_id,
            query,
            final_text,
            items=items,
            activity=activity_rec.events,
            elapsed_ms=elapsed_ms,
            tokens=tokens,
            session_dir=session_dir,
            # 参考图文件名随 user 轮落库，回看时前端据此把图取回来画进气泡。
            images=list(image_paths or ()),
        )
        save_full_trace(session_dir, messages)

        # 记一条 search 行为历史：机制性写，不靠模型调工具。仅购物终结（summary 非空）时记，
        # 非购物（chat_fallback）跳过。下次会话注入，让 Agent「记得你最近搜过什么」——用途只有
        # 跨会话指代消解（「上次那个」）和知道用户近期关注的品类。
        #
        # **只记 query，不记结果**。曾经这里还写「精选 N 件，首选：<items[0].title>」，是错的：
        # items[0] 是**系统排序的第一名**，用户从未表过态（可能压根没看上、甚至反感），把它记成
        # 「你上次选的」会让烂召回结果反过来污染下一轮上下文。用户真正认可的信号是收藏（♡），
        # 那是另一条数据、且刻意不进 prompt。件数同理无决策价值，一并去掉。
        if summary is not None:
            await record_search_history(user_id or "", f"搜了「{query[:60]}」")

        await monitor.report_task_result(
            final_text, items=items, elapsed_ms=elapsed_ms, tokens=tokens
        )

        # 记忆判定（后处理异步）：主回复已 report_task_result 下发，用户零感知延迟。记忆管家 curator
        # 只判**长期库**——扫本轮对话，把「一贯取向」经单一落库口提升为长期偏好；P_t 不归它管
        # （planner 单写者，当轮已落盘）。仍在 thread_scope 与被 track 的 _runner task 内
        # （取消 / 清理语义不变），不另起 detached task。curator 内部全兜底（失败只降级、绝不抛），
        # 不反噬主链路。其经 persist 写库时会 record_learned_pref，故下面 get_learned_prefs() 能取。
        await curate_turn(
            user_id or "",
            query,
            final_text,
            prev_pt=pt,
            # 上面 finally 已经把品类域从模块级 dict 里清了，只能用清理前的快照——直接在
            # curator 里调 get_session_domains() 会永远读到空。
            session_domains=session_domains,
        )
        # 沉淀了新长期偏好就告诉用户——前端在回复下方画一行「记住了 … ✕」，可一键撤销。
        # 自动写入 + 看得见 + 撤得掉，替代「弹窗问用户要不要记」：收尾那一刻用户的注意力在
        # 「买不买这件」上，此时弹确认框只会被无脑点掉，筛不出任何东西。
        await monitor.report_memory_updated(get_learned_pref_items())

        reset_phase_machine()

        return {
            "thread_id": thread_id,
            # 本轮 Langfuse trace_id（未启用观测则 None）。显式返回、不让下游读 ContextVar：评测拿它
            # 把 Rubric 分数作为 score 挂回这条 trace（见 tracing.record_rubric_scores 的注释）。
            "trace_id": current_trace_id(),
            "final_text": final_text,
            "messages": messages,
            "items": items,
            # 本轮提升为长期的偏好（curator 判定 + 落库，去重后）。会话级约束进 pt.json、不在此列。
            "learned_preferences": get_learned_prefs(),
        }
