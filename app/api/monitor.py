"""AGUI 事件上报 —— 统一封装「Agent 在做什么」并经 WebSocket 推给前端。

设计立场（M4 留桩、M8 接真，工具侧一行不改）：上报接口是**模块级函数**
（``report_tool_start`` / ``report_tool_end`` / ……），工具里只写一句
``await monitor.report_tool_start(...)``，既不必拿 ``thread_id``、也不必知道连接在哪——
这两件事分别由 :mod:`app.api.context` 的 ContextVar 与 :class:`ConnectionManager` 透明处理。

一条事件的统一结构（前端只看 ``event`` 分发、看 ``data`` 取业务字段）::

    {
      "type": "monitor_event",
      "event": "tool_start",
      "message": "正在调用 item_search",
      "data": {"tool": "item_search", "query": "旅行收纳袋"},
      "thread_id": "abc123",
      "timestamp": "2026-06-25T14:23:45.123456+00:00"
    }

为什么全做成 async：推 WebSocket 是 IO，要 await；定成 async 后工具调用点到了真推送也
不用改。无 ``thread_id``（离线脚本/测试）或该 thread 没有活跃连接时，上报自动降级为
**只打日志、不推送**，绝不抛异常——监控失败不该拖垮主任务。
"""

from __future__ import annotations

import logging
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.api import event_log
from app.api.connection import ConnectionManager
from app.api.context import get_session_dir, get_thread_id
from app.observability import metrics

logger = logging.getLogger("shoppingx.monitor")

# AGUI 七类标准事件 + fork（同质子 Agent 派发）。前端按这些类型分发展示。
EVENT_SESSION_CREATED = "session_created"
EVENT_ASSISTANT_CALL = "assistant_call"
EVENT_TOOL_START = "tool_start"
EVENT_TOOL_END = "tool_end"
EVENT_FORK = "fork"
EVENT_CLARIFICATION_REQUEST = "clarification_request"
EVENT_QUEUE_STATUS = "queue_status"
EVENT_MEMORY_UPDATED = "memory_updated"
EVENT_MEMORY_APPLIED = "memory_applied"
# 会话级 P_t 约束集快照（每轮 planner 落 P_t 后推送）：偏好面板的「本次会话」区据此实时刷新。
# **瞬态**（不进活动流、不进回放存档）：断线重连后面板走 GET /api/session/{thread}/constraints
# 主动拉，一条随时可重建的状态快照不值得进存档。
EVENT_SESSION_CONSTRAINTS = "session_constraints"
EVENT_TASK_RESULT = "task_result"
EVENT_TASK_CANCELLED = "task_cancelled"
EVENT_ERROR = "error"
# 收尾文案的流式增量（杠杆3·感知延迟）：shopping_summary 内部 LLM 边生成边推，用户提前 ~10s
# 看到清单文案逐字出现。**瞬态事件**：不进活动流（回看由 task_result 的定稿承担）、不进事件
# 回放存档（断线重连补发几十条渐进全文毫无意义，重连后 task_result 自带完整文案）。
EVENT_SUMMARY_DELTA = "summary_delta"
# 商品卡先出货（感知延迟·同 summary_delta 一条思路：那个管文案、这个管卡片）：item_picker 一定稿，
# 清单里到底是哪几件**就已经确定了**——可用户还得再等主 loop 收尾那轮解码 + summary 生成才看得见。
# 这条事件在定稿的那一刻先把商品卡推出去，文案随后由 summary_delta 逐字补上。
# **不瞬态**（进回放存档）：跑到一半刷新页面 / 断线重连时，卡片要能跟着 inflight 事件回放一起回来，
# 否则用户会看着卡片凭空消失、再等收尾才重现。不进活动流（它不是「思考行」，是结果本身）。
EVENT_ITEMS_PREVIEW = "items_preview"

# 事件里携带的自由文本（demands / preview / 最终答案）截断上限，避免单条事件灌爆前端。
_MAX_TEXT = 2000

# 「思考过程」事件白名单：只把这几类录进活动流供历史回看（与 ActivityFeed 画的「思考行 / 搜索卡」
# 一一对应）。终结类 task_result / task_cancelled / error 是驱动前端状态的信号、不画思考行，且商品卡
# 走 items 独立持久化，故不录——免得回看时多出一行「工具调用·运行中」的脏行。
_ACTIVITY_EVENTS = frozenset(
    {
        EVENT_SESSION_CREATED,
        EVENT_ASSISTANT_CALL,
        EVENT_TOOL_START,
        EVENT_TOOL_END,
        EVENT_FORK,
        EVENT_CLARIFICATION_REQUEST,
        EVENT_MEMORY_APPLIED,
    }
)


@dataclass
class ActivityRecorder:
    """一段任务里「Agent 干了什么」的活动流录制器（供收尾持久化、前端回看时还原思考过程）。

    只收与录制起点**同一 thread_id** 的事件：fork 子 Agent 在子 ``thread_scope`` 下产生的内部
    事件天然被过滤掉，与实时前端「父任务页只显示父 thread 事件」的口径一致（见 ``report_fork``）。
    """

    root_thread_id: str | None
    events: list[dict[str, Any]] = field(default_factory=list)


# 当前任务的活动流录制器（None=不录制，如离线脚本 / 测试）。run_agent 在主 loop 开局
# 用 begin_activity_capture() 设上，收尾读 recorder.events 落进 turns.json。task-local：
# fork 子任务 copy_context 拿到同一引用，但子事件因 thread_id 不匹配被滤掉，无并发写竞争。
_activity_recorder: ContextVar[ActivityRecorder | None] = ContextVar(
    "shoppingx_activity_recorder", default=None
)


def begin_activity_capture() -> ActivityRecorder:
    """在当前 thread 上下文开一段活动流录制，返回 recorder（其 ``events`` 随上报实时累积）。

    绑定调用时刻的 ``thread_id`` 为「根」，只录该 thread 直接产生的事件。一请求一 async task、
    run_agent 不在同 task 内重入，故无需显式 reset——录制器随 task 上下文销毁。
    """
    rec = ActivityRecorder(root_thread_id=get_thread_id())
    _activity_recorder.set(rec)
    return rec


# 全局连接管理器单例：monitor 经它推送，API 层的 WS 端点用同一个实例登记连接。
# 用 set/get 包一层，测试可替换为带假连接的实例。
_manager = ConnectionManager()


def get_connection_manager() -> ConnectionManager:
    """返回当前生效的连接管理器（API 层 WS 端点与 monitor 共用同一个）。"""
    return _manager


def set_connection_manager(manager: ConnectionManager) -> None:
    """替换连接管理器（主要给测试用；生产用默认单例即可）。"""
    global _manager
    _manager = manager


def _clip(text: str | None) -> str | None:
    """长文本截断，事件只带摘要，不灌全量。"""
    if text is None:
        return None
    return text if len(text) <= _MAX_TEXT else text[:_MAX_TEXT] + "…[truncated]"


async def _emit(
    event: str,
    message: str,
    data: dict[str, Any],
    thread_id: str | None = None,
    transient: bool = False,
) -> None:
    """组装统一结构并推给当前 thread 的连接；无上下文/无连接则只记日志。

    ``thread_id`` 显式传入时覆盖 ContextVar——``queue_status`` 在任务真正开跑（进 ``thread_scope``）
    **之前**就要推给前端，那时 ContextVar 还没绑定，只能由调用方把 thread_id 递进来。

    ``transient=True``：只直播、不落 Redis 事件回放存档（summary_delta 这类渐进增量，断线补发
    几十条渐进全文毫无意义——重连后 task_result 自带定稿）。活动流白名单本就不含它们。
    """
    if thread_id is None:
        thread_id = get_thread_id()
    payload: dict[str, Any] = {
        "type": "monitor_event",
        "event": event,
        "message": message,
        "data": data,
        "thread_id": thread_id,
        "timestamp": datetime.now(UTC).isoformat(),
    }

    # 持久化进该 thread 的 Redis Stream（D 块），拿到 stream id 回填进 payload —— 前端把它当
    # last_event_id 记住，断线重连时带回来补发缺口。
    #
    # **只持久化「根 thread」的事件**：子 fork 内部事件的流没人会拿 last_event_id 去回放（前端只连
    # 根 thread），写了纯属浪费 + 留下没人读的 Stream。判据复用活动流录制的根 thread 口径——
    # report_fork 的 thread_id 是父=根，照样持久化；子 Agent 内部事件的 thread_id 是子，跳过。
    # 无录制上下文（直接调 run_agent 的测试 / 离线）时回退为持久化，有 TTL 兜底。无 thread_id 或
    # Redis 降级时 payload 无 id、退回现状（只直播）。放在直播之前，确保推出去的事件就带 id。
    rec = _activity_recorder.get()
    is_root_event = rec is None or rec.root_thread_id is None or thread_id == rec.root_thread_id
    if thread_id is not None and is_root_event and not transient:
        event_id = await event_log.append(thread_id, payload)
        if event_id is not None:
            payload["id"] = event_id

    logger.debug("AGUI %s", payload)

    # 录进活动流（若本任务在录制）：只收根 thread 的思考过程事件，供历史回看还原。
    # 放在推送之前、且与连接是否存在无关——离线/无连接也照样攒，回看不依赖当时是否有人在看。
    if rec is not None and event in _ACTIVITY_EVENTS and thread_id == rec.root_thread_id:
        rec.events.append(payload)

    if thread_id is None:
        return
    try:
        await _manager.send_to_thread(thread_id, payload)
    except Exception:  # 兜底：上报链路任何异常都不许冒泡进 AgentLoop
        logger.exception("monitor emit failed: event=%s thread_id=%s", event, thread_id)


# --- 七类标准事件 + fork 的上报入口 -------------------------------------------


async def report_session_created(session_dir: Path | None = None) -> None:
    """后台任务建好、会话目录就绪时上报（前端显示「会话已创建」）。"""
    sd = session_dir if session_dir is not None else get_session_dir()
    await _emit(EVENT_SESSION_CREATED, "会话已创建", {"session_dir": str(sd) if sd else None})


async def report_assistant_call(step: str = "thinking", preview: str | None = None) -> None:
    """主 AgentLoop 进入 Think 阶段时上报（前端显示「Agent 思考中…」）。"""
    await _emit(EVENT_ASSISTANT_CALL, "Agent 思考中", {"step": step, "preview": _clip(preview)})


async def report_tool_start(tool: str, **fields: Any) -> None:
    """上报「某工具开始执行」。``fields`` 放该工具关键入参（如 query / platform）。"""
    await _emit(EVENT_TOOL_START, f"正在调用 {tool}", {"tool": tool, **fields})


async def report_tool_end(tool: str, **fields: Any) -> None:
    """上报「某工具执行完毕」。``fields`` 放结果摘要（召回条数/是否截断），不灌全量结果。"""
    await _emit(EVENT_TOOL_END, f"{tool} 完成", {"tool": tool, **fields})


async def report_summary_delta(text: str) -> None:
    """收尾文案的流式增量（**累计全文**，非 delta 片段）。

    发累计全文而非增量片段：WS 消息乱序 / 丢一条时前端不会拼出错位文本，收到哪条就渲染哪条，
    天然幂等。文案本身只有几百字，累计重发的带宽成本可忽略。瞬态：不进回放存档与活动流。
    """
    await _emit(EVENT_SUMMARY_DELTA, "清单生成中", {"text": _clip(text)}, transient=True)


async def report_items_preview(items: list[dict[str, Any]]) -> None:
    """商品卡定稿即先出货（``item_picker`` 挑完就推，不等收尾文案）。

    ``items`` 的字段与收尾 ``task_result`` 里的商品卡**同构**（item_id / platform / title /
    price_usd / landed_usd / reason / image_url / url），前端可以用同一个 ProductCards 渲染，
    收尾时再被定稿那批原样覆盖。

    **显式路由到根 thread**：精挑万一发生在 fork 出的子 loop 里（子 thread 没有前端连接，事件会
    静默丢掉），卡片就永远推不出去。这与 ``report_fork`` 的取舍一致——用户看的是根 thread 那个页面，
    结果类事件就该送到那里。无录制上下文（离线 / 单测）时退回当前 ContextVar。
    """
    rec = _activity_recorder.get()
    root = rec.root_thread_id if rec is not None and rec.root_thread_id else None
    await _emit(
        EVENT_ITEMS_PREVIEW,
        f"精选 {len(items)} 件商品",
        {"items": items},
        thread_id=root,
    )


async def report_fork(sub_thread_id: str, demands: str) -> None:
    """主 loop 派发同质子 AgentLoop 时上报（前端显示「派发子任务并行处理」）。

    在进入子 ``thread_scope`` 之前调用，故事件路由到**父 thread** 的连接——用户在父任务
    页面就能看到「分叉出一个子任务」。子 Agent 内部的事件因其 thread 无前端连接而静默
    （上下文隔离），这是有意为之。
    """
    metrics.inc_fork()  # A 块：fork 派发计数
    await _emit(
        EVENT_FORK,
        "派发子任务并行处理",
        {"sub_thread_id": sub_thread_id, "demands": _clip(demands)},
    )


async def report_queue_status(
    thread_id: str, position: int, estimated_wait_seconds: int, kind: str
) -> None:
    """任务因槽位占满进入等待队列时上报（前端显示「排队中，前面还有 N 个任务」）。

    **必须显式传 thread_id**：这一刻任务还没进 ``thread_scope``（``run_agent`` 都还没开始跑），
    ContextVar 是空的。排队反馈的全部价值就在于「用户不必对着空白屏幕猜」——所以它必须早于
    ``session_created`` 推出去。
    """
    await _emit(
        EVENT_QUEUE_STATUS,
        f"排队中，前面还有 {position - 1} 个任务",
        {"position": position, "estimated_wait_seconds": estimated_wait_seconds, "kind": kind},
        thread_id=thread_id,
    )


async def report_clarification_request(
    question: str,
    options: list[str] | None = None,
    multi_select: bool = False,
    preselected: list[str] | None = None,
) -> None:
    """Agent 需要用户澄清时上报，等待用户回复。

    无 options → 前端渲染问题气泡并激活输入框（老式自由文本回复）。
    有 options → 前端在展示区内嵌一张**可点选卡片**（单选按钮 / 多选清单），用户点鼠标作答、
    不复用聊天框；回传的仍是一段文本（卡片据勾选拼成自然语言），后端契约不变。
    """
    data: dict[str, object] = {"question": question}
    if options:
        data["options"] = options
        data["multi_select"] = multi_select
        if preselected:
            data["preselected"] = preselected
    await _emit(EVENT_CLARIFICATION_REQUEST, "等待用户澄清", data)


async def report_memory_updated(prefs: list[dict[str, str]]) -> None:
    """curator 本轮沉淀了新长期偏好时上报（前端在回复下方画一行「记住了 … ✕」）。

    透明度设计的落点：写入是自动的（不打断用户去点确认框），但**必须看得见、且一键撤得掉**——
    每条带 ``dedup_key``，✕ 直接打 ``DELETE /api/preferences/{uid}/{key}``。当撤销成本只有一次
    点击时，事前确认就不值得存在（同 ChatGPT 的 "Memory updated"）。

    在 ``report_task_result`` **之后**发（curator 是后处理，跑在主回复下发之后）——前端据此把这行
    追加到已渲染的回复下面。空列表不发，免得每轮都闪一个空条。
    """
    if not prefs:
        return
    await _emit(EVENT_MEMORY_UPDATED, f"记住了 {len(prefs)} 条新偏好", {"preferences": prefs})


async def report_memory_applied(
    domains: list[str], excluded: list[str], attenuated: list[str]
) -> None:
    """本轮**用到了**哪些长期记忆——把「记忆如何影响了这次结果」摆到台面上。

    与 :func:`report_memory_updated`（写入侧：「记住了 X」）互补，这条是**读取侧**：本轮判定的
    品类域、哪些偏好词把商品淘汰了、哪些只是压低了排序。

    **为什么这条事件不能省。** 这套记忆系统真正的病不是复杂，是**复杂且不可观测**：改造前
    ``domain`` 字段被写入端精心维护、读取端一个都没消费，而这个 bug 静默存在了很久——因为一条
    偏好没生效 / 误杀了一批商品，前端不会有任何提示，用户只会觉得「这破 Agent 老是搜不出东西」，
    且**归因不到记忆头上**。把生效情况变成一个事件，以后任何一个维度接漏了，你和用户都能立刻
    看见。空信息不发（没记忆生效就别闪一行噪声）。
    """
    if not (excluded or attenuated):
        return
    parts = []
    if excluded:
        parts.append(f"排除 {len(excluded)} 项")
    if attenuated:
        parts.append(f"降权 {len(attenuated)} 项")
    await _emit(
        EVENT_MEMORY_APPLIED,
        "按你的长期偏好：" + "、".join(parts),
        {"domains": domains, "excluded": excluded, "attenuated": attenuated},
    )


async def report_session_constraints(pt: Any, thread_id: str | None = None) -> None:
    """P_t 约束集变化（新增 / 撤回 / 换代 / 面板删除）后推当前快照——偏好面板「本次会话」区实时刷新。

    可见可纠的第二腿（步骤三①的事件侧）：约束「录入」仍过 LLM 的手（极性判反 / keywords 抽漏
    照样进 P_t，且无自愈性），抽错时唯一的兜底是**用户看得见、点得掉**——看得见的前提是推送。
    每条带 ``id``（面板删除按 id 打 DELETE）与 ``source_quote``（让用户看懂是自己哪句话）。

    瞬态：面板打开 / 断线重连走 GET 主动拉，快照不进回放存档。空约束集也推——撤回 / 换代后
    面板要能清空，不推就永远停在删除前的样子。``thread_id`` 显式传入供 API 层（面板删除）使用，
    那里不在 thread_scope 里。
    """
    await _emit(
        EVENT_SESSION_CONSTRAINTS,
        f"本会话累积约束 {len(pt.constraints)} 条",
        {
            "epoch": pt.epoch,
            "budget_usd": pt.budget_usd,
            "category": pt.category,
            "constraints": [
                {
                    "id": c.id,
                    "content": c.content,
                    "source_quote": c.source_quote,
                    "polarity": c.polarity,
                    "blocking": c.blocking,
                }
                for c in pt.constraints
            ],
        },
        thread_id=thread_id,
        transient=True,
    )


async def report_task_result(
    final_answer: str,
    items: list[dict[str, Any]] | None = None,
    elapsed_ms: int | None = None,
    tokens: dict[str, Any] | None = None,
) -> None:
    """任务完成、给出最终回答时上报（前端渲染最终清单 + 商品卡）。

    ``items`` 是 ``shopping_summary`` 结构化产出里的精选商品（平台/标题/到手价/选购理由），
    前端据此渲染商品卡——文本清单给人读、结构化 items 给机器画卡，一条事件两用。M8 老调用
    点（只传 final_answer）不受影响：缺省 ``None`` 即不带 items 字段。

    ``elapsed_ms`` 是本轮总耗时（毫秒），前端在该轮右下角显示「用时」。缺省 ``None`` 即不带。

    ``tokens`` 是本轮**全树**（主 + 各 fork 子 Agent）token 用量（``input`` / ``output`` /
    ``total`` / ``cost_usd`` / ``cache_read`` / ``cache_hit_rate``），前端在该轮右下角与「用时」
    并排显示「token 消耗」，hover 可见输入/输出/成本/缓存命中率拆分。缺省即不带。
    """
    data: dict[str, Any] = {"final_answer": _clip(final_answer)}
    if items:
        data["items"] = items
    if elapsed_ms is not None:
        data["elapsed_ms"] = elapsed_ms
    if tokens is not None:
        data["tokens"] = tokens
    await _emit(EVENT_TASK_RESULT, "任务完成", data)


async def report_task_cancelled() -> None:
    """任务被用户取消时上报（AgentLoop 捕获 CancelledError 后发）。"""
    await _emit(EVENT_TASK_CANCELLED, "任务已取消", {})


async def report_error(error_type: str, message: str) -> None:
    """执行异常时上报（前端显示错误，便于定位卡在哪一步）。"""
    await _emit(EVENT_ERROR, "执行出错", {"error_type": error_type, "message": _clip(message)})
