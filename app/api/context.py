"""请求级上下文：用 ContextVar 保存当前任务的 thread_id 与 session_dir。

ShoppingX 是 asyncio 单线程协程并发服务，同一事件循环里多个用户任务交替推进，
主 AgentLoop 同一轮还会并发跑多个工具调用（``is_concurrency_safe=True``）。若用普通
全局变量保存 thread_id / session_dir 会立刻串台。ContextVar 为每个 asyncio Task 维护
独立副本，天然隔离；且 ``asyncio.create_task`` 会复制当前 Task 的 ContextVar 快照，
框架并发执行的工具协程自动继承。

（2026-09-16：曾用于「主 loop fork 同质子 AgentLoop」的隔离，派发已随单环收敛删除，
ContextVar 现在只服务多用户隔离与产物归档。）

写入封装见 :mod:`app.utils.thread_ctx` 的 ``thread_scope`` 上下文管理器。
"""

import os
import time
import uuid
from collections.abc import Sequence
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from app.api.run_state import peek_run_slot, run_slot
from app.utils.env import env_bool

if TYPE_CHECKING:
    from app.memory.turn_constraints import TurnConstraints

# 当前请求的 thread_id（由 /api/task 入口或 thread_scope 设置）。
_thread_id_var: ContextVar[str | None] = ContextVar("shoppingx_thread_id", default=None)

# 当前请求的会话目录（本次任务的产物落到这里）。
_session_dir_var: ContextVar[Path | None] = ContextVar("shoppingx_session_dir", default=None)

# 当前请求的登录用户（用于工具层读长期偏好 / 黑名单）。匿名任务为 None。
_user_id_var: ContextVar[str | None] = ContextVar("shoppingx_user_id", default=None)

# 当前这一轮的 run_id（队列模式下 = task_id）。除了 credit 预扣结算的幂等键，它还是**写工具的
# 幂等锚**：消息被 PEL 重投、整轮重跑时 run_id 不变，同一轮重复调 create_order 只落一张确认卡
# （见 app.trade.confirmations 的 request_key）。空串 = 没有 run 作用域（HTTP 表单入口 / 离线
# 脚本 / 单测），此时写工具退回「每次新建一张卡」的老行为。
_run_id_var: ContextVar[str] = ContextVar("shoppingx_run_id", default="")

# 跨进程的日志关联 id：HTTP 入口生成一次，随 IntentTask 进队列，worker 消费时绑回
# 日志上下文。**不是 run_id 的重复**：run_id 只在幂等判定通过、真开出一个 run 之后才有意义，而
# 被判成 already_running / duplicate 的请求同样要能在日志里被找到——排查「用户点了三次，为什么
# 只跑了一次」时，串起那三条 HTTP 请求的就是它。也**不是 Langfuse 的 trace_id**（那个由 worker
# 侧根 span 生成、只覆盖 run_agent 内部，且已经占用了 trace_id 这个名字）。
_request_id_var: ContextVar[str] = ContextVar("shoppingx_request_id", default="")


def new_request_id() -> str:
    """生成一个新的日志关联 id（16 位十六进制，够短好在日志里扫、碰撞概率可忽略）。"""
    return uuid.uuid4().hex[:16]


def get_request_id() -> str:
    """当前请求的日志关联 id；不在请求作用域内返回空串。"""
    return _request_id_var.get()


# 首事件延迟的计时盒（SLO 第二条）：``{"started_at": <入队时刻的 wall clock>}``，
# 第一条 ``assistant_call`` 上报时把它取走并记进 Histogram，之后的事件读到空盒直接跳过。
#
# **为什么是可变 dict 而不是两个裸 ContextVar**：``report_assistant_call`` 由主 loop 的钩子触发，
# 可能跑在 ``create_task`` 派生出的子 context 里，在那里 ``set`` 的值回不到父 context——用「父置
# 一个盒子、子往里掏」的形态，写才跨得回来（同 ``_learned_prefs_var``）。
#
# 刻度是 wall clock 不是 monotonic：起点来自**另一个进程**（API 入队），monotonic 跨进程没有可比性。
# 同机部署下时钟一致；真跨机时 NTP 偏差会算进延迟里，这是已知误差，不值得为它上一套时钟同步。
_first_event_var: ContextVar[dict[str, float] | None] = ContextVar(
    "shoppingx_first_event", default=None
)


def begin_first_event_timer(started_at: float | None = None) -> None:
    """开一个首事件计时盒（``run_agent`` 入口调）。``started_at`` 缺省取当下。"""
    _first_event_var.set({"started_at": started_at if started_at is not None else time.time()})


def take_first_event_latency() -> float | None:
    """取走首事件延迟（秒）。**只有第一次调返回数**，之后返回 ``None``——「首」事件只有一个。

    没开计时盒（离线脚本 / 单测直调 monitor）时也返回 ``None``。
    """
    box = _first_event_var.get()
    if box is None:
        return None
    started = box.pop("started_at", None)
    return None if started is None else time.time() - started


@dataclass
class _RunScope:
    """本模块在**一次 run** 里持有的全部状态（住 :mod:`app.api.run_state` 的总表）。

    这四格都按 session_dir 聚合而非裸 ContextVar：planner 与 item_picker / shipping_calc 是几个
    不同的工具、各自在独立 context 里跑，前者 ``set`` 的 ContextVar 后者读不到（子 task 建立时
    拷贝一份 context，回写不冒泡）。P_t 原本是裸 ContextVar 且侥幸没暴露这个坑——它此前只由
    run_agent 入口（主 context）写一次；planner 一开始写 P_t，同一个坑就要踩第三次。
    """

    # 本轮生效约束（沿用 P_t 的名字与形状）——**只由 planner 写**，每轮从前几轮原话 + 本轮原话
    # 整体重算，不跨轮累积、不落盘；供 item_picker 等工具机制性读取并强制执行
    # （把「不要塑料」「预算 ≤X」从 prompt 建议升为硬保证，不靠模型每轮转述）。
    constraints: "TurnConstraints | None" = None

    # 前几轮的用户原话（旧 → 新，不含本轮）——run_agent 入口从 session.json 读回后写入，
    # planner 据它重算仍生效的约束。存原话而不从 messages 里抠：messages 里是拼了运行时
    # 上下文的版本，且会被框架压缩掉。
    prior_queries: list[str] = field(default_factory=list)

    # planner 本轮判定的任务清单（recommend / price_compare / landed_cost / ...）——「用户要不要
    # 比价」同样是意图判断，只有 planner 有依据。收线通告读它来定向（无比价诉求时提示模型跳过
    # price_compare / shipping_calc，见 harness.hooks.progress）。
    tasks: list[str] = field(default_factory=list)

    # planner 本轮判定的收货国：(ISO 码, 是否为系统假设值)。收货国决定关税免征额（US $0 vs
    # CN $7 vs AU $660，差两个数量级）。assumed=True 表示用户从没说过、是 env 默认兜的，此时
    # 回复必须标注假设，且**不该**把它当用户事实沉进会话 slots / 长期记忆。
    dest_country: tuple[str, bool] | None = None

    # 本轮**原始用户 query**（未经任何 LLM 转述）——工具侧唯一的「用户到底说了什么」确定性
    # 信号源。planner 的 category 是 LLM 结构化输出，「合法但错」时下游拿它当锚会
    # 静默反转（品类门反着杀）；反证只能靠独立信号，而独立信号只有原文词面。
    original_query: str = ""

# 本轮「已沉淀事实」累加器——curator 写成功即把 (content, key) 记这里，
# 供 run_agent 收尾时汇总进 learned_preferences 返回、并经 AGUI 推给前端「记住了 … ✕」那一行。
# 存 key 是因为那一行的 ✕ 要能删掉这条——只有 content 的话前端拿不到删除 handle。
# 默认 None（非 run_agent 上下文，如离线脚本 / 单测直调 persist）→ 记录端 no-op，避免
# 「模块级可变默认列表跨任务串台」的经典坑。
# fork 子 Agent 通过 Task 的 ContextVar 快照继承同一个 list 引用，子里记的偏好自然冒泡回主轮。
_learned_prefs_var: ContextVar[list[dict[str, str]] | None] = ContextVar(
    "shoppingx_learned_prefs", default=None
)

def get_thread_id() -> str | None:
    """读取当前任务的 thread_id；无上下文（如离线脚本）时返回 None。"""
    return _thread_id_var.get()


def get_user_id() -> str | None:
    """读取当前任务的登录用户 id；匿名 / 无上下文时返回 None。"""
    return _user_id_var.get()


def get_run_id() -> str:
    """读取本轮 run_id；无 run 作用域（HTTP 表单入口 / 离线脚本 / 单测）时返回空串。

    空串是安全侧：写工具据它决定**要不要**按幂等键复用确认卡，判不出来就按老行为每次新建一张
    ——复用错了是「用户要的第二张单被吞掉」，不复用最多是重投时多一张待决议的卡。
    """
    return _run_id_var.get()


# 本轮的截止时刻（``time.monotonic()`` 刻度），None = 没有 deadline 作用域。
#
# 各出站超时各自为政会出现这种事：主 loop 还剩 3 秒，某次检索照样按自己的 5 秒等下去——那 5 秒
# 注定等不到结果被用上，纯属让用户多晾 3 秒再看到一句「超时」。deadline 由 ``run_agent`` 入口写一次
# （= now + MAIN_AGENT_TIMEOUT_SEC），各出站点取 ``min(自身超时, 剩余)``。
#
# 形态照抄 ``_run_id_var``：入口写一次、下游只读，ContextVar 的快照继承正好够用（工具在各自的
# context 里跑，读得到父的值，回写不冒泡——这里不需要回写）。
#
# 刻度用 monotonic 不用 wall clock：这是个纯相对量，而后者会被改系统时间 / NTP 回拨拽着跳。
_deadline_var: ContextVar[float | None] = ContextVar("shoppingx_deadline", default=None)

# 剩余时间见底后仍要给出站一个正数超时：0 或负数传进 httpx / Qdrant 有的当「无限等」、有的直接
# 抛参数错误，两种都比「立刻失败」难查。给 50ms 让它按正常路径超时返回。
_DEADLINE_FLOOR = 0.05


def deadline_enabled() -> bool:
    """这道闸开着没有（``DEADLINE_ENABLED``，默认开）。

    关掉 = 各出站点照自己的超时来。默认开是因为它只会把超时**往小了收**，而收掉的那一段本来也会
    被主 loop 的 ``asyncio.timeout`` 整体掐掉——区别只在「早几秒知道」还是「白等几秒」。
    """
    return env_bool("DEADLINE_ENABLED", True)


def set_deadline(seconds: float) -> None:
    """把本轮的截止时刻设为 ``now + seconds``（``run_agent`` 入口调，一轮一次）。"""
    _deadline_var.set(time.monotonic() + seconds)


def reset_deadline() -> None:
    """清掉本轮 deadline（离线脚本 / 单测复位用）。

    线上不需要显式清：``run_agent`` 每轮进来重设一次，而 ContextVar 随 task 结束自然回收。
    """
    _deadline_var.set(None)


def remaining_seconds() -> float | None:
    """距本轮截止还剩几秒；没有 deadline 作用域（工具单测 / 离线脚本）时返回 ``None``。"""
    deadline = _deadline_var.get()
    return None if deadline is None else deadline - time.monotonic()


def clamp_timeout(base: float) -> float:
    """把一个出站超时收到本轮剩余时间以内——**出站超时的唯一入口**，调用方别自己比大小。

    没有 deadline、闸关着、或 ``base`` 本就比剩余小，都原样返回 ``base``：这道闸只收紧，不放宽。
    一个出站点该等多久是它自己的事（对面正常响应要多久），deadline 只管「再等也没意义了」。
    """
    if not deadline_enabled():
        return base
    left = remaining_seconds()
    if left is None or base <= left:
        return base
    return max(left, _DEADLINE_FLOOR)


def set_prior_queries(queries: Sequence[str]) -> None:
    """记下前几轮的用户原话（``run_agent`` 入口写，旧 → 新）。"""
    st = run_slot(_RunScope)
    if st is not None:
        st.prior_queries = list(queries)


def get_prior_queries() -> list[str]:
    """读前几轮的用户原话；无会话作用域（单测直调）或首轮返回空列表。"""
    st = peek_run_slot(_RunScope)
    return list(st.prior_queries) if st is not None else []


def set_turn_constraints(pt: "TurnConstraints | None") -> None:
    """写入本轮生效约束 P_t。唯一写者是 ``planner``（每轮整体重算后覆盖）。按 session_dir
    聚合，故**跨工具可见**。无 session_dir（单测直调工具）时静默丢弃。"""
    st = run_slot(_RunScope)
    if st is not None:
        st.constraints = pt


def get_turn_constraints() -> "TurnConstraints | None":
    """读取本会话的 P_t；未设置（无会话上下文 / 首轮空态）时返回 None。"""
    st = peek_run_slot(_RunScope)
    return st.constraints if st is not None else None


def reset_turn_constraints() -> None:
    """清掉本会话的 P_t（run_agent 收尾，与 reset_session_tasks 对称——run 状态表不像
    ContextVar 会随 task 结束自动回收，不清就会按 session_dir 一直攒着）。"""
    st = peek_run_slot(_RunScope)
    if st is not None:
        st.constraints = None


def get_session_dir() -> Path | None:
    """读取当前任务的会话目录；无上下文时返回 None。"""
    return _session_dir_var.get()


def set_original_query(query: str) -> None:
    """记下本轮原始用户 query（``run_agent`` 入口写，每轮覆盖）。

    给 item_picker 的品类门锚核验当独立信号：planner 的 category 是 LLM 结构化输出，
    能反证它的只有用户原文的词面。
    """
    st = run_slot(_RunScope)
    if st is not None:
        st.original_query = query


def get_original_query() -> str:
    """读本轮原始用户 query；无会话作用域（单测）返回空串 = 无反证证据，一切照旧。"""
    st = peek_run_slot(_RunScope)
    return st.original_query if st is not None else ""


def reset_original_query() -> None:
    """收尾清理（run 状态按 session_dir 为键，不清会无界增长）。"""
    st = peek_run_slot(_RunScope)
    if st is not None:
        st.original_query = ""


def set_session_tasks(tasks: Sequence[str]) -> None:
    """记下 planner 本轮判定的任务清单。由 planner 工具写，收线通告读。"""
    st = run_slot(_RunScope)
    if st is not None:
        st.tasks = list(tasks)


def get_session_tasks() -> list[str]:
    """读 planner 本轮的任务判定；planner 还没跑（或无会话作用域）时返回空列表。

    空列表是安全侧：读方（转移通告）只在**确定无比价诉求**时才提示跳过 price_compare，
    判不出来就不提示——多调一次工具只是慢，错误提示跳过会漏掉用户真要的比价。
    """
    st = peek_run_slot(_RunScope)
    return list(st.tasks) if st is not None else []


def reset_session_tasks() -> None:
    """清掉本会话的任务判定（``run_agent`` 开局 + 收尾调）。

    开局清防上一轮残留、收尾清防 run 状态表无界增长。
    """
    st = peek_run_slot(_RunScope)
    if st is not None:
        st.tasks = []


def set_dest_country(country: str, assumed: bool = False) -> None:
    """记下 planner 本轮确定的收货国（ISO 码）+ 它是不是系统假设的。由 planner 工具写。

    收货国是**确定性判断**（用户原话规则解析 > 会话 slots > 长期记忆 > env 默认），不让模型
    每轮自由填——同 currency 的老教训（「预算 500」曾被轮流猜成 ₹/¥/$）。判完写这里，让
    shipping_calc 读得到，而不是指望模型每次都记得把参数传对。
    """
    st = run_slot(_RunScope)
    if st is not None:
        st.dest_country = (country.strip().upper(), assumed)


def get_dest_country() -> str:
    """读本轮收货国；planner 还没跑（或无会话作用域，如单测 / examples）时回落 env 默认值。

    这是 shipping_calc 的**机制兜底**：即便模型漏传 / 传错 dest_country，工具拿到的仍是系统
    认定的那个国家。默认值与 ``app.recall.geo.DEFAULT_DEST_COUNTRY`` 同源——这里单独读一次 env
    而不 import geo，是为了不把整个 recall 包（qdrant / towers 等重模块）拖进 api 底层。
    """
    st = peek_run_slot(_RunScope)
    if st is not None and st.dest_country:
        return st.dest_country[0]
    return (os.getenv("DEFAULT_DEST_COUNTRY", "CN") or "CN").strip().upper()


def is_dest_country_assumed() -> bool:
    """本轮收货国是不是系统假设的（用户从没说过）。planner 没跑过时按「是」算。

    curate_turn 用它决定要不要把收货国沉进会话 slots：假设值不是用户事实，沉下去会让
    「系统默认」在下一轮伪装成「用户说过」，越滚越真。
    """
    st = peek_run_slot(_RunScope)
    if st is not None and st.dest_country:
        return st.dest_country[1]
    return True


def reset_dest_country() -> None:
    """清掉本会话的收货国（``run_agent`` 开局 + 收尾调）。

    开局清：同 thread 续聊换了收货国时，别让上一轮的国家赖着不走。
    收尾清：run 状态按 session_dir 为键，不清会无界增长。
    """
    st = peek_run_slot(_RunScope)
    if st is not None:
        st.dest_country = None


def begin_learned_prefs() -> None:
    """在 ``run_agent`` 入口开一份空的「本轮已沉淀偏好」累加器（须在派生任何子任务之前调）。

    置一个**新** list 而非复用默认——这样框架并发跑工具协程时快照到的是本轮这份、且各轮互不串。
    """
    _learned_prefs_var.set([])


def record_learned_pref(content: str, key: str = "") -> None:
    """把一条刚落库成功的事实记进本轮累加器（按 content 保序去重）。

    ``key`` 是事实的 key，也是前端「记住了 … ✕」那一行的删除 handle
    （DELETE /api/preferences/{uid}/{key}）。非 ``run_agent`` 上下文（累加器为 None，
    如离线脚本 / 单测直调）下 no-op。
    """
    lst = _learned_prefs_var.get()
    if lst is None or not content:
        return
    if any(p["content"] == content for p in lst):
        return
    lst.append({"content": content, "key": key})


def get_learned_prefs() -> list[str]:
    """读取本轮已沉淀偏好的 content 列表（run_agent 收尾汇总进返回）。未开累加器时为空。"""
    return [p["content"] for p in (_learned_prefs_var.get() or [])]


def get_learned_pref_items() -> list[dict[str, str]]:
    """同上，但带 ``dedup_key``——供 AGUI ``memory_updated`` 事件让前端能一键撤销。"""
    return [dict(p) for p in (_learned_prefs_var.get() or [])]
