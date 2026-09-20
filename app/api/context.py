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
from collections.abc import Sequence
from contextvars import ContextVar
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.memory.session_state import SessionPrefState

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

# 当前会话的短期偏好状态 P_t（本会话逐轮累积的约束）——run_agent 入口从 session.json 读回后
# 写入、**planner 在识别出本轮约束后当轮改写**，供 item_picker 等工具机制性读取并强制执行
# （把「不要塑料」「预算 ≤X」从 prompt 建议升为硬保证，不靠模型每轮转述）。
#
# 同 _DEST_COUNTRY 用「按 session_dir 聚合的模块级 dict」而非裸 ContextVar，理由见下面那段
# 注释——planner 与 item_picker 各自在独立 context 里跑，前者 set 的 ContextVar 后者读不到。
# P_t 原本是裸 ContextVar 且侥幸没暴露这个坑：它此前只由 run_agent 入口（主 context）写一次，
# 从没有工具写过它。planner 一开始写 P_t，同一个坑就踩第三次了。
_SESSION_PT: dict[str, "SessionPrefState"] = {}

# 本轮「已沉淀事实」累加器——curator 写成功即把 (content, key) 记这里，
# 供 run_agent 收尾时汇总进 learned_preferences 返回、并经 AGUI 推给前端「记住了 … ✕」那一行。
# 存 key 是因为那一行的 ✕ 要能删掉这条——只有 content 的话前端拿不到删除 handle。
# 默认 None（非 run_agent 上下文，如离线脚本 / 单测直调 persist）→ 记录端 no-op，避免
# 「模块级可变默认列表跨任务串台」的经典坑。
# fork 子 Agent 通过 Task 的 ContextVar 快照继承同一个 list 引用，子里记的偏好自然冒泡回主轮。
_learned_prefs_var: ContextVar[list[dict[str, str]] | None] = ContextVar(
    "shoppingx_learned_prefs", default=None
)

# planner 本轮判定的任务清单（recommend / price_compare / landed_cost / ...）——「用户要不要比价」
# 同样是意图判断，只有 planner 有依据。阶段机的转移通告读它来定向（无比价诉求时提示模型跳过
# price_compare / shipping_calc，见 harness.hooks.progress）。按 session_dir 聚合（理由见
# _DEST_COUNTRY）。
_SESSION_TASKS: dict[str, list[str]] = {}

# planner 本轮判定的收货国（ISO 码）——决定关税免征额（US $0 vs CN $7 vs AU $660，差两个数量级）。
# 同 _RETRIEVAL_MODE 用「按 session_dir 聚合的模块级 dict」而非裸 ContextVar：planner 与
# shipping_calc 是两个工具、各自在独立 context 里跑，前者 set 的 ContextVar 后者读不到。
# 存 (ISO 码, 是否为系统假设值)：assumed=True 表示用户从没说过收货国、是 env 默认兜的，
# 此时回复必须标注假设，且**不该**把它当用户事实沉进会话 slots / 长期记忆。
_DEST_COUNTRY: dict[str, tuple[str, bool]] = {}

# 本轮**原始用户 query**（未经任何 LLM 转述）——工具侧唯一的「用户到底说了什么」确定性信号源。
# planner 的 domains / category 都是 LLM 结构化输出，「合法但错」时下游拿它当锚会静默反转
# （品类门反着杀）；反证只能靠独立信号，而独立信号只有原文词面。按 session_dir 聚合（理由见
# _DEST_COUNTRY）。
_ORIGINAL_QUERY: dict[str, str] = {}


def set_thread_context(thread_id: str, session_dir: Path, user_id: str | None = None) -> None:
    """在请求入口写入本次任务的身份信息。

    一般通过 ``thread_scope`` 调用以保证离开作用域时自动还原；直接调用时不返回
    token，无法 reset，仅适用于进程级一次性绑定（如离线脚本）。
    """
    _thread_id_var.set(thread_id)
    _session_dir_var.set(session_dir)
    _user_id_var.set(user_id)


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


def set_session_pt(pt: "SessionPrefState | None") -> None:
    """写入本会话的短期状态 P_t。两个写入点：``run_agent`` 入口（从 session.json 读回后）与
    ``planner``（识别出本轮约束后当轮改写）。按 session_dir 聚合，故**跨工具可见**；fork 子 Agent
    继承父 session_dir，因此天然读到同一份。无 session_dir（单测直调工具）时静默丢弃。"""
    sd = get_session_dir()
    if sd is None:
        return
    if pt is None:
        _SESSION_PT.pop(str(sd), None)
    else:
        _SESSION_PT[str(sd)] = pt


def get_session_pt() -> "SessionPrefState | None":
    """读取本会话的 P_t；未设置（无会话上下文 / 首轮空态）时返回 None。"""
    sd = get_session_dir()
    if sd is None:
        return None
    return _SESSION_PT.get(str(sd))


def reset_session_pt() -> None:
    """清掉本会话的 P_t（run_agent 收尾，与 reset_session_tasks 对称——模块级 dict 不像
    ContextVar 会随 task 结束自动回收，不清就会按 session_dir 一直攒着）。"""
    sd = get_session_dir()
    if sd is not None:
        _SESSION_PT.pop(str(sd), None)


def get_session_dir() -> Path | None:
    """读取当前任务的会话目录；无上下文时返回 None。"""
    return _session_dir_var.get()


def set_original_query(query: str) -> None:
    """记下本轮原始用户 query（``run_agent`` 入口写，每轮覆盖）。

    给 planner 的域反证与 item_picker 的品类门锚核验当独立信号：LLM 结构化输出互相印证
    没有意义（domains 与 category 同出一张嘴），能反证它们的只有用户原文的词面。
    """
    sd = get_session_dir()
    if sd is not None:
        _ORIGINAL_QUERY[str(sd)] = query


def get_original_query() -> str:
    """读本轮原始用户 query；无会话作用域（单测）返回空串 = 无反证证据，一切照旧。"""
    sd = get_session_dir()
    if sd is None:
        return ""
    return _ORIGINAL_QUERY.get(str(sd), "")


def reset_original_query() -> None:
    """收尾清理（模块级 dict 按 session_dir 为键，不清会无界增长）。"""
    sd = get_session_dir()
    if sd is not None:
        _ORIGINAL_QUERY.pop(str(sd), None)


def set_session_tasks(tasks: Sequence[str]) -> None:
    """记下 planner 本轮判定的任务清单。由 planner 工具写，阶段机的转移通告读。"""
    sd = get_session_dir()
    if sd is not None:
        _SESSION_TASKS[str(sd)] = list(tasks)


def get_session_tasks() -> list[str]:
    """读 planner 本轮的任务判定；planner 还没跑（或无会话作用域）时返回空列表。

    空列表是安全侧：读方（转移通告）只在**确定无比价诉求**时才提示跳过 price_compare，
    判不出来就不提示——多调一次工具只是慢，错误提示跳过会漏掉用户真要的比价。
    """
    sd = get_session_dir()
    if sd is None:
        return []
    return list(_SESSION_TASKS.get(str(sd), []))


def reset_session_tasks() -> None:
    """清掉本会话的任务判定（``run_agent`` 开局 + 收尾调）。

    开局清防上一轮残留、收尾清防模块级 dict 无界增长。
    """
    sd = get_session_dir()
    if sd is not None:
        _SESSION_TASKS.pop(str(sd), None)


def set_dest_country(country: str, assumed: bool = False) -> None:
    """记下 planner 本轮确定的收货国（ISO 码）+ 它是不是系统假设的。由 planner 工具写。

    收货国是**确定性判断**（用户原话规则解析 > 会话 slots > 长期记忆 > env 默认），不让模型
    每轮自由填——同 currency 的老教训（「预算 500」曾被轮流猜成 ₹/¥/$）。判完写这里，让
    shipping_calc 读得到，而不是指望模型每次都记得把参数传对。
    """
    sd = get_session_dir()
    if sd is not None:
        _DEST_COUNTRY[str(sd)] = (country.strip().upper(), assumed)


def get_dest_country() -> str:
    """读本轮收货国；planner 还没跑（或无会话作用域，如单测 / examples）时回落 env 默认值。

    这是 shipping_calc 的**机制兜底**：即便模型漏传 / 传错 dest_country，工具拿到的仍是系统
    认定的那个国家。默认值与 ``app.recall.geo.DEFAULT_DEST_COUNTRY`` 同源——这里单独读一次 env
    而不 import geo，是为了不把整个 recall 包（qdrant / towers 等重模块）拖进 api 底层。
    """
    sd = get_session_dir()
    if sd is not None:
        hit = _DEST_COUNTRY.get(str(sd))
        if hit:
            return hit[0]
    return (os.getenv("DEFAULT_DEST_COUNTRY", "CN") or "CN").strip().upper()


def is_dest_country_assumed() -> bool:
    """本轮收货国是不是系统假设的（用户从没说过）。planner 没跑过时按「是」算。

    curate_turn 用它决定要不要把收货国沉进会话 slots：假设值不是用户事实，沉下去会让
    「系统默认」在下一轮伪装成「用户说过」，越滚越真。
    """
    sd = get_session_dir()
    if sd is not None:
        hit = _DEST_COUNTRY.get(str(sd))
        if hit:
            return hit[1]
    return True


def reset_dest_country() -> None:
    """清掉本会话的收货国（``run_agent`` 开局 + 收尾调）。

    开局清：同 thread 续聊换了收货国时，别让上一轮的国家赖着不走。
    收尾清：模块级 dict 按 session_dir 为键，不清会无界增长。
    """
    sd = get_session_dir()
    if sd is not None:
        _DEST_COUNTRY.pop(str(sd), None)


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
