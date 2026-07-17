"""跨整棵 fork 树共享的「商品检索」预算 + 召回信号（按 session_dir 聚合）。

为什么把预算打在「检索总量」而不是「fork 机制」上：``MAX_FORK_DEPTH`` / ForkBudget 只堵了
**fork 这个机制**，但「再找找更好的」这个**动机**没消失——堵死 fork 口，压力就顶到主 loop
还握着的直调 ``item_search`` / ``web_search``（挤气球）。所以把主 loop 直调与子里的 item_search
都计进**同一个计数器**，过阈值由 middleware 注入强制收尾信号：fork 渠道和直调渠道一起兜。

为什么用「按 session_dir 为键的模块级 dict」而不是裸 ContextVar：asyncio 子任务创建时会**拷贝**
一份 context，子里对 ContextVar 的 ``set`` 不会回传父 loop——想跨 fork 聚合（连子的 item_search
也一起兜）就会静默地数不到子。``session_dir`` 是显式透传给子 Agent 的（``thread_scope`` 让子
**继承父 session_dir**），主和子都按同一 key 自增，才能真正全树聚合。

模块级 dict 需要收尾清理（防无界增长）：``run_agent`` 结束时调 :func:`reset_tree`。
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

from app.api.context import get_session_dir, get_session_tasks, get_thread_id
from app.utils.env import env_int

# 任务口径的 web_search 小配额（窄口径用途门）：planner 判 evaluate / category_intel 时，
# prompt 明确要求「+ web_search 口碑」，但「有候选就拦」的位置门会罚这条路——只能靠连拒 2 次
# 的逃生门走通，每次白花 2 轮往返。判据用 planner 落进 session 的 tasks（确定性信号，不是模型
# 自由意志），配额防挤气球：recommend 主链路照旧拦死，延迟零回退。
WEB_SEARCH_TASK_QUOTA = env_int("WEB_SEARCH_TASK_QUOTA", 2)
_TASKS_WANT_WEB = frozenset({"evaluate", "category_intel"})

# 隔离检索作用域标记：串行 dispatch_tool（独立子任务，如定点商品调查）打开，
# parallel_dispatch_tool（跨平台泛搜，共享收敛信号）不开、维持全树共享语义。
# 见 :func:`isolated_retrieval_scope`。
_isolated_var: ContextVar[bool] = ContextVar("shoppingx_retrieval_isolated", default=False)


@contextmanager
def isolated_retrieval_scope() -> Iterator[None]:
    """独立子任务（串行 dispatch_tool）作用域：本子任务的 web_search 门控只看**自己**的召回结果，
    不受全树其它子任务（如兄弟平台 / 兄弟商品）是否已找到候选影响。

    用于纠正 :func:`web_search_allowed` 的全树共享语义在「多个独立子任务各自调查互不相关的对象」
    场景下的错配——商品 A 搜到了，不该连带拦掉商品 B（B 自己没搜到）的 web_search 兜底。
    只影响 web_search 门控判定，不影响 ``TREE_RETRIEVAL_BUDGET`` 那条真实的全树总成本上限
    （见 :func:`charge_tree_retrieval`，本作用域不碰它）。
    """
    token = _isolated_var.set(True)
    try:
        yield
    finally:
        _isolated_var.reset(token)


@dataclass
class _TreeRetrieval:
    count: int = 0  # item_search + web_search 全树累计（预算计数）
    item_search_runs: int = 0  # item_search 调用次数（含召回为空的）
    web_search_runs: int = 0  # web_search 已执行次数（任务口径配额用，全树共享）
    nonempty_item_search: int = 0  # 召回到 ≥1 候选的 item_search 次数（web_search 兜底门用）
    # 隔离作用域内，按 thread_id 记「这个子任务自己是否搜到过候选」——只在 isolated_retrieval_scope
    # 内才写入 / 读取，供 web_search_allowed 在隔离场景下只看自己、不看全树。
    scoped_nonempty: dict[str, int] = field(default_factory=dict)


# session_dir(str) → 该任务一棵 fork 树的检索状态。主 / 各子 Agent 共享同一条目。
_STATE: dict[str, _TreeRetrieval] = {}


def _key() -> str | None:
    sd = get_session_dir()
    return str(sd) if sd is not None else None


def _state(create: bool = True) -> _TreeRetrieval | None:
    """取当前 session 的检索状态；无 session 作用域（单测）返回 None。"""
    k = _key()
    if k is None:
        return None
    st = _STATE.get(k)
    if st is None and create:
        st = _TreeRetrieval()
        _STATE[k] = st
    return st


def charge_tree_retrieval() -> int | None:
    """item_search / web_search 计一次，返回当前全树累计；无 session 作用域返回 None。"""
    st = _state()
    if st is None:
        return None
    st.count += 1
    return st.count


def peek_tree_retrieval() -> int | None:
    """只读当前全树检索累计（不自增），供「耗尽即夺权」在请求模型前判断要不要摘掉检索工具。

    与 :func:`charge_tree_retrieval` 区别：charge 在工具**执行时**计数，peek 在**请求模型前**
    读数——用它决定下一轮还把不把 item_search/web_search 放进模型可见工具表。无 session 作用域
    或尚未检索过返回 None。
    """
    st = _state(create=False)
    return st.count if st is not None else None


def note_web_search() -> None:
    """web_search 执行时计一次（配额消耗）。挂在 retrieval_charge(45)——门控 websearch_gate(15)
    读的是自增前值（＝之前已完成次数），「已完成 < 配额」即放行，与 item_search_calls 同一套
    顺序契约。无 session 作用域（单测）不计。"""
    st = _state()
    if st is not None:
        st.web_search_runs += 1


def note_item_search(total_recall: int) -> None:
    """item_search 完成后登记一次召回信号（供 web_search 兜底门判定）。

    隔离作用域内（``isolated_retrieval_scope``）额外按当前 thread_id 记一份局部信号，
    供 :func:`web_search_allowed` 在该场景下只看自己、不看全树。
    """
    st = _state()
    if st is None:
        return
    st.item_search_runs += 1
    if total_recall > 0:
        st.nonempty_item_search += 1
        if _isolated_var.get():
            tid = get_thread_id()
            if tid is not None:
                st.scoped_nonempty[tid] = st.scoped_nonempty.get(tid, 0) + 1


def web_search_allowed() -> bool:
    """web_search 此刻是否允许调用。三种合法场景：

    1. **独立知识查询**：还没进入购物检索流程（item_search 未跑过），用户可能在问品牌口碑、
       评测、趋势等外部事实，或按 plan 的 intent_grounding=web 做意图翻译——放行。
    2. **任务口径配额**（窄口径用途门）：planner 判了 evaluate / category_intel——prompt 明确
       要求 web_search 口碑佐证的任务，在 ``WEB_SEARCH_TASK_QUOTA`` 内放行，不再罚它走
       连拒 2 次的逃生门。判据是 planner 落 session 的确定性 tasks，不是模型自由意志。
    3. **购物流程内兜底**：item_search 跑过但全部召回为空——放行，作为兜底补线索。

    recommend 主链路已有候选时仍然拦截——web_search 不是「找更好」的渠道。

    隔离作用域内（独立子任务，如定点商品调查）改按**本子任务自己**的召回结果判定，不受
    全树其它子任务是否已找到候选影响——否则商品 A 搜到了会连带拦掉商品 B 自己的兜底。
    """
    if _key() is None:
        return True  # 无 session 作用域（单测）
    st = _state(create=False)
    if st is None:
        return True  # 还没进入购物检索流程 → 允许独立知识查询
    if st.item_search_runs == 0:
        return True  # 同上：session 存在但还没搜过商品
    if _TASKS_WANT_WEB & set(get_session_tasks()) and st.web_search_runs < WEB_SEARCH_TASK_QUOTA:
        return True  # 评价 / 行情任务的口碑配额（配额尽则落回下面的兜底判定）
    if _isolated_var.get():
        tid = get_thread_id()
        return st.scoped_nonempty.get(tid, 0) == 0 if tid is not None else True
    return st.nonempty_item_search == 0  # 搜过但全空 → 兜底放行；有候选 → 拦


def reset_tree() -> None:
    """清掉本 session 的检索预算条目（任务收尾时调，防模块级 dict 无界增长）。"""
    k = _key()
    if k is not None:
        _STATE.pop(k, None)
