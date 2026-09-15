"""fork 预算 / 并发闸 / 各类受限工具集与上限常量（Feedforward / Computational）。

这些是「弱模型的职责边界与死循环动机必须用机制兜」的落点——prompt 只当辅助。各闸的实际拦截
发生在 ``app/harness/hooks/budget.py`` 的 pre_tool_call Hook 里，本模块只提供状态与常量。
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

from app.agent.constants import TERMINAL_TOOLS as _TERMINAL_TOOLS
from app.harness.sentinels import FORK_EXHAUSTED
from app.utils.env import env_int

# 「商品检索」类工具：拿信息但不推进收尾，是「再找找更好的」这个动机最爱漏出来的两个口子。
# 预算打在**检索总量**（动机）而非 fork（机制）上：堵死 fork 口，动机会改道成主 loop 直调
# item_search/web_search 漏出来（挤气球）。所以把主 loop 直调与子里的 item_search 都计进同一个
# 全树计数（见 retrieval_budget.py，按 session_dir 聚合），fork 渠道与直调渠道一起兜。
# 不计 category_insight：它是品类常识（RAG）、不是「找更好商品」的渠道。
RETRIEVAL_TOOLS = frozenset({"item_search", "web_search"})

# 一棵 fork 树（一次 run_agent）的商品检索总量上限。
TREE_RETRIEVAL_BUDGET = env_int("RETRIEVAL_BUDGET", 8)

# 无 session 作用域（单测 / 无树）时的 per-instance 回退上限。
DEFAULT_RETRIEVAL_CAP = 6

# 子 Agent（depth≥1）单平台 item_search 上限 = 1：强制单平台**恰好一次召回**。**实测**：子不靠
# 软提示收敛，哪怕召回非空也会一直「再找找更好」搜到迭代上限。故收成 1：搜满 1 次后执行层硬挡。
# 可经 env 调（设 2 即恢复「1 初搜 + 1 纠偏」）。
SUB_ITEM_SEARCH_CAP = env_int("SUB_ITEM_SEARCH_CAP", 1)

# 聚合 / 终结类工具：本质上需要**跨平台 / 跨商品合流后的全局视图**——只搜了一个平台的子 Agent
# 没有别家数据，去比价/算到手价/精挑/收尾在逻辑上就是错的。故做一道**深度闸（权限闸）**：仅
# depth==0（主 loop）可调。这不破坏同质 fork——子依然握着全部检索工具（能力同质），只是收回
# 「终结授权」（授权不同质）。
DEPTH0_ONLY_TOOLS = frozenset({"price_compare", "shipping_calc", "item_picker", "shopping_summary"})

# 平台无关的「上下文」工具：planner（意图拆解）与 category_insight（品类常识）都不依赖具体平台，
# 主流程跑一次就够，结果经 demands 喂给所有子。子 Agent 调即硬挡——否则 N 个平台子各跑一遍，
# 纯属重复解码。
MAIN_ONLY_CONTEXT_TOOLS = frozenset({"planner", "category_insight"})

# 派发元工具：本项目里只有主 Agent 会调（worker 的工具集里根本没有它，深度闸是二道保险）。
FORK_TOOLS = frozenset({"task_dispatch"})

# 「成本放大器」工具：会派生更多模型调用 / 外呼、让 token 成本乘法累积的几个口子。token 预算越
# 硬线时执行层硬挡这些工具，逼 Agent 用现有候选走收尾。便宜的收尾 / 精挑工具与终结工具保留，
# 让任务能「花得起地」结束，而非硬停丢掉已收敛的候选。
COST_AMPLIFIER_TOOLS = FORK_TOOLS | RETRIEVAL_TOOLS | frozenset({"category_insight"})

# 终结工具集从 ``app.agent.constants`` 读（无依赖模块，正是为打破 tool_registry → dispatch_tool
# → harness 这个环而设）。这里曾是一份**只有 2 个**的复制品：复制时说好「与 tool_registry 一致」，
# 之后那边加了 create_order / cancel_order，这边没跟——交易轮的收尾判定因此走的是另一套。
# 复制常量的成本从来不在复制那一刻，在此后每一次只改了一处的修改。
TERMINAL_TOOLS = _TERMINAL_TOOLS

# 主 loop 没调终结工具就打算用纯文字收尾时，最多提醒一次——避免模型持续不听指令时无限重试。
MAX_TERMINAL_NUDGE_RETRIES = 1

# 一棵树允许的派发**总次数**。一条 demand 就是一次 task_dispatch 调用（同轮多条由框架并发
# 跑），所以额度按调用数给：6 ≈ 一次铺满 5 个启用平台
# + 1 条补派。给得偏松是有意的——派发额度是**动机闸**（挡「再找找更好的」），不该在正常的
# 跨平台铺开时就咬人；真正的资源背压在并发信号量那边（见 fork_concurrency_scope）。
DEFAULT_MAX_DISPATCH = env_int("MAX_DISPATCH_CALLS", 6)


class ForkBudget:
    """一棵派发树共享的调用计数（可变对象，靠 ContextVar 把同一引用传给所有子任务）。

    只有主 Agent 会 charge 它（worker 的工具集里没有 task_dispatch）。语义就一条：整棵树最多
    派 ``max_calls`` 次，超了硬挡并回哨兵文案。
    """

    __slots__ = ("calls", "max_calls")

    def __init__(self, max_calls: int) -> None:
        self.max_calls = max_calls
        self.calls = 0

    def charge(self, tool_name: str) -> str | None:
        """记一次派发，返回 None=放行 / 哨兵文案=拒（应硬挡）。"""
        self.calls += 1
        return None if self.calls <= self.max_calls else FORK_EXHAUSTED

    @property
    def dispatched(self) -> bool:
        """本轮是否已经派出去过——「派过就别再自己直搜」那道闸的判据。"""
        return self.calls > 0


# ContextVar 存的是可变对象的引用：asyncio 子任务复制 context 拿到的是**同一个** ForkBudget，
# 主 loop 的多轮 fork 累加到一处。未开作用域为 None → fork 闸不设限。
_fork_budget: ContextVar[ForkBudget | None] = ContextVar("shoppingx_fork_budget", default=None)


def get_fork_budget() -> ForkBudget | None:
    """取当前 fork 树的 fork 预算；无作用域返回 None（不设限）。"""
    return _fork_budget.get()


@contextmanager
def fork_budget_scope(max_calls: int = DEFAULT_MAX_DISPATCH) -> Iterator[ForkBudget]:
    """开一棵派发树的预算作用域：``run_agent`` 入口套一次，拦住主 Agent 一轮轮重复派发。"""
    budget = ForkBudget(max_calls)
    token = _fork_budget.set(budget)
    try:
        yield budget
    finally:
        _fork_budget.reset(token)


# 单任务内同时在跑的子 Agent 数上限（fork 级背压）。与 ``ForkBudget`` **正交**：ForkBudget 限
# 「这棵树总共能 fork 几次/几轮」（动机闸），这里限「同一时刻最多几个子 Agent 并发执行」（资源闸）。
DEFAULT_FORK_CONCURRENCY = env_int("FORK_CONCURRENCY", 5)

_fork_semaphore: ContextVar[asyncio.Semaphore | None] = ContextVar(
    "shoppingx_fork_semaphore", default=None
)


@contextmanager
def fork_concurrency_scope(limit: int = DEFAULT_FORK_CONCURRENCY) -> Iterator[asyncio.Semaphore]:
    """开一棵 fork 树共享的子 Agent 并发闸：超出**排队**（非拒绝）。

    在 ``run_agent`` 入口与 :func:`fork_budget_scope` 并列套一次。Semaphore 在事件循环运行中创建，
    符合「Semaphore 应绑定运行中的 loop」的要求。
    """
    sem = asyncio.Semaphore(limit)
    token = _fork_semaphore.set(sem)
    try:
        yield sem
    finally:
        _fork_semaphore.reset(token)


def get_fork_semaphore() -> asyncio.Semaphore | None:
    """取当前 fork 树的子 Agent 并发闸；无作用域（单测/示例）返回 None（不限并发）。"""
    return _fork_semaphore.get()
