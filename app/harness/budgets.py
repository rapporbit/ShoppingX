"""fork 预算 / 并发闸 / 各类受限工具集与上限常量（Feedforward / Computational）。

这些是「弱模型的职责边界与死循环动机必须用机制兜」的落点——prompt 只当辅助。各闸的实际拦截
发生在 ``app/harness/hooks/tool_gates.py`` 的 pre_tool_call Hook 里，本模块只提供状态与常量。
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

from app.harness.sentinels import FORK_EXHAUSTED_PARALLEL, FORK_EXHAUSTED_SERIAL
from app.utils.env import env_int

# 「商品检索」类工具：拿信息但不推进收尾，是「再找找更好的」这个动机最爱漏出来的两个口子。
# 预算打在**检索总量**（动机）而非 fork（机制）上：堵死 fork 口，动机会改道成主 loop 直调
# item_search/web_search 漏出来（挤气球）。所以把主 loop 直调与子里的 item_search 都计进同一个
# 全树计数（见 retrieval_budget.py，按 session_dir 聚合），fork 渠道与直调渠道一起兜。
# 不计 category_insight：它是品类常识（RAG）、不是「找更好商品」的渠道。
RETRIEVAL_TOOLS = frozenset({"item_search", "web_search"})

# 一棵 fork 树（一次 run_agent）的商品检索总量上限。
TREE_RETRIEVAL_BUDGET = env_int("RETRIEVAL_BUDGET", 8)

# 复用轮（planner 判 retrieval=reuse）的全树检索小预算。阶段白名单禁令已撤（重构第三段），
# reuse 轮「不重新检索」从硬禁令降为经济约束：给一笔小预算而不是锁死——planner 的 reuse 判定
# 是**假设不是承诺**（2026-07-14 线上死锁即换品类被误判 reuse），模型确认旧候选不适用时应能
# 立即补搜，而不是攒 2 次被拒换一次逃生。``max(1, ...)`` 钉死「永不为 0」：配 0 等于把预算制
# 又改回禁令制，死锁风险回归。refine_backfill / phase_rollback 把 mode 改写为 augment 后自动
# 恢复全树预算（见 tool_gates.charge_retrieval）。
REUSE_RETRIEVAL_BUDGET = max(1, env_int("REUSE_RETRIEVAL_BUDGET", 1))

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

# fork 元工具：本项目里只有主 loop 会调（子 Agent 在 MAX_FORK_DEPTH=1 下 fork 即被深度护栏拒）。
FORK_TOOLS = frozenset({"dispatch_tool", "parallel_dispatch_tool"})

# 「成本放大器」工具：会派生更多模型调用 / 外呼、让 token 成本乘法累积的几个口子。token 预算越
# 硬线时执行层硬挡这些工具，逼 Agent 用现有候选走收尾。便宜的收尾 / 精挑工具与终结工具保留，
# 让任务能「花得起地」结束，而非硬停丢掉已收敛的候选。
COST_AMPLIFIER_TOOLS = FORK_TOOLS | RETRIEVAL_TOOLS | frozenset({"category_insight"})

# 主 loop 专属的终结工具集：与 tool_registry.TERMINAL_TOOLS 一致的字面量，本地重复定义是为了避免
# 循环导入（tool_registry → dispatch_tool → harness）。
TERMINAL_TOOLS = frozenset({"shopping_summary", "chat_fallback"})

# 主 loop 没调终结工具就打算用纯文字收尾时，最多提醒一次——避免模型持续不听指令时无限重试。
MAX_TERMINAL_NUDGE_RETRIES = 1

# 并行 fork 轮数上限（标准购物流程：跨平台检索只用一次 parallel_dispatch_tool）。
DEFAULT_MAX_PARALLEL_FORK = 1
# 串行 dispatch_tool 上限（少量独立深子任务用；并行轮一旦跑过则一律不再放行）。
DEFAULT_MAX_SERIAL_FORK = 4


class ForkBudget:
    """一棵 fork 树共享的 fork 计数（可变对象，靠 ContextVar 把同一引用传给所有子任务）。

    只有主 loop 会 charge 它（子 Agent 在 MAX_FORK_DEPTH=1 下无法再 fork）。语义：
    - 跨平台并行 fork（``parallel_dispatch_tool``）只放行 ``max_parallel`` 轮（默认 1）。
    - 串行 ``dispatch_tool`` 在并行轮之前可用 ``max_serial`` 次；并行轮一旦跑过，之后任何 fork
      一律拒——此时该进比价/精挑/收尾，不该再拓宽。
    """

    __slots__ = ("max_parallel", "max_serial", "parallel_calls", "serial_calls")

    def __init__(self, max_parallel: int, max_serial: int) -> None:
        self.max_parallel = max_parallel
        self.max_serial = max_serial
        self.parallel_calls = 0
        self.serial_calls = 0

    def charge(self, tool_name: str) -> str | None:
        """记一次 fork 调用，返回 None=放行 / 拦截哨兵文案=拒（应硬挡）。

        耗尽原因有两种、文案不同（拒的理由对模型必须真实，不能张冠李戴）：
        - 并行轮已跑过一轮 → ``FORK_EXHAUSTED_PARALLEL``。
        - 纯串行超过 ``max_serial``（可能从未跑过并行轮）→ ``FORK_EXHAUSTED_SERIAL``。
        """
        if tool_name == "parallel_dispatch_tool":
            self.parallel_calls += 1
            return None if self.parallel_calls <= self.max_parallel else FORK_EXHAUSTED_PARALLEL
        if self.parallel_calls >= self.max_parallel:
            return FORK_EXHAUSTED_PARALLEL
        self.serial_calls += 1
        return None if self.serial_calls <= self.max_serial else FORK_EXHAUSTED_SERIAL


# ContextVar 存的是可变对象的引用：asyncio 子任务复制 context 拿到的是**同一个** ForkBudget，
# 主 loop 的多轮 fork 累加到一处。未开作用域为 None → fork 闸不设限。
_fork_budget: ContextVar[ForkBudget | None] = ContextVar("shoppingx_fork_budget", default=None)


def get_fork_budget() -> ForkBudget | None:
    """取当前 fork 树的 fork 预算；无作用域返回 None（不设限）。"""
    return _fork_budget.get()


@contextmanager
def fork_budget_scope(
    max_parallel: int = DEFAULT_MAX_PARALLEL_FORK,
    max_serial: int = DEFAULT_MAX_SERIAL_FORK,
) -> Iterator[ForkBudget]:
    """开一棵 fork 树的 fork 预算作用域：``run_agent`` 入口套一次，拦住主 loop 多轮 re-fork。"""
    budget = ForkBudget(max_parallel, max_serial)
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
