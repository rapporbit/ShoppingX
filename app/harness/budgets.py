"""受限工具集与上限常量（Feedforward / Computational）。

这些是「弱模型的死循环动机必须用机制兜」的落点——prompt 只当辅助。各闸的实际拦截发生在
``app/harness/hooks/budget.py`` 的 pre_tool_call Hook 里，本模块只提供常量。

A4 删掉子 Agent 时，派发次数 ``ForkBudget``、子 Agent 并发信号量、子搜上限与两道 worker 工具集
（``DEPTH0_ONLY_TOOLS`` / ``MAIN_ONLY_CONTEXT_TOOLS``）随之删除。
"""

from __future__ import annotations

from app.agent.constants import TERMINAL_TOOLS as _TERMINAL_TOOLS
from app.utils.env import env_int

# 「商品检索」类工具：拿信息但不推进收尾，是「再找找更好的」这个动机最爱漏出来的两个口子。
# 预算打在**检索总量**（动机）上，两个口子计进同一个全树计数（见 retrieval_budget.py，按
# session_dir 聚合）。不计 category_insight：它是品类常识（RAG）、不是「找更好商品」的渠道。
RETRIEVAL_TOOLS = frozenset({"item_search", "web_search"})

# 一次 run_agent 的商品检索总量上限。
TREE_RETRIEVAL_BUDGET = env_int("RETRIEVAL_BUDGET", 8)

# 无 session 作用域（单测）时的 per-instance 回退上限。
DEFAULT_RETRIEVAL_CAP = 6

# 「成本放大器」工具：会派生更多模型调用 / 外呼、让 token 成本乘法累积的几个口子。token 预算越
# 硬线时执行层硬挡这些工具，逼 Agent 用现有候选走收尾。便宜的收尾 / 精挑工具与终结工具保留，
# 让任务能「花得起地」结束，而非硬停丢掉已收敛的候选。
COST_AMPLIFIER_TOOLS = RETRIEVAL_TOOLS | frozenset({"category_insight"})

# 终结工具集从 ``app.agent.constants`` 读（无依赖模块，harness 与 tool_registry 共用一份）。
# 这里曾是一份**只有 2 个**的复制品：复制时说好「与 tool_registry 一致」，之后那边加了
# create_order / cancel_order，这边没跟——交易轮的收尾判定因此走的是另一套。
# 复制常量的成本从来不在复制那一刻，在此后每一次只改了一处的修改。
TERMINAL_TOOLS = _TERMINAL_TOOLS

# 主 loop 没调终结工具就打算用纯文字收尾时，最多提醒一次——避免模型持续不听指令时无限重试。
MAX_TERMINAL_NUDGE_RETRIES = 1
