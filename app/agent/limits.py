"""防失控的**上限**：一个 loop 最多跑多久、最多几轮、最多派几层。全在这一页上。

起因是审查报告 P1-6：「派发安全四层」的四个数字此前分在四个文件（超时在 dispatch_tool、迭代
上限在 agents、深度在 fork_guard、派发次数在 harness/budgets），要回答「一次失控最多烧多少」
得翻四个地方——而这正是最需要一眼看全的那组数字。

**这里只放定义，不放消费。** 各模块照旧 ``from app.agent.limits import X`` 后用本模块名引用，
所以 ``monkeypatch.setattr(agents, "MAIN_MAX_ITERS", 2)`` 这类既有写法仍然有效（消费点没动）。

**两类数字刻意没有搬进来**，别再「顺手补齐」：

- 检索预算（``harness/budgets.py``）与 token 档位阈值（``model_router._thresholds``）是**每次
  现算**的 env 读取，不是模块常量——后台热更新改了 env 当场生效靠的就是这一点。搬成模块级
  字面量会把它们冻在进程启动那一刻，且没有任何测试会红。
- ``MAX_DISPATCH_CALLS`` 留在 ``harness/budgets.py``：它跟 ForkBudget 的计数状态是一体的
  （额度与用量写在同一处才对得上账），单独把额度搬走反而更难读。
"""

from app.utils.env import env_int

# ── 主 loop ──
# 一次任务的墙钟上限。到点抛 TimeoutError 收场（见 orchestrator 的 asyncio.timeout）——
# 看门狗会在远早于它的位置先给用户一个交代，这条是最后的兜底。
MAIN_AGENT_TIMEOUT_SEC = env_int("MAIN_AGENT_TIMEOUT_SEC", 300)

# 主 loop 的迭代上限（防失控之②）。这是**真·迭代数**（一轮 Think→Act 算一次），
# 不是某些框架里按「超步」计数的那种口径。
MAIN_MAX_ITERS = env_int("MAIN_AGENT_MAX_ITERATIONS", 30)

# ── worker（派发安全四层之①②）──
SUB_AGENT_TIMEOUT_SEC = env_int("SUB_AGENT_TIMEOUT_SEC", 90)

# worker 的迭代上限，**按 kind 分档**：检索子任务要留出「召回跑题换一次词重搜」的余量；
# 交易子任务是查→改两跳的确定性动作（query_order → cancel_order），4 轮还收不住说明它在
# 里面乱试，早掐比让它继续试更安全（写工具的每一次试都在改真实状态）。
WORKER_MAX_ITERS = env_int("SUB_AGENT_MAX_ITERATIONS", 6)
TRADE_MAX_ITERS = env_int("TRADE_AGENT_MAX_ITERATIONS", 4)

# ── 派发深度（派发安全四层之③）──
# 1 = 主 loop 可以派 worker，worker 不能再派。这条**在结构上已经由 Toolkit 发放范围保证**
# （worker 的工具集里没有 task_dispatch），本常量是二道保险，见 fork_guard。
MAX_FORK_DEPTH = 1
