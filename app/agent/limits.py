"""防失控的**上限**：一个 loop 最多跑多久、最多几轮。全在这一页上。

起因是审查报告 P1-6：防失控的数字此前分在几个文件里，要回答「一次失控最多烧多少」得翻好几处
——而这正是最需要一眼看全的那组数字。A4 删掉子 Agent 后，派发超时 / worker 迭代上限 / 派发深度
三个数字随之删除，只剩主 loop 两个。

**这里只放定义，不放消费。** 各模块照旧 ``from app.agent.limits import X`` 后用本模块名引用，
所以 ``monkeypatch.setattr(agents, "MAIN_MAX_ITERS", 2)`` 这类既有写法仍然有效（消费点没动）。

**检索预算刻意没有搬进来**，别再「顺手补齐」：检索预算（``harness/budgets.py``）与 token 档位
阈值（``model_router._thresholds``）是**每次现算**的 env 读取，不是模块常量——后台热更新改了
env 当场生效靠的就是这一点。搬成模块级字面量会把它们冻在进程启动那一刻，且没有任何测试会红。
"""

from app.utils.env import env_int

# 一次任务的墙钟上限。到点抛 TimeoutError 收场（见 orchestrator 的 asyncio.timeout）——
# 看门狗会在远早于它的位置先给用户一个交代，这条是最后的兜底。
MAIN_AGENT_TIMEOUT_SEC = env_int("MAIN_AGENT_TIMEOUT_SEC", 300)

# 主 loop 的迭代上限。这是**真·迭代数**（一轮 Think→Act 算一次），
# 不是某些框架里按「超步」计数的那种口径。
MAIN_MAX_ITERS = env_int("MAIN_AGENT_MAX_ITERATIONS", 30)
