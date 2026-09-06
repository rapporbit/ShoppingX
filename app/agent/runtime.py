"""主链路跑在哪个运行时——``run_agent`` 的唯一选择点（批 0 迁移期）。

原来这个 if-else 只写在 ``app/api/server.py`` 里，于是**只有 HTTP 入口**能切运行时；离线的
评测 / 训练脚本（``run_rubric`` / ``pt_longrun`` / ``tool_rt_baseline``）各自 ``from
app.agent.main_agent import run_agent`` 写死了 LangChain。这会让 L8 的验收无从做起——手册要求
「对同一 15 条子集跑 Rubric 与迁移前基线对照」，可脚本压根跑不到新链路上去。

所以把选择点收进本模块，四个入口共用一个函数：**换运行时只改 ``.env`` 一处**。

**进程级、启动时读一次**，刻意不进后台热更新那套旋钮：换运行时会换掉 Agent 装配、事件流与
会话恢复格式，跑到一半切过去只会得到一个半新半旧的会话。要切就重启进程——这与
``config_overrides`` 那类「只盖运行中进程」的参数是两类东西，混在一起会让线上行为与 .env
对不上（记忆 structured-output-method-must-be-pinned 就是这么踩的）。

L8 摘掉 LangChain 时，本模块连同 ``AGENT_RUNTIME`` 一起删，``orchestrator.run_agent`` 占回唯一实现。
"""

import os
from collections.abc import Awaitable, Callable
from typing import Any

RUNTIME_AGENTSCOPE = "agentscope"
RUNTIME_LANGCHAIN = "langchain"

RunAgent = Callable[..., Awaitable[dict[str, Any]]]


def agent_runtime() -> str:
    """当前进程选中的运行时名（``.env`` 的 ``AGENT_RUNTIME``，缺省 ``langchain``）。

    每次现读 env 而不缓存：值本身极轻，而缓存会让测试里的 ``monkeypatch.setenv`` 不生效——
    真正「只读一次」的语义由调用方（模块级常量 / 启动时解析）表达，不该由本函数强加。
    """
    return (os.environ.get("AGENT_RUNTIME") or RUNTIME_LANGCHAIN).strip().lower()


def resolve_run_agent() -> RunAgent:
    """按 ``AGENT_RUNTIME`` 取 ``run_agent`` 实现。两个实现签名逐字相同，调用方无需分支。

    import 放在函数内：两条链路各自拖着不轻的依赖树（LangChain 那条更是 L8 要摘的），
    模块级 import 会让「只用 AgentScope」的进程也被迫加载 LangChain。
    """
    if agent_runtime() == RUNTIME_AGENTSCOPE:
        from app.agent.orchestrator import run_agent as as_run_agent

        return as_run_agent
    from app.agent.main_agent import run_agent as lc_run_agent

    return lc_run_agent
