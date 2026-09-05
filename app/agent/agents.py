"""主 Agent / worker 的装配（批 0 / L3）。

一次 AgentLoop 需要三样东西各建一份、彼此对应：**一个 ``HarnessSession``**（控制面状态）、
**一份 Toolkit**（工具实例上挂着那个 session 的工具适配器）、**一个 Agent**（模型适配器也拿
同一个 session）。三者绑成一套是硬要求：AgentScope 把模型钩子与工具钩子拆成了两个类，它们
之间的三条接力通道全靠共享的 session 传（见 ``app/harness/adapter.py``），错配就是 worker
的断言流进主 loop、或者两个并发 worker 互相污染循环检测。

**批 0 的 worker 是主 Agent 的克隆**（同工具集、同 system prompt），只在 thread / 上下文 /
控制面状态上隔离——这是 Q13「为什么不用同质 fork」的**同框架对照基准**，批 1 切读写分工后
它由 ``WORKER_MODE=clone`` 保留，用来在同一运行时下量同一批 query 的差异。
"""

import os

from agentscope.agent import Agent, ReActConfig
from agentscope.middleware import MiddlewareBase
from agentscope.state import AgentState

from app.agent.llm import get_as_fast_llm, get_as_llm, get_model_config
from app.agent.permissions import allow_tools
from app.agent.prompts import get_system_prompt
from app.agent.tool_registry import build_toolkit
from app.harness.adapter import HarnessAgentAdapter, HarnessSession, HarnessToolAdapter
from app.utils.env import env_int

# 主 loop 的迭代上限（防失控之②）。与 LangChain 版的 MAIN_AGENT_MAX_ITERATIONS 同口径，
# 但这里是**真·迭代数**，不必再换算 langgraph 的「超步」。
MAIN_MAX_ITERS = env_int("MAIN_AGENT_MAX_ITERATIONS", 30)
# worker 的迭代上限：单平台检索子任务 category_insight 校准 + 几次自我纠偏 item_search 就够收敛。
WORKER_MAX_ITERS = env_int("SUB_AGENT_MAX_ITERATIONS", 6)

# 批 0 = clone（worker 是主 Agent 的完整克隆）；批 1 起默认 split（读写切分）。
# 开关留着不是为了「以后可能要用」，而是为了能在**同一运行时、同一批 query** 上量出两种
# 结构的差异——否则「Supervisor-Workers 比同质 fork 好」就只是一句话。
WORKER_MODE = os.environ.get("WORKER_MODE", "clone")


async def _assemble(
    *,
    name: str,
    role: str,
    max_iters: int,
    original_query: str = "",
    state: AgentState | None = None,
    fast_model: bool = False,
) -> tuple[Agent, HarnessSession]:
    """按「一个 session + 一份 Toolkit + 一个 Agent」装一套，返回 Agent 与它的 session。

    ``state`` 非空即**会话恢复**：把落盘读回来的那份 ``AgentState`` 原样交给 Agent，它的
    context / permission / tool 上下文一并接上（见 orchestrator 的 agent_state.json）。
    """
    session = HarnessSession(original_query=original_query)
    # 工具适配器挂在**工具实例**上，所以工具实例不能跨 loop 复用 —— build_toolkit 每次按需
    # 重建一批壳（壳很薄，底下的实现函数与 schema 仍是同一份，见 tool_registry）。
    toolkit = await build_toolkit(role, tool_middlewares=[HarnessToolAdapter(session)])
    agent_state = state if state is not None else AgentState()
    # 写工具精准放行：不用 BYPASS，见 app/agent/permissions.py。
    allow_tools(agent_state)
    # 观测（Langfuse）**不在这层挂**：LangChain 版靠 callback handler 走 config，AgentScope 侧
    # 要换成 OTel 的 TracingMiddleware + OTLP 后端，那是 L7 的活。这里先只挂控制面，免得半截
    # 接线让 trace 里出现「有的轮有、有的轮没有」的空洞。
    middlewares: list[MiddlewareBase] = [HarnessAgentAdapter(session)]
    agent = Agent(
        name=name,
        # system prompt 纯静态（无运行时注入）→ 跨轮 / 跨会话字节稳定、可命中 prompt cache；
        # 主与 worker 的 system 段字节相同，worker 也能命中主 Agent 的缓存。
        system_prompt=get_system_prompt(),
        # 模型分层：worker 用快档（关思考）砍解码延迟——它在收窄后的子任务里只做 1~2 跳检索，
        # 不需要深推理。换的只是「档位」，工具集与 prompt 仍与主 Agent 一致。
        model=get_as_fast_llm() if fast_model else get_as_llm(),
        toolkit=toolkit,
        middlewares=middlewares,
        state=agent_state,
        model_config=get_model_config(),
        react_config=ReActConfig(max_iters=max_iters),
    )
    return agent, session


async def build_main_agent(
    *,
    original_query: str = "",
    state: AgentState | None = None,
) -> tuple[Agent, HarnessSession]:
    """装配主 Agent（Supervisor）。

    ``original_query`` 是**未经 LLM 转述**的本轮用户原文，交给控制面当漂移检测与语义断言的
    对齐基准（见 harness 的 drift_detector）——不是给模型看的，模型看的是 orchestrator 拼的
    那条 human message。

    基座模型是主档（开思考）：主 loop 第 1 轮是全链路唯一没被机制锁死的决策（购物还是闲聊、
    先拆解还是先查品类、自己干还是派 worker），值得让它想清楚；第 2 轮起决策空间已被阶段机
    与候选 id 化夹死，由 ``harness.hooks.reasoning_boost`` 决定还要不要继续开。
    """
    return await _assemble(
        name="shoppingx",
        role="main",
        max_iters=MAIN_MAX_ITERS,
        original_query=original_query,
        state=state,
    )


async def build_worker_agent(kind: str = "search") -> Agent:
    """装配一个 worker（``search`` / ``trade``）。

    批 0 的 ``clone`` 模式下 ``kind`` 只影响名字与事件里的标识——工具集仍是全集（见
    ``tool_registry._ROLE_TOOLS``），这正是「同质克隆」的定义。批 1 把 ``_ROLE_TOOLS`` 切开
    之后，同一个 ``kind`` 才真正决定它拿得到哪些工具，**切的是发放范围，不是这里的装配代码**。

    只回 Agent 不回 session：worker 的控制面状态是它自己的私事，派发方（``_run_worker``）
    只关心最终那条回复。
    """
    role = "main" if WORKER_MODE == "clone" else kind
    agent, _ = await _assemble(
        name=f"shoppingx-{kind}",
        role=role,
        max_iters=WORKER_MAX_ITERS,
        fast_model=True,
    )
    return agent
