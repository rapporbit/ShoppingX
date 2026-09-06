"""主 Agent / worker 的装配（批 0 / L3）。

一次 AgentLoop 需要三样东西各建一份、彼此对应：**一个 ``HarnessSession``**（控制面状态）、
**一份 Toolkit**（工具实例上挂着那个 session 的工具适配器）、**一个 Agent**（模型适配器也拿
同一个 session）。三者绑成一套是硬要求：AgentScope 把模型钩子与工具钩子拆成了两个类，它们
之间的三条接力通道全靠共享的 session 传（见 ``app/harness/adapter.py``），错配就是 worker
的断言流进主 loop、或者两个并发 worker 互相污染循环检测。

**批 1 起是 Supervisor-Workers**：主 Agent 持全部业务工具、**单干优先**；worker 按读写属性切成
两种，边界靠三样结构性保证（Toolkit 发放范围 / ``is_read_only`` 标记 / ``PermissionEngine``
精准放行），不靠提示词劝退。SearchAgent 手上根本没有写工具，TradeAgent 手上根本没有检索工具。

``WORKER_MODE=clone`` 保留批 0 的同质克隆形态（同工具集、同 system prompt，只隔离 thread /
上下文 / 控制面状态），用途只有一个：在**同一运行时、同一批 query** 上量出两种结构的差异，
作 Q13「为什么不用同质 fork」的同框架对照基准。
"""

import os
from collections.abc import Sequence

from agentscope.agent import Agent, ContextConfig, ReActConfig
from agentscope.middleware import MiddlewareBase
from agentscope.state import AgentState

from app.agent.llm import get_fast_llm, get_llm, get_model_config
from app.agent.permissions import allow_tools
from app.agent.prompts import get_system_prompt, get_worker_system_prompt
from app.agent.tool_registry import build_toolkit
from app.agent.tracing import tracing_middlewares
from app.harness.adapter import HarnessAgentAdapter, HarnessSession, HarnessToolAdapter
from app.utils.env import env_int

# 主 loop 的迭代上限（防失控之②）。这是**真·迭代数**（一轮 Think→Act 算一次），
# 不是某些框架里按「超步」计数的那种口径。
MAIN_MAX_ITERS = env_int("MAIN_AGENT_MAX_ITERATIONS", 30)
# worker 的迭代上限，**按 kind 分档**：检索子任务要留出「召回跑题换一次词重搜」的余量；
# 交易子任务是查→改两跳的确定性动作（query_order → cancel_order），4 轮还收不住说明它在
# 里面乱试，早掐比让它继续试更安全（写工具的每一次试都在改真实状态）。
WORKER_MAX_ITERS = env_int("SUB_AGENT_MAX_ITERATIONS", 6)
TRADE_MAX_ITERS = env_int("TRADE_AGENT_MAX_ITERATIONS", 4)

# ``split`` = 读写切分（批 1 起的默认）；``clone`` = worker 是主 Agent 的完整克隆（批 0 的
# 过渡形态）。开关留着不是为了「以后可能要用」，而是为了能在**同一运行时、同一批 query**
# 上量出两种结构的差异——否则「Supervisor-Workers 比同质 fork 好」就只是一句话。
WORKER_MODE = os.environ.get("WORKER_MODE", "split")


async def _assemble(
    *,
    name: str,
    role: str,
    max_iters: int,
    original_query: str = "",
    image_paths: Sequence[str] = (),
    state: AgentState | None = None,
    fast_model: bool = False,
    system_prompt: str | None = None,
) -> tuple[Agent, HarnessSession]:
    """按「一个 session + 一份 Toolkit + 一个 Agent」装一套，返回 Agent 与它的 session。

    ``state`` 非空即**会话恢复**：把落盘读回来的那份 ``AgentState`` 原样交给 Agent，它的
    context / permission / tool 上下文一并接上（见 orchestrator 的 agent_state.json）。
    """
    session = HarnessSession(original_query=original_query, image_paths=image_paths)
    # 工具适配器挂在**工具实例**上，所以工具实例不能跨 loop 复用 —— build_toolkit 每次按需
    # 重建一批壳（壳很薄，底下的实现函数与 schema 仍是同一份，见 tool_registry）。
    toolkit = await build_toolkit(role, tool_middlewares=[HarnessToolAdapter(session)])
    agent_state = state if state is not None else AgentState()
    # 写工具精准放行：不用 BYPASS，见 app/agent/permissions.py。
    allow_tools(agent_state)
    # 观测：框架原生的 ``TracingMiddleware`` 打标准 GenAI 语义属性，Langfuse（本身是 OTEL SDK
    # 包装）的 span 过滤器按 ``gen_ai.*`` 放行 —— 两头自动对上，不需要胶水（见 tracing.py 尾部）。
    # 未启用观测时返回空表，主 + worker 一视同仁：trace 里不会出现「有的轮有、有的轮没有」的空洞。
    # 顺序上放在控制面**后面**：适配器改写 messages / 换档发生在前，trace 记的是真正发出去的那份。
    middlewares: list[MiddlewareBase] = [HarnessAgentAdapter(session), *tracing_middlewares()]
    agent = Agent(
        name=name,
        # system prompt 纯静态（无运行时注入）→ 跨轮 / 跨会话字节稳定、可命中 prompt cache。
        # 主 Agent 用主 prompt；split 模式的 worker 用自己那段专职 prompt（同类 worker 之间
        # 共用一条前缀），clone 模式的 worker 仍与主 Agent 逐字相同——那是对照组的定义。
        system_prompt=system_prompt or get_system_prompt(),
        # 模型分层：worker 用快档（关思考）砍解码延迟——它在收窄后的子任务里只做 1~2 跳检索，
        # 不需要深推理。换的只是「档位」，工具集与 prompt 仍与主 Agent 一致。
        model=get_fast_llm() if fast_model else get_llm(),
        toolkit=toolkit,
        middlewares=middlewares,
        state=agent_state,
        model_config=get_model_config(),
        react_config=ReActConfig(max_iters=max_iters),
        # 上下文治理由本仓自己做（Cache Breakpoint + block 级压缩，挂在 pre_think）。这里把框架
        # 自带的摘要压缩触发线顶到允许的最高值（0.9 × 128k ≈ 115k），让它成为纯兜底；真撞上了
        # 也不会跑框架的 LLM 摘要——``HarnessAgentAdapter.on_compress_context`` 已接管，理由见那里。
        context_config=ContextConfig(trigger_ratio=0.9),
    )
    return agent, session


async def build_main_agent(
    *,
    original_query: str = "",
    image_paths: Sequence[str] = (),
    state: AgentState | None = None,
) -> tuple[Agent, HarnessSession]:
    """装配主 Agent（Supervisor）。

    ``original_query`` 是**未经 LLM 转述**的本轮用户原文，交给控制面当漂移检测与语义断言的
    对齐基准（见 harness 的 drift_detector）——不是给模型看的，模型看的是 orchestrator 拼的
    那条用户消息。``image_paths`` 同理交给控制面：开局预置要先把图看掉再拆意图
    （见 ``HarnessAgentAdapter._prefill``）。

    基座模型是主档（开思考）：主 loop 第 1 轮是全链路唯一没被机制锁死的决策（购物还是闲聊、
    先拆解还是先查品类、自己干还是派 worker），值得让它想清楚；第 2 轮起决策空间已被阶段机
    与候选 id 化夹死，由 ``harness.hooks.reasoning_boost`` 决定还要不要继续开。
    """
    return await _assemble(
        name="shoppingx",
        role="main",
        max_iters=MAIN_MAX_ITERS,
        original_query=original_query,
        image_paths=image_paths,
        state=state,
    )


async def build_worker_agent(kind: str = "search") -> Agent:
    """装配一个 worker（``search`` 只读 / ``trade`` 写）。

    ``split``（默认）：``kind`` 决定三件事——**拿得到哪些工具**（``tool_registry._ROLE_TOOLS``，
    这是读写边界的结构性保证）、**哪段 system prompt**、**几轮上限**。SearchAgent 的 Toolkit 里
    根本没有写工具，TradeAgent 的 Toolkit 里根本没有检索工具（所以「买第 2 个」的候选定位必须由
    主 Agent 在 demands 里给定 item_id）。

    ``clone``：批 0 的过渡形态，worker = 主 Agent 的完整克隆（全集工具 + 同一段 system prompt），
    只在 thread / 上下文 / 控制面状态上隔离。留着是为了在同一运行时、同一批 query 上量出两种
    结构的差异（Q13 的同框架对照基准），**不是**为了将来还要用。

    只回 Agent 不回 session：worker 的控制面状态是它自己的私事，派发方（``_run_worker``）
    只关心最终那条回复。
    """
    if WORKER_MODE == "clone":
        agent, _ = await _assemble(
            name=f"shoppingx-{kind}",
            role="main",
            max_iters=WORKER_MAX_ITERS,
            fast_model=True,
        )
        return agent
    agent, _ = await _assemble(
        name=f"shoppingx-{kind}",
        role=kind,
        max_iters=TRADE_MAX_ITERS if kind == "trade" else WORKER_MAX_ITERS,
        fast_model=True,
        system_prompt=get_worker_system_prompt(kind),
    )
    return agent
