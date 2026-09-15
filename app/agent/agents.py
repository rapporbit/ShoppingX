"""主 Agent 的装配（批 0 / L3）。

一次 AgentLoop 需要三样东西各建一份、彼此对应：**一个 ``HarnessSession``**（控制面状态）、
**一份 Toolkit**（工具实例上挂着那个 session 的工具适配器）、**一个 Agent**（模型适配器也拿
同一个 session）。三者绑成一套是硬要求：AgentScope 把模型钩子与工具钩子拆成了两个类，它们
之间的三条接力通道全靠共享的 session 传（见 ``app/harness/adapter.py``）。

**A4 起是单环**：一个模型一个 loop，持全部业务工具；并行靠同一轮发多个工具调用（框架并发执行）。
做过 Supervisor-Workers（批 1）与同质 fork（批 0），440 个会话实测派发全是单跳壳后删掉，
数据留在 ``docs/milestones/``。
"""

from collections.abc import Sequence

from agentscope.agent import Agent, ReActConfig
from agentscope.middleware import MiddlewareBase
from agentscope.state import AgentState

from app.agent.limits import MAIN_MAX_ITERS
from app.agent.llm import get_model_config, get_tier_llm, main_loop_tier_base
from app.agent.permissions import allow_tools
from app.agent.prompts import get_system_prompt
from app.agent.tool_registry import build_toolkit
from app.agent.tracing import tracing_middlewares
from app.harness.adapter import HarnessAgentAdapter, HarnessSession, HarnessToolAdapter
from app.harness.middleware import harness
from app.harness.setup import setup_harness

# 迭代上限的定义与理由在 ``app.agent.limits``（防失控的上限全在那一页）。这里按本名引入，
# 消费点仍是本模块的名字——``monkeypatch.setattr(agents, "MAIN_MAX_ITERS", 2)`` 照旧有效。


async def _run_system_prompt_hooks(prompt: str, *, role: str, query: str) -> str:
    """跑 ``on_system_prompt`` 钩子，返回定稿后的 system prompt。

    钩子的契约是**只许往末尾追加**（``context["append"]`` 收集，本函数负责拼）——不给它们
    ``system_prompt`` 的改写权。让钩子随便重写整段的代价太大：谁都能悄悄删掉 ``<termination>``
    那一段，而那正是本仓最贵的一条纪律（Agent 最常见的失败是不收尾死循环）。只许追加，最坏
    情况也只是尾巴上多了段废话。

    追加内容按注册顺序（priority 升序）拼接，所以同一批策略每轮渲染出的字节完全一致。
    """
    setup_harness()  # 幂等；服务启动时已初始化过，这里只兜离线脚本 / 单测
    ctx = await harness.run(
        "on_system_prompt", {"role": role, "query": query, "system_prompt": prompt, "append": []}
    )
    extra = [str(x).strip() for x in (ctx.get("append") or []) if str(x).strip()]
    return prompt + "\n\n" + "\n\n".join(extra) if extra else prompt


async def _assemble(
    *,
    name: str,
    role: str,
    max_iters: int,
    original_query: str = "",
    image_paths: Sequence[str] = (),
    state: AgentState | None = None,
    tier: str,
) -> tuple[Agent, HarnessSession]:
    """按「一个 session + 一份 Toolkit + 一个 Agent」装一套，返回 Agent 与它的 session。

    ``state`` 非空即**会话恢复**：把落盘读回来的那份 ``AgentState`` 原样交给 Agent，它的
    context / permission / tool 上下文一并接上（见 orchestrator 的 session.json）。
    """
    session = HarnessSession(original_query=original_query, image_paths=image_paths)
    base_prompt = await _run_system_prompt_hooks(
        get_system_prompt(), role=role, query=original_query
    )
    # 工具适配器挂在**工具实例**上，所以工具实例不能跨 loop 复用 —— build_toolkit 每次按需
    # 重建一批壳（壳很薄，底下的实现函数与 schema 仍是同一份，见 tool_registry）。
    toolkit = await build_toolkit(role, tool_middlewares=[HarnessToolAdapter(session)])
    agent_state = state if state is not None else AgentState()
    # 写工具精准放行：不用 BYPASS，见 app/agent/permissions.py。
    allow_tools(agent_state)
    # 观测：框架原生的 ``TracingMiddleware`` 打标准 GenAI 语义属性，Langfuse（本身是 OTEL SDK
    # 包装）的 span 过滤器按 ``gen_ai.*`` 放行 —— 两头自动对上，不需要胶水（见 tracing.py 尾部）。
    # 未启用观测时返回空表：trace 里不会出现「有的轮有、有的轮没有」的空洞。
    # 顺序上放在控制面**后面**：适配器改写 messages / 换档发生在前，trace 记的是真正发出去的那份。
    middlewares: list[MiddlewareBase] = [HarnessAgentAdapter(session), *tracing_middlewares()]
    agent = Agent(
        name=name,
        # system prompt 在**装配期**定稿（``on_system_prompt`` 钩子跑完就不再动）→ 一次任务内
        # 跨轮字节稳定、可命中 prompt cache；钩子唯一允许的动作是往**末尾**追加，前面那段
        # （role / workflow / tool_policy / …）逐字不变，所以缓存前缀照常从头命中。
        system_prompt=base_prompt,
        # 模型分层只换「档位」，工具集与 prompt 不动。取值由 llm.py 的档位策略表统一给
        # （``MAIN_LOOP_TIER_BASE``），装配处不再自己判断该用哪档——这正是
        # P0-1 的教训：装配写死一档、Hook 假设另一档，两边都不会报错。
        model=get_tier_llm(tier),
        toolkit=toolkit,
        middlewares=middlewares,
        state=agent_state,
        model_config=get_model_config(),
        react_config=ReActConfig(max_iters=max_iters),
        # 上下文压缩交给框架默认的 ``ContextConfig``（trigger_ratio 0.8）：超阈值时 LLM 写一份
        # continuation summary 进 ``state.summary``，随 session.json 持久化。摘要是二手上下文，
        # 这是有意接受的取舍（见 docs/plans 会话状态重构）。
    )
    return agent, session


async def build_main_agent(
    *,
    original_query: str = "",
    image_paths: Sequence[str] = (),
    state: AgentState | None = None,
) -> tuple[Agent, HarnessSession]:
    """装配主 Agent。

    ``original_query`` 是**未经 LLM 转述**的本轮用户原文，交给控制面当漂移检测与语义断言的
    对齐基准（见 harness 的 drift_detector）——不是给模型看的，模型看的是 orchestrator 拼的
    那条用户消息。``image_paths`` 同理交给控制面：开局预置要先把图看掉再拆意图
    （见 ``HarnessAgentAdapter._prefill``）。

    基座档由 ``MAIN_LOOP_TIER_BASE`` 决定（默认 fast，关思考）：第 2 轮起决策空间已被阶段机与
    候选 id 化夹死，thinking token 买不到东西。第 1 轮是全链路唯一没被机制锁死的决策（购物还是
    闲聊、先拆解还是先查品类），由 ``MAIN_LOOP_TIER_FIRST`` 单独加档，
    落点在 ``HarnessAgentAdapter.on_model_call``。
    """
    return await _assemble(
        name="shoppingx",
        role="main",
        max_iters=MAIN_MAX_ITERS,
        original_query=original_query,
        image_paths=image_paths,
        state=state,
        tier=main_loop_tier_base(),
    )
