"""fork 入口（元工具）+ 安全四层之②：子 Agent 超时 + 迭代上限。

``dispatch_tool(demands)`` 对主 loop 而言就是「另一个工具」：传需求、拿最终回复。
内部 spawn 一个子 AgentLoop（同 FULL_TOOL_SET + 同 system prompt + 同中间件 = **能力同质**），
独立 thread_id、继承父 session_dir，只回传精简的最终消息（上下文隔离）。
模型档位上子用 ``get_fast_llm``（非推理快模型，见 llm.py 的 perf/model-tiering 说明）——
有意识地破「主子模型同质」以砍解码延迟；未配 ``LLM_FAST`` 时回退主模型、退回全同质。

同质 + 递归如何不循环依赖：工具集通过 ``tools_provider`` 闭包**延迟获取**——provider
返回 live 的 FULL_TOOL_SET（含 dispatch_tool 自身），既保证子 Agent 是主 Agent 的完整
克隆、又支持子再 fork 孙，且 dispatch_tool.py 不在模块层 import tool_registry。

安全四层在这里收口：① 深度（enter_fork）② 超时（wait_for）+ 迭代上限（recursion_limit）
③ 结果截断（truncate_tool_result）。**任何异常都转成字符串回传**，让主 loop 把「子任务
失败」当普通工具结果处理，而不是整个 Agent 崩。
"""

import asyncio
from collections.abc import Callable
from contextlib import nullcontext
from typing import Any
from uuid import uuid4

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, tool
from langgraph.errors import GraphRecursionError

from app.agent.fork_guard import ForkLimitExceeded, enter_fork
from app.agent.llm import get_fast_llm
from app.agent.platform_scope import get_enabled_platforms
from app.agent.prompts import get_sub_agent_brief, get_system_prompt
from app.agent.retrieval_budget import isolated_retrieval_scope
from app.agent.tracing import apply_tracing
from app.api import monitor
from app.api.context import get_session_dir
from app.harness.agent_middleware import build_agent_middleware
from app.harness.budgets import get_fork_semaphore
from app.harness.truncation import truncate_tool_result
from app.tools._args import StrListArg
from app.tools._bundle import detect_slot, slot_scope
from app.utils.clean import PLATFORMS
from app.utils.thread_ctx import thread_scope

# 子 Agent 防失控参数。
SUB_AGENT_TIMEOUT_SEC = 90
# 单平台检索子任务：category_insight 校准 + 几次自我纠偏 item_search 就够收敛，6 轮留足余量。
# 比早先的 12 紧一半——越界子（跑完整购物流程的）会更早撞上限被掐，也压住内部刷检索。
SUB_AGENT_MAX_ITERATIONS = 6
# langgraph 按「超步」计数，一轮 Think→Act 约 2 步，故上限 ≈ 2×迭代 + 1。
SUB_AGENT_RECURSION_LIMIT = SUB_AGENT_MAX_ITERATIONS * 2 + 1

ToolsProvider = Callable[[], list[BaseTool]]


def _detect_platform(demand: str) -> str | None:
    """从一条 demands 文本里识别它针对的平台（命中 PLATFORMS 里的平台名，小写匹配）。"""
    low = demand.lower()
    for p in PLATFORMS:
        if p in low:
            return p
    return None


def _ensure_platform_coverage(demands_list: list[str]) -> list[str]:
    """跨平台并行检索时，机制兜底「不多不少，正好覆盖**本轮启用的**平台」——模型既会少列平台、也会
    照着 prompt 惯性去派用户根本没勾的平台，两头都靠机制收口而非靠模型自觉
    （见 fork-guardrails-mechanism-not-prompt）。

    口径：
    - 仅当这批 demands **是平台检索**（至少一条提到某平台名）时才处理；否则原样返回（定点调查那批
      不写平台名，正是靠这点识别，别强塞平台）。
    - **未启用平台的 demand 直接丢弃**：用户在设置里没勾它，派出去也是空军——这正是「只 fork 有货
      的平台」那一刀（语料 99.75% 是 amazon，默认单平台时 5 派 4 空）。
    - 启用且模型已列的平台：原样保留它写的 demand（含它针对该平台的措辞）。
    - 启用但模型漏了的平台：克隆一条已覆盖的 demand 作模板（字段平台无关：同预算/关键词/硬约束/
      软偏好），前缀一条强制平台指令——子据此 item_search(platform=该平台)。
    - 若丢完一条不剩（模型列的平台全没启用），退化成一条「只搜第一个启用平台」的 demand，不至于把
      整批检索派空。
    """
    detected = [(_detect_platform(d), d) for d in demands_list]
    if not any(p for p, _ in detected):
        # 这批不是平台检索（没有任何平台名）——原样放行。
        return demands_list
    enabled = get_enabled_platforms()
    template = next(d for p, d in detected if p is not None)
    kept = [(p, d) for p, d in detected if p is None or p in enabled]
    covered = {p for p, _ in kept if p is not None}
    out = [d for _, d in kept]
    for p in enabled:
        if p not in covered:
            out.append(
                f"只在 {p} 这一个平台检索（下文模板里如提到其它平台名一律忽略，只搜 {p}）。"
                f"\n{template}"
            )
    return out


def _slot_digest(slot: str) -> str | None:
    """从会话候选登记表给一个套装槽位生成确定性检索摘要（子 Agent 文本的替身，零 LLM）。

    在父上下文调用（``thread_scope`` 已退出，登记表按 session_dir 聚合、父子共享）。
    该槽没有任何入库候选时返回 None——调用方保留子 Agent 原文（失败/空召回的说明有信息量）。
    """
    from app.tools._bundle import resolve_slot  # 延迟导入，同 create_agent 的循环兜法
    from app.tools._candidates import registry_snapshot

    # 入参是槽引用（detect_slot 给的 id，或「套装槽位：X」标记的原文）；盖章是稳定 id——
    # 解析成同一副身份再过滤，原文引用也对得上章。解析不出就按原文匹配（旧数据名字章）。
    s = resolve_slot(slot)
    sid, disp = (s.id, s.name) if s is not None else (slot, slot)
    cands = [c for c in registry_snapshot() if c.slot in (sid, slot)]
    if not cands:
        return None
    prices = sorted(p for p in (c.price_usd for c in cands) if p is not None)
    head = f"[套装槽位「{disp}」检索完成] 召回 {len(cands)} 件候选，已入库（比价 / 精挑直接可用）"
    if prices:
        head += f"，价格区间 ${prices[0]:.2f}–${prices[-1]:.2f}"
    top = sorted(cands, key=lambda c: c.score, reverse=True)[:3]
    lines = [head] + [
        f"· {c.title[:70]}"
        + (f"（${c.price_usd:.2f}" if c.price_usd is not None else "（价格未知")
        + (f"，评分 {c.rating}）" if c.rating is not None else "）")
        for c in top
    ]
    return "\n".join(lines)


async def _run_sub_agent(
    demands: str,
    tools_provider: ToolsProvider,
    system_prompt: str | None = None,
    *,
    isolated_retrieval: bool = False,
    slot: str | None = None,
) -> str:
    """fork 一个同质子 Agent 执行 demands，回传截断后的最终消息；任何失败转字符串。

    ``system_prompt`` 默认取全局 system prompt（与主 loop 同质）；示例/测试可注入与各自主
    Agent 一致的 prompt，以保持「子 = 主的克隆」这一同质 fork 约束。

    ``isolated_retrieval``：为真时子 Agent 的 web_search 门控只看**自己**的召回结果，不受
    全树其它子任务（兄弟平台 / 兄弟商品）是否已找到候选影响（见
    :func:`app.agent.retrieval_budget.isolated_retrieval_scope`）。单个独立子任务
    （``dispatch_tool``，或 ``parallel_dispatch_tool`` 里识别出的点名商品定点调查批次）传
    True；跨平台泛搜批次（子任务间本就该共享「已有候选就别再找」的收敛信号）保持 False。
    """
    try:
        # 在 fork 前捕获父会话目录，子 Agent 继承同一目录。放在 try 内确保任何意外都被兜底。
        parent_session_dir = get_session_dir()
        with enter_fork() as depth:
            sub_thread_id = f"sub-{uuid4().hex[:8]}-d{depth}"
            # fork 事件在进入子 thread_scope 之前上报：此刻 ContextVar 仍是父 thread_id，
            # 事件路由到父任务的前端连接，用户能看见「分叉出一个子任务」。
            await monitor.report_fork(sub_thread_id, demands)
            # 延迟导入，避免与 tool_registry 的模块级循环依赖。
            from langchain.agents import create_agent

            sub_agent = create_agent(
                # 模型分层（perf/model-tiering）：子用非推理快模型砍解码延迟。子在 #2/#2.5 后已塌成
                # 1-2 跳 item_search、不需深推理。换的是「模型档位」，工具集 + system prompt 仍与主
                # 一致（能力同质），只破「主子模型同质」；未配 LLM_FAST 回退 LLM_MAIN（全同质）。
                model=get_fast_llm(),
                tools=tools_provider(),
                system_prompt=system_prompt or get_system_prompt(),
                # 子 loop 挂与主 loop 同一套中间件（压缩 + 逐工具截断 + 循环检测 + 检索收敛）。
                # 每次 fork 新建一份（LoopDetector / per-sub 检索计数有状态，子任务各自独立计数）。
                middleware=build_agent_middleware(),
            )
            # 覆盖子 thread_id（事件/checkpoint 隔离），但继承父 session_dir（产物归同一会话）。
            scope = (
                thread_scope(sub_thread_id, parent_session_dir)
                if parent_session_dir is not None
                else nullcontext()
            )
            config: RunnableConfig = {"recursion_limit": SUB_AGENT_RECURSION_LIMIT}
            # 子 loop 复用父在 ContextVar 里的 trace_id（root=False，默认）：子 Agent 的独立 ainvoke
            # 归并进父的同一条 trace，而非另起一条顶层 trace。不设 session/user（那是 root 的事）。
            apply_tracing(config)
            # 元组消息形式 ("user", demands) 运行时由 LangChain 归一，类型上以 Any 放行。
            # slim variant 下，子 Agent 的「执行方通则」在这里拼进 demands 前缀，而不是塞进主 / 子
            # 共享的 system prompt——主 loop 因此不必为一段它永远用不上的协议付 token。full variant
            # 下 get_sub_agent_brief() 返回 ""（那段仍在 system prompt 里），行为完全不变。
            payload: Any = {"messages": [("user", get_sub_agent_brief() + demands)]}
            # fork 级并发闸（C 块）：跨平台并行补齐到 5 平台时，用它限同时在跑的子 Agent 数、
            # 给下游（embedding/rerank/LLM）背压。排队等槽的时间**不计入**子任务超时——超时只
            # 衡量真正的执行时长，故 async with 包在 wait_for 外层。未开作用域时为 None → 不限。
            fork_sem = get_fork_semaphore()
            sem_ctx: Any = fork_sem if fork_sem is not None else nullcontext()
            isolate_ctx: Any = isolated_retrieval_scope() if isolated_retrieval else nullcontext()
            # 套装槽位批：把槽名设进 ContextVar，子 loop 内的 item_search（task 继承快照）据此
            # 给候选盖章——机制打标，不依赖子 Agent 记得转述 slot 参数（见 app.tools._bundle）。
            slot_ctx: Any = slot_scope(slot) if slot else nullcontext()
            with scope, isolate_ctx, slot_ctx:
                async with sem_ctx:
                    result: dict[str, Any] = await asyncio.wait_for(
                        sub_agent.ainvoke(payload, config=config),
                        timeout=SUB_AGENT_TIMEOUT_SEC,
                    )
            # 套装槽位批：子 Agent 的叙述文本对主 loop 是冗余的（候选已经由 item_search 打标
            # 入库，比价/精挑直接从登记表拿），还常泄漏内心独白（badcase 4c0ac682：
            # "The system says I've used my 1 search…"）。改用登记表生成**确定性摘要**（零
            # LLM、零泄漏、省主 loop token）；该槽一件候选都没入库时保留原文——失败说明有
            # 信息量，不能抹。非槽位派发（定点调查等，文本才是主产物）不走这里。
            if slot:
                digest = _slot_digest(slot)
                if digest is not None:
                    return digest
            return truncate_tool_result(str(result["messages"][-1].content))
    except ForkLimitExceeded as e:
        return f"[dispatch_tool 拒绝] {e}"
    except TimeoutError:
        return f"[dispatch_tool 超时] 子任务超过 {SUB_AGENT_TIMEOUT_SEC}s 未完成，请拆小再试"
    except GraphRecursionError:
        return "[dispatch_tool 拒绝] 子任务迭代超限（疑似打转），请缩小范围或换思路"
    except Exception as e:  # 兜底：子任务任何异常都转字符串，不让主 loop 崩
        return f"[dispatch_tool 错误] {type(e).__name__}: {e}"


def make_dispatch_tools(
    tools_provider: ToolsProvider, system_prompt: str | None = None
) -> tuple[BaseTool, BaseTool]:
    """绑定工具集 provider（+ 可选 system_prompt），返回两个元工具。

    provider 在每次调用时求值，应返回主/子共用的完整工具集（含返回的这两个元工具本身），
    从而保证「同质 fork」并支持递归。``system_prompt`` 应与对应主 Agent 的一致（默认全局
    system prompt）。app 用 tool_registry 的 FULL_TOOL_SET，示例可注入玩具集 + 玩具 prompt。
    """

    @tool
    async def dispatch_tool(demands: str) -> str:
        """派一个同质子 AgentLoop 去执行 demands，返回它的最终回复。

        何时调用（fork 三件事，满足任一即可）：
          1. 能并行：多个独立子任务可同时跑（如跨多个平台同时检索）。
          2. 要隔离：子任务输出很大，不该污染主 loop 上下文（如爬十几页条款做对比）。
          3. 链够深：子任务自己内部还需 ≥3 层工具调用。
        都不满足就自己处理，不要为 fork 而 fork。
        """
        return await _run_sub_agent(
            demands,
            tools_provider,
            system_prompt,
            isolated_retrieval=True,
            # 单条派发也可能是套装的一个槽（用户确认组成后补搜新槽），照样机制打标。
            slot=detect_slot(demands),
        )

    @tool
    async def parallel_dispatch_tool(demands_list: StrListArg) -> str:
        """并行派发多个子任务给独立的同质子 AgentLoop，合并回传各自最终回复。

        何时调用：多个**彼此独立**的子任务可同时跑。两种典型场景，写法见对应协议块：
          1. 跨平台泛搜：demands_list 每项是一条平台子目标（<fork_protocol>）。系统会
             **机制兜底补齐全部 5 个平台**（模型少列的自动补一条），不必担心漏平台。
          2. 点名商品定点调查：demands_list 每项是一件被点名商品的调查任务
             （<target_protocol>）——哪怕只列 1 件也可以调，不必凑够多件才用。
          3. 套装（一套齐）槽位检索：demands_list 一槽一条，每条**开头写「套装槽位：<槽名>」**、
             不写平台名——系统按这个标记给该槽召回的候选打标，供跨槽组合优选。
        场景由系统按 demands 文本**自动识别**（槽位标记 / 平台名），并分别处理检索收敛信号
        是否跨子任务共享，不必自己判断隔不隔离。
        """
        # 机制识别场景（不靠模型自报），优先级：套装槽位批 > 跨平台泛搜 > 定点调查。
        # ① 任一条 demands 解析得出槽位（「套装槽位：X」标记 / 已登记槽名）→ 套装批：一槽一条、
        #    平台无关——**跳过平台补齐**（那会把一条床品 demand 克隆成 5 个平台），收敛信号各槽
        #    独立（床品搜到货不该让台灯槽收手），并把槽名经 slot_scope 传给子里的 item_search。
        # ② 否则任一条提到平台名 → 跨平台泛搜：补齐启用平台，收敛信号全树共享。
        # ③ 都不是 → 定点调查批：各自独立（见 target-refs-serial-dispatch-fork-done 的教训）。
        slots_detected = [detect_slot(d) for d in demands_list]
        if any(slots_detected):
            pairs = list(zip(demands_list, slots_detected, strict=True))
            isolated = True
        else:
            is_platform_search = any(_detect_platform(d) is not None for d in demands_list)
            demands_list = _ensure_platform_coverage(demands_list)
            pairs = [(d, None) for d in demands_list]
            isolated = not is_platform_search
        # return_exceptions=True：一个子任务抛异常（罕见，_run_sub_agent 已兜大部分）不连累其余——
        # gather 默认会把首个异常抛出并丢弃其它结果，这里改成各子独立降级，超时/出错的子返回降级串。
        results = await asyncio.gather(
            *(
                _run_sub_agent(
                    d, tools_provider, system_prompt, isolated_retrieval=isolated, slot=s
                )
                for d, s in pairs
            ),
            return_exceptions=True,
        )
        parts = [
            r if isinstance(r, str) else f"[dispatch_tool 错误] {type(r).__name__}: {r}"
            for r in results
        ]
        return "\n---\n".join(parts)

    return dispatch_tool, parallel_dispatch_tool
