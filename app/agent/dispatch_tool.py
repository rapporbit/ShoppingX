"""派发入口（元工具）+ 派发安全四层之②：子 Agent 超时 + 迭代上限。

``task_dispatch(demands, subagent_type)`` 对主 Agent 而言就是「另一个工具」：传需求、拿最终
回复。内部起一个 worker（专职工具集 + 专职 prompt 段，见 ``agents.build_worker_agent``），
独立 thread_id、继承父 session_dir，只回传截断后的最终文本（上下文隔离）。

**没有 parallel 版**：工具标了 ``is_concurrency_safe=True``，主 Agent 同一轮发多个 tool_call
就由框架并发执行，不需要一个「入参是列表」的元工具替它排并发。代价是丢了「批次视角」——见
:func:`_platform_guard` 对平台补齐那条机制的诚实交代。

安全四层在这里收口：① 深度（``enter_fork``，且 worker 的工具集里根本没有 task_dispatch）
② 超时（``wait_for``）+ 迭代上限（``ReActConfig.max_iters``）③ 结果截断 ④ 循环检测。
**任何异常都转成字符串回传**，让主 Agent 把「子任务失败」当普通工具结果处理，而不是整个 loop 崩。
"""

import asyncio
from contextlib import nullcontext
from typing import Any, Literal
from uuid import uuid4

from agentscope.message import Msg, TextBlock
from agentscope.tool._response import ToolChunk, ToolResultState

from app.agent.fork_guard import ForkLimitExceeded, enter_fork
from app.agent.platform_scope import get_enabled_platforms
from app.agent.prompts import get_sub_agent_brief
from app.agent.retrieval_budget import isolated_retrieval_scope
from app.api import monitor
from app.api.context import get_session_dir
from app.harness.budgets import get_fork_semaphore
from app.harness.truncation import truncate_tool_result
from app.tools._bundle import detect_slot, slot_scope
from app.utils.clean import PLATFORMS
from app.utils.thread_ctx import thread_scope

# 子 Agent 防失控参数。
SUB_AGENT_TIMEOUT_SEC = 90
# 单平台检索子任务：category_insight 校准 + 几次自我纠偏 item_search 就够收敛，6 轮留足余量。
# 比早先的 12 紧一半——越界子（跑完整购物流程的）会更早撞上限被掐，也压住内部刷检索。
SUB_AGENT_MAX_ITERATIONS = 6


def _detect_platform(demand: str) -> str | None:
    """从一条 demands 文本里识别它针对的平台（命中 PLATFORMS 里的平台名，小写匹配）。"""
    low = demand.lower()
    for p in PLATFORMS:
        if p in low:
            return p
    return None


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


def _platform_guard(demands: str) -> str | None:
    """单条 demand 的平台收口：指名了**未启用**平台就拒派，返回拒绝文案。

    被取代的 ``parallel_dispatch_tool``（入参是一批 demands）在这里还做了另一半——**补齐**
    模型漏派的启用平台（一次拿到整批，才知道少了谁）。拆成一条一次派发后，那个批次视角没有了：
    本函数只保得住
    「不派用户没勾的平台」（单条可判），保不住「模型少派了一个平台」。
    这是 L3 的一处**能力回退**，不藏着：
    - 影响面小——线上默认单平台（amazon），语料 99.75% 也在 amazon，补齐几乎不触发；
    - 动机侧仍在——``<enabled_platforms>`` 块每轮列出该派哪些平台（见 orchestrator）；
    - 机制侧的补法留给批 1：post_reflect 里数「启用 n 个平台、本轮只派了 k 条」，不足就催一轮。
    """
    target = _detect_platform(demands)
    if target is None or target in get_enabled_platforms():
        return None
    return (
        f"[task_dispatch 拒绝] 用户本轮未启用 {target} 平台，该子任务不派。"
        f"可派的平台：{' / '.join(get_enabled_platforms())}。"
    )


async def _run_worker(demands: str, kind: str) -> str:
    """派一个 worker 执行 demands，回传截断后的最终文本；任何失败都转字符串。

    安全四层在这里收口，与旧版逐条对齐：① 深度（``enter_fork``）② 超时（``wait_for``）+
    迭代上限（``ReActConfig.max_iters``，见 agents.py）③ 结果截断 ④ 循环检测（worker 自己那份
    ``HarnessSession`` 里的 LoopDetector）。**任何异常都转成字符串回传**，让主 Agent 把「子任务
    失败」当普通工具结果处理，而不是整个 loop 崩。
    """
    rejected = _platform_guard(demands)
    if rejected is not None:
        return rejected
    try:
        # fork 前捕获父会话目录，worker 继承同一目录（产物归同一会话）。
        parent_session_dir = get_session_dir()
        with enter_fork() as depth:
            sub_thread_id = f"{kind}-{uuid4().hex[:8]}-d{depth}"
            # fork 事件在进入子 thread_scope 之前上报：此刻 ContextVar 仍是父 thread_id，
            # 事件路由到父任务的前端连接，用户能看见「派出去了一个子任务」。
            await monitor.report_fork(sub_thread_id, demands)
            from app.agent.agents import build_worker_agent  # 延迟导入，破模块级循环

            # 收敛信号是否跨子任务共享，仍按「这条 demand 提没提平台名」自动识别（口径与旧版
            # 的批次判定等价，只是粒度从批降到条）：跨平台泛搜的兄弟之间该共享「已经找到货了
            # 就别再找」，定点调查 / 套装槽位则各查各的，不许一个搜到就让别人收手。
            isolated = _detect_platform(demands) is None
            slot = detect_slot(demands)
            scope: Any = (
                thread_scope(sub_thread_id, parent_session_dir)
                if parent_session_dir is not None
                else nullcontext()
            )
            fork_sem = get_fork_semaphore()
            sem_ctx: Any = fork_sem if fork_sem is not None else nullcontext()
            isolate_ctx: Any = isolated_retrieval_scope() if isolated else nullcontext()
            slot_ctx: Any = slot_scope(slot) if slot else nullcontext()
            with scope, isolate_ctx, slot_ctx:
                async with sem_ctx:
                    # Agent 在子 scope **内**建：它自带的 HarnessSession / Toolkit 都是 per-loop
                    # 的，建在外面会让 worker 与主 loop 共用控制面状态（断言、循环检测全串味）。
                    agent = await build_worker_agent(kind)
                    msg = Msg(
                        name="user",
                        role="user",
                        content=[
                            TextBlock(type="text", text=get_sub_agent_brief() + demands),
                        ],
                    )
                    # 用 reply 而非 reply_stream：worker 的 thread 没有前端连接，事件转发出去
                    # 也无人接收（上下文隔离本就是它的目的）。排队等 fork 槽的时间不计入超时。
                    final = await asyncio.wait_for(agent.reply(msg), timeout=SUB_AGENT_TIMEOUT_SEC)
            if slot:
                # 套装槽位批：候选已由 item_search 打标入库，worker 的叙述文本对主 Agent 是冗余
                # 的、还常泄漏内心独白（badcase 4c0ac682），改用登记表生成确定性摘要。该槽一件
                # 都没入库时保留原文——失败说明有信息量。
                digest = _slot_digest(slot)
                if digest is not None:
                    return digest
            return truncate_tool_result(final.get_text_content() or "")
    except ForkLimitExceeded as e:
        return f"[task_dispatch 拒绝] {e}"
    except TimeoutError:
        return f"[task_dispatch 超时] 子任务超过 {SUB_AGENT_TIMEOUT_SEC}s 未完成，请拆小再试"
    except asyncio.CancelledError:
        raise  # 用户取消要一路传上去，不能被下面的兜底吞成一条「工具出错」
    except Exception as e:  # 兜底：子任务任何异常都转字符串，不让主 loop 崩
        return f"[task_dispatch 错误] {type(e).__name__}: {e}"


async def task_dispatch(
    demands: str,
    subagent_type: Literal["search", "trade"] = "search",
) -> ToolChunk:
    """把一个子任务派给专职的子 Agent 执行，返回它的最终回复。

    何时调用（三件事判断，满足**任一**即可）：
      1. 能并行：多个独立子任务可同时跑（如在多个平台同时检索）。
      2. 要隔离：子任务会产出大量中间数据，不该污染你的上下文。
      3. 链够深：子任务自己内部还需要 ≥3 层工具调用。
    都不满足就**自己直接调工具**——你手上有全部业务工具，小事派一趟只是白花一轮往返。

    同一轮要派多个独立子任务时，**在这一轮里一次性发多个 task_dispatch 调用**（它们会被并行
    执行），不要一轮派一个串着等。

    参数：
      - demands：交给子 Agent 的完整需求描述。它看不到你的上下文，所以预算 / 品类 / 硬约束 /
        软偏好 / 目标平台都要在这段文字里写全。跨平台检索时**一条只写一个平台**。
      - subagent_type：派给哪种子 Agent。``search`` = 只读检索（商品检索 / 品类调研 / 比价 /
        运费）；``trade`` = 交易操作（下单 / 查单 / 取消，需在 demands 里给定 item_id）。
    """
    text = await _run_worker(demands, subagent_type)
    # 派发失败已在 _run_worker 里转成 "[task_dispatch …]" 文案。判 ERROR 状态而不是只回文本：
    # harness 的 result_nudges 与循环检测靠 state 分辨「这一趟到底有没有拿到东西」。
    failed = text.startswith("[task_dispatch ")
    return ToolChunk(
        content=[TextBlock(type="text", text=text)],
        state=ToolResultState.ERROR if failed else ToolResultState.SUCCESS,
        metadata={"tool": "task_dispatch", "subagent_type": subagent_type},
    )
