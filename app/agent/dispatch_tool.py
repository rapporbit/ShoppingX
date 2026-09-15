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

from app.agent.limits import SUB_AGENT_TIMEOUT_SEC
from app.agent.platform_scope import get_enabled_platforms
from app.api import monitor
from app.api.context import get_session_dir, get_user_id
from app.harness.budgets import get_fork_semaphore
from app.harness.fork_guard import ForkLimitExceeded, enter_fork
from app.harness.truncation import truncate_tool_result
from app.memory.injector import PREF_EMPTY, build_preference_block
from app.tools._bundle import detect_slot, ensure_dispatch_slot, slot_scope
from app.utils.clean import PLATFORMS
from app.utils.thread_ctx import thread_scope

# 子 Agent 防失控参数之①（超时）。定义与其余三层一起在 ``app.agent.limits``；本文件曾另存
# 一份 ``SUB_AGENT_MAX_ITERATIONS = 6`` 的字面量，零引用却让人以为「改这里就生效」，已删。


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

    只保得住「不派用户没勾的平台」（单条可判），保不住「模型少派了一个平台」——后者需要批次
    视角，随「一次一批」改「一条一次」一起丢了。这是一处**已知的能力回退**，代价、理由与未补
    的待办见 docs/decisions/0002-派发从批量改单条的能力回退.md。
    """
    target = _detect_platform(demands)
    if target is None or target in get_enabled_platforms():
        return None
    return (
        f"[task_dispatch 拒绝] 用户本轮未启用 {target} 平台，该子任务不派。"
        f"可派的平台：{' / '.join(get_enabled_platforms())}。"
    )


async def _buyer_preferences() -> str:
    """在**父上下文**取本轮域内的长期偏好，渲染成 ``<buyer-preferences>`` 块。

    **偏好由服务端注入，不由子 Agent 自己去读 Store**：worker 拿到的应该是一份已经按买家、按
    本轮品类域裁好的事实，而不是一个「你自己去查」的授权——后者等于把偏好读取这件事的正确性
    押在模型愿不愿意调、调得对不对上。

    在父上下文取有两个原因：① 域（``session_domains``）按 session_dir 聚合，父子共享，但父这边
    是 planner 判完域之后的确定态；② 取偏好是纯读，放在派发前做不占子任务的超时预算。

    """
    user_id = get_user_id() or ""
    if not user_id:
        return ""
    block = await build_preference_block(user_id)
    if not block or block == PREF_EMPTY:
        return ""
    return (
        "<buyer-preferences>\n"
        f"{block}\n"
        "</buyer-preferences>\n"
        "以上是该买家与本轮品类相关的长期偏好，**系统已在检索与打分里自动并入**（见 "
        "memory.assemble）。它在这里只为一件事：让你判断召回是否跑题时有依据。**不要**再把它们\n"
        "转述进任何工具参数——重复一遍不会让它们更生效，只会替用户做他没授权的决定。\n\n"
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
    prefs = await _buyer_preferences()
    try:
        # fork 前捕获父会话目录，worker 继承同一目录（产物归同一会话）。
        parent_session_dir = get_session_dir()
        with enter_fork() as depth:
            sub_thread_id = f"{kind}-{uuid4().hex[:8]}-d{depth}"
            # fork 事件在进入子 thread_scope 之前上报：此刻 ContextVar 仍是父 thread_id，
            # 事件路由到父任务的前端连接，用户能看见「派出去了一个子任务」。
            await monitor.report_fork(sub_thread_id, demands)
            from app.agent.agents import build_worker_agent  # 延迟导入，破模块级循环

            # 槽位打标：解析出标记后**先把槽落实**（planner 没拆槽时兜底登记，见
            # ensure_dispatch_slot），再把稳定 id 传进子作用域——传名字的话，兜底那条路上
            # item_search 盖章时槽表还是空的，章照样盖不上。
            slot = ensure_dispatch_slot(detect_slot(demands) or "")
            scope: Any = (
                thread_scope(sub_thread_id, parent_session_dir)
                if parent_session_dir is not None
                else nullcontext()
            )
            fork_sem = get_fork_semaphore()
            sem_ctx: Any = fork_sem if fork_sem is not None else nullcontext()
            slot_ctx: Any = slot_scope(slot) if slot else nullcontext()
            with scope, slot_ctx:
                async with sem_ctx:
                    # Agent 在子 scope **内**建：它自带的 HarnessSession / Toolkit 都是 per-loop
                    # 的，建在外面会让 worker 与主 loop 共用控制面状态（断言、循环检测全串味）。
                    agent = await build_worker_agent(kind)
                    # 「执行方通则」批 1 起住在 worker 自己的 system prompt 里（``sub_agents.*``
                    # 段），user 消息只剩「这一条子任务 + 服务端注入的买家偏好」——system 段因此
                    # 跨调用逐字稳定，同类 worker 共用一条缓存前缀。
                    msg = Msg(
                        name="user",
                        role="user",
                        content=[TextBlock(type="text", text=prefs + demands)],
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
    subagent_type: Literal["search"],
) -> ToolChunk:
    """把子任务派给专职子 Agent（看不到你的上下文），回传其最终回复；同轮多个独立子任务一次性多发即并行。
    参数：demands 写全预算/品类/硬约束/软偏好/目标平台（跨平台检索一条只写一个平台）；
    subagent_type：search=只读检索（只有 item_search / web_search，无下单工具）。
    下单 / 查单 / 取消不派发，你自己调交易工具。
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
