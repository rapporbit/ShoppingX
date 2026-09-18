"""开局预置：planner（有图时连同 image_understand）在第 1 次模型调用之前就跑掉。

从 ``HarnessAgentAdapter`` 抽出来的一段——它是「机制替模型做的一次工具调用」，不是 AgentScope
中间件桥接的一部分。适配器的 ``on_reply`` 开头调 :func:`prefill`；产物直接写进 ``state.context``。
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import TYPE_CHECKING, Any

from agentscope.message import Msg
from pydantic import BaseModel

from app.harness.middleware import harness
from app.harness.msgs import tool_blocks
from app.harness.signals import _summarize_call

if TYPE_CHECKING:
    from agentscope.agent import Agent

    from app.harness.session import HarnessSession

logger = logging.getLogger("shoppingx.harness.prefill")

# ── 开局预置：planner（与参考图）在第 1 次模型调用之前就跑掉 ──

# 一次任务最多看几张图：每张都是一次 VL 往返 + 一段上下文，传一堆图既烧预算又稀释意图。
MAX_PREFILL_IMAGES = 3

#: 套装轮预注入的 skill 名（``skills/bundle-planning/SKILL.md`` 的 frontmatter name）。
BUNDLE_SKILL_NAME = "bundle-planning"


async def prefill(session: HarnessSession, agent: Agent) -> None:
    """把 planner（有图时连同 image_understand）预先跑掉，结果写进 ``state.context``。

    **省掉的是一次纯仪式性的模型往返**：第 1 轮模型面对的问题本来是「我该调什么工具」，而
    这个答案不需要模型给——planner 判 ``retrieval``（reuse / augment / search）靠的是系统
    确定性注入的会话状态 + 候选登记表，它**不消费主 loop 模型的任何输出**，入参只有用户原话。
    预置之后模型第 1 轮面对的是「plan 已在手，我该怎么检索」，那才是真需要推理的一步。

    **为什么在 loop 内（中间件）而不是在 orchestrator 里手工预跑**：域内长期偏好注入
    （``hooks/context_shaping``）与阶段机 PLANNING→SEARCHING（``hooks/progress``
    读 ``planner_output_ready``）都挂在「planner 在 loop 内被调过」这个事实上。走这里、照常
    跑 ``post_tool_call`` 与阶段信号，它们一行都不用复刻。

    **有图时看图必须先于 planner**：planner 是拿用户原话拆结构化字段的，若图的结论晚于它
    产出，「只发一张图 + 想买这个」这类 query 会让 planner 拆出一片空白，后面全链路空转。

    **跳过 pre_tool_call 是有意的**：那一层的闸（白名单 / 阶段门 / 熔断 / 循环检测 / 检索
    预算）管的是模型的自由发挥，而这次调用是机制自己决定的——让它去过一道为约束模型而设的
    闸，只会平添「机制被自己的护栏拦下」这种荒诞失败。

    降级：planner 抛错则回到老路（模型自己决定调 planner，prompt 里那条规则仍在）。预置是快路径
    不是唯一路径。
    """
    s = session
    if s.prefilled or not s.original_query:
        return
    s.prefilled = True

    blocks: list[Any] = []
    intent = s.original_query
    if s.image_paths:
        vision_blocks, hint = await _prefill_vision(session)
        blocks.extend(vision_blocks)
        if hint:
            intent = f"{intent}\n\n[用户上传的参考图，已识别] {hint}"

    from app.tools.planner import planner as planner_tool  # 懒 import：防注册期导入环

    args = {"intent": intent}
    call_id = "prefill_planner"
    try:
        out = await planner_tool.ainvoke(args)
    except Exception:
        logger.warning("planner 预置失败，回退为模型自行调用（老路径）", exc_info=True)
        append_prefilled(agent, blocks)  # 图的结论已经拿到了，别连它一起丢
        return
    text = out.model_dump_json() if isinstance(out, BaseModel) else str(out)

    # 阶段信号与行为摘要：与工具适配器里真调一次 planner 记的东西完全一致——第 1 轮
    # post_reflect 据 planner_output_ready 把阶段从 PLANNING 推到 SEARCHING。
    s.planner_done = True
    s.called_tools.add("planner")
    s.recent_actions.append(_summarize_call("planner", args))

    ctx = s.base_context()
    ctx["tool_name"] = "planner"
    ctx["tool_args"] = args
    ctx["tool_result"] = text
    # round3 刀 4：planner 的 post_tool_call（域内长期偏好读取 + 注入，走 DB）与品类知识库预取
    # （OpenSearch 两段式检索）互不依赖，并发跑；KB 预取的结果作为第二对 tool 块预置进上下文，
    # 模型第 1 轮就拿着 plan + 品类行情直接检索（改前 9/9 遍第 1 轮都在调 category_insight）。
    kb_task = asyncio.create_task(_prefetch_kb(s, out)) if _kb_prefetch_due(out) else None
    # 套装 skill 同理预载：信号（槽位表）与 KB 预取来自同一次 planner，两件事互不依赖，一起跑。
    skill_task = (
        asyncio.create_task(_prefetch_skill(BUNDLE_SKILL_NAME)) if _skill_prefetch_due(out) else None
    )
    ctx = await harness.run("post_tool_call", ctx)
    # 偏好注入落 pending_inject，由下一次 on_model_call 开头消费——那正是第 1 轮。
    s.collect(ctx)
    guarded = ctx.get("tool_result")
    if isinstance(guarded, str) and guarded:
        text = guarded

    blocks.extend(tool_blocks(call_id, "planner", args, text))
    if kb_task is not None:
        blocks.extend(await kb_task)
    if skill_task is not None:
        blocks.extend(await skill_task)
    append_prefilled(agent, blocks)


def _kb_prefetch_due(plan: Any) -> bool:
    """要不要预取品类知识库：开关开 + planner 判出品类 + 本轮有购物类任务（纯交易 / 闲聊不取）。"""
    if os.getenv("KB_PREFETCH", "1").strip().lower() in {"0", "false", "off"}:
        return False
    category = str(getattr(plan, "category", "") or "").strip()
    tasks = set(getattr(plan, "tasks", None) or [])
    return bool(category) and bool(tasks & {"recommend", "evaluate", "category_intel"})


async def _prefetch_kb(s: HarnessSession, plan: Any) -> list[Any]:
    """按 planner 的品类预取 category_insight（quick），走与真实调用同一条成功后管线。失败即空。"""
    from app.harness.adapter import after_tool_success
    from app.tools._shell import _to_text
    from app.tools.category_insight import category_insight

    args = {"category": str(plan.category).strip(), "depth": "quick"}
    try:
        out = await category_insight.ainvoke(args)
        text = await after_tool_success(s, "category_insight", args, _to_text(out))
    except Exception:
        logger.warning("品类知识库预取失败，交回模型自行决定是否调 category_insight", exc_info=True)
        return []
    return tool_blocks("prefill_category_insight", "category_insight", args, text)


def _skill_prefetch_due(plan: Any) -> bool:
    """要不要预注入套装 skill：开关开 + planner 拆出 ≥2 个槽位。

    判据与下游对齐（``autopick._is_bundle_turn`` 与 ``planner._clean_bundle_slots`` 都按
    ``len(bundle_slots) >= 2`` 认槽位轮），不看 ``slot_mode``——它区分的是「一套齐」与「多品类
    并列」两种形态，而两种形态的打法写在同一份 SKILL.md 里，取哪种都要读同一段正文。
    """
    if os.getenv("SKILL_PREFETCH", "1").strip().lower() in {"0", "false", "off"}:
        return False
    return len(getattr(plan, "bundle_slots", None) or []) >= 2


async def _prefetch_skill(name: str) -> list[Any]:
    """把 skill 正文预注入成一次「模型已经调过 ``Skill`` 工具」的工具返回。失败即空。

    Anthropic 的口径是「加载一个 skill 要花掉模型一轮」：模型先读 ``<agent-skills>`` 目录里的
    description，判断这次用得上，再调 ``Skill(skill=…)`` 把正文取进来——判断那一步是纯仪式，
    因为 planner 拆出的槽位表已经确定性地告诉了系统「这轮是套装轮」。同一个信号预加载 skill，
    省掉的正是这次往返（q_backpack 刚从 9 次模型调用压到 4 次，不能再往回加）。

    走 loader 而不是直接读 SKILL.md：``Skill.markdown`` 是框架剥掉 frontmatter 后的正文，与模型
    亲手调一次拿到的字节完全一致。自己读文件会把 frontmatter 也塞进去，预注入版和手调版讲的就
    不是一份东西了。``SKILLS_ENABLED=0`` 时 ``skill_loaders`` 返回空表，这里自然降级为不注入。
    """
    from app.agent.skills import skill_loaders

    try:
        for loader in skill_loaders("main"):
            for skill in await loader.list_skills():
                if getattr(skill, "name", "") != name:
                    continue
                markdown = str(getattr(skill, "markdown", "") or "")
                if not markdown:
                    return []
                return tool_blocks(f"prefill_skill_{name}", "Skill", {"skill": name}, markdown)
    except Exception:
        logger.warning("skill 预注入失败（%s），交回模型自行决定是否调 Skill", name, exc_info=True)
    return []


async def _prefill_vision(session: HarnessSession) -> tuple[list[Any], str]:
    """开局把参考图逐张看掉，返回（要写进上下文的 blocks, 给 planner 的一句话线索）。

    block 形状与真调一次工具逐字同构（tool_call + tool_result），主 loop 因此能像读任何
    工具结果一样读到图的结论；AGUI 事件由 image_understand 内部照常上报，前端看得见「正在
    看图」这一步。看图失败（未配 LLM_VISION / 图读不到 / 模型抽风）不阻断——工具自身已降级
    返回 note，主 loop 照常按文字意图往下走。
    """
    from app.tools.image_understand import image_understand  # 懒 import：防注册期导入环

    s = session
    blocks: list[Any] = []
    hints: list[str] = []
    for idx, name in enumerate(s.image_paths[:MAX_PREFILL_IMAGES]):
        args = {"filename": name}
        try:
            out = await image_understand.ainvoke(args)
        except Exception:
            logger.warning("参考图预读失败：%s", name, exc_info=True)
            continue
        blocks.extend(
            tool_blocks(
                f"prefill_vision_{idx}",
                "image_understand",
                args,
                out.model_dump_json(exclude_none=True),
            )
        )
        s.called_tools.add("image_understand")
        s.recent_actions.append(_summarize_call("image_understand", args))
        if not out.degraded and out.search_query:
            hints.append(f"{out.subject or out.category}（检索词：{out.search_query}）")
    return blocks, "；".join(hints)


def append_prefilled(agent: Agent, blocks: list[Any]) -> None:
    """预置产物写进 ``state.context``——**不落 state 就等于没发生**。

    每轮的 messages 都从 ``state.context`` 重建（见 L5 那条注入蒸发的坑），只塞进本次请求
    的 kwargs 里，下一轮就没了：模型会发现自己「调过 planner 却看不到结果」。
    一整轮的 tool_call / tool_result 同住一条 assistant 消息，这里照这个形状拼。
    """
    if blocks:
        agent.state.context.append(Msg(name=agent.name, role="assistant", content=blocks))
