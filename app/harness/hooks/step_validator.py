"""三类单步断言：Schema / Sequencing / Semantic。

- Schema Assertion（post_tool_call, <1ms）：工具返回是否能解析为合法 JSON、关键字段是否缺失。
- Sequencing Assertion（pre_tool_call, <1ms）：工具调用顺序是否满足前置条件。
- Semantic Assertion（post_tool_call, ~50ms）：工具返回内容与用户 query 的语义对齐度（轻量 LLM）。

断言失败不中断 Agent——记录到 ``context["assertions_failed"]``，由下游的
``assertion_handler`` Hook 汇总后注入纠正提示，让模型自行修正。
"""

from __future__ import annotations

import json
import logging
from typing import Any

from pydantic import ValidationError

from app.harness.middleware import harness_hook
from app.harness.signals import candidate_count
from app.utils.env import env_bool

logger = logging.getLogger("shoppingx.harness.step_validator")

# ---------- Schema Assertion ----------

_SCHEMA_TOOLS: dict[str, str] = {
    "item_search": "app.tools.item_search.ItemSearchOutput",
    "price_compare": "app.tools.price_compare.PriceCompareOutput",
    "shipping_calc": "app.tools.shipping_calc.ShippingCalcOutput",
    "category_insight": "app.tools.category_insight.CategoryInsightOutput",
    "item_picker": "app.tools.item_picker.ItemPickerOutput",
    "shopping_summary": "app.tools.shopping_summary.ShoppingSummaryOutput",
    "planner": "app.tools.planner.PlanOutput",
    "web_search": "app.tools.web_search.WebSearchOutput",
}

_resolved_schemas: dict[str, type] = {}


def _resolve_schema(dotted: str) -> type | None:
    """延迟解析 Pydantic model（避免循环导入）。"""
    if dotted in _resolved_schemas:
        return _resolved_schemas[dotted]
    try:
        module_path, cls_name = dotted.rsplit(".", 1)
        import importlib

        mod = importlib.import_module(module_path)
        cls = getattr(mod, cls_name)
        _resolved_schemas[dotted] = cls
        return cls
    except Exception:
        logger.debug("无法解析 schema %s", dotted, exc_info=True)
        return None


def _restore_render_projection(tool_name: str, data: dict[str, Any]) -> None:
    """把渲染层刻意省略、但可从上下文推导的字段回填，再交给完整 schema 验证。

    验证对象是**给模型看的渲染串**，而渲染契约允许裁剪冗余（如 item_search 单平台时每条候选的
    ``platform``——顶层已写一次，见 ``ItemSearchOutput.__str__``）。不回填就拿完整 schema 去验
    投影，等于要求「字面回显」：每次单平台检索必假阳性，一轮一条「格式问题」纠正灌进上下文
    （eval q05 实测由此每轮斩断前缀缓存）。回填后验的才是本断言的本意——信息完整性。"""
    if tool_name != "item_search":
        return
    platform = data.get("platform")
    if not isinstance(platform, str) or platform == "all":
        return  # "all" 合流时渲染必须逐条带 platform，缺了就是真错，不回填
    candidates = data.get("candidates")
    if not isinstance(candidates, list):
        return
    for c in candidates:
        if isinstance(c, dict):
            c.setdefault("platform", platform)


@harness_hook("post_tool_call", name="schema_assertion", priority=40)
async def check_schema(context: dict[str, Any]) -> dict[str, Any] | None:
    """验证工具返回是否符合预期 Pydantic schema。"""
    tool_name = context.get("tool_name", "")
    tool_result = context.get("tool_result", "")

    dotted = _SCHEMA_TOOLS.get(tool_name)
    if not dotted:
        return None
    schema_cls = _resolve_schema(dotted)
    if schema_cls is None:
        return None

    # ToolMessage content 是字符串——解析开头的 JSON 对象。用 raw_decode 而不是 loads：
    # 先跑的 Hook（transition_notice/result_nudges，priority < 40）会在结果尾部追加通告，
    # loads 会因 Extra data 抛错、令断言静默跳过——验不验居然取决于有没有别的 Hook 贴过话。
    if isinstance(tool_result, str):
        try:
            data, _ = json.JSONDecoder().raw_decode(tool_result.lstrip())
        except (json.JSONDecodeError, ValueError):
            # 开头就不是 JSON 也不一定是错——有些工具返回纯文本摘要（如 planner 走 LLM 产出）
            return None
    elif isinstance(tool_result, dict):
        data = tool_result
    else:
        return None

    if isinstance(data, dict):
        _restore_render_projection(tool_name, data)

    try:
        schema_cls.model_validate(data)  # type: ignore[attr-defined]
    except ValidationError as exc:
        context.setdefault("assertions_failed", []).append(
            {
                "type": "schema",
                "tool": tool_name,
                "reason": str(exc.errors()[:2]),  # 只保留前 2 条错误避免膨胀
            }
        )
        logger.info("Schema assertion failed: %s → %s", tool_name, exc.errors()[:2])
    return context


# ---------- Sequencing Assertion ----------

# 工具名 → 前置工具候选列表，**满足其一即可**（不是全部都要）。
#
# 检索的前置必须把 fork 通路算进去：本项目跨平台检索的主路径是 dispatch_tool /
# parallel_dispatch_tool 派子 Agent 去 item_search，主 loop 自己从头到尾可能一次 item_search
# 都没调过。只认 item_search 会让「fork 检索 → item_picker」这条正常链路每次都被误报顺序错误。
PREREQUISITES: dict[str, list[str]] = {
    "shopping_summary": ["item_picker"],
    "price_compare": ["item_search", "dispatch_tool", "parallel_dispatch_tool"],
    "shipping_calc": ["price_compare"],
    "item_picker": ["item_search", "dispatch_tool", "parallel_dispatch_tool"],
}

# 这些工具的真实前置是「登记表里有候选」，工具名只是达成它的若干条路径之一。
# 有候选即视为前置已满足——结构化信号比工具名可靠（候选也可能来自续聊的历史轮次）。
_CANDIDATE_CONSUMERS = frozenset({"item_picker", "price_compare"})


@harness_hook("pre_tool_call", name="sequencing_assertion", priority=25)
async def check_sequencing(context: dict[str, Any]) -> dict[str, Any] | None:
    """验证工具调用顺序是否满足前置条件（满足任一前置即通过）。

    不硬拒绝——有些场景确实需要跳步（如用户直接给了候选列表）。
    只注入警告让模型自己判断是否继续。
    """
    tool_name = context.get("tool_name", "")
    prerequisites = PREREQUISITES.get(tool_name)
    if not prerequisites:
        return None

    called: set[str] = context.get("called_tools", set())
    if any(p in called for p in prerequisites):
        return None

    if tool_name in _CANDIDATE_CONSUMERS and candidate_count() > 0:
        return None

    context.setdefault("assertions_failed", []).append(
        {
            "type": "sequencing",
            "tool": tool_name,
            "reason": (
                f"{tool_name} 通常在 {' 或 '.join(prerequisites)} 之后调用，但它们都还没执行过"
            ),
        }
    )
    logger.info("Sequencing warning: %s called before any of %s", tool_name, prerequisites)
    return context


# ---------- Semantic Assertion ----------

SEMANTIC_CHECK_TOOLS = frozenset({"item_search", "category_insight"})

_SEMANTIC_PROMPT = """判断以下工具返回是否和用户需求相关。
用户需求：{query}
工具返回摘要（前 200 字）：{preview}
只回答"相关"或"不相关"，不要解释。"""

SEMANTIC_ENABLED = env_bool("HARNESS_SEMANTIC_ASSERTION", False)


@harness_hook("post_tool_call", name="semantic_assertion", priority=45)
async def check_semantic_alignment(context: dict[str, Any]) -> dict[str, Any] | None:
    """轻量语义对齐检查（只对高价值工具执行，默认关闭，可经 env 开启）。"""
    if not SEMANTIC_ENABLED:
        return None

    tool_name = context.get("tool_name", "")
    if tool_name not in SEMANTIC_CHECK_TOOLS:
        return None

    query = context.get("original_query", "")
    result = context.get("tool_result", "")
    if not query or not result:
        return None

    preview = str(result)[:200]

    try:
        # fast 档而非 judge 强模型：refdocs 17-3 §2 把 Semantic Assertion 的延迟预算定在 ~50ms，
        # 它只需回一个「相关 / 不相关」的标签，用强模型是把在线延迟花在不需要的地方。
        from app.agent.llm import get_fast_llm

        llm = get_fast_llm()
        resp = await llm.ainvoke(
            [
                ("user", _SEMANTIC_PROMPT.format(query=query, preview=preview)),
            ]
        )
        raw = resp.content
        verdict = raw.strip() if isinstance(raw, str) else str(raw)
    except Exception:
        logger.debug("Semantic assertion LLM 调用失败，跳过", exc_info=True)
        return None

    if "不相关" in verdict:
        context.setdefault("assertions_failed", []).append(
            {
                "type": "semantic",
                "tool": tool_name,
                "reason": f"工具返回与用户需求「{query[:50]}」语义不相关",
            }
        )
        logger.info("Semantic assertion: %s result misaligned with query", tool_name)
    return context
