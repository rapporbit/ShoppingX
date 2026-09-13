"""单步断言：Schema（post_tool_call）+ 失败汇总（post_reflect）。确定性、微秒级、零 LLM。

    post_tool_call  40  schema_assertion    工具返回能否按 Pydantic *Output 解析
                                            （raw_decode 容忍尾部通告）
    post_reflect    15  assertion_handler   汇总本轮 schema / sequencing 失败，注入一条纠正提示

顺序断言本体在 ``sequencing.py``（它挂在 pre_tool_call），失败同样记入 ``assertions_failed``
由这里汇总。
Semantic Assertion 已删，理由见文件末尾。
"""

from __future__ import annotations

import json
import logging
from typing import Any

from pydantic import ValidationError

from app.harness.middleware import harness_hook

logger = logging.getLogger("shoppingx.harness.validation")


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


@harness_hook("post_reflect", name="assertion_handler", priority=15)
async def handle_failed_assertions(context: dict[str, Any]) -> dict[str, Any] | None:
    """汇总本轮所有 assertion 失败，注入纠正提示让模型自行修正。"""
    failures: list[dict] = context.pop("assertions_failed", [])
    if not failures:
        return None

    schema_fails = [f for f in failures if f["type"] == "schema"]
    seq_fails = [f for f in failures if f["type"] == "sequencing"]
    semantic_fails = [f for f in failures if f["type"] == "semantic"]

    messages: list[str] = []
    if schema_fails:
        f = schema_fails[0]
        messages.append(
            f"[格式问题] {f['tool']} 的返回格式不符合预期：{f['reason'][:120]}。"
            "请检查工具参数是否正确。"
        )
    if seq_fails:
        f = seq_fails[0]
        messages.append(f"[顺序问题] {f['reason']}")
    if semantic_fails:
        f = semantic_fails[0]
        messages.append(
            f"[相关性问题] {f['tool']} 的返回和用户需求不太对齐。考虑调整搜索词或换一个检索方向。"
        )

    if messages:
        context.setdefault("inject_messages", []).extend(
            {"role": "system", "content": m} for m in messages
        )
        logger.info("Assertion handler: injected %d correction(s)", len(messages))
    return context


# ---------- Semantic Assertion（已删，2026-09-10）----------
#
# refdocs 17-3 §2 的第三类断言：拿快档模型判一句「这个工具返回跟用户需求相不相关」。
# 本仓实现过（``HARNESS_SEMANTIC_ASSERTION`` 默认关），**301 个会话零触发**，删除理由有三条：
#
# 1. 它要付的是**在线延迟**：每次 item_search / category_insight 返回后多一次 LLM 往返，
#    换回来的只是一个「相关 / 不相关」标签，而后续只是往 assertions_failed 里记一笔。
# 2. 同一件事已经有确定性实现且在真实跑：品类门（``item_picker`` 的 rerank 软降权）用
#    cross-encoder 分数判「跑没跑题」，判据可复现、零 LLM；语义断言是它的模糊版本。
# 3. 默认关 = 从未被验证过。留着一段没人跑过的 LLM 调用，比没有它更危险。
#
# 要重新引入的话，先答一个问题：判出「不相关」之后**做什么**？当年的答案是「记一笔」——
# 那就不值一次 LLM 往返。
