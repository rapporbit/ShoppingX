"""assertion 失败汇总：把 Schema / Sequencing / Semantic 断言失败转成纠正提示。"""

from __future__ import annotations

import logging
from typing import Any

from app.harness.middleware import harness_hook

logger = logging.getLogger("shoppingx.harness.assertion_handler")


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
