"""工具结果截断原语（Feedback / Computational）。

单个工具一次返回过长（如爬十几页条款）会灌爆上下文、废掉后续轮。超过 token 预算就尾部截断
并留提示，让模型知道「结果被截断、可缩小查询」。

两处用它：``post_tool_call`` 的 ``truncate_result`` Hook（主/子 loop 内逐个工具结果），以及
``dispatch_tool`` 在 fork 边界回传子 Agent 最终结果前。
"""

from __future__ import annotations

from app.utils.tokens import truncate_to_token_budget

# 单工具结果上限：约 4000 token。token 数走 app.utils.tokens（Qwen 本地分词器为主、CJK 启发式
# 兜底），不再用 char/4 粗估（后者对中文低估约 3 倍）。
MAX_TOOL_RESULT_TOKENS = 4000

_TRUNCATE_HINT = "\n\n[…工具结果过长已截断，可用更窄的查询参数重试]"


def truncate_tool_result(text: str, max_tokens: int = MAX_TOOL_RESULT_TOKENS) -> str:
    """工具返回过长时按 token 预算截断并加省略提示；未超长则原样返回。"""
    return truncate_to_token_budget(text, max_tokens, _TRUNCATE_HINT)
