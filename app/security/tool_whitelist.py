"""L1 工具白名单：模型只能调用已注册的工具，别的名字一律拒。

**这层在当前依赖下是纵深防御，不是唯一防线——诚实标注。** LangChain 的 tool node 本来就只会
按 ``tools=`` 里注册的那些去查找并执行，模型幻觉出一个 ``rm_database`` 通常在框架层就落不了地。
那为什么还要写？三个理由：

1. **不把安全性寄托在上游框架的实现细节上。** 「框架恰好会拦」和「我们明确拒绝」是两回事，
   前者随版本变化，后者是我们自己的契约。
2. **工具表将来可能动态化。** 一旦引入 MCP / 插件式工具注册，工具名就成了外部可影响的输入。
3. **它是唯一能把「模型试图越界」这件事变成可观测信号的地方**——拦下即打日志 + metric，
   而不是被框架静默丢弃。

白名单**懒加载**（首次校验时才 import ``tool_registry``）：``tool_registry`` 会 import 全部九件
工具，进而拉起召回 / RAG 客户端。在模块导入期就做这件事会让「只想校验一个工具名」的单测被迫
初始化半个系统。
"""

from __future__ import annotations

import logging
from functools import lru_cache

logger = logging.getLogger("shoppingx.security.whitelist")


@lru_cache(maxsize=1)
def allowed_tools() -> frozenset[str]:
    """当前进程允许调用的全部工具名（= ``FULL_TOOL_SET`` 的工具名集合）。

    主 Agent 与所有 fork 子 Agent 共用同一份 ``FULL_TOOL_SET``（同质 fork 的硬约束），所以白名单
    对主 / 子是同一个——**授权的差异由深度闸表达**（见 ``harness/hooks/tool_gates.py``），不在这里
    分叉。这层只回答「这个名字是不是本系统的工具」。
    """
    from app.agent.tool_registry import FULL_TOOL_SET

    return frozenset(t.name for t in FULL_TOOL_SET)


def validate_tool_call(tool_name: str) -> bool:
    """工具名是否在白名单内。空名 / 未注册名一律 ``False``。

    取不到白名单（``tool_registry`` 导入失败，例如缺 env 的离线单测）时**放行**：安全层绝不能
    因为自身不可用而把主链路锁死——这道闸的定位是纵深防御，不是唯一防线（见模块 docstring）。
    """
    if not tool_name:
        return False
    try:
        return tool_name in allowed_tools()
    except Exception:
        logger.warning("工具白名单不可用，本次放行 tool=%s", tool_name, exc_info=True)
        return True
