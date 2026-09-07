"""L1 工具白名单：模型只能调用已注册的工具，别的名字一律拒。

**这层在当前依赖下是纵深防御，不是唯一防线——诚实标注。** AgentScope 的 ``Toolkit`` 本来就只
按注册进去的那些工具查找并执行，模型幻觉出一个 ``rm_database`` 通常在框架层就落不了地。
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
    """当前进程允许调用的全部工具名（= 工具注册表里的全集，与角色发放无关）。

    白名单对主 Agent 与各类 worker 是同一个——**授权的差异由发放范围表达**（``build_toolkit(role)``
    发给谁哪些工具，见 ``tool_registry``），不在这里分叉。这层只回答「这个名字是不是本系统的工具」。

    批 4-3 起还包含两类**非本仓实现**、但由本仓主动挂进 Toolkit 的工具：框架内置的 skill
    阅读器（``Skill``）与 MCP 工具（``mcp__{server}__{tool}``）。它们今天走不到这道闸——
    harness 的工具中间件挂在本仓自己的 ``FunctionTool`` 实例上，框架内部造的对象够不着（见
    ``app/agent/mcp_registry.py`` 的诚实标注）。写进来是为了这道闸的**判据保持正确**：它回答
    的是「这个名字是不是本系统的工具」，而它们确实是。等哪天控制面能接上，第一道闸不该反过来
    把自家挂的工具当幻觉拒掉——那种失效方向的 bug 只会在切换的那一刻才现形。

    MCP 名单**不查 server**（走本地配置推导）：白名单不能依赖一次网络往返，对端一挂第一道
    安全闸自己先不可用。
    """
    from app.agent.mcp_registry import MCP_ROLES, mcp_tool_names
    from app.agent.skills import SKILL_VIEWER_TOOL_NAME
    from app.agent.tool_registry import TOOLS

    names = {t.name for t in TOOLS}
    names.add(SKILL_VIEWER_TOOL_NAME)
    for role in sorted(MCP_ROLES):
        names.update(mcp_tool_names(role))
    return frozenset(names)


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
