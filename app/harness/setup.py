"""Harness Hook 注册入口：导入各 Hook 模块即触发 @harness_hook 装饰器注册。

只需 import 一次（进程级）。由 ``orchestrator.run_agent`` 与 ``agents._assemble`` 各调一次
``setup_harness()``（幂等，后者兜离线脚本 / 单测直接装配 Agent 的路径）。
"""

from __future__ import annotations

import logging

logger = logging.getLogger("shoppingx.harness.setup")

_initialized = False


def setup_harness() -> None:
    """导入所有 Hook 模块，触发 @harness_hook 装饰器的自动注册。幂等。"""
    global _initialized
    if _initialized:
        return
    _initialized = True

    # 按关切分文件（每文件 = 一个关切，内含它在各 hook 点上的全部钩子）。装饰器在 import 时注册。
    import app.harness.hooks.budget  # noqa: F401  检索 / fork / token 预算闸 + 预算档位路由
    import app.harness.hooks.context_shaping  # noqa: F401  上下文压缩 + 偏好注入 + 成功策略注入/结账
    import app.harness.hooks.drift  # noqa: F401  Silent Drift 漂移检测 + 结果信号追踪
    import app.harness.hooks.progress  # noqa: F401  阶段机：复位 / 转移 / 回退 / 补搜 / 收线通告 / 收尾资格
    import app.harness.hooks.repetition  # noqa: F401  循环检测提示 + 同参数回放 + 工具熔断
    import app.harness.hooks.safety  # noqa: F401  白名单 / 深度断言 / 内容过滤 / 截断 / 输出审核与脱敏
    import app.harness.hooks.sequencing  # noqa: F401  工具前置条件：软断言 + 取消前必先查单硬拒
    import app.harness.hooks.termination  # noqa: F401  终结硬停 / 终结置位 / 终结纪律 / liveness 看门狗
    import app.harness.hooks.validation  # noqa: F401  Schema 断言 + 断言失败汇总纠正
    from app.harness.middleware import harness

    hooks = harness.list_hooks()
    logger.info("Harness 初始化完成，注册 %d 个 Hook", len(hooks))
    for hp, name, prio in hooks:
        logger.debug("  %s: %s (priority=%d)", hp, name, prio)
