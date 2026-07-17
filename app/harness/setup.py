"""Harness Hook 注册入口：导入各 Hook 模块即触发 @harness_hook 装饰器注册。

只需 import 一次（进程级）。由 ``build_agent_middleware`` 在首次调用时触发。
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

    # 按模块导入——装饰器在 import 时自动注册到全局 harness 单例
    import app.harness.hooks.assertion_handler  # noqa: F401  assertion 失败汇总
    import app.harness.hooks.context_compress  # noqa: F401  pre_think: 预算 hint + 上下文压缩
    import app.harness.hooks.drift_detector  # noqa: F401  Silent Drift + 结果信号追踪
    import app.harness.hooks.phase_check  # noqa: F401  阶段权限拦截
    import app.harness.hooks.phase_transition  # noqa: F401  阶段转移 + 回退 + 信号追踪
    import app.harness.hooks.preference_inject  # noqa: F401  planner 后注入域内长期偏好
    import app.harness.hooks.reasoning_boost  # noqa: F401  fast 档：只给主 loop 第一轮开 reasoning
    import app.harness.hooks.result_guard  # noqa: F401  截断 + 循环检测 + 分级提示 + 终结标记
    import app.harness.hooks.security  # noqa: F401  安全护栏 L1 白名单 / L3 过滤 / L4 脱敏
    import app.harness.hooks.session_hooks  # noqa: F401  阶段复位 + 输出审核
    import app.harness.hooks.step_validator  # noqa: F401  Schema/Sequencing/Semantic
    import app.harness.hooks.terminal_enforce  # noqa: F401  终结纪律（当场重发模型）
    import app.harness.hooks.tool_breaker  # noqa: F401  工具级熔断（闸 + 计数）
    import app.harness.hooks.tool_gates  # noqa: F401  终结硬停/深度/检索/fork/预算 各硬闸
    import app.harness.hooks.tool_memo  # noqa: F401  同参数重复调用回放（幂等工具不重复执行）
    import app.harness.hooks.watchdog  # noqa: F401  liveness 看门狗（停滞→收敛指令→硬停交部分结果）
    from app.harness.middleware import harness

    hooks = harness.list_hooks()
    logger.info("Harness 初始化完成，注册 %d 个 Hook", len(hooks))
    for hp, name, prio in hooks:
        logger.debug("  %s: %s (priority=%d)", hp, name, prio)
