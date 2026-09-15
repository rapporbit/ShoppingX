"""Harness 的钩子，按**关切**分文件：一个文件 = 一个关切 = 它在各 hook 点上的全部钩子。

    safety.py           安全底线：内容过滤 / 截断 / 输出审核与脱敏
                        （硬，无逃生门）
    termination.py      终结：终结硬停闸 + 置位 + 终结纪律重发 + liveness 看门狗
    budget.py           预算：检索 / fork / token 三类额度闸 + 预算档位路由
    sequencing.py       工具前置条件：PREREQUISITES 一张表（软警告）+ 取消前必先查单（硬拒）
    validation.py       单步断言：Schema 断言 + 断言失败汇总纠正
    repetition.py       重复调用：LoopDetector 提示 + 同参数回放 + 工具熔断
    progress.py         检索进度机：复位 / 转移 / 回退 / 补搜 / 收线通告 / 收尾资格（不是权限机）
    drift.py            Silent Drift 漂移检测 + 结果信号追踪
    context_shaping.py  塑形上下文：压缩 + 偏好注入 + 成功策略注入 / 结账

按文件切而不是按 hook 点切，是因为一个关切天然横跨 3～4 个 hook 点（终结要在 pre_tool_call 拦、在
post_tool_call 置位、在 post_reflect 催），按点切会把它散在 4 个文件里。代价是「同一个 hook 点上
谁先谁后」
不在一页上——**跨文件的顺序契约**集中列在这里，改 priority 前先看：

pre_tool_call（低先执行）：
    5 terminal_reached · 12 trade_sequence · 15 websearch
    · 20 phase_check
    · 25 sequencing · 27 tool_memo_replay · 30 search_authority · 33 token_budget · 35 fork_budget
    · 45 retrieval_charge · 48 tool_breaker
  - search_authority(30) 读 item_search_calls 的**自增前**值，自增在 retrieval_charge(45)——
    挪了顺序，
    子 Agent 的「恰好放行一次」塌成「放行 0 次」。
  - token_budget(33) 必须早于 fork_budget(35)：fork 闸 charge 即扣槽。
  - tool_memo_replay(27) 早于 retrieval_charge(45)：回放不是真实检索，不该扣预算；
    但晚于 phase_check(20)。
  - tool_breaker(48) 必须最后：allow() 有副作用。

post_tool_call：
    5 tool_breaker_record · 5 content_filter · 10 truncate_result · 15 tool_memo_record
    · 19 transition_notice · 20 result_nudges · 30 mark_terminal · 40 schema_assertion
    · 50 preference_inject · 50 drift_result_tracker
  - truncate_result(10) 必须早于所有追加提示的钩子（19 / 20），否则刚贴的提示被截掉。
  - schema_assertion(40) 用 raw_decode 容忍 19 / 20 缀在尾部的通告。

pre_think：5 liveness_watchdog · 20 budget_router。（上下文压缩交框架 compress_context，无 Hook。）
post_reflect：15 assertion_handler · 20 drift_detector · 39 refine_backfill · 40 phase_transition
    · 41 phase_rollback · 60 terminal_enforcer。
on_session_start：10 phase_init。
on_session_end：10 output_guard · 20 output_audit · 90 strategy_feedback。
on_system_prompt（装配期）：50 strategy_inject · 60 trade_state_inject。

注册与执行见 ``app/harness/middleware.py``；各钩子在 AgentScope 上的落点见
``app/harness/adapter.py``。
"""
