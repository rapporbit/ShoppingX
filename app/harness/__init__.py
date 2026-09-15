"""Harness 治理框架：统一 Hook Pipeline（middleware.py）+ 按关切分文件的钩子（hooks/，地图与
顺序契约见 hooks/__init__.py）+ AgentScope 桥接（adapter.py / session.py / prefill.py / streaming.py / tiering.py）
+ 控制面状态源（fork_guard / token_budget / retrieval_budget / model_router）。

状态源四个模块 2026-09-15 从 app/agent 挪入：harness 依赖 agent 是方向反了，
它们本就是各闸的状态与档位来源。"""
