# examples/

各项能力的独立示例脚本（与 `app/` 主线分离，不互相污染）：

- `min_loop.py` —— 最小 AgentLoop：玩具工具跑通 Think→Act→Observe→Reflect
- `stream.py` —— 用 `reply_stream` 实时观察循环过程
- `tools_pipeline.py` —— 业务工具的主链路串联（离线，不调 LLM）
- `category_rag.py` —— CategoryInsight 的 RAG 链路（召回 → 精排 → 结构化提炼）
- `memory.py` —— 长期记忆「跨会话记住偏好」闭环（离线）
- `agui_events.py` —— AGUI 事件协议 + WebSocket 实时推送
- `main_agent.py` —— 主 AgentLoop 从入口跑到收尾（需真实 LLM）
- `server_e2e.py` —— FastAPI 前后端闭环的协议级端到端（桩 Agent，零 LLM）
