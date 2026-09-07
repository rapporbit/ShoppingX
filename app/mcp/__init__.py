"""MCP 两侧（批 4-3）。

- :mod:`app.mcp.server` —— **生产侧**：把本仓的只读三工具（``item_search`` / ``price_compare``
  / ``shipping_calc``）暴露成一个 MCP server，供仓外的 Agent / IDE / 别的编排框架消费。
- :mod:`app.mcp.fx_server` —— **消费侧的对端**：自建汇率 MCP，被 SearchAgent 的 Toolkit 挂进
  来（接线在 :mod:`app.agent.mcp_registry`）。挑汇率是因为它零外部依赖（纯静态表，见
  ``app/recall/fx.py``），验的是「本仓能不能吃外部 MCP」这条通路，而不是某个第三方服务今天
  在不在线。

两侧刻意分成两个进程/两个 server：生产侧那三个工具要跑真实召回（Qdrant / reranker），把它
和一个「零依赖、随时可起」的对端塞进同一个 server，会让消费侧的验收被召回栈的可用性绑架。
"""
