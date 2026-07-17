"""可观测性（A 块）：Prometheus metrics + 结构化日志。

不自建 OpenTelemetry system trace——Langfuse v4 已基于 OTel 归并调用树（见 app/agent/tracing.py）。
本包只补两样真正缺位的：系统级 metrics（/metrics）与带上下文的结构化日志（structlog）。
"""
