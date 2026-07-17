"""web_search —— 检索外部事实（评测 / 博主推荐 / 价格趋势）。

商品库里没有的外部信息（「这双鞋值不值」「今年这品类流行什么」「最近降价了吗」）走公网搜。
接 Tavily（一个面向 LLM 的搜索 API，直接返回抽取好的正文，省去自己爬页解析）。

**优雅降级**：没配 ``TAVILY_API_KEY`` 时不报错、不崩——返回空结果 + 一句说明，让主 loop
知道「这条外部信息暂时拿不到」，照样能基于已有信息收尾。这和召回层 endpoint 未就绪退化到
本地编码是同一套「缺外部依赖也能跑」的思路，保证离线 / CI 可用。
"""

from __future__ import annotations

import os

import httpx
from langchain_core.tools import tool
from pydantic import BaseModel

from app.api import monitor
from app.utils.circuit_breaker import CircuitBreaker
from app.utils.env import env_float, env_int
from app.utils.retry import call_with_retry

_TAVILY_URL = "https://api.tavily.com/search"

# 韧性（B 块）：Tavily 外呼的断路器（模块级单例——web_search 是函数工具，主/子 Agent 共用）。
# 连续失败到阈值即熔断，OPEN 期直接走降级 note、不再每次干等 20s 超时。
_breaker = CircuitBreaker(
    "web_search",
    failure_threshold=env_int("WEB_SEARCH_CB_THRESHOLD", 5),
    recovery_timeout=env_float("WEB_SEARCH_CB_RECOVERY", 30.0),
)


class WebResult(BaseModel):
    """一条网页搜索结果。"""

    title: str
    url: str
    content: str = ""
    score: float = 0.0


class WebSearchOutput(BaseModel):
    """web_search 的结构化返回。"""

    query: str
    results: list[WebResult]
    answer: str = ""  # Tavily 给的概括性答案（可能为空）
    note: str = ""  # 降级 / 异常时的说明


@tool
async def web_search(query: str, max_results: int = 5) -> WebSearchOutput:
    """检索公网外部事实（评测、博主推荐、价格趋势等商品库里没有的信息）。

    何时调用：① plan 判 intent_grounding=web——意图含没把握的新说法 / 潮流词 / 时效诉求时，
    检索前先用它把说法翻译成品类词 / 英文检索词（结果只用于改写检索词，不当商品候选）；
    ② 需要商品检索之外的外部信息（口碑 / 评测 / 趋势）来佐证或补充时。
    参数：
      - query：搜索词。
      - max_results：返回条数，默认 5。
    """
    await monitor.report_tool_start("web_search", query=query)
    api_key = os.environ.get("TAVILY_API_KEY", "").strip()
    if not api_key:
        out = WebSearchOutput(
            query=query,
            results=[],
            note="未配置 TAVILY_API_KEY，web_search 已跳过；请基于已有信息判断或如实说明不确定。",
        )
        await monitor.report_tool_end("web_search", results=0, degraded=True, result=out.note)
        return out

    degraded = False
    try:

        async def _do() -> dict:
            async with httpx.AsyncClient(timeout=20.0) as client:
                resp = await client.post(
                    _TAVILY_URL,
                    json={
                        "api_key": api_key,
                        "query": query,
                        "search_depth": "basic",
                        "max_results": max(1, min(max_results, 10)),
                    },
                )
                resp.raise_for_status()  # 4xx 立即抛，5xx 由 call_with_retry 退避重试
                return resp.json()

        # 断路器包「含重试的远程调用」：OPEN 期抛 CircuitOpenError，退避只兜瞬时抖动（超时/5xx）。
        data = await _breaker.call(lambda: call_with_retry(_do))
        results = [
            WebResult(
                title=r.get("title", ""),
                url=r.get("url", ""),
                content=r.get("content", ""),
                score=float(r.get("score", 0.0)),
            )
            for r in data.get("results", [])
        ]
        out = WebSearchOutput(query=query, results=results, answer=data.get("answer", "") or "")
    except Exception as e:  # 外部依赖失败（或已熔断）不该崩主 loop，转成可读 note + 标降级
        degraded = True
        out = WebSearchOutput(
            query=query,
            results=[],
            note=f"web_search 调用失败（{type(e).__name__}），请如实说明不确定。",
        )

    # 思考结果摘要：Tavily 概括答案 + 头部几条标题（供前端展开看这一步「查到什么外部事实」）；
    # 降级 / 无果时退回 note。
    ws_lines: list[str] = []
    if out.answer:
        ws_lines.append(out.answer)
    ws_lines.extend(f"· {r.title}" for r in out.results[:3] if r.title)
    ws_result = "\n".join(ws_lines) or out.note
    # 异常 / 熔断降级时标 degraded=True（修正现状：原异常路径未标，前端/监控看不出这次是降级）。
    await monitor.report_tool_end(
        "web_search", results=len(out.results), degraded=degraded, result=ws_result
    )
    return out
