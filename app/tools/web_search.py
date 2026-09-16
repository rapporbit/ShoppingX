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
from pydantic import BaseModel

from app.api import monitor
from app.tools._shell import tool
from app.utils.circuit_breaker import CircuitBreaker
from app.utils.env import env_float, env_int
from app.utils.retry import call_with_retry

_TAVILY_URL = "https://api.tavily.com/search"

# 返回截断：web_search 结果原样进主 loop 上下文，是最大的外部文本口子。实测（2026-09-15，5 条样本）
# basic 深度每条 content 86~1482 字符，长的是几段正文用 [...] 拼成、封顶约 1500。上限定得略高于
# 正常值——正常结果不截，只防 advanced 深度 / raw_content / 换搜索源时的异常长文。
# 最坏 = max_results 上限 10 × 单条 1500 = 总上限 15000。
_CONTENT_MAX_CHARS = env_int("WEB_SEARCH_CONTENT_MAX_CHARS", 1500)
_TOTAL_MAX_CHARS = env_int("WEB_SEARCH_TOTAL_MAX_CHARS", 15000)
_TRUNCATED_MARK = " [truncated]"


def _clip_contents(contents: list[str]) -> list[str]:
    """逐条截到单条上限，同时整批累计不超总上限；超出的条目 content 置空（标题 / url 保留）。"""
    clipped: list[str] = []
    remaining = _TOTAL_MAX_CHARS
    for text in contents:
        limit = min(_CONTENT_MAX_CHARS, max(remaining, 0))
        if len(text) > limit:
            text = text[:limit] + _TRUNCATED_MARK if limit > 0 else ""
        remaining -= min(len(text), limit)
        clipped.append(text)
    return clipped


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


async def search_web(query: str, max_results: int = 5) -> tuple[WebSearchOutput, bool]:
    """发一次 Tavily 搜索，返回 ``(结果, 是否降级)``。**不报 AGUI 事件**——上报归调用方。

    抽出来是给 ``research``（C2）复用：外呼 / 断路器 / 截断 / 降级 note 只有这一套实现，
    否则两个工具各写一遍，将来改截断定数必漏一边。降级（缺 key / 异常 / 熔断）一律返回
    空结果 + note，**不抛**——调用方据 ``note`` 决定怎么说「这条外部信息暂时拿不到」。
    """
    api_key = os.environ.get("TAVILY_API_KEY", "").strip()
    if not api_key:
        note = "未配置 TAVILY_API_KEY，web_search 已跳过；请基于已有信息判断或如实说明不确定。"
        return WebSearchOutput(query=query, results=[], note=note), True

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
        raw = data.get("results", [])
        contents = _clip_contents([r.get("content", "") or "" for r in raw])
        results = [
            WebResult(
                title=r.get("title", ""),
                url=r.get("url", ""),
                content=content,
                score=float(r.get("score", 0.0)),
            )
            for r, content in zip(raw, contents, strict=True)
        ]
        out = WebSearchOutput(query=query, results=results, answer=data.get("answer", "") or "")
    except Exception as e:  # 外部依赖失败（或已熔断）不该崩主 loop，转成可读 note + 标降级
        degraded = True
        out = WebSearchOutput(
            query=query,
            results=[],
            note=f"web_search 调用失败（{type(e).__name__}），请如实说明不确定。",
        )
    return out, degraded


@tool
async def web_search(query: str, max_results: int = 5) -> WebSearchOutput:
    """查公网外部事实（评测/口碑/趋势/新说法翻译成品类词）；不产候选。参数 query、max_results。"""
    await monitor.report_tool_start("web_search", query=query)
    out, degraded = await search_web(query, max_results)

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
