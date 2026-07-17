"""Cross-encoder 精排客户端（BGE-Reranker-v2-m3，refdocs 13-1 §4）。

**为什么要精排：** Hybrid 粗排出来的 Top-30 里常夹着 1-2 张「相关但跑题」的卡片
（查 travel accessories 混进一张只命中 "travel" 的洗漱包卡）。双塔召回是「你算你的、
我算我的、最后比相似度」，没有交叉特征；cross-encoder 把 query 和候选**拼进同一个
模型**判相关性，更细，但更慢——所以前者做粗排（几十条），后者做精排（个位数）。

**无 GPU 约束（同 M3 编码器策略）：** 不自训、不本地拉模型权重。配了
``RERANKER_ENDPOINT`` 就走远程现成 BGE-Reranker 服务；没配则退化到**确定性本地打分**
（query×候选的 token 重叠），保证离线/CI 可跑、结果可复现。本地分不是「真精排」，但能
让短路逻辑、调用链在没有服务时照样测得通——诚实地以 :attr:`remote` 标注当前是否走真模型。

**远程协议（默认走 Cohere/Jina/硅基流动同构的标准 rerank API）：** 配了
``RERANKER_MODEL`` + ``RERANKER_API_KEY`` 即按标准 ``POST /v1/rerank`` 发请求——
``{"model", "query", "documents", "top_n", "return_documents": false}``，Bearer 鉴权，
返回 ``{"results": [{"index", "relevance_score"}, ...]}``（按相关度降序、可能截断），
本客户端按 ``index`` 映射回**候选原序**。未配 model/key 时退回最朴素的自定义契约
``{"query","candidates"} -> {"scores"}``（同序），两种返回体都做了兼容解析。
"""

from __future__ import annotations

import logging
import os
import re
from functools import lru_cache

import httpx

from app.utils.circuit_breaker import CircuitBreaker, CircuitOpenError
from app.utils.env import env_float, env_int
from app.utils.retry import call_with_retry

logger = logging.getLogger("shoppingx.reranker")

# 本地回退分词：抓连续字母数字片段（中英混排够用，中文按单字切由下面的 bigram 兜）。
_WORD_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def _tokens(text: str) -> set[str]:
    """切出用于 token 重叠打分的词集合：英文按词、其余按单字符（覆盖中文）。"""
    low = text.lower()
    words = set(_WORD_RE.findall(low))
    # 非字母数字字符（如中文）逐字加入，让中文 query×中文摘要也能有重叠信号。
    words.update(ch for ch in low if not ch.isspace() and not _WORD_RE.match(ch))
    return words


class RerankerClient:
    """BGE-Reranker 的统一壳子：远程 HTTP 精排 + 确定性本地回退。

    远程服务约定（refdocs 13-1 §4.3）：``POST {endpoint}`` 入参
    ``{"query": str, "candidates": list[str]}``，出参 ``{"scores": list[float]}``
    与 candidates **同序**，分越大越相关。
    """

    def __init__(
        self,
        endpoint: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        timeout: float = 10.0,
    ) -> None:
        self._endpoint = endpoint if endpoint is not None else os.environ.get("RERANKER_ENDPOINT")
        # 标准 rerank API 需要 model 字段；配了就走标准协议，否则走自定义 {candidates}->{scores}。
        self._model = model if model is not None else os.environ.get("RERANKER_MODEL")
        # rerank 与 embedding 常同供应商，key 缺省回退 EMBED_API_KEY / OPENAI_API_KEY。
        self._api_key = (
            api_key
            if api_key is not None
            else (
                os.environ.get("RERANKER_API_KEY")
                or os.environ.get("EMBED_API_KEY")
                or os.environ.get("OPENAI_API_KEY")
            )
        )
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None
        # 韧性（B 块）：断路器 + 退避重试。连续失败到阈值即熔断，OPEN 期直接走本地兜底、
        # 不再每次干等 10s 超时；瞬时抖动（超时 / 5xx）先由 call_with_retry 消化。
        self._breaker = CircuitBreaker(
            "reranker",
            failure_threshold=env_int("RERANKER_CB_THRESHOLD", 5),
            recovery_timeout=env_float("RERANKER_CB_RECOVERY", 30.0),
        )
        self._retry_attempts = env_int("RERANKER_RETRY_ATTEMPTS", 3)

    @property
    def remote(self) -> bool:
        """是否走远程真实 reranker（否则为确定性本地回退）。"""
        return bool(self._endpoint)

    async def score(self, query: str, candidates: list[str]) -> list[float]:
        """给每个候选打相关性分，返回与 ``candidates`` 同序的分数列表。

        远程失败（超时/连不上/状态码异常）不抛崩溃——退化到本地打分，让上层
        （category_insight）至少拿到一个能排序的结果，符合 refdocs §6.4「不抛异常、
        给低置信度结构化结果」的兜底原则。需要知道「本次是否真走了远程」的调用方用
        :meth:`score_detailed`。
        """
        scores, _ = await self.score_detailed(query, candidates)
        return scores

    async def score_detailed(self, query: str, candidates: list[str]) -> tuple[list[float], bool]:
        """同 :meth:`score`，但额外返回「本次是否真用了远程精排」。

        返回 ``(scores, used_remote)``。``used_remote=False`` 表示走的是本地 token 重叠
        回退（要么没配 endpoint，要么远程调用失败降级）——调用方据此**如实**上报，避免
        在远程静默降级时仍对前端/监控声称「已精排」（G2）。``used_remote`` 是本次调用的
        局部返回值、非共享实例状态，并发调用之间不串扰。
        """
        if not candidates:
            return [], False
        if self.remote:
            try:
                # 断路器包远程：OPEN 期抛 CircuitOpenError，直接落进下面的 except 走本地兜底。
                scores = await self._breaker.call(lambda: self._score_remote(query, candidates))
                return scores, True
            except (httpx.HTTPError, KeyError, ValueError, CircuitOpenError) as exc:
                # 远程精排挂了（或已熔断）就退化粗排序，不让整条链路崩——记日志且 used_remote=False。
                logger.warning(
                    "远程精排失败，本次降级为本地 token 重叠打分（候选 %d 条）：%s: %s",
                    len(candidates),
                    type(exc).__name__,
                    exc,
                )
                return self._score_local(query, candidates), False
        return self._score_local(query, candidates), False

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else None
            self._client = httpx.AsyncClient(timeout=self._timeout, headers=headers)
        return self._client

    async def _score_remote(self, query: str, candidates: list[str]) -> list[float]:
        client = self._get_client()
        if self._model:
            # 标准 rerank API（Cohere/Jina 同构）：model + documents，返回按相关度排序的 results。
            payload: dict = {
                "model": self._model,
                "query": query,
                "documents": candidates,
                "top_n": len(candidates),  # 要回全部候选的分，否则只回 top_n 条
                "return_documents": False,
            }
        else:
            # 朴素自定义契约（无 model/key 时）：同序 candidates -> scores。
            payload = {"query": query, "candidates": candidates}

        async def _do() -> httpx.Response:
            resp = await client.post(self._endpoint or "", json=payload)
            resp.raise_for_status()  # 4xx 立即抛（不重试），5xx 交给 call_with_retry 退避重试
            return resp

        # 退避重试只兜瞬时抖动（超时 / 5xx）；解析放在重试外——解析报错是数据问题，重试无意义。
        resp = await call_with_retry(_do, attempts=self._retry_attempts)
        return self._parse_remote(resp.json(), len(candidates))

    @staticmethod
    def _parse_remote(body: dict, n: int) -> list[float]:
        """把远程返回体解析成与候选**同序、同长**的分数列表，两种契约都兼容。

        - 标准 rerank：``{"results": [{"index", "relevance_score"}, ...]}``，按 index 散回原位，
          缺失的候选（被 top_n 截断时）补 0.0（bge-reranker 分域 0~1，0 即最不相关）。
        - 朴素契约：``{"scores": [...]}``，要求与候选同序同长。
        """
        if "results" in body:
            scores = [0.0] * n
            for item in body["results"]:
                idx = int(item["index"])
                if 0 <= idx < n:
                    # 不同实现字段名略有出入，优先 relevance_score，回退 score。
                    raw = item.get("relevance_score", item.get("score", 0.0))
                    scores[idx] = float(raw)
            return scores
        scores = body["scores"]
        if len(scores) != n:
            raise ValueError("reranker 返回分数条数与候选不一致")
        return [float(s) for s in scores]

    def _score_local(self, query: str, candidates: list[str]) -> list[float]:
        """确定性本地相关性分：query 与候选的 token Jaccard 重叠（0-1）。

        Jaccard = |交集| / |并集|：对长度不一的候选更公平（纯交集数会偏向长文本）。
        和远程 cross-encoder 的语义判断不可同日而语，但**同输入恒同输出**，足够让
        精排短路与排序在离线下被断言。
        """
        q = _tokens(query)
        if not q:
            return [0.0] * len(candidates)
        out: list[float] = []
        for cand in candidates:
            c = _tokens(cand)
            if not c:
                out.append(0.0)
                continue
            inter = len(q & c)
            union = len(q | c)
            out.append(inter / union if union else 0.0)
        return out

    async def aclose(self) -> None:
        """释放底层 HTTP 连接（远程模式用过才有）。"""
        if self._client is not None:
            await self._client.aclose()
            self._client = None


@lru_cache(maxsize=1)
def get_reranker() -> RerankerClient:
    """进程内共享的 reranker 客户端（主 / 子 Agent 及各工具复用）。"""
    return RerankerClient()
