"""品类知识库检索客户端（OpenSearch Hybrid + 进程内本地回退）。

refdocs 04-1 定型：**应用层走 OpenSearch**——它把「语义召回(KNN) + 全文匹配(BM25) +
标量过滤 + 线性加权融合」装进同一套 DSL，且融合权重在引擎层运行时可调，这是 Faiss
等纯向量库做不到的（详见 04-1 §3 三种标量+向量结合模式）。

**两段式检索（本客户端的主用法）：** 知识库按品类组织（每品类固定几类卡），检索的本质是
「定位品类」而非「检索卡片」。:meth:`resolve_category` 用 hybrid 命中按品类投票定位，
:meth:`fetch_cards` 再按品类 term 精确取全——跨品类污染、同类卡挤出 top-K 这两类全局
top-K 的老毛病从结构上消掉。裸 :meth:`search` 保留给投票内部与评测用。

**双后端（沿用 M3 解耦套路）：**

- 配了 ``OPENSEARCH_HOST`` → 走真 OpenSearch：``hybrid`` query + ``search_pipeline``
  在引擎层做 min_max 归一 + 加权融合（KNN 0.7 / BM25 0.3）。
- 没配 → 退化到**进程内本地 hybrid**：用 TowerClient 编码做语义召回 + token 重叠做
  全文召回，各自 min_max 归一后同权重融合——**和引擎层算的是同一个公式**，只是规模小、
  在本进程算。保证离线 / CI 不依赖 docker 也能跑通整条 RAG 链路。

**语料是英文（主动更正 refdocs）：** refdocs 假设中文卡片用 ``ik_max_word`` 分词；本项目
``data/rag`` 是英文 Amazon 商品，故 BM25 那一路用 OpenSearch 内置 ``english`` 分词器。
中文 query（如「旅行三件套」）跨语言匹配**靠 KNN 多语言向量兜底**——这正是 Hybrid 双路
互相代偿的设计意图（一路命不中，另一路补上）。
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from app.recall.category_kb import CategoryCard
from app.recall.towers import TowerClient, get_tower_client

logger = logging.getLogger("shoppingx.kb")

# OpenSearch 侧的固定命名（建库脚本与检索共用，避免两处写歪）。
INDEX_NAME = "globex_category_kb"
HYBRID_PIPELINE = "globex_hybrid_pipeline"
VECTOR_FIELD = "content_vector"

# 默认卡片落盘位置（ETL 产出，已 gitignore；本地后端从这里读）。
DEFAULT_CARDS_PATH = "./data/rag/category_cards.jsonl"

# 融合权重 [KNN, BM25]，对应 refdocs 13-1 §3 的引擎层 weights，本地后端复用同一组。
HYBRID_WEIGHTS = (0.7, 0.3)

# 两段式检索（品类定位 → 结构化取卡）的参数：
# 第一段只为「投票定品类」，不需要粗排 30 条——top-15 里的品类分布已足够投票。
RESOLVE_COARSE_K = 15
# 第二段按品类取卡的上限（一个品类当前 ≈8 张卡，32 给足余量）。
FETCH_SIZE = 32

# query 含这些「语义化 token」时关掉 BM25 子路：纯气质/口语 query 下 BM25 几乎全是
# 字面命中的杂卡，反把 KNN 准命中的卡挤出 Top-K（refdocs 13-1 §3.3）。
SEMANTIC_TOKENS = ("气质", "感觉", "风格", "氛围", "适合", "送", "vibe", "aesthetic", "minimal")


def should_disable_bm25(query: str) -> bool:
    """判定型分支：query 偏纯语义时关掉 BM25 子路，只走 KNN（大小写不敏感的子串匹配）。"""
    low = query.lower()
    return any(t in low for t in SEMANTIC_TOKENS)


def _min_max_norm(scores: list[float]) -> list[float]:
    """把一路原始分线性映射到 [0,1]（KNN 余弦与 BM25 量纲不同，融合前先各自归一）。"""
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if hi - lo < 1e-9:
        # 全相等（含单条候选 / 某路全 0，如中文 query 对英文卡的 BM25 全不命中）：这一路
        # 没有区分信号，归零让它**不贡献**融合分，把排序交给另一路——避免给所有候选注入
        # 一个常数把绝对分抬高、把短路用的首尾差距抹平。
        return [0.0 for _ in scores]
    return [(s - lo) / (hi - lo) for s in scores]


def _overlap_score(query: str, text: str) -> float:
    """本地 BM25 替身：query 与卡片文本的 token 交集数（min_max 前的原始分）。"""
    q = set(query.lower().split())
    t = set(text.lower().split())
    return float(len(q & t))


class KBClient:
    """品类知识库检索：对外只暴露一个 :meth:`search`，内部按配置选后端。

    构造参数全可选，缺省从 env 读；``cards`` / ``cards_path`` 用于测试时直接注入卡片
    （不依赖外部文件）。
    """

    def __init__(
        self,
        cards: list[CategoryCard] | None = None,
        cards_path: str | Path | None = None,
        host: str | None = None,
        tower: TowerClient | None = None,
    ) -> None:
        self._host = host if host is not None else os.environ.get("OPENSEARCH_HOST")
        self._tower = tower or get_tower_client()
        self._cards_path = Path(
            cards_path or os.environ.get("CATEGORY_CARDS_PATH", DEFAULT_CARDS_PATH)
        )
        # 本地后端的卡片缓存（远程后端不用）。
        self._cards: list[CategoryCard] | None = cards
        self._os_client: Any | None = None

    @property
    def remote(self) -> bool:
        """是否走真 OpenSearch（否则为进程内本地回退）。"""
        return bool(self._host)

    async def search(
        self, query: str, coarse_k: int, disable_bm25: bool | None = None
    ) -> list[tuple[CategoryCard, float]]:
        """Hybrid 召回：返回 ``(卡片, 融合分)`` 列表，按融合分降序，最多 ``coarse_k`` 条。

        ``disable_bm25=None`` 时按 :func:`should_disable_bm25` 自动判定。
        """
        if disable_bm25 is None:
            disable_bm25 = should_disable_bm25(query)
        if self.remote:
            return await self._search_remote(query, coarse_k, disable_bm25)
        return await self._search_local(query, coarse_k, disable_bm25)

    # ---------------------- 两段式：品类定位 + 结构化取卡 ----------------------
    async def resolve_category(self, query: str, top_n: int = 2) -> list[tuple[str, float]]:
        """第一段：定位 query 指向的品类。返回 ``(品类, 置信度)`` 降序，最多 ``top_n`` 个。

        用 hybrid 命中做**按品类投票**：同品类各卡的融合分求和（卡多的品类天然多票——
        同品类卡片互为佐证，这正是想要的），置信度 = 该品类得分在全部命中里的占比。
        比直接拿卡片 top-K 稳：单张跑题卡抢不动整个品类的票仓。
        """
        hits = await self.search(query, coarse_k=RESOLVE_COARSE_K)
        if not hits:
            return []
        votes: dict[str, float] = {}
        for card, score in hits:
            votes[card.category] = votes.get(card.category, 0.0) + score
        total = sum(votes.values())
        if total <= 0:
            # 全 0 分（如双路都无区分信号）：退化为按命中顺序均分票（极小概率路径）。
            return [(c, 1.0 / len(votes)) for c in list(votes)[:top_n]]
        ranked = sorted(votes.items(), key=lambda x: x[1], reverse=True)
        return [(cat, round(v / total, 3)) for cat, v in ranked[:top_n]]

    async def fetch_cards(self, category: str) -> list[CategoryCard]:
        """第二段：按已定位的品类**精确取全**该品类卡片（零漏召、零跨品类污染）。

        同 card_type 多卡按 confidence 降序，下游「只取首卡」的消费口径直接受益。
        """
        if self.remote:
            cards = await self._fetch_remote(category)
        else:
            cards = [c for c in await self._load_cards() if c.category == category]
        return sorted(cards, key=lambda c: c.confidence, reverse=True)[:FETCH_SIZE]

    # ---------------------- 本地回退后端 ----------------------
    async def _load_cards(self) -> list[CategoryCard]:
        # 首次：从文件读卡（注入的 cards 已在 __init__ 落到 self._cards，跳过读盘）。
        if self._cards is None:
            cards: list[CategoryCard] = []
            if self._cards_path.exists():
                for line in self._cards_path.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if line:
                        cards.append(CategoryCard.model_validate_json(line))
            else:
                logger.warning("品类卡片文件不存在：%s（本地后端将返回空召回）", self._cards_path)
            self._cards = cards
        # 补齐缺失向量：建库时一般已写入 content_vector；测试直接注入的卡可能没有，
        # 用同一个 TowerClient 即时编码摘要文本补上（一次性，补完后续 search 直接用）。
        missing = [c for c in self._cards if not c.content_vector]
        if missing:
            mat = await self._tower.encode_texts([c.search_text() for c in missing])
            for card, vec in zip(missing, mat, strict=True):
                card.content_vector = [float(x) for x in vec]
        return self._cards

    async def _search_local(
        self, query: str, coarse_k: int, disable_bm25: bool
    ) -> list[tuple[CategoryCard, float]]:
        cards = await self._load_cards()
        if not cards:
            return []
        qvec = await self._tower.encode_query(query)
        mat = np.asarray([c.content_vector for c in cards], dtype=np.float32)
        # 向量已 L2 归一化 → 内积即余弦（语义子路原始分）。
        knn_raw = (mat @ np.asarray(qvec, dtype=np.float32)).tolist()
        knn = _min_max_norm(knn_raw)

        if disable_bm25:
            fused = [(card, k) for card, k in zip(cards, knn, strict=True)]
        else:
            bm25_raw = [_overlap_score(query, c.search_text()) for c in cards]
            bm25 = _min_max_norm(bm25_raw)
            w_knn, w_bm25 = HYBRID_WEIGHTS
            denom = w_knn + w_bm25
            fused = [
                (card, (w_knn * k + w_bm25 * b) / denom)
                for card, k, b in zip(cards, knn, bm25, strict=True)
            ]
        fused.sort(key=lambda x: x[1], reverse=True)
        return fused[:coarse_k]

    # ---------------------- OpenSearch 后端 ----------------------
    def _client(self) -> Any:
        if self._os_client is None:
            from opensearchpy import OpenSearch  # 懒导入：本地后端不强依赖 opensearch-py

            self._os_client = OpenSearch(
                hosts=[
                    {"host": self._host, "port": int(os.environ.get("OPENSEARCH_PORT", "9200"))}
                ],
                http_auth=(
                    os.environ.get("OPENSEARCH_USER", "admin"),
                    os.environ.get("OPENSEARCH_PASS", "admin"),
                ),
                use_ssl=os.environ.get("OPENSEARCH_USE_SSL", "false").lower() == "true",
                verify_certs=False,
                ssl_show_warn=False,
            )
        return self._os_client

    def _hybrid_body(
        self, query: str, qvec: list[float], coarse_k: int, disable_bm25: bool
    ) -> dict:
        """组装 OpenSearch ``hybrid`` query：子路顺序必须与 pipeline weights 顺序一致。"""
        knn_q: dict[str, Any] = {"knn": {VECTOR_FIELD: {"vector": qvec, "k": coarse_k * 3}}}
        queries: list[dict[str, Any]] = [knn_q]
        if not disable_bm25:
            # aliases 权重介于 category 与 summary 之间：别名是品类锚点（强于 summary 的
            # 通用长串），但经 LLM 生成、可信度略逊标准品类名。
            queries.append(
                {
                    "multi_match": {
                        "query": query,
                        "fields": ["category^2", "aliases^1.5", "summary"],
                    }
                }
            )
        return {"size": coarse_k, "query": {"hybrid": {"queries": queries}}}

    async def _search_remote(
        self, query: str, coarse_k: int, disable_bm25: bool
    ) -> list[tuple[CategoryCard, float]]:
        qvec = [float(x) for x in await self._tower.encode_query(query)]
        # 关 BM25 后只剩单子路：**不能**再走 hybrid + pipeline——归一化管道的融合权重
        # 是两个（KNN/BM25），子路数与权重数不匹配直接 400（口语金标的语义 token query
        # 实测踩中）。退成裸 KNN 查询，分数即原始余弦，排序语义不变。
        if disable_bm25:
            body: dict[str, Any] = {
                "size": coarse_k,
                "query": {"knn": {VECTOR_FIELD: {"vector": qvec, "k": coarse_k * 3}}},
            }
            params: dict[str, str] = {}
        else:
            body = self._hybrid_body(query, qvec, coarse_k, disable_bm25)
            params = {"search_pipeline": HYBRID_PIPELINE}
        try:
            resp = self._client().search(index=INDEX_NAME, body=body, params=params)
        except Exception as exc:  # noqa: BLE001 —— OpenSearch 不可用不该让工具崩
            # refdocs §6.4：检索后端挂了不抛异常，返回空让上层给低置信度结果。
            logger.warning("OpenSearch 检索失败，降级为空召回：%s", exc)
            return []
        out: list[tuple[CategoryCard, float]] = []
        for hit in resp.get("hits", {}).get("hits", []):
            src = dict(hit.get("_source", {}))
            src.pop(VECTOR_FIELD, None)  # 向量不进结构化输出
            out.append((CategoryCard(**src), float(hit.get("_score", 0.0))))
        return out

    async def _fetch_remote(self, category: str) -> list[CategoryCard]:
        """term 精确取一个品类的全部卡片（不走 hybrid pipeline，纯结构化查询）。"""
        body = {"size": FETCH_SIZE, "query": {"term": {"category.raw": category}}}
        try:
            resp = self._client().search(index=INDEX_NAME, body=body)
        except Exception as exc:  # noqa: BLE001 —— 同 _search_remote：后端挂了降级为空
            logger.warning("OpenSearch 取卡失败，降级为空：%s", exc)
            return []
        cards: list[CategoryCard] = []
        for hit in resp.get("hits", {}).get("hits", []):
            src = dict(hit.get("_source", {}))
            src.pop(VECTOR_FIELD, None)
            cards.append(CategoryCard(**src))
        return cards

    async def aclose(self) -> None:
        """释放编码器连接（OpenSearch 客户端是同步的，无需 await 关闭）。"""
        await self._tower.aclose()


@lru_cache(maxsize=1)
def get_kb_client() -> KBClient:
    """进程内共享的知识库客户端（按 env 选 OpenSearch / 本地后端，主+子 Agent 复用）。"""
    return KBClient()
