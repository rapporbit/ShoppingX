"""品类知识库检索客户端（进程内 Hybrid：KNN + 词面重叠）。

**为什么不用搜索引擎（2026-09-22 主动更正原方案）：** 原方案让应用层走 OpenSearch，
理由是它把「语义召回(KNN) + 全文匹配(BM25) + 标量过滤 + 线性加权融合」装进同一套 DSL。
但本仓的知识库只有 1935 张品类卡，整个语料的向量矩阵不到 10MB，全量精确点积比 HNSW 近似
还准；两段式结构下指标只取决于「品类定位对不对」，而定位由 KNN 0.7 主导。40 条口语金标集
实测两种后端 recall / mrr / ndcg 三个数完全一致（0.800 / 0.800 / 0.769），于是把 OpenSearch
连同 512MB 堆的容器、断路器、灌库脚本一起删掉，只留这一条进程内路。

**两段式检索（本客户端的主用法）：** 知识库按品类组织（每品类固定几类卡），检索的本质是
「定位品类」而非「检索卡片」。:meth:`resolve_category` 用 hybrid 命中按品类投票定位，
:meth:`fetch_cards` 再按品类精确取全——跨品类污染、同类卡挤出 top-K 这两类全局
top-K 的老毛病从结构上消掉。裸 :meth:`search` 保留给投票内部与评测用。

**Hybrid 公式：** KNN 余弦与词面重叠各自 min_max 归一后按 :data:`HYBRID_WEIGHTS` 加权平均。
词面那一路是「query 与卡片文本的 token 交集数」，没有词干化 / IDF / 字段加权；字段加权由
:meth:`CategoryCard.search_text` 的「品类名重复一次 + 并入别名」代替。

**语料是英文：** ``data/rag`` 是英文 Amazon 商品卡。中文 query（如「旅行三件套」）词面路
基本不命中，跨语言匹配靠 KNN 多语言向量兜底——这正是 Hybrid 双路互相代偿的设计意图。
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path

import numpy as np

from app.recall.category_kb import CategoryCard
from app.recall.towers import TowerClient, get_tower_client

logger = logging.getLogger("shoppingx.kb")

# 默认卡片落盘位置（ETL 产出，已 gitignore）。
DEFAULT_CARDS_PATH = "./data/rag/category_cards.jsonl"

# 融合权重 [KNN, 词面]，沿用原方案的引擎层 weights。
HYBRID_WEIGHTS = (0.7, 0.3)

# 两段式检索（品类定位 → 结构化取卡）的参数：
# 第一段只为「投票定品类」，不需要粗排 30 条——top-15 里的品类分布已足够投票。
RESOLVE_COARSE_K = 15
# 第二段按品类取卡的上限（一个品类当前 ≈8 张卡，32 给足余量）。
FETCH_SIZE = 32

# query 含这些「语义化 token」时关掉词面子路：纯气质/口语 query 下词面几乎全是
# 字面命中的杂卡，反把 KNN 准命中的卡挤出 Top-K。
SEMANTIC_TOKENS = ("气质", "感觉", "风格", "氛围", "适合", "送", "vibe", "aesthetic", "minimal")


def should_disable_bm25(query: str) -> bool:
    """判定型分支：query 偏纯语义时关掉词面子路，只走 KNN（大小写不敏感的子串匹配）。"""
    low = query.lower()
    return any(t in low for t in SEMANTIC_TOKENS)


def _min_max_norm(scores: list[float]) -> list[float]:
    """把一路原始分线性映射到 [0,1]（KNN 余弦与词面计数量纲不同，融合前先各自归一）。"""
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if hi - lo < 1e-9:
        # 全相等（含单条候选 / 某路全 0，如中文 query 对英文卡的词面全不命中）：这一路
        # 没有区分信号，归零让它**不贡献**融合分，把排序交给另一路——避免给所有候选注入
        # 一个常数把绝对分抬高、把短路用的首尾差距抹平。
        return [0.0 for _ in scores]
    return [(s - lo) / (hi - lo) for s in scores]


def _overlap_score(query: str, text: str) -> float:
    """词面子路：query 与卡片文本的 token 交集数（min_max 前的原始分）。"""
    q = set(query.lower().split())
    t = set(text.lower().split())
    return float(len(q & t))


class KBClient:
    """品类知识库检索：对外暴露 :meth:`search` / :meth:`resolve_category` / :meth:`fetch_cards`。

    构造参数全可选，缺省从 env 读；``cards`` / ``cards_path`` 用于测试时直接注入卡片
    （不依赖外部文件）。
    """

    def __init__(
        self,
        cards: list[CategoryCard] | None = None,
        cards_path: str | Path | None = None,
        tower: TowerClient | None = None,
    ) -> None:
        self._tower = tower or get_tower_client()
        self._cards_path = Path(
            cards_path or os.environ.get("CATEGORY_CARDS_PATH", DEFAULT_CARDS_PATH)
        )
        self._cards: list[CategoryCard] | None = cards

    async def search(
        self, query: str, coarse_k: int, disable_bm25: bool | None = None
    ) -> list[tuple[CategoryCard, float]]:
        """Hybrid 召回：返回 ``(卡片, 融合分)`` 列表，按融合分降序，最多 ``coarse_k`` 条。

        ``disable_bm25=None`` 时按 :func:`should_disable_bm25` 自动判定。
        """
        if disable_bm25 is None:
            disable_bm25 = should_disable_bm25(query)
        cards = await self._load_cards()
        if not cards:
            return []
        qvec = await self._tower.encode_query(query)
        mat = np.asarray([c.content_vector for c in cards], dtype=np.float32)
        # 向量已 L2 归一化 → 内积即余弦（语义子路原始分）。全量精确计算，不做近似。
        knn_raw = (mat @ np.asarray(qvec, dtype=np.float32)).tolist()
        knn = _min_max_norm(knn_raw)
        if disable_bm25:
            fused = [(card, k) for card, k in zip(cards, knn, strict=True)]
        else:
            lex_raw = [_overlap_score(query, c.search_text()) for c in cards]
            lex = _min_max_norm(lex_raw)
            w_knn, w_lex = HYBRID_WEIGHTS
            denom = w_knn + w_lex
            fused = [
                (card, (w_knn * k + w_lex * b) / denom)
                for card, k, b in zip(cards, knn, lex, strict=True)
            ]
        fused.sort(key=lambda x: x[1], reverse=True)
        return fused[:coarse_k]

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
        cards = [c for c in await self._load_cards() if c.category == category]
        return sorted(cards, key=lambda c: c.confidence, reverse=True)[:FETCH_SIZE]

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
                logger.warning("品类卡片文件不存在：%s（知识库将返回空召回）", self._cards_path)
            self._cards = cards
        # 补齐缺失向量：建库时一般已写入 content_vector；测试直接注入的卡可能没有，
        # 用同一个 TowerClient 即时编码摘要文本补上（一次性，补完后续 search 直接用）。
        missing = [c for c in self._cards if not c.content_vector]
        if missing:
            mat = await self._tower.encode_texts([c.search_text() for c in missing])
            for card, vec in zip(missing, mat, strict=True):
                card.content_vector = [float(x) for x in vec]
        return self._cards

    async def aclose(self) -> None:
        """释放编码器连接。"""
        await self._tower.aclose()


@lru_cache(maxsize=1)
def get_kb_client() -> KBClient:
    """进程内共享的知识库客户端。"""
    return KBClient()
