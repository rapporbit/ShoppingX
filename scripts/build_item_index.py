"""从干净商品表构建 Qdrant dense 召回库（BGE-M3 dense + payload）。

对齐 `docs/plans/召回引擎选型思路.md`：召回 = dense + filter（无 sparse 打分腿，§4）。
**主动更正 refdocs 04-1**：召回从 Faiss 升级到 Qdrant（理由见 `qdrant_store.py`）。流程：
读干净分平台表 → 拼 embed_text（title|brand|尾3类|描述边界截断 + 轻量归一）→ BGE-M3 编码
dense → 连同 payload upsert 进 Qdrant 单 collection。

编码器：配了 ``EMBED_MODEL`` 走真实 embedding（带 TPM/RPM 限流 + 重试）；否则本地确定性回退
（离线可跑、维度 256）。Qdrant 后端由 env 决定（server / on-disk 本地 / 内存），见 qdrant_store。

前置：先 ``uv run python scripts/clean_platforms.py`` 生成干净表。

用法：
    uv run python scripts/build_item_index.py                 # 全部 5 平台
    uv run python scripts/build_item_index.py amazon          # 单平台（冒烟）
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from collections.abc import Iterator
from itertools import islice
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# 离线建库脚本不经 app.agent.llm 那条链，必须自己加载 .env，否则 EMBED_* 读不到、
# TowerClient 静默退化成本地哈希回退——会把「假向量」灌进 Qdrant。
load_dotenv()

from app.recall.fx import to_base_or_none  # noqa: E402
from app.recall.qdrant_store import QdrantRecall  # noqa: E402
from app.recall.schemas import ItemRecord  # noqa: E402
from app.recall.text import clip_sentence, tail_category  # noqa: E402
from app.recall.towers import DEFAULT_LOCAL_DIM, TowerClient  # noqa: E402
from app.utils.clean import PLATFORMS, CleanItem, clean_text  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLEAN_DIR = PROJECT_ROOT / "data" / "platforms" / "clean" / "by_platform"
MANIFEST_PATH = PROJECT_ROOT / "data" / "index" / "qdrant_manifest.json"

DESC_EMBED_CLIP = 300  # 描述参与编码的截断长度（边界截断，防半句）
EMBED_BATCH = 64
# 流式建库：一个 chunk 编码完立刻 upsert 再丢弃，峰值内存 ≈ CHUNK_RECORDS × dim × 4B。
# 4096 × 1024 × 4B ≈ 16MB；全量 137 万条不再需要 5.4GB × 2 的 vstack 峰值。
CHUNK_RECORDS = 4096
ENCODE_CONCURRENCY = 4
# embedding API 限流：TPM 是活约束（RPM 在 batch=64 下远不触发）。用满额度的 50%（留 50% 余量）。
MAX_TPM = 500_000
MAX_RPM = 2_000
_RATE_MARGIN = 0.5
_RATE_WINDOW = 60.0


class _RateLimiter:
    """滑动 60s 窗口节流：同时盯 token 与请求两个上限，发批前等到窗口腾出。"""

    def __init__(self) -> None:
        self._cap_tok = MAX_TPM * _RATE_MARGIN
        self._cap_req = MAX_RPM * _RATE_MARGIN
        self._events: list[tuple[float, int]] = []  # (ts, tokens)

    def _reserve(self, tokens: int) -> float:
        now = time.monotonic()
        self._events = [e for e in self._events if now - e[0] < _RATE_WINDOW]
        cur_tok = sum(n for _, n in self._events)
        if cur_tok + tokens <= self._cap_tok and len(self._events) + 1 <= self._cap_req:
            self._events.append((now, tokens))
            return 0.0
        # 等到最旧事件移出窗口再重试登记。
        return max(_RATE_WINDOW - (now - self._events[0][0]), 0.0) if self._events else 0.0

    async def acquire(self, tokens: int) -> None:
        # 限流器本就要「睡到窗口腾出再重试登记」，while+sleep 是正确形态。
        while (wait := self._reserve(tokens)) > 0:  # noqa: ASYNC110
            await asyncio.sleep(wait)


def _embed_text(item: CleanItem) -> str:
    """dense 编码文本：title | brand | 尾3类 | 描述(边界截断)，整串轻量归一。"""
    parts = [
        item.title,
        item.brand,
        tail_category(item.category),
        clip_sentence(item.description, DESC_EMBED_CLIP),
    ]
    composed = " | ".join(p for p in parts if p)
    return clean_text(composed, strip_html=False)


def _clean_to_record(item: CleanItem) -> ItemRecord:
    return ItemRecord(
        item_id=item.item_id,
        platform=item.platform,
        title=item.title,
        brand=item.brand,
        price=item.price,
        currency=item.currency,
        rating=item.rating,
        reviews_count=item.reviews_count,
        category=item.category,  # 全 breadcrumb（payload 用；尾3类只在检索文本里取）
        url=item.url,
        image_url=item.image_url,
        price_usd=to_base_or_none(item.price, item.currency, "USD"),
        embed_text=_embed_text(item),
    )


def _platform_paths(platform: str) -> list[Path]:
    # amazon 额外并入 RAG 扩充源（amazon_rag.jsonl），平台名同为 amazon；
    # 缺该文件不报错（没跑 sample_rag_amazon.py 时退化为仅 CSV 源）。
    paths = [CLEAN_DIR / f"{platform}.jsonl"]
    if platform == "amazon":
        paths.append(CLEAN_DIR / "amazon_rag.jsonl")
    return [p for p in paths if p.exists()]


def _count_platform(platform: str) -> int:
    """预扫行数：只为算进度和 ETA，不构造对象。"""
    return sum(
        sum(1 for ln in p.open(encoding="utf-8") if ln.strip()) for p in _platform_paths(platform)
    )


def _iter_platform(platform: str) -> Iterator[ItemRecord]:
    """逐行流式产出 ItemRecord。

    全量 amazon 有 137 万行，一次性 load 成 pydantic 对象要几 GB；这里边读边吐，
    调用方按 chunk 消费完即弃，内存占用与语料规模无关。
    """
    for path in _platform_paths(platform):
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield _clean_to_record(CleanItem.model_validate_json(line))


def _chunks(it: Iterator[ItemRecord], size: int) -> Iterator[list[ItemRecord]]:
    while chunk := list(islice(it, size)):
        yield chunk


async def _encode_records(
    records: list[ItemRecord], tower: TowerClient, limiter: _RateLimiter
) -> np.ndarray:
    """编码一个 chunk：内部按 EMBED_BATCH 切子批，最多 ENCODE_CONCURRENCY 个在飞。

    并发只在「未被 TPM 限流卡住」时才有收益——限流器是全局的，真被 TPM 顶住时
    并发批会一起排队，不会超发。
    """
    sem = asyncio.Semaphore(ENCODE_CONCURRENCY)

    async def _one(batch: list[str]) -> np.ndarray:
        async with sem:
            if tower.remote:
                # token 保守上估：字符数/3（英文偏高 → 安全）。
                await limiter.acquire(sum(len(t) for t in batch) // 3)
            return await tower.encode_texts(batch)

    batches = [
        [r.embed_text for r in records[i : i + EMBED_BATCH]]
        for i in range(0, len(records), EMBED_BATCH)
    ]
    vectors = await asyncio.gather(*(_one(b) for b in batches))
    return np.ascontiguousarray(np.vstack(vectors), dtype=np.float32)


async def main(platforms: list[str], require_remote: bool = False) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    tower = TowerClient()
    recall = QdrantRecall()
    limiter = _RateLimiter()
    print(f"编码器：{'远程真实 embedding' if tower.remote else f'本地确定性回退 dim={tower.dim}'}")
    # 与 build_category_kb 同款门禁：建生产库时拒绝静默退化成本地哈希回退，否则把假向量灌进 Qdrant。
    if require_remote and not tower.remote:
        raise SystemExit(
            "❌ --require-remote：EMBED_MODEL 未生效，当前是本地哈希回退。"
            "拒绝把假向量灌进 Qdrant。\n"
            "   检查 .env 的 EMBED_MODEL / EMBED_BASE_URL / EMBED_API_KEY 是否已配置且被加载。"
        )

    todo = {p: n for p in platforms if (n := _count_platform(p))}
    if not todo:
        print("没有可建索引的平台，退出。")
        await tower.aclose()
        return
    total = sum(todo.values())

    # 先用一条样本探真实维度：全量 amazon 137 万条，不能等整平台编完才建库。
    probe = next(_iter_platform(next(iter(todo))))
    dim = int((await _encode_records([probe], tower, limiter)).shape[1])
    if require_remote and dim == DEFAULT_LOCAL_DIM:
        raise SystemExit(
            f"❌ --require-remote：向量维度={dim} 等于本地回退维度，疑似静默回退，已中止。"
        )
    recall.ensure_collection(dim, recreate=True)
    print(f"collection 已重建：dim={dim}，向量 on-disk；待灌 {total:,} 条")

    counts: dict[str, int] = {}
    start_id = 0
    done = 0
    t0 = time.monotonic()
    for platform in todo:
        for chunk in _chunks(_iter_platform(platform), CHUNK_RECORDS):
            dense = await _encode_records(chunk, tower, limiter)
            recall.upsert(chunk, dense, start_id=start_id)
            start_id += len(chunk)
            done += len(chunk)
            counts[platform] = counts.get(platform, 0) + len(chunk)
            el = time.monotonic() - t0
            eta = (total - done) / (done / el) if done else 0
            print(
                f"  [{platform}] {done:,}/{total:,} "
                f"({done / total:.1%})  {done / el:.0f} 条/s  ETA {eta / 60:.0f} min",
                flush=True,
            )
    await tower.aclose()

    if not counts:
        print("没有可建索引的平台，退出。")
        return

    manifest = {
        "model": tower._model or "local",  # noqa: SLF001 (建索引留痕，serve 启动校验用)
        "dim": dim,
        "embed": "remote" if tower.remote else "local",
        "platforms": sorted(counts),
        "counts": counts,
        "points": sum(counts.values()),
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"完成：{sum(counts.values())} 条商品 → Qdrant collection，清单 {MANIFEST_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="构建 Qdrant hybrid 召回库")
    parser.add_argument("platforms", nargs="*", help="平台名（默认全部 5 平台）")
    parser.add_argument(
        "--require-remote",
        action="store_true",
        help="强制走真实 embedding：退化到本地哈希回退则直接报错（建生产库时用）",
    )
    args = parser.parse_args()
    asyncio.run(main(args.platforms or list(PLATFORMS), require_remote=args.require_remote))
