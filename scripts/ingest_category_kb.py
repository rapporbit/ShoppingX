"""把已产出的 category_cards.jsonl 直接灌进 OpenSearch（不重新抽卡、不重新编码）。

与 build_category_kb.py 的分工：后者是**全量离线流程**（读原始 CSV → 聚合抽卡 → 调 embedding
编码 → 写 JSONL → 灌库），跑一次要重刷全部向量，慢且花 API 钱，且依赖几百 MB 的原始 CSV。
部署到新机器时这些都不必重来——JSONL 里已经带着 content_vector，缺的只是「灌库」这一步。

用法（容器内）：python scripts/ingest_category_kb.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from app.recall.category_kb import CategoryCard  # noqa: E402
from app.recall.kb_client import DEFAULT_CARDS_PATH  # noqa: E402
from scripts.etl.os_setup import bulk_cards, make_client, recreate_index, register_pipeline  # noqa: E402


def main() -> None:
    if not os.environ.get("OPENSEARCH_HOST"):
        sys.exit("未配置 OPENSEARCH_HOST —— 本脚本只负责灌库，无 OpenSearch 可灌。")

    path = Path(os.environ.get("CATEGORY_CARDS_PATH", DEFAULT_CARDS_PATH))
    if not path.exists():
        sys.exit(f"卡片文件不存在：{path}")

    cards: list[CategoryCard] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cards.append(CategoryCard.model_validate(json.loads(line)))

    if not cards:
        sys.exit(f"{path} 里没有卡片。")

    # content_vector 是 model 的 exclude 字段，需从原始 JSON 并回（bulk_cards 要用它建 KNN 向量）。
    with path.open(encoding="utf-8") as f:
        for card, line in zip(cards, (ln for ln in f if ln.strip()), strict=True):
            card.content_vector = json.loads(line).get("content_vector") or []

    missing = [c.category for c in cards if not c.content_vector]
    if missing:
        sys.exit(f"{len(missing)} 张卡片缺 content_vector（如 {missing[:3]}），JSONL 不完整。")

    dim = len(cards[0].content_vector)
    print(f"读入 {len(cards)} 张卡片（dim={dim}），灌入 OpenSearch…")

    client = make_client()
    recreate_index(client, dim)
    register_pipeline(client)
    n = bulk_cards(client, cards)
    print(f"已索引 {n} 张 → {client.transport.hosts}")


if __name__ == "__main__":
    main()
