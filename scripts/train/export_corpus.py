"""导出全库语料（item_id + 编码文本），供 GPU 机器做训后评测。

评测必须在**全库 137 万**里检索才有意义（口径见 ``eval_recall.py``），而训完的新模型在 huzhou 上，
那台机器没有商品库也没有 Qdrant。与其把 5.6 GB 的向量传来传去，不如把 200 MB 的文本传过去，在
A100 上重编 + 暴力检索——137 万 × 1024 维的矩阵乘对 A100 来说是小活，比搬向量省事得多。

文本用 ``embed_text``，与线上索引、与训练集三处同源。

用法：``uv run --group train python scripts/train/export_corpus.py``
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from app.recall.text import embed_text  # noqa: E402
from app.utils.clean import CleanItem  # noqa: E402

CLEAN_DIR = PROJECT_ROOT / "data" / "platforms" / "clean" / "by_platform"
OUT_PATH = PROJECT_ROOT / "data" / "train" / "corpus.jsonl"


def main() -> None:
    n = 0
    seen: set[str] = set()
    with OUT_PATH.open("w", encoding="utf-8") as out:
        for path in sorted(CLEAN_DIR.glob("*.jsonl")):
            for line in path.open(encoding="utf-8"):
                if not line.strip():
                    continue
                try:
                    item = CleanItem(**json.loads(line))
                except Exception:
                    continue
                text = embed_text(item)
                if not text or item.item_id in seen:
                    continue
                seen.add(item.item_id)
                out.write(
                    json.dumps({"item_id": item.item_id, "text": text}, ensure_ascii=False) + "\n"
                )
                n += 1
            print(f"  {path.name}: 累计 {n}")
    print(f"\n共 {n} 条，{OUT_PATH.stat().st_size / 1e6:.1f} MB → {OUT_PATH}")


if __name__ == "__main__":
    main()
