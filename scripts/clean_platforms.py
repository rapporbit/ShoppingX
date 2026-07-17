"""前置清洗步骤：把 5 平台原始 CSV 洗成统一干净商品表（可复现，产物 gitignore）。

读取 ``data/platforms/*.csv`` → 调 :mod:`app.utils.clean`（映射 + 归一 + 严格质量闸 +
平台内去重）→ 落两份产物，并打印每平台「进 / 出 / 各类丢弃」报告：

- ``data/platforms/clean/products.jsonl`` —— 合表（一行一条 :class:`CleanItem`），通用加载入口。
- ``data/platforms/clean/by_platform/{platform}.jsonl`` —— 分平台表（读取便利）；
  ``build_item_index.py`` 从这批表逐平台编码、灌进 Qdrant 单 collection（平台维度靠 payload
  过滤区分，非物理分库——见 ``app/recall/qdrant_store.py``）。

用法：
    uv run python scripts/clean_platforms.py            # 全部 5 平台
    uv run python scripts/clean_platforms.py amazon     # 指定平台
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.utils.clean import PLATFORMS, CleanItem, clean_platform  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PLATFORM_DIR = PROJECT_ROOT / "data" / "platforms"
OUT_DIR = PLATFORM_DIR / "clean"
OUT_FILE = OUT_DIR / "products.jsonl"
BY_PLATFORM_DIR = OUT_DIR / "by_platform"

# CSV 单元格内嵌大段 JSON / 评论，默认字段上限会炸；调高（不用 maxsize 防 OverflowError）。
csv.field_size_limit(10 * 1024 * 1024)

# 丢弃原因的展示顺序（与 clean.admit / is_available 的 reason 对齐）。
_DROP_KEYS = (
    "drop_unavailable",
    "drop_no_title",
    "drop_no_id",
    "drop_no_price",
    "drop_bad_price",
    "drop_no_currency",
    "drop_no_image",
    "drop_no_url",
    "drop_dup",
)


def _read_csv(platform: str) -> list[dict[str, str]]:
    path = PLATFORM_DIR / f"{platform}-products.csv"
    if not path.exists():
        print(f"  [跳过] 找不到 {path}")
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_jsonl(path: Path, items: list[CleanItem]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item.model_dump(), ensure_ascii=False) + "\n")


def main(platforms: list[str]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    BY_PLATFORM_DIR.mkdir(parents=True, exist_ok=True)
    all_items: list[CleanItem] = []
    all_stats: dict[str, dict[str, int]] = {}

    for platform in platforms:
        rows = _read_csv(platform)
        if not rows:
            continue
        items, stats = clean_platform(platform, rows)
        all_items.extend(items)
        all_stats[platform] = stats
        # 分平台表：对齐召回层「按平台分库」，供 build_item_index.py 逐平台建索引。
        _write_jsonl(BY_PLATFORM_DIR / f"{platform}.jsonl", items)

    # 合表：通用加载入口（一份全平台 JSONL）。
    _write_jsonl(OUT_FILE, all_items)

    # —— 报告 ——
    print(f"\n{'平台':<10}{'读入':>7}{'保留':>7}{'留存率':>8}   丢弃明细")
    print("-" * 72)
    for platform, st in all_stats.items():
        read, kept = st.get("read", 0), st.get("kept", 0)
        rate = f"{100 * kept / read:.0f}%" if read else "—"
        drops = "  ".join(f"{k.removeprefix('drop_')}={st[k]}" for k in _DROP_KEYS if st.get(k))
        print(f"{platform:<10}{read:>7}{kept:>7}{rate:>8}   {drops or '（无丢弃）'}")
    print("-" * 72)
    total_read = sum(s.get("read", 0) for s in all_stats.values())
    print(f"{'合计':<10}{total_read:>7}{len(all_items):>7}")
    print(f"\n干净表 → {OUT_FILE}")
    print(f"分平台 → {BY_PLATFORM_DIR}/{{platform}}.jsonl（{len(all_stats)} 个平台）")


if __name__ == "__main__":
    targets = sys.argv[1:] or list(PLATFORMS)
    main(targets)
