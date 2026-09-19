"""把旧 `preferences` 表搬进 `memory_facts`（M1）。

映射规则（计划 §3.3 / §4.1 C1）：

| 旧 | 新 |
| --- | --- |
| `blocking=True` 或 `domain=global` | `category=constraint` |
| `category=location` 且 `polarity=like` | `category=context`，key 固定 `default_ship_to` |
| 其余 | `category=preference` |
| key | `{category}_{slug}`，slug 空则退回 `{category}_{域}` |
| value | `content` |

**旧表不删**：这个脚本只读它、只写新表，`preferences` 原样留着作回滚依据。

**逐条过 `validate_fact`**：被 PII 过滤器拒的记一行日志（只报 key，不回显 value）后跳过——
迁移不是绕开写入单门的后门，老库里躺着的手机号同样不该进新表。

同 key 冲突时**后写覆盖先写**，所以按 `last_confirmed_at` 升序搬，让最新确认的那条赢。

用法：
    uv run python scripts/migrate_preferences_to_facts.py            # 预演，不写库
    uv run python scripts/migrate_preferences_to_facts.py --apply    # 真写
    uv run python scripts/migrate_preferences_to_facts.py --apply --user-id u1
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import select

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# 脚本不走 FastAPI 的 lifespan，DATABASE_URL 要自己从 .env 读，否则会连到内存库、搬完等于没搬。
load_dotenv()

from app.db.models import Preference  # noqa: E402
from app.db.session import init_db, session_factory  # noqa: E402
from app.memory.fact_store import get_fact_store  # noqa: E402
from app.memory.facts import MemoryWriteRejected, validate_fact  # noqa: E402

logger = logging.getLogger("migrate_prefs")

# C1：收货国原先靠 `category == "location" and polarity == "like"` 被 planner 第 3 层读到。
# 新模型里它必须落在一个**固定 key** 上，否则 planner 查不到就静默退到默认国，到手价整个算错。
SHIP_TO_KEY = "default_ship_to"


def _to_fact_args(row: Preference) -> tuple[str, str, str]:
    """一行旧偏好 → (key, value, category)。"""
    if row.category == "location" and row.polarity == "like":
        return SHIP_TO_KEY, row.content, "context"
    slug = (row.slug or "").strip() or (row.domain or "other")
    key = f"{row.category}_{slug}"
    category = "constraint" if (row.blocking or row.domain == "global") else "preference"
    return key, row.content, category


async def migrate(apply: bool, user_id: str | None) -> Counter[str]:
    stats: Counter[str] = Counter()
    store = get_fact_store()
    async with session_factory()() as db:
        stmt = select(Preference).order_by(Preference.last_confirmed_at.asc())
        if user_id:
            stmt = stmt.where(Preference.user_id == user_id)
        rows = list((await db.execute(stmt)).scalars())

    for row in rows:
        stats["read"] += 1
        key, value, category = _to_fact_args(row)
        try:
            fact = validate_fact(key, value, category, source_session=row.source_session or "")
        except MemoryWriteRejected:
            # 只报 key 与用户，不回显 value——被拒的正是不该扩散的内容。
            logger.warning("拒绝迁移（user=%s，key=%s）：命中写入过滤器", row.user_id, key)
            stats["rejected"] += 1
            continue
        stats[f"cat:{fact.category.value}"] += 1
        if apply:
            await store.upsert_facts(row.user_id, [fact])
            stats["written"] += 1
        else:
            logger.info(
                "预演 user=%s %s → [%s] %s",
                row.user_id,
                row.dedup_key,
                fact.category.value,
                fact.key,
            )
    return stats


async def main() -> None:
    parser = argparse.ArgumentParser(description="preferences → memory_facts 迁移")
    parser.add_argument("--apply", action="store_true", help="真写库；不给就是预演")
    parser.add_argument("--user-id", default=None, help="只搬这一个用户")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    await init_db()
    stats = await migrate(args.apply, args.user_id)
    mode = "已写入" if args.apply else "预演（未写库）"
    logger.info("%s：%s", mode, dict(sorted(stats.items())))


if __name__ == "__main__":
    asyncio.run(main())
