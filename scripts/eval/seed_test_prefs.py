"""给 Rubric 评测的测试 user 预置长期偏好，让「记忆注入」类 query（q12）评测有 ground truth。

q12「还是按我之前说的偏好，再帮我推荐两件家居好物」本身不带显式约束——靠注入的长期偏好兜。
没有预置偏好时，Agent 无从尊重、judge 的 P0「违背黑名单」也没有判定基准，这条评测恒判 0、无效。
本脚本给固定测试 user（``EVAL_USER_ID``）写入一条黑名单 + 两条家居 like 偏好；评测时用
``run_rubric.py --user-id eval_user`` 跑，q12 的偏好注入与红线判定才成立。

偏好是 read_relevant 按 query 语义召回的：家居/材质偏好对「买耳机」等无关 query 召不回，
故给所有条目统一用同一 ``--user-id`` 不会污染其他评测。

用法：
    uv run python scripts/eval/seed_test_prefs.py            # 写入
    uv run python scripts/eval/seed_test_prefs.py --clear    # 清掉（按 key 删除）
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.memory.store import PreferenceEntry, get_store  # noqa: E402

EVAL_USER_ID = "eval_user"

# 一条黑名单（dislike，带可硬过滤 keywords）+ 两条家居 like：让 q12 的「按我之前偏好」有内容，
# 且 P0「违背黑名单（推了塑料家居）」有判定基准。
SEED_PREFS = [
    {
        "slug": "plastic",
        "domain": "home",
        "content": "不接受塑料材质",
        "category": "material",
        "polarity": "dislike",
        "keywords": ["塑料", "plastic"],
    },
    {
        "slug": "natural_material",
        "domain": "home",
        "content": "偏好原木 / 藤编等自然材质的家居",
        "category": "material",
        "polarity": "like",
        "keywords": ["原木", "实木", "藤编", "rattan", "wood"],
    },
    {
        "slug": "niche_designer",
        "domain": "home",
        "content": "喜欢小众设计师品牌、不爱大路货",
        "category": "brand",
        "polarity": "like",
        "keywords": ["小众", "设计师", "designer"],
    },
]


async def main(clear: bool) -> None:
    store = get_store()
    entries = [PreferenceEntry.create(**p) for p in SEED_PREFS]  # type: ignore[arg-type]
    if clear:
        for e in entries:
            await store.delete(EVAL_USER_ID, e.dedup_key)
        print(f"已清除测试 user「{EVAL_USER_ID}」的 {len(entries)} 条预置偏好")
        return
    for e in entries:
        await store.write(EVAL_USER_ID, e)
    print(f"已为测试 user「{EVAL_USER_ID}」写入 {len(entries)} 条偏好：")
    for e in entries:
        print(f"  - [{e.polarity}/{e.category}] {e.content}（keywords={e.keywords}）")
    print(f"\n评测 q12 时用：uv run python scripts/eval/run_rubric.py --user-id {EVAL_USER_ID}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clear", action="store_true", help="删除预置偏好而非写入")
    args = parser.parse_args()
    asyncio.run(main(args.clear))
