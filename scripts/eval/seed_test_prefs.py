"""给 Rubric 评测的测试 user 预置长期记忆，让「记忆注入」类 query（q12）评测有 ground truth。

q12「还是按我之前说的偏好，再帮我推荐两件家居好物」本身不带显式约束——靠注入的长期记忆兜。
没有预置事实时，Agent 无从尊重、judge 的 P0「违背硬规则」也没有判定基准，这条评测恒判 0、无效。
本脚本给固定测试 user（``EVAL_USER_ID``）写入一条 constraint + 两条 preference；评测时用
``run_rubric.py --user-id eval_user`` 跑，记忆注入与红线判定才成立。

**没有域过滤了**（M4）：注入的是 tier-one 那批（全部 constraint + 最近几条其余事实），
不再按 planner 判的品类域筛。所以这里每条只写一份，而不是像旧版那样同一条在两个域各种一份。
代价是这三条在「买耳机」等无关 query 上也会注入——这正是参考实现的口径：事实是默认值，
与本轮无关时模型自己会忽略它。

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

from app.memory.fact_store import get_fact_store  # noqa: E402
from app.memory.facts import validate_fact  # noqa: E402

EVAL_USER_ID = "eval_user"

# 一条 constraint（每轮必注入，P0「推了塑料家居」的判定基准）+ 两条 preference。
# key 用英文小写下划线，与 save_memory / curator 的产物同形态；value 写成几个月后单看也成立的句子。
SEED_FACTS = [
    ("material_avoid", "不接受塑料材质的家居用品", "constraint"),
    ("material_taste", "偏好原木 / 实木 / 藤编等自然材质", "preference"),
    ("brand_taste", "喜欢小众设计师品牌，不爱大路货", "preference"),
]


async def main(clear: bool) -> None:
    store = get_fact_store()
    facts = [validate_fact(k, v, c) for k, v, c in SEED_FACTS]
    if clear:
        for f in facts:
            await store.delete_fact(EVAL_USER_ID, f.key)
        print(f"已清除测试 user「{EVAL_USER_ID}」的 {len(facts)} 条预置记忆")
        return
    await store.upsert_facts(EVAL_USER_ID, facts)
    print(f"已为测试 user「{EVAL_USER_ID}」写入 {len(facts)} 条事实：")
    for f in facts:
        print(f"  - [{f.category.value}] {f.key}: {f.value}")
    print(f"\n评测 q12 时用：uv run python scripts/eval/run_rubric.py --user-id {EVAL_USER_ID}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clear", action="store_true", help="删除预置记忆而非写入")
    args = parser.parse_args()
    asyncio.run(main(args.clear))
