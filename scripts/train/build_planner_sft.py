"""S2 冷启动数据：golden + 教师产出 → SFT 样本（ms-swift messages 格式）。

**为什么是「蒸馏 + golden 修正」而不是纯 golden**：S0-2 刻意没给 `keywords` / `exclude_terms`
标字面 golden（检索词天然多解，比字面就是在罚同义改写）。可 SFT 要有个具体的目标串才能训。
解法是分工——
- `category` / `domains` / `budget_*`：**用 golden 覆盖教师**。这三组有确定答案，教师答错的
  地方正是我们要纠的。
- `keywords` / `exclude_terms`：用**教师产出**（线上 API 模型）。它们没有唯一解，SFT 阶段
  只需要学会「长什么样」，学准是 S3 GRPO 用 R_retrieval 去顶的事。

这也正是 S2 的定位：**冷启动只学 schema 与形态**（验收线是格式正确率 ≥98%），不指望它学对
判定。把这两件事混在一起要，SFT 就会去死记 golden，反而压缩了 GRPO 的探索空间。

**只训三组字段，system prompt 也只讲这三组**：线上双通道方案里本地 4B 只出这几项，完整
2k prompt 里 tasks / retrieval / target_refs / bundle_slots 的大段说明它一句用不上。砍掉后
序列从约 2400 token 降到约 900，训练与 rollout 一起快 2.5 倍——这不是偷工，是与双通道
的职责划分保持一致。

用法：``uv run python scripts/train/build_planner_sft.py --limit 200``（先小批给 GPU profile）
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")

from app.memory.domains import ALL_DOMAINS, DOMAIN_LABELS  # noqa: E402

GOLDEN = PROJECT_ROOT / "data" / "train" / "planner_golden.jsonl"
OUT_DIR = PROJECT_ROOT / "data" / "train"
CONCURRENCY = 16
REQ_TIMEOUT = 120

# 训练专用精简 prompt：只讲本地 4B 负责的三组字段。**域清单必须原样保留**——它是封闭枚举，
# 少一项模型就可能造词，而域判错会让长期偏好在该轮静默失效（线上踩过）。
SFT_SYSTEM = """你是购物 Agent 的意图拆解器。把用户这轮的购物意图拆成结构化字段，输出 JSON。

字段：
- category：主品类，中文 2~10 字（「跑鞋」「保温杯」「笔记本电脑包」）。追问轮没换品类就
  沿用上一轮的；只给场景没给品类时按最贴近的品类假设填；纯闲聊留空串。
- domains：品类域，从下面封闭清单里挑，可多选，**按商品本体归域**，别被使用场景带偏
  （手表 → jewelry_watches，不是 furniture）。归不进具体域填 ["other"]。**绝不填 global**。
{menu}
- budget_amount：用户原话给的预算金额，**照原数填不要换算**，本轮没提就填 null。
- clear_budget：用户明确取消/放开预算（「不限预算」「贵点也行」）为 true，只是没提填 false。
- keywords：给商品检索用的英文关键词 2~6 个。**商品标题基本是英文**，中文词搜不到东西。
  别把整句塞进来，也别堆同义词。
- exclude_terms：用户本轮说的「不要 X」里的 X，每项 {{"word": 英文词, "evidence": 用户原话片段}}。
  evidence 必须是原话里真有的片段。弱表达（「不太喜欢」「尽量别」）不算，留空数组。

只输出 JSON，不要解释。"""

USER_TMPL = "{prior}本轮用户：{text}"


def build_system() -> str:
    menu = "\n".join(f"  - {d}：{DOMAIN_LABELS[d]}" for d in ALL_DOMAINS if d != "global")
    return SFT_SYSTEM.format(menu=menu)


async def _teacher(row: dict) -> dict | None:
    """教师产出：线上 API 模型跑一遍真实 planner，取它的 keywords / exclude_terms。

    复用**线上 prompt + schema**，不碰工具体（那里有 P_t 写入 / 计费 / AGUI 上报等会话副作用，
    批量跑会互相污染）——与 M21「训练与线上共用 embed_text、不共用会话层」同一条纪律。
    """
    from app.agent.invoke import call_structured
    from app.agent.llm import get_fast_llm
    from app.tools.planner import PlanOutput, get_planner_prompt

    prior = "".join(f"用户上一轮：{t}\n" for t in row.get("prior_turns") or [])
    # **必须重试**：首次全量跑（并发 16）教师失败 411/1621 = 25.4%，而 120 条 smoke 时是 0 ——
    # 典型的限流/超时，不是这些样本本身有问题。失败直接丢等于白扔四分之一训练集。
    for attempt in range(3):
        try:
            out = await asyncio.wait_for(
                call_structured(
                    get_fast_llm(),
                    [("system", get_planner_prompt()), ("user", prior + row["text"])],
                    PlanOutput,
                ),
                timeout=REQ_TIMEOUT,
            )
            return out.model_dump()
        except Exception:
            if attempt < 2:
                await asyncio.sleep(2 * (attempt + 1))  # 退避：限流下立刻重试只会继续撞墙
    return None


def _target(row: dict, teacher: dict) -> dict:
    """拼 SFT 目标：三组确定字段用 golden 覆盖教师，检索词形态用教师的。"""
    g = row["golden"]
    tgt = {
        # golden 弃权（None）的样本，退回教师值——**不能填 null 当目标**，那会教模型
        # 在「上文缺失」时输出空品类，而线上那一轮是带 P_t 上文调用的，本该继承品类。
        "category": g["category"]
        if g.get("category") is not None
        else (teacher.get("category") or ""),
        "domains": g["domains"] if g.get("domains") is not None else (teacher.get("domains") or []),
        "budget_amount": (
            teacher.get("budget_amount") if g.get("budget_uncertain") else g.get("budget_amount")
        ),
        "clear_budget": bool(g.get("clear_budget")),
        "keywords": [str(k) for k in (teacher.get("keywords") or [])][:6],
        "exclude_terms": [
            # 字段名必须是 word——线上 ExcludeTerm 就叫 word。第一版写成 term，教师产出被
            # 整批过滤成空（1621 条里 0 条带排除项），且真训出来字段名也对不上线上 schema，
            # 双通道合并会直接失效。训练与线上共用一份 schema，不是共用「差不多的」。
            {"word": str(t.get("word", "")), "evidence": str(t.get("evidence", ""))}
            for t in (teacher.get("exclude_terms") or [])
            if str(t.get("word", "")).strip()
        ],
    }
    return tgt


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="只导前 N 条（0=全量）")
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", default="")
    ap.add_argument("--resume", action="store_true", help="跳过已导好的 id，只补失败的")
    ap.add_argument(
        "--concurrency",
        type=int,
        default=CONCURRENCY,
        help="教师并发。首次全量跑 16 并发触发限流、失败 25%%，补跑时调小",
    )
    args = ap.parse_args()

    rows = [json.loads(x) for x in GOLDEN.open(encoding="utf-8") if x.strip()]
    rows = [r for r in rows if r["split"] == args.split]
    # review 状态的样本降权处理 = 直接排除：它们的 category/domains 三票没谈拢，拿去当
    # SFT 的确定目标不合适（S0-2 的决定：train 里的分歧不裁决，训练时排除）。
    rows = [r for r in rows if r["status"] != "review"]
    if args.limit:
        rows = rows[: args.limit]

    system, sem = build_system(), asyncio.Semaphore(args.concurrency)
    out_path = OUT_DIR / (args.out or f"planner_sft_{args.split}.jsonl")
    # resume：教师调用是花钱的，补跑失败样本时不该把已导好的重来一遍
    done_ids: set[str] = set()
    if args.resume and out_path.exists():
        done_ids = {json.loads(x)["id"] for x in out_path.open(encoding="utf-8") if x.strip()}
        rows = [r for r in rows if r["id"] not in done_ids]
        print(f"resume：已有 {len(done_ids)} 条，补 {len(rows)} 条")
    fh = out_path.open("a" if done_ids else "w", encoding="utf-8")
    # 教师产出**原样落盘**：这次因为一个字段名写错（term/word）就得把 1600 次 API 全重跑一遍。
    # 存下原始产出，以后改目标拼装只要重跑 _target，一分钱不用再花。
    raw_path = OUT_DIR / f"planner_teacher_raw_{args.split}.jsonl"
    raw_fh = raw_path.open("a" if done_ids else "w", encoding="utf-8")
    stat = {"ok": 0, "teacher_fail": 0}

    async def one(row: dict) -> None:
        async with sem:
            teacher = await _teacher(row)
        if teacher is None:
            stat["teacher_fail"] += 1
            return
        raw_fh.write(json.dumps({"id": row["id"], "teacher": teacher}, ensure_ascii=False) + "\n")
        prior = "".join(f"用户上一轮：{t}\n" for t in row.get("prior_turns") or [])
        fh.write(
            json.dumps(
                {
                    "id": row["id"],
                    "messages": [
                        {"role": "system", "content": system},
                        {
                            "role": "user",
                            "content": USER_TMPL.format(prior=prior, text=row["text"]),
                        },
                        {
                            "role": "assistant",
                            "content": json.dumps(_target(row, teacher), ensure_ascii=False),
                        },
                    ],
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        fh.flush()
        stat["ok"] += 1
        if stat["ok"] % 100 == 0:
            print(f"  已导 {stat['ok']}/{len(rows)}", flush=True)

    print(f"导 {len(rows)} 条（{args.split}，已排除 review），system prompt {len(system)} 字符")
    await asyncio.gather(*(one(r) for r in rows))
    fh.close()
    raw_fh.close()

    # 连 meta 一起导：GPU 机上没有本仓库的 app 包，格式检查器要有个地方读到「域枚举有哪些、
    # 该有哪些字段」。硬编码进那边的脚本就有两份事实来源，改了枚举必忘同步一处。
    meta = OUT_DIR / "planner_sft_meta.json"
    meta.write_text(
        json.dumps(
            {
                "system": system,
                "domains": [d for d in ALL_DOMAINS if d != "global"],
                "forbidden_domains": ["global"],
                "required_fields": [
                    "category",
                    "domains",
                    "budget_amount",
                    "clear_budget",
                    "keywords",
                    "exclude_terms",
                ],
                "field_types": {
                    "category": "str",
                    "domains": "list[str]",
                    "budget_amount": "float|null",
                    "clear_budget": "bool",
                    "keywords": "list[str]",
                    "exclude_terms": "list[{word,evidence}]",
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"完成：{stat}\n→ {out_path.relative_to(PROJECT_ROOT)}")
    print(f"→ {meta.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    asyncio.run(main())
