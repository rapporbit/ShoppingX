"""LLM 合成 query：给没被 ESCI 覆盖的商品造训练对（refdocs 04-2 §5.5.2 五档生成）。

**为什么做这个**：三轮实验证明唯一有效的杠杆是「数据量/正例利用率」（正例展开 ×3.94 拿到
R@100 +11.7%），而展开已经到头。下一个数据来源只能是造——ESCI 只覆盖 12.67 万 ASIN，
占 138 万商品库的 9.2%，**剩下 90.8% 的商品从没进过训练数据**。

**语言是刻意分档的**（refdocs 没这么写，是对着我们自己的 train/serve skew 定的）：

- L1/L2（核心词、属性）→ **英文**：贴近 ESCI 评测集形态，涨的分能在现有尺子上量出来
- L3/L4（场景、口语）→ **中文**：贴近 ShoppingX 线上的真实输入（"便宜又抗造的旅行三件套"）

ESCI query 中位数只有 3 个词、是关键词搜索，而线上是自然语言购物意图——这个分布差是
「离线涨 11.7% 但线上可能不涨」的病根。L3/L4 是唯一能直接攻击它的手段，顺带还强化跨语言对齐。

**负例用随机采样，不挖 hard**：这不是偷懒，是实测结论——hard 占比 16.4% 的配方全面胜过
71.2%，hard negative 是拿主召回换近义干扰抑制的 trade-off 工具，不是提分工具。

用法：``uv run --group train python scripts/train/build_synth_queries.py --limit 10000``
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from app.agent.llm import get_llm  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data" / "train"
BATCH = 8
CONCURRENCY = 12
N_NEG = 5
REQ_TIMEOUT = 150  # 秒。没有它会死得很难看——见下方 synth() 注释

PROMPT = """你是电商搜索的数据标注员。下面每个商品，按四档分别生成 1 条用户可能输入的搜索词。

[L1] 英文·核心词：用户记得商品大致名称时怎么搜，2-5 个词，关键词形态
[L2] 英文·属性搜：用户记得 1-2 个属性但不确定品类时怎么搜
[L3] 中文·场景搜：用户从使用场景出发怎么搜，**不要出现品类词本身**
[L4] 中文·口语搜：用户用不完整、口语化的说法怎么搜，像在跟朋友描述

硬约束：
- 禁止照抄商品标题里连续 4 个词以上的片段
- 每条句式必须不同，不要套用「我想要 X 的 Y」这类模板
- 真实用户口吻，不要营销语、不要书面语
- L3/L4 必须是简体中文，L1/L2 必须是英文

严格输出 JSON 数组，每个元素形如 {{"L1":"...","L2":"...","L3":"...","L4":"..."}}，
数组长度必须等于商品数量，顺序一一对应。

商品列表：
{items}"""


def _parse(text: str, expect: int) -> list[dict] | None:
    m = re.search(r"\[.*]", text, re.S)
    if not m:
        return None
    try:
        arr = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(arr, list) or len(arr) != expect:
        return None
    if not all(isinstance(x, dict) and all(k in x for k in ("L1", "L2", "L3", "L4")) for x in arr):
        return None
    return arr


async def synth(items: list[dict], sink) -> int:
    """生成并**逐批落盘**。

    第一版栽了个跟头，两个缺陷叠在一起代价极大：``llm.ainvoke`` 没有超时，而结果要等
    ``asyncio.gather`` 全部返回才写文件。跑到 6400/10000 时 16 个并发槽全被挂起的请求占死
    （实测 16 条 TCP 连接全在等、进程 CPU 时间只有 11 秒），**gather 永远不会返回，2.6 小时
    的 API 调用一条都落不了盘**。

    所以现在：① 每个请求套 ``wait_for`` 超时；② 每批一完成立刻写文件，进程随时可杀可续。
    长跑任务只要没有增量落盘，任何一个挂起点都会让全部产出归零。
    """
    llm, sem = get_llm(), asyncio.Semaphore(CONCURRENCY)
    batches = [items[i : i + BATCH] for i in range(0, len(items), BATCH)]
    done_n = [0]

    async def one(batch: list[dict]) -> None:
        listing = "\n".join(f"{i + 1}. {it['text'][:220]}" for i, it in enumerate(batch))
        async with sem:
            for _ in range(2):
                try:
                    resp = await asyncio.wait_for(
                        llm.ainvoke(PROMPT.format(items=listing)), timeout=REQ_TIMEOUT
                    )
                except (TimeoutError, Exception):
                    continue
                arr = _parse(str(resp.content), len(batch))
                if arr:
                    sink([{"item": it, "q": a} for it, a in zip(batch, arr, strict=True)])
                    done_n[0] += len(batch)
                    if done_n[0] % 400 < BATCH:
                        print(f"  已生成 {done_n[0]}/{len(items)} 个商品", flush=True)
                    return

    await asyncio.gather(*(one(b) for b in batches))
    return done_n[0]


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=10000, help="采样多少个商品")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="synth_train.jsonl")
    args = ap.parse_args()

    # 排除 ESCI 已覆盖的商品——要造的正是那 90.8% 没进过训练数据的
    covered: set[str] = set()
    for name in ("esci_train.jsonl", "esci_eval_qrels.jsonl"):
        for line in (DATA_DIR / name).open(encoding="utf-8"):
            r = json.loads(line)
            covered.update(r.get("pos_ids") or [])
            covered.update(r.get("positives") or [])
    print(f"ESCI 已覆盖商品 {len(covered)} 个，从其余商品中采样")

    corpus = [json.loads(x) for x in (DATA_DIR / "corpus.jsonl").open(encoding="utf-8") if x.strip()]
    pool = [c for c in corpus if c["item_id"] not in covered]
    rng = random.Random(args.seed)
    rng.shuffle(pool)
    sample = pool[: args.limit]
    print(f"候选池 {len(pool)}，采样 {len(sample)} 个商品，开始生成…")

    texts = [c["text"] for c in corpus]
    out = DATA_DIR / args.out
    state = {"qid": 900_000_000, "rows": 0}
    fh = out.open("w", encoding="utf-8")

    def sink(packs: list[dict]) -> None:
        for p in packs:
            for lv in ("L1", "L2", "L3", "L4"):
                q = str(p["q"][lv]).strip()
                if not q:
                    continue
                state["qid"] += 1
                fh.write(json.dumps({
                    "query": q,
                    "query_id": state["qid"],
                    "pos": [p["item"]["text"]],
                    "pos_ids": [p["item"]["item_id"]],
                    "neg": [texts[rng.randrange(len(texts))] for _ in range(N_NEG)],
                    "level": lv,
                }, ensure_ascii=False) + "\n")
                state["rows"] += 1
        fh.flush()  # 每批刷盘，进程被杀也保得住已产出的部分

    ok = await synth(sample, sink)
    fh.close()
    print(f"\n生成成功 {ok}/{len(sample)} 个商品，"
          f"{state['rows']} 条训练对 → {out.name} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    asyncio.run(main())
