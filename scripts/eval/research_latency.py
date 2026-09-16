"""C2 遗留的延迟账：research 一次 vs 模型自己多调几次裸 web_search，到底谁慢、贵多少上下文。

``RESEARCH_SEARCH_QUOTA=6`` 是按**上下文体量**推的（research 返回 schema，裸搜返回整页正文，
单次上下文增量约 1/10），当时没测延迟——万一 research 因为多一次归纳解码而明显更慢，这个配额
就该按别的量级定。这个脚本把两条路各跑一遍、把数摆出来。

**口径（不要读成 research 赢了多少）**：
- 两边都只测**工具侧**墙钟。research 是「并行搜 N 条 + 一次 fast 模型归纳」，一次调用测完；
  裸 web_search 这边只算 N 次搜索本身，**不含主环往返**——模型每调一次 web_search 就要多解码
  一轮才能发下一条，那部分（每轮数秒）全算在 web_search 头上却没计进来。所以这里的对比对
  裸搜有利，research 的真实优势比数字更大。
- 上下文增量按**返回给主环的文本字符数**算：research 是那段 JSON（正文只进归纳模型），
  web_search 是截断后的整页正文。

跑法（打 Tavily + 一次 fast 模型，约 $0.001 / 遍）：
    uv run python scripts/eval/research_latency.py
"""

from __future__ import annotations

import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from app.tools.research import RESEARCH_RESULTS_PER_TARGET, research  # noqa: E402
from app.tools.web_search import search_web  # noqa: E402
from app.utils.path_utils import ensure_session_dir  # noqa: E402
from app.utils.thread_ctx import thread_scope  # noqa: E402

TARGETS = ["Sony WH-1000XM5", "Bose QuietComfort 45", "Sennheiser Momentum 4"]
ASPECTS = ["降噪", "佩戴舒适度", "续航"]
#: 裸 web_search 这边发什么查询——照真实会话里模型自己写的形态（换措辞重搜同一件事）。
BARE_QUERIES = [
    "Sony WH-1000XM5 vs Bose QuietComfort 45 降噪 对比",
    "Sennheiser Momentum 4 降噪 佩戴 评测",
    "头戴降噪耳机 2025 推荐 续航对比",
]
ROUNDS = 2


async def _run_research() -> tuple[float, int, int]:
    t0 = time.perf_counter()
    out = await research(targets=list(TARGETS), aspects=list(ASPECTS))
    wall = time.perf_counter() - t0
    text = out.model_dump_json() if hasattr(out, "model_dump_json") else str(out)
    return wall, len(text), int(getattr(out, "searched", 0))


async def _run_parallel_search() -> float:
    """research 搜索段的代理：同样 3 条查询并行发，不归纳。

    用来把 research 的墙钟拆成「搜索段 + 归纳解码段」——不拆的话，看到 research 比裸搜慢只会
    得出「有界函数更慢」的错结论，实际慢的是那次归纳解码，而它换回的是 1/5 的上下文。
    """
    t0 = time.perf_counter()
    await asyncio.gather(*(search_web(q, RESEARCH_RESULTS_PER_TARGET) for q in BARE_QUERIES))
    return time.perf_counter() - t0


async def _run_bare() -> tuple[float, int, int]:
    """串行发——模型只能一轮发一条、看完再想下一条，这就是它的真实形态。"""
    t0 = time.perf_counter()
    chars = 0
    for q in BARE_QUERIES:
        ws, _ = await search_web(q, RESEARCH_RESULTS_PER_TARGET)
        chars += len(ws.model_dump_json())
    return time.perf_counter() - t0, chars, len(BARE_QUERIES)


async def main() -> None:
    rows: list[dict] = []
    for i in range(ROUNDS):
        sd = ensure_session_dir(f"latency_research_{i}")
        with thread_scope(f"latency_research_{i}", sd):
            r_wall, r_chars, r_searched = await _run_research()
            par_wall = await _run_parallel_search()
            b_wall, b_chars, b_searched = await _run_bare()
        rows.append(
            {
                "round": i + 1,
                "research_s": round(r_wall, 1),
                "research_search_s": round(par_wall, 1),
                "research_summarize_s": round(r_wall - par_wall, 1),
                "research_chars": r_chars,
                "research_searches": r_searched,
                "bare_s": round(b_wall, 1),
                "bare_chars": b_chars,
                "bare_searches": b_searched,
            }
        )
        print(json.dumps(rows[-1], ensure_ascii=False))

    med = lambda k: statistics.median(r[k] for r in rows)  # noqa: E731
    print("\n中位数：")
    print(
        f"  research  {med('research_s'):.1f}s"
        f"（搜索段 {med('research_search_s'):.1f}s + 归纳段 {med('research_summarize_s'):.1f}s）"
        f" / {med('research_chars'):.0f} 字符进主环"
    )
    print(f"  裸 3 搜   {med('bare_s'):.1f}s / {med('bare_chars'):.0f} 字符进主环（不含主环往返）")
    print(f"  上下文比  research = 裸搜的 {med('research_chars') / max(1, med('bare_chars')):.2f}x")
    print(f"  延迟比    research = 裸搜的 {med('research_s') / max(0.01, med('bare_s')):.2f}x")
    print(
        "  补一句口径：裸搜那 3 次各占一轮主环往返（实测单轮约 7s，见 snapshot_runs.jsonl），"
        "把它算进来才是模型自由发挥的真实代价。"
    )


if __name__ == "__main__":
    asyncio.run(main())
