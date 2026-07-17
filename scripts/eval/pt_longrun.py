"""P_t 长程对抗会话：一条 20 轮的真实会话打满换题 / 撤回 / 升级 / 预算变化，逐轮 dump P_t 形态。

执行计划 §5 的「长程对抗会话」腿：M1~M5 都太短（2~3 轮），epoch 误判频率 / 归并误并率 /
死约束堆积这几个只有拉长了才暴露。观测点：早轮约束是否全程存续（I1）、换题是否精确清代
不误清（I3）、约束集是否只增不减地堆脏、误并有无发生。

用法：
    LLM_REQUEST_TIMEOUT=300 uv run python scripts/eval/pt_longrun.py [thread_id]

产物：逐轮快照打到 stdout（JSONL），并落 output/<thread_id>/pt_trace.jsonl 供肉眼审。
"""

from __future__ import annotations

import asyncio
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sqlalchemy import delete  # noqa: E402

from app.agent.main_agent import run_agent  # noqa: E402
from app.db.models import Message  # noqa: E402
from app.db.session import session_factory  # noqa: E402

# 20 轮剧本：保温杯（1~6，含撤回粉色+软升硬花哨）→ 换题双肩包（7~12，含放宽预算+撤回皮革）
# → 换题机械键盘（13~20，含 clear_budget+撤回白色）。每段都有「补充轮」验存续。
TURNS = [
    "想买不锈钢保温杯，不要粉色的，预算150",
    "最好能保温12小时以上",
    "尽量别太花哨",
    "算了，粉色的也可以接受",
    "还是说死吧：绝对不要花哨的",
    "他们的到手价是多少？",
    "保温杯不买了，看看通勤双肩包吧，预算400",
    "要能装16寸笔记本的",
    "不要皮革的",
    "预算提到600吧",
    "算了，皮革的也行",
    "要防泼水的",
    "换个方向，想要机械键盘，预算50美元",
    "要红轴的",
    "不限预算了，直接上最好的",
    "尽量别太重",
    "最好带蓝牙",
    "不要白色的",
    "想了想白色其实也行",
    "就这样，给我最终推荐吧",
]


def _snapshot(session_dir: Path, turn_no: int, query: str) -> dict:
    pt_path = session_dir / "pt.json"
    pt = json.loads(pt_path.read_text(encoding="utf-8")) if pt_path.exists() else {}
    return {
        "turn": turn_no,
        "query": query,
        "epoch": pt.get("epoch"),
        "budget_usd": pt.get("budget_usd"),
        "category": pt.get("category"),
        "active": [
            f"[{c['id']}]{'硬' if c.get('blocking') else '软'}"
            f"{'-' if c.get('polarity') == 'dislike' else '+'} {c.get('content', '')}"
            for c in pt.get("constraints", [])
        ],
        "archived": [c["id"] for c in pt.get("archived", [])],
        "next_id": pt.get("next_id"),
    }


async def main() -> None:
    thread_id = sys.argv[1] if len(sys.argv) > 1 else "pt_longrun"
    session_dir = Path("output") / thread_id
    # 回零（同 run_rubric._reset_thread）：清对话历史 + 会话产物，保证从干净第 1 轮开始。
    async with session_factory()() as db:
        await db.execute(delete(Message).where(Message.thread_id == thread_id))
        await db.commit()
    shutil.rmtree(session_dir, ignore_errors=True)

    snaps: list[dict] = []
    for i, q in enumerate(TURNS, 1):
        try:
            await run_agent(q, thread_id=thread_id)
        except Exception as exc:  # noqa: BLE001 —— 单轮失败不终止剧本，记录后继续
            print(f"!! turn {i} 失败：{exc}", flush=True)
        snap = _snapshot(session_dir, i, q)
        snaps.append(snap)
        print(json.dumps(snap, ensure_ascii=False), flush=True)

    (session_dir / "pt_trace.jsonl").write_text(
        "\n".join(json.dumps(s, ensure_ascii=False) for s in snaps), encoding="utf-8"
    )
    print(f"\n轨迹已落 {session_dir / 'pt_trace.jsonl'}（{len(snaps)} 轮）")


if __name__ == "__main__":
    asyncio.run(main())
