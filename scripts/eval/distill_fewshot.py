"""从 Rubric 评测的高分轨迹自动蒸馏 few-shot 示例——上下文级飞轮第三腿的闭环。

飞轮第三腿原本是「高分轨迹 → 沉淀 few-shot（in-context，不改权重）」。人工种子（``few_shot.yml``）
开了头，本脚本把它**自动化**：读评测报告里 ``is_high_score`` 的条目，取其成功轨迹（query + 工具
调用序列 + 最终回复摘要），用 judge 模型蒸馏成精短的「✅正确决策 / ❌常见反例」范式，写入
``few_shot_distilled.yml``（与人工种子分文件，``fewshot.py`` 加载时按 intent 去重合并）。

闭环：跑评测（run_rubric）→ 高分轨迹 → 本脚本蒸馏 → 注入 system prompt → 再评测看是否抬分。

用法：
    uv run python scripts/eval/distill_fewshot.py                 # dry-run，打印蒸馏候选
    uv run python scripts/eval/distill_fewshot.py --write         # 落盘 few_shot_distilled.yml
    uv run python scripts/eval/distill_fewshot.py --top-n 3 --report data/eval/rubric_report.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from langchain_core.messages import AIMessage, messages_from_dict  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from app.eval.rubric import extract_tool_calls  # noqa: E402

REPORT_PATH = Path("data/eval/rubric_report.json")
DISTILLED_PATH = Path("prompt/few_shot_distilled.yml")


class FewShotExample(BaseModel):
    intent: str = Field(description="一句话场景标签，如「跨平台比价·先澄清型号」")
    text: str = Field(description="精短范式：✅正确决策 / ❌常见反例，2-4 行，照搬人工种子的口吻")


class _FewShotSet(BaseModel):
    examples: list[FewShotExample] = Field(default_factory=list)


_DISTILL_PROMPT = """你在为电商购物 Agent 提炼 few-shot 范例。下面是若干**评测高分轨迹**\
（每条含用户 query、Agent 的工具调用序列、最终回复摘要）——它们是「该照着做」的成功样本。

请提炼出最多 {n} 条**精短**的理想范式示例，要求：
- 每条 intent 是一句话场景标签；text 用「✅正确：…\\n❌反例：…」两三行，点出**决策方式**\
（何时直接执行、何时收尾、调哪些工具），不要复述具体商品。
- 覆盖不同场景、彼此去重；优先提炼能教会模型「信息够就直接检索别过度澄清」「item_picker 后真调\
shopping_summary 别口头收尾」「非购物直接兜底」这类高频要点。

高分轨迹：
{traces}

只输出一个 json 对象，形如：
{{"examples": [{{"intent": "场景标签", "text": "✅正确：…\\n❌反例：…"}}]}}
不要输出 json 以外的任何文字。"""


def _load_high_score_traces(report_path: Path) -> list[dict]:
    """从评测报告读 is_high_score 条目，配上其轨迹摘要（工具序列 + 最终回复片段）。"""
    if not report_path.exists():
        raise SystemExit(f"找不到评测报告 {report_path}，请先跑 scripts/eval/run_rubric.py")
    records = json.loads(report_path.read_text(encoding="utf-8"))
    traces: list[dict] = []
    for r in records:
        if not r.get("ok") or not r["result"].get("is_high_score"):
            continue
        qid = r["id"]
        hist = Path(f"output/eval_{qid}/history.json")
        tools: list[str] = []
        final = ""
        if hist.exists():
            msgs = messages_from_dict(json.loads(hist.read_text(encoding="utf-8")))
            tools = [c["name"] for c in extract_tool_calls(msgs)]
            final = next(
                (m.text for m in reversed(msgs) if isinstance(m, AIMessage) and m.text), ""
            )
        traces.append(
            {
                "id": qid,
                "query": r["result"].get("query", ""),
                "total": r["result"]["total"],
                "tools": tools,
                "final_head": final[:200],
            }
        )
    return traces


def _render_traces(traces: list[dict]) -> str:
    blocks = []
    for t in traces:
        blocks.append(
            f"- query：{t['query']}（得分 {t['total']}）\n"
            f"  工具序列：{' → '.join(t['tools']) or '（无）'}\n"
            f"  最终回复摘要：{t['final_head']}"
        )
    return "\n".join(blocks)


async def _distill(traces: list[dict], top_n: int) -> list[FewShotExample]:
    from app.agent.llm import get_judge_llm

    structured = get_judge_llm().with_structured_output(_FewShotSet, method="json_mode")
    result = await structured.ainvoke(
        _DISTILL_PROMPT.format(n=top_n, traces=_render_traces(traces))
    )
    fs = result if isinstance(result, _FewShotSet) else _FewShotSet.model_validate(result)
    return fs.examples[:top_n]


def _write_distilled(examples: list[FewShotExample]) -> None:
    payload = {"examples": [{"intent": e.intent, "text": e.text.rstrip() + "\n"} for e in examples]}
    header = (
        "# 自动蒸馏的 few-shot 示例（scripts/eval/distill_fewshot.py 从评测高分轨迹生成）。\n"
        "# 勿手改——会被下次蒸馏覆盖；人工种子写 few_shot.yml。fewshot.py 按 intent 去重合并。\n"
    )
    DISTILLED_PATH.write_text(
        header + yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )


async def main(report_path: Path, top_n: int, write: bool) -> None:
    traces = _load_high_score_traces(report_path)
    if not traces:
        print(f"报告 {report_path} 里没有 is_high_score 的高分轨迹——先跑出一些高分条目再蒸馏。")
        return
    print(f"读到 {len(traces)} 条高分轨迹：{', '.join(t['id'] for t in traces)}")
    examples = await _distill(traces, top_n)
    print(f"\n蒸馏出 {len(examples)} 条 few-shot 候选：\n")
    print(
        yaml.safe_dump(
            {"examples": [e.model_dump() for e in examples]}, allow_unicode=True, sort_keys=False
        )
    )
    if write:
        _write_distilled(examples)
        print(f"已写入 {DISTILLED_PATH}（fewshot.py 会按 intent 去重合并进 system prompt）")
    else:
        print("（dry-run，未落盘；加 --write 写入 few_shot_distilled.yml）")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=str, default=str(REPORT_PATH), help="评测报告路径")
    parser.add_argument("--top-n", type=int, default=5, help="最多蒸馏几条示例")
    parser.add_argument(
        "--write", action="store_true", help="落盘 few_shot_distilled.yml（默认 dry-run）"
    )
    args = parser.parse_args()
    asyncio.run(main(Path(args.report), args.top_n, args.write))
