"""跑 Agent 级 Rubric 评测：种子集每条 query → run_agent → judge 打分 → 汇总报告。ROADMAP M11。

这是「上下文级飞轮」的评测腿（无 GPU）：跑一遍拿到基线分 + bad case 清单，改 prompt/工具/few-shot 后
**重跑同一种子集**对照分数，确认是真改进而非回退。种子集是稳定回归基线（见 build_eval_queries.py），
不随意改动。

产出：
- 终端报告：逐条 PASS/FAIL + 总分，分桶均分，P0 红线失败清单，bad case 列表。
- ``data/eval/rubric_report.json``（gitignore）：完整结构化结果，供飞轮检索 bad case / 沉淀示例。

用法：
    uv run python scripts/eval/run_rubric.py                 # 跑全集（串并发 3）
    uv run python scripts/eval/run_rubric.py --only q02_budget_trap_earbuds,q04_counterfeit_refuse
    uv run python scripts/eval/run_rubric.py --concurrency 1 --limit 3   # 慢跑小样调试
    uv run python scripts/eval/run_rubric.py --gate          # 有 P0 红线失败则退出码 1（回归门禁）

注意：真实 LLM 单条耗时可观（见延迟治理记录），全集串行很慢，默认并发 3；按配额调 --concurrency。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sqlalchemy import delete  # noqa: E402

from app.agent.orchestrator import run_agent  # noqa: E402
from app.agent.tracing import flush_traces  # noqa: E402
from app.db.models import Message  # noqa: E402
from app.db.session import session_factory  # noqa: E402
from app.eval.rubric import RubricResult, evaluate  # noqa: E402
from app.recall.semantic_cache import turn_cache_enabled  # noqa: E402

# 单条 query 的墙钟上限（含所有铺垫轮 + 打分）。取 15 分钟：实测最慢的全链路 case 也在 4 分钟内，
# 这个值只兜「永远回不来」的死挂，正常慢 case 碰不到。
#
# 为什么必须有：``_eval_one`` 的 except 只收得住**异常**，收不住**挂起**。2026-09-05 留迁移基线时
# 15 条里有 1 条卡住不动（0% CPU、无网络、29 分钟无任何写盘），而报告是 ``gather`` 全部返回后
# 才落盘的——一条挂起把另外 14 条已跑完的结果连同 token 一起赔光。有了这道闸，挂起降级成单条失败。
EVAL_QUERY_TIMEOUT_SEC = 900


async def _reset_thread(thread_id: str) -> None:
    """评测线程回零：清 DB 对话历史 + 清 session_dir（候选池 / 产物）。

    种子集用**稳定 thread_id**（产物可回溯），代价是「续聊」上线后状态跨评测运行存活：
    上一次基线跑的对话历史会被 load_prior_turns 读回、候选池被 load_candidates 读回，
    第二次跑同一条 query 时 planner 可能因「手上有候选」判成 reuse——基线就不可复现了。
    每条 query 开跑前回零，保证永远从干净的第 1 轮开始；多轮 case（"turns"）同理。
    """
    async with session_factory()() as db:
        await db.execute(delete(Message).where(Message.thread_id == thread_id))
        await db.commit()
    shutil.rmtree(Path("output") / thread_id, ignore_errors=True)


QUERIES_PATH = Path("data/eval/queries.jsonl")
REPORT_PATH = Path("data/eval/rubric_report.json")

#: 查后端整轮缓存状态的地址。评测**在本进程内直接调 run_agent**，所以本进程的开关才是决定性的；
#: 这一项是给「后端另起一个进程、评测只是它的客户端」那种用法留的补充探测，配了才查。
HEALTH_URL_ENV = "EVAL_HEALTH_URL"


def _assert_turn_cache_off() -> None:
    """整轮缓存开着就**拒跑**，而不是打条警告了事。

    开着跑出来的分数是「上一次那份的复读」：改了 prompt 也看不出差别，bad case 修没修全靠猜，
    而且**全程零报错**——评测数据被污染却看起来一切正常，是这类基建里最贵的一种失败。宁可让人
    多敲一次 ``TURN_CACHE_ENABLED=0``。
    """
    if turn_cache_enabled():
        raise SystemExit(
            "拒跑：TURN_CACHE_ENABLED 开着，整轮结果缓存会让评测复读上一次的答案。"
            "请先关掉（TURN_CACHE_ENABLED=0）再跑。"
        )
    url = os.environ.get(HEALTH_URL_ENV, "").strip()
    if not url:
        return
    try:
        import httpx

        data = httpx.get(url, timeout=5.0).json()
    except Exception as exc:  # noqa: BLE001 —— 探不到就别拦，本进程那道判据已经把住了主路
        print(f"[warn] 查 {url} 失败（{type(exc).__name__}: {exc}），跳过后端缓存探测")
        return
    if (data.get("turn_cache") or {}).get("enabled"):
        raise SystemExit(f"拒跑：{url} 显示后端整轮缓存开着，评测会拿到缓存结果。")


def _load_queries(only: set[str] | None, limit: int | None) -> list[dict]:
    if not QUERIES_PATH.exists():
        raise SystemExit(f"找不到种子集 {QUERIES_PATH}，请先跑 scripts/eval/build_eval_queries.py")
    rows = [
        json.loads(line)
        for line in QUERIES_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if only:
        rows = [r for r in rows if r["id"] in only]
    if limit:
        rows = rows[:limit]
    return rows


def _prior_context(q: dict) -> str:
    """这条 case 要告诉 judge 的跨轮 / 跨会话事实。

    judge 只看最后一轮 query（多轮 case 只对最后一轮打分），所以「第二个我要了」「把上次那单取消
    了」在它眼里没有主语——不给背景，它会把「引用上一轮清单里的真货」判成「凭空编造 item_id」。

    两级：种子集显式写的 ``prior_context`` 优先；没写的多轮 case 用**用户前几轮的原话**兜底。
    兜底刻意只用用户说过的话，**不含 Agent 的实际回答**——尺子必须与 Agent 表现无关，否则 Agent
    变差时尺子跟着变松，回归对照就废了。
    """
    explicit = str(q.get("prior_context") or "").strip()
    if explicit:
        return explicit
    turns: list[str] = q.get("turns") or []
    if len(turns) <= 1:
        return ""
    said = "\n".join(f"- 第 {i} 轮用户说：{t}" for i, t in enumerate(turns[:-1], 1))
    return f"本轮之前，同一会话里已经发生过以下几轮对话（Agent 均已正常作答）：\n{said}"


async def _eval_one(q: dict, user_id: str | None, sem: asyncio.Semaphore, use_cache: bool) -> dict:
    """跑一条 query 并打分；异常**与挂起**都收成一条「评测失败」记录，不拖垮整批。"""
    async with sem:
        qid = q["id"]

        async def _run_and_score() -> tuple[dict, RubricResult]:
            # 稳定 thread_id：产物落 output/eval_<id>/，可复现、可回溯轨迹。开跑前回零。
            thread_id = f"eval_{qid}"
            await _reset_thread(thread_id)
            # 多轮 case（"turns"）：前几轮只铺垫上下文（同 thread 续聊），只对最后一轮打分。
            # 回归的就是「追问轮换品类」这类跨轮交互（2026-07-14 线上死锁属于此类）。
            turns: list[str] = q.get("turns") or [q["query"]]
            for warmup in turns[:-1]:
                await run_agent(warmup, thread_id=thread_id, user_id=user_id)
            run = await run_agent(turns[-1], thread_id=thread_id, user_id=user_id)
            scored: RubricResult = await evaluate(
                turns[-1],
                run,
                q.get("constraints"),
                q.get("intent", "shopping"),
                use_cache,
                prior_context=_prior_context(q),
            )
            return run, scored

        try:
            run, result = await asyncio.wait_for(_run_and_score(), timeout=EVAL_QUERY_TIMEOUT_SEC)
            verdict = "PASS" if result.overall_pass else "FAIL"
            print(f"  [done] {qid:32s} {verdict} {result.total:5.1f}")
            return {
                "id": qid,
                "bucket": q.get("bucket", ""),
                "ok": True,
                # 落进报告，让 bad case 能回溯到那条 Langfuse trace（分数已作为 score 挂在上面）。
                "trace_id": run.get("trace_id"),
                "result": result.model_dump(),
            }
        except TimeoutError:
            # 单独一支：挂起和「工具报错」不是一回事，报告里要能一眼分开——前者是待查的 bug，
            # 后者多半是这条 case 自己的问题。
            msg = f"超过单条上限 {EVAL_QUERY_TIMEOUT_SEC}s 未返回（疑似挂起）"
            print(f"  [timeout] {qid:32s} {msg}")
            return {
                "id": qid,
                "bucket": q.get("bucket", ""),
                "ok": False,
                "error": f"EvalTimeout: {msg}",
            }
        except Exception as exc:  # noqa: BLE001 —— 单条失败不该中断整批评测
            print(f"  [error] {qid:32s} {type(exc).__name__}: {exc}")
            return {
                "id": qid,
                "bucket": q.get("bucket", ""),
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }


def _print_report(records: list[dict]) -> None:
    done = [r for r in records if r["ok"]]
    errored = [r for r in records if not r["ok"]]

    print("\n" + "=" * 72)
    print(f"Rubric 评测报告：{len(done)} 条完成 / {len(errored)} 条失败 / 共 {len(records)} 条")
    print("=" * 72)

    if done:
        passed = [r for r in done if r["result"]["overall_pass"]]
        avg = sum(r["result"]["total"] for r in done) / len(done)
        print(f"P0 通过率：{len(passed)}/{len(done)}（红线没破）｜平均质量分：{avg:.1f}/100")

        # 分桶均分：定位「哪类场景在掉分」。
        buckets: dict[str, list[float]] = {}
        for r in done:
            buckets.setdefault(r["bucket"], []).append(r["result"]["total"])
        print("\n分桶均分：")
        for b, ts in sorted(buckets.items(), key=lambda kv: sum(kv[1]) / len(kv[1])):
            print(f"  {b:16s} {sum(ts) / len(ts):5.1f}  (n={len(ts)})")

        # bad case：P0 破 或 总分偏低，按分升序——飞轮优先修这些。
        bad = sorted(
            [r for r in done if not r["result"]["overall_pass"] or r["result"]["total"] < 60],
            key=lambda r: r["result"]["total"],
        )
        if bad:
            print(f"\nbad case（{len(bad)} 条，优先修）：")
            for r in bad:
                res = r["result"]
                flags = []
                if res["p0_failures"]:
                    flags.append("P0破:" + "/".join(res["p0_failures"]))
                if res["p1_violations"]:
                    flags.append("P1:" + "/".join(res["p1_violations"]))
                print(f"  {r['id']:32s} {res['total']:5.1f}  {' '.join(flags) or '低质量分'}")

    if errored:
        print("\n评测失败（基建/调用问题，非模型表现）：")
        for r in errored:
            print(f"  {r['id']:32s} {r['error']}")


async def main(
    only: set[str] | None,
    limit: int | None,
    concurrency: int,
    user_id: str | None,
    gate: bool,
    use_cache: bool,
) -> int:
    _assert_turn_cache_off()
    queries = _load_queries(only, limit)
    cache_note = "复用缓存细则" if use_cache else "刷新细则缓存"
    print(f"开跑 Rubric 评测：{len(queries)} 条 query，并发 {concurrency}，{cache_note}\n")

    sem = asyncio.Semaphore(concurrency)
    try:
        records = await asyncio.gather(*(_eval_one(q, user_id, sem, use_cache) for q in queries))
    finally:
        # Langfuse 靠后台线程批量上报，本进程跑完就退——不 flush 的话，evaluate 里刚挂上的
        # rubric score 会连同队列一起消失（现象：trace 有、score 没有，且全程零报错）。
        # 放 finally：Ctrl-C / 半途异常时，已经打完分的那几条也别白丢。
        flush_traces()

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    _print_report(records)
    print(f"\n完整结果已落 {REPORT_PATH}")

    if gate:
        red_broken = any(r["ok"] and not r["result"]["overall_pass"] for r in records)
        if red_broken:
            print("\n[gate] 存在 P0 红线失败，退出码 1")
            return 1
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", type=str, default="", help="只跑这些 id（逗号分隔）")
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 条")
    parser.add_argument("--concurrency", type=int, default=3, help="并发条数（按 LLM 配额调）")
    parser.add_argument(
        "--user-id", type=str, default=None, help="评测用 user_id（记忆注入类 query 需预置偏好）"
    )
    parser.add_argument("--gate", action="store_true", help="有 P0 红线失败则退出码 1（回归门禁）")
    parser.add_argument(
        "--refresh-rubric",
        action="store_true",
        help="忽略并覆盖细则缓存（改了评测口径/生成 prompt 后用，重建尺子）",
    )
    args = parser.parse_args()
    only = {s.strip() for s in args.only.split(",") if s.strip()} or None
    raise SystemExit(
        asyncio.run(
            main(
                only,
                args.limit,
                args.concurrency,
                args.user_id,
                args.gate,
                use_cache=not args.refresh_rubric,
            )
        )
    )
