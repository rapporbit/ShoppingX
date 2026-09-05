"""S2 spike：AgentScope 结构化输出对本仓真实 schema（PlanOutput）稳不稳。

手册 §6-L0 的 S2：对 LLM_MAIN 与「快档」（= LLM_MAIN + enable_thinking=False）
各跑 3 次，看 3/3 解析成功。记忆 structured-output-method-must-be-pinned 的坑是
「with_structured_output 默认 method 随模型浮动 → qwen 系走 json_object 打挂 planner」，
所以这里额外记录 AgentScope 实际用的策略（forced / auto / no_think / none）。

最后补一次 Agent.reply(structured_schema=...) 端到端，确认落在 msg.structured_output。

跑法：uv run python scripts/spikes/agentscope_s2_structured_output.py
"""

import asyncio
import json
import os
import time

from agentscope.agent import Agent
from agentscope.credential import OpenAICredential
from agentscope.message import Msg, TextBlock
from agentscope.model import OpenAIChatModel
from agentscope.tool import Toolkit
from dotenv import load_dotenv

load_dotenv()

from app.tools.planner import PlanOutput  # noqa: E402  (需先 load_dotenv)

QUERY = (
    "想买便宜又抗造的旅行三件套，预算 300，不要塑料的，喜欢小众牌子；"
    "顺便告诉我 amazon 和 shein 哪边便宜，含税含运到手多少钱。"
)


def _msg(role: str, text: str) -> Msg:
    return Msg(name=role, role=role, content=[TextBlock(type="text", text=text)])


def _model(no_think: bool) -> OpenAIChatModel:
    return OpenAIChatModel(
        credential=OpenAICredential(
            api_key=os.environ["OPENAI_API_KEY"],
            base_url=os.environ["OPENAI_BASE_URL"],
        ),
        model=os.environ.get("LLM_MAIN", "deepseek-v4-flash"),
        stream=True,
        extra_body={"enable_thinking": False} if no_think else None,
    )


async def _try_once(no_think: bool) -> dict:
    msgs = [
        _msg("system", "你是购物意图解析器，把用户需求拆成结构化字段，不臆造。"),
        _msg("user", QUERY),
    ]
    t0 = time.perf_counter()
    try:
        res = await _model(no_think).generate_structured_output(msgs, PlanOutput)
        # 结构化结果在 .content（不是 .metadata——踩过：空 dict 会被默认值验证成假通过）
        parsed = PlanOutput.model_validate(res.content)
        return {
            # 空壳不算过：PlanOutput 全字段有默认值，{} 也能 validate
            "ok": bool(parsed.tasks) and bool(parsed.category),
            "raw_keys": len(res.content),
            "seconds": round(time.perf_counter() - t0, 1),
            "tasks": [t.value if hasattr(t, "value") else t for t in parsed.tasks],
            "category": parsed.category,
            "retrieval": parsed.retrieval,
        }
    except Exception as exc:  # noqa: BLE001 — spike 要记录失败形态
        return {
            "ok": False,
            "seconds": round(time.perf_counter() - t0, 1),
            "error": f"{type(exc).__name__}: {exc}"[:300],
        }


async def _agent_path() -> dict:
    """端到端：Agent.reply(structured_schema=PlanOutput)。"""
    agent = Agent(
        name="planner_spike",
        system_prompt="你是购物意图解析器，拿到需求后立刻产出结构化结果，不要反问。",
        model=_model(no_think=True),
        toolkit=Toolkit(),
    )
    t0 = time.perf_counter()
    try:
        reply = await agent.reply(_msg("user", QUERY), structured_schema=PlanOutput)
        out = reply.structured_output
        parsed = PlanOutput.model_validate(out) if out else None
        return {
            "ok": bool(parsed and parsed.tasks and parsed.category),
            "seconds": round(time.perf_counter() - t0, 1),
            "structured_output_keys": sorted(out)[:8] if out else None,
            "tasks": (PlanOutput.model_validate(out).tasks if out else None),
            "category": (PlanOutput.model_validate(out).category if out else None),
        }
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:300]}


async def main() -> None:
    result = {"schema_fields": len(PlanOutput.model_fields)}
    for label, no_think in (("main_thinking", False), ("fast_no_think", True)):
        runs = [await _try_once(no_think) for _ in range(3)]
        result[label] = {"pass": sum(r["ok"] for r in runs), "runs": runs}
    result["agent_reply_path"] = await _agent_path()
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(main())
