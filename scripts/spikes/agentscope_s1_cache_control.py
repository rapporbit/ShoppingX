"""S1 spike：AgentScope 的 formatter 加 cache_control 能否透传到网关并命中缓存。

手册 §6-L0 的 S1：子类化 OpenAIChatFormatter.format()，把前 N 条消息的 content
转成带 ``{"cache_control": {"type": "ephemeral"}}`` 的 content-block，用 .env 的
OPENAI_BASE_URL 发两次同前缀请求，看 usage 里的 cache_input_tokens。

A/B 两组各用不同 nonce 前缀，避免互相命中：
  A = 带 cache_control 的 formatter；B = 原生 formatter（对照隐式缓存）。

跑法：uv run python scripts/spikes/agentscope_s1_cache_control.py
"""

import asyncio
import json
import os
import uuid
from typing import Any

from agentscope.credential import OpenAICredential
from agentscope.formatter import OpenAIChatFormatter
from agentscope.message import Msg, TextBlock
from agentscope.model import OpenAIChatModel
from dotenv import load_dotenv

load_dotenv()

_EPHEMERAL = {"type": "ephemeral"}
# DashScope 显式缓存有最小写入阈值，前缀要够长（本仓 M6.1 口径）
_FILLER = (
    "你是 ShoppingX 的购物助理，负责跨平台检索、比价、算到手价并给出选购理由。"
    "以下是平台与品类的背景知识，需逐条遵守，不得省略任何一条约束。"
) * 40


class CacheAwareOpenAIFormatter(OpenAIChatFormatter):
    """在 format() 产出的 dict 上打 cache_control（断点前最后一条）。"""

    async def format(self, msgs: list[Msg]) -> list[dict[str, Any]]:
        formatted = await super().format(msgs)
        if not formatted:
            return formatted
        target = formatted[0]  # 只标 system（= 断点前最后一条）
        content = target.get("content")
        if isinstance(content, str):
            target["content"] = [
                {"type": "text", "text": content, "cache_control": dict(_EPHEMERAL)},
            ]
        elif isinstance(content, list) and content:
            content[-1] = {**content[-1], "cache_control": dict(_EPHEMERAL)}
        return formatted


def _msg(role: str, text: str) -> Msg:
    return Msg(name=role, role=role, content=[TextBlock(type="text", text=text)])


def _build_model(formatter: OpenAIChatFormatter) -> OpenAIChatModel:
    return OpenAIChatModel(
        credential=OpenAICredential(
            api_key=os.environ["OPENAI_API_KEY"],
            base_url=os.environ["OPENAI_BASE_URL"],
        ),
        model=os.environ.get("LLM_MAIN", "deepseek-v4-flash"),
        stream=True,
        formatter=formatter,
    )


async def _one_call(model: OpenAIChatModel, msgs: list[Msg]) -> dict:
    # 注意：model.__call__ 内部会自己调 formatter.format(msgs)，此处传 Msg 列表
    last = None
    async for res in await model(msgs):
        last = res
    usage = last.usage if last else None
    return {
        "input_tokens": getattr(usage, "input_tokens", None),
        "cache_input_tokens": getattr(usage, "cache_input_tokens", None),
        "output_tokens": getattr(usage, "output_tokens", None),
    }


async def _round(label: str, formatter: OpenAIChatFormatter) -> dict:
    nonce = uuid.uuid4().hex
    model = _build_model(formatter)
    msgs = [
        _msg("system", f"[{label}-{nonce}] {_FILLER}"),
        _msg("user", "只回一个字：好"),
    ]
    first = await _one_call(model, msgs)
    await asyncio.sleep(3)  # 给网关落缓存的时间
    second = await _one_call(model, msgs)
    return {"first": first, "second": second}


async def main() -> None:
    # 先确认带 cache_control 的 payload 长什么样（透传形态）
    probe = await CacheAwareOpenAIFormatter().format(
        [_msg("system", "abc"), _msg("user", "hi")],
    )
    result = {
        "payload_shape": probe[0],
        "A_with_cache_control": await _round("A", CacheAwareOpenAIFormatter()),
        "B_plain": await _round("B", OpenAIChatFormatter()),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(main())
