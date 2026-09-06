"""ETL 共享的 LLM 构造与并发限流（llm_attributes / llm_aliases / shopify_attrs 共用）。

建库走便宜快模型（``LLM_FAST``，缺省回落 ``LLM_MAIN``）；信号量全 ETL 共享，
避免多个生成阶段叠加把 API rate limit 打爆。

模型对象是 AgentScope 的（批 0 / L7）：调用一律经 :mod:`app.agent.invoke` 的
``call_text`` / ``call_structured``，别自己 ``await model(...)`` ——流式模型直接 await
拿到的是异步生成器，要迭代到最后一个 chunk 才是完整回答。
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

from dotenv import load_dotenv

load_dotenv()

# 全 ETL 共享的并发上限（多个生成阶段串行跑，但共享一个上限最稳）。
LLM_SEM = asyncio.Semaphore(int(os.environ.get("LLM_ATTR_CONCURRENCY", "5")))


def etl_llm_model() -> str:
    """当前 ETL 用的模型名（缓存键要连模型一起 key，换模型自动失效）。"""
    return os.environ.get("LLM_FAST") or os.environ["LLM_MAIN"]


def get_etl_llm() -> Any:
    """建库档模型：不关思考、temperature 0.3（属性抽取要一点多样性但不能发散）。

    走 ``role="etl"`` 独立标记，网关的并发闸门与线上主链路共用——离线跑批时压不垮线上。
    """
    from app.agent.llm import build_model

    return build_model(etl_llm_model(), temperature=0.3, role="etl", thinking=False)
