"""ETL 共享的 LLM 构造与并发限流（llm_attributes / llm_aliases / shopify_attrs 共用）。

建库走便宜快模型（``LLM_FAST``，缺省回落 ``LLM_MAIN``）；信号量全 ETL 共享，
避免多个生成阶段叠加把 API rate limit 打爆。
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

# 全 ETL 共享的并发上限（多个生成阶段串行跑，但共享一个上限最稳）。
LLM_SEM = asyncio.Semaphore(int(os.environ.get("LLM_ATTR_CONCURRENCY", "5")))


def etl_llm_model() -> str:
    """当前 ETL 用的模型名（缓存键要连模型一起 key，换模型自动失效）。"""
    return os.environ.get("LLM_FAST") or os.environ["LLM_MAIN"]


def get_etl_llm() -> Any:
    return init_chat_model(
        etl_llm_model(),
        model_provider="openai",
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ["OPENAI_BASE_URL"],
        temperature=0.3,
        timeout=60.0,
        max_retries=2,
    )
