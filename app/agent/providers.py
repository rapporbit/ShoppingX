"""provider / model 寻址（阶段 2 第 2 条）。

**这层解决什么**：改造前全仓只有一个模型出口——``OPENAI_BASE_URL`` + ``OPENAI_API_KEY``。
换供应商等于改全局 env，一条链路上不可能同时用两家，fallback 只能在同一家里换个模型名
（``get_fallback_llm`` 的 docstring 早就写了「最好是不同供应商」，但机制上办不到）。
把出口信息挪进**模型名**本身（``dashscope/qwen3.8-flash``），跨家 fallback 才成立。

**向后兼容是硬要求**：不带前缀的模型名（``qwen3.8-flash``）照旧走 ``OPENAI_*`` 那对，
``.env`` 一个字不改也能跑。生产 env 里那几个键不动，是这次改造能安全上线的前提。

**为什么前缀不能无脑按第一个 ``/`` 切**：siliconflow 的模型名本身就带斜杠
（``Qwen/Qwen3-8B``）。所以判据是「第一段**配过** ``PROVIDER_<NAME>_BASE_URL`` 才算前缀」，
没配过就整串当模型名。拿配置当判据而不是拿字符串形状当判据，才不会把模型名吃掉一截。
"""

import os
from dataclasses import dataclass
from typing import Any

SEP = "/"


@dataclass(frozen=True)
class Endpoint:
    """一个模型的完整出口：打哪个 base_url、用哪把 key、模型名叫什么。"""

    provider: str
    """供应商标识（小写）。未带前缀时是 ``"default"``，指 ``OPENAI_*`` 那对。"""

    model: str
    """供应商侧的模型名（已剥掉前缀）。"""

    base_url: str
    api_key: str

    @property
    def ref(self) -> str:
        """回到 ``provider/model`` 形态；default 家不加前缀（保持与老配置逐字一致）。"""
        return self.model if self.provider == "default" else f"{self.provider}{SEP}{self.model}"


def _env(key: str) -> str:
    return (os.environ.get(key) or "").strip()


def provider_configured(provider: str) -> bool:
    """这个 provider 名配过出口吗（``PROVIDER_<NAME>_BASE_URL``）。

    解析前缀时**只认配过的**——见模块 docstring 里 ``Qwen/Qwen3-8B`` 那段。
    """
    return bool(_env(f"PROVIDER_{provider.upper()}_BASE_URL"))


def parse_model_ref(ref: str) -> tuple[str, str]:
    """``"dashscope/qwen3.8-flash"`` → ``("dashscope", "qwen3.8-flash")``。

    第一段没配过出口（或压根没有 ``/``）→ ``("default", 整串)``。
    """
    text = (ref or "").strip()
    if SEP in text:
        head, _, rest = text.partition(SEP)
        if head and rest and provider_configured(head):
            return head.lower(), rest
    return "default", text


def resolve_endpoint(ref: str) -> Endpoint:
    """把模型名解析成完整出口。

    default 家读 ``OPENAI_BASE_URL`` / ``OPENAI_API_KEY``——**故意直接 KeyError 式取值**
    （缺了就该炸在启动期，而不是等第一条用户请求才发现没配 key）。
    """
    provider, model = parse_model_ref(ref)
    if provider == "default":
        return Endpoint(
            provider="default",
            model=model,
            base_url=os.environ["OPENAI_BASE_URL"],
            api_key=os.environ["OPENAI_API_KEY"],
        )
    up = provider.upper()
    key = _env(f"PROVIDER_{up}_API_KEY") or _env("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(f"provider {provider} 配了 BASE_URL 却没有可用的 API key")
    return Endpoint(
        provider=provider,
        model=model,
        base_url=_env(f"PROVIDER_{up}_BASE_URL"),
        api_key=key,
    )


def fallback_chain() -> list[str]:
    """``LLM_FALLBACK_CHAIN=a/b,c/d`` → ``["a/b", "c/d"]``（顺序即优先级）。

    与老的 ``LLM_FALLBACK_MODEL`` 并存：这里返回空列表时，``llm.py`` 仍走老那条
    ``ModelConfig.fallback_model``。两套不叠加——链配了就以链为准，理由见 ``llm.py``。
    """
    raw = _env("LLM_FALLBACK_CHAIN")
    return [item.strip() for item in raw.split(",") if item.strip()]


def router_enabled() -> bool:
    """总开关。默认**开**：没配任何 ``PROVIDER_*`` 时 Router 只有一个 deployment、
    指向 ``OPENAI_*``，与直连逐字等价（spike 的 P1~P4 + R1~R4 量的就是这条路）。
    关掉即回退到直连 ``OpenAIChatModel``，这是本条的回滚开关。
    """
    raw = _env("LLM_PROVIDER_ROUTER")
    if not raw:
        return True
    return raw.lower() in {"1", "true", "yes", "on"}


def build_model_list(primary: Endpoint, fallbacks: list[Endpoint]) -> list[dict[str, Any]]:
    """Router 的 ``model_list``：每个出口一条 deployment，``model_name`` 就用 ``ref``。

    用 ref 当 deployment 名（而不是 "main"/"backup" 这种角色名）是刻意的：调用方手里只有
    模型名，用它当键就不需要再维护一张角色表；``model_fallback`` 事件也能直接报出真名。
    """
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for ep in [primary, *fallbacks]:
        if ep.ref in seen:
            continue
        seen.add(ep.ref)
        out.append(
            {
                "model_name": ep.ref,
                "litellm_params": {
                    # ``openai/`` 前缀是告诉 litellm「按 OpenAI 兼容协议发」，与 provider 无关：
                    # 本仓的出口全是 OpenAI 兼容端点（dashscope / siliconflow 都是）。
                    "model": f"openai/{ep.model}",
                    "api_key": ep.api_key,
                    "api_base": ep.base_url,
                },
            }
        )
    return out
