"""统一的大模型工厂。

同质 fork 的硬约束：主 loop 与所有 fork 出去的子 loop 必须用**完全相同**的模型与
温度，子 Agent 才是主 loop 的真克隆。因此 :func:`get_llm` 全局只建一次实例
（``lru_cache``），主/子共享，也避免每次 fork 重建连接池。

模型、endpoint、温度全部走 ``.env``（见 ``.env.example``），代码里不写死。
判官模型 :func:`get_judge_llm` 给 Rubric 评测用，默认更强、temperature=0 保证评分稳定。
"""

import os
from functools import lru_cache

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

# 模块导入即加载 .env，使后续 os.environ 读取生效（已设置的环境变量优先，不覆盖）。
load_dotenv()


def _env_float(key: str, default: float) -> float:
    """读取浮点型环境变量，缺失或非法时回退默认值。"""
    raw = os.environ.get(key)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(key: str, default: int) -> int:
    """读取整型环境变量，缺失或非法时回退默认值（与 utils.env.env_int 同义）。"""
    raw = os.environ.get(key)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


# 单次请求超时 + 有限重试：这是「一次卡死的 API 调用拖垮整条任务」的根因防线。
# 同质 fork 下主/子共享一个连接池，高并发偶发某条连接焊死、服务端迟迟不回响应头时，
# 没有请求超时就会一直挂到主 loop 的全局预算（MAIN_AGENT_TIMEOUT_SEC）被 wait_for 杀掉，
# 整条购物任务白跑。设了超时，卡住的调用 60s 内中断并换连接重试，任务得以自愈。
LLM_REQUEST_TIMEOUT = _env_float("LLM_REQUEST_TIMEOUT", 60.0)
LLM_MAX_RETRIES = _env_int("LLM_MAX_RETRIES", 2)


def _load_params() -> None:
    """重新从 env 求值本模块参数，并**清空模型实例缓存**（后台管理页面热更新入口）。

    模型档位与温度是在 ``get_*_llm()`` 内部直读 env 的，但那些函数挂着 ``lru_cache``——不清缓存
    的话，改完 ``LLM_MAIN`` 只会继续拿到用旧模型建好的那个实例。清掉后下次调用即按新 env 重建。

    只对**新任务**生效：进行中的 loop 早已持有旧实例的引用，中途换模型反而会让同一条任务前后
    半段用不同模型（同质 fork 的硬约束也就破了），故不追求「立刻换掉在跑的」。
    """
    global LLM_REQUEST_TIMEOUT, LLM_MAX_RETRIES
    LLM_REQUEST_TIMEOUT = _env_float("LLM_REQUEST_TIMEOUT", 60.0)
    LLM_MAX_RETRIES = _env_int("LLM_MAX_RETRIES", 2)
    for factory in (get_llm, get_fast_llm, get_vision_llm, get_judge_llm):
        factory.cache_clear()


@lru_cache(maxsize=1)
def get_llm() -> BaseChatModel:
    """主 / 子 AgentLoop 共享的大模型实例。

    需要 env：``LLM_MAIN`` / ``OPENAI_API_KEY`` / ``OPENAI_BASE_URL``；
    可选 ``LLM_TEMPERATURE``（默认 0.3）、``LLM_REQUEST_TIMEOUT``（默认 60s）、
    ``LLM_MAX_RETRIES``（默认 2）。
    """
    return init_chat_model(
        os.environ["LLM_MAIN"],
        model_provider="openai",
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ["OPENAI_BASE_URL"],
        temperature=_env_float("LLM_TEMPERATURE", 0.3),
        timeout=LLM_REQUEST_TIMEOUT,
        max_retries=LLM_MAX_RETRIES,
    )


def _env_bool(key: str, default: bool) -> bool:
    """读取布尔环境变量（1/true/yes/on 为真，0/false/no/off 为假），缺失回退默认值。"""
    raw = os.environ.get(key)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@lru_cache(maxsize=1)
def get_fast_llm() -> BaseChatModel:
    """子 Agent 执行 + 文案生成（shopping_summary）用的「快档」模型。

    延迟治理（perf/model-tiering）：实测端到端 ~96% 耗时是 reasoning 模型逐字解码。子 Agent 在
    #2/#2.5 后已塌成 1-2 跳 item_search、shopping_summary 只按既定候选生成文案——这两处不需深推理。

    **关键取舍（实测教训）**：不要为了「快」换一个更弱的小模型——实测 qwen-turbo / qwen3.5-flash
    在子任务上反而更慢且降级（弱模型多轮 + 自纠偏差）。正确做法是**同一个 LLM_MAIN（如 deepseek-v4
    这类 hybrid 模型）只把 reasoning 关掉**：能力不变（不降质），只省掉那段昂贵的 thinking 解码。
    故默认 ``LLM_FAST`` 不配 → 用 ``LLM_MAIN`` 同款；``LLM_FAST_REASONING=off``（默认）→ 经
    ``extra_body={"enable_thinking": false}`` 关推理（DashScope / Qwen / DeepSeek hybrid 口径）。

    破的是「主子**推理档**同质」：工具集 / prompt / 模型都与主一致（能力同质），只是子不开思考。
    需要 env：``OPENAI_API_KEY`` / ``OPENAI_BASE_URL``；可选 ``LLM_FAST``（缺省同 ``LLM_MAIN``）、
    ``LLM_FAST_REASONING``（默认 off=关推理）、``LLM_FAST_TEMPERATURE``（默认同主温度）。
    """
    fast_model = os.environ.get("LLM_FAST") or os.environ["LLM_MAIN"]
    kwargs: dict = {
        "model_provider": "openai",
        "api_key": os.environ["OPENAI_API_KEY"],
        "base_url": os.environ["OPENAI_BASE_URL"],
        "temperature": _env_float("LLM_FAST_TEMPERATURE", _env_float("LLM_TEMPERATURE", 0.3)),
        "timeout": LLM_REQUEST_TIMEOUT,
        "max_retries": LLM_MAX_RETRIES,
    }
    # 默认关掉快档的 reasoning：hybrid 模型（DashScope/Qwen/DeepSeek）经 OpenAI 兼容层读 extra_body
    # 里的 enable_thinking。不支持的供应商会忽略该字段（无害）；要保留推理设 LLM_FAST_REASONING=on。
    if not _env_bool("LLM_FAST_REASONING", False):
        kwargs["extra_body"] = {"enable_thinking": False}
    return init_chat_model(fast_model, **kwargs)


def vision_enabled() -> bool:
    """是否配了多模态模型（``LLM_VISION``）——没配则图片理解整条腿优雅降级、不崩。"""
    return bool(os.environ.get("LLM_VISION", "").strip())


@lru_cache(maxsize=1)
def get_vision_llm() -> BaseChatModel:
    """看图专用的多模态模型（image_understand 工具内部用，主 loop 永不直接见图）。

    **为什么单开一档**：主模型（``LLM_MAIN``，如 deepseek 系）是纯文本的，把图片塞进主 loop 的
    messages 只会报错或被静默忽略。所以图片只在工具内部喂给 VL 模型，主 loop 只看到它吐出的
    结构化文本结论——顺带也符合「工具中间产物不污染主上下文」的既有架构。

    温度压到 0.1：看图是「读事实」不是「做创作」，同一张图两次调用该给同一个品类。

    **默认关 reasoning**（同 :func:`get_fast_llm` 的口径）。现在的 VL 模型多是 hybrid 的
    （``qwen3.5-flash`` 即是），而「这张图里是什么」是感知题不是推理题：实测同一张手环图，开思考
    要 6.5s / 790 token（其中 668 是思考），关掉后 2.2s / 185 token，**答案一模一样**。不关就是
    白付一段最贵的解码。要保留推理设 ``LLM_VISION_REASONING=on``。

    需要 env：``LLM_VISION``（如 qwen3.5-flash）；endpoint / key 默认复用 ``OPENAI_*``，
    也可用 ``VISION_BASE_URL`` / ``VISION_API_KEY`` 单独指向别的供应商。
    """
    kwargs: dict = {
        "model_provider": "openai",
        "api_key": os.environ.get("VISION_API_KEY") or os.environ["OPENAI_API_KEY"],
        "base_url": os.environ.get("VISION_BASE_URL") or os.environ["OPENAI_BASE_URL"],
        "temperature": _env_float("LLM_VISION_TEMPERATURE", 0.1),
        "timeout": LLM_REQUEST_TIMEOUT,
        "max_retries": LLM_MAX_RETRIES,
    }
    # 不支持该字段的供应商会忽略它（无害）——纯视觉模型如 qwen-vl-plus 本就没有思考档。
    if not _env_bool("LLM_VISION_REASONING", False):
        kwargs["extra_body"] = {"enable_thinking": False}
    return init_chat_model(os.environ["LLM_VISION"], **kwargs)


@lru_cache(maxsize=1)
def get_judge_llm() -> BaseChatModel:
    """Rubric 评测专用的判官模型（更强、temperature=0）。

    需要 env：``OPENAI_API_KEY`` / ``OPENAI_BASE_URL``；
    可选 ``LLM_JUDGE``（默认回退到 ``LLM_MAIN``）、``LLM_JUDGE_TEMPERATURE``（默认 0.0）。
    """
    judge_model = os.environ.get("LLM_JUDGE") or os.environ["LLM_MAIN"]
    return init_chat_model(
        judge_model,
        model_provider="openai",
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ["OPENAI_BASE_URL"],
        temperature=_env_float("LLM_JUDGE_TEMPERATURE", 0.0),
        timeout=LLM_REQUEST_TIMEOUT,
        max_retries=LLM_MAX_RETRIES,
    )
