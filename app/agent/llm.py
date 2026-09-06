"""统一的大模型工厂。

主 Agent 与 worker 共用同一个模型实例（``lru_cache`` 全局只建一次），一来省掉每次派发重建
连接池的开销，二来「同一批工具、同一个模型」是 Supervisor-Workers 里 worker 不降智的前提——
切的是**工具发放范围**，不是模型能力。快档 :func:`get_fast_llm` 是唯一的例外，它同款模型只关
思考（见该函数 docstring 的实测取舍）。

模型、endpoint、温度全部走 ``.env``（见 ``.env.example``），代码里不写死。
判官模型 :func:`get_judge_llm` 给 Rubric 评测用，默认更强、temperature=0 保证评分稳定。

**本模块有两套并存的工厂**（批 0 迁移期）：上半部分 ``get_*`` 返回 LangChain 的
``BaseChatModel``（旧运行时在用），下半部分 ``get_as_*`` 返回 AgentScope 的
``ThrottledChatModel``（新运行时）。并存的理由与摘除时机见下半部分的分隔注释。
"""

import os
from functools import lru_cache

from agentscope.agent import ModelConfig
from agentscope.credential import OpenAICredential
from agentscope.formatter import OpenAIChatFormatter
from agentscope.model import OpenAIChatModel
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from pydantic import SecretStr

from app.agent.gateway import GatewayThrottle, ThrottledChatModel

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
    global LLM_REQUEST_TIMEOUT, LLM_MAX_RETRIES, _as_throttle
    LLM_REQUEST_TIMEOUT = _env_float("LLM_REQUEST_TIMEOUT", 60.0)
    LLM_MAX_RETRIES = _env_int("LLM_MAX_RETRIES", 2)
    for factory in (
        get_llm,
        get_fast_llm,
        get_vision_llm,
        get_judge_llm,
        get_as_llm,
        get_as_fast_llm,
        get_as_lite_llm,
        get_as_vision_llm,
        get_as_judge_llm,
        get_as_fallback_llm,
    ):
        factory.cache_clear()
    # 闸门也要重建：并发数 / 间隔改了，旧实例里的信号量容量是改不动的。
    # 只影响**新建**的模型实例；在跑的任务仍持有旧闸门（同上：不追求「立刻换掉在跑的」）。
    _as_throttle = None


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


# ============================================================================
# AgentScope 运行时的模型工厂（批 0 迁移期与上面的 LangChain 工厂**并存**）
# ----------------------------------------------------------------------------
# 为什么并存而不是原地改返回类型：旧运行时有 39 个文件、1042 个测试挂在上面的
# ``get_llm()`` 上。原地换类型会让整仓一夜全红，「红的不许进下一层」的规矩就守不住了。
# 所以新壳一律 ``get_as_*``（与工具层的 ``AS_TOOLS`` 同一命名口径），L8 摘掉 LangChain 时
# 再把 ``as_`` 前缀去掉、占回本名。
# ============================================================================

_as_throttle: GatewayThrottle | None = None


def get_gateway_throttle() -> GatewayThrottle:
    """进程内共享的一份闸门（主 / 快档 / 判官 / 备用共用一个并发池）。

    共享是刻意的：网关的 RPM 是按 API key 算的，分开各建各的池子等于把限流让给运气。
    """
    global _as_throttle
    if _as_throttle is None:
        _as_throttle = GatewayThrottle(
            max_concurrency=_env_int("LLM_MAX_CONCURRENCY", 4),
            min_interval=_env_float("LLM_MIN_INTERVAL_SECONDS", 0.0),
        )
    return _as_throttle


def _as_credential(vision: bool = False) -> OpenAICredential:
    """凭据：视觉档可用 ``VISION_*`` 单独指向别的供应商，其余复用 ``OPENAI_*``。

    key 包成 ``SecretStr``——AgentScope 的凭据类型要求如此，顺带也让 key 不会因为某处
    ``repr()`` / 异常回溯就明文躺进日志。
    """
    if vision:
        return OpenAICredential(
            api_key=SecretStr(os.environ.get("VISION_API_KEY") or os.environ["OPENAI_API_KEY"]),
            base_url=os.environ.get("VISION_BASE_URL") or os.environ["OPENAI_BASE_URL"],
        )
    return OpenAICredential(
        api_key=SecretStr(os.environ["OPENAI_API_KEY"]),
        base_url=os.environ["OPENAI_BASE_URL"],
    )


def _as_formatter() -> OpenAIChatFormatter:
    """格式化层：开了 ``COMPRESS_CACHE_CONTROL`` 就换成会打断点标记的那版。

    每档模型各持一个实例（formatter 无状态，共享与否都行；各持一份省得将来有人往里加状态时
    踩到跨档串扰）。``keep_recent`` 与压缩 Hook 同源，两处用同一把尺子量断点，否则标记会打在
    压缩边界之外——缓存前缀里混进易变内容，命中率白丢。
    """
    if not _env_bool("COMPRESS_CACHE_CONTROL", False):
        return OpenAIChatFormatter()
    from app.harness.formatter import CacheAwareOpenAIFormatter

    return CacheAwareOpenAIFormatter(keep_recent=_env_int("COMPRESS_KEEP_RECENT", 3))


def _build_as_model(
    model: str,
    *,
    temperature: float,
    role: str,
    thinking: bool,
    vision: bool = False,
) -> ThrottledChatModel:
    """统一装配：闸门 + 超时 + 重试 + hybrid 模型的思考开关。

    ``max_retries`` 交给模型自己的重试环（``ChatModelBase.__call__``），Agent 层的
    ``ModelConfig.max_retries`` 另设 0，避免两层重试相乘——同一个 429 被试 9 次那种。
    """
    return ThrottledChatModel(
        credential=_as_credential(vision=vision),
        model=model,
        parameters=OpenAIChatModel.Parameters(temperature=temperature),
        stream=True,
        formatter=_as_formatter(),
        max_retries=LLM_MAX_RETRIES,
        client_kwargs={"timeout": LLM_REQUEST_TIMEOUT},
        # hybrid 模型（DashScope / Qwen / DeepSeek）经 OpenAI 兼容层读 extra_body 里的
        # enable_thinking；不支持的供应商忽略该字段（无害）。口径与上面的 LangChain 档一致。
        extra_body=None if thinking else {"enable_thinking": False},
        throttle=get_gateway_throttle(),
        role=role,
    )


@lru_cache(maxsize=1)
def get_as_llm() -> ThrottledChatModel:
    """主 AgentLoop 的模型（AgentScope 侧），对应 :func:`get_llm`。"""
    return _build_as_model(
        os.environ["LLM_MAIN"],
        temperature=_env_float("LLM_TEMPERATURE", 0.3),
        role="main",
        thinking=True,
    )


@lru_cache(maxsize=1)
def get_as_fast_llm() -> ThrottledChatModel:
    """快档（AgentScope 侧），对应 :func:`get_fast_llm`——同款模型只关思考，不换弱模型。

    L0 的 S2 spike 实测：同一条 planner 请求，主档 12.7~17.2s，关思考后 4.5~5.1s，
    结构化结果质量无差。这一档的收益是实打实的解码时间，不是玄学。
    """
    return _build_as_model(
        os.environ.get("LLM_FAST") or os.environ["LLM_MAIN"],
        temperature=_env_float("LLM_FAST_TEMPERATURE", _env_float("LLM_TEMPERATURE", 0.3)),
        role="fast",
        thinking=_env_bool("LLM_FAST_REASONING", False),
    )


@lru_cache(maxsize=1)
def get_as_lite_llm() -> ThrottledChatModel:
    """便宜档（AgentScope 侧），对应 LangChain 的 ``model_router._lite_llm``。

    预算降档时由 ``HarnessAgentAdapter`` 换上。默认就是快档（同模型关思考）；配了 ``LLM_LITE``
    才真换一个更便宜的模型名——**换模型名会打断前缀缓存**，所以默认不换，只关思考。
    """
    name = os.environ.get("LLM_LITE", "").strip()
    if not name:
        return get_as_fast_llm()
    return _build_as_model(
        name,
        temperature=_env_float("LLM_FAST_TEMPERATURE", _env_float("LLM_TEMPERATURE", 0.3)),
        role="lite",
        thinking=False,
    )


@lru_cache(maxsize=1)
def get_as_vision_llm() -> ThrottledChatModel:
    """看图档（AgentScope 侧），对应 :func:`get_vision_llm`。"""
    return _build_as_model(
        os.environ["LLM_VISION"],
        temperature=_env_float("LLM_VISION_TEMPERATURE", 0.1),
        role="vision",
        thinking=_env_bool("LLM_VISION_REASONING", False),
        vision=True,
    )


@lru_cache(maxsize=1)
def get_as_judge_llm() -> ThrottledChatModel:
    """判官档（AgentScope 侧），对应 :func:`get_judge_llm`——temperature=0，尺子不能自己抖。"""
    return _build_as_model(
        os.environ.get("LLM_JUDGE") or os.environ["LLM_MAIN"],
        temperature=_env_float("LLM_JUDGE_TEMPERATURE", 0.0),
        role="judge",
        thinking=True,
    )


def build_as_judge_llm(temperature: float) -> ThrottledChatModel:
    """判官档、**温度由调用方指定**（离线标注 / 多票投票用），AgentScope 侧。

    与 :func:`get_as_judge_llm` 的区别只在温度来源：那个钉在 env 上（0.0，线上评测的尺子
    不许自己抖），而 S0-2 的 golden 标注**要的恰恰是三档不同温度**——温度是三票投票的扰动源，
    三票同温等于把同一次调用重复三遍，一致率虚高、分歧根本暴露不出来。

    刻意不加 ``lru_cache``：温度是入参，缓存键会随之膨胀，而这类离线脚本一轮只建三个模型。
    """
    return _build_as_model(
        os.environ.get("LLM_JUDGE") or os.environ["LLM_MAIN"],
        temperature=temperature,
        role="judge",
        thinking=True,
    )


@lru_cache(maxsize=1)
def get_as_fallback_llm() -> ThrottledChatModel | None:
    """备用模型：主模型重试用尽后由 ``ModelConfig.fallback_model`` 接手。

    **不配就返回 None**（不默默拿 ``LLM_MAIN`` 当备用）——同一个模型当自己的备用毫无意义，
    主模型垮了通常是网关或该模型本身的问题，换个名字再撞一次只是多烧一次钱、多等一轮。
    要用就在 ``.env`` 里配一个**真正不同**的模型（最好是不同系列或不同供应商）。
    """
    name = os.environ.get("LLM_FALLBACK_MODEL", "").strip()
    if not name or name == os.environ.get("LLM_MAIN"):
        return None
    return _build_as_model(
        name,
        temperature=_env_float("LLM_TEMPERATURE", 0.3),
        role="fallback",
        thinking=True,
    )


def get_model_config() -> ModelConfig:
    """挂到 Agent 的模型配置：备用模型 + Agent 层重试次数。

    Agent 层 ``max_retries=0``：模型自己那层已经按 ``LLM_MAX_RETRIES`` 重试过了，两层相乘会把
    「重试 2 次」变成 9 次请求——限流时这等于火上浇油。这里只负责「主模型彻底不行了就换备用」。
    """
    return ModelConfig(max_retries=0, fallback_model=get_as_fallback_llm())
