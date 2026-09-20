"""LLM 令牌桶（阶段 2 第 4 条）：RPM + TPM 双桶，跨副本共享一份账。

**解决什么。** :class:`~app.agent.gateway.GatewayThrottle` 管的是**每进程**的并发位与起点间隔，
多副本一起跑时它谁也不认识谁——N 个副本各发各的，供应商那边看到的是 N 倍速率，429 由此而来。
把「这一分钟还能发几条 / 还能烧多少 token」放进 Redis，副本之间才是同一本账。

**为什么是两个桶而不是一个。** 供应商的配额本来就是两维（百炼：RPM 与 TPM 各一条线，还派生出
RPS/TPS）。只限条数，一条 30k 输入的长请求和一条 500 token 的短请求同价，TPM 那条线照样会撞；
只限 token，短请求可以每秒几十条打满 RPM。两个桶**要么都扣、要么都不扣**（见 :data:`_LUA_TAKE`）——
扣了 RPM 却因 TPM 不足退回，那一格就凭空漏了，攒几百次桶就空转。

**TPM 是预扣 + 结算的。** 发请求之前不知道会烧多少 token，只能估（输入用分词器数，输出按
``LLM_BUCKET_EST_OUTPUT`` 记一笔）；响应回来后拿真实 ``usage`` 补差额（:func:`settle`）。
真实用量超出预估时桶里的令牌**允许为负**——账不能丢，欠的那部分让下一次多等一会儿还回来。

**降级方向：Redis 一慢就退进程内桶。** Lua 调用超过 ``LLM_BUCKET_LUA_TIMEOUT_MS``（默认 50ms）
即放弃这次共享判定，改用本进程的一份桶（限额除以副本数？**不除**——退化期宁可放宽也不要卡死，
这一层的定位是优化不是正确性前提，与 :mod:`app.utils.shared_breaker` 同口径），并记一次
``llm_bucket_degraded``。

**等不到令牌怎么办：等满 ``LLM_BUCKET_WAIT_MAX_SEC``（默认 30s）就放行**，不抛错。理由同上：
桶挡不住的那几条会撞 429，而 429 有 ``GatewayThrottle.penalize`` 兜着；反过来把用户的任务饿死
在门口，没有任何东西兜。放行时记一条 warning，那是「限额配小了 / 副本开多了」的直接信号。

**键 = ``provider/model``**（与 :mod:`app.agent.llm_breaker` 同键）。计划里写的是 (provider, key)，
在本仓是同一件事——一个 provider 一把 key；但限额是供应商**按模型**给的，model 粒度更贴。

默认关（``LLM_BUCKET_ENABLED=0``），或限额未配（``LLM_RPM`` / ``LLM_TPM`` 都是 0）时整层空转，
一次 Redis 都不碰。
"""

import asyncio
import json
import logging
import os
import time
from typing import Any

from app.agent.providers import parse_model_ref
from app.utils.env import env_bool, env_float, env_int
from app.utils.tokens import count_tokens

__all__ = [
    "BucketLimits",
    "TokenBucket",
    "acquire",
    "bucket_enabled",
    "estimate_cost_tokens",
    "estimate_prompt_tokens",
    "get_bucket",
    "limits_for",
    "reset_buckets",
    "settle",
]

logger = logging.getLogger("shoppingx.llm.bucket")

#: 一个模型 ref 两个键：``:r`` 是 RPM 桶，``:t`` 是 TPM 桶。前缀与队列 / 熔断同一套 ``globex:``。
KEY_PREFIX = "globex:llmbucket:"


def bucket_enabled() -> bool:
    """总开关 ``LLM_BUCKET_ENABLED``，默认**关**（本条的回滚开关）。"""
    return env_bool("LLM_BUCKET_ENABLED", False)


class BucketLimits:
    """一个模型 ref 的两条线。``0`` = 该维度不限。"""

    __slots__ = ("rpm", "tpm")

    def __init__(self, rpm: int = 0, tpm: int = 0) -> None:
        self.rpm = max(0, int(rpm))
        self.tpm = max(0, int(tpm))

    @property
    def active(self) -> bool:
        return self.rpm > 0 or self.tpm > 0

    def __repr__(self) -> str:  # pragma: no cover - 只给日志看
        return f"BucketLimits(rpm={self.rpm}, tpm={self.tpm})"


def _env_token(name: str) -> str:
    """模型名转成能当 env 键的一段：``deepseek-v4-flash`` → ``DEEPSEEK_V4_FLASH``。"""
    return "".join(c if c.isalnum() else "_" for c in name).upper()


def limits_for(ref: str) -> BucketLimits:
    """读这个 ref 的限额，**逐维度**按 模型 → provider → 全局 取第一个非 0 的值。

    三级里模型那一级是必须的：供应商的配额是按模型给的（百炼 deepseek-v4-flash 的 TPM 是
    120 万，qwen3.8-flash 是 500 万，差 4 倍），而这两个在本仓走的是同一个出口——只按 provider
    配就只能取小的那个，把大模型白白压到四分之一。

    逐维度取意味着可以只配 ``MODEL_<X>_TPM`` 而让 RPM 落回全局，不必每一级都写全。
    每次现读不缓存，后台热更新（``config_overrides``）改完即生效，与
    :func:`~app.agent.llm_breaker.first_token_timeout` 同口径。
    """
    provider, model = parse_model_ref(ref)
    up = provider.upper()
    mod = _env_token(model)
    rpm = (
        env_int(f"MODEL_{mod}_RPM", 0) or env_int(f"PROVIDER_{up}_RPM", 0) or env_int("LLM_RPM", 0)
    )
    tpm = (
        env_int(f"MODEL_{mod}_TPM", 0) or env_int(f"PROVIDER_{up}_TPM", 0) or env_int("LLM_TPM", 0)
    )
    return BucketLimits(rpm, tpm)


def estimate_cost_tokens(prompt_tokens: int) -> int:
    """本次请求向 TPM 桶预扣多少：估出来的输入 + 一笔输出预留（``LLM_BUCKET_EST_OUTPUT``）。

    输出那笔是猜的，猜多猜少都由 :func:`settle` 按真实 ``usage`` 找平；预留为 0 会让每一轮都
    先透支再回填，桶始终处在「刚好卡住」的状态。
    """
    reserve = env_int("LLM_BUCKET_EST_OUTPUT", 1024)
    return max(1, int(prompt_tokens) + max(0, reserve))


def _content_tokens(content: Any) -> int:
    """一条消息正文的 token 估算。正文是 str，或一串内容块（文本 / 工具调用 / 工具结果）。"""
    if isinstance(content, str):
        return count_tokens(content)
    if not isinstance(content, list):
        return count_tokens(str(content)) if content else 0
    total = 0
    for block in content:
        text = block.get("text") if isinstance(block, dict) else getattr(block, "text", None)
        if isinstance(text, str):
            total += count_tokens(text)
            continue
        # 工具调用的入参、工具结果的正文都在输入里，且常常比文本块大得多。整块序列化着数，
        # 比只认 text 字段稳——块的形状随框架版本漂移，漏认一种就少算一大截。
        total += count_tokens(json.dumps(block, ensure_ascii=False, default=str))
    return total


def estimate_prompt_tokens(messages: Any, tools: Any = None) -> int:
    """粗估这次请求的输入 token：消息正文 + 工具 schema。

    **入参可能是 ``Msg`` 对象，不是 OpenAI 的 dict**：AgentScope 把 ``formatter.format()`` 放在
    ``OpenAIChatModel._call_api`` **内部**，所以闸门这一层（``__call__``）看到的还是没格式化的
    ``Msg``。只认 dict 的话这里会恒返回 0——不报错、不告警，只是 TPM 预扣悄悄只剩输出那一笔预留，
    桶在突发时挡不住任何东西（2026-09-20 部署时实测撞到）。两种形态都收。

    **工具 schema 必须算进来**：本仓 18 个工具的 JSON schema 每轮都随请求发，量级和一段中等
    长度的对话相当，漏掉它预扣就会系统性偏小（再由 :func:`settle` 一次次补扣，桶永远滞后一轮）。

    形状认不出来就返回 0——预估是为了少撞 429，不是记账，宁可估少也不能在这里抛。
    """
    try:
        total = 0
        for msg in messages or []:
            if isinstance(msg, dict):
                content, calls = msg.get("content"), msg.get("tool_calls")
            else:
                content, calls = getattr(msg, "content", None), getattr(msg, "tool_calls", None)
            total += _content_tokens(content)
            if calls:
                total += count_tokens(json.dumps(calls, ensure_ascii=False, default=str))
        if tools:
            total += count_tokens(json.dumps(tools, ensure_ascii=False, default=str))
        return total
    except Exception:  # pragma: no cover - 形状随框架版本漂移，估不出就算了
        logger.debug("输入 token 估算失败，按 0 记", exc_info=True)
        return 0


def _redis_url() -> str:
    """复用队列 / 事件那个实例（键名带 ``globex:`` 前缀，同一个 db 不会撞）。"""
    return (
        os.environ.get("BUCKET_REDIS_URL")
        or os.environ.get("QUEUE_REDIS_URL")
        or os.environ.get("EVENT_REDIS_URL", "redis://localhost:6379/2")
    )


# ── Lua ───────────────────────────────────────────────────────────────────────
# 时钟一律用 ``redis.call('TIME')``：副本的系统时间各差几秒是常态，让每个副本拿自己的 now 去
# 填同一个桶，等于凭空多发或少发令牌。Redis 是这本账唯一的时钟源。
#
# 桶本身是 hash：``tokens``（当前令牌，**可为负**）与 ``ts``（上次填充时刻，Redis 秒）。
_LUA_REFILL = """
local function refill(key, capacity, rate, now)
  local data = redis.call('HMGET', key, 'tokens', 'ts')
  local tokens = tonumber(data[1])
  local ts = tonumber(data[2])
  if tokens == nil or ts == nil then return capacity end
  local delta = now - ts
  if delta > 0 then tokens = tokens + delta * rate end
  if tokens > capacity then tokens = capacity end
  return tokens
end
local function save(key, tokens, now, ttl)
  redis.call('HSET', key, 'tokens', tokens, 'ts', now)
  redis.call('EXPIRE', key, ttl)
end
local now_raw = redis.call('TIME')
local now = tonumber(now_raw[1]) + tonumber(now_raw[2]) / 1000000
"""

#: 取一格 RPM + ``cost`` 个 TPM。返回 ``{允许?, 建议等待毫秒}``。
#:
#: **两个桶同生共死**：任一不够就一格都不扣，只回等待时间。扣一半会让另一维凭空漏格——
#: 这类漏账不会报错，只会表现成「桶明明没满却老是等」，几百次之后才看得出来。
_LUA_TAKE = (
    _LUA_REFILL
    + """
local rpm_cap = tonumber(ARGV[1])
local tpm_cap = tonumber(ARGV[2])
local cost = tonumber(ARGV[3])
local ttl = tonumber(ARGV[4])
local wait = 0
local r_tokens, t_tokens
if rpm_cap > 0 then
  local rate = rpm_cap / 60.0
  r_tokens = refill(KEYS[1], rpm_cap, rate, now)
  if r_tokens < 1 then wait = math.max(wait, (1 - r_tokens) / rate) end
end
if tpm_cap > 0 then
  local rate = tpm_cap / 60.0
  -- cost 超过整桶容量时永远等不到：钳到容量，让它「攒满就发」而不是死等。
  if cost > tpm_cap then cost = tpm_cap end
  t_tokens = refill(KEYS[2], tpm_cap, rate, now)
  if t_tokens < cost then wait = math.max(wait, (cost - t_tokens) / rate) end
end
if wait > 0 then return {0, math.ceil(wait * 1000)} end
if r_tokens ~= nil then save(KEYS[1], r_tokens - 1, now, ttl) end
if t_tokens ~= nil then save(KEYS[2], t_tokens - cost, now, ttl) end
return {1, 0}
"""
)

#: 结算：把「真实用量 - 预估用量」这笔差额补进 TPM 桶（RPM 一次调用就是一格，无需结算）。
#:
#: ``delta > 0``（估少了）→ 扣，**允许扣成负数**：欠的那部分必须留在账上，下一次自然多等。
#: ``delta < 0``（估多了）→ 回补，封顶到容量。
_LUA_SETTLE = (
    _LUA_REFILL
    + """
local tpm_cap = tonumber(ARGV[1])
local delta = tonumber(ARGV[2])
local ttl = tonumber(ARGV[3])
if tpm_cap <= 0 or delta == 0 then return 0 end
local tokens = refill(KEYS[1], tpm_cap, tpm_cap / 60.0, now) - delta
if tokens > tpm_cap then tokens = tpm_cap end
save(KEYS[1], tokens, now, ttl)
return 1
"""
)


class _LocalBucket:
    """Redis 不可用时顶上的进程内双桶。算法与 Lua 那份一致，时钟换 ``time.monotonic()``。

    **限额不按副本数分摊**：退化期的目标是「别把在跑的 run 掐断」，宁可整体放宽 N 倍——
    真撞上 429 还有 ``penalize`` 这一层。把它做成精确的，反而要引入「我是第几个副本」这种
    比 Redis 本身还脆的状态。
    """

    __slots__ = ("_rpm_tokens", "_tpm_tokens", "_ts")

    def __init__(self) -> None:
        self._rpm_tokens: float | None = None
        self._tpm_tokens: float | None = None
        self._ts = time.monotonic()

    def take(self, limits: BucketLimits, cost: int) -> float:
        """扣一次；返回 ``0`` = 放行，``>0`` = 建议等待秒数（本次未扣）。"""
        now = time.monotonic()
        elapsed = max(0.0, now - self._ts)
        self._ts = now
        wait = 0.0
        if limits.rpm > 0:
            rate = limits.rpm / 60.0
            cur = limits.rpm if self._rpm_tokens is None else self._rpm_tokens + elapsed * rate
            self._rpm_tokens = min(float(limits.rpm), cur)
            if self._rpm_tokens < 1:
                wait = max(wait, (1 - self._rpm_tokens) / rate)
        if limits.tpm > 0:
            rate = limits.tpm / 60.0
            need = min(float(cost), float(limits.tpm))
            cur = limits.tpm if self._tpm_tokens is None else self._tpm_tokens + elapsed * rate
            self._tpm_tokens = min(float(limits.tpm), cur)
            if self._tpm_tokens < need:
                wait = max(wait, (need - self._tpm_tokens) / rate)
        if wait > 0:
            return wait
        if self._rpm_tokens is not None:
            self._rpm_tokens -= 1
        if self._tpm_tokens is not None:
            self._tpm_tokens -= min(float(cost), float(limits.tpm))
        return 0.0

    def settle(self, limits: BucketLimits, delta: int) -> None:
        """把差额补进本地 TPM 桶（口径同 :data:`_LUA_SETTLE`）。"""
        if limits.tpm <= 0 or delta == 0 or self._tpm_tokens is None:
            return
        self._tpm_tokens = min(float(limits.tpm), self._tpm_tokens - delta)


# ── Redis 客户端与脚本（进程级一份，所有 ref 共用）────────────────────────────────
_client: Any | None = None
_scripts: tuple[Any, Any] | None = None
_resolved = False
#: 降级冷却截止时刻（monotonic）。Redis 卡住时不该每次调用都再赔 50ms 进去。
_degraded_until = 0.0


def _lua_timeout() -> float:
    return max(0.001, env_int("LLM_BUCKET_LUA_TIMEOUT_MS", 50) / 1000.0)


def _key_ttl() -> int:
    """桶键的存活时长。远大于一分钟窗口即可——没人用的模型不该永久占着键。"""
    return env_int("LLM_BUCKET_KEY_TTL", 300)


def _get_scripts() -> tuple[Any, Any] | None:
    """懒建客户端并注册两个脚本；建不起来返回 ``None``（整层退本地桶）。"""
    global _client, _scripts, _resolved
    if _resolved:
        return _scripts
    _resolved = True
    try:
        import redis.asyncio as aredis  # 可选依赖，懒加载（与 queue / dedup / control 同源）

        client = aredis.from_url(
            _redis_url(),
            decode_responses=True,
            socket_connect_timeout=1.0,
            # 与 shared_breaker 同理由：这一层坐在每次模型调用的必经之路上，Redis 卡住必须立刻
            # 退本地，而不是让每条请求先干等——那正是它要消除的那种等待。
            socket_timeout=1.0,
        )
        _scripts = (client.register_script(_LUA_TAKE), client.register_script(_LUA_SETTLE))
    except Exception as exc:
        logger.warning("令牌桶 Redis 初始化失败，退进程内桶：%s", exc)
        _scripts = None
        return None
    _client = client
    logger.info("LLM 令牌桶启用：%s（前缀 %s）", _redis_url(), KEY_PREFIX)
    return _scripts


def set_client(client: Any | None) -> None:
    """注入客户端（测试 / 手工装配用）；传 ``None`` 即彻底退本地桶。"""
    global _client, _scripts, _resolved, _degraded_until
    _resolved = True
    _degraded_until = 0.0
    _client = client
    _scripts = (
        (client.register_script(_LUA_TAKE), client.register_script(_LUA_SETTLE))
        if client is not None
        else None
    )


def reset_buckets() -> None:
    """复位客户端与进程内桶注册表（测试收尾用）。"""
    global _client, _scripts, _resolved, _degraded_until
    _client = None
    _scripts = None
    _resolved = False
    _degraded_until = 0.0
    _buckets.clear()


def _mark_degraded(exc: BaseException | None = None) -> None:
    """记一次降级并进入冷却：冷却期内直接走本地桶，不再为每次调用赔一个超时。"""
    global _degraded_until
    cooldown = env_float("LLM_BUCKET_DEGRADE_COOLDOWN_SEC", 10.0)
    first = time.monotonic() >= _degraded_until
    _degraded_until = time.monotonic() + max(0.0, cooldown)
    try:
        from app.observability import metrics

        metrics.record_llm_bucket("degraded")
    except Exception:  # pragma: no cover - 观测是附属品
        logger.debug("令牌桶降级计数失败", exc_info=True)
    if first:
        logger.warning("令牌桶退进程内桶 %.0fs（Redis 不可用 / 超时）：%s", cooldown, exc)


class TokenBucket:
    """一个模型 ref 的双桶把手：共享账在 Redis，退化账在 :class:`_LocalBucket`。"""

    def __init__(self, ref: str) -> None:
        self.ref = ref
        self._rk = f"{KEY_PREFIX}{ref}:r"
        self._tk = f"{KEY_PREFIX}{ref}:t"
        self._local = _LocalBucket()

    async def _take_remote(self, limits: BucketLimits, cost: int) -> float | None:
        """共享桶取一次。返回等待秒数（``0`` = 放行）；``None`` = 这条路现在不可用。"""
        if time.monotonic() < _degraded_until:
            return None
        scripts = _get_scripts()
        if scripts is None:
            return None
        take = scripts[0]
        try:
            res = await asyncio.wait_for(
                take(keys=[self._rk, self._tk], args=[limits.rpm, limits.tpm, cost, _key_ttl()]),
                timeout=_lua_timeout(),
            )
        except asyncio.CancelledError:
            raise  # 用户点了停止 / 上游超时，不是 Redis 的问题，别当降级记
        except Exception as exc:
            _mark_degraded(exc)
            return None
        allowed, wait_ms = int(res[0]), int(res[1])
        return 0.0 if allowed else max(0.0, wait_ms / 1000.0)

    async def take(self, limits: BucketLimits, cost: int) -> float:
        """取一次令牌：``0`` = 放行，``>0`` = 建议等待秒数（本次没扣）。"""
        remote = await self._take_remote(limits, cost)
        if remote is not None:
            return remote
        return self._local.take(limits, cost)

    async def settle(self, limits: BucketLimits, delta: int) -> None:
        """按「真实 - 预估」的差额找平 TPM 桶。失败静默：结算丢一笔不该让一轮对话挂掉。

        预扣与结算之间发生降级切换时，这笔差额会落到另一本账上（预扣在 Redis、结算在本地，或
        反过来）。不去追：那只会让 Redis 桶紧一会儿或松一会儿，而键 TTL 一到就清零；为此维护
        「这次是在哪本账上扣的」反而要在主链路上多存一份状态。
        """
        if delta == 0 or limits.tpm <= 0:
            return
        if time.monotonic() >= _degraded_until:
            scripts = _get_scripts()
            if scripts is not None:
                try:
                    await asyncio.wait_for(
                        scripts[1](keys=[self._tk], args=[limits.tpm, delta, _key_ttl()]),
                        timeout=_lua_timeout(),
                    )
                    return
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    _mark_degraded(exc)
        self._local.settle(limits, delta)


_buckets: dict[str, TokenBucket] = {}


def get_bucket(ref: str) -> TokenBucket:
    """取（或建）这个 ref 的桶把手。进程内桶的状态要跨调用留着，所以必须有这张表。"""
    bucket = _buckets.get(ref)
    if bucket is None:
        bucket = TokenBucket(ref)
        _buckets[ref] = bucket
    return bucket


async def acquire(ref: str, prompt_tokens: int) -> int:
    """发请求前过闸：拿到令牌才返回，返回值是**预扣的 token 数**（交给 :func:`settle` 找平）。

    返回 ``0`` 表示这一层没生效（开关关着 / 没配限额），调用方无需结算。

    等满 ``LLM_BUCKET_WAIT_MAX_SEC`` 仍拿不到就**放行并记 warning**——理由见模块 docstring。
    放行时照样返回预扣值：这一格虽然没从桶里扣走，账还是要记在结算里，否则真实用量会凭空消失。
    """
    if not bucket_enabled():
        return 0
    limits = limits_for(ref)
    if not limits.active:
        return 0
    cost = estimate_cost_tokens(prompt_tokens)
    bucket = get_bucket(ref)
    deadline = time.monotonic() + max(0.0, env_float("LLM_BUCKET_WAIT_MAX_SEC", 30.0))
    waited = 0.0
    while True:
        wait = await bucket.take(limits, cost)
        if wait <= 0:
            if waited > 0:
                logger.debug("令牌桶 %s 等待 %.2fs 后放行", ref, waited)
            return cost
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            logger.warning(
                "令牌桶 %s 等满 %.0fs 仍无令牌（限额 %s），本次放行", ref, waited, limits
            )
            try:
                from app.observability import metrics

                metrics.record_llm_bucket("overflow")
            except Exception:  # pragma: no cover - 观测是附属品
                logger.debug("令牌桶放行计数失败", exc_info=True)
            return cost
        # 多睡 20ms：Lua 算出来的是「刚好够」的时刻，扣着点醒来只会白跑一趟 Redis。
        nap = min(wait + 0.02, remaining)
        await asyncio.sleep(nap)
        waited += nap


async def settle(ref: str, reserved: int, actual_tokens: int) -> None:
    """响应回来后找平：``reserved`` 是 :func:`acquire` 的返回值，``actual_tokens`` 是真实用量。

    ``reserved <= 0``（这一层没生效）或拿不到 usage 时什么都不做——**没有 usage 不等于零用量**，
    按 0 结算会把预扣的那笔整个还回去，等于这次调用没花 token。
    """
    if reserved <= 0 or actual_tokens <= 0:
        return
    delta = int(actual_tokens) - int(reserved)
    if delta == 0:
        return
    await get_bucket(ref).settle(limits_for(ref), delta)
