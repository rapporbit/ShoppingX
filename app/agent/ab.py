"""Prompt 版本 A/B：按 ``user_id`` 稳定哈希分桶，配置驱动放量（18-3）。

**为什么按人分桶、而不是按会话或按轮**。提示词改的是 Agent 的行为风格，用户是跨轮累积感知的：
同一个人这轮拿 1.0.0、下轮拿 1.1.0，他体验到的是「这 Agent 时好时坏」，而我们量到的是两个版本
的混合——既伤体验又测不出东西。按人分桶还顺带保住 prompt cache：同一个人的 system 前缀逐字稳定。

**桶号只由 (盐, user_id) 决定，与当前放量比例无关**。这是「达标扩桶」能成立的前提：把候选版本
从 10% 调到 30%，桶 0~9 的人原地不动、桶 10~29 的人从对照组迁进来，没有人被重新洗牌。若按
「哈希 % 变体数」分配，每改一次比例所有人重排，前后两段数据就不可比了。

**匿名 / 未登录一律归对照组**（``user_id`` 为空 → 桶号 :data:`ANON_BUCKET` = -1，版本 = 默认版）。
理由：① 关掉鉴权时全体共用假身份 ``demo-user``，对它哈希等于把所有匿名流量塞进同一个桶，比例
完全失真；② 匿名用量本来就不进 :class:`~app.db.models.UsageLedger`（配额只在鉴权开启时记账），
实验组捞不到成本与 token，只剩半张表；③ 实验结论要能落到「某个人的体验变好了」，匿名没有这个
主语。要在本地手工试某一版，用 ``PROMPT_VERSION`` 直接钉死，不必伪造身份。

配置（``.env``，热更新见 :mod:`app.config.registry` 之外的普通 env 读取——每次任务重新求值）：

- ``PROMPT_VERSION``：默认版本 / 对照组，缺省 ``1.0.0``。
- ``PROMPT_AB_VARIANTS``：候选版本与百分比，形如 ``"1.1.0:10"`` 或 ``"1.1.0:10,1.2.0:5"``。
  **对照组不写**——它隐式吃掉剩下的桶，所以放量只改这一个数字，改不出「加起来不等于 100」的账。
- ``PROMPT_AB_SALT``：哈希盐。换盐 = 整体重新分桶，只有在「想开一轮全新实验、不希望沿用上轮
  分组」时才动它。
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass

from app.agent.prompts import BASE_VERSION, available_versions
from app.api.context import get_user_id
from app.utils.env import env_str

logger = logging.getLogger("shoppingx.ab")

#: 分桶总数。固定 100 → 配置里的百分比就是桶数，看得见摸得着。
BUCKETS = 100
#: 匿名 / 未登录的哨兵桶号（不参与实验，恒走默认版本）。
ANON_BUCKET = -1


@dataclass(frozen=True)
class Assignment:
    """一次分桶结果。``bucket`` 与 ``version`` 一起进 trace 与账本。"""

    bucket: int
    version: str
    #: 是否处于实验组（``False`` = 对照组 / 匿名 / 实验未开）。
    in_experiment: bool


def default_version() -> str:
    """对照组版本（``PROMPT_VERSION``）。配了不存在的版本 → 退回基线并告警，不让服务起不来。"""
    version = env_str("PROMPT_VERSION", BASE_VERSION).strip() or BASE_VERSION
    if version not in available_versions():
        logger.warning("PROMPT_VERSION=%s 不存在，退回基线 %s", version, BASE_VERSION)
        return BASE_VERSION
    return version


def variant_weights() -> list[tuple[str, int]]:
    """解析 ``PROMPT_AB_VARIANTS`` → ``[(版本, 百分比), ...]``，保持配置里的书写顺序。

    顺序即桶区间的分配顺序（第一个候选拿 ``[0, w1)``，第二个拿 ``[w1, w1+w2)`` ……），所以
    **别调换已上线候选的先后**——那等于把两组人对调，比换盐还狠。写坏的条目跳过并告警：A/B
    配置写错不该把服务打挂，但也不能静默当成 0%（那就是「实验开了却什么也没测」）。
    """
    raw = env_str("PROMPT_AB_VARIANTS", "").strip()
    if not raw:
        return []
    known = set(available_versions())
    out: list[tuple[str, int]] = []
    total = 0
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        version, _, pct_str = chunk.partition(":")
        version = version.strip()
        try:
            pct = int(pct_str.strip())
        except ValueError:
            logger.warning("PROMPT_AB_VARIANTS 条目 %r 百分比不是整数，跳过", chunk)
            continue
        if version not in known:
            logger.warning("PROMPT_AB_VARIANTS 里的版本 %s 不存在，跳过", version)
            continue
        if pct <= 0:
            continue
        if total + pct > BUCKETS:
            logger.warning("PROMPT_AB_VARIANTS 百分比累计超过 100%%，%s 之后的条目忽略", version)
            break
        out.append((version, pct))
        total += pct
    return out


def bucket_of(user_id: str | None) -> int:
    """``user_id`` → 稳定桶号 ``[0, 100)``；空身份返回 :data:`ANON_BUCKET`。

    用 sha256 而不是内置 ``hash()``：后者对 str 每进程随机加盐（PYTHONHASHSEED），重启一次
    所有人换组，跨进程（api / worker）更是各分各的——那种「分桶」等于没分。
    """
    if not user_id:
        return ANON_BUCKET
    salt = env_str("PROMPT_AB_SALT", "")
    digest = hashlib.sha256(f"{salt}:{user_id}".encode()).hexdigest()
    return int(digest[:8], 16) % BUCKETS


def assign(user_id: str | None = None) -> Assignment:
    """当前用户的分桶与版本。``user_id`` 省略时取 ContextVar 里的本轮身份。"""
    uid = user_id if user_id is not None else get_user_id()
    bucket = bucket_of(uid)
    base = default_version()
    if bucket == ANON_BUCKET:
        return Assignment(bucket=ANON_BUCKET, version=base, in_experiment=False)
    lower = 0
    for version, pct in variant_weights():
        if lower <= bucket < lower + pct:
            return Assignment(bucket=bucket, version=version, in_experiment=True)
        lower += pct
    return Assignment(bucket=bucket, version=base, in_experiment=False)


def active_version() -> str:
    """本轮该用哪个版本的提示词（:mod:`app.agent.prompts` 的缺省来源）。

    **每次调用重新求值**，不缓存：热更新配置后新任务立刻按新比例分流，正在跑的那轮不受影响
    （它的 system prompt 早已装配完毕）——与 :mod:`app.config.registry` 的口径一致。
    """
    try:
        return assign().version
    except Exception:  # noqa: BLE001 —— 分桶炸了也绝不能让 Agent 起不来
        logger.warning("A/B 分桶失败，退回基线提示词", exc_info=True)
        return BASE_VERSION
