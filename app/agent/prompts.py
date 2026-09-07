"""提示词读取层：从 ``prompt/prompts.yml`` 加载并渲染。

System prompt 用 XML 分块，且**纯静态**——不含任何运行时注入位。
长期偏好 / 近期行为历史 / 会话级 P_t 这些**每轮必变**的运行时上下文一律**不进 system prompt**，
改由 ``main_agent._inject_runtime_context`` 拼进当轮 human message（见该函数的 prompt cache 说明）：
system prompt 逐字稳定才能成为跨轮 / 跨会话都命中的缓存前缀。主 / 子 AgentLoop 共用同一份 system
prompt（同质 fork 的硬约束）——静态化后主与子的 system 段字节相同，子 Agent 也能命中主 Agent 的缓存。
"""

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

# prompts.yml 在仓库根的 prompt/ 下；本文件位于 app/agent/，向上两级到根。
# **提示词正文全仓只此一份**，system_prompt 即原「冲 2k」版：<2000 tok、无 few-shot。
#
# 与已删的 PROMPT_VARIANT 机制的区别（别把旧机制原样复活）：旧机制是「几份互相独立的完整
# prompts_*.yml，按 env 整份切换」——改一处要同步 N 份，且切换是全局的、没有分桶也没有对照。
# 现在是「一份正文 + prompt/versions/<semver>.yml 叠加层 + 按 user_id 分桶」（批 4 / 18-3）：
# 版本文件只存差异，A/B 按人分流，桶号与版本进 trace 与配额账本，用 Rubric 分桶对照来判优劣。
# 更早的 full / slim 全量副本仍在 prompt/archive/，仅供翻阅、不参与运行。
_PROMPT_DIR = Path(__file__).resolve().parents[2] / "prompt"
_PROMPTS_PATH = _PROMPT_DIR / "prompts.yml"
_VERSIONS_DIR = _PROMPT_DIR / "versions"

#: 基线版本号：``prompt/versions/1.0.0.yml`` 是零覆盖的 ``prompts.yml`` 本身。
BASE_VERSION = "1.0.0"
#: ``base: prompts.yml`` 的字面量——版本链的终点。
_BASE_DOC = "prompts.yml"
_MAX_CHAIN = 16  # 版本链深度上限，兜住 A→B→A 这类环


@lru_cache(maxsize=1)
def _load_base_prompts() -> dict[str, Any]:
    with _PROMPTS_PATH.open("r", encoding="utf-8") as f:
        return dict(yaml.safe_load(f))


def available_versions() -> list[str]:
    """``prompt/versions/`` 下已声明的版本号（字典序；目录不存在时只有基线版）。"""
    if not _VERSIONS_DIR.is_dir():
        return [BASE_VERSION]
    return sorted(p.stem for p in _VERSIONS_DIR.glob("*.yml"))


def _merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """把叠加层合进基线：顶层键替换，值为 dict 的键（如 ``sub_agents``）做一层递归合并。

    **覆盖的键必须在 base 里已存在**——拼错一个键名（``sytem_prompt``）若被静默接受，那个桶的
    用户就会一直跑在未改动的提示词上，而 A/B 报告照样出数、看起来一切正常。宁可开机就炸。
    """
    merged = dict(base)
    for key, value in overrides.items():
        if key not in base:
            raise KeyError(f"prompt 版本覆盖了不存在的键 {key!r}（基线里没有它，八成是拼错了）")
        if isinstance(value, dict) and _PATCH_KEY in value:
            merged[key] = _apply_patches(key, base[key], value[_PATCH_KEY])
        elif isinstance(value, dict) and isinstance(base[key], dict):
            merged[key] = _merge(base[key], value)
        else:
            merged[key] = value
    return merged


#: 字符串键的**补丁**形态：``system_prompt: {__patch__: [{old: …, new: …}, …]}``。
#: 整键替换对 ``system_prompt`` 这种几百行的长文本是错的——它会把该版本**冻结**在写它那天的
#: 全文上，此后基线修一个错别字、加一条硬约束，这个桶的用户都拿不到；A/B 量到的也不再是
#: 「那一句话的差异」而是「一整份旧文 vs 新文」。补丁只写改的那一句，其余逐字跟着基线走。
_PATCH_KEY = "__patch__"


def _apply_patches(key: str, text: Any, patches: Any) -> str:
    """按顺序把 ``{old, new}`` 补丁打到基线文本上。**每条 ``old`` 必须恰好命中一次**：
    命不中说明基线那句话已经改掉了（补丁失效，得重写），命中多次说明锚点太短（会改到别处）——
    两种情况都不该静默，宁可开机就炸。"""
    if not isinstance(text, str):
        raise TypeError(f"prompt 版本对非字符串键 {key!r} 用了 {_PATCH_KEY}")
    if not isinstance(patches, list):
        raise TypeError(f"prompt 版本 {key!r} 的 {_PATCH_KEY} 必须是 [{{old, new}}, …] 列表")
    out = text
    for idx, patch in enumerate(patches):
        if not isinstance(patch, dict) or "old" not in patch or "new" not in patch:
            raise ValueError(f"prompt 版本 {key!r} 的第 {idx} 条补丁缺 old / new")
        old, new = str(patch["old"]), str(patch["new"])
        hits = out.count(old)
        if hits != 1:
            raise ValueError(
                f"prompt 版本 {key!r} 的第 {idx} 条补丁锚点命中 {hits} 次（须恰好 1 次）："
                f"{old[:60]!r}"
            )
        out = out.replace(old, new, 1)
    return out


@lru_cache(maxsize=8)
def _load_prompts(version: str | None = None) -> dict[str, Any]:
    """解析某个版本的完整提示词表（``None`` / 基线版 → 主文件原样）。

    版本文件只存差异，正文永远来自 ``prompts.yml``：**主文件改一个字，所有版本同时跟着变**，
    没有「手工同步 N 份副本」这道会漂的工序（见 ``prompt/versions/1.0.0.yml`` 头部）。
    """
    if version is None:
        return _load_base_prompts()
    chain: list[dict[str, Any]] = []
    seen: set[str] = set()
    cur = version
    while cur != _BASE_DOC:
        if cur in seen:
            raise ValueError(f"prompt 版本链成环：{cur} 已在链上 {sorted(seen)}")
        if len(chain) >= _MAX_CHAIN:
            raise ValueError(f"prompt 版本链过深（>{_MAX_CHAIN}），八成是配错了 base")
        seen.add(cur)
        path = _VERSIONS_DIR / f"{cur}.yml"
        if not path.exists():
            raise KeyError(f"找不到 prompt 版本 {cur}（期望 {path}）")
        with path.open("r", encoding="utf-8") as f:
            doc = dict(yaml.safe_load(f) or {})
        chain.append(doc)
        cur = str(doc.get("base") or _BASE_DOC)
    resolved = _load_base_prompts()
    for doc in reversed(chain):  # 从最靠近基线的那层往外叠
        resolved = _merge(resolved, dict(doc.get("overrides") or {}))
    return resolved


# curator 与 preference_parse 共用的「长期偏好字段规则」占位符——两者落同一张表，规则必须逐字
# 一致，曾各抄一份已开始措辞漂移。YAML 锚点没法插进 block scalar 中间，故用显式占位符 + replace
# （不用 str.format：prompt 里有 {"size": "42"} 这类字面大括号，format 会炸）。
_PREF_RULES_PLACEHOLDER = "<<PREF_FIELD_RULES>>"


def _inject_pref_rules(text: str, version: str | None) -> str:
    rules = str(_resolved(version).get("pref_field_rules", "")).rstrip()
    return text.replace(_PREF_RULES_PLACEHOLDER, rules)


def _resolved(version: str | None) -> dict[str, Any]:
    """``version=None`` → 取**当前分桶**的版本（A/B）。

    默认不是「基线版」而是「分桶版」，为的是让 A/B 覆盖到任意一个键：若这里退回基线，某个变体
    改了 ``planner_prompt`` 却因为 planner 工具没显式传版本而**静默不生效**——报告照样出数，
    实验却什么也没测。函数内延迟 import：``ab`` 要用本模块的 :func:`available_versions` 校验，
    模块级 import 会成环。
    """
    if version is not None:
        return _load_prompts(version)
    from app.agent.ab import active_version

    return _load_prompts(active_version())


def get_worker_system_prompt(kind: str, version: str | None = None) -> str:
    """worker 的**专职** system prompt（``sub_agents.search`` / ``sub_agents.trade``）。

    批 1 起 worker 不再复用主 prompt：读写切分之后，主 prompt 里的收尾判据、bundle 槽位流程、
    派发策略对 worker 全是噪声——更糟的是**指挥它去调根本没发给它的工具**（worker 手上没有
    shopping_summary / task_dispatch），白烧一轮撞 tool-not-found。

    对 prompt cache 的影响是**正的**：worker 不再蹭主 Agent 的前缀，但它自己那段短得多且逐字
    稳定，同类 worker 之间（跨调用、跨会话）共用同一条前缀。

    批 0 的 ``clone`` 模式不走这里——它的定义就是「与主 Agent 同工具集、同 system prompt」，
    换 prompt 就不是对照组了（见 :func:`app.agent.agents.build_worker_agent`）。
    """
    sub_agents = _resolved(version).get("sub_agents", {})
    if kind not in sub_agents:
        raise KeyError(f"prompts.yml 缺少 sub_agents.{kind} 段")
    return str(sub_agents[kind])


def get_system_prompt(version: str | None = None) -> str:
    """主 / 子 AgentLoop 共用的**纯静态** system prompt（无任何运行时变量，逐字稳定）。

    **不注入 few-shot**：当前 system prompt 是 2k 版，为压 token 整段删掉了 ``<examples>``。
    ``app.agent.fewshot`` 模块仍在（评测飞轮的「高分轨迹蒸馏」一腿还用它产出示例），只是不再拼进
    system prompt——要恢复注入，得先给模板加回 ``<examples>`` 段。

    **长期偏好 / 近期行为历史 / 会话级 P_t 一律不在此注入**：三者每轮必变（偏好按本轮 query 语义
    裁剪、历史每轮收尾覆盖、P_t 每轮 curator 更新），混进 system prompt 会打断本该跨轮稳定的
    prompt cache 前缀。它们改由 ``main_agent._inject_runtime_context`` 拼进当轮 human message——
    那是缓存断点之后、永不缓存的部分，把「每轮必变」彻底隔离在缓存区外（对齐 refdocs/05 §4.4
    「按易变性分层，越易变越靠后」）。

    ``version`` 缺省 = 当前用户所在 A/B 桶的版本（见 :mod:`app.agent.ab`）。**版本按 user_id
    稳定**，所以同一个人跨轮拿到的 system prompt 逐字不变，prompt cache 前缀照旧命中；变的只是
    「不同人前缀不同」，那本来就是多用户的常态。
    """
    return str(_resolved(version)["system_prompt"])


def get_planner_prompt(version: str | None = None) -> str:
    """planner 工具的提示词。"""
    return _resolved(version)["planner_prompt"]


def get_shopping_summary_prompt(version: str | None = None) -> str:
    """shopping_summary 工具的提示词。"""
    return _resolved(version)["shopping_summary_prompt"]


def get_memory_curator_prompt(version: str | None = None) -> str:
    """记忆管家（curator）的提示词——独立于购物工作流的偏好判定器。"""
    return _inject_pref_rules(_resolved(version)["memory_curator_prompt"], version)


def get_preference_parse_prompt(version: str | None = None) -> str:
    """把用户手填的一句话拆成结构化偏好条目（偏好页面的「添加」入口用）。"""
    return _inject_pref_rules(_resolved(version)["preference_parse_prompt"], version)
