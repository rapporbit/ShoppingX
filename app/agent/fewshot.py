"""few-shot 范例库：高分轨迹沉淀为可注入 system prompt 的 in-context 示例（飞轮第三腿）。

与长期偏好（``injector``）的关键区别——**few-shot 是全局静态的**（不依赖具体 user / query），
所以由 :func:`app.agent.prompts.get_system_prompt` 内部默认加载注入：主 Agent 与 fork 子 Agent
（fallback 到 ``get_system_prompt()``）都自动带上同一份，天然同质、且内容固定不破坏 prompt cache。

示例来自 M11 Rubric 评测定位的 bad case（过度澄清 / 口头收尾 / 非购物乱检索），是「上下文级飞轮」
的「定位 bad case → 沉淀 few-shot」一环。文件 ``prompt/few_shot.yml`` 可由飞轮迭代更新。
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# 两个示例源（都在仓库根的 prompt/ 下；本文件位于 app/agent/，向上两级到根）：
# - few_shot.yml：人工种子（带注释、稳定，手维护）。
# - few_shot_distilled.yml：scripts/eval/distill_fewshot.py 从评测高分轨迹自动蒸馏的产物（飞轮第三腿
#   闭环，可被反复重写）。分两文件，使自动蒸馏不冲掉人工种子的注释与精心措辞。
_FEWSHOT_PATH = Path(__file__).resolve().parents[2] / "prompt" / "few_shot.yml"
_DISTILLED_PATH = Path(__file__).resolve().parents[2] / "prompt" / "few_shot_distilled.yml"


def _read_examples(path: Path) -> list[dict[str, Any]]:
    """读单个 yml 的 examples 列表；缺失 / 解析失败降级为空（注入是附属品，绝不反噬主链路）。"""
    if not path.exists():
        return []
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        examples = data.get("examples", [])
        return examples if isinstance(examples, list) else []
    except Exception:
        logger.warning("加载 few-shot 示例失败（path=%s），降级为空", path, exc_info=True)
        return []


@lru_cache(maxsize=1)
def _load_examples() -> list[dict[str, Any]]:
    """合并人工种子 + 自动蒸馏的示例，按 intent 去重（种子优先，蒸馏只补种子没覆盖的场景）。"""
    seed = _read_examples(_FEWSHOT_PATH)
    seen = {e.get("intent") for e in seed}
    distilled = [e for e in _read_examples(_DISTILLED_PATH) if e.get("intent") not in seen]
    return seed + distilled


def render_fewshot(limit: int | None = None) -> str:
    """把 few-shot 示例渲染成注入 system prompt 的一段文本；无示例时给占位。

    ``FEWSHOT_ENABLED=0`` 可整体关掉注入（feature flag）——既给 A/B 回归对照做对照组，也作生产
    可控开关（如发现 few-shot 反而拉低某场景时快速回退）。

    ``limit`` 截取前 N 条（种子在前、蒸馏在后，故截的是蒸馏那一头）。slim variant 用它把 7 条压到
    2 条：Comet / Cursor 这类成熟 agent 都保留了可观的 examples 段，全删大概率伤 flash 档模型的
    指令遵循度，所以只减不删。``None``（默认）= 全量，行为与瘦身前一致。
    """
    from app.utils.env import env_bool

    if not env_bool("FEWSHOT_ENABLED", True):
        return "（暂无示例）"
    examples = _load_examples()
    if limit is not None:
        examples = examples[:limit]
    if not examples:
        return "（暂无示例）"
    blocks = [
        f"示例{i}（{e.get('intent', '')}）：\n{str(e.get('text', '')).rstrip()}"
        for i, e in enumerate(examples, 1)
    ]
    return "\n\n".join(blocks)
