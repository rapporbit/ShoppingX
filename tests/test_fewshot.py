"""M11 飞轮第三腿：few-shot 渲染的确定性测试（不依赖真实 LLM）。

锁住三件事：① 种子 + 蒸馏两源的渲染与去重；② system prompt 当前**不注入** few-shot（2k 版整段
删掉了 <examples>，模块只服务评测飞轮的蒸馏产出）；③ 文件缺失时降级为占位、不抛（渲染是附属品，
绝不反噬主链路）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.agent import fewshot
from app.agent.prompts import get_system_prompt


def test_render_fewshot_has_seed_examples() -> None:
    # 种子库专打的失败类别都应在渲染文本里出现（含意图化改造后新增的评价/组合意图范例）。
    rendered = fewshot.render_fewshot()
    assert "别自作主张" in rendered  # 纯推荐·只做用户要的
    assert "口头" in rendered or "现在为你生成清单" in rendered  # 收尾·别口头宣告
    assert "非购物" in rendered  # 非购物·直接兜底
    assert "评价" in rendered  # 评价单品·给结论 + 推荐替代品


def test_system_prompt_does_not_inject_fewshot() -> None:
    # 2k 版 system prompt 为压 token 整段删掉 <examples>：默认不再注入 few-shot。
    # 「反例」只出现在 few-shot 示例里、不在 prompt 正文，用它作注入与否的标记。
    sp = get_system_prompt()
    assert "<examples>" not in sp
    assert "反例" not in sp


def test_main_and_sub_share_same_system_prompt() -> None:
    # system prompt 纯静态：主与子完全同参（无运行时注入）→ system 段字节相同（同质 fork 硬约束）。
    main_sp = get_system_prompt()
    sub_sp = get_system_prompt()  # dispatch_tool 子 Agent 的 fallback 路径
    assert main_sp == sub_sp


def test_missing_files_degrade_to_placeholder(tmp_path: Path, monkeypatch: Any) -> None:
    # 两个源都缺失 → render 给占位、不抛。改路径后须清 lru_cache，测完复原。
    monkeypatch.setattr(fewshot, "_FEWSHOT_PATH", tmp_path / "nope.yml")
    monkeypatch.setattr(fewshot, "_DISTILLED_PATH", tmp_path / "nope2.yml")
    fewshot._load_examples.cache_clear()
    try:
        assert fewshot.render_fewshot() == "（暂无示例）"
    finally:
        fewshot._load_examples.cache_clear()  # 复原缓存，不污染其他用例


def test_seed_and_distilled_merge_dedup_by_intent(tmp_path: Path, monkeypatch: Any) -> None:
    # 合并两源：蒸馏只补种子没覆盖的 intent；与种子同 intent 的被去掉（种子优先）。
    seed = tmp_path / "seed.yml"
    distilled = tmp_path / "distilled.yml"
    seed.write_text("examples:\n  - intent: A\n    text: 种子A\n", encoding="utf-8")
    distilled.write_text(
        "examples:\n  - intent: A\n    text: 蒸馏A（应被去重）\n  - intent: B\n    text: 蒸馏B\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(fewshot, "_FEWSHOT_PATH", seed)
    monkeypatch.setattr(fewshot, "_DISTILLED_PATH", distilled)
    fewshot._load_examples.cache_clear()
    try:
        merged = fewshot._load_examples()
        intents = [e["intent"] for e in merged]
        assert intents == ["A", "B"]  # A 来自种子、B 来自蒸馏；蒸馏的 A 被去重
        assert merged[0]["text"] == "种子A"  # 种子优先
    finally:
        fewshot._load_examples.cache_clear()
