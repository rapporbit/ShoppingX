"""SKILL 触发标注集与判定口径（阶段 S4）。

这里**不跑模型**——触发率只能真跑（见 ``scripts/eval/run_skill_trigger.py`` 的模块 docstring）。
本单守的是那条评测链路自己别坏掉的三件事：标注集覆盖得全、判定函数分得清四种错法、以及探针
不会因为「被 import 了一下」就插进生产 pipeline。
"""

from types import SimpleNamespace

import pytest

from app.agent.skills import SKILLS_DIR
from app.harness.middleware import harness
from scripts.eval.build_skill_trigger import CASES, UNCOVERED
from scripts.eval.run_skill_trigger import _tool_calls, _verdict


def _block(type_: str, name: str = "", input_=None):  # noqa: ANN001, ANN202
    return SimpleNamespace(type=type_, name=name, input=input_)


def _msg(*blocks):  # noqa: ANN002, ANN202
    return SimpleNamespace(content=list(blocks))


def test_every_skill_has_a_positive_case_or_a_reason() -> None:
    """新增一份 SKILL.md 就得给它出正例，或在 ``UNCOVERED`` 里写明为什么不测。

    白名单式守栏（同 ``test_skills.EXPECTED_SKILLS``）：没有这条，新 skill 会悄悄落在验收集之外，
    报告上的「正例触发率 100%」量的却是少了一份的集合。
    """
    on_disk = {p.name for p in SKILLS_DIR.iterdir() if (p / "SKILL.md").is_file()}
    covered = {c["skill"] for c in CASES if c["skill"]}
    assert on_disk - covered - set(UNCOVERED) == set()
    assert covered <= on_disk, "标注集引用了不存在的 skill"


def test_dataset_has_unique_ids_and_negative_cases() -> None:
    ids = [c["id"] for c in CASES]
    assert len(ids) == len(set(ids))
    # 负例是这套验收的另一半：只量正例的话，「每轮都读一份 skill」也能拿满分。
    assert sum(1 for c in CASES if not c["skill"]) >= 3


def test_tool_calls_accepts_dict_and_json_input() -> None:
    """``Skill`` 的入参可能是 dict 也可能是 JSON 串，取决于框架版本怎么存，两种都要认。"""
    msg = _msg(
        _block("tool_call", "Skill", {"skill": "order-care"}),
        _block("tool_call", "item_search", '{"query": "背包"}'),
        _block("text"),
    )
    assert _tool_calls(msg) == [
        {"name": "Skill", "skill": "order-care"},
        {"name": "item_search", "skill": ""},
    ]
    assert (
        _tool_calls(_msg(_block("tool_call", "Skill", '{"skill": "search-discovery"}')))[0]["skill"]
        == "search-discovery"
    )
    assert _tool_calls(None) == []


POSITIVE = {"id": "x", "skill": "order-care"}
NEGATIVE = {"id": "y", "skill": None}


@pytest.mark.parametrize(
    ("case", "calls", "expected"),
    [
        # 正例：读对了那份 → PASS
        (
            POSITIVE,
            [{"name": "Skill", "skill": "order-care"}, {"name": "query_order", "skill": ""}],
            "PASS",
        ),
        # 正例：一个工具都没调 / 只调了业务工具 → MISS
        (POSITIVE, [{"name": "item_search", "skill": ""}], "MISS"),
        (POSITIVE, [], "MISS"),
        # 正例：读成了别的 skill → WRONG_SKILL（要修的是两份 description 撞车，与 MISS 不同因）
        (POSITIVE, [{"name": "Skill", "skill": "search-discovery"}], "WRONG_SKILL"),
        # 负例：不该读却读了
        (NEGATIVE, [{"name": "Skill", "skill": "purchase-research"}], "FALSE_FIRE"),
        (NEGATIVE, [{"name": "item_search", "skill": ""}], "PASS"),
        (NEGATIVE, [], "PASS"),
    ],
)
def test_verdict_tells_the_four_failure_modes_apart(case: dict, calls: list, expected: str) -> None:
    assert _verdict(case, calls)[0] == expected


def test_lone_skill_round_is_passed_but_noted() -> None:
    """只发了 ``Skill``、没带业务工具：判 PASS 但留 note。

    S2 的分流表要求「和本轮第一个检索/读取工具同一轮发出」，白等一次往返是延迟账上的损失、
    不是触发错误——混进 MISS 会让触发率背上不属于它的锅，所以单列成 ``lone_skill_round``。
    """
    verdict, note = _verdict(POSITIVE, [{"name": "Skill", "skill": "order-care"}])
    assert verdict == "PASS"
    assert note


def test_probe_is_not_installed_on_import() -> None:
    """导入评测脚本不得往全局 harness 插探针——它抛的是 BaseException，会掐断别人的用例。"""
    names = {name for _, name, _ in harness.list_hooks("post_reflect")}
    assert "skill_trigger_probe" not in names
