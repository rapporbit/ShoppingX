"""约束类型学分诊（第一病根：embedding 盲区成体系、逐坑发现）。

分诊器只负责「认出是什么类型」；gap 类现状回退 topic 通路（行为不变、日志留痕），
补专道的优先级由巡检种子（build_eval_queries.py「约束类型学巡检」桶）的评测证据决定。
"""

from __future__ import annotations

from typing import get_args

from app.utils.terms import CONSTRAINT_LANES, ConstraintKind, classify_constraint


class TestClassify:
    def test_numeric_spec_reuses_spec_lane(self) -> None:
        assert classify_constraint("16寸") == "numeric_spec"
        assert classify_constraint("16 inch") == "numeric_spec"
        assert classify_constraint("40L") == "numeric_spec"

    def test_numeric_range(self) -> None:
        assert classify_constraint("17寸以上") == "numeric_range"
        assert classify_constraint("不超过20寸") == "numeric_range"
        assert classify_constraint("under 100") == "numeric_range"

    def test_range_hint_without_digit_is_topic(self) -> None:
        """「预算以内」这类无数字的表述不是范围约束。"""
        assert classify_constraint("预算以内") == "topic"

    def test_enum_size(self) -> None:
        assert classify_constraint("M码") == "enum_size"
        assert classify_constraint("size m") == "enum_size"
        assert classify_constraint("XL") == "enum_size"
        assert classify_constraint("均码") == "enum_size"

    def test_bare_single_letter_is_not_size(self) -> None:
        """裸 s/m/l 歧义太大（品牌名/型号缩写），必须带 码/号/size 语境。"""
        assert classify_constraint("m") == "topic"
        assert classify_constraint("l") == "topic"

    def test_count_pack(self) -> None:
        assert classify_constraint("两个装") == "count_pack"
        assert classify_constraint("2件装") == "count_pack"
        assert classify_constraint("3-pack") == "count_pack"
        assert classify_constraint("pack of 2") == "count_pack"

    def test_set_concept_is_topic_not_pack(self) -> None:
        """「旅行三件套」是成品概念（topic/bundle），不是单品装量硬约束。"""
        assert classify_constraint("旅行三件套") == "topic"

    def test_generation(self) -> None:
        assert classify_constraint("第3代") == "generation"
        assert classify_constraint("gen 3") == "generation"

    def test_plain_terms_stay_topic(self) -> None:
        for t in ("纯棉", "plastic", "小众设计", ""):
            assert classify_constraint(t) == "topic"


class TestLanesContract:
    def test_every_kind_has_a_lane_row(self) -> None:
        """契约表完整性：分诊得出的每个类型都必须有「通路/失效方向/状态」登记。"""
        for kind in get_args(ConstraintKind):
            assert kind in CONSTRAINT_LANES, f"{kind} 未在 CONSTRAINT_LANES 登记"

    def test_negation_documented_even_if_not_classified(self) -> None:
        """否定由入口分桶决定、不是 classify 输出，但契约表必须登记它的通路。"""
        assert "negation" in CONSTRAINT_LANES

    def test_status_values_are_honest(self) -> None:
        """状态只有 live/gap 两档；gap 行必须写明「回退 topic=双盲」的失效方向。"""
        for kind, (lane, failure, status) in CONSTRAINT_LANES.items():
            assert status in {"live", "gap"}, f"{kind} 状态非法：{status}"
            assert lane and failure
            if status == "gap":
                assert "双盲" in failure, f"gap 类 {kind} 必须如实标注失效方向"
