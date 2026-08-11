"""M23 S0-2：planner golden 标注里**零 LLM 的那半边**（预算 / 硬排除）。

这半边是规则跑出来的，所以它的 bug 不会崩、只会**往训练集里灌错标签**——模型照着学，
线下指标还很好看。三个已踩坑（都是实测出来的，不是设想）在这里钉死：

1. 规格数字被当成钱：「16寸的笔记本包」→ 预算 16；「重量不超过2kg」→ 预算 2000（k 被当「千」）。
2. 区间只取到下限：「预算300到500」→ 500 才是上限，标 300 等于教模型把预算标小。
3. 中文数词一律弃权太亏：「预算十二美元」可确定性解析；但「十五六块」是口语区间，
   必须弃权（``budget_uncertain``），猜哪个都是往 golden 里灌噪声。
"""

from __future__ import annotations

import pytest

from scripts.train.build_planner_golden import _cjk_value, budget_golden, exclude_expected


class TestBudgetAmount:
    @pytest.mark.parametrize(
        ("text", "amount", "currency"),
        [
            ("帮我挑几件夏天穿的男士短袖，预算 40 美元以内", 40.0, "USD"),
            ("推荐几个适合旅行的降噪耳机，预算1000", 1000.0, "CNY"),  # 裸数字落默认币种
            ("预算提到600吧", 600.0, "CNY"),  # 语境词不紧邻数字
            ("寄到日本，含税到手价1万日元以内", 10000.0, "JPY"),
            ("预算300到500的机械键盘", 500.0, "CNY"),  # 区间取上限
            ("256GB的手机，2千以内", 2000.0, "CNY"),  # 规格与预算同句
        ],
    )
    def test_extracts_money(self, text: str, amount: float, currency: str) -> None:
        g = budget_golden(text)
        assert g["budget_amount"] == amount
        assert g["currency"] == currency

    @pytest.mark.parametrize(
        "text",
        [
            "16寸的笔记本电脑包，别太贵",  # 尺寸
            "保温12小时的杯子",  # 时长
            "预算300元，重量不超过2kg",  # 2kg 的 k 不是「千」——它是单位的一部分
            "买个杯子",  # 压根没数字
        ],
    )
    def test_spec_numbers_are_not_money(self, text: str) -> None:
        assert budget_golden(text)["budget_amount"] in (None, 300.0)

    def test_spec_and_budget_in_one_sentence(self) -> None:
        """同句里既有规格又有钱：钱要抽对，规格不能污染。"""
        assert budget_golden("预算300元，重量不超过2kg")["budget_amount"] == 300.0


class TestChineseNumerals:
    @pytest.mark.parametrize(
        ("run", "value"),
        [("十二", 12), ("三百", 300), ("一千五", 1500), ("两万", 20000),
         ("一万五", 15000), ("一百二十三", 123), ("十", 10)],
    )
    def test_parses_regular_forms(self, run: str, value: float) -> None:
        assert _cjk_value(run) == value

    @pytest.mark.parametrize("run", ["十五六", "三四百"])
    def test_abstains_on_spoken_ranges(self, run: str) -> None:
        """「十五六块」是口语区间——解析成 15 或 16 都是编的，只能弃权。"""
        assert _cjk_value(run) is None

    def test_uncertain_flag_marks_abstention(self) -> None:
        g = budget_golden("送男朋友的挎包，十五六块就行")
        assert g["budget_amount"] is None
        # 关键：不是「本轮没提预算」，是「提了但规则判不准」。reward 侧据此跳过这一维，
        # 而不是拿 None 去罚一个答对 15 的模型。
        assert g["budget_uncertain"] is True

    def test_grounded_gate_does_not_eat_cjk_amount(self) -> None:
        """句里另有阿拉伯数字（16寸）时，中文数词预算不能被 grounded 闸误杀。"""
        assert budget_golden("16寸的笔记本包，预算三百")["budget_amount"] == 300.0


class TestClearBudget:
    def test_explicit_release_only(self) -> None:
        assert budget_golden("算了不限预算，直接上最好的")["clear_budget"] is True
        assert budget_golden("贵点也行")["clear_budget"] is True

    def test_cheap_preference_is_not_release(self) -> None:
        """「便宜点」「不要太贵」是软偏好，不是取消预算——判成取消会把预算约束整条抹掉。"""
        assert budget_golden("要便宜点的")["clear_budget"] is False
        assert budget_golden("不要太贵的耳机")["clear_budget"] is False


class TestExcludeExpected:
    @pytest.mark.parametrize("text", ["不要塑料的", "别给我皮革的", "把带logo的去掉"])
    def test_hard_negation(self, text: str) -> None:
        assert exclude_expected(text) is True

    @pytest.mark.parametrize(
        "text",
        ["不太喜欢塑料感的", "尽量别太花哨", "不要太贵的耳机", "要红轴的", "算了，塑料的也行"],
    )
    def test_soft_or_positive_is_not_exclusion(self, text: str) -> None:
        """软表达进 soft_dislikes（减分），判成硬排除会让 golden 教模型去淘汰商品。"""
        assert exclude_expected(text) is False
