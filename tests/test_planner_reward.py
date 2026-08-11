"""M23 planner RL 的 reward（`app/eval/planner_reward.py`）。

**为什么这份测试值得写细**：reward 的 bug 不会崩，只会**给出一个方向错误的梯度**——模型老老实实
朝着错的地方学，训练曲线还很好看。等到 S4 端到端 A/B 才发现，一轮 GRPO 已经烧掉几十 GPU 小时。

四条重点：弃权维度不能当 0 分罚（否则无解的题系统性拉低分、污染 GRPO 的组内优势）、
反 hacking 门禁真的掐得住、集合 F1 要罚多判（不然模型学会「把所有域都填上」）、
命中口径复用 term_hits（watch ≠ watching）。
"""

from __future__ import annotations

from app.eval.planner_reward import (
    PARSE_FAIL_REWARD,
    compute_reward,
    score_econ,
    score_field,
    score_format,
    score_retrieval,
)

GOLD = {
    "category": "双肩包",
    "domains": ["bags"],
    "budget_amount": 400.0,
    "clear_budget": False,
    "budget_uncertain": False,
    "must_have": ["backpack"],
    "category_anchor": "backpacks",
}
GOOD_PLAN = {
    "category": "双肩包",
    "domains": ["bags"],
    "budget_amount": 400.0,
    "keywords": ["laptop backpack", "travel daypack"],
    "exclude_terms": [],
}
TITLES = ([f"Travel Laptop Backpack {i}" for i in range(10)]
          + [f"Random Gadget {i}" for i in range(10)])


class TestAbstention:
    def test_none_category_is_skipped_not_zeroed(self) -> None:
        """无上文碎片（golden category=None）不该把模型拖低分——那题本来就无解。"""
        # 模型给了品类与域，但 golden 在这两维弃权 → 只按 budget 计分，不该被判错
        plan = {"budget_amount": 400.0, "category": "双肩包", "domains": ["bags"]}
        skipped, parts = score_field(plan, {**GOLD, "category": None, "domains": None})
        assert skipped == 1.0 and set(parts) == {"budget"}
        # 同一个 plan 遇上「品类判得出」的 golden（且答错）就该扣分——对照证明弃权真的生效
        scored, _ = score_field(plan, {**GOLD, "category": "跑鞋", "domains": ["footwear"]})
        assert scored < skipped

    def test_uncertain_budget_skipped(self) -> None:
        gold = {**GOLD, "budget_uncertain": True, "budget_amount": None}
        score, parts = score_field(GOOD_PLAN, gold)
        assert "budget" not in parts and score == 1.0

    def test_all_dims_abstained_returns_none(self) -> None:
        gold = {"category": None, "domains": None, "budget_uncertain": True}
        assert score_field(GOOD_PLAN, gold)[0] is None

    def test_weight_redistribution(self) -> None:
        """检索维弃权时，剩下三维要在**自己的权重和**上归一，不能拿 0.45 当分母的一部分。"""
        br = compute_reward(GOOD_PLAN, GOLD, "想买双肩包，预算400", titles=None)
        assert br.retrieval is None
        assert br.total > 0.9  # 其余三维接近满分 → 总分也该接近满分，而不是被 0.45 稀释成 0.55


class TestAntiHacking:
    def test_copycat_keywords_halved(self) -> None:
        """整句照抄是 RL 最容易发现的捷径：字面不吃亏，召回全废。"""
        text = "想买个能装16寸笔记本的双肩包，预算400"
        plan = {**GOOD_PLAN, "keywords": [text]}
        br = compute_reward(plan, GOLD, text, titles=TITLES)
        assert any("照抄" in p for p in br.penalties)
        clean = compute_reward(GOOD_PLAN, GOLD, text, titles=TITLES)
        assert br.retrieval < clean.retrieval

    def test_fabricated_evidence_zeroes_field(self) -> None:
        plan = {**GOOD_PLAN,
                "exclude_terms": [{"word": "leather", "evidence": "用户说不要皮革"}]}
        br = compute_reward(plan, GOLD, "想买双肩包，预算400", titles=TITLES)
        assert br.field_score == 0.0
        assert any("evidence" in p for p in br.penalties)

    def test_real_evidence_survives(self) -> None:
        text = "想买双肩包，不要皮革的，预算400"
        plan = {**GOOD_PLAN,
                "exclude_terms": [{"word": "leather", "evidence": "不要皮革的"}]}
        br = compute_reward(plan, GOLD, text, titles=TITLES)
        assert br.field_score and br.field_score > 0.9


class TestFieldScoring:
    def test_domains_f1_punishes_overfill(self) -> None:
        """多判要罚。只算交集/召回，模型会学会「把 20 个域全填上」白拿满分。"""
        one, _ = score_field({**GOOD_PLAN, "domains": ["bags"]}, GOLD)
        many, _ = score_field(
            {**GOOD_PLAN, "domains": ["bags", "apparel", "electronics"]}, GOLD
        )
        assert many < one

    def test_category_loose_match(self) -> None:
        """「剪刀 / 工具」这类粒度差异在标注阶段就判为一致，reward 必须同口径。"""
        exact, _ = score_field(GOOD_PLAN, GOLD)
        loose, _ = score_field({**GOOD_PLAN, "category": "男士双肩包"}, GOLD)
        wrong, _ = score_field({**GOOD_PLAN, "category": "跑鞋"}, GOLD)
        assert exact == loose == 1.0 and wrong < 0.7

    def test_budget_must_be_exact(self) -> None:
        """预算是确定性字段，没有部分分——400 和 4000 差着一个数量级。"""
        off, parts = score_field({**GOOD_PLAN, "budget_amount": 4000.0}, GOLD)
        assert parts["budget"] < 0.5


class TestRetrieval:
    def test_empty_recall_scores_zero(self) -> None:
        assert score_retrieval([], GOLD)[0] == 0.0

    def test_no_anchor_abstains(self) -> None:
        assert score_retrieval(TITLES, {"must_have": [], "category_anchor": ""})[0] is None

    def test_term_hits_word_boundary(self) -> None:
        """复用 term_hits：watching 不算 watch 命中。裸 in 会让这条过。"""
        gold = {"must_have": ["watch"], "category_anchor": ""}
        assert score_retrieval(["i enjoy watching movies"] * 5, gold)[0] == 0.0
        assert score_retrieval(["mens dive watch"] * 5, gold)[0] > 0.0

    def test_more_hits_scores_higher(self) -> None:
        few = score_retrieval(["Backpack"] + ["Unrelated"] * 19, GOLD)[0]
        many = score_retrieval(["Backpack"] * 20, GOLD)[0]
        assert many > few


class TestFormatAndEcon:
    def test_global_domain_penalized(self) -> None:
        assert score_format({"domains": ["global"]})[0] < 1.0

    def test_system_owned_fields_penalized(self) -> None:
        """currency / budget_usd 是系统回填的，模型抢填说明没读懂分工。"""
        assert score_format({"domains": ["bags"], "currency": "CNY"})[0] < 1.0

    def test_econ_punishes_synonym_stuffing(self) -> None:
        clean, _ = score_econ({"keywords": ["laptop backpack", "school bag"]})
        stuffed, _ = score_econ({"keywords": [
            "laptop backpack", "laptop bag", "laptop rucksack", "laptop daypack"
        ]})
        assert stuffed < clean

    def test_econ_punishes_whole_sentence(self) -> None:
        long_kw, _ = score_econ({"keywords": [
            "a very large waterproof travel laptop backpack for men", "bag"
        ]})
        short, _ = score_econ({"keywords": ["travel backpack", "bag"]})
        assert long_kw < short


def test_parse_failure_is_vetoed() -> None:
    assert compute_reward(None, GOLD, "想买双肩包").total == PARSE_FAIL_REWARD
