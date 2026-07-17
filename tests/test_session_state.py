"""会话级短期偏好状态 P_t（``app.memory.session_state``）的确定性测试。

刀2（单写者 + id 增量）后的语义总纲 = 四条不变量：
- I1 存续：说过且未撤回的约束跨轮存活（merge 从不要求重放）。
- I2 撤回：引用代码发的 id + 词面呼应双闸，核验过才删。
- I3 过期：search 换代（epoch）+ TTL，约束不跨意图 / 不跨会话。
- I4 失效方向：核验不过=不删（宁松）；旧格式 / 损坏 / 过期=按空开局。

测试照**真实数据形态**写（中文原话 + 英文标题 token，复用 normalize_terms/term_hits 的行为），
不造理想化输入——见记忆 memory-bugs-are-silent-inversions：这层的 bug 不崩，只把推荐做反。
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from app.memory.session_state import (
    SessionConstraint,
    SessionPrefState,
    load_pt,
    merge_pt,
    save_pt,
)


def _c(
    content: str,
    polarity: str = "like",
    *,
    keywords: list[str] | None = None,
    blocking: bool = True,
    quote: str = "",
) -> SessionConstraint:
    # id 留空：跨轮身份由 merge_pt 发号，测试与生产（planner._plan_constraints）同一形态。
    return SessionConstraint(
        content=content,
        polarity=polarity,  # type: ignore[arg-type]
        keywords=keywords or [],
        blocking=blocking,
        source_quote=quote,
    )


# ---------- 发号：唯一、跨轮稳定、跨 epoch 单调 ----------
def test_merge_assigns_stable_monotonic_ids() -> None:
    s = merge_pt(
        SessionPrefState(),
        [_c("不要塑料、plastic", "dislike", keywords=["塑料", "plastic"])],
        retrieval="search",
    )
    assert [c.id for c in s.constraints] == ["c1"]
    s2 = merge_pt(s, [_c("偏好帆布、canvas", keywords=["帆布", "canvas"])], retrieval="augment")
    # 旧约束 id 不变（跨轮稳定），新约束续号
    assert [c.id for c in s2.constraints] == ["c1", "c2"]
    assert s2.next_id == 3
    # 换代后发号继续单调，id 永不复用（archived 里的 c1/c2 与新代不撞号）
    s3 = merge_pt(
        s2, [_c("不要皮革、leather", "dislike", keywords=["皮革", "leather"])], retrieval="search"
    )
    assert [c.id for c in s3.constraints] == ["c3"]


# ---------- I1 存续：后续轮不重放，旧约束原样活着 ----------
def test_constraints_survive_turns_without_replay() -> None:
    s = merge_pt(
        SessionPrefState(),
        [_c("不要塑料、plastic", "dislike", keywords=["塑料", "plastic"])],
        retrieval="search",
        budget_usd=14.0,
    )
    # 第二轮只补颜色偏好（M1 形态：planner 只输出本轮新说的，旧约束一个字不重放）
    s2 = merge_pt(s, [_c("偏好深色、dark", keywords=["深色", "dark"])], retrieval="reuse")
    assert s2.dislike_terms() == ["塑料", "plastic"]  # T1 硬排除仍生效
    assert s2.budget_usd == 14.0  # 预算未提及 → 保持
    # 第三轮什么约束都没说（纯追问）
    s3 = merge_pt(s2, [], retrieval="reuse")
    assert s3.dislike_terms() == ["塑料", "plastic"]
    assert [c.id for c in s3.constraints] == ["c1", "c2"]


# ---------- I2 撤回三态 ----------
def test_retract_verified_deletes() -> None:
    s = merge_pt(
        SessionPrefState(),
        [_c("不要塑料、plastic", "dislike", keywords=["塑料", "plastic"], quote="不要塑料的")],
        retrieval="search",
    )
    s2 = merge_pt(s, [], retract_ids=["c1"], user_utterance="算了，塑料的也行", retrieval="reuse")
    assert s2.constraints == []  # 词面呼应（塑料 ∈ 原话）→ 删
    assert s2.dislike_terms() == []


def test_retract_hallucinated_id_ignored() -> None:
    s = merge_pt(
        SessionPrefState(),
        [_c("不要塑料、plastic", "dislike", keywords=["塑料", "plastic"])],
        retrieval="search",
    )
    s2 = merge_pt(s, [], retract_ids=["c99"], user_utterance="算了，塑料的也行", retrieval="reuse")
    assert [c.id for c in s2.constraints] == ["c1"]  # 幻觉 id → 不删，约束可见可纠


def test_retract_without_word_echo_refused() -> None:
    """LLM 抄错 id（用户说的是放开预算，它却把塑料那条填进 retract_ids）→ 词面闸挡下。"""
    s = merge_pt(
        SessionPrefState(),
        [_c("不要塑料、plastic", "dislike", keywords=["塑料", "plastic"])],
        retrieval="search",
    )
    s2 = merge_pt(
        s, [], retract_ids=["c1"], user_utterance="预算放开点，贵的也看看", retrieval="augment"
    )
    assert [c.id for c in s2.constraints] == ["c1"]  # 原话与「塑料」无呼应 → 不删


def test_retract_matches_english_utterance_via_norm() -> None:
    """撤回核验走 norm_keywords（中→英扩词）：英文原话撤中文记的约束也对得上。"""
    s = merge_pt(
        SessionPrefState(),
        [_c("不要塑料", "dislike", keywords=["塑料"])],  # 只记了中文原子词
        retrieval="search",
    )
    s2 = merge_pt(
        s, [], retract_ids=["c1"], user_utterance="plastic is fine now", retrieval="reuse"
    )
    assert s2.constraints == []  # normalize_terms(塑料)⊇plastic → 呼应成立


# ---------- 归并：改口升级「只 add 不 retract」也收敛成一条 ----------
def test_merge_soft_to_hard_upgrade_inherits_id() -> None:
    """M3 形态：「尽量别花哨」→「绝对不要花哨」。升级轮 planner 只输出新的硬约束（不一定
    记得同时 retract），词面重叠归并必须兜住——P_t 里只剩一条硬的，且身份沿用旧 id。"""
    s = merge_pt(
        SessionPrefState(),
        [_c("尽量避免花哨、flashy", "dislike", keywords=["花哨", "flashy"], blocking=False)],
        retrieval="search",
    )
    s2 = merge_pt(
        s,
        [_c("不要花哨、flashy", "dislike", keywords=["花哨", "flashy"], blocking=True)],
        retrieval="reuse",
    )
    assert len(s2.constraints) == 1
    assert s2.constraints[0].id == "c1"  # 继承旧身份
    assert s2.constraints[0].blocking is True  # 档位升级生效
    assert s2.dislike_terms() == ["花哨", "flashy"]
    assert s2.soft_dislike_terms() == []


def test_merge_cross_language_restatement_dedupes() -> None:
    """真实形态：第一轮记了中文原子词，第二轮 LLM 吐的是英文 token——同一件事必须归并，
    否则 P_t 渲染出两行近似噪声（改造前 slug 漂移的经典长法）。"""
    s = merge_pt(
        SessionPrefState(), [_c("不要塑料", "dislike", keywords=["塑料"])], retrieval="search"
    )
    s2 = merge_pt(s, [_c("不要plastic", "dislike", keywords=["plastic"])], retrieval="reuse")
    assert len(s2.constraints) == 1
    assert s2.constraints[0].id == "c1"


def test_merge_unrelated_constraint_appends() -> None:
    s = merge_pt(
        SessionPrefState(),
        [_c("不要塑料、plastic", "dislike", keywords=["塑料", "plastic"])],
        retrieval="search",
    )
    s2 = merge_pt(
        s, [_c("不要皮革、leather", "dislike", keywords=["皮革", "leather"])], retrieval="reuse"
    )
    assert [c.id for c in s2.constraints] == ["c1", "c2"]  # 无词面重叠 → 各是各的


# ---------- I3 换代：search 清代归档，reuse/augment 不动 ----------
def test_epoch_bumps_on_search_and_archives_one_generation() -> None:
    """M4 形态：保温杯聊完换跑步鞋。旧约束整体挪 archived（只留一代），消费接口零感知。"""
    s = merge_pt(
        SessionPrefState(),
        [_c("不要粉色、pink", "dislike", keywords=["粉色", "pink"])],
        retrieval="search",
        category="保温杯",
    )
    assert s.epoch == 0  # 首轮 search：没有旧约束，不算换代
    s2 = merge_pt(
        s,
        [_c("偏好透气、breathable", keywords=["透气", "breathable"])],
        retrieval="search",
        category="跑步鞋",
    )
    assert s2.epoch == 1
    assert [c.id for c in s2.constraints] == ["c2"]  # 新代只有本轮约束
    assert [c.id for c in s2.archived] == ["c1"]  # 上代归档（排障可见）
    assert s2.dislike_terms() == []  # 旧代不再进任何消费接口——不杀新品类
    # 再换代：archived 只留一代（上上代直接丢）
    s3 = merge_pt(s2, [], retrieval="search", category="键盘")
    assert [c.id for c in s3.archived] == ["c2"]


def test_reuse_and_augment_do_not_bump_epoch() -> None:
    s = merge_pt(
        SessionPrefState(), [_c("不要塑料", "dislike", keywords=["塑料"])], retrieval="search"
    )
    for mode in ("reuse", "augment"):
        s2 = merge_pt(s, [], retrieval=mode)
        assert s2.epoch == 0
        assert [c.id for c in s2.constraints] == ["c1"]  # 收紧/放宽是同一意图的延续


# ---------- 预算 / slots 语义（沿用，含 clear_budget 单列的论证）----------
def test_merge_budget_none_keeps_prev() -> None:
    s = merge_pt(SessionPrefState(), [], budget_usd=100.0)
    assert merge_pt(s, [], budget_usd=None).budget_usd == 100.0  # 未提及 → 保持
    assert merge_pt(s, [], budget_usd=200.0).budget_usd == 200.0  # 明确改 → 更新


def test_merge_clear_budget_removes_it() -> None:
    """用户说「算了，不限预算」——这**必须能表达**（M5 形态）。

    ``budget_usd=None`` 的语义已经被「本轮没提预算」占了，所以取消预算需要独立的 clear_budget。
    没有它，撤销要么给 None（=沿用旧预算，错）要么编大数字（更错），而 item_picker 里那条
    「本轮没传预算就用 P_t 兜底」的逻辑会让旧预算在后续每一轮继续悄悄卡人。
    """
    s = merge_pt(SessionPrefState(), [], budget_usd=100.0)
    assert merge_pt(s, [], clear_budget=True).budget_usd is None


def test_merge_slots_patch_is_incremental() -> None:
    s = merge_pt(SessionPrefState(), [], slots_patch={"size": "42", "dest_country": "JP"})
    s2 = merge_pt(s, [], slots_patch={"size": "43"})  # 只提 size → dest_country 不动
    assert s2.slots == {"size": "43", "dest_country": "JP"}


# ---------- 会话内约束一律硬执行（供 item_picker 机制性强制）----------
def test_dislike_terms_all_hard() -> None:
    """P_t 的 dislike **全部**进硬淘汰——用户本轮刚亲口说的，明确性极高。

    这与长期库「agent 学到的只减分」是**刻意的不对称**：硬 / 软的分界是**信息的来源**，
    不是 LLM 的置信度。
    """
    s = merge_pt(
        SessionPrefState(),
        [_c("不要塑料、plastic", "dislike", keywords=["塑料", "plastic"])],
    )
    assert s.dislike_terms() == ["塑料", "plastic"]


def test_like_terms_from_keywords() -> None:
    s = merge_pt(SessionPrefState(), [_c("偏好金属", keywords=["金属"])])
    assert s.like_terms() == ["金属"]


# ---------- pt.json roundtrip 与容错 ----------
def test_save_load_roundtrip(tmp_path: Path) -> None:
    s = merge_pt(
        SessionPrefState(),
        [_c("不要塑料、plastic", "dislike", keywords=["塑料", "plastic"], quote="不要塑料的")],
        budget_usd=42.0,
        category="旅行收纳",
        retrieval="search",
    )
    save_pt(tmp_path, s)
    loaded = load_pt(tmp_path)
    assert loaded.budget_usd == 42.0
    assert loaded.category == "旅行收纳"
    assert [(c.id, c.source_quote) for c in loaded.constraints] == [("c1", "不要塑料的")]
    assert loaded.next_id == 2 and loaded.epoch == 0
    assert loaded.updated_at  # save 刷新了时间戳


def test_load_missing_returns_empty(tmp_path: Path) -> None:
    assert load_pt(tmp_path).is_empty()


def test_load_corrupt_degrades_to_empty(tmp_path: Path) -> None:
    (tmp_path / "pt.json").write_text("{ 坏 json", encoding="utf-8")
    assert load_pt(tmp_path).is_empty()  # 损坏 → 降级空，不抛


def test_load_old_slug_format_degrades_to_empty(tmp_path: Path) -> None:
    """旧格式 pt.json（id 化之前：约束带 slug、状态带 open_questions）→ 按空开局，不写迁移。

    这是**钉行为**的测试：extra="forbid" 使旧字段触发 ValidationError → 走「读不出当空」容错。
    若未来有人把 forbid 放松成 ignore，旧文件会静默载入成「无 id 约束」，撤回 / 归并全部失灵
    ——这条测试就是那时的绊线。
    """
    old = {
        "current_intent": "买旅行收纳",
        "budget_usd": 42.0,
        "category": "旅行收纳",
        "slots": {},
        "constraints": [
            {
                "slug": "plastic",
                "content": "不要塑料",
                "polarity": "dislike",
                "keywords": ["塑料", "plastic"],
                "category": "material",
                "turn_added": 1,
                "blocking": True,
            }
        ],
        "open_questions": [],
        "turn": 1,
        "updated_at": datetime.now(UTC).isoformat(),
    }
    (tmp_path / "pt.json").write_text(json.dumps(old, ensure_ascii=False), encoding="utf-8")
    assert load_pt(tmp_path).is_empty()


def test_load_expired_returns_empty(tmp_path: Path) -> None:
    """同一 thread 隔很久再来 → 视为新一段选购，旧约束不该复活。"""
    stale = SessionPrefState(
        constraints=[_c("要蓝色", keywords=["蓝色"])],
        updated_at=(datetime.now(UTC) - timedelta(hours=999)).isoformat(),
    )
    (tmp_path / "pt.json").write_text(stale.model_dump_json(), encoding="utf-8")
    assert load_pt(tmp_path).is_empty()


# ---------- render ----------
def test_render_empty_placeholder() -> None:
    assert "尚无累积约束" in SessionPrefState().render()


def test_render_shows_constraints_and_budget_without_ids() -> None:
    s = merge_pt(
        SessionPrefState(),
        [
            _c("不要塑料、plastic", "dislike", keywords=["塑料", "plastic"]),
            _c("偏好蓝色、blue", keywords=["蓝色", "blue"]),
        ],
        budget_usd=42.0,
    )
    text = s.render()
    assert "预算" in text and "$42" in text
    assert "不要塑料、plastic" in text
    assert "偏好蓝色、blue" in text
    # 主 loop 的可读渲染不带内部 id——那是 planner 撤回通道的引用键，混进来只会被模型念给用户
    assert "[c1]" not in text


def test_render_distinguishes_hard_exclude_from_soft_dislike() -> None:
    """render 的标签必须带 blocking 档位——**续聊轮的 planner 是照着这段文字重新识别约束的**。

    真实 bug（端到端抓到）：标签只按 polarity 分「排除 / 需要」两档时，软避讳「尽量别太花哨」被
    渲染成「本轮约束（排除）：…」，第二轮 planner 读到「排除」二字，就把「花哨」原样填进
    exclude_keywords，于是它从减分升级成硬淘汰，用户被误杀一大片商品还看不见原因。
    """
    s = merge_pt(
        SessionPrefState(),
        [
            _c("不要塑料", "dislike", keywords=["塑料"], blocking=True),
            _c("尽量避免花哨", "dislike", keywords=["花哨"], blocking=False),
        ],
    )
    text = s.render()
    assert "硬排除" in text and "软避讳" in text
