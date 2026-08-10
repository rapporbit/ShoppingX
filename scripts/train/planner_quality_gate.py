"""S0-3：合并三源 → 过质量门 → 切分。planner 训练集的最后一道关。

三源（见 ROADMAP M23 S0-1）：
- `planner_anchors.jsonl`   真实线上 query（94 条）——量小但**分布最真**
- `planner_queries.jsonl`   维度矩阵合成——保覆盖面
- `planner_adversarial.jsonl` bad case 族定向——保长尾

**切分不是随机切的**：94 条真实锚**整体进 dev**。理由是 dev 的职责是回答「训出来的模型在
线上真实分布上行不行」，用合成数据当 dev 只能回答「在合成分布上行不行」——M21 栽过一次同款：
ESCI（3 词英文关键词）上涨了 6.6%，但那把尺子量不出线上中文口语的表现。真实数据稀缺时，
它的最高价值用法是当尺子，不是当训练料。

质量门七项，全部**只报告不静默丢弃**（除精确重复）：数据被门挡掉多少、为什么挡，
得让人看见。M21 的合成管线是「质量门全过」才敢用的，这里沿用同一条规矩。

用法：``uv run python scripts/train/planner_quality_gate.py``
"""

from __future__ import annotations

import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data" / "train"
ANCHORS = DATA_DIR / "planner_anchors.jsonl"
SYNTH = DATA_DIR / "planner_queries.jsonl"
ADV = DATA_DIR / "planner_adversarial.jsonl"
OUT = DATA_DIR / "planner_dataset.jsonl"
REPORT = DATA_DIR / "planner_dataset_report.json"

_NORM = re.compile(r"[\s，。,.、！!？?]+")
# 模板腔：LLM 造数据的通病，一旦混进训练集，planner 会学到线上不存在的输入形态
_TEMPLATE_TONE = re.compile(r"我需要一款|请帮我推荐一款|符合以下要求|如下要求|以下是我的需求")
_CJK = re.compile(r"[一-鿿]")

# 长度红线：真实锚实测中位 17 / p90 31。首轮超过 60 字基本可以断定是需求文档腔
MAX_FIRST_TURN = 60
MIN_FIRST_TURN = 5


def _norm(s: str) -> str:
    return _NORM.sub("", str(s))


def _load(path: Path, source: str) -> list[dict]:
    """三源 schema 不同，在这里统一成 {id, source, category, family, dims, turns}。"""
    if not path.exists():
        print(f"  [warn] 缺 {path.name}，跳过该源")
        return []
    rows = []
    for line in path.open(encoding="utf-8"):
        if not line.strip():
            continue
        d = json.loads(line)
        if source == "real":  # 锚文件是「一行一条 query」，其余两源是「一行一个会话」
            rows.append({
                "id": d["id"], "source": "real", "category": "", "family": "",
                "dims": {}, "bucket": d.get("bucket", ""),
                "turns": [{"turn": 0, "text": d["text"], "followup_type": None}],
                "is_followup_fragment": d.get("is_followup", False),
            })
        else:
            rows.append({
                "id": d["id"], "source": source, "category": d.get("category", ""),
                "family": d.get("family", ""), "dims": d.get("dims", {}),
                "turns": d["turns"], "is_followup_fragment": False,
            })
    return rows


def _gate(rows: list[dict]) -> tuple[list[dict], Counter]:
    """七项质量门。只有「精确重复」和「空/超长」会真的丢，其余记账。"""
    seen: set[str] = set()
    kept, stats = [], Counter()
    for r in rows:
        first = r["turns"][0]["text"] if r["turns"] else ""
        key = _norm(first)
        if not key:
            stats["空 query"] += 1
            continue
        if key in seen:
            stats["精确重复"] += 1
            continue
        n = len(key)
        if n > MAX_FIRST_TURN or n < MIN_FIRST_TURN:
            stats[f"长度越界(<{MIN_FIRST_TURN} 或 >{MAX_FIRST_TURN})"] += 1
            continue
        seen.add(key)
        if _TEMPLATE_TONE.search(first):
            stats["模板腔(保留但标记)"] += 1
            r["flag_template_tone"] = True
        if not _CJK.search(first):
            stats["无中文(保留但标记)"] += 1
            r["flag_no_cjk"] = True
        # 追问轮只写增量：与首轮字面重合过高说明 LLM 在复述，训练价值低
        for t in r["turns"][1:]:
            if _norm(t["text"]) and _norm(t["text"]) in key:
                stats["追问轮复述首轮(标记)"] += 1
                r["flag_echo"] = True
        kept.append(r)
    return kept, stats


def _split(rows: list[dict], rng: random.Random) -> dict[str, list[dict]]:
    """真实锚整体进 dev；合成与对抗按 85/15 切 train/test，且每个 family 都留够 test。"""
    dev = [r for r in rows if r["source"] == "real"]
    rest = [r for r in rows if r["source"] != "real"]
    by_fam: dict[str, list[dict]] = defaultdict(list)
    for r in rest:
        by_fam[r["family"] or "synth"].append(r)
    train, test = [], []
    for fam, group in by_fam.items():
        rng.shuffle(group)
        n_test = max(2, round(len(group) * 0.15)) if len(group) > 4 else 0
        test += group[:n_test]
        train += group[n_test:]
    return {"train": train, "dev": dev, "test": test}


def _profile(rows: list[dict]) -> dict:
    """画像：与真实锚对比的那几个数，是判断「合成分布跑没跑偏」的唯一依据。"""
    firsts = [_norm(r["turns"][0]["text"]) for r in rows if r["turns"]]
    lens = sorted(len(f) for f in firsts)
    n = len(rows) or 1
    multi = sum(1 for r in rows if len(r["turns"]) > 1)
    frag = sum(1 for r in rows if r.get("is_followup_fragment"))
    return {
        "条数": len(rows),
        "首轮长度中位": lens[len(lens) // 2] if lens else 0,
        "首轮长度p90": lens[int(len(lens) * 0.9)] if lens else 0,
        "带追问轮比例": round((multi + frag) / n, 3),
        "带预算比例": round(sum(1 for f in firsts if re.search(r"预算|不超|以内|以下|块|元|美元|人民币|\$", f)) / n, 3),
        "带排除比例": round(sum(1 for f in firsts if re.search(r"不要|不想|别太|除了|不能", f)) / n, 3),
    }


def main() -> None:
    rows = _load(ANCHORS, "real") + _load(SYNTH, "synth") + _load(ADV, "adversarial")
    print(f"三源合计 {len(rows)} 条")

    kept, stats = _gate(rows)
    print("\n质量门：")
    for k, v in stats.most_common():
        print(f"  {k:32s} {v}")
    print(f"  {'通过':32s} {len(kept)}")

    rng = random.Random(42)
    splits = _split(kept, rng)

    # 真实锚是尺子：合成数据的画像要贴着它，偏了就说明维度矩阵或 prompt 需要回调
    print("\n分布画像（真实锚 = 基准尺子）：")
    real = [r for r in kept if r["source"] == "real"]
    synth = [r for r in kept if r["source"] == "synth"]
    adv = [r for r in kept if r["source"] == "adversarial"]
    prof = {"real": _profile(real), "synth": _profile(synth), "adversarial": _profile(adv)}
    keys = list(prof["real"])
    print(f"  {'指标':16s}" + "".join(f"{s:>14s}" for s in prof))
    for k in keys:
        print(f"  {k:16s}" + "".join(f"{prof[s][k]:>14}" for s in prof))

    fam = Counter(r["family"] for r in adv)
    print("\nbad case 族分布：", dict(fam))
    print("品类覆盖：", len({r["category"] for r in kept if r["category"]}), "个")

    with OUT.open("w", encoding="utf-8") as f:
        for split, group in splits.items():
            for r in group:
                f.write(json.dumps({**r, "split": split}, ensure_ascii=False) + "\n")
    REPORT.write_text(json.dumps({
        "gate": dict(stats), "profile": prof,
        "splits": {k: len(v) for k, v in splits.items()},
        "families": dict(fam),
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n切分：" + " / ".join(f"{k}={len(v)}" for k, v in splits.items()))
    print(f"→ {OUT.relative_to(PROJECT_ROOT)}（含 split 字段）")
    print("下一步 S0-2：跑 golden 标注（budget 走规则、category/domains 走三次投票）")


if __name__ == "__main__":
    main()
