"""合成 query 质量门（refdocs 04-2 §5.5.3 四个坑 + §5.5.7 五条红线）。

合成数据不清洗就训，"训出来的模型大概率比通用底座还差"——而且坏味道会在后续挖负例时被指数
放大。所以这道门是**卡在训练之前**的，不达标就整批废掉重生成，不要试图修一修再用。

四项检查（阈值按 refdocs，语言分档那条是我们自己加的）：

1. **模板化率** < 15%：句式雷同的 query 只会让模型学模板。用 3-gram 重合度近似 SimHash。
2. **标题照抄率** < 10%：LLM 最爱直接抄标题，抄了 embedding 就退化成字面相似度模型。
   判定用**连续 4 词**（中文 5 字）——refdocs 原文就是"连续 4 字以上"。第一版我写成 3-gram，
   过严了一档，把正常的核心词命中也算成抄。
   **且 L1 不计入判定**：L1 的定义就是"用户记得商品大致名称时怎么搜、包含核心词"，与标题
   重合是设计使然，拿照抄率卡它等于否定它自己的设计目标。L1 的重合率照常报告、只是不作
   为门禁——它偏高时该做的是降低 L1 在训练集里的权重，不是打回整批重生成。
3. **语言分档正确率** ≥ 90%：L1/L2 必须英文（贴 ESCI 尺子）、L3/L4 必须中文（贴线上分布）。
   这条 refdocs 没有，是我们为了修 train/serve skew 专门设计的，必须验。
4. **L3 品类词泄漏率** < 20%：L3 是场景搜，出现品类词本身就失去了"语义跨度"的训练价值。

用法：``uv run --group train python scripts/train/synth_quality_gate.py --input synth_train.jsonl``
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "train"

THRESHOLDS = {"template": 0.15, "copy": 0.10, "lang": 0.90, "leak": 0.20}
CJK = re.compile(r"[一-鿿]")


def has_cjk(s: str) -> bool:
    return bool(CJK.search(s))


def word_ngrams(s: str, n: int = 4) -> set[str]:
    w = re.findall(r"[a-z0-9]+", s.lower())
    return {" ".join(w[i : i + n]) for i in range(max(0, len(w) - n + 1))}


def is_copied(q: str, title: str) -> bool:
    """连续 4 词（中文 5 字）命中标题即判为照抄，口径对齐 refdocs §5.5.3。"""
    if has_cjk(q):
        return len(q) >= 5 and any(q[i : i + 5] in title for i in range(len(q) - 4))
    return bool(word_ngrams(q) & word_ngrams(title))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="synth_train.jsonl")
    args = ap.parse_args()

    rows = [json.loads(x) for x in (DATA_DIR / args.input).open(encoding="utf-8") if x.strip()]
    by_lv: dict[str, list[dict]] = {}
    for r in rows:
        by_lv.setdefault(r["level"], []).append(r)
    print(f"总计 {len(rows)} 条，分档：" + "  ".join(f"{k}={len(v)}" for k, v in sorted(by_lv.items())))

    # 1. 模板化率：取 query 前 3 个词作句式指纹，重复度过高即模板化
    heads = Counter()
    for r in rows:
        toks = re.findall(r"[\w一-鿿]+", r["query"])[:3]
        heads[" ".join(toks)] += 1
    dup = sum(c for h, c in heads.items() if c > len(rows) * 0.01)
    template_rate = dup / len(rows)

    # 2. 标题照抄率：L1 报告但不计入门禁（见模块 docstring）
    gated = [r for r in rows if r["level"] != "L1"]
    copy_rate = sum(is_copied(r["query"], r["pos"][0]) for r in gated) / max(1, len(gated))
    per_level = {
        lv: sum(is_copied(r["query"], r["pos"][0]) for r in rs) / len(rs)
        for lv, rs in sorted(by_lv.items())
    }

    # 3. 语言分档正确率
    lang_ok = sum(
        (not has_cjk(r["query"])) if r["level"] in ("L1", "L2") else has_cjk(r["query"])
        for r in rows
    )
    lang_rate = lang_ok / len(rows)

    # 4. L3 品类词泄漏：标题尾部类目词直接出现在 L3 query 里
    l3 = by_lv.get("L3", [])
    leak = 0
    for r in l3:
        cats = re.findall(r"[一-鿿]{2,}", r["pos"][0].split("|")[-1])
        leak += any(c in r["query"] for c in cats)
    leak_rate = leak / len(l3) if l3 else 0.0

    results = {
        "template": (template_rate, THRESHOLDS["template"], "<"),
        "copy": (copy_rate, THRESHOLDS["copy"], "<"),
        "lang": (lang_rate, THRESHOLDS["lang"], ">="),
        "leak": (leak_rate, THRESHOLDS["leak"], "<"),
    }
    names = {
        "template": "模板化率", "copy": "标题照抄率",
        "lang": "语言分档正确率", "leak": "L3 品类词泄漏率",
    }
    print("\n=== 质量门 ===")
    failed = []
    for k, (val, thr, op) in results.items():
        ok = val < thr if op == "<" else val >= thr
        print(f"{names[k]:<16} {val:6.2%}  阈值 {op}{thr:.0%}   {'✅' if ok else '❌'}")
        if not ok:
            failed.append(names[k])

    print("\n分档照抄率（L1 仅报告，不计入门禁）：" +
          "  ".join(f"{lv}={v:.1%}" for lv, v in per_level.items()))
    if per_level.get("L1", 0) > 0.20:
        print("  ⚠️ L1 照抄率偏高——训练时建议降低 L1 权重或直接只用 L3/L4")
    print("\n判定：" + ("✅ 全部通过，可进训练" if not failed else f"❌ 不通过：{'、'.join(failed)}"))
    for lv in sorted(by_lv):
        print(f"\n[{lv}] 样例：" + " / ".join(r["query"][:34] for r in by_lv[lv][:3]))


if __name__ == "__main__":
    main()
