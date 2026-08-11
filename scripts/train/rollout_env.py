"""S3 的 **rollout 环境**：把模型吐的一段文本，变成一个 reward 数字。

三件事，缺一不可：
1. **解析**：从 completion 里抠出 PlanOutput 的 JSON（口径与 `eval_planner_format.py` 一致，
   允许 ```json 围栏与前后废话——线上有 structured output 兜底，判太严会低估真实可用性）；
2. **检索**：把 plan 的 keywords 真打进 Qdrant（GPU 机本地那份 138 万点），拿 top20 标题。
   这是 R_retrieval 那 45% 的来源，也是本项目相对 refdocs 08-2「录制回放」的实质差别；
3. **打分**：调 `app/eval/planner_reward.py` 的 `compute_reward`。

**reward 代码不复制一份到 GPU 机**：训练侧和线上侧必须共用同一个 `planner_reward.py`，两份
实现迟早漂移，而 reward 漂移是「训练分涨了、线上没动」这种最难查的事故。做法是把仓库 rsync
过来后**按文件路径加载模块**，并预先塞一个空的 `app` 父包——直接 `import app.eval.planner_reward`
会先执行 `app/memory/__init__.py`，那里 import 了 store/injector 一串 sqlalchemy 依赖，GPU
机上装不起也没必要装。reward 真正用到的 `app/utils/terms.py` 与 `app/memory/domains.py`
本身只依赖标准库。

**query 编码走 GPU 机本地 bge-m3**（不是线上 embedding API）：等价性由
`scripts/train/verify_embed_parity.py` 实测背书——余弦 1.0、top20 重合 97.5%、top1 100%。
不等价的话 reward 会静默失真，所以那条验证是这条通路的前置件，不是可选项。

用法（GPU 机）::

    python rollout_env.py --selftest --data planner_grpo_dev.jsonl --limit 20
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import sys
import types
from pathlib import Path
from typing import Any

import requests

REPO = Path(os.environ.get("GLOBEX_REPO", str(Path(__file__).resolve().parents[2])))
EMBED_URL = os.environ.get("ROLLOUT_EMBED_URL", "http://127.0.0.1:8095/v1/embeddings")
QDRANT_URL = os.environ.get("QDRANT_URL", "http://127.0.0.1:6333")
COLLECTION = os.environ.get("QDRANT_COLLECTION", "globex_items")
TOP_K = 20
HTTP_TIMEOUT = 30


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"❌ 加载不了 {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_reward():
    """按文件加载仓库里的 reward，绕开 app 包的重依赖 __init__（理由见模块 docstring）。"""
    for pkg in ("app", "app.eval", "app.memory", "app.utils"):
        if pkg not in sys.modules:
            m = types.ModuleType(pkg)
            m.__path__ = [str(REPO / Path(*pkg.split(".")))]  # 让子模块可被定位
            sys.modules[pkg] = m
    _load_module("app.utils.terms", REPO / "app/utils/terms.py")
    _load_module("app.memory.domains", REPO / "app/memory/domains.py")
    return _load_module("app.eval.planner_reward", REPO / "app/eval/planner_reward.py")


def extract_json(text: str) -> dict | None:
    """与 eval_planner_format.py 逐字同口径——两把尺子量的必须是同一个东西。"""
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.S)
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


class Retriever:
    """keywords → 本地 bge-m3 编码 → Qdrant top-k 标题。

    **批量编码**：GRPO 一步要给几十上百条 rollout 打分，逐条 HTTP 会把 reward 变成瓶颈
    （每次往返 ~5ms，256 条就是 1.3s，和整步 rollout 一个量级）。embed server 的
    ``input`` 支持 list，一次发完。
    """

    def __init__(self, embed_url: str = EMBED_URL, qdrant_url: str = QDRANT_URL,
                 collection: str = COLLECTION, top_k: int = TOP_K) -> None:
        self.embed_url, self.qdrant_url = embed_url, qdrant_url.rstrip("/")
        self.collection, self.top_k = collection, top_k
        self.sess = requests.Session()

    def encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        r = self.sess.post(self.embed_url, json={"input": texts}, timeout=HTTP_TIMEOUT)
        r.raise_for_status()
        data = r.json()["data"]
        # OpenAI 兼容响应不保证按序返回，按 index 排一次——顺序错位会让每条 rollout 拿到
        # 别人的检索结果，分数照样算得出来，只是全错。
        data.sort(key=lambda d: d.get("index", 0))
        return [d["embedding"] for d in data]

    def search_batch(self, vecs: list[list[float]]) -> list[list[str]]:
        """**一次请求查完整批**。逐条查也能跑，但实测 128 条要 5.5s（打分比生成还慢，
        整步 rollout 的时间会被 reward 侧吃掉一半）——瓶颈全在 HTTP 往返，不在 Qdrant。"""
        if not vecs:
            return []
        body = {"searches": [
            {"query": v, "using": "dense", "limit": self.top_k, "with_payload": ["title"]}
            for v in vecs
        ]}
        r = self.sess.post(f"{self.qdrant_url}/collections/{self.collection}/points/query/batch",
                           json=body, timeout=HTTP_TIMEOUT)
        r.raise_for_status()
        return [
            [(p.get("payload") or {}).get("title", "") for p in res["points"]]
            for res in r.json()["result"]
        ]

    def titles_for(self, keyword_lists: list[list[str] | None]) -> list[list[str] | None]:
        """一组 plan 的 keywords → 各自的 top-k 标题。空 keywords 回 None（该维弃权）。"""
        idx, texts = [], []
        for i, kws in enumerate(keyword_lists):
            text = " ".join(str(k) for k in (kws or []) if str(k).strip())
            if text:
                idx.append(i)
                texts.append(text)
        out: list[list[str] | None] = [None] * len(keyword_lists)
        for i, titles in zip(idx, self.search_batch(self.encode(texts)), strict=True):
            out[i] = titles
        return out


def score_batch(completions: list[str], golds: list[dict], texts: list[str],
                retriever: Retriever | None, reward_mod: Any) -> list[Any]:
    """一批 rollout → 一批 RewardBreakdown。检索一次批量做完，再逐条打分。"""
    plans = [extract_json(c) for c in completions]
    if retriever is None:
        titles: list[list[str] | None] = [None] * len(plans)
    else:
        titles = retriever.titles_for([(p or {}).get("keywords") for p in plans])
    return [
        reward_mod.compute_reward(p, g, t, ti)
        for p, g, t, ti in zip(plans, golds, texts, titles, strict=True)
    ]


def _selftest(args) -> None:
    """S1 验收：**单条 rollout 延迟** + **reward 可复现** + **组内区分度**。

    第三项是 S0-5 桩数据标定管不到的：那次证的是「好答案和坏答案分得开」，这里问的是
    「**同一个模型自己采 8 条，彼此分不分得开**」——GRPO 的优势是组内相对量算的，组内全挤
    在一起的话，梯度就是纯噪声，训练只会原地抖。
    """
    import statistics
    import time

    rows = [json.loads(x) for x in Path(args.data).open(encoding="utf-8") if x.strip()]
    if args.limit:
        rows = rows[: args.limit]
    reward_mod = load_reward()
    retriever = None if args.no_retrieval else Retriever(top_k=args.top_k)

    from vllm import LLM, SamplingParams

    kw: dict[str, Any] = dict(model=args.model, dtype="bfloat16", max_model_len=args.max_len,
                              gpu_memory_utilization=args.gpu_util, enable_prefix_caching=True)
    lora = None
    if args.adapter:
        from vllm.lora.request import LoRARequest
        kw.update(enable_lora=True, max_lora_rank=args.lora_rank)
        lora = LoRARequest("sft", 1, args.adapter)
    llm = LLM(**kw)
    prompts = [
        f"<|im_start|>system\n{r['messages'][0]['content']}<|im_end|>\n"
        f"<|im_start|>user\n{r['messages'][1]['content']}<|im_end|>\n<|im_start|>assistant\n"
        for r in rows
    ]
    sp = SamplingParams(n=args.group_size, temperature=args.temperature, top_p=0.95,
                        max_tokens=args.gen_tokens, seed=args.seed)

    def gen_pass() -> tuple[list[str], list[dict], list[str], float]:
        t0 = time.perf_counter()
        outs = llm.generate(prompts, sp, lora_request=lora, use_tqdm=False)
        t_gen = time.perf_counter() - t0
        comps, golds, texts = [], [], []
        for r, o in zip(rows, outs, strict=True):
            gold = json.loads(r["golden_json"])
            for c in o.outputs:
                comps.append(c.text)
                golds.append(gold)
                texts.append(r["text"])
        return comps, golds, texts, t_gen

    def score_pass(comps, golds, texts) -> tuple[list[float], float]:
        t0 = time.perf_counter()
        brs = score_batch(comps, golds, texts, retriever, reward_mod)
        return [b.total for b in brs], time.perf_counter() - t0

    c1, golds, texts, t_gen1 = gen_pass()
    r1, t_rw1 = score_pass(c1, golds, texts)
    # **reward 复现性**（验收项）：同一批 completion 再打一遍，必须逐位一致。
    r1b, _ = score_pass(c1, golds, texts)
    # **生成复现性**（观测项，不是验收项）：vLLM 即使固定 seed，两次调用的 batch 组合与
    # 前缀缓存状态不同，logits 会有浮点级差异、采样点可能分叉。GRPO 本来就要随机采样，
    # 这条不一致不影响训练；真正不能抖的是 reward 函数本身。
    c2, _, _, t_gen2 = gen_pass()
    r2, t_rw2 = score_pass(c2, golds, texts)

    g = args.group_size
    groups = [r1[i * g:(i + 1) * g] for i in range(len(rows))]
    within = [statistics.pstdev(x) for x in groups if len(x) > 1]
    n = len(c1)
    report = {
        "样本": {"prompt 数": len(rows), "group_size": g, "temperature": args.temperature,
                 "检索": "关闭（--no-retrieval）" if retriever is None else COLLECTION,
                 "rollout 条数": n},
        "耗时": {
            "生成 s": [round(t_gen1, 3), round(t_gen2, 3)],
            "打分 s（含检索）": [round(t_rw1, 3), round(t_rw2, 3)],
            "端到端每条 ms": round((t_gen1 + t_rw1) * 1000 / max(n, 1), 1),
            "一步 rollout 总计 s": round(t_gen1 + t_rw1, 3),
        },
        "parse 失败率": round(sum(x == reward_mod.PARSE_FAIL_REWARD for x in r1) / max(n, 1), 4),
        "reward 均值": round(statistics.mean(r1), 4),
        "reward 全局 σ": round(statistics.pstdev(r1), 4),
        "组内 σ 均值": round(statistics.mean(within), 4) if within else None,
        "组内 σ 为 0 的组数": sum(s == 0 for s in within),
        "【验收】reward 函数可复现": r1 == r1b,
        "生成可复现（观测，非验收）": {
            "completion 逐字一致": c1 == c2,
            "reward 均值两遍": [round(statistics.mean(r1), 4), round(statistics.mean(r2), 4)],
        },
    }
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--model", default="")
    ap.add_argument("--adapter", default="")
    ap.add_argument("--data", default="planner_grpo_dev.jsonl")
    ap.add_argument("--limit", type=int, default=16)
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--gen-tokens", type=int, default=256)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--gpu-util", type=float, default=0.6)
    ap.add_argument("--top-k", type=int, default=TOP_K)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-retrieval", action="store_true")
    ap.add_argument("--out", default="rollout_selftest.json")
    args = ap.parse_args()
    if not args.selftest:
        raise SystemExit("本模块主要给 GRPO 插件 import；跑自检请加 --selftest")
    if not args.model:
        raise SystemExit("--selftest 需要 --model")
    _selftest(args)


if __name__ == "__main__":
    main()
