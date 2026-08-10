"""按 refdocs §10.3 Stage C 训 reranker：分级 label + ApproxNDCG，跑在 GPU 机。

**为什么不继续用 ms-swift。** refdocs Stage C 的精髓不是 ApproxNDCG 这个 loss 本身，而是它
配套的分级采样（§10.3「1个3分 + 1个2分 + 2个1分 + 3个0分，保持相关性分布真实」）。
ApproxNDCG 在二值 label 下会退化成近似 MRR，跟 listwise CE 拉不开差距——**分级标注才是它
发挥价值的前提**。而 ms-swift 的 reranker 模板只吃 positive/negative 二值，四档表达不了。
自写一份反而更短更可控。

**2×2 析因，把两个因素分开量：**

              CE loss            ApproxNDCG
  二值 label   r1（已跑，swift）   r4
  分级 label   r3（ListNet）       r2 ← refdocs Stage C

只跑 r2 的话，涨了不知道是分级的功劳还是 loss 的功劳，跌了也不知道该怪谁。

**四档从哪来**：ESCI 人工标注天然对应 refdocs §10.2 的 3/2/1/0——E=3、S=2（可替代但不是要
的那个）、C=1（互补配件）、I 与 ANN 挖的=0。**但覆盖率要诚实**：实测只有 33% 的组含 S 或 C
档（29.9% 有 S、3.3% 最高只到 C），其余 66.8% 的组除正例外全是 0 档。分级信号只在这三分之一
的组里起作用，r2 的天花板由此封顶。

用法::

    CUDA_VISIBLE_DEVICES=5 python train_reranker_graded.py \\
        --train graded_r2_train.jsonl --val graded_r2_val.jsonl \\
        --loss approxndcg --graded --out output/r2
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_scheduler

MODEL_NAME = "BAAI/bge-reranker-v2-m3"
MAX_LEN = 160  # 实测 query 中位 8 token、doc p99 79，160 覆盖到 p99 还有富余


class GroupDataset(Dataset):
    """一条样本 = 一个 query 组：docs 列表 + 同长的 gain 列表。"""

    def __init__(self, path: str, graded: bool) -> None:
        self.rows = [json.loads(x) for x in Path(path).open(encoding="utf-8") if x.strip()]
        self.graded = graded

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        r = self.rows[i]
        labels = r["labels"]
        if not self.graded:
            # 二值化：只有正例算 1，S/C 一并压成 0 —— 这正是 r1（ms-swift listwise）的口径，
            # 保持一致才能让 2×2 的四格可比。
            labels = [1 if g >= 3 else 0 for g in labels]
        return {"query": r["query"], "docs": r["docs"], "labels": labels}


def collate(batch: list[dict], tok, max_len: int) -> dict:
    """把若干组拍平成 pair 序列，另存每组长度用于组内归约。"""
    queries, docs, labels, sizes = [], [], [], []
    for g in batch:
        queries += [g["query"]] * len(g["docs"])
        docs += g["docs"]
        labels += g["labels"]
        sizes.append(len(g["docs"]))
    enc = tok(queries, docs, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
    enc["group_labels"] = torch.tensor(labels, dtype=torch.float)
    enc["group_sizes"] = torch.tensor(sizes, dtype=torch.long)
    return enc


def approx_ndcg_loss(scores: torch.Tensor, gains: torch.Tensor, temp: float = 0.1) -> torch.Tensor:
    """ApproxNDCG（Qin et al. 2010）：用 sigmoid 近似排名，让 NDCG 可微。

    approx_rank_i = 1 + Σ_{j≠i} σ((s_j − s_i)/T)，再按标准 DCG 公式代入。
    直接优化业务指标（我们的主指标就是 ndcg@8），而不是 CE 那种代理目标——这是 refdocs
    §10.3 Stage C 的核心主张。
    """
    diff = scores.unsqueeze(0) - scores.unsqueeze(1)  # [n,n]，diff[i,j] = s_j - s_i
    pairs = torch.sigmoid(diff / temp)
    ranks = 1.0 + (pairs.sum(dim=1) - 0.5)  # 减去 j=i 时的 σ(0)=0.5
    dcg = ((2.0**gains - 1.0) / torch.log2(ranks + 1.0)).sum()
    ideal = torch.sort(gains, descending=True).values
    positions = torch.arange(len(ideal), device=gains.device, dtype=gains.dtype)
    idcg = ((2.0**ideal - 1.0) / torch.log2(positions + 2.0)).sum()
    return 1.0 - dcg / idcg.clamp(min=1e-6)


def listwise_ce_loss(scores: torch.Tensor, gains: torch.Tensor) -> torch.Tensor:
    """组内 softmax 交叉熵。

    二值 gain 下即标准 listwise CE（目标 one-hot，等价 ms-swift 的 listwise_reranker）；
    分级 gain 下目标变成 softmax(gain) 的软分布，即 ListNet——这样「分级」这个因素在 CE
    这一列也真的起作用，2×2 才不是假的。
    """
    target = torch.softmax(gains, dim=0) if gains.max() > 1 else gains / gains.sum().clamp(min=1e-6)
    return -(target * torch.log_softmax(scores, dim=0)).sum()


def pointwise_bce_loss(scores: torch.Tensor, gains: torch.Tensor) -> torch.Tensor:
    """Pointwise BCE，目标 = gain/3（E=1 / S=.67 / C=.33 / 其余=0）。

    **它在这里不是 refdocs §10.3 说的「热启」——那个理由对续训不成立——而是为了保住绝对分数
    校准。** 纯 listwise 只约束组内相对序，绝对分数怎么漂都不影响 loss，实测 r3 训完分数分布
    被压到 .17~.85（原版跨满 0~1），直接导致下游 ``item_picker`` 那道 ``PICK_RERANK_FLOOR=0.2``
    的品类门失效：真实候选低于 .2 的比例从 base 的 58.9% 掉到 11.8%，门等于没开。
    BCE 把分数钉回「相关度」的绝对语义上，排序腿仍由 listwise 负责。
    """
    return F.binary_cross_entropy_with_logits(scores, (gains / 3.0).clamp(0, 1))


def batch_loss(logits: torch.Tensor, enc: dict, loss_name: str, bce_w: float = 0.0) -> torch.Tensor:
    """按组切开逐组算 loss 再平均——组是 listwise 的最小单位，跨组混算没有意义。"""
    losses = []
    off = 0
    for size in enc["group_sizes"].tolist():
        # loss 一律在 fp32 上算：ApproxNDCG 里 diff/temp 会把差值放大 10 倍，
        # bf16 那 8 位尾数在 sigmoid 饱和区直接丢精度，梯度会变成噪声。
        s = logits[off : off + size].float()
        g = enc["group_labels"][off : off + size].to(s.device, torch.float32)
        if g.max() > 0:  # 组内一个正例都没有（正例被截断）时跳过，否则 IDCG=0
            fn = approx_ndcg_loss if loss_name == "approxndcg" else listwise_ce_loss
            loss = fn(s, g)
            if bce_w:  # 联合 loss：排序由 listwise 管，绝对校准由 BCE 管
                loss = loss + bce_w * pointwise_bce_loss(s, g)
            losses.append(loss)
        off += size
    if not losses:
        return logits.sum() * 0.0
    return torch.stack(losses).mean()


@torch.inference_mode()
def evaluate(model, loader, loss_name: str, bce_w: float = 0.0) -> dict:
    """val 集上的 loss + 组内 NDCG + 命中率（正例排到第一的比例）。"""
    model.eval()
    tot_loss = tot_ndcg = tot_hit = n = 0.0
    for enc in loader:
        enc = {k: v.cuda() for k, v in enc.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(
                input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]
            ).logits.view(-1)
        logits = logits.float()
        tot_loss += float(batch_loss(logits, enc, loss_name, bce_w))
        off = 0
        for size in enc["group_sizes"].tolist():
            s = logits[off : off + size].float()
            g = enc["group_labels"][off : off + size].float()
            order = torch.argsort(s, descending=True)
            gs = g[order]
            pos = torch.arange(len(gs), device=gs.device, dtype=gs.dtype)
            dcg = ((2**gs - 1) / torch.log2(pos + 2)).sum()
            ideal = torch.sort(g, descending=True).values
            idcg = ((2**ideal - 1) / torch.log2(pos + 2)).sum().clamp(min=1e-6)
            tot_ndcg += float(dcg / idcg)
            tot_hit += float(gs[0] == g.max())
            n += 1
            off += size
    model.train()
    return {
        "loss": tot_loss / max(1, len(loader)),
        "ndcg": tot_ndcg / max(1.0, n),
        "hit@1": tot_hit / max(1.0, n),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--loss", choices=("ce", "approxndcg"), default="approxndcg")
    ap.add_argument("--graded", action="store_true", help="不加则把 label 二值化（S/C 压成 0）")
    ap.add_argument("--model", default=MODEL_NAME)
    ap.add_argument("--epochs", type=int, default=1)  # r1 实测 2ep 无增益，默认 1
    ap.add_argument("--batch", type=int, default=8, help="组数，不是 pair 数")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--max-len", type=int, default=MAX_LEN)
    ap.add_argument(
        "--bce-weight",
        type=float,
        default=0.0,
        help="pointwise BCE 辅助权重（>0 即联合 loss，保住绝对分数校准）",
    )
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    # 权重存 fp32、前向用 bf16 autocast（混合精度）。纯 bf16 权重 + AdamW 没有 fp32 master
    # weights，1e-5 这个量级的更新会被舍入吃掉——ms-swift 内部由 accelerate 兜着，手写循环得自己来。
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=1, torch_dtype=torch.float32
    )
    model.gradient_checkpointing_enable()
    model.cuda().train()

    def make_loader(path: str, shuffle: bool) -> DataLoader:
        return DataLoader(
            GroupDataset(path, args.graded),
            batch_size=args.batch,
            shuffle=shuffle,
            num_workers=4,
            collate_fn=lambda b: collate(b, tok, args.max_len),
        )

    train_loader, val_loader = make_loader(args.train, True), make_loader(args.val, False)
    steps = len(train_loader) * args.epochs
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = get_scheduler("linear", opt, int(steps * 0.05), steps)
    print(
        f"loss={args.loss} bce_w={args.bce_weight} graded={args.graded} "
        f"steps={steps} groups={len(train_loader.dataset)}"
    )

    t0, done = time.time(), 0
    for ep in range(args.epochs):
        for enc in train_loader:
            enc = {k: v.cuda() for k, v in enc.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(
                    input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]
                ).logits.view(-1)
            loss = batch_loss(logits, enc, args.loss, args.bce_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            done += 1
            if done % 200 == 0:
                rate = done / (time.time() - t0)
                eta = (steps - done) / rate / 60
                print(
                    f"  {done}/{steps}  loss {float(loss):.4f}  {rate:.2f} it/s  ETA {eta:.0f}min",
                    flush=True,
                )
        m = evaluate(model, val_loader, args.loss, args.bce_weight)
        print(f"[epoch {ep + 1}] val {m}", flush=True)
        out = Path(args.out) / f"epoch-{ep + 1}"
        out.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(out)
        tok.save_pretrained(out)
        print(f"  已存 {out}", flush=True)

    print(f"完成，用时 {(time.time() - t0) / 60:.1f} 分钟")


if __name__ == "__main__":
    main()
