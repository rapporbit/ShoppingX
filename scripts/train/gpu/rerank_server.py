"""在 GPU 机器上起标准 rerank API，暴露微调后的 reranker。姊妹脚本：``embed_server.py``。

**为什么走标准协议而不是自定义**：``app/recall/reranker.py`` 配了 ``RERANKER_MODEL`` 就按
Cohere/Jina 同构的 ``POST /v1/rerank`` 发请求（``{model, query, documents, top_n,
return_documents}`` → ``{"results": [{"index", "relevance_score"}]}``）。照它实现，线上切换
自训模型就只是改两行 ``.env``——不用动一行应用代码，A/B 也才干净。

**打分口径必须与训练/评测三处一致**：sigmoid(logit)，与 ``rerank_candidates.py``、
``score_negatives.py`` 同源。相对排序其实不受单调变换影响，但 ``item_picker`` 的相对门会拿
分数做比较与缓存，口径漂了会静默改变门的行为。

本地怎么连（不暴露公网）::

    ssh -N -L 8091:localhost:8091 huzhouet
    # .env: RERANKER_ENDPOINT=http://127.0.0.1:8091/v1/rerank
    #       RERANKER_MODEL=globex-reranker-r1

用法（GPU 机器）::

    CUDA_VISIBLE_DEVICES=6 python rerank_server.py --model output/r1/xxx/checkpoint-15670
"""

from __future__ import annotations

import argparse

import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MAX_LEN = 320  # 与 rerank_candidates.py 一致；实测 p99 才约 100 token，够用有余
BATCH = 64  # refdocs §10.6：每批 64-128 打包，单条推理 GPU 利用率 < 20%

app = FastAPI()
STATE: dict = {}


class RerankRequest(BaseModel):
    query: str
    documents: list[str]
    model: str | None = None
    top_n: int | None = None
    return_documents: bool = False


@app.get("/health")
def health() -> dict:
    return {"ok": True, "model": STATE.get("name")}


@app.post("/v1/rerank")
@torch.inference_mode()
def rerank(req: RerankRequest) -> dict:
    tok, model = STATE["tok"], STATE["model"]
    scores: list[float] = []
    for i in range(0, len(req.documents), BATCH):
        chunk = req.documents[i : i + BATCH]
        enc = tok(
            [req.query] * len(chunk),
            chunk,
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        ).to("cuda")
        logits = model(**enc).logits.view(-1).float()
        scores.extend(torch.sigmoid(logits).cpu().tolist())

    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    if req.top_n:
        ranked = ranked[: req.top_n]
    results = [{"index": i, "relevance_score": scores[i]} for i in ranked]
    if req.return_documents:
        for r in results:
            r["document"] = req.documents[int(r["index"])]
    return {"model": STATE.get("name"), "results": results}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", default="globex-reranker")
    ap.add_argument("--port", type=int, default=8091)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, torch_dtype=torch.float16
    )
    model.eval().cuda()
    STATE.update(tok=tok, model=model, name=args.name)

    # refdocs §10.6：服务起来先预热，否则第一批真实请求会吃满冷启动延迟
    warm = ["warmup document about a backpack"] * 8
    rerank(RerankRequest(query="warmup", documents=warm))
    print(f"预热完成，模型 {args.model}，监听 :{args.port}", flush=True)

    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
