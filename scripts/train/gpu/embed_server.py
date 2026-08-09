"""在 GPU 机器上起一个 OpenAI 兼容的 /v1/embeddings 服务，暴露微调后的模型。

**为什么必须有它**：换模型做端到端对照时，query 和商品必须由**同一个模型**编码，否则两者不在
一个向量空间里——召回不会报错，只会静默变成垃圾。商品侧我们离线编码好了，query 侧是线上运行
时才编码的，所以微调模型必须能被 `app/recall/towers.py` 以 API 方式调用。

本地怎么连：SSH 端口转发即可，不用暴露公网::

    ssh -N -L 8090:localhost:8090 huzhouet          # 本地另开一个终端
    EMBED_BASE_URL=http://localhost:8090/v1 uv run python scripts/eval/run_rubric.py

编码口径与 `eval_on_gpu.py` / `encode_corpus_gpu.py` 严格一致：CLS 池化 + L2 归一化。三处必须
同步改，任何一处不一致都会造成"离线评测好好的、线上召回崩了"这种最难查的事故。

用法（GPU 机器）::

    CUDA_VISIBLE_DEVICES=5 python embed_server.py --model output/e10_fullexp/xxx/checkpoint-9284
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModel, AutoTokenizer

MAX_LEN = 512

app = FastAPI()
STATE: dict = {}


class EmbedRequest(BaseModel):
    input: str | list[str]
    model: str | None = None


@app.get("/health")
def health() -> dict:
    return {"ok": True, "model": STATE.get("name")}


@app.post("/v1/embeddings")
@torch.inference_mode()
def embeddings(req: EmbedRequest) -> dict:
    texts = [req.input] if isinstance(req.input, str) else list(req.input)
    tok, model = STATE["tok"], STATE["model"]
    out = []
    for i in range(0, len(texts), 64):
        enc = tok(
            texts[i : i + 64],
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        ).to("cuda")
        vec = model(**enc).last_hidden_state[:, 0]
        out.extend(F.normalize(vec, dim=-1).float().cpu().tolist())
    return {
        "object": "list",
        "model": STATE.get("name"),
        "data": [{"object": "embedding", "index": i, "embedding": v} for i, v in enumerate(out)],
        "usage": {"prompt_tokens": 0, "total_tokens": 0},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--port", type=int, default=8090)
    args = ap.parse_args()

    STATE["name"] = args.model
    STATE["tok"] = AutoTokenizer.from_pretrained(args.model)
    STATE["model"] = AutoModel.from_pretrained(args.model, torch_dtype=torch.float16).eval().cuda()
    print(f"模型已加载：{args.model}，监听 :{args.port}", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
