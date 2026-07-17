"""向量编码客户端（query / 商品文本共用一个现成 embedding）。

源自 refdocs 04-0 的「三塔 + 双通道」设计，落地后已收敛为**单编码器**：

- 本项目**不自训模型**（无 GPU），query 与商品文本共享同一个现成预训练
  embedding（BGE-M3 等，走 OpenAI 兼容 ``/embeddings`` 在线推理）。
- endpoint 未就绪时退化到**确定性本地编码**（字符 n-gram 哈希），让建索引与检索
  在离线 / CI 下也能跑通且结果可复现，从而和真实链路解耦（refdocs M3 条目要求的 stub）。

**个性化通路（user 塔 + fuse 加权融合）已删**：实践中个性化改走「偏好词并入检索词」
（见 item_search）+ Qdrant payload filter，向量级融合通路长期零调用，按减法原则移除。
"""

from __future__ import annotations

import asyncio
import hashlib
import os
from functools import lru_cache

import httpx
import numpy as np

# 远程编码瞬时错误（429/5xx/连接抖动）的重试次数。
_REMOTE_RETRIES = 3

# 本地回退编码的默认维度。真实 embedding 模型维度由其自身决定（见 _remote_dim）。
DEFAULT_LOCAL_DIM = 256
# 字符 n-gram 的 n（本地哈希编码用），3-gram 对中英混排都还稳。
_NGRAM = 3


def _env_int(key: str, default: int) -> int:
    """读取整型环境变量，缺失或非法（如 EMBED_DIM=auto 笔误）时回退默认值，不崩。"""
    raw = os.environ.get(key)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    """对单条或一批向量做 L2 归一化（归一化后内积 == 余弦相似度）。"""
    norm = np.linalg.norm(vec, axis=-1, keepdims=True)
    # 避免零向量除零：范数为 0 时原样返回。
    return np.where(norm == 0, vec, vec / norm)


class TowerClient:
    """编码器的统一壳子：对外暴露 query 单条与批量两个编码入口。

    构造参数全部可选，缺省从 env 读：

    - ``EMBED_MODEL``：填了就走远程 OpenAI 兼容 embedding；不填走本地确定性回退。
    - ``EMBED_BASE_URL`` / ``EMBED_API_KEY``：缺省回退到 ``OPENAI_BASE_URL`` / ``OPENAI_API_KEY``。
    - ``EMBED_DIM``：仅本地回退用，默认 256。
    """

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        local_dim: int | None = None,
    ) -> None:
        self._model = model if model is not None else os.environ.get("EMBED_MODEL")
        self._base_url = (
            base_url or os.environ.get("EMBED_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
        )
        self._api_key = (
            api_key or os.environ.get("EMBED_API_KEY") or os.environ.get("OPENAI_API_KEY")
        )
        self._local_dim = local_dim or _env_int("EMBED_DIM", DEFAULT_LOCAL_DIM)
        self._client: httpx.AsyncClient | None = None
        # 远程模型维度首次编码后探明并缓存（不同模型维度不同）。
        self._remote_dim: int | None = None

    @property
    def remote(self) -> bool:
        """是否走远程真实 embedding（否则为本地确定性回退）。"""
        return bool(self._model)

    @property
    def dim(self) -> int:
        """当前编码维度。远程模型若尚未探明则返回 -1（首次编码后即确定）。"""
        if self.remote:
            return self._remote_dim if self._remote_dim is not None else -1
        return self._local_dim

    # ---------------------- 编码入口 ----------------------
    async def encode_query(self, text: str) -> np.ndarray:
        """编码单条文本（用户这次的搜索意图 / 意图短语）。"""
        return (await self.encode_texts([text]))[0]

    async def encode_texts(self, texts: list[str]) -> np.ndarray:
        """批量编码，返回形状 ``(len(texts), dim)`` 的 L2 归一化向量矩阵。

        建索引时一次喂一批，省去逐条 HTTP 往返。
        """
        if not texts:
            return np.zeros((0, max(self.dim, 1)), dtype=np.float32)
        if self.remote:
            mat = await self._encode_remote(texts)
        else:
            mat = self._encode_local(texts)
        return _l2_normalize(mat.astype(np.float32))

    # ---------------------- 后端实现 ----------------------
    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._base_url or "",
                headers={"Authorization": f"Bearer {self._api_key}"},
                timeout=30.0,
            )
        return self._client

    async def _encode_remote(self, texts: list[str]) -> np.ndarray:
        """走 OpenAI 兼容 ``/embeddings`` 接口编码一批文本（瞬时错误指数退避重试）。"""
        last_exc: Exception | None = None
        for attempt in range(_REMOTE_RETRIES):
            try:
                resp = await self._get_client().post(
                    "/embeddings", json={"model": self._model, "input": texts}
                )
                resp.raise_for_status()
                data = resp.json()["data"]
                # 接口不保证按输入顺序返回，按 index 排序后取 embedding。
                ordered = sorted(data, key=lambda d: d["index"])
                mat = np.asarray([d["embedding"] for d in ordered], dtype=np.float32)
                self._remote_dim = mat.shape[1]
                return mat
            except httpx.HTTPError as exc:  # 含连接错误与 raise_for_status 的 4xx/5xx
                last_exc = exc
                if attempt < _REMOTE_RETRIES - 1:
                    await asyncio.sleep(2**attempt)  # 1s → 2s → 4s
        raise last_exc  # type: ignore[misc]

    def _encode_local(self, texts: list[str]) -> np.ndarray:
        """确定性本地回退编码：字符 n-gram 哈希进固定维度的词袋向量。

        不是「好」的语义编码，但**确定性**（同输入恒同输出）且零依赖零网络，
        足以让建索引→检索全链路在离线 / CI 下跑通、被测试断言。
        """
        dim = self._local_dim
        mat = np.zeros((len(texts), dim), dtype=np.float32)
        for row, text in enumerate(texts):
            norm = text.lower().strip()
            if not norm:
                continue
            for i in range(max(len(norm) - _NGRAM + 1, 1)):
                gram = norm[i : i + _NGRAM]
                # 用稳定哈希（md5）落桶，避免 Python hash 随机化导致跨进程不一致。
                bucket = int(hashlib.md5(gram.encode("utf-8")).hexdigest(), 16) % dim
                mat[row, bucket] += 1.0
        return mat

    async def aclose(self) -> None:
        """释放底层 HTTP 连接（远程模式下用过才有）。"""
        if self._client is not None:
            await self._client.aclose()
            self._client = None


@lru_cache(maxsize=1)
def get_tower_client() -> TowerClient:
    """进程内共享的三塔客户端（主 / 子 Agent 共用，复用连接池）。"""
    return TowerClient()
