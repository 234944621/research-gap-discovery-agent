"""Embedding client for Chroma RAG (DashScope OpenAI-compatible)."""

from __future__ import annotations

import logging
import os
from typing import Sequence

logger = logging.getLogger(__name__)


def embed_texts(texts: Sequence[str]) -> list[list[float]]:
    """Embed texts via DashScope/OpenAI-compatible embeddings API."""

    from openai import OpenAI

    api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
    base_url = (
        os.getenv("EMBEDDING_BASE_URL")
        or os.getenv("LLM_BASE_URL")
        or "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    model = os.getenv("EMBEDDING_MODEL", "text-embedding-v3")
    client = OpenAI(api_key=api_key, base_url=base_url)

    cleaned = [t if (t or "").strip() else " " for t in texts]
    # DashScope batch limit is modest; chunk requests
    out: list[list[float]] = []
    batch_size = int(os.getenv("EMBEDDING_BATCH_SIZE", "8"))
    for i in range(0, len(cleaned), batch_size):
        batch = cleaned[i : i + batch_size]
        resp = client.embeddings.create(model=model, input=batch)
        # API may not preserve order in all providers; sort by index if present
        data = sorted(resp.data, key=lambda d: getattr(d, "index", 0))
        out.extend([list(d.embedding) for d in data])
    return out


class DashScopeEmbeddingFunction:
    """chromadb EmbeddingFunction compatible wrapper."""

    def __init__(self) -> None:
        self._dim: int | None = None

    def name(self) -> str:
        return os.getenv("EMBEDDING_MODEL", "text-embedding-v3")

    def __call__(self, input: list[str]) -> list[list[float]]:
        vectors = embed_texts(input)
        if vectors and self._dim is None:
            self._dim = len(vectors[0])
        return vectors
