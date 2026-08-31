"""Paper RAG: Embedding + Chroma (with lexical fallback).

Purpose: Gap verification evidence retrieval — NOT generic chatbot QA.
Pipeline: PaperCard / memory chunk → embedding → Chroma → Top-K → ContextBuilder
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]{2,}", text.lower())


def json_safe(obj: Any) -> str:
    import json

    try:
        return json.dumps(obj, ensure_ascii=False)
    except TypeError:
        return str(obj)


class LexicalIndex:
    """In-memory lexical fallback when Chroma/embedding unavailable."""

    def __init__(self) -> None:
        self.chunks: list[dict[str, Any]] = []

    def clear(self) -> None:
        self.chunks = []

    def add(self, chunk: dict[str, Any]) -> None:
        chunk = dict(chunk)
        chunk["tokens"] = Counter(_tokenize(chunk.get("text") or ""))
        self.chunks.append(chunk)

    def retrieve(self, query: str, *, top_k: int = 5) -> list[dict[str, Any]]:
        q = Counter(_tokenize(query))
        if not q or not self.chunks:
            return []
        scored: list[tuple[float, dict[str, Any]]] = []
        for chunk in self.chunks:
            score = _cosine(q, chunk.get("tokens") or Counter())
            if score > 0:
                scored.append((score, chunk))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [_hit(score, chunk) for score, chunk in scored[:top_k]]


class ChromaIndex:
    """Persistent Chroma collection with DashScope embeddings."""

    def __init__(self, collection_name: str = "paper_cards") -> None:
        import chromadb

        from gap_discovery.embeddings import DashScopeEmbeddingFunction

        root = Path(__file__).resolve().parents[2]
        persist = Path(
            os.getenv("CHROMA_DIR", str(root / "workspace" / "chroma_papers"))
        )
        persist.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(persist))
        self._embedder = DashScopeEmbeddingFunction()
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._pending_ids: list[str] = []
        self._pending_docs: list[str] = []
        self._pending_meta: list[dict[str, Any]] = []

    def clear(self) -> None:
        name = self._collection.name
        try:
            self._client.delete_collection(name)
        except Exception:  # noqa: BLE001
            pass
        self._collection = self._client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )
        self._pending_ids.clear()
        self._pending_docs.clear()
        self._pending_meta.clear()

    def add(self, chunk: dict[str, Any]) -> None:
        chunk_id = str(chunk.get("chunk_id") or _stable_id(chunk.get("text") or ""))
        text = (chunk.get("text") or "").strip()
        if not text:
            return
        meta = {
            "paper_id": str(chunk.get("paper_id") or ""),
            "title": str(chunk.get("title") or "")[:200],
            "year": str(chunk.get("year") or ""),
            "section": str(chunk.get("section") or ""),
        }
        self._pending_ids.append(chunk_id)
        self._pending_docs.append(text[:4000])
        self._pending_meta.append(meta)

    def flush(self) -> None:
        if not self._pending_ids:
            return
        vectors = self._embedder(self._pending_docs)
        # upsert in batches
        batch = 16
        for i in range(0, len(self._pending_ids), batch):
            self._collection.upsert(
                ids=self._pending_ids[i : i + batch],
                documents=self._pending_docs[i : i + batch],
                embeddings=vectors[i : i + batch],
                metadatas=self._pending_meta[i : i + batch],
            )
        self._pending_ids.clear()
        self._pending_docs.clear()
        self._pending_meta.clear()

    def retrieve(self, query: str, *, top_k: int = 5) -> list[dict[str, Any]]:
        self.flush()
        if self._collection.count() == 0:
            return []
        qvec = self._embedder([query or " "])[0]
        result = self._collection.query(
            query_embeddings=[qvec],
            n_results=min(top_k, max(1, self._collection.count())),
            include=["documents", "metadatas", "distances"],
        )
        hits: list[dict[str, Any]] = []
        docs = (result.get("documents") or [[]])[0]
        metas = (result.get("metadatas") or [[]])[0]
        dists = (result.get("distances") or [[]])[0]
        for doc, meta, dist in zip(docs, metas, dists):
            # cosine distance → similarity-ish score
            score = round(1.0 - float(dist), 4) if dist is not None else 0.0
            hits.append(
                {
                    "score": score,
                    "paper_id": (meta or {}).get("paper_id"),
                    "title": (meta or {}).get("title"),
                    "section": (meta or {}).get("section"),
                    "year": (meta or {}).get("year") or None,
                    "text": (doc or "")[:800],
                    "backend": "chroma",
                }
            )
        return hits


class PaperRAG:
    """Vector RAG over PaperCards; falls back to lexical if embedding/chroma fails."""

    def __init__(self) -> None:
        self.backend = os.getenv("RAG_BACKEND", "chroma").strip().lower()
        self._lexical = LexicalIndex()
        self._chroma: ChromaIndex | None = None
        self._active = "lexical"
        if self.backend in {"chroma", "embedding", "vector"}:
            try:
                self._chroma = ChromaIndex(
                    collection_name=os.getenv("CHROMA_COLLECTION", "paper_cards")
                )
                # ephemeral session collection: clear so this run indexes fresh cards
                if os.getenv("CHROMA_CLEAR_ON_INIT", "true").lower() in {
                    "1",
                    "true",
                    "yes",
                }:
                    self._chroma.clear()
                self._active = "chroma"
            except Exception as exc:  # noqa: BLE001
                logger.warning("Chroma RAG init failed, lexical fallback: %s", exc)
                self._chroma = None
                self._active = "lexical"

    @property
    def active_backend(self) -> str:
        return self._active

    def clear(self) -> None:
        self._lexical.clear()
        if self._chroma is not None:
            try:
                self._chroma.clear()
            except Exception as exc:  # noqa: BLE001
                logger.debug("chroma clear: %s", exc)

    def add_paper_card(self, card: dict[str, Any]) -> None:
        paper_id = card.get("paper_id")
        title = card.get("title") or ""
        abstract = card.get("abstract") or card.get("content") or ""
        self._add_chunk(
            {
                "chunk_id": f"{paper_id}:abstract",
                "paper_id": paper_id,
                "title": title,
                "year": card.get("year"),
                "section": "abstract",
                "text": f"{title}\n{abstract}",
            }
        )
        for section, value in [
            ("method", card.get("method")),
            ("problem", card.get("research_problem")),
            ("limitations", _limitations_text(card)),
        ]:
            if value:
                self._add_chunk(
                    {
                        "chunk_id": f"{paper_id}:{section}",
                        "paper_id": paper_id,
                        "title": title,
                        "year": card.get("year"),
                        "section": section,
                        "text": str(value),
                    }
                )

    def add_memory_items(self, items: list[dict[str, Any]]) -> None:
        for idx, item in enumerate(items):
            self._add_chunk(
                {
                    "chunk_id": f"memory:{idx}:{_stable_id(str(item))[:8]}",
                    "paper_id": None,
                    "title": f"memory:{item.get('type')}",
                    "year": None,
                    "section": "memory",
                    "text": json_safe(item),
                }
            )

    def _add_chunk(self, chunk: dict[str, Any]) -> None:
        self._lexical.add(chunk)
        if self._chroma is not None:
            try:
                self._chroma.add(chunk)
            except Exception as exc:  # noqa: BLE001
                logger.warning("chroma add failed: %s", exc)

    def retrieve(self, query: str, *, top_k: int = 5) -> list[dict[str, Any]]:
        if self._chroma is not None and self._active == "chroma":
            try:
                hits = self._chroma.retrieve(query, top_k=top_k)
                if hits:
                    return hits
            except Exception as exc:  # noqa: BLE001
                logger.warning("chroma retrieve failed, lexical fallback: %s", exc)
                self._active = "lexical"
        return self._lexical.retrieve(query, top_k=top_k)

    def add_fulltext_chunks(self, chunks: list[dict[str, Any]]) -> None:
        """Index section chunks produced by FullTextService / pdf_parser."""

        for chunk in chunks or []:
            self._add_chunk(
                {
                    "chunk_id": chunk.get("chunk_id"),
                    "paper_id": chunk.get("paper_id"),
                    "title": chunk.get("title"),
                    "year": chunk.get("year"),
                    "section": chunk.get("section") or "fulltext",
                    "text": chunk.get("text") or "",
                }
            )

    def retrieve_section(
        self, query: str, *, section: str | None = None, top_k: int = 5
    ) -> list[dict[str, Any]]:
        hits = self.retrieve(query, top_k=top_k * 3)
        if section:
            filtered = [h for h in hits if (h.get("section") or "") == section]
            if filtered:
                return filtered[:top_k]
        return hits[:top_k]


class ContextBuilder:
    """gather → select → structure → compress for one LLM call."""

    def __init__(self, rag: PaperRAG) -> None:
        self.rag = rag

    def build_for_gap(
        self,
        *,
        gap: dict[str, Any],
        paper_cards: list[dict[str, Any]],
        memory_items: list[dict[str, Any]],
        search_results: list[dict[str, Any]] | None = None,
        rag_hits: list[dict[str, Any]] | None = None,
        max_chars: int = 6000,
    ) -> str:
        related_cards = [
            c
            for c in paper_cards
            if c.get("paper_id") in set(gap.get("supporting_papers") or [])
        ] or paper_cards[:5]
        hits = rag_hits or self.rag.retrieve(
            f"{gap.get('description')} {' '.join(gap.get('verification_queries') or [])}",
            top_k=5,
        )

        parts = [
            f"## RAG Backend: {self.rag.active_backend}",
            "## Current Gap",
            json_safe(gap),
            "",
            "## Related Paper Cards (selected)",
        ]
        for card in related_cards[:5]:
            parts.append(
                (
                    f"- {card.get('title')} ({card.get('year')}) "
                    f"[fulltext={card.get('fulltext_status')}]\n"
                    f"  problem: {card.get('research_problem')}\n"
                    f"  method: {card.get('method')}\n"
                    f"  self_reported: {card.get('self_reported_limitations')}\n"
                    f"  inferred: {card.get('inferred_weaknesses')}"
                )
            )
        parts.extend(["", "## RAG Hits"])
        for hit in hits:
            parts.append(
                f"- [{hit.get('section')}] {hit.get('title')} (score={hit.get('score')}): "
                f"{(hit.get('text') or '')[:300]}"
            )
        if memory_items:
            parts.extend(["", "## Research Memory (related)"])
            for item in memory_items[:8]:
                parts.append(f"- {json_safe(item)[:300]}")
        if search_results:
            parts.extend(["", "## Fresh Search Results"])
            for paper in search_results[:6]:
                parts.append(
                    f"- [{paper.get('year')}] {paper.get('title')} :: "
                    f"{(paper.get('content') or '')[:220]}"
                )

        text = "\n".join(parts)
        if len(text) > max_chars:
            text = text[: max_chars - 20] + "\n...[compressed]"
        return text


def _limitations_text(card: dict[str, Any]) -> str:
    lims = card.get("self_reported_limitations") or card.get("inferred_weaknesses") or []
    if isinstance(lims, list):
        return "\n".join(str(x) for x in lims if x)
    return str(lims or "")


def _stable_id(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


def _hit(score: float, chunk: dict[str, Any]) -> dict[str, Any]:
    return {
        "score": round(score, 4),
        "paper_id": chunk.get("paper_id"),
        "title": chunk.get("title"),
        "section": chunk.get("section"),
        "year": chunk.get("year"),
        "text": (chunk.get("text") or "")[:800],
        "backend": "lexical",
    }


def _cosine(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    dot = sum(a[t] * b[t] for t in a if t in b)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)
