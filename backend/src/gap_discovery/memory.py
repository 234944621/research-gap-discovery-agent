"""SQLite Research Memory — topic-scoped long-term store.

Layers (interview-aligned, scenario-sized):
- Episodic: gaps / queries / papers / artifacts (concrete run experiences)
- Semantic: semantic_lessons (distilled REJECTED rules, consolidated)
- Procedural: procedures (reusable verify / search SOPs per topic)
- Entity: entities (structured facts: counts, keep directions, query seeds)

Working / short-term memory lives in ResearchState + verify `messages`, not here.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
import time
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


def normalize_topic(topic: str) -> str:
    """Exact topic key for storage/recall (trim + collapse space + lower)."""

    return re.sub(r"\s+", " ", (topic or "").strip().lower())


def _token_set(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]{2,}", (text or "").lower()))


def texts_similar(a: str, b: str, *, threshold: float = 0.5) -> bool:
    """Lexical overlap similarity used for gap de-duplication."""

    ta, tb = _token_set(a), _token_set(b)
    if not ta or not tb:
        return False
    inter = len(ta & tb)
    return inter / max(len(ta), 1) > threshold or inter / max(len(tb), 1) > threshold


def _cosine(u: list[float], v: list[float]) -> float:
    if not u or not v or len(u) != len(v):
        return 0.0
    dot = sum(a * b for a, b in zip(u, v))
    nu = sum(a * a for a in u) ** 0.5
    nv = sum(b * b for b in v) ** 0.5
    if nu == 0 or nv == 0:
        return 0.0
    return dot / (nu * nv)


class ResearchMemoryStore:
    """Persist research artifacts so later runs can avoid repeating rejected gaps."""

    def __init__(self, db_path: Optional[str] = None) -> None:
        root = Path(__file__).resolve().parents[2]
        default = root / "workspace" / "research_memory.db"
        self.db_path = Path(db_path or os.getenv("RESEARCH_MEMORY_DB", str(default)))
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.max_queries_per_topic = int(os.getenv("MEMORY_MAX_QUERIES_PER_TOPIC", "40"))
        self.max_lessons_per_topic = int(os.getenv("MEMORY_MAX_LESSONS_PER_TOPIC", "30"))
        self.gap_sim_threshold = float(os.getenv("MEMORY_GAP_SIM_THRESHOLD", "0.5"))
        self.embed_sim_threshold = float(os.getenv("MEMORY_GAP_EMBED_SIM", "0.82"))
        self.lesson_sim_threshold = float(os.getenv("MEMORY_LESSON_SIM_THRESHOLD", "0.45"))
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS papers (
                    paper_id TEXT PRIMARY KEY,
                    topic TEXT,
                    title TEXT,
                    year INTEGER,
                    payload TEXT,
                    updated_at REAL
                );
                CREATE TABLE IF NOT EXISTS gaps (
                    gap_id TEXT PRIMARY KEY,
                    topic TEXT,
                    description TEXT,
                    status TEXT,
                    reason TEXT,
                    payload TEXT,
                    updated_at REAL
                );
                CREATE TABLE IF NOT EXISTS queries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    topic TEXT,
                    query TEXT,
                    purpose TEXT,
                    created_at REAL
                );
                CREATE TABLE IF NOT EXISTS artifacts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    topic TEXT,
                    kind TEXT,
                    key TEXT,
                    payload TEXT,
                    updated_at REAL
                );
                CREATE TABLE IF NOT EXISTS semantic_lessons (
                    lesson_id TEXT PRIMARY KEY,
                    topic TEXT NOT NULL,
                    rule_text TEXT NOT NULL,
                    source_gap_ids TEXT,
                    evidence_count INTEGER DEFAULT 1,
                    updated_at REAL
                );
                CREATE TABLE IF NOT EXISTS entities (
                    entity_key TEXT PRIMARY KEY,
                    topic TEXT NOT NULL,
                    entity_type TEXT NOT NULL,
                    name TEXT NOT NULL,
                    value_json TEXT,
                    updated_at REAL
                );
                CREATE TABLE IF NOT EXISTS procedures (
                    procedure_id TEXT PRIMARY KEY,
                    topic TEXT NOT NULL,
                    name TEXT NOT NULL,
                    steps_json TEXT NOT NULL,
                    updated_at REAL
                );
                CREATE INDEX IF NOT EXISTS idx_gaps_topic ON gaps(topic);
                CREATE INDEX IF NOT EXISTS idx_papers_topic ON papers(topic);
                CREATE INDEX IF NOT EXISTS idx_artifacts_topic ON artifacts(topic, kind);
                CREATE INDEX IF NOT EXISTS idx_queries_topic ON queries(topic);
                CREATE INDEX IF NOT EXISTS idx_lessons_topic ON semantic_lessons(topic);
                CREATE INDEX IF NOT EXISTS idx_entities_topic ON entities(topic, entity_type);
                CREATE INDEX IF NOT EXISTS idx_procedures_topic ON procedures(topic);
                """
            )

    def _topic_clause(self, topic: str) -> tuple[str, tuple[Any, ...]]:
        """Prefer exact normalized topic match (no broad LIKE)."""

        key = normalize_topic(topic)
        raw = (topic or "").strip()
        # Match normalized equality; also accept exact original string for legacy rows
        sql = "(lower(trim(topic)) = ? OR topic = ?)"
        return sql, (key, raw)

    def recall_bundle(self, topic: str, *, limit: int = 20) -> dict[str, Any]:
        """Structured recall for UI / planner / discover.

        Returns:
          {
            topic_key,
            rejected_gaps: [{description, reason, gap_id, status}, ...],
            other_gaps: [...],
            recent_queries: [...],
            flat: [...]  # backward-compatible flat list for RAG / tools
          }
        """

        topic_sql, topic_params = self._topic_clause(topic)
        rejected: list[dict[str, Any]] = []
        others: list[dict[str, Any]] = []
        queries: list[dict[str, Any]] = []
        flat: list[dict[str, Any]] = []

        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT gap_id, description, status, reason, payload, updated_at
                FROM gaps
                WHERE {topic_sql}
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (*topic_params, limit * 2),
            ).fetchall()
            for row in rows:
                item = {
                    "type": "gap",
                    "gap_id": row["gap_id"],
                    "description": row["description"],
                    "status": row["status"],
                    "reason": row["reason"] or "",
                    "payload": json.loads(row["payload"] or "{}"),
                    "updated_at": row["updated_at"],
                }
                flat.append(item)
                slim = {
                    "gap_id": item["gap_id"],
                    "description": item["description"],
                    "status": item["status"],
                    "reason": item["reason"],
                }
                if (row["status"] or "").upper() == "REJECTED":
                    rejected.append(slim)
                else:
                    others.append(slim)

            qrows = conn.execute(
                f"""
                SELECT query, purpose, created_at FROM queries
                WHERE {topic_sql}
                ORDER BY created_at DESC LIMIT ?
                """,
                (*topic_params, limit),
            ).fetchall()
            for row in qrows:
                qitem = {
                    "type": "query",
                    "query": row["query"],
                    "purpose": row["purpose"],
                    "created_at": row["created_at"],
                }
                queries.append(qitem)
                flat.append(qitem)

        # Put REJECTED first in flat list so tools/RAG see them early
        flat_sorted = [
            x for x in flat if x.get("type") == "gap" and x.get("status") == "REJECTED"
        ] + [
            x for x in flat if not (x.get("type") == "gap" and x.get("status") == "REJECTED")
        ]

        lessons = self.recall_semantic_lessons(topic, limit=min(limit, 12))
        entities = self.recall_entities(topic, limit=min(limit, 20))
        procedures = self.recall_procedures(topic, limit=6)
        for lesson in lessons:
            flat_sorted.append(
                {
                    "type": "semantic_lesson",
                    "lesson_id": lesson.get("lesson_id"),
                    "rule_text": lesson.get("rule_text"),
                    "evidence_count": lesson.get("evidence_count"),
                }
            )
        for ent in entities[:8]:
            flat_sorted.append(
                {
                    "type": "entity",
                    "entity_type": ent.get("entity_type"),
                    "name": ent.get("name"),
                    "value": ent.get("value"),
                }
            )
        for proc in procedures:
            flat_sorted.append(
                {
                    "type": "procedure",
                    "name": proc.get("name"),
                    "steps": proc.get("steps"),
                }
            )

        return {
            "topic_key": normalize_topic(topic),
            "rejected_gaps": rejected[:limit],
            "other_gaps": others[:limit],
            "recent_queries": queries[:limit],
            "semantic_lessons": lessons,
            "entities": entities,
            "procedures": procedures,
            "flat": flat_sorted[: limit * 3],
            "summary": {
                "rejected_count": len(rejected),
                "other_gap_count": len(others),
                "query_count": len(queries),
                "lesson_count": len(lessons),
                "entity_count": len(entities),
                "procedure_count": len(procedures),
            },
        }

    def recall(self, topic: str, *, limit: int = 20) -> list[dict[str, Any]]:
        """Flat list recall (compat). Prefer recall_bundle for structured use."""

        return self.recall_bundle(topic, limit=limit)["flat"]

    def rejected_gaps_with_reasons(self, topic: str) -> list[dict[str, Any]]:
        return self.recall_bundle(topic, limit=50)["rejected_gaps"]

    def rejected_gap_descriptions(self, topic: str) -> list[str]:
        return [g["description"] for g in self.rejected_gaps_with_reasons(topic) if g.get("description")]

    def find_similar_rejected(
        self, topic: str, description: str
    ) -> Optional[dict[str, Any]]:
        """Return a REJECTED gap similar to description, if any."""

        if not (description or "").strip():
            return None
        rejected = self.rejected_gaps_with_reasons(topic)
        # 1) lexical
        for g in rejected:
            if texts_similar(
                description, g.get("description") or "", threshold=self.gap_sim_threshold
            ):
                return {**g, "match": "lexical"}

        # 2) optional embedding (fail open)
        if os.getenv("MEMORY_USE_EMBED_DEDUP", "true").lower() not in {"1", "true", "yes"}:
            return None
        if len(rejected) == 0:
            return None
        try:
            from gap_discovery.embeddings import embed_texts

            texts = [description] + [g.get("description") or "" for g in rejected]
            vectors = embed_texts(texts)
            if len(vectors) != len(texts):
                return None
            probe = vectors[0]
            best_i, best_s = -1, 0.0
            for i, vec in enumerate(vectors[1:], start=0):
                score = _cosine(probe, vec)
                if score > best_s:
                    best_s, best_i = score, i
            if best_i >= 0 and best_s >= self.embed_sim_threshold:
                return {**rejected[best_i], "match": "embedding", "score": round(best_s, 4)}
        except Exception as exc:  # noqa: BLE001
            logger.debug("embed dedup skipped: %s", exc)
        return None

    def is_duplicate_rejected(self, topic: str, description: str) -> bool:
        return self.find_similar_rejected(topic, description) is not None

    def save_paper(self, topic: str, paper_card: dict[str, Any]) -> None:
        paper_id = str(paper_card.get("paper_id") or paper_card.get("title"))
        topic_store = (topic or "").strip()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO papers(paper_id, topic, title, year, payload, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    paper_id,
                    topic_store,
                    paper_card.get("title"),
                    paper_card.get("year"),
                    json.dumps(paper_card, ensure_ascii=False),
                    time.time(),
                ),
            )

    def save_gap(
        self,
        topic: str,
        *,
        gap_id: str,
        description: str,
        status: str,
        reason: str = "",
        payload: Optional[dict[str, Any]] = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO gaps(gap_id, topic, description, status, reason, payload, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    gap_id,
                    (topic or "").strip(),
                    description,
                    status,
                    reason,
                    json.dumps(payload or {}, ensure_ascii=False),
                    time.time(),
                ),
            )
        # Episodic write → light consolidate into semantic / entity / procedural
        try:
            self.consolidate_after_episode(
                topic,
                gap_id=gap_id,
                description=description,
                status=status,
                reason=reason,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("memory consolidate skipped: %s", exc)

    def save_query(self, topic: str, query: str, purpose: str) -> None:
        topic_store = (topic or "").strip()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO queries(topic, query, purpose, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (topic_store, query, purpose, time.time()),
            )
            self._prune_queries(conn, topic_store)

    def _prune_queries(self, conn: sqlite3.Connection, topic: str) -> None:
        """Keep only the newest N queries for this topic (exact + normalized)."""

        topic_sql, topic_params = self._topic_clause(topic)
        rows = conn.execute(
            f"""
            SELECT id FROM queries
            WHERE {topic_sql}
            ORDER BY created_at DESC
            """,
            topic_params,
        ).fetchall()
        if len(rows) <= self.max_queries_per_topic:
            return
        drop_ids = [r["id"] for r in rows[self.max_queries_per_topic :]]
        conn.executemany("DELETE FROM queries WHERE id = ?", [(i,) for i in drop_ids])

    def save_json_artifact(
        self, topic: str, *, kind: str, key: str, payload: dict[str, Any]
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO artifacts(topic, kind, key, payload, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    (topic or "").strip(),
                    kind,
                    key,
                    json.dumps(payload, ensure_ascii=False),
                    time.time(),
                ),
            )

    def recall_artifacts(
        self, topic: str, *, kind: Optional[str] = None, limit: int = 20
    ) -> list[dict[str, Any]]:
        topic_sql, topic_params = self._topic_clause(topic)
        with self._connect() as conn:
            if kind:
                rows = conn.execute(
                    f"""
                    SELECT kind, key, payload FROM artifacts
                    WHERE {topic_sql} AND kind = ?
                    ORDER BY updated_at DESC LIMIT ?
                    """,
                    (*topic_params, kind, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    f"""
                    SELECT kind, key, payload FROM artifacts
                    WHERE {topic_sql}
                    ORDER BY updated_at DESC LIMIT ?
                    """,
                    (*topic_params, limit),
                ).fetchall()
        out = []
        for row in rows:
            out.append(
                {
                    "type": row["kind"],
                    "key": row["key"],
                    "payload": json.loads(row["payload"] or "{}"),
                }
            )
        return out

    # ----- Semantic / Entity / Procedural layers -----

    def recall_semantic_lessons(self, topic: str, *, limit: int = 12) -> list[dict[str, Any]]:
        topic_sql, topic_params = self._topic_clause(topic)
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT lesson_id, rule_text, source_gap_ids, evidence_count, updated_at
                FROM semantic_lessons
                WHERE {topic_sql}
                ORDER BY evidence_count DESC, updated_at DESC
                LIMIT ?
                """,
                (*topic_params, limit),
            ).fetchall()
        out = []
        for row in rows:
            out.append(
                {
                    "lesson_id": row["lesson_id"],
                    "rule_text": row["rule_text"],
                    "source_gap_ids": json.loads(row["source_gap_ids"] or "[]"),
                    "evidence_count": row["evidence_count"] or 1,
                    "updated_at": row["updated_at"],
                }
            )
        return out

    def recall_entities(
        self, topic: str, *, entity_type: Optional[str] = None, limit: int = 20
    ) -> list[dict[str, Any]]:
        topic_sql, topic_params = self._topic_clause(topic)
        with self._connect() as conn:
            if entity_type:
                rows = conn.execute(
                    f"""
                    SELECT entity_key, entity_type, name, value_json, updated_at
                    FROM entities
                    WHERE {topic_sql} AND entity_type = ?
                    ORDER BY updated_at DESC LIMIT ?
                    """,
                    (*topic_params, entity_type, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    f"""
                    SELECT entity_key, entity_type, name, value_json, updated_at
                    FROM entities
                    WHERE {topic_sql}
                    ORDER BY updated_at DESC LIMIT ?
                    """,
                    (*topic_params, limit),
                ).fetchall()
        out = []
        for row in rows:
            out.append(
                {
                    "entity_key": row["entity_key"],
                    "entity_type": row["entity_type"],
                    "name": row["name"],
                    "value": json.loads(row["value_json"] or "{}"),
                    "updated_at": row["updated_at"],
                }
            )
        return out

    def recall_procedures(self, topic: str, *, limit: int = 6) -> list[dict[str, Any]]:
        topic_sql, topic_params = self._topic_clause(topic)
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT procedure_id, name, steps_json, updated_at
                FROM procedures
                WHERE {topic_sql}
                ORDER BY updated_at DESC LIMIT ?
                """,
                (*topic_params, limit),
            ).fetchall()
        out = []
        for row in rows:
            out.append(
                {
                    "procedure_id": row["procedure_id"],
                    "name": row["name"],
                    "steps": json.loads(row["steps_json"] or "[]"),
                    "updated_at": row["updated_at"],
                }
            )
        return out

    def upsert_semantic_lesson(
        self,
        topic: str,
        *,
        rule_text: str,
        source_gap_id: str = "",
        lesson_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Insert or merge a semantic lesson (dedupe by lexical similarity)."""

        rule = (rule_text or "").strip()
        if not rule:
            return {}
        topic_store = (topic or "").strip()
        existing = self.recall_semantic_lessons(topic_store, limit=self.max_lessons_per_topic)
        for lesson in existing:
            if texts_similar(rule, lesson.get("rule_text") or "", threshold=self.lesson_sim_threshold):
                sources = list(lesson.get("source_gap_ids") or [])
                added = False
                if source_gap_id and source_gap_id not in sources:
                    sources.append(source_gap_id)
                    added = True
                merged = lesson["rule_text"]
                # Prefer longer rule when merging near-duplicates
                if len(rule) > len(merged) + 20:
                    merged = rule
                new_count = int(lesson.get("evidence_count") or 1)
                if added:
                    new_count = max(new_count + 1, len(sources) or 1)
                with self._connect() as conn:
                    conn.execute(
                        """
                        UPDATE semantic_lessons
                        SET rule_text = ?, source_gap_ids = ?, evidence_count = ?, updated_at = ?
                        WHERE lesson_id = ?
                        """,
                        (
                            merged,
                            json.dumps(sources, ensure_ascii=False),
                            new_count,
                            time.time(),
                            lesson["lesson_id"],
                        ),
                    )
                return {
                    "lesson_id": lesson["lesson_id"],
                    "rule_text": merged,
                    "merged": True,
                    "evidence_count": new_count,
                }

        lid = lesson_id or f"lesson:{normalize_topic(topic_store)[:40]}:{_stable_short(rule)}"
        sources = [source_gap_id] if source_gap_id else []
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO semantic_lessons(
                    lesson_id, topic, rule_text, source_gap_ids, evidence_count, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    lid,
                    topic_store,
                    rule,
                    json.dumps(sources, ensure_ascii=False),
                    max(1, len(sources)),
                    time.time(),
                ),
            )
            self._prune_lessons(conn, topic_store)
        return {"lesson_id": lid, "rule_text": rule, "merged": False, "evidence_count": 1}

    def upsert_entity(
        self,
        topic: str,
        *,
        entity_type: str,
        name: str,
        value: Any,
        entity_key: Optional[str] = None,
    ) -> None:
        topic_store = (topic or "").strip()
        key = entity_key or f"{normalize_topic(topic_store)}:{entity_type}:{_stable_short(name)}"
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO entities(entity_key, topic, entity_type, name, value_json, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    key,
                    topic_store,
                    entity_type,
                    name,
                    json.dumps(value, ensure_ascii=False, default=str),
                    time.time(),
                ),
            )

    def upsert_procedure(
        self,
        topic: str,
        *,
        name: str,
        steps: list[Any],
        procedure_id: Optional[str] = None,
    ) -> None:
        topic_store = (topic or "").strip()
        pid = procedure_id or f"proc:{normalize_topic(topic_store)[:40]}:{_stable_short(name)}"
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO procedures(procedure_id, topic, name, steps_json, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    pid,
                    topic_store,
                    name,
                    json.dumps(steps, ensure_ascii=False),
                    time.time(),
                ),
            )

    def _prune_lessons(self, conn: sqlite3.Connection, topic: str) -> None:
        topic_sql, topic_params = self._topic_clause(topic)
        rows = conn.execute(
            f"""
            SELECT lesson_id FROM semantic_lessons
            WHERE {topic_sql}
            ORDER BY evidence_count DESC, updated_at DESC
            """,
            topic_params,
        ).fetchall()
        if len(rows) <= self.max_lessons_per_topic:
            return
        drop_ids = [r["lesson_id"] for r in rows[self.max_lessons_per_topic :]]
        conn.executemany("DELETE FROM semantic_lessons WHERE lesson_id = ?", [(i,) for i in drop_ids])

    def consolidate_after_episode(
        self,
        topic: str,
        *,
        gap_id: str,
        description: str,
        status: str,
        reason: str = "",
    ) -> dict[str, Any]:
        """Light integrate after one gap write: semantic lesson + entities + verify SOP."""

        status_u = (status or "").upper()
        out: dict[str, Any] = {"lesson": None, "entities": [], "procedure": None}

        if status_u == "REJECTED" and (description or reason):
            rule = (
                f"Do not reopen similar gaps: {str(description)[:180].strip()}. "
                f"Prior reject reason: {str(reason)[:220].strip() or 'already covered / not open'}."
            )
            out["lesson"] = self.upsert_semantic_lesson(
                topic, rule_text=rule, source_gap_id=gap_id
            )
            self.upsert_entity(
                topic,
                entity_type="rejected_pattern",
                name=str(description)[:120] or gap_id,
                value={"gap_id": gap_id, "reason": reason, "status": "REJECTED"},
                entity_key=f"{normalize_topic(topic)}:rejected:{gap_id}",
            )
            out["entities"].append("rejected_pattern")

        if status_u in {"KEEP", "REFINED"}:
            self.upsert_entity(
                topic,
                entity_type="keep_direction",
                name=str(description)[:120] or gap_id,
                value={"gap_id": gap_id, "status": status_u, "reason": reason},
                entity_key=f"{normalize_topic(topic)}:keep:{gap_id}",
            )
            out["entities"].append("keep_direction")

        # Topic stats entity (precise query)
        bundle_counts = self._topic_gap_counts(topic)
        self.upsert_entity(
            topic,
            entity_type="topic_stats",
            name="gap_status_counts",
            value=bundle_counts,
            entity_key=f"{normalize_topic(topic)}:stats:gap_counts",
        )
        out["entities"].append("topic_stats")

        # Procedural: refresh verify SOP with preferred query seeds
        recent_q = self._recent_query_texts(topic, limit=5)
        steps = [
            "Load semantic_lessons + REJECTED reasons for this topic",
            "retrieve_rag on gap description + verification_queries",
            "If RAG empty: search_papers with short English keywords (rewrite ≤2)",
            "Optional: find_citing_papers / get_citation_context on key seeds",
            "Decide KEEP | REFINED | REJECTED; never claim global novelty",
        ]
        if recent_q:
            steps.insert(2, f"Prefer historically useful queries: {recent_q}")
        self.upsert_procedure(
            topic,
            name="gap_verify_sop",
            steps=steps,
            procedure_id=f"proc:{normalize_topic(topic)[:48]}:gap_verify_sop",
        )
        out["procedure"] = "gap_verify_sop"

        # Landscape search SOP from landscape queries
        landscape = self._recent_query_texts(topic, limit=8, purpose_prefix="landscape")
        self.upsert_procedure(
            topic,
            name="landscape_search_sop",
            steps=[
                "sanitize_academic_query(topic)",
                "Primary AcademicSearch (OpenAlex↔S2), seed fallback if empty",
                *( [f"Reuse landscape seeds: {landscape}"] if landscape else [] ),
                "Write queries into episodic memory (purpose=landscape_search)",
            ],
            procedure_id=f"proc:{normalize_topic(topic)[:48]}:landscape_search_sop",
        )
        return out

    def _recent_query_texts(
        self, topic: str, *, limit: int = 5, purpose_prefix: Optional[str] = None
    ) -> list[str]:
        topic_sql, topic_params = self._topic_clause(topic)
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT query, purpose FROM queries
                WHERE {topic_sql}
                ORDER BY created_at DESC LIMIT ?
                """,
                (*topic_params, max(limit * 3, limit)),
            ).fetchall()
        out: list[str] = []
        for row in rows:
            if purpose_prefix and not str(row["purpose"] or "").startswith(purpose_prefix):
                continue
            q = (row["query"] or "").strip()
            if q and q not in out:
                out.append(q)
            if len(out) >= limit:
                break
        return out

    def consolidate_topic(self, topic: str) -> dict[str, Any]:
        """Heavier integrate: re-distill all REJECTED → lessons, refresh entities/SOPs."""

        rejected = self.rejected_gaps_with_reasons(topic)
        merged = 0
        for g in rejected:
            self.consolidate_after_episode(
                topic,
                gap_id=str(g.get("gap_id") or ""),
                description=str(g.get("description") or ""),
                status="REJECTED",
                reason=str(g.get("reason") or ""),
            )
            merged += 1
        # Also refresh SOPs using latest KEEP gaps
        for g in self.recall_bundle(topic, limit=30).get("other_gaps") or []:
            if (g.get("status") or "").upper() in {"KEEP", "REFINED"}:
                self.consolidate_after_episode(
                    topic,
                    gap_id=str(g.get("gap_id") or ""),
                    description=str(g.get("description") or ""),
                    status=str(g.get("status") or "KEEP"),
                    reason=str(g.get("reason") or ""),
                )
        return {
            "topic_key": normalize_topic(topic),
            "rejected_processed": merged,
            "lessons": len(self.recall_semantic_lessons(topic)),
            "entities": len(self.recall_entities(topic)),
            "procedures": len(self.recall_procedures(topic)),
        }

    def _topic_gap_counts(self, topic: str) -> dict[str, int]:
        topic_sql, topic_params = self._topic_clause(topic)
        counts: dict[str, int] = {}
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT status, COUNT(*) AS n FROM gaps
                WHERE {topic_sql}
                GROUP BY status
                """,
                topic_params,
            ).fetchall()
        for row in rows:
            counts[str(row["status"] or "UNKNOWN").upper()] = int(row["n"])
        return counts


def _stable_short(text: str) -> str:
    import hashlib

    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:10]