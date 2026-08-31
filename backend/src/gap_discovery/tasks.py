"""Task registry + node-level JSON checkpoints (table-isolated from research_memory)."""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class TaskStore:
    """Persist research tasks, thread mapping, events, and state snapshots."""

    def __init__(self, db_path: Optional[str] = None) -> None:
        root = Path(__file__).resolve().parents[2]
        default = root / "workspace" / "research_tasks.db"
        self.db_path = Path(db_path or os.getenv("RESEARCH_TASKS_DB", str(default)))
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._lock, self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS research_tasks (
                    task_id TEXT PRIMARY KEY,
                    thread_id TEXT NOT NULL UNIQUE,
                    topic TEXT NOT NULL,
                    status TEXT NOT NULL,
                    current_node TEXT,
                    termination_reason TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    meta_json TEXT
                );
                CREATE TABLE IF NOT EXISTS task_checkpoints (
                    checkpoint_id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL,
                    thread_id TEXT NOT NULL,
                    node_name TEXT,
                    state_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(task_id) REFERENCES research_tasks(task_id)
                );
                CREATE TABLE IF NOT EXISTS task_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL,
                    seq INTEGER NOT NULL,
                    event_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS side_effect_log (
                    idempotency_key TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_task_events_task ON task_events(task_id, seq);
                CREATE INDEX IF NOT EXISTS idx_ckpt_task ON task_checkpoints(task_id, created_at);
                """
            )

    def create_task(self, *, topic: str, task_id: Optional[str] = None, thread_id: Optional[str] = None) -> dict[str, Any]:
        tid = task_id or str(uuid4())
        thr = thread_id or tid
        now = _utcnow()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO research_tasks(task_id, thread_id, topic, status, current_node, created_at, updated_at, meta_json)
                VALUES (?, ?, ?, 'running', '', ?, ?, '{}')
                """,
                (tid, thr, topic, now, now),
            )
        return {"task_id": tid, "thread_id": thr, "topic": topic, "status": "running"}

    def get_task(self, task_id: str) -> Optional[dict[str, Any]]:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM research_tasks WHERE task_id = ?", (task_id,)
            ).fetchone()
        if not row:
            return None
        return dict(row)

    def update_task(
        self,
        task_id: str,
        *,
        status: Optional[str] = None,
        current_node: Optional[str] = None,
        termination_reason: Optional[str] = None,
        meta: Optional[dict[str, Any]] = None,
    ) -> None:
        task = self.get_task(task_id)
        if not task:
            raise KeyError(task_id)
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                UPDATE research_tasks
                SET status = ?, current_node = ?, termination_reason = ?, updated_at = ?, meta_json = ?
                WHERE task_id = ?
                """,
                (
                    status if status is not None else task["status"],
                    current_node if current_node is not None else task["current_node"],
                    termination_reason
                    if termination_reason is not None
                    else task["termination_reason"],
                    _utcnow(),
                    json.dumps(meta if meta is not None else json.loads(task["meta_json"] or "{}"), ensure_ascii=False),
                    task_id,
                ),
            )

    def save_checkpoint(
        self,
        *,
        task_id: str,
        thread_id: str,
        node_name: str,
        state: dict[str, Any],
    ) -> str:
        ckpt_id = str(uuid4())
        payload = json.dumps(state, ensure_ascii=False, default=str)
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO task_checkpoints(checkpoint_id, task_id, thread_id, node_name, state_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (ckpt_id, task_id, thread_id, node_name, payload, _utcnow()),
            )
            conn.execute(
                "UPDATE research_tasks SET current_node = ?, updated_at = ? WHERE task_id = ?",
                (node_name, _utcnow(), task_id),
            )
        return ckpt_id

    def latest_checkpoint(self, task_id: str) -> Optional[dict[str, Any]]:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM task_checkpoints
                WHERE task_id = ?
                ORDER BY created_at DESC LIMIT 1
                """,
                (task_id,),
            ).fetchone()
        if not row:
            return None
        data = dict(row)
        data["state"] = json.loads(data.pop("state_json"))
        return data

    def append_event(self, task_id: str, event: dict[str, Any]) -> int:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT COALESCE(MAX(seq), 0) AS m FROM task_events WHERE task_id = ?",
                (task_id,),
            ).fetchone()
            seq = int(row["m"]) + 1
            conn.execute(
                """
                INSERT INTO task_events(task_id, seq, event_json, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (task_id, seq, json.dumps(event, ensure_ascii=False, default=str), _utcnow()),
            )
        return seq

    def list_events(self, task_id: str, *, after_seq: int = 0, limit: int = 500) -> list[dict[str, Any]]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT seq, event_json FROM task_events
                WHERE task_id = ? AND seq > ?
                ORDER BY seq ASC LIMIT ?
                """,
                (task_id, after_seq, limit),
            ).fetchall()
        out = []
        for row in rows:
            ev = json.loads(row["event_json"])
            ev["_seq"] = row["seq"]
            out.append(ev)
        return out

    def claim_side_effect(self, task_id: str, key: str, kind: str) -> bool:
        """Return True if this is the first time the side effect runs (idempotent)."""

        with self._lock, self._connect() as conn:
            try:
                conn.execute(
                    """
                    INSERT INTO side_effect_log(idempotency_key, task_id, kind, created_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (f"{task_id}:{key}", task_id, kind, _utcnow()),
                )
                return True
            except sqlite3.IntegrityError:
                return False


_STORE: TaskStore | None = None


def get_task_store() -> TaskStore:
    global _STORE
    if _STORE is None:
        _STORE = TaskStore()
    return _STORE
