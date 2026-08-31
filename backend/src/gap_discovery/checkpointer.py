"""LangGraph SQLite checkpointer (isolated DB from research_memory / research_tasks)."""

from __future__ import annotations

import os
import sqlite3
import threading
from pathlib import Path
from typing import Any, Optional

_LOCK = threading.Lock()
_CHECKPOINTER = None
_CONN: Optional[sqlite3.Connection] = None


def checkpoint_db_path() -> Path:
    root = Path(__file__).resolve().parents[2]
    default = root / "workspace" / "langgraph_checkpoints.db"
    return Path(os.getenv("LANGGRAPH_CHECKPOINT_DB", str(default)))


def get_sqlite_checkpointer():
    """Return a process-wide SqliteSaver (connection kept open)."""

    global _CHECKPOINTER, _CONN
    with _LOCK:
        if _CHECKPOINTER is not None:
            return _CHECKPOINTER
        path = checkpoint_db_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        _CONN = sqlite3.connect(str(path), check_same_thread=False)
        from langgraph.checkpoint.sqlite import SqliteSaver

        _CHECKPOINTER = SqliteSaver(_CONN)
        return _CHECKPOINTER


def invoke_config(thread_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": thread_id}}
