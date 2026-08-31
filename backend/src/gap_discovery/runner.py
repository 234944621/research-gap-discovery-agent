"""Runner that streams Research Agent events to FastAPI SSE (+ task checkpoints)."""

from __future__ import annotations

import logging
import os
import threading
from typing import Any, Iterator, Optional

from config import Configuration
from gap_discovery.graph import GRAPH_EDGES, GRAPH_NODES, iter_pipeline, langgraph_engine, run_pipeline
from gap_discovery.state import ResearchState, initial_state
from gap_discovery.state_view import diff_state, snapshot_state
from gap_discovery.tasks import get_task_store

logger = logging.getLogger(__name__)


def _structured_event(event_type: str, **payload: Any) -> dict[str, Any]:
    return {"type": event_type, **payload}


class GapDiscoveryRunner:
    """Thin adapter: Configuration + topic → SSE event iterator / background tasks."""

    def __init__(self, config: Configuration | None = None) -> None:
        self.config = config or Configuration.from_env()
        self.max_iterations = int(os.getenv("GAP_MAX_ITERATIONS", "6"))
        self.store = get_task_store()

    def run_stream(
        self,
        topic: str,
        *,
        task_id: Optional[str] = None,
        thread_id: Optional[str] = None,
        resume: bool = False,
    ) -> Iterator[dict[str, Any]]:
        state, meta = self._prepare_state(topic, task_id=task_id, thread_id=thread_id, resume=resume)
        yield from self._iter_events(state, persist=True, meta=meta)

    def _prepare_state(
        self,
        topic: str,
        *,
        task_id: Optional[str] = None,
        thread_id: Optional[str] = None,
        resume: bool = False,
    ) -> tuple[ResearchState, dict[str, Any]]:
        if resume and task_id:
            task = self.store.get_task(task_id)
            if not task:
                raise KeyError(task_id)
            if task["status"] == "completed":
                raise RuntimeError("TASK_ALREADY_COMPLETED")
            ckpt = self.store.latest_checkpoint(task_id)
            if not ckpt:
                raise RuntimeError("NO_CHECKPOINT")
            state = ckpt["state"]  # type: ignore[assignment]
            state["status"] = "running"
            if state.get("termination_reason") == "NEEDS_USER_INPUT":
                # Resume after user narrowed topic: allow override via topic arg
                if topic and topic != state.get("topic"):
                    state["topic"] = topic
                state["termination_reason"] = None
                state["verification_status"] = "pending"
                # Re-run planner if paused there
                completed = [n for n in (state.get("completed_nodes") or []) if n != "planner"]
                state["completed_nodes"] = completed
            self.store.update_task(task_id, status="running")
            self.store.append_event(
                task_id,
                _structured_event("task_resumed", task_id=task_id, thread_id=task["thread_id"]),
            )
            return state, {"task_id": task_id, "thread_id": task["thread_id"], "resumed": True}

        created = self.store.create_task(topic=topic, task_id=task_id, thread_id=thread_id)
        state = initial_state(
            topic,
            max_iterations=self.max_iterations,
            task_id=created["task_id"],
            thread_id=created["thread_id"],
        )
        return state, {"task_id": created["task_id"], "thread_id": created["thread_id"], "resumed": False}

    def _iter_events(
        self,
        state: ResearchState,
        *,
        persist: bool = True,
        meta: Optional[dict[str, Any]] = None,
    ) -> Iterator[dict[str, Any]]:
        meta = meta or {}
        task_id = str(state.get("task_id") or meta.get("task_id") or "")
        thread_id = str(state.get("thread_id") or meta.get("thread_id") or "")
        engine = langgraph_engine()

        def emit(ev: dict[str, Any]) -> dict[str, Any]:
            if persist and task_id:
                try:
                    self.store.append_event(task_id, ev)
                except Exception:  # noqa: BLE001
                    logger.debug("append_event failed", exc_info=True)
            return ev

        yield emit(
            _structured_event(
                "status",
                message=f"初始化 Research Gap Discovery Agent（engine={engine}）",
                stage="init",
                disclaimer="帮助发现和验证候选 Research Gap，不宣称自动发现真正创新点",
                engine=engine,
                task_id=task_id,
                thread_id=thread_id,
            )
        )
        if meta.get("resumed"):
            yield emit(
                _structured_event(
                    "task_resumed",
                    task_id=task_id,
                    thread_id=thread_id,
                    current_node=state.get("current_node"),
                    completed_nodes=state.get("completed_nodes") or [],
                )
            )
        yield emit(
            _structured_event(
                "pipeline",
                message="LangGraph StateGraph topology",
                engine=engine,
                nodes=GRAPH_NODES,
                edges=GRAPH_EDGES,
                entry="memory_recall",
                conditional_from="gap_verify",
                task_id=task_id,
                thread_id=thread_id,
            )
        )
        yield emit(
            _structured_event(
                "state_snapshot",
                node="__init__",
                state=snapshot_state(dict(state)),
                diff=[],
                task_id=task_id,
                thread_id=thread_id,
            )
        )

        emitted = 0
        prev_snap = snapshot_state(dict(state))
        try:
            for phase, name, state in iter_pipeline(state):
                if phase == "start":
                    yield emit(
                        _structured_event(
                            "node_started",
                            node=name,
                            stage=state.get("stage"),
                            message=f"进入节点：{name}",
                            task_id=task_id,
                            thread_id=thread_id,
                        )
                    )
                    # Back-compat alias
                    yield emit(
                        _structured_event(
                            "node_start",
                            node=name,
                            stage=state.get("stage"),
                            message=f"进入节点：{name}",
                            task_id=task_id,
                            thread_id=thread_id,
                        )
                    )
                    continue

                events = state.get("events") or []
                for event in events[emitted:]:
                    # Preserve structured types from pipeline
                    yield emit(dict(event))
                emitted = len(events)

                snap = snapshot_state(dict(state))
                changes = diff_state(prev_snap, snap)
                prev_snap = snap
                yield emit(
                    _structured_event(
                        "state_patch",
                        node=name,
                        diff=changes,
                        state=snap,
                        task_id=task_id,
                        thread_id=thread_id,
                        current_node=name,
                        verification_round=state.get("verification_round"),
                        max_verification_rounds=state.get("max_verification_rounds"),
                        tool_call_count=state.get("tool_call_count"),
                        max_tool_calls=state.get("max_tool_calls"),
                        warnings=state.get("warnings") or [],
                        termination_reason=state.get("termination_reason"),
                    )
                )
                yield emit(
                    _structured_event(
                        "node_completed",
                        node=name,
                        stage=state.get("stage"),
                        updated_fields=[c["field"] for c in changes],
                        task_id=task_id,
                        thread_id=thread_id,
                    )
                )
                yield emit(
                    _structured_event(
                        "node_done",
                        node=name,
                        stage=state.get("stage"),
                        updated_fields=[c["field"] for c in changes],
                        task_id=task_id,
                        thread_id=thread_id,
                    )
                )

                if persist and task_id:
                    ckpt_id = self.store.save_checkpoint(
                        task_id=task_id,
                        thread_id=thread_id,
                        node_name=name,
                        state=dict(state),
                    )
                    state["last_checkpoint_id"] = ckpt_id
                    yield emit(
                        _structured_event(
                            "checkpoint_saved",
                            checkpoint_id=ckpt_id,
                            node=name,
                            task_id=task_id,
                            thread_id=thread_id,
                        )
                    )
                    self.store.update_task(
                        task_id,
                        status="paused" if state.get("status") == "paused" else "running",
                        current_node=name,
                        termination_reason=state.get("termination_reason"),
                    )

                if state.get("status") == "paused":
                    yield emit(
                        _structured_event(
                            "task_paused",
                            task_id=task_id,
                            thread_id=thread_id,
                            reason=state.get("termination_reason") or "paused",
                            message="任务已暂停，可通过 Resume 继续",
                        )
                    )
                    return

                if state.get("status") == "failed":
                    if persist and task_id:
                        self.store.update_task(
                            task_id,
                            status="failed",
                            current_node=name,
                            termination_reason=state.get("termination_reason") or "failed",
                        )
                    yield emit(
                        _structured_event(
                            "task_failed",
                            detail=state.get("error") or f"failed at {name}",
                            task_id=task_id,
                            thread_id=thread_id,
                        )
                    )
                    yield emit(
                        _structured_event(
                            "error",
                            detail=state.get("error") or f"failed at {name}",
                        )
                    )
                    return
        except Exception as exc:
            logger.exception("Gap discovery streaming failed")
            if persist and task_id:
                self.store.update_task(task_id, status="failed", termination_reason="TOOL_FAILURE")
            yield emit(_structured_event("task_failed", detail=str(exc), task_id=task_id))
            yield emit(_structured_event("error", detail=str(exc)))
            return

        if persist and task_id:
            self.store.update_task(
                task_id,
                status="completed" if state.get("status") == "completed" else state.get("status") or "completed",
                current_node=state.get("current_node") or "report",
                termination_reason=state.get("termination_reason") or "COMPLETED",
            )
            self.store.save_checkpoint(
                task_id=task_id,
                thread_id=thread_id,
                node_name=state.get("current_node") or "report",
                state=dict(state),
            )

        if not any(e.get("type") == "done" for e in (state.get("events") or [])):
            if state.get("final_report"):
                yield emit(_structured_event("report", report_markdown=state["final_report"]))
            yield emit(_structured_event("done"))
            yield emit(
                _structured_event(
                    "task_completed",
                    task_id=task_id,
                    thread_id=thread_id,
                    termination_reason=state.get("termination_reason"),
                )
            )

    def start_background_task(self, topic: str) -> dict[str, Any]:
        created = self.store.create_task(topic=topic)
        state = initial_state(
            topic,
            max_iterations=self.max_iterations,
            task_id=created["task_id"],
            thread_id=created["thread_id"],
        )

        def _worker() -> None:
            try:
                for _ in self._iter_events(state, persist=True, meta=created):
                    pass
            except Exception:
                logger.exception("background task failed: %s", created["task_id"])

        threading.Thread(target=_worker, daemon=True, name=f"research-{created['task_id'][:8]}").start()
        return created

    def resume_background_task(self, task_id: str, *, topic: Optional[str] = None) -> dict[str, Any]:
        task = self.store.get_task(task_id)
        if not task:
            raise KeyError(task_id)
        if task["status"] == "completed":
            return {"task_id": task_id, "status": "completed", "resumed": False, "reason": "already_completed"}
        # Idempotent: if already running, no-op
        if task["status"] == "running":
            return {"task_id": task_id, "status": "running", "resumed": False, "reason": "already_running"}

        state, meta = self._prepare_state(
            topic or task["topic"], task_id=task_id, resume=True
        )

        def _worker() -> None:
            try:
                for _ in self._iter_events(state, persist=True, meta=meta):
                    pass
            except Exception:
                logger.exception("resume worker failed: %s", task_id)

        threading.Thread(target=_worker, daemon=True, name=f"resume-{task_id[:8]}").start()
        return {"task_id": task_id, "thread_id": task["thread_id"], "status": "running", "resumed": True}

    def get_task_snapshot(self, task_id: str) -> dict[str, Any]:
        task = self.store.get_task(task_id)
        if not task:
            raise KeyError(task_id)
        ckpt = self.store.latest_checkpoint(task_id)
        state = ckpt["state"] if ckpt else {}
        return {
            "task_id": task_id,
            "thread_id": task["thread_id"],
            "topic": task["topic"],
            "status": task["status"],
            "current_node": task.get("current_node"),
            "termination_reason": task.get("termination_reason"),
            "updated_at": task.get("updated_at"),
            "last_checkpoint_id": ckpt.get("checkpoint_id") if ckpt else None,
            "state": snapshot_state(state) if state else {},
            "raw_state_keys": list(state.keys()) if isinstance(state, dict) else [],
            "verification_round": state.get("verification_round") if isinstance(state, dict) else None,
            "max_verification_rounds": state.get("max_verification_rounds") if isinstance(state, dict) else None,
            "tool_call_count": state.get("tool_call_count") if isinstance(state, dict) else None,
            "max_tool_calls": state.get("max_tool_calls") if isinstance(state, dict) else None,
            "warnings": state.get("warnings") if isinstance(state, dict) else [],
            "tool_traces": (state.get("tool_traces") or [])[-20:] if isinstance(state, dict) else [],
        }

    def run(self, topic: str) -> ResearchState:
        created = self.store.create_task(topic=topic)
        state = initial_state(
            topic,
            max_iterations=self.max_iterations,
            task_id=created["task_id"],
            thread_id=created["thread_id"],
        )
        final = run_pipeline(state)
        self.store.save_checkpoint(
            task_id=created["task_id"],
            thread_id=created["thread_id"],
            node_name=final.get("current_node") or "report",
            state=dict(final),
        )
        self.store.update_task(
            created["task_id"],
            status=final.get("status") or "completed",
            current_node=final.get("current_node"),
            termination_reason=final.get("termination_reason"),
        )
        return final
