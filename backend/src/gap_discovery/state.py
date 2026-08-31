"""ResearchState schema, defaults, and list merge helpers."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Literal, Optional, TypedDict
from uuid import uuid4


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ResearchState(TypedDict, total=False):
    topic: str

    # Task / checkpoint identity
    task_id: str
    thread_id: str
    current_node: str

    # Planner
    research_questions: list[str]
    search_keywords: list[str]
    plan: dict[str, Any]

    # Search / papers
    papers: list[dict[str, Any]]
    paper_cards: list[dict[str, Any]]

    # Analyzer
    method_taxonomy: list[dict[str, Any]]
    analysis: dict[str, Any]
    limitations: list[dict[str, Any]]

    # Gap + verification
    candidate_gaps: list[dict[str, Any]]
    verified_gaps: list[dict[str, Any]]
    current_gap: Optional[dict[str, Any]]
    gap_verification_results: list[dict[str, Any]]
    cross_domain_methods: list[dict[str, Any]]
    final_candidates: list[dict[str, Any]]

    # Memory / RAG
    research_memory: list[dict[str, Any]]
    memory_semantic_lessons: list[dict[str, Any]]
    memory_entities: list[dict[str, Any]]
    memory_procedures: list[dict[str, Any]]
    rag_hits: list[dict[str, Any]]

    # Control (legacy + production)
    iteration_count: int
    max_iterations: int
    verification_round: int
    max_verification_rounds: int
    tool_call_count: int
    max_tool_calls: int
    token_usage: Optional[int]
    max_token_budget: Optional[int]
    visited_actions: list[str]
    tool_traces: list[dict[str, Any]]
    retry_counts: dict[str, int]
    last_error: Optional[dict[str, Any]]
    warnings: list[str]
    evidence_status: str
    verification_status: str
    termination_reason: Optional[str]
    started_at: Optional[str]
    updated_at: Optional[str]
    needs_more_evidence: bool
    final_report: str
    stage: str
    notices: list[str]
    events: list[dict[str, Any]]
    error: Optional[str]
    status: Literal["running", "completed", "failed", "paused"]

    # Idempotency / resume
    completed_nodes: list[str]
    executed_side_effects: list[str]
    last_checkpoint_id: Optional[str]

    # Legacy compatibility
    scope: dict[str, Any]
    research_directions: list[dict[str, Any]]
    selected_direction: Optional[str]
    landscape_papers: list[dict[str, Any]]
    candidate_papers: list[dict[str, Any]]
    evolution_chain: list[dict[str, Any]]
    external_critiques: list[dict[str, Any]]
    limitation_lifecycles: list[dict[str, Any]]


# List fields that should merge by append+dedupe rather than wholesale replace
LIST_MERGE_FIELDS = {
    "events",
    "notices",
    "warnings",
    "visited_actions",
    "tool_traces",
    "executed_side_effects",
    "completed_nodes",
}


def merge_list_unique(existing: list[Any] | None, incoming: list[Any] | None) -> list[Any]:
    """Append reducer for parallel-safe-ish list updates (JSON-serializable items)."""

    out: list[Any] = list(existing or [])
    seen = {json_key(x) for x in out}
    for item in incoming or []:
        key = json_key(item)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def json_key(item: Any) -> str:
    import json

    try:
        return json.dumps(item, sort_keys=True, ensure_ascii=False, default=str)
    except TypeError:
        return str(item)


def merge_state_update(base: ResearchState, update: dict[str, Any]) -> ResearchState:
    """Apply a node partial update onto base without dropping sibling list history."""

    merged: ResearchState = dict(base)  # type: ignore[assignment]
    for key, value in update.items():
        if key in LIST_MERGE_FIELDS and isinstance(value, list):
            merged[key] = merge_list_unique(merged.get(key), value)  # type: ignore[literal-required]
        elif key == "retry_counts" and isinstance(value, dict):
            cur = dict(merged.get("retry_counts") or {})
            cur.update(value)
            merged["retry_counts"] = cur
        else:
            merged[key] = value  # type: ignore[literal-required]
    merged["updated_at"] = _utcnow_iso()
    return merged


def initial_state(
    topic: str,
    *,
    max_iterations: int = 6,
    task_id: Optional[str] = None,
    thread_id: Optional[str] = None,
    max_verification_rounds: Optional[int] = None,
    max_tool_calls: Optional[int] = None,
    max_token_budget: Optional[int] = None,
) -> ResearchState:
    import os

    tid = task_id or str(uuid4())
    thr = thread_id or tid
    now = _utcnow_iso()
    max_vr = max_verification_rounds
    if max_vr is None:
        max_vr = int(os.getenv("GAP_MAX_VERIFICATION_ROUNDS", str(max_iterations)))
    max_tc = max_tool_calls
    if max_tc is None:
        max_tc = int(os.getenv("GAP_MAX_TOOL_CALLS", "24"))
    token_budget = max_token_budget
    if token_budget is None:
        raw = os.getenv("GAP_MAX_TOKEN_BUDGET", "").strip()
        token_budget = int(raw) if raw.isdigit() else None

    return ResearchState(
        topic=topic,
        task_id=tid,
        thread_id=thr,
        current_node="",
        research_questions=[],
        search_keywords=[],
        plan={},
        papers=[],
        paper_cards=[],
        method_taxonomy=[],
        analysis={},
        limitations=[],
        candidate_gaps=[],
        verified_gaps=[],
        current_gap=None,
        gap_verification_results=[],
        cross_domain_methods=[],
        final_candidates=[],
        research_memory=[],
        memory_semantic_lessons=[],
        memory_entities=[],
        memory_procedures=[],
        rag_hits=[],
        iteration_count=0,
        max_iterations=max_iterations,
        verification_round=0,
        max_verification_rounds=max_vr,
        tool_call_count=0,
        max_tool_calls=max_tc,
        token_usage=0,
        max_token_budget=token_budget,
        visited_actions=[],
        tool_traces=[],
        retry_counts={},
        last_error=None,
        warnings=[],
        evidence_status="unknown",
        verification_status="pending",
        termination_reason=None,
        started_at=now,
        updated_at=now,
        needs_more_evidence=False,
        final_report="",
        stage="init",
        notices=[],
        events=[],
        error=None,
        status="running",
        completed_nodes=[],
        executed_side_effects=[],
        last_checkpoint_id=None,
        scope={},
        research_directions=[],
        selected_direction=None,
        landscape_papers=[],
        candidate_papers=[],
        evolution_chain=[],
        external_critiques=[],
        limitation_lifecycles=[],
    )


def append_event(state: ResearchState, event: dict[str, Any]) -> ResearchState:
    events = list(state.get("events") or [])
    events.append(event)
    state["events"] = events
    state["updated_at"] = _utcnow_iso()
    return state


def emit(state: ResearchState, message: str, **extra: Any) -> ResearchState:
    stage = extra.pop("stage", state.get("stage", ""))
    state["stage"] = stage
    state["updated_at"] = _utcnow_iso()
    return append_event(
        state,
        {"type": "status", "message": message, "stage": stage, **extra},
    )


def touch(state: ResearchState, *, node: Optional[str] = None) -> ResearchState:
    state["updated_at"] = _utcnow_iso()
    if node is not None:
        state["current_node"] = node
    return state


def clone_state(state: ResearchState) -> ResearchState:
    return deepcopy(state)
