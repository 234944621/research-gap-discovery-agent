"""Summarize ResearchState for SSE UI (avoid dumping full papers/events)."""

from __future__ import annotations

from typing import Any, Optional

WATCHED_FIELDS = [
    "topic",
    "task_id",
    "thread_id",
    "current_node",
    "stage",
    "status",
    "research_questions",
    "search_keywords",
    "papers",
    "paper_cards",
    "method_taxonomy",
    "limitations",
    "evolution_chain",
    "limitation_lifecycles",
    "external_critiques",
    "candidate_gaps",
    "verified_gaps",
    "gap_verification_results",
    "cross_domain_methods",
    "final_candidates",
    "iteration_count",
    "verification_round",
    "max_verification_rounds",
    "tool_call_count",
    "max_tool_calls",
    "token_usage",
    "max_token_budget",
    "visited_actions",
    "tool_traces",
    "retry_counts",
    "warnings",
    "evidence_status",
    "verification_status",
    "termination_reason",
    "last_checkpoint_id",
    "completed_nodes",
    "needs_more_evidence",
    "rag_hits",
    "final_report",
]


def summarize_value(field: str, value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, bool) or isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        if field == "final_report":
            return {"chars": len(value), "head": value[:160]}
        return value if len(value) <= 240 else value[:240] + "…"
    if isinstance(value, list):
        preview: list[Any] = []
        for item in value[:3]:
            if isinstance(item, dict):
                preview.append(
                    {
                        k: item.get(k)
                        for k in (
                            "paper_id",
                            "title",
                            "year",
                            "gap_id",
                            "status",
                            "current_status",
                            "limitation_id",
                            "description",
                            "fulltext_status",
                            "name",
                        )
                        if k in item
                    }
                    or {"keys": list(item.keys())[:6]}
                )
            else:
                preview.append(str(item)[:120])
        return {"count": len(value), "preview": preview}
    if isinstance(value, dict):
        return {
            "keys": list(value.keys())[:16],
            "preview": {
                k: summarize_value(k, value.get(k))
                for k in list(value.keys())[:6]
            },
        }
    return str(value)[:200]


def snapshot_state(state: dict[str, Any]) -> dict[str, Any]:
    return {field: summarize_value(field, state.get(field)) for field in WATCHED_FIELDS}


def diff_state(
    before: Optional[dict[str, Any]], after: dict[str, Any]
) -> list[dict[str, Any]]:
    prev = before or {}
    changes: list[dict[str, Any]] = []
    for field in WATCHED_FIELDS:
        left = summarize_value(field, prev.get(field))
        right = summarize_value(field, after.get(field))
        if left != right:
            changes.append(
                {
                    "field": field,
                    "before": left,
                    "after": right,
                }
            )
    return changes
