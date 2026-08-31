"""Workflow-level hard-rule checks."""

from __future__ import annotations

from typing import Any


def evaluate_workflow(case: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    expect = case.get("expect") or {}
    hard_fail: list[str] = []
    notes: list[str] = []

    status = result.get("status")
    term = result.get("termination_reason")
    if expect.get("termination_reason") and term != expect["termination_reason"]:
        hard_fail.append(f"termination_reason want={expect['termination_reason']} got={term}")
    if expect.get("status") and status != expect["status"]:
        hard_fail.append(f"status want={expect['status']} got={status}")

    if expect.get("task_completes"):
        if status not in {"completed", "paused"} and term not in {
            "COMPLETED",
            "NEEDS_USER_INPUT",
            "INSUFFICIENT_EVIDENCE",
            "BUDGET_EXCEEDED",
        }:
            hard_fail.append("task did not complete or pause cleanly")

    completed = set(result.get("completed_nodes") or [])
    route_ok = True
    if expect.get("route_in"):
        route_ok = any(r in completed or r == result.get("current_node") for r in expect["route_in"])
        if not route_ok:
            notes.append("route_in soft miss")

    # Hard rule: thread isolation marker
    if result.get("_pollution"):
        hard_fail.append("thread_id state pollution detected")

    return {
        "pass": not hard_fail,
        "hard_fail": hard_fail,
        "notes": notes,
        "route_ok": route_ok,
        "latency_ms": result.get("latency_ms") or 0,
    }
