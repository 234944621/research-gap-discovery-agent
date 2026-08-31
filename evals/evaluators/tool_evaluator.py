"""Tool-level hard-rule checks."""

from __future__ import annotations

from typing import Any


def evaluate_tools(case: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    expect = case.get("expect") or {}
    hard_fail: list[str] = []
    traces = result.get("tool_traces") or []
    unauthorized = [
        t for t in traces if t.get("status") == "rejected" or t.get("error_type") == "unauthorized_tool"
    ]
    if unauthorized and not expect.get("no_unauthorized_tool", True):
        hard_fail.append("unauthorized tool executed")
    # Hard: unauthorized must never succeed
    for t in traces:
        if t.get("tool_name") not in {
            None,
            "search_papers",
            "recall_memory",
            "retrieve_rag",
            "find_citing_papers",
            "get_citation_context",
            "read_fulltext_section",
        } and t.get("status") == "success":
            hard_fail.append(f"unauthorized tool success: {t.get('tool_name')}")

    max_tools = result.get("max_tool_calls")
    count = int(result.get("tool_call_count") or 0)
    if max_tools is not None and count > int(max_tools):
        hard_fail.append("exceeded max_tool_calls")

    max_vr = result.get("max_verification_rounds")
    vr = int(result.get("verification_round") or 0)
    if max_vr is not None and vr > int(max_vr):
        hard_fail.append("exceeded max_verification_rounds")

    success_n = sum(1 for t in traces if t.get("status") in {"success", "empty"})
    total = len(traces) or 1
    success_rate = success_n / total

    # duplicate rate
    sigs = []
    for t in traces:
        sigs.append(f"{t.get('tool_name')}:{t.get('arguments')}")
    dup_rate = 1 - (len(set(map(str, sigs))) / max(len(sigs), 1))

    if expect.get("error_type"):
        # unit-style fixture path
        err = result.get("classified") or {}
        if err.get("error_type") != expect["error_type"]:
            hard_fail.append("error classification mismatch")

    return {
        "pass": not hard_fail,
        "hard_fail": hard_fail,
        "tool_call_success_rate": success_rate,
        "average_tool_calls": count,
        "duplicate_call_rate": dup_rate,
    }
