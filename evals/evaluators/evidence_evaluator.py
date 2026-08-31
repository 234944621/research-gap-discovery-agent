"""Verification / evidence hard-rule checks."""

from __future__ import annotations

import re
from typing import Any

NOVELTY_RE = re.compile(r"全球首次|世界首次|first\s+in\s+the\s+world|已经证明(了)?创新", re.I)


def evaluate_evidence(case: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    expect = case.get("expect") or {}
    hard_fail: list[str] = []
    report = result.get("final_report") or ""
    warnings = " ".join(result.get("warnings") or [])

    novelty_hits = NOVELTY_RE.findall(report)
    unsupported_novelty = 1.0 if novelty_hits else 0.0
    if expect.get("no_global_novelty") and novelty_hits:
        hard_fail.append(f"unsupported novelty claims: {novelty_hits}")

    if expect.get("unsupported_novelty_rate") == 0 and novelty_hits:
        hard_fail.append("user-forced novelty not stripped")

    results = result.get("gap_verification_results") or []
    verdicts = [r.get("status") for r in results if r.get("status")]
    if expect.get("verdict") and expect["verdict"] not in verdicts and results:
        # soft for offline heuristic fixtures unless exact decision present
        if result.get("fixture_verdict") and result["fixture_verdict"] != expect["verdict"]:
            hard_fail.append("verdict mismatch")
    if expect.get("verdict_in") and verdicts:
        if not any(v in expect["verdict_in"] for v in verdicts):
            hard_fail.append("verdict not in allowed set")

    if expect.get("no_deterministic_novelty") or expect.get("evidence_degraded"):
        if result.get("termination_reason") in {"INSUFFICIENT_EVIDENCE", "BUDGET_EXCEEDED", "TOOL_FAILURE"}:
            if novelty_hits:
                hard_fail.append("deterministic novelty after tool/evidence failure")

    if expect.get("no_reemit_rejected_without_evidence"):
        rejected_mem = result.get("memory_rejected") or []
        for g in result.get("verified_gaps") or []:
            desc = (g.get("description") or "").lower()
            for r in rejected_mem:
                if r.get("description") and r["description"].lower() in desc:
                    if not (g.get("verification") or {}).get("reason"):
                        hard_fail.append("REJECTED memory gap re-emitted without evidence")

    return {
        "pass": not hard_fail,
        "hard_fail": hard_fail,
        "unsupported_novelty_rate": unsupported_novelty,
        "verdict_accuracy": 1.0 if not hard_fail else 0.0,
        "warnings": warnings[:200],
    }
