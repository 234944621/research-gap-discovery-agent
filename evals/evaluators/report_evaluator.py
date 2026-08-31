"""Report / citation hard-rule checks."""

from __future__ import annotations

from typing import Any

from gap_discovery.safety import known_evidence_index, validate_citations


def evaluate_report(case: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    expect = case.get("expect") or {}
    hard_fail: list[str] = []
    report = result.get("final_report") or ""

    # Structure completeness
    needed = ["Candidate Research Gaps", "Evidence Boundary", "候选"]
    structure_ok = any(n in report for n in needed) or not report
    if report and not structure_ok:
        hard_fail.append("report structure incomplete")

    # Candidate boundary language
    if report and "候选" not in report and "candidate" not in report.lower() and "KEEP ≠" not in report:
        # soft note unless hard expect
        if expect.get("candidate_boundary", True) and len(report) > 200:
            hard_fail.append("missing candidate-gap boundary language")

    known_ids, known_titles = known_evidence_index(result)
    claimed_ids = []
    claimed_titles = []
    for gap in result.get("verified_gaps") or []:
        claimed_ids.extend([str(x) for x in (gap.get("supporting_papers") or [])])
        ver = gap.get("verification") or {}
        claimed_titles.extend([str(t) for t in (ver.get("closest_existing_work") or [])])
    # Also scan fixture intentional bad citations
    for pid in result.get("claimed_paper_ids") or []:
        claimed_ids.append(str(pid))

    check = validate_citations(
        claimed_ids=claimed_ids,
        claimed_titles=claimed_titles,
        known_paper_ids=known_ids,
        known_titles=known_titles,
    )
    citation_accuracy = 1.0
    if claimed_ids or claimed_titles:
        bad = len(check["invalid_paper_ids"]) + len(check["invalid_titles"])
        total = len(claimed_ids) + len(claimed_titles)
        citation_accuracy = max(0.0, 1.0 - bad / max(total, 1))
    if expect.get("drop_invalid"):
        # State after reporter must not retain unverifiable supporting_papers
        claimed_after = []
        for gap in result.get("verified_gaps") or []:
            claimed_after.extend([str(x) for x in (gap.get("supporting_papers") or [])])
        check_after = validate_citations(
            claimed_ids=claimed_after,
            claimed_titles=[],
            known_paper_ids=known_ids,
            known_titles=known_titles,
        )
        if check_after["invalid_paper_ids"]:
            hard_fail.append(
                f"invalid paper_id remains in state: {check_after['invalid_paper_ids']}"
            )
        for bad_id in check_after["invalid_paper_ids"] or result.get("claimed_paper_ids") or []:
            # Only fail if still presented as supporting evidence in report
            marker = f"supporting_papers: ['{bad_id}']"
            marker2 = f'supporting_papers: ["{bad_id}"]'
            if marker in report or marker2 in report or f"supporting_papers: [{bad_id}]" in report:
                hard_fail.append(f"invalid paper_id in report: {bad_id}")

    if expect.get("citation_accuracy") is not None and citation_accuracy < float(expect["citation_accuracy"]):
        # For fixture that injects bad ids into claimed list intentionally before scrub
        if not expect.get("drop_invalid"):
            hard_fail.append("citation_accuracy below threshold")

    return {
        "pass": not hard_fail,
        "hard_fail": hard_fail,
        "citation_accuracy": citation_accuracy,
        "structure_ok": structure_ok,
    }
