"""Phase checks for evidence-chain upgrades (fulltext → citation → lifecycle)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env", override=True)


def phase1_fulltext() -> None:
    from gap_discovery.fulltext import FullTextService

    svc = FullTextService()
    paper = {
        "paper_id": "test-ityfuzz",
        "title": "ItyFuzz: Snapshot-Based Fuzzer for Smart Contract",
        "year": 2023,
        "arxiv_id": "2306.17135",
        "content": "abstract only placeholder",
    }
    out = svc.enrich_paper(paper)
    print(
        f"[P1] fulltext_status={out.get('fulltext_status')} "
        f"source={out.get('fulltext_source')} "
        f"sections={list((out.get('sections') or {}).keys())[:8]}"
    )
    assert out.get("fulltext_status") == "FULLTEXT", out.get("fulltext_error")
    assert out.get("sections"), "expected sections"
    print("[OK] Phase1 FullTextService")


def phase2_sections() -> None:
    from gap_discovery.pdf_parser import read_section_text, split_sections

    sample = """
    Abstract
    This is an abstract about bridge security.

    1 Introduction
    Cross-chain bridges are risky.

    3 Method
    We use static analysis.

    5 Limitations
    However, our tool only supports EVM contracts and requires manual target identification.

    6 Conclusion
    Future work will expand architectures.
    References
    [1] Foo
    """
    sections = split_sections(sample)
    lim = read_section_text(sections, "limitations", "discussion")
    assert "manual" in lim.lower() or "evm" in lim.lower(), sections
    print(f"[OK] Phase2 section parser keys={list(sections.keys())}")


def phase_graph_has_evidence_chain() -> None:
    from gap_discovery.graph import build_gap_discovery_graph, langgraph_engine

    g = build_gap_discovery_graph()
    assert g is not None
    nodes = set(g.get_graph().nodes)
    assert "evidence_chain" in nodes, nodes
    print(f"[OK] LangGraph includes evidence_chain; engine={langgraph_engine()}")


def main() -> None:
    print("=== Evidence-chain phase smoke ===\n")
    phase2_sections()
    phase1_fulltext()
    phase_graph_has_evidence_chain()
    print("\nPhase smoke passed.")


if __name__ == "__main__":
    main()
