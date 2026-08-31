"""Unit test for PaperCard model (no network)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from gap_discovery.models import (
    EvidenceSourceType,
    PaperCard,
    paper_card_from_search_result,
)


def test_paper_card_fields_and_provenance() -> None:
    raw = {
        "paper_id": "demo-smartaxe",
        "title": "SmartAxe: Detecting Cross-Chain Vulnerabilities in Bridge Smart Contracts",
        "authors": ["A", "B"],
        "year": 2024,
        "venue": "Demo Venue",
        "url": "https://example.com/smartaxe",
        "content": (
            "However, existing tools fail to analyze cross-chain bridge contracts. "
            "We propose SmartAxe, a fine-grained static analysis approach for detecting "
            "vulnerabilities in bridge smart contracts. Future work includes expanding datasets."
        ),
        "citation_count": 27,
        "evidence_level": "abstract_only",
        "source": "openalex",
        "doi": "10.1000/demo",
    }
    card = paper_card_from_search_result(raw, research_direction="静态分析 / Static Analysis")
    assert isinstance(card, PaperCard)
    assert card.paper_id == "demo-smartaxe"
    assert card.evidence_level.value == "abstract_only"
    assert card.research_direction.startswith("静态分析")
    assert card.core_technique and "static analysis" in card.core_technique
    assert card.contributions, "should extract contribution cue from abstract"
    assert card.inferred_weaknesses, "abstract_only should add AI inferred note"
    assert card.inferred_weaknesses[0].source_type == EvidenceSourceType.AI_INFERRED
    assert "AI inferred" in (card.inferred_weaknesses[0].notes or "")
    # self-reported only when abstract explicitly cues limitation/future work
    assert any(
        e.source_type == EvidenceSourceType.SELF_REPORTED for e in card.self_reported_limitations
    )
    row = card.to_matrix_row()
    assert "paper_id" in row and "inferred_weaknesses" in row
    print("PaperCard test OK")
    print("method:", card.method)
    print("inferred:", card.inferred_weaknesses[0].as_labeled_text())


if __name__ == "__main__":
    test_paper_card_fields_and_provenance()
