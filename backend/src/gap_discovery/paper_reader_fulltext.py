"""Fulltext-aware PaperCard enrichment (section-scoped, no whole-PDF dumps)."""

from __future__ import annotations

import logging
import re
from typing import Any

from gap_discovery.models import (
    EvidenceItem,
    EvidenceLevel,
    EvidenceRef,
    EvidenceSourceType,
    PaperCard,
)
from gap_discovery.pdf_parser import read_section_text

logger = logging.getLogger(__name__)


def enrich_card_from_fulltext(card: PaperCard, paper: dict[str, Any]) -> PaperCard:
    """Fill PaperCard fields from section chunks when FULLTEXT is available."""

    if paper.get("fulltext_status") != "FULLTEXT":
        return card

    sections = paper.get("sections") or {}
    card.fulltext_status = "FULLTEXT"
    card.fulltext_source = paper.get("fulltext_source")
    card.pdf_url = paper.get("pdf_url") or card.pdf_url
    card.evidence_level = EvidenceLevel.FULL_TEXT
    card.sections_present = list(sections.keys())

    intro = read_section_text(sections, "abstract", "introduction", max_chars=2500)
    method = read_section_text(sections, "method", max_chars=2500)
    evaluation = read_section_text(sections, "evaluation", max_chars=2000)
    limitations = read_section_text(
        sections, "limitations", "discussion", "threats_to_validity", max_chars=2500
    )
    future = read_section_text(sections, "conclusion", "future_work", max_chars=2000)

    if intro and not card.research_problem:
        card.research_problem = _first_sentence(intro, prefer=("challenge", "problem", "however"))
        card.evidence_refs.append(
            EvidenceRef(
                claim=card.research_problem or "research problem from intro",
                evidence_type=EvidenceSourceType.ABSTRACT,
                paper_id=card.paper_id,
                section="introduction",
                quote_or_summary=(intro[:220]),
                evidence_level=EvidenceLevel.FULL_TEXT,
            )
        )

    if method:
        card.method = _compress(method, 320)
        card.evidence_refs.append(
            EvidenceRef(
                claim="method description",
                evidence_type=EvidenceSourceType.METADATA,
                paper_id=card.paper_id,
                section="method",
                quote_or_summary=method[:220],
                evidence_level=EvidenceLevel.FULL_TEXT,
            )
        )

    if evaluation and not card.main_results:
        card.main_results = _compress(evaluation, 280)

    # self-reported limitations strictly from limitation-ish sections
    lim_items = _extract_limitation_bullets(limitations) if limitations else []
    for text in lim_items[:5]:
        item = EvidenceItem(
            text=text,
            source_type=EvidenceSourceType.SELF_REPORTED,
            evidence_level=EvidenceLevel.FULL_TEXT,
            location="Limitations/Discussion",
            source_paper_id=card.paper_id,
            source_paper_title=card.title,
            confidence="medium",
            notes="extracted from fulltext limitation-related sections",
        )
        card.self_reported_limitations.append(item)
        card.evidence_refs.append(
            EvidenceRef(
                claim=text,
                evidence_type=EvidenceSourceType.SELF_REPORTED,
                paper_id=card.paper_id,
                section="limitations",
                quote_or_summary=text[:220],
                evidence_level=EvidenceLevel.FULL_TEXT,
            )
        )

    if future:
        for text in _extract_limitation_bullets(future)[:3]:
            if "future" in text.lower() or "we plan" in text.lower() or "open" in text.lower():
                card.future_work.append(text[:220])

    # drop generic abstract_only AI placeholder if we now have real self-reported
    if card.self_reported_limitations:
        card.inferred_weaknesses = [
            w
            for w in card.inferred_weaknesses
            if "尚未从全文" not in (w.text or "")
        ]

    return card


def _extract_limitation_bullets(text: str) -> list[str]:
    if not text:
        return []
    # sentence split with limitation cues preferred
    sentences = re.split(r"(?<=[\.\!\?])\s+", text)
    scored: list[tuple[int, str]] = []
    cues = (
        "limit",
        "however",
        "only",
        "cannot",
        "unable",
        "fail",
        "lack",
        "not support",
        "manual",
        "assume",
        "future work",
        "threat",
    )
    for s in sentences:
        s = s.strip()
        if len(s) < 40 or len(s) > 280:
            continue
        score = sum(1 for c in cues if c in s.lower())
        if score:
            scored.append((score, s))
    scored.sort(key=lambda x: -x[0])
    out = [s for _, s in scored[:6]]
    if not out:
        # fallback: first 2 medium sentences
        out = [s.strip() for s in sentences if 60 <= len(s.strip()) <= 240][:2]
    return out


def _first_sentence(text: str, prefer: tuple[str, ...] = ()) -> str:
    sentences = re.split(r"(?<=[\.\!\?])\s+", text)
    for s in sentences:
        if any(p in s.lower() for p in prefer) and len(s) > 40:
            return s.strip()[:240]
    return (sentences[0].strip()[:240] if sentences else text[:240])


def _compress(text: str, n: int) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text[:n]
