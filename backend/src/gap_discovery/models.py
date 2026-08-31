"""PaperCard data model for Research Gap Discovery.

证据来源约束（必须遵守）：
- self_reported_limitations: 仅来自论文明确表述
- external_critiques: 仅来自后续论文对该文的引用语境
- inferred_weaknesses: Agent 推断，必须标记为 AI inferred，不可伪装成作者观点
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class EvidenceLevel(str, Enum):
    ABSTRACT_ONLY = "abstract_only"
    FULL_TEXT = "full_text"
    MIXED = "mixed"


class EvidenceSourceType(str, Enum):
    SELF_REPORTED = "self_reported"
    EXTERNAL_CRITIQUE = "external_critique"
    AI_INFERRED = "ai_inferred"
    METADATA = "metadata"
    ABSTRACT = "abstract"


class EvidenceItem(BaseModel):
    """A single piece of evidence with provenance."""

    text: str
    source_type: EvidenceSourceType
    evidence_level: EvidenceLevel = EvidenceLevel.ABSTRACT_ONLY
    location: Optional[str] = Field(
        default=None,
        description="e.g. Limitations / Discussion / abstract / citation context",
    )
    source_paper_id: Optional[str] = None
    source_paper_title: Optional[str] = None
    confidence: Optional[str] = Field(default=None, description="high|medium|low")
    notes: Optional[str] = None

    def as_labeled_text(self) -> str:
        prefix = {
            EvidenceSourceType.SELF_REPORTED: "[Self-reported]",
            EvidenceSourceType.EXTERNAL_CRITIQUE: "[External critique]",
            EvidenceSourceType.AI_INFERRED: "[AI inferred weakness]",
            EvidenceSourceType.METADATA: "[Metadata]",
            EvidenceSourceType.ABSTRACT: "[Abstract]",
        }[self.source_type]
        loc = f" ({self.location})" if self.location else ""
        return f"{prefix}{loc} {self.text}"


class EvidenceRef(BaseModel):
    """Short, traceable claim↔evidence pointer (no long verbatim quotes)."""

    claim: str
    evidence_type: EvidenceSourceType
    paper_id: str
    section: Optional[str] = None
    quote_or_summary: str = Field(
        default="",
        description="Short fragment or reliable summary (<=240 chars)",
    )
    evidence_level: EvidenceLevel = EvidenceLevel.ABSTRACT_ONLY


class ExternalCritique(BaseModel):
    target_paper_id: str
    citing_paper_id: Optional[str] = None
    citing_paper_title: Optional[str] = None
    citing_year: Optional[int] = None
    citation_context: Optional[str] = None
    critique_summary: Optional[str] = None
    critique_type: str = "UNKNOWN"  # NEUTRAL|POSITIVE|EXTENSION|CRITIQUE|UNKNOWN
    source_section: Optional[str] = None
    confidence: Optional[str] = None
    fulltext_status: Optional[str] = None


class PaperCard(BaseModel):
    """Structured fast-reading card for one paper."""

    paper_id: str
    title: str
    authors: list[str] = Field(default_factory=list)
    year: Optional[int] = None
    venue: Optional[str] = None
    url: Optional[str] = None

    research_problem: Optional[str] = None
    method: Optional[str] = None
    core_technique: Optional[str] = None

    contributions: list[str] = Field(default_factory=list)

    experimental_objects: Optional[str] = None
    dataset: Optional[str] = None
    metrics: Optional[str] = None
    main_results: Optional[str] = None

    self_reported_limitations: list[EvidenceItem] = Field(default_factory=list)
    future_work: list[str] = Field(default_factory=list)

    external_critiques: list[EvidenceItem] = Field(default_factory=list)
    inferred_weaknesses: list[EvidenceItem] = Field(default_factory=list)
    evidence_refs: list[EvidenceRef] = Field(default_factory=list)

    assumptions: list[str] = Field(default_factory=list)
    scope: Optional[str] = None
    uncovered_scenarios: list[str] = Field(default_factory=list)

    references: list[str] = Field(default_factory=list)

    # 运行/检索元数据
    abstract: str = ""
    citation_count: int = 0
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    pdf_url: Optional[str] = None
    fulltext_status: str = "ABSTRACT_ONLY"  # FULLTEXT|ABSTRACT_ONLY|FAILED
    fulltext_source: Optional[str] = None
    evidence_level: EvidenceLevel = EvidenceLevel.ABSTRACT_ONLY
    source: Optional[str] = Field(default=None, description="semantic_scholar|openalex|...")
    citing_papers: list[dict[str, Any]] = Field(default_factory=list)
    research_direction: Optional[str] = None
    sections_present: list[str] = Field(default_factory=list)

    def to_matrix_row(self) -> dict[str, Any]:
        """Compact row for Literature Matrix UI/report."""

        return {
            "paper_id": self.paper_id,
            "title": self.title,
            "year": self.year,
            "venue": self.venue,
            "citation_count": self.citation_count,
            "evidence_level": self.evidence_level.value,
            "research_problem": self.research_problem,
            "method": self.method,
            "core_technique": self.core_technique,
            "contributions": self.contributions,
            "self_reported_limitations": [
                e.as_labeled_text() for e in self.self_reported_limitations
            ],
            "external_critiques": [e.as_labeled_text() for e in self.external_critiques],
            "inferred_weaknesses": [e.as_labeled_text() for e in self.inferred_weaknesses],
            "url": self.url,
        }

    def to_state_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


def paper_card_from_search_result(
    paper: dict[str, Any],
    *,
    research_direction: Optional[str] = None,
    extract_hints: bool = True,
) -> PaperCard:
    """Build a PaperCard from academic search payload (abstract-first)."""

    paper_id = (
        paper.get("paper_id")
        or paper.get("openalex_id")
        or paper.get("doi")
        or paper.get("arxiv_id")
        or f"paper-{uuid4().hex[:10]}"
    )
    abstract = (paper.get("content") or paper.get("abstract") or "").strip()
    evidence_level = EvidenceLevel(
        paper.get("evidence_level") or EvidenceLevel.ABSTRACT_ONLY.value
    )

    card = PaperCard(
        paper_id=str(paper_id),
        title=paper.get("title") or "Untitled",
        authors=list(paper.get("authors") or []),
        year=paper.get("year"),
        venue=paper.get("venue"),
        url=paper.get("url"),
        abstract=abstract,
        citation_count=int(paper.get("citation_count") or 0),
        doi=paper.get("doi"),
        arxiv_id=paper.get("arxiv_id"),
        pdf_url=paper.get("pdf_url"),
        fulltext_status=str(paper.get("fulltext_status") or "ABSTRACT_ONLY"),
        fulltext_source=paper.get("fulltext_source"),
        evidence_level=evidence_level,
        source=paper.get("source"),
        citing_papers=list(paper.get("citing_papers") or []),
        research_direction=research_direction,
        sections_present=list((paper.get("sections") or {}).keys()),
    )

    if extract_hints and abstract:
        card = _fill_abstract_hints(card)

    # 尚无全文时：不得伪造 self_reported_limitations
    # inferred 只能显式标记 AI inferred，且保持谨慎
    if evidence_level == EvidenceLevel.ABSTRACT_ONLY and not card.inferred_weaknesses:
        card.inferred_weaknesses = [
            EvidenceItem(
                text=(
                    "当前仅有摘要级证据（evidence_level=abstract_only），"
                    "作者自述局限与外部批评尚未从全文/引用语境中核验。"
                ),
                source_type=EvidenceSourceType.AI_INFERRED,
                evidence_level=EvidenceLevel.ABSTRACT_ONLY,
                location="pipeline",
                confidence="low",
                notes="AI inferred weakness — not an author claim",
            )
        ]

    return card


def _fill_abstract_hints(card: PaperCard) -> PaperCard:
    """Lightweight heuristic extraction from abstract only.

    Step 3 只建立模型与可追溯字段；完整 PaperReader（按章节抽取）在 Step 5。
    """

    text = card.abstract
    lower = text.lower()

    # research problem hints
    for pattern in [
        r"(?:however|but|yet)[, ]+(.{20,160}?)(?:\.|$)",
        r"(?:challenge|problem|issue)[s]?[: ]+(.{20,160}?)(?:\.|$)",
        r"(?:漏洞|问题|挑战)[：: ]*(.{8,80})",
    ]:
        m = re.search(pattern, text, flags=re.IGNORECASE)
        if m:
            card.research_problem = m.group(1).strip()
            break

    # method / technique hints
    method_keywords = [
        "static analysis",
        "symbolic execution",
        "fuzzing",
        "dynamic analysis",
        "machine learning",
        "deep learning",
        "taint analysis",
        "formal verification",
        "graph neural",
        "llm",
        "静态分析",
        "符号执行",
        "模糊测试",
    ]
    found = [k for k in method_keywords if k in lower]
    if found:
        card.core_technique = ", ".join(found[:3])
        card.method = (
            f"Abstract mentions technique(s): {card.core_technique} "
            f"[Abstract-only heuristic; pending full-text confirmation]"
        )

    # contribution sentence
    for pattern in [
        r"(?:we propose|we present|this paper proposes|this paper presents)(.{20,180}?)(?:\.|$)",
        r"(?:本文提出|本文提出了|本文设计)(.{8,80})",
    ]:
        m = re.search(pattern, text, flags=re.IGNORECASE)
        if m:
            card.contributions = [
                m.group(0).strip() + " [from abstract; not verified against full text]"
            ]
            break

    # future work / limitation cues in abstract (rare) — still treat carefully
    if re.search(r"future work|limitation|threats to validity|不足|未来工作", lower):
        # 只有摘要里显式出现时，才记为 self-reported 候选，并标注来源是 abstract
        m = re.search(
            r"((?:future work|limitation|threats to validity|不足|未来工作).{0,160})",
            text,
            flags=re.IGNORECASE,
        )
        if m:
            card.self_reported_limitations.append(
                EvidenceItem(
                    text=m.group(1).strip(),
                    source_type=EvidenceSourceType.SELF_REPORTED,
                    evidence_level=EvidenceLevel.ABSTRACT_ONLY,
                    location="abstract",
                    source_paper_id=card.paper_id,
                    source_paper_title=card.title,
                    confidence="low",
                    notes="Extracted from abstract cue; needs full-text confirmation",
                )
            )

    return card


def cards_to_literature_matrix(cards: list[PaperCard]) -> list[dict[str, Any]]:
    return [c.to_matrix_row() for c in cards]
