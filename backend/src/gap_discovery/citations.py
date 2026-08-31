"""Forward citations + citation-context critique extraction."""

from __future__ import annotations

import logging
import os
import re
from datetime import datetime
from typing import Any, Optional

import requests

from gap_discovery.fulltext import FullTextService
from gap_discovery.llm_utils import build_llm, extract_json, llm_chat
from gap_discovery.pdf_parser import read_section_text
from config import Configuration

logger = logging.getLogger(__name__)


def paper_age_years(year: Optional[int], *, now: Optional[int] = None) -> Optional[int]:
    if not year:
        return None
    return max(0, (now or datetime.now().year) - int(year))


def citation_budget_for_age(age: Optional[int]) -> dict[str, Any]:
    """Age-aware citation strategy (MVP)."""

    max_citing = int(os.getenv("MAX_CITING_PAPERS_PER_TARGET", "6"))
    if age is None:
        return {"mode": "standard", "max_citing": max_citing, "critique": True}
    if age < 1:
        return {"mode": "young", "max_citing": min(3, max_citing), "critique": False}
    if age <= 3:
        return {"mode": "prime", "max_citing": max_citing, "critique": True}
    return {
        "mode": "mature",
        "max_citing": min(5, max_citing),
        "critique": True,
        "prefer_recent_years": 3,
        "prefer_survey": True,
    }


class CitationService:
    """find_citing_papers + get_citation_context."""

    def __init__(self) -> None:
        self.timeout = int(os.getenv("ACADEMIC_TIMEOUT", "20"))
        self.s2_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
        self._session = requests.Session()
        headers = {
            "User-Agent": "HelloAgents-ResearchGap/0.2",
            "Accept": "application/json",
        }
        if self.s2_key:
            headers["x-api-key"] = self.s2_key
        self._session.headers.update(headers)
        self.fulltext = FullTextService()
        self.max_contexts = int(os.getenv("MAX_CITATION_CONTEXTS", "4"))

    def find_citing_papers(
        self,
        *,
        paper_id: Optional[str] = None,
        title: Optional[str] = None,
        doi: Optional[str] = None,
        year: Optional[int] = None,
        max_results: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        age = paper_age_years(year)
        budget = citation_budget_for_age(age)
        limit = max_results or int(budget["max_citing"])

        s2_id = self._resolve_s2_id(paper_id=paper_id, title=title, doi=doi)
        papers: list[dict[str, Any]] = []
        if s2_id:
            papers = self._s2_citations(s2_id, limit=max(limit * 2, 10))
        if not papers and title:
            # soft fallback: search related newer works (not true citations)
            papers = self._related_search(title, limit=limit)
            for p in papers:
                p["citation_relation"] = "related_fallback_not_verified_citation"

        ranked = self._rank_citing(papers, age=age, budget=budget)
        return ranked[:limit]

    def get_citation_context(
        self,
        *,
        target_paper: dict[str, Any],
        citing_paper: dict[str, Any],
    ) -> Optional[dict[str, Any]]:
        """Extract citation context from citing paper fulltext; classify intent."""

        citing = self.fulltext.enrich_paper(citing_paper)
        if citing.get("fulltext_status") != "FULLTEXT":
            return {
                "target_paper_id": target_paper.get("paper_id"),
                "citing_paper_id": citing.get("paper_id"),
                "citing_paper_title": citing.get("title"),
                "citing_year": citing.get("year"),
                "citation_context": None,
                "critique_summary": None,
                "critique_type": "UNKNOWN",
                "source_section": None,
                "confidence": "low",
                "fulltext_status": citing.get("fulltext_status") or "FAILED",
                "notes": "citing paper fulltext unavailable; no external_critique emitted",
            }

        target_title = (target_paper.get("title") or "").strip()
        sections = citing.get("sections") or {}
        body = read_section_text(
            sections,
            "introduction",
            "related_work",
            "method",
            "discussion",
            "body",
            max_chars=12000,
        )
        contexts = _find_title_contexts(body, target_title, window=280)
        if not contexts:
            # try short author-year / keyword from title
            key = _title_keywords(target_title)
            contexts = _find_keyword_contexts(body, key, window=280) if key else []

        if not contexts:
            return {
                "target_paper_id": target_paper.get("paper_id"),
                "citing_paper_id": citing.get("paper_id"),
                "citing_paper_title": citing.get("title"),
                "citing_year": citing.get("year"),
                "citation_context": None,
                "critique_summary": None,
                "critique_type": "UNKNOWN",
                "source_section": "fulltext",
                "confidence": "low",
                "fulltext_status": "FULLTEXT",
                "notes": "no citation context span located",
            }

        context = contexts[0]
        intent = self._classify_intent(target_title, context, citing.get("title") or "")
        return {
            "target_paper_id": target_paper.get("paper_id"),
            "citing_paper_id": citing.get("paper_id"),
            "citing_paper_title": citing.get("title"),
            "citing_year": citing.get("year"),
            "citation_context": context[:500],
            "critique_summary": intent.get("critique_summary"),
            "critique_type": intent.get("critique_type") or "NEUTRAL",
            "source_section": intent.get("source_section") or "body",
            "confidence": intent.get("confidence") or "medium",
            "fulltext_status": "FULLTEXT",
        }

    def _classify_intent(self, target_title: str, context: str, citing_title: str) -> dict[str, Any]:
        # rule-first to avoid hallucinated critique
        lower = context.lower()
        critique_cues = [
            "however",
            "limitation",
            "limited",
            "fail",
            "unable",
            "only",
            "still",
            "does not",
            "cannot",
            "lack",
            "weak",
            "drawback",
            "require manual",
            "manually",
        ]
        positive_cues = ["inspired by", "based on", "extends", "improve upon", "following"]
        if any(c in lower for c in critique_cues):
            rule_type = "CRITIQUE"
        elif any(c in lower for c in positive_cues):
            rule_type = "EXTENSION"
        else:
            rule_type = "NEUTRAL"

        try:
            llm = build_llm(Configuration.from_env())
            raw = llm_chat(
                llm,
                system=(
                    "Classify citation intent of a citing paper toward a target paper. "
                    "Return STRICT JSON: "
                    "{critique_type: NEUTRAL|POSITIVE|EXTENSION|CRITIQUE, "
                    "critique_summary: string|null, confidence: high|medium|low, "
                    "source_section: string}. "
                    "critique_summary ONLY if critique_type=CRITIQUE and explicitly supported by the context. "
                    "Do NOT invent criticism."
                ),
                user=(
                    f"Target: {target_title}\nCiting: {citing_title}\n"
                    f"Context:\n{context}\nRule prior: {rule_type}"
                ),
            )
            data = extract_json(raw)
            if isinstance(data, dict) and data.get("critique_type") in {
                "NEUTRAL",
                "POSITIVE",
                "EXTENSION",
                "CRITIQUE",
            }:
                # refuse LLM upgrade to CRITIQUE without cues
                if data["critique_type"] == "CRITIQUE" and rule_type != "CRITIQUE":
                    data["critique_type"] = rule_type
                    data["critique_summary"] = None
                    data["confidence"] = "low"
                    data["notes"] = "downgraded: no explicit critique cue in context"
                return data
        except Exception as exc:  # noqa: BLE001
            logger.debug("citation intent LLM failed: %s", exc)

        summary = None
        if rule_type == "CRITIQUE":
            summary = context[:180]
        return {
            "critique_type": rule_type,
            "critique_summary": summary,
            "confidence": "low",
            "source_section": "body",
        }

    def _resolve_s2_id(
        self,
        *,
        paper_id: Optional[str],
        title: Optional[str],
        doi: Optional[str],
    ) -> Optional[str]:
        if paper_id and not str(paper_id).startswith("seed-"):
            # may already be S2 id
            if re.fullmatch(r"[a-f0-9]{40}", str(paper_id)):
                return str(paper_id)
        if doi:
            try:
                url = f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}"
                resp = self._session.get(
                    url, params={"fields": "paperId,title"}, timeout=self.timeout
                )
                if resp.status_code == 200:
                    return resp.json().get("paperId")
            except Exception as exc:  # noqa: BLE001
                logger.debug("s2 doi resolve failed: %s", exc)
        if title:
            try:
                resp = self._session.get(
                    "https://api.semanticscholar.org/graph/v1/paper/search",
                    params={"query": title, "limit": 1, "fields": "paperId,title"},
                    timeout=self.timeout,
                )
                if resp.status_code == 200:
                    data = (resp.json().get("data") or [])
                    if data:
                        return data[0].get("paperId")
            except Exception as exc:  # noqa: BLE001
                logger.debug("s2 title resolve failed: %s", exc)
        return None

    def _s2_citations(self, paper_id: str, *, limit: int) -> list[dict[str, Any]]:
        url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}/citations"
        fields = (
            "citingPaper.paperId,citingPaper.title,citingPaper.abstract,"
            "citingPaper.year,citingPaper.citationCount,citingPaper.url,"
            "citingPaper.externalIds,citingPaper.openAccessPdf"
        )
        try:
            resp = self._session.get(
                url,
                params={"fields": fields, "limit": min(limit, 50)},
                timeout=self.timeout,
            )
            if resp.status_code != 200:
                logger.warning("s2 citations %s: %s", resp.status_code, resp.text[:200])
                return []
            out = []
            for row in resp.json().get("data") or []:
                p = row.get("citingPaper") or {}
                if not p.get("title"):
                    continue
                external = p.get("externalIds") or {}
                oa = p.get("openAccessPdf") or {}
                out.append(
                    {
                        "paper_id": p.get("paperId"),
                        "title": p.get("title"),
                        "year": p.get("year"),
                        "abstract": p.get("abstract") or "",
                        "content": p.get("abstract") or "",
                        "citation_count": p.get("citationCount") or 0,
                        "url": p.get("url"),
                        "doi": external.get("DOI"),
                        "arxiv_id": external.get("ArXiv"),
                        "pdf_url": oa.get("url"),
                        "source": "semantic_scholar_citation",
                        "citation_relation": "cites",
                        "fulltext_status": "ABSTRACT_ONLY",
                    }
                )
            return out
        except Exception as exc:  # noqa: BLE001
            logger.warning("s2 citations failed: %s", exc)
            return []

    def _related_search(self, title: str, *, limit: int) -> list[dict[str, Any]]:
        try:
            from services.academic_search import AcademicSearchService

            svc = AcademicSearchService(max_results=limit, include_citing=False)
            # short keyword query
            q = " ".join(re.findall(r"[A-Za-z0-9\-]+", title)[:8])
            payload = svc.search(q, backend="academic", max_results=limit)
            return payload.get("results") or []
        except Exception as exc:  # noqa: BLE001
            logger.debug("related search failed: %s", exc)
            return []

    def _rank_citing(
        self, papers: list[dict[str, Any]], *, age: Optional[int], budget: dict[str, Any]
    ) -> list[dict[str, Any]]:
        now = datetime.now().year
        prefer_recent = int(budget.get("prefer_recent_years") or 0)

        def score(p: dict[str, Any]) -> float:
            s = float(p.get("citation_count") or 0) ** 0.5
            y = p.get("year") or 0
            if prefer_recent and y and (now - int(y)) <= prefer_recent:
                s += 5
            title = (p.get("title") or "").lower()
            if budget.get("prefer_survey") and any(
                k in title for k in ("survey", "review", "systematic")
            ):
                s += 4
            if p.get("pdf_url") or p.get("arxiv_id"):
                s += 2
            if p.get("citation_relation") == "cites":
                s += 3
            return s

        return sorted(papers, key=score, reverse=True)


def _title_keywords(title: str) -> str:
    stop = {"the", "a", "an", "of", "for", "and", "in", "on", "to", "via"}
    toks = [t for t in re.findall(r"[A-Za-z][A-Za-z0-9\-]{3,}", title) if t.lower() not in stop]
    # prefer distinctive token (often system name)
    for t in toks:
        if t[0].isupper() and t.lower() not in stop:
            return t
    return " ".join(toks[:2])


def _find_title_contexts(text: str, title: str, *, window: int) -> list[str]:
    if not text or not title:
        return []
    # try full title then shortened
    candidates = [title]
    short = re.sub(r"[:\-].*$", "", title).strip()
    if short and short != title:
        candidates.append(short)
    key = _title_keywords(title)
    if key:
        candidates.append(key)
    out: list[str] = []
    lower = text.lower()
    for cand in candidates:
        idx = lower.find(cand.lower())
        if idx < 0:
            continue
        start = max(0, idx - window)
        end = min(len(text), idx + len(cand) + window)
        out.append(text[start:end].strip())
        if len(out) >= 3:
            break
    return out


def _find_keyword_contexts(text: str, key: str, *, window: int) -> list[str]:
    return _find_title_contexts(text, key, window=window)
