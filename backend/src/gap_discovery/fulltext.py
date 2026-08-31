"""Open-access fulltext fetch (no paywall bypass)."""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any, Optional

import requests

from gap_discovery.pdf_parser import chunk_sections, extract_pdf_text, split_sections

logger = logging.getLogger(__name__)

FulltextStatus = str  # FULLTEXT | ABSTRACT_ONLY | FAILED


class FullTextService:
    """Resolve OA PDF → download → parse → section chunks."""

    def __init__(self, cache_dir: Optional[str | Path] = None) -> None:
        root = Path(__file__).resolve().parents[2]
        self.cache_dir = Path(
            cache_dir or os.getenv("FULLTEXT_CACHE_DIR", str(root / "workspace" / "pdfs"))
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = int(os.getenv("FULLTEXT_TIMEOUT", "45"))
        self.max_papers = int(os.getenv("MAX_FULLTEXT_PAPERS", "6"))
        self._session = requests.Session()
        self._session.headers.update(
            {
                "User-Agent": "HelloAgents-ResearchGap/0.2 (OA fulltext; mailto=research-gap-agent@local.dev)",
                "Accept": "application/pdf,*/*",
            }
        )

    def enrich_paper(self, paper: dict[str, Any]) -> dict[str, Any]:
        """Attach fulltext fields onto a paper dict (mutates copy)."""

        out = dict(paper)
        if out.get("fulltext_status") == "FULLTEXT" and out.get("sections"):
            return out

        pdf_url, source = self.resolve_pdf_url(out)
        out["pdf_url"] = pdf_url
        out["fulltext_source"] = source
        if not pdf_url:
            out["fulltext_status"] = "ABSTRACT_ONLY"
            out["evidence_level"] = out.get("evidence_level") or "abstract_only"
            return out

        try:
            path = self.download_pdf(pdf_url, paper_id=str(out.get("paper_id") or "paper"))
            text = extract_pdf_text(path)
            if len(text) < 400:
                raise ValueError("extracted text too short")
            sections = split_sections(text)
            chunks = chunk_sections(
                sections,
                paper_id=str(out.get("paper_id")),
                title=str(out.get("title") or ""),
                year=out.get("year"),
            )
            out["fulltext_status"] = "FULLTEXT"
            out["fulltext_path"] = str(path)
            out["sections"] = {k: v[:8000] for k, v in sections.items()}
            out["fulltext_chunks"] = chunks
            out["evidence_level"] = "full_text"
        except Exception as exc:  # noqa: BLE001
            logger.warning("fulltext failed for %s: %s", out.get("title"), exc)
            out["fulltext_status"] = "FAILED"
            out["fulltext_error"] = str(exc)
            out["evidence_level"] = out.get("evidence_level") or "abstract_only"
        return out

    def resolve_pdf_url(self, paper: dict[str, Any]) -> tuple[Optional[str], Optional[str]]:
        """Priority: known arXiv → openAccessPdf → DOI OA landing (best-effort, no paywall)."""

        if paper.get("pdf_url") and _looks_like_pdf_url(str(paper["pdf_url"])):
            return str(paper["pdf_url"]), "provided_pdf_url"

        arxiv_id = paper.get("arxiv_id") or _extract_arxiv(paper.get("url") or "")
        if arxiv_id:
            aid = str(arxiv_id).replace("arXiv:", "").strip()
            return f"https://arxiv.org/pdf/{aid}.pdf", "arxiv"

        # Semantic Scholar / OpenAlex OA pdf field
        for key in ("open_access_pdf", "oa_pdf_url"):
            if paper.get(key):
                return str(paper[key]), key

        pdf_url = paper.get("pdf_url")
        if pdf_url and "arxiv.org" in str(pdf_url):
            return str(pdf_url), "arxiv_url"

        # Unpaywall-style: only if explicitly enabled (still OA-only)
        doi = paper.get("doi")
        if doi and os.getenv("ENABLE_UNPAYWALL", "false").lower() in {"1", "true", "yes"}:
            oa = self._unpaywall_pdf(str(doi))
            if oa:
                return oa, "unpaywall"

        return None, None

    def download_pdf(self, url: str, *, paper_id: str) -> Path:
        safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", paper_id)[:80]
        path = self.cache_dir / f"{safe}.pdf"
        if path.exists() and path.stat().st_size > 1000:
            return path
        resp = self._session.get(url, timeout=self.timeout, allow_redirects=True)
        resp.raise_for_status()
        ctype = (resp.headers.get("Content-Type") or "").lower()
        if "pdf" not in ctype and not url.lower().endswith(".pdf"):
            # some mirrors still return PDF bytes
            if not resp.content.startswith(b"%PDF"):
                raise ValueError(f"not a PDF content-type={ctype}")
        path.write_bytes(resp.content)
        return path

    def _unpaywall_pdf(self, doi: str) -> Optional[str]:
        email = os.getenv("OPENALEX_MAILTO") or "research-gap-agent@local.dev"
        url = f"https://api.unpaywall.org/v2/{doi}"
        try:
            resp = self._session.get(url, params={"email": email}, timeout=20)
            if resp.status_code != 200:
                return None
            data = resp.json()
            loc = data.get("best_oa_location") or {}
            pdf = loc.get("url_for_pdf") or loc.get("url")
            if pdf and _looks_like_pdf_url(pdf):
                return pdf
        except Exception as exc:  # noqa: BLE001
            logger.debug("unpaywall failed: %s", exc)
        return None


def _extract_arxiv(url: str) -> Optional[str]:
    m = re.search(r"arxiv\.org/(?:abs|pdf)/([0-9]+\.[0-9]+)(v\d+)?", url or "")
    if m:
        return m.group(1)
    return None


def _looks_like_pdf_url(url: str) -> bool:
    u = url.lower()
    return u.endswith(".pdf") or "/pdf/" in u or "arxiv.org/pdf" in u
