"""PDF text extraction + section-aware chunking for Paper RAG."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

SECTION_ALIASES: list[tuple[str, tuple[str, ...]]] = [
    ("abstract", ("abstract",)),
    (
        "introduction",
        ("introduction", "1 introduction", "i. introduction"),
    ),
    (
        "related_work",
        ("related work", "related works", "background", "2 related"),
    ),
    (
        "method",
        (
            "method",
            "methods",
            "approach",
            "methodology",
            "proposed method",
            "system design",
            "design",
        ),
    ),
    (
        "evaluation",
        (
            "evaluation",
            "experiment",
            "experiments",
            "experimental setup",
            "results",
            "implementation",
        ),
    ),
    (
        "discussion",
        ("discussion", "analysis"),
    ),
    (
        "limitations",
        ("limitation", "limitations", "threats to validity", "threats to validity"),
    ),
    (
        "threats_to_validity",
        ("threats to validity", "threats to the validity"),
    ),
    (
        "conclusion",
        ("conclusion", "conclusions", "concluding remarks"),
    ),
    (
        "future_work",
        ("future work", "future directions", "open challenges"),
    ),
    (
        "references",
        ("references", "bibliography"),
    ),
]


def extract_pdf_text(pdf_path: str | Path) -> str:
    """Extract plain text from a local PDF file."""

    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    parts: list[str] = []
    for page in reader.pages:
        try:
            parts.append(page.extract_text() or "")
        except Exception as exc:  # noqa: BLE001
            logger.debug("page extract failed: %s", exc)
    text = "\n".join(parts)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_sections(full_text: str) -> dict[str, str]:
    """Heuristic section splitter; unknown blocks go to 'body'."""

    if not full_text:
        return {}

    lines = [ln.strip() for ln in full_text.splitlines()]
    headings: list[tuple[int, str]] = []
    for i, line in enumerate(lines):
        if not line or len(line) > 120:
            continue
        canon = _match_section(line)
        if canon:
            headings.append((i, canon))

    if not headings:
        # semantic/rule fallback: treat whole text as body + try abstract cue
        sections = {"body": full_text}
        m = re.search(
            r"(?is)\babstract\b[:\s]*(.{200,2500}?)(?=\b(?:1\.|introduction|keywords)\b)",
            full_text,
        )
        if m:
            sections["abstract"] = m.group(1).strip()
        # limitation cue anywhere
        for cue, key in [
            (r"(?is)\blimitations?\b[:\s]*(.{120,2000}?)(?=\n\s*\n|\bconclusion|\breferences\b)", "limitations"),
            (r"(?is)\bfuture work\b[:\s]*(.{80,1500}?)(?=\n\s*\n|\breferences\b)", "future_work"),
        ]:
            mm = re.search(cue, full_text)
            if mm:
                sections[key] = mm.group(1).strip()
        return sections

    sections: dict[str, str] = {}
    for idx, (start, name) in enumerate(headings):
        end = headings[idx + 1][0] if idx + 1 < len(headings) else len(lines)
        body = "\n".join(lines[start + 1 : end]).strip()
        if not body:
            continue
        if name in sections:
            sections[name] = sections[name] + "\n" + body
        else:
            sections[name] = body
    return sections


def chunk_sections(
    sections: dict[str, str],
    *,
    paper_id: str,
    title: str,
    year: Optional[int],
    max_chars: int = 1200,
) -> list[dict[str, Any]]:
    """Turn sections into RAG chunks with metadata."""

    chunks: list[dict[str, Any]] = []
    for section, text in sections.items():
        if section == "references":
            # keep short refs index only
            text = text[:2000]
        pieces = _window(text, max_chars=max_chars)
        for i, piece in enumerate(pieces):
            chunks.append(
                {
                    "chunk_id": f"{paper_id}:{section}:{i}",
                    "paper_id": paper_id,
                    "title": title,
                    "year": year,
                    "section": section,
                    "chunk_index": i,
                    "source": "fulltext",
                    "text": piece,
                }
            )
    return chunks


def read_section_text(sections: dict[str, str], *names: str, max_chars: int = 3500) -> str:
    parts = []
    for name in names:
        if sections.get(name):
            parts.append(sections[name])
    text = "\n\n".join(parts).strip()
    return text[:max_chars]


def _match_section(line: str) -> Optional[str]:
    cleaned = re.sub(r"^[\dIVXivx\.\)\-\s]+", "", line).strip().lower()
    cleaned = re.sub(r"[^a-z\s]", "", cleaned).strip()
    if not cleaned or len(cleaned.split()) > 8:
        return None
    for canon, aliases in SECTION_ALIASES:
        for alias in aliases:
            if cleaned == alias or cleaned.startswith(alias + " "):
                return canon
            # numbered: "3 method"
            if cleaned.endswith(alias) and len(cleaned) <= len(alias) + 4:
                return canon
    return None


def _window(text: str, *, max_chars: int) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    out: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        if end < len(text):
            cut = text.rfind(". ", start, end)
            if cut > start + max_chars // 2:
                end = cut + 1
        out.append(text[start:end].strip())
        start = end
    return [x for x in out if x]
