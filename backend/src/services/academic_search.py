"""Academic paper search via Semantic Scholar and OpenAlex."""

from __future__ import annotations

import logging
import os
import re
import time
from typing import Any, Optional

import requests

logger = logging.getLogger(__name__)

SEMANTIC_SCHOLAR_SEARCH = "https://api.semanticscholar.org/graph/v1/paper/search"
SEMANTIC_SCHOLAR_PAPER = "https://api.semanticscholar.org/graph/v1/paper"
OPENALEX_WORKS = "https://api.openalex.org/works"

S2_FIELDS = (
    "paperId,title,abstract,year,citationCount,authors,url,venue,"
    "externalIds,publicationTypes,isOpenAccess,openAccessPdf"
)
S2_CITATION_FIELDS = "citingPaper.paperId,citingPaper.title,citingPaper.year,citingPaper.citationCount"


def sanitize_academic_query(query: str, *, max_len: int = 120) -> str:
    """Turn NL questions into short keyword queries safe for OpenAlex/S2."""

    q = (query or "").strip()
    if not q:
        return "cross-chain bridge smart contract vulnerability"

    # Drop question wrappers that blow up OpenAlex / waste tokens
    q = re.sub(r"^[?\s]*(Does|Do|Is|Are|Can|How|What|Which|Why)\b[\s:]*", "", q, flags=re.I)
    q = q.replace("?", " ")
    q = re.sub(r"[()'\"“”‘’]", " ", q)
    q = re.sub(r"\s+", " ", q).strip()
    if len(q) > max_len:
        q = q[:max_len].rsplit(" ", 1)[0] or q[:max_len]
    return q or "cross-chain bridge smart contract vulnerability"


class AcademicSearchService:
    """Keyword search for scholarly papers with abstract-first evidence."""

    def __init__(
        self,
        *,
        semantic_scholar_api_key: Optional[str] = None,
        openalex_mailto: Optional[str] = None,
        timeout: int = 30,
        max_results: int = 10,
        include_citing: bool = True,
        citing_limit: int = 5,
    ) -> None:
        self.semantic_scholar_api_key = semantic_scholar_api_key
        self.openalex_mailto = openalex_mailto or "research-gap-agent@local.dev"
        self.timeout = int(os.getenv("ACADEMIC_TIMEOUT", str(timeout)))
        self.max_retries = int(os.getenv("ACADEMIC_MAX_RETRIES", "2"))
        self.max_results = max_results
        self.include_citing = include_citing
        self.citing_limit = citing_limit
        self._session = requests.Session()
        self._session.headers.update(
            {
                "User-Agent": "HelloAgents-ResearchGap/0.1 (academic search)",
                "Accept": "application/json",
            }
        )

    def search(
        self,
        query: str,
        *,
        backend: str = "semantic_scholar",
        max_results: Optional[int] = None,
    ) -> dict[str, Any]:
        """Search papers; supports offline seed-first mode for demos."""

        limit = max_results or self.max_results
        notices: list[str] = []
        papers: list[dict[str, Any]] = []
        resolved = backend
        query = sanitize_academic_query(query)

        force_seed = os.getenv("FORCE_SEED_CORPUS", "false").lower() in {
            "1",
            "true",
            "yes",
        }
        academic_offline = os.getenv("ACADEMIC_OFFLINE", "false").lower() in {
            "1",
            "true",
            "yes",
        }
        allow_seed = os.getenv("ALLOW_SEED_CORPUS_FALLBACK", "true").lower() in {
            "1",
            "true",
            "yes",
        }

        if force_seed or academic_offline:
            from gap_discovery.seed_corpus import seed_search

            papers = seed_search(query, limit=limit)
            return {
                "results": papers,
                "backend": "seed_corpus",
                "answer": None,
                "notices": ["ACADEMIC_OFFLINE/FORCE_SEED_CORPUS：跳过在线 API，使用本地种子论文库"],
                "query": query,
            }

        primary = os.getenv("ACADEMIC_PRIMARY", "openalex").strip().lower()
        order: list[str]
        if backend == "semantic_scholar":
            order = ["semantic_scholar", "openalex"]
        elif backend == "openalex":
            order = ["openalex"]
        elif primary == "semantic_scholar":
            order = ["semantic_scholar", "openalex"]
        else:
            order = ["openalex", "semantic_scholar"]

        for name in order:
            if papers:
                break
            for attempt in range(1, self.max_retries + 1):
                try:
                    if name == "semantic_scholar":
                        papers = self.search_semantic_scholar(query, limit=limit)
                        resolved = "semantic_scholar"
                    else:
                        papers = self.search_openalex(query, limit=limit)
                        resolved = "openalex"
                    break
                except Exception as exc:
                    logger.warning("%s search failed (attempt %s): %s", name, attempt, exc)
                    if attempt >= self.max_retries:
                        notices.append(f"{name} 不可用: {exc}")
                    else:
                        time.sleep(0.4 * attempt)

        if not papers and allow_seed:
            from gap_discovery.seed_corpus import seed_search

            papers = seed_search(query, limit=limit)
            resolved = "seed_corpus"
            notices.append(
                "在线学术 API 不可用或无结果，已启用本地 seed_corpus 兜底（面试/演示可复现）"
            )

        if not papers:
            notices.append("未检索到相关学术论文")

        return {
            "results": papers,
            "backend": resolved,
            "answer": None,
            "notices": notices,
            "query": query,
        }

    def search_semantic_scholar(self, query: str, *, limit: int = 10) -> list[dict[str, Any]]:
        headers = {}
        if self.semantic_scholar_api_key:
            headers["x-api-key"] = self.semantic_scholar_api_key

        params = {
            "query": query,
            "limit": min(limit, 100),
            "fields": S2_FIELDS,
        }
        response = self._session.get(
            SEMANTIC_SCHOLAR_SEARCH,
            params=params,
            headers=headers,
            timeout=self.timeout,
        )
        if response.status_code == 429:
            time.sleep(1.5)
            response = self._session.get(
                SEMANTIC_SCHOLAR_SEARCH,
                params=params,
                headers=headers,
                timeout=self.timeout,
            )
        response.raise_for_status()
        data = response.json()
        papers: list[dict[str, Any]] = []
        for item in data.get("data") or []:
            paper = self._normalize_semantic_scholar(item)
            if self.include_citing and paper.get("paper_id"):
                paper["citing_papers"] = self._fetch_s2_citations(paper["paper_id"])
            papers.append(paper)
        return papers

    def search_openalex(self, query: str, *, limit: int = 10) -> list[dict[str, Any]]:
        params = {
            "search": query,
            "per_page": min(limit, 50),
            "mailto": self.openalex_mailto,
            "sort": "relevance_score:desc",
        }
        response = self._session.get(OPENALEX_WORKS, params=params, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()
        papers: list[dict[str, Any]] = []
        for item in data.get("results") or []:
            paper = self._normalize_openalex(item)
            if self.include_citing and paper.get("openalex_id"):
                paper["citing_papers"] = self._fetch_openalex_citations(paper["openalex_id"])
            papers.append(paper)
        return papers

    def _fetch_s2_citations(self, paper_id: str) -> list[dict[str, Any]]:
        headers = {}
        if self.semantic_scholar_api_key:
            headers["x-api-key"] = self.semantic_scholar_api_key
        try:
            response = self._session.get(
                f"{SEMANTIC_SCHOLAR_PAPER}/{paper_id}/citations",
                params={"fields": S2_CITATION_FIELDS, "limit": self.citing_limit},
                headers=headers,
                timeout=self.timeout,
            )
            if response.status_code == 429:
                time.sleep(1.2)
                response = self._session.get(
                    f"{SEMANTIC_SCHOLAR_PAPER}/{paper_id}/citations",
                    params={"fields": S2_CITATION_FIELDS, "limit": self.citing_limit},
                    headers=headers,
                    timeout=self.timeout,
                )
            if not response.ok:
                return []
            citing: list[dict[str, Any]] = []
            for row in response.json().get("data") or []:
                citing_paper = row.get("citingPaper") or {}
                if not citing_paper.get("title"):
                    continue
                citing.append(
                    {
                        "paper_id": citing_paper.get("paperId"),
                        "title": citing_paper.get("title"),
                        "year": citing_paper.get("year"),
                        "citation_count": citing_paper.get("citationCount"),
                    }
                )
            return citing
        except Exception as exc:  # pragma: no cover - best effort
            logger.debug("Fetch S2 citations failed for %s: %s", paper_id, exc)
            return []

    def _fetch_openalex_citations(self, openalex_id: str) -> list[dict[str, Any]]:
        """Fetch papers that cite this work (OpenAlex cited_by)."""
        try:
            # openalex_id may be full URL or W123 style
            work_id = openalex_id.rstrip("/").split("/")[-1]
            params = {
                "filter": f"cites:{work_id}",
                "per_page": self.citing_limit,
                "mailto": self.openalex_mailto,
                "sort": "cited_by_count:desc",
            }
            response = self._session.get(OPENALEX_WORKS, params=params, timeout=self.timeout)
            if not response.ok:
                return []
            citing: list[dict[str, Any]] = []
            for item in response.json().get("results") or []:
                citing.append(
                    {
                        "paper_id": item.get("id"),
                        "title": item.get("display_name") or item.get("title"),
                        "year": item.get("publication_year"),
                        "citation_count": item.get("cited_by_count"),
                    }
                )
            return citing
        except Exception as exc:  # pragma: no cover
            logger.debug("Fetch OpenAlex citations failed for %s: %s", openalex_id, exc)
            return []

    def _normalize_semantic_scholar(self, item: dict[str, Any]) -> dict[str, Any]:
        external = item.get("externalIds") or {}
        authors = [a.get("name") for a in (item.get("authors") or []) if a.get("name")]
        paper_id = item.get("paperId")
        url = (
            item.get("url")
            or (f"https://www.semanticscholar.org/paper/{paper_id}" if paper_id else None)
            or (f"https://arxiv.org/abs/{external.get('ArXiv')}" if external.get("ArXiv") else None)
            or (f"https://doi.org/{external.get('DOI')}" if external.get("DOI") else "")
        )
        abstract = (item.get("abstract") or "").strip()
        pdf_url = None
        open_access = item.get("openAccessPdf") or {}
        if isinstance(open_access, dict):
            pdf_url = open_access.get("url")

        evidence_level = "abstract_only"
        # 当前未拉取全文 PDF，统一标记 abstract_only；有 PDF 链接时仅记录可获取性
        return self._build_paper_record(
            paper_id=paper_id,
            title=item.get("title") or "Untitled",
            abstract=abstract,
            year=item.get("year"),
            citation_count=item.get("citationCount") or 0,
            authors=authors,
            venue=item.get("venue"),
            url=url or "",
            doi=external.get("DOI"),
            arxiv_id=external.get("ArXiv"),
            source="semantic_scholar",
            evidence_level=evidence_level,
            pdf_url=pdf_url,
            openalex_id=None,
        )

    def _normalize_openalex(self, item: dict[str, Any]) -> dict[str, Any]:
        authorships = item.get("authorships") or []
        authors = []
        for authorship in authorships:
            author = authorship.get("author") or {}
            name = author.get("display_name")
            if name:
                authors.append(name)

        ids = item.get("ids") or {}
        doi = None
        if ids.get("doi"):
            doi = ids["doi"].replace("https://doi.org/", "")
        arxiv_id = None
        for loc in item.get("locations") or []:
            landing = loc.get("landing_page_url") or ""
            if "arxiv.org" in landing:
                arxiv_id = landing.rstrip("/").split("/")[-1]
                break

        abstract = self._rebuild_openalex_abstract(item.get("abstract_inverted_index"))
        primary = item.get("primary_location") or {}
        url = (
            ids.get("doi")
            or primary.get("landing_page_url")
            or item.get("id")
            or ""
        )
        pdf_url = primary.get("pdf_url")
        openalex_id = item.get("id")

        return self._build_paper_record(
            paper_id=openalex_id,
            title=item.get("display_name") or item.get("title") or "Untitled",
            abstract=abstract,
            year=item.get("publication_year"),
            citation_count=item.get("cited_by_count") or 0,
            authors=authors,
            venue=((item.get("primary_location") or {}).get("source") or {}).get("display_name"),
            url=url,
            doi=doi,
            arxiv_id=arxiv_id,
            source="openalex",
            evidence_level="abstract_only",
            pdf_url=pdf_url,
            openalex_id=openalex_id,
        )

    @staticmethod
    def _rebuild_openalex_abstract(inverted: Optional[dict[str, list[int]]]) -> str:
        if not inverted:
            return ""
        positions: list[tuple[int, str]] = []
        for token, idxs in inverted.items():
            for idx in idxs:
                positions.append((idx, token))
        positions.sort(key=lambda x: x[0])
        return " ".join(token for _, token in positions)

    def _build_paper_record(
        self,
        *,
        paper_id: Optional[str],
        title: str,
        abstract: str,
        year: Optional[int],
        citation_count: int,
        authors: list[str],
        venue: Optional[str],
        url: str,
        doi: Optional[str],
        arxiv_id: Optional[str],
        source: str,
        evidence_level: str,
        pdf_url: Optional[str],
        openalex_id: Optional[str],
    ) -> dict[str, Any]:
        content = abstract or "（无摘要，仅有题录信息）"
        meta_lines = [
            f"Title: {title}",
            f"Year: {year or 'N/A'}",
            f"Citations: {citation_count}",
            f"Authors: {', '.join(authors[:12]) if authors else 'N/A'}",
            f"Venue: {venue or 'N/A'}",
            f"DOI: {doi or 'N/A'}",
            f"arXiv: {arxiv_id or 'N/A'}",
            f"Evidence level: {evidence_level}",
            f"Source: {source}",
            "",
            "Abstract:",
            content,
        ]
        if pdf_url:
            meta_lines.extend(["", f"PDF URL (not fetched): {pdf_url}"])

        return {
            # 兼容原 SearchTool 结果格式
            "title": title,
            "url": url,
            "content": content,
            "raw_content": "\n".join(meta_lines),
            # 学术字段
            "paper_id": paper_id,
            "openalex_id": openalex_id,
            "year": year,
            "citation_count": citation_count,
            "authors": authors,
            "venue": venue,
            "doi": doi,
            "arxiv_id": arxiv_id,
            "evidence_level": evidence_level,
            "pdf_url": pdf_url,
            "source": source,
            "citing_papers": [],
        }


def format_academic_sources(search_results: dict[str, Any] | None) -> str:
    """Bullet summary tailored for paper metadata."""

    if not search_results:
        return ""

    lines: list[str] = []
    for item in search_results.get("results") or []:
        title = item.get("title") or "Untitled"
        year = item.get("year") or "?"
        cites = item.get("citation_count", 0)
        evidence = item.get("evidence_level", "abstract_only")
        url = item.get("url") or ""
        lines.append(f"* [{year}] {title} (citations={cites}, {evidence}) : {url}")
        citing = item.get("citing_papers") or []
        if citing:
            preview = "; ".join(
                f"{c.get('year') or '?'}: {c.get('title')}" for c in citing[:3] if c.get("title")
            )
            lines.append(f"  - cited by: {preview}")
    return "\n".join(lines)
