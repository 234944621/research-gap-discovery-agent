"""Topic query mapping and paper relevance filtering."""

from __future__ import annotations

import re
from typing import Any

_TOPIC_QUERY_MAP = [
    (
        ["跨链", "cross-chain", "cross chain", "bridge"],
        "cross-chain bridge smart contract vulnerability detection security",
        ["cross-chain", "bridge", "smart contract", "blockchain", "vulnerability", "security", "solidity", "defi", "合约", "跨链", "漏洞"],
    ),
    (
        ["智能合约", "smart contract"],
        "smart contract vulnerability detection static analysis",
        ["smart contract", "vulnerability", "solidity", "ethereum", "blockchain", "合约", "漏洞"],
    ),
    (
        ["区块链", "blockchain"],
        "blockchain security smart contract vulnerability",
        ["blockchain", "security", "smart contract", "vulnerability", "区块链"],
    ),
]


def _to_academic_query(topic: str) -> str:
    """Map Chinese/short topics to English academic search queries."""

    lower = topic.lower()
    for keys, query, _ in _TOPIC_QUERY_MAP:
        if any(k.lower() in lower for k in keys):
            return query
    # keep original if already latin-heavy
    latin_ratio = sum(ch.isascii() and ch.isalpha() for ch in topic) / max(len(topic), 1)
    if latin_ratio > 0.5:
        return topic
    return f"{topic} research survey"


def _topic_relevance_terms(topic: str) -> list[str]:
    lower = topic.lower()
    for keys, _, terms in _TOPIC_QUERY_MAP:
        if any(k.lower() in lower for k in keys):
            return terms
    return _extract_keywords(topic)


def _core_topic_terms(topic: str) -> list[str]:
    """Strong anchors that must usually appear for a paper to stay in-scope."""

    lower = topic.lower()
    if any(k in lower for k in ["跨链", "cross-chain", "cross chain", "bridge"]):
        return ["cross-chain", "bridge", "smart contract", "solidity", "ethereum", "defi", "跨链", "合约"]
    if any(k in lower for k in ["智能合约", "smart contract"]):
        return ["smart contract", "solidity", "ethereum", "vulnerability", "合约", "漏洞"]
    if any(k in lower for k in ["区块链", "blockchain"]):
        return ["blockchain", "smart contract", "blockchain security", "区块链"]
    return _topic_relevance_terms(topic)[:4]


def _paper_blob(paper: dict[str, Any]) -> str:
    return " ".join(
        [
            str(paper.get("title") or ""),
            str(paper.get("content") or ""),
            str(paper.get("venue") or ""),
        ]
    ).lower()


def _relevance_score(paper: dict[str, Any], topic: str) -> int:
    blob = _paper_blob(paper)
    if _is_hard_noise(blob, topic):
        return 0

    terms = _topic_relevance_terms(topic)
    score = 0
    for term in terms:
        if term.lower() in blob:
            # 宽泛词权重更低，避免只靠 security 命中
            if term.lower() in {"security", "安全", "research", "survey"}:
                score += 1
            else:
                score += 2
    # 必须至少命中一个核心锚点，否则大幅降权
    if not any(t.lower() in blob for t in _core_topic_terms(topic)):
        return 0
    return score


def _is_hard_noise(blob: str, topic: str) -> bool:
    """Reject clearly off-domain hits that share the word 'security'."""

    noise_markers = [
        "o-ran",
        "oran",
        "industrial control",
        "ics security",
        "network slicing",
        "smart transportation",
        "smart factory",
        "explainable artificial intelligence applications in cyber security",
        "beeswax",
        "object detection",
        "canny",
        "recombinase",
        "parking demand",
    ]
    if any(m in blob for m in noise_markers):
        # 跨链主题下直接剔除；其他主题仅当不含核心锚点时剔除
        lower = topic.lower()
        if any(k in lower for k in ["跨链", "bridge", "smart contract", "智能合约", "blockchain", "区块链"]):
            return True
        if not any(t.lower() in blob for t in _core_topic_terms(topic)):
            return True
    return False


def _filter_relevant_papers(
    papers: list[dict[str, Any]],
    topic: str,
    *,
    min_score: int = 1,
) -> list[dict[str, Any]]:
    # 跨链/合约主题默认要求更高分，过滤 O-RAN / ICS 等噪声
    lower = topic.lower()
    effective_min = min_score
    if any(k in lower for k in ["跨链", "bridge", "smart contract", "智能合约", "blockchain", "区块链"]):
        effective_min = max(min_score, 3)

    scored = [(_relevance_score(p, topic), p) for p in papers]
    filtered = [p for s, p in scored if s >= effective_min]
    filtered.sort(
        key=lambda p: (_relevance_score(p, topic), p.get("citation_count") or 0),
        reverse=True,
    )
    return filtered


def _extract_keywords(text: str) -> list[str]:
    # Prefer known domain terms, then token fragments
    domain = _topic_relevance_terms(text)
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}|[\u4e00-\u9fff]{2,}", text.lower())
    stop = {
        "the", "and", "for", "with", "from", "into", "research", "survey",
        "研究", "基于", "方法", "分析", "检测", "detection", "method", "methods",
    }
    keywords = [t for t in tokens if t not in stop]
    seen: set[str] = set()
    ordered: list[str] = []
    for k in domain + keywords:
        if k not in seen:
            seen.add(k)
            ordered.append(k)
    return ordered[:16]


# Public aliases used by pipeline / other modules
to_academic_query = _to_academic_query
filter_relevant_papers = _filter_relevant_papers

