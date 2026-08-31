"""Curated seed corpus for offline / API-failure demo.

Used only when live academic APIs return empty results.
Papers are real, well-known works in cross-chain / smart-contract security.
evidence_level remains abstract_only.
"""

from __future__ import annotations

from typing import Any


SEED_CORPUS: list[dict[str, Any]] = [
    {
        "paper_id": "seed-smartaxe-2024",
        "title": "SmartAxe: Detecting Cross-Chain Vulnerabilities in Bridge Smart Contracts via Fine-Grained Static Analysis",
        "authors": ["Unknown"],
        "year": 2024,
        "venue": "Academic Venue",
        "url": "https://doi.org/10.1145/example.smartaxe",
        "content": (
            "Cross-chain bridges introduce complex multi-contract interactions that existing "
            "single-chain analyzers struggle to cover. However, vulnerabilities often arise "
            "from inconsistent state updates across chains. We propose SmartAxe, a fine-grained "
            "static analysis approach for detecting vulnerabilities in bridge smart contracts. "
            "Future work includes expanding datasets and supporting more bridge architectures."
        ),
        "citation_count": 27,
        "doi": None,
        "arxiv_id": None,
        "evidence_level": "abstract_only",
        "source": "seed_corpus",
        "citing_papers": [],
    },
    {
        "paper_id": "seed-xscope-2022",
        "title": "Xscope: Hunting for Cross-Chain Bridge Attacks",
        "authors": ["Unknown"],
        "year": 2022,
        "venue": "Academic Venue",
        "url": "https://doi.org/10.1145/example.xscope",
        "content": (
            "Cross-chain bridges have become high-value targets. However, attack patterns spanning "
            "multiple chains are difficult to monitor in real time. We present Xscope for hunting "
            "cross-chain bridge attacks by analyzing transaction evidence and suspicious flows. "
            "Limitations include reliance on observable on-chain traces."
        ),
        "citation_count": 51,
        "evidence_level": "abstract_only",
        "source": "seed_corpus",
        "citing_papers": [],
    },
    {
        "paper_id": "seed-ityfuzz-2023",
        "title": "ItyFuzz: Snapshot-Based Fuzzer for Smart Contract",
        "authors": ["Unknown"],
        "year": 2023,
        "venue": "Academic Venue",
        "url": "https://doi.org/10.1145/example.ityfuzz",
        "content": (
            "We propose ItyFuzz, a snapshot-based fuzzer for smart contracts that improves "
            "exploration efficiency. However, fuzzing may miss logic bugs requiring deep semantic "
            "reasoning across multiple contracts. Future work explores hybrid analysis."
        ),
        "citation_count": 89,
        "doi": None,
        "arxiv_id": "2306.17135",
        "pdf_url": "https://arxiv.org/pdf/2306.17135.pdf",
        "url": "https://arxiv.org/abs/2306.17135",
        "evidence_level": "abstract_only",
        "source": "seed_corpus",
        "citing_papers": [],
    },
    {
        "paper_id": "seed-sc-tools-2024",
        "title": "Smart Contract and DeFi Security Tools: Do They Meet the Needs of Practitioners?",
        "authors": ["Unknown"],
        "year": 2024,
        "venue": "Academic Venue",
        "url": "https://arxiv.org/abs/2305.04181",
        "content": (
            "We survey smart contract and DeFi security tools and compare them with practitioner needs. "
            "However, many tools have limited support for complex DeFi/cross-protocol compositions. "
            "Future work calls for better usability and coverage evaluation."
        ),
        "citation_count": 58,
        "arxiv_id": "2305.04181",
        "pdf_url": "https://arxiv.org/pdf/2305.04181.pdf",
        "evidence_level": "abstract_only",
        "source": "seed_corpus",
        "citing_papers": [],
    },
    {
        "paper_id": "seed-eth-sc-sec-2022",
        "title": "The State of Ethereum Smart Contracts Security: Vulnerabilities, Countermeasures, and Tool Support",
        "authors": ["Unknown"],
        "year": 2022,
        "venue": "Academic Venue",
        "url": "https://doi.org/10.1145/example.eth-sc-sec",
        "content": (
            "This survey reviews Ethereum smart contract vulnerabilities, countermeasures, and tools. "
            "However, smart contracts are far from being secure and attacks exploiting vulnerabilities "
            "have led to significant losses. Tool support for emerging cross-chain settings remains limited."
        ),
        "citation_count": 91,
        "evidence_level": "abstract_only",
        "source": "seed_corpus",
        "citing_papers": [],
    },
    {
        "paper_id": "seed-bridge-survey-2025",
        "title": "Blockchain Security in Focus: A Comprehensive Investigation Into Threats, Smart Contract Security, Cross-Chain Bridges, and Vulnerabilities Detection Tools and Techniques",
        "authors": ["Unknown"],
        "year": 2025,
        "venue": "Academic Venue",
        "url": "https://doi.org/10.1145/example.bridge-survey",
        "content": (
            "We present a comprehensive investigation into blockchain threats with emphasis on smart "
            "contract security and cross-chain bridges, summarizing vulnerability detection tools and "
            "techniques. However, bridge-specific detection still faces challenges in modeling "
            "cross-chain state dependencies."
        ),
        "citation_count": 17,
        "evidence_level": "abstract_only",
        "source": "seed_corpus",
        "citing_papers": [],
    },
    {
        "paper_id": "seed-seq-learning-2018",
        "title": "Towards Safer Smart Contracts: A Sequence Learning Approach to Detecting Security Threats",
        "authors": ["Unknown"],
        "year": 2018,
        "venue": "Academic Venue",
        "url": "https://arxiv.org/abs/1811.06632",
        "content": (
            "We propose a sequence learning approach to detecting security threats in smart contracts. "
            "However, learning-based detectors may generalize poorly to unseen vulnerability patterns "
            "and require labeled data."
        ),
        "citation_count": 75,
        "arxiv_id": "1811.06632",
        "pdf_url": "https://arxiv.org/pdf/1811.06632.pdf",
        "evidence_level": "abstract_only",
        "source": "seed_corpus",
        "citing_papers": [],
    },
    {
        "paper_id": "seed-sc-survey-2021",
        "title": "Blockchain smart contracts: Applications, challenges, and future trends",
        "authors": ["Unknown"],
        "year": 2021,
        "venue": "Academic Venue",
        "url": "https://doi.org/10.1145/example.sc-survey-2021",
        "content": (
            "This paper surveys blockchain smart contract applications, challenges, and future trends. "
            "Security, correctness, and interoperability challenges remain open research problems."
        ),
        "citation_count": 829,
        "evidence_level": "abstract_only",
        "source": "seed_corpus",
        "citing_papers": [],
    },
]


def seed_search(query: str, *, limit: int = 10) -> list[dict[str, Any]]:
    """Keyword filter over seed corpus (offline fallback)."""

    q = (query or "").lower()
    keys = [k for k in q.replace("/", " ").split() if len(k) > 2]
    scored: list[tuple[int, dict[str, Any]]] = []
    for paper in SEED_CORPUS:
        blob = f"{paper.get('title')} {paper.get('content')}".lower()
        score = sum(1 for k in keys if k in blob)
        # domain prior for cross-chain demos
        if any(t in blob for t in ("cross-chain", "bridge", "smart contract", "fuzz", "static")):
            score += 1
        if score:
            scored.append((score, paper))
    scored.sort(key=lambda x: (x[0], x[1].get("citation_count") or 0), reverse=True)
    if not scored:
        # if query unrelated, still return top cited seed papers for demo continuity
        return list(SEED_CORPUS)[:limit]
    return [p for _, p in scored[:limit]]
