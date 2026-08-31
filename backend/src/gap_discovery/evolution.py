"""Research evolution chain + limitation lifecycle builders."""

from __future__ import annotations

import logging
import re
from typing import Any, Optional
from uuid import uuid4

from gap_discovery.llm_utils import build_llm, extract_json, llm_chat
from config import Configuration

logger = logging.getLogger(__name__)


def build_evolution_and_lifecycles(
    *,
    topic: str,
    paper_cards: list[dict[str, Any]],
    external_critiques: list[dict[str, Any]],
) -> dict[str, Any]:
    """Return {evolution_chain, limitation_lifecycles} with evidence-aware MVP."""

    cards = sorted(
        [c for c in paper_cards if c.get("title")],
        key=lambda c: (c.get("year") or 0, -(c.get("citation_count") or 0)),
    )
    chain = _heuristic_evolution(cards, external_critiques)
    lifecycles = _heuristic_lifecycles(cards, external_critiques, chain)

    # optional LLM refine (non-authoritative)
    try:
        llm = build_llm(Configuration.from_env())
        raw = llm_chat(
            llm,
            system=(
                "Build research evolution edges and limitation lifecycles. "
                "Return STRICT JSON: {evolution_chain:[{from_paper_id,to_paper_id,relationship,"
                "previous_limitation,what_new_work_changed,remaining_problem,evidence}], "
                "limitation_lifecycles:[{limitation_id,description,first_reported_by,"
                "current_status,remaining_problem,supporting_papers,confidence,"
                "subsequent_attempts:[{paper_id,year,what_changed}]}]}. "
                "relationship in EXTENDS|IMPROVES|ADDRESSES_LIMITATION|PARTIALLY_SOLVES|ALTERNATIVE_APPROACH. "
                "current_status in SOLVED|PARTIALLY_SOLVED|STILL_OPEN|UNCERTAIN. "
                "Do NOT invent papers or critiques; use only provided evidence."
            ),
            user=(
                f"Topic: {topic}\nCards: {_slim_cards(cards)}\n"
                f"Critiques: {external_critiques[:12]}\nHeuristicChain: {chain[:8]}"
            ),
        )
        data = extract_json(raw)
        if isinstance(data, dict):
            if data.get("evolution_chain"):
                chain = data["evolution_chain"]
            if data.get("limitation_lifecycles"):
                lifecycles = data["limitation_lifecycles"]
    except Exception as exc:  # noqa: BLE001
        logger.warning("evolution LLM refine skipped: %s", exc)

    return {"evolution_chain": chain, "limitation_lifecycles": lifecycles}


def _slim_cards(cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for c in cards[:12]:
        lims = []
        for item in c.get("self_reported_limitations") or []:
            if isinstance(item, dict):
                lims.append(item.get("text") or item.get("claim"))
            else:
                lims.append(str(item))
        out.append(
            {
                "paper_id": c.get("paper_id"),
                "title": c.get("title"),
                "year": c.get("year"),
                "method": c.get("method"),
                "self_reported_limitations": lims[:4],
                "evidence_level": c.get("evidence_level"),
                "fulltext_status": c.get("fulltext_status"),
            }
        )
    return out


def _limitation_texts(card: dict[str, Any]) -> list[str]:
    texts = []
    for item in card.get("self_reported_limitations") or []:
        if isinstance(item, dict):
            t = item.get("text") or item.get("claim") or item.get("quote_or_summary")
        else:
            t = str(item)
        if t:
            texts.append(str(t).strip())
    for item in card.get("inferred_weaknesses") or []:
        if isinstance(item, dict) and item.get("source_type") == "ai_inferred":
            continue  # do not seed lifecycle primarily from ai_inferred
    return texts


def _heuristic_evolution(
    cards: list[dict[str, Any]], critiques: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    edges: list[dict[str, Any]] = []
    if len(cards) < 2:
        return edges
    # connect consecutive years when later card mentions earlier title keyword / critique link
    for i, older in enumerate(cards[:-1]):
        newer = cards[i + 1]
        rel = "EXTENDS"
        prev_lim = (_limitation_texts(older) or ["unspecified limitation"])[0][:180]
        remaining = (_limitation_texts(newer) or ["unspecified remaining issue"])[0][:180]
        evidence = f"chronological adjacency {older.get('year')}→{newer.get('year')}"
        for c in critiques:
            if c.get("target_paper_id") == older.get("paper_id") and c.get(
                "citing_paper_id"
            ) == newer.get("paper_id"):
                rel = (
                    "ADDRESSES_LIMITATION"
                    if c.get("critique_type") == "CRITIQUE"
                    else "EXTENDS"
                )
                evidence = f"citation context: {(c.get('critique_summary') or c.get('citation_context') or '')[:160]}"
                break
        edges.append(
            {
                "from_paper_id": older.get("paper_id"),
                "to_paper_id": newer.get("paper_id"),
                "from_title": older.get("title"),
                "to_title": newer.get("title"),
                "relationship": rel,
                "previous_limitation": prev_lim,
                "what_new_work_changed": (newer.get("method") or newer.get("title") or "")[:180],
                "remaining_problem": remaining,
                "evidence": evidence,
            }
        )
    return edges[:8]


def _normalize_lim_key(text: str) -> str:
    toks = re.findall(r"[a-z0-9]+", text.lower())
    return " ".join(toks[:8])


def _heuristic_lifecycles(
    cards: list[dict[str, Any]],
    critiques: list[dict[str, Any]],
    chain: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    buckets: dict[str, dict[str, Any]] = {}
    for card in cards:
        for text in _limitation_texts(card):
            key = _normalize_lim_key(text)
            if len(key) < 8:
                continue
            if key not in buckets:
                buckets[key] = {
                    "limitation_id": f"lim-{uuid4().hex[:8]}",
                    "description": text[:240],
                    "first_reported_by": card.get("paper_id"),
                    "first_reported_title": card.get("title"),
                    "first_year": card.get("year"),
                    "self_reported_evidence": [
                        {
                            "paper_id": card.get("paper_id"),
                            "evidence_level": card.get("evidence_level"),
                            "text": text[:200],
                        }
                    ],
                    "external_critiques": [],
                    "subsequent_attempts": [],
                    "supporting_papers": [card.get("paper_id")],
                    "current_status": "STILL_OPEN",
                    "remaining_problem": text[:200],
                    "confidence": "low",
                }
            else:
                b = buckets[key]
                if card.get("paper_id") not in b["supporting_papers"]:
                    b["supporting_papers"].append(card.get("paper_id"))
                b["self_reported_evidence"].append(
                    {
                        "paper_id": card.get("paper_id"),
                        "evidence_level": card.get("evidence_level"),
                        "text": text[:200],
                    }
                )

    # attach critiques
    for c in critiques:
        if c.get("critique_type") != "CRITIQUE":
            continue
        summary = c.get("critique_summary") or c.get("citation_context") or ""
        key = _normalize_lim_key(summary)
        # attach to closest bucket or create
        target = None
        for k, b in buckets.items():
            if len(set(k.split()) & set(key.split())) >= 2:
                target = b
                break
        if target is None and summary:
            target = {
                "limitation_id": f"lim-{uuid4().hex[:8]}",
                "description": summary[:240],
                "first_reported_by": c.get("target_paper_id"),
                "self_reported_evidence": [],
                "external_critiques": [],
                "subsequent_attempts": [],
                "supporting_papers": [c.get("target_paper_id"), c.get("citing_paper_id")],
                "current_status": "STILL_OPEN",
                "remaining_problem": summary[:200],
                "confidence": "medium",
            }
            buckets[key or target["limitation_id"]] = target
        if target is not None:
            target["external_critiques"].append(c)
            if c.get("citing_paper_id"):
                target["subsequent_attempts"].append(
                    {
                        "paper_id": c.get("citing_paper_id"),
                        "year": c.get("citing_year"),
                        "what_changed": (c.get("critique_summary") or "")[:180],
                    }
                )

    # status from chain + repeats
    for b in buckets.values():
        n_self = len(b.get("self_reported_evidence") or [])
        n_ext = len(b.get("external_critiques") or [])
        n_att = len(b.get("subsequent_attempts") or [])
        if n_att >= 2 and n_ext >= 1:
            b["current_status"] = "PARTIALLY_SOLVED"
            b["confidence"] = "medium"
        elif n_self >= 2 or n_ext >= 2:
            b["current_status"] = "STILL_OPEN"
            b["confidence"] = "medium"
        elif n_att == 0 and n_self == 1 and n_ext == 0:
            b["current_status"] = "UNCERTAIN"
            b["confidence"] = "low"
        # chain addresses
        for edge in chain:
            if edge.get("from_paper_id") == b.get("first_reported_by"):
                if edge.get("relationship") in {"ADDRESSES_LIMITATION", "PARTIALLY_SOLVES", "IMPROVES"}:
                    b["current_status"] = "PARTIALLY_SOLVED"
                    b["remaining_problem"] = edge.get("remaining_problem") or b.get(
                        "remaining_problem"
                    )

    # prefer multi-evidence
    ranked = sorted(
        buckets.values(),
        key=lambda x: (
            0 if x.get("current_status") in {"STILL_OPEN", "PARTIALLY_SOLVED"} else 1,
            -len(x.get("supporting_papers") or []),
            -len(x.get("external_critiques") or []),
        ),
    )
    return ranked[:8]
