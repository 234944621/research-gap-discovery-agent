"""Interview-oriented Research Agent pipeline nodes.

Flow:
Planner → Search → PaperReader → Analyzer → GapDiscover
→ VerifyLoop (tool calling / re-search) → CrossDomain (optional)
→ Reporter

Memory + RAG support verification and historical recall.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any
from uuid import uuid4

from config import Configuration
from gap_discovery.llm_utils import build_llm, env_int, extract_json, llm_chat
from gap_discovery.memory import ResearchMemoryStore
from gap_discovery.models import paper_card_from_search_result
from gap_discovery.rag import ContextBuilder, PaperRAG
from gap_discovery.state import ResearchState, append_event, emit
from services.academic_search import AcademicSearchService

logger = logging.getLogger(__name__)

from gap_discovery.relevance import (
    filter_relevant_papers as _filter_relevant_papers,
    to_academic_query as _to_academic_query,
)


def _search_service() -> AcademicSearchService:
    return AcademicSearchService(
        semantic_scholar_api_key=os.getenv("SEMANTIC_SCHOLAR_API_KEY"),
        openalex_mailto=os.getenv("OPENALEX_MAILTO"),
        max_results=env_int("LANDSCAPE_MAX_PAPERS", 20),
        include_citing=False,
    )


def node_recall_memory(state: ResearchState) -> ResearchState:
    store = ResearchMemoryStore()
    # Optional heavier integrate before read (default off; save_gap already consolidates)
    if os.getenv("MEMORY_CONSOLIDATE_ON_RECALL", "false").lower() in {"1", "true", "yes"}:
        try:
            store.consolidate_topic(state["topic"])
        except Exception as exc:  # noqa: BLE001
            logger.debug("consolidate_topic on recall skipped: %s", exc)
    else:
        # One-shot backfill: episodic REJECTED exists but semantic layer empty
        try:
            if store.rejected_gaps_with_reasons(state["topic"]) and not store.recall_semantic_lessons(
                state["topic"], limit=1
            ):
                store.consolidate_topic(state["topic"])
        except Exception as exc:  # noqa: BLE001
            logger.debug("memory backfill consolidate skipped: %s", exc)
    bundle = store.recall_bundle(state["topic"])
    # Flat list for RAG/tools; rejected (with reasons) first; includes lessons/entities/procedures
    state["research_memory"] = bundle.get("flat") or []
    state["memory_semantic_lessons"] = bundle.get("semantic_lessons") or []
    state["memory_entities"] = bundle.get("entities") or []
    state["memory_procedures"] = bundle.get("procedures") or []
    rejected = bundle.get("rejected_gaps") or []
    summary = bundle.get("summary") or {}
    state = emit(
        state,
        (
            f"已召回 Memory：情节 gaps="
            f"{summary.get('other_gap_count', 0)}+REJECTED={len(rejected)}，"
            f"queries={summary.get('query_count', 0)}；"
            f"语义 lessons={summary.get('lesson_count', 0)}；"
            f"实体={summary.get('entity_count', 0)}；"
            f"程序 SOP={summary.get('procedure_count', 0)}"
            f"（topic_key={bundle.get('topic_key')}）"
        ),
        stage="memory_recall",
    )
    if rejected:
        state = append_event(
            state,
            {
                "type": "artifact",
                "artifact": "memory_rejected_gaps",
                "data": [
                    {
                        "description": g.get("description"),
                        "reason": g.get("reason"),
                        "gap_id": g.get("gap_id"),
                    }
                    for g in rejected[:10]
                ],
            },
        )
    if bundle.get("semantic_lessons"):
        state = append_event(
            state,
            {
                "type": "artifact",
                "artifact": "memory_semantic_lessons",
                "data": bundle["semantic_lessons"][:8],
            },
        )
    if bundle.get("entities"):
        state = append_event(
            state,
            {
                "type": "artifact",
                "artifact": "memory_entities",
                "data": bundle["entities"][:10],
            },
        )
    if bundle.get("procedures"):
        state = append_event(
            state,
            {
                "type": "artifact",
                "artifact": "memory_procedures",
                "data": bundle["procedures"][:4],
            },
        )
    if bundle.get("recent_queries"):
        state = append_event(
            state,
            {
                "type": "artifact",
                "artifact": "memory_recent_queries",
                "data": bundle["recent_queries"][:8],
            },
        )
    return state

def _rejected_from_state(state: ResearchState) -> list[dict[str, Any]]:
    """Prefer structured REJECTED entries (description + reason) from research_memory."""

    out: list[dict[str, Any]] = []
    for m in state.get("research_memory") or []:
        if not isinstance(m, dict):
            continue
        if m.get("type") == "gap" and (m.get("status") or "").upper() == "REJECTED":
            out.append(
                {
                    "description": m.get("description") or "",
                    "reason": m.get("reason") or "",
                    "gap_id": m.get("gap_id"),
                }
            )
        elif m.get("status") == "REJECTED" and m.get("description"):
            out.append(
                {
                    "description": m.get("description") or "",
                    "reason": m.get("reason") or "",
                    "gap_id": m.get("gap_id"),
                }
            )
    if out:
        return out
    store = ResearchMemoryStore()
    return store.rejected_gaps_with_reasons(state["topic"])


def node_planner(state: ResearchState) -> ResearchState:
    state = emit(state, "Research Planner：拆解研究问题与检索关键词...", stage="planner")
    topic = state["topic"]
    # Overly broad / ambiguous topics → pause for user input (keep checkpoint)
    broad_markers = ("一切", "所有领域", "全世界", "任意", "帮我研究一下", "something interesting")
    topic_stripped = (topic or "").strip()
    if len(topic_stripped) < 4 or any(m in topic_stripped.lower() for m in broad_markers) or topic_stripped in {
        "AI",
        "人工智能",
        "科研",
        "research",
    }:
        state["status"] = "paused"
        state["verification_status"] = "NEEDS_USER_INPUT"
        state["termination_reason"] = "NEEDS_USER_INPUT"
        state["warnings"] = list(state.get("warnings") or []) + [
            "topic too broad or ambiguous; waiting for user to narrow scope"
        ]
        state = emit(
            state,
            "主题范围过大或不明确，已暂停并等待用户补充（NEEDS_USER_INPUT）",
            stage="planner",
        )
        state = append_event(
            state,
            {
                "type": "task_paused",
                "reason": "NEEDS_USER_INPUT",
                "message": "请缩小主题范围或明确研究目标后 Resume",
            },
        )
        return state

    academic_query = _to_academic_query(topic)
    cfg = Configuration.from_env()
    rejected = _rejected_from_state(state)
    rejected_for_prompt = [
        {"description": r.get("description"), "reason": r.get("reason")}
        for r in rejected[:5]
    ]
    lessons_for_prompt = [
        {"rule": x.get("rule_text"), "evidence_count": x.get("evidence_count")}
        for x in (state.get("memory_semantic_lessons") or [])[:5]
    ]
    entities_for_prompt = [
        {
            "type": x.get("entity_type"),
            "name": x.get("name"),
            "value": x.get("value"),
        }
        for x in (state.get("memory_entities") or [])[:6]
        if x.get("entity_type") in {"keep_direction", "topic_stats", "rejected_pattern"}
    ]
    proc_for_prompt = [
        {"name": p.get("name"), "steps": (p.get("steps") or [])[:4]}
        for p in (state.get("memory_procedures") or [])[:2]
    ]
    plan = {
        "research_questions": [
            f"What are the mainstream methods for {topic}?",
            f"What limitations remain unsolved in {topic}?",
            f"Which candidate research gaps are worth verifying for {topic}?",
        ],
        "search_keywords": [
            academic_query,
            f"{academic_query} survey",
            f"{academic_query} static analysis",
            f"{academic_query} fuzzing",
        ],
        "notes": "fallback plan",
    }
    try:
        llm = build_llm(cfg)
        raw = llm_chat(
            llm,
            system=(
                "You are a research planner for a Gap Discovery Agent. "
                "Output STRICT JSON with keys research_questions (string[]), "
                "search_keywords (string[], English academic queries), taxonomy_hint (string[]). "
                "Do NOT invent paper titles. Do NOT claim true novelty. "
                "Avoid proposing directions that only restate historically REJECTED gaps. "
                "Obey semantic_lessons (distilled rules) and prefer procedure query seeds when present."
            ),
            user=(
                f"Research topic: {topic}\n"
                f"Preferred English query seed: {academic_query}\n"
                f"Historical REJECTED gaps (episodic):\n{rejected_for_prompt}\n"
                f"Semantic lessons:\n{lessons_for_prompt}\n"
                f"Entity memory:\n{entities_for_prompt}\n"
                f"Procedural SOPs:\n{proc_for_prompt}"
            ),
        )
        data = extract_json(raw)
        if isinstance(data, dict):
            plan["research_questions"] = data.get("research_questions") or plan["research_questions"]
            plan["search_keywords"] = data.get("search_keywords") or plan["search_keywords"]
            plan["taxonomy_hint"] = data.get("taxonomy_hint") or []
            plan["notes"] = "llm plan"
    except Exception as exc:
        logger.warning("Planner LLM failed, using fallback: %s", exc)
        state = emit(state, f"Planner LLM 不可用，使用规则兜底：{exc}", stage="planner")

    state["plan"] = plan
    state["research_questions"] = list(plan.get("research_questions") or [])
    state["search_keywords"] = list(plan.get("search_keywords") or [])
    state["scope"] = {
        "topic": topic,
        "academic_query": academic_query,
        "disclaimer": "帮助发现和验证候选 Research Gap，不宣称自动发现真正创新点",
    }
    state = emit(
        state,
        f"Planner 完成：{len(state['research_questions'])} 个问题，{len(state['search_keywords'])} 个关键词",
        stage="planner",
    )
    state = append_event(state, {"type": "artifact", "artifact": "plan", "data": plan})
    return state


def node_search(state: ResearchState) -> ResearchState:
    state = emit(state, "Literature Search：调用学术搜索工具...", stage="search")
    topic = state["topic"]
    keywords = state.get("search_keywords") or [_to_academic_query(topic)]
    max_papers = env_int("LANDSCAPE_MAX_PAPERS", 20)
    service = _search_service()
    store = ResearchMemoryStore()
    collected: list[dict[str, Any]] = []

    for query in keywords[:2]:
        store.save_query(topic, query, purpose="landscape_search")
        payload = service.search(query, backend="academic", max_results=min(12, max_papers))
        for notice in payload.get("notices") or []:
            state = emit(state, notice, stage="search")
        collected.extend(payload.get("results") or [])
        # 轻微限速，降低 429
        import time

        time.sleep(0.8)

    papers = _filter_relevant_papers(collected, topic, min_score=1)
    # dedupe by title/doi
    seen = set()
    unique = []
    for p in papers:
        key = (p.get("doi") or p.get("paper_id") or (p.get("title") or "").lower()).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(p)
    unique = unique[:max_papers]

    state["papers"] = unique
    state["candidate_papers"] = unique
    state["landscape_papers"] = unique
    state = emit(
        state,
        f"搜索完成：保留 {len(unique)} 篇主题相关论文",
        stage="search",
        paper_count=len(unique),
    )
    state = append_event(
        state,
        {
            "type": "artifact",
            "artifact": "papers",
            "data": [
                {
                    "title": p.get("title"),
                    "year": p.get("year"),
                    "citation_count": p.get("citation_count"),
                    "url": p.get("url"),
                }
                for p in unique
            ],
        },
    )
    return state


def node_paper_reader(state: ResearchState) -> ResearchState:
    state = emit(state, "Paper Reader：全文获取 + PaperCard...", stage="paper_reader")
    papers = (state.get("papers") or state.get("candidate_papers") or [])[
        : env_int("FOCUS_MAX_PAPERS", 10)
    ]
    cards = []
    store = ResearchMemoryStore()
    rag = PaperRAG()
    from gap_discovery.fulltext import FullTextService
    from gap_discovery.paper_reader_fulltext import enrich_card_from_fulltext

    fulltext = FullTextService()
    max_ft = int(os.getenv("MAX_FULLTEXT_PAPERS", "6"))
    ft_ok = 0

    enriched_papers: list[dict[str, Any]] = []
    for idx, paper in enumerate(papers, start=1):
        state = emit(
            state,
            f"正在尝试获取全文 {idx}/{len(papers)}: {(paper.get('title') or '')[:80]}",
            stage="paper_reader",
        )
        paper_e = dict(paper)
        if ft_ok < max_ft:
            paper_e = fulltext.enrich_paper(paper_e)
        else:
            paper_e["fulltext_status"] = paper_e.get("fulltext_status") or "ABSTRACT_ONLY"

        status = paper_e.get("fulltext_status") or "ABSTRACT_ONLY"
        if status == "FULLTEXT":
            ft_ok += 1
            state = emit(
                state,
                f"✓ Fulltext available ({paper_e.get('fulltext_source')}): {(paper_e.get('title') or '')[:70]}",
                stage="paper_reader",
            )
            rag.add_fulltext_chunks(paper_e.get("fulltext_chunks") or [])
        elif status == "FAILED":
            state = emit(
                state,
                f"全文失败→abstract_only: {(paper_e.get('title') or '')[:70]}",
                stage="paper_reader",
            )
        enriched_papers.append(paper_e)

        card = paper_card_from_search_result(paper_e, extract_hints=True)
        if status == "FULLTEXT":
            state = emit(state, f"正在读取 Limitations/Method sections: {card.title[:60]}", stage="paper_reader")
            card = enrich_card_from_fulltext(card, paper_e)

        try:
            if card.abstract and os.getenv("ENABLE_LLM_PAPER_READER", "true").lower() in {
                "1",
                "true",
                "yes",
            }:
                # only enrich fields still empty; keep provenance
                if not card.method or card.evidence_level.value == "abstract_only":
                    card = _llm_enrich_card(card)
        except Exception as exc:
            logger.debug("LLM paper enrich skipped: %s", exc)

        payload = card.to_state_dict()
        # keep sections pointer small for later tools
        payload["sections_present"] = list((paper_e.get("sections") or {}).keys())
        payload["fulltext_status"] = status
        payload["pdf_url"] = paper_e.get("pdf_url")
        payload["fulltext_source"] = paper_e.get("fulltext_source")
        cards.append(payload)
        store.save_paper(state["topic"], payload)
        rag.add_paper_card(payload)

    state["papers"] = enriched_papers
    state["paper_cards"] = cards
    state = append_event(
        state,
        {
            "type": "artifact",
            "artifact": "literature_matrix",
            "data": {
                "count": len(cards),
                "fulltext_count": sum(1 for c in cards if c.get("fulltext_status") == "FULLTEXT"),
                "papers": cards,
            },
        },
    )
    state = emit(
        state,
        f"Paper Reader 完成：{len(cards)} 张 PaperCard（fulltext={ft_ok}）",
        stage="paper_reader",
    )
    return state


def _llm_enrich_card(card):
    cfg = Configuration.from_env()
    llm = build_llm(cfg)
    raw = llm_chat(
        llm,
        system=(
            "Extract structured fields from a paper abstract. "
            "Return STRICT JSON with keys: research_problem, method, core_technique, "
            "contributions (string[]), possible_self_reported_limitation (string|null). "
            "If uncertain, use null. Never invent experimental numbers. "
            "If limitation is not explicitly in abstract, set possible_self_reported_limitation=null."
        ),
        user=f"Title: {card.title}\nAbstract: {card.abstract}",
    )
    data = extract_json(raw)
    if not isinstance(data, dict):
        return card
    card.research_problem = data.get("research_problem") or card.research_problem
    card.method = data.get("method") or card.method
    card.core_technique = data.get("core_technique") or card.core_technique
    if data.get("contributions"):
        card.contributions = list(data.get("contributions") or [])
    lim = data.get("possible_self_reported_limitation")
    if lim:
        from gap_discovery.models import EvidenceItem, EvidenceLevel, EvidenceSourceType

        card.self_reported_limitations.append(
            EvidenceItem(
                text=str(lim),
                source_type=EvidenceSourceType.SELF_REPORTED,
                evidence_level=EvidenceLevel.ABSTRACT_ONLY,
                location="abstract",
                source_paper_id=card.paper_id,
                source_paper_title=card.title,
                confidence="low",
                notes="LLM extracted from abstract; needs full-text confirmation",
            )
        )
    return card


def node_analyzer(state: ResearchState) -> ResearchState:
    """Merged Literature Comparison + Limitation Mining."""

    state = emit(state, "Research Analyzer：方法分类 / 优缺点 / 瓶颈...", stage="analyzer")
    cards = state.get("paper_cards") or []
    summary_lines = []
    for c in cards[:10]:
        summary_lines.append(
            {
                "paper_id": c.get("paper_id"),
                "title": c.get("title"),
                "year": c.get("year"),
                "method": c.get("method"),
                "core_technique": c.get("core_technique"),
                "research_problem": c.get("research_problem"),
                "inferred_weaknesses": c.get("inferred_weaknesses"),
                "self_reported_limitations": c.get("self_reported_limitations"),
            }
        )

    analysis = {
        "method_groups": _heuristic_method_groups(cards),
        "bottlenecks": [],
        "shared_limitations": [],
        "notes": "heuristic analyzer",
    }
    try:
        llm = build_llm(Configuration.from_env())
        raw = llm_chat(
            llm,
            system=(
                "You are a research analyzer. Given PaperCards (mostly abstract_only), "
                "return STRICT JSON with: method_groups ([{name, paper_ids, summary}]), "
                "shared_limitations ([{text, supporting_paper_ids, evidence_type}]), "
                "bottlenecks (string[]). "
                "evidence_type must be one of self_reported|external_critique|ai_inferred. "
                "Do not claim a limitation is self_reported unless explicitly present. "
                "Do NOT invent citations."
            ),
            user=f"Topic: {state['topic']}\nPaperCards:\n{summary_lines}",
        )
        data = extract_json(raw)
        if isinstance(data, dict):
            analysis.update({k: data.get(k, analysis.get(k)) for k in analysis})
            analysis["notes"] = "llm analyzer"
    except Exception as exc:
        logger.warning("Analyzer LLM failed: %s", exc)
        state = emit(state, f"Analyzer LLM 回退启发式：{exc}", stage="analyzer")
        analysis["shared_limitations"] = _heuristic_shared_limitations(cards)
        analysis["bottlenecks"] = [
            "Cross-component / cross-chain semantic tracking remains difficult",
            "Many tools lack strong evidence beyond single-chain settings",
        ]

    state["analysis"] = analysis
    state["method_taxonomy"] = list(analysis.get("method_groups") or [])
    state["research_directions"] = [
        {
            "direction_name": g.get("name"),
            "description": g.get("summary"),
            "paper_count": len(g.get("paper_ids") or []),
            "representative_papers": g.get("paper_ids") or [],
            "main_methods": [g.get("name")],
        }
        for g in state["method_taxonomy"]
        if g.get("name")
    ]
    state["limitations"] = list(analysis.get("shared_limitations") or [])
    state["selected_direction"] = (
        state["research_directions"][0]["direction_name"]
        if state["research_directions"]
        else state["topic"]
    )
    state = append_event(
        state,
        {"type": "artifact", "artifact": "analysis", "data": analysis},
    )
    state = emit(
        state,
        f"Analyzer 完成：{len(state['method_taxonomy'])} 类方法，{len(state['limitations'])} 条共享局限",
        stage="analyzer",
    )
    return state


def node_evidence_chain(state: ResearchState) -> ResearchState:
    """Forward citations + citation contexts + evolution + limitation lifecycles."""

    state = emit(
        state,
        "Evidence Chain：查找后续引用 / 抽取 citation context / 构建演进与局限生命周期...",
        stage="evidence_chain",
    )
    from gap_discovery.citations import CitationService, paper_age_years
    from gap_discovery.evolution import build_evolution_and_lifecycles
    from gap_discovery.models import EvidenceItem, EvidenceLevel, EvidenceSourceType

    cards = list(state.get("paper_cards") or [])
    cite = CitationService()
    external_critiques: list[dict[str, Any]] = []
    max_targets = int(os.getenv("MAX_CITATION_TARGETS", "3"))
    max_contexts = int(os.getenv("MAX_CITATION_CONTEXTS", "4"))
    contexts_done = 0

    # Prefer older-but-not-ancient papers for citation critique (1~8 years)
    ranked = sorted(
        cards,
        key=lambda c: (
            0 if (1 <= (paper_age_years(c.get("year")) or 99) <= 8) else 1,
            -(c.get("citation_count") or 0),
        ),
    )

    for card in ranked[:max_targets]:
        age = paper_age_years(card.get("year"))
        state = emit(
            state,
            f"正在查找后续引用：{(card.get('title') or '')[:70]} (age={age})",
            stage="evidence_chain",
        )
        citing = cite.find_citing_papers(
            paper_id=card.get("paper_id"),
            title=card.get("title"),
            doi=card.get("doi"),
            year=card.get("year"),
        )
        card["citing_papers"] = citing[:8]
        state = emit(
            state,
            f"找到 {len(citing)} 篇高相关 citing/related papers",
            stage="evidence_chain",
        )

        for cp in citing[:3]:
            if contexts_done >= max_contexts:
                break
            # skip optional critique for very young targets already handled in budget
            if age is not None and age < 1:
                continue
            state = emit(
                state,
                f"正在分析 citation context：{(cp.get('title') or '')[:70]}",
                stage="evidence_chain",
            )
            ctx = cite.get_citation_context(target_paper=card, citing_paper=cp)
            if not ctx:
                continue
            contexts_done += 1
            if ctx.get("critique_type") == "CRITIQUE" and ctx.get("critique_summary"):
                external_critiques.append(ctx)
                state = emit(
                    state,
                    f"✓ External critique: {str(ctx.get('critique_summary'))[:100]}",
                    stage="evidence_chain",
                )
                # attach onto target card as EvidenceItem
                try:
                    item = EvidenceItem(
                        text=str(ctx.get("critique_summary")),
                        source_type=EvidenceSourceType.EXTERNAL_CRITIQUE,
                        evidence_level=(
                            EvidenceLevel.FULL_TEXT
                            if ctx.get("fulltext_status") == "FULLTEXT"
                            else EvidenceLevel.ABSTRACT_ONLY
                        ),
                        location="citation_context",
                        source_paper_id=str(ctx.get("citing_paper_id") or ""),
                        source_paper_title=ctx.get("citing_paper_title"),
                        confidence=ctx.get("confidence") or "medium",
                        notes=f"cites {card.get('paper_id')}",
                    )
                    card.setdefault("external_critiques", []).append(item.model_dump(mode="json"))
                except Exception as exc:  # noqa: BLE001
                    logger.debug("attach critique failed: %s", exc)
            elif ctx.get("critique_type") == "UNKNOWN":
                state = emit(
                    state,
                    "citation context UNKNOWN（无全文或未定位，不生成 external_critique）",
                    stage="evidence_chain",
                )

    state = emit(state, "正在构建技术演进链与 Limitation Lifecycle...", stage="evidence_chain")
    built = build_evolution_and_lifecycles(
        topic=state["topic"],
        paper_cards=cards,
        external_critiques=external_critiques,
    )
    state["paper_cards"] = cards
    state["external_critiques"] = external_critiques
    state["evolution_chain"] = built.get("evolution_chain") or []
    state["limitation_lifecycles"] = built.get("limitation_lifecycles") or []

    # persist lifecycles into memory
    store = ResearchMemoryStore()
    for life in state["limitation_lifecycles"]:
        store.save_json_artifact(
            state["topic"],
            kind="limitation_lifecycle",
            key=str(life.get("limitation_id")),
            payload=life,
        )
    for edge in state["evolution_chain"]:
        store.save_json_artifact(
            state["topic"],
            kind="evolution_relationship",
            key=f"{edge.get('from_paper_id')}->{edge.get('to_paper_id')}",
            payload=edge,
        )
    for c in external_critiques:
        store.save_json_artifact(
            state["topic"],
            kind="citation_critique",
            key=f"{c.get('target_paper_id')}:{c.get('citing_paper_id')}",
            payload=c,
        )

    for life in state["limitation_lifecycles"][:5]:
        state = emit(
            state,
            f"Limitation {(life.get('description') or '')[:70]} => {life.get('current_status')}",
            stage="evidence_chain",
        )

    state = append_event(
        state,
        {
            "type": "artifact",
            "artifact": "evidence_chain",
            "data": {
                "external_critiques": external_critiques,
                "evolution_chain": state["evolution_chain"],
                "limitation_lifecycles": state["limitation_lifecycles"],
            },
        },
    )
    state = emit(
        state,
        (
            f"Evidence Chain 完成：critiques={len(external_critiques)}, "
            f"edges={len(state['evolution_chain'])}, lifecycles={len(state['limitation_lifecycles'])}"
        ),
        stage="evidence_chain",
    )
    return state


def _heuristic_method_groups(cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets = {
        "Static Analysis": ["static", "symbolic", "formal", "smartaxe"],
        "Fuzzing / Dynamic": ["fuzz", "dynamic", "ityfuzz"],
        "Attack Monitoring": ["attack", "xscope", "monitor", "exploit"],
        "Survey / Tool Landscape": ["survey", "tool", "taxonomy"],
    }
    groups = {k: [] for k in buckets}
    for c in cards:
        blob = f"{c.get('title')} {c.get('method')} {c.get('core_technique')}".lower()
        placed = False
        for name, keys in buckets.items():
            if any(k in blob for k in keys):
                groups[name].append(c.get("paper_id"))
                placed = True
                break
        if not placed:
            groups.setdefault("Other", []).append(c.get("paper_id"))
    return [
        {"name": name, "paper_ids": ids, "summary": f"{len(ids)} papers related to {name}"}
        for name, ids in groups.items()
        if ids
    ]


def _heuristic_shared_limitations(cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "text": "Many existing evaluations remain abstract-level or single-chain oriented; cross-chain bridge-specific coverage is uneven.",
            "supporting_paper_ids": [c.get("paper_id") for c in cards[:5]],
            "evidence_type": "ai_inferred",
        }
    ]


def node_gap_discover(state: ResearchState) -> ResearchState:
    state = emit(
        state,
        "Gap Discovery：优先从 Limitation Lifecycle / 多源证据形成 Candidate Gaps...",
        stage="gap_discover",
    )
    store = ResearchMemoryStore()
    rejected_records = _rejected_from_state(state)
    rejected_descs = [r.get("description") or "" for r in rejected_records if r.get("description")]
    cards = state.get("paper_cards") or []
    limitations = state.get("limitations") or []
    lifecycles = list(state.get("limitation_lifecycles") or [])
    critiques = list(state.get("external_critiques") or [])

    def _skip_duplicate(desc: str) -> bool:
        if not desc:
            return True
        # Prefer store similarity (lexical + optional embedding) against REJECTED
        hit = store.find_similar_rejected(state["topic"], desc)
        if hit:
            return True
        return any(_similar(desc, r) for r in rejected_descs)

    gaps = []
    # Priority: STILL_OPEN / PARTIALLY_SOLVED lifecycles
    prioritized = [
        life
        for life in lifecycles
        if life.get("current_status") in {"STILL_OPEN", "PARTIALLY_SOLVED"}
    ]
    # de-prioritize single old mention with no follow-up
    for idx, life in enumerate(prioritized[:5], start=1):
        desc = life.get("remaining_problem") or life.get("description") or ""
        if _skip_duplicate(desc):
            continue
        gaps.append(
            {
                "gap_id": f"gap-life-{idx}-{uuid4().hex[:6]}",
                "description": desc,
                "supporting_evidence": [
                    {
                        "type": "limitation_lifecycle",
                        "status": life.get("current_status"),
                        "first_reported_by": life.get("first_reported_by"),
                        "text": life.get("description"),
                    }
                ],
                "supporting_papers": life.get("supporting_papers") or [],
                "self_reported_count": len(life.get("self_reported_evidence") or []),
                "external_critique_count": len(life.get("external_critiques") or []),
                "existing_attempts": life.get("subsequent_attempts") or [],
                "lifecycle_status": life.get("current_status"),
                "first_reported_by": life.get("first_reported_by"),
                "remaining_problem": life.get("remaining_problem"),
                "current_status": "CANDIDATE",
                "confidence": life.get("confidence") or "medium",
                "verification_queries": _gap_queries(state["topic"], desc),
                "human_verification_needed": True,
            }
        )

    # secondary: shared analyzer limitations if still short
    if len(gaps) < 2:
        for idx, lim in enumerate(limitations[:4], start=1):
            desc = lim.get("text") if isinstance(lim, dict) else str(lim)
            if _skip_duplicate(desc):
                continue
            # skip low-priority single-paper uncertain items when lifecycle exists
            gaps.append(
                {
                    "gap_id": f"gap-{idx}-{uuid4().hex[:6]}",
                    "description": desc,
                    "supporting_evidence": [lim],
                    "supporting_papers": lim.get("supporting_paper_ids")
                    if isinstance(lim, dict)
                    else [c.get("paper_id") for c in cards[:4]],
                    "self_reported_count": 1
                    if isinstance(lim, dict) and lim.get("evidence_type") == "self_reported"
                    else 0,
                    "external_critique_count": 1
                    if isinstance(lim, dict) and lim.get("evidence_type") == "external_critique"
                    else 0,
                    "existing_attempts": [],
                    "current_status": "CANDIDATE",
                    "confidence": "low",
                    "verification_queries": _gap_queries(state["topic"], desc),
                    "human_verification_needed": True,
                }
            )

    if not gaps:
        gaps.append(
            {
                "gap_id": f"gap-default-{uuid4().hex[:6]}",
                "description": (
                    f"For '{state['topic']}', repeatedly reported bottlenecks around "
                    "cross-chain semantic correlation remain under-resolved in current evidence scope."
                ),
                "supporting_evidence": [{"type": "ai_inferred", "text": "default synthesized gap"}],
                "supporting_papers": [c.get("paper_id") for c in cards[:5]],
                "self_reported_count": 0,
                "external_critique_count": len(critiques),
                "existing_attempts": [],
                "current_status": "CANDIDATE",
                "confidence": "low",
                "verification_queries": _gap_queries(state["topic"], "cross-chain state tracking"),
                "human_verification_needed": True,
            }
        )

    # LLM refine top gaps with lifecycle context
    rejected_for_llm = [
        {"description": r.get("description"), "reason": r.get("reason")}
        for r in rejected_records[:5]
    ]
    try:
        llm = build_llm(Configuration.from_env())
        raw = llm_chat(
            llm,
            system=(
                "Create candidate research gaps from Limitation Lifecycles and critiques. "
                "Return STRICT JSON: {gaps:[{gap_id, description, supporting_paper_ids, "
                "verification_queries, confidence, lifecycle_status, first_reported_by, "
                "subsequent_attempts, remaining_problem}]}. "
                "Prefer STILL_OPEN / PARTIALLY_SOLVED multi-paper issues. "
                "verification_queries must be SHORT English KEYWORD queries "
                "(max 12 words, no full questions, no parentheses). "
                "Do not restate historically REJECTED gaps (see reasons)."
            ),
            user=(
                f"Topic: {state['topic']}\nLifecycles: {lifecycles[:6]}\n"
                f"Critiques: {critiques[:6]}\nLimitations: {limitations[:6]}\n"
                f"Avoid repeating rejected (description+reason): {rejected_for_llm}"
            ),
        )
        data = extract_json(raw)
        llm_gaps = data.get("gaps") if isinstance(data, dict) else None
        if llm_gaps:
            refined = []
            for i, g in enumerate(llm_gaps[:4], start=1):
                desc = g.get("description") or ""
                if _skip_duplicate(desc):
                    continue
                refined.append(
                    {
                        "gap_id": g.get("gap_id") or f"gap-llm-{i}",
                        "description": desc,
                        "supporting_evidence": lifecycles[:2] or limitations[:2],
                        "supporting_papers": g.get("supporting_paper_ids")
                        or [c.get("paper_id") for c in cards[:4]],
                        "self_reported_count": 0,
                        "external_critique_count": len(critiques),
                        "existing_attempts": g.get("subsequent_attempts") or [],
                        "lifecycle_status": g.get("lifecycle_status"),
                        "first_reported_by": g.get("first_reported_by"),
                        "remaining_problem": g.get("remaining_problem") or desc,
                        "current_status": "CANDIDATE",
                        "confidence": g.get("confidence") or "low",
                        "verification_queries": g.get("verification_queries")
                        or _gap_queries(state["topic"], desc),
                        "human_verification_needed": True,
                    }
                )
            if refined:
                gaps = refined
    except Exception as exc:
        logger.warning("Gap LLM refine failed: %s", exc)

    state["candidate_gaps"] = gaps
    state = append_event(
        state,
        {"type": "artifact", "artifact": "gap_matrix", "data": {"gaps": gaps}},
    )
    state = emit(state, f"发现 {len(gaps)} 个 Candidate Gaps（待验证）", stage="gap_discover")
    return state


def node_gap_verify(state: ResearchState) -> ResearchState:
    """Gap Verification Agent: LLM autonomously chooses tools (search/memory/rag)."""

    import time

    from gap_discovery.tasks import get_task_store

    state = emit(
        state,
        "Gap Verification Agent：LLM 自主选择 search / memory / RAG 工具...",
        stage="gap_verify",
    )
    gaps = list(state.get("candidate_gaps") or [])
    max_iter = int(state.get("max_iterations") or 6)
    max_vr = int(state.get("max_verification_rounds") or max_iter)
    max_tool_calls = int(state.get("max_tool_calls") or 24)
    max_token_budget = state.get("max_token_budget")
    iteration = int(state.get("iteration_count") or 0)
    verification_round = int(state.get("verification_round") or 0)
    service = _search_service()
    store = ResearchMemoryStore()
    task_store = get_task_store()
    task_id = str(state.get("task_id") or "")
    rag = PaperRAG()
    for card in state.get("paper_cards") or []:
        rag.add_paper_card(card)
    rag.add_memory_items(state.get("research_memory") or [])
    state = emit(
        state,
        f"RAG backend={rag.active_backend}（PaperCard→Embedding→Chroma/Lexical）",
        stage="gap_verify",
    )

    verified = list(state.get("verified_gaps") or [])
    results = list(state.get("gap_verification_results") or [])
    needs_more = False
    warnings = list(state.get("warnings") or [])
    tool_traces = list(state.get("tool_traces") or [])
    visited = list(state.get("visited_actions") or [])
    retry_counts = dict(state.get("retry_counts") or {})
    tool_call_count = int(state.get("tool_call_count") or 0)
    token_usage = int(state.get("token_usage") or 0)
    executed = list(state.get("executed_side_effects") or [])
    termination_reason = state.get("termination_reason")
    verification_status = state.get("verification_status") or "pending"
    evidence_status = state.get("evidence_status") or "unknown"

    task_timeout_s = float(os.getenv("GAP_TASK_TIMEOUT_S", "900"))
    started = state.get("started_at")
    deadline = time.time() + task_timeout_s
    # Prefer absolute deadline from started_at if parseable
    try:
        if started:
            from datetime import datetime

            t0 = datetime.fromisoformat(started.replace("Z", "+00:00")).timestamp()
            deadline = t0 + task_timeout_s
    except Exception:
        pass

    from gap_discovery.verify_agent import run_gap_verify_agent

    def _search(q: str) -> dict[str, Any]:
        queries = _normalize_search_queries([q], state["topic"])
        query = queries[0] if queries else q
        side_key = f"save_query:verify:{query}"
        if side_key not in executed and (
            not task_id or task_store.claim_side_effect(task_id, side_key, "save_query")
        ):
            store.save_query(state["topic"], query, purpose="verify:tool")
            executed.append(side_key)
        payload = service.search(query, backend="academic", max_results=5)
        papers = _filter_relevant_papers(
            payload.get("results") or [], state["topic"], min_score=1
        )
        return {**payload, "results": papers[:8]}

    def _memory_recall(q: str) -> list[dict[str, Any]]:
        bundle = store.recall_bundle(state["topic"])
        items = list(bundle.get("flat") or [])
        rejected = [
            {
                "type": "rejected_gap",
                "description": g.get("description"),
                "reason": g.get("reason"),
                "gap_id": g.get("gap_id"),
                "status": "REJECTED",
            }
            for g in (bundle.get("rejected_gaps") or [])
        ]
        merged = rejected + [x for x in items if x.get("status") != "REJECTED"]
        if not q:
            return merged
        ql = q.lower()
        ranked = [it for it in merged if ql in json_dumps_safe(it).lower()]
        return ranked or merged

    # verify top 2 gaps this pass via tool-calling agent
    for gap in gaps[:2]:
        if time.time() > deadline:
            termination_reason = "BUDGET_EXCEEDED"
            verification_status = "INSUFFICIENT_EVIDENCE"
            warnings.append("task timeout before finishing all gap verifications")
            break
        if verification_round >= max_vr:
            termination_reason = "BUDGET_EXCEEDED"
            verification_status = "INSUFFICIENT_EVIDENCE"
            warnings.append("max_verification_rounds reached")
            break
        if tool_call_count >= max_tool_calls:
            termination_reason = "BUDGET_EXCEEDED"
            verification_status = "INSUFFICIENT_EVIDENCE"
            break

        gap_id = str(gap.get("gap_id") or "")
        side_key = f"verify_gap:{gap_id}"
        if side_key in executed:
            # Idempotent resume: skip re-verification & re-write
            if gap.get("current_status") in {"KEEP", "REFINED"} and gap not in verified:
                verified.append(gap)
            continue

        iteration += 1
        verification_round += 1
        if iteration > max_iter:
            needs_more = True
            break
        state = emit(
            state,
            f"正在验证 {gap_id}（round={verification_round}/{max_vr}, tools={tool_call_count}/{max_tool_calls}）...",
            stage="gap_verify",
        )
        try:
            decision = run_gap_verify_agent(
                topic=state["topic"],
                gap=gap,
                paper_cards=state.get("paper_cards") or [],
                memory_items=state.get("research_memory") or [],
                rag=rag,
                search_fn=_search,
                memory_recall_fn=_memory_recall,
                max_tool_rounds=max_vr,
                max_tool_calls=max(1, max_tool_calls - tool_call_count),
                max_token_budget=max_token_budget,
                task_deadline_ts=deadline,
                lifecycles=state.get("limitation_lifecycles") or [],
                external_critiques=state.get("external_critiques") or [],
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Tool-calling verify agent failed, fallback: %s", exc)
            decision = _fallback_verify_without_tools(
                state=state,
                gap=gap,
                rag=rag,
                service=service,
                store=store,
            )
            decision["reason"] = f"{decision.get('reason')} (fallback after: {exc})"
            termination_reason = termination_reason or "TOOL_FAILURE"

        runtime_meta = decision.pop("_runtime", {}) or {}
        tool_call_count += int(runtime_meta.get("tool_call_count") or 0)
        token_usage += int(runtime_meta.get("token_usage") or 0)
        tool_traces.extend(runtime_meta.get("tool_traces") or decision.get("tool_trace") or [])
        visited.extend(runtime_meta.get("visited_actions") or [])
        for k, v in (runtime_meta.get("retry_counts") or {}).items():
            retry_counts[k] = retry_counts.get(k, 0) + int(v)
        warnings.extend(runtime_meta.get("warnings") or [])
        if runtime_meta.get("termination_reason"):
            termination_reason = runtime_meta["termination_reason"]
        if runtime_meta.get("verification_status"):
            verification_status = runtime_meta["verification_status"]

        for step in decision.get("tool_trace") or []:
            name = step.get("tool_name") or step.get("tool")
            n = step.get("result_count", step.get("n"))
            state = append_event(
                state,
                {
                    "type": "tool_completed" if step.get("status") in {None, "success", "empty"} else "tool_failed",
                    "tool_name": name,
                    "trace": step,
                    "message": f"tool:{name} → n={n}",
                    "stage": "gap_verify",
                },
            )

        fresh_papers = decision.pop("_fresh_papers", []) or []
        rag_hits = decision.pop("_rag_hits", []) or []
        state["rag_hits"] = rag_hits or rag.retrieve(gap.get("description") or "", top_k=5)

        status = decision.get("status") or "KEEP"
        if status not in {"REJECTED", "REFINED", "KEEP"}:
            status = "KEEP"
            decision["status"] = status
        # Budget / insufficient evidence: never upgrade to strong novelty claims
        if termination_reason in {"BUDGET_EXCEEDED", "INSUFFICIENT_EVIDENCE", "TOOL_FAILURE"}:
            if status == "REJECTED" and not (fresh_papers or rag_hits):
                status = "KEEP"
                decision["status"] = status
                decision["reason"] = (
                    str(decision.get("reason") or "")
                    + " [downgraded: insufficient evidence for REJECTED]"
                )
            evidence_status = "insufficient"
            verification_status = termination_reason if termination_reason != "COMPLETED" else "INSUFFICIENT_EVIDENCE"

        gap["current_status"] = status
        if status == "REFINED" and decision.get("refined_description"):
            gap["description"] = decision["refined_description"]
        gap["verification"] = decision
        results.append({**decision, "iteration": iteration, "verification_round": verification_round})

        save_key = f"save_gap:{gap_id}:{status}"
        if save_key not in executed and (
            not task_id or task_store.claim_side_effect(task_id, save_key, "save_gap")
        ):
            store.save_gap(
                state["topic"],
                gap_id=gap_id,
                description=str(gap.get("description")),
                status=status,
                reason=str(decision.get("reason") or ""),
                payload=gap,
            )
            executed.append(save_key)
        executed.append(side_key)

        if status in {"KEEP", "REFINED"}:
            verified.append(gap)
        state = emit(
            state,
            f"Gap {gap_id} => {status}: {str(decision.get('reason'))[:120]}",
            stage="gap_verify",
        )
        if fresh_papers:
            state = emit(
                state,
                f"Agent 检索补充论文 {len(fresh_papers)} 篇（tool=search_papers）",
                stage="gap_verify",
            )

    if not termination_reason:
        termination_reason = "COMPLETED"
        verification_status = verification_status if verification_status != "pending" else "COMPLETED"
        evidence_status = evidence_status if evidence_status != "unknown" else "partial"

    state["iteration_count"] = iteration
    state["verification_round"] = verification_round
    state["tool_call_count"] = tool_call_count
    state["token_usage"] = token_usage
    state["tool_traces"] = tool_traces
    state["visited_actions"] = visited
    state["retry_counts"] = retry_counts
    state["warnings"] = warnings
    state["executed_side_effects"] = executed
    state["termination_reason"] = termination_reason
    state["verification_status"] = verification_status
    state["evidence_status"] = evidence_status
    state["gap_verification_results"] = results
    state["verified_gaps"] = verified
    state["candidate_gaps"] = gaps
    state["needs_more_evidence"] = needs_more or (
        bool(verified) and iteration < max_iter and not state.get("cross_domain_methods")
    )
    state = append_event(
        state,
        {
            "type": "artifact",
            "artifact": "gap_verification",
            "data": results,
        },
    )
    return state


def _fallback_verify_without_tools(
    *,
    state: ResearchState,
    gap: dict[str, Any],
    rag: PaperRAG,
    service: AcademicSearchService,
    store: ResearchMemoryStore,
) -> dict[str, Any]:
    """Deterministic fallback if tool-calling agent cannot run."""

    ctx_builder = ContextBuilder(rag)
    raw_queries = gap.get("verification_queries") or _gap_queries(
        state["topic"], gap.get("description") or ""
    )
    queries = _normalize_search_queries(raw_queries, state["topic"])
    fresh_papers: list[dict[str, Any]] = []
    for q in queries[:2]:
        store.save_query(state["topic"], q, purpose=f"verify:{gap.get('gap_id')}")
        payload = service.search(q, backend="academic", max_results=5)
        fresh_papers.extend(payload.get("results") or [])
    fresh_papers = _filter_relevant_papers(fresh_papers, state["topic"], min_score=1)[:8]
    context = ctx_builder.build_for_gap(
        gap=gap,
        paper_cards=state.get("paper_cards") or [],
        memory_items=state.get("research_memory") or [],
        search_results=fresh_papers,
    )
    decision = {
        "gap_id": gap.get("gap_id"),
        "status": "KEEP",
        "reason": "No clearly resolving work found in current retrieval scope.",
        "closest_existing_work": [p.get("title") for p in fresh_papers[:3]],
        "tool_trace": [{"tool": "fallback_fixed_search", "n": len(fresh_papers)}],
        "_fresh_papers": fresh_papers,
        "_rag_hits": rag.retrieve(gap.get("description") or "", top_k=5),
    }
    try:
        llm = build_llm(Configuration.from_env())
        raw = llm_chat(
            llm,
            system=(
                "You verify candidate research gaps. Return STRICT JSON: "
                "{status: REJECTED|REFINED|KEEP, reason, refined_description?, "
                "closest_existing_work: string[]}."
            ),
            user=context,
        )
        data = extract_json(raw)
        if isinstance(data, dict) and data.get("status") in {"REJECTED", "REFINED", "KEEP"}:
            decision.update(data)
    except Exception as exc:  # noqa: BLE001
        decision["reason"] += f" (LLM fallback: {exc})"
    return decision


def json_dumps_safe(obj: Any) -> str:
    import json

    try:
        return json.dumps(obj, ensure_ascii=False)
    except TypeError:
        return str(obj)


def node_cross_domain(state: ResearchState) -> ResearchState:
    state = emit(
        state,
        "Cross-domain Explorer：抽象问题并检索相邻领域方法...",
        stage="cross_domain",
    )
    keep_gaps = [
        g
        for g in (state.get("verified_gaps") or state.get("candidate_gaps") or [])
        if g.get("current_status") in {"KEEP", "REFINED", "CANDIDATE"}
    ][:2]
    service = _search_service()
    store = ResearchMemoryStore()
    methods: list[dict[str, Any]] = []

    for gap in keep_gaps:
        abstraction = {
            "original_problem": gap.get("description"),
            "abstract_problem": "How to track security semantics propagating across components/systems?",
            "candidate_domains": [
                "program analysis",
                "taint analysis",
                "Android inter-component analysis",
                "API misuse detection",
            ],
        }
        try:
            llm = build_llm(Configuration.from_env())
            raw = llm_chat(
                llm,
                system=(
                    "Abstract a domain-specific research gap into a transferable problem. "
                    "Return STRICT JSON: {abstract_problem, domains: string[], queries: string[]}."
                ),
                user=f"Topic: {state['topic']}\nGap: {gap.get('description')}",
            )
            data = extract_json(raw)
            if isinstance(data, dict):
                abstraction["abstract_problem"] = data.get("abstract_problem") or abstraction[
                    "abstract_problem"
                ]
                abstraction["candidate_domains"] = data.get("domains") or abstraction[
                    "candidate_domains"
                ]
                queries = data.get("queries") or []
            else:
                queries = []
        except Exception:
            queries = [
                f"{abstraction['abstract_problem']} program analysis",
                "inter-component taint analysis vulnerability",
            ]

        if not queries:
            queries = [
                f"{d} {abstraction['abstract_problem']}"
                for d in abstraction["candidate_domains"][:3]
            ]

        related_papers = []
        for q in _normalize_search_queries(queries, state["topic"])[:2]:
            store.save_query(state["topic"], q, purpose=f"cross_domain:{gap.get('gap_id')}")
            payload = service.search(q, backend="academic", max_results=5)
            related_papers.extend(payload.get("results") or [])

        transfer = {
            "gap_id": gap.get("gap_id"),
            "abstraction": abstraction,
            "related_papers": [
                {"title": p.get("title"), "year": p.get("year"), "url": p.get("url")}
                for p in related_papers[:6]
            ],
            "transferability": "Medium",
            "transferable_parts": ["dependency/semantic tracking ideas"],
            "required_adaptation": [
                "Map component boundaries to bridge contracts / message relays"
            ],
            "risks": [
                "Domain assumptions may not hold for adversarial cross-chain settings"
            ],
            "feasibility": "needs human verification",
        }
        try:
            llm = build_llm(Configuration.from_env())
            raw = llm_chat(
                llm,
                system=(
                    "Assess cross-domain method transferability. Return STRICT JSON with keys: "
                    "transferability (High|Medium|Low), transferable_parts, required_adaptation, "
                    "risks, feasibility. Do NOT say 'use GNN because others used GNN'."
                ),
                user=str(transfer),
            )
            data = extract_json(raw)
            if isinstance(data, dict):
                transfer.update(data)
        except Exception as exc:
            logger.debug("transfer LLM skipped: %s", exc)

        methods.append(transfer)
        state = emit(
            state,
            f"跨域探索完成：{gap.get('gap_id')} transferability={transfer.get('transferability')}",
            stage="cross_domain",
        )

    state["cross_domain_methods"] = methods
    state["needs_more_evidence"] = False
    state = append_event(
        state,
        {"type": "artifact", "artifact": "cross_domain", "data": methods},
    )
    return state


def node_finalize_candidates(state: ResearchState) -> ResearchState:
    verified = state.get("verified_gaps") or []
    cross = {m.get("gap_id"): m for m in state.get("cross_domain_methods") or []}
    finals = []
    for gap in verified:
        if gap.get("current_status") == "REJECTED":
            continue
        cd = cross.get(gap.get("gap_id")) or {}
        finals.append(
            {
                "title": f"Candidate direction around: {str(gap.get('description'))[:80]}",
                "research_problem": gap.get("description"),
                "gap": gap.get("description"),
                "supporting_evidence": gap.get("supporting_evidence"),
                "closest_existing_work": (gap.get("verification") or {}).get(
                    "closest_existing_work"
                )
                or [],
                "what_existing_work_has_solved": (gap.get("verification") or {}).get("reason"),
                "what_remains_unsolved": gap.get("description"),
                "cross_domain_inspiration": cd.get("abstraction"),
                "potential_method": cd.get("transferable_parts"),
                "why_it_may_work": cd.get("feasibility"),
                "difference_from_existing_work": "Requires adaptation to cross-chain bridge semantics",
                "novelty_verification": gap.get("current_status"),
                "confidence": gap.get("confidence") or "low",
                "feasibility": cd.get("feasibility") or "unknown",
                "risks": cd.get("risks") or [],
                "transferability": cd.get("transferability"),
                "human_verification_needed": True,
                "disclaimer": (
                    "在当前检索到的文献范围内暂未发现直接解决方案，"
                    "不代表该方向在全球范围内绝对不存在已有工作。"
                ),
            }
        )
    state["final_candidates"] = finals
    state = emit(
        state,
        f"形成 {len(finals)} 个 Candidate Research Directions（均需人工复核）",
        stage="finalize",
    )
    return state


def node_reporter(state: ResearchState) -> ResearchState:
    state = emit(state, "Report Generator：输出证据链研究报告...", stage="report")
    from gap_discovery.safety import (
        known_evidence_index,
        strip_unsupported_novelty,
        validate_citations,
    )

    topic = state["topic"]
    cards = state.get("paper_cards") or []
    ft_n = sum(1 for c in cards if c.get("fulltext_status") == "FULLTEXT")
    abs_n = sum(1 for c in cards if c.get("fulltext_status") != "FULLTEXT")
    known_ids, known_titles = known_evidence_index(dict(state))
    warnings = list(state.get("warnings") or [])
    term = state.get("termination_reason")
    vstatus = state.get("verification_status")

    # Validate citations appearing in gap verification / final candidates
    claimed_ids: list[str] = []
    claimed_titles: list[str] = []
    for gap in state.get("verified_gaps") or state.get("candidate_gaps") or []:
        claimed_ids.extend([str(x) for x in (gap.get("supporting_papers") or []) if x])
        ver = gap.get("verification") or {}
        for t in ver.get("closest_existing_work") or []:
            claimed_titles.append(str(t))
    cite_check = validate_citations(
        claimed_ids=claimed_ids,
        claimed_titles=claimed_titles,
        known_paper_ids=known_ids,
        known_titles=known_titles,
    )
    if cite_check["invalid_paper_ids"] or cite_check["invalid_titles"]:
        warnings.append(
            "dropped unverifiable citations: "
            f"invalid_ids={len(cite_check['invalid_paper_ids'])} "
            f"invalid_titles={len(cite_check['invalid_titles'])}"
        )
        invalid_id_set = {str(x) for x in cite_check["invalid_paper_ids"]}
        invalid_title_set = {t.lower().strip() for t in cite_check["invalid_titles"]}
        for gap in list(state.get("verified_gaps") or []) + list(state.get("candidate_gaps") or []):
            if not isinstance(gap, dict):
                continue
            gap["supporting_papers"] = [
                pid for pid in (gap.get("supporting_papers") or []) if str(pid) not in invalid_id_set
            ]
            ver = gap.get("verification") or {}
            works = [
                t
                for t in (ver.get("closest_existing_work") or [])
                if str(t).lower().strip() not in invalid_title_set
            ]
            ver["closest_existing_work"] = works
            gap["verification"] = ver

    lines = [
        "# Research Gap Discovery Report",
        "",
        "> 本系统用于**帮助研究者发现和验证候选 Research Gap**，不宣称自动发现真正创新点。",
        "> KEEP/REFINED 仅相对当前检索范围与证据链成立，需人工复核。",
        "",
        f"## 0. Run Control\n"
        f"- task_id: `{state.get('task_id')}`\n"
        f"- thread_id: `{state.get('thread_id')}`\n"
        f"- verification_status: **{vstatus}**\n"
        f"- termination_reason: **{term}**\n"
        f"- tool_calls: {state.get('tool_call_count')}/{state.get('max_tool_calls')}\n"
        f"- verification_rounds: {state.get('verification_round')}/{state.get('max_verification_rounds')}\n"
        f"- evidence_status: {state.get('evidence_status')}",
        "",
        f"## 1. Research Landscape / Topic\n- Topic: {topic}",
        f"- Papers: {len(cards)} (fulltext={ft_n}, abstract_only/failed={abs_n})",
        "",
        "## 2. Method Taxonomy",
    ]
    if term in {"BUDGET_EXCEEDED", "INSUFFICIENT_EVIDENCE", "TOOL_FAILURE"}:
        lines.insert(
            4,
            "> ⚠️ 验证未充分完成或证据不足：下文仅保留已获取证据，**不得**解读为已证明创新或全球首次。",
        )
    analysis = state.get("analysis") or {}
    for g in analysis.get("method_groups") or []:
        lines.append(f"- **{g.get('name')}**: {g.get('summary')}")

    lines.extend(["", "## 3. Research Evolution"])
    for edge in state.get("evolution_chain") or []:
        lines.append(
            f"- `{edge.get('from_paper_id')}` → `{edge.get('to_paper_id')}` "
            f"**{edge.get('relationship')}**\n"
            f"  - previous_limitation: {edge.get('previous_limitation')}\n"
            f"  - what_changed: {edge.get('what_new_work_changed')}\n"
            f"  - remaining: {edge.get('remaining_problem')}\n"
            f"  - evidence: {edge.get('evidence')}"
        )
    if not state.get("evolution_chain"):
        lines.append("- （当前证据不足以形成稳健演进边；见 Evidence Boundary）")

    lines.extend(["", "## 4. Limitation Lifecycle"])
    for life in state.get("limitation_lifecycles") or []:
        lines.append(
            f"### {life.get('limitation_id')}: {life.get('description')}\n"
            f"- first_reported_by: {life.get('first_reported_by')}\n"
            f"- status: **{life.get('current_status')}**\n"
            f"- remaining_problem: {life.get('remaining_problem')}\n"
            f"- supporting_papers: {life.get('supporting_papers')}\n"
            f"- subsequent_attempts: {life.get('subsequent_attempts')}\n"
            f"- confidence: {life.get('confidence')}"
        )

    lines.extend(["", "## 5. External Critiques"])
    critiques = state.get("external_critiques") or []
    if not critiques:
        lines.append("- 无（citation context 未定位或 citing fulltext 不可用 → 不生成虚构批评）")
    for c in critiques:
        lines.append(
            f"- target=`{c.get('target_paper_id')}` ← citing=`{c.get('citing_paper_title')}` "
            f"({c.get('citing_year')}) type=**{c.get('critique_type')}**\n"
            f"  - summary: {c.get('critique_summary')}\n"
            f"  - context: {(c.get('citation_context') or '')[:220]}\n"
            f"  - fulltext_status: {c.get('fulltext_status')}"
        )

    lines.extend(["", "## 6. Still-open / Partially-solved Problems"])
    for life in state.get("limitation_lifecycles") or []:
        if life.get("current_status") in {"STILL_OPEN", "PARTIALLY_SOLVED"}:
            lines.append(
                f"- **{life.get('current_status')}**: {life.get('remaining_problem') or life.get('description')}"
            )

    lines.extend(["", "## 7. Candidate Research Gaps"])
    for gap in state.get("verified_gaps") or state.get("candidate_gaps") or []:
        ver = gap.get("verification") or {}
        support = [
            pid
            for pid in (gap.get("supporting_papers") or [])
            if str(pid) in known_ids or not known_ids
        ]
        lines.append(
            f"### {gap.get('gap_id')} — {gap.get('current_status')}\n"
            f"- description: {gap.get('description')}\n"
            f"- first_reported_by: {gap.get('first_reported_by')}\n"
            f"- lifecycle_status: {gap.get('lifecycle_status')}\n"
            f"- subsequent_attempts: {gap.get('existing_attempts')}\n"
            f"- remaining_problem: {gap.get('remaining_problem')}\n"
            f"- self_reported_count: {gap.get('self_reported_count')}\n"
            f"- external_critique_count: {gap.get('external_critique_count')}\n"
            f"- supporting_papers: {support}\n"
            f"- closest_existing_work: {ver.get('closest_existing_work')}\n"
            f"- verify_reason: {ver.get('reason')}\n"
            f"- tool_trace_count: {len(ver.get('tool_trace') or [])}\n"
            f"- why_candidate: multi-source / lifecycle-backed within current evidence scope\n"
            f"- human_verification_needed: true"
        )

    lines.extend(["", "## 8. Cross-domain Inspiration"])
    for m in state.get("cross_domain_methods") or []:
        lines.append(
            f"- gap={m.get('gap_id')} transferability={m.get('transferability')} "
            f"abstract={((m.get('abstraction') or {}).get('abstract_problem'))}"
        )

    lines.extend(["", "## 9. Candidate Research Directions"])
    for cand in state.get("final_candidates") or []:
        lines.append(f"### {cand.get('title')}")
        lines.append(f"- gap: {cand.get('gap')}")
        lines.append(f"- status: {cand.get('novelty_verification')}")
        lines.append(f"- confidence: {cand.get('confidence')}")
        lines.append(f"- disclaimer: {cand.get('disclaimer')}")

    lines.extend(
        [
            "",
            "## 10. Evidence Boundary",
            f"- fulltext papers: {ft_n}",
            f"- abstract_only/failed: {abs_n}",
            f"- external critiques extracted: {len(critiques)}",
            f"- evolution edges: {len(state.get('evolution_chain') or [])}",
            f"- limitation lifecycles: {len(state.get('limitation_lifecycles') or [])}",
            f"- warnings: {warnings[:8]}",
            "- ai_inferred claims are labeled and never presented as author statements",
            "- no paywall bypass; OA PDF only",
            "- KEEP ≠ global novelty",
        ]
    )

    report = "\n".join(lines)
    report, novelty_warns = strip_unsupported_novelty(report)
    warnings.extend(novelty_warns)
    try:
        if os.getenv("GAP_SKIP_REPORT_LLM", "").lower() in {"1", "true", "yes"}:
            raise RuntimeError("report polish skipped by env")
        llm = build_llm(Configuration.from_env())
        polished = llm_chat(
            llm,
            system=(
                "Polish the research gap report in Chinese markdown. Keep section numbering 0-10. "
                "Keep all evidence disclaimers. Do not invent papers or critiques. "
                "Never claim 全球首次 or proven novelty."
            ),
            user=report[:14000],
        )
        if polished and len(polished) > 200:
            polished, more_warns = strip_unsupported_novelty(polished)
            warnings.extend(more_warns)
            report = polished
    except Exception as exc:
        logger.debug("report polish skipped: %s", exc)

    state["warnings"] = warnings
    state["final_report"] = report
    if state.get("status") != "paused":
        state["status"] = "completed"
        if not state.get("termination_reason"):
            state["termination_reason"] = "COMPLETED"
    state = append_event(state, {"type": "report", "report_markdown": report})
    state = emit(state, "研究报告生成完成。", stage="completed", percentage=100)
    state = append_event(state, {"type": "done"})
    state = append_event(state, {"type": "task_completed", "termination_reason": state.get("termination_reason")})
    return state


def route_after_verify(state: ResearchState) -> str:
    """Conditional edge target after verification."""

    if state.get("needs_more_evidence") and int(state.get("iteration_count") or 0) < int(
        state.get("max_iterations") or 6
    ):
        # if we have KEEP/REFINED gaps and no cross-domain yet → cross domain
        if not state.get("cross_domain_methods"):
            return "cross_domain"
        return "finalize"
    if state.get("verified_gaps") and not state.get("cross_domain_methods"):
        return "cross_domain"
    return "finalize"


def _normalize_search_queries(queries: list[Any], topic: str) -> list[str]:
    """LLM often returns long questions; keep short keyword queries only."""

    from services.academic_search import sanitize_academic_query

    seed = _to_academic_query(topic)
    out: list[str] = []
    for q in queries or []:
        text = sanitize_academic_query(str(q), max_len=100)
        # Reject leftover interrogative / too-long NL that still looks like a question
        words = text.split()
        if len(words) > 14 or text.lower().startswith(("does ", "is ", "are ", "how ")):
            text = seed
        if text and text not in out:
            out.append(text)
    if not out:
        out = _gap_queries(topic, "")
    return out


def _gap_queries(topic: str, description: str) -> list[str]:
    seed = _to_academic_query(topic)
    # Prefer short keyword fragments from description, not full sentences
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", description or "")
    focus = " ".join(tokens[:8]) if tokens else "evaluation limitation"
    return [
        f"{seed} {focus}".strip()[:100],
        f"{seed} open problem survey",
        "cross-chain bridge vulnerability detection benchmark",
    ]


def _similar(a: str, b: str) -> bool:
    ta = set(re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]{2,}", a.lower()))
    tb = set(re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]{2,}", b.lower()))
    if not ta or not tb:
        return False
    inter = len(ta & tb)
    return inter / max(len(ta), 1) > 0.5 or inter / max(len(tb), 1) > 0.5
