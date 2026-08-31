"""Gap Verification Tool-Calling Agent (LangGraph ReAct) with budgets & traces."""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

ALLOWED_TOOLS = {
    "search_papers",
    "recall_memory",
    "retrieve_rag",
    "find_citing_papers",
    "get_citation_context",
    "read_fulltext_section",
}


def _compress_agent_messages(
    messages: list[Any],
    *,
    keep_recent: int = 6,
    max_tool_chars: int = 700,
) -> list[Any]:
    """Short-term context compression: keep system/human, shrink old tool payloads.

    Not LLM summarization — deterministic truncate of older ToolMessage bodies so
    the verify loop stays under practical context limits.
    """

    from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

    if len(messages) <= keep_recent + 2:
        # Still trim oversized tool bodies
        out = []
        for m in messages:
            if isinstance(m, ToolMessage) and len(str(m.content or "")) > max_tool_chars:
                out.append(
                    ToolMessage(
                        content=str(m.content)[: max_tool_chars - 20] + "\n...[truncated]",
                        tool_call_id=m.tool_call_id,
                    )
                )
            else:
                out.append(m)
        return out

    head: list[Any] = []
    middle: list[Any] = []
    for m in messages:
        if isinstance(m, (SystemMessage, HumanMessage)) and len(head) < 2:
            head.append(m)
        else:
            middle.append(m)

    recent = middle[-keep_recent:]
    older = middle[:-keep_recent]
    compressed_older: list[Any] = []
    for m in older:
        if isinstance(m, ToolMessage):
            text = str(m.content or "")
            compressed_older.append(
                ToolMessage(
                    content=(text[:180] + "\n...[compressed early tool]")
                    if len(text) > 180
                    else text,
                    tool_call_id=m.tool_call_id,
                )
            )
        else:
            # Drop bulky early AI tool-call chatter; keep a stub note via short AI if needed
            content = getattr(m, "content", None)
            if content and len(str(content)) > 240:
                try:
                    from langchain_core.messages import AIMessage

                    compressed_older.append(
                        AIMessage(content=str(content)[:200] + "...[compressed]")
                    )
                except Exception:  # noqa: BLE001
                    compressed_older.append(m)
            else:
                compressed_older.append(m)

    trimmed_recent = []
    for m in recent:
        if isinstance(m, ToolMessage) and len(str(m.content or "")) > max_tool_chars:
            trimmed_recent.append(
                ToolMessage(
                    content=str(m.content)[: max_tool_chars - 20] + "\n...[truncated]",
                    tool_call_id=m.tool_call_id,
                )
            )
        else:
            trimmed_recent.append(m)
    return head + compressed_older + trimmed_recent


def _build_chat_model():
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=os.getenv("LLM_MODEL_ID", "qwen-plus"),
        api_key=os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL")
        or "https://dashscope.aliyuncs.com/compatible-mode/v1",
        temperature=0.2,
    )


def run_gap_verify_agent(
    *,
    topic: str,
    gap: dict[str, Any],
    paper_cards: list[dict[str, Any]],
    memory_items: list[dict[str, Any]],
    rag,
    search_fn: Callable[[str], dict[str, Any]],
    memory_recall_fn: Callable[[str], list[dict[str, Any]]],
    max_tool_rounds: int = 4,
    max_tool_calls: Optional[int] = None,
    max_token_budget: Optional[int] = None,
    task_deadline_ts: Optional[float] = None,
    lifecycles: Optional[list[dict[str, Any]]] = None,
    external_critiques: Optional[list[dict[str, Any]]] = None,
) -> dict[str, Any]:
    """Run a ReAct-style tool agent for one candidate gap."""

    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
    from langchain_core.tools import tool

    from gap_discovery.citations import CitationService
    from gap_discovery.safety import partition_prompt_blocks, scan_untrusted_text
    from gap_discovery.tool_runtime import (
        ToolBudgetExceeded,
        ToolLoopDetected,
        ToolRuntime,
        classify_error,
    )

    runtime = ToolRuntime(
        max_tool_calls=int(
            max_tool_calls
            if max_tool_calls is not None
            else os.getenv("GAP_MAX_TOOL_CALLS", "24")
        ),
        allowed_tools=ALLOWED_TOOLS,
    )
    observations: dict[str, Any] = {
        "search_results": [],
        "memory_hits": [],
        "rag_hits": [],
        "citing": [],
        "contexts": [],
    }
    cite = CitationService()
    cards_by_id = {c.get("paper_id"): c for c in paper_cards}
    lifecycles = lifecycles or []
    external_critiques = external_critiques or []
    rewrite_budget = int(os.getenv("GAP_QUERY_REWRITE_BUDGET", "2"))
    rewrites_used = 0
    token_usage = 0
    termination_reason: Optional[str] = None
    verification_status = "KEEP"

    def _deadline_ok() -> bool:
        if task_deadline_ts is None:
            return True
        return time.time() < task_deadline_ts

    @tool
    def search_papers(query: str) -> str:
        """Search academic papers for evidence related to this gap. Use short English keywords."""

        nonlocal rewrites_used

        def _call():
            nonlocal rewrites_used
            q = (query or "").strip()
            if len(q) > 120 or any(ch in q for ch in "{}[];"):
                if rewrites_used >= rewrite_budget:
                    return json.dumps(
                        {
                            "backend": "none",
                            "papers": [],
                            "notices": ["query rewrite budget exceeded"],
                            "error_type": "invalid_query",
                            "suggested_action": "stop",
                        },
                        ensure_ascii=False,
                    )
                rewrites_used += 1
                q = " ".join(q.replace("{", " ").replace("}", " ").split())[:80]
            payload = search_fn(q)
            papers = payload.get("results") or []
            if not papers and rewrites_used < rewrite_budget:
                rewrites_used += 1
                alt = f"{topic} {(gap.get('description') or '')}"[:80]
                payload = search_fn(alt)
                papers = payload.get("results") or []
                payload = {**payload, "notices": list(payload.get("notices") or []) + ["query_rewritten"]}
            observations["search_results"].extend(papers)
            slim = [
                {
                    "title": p.get("title"),
                    "year": p.get("year"),
                    "url": p.get("url"),
                    "snippet": (p.get("content") or p.get("abstract") or "")[:280],
                }
                for p in papers[:5]
            ]
            return json.dumps(
                {
                    "backend": payload.get("backend"),
                    "papers": slim,
                    "notices": payload.get("notices"),
                    "result_count": len(papers),
                    "status": "success" if papers else "empty",
                },
                ensure_ascii=False,
            )

        result, _trace = runtime.execute(
            "search_papers", {"query": query}, _call
        )
        return str(result)

    @tool
    def recall_memory(query: str = "") -> str:
        """Recall historical Research Memory gaps/queries/lifecycles for this topic."""

        def _call():
            items = memory_recall_fn(query or topic)
            observations["memory_hits"] = items
            slim = []
            for item in items[:10]:
                slim.append(
                    {
                        "type": item.get("type"),
                        "status": item.get("status"),
                        "description": (item.get("description") or "")[:220],
                        "reason": (item.get("reason") or "")[:160],
                    }
                )
            life_slim = [
                {
                    "limitation_id": x.get("limitation_id"),
                    "status": x.get("current_status"),
                    "remaining": (x.get("remaining_problem") or "")[:160],
                }
                for x in lifecycles[:5]
            ]
            return json.dumps(
                {"memory": slim, "lifecycles": life_slim, "result_count": len(slim)},
                ensure_ascii=False,
            )

        result, _ = runtime.execute("recall_memory", {"query": query}, _call)
        return str(result)

    @tool
    def retrieve_rag(query: str) -> str:
        """Retrieve Top-K evidence chunks from vector RAG over PaperCards / fulltext sections."""

        def _call():
            hits = rag.retrieve(query or (gap.get("description") or ""), top_k=5)
            observations["rag_hits"] = hits
            slim = [
                {
                    "score": h.get("score"),
                    "title": h.get("title"),
                    "section": h.get("section"),
                    "text": (h.get("text") or "")[:320],
                }
                for h in hits
            ]
            return json.dumps(
                {"rag_hits": slim, "result_count": len(slim)}, ensure_ascii=False
            )

        result, _ = runtime.execute("retrieve_rag", {"query": query}, _call)
        return str(result)

    @tool
    def find_citing_papers(paper_id: str = "", title: str = "") -> str:
        """Find subsequent citing/related papers for a target paper (age-aware budget)."""

        def _call():
            card = cards_by_id.get(paper_id) or {}
            t = title or card.get("title") or ""
            papers = cite.find_citing_papers(
                paper_id=paper_id or card.get("paper_id"),
                title=t,
                doi=card.get("doi"),
                year=card.get("year"),
            )
            observations["citing"].extend(papers)
            slim = [
                {
                    "paper_id": p.get("paper_id"),
                    "title": p.get("title"),
                    "year": p.get("year"),
                    "citation_count": p.get("citation_count"),
                    "relation": p.get("citation_relation"),
                }
                for p in papers[:6]
            ]
            return json.dumps(
                {"citing_papers": slim, "result_count": len(slim)}, ensure_ascii=False
            )

        result, _ = runtime.execute(
            "find_citing_papers", {"paper_id": paper_id, "title": title}, _call
        )
        return str(result)

    @tool
    def get_citation_context(
        target_paper_id: str, citing_paper_id: str = "", citing_title: str = ""
    ) -> str:
        """Extract citation context from a citing paper fulltext; classify CRITIQUE/NEUTRAL/..."""

        def _call():
            target = cards_by_id.get(target_paper_id) or {"paper_id": target_paper_id}
            citing = None
            for p in observations.get("citing") or []:
                if p.get("paper_id") == citing_paper_id or (
                    citing_title and citing_title.lower() in (p.get("title") or "").lower()
                ):
                    citing = p
                    break
            if citing is None:
                citing = {"paper_id": citing_paper_id, "title": citing_title}
            ctx = cite.get_citation_context(target_paper=target, citing_paper=citing)
            if ctx:
                observations["contexts"].append(ctx)
            return json.dumps(ctx or {"critique_type": "UNKNOWN"}, ensure_ascii=False)

        result, _ = runtime.execute(
            "get_citation_context",
            {
                "target_paper_id": target_paper_id,
                "citing_paper_id": citing_paper_id,
                "citing_title": citing_title,
            },
            _call,
        )
        return str(result)

    @tool
    def read_fulltext_section(paper_id: str, section: str = "limitations") -> str:
        """Read a specific fulltext section already indexed for a paper (limitations/method/...)."""

        def _call():
            card = cards_by_id.get(paper_id) or {}
            hits = []
            if hasattr(rag, "retrieve_section"):
                hits = rag.retrieve_section(
                    gap.get("description") or section, section=section, top_k=3
                )
                hits = [h for h in hits if h.get("paper_id") == paper_id] or hits
            if hits:
                texts = [(h.get("text") or "")[:400] for h in hits[:3]]
                for t in texts:
                    if scan_untrusted_text(t):
                        runtime.warnings.append(
                            "untrusted fulltext contained control-like phrases (ignored as instructions)"
                        )
                return json.dumps(
                    {
                        "paper_id": paper_id,
                        "section": section,
                        "snippets": texts,
                        "fulltext_status": card.get("fulltext_status"),
                        "result_count": len(hits),
                    },
                    ensure_ascii=False,
                )
            refs = [
                r
                for r in (card.get("evidence_refs") or [])
                if isinstance(r, dict) and (r.get("section") or "") == section
            ]
            return json.dumps(
                {
                    "paper_id": paper_id,
                    "section": section,
                    "snippets": [r.get("quote_or_summary") for r in refs[:3]],
                    "fulltext_status": card.get("fulltext_status") or "ABSTRACT_ONLY",
                    "note": "no section chunks; returning evidence_refs if any",
                    "result_count": len(refs),
                },
                ensure_ascii=False,
            )

        result, _ = runtime.execute(
            "read_fulltext_section", {"paper_id": paper_id, "section": section}, _call
        )
        return str(result)

    tools = [
        search_papers,
        recall_memory,
        retrieve_rag,
        find_citing_papers,
        get_citation_context,
        read_fulltext_section,
    ]
    tool_map = {t.name: t for t in tools}

    system_base = (
        "You are the Gap Verification Agent for a research assistant.\n"
        "Goal: decide if a candidate Research Gap is REJECTED, REFINED, or KEEP "
        "within the CURRENT retrieval scope (never claim global absolute novelty).\n\n"
        "Tools (whitelist only):\n"
        "- search_papers\n- recall_memory\n- retrieve_rag\n"
        "- find_citing_papers\n- get_citation_context\n- read_fulltext_section\n\n"
        "Prefer evidence chain: self-reported limitations → subsequent attempts → "
        "citation critiques → decide remaining openness.\n"
        "If evidence is insufficient, prefer KEEP/REFINED with uncertainty — "
        "do NOT assert 全球首次 / proven novelty.\n"
        "When ready, return ONLY strict JSON:\n"
        '{"status":"REJECTED|REFINED|KEEP","reason":"...","refined_description":"...",'
        '"closest_existing_work":["..."]}\n'
        "REJECTED only if evidence shows sufficiently solved; otherwise REFINED/KEEP."
    )

    evidence_blobs = [
        str(gap.get("description") or ""),
        json.dumps(memory_items[:5], ensure_ascii=False, default=str)[:2000],
    ]
    semantic_bits = [
        m.get("rule_text") or m
        for m in memory_items
        if isinstance(m, dict) and m.get("type") == "semantic_lesson"
    ][:5]
    procedure_bits = [
        m
        for m in memory_items
        if isinstance(m, dict) and m.get("type") == "procedure"
    ][:2]
    parts = partition_prompt_blocks(
        system=system_base,
        user_request=(
            f"Topic: {topic}\n"
            f"Gap ID: {gap.get('gap_id')}\n"
            f"Gap description: {gap.get('description')}\n"
            f"lifecycle_status: {gap.get('lifecycle_status')}\n"
            f"first_reported_by: {gap.get('first_reported_by')}\n"
            f"remaining_problem: {gap.get('remaining_problem')}\n"
            f"Suggested keyword seeds: {gap.get('verification_queries') or []}\n"
            f"Supporting paper ids: {gap.get('supporting_papers') or []}\n"
            f"Known external critiques: {len(external_critiques)}\n"
            f"PaperCard count: {len(paper_cards)}\n"
            f"Semantic lessons: {json.dumps(semantic_bits, ensure_ascii=False)[:1200]}\n"
            f"Procedural SOP hints: {json.dumps(procedure_bits, ensure_ascii=False, default=str)[:900]}\n"
            "Start by deciding which tool(s) to call. Prefer recall_memory then retrieve_rag."
        ),
        evidence_blocks=evidence_blobs,
    )
    if parts.get("injection_flags"):
        runtime.warnings.append(f"injection_flags:{parts['injection_flags']}")

    llm = _build_chat_model().bind_tools(tools)
    messages: list[Any] = [
        SystemMessage(content=parts["system"]),
        HumanMessage(content=parts["user"]),
    ]

    decision: Optional[dict[str, Any]] = None
    max_rounds = max(1, int(os.getenv("GAP_VERIFY_TOOL_ROUNDS", str(max_tool_rounds))))
    # Stay below typical LangGraph recursion limits for nested agent loops
    max_rounds = min(max_rounds, int(os.getenv("GAP_AGENT_RECURSION_CAP", "12")))
    rounds_used = 0

    for round_i in range(max_rounds):
        rounds_used = round_i + 1
        if not _deadline_ok():
            termination_reason = "BUDGET_EXCEEDED"
            verification_status = "INSUFFICIENT_EVIDENCE"
            runtime.warnings.append("task deadline reached during verify agent")
            break
        if runtime.empty_streak >= runtime.max_empty_streak:
            termination_reason = "INSUFFICIENT_EVIDENCE"
            verification_status = "INSUFFICIENT_EVIDENCE"
            break
        if max_token_budget is not None and token_usage >= max_token_budget:
            termination_reason = "BUDGET_EXCEEDED"
            verification_status = "INSUFFICIENT_EVIDENCE"
            break

        runtime.begin_round()
        try:
            if os.getenv("GAP_COMPRESS_VERIFY_MESSAGES", "true").lower() in {
                "1",
                "true",
                "yes",
            }:
                messages = _compress_agent_messages(messages)
            ai: AIMessage = llm.invoke(messages)
        except Exception as exc:  # noqa: BLE001
            err = classify_error(exc)
            runtime.warnings.append(f"llm_error:{err['error_type']}")
            termination_reason = "TOOL_FAILURE"
            verification_status = "INSUFFICIENT_EVIDENCE"
            break

        messages.append(ai)
        usage = getattr(ai, "usage_metadata", None) or {}
        if isinstance(usage, dict):
            token_usage += int(usage.get("total_tokens") or usage.get("output_tokens") or 0)

        tool_calls = getattr(ai, "tool_calls", None) or []
        if tool_calls:
            for call in tool_calls:
                name = call.get("name") if isinstance(call, dict) else call["name"]
                args = call.get("args") if isinstance(call, dict) else call["args"]
                call_id = call.get("id") if isinstance(call, dict) else call["id"]
                tool_fn = tool_map.get(name)
                if tool_fn is None or name not in ALLOWED_TOOLS:
                    runtime.warnings.append(f"rejected unauthorized tool: {name}")
                    result = json.dumps(
                        {
                            "error_type": "unauthorized_tool",
                            "message": f"tool {name} not allowed",
                            "retryable": False,
                            "suggested_action": "stop",
                        }
                    )
                    messages.append(ToolMessage(content=str(result), tool_call_id=call_id))
                    termination_reason = "TOOL_FAILURE"
                    continue
                try:
                    # Tools already wrap ToolRuntime.execute internally
                    result = tool_fn.invoke(args or {})
                except ToolBudgetExceeded:
                    termination_reason = "BUDGET_EXCEEDED"
                    verification_status = "INSUFFICIENT_EVIDENCE"
                    result = json.dumps(
                        {
                            "error_type": "budget_exceeded",
                            "message": "max tool calls reached",
                            "retryable": False,
                            "suggested_action": "stop",
                        }
                    )
                    messages.append(ToolMessage(content=str(result), tool_call_id=call_id))
                    break
                except ToolLoopDetected as exc:
                    termination_reason = "BUDGET_EXCEEDED"
                    verification_status = "INSUFFICIENT_EVIDENCE"
                    runtime.warnings.append(str(exc))
                    result = json.dumps(
                        {
                            "error_type": "duplicate_call",
                            "message": str(exc),
                            "retryable": False,
                            "suggested_action": "stop",
                        }
                    )
                    messages.append(ToolMessage(content=str(result), tool_call_id=call_id))
                    break
                except Exception as exc:  # noqa: BLE001
                    err = classify_error(exc)
                    logger.warning("Tool %s failed: %s", name, exc)
                    if err.get("suggested_action") == "stop":
                        termination_reason = "TOOL_FAILURE"
                    result = json.dumps(err, ensure_ascii=False)
                messages.append(ToolMessage(content=str(result), tool_call_id=call_id))
            if termination_reason in {"BUDGET_EXCEEDED", "TOOL_FAILURE"}:
                break
            continue

        content = ai.content if isinstance(ai.content, str) else str(ai.content)
        try:
            from gap_discovery.llm_utils import extract_json

            data = extract_json(content)
            if isinstance(data, dict) and data.get("status") in {"REJECTED", "REFINED", "KEEP"}:
                decision = data
                verification_status = str(data["status"])
                break
        except Exception:
            messages.append(
                HumanMessage(
                    content=(
                        "Your last reply was not valid decision JSON. "
                        "Either call a tool or return ONLY the decision JSON."
                    )
                )
            )

    if decision is None:
        if termination_reason is None:
            termination_reason = "INSUFFICIENT_EVIDENCE"
            verification_status = "INSUFFICIENT_EVIDENCE"
        decision = {
            "status": "KEEP",
            "reason": (
                "Verification did not fully complete "
                f"(reason={termination_reason}). Retaining as candidate gap within "
                "current retrieval scope; do not treat as proven novelty."
            ),
            "closest_existing_work": [
                p.get("title") for p in (observations.get("search_results") or [])[:3]
            ],
        }
    else:
        if termination_reason is None:
            termination_reason = "COMPLETED"

    # Hard block unsupported novelty language in reason
    from gap_discovery.safety import strip_unsupported_novelty

    reason, novelty_warns = strip_unsupported_novelty(str(decision.get("reason") or ""))
    decision["reason"] = reason
    runtime.warnings.extend(novelty_warns)

    decision["gap_id"] = gap.get("gap_id")
    decision["tool_trace"] = runtime.tool_traces
    decision["observations"] = {
        "search_n": len(observations.get("search_results") or []),
        "memory_n": len(observations.get("memory_hits") or []),
        "rag_n": len(observations.get("rag_hits") or []),
        "citing_n": len(observations.get("citing") or []),
        "context_n": len(observations.get("contexts") or []),
    }
    decision["_fresh_papers"] = observations.get("search_results") or []
    decision["_rag_hits"] = observations.get("rag_hits") or []
    decision["_runtime"] = {
        **runtime.export_state_fields(),
        "token_usage": token_usage,
        "termination_reason": termination_reason,
        "verification_status": verification_status,
        "verification_rounds_used": rounds_used,
    }
    return decision


def langgraph_react_available() -> bool:
    try:
        from langchain_openai import ChatOpenAI  # noqa: F401
        from langchain_core.tools import tool  # noqa: F401

        return True
    except ImportError:
        return False
