"""LangGraph workflow for interview-oriented Research Agent.

Real path (preferred):
  StateGraph → nodes → conditional edge after gap_verify → cross_domain|finalize → report → END

Fallback (only if langgraph missing): sequential runner with the same routing semantics.
Set REQUIRE_LANGGRAPH=true to fail fast instead of falling back.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Iterator

from gap_discovery import pipeline
from gap_discovery.state import ResearchState, touch

logger = logging.getLogger(__name__)

_COMPILED = None
_ENGINE = "uninitialized"


def _pipeline_steps() -> list[tuple[str, Callable[[ResearchState], ResearchState]]]:
    return [
        ("memory_recall", pipeline.node_recall_memory),
        ("planner", pipeline.node_planner),
        ("search", pipeline.node_search),
        ("paper_reader", pipeline.node_paper_reader),
        ("analyzer", pipeline.node_analyzer),
        ("evidence_chain", pipeline.node_evidence_chain),
        ("gap_discover", pipeline.node_gap_discover),
        ("gap_verify", pipeline.node_gap_verify),
    ]


def langgraph_engine() -> str:
    """Return 'langgraph' | 'sequential' | 'uninitialized'."""

    global _ENGINE
    if _ENGINE == "uninitialized":
        build_gap_discovery_graph()
    return _ENGINE


def build_gap_discovery_graph(*, with_checkpointer: bool = True):
    """Compile LangGraph with verification conditional edge."""

    global _COMPILED, _ENGINE
    if _COMPILED is not None:
        return _COMPILED

    require = os.getenv("REQUIRE_LANGGRAPH", "true").lower() in {"1", "true", "yes"}
    try:
        from langgraph.graph import END, StateGraph
    except ImportError as exc:  # pragma: no cover
        _ENGINE = "sequential"
        if require:
            raise RuntimeError(
                "REQUIRE_LANGGRAPH=true but langgraph is not installed. "
                "pip install langgraph langchain-core"
            ) from exc
        logger.warning("langgraph not installed; using sequential fallback runner")
        return None

    graph = StateGraph(ResearchState)
    for name, fn in _pipeline_steps():
        graph.add_node(name, fn)
    graph.add_node("cross_domain", pipeline.node_cross_domain)
    graph.add_node("finalize", pipeline.node_finalize_candidates)
    graph.add_node("report", pipeline.node_reporter)

    graph.set_entry_point("memory_recall")
    linear = [n for n, _ in _pipeline_steps()]
    for a, b in zip(linear, linear[1:]):
        graph.add_edge(a, b)

    graph.add_conditional_edges(
        "gap_verify",
        pipeline.route_after_verify,
        {
            "cross_domain": "cross_domain",
            "finalize": "finalize",
        },
    )
    graph.add_edge("cross_domain", "finalize")
    graph.add_edge("finalize", "report")
    graph.add_edge("report", END)

    checkpointer = None
    if with_checkpointer and os.getenv("GAP_DISABLE_CHECKPOINTER", "").lower() not in {
        "1",
        "true",
        "yes",
    }:
        try:
            from gap_discovery.checkpointer import get_sqlite_checkpointer

            checkpointer = get_sqlite_checkpointer()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Sqlite checkpointer unavailable: %s", exc)

    _COMPILED = graph.compile(checkpointer=checkpointer) if checkpointer else graph.compile()
    _ENGINE = "langgraph"
    logger.info(
        "LangGraph compiled: memory_recall→…→gap_verify -conditional→ "
        "cross_domain|finalize → report → END (checkpointer=%s)",
        bool(checkpointer),
    )
    return _COMPILED


def reset_compiled_graph() -> None:
    """Test helper: drop cached compiled graph."""

    global _COMPILED, _ENGINE
    _COMPILED = None
    _ENGINE = "uninitialized"


def run_pipeline_sequential(state: ResearchState) -> ResearchState:
    """Execute with explicit verification routing (works without langgraph)."""

    for _phase, _name, state in iter_pipeline(state):
        pass
    return state


def run_pipeline(state: ResearchState) -> ResearchState:
    compiled = build_gap_discovery_graph()
    if compiled is not None and not (state.get("completed_nodes") or []):
        logger.info("run_pipeline via LangGraph engine")
        thread_id = state.get("thread_id") or state.get("task_id") or "default"
        from gap_discovery.checkpointer import invoke_config

        try:
            return compiled.invoke(state, config=invoke_config(str(thread_id)))
        except TypeError:
            return compiled.invoke(state)
    # Resume / partial completion uses sequential path with completed_nodes skip
    return run_pipeline_sequential(state)


def iter_pipeline(state: ResearchState) -> Iterator[tuple[str, str, ResearchState]]:
    """Yield (phase, node_name, state) for SSE.

    phase is 'start' | 'done'.
    Uses the same node functions + conditional routing as LangGraph.
    Skips nodes listed in state['completed_nodes'] for resume.
    """

    logger.info("iter_pipeline traced sequential (same nodes/routing as LangGraph)")
    completed = set(state.get("completed_nodes") or [])

    for name, fn in _pipeline_steps():
        if name in completed:
            continue
        yield "start", name, state
        logger.info("Running node: %s", name)
        state = touch(fn(state), node=name)
        done = list(state.get("completed_nodes") or [])
        if name not in done:
            done.append(name)
        state["completed_nodes"] = done
        yield "done", name, state
        if state.get("status") in {"failed", "paused"}:
            return

    route = pipeline.route_after_verify(state)
    done_set = set(state.get("completed_nodes") or [])
    if route == "cross_domain" and "cross_domain" not in done_set:
        yield "start", "cross_domain", state
        state = touch(pipeline.node_cross_domain(state), node="cross_domain")
        state["completed_nodes"] = list(state.get("completed_nodes") or []) + ["cross_domain"]
        yield "done", "cross_domain", state

    done_set = set(state.get("completed_nodes") or [])
    if "finalize" not in done_set:
        yield "start", "finalize", state
        state = touch(pipeline.node_finalize_candidates(state), node="finalize")
        state["completed_nodes"] = list(state.get("completed_nodes") or []) + ["finalize"]
        yield "done", "finalize", state

    done_set = set(state.get("completed_nodes") or [])
    if "report" not in done_set:
        yield "start", "report", state
        state = touch(pipeline.node_reporter(state), node="report")
        state["completed_nodes"] = list(state.get("completed_nodes") or []) + ["report"]
        yield "done", "report", state


def describe_pipeline() -> list[dict[str, Any]]:
    nodes = [n for n, _ in _pipeline_steps()] + [
        "cross_domain(conditional)",
        "finalize",
        "report",
    ]
    return [{"node": n, "engine": langgraph_engine()} for n in nodes]


GRAPH_NODES = [
    "memory_recall",
    "planner",
    "search",
    "paper_reader",
    "analyzer",
    "evidence_chain",
    "gap_discover",
    "gap_verify",
    "cross_domain",
    "finalize",
    "report",
]

GRAPH_EDGES = [
    {"from": "memory_recall", "to": "planner", "kind": "normal"},
    {"from": "planner", "to": "search", "kind": "normal"},
    {"from": "search", "to": "paper_reader", "kind": "normal"},
    {"from": "paper_reader", "to": "analyzer", "kind": "normal"},
    {"from": "analyzer", "to": "evidence_chain", "kind": "normal"},
    {"from": "evidence_chain", "to": "gap_discover", "kind": "normal"},
    {"from": "gap_discover", "to": "gap_verify", "kind": "normal"},
    {"from": "gap_verify", "to": "cross_domain", "kind": "conditional", "label": "needs cross-domain"},
    {"from": "gap_verify", "to": "finalize", "kind": "conditional", "label": "skip"},
    {"from": "cross_domain", "to": "finalize", "kind": "normal"},
    {"from": "finalize", "to": "report", "kind": "normal"},
    {"from": "report", "to": "END", "kind": "normal"},
]
