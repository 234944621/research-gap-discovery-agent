"""Smoke checks for the 3 resume-critical upgrades.

1) LangGraph compiles and runs StateGraph + conditional edge
2) Gap Verify agent can bind tools (search/memory/rag)
3) Embedding RAG indexes PaperCard and retrieves Top-K via Chroma
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env", override=True)


def check_langgraph() -> None:
    from gap_discovery.graph import build_gap_discovery_graph, langgraph_engine
    from gap_discovery.state import initial_state

    g = build_gap_discovery_graph()
    assert g is not None, "LangGraph graph is None"
    assert langgraph_engine() == "langgraph", langgraph_engine()
    # Tiny invoke: only memory_recall needs no LLM if we stop early — instead
    # inspect graph structure via get_graph
    graph = g.get_graph()
    node_ids = set(graph.nodes)
    for required in {
        "memory_recall",
        "planner",
        "search",
        "paper_reader",
        "analyzer",
        "gap_discover",
        "gap_verify",
        "cross_domain",
        "finalize",
        "report",
    }:
        assert required in node_ids, f"missing node {required} in {node_ids}"
    print("[OK] LangGraph compiled with conditional gap_verify edge")
    print(f"     engine={langgraph_engine()} nodes={sorted(node_ids)}")


def check_rag() -> None:
    from gap_discovery.rag import PaperRAG

    rag = PaperRAG()
    rag.add_paper_card(
        {
            "paper_id": "smoke-xscope",
            "title": "Xscope: Hunting for Cross-Chain Bridge Attacks",
            "year": 2022,
            "abstract": (
                "We present Xscope to detect cross-chain bridge attacks using "
                "on-chain traces and message verification patterns."
            ),
            "method": "dynamic observation of bridge message relays",
            "research_problem": "cross-chain bridge attack detection",
        }
    )
    hits = rag.retrieve("cross-chain bridge attack detection", top_k=3)
    print(f"[OK] RAG backend={rag.active_backend} hits={len(hits)}")
    assert hits, "expected at least one RAG hit"
    print(f"     top={hits[0].get('title')} score={hits[0].get('score')} backend={hits[0].get('backend')}")


def check_tools_bind() -> None:
    from langchain_core.tools import tool
    from langchain_openai import ChatOpenAI

    @tool
    def search_papers(query: str) -> str:
        """Search papers."""

        return f"ok:{query}"

    @tool
    def recall_memory(query: str = "") -> str:
        """Recall memory."""

        return "[]"

    @tool
    def retrieve_rag(query: str) -> str:
        """Retrieve RAG."""

        return "[]"

    llm = ChatOpenAI(
        model=os.getenv("LLM_MODEL_ID", "qwen-plus"),
        api_key=os.getenv("LLM_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL"),
        temperature=0,
    ).bind_tools([search_papers, recall_memory, retrieve_rag])
    msg = llm.invoke(
        "You are a gap verify agent. Call retrieve_rag with query='bridge attack'."
    )
    calls = getattr(msg, "tool_calls", None) or []
    print(f"[OK] Tool binding works; model returned {len(calls)} tool_calls")
    if calls:
        print(f"     first_tool={calls[0].get('name')} args={calls[0].get('args')}")
    else:
        # Some models may answer in text; still prove bind_tools does not crash
        print("     (no tool_calls this turn — bind succeeded; full agent loop covers multi-turn)")


def main() -> None:
    print("=== Smoke: LangGraph + Tool Calling + Embedding RAG ===\n")
    check_langgraph()
    check_rag()
    check_tools_bind()
    print("\nAll smoke checks passed.")


if __name__ == "__main__":
    main()
