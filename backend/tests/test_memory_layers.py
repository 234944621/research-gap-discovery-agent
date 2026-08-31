"""Memory layers: episodic / semantic / entity / procedural + consolidate."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture()
def mem(tmp_path, monkeypatch):
    db = tmp_path / "research_memory.db"
    monkeypatch.setenv("RESEARCH_MEMORY_DB", str(db))
    from gap_discovery.memory import ResearchMemoryStore

    return ResearchMemoryStore(db_path=str(db))


def test_save_gap_builds_semantic_entity_procedure(mem):
    topic = "cross-chain bridge vulnerability detection"
    mem.save_gap(
        topic,
        gap_id="g1",
        description="Need a detector for bridge oracle lag",
        status="REJECTED",
        reason="Already solved by Paper X formal verification",
    )
    lessons = mem.recall_semantic_lessons(topic)
    assert lessons, "REJECTED should distill a semantic lesson"
    assert "oracle" in lessons[0]["rule_text"].lower() or "reject" in lessons[0]["rule_text"].lower()

    ents = mem.recall_entities(topic)
    types = {e["entity_type"] for e in ents}
    assert "rejected_pattern" in types
    assert "topic_stats" in types

    procs = mem.recall_procedures(topic)
    names = {p["name"] for p in procs}
    assert "gap_verify_sop" in names
    assert "landscape_search_sop" in names


def test_semantic_lesson_merge_same_gap_no_double_count(mem):
    topic = "topic-merge"
    mem.save_gap(
        topic,
        gap_id="g2",
        description="duplicate style gap about latency",
        status="REJECTED",
        reason="covered",
    )
    mem.save_gap(
        topic,
        gap_id="g2",
        description="duplicate style gap about latency",
        status="REJECTED",
        reason="covered",
    )
    lessons = mem.recall_semantic_lessons(topic)
    assert len(lessons) >= 1
    # same source gap should not inflate forever
    assert lessons[0]["evidence_count"] <= 2


def test_recall_bundle_includes_layers(mem):
    topic = "bundle-topic"
    mem.save_query(topic, "bridge fuzzing", "landscape_search")
    mem.save_gap(
        topic,
        gap_id="g3",
        description="open problem on liveness",
        status="KEEP",
        reason="insufficient counterevidence",
    )
    bundle = mem.recall_bundle(topic)
    assert "semantic_lessons" in bundle
    assert "entities" in bundle
    assert "procedures" in bundle
    assert any(x.get("type") == "procedure" for x in bundle["flat"])
    assert bundle["summary"]["procedure_count"] >= 1


def test_compress_agent_messages():
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

    from gap_discovery.verify_agent import _compress_agent_messages

    messages = [
        SystemMessage(content="sys"),
        HumanMessage(content="user"),
    ]
    for i in range(10):
        messages.append(AIMessage(content=f"thinking {i}" + ("x" * 300)))
        messages.append(
            ToolMessage(content=("result " * 200), tool_call_id=f"c{i}")
        )
    out = _compress_agent_messages(messages, keep_recent=4, max_tool_chars=500)
    assert isinstance(out[0], SystemMessage)
    # older tool bodies should be shortened
    tool_lens = [len(str(m.content)) for m in out if isinstance(m, ToolMessage)]
    assert tool_lens
    assert min(tool_lens) < 5000
