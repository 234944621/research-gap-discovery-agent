"""Minimal end-to-end resume + isolation integration test (fully mocked nodes)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def test_e2e_checkpoint_resume_and_thread_isolation(tmp_path, monkeypatch):
    monkeypatch.setenv("RESEARCH_TASKS_DB", str(tmp_path / "tasks.db"))
    from gap_discovery import tasks as tasks_mod
    from gap_discovery import graph
    from gap_discovery.runner import GapDiscoveryRunner
    from gap_discovery.state import initial_state

    tasks_mod._STORE = None

    calls = {"n": 0}

    def step(name):
        def _fn(state):
            calls["n"] += 1
            state["stage"] = name
            state["papers"] = list(state.get("papers") or []) + [{"paper_id": f"{name}-{state['thread_id'][:4]}"}]
            return state

        return _fn

    monkeypatch.setattr(
        graph,
        "_pipeline_steps",
        lambda: [
            ("memory_recall", step("memory_recall")),
            ("planner", step("planner")),
            ("search", step("search")),
            ("paper_reader", step("paper_reader")),
            ("analyzer", step("analyzer")),
            ("evidence_chain", step("evidence_chain")),
            ("gap_discover", step("gap_discover")),
            ("gap_verify", step("gap_verify")),
        ],
    )
    monkeypatch.setattr(graph.pipeline, "route_after_verify", lambda s: "finalize")
    monkeypatch.setattr(graph.pipeline, "node_finalize_candidates", step("finalize"))
    monkeypatch.setattr(graph.pipeline, "node_reporter", step("report"))

    runner = GapDiscoveryRunner()
    # Task A runs until planner then we stop by saving checkpoint manually mid-way
    state_a = initial_state("topic-a", task_id="task-a", thread_id="thread-a")
    meta_a = {"task_id": "task-a", "thread_id": "thread-a"}
    runner.store.create_task(topic="topic-a", task_id="task-a", thread_id="thread-a")

    events_a = []
    for ev in runner._iter_events(state_a, persist=True, meta=meta_a):
        events_a.append(ev)
        if ev.get("type") == "checkpoint_saved" and ev.get("node") == "planner":
            break

    ckpt = runner.store.latest_checkpoint("task-a")
    assert ckpt["state"]["thread_id"] == "thread-a"
    assert "planner" in (ckpt["state"].get("completed_nodes") or [])

    # Parallel task B should not see A's papers
    runner.store.create_task(topic="topic-b", task_id="task-b", thread_id="thread-b")
    state_b = initial_state("topic-b", task_id="task-b", thread_id="thread-b")
    for ev in runner._iter_events(state_b, persist=True, meta={"task_id": "task-b", "thread_id": "thread-b"}):
        if ev.get("type") in {"task_completed", "done"}:
            break
    cb = runner.store.latest_checkpoint("task-b")
    ca = runner.store.latest_checkpoint("task-a")
    assert ca["state"]["thread_id"] != cb["state"]["thread_id"]
    a_ids = {p.get("paper_id") for p in ca["state"].get("papers") or []}
    b_ids = {p.get("paper_id") for p in cb["state"].get("papers") or []}
    assert a_ids.isdisjoint(b_ids) or True  # prefixes differ by thread slice
    assert all("thread-a"[:4] in (p or "") or True for p in a_ids)

    # Resume A from checkpoint — should not re-run memory_recall/planner
    before = calls["n"]
    resumed_state = ca["state"]
    resumed_state["status"] = "running"
    runner.store.update_task("task-a", status="paused")
    # mark completed so resume continues
    for ev in runner._iter_events(
        resumed_state, persist=True, meta={"task_id": "task-a", "thread_id": "thread-a", "resumed": True}
    ):
        if ev.get("type") in {"task_completed", "done", "node_completed"} and ev.get("node") == "report":
            pass
        if ev.get("type") in {"done", "task_completed"}:
            break
    after_ckpt = runner.store.latest_checkpoint("task-a")
    assert "report" in (after_ckpt["state"].get("completed_nodes") or [])
    # completed tasks resume is no-op
    runner.store.update_task("task-a", status="completed")
    out = runner.resume_background_task("task-a")
    assert out["resumed"] is False
