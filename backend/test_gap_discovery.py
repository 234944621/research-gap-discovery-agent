"""End-to-end smoke test for interview-oriented Research Agent."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env", override=True)

from gap_discovery.runner import GapDiscoveryRunner


def main() -> None:
    topic = "跨链桥智能合约漏洞检测"
    # Faster smoke: disable expensive LLM paper enrich if needed via env
    runner = GapDiscoveryRunner()
    print(f"=== Research Agent E2E: {topic} ===\n")
    for event in runner.run_stream(topic):
        etype = event.get("type")
        if etype == "status":
            print(f"[status] {event.get('message')}")
        elif etype == "artifact":
            print(f"[artifact] {event.get('artifact')}")
        elif etype == "node_done":
            print(f"[node_done] {event.get('node')}")
        elif etype == "report":
            report = event.get("report_markdown") or ""
            print("\n===== REPORT (head) =====")
            print("\n".join(report.splitlines()[:60]))
        elif etype == "error":
            print(f"[error] {event.get('detail')}")
        elif etype == "done":
            print("\n[done]")
        elif etype == "pipeline":
            print(f"[pipeline] {event.get('nodes')}")


if __name__ == "__main__":
    main()
