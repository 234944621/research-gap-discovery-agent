"""Quick smoke test for academic search (run from backend/ with venv)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from services.academic_search import AcademicSearchService


def main() -> None:
    svc = AcademicSearchService(include_citing=False, max_results=3)
    result = svc.search(
        "cross-chain bridge smart contract vulnerability detection",
        backend="academic",
    )
    print("backend:", result["backend"])
    print("notices:", result["notices"])
    print("count:", len(result["results"]))
    for paper in result["results"]:
        print(
            f"- [{paper.get('year')}] {paper.get('title')[:90]} "
            f"| cites={paper.get('citation_count')} "
            f"| evidence={paper.get('evidence_level')}"
        )


if __name__ == "__main__":
    main()
