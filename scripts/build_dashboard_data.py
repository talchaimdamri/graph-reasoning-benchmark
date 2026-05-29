"""Build figures/ and dashboard/public/results.json.

If a real SQLite results DB is given (``--db path.sqlite``), its results are
used and enriched with the on-disk graphs (``--graph-dir``) and questions
(``--question-dir``) that produced the run, so the Explorer view and the
by_difficulty / by_category metrics populate from real data. Otherwise a
synthetic fixture is generated so the dashboard always has something to render.

Usage::

    python scripts/build_dashboard_data.py            # synthetic fixture
    python scripts/build_dashboard_data.py --db x.db  # real results
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from grb.fixtures import make_full_fixture
from grb.metrics import export_dashboard_json, generate_figures, results_to_frame
from grb.models import BenchGraph, Question

ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = ROOT / "figures"
DASHBOARD_JSON = ROOT / "dashboard" / "public" / "results.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", help="Optional SQLite results DB.")
    parser.add_argument("--run-id", help="Restrict to one run_id (with --db).")
    parser.add_argument(
        "--graph-dir",
        default=str(ROOT / "data" / "graphs"),
        help="Graphs that produced the run (for Explorer + metric enrichment).",
    )
    parser.add_argument(
        "--question-dir",
        default=str(ROOT / "data" / "questions"),
        help="Questions that produced the run (for difficulty/category enrichment).",
    )
    args = parser.parse_args()

    if args.db:
        from grb.storage import Storage

        with Storage(args.db) as store:
            results = store.list_results(args.run_id)
        # Enrich from the on-disk graph/question sets that produced this run so the
        # Explorer view and by_difficulty/by_category metrics populate from real data.
        graphs = [
            BenchGraph.model_validate_json(p.read_text())
            for p in sorted(Path(args.graph_dir).glob("*.json"))
        ]
        questions = [
            Question.model_validate(q)
            for p in sorted(Path(args.question_dir).glob("*.json"))
            for q in json.loads(p.read_text())
        ]
    else:
        fixture = make_full_fixture()
        results = fixture["results"]
        graphs = fixture["graphs"]
        questions = fixture["questions"]

    frame = results_to_frame(results, questions=questions, graphs=graphs)
    figs = generate_figures(frame, FIGURES_DIR)
    export_dashboard_json(DASHBOARD_JSON, results, questions=questions, graphs=graphs)

    print(f"results: {len(results)}")
    print(f"figures: {len(figs)} -> {FIGURES_DIR}")
    print(f"dashboard json -> {DASHBOARD_JSON}")


if __name__ == "__main__":
    main()
