"""Build figures/ and dashboard/public/results.json.

If a real SQLite results DB is given (``--db path.sqlite``), its results are
used (joined to freshly generated questions/graphs is not attempted — real runs
should pass real graphs/questions, so for now real DBs export results-only with
'unknown' enrichment). Otherwise a synthetic fixture is generated so the
dashboard always has something to render.

Usage::

    python scripts/build_dashboard_data.py            # synthetic fixture
    python scripts/build_dashboard_data.py --db x.db  # real results
"""

from __future__ import annotations

import argparse
from pathlib import Path

from grb.fixtures import make_full_fixture
from grb.metrics import export_dashboard_json, generate_figures, results_to_frame

ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = ROOT / "figures"
DASHBOARD_JSON = ROOT / "dashboard" / "public" / "results.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", help="Optional SQLite results DB.")
    parser.add_argument("--run-id", help="Restrict to one run_id (with --db).")
    args = parser.parse_args()

    if args.db:
        from grb.storage import Storage

        with Storage(args.db) as store:
            results = store.list_results(args.run_id)
        graphs: list = []
        questions: list = []
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
