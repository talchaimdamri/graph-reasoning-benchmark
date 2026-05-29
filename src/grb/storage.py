"""SQLite-backed persistence for benchmark runs and results, plus JSON export.

Schema
------
``runs``    one row per benchmark run (a ``run_id`` groups many results).
``results`` one row per (graph, encoding, question, model) evaluation.

The ``results`` table mirrors :class:`grb.models.Result`. A unique index on
``(run_id, graph_id, encoding, question_id, model)`` makes the store
idempotent and enables the pipeline's resume/cache behaviour: re-inserting a
completed cell is a no-op, and :meth:`Storage.get_result` answers the cache.

``ground_truth`` and ``model_answer`` are stored as JSON text so any answer type
(list, bool, number, string) round-trips faithfully.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Iterable, Optional

from grb.models import Result

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id      TEXT PRIMARY KEY,
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL,
    config      TEXT,
    status      TEXT NOT NULL DEFAULT 'running',
    total_calls INTEGER NOT NULL DEFAULT 0,
    total_tokens INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS results (
    result_id     TEXT PRIMARY KEY,
    run_id        TEXT NOT NULL,
    graph_id      TEXT NOT NULL,
    encoding      TEXT NOT NULL,
    question_id   TEXT NOT NULL,
    question_text TEXT NOT NULL,
    ground_truth  TEXT,
    model         TEXT NOT NULL,
    model_answer  TEXT,
    correct       INTEGER NOT NULL,
    tokens_used   INTEGER NOT NULL,
    latency_ms    REAL NOT NULL,
    error         TEXT,
    created_at    REAL NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS ux_results_cell
    ON results (run_id, graph_id, encoding, question_id, model);
"""


def _dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _loads(text: Optional[str]) -> Any:
    if text is None:
        return None
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return text


class Storage:
    """A thin wrapper around a SQLite database file (thread-safe for writes)."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False so a ThreadPoolExecutor can share it; we
        # serialize writes ourselves via the connection's implicit locking.
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    # -- runs ---------------------------------------------------------------

    def create_run(self, run_id: str, config: dict | None = None) -> None:
        """Insert a run row (idempotent: keeps the original created_at)."""
        now = time.time()
        self.conn.execute(
            """
            INSERT INTO runs (run_id, created_at, updated_at, config, status)
            VALUES (?, ?, ?, ?, 'running')
            ON CONFLICT(run_id) DO UPDATE SET updated_at=excluded.updated_at
            """,
            (run_id, now, now, _dumps(config or {})),
        )
        self.conn.commit()

    def update_run_stats(
        self, run_id: str, *, total_calls: int, total_tokens: int, status: str = "running"
    ) -> None:
        self.conn.execute(
            """
            UPDATE runs
               SET total_calls=?, total_tokens=?, status=?, updated_at=?
             WHERE run_id=?
            """,
            (total_calls, total_tokens, status, time.time(), run_id),
        )
        self.conn.commit()

    def get_run(self, run_id: str) -> Optional[dict]:
        row = self.conn.execute(
            "SELECT * FROM runs WHERE run_id=?", (run_id,)
        ).fetchone()
        if row is None:
            return None
        d = dict(row)
        d["config"] = _loads(d.get("config"))
        return d

    # -- results ------------------------------------------------------------

    def save_result(self, result: Result) -> None:
        """Upsert one result row keyed by its unique cell."""
        self.conn.execute(
            """
            INSERT INTO results (
                result_id, run_id, graph_id, encoding, question_id,
                question_text, ground_truth, model, model_answer, correct,
                tokens_used, latency_ms, error, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id, graph_id, encoding, question_id, model)
            DO UPDATE SET
                result_id=excluded.result_id,
                question_text=excluded.question_text,
                ground_truth=excluded.ground_truth,
                model_answer=excluded.model_answer,
                correct=excluded.correct,
                tokens_used=excluded.tokens_used,
                latency_ms=excluded.latency_ms,
                error=excluded.error
            """,
            (
                result.result_id,
                result.run_id,
                result.graph_id,
                result.encoding,
                result.question_id,
                result.question_text,
                _dumps(result.ground_truth),
                result.model,
                _dumps(result.model_answer),
                1 if result.correct else 0,
                int(result.tokens_used),
                float(result.latency_ms),
                result.error,
                time.time(),
            ),
        )
        self.conn.commit()

    def get_result(
        self, run_id: str, graph_id: str, encoding: str, question_id: str, model: str
    ) -> Optional[Result]:
        """Fetch a cached result for a single cell, or ``None``."""
        row = self.conn.execute(
            """
            SELECT * FROM results
             WHERE run_id=? AND graph_id=? AND encoding=? AND question_id=? AND model=?
            """,
            (run_id, graph_id, encoding, question_id, model),
        ).fetchone()
        return self._row_to_result(row) if row else None

    def list_results(self, run_id: Optional[str] = None) -> list[Result]:
        if run_id is None:
            rows = self.conn.execute("SELECT * FROM results").fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM results WHERE run_id=?", (run_id,)
            ).fetchall()
        return [self._row_to_result(r) for r in rows]

    @staticmethod
    def _row_to_result(row: sqlite3.Row) -> Result:
        return Result(
            result_id=row["result_id"],
            run_id=row["run_id"],
            graph_id=row["graph_id"],
            encoding=row["encoding"],
            question_id=row["question_id"],
            question_text=row["question_text"],
            ground_truth=_loads(row["ground_truth"]),
            model=row["model"],
            model_answer=_loads(row["model_answer"]),
            correct=bool(row["correct"]),
            tokens_used=int(row["tokens_used"]),
            latency_ms=float(row["latency_ms"]),
            error=row["error"],
        )

    # -- export -------------------------------------------------------------

    def export_json(self, out_path: str | Path, run_id: Optional[str] = None) -> Path:
        """Write all results (optionally for one run) to a JSON file."""
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        results = [r.model_dump() for r in self.list_results(run_id)]
        payload: dict[str, Any] = {
            "run_id": run_id,
            "count": len(results),
            "results": results,
        }
        if run_id is not None:
            run = self.get_run(run_id)
            if run is not None:
                payload["run"] = run
        out_path.write_text(_dumps(payload) + "\n", encoding="utf-8")
        return out_path

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "Storage":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def save_results(db_path: str | Path, results: Iterable[Result]) -> None:
    """Helper: open ``db_path`` and persist an iterable of results."""
    with Storage(db_path) as store:
        for r in results:
            store.save_result(r)
