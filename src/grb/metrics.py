"""Phase 5 metrics & analysis for the Graph Reasoning Benchmark.

This module turns a list of :class:`grb.models.Result` rows (plus the graphs
and questions they reference) into:

* a tidy pandas ``DataFrame`` (:func:`results_to_frame`),
* a family of pre-aggregated metric tables (:func:`compute_metrics`):
  accuracy by encoding / model / difficulty / category / tier, token
  efficiency, an accuracy-vs-tokens scatter table, and an error-type
  breakdown,
* publication-quality matplotlib figures written to ``figures/``
  (:func:`generate_figures`),
* a single ``results.json`` the Hebrew dashboard reads
  (:func:`export_dashboard_json`).

All identifiers, comments and graph node labels stay in English. The Hebrew
lives only in the dashboard UI layer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import pandas as pd

from grb.models import BenchGraph, Question, Result

# ---------------------------------------------------------------------------
# Tidy frame
# ---------------------------------------------------------------------------

#: Columns that always exist on the tidy results frame.
RESULT_COLUMNS = [
    "result_id",
    "run_id",
    "graph_id",
    "encoding",
    "question_id",
    "question_text",
    "model",
    "correct",
    "tokens_used",
    "latency_ms",
    "error",
]


def results_to_frame(
    results: Iterable[Result | dict],
    *,
    questions: Optional[Sequence[Question]] = None,
    graphs: Optional[Sequence[BenchGraph]] = None,
) -> pd.DataFrame:
    """Build a tidy analysis frame from results.

    ``questions`` and ``graphs`` are optional enrichments: when supplied we
    join in each question's ``category``/``difficulty`` and each graph's
    ``tier``/``num_nodes``/edge-count so the aggregations have something to
    group by. Missing joins fall back to ``"unknown"``.
    """
    rows: list[dict[str, Any]] = []
    for r in results:
        d = r.model_dump() if isinstance(r, Result) else dict(r)
        rows.append({k: d.get(k) for k in RESULT_COLUMNS})

    frame = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    if frame.empty:
        # Still attach the enrichment columns so downstream code is uniform.
        for col in ("category", "difficulty", "tier", "num_nodes", "num_edges"):
            frame[col] = pd.Series(dtype="object")
        return frame

    frame["correct"] = frame["correct"].astype(bool)
    frame["tokens_used"] = pd.to_numeric(frame["tokens_used"], errors="coerce").fillna(0).astype(int)
    frame["latency_ms"] = pd.to_numeric(frame["latency_ms"], errors="coerce").fillna(0.0)

    # Question enrichment.
    q_cat: dict[str, str] = {}
    q_diff: dict[str, str] = {}
    for q in questions or []:
        q_cat[q.id] = q.category
        q_diff[q.id] = q.difficulty
    frame["category"] = frame["question_id"].map(q_cat).fillna("unknown")
    frame["difficulty"] = frame["question_id"].map(q_diff).fillna("unknown")

    # Graph enrichment.
    g_tier: dict[str, str] = {}
    g_nodes: dict[str, int] = {}
    g_edges: dict[str, int] = {}
    for g in graphs or []:
        g_tier[g.id] = g.metadata.tier
        g_nodes[g.id] = g.metadata.num_nodes
        g_edges[g.id] = len(g.edges)
    frame["tier"] = frame["graph_id"].map(g_tier).fillna("unknown")
    frame["num_nodes"] = frame["graph_id"].map(g_nodes)
    frame["num_edges"] = frame["graph_id"].map(g_edges)

    return frame


# ---------------------------------------------------------------------------
# Aggregations
# ---------------------------------------------------------------------------


def _accuracy_by(frame: pd.DataFrame, key: str) -> pd.DataFrame:
    """Accuracy + sample size grouped by a single column."""
    if frame.empty or key not in frame:
        return pd.DataFrame(columns=[key, "accuracy", "n", "correct"])
    grouped = (
        frame.groupby(key, dropna=False)["correct"]
        .agg(accuracy="mean", n="count", correct="sum")
        .reset_index()
        .sort_values("accuracy", ascending=False)
    )
    grouped["accuracy"] = grouped["accuracy"].round(4)
    return grouped


def accuracy_by_encoding(frame: pd.DataFrame) -> pd.DataFrame:
    return _accuracy_by(frame, "encoding")


def accuracy_by_model(frame: pd.DataFrame) -> pd.DataFrame:
    return _accuracy_by(frame, "model")


def accuracy_by_difficulty(frame: pd.DataFrame) -> pd.DataFrame:
    return _accuracy_by(frame, "difficulty")


def accuracy_by_category(frame: pd.DataFrame) -> pd.DataFrame:
    return _accuracy_by(frame, "category")


def accuracy_by_tier(frame: pd.DataFrame) -> pd.DataFrame:
    return _accuracy_by(frame, "tier")


def token_efficiency(frame: pd.DataFrame) -> pd.DataFrame:
    """Mean tokens per call and accuracy-per-1k-tokens, by encoding.

    ``accuracy_per_1k`` is a crude cost-effectiveness signal: how much
    correctness you buy per thousand prompt+completion tokens.
    """
    if frame.empty:
        return pd.DataFrame(
            columns=["encoding", "mean_tokens", "accuracy", "n", "accuracy_per_1k"]
        )
    g = (
        frame.groupby("encoding", dropna=False)
        .agg(
            mean_tokens=("tokens_used", "mean"),
            accuracy=("correct", "mean"),
            n=("correct", "count"),
        )
        .reset_index()
    )
    g["mean_tokens"] = g["mean_tokens"].round(1)
    g["accuracy"] = g["accuracy"].round(4)
    safe_tokens = g["mean_tokens"].where(g["mean_tokens"] != 0)
    g["accuracy_per_1k"] = (
        (g["accuracy"] / safe_tokens * 1000).fillna(0.0).round(4)
    )
    return g.sort_values("accuracy_per_1k", ascending=False)


def accuracy_vs_tokens(frame: pd.DataFrame) -> pd.DataFrame:
    """Per (model, encoding) accuracy against mean token cost (scatter table)."""
    if frame.empty:
        return pd.DataFrame(
            columns=["model", "encoding", "mean_tokens", "accuracy", "n"]
        )
    g = (
        frame.groupby(["model", "encoding"], dropna=False)
        .agg(
            mean_tokens=("tokens_used", "mean"),
            accuracy=("correct", "mean"),
            n=("correct", "count"),
        )
        .reset_index()
    )
    g["mean_tokens"] = g["mean_tokens"].round(1)
    g["accuracy"] = g["accuracy"].round(4)
    return g


def error_breakdown(frame: pd.DataFrame) -> pd.DataFrame:
    """Count rows by ``error`` label (``None`` -> ``"ok"``)."""
    if frame.empty:
        return pd.DataFrame(columns=["error", "count"])
    labels = frame["error"].fillna("ok").replace("", "ok")
    g = labels.value_counts().rename_axis("error").reset_index(name="count")
    return g


def model_x_format(frame: pd.DataFrame) -> pd.DataFrame:
    """Accuracy heatmap table: rows=model, columns=encoding."""
    if frame.empty:
        return pd.DataFrame()
    pivot = frame.pivot_table(
        index="model", columns="encoding", values="correct", aggfunc="mean"
    ).round(4)
    return pivot


@dataclass
class MetricTables:
    """Bundle of every aggregated table, ready to serialize."""

    by_encoding: pd.DataFrame
    by_model: pd.DataFrame
    by_difficulty: pd.DataFrame
    by_category: pd.DataFrame
    by_tier: pd.DataFrame
    token_efficiency: pd.DataFrame
    accuracy_vs_tokens: pd.DataFrame
    error_breakdown: pd.DataFrame
    model_x_format: pd.DataFrame

    def to_json_dict(self) -> dict[str, Any]:
        """Each table as a list of record dicts (heatmap keeps its index)."""
        mxf = self.model_x_format
        mxf_records: list[dict[str, Any]] = []
        if not mxf.empty:
            mxf_records = mxf.reset_index().to_dict(orient="records")
        return {
            "by_encoding": self.by_encoding.to_dict(orient="records"),
            "by_model": self.by_model.to_dict(orient="records"),
            "by_difficulty": self.by_difficulty.to_dict(orient="records"),
            "by_category": self.by_category.to_dict(orient="records"),
            "by_tier": self.by_tier.to_dict(orient="records"),
            "token_efficiency": self.token_efficiency.to_dict(orient="records"),
            "accuracy_vs_tokens": self.accuracy_vs_tokens.to_dict(orient="records"),
            "error_breakdown": self.error_breakdown.to_dict(orient="records"),
            "model_x_format": {
                "models": list(mxf.index) if not mxf.empty else [],
                "encodings": list(mxf.columns) if not mxf.empty else [],
                "rows": mxf_records,
            },
        }


def compute_metrics(frame: pd.DataFrame) -> MetricTables:
    """Run every aggregation over the tidy frame."""
    return MetricTables(
        by_encoding=accuracy_by_encoding(frame),
        by_model=accuracy_by_model(frame),
        by_difficulty=accuracy_by_difficulty(frame),
        by_category=accuracy_by_category(frame),
        by_tier=accuracy_by_tier(frame),
        token_efficiency=token_efficiency(frame),
        accuracy_vs_tokens=accuracy_vs_tokens(frame),
        error_breakdown=error_breakdown(frame),
        model_x_format=model_x_format(frame),
    )


def overall_summary(frame: pd.DataFrame) -> dict[str, Any]:
    """Scalar headline numbers for the dashboard hero cards."""
    if frame.empty:
        return {
            "total_results": 0,
            "overall_accuracy": 0.0,
            "n_models": 0,
            "n_encodings": 0,
            "n_graphs": 0,
            "total_tokens": 0,
            "error_rate": 0.0,
        }
    errors = frame["error"].fillna("").replace("ok", "")
    return {
        "total_results": int(len(frame)),
        "overall_accuracy": round(float(frame["correct"].mean()), 4),
        "n_models": int(frame["model"].nunique()),
        "n_encodings": int(frame["encoding"].nunique()),
        "n_graphs": int(frame["graph_id"].nunique()),
        "total_tokens": int(frame["tokens_used"].sum()),
        "error_rate": round(float((errors != "").mean()), 4),
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def generate_figures(frame: pd.DataFrame, out_dir: str | Path) -> list[Path]:
    """Render publication-quality PNG figures into ``out_dir``.

    Uses a non-interactive backend so it is safe in CI / headless. Returns the
    list of written paths. Skips a figure when its source table is empty.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 140,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    def _save(fig, name: str) -> None:
        path = out_dir / name
        fig.tight_layout()
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        written.append(path)

    if frame.empty:
        return written

    # 1. Accuracy by encoding.
    enc = accuracy_by_encoding(frame)
    if not enc.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(enc["encoding"], enc["accuracy"], color="#3b6ea5")
        ax.set_ylabel("Accuracy")
        ax.set_title("Accuracy by Encoding")
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", rotation=30)
        _save(fig, "accuracy_by_encoding.png")

    # 2. Accuracy by model (leaderboard).
    mod = accuracy_by_model(frame)
    if not mod.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.barh(mod["model"], mod["accuracy"], color="#a5673b")
        ax.set_xlabel("Accuracy")
        ax.set_title("Model Leaderboard")
        ax.set_xlim(0, 1)
        ax.invert_yaxis()
        _save(fig, "model_leaderboard.png")

    # 3. Accuracy by difficulty.
    diff = accuracy_by_difficulty(frame)
    if not diff.empty:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.bar(diff["difficulty"], diff["accuracy"], color="#4a8c5e")
        ax.set_ylabel("Accuracy")
        ax.set_title("Accuracy by Difficulty")
        ax.set_ylim(0, 1)
        _save(fig, "accuracy_by_difficulty.png")

    # 4. Accuracy by tier.
    tier = accuracy_by_tier(frame)
    if not tier.empty:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.bar(tier["tier"], tier["accuracy"], color="#7a5c99")
        ax.set_ylabel("Accuracy")
        ax.set_title("Accuracy by Graph Tier")
        ax.set_ylim(0, 1)
        _save(fig, "accuracy_by_tier.png")

    # 5. Accuracy vs tokens (cost/quality scatter).
    avt = accuracy_vs_tokens(frame)
    if not avt.empty:
        fig, ax = plt.subplots(figsize=(7, 5))
        for enc_name, grp in avt.groupby("encoding"):
            ax.scatter(grp["mean_tokens"], grp["accuracy"], label=enc_name, s=70, alpha=0.8)
        ax.set_xlabel("Mean tokens per call")
        ax.set_ylabel("Accuracy")
        ax.set_title("Accuracy vs Token Cost")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8, loc="best")
        _save(fig, "accuracy_vs_tokens.png")

    # 6. Error-type breakdown.
    err = error_breakdown(frame)
    if not err.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(err["error"], err["count"], color="#b5453b")
        ax.set_ylabel("Count")
        ax.set_title("Error-type Breakdown")
        ax.tick_params(axis="x", rotation=30)
        _save(fig, "error_breakdown.png")

    # 7. Model x format heatmap.
    mxf = model_x_format(frame)
    if not mxf.empty:
        fig, ax = plt.subplots(figsize=(1.2 * len(mxf.columns) + 3, 0.7 * len(mxf.index) + 2))
        data = mxf.fillna(0).values
        im = ax.imshow(data, cmap="viridis", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(mxf.columns)))
        ax.set_xticklabels(mxf.columns, rotation=30, ha="right")
        ax.set_yticks(range(len(mxf.index)))
        ax.set_yticklabels(mxf.index)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", color="w", fontsize=8)
        ax.set_title("Accuracy: Model x Format")
        fig.colorbar(im, ax=ax, label="Accuracy")
        _save(fig, "model_x_format_heatmap.png")

    return written


# ---------------------------------------------------------------------------
# Graph payloads for the Cytoscape explorer
# ---------------------------------------------------------------------------


def graph_to_payload(graph: BenchGraph) -> dict[str, Any]:
    """Serialize a graph into a Cytoscape-friendly payload."""
    return {
        "id": graph.id,
        "metadata": graph.metadata.model_dump(),
        "nodes": list(graph.nodes),
        "edges": [
            {
                "source": e.source,
                "target": e.target,
                "type": e.type,
                "weight": e.weight,
            }
            for e in graph.edges
        ],
    }


# ---------------------------------------------------------------------------
# Dashboard export
# ---------------------------------------------------------------------------


def build_dashboard_payload(
    results: Sequence[Result | dict],
    *,
    questions: Optional[Sequence[Question]] = None,
    graphs: Optional[Sequence[BenchGraph]] = None,
) -> dict[str, Any]:
    """Assemble the full ``results.json`` payload the dashboard reads."""
    frame = results_to_frame(results, questions=questions, graphs=graphs)
    tables = compute_metrics(frame)

    result_rows = frame.to_dict(orient="records")
    # Make NaN JSON-safe.
    for row in result_rows:
        for k, v in list(row.items()):
            if isinstance(v, float) and v != v:  # NaN
                row[k] = None

    return {
        "generated_at": pd.Timestamp.now("UTC").isoformat(),
        "summary": overall_summary(frame),
        "metrics": tables.to_json_dict(),
        "results": result_rows,
        "graphs": [graph_to_payload(g) for g in (graphs or [])],
    }


def export_dashboard_json(
    out_path: str | Path,
    results: Sequence[Result | dict],
    *,
    questions: Optional[Sequence[Question]] = None,
    graphs: Optional[Sequence[BenchGraph]] = None,
) -> Path:
    """Write the dashboard payload to ``out_path`` (creating parent dirs)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_dashboard_payload(results, questions=questions, graphs=graphs)
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return out_path
