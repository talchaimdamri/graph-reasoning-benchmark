"""Synthetic benchmark fixtures.

Generates a deterministic, plausible set of graphs / questions / results so the
dashboard renders before any real (and expensive) model run has happened. The
numbers are fabricated but internally consistent: encodings differ in token
cost, models differ in skill, harder questions are answered less often.

Used both by :mod:`tests.test_metrics` and by ``scripts/build_dashboard_data.py``.
"""

from __future__ import annotations

import hashlib
import random
from typing import Any

from grb.encoder import ALL_FORMATS
from grb.generator import make_tiered_graph
from grb.models import BenchGraph, Question, Result

# Plausible relative skill (base accuracy) per model.
_MODEL_SKILL = {
    "claude-opus-4": 0.86,
    "claude-sonnet-4": 0.78,
    "gpt-4o": 0.74,
}

# Relative token cost and a per-encoding accuracy nudge.
_ENCODING_PROFILE = {
    "adjacency_list": (1.0, 0.04),
    "edge_list": (1.1, 0.02),
    "mermaid": (1.4, 0.0),
    "dot": (1.5, -0.01),
    "natural_language": (2.2, 0.06),
    "matrix": (1.8, -0.05),
    "visual": (1.0, -0.02),
}


def _hash_id(*parts: str) -> str:
    return hashlib.sha1("::".join(parts).encode()).hexdigest()[:8]


def make_fixture_graphs() -> list[BenchGraph]:
    """One graph per tier."""
    return [make_tiered_graph(tier, seed=i + 1) for i, tier in enumerate(["small", "medium", "large"])]


def _questions_for_graph(graph: BenchGraph) -> list[Question]:
    g = graph.to_networkx()
    n = graph.metadata.num_nodes
    m = len(graph.edges)
    specs: list[dict[str, Any]] = [
        dict(category="node_count", difficulty="trivial", answer_type="int",
             text="How many nodes does the graph have?", gt=n, comp="G.number_of_nodes()"),
        dict(category="edge_count", difficulty="trivial", answer_type="int",
             text="How many edges does the graph have?", gt=m, comp="G.number_of_edges()"),
        dict(category="degree", difficulty="nontrivial", answer_type="int",
             text="What is the maximum node degree?",
             gt=max((d for _, d in g.degree()), default=0), comp="max(degree)"),
        dict(category="connectivity", difficulty="nontrivial", answer_type="bool",
             text="Is the graph connected (ignoring direction)?",
             gt=bool(n) and (g.number_of_edges() > 0), comp="nx.is_connected(G.to_undirected())"),
    ]
    out: list[Question] = []
    for s in specs:
        qid = f"{graph.id}__{s['category']}__{_hash_id(graph.id, s['category'])}"
        out.append(
            Question(
                id=qid,
                graph_id=graph.id,
                text=s["text"],
                category=s["category"],
                difficulty=s["difficulty"],
                answer_type=s["answer_type"],
                ground_truth=s["gt"],
                computation=s["comp"],
            )
        )
    return out


def make_fixture_questions(graphs: list[BenchGraph]) -> list[Question]:
    out: list[Question] = []
    for g in graphs:
        out.extend(_questions_for_graph(g))
    return out


def make_fixture_results(
    graphs: list[BenchGraph],
    questions: list[Question],
    *,
    run_id: str = "synthetic-demo",
    seed: int = 7,
) -> list[Result]:
    """Fabricate one result per (graph, encoding, question, model) cell."""
    rng = random.Random(seed)
    models = list(_MODEL_SKILL)
    tier_penalty = {"small": 0.0, "medium": -0.08, "large": -0.18, "unknown": 0.0}
    by_graph = {g.id: g for g in graphs}

    results: list[Result] = []
    for q in questions:
        graph = by_graph[q.graph_id]
        tier = graph.metadata.tier
        base_tokens = 60 + len(graph.edges) * 6
        for fmt in ALL_FORMATS:
            cost_mult, enc_acc = _ENCODING_PROFILE[fmt]
            for model in models:
                # Visual is unsupported in headless -> record an error row.
                if fmt == "visual":
                    results.append(
                        Result(
                            result_id=f"{run_id}::{graph.id}::{fmt}::{q.id}::{model}",
                            run_id=run_id,
                            graph_id=graph.id,
                            encoding=fmt,
                            question_id=q.id,
                            question_text=q.text,
                            ground_truth=q.ground_truth,
                            model=model,
                            model_answer=None,
                            correct=False,
                            tokens_used=0,
                            latency_ms=0.0,
                            error="vision-unsupported-in-headless",
                        )
                    )
                    continue

                p = _MODEL_SKILL[model] + enc_acc + tier_penalty.get(tier, 0.0)
                if q.difficulty == "nontrivial":
                    p -= 0.18
                p = max(0.02, min(0.98, p))
                correct = rng.random() < p

                # Small chance of a parse error on a wrong answer.
                error: str | None = None
                if not correct and rng.random() < 0.08:
                    error = "answer-parse-failed"

                tokens = int(base_tokens * cost_mult * rng.uniform(0.9, 1.1))
                latency = round(rng.uniform(400, 2500), 1)
                model_answer = q.ground_truth if correct else _wrong_answer(q, rng)

                results.append(
                    Result(
                        result_id=f"{run_id}::{graph.id}::{fmt}::{q.id}::{model}",
                        run_id=run_id,
                        graph_id=graph.id,
                        encoding=fmt,
                        question_id=q.id,
                        question_text=q.text,
                        ground_truth=q.ground_truth,
                        model=model,
                        model_answer=None if error else model_answer,
                        correct=correct,
                        tokens_used=tokens,
                        latency_ms=latency,
                        error=error,
                    )
                )
    return results


def _wrong_answer(q: Question, rng: random.Random) -> Any:
    gt = q.ground_truth
    if isinstance(gt, bool):
        return not gt
    if isinstance(gt, int):
        return gt + rng.choice([-2, -1, 1, 2])
    if isinstance(gt, float):
        return round(gt * rng.uniform(0.5, 1.5), 2)
    return "unknown"


def make_full_fixture() -> dict[str, Any]:
    """Convenience: graphs + questions + results in one call."""
    graphs = make_fixture_graphs()
    questions = make_fixture_questions(graphs)
    results = make_fixture_results(graphs, questions)
    return {"graphs": graphs, "questions": questions, "results": results}
