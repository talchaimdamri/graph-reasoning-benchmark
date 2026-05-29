"""Benchmark execution pipeline.

``evaluate_single`` runs one (graph, encoding, question, model) cell; it builds
the prompt, calls Claude headless, grades the answer, and returns a
:class:`grb.models.Result`. ``run_benchmark`` iterates the full grid of
graphs x encodings x questions x models with:

* a configurable concurrency cap (``ThreadPoolExecutor``, default 4 workers),
* caching keyed on (graph_id, encoding, question_id, model) via SQLite so reruns
  of already-completed cells cost nothing,
* a progress bar and a running cost/quota counter (calls + tokens),
* checkpoint/resume by ``run_id`` (cached cells are skipped on resume),
* graceful handling of the ``visual`` format (marked
  ``error='vision-unsupported-in-headless'`` without calling the model).
"""

from __future__ import annotations

import sys
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from grb.answers import grade, is_valid_shortest_path, parse_response
from grb.llm.headless import call_claude as _default_call_claude
from grb.models import BenchGraph, Encoding, Question, Result
from grb.prompts import (
    VISION_UNSUPPORTED_ERROR,
    VISUAL_FORMAT,
    VisionUnsupported,
    build_prompt,
)
from grb.storage import Storage

DEFAULT_MODELS = ("opus", "sonnet", "haiku")


@dataclass
class BenchmarkConfig:
    """Inputs for :func:`run_benchmark`.

    ``graphs`` and ``questions`` describe the grid; ``encodings`` maps each
    graph_id to its format -> Encoding dict (as returned by
    :func:`grb.encoder.encode_graph`).
    """

    graphs: list[BenchGraph]
    encodings: dict[str, dict[str, Encoding]]
    questions: dict[str, list[Question]]
    models: tuple[str, ...] = DEFAULT_MODELS
    db_path: str = "data/results/benchmark.db"
    run_id: Optional[str] = None
    max_workers: int = 4
    formats: Optional[list[str]] = None  # restrict to these formats if set
    retries: int = 2
    backoff_s: float = 2.0
    timeout_s: float = 120.0
    show_progress: bool = True


@dataclass
class Quota:
    """Running cost/quota counter for a benchmark run."""

    calls: int = 0
    tokens: int = 0
    cached: int = 0
    errors: int = 0
    skipped: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "calls": self.calls,
            "tokens": self.tokens,
            "cached": self.cached,
            "errors": self.errors,
            "skipped": self.skipped,
        }


@dataclass
class Cell:
    """One unit of work in the grid."""

    graph_id: str
    encoding: Encoding
    fmt: str
    question: Question
    model: str
    graph: Optional[BenchGraph] = None


def _result_id(run_id: str, graph_id: str, fmt: str, question_id: str, model: str) -> str:
    return f"{run_id}::{graph_id}::{fmt}::{question_id}::{model}"


def _grade_answer(
    text: str,
    question: Question,
    graph: Optional[BenchGraph],
) -> tuple[Any, bool]:
    """Grade ``text`` against ``question``.

    For the ``shortest_path`` category (where the ground truth is an *ordered*
    path) graphs are graded structurally when ``graph`` is available: any valid
    shortest path is accepted and non-paths are rejected. All other categories
    fall back to the generic type-aware :func:`grade`.
    """
    if question.category == "shortest_path" and graph is not None:
        gt = question.ground_truth
        if isinstance(gt, (list, tuple)) and len(gt) >= 1:
            parsed = parse_response(text, question.answer_type)
            src, dst = str(gt[0]), str(gt[-1])
            target_len = len(gt) - 1
            correct = is_valid_shortest_path(
                parsed, src, dst, graph.to_networkx(), target_len
            )
            return parsed, correct
    return grade(text, question.ground_truth, question.answer_type)


def evaluate_single(
    *,
    run_id: str,
    encoding: Encoding,
    fmt: str,
    question: Question,
    model: str,
    graph: Optional[BenchGraph] = None,
    call_fn: Callable[..., dict[str, Any]] = _default_call_claude,
    retries: int = 2,
    backoff_s: float = 2.0,
    timeout_s: float = 120.0,
) -> Result:
    """Evaluate one cell and return a :class:`Result` (never raises on LLM error).

    The ``visual`` format short-circuits to a skipped result with
    ``error='vision-unsupported-in-headless'`` and no model call.

    When ``graph`` is supplied, ``shortest_path`` questions are graded
    structurally (any valid shortest path is accepted) rather than by the
    order-insensitive list comparison.
    """
    rid = _result_id(run_id, encoding.graph_id, fmt, question.id, model)
    base = dict(
        result_id=rid,
        run_id=run_id,
        graph_id=encoding.graph_id,
        encoding=fmt,
        question_id=question.id,
        question_text=question.text,
        ground_truth=question.ground_truth,
        model=model,
    )

    # Visual format: cannot send image to headless claude.
    if fmt == VISUAL_FORMAT:
        return Result(
            **base,
            model_answer=None,
            correct=False,
            tokens_used=0,
            latency_ms=0.0,
            error=VISION_UNSUPPORTED_ERROR,
        )

    try:
        prompt = build_prompt(encoding, question, fmt)
    except VisionUnsupported:
        return Result(
            **base,
            model_answer=None,
            correct=False,
            tokens_used=0,
            latency_ms=0.0,
            error=VISION_UNSUPPORTED_ERROR,
        )

    resp = call_fn(
        prompt,
        model,
        retries=retries,
        backoff_s=backoff_s,
        timeout_s=timeout_s,
    )
    text = resp.get("text", "") or ""
    error = resp.get("error")

    if error and not text:
        return Result(
            **base,
            model_answer=None,
            correct=False,
            tokens_used=int(resp.get("tokens_used", 0) or 0),
            latency_ms=float(resp.get("latency_ms", 0.0) or 0.0),
            error=error,
        )

    parsed, correct = _grade_answer(text, question, graph)
    return Result(
        **base,
        model_answer=parsed if parsed is not None else text,
        correct=correct,
        tokens_used=int(resp.get("tokens_used", 0) or 0),
        latency_ms=float(resp.get("latency_ms", 0.0) or 0.0),
        error=error,
    )


def _build_cells(config: BenchmarkConfig) -> list[Cell]:
    """Expand the grid into a flat list of work cells."""
    cells: list[Cell] = []
    for graph in config.graphs:
        enc_map = config.encodings.get(graph.id, {})
        qs = config.questions.get(graph.id, [])
        for fmt, encoding in enc_map.items():
            if config.formats is not None and fmt not in config.formats:
                continue
            for question in qs:
                for model in config.models:
                    cells.append(
                        Cell(
                            graph_id=graph.id,
                            encoding=encoding,
                            fmt=fmt,
                            question=question,
                            model=model,
                            graph=graph,
                        )
                    )
    return cells


# Rough per-call output token allowance (answers are short; a modest budget
# covers reasoning slippage without claiming to be exact).
OUTPUT_TOKEN_ALLOWANCE = 64


def estimate_cost(config: BenchmarkConfig) -> dict[str, Any]:
    """Project the grid size, total calls and approximate input/output tokens.

    No API calls are made. Input tokens are estimated by building the real
    prompt for each non-visual cell and counting its tokens with the same
    ``cl100k_base`` encoder used elsewhere; visual cells are counted as skipped.
    Output tokens use a flat :data:`OUTPUT_TOKEN_ALLOWANCE` per call. Returns a
    summary with a per-tier breakdown and totals.
    """
    from grb.encoder import count_tokens

    cells = _build_cells(config)

    tiers: dict[str, dict[str, int]] = {}

    def _tier_of(cell: Cell) -> str:
        if cell.graph is not None:
            return cell.graph.metadata.tier
        return "unknown"

    totals = {"cells": 0, "calls": 0, "skipped": 0, "input_tokens": 0, "output_tokens": 0}

    # Cache prompt token counts per (graph_id, fmt, question_id) since they do
    # not depend on the model.
    _prompt_cache: dict[tuple[str, str, str], int] = {}

    for cell in cells:
        tier = _tier_of(cell)
        row = tiers.setdefault(
            tier,
            {"cells": 0, "calls": 0, "skipped": 0, "input_tokens": 0, "output_tokens": 0},
        )
        row["cells"] += 1
        totals["cells"] += 1

        if cell.fmt == VISUAL_FORMAT:
            row["skipped"] += 1
            totals["skipped"] += 1
            continue

        key = (cell.graph_id, cell.fmt, cell.question.id)
        in_tokens = _prompt_cache.get(key)
        if in_tokens is None:
            try:
                prompt = build_prompt(cell.encoding, cell.question, cell.fmt)
                in_tokens = count_tokens(prompt)
            except VisionUnsupported:
                row["skipped"] += 1
                totals["skipped"] += 1
                _prompt_cache[key] = 0
                continue
            _prompt_cache[key] = in_tokens

        row["calls"] += 1
        row["input_tokens"] += in_tokens
        row["output_tokens"] += OUTPUT_TOKEN_ALLOWANCE
        totals["calls"] += 1
        totals["input_tokens"] += in_tokens
        totals["output_tokens"] += OUTPUT_TOKEN_ALLOWANCE

    return {
        "models": list(config.models),
        "tiers": tiers,
        "totals": totals,
        "output_allowance_per_call": OUTPUT_TOKEN_ALLOWANCE,
    }


def format_estimate(est: dict[str, Any]) -> str:
    """Render :func:`estimate_cost` output as a per-tier and total table."""
    lines = [
        f"Cost estimate (models: {', '.join(est['models'])}; "
        f"output allowance {est['output_allowance_per_call']} tok/call):",
        f"  {'tier':<10}{'cells':>8}{'calls':>8}{'skipped':>9}"
        f"{'in_tokens':>12}{'out_tokens':>12}",
    ]
    for tier in sorted(est["tiers"]):
        row = est["tiers"][tier]
        lines.append(
            f"  {tier:<10}{row['cells']:>8}{row['calls']:>8}{row['skipped']:>9}"
            f"{row['input_tokens']:>12}{row['output_tokens']:>12}"
        )
    t = est["totals"]
    lines.append(
        f"  {'TOTAL':<10}{t['cells']:>8}{t['calls']:>8}{t['skipped']:>9}"
        f"{t['input_tokens']:>12}{t['output_tokens']:>12}"
    )
    return "\n".join(lines)


def _progress(done: int, total: int, quota: Quota, *, stream=sys.stderr) -> None:
    width = 28
    frac = (done / total) if total else 1.0
    filled = int(width * frac)
    bar = "#" * filled + "-" * (width - filled)
    stream.write(
        f"\r[{bar}] {done}/{total} "
        f"calls={quota.calls} tok={quota.tokens} "
        f"cached={quota.cached} skip={quota.skipped} err={quota.errors}   "
    )
    stream.flush()
    if done >= total:
        stream.write("\n")
        stream.flush()


def run_benchmark(
    config: BenchmarkConfig,
    *,
    call_fn: Callable[..., dict[str, Any]] = _default_call_claude,
) -> dict[str, Any]:
    """Run the full benchmark grid with caching, concurrency and progress.

    Returns a summary dict with ``run_id``, ``quota``, and ``results`` (the list
    of :class:`Result` for this run, including cached cells).
    """
    run_id = config.run_id or uuid.uuid4().hex[:12]
    store = Storage(config.db_path)
    store.create_run(run_id, config={"models": list(config.models)})

    cells = _build_cells(config)
    total = len(cells)
    quota = Quota()

    # Partition into cached vs to-run (resume support).
    to_run: list[Cell] = []
    done = 0
    for cell in cells:
        cached = store.get_result(
            run_id, cell.graph_id, cell.fmt, cell.question.id, cell.model
        )
        if cached is not None:
            quota.cached += 1
            quota.tokens += cached.tokens_used
            if cached.error:
                quota.errors += 1
            if cached.error == VISION_UNSUPPORTED_ERROR:
                quota.skipped += 1
            done += 1
        else:
            to_run.append(cell)

    if config.show_progress:
        _progress(done, total, quota)

    def _work(cell: Cell) -> Result:
        return evaluate_single(
            run_id=run_id,
            encoding=cell.encoding,
            fmt=cell.fmt,
            question=cell.question,
            model=cell.model,
            graph=cell.graph,
            call_fn=call_fn,
            retries=config.retries,
            backoff_s=config.backoff_s,
            timeout_s=config.timeout_s,
        )

    if to_run:
        with ThreadPoolExecutor(max_workers=max(1, config.max_workers)) as pool:
            futures = {pool.submit(_work, cell): cell for cell in to_run}
            for fut in as_completed(futures):
                result = fut.result()
                store.save_result(result)
                done += 1
                if result.error == VISION_UNSUPPORTED_ERROR:
                    quota.skipped += 1
                else:
                    quota.calls += 1
                    quota.tokens += result.tokens_used
                if result.error and result.error != VISION_UNSUPPORTED_ERROR:
                    quota.errors += 1
                store.update_run_stats(
                    run_id, total_calls=quota.calls, total_tokens=quota.tokens
                )
                if config.show_progress:
                    _progress(done, total, quota)

    store.update_run_stats(
        run_id, total_calls=quota.calls, total_tokens=quota.tokens, status="done"
    )
    results = store.list_results(run_id)
    store.close()

    correct = sum(1 for r in results if r.correct)
    return {
        "run_id": run_id,
        "total": total,
        "correct": correct,
        "accuracy": round(correct / total, 4) if total else 0.0,
        "quota": quota.as_dict(),
        "results": results,
    }
