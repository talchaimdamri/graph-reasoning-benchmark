"""Command-line interface for the Graph Reasoning Benchmark (grb).

Commands
--------
generate   Generate tiered graphs and write them as JSON.
encode     Encode stored graphs into the benchmark's text formats.
questions  Generate deterministic questions (with NetworkX ground truth).
smoke      Run exactly 6 real ``claude -p`` calls (2 questions x 1 graph x 1
           encoding x 3 models) and print pass/fail.
run        Run the full benchmark grid (graphs x encodings x questions x models).
export     Export stored results to JSON.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(help="Graph Reasoning Benchmark CLI.", no_args_is_help=True)

DEFAULT_GRAPH_DIR = "data/graphs"
DEFAULT_ENCODING_DIR = "data/encodings"
DEFAULT_QUESTION_DIR = "data/questions"
DEFAULT_DB = "data/results/benchmark.db"
TIERS = ("small", "medium", "large")


@app.command()
def version() -> None:
    """Print the grb version."""
    from grb import __version__

    typer.echo(__version__)


# --------------------------------------------------------------------------- #
# generate
# --------------------------------------------------------------------------- #
@app.command()
def generate(
    tier: str = typer.Option("small", help="Tier: small | medium | large."),
    seed: int = typer.Option(0, help="Random seed."),
    out_dir: str = typer.Option(DEFAULT_GRAPH_DIR, help="Output directory."),
) -> None:
    """Generate one tier-calibrated graph and write it to ``out_dir``."""
    from grb.generator import make_tiered_graph

    if tier not in TIERS:
        raise typer.BadParameter(f"tier must be one of {TIERS}")
    graph = make_tiered_graph(tier, seed=seed)  # type: ignore[arg-type]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{graph.id}.json"
    path.write_text(graph.model_dump_json(indent=2) + "\n", encoding="utf-8")
    typer.echo(f"Wrote {path} ({len(graph.nodes)} nodes, {len(graph.edges)} edges)")


def _load_graphs(graph_dir: str):
    from grb.models import BenchGraph

    paths = sorted(Path(graph_dir).glob("*.json"))
    return [BenchGraph.model_validate_json(p.read_text()) for p in paths]


# --------------------------------------------------------------------------- #
# encode
# --------------------------------------------------------------------------- #
@app.command()
def encode(
    graph_dir: str = typer.Option(DEFAULT_GRAPH_DIR, help="Directory of graph JSON."),
    out_dir: str = typer.Option(DEFAULT_ENCODING_DIR, help="Output directory."),
    formats: Optional[str] = typer.Option(
        None, help="Comma-separated formats; default = all text formats."
    ),
) -> None:
    """Encode every graph in ``graph_dir`` into text formats; write JSON per graph."""
    from grb.encoder import encode_graph

    fmt_list = [f.strip() for f in formats.split(",")] if formats else None
    graphs = _load_graphs(graph_dir)
    if not graphs:
        typer.echo(f"No graphs found in {graph_dir}")
        raise typer.Exit(code=1)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for g in graphs:
        enc = encode_graph(g, formats=fmt_list)
        payload = {fmt: e.model_dump() for fmt, e in enc.items()}
        path = out / f"{g.id}.json"
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        typer.echo(f"Encoded {g.id}: {', '.join(payload.keys())}")


def _load_encodings(encoding_dir: str):
    from grb.models import Encoding

    out: dict[str, dict[str, Encoding]] = {}
    for p in sorted(Path(encoding_dir).glob("*.json")):
        data = json.loads(p.read_text())
        out[p.stem] = {fmt: Encoding.model_validate(e) for fmt, e in data.items()}
    return out


# --------------------------------------------------------------------------- #
# questions
# --------------------------------------------------------------------------- #
@app.command()
def questions(
    graph_dir: str = typer.Option(DEFAULT_GRAPH_DIR, help="Directory of graph JSON."),
    out_dir: str = typer.Option(DEFAULT_QUESTION_DIR, help="Output directory."),
    n: int = typer.Option(15, help="Questions per graph."),
    seed: int = typer.Option(0, help="Seed for question sampling."),
) -> None:
    """Generate ``n`` deterministic questions per graph; write JSON per graph."""
    from grb.ground_truth import generate_questions

    graphs = _load_graphs(graph_dir)
    if not graphs:
        typer.echo(f"No graphs found in {graph_dir}")
        raise typer.Exit(code=1)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for g in graphs:
        qs = generate_questions(g, n=n, seed=seed)
        path = out / f"{g.id}.json"
        path.write_text(
            json.dumps([q.model_dump() for q in qs], indent=2) + "\n",
            encoding="utf-8",
        )
        typer.echo(f"{g.id}: {len(qs)} questions")


def _load_questions(question_dir: str):
    from grb.models import Question

    out: dict[str, list[Question]] = {}
    for p in sorted(Path(question_dir).glob("*.json")):
        data = json.loads(p.read_text())
        out[p.stem] = [Question.model_validate(q) for q in data]
    return out


# --------------------------------------------------------------------------- #
# smoke
# --------------------------------------------------------------------------- #
@app.command()
def smoke(
    encoding_format: str = typer.Option("edge_list", help="Single format to test."),
    seed: int = typer.Option(0, help="Seed."),
) -> None:
    """Run exactly 6 real ``claude -p`` calls (2 questions x 1 graph x 3 models).

    Prints pass/fail per call. If the nested ``claude`` invocation is blocked in
    this environment it is reported clearly as 'nested-claude-blocked' and the
    command exits 0 (a blocked nested call is not a build failure).
    """
    from grb.encoder import encode_graph
    from grb.generator import make_tiered_graph
    from grb.ground_truth import generate_questions
    from grb.llm.headless import NESTED_BLOCKED
    from grb.pipeline import evaluate_single

    models = ("opus", "sonnet", "haiku")

    graph = make_tiered_graph("small", seed=seed)
    enc = encode_graph(graph, formats=[encoding_format])[encoding_format]
    qs = generate_questions(graph, n=6, seed=seed)[:2]
    if len(qs) < 2:
        typer.echo("Could not generate 2 questions for the smoke graph.")
        raise typer.Exit(code=1)

    typer.echo(f"Smoke test: graph={graph.id} format={encoding_format}")
    typer.echo(f"  q1: {qs[0].text}  (truth={qs[0].ground_truth})")
    typer.echo(f"  q2: {qs[1].text}  (truth={qs[1].ground_truth})")
    typer.echo("")

    blocked = False
    passed = 0
    total = 0
    for q in qs:
        for model in models:
            total += 1
            res = evaluate_single(
                run_id="smoke",
                encoding=enc,
                fmt=encoding_format,
                question=q,
                model=model,
                retries=1,
                backoff_s=1.0,
                timeout_s=120.0,
            )
            if res.error == NESTED_BLOCKED:
                blocked = True
                typer.echo(f"  [{model}] nested-claude-blocked")
                continue
            mark = "PASS" if res.correct else "FAIL"
            if res.correct:
                passed += 1
            extra = f" error={res.error}" if res.error else ""
            typer.echo(
                f"  [{model}] {mark} answer={res.model_answer!r} "
                f"tok={res.tokens_used} {res.latency_ms:.0f}ms{extra}"
            )

    typer.echo("")
    if blocked:
        typer.echo("RESULT: nested-claude-blocked (live smoke could not run here)")
        raise typer.Exit(code=0)
    typer.echo(f"RESULT: {passed}/{total} calls passed")


# --------------------------------------------------------------------------- #
# run
# --------------------------------------------------------------------------- #
def _build_run_config(
    graph_dir: str,
    encoding_dir: str,
    question_dir: str,
    db_path: str,
    run_id: Optional[str],
    max_workers: int,
    models: str,
    formats: Optional[str],
):
    """Load inputs and assemble a BenchmarkConfig (or exit if anything missing)."""
    from grb.pipeline import BenchmarkConfig

    graphs = _load_graphs(graph_dir)
    encodings = _load_encodings(encoding_dir)
    qs = _load_questions(question_dir)
    if not graphs or not encodings or not qs:
        typer.echo("Missing graphs, encodings, or questions. Run generate/encode/questions first.")
        raise typer.Exit(code=1)

    return BenchmarkConfig(
        graphs=graphs,
        encodings=encodings,
        questions=qs,
        models=tuple(m.strip() for m in models.split(",")),
        db_path=db_path,
        run_id=run_id,
        max_workers=max_workers,
        formats=[f.strip() for f in formats.split(",")] if formats else None,
    )


@app.command()
def estimate(
    graph_dir: str = typer.Option(DEFAULT_GRAPH_DIR),
    encoding_dir: str = typer.Option(DEFAULT_ENCODING_DIR),
    question_dir: str = typer.Option(DEFAULT_QUESTION_DIR),
    db_path: str = typer.Option(DEFAULT_DB),
    run_id: Optional[str] = typer.Option(None, help="Resume/checkpoint id."),
    max_workers: int = typer.Option(4),
    models: str = typer.Option("opus,sonnet,haiku"),
    formats: Optional[str] = typer.Option(None, help="Restrict to these formats."),
) -> None:
    """Estimate the benchmark grid size, total calls and approximate tokens.

    Makes NO API calls; computes the (graphs x encodings x questions x models)
    grid and projects input tokens from the real prompts plus an output
    allowance per call.
    """
    from grb.pipeline import estimate_cost, format_estimate

    config = _build_run_config(
        graph_dir, encoding_dir, question_dir, db_path, run_id, max_workers,
        models, formats,
    )
    typer.echo(format_estimate(estimate_cost(config)))


@app.command()
def run(
    graph_dir: str = typer.Option(DEFAULT_GRAPH_DIR),
    encoding_dir: str = typer.Option(DEFAULT_ENCODING_DIR),
    question_dir: str = typer.Option(DEFAULT_QUESTION_DIR),
    db_path: str = typer.Option(DEFAULT_DB),
    run_id: Optional[str] = typer.Option(None, help="Resume/checkpoint id."),
    max_workers: int = typer.Option(4),
    models: str = typer.Option("opus,sonnet,haiku"),
    formats: Optional[str] = typer.Option(None, help="Restrict to these formats."),
    estimate_only: bool = typer.Option(
        False, "--estimate", help="Print the cost estimate and exit (no API calls)."
    ),
    yes: bool = typer.Option(
        False, "--yes", help="Confirm execution of real API calls."
    ),
) -> None:
    """Run the full benchmark grid (with caching, concurrency and resume).

    Always prints a cost estimate first. With ``--estimate`` it stops there.
    Otherwise ``--yes`` is required to proceed with real API calls.
    """
    from grb.pipeline import estimate_cost, format_estimate, run_benchmark

    config = _build_run_config(
        graph_dir, encoding_dir, question_dir, db_path, run_id, max_workers,
        models, formats,
    )

    typer.echo(format_estimate(estimate_cost(config)))

    if estimate_only:
        return
    if not yes:
        typer.echo(
            "\nThis will make real API calls. Re-run with --yes to proceed "
            "(or --estimate to only see the projection)."
        )
        raise typer.Exit(code=1)

    summary = run_benchmark(config)
    typer.echo(
        f"run_id={summary['run_id']} "
        f"accuracy={summary['accuracy']} "
        f"({summary['correct']}/{summary['total']}) "
        f"quota={summary['quota']}"
    )


# --------------------------------------------------------------------------- #
# export
# --------------------------------------------------------------------------- #
@app.command()
def export(
    db_path: str = typer.Option(DEFAULT_DB),
    out_path: str = typer.Option("data/results/export.json"),
    run_id: Optional[str] = typer.Option(None, help="Limit to one run."),
) -> None:
    """Export stored results to a JSON file."""
    from grb.storage import Storage

    with Storage(db_path) as store:
        path = store.export_json(out_path, run_id=run_id)
    typer.echo(f"Exported to {path}")


if __name__ == "__main__":
    app()
