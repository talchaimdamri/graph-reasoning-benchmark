"""Command-line interface for grb (stub)."""

import typer

app = typer.Typer(help="Graph Reasoning Benchmark CLI.")


@app.command()
def version() -> None:
    """Print the grb version."""
    from grb import __version__

    typer.echo(__version__)


if __name__ == "__main__":
    app()
