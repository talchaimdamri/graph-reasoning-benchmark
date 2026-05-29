"""Tests for grb.visualizer rendering (matplotlib fallback always works)."""

from __future__ import annotations

from grb.generator import generate_graph, make_tiered_graph
from grb.visualizer import render


def test_render_matplotlib_png_svg(tmp_path):
    g = generate_graph(8, seed=1, directed=True, weighted=True)
    paths = render(g, tmp_path / "g", fmt="both", force_backend="matplotlib")
    assert len(paths) == 2
    exts = {p.suffix for p in paths}
    assert exts == {".png", ".svg"}
    for p in paths:
        assert p.exists() and p.stat().st_size > 0


def test_render_hierarchical(tmp_path):
    g = make_tiered_graph("medium", seed=2)
    paths = render(g, tmp_path / "h", fmt="png", layout="hierarchical",
                   force_backend="matplotlib")
    assert len(paths) == 1
    assert paths[0].exists() and paths[0].stat().st_size > 0


def test_render_auto_backend(tmp_path):
    g = generate_graph(6, seed=3)
    paths = render(g, tmp_path / "a", fmt="svg")
    assert len(paths) == 1
    assert paths[0].exists()
