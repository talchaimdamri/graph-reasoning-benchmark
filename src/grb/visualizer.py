"""Render a :class:`~grb.models.BenchGraph` to PNG and SVG.

Graphviz (via the ``graphviz`` Python package + the ``dot`` binary) is preferred
because it produces clean layouts and supports a true hierarchical layout for
tree-shaped graphs. When the ``dot`` executable is unavailable we fall back to a
Matplotlib + NetworkX rendering that still draws node labels and edge
weight/type labels.

Public API
----------
``render(graph, out_stem, fmt="both", layout="auto") -> list[Path]``
    Render to ``<out_stem>.png`` and/or ``<out_stem>.svg`` and return paths.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from grb.models import BenchGraph

RenderFormat = Literal["png", "svg", "both"]
Layout = Literal["auto", "hierarchical", "spring"]


def _edge_label(etype, weight) -> str:
    """Combine type and weight into a single short edge label."""
    parts = []
    if etype is not None:
        parts.append(str(etype))
    if weight is not None:
        parts.append(str(weight))
    return " ".join(parts)


def _is_hierarchical(graph: BenchGraph, layout: Layout) -> bool:
    if layout == "hierarchical":
        return True
    if layout == "spring":
        return False
    return graph.metadata.hierarchy_depth > 0


def _graphviz_available() -> bool:
    """True if the ``dot`` executable can be found."""
    import shutil

    return shutil.which("dot") is not None


# --------------------------------------------------------------------------- #
# Graphviz backend (preferred)
# --------------------------------------------------------------------------- #
def _render_graphviz(
    graph: BenchGraph, out_stem: Path, formats: list[str], hierarchical: bool
) -> list[Path]:
    import graphviz

    directed = graph.metadata.directed
    dot = graphviz.Digraph() if directed else graphviz.Graph()
    dot.attr(rankdir="TB" if hierarchical else "LR")
    if not hierarchical:
        dot.attr(layout="dot")
    dot.attr("node", shape="circle", fontsize="10")
    dot.attr("edge", fontsize="8")

    for n in graph.nodes:
        dot.node(n, label=n)
    for e in graph.edges:
        label = _edge_label(e.type, e.weight)
        dot.edge(e.source, e.target, label=label or None)

    written: list[Path] = []
    for fmt in formats:
        dot.format = fmt
        # graphviz appends the extension itself; render returns the path.
        rendered = dot.render(filename=str(out_stem), cleanup=True)
        written.append(Path(rendered))
    return written


# --------------------------------------------------------------------------- #
# Matplotlib fallback
# --------------------------------------------------------------------------- #
def _hierarchical_layout(g, graph: BenchGraph):
    """Layered positions: BFS depth from roots (nodes with no in-edges)."""
    import networkx as nx

    if graph.metadata.directed:
        roots = [n for n in g.nodes() if g.in_degree(n) == 0]
    else:
        roots = [graph.nodes[0]] if graph.nodes else []
    if not roots and graph.nodes:
        roots = [graph.nodes[0]]

    depth: dict = {}
    for r in roots:
        depth.setdefault(r, 0)
    # Simple BFS over an undirected view for layering.
    und = g.to_undirected()
    frontier = list(roots)
    seen = set(roots)
    while frontier:
        nxt = []
        for node in frontier:
            for nb in und.neighbors(node):
                if nb not in seen:
                    seen.add(nb)
                    depth[nb] = depth[node] + 1
                    nxt.append(nb)
        frontier = nxt
    for n in g.nodes():
        depth.setdefault(n, 0)

    # Group by level and spread horizontally.
    from collections import defaultdict

    levels = defaultdict(list)
    for n, d in depth.items():
        levels[d].append(n)
    pos = {}
    for d, members in levels.items():
        members = sorted(members)
        count = len(members)
        for i, n in enumerate(members):
            x = (i - (count - 1) / 2.0)
            pos[n] = (x, -float(d))
    return pos


def _render_matplotlib(
    graph: BenchGraph, out_stem: Path, formats: list[str], hierarchical: bool
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import networkx as nx

    g = graph.to_networkx()
    n = len(graph.nodes)
    size = max(6, min(20, n * 0.3 + 4))

    if hierarchical:
        pos = _hierarchical_layout(g, graph)
    else:
        pos = nx.spring_layout(g, seed=graph.metadata.seed, k=1.5 / max(1, n ** 0.5))

    node_size = max(80, 900 - n * 4)
    show_labels = n <= 60  # avoid an unreadable mess for large graphs

    fig, ax = plt.subplots(figsize=(size, size))
    nx.draw_networkx_nodes(g, pos, ax=ax, node_color="#cfe3ff",
                           edgecolors="#274472", node_size=node_size)
    edge_kwargs = dict(ax=ax, edge_color="#888888")
    if graph.metadata.directed:
        edge_kwargs.update(
            arrows=True, arrowstyle="-|>", arrowsize=10,
            connectionstyle="arc3,rad=0.08",
        )
    nx.draw_networkx_edges(g, pos, **edge_kwargs)
    if show_labels:
        nx.draw_networkx_labels(g, pos, ax=ax, font_size=max(6, 11 - n // 10))

    # Edge labels (type/weight). For multigraphs collapse to first key.
    if show_labels:
        edge_labels = {}
        for e in graph.edges:
            lbl = _edge_label(e.type, e.weight)
            if lbl:
                edge_labels[(e.source, e.target)] = lbl
        if edge_labels:
            try:
                nx.draw_networkx_edge_labels(
                    g, pos, edge_labels=edge_labels, ax=ax, font_size=6,
                    label_pos=0.5, rotate=False,
                )
            except Exception:
                pass

    ax.set_title(graph.id, fontsize=10)
    ax.axis("off")
    fig.tight_layout()

    written: list[Path] = []
    for fmt in formats:
        path = out_stem.with_suffix(f".{fmt}")
        fig.savefig(path, format=fmt, dpi=120, bbox_inches="tight")
        written.append(path)
    plt.close(fig)
    return written


def render(
    graph: BenchGraph,
    out_stem: str | Path,
    fmt: RenderFormat = "both",
    layout: Layout = "auto",
    force_backend: Literal["auto", "graphviz", "matplotlib"] = "auto",
) -> list[Path]:
    """Render ``graph`` to image files sharing ``out_stem`` (no extension).

    Returns the list of written file paths. Uses graphviz when available unless
    ``force_backend`` overrides it; otherwise falls back to matplotlib.
    """
    out_stem = Path(out_stem)
    out_stem.parent.mkdir(parents=True, exist_ok=True)

    formats = ["png", "svg"] if fmt == "both" else [fmt]
    hierarchical = _is_hierarchical(graph, layout)

    use_graphviz = force_backend == "graphviz" or (
        force_backend == "auto" and _graphviz_available()
    )
    if use_graphviz:
        try:
            return _render_graphviz(graph, out_stem, formats, hierarchical)
        except Exception:
            if force_backend == "graphviz":
                raise
            # fall through to matplotlib
    return _render_matplotlib(graph, out_stem, formats, hierarchical)
