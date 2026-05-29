"""Graphviz DOT-language encoder.

Emits a ``digraph`` or ``graph`` block. Node ids are quoted so any label is
valid. Edge attributes ``label`` (type) and ``weight`` are attached when present.
Isolated nodes are declared on their own statement line.

Format (example)::

    digraph G {
        "N0";
        "N1";
        "N0" -> "N1" [label="calls", weight=2.5];
    }
"""

from __future__ import annotations

from grb.models import BenchGraph


def _q(label: str) -> str:
    return '"' + label.replace("\\", "\\\\").replace('"', '\\"') + '"'


def encode(graph: BenchGraph) -> str:
    """Serialize ``graph`` as Graphviz DOT."""
    directed = graph.metadata.directed
    keyword = "digraph" if directed else "graph"
    connector = "->" if directed else "--"
    lines = [f"{keyword} G {{"]

    for n in graph.nodes:
        lines.append(f"    {_q(n)};")

    for e in graph.edges:
        attrs = []
        if e.type is not None:
            attrs.append(f'label="{e.type}"')
        if e.weight is not None:
            attrs.append(f"weight={e.weight}")
        attr_str = f" [{', '.join(attrs)}]" if attrs else ""
        lines.append(f"    {_q(e.source)} {connector} {_q(e.target)}{attr_str};")

    lines.append("}")
    return "\n".join(lines)
