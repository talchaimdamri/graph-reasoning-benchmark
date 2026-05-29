"""Mermaid flowchart encoder.

Emits a ``flowchart`` (directed) or ``graph`` block. Edge labels combine type and
weight; undirected graphs use ``---`` links, directed graphs use ``-->``. Isolated
nodes are declared on their own line so they appear in the diagram.

Mermaid node ids must be alphanumeric/underscore, so labels are emitted as
``N0["N0"]`` keeping the original label as display text.

Format (example)::

    flowchart LR
        N0["N0"]
        N0 -->|calls, w=2.5| N1["N1"]
"""

from __future__ import annotations

import re

from grb.models import BenchGraph

_SAFE = re.compile(r"[^A-Za-z0-9_]")


def _node_id(label: str) -> str:
    sid = _SAFE.sub("_", label)
    if not sid or sid[0].isdigit():
        sid = "n_" + sid
    return sid


def _edge_label(etype, weight) -> str:
    parts = []
    if etype is not None:
        parts.append(str(etype))
    if weight is not None:
        parts.append(f"w={weight}")
    return ", ".join(parts)


def encode(graph: BenchGraph) -> str:
    """Serialize ``graph`` as a Mermaid flowchart."""
    directed = graph.metadata.directed
    link = "-->" if directed else "---"
    lines = ["flowchart LR" if directed else "graph LR"]

    # Declare every node so isolated ones render too.
    for n in graph.nodes:
        lines.append(f'    {_node_id(n)}["{n}"]')

    for e in graph.edges:
        sid, tid = _node_id(e.source), _node_id(e.target)
        label = _edge_label(e.type, e.weight)
        if label:
            # Mermaid edge-label syntax: A -->|label| B  /  A ---|label| B
            lines.append(f"    {sid} {link}|{label}| {tid}")
        else:
            lines.append(f"    {sid} {link} {tid}")
    return "\n".join(lines)
