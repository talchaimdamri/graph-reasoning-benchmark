"""Natural-language encoder.

Describes the graph in plain English prose: a summary sentence, the node list,
and one sentence per edge mentioning relation type and weight when present.
Self-loops, multi-edges, isolated nodes, directedness and weighting are all
handled by the phrasing.
"""

from __future__ import annotations

from grb.models import BenchGraph


def encode(graph: BenchGraph) -> str:
    """Serialize ``graph`` as an English-prose description."""
    m = graph.metadata
    kind = "directed" if m.directed else "undirected"
    weighting = "weighted" if m.weighted else "unweighted"
    arrow_verb = "points to" if m.directed else "is connected to"

    lines = [
        f"This is a {kind}, {weighting} graph named {graph.id} "
        f"with {len(graph.nodes)} nodes and {len(graph.edges)} edges.",
    ]
    if graph.nodes:
        lines.append(f"The nodes are: {', '.join(graph.nodes)}.")
    else:
        lines.append("The graph has no nodes.")

    incident = set()
    for e in graph.edges:
        incident.add(e.source)
        incident.add(e.target)
    isolated = [n for n in graph.nodes if n not in incident]

    if graph.edges:
        lines.append("The edges are described below:")
        for e in graph.edges:
            if e.source == e.target:
                clause = f"Node {e.source} has a self-loop"
            else:
                clause = f"Node {e.source} {arrow_verb} node {e.target}"
            extras = []
            if e.type is not None:
                extras.append(f"with relation type '{e.type}'")
            if e.weight is not None:
                extras.append(f"with weight {e.weight}")
            if extras:
                clause += " " + " ".join(extras)
            lines.append(clause + ".")
    else:
        lines.append("The graph has no edges.")

    if isolated:
        lines.append(
            f"The following nodes have no edges: {', '.join(isolated)}."
        )
    return "\n".join(lines)
