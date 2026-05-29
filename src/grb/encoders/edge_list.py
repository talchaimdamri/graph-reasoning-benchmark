"""Edge-list encoder.

One line per edge: ``source <arrow> target`` plus optional ``type`` / ``weight``
columns. Isolated nodes (no incident edges) are listed in a trailing comment so
they survive a round-trip.

Format (example)::

    # directed edge list (3 nodes, 2 edges)
    N0 -> N1 [type=calls] [weight=2.5]
    N0 -> N2
    # isolated: N3
"""

from __future__ import annotations

from grb.models import BenchGraph, Edge, GraphMeta


def encode(graph: BenchGraph) -> str:
    """Serialize ``graph`` as a flat edge list."""
    directed = graph.metadata.directed
    arrow = "->" if directed else "--"
    header = (
        f"# {'directed' if directed else 'undirected'} edge list "
        f"({len(graph.nodes)} nodes, {len(graph.edges)} edges)"
    )
    lines = [header]
    for e in graph.edges:
        line = f"{e.source} {arrow} {e.target}"
        if e.type is not None:
            line += f" [type={e.type}]"
        if e.weight is not None:
            line += f" [weight={e.weight}]"
        lines.append(line)

    incident = set()
    for e in graph.edges:
        incident.add(e.source)
        incident.add(e.target)
    isolated = [n for n in graph.nodes if n not in incident]
    if isolated:
        lines.append(f"# isolated: {', '.join(isolated)}")
    return "\n".join(lines)


def parse(content: str) -> BenchGraph:
    """Parse an edge list back into a :class:`BenchGraph`."""
    lines = content.splitlines()
    directed = True
    if lines and lines[0].startswith("#"):
        directed = "undirected" not in lines[0].lower()

    arrow = "->" if directed else "--"
    nodes: list[str] = []
    seen: set[str] = set()
    edges: list[Edge] = []

    def add_node(n: str) -> None:
        if n not in seen:
            seen.add(n)
            nodes.append(n)

    for ln in lines:
        stripped = ln.strip()
        if not stripped:
            continue
        if stripped.startswith("# isolated:"):
            for n in stripped[len("# isolated:"):].split(","):
                n = n.strip()
                if n:
                    add_node(n)
            continue
        if stripped.startswith("#"):
            continue
        if arrow not in stripped:
            continue
        core = stripped
        etype = None
        weight = None
        # Strip bracketed attributes from the right.
        while core.rstrip().endswith("]") and "[" in core:
            head, attr = core.rsplit("[", 1)
            attr = attr.rstrip().rstrip("]").strip()
            core = head.strip()
            if attr.startswith("type="):
                etype = attr[len("type="):]
            elif attr.startswith("weight="):
                weight = float(attr[len("weight="):])
        src, tgt = core.split(arrow, 1)
        src, tgt = src.strip(), tgt.strip()
        add_node(src)
        add_node(tgt)
        edges.append(Edge(source=src, target=tgt, type=etype, weight=weight))

    weighted = any(e.weight is not None for e in edges)
    multi = any(e.type is not None for e in edges)
    meta = GraphMeta(
        directed=directed,
        weighted=weighted,
        multi_edge=multi,
        hierarchy_depth=0,
        seed=0,
        tier="small",
        num_nodes=len(nodes),
    )
    return BenchGraph(id="parsed", nodes=nodes, edges=edges, metadata=meta)
