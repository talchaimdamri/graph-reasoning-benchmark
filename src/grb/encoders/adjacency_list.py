"""Adjacency-list encoder.

Renders one line per node listing its outgoing neighbours (for directed graphs)
or all incident neighbours (for undirected graphs). Edge type/weight are shown in
parentheses when present, and repeated neighbours are listed once per parallel
edge so multi-edges survive a round-trip.

Format (example)::

    # directed adjacency list
    N0 -> N1 (calls, w=2.5), N2
    N1 -> N0
    N2 ->

The header records directedness so :func:`parse` can reconstruct the structure.
"""

from __future__ import annotations

from grb.models import BenchGraph, Edge, GraphMeta


def _neighbour_token(target: str, etype, weight) -> str:
    parts = []
    if etype is not None:
        parts.append(str(etype))
    if weight is not None:
        parts.append(f"w={weight}")
    if parts:
        return f"{target} ({', '.join(parts)})"
    return target


def encode(graph: BenchGraph) -> str:
    """Serialize ``graph`` as an adjacency list (one line per node)."""
    directed = graph.metadata.directed
    arrow = "->" if directed else "--"
    header = (
        f"# {'directed' if directed else 'undirected'} adjacency list "
        f"({len(graph.nodes)} nodes, {len(graph.edges)} edges)"
    )

    # Map node -> ordered list of (neighbour, type, weight) tokens.
    out: dict[str, list[str]] = {n: [] for n in graph.nodes}
    for e in graph.edges:
        out.setdefault(e.source, [])
        out.setdefault(e.target, [])
        out[e.source].append(_neighbour_token(e.target, e.type, e.weight))
        if not directed and e.source != e.target:
            # Undirected: also record the reverse direction.
            out[e.target].append(_neighbour_token(e.source, e.type, e.weight))

    lines = [header]
    for n in graph.nodes:
        neighbours = out.get(n, [])
        lines.append(f"{n} {arrow} {', '.join(neighbours)}")
    return "\n".join(lines)


def parse(content: str) -> BenchGraph:
    """Parse adjacency-list ``content`` back into a :class:`BenchGraph`.

    Only structural fidelity is guaranteed (nodes, directed edges with their
    type/weight). For undirected graphs the reverse-direction lines are folded
    back into single undirected edges.
    """
    lines = [ln for ln in content.splitlines() if ln.strip()]
    directed = True
    if lines and lines[0].startswith("#"):
        header = lines[0].lower()
        directed = "undirected" not in header
        lines = lines[1:]

    arrow = "->" if directed else "--"
    nodes: list[str] = []
    seen: set[str] = set()
    raw_edges: list[Edge] = []

    def add_node(n: str) -> None:
        if n not in seen:
            seen.add(n)
            nodes.append(n)

    for ln in lines:
        if arrow not in ln:
            continue
        src, rest = ln.split(arrow, 1)
        src = src.strip()
        add_node(src)
        rest = rest.strip()
        if not rest:
            continue
        for token in _split_neighbours(rest):
            target, etype, weight = _parse_neighbour(token)
            add_node(target)
            raw_edges.append(
                Edge(source=src, target=target, type=etype, weight=weight)
            )

    if directed:
        edges = raw_edges
    else:
        # Deduplicate symmetric edges: keep one per unordered pair (incl. attrs).
        edges = []
        used: set[tuple] = set()
        for e in raw_edges:
            key = (frozenset((e.source, e.target)), e.type, e.weight)
            if e.source == e.target:
                # self-loop appears once per line; keep all occurrences distinct
                edges.append(e)
                continue
            if key in used:
                used.discard(key)  # cancel the matched reverse
                continue
            used.add(key)
            edges.append(e)

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


def _split_neighbours(rest: str) -> list[str]:
    """Split a neighbour list on commas that are NOT inside parentheses."""
    tokens: list[str] = []
    depth = 0
    cur = []
    for ch in rest:
        if ch == "(":
            depth += 1
            cur.append(ch)
        elif ch == ")":
            depth = max(0, depth - 1)
            cur.append(ch)
        elif ch == "," and depth == 0:
            tokens.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    if "".join(cur).strip():
        tokens.append("".join(cur).strip())
    return tokens


def _parse_neighbour(token: str):
    """Parse ``"N1 (calls, w=2.5)"`` -> (target, type, weight)."""
    etype = None
    weight = None
    if "(" in token and token.endswith(")"):
        name, attrs = token[:-1].split("(", 1)
        target = name.strip()
        for part in attrs.split(","):
            part = part.strip()
            if part.startswith("w="):
                weight = float(part[2:])
            elif part:
                etype = part
    else:
        target = token.strip()
    return target, etype, weight
