"""Adjacency-matrix encoder.

Produces a labelled square matrix. For unweighted graphs the cell value is an
edge count (so multi-edges and self-loops are visible); for weighted graphs the
cell holds the weight of the (last) edge, with multi-edges still counted via the
sum is avoided — instead weighted cells show the weight and the count is encoded
by repetition only in unweighted mode. To keep the round-trip lossless for the
common (single-edge) case we store the weight when present and 1/0 otherwise.

Format (example)::

    # directed adjacency matrix, weighted=True
        N0   N1   N2
    N0   .    2.5  .
    N1   .    .    .
    N2   1    .    .
"""

from __future__ import annotations

from grb.models import BenchGraph, Edge, GraphMeta

_EMPTY = "."


def encode(graph: BenchGraph) -> str:
    """Serialize ``graph`` as a labelled adjacency matrix."""
    directed = graph.metadata.directed
    weighted = graph.metadata.weighted
    nodes = graph.nodes
    idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)

    # cell[i][j] accumulates either a count (unweighted) or last weight (weighted).
    counts = [[0 for _ in range(n)] for _ in range(n)]
    weights: list[list] = [[None for _ in range(n)] for _ in range(n)]
    for e in graph.edges:
        i, j = idx[e.source], idx[e.target]
        counts[i][j] += 1
        if e.weight is not None:
            weights[i][j] = e.weight
        if not directed and i != j:
            counts[j][i] += 1
            if e.weight is not None:
                weights[j][i] = e.weight

    def cell(i: int, j: int) -> str:
        if counts[i][j] == 0:
            return _EMPTY
        if weighted and weights[i][j] is not None:
            return str(weights[i][j])
        return str(counts[i][j])

    width = max([2] + [len(x) for x in nodes])
    body_width = max(
        [width]
        + [len(cell(i, j)) for i in range(n) for j in range(n)]
    )
    col_w = max(width, body_width) + 1

    header = (
        f"# {'directed' if directed else 'undirected'} adjacency matrix, "
        f"weighted={weighted}"
    )
    # Column header row.
    top = " " * (width + 1) + "".join(c.ljust(col_w) for c in nodes)
    lines = [header, top.rstrip()]
    for i, row_label in enumerate(nodes):
        row = row_label.ljust(width + 1)
        row += "".join(cell(i, j).ljust(col_w) for j in range(n))
        lines.append(row.rstrip())
    return "\n".join(lines)


def parse(content: str) -> BenchGraph:
    """Parse an adjacency matrix back into a :class:`BenchGraph`."""
    lines = [ln for ln in content.splitlines() if ln.strip()]
    directed = True
    weighted = False
    if lines and lines[0].startswith("#"):
        header = lines[0].lower()
        directed = "undirected" not in header
        weighted = "weighted=true" in header.replace(" ", "")
        lines = lines[1:]

    if not lines:
        meta = GraphMeta(
            directed=directed, weighted=weighted, multi_edge=False,
            hierarchy_depth=0, seed=0, tier="small", num_nodes=0,
        )
        return BenchGraph(id="parsed", nodes=[], edges=[], metadata=meta)

    col_labels = lines[0].split()
    nodes = list(col_labels)
    edges: list[Edge] = []

    for row_line in lines[1:]:
        toks = row_line.split()
        if not toks:
            continue
        row_label = toks[0]
        if row_label not in nodes:
            nodes.append(row_label)
        cells = toks[1:]
        i = col_labels.index(row_label) if row_label in col_labels else None
        for col_j, raw in enumerate(cells):
            if raw == _EMPTY:
                continue
            if col_j >= len(col_labels):
                continue
            target = col_labels[col_j]
            if weighted:
                try:
                    weight = float(raw)
                except ValueError:
                    weight = None
                count = 1
            else:
                weight = None
                try:
                    count = int(float(raw))
                except ValueError:
                    count = 1
            if not directed:
                # Only emit the upper triangle (i <= j) to avoid duplicates.
                if i is not None and i > col_j:
                    continue
            for _ in range(max(1, count)):
                edges.append(
                    Edge(source=row_label, target=target,
                         type=None, weight=weight)
                )

    multi = False  # matrix cannot distinguish edge types
    meta = GraphMeta(
        directed=directed, weighted=weighted, multi_edge=multi,
        hierarchy_depth=0, seed=0, tier="small", num_nodes=len(nodes),
    )
    return BenchGraph(id="parsed", nodes=nodes, edges=edges, metadata=meta)
