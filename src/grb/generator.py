"""Seed-reproducible graph generation for the Graph Reasoning Benchmark.

Provides :func:`generate_graph` (random / hierarchical / scale-free models) and
:func:`make_tiered_graph`, a tier-calibrated helper that targets a typical
encoded token budget per tier (small ~500, medium ~2000, large ~20000) measured
with the ``cl100k_base`` tiktoken encoding on a natural-language-ish description.

Everything is keyed off an integer ``seed`` so that the same inputs always
produce byte-identical :class:`~grb.models.BenchGraph` objects.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Literal, Optional

from grb.models import BenchGraph, Edge, GraphMeta, Tier

GraphModel = Literal["random", "hierarchical", "scale_free"]

# Repository-relative path where tier calibration data is recorded.
_CALIBRATION_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "tier_calibration.json"
)


def _node_label(i: int) -> str:
    """Stable, human-readable node label (English): N0, N1, ..."""
    return f"N{i}"


def _round_weight(value: float) -> float:
    """Round weights to 2 decimals for stable, readable serialization."""
    return round(value, 2)


def _make_edge(
    rng: random.Random,
    source: str,
    target: str,
    weighted: bool,
    weight_range: tuple[float, float],
    multi_edge_types: Optional[list[str]],
) -> Edge:
    """Construct one Edge, sampling weight / type deterministically from ``rng``."""
    weight = None
    if weighted:
        lo, hi = weight_range
        weight = _round_weight(rng.uniform(lo, hi))
    etype = None
    if multi_edge_types:
        etype = rng.choice(multi_edge_types)
    return Edge(source=source, target=target, type=etype, weight=weight)


def _validate_params(
    nodes: int,
    hierarchy_depth: int,
    weight_range: tuple[float, float],
    multi_edge_types: Optional[list[str]],
    model: GraphModel,
) -> None:
    if nodes < 1:
        raise ValueError(f"nodes must be >= 1, got {nodes}")
    if hierarchy_depth < 0:
        raise ValueError(f"hierarchy_depth must be >= 0, got {hierarchy_depth}")
    if model == "hierarchical" and hierarchy_depth < 1:
        raise ValueError("hierarchical model requires hierarchy_depth >= 1")
    lo, hi = weight_range
    if lo > hi:
        raise ValueError(f"weight_range lower bound {lo} > upper bound {hi}")
    if multi_edge_types is not None:
        if len(multi_edge_types) == 0:
            raise ValueError("multi_edge_types, if provided, must be non-empty")
        if len(set(multi_edge_types)) != len(multi_edge_types):
            raise ValueError("multi_edge_types must be unique")
    if model not in ("random", "hierarchical", "scale_free"):
        raise ValueError(f"unknown model: {model!r}")


def _gen_random(
    rng: random.Random,
    node_list: list[str],
    directed: bool,
    edge_prob: float,
    weighted: bool,
    weight_range: tuple[float, float],
    multi_edge_types: Optional[list[str]],
) -> list[Edge]:
    """Erdos-Renyi style random graph. Deterministic ordered iteration."""
    edges: list[Edge] = []
    n = len(node_list)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if not directed and j <= i:
                continue
            if rng.random() < edge_prob:
                edges.append(
                    _make_edge(
                        rng,
                        node_list[i],
                        node_list[j],
                        weighted,
                        weight_range,
                        multi_edge_types,
                    )
                )
    return edges


def _gen_hierarchical(
    rng: random.Random,
    node_list: list[str],
    hierarchy_depth: int,
    directed: bool,
    weighted: bool,
    weight_range: tuple[float, float],
    multi_edge_types: Optional[list[str]],
    cross_edge_prob: float,
) -> list[Edge]:
    """Tree of the requested depth plus a sprinkle of cross-edges.

    Nodes are assigned to levels by repeatedly attaching each new node to a
    random already-placed node, capping the tree depth at ``hierarchy_depth``.
    Cross-edges connect nodes at the same or adjacent levels to add structure.
    """
    edges: list[Edge] = []
    n = len(node_list)
    if n == 0:
        return edges

    # level[idx] = depth of node_list[idx]; parent[idx] = index of parent or None.
    level = [0] * n
    parents: list[Optional[int]] = [None] * n
    # Candidate parents are nodes whose level < hierarchy_depth (so children fit).
    placed = [0]
    for idx in range(1, n):
        candidates = [p for p in placed if level[p] < hierarchy_depth]
        if not candidates:
            candidates = placed  # fallback: attach anywhere (keeps it connected)
        parent = rng.choice(candidates)
        parents[idx] = parent
        level[idx] = min(level[parent] + 1, hierarchy_depth)
        placed.append(idx)
        edges.append(
            _make_edge(
                rng,
                node_list[parent],
                node_list[idx],
                weighted,
                weight_range,
                multi_edge_types,
            )
        )

    # Cross-edges between distinct non-parent/child pairs at close levels.
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if not directed and j <= i:
                continue
            if parents[j] == i or parents[i] == j:
                continue
            if abs(level[i] - level[j]) > 1:
                continue
            if rng.random() < cross_edge_prob:
                edges.append(
                    _make_edge(
                        rng,
                        node_list[i],
                        node_list[j],
                        weighted,
                        weight_range,
                        multi_edge_types,
                    )
                )
    return edges


def _gen_scale_free(
    rng: random.Random,
    node_list: list[str],
    directed: bool,
    m: int,
    weighted: bool,
    weight_range: tuple[float, float],
    multi_edge_types: Optional[list[str]],
) -> list[Edge]:
    """Barabasi-Albert style preferential attachment (deterministic).

    Each new node connects to ``m`` existing nodes chosen with probability
    proportional to their current degree.
    """
    edges: list[Edge] = []
    n = len(node_list)
    if n <= 1:
        return edges

    m = max(1, min(m, n - 1))
    # Seed with a small connected core of size m (a path).
    repeated: list[int] = []  # degree-weighted bag of node indices
    for i in range(1, m):
        edges.append(
            _make_edge(
                rng,
                node_list[i - 1],
                node_list[i],
                weighted,
                weight_range,
                multi_edge_types,
            )
        )
        repeated.extend([i - 1, i])
    if not repeated:
        repeated = [0]

    for new in range(m, n):
        targets: set[int] = set()
        attempts = 0
        while len(targets) < m and attempts < 10 * m:
            attempts += 1
            targets.add(rng.choice(repeated))
        for t in sorted(targets):
            edges.append(
                _make_edge(
                    rng,
                    node_list[new],
                    node_list[t],
                    weighted,
                    weight_range,
                    multi_edge_types,
                )
            )
            repeated.extend([new, t])
    return edges


def generate_graph(
    nodes: int,
    *,
    hierarchy_depth: int = 0,
    directed: bool = False,
    weighted: bool = False,
    weight_range: tuple[float, float] = (1.0, 10.0),
    multi_edge_types: Optional[list[str]] = None,
    seed: int = 0,
    model: GraphModel = "random",
    edge_prob: float = 0.3,
    cross_edge_prob: float = 0.1,
    scale_free_m: int = 2,
    tier: Tier = "small",
    graph_id: Optional[str] = None,
) -> BenchGraph:
    """Generate a reproducible :class:`BenchGraph`.

    Parameters
    ----------
    nodes:
        Number of nodes (>= 1).
    hierarchy_depth:
        Tree depth for the ``hierarchical`` model; recorded in metadata for all
        models.
    directed / weighted:
        Toggle a directed and/or weighted graph.
    weight_range:
        ``(low, high)`` inclusive bounds for sampled edge weights.
    multi_edge_types:
        If provided, each edge is tagged with one of these type labels and the
        graph is marked ``multi_edge`` (a NetworkX multigraph).
    seed:
        Integer seed; identical inputs (incl. seed) yield identical graphs.
    model:
        ``"random"``, ``"hierarchical"`` or ``"scale_free"``.

    Returns
    -------
    BenchGraph
        Fully populated, validated graph object.
    """
    _validate_params(nodes, hierarchy_depth, weight_range, multi_edge_types, model)

    rng = random.Random(seed)
    node_list = [_node_label(i) for i in range(nodes)]
    multi_edge = multi_edge_types is not None

    if model == "random":
        edges = _gen_random(
            rng, node_list, directed, edge_prob, weighted, weight_range, multi_edge_types
        )
    elif model == "hierarchical":
        edges = _gen_hierarchical(
            rng,
            node_list,
            hierarchy_depth,
            directed,
            weighted,
            weight_range,
            multi_edge_types,
            cross_edge_prob,
        )
    else:  # scale_free
        edges = _gen_scale_free(
            rng,
            node_list,
            directed,
            scale_free_m,
            weighted,
            weight_range,
            multi_edge_types,
        )

    meta = GraphMeta(
        directed=directed,
        weighted=weighted,
        multi_edge=multi_edge,
        hierarchy_depth=hierarchy_depth,
        seed=seed,
        tier=tier,
        num_nodes=nodes,
    )
    gid = graph_id or f"{model}-{tier}-n{nodes}-s{seed}"
    return BenchGraph(id=gid, nodes=node_list, edges=edges, metadata=meta)


# --------------------------------------------------------------------------- #
# Tier calibration
# --------------------------------------------------------------------------- #

# Empirically chosen so the NL description lands near the target token budget.
# Tuned against describe_graph + cl100k_base (see calibrate_tiers()).
_TIER_SPEC: dict[str, dict] = {
    "small": {
        "nodes": 8,
        "model": "random",
        "directed": True,
        "weighted": True,
        "multi_edge_types": None,
        "hierarchy_depth": 0,
        "edge_prob": 0.6,
        "target_tokens": 500,
    },
    "medium": {
        "nodes": 20,
        "model": "hierarchical",
        "directed": True,
        "weighted": True,
        "multi_edge_types": ["calls", "imports", "extends"],
        "hierarchy_depth": 3,
        "edge_prob": 0.3,
        "cross_edge_prob": 0.32,
        "target_tokens": 2000,
    },
    "large": {
        "nodes": 150,
        "model": "scale_free",
        "directed": True,
        "weighted": True,
        "multi_edge_types": ["calls", "imports", "extends", "uses"],
        "hierarchy_depth": 0,
        "scale_free_m": 7,
        "target_tokens": 20000,
    },
}


def describe_graph(graph: BenchGraph) -> str:
    """Natural-language-ish description of a graph used for token calibration.

    This intentionally reads like prose so the token measurement reflects what
    an LLM-facing encoding would roughly cost, independent of the formal
    encoder modules built later.
    """
    m = graph.metadata
    kind = "directed" if m.directed else "undirected"
    weighting = "weighted" if m.weighted else "unweighted"
    lines = [
        f"This is a {kind}, {weighting} graph named {graph.id} "
        f"with {len(graph.nodes)} nodes and {len(graph.edges)} edges.",
        f"The nodes are: {', '.join(graph.nodes)}.",
        "The edges are described below:",
    ]
    arrow = "->" if m.directed else "--"
    for e in graph.edges:
        parts = [f"Node {e.source} {arrow} node {e.target}"]
        if e.type is not None:
            parts.append(f"with relation type '{e.type}'")
        if e.weight is not None:
            parts.append(f"and weight {e.weight}")
        lines.append(" ".join(parts) + ".")
    return "\n".join(lines)


def count_tokens(text: str) -> int:
    """Token count using the ``cl100k_base`` encoding (lazy import)."""
    import tiktoken

    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


def make_tiered_graph(tier: Tier, seed: int = 0) -> BenchGraph:
    """Produce a graph whose NL description targets the tier's token budget."""
    if tier not in _TIER_SPEC:
        raise ValueError(f"unknown tier: {tier!r}")
    spec = dict(_TIER_SPEC[tier])
    spec.pop("target_tokens", None)
    nodes = spec.pop("nodes")
    return generate_graph(nodes, seed=seed, tier=tier, **spec)


def calibrate_tiers(
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
    write: bool = True,
) -> dict:
    """Measure encoded token sizes per tier over several seeds; record results.

    Returns the calibration dict and (optionally) writes it to
    ``data/tier_calibration.json``.
    """
    report: dict = {
        "encoding": "cl100k_base",
        "description": "Token counts of describe_graph() output per tier.",
        "tiers": {},
    }
    for tier, spec in _TIER_SPEC.items():
        counts: list[int] = []
        node_counts: list[int] = []
        edge_counts: list[int] = []
        for s in seeds:
            g = make_tiered_graph(tier, seed=s)  # type: ignore[arg-type]
            counts.append(count_tokens(describe_graph(g)))
            node_counts.append(len(g.nodes))
            edge_counts.append(len(g.edges))
        report["tiers"][tier] = {
            "nodes": spec["nodes"],
            "model": spec["model"],
            "target_tokens": spec["target_tokens"],
            "seeds": list(seeds),
            "token_counts": counts,
            "mean_tokens": round(sum(counts) / len(counts), 1),
            "min_tokens": min(counts),
            "max_tokens": max(counts),
            "mean_edges": round(sum(edge_counts) / len(edge_counts), 1),
        }
    if write:
        _CALIBRATION_PATH.parent.mkdir(parents=True, exist_ok=True)
        _CALIBRATION_PATH.write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":  # pragma: no cover
    rep = calibrate_tiers()
    for t, info in rep["tiers"].items():
        print(
            f"{t:7s} target={info['target_tokens']:>6} "
            f"mean={info['mean_tokens']:>8} "
            f"range=[{info['min_tokens']},{info['max_tokens']}] "
            f"nodes={info['nodes']} mean_edges={info['mean_edges']}"
        )
