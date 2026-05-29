"""Tests for grb.encoder and grb.encoders.

Covers: structural round-trips for parseable formats (adjacency_list, edge_list,
matrix), all formats running without error across directed/undirected/weighted/
multi-edge/self-loop graphs, and positive token counts for text formats.
"""

from __future__ import annotations

import pytest

from grb.encoder import ALL_FORMATS, TEXT_ENCODERS, encode_graph
from grb.encoders import adjacency_list, edge_list, matrix
from grb.generator import generate_graph, make_tiered_graph
from grb.models import BenchGraph, Edge, GraphMeta


# --------------------------------------------------------------------------- #
# Fixtures: a representative spread of graph shapes.
# --------------------------------------------------------------------------- #
def _meta(**kw):
    base = dict(
        directed=True, weighted=False, multi_edge=False,
        hierarchy_depth=0, seed=0, tier="small", num_nodes=0,
    )
    base.update(kw)
    return GraphMeta(**base)


def graph_directed_weighted():
    return generate_graph(8, seed=1, directed=True, weighted=True)


def graph_undirected_unweighted():
    return generate_graph(8, seed=2, directed=False, weighted=False)


def graph_undirected_weighted():
    return generate_graph(7, seed=3, directed=False, weighted=True)


def graph_multi_edge():
    return generate_graph(
        6, seed=4, directed=True,
        multi_edge_types=["calls", "imports"],
    )


def graph_self_loop():
    nodes = ["N0", "N1", "N2", "N3"]
    edges = [
        Edge(source="N0", target="N0", weight=2.5),  # self-loop
        Edge(source="N0", target="N1", weight=1.0),
        Edge(source="N2", target="N2", weight=3.0),  # self-loop
        # N3 isolated
    ]
    return BenchGraph(
        id="selfloop",
        nodes=nodes,
        edges=edges,
        metadata=_meta(directed=True, weighted=True, num_nodes=4),
    )


def graph_with_isolated():
    nodes = ["N0", "N1", "N2", "N3", "N4"]
    edges = [Edge(source="N0", target="N1"), Edge(source="N1", target="N2")]
    return BenchGraph(
        id="iso",
        nodes=nodes,
        edges=edges,
        metadata=_meta(directed=True, num_nodes=5),
    )


ALL_GRAPHS = [
    graph_directed_weighted,
    graph_undirected_unweighted,
    graph_undirected_weighted,
    graph_multi_edge,
    graph_self_loop,
    graph_with_isolated,
]


# --------------------------------------------------------------------------- #
# All formats run without error & produce content.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("make_graph", ALL_GRAPHS, ids=lambda f: f.__name__)
@pytest.mark.parametrize("fmt", list(TEXT_ENCODERS.keys()))
def test_text_format_runs_and_has_tokens(make_graph, fmt):
    g = make_graph()
    enc = encode_graph(g, formats=[fmt])[fmt]
    assert enc.format == fmt
    assert enc.graph_id == g.id
    assert isinstance(enc.content, str) and enc.content.strip()
    assert enc.token_count > 0
    assert enc.tokens_per_node > 0
    assert enc.tokens_per_edge > 0


def test_encode_all_text_formats_at_once():
    g = graph_directed_weighted()
    fmts = list(TEXT_ENCODERS.keys())
    out = encode_graph(g, formats=fmts)
    assert set(out.keys()) == set(fmts)
    for enc in out.values():
        assert enc.token_count > 0


def test_unknown_format_raises():
    g = graph_directed_weighted()
    with pytest.raises(ValueError):
        encode_graph(g, formats=["nonsense"])


# --------------------------------------------------------------------------- #
# Visual format: returns a real file path, token_count == 0.
# --------------------------------------------------------------------------- #
def test_visual_format(tmp_path):
    g = generate_graph(6, seed=5, directed=True)
    out = encode_graph(g, formats=["visual"], visual_out_dir=tmp_path)
    enc = out["visual"]
    assert enc.format == "visual"
    assert enc.token_count == 0
    from pathlib import Path

    p = Path(enc.content)
    assert p.exists() and p.stat().st_size > 0


def test_all_formats_constant_complete():
    expected = {
        "adjacency_list", "edge_list", "mermaid", "dot",
        "natural_language", "matrix", "visual",
    }
    assert set(ALL_FORMATS) == expected


# --------------------------------------------------------------------------- #
# Structural round-trips.
# --------------------------------------------------------------------------- #
def _edge_key_set(g: BenchGraph, directed: bool, with_type=True, with_weight=True):
    """Multiset of edge keys for structural comparison."""
    from collections import Counter

    keys = []
    for e in g.edges:
        endpoints = (e.source, e.target) if directed else frozenset((e.source, e.target))
        key = [endpoints]
        if with_type:
            key.append(e.type)
        if with_weight:
            key.append(e.weight)
        keys.append(tuple(key))
    return Counter(keys)


@pytest.mark.parametrize(
    "make_graph",
    [graph_directed_weighted, graph_undirected_unweighted,
     graph_undirected_weighted, graph_multi_edge,
     graph_self_loop, graph_with_isolated],
    ids=lambda f: f.__name__,
)
def test_adjacency_list_roundtrip(make_graph):
    g = make_graph()
    content = adjacency_list.encode(g)
    back = adjacency_list.parse(content)
    assert set(back.nodes) == set(g.nodes)
    directed = g.metadata.directed
    assert _edge_key_set(back, directed) == _edge_key_set(g, directed)


@pytest.mark.parametrize(
    "make_graph",
    [graph_directed_weighted, graph_undirected_unweighted,
     graph_undirected_weighted, graph_multi_edge,
     graph_self_loop, graph_with_isolated],
    ids=lambda f: f.__name__,
)
def test_edge_list_roundtrip(make_graph):
    g = make_graph()
    content = edge_list.encode(g)
    back = edge_list.parse(content)
    assert set(back.nodes) == set(g.nodes)
    directed = g.metadata.directed
    # Edge list stores edges verbatim in source order; compare as multiset.
    assert _edge_key_set(back, directed) == _edge_key_set(g, directed)


@pytest.mark.parametrize(
    "make_graph",
    [graph_directed_weighted, graph_undirected_unweighted,
     graph_undirected_weighted, graph_self_loop, graph_with_isolated],
    ids=lambda f: f.__name__,
)
def test_matrix_roundtrip(make_graph):
    """Matrix round-trips structure (endpoints + weight), ignoring edge type.

    Multi-edge type graphs are excluded because the matrix cannot carry types;
    those are covered by the run-without-error tests instead.
    """
    g = make_graph()
    content = matrix.encode(g)
    back = matrix.parse(content)
    assert set(back.nodes) == set(g.nodes)
    directed = g.metadata.directed
    # Compare endpoints + weight (type is not representable in a matrix).
    assert _edge_key_set(back, directed, with_type=False) == _edge_key_set(
        g, directed, with_type=False
    )


def test_matrix_multi_edge_counts_unweighted():
    """Parallel edges in an unweighted graph survive as a cell count."""
    nodes = ["N0", "N1"]
    edges = [
        Edge(source="N0", target="N1"),
        Edge(source="N0", target="N1"),
        Edge(source="N0", target="N1"),
    ]
    g = BenchGraph(
        id="multi", nodes=nodes, edges=edges,
        metadata=_meta(directed=True, weighted=False, multi_edge=True, num_nodes=2),
    )
    content = matrix.encode(g)
    back = matrix.parse(content)
    assert len(back.edges) == 3


# --------------------------------------------------------------------------- #
# Tier graphs encode end-to-end.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("tier", ["small", "medium", "large"])
def test_tier_graphs_encode(tier):
    g = make_tiered_graph(tier, seed=0)
    out = encode_graph(g, formats=list(TEXT_ENCODERS.keys()))
    for enc in out.values():
        assert enc.token_count > 0
