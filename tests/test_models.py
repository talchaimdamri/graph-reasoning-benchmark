"""Round-trip tests for BenchGraph <-> NetworkX conversion."""

import networkx as nx

from grb.models import BenchGraph, Edge, GraphMeta


def test_directed_weighted_multi_edge_round_trip():
    meta = GraphMeta(
        directed=True,
        weighted=True,
        multi_edge=True,
        hierarchy_depth=2,
        seed=42,
        tier="small",
        num_nodes=3,
    )
    bg = BenchGraph(
        id="g-dwm",
        nodes=["A", "B", "C"],
        edges=[
            Edge(source="A", target="B", type="calls", weight=1.5),
            Edge(source="A", target="B", type="imports", weight=2.0),
            Edge(source="B", target="C", weight=3.0),
        ],
        metadata=meta,
    )

    g = bg.to_networkx()
    assert isinstance(g, nx.MultiDiGraph)
    assert g.is_directed() and g.is_multigraph()
    assert g.number_of_nodes() == 3
    assert g.number_of_edges() == 3
    # Parallel A->B edges preserved.
    assert g.number_of_edges("A", "B") == 2

    rt = BenchGraph.from_networkx(
        g, id="g-dwm", metadata=meta
    )
    rt_g = rt.to_networkx()
    assert nx.is_isomorphic(g, rt_g)
    assert sorted(rt.nodes) == ["A", "B", "C"]
    assert len(rt.edges) == 3
    weights = sorted(e.weight for e in rt.edges)
    assert weights == [1.5, 2.0, 3.0]
    assert rt.metadata.directed and rt.metadata.multi_edge and rt.metadata.weighted


def test_undirected_unweighted_round_trip():
    meta = GraphMeta(
        directed=False,
        weighted=False,
        multi_edge=False,
        hierarchy_depth=0,
        seed=7,
        tier="small",
        num_nodes=4,
    )
    bg = BenchGraph(
        id="g-uu",
        nodes=["N1", "N2", "N3", "N4"],
        edges=[
            Edge(source="N1", target="N2"),
            Edge(source="N2", target="N3"),
            Edge(source="N3", target="N4"),
        ],
        metadata=meta,
    )

    g = bg.to_networkx()
    assert isinstance(g, nx.Graph)
    assert not g.is_directed() and not g.is_multigraph()
    assert g.number_of_nodes() == 4
    assert g.number_of_edges() == 3

    # Infer metadata from the graph (no metadata passed).
    rt = BenchGraph.from_networkx(g, id="g-uu", seed=7, tier="small")
    assert rt.metadata.directed is False
    assert rt.metadata.multi_edge is False
    assert rt.metadata.weighted is False
    assert rt.metadata.num_nodes == 4
    assert all(e.weight is None for e in rt.edges)
    assert nx.is_isomorphic(g, rt.to_networkx())
