"""Tests for grb.generator: reproducibility, params, multi-edge, tiers."""

from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import pytest

from grb.generator import (
    calibrate_tiers,
    count_tokens,
    describe_graph,
    generate_graph,
    make_tiered_graph,
)
from grb.models import BenchGraph

MODELS = ["random", "hierarchical", "scale_free"]


def _kw(model):
    return {"hierarchy_depth": 3} if model == "hierarchical" else {}


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("model", MODELS)
def test_same_seed_identical(model):
    g1 = generate_graph(15, seed=42, weighted=True, directed=True,
                        model=model, **_kw(model))
    g2 = generate_graph(15, seed=42, weighted=True, directed=True,
                        model=model, **_kw(model))
    assert g1.model_dump() == g2.model_dump()


@pytest.mark.parametrize("model", MODELS)
def test_different_seed_differs(model):
    g1 = generate_graph(15, seed=1, model=model, **_kw(model))
    g2 = generate_graph(15, seed=2, model=model, **_kw(model))
    # Edge sets almost certainly differ for these sizes.
    assert g1.model_dump() != g2.model_dump()


def test_tiered_reproducible():
    a = make_tiered_graph("medium", seed=7)
    b = make_tiered_graph("medium", seed=7)
    assert a.model_dump() == b.model_dump()


# --------------------------------------------------------------------------- #
# Parameter combinations & validation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("directed", [True, False])
@pytest.mark.parametrize("weighted", [True, False])
def test_param_combos(model, directed, weighted):
    g = generate_graph(12, seed=3, directed=directed, weighted=weighted,
                       model=model, **_kw(model))
    assert g.metadata.directed is directed
    assert g.metadata.weighted is weighted
    assert g.metadata.num_nodes == 12
    assert len(g.nodes) == 12
    # Edges reference only declared nodes.
    valid = set(g.nodes)
    for e in g.edges:
        assert e.source in valid and e.target in valid
        assert e.source != e.target
        if weighted:
            assert e.weight is not None
            assert 1.0 <= e.weight <= 10.0
        else:
            assert e.weight is None


def test_weight_range_respected():
    g = generate_graph(20, seed=5, weighted=True, weight_range=(2.5, 4.0))
    assert g.edges
    assert all(2.5 <= e.weight <= 4.0 for e in g.edges)


def test_validation_errors():
    with pytest.raises(ValueError):
        generate_graph(0)
    with pytest.raises(ValueError):
        generate_graph(5, hierarchy_depth=-1)
    with pytest.raises(ValueError):
        generate_graph(5, model="hierarchical", hierarchy_depth=0)
    with pytest.raises(ValueError):
        generate_graph(5, weight_range=(9.0, 1.0))
    with pytest.raises(ValueError):
        generate_graph(5, multi_edge_types=[])
    with pytest.raises(ValueError):
        generate_graph(5, multi_edge_types=["a", "a"])
    with pytest.raises(ValueError):
        generate_graph(5, model="bogus")


# --------------------------------------------------------------------------- #
# Multi-edge correctness
# --------------------------------------------------------------------------- #
def test_multi_edge_types_assigned():
    types = ["calls", "imports", "extends"]
    g = generate_graph(18, seed=11, directed=True, model="hierarchical",
                       hierarchy_depth=3, multi_edge_types=types)
    assert g.metadata.multi_edge is True
    assert g.edges
    assert all(e.type in types for e in g.edges)


def test_multi_edge_builds_multigraph():
    g = generate_graph(15, seed=9, multi_edge_types=["a", "b"], directed=True)
    nxg = g.to_networkx()
    assert isinstance(nxg, nx.MultiDiGraph)
    # Parallel edges (same pair) are preserved by the multigraph.
    pair_counts = {}
    for e in g.edges:
        pair_counts[(e.source, e.target)] = pair_counts.get((e.source, e.target), 0) + 1
    # Total stored edges equal generated edges (no collapse).
    assert nxg.number_of_edges() == len(g.edges)


def test_non_multi_is_simple_graph():
    g = generate_graph(10, seed=1, directed=False)
    assert g.metadata.multi_edge is False
    assert isinstance(g.to_networkx(), nx.Graph)


def _max_parallel(g) -> int:
    """Largest number of edges generated for any single ordered pair."""
    pair_counts: dict[tuple[str, str], int] = {}
    for e in g.edges:
        key = (e.source, e.target)
        pair_counts[key] = pair_counts.get(key, 0) + 1
    return max(pair_counts.values()) if pair_counts else 0


def test_multi_edge_emits_parallel_edges():
    types = ["a", "b", "c"]
    g = generate_graph(20, seed=11, directed=True, model="random",
                       weighted=True, edge_prob=0.4, multi_edge_types=types)
    # At least one ordered pair carries more than one (parallel) edge.
    assert _max_parallel(g) > 1
    nxg = g.to_networkx()
    assert isinstance(nxg, nx.MultiDiGraph)
    assert nxg.number_of_edges() == len(g.edges)


@pytest.mark.parametrize("tier", ["medium", "large"])
def test_tier_multi_edge_has_parallel_edges(tier):
    g = make_tiered_graph(tier, seed=3)
    assert g.metadata.multi_edge is True
    assert _max_parallel(g) > 1, f"{tier} tier has no parallel edges"


def test_parallel_edge_weight_sum_fires_on_real_data():
    """templates.py _edge_weight / total_ownership 'sum across parallel edges'
    must actually aggregate >1 weight for a multigraph pair."""
    from grb.questions.templates import _edge_weight, _node_weight_by_type

    g = generate_graph(20, seed=11, directed=True, model="random",
                       weighted=True, edge_prob=0.4,
                       multi_edge_types=["a", "b", "c"])
    nxg = g.to_networkx()
    # Find an ordered pair with parallel edges.
    parallel_pair = None
    for u, v in nxg.edges():
        if len(nxg[u][v]) > 1:
            parallel_pair = (u, v)
            break
    assert parallel_pair is not None, "expected a parallel-edge pair"
    u, v = parallel_pair
    indiv = [
        float(d["weight"])
        for d in nxg[u][v].values()
        if d.get("weight") is not None
    ]
    assert len(indiv) > 1
    # Edge weight sums across the parallel edges.
    assert _edge_weight(nxg, u, v) == round(sum(indiv), 4)
    # total_ownership aggregation also includes all parallel edges.
    out_w = _node_weight_by_type(nxg, u, incoming=False, etype=None)
    assert out_w >= round(sum(indiv), 4) - 1e-6


# --------------------------------------------------------------------------- #
# Hierarchical structure
# --------------------------------------------------------------------------- #
def test_hierarchical_has_tree_backbone():
    g = generate_graph(20, seed=4, directed=True, model="hierarchical",
                       hierarchy_depth=3)
    # A tree backbone means at least n-1 edges exist.
    assert len(g.edges) >= len(g.nodes) - 1


# --------------------------------------------------------------------------- #
# Scale-free
# --------------------------------------------------------------------------- #
def test_scale_free_connected_core():
    g = generate_graph(30, seed=2, directed=False, model="scale_free",
                       scale_free_m=2)
    nxg = g.to_networkx()
    # Preferential attachment yields a single connected component.
    assert nx.number_connected_components(nxg) == 1


# --------------------------------------------------------------------------- #
# Round-trip via networkx
# --------------------------------------------------------------------------- #
def test_networkx_roundtrip_node_count():
    g = generate_graph(14, seed=8, directed=True, weighted=True)
    nxg = g.to_networkx()
    assert nxg.number_of_nodes() == 14
    rt = BenchGraph.from_networkx(nxg, id="rt", seed=8)
    assert set(rt.nodes) == set(g.nodes)


# --------------------------------------------------------------------------- #
# Tier calibration
# --------------------------------------------------------------------------- #
def test_tier_token_targets():
    # Each tier's mean encoded size should be in a sane band around its target.
    bands = {"small": (250, 900), "medium": (1200, 3200), "large": (12000, 30000)}
    for tier, (lo, hi) in bands.items():
        sizes = [count_tokens(describe_graph(make_tiered_graph(tier, seed=s)))
                 for s in range(3)]
        mean = sum(sizes) / len(sizes)
        assert lo <= mean <= hi, f"{tier} mean {mean} outside [{lo},{hi}]"


def test_calibration_writes_file(tmp_path, monkeypatch):
    import grb.generator as gen
    target = tmp_path / "tier_calibration.json"
    monkeypatch.setattr(gen, "_CALIBRATION_PATH", target)
    report = gen.calibrate_tiers(seeds=(0, 1), write=True)
    assert target.exists()
    data = json.loads(target.read_text())
    assert set(data["tiers"]) == {"small", "medium", "large"}
    assert report["encoding"] == "cl100k_base"
