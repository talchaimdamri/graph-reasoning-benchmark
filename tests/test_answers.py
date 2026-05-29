"""Tests for graph-aware shortest-path grading and multiset list comparison."""

from __future__ import annotations

import networkx as nx

from grb.answers import compare_answers, is_valid_shortest_path


def _diamond() -> nx.DiGraph:
    """A -> B -> D and A -> C -> D: two distinct shortest paths of length 2."""
    g = nx.DiGraph()
    g.add_edge("A", "B")
    g.add_edge("B", "D")
    g.add_edge("A", "C")
    g.add_edge("C", "D")
    return g


# --------------------------------------------------------------------------- #
# is_valid_shortest_path
# --------------------------------------------------------------------------- #
def test_exact_path_accepted():
    g = _diamond()
    assert is_valid_shortest_path(["A", "B", "D"], "A", "D", g, 2)


def test_valid_alternative_shortest_path_accepted():
    g = _diamond()
    # NetworkX might return A,B,D; the alternative A,C,D is equally shortest.
    assert is_valid_shortest_path(["A", "C", "D"], "A", "D", g, 2)


def test_scrambled_non_path_rejected():
    g = _diamond()
    # Right node set, but B->C is not an edge -> not a valid path.
    assert not is_valid_shortest_path(["A", "C", "B", "D"], "A", "D", g, 2)


def test_wrong_length_path_rejected():
    g = _diamond()
    # A longer walk that is a real path but not shortest.
    g.add_edge("B", "C")
    assert nx.has_path(g, "A", "D")
    assert not is_valid_shortest_path(["A", "B", "C", "D"], "A", "D", g, 2)


def test_wrong_endpoints_rejected():
    g = _diamond()
    assert not is_valid_shortest_path(["B", "D"], "A", "D", g, 2)


def test_string_answer_parsed():
    g = _diamond()
    assert is_valid_shortest_path("A -> C -> D", "A", "D", g, 2)


def test_unknown_node_rejected():
    g = _diamond()
    assert not is_valid_shortest_path(["A", "Z", "D"], "A", "D", g, 2)


# --------------------------------------------------------------------------- #
# multiset list comparison
# --------------------------------------------------------------------------- #
def test_multiset_rejects_duplicate_mismatch():
    # Order-insensitive but duplicate-count-aware: ['N1','N1'] != ['N1'].
    assert not compare_answers(["N1", "N1"], ["N1"], "list")


def test_multiset_order_insensitive_still_passes():
    assert compare_answers(["N2", "N1"], ["N1", "N2"], "list")


def test_multiset_matching_duplicates_pass():
    assert compare_answers(["N1", "N1", "N2"], ["N2", "N1", "N1"], "list")
