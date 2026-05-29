"""Deterministic question templates with NetworkX-computed ground truth.

Each template describes one *kind* of question. A template is a small,
declarative object that knows:

* its ``category`` and ``difficulty``,
* the ``answer_type`` of its ground truth,
* whether it ``applies`` to a given graph (e.g. weight templates need a
  weighted graph),
* how to ``sample`` concrete parameters (deterministically, from a seeded
  ``random.Random``) — usually picking node(s) or edge(s),
* how to render the question ``text`` for chosen parameters,
* how to ``compute`` the ground truth via NetworkX on ``graph.to_networkx()``,
* a short ``computation`` code string documenting that computation.

Ground truth is *never* AI-generated: it is always the result of running a
NetworkX call on the graph. This module is the single source of truth for the
benchmark's correct answers.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Callable, Optional

import networkx as nx

from grb.models import AnswerType, BenchGraph, Difficulty

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _edge_types(g: nx.Graph) -> list[str]:
    """Distinct, sorted edge ``type`` values present in the graph."""
    types: set[str] = set()
    for *_nodes, data in g.edges(data=True):
        t = data.get("type")
        if t is not None:
            types.add(t)
    return sorted(types)


def _has_weights(g: nx.Graph) -> bool:
    for *_nodes, data in g.edges(data=True):
        if data.get("weight") is not None:
            return True
    return False


def _edge_weight(g: nx.Graph, u: str, v: str) -> float:
    """Total weight on edge ``u -> v`` (summed across parallel edges).

    Returns ``0.0`` when there is no such edge. For multigraphs the parallel
    edges are summed so the answer is well defined.
    """
    if not g.has_edge(u, v):
        return 0.0
    if g.is_multigraph():
        total = 0.0
        for _k, data in g[u][v].items():
            w = data.get("weight")
            total += float(w) if w is not None else 0.0
        return round(total, 4)
    w = g[u][v].get("weight")
    return float(w) if w is not None else 0.0


def _node_weight_by_type(
    g: nx.Graph, node: str, *, incoming: bool, etype: Optional[str]
) -> float:
    """Sum of edge weights touching ``node`` in one direction, optionally
    filtered to a single edge ``type``.

    For undirected graphs ``incoming`` is ignored and all incident edges are
    summed. Parallel edges in a multigraph are all included.
    """
    total = 0.0
    if g.is_directed():
        view = g.in_edges(node, data=True) if incoming else g.out_edges(node, data=True)
    else:
        view = g.edges(node, data=True)
    for *_n, data in view:
        if etype is not None and data.get("type") != etype:
            continue
        w = data.get("weight")
        if w is not None:
            total += float(w)
    return round(total, 4)


# ---------------------------------------------------------------------------
# Template definition
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TemplateInstance:
    """A concrete, parameter-bound question ready to be turned into a Question."""

    text: str
    ground_truth: Any
    computation: str


@dataclass(frozen=True)
class Template:
    """A reusable question template.

    ``sample`` returns a dict of parameters (or ``None`` when the graph offers
    no valid binding, e.g. asking for a child of a leaf node). ``build`` turns
    the graph + params into a :class:`TemplateInstance`.
    """

    name: str
    category: str
    difficulty: Difficulty
    answer_type: AnswerType
    applies: Callable[[nx.Graph], bool]
    sample: Callable[[nx.Graph, random.Random], Optional[dict]]
    build: Callable[[nx.Graph, dict], TemplateInstance]

    def make(
        self, bench: BenchGraph, rng: random.Random
    ) -> Optional[TemplateInstance]:
        g = bench.to_networkx()
        if not self.applies(g):
            return None
        params = self.sample(g, rng)
        if params is None:
            return None
        return self.build(g, params)


# Always-applicable predicate.
def _any(_g: nx.Graph) -> bool:
    return True


def _is_directed(g: nx.Graph) -> bool:
    return g.is_directed()


# ===========================================================================
# Trivial templates
# ===========================================================================


def _t_node_count() -> Template:
    return Template(
        name="node_count",
        category="node_count",
        difficulty="trivial",
        answer_type="int",
        applies=_any,
        sample=lambda g, r: {},
        build=lambda g, p: TemplateInstance(
            text="How many nodes does the graph have?",
            ground_truth=g.number_of_nodes(),
            computation="G.number_of_nodes()",
        ),
    )


def _t_list_nodes() -> Template:
    def build(g, p):
        nodes = sorted(g.nodes())
        return TemplateInstance(
            text="List all node labels in the graph.",
            ground_truth=nodes,
            computation="sorted(G.nodes())",
        )

    return Template(
        name="list_nodes",
        category="list_nodes",
        difficulty="trivial",
        answer_type="list",
        applies=_any,
        sample=lambda g, r: {},
        build=build,
    )


def _t_node_exists() -> Template:
    def sample(g, r):
        nodes = sorted(g.nodes())
        if not nodes:
            return None
        # Half the time ask about a real node, half about a fake one.
        if r.random() < 0.5:
            return {"node": r.choice(nodes), "real": True}
        # Build a label that does not exist.
        fake = f"X{r.randint(1000, 9999)}"
        while fake in g:
            fake = f"X{r.randint(1000, 9999)}"
        return {"node": fake, "real": False}

    def build(g, p):
        node = p["node"]
        return TemplateInstance(
            text=f"Is there a node labelled {node!r} in the graph?",
            ground_truth=g.has_node(node),
            computation=f"G.has_node({node!r})",
        )

    return Template(
        name="node_exists",
        category="node_exists",
        difficulty="trivial",
        answer_type="bool",
        applies=_any,
        sample=sample,
        build=build,
    )


def _t_edge_count() -> Template:
    return Template(
        name="edge_count",
        category="edge_count",
        difficulty="trivial",
        answer_type="int",
        applies=_any,
        sample=lambda g, r: {},
        build=lambda g, p: TemplateInstance(
            text="How many edges does the graph have?",
            ground_truth=g.number_of_edges(),
            computation="G.number_of_edges()",
        ),
    )


def _t_list_children() -> Template:
    """Successors of a node (directed) / neighbours (undirected)."""

    def sample(g, r):
        nodes = [n for n in g.nodes()]
        if not nodes:
            return None
        return {"node": r.choice(sorted(nodes))}

    def build(g, p):
        node = p["node"]
        if g.is_directed():
            children = sorted(set(g.successors(node)))
            comp = f"sorted(set(G.successors({node!r})))"
            text = f"List the direct children (successors) of node {node!r}."
        else:
            children = sorted(set(g.neighbors(node)))
            comp = f"sorted(set(G.neighbors({node!r})))"
            text = f"List the neighbours of node {node!r}."
        return TemplateInstance(text=text, ground_truth=children, computation=comp)

    return Template(
        name="list_children",
        category="list_children",
        difficulty="trivial",
        answer_type="list",
        applies=_any,
        sample=sample,
        build=build,
    )


def _t_edge_exists() -> Template:
    def sample(g, r):
        nodes = sorted(g.nodes())
        if len(nodes) < 2:
            return None
        if r.random() < 0.5 and g.number_of_edges() > 0:
            edges = list(g.edges())
            u, v = r.choice(edges)[:2]
            return {"u": str(u), "v": str(v)}
        u, v = r.sample(nodes, 2)
        return {"u": u, "v": v}

    def build(g, p):
        u, v = p["u"], p["v"]
        arrow = "->" if g.is_directed() else "--"
        return TemplateInstance(
            text=f"Is there an edge {u} {arrow} {v}?",
            ground_truth=g.has_edge(u, v),
            computation=f"G.has_edge({u!r}, {v!r})",
        )

    return Template(
        name="edge_exists",
        category="edge_exists",
        difficulty="trivial",
        answer_type="bool",
        applies=_any,
        sample=sample,
        build=build,
    )


def _t_edge_weight() -> Template:
    def sample(g, r):
        weighted_edges = [
            (u, v) for u, v, d in g.edges(data=True) if d.get("weight") is not None
        ]
        if not weighted_edges:
            return None
        u, v = r.choice(weighted_edges)
        return {"u": str(u), "v": str(v)}

    def build(g, p):
        u, v = p["u"], p["v"]
        arrow = "->" if g.is_directed() else "--"
        note = (
            " (sum across parallel edges)" if g.is_multigraph() else ""
        )
        return TemplateInstance(
            text=f"What is the weight of edge {u} {arrow} {v}?{note}",
            ground_truth=_edge_weight(g, u, v),
            computation=f"sum of weight on edge ({u!r}, {v!r})",
        )

    return Template(
        name="edge_weight",
        category="edge_weight",
        difficulty="trivial",
        answer_type="float",
        applies=_has_weights,
        sample=sample,
        build=build,
    )


def _t_in_degree() -> Template:
    def sample(g, r):
        return {"node": r.choice(sorted(g.nodes()))} if g.nodes() else None

    def build(g, p):
        node = p["node"]
        return TemplateInstance(
            text=f"What is the in-degree of node {node!r}?",
            ground_truth=g.in_degree(node),
            computation=f"G.in_degree({node!r})",
        )

    return Template(
        name="in_degree",
        category="in_degree",
        difficulty="trivial",
        answer_type="int",
        applies=_is_directed,
        sample=sample,
        build=build,
    )


def _t_out_degree() -> Template:
    def sample(g, r):
        return {"node": r.choice(sorted(g.nodes()))} if g.nodes() else None

    def build(g, p):
        node = p["node"]
        return TemplateInstance(
            text=f"What is the out-degree of node {node!r}?",
            ground_truth=g.out_degree(node),
            computation=f"G.out_degree({node!r})",
        )

    return Template(
        name="out_degree",
        category="out_degree",
        difficulty="trivial",
        answer_type="int",
        applies=_is_directed,
        sample=sample,
        build=build,
    )


def _t_max_degree_node() -> Template:
    def build(g, p):
        # Deterministic tie-break: highest degree, then smallest label.
        best = min(g.nodes(), key=lambda n: (-g.degree(n), str(n)))
        return TemplateInstance(
            text=(
                "Which node has the highest total degree? "
                "(break ties by smallest label)"
            ),
            ground_truth=str(best),
            computation="argmax_n G.degree(n), tie-break smallest label",
        )

    return Template(
        name="max_degree_node",
        category="max_degree_node",
        difficulty="trivial",
        answer_type="string",
        applies=lambda g: g.number_of_nodes() > 0,
        sample=lambda g, r: {},
        build=build,
    )


# ===========================================================================
# Non-trivial templates
# ===========================================================================


def _t_shortest_path() -> Template:
    def sample(g, r):
        # Choose a reachable (u, v) pair if possible.
        nodes = sorted(g.nodes())
        if len(nodes) < 2:
            return None
        for _ in range(20):
            u, v = r.sample(nodes, 2)
            if nx.has_path(g, u, v):
                return {"u": u, "v": v}
        return None

    def build(g, p):
        u, v = p["u"], p["v"]
        path = nx.shortest_path(g, u, v)
        return TemplateInstance(
            text=f"What is a shortest path (by hop count) from {u!r} to {v!r}? "
            "Give the node sequence.",
            ground_truth=[str(n) for n in path],
            computation=f"nx.shortest_path(G, {u!r}, {v!r})",
        )

    return Template(
        name="shortest_path",
        category="shortest_path",
        difficulty="nontrivial",
        answer_type="list",
        applies=_any,
        sample=sample,
        build=build,
    )


def _t_shortest_path_len() -> Template:
    def sample(g, r):
        nodes = sorted(g.nodes())
        if len(nodes) < 2:
            return None
        for _ in range(20):
            u, v = r.sample(nodes, 2)
            if nx.has_path(g, u, v):
                return {"u": u, "v": v}
        return None

    def build(g, p):
        u, v = p["u"], p["v"]
        length = nx.shortest_path_length(g, u, v)
        return TemplateInstance(
            text=f"What is the length (number of hops) of the shortest path "
            f"from {u!r} to {v!r}?",
            ground_truth=length,
            computation=f"nx.shortest_path_length(G, {u!r}, {v!r})",
        )

    return Template(
        name="shortest_path_len",
        category="shortest_path_len",
        difficulty="nontrivial",
        answer_type="int",
        applies=_any,
        sample=sample,
        build=build,
    )


def _t_num_paths() -> Template:
    """Number of simple paths between two nodes (capped search)."""

    def sample(g, r):
        nodes = sorted(g.nodes())
        if len(nodes) < 2:
            return None
        for _ in range(20):
            u, v = r.sample(nodes, 2)
            if nx.has_path(g, u, v):
                return {"u": u, "v": v}
        return None

    def build(g, p):
        u, v = p["u"], p["v"]
        # cutoff keeps enumeration tractable on dense graphs.
        count = sum(1 for _ in nx.all_simple_paths(g, u, v, cutoff=6))
        return TemplateInstance(
            text=f"How many distinct simple paths of length <= 6 hops connect "
            f"{u!r} to {v!r}?",
            ground_truth=count,
            computation=f"len(list(nx.all_simple_paths(G, {u!r}, {v!r}, cutoff=6)))",
        )

    return Template(
        name="num_paths",
        category="num_paths",
        difficulty="nontrivial",
        answer_type="int",
        applies=_any,
        sample=sample,
        build=build,
    )


def _t_reachable_count() -> Template:
    def sample(g, r):
        return {"node": r.choice(sorted(g.nodes()))} if g.nodes() else None

    def build(g, p):
        node = p["node"]
        reachable = nx.descendants(g, node) if g.is_directed() else (
            set(nx.node_connected_component(g, node)) - {node}
        )
        comp = (
            f"len(nx.descendants(G, {node!r}))"
            if g.is_directed()
            else f"len(nx.node_connected_component(G, {node!r})) - 1"
        )
        return TemplateInstance(
            text=f"How many other nodes are reachable from {node!r}?",
            ground_truth=len(reachable),
            computation=comp,
        )

    return Template(
        name="reachable_count",
        category="reachable_count",
        difficulty="nontrivial",
        answer_type="int",
        applies=_any,
        sample=sample,
        build=build,
    )


def _t_can_reach() -> Template:
    def sample(g, r):
        nodes = sorted(g.nodes())
        if len(nodes) < 2:
            return None
        u, v = r.sample(nodes, 2)
        return {"u": u, "v": v}

    def build(g, p):
        u, v = p["u"], p["v"]
        return TemplateInstance(
            text=f"Can node {v!r} be reached from node {u!r}?",
            ground_truth=nx.has_path(g, u, v),
            computation=f"nx.has_path(G, {u!r}, {v!r})",
        )

    return Template(
        name="can_reach",
        category="can_reach",
        difficulty="nontrivial",
        answer_type="bool",
        applies=_any,
        sample=sample,
        build=build,
    )


def _t_max_distance() -> Template:
    """Eccentricity of a node: greatest shortest-path distance to a reachable
    node."""

    def sample(g, r):
        return {"node": r.choice(sorted(g.nodes()))} if g.nodes() else None

    def build(g, p):
        node = p["node"]
        lengths = nx.single_source_shortest_path_length(g, node)
        max_d = max(lengths.values()) if lengths else 0
        return TemplateInstance(
            text=f"What is the greatest shortest-path distance (in hops) from "
            f"{node!r} to any node it can reach?",
            ground_truth=max_d,
            computation=f"max(nx.single_source_shortest_path_length(G, {node!r}).values())",
        )

    return Template(
        name="max_distance",
        category="max_distance",
        difficulty="nontrivial",
        answer_type="int",
        applies=lambda g: g.number_of_nodes() > 0,
        sample=sample,
        build=build,
    )


def _t_leaf_nodes() -> Template:
    """Leaves: out-degree 0 (directed) / degree 1 (undirected)."""

    def build(g, p):
        if g.is_directed():
            leaves = sorted(n for n in g.nodes() if g.out_degree(n) == 0)
            comp = "sorted(n for n in G if G.out_degree(n) == 0)"
            text = "List all leaf nodes (out-degree 0)."
        else:
            leaves = sorted(n for n in g.nodes() if g.degree(n) == 1)
            comp = "sorted(n for n in G if G.degree(n) == 1)"
            text = "List all leaf nodes (degree 1)."
        return TemplateInstance(
            text=text, ground_truth=[str(n) for n in leaves], computation=comp
        )

    return Template(
        name="leaf_nodes",
        category="leaf_nodes",
        difficulty="nontrivial",
        answer_type="list",
        applies=_any,
        sample=lambda g, r: {},
        build=build,
    )


def _t_root_nodes() -> Template:
    """Roots: in-degree 0 nodes (directed only)."""

    def build(g, p):
        roots = sorted(n for n in g.nodes() if g.in_degree(n) == 0)
        return TemplateInstance(
            text="List all root nodes (in-degree 0).",
            ground_truth=[str(n) for n in roots],
            computation="sorted(n for n in G if G.in_degree(n) == 0)",
        )

    return Template(
        name="root_nodes",
        category="root_nodes",
        difficulty="nontrivial",
        answer_type="list",
        applies=_is_directed,
        sample=lambda g, r: {},
        build=build,
    )


def _t_num_components() -> Template:
    def build(g, p):
        if g.is_directed():
            n = nx.number_weakly_connected_components(g)
            comp = "nx.number_weakly_connected_components(G)"
            text = "How many weakly connected components does the graph have?"
        else:
            n = nx.number_connected_components(g)
            comp = "nx.number_connected_components(G)"
            text = "How many connected components does the graph have?"
        return TemplateInstance(text=text, ground_truth=n, computation=comp)

    return Template(
        name="num_components",
        category="num_components",
        difficulty="nontrivial",
        answer_type="int",
        applies=_any,
        sample=lambda g, r: {},
        build=build,
    )


def _t_is_cyclic() -> Template:
    def build(g, p):
        if g.is_directed():
            has_cycle = not nx.is_directed_acyclic_graph(g)
            comp = "not nx.is_directed_acyclic_graph(G)"
        else:
            try:
                nx.find_cycle(g)
                has_cycle = True
            except nx.NetworkXNoCycle:
                has_cycle = False
            comp = "try nx.find_cycle(G) -> True else False"
        return TemplateInstance(
            text="Does the graph contain a cycle?",
            ground_truth=has_cycle,
            computation=comp,
        )

    return Template(
        name="is_cyclic",
        category="is_cyclic",
        difficulty="nontrivial",
        answer_type="bool",
        applies=_any,
        sample=lambda g, r: {},
        build=build,
    )


# ===========================================================================
# Weighted / multi-edge aggregation templates
# ===========================================================================


def _t_total_ownership() -> Template:
    """Sum of incoming or outgoing edge weights of a node, by edge type.

    Models e.g. "how much of company X is owned" (incoming ``owns`` weight) in
    an ownership graph, generalised to any edge type / direction.
    """

    def applies(g):
        return _has_weights(g)

    def sample(g, r):
        nodes = sorted(g.nodes())
        if not nodes:
            return None
        node = r.choice(nodes)
        incoming = g.is_directed() and r.random() < 0.5
        types = _edge_types(g)
        etype = r.choice(types) if types and r.random() < 0.5 else None
        return {"node": node, "incoming": incoming, "etype": etype}

    def build(g, p):
        node, incoming, etype = p["node"], p["incoming"], p["etype"]
        value = _node_weight_by_type(g, node, incoming=incoming, etype=etype)
        if g.is_directed():
            direction = "incoming" if incoming else "outgoing"
        else:
            direction = "incident"
        type_clause = f" of type {etype!r}" if etype else ""
        return TemplateInstance(
            text=f"What is the total weight of {direction} edges{type_clause} "
            f"at node {node!r}?",
            ground_truth=value,
            computation=(
                f"sum weight of {direction} edges"
                + (f" with type=={etype!r}" if etype else "")
                + f" at {node!r}"
            ),
        )

    return Template(
        name="total_ownership",
        category="total_ownership",
        difficulty="nontrivial",
        answer_type="float",
        applies=applies,
        sample=sample,
        build=build,
    )


def _t_avg_weight() -> Template:
    def build(g, p):
        weights = [
            float(d["weight"])
            for *_n, d in g.edges(data=True)
            if d.get("weight") is not None
        ]
        avg = round(sum(weights) / len(weights), 4) if weights else 0.0
        return TemplateInstance(
            text="What is the average weight across all weighted edges? "
            "(rounded to 4 decimals)",
            ground_truth=avg,
            computation="mean(d['weight'] for *_ , d in G.edges(data=True))",
        )

    return Template(
        name="avg_weight",
        category="avg_weight",
        difficulty="nontrivial",
        answer_type="float",
        applies=_has_weights,
        sample=lambda g, r: {},
        build=build,
    )


def _t_controls_most() -> Template:
    """Which node has the greatest total outgoing (or incident) edge weight.

    Interprets "controls" as outgoing weight in a directed graph (e.g. an
    ownership / influence graph), or total incident weight when undirected.
    """

    def build(g, p):
        def node_out_weight(n):
            if g.is_directed():
                view = g.out_edges(n, data=True)
            else:
                view = g.edges(n, data=True)
            return sum(
                float(d["weight"])
                for *_x, d in view
                if d.get("weight") is not None
            )

        best = min(g.nodes(), key=lambda n: (-node_out_weight(n), str(n)))
        direction = "outgoing" if g.is_directed() else "incident"
        return TemplateInstance(
            text=f"Which node has the greatest total {direction} edge weight? "
            "(break ties by smallest label)",
            ground_truth=str(best),
            computation=f"argmax_n sum of {direction} edge weights at n",
        )

    return Template(
        name="controls_most",
        category="controls_most",
        difficulty="nontrivial",
        answer_type="string",
        applies=lambda g: _has_weights(g) and g.number_of_nodes() > 0,
        sample=lambda g, r: {},
        build=build,
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

TRIVIAL_TEMPLATES: list[Template] = [
    _t_node_count(),
    _t_list_nodes(),
    _t_node_exists(),
    _t_edge_count(),
    _t_list_children(),
    _t_edge_exists(),
    _t_edge_weight(),
    _t_in_degree(),
    _t_out_degree(),
    _t_max_degree_node(),
]

NONTRIVIAL_TEMPLATES: list[Template] = [
    _t_shortest_path(),
    _t_shortest_path_len(),
    _t_num_paths(),
    _t_reachable_count(),
    _t_can_reach(),
    _t_max_distance(),
    _t_leaf_nodes(),
    _t_root_nodes(),
    _t_num_components(),
    _t_is_cyclic(),
    _t_total_ownership(),
    _t_avg_weight(),
    _t_controls_most(),
]

ALL_TEMPLATES: list[Template] = TRIVIAL_TEMPLATES + NONTRIVIAL_TEMPLATES

TEMPLATES_BY_NAME: dict[str, Template] = {t.name: t for t in ALL_TEMPLATES}
