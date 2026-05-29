"""Shared data model for the Graph Reasoning Benchmark (grb).

Every module imports from ``grb.models``. All models are Pydantic v2.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

import networkx as nx
from pydantic import BaseModel, Field

Tier = Literal["small", "medium", "large"]
Difficulty = Literal["trivial", "nontrivial"]
AnswerType = Literal["int", "float", "bool", "list", "string"]


class Edge(BaseModel):
    """A single edge. ``type`` and ``weight`` are optional attributes."""

    source: str
    target: str
    type: Optional[str] = None
    weight: Optional[float] = None


class GraphMeta(BaseModel):
    """Structural metadata describing how a graph was generated."""

    directed: bool
    weighted: bool
    multi_edge: bool
    hierarchy_depth: int
    seed: int
    tier: Tier
    num_nodes: int


class BenchGraph(BaseModel):
    """A benchmark graph: nodes, edges and structural metadata."""

    id: str
    nodes: list[str]
    edges: list[Edge]
    metadata: GraphMeta

    def _nx_class(self) -> type:
        """Pick the right NetworkX class from the metadata."""
        directed = self.metadata.directed
        multi = self.metadata.multi_edge
        if directed and multi:
            return nx.MultiDiGraph
        if directed and not multi:
            return nx.DiGraph
        if not directed and multi:
            return nx.MultiGraph
        return nx.Graph

    def to_networkx(self) -> nx.Graph:
        """Build a NetworkX graph using the class implied by the metadata.

        Edge ``type`` and ``weight`` are stored as edge attributes (only
        when present). All declared nodes are added even if isolated.
        """
        g = self._nx_class()()
        g.add_nodes_from(self.nodes)
        for e in self.edges:
            attrs: dict[str, Any] = {}
            if e.type is not None:
                attrs["type"] = e.type
            if e.weight is not None:
                attrs["weight"] = e.weight
            g.add_edge(e.source, e.target, **attrs)
        return g

    @classmethod
    def from_networkx(
        cls,
        g: nx.Graph,
        *,
        id: str,
        metadata: GraphMeta | None = None,
        seed: int = 0,
        tier: Tier = "small",
        hierarchy_depth: int = 0,
    ) -> "BenchGraph":
        """Construct a :class:`BenchGraph` from a NetworkX graph.

        If ``metadata`` is not supplied it is inferred from the graph class
        and the presence of edge weights, using the keyword fallbacks for the
        fields NetworkX does not track (``seed``, ``tier``, ``hierarchy_depth``).
        """
        directed = g.is_directed()
        multi = g.is_multigraph()
        nodes = [str(n) for n in g.nodes()]

        edges: list[Edge] = []
        weighted = False
        if multi:
            edge_iter = g.edges(keys=False, data=True)
        else:
            edge_iter = g.edges(data=True)
        for u, v, data in edge_iter:
            weight = data.get("weight")
            if weight is not None:
                weighted = True
            edges.append(
                Edge(
                    source=str(u),
                    target=str(v),
                    type=data.get("type"),
                    weight=weight,
                )
            )

        if metadata is None:
            metadata = GraphMeta(
                directed=directed,
                weighted=weighted,
                multi_edge=multi,
                hierarchy_depth=hierarchy_depth,
                seed=seed,
                tier=tier,
                num_nodes=len(nodes),
            )

        return cls(id=id, nodes=nodes, edges=edges, metadata=metadata)


class Question(BaseModel):
    """A deterministic, template-generated question about one graph."""

    id: str
    graph_id: str
    text: str
    category: str
    difficulty: Difficulty
    answer_type: AnswerType
    ground_truth: Any
    computation: str = Field(
        description="Human-readable description of how ground_truth was computed."
    )


class Encoding(BaseModel):
    """One serialized representation of a graph in a given format."""

    graph_id: str
    format: str
    content: str
    token_count: int
    tokens_per_node: float
    tokens_per_edge: float


class Result(BaseModel):
    """A single (graph, encoding, question, model) evaluation outcome."""

    result_id: str
    run_id: str
    graph_id: str
    encoding: str
    question_id: str
    question_text: str
    ground_truth: Any
    model: str
    model_answer: Any
    correct: bool
    tokens_used: int
    latency_ms: float
    error: Optional[str] = None
