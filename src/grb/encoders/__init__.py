"""Graph encoders: serialize a BenchGraph into the benchmark formats.

Each text-format module exposes ``encode(graph) -> str``; the structurally
parseable ones (``adjacency_list``, ``edge_list``, ``matrix``) also expose
``parse(content) -> BenchGraph``. The ``visual`` module exposes
``encode(graph, ...) -> str`` returning an image file path.
"""

from grb.encoders import (  # noqa: F401
    adjacency_list,
    dot,
    edge_list,
    matrix,
    mermaid,
    natural_language,
    visual,
)

__all__ = [
    "adjacency_list",
    "dot",
    "edge_list",
    "matrix",
    "mermaid",
    "natural_language",
    "visual",
]
