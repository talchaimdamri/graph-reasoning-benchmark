"""Top-level multi-format graph encoder.

:func:`encode_graph` serializes a :class:`~grb.models.BenchGraph` into one or
more of the benchmark's encoding formats and wraps each result in an
:class:`~grb.models.Encoding` with ``cl100k_base`` token statistics.

Text formats: ``adjacency_list``, ``edge_list``, ``mermaid``, ``dot``,
``natural_language``, ``matrix``. The ``visual`` format renders an image via
:mod:`grb.visualizer`; its ``content`` is the image path and ``token_count`` is 0.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

from grb.encoders import (
    adjacency_list,
    dot,
    edge_list,
    matrix,
    mermaid,
    natural_language,
    visual,
)
from grb.models import BenchGraph, Encoding

# Registry of text encoders: format name -> callable(graph) -> str.
TEXT_ENCODERS = {
    "adjacency_list": adjacency_list.encode,
    "edge_list": edge_list.encode,
    "mermaid": mermaid.encode,
    "dot": dot.encode,
    "natural_language": natural_language.encode,
    "matrix": matrix.encode,
}

# Formats that support structural round-trip parsing.
PARSEABLE = {
    "adjacency_list": adjacency_list.parse,
    "edge_list": edge_list.parse,
    "matrix": matrix.parse,
}

VISUAL_FORMAT = "visual"

ALL_FORMATS = list(TEXT_ENCODERS.keys()) + [VISUAL_FORMAT]


@lru_cache(maxsize=1)
def _encoder():
    import tiktoken

    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Token count using the ``cl100k_base`` encoding."""
    return len(_encoder().encode(text))


def _make_encoding(graph: BenchGraph, fmt: str, content: str, tokens: int) -> Encoding:
    n_nodes = max(1, len(graph.nodes))
    n_edges = max(1, len(graph.edges))
    return Encoding(
        graph_id=graph.id,
        format=fmt,
        content=content,
        token_count=tokens,
        tokens_per_node=round(tokens / n_nodes, 4),
        tokens_per_edge=round(tokens / n_edges, 4),
    )


def encode_graph(
    graph: BenchGraph,
    formats: Optional[list[str]] = None,
    *,
    visual_out_dir: Optional[str | Path] = None,
) -> dict[str, Encoding]:
    """Encode ``graph`` into the requested formats.

    Parameters
    ----------
    graph:
        The graph to encode.
    formats:
        Subset of :data:`ALL_FORMATS`. Defaults to all text formats (the
        ``visual`` format is excluded by default because it writes files).
    visual_out_dir:
        Directory for the ``visual`` format's rendered image, if requested.

    Returns
    -------
    dict[str, Encoding]
        Mapping of format name to its :class:`Encoding`.
    """
    if formats is None:
        formats = list(TEXT_ENCODERS.keys())

    out: dict[str, Encoding] = {}
    for fmt in formats:
        if fmt in TEXT_ENCODERS:
            content = TEXT_ENCODERS[fmt](graph)
            tokens = count_tokens(content)
            out[fmt] = _make_encoding(graph, fmt, content, tokens)
        elif fmt == VISUAL_FORMAT:
            path = visual.encode(graph, out_dir=visual_out_dir)
            out[fmt] = _make_encoding(graph, fmt, path, 0)
        else:
            raise ValueError(f"unknown format: {fmt!r}")
    return out
