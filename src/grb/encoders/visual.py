"""Visual encoder: render the graph to an image and return the file path.

Unlike the text encoders, this returns a filesystem path (string) to a rendered
PNG produced by :mod:`grb.visualizer`. The path is what gets stored in the
:class:`~grb.models.Encoding` ``content`` field; ``token_count`` is set to 0 by
the top-level :func:`grb.encoder.encode_graph`.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Optional

from grb.models import BenchGraph
from grb.visualizer import render


def encode(
    graph: BenchGraph,
    out_dir: Optional[str | Path] = None,
    fmt: str = "png",
) -> str:
    """Render ``graph`` to an image and return the absolute file path as a string.

    Parameters
    ----------
    out_dir:
        Directory to write into. Defaults to a temp directory keyed off the
        system temp root so repeated calls for the same graph id overwrite.
    fmt:
        ``"png"`` or ``"svg"``.
    """
    if out_dir is None:
        out_dir = Path(tempfile.gettempdir()) / "grb_visual"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Sanitize id for use as a filename stem.
    stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in graph.id)
    out_stem = out_dir / stem
    paths = render(graph, out_stem, fmt=fmt)
    return str(paths[0].resolve())
