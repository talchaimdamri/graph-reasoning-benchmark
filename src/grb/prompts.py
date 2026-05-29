"""Prompt construction for the benchmark.

``build_prompt(encoding, question, format)`` produces the full text prompt sent
to the model. Each prompt has:

* a per-format header explaining how the graph is encoded,
* the encoded graph content,
* the question,
* a strict instruction to answer with ONLY the direct answer.

The ``visual`` format is special: headless ``claude -p`` cannot (in this
pipeline) accept an image, so :func:`build_prompt` for ``visual`` raises
:class:`VisionUnsupported`, and the pipeline marks such results skipped with
``error='vision-unsupported-in-headless'``.
"""

from __future__ import annotations

from grb.models import Encoding, Question

VISION_UNSUPPORTED_ERROR = "vision-unsupported-in-headless"

VISUAL_FORMAT = "visual"

# Per-format human-readable descriptions used in the prompt header.
FORMAT_DESCRIPTIONS: dict[str, str] = {
    "adjacency_list": (
        "an adjacency list, where each line lists a node followed by the nodes "
        "it connects to"
    ),
    "edge_list": (
        "an edge list, where each line describes one edge between two nodes"
    ),
    "mermaid": (
        "a Mermaid flowchart diagram, where arrows denote edges between nodes"
    ),
    "dot": (
        "a Graphviz DOT description, where statements denote edges between nodes"
    ),
    "natural_language": (
        "a natural-language description of the nodes and edges"
    ),
    "matrix": (
        "an adjacency matrix, where rows and columns are nodes and a nonzero "
        "cell denotes an edge"
    ),
    "visual": "an image (rendered diagram) of the graph",
}

# Per-answer-type hint appended to the instruction so output is easy to parse.
_ANSWER_TYPE_HINTS: dict[str, str] = {
    "int": "Answer with a single integer.",
    "float": "Answer with a single number.",
    "bool": "Answer with exactly 'yes' or 'no'.",
    "list": (
        "Answer with a comma-separated list of items (order does not matter). "
        "If the answer is empty, write 'none'."
    ),
    "string": "Answer with a single label or word.",
}

_INSTRUCTION = (
    "Answer the question with ONLY the direct answer (a number, a list, or "
    "yes/no). Do not show any reasoning, explanation, or extra words."
)


class VisionUnsupported(Exception):
    """Raised when a prompt is requested for the visual format in headless mode."""

    def __init__(self, message: str = VISION_UNSUPPORTED_ERROR) -> None:
        super().__init__(message)


def _format_header(fmt: str) -> str:
    desc = FORMAT_DESCRIPTIONS.get(fmt, f"the {fmt} format")
    return f"The graph is encoded as {fmt} ({desc})."


def build_prompt(encoding: Encoding, question: Question, fmt: str) -> str:
    """Build the full prompt string for one (encoding, question) pair.

    Parameters
    ----------
    encoding:
        The :class:`Encoding` whose ``content`` is the serialized graph (for the
        ``visual`` format ``content`` is an image path, which is unsupported).
    question:
        The :class:`Question` to ask.
    fmt:
        The encoding format name (e.g. ``edge_list``); usually
        ``encoding.format``.

    Raises
    ------
    VisionUnsupported
        If ``fmt`` is the visual format (image input is not available here).
    """
    if fmt == VISUAL_FORMAT:
        raise VisionUnsupported()

    header = _format_header(fmt)
    answer_hint = _ANSWER_TYPE_HINTS.get(question.answer_type, "")

    parts = [
        header,
        "",
        "Graph:",
        encoding.content,
        "",
        f"Question: {question.text}",
        "",
        _INSTRUCTION,
    ]
    if answer_hint:
        parts.append(answer_hint)
    return "\n".join(parts)
