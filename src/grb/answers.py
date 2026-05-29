"""Parse model responses and compare them to ground truth.

The model is asked to answer with only the direct answer, but real responses
are noisy. :func:`parse_response` extracts a usable value given the expected
``answer_type``, and :func:`compare_answers` decides correctness with type-aware
normalization:

* ``int`` / ``float`` — numeric comparison with tolerance (floats within
  ``rel/abs`` tolerance, ints exact after rounding).
* ``bool`` — yes/no/true/false/1/0 mapped to a boolean.
* ``list`` — order-insensitive set comparison of normalized items.
* ``string`` — case- and whitespace-insensitive comparison.
"""

from __future__ import annotations

import re
from typing import Any

# Tolerances for float comparison.
FLOAT_ABS_TOL = 1e-2
FLOAT_REL_TOL = 1e-3

_TRUE_WORDS = {"yes", "true", "y", "t", "1", "correct"}
_FALSE_WORDS = {"no", "false", "n", "f", "0", "incorrect"}

_NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def _strip(text: Any) -> str:
    return str(text).strip()


def _first_number(text: str) -> str | None:
    """Return the first numeric token in ``text`` (handles 1,234 thousands)."""
    cleaned = text.replace(",", "")
    m = _NUM_RE.search(cleaned)
    return m.group(0) if m else None


def _to_bool(text: str) -> bool | None:
    """Interpret free text as a boolean, or ``None`` if undecidable."""
    t = text.strip().lower()
    t = t.strip(" .!\"'`)(")
    if t in _TRUE_WORDS:
        return True
    if t in _FALSE_WORDS:
        return False
    # Look at the first word.
    first = re.split(r"[\s,.;:]+", t, maxsplit=1)[0] if t else ""
    if first in _TRUE_WORDS:
        return True
    if first in _FALSE_WORDS:
        return False
    return None


def _split_list(text: str) -> list[str]:
    """Split a free-text list answer into normalized item strings."""
    t = text.strip()
    if not t:
        return []
    low = t.lower().strip(" .!\"'`")
    if low in {"none", "[]", "empty", "no nodes", "nothing"}:
        return []
    # Remove surrounding brackets if present.
    t = t.strip()
    if t.startswith("[") and t.endswith("]"):
        t = t[1:-1]
    # Split on commas, semicolons, arrows, or whitespace runs.
    raw = re.split(r"[,;]|->|=>|\s*\|\s*", t)
    if len(raw) == 1:
        # Maybe space-separated.
        raw = re.split(r"\s+", t)
    items = []
    for r in raw:
        item = r.strip().strip("'\"`()[]")
        if item:
            items.append(item)
    return items


def _norm_token(s: str) -> str:
    """Normalize one list/string token for comparison."""
    return re.sub(r"\s+", "", str(s).strip().lower())


def parse_response(text: str, answer_type: str) -> Any:
    """Parse raw model output into a typed value for the given answer type.

    Returns ``None`` when nothing usable can be extracted.
    """
    if text is None:
        return None
    s = _strip(text)
    if not s:
        return None

    if answer_type == "bool":
        return _to_bool(s)

    if answer_type == "int":
        num = _first_number(s)
        if num is None:
            return None
        try:
            return int(round(float(num)))
        except ValueError:
            return None

    if answer_type == "float":
        num = _first_number(s)
        if num is None:
            return None
        try:
            return float(num)
        except ValueError:
            return None

    if answer_type == "list":
        return _split_list(s)

    if answer_type == "string":
        # Take the last non-empty line, stripped of trailing punctuation.
        line = s.splitlines()[-1].strip() if s.splitlines() else s
        return line.strip(" .!\"'`")

    return s


def _floats_close(a: float, b: float) -> bool:
    return abs(a - b) <= max(FLOAT_ABS_TOL, FLOAT_REL_TOL * max(abs(a), abs(b)))


def compare_answers(parsed: Any, ground_truth: Any, answer_type: str) -> bool:
    """Return True iff ``parsed`` matches ``ground_truth`` for ``answer_type``."""
    if parsed is None:
        return False

    if answer_type == "bool":
        gt = ground_truth if isinstance(ground_truth, bool) else _to_bool(str(ground_truth))
        return bool(parsed) == bool(gt)

    if answer_type == "int":
        try:
            return int(parsed) == int(round(float(ground_truth)))
        except (ValueError, TypeError):
            return False

    if answer_type == "float":
        try:
            return _floats_close(float(parsed), float(ground_truth))
        except (ValueError, TypeError):
            return False

    if answer_type == "list":
        if not isinstance(parsed, (list, tuple, set)):
            parsed = _split_list(str(parsed))
        gt_list = ground_truth if isinstance(ground_truth, (list, tuple, set)) else [ground_truth]
        return {_norm_token(x) for x in parsed} == {_norm_token(x) for x in gt_list}

    if answer_type == "string":
        return _norm_token(parsed) == _norm_token(ground_truth)

    return _norm_token(parsed) == _norm_token(ground_truth)


def grade(text: str, ground_truth: Any, answer_type: str) -> tuple[Any, bool]:
    """Convenience: parse ``text`` then compare. Returns (parsed, correct)."""
    parsed = parse_response(text, answer_type)
    return parsed, compare_answers(parsed, ground_truth, answer_type)
