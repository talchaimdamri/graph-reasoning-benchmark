"""Deterministic question generation with NetworkX-computed ground truth.

``generate_questions`` draws a balanced mix of trivial and non-trivial
templates, binds each to concrete parameters with a seeded RNG, executes the
ground-truth computation via NetworkX, validates that the computed answer
matches the declared ``answer_type``, and emits :class:`grb.models.Question`
objects.

Nothing here is AI-generated: every ``ground_truth`` value is the result of a
NetworkX call performed at generation time, recorded alongside a short
``computation`` string documenting how it was produced.
"""

from __future__ import annotations

import random
from typing import Any

from grb.models import AnswerType, BenchGraph, Question
from grb.questions.templates import (
    ALL_TEMPLATES,
    NONTRIVIAL_TEMPLATES,
    TRIVIAL_TEMPLATES,
    Template,
    TemplateInstance,
)


def validate_answer_type(value: Any, answer_type: AnswerType) -> bool:
    """Return True iff ``value`` is consistent with ``answer_type``.

    Used as a sanity check after computing a ground truth. ``bool`` is checked
    before ``int`` because ``bool`` is a subclass of ``int`` in Python.
    """
    if answer_type == "bool":
        return isinstance(value, bool)
    if answer_type == "int":
        return isinstance(value, int) and not isinstance(value, bool)
    if answer_type == "float":
        # Accept ints too: a whole-number weight is a valid float answer.
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if answer_type == "list":
        return isinstance(value, list)
    if answer_type == "string":
        return isinstance(value, str)
    return False


def _instantiate(
    template: Template, graph: BenchGraph, rng: random.Random
) -> Question | None:
    """Try to build one Question from a template, validating its ground truth.

    Returns ``None`` when the template cannot bind to this graph (e.g. no
    weighted edges) — the caller should then try another template.
    """
    inst: TemplateInstance | None = template.make(graph, rng)
    if inst is None:
        return None

    if not validate_answer_type(inst.ground_truth, template.answer_type):
        raise ValueError(
            f"Template {template.name!r} produced ground_truth "
            f"{inst.ground_truth!r} which is not of declared answer_type "
            f"{template.answer_type!r} for graph {graph.id!r}."
        )

    qid = f"{graph.id}__{template.name}__{rng.getrandbits(32):08x}"
    return Question(
        id=qid,
        graph_id=graph.id,
        text=inst.text,
        category=template.category,
        difficulty=template.difficulty,
        answer_type=template.answer_type,
        ground_truth=inst.ground_truth,
        computation=inst.computation,
    )


def generate_questions(
    graph: BenchGraph,
    n: int = 15,
    *,
    seed: int = 0,
    balance: float = 0.5,
) -> list[Question]:
    """Generate ``n`` deterministic questions for ``graph``.

    Args:
        graph: the benchmark graph to ask about.
        n: target number of questions (15-20 is the intended range).
        seed: RNG seed; the same (graph, n, seed) always yields identical output.
        balance: target fraction of *trivial* questions in ``[0, 1]``.

    Strategy: aim for ``round(n * balance)`` trivial and the rest non-trivial.
    For each slot we shuffle the applicable templates and take the first that
    successfully binds, retrying with fresh parameters if needed. Templates may
    repeat across slots (with different parameters) when the pool is small, but
    we prefer unused templates first to maximise category coverage.
    """
    if n <= 0:
        return []

    rng = random.Random(seed)
    n_trivial = round(n * balance)
    n_nontrivial = n - n_trivial

    questions: list[Question] = []
    questions += _fill(graph, TRIVIAL_TEMPLATES, n_trivial, rng)
    questions += _fill(graph, NONTRIVIAL_TEMPLATES, n_nontrivial, rng)

    # If a difficulty pool was too thin for this graph, top up from the other
    # pool so we still return a useful number of questions.
    if len(questions) < n:
        questions += _fill(
            graph, ALL_TEMPLATES, n - len(questions), rng, seen_texts={q.text for q in questions}
        )

    return questions[:n]


def _fill(
    graph: BenchGraph,
    pool: list[Template],
    count: int,
    rng: random.Random,
    seen_texts: set[str] | None = None,
) -> list[Question]:
    """Produce up to ``count`` questions from ``pool`` avoiding duplicate text."""
    if count <= 0:
        return []
    out: list[Question] = []
    seen: set[str] = set() if seen_texts is None else set(seen_texts)

    # Prefer using each template once before repeating, for category coverage.
    order = list(pool)
    rng.shuffle(order)
    queue = list(order)

    attempts = 0
    max_attempts = count * 40 + 50
    while len(out) < count and attempts < max_attempts:
        attempts += 1
        if not queue:
            queue = list(order)
            rng.shuffle(queue)
        template = queue.pop(0)
        q = _instantiate(template, graph, rng)
        if q is None:
            continue
        if q.text in seen:
            # Same parameterisation already used; re-queue and retry.
            queue.append(template)
            continue
        seen.add(q.text)
        out.append(q)
    return out
