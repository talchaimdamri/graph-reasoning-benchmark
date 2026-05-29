"""Unit tests for Phase 4: answers, prompts, storage, and the pipeline.

All LLM access is MOCKED — no real ``claude`` calls are made here.
"""

from __future__ import annotations

import pytest

from grb.answers import compare_answers, grade, parse_response
from grb.encoder import encode_graph
from grb.generator import make_tiered_graph
from grb.ground_truth import generate_questions
from grb.models import Encoding, Question, Result
from grb.pipeline import BenchmarkConfig, evaluate_single, run_benchmark
from grb.prompts import (
    VISION_UNSUPPORTED_ERROR,
    VisionUnsupported,
    build_prompt,
)
from grb.storage import Storage


# --------------------------------------------------------------------------- #
# answers: parse_response
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "text,atype,expected",
    [
        ("42", "int", 42),
        ("The answer is 42.", "int", 42),
        ("1,234", "int", 1234),
        ("3.14", "float", 3.14),
        ("yes", "bool", True),
        ("No.", "bool", False),
        ("TRUE", "bool", True),
        ("N1, N2, N3", "list", ["N1", "N2", "N3"]),
        ("[N1, N2]", "list", ["N1", "N2"]),
        ("none", "list", []),
        ("  Hello ", "string", "Hello"),
    ],
)
def test_parse_response(text, atype, expected):
    assert parse_response(text, atype) == expected


def test_parse_response_empty_returns_none():
    assert parse_response("", "int") is None
    assert parse_response("   ", "list") is None  # whitespace-only -> None
    assert parse_response("not a number", "int") is None


# --------------------------------------------------------------------------- #
# answers: compare_answers
# --------------------------------------------------------------------------- #
def test_compare_int_exact_and_rounding():
    assert compare_answers(5, 5, "int")
    assert compare_answers(5, 5.0, "int")
    assert not compare_answers(5, 6, "int")


def test_compare_float_tolerance():
    assert compare_answers(3.14, 3.141, "float")  # within abs tol
    assert not compare_answers(3.14, 9.99, "float")


def test_compare_bool_variants():
    assert compare_answers(True, True, "bool")
    assert compare_answers(parse_response("yes", "bool"), True, "bool")
    assert compare_answers(parse_response("no", "bool"), False, "bool")
    assert not compare_answers(parse_response("yes", "bool"), False, "bool")


def test_compare_list_order_insensitive():
    assert compare_answers(["N2", "N1"], ["N1", "N2"], "list")
    assert compare_answers(parse_response("N1, N2, N3", "list"), ["N3", "N2", "N1"], "list")
    assert not compare_answers(["N1"], ["N1", "N2"], "list")
    assert compare_answers([], [], "list")


def test_compare_string_case_space_insensitive():
    assert compare_answers("N1", "n1", "string")
    assert compare_answers("  Hello World ", "helloworld", "string")
    assert not compare_answers("N1", "N2", "string")


def test_grade_helper():
    parsed, correct = grade("yes", True, "bool")
    assert parsed is True and correct is True
    parsed, correct = grade("N1, N2", ["N2", "N1"], "list")
    assert correct is True


# --------------------------------------------------------------------------- #
# prompts
# --------------------------------------------------------------------------- #
def _make_encoding(fmt="edge_list", content="N0 -> N1") -> Encoding:
    return Encoding(
        graph_id="g1",
        format=fmt,
        content=content,
        token_count=10,
        tokens_per_node=1.0,
        tokens_per_edge=1.0,
    )


def _make_question(atype="int", text="How many nodes?") -> Question:
    return Question(
        id="q1",
        graph_id="g1",
        text=text,
        category="node_count",
        difficulty="trivial",
        answer_type=atype,
        ground_truth=2,
        computation="G.number_of_nodes()",
    )


def test_build_prompt_includes_header_content_question_instruction():
    prompt = build_prompt(_make_encoding(), _make_question(), "edge_list")
    assert "encoded as edge_list" in prompt
    assert "N0 -> N1" in prompt
    assert "How many nodes?" in prompt
    assert "ONLY the direct answer" in prompt
    assert "single integer" in prompt


def test_build_prompt_visual_raises():
    with pytest.raises(VisionUnsupported):
        build_prompt(_make_encoding(fmt="visual", content="/tmp/x.png"),
                     _make_question(), "visual")


# --------------------------------------------------------------------------- #
# storage
# --------------------------------------------------------------------------- #
def _make_result(run_id="r1", correct=True, qid="q1", model="opus") -> Result:
    return Result(
        result_id=f"{run_id}::g1::edge_list::{qid}::{model}",
        run_id=run_id,
        graph_id="g1",
        encoding="edge_list",
        question_id=qid,
        question_text="How many nodes?",
        ground_truth=2,
        model=model,
        model_answer=2,
        correct=correct,
        tokens_used=37,
        latency_ms=12.5,
        error=None,
    )


def test_storage_save_and_get(tmp_path):
    db = tmp_path / "t.db"
    with Storage(db) as store:
        store.create_run("r1", config={"models": ["opus"]})
        store.save_result(_make_result())
        got = store.get_result("r1", "g1", "edge_list", "q1", "opus")
        assert got is not None
        assert got.correct is True
        assert got.ground_truth == 2
        assert got.model_answer == 2
        # Missing cell -> None.
        assert store.get_result("r1", "g1", "edge_list", "q1", "haiku") is None


def test_storage_upsert_is_idempotent(tmp_path):
    db = tmp_path / "t.db"
    with Storage(db) as store:
        store.create_run("r1")
        store.save_result(_make_result(correct=False))
        store.save_result(_make_result(correct=True))  # same cell, updated
        results = store.list_results("r1")
        assert len(results) == 1
        assert results[0].correct is True


def test_storage_list_handles_list_ground_truth(tmp_path):
    db = tmp_path / "t.db"
    r = _make_result()
    r = r.model_copy(update={"ground_truth": ["N1", "N2"], "model_answer": ["N2", "N1"]})
    with Storage(db) as store:
        store.create_run("r1")
        store.save_result(r)
        got = store.get_result("r1", "g1", "edge_list", "q1", "opus")
        assert got.ground_truth == ["N1", "N2"]
        assert got.model_answer == ["N2", "N1"]


def test_storage_export_json(tmp_path):
    db = tmp_path / "t.db"
    out = tmp_path / "export.json"
    with Storage(db) as store:
        store.create_run("r1")
        store.save_result(_make_result())
        path = store.export_json(out, run_id="r1")
    assert path.exists()
    import json

    data = json.loads(path.read_text())
    assert data["count"] == 1
    assert data["results"][0]["graph_id"] == "g1"


# --------------------------------------------------------------------------- #
# pipeline (mocked call_claude)
# --------------------------------------------------------------------------- #
def _mock_call_correct(prompt, model, **kwargs):
    """A mock that 'knows' the answer is 2 (matches the question ground truth)."""
    return {"text": "2", "tokens_used": 5, "latency_ms": 1.0, "error": None}


def _mock_call_wrong(prompt, model, **kwargs):
    return {"text": "999", "tokens_used": 5, "latency_ms": 1.0, "error": None}


def _mock_call_error(prompt, model, **kwargs):
    return {"text": "", "tokens_used": 0, "latency_ms": 0.0, "error": "boom"}


def test_evaluate_single_correct():
    res = evaluate_single(
        run_id="r1",
        encoding=_make_encoding(),
        fmt="edge_list",
        question=_make_question(),
        model="opus",
        call_fn=_mock_call_correct,
    )
    assert res.correct is True
    assert res.model_answer == 2
    assert res.tokens_used == 5
    assert res.error is None


def test_evaluate_single_wrong():
    res = evaluate_single(
        run_id="r1",
        encoding=_make_encoding(),
        fmt="edge_list",
        question=_make_question(),
        model="opus",
        call_fn=_mock_call_wrong,
    )
    assert res.correct is False
    assert res.model_answer == 999


def test_evaluate_single_error():
    res = evaluate_single(
        run_id="r1",
        encoding=_make_encoding(),
        fmt="edge_list",
        question=_make_question(),
        model="opus",
        call_fn=_mock_call_error,
    )
    assert res.correct is False
    assert res.error == "boom"


def test_evaluate_single_visual_skipped():
    res = evaluate_single(
        run_id="r1",
        encoding=_make_encoding(fmt="visual", content="/tmp/x.png"),
        fmt="visual",
        question=_make_question(),
        model="opus",
        call_fn=_mock_call_correct,  # must NOT be invoked
    )
    assert res.error == VISION_UNSUPPORTED_ERROR
    assert res.correct is False


def _tiny_config(tmp_path, call_count_box=None):
    graph = make_tiered_graph("small", seed=0)
    enc = encode_graph(graph, formats=["edge_list"])
    qs = generate_questions(graph, n=2, seed=0)[:2]
    return BenchmarkConfig(
        graphs=[graph],
        encodings={graph.id: enc},
        questions={graph.id: qs},
        models=("opus", "sonnet", "haiku"),
        db_path=str(tmp_path / "bench.db"),
        run_id="run1",
        max_workers=2,
        formats=["edge_list"],
        show_progress=False,
    )


def test_run_benchmark_grid_size_and_caching(tmp_path):
    calls = {"n": 0}

    def mock(prompt, model, **kwargs):
        calls["n"] += 1
        return {"text": "2", "tokens_used": 3, "latency_ms": 1.0, "error": None}

    config = _tiny_config(tmp_path)
    summary = run_benchmark(config, call_fn=mock)
    # 1 graph x 1 encoding x 2 questions x 3 models = 6 cells.
    assert summary["total"] == 6
    assert calls["n"] == 6
    assert summary["quota"]["calls"] == 6
    assert summary["quota"]["tokens"] == 18

    # Rerun with the same run_id -> everything cached, no new calls.
    calls["n"] = 0
    summary2 = run_benchmark(config, call_fn=mock)
    assert calls["n"] == 0
    assert summary2["quota"]["cached"] == 6
    assert summary2["total"] == 6


def test_run_benchmark_visual_skipped(tmp_path):
    graph = make_tiered_graph("small", seed=0)
    # Add a fake visual encoding (content = a path); pipeline must skip it.
    enc = encode_graph(graph, formats=["edge_list"])
    enc["visual"] = Encoding(
        graph_id=graph.id,
        format="visual",
        content="/tmp/fake.png",
        token_count=0,
        tokens_per_node=0.0,
        tokens_per_edge=0.0,
    )
    qs = generate_questions(graph, n=1, seed=0)[:1]
    config = BenchmarkConfig(
        graphs=[graph],
        encodings={graph.id: enc},
        questions={graph.id: qs},
        models=("opus",),
        db_path=str(tmp_path / "b.db"),
        run_id="rv",
        formats=None,
        show_progress=False,
    )

    def mock(prompt, model, **kwargs):
        return {"text": "2", "tokens_used": 1, "latency_ms": 1.0, "error": None}

    summary = run_benchmark(config, call_fn=mock)
    # 2 formats x 1 question x 1 model = 2 cells; visual one is skipped.
    assert summary["total"] == 2
    assert summary["quota"]["skipped"] == 1
    visual_results = [r for r in summary["results"] if r.encoding == "visual"]
    assert visual_results and visual_results[0].error == VISION_UNSUPPORTED_ERROR
