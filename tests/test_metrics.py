"""Tests for grb.metrics over a small synthetic results frame."""

from __future__ import annotations

import json
import math

import pandas as pd
import pytest

from grb import metrics
from grb.fixtures import make_full_fixture
from grb.models import Question, Result


def _result(**kw) -> Result:
    base = dict(
        result_id="r",
        run_id="run1",
        graph_id="g1",
        encoding="edge_list",
        question_id="q1",
        question_text="?",
        ground_truth=1,
        model="m1",
        model_answer=1,
        correct=True,
        tokens_used=100,
        latency_ms=10.0,
        error=None,
    )
    base.update(kw)
    return Result(**base)


@pytest.fixture
def small_results() -> list[Result]:
    return [
        _result(result_id="a", encoding="edge_list", model="m1", correct=True, tokens_used=80),
        _result(result_id="b", encoding="edge_list", model="m1", correct=False, tokens_used=90,
                 error="answer-parse-failed"),
        _result(result_id="c", encoding="matrix", model="m1", correct=True, tokens_used=200),
        _result(result_id="d", encoding="matrix", model="m2", correct=False, tokens_used=210),
        _result(result_id="e", encoding="edge_list", model="m2", correct=True, tokens_used=85),
    ]


@pytest.fixture
def small_questions() -> list[Question]:
    return [
        Question(id="q1", graph_id="g1", text="?", category="node_count",
                 difficulty="trivial", answer_type="int", ground_truth=1, computation="n"),
    ]


def test_results_to_frame_empty():
    frame = metrics.results_to_frame([])
    assert frame.empty
    assert "category" in frame.columns and "tier" in frame.columns


def test_results_to_frame_basic(small_results, small_questions):
    frame = metrics.results_to_frame(small_results, questions=small_questions)
    assert len(frame) == 5
    assert frame["correct"].dtype == bool
    # category enrichment from question join.
    assert (frame["category"] == "node_count").all()
    # difficulty present.
    assert (frame["difficulty"] == "trivial").all()


def test_accuracy_by_encoding(small_results):
    frame = metrics.results_to_frame(small_results)
    acc = metrics.accuracy_by_encoding(frame)
    row = acc.set_index("encoding").loc["edge_list"]
    # 2/3 correct on edge_list.
    assert math.isclose(row["accuracy"], 2 / 3, abs_tol=1e-4)
    assert row["n"] == 3


def test_accuracy_by_model(small_results):
    frame = metrics.results_to_frame(small_results)
    acc = metrics.accuracy_by_model(frame).set_index("model")
    assert math.isclose(acc.loc["m1", "accuracy"], 2 / 3, abs_tol=1e-4)
    assert math.isclose(acc.loc["m2", "accuracy"], 0.5, abs_tol=1e-4)


def test_token_efficiency(small_results):
    frame = metrics.results_to_frame(small_results)
    te = metrics.token_efficiency(frame).set_index("encoding")
    assert "edge_list" in te.index
    assert te.loc["edge_list", "accuracy_per_1k"] > 0


def test_accuracy_vs_tokens(small_results):
    frame = metrics.results_to_frame(small_results)
    avt = metrics.accuracy_vs_tokens(frame)
    assert {"model", "encoding", "mean_tokens", "accuracy"}.issubset(avt.columns)
    # (m1,edge_list), (m1,matrix), (m2,edge_list), (m2,matrix)
    assert len(avt) == 4


def test_error_breakdown(small_results):
    frame = metrics.results_to_frame(small_results)
    eb = metrics.error_breakdown(frame).set_index("error")
    assert eb.loc["ok", "count"] == 4
    assert eb.loc["answer-parse-failed", "count"] == 1


def test_model_x_format(small_results):
    frame = metrics.results_to_frame(small_results)
    mxf = metrics.model_x_format(frame)
    assert "edge_list" in mxf.columns
    assert "m1" in mxf.index


def test_overall_summary(small_results):
    frame = metrics.results_to_frame(small_results)
    summ = metrics.overall_summary(frame)
    assert summ["total_results"] == 5
    assert math.isclose(summ["overall_accuracy"], 3 / 5, abs_tol=1e-4)
    assert summ["n_models"] == 2
    assert summ["n_encodings"] == 2


def test_compute_metrics_json_serializable(small_results, small_questions):
    frame = metrics.results_to_frame(small_results, questions=small_questions)
    tables = metrics.compute_metrics(frame)
    payload = tables.to_json_dict()
    # round-trips through JSON.
    json.dumps(payload)
    assert "model_x_format" in payload
    assert "models" in payload["model_x_format"]


def test_build_dashboard_payload_is_json(small_results, small_questions):
    payload = metrics.build_dashboard_payload(small_results, questions=small_questions)
    text = json.dumps(payload)
    assert '"summary"' in text
    assert '"results"' in text
    assert '"metrics"' in text


def test_export_dashboard_json_writes_file(tmp_path, small_results, small_questions):
    out = tmp_path / "results.json"
    metrics.export_dashboard_json(out, small_results, questions=small_questions)
    data = json.loads(out.read_text())
    assert data["summary"]["total_results"] == 5
    assert isinstance(data["results"], list)


def test_generate_figures(tmp_path, small_results):
    frame = metrics.results_to_frame(small_results)
    figs = metrics.generate_figures(frame, tmp_path)
    assert len(figs) >= 4
    for p in figs:
        assert p.exists() and p.stat().st_size > 0


def test_generate_figures_empty(tmp_path):
    figs = metrics.generate_figures(metrics.results_to_frame([]), tmp_path)
    assert figs == []


def test_full_fixture_end_to_end(tmp_path):
    fx = make_full_fixture()
    assert len(fx["graphs"]) == 3
    assert len(fx["questions"]) == 12  # 4 questions x 3 graphs
    assert len(fx["results"]) > 0
    payload = metrics.build_dashboard_payload(
        fx["results"], questions=fx["questions"], graphs=fx["graphs"]
    )
    json.dumps(payload, default=str)
    # graphs carry nodes/edges for the cytoscape view.
    assert payload["graphs"][0]["nodes"]
    assert "edges" in payload["graphs"][0]
    assert payload["summary"]["n_graphs"] == 3


def test_graph_payload_shape():
    fx = make_full_fixture()
    p = metrics.graph_to_payload(fx["graphs"][0])
    assert set(p) == {"id", "metadata", "nodes", "edges"}
    if p["edges"]:
        assert {"source", "target"}.issubset(p["edges"][0])
