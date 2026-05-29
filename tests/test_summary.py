"""Validate that papers/SUMMARY.md is a well-formed synthesis of all 12 papers.

The SUMMARY.md is the deliverable for the literature-review task. These tests
guard its structure so the synthesis stays complete and navigable.
"""

from pathlib import Path

import pytest

SUMMARY = Path(__file__).resolve().parents[1] / "papers" / "SUMMARY.md"

# Distinctive token per paper (case-insensitive substring match).
PAPER_MARKERS = [
    "GraphARC",
    "GraphInstruct",
    "KG-LLM-Bench",
    "GraphArena",
    "Rethinking and Benchmarking",
    "GraphToken",
    "MermaidSeqBench",
    "How Do LLMs Understand Graph Patterns",
    "When Structure Doesn't Help",
    "NLGift",
    "EstGraph",
    "GraphAgent-Reasoner",
]

REQUIRED_SECTIONS = [
    "## 1. Paper-by-Paper Summaries",
    "## 2. Cross-Paper Synthesis: Which Encoding Formats Win / Lose",
    "## 3. Cross-Paper Synthesis: Which Question Types Are Hard for LLMs",
    "## 4. Cross-Paper Synthesis: Graph-Size Effects",
    "## 5. The Gap Our Benchmark Fills",
]

GAP_PILLARS = [
    "Claude-only",
    "Head-to-head encodings",
    "direct-reading",
    "Token-efficiency-vs-accuracy".lower(),
]


@pytest.fixture(scope="module")
def text() -> str:
    assert SUMMARY.exists(), f"missing deliverable: {SUMMARY}"
    return SUMMARY.read_text(encoding="utf-8")


def test_all_twelve_papers_present(text: str) -> None:
    low = text.lower()
    missing = [m for m in PAPER_MARKERS if m.lower() not in low]
    assert not missing, f"papers not covered: {missing}"


def test_twelve_numbered_paper_subsections(text: str) -> None:
    # Subsections are numbered 1.1 .. 1.12 under section 1.
    subsections = [f"### 1.{i} " for i in range(1, 13)]
    missing = [s for s in subsections if s not in text]
    assert not missing, f"missing numbered subsections: {missing}"


def test_required_top_level_sections(text: str) -> None:
    missing = [s for s in REQUIRED_SECTIONS if s not in text]
    assert not missing, f"missing sections: {missing}"


def test_gap_section_covers_four_pillars(text: str) -> None:
    gap = text.split("## 5. The Gap Our Benchmark Fills", 1)[1].lower()
    missing = [p for p in GAP_PILLARS if p.lower() not in gap]
    assert not missing, f"gap section missing pillars: {missing}"


def test_synthesis_mentions_concrete_encodings(text: str) -> None:
    low = text.lower()
    for enc in ["edge list", "adjacency", "json", "yaml", "rdf turtle", "json-ld"]:
        assert enc in low, f"encoding not discussed: {enc}"
