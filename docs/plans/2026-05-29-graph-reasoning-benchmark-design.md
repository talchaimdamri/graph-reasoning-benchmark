# Graph Reasoning Benchmark — Design Record

Date: 2026-05-29
Status: Accepted

## Purpose

Measure how well LLMs reason over graphs, and how that ability varies with the
**textual/visual encoding** of the same graph. The benchmark holds the graph and
the question fixed while varying the encoding, so accuracy differences are
attributable to representation rather than content.

## Key Decisions

### 1. Models — Claude only

We evaluate Claude models exclusively (no cross-vendor comparison). The three
tiers are `opus`, `sonnet`, and `haiku`. Inference runs through the **headless
Claude CLI**:

```
claude -p --model <opus|sonnet|haiku> "<encoded prompt>"
```

Rationale: single auth path, deterministic-enough harness, no per-vendor SDK
divergence. The model identifier stored on `Result.model` is the CLI model string.

### 2. Tiers by token budget (not raw node count)

Tiers are defined by the **token budget of the encoded graph**, then mapped to
approximate node counts. Token budget is the controlled variable because it is
what the model actually pays for and is bounded by context.

| Tier   | Token budget | Approx. nodes |
|--------|--------------|---------------|
| small  | ~500         | ~10–15        |
| medium | ~2000        | ~40–60        |
| large  | ~20000       | ~300–500      |

Node-count ranges are calibrated empirically per encoding (a `matrix` encoding
costs far more tokens per node than an `edge_list`), so the generator targets the
token budget and records the achieved `num_nodes` in `GraphMeta`.

### 3. Seven encodings

Each graph is serialized into 7 formats. The model sees exactly one per trial:

1. `adjacency_list` — node: [neighbors]
2. `edge_list` — one `u -> v` (or `u -- v`) per line
3. `mermaid` — Mermaid `graph`/`flowchart` syntax
4. `dot` — Graphviz DOT
5. `natural_language` — prose description of nodes and relationships
6. `matrix` — adjacency matrix
7. `visual` — rendered image (via Graphviz/matplotlib) passed to the model

Token cost per format is recorded on `Encoding` (`token_count`,
`tokens_per_node`, `tokens_per_edge`) using `tiktoken`.

### 4. Questions — template-based, deterministic, NetworkX ground truth

Questions are generated from **fixed templates**, not authored by an AI. Ground
truth is computed by **NetworkX** (e.g. shortest path length, degree, connectivity,
cycle existence, topological order). This makes correctness checkable
programmatically and removes LLM grading noise.

- `difficulty`: `trivial` (single-hop / direct lookup) vs `nontrivial`
  (multi-hop / global property).
- `answer_type` constrains parsing of the model answer: `int | float | bool |
  list | string`.
- `computation` records the NetworkX call used, for auditability.

### 5. Results storage — SQLite + JSON

Raw per-trial `Result` rows are written to **SQLite** (`data/results/*.db`, the
DB files are git-ignored) for queryable analysis, and mirrored to **JSON** under
`data/results/` for portability and diffing.

### 6. Reporting surface — analysis + dashboard, no paper

Deliverables are (a) an analysis layer (pandas + matplotlib figures into
`figures/`) and (b) a **Vite + React + TypeScript** dashboard rendered
**Hebrew, right-to-left (RTL)**. There is **no academic paper**; `papers/` is a
placeholder only.

## Repository Layout

```
graph-reasoning-benchmark/
├── pyproject.toml            # package "grb", console script grb = grb.cli:app
├── .gitignore
├── README.md
├── src/grb/
│   ├── __init__.py
│   ├── models.py             # SHARED DATA MODEL (Pydantic v2)
│   ├── cli.py                # Typer app
│   ├── encoders/__init__.py  # 7 encodings
│   └── questions/__init__.py # template question generation + NetworkX truth
├── tests/                    # pytest
├── data/
│   ├── graphs/               # serialized BenchGraph JSON
│   ├── questions/            # serialized Question JSON
│   ├── encodings/            # serialized Encoding JSON
│   └── results/              # SQLite (.db, git-ignored) + JSON
├── docs/plans/               # design records (this file)
├── figures/                  # analysis output
└── papers/                   # placeholder (no paper)
```

## Shared Data Model

All models are Pydantic v2 and live in `src/grb/models.py`; every module imports
from `grb.models` and must not redefine them.

- **Edge**: `{source: str, target: str, type: str|None=None, weight: float|None=None}`
- **GraphMeta**: `{directed: bool, weighted: bool, multi_edge: bool,
  hierarchy_depth: int, seed: int, tier: 'small'|'medium'|'large', num_nodes: int}`
- **BenchGraph**: `{id: str, nodes: list[str], edges: list[Edge],
  metadata: GraphMeta}`
  - `to_networkx() -> nx.(Multi)(Di)Graph` — class chosen from
    `directed`/`multi_edge` (`Graph`, `DiGraph`, `MultiGraph`, `MultiDiGraph`);
    `type`/`weight` become edge attributes.
  - `from_networkx(g, *, id, metadata=None, ...)` — inverse; infers metadata
    (directed/multi/weighted/num_nodes) from the graph when not supplied.
- **Question**: `{id, graph_id, text, category, difficulty('trivial'|'nontrivial'),
  answer_type('int'|'float'|'bool'|'list'|'string'), ground_truth: Any,
  computation: str}`
- **Encoding**: `{graph_id, format, content, token_count, tokens_per_node,
  tokens_per_edge}`
- **Result**: `{result_id, run_id, graph_id, encoding, question_id,
  question_text, ground_truth, model, model_answer, correct: bool,
  tokens_used: int, latency_ms: float, error: str|None}`

## Open / Deferred

- Empirical node-count calibration per encoding to hit token budgets.
- Answer parsing/normalization rules per `answer_type`.
- Dashboard data contract (read SQLite vs. exported JSON).
