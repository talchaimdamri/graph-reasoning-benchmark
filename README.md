# Graph Reasoning Benchmark

A benchmark for evaluating how well Large Language Models understand and reason about graphs across different encoding formats.

## Overview

This benchmark focuses on **direct graph reading** - not complex reasoning or pattern memorization. The goal is to systematically compare how different graph encoding formats affect LLM performance on simple, fact-based queries.

### What Makes This Different

Most existing benchmarks focus on complex graph algorithms (shortest path, cycle detection, etc.). This benchmark tests something more fundamental: **Can the model accurately read and understand the graph structure itself?**

We test:
- ✅ Simple queries that can be answered by direct graph traversal
- ✅ Multiple encoding formats (adjacency list, Mermaid, DOT, natural language, visual, etc.)
- ✅ Synthetic graphs with controlled parameters
- ✅ Ground truth computed via graph algorithms
- ✅ Token efficiency + accuracy trade-offs

## Research Context

### Existing Work

This benchmark builds on recent research in graph reasoning for LLMs:

**General Benchmarks:**
- [GraphARC](https://openreview.net/pdf?id=CULN1wh2tw) - Comprehensive benchmark for graph-based abstract reasoning
- [GraphInstruct](https://arxiv.org/html/2605.09997v1) - Progressive benchmark for LLM graph generation
- [KG-LLM-Bench](https://arxiv.org/pdf/2504.07087) - Evaluating LLM reasoning on textualized knowledge graphs
- [GraphArena](https://arxiv.org/html/2407.00379v1) - Benchmarking on graph computational problems

**Encoding Strategies:**
- ["Let Your Graph Do the Talking"](https://arxiv.org/pdf/2402.05862) - Analysis of graph-to-text encoding strategies
- [MermaidSeqBench](https://arxiv.org/pdf/2511.14967) - Evaluation of LLM-to-Mermaid generation

**Graph Pattern Understanding:**
- ["How Do Large Language Models Understand Graph Patterns?"](https://arxiv.org/pdf/2410.05298) - Benchmark for graph pattern comprehension
- ["When Structure Doesn't Help"](https://arxiv.org/pdf/2511.16767) - Challenges in text-attributed graph reading

### Research Gap

While existing benchmarks test complex reasoning, few systematically compare:
1. **Multiple encoding formats head-to-head** on the same graphs
2. **Simple "flash" queries** that require only direct reading (no reasoning)
3. **Token efficiency vs accuracy** trade-offs per format
4. **Visual graph understanding** alongside textual encodings

## Benchmark Design

### Phase 1: Graph Generation

Generate synthetic graphs with configurable parameters:

**Core Parameters:**
- **Size**: Number of nodes (5-50 range)
- **Hierarchy depth**: Levels in the graph (flat vs deep)
- **Directionality**: Directed vs undirected
- **Edge weights**: Weighted edges (e.g., company ownership percentages)
- **Multi-edges**: Multiple edge types between nodes (e.g., control, management, ownership)

**Example Use Cases:**
- Company ownership structures (weighted, directed, multi-edge)
- Social networks (undirected, unweighted)
- Hierarchical organizations (directed, hierarchical)

**Output:**
- Graph data structure (NetworkX or similar)
- Visualization (automatic rendering)

### Phase 2: Multi-Format Encoding

Translate each graph into multiple formats:

1. **Adjacency List** (JSON)
   ```json
   {
     "A": ["B", "C"],
     "B": ["D"],
     "C": ["D"]
   }
   ```

2. **Edge List** (JSON)
   ```json
   [
     {"from": "A", "to": "B", "weight": 0.5},
     {"from": "A", "to": "C", "weight": 0.5}
   ]
   ```

3. **Mermaid Syntax**
   ```
   graph TD
     A --> B
     A --> C
     B --> D
     C --> D
   ```

4. **DOT (Graphviz)**
   ```
   digraph G {
     A -> B;
     A -> C;
     B -> D;
     C -> D;
   }
   ```

5. **Natural Language**
   ```
   Node A connects to nodes B and C.
   Node B connects to node D.
   Node C connects to node D.
   ```

6. **Matrix Representation**
   ```
   Adjacency Matrix:
      A B C D
   A [0 1 1 0]
   B [0 0 0 1]
   C [0 0 0 1]
   D [0 0 0 0]
   ```

7. **Visual (Image)**
   - Rendered graph image (PNG/SVG)
   - Tests vision-language models

### Phase 3: Question Generation

Automatically generate questions from graphs using an AI agent + graph algorithms.

**Question Types:**

**Trivial Queries:**
- "How many nodes are in the graph?"
- "List all children of node B"
- "What is the in-degree of node D?"
- "Does an edge exist from A to C?"

**Non-Trivial Queries:**
- "What is the shortest path from A to D?"
- "How many nodes are reachable from A?"
- "What is the total weight along path A→B→D?"
- "Which nodes have no outgoing edges?"

**Agent-Generated Questions:**
- AI agent analyzes the graph structure
- Identifies interesting patterns/features
- Generates 10-20 diverse questions
- Writes graph query code (e.g., NetworkX) to compute ground truth

**Ground Truth:**
- All answers computed via graph algorithms (NetworkX, igraph, etc.)
- 100% accuracy guaranteed
- No manual labeling needed

### Phase 4: Benchmark Execution

For each (graph, encoding, question) tuple:
1. Encode graph in specific format
2. Present to LLM with question
3. Collect LLM answer
4. Compare to ground truth
5. Measure: accuracy, token count, latency

**Metrics:**
- **Accuracy**: % correct answers
- **Token Efficiency**: Tokens used per encoding
- **Latency**: Time to answer
- **Error Types**: Categorize failure modes

**Models to Test:**

Claude only — three tiers: **Opus**, **Sonnet**, **Haiku**. There is no
cross-vendor comparison. Inference runs through the local **headless Claude
CLI** (`claude -p --model <opus|sonnet|haiku> "<encoded prompt>"`), so **no API
keys are required** — the harness reuses your existing `claude` login.

## Roadmap

- [x] **Phase 0**: Package scaffold + shared Pydantic data model
- [x] **Phase 1**: Graph generation + tier calibration + visualization
- [x] **Phase 2**: Multi-format encoding (7 formats)
- [x] **Phase 3**: Template-based question generation + NetworkX ground truth
- [x] **Phase 4**: Benchmark execution framework (headless Claude CLI)
- [x] **Phase 5**: Metrics analysis + Hebrew RTL dashboard

## Installation

```bash
python -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/pytest
```

This installs the `grb` package (and the `grb` console script) plus the dev
extras, then runs the test suite to confirm the install.

## Quickstart

The benchmark is a pipeline of `grb` subcommands. A typical end-to-end flow:

```bash
grb generate           # synthesize tiered graphs (small/medium/large) -> data/graphs/
grb encode             # serialize each graph into the 7 formats -> data/encodings/
grb questions          # generate templated questions + NetworkX ground truth -> data/questions/
grb estimate           # project grid size, total calls and tokens (no API calls)
grb run --yes          # execute the grid via headless Claude, write results to SQLite
grb export             # mirror SQLite results to portable JSON under data/results/
```

`grb run` always prints the cost estimate first; pass `--estimate` to stop there
without making any calls, and `--yes` to actually execute. Restrict the grid
with `--models opus,sonnet,haiku` and `--formats edge_list,mermaid` as needed.

> ⚠️ **`grb run` consumes real Claude quota.** Cost scales with the full grid:
> **graphs × encodings × questions × models**. For reference, one observed run of
> **48 cells used ~2.0M tokens**. Always run `grb run --estimate` first to see the
> projection before committing real quota.

## Disclosure: committed results are SYNTHETIC demo fixtures

The checked-in `figures/` PNGs and `dashboard/public/results.json` are **not real
model output**. They are **synthetic demo fixtures** produced by `grb.fixtures`
with a fixed `seed=7`, so the dashboard and analysis layer render meaningful data
without spending any quota. Treat their accuracy/token numbers as illustrative
only.

To regenerate the artifacts from **real** benchmark results:

```bash
grb run --yes                                   # produces a real SQLite results DB
.venv/bin/python scripts/build_dashboard_data.py --db data/results/<run>.db
```

Running `scripts/build_dashboard_data.py` with **no** `--db` argument
regenerates the synthetic fixtures (seed=7) instead.

## Contributing

This is an open research project. Contributions welcome!

## License

MIT

---

**Related Projects:**
- [GraphARC](https://openreview.net/pdf?id=CULN1wh2tw)
- [Let Your Graph Do the Talking](https://arxiv.org/pdf/2402.05862)
- [MermaidSeqBench](https://arxiv.org/pdf/2511.14967)
