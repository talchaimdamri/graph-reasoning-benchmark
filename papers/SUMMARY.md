# Literature Synthesis: LLM Graph Reasoning Across Encodings

This document synthesizes 12 papers on how large language models read, reason over,
and generate graphs. It distills (1) per-paper findings, (2) which **encoding formats**
tend to win or lose, (3) which **question types** are hard, (4) **graph-size effects**,
and (5) the explicit **gap** that our benchmark fills: a Claude-only, head-to-head
encoding comparison on simple direct-reading queries, scored on the
token-efficiency-vs-accuracy frontier.

---

## 1. Paper-by-Paper Summaries

### 1.1 GraphARC (Peltonen, Ronberg, Plesner, Wattenhofer; ETH Zurich; NeurIPS 2025 Workshop)
A benchmark for graph-based *abstract reasoning* (ARC-style transformations), comparing two
text encodings -- **adjacency list** (global, edge-centric) and **incident list** (local,
node-centric per-node neighborhoods). Its central result is a **comprehension-execution gap**:
models answer questions about graph *properties* far better than they can *generate* a fully
transformed graph (Qwen3-1.7B: 54.4% question accuracy vs 8.3% full-output, a 6.55x gap; even
o4-mini 91.2% vs 86.5%). There is also an **input-output asymmetry** (models reason better
about the visible input than the inferred output) and a striking **transformation bias** in
which stronger reasoning models over-apply an implied operation, answering as if the input were
already transformed (GPT-5 55-85% of the time). Encoding effect is near-neutral on average
(ratio ~ 1.0) but swings up to +/-32% for specific model-task pairs (e.g. Qwen3-4B). System-prompt
role framing had minimal, inconsistent effect. Hardest tasks: removeDegree3, colorEquidistant,
colorDegree3; easiest: colorDegree1, addHub, colorNeighbors.

### 1.2 GraphInstruct (Wei, Xiang, Zhang, Jiang; arXiv 2605.09997)
A *generation* (not reading) benchmark: LLMs must **synthesize** graphs satisfying instructions
across 6 progressive complexity levels (800 hand-authored instructions, 1,582 reference
solutions). Headline finding: discriminative power peaks at **multi-constraint composition**,
not reasoning depth -- composing several simultaneous constraints separates models more than
deeper algorithmic reasoning. **No single prompting strategy dominates** across levels/families.
Domain-semantic constraints are **iteration-invariant** (more compute does not fix them; the
authors argue retrieval is the next frontier). Lower-confidence notes from full-text:
performance drops sharply above Level 3, and JSON output is more reliable than free-form
adjacency matrices. Inverted relative to our scope (generation, multi-LLM) but its format set
and "composition is the discriminative axis" lesson transfer.

### 1.3 KG-LLM-Bench (Markowitz, Galiya, Ver Steeg, Galstyan; USC/UCR; arXiv:2504.07087)
The closest precedent to our work: a scalable benchmark evaluating LLM reasoning over
**textualized knowledge graphs** across five encodings -- **List-of-Edges, Structured JSON,
Structured YAML, RDF Turtle, JSON-LD** -- on five tasks (TripleRetrieval, ShortestPath,
AggByRelation, AggNeighborProperty, HighestDegree). Textualization choice changes overall
accuracy by **up to 17.5% absolute**, and **no single format wins** for all model/task pairs.
Average ranking: Structured JSON (0.42) best, then YAML and List-of-Edges, with RDF Turtle
(0.35) and JSON-LD (0.34) worst -- and these verbose semantic-web formats are **3-5x more
tokens** (JSON-LD 13,503 vs List-of-Edges 2,644 mean tokens), so part of their loss is pure
verbosity. **Best format is model-specific**: notably Claude-3.5-Sonnet's best was RDF Turtle,
and Claude led HighestDegree (61.5% vs 16% next). Structure helps aggregation; List-of-Edges
wins HighestDegree (the hub literally appears most often). Aggregation collapses beyond degree
~4 (~10%). Outgoing-edge aggregation is far easier than incoming. Pseudonymization had ~0.2%
effect -> models answer from the in-context graph, not memory.

### 1.4 GraphArena (ICLR 2025; arXiv:2407.00379)
Benchmarks LLMs on **graph computational problems** -- 4 polynomial-time (Common Neighbor,
Shortest Distance, Connected Component, Diameter) and 6 NP-complete (MCP, MIS, MVC, MCS, GED,
TSP) -- using a single **edge-list** encoding with real-world named nodes. Its key methodological
contribution is a **four-way output taxonomy: CORRECT / SUBOPTIMAL / HALLUCINATORY (well-formed
but infeasible) / MISSING**, which exposes failure modes plain accuracy hides. Even the best
model (Claude-3.5-Sonnet) drops from ~82.2% on small polynomial tasks to ~58.7% on large;
NP-complete tasks are far worse. **Hallucination** (using non-existent edges/nodes) is a primary
large-graph failure. CoT helps only marginally and can *degrade* with more examples; SFT helps
on small graphs only; **code-writing is the most effective anti-hallucination remedy**. Does not
compare encodings -- leaving that axis open.

### 1.5 Rethinking and Benchmarking LLMs for Graph Reasoning (Hu et al.; RUC / Ant Group; arXiv:2509.24260)
Argues base models are **underestimated**: the problem is "improper reasoning focus," not
capability. When redirected from **simulating** an algorithm step-by-step to **designing** one
(then coding it), accuracy jumps. Language-based simulation fails on **repetitive iterative and
backtracking operations** -- error compounds with graph size/density, and long graph encodings
cause "lost-in-the-middle" interference. Code-augmented (NetworkX/API) and fine-tuned methods
overfit and collapse out-of-distribution (GraphWiz scored 0.0% on their new GraphAlgorithm
suite). Their **Simple-RTC** (Reasoning-Then-Coding) decouples reasoning from data -- the
reasoning step never sees the graph, so it is size-invariant -- lifting GPT-4o-mini by 39-62%.
Reasoning-model strength scales results (DeepSeek-R1 > o3-mini-high > 4o-mini). Default encoding
is edge list; GML/GraphML/visual/soft-token alternatives are cited, not systematically compared.

### 1.6 GraphToken -- "Let Your Graph Do the Talking" (Perozzi et al.; Google; ICML 2024)
Not a text-encoding study: proposes a **learned GNN encoder** producing continuous "soft" graph
tokens prepended to a frozen LLM, yielding gains of **up to 73 percentage points** over text
prompting on the GraphQA 9-task suite (node/edge/cycle/triangle counts, degree, connected nodes,
reachability, edge existence, shortest path). Confirms the failure mode we target: **frozen LLMs
read text-encoded graphs poorly** on algorithmic tasks, and CoT/CoT-BAG give only small lifts.
"No free lunch" across GNN encoders (GCN/MPNN/GIN/MHA/HGT) -- best is task-dependent. Notably,
**breaking permutation-equivariance with explicit node identifiers (IDX) helps** more than
spectral (Laplacian) features -- hinting that stable, explicit node IDs aid LLM reading. The soft-
token method itself is out of scope for a closed Claude API, but its task taxonomy is reusable.

### 1.7 MermaidSeqBench (Shbita, Ahmed, DeLuca; IBM Research; arXiv:2511.14967)
A **generation** benchmark (NL -> Mermaid sequence diagrams) on a single encoding, scored by
LLM-judges across 6 dimensions: Syntax, Mermaid-Only, Logic, Completeness, Activation Handling,
Error/Status Tracking (132 pairs). Findings: **syntax saturates** (~88-90% even mid-size) while
**stateful/structural dimensions (activation handling, error-branch tracking) stay hard and are
most size-sensitive** -- small models produce syntactically plausible but logically wrong
diagrams. **Judge disagreement is large** (DeepSeek-V3 lenient, GPT-OSS strict), a strong caution
against LLM-as-judge and in favor of deterministic ground truth. Leaves PlantUML/other syntaxes
and reading/comprehension untested.

### 1.8 How Do LLMs Understand Graph Patterns? (Dai et al.; ICLR 2025)
A strong design template. Tests **adjacency list vs edge list** plus **terminology-based vs
topology-based** descriptions across 11 pattern tasks (translation, modification, detection,
isomorphic mapping, k-core, frequent/discriminative subgraphs) and 9 motifs (triangle, square,
diamond, house, FFL, etc.). Key results: **encoding winner flips by task** -- edge list helps
edge-comparison/isomorphism, adjacency list helps degree-based (k-core) tasks. **Terminology
(named patterns) beats topology** because it taps pretrained knowledge -> a confound that inflates
scores if not separated from actual structure reading. O1-mini's long CoT makes it
size-*robust* for one-by-one search but it fails discriminative learning. **Models miscount
degrees**: in isomorphic mapping, Claude-3-opus used direct edge comparison (~77% of cases ->
96% accuracy) while O1-mini used degree-counting (89% -> only 30%). Difficulty grows with motif
complexity ("house" is hardest) and graph size.

### 1.9 When Structure Doesn't Help (Xu, You, Ma; LoG 2025 Extended Abstract; arXiv:2511.16767)
On text-attributed graphs (TAGs), **structure-agnostic templates beat structure-aware ones**:
node classification ND (k-hop + Laplacian) 69.48% vs HN-1 (1-hop random subset) 74.49% vs
Center-Only 75.29%. A plain **MLP adapter beats GNNs** (no message passing); deeper GNNs degrade
via over-smoothing. Semantic embedding quality dominates topology even on molecules. Core thesis:
**LLMs treat graphs as unordered *sets* of node descriptions and exploit node semantics far more
than explicit topology.** This is the strongest argument that to measure *structure reading* we
must **anonymize node labels** (so connectivity can't be inferred from text), **vary
node/edge ordering**, and **favor genuine multi-hop tasks** where node names cannot leak the
answer. This is an adapter/embedding paper, not a serialization-format study -- complementary
to our axis.

### 1.10 NLGift -- Generalize beyond Pattern Memorization? (EMNLP Findings 2024; arXiv:2406.15992)
Tests whether tuned graph reasoning **generalizes** across 5 distribution-shift "patterns":
SEMANTIC (encoding format: adjacency/incident/expert/friendship framing), NUMERIC (weight
distribution: small-int/large-int/float), STRUCTURAL (size 3-10 vs 11-25, ER vs Barabasi-Albert,
transitivity), REASONING (cross-task transfer), and REAL-WORLD. Tasks: connectivity, shortest
path, topological sort, max flow (~37k problems). Findings: easier shifts meet a basic transfer
bar ~75% of the time but **Strong Recovery only ~35%** (transfer is shallow); cross-task
reasoning *never* meets Strong Recovery; and **synthetic graph instruction-tuning is actively
counterproductive on real downstream tasks in 69% of cases**. Apparent competence is largely
**memorized surface patterns / keyword frequency**. Graph size is the most damaging structural
shift. (LLaMA2-era models -- re-measure for Opus-class.)

### 1.11 EstGraph -- Large-Scale Graph Property Estimation via Random Walks
The opposite of direct reading: for graphs **above ~50 nodes you cannot serialize structure**, so
it substitutes **aggregated random-walk statistics** (simple RW, Metropolis-Hastings, max-degree;
up to 559x prompt-length reduction) for the graph itself. Tasks: size estimation, community
count, structure classification, influential-node ranking. Findings: **walk statistics beat raw
walk sequences** (errors cut 9-78%); more walk budget = lower error; accuracy degrades sharply
with size (node-count relative error ~12% small -> 33-51% on million-node graphs, with
order-of-magnitude hallucinations). **Task difficulty is uneven and encoding-driven**: PageRank
P@20 ~80% (it aligns with walk visitation) while betweenness/closeness are ~15-35%.
Reasoning model o3 led; **claude-sonnet-4 underperformed** o3 on these estimation tasks.
Establishes the **~50-node ceiling** that justifies why direct-reading work is bounded to small
graphs.

### 1.12 GraphAgent-Reasoner / GAR (Hu et al.; AAAI 2026; arXiv:2410.05130)
A **multi-agent scaffold**: a Master LLM spawns one agent per node, retrieves a distributed
algorithm, and runs message-passing rounds -- scaling by adding agents, not lengthening one
prompt. Its **Figure 2 memory probe** is directly replicable and damning: single-LLM 1-hop
neighbor recall drops from ~90% at 10 nodes to **near 0% at 100 nodes** for Claude-3.5-Sonnet,
GPT-4-turbo, and Gemini-1.5-Pro -- if a model can't recall basic structure, it can't reason.
Edge/adjacency lists grow **quadratically**, flooding context and causing attention misallocation;
LLMs process linearized graph text "as paragraphs/keywords, not structure." GAR reaches 98% avg
on GraphInstruct (vs GPT-4 2-shot 44%) and stays flat to 1000 nodes while single LLMs collapse to
0/20. Out of scope as a method (multi-agent), but sets the **with-scaffolding upper bound** and
motivates measuring the unscaffolded gap.

---

## 2. Cross-Paper Synthesis: Which Encoding Formats Win / Lose

**Headline: there is no globally best encoding -- the winner flips by model and by task -- but
verbosity reliably loses, and structure helps aggregation.**

- **Token-bloated semantic-web formats lose, partly for the wrong reason.** In KG-LLM-Bench,
  RDF Turtle (0.35) and JSON-LD (0.34) ranked worst *and* cost 3-5x the tokens of List-of-Edges
  (13,503 vs 2,644). Much of their disadvantage is verbosity, not semantics -- so any honest
  comparison must control for token count.
- **Structured formats (JSON / YAML) win aggregation; flat edge lists win hub-finding.**
  KG-LLM-Bench: Structured JSON/YAML beat List-of-Edges on AggByRelation / AggNeighborProperty
  because related edges are grouped, while List-of-Edges wins HighestDegree (the hub literally
  recurs most in a flat list). Dai et al. echo this: adjacency list helps **degree-based**
  (k-core) tasks; edge list helps **edge-comparison / isomorphism**.
- **Average encoding effect is small but conditional swings are large.** GraphARC found a
  near-neutral average (ratio ~ 1.0) but +/-32% swings for specific model-task pairs. NLGift's
  SEMANTIC pattern (adjacency / incident / expert / friendship) showed only *shallow* transfer
  across framings. Conclusion: encoding effects are **real but conditional**, so report
  **per-encoding x per-task variance**, never a single aggregate.
- **Model-specific winners matter for a Claude study.** KG-LLM-Bench: Claude-3.5-Sonnet's best
  format was **RDF Turtle** (against a global-best of JSON), and it led HighestDegree. Do **not**
  assume JSON is best for Claude; sweep formats and test newer Claude models.
- **Explicit, stable node identifiers help.** GraphToken's IDX result (breaking equivariance with
  explicit IDs beats spectral features) and the general "LLMs treat graphs as unordered node
  sets" finding (Xu et al.) suggest stable named IDs aid reading -- but those same names can
  **leak connectivity**, so anonymize when measuring pure structure.
- **Visual/soft-token encodings** (GITA images, GraphToken) help only for tiny graphs or require
  training -- out of scope for a frozen-API text benchmark.

## 3. Cross-Paper Synthesis: Which Question Types Are Hard for LLMs

Roughly easiest -> hardest, as a consistent ordering across papers:

1. **Local / single-fact lookups are easy.** Edge existence / TripleRetrieval (KG-LLM-Bench
   0.9+), node/edge counts, degree, Center-Only node classification, basic syntax parsing
   (MermaidSeqBench saturates ~90%).
2. **Generation/comprehension split.** GraphARC's comprehension-execution gap: answering
   *about* a graph is far easier than *producing* a transformed graph.
3. **Counting / aggregation breaks with scale.** KG-LLM-Bench aggregation collapses beyond
   degree ~4 (~10%); Dai et al. show degree-counting is unreliable; HighestDegree is
   "surprisingly hard" (<20% for most models). Counting is a core LLM weakness.
4. **Direction/locality asymmetry.** Outgoing-edge aggregation >> incoming (adjacency in the
   serialization); local tasks are weak discriminators (Center-Only often suffices).
5. **Multi-hop / global / stateful tasks are hard.** ShortestPath is the single hardest in
   KG-LLM-Bench (models under-predict length, hallucinate edges; Claude often answers "no path");
   connectivity, cycles, topological sort, max flow degrade fast (NLGift, GraphArena);
   activation/error-branch (stateful) tracking is hardest in MermaidSeqBench.
6. **NP-complete / combinatorial-optimization tasks** (MCP, MIS, TSP, GED) are worst across all
   models -- but failure here may reflect *reasoning*, not *reading*, so they are a confound for a
   pure reading study.
7. **Encoding-driven difficulty traps.** EstGraph: PageRank looks easy only because it aligns
   with walk visitation; betweenness/closeness are hard. Metric choice can leak structure and
   inflate scores.
8. **Named-pattern leakage.** Dai et al.: "terminology" (named-motif) questions tap pretrained
   knowledge, inflating apparent reading ability -- must be separated from topology reading.

## 4. Cross-Paper Synthesis: Graph-Size Effects

**Universal, monotonic degradation with size -- this is the dominant signal.**

- **Hard ceiling around ~50-100 nodes for direct serialization.** GAR's memory probe: 1-hop
  neighbor recall falls from ~90% (10 nodes) to ~0% (100 nodes) for Claude-3.5-Sonnet, GPT-4-turbo,
  Gemini-1.5-Pro. EstGraph treats ~50 nodes as the point past which structure cannot be serialized
  at all.
- **Top models degrade but the cliff varies.** GraphArena: Claude-3.5-Sonnet ~82% -> ~59% (small ->
  large polynomial). GraphARC: o1-mini 88% (n=10) -> 18% (n=250), while **GPT-5 stayed ~91% to
  n=250** -- strong reasoning models flatten the curve. GAR shows single LLMs hit 0/20 on
  shortest-path by 100-1000 nodes purely from context/attention limits.
- **Mechanism: quadratic token growth + iterative-reasoning collapse.** Edge/adjacency lists grow
  O(N^2); long encodings cause "lost-in-the-middle" and attention misallocation (Hu et al., GAR).
  Language-based algorithm *simulation* compounds error with size/density via repetitive
  backtracking.
- **Hallucination scales with size** (GraphArena): infeasible-but-formatted answers and
  order-of-magnitude estimate errors (EstGraph) appear mainly on large graphs.
- **Some global tasks stay robust; degree-sensitive tasks collapse fastest.** GraphARC:
  colorPath/colorComponents near-perfect to ~100 nodes; removeDegree3 96% -> 25% (n=10 -> 250).
  Difficulty is **not purely locality-driven**.
- **Size is the most damaging distribution shift** (NLGift: train-small -> test-large barely
  recovers) and **scaling doesn't auto-fix it** (bigger base models, more in-context examples,
  test-time compute give only modest help).
- **Some tasks are counterintuitively easier on larger graphs** (Dai et al. k-core: more nodes
  clear the degree-3 threshold) -- size effects are task-dependent in sign, not just magnitude.

## 5. The Gap Our Benchmark Fills

Every paper above leaves at least one of our four pillars open; together they define a clean,
unoccupied niche:

1. **Claude-only, controlled.** Prior benchmarks are multi-LLM and use older Claude checkpoints
   (KG-LLM-Bench: Claude-3.5-Sonnet-v2; Dai et al.: Claude-3-opus; GAR: Claude-3.5-Sonnet;
   EstGraph: Claude-sonnet-4). None hold the model family fixed to isolate the encoding/size axes
   on **current** Claude (with and without extended thinking). KG-LLM-Bench already shows Claude
   behaves *differently* from the global ranking (its best format was RDF Turtle, not JSON),
   so a Claude-specific sweep is warranted and not redundant.

2. **Head-to-head encodings, fairly compared.** Only KG-LLM-Bench (5 KG formats) and Dai et al.
   (adjacency vs edge list) systematically vary serialization; GraphARC tests just two
   (adjacency vs incident); GraphArena, Rethinking, and GAR **lock a single edge-list encoding**.
   None run the full sweep -- **edge list, adjacency list, adjacency matrix, JSON, YAML, GraphML,
   DOT, natural language** -- head-to-head on the *same* graphs. We own this axis. Crucially, we
   **control for token count** (report accuracy at fixed token budget), so we measure the
   encoding, not its verbosity (the lesson from KG-LLM-Bench's Turtle/JSON-LD loss).

3. **Simple, direct-reading queries (not algorithm-solving).** We deliberately isolate *parsing
   the encoding* from *reasoning*. Prior work conflates the two and then finds remedies that
   measure something else: code-writing (GraphArena, Hu et al.) and multi-agent scaffolds (GAR)
   reach ~98% but measure **tool use / scaffolding, not graph reading**. To isolate direct
   reading we (a) forbid code execution, (b) keep the graph in context during reasoning, (c) lean
   on poly-time/retrieval-style probes (edge existence, degree, neighbors, connectivity,
   shortest distance, component/cycle/node/edge counts -- the GraphToken/GraphARC question banks),
   and (d) treat NP-complete tasks as a confound, not a target. We also guard the confounds the
   literature flagged: **anonymized node labels** (Xu et al.: semantics leak; KG-LLM-Bench: ~free
   at 0.2%), **shuffled node/edge order** (Xu et al.: graphs read as unordered sets),
   **no named-pattern leakage** (Dai et al.: terminology inflates scores), and prompts that are
   unambiguous that **no transformation is requested** (GraphARC's transformation bias).

4. **Token-efficiency-vs-accuracy as a first-class metric.** No prior reading benchmark plots the
   **accuracy-per-token frontier** per encoding. Given the O(N^2) token growth and verbosity
   penalty documented in KG-LLM-Bench, and the ~50-100 node serialization ceiling (GAR, EstGraph),
   the practically useful question is not "which format is most accurate" but "which format gives
   the **most accuracy per token** at a given graph size." We score every encoding on both axes
   and report the Pareto frontier across our size tiers.

**Method choices borrowed wholesale:** GraphArena's 4-way CORRECT/SUBOPTIMAL/HALLUCINATORY/MISSING
rubric (hallucination is exactly what direct-reading must catch); deterministic exact-match /
isomorphism scoring with **no LLM-as-judge** (MermaidSeqBench's judge-disagreement caution);
seeded generators (ER p=0.3, Watts-Strogatz, tree, star, bipartite, 2-component from GraphARC;
Barabasi-Albert from NLGift); tractability-aware per-task size tiers (GraphArena); GAR's Figure-2
1-hop recall probe as a structure-comprehension floor; and binning results by aggregation size
and edge direction (KG-LLM-Bench's most actionable diagnostics).

---

*Confidence note:* some quantitative figures for GraphInstruct, GraphArena, and the Rethinking
paper were extracted via fast-model summarization of HTML/PDF and should be re-verified against
the saved PDFs in this directory before being cited as exact numbers.
