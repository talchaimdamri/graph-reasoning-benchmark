"""Generate ~6 example graphs across tiers/models and save JSON + images.

Run:
    .venv/bin/python examples/generate_examples.py

Outputs land in ``examples/graphs/*.json`` and ``examples/images/*.{png,svg}``.
"""

from __future__ import annotations

import json
from pathlib import Path

from grb.generator import describe_graph, generate_graph, make_tiered_graph
from grb.visualizer import render

HERE = Path(__file__).resolve().parent
GRAPH_DIR = HERE / "graphs"
IMAGE_DIR = HERE / "images"


def _save(graph, name: str) -> dict:
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    json_path = GRAPH_DIR / f"{name}.json"
    json_path.write_text(json.dumps(graph.model_dump(), indent=2) + "\n")
    images = render(graph, IMAGE_DIR / name, fmt="both")
    return {
        "name": name,
        "id": graph.id,
        "tier": graph.metadata.tier,
        "model": name.split("__")[1] if "__" in name else "",
        "nodes": len(graph.nodes),
        "edges": len(graph.edges),
        "json": str(json_path),
        "images": [str(p) for p in images],
    }


def main() -> None:
    manifest = []

    # Three tier-calibrated graphs.
    manifest.append(_save(make_tiered_graph("small", seed=0), "small__random"))
    manifest.append(_save(make_tiered_graph("medium", seed=0), "medium__hierarchical"))
    manifest.append(_save(make_tiered_graph("large", seed=0), "large__scale_free"))

    # Three more illustrating specific model features.
    manifest.append(
        _save(
            generate_graph(10, seed=1, directed=False, weighted=True,
                           model="random", edge_prob=0.35),
            "demo__random_undirected_weighted",
        )
    )
    manifest.append(
        _save(
            generate_graph(16, seed=2, directed=True, model="hierarchical",
                           hierarchy_depth=3,
                           multi_edge_types=["calls", "imports", "extends"]),
            "demo__hierarchical_multiedge",
        )
    )
    manifest.append(
        _save(
            generate_graph(40, seed=3, directed=True, weighted=True,
                           model="scale_free", scale_free_m=3),
            "demo__scale_free_directed",
        )
    )

    (HERE / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for m in manifest:
        print(f"{m['name']:35s} tier={m['tier']:6s} "
              f"nodes={m['nodes']:3d} edges={m['edges']:4d}")


if __name__ == "__main__":
    main()
