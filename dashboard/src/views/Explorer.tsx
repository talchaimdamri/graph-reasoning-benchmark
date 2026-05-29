import { useEffect, useRef, useState } from "react";
import cytoscape from "cytoscape";
import type { Core, ElementDefinition } from "cytoscape";
import type { DashboardData, GraphPayload } from "../types";
import { he } from "../i18n/he";

function toElements(graph: GraphPayload): ElementDefinition[] {
  const els: ElementDefinition[] = graph.nodes.map((id) => ({
    data: { id, label: id },
  }));
  graph.edges.forEach((e, i) => {
    els.push({
      data: {
        id: `e${i}`,
        source: e.source,
        target: e.target,
        label: e.weight != null ? String(e.weight) : "",
      },
    });
  });
  return els;
}

const LAYOUTS = ["cose", "circle", "grid", "concentric", "breadthfirst"];

export function Explorer({ data }: { data: DashboardData }) {
  const t = he.explorer;
  const [selected, setSelected] = useState(data.graphs[0]?.id ?? "");
  const [layout, setLayout] = useState("cose");
  const containerRef = useRef<HTMLDivElement>(null);
  const cyRef = useRef<Core | null>(null);

  const graph = data.graphs.find((g) => g.id === selected);

  useEffect(() => {
    if (!containerRef.current || !graph) return;
    const directed = graph.metadata.directed;
    const cy = cytoscape({
      container: containerRef.current,
      elements: toElements(graph),
      style: [
        {
          selector: "node",
          style: {
            "background-color": "#5b8def",
            label: "data(label)",
            color: "#f1f4fa",
            "font-size": "10px",
            "text-valign": "center",
            "text-halign": "center",
            width: 26,
            height: 26,
          },
        },
        {
          selector: "edge",
          style: {
            width: 1.5,
            "line-color": "#4ec9a8",
            "target-arrow-color": "#4ec9a8",
            "target-arrow-shape": directed ? "triangle" : "none",
            "curve-style": "bezier",
            label: "data(label)",
            "font-size": "8px",
            color: "#8a92a3",
            "text-background-color": "#171a23",
            "text-background-opacity": 1,
          },
        },
      ],
      layout: { name: layout, animate: false } as cytoscape.LayoutOptions,
      wheelSensitivity: 0.2,
    });
    cyRef.current = cy;
    return () => {
      cy.destroy();
      cyRef.current = null;
    };
  }, [graph, layout]);

  return (
    <div>
      <div className="view-head">
        <h2>{t.title}</h2>
        <p>{t.description}</p>
      </div>

      <div className="filters">
        <label>
          {t.selectGraph}
          <select value={selected} onChange={(e) => setSelected(e.target.value)}>
            {data.graphs.map((g) => (
              <option key={g.id} value={g.id}>
                {g.id}
              </option>
            ))}
          </select>
        </label>
        <label>
          {t.layout}
          <select value={layout} onChange={(e) => setLayout(e.target.value)}>
            {LAYOUTS.map((l) => (
              <option key={l} value={l}>
                {l}
              </option>
            ))}
          </select>
        </label>
      </div>

      {graph && (
        <div className="cy-meta">
          <span>
            {t.nodes}: <b>{graph.metadata.num_nodes}</b>
          </span>
          <span>
            {t.edges}: <b>{graph.edges.length}</b>
          </span>
          <span>
            {t.tier}: <b>{graph.metadata.tier}</b>
          </span>
          <span>
            {t.directed}: <b>{graph.metadata.directed ? he.experiments.yes : he.experiments.no}</b>
          </span>
          <span>
            {t.weighted}: <b>{graph.metadata.weighted ? he.experiments.yes : he.experiments.no}</b>
          </span>
        </div>
      )}

      <div className="cy-wrap" ref={containerRef} />
    </div>
  );
}
