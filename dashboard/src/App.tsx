import { useState } from "react";
import { he } from "./i18n/he";
import { useData } from "./useData";
import type { Summary } from "./types";
import { Overview } from "./views/Overview";
import { Experiments } from "./views/Experiments";
import { Pipeline } from "./views/Pipeline";
import { Explorer } from "./views/Explorer";
import { MetricsView } from "./views/MetricsView";

type Tab = "overview" | "experiments" | "pipeline" | "explorer" | "metrics";

const TABS: { key: Tab; label: string }[] = [
  { key: "overview", label: he.nav.overview },
  { key: "metrics", label: he.nav.metrics },
  { key: "experiments", label: he.nav.experiments },
  { key: "explorer", label: he.nav.explorer },
  { key: "pipeline", label: he.nav.pipeline },
];

function fmtInt(n: number) {
  return n.toLocaleString("he-IL");
}
function pct(v: number) {
  return `${Math.round(v * 100)}%`;
}

function Cards({ s }: { s: Summary }) {
  const items = [
    { label: he.summary.overallAccuracy, value: pct(s.overall_accuracy) },
    { label: he.summary.totalResults, value: fmtInt(s.total_results) },
    { label: he.summary.models, value: fmtInt(s.n_models) },
    { label: he.summary.encodings, value: fmtInt(s.n_encodings) },
    { label: he.summary.graphs, value: fmtInt(s.n_graphs) },
    { label: he.summary.totalTokens, value: fmtInt(s.total_tokens) },
    { label: he.summary.errorRate, value: pct(s.error_rate) },
  ];
  return (
    <div className="cards">
      {items.map((it) => (
        <div className="card" key={it.label}>
          <div className="label">{it.label}</div>
          <div className="value">{it.value}</div>
        </div>
      ))}
    </div>
  );
}

export default function App() {
  const [tab, setTab] = useState<Tab>("overview");
  const { data, loading, error } = useData();

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="brand">
          <h1>{he.appTitle}</h1>
          <p>Graph Reasoning Benchmark</p>
        </div>
        {TABS.map((t) => (
          <button
            key={t.key}
            className={`nav-btn ${tab === t.key ? "active" : ""}`}
            onClick={() => setTab(t.key)}
          >
            {t.label}
          </button>
        ))}
        {data && (
          <p style={{ marginTop: "auto", fontSize: 11, color: "var(--text-dim)" }}>
            {he.generatedAt}:<br />
            <span dir="ltr">{data.generated_at.slice(0, 19).replace("T", " ")}</span>
          </p>
        )}
      </aside>

      <main className="main">
        {loading && <div className="status">{he.loading}</div>}
        {error && <div className="status error">{he.loadError}</div>}
        {data && (
          <>
            {(tab === "overview" || tab === "metrics") && (
              <Cards s={data.summary} />
            )}
            {tab === "overview" && <Overview data={data} />}
            {tab === "metrics" && <MetricsView data={data} />}
            {tab === "experiments" && <Experiments data={data} />}
            {tab === "explorer" && <Explorer data={data} />}
            {tab === "pipeline" && <Pipeline />}
          </>
        )}
      </main>
    </div>
  );
}
