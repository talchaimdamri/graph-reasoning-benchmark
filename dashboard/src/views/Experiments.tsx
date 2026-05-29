import { useMemo, useState } from "react";
import type { DashboardData } from "../types";
import { he } from "../i18n/he";

const ALL = "__all__";

function uniq(values: string[]): string[] {
  return Array.from(new Set(values)).sort();
}

export function Experiments({ data }: { data: DashboardData }) {
  const t = he.experiments;
  const [model, setModel] = useState(ALL);
  const [encoding, setEncoding] = useState(ALL);
  const [difficulty, setDifficulty] = useState(ALL);
  const [correct, setCorrect] = useState(ALL);

  const models = useMemo(() => uniq(data.results.map((r) => r.model)), [data]);
  const encodings = useMemo(
    () => uniq(data.results.map((r) => r.encoding)),
    [data],
  );
  const difficulties = useMemo(
    () => uniq(data.results.map((r) => r.difficulty)),
    [data],
  );

  const rows = useMemo(() => {
    return data.results.filter((r) => {
      if (model !== ALL && r.model !== model) return false;
      if (encoding !== ALL && r.encoding !== encoding) return false;
      if (difficulty !== ALL && r.difficulty !== difficulty) return false;
      if (correct === "yes" && !r.correct) return false;
      if (correct === "no" && r.correct) return false;
      return true;
    });
  }, [data, model, encoding, difficulty, correct]);

  // Cap rendered rows for performance; note tells the user the full count.
  const shown = rows.slice(0, 500);

  return (
    <div>
      <div className="view-head">
        <h2>{t.title}</h2>
        <p>{t.description}</p>
      </div>

      <div className="filters">
        <label>
          {t.filterModel}
          <select value={model} onChange={(e) => setModel(e.target.value)}>
            <option value={ALL}>{t.all}</option>
            {models.map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </select>
        </label>
        <label>
          {t.filterEncoding}
          <select value={encoding} onChange={(e) => setEncoding(e.target.value)}>
            <option value={ALL}>{t.all}</option>
            {encodings.map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </select>
        </label>
        <label>
          {t.filterDifficulty}
          <select
            value={difficulty}
            onChange={(e) => setDifficulty(e.target.value)}
          >
            <option value={ALL}>{t.all}</option>
            {difficulties.map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </select>
        </label>
        <label>
          {t.filterCorrect}
          <select value={correct} onChange={(e) => setCorrect(e.target.value)}>
            <option value={ALL}>{t.all}</option>
            <option value="yes">{t.onlyCorrect}</option>
            <option value="no">{t.onlyWrong}</option>
          </select>
        </label>
      </div>

      <p className="count-note">
        {t.showing} {shown.length} {t.of} {rows.length} {t.rows}
      </p>

      <div className="table-scroll">
        <table>
          <thead>
            <tr>
              <th>{t.columns.graph}</th>
              <th>{t.columns.encoding}</th>
              <th>{t.columns.model}</th>
              <th>{t.columns.question}</th>
              <th>{t.columns.difficulty}</th>
              <th>{t.columns.category}</th>
              <th>{t.columns.correct}</th>
              <th className="num">{t.columns.tokens}</th>
              <th className="num">{t.columns.latency}</th>
              <th>{t.columns.error}</th>
            </tr>
          </thead>
          <tbody>
            {shown.map((r) => (
              <tr key={r.result_id}>
                <td>{r.graph_id}</td>
                <td>
                  <span className="badge tag">{r.encoding}</span>
                </td>
                <td>{r.model}</td>
                <td
                  style={{
                    whiteSpace: "normal",
                    maxWidth: 280,
                  }}
                  dir="ltr"
                >
                  {r.question_text}
                </td>
                <td>{r.difficulty}</td>
                <td>{r.category}</td>
                <td>
                  <span className={`badge ${r.correct ? "ok" : "bad"}`}>
                    {r.correct ? t.yes : t.no}
                  </span>
                </td>
                <td className="num">{r.tokens_used}</td>
                <td className="num">{Math.round(r.latency_ms)}</td>
                <td dir="ltr">{r.error ?? t.none}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
