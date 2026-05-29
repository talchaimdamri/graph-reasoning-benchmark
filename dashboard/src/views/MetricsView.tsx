import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { DashboardData } from "../types";
import { he } from "../i18n/he";
import { accuracyColor, CHART_GRID, CHART_TEXT, SERIES_COLORS } from "../palette";

const axis = { stroke: CHART_TEXT, fontSize: 12 };
const tooltipStyle = {
  background: "#1e222d",
  border: "1px solid #2a2f3c",
  borderRadius: 8,
  color: "#f1f4fa",
  direction: "rtl" as const,
};

function pct(v: number | undefined) {
  if (v == null || Number.isNaN(v)) return "—";
  return `${Math.round(v * 100)}%`;
}

// recharts hands formatters a loosely-typed value.
const pctFmt = (v: unknown) => pct(Number(v));

export function MetricsView({ data }: { data: DashboardData }) {
  const t = he.metricsView;
  const mxf = data.metrics.model_x_format;

  // Lookup accuracy from the row records (keyed by encoding column).
  const cell = (model: string, enc: string): number => {
    const row = mxf.rows.find((r) => r.model === model);
    if (!row) return NaN;
    const v = row[enc];
    return typeof v === "number" ? v : NaN;
  };

  const byCategory = data.metrics.by_category.map((r) => ({
    category: r.category,
    accuracy: r.accuracy,
  }));

  return (
    <div>
      <div className="view-head">
        <h2>{t.title}</h2>
        <p>{t.description}</p>
      </div>

      <div className="panel">
        <h3>{t.heatmap}</h3>
        <div style={{ overflowX: "auto" }}>
          <table className="heatmap">
            <thead>
              <tr>
                <th>{t.model}</th>
                {mxf.encodings.map((e) => (
                  <th key={e}>{e}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {mxf.models.map((m) => (
                <tr key={m}>
                  <th>{m}</th>
                  {mxf.encodings.map((e) => {
                    const v = cell(m, e);
                    return (
                      <td
                        key={e}
                        style={{ background: accuracyColor(v) }}
                        title={`${m} × ${e}`}
                      >
                        {pct(v)}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="grid-2">
        <div className="panel">
          <h3>{t.byCategory}</h3>
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={byCategory} layout="vertical" margin={{ right: 16 }}>
              <CartesianGrid stroke={CHART_GRID} horizontal={false} />
              <XAxis type="number" domain={[0, 1]} tickFormatter={pctFmt} {...axis} />
              <YAxis type="category" dataKey="category" width={130} {...axis} />
              <Tooltip formatter={pctFmt} contentStyle={tooltipStyle} />
              <Bar dataKey="accuracy" fill={SERIES_COLORS[0]} radius={[0, 4, 4, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="panel">
          <h3>{t.errorBreakdown}</h3>
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={data.metrics.error_breakdown} margin={{ bottom: 30 }}>
              <CartesianGrid stroke={CHART_GRID} vertical={false} />
              <XAxis
                dataKey="error"
                angle={-20}
                textAnchor="end"
                interval={0}
                {...axis}
              />
              <YAxis {...axis} />
              <Tooltip contentStyle={tooltipStyle} />
              <Bar dataKey="count" name={he.chart.count} fill={SERIES_COLORS[4]} radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="panel">
        <h3>{t.table}</h3>
        <div className="table-scroll">
          <table>
            <thead>
              <tr>
                <th>{t.model}</th>
                {mxf.encodings.map((e) => (
                  <th className="num" key={e}>
                    {e}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {mxf.models.map((m) => (
                <tr key={m}>
                  <td>{m}</td>
                  {mxf.encodings.map((e) => (
                    <td className="num" key={e}>
                      {pct(cell(m, e))}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
