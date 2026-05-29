import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from "recharts";
import type { DashboardData } from "../types";
import { he } from "../i18n/he";
import { CHART_GRID, CHART_TEXT, SERIES_COLORS } from "../palette";

const axis = { stroke: CHART_TEXT, fontSize: 12 };
const tooltipStyle = {
  background: "#1e222d",
  border: "1px solid #2a2f3c",
  borderRadius: 8,
  color: "#f1f4fa",
  direction: "rtl" as const,
};

function pct(v: number) {
  return `${Math.round(v * 100)}%`;
}

// Tooltip/axis formatters receive recharts' loosely-typed value.
const pctFmt = (v: unknown) => pct(Number(v));

export function Overview({ data }: { data: DashboardData }) {
  const { metrics } = data;

  const leaderboard = metrics.by_model.map((r) => ({
    model: r.model,
    accuracy: r.accuracy,
  }));
  const encodings = metrics.by_encoding.map((r) => ({
    encoding: r.encoding,
    accuracy: r.accuracy,
  }));
  const tokenEff = metrics.token_efficiency.map((r) => ({
    encoding: r.encoding,
    accuracy_per_1k: r.accuracy_per_1k,
  }));

  // Build per-encoding scatter series for accuracy vs tokens.
  const encNames = Array.from(
    new Set(metrics.accuracy_vs_tokens.map((r) => r.encoding)),
  );

  return (
    <div>
      <div className="view-head">
        <h2>{he.nav.overview}</h2>
        <p>{he.appSubtitle}</p>
      </div>

      <div className="grid-2">
        <div className="panel">
          <h3>{he.overview.leaderboard}</h3>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={leaderboard} layout="vertical" margin={{ right: 16 }}>
              <CartesianGrid stroke={CHART_GRID} horizontal={false} />
              <XAxis type="number" domain={[0, 1]} tickFormatter={pctFmt} {...axis} />
              <YAxis type="category" dataKey="model" width={120} {...axis} />
              <Tooltip formatter={pctFmt} contentStyle={tooltipStyle} />
              <Bar dataKey="accuracy" radius={[0, 4, 4, 0]}>
                {leaderboard.map((_, i) => (
                  <Cell key={i} fill={SERIES_COLORS[i % SERIES_COLORS.length]} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="panel">
          <h3>{he.overview.encodingComparison}</h3>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={encodings} margin={{ bottom: 28 }}>
              <CartesianGrid stroke={CHART_GRID} vertical={false} />
              <XAxis
                dataKey="encoding"
                angle={-25}
                textAnchor="end"
                interval={0}
                {...axis}
              />
              <YAxis domain={[0, 1]} tickFormatter={pctFmt} {...axis} />
              <Tooltip formatter={pctFmt} contentStyle={tooltipStyle} />
              <Bar dataKey="accuracy" fill={SERIES_COLORS[1]} radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="panel">
          <h3>{he.overview.tokenEfficiency}</h3>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={tokenEff} margin={{ bottom: 28 }}>
              <CartesianGrid stroke={CHART_GRID} vertical={false} />
              <XAxis
                dataKey="encoding"
                angle={-25}
                textAnchor="end"
                interval={0}
                {...axis}
              />
              <YAxis {...axis} />
              <Tooltip contentStyle={tooltipStyle} />
              <Bar
                dataKey="accuracy_per_1k"
                name={he.chart.accuracyPer1k}
                fill={SERIES_COLORS[2]}
                radius={[4, 4, 0, 0]}
              />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="panel">
          <h3>{he.overview.accuracyVsTokens}</h3>
          <ResponsiveContainer width="100%" height={260}>
            <ScatterChart margin={{ bottom: 16, left: 4 }}>
              <CartesianGrid stroke={CHART_GRID} />
              <XAxis
                type="number"
                dataKey="mean_tokens"
                name={he.chart.meanTokens}
                {...axis}
              />
              <YAxis
                type="number"
                dataKey="accuracy"
                name={he.chart.accuracy}
                domain={[0, 1]}
                tickFormatter={pctFmt}
                {...axis}
              />
              <ZAxis type="number" range={[60, 60]} />
              <Tooltip
                contentStyle={tooltipStyle}
                formatter={(v: unknown, n: unknown) =>
                  n === he.chart.accuracy ? pct(Number(v)) : String(v)
                }
              />
              <Legend wrapperStyle={{ fontSize: 12 }} />
              {encNames.map((enc, i) => (
                <Scatter
                  key={enc}
                  name={enc}
                  data={metrics.accuracy_vs_tokens.filter(
                    (r) => r.encoding === enc,
                  )}
                  fill={SERIES_COLORS[i % SERIES_COLORS.length]}
                />
              ))}
            </ScatterChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}
