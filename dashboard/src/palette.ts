// Shared chart palette + a viridis-ish scale for the accuracy heatmap.

export const SERIES_COLORS = [
  "#5b8def",
  "#4ec9a8",
  "#e0a04a",
  "#c084fc",
  "#e0584a",
  "#4ec98a",
  "#f06ea9",
];

// Map a 0..1 accuracy to a heatmap color (purple -> teal -> yellow).
export function accuracyColor(v: number): string {
  if (Number.isNaN(v)) return "#2a2f3c";
  const stops: Array<[number, [number, number, number]]> = [
    [0, [68, 1, 84]],
    [0.5, [33, 145, 140]],
    [1, [253, 231, 37]],
  ];
  let lo = stops[0];
  let hi = stops[stops.length - 1];
  for (let i = 0; i < stops.length - 1; i++) {
    if (v >= stops[i][0] && v <= stops[i + 1][0]) {
      lo = stops[i];
      hi = stops[i + 1];
      break;
    }
  }
  const t = (v - lo[0]) / (hi[0] - lo[0] || 1);
  const c = lo[1].map((x, i) => Math.round(x + (hi[1][i] - x) * t));
  return `rgb(${c[0]}, ${c[1]}, ${c[2]})`;
}

export const CHART_GRID = "#2a2f3c";
export const CHART_TEXT = "#8a92a3";
