// Mirrors the payload written by grb.metrics.export_dashboard_json.
// All field names stay English (data layer); Hebrew lives in the UI only.

export interface Summary {
  total_results: number;
  overall_accuracy: number;
  n_models: number;
  n_encodings: number;
  n_graphs: number;
  total_tokens: number;
  error_rate: number;
}

export interface AccuracyRow {
  accuracy: number;
  n: number;
  correct: number;
  encoding?: string;
  model?: string;
  difficulty?: string;
  category?: string;
  tier?: string;
}

export interface TokenEfficiencyRow {
  encoding: string;
  mean_tokens: number;
  accuracy: number;
  n: number;
  accuracy_per_1k: number;
}

export interface AccuracyVsTokensRow {
  model: string;
  encoding: string;
  mean_tokens: number;
  accuracy: number;
  n: number;
}

export interface ErrorRow {
  error: string;
  count: number;
}

export interface ModelXFormat {
  models: string[];
  encodings: string[];
  rows: Array<Record<string, number | string>>;
}

export interface Metrics {
  by_encoding: AccuracyRow[];
  by_model: AccuracyRow[];
  by_difficulty: AccuracyRow[];
  by_category: AccuracyRow[];
  by_tier: AccuracyRow[];
  token_efficiency: TokenEfficiencyRow[];
  accuracy_vs_tokens: AccuracyVsTokensRow[];
  error_breakdown: ErrorRow[];
  model_x_format: ModelXFormat;
}

export interface ResultRow {
  result_id: string;
  run_id: string;
  graph_id: string;
  encoding: string;
  question_id: string;
  question_text: string;
  model: string;
  correct: boolean;
  tokens_used: number;
  latency_ms: number;
  error: string | null;
  category: string;
  difficulty: string;
  tier: string;
  num_nodes: number | null;
  num_edges: number | null;
}

export interface GraphEdge {
  source: string;
  target: string;
  type: string | null;
  weight: number | null;
}

export interface GraphMeta {
  directed: boolean;
  weighted: boolean;
  multi_edge: boolean;
  hierarchy_depth: number;
  seed: number;
  tier: string;
  num_nodes: number;
}

export interface GraphPayload {
  id: string;
  metadata: GraphMeta;
  nodes: string[];
  edges: GraphEdge[];
}

export interface DashboardData {
  generated_at: string;
  summary: Summary;
  metrics: Metrics;
  results: ResultRow[];
  graphs: GraphPayload[];
}
