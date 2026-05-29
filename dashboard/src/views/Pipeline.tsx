import { he } from "../i18n/he";

const STAGE_CMD: Record<string, string> = {
  generate: "make_tiered_graph()",
  encode: "encode_graph()",
  question: "generate_questions()",
  benchmark: "run_benchmark()",
  analyze: "build_dashboard_data.py",
};

export function Pipeline() {
  const t = he.pipeline;
  return (
    <div>
      <div className="view-head">
        <h2>{t.title}</h2>
        <p>{t.description}</p>
      </div>

      <div className="pipeline">
        {t.stages.map((s, i) => (
          <div className="stage" key={s.key}>
            <span className="step">{i + 1}</span>
            <h3>{s.title}</h3>
            <p>{s.body}</p>
            <p style={{ marginTop: 10 }}>
              <code>{STAGE_CMD[s.key]}</code>
            </p>
          </div>
        ))}
      </div>
    </div>
  );
}
