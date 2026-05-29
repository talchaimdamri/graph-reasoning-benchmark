import { useEffect, useState } from "react";
import type { DashboardData } from "./types";

interface State {
  data: DashboardData | null;
  loading: boolean;
  error: string | null;
}

// Loads /results.json (served from dashboard/public). Vite's BASE_URL keeps it
// correct under a sub-path deploy.
export function useData(): State {
  const [state, setState] = useState<State>({
    data: null,
    loading: true,
    error: null,
  });

  useEffect(() => {
    let alive = true;
    const url = `${import.meta.env.BASE_URL}results.json`;
    fetch(url)
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then((data: DashboardData) => {
        if (alive) setState({ data, loading: false, error: null });
      })
      .catch((e: unknown) => {
        if (alive)
          setState({ data: null, loading: false, error: String(e) });
      });
    return () => {
      alive = false;
    };
  }, []);

  return state;
}
