import { performanceSummary, type PerfSummary } from "@/lib/outcomes";
import type { ResultRow } from "@/lib/types";

const pct = (v: number | null) => (v == null ? "—" : `${(v * 100).toFixed(1)}%`);

export function PerfCards({ rows }: { rows: ResultRow[] }) {
  const s: PerfSummary = performanceSummary(rows);
  if (s.nLabeled === 0) return null;
  const cards: [string, string, string][] = [
    ["Recall (smoke kept)", pct(s.recall), `${s.keptSmoke}/${s.nSmoke}`],
    ["FP filtered", pct(s.specificity), `${s.discardedFp}/${s.nFp}`],
    ["Precision", pct(s.precision), `${s.keptSmoke}/${s.keptSmoke + s.keptFp}`],
  ];
  return (
    <div className="grid grid-cols-3 gap-2">
      {cards.map(([label, value, frac]) => (
        <div key={label} className="rounded-lg border border-slate-200 bg-white p-2">
          <div className="text-[11px] uppercase tracking-wide text-slate-500">{label}</div>
          <div className="text-lg font-semibold text-slate-900">{value}</div>
          <div className="text-[11px] text-slate-400">{frac}</div>
        </div>
      ))}
    </div>
  );
}
