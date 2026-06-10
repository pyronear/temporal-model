import { performanceSummary, type PerfSummary } from "@/lib/outcomes";
import type { ResultRow } from "@/lib/types";

const pct = (v: number | null) => (v == null ? "—" : `${(v * 100).toFixed(1)}%`);

interface Card {
  label: string;
  value: number | null;
  frac: string;
  color: string;
}

export function PerfCards({ rows }: { rows: ResultRow[] }) {
  const s: PerfSummary = performanceSummary(rows);
  if (s.nLabeled === 0) return null;
  const cards: Card[] = [
    { label: "Recall", value: s.recall, frac: `${s.keptSmoke}/${s.nSmoke} smoke kept`, color: "#059669" },
    { label: "FP filtered", value: s.specificity, frac: `${s.discardedFp}/${s.nFp} fp`, color: "#0d9488" },
    {
      label: "Precision",
      value: s.precision,
      frac: `${s.keptSmoke}/${s.keptSmoke + s.keptFp} kept`,
      color: "#6366f1",
    },
  ];
  return (
    <div className="grid grid-cols-3 gap-2">
      {cards.map((c) => (
        <div key={c.label} className="flex flex-col rounded-xl border border-slate-200 bg-white p-2.5">
          <div className="flex h-8 items-start text-[10px] font-medium uppercase leading-tight tracking-wide text-slate-500">
            {c.label}
          </div>
          <div className="text-xl font-semibold tabular-nums" style={{ color: c.color }}>
            {pct(c.value)}
          </div>
          <div className="mt-1.5 h-1 overflow-hidden rounded-full bg-slate-100">
            <div
              className="h-full rounded-full"
              style={{ width: `${(c.value ?? 0) * 100}%`, background: c.color }}
            />
          </div>
          <div className="mt-1 text-[10px] text-slate-400">{c.frac}</div>
        </div>
      ))}
    </div>
  );
}
