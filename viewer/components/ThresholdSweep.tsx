import { useMemo } from "react";
import type { ResultRow } from "@/lib/types";

// Dense 0.05 steps around the ~0.45 operating point, bigger gaps at the extremes.
const THRESHOLDS = [
  0.1, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9,
];

/**
 * Compact tradeoff table under the triage slider: at each standard threshold,
 * how many sequences land in To Review vs Unlabel. Counts depend only on the
 * scores (not the live slider), so they're stable while dragging; the row
 * nearest the current threshold is highlighted.
 */
export function ThresholdSweep({
  rows,
  current,
}: {
  rows: ResultRow[];
  current: number;
}) {
  const total = rows.length;
  const counts = useMemo(() => {
    const probs = rows
      .map((r) => r.probability)
      .filter((p): p is number => p != null);
    return THRESHOLDS.map((t) => {
      const review = probs.filter((p) => p >= t).length;
      return { t, review, unlabel: total - review };
    });
  }, [rows, total]);

  // Highlight the standard threshold closest to the live slider value.
  const nearest = THRESHOLDS.reduce((a, b) =>
    Math.abs(b - current) < Math.abs(a - current) ? b : a,
  );

  return (
    <div className="rounded-xl border border-slate-200 bg-white p-2.5">
      <div className="mb-1 text-[9px] font-medium uppercase tracking-tight text-slate-500">
        Threshold sweep
      </div>
      <table className="w-full text-[11px] tabular-nums">
        <thead>
          <tr className="text-slate-400">
            <th className="text-left font-normal">thr</th>
            <th className="text-right font-normal">review</th>
            <th className="text-right font-normal">unlabel</th>
            <th className="text-right font-normal">%</th>
          </tr>
        </thead>
        <tbody>
          {counts.map(({ t, review, unlabel }) => {
            const active = t === nearest;
            const pct = total ? Math.round((review / total) * 100) : 0;
            return (
              <tr
                key={t}
                className={active ? "bg-slate-100 font-medium" : ""}
                aria-current={active ? "true" : undefined}
              >
                <td className="text-left text-slate-600">{t.toFixed(2)}</td>
                <td className="text-right" style={{ color: "#047857" }}>
                  {review.toLocaleString()}
                </td>
                <td className="text-right text-slate-400">
                  {unlabel.toLocaleString()}
                </td>
                <td className="text-right text-slate-400">{pct}%</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
