import type { Decision, Label, Outcome, ResultRow } from "@/lib/types";

export function computeOutcome(decision: Decision, label: Label): Outcome {
  if (label === "smoke")
    return decision === "keep" ? "kept-smoke" : "discarded-smoke";
  if (label === "fp") return decision === "keep" ? "kept-fp" : "discarded-fp";
  return "n/a";
}

/** Re-decide every row at a logistic threshold (does not mutate input). */
export function applyThreshold(
  rows: ResultRow[],
  threshold: number,
): ResultRow[] {
  return rows.map((r) => {
    const decision: Decision =
      r.probability != null && r.probability >= threshold ? "keep" : "discard";
    return { ...r, decision, outcome: computeOutcome(decision, r.label) };
  });
}

export interface PerfSummary {
  nLabeled: number;
  nSmoke: number;
  nFp: number;
  keptSmoke: number;
  discardedSmoke: number;
  discardedFp: number;
  keptFp: number;
  recall: number | null;
  specificity: number | null;
  precision: number | null;
}

export function performanceSummary(rows: ResultRow[]): PerfSummary {
  const labeled = rows.filter((r) => r.label === "smoke" || r.label === "fp");
  const n = (o: Outcome) => labeled.filter((r) => r.outcome === o).length;
  const keptSmoke = n("kept-smoke");
  const discardedSmoke = n("discarded-smoke");
  const discardedFp = n("discarded-fp");
  const keptFp = n("kept-fp");
  const nSmoke = keptSmoke + discardedSmoke;
  const nFp = discardedFp + keptFp;
  const nKept = keptSmoke + keptFp;
  return {
    nLabeled: nSmoke + nFp,
    nSmoke,
    nFp,
    keptSmoke,
    discardedSmoke,
    discardedFp,
    keptFp,
    recall: nSmoke ? keptSmoke / nSmoke : null,
    specificity: nFp ? discardedFp / nFp : null,
    precision: nKept ? keptSmoke / nKept : null,
  };
}
