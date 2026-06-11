import type { Decision, Label, Outcome, ResultRow } from "@/lib/types";

/** All correctness classes, in display order. */
export const ALL_OUTCOMES: Outcome[] = [
  "kept-smoke",
  "discarded-fp",
  "kept-fp",
  "discarded-smoke",
  "n/a",
];

export interface Filters {
  outcomes: Outcome[]; // enabled correctness classes (a row shows iff its outcome is in here)
  label: "all" | Label;
  verdict: "all" | Decision;
  camera: "all" | string;
}

export const defaultFilters = (): Filters => ({
  outcomes: [...ALL_OUTCOMES],
  label: "all",
  verdict: "all",
  camera: "all",
});

export function applyFilters(rows: ResultRow[], f: Filters): ResultRow[] {
  return rows.filter(
    (r) =>
      f.outcomes.includes(r.outcome) &&
      (f.label === "all" || r.label === f.label) &&
      (f.verdict === "all" || r.decision === f.verdict) &&
      (f.camera === "all" || r.camera_name === f.camera),
  );
}

/** Sorted distinct non-null camera names present in the rows. */
export function cameraOptions(rows: ResultRow[]): string[] {
  return [
    ...new Set(
      rows.map((r) => r.camera_name).filter((c): c is string => c != null),
    ),
  ].sort();
}
