import type { ResultRow } from "@/lib/types";

export type SortCol =
  | "camera"
  | "label"
  | "decision"
  | "outcome"
  | "probability"
  | "tubes";
export interface Sort {
  col: SortCol;
  dir: "asc" | "desc";
}

const str = (r: ResultRow, col: SortCol): string => {
  switch (col) {
    case "camera":
      return r.camera_name ?? "";
    case "label":
      return r.label;
    case "decision":
      return r.decision;
    case "outcome":
      return r.outcome;
    default:
      return "";
  }
};

/**
 * Stable copy of `rows` sorted by `sort`. Null probabilities always sort last
 * (both directions), so "probability desc" surfaces the highest-confidence rows
 * and pushes no-kept-tube sequences to the bottom.
 */
export function sortRows(rows: ResultRow[], sort: Sort | null): ResultRow[] {
  if (!sort) return [...rows];
  const sign = sort.dir === "asc" ? 1 : -1;
  return [...rows].sort((a, b) => {
    if (sort.col === "probability") {
      const av = a.probability;
      const bv = b.probability;
      if (av == null && bv == null) return 0;
      if (av == null) return 1;
      if (bv == null) return -1;
      return sign * (av - bv);
    }
    if (sort.col === "tubes") {
      return sign * ((a.num_tubes_kept ?? 0) - (b.num_tubes_kept ?? 0));
    }
    return sign * str(a, sort.col).localeCompare(str(b, sort.col));
  });
}

/** Click behaviour: same column toggles asc/desc; a new column starts desc. */
export function nextSort(current: Sort | null, col: SortCol): Sort {
  if (current && current.col === col) {
    return { col, dir: current.dir === "asc" ? "desc" : "asc" };
  }
  return { col, dir: "desc" };
}
