"use client";
import { useMemo, useRef, type ReactNode } from "react";
import { correctnessLabel, outcomeTokens, rowTokens } from "@/lib/correctness";
import type { Sort, SortCol } from "@/lib/sort";
import type { ResultRow } from "@/lib/types";

const num = (v: number | null) => (v == null ? "—" : v.toFixed(3));

interface Column {
  header: string;
  sortCol: SortCol | null;
  render: (r: ResultRow) => ReactNode;
  cellStyle?: (r: ResultRow) => React.CSSProperties;
}

const COLUMNS: Column[] = [
  { header: "camera", sortCol: "camera", render: (r) => r.camera_name ?? "—" },
  { header: "ground truth", sortCol: "label", render: (r) => r.label },
  { header: "verdict", sortCol: "decision", render: (r) => r.decision },
  {
    header: "correctness",
    sortCol: "outcome",
    cellStyle: (r) => ({ color: rowTokens(r.outcome, r.decision).text }),
    render: (r) => (
      <>
        <span
          className="mr-1.5 inline-block h-2 w-2 rounded-full align-middle"
          style={{ background: outcomeTokens[r.outcome].dot }}
        />
        {correctnessLabel(r.outcome)}
      </>
    ),
  },
  { header: "tubes", sortCol: "tubes", render: (r) => r.num_tubes_kept },
  { header: "prob", sortCol: "probability", render: (r) => num(r.probability) },
];

const MONITOR_COLUMNS: Column[] = [
  {
    header: "prod prob",
    sortCol: null,
    render: (r) => num(r.recorded_probability ?? null),
  },
  {
    header: "match",
    sortCol: null,
    render: (r) =>
      r.replay_matches == null ? "—" : r.replay_matches ? "=" : "≠",
    cellStyle: (r) => ({
      color: r.replay_matches === false ? "#b91c1c" : undefined,
    }),
  },
];

export function SequenceTable({
  rows,
  selectedKey,
  onSelect,
  sort = null,
  onSort,
}: {
  rows: ResultRow[];
  selectedKey: string | null;
  onSelect: (key: string) => void;
  sort?: Sort | null;
  onSort?: (col: SortCol) => void;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const order = useMemo(() => rows.map((r) => r.key), [rows]);
  const hasProvenance = rows.some((r) => r.recorded_probability !== undefined);
  const columns = hasProvenance ? [...COLUMNS, ...MONITOR_COLUMNS] : COLUMNS;

  function move(delta: number) {
    if (!order.length) return;
    const i = Math.max(0, order.indexOf(selectedKey ?? order[0]));
    onSelect(order[Math.min(order.length - 1, Math.max(0, i + delta))]);
  }

  const arrow = (col: SortCol | null) =>
    col && sort?.col === col ? (sort.dir === "asc" ? " ▲" : " ▼") : "";

  return (
    <div
      ref={ref}
      tabIndex={0}
      onKeyDown={(e) => {
        if (e.key === "ArrowDown") {
          e.preventDefault();
          move(1);
        }
        if (e.key === "ArrowUp") {
          e.preventDefault();
          move(-1);
        }
      }}
      className="h-full overflow-auto rounded-lg border border-slate-200 outline-none"
    >
      <table className="w-full border-collapse text-sm">
        <thead className="sticky top-0 bg-white text-left text-xs uppercase tracking-wide text-slate-500">
          <tr>
            {columns.map((c) => (
              <th
                key={c.header}
                onClick={() => c.sortCol && onSort?.(c.sortCol)}
                className={`border-b border-slate-200 px-3 py-2 ${
                  c.sortCol && onSort
                    ? "cursor-pointer select-none hover:text-slate-700"
                    : ""
                }`}
              >
                {c.header}
                {arrow(c.sortCol)}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => {
            const t = rowTokens(r.outcome, r.decision);
            const sel = r.key === selectedKey;
            return (
              <tr
                key={r.key}
                onClick={() => onSelect(r.key)}
                style={{ background: t.bg }}
                className={`cursor-pointer ${sel ? "ring-2 ring-inset ring-slate-400" : ""}`}
              >
                {columns.map((c) => (
                  <td
                    key={c.header}
                    className="px-3 py-1.5 text-slate-700"
                    style={c.cellStyle?.(r)}
                  >
                    {c.render(r)}
                  </td>
                ))}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
