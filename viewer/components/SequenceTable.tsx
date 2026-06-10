"use client";
import { useMemo, useRef } from "react";
import { correctnessLabel, outcomeTokens, rowTokens } from "@/lib/correctness";
import type { ResultRow } from "@/lib/types";

const num = (v: number | null) => (v == null ? "—" : v.toFixed(3));

export function SequenceTable({
  rows,
  selectedKey,
  onSelect,
}: {
  rows: ResultRow[];
  selectedKey: string | null;
  onSelect: (key: string) => void;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const order = useMemo(() => rows.map((r) => r.key), [rows]);

  function move(delta: number) {
    if (!order.length) return;
    const i = Math.max(0, order.indexOf(selectedKey ?? order[0]));
    onSelect(order[Math.min(order.length - 1, Math.max(0, i + delta))]);
  }

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
            {["camera", "ground truth", "verdict", "correctness", "prob"].map((h) => (
              <th key={h} className="border-b border-slate-200 px-3 py-2">
                {h}
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
                <td className="px-3 py-1.5 text-slate-700">{r.camera_name ?? "—"}</td>
                <td className="px-3 py-1.5 text-slate-700">{r.label}</td>
                <td className="px-3 py-1.5 text-slate-700">{r.decision}</td>
                <td className="px-3 py-1.5" style={{ color: t.text }}>
                  <span
                    className="mr-1.5 inline-block h-2 w-2 rounded-full align-middle"
                    style={{ background: outcomeTokens[r.outcome].dot }}
                  />
                  {correctnessLabel(r.outcome)}
                </td>
                <td className="px-3 py-1.5 text-slate-600">{num(r.probability)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
