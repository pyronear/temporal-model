import { useMemo } from "react";
import type { ResultRow } from "@/lib/types";

/**
 * Left-rail cards for the triage reporting tree: the Review / Unlabel split, a
 * per-organization breakdown, and a triage-specific explainer. Mirrors
 * MonitorCards structurally; "keep" maps to Review (>= threshold, needs a human)
 * and "discard" to Unlabel (< threshold, auto false-positive).
 */
export function TriageCards({
  rows,
  threshold,
  selectedOrganization = "all",
  onSelectOrganization,
}: {
  rows: ResultRow[];
  threshold?: number | null;
  selectedOrganization?: string;
  onSelectOrganization?: (org: string) => void;
}) {
  // Memoized on rows so re-renders for the live header value (threshold) don't
  // re-scan all rows each tick.
  const { review, unlabel, orgs } = useMemo(() => {
    const review = rows.filter((r) => r.decision === "keep").length;
    // Per-org breakdown (review/unlabel), sorted by total desc.
    const orgMap = new Map<string, { review: number; unlabel: number }>();
    for (const r of rows) {
      const org = r.organization_name ?? "—";
      const entry = orgMap.get(org) ?? { review: 0, unlabel: 0 };
      if (r.decision === "keep") entry.review++;
      else entry.unlabel++;
      orgMap.set(org, entry);
    }
    const orgs = [...orgMap.entries()].sort(
      (a, b) => b[1].review + b[1].unlabel - (a[1].review + a[1].unlabel),
    );
    return { review, unlabel: rows.length - review, orgs };
  }, [rows]);

  const pctOf = (n: number) =>
    rows.length ? Math.round((n / rows.length) * 100) : 0;

  return (
    <div className="flex flex-col gap-3">
      <div className="rounded-xl border border-slate-200 bg-white p-2.5">
        <div className="text-[9px] font-medium uppercase tracking-tight text-slate-500">
          Backlog
        </div>
        <div className="text-sm font-medium tabular-nums text-slate-700">
          {rows.length} sequences
          {threshold != null && (
            <span className="ml-1 text-[11px] font-normal text-slate-400">
              · threshold {threshold}
            </span>
          )}
        </div>
      </div>

      <div className="grid grid-cols-2 gap-2">
        <div className="flex flex-col rounded-xl border border-slate-200 bg-white p-2.5">
          <div className="flex h-4 items-center whitespace-nowrap text-[9px] font-medium uppercase tracking-tight text-slate-500">
            To Review
          </div>
          <div
            className="text-base font-semibold tabular-nums"
            style={{ color: "#047857" }}
          >
            {review}
            {rows.length > 0 && (
              <span className="ml-1 text-[11px] font-normal text-slate-400">
                {pctOf(review)}%
              </span>
            )}
          </div>
          <div className="mt-1.5 h-1 overflow-hidden rounded-full bg-slate-100">
            <div
              className="h-full rounded-full"
              style={{
                width: rows.length ? `${(review / rows.length) * 100}%` : "0%",
                background: "#047857",
              }}
            />
          </div>
        </div>
        <div className="flex flex-col rounded-xl border border-slate-200 bg-white p-2.5">
          <div className="flex h-4 items-center whitespace-nowrap text-[9px] font-medium uppercase tracking-tight text-slate-500">
            Unlabel
          </div>
          <div className="text-base font-semibold tabular-nums text-slate-500">
            {unlabel}
            {rows.length > 0 && (
              <span className="ml-1 text-[11px] font-normal text-slate-400">
                {pctOf(unlabel)}%
              </span>
            )}
          </div>
          <div className="mt-1.5 h-1 overflow-hidden rounded-full bg-slate-100">
            <div
              className="h-full rounded-full bg-slate-400"
              style={{
                width: rows.length
                  ? `${(unlabel / rows.length) * 100}%`
                  : "0%",
              }}
            />
          </div>
        </div>
      </div>

      {orgs.length > 0 && (
        <div className="flex flex-col gap-0.5 rounded-xl border border-slate-200 bg-white p-2.5">
          <div className="mb-1 text-[9px] font-medium uppercase tracking-tight text-slate-500">
            By organization
          </div>
          {orgs.map(([org, counts]) => {
            const total = counts.review + counts.unlabel;
            const pct = total ? Math.round((counts.review / total) * 100) : 0;
            const clickable = onSelectOrganization && org !== "—";
            const active = org === selectedOrganization;
            return (
              <button
                key={org}
                disabled={!clickable}
                aria-pressed={active}
                onClick={() =>
                  onSelectOrganization?.(active ? "all" : org)
                }
                className={`-mx-2.5 flex items-center justify-between px-2.5 py-0.5 text-left text-[11px] ${
                  active ? "bg-slate-100" : ""
                } ${clickable ? "cursor-pointer hover:bg-slate-50" : ""}`}
              >
                <span className="truncate text-slate-700">{org}</span>
                <span className="ml-2 shrink-0 tabular-nums">
                  <span style={{ color: "#047857" }}>{counts.review}</span>
                  <span className="text-slate-300">/</span>
                  <span className="text-slate-400">{counts.unlabel}</span>
                  <span className="ml-1 text-slate-400">({pct}%)</span>
                </span>
              </button>
            );
          })}
        </div>
      )}

      <details
        open
        className="rounded-xl border border-slate-200 bg-white p-2.5"
      >
        <summary className="cursor-pointer text-[9px] font-medium uppercase tracking-tight text-slate-500">
          How this view works
        </summary>
        <div className="mt-1.5 flex flex-col gap-1.5 text-[11px] leading-snug text-slate-600">
          <p>
            These are unannotated sequences pulled from the pyro-annotator
            (read-only), each scored by the temporal model.
          </p>
          <p>
            <span style={{ color: "#047857" }}>To review</span> = score ≥
            threshold: worth a human&apos;s eyes.{" "}
            <span className="text-slate-500">Unlabel</span>{" "}
            = score &lt; threshold: almost certainly a false positive, queued to
            be marked <em>unlabeled</em> in the annotator.
          </p>
          <p>
            Verdicts are the model&apos;s; the per-tube boxes and crops on the
            right show why each sequence scored as it did.
          </p>
        </div>
      </details>
    </div>
  );
}
