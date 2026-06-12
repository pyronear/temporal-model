import type { ResultRow } from "@/lib/types";

/** "2026-06-12" or "2026-06-10 → 2026-06-12"; null without any started_at. */
function dayRange(rows: ResultRow[]): string | null {
  const days = rows
    .map((r) => r.started_at?.slice(0, 10))
    .filter((d): d is string => d != null)
    .sort();
  if (!days.length) return null;
  const [first, last] = [days[0], days[days.length - 1]];
  return first === last ? first : `${first} → ${last}`;
}

export function MonitorCards({
  rows,
  selectedOrganization = "all",
  onSelectOrganization,
}: {
  rows: ResultRow[];
  selectedOrganization?: string;
  onSelectOrganization?: (org: string) => void;
}) {
  const kept = rows.filter((r) => r.decision === "keep").length;
  const discarded = rows.filter((r) => r.decision === "discard").length;
  const span = dayRange(rows);

  // Per-org breakdown, sorted by total desc.
  const orgMap = new Map<string, { kept: number; discarded: number }>();
  for (const r of rows) {
    const org = r.organization_name ?? "—";
    const entry = orgMap.get(org) ?? { kept: 0, discarded: 0 };
    if (r.decision === "keep") entry.kept++;
    else entry.discarded++;
    orgMap.set(org, entry);
  }
  const orgs = [...orgMap.entries()].sort(
    (a, b) => b[1].kept + b[1].discarded - (a[1].kept + a[1].discarded),
  );

  return (
    <div className="flex flex-col gap-3">
      {span && (
        <div className="rounded-xl border border-slate-200 bg-white p-2.5">
          <div className="text-[9px] font-medium uppercase tracking-tight text-slate-500">
            Monitoring span
          </div>
          <div className="text-sm font-medium tabular-nums text-slate-700">
            {span}
          </div>
        </div>
      )}
      <div className="grid grid-cols-2 gap-2">
        <div className="flex flex-col rounded-xl border border-slate-200 bg-white p-2.5">
          <div className="flex h-4 items-center whitespace-nowrap text-[9px] font-medium uppercase tracking-tight text-slate-500">
            Kept
          </div>
          <div
            className="text-base font-semibold tabular-nums"
            style={{ color: "#047857" }}
          >
            {kept}
            {rows.length > 0 && (
              <span className="ml-1 text-[11px] font-normal text-slate-400">
                {Math.round((kept / rows.length) * 100)}%
              </span>
            )}
          </div>
          <div className="mt-1.5 h-1 overflow-hidden rounded-full bg-slate-100">
            <div
              className="h-full rounded-full"
              style={{
                width: rows.length ? `${(kept / rows.length) * 100}%` : "0%",
                background: "#047857",
              }}
            />
          </div>
        </div>
        <div className="flex flex-col rounded-xl border border-slate-200 bg-white p-2.5">
          <div className="flex h-4 items-center whitespace-nowrap text-[9px] font-medium uppercase tracking-tight text-slate-500">
            Discarded
          </div>
          <div className="text-base font-semibold tabular-nums text-slate-500">
            {discarded}
            {rows.length > 0 && (
              <span className="ml-1 text-[11px] font-normal text-slate-400">
                {Math.round((discarded / rows.length) * 100)}%
              </span>
            )}
          </div>
          <div className="mt-1.5 h-1 overflow-hidden rounded-full bg-slate-100">
            <div
              className="h-full rounded-full bg-slate-400"
              style={{
                width: rows.length
                  ? `${(discarded / rows.length) * 100}%`
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
            const total = counts.kept + counts.discarded;
            const pct = total ? Math.round((counts.kept / total) * 100) : 0;
            const clickable = onSelectOrganization && org !== "—";
            const active = org === selectedOrganization;
            return (
              <button
                key={org}
                disabled={!clickable}
                onClick={() =>
                  // clicking the active org clears the filter back to "all"
                  onSelectOrganization?.(active ? "all" : org)
                }
                className={`flex items-center justify-between rounded px-1 py-0.5 text-left text-[11px] ${
                  active ? "bg-slate-100" : ""
                } ${clickable ? "cursor-pointer hover:bg-slate-50" : ""}`}
              >
                <span className="truncate text-slate-700">{org}</span>
                <span className="ml-2 shrink-0 tabular-nums">
                  <span style={{ color: "#047857" }}>{counts.kept}</span>
                  <span className="text-slate-300">/</span>
                  <span className="text-slate-400">{counts.discarded}</span>
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
            In production, alert-api asks the temporal model to validate each
            detection sequence, but it only stores the resulting probability —
            not the <em>why</em>.
          </p>
          <p>
            This view re-runs every sequence locally through the exact api+model
            release that scored it, with verbose output, to extract the tubes,
            boxes and crops shown on the right.
          </p>
          <p>
            Verdicts and probabilities in the table are production&apos;s own;
            the re-run is only the microscope (its diagnostics live in the
            detail pane).
          </p>
        </div>
      </details>
    </div>
  );
}
