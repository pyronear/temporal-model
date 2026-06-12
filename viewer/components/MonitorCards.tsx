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

export function MonitorCards({ rows }: { rows: ResultRow[] }) {
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
          {orgs.map(([org, counts]) => (
            <div
              key={org}
              className="flex items-center justify-between text-[11px]"
            >
              <span className="truncate text-slate-700">{org}</span>
              <span className="ml-2 shrink-0 tabular-nums">
                <span style={{ color: "#047857" }}>{counts.kept}</span>
                <span className="text-slate-300">/</span>
                <span className="text-slate-400">{counts.discarded}</span>
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
