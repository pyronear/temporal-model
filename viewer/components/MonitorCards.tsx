import type { ResultRow } from "@/lib/types";

export function MonitorCards({ rows }: { rows: ResultRow[] }) {
  const kept = rows.filter((r) => r.decision === "keep").length;
  const discarded = rows.filter((r) => r.decision === "discard").length;

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
