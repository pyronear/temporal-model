"use client";
import { correctnessLabel, outcomeTokens } from "@/lib/correctness";
import { ALL_OUTCOMES, type Filters } from "@/lib/filters";
import type { Decision, Label, Outcome } from "@/lib/types";

function Select<T extends string>({
  label,
  value,
  options,
  onChange,
}: {
  label: string;
  value: T;
  options: { value: T; label: string }[];
  onChange: (v: T) => void;
}) {
  return (
    <label className="flex items-center gap-1.5 text-xs text-slate-500">
      {label}
      <span className="relative">
        <select
          className="cursor-pointer appearance-none rounded-lg border border-slate-200 bg-white py-1 pl-2.5 pr-7 text-sm text-slate-800 shadow-sm transition-colors hover:border-slate-300 focus:border-slate-400 focus:outline-none focus:ring-2 focus:ring-slate-200"
          value={value}
          onChange={(e) => onChange(e.target.value as T)}
        >
          {options.map((o) => (
            <option key={o.value} value={o.value}>
              {o.label}
            </option>
          ))}
        </select>
        <span className="pointer-events-none absolute inset-y-0 right-2 flex items-center text-[10px] text-slate-400">
          ▾
        </span>
      </span>
    </label>
  );
}

export function FilterBar({
  filters,
  cameras,
  organizations = [],
  onChange,
  shownCount,
  totalCount,
  monitorMode = false,
}: {
  filters: Filters;
  cameras: string[];
  organizations?: string[];
  onChange: (f: Filters) => void;
  shownCount: number;
  totalCount: number;
  monitorMode?: boolean;
}) {
  const toggleOutcome = (o: Outcome) => {
    const on = filters.outcomes.includes(o);
    const outcomes = on
      ? filters.outcomes.filter((x) => x !== o)
      : [...filters.outcomes, o];
    onChange({ ...filters, outcomes });
  };

  return (
    <div className="mb-2 flex flex-wrap items-center gap-x-4 gap-y-2 rounded-lg border border-slate-200 bg-white px-3 py-2">
      <span className="text-xs font-medium text-slate-600">
        {shownCount}/{totalCount} sequences
      </span>

      {!monitorMode && (
        <div className="flex flex-wrap items-center gap-1.5">
          {ALL_OUTCOMES.map((o) => {
            const on = filters.outcomes.includes(o);
            const t = outcomeTokens[o];
            return (
              <button
                key={o}
                onClick={() => toggleOutcome(o)}
                aria-pressed={on}
                className={`flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 text-xs ${
                  on ? "border-slate-300" : "border-slate-200 opacity-40"
                }`}
                style={on ? { background: t.bg, color: t.text } : undefined}
              >
                <span
                  className="inline-block h-2 w-2 rounded-full"
                  style={{ background: t.dot }}
                />
                {correctnessLabel(o)}
              </button>
            );
          })}
        </div>
      )}

      {!monitorMode && (
        <Select<"all" | Label>
          label="GT"
          value={filters.label}
          onChange={(v) => onChange({ ...filters, label: v })}
          options={[
            { value: "all", label: "all" },
            { value: "smoke", label: "smoke" },
            { value: "fp", label: "fp" },
            { value: "unknown", label: "unknown" },
          ]}
        />
      )}
      {monitorMode && organizations.length > 0 && (
        <Select<string>
          label="organization"
          value={filters.organization}
          onChange={(v) =>
            onChange({ ...filters, organization: v, camera: "all" })
          }
          options={[
            { value: "all", label: "all" },
            ...organizations.map((o) => ({ value: o, label: o })),
          ]}
        />
      )}
      <Select<"all" | Decision>
        label="verdict"
        value={filters.verdict}
        onChange={(v) => onChange({ ...filters, verdict: v })}
        options={[
          { value: "all", label: "all" },
          { value: "keep", label: "keep" },
          { value: "discard", label: "discard" },
        ]}
      />
      {cameras.length > 0 && (
        <Select<string>
          label="camera"
          value={filters.camera}
          onChange={(v) => onChange({ ...filters, camera: v })}
          options={[
            { value: "all", label: "all" },
            ...cameras.map((c) => ({ value: c, label: c })),
          ]}
        />
      )}
    </div>
  );
}
