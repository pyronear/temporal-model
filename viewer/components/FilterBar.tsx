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
      <select
        className="rounded-md border border-slate-300 bg-white px-2 py-1 text-sm text-slate-800"
        value={value}
        onChange={(e) => onChange(e.target.value as T)}
      >
        {options.map((o) => (
          <option key={o.value} value={o.value}>
            {o.label}
          </option>
        ))}
      </select>
    </label>
  );
}

export function FilterBar({
  filters,
  cameras,
  onChange,
  shownCount,
  totalCount,
}: {
  filters: Filters;
  cameras: string[];
  onChange: (f: Filters) => void;
  shownCount: number;
  totalCount: number;
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
          options={[{ value: "all", label: "all" }, ...cameras.map((c) => ({ value: c, label: c }))]}
        />
      )}
    </div>
  );
}
