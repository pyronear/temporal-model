"use client";

export function SourceSelect({
  sources,
  value,
  onChange,
}: {
  sources: string[];
  value: string;
  onChange: (s: string) => void;
}) {
  return (
    <div className="space-y-1">
      <label className="text-[11px] uppercase tracking-wide text-slate-500">source</label>
      <select
        className="w-full rounded-md border border-slate-300 bg-white px-2 py-1.5 text-sm"
        value={value}
        onChange={(e) => onChange(e.target.value)}
      >
        {sources.map((s) => (
          <option key={s} value={s}>
            {s}
          </option>
        ))}
      </select>
    </div>
  );
}
