"use client";

// Human-readable blurb per known source; unknown sources just show their name.
const SOURCE_INFO: Record<string, string> = {
  "pyro-annotator": "human-labeled platform sequences (smoke / fp / unknown)",
  val: "held-out validation split",
  train: "training split",
};

const blurb = (s: string) => SOURCE_INFO[s] ?? "";

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
          <option key={s} value={s} title={blurb(s)}>
            {blurb(s) ? `${s} — ${blurb(s)}` : s}
          </option>
        ))}
      </select>
      {blurb(value) && <p className="text-[11px] leading-snug text-slate-400">{blurb(value)}</p>}
    </div>
  );
}
