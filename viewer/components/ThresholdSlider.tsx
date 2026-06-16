"use client";

export function ThresholdSlider({
  value,
  defaultValue,
  onChange,
  onReset,
  label = "logistic threshold",
  defaultLabel = "model default",
}: {
  value: number;
  defaultValue: number;
  onChange: (v: number) => void;
  onReset: () => void;
  label?: string;
  defaultLabel?: string;
}) {
  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between">
        <label className="text-[11px] uppercase tracking-wide text-slate-500">
          {label}
        </label>
        <button
          onClick={onReset}
          className="rounded border border-slate-200 px-2 py-0.5 text-xs text-slate-600 hover:bg-slate-50"
        >
          ↺ reset
        </button>
      </div>
      <input
        type="range"
        min={0}
        max={1}
        step={0.01}
        value={value}
        aria-label={label}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full"
      />
      <div className="flex justify-between text-[11px] text-slate-400">
        <span>{value.toFixed(2)}</span>
        <span>
          {defaultLabel}: {defaultValue.toFixed(2)}
        </span>
      </div>
    </div>
  );
}
