export interface TimelineRow {
  label: string;
  color: string;
  frames: number[];
  trigger: "decisive" | "would" | null;
}

export function TubeTimeline({
  rows,
  n,
  current,
  trigger,
}: {
  rows: TimelineRow[];
  n: number;
  current: number;
  trigger: number | null;
}) {
  if (!rows.length) return <p className="text-xs text-slate-400">no smoke tubes extracted</p>;
  const rh = 26;
  const pad = 70;
  const w = 600;
  const plot = w - pad;
  const x = (f: number) => pad + (plot * f) / Math.max(1, n);
  const barW = Math.max(2, plot / Math.max(1, n) - 1);
  return (
    <svg viewBox={`0 0 ${w} ${rows.length * rh + 16}`} className="w-full">
      {rows.map((r, ri) => (
        <g key={r.label} transform={`translate(0 ${ri * rh + 4})`}>
          <text x={0} y={16} className="fill-slate-600 text-[11px]">
            {r.label}
          </text>
          {r.frames.map((f) => (
            <rect
              key={f}
              x={x(f)}
              y={4}
              width={barW}
              height={16}
              rx={2}
              fill={r.color}
              stroke={
                r.trigger === "decisive" ? "#111827" : r.trigger === "would" ? "#6b7280" : "none"
              }
              strokeWidth={r.trigger === "decisive" ? 2 : r.trigger === "would" ? 1 : 0}
            />
          ))}
        </g>
      ))}
      {trigger != null && (
        <line
          x1={x(trigger) + 1}
          x2={x(trigger) + 1}
          y1={0}
          y2={rows.length * rh}
          stroke="#c62828"
          strokeWidth={2}
        />
      )}
      <line
        x1={x(current) + 1}
        x2={x(current) + 1}
        y1={0}
        y2={rows.length * rh}
        stroke="#111827"
        strokeWidth={2}
        strokeDasharray="4 3"
      />
    </svg>
  );
}
