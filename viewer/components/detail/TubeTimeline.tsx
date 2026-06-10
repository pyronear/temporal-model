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

  const rh = 32;
  const gutter = 56;
  const w = 660;
  const axisH = 18;
  const plot = w - gutter;
  const trackH = 14;
  const x = (f: number) => gutter + (plot * f) / Math.max(1, n);
  const barW = Math.max(2, plot / Math.max(1, n) - 1.5);
  const bodyH = rows.length * rh;
  const step = Math.max(1, Math.ceil(n / 10));
  const ticks = Array.from({ length: n }, (_, f) => f).filter((f) => f % step === 0);

  return (
    <svg viewBox={`0 0 ${w} ${bodyH + axisH}`} className="w-full">
      {/* current-frame band */}
      <rect x={x(current)} y={0} width={barW} height={bodyH} fill="#0f172a" opacity={0.06} />

      {rows.map((r, ri) => {
        const y = ri * rh;
        const stroke =
          r.trigger === "decisive" ? "#0f172a" : r.trigger === "would" ? "#94a3b8" : "none";
        const sw = r.trigger === "decisive" ? 1.5 : r.trigger === "would" ? 1 : 0;
        return (
          <g key={r.label}>
            <text x={0} y={y + 17} fontSize={12} fontWeight={600} fill={r.color}>
              {r.label}
            </text>
            {/* emoji-only trigger marker (full words live next to the tube image) */}
            {r.trigger && (
              <text
                x={28}
                y={y + 17}
                fontSize={12}
                opacity={r.trigger === "decisive" ? 1 : 0.4}
              >
                ⚡
              </text>
            )}
            {/* full-range track so absent frames read as gaps */}
            <rect x={gutter} y={y + 6} width={plot} height={trackH} rx={3} fill="#f1f5f9" />
            {r.frames.map((f) => (
              <rect
                key={f}
                x={x(f)}
                y={y + 6}
                width={barW}
                height={trackH}
                rx={2}
                fill={r.color}
                stroke={stroke}
                strokeWidth={sw}
              />
            ))}
          </g>
        );
      })}

      {trigger != null && (
        <line
          x1={x(trigger) + barW / 2}
          x2={x(trigger) + barW / 2}
          y1={0}
          y2={bodyH}
          stroke="#e11d48"
          strokeWidth={2}
        />
      )}
      <line
        x1={x(current) + barW / 2}
        x2={x(current) + barW / 2}
        y1={0}
        y2={bodyH}
        stroke="#0f172a"
        strokeWidth={1.5}
        strokeDasharray="3 3"
      />

      {/* frame axis */}
      {ticks.map((f) => (
        <g key={f}>
          <line
            x1={x(f) + barW / 2}
            x2={x(f) + barW / 2}
            y1={bodyH}
            y2={bodyH + 4}
            stroke="#cbd5e1"
          />
          <text x={x(f) + barW / 2} y={bodyH + 15} textAnchor="middle" fontSize={9} fill="#94a3b8">
            {f}
          </text>
        </g>
      ))}
    </svg>
  );
}
