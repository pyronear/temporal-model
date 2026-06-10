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
  const headH = 16; // top band for the trigger label
  const axisH = 18;
  const titleH = 14;
  const trackH = 14;
  const plot = w - gutter;
  const x = (f: number) => gutter + (plot * f) / Math.max(1, n);
  const barW = Math.max(2, plot / Math.max(1, n) - 1.5);
  const bodyTop = headH;
  const bodyH = rows.length * rh;
  const bodyBottom = bodyTop + bodyH;
  const step = Math.max(1, Math.ceil(n / 10));
  const ticks = Array.from({ length: n }, (_, f) => f).filter((f) => f % step === 0);
  const cx = x(current) + barW / 2;
  const tx = trigger != null ? x(trigger) + barW / 2 : null;

  return (
    <svg viewBox={`0 0 ${w} ${bodyBottom + axisH + titleH}`} className="w-full">
      {/* current-frame band */}
      <rect x={x(current)} y={bodyTop} width={barW} height={bodyH} fill="#0f172a" opacity={0.06} />

      {/* trigger legend (the red line itself is drawn last, on top of the bars) */}
      {tx != null && (
        <text x={tx} y={bodyTop - 4} textAnchor="middle" fontSize={10} fontWeight={600} fill="#e11d48">
          ⚡ triggered
        </text>
      )}

      {rows.map((r, ri) => {
        const y = bodyTop + ri * rh;
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
              <text x={28} y={y + 17} fontSize={12} opacity={r.trigger === "decisive" ? 1 : 0.4}>
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

      {/* current-frame line */}
      <line
        x1={cx}
        x2={cx}
        y1={bodyTop}
        y2={bodyBottom}
        stroke="#0f172a"
        strokeWidth={1.5}
        strokeDasharray="3 3"
      />

      {/* trigger line — drawn last so it sits on top of the bars */}
      {tx != null && (
        <line x1={tx} x2={tx} y1={bodyTop} y2={bodyBottom} stroke="#e11d48" strokeWidth={2} />
      )}

      {/* frame axis */}
      {ticks.map((f) => (
        <g key={f}>
          <line x1={x(f) + barW / 2} x2={x(f) + barW / 2} y1={bodyBottom} y2={bodyBottom + 4} stroke="#cbd5e1" />
          <text x={x(f) + barW / 2} y={bodyBottom + 15} textAnchor="middle" fontSize={9} fill="#94a3b8">
            {f}
          </text>
        </g>
      ))}
      {/* trigger frame number, in red, under the trigger line */}
      {tx != null && trigger != null && (
        <text x={tx} y={bodyBottom + 15} textAnchor="middle" fontSize={9} fontWeight={700} fill="#e11d48">
          {trigger}
        </text>
      )}
      <text
        x={gutter + plot / 2}
        y={bodyBottom + axisH + titleH - 1}
        textAnchor="middle"
        fontSize={10}
        fill="#64748b"
      >
        frames
      </text>
    </svg>
  );
}
