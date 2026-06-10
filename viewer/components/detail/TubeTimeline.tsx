export interface TimelineRow {
  label: string;
  color: string;
  frames: number[];
  trigger: "decisive" | "would" | null;
}

// Horizontal gutters as a fraction of the timeline width. The plot fills the
// band between them. The frame-strip controls reuse these exact percentages so
// the slider track lines up pixel-for-pixel with the tube track at any width.
export const TIMELINE_LEFT_PCT = 10; // labels (T0…) + play button
export const TIMELINE_RIGHT_PCT = 10; // frame count

// Group a tube's present frames into contiguous [start, end] runs so each run
// renders as one continuous bar instead of a row of per-frame pills.
function runs(frames: number[]): [number, number][] {
  const sorted = [...frames].sort((a, b) => a - b);
  const out: [number, number][] = [];
  for (const f of sorted) {
    const last = out[out.length - 1];
    if (last && f === last[1] + 1) last[1] = f;
    else out.push([f, f]);
  }
  return out;
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
  if (!rows.length)
    return <p className="text-xs text-slate-400">no smoke tubes extracted</p>;

  const rh = 32;
  const w = 660;
  const gutter = (w * TIMELINE_LEFT_PCT) / 100;
  const gutterR = (w * TIMELINE_RIGHT_PCT) / 100;
  const headH = 16; // top band for the trigger label
  const axisH = 18;
  const titleH = 14;
  const trackH = 14;
  const plot = w - gutter - gutterR;
  const x = (f: number) => gutter + (plot * f) / Math.max(1, n);
  const cellW = plot / Math.max(1, n);
  const center = (f: number) => x(f) + cellW / 2;
  const bodyTop = headH;
  const bodyH = rows.length * rh;
  const bodyBottom = bodyTop + bodyH;
  const step = Math.max(1, Math.ceil(n / 10));
  const ticks = Array.from({ length: n }, (_, f) => f).filter(
    (f) => f % step === 0,
  );
  const cx = center(current);
  const tx = trigger != null ? center(trigger) : null;

  return (
    <svg viewBox={`0 0 ${w} ${bodyBottom + axisH + titleH}`} className="w-full">
      {/* trigger legend (the red line itself is drawn last, on top of the bars) */}
      {tx != null && (
        <text
          x={tx}
          y={bodyTop - 4}
          textAnchor="middle"
          fontSize={10}
          fontWeight={600}
          fill="#e11d48"
        >
          ⚡ triggered
        </text>
      )}

      {rows.map((r, ri) => {
        const y = bodyTop + ri * rh;
        const trackFill = r.trigger === "decisive" ? "#fff7ed" : "#f1f5f9";
        return (
          <g key={r.label}>
            <text
              x={0}
              y={y + 17}
              fontSize={12}
              fontWeight={600}
              fill={r.color}
            >
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
            <rect
              x={gutter}
              y={y + 6}
              width={plot}
              height={trackH}
              rx={trackH / 2}
              fill={trackFill}
            />
            {/* one continuous bar per contiguous run of present frames */}
            {runs(r.frames).map(([a, b]) => (
              <rect
                key={a}
                x={x(a)}
                y={y + 6}
                width={x(b + 1) - x(a)}
                height={trackH}
                rx={trackH / 2}
                fill={r.color}
              />
            ))}
          </g>
        );
      })}

      {/* current-frame guide + caret — quiet, just enough to locate the frame */}
      <line
        x1={cx}
        x2={cx}
        y1={bodyTop}
        y2={bodyBottom}
        stroke="#cbd5e1"
        strokeWidth={1}
      />
      <polygon
        points={`${cx - 4},${bodyBottom} ${cx + 4},${bodyBottom} ${cx},${bodyBottom + 5}`}
        fill="#0f172a"
      />

      {/* trigger line + dot — drawn last so it sits on top of the bars */}
      {tx != null && (
        <g>
          <line
            x1={tx}
            x2={tx}
            y1={bodyTop}
            y2={bodyBottom}
            stroke="#e11d48"
            strokeWidth={2}
          />
          <circle cx={tx} cy={bodyTop} r={3} fill="#e11d48" />
        </g>
      )}

      {/* frame axis */}
      {ticks.map((f) => (
        <g key={f}>
          <line
            x1={center(f)}
            x2={center(f)}
            y1={bodyBottom}
            y2={bodyBottom + 4}
            stroke="#cbd5e1"
          />
          <text
            x={center(f)}
            y={bodyBottom + 15}
            textAnchor="middle"
            fontSize={9}
            fill="#94a3b8"
          >
            {f}
          </text>
        </g>
      ))}
      {/* trigger frame number, in red, under the trigger line */}
      {tx != null && trigger != null && (
        <text
          x={tx}
          y={bodyBottom + 15}
          textAnchor="middle"
          fontSize={9}
          fontWeight={700}
          fill="#e11d48"
        >
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
