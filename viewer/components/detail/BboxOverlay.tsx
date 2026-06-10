import type { TriggerState } from "@/lib/details";

export interface OverlayBox {
  bbox: [number, number, number, number];
  color: string;
  trigger: TriggerState;
  confidence: number | null;
  /** dashed = the tube's stabilized crop window on a frame where it has no detection */
  dashed?: boolean;
}

/** SVG overlay in normalized [0..1] space (viewBox 0 0 1 1), sits over the frame. */
export function BboxOverlay({ boxes }: { boxes: OverlayBox[] }) {
  return (
    <svg
      viewBox="0 0 1 1"
      preserveAspectRatio="none"
      className="pointer-events-none absolute inset-0 h-full w-full"
    >
      {boxes.map((b, i) => {
        const [cx, cy, w, h] = b.bbox;
        const sw = b.dashed ? 1.5 : b.trigger === "decisive" ? 3 : b.trigger === "would" ? 2 : 1.5;
        return (
          <rect
            key={i}
            x={cx - w / 2}
            y={cy - h / 2}
            width={w}
            height={h}
            fill="none"
            stroke={b.color}
            strokeWidth={sw}
            strokeDasharray={b.dashed ? "5 4" : undefined}
            opacity={b.dashed ? 0.6 : 1}
            vectorEffect="non-scaling-stroke"
          />
        );
      })}
    </svg>
  );
}
