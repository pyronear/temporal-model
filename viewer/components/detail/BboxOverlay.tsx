import type { TriggerState } from "@/lib/details";

export interface OverlayBox {
  bbox: [number, number, number, number];
  color: string;
  trigger: TriggerState;
  confidence: number | null;
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
        const sw = b.trigger === "decisive" ? 3 : b.trigger === "would" ? 2 : 1.5;
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
            vectorEffect="non-scaling-stroke"
          />
        );
      })}
    </svg>
  );
}
