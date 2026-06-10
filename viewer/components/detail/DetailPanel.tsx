"use client";
import { useMemo, useState } from "react";
import type { OverlayBox } from "@/components/detail/BboxOverlay";
import { FrameViewer } from "@/components/detail/FrameViewer";
import { TubeCrops } from "@/components/detail/TubeCrops";
import { TubeTimeline, type TimelineRow } from "@/components/detail/TubeTimeline";
import { correctnessLabel, outcomeTokens } from "@/lib/correctness";
import {
  frameBboxesByInputIndex,
  processedToInputIndex,
  triggerState,
  triggeringTubeIds,
  tubeInputBoxes,
} from "@/lib/details";
import type { BboxTubeDetails, ResultRow, SequenceView } from "@/lib/types";

const PALETTE = [
  "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
  "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
];
const tubeColor = (id: number) => PALETTE[id % PALETTE.length];

function Stat({ label, value, color }: { label: string; value: string; color?: string }) {
  return (
    <div className="rounded-lg border border-slate-200 bg-white p-2">
      <div className="text-[10px] uppercase tracking-wide text-slate-500">{label}</div>
      <div className="text-sm font-medium" style={color ? { color } : undefined}>
        {value}
      </div>
    </div>
  );
}

export function DetailPanel({
  details,
  view,
  row,
}: {
  details: BboxTubeDetails | null;
  view: SequenceView | null;
  row: ResultRow;
}) {
  const frames = view?.frames ?? [];
  const padded = details?.preprocessing?.padded_frame_indices ?? [];
  const kept = details?.tubes?.kept ?? [];
  const triggerTubeId = details?.decision?.trigger_tube_id ?? null;
  const wouldIds = useMemo(() => triggeringTubeIds(details), [details]);
  const bbmap = useMemo(() => frameBboxesByInputIndex(details), [details]);
  const trig =
    row.trigger_frame_index != null
      ? processedToInputIndex(row.trigger_frame_index, padded)
      : null;

  const timeline: TimelineRow[] = kept.map((t) => ({
    label: `T${t.tube_id}`,
    color: tubeColor(t.tube_id),
    frames: tubeInputBoxes(t, padded).map((b) => b.inputIdx),
    trigger: triggerState(t.tube_id, triggerTubeId, wouldIds),
  }));

  const boxesByFrame = (i: number): OverlayBox[] =>
    (bbmap.get(i) ?? []).map((b) => ({
      bbox: b.bbox,
      color: tubeColor(b.tubeId),
      trigger: triggerState(b.tubeId, triggerTubeId, wouldIds),
      confidence: b.confidence,
    }));

  const activeBoxByTube = (frame: number) => {
    const m = new Map<number, [number, number, number, number]>();
    for (const t of kept) {
      const hit = tubeInputBoxes(t, padded).find((b) => b.inputIdx === frame);
      if (hit) m.set(t.tube_id, hit.bbox);
    }
    return m;
  };

  const triggerByTube = new Map(
    kept.map((t) => [t.tube_id, triggerState(t.tube_id, triggerTubeId, wouldIds)] as const),
  );

  // The parent owns the frame index so the viewer, timeline, and crops stay in sync.
  const [i, setI] = useState(0);
  const n = frames.length;
  const cur = n ? i % n : 0;

  return (
    <section className="flex h-full flex-col gap-3 overflow-auto p-4">
      <header className="flex items-center justify-between">
        <h2 className="text-lg font-semibold">
          {row.decision === "keep" ? "💨 KEEP (smoke)" : "🚫 DISCARD (no smoke)"}
        </h2>
        <code className="text-xs text-slate-500">{row.key}</code>
      </header>
      <div className="grid grid-cols-4 gap-2 text-sm">
        <Stat label="verdict" value={row.decision} />
        <Stat
          label="correctness"
          value={correctnessLabel(row.outcome)}
          color={outcomeTokens[row.outcome].text}
        />
        <Stat label="trigger frame" value={trig == null ? "—" : String(trig)} />
        <Stat
          label="probability"
          value={row.probability == null ? "—" : row.probability.toFixed(3)}
        />
      </div>
      {n > 0 && (
        <div className="grid grid-cols-[2fr_1fr] gap-3">
          <div className="space-y-2">
            <FrameViewer frames={frames} boxesByFrame={boxesByFrame} i={i} setI={setI} />
            <TubeTimeline rows={timeline} n={n} current={cur} trigger={trig} />
          </div>
          <TubeCrops
            tubes={kept}
            framePath={frames[cur]}
            activeBoxByTube={activeBoxByTube(cur)}
            triggerByTube={triggerByTube}
          />
        </div>
      )}
    </section>
  );
}
