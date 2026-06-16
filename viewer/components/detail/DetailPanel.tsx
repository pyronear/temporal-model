"use client";
import { useMemo, useState } from "react";
import type { OverlayBox } from "@/components/detail/BboxOverlay";
import { FrameViewer } from "@/components/detail/FrameViewer";
import { TubeCrops } from "@/components/detail/TubeCrops";
import {
  TubeTimeline,
  type TimelineRow,
} from "@/components/detail/TubeTimeline";
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
  "#1f77b4",
  "#ff7f0e",
  "#2ca02c",
  "#d62728",
  "#9467bd",
  "#8c564b",
  "#e377c2",
  "#7f7f7f",
  "#bcbd22",
  "#17becf",
];
const tubeColor = (id: number) => PALETTE[id % PALETTE.length];

function Stat({
  label,
  value,
  hint,
  color,
}: {
  label: string;
  value: string;
  hint: string;
  color?: string;
}) {
  return (
    <div
      className="flex flex-col rounded-xl border border-slate-200 bg-white p-2.5"
      title={hint}
    >
      <div className="text-[10px] font-medium uppercase tracking-wide text-slate-500">
        {label}
      </div>
      <div
        className="mt-0.5 text-sm font-semibold"
        style={color ? { color } : undefined}
      >
        {value}
      </div>
      <div className="mt-1 text-[10px] leading-snug text-slate-400">{hint}</div>
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

  const boxesByFrame = (i: number): OverlayBox[] => {
    const active = bbmap.get(i) ?? [];
    const activeIds = new Set(active.map((b) => b.tubeId));
    const out: OverlayBox[] = active.map((b) => ({
      bbox: b.bbox,
      color: tubeColor(b.tubeId),
      trigger: triggerState(b.tubeId, triggerTubeId, wouldIds),
      confidence: b.confidence,
    }));
    // For tubes with no detection at this frame, show their stabilized crop
    // window as a dashed box (where the crop is / will be).
    for (const t of kept) {
      if (!activeIds.has(t.tube_id) && t.stabilized_window) {
        out.push({
          bbox: t.stabilized_window,
          color: tubeColor(t.tube_id),
          trigger: triggerState(t.tube_id, triggerTubeId, wouldIds),
          confidence: null,
          dashed: true,
        });
      }
    }
    return out;
  };

  const activeBoxByTube = (frame: number) => {
    const m = new Map<number, [number, number, number, number]>();
    for (const t of kept) {
      const hit = tubeInputBoxes(t, padded).find((b) => b.inputIdx === frame);
      if (hit) m.set(t.tube_id, hit.bbox);
    }
    return m;
  };

  const triggerByTube = new Map(
    kept.map(
      (t) =>
        [t.tube_id, triggerState(t.tube_id, triggerTubeId, wouldIds)] as const,
    ),
  );
  const colorByTube = new Map(
    kept.map((t) => [t.tube_id, tubeColor(t.tube_id)] as const),
  );

  // The parent owns the frame index so the viewer, timeline, and crops stay in sync.
  const [i, setI] = useState(0);
  const n = frames.length;
  const cur = n ? i % n : 0;

  return (
    <section className="flex h-full flex-col gap-3 overflow-auto p-4">
      <header className="flex items-center justify-between border-b border-slate-100 pb-2">
        <code className="text-sm font-medium text-slate-700">{row.key}</code>
      </header>
      <div className="grid grid-cols-4 gap-2 text-sm">
        <Stat
          label="verdict"
          value={row.decision}
          color={row.decision === "keep" ? "#059669" : "#475569"}
          hint="the model's keep / discard decision for this sequence"
        />
        {/* Eval-only: triage rows are unlabeled (no ground truth), monitor
            rows carry production's verdict — neither has a correctness. */}
        {row.replayed_probability === undefined &&
          row.triage_bucket === undefined && (
            <Stat
              label="correctness"
              value={correctnessLabel(row.outcome)}
              color={outcomeTokens[row.outcome].text}
              hint="the verdict vs. the ground-truth label"
            />
          )}
        {(row.replayed_probability === undefined || trig != null) && (
          <Stat
            label="trigger frame"
            value={trig == null ? "—" : String(trig)}
            hint="first frame the model fired (— if discarded)"
          />
        )}
        <Stat
          label="probability"
          value={row.probability == null ? "—" : row.probability.toFixed(3)}
          hint="max calibrated tube probability driving the decision"
        />
        {row.replayed_probability !== undefined && (
          <>
            <Stat
              label="replay prob"
              value={
                row.replayed_probability == null
                  ? "—"
                  : row.replayed_probability.toFixed(3)
              }
              hint="probability from the local re-run (diagnostic)"
            />
            <Stat
              label="replay"
              value={
                row.replay_matches == null
                  ? "—"
                  : row.replay_matches
                    ? "matches"
                    : "MISMATCH"
              }
              hint={`api ${row.temporal_api_version ?? "?"} · model ${row.temporal_model_version ?? "?"}`}
            />
            {row.matched_window_frames != null && (
              <Stat
                label="scored window"
                value={`first ${row.matched_window_frames} frames`}
                hint="production scored this window; the sequence kept growing afterwards"
              />
            )}
          </>
        )}
      </div>
      {n > 0 && (
        <div className="grid grid-cols-[2fr_1fr] gap-3">
          <div className="space-y-2">
            <FrameViewer
              frames={frames}
              boxesByFrame={boxesByFrame}
              i={i}
              setI={setI}
            />
            <TubeTimeline rows={timeline} n={n} current={cur} trigger={trig} />
          </div>
          <TubeCrops
            tubes={kept}
            framePath={frames[cur]}
            activeBoxByTube={activeBoxByTube(cur)}
            triggerByTube={triggerByTube}
            colorByTube={colorByTube}
          />
        </div>
      )}
    </section>
  );
}
