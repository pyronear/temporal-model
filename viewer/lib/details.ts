import type { BboxTubeDetails, KeptTube } from "@/lib/types";

export function processedToInputIndex(
  frameIdx: number,
  padded: number[],
): number | null {
  if (padded.includes(frameIdx)) return null;
  return frameIdx - padded.filter((p) => p < frameIdx).length;
}

export interface FrameBox {
  bbox: [number, number, number, number];
  confidence: number | null;
  tubeId: number;
}

export function frameBboxesByInputIndex(
  details: BboxTubeDetails | null,
): Map<number, FrameBox[]> {
  const padded = details?.preprocessing?.padded_frame_indices ?? [];
  const out = new Map<number, FrameBox[]>();
  for (const tube of details?.tubes?.kept ?? []) {
    for (const e of tube.entries) {
      if (e.bbox == null) continue;
      const inp = processedToInputIndex(e.frame_idx, padded);
      if (inp == null) continue;
      const list = out.get(inp) ?? [];
      list.push({
        bbox: e.bbox,
        confidence: e.confidence,
        tubeId: tube.tube_id,
      });
      out.set(inp, list);
    }
  }
  return out;
}

export interface TubeBox {
  inputIdx: number;
  bbox: [number, number, number, number];
  confidence: number | null;
}

export function tubeInputBoxes(tube: KeptTube, padded: number[]): TubeBox[] {
  const boxes: TubeBox[] = [];
  for (const e of tube.entries) {
    if (e.bbox == null) continue;
    const inp = processedToInputIndex(e.frame_idx, padded);
    if (inp != null)
      boxes.push({ inputIdx: inp, bbox: e.bbox, confidence: e.confidence });
  }
  return boxes;
}

export function triggeringTubeIds(
  details: BboxTubeDetails | null,
): Set<number> {
  const dec = details?.decision;
  if (!dec || dec.threshold == null) return new Set();
  const useProb = dec.aggregation === "logistic";
  const ids = new Set<number>();
  for (const t of details?.tubes?.kept ?? []) {
    const v = useProb ? t.probability : t.logit;
    if (v != null && v >= dec.threshold) ids.add(t.tube_id);
  }
  return ids;
}

export type TriggerState = "decisive" | "would" | null;

export function triggerState(
  tubeId: number | null,
  triggerTubeId: number | null,
  wouldIds: Set<number>,
): TriggerState {
  if (tubeId != null && tubeId === triggerTubeId) return "decisive";
  return tubeId != null && wouldIds.has(tubeId) ? "would" : null;
}
