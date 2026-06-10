import { describe, expect, it } from "vitest";
import {
  frameBboxesByInputIndex,
  processedToInputIndex,
  triggeringTubeIds,
  tubeInputBoxes,
} from "@/lib/details";
import type { BboxTubeDetails, KeptTube } from "@/lib/types";

describe("processedToInputIndex", () => {
  it("identity with no padding", () => {
    expect(processedToInputIndex(5, [])).toBe(5);
  });
  it("maps real slots, null for synthetic", () => {
    const padded = [0, 3];
    expect(processedToInputIndex(1, padded)).toBe(0);
    expect(processedToInputIndex(2, padded)).toBe(1);
    expect(processedToInputIndex(0, padded)).toBeNull();
    expect(processedToInputIndex(3, padded)).toBeNull();
  });
});

describe("frameBboxesByInputIndex", () => {
  it("groups kept-tube boxes by input frame, skipping null bbox", () => {
    const d = {
      preprocessing: { padded_frame_indices: [] },
      tubes: {
        kept: [
          {
            tube_id: 0,
            entries: [
              {
                frame_idx: 0,
                bbox: [0.5, 0.5, 0.1, 0.1],
                confidence: 0.7,
                is_gap: false,
              },
              { frame_idx: 1, bbox: null, confidence: null, is_gap: true },
            ],
          },
        ],
      },
    } as unknown as BboxTubeDetails;
    const out = frameBboxesByInputIndex(d);
    expect(out.get(0)).toEqual([
      { bbox: [0.5, 0.5, 0.1, 0.1], confidence: 0.7, tubeId: 0 },
    ]);
    expect(out.has(1)).toBe(false);
  });
});

describe("triggeringTubeIds", () => {
  it("logistic uses probability", () => {
    const d = {
      decision: { aggregation: "logistic", threshold: 0.5 },
      tubes: {
        kept: [
          { tube_id: 0, probability: 0.9 },
          { tube_id: 2, probability: 0.03 },
        ],
      },
    } as unknown as BboxTubeDetails;
    expect([...triggeringTubeIds(d)]).toEqual([0]);
  });
  it("max_logit uses logit", () => {
    const d = {
      decision: { aggregation: "max_logit", threshold: 1.0 },
      tubes: {
        kept: [
          { tube_id: 0, logit: 2.5 },
          { tube_id: 1, logit: 0.2 },
        ],
      },
    } as unknown as BboxTubeDetails;
    expect([...triggeringTubeIds(d)]).toEqual([0]);
  });
});

describe("tubeInputBoxes", () => {
  it("returns input-index boxes for real entries", () => {
    const tube = {
      entries: [
        {
          frame_idx: 0,
          bbox: [0.5, 0.5, 0.1, 0.1],
          confidence: 0.7,
          is_gap: false,
        },
        { frame_idx: 1, bbox: null, confidence: null, is_gap: true },
      ],
    } as unknown as KeptTube;
    expect(tubeInputBoxes(tube, [])).toEqual([
      { inputIdx: 0, bbox: [0.5, 0.5, 0.1, 0.1], confidence: 0.7 },
    ]);
  });
});
