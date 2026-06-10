import { describe, expect, it } from "vitest";
import { applyFilters, cameraOptions, defaultFilters } from "@/lib/filters";
import type { ResultRow } from "@/lib/types";

const r = (p: Partial<ResultRow>): ResultRow => ({
  key: "k", source: "s", label: "smoke", decision: "keep", outcome: "kept-smoke",
  score: 1, probability: 0.9, num_tubes_kept: 1, trigger_frame_index: 0,
  organization_name: null, camera_name: "cam-1", started_at: null, ...p,
});

const rows: ResultRow[] = [
  r({ key: "a", outcome: "kept-smoke", label: "smoke", decision: "keep", camera_name: "cam-1" }),
  r({ key: "b", outcome: "kept-fp", label: "fp", decision: "keep", camera_name: "cam-2" }),
  r({ key: "c", outcome: "discarded-smoke", label: "smoke", decision: "discard", camera_name: "cam-1" }),
  r({ key: "d", outcome: "n/a", label: "unknown", decision: "discard", camera_name: null }),
];

describe("applyFilters", () => {
  it("default keeps everything", () => {
    expect(applyFilters(rows, defaultFilters()).map((x) => x.key)).toEqual(["a", "b", "c", "d"]);
  });
  it("keeps only the selected correctness classes", () => {
    const f = { ...defaultFilters(), outcomes: ["kept-fp", "discarded-smoke"] as const };
    expect(applyFilters(rows, { ...f, outcomes: [...f.outcomes] }).map((x) => x.key)).toEqual(["b", "c"]);
  });
  it("filters by ground truth, verdict, camera", () => {
    expect(applyFilters(rows, { ...defaultFilters(), label: "smoke" }).map((x) => x.key)).toEqual(["a", "c"]);
    expect(applyFilters(rows, { ...defaultFilters(), verdict: "discard" }).map((x) => x.key)).toEqual(["c", "d"]);
    expect(applyFilters(rows, { ...defaultFilters(), camera: "cam-2" }).map((x) => x.key)).toEqual(["b"]);
  });
});

describe("cameraOptions", () => {
  it("returns sorted distinct non-null cameras", () => {
    expect(cameraOptions(rows)).toEqual(["cam-1", "cam-2"]);
  });
});
