import { describe, expect, it } from "vitest";
import { applyThreshold, computeOutcome, performanceSummary } from "@/lib/outcomes";
import type { ResultRow } from "@/lib/types";

function row(p: Partial<ResultRow>): ResultRow {
  return {
    key: "k",
    source: "s",
    label: "smoke",
    decision: "keep",
    outcome: "kept-smoke",
    score: 1,
    probability: 0.9,
    num_tubes_kept: 1,
    trigger_frame_index: 0,
    organization_name: null,
    camera_name: null,
    started_at: null,
    ...p,
  };
}

describe("computeOutcome", () => {
  it("maps decision+label", () => {
    expect(computeOutcome("keep", "smoke")).toBe("kept-smoke");
    expect(computeOutcome("discard", "smoke")).toBe("discarded-smoke");
    expect(computeOutcome("keep", "fp")).toBe("kept-fp");
    expect(computeOutcome("discard", "fp")).toBe("discarded-fp");
    expect(computeOutcome("keep", "unknown")).toBe("n/a");
  });
});

describe("applyThreshold", () => {
  it("re-decides keep iff probability >= thr (null -> discard)", () => {
    const rows = [
      row({ label: "smoke", probability: 0.9 }),
      row({ label: "smoke", probability: 0.2 }),
      row({ label: "fp", probability: 0.8 }),
      row({ label: "smoke", probability: null }),
    ];
    const out = applyThreshold(rows, 0.5);
    expect(out.map((r) => r.decision)).toEqual(["keep", "discard", "keep", "discard"]);
    expect(out.map((r) => r.outcome)).toEqual([
      "kept-smoke",
      "discarded-smoke",
      "kept-fp",
      "discarded-smoke",
    ]);
    expect(rows[1].decision).toBe("keep"); // input not mutated
  });
});

describe("performanceSummary", () => {
  it("computes recall / specificity / precision over labeled rows", () => {
    const rows = [
      row({ label: "smoke", outcome: "kept-smoke" }),
      row({ label: "smoke", outcome: "discarded-smoke" }),
      row({ label: "fp", outcome: "discarded-fp" }),
      row({ label: "fp", outcome: "kept-fp" }),
      row({ label: "unknown", outcome: "n/a" }),
    ];
    const s = performanceSummary(rows);
    expect(s.nSmoke).toBe(2);
    expect(s.nFp).toBe(2);
    expect(s.recall).toBe(0.5);
    expect(s.specificity).toBe(0.5);
    expect(s.precision).toBe(0.5);
  });
});
