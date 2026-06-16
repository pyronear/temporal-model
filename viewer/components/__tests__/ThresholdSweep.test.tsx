import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { ThresholdSweep } from "@/components/ThresholdSweep";
import type { ResultRow } from "@/lib/types";

const row = (probability: number | null): ResultRow => ({
  key: `k${probability}`,
  source: "pyro-annotator",
  label: "unknown",
  decision: "keep",
  outcome: "n/a",
  score: probability,
  probability,
  triage_score: probability,
  triage_bucket: "review",
  num_tubes_kept: 1,
  trigger_frame_index: 0,
  organization_name: "adf",
  camera_name: null,
  started_at: null,
});

// probabilities: 0.05, 0.30, 0.40, 0.60, 0.95  (n=5)
const rows = [row(0.05), row(0.3), row(0.4), row(0.6), row(0.95)];

describe("ThresholdSweep", () => {
  it("counts To Review (prob >= threshold) at each standard threshold", () => {
    render(<ThresholdSweep rows={rows} current={0.35} />);
    // threshold 0.35 -> review = 0.40, 0.60, 0.95 = 3, unlabel = 2
    const row035 = screen.getByText("0.35").closest("tr")!;
    expect(within(row035).getByText("3")).toBeInTheDocument();
    expect(within(row035).getByText("2")).toBeInTheDocument();
    expect(within(row035).getByText("60%")).toBeInTheDocument();
  });

  it("highlights the standard threshold nearest the current slider value", () => {
    render(<ThresholdSweep rows={rows} current={0.34} />);
    const row035 = screen.getByText("0.35").closest("tr")!;
    expect(row035).toHaveAttribute("aria-current", "true");
    const row070 = screen.getByText("0.70").closest("tr")!;
    expect(row070).not.toHaveAttribute("aria-current");
  });

  it("ignores null probabilities and renders without crashing when empty", () => {
    render(<ThresholdSweep rows={[]} current={0.35} />);
    expect(screen.getByText("Threshold sweep")).toBeInTheDocument();
  });
});
