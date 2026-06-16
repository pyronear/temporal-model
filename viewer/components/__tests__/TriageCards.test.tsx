import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { TriageCards } from "@/components/TriageCards";
import type { ResultRow } from "@/lib/types";

const r = (
  key: string,
  decision: ResultRow["decision"],
  organization_name: string | null = null,
): ResultRow => ({
  key,
  source: "pyro-annotator",
  label: "unknown",
  decision,
  outcome: "n/a",
  score: 1,
  probability: 0.9,
  triage_score: 0.9,
  triage_bucket: decision === "keep" ? "review" : "unlabeled",
  num_tubes_kept: 1,
  trigger_frame_index: 0,
  organization_name,
  camera_name: null,
  started_at: null,
});

const rows: ResultRow[] = [
  r("a", "keep", "adf"),
  r("b", "keep", "adf"),
  r("c", "discard", "adf"),
  r("d", "keep", "sis-67"),
  r("e", "discard", "sis-67"),
  r("f", "discard", null),
];

describe("TriageCards", () => {
  it("labels the buckets Review and Unlabel", () => {
    render(<TriageCards rows={rows} threshold={0.35} />);
    // "Review"/"Unlabel" appear both as card labels and in the explainer text.
    expect(screen.getAllByText("Review").length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText("Unlabel").length).toBeGreaterThanOrEqual(1);
  });

  it("shows correct review (3) and unlabel (3) counts", () => {
    render(<TriageCards rows={rows} threshold={0.35} />);
    expect(screen.getAllByText("3").length).toBeGreaterThanOrEqual(2);
  });

  it("renders the backlog total and threshold", () => {
    render(<TriageCards rows={rows} threshold={0.35} />);
    expect(screen.getByText(/6 sequences/)).toBeInTheDocument();
    expect(screen.getByText(/threshold 0\.35/)).toBeInTheDocument();
  });

  it("renders a per-org breakdown sorted by total desc (adf first)", () => {
    render(<TriageCards rows={rows} threshold={0.35} />);
    const items = screen.getAllByText(/adf|sis-67/);
    expect(items[0].textContent).toBe("adf");
    expect(items[1].textContent).toBe("sis-67");
  });

  it("renders empty without crashing", () => {
    render(<TriageCards rows={[]} />);
    expect(screen.getAllByText("Review").length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText("Unlabel").length).toBeGreaterThanOrEqual(1);
  });
});
