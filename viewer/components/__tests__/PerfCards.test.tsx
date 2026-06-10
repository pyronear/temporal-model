import { render, screen } from "@testing-library/react";
import { it, expect } from "vitest";
import { PerfCards } from "@/components/PerfCards";
import type { ResultRow } from "@/lib/types";

const r = (
  label: ResultRow["label"],
  outcome: ResultRow["outcome"],
): ResultRow => ({
  key: "k",
  source: "s",
  label,
  decision: "keep",
  outcome,
  score: 1,
  probability: 0.9,
  num_tubes_kept: 1,
  trigger_frame_index: 0,
  organization_name: null,
  camera_name: null,
  started_at: null,
});

it("renders recall over labeled rows", () => {
  render(
    <PerfCards
      rows={[r("smoke", "kept-smoke"), r("smoke", "discarded-smoke")]}
    />,
  );
  expect(screen.getByText("50.0%")).toBeInTheDocument();
});
