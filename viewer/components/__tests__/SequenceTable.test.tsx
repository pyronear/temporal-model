import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { expect, it, vi } from "vitest";
import { SequenceTable } from "@/components/SequenceTable";
import type { ResultRow } from "@/lib/types";

const r = (key: string, outcome: ResultRow["outcome"]): ResultRow => ({
  key,
  source: "s",
  label: "smoke",
  decision: "keep",
  outcome,
  score: 1,
  probability: 0.9,
  num_tubes_kept: 1,
  trigger_frame_index: 0,
  organization_name: null,
  camera_name: "cam-1",
  started_at: null,
});

const rows = [r("a", "kept-smoke"), r("b", "kept-fp")];

it("renders rows and fires onSelect on click", () => {
  const onSelect = vi.fn();
  render(
    <SequenceTable
      rows={[r("a", "kept-smoke"), r("b", "kept-fp")]}
      selectedKey="a"
      onSelect={onSelect}
    />,
  );
  expect(screen.getByText("smoke kept")).toBeInTheDocument();
  expect(screen.getByText("false alarm")).toBeInTheDocument();
  fireEvent.click(screen.getByText("false alarm"));
  expect(onSelect).toHaveBeenCalledWith("b");
});

it("shows provenance columns only for monitor rows", () => {
  // eval rows: no prod prob column
  render(<SequenceTable rows={rows} selectedKey={null} onSelect={() => {}} />);
  expect(screen.queryByText("prod prob")).toBeNull();
  cleanup();
  const monitorRow = {
    ...rows[0],
    key: "platform_1",
    recorded_probability: 0.931,
    replay_matches: false,
  };
  render(
    <SequenceTable
      rows={[monitorRow]}
      selectedKey={null}
      onSelect={() => {}}
    />,
  );
  expect(screen.getByText("prod prob")).toBeTruthy();
  expect(screen.getByText("0.931")).toBeTruthy();
  expect(screen.getByText("≠")).toBeTruthy(); // mismatch marker
});
