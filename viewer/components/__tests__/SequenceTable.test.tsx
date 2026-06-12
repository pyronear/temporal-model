import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { expect, it, vi } from "vitest";
import { SequenceTable } from "@/components/SequenceTable";
import type { ResultRow } from "@/lib/types";

const r = (
  key: string,
  outcome: ResultRow["outcome"],
  organization_name: string | null = null,
): ResultRow => ({
  key,
  source: "s",
  label: "smoke",
  decision: "keep",
  outcome,
  score: 1,
  probability: 0.9,
  num_tubes_kept: 1,
  trigger_frame_index: 0,
  organization_name,
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

it("never shows prod prob column (match signal lives in detail pane)", () => {
  // eval rows
  render(<SequenceTable rows={rows} selectedKey={null} onSelect={() => {}} />);
  expect(screen.queryByText("prod prob")).toBeNull();
  cleanup();
  // monitor rows with provenance fields
  const monitorRow = {
    ...rows[0],
    key: "platform_1",
    replayed_probability: 0.931,
    replayed_decision: "keep" as const,
    replay_matches: false,
  };
  render(
    <SequenceTable
      rows={[monitorRow]}
      selectedKey={null}
      onSelect={() => {}}
    />,
  );
  expect(screen.queryByText("prod prob")).toBeNull();
});

it("shows organization column only in monitorMode", () => {
  const monitorRows = [r("a", "kept-smoke", "sis-67")];
  // eval mode: no organization column
  render(
    <SequenceTable rows={monitorRows} selectedKey={null} onSelect={() => {}} />,
  );
  expect(screen.queryByText("organization")).toBeNull();
  expect(screen.queryByText("sis-67")).toBeNull();
  cleanup();
  // monitor mode: organization column present with value
  render(
    <SequenceTable
      rows={monitorRows}
      selectedKey={null}
      onSelect={() => {}}
      monitorMode
    />,
  );
  expect(screen.getByText("organization")).toBeInTheDocument();
  expect(screen.getByText("sis-67")).toBeInTheDocument();
});

it("hides correctness column in monitorMode, shows it otherwise", () => {
  // eval mode: correctness column header present
  render(<SequenceTable rows={rows} selectedKey={null} onSelect={() => {}} />);
  expect(screen.getByText("correctness")).toBeInTheDocument();
  cleanup();
  // monitor mode: correctness column absent
  render(
    <SequenceTable
      rows={rows}
      selectedKey={null}
      onSelect={() => {}}
      monitorMode
    />,
  );
  expect(screen.queryByText("correctness")).toBeNull();
});
