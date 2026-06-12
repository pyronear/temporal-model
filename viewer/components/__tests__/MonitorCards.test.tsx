import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { MonitorCards } from "@/components/MonitorCards";
import type { ResultRow } from "@/lib/types";

const r = (
  key: string,
  decision: ResultRow["decision"],
  organization_name: string | null = null,
): ResultRow => ({
  key,
  source: "s",
  label: "smoke",
  decision,
  outcome: decision === "keep" ? "kept-smoke" : "discarded-fp",
  score: 1,
  probability: 0.9,
  num_tubes_kept: 1,
  trigger_frame_index: 0,
  organization_name,
  camera_name: null,
  started_at: null,
});

const rows: ResultRow[] = [
  r("a", "keep", "org-alpha"),
  r("b", "keep", "org-alpha"),
  r("c", "discard", "org-alpha"),
  r("d", "keep", "org-beta"),
  r("e", "discard", "org-beta"),
  r("f", "discard", null),
];

describe("MonitorCards", () => {
  it("shows correct kept and discarded counts", () => {
    render(<MonitorCards rows={rows} />);
    // 3 kept, 3 discarded
    const counts = screen.getAllByText("3");
    expect(counts.length).toBeGreaterThanOrEqual(2);
  });

  it("renders per-org breakdown with kept/discarded numbers", () => {
    render(<MonitorCards rows={rows} />);
    expect(screen.getByText("org-alpha")).toBeInTheDocument();
    expect(screen.getByText("org-beta")).toBeInTheDocument();
    // null org renders as "—"
    expect(screen.getByText("—")).toBeInTheDocument();
  });

  it("sorts orgs by total desc (org-alpha first)", () => {
    render(<MonitorCards rows={rows} />);
    const items = screen.getAllByText(/org-/);
    expect(items[0].textContent).toBe("org-alpha");
    expect(items[1].textContent).toBe("org-beta");
  });

  it("renders empty without crashing", () => {
    render(<MonitorCards rows={[]} />);
    expect(screen.getByText("Kept")).toBeInTheDocument();
    expect(screen.getByText("Discarded")).toBeInTheDocument();
  });
});
