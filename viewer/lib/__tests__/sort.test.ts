import { describe, expect, it } from "vitest";
import { nextSort, sortRows } from "@/lib/sort";
import type { ResultRow } from "@/lib/types";

const r = (
  key: string,
  probability: number | null,
  camera: string | null = "c",
): ResultRow => ({
  key,
  source: "s",
  label: "smoke",
  decision: "keep",
  outcome: "kept-smoke",
  score: 1,
  probability,
  num_tubes_kept: 1,
  trigger_frame_index: 0,
  organization_name: null,
  camera_name: camera,
  started_at: null,
});

describe("sortRows by probability", () => {
  const rows = [r("a", 0.2), r("b", 0.9), r("c", null), r("d", 0.5)];
  it("desc puts highest first, nulls last", () => {
    expect(
      sortRows(rows, { col: "probability", dir: "desc" }).map((x) => x.key),
    ).toEqual(["b", "d", "a", "c"]);
  });
  it("asc puts lowest first, nulls still last", () => {
    expect(
      sortRows(rows, { col: "probability", dir: "asc" }).map((x) => x.key),
    ).toEqual(["a", "d", "b", "c"]);
  });
  it("no sort returns a copy in original order", () => {
    const out = sortRows(rows, null);
    expect(out.map((x) => x.key)).toEqual(["a", "b", "c", "d"]);
    expect(out).not.toBe(rows);
  });
});

describe("nextSort", () => {
  it("new column starts desc; same column toggles", () => {
    expect(nextSort(null, "probability")).toEqual({
      col: "probability",
      dir: "desc",
    });
    expect(
      nextSort({ col: "probability", dir: "desc" }, "probability"),
    ).toEqual({
      col: "probability",
      dir: "asc",
    });
    expect(nextSort({ col: "probability", dir: "asc" }, "camera")).toEqual({
      col: "camera",
      dir: "desc",
    });
  });
});
