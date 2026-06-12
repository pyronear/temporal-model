import { expect, it } from "vitest";
import { correctnessLabel, outcomeTokens, rowTokens } from "@/lib/correctness";
import type { Outcome } from "@/lib/types";

it("labels", () => {
  expect(correctnessLabel("discarded-smoke")).toBe("missed smoke");
  expect(correctnessLabel("kept-fp")).toBe("false alarm");
  expect(correctnessLabel("n/a")).toBe("—");
});

it("tokens exist for every outcome", () => {
  const outcomes: Outcome[] = [
    "kept-smoke",
    "discarded-fp",
    "kept-fp",
    "discarded-smoke",
    "n/a",
  ];
  for (const o of outcomes) {
    expect(outcomeTokens[o]).toBeDefined();
    expect(outcomeTokens[o].dot).toMatch(/^#/);
  }
});

it("rowTokens: n/a keep is green-tinted, n/a discard is gray", () => {
  const keep = rowTokens("n/a", "keep");
  const discard = rowTokens("n/a", "discard");
  // keep → green family (matches kept-smoke bg)
  expect(keep).toEqual({ bg: "#ecfdf5", dot: "#10b981", text: "#047857" });
  // discard → muted gray
  expect(discard).toEqual({ bg: "#f8fafc", dot: "#94a3b8", text: "#64748b" });
  // they must be distinct
  expect(keep.bg).not.toBe(discard.bg);
  expect(keep.dot).not.toBe(discard.dot);
});

it("rowTokens: labeled outcomes are byte-identical to outcomeTokens", () => {
  expect(rowTokens("kept-smoke", "keep")).toEqual(outcomeTokens["kept-smoke"]);
  expect(rowTokens("discarded-fp", "discard")).toEqual(
    outcomeTokens["discarded-fp"],
  );
  expect(rowTokens("kept-fp", "keep")).toEqual(outcomeTokens["kept-fp"]);
  expect(rowTokens("discarded-smoke", "discard")).toEqual(
    outcomeTokens["discarded-smoke"],
  );
});
