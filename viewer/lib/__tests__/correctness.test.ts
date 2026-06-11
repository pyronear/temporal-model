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

it("rowTokens falls back to verdict tint for n/a (GT unknown)", () => {
  expect(rowTokens("n/a", "keep").bg).not.toBe(rowTokens("n/a", "discard").bg);
});
