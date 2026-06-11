import { describe, expect, it } from "vitest";
import { squarePixelBox, stabilizedCropStyle } from "@/lib/crop";

describe("squarePixelBox", () => {
  it("expands a normalized window and squares it in pixels", () => {
    const b = squarePixelBox([0.5, 0.5, 0.1, 0.1], 1000, 1000, 2.0);
    expect(b.side).toBeCloseTo(200, 5);
    expect(b.x0).toBeCloseTo(400, 5);
    expect(b.y0).toBeCloseTo(400, 5);
  });
  it("uses the larger pixel extent for the square on non-square images", () => {
    const b = squarePixelBox([0.5, 0.5, 0.1, 0.1], 2000, 1000, 1.0);
    expect(b.side).toBeCloseTo(200, 5);
  });
});

describe("stabilizedCropStyle", () => {
  it("scales the image so the square box fills the display box", () => {
    const s = stabilizedCropStyle([0.5, 0.5, 0.1, 0.1], 1000, 1000, 2.0, 220);
    expect(s.width).toBeCloseTo(1100, 4);
    expect(s.height).toBeCloseTo(1100, 4);
    expect(s.left).toBeCloseTo(-440, 4);
    expect(s.top).toBeCloseTo(-440, 4);
  });
});
