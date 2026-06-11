import { render } from "@testing-library/react";
import { expect, it } from "vitest";
import { BboxOverlay } from "@/components/detail/BboxOverlay";

it("renders a rect per box at normalized coords", () => {
  const { container } = render(
    <BboxOverlay
      boxes={[
        {
          bbox: [0.5, 0.5, 0.2, 0.2],
          color: "#059669",
          trigger: "decisive",
          confidence: 0.9,
        },
      ]}
    />,
  );
  const rect = container.querySelector("rect")!;
  expect(rect.getAttribute("x")).toBe("0.4");
  expect(rect.getAttribute("width")).toBe("0.2");
});
