import { fireEvent, render, screen } from "@testing-library/react";
import { expect, it, vi } from "vitest";
import { ThresholdSlider } from "@/components/ThresholdSlider";

it("emits onChange and onReset", () => {
  const onChange = vi.fn();
  const onReset = vi.fn();
  render(
    <ThresholdSlider
      value={0.47}
      defaultValue={0.47}
      onChange={onChange}
      onReset={onReset}
    />,
  );
  fireEvent.change(screen.getByLabelText("logistic threshold"), {
    target: { value: "0.3" },
  });
  expect(onChange).toHaveBeenCalledWith(0.3);
  fireEvent.click(screen.getByText("↺ reset"));
  expect(onReset).toHaveBeenCalled();
});
