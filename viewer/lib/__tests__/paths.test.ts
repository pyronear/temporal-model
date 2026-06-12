import path from "node:path";
import { describe, expect, it } from "vitest";
import { MODEL_NAME, resolveFramePath } from "@/lib/paths";

const ROOT = "/tmp/evalroot";

describe("resolveFramePath", () => {
  it("joins a relative frame path under DATA_ROOT", () => {
    expect(resolveFramePath(ROOT, "data/01_raw/x/images/f.jpg")).toBe(
      path.join(ROOT, "data/01_raw/x/images/f.jpg"),
    );
  });
  it("rejects traversal outside DATA_ROOT", () => {
    expect(() => resolveFramePath(ROOT, "../../etc/passwd")).toThrow();
    expect(() => resolveFramePath(ROOT, "/etc/passwd")).toThrow();
  });
  it("exposes the model name", () => {
    expect(MODEL_NAME).toBe("vit_dinov2_finetune");
  });
});
