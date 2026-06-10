# React/Next.js Eval Viewer Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A local Next.js + TypeScript + Tailwind app (`viewer/`) that replaces the Streamlit eval viewer at feature parity, with a master–detail layout, light palette-A theme, and client-side frame/bbox/crop rendering.

**Architecture:** Next.js App Router app run with `npm run dev`. Route handlers read the local eval reporting tree (`DATA_ROOT` → `eval/`) and stream frame jpgs. The React client renders a master–detail UI; bboxes are an SVG overlay and stabilized crops are CSS transforms. Pure presentation/decision helpers are ported from the Python viewer to TypeScript with parity unit tests. Consumes the unchanged PR-#39 data contract; no Python changes.

**Tech Stack:** Next.js (App Router) + TypeScript + Tailwind CSS + TanStack Table + Vitest + React Testing Library. Node 22, npm.

> **Branch:** create branch `arthur/feat-eval-viewer-nextjs` **from `arthur/feat-eval-qualitative-viewer`** (stacked PR; base = that branch). Do this at worktree-creation time — the new branch must include #39's commits (the data contract), so do NOT branch fresh from `origin/main`.

---

## File Structure (all under `viewer/`)

- `package.json`, `tsconfig.json`, `next.config.mjs`, `tailwind.config.ts`, `postcss.config.mjs`, `.env.local.example`, `vitest.config.ts`, `README.md`
- `app/layout.tsx`, `app/globals.css`, `app/page.tsx` — App Router shell + the master–detail page.
- `app/api/sources/route.ts`, `app/api/results/route.ts`, `app/api/sequence/[source]/[key]/route.ts`, `app/api/model-config/[source]/route.ts`, `app/api/frame/route.ts` — data + image handlers.
- `lib/types.ts` — TS mirror of the data contract.
- `lib/paths.ts` — `DATA_ROOT`, reporting-tree paths, frame-path resolution + validation.
- `lib/outcomes.ts` — `computeOutcome`, `applyThreshold`, `performanceSummary`.
- `lib/details.ts` — `processedToInputIndex`, `frameBboxesByInputIndex`, `tubeInputBoxes`, `triggeringTubeIds`, `triggerState`.
- `lib/crop.ts` — `stabilizedCropStyle` (CSS transform for a normalized window).
- `lib/correctness.ts` — outcome → label + Tailwind colour tokens.
- `lib/api.ts` — typed client fetchers.
- `components/ControlRail.tsx`, `SourceSelect.tsx`, `PerfCards.tsx`, `ThresholdSlider.tsx`, `ModelConfigPanel.tsx`
- `components/SequenceTable.tsx`
- `components/detail/DetailPanel.tsx`, `FrameViewer.tsx`, `BboxOverlay.tsx`, `TubeTimeline.tsx`, `TubeCrops.tsx`
- `lib/__tests__/*.test.ts`, `components/__tests__/*.test.tsx`

All commands run from `viewer/`.

---

## Task 1: Scaffold the Next.js app

**Files:** create the `viewer/` project.

- [ ] **Step 1: Scaffold**

From the repo root:

```bash
npx create-next-app@latest viewer --ts --tailwind --eslint --app --src-dir=false \
  --import-alias "@/*" --no-turbopack --use-npm
```

Accept defaults. This creates `viewer/` with App Router, TS, Tailwind, ESLint.

- [ ] **Step 2: Add Vitest + RTL + TanStack Table**

```bash
cd viewer
npm install @tanstack/react-table
npm install -D vitest @vitejs/plugin-react @testing-library/react @testing-library/jest-dom jsdom
```

- [ ] **Step 3: Vitest config**

Create `viewer/vitest.config.ts`:

```ts
import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

export default defineConfig({
  plugins: [react()],
  test: { environment: "jsdom", globals: true, setupFiles: ["./vitest.setup.ts"] },
  resolve: { alias: { "@": new URL(".", import.meta.url).pathname } },
});
```

Create `viewer/vitest.setup.ts`:

```ts
import "@testing-library/jest-dom/vitest";
```

Add to `viewer/package.json` `"scripts"`: `"test": "vitest run"`, `"test:watch": "vitest"`.

- [ ] **Step 4: Sanity test**

Create `viewer/lib/__tests__/smoke.test.ts`:

```ts
import { describe, expect, it } from "vitest";
it("vitest runs", () => {
  expect(1 + 1).toBe(2);
});
```

Run: `npm test`
Expected: 1 passed.

- [ ] **Step 5: Verify build + dev**

Run: `npm run build`
Expected: build succeeds (default app).
Run: `npm run dev` then `curl -s -o /dev/null -w "%{http_code}" http://localhost:3000` → `200`. Stop dev.

- [ ] **Step 6: Commit**

```bash
git add viewer
git commit -m "feat(viewer): scaffold Next.js + TS + Tailwind app"
```

---

## Task 2: Data-contract types

**Files:** Create `viewer/lib/types.ts`.

- [ ] **Step 1: Write the types**

```ts
export type Outcome =
  | "kept-smoke"
  | "discarded-fp"
  | "kept-fp"
  | "discarded-smoke"
  | "n/a";

export type Decision = "keep" | "discard";
export type Label = "smoke" | "fp" | "unknown";

export interface ResultRow {
  key: string;
  source: string;
  label: Label;
  decision: Decision;
  outcome: Outcome;
  score: number | null;
  probability: number | null;
  trigger_frame_index: number | null;
  organization_name: string | null;
  camera_name: string | null;
  started_at: string | null;
}

export interface KeptTubeEntry {
  frame_idx: number;
  bbox: [number, number, number, number] | null;
  is_gap: boolean;
  confidence: number | null;
}

export interface KeptTube {
  tube_id: number;
  start_frame: number;
  end_frame: number;
  logit: number;
  probability: number | null;
  first_crossing_frame: number | null;
  stabilized_window: [number, number, number, number] | null;
  entries: KeptTubeEntry[];
}

export interface BboxTubeDetails {
  preprocessing: { num_frames_input: number; num_truncated: number; padded_frame_indices: number[] };
  tubes: { num_candidates: number; kept: KeptTube[] };
  decision: {
    aggregation: "max_logit" | "logistic";
    threshold: number;
    logistic_threshold?: number;
    trigger_tube_id: number | null;
  };
}

export interface SequenceView {
  key: string;
  source: string;
  label: Label;
  organization_name: string | null;
  camera_name: string | null;
  started_at: string | null;
  frames: string[];
}

export interface ModelConfig {
  detector?: { source?: string; type?: string } | null;
  variant?: string | null;
  train_git_sha?: string | null;
  decision?: { aggregation?: string; threshold?: number; logistic_threshold?: number | null } | null;
  infer?: { pad_strategy?: string; pad_to_min_frames?: number } | null;
  model_input?: { stabilize?: boolean; context_factor?: number } | null;
  classifier?: { max_frames?: number; backbone?: string } | null;
  tubes?: Record<string, unknown> | null;
  calibrator?: unknown;
}
```

- [ ] **Step 2: Typecheck + commit**

Run: `npx tsc --noEmit`
Expected: no errors.

```bash
git add viewer/lib/types.ts
git commit -m "feat(viewer): data-contract TypeScript types"
```

---

## Task 3: Paths + frame resolution (with traversal guard)

**Files:** Create `viewer/lib/paths.ts`, `viewer/lib/__tests__/paths.test.ts`, `viewer/.env.local.example`.

- [ ] **Step 1: Write the failing test**

`viewer/lib/__tests__/paths.test.ts`:

```ts
import path from "node:path";
import { describe, expect, it } from "vitest";
import { resolveFramePath, MODEL_NAME } from "@/lib/paths";

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
```

- [ ] **Step 2: Run it — fails (module missing).**

Run: `npm test -- paths`
Expected: FAIL (cannot find `@/lib/paths`).

- [ ] **Step 3: Implement**

`viewer/lib/paths.ts`:

```ts
import path from "node:path";

export const MODEL_NAME = "vit_dinov2_finetune";

/** Absolute path to the eval/ dir. Configurable via DATA_ROOT (default ../eval). */
export function dataRoot(): string {
  return path.resolve(process.env.DATA_ROOT ?? path.join(process.cwd(), "..", "eval"));
}

export function reportingRoot(root = dataRoot()): string {
  return path.join(root, "data", "08_reporting");
}

export function sourceDir(source: string, root = dataRoot()): string {
  return path.join(reportingRoot(root), source, MODEL_NAME);
}

/** Resolve a frame path (relative to DATA_ROOT) and refuse anything escaping it. */
export function resolveFramePath(root: string, rel: string): string {
  const abs = path.resolve(root, rel);
  const base = path.resolve(root);
  if (abs !== base && !abs.startsWith(base + path.sep)) {
    throw new Error(`path escapes DATA_ROOT: ${rel}`);
  }
  return abs;
}
```

`viewer/.env.local.example`:

```
# Path to the eval/ package dir (holds data/08_reporting and the frame images).
DATA_ROOT=../eval
```

- [ ] **Step 4: Run test — passes. Commit.**

Run: `npm test -- paths` → PASS.

```bash
git add viewer/lib/paths.ts viewer/lib/__tests__/paths.test.ts viewer/.env.local.example
git commit -m "feat(viewer): data-root paths + frame-path traversal guard"
```

---

## Task 4: Ported pure logic — outcomes

**Files:** Create `viewer/lib/outcomes.ts`, `viewer/lib/__tests__/outcomes.test.ts`. Ports `eval/.../outcomes.py`.

- [ ] **Step 1: Write the failing tests**

`viewer/lib/__tests__/outcomes.test.ts`:

```ts
import { describe, expect, it } from "vitest";
import { applyThreshold, computeOutcome, performanceSummary } from "@/lib/outcomes";
import type { ResultRow } from "@/lib/types";

function row(p: Partial<ResultRow>): ResultRow {
  return {
    key: "k", source: "s", label: "smoke", decision: "keep", outcome: "kept-smoke",
    score: 1, probability: 0.9, trigger_frame_index: 0,
    organization_name: null, camera_name: null, started_at: null, ...p,
  };
}

describe("computeOutcome", () => {
  it("maps decision+label", () => {
    expect(computeOutcome("keep", "smoke")).toBe("kept-smoke");
    expect(computeOutcome("discard", "smoke")).toBe("discarded-smoke");
    expect(computeOutcome("keep", "fp")).toBe("kept-fp");
    expect(computeOutcome("discard", "fp")).toBe("discarded-fp");
    expect(computeOutcome("keep", "unknown")).toBe("n/a");
  });
});

describe("applyThreshold", () => {
  it("re-decides keep iff probability >= thr (null -> discard)", () => {
    const rows = [
      row({ label: "smoke", probability: 0.9 }),
      row({ label: "smoke", probability: 0.2 }),
      row({ label: "fp", probability: 0.8 }),
      row({ label: "smoke", probability: null }),
    ];
    const out = applyThreshold(rows, 0.5);
    expect(out.map((r) => r.decision)).toEqual(["keep", "discard", "keep", "discard"]);
    expect(out.map((r) => r.outcome)).toEqual([
      "kept-smoke", "discarded-smoke", "kept-fp", "discarded-smoke",
    ]);
    // input not mutated
    expect(rows[1].decision).toBe("keep");
  });
});

describe("performanceSummary", () => {
  it("computes recall / specificity / precision over labeled rows", () => {
    const rows = [
      row({ label: "smoke", outcome: "kept-smoke" }),
      row({ label: "smoke", outcome: "discarded-smoke" }),
      row({ label: "fp", outcome: "discarded-fp" }),
      row({ label: "fp", outcome: "kept-fp" }),
      row({ label: "unknown", outcome: "n/a" }),
    ];
    const s = performanceSummary(rows);
    expect(s.nSmoke).toBe(2);
    expect(s.nFp).toBe(2);
    expect(s.recall).toBe(0.5);
    expect(s.specificity).toBe(0.5);
    expect(s.precision).toBe(0.5);
  });
});
```

- [ ] **Step 2: Run — fails.** `npm test -- outcomes` → FAIL.

- [ ] **Step 3: Implement**

`viewer/lib/outcomes.ts`:

```ts
import type { Decision, Label, Outcome, ResultRow } from "@/lib/types";

export function computeOutcome(decision: Decision, label: Label): Outcome {
  if (label === "smoke") return decision === "keep" ? "kept-smoke" : "discarded-smoke";
  if (label === "fp") return decision === "keep" ? "kept-fp" : "discarded-fp";
  return "n/a";
}

/** Re-decide every row at a logistic threshold (does not mutate input). */
export function applyThreshold(rows: ResultRow[], threshold: number): ResultRow[] {
  return rows.map((r) => {
    const decision: Decision =
      r.probability != null && r.probability >= threshold ? "keep" : "discard";
    return { ...r, decision, outcome: computeOutcome(decision, r.label) };
  });
}

export interface PerfSummary {
  nLabeled: number; nSmoke: number; nFp: number;
  keptSmoke: number; discardedSmoke: number; discardedFp: number; keptFp: number;
  recall: number | null; specificity: number | null; precision: number | null;
}

export function performanceSummary(rows: ResultRow[]): PerfSummary {
  const labeled = rows.filter((r) => r.label === "smoke" || r.label === "fp");
  const n = (o: Outcome) => labeled.filter((r) => r.outcome === o).length;
  const keptSmoke = n("kept-smoke"), discardedSmoke = n("discarded-smoke");
  const discardedFp = n("discarded-fp"), keptFp = n("kept-fp");
  const nSmoke = keptSmoke + discardedSmoke;
  const nFp = discardedFp + keptFp;
  const nKept = keptSmoke + keptFp;
  return {
    nLabeled: nSmoke + nFp, nSmoke, nFp, keptSmoke, discardedSmoke, discardedFp, keptFp,
    recall: nSmoke ? keptSmoke / nSmoke : null,
    specificity: nFp ? discardedFp / nFp : null,
    precision: nKept ? keptSmoke / nKept : null,
  };
}
```

- [ ] **Step 4: Run — passes. Commit.**

Run: `npm test -- outcomes` → PASS.

```bash
git add viewer/lib/outcomes.ts viewer/lib/__tests__/outcomes.test.ts
git commit -m "feat(viewer): port outcome/threshold/perf helpers (parity)"
```

---

## Task 5: Ported pure logic — details (frame/tube mapping)

**Files:** Create `viewer/lib/details.ts`, `viewer/lib/__tests__/details.test.ts`. Ports `eval/.../render.py`.

- [ ] **Step 1: Write the failing tests**

`viewer/lib/__tests__/details.test.ts`:

```ts
import { describe, expect, it } from "vitest";
import {
  processedToInputIndex, frameBboxesByInputIndex, tubeInputBoxes, triggeringTubeIds,
} from "@/lib/details";
import type { BboxTubeDetails } from "@/lib/types";

describe("processedToInputIndex", () => {
  it("identity with no padding", () => {
    expect(processedToInputIndex(5, [])).toBe(5);
  });
  it("maps real slots, null for synthetic", () => {
    const padded = [0, 3];
    expect(processedToInputIndex(1, padded)).toBe(0);
    expect(processedToInputIndex(2, padded)).toBe(1);
    expect(processedToInputIndex(0, padded)).toBeNull();
    expect(processedToInputIndex(3, padded)).toBeNull();
  });
});

describe("frameBboxesByInputIndex", () => {
  it("groups kept-tube boxes by input frame, skipping null bbox", () => {
    const d = {
      preprocessing: { padded_frame_indices: [] },
      tubes: { kept: [{ tube_id: 0, entries: [
        { frame_idx: 0, bbox: [0.5, 0.5, 0.1, 0.1], confidence: 0.7, is_gap: false },
        { frame_idx: 1, bbox: null, confidence: null, is_gap: true },
      ] }] },
    } as unknown as BboxTubeDetails;
    const out = frameBboxesByInputIndex(d);
    expect(out.get(0)).toEqual([{ bbox: [0.5, 0.5, 0.1, 0.1], confidence: 0.7, tubeId: 0 }]);
    expect(out.has(1)).toBe(false);
  });
});

describe("triggeringTubeIds", () => {
  it("logistic uses probability", () => {
    const d = { decision: { aggregation: "logistic", threshold: 0.5 },
      tubes: { kept: [{ tube_id: 0, probability: 0.9 }, { tube_id: 2, probability: 0.03 }] } } as unknown as BboxTubeDetails;
    expect([...triggeringTubeIds(d)]).toEqual([0]);
  });
  it("max_logit uses logit", () => {
    const d = { decision: { aggregation: "max_logit", threshold: 1.0 },
      tubes: { kept: [{ tube_id: 0, logit: 2.5 }, { tube_id: 1, logit: 0.2 }] } } as unknown as BboxTubeDetails;
    expect([...triggeringTubeIds(d)]).toEqual([0]);
  });
});

describe("tubeInputBoxes", () => {
  it("returns input-index boxes for real entries", () => {
    const tube = { entries: [
      { frame_idx: 0, bbox: [0.5, 0.5, 0.1, 0.1], confidence: 0.7, is_gap: false },
      { frame_idx: 1, bbox: null, confidence: null, is_gap: true },
    ] } as any;
    expect(tubeInputBoxes(tube, [])).toEqual([
      { inputIdx: 0, bbox: [0.5, 0.5, 0.1, 0.1], confidence: 0.7 },
    ]);
  });
});
```

- [ ] **Step 2: Run — fails.** `npm test -- details` → FAIL.

- [ ] **Step 3: Implement**

`viewer/lib/details.ts`:

```ts
import type { BboxTubeDetails, KeptTube } from "@/lib/types";

export function processedToInputIndex(frameIdx: number, padded: number[]): number | null {
  if (padded.includes(frameIdx)) return null;
  return frameIdx - padded.filter((p) => p < frameIdx).length;
}

export interface FrameBox { bbox: [number, number, number, number]; confidence: number | null; tubeId: number }

export function frameBboxesByInputIndex(details: BboxTubeDetails | null): Map<number, FrameBox[]> {
  const padded = details?.preprocessing?.padded_frame_indices ?? [];
  const out = new Map<number, FrameBox[]>();
  for (const tube of details?.tubes?.kept ?? []) {
    for (const e of tube.entries) {
      if (e.bbox == null) continue;
      const inp = processedToInputIndex(e.frame_idx, padded);
      if (inp == null) continue;
      const list = out.get(inp) ?? [];
      list.push({ bbox: e.bbox, confidence: e.confidence, tubeId: tube.tube_id });
      out.set(inp, list);
    }
  }
  return out;
}

export interface TubeBox { inputIdx: number; bbox: [number, number, number, number]; confidence: number | null }

export function tubeInputBoxes(tube: KeptTube, padded: number[]): TubeBox[] {
  const boxes: TubeBox[] = [];
  for (const e of tube.entries) {
    if (e.bbox == null) continue;
    const inp = processedToInputIndex(e.frame_idx, padded);
    if (inp != null) boxes.push({ inputIdx: inp, bbox: e.bbox, confidence: e.confidence });
  }
  return boxes;
}

export function triggeringTubeIds(details: BboxTubeDetails | null): Set<number> {
  const dec = details?.decision;
  if (!dec || dec.threshold == null) return new Set();
  const useProb = dec.aggregation === "logistic";
  const ids = new Set<number>();
  for (const t of details?.tubes?.kept ?? []) {
    const v = useProb ? t.probability : t.logit;
    if (v != null && v >= dec.threshold) ids.add(t.tube_id);
  }
  return ids;
}

export type TriggerState = "decisive" | "would" | null;

export function triggerState(
  tubeId: number | null, triggerTubeId: number | null, wouldIds: Set<number>,
): TriggerState {
  if (tubeId != null && tubeId === triggerTubeId) return "decisive";
  return tubeId != null && wouldIds.has(tubeId) ? "would" : null;
}
```

- [ ] **Step 4: Run — passes. Commit.**

Run: `npm test -- details` → PASS.

```bash
git add viewer/lib/details.ts viewer/lib/__tests__/details.test.ts
git commit -m "feat(viewer): port frame/tube detail helpers (parity)"
```

---

## Task 6: Stabilized-crop transform

**Files:** Create `viewer/lib/crop.ts`, `viewer/lib/__tests__/crop.test.ts`. Mirrors `core.crop` (expand → square → crop) as a CSS transform.

- [ ] **Step 1: Write the failing test**

`viewer/lib/__tests__/crop.test.ts`:

```ts
import { describe, expect, it } from "vitest";
import { squarePixelBox, stabilizedCropStyle } from "@/lib/crop";

describe("squarePixelBox", () => {
  it("expands a normalized window and squares it in pixels", () => {
    // window centred at (0.5,0.5), w=h=0.1; context 2.0 -> 0.2 norm.
    // imgW=imgH=1000 -> side = max(0.2*1000, 0.2*1000)=200; centre px (500,500)
    const b = squarePixelBox([0.5, 0.5, 0.1, 0.1], 1000, 1000, 2.0);
    expect(b.side).toBeCloseTo(200, 5);
    expect(b.x0).toBeCloseTo(400, 5);
    expect(b.y0).toBeCloseTo(400, 5);
  });
  it("uses the larger pixel extent for the square on non-square images", () => {
    // w=0.1 over 2000px = 200; h=0.1 over 1000px = 100 -> side = 200
    const b = squarePixelBox([0.5, 0.5, 0.1, 0.1], 2000, 1000, 1.0);
    expect(b.side).toBeCloseTo(200, 5);
  });
});

describe("stabilizedCropStyle", () => {
  it("scales the image so the square box fills the display box", () => {
    const s = stabilizedCropStyle([0.5, 0.5, 0.1, 0.1], 1000, 1000, 2.0, 220);
    // scale = 220/200 = 1.1 -> img width = 1100
    expect(s.width).toBeCloseTo(1100, 4);
    expect(s.height).toBeCloseTo(1100, 4);
    // translate = -x0*scale = -400*1.1 = -440
    expect(s.left).toBeCloseTo(-440, 4);
    expect(s.top).toBeCloseTo(-440, 4);
  });
});
```

- [ ] **Step 2: Run — fails.** `npm test -- crop` → FAIL.

- [ ] **Step 3: Implement**

`viewer/lib/crop.ts`:

```ts
export interface SquareBox { x0: number; y0: number; side: number }

/**
 * Pixel-space square crop for a normalized window (cx,cy,w,h), expanded by
 * `context`. Mirrors core.crop: expand_bbox (scale w,h by context) ->
 * norm_bbox_to_pixel_square (side = max pixel extent, centred on the bbox).
 */
export function squarePixelBox(
  window: [number, number, number, number],
  imgW: number,
  imgH: number,
  context: number,
): SquareBox {
  const [cx, cy, w, h] = window;
  const ew = w * context;
  const eh = h * context;
  const side = Math.max(ew * imgW, eh * imgH);
  const cxPx = cx * imgW;
  const cyPx = cy * imgH;
  return { x0: cxPx - side / 2, y0: cyPx - side / 2, side };
}

export interface CropStyle { width: number; height: number; left: number; top: number }

/**
 * Style for an <img> inside a `displaySize`x`displaySize` overflow-hidden box so
 * the stabilized window square fills the box. Apply width/height (px) + absolute
 * left/top (px) to the <img>.
 */
export function stabilizedCropStyle(
  window: [number, number, number, number],
  imgW: number,
  imgH: number,
  context: number,
  displaySize: number,
): CropStyle {
  const { x0, y0, side } = squarePixelBox(window, imgW, imgH, context);
  const scale = displaySize / side;
  return { width: imgW * scale, height: imgH * scale, left: -x0 * scale, top: -y0 * scale };
}
```

- [ ] **Step 4: Run — passes. Commit.**

Run: `npm test -- crop` → PASS.

```bash
git add viewer/lib/crop.ts viewer/lib/__tests__/crop.test.ts
git commit -m "feat(viewer): stabilized-crop CSS transform helper"
```

---

## Task 7: Correctness colour tokens

**Files:** Create `viewer/lib/correctness.ts`, `viewer/lib/__tests__/correctness.test.ts`. Palette A.

- [ ] **Step 1: Write the failing test**

`viewer/lib/__tests__/correctness.test.ts`:

```ts
import { describe, expect, it } from "vitest";
import { correctnessLabel, outcomeTokens, rowTokens } from "@/lib/correctness";

it("labels", () => {
  expect(correctnessLabel("discarded-smoke")).toBe("missed smoke");
  expect(correctnessLabel("kept-fp")).toBe("false alarm");
  expect(correctnessLabel("n/a")).toBe("—");
});

it("tokens exist for every outcome", () => {
  for (const o of ["kept-smoke", "discarded-fp", "kept-fp", "discarded-smoke", "n/a"] as const) {
    expect(outcomeTokens[o]).toBeDefined();
    expect(outcomeTokens[o].dot).toMatch(/^#/);
  }
});

it("rowTokens falls back to verdict tint for n/a (GT unknown)", () => {
  expect(rowTokens("n/a", "keep").bg).not.toBe(rowTokens("n/a", "discard").bg);
});
```

- [ ] **Step 2: Run — fails.** `npm test -- correctness` → FAIL.

- [ ] **Step 3: Implement**

`viewer/lib/correctness.ts`:

```ts
import type { Decision, Outcome } from "@/lib/types";

export const correctnessLabel = (o: Outcome): string =>
  ({
    "kept-smoke": "smoke kept",
    "discarded-fp": "fp filtered",
    "kept-fp": "false alarm",
    "discarded-smoke": "missed smoke",
    "n/a": "—",
  })[o] ?? o;

export interface Tokens { bg: string; dot: string; text: string }

// Palette A (light). bg = 50-tint, dot = saturated accent, text = 800.
export const outcomeTokens: Record<Outcome, Tokens> = {
  "kept-smoke": { bg: "#ecfdf5", dot: "#059669", text: "#065f46" },
  "discarded-fp": { bg: "#f0fdfa", dot: "#0d9488", text: "#115e59" },
  "kept-fp": { bg: "#fffbeb", dot: "#f59e0b", text: "#92400e" },
  "discarded-smoke": { bg: "#fff1f2", dot: "#e11d48", text: "#9f1239" },
  "n/a": { bg: "#f8fafc", dot: "#94a3b8", text: "#475569" },
};

const UNKNOWN_KEEP: Tokens = { bg: "#eff6ff", dot: "#3b82f6", text: "#1e40af" };

/** Row colours: errors/correct from outcome; GT-unknown tinted by verdict. */
export function rowTokens(outcome: Outcome, verdict: Decision): Tokens {
  if (outcome === "n/a") return verdict === "keep" ? UNKNOWN_KEEP : outcomeTokens["n/a"];
  return outcomeTokens[outcome];
}
```

- [ ] **Step 4: Run — passes. Commit.**

Run: `npm test -- correctness` → PASS.

```bash
git add viewer/lib/correctness.ts viewer/lib/__tests__/correctness.test.ts
git commit -m "feat(viewer): palette-A correctness tokens"
```

---

## Task 8: API route handlers + client fetchers

**Files:** Create the five `app/api/.../route.ts` handlers + `viewer/lib/api.ts`.

- [ ] **Step 1: Implement the handlers**

`app/api/sources/route.ts`:

```ts
import { promises as fs } from "node:fs";
import { NextResponse } from "next/server";
import { reportingRoot, MODEL_NAME } from "@/lib/paths";

export async function GET() {
  const root = reportingRoot();
  let entries: string[] = [];
  try {
    entries = await fs.readdir(root);
  } catch {
    return NextResponse.json([]);
  }
  const sources: string[] = [];
  for (const name of entries) {
    try {
      await fs.access(`${root}/${name}/${MODEL_NAME}/results.json`);
      sources.push(name);
    } catch {
      /* skip */
    }
  }
  // pyro-annotator first, then alphabetical (mirrors the Streamlit default).
  sources.sort((a, b) =>
    a === "pyro-annotator" ? -1 : b === "pyro-annotator" ? 1 : a.localeCompare(b));
  return NextResponse.json(sources);
}
```

`app/api/results/route.ts`:

```ts
import { promises as fs } from "node:fs";
import { NextResponse } from "next/server";
import { sourceDir, reportingRoot, MODEL_NAME } from "@/lib/paths";

export async function GET() {
  const root = reportingRoot();
  let names: string[] = [];
  try { names = await fs.readdir(root); } catch { return NextResponse.json([]); }
  const rows: unknown[] = [];
  for (const source of names) {
    try {
      const txt = await fs.readFile(`${sourceDir(source)}/results.json`, "utf8");
      void MODEL_NAME;
      rows.push(...JSON.parse(txt));
    } catch { /* skip sources without results */ }
  }
  return NextResponse.json(rows);
}
```

`app/api/sequence/[source]/[key]/route.ts`:

```ts
import { promises as fs } from "node:fs";
import { NextResponse } from "next/server";
import { sourceDir } from "@/lib/paths";

export async function GET(
  _req: Request,
  { params }: { params: Promise<{ source: string; key: string }> },
) {
  const { source, key } = await params;
  const dir = sourceDir(source);
  const read = async (p: string) => {
    try { return JSON.parse(await fs.readFile(p, "utf8")); } catch { return null; }
  };
  const details = await read(`${dir}/details/${key}.json`);
  const view = await read(`${dir}/sequences/${key}.json`);
  return NextResponse.json({ details, view });
}
```

`app/api/model-config/[source]/route.ts`:

```ts
import { promises as fs } from "node:fs";
import { NextResponse } from "next/server";
import { sourceDir } from "@/lib/paths";

export async function GET(
  _req: Request,
  { params }: { params: Promise<{ source: string }> },
) {
  const { source } = await params;
  try {
    return NextResponse.json(JSON.parse(await fs.readFile(`${sourceDir(source)}/model_config.json`, "utf8")));
  } catch {
    return NextResponse.json({});
  }
}
```

`app/api/frame/route.ts`:

```ts
import { promises as fs } from "node:fs";
import { NextResponse } from "next/server";
import { dataRoot, resolveFramePath } from "@/lib/paths";

export async function GET(req: Request) {
  const rel = new URL(req.url).searchParams.get("path");
  if (!rel) return new NextResponse("missing path", { status: 400 });
  let abs: string;
  try { abs = resolveFramePath(dataRoot(), rel); }
  catch { return new NextResponse("forbidden", { status: 400 }); }
  try {
    const buf = await fs.readFile(abs);
    return new NextResponse(buf, {
      headers: { "Content-Type": "image/jpeg", "Cache-Control": "public, max-age=31536000, immutable" },
    });
  } catch {
    return new NextResponse("not found", { status: 404 });
  }
}
```

- [ ] **Step 2: Client fetchers**

`viewer/lib/api.ts`:

```ts
import type { BboxTubeDetails, ModelConfig, ResultRow, SequenceView } from "@/lib/types";

async function getJSON<T>(url: string): Promise<T> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  return r.json() as Promise<T>;
}

export const fetchSources = () => getJSON<string[]>("/api/sources");
export const fetchResults = () => getJSON<ResultRow[]>("/api/results");
export const fetchModelConfig = (source: string) =>
  getJSON<ModelConfig>(`/api/model-config/${encodeURIComponent(source)}`);
export const fetchSequence = (source: string, key: string) =>
  getJSON<{ details: BboxTubeDetails | null; view: SequenceView | null }>(
    `/api/sequence/${encodeURIComponent(source)}/${encodeURIComponent(key)}`);
export const frameUrl = (relPath: string) => `/api/frame?path=${encodeURIComponent(relPath)}`;
```

- [ ] **Step 3: Manual verification against real data**

With the eval reporting tree present (it is, in this repo), run from `viewer/`:

```bash
DATA_ROOT=../eval npm run dev &
sleep 4
curl -s http://localhost:3000/api/sources           # ["pyro-annotator","val"]
curl -s http://localhost:3000/api/results | head -c 200   # array of rows
curl -s -o /dev/null -w "%{http_code}\n" \
  "http://localhost:3000/api/frame?path=$(curl -s http://localhost:3000/api/sequence/pyro-annotator/$(curl -s http://localhost:3000/api/results | python3 -c 'import sys,json;print(json.load(sys.stdin)[0]["key"])') | python3 -c 'import sys,json;print(json.load(sys.stdin)["view"]["frames"][0])')"   # 200
curl -s -o /dev/null -w "%{http_code}\n" "http://localhost:3000/api/frame?path=../../etc/passwd"  # 400
kill %1
```

Expected: sources list, a results array, frame 200, traversal 400.

- [ ] **Step 4: Typecheck + commit**

Run: `npx tsc --noEmit` → clean.

```bash
git add viewer/app/api viewer/lib/api.ts
git commit -m "feat(viewer): API route handlers + client fetchers"
```

---

## Task 9: Control rail components

**Files:** Create `components/SourceSelect.tsx`, `PerfCards.tsx`, `ThresholdSlider.tsx`, `ModelConfigPanel.tsx`, `ControlRail.tsx`, and `components/__tests__/ThresholdSlider.test.tsx`, `PerfCards.test.tsx`.

- [ ] **Step 1: PerfCards (pure render) + test**

`components/PerfCards.tsx`:

```tsx
import { performanceSummary, type PerfSummary } from "@/lib/outcomes";
import type { ResultRow } from "@/lib/types";

const pct = (v: number | null) => (v == null ? "—" : `${(v * 100).toFixed(1)}%`);

export function PerfCards({ rows }: { rows: ResultRow[] }) {
  const s: PerfSummary = performanceSummary(rows);
  if (s.nLabeled === 0) return null;
  const cards = [
    ["Recall (smoke kept)", pct(s.recall), `${s.keptSmoke}/${s.nSmoke}`],
    ["FP filtered", pct(s.specificity), `${s.discardedFp}/${s.nFp}`],
    ["Precision", pct(s.precision), `${s.keptSmoke}/${s.keptSmoke + s.keptFp}`],
  ];
  return (
    <div className="grid grid-cols-3 gap-2">
      {cards.map(([label, value, frac]) => (
        <div key={label} className="rounded-lg border border-slate-200 bg-white p-2">
          <div className="text-[11px] uppercase tracking-wide text-slate-500">{label}</div>
          <div className="text-lg font-semibold text-slate-900">{value}</div>
          <div className="text-[11px] text-slate-400">{frac}</div>
        </div>
      ))}
    </div>
  );
}
```

`components/__tests__/PerfCards.test.tsx`:

```tsx
import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { PerfCards } from "@/components/PerfCards";
import type { ResultRow } from "@/lib/types";

const r = (label: ResultRow["label"], outcome: ResultRow["outcome"]): ResultRow => ({
  key: "k", source: "s", label, decision: "keep", outcome, score: 1, probability: 0.9,
  trigger_frame_index: 0, organization_name: null, camera_name: null, started_at: null,
});

it("renders recall over labeled rows", () => {
  render(<PerfCards rows={[r("smoke", "kept-smoke"), r("smoke", "discarded-smoke")]} />);
  expect(screen.getByText("50.0%")).toBeInTheDocument();
});
```

Run: `npm test -- PerfCards` → PASS.

- [ ] **Step 2: ThresholdSlider (client) + test**

`components/ThresholdSlider.tsx`:

```tsx
"use client";

export function ThresholdSlider({
  value, defaultValue, onChange, onReset,
}: {
  value: number; defaultValue: number;
  onChange: (v: number) => void; onReset: () => void;
}) {
  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between">
        <label className="text-[11px] uppercase tracking-wide text-slate-500">
          logistic threshold
        </label>
        <button
          onClick={onReset}
          className="rounded border border-slate-200 px-2 py-0.5 text-xs text-slate-600 hover:bg-slate-50"
        >↺ reset</button>
      </div>
      <input
        type="range" min={0} max={1} step={0.01} value={value}
        aria-label="logistic threshold"
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full"
      />
      <div className="flex justify-between text-[11px] text-slate-400">
        <span>{value.toFixed(2)}</span>
        <span>model default: {defaultValue.toFixed(3)}</span>
      </div>
    </div>
  );
}
```

`components/__tests__/ThresholdSlider.test.tsx`:

```tsx
import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { ThresholdSlider } from "@/components/ThresholdSlider";

it("emits onChange and onReset", () => {
  const onChange = vi.fn(); const onReset = vi.fn();
  render(<ThresholdSlider value={0.47} defaultValue={0.47} onChange={onChange} onReset={onReset} />);
  fireEvent.change(screen.getByLabelText("logistic threshold"), { target: { value: "0.3" } });
  expect(onChange).toHaveBeenCalledWith(0.3);
  fireEvent.click(screen.getByText("↺ reset"));
  expect(onReset).toHaveBeenCalled();
});
```

Run: `npm test -- ThresholdSlider` → PASS.

- [ ] **Step 3: ModelConfigPanel + SourceSelect + ControlRail**

`components/ModelConfigPanel.tsx` — renders the headline fields with `title` tooltips
(port `MODEL_CONFIG_HELP` text + field list from `app.py`; each row: muted uppercase
label over value, `title` = help string). `components/SourceSelect.tsx` — a `<select>`
over sources. `components/ControlRail.tsx` — stacks `SourceSelect`, `PerfCards`,
`ThresholdSlider` (only when `showSlider`), `ModelConfigPanel` with a spacer so the
config sits lower. Full TSX:

```tsx
// components/SourceSelect.tsx
"use client";
export function SourceSelect({
  sources, value, onChange,
}: { sources: string[]; value: string; onChange: (s: string) => void }) {
  return (
    <div className="space-y-1">
      <label className="text-[11px] uppercase tracking-wide text-slate-500">source</label>
      <select
        className="w-full rounded-md border border-slate-300 bg-white px-2 py-1.5 text-sm"
        value={value} onChange={(e) => onChange(e.target.value)}
      >
        {sources.map((s) => <option key={s} value={s}>{s}</option>)}
      </select>
    </div>
  );
}
```

```tsx
// components/ModelConfigPanel.tsx
import type { ModelConfig } from "@/lib/types";

const HELP: Record<string, string> = {
  detector: "Stage-1 detector that proposes smoke boxes per frame; value is its source.",
  variant: "Packaged model variant name.",
  "train sha": "Git commit of the training run that produced this model.",
  aggregation: "Rule combining per-tube scores into the sequence keep/discard decision.",
  threshold: "Keep cutoff on a tube's raw classifier logit (max_logit aggregation).",
  "logistic threshold": "Keep cutoff on a tube's calibrated probability (logistic aggregation).",
  stabilize: "If true, each tube is cropped from one fixed window (stabilized).",
  "context factor": "How much the bbox is expanded before cropping the classifier patch.",
  "max frames": "Input sequence truncated to its first N frames; also caps frames per tube.",
  pad: "Short sequences are padded up to a minimum number of frames before detection.",
};

export function ModelConfigPanel({ cfg }: { cfg: ModelConfig }) {
  if (!cfg || Object.keys(cfg).length === 0)
    return <p className="text-xs text-slate-400">model config unavailable</p>;
  const d = cfg.decision ?? {}, mi = cfg.model_input ?? {}, inf = cfg.infer ?? {}, cl = cfg.classifier ?? {};
  const rows: [string, string][] = [
    ["detector", cfg.detector?.source ?? "—"],
    ["variant", cfg.variant ?? "—"],
    ["train sha", (cfg.train_git_sha ?? "").slice(0, 8) || "—"],
    ["aggregation", String(d.aggregation ?? "—")],
    ["threshold", String(d.threshold ?? "—")],
    ["logistic threshold", String(d.logistic_threshold ?? "—")],
    ["stabilize", String(mi.stabilize ?? "—")],
    ["context factor", String(mi.context_factor ?? "—")],
    ["max frames", String(cl.max_frames ?? "—")],
    ["pad", `${inf.pad_strategy ?? "—"} / min ${inf.pad_to_min_frames ?? "—"}`],
  ];
  return (
    <div className="space-y-2">
      <div className="text-[11px] uppercase tracking-wide text-slate-500">
        Model config <span className="normal-case text-slate-400">· hover a field</span>
      </div>
      {rows.map(([label, value]) => (
        <div key={label} title={HELP[label]} className="cursor-help leading-tight">
          <div className="text-[10px] uppercase tracking-wide text-slate-400">{label}</div>
          <div className="text-sm text-slate-800">{value}</div>
        </div>
      ))}
    </div>
  );
}
```

```tsx
// components/ControlRail.tsx
import { PerfCards } from "@/components/PerfCards";
import { SourceSelect } from "@/components/SourceSelect";
import { ThresholdSlider } from "@/components/ThresholdSlider";
import { ModelConfigPanel } from "@/components/ModelConfigPanel";
import type { ModelConfig, ResultRow } from "@/lib/types";

export function ControlRail(props: {
  sources: string[]; source: string; onSource: (s: string) => void;
  rows: ResultRow[]; cfg: ModelConfig;
  showSlider: boolean; threshold: number; defaultThreshold: number;
  onThreshold: (v: number) => void; onReset: () => void;
}) {
  return (
    <aside className="flex h-full w-72 shrink-0 flex-col gap-4 border-r border-slate-200 bg-slate-50 p-4">
      <SourceSelect sources={props.sources} value={props.source} onChange={props.onSource} />
      <PerfCards rows={props.rows} />
      {props.showSlider && (
        <ThresholdSlider
          value={props.threshold} defaultValue={props.defaultThreshold}
          onChange={props.onThreshold} onReset={props.onReset}
        />
      )}
      <div className="mt-auto" />
      <ModelConfigPanel cfg={props.cfg} />
    </aside>
  );
}
```

- [ ] **Step 4: Typecheck + tests + commit**

Run: `npx tsc --noEmit` and `npm test` → all pass.

```bash
git add viewer/components
git commit -m "feat(viewer): control-rail components (source, perf, slider, model config)"
```

---

## Task 10: Sequence table

**Files:** Create `components/SequenceTable.tsx`, `components/__tests__/SequenceTable.test.tsx`.

- [ ] **Step 1: Write the failing test**

`components/__tests__/SequenceTable.test.tsx`:

```tsx
import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { SequenceTable } from "@/components/SequenceTable";
import type { ResultRow } from "@/lib/types";

const r = (key: string, outcome: ResultRow["outcome"]): ResultRow => ({
  key, source: "s", label: "smoke", decision: "keep", outcome, score: 1, probability: 0.9,
  trigger_frame_index: 0, organization_name: null, camera_name: "cam-1", started_at: null,
});

it("renders rows and fires onSelect on click", () => {
  const onSelect = vi.fn();
  render(<SequenceTable rows={[r("a", "kept-smoke"), r("b", "kept-fp")]}
    selectedKey="a" onSelect={onSelect} />);
  expect(screen.getByText("smoke kept")).toBeInTheDocument();
  expect(screen.getByText("false alarm")).toBeInTheDocument();
  fireEvent.click(screen.getByText("false alarm"));
  expect(onSelect).toHaveBeenCalledWith("b");
});
```

- [ ] **Step 2: Run — fails.** `npm test -- SequenceTable` → FAIL.

- [ ] **Step 3: Implement**

`components/SequenceTable.tsx` — a TanStack Table over `ResultRow[]`. Columns:
camera, ground truth (label), model verdict (decision), correctness (chip via
`correctnessLabel` + `outcomeTokens[outcome].dot`), score, probability. Each row gets
`style={{ background: rowTokens(outcome, decision).bg }}`. `onSelect(key)` on row
click; the selected row gets a ring. Keyboard ↑/↓ on the table container moves the
selection through the *current row order* and calls `onSelect`. Full code:

```tsx
"use client";
import { useMemo, useRef } from "react";
import { correctnessLabel, outcomeTokens, rowTokens } from "@/lib/correctness";
import type { ResultRow } from "@/lib/types";

const num = (v: number | null) => (v == null ? "—" : v.toFixed(3));

export function SequenceTable({
  rows, selectedKey, onSelect,
}: { rows: ResultRow[]; selectedKey: string | null; onSelect: (key: string) => void }) {
  const ref = useRef<HTMLDivElement>(null);
  const order = useMemo(() => rows.map((r) => r.key), [rows]);

  function move(delta: number) {
    if (!order.length) return;
    const i = Math.max(0, order.indexOf(selectedKey ?? order[0]));
    onSelect(order[Math.min(order.length - 1, Math.max(0, i + delta))]);
  }

  return (
    <div
      ref={ref} tabIndex={0}
      onKeyDown={(e) => {
        if (e.key === "ArrowDown") { e.preventDefault(); move(1); }
        if (e.key === "ArrowUp") { e.preventDefault(); move(-1); }
      }}
      className="h-full overflow-auto rounded-lg border border-slate-200 outline-none"
    >
      <table className="w-full border-collapse text-sm">
        <thead className="sticky top-0 bg-white text-left text-xs uppercase tracking-wide text-slate-500">
          <tr>{["camera", "ground truth", "verdict", "correctness", "score", "prob"].map((h) => (
            <th key={h} className="border-b border-slate-200 px-3 py-2">{h}</th>))}</tr>
        </thead>
        <tbody>
          {rows.map((r) => {
            const t = rowTokens(r.outcome, r.decision);
            const sel = r.key === selectedKey;
            return (
              <tr
                key={r.key} onClick={() => onSelect(r.key)}
                style={{ background: t.bg }}
                className={`cursor-pointer ${sel ? "ring-2 ring-inset ring-slate-400" : ""}`}
              >
                <td className="px-3 py-1.5 text-slate-700">{r.camera_name ?? "—"}</td>
                <td className="px-3 py-1.5 text-slate-700">{r.label}</td>
                <td className="px-3 py-1.5 text-slate-700">{r.decision}</td>
                <td className="px-3 py-1.5" style={{ color: t.text }}>
                  <span className="mr-1.5 inline-block h-2 w-2 rounded-full align-middle"
                    style={{ background: outcomeTokens[r.outcome].dot }} />
                  {correctnessLabel(r.outcome)}
                </td>
                <td className="px-3 py-1.5 text-slate-600">{num(r.score)}</td>
                <td className="px-3 py-1.5 text-slate-600">{num(r.probability)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
```

(Filtering by ground truth / verdict / correctness / camera is applied by the page in
Task 12 before passing `rows`; sorting is a follow-up — TanStack header sorting can be
added without changing the row markup.)

- [ ] **Step 4: Run — passes. Commit.**

Run: `npm test -- SequenceTable` → PASS.

```bash
git add viewer/components/SequenceTable.tsx viewer/components/__tests__/SequenceTable.test.tsx
git commit -m "feat(viewer): error-coloured sequence table with keyboard nav"
```

---

## Task 11: Detail panel (frame viewer, bbox overlay, timeline, crops)

**Files:** Create `components/detail/BboxOverlay.tsx`, `FrameViewer.tsx`, `TubeTimeline.tsx`, `TubeCrops.tsx`, `DetailPanel.tsx`, and `components/__tests__/BboxOverlay.test.tsx`.

- [ ] **Step 1: BboxOverlay + test**

`components/detail/BboxOverlay.tsx`:

```tsx
import type { TriggerState } from "@/lib/details";

export interface OverlayBox {
  bbox: [number, number, number, number];
  color: string;
  trigger: TriggerState;
  confidence: number | null;
}

/** SVG overlay in normalized [0..1] space (viewBox 0 0 1 1), sits over the frame. */
export function BboxOverlay({ boxes }: { boxes: OverlayBox[] }) {
  return (
    <svg viewBox="0 0 1 1" preserveAspectRatio="none"
      className="pointer-events-none absolute inset-0 h-full w-full">
      {boxes.map((b, i) => {
        const [cx, cy, w, h] = b.bbox;
        const sw = b.trigger === "decisive" ? 0.006 : b.trigger === "would" ? 0.004 : 0.003;
        return (
          <rect key={i} x={cx - w / 2} y={cy - h / 2} width={w} height={h}
            fill="none" stroke={b.color} strokeWidth={sw} vectorEffect="non-scaling-stroke" />
        );
      })}
    </svg>
  );
}
```

`components/__tests__/BboxOverlay.test.tsx`:

```tsx
import { render } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { BboxOverlay } from "@/components/detail/BboxOverlay";

it("renders a rect per box at normalized coords", () => {
  const { container } = render(
    <BboxOverlay boxes={[{ bbox: [0.5, 0.5, 0.2, 0.2], color: "#059669", trigger: "decisive", confidence: 0.9 }]} />);
  const rect = container.querySelector("rect")!;
  expect(rect.getAttribute("x")).toBe("0.4");
  expect(rect.getAttribute("width")).toBe("0.2");
});
```

Run: `npm test -- BboxOverlay` → PASS.

- [ ] **Step 2: TubeTimeline (SVG)**

`components/detail/TubeTimeline.tsx` — one colour-coded bar row per kept tube across
frames, plus a vertical rule at the current frame and (if any) the trigger frame.
Input: `rows: { label: string; color: string; frames: number[]; trigger: TriggerState }[]`,
`n` (frame count), `current`, `trigger`. Render an SVG `width=100%`, height = rows*28;
each present frame → a unit-width `<rect>`; decisive rows get a dark stroke, would rows
a grey stroke. Full code:

```tsx
export interface TimelineRow { label: string; color: string; frames: number[]; trigger: "decisive" | "would" | null }

export function TubeTimeline({
  rows, n, current, trigger,
}: { rows: TimelineRow[]; n: number; current: number; trigger: number | null }) {
  if (!rows.length) return <p className="text-xs text-slate-400">no smoke tubes extracted</p>;
  const rh = 26, pad = 70, w = 600, plot = w - pad;
  const x = (f: number) => pad + (plot * f) / Math.max(1, n);
  return (
    <svg viewBox={`0 0 ${w} ${rows.length * rh + 16}`} className="w-full">
      {rows.map((r, ri) => (
        <g key={r.label} transform={`translate(0 ${ri * rh + 4})`}>
          <text x={0} y={16} className="fill-slate-600 text-[11px]">{r.label}</text>
          {r.frames.map((f) => (
            <rect key={f} x={x(f)} y={4} width={Math.max(2, plot / Math.max(1, n) - 1)} height={16}
              rx={2} fill={r.color}
              stroke={r.trigger === "decisive" ? "#111827" : r.trigger === "would" ? "#6b7280" : "none"}
              strokeWidth={r.trigger === "decisive" ? 2 : r.trigger === "would" ? 1 : 0} />
          ))}
        </g>
      ))}
      {trigger != null && (
        <line x1={x(trigger) + 1} x2={x(trigger) + 1} y1={0} y2={rows.length * rh}
          stroke="#c62828" strokeWidth={2} />
      )}
      <line x1={x(current) + 1} x2={x(current) + 1} y1={0} y2={rows.length * rh}
        stroke="#111827" strokeWidth={2} strokeDasharray="4 3" />
    </svg>
  );
}
```

- [ ] **Step 3: TubeCrops (CSS stabilized crop)**

`components/detail/TubeCrops.tsx` — for each kept tube active at the current input
frame, render a `displaySize` box with the frame `<img>` styled by
`stabilizedCropStyle(window, naturalW, naturalH, 2.0, displaySize)`. Needs the frame's
natural dimensions; obtain via an `onLoad` handler storing `{w,h}` in state (shared with
FrameViewer or measured per crop). Full code:

```tsx
"use client";
import { useState } from "react";
import { stabilizedCropStyle } from "@/lib/crop";
import { frameUrl } from "@/lib/api";
import type { KeptTube } from "@/lib/types";

const SIZE = 200, CONTEXT = 2.0;

export function TubeCrop({
  framePath, window, fallbackBbox,
}: { framePath: string; window: [number, number, number, number] | null;
     fallbackBbox: [number, number, number, number] | null }) {
  const [dim, setDim] = useState<{ w: number; h: number } | null>(null);
  const box = window ?? fallbackBbox;
  if (!box) return <p className="text-xs text-slate-400">inactive at this frame</p>;
  const style = dim ? stabilizedCropStyle(box, dim.w, dim.h, CONTEXT, SIZE) : null;
  return (
    <div className="relative overflow-hidden rounded-md bg-slate-100"
      style={{ width: SIZE, height: SIZE }}>
      <img src={frameUrl(framePath)} alt="" onLoad={(e) =>
        setDim({ w: e.currentTarget.naturalWidth, h: e.currentTarget.naturalHeight })}
        style={style ? { position: "absolute", width: style.width, height: style.height, left: style.left, top: style.top, maxWidth: "none" } : { opacity: 0 }} />
    </div>
  );
}

export function TubeCrops({
  tubes, framePath, activeBoxByTube,
}: { tubes: KeptTube[]; framePath: string;
     activeBoxByTube: Map<number, [number, number, number, number]> }) {
  return (
    <div className="space-y-3">
      {tubes.map((t) => (
        <TubeCrop key={t.tube_id} framePath={framePath}
          window={t.stabilized_window} fallbackBbox={activeBoxByTube.get(t.tube_id) ?? null} />
      ))}
    </div>
  );
}
```

- [ ] **Step 4: FrameViewer (autoplay) + DetailPanel**

`components/detail/FrameViewer.tsx` — an `<img>` (`frameUrl(frames[i])`) wrapped in a
`relative` box with `<BboxOverlay>`; an autoplay toggle advancing `i` every 1 s via
`setInterval`; a range slider for `i`. `DetailPanel.tsx` — given `details`, `view`,
`row`: compute `padded`, `kept`, `triggerTubeId`, `wouldIds` (from `lib/details`),
`bbmap`, the timeline rows, the decision header (verdict, correctness, probability,
trigger frame), and render `FrameViewer` + `TubeTimeline` + `TubeCrops`. The drill-down
shows the model's original `row` (not threshold-adjusted), matching the Streamlit
behaviour. Full code:

`FrameViewer` is **controlled** — the parent owns the frame index `i` (so the timeline
and crops stay in sync); the viewer just drives autoplay + the slider via `setI`.

```tsx
// components/detail/FrameViewer.tsx
"use client";
import { useEffect, useState } from "react";
import { frameUrl } from "@/lib/api";
import { BboxOverlay, type OverlayBox } from "@/components/detail/BboxOverlay";

export function FrameViewer({
  frames, boxesByFrame, i, setI,
}: {
  frames: string[];
  boxesByFrame: (i: number) => OverlayBox[];
  i: number;
  setI: (updater: (prev: number) => number) => void;
}) {
  const n = frames.length;
  const [playing, setPlaying] = useState(true);
  useEffect(() => {
    if (!playing || n === 0) return;
    const t = setInterval(() => setI((p) => (p + 1) % n), 1000);
    return () => clearInterval(t);
  }, [playing, n, setI]);
  if (n === 0) return null;
  const cur = i % n;
  return (
    <div className="space-y-2">
      <div className="flex items-center gap-3">
        <button onClick={() => setPlaying((p) => !p)}
          className="rounded border border-slate-300 px-2 py-0.5 text-sm">
          {playing ? "⏸ pause" : "▶ play"}</button>
        <input type="range" min={0} max={n - 1} value={cur}
          onChange={(e) => setI(() => parseInt(e.target.value, 10))} className="flex-1" />
        <span className="text-xs text-slate-500">{cur + 1}/{n}</span>
      </div>
      <div className="relative w-full overflow-hidden rounded-lg bg-slate-100">
        <img src={frameUrl(frames[cur])} alt={`frame ${cur}`} className="block w-full" />
        <BboxOverlay boxes={boxesByFrame(cur)} />
      </div>
    </div>
  );
}
```

```tsx
// components/detail/DetailPanel.tsx
"use client";
import { useMemo, useState } from "react";
import { FrameViewer } from "@/components/detail/FrameViewer";
import { TubeTimeline, type TimelineRow } from "@/components/detail/TubeTimeline";
import { TubeCrops } from "@/components/detail/TubeCrops";
import type { OverlayBox } from "@/components/detail/BboxOverlay";
import { correctnessLabel, outcomeTokens } from "@/lib/correctness";
import {
  frameBboxesByInputIndex, processedToInputIndex, triggerState, triggeringTubeIds, tubeInputBoxes,
} from "@/lib/details";
import type { BboxTubeDetails, ResultRow, SequenceView } from "@/lib/types";

const PALETTE = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf"];
const tubeColor = (id: number) => PALETTE[id % PALETTE.length];

export function DetailPanel({
  details, view, row,
}: { details: BboxTubeDetails | null; view: SequenceView | null; row: ResultRow }) {
  const frames = view?.frames ?? [];
  const padded = details?.preprocessing?.padded_frame_indices ?? [];
  const kept = details?.tubes?.kept ?? [];
  const triggerTubeId = details?.decision?.trigger_tube_id ?? null;
  const wouldIds = useMemo(() => triggeringTubeIds(details), [details]);
  const bbmap = useMemo(() => frameBboxesByInputIndex(details), [details]);
  const trig = row.trigger_frame_index != null
    ? processedToInputIndex(row.trigger_frame_index, padded) : null;

  const timeline: TimelineRow[] = kept.map((t) => ({
    label: `T${t.tube_id}`, color: tubeColor(t.tube_id),
    frames: tubeInputBoxes(t, padded).map((b) => b.inputIdx),
    trigger: triggerState(t.tube_id, triggerTubeId, wouldIds),
  }));

  const boxesByFrame = (i: number): OverlayBox[] =>
    (bbmap.get(i) ?? []).map((b) => ({
      bbox: b.bbox, color: tubeColor(b.tubeId),
      trigger: triggerState(b.tubeId, triggerTubeId, wouldIds), confidence: b.confidence,
    }));

  const activeBoxByTube = (frame: number) => {
    const m = new Map<number, [number, number, number, number]>();
    for (const t of kept) {
      const hit = tubeInputBoxes(t, padded).find((b) => b.inputIdx === frame);
      if (hit) m.set(t.tube_id, hit.bbox);
    }
    return m;
  };

  // The parent owns the frame index so the viewer, timeline, and crops stay in sync.
  const [i, setI] = useState(0);

  return (
    <section className="flex h-full flex-col gap-3 overflow-auto p-4">
      <header className="flex items-center justify-between">
        <h2 className="text-lg font-semibold">
          {row.decision === "keep" ? "💨 KEEP (smoke)" : "🚫 DISCARD (no smoke)"}
        </h2>
        <code className="text-xs text-slate-500">{row.key}</code>
      </header>
      <div className="grid grid-cols-4 gap-2 text-sm">
        <Stat label="verdict" value={row.decision} />
        <Stat label="correctness" value={correctnessLabel(row.outcome)}
          color={outcomeTokens[row.outcome].text} />
        <Stat label="trigger frame" value={trig == null ? "—" : String(trig)} />
        <Stat label="probability" value={row.probability == null ? "—" : row.probability.toFixed(3)} />
      </div>
      {frames.length > 0 && (
        <div className="grid grid-cols-[2fr_1fr] gap-3">
          <div className="space-y-2">
            <FrameViewer frames={frames} boxesByFrame={boxesByFrame} i={i} setI={setI} />
            <TubeTimeline rows={timeline} n={frames.length} current={i % frames.length} trigger={trig} />
          </div>
          <TubeCrops tubes={kept} framePath={frames[i % frames.length]}
            activeBoxByTube={activeBoxByTube(i % frames.length)} />
        </div>
      )}
    </section>
  );
}

function Stat({ label, value, color }: { label: string; value: string; color?: string }) {
  return (
    <div className="rounded-lg border border-slate-200 bg-white p-2">
      <div className="text-[10px] uppercase tracking-wide text-slate-500">{label}</div>
      <div className="text-sm font-medium" style={color ? { color } : undefined}>{value}</div>
    </div>
  );
}
```

- [ ] **Step 5: Typecheck + tests + commit**

Run: `npx tsc --noEmit` and `npm test` → all pass.

```bash
git add viewer/components/detail viewer/components/__tests__/BboxOverlay.test.tsx
git commit -m "feat(viewer): detail panel (frame viewer, bbox overlay, timeline, crops)"
```

---

## Task 12: Page assembly + README + final verification

**Files:** `app/page.tsx`, `app/layout.tsx`, `app/globals.css`, `viewer/README.md`, root `Makefile`/`eval` README pointer (optional).

- [ ] **Step 1: Assemble the master–detail page**

`app/page.tsx` (client component): on mount fetch sources → pick default (first);
fetch results + model config; derive `decision.aggregation`/`logistic_threshold`;
`showSlider = aggregation === "logistic" && rows.some(r => r.probability != null)`;
hold `threshold` (default from config), `source`, `selectedKey`, filter state. Compute
`view = showSlider ? applyThreshold(sourceRows, threshold) : sourceRows`, apply filters,
render `<ControlRail>`, `<SequenceTable>`, and `<DetailPanel>` (fetch the selected
sequence's `{details, view}` via `fetchSequence`, using the **original** row for the
header). Full code:

```tsx
"use client";
import { useEffect, useMemo, useState } from "react";
import { ControlRail } from "@/components/ControlRail";
import { SequenceTable } from "@/components/SequenceTable";
import { DetailPanel } from "@/components/detail/DetailPanel";
import { fetchModelConfig, fetchResults, fetchSequence, fetchSources } from "@/lib/api";
import { applyThreshold } from "@/lib/outcomes";
import type { BboxTubeDetails, ModelConfig, ResultRow, SequenceView } from "@/lib/types";

const DEFAULT_LOGISTIC_THRESHOLD = 0.5;

export default function Page() {
  const [sources, setSources] = useState<string[]>([]);
  const [source, setSource] = useState<string>("");
  const [allRows, setAllRows] = useState<ResultRow[]>([]);
  const [cfg, setCfg] = useState<ModelConfig>({});
  const [selectedKey, setSelectedKey] = useState<string | null>(null);
  const [seq, setSeq] = useState<{ details: BboxTubeDetails | null; view: SequenceView | null }>({ details: null, view: null });

  const defaultThr = (() => {
    const v = cfg.decision?.logistic_threshold;
    return typeof v === "number" ? v : DEFAULT_LOGISTIC_THRESHOLD;
  })();
  const [threshold, setThreshold] = useState(defaultThr);

  useEffect(() => { fetchSources().then((s) => { setSources(s); setSource(s[0] ?? ""); }); }, []);
  useEffect(() => { fetchResults().then(setAllRows); }, []);
  useEffect(() => { if (source) fetchModelConfig(source).then(setCfg); }, [source]);
  useEffect(() => { setThreshold(defaultThr); }, [defaultThr]);

  const sourceRows = useMemo(() => allRows.filter((r) => r.source === source), [allRows, source]);
  const showSlider = cfg.decision?.aggregation === "logistic" && sourceRows.some((r) => r.probability != null);
  const rows = useMemo(
    () => (showSlider ? applyThreshold(sourceRows, threshold) : sourceRows),
    [sourceRows, showSlider, threshold]);

  // keep a selection; default to the first row of the current source
  useEffect(() => {
    if (!rows.length) { setSelectedKey(null); return; }
    if (!selectedKey || !rows.some((r) => r.key === selectedKey)) setSelectedKey(rows[0].key);
  }, [rows, selectedKey]);

  useEffect(() => {
    if (source && selectedKey) fetchSequence(source, selectedKey).then(setSeq);
  }, [source, selectedKey]);

  const originalRow = sourceRows.find((r) => r.key === selectedKey) ?? null;

  return (
    <main className="flex h-screen">
      <ControlRail
        sources={sources} source={source} onSource={setSource}
        rows={rows} cfg={cfg}
        showSlider={showSlider} threshold={threshold} defaultThreshold={defaultThr}
        onThreshold={setThreshold} onReset={() => setThreshold(defaultThr)} />
      <div className="min-w-0 flex-1 p-4">
        <SequenceTable rows={rows} selectedKey={selectedKey} onSelect={setSelectedKey} />
      </div>
      <div className="w-[40%] shrink-0 border-l border-slate-200">
        {originalRow ? <DetailPanel details={seq.details} view={seq.view} row={originalRow} />
          : <p className="p-4 text-slate-400">No sequences.</p>}
      </div>
    </main>
  );
}
```

(Filter controls — ground truth / verdict / correctness / camera — can be added as a
small popover that narrows `rows` before the table; out of scope for the first green
build but trivial to add against `rows`.)

- [ ] **Step 2: README**

`viewer/README.md`: what it is (local read-only viewer over the eval reporting tree),
prerequisites (Node 22, the eval reporting tree present), `cp .env.local.example
.env.local` (set `DATA_ROOT`), `npm install`, `npm run dev` → http://localhost:3000,
`npm test`, `npm run build`. Note it consumes the same artifacts as the Streamlit app
and is the React port of it.

- [ ] **Step 3: Final verification**

Run from `viewer/`:

```bash
npm test            # all unit/component tests pass
npx tsc --noEmit    # clean
npm run lint        # clean
npm run build       # succeeds
```

Then a manual smoke against real data:

```bash
DATA_ROOT=../eval npm run dev &
sleep 4
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:3000   # 200
kill %1
```

Open http://localhost:3000, confirm: source selector lists train/val/pyro-annotator;
table is error-coloured; selecting a row (or ↑/↓) shows the frame viewer with bbox
overlay, tube timeline, and stabilized crops; the threshold slider re-decides the
cards + table live and reset works; the model-config panel shows the fields with hover
tooltips.

- [ ] **Step 4: Commit**

```bash
git add viewer/app viewer/README.md
git commit -m "feat(viewer): master-detail page assembly + README"
```

---

## Notes for the implementer

- **Server vs client components:** route handlers and `lib/paths.ts` are server-only
  (Node `fs`). All `components/*` that use state/effects declare `"use client"`. `lib/`
  pure helpers are isomorphic (no `fs`) and safe to import in client components.
- **Frame natural size:** the bbox overlay uses `preserveAspectRatio="none"` over a
  full-width `<img>`, so it stretches with the image — correct because bboxes are
  normalized to the image. The crop needs the natural size (captured via `onLoad`).
- **Parity:** the `lib/` ports mirror `eval/.../{outcomes,render}.py` and
  `core.crop`; the unit tests reuse the Python tests' cases. If a number disagrees,
  the Python is the source of truth.
- **Filters/sort** are intentionally deferred to keep the first build green; both layer
  onto `rows`/the table without touching the data or detail code.
