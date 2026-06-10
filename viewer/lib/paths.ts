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
