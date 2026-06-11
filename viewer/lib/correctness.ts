import type { Decision, Outcome } from "@/lib/types";

export const correctnessLabel = (o: Outcome): string =>
  ({
    "kept-smoke": "smoke kept",
    "discarded-fp": "fp filtered",
    "kept-fp": "false alarm",
    "discarded-smoke": "missed smoke",
    "n/a": "—",
  })[o] ?? o;

export interface Tokens {
  bg: string;
  dot: string;
  text: string;
}

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
  if (outcome === "n/a")
    return verdict === "keep" ? UNKNOWN_KEEP : outcomeTokens["n/a"];
  return outcomeTokens[outcome];
}
