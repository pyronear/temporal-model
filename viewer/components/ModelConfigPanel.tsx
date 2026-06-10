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
  const d = cfg.decision ?? {};
  const mi = cfg.model_input ?? {};
  const inf = cfg.infer ?? {};
  const cl = cfg.classifier ?? {};
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
