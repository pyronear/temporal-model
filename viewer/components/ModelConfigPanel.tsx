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

interface Field {
  label: string;
  value: string;
  mono?: boolean;
  wide?: boolean;
}

export function ModelConfigPanel({ cfg }: { cfg: ModelConfig }) {
  if (!cfg || Object.keys(cfg).length === 0)
    return <p className="text-xs text-slate-400">model config unavailable</p>;
  const d = cfg.decision ?? {};
  const mi = cfg.model_input ?? {};
  const inf = cfg.infer ?? {};
  const cl = cfg.classifier ?? {};
  const round3 = (v: number | null | undefined) =>
    typeof v === "number" ? v.toFixed(3) : "—";
  const fields: Field[] = [
    { label: "detector", value: cfg.detector?.source ?? "—", mono: true, wide: true },
    { label: "variant", value: cfg.variant ?? "—" },
    { label: "train sha", value: (cfg.train_git_sha ?? "").slice(0, 8) || "—", mono: true },
    { label: "aggregation", value: String(d.aggregation ?? "—") },
    { label: "threshold", value: round3(d.threshold) },
    { label: "logistic threshold", value: round3(d.logistic_threshold) },
    { label: "stabilize", value: String(mi.stabilize ?? "—") },
    { label: "context factor", value: String(mi.context_factor ?? "—") },
    { label: "max frames", value: String(cl.max_frames ?? "—") },
    { label: "pad", value: `${inf.pad_strategy ?? "—"} / min ${inf.pad_to_min_frames ?? "—"}` },
  ];

  return (
    <details className="group rounded-lg border border-slate-200 bg-white">
      <summary className="flex cursor-pointer list-none items-center justify-between px-3 py-2 text-[11px] font-medium uppercase tracking-wide text-slate-500 select-none hover:text-slate-700">
        Model config
        <span className="text-slate-400 transition-transform group-open:rotate-90">▸</span>
      </summary>
      <dl className="grid grid-cols-2 gap-x-3 gap-y-2 border-t border-slate-100 px-3 py-2">
        {fields.map((f) => (
          <div key={f.label} title={HELP[f.label]} className={`cursor-help ${f.wide ? "col-span-2" : ""}`}>
            <dt className="text-[10px] uppercase tracking-wide text-slate-400">{f.label}</dt>
            <dd className={`text-sm break-words text-slate-800 ${f.mono ? "font-mono" : ""}`}>
              {f.value}
            </dd>
          </div>
        ))}
      </dl>
    </details>
  );
}
