import { ModelConfigPanel } from "@/components/ModelConfigPanel";
import { PerfCards } from "@/components/PerfCards";
import { SourceSelect } from "@/components/SourceSelect";
import { ThresholdSlider } from "@/components/ThresholdSlider";
import type { ModelConfig, ResultRow } from "@/lib/types";

export function ControlRail(props: {
  sources: string[];
  source: string;
  onSource: (s: string) => void;
  rows: ResultRow[];
  cfg: ModelConfig;
  showSlider: boolean;
  threshold: number;
  defaultThreshold: number;
  onThreshold: (v: number) => void;
  onReset: () => void;
}) {
  return (
    <aside className="flex h-full w-72 shrink-0 flex-col gap-4 overflow-auto border-r border-slate-200 bg-slate-50 p-4">
      <SourceSelect
        sources={props.sources}
        value={props.source}
        onChange={props.onSource}
      />
      <PerfCards rows={props.rows} />
      {props.showSlider && (
        <ThresholdSlider
          value={props.threshold}
          defaultValue={props.defaultThreshold}
          onChange={props.onThreshold}
          onReset={props.onReset}
        />
      )}
      <div className="mt-auto" />
      <ModelConfigPanel cfg={props.cfg} />
    </aside>
  );
}
