import { ModelConfigPanel } from "@/components/ModelConfigPanel";
import { MonitorCards } from "@/components/MonitorCards";
import { PerfCards } from "@/components/PerfCards";
import { SourceSelect } from "@/components/SourceSelect";
import { ThresholdSlider } from "@/components/ThresholdSlider";
import { TriageCards } from "@/components/TriageCards";
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
  monitorMode?: boolean;
  triageMode?: boolean;
  selectedOrganization?: string;
  onSelectOrganization?: (org: string) => void;
}) {
  return (
    <aside className="flex h-full w-72 shrink-0 flex-col gap-4 overflow-auto border-r border-slate-200 bg-slate-50 p-4">
      {/* A single source (the monitor's alert-api tree) needs no picker. */}
      {props.sources.length > 1 && (
        <SourceSelect
          sources={props.sources}
          value={props.source}
          onChange={props.onSource}
        />
      )}
      {props.triageMode ? (
        <TriageCards
          rows={props.rows}
          threshold={props.cfg.threshold}
          selectedOrganization={props.selectedOrganization}
          onSelectOrganization={props.onSelectOrganization}
        />
      ) : props.monitorMode ? (
        <MonitorCards
          rows={props.rows}
          selectedOrganization={props.selectedOrganization}
          onSelectOrganization={props.onSelectOrganization}
        />
      ) : (
        <PerfCards rows={props.rows} />
      )}
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
