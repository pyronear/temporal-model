"use client";
import { useEffect, useMemo, useState } from "react";
import { ControlRail } from "@/components/ControlRail";
import { DetailPanel } from "@/components/detail/DetailPanel";
import { SequenceTable } from "@/components/SequenceTable";
import { fetchModelConfig, fetchResults, fetchSequence, fetchSources } from "@/lib/api";
import { applyThreshold } from "@/lib/outcomes";
import type { BboxTubeDetails, ModelConfig, ResultRow, SequenceView } from "@/lib/types";

const DEFAULT_LOGISTIC_THRESHOLD = 0.5;

export default function Page() {
  const [sources, setSources] = useState<string[]>([]);
  const [source, setSource] = useState("");
  const [allRows, setAllRows] = useState<ResultRow[]>([]);
  const [cfg, setCfg] = useState<ModelConfig>({});
  const [selectedKey, setSelectedKey] = useState<string | null>(null);
  const [seq, setSeq] = useState<{ details: BboxTubeDetails | null; view: SequenceView | null }>({
    details: null,
    view: null,
  });

  const defaultThr = useMemo(() => {
    const v = cfg.decision?.logistic_threshold;
    return typeof v === "number" ? v : DEFAULT_LOGISTIC_THRESHOLD;
  }, [cfg]);
  // Reset the threshold to the model default whenever that default changes (i.e. a
  // new source's config loads) — done during render, not in an effect.
  const [threshold, setThreshold] = useState(defaultThr);
  const [prevDefault, setPrevDefault] = useState(defaultThr);
  if (defaultThr !== prevDefault) {
    setPrevDefault(defaultThr);
    setThreshold(defaultThr);
  }

  useEffect(() => {
    fetchSources().then((s) => {
      setSources(s);
      setSource((cur) => cur || s[0] || "");
    });
  }, []);
  useEffect(() => {
    fetchResults().then(setAllRows);
  }, []);
  useEffect(() => {
    if (source) fetchModelConfig(source).then(setCfg);
  }, [source]);

  const sourceRows = useMemo(
    () => allRows.filter((r) => r.source === source),
    [allRows, source],
  );
  const showSlider =
    cfg.decision?.aggregation === "logistic" && sourceRows.some((r) => r.probability != null);
  const rows = useMemo(
    () => (showSlider ? applyThreshold(sourceRows, threshold) : sourceRows),
    [sourceRows, showSlider, threshold],
  );

  // Effective selection is derived: fall back to the first row when the user's
  // pick is absent (e.g. after switching source) — no state-sync effect needed.
  const selected = useMemo(() => {
    if (!rows.length) return null;
    if (selectedKey && rows.some((r) => r.key === selectedKey)) return selectedKey;
    return rows[0].key;
  }, [rows, selectedKey]);

  useEffect(() => {
    if (source && selected) fetchSequence(source, selected).then(setSeq);
  }, [source, selected]);

  const originalRow = sourceRows.find((r) => r.key === selected) ?? null;

  return (
    <main className="flex h-screen">
      <ControlRail
        sources={sources}
        source={source}
        onSource={setSource}
        rows={rows}
        cfg={cfg}
        showSlider={showSlider}
        threshold={threshold}
        defaultThreshold={defaultThr}
        onThreshold={setThreshold}
        onReset={() => setThreshold(defaultThr)}
      />
      <div className="min-w-0 flex-1 p-4">
        <SequenceTable rows={rows} selectedKey={selected} onSelect={setSelectedKey} />
      </div>
      <div className="w-[40%] shrink-0 border-l border-slate-200">
        {originalRow ? (
          <DetailPanel details={seq.details} view={seq.view} row={originalRow} />
        ) : (
          <p className="p-4 text-slate-400">No sequences.</p>
        )}
      </div>
    </main>
  );
}
