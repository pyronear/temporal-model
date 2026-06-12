"use client";
import { useEffect, useMemo, useState } from "react";
import { ControlRail } from "@/components/ControlRail";
import { DetailPanel } from "@/components/detail/DetailPanel";
import { FilterBar } from "@/components/FilterBar";
import { SequenceTable } from "@/components/SequenceTable";
import {
  fetchModelConfig,
  fetchResults,
  fetchSequence,
  fetchSources,
} from "@/lib/api";
import {
  applyFilters,
  cameraOptions,
  defaultFilters,
  organizationOptions,
  type Filters,
} from "@/lib/filters";
import { applyThreshold } from "@/lib/outcomes";
import { nextSort, sortRows, type Sort } from "@/lib/sort";
import type {
  BboxTubeDetails,
  ModelConfig,
  ResultRow,
  SequenceView,
} from "@/lib/types";

const DEFAULT_LOGISTIC_THRESHOLD = 0.5;

export default function Page() {
  const [sources, setSources] = useState<string[]>([]);
  const [source, setSource] = useState("");
  const [allRows, setAllRows] = useState<ResultRow[]>([]);
  const [cfg, setCfg] = useState<ModelConfig>({});
  const [selectedKey, setSelectedKey] = useState<string | null>(null);
  const [seq, setSeq] = useState<{
    details: BboxTubeDetails | null;
    view: SequenceView | null;
  }>({
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
  const monitorMode = useMemo(
    () => sourceRows.some((r) => r.replayed_probability !== undefined),
    [sourceRows],
  );
  // Monitor rows carry production's verdict — never re-decided locally, so
  // the threshold slider is eval-only.
  const showSlider =
    !monitorMode &&
    cfg.decision?.aggregation === "logistic" &&
    sourceRows.some((r) => r.probability != null);
  const rows = useMemo(
    () => (showSlider ? applyThreshold(sourceRows, threshold) : sourceRows),
    [sourceRows, showSlider, threshold],
  );

  // Filters reset when the source changes (camera options are source-specific).
  const [filters, setFilters] = useState<Filters>(defaultFilters);
  const [prevFilterSource, setPrevFilterSource] = useState(source);
  if (source !== prevFilterSource) {
    setPrevFilterSource(source);
    setFilters(defaultFilters());
  }
  const cameras = useMemo(
    () =>
      cameraOptions(
        filters.organization === "all"
          ? sourceRows
          : sourceRows.filter(
              (r) => r.organization_name === filters.organization,
            ),
      ),
    [sourceRows, filters.organization],
  );
  const organizations = useMemo(
    () => organizationOptions(sourceRows),
    [sourceRows],
  );
  const [sort, setSort] = useState<Sort | null>(null);
  const tableRows = useMemo(
    () => sortRows(applyFilters(rows, filters), sort),
    [rows, filters, sort],
  );

  // Effective selection is derived: fall back to the first visible row when the
  // user's pick is filtered out / absent — no state-sync effect needed.
  const selected = useMemo(() => {
    if (!tableRows.length) return null;
    if (selectedKey && tableRows.some((r) => r.key === selectedKey))
      return selectedKey;
    return tableRows[0].key;
  }, [tableRows, selectedKey]);

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
        monitorMode={monitorMode}
      />
      <div className="flex min-w-0 flex-1 flex-col p-4">
        <FilterBar
          filters={filters}
          cameras={cameras}
          organizations={organizations}
          onChange={setFilters}
          shownCount={tableRows.length}
          totalCount={rows.length}
          monitorMode={monitorMode}
        />
        <div className="min-h-0 flex-1">
          <SequenceTable
            rows={tableRows}
            selectedKey={selected}
            onSelect={setSelectedKey}
            sort={sort}
            onSort={(col) => setSort((cur) => nextSort(cur, col))}
            monitorMode={monitorMode}
          />
        </div>
      </div>
      <div className="w-[40%] shrink-0 border-l border-slate-200">
        {originalRow ? (
          <DetailPanel
            details={seq.details}
            view={seq.view}
            row={originalRow}
          />
        ) : (
          <p className="p-4 text-slate-400">No sequences.</p>
        )}
      </div>
    </main>
  );
}
