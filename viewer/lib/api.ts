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
    `/api/sequence/${encodeURIComponent(source)}/${encodeURIComponent(key)}`,
  );
export const frameUrl = (relPath: string) => `/api/frame?path=${encodeURIComponent(relPath)}`;
