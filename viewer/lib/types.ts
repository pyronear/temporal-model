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
  num_tubes_kept: number;
  trigger_frame_index: number | null;
  organization_name: string | null;
  camera_name: string | null;
  started_at: string | null;
  // Triage-only (absent in eval/monitor trees): the per-sequence triage score
  // and its bucket. Their presence is what switches the viewer to triage mode.
  triage_score?: number | null;
  triage_bucket?: "review" | "unlabeled";
  // Monitor-only provenance (absent in eval reporting trees).
  replayed_probability?: number | null;
  replayed_decision?: Decision | null;
  replay_matches?: boolean | null;
  matched_window_frames?: number | null;
  temporal_model_version?: string | null;
  temporal_api_version?: string | null;
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
  preprocessing: {
    num_frames_input: number;
    num_truncated: number;
    padded_frame_indices: number[];
  };
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
  decision?: {
    aggregation?: string;
    threshold?: number;
    logistic_threshold?: number | null;
  } | null;
  infer?: { pad_strategy?: string; pad_to_min_frames?: number } | null;
  model_input?: { stabilize?: boolean; context_factor?: number } | null;
  classifier?: { max_frames?: number; backbone?: string } | null;
  tubes?: Record<string, unknown> | null;
  calibrator?: unknown;
  // Triage-only: the fixed triage split threshold written into model_config.json.
  threshold?: number | null;
}
