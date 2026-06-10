"use client";
import { useEffect, useState } from "react";
import { frameUrl } from "@/lib/api";
import { BboxOverlay, type OverlayBox } from "@/components/detail/BboxOverlay";
import { FrameModal } from "@/components/detail/FrameModal";

export function FrameViewer({
  frames,
  boxesByFrame,
  i,
  setI,
}: {
  frames: string[];
  boxesByFrame: (i: number) => OverlayBox[];
  i: number;
  setI: (updater: (prev: number) => number) => void;
}) {
  const n = frames.length;
  const [playing, setPlaying] = useState(true);
  const [zoom, setZoom] = useState(false);

  useEffect(() => {
    // Autoplay keeps running even while the modal is open, so the enlarged view
    // plays the sequence; the modal's zoom/pan is its own state and persists.
    if (!playing || n === 0) return;
    const t = setInterval(() => setI((p) => (p + 1) % n), 1000);
    return () => clearInterval(t);
  }, [playing, n, setI]);

  if (n === 0) return null;
  const cur = i % n;

  return (
    <div className="space-y-2">
      <div className="flex items-center gap-3">
        <button
          onClick={() => setPlaying((p) => !p)}
          className="rounded border border-slate-300 px-2 py-0.5 text-sm"
        >
          {playing ? "⏸ pause" : "▶ play"}
        </button>
        <input
          type="range"
          min={0}
          max={n - 1}
          value={cur}
          aria-label="frame"
          onChange={(e) => setI(() => parseInt(e.target.value, 10))}
          className="flex-1"
        />
        <span className="text-xs text-slate-500">
          {cur + 1}/{n}
        </span>
      </div>
      <div
        className="relative w-full cursor-zoom-in overflow-hidden bg-slate-100"
        onClick={() => setZoom(true)}
        title="click to enlarge"
      >
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img src={frameUrl(frames[cur])} alt={`frame ${cur}`} className="block w-full" />
        <BboxOverlay boxes={boxesByFrame(cur)} />
      </div>

      {zoom && (
        <FrameModal
          src={frameUrl(frames[cur])}
          boxes={boxesByFrame(cur)}
          label={`frame ${cur + 1}/${n}`}
          onClose={() => setZoom(false)}
        />
      )}
    </div>
  );
}
