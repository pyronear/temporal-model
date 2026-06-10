"use client";
import { useState } from "react";
import { frameUrl } from "@/lib/api";
import { stabilizedCropStyle } from "@/lib/crop";
import type { KeptTube } from "@/lib/types";

const SIZE = 200;
const CONTEXT = 2.0;

function TubeCrop({
  framePath,
  window,
  fallbackBbox,
}: {
  framePath: string;
  window: [number, number, number, number] | null;
  fallbackBbox: [number, number, number, number] | null;
}) {
  const [dim, setDim] = useState<{ w: number; h: number } | null>(null);
  const box = window ?? fallbackBbox;
  if (!box) return <p className="text-xs text-slate-400">inactive at this frame</p>;
  const style = dim ? stabilizedCropStyle(box, dim.w, dim.h, CONTEXT, SIZE) : null;
  return (
    <div className="relative overflow-hidden rounded-md bg-slate-100" style={{ width: SIZE, height: SIZE }}>
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img
        src={frameUrl(framePath)}
        alt=""
        onLoad={(e) => setDim({ w: e.currentTarget.naturalWidth, h: e.currentTarget.naturalHeight })}
        style={
          style
            ? {
                position: "absolute",
                width: style.width,
                height: style.height,
                left: style.left,
                top: style.top,
                maxWidth: "none",
              }
            : { opacity: 0 }
        }
      />
    </div>
  );
}

export function TubeCrops({
  tubes,
  framePath,
  activeBoxByTube,
}: {
  tubes: KeptTube[];
  framePath: string;
  activeBoxByTube: Map<number, [number, number, number, number]>;
}) {
  return (
    <div className="space-y-3">
      {tubes.map((t) => (
        <div key={t.tube_id} className="space-y-1">
          <div className="text-xs text-slate-500">
            T{t.tube_id} ·{" "}
            {t.probability != null ? `prob ${t.probability.toFixed(2)}` : `logit ${t.logit.toFixed(2)}`}
          </div>
          <TubeCrop
            framePath={framePath}
            window={t.stabilized_window}
            fallbackBbox={activeBoxByTube.get(t.tube_id) ?? null}
          />
        </div>
      ))}
    </div>
  );
}
