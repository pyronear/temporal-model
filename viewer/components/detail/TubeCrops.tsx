"use client";
import { useState } from "react";
import { frameUrl } from "@/lib/api";
import { stabilizedCropStyle } from "@/lib/crop";
import type { TriggerState } from "@/lib/details";
import type { KeptTube } from "@/lib/types";

function TriggerBadge({ state }: { state: TriggerState }) {
  if (state === "decisive")
    return <span className="font-semibold text-amber-600">⚡ triggered</span>;
  if (state === "would")
    return <span className="text-slate-400">would trigger</span>;
  return null;
}

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
  if (!box)
    return <p className="text-xs text-slate-400">inactive at this frame</p>;
  const style = dim
    ? stabilizedCropStyle(box, dim.w, dim.h, CONTEXT, SIZE)
    : null;
  return (
    <div
      className="relative overflow-hidden bg-slate-100"
      style={{ width: SIZE, height: SIZE }}
    >
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img
        src={frameUrl(framePath)}
        alt=""
        onLoad={(e) =>
          setDim({
            w: e.currentTarget.naturalWidth,
            h: e.currentTarget.naturalHeight,
          })
        }
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
  triggerByTube,
  colorByTube,
}: {
  tubes: KeptTube[];
  framePath: string;
  activeBoxByTube: Map<number, [number, number, number, number]>;
  triggerByTube: Map<number, TriggerState>;
  colorByTube: Map<number, string>;
}) {
  return (
    <div className="space-y-2">
      {tubes.map((t) => {
        const tr = triggerByTube.get(t.tube_id) ?? null;
        const decisive = tr === "decisive";
        const color = colorByTube.get(t.tube_id) ?? "#64748b";
        return (
          <div
            key={t.tube_id}
            className={`space-y-1 rounded-lg p-2 ${
              decisive ? "bg-amber-50 ring-1 ring-amber-300" : ""
            }`}
          >
            <div className="flex items-baseline gap-2 text-xs text-slate-500">
              <span className="inline-flex items-center gap-1.5">
                <span
                  className="inline-block h-2 w-2 shrink-0 rounded-full"
                  style={{ background: color }}
                />
                <span className="font-semibold" style={{ color }}>
                  T{t.tube_id}
                </span>
                <span>
                  ·{" "}
                  {t.probability != null
                    ? `prob ${t.probability.toFixed(2)}`
                    : `logit ${t.logit.toFixed(2)}`}
                </span>
              </span>
              <TriggerBadge state={tr} />
            </div>
            <TubeCrop
              framePath={framePath}
              window={t.stabilized_window}
              fallbackBbox={activeBoxByTube.get(t.tube_id) ?? null}
            />
          </div>
        );
      })}
    </div>
  );
}
