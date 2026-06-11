"use client";
import { useEffect, useRef, useState } from "react";
import { BboxOverlay, type OverlayBox } from "@/components/detail/BboxOverlay";

/** Fullscreen frame view with scroll-to-zoom + drag-to-pan. */
export function FrameModal({
  src,
  boxes,
  label,
  onClose,
}: {
  src: string;
  boxes: OverlayBox[];
  label: string;
  onClose: () => void;
}) {
  const [scale, setScale] = useState(1);
  const [off, setOff] = useState({ x: 0, y: 0 });
  const drag = useRef<{ x: number; y: number; ox: number; oy: number } | null>(
    null,
  );
  const viewport = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  // Non-passive wheel listener so we can preventDefault and zoom instead of scroll.
  useEffect(() => {
    const el = viewport.current;
    if (!el) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      setScale((s) => {
        const next = Math.min(
          8,
          Math.max(1, s * (e.deltaY < 0 ? 1.15 : 1 / 1.15)),
        );
        if (next === 1) setOff({ x: 0, y: 0 });
        return next;
      });
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, []);

  const reset = () => {
    setScale(1);
    setOff({ x: 0, y: 0 });
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/80"
      onClick={onClose}
    >
      <div className="absolute right-3 top-3 z-10 flex gap-2 text-xs text-white">
        <button
          onClick={(e) => {
            e.stopPropagation();
            reset();
          }}
          className="rounded bg-white/15 px-2 py-1 hover:bg-white/25"
        >
          reset
        </button>
        <button
          onClick={(e) => {
            e.stopPropagation();
            onClose();
          }}
          className="rounded bg-white/15 px-2 py-1 hover:bg-white/25"
        >
          ✕ close
        </button>
      </div>

      <div
        ref={viewport}
        className="relative max-h-[92vh] max-w-[92vw] overflow-hidden"
        onClick={(e) => e.stopPropagation()}
        onMouseDown={(e) => {
          drag.current = { x: e.clientX, y: e.clientY, ox: off.x, oy: off.y };
        }}
        onMouseMove={(e) => {
          if (!drag.current) return;
          setOff({
            x: drag.current.ox + (e.clientX - drag.current.x),
            y: drag.current.oy + (e.clientY - drag.current.y),
          });
        }}
        onMouseUp={() => {
          drag.current = null;
        }}
        onMouseLeave={() => {
          drag.current = null;
        }}
        style={{ cursor: scale > 1 ? "grab" : "default" }}
      >
        <div
          className="relative inline-block"
          style={{
            transform: `translate(${off.x}px, ${off.y}px) scale(${scale})`,
            transformOrigin: "center center",
          }}
        >
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={src}
            alt={label}
            draggable={false}
            className="block max-h-[92vh] max-w-[92vw] select-none object-contain"
          />
          <BboxOverlay boxes={boxes} />
        </div>
      </div>

      <span className="absolute bottom-3 left-1/2 -translate-x-1/2 rounded bg-black/60 px-2 py-1 text-xs text-white">
        {label} · scroll to zoom · drag to pan · Esc to close
      </span>
    </div>
  );
}
