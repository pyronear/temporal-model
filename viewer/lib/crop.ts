export interface SquareBox {
  x0: number;
  y0: number;
  side: number;
}

/**
 * Pixel-space square crop for a normalized window (cx,cy,w,h), expanded by
 * `context`. Mirrors core.crop: expand_bbox (scale w,h by context) ->
 * norm_bbox_to_pixel_square (side = max pixel extent, centred on the bbox).
 */
export function squarePixelBox(
  window: [number, number, number, number],
  imgW: number,
  imgH: number,
  context: number,
): SquareBox {
  const [cx, cy, w, h] = window;
  const ew = w * context;
  const eh = h * context;
  const side = Math.max(ew * imgW, eh * imgH);
  const cxPx = cx * imgW;
  const cyPx = cy * imgH;
  return { x0: cxPx - side / 2, y0: cyPx - side / 2, side };
}

export interface CropStyle {
  width: number;
  height: number;
  left: number;
  top: number;
}

/**
 * Style for an <img> inside a `displaySize`x`displaySize` overflow-hidden box so
 * the stabilized window square fills the box. Apply width/height (px) + absolute
 * left/top (px) to the <img>.
 */
export function stabilizedCropStyle(
  window: [number, number, number, number],
  imgW: number,
  imgH: number,
  context: number,
  displaySize: number,
): CropStyle {
  const { x0, y0, side } = squarePixelBox(window, imgW, imgH, context);
  const scale = displaySize / side;
  return {
    width: imgW * scale,
    height: imgH * scale,
    left: -x0 * scale,
    top: -y0 * scale,
  };
}
