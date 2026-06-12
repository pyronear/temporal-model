import { promises as fs } from "node:fs";
import { NextResponse } from "next/server";
import { dataRoot, resolveFramePath } from "@/lib/paths";

export async function GET(req: Request) {
  const rel = new URL(req.url).searchParams.get("path");
  if (!rel) return new NextResponse("missing path", { status: 400 });
  let abs: string;
  try {
    abs = resolveFramePath(dataRoot(), rel);
  } catch {
    return new NextResponse("forbidden", { status: 400 });
  }
  try {
    const buf = await fs.readFile(abs);
    return new NextResponse(new Uint8Array(buf), {
      headers: {
        "Content-Type": "image/jpeg",
        "Cache-Control": "public, max-age=31536000, immutable",
      },
    });
  } catch {
    return new NextResponse("not found", { status: 404 });
  }
}
