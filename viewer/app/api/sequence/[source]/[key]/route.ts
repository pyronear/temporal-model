import { promises as fs } from "node:fs";
import { NextResponse } from "next/server";
import { sourceDir } from "@/lib/paths";

export async function GET(
  _req: Request,
  { params }: { params: Promise<{ source: string; key: string }> },
) {
  const { source, key } = await params;
  const dir = sourceDir(source);
  const read = async (p: string) => {
    try {
      return JSON.parse(await fs.readFile(p, "utf8"));
    } catch {
      return null;
    }
  };
  const details = await read(`${dir}/details/${key}.json`);
  const view = await read(`${dir}/sequences/${key}.json`);
  return NextResponse.json({ details, view });
}
