import { promises as fs } from "node:fs";
import { NextResponse } from "next/server";
import { sourceDir } from "@/lib/paths";

export async function GET(
  _req: Request,
  { params }: { params: Promise<{ source: string }> },
) {
  const { source } = await params;
  try {
    const txt = await fs.readFile(
      `${sourceDir(source)}/model_config.json`,
      "utf8",
    );
    return NextResponse.json(JSON.parse(txt));
  } catch {
    return NextResponse.json({});
  }
}
