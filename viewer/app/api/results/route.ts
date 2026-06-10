import { promises as fs } from "node:fs";
import { NextResponse } from "next/server";
import { reportingRoot, sourceDir } from "@/lib/paths";

export async function GET() {
  const root = reportingRoot();
  let names: string[] = [];
  try {
    names = await fs.readdir(root);
  } catch {
    return NextResponse.json([]);
  }
  const rows: unknown[] = [];
  for (const source of names) {
    try {
      const txt = await fs.readFile(`${sourceDir(source)}/results.json`, "utf8");
      rows.push(...JSON.parse(txt));
    } catch {
      /* skip sources without results */
    }
  }
  return NextResponse.json(rows);
}
