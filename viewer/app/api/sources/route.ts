import { promises as fs } from "node:fs";
import { NextResponse } from "next/server";
import { MODEL_NAME, reportingRoot } from "@/lib/paths";

export async function GET() {
  const root = reportingRoot();
  let entries: string[] = [];
  try {
    entries = await fs.readdir(root);
  } catch {
    return NextResponse.json([]);
  }
  const sources: string[] = [];
  for (const name of entries) {
    try {
      await fs.access(`${root}/${name}/${MODEL_NAME}/results.json`);
      sources.push(name);
    } catch {
      /* skip dirs without a results.json */
    }
  }
  // pyro-annotator first, then alphabetical (mirrors the Streamlit default).
  sources.sort((a, b) =>
    a === "pyro-annotator" ? -1 : b === "pyro-annotator" ? 1 : a.localeCompare(b),
  );
  return NextResponse.json(sources);
}
