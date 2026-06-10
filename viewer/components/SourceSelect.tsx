"use client";
import {
  Listbox,
  ListboxButton,
  ListboxOption,
  ListboxOptions,
} from "@headlessui/react";

// Human-readable blurb per known source; unknown sources just show their name.
const SOURCE_INFO: Record<string, string> = {
  "pyro-annotator": "human-labeled platform sequences (smoke / fp / unknown)",
  val: "held-out validation split",
  train: "training split",
};

const blurb = (s: string) => SOURCE_INFO[s] ?? "";

export function SourceSelect({
  sources,
  value,
  onChange,
}: {
  sources: string[];
  value: string;
  onChange: (s: string) => void;
}) {
  return (
    <div className="space-y-1">
      <label className="text-[11px] uppercase tracking-wide text-slate-500">
        source
      </label>
      <Listbox value={value} onChange={onChange}>
        <div className="relative">
          <ListboxButton className="flex w-full items-center justify-between rounded-md border border-slate-300 bg-white px-2.5 py-1.5 text-left text-sm hover:border-slate-400 focus:outline-none focus-visible:ring-2 focus-visible:ring-slate-300">
            <span className="min-w-0">
              <span className="block truncate font-medium text-slate-800">
                {value}
              </span>
              {blurb(value) && (
                <span className="block truncate text-[11px] text-slate-400">
                  {blurb(value)}
                </span>
              )}
            </span>
            <span className="ml-2 shrink-0 text-slate-400">▾</span>
          </ListboxButton>
          <ListboxOptions className="absolute z-20 mt-1 w-full overflow-hidden rounded-md border border-slate-200 bg-white shadow-lg focus:outline-none">
            {sources.map((s) => (
              <ListboxOption
                key={s}
                value={s}
                className="cursor-pointer px-2.5 py-2 data-[focus]:bg-slate-50 data-[selected]:bg-slate-100"
              >
                <span className="block text-sm font-medium text-slate-800">
                  {s}
                </span>
                {blurb(s) && (
                  <span className="block text-[11px] leading-snug text-slate-500">
                    {blurb(s)}
                  </span>
                )}
              </ListboxOption>
            ))}
          </ListboxOptions>
        </div>
      </Listbox>
    </div>
  );
}
