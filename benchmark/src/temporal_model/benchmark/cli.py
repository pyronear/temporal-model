"""`temporal-benchmark` CLI. Phase 1 implements the `core` subcommand."""

import argparse
import logging
import os
from pathlib import Path

import pandas as pd
import torch

from .machine import machine_info
from .report import write_report
from .resources import ResourceSampler
from .run_core import resolve_device, run_core


def _run_core_cmd(args: argparse.Namespace) -> None:
    threads = args.threads or os.cpu_count()
    if threads is not None:
        torch.set_num_threads(threads)
    device = resolve_device(args.device)

    with ResourceSampler(interval=args.sample_interval) as sampler:
        df = run_core(
            args.store,
            args.model,
            device=device,
            reps=args.reps,
            warmup=args.warmup,
            limit=args.limit,
        )
    resources = pd.DataFrame(sampler.timeline())

    stamp = args.timestamp
    machine = machine_info(device=device)
    out_dir = args.out / f"{machine['hostname']}-{stamp}"
    summary = write_report(df, resources, machine, out_dir)

    print(f"wrote {out_dir}")
    print(
        f"  p50 {summary['total_ms']['p50']}ms · "
        f"{summary['throughput']['sequences_per_sec']} seq/s · "
        f"{summary['n_failed']} failed"
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(prog="temporal-benchmark")
    sub = ap.add_subparsers(dest="command", required=True)

    core = sub.add_parser("core", help="in-process predict() stage breakdown")
    core.add_argument("--store", type=Path, default=Path("data/03_primary/sequences"))
    core.add_argument("--model", type=Path, required=True)
    core.add_argument("--device", default="auto", help="cpu, cuda, or auto")
    core.add_argument("--reps", type=int, default=5)
    core.add_argument("--warmup", type=int, default=3)
    core.add_argument("--limit", type=int, default=None)
    core.add_argument(
        "--threads",
        type=int,
        default=None,
        help="torch.set_num_threads(); default = all CPU cores (os.cpu_count())",
    )
    core.add_argument("--sample-interval", type=float, default=0.1)
    core.add_argument("--out", type=Path, default=Path("data/08_reporting"))
    core.add_argument(
        "--timestamp",
        default="run",
        help="label appended to the results dir (e.g. 20260609-1530)",
    )
    core.set_defaults(func=_run_core_cmd)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
