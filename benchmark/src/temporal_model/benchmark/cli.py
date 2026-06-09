"""`temporal-benchmark` CLI. Phase 1 implements the `core` subcommand."""

import argparse
import logging
import os
from pathlib import Path

import pandas as pd
import torch

from .machine import machine_info
from .report import write_api_report, write_report
from .resources import ResourceSampler
from .run_api import run_api
from .run_core import resolve_device, run_core


def _run_api_cmd(args: argparse.Namespace) -> None:
    with ResourceSampler(interval=args.sample_interval) as sampler:
        df = run_api(
            args.store,
            args.url,
            passes=tuple(args.passes.split(",")),
            warmup=args.warmup,
            limit=args.limit,
            warm_min_frames=args.warm_min_frames,
        )
    resources = pd.DataFrame(sampler.timeline())
    machine = machine_info(device="cpu")
    out_dir = args.out / f"{machine['hostname']}-api-{args.timestamp}"
    summary = write_api_report(df, resources, machine, out_dir)
    print(f"wrote {out_dir}")
    for name, p in summary["passes"].items():
        print(f"  {name}: p50 {p['e2e_ms']['p50']}ms · {p['n_errors']} errors")


def _run_core_cmd(args: argparse.Namespace) -> None:
    threads = args.threads if args.threads is not None else os.cpu_count()
    if threads:
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

    api = sub.add_parser("api", help="API end-to-end (HTTP) benchmark")
    api.add_argument("--url", default="http://localhost:8000")
    api.add_argument("--store", type=Path, default=Path("data/03_primary/sequences"))
    api.add_argument("--passes", default="cold,warm", help="comma list: cold,warm")
    api.add_argument("--warmup", type=int, default=3)
    api.add_argument("--limit", type=int, default=None)
    api.add_argument("--warm-min-frames", type=int, default=3)
    api.add_argument("--sample-interval", type=float, default=0.1)
    api.add_argument("--out", type=Path, default=Path("data/08_reporting"))
    api.add_argument("--timestamp", default="run")
    api.set_defaults(func=_run_api_cmd)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
