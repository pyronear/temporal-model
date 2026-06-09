"""Aggregate the raw benchmark table into a summary, plots, and a markdown report."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from temporal_model.core.stage_timer import STAGES  # noqa: E402


def _ok_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Successful rows that carry timing columns.

    Failed rows are recorded as bare ``{key, rep, failed}`` dicts, so an
    all-failed run yields a DataFrame with no ``total_ms`` column at all. Return
    an empty frame in that case rather than letting callers ``KeyError``.
    """
    ok = df[~df["failed"]] if "failed" in df else df
    if "total_ms" not in ok.columns:
        return ok.iloc[0:0]
    return ok


def _pct(series: pd.Series, q: float) -> float:
    return round(float(series.quantile(q)), 3)


def summarize(df: pd.DataFrame) -> dict:
    """Compute latency percentiles, throughput, and mean stage shares.

    Degrades gracefully when every sequence failed: latency/throughput/stage
    figures are zero and ``n_failed`` reflects the failures, so a run that
    blew up still produces a valid (loadable) report instead of a traceback.
    """
    ok = _ok_rows(df)
    has_data = not ok.empty

    total = ok["total_ms"] if has_data else pd.Series(dtype=float)
    mean_total_ms = float(total.mean()) if has_data else 0.0
    mean_frames = float(ok["frame_count"].mean()) if has_data else 0.0
    seq_per_sec = 1000.0 / mean_total_ms if mean_total_ms else 0.0

    stage_means = (
        {s: float(ok[f"{s}_ms"].mean()) for s in STAGES}
        if has_data
        else dict.fromkeys(STAGES, 0.0)
    )
    stage_total = sum(stage_means.values()) or 1.0

    latency = (
        {
            "p50": _pct(total, 0.50),
            "p90": _pct(total, 0.90),
            "p99": _pct(total, 0.99),
            "mean": round(mean_total_ms, 3),
        }
        if has_data
        else {"p50": 0.0, "p90": 0.0, "p99": 0.0, "mean": 0.0}
    )

    return {
        "n_sequences": int(df["key"].nunique()) if "key" in df else 0,
        "n_failed": int(df["failed"].sum()) if "failed" in df else 0,
        "total_ms": latency,
        "stage_ms_mean": {s: round(v, 3) for s, v in stage_means.items()},
        "stage_share_pct": {
            s: round(100.0 * v / stage_total, 1) for s, v in stage_means.items()
        },
        "throughput": {
            "sequences_per_sec": round(seq_per_sec, 3),
            "frames_per_sec": round(seq_per_sec * mean_frames, 3),
        },
    }


def _plot_latency_hist(df: pd.DataFrame, out: Path) -> None:
    ok = _ok_rows(df)
    if ok.empty:
        return
    fig, ax = plt.subplots()
    ok["total_ms"].plot.hist(bins=30, ax=ax)
    ax.set_xlabel("total latency (ms)")
    ax.set_title("Per-sequence latency distribution")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def _plot_stage_breakdown(summary: dict, out: Path) -> None:
    fig, ax = plt.subplots()
    means = summary["stage_ms_mean"]
    ax.bar(list(means), list(means.values()))
    ax.set_ylabel("mean ms")
    ax.set_title("Mean time per stage")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def _plot_latency_vs_frames(df: pd.DataFrame, out: Path) -> None:
    ok = _ok_rows(df)
    if ok.empty:
        return
    fig, ax = plt.subplots()
    ax.scatter(ok["frame_count"], ok["total_ms"], s=8, alpha=0.5)
    ax.set_xlabel("frame count")
    ax.set_ylabel("total latency (ms)")
    ax.set_title("Latency vs frame count")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def _plot_resources(resources: pd.DataFrame, out: Path) -> None:
    if resources.empty:
        return
    fig, ax = plt.subplots()
    ax.plot(resources["t"], resources["cpu_pct"], label="CPU %")
    if "gpu_util" in resources:
        ax.plot(resources["t"], resources["gpu_util"], label="GPU %")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("utilisation %")
    ax.set_title("Resource utilisation")
    ax.legend()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def write_report(
    df: pd.DataFrame,
    resources: pd.DataFrame,
    machine: dict,
    out_dir: Path,
) -> dict:
    """Write raw.parquet, resources.parquet, summary.json, plots, report.md."""
    out_dir.mkdir(parents=True, exist_ok=True)
    plots = out_dir / "plots"
    plots.mkdir(exist_ok=True)

    df.to_parquet(out_dir / "raw.parquet")
    resources.to_parquet(out_dir / "resources.parquet")

    summary = summarize(df)
    summary["machine"] = machine
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    _plot_latency_hist(df, plots / "latency_hist.png")
    _plot_stage_breakdown(summary, plots / "stage_breakdown.png")
    _plot_latency_vs_frames(df, plots / "latency_vs_frames.png")
    _plot_resources(resources, plots / "resources.png")

    (out_dir / "report.md").write_text(_render_markdown(summary))
    return summary


def _render_markdown(summary: dict) -> str:
    m = summary["machine"]
    lat = summary["total_ms"]
    tp = summary["throughput"]
    lines = [
        f"# Benchmark report — {m['hostname']}",
        "",
        "## Machine",
        f"- CPU: {m['cpu_model']} ({m['cpu_count_physical']} phys / "
        f"{m['cpu_count_logical']} logical, {m['torch_num_threads']} torch threads)",
        f"- RAM: {m['ram_total_gb']} GB",
        f"- GPU: {m['gpu_name'] or 'none'}",
        f"- device: {m['device']} · torch {m['torch_version']} · "
        f"python {m['python_version']}",
        "",
        "## Latency (total, ms)",
        f"- p50 {lat['p50']} · p90 {lat['p90']} · p99 {lat['p99']} · "
        f"mean {lat['mean']}",
        "",
        "## Throughput",
        f"- {tp['sequences_per_sec']} seq/s · {tp['frames_per_sec']} frames/s",
        "",
        "## Stage breakdown (mean ms · share)",
    ]
    for stage in STAGES:
        ms = summary["stage_ms_mean"][stage]
        pct = summary["stage_share_pct"][stage]
        lines.append(f"- {stage}: {ms} ms ({pct}%)")
    lines += [
        "",
        f"Sequences: {summary['n_sequences']} · failed: {summary['n_failed']}",
        "",
        "## Plots",
        "![latency](plots/latency_hist.png)",
        "![stages](plots/stage_breakdown.png)",
        "![latency vs frames](plots/latency_vs_frames.png)",
        "![resources](plots/resources.png)",
    ]
    return "\n".join(lines) + "\n"
