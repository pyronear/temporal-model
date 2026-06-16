"""triage CLI: ``pull`` (read-only fetch) and ``score`` (model + report)."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from temporal_model.triage.annotator_api import AnnotatorApiClient, AnnotatorApiConfig
from temporal_model.triage.model_config import read_model_config
from temporal_model.triage.pull import DEFAULT_WORKERS, pull_unannotated
from temporal_model.triage.report import write_triage_report
from temporal_model.triage.score import score_sequences

DEFAULT_STORE = Path("data/01_raw/sequences")
DEFAULT_OUTPUT = Path("data/08_reporting")
DEFAULT_MODEL_ZIP = Path("../api/models/model.zip")
DEFAULT_THRESHOLD = 0.35


def _build_client() -> AnnotatorApiClient:
    client = AnnotatorApiClient(AnnotatorApiConfig.from_env())
    client.login()
    return client


def _load_model(model_zip: Path, device: str | None):
    # Local import: keep torch/ultralytics out of the `pull` path.
    from temporal_model.core.model import BboxTubeTemporalModel  # noqa: PLC0415

    return BboxTubeTemporalModel.from_archive(model_zip, device=device)


def _cmd_pull(args: argparse.Namespace) -> None:
    client = _build_client()
    counts = pull_unannotated(
        client,
        args.store,
        processing_stage=args.stage,
        limit=args.limit,
        page_size=args.page_size,
        workers=args.workers,
    )
    print(json.dumps(counts, indent=2))


def _cmd_score(args: argparse.Namespace) -> None:
    model = _load_model(args.model_zip, args.device)
    scored, dropped = score_sequences(model, args.store, threshold=args.threshold)
    write_triage_report(
        args.output_dir,
        scored,
        dropped=dropped,
        threshold=args.threshold,
        model_config=read_model_config(args.model_zip),
    )
    n_low = sum(1 for s in scored if s.bucket == "unlabeled")
    n_high = sum(1 for s in scored if s.bucket == "review")
    print(
        json.dumps(
            {
                "scored": len(scored),
                "dropped": len(dropped),
                "unlabeled": n_low,
                "review": n_high,
                "threshold": args.threshold,
            },
            indent=2,
        )
    )


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(prog="temporal-triage", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_pull = sub.add_parser("pull", help="fetch unannotated sequences (read-only)")
    p_pull.add_argument("--store", type=Path, default=DEFAULT_STORE)
    p_pull.add_argument(
        "--stage",
        default="ready_to_annotate",
        help="annotator processing_stage to pull (default: ready_to_annotate)",
    )
    p_pull.add_argument("--limit", type=int, default=None, help="cap sequences pulled")
    p_pull.add_argument("--page-size", type=int, default=100)
    p_pull.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"concurrent frame downloads per sequence (default: {DEFAULT_WORKERS})",
    )
    p_pull.set_defaults(func=_cmd_pull)

    p_score = sub.add_parser("score", help="score the store + write report")
    p_score.add_argument("--store", type=Path, default=DEFAULT_STORE)
    p_score.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    p_score.add_argument("--model-zip", type=Path, default=DEFAULT_MODEL_ZIP)
    p_score.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    p_score.add_argument("--device", default=None, help="cuda/mps/cpu (default auto)")
    p_score.set_defaults(func=_cmd_score)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
