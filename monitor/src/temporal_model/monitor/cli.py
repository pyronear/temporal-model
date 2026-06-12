"""temporal-monitor CLI: import sequences from alert-api, replay them locally."""

from __future__ import annotations

import argparse
import datetime as dt
import logging
from pathlib import Path

DEFAULT_STORE = Path("data/01_raw/sequences")
DEFAULT_OUTPUT = Path("data/08_reporting")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="temporal-monitor", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    imp = sub.add_parser("import", help="fetch scored sequences from alert-api")
    today = dt.date.today().isoformat()
    yesterday = (dt.date.today() - dt.timedelta(days=1)).isoformat()
    imp.add_argument("--date-from", default=yesterday, help="YYYY-MM-DD (inclusive)")
    imp.add_argument("--date-to", default=today, help="YYYY-MM-DD (inclusive)")
    imp.add_argument("--store", type=Path, default=DEFAULT_STORE)
    imp.add_argument(
        "--force", action="store_true", help="re-download already-stored sequences"
    )
    imp.add_argument(
        "--all-orgs",
        action="store_true",
        help="scan the global sequence-id space to import every organization "
        "(admin token required); default imports only the account's own org",
    )
    imp.add_argument(
        "--seed-id",
        type=int,
        default=None,
        help="recent sequence id to seed the --all-orgs scan (only needed on "
        "an empty store when the own-org listing is empty)",
    )
    imp.add_argument(
        "--exclude-org",
        action="append",
        metavar="ORG",
        default=None,
        help="organization to skip (slug or name); repeatable",
    )

    rep = sub.add_parser(
        "replay", help="re-run stored sequences through their pinned api release"
    )
    rep.add_argument("--store", type=Path, default=DEFAULT_STORE)
    rep.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    rep.add_argument(
        "--compose-file",
        type=Path,
        # cli.py -> monitor(pkg) -> temporal_model -> src -> monitor/ root
        default=Path(__file__).resolve().parents[3] / "docker-compose.yml",
    )
    rep.add_argument(
        "--trigger-image",
        default=None,
        help=(
            "newer serving image used ONLY to compute trigger fields (same model.zip); "
            "merged only when its probability reproduces the pinned replay's"
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args(argv)
    if args.command == "import":
        # Imported lazily so `replay` does not need alert-api credentials.
        from temporal_model.monitor.alert_api import (  # noqa: PLC0415
            AlertApiClient,
            AlertApiConfig,
        )
        from temporal_model.monitor.import_alert_api import (  # noqa: PLC0415
            import_alert_api,
            import_all_orgs,
        )
        from temporal_model.monitor.store import slugify  # noqa: PLC0415

        exclude_orgs = (
            {slugify(o) for o in args.exclude_org} if args.exclude_org else None
        )
        client = AlertApiClient(AlertApiConfig.from_env())
        client.login()
        if args.all_orgs:
            import_all_orgs(
                client,
                args.store,
                args.date_from,
                args.date_to,
                force=args.force,
                seed_id=args.seed_id,
                exclude_orgs=exclude_orgs,
            )
        else:
            import_alert_api(
                client,
                args.store,
                args.date_from,
                args.date_to,
                force=args.force,
                exclude_orgs=exclude_orgs,
            )
    elif args.command == "replay":
        from temporal_model.monitor.replay import run_replay  # noqa: PLC0415

        run_replay(
            store_dir=args.store,
            output_dir=args.output_dir,
            compose_file=args.compose_file,
            trigger_image=args.trigger_image,
        )


if __name__ == "__main__":
    main()
