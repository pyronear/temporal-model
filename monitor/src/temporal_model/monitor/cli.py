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
        from temporal_model.monitor.import_platform import (  # noqa: PLC0415
            import_platform,
        )

        client = AlertApiClient(AlertApiConfig.from_env())
        client.login()
        import_platform(
            client, args.store, args.date_from, args.date_to, force=args.force
        )
    elif args.command == "replay":
        raise SystemExit("replay is not implemented yet")


if __name__ == "__main__":
    main()
