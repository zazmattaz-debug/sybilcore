"""Launcher for the SybilCore REST API.

Run with:
    python -m sybilcore.api.run_server
    # or:
    python -m sybilcore.api.run_server --port 8765 --host 0.0.0.0

Environment variables:
    SYBILCORE_API_KEY  Optional bearer token to require on all requests.
    SYBILCORE_PORT     Default port if --port is omitted (defaults to 8765).
    SYBILCORE_HOST     Default host if --host is omitted (defaults to 0.0.0.0).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

DEFAULT_PORT = 8765
DEFAULT_HOST = "0.0.0.0"  # noqa: S104  — opt-in for SDK consumers


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the SybilCore API server")
    parser.add_argument(
        "--host",
        default=os.environ.get("SYBILCORE_HOST", DEFAULT_HOST),
        help="Bind host (default: %(default)s)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("SYBILCORE_PORT", DEFAULT_PORT)),
        help="Bind port (default: %(default)s)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable uvicorn auto-reload (development)",
    )
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["critical", "error", "warning", "info", "debug", "trace"],
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    )

    try:
        import uvicorn
    except ImportError:
        print(
            "uvicorn is required to run the SybilCore API server.\n"
            "Install with: pip install 'uvicorn[standard]'",
            file=sys.stderr,
        )
        return 2

    # Eagerly import the app so brain loading happens before bind.
    from sybilcore.api.server import app  # noqa: F401

    uvicorn.run(
        "sybilcore.api.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
