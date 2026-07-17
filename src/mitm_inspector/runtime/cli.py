"""CLI boundary; live proxy startup is intentionally deferred beyond S0."""

from __future__ import annotations

import argparse

from mitm_inspector import __version__


def main() -> int:
    parser = argparse.ArgumentParser(prog="mitm-inspector")
    parser.add_argument("--version", action="version", version=__version__)
    parser.parse_args()
    parser.error("live capture is not implemented in S0")
    return 2
