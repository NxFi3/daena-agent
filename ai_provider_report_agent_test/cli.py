"""Command line interface for the ai_provider_report package.

Running ``python -m ai_provider_report`` prints a human‑readable
comparison report followed by a concise table of differences.
"""

from __future__ import annotations

import sys

from .report import build_report, compare_providers


def main(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]
    # For now we ignore any arguments and simply print the reports.
    print(build_report())
    print("\n" + compare_providers())


if __name__ == "__main__":
    main()
