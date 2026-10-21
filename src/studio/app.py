from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Daena Studio — desktop agent workspace"
    )
    parser.add_argument("--cwd", default=str(Path.cwd()), help="Agent workspace")
    parser.add_argument(
        "--config",
        type=Path,
        help="Initial JSON config; workspace profile takes precedence",
    )
    parser.add_argument("--model", help="Override the selected profile model")
    parser.add_argument("--no-think", action="store_true")
    from src import __version__

    parser.add_argument(
        "--version", action="version", version=f"Daena Studio {__version__}"
    )
    args = parser.parse_args(argv)
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError as exc:
        parser.exit(
            1,
            f"Cannot load PySide6: {exc}\nInstall the desktop extra: python -m pip install 'daena-agent[desktop]'\nFor Linux native libraries, see docs/studio.md.\n",
        )
    from src.studio.window import StudioWindow

    config = json.loads(args.config.read_text()) if args.config else None
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("Daena Studio")
    app.setOrganizationName("Daena")
    try:
        window = StudioWindow(args.cwd, config)
    except Exception as exc:
        parser.exit(1, f"Cannot open workspace: {exc}\n")
    if args.model or args.no_think:
        from copy import deepcopy

        selected = deepcopy(window.config)
        provider = selected.setdefault("llm", {}).setdefault("provider_config", {})
        if args.model:
            provider["model_name"] = args.model
        if args.no_think:
            provider.setdefault("generation_config", {})["think"] = False
        window.config = selected
        window.settings.populate(selected)
        window._update_badge()
    window.show()
    return app.exec()
