"""Resolve an installed Chromium executable without downloading or changing profiles."""

import os
from pathlib import Path
import shutil
import sys

from .contracts import BrowserError


def resolve_executable(config, bundled=None):
    explicit = config.get("executable_path")
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_file() or not os.access(path, os.X_OK):
            raise BrowserError(
                "executable_missing",
                f"Configured browser executable is unavailable: {path}",
            )
        return str(path), "configured"
    if bundled and Path(bundled).is_file() and os.access(bundled, os.X_OK):
        return str(bundled), "playwright"
    for name in (
        "google-chrome-stable",
        "google-chrome",
        "chromium",
        "chromium-browser",
        "chrome",
        "msedge",
    ):
        path = shutil.which(name)
        if path:
            return path, "system"
    candidates = []
    if sys.platform == "darwin":
        candidates = [
            Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
            Path("/Applications/Chromium.app/Contents/MacOS/Chromium"),
        ]
    elif os.name == "nt":
        for variable in ("PROGRAMFILES", "PROGRAMFILES(X86)", "LOCALAPPDATA"):
            if os.environ.get(variable):
                root = Path(os.environ[variable])
                candidates.extend(
                    [
                        root / "Google/Chrome/Application/chrome.exe",
                        root / "Microsoft/Edge/Application/msedge.exe",
                    ]
                )
    for path in candidates:
        if path.is_file() and os.access(path, os.X_OK):
            return str(path), "system"
    raise BrowserError(
        "executable_missing",
        "No Chromium browser was found. Install Chrome/Chromium, set browser.executable_path, or run python -m playwright install chromium. Firefox is not supported by this Chromium backend.",
    )
