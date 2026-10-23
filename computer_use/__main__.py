"""JSON CLI and persistent JSONL bridge for local agent integrations."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

from .computer_use import ComputerUseTool
from .backend import create_backend


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Daena Computer Use: snapshot-bound desktop control"
    )
    parser.add_argument("--workspace", default=str(Path.cwd()))
    parser.add_argument("--backend", choices=["cua", "x11"], default=None)
    parser.add_argument("--session", default="cli")
    parser.add_argument(
        "command",
        nargs="?",
        choices=["doctor", "capture", "call", "serve", "viewer"],
        default="doctor",
    )
    parser.add_argument(
        "--json", help="Request JSON. Omit to read it from stdin (recommended for typed text)."
    )
    args = parser.parse_args(argv)
    if args.command == "viewer":
        from .viewer import main as view

        return view([args.workspace])
    tool = ComputerUseTool(create_backend(args.backend), workspace=args.workspace)
    tool.set_context_store(None, lambda: args.session)
    if args.command == "serve":
        for line in sys.stdin:
            try:
                request = json.loads(line)
                result = asdict(tool.execute(**request))
            except (ValueError, TypeError) as exc:
                result = {"success": False, "error": str(exc)}
            print(json.dumps(result, ensure_ascii=False), flush=True)
        return 0
    try:
        request = (
            json.loads(args.json or sys.stdin.read())
            if args.command == "call"
            else {"action": args.command}
        )
        result = tool.execute(**request)
    except (ValueError, TypeError) as exc:
        parser.error(str(exc))
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
