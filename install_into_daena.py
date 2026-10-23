"""Install this source package into a Daena checkout, with backups and optional hooks.

No dependencies are installed and no network/model call is made by this script.
"""

import argparse
import ast
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import uuid


def method(tree, class_name, method_name):
    cls = next((n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name), None)
    if cls is None:
        raise ValueError(f"Missing class {class_name}; host is incompatible with automatic wiring.")
    node = next(
        (n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method_name), None
    )
    if node is None:
        raise ValueError(
            f"Missing {class_name}.{method_name}; use the manual hooks in docs/integration.md."
        )
    return node


def add_hook(source, marker, class_name, name, snippet, *, where="end"):
    if marker in source:
        return source
    node = method(ast.parse(source), class_name, name)
    lines = source.splitlines(keepends=True)
    if where == "start":
        first = node.body[0]
        is_doc = (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        )
        position = first.end_lineno if is_doc else first.lineno - 1
    elif where == "before_invoke":
        positions = [
            n.lineno - 1
            for n in ast.walk(node)
            if isinstance(n, ast.Assign)
            and isinstance(n.value, ast.Call)
            and isinstance(n.value.func, ast.Name)
            and n.value.func.id == "invoke"
        ]
        if len(positions) != 1:
            raise ValueError("Cannot identify one provider invocation; no files were changed.")
        position = positions[0]
    else:
        position = node.end_lineno
    lines.insert(position, f"        # {marker}\n" + snippet + "\n")
    updated = "".join(lines)
    ast.parse(updated)
    return updated


def prepare_hooks(repo):
    changes = {}
    path = Path("src/tools/ToolManager.py")
    changes[path] = add_hook(
        (repo / path).read_text(),
        "DAENA_COMPUTER_BIND_V1",
        "ToolManager",
        "_bind_runtime_services",
        "        from src.tools.builtin.computer_use.integration import bind_manager\n        bind_manager(self)",
    )
    path = Path("src/agent/agentloop.py")
    source = add_hook(
        (repo / path).read_text(),
        "DAENA_COMPUTER_RECEIPT_V1",
        "Loop",
        "_apply_result",
        "        from src.tools.builtin.computer_use.integration import remember_result\n        remember_result(self, call, result)",
        where="start",
    )
    source = add_hook(
        source,
        "DAENA_COMPUTER_VISION_V1",
        "Loop",
        "_generate_next_action",
        "        from src.tools.builtin.computer_use.integration import append_pending_visual\n        messages = append_pending_visual(self, messages)",
        where="before_invoke",
    )
    changes[path] = source
    path = Path("src/engine/providers/builtin/gemini/geminiprovider.py")
    source = (repo / path).read_text()
    marker = "DAENA_COMPUTER_GEMINI_V1"
    if marker not in source:
        tree = ast.parse(source)
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
        convert = method(tree, cls.name, "_convert_messages")
        loops = [n for n in convert.body if isinstance(n, ast.For)]
        if (
            len(loops) != 1
            or not isinstance(loops[0].target, ast.Name)
            or loops[0].target.id != "m"
        ):
            raise ValueError("Unsupported Gemini conversion layout; no files were changed.")
        lines = source.splitlines(keepends=True)
        lines.insert(
            loops[0].body[0].lineno - 1,
            f"            # {marker}\n"
            "            if m.get('role') == 'user' and m.get('computer_image'):\n"
            "                from src.tools.builtin.computer_use.integration import gemini_image_parts\n"
            "                contents.append(types.Content(role='user', parts=gemini_image_parts(m, types)))\n"
            "                continue\n",
        )
        source = "".join(lines)
        ast.parse(source)
    changes[path] = source
    return changes


def default_tools(repo):
    tree = ast.parse((repo / "src/security/Policy.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "DEFAULT_ALLOWED_TOOLS" for t in node.targets
        ):
            value = ast.literal_eval(node.value)
            if isinstance(value, (set, list, tuple)) and all(isinstance(x, str) for x in value):
                return sorted(value)
    raise ValueError("Cannot read host's default tool scope; no files were changed.")


def install(repo, *, wire_runtime=False, enable=False):
    repo = Path(repo).expanduser().resolve()
    if not (repo / "src/tools/Tool.py").is_file():
        raise ValueError("--repo must point to a Daena checkout containing src/tools/Tool.py")
    source = Path(__file__).resolve().parent / "computer_use"
    target = repo / "src/tools/builtin/computer_use"
    changes = prepare_hooks(repo) if wire_runtime else {}
    if enable:
        defaults = default_tools(repo)
        for relative in (Path("config.json"), Path("src/default_config.json")):
            path = repo / relative
            if not path.exists():
                continue
            config = json.loads(path.read_text())
            security = config.setdefault("security", {})
            allowed = security.get("allowed_tools", defaults)
            if not isinstance(allowed, list):
                raise ValueError(f"{relative}: security.allowed_tools must be a list")
            security["allowed_tools"] = list(dict.fromkeys([*allowed, "computer_use"]))
            changes[relative] = json.dumps(config, ensure_ascii=False, indent=2) + "\n"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]
    backup = repo / ".daena-computer-use-backups" / stamp
    backup.mkdir(parents=True)
    for relative in changes:
        destination = backup / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repo / relative, destination)
    old = backup / "src/tools/builtin/computer_use"
    had_package = target.exists()
    if had_package:
        old.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(target, old)
    try:
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(source, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        for relative, text in changes.items():
            (repo / relative).write_text(text, encoding="utf-8")
    except BaseException:
        if target.exists():
            shutil.rmtree(target)
        if had_package:
            shutil.copytree(old, target)
        for relative in changes:
            shutil.copy2(backup / relative, repo / relative)
        raise
    manifest = {
        "repo": str(repo),
        "had_package": had_package,
        "files": [str(p) for p in changes],
        "wire_runtime": wire_runtime,
        "enable": enable,
    }
    (backup / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return backup


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument(
        "--wire-runtime",
        action="store_true",
        help="Add workspace/cancellation and one-shot screenshot forwarding hooks; use for the stock Daena 0.6 host.",
    )
    parser.add_argument(
        "--enable",
        action="store_true",
        help="Add computer_use to config.json and bundled default_config.json tool lists.",
    )
    args = parser.parse_args()
    try:
        backup = install(args.repo, wire_runtime=args.wire_runtime, enable=args.enable)
    except (ValueError, OSError, StopIteration, SyntaxError) as exc:
        parser.exit(1, f"Installation failed: {exc}\n")
    print(
        f"Installed. Backup: {backup}\nRestart Daena. Existing workspace profiles keep their own tool selection."
    )


if __name__ == "__main__":
    main()
