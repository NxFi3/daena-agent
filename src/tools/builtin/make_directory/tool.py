from pathlib import Path

from src.models.ToolResult import ToolResult
from src.tools.Tool import Tool


class MakeDirectory(Tool):
    name = "make_directory"
    action = "create"
    description = (
        "Create directories and missing parents. Supply each literal path separately, "
        'for example {"paths":["public/css","public/js","routes","data"]}. '
        "Existing directories are accepted. No shell, brace or wildcard expansion is performed."
    )
    parameters = {
        "type": "object",
        "properties": {
            "paths": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "maxItems": 64,
            }
        },
        "required": ["paths"],
        "additionalProperties": False,
    }

    def validate(self, arguments):
        paths = arguments.get("paths")
        return (
            set(arguments) == {"paths"}
            and isinstance(paths, list)
            and 1 <= len(paths) <= 64
            and all(isinstance(p, str) and p.strip() for p in paths)
        )

    def execute(self, paths):
        # ToolManager resolves and authorizes the entire list before execution.
        targets = [Path(p) for p in paths]
        for target in targets:
            if any(p.exists() and not p.is_dir() for p in (target, *target.parents)):
                raise ValueError(f"A file occupies a directory path: {target}")
        created, existing = [], []
        try:
            for target in targets:
                if target.is_dir():
                    existing.append(str(target))
                    continue
                target.mkdir(parents=True, exist_ok=True)
                created.append(str(target))
        except OSError as exc:
            return ToolResult(
                success=False,
                name=self.name,
                content={
                    "created": created,
                    "existing": existing,
                    "error": {"type": "filesystem_error", "message": str(exc)},
                },
                effects=[{"action": "create", "target": p} for p in created],
            )
        return ToolResult(
            success=True,
            name=self.name,
            content={"created": created, "existing": existing},
            evidence={"created": created, "existing": existing},
            effects=[{"action": "create", "target": p} for p in created],
            summary=f"Created {len(created)} directories; {len(existing)} already existed.",
        )
