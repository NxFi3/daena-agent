from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.models.ToolResult import ToolResult
from src.tools.Tool import Tool


@dataclass
class PatchOperation:
    operation: str
    path: str
    hunks: list[list[str]]
    content: str | None = None


class ApplyPatch(Tool):
    """
    Apply Codex-style file patches.

    Supported operations:

        *** Add File: path/to/file
        *** Update File: path/to/file
        *** Delete File: path/to/file

    Patch format:

        *** Begin Patch
        *** Add File: README.md
        +# Hello
        +
        +World

        *** Update File: app.py
        @@
         print("hello")
        -print("old")
        +print("new")

        *** Delete File: old.py
        *** End Patch

    The parser intentionally keeps the Codex-style format, but is tolerant
    of common LLM formatting mistakes such as:

        - omitted *** Begin Patch
        - omitted *** End Patch
        - markdown code fences
        - blank lines in Add File
        - blank lines inside Update hunks

    It remains strict about actual content prefixes, exact update matching,
    duplicate operations, ambiguous matches, and invalid operation syntax.
    """

    name = "apply_patch"
    action = "modify"

    description = (
        "Apply Codex-style patches to text files. "
        "Supports Add File, Update File, and Delete File. "
        "Update hunks use exact context matching. "
        "Multiple file operations are allowed. "
        "The patch may be wrapped in a markdown code fence. "
        "For updates, use exact file context. "
        "The tool fails rather than guessing when an update is ambiguous."
    )

    parameters = {
        "type": "object",
        "properties": {
            "patch": {
                "type": "string",
                "description": (
                    "Codex-style patch text. Example:\n"
                    "*** Begin Patch\n"
                    "*** Add File: example.py\n"
                    '+example = "hello"\n'
                    "+\n"
                    "*** Update File: app.py\n"
                    "@@\n"
                    " context line\n"
                    "-old line\n"
                    "+new line\n"
                    "*** Delete File: old.py\n"
                    "*** End Patch\n"
                    "\n"
                    "For Add File, prefix each content line with '+'. "
                    "A blank line may also be written as an empty line. "
                    "For Update File, start each hunk with '@@' and use "
                    "' ' for context, '-' for removed lines, and '+' for "
                    "added lines. Exact context is required."
                ),
            }
        },
        "required": ["patch"],
        "additionalProperties": False,
    }

    _BEGIN = "*** Begin Patch"
    _END = "*** End Patch"

    _UPDATE = "*** Update File: "
    _ADD = "*** Add File: "
    _DELETE = "*** Delete File: "

    MAX_CONTENT_PREVIEW_CHARS = 4000

    def execute(self, patch: str) -> ToolResult:
        if not isinstance(patch, str):
            return self._failure(
                error_type="invalid_argument",
                message="patch must be a string.",
            )

        if not patch.strip():
            return self._failure(
                error_type="invalid_patch",
                message="Patch cannot be empty.",
            )

        try:
            operations = self._parse_patch(patch)
        except ValueError as exc:
            return self._failure(
                error_type="parse_error",
                message=str(exc),
            )

        if not operations:
            return self._failure(
                error_type="invalid_patch",
                message="Patch contains no file operations.",
            )

        try:
            plan = self._build_plan(operations)
        except ValueError as exc:
            return self._failure(
                error_type="validation_error",
                message=str(exc),
            )

        # _plan_update() can return a structured ToolResult for recoverable
        # patch-context failures. Propagate it directly instead of treating it
        # as a planned file-operation dictionary.
        if isinstance(plan, ToolResult):
            return plan

        try:
            results = self._apply_plan(plan)
        except OSError as exc:
            return self._failure(
                error_type="filesystem_error",
                message=str(exc),
            )
        except Exception as exc:
            return self._failure(
                error_type="application_error",
                message=str(exc),
            )

        files_changed = len(results)
        added = sum(item["added"] for item in results)
        removed = sum(item["removed"] for item in results)

        ops_summary = []

        for item in results:
            filename = Path(item["path"]).name
            ops_summary.append(f"{item['operation']}:{filename}")

        return self._success(
            summary=(
                f"Applied patch to {files_changed} file(s): "
                f"{added} addition(s), {removed} deletion(s). "
                f"Operations: [{', '.join(ops_summary)}]. "
                f"Use the returned file preview when available; "
                f"re-read large files only when necessary."
            ),
            files=results,
            statistics={
                "files_changed": files_changed,
                "added": added,
                "removed": removed,
            },
        )

    def _parse_patch(self, patch: str) -> list[PatchOperation]:
        normalized = self._normalize_patch_text(patch)

        lines = normalized.split("\n")

        if not lines:
            raise ValueError("Patch is empty.")

        operations = self._parse_operations(lines)

        if not operations:
            raise ValueError(
                "Patch contains no Add File, Update File, or Delete File operation."
            )

        return operations

    def _normalize_patch_text(self, patch: str) -> str:
        """
        Normalize common LLM formatting mistakes without changing the
        actual Codex-style operation syntax.
        """

        text = patch.replace("\r\n", "\n").replace("\r", "\n")

        # Remove BOM if present.
        text = text.lstrip("\ufeff")

        # Remove surrounding whitespace/newlines first.
        text = text.strip()

        if not text:
            raise ValueError("Patch is empty.")

        lines = text.split("\n")

        # Accept markdown fences:
        #
        # ```patch
        # *** Begin Patch
        # ...
        # *** End Patch
        # ```
        if lines:
            first = lines[0].strip().lower()

            if first.startswith("```"):
                lines = lines[1:]

                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]

        # Remove only surrounding empty lines.
        while lines and not lines[0].strip():
            lines.pop(0)

        while lines and not lines[-1].strip():
            lines.pop()

        if not lines:
            raise ValueError("Patch is empty.")

        # Be tolerant if the model omitted Begin Patch.
        if lines[0].strip() != self._BEGIN:
            if self._is_operation_line(lines[0]):
                lines.insert(0, self._BEGIN)
            else:
                raise ValueError(
                    f"Patch must start with '{self._BEGIN}' "
                    f"or directly with a file operation."
                )

        # Be tolerant if the model omitted End Patch.
        #
        # We only auto-add it when there is no existing End marker.
        # If End exists in the middle, keep parsing strict and report it.
        end_positions = [
            index for index, line in enumerate(lines) if line.strip() == self._END
        ]

        if not end_positions:
            lines.append(self._END)
        elif end_positions[-1] != len(lines) - 1:
            raise ValueError(f"'{self._END}' must be the final patch marker.")

        return "\n".join(lines)

    def _parse_operations(
        self,
        lines: list[str],
    ) -> list[PatchOperation]:

        if lines[0].strip() != self._BEGIN:
            raise ValueError(f"Patch must start with '{self._BEGIN}'.")

        if lines[-1].strip() != self._END:
            raise ValueError(f"Patch must end with '{self._END}'.")

        operations: list[PatchOperation] = []

        i = 1

        while i < len(lines) - 1:
            line = lines[i]

            # Ignore purely empty separator lines between operations.
            if not line.strip():
                i += 1
                continue

            if line.startswith(self._UPDATE):
                path = line[len(self._UPDATE) :].strip()

                if not path:
                    raise ValueError("Update File path cannot be empty.")

                i += 1

                hunks, i = self._parse_hunks(
                    lines=lines,
                    start_index=i,
                )

                if not hunks:
                    raise ValueError(
                        f"Update File '{path}' has no hunks. "
                        "Start the update content with '@@'."
                    )

                operations.append(
                    PatchOperation(
                        operation="update",
                        path=path,
                        hunks=hunks,
                    )
                )

                continue

            if line.startswith(self._ADD):
                path = line[len(self._ADD) :].strip()

                if not path:
                    raise ValueError("Add File path cannot be empty.")

                i += 1

                content_lines: list[str] = []

                while i < len(lines) - 1:
                    current = lines[i]

                    if self._is_operation_line(current):
                        break

                    if current.startswith("@@"):
                        raise ValueError(
                            f"Add File '{path}' cannot contain hunks. "
                            "Prefix file content lines with '+'."
                        )

                    # Important LLM tolerance:
                    #
                    # Standard Codex form for an empty line is:
                    #
                    # +
                    #
                    # Some models emit a truly empty line instead:
                    #
                    # <empty>
                    #
                    # Accept both.
                    if current == "":
                        content_lines.append("")
                        i += 1
                        continue

                    if not current.startswith("+"):
                        raise ValueError(
                            f"Invalid Add File line in '{path}': "
                            f"{current!r}. "
                            "Every non-empty content line must start with '+'. "
                            "Use '+' for a blank line."
                        )

                    content_lines.append(current[1:])
                    i += 1

                content = self._join_added_lines(content_lines)

                operations.append(
                    PatchOperation(
                        operation="add",
                        path=path,
                        hunks=[],
                        content=content,
                    )
                )

                continue

            if line.startswith(self._DELETE):
                path = line[len(self._DELETE) :].strip()

                if not path:
                    raise ValueError("Delete File path cannot be empty.")

                operations.append(
                    PatchOperation(
                        operation="delete",
                        path=path,
                        hunks=[],
                    )
                )

                i += 1
                continue

            if line.strip() == self._END:
                break

            raise ValueError(
                f"Unexpected patch line: {line!r}. "
                f"Expected Add File, Update File, Delete File, "
                f"or '{self._END}'."
            )

        return operations

    def _parse_hunks(
        self,
        lines: list[str],
        start_index: int,
    ) -> tuple[list[list[str]], int]:

        hunks: list[list[str]] = []

        current_hunk: list[str] | None = None

        i = start_index

        while i < len(lines) - 1:
            line = lines[i]

            if self._is_operation_line(line):
                break

            if line.strip() == self._END:
                break

            if line.startswith("@@"):
                if current_hunk is not None:
                    if not current_hunk:
                        raise ValueError("Empty patch hunk.")

                    hunks.append(current_hunk)

                current_hunk = []

                i += 1
                continue

            if current_hunk is None:
                raise ValueError("Update File content must begin with '@@'.")

            if line == "":
                current_hunk.append(" ")
                i += 1
                continue

            prefix = line[0]

            if prefix not in (" ", "+", "-"):
                raise ValueError(
                    f"Invalid hunk line: {line!r}. "
                    "Expected ' ' for context, '-' for removal, "
                    "or '+' for addition."
                )

            current_hunk.append(line)

            i += 1

        if current_hunk is not None:
            if not current_hunk:
                raise ValueError("Empty patch hunk.")

            hunks.append(current_hunk)

        return hunks, i

    @classmethod
    def _is_operation_line(
        cls,
        line: str,
    ) -> bool:
        return (
            line.startswith(cls._UPDATE)
            or line.startswith(cls._ADD)
            or line.startswith(cls._DELETE)
        )

    def _build_plan(
        self,
        operations: list[PatchOperation],
    ) -> list[dict[str, Any]] | ToolResult:

        seen_paths: set[str] = set()

        plan: list[dict[str, Any]] = []

        for operation in operations:
            normalized_path = self._normalize_path(operation.path)

            if normalized_path in seen_paths:
                raise ValueError(
                    f"Duplicate operation for path " f"'{operation.path}'."
                )

            seen_paths.add(normalized_path)

            path = Path(normalized_path)

            if operation.operation == "add":
                plan.append(
                    self._plan_add(
                        path=path,
                        operation=operation,
                    )
                )

            elif operation.operation == "update":
                planned_update = self._plan_update(
                    path=path,
                    operation=operation,
                )
                if isinstance(planned_update, ToolResult):
                    return planned_update
                plan.append(planned_update)

            elif operation.operation == "delete":
                plan.append(
                    self._plan_delete(
                        path=path,
                    )
                )

            else:
                raise ValueError(
                    f"Unsupported patch operation " f"'{operation.operation}'."
                )

        return plan

    def _plan_add(
        self,
        path: Path,
        operation: PatchOperation,
    ) -> dict[str, Any]:

        if path.exists():
            raise ValueError(f"Cannot add '{path}': file already exists.")

        if path.is_dir():
            raise ValueError(f"Cannot add '{path}': path is a directory.")

        content = operation.content or ""

        return {
            "operation": "add",
            "path": path,
            "content": content,
            "added": self._count_lines(content),
            "removed": 0,
        }

    def _plan_update(
        self,
        path: Path,
        operation: PatchOperation,
    ) -> dict[str, Any]:

        if not path.exists():
            raise ValueError(f"Cannot update '{path}': file does not exist.")

        if not path.is_file():
            raise ValueError(f"Cannot update '{path}': path is not a file.")

        try:
            original = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"Cannot update non-UTF-8/binary file '{path}'.") from exc

        try:
            updated, added, removed = self._apply_hunks(
                original=original,
                hunks=operation.hunks,
                path=str(path),
            )
        except ValueError as exc:
            message = str(exc)
            if (
                "Patch context did not match" in message
                or "Ambiguous patch" in message
            ):
                try:
                    current_content = path.read_text(
                        encoding="utf-8",
                    )
                except (OSError, UnicodeDecodeError):
                    current_content = ""

                return self._failure(
                    error_type="patch_context_mismatch",
                    message=message,
                    details={
                        "path": str(path),
                        "type": "file",
                        "content": self._bounded_preview(current_content),
                        "total_lines": self._count_lines(current_content),
                    },
                )
            raise

        return {
            "operation": "update",
            "path": path,
            "content": updated,
            "added": added,
            "removed": removed,
        }

    def _plan_delete(
        self,
        path: Path,
    ) -> dict[str, Any]:

        if not path.exists():
            raise ValueError(f"Cannot delete '{path}': file does not exist.")

        if not path.is_file():
            raise ValueError(f"Cannot delete '{path}': path is not a file.")

        try:
            file_content = path.read_text(encoding="utf-8")

            removed_lines = self._count_lines(file_content)

        except (UnicodeDecodeError, OSError):
            removed_lines = 0

        return {
            "operation": "delete",
            "path": path,
            "content": None,
            "added": 0,
            "removed": removed_lines,
        }

    def _apply_hunks(
        self,
        original: str,
        hunks: list[list[str]],
        path: str,
    ) -> tuple[str, int, int]:

        newline = self._detect_newline(original)

        file_lines = original.splitlines(keepends=True)

        additions = 0
        removals = 0

        search_from = 0

        for hunk_index, hunk in enumerate(
            hunks,
            start=1,
        ):
            old_lines: list[str] = []
            new_lines: list[str] = []

            for raw_line in hunk:
                prefix = raw_line[0]
                value = raw_line[1:]

                if prefix == " ":
                    old_lines.append(value)
                    new_lines.append(value)

                elif prefix == "-":
                    old_lines.append(value)

                elif prefix == "+":
                    new_lines.append(value)

                else:
                    raise ValueError(
                        f"Invalid hunk line in '{path}', "
                        f"hunk {hunk_index}: {raw_line!r}"
                    )

            if not old_lines:
                raise ValueError(
                    f"Hunk {hunk_index} in '{path}' "
                    "contains only additions. "
                    "Add at least one context or removed line "
                    "so the exact insertion point is unambiguous."
                )

            old_normalized = [self._remove_newline(line) for line in old_lines]

            matches = self._find_matches(
                file_lines=file_lines,
                target=old_normalized,
            )

            forward_matches = [
                position for position in matches if position >= search_from
            ]

            if not forward_matches:
                raise ValueError(
                    f"Patch context did not match in '{path}', "
                    f"hunk {hunk_index}. "
                    "The file may have changed or the context may be incorrect. "
                    "Re-read the relevant file section and retry."
                )

            if len(forward_matches) > 1:
                raise ValueError(
                    f"Ambiguous patch in '{path}', "
                    f"hunk {hunk_index}: "
                    f"context matched {len(forward_matches)} times. "
                    "Use more specific context."
                )

            position = forward_matches[0]

            replacement = [
                self._format_new_line(
                    line=line,
                    newline=newline,
                )
                for line in new_lines
            ]

            file_lines[position : position + len(old_normalized)] = replacement

            search_from = position + len(replacement)

            additions += sum(1 for line in hunk if line.startswith("+"))

            removals += sum(1 for line in hunk if line.startswith("-"))

        return (
            "".join(file_lines),
            additions,
            removals,
        )

    def _find_matches(
        self,
        file_lines: list[str],
        target: list[str],
    ) -> list[int]:

        if not target:
            return []

        normalized_file = [self._remove_newline(line) for line in file_lines]

        matches: list[int] = []

        limit = len(normalized_file) - len(target) + 1

        if limit <= 0:
            return matches

        for position in range(limit):
            candidate = normalized_file[position : position + len(target)]

            if candidate == target:
                matches.append(position)

        return matches

    def _apply_plan(
        self,
        plan: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:

        results: list[dict[str, Any]] = []

        for item in plan:
            path: Path = item["path"]
            operation = item["operation"]

            if operation == "add":
                path.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                path.write_text(
                    item["content"],
                    encoding="utf-8",
                )

                content = item.get("content") or ""

                results.append(
                    self._build_file_result(
                        operation="add",
                        path=path,
                        content=content,
                        added=item["added"],
                        removed=item["removed"],
                    )
                )

            elif operation == "update":
                path.write_text(
                    item["content"],
                    encoding="utf-8",
                )

                content = item.get("content") or ""

                results.append(
                    self._build_file_result(
                        operation="update",
                        path=path,
                        content=content,
                        added=item["added"],
                        removed=item["removed"],
                    )
                )

            elif operation == "delete":
                path.unlink()

                results.append(
                    {
                        "path": str(path),
                        "operation": "delete",
                        "added": item["added"],
                        "removed": item["removed"],
                    }
                )

            else:
                raise RuntimeError(f"Unknown planned operation: " f"{operation}")

        return results

    def _build_file_result(
        self,
        *,
        operation: str,
        path: Path,
        content: str,
        added: int,
        removed: int,
    ) -> dict[str, Any]:

        result: dict[str, Any] = {
            "path": str(path),
            "operation": operation,
            "added": added,
            "removed": removed,
            "content_preview": self._bounded_preview(content),
            "total_lines": self._count_lines(content),
        }

        # Do not send huge complete files back into the model context.
        #
        # Small files keep the previous convenient behavior.
        if len(content) <= self.MAX_CONTENT_PREVIEW_CHARS:
            result["content"] = content
            result["content_truncated"] = False
        else:
            result["content"] = None
            result["content_truncated"] = True

        return result

    @staticmethod
    def _normalize_path(
        value: str,
    ) -> str:

        value = value.strip()

        if not value:
            raise ValueError("File path cannot be empty.")

        return str(Path(value))

    @staticmethod
    def _detect_newline(
        content: str,
    ) -> str:

        if "\r\n" in content:
            return "\r\n"

        return "\n"

    @staticmethod
    def _remove_newline(
        value: str,
    ) -> str:

        if value.endswith("\r\n"):
            return value[:-2]

        if value.endswith("\n"):
            return value[:-1]

        if value.endswith("\r"):
            return value[:-1]

        return value

    @staticmethod
    def _format_new_line(
        line: str,
        newline: str,
    ) -> str:

        return ApplyPatch._remove_newline(line) + newline

    @staticmethod
    def _join_added_lines(
        lines: list[str],
    ) -> str:

        if not lines:
            return ""

        return "\n".join(lines) + "\n"

    @staticmethod
    def _count_lines(
        content: str,
    ) -> int:

        if not content:
            return 0

        return len(content.splitlines())

    @classmethod
    def _bounded_preview(
        cls,
        content: str,
    ) -> str:

        if not content:
            return ""

        limit = cls.MAX_CONTENT_PREVIEW_CHARS

        if len(content) <= limit:
            return content

        head = int(limit * 0.70)

        tail = limit - head

        omitted = len(content) - head - tail

        return (
            content[:head].rstrip()
            + "\n... "
            + str(omitted)
            + " chars omitted ...\n"
            + content[-tail:].lstrip()
        )

    def _success(
        self,
        *,
        summary: str,
        files: list[dict[str, Any]],
        statistics: dict[str, int],
    ) -> ToolResult:

        return ToolResult(
            success=True,
            name=self.name,
            content={
                "success": True,
                "operation": self.name,
                "summary": summary,
                "files": files,
                "statistics": statistics,
            },
            metadata={},
        )

    def _failure(
        self,
        *,
        error_type: str,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> ToolResult:

        content: dict[str, Any] = {
            "success": False,
            "operation": self.name,
            "error": {
                "type": error_type,
                "message": message,
            },
        }

        if isinstance(details, dict):
            content.update(details)

        return ToolResult(
            success=False,
            name=self.name,
            content=content,
            metadata={},
        )

    def describe_call(
        self,
        arguments: dict[str, Any],
    ) -> dict[str, str]:

        patch = arguments.get(
            "patch",
            "",
        )

        if not isinstance(
            patch,
            str,
        ):
            return {
                "action": "modify",
                "target": "",
            }

        try:
            operations = self._parse_patch(patch)
        except Exception:
            return {
                "action": "modify",
                "target": "",
            }

        if not operations:
            return {
                "action": "modify",
                "target": "",
            }

        operation_types = {operation.operation for operation in operations}

        if operation_types == {"add"}:
            action = "create"

        elif operation_types == {"delete"}:
            action = "delete"

        else:
            action = "modify"

        targets = [operation.path for operation in operations if operation.path]

        return {
            "action": action,
            "target": ", ".join(targets),
        }

    def __repr__(self) -> str:
        return f"<Tool name='{self.name}'>"
