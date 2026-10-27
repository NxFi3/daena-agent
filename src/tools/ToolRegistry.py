from __future__ import annotations

import importlib
import inspect
import re
from pathlib import Path
from types import ModuleType

from src.tools.Tool import Tool
from src.tools.contract import validate_catalog
from src.utils.logger import get_logger


class ToolRegistry:
    """Discover built-in tools without relying on fragile package boilerplate.

    Packages may explicitly export tools through __all__. When they do not,
    the registry falls back to scanning Python modules inside that tool
    directory and discovers concrete Tool subclasses defined there.
    """

    PACKAGE_ROOT = "src.tools.builtin"

    def __init__(self):
        self.tools_path = Path(__file__).parent / "builtin"
        self.package_root = self.PACKAGE_ROOT
        self.logger = get_logger("[TOOLREGISTRY]")
        self.tools: dict[str, Tool] = {}

    @staticmethod
    def _is_tool_class(obj) -> bool:
        return (
            inspect.isclass(obj)
            and obj is not Tool
            and issubclass(obj, Tool)
            and not inspect.isabstract(obj)
        )

    def _explicit_classes(self, module: ModuleType) -> list[type[Tool]]:
        exports = getattr(module, "__all__", None)
        if not isinstance(exports, (list, tuple)):
            return []

        classes: list[type[Tool]] = []
        for export_name in exports:
            obj = getattr(module, str(export_name), None)
            if obj is None:
                self.logger.warning(
                    "Skipping '%s': not found in module %s",
                    export_name,
                    module.__name__,
                )
                continue
            if not self._is_tool_class(obj):
                self.logger.warning(
                    "Skipping '%s': export is not a concrete Tool",
                    export_name,
                )
                continue
            classes.append(obj)
        return classes

    def _module_classes(self, module: ModuleType) -> list[type[Tool]]:
        classes: list[type[Tool]] = []
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if getattr(obj, "__module__", None) != module.__name__:
                continue
            if self._is_tool_class(obj):
                classes.append(obj)
        return classes

    def _fallback_modules(
        self,
        package_dir: Path,
        package_module: ModuleType,
    ):
        """Yield modules to inspect when a package has no explicit exports."""
        yielded = {package_module.__name__}
        yield package_module

        for file_path in sorted(package_dir.rglob("*.py")):
            try:
                relative = file_path.relative_to(self.tools_path).with_suffix("")
            except ValueError:
                continue

            parts = list(relative.parts)
            if not parts:
                continue

            if file_path.name == "__init__.py":
                parts = parts[:-1]
                if not parts:
                    continue
            elif file_path.stem.startswith("_"):
                continue

            if any(part == "__pycache__" for part in parts):
                continue

            module_name = ".".join([self.package_root, *parts])
            if not module_name or module_name in yielded:
                continue

            try:
                module = importlib.import_module(module_name)
            except Exception as exc:
                self.logger.warning(
                    "Failed to load fallback module '%s': %s",
                    module_name,
                    exc,
                )
                continue

            yielded.add(module_name)
            yield module

    def _discover_package_classes(
        self,
        package_dir: Path,
        package_module: ModuleType,
    ) -> list[type[Tool]]:
        if hasattr(package_module, "__all__"):
            self.logger.info("Exports: %s", package_module.__all__)
            return self._explicit_classes(package_module)

        self.logger.warning(
            "Tool package '%s' has no __all__; falling back to module discovery",
            package_dir.name,
        )

        classes: list[type[Tool]] = []
        seen: set[type[Tool]] = set()

        for module in self._fallback_modules(package_dir, package_module):
            for cls in self._module_classes(module):
                if cls in seen:
                    continue
                seen.add(cls)
                classes.append(cls)

        return classes

    def _discover_tools(self) -> list[Tool]:
        discovered_tools: list[Tool] = []

        if not self.tools_path.exists():
            self.logger.error("Tools directory does not exist: %s", self.tools_path)
            return discovered_tools

        for item in sorted(self.tools_path.iterdir()):
            if not item.is_dir() or item.name.startswith("_"):
                continue

            self.logger.info("Found tool package: %s", item.name)
            module_name = f"{self.package_root}.{item.name}"

            try:
                module = importlib.import_module(module_name)
            except Exception as exc:
                self.logger.error(
                    "Failed to load tool package '%s': %s",
                    item.name,
                    exc,
                )
                continue

            classes = self._discover_package_classes(item, module)
            if not classes:
                self.logger.warning(
                    "Skipping '%s': no concrete Tool subclasses found",
                    item.name,
                )
                continue

            for cls in classes:
                try:
                    tool = cls()
                except Exception as exc:
                    self.logger.error(
                        "Failed to instantiate '%s': %s",
                        cls.__name__,
                        exc,
                    )
                    continue

                name = str(getattr(tool, "name", "") or "").strip().lower()
                if not name:
                    self.logger.warning(
                        "Skipping '%s': tool has no name",
                        cls.__name__,
                    )
                    continue

                self.logger.info("Instantiated: %s", name)
                discovered_tools.append(tool)

        return discovered_tools

    def discover(self) -> list[str]:
        self.tools.clear()

        for tool in self._discover_tools():
            name = str(tool.name).strip().lower()
            existing = self.tools.get(name)
            if existing is not None and type(existing) is not type(tool):
                self.logger.warning(
                    "Duplicate tool name '%s'; keeping %s and ignoring %s",
                    name,
                    type(existing).__name__,
                    type(tool).__name__,
                )
                continue
            self.tools[name] = tool

        self.logger.info("Discovered %s unique tool(s)", len(self.tools))
        return list(self.tools.keys())

    def register(self, tool: Tool) -> str:
        """Register a host-owned tool or adapter without replacing another tool.

        Registration is a setup operation: configure adapters before agent runs.
        ToolManager applies the usual allowlist, schema and runtime policy gates.
        """
        from jsonschema.validators import validator_for

        if not isinstance(tool, Tool):
            raise TypeError("Registered tools must implement Tool")
        name = tool.name
        if not isinstance(name, str) or not re.fullmatch(
            r"[a-z][a-z0-9_-]{0,63}", name
        ):
            raise ValueError(
                "Tool names must be lowercase identifiers of 1–64 characters"
            )
        if not isinstance(tool.parameters, dict):
            raise ValueError("Tool parameters must be a JSON Schema object")
        validator_for(tool.parameters).check_schema(tool.parameters)
        existing = self.tools.get(name)
        if existing is not None and existing is not tool:
            raise ValueError(f"Tool name already registered: {name}")
        self.tools[name] = tool
        return name

    def is_available(self, toolname: str) -> bool:
        if not isinstance(toolname, str):
            return False
        return toolname.strip().lower() in self.tools

    def get(self, toolname: str):
        if not isinstance(toolname, str):
            return None
        return self.tools.get(toolname.strip().lower())

    def get_definitions(self):
        return validate_catalog([tool.get_definition() for tool in self.tools.values()])
