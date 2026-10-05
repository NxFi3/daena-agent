from pathlib import Path
import importlib
import inspect

from src.utils.logger import get_logger
from src.tools.Tool import Tool


class ToolRegistry:

    def __init__(self):
        self.tools_path = Path(__file__).parent / "builtin"
        self.logger = get_logger("[TOOLREGISTRY]")
        self.tools = {}

    def _discover_tools(self):

        discovered_tools = []

        if not self.tools_path.exists():
            self.logger.error(f"Tools directory does not exist: {self.tools_path}")
            return discovered_tools

        for item in self.tools_path.iterdir():

            if not item.is_dir():
                continue

            if item.name.startswith("_"):
                continue

            self.logger.info(f"Found tool package: {item.name}")

            module_name = f"src.tools.builtin.{item.name}"

            try:
                module = importlib.import_module(module_name)

            except Exception as e:
                self.logger.error(f"Failed to load tool package " f"'{item.name}': {e}")
                continue

            self.logger.info(f"Loaded module: {module}")

            if not hasattr(module, "__all__"):
                self.logger.warning(f"Skipping '{item.name}': " "__all__ not found")
                continue

            self.logger.info(f"Exports: {module.__all__}")

            for export_name in module.__all__:

                obj = getattr(
                    module,
                    export_name,
                    None,
                )

                if obj is None:
                    self.logger.warning(
                        f"Skipping '{export_name}': " "not found in module"
                    )
                    continue

                if not inspect.isclass(obj):
                    self.logger.warning(f"Skipping '{export_name}': " "not a class")
                    continue

                if not issubclass(obj, Tool):
                    self.logger.warning(f"Skipping '{export_name}': " "not a Tool")
                    continue

                if inspect.isabstract(obj):
                    self.logger.warning(
                        f"Skipping '{export_name}': " "Tool is abstract"
                    )
                    continue

                self.logger.info(f"Found Tool class: {obj}")

                try:
                    tool = obj()

                except Exception as e:
                    self.logger.error(f"Failed to instantiate " f"'{export_name}': {e}")
                    continue

                if not getattr(tool, "name", None):
                    self.logger.warning(
                        f"Skipping '{export_name}': " "tool has no name"
                    )
                    continue

                self.logger.info(f"Instantiated: {tool.name}")

                discovered_tools.append(tool)

        return discovered_tools

    def discover(self):
        # Rebuild the registry from the current filesystem. This prevents a
        # tool removed or disabled during development from surviving in a
        # long-lived registry instance.
        self.tools.clear()

        discovered_tools = self._discover_tools()

        for tool in discovered_tools:
            self.tools[str(tool.name).strip().lower()] = tool

        self.logger.info(f"Discovered " f"{len(self.tools)} unique tool(s)")

        return list(self.tools.keys())

    def is_available(self, toolname: str):
        if not isinstance(toolname, str):
            return False

        return toolname.strip().lower() in self.tools

    def get(self, toolname: str):
        if not isinstance(toolname, str):
            return None

        return self.tools.get(toolname.strip().lower())

    def get_definitions(self):
        return [tool.get_definition() for tool in self.tools.values()]
