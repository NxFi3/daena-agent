# src/Engine/providers/Registry.py

from pathlib import Path
import importlib
import inspect
from src.utils.logger import get_logger
from src.engine.providers.ProviderBase import ProviderBase


class ProviderRegistry:

    def __init__(self):
        self.providers_path = Path(__file__).parent / "builtin"
        self.logger = get_logger("[PROVIDERREGISTRY]")
        self.providers = {}

    def _discover_providers(self):

        discovered_providers = []

        for item in self.providers_path.iterdir():

            if not item.is_dir():
                continue

            if item.name.startswith("_"):
                continue

            self.logger.info(f"Found provider package: {item.name}")
            module_name = f"src.engine.providers.builtin.{item.name}"

            try:
                module = importlib.import_module(module_name)

            except Exception as e:
                self.logger.warning(
                    f"Failed to load provider package " f"'{item.name}': {e}"
                )
                continue
            self.logger.info(f"Loaded module: {module}")
            if not hasattr(module, "__all__"):
                self.logger.warning(f"Skipping '{item.name}': " f"__all__ not found")
                continue

            self.logger.info(f"Exports: {module.__all__}")

            for export_name in module.__all__:
                obj = getattr(module, export_name, None)

                if obj is None:
                    self.logger.warning(
                        f"Skipping '{export_name}': " f"not found in module"
                    )
                    continue

                if not inspect.isclass(obj):
                    self.logger.warning(f"Skipping '{export_name}': " f"not a class")
                    continue
                if not issubclass(obj, ProviderBase):
                    self.logger.warning(f"Skipping '{export_name}': " f"not a provider")
                    continue

                if inspect.isabstract(obj):
                    self.logger.warning(
                        f"Skipping '{export_name}': " f"provider is abstract"
                    )
                    continue

                self.logger.info(f"Found provider class: {obj}")
                try:
                    provider = obj()

                except Exception as e:
                    self.logger.error(f"Failed to instantiate " f"'{export_name}': {e}")
                    continue

                self.logger.info(f"Instantiated: " f"{provider.name}")

                discovered_providers.append(provider)

        return discovered_providers

    def discover(self):
        # Rebuild the registry from the current provider packages so a failed
        # or removed provider cannot remain available in a long-lived runtime.
        self.providers.clear()

        discovered_providers = self._discover_providers()

        for provider in discovered_providers:
            self.providers[provider.name] = provider

        self.logger.info(f"Discovered {len(self.providers)} provider(s)")
        return list(self.providers.keys())

    def is_available(self, providername: str):
        return providername in self.providers

    def get(self, providername: str):
        return self.providers.get(providername)
