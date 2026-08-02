from importlib import import_module
from pathlib import Path
from types import ModuleType
from typing import Any

from spectrum_app.modules.base import BaseModule


class ModuleLoadError(RuntimeError):
    pass


class ModuleManager:
    """Discovers measurement modules and provides read-only registry access."""

    def __init__(
        self,
        path: Path | None = None,
        package: str = "spectrum_app.modules",
    ) -> None:
        self.path = path or Path(__file__).parent
        self.package = package
        self._modules: dict[str, BaseModule] = {}

    @property
    def modules(self) -> tuple[BaseModule, ...]:
        return tuple(self._modules.values())

    @property
    def module_ids(self) -> tuple[str, ...]:
        return tuple(self._modules)

    def module(self, module_id: str) -> BaseModule:
        try:
            return self._modules[module_id]
        except KeyError as error:
            raise ValueError(f"Unknown module: {module_id}") from error

    def discover(self) -> None:
        discovered: dict[str, BaseModule] = {}
        for directory in sorted(self.path.iterdir(), key=lambda item: item.name):
            if not directory.is_dir() or not (directory / "__init__.py").is_file():
                continue
            package = import_module(f"{self.package}.{directory.name}")
            module = self._create_module(package)
            if module.id in discovered:
                raise ModuleLoadError(f"Duplicate module id: {module.id}")
            if any(item.name == module.name for item in discovered.values()):
                raise ModuleLoadError(f"Duplicate module name: {module.name}")
            discovered[module.id] = module
        self._modules = discovered

    @staticmethod
    def _create_module(package: ModuleType) -> BaseModule:
        module_class: Any = getattr(package, "MODULE_CLASS", None)
        if not isinstance(module_class, type) or not issubclass(
            module_class, BaseModule
        ):
            raise ModuleLoadError(
                f"{package.__name__} must export a BaseModule as MODULE_CLASS"
            )
        if module_class is BaseModule:
            raise ModuleLoadError(f"{package.__name__} exports BaseModule itself")

        module = module_class()
        if not module.id:
            raise ModuleLoadError(f"{package.__name__} has an empty module id")
        if not module.name:
            raise ModuleLoadError(f"{package.__name__} has an empty module name")
        return module
