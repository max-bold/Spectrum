from importlib import import_module
from pathlib import Path
from pkgutil import iter_modules
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
        self._custom_path = path is not None
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
        for module_name in self._module_names():
            package = import_module(f"{self.package}.{module_name}")
            module = self._create_module(package)
            if module.id in discovered:
                raise ModuleLoadError(f"Duplicate module id: {module.id}")
            if any(item.name == module.name for item in discovered.values()):
                raise ModuleLoadError(f"Duplicate module name: {module.name}")
            discovered[module.id] = module
        self._modules = discovered

    def _module_names(self) -> list[str]:
        if self._custom_path:
            return sorted(
                directory.name
                for directory in self.path.iterdir()
                if directory.is_dir() and (directory / "__init__.py").is_file()
            )
        package = import_module(self.package)
        package_path = getattr(package, "__path__", None)
        if package_path is None:
            raise ModuleLoadError(f"{self.package} is not a package")
        registered = set(getattr(package, "BUILTIN_MODULE_NAMES", ()))
        discovered = {
            item.name for item in iter_modules(package_path) if item.ispkg
        }
        return sorted(registered | discovered)

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
