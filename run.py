from collections.abc import Sequence
import sys

from spectrum_app.application import SpectrumApplication
from spectrum_app.modules.manager import ModuleManager


REQUIRED_MODULE_IDS = {"impedance", "phase", "rta", "spectrum", "thd"}


def _check_modules() -> int:
    manager = ModuleManager()
    manager.discover()
    return 0 if set(manager.module_ids) == REQUIRED_MODULE_IDS else 1


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments == ["--check-modules"]:
        return _check_modules()

    app = SpectrumApplication()
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
