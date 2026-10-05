from collections.abc import Sequence
import sys


REQUIRED_MODULE_IDS = {"impedance", "phase", "rta", "spectrum", "thd"}


def _check_modules() -> int:
    from spectrum_app.modules.manager import ModuleManager
    try:
        manager = ModuleManager()
        manager.discover()
        return 0 if set(manager.module_ids) == REQUIRED_MODULE_IDS else 1
    except Exception:
        return 1


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments == ["--check-modules"]:
        return _check_modules()
    from spectrum_app.startup import StartupSplash, show_splash
    if len(arguments) == 2 and arguments[0] == "--splash":
        show_splash(arguments[1])
        return 0
    splash = StartupSplash()
    splash.start()
    try:
        splash.update("Loading analysis libraries", 0.05)
        from spectrum_app.application import SpectrumApplication
        def ready() -> None:
            splash.close()
            if arguments == ["--check-startup"]:
                app.dpg.backend.stop_dearpygui()
        app = SpectrumApplication(startup_progress=splash.update, startup_ready=ready)
        app.run()
    finally:
        splash.close()
    return 0
