from collections.abc import Callable
from pathlib import Path

from .core.audio import AudioInput, AudioOutput, AudioService
from .core.dpg import DearPyGuiRuntime
from .core.model import AppState, Measurement
from .core.project import (
    ProjectError,
    load_project as load_app_state,
    save_project as save_app_state,
)
from .core.settings import AppSettings
from .gui.main_window import MainWindow
from .modules.manager import ModuleManager
from .modules.base import BaseModule


class SpectrumApplication:
    """Owns and coordinates the application lifecycle."""

    DEFAULT_MODULE_ID = "spectrum"

    def __init__(self) -> None:
        self._running = False
        self.frame_callbacks: list[Callable[[], None]] = []
        self.app_state = AppState()
        self.settings = AppSettings(on_change=self._settings_changed)
        self._audio_service = AudioService(self.settings)
        self.audio_input = AudioInput(self._audio_service)
        self.audio_output = AudioOutput(self._audio_service)
        self.module_manager = ModuleManager()
        self.module_manager.discover()
        self._initialized_modules: list[BaseModule] = []
        self.dpg = DearPyGuiRuntime()
        self.main_window = MainWindow(self)

    @property
    def running(self) -> bool:
        return self._running

    def run(self) -> None:
        if self._running:
            raise RuntimeError("SpectrumApplication is already running")

        self._running = True
        try:
            self._initialize()
            self._run_main_loop()
        finally:
            try:
                self._shutdown()
            finally:
                self._running = False

    def _initialize(self) -> None:
        self.settings.load()
        self._audio_service.start()
        self.dpg.create_context()
        if not self.app_state.measurements:
            self.create_measurement()
        self.main_window.build()
        self._initialize_modules()
        self.main_window.measurement_panel.modules_initialized()
        self.dpg.show_viewport(
            title=self.main_window.TITLE,
            width=self.main_window.WIDTH,
            height=self.main_window.HEIGHT,
            primary_window=self.main_window.tag,
        )

    def _run_main_loop(self) -> None:
        while self.dpg.running:
            self.dpg.process_callbacks()
            self._process_frame_callbacks()
            self.main_window.update()
            self.dpg.render_frame()

    def _process_frame_callbacks(self) -> None:
        for callback in self.frame_callbacks.copy():
            callback()

    def _shutdown(self) -> None:
        try:
            self.main_window.measurement_panel.shutdown()
        finally:
            try:
                self._shutdown_modules()
            finally:
                try:
                    self._audio_service.shutdown()
                finally:
                    self.dpg.destroy_context()

    def _initialize_modules(self) -> None:
        for module in self.module_manager.modules:
            try:
                module.initialize(self)
            except Exception:
                try:
                    module.shutdown()
                finally:
                    self._shutdown_modules()
                raise
            self._initialized_modules.append(module)

    def _shutdown_modules(self) -> None:
        first_error: Exception | None = None
        for module in reversed(self._initialized_modules):
            try:
                module.shutdown()
            except Exception as error:
                if first_error is None:
                    first_error = error
        self._initialized_modules.clear()
        if first_error is not None:
            raise first_error

    def _settings_changed(self) -> None:
        self.app_state.graph_data_changed = True

    def save_project(self, path: str | Path | None = None) -> Path:
        project_path = path or self.app_state.project_path
        if project_path is None:
            raise ProjectError("Project path is not selected")
        return save_app_state(self.app_state, project_path)

    def load_project(self, path: str | Path) -> None:
        state = load_app_state(path)
        unknown_modules = sorted(
            {
                measurement.module_id
                for measurement in state.measurements
                if measurement.module_id not in self.module_manager.module_ids
            }
        )
        if unknown_modules:
            raise ProjectError(
                f"Project uses unavailable modules: {', '.join(unknown_modules)}"
            )

        measurement_ids = {measurement.id for measurement in state.measurements}
        if state.active_measurement_id not in measurement_ids:
            state.active_measurement_id = (
                state.measurements[0].id if state.measurements else None
            )
        graph_ids = {
            graph.id
            for measurement in state.measurements
            for graph in measurement.graphs
        }
        state.visible_graph_ids = [
            graph_id for graph_id in state.visible_graph_ids if graph_id in graph_ids
        ]
        state.measuring = False
        state.graph_data_changed = True

        self.main_window.measurement_panel.deactivate()
        self.app_state = state
        self.main_window.project_loaded()

    def create_measurement(self, module_id: str | None = None) -> Measurement:
        selected_module_id = module_id or self.DEFAULT_MODULE_ID
        self.module_manager.module(selected_module_id)
        measurement = Measurement(
            module_id=selected_module_id,
            name=f"Measurement {len(self.app_state.measurements) + 1}",
        )
        self.app_state.measurements.append(measurement)
        self.app_state.active_measurement_id = measurement.id
        self.app_state.graph_data_changed = True
        return measurement

    def delete_measurement(self, measurement_id: str) -> None:
        for index, measurement in enumerate(self.app_state.measurements):
            if measurement.id == measurement_id:
                break
        else:
            raise ValueError(f"Unknown measurement: {measurement_id}")

        self.app_state.measurements.pop(index)
        graph_ids = {graph.id for graph in measurement.graphs}
        self.app_state.visible_graph_ids = [
            graph_id
            for graph_id in self.app_state.visible_graph_ids
            if graph_id not in graph_ids
        ]
        self.app_state.graph_data_changed = True

        if self.app_state.active_measurement_id == measurement_id:
            if self.app_state.measurements:
                new_index = min(index, len(self.app_state.measurements) - 1)
                self.app_state.active_measurement_id = (
                    self.app_state.measurements[new_index].id
                )
            else:
                self.app_state.active_measurement_id = None


if __name__ == "__main__":
    app = SpectrumApplication()
    app.run()
