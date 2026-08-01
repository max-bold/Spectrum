from collections.abc import Callable

from .core.dpg import DearPyGuiRuntime
from .core.model import AppState, Measurement
from .gui.main_window import MainWindow


class SpectrumApplication:
    """Owns and coordinates the application lifecycle."""

    DEFAULT_MODULE_ID = "spectrum"

    def __init__(self) -> None:
        self._running = False
        self.frame_callbacks: list[Callable[[], None]] = []
        self.app_state = AppState()
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
        self.dpg.create_context()
        if not self.app_state.measurements:
            self.create_measurement()
        self.main_window.build()
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
        self.dpg.destroy_context()

    def create_measurement(self, module_id: str | None = None) -> Measurement:
        measurement = Measurement(
            module_id=module_id or self.DEFAULT_MODULE_ID,
            name=f"Measurement {len(self.app_state.measurements) + 1}",
        )
        self.app_state.measurements.append(measurement)
        self.app_state.active_measurement_id = measurement.id
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
