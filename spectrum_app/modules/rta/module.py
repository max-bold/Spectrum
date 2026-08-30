from __future__ import annotations

from copy import deepcopy
from threading import Lock
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from audioanalysis import ASignal, FrequencyBand, RTAConfig, RTAResult
from spectrum_app.core.model import AxisSpec, GraphData, Measurement, PlotType
from spectrum_app.modules.base import BaseModule
from spectrum_app.modules.rta.jobs import RTAAnalyzer, RTAIOWorker, RTARuntimeConfig
from spectrum_app.modules.rta.settings import RTASettings, RTASettingsWindow
from spectrum_app.modules.rta.types import PERIODIC_IFFT_GENERATOR
from spectrum_app.modules.rta.view import RTAView

if TYPE_CHECKING:
    from spectrum_app.application import SpectrumApplication


class RTAModule(BaseModule):
    id = "rta"
    name = "RTA"

    DEFAULT_STATE: dict[str, Any] = {
        "noise": True,
        "band": (20, 20_000),
        "level_db": 0.0,
        "window_width": 1.0,
        "window_hop": 0.1,
        "points": 31,
        "smoothing_octaves": 0.1,
        "recording": None,
        "result_frequency": None,
        "result_level_db": None,
        "status": "Ready",
    }
    REANALYZE_SETTINGS = {"points", "smoothing_octaves"}
    INVALIDATE_SETTINGS = {"band", "window_width"}

    def __init__(self) -> None:
        super().__init__()
        self._view: RTAView | None = None
        self._settings: RTASettings | None = None
        self._settings_window: RTASettingsWindow | None = None
        self._io_worker: RTAIOWorker | None = None
        self._analyzer: RTAAnalyzer | None = None
        self._runtime_lock = Lock()
        self._pending_level: tuple[int, Measurement, tuple[float, float]] | None = None
        self._pending_completion: tuple[int, Measurement, str | None, bool] | None = (
            None
        )
        self._pending_clipping: tuple[int, Measurement, str] | None = None
        self._revision = 0
        self._analysis_measurement: Measurement | None = None
        self._runtime_analysis_config: RTAConfig | None = None
        self._clipping_reported = False

    @property
    def settings(self) -> RTASettings:
        if self._settings is None:
            raise RuntimeError("RTA settings are not initialized")
        return self._settings

    def initialize(self, app: "SpectrumApplication") -> None:
        super().initialize(app)
        self._settings = RTASettings(app.settings, self._settings_changed)
        self._settings_window = RTASettingsWindow(app, self._settings)
        self._settings_window.build()
        self._view = RTAView(self)
        self._analyzer = RTAAnalyzer()
        self._analyzer.start()

    def activate(self, measurement: Measurement) -> None:
        super().activate(measurement)
        self._ensure_state(measurement.module_state)
        if self.settings.generator == PERIODIC_IFFT_GENERATOR:
            measurement.module_state["level_db"] = 0.0
        if self._view is None:
            raise RuntimeError("RTA module is not initialized")
        self._view.build(
            self.app.main_window.module_gui_host,
            self.app.main_window.bottom_host,
            measurement.module_state,
        )
        self._restore_graphs(measurement)

    def start_measurement(self) -> None:
        if self.app.app_state.measuring:
            return
        if self._io_worker is not None and self._io_worker.is_alive():
            self._set_status("RTA is still stopping")
            return
        try:
            measurement = self.measurement
            runtime = self._build_runtime_config(measurement.module_state)
            analysis = self._build_analysis_config(measurement.module_state)
            analysis.validate(self.app.audio_input.sample_rate)
            self._revision += 1
            revision = self._revision
            self._analysis_measurement = measurement
            self._runtime_analysis_config = analysis
            self._clipping_reported = False
            worker = RTAIOWorker(
                self.app.audio_input,
                self.app.audio_output,
                runtime,
                on_level=lambda levels: self._receive_level(
                    revision,
                    measurement,
                    levels,
                ),
                on_snapshot=lambda recording: self._receive_snapshot(
                    revision,
                    measurement,
                    recording,
                ),
                on_complete=lambda error, cancelled: self._receive_completion(
                    revision,
                    measurement,
                    error,
                    cancelled,
                ),
                on_clipping=lambda message: self._receive_clipping(
                    revision,
                    measurement,
                    message,
                ),
            )
            with self._runtime_lock:
                self._pending_level = None
                self._pending_completion = None
                self._pending_clipping = None
            self._io_worker = worker
            self.app.app_state.measuring = True
            self._set_controls_enabled(False)
            status = "RTA measurement started"
            if (
                runtime.noise
                and runtime.generator == PERIODIC_IFFT_GENERATOR
                and self.app.audio_input.sample_rate
                != self.app.audio_output.sample_rate
            ):
                status += "; warning: sample-rate mismatch may cause spectral leakage"
            self._set_status(status)
            worker.start()
        except Exception as error:
            self.app.app_state.measuring = False
            self._set_controls_enabled(True)
            self._show_error(str(error))

    def stop_measurement(self) -> None:
        worker = self._io_worker
        if worker is not None and worker.is_alive():
            self._set_status("Stopping RTA measurement...")
            worker.request_stop()
            return
        self.app.app_state.measuring = False
        self._set_controls_enabled(True)

    def update(self) -> None:
        if self._view is not None:
            self._view.update()
        self._process_analysis_response()
        with self._runtime_lock:
            level = self._pending_level
            completion = self._pending_completion
            clipping = self._pending_clipping
            self._pending_level = None
            self._pending_completion = None
            self._pending_clipping = None
        if level is not None:
            self._process_level(*level)
        if clipping is not None:
            self._process_clipping(*clipping)
        if completion is not None:
            self._process_completion(*completion)

    def deactivate(self) -> None:
        self._revision += 1
        self._stop_io_worker()
        self.app.app_state.measuring = False
        self._analysis_measurement = None
        self._runtime_analysis_config = None
        with self._runtime_lock:
            self._pending_level = None
            self._pending_completion = None
            self._pending_clipping = None
        if self._view is not None:
            self._view.destroy()
        super().deactivate()

    def shutdown(self) -> None:
        self._stop_io_worker()
        if self._analyzer is not None:
            self._analyzer.shutdown()
            self._analyzer = None
        if self._settings_window is not None:
            self._settings_window.destroy()
            self._settings_window = None
        self._settings = None
        self._view = None
        self.app.app_state.measuring = False
        super().shutdown()

    def set_setting(self, key: str, value: Any) -> Any:
        state = self.measurement.module_state
        normalized = self._normalize_setting(key, value)
        if state.get(key) == normalized:
            return normalized
        state[key] = normalized
        if key in self.INVALIDATE_SETTINGS:
            self._clear_measurement_data(self.measurement)
            self._set_status("RTA capture settings changed; measure again")
        elif key in self.REANALYZE_SETTINGS:
            if self._io_worker is not None and self._io_worker.is_alive():
                config = self._build_analysis_config(state)
                config.validate(self.app.audio_input.sample_rate)
                with self._runtime_lock:
                    self._runtime_analysis_config = config
                if key == "points":
                    self._restore_graphs(self.measurement)
                self._set_status("RTA analysis settings updated")
            elif isinstance(state.get("recording"), ASignal):
                self._submit_stored_recording(self.measurement)
        else:
            self._set_status("RTA setting will apply to the next measurement")
        return normalized

    def _receive_level(
        self,
        revision: int,
        measurement: Measurement,
        levels: tuple[float, float],
    ) -> None:
        with self._runtime_lock:
            self._pending_level = revision, measurement, levels

    def _receive_snapshot(
        self,
        revision: int,
        measurement: Measurement,
        recording: ASignal,
    ) -> None:
        analyzer = self._analyzer
        with self._runtime_lock:
            config = self._runtime_analysis_config
        if analyzer is None or config is None or revision != self._revision:
            return
        self._analysis_measurement = measurement
        analyzer.submit(revision, recording, config)

    def _receive_completion(
        self,
        revision: int,
        measurement: Measurement,
        error: str | None,
        cancelled: bool,
    ) -> None:
        with self._runtime_lock:
            self._pending_completion = revision, measurement, error, cancelled

    def _receive_clipping(
        self,
        revision: int,
        measurement: Measurement,
        message: str,
    ) -> None:
        with self._runtime_lock:
            self._pending_clipping = revision, measurement, message

    def _process_level(
        self,
        revision: int,
        measurement: Measurement,
        levels: tuple[float, float],
    ) -> None:
        if revision != self._revision or measurement is not self._active_measurement():
            return
        if self._view is not None:
            self._view.update_levels(levels)

    def _process_clipping(
        self,
        revision: int,
        measurement: Measurement,
        message: str,
    ) -> None:
        if revision != self._revision or measurement is not self._active_measurement():
            return
        self._clipping_reported = True
        worker = self._io_worker
        if worker is not None:
            worker.request_stop()
        self._show_error(message)

    def _process_completion(
        self,
        revision: int,
        measurement: Measurement,
        error: str | None,
        cancelled: bool,
    ) -> None:
        if revision != self._revision:
            return
        self._io_worker = None
        self.app.app_state.measuring = False
        self._set_controls_enabled(True)
        if measurement is not self._active_measurement():
            return
        if error is not None:
            if not self._clipping_reported:
                self._show_error(error)
        elif cancelled:
            self._set_status("RTA measurement stopped")
        else:
            self._set_status("RTA measurement completed")

    def _process_analysis_response(self) -> None:
        if self._analyzer is None:
            return
        response = self._analyzer.poll()
        if response is None:
            return
        revision, recording, result, error = response
        measurement = self._analysis_measurement
        if (
            revision != self._revision
            or measurement is None
            or measurement is not self._active_measurement()
        ):
            return
        if error is not None or result is None:
            self._show_error(error or "RTA analysis returned no result")
            return
        self._store_result(measurement, recording, result)

    def _store_result(
        self,
        measurement: Measurement,
        recording: ASignal,
        result: RTAResult,
    ) -> None:
        state = measurement.module_state
        state["recording"] = recording
        state["result_frequency"] = result.frequency
        state["result_level_db"] = result.level_db
        self._update_graphs(measurement, result.frequency, result.level_db)

    def _restore_graphs(self, measurement: Measurement) -> None:
        state = measurement.module_state
        frequency = state.get("result_frequency")
        level_db = state.get("result_level_db")
        if frequency is not None and level_db is not None:
            self._update_graphs(
                measurement,
                np.asarray(frequency, dtype=np.float64),
                np.asarray(level_db, dtype=np.float64),
            )

    def _update_graphs(
        self,
        measurement: Measurement,
        frequency: np.ndarray,
        level_db: np.ndarray,
    ) -> None:
        measurement.remember_graph_colors()
        values = np.asarray(level_db, dtype=np.float64)
        if values.ndim == 1:
            values = values[:, None]
        mode = self.settings.mode
        desired = (
            [("RTA", values[:, 0])]
            if mode == "mono"
            else [("A", values[:, 0]), ("B", values[:, min(1, values.shape[1] - 1)])]
        )
        plot_type = self._effective_plot_type(int(measurement.module_state["points"]))
        existing = {graph.name: graph for graph in measurement.graphs}
        stale_ids = {
            graph.id
            for graph in measurement.graphs
            if graph.name not in {name for name, _ in desired}
        }
        graphs: list[GraphData] = []
        new_graph_ids: list[str] = []
        for name, values_y in desired:
            graph = existing.get(name)
            if graph is None:
                graph = GraphData(
                    name=name,
                    x=frequency,
                    y=values_y,
                    x_axis=AxisSpec.FREQ,
                    y_axis=AxisSpec.LEVEL,
                    plot_type=plot_type,
                    color=measurement.color_for_graph(name),
                )
                new_graph_ids.append(graph.id)
            else:
                graph.x = frequency
                graph.y = values_y
                graph.x_axis = AxisSpec.FREQ
                graph.y_axis = AxisSpec.LEVEL
                graph.plot_type = plot_type
            graphs.append(graph)
        measurement.graphs = graphs
        self.app.app_state.visible_graph_ids = [
            graph_id
            for graph_id in self.app.app_state.visible_graph_ids
            if graph_id not in stale_ids
        ]
        for graph_id in new_graph_ids:
            if graph_id not in self.app.app_state.visible_graph_ids:
                self.app.app_state.visible_graph_ids.append(graph_id)
        self.app.app_state.graph_data_changed = True

    def _effective_plot_type(self, points: int) -> PlotType:
        if self.settings.mode == "stereo":
            return PlotType.LINE
        setting = self.settings.plot_type
        if setting == "bars" or (setting == "auto" and points < 100):
            return PlotType.BARS
        return PlotType.LINE

    def _settings_changed(self, key: str) -> None:
        if key == "generator":
            for item in self.app.app_state.measurements:
                if item.module_id == self.id:
                    item.module_state["level_db"] = 0.0
            if self._view is not None:
                self._view.update_generator_visibility(self.settings.generator)
            if self._active_measurement() is not None:
                self._set_status("RTA generator changed; level reset to 0 dB")
            return
        measurement = self._active_measurement()
        if measurement is None:
            return
        if key in ("plot_type", "mode"):
            if self._view is not None:
                self._view.update_smoothing_visibility(
                    int(measurement.module_state["points"])
                )
            if isinstance(measurement.module_state.get("recording"), ASignal):
                self._submit_stored_recording(measurement)
            else:
                self._restore_graphs(measurement)
        elif key == "window_function" and isinstance(
            measurement.module_state.get("recording"), ASignal
        ):
            self._submit_stored_recording(measurement)
        else:
            self._set_status("RTA setting will apply to the next measurement")

    def _submit_stored_recording(self, measurement: Measurement) -> None:
        recording = measurement.module_state.get("recording")
        if not isinstance(recording, ASignal) or self._analyzer is None:
            return
        self._revision += 1
        revision = self._revision
        self._analysis_measurement = measurement
        config = self._build_analysis_config(measurement.module_state)
        self._runtime_analysis_config = config
        self._set_status("Recalculating RTA...")
        self._analyzer.submit(revision, recording, config)

    def _build_runtime_config(self, state: dict[str, Any]) -> RTARuntimeConfig:
        band = FrequencyBand(cast(tuple[float, float], state["band"]))
        input_rate = self.app.audio_input.sample_rate
        if input_rate <= 0:
            raise ValueError("Select an audio input device")
        band.validate(nyquist=input_rate / 2.0)
        if bool(state["noise"]) and self.app.audio_output.sample_rate <= 0:
            raise ValueError("Select an audio output device")
        return RTARuntimeConfig(
            band=band,
            noise=bool(state["noise"]),
            generator=self.settings.generator,
            level_db=(
                0.0
                if self.settings.generator == PERIODIC_IFFT_GENERATOR
                else float(state["level_db"])
            ),
            window_seconds=float(state["window_width"]),
            hop_seconds=float(state["window_hop"]),
            pre_silence=self.settings.pre_silence,
            fade_in=self.settings.fade_in,
            fade_out=self.settings.fade_out,
        )

    def _build_analysis_config(self, state: dict[str, Any]) -> RTAConfig:
        points = int(state["points"])
        band = FrequencyBand(cast(tuple[float, float], state["band"]))
        if self._effective_plot_type(points) == PlotType.BARS:
            width = float(np.log2(band.high / band.low) / (points - 1))
        else:
            width = float(state["smoothing_octaves"])
        return RTAConfig(
            band=band,
            points=points,
            smoothing_width=width,
            fft_window=self.settings.window_function,
        )

    def _clear_measurement_data(self, measurement: Measurement) -> None:
        measurement.remember_graph_colors()
        state = measurement.module_state
        state["recording"] = None
        state["result_frequency"] = None
        state["result_level_db"] = None
        graph_ids = {graph.id for graph in measurement.graphs}
        measurement.graphs.clear()
        self.app.app_state.visible_graph_ids = [
            graph_id
            for graph_id in self.app.app_state.visible_graph_ids
            if graph_id not in graph_ids
        ]
        self.app.app_state.graph_data_changed = True

    def _stop_io_worker(self) -> None:
        worker = self._io_worker
        if worker is None:
            return
        if worker.is_alive():
            worker.request_stop()
            worker.join(timeout=max(2.0, self.settings.fade_out + 1.0))
        if worker.is_alive():
            worker.abort()
            worker.join(timeout=1.0)
        self._io_worker = None

    def _active_measurement(self) -> Measurement | None:
        try:
            return self.measurement
        except RuntimeError:
            return None

    def _set_controls_enabled(self, enabled: bool) -> None:
        if self._view is not None:
            self._view.set_enabled(enabled)

    def _set_status(self, text: str) -> None:
        measurement = self._active_measurement()
        if measurement is not None:
            measurement.module_state["status"] = text
        self.app.main_window.set_status_text(text)

    def _show_error(self, message: str) -> None:
        clipping = "clipping" in message.lower()
        self._set_status("RTA stopped: clipping" if clipping else "RTA failed")
        self.app.main_window.show_error(
            "RTA clipping" if clipping else "RTA error",
            message,
        )

    @classmethod
    def _ensure_state(cls, state: dict[str, Any]) -> None:
        for key, value in cls.DEFAULT_STATE.items():
            state.setdefault(key, deepcopy(value))

    @staticmethod
    def _normalize_setting(key: str, value: Any) -> Any:
        if key == "noise":
            return bool(value)
        if key == "band":
            low, high = map(int, value)
            low = max(1, low)
            return low, max(low + 1, high)
        if key == "level_db":
            return min(10.0, max(-10.0, float(value)))
        if key == "window_width":
            return min(60.0, max(0.01, float(value)))
        if key == "window_hop":
            return min(10.0, max(0.001, float(value)))
        if key == "points":
            return min(2048, max(24, int(value)))
        if key == "smoothing_octaves":
            return min(4.0, max(0.01, float(value)))
        raise ValueError(f"Unknown RTA setting: {key}")
