from copy import deepcopy
from threading import Lock
from typing import TYPE_CHECKING, Any

import numpy as np

from audioanalysis import (
    ASignal,
    FrequencyBand,
    PhaseConfig,
    PhaseResult,
)
from spectrum_app.core.model import AxisSpec, GraphData, Measurement
from spectrum_app.modules.base import BaseModule
from spectrum_app.modules.phase.jobs import PhaseAcquisition, PhaseAnalyzer
from spectrum_app.modules.phase.settings import PhaseSettings, PhaseSettingsWindow
from spectrum_app.modules.phase.view import PhaseView

if TYPE_CHECKING:
    from spectrum_app.application import SpectrumApplication


class PhaseModule(BaseModule):
    id = "phase"
    name = "Phase"
    MINIMUM_INPUT_LEVEL = 10.0 ** (-60.0 / 20.0)

    DEFAULT_STATE: dict[str, Any] = {
        "band": (20, 20_000),
        "duration": 5.0,
        "smoothing_octaves": 1.0 / 3.0,
        "points": 1024,
        "delay_fit_band": (80, 15_000),
        "delay_correction_meters": 0.0,
        "recording": None,
        "generator": None,
        "level_time": np.empty(0, dtype=np.float64),
        "level_values": np.empty((0, 2), dtype=np.float64),
        "result_frequency": None,
        "result_magnitude_db": None,
        "result_phase_degrees": None,
        "estimated_delay_seconds": None,
        "estimated_delay_meters": None,
        "analysis_signature": None,
        "status": "Ready",
    }
    CAPTURE_SETTINGS = {"band", "duration"}
    ANALYSIS_SETTINGS = {
        "smoothing_octaves",
        "points",
        "delay_fit_band",
        "delay_correction_meters",
    }

    def __init__(self) -> None:
        super().__init__()
        self._view: PhaseView | None = None
        self._settings: PhaseSettings | None = None
        self._settings_window: PhaseSettingsWindow | None = None
        self._acquisition: PhaseAcquisition | None = None
        self._analyzer: PhaseAnalyzer | None = None
        self._runtime_lock = Lock()
        self._pending_level: (
            tuple[int, Measurement, np.ndarray, np.ndarray, tuple[float, float]] | None
        ) = None
        self._pending_completion: (
            tuple[
                int,
                Measurement,
                ASignal | None,
                ASignal | None,
                np.ndarray,
                np.ndarray,
                str | None,
                bool,
            ]
            | None
        ) = None
        self._capture_revision = 0
        self._analysis_revision = 0
        self._current_analysis_revision = 0
        self._finishing_revision: int | None = None
        self._analysis_measurement: Measurement | None = None
        self._reanalysis_requested = False

    def initialize(self, app: "SpectrumApplication") -> None:
        super().initialize(app)
        self._settings = PhaseSettings(app.settings, self._settings_changed)
        self._settings_window = PhaseSettingsWindow(app, self._settings)
        self._settings_window.build()
        self._view = PhaseView(self)
        self._analyzer = PhaseAnalyzer()
        self._analyzer.start()

    def activate(self, measurement: Measurement) -> None:
        super().activate(measurement)
        self._ensure_state(measurement.module_state)
        self._constrain_delay_fit(measurement.module_state)
        if self._view is None:
            raise RuntimeError("Phase module is not initialized")
        self._view.build(
            self.app.main_window.module_gui_host,
            self.app.main_window.bottom_host,
            measurement.module_state,
        )
        recording = measurement.module_state.get("recording")
        if isinstance(recording, ASignal):
            signature = self._analysis_signature(
                self._build_config(recording.sample_rate)
            )
            if (
                not measurement.graphs
                or measurement.module_state.get("analysis_signature") != signature
            ):
                self._reanalysis_requested = True

    def start_measurement(self) -> None:
        if self.app.app_state.measuring:
            return
        if self._acquisition is not None and self._acquisition.is_alive():
            self._set_status("Phase measurement is still stopping")
            return
        try:
            self._validate_audio_settings()
            measurement = self.measurement
            state = measurement.module_state
            self._capture_revision += 1
            revision = self._capture_revision
            acquisition = PhaseAcquisition(
                self.app.audio_input,
                self.app.audio_output,
                band=FrequencyBand(*state["band"]),
                duration=float(state["duration"]),
                pre_silence=self.settings.pre_silence,
                post_silence=self.settings.post_silence,
                fade=self.settings.fade,
                on_level=lambda times, levels, current: self._receive_level(
                    revision,
                    measurement,
                    times,
                    levels,
                    current,
                ),
                on_complete=lambda recording,
                generator,
                times,
                levels,
                error,
                cancelled: self._receive_completion(
                    revision,
                    measurement,
                    recording,
                    generator,
                    times,
                    levels,
                    error,
                    cancelled,
                ),
            )
            with self._runtime_lock:
                self._pending_level = None
                self._pending_completion = None
            self._invalidate_analysis()
            self._clear_level_history(measurement)
            self._acquisition = acquisition
            self.app.app_state.measuring = True
            self._set_controls_enabled(False)
            self._set_status("Phase measurement started")
            acquisition.start()
        except Exception as error:
            self.app.app_state.measuring = False
            self._set_controls_enabled(True)
            self._set_status(f"Phase error: {error}")

    def stop_measurement(self) -> None:
        acquisition = self._acquisition
        if acquisition is not None and acquisition.is_alive():
            self._set_status("Stopping Phase measurement...")
            acquisition.stop()
            return
        if self._finishing_revision is not None:
            self._invalidate_analysis()
            self.app.app_state.measuring = False
            self._set_controls_enabled(True)
            self._set_status("Phase calculation stopped")
            return
        self.app.app_state.measuring = False
        self._set_controls_enabled(True)

    def update(self) -> None:
        if self._view is not None:
            self._view.update()
        self._process_analysis_response()
        level, completion = self._take_worker_updates()
        if level is not None:
            self._process_level(*level)
        if completion is not None:
            self._process_completion(*completion)
        if self._reanalysis_requested and not self.app.app_state.measuring:
            self._reanalysis_requested = False
            self._reanalyze_stored_recording()

    def deactivate(self) -> None:
        self._capture_revision += 1
        acquisition = self._acquisition
        if acquisition is not None and acquisition.is_alive():
            acquisition.stop()
            acquisition.join(timeout=2.0)
        self.app.audio_input.close()
        self.app.audio_output.close()
        self.app.app_state.measuring = False
        self._acquisition = None
        self._invalidate_analysis()
        self._reanalysis_requested = False
        with self._runtime_lock:
            self._pending_level = None
            self._pending_completion = None
        if self._view is not None:
            self._view.destroy()
        super().deactivate()

    def shutdown(self) -> None:
        acquisition = self._acquisition
        if acquisition is not None and acquisition.is_alive():
            acquisition.stop()
            acquisition.join(timeout=2.0)
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

    @property
    def settings(self) -> PhaseSettings:
        if self._settings is None:
            raise RuntimeError("Phase settings are not initialized")
        return self._settings

    def set_setting(self, key: str, value: Any) -> Any:
        state = self.measurement.module_state
        normalized = self._normalize_setting(key, value, state)
        if state.get(key) == normalized:
            return normalized
        state[key] = normalized
        if key == "band":
            self._constrain_delay_fit(state)
        if key in self.CAPTURE_SETTINGS:
            self._clear_measurement_data(self.measurement)
            self._set_status("Phase measurement settings changed; measure again")
        elif key in self.ANALYSIS_SETTINGS and isinstance(
            state.get("recording"), ASignal
        ):
            self._reanalysis_requested = True
            self._set_status("Recalculating Phase...")
        return normalized

    def _settings_changed(self, key: str) -> None:
        if self._active_measurement() is not None:
            self._set_status(
                "Phase generator settings will apply to the next measurement"
            )

    def _receive_level(
        self,
        revision: int,
        measurement: Measurement,
        times: np.ndarray,
        levels: np.ndarray,
        current: tuple[float, float],
    ) -> None:
        with self._runtime_lock:
            self._pending_level = revision, measurement, times, levels, current

    def _receive_completion(
        self,
        revision: int,
        measurement: Measurement,
        recording: ASignal | None,
        generator: ASignal | None,
        times: np.ndarray,
        levels: np.ndarray,
        error: str | None,
        cancelled: bool,
    ) -> None:
        with self._runtime_lock:
            self._pending_completion = (
                revision,
                measurement,
                recording,
                generator,
                times,
                levels,
                error,
                cancelled,
            )

    def _take_worker_updates(self):
        with self._runtime_lock:
            level = self._pending_level
            completion = self._pending_completion
            self._pending_level = None
            self._pending_completion = None
        return level, completion

    def _process_level(
        self,
        revision: int,
        measurement: Measurement,
        times: np.ndarray,
        levels: np.ndarray,
        current: tuple[float, float],
    ) -> None:
        if (
            revision != self._capture_revision
            or measurement is not self._active_measurement()
        ):
            return
        measurement.module_state["level_time"] = times
        measurement.module_state["level_values"] = levels
        if self._view is not None:
            self._view.update_levels(current)

    def _process_completion(
        self,
        revision: int,
        measurement: Measurement,
        recording: ASignal | None,
        generator: ASignal | None,
        times: np.ndarray,
        levels: np.ndarray,
        error: str | None,
        cancelled: bool,
    ) -> None:
        if revision != self._capture_revision:
            return
        self._acquisition = None
        if measurement is not self._active_measurement():
            self.app.app_state.measuring = False
            return
        current = (
            (float(levels[-1, 0]), float(levels[-1, 1])) if levels.size else (0.0, 0.0)
        )
        self._process_level(revision, measurement, times, levels, current)
        if cancelled:
            self.app.app_state.measuring = False
            self._set_controls_enabled(True)
            self._set_status("Phase measurement stopped")
            return
        if error is not None or recording is None or generator is None:
            self.app.app_state.measuring = False
            self._set_controls_enabled(True)
            self._set_status(f"Phase error: {error or 'Audio recording is empty'}")
            return
        peaks = recording.max()
        labels = ("A", "B")
        for index, label in enumerate(labels):
            if float(peaks[index]) >= 0.999:
                self.app.app_state.measuring = False
                self._set_controls_enabled(True)
                self._set_status(f"Phase error: clipping detected on input {label}")
                return
            if float(peaks[index]) < self.MINIMUM_INPUT_LEVEL:
                self.app.app_state.measuring = False
                self._set_controls_enabled(True)
                self._set_status(f"Phase error: input {label} is below -60 dBFS")
                return

        measurement.module_state["recording"] = recording
        measurement.module_state["generator"] = generator
        self._set_status("Calculating Phase...")
        self._submit_analysis(measurement, recording, finishing=True)

    def _submit_analysis(
        self,
        measurement: Measurement,
        recording: ASignal,
        *,
        finishing: bool,
    ) -> None:
        if self._analyzer is None:
            raise RuntimeError("Phase analyzer is not initialized")
        try:
            config = self._build_config(recording.sample_rate)
            config.validate(recording.sample_rate)
        except Exception as error:
            self._set_status(f"Phase error: {error}")
            if finishing:
                self.app.app_state.measuring = False
                self._set_controls_enabled(True)
            return
        self._analysis_revision += 1
        revision = self._analysis_revision
        self._current_analysis_revision = revision
        self._analysis_measurement = measurement
        self._finishing_revision = revision if finishing else None
        self._analyzer.submit(revision, finishing, recording, config)

    def _process_analysis_response(self) -> None:
        if self._analyzer is None:
            return
        response = self._analyzer.poll()
        if response is None:
            return
        revision, finishing, result, error = response
        if revision != self._current_analysis_revision:
            return
        measurement = self._analysis_measurement
        if measurement is not None and measurement is self._active_measurement():
            if error is not None or result is None:
                self._set_status(f"Phase error: {error or 'Calculation failed'}")
            else:
                self._store_result(measurement, result)
                self._update_graph(measurement, result)
                self._set_status(
                    "Phase delay: "
                    f"{result.estimated_delay_seconds * 1000.0:.3f} ms, "
                    f"{result.estimated_delay_meters:.3f} m"
                )
        if finishing and revision == self._finishing_revision:
            self._finishing_revision = None
            self.app.app_state.measuring = False
            self._set_controls_enabled(True)

    def _store_result(self, measurement: Measurement, result: PhaseResult) -> None:
        state = measurement.module_state
        state["result_frequency"] = result.frequency
        state["result_magnitude_db"] = result.magnitude_db
        state["result_phase_degrees"] = result.phase_degrees
        state["estimated_delay_seconds"] = result.estimated_delay_seconds
        state["estimated_delay_meters"] = result.estimated_delay_meters
        state["analysis_signature"] = self._analysis_signature(
            self._build_config(measurement.module_state["recording"].sample_rate)
        )
        if self._view is not None:
            self._view.update_result(
                result.frequency,
                result.magnitude_db,
                result.estimated_delay_seconds,
                result.estimated_delay_meters,
                band=state["band"],
            )

    def _reanalyze_stored_recording(self) -> None:
        measurement = self._active_measurement()
        if measurement is None:
            return
        recording = measurement.module_state.get("recording")
        if isinstance(recording, ASignal):
            self._set_status("Recalculating Phase...")
            self._submit_analysis(measurement, recording, finishing=False)

    def _build_config(self, sample_rate: int) -> PhaseConfig:
        state = self.measurement.module_state
        return PhaseConfig(
            band=FrequencyBand(*state["band"]),
            delay_fit_band=FrequencyBand(*state["delay_fit_band"]),
            points=int(state["points"]),
            smoothing_octaves=float(state["smoothing_octaves"]),
            delay_correction_meters=float(state["delay_correction_meters"]),
            minimum_a_db=-60.0,
            minimum_b_db=-60.0,
        )

    def _validate_audio_settings(self) -> None:
        input_rate = self.app.audio_input.sample_rate
        output_rate = self.app.audio_output.sample_rate
        if input_rate <= 0:
            raise ValueError("Audio input device is unavailable")
        if output_rate <= 0:
            raise ValueError("Audio output device is unavailable")
        state = self.measurement.module_state
        band = FrequencyBand(*state["band"])
        band.validate(nyquist=min(input_rate, output_rate) / 2)
        self._build_config(input_rate).validate(input_rate)

    @staticmethod
    def _analysis_signature(config: PhaseConfig) -> tuple[object, ...]:
        return (
            config.band,
            config.delay_fit_band,
            config.points,
            config.smoothing_octaves,
            config.delay_correction_meters,
        )

    def _update_graph(self, measurement: Measurement, result: PhaseResult) -> None:
        if measurement.graphs:
            graph = measurement.graphs[0]
            graph.name = "Phase"
            graph.x = result.frequency
            graph.y = result.phase_degrees
            graph.x_axis = AxisSpec.FREQ
            graph.y_axis = AxisSpec.PHASE
        else:
            graph = GraphData(
                name="Phase",
                x=result.frequency,
                y=result.phase_degrees,
                x_axis=AxisSpec.FREQ,
                y_axis=AxisSpec.PHASE,
            )
            measurement.graphs.append(graph)
        if graph.id not in self.app.app_state.visible_graph_ids:
            self.app.app_state.visible_graph_ids.append(graph.id)
        self.app.app_state.graph_data_changed = True

    def _clear_measurement_data(self, measurement: Measurement) -> None:
        state = measurement.module_state
        for key in (
            "recording",
            "generator",
            "result_frequency",
            "result_magnitude_db",
            "result_phase_degrees",
            "estimated_delay_seconds",
            "estimated_delay_meters",
            "analysis_signature",
        ):
            state[key] = None
        graph_ids = {graph.id for graph in measurement.graphs}
        measurement.graphs.clear()
        self.app.app_state.visible_graph_ids = [
            graph_id
            for graph_id in self.app.app_state.visible_graph_ids
            if graph_id not in graph_ids
        ]
        self.app.app_state.graph_data_changed = True
        if self._view is not None and measurement is self._active_measurement():
            self._view.update_result(None, None, None, None, band=state["band"])

    def _clear_level_history(self, measurement: Measurement) -> None:
        measurement.module_state["level_time"] = np.empty(0, dtype=np.float64)
        measurement.module_state["level_values"] = np.empty((0, 2), dtype=np.float64)
        if self._view is not None and measurement is self._active_measurement():
            self._view.update_levels((0.0, 0.0))

    def _invalidate_analysis(self) -> None:
        self._analysis_revision += 1
        self._current_analysis_revision = self._analysis_revision
        self._finishing_revision = None
        self._analysis_measurement = None

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

    @classmethod
    def _ensure_state(cls, state: dict[str, Any]) -> None:
        for key, value in cls.DEFAULT_STATE.items():
            state.setdefault(key, deepcopy(value))

    @staticmethod
    def _constrain_delay_fit(state: dict[str, Any]) -> tuple[int, int]:
        band_low, band_high = map(int, state["band"])
        fit_low, fit_high = map(int, state["delay_fit_band"])
        fit_low = min(max(band_low, fit_low), band_high - 1)
        fit_high = max(fit_low + 1, min(band_high, fit_high))
        state["delay_fit_band"] = fit_low, fit_high
        return fit_low, fit_high

    @staticmethod
    def _normalize_setting(key: str, value: Any, state: dict[str, Any]) -> Any:
        if key == "band":
            low, high = int(value[0]), int(value[1])
            low = max(1, low)
            return low, max(low + 1, high)
        if key == "duration":
            return min(600.0, max(0.1, float(value)))
        if key == "smoothing_octaves":
            return min(3.0, max(0.01, float(value)))
        if key == "points":
            return min(100_000, max(2, int(value)))
        if key == "delay_fit_band":
            band_low, band_high = map(int, state["band"])
            low, high = int(value[0]), int(value[1])
            low = min(max(band_low, low), band_high - 1)
            return low, max(low + 1, min(band_high, high))
        if key == "delay_correction_meters":
            return min(1_000.0, max(-1_000.0, float(value)))
        raise ValueError(f"Unknown Phase setting: {key}")
