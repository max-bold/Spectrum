from copy import deepcopy
from threading import Lock
from typing import TYPE_CHECKING, Any

import numpy as np

from audioanalysis import (
    ASignal,
    FrequencyBand,
    SemiAnalogTHDConfig,
    SemiAnalogTHDResult,
)
from spectrum_app.core.model import AxisSpec, GraphData, Measurement
from spectrum_app.modules.base import BaseModule
from spectrum_app.modules.thd.jobs import THDAcquisition, THDAnalyzer
from spectrum_app.modules.thd.settings import THDSettings, THDSettingsWindow
from spectrum_app.modules.thd.view import THDView

if TYPE_CHECKING:
    from spectrum_app.application import SpectrumApplication


class THDModule(BaseModule):
    id = "thd"
    name = "THD+N"

    DEFAULT_STATE: dict[str, Any] = {
        "band": (20, 20_000),
        "duration": 30.0,
        "smoothing_octaves": 0.1,
        "recording": None,
        "generator": None,
        "level_time": np.empty(0, dtype=np.float64),
        "level_values": np.empty(0, dtype=np.float64),
        "result_frequency": None,
        "result_ratio": None,
        "integrated_ratio": None,
        "analysis_signature": None,
        "status": "Ready",
    }
    CAPTURE_SETTINGS = {"band", "duration"}

    def __init__(self) -> None:
        super().__init__()
        self._view: THDView | None = None
        self._settings: THDSettings | None = None
        self._settings_window: THDSettingsWindow | None = None
        self._acquisition: THDAcquisition | None = None
        self._analyzer: THDAnalyzer | None = None
        self._runtime_lock = Lock()
        self._pending_level: (
            tuple[int, Measurement, np.ndarray, np.ndarray, float] | None
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
        self._stop_requested = False

    def initialize(self, app: "SpectrumApplication") -> None:
        super().initialize(app)
        self._settings = THDSettings(app.settings, self._settings_changed)
        self._settings_window = THDSettingsWindow(app, self._settings)
        self._settings_window.build()
        self._view = THDView(self)
        self._analyzer = THDAnalyzer()
        self._analyzer.start()

    def activate(self, measurement: Measurement) -> None:
        super().activate(measurement)
        self._ensure_state(measurement.module_state)
        if self._view is None:
            raise RuntimeError("THD module is not initialized")
        self._view.build(
            self.app.main_window.module_gui_host,
            self.app.main_window.bottom_host,
            measurement.module_state,
        )
        self._view.update_levels(
            np.asarray(measurement.module_state["level_time"], dtype=np.float64),
            np.asarray(measurement.module_state["level_values"], dtype=np.float64),
            0.0,
            duration=self._recording_duration(measurement),
        )
        recording = measurement.module_state.get("recording")
        if isinstance(recording, ASignal):
            current_signature = self._analysis_signature(
                self._build_config(recording.sample_rate)
            )
            if (
                not measurement.graphs
                or measurement.module_state.get("analysis_signature")
                != current_signature
            ):
                self._reanalysis_requested = True

    def start_measurement(self) -> None:
        if self.app.app_state.measuring:
            return
        if self._acquisition is not None and self._acquisition.is_alive():
            self._set_status("THD measurement is still stopping")
            return
        try:
            output_config = self._validate_audio_and_build_output_config()
            measurement = self.measurement
            self._capture_revision += 1
            revision = self._capture_revision
            acquisition = THDAcquisition(
                self.app.audio_input,
                self.app.audio_output,
                output_config,
                on_level=lambda times, levels, current: self._receive_level(
                    revision,
                    measurement,
                    times,
                    levels,
                    current,
                ),
                on_complete=lambda recording, generator, times, levels, error, cancelled: self._receive_completion(
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
            self._acquisition = acquisition
            self._stop_requested = False
            self._clear_level_history(measurement)
            self.app.app_state.measuring = True
            self._set_controls_enabled(False)
            self._set_status("THD measurement started")
            acquisition.start()
        except Exception as error:
            self.app.app_state.measuring = False
            self._set_controls_enabled(True)
            self._show_error(str(error))

    def stop_measurement(self) -> None:
        acquisition = self._acquisition
        if acquisition is not None and acquisition.is_alive():
            self._stop_requested = True
            self._set_status("Stopping THD measurement...")
            acquisition.request_stop()
            return
        if self._finishing_revision is not None:
            self._invalidate_analysis()
            self.app.app_state.measuring = False
            self._set_controls_enabled(True)
            self._set_status("THD calculation stopped")
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
            acquisition.request_stop()
            acquisition.join()
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
            acquisition.request_stop()
            acquisition.join()
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
        if key in self.CAPTURE_SETTINGS:
            self._clear_measurement_data(self.measurement)
            self._set_status("THD measurement settings changed; measure again")
        elif key == "smoothing_octaves" and isinstance(
            state.get("recording"), ASignal
        ):
            self._reanalysis_requested = True
            self._set_status("Recalculating THD+N...")
        return normalized

    @property
    def settings(self) -> THDSettings:
        if self._settings is None:
            raise RuntimeError("THD settings are not initialized")
        return self._settings

    def _settings_changed(self, key: str) -> None:
        measurement = self._active_measurement()
        if measurement is None:
            return
        if key in ("fade_in_seconds", "fade_out_seconds"):
            self._set_status("Sweep fades will apply to new THD measurements")
            return
        if isinstance(measurement.module_state.get("recording"), ASignal):
            self._reanalysis_requested = True
            self._set_status("Recalculating THD+N...")

    def _receive_level(
        self,
        revision: int,
        measurement: Measurement,
        times: np.ndarray,
        levels: np.ndarray,
        current: float,
    ) -> None:
        with self._runtime_lock:
            self._pending_level = (
                revision,
                measurement,
                times,
                levels,
                current,
            )

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
        current: float,
    ) -> None:
        if revision != self._capture_revision or measurement is not self._active_measurement():
            return
        measurement.module_state["level_time"] = times
        measurement.module_state["level_values"] = levels
        if self._view is not None:
            self._view.update_levels(
                times,
                levels,
                current,
                duration=self._recording_duration(measurement),
            )

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
        self._process_level(revision, measurement, times, levels, 0.0)
        if cancelled:
            self.app.app_state.measuring = False
            self._set_controls_enabled(True)
            self._set_status("THD measurement stopped")
            return
        if error is not None or recording is None or generator is None:
            self.app.app_state.measuring = False
            self._set_controls_enabled(True)
            self._show_error(error or "Audio recording is empty")
            return
        peak = float(recording.max()[0])
        if peak >= 0.999:
            self.app.app_state.measuring = False
            self._set_controls_enabled(True)
            self._show_error("Input clipping detected on channel A")
            return
        if peak < 1e-4:
            self.app.app_state.measuring = False
            self._set_controls_enabled(True)
            self._show_error("Input A signal is too quiet")
            return

        state = measurement.module_state
        state["recording"] = recording
        state["generator"] = generator
        self._set_status("Calculating THD+N...")
        self._submit_analysis(measurement, recording, finishing=True)

    def _submit_analysis(
        self,
        measurement: Measurement,
        recording: ASignal,
        *,
        finishing: bool,
    ) -> None:
        if self._analyzer is None:
            raise RuntimeError("THD analyzer is not initialized")
        try:
            config = self._build_config(recording.sample_rate)
            config.validate()
        except Exception as error:
            self._show_error(str(error))
            if finishing:
                self.app.app_state.measuring = False
                self._set_controls_enabled(True)
            return

        analysis_signature = self._analysis_signature(config)
        self._analysis_revision += 1
        revision = self._analysis_revision
        self._current_analysis_revision = revision
        self._analysis_measurement = measurement
        self._finishing_revision = revision if finishing else None
        self._analyzer.submit(
            revision,
            finishing,
            recording,
            config,
            analysis_signature,
        )

    def _process_analysis_response(self) -> None:
        if self._analyzer is None:
            return
        response = self._analyzer.poll()
        if response is None:
            return
        (
            revision,
            finishing,
            result,
            analysis_signature,
            error,
        ) = response
        if revision != self._current_analysis_revision:
            return
        measurement = self._analysis_measurement
        if measurement is not None and measurement is self._active_measurement():
            if error is not None or result is None:
                self._show_error(error or "Calculation failed")
            else:
                state = measurement.module_state
                state["result_frequency"] = result.frequency
                state["result_ratio"] = result.ratio
                state["integrated_ratio"] = result.integrated_ratio
                state["analysis_signature"] = analysis_signature
                self._update_graph(measurement, result)
                self._set_status(
                    f"THD+N: {result.integrated_percent:.4g}%"
                )
        if finishing and revision == self._finishing_revision:
            self._finishing_revision = None
            self.app.app_state.measuring = False
            self._set_controls_enabled(True)

    def _reanalyze_stored_recording(self) -> None:
        measurement = self._active_measurement()
        if measurement is None:
            return
        recording = measurement.module_state.get("recording")
        if not isinstance(recording, ASignal):
            return
        self._set_status("Recalculating THD+N...")
        self._submit_analysis(measurement, recording, finishing=False)

    def _build_config(
        self,
        sample_rate: int,
        *,
        fade_in_seconds: float = 0.0,
        fade_out_seconds: float = 0.0,
    ) -> SemiAnalogTHDConfig:
        state = self.measurement.module_state
        return SemiAnalogTHDConfig(
            sample_rate=sample_rate,
            duration=float(state["duration"]),
            band=FrequencyBand(*state["band"]),
            smoothing_octaves=float(state["smoothing_octaves"]),
            segment_seconds=self.settings.segment_seconds,
            overlap=self.settings.overlap_percent / 100.0,
            fade_in_seconds=fade_in_seconds,
            fade_out_seconds=fade_out_seconds,
            notch_ratio=self.settings.notch_ratio,
            points=self.settings.points,
        )

    def _validate_audio_and_build_output_config(self) -> SemiAnalogTHDConfig:
        input_rate = self.app.audio_input.sample_rate
        output_rate = self.app.audio_output.sample_rate
        if input_rate <= 0:
            raise ValueError("Audio input device is unavailable")
        if output_rate <= 0:
            raise ValueError("Audio output device is unavailable")
        input_config = self._build_config(input_rate)
        output_config = self._build_config(
            output_rate,
            fade_in_seconds=self.settings.fade_in_seconds,
            fade_out_seconds=self.settings.fade_out_seconds,
        )
        input_config.validate()
        output_config.validate()
        return output_config

    @staticmethod
    def _analysis_signature(config: SemiAnalogTHDConfig) -> tuple[object, ...]:
        return (
            config.sample_rate,
            config.duration,
            config.band,
            config.segment_seconds,
            config.overlap,
            config.notch_ratio,
            config.smoothing_octaves,
            config.points,
        )

    def _update_graph(
        self,
        measurement: Measurement,
        result: SemiAnalogTHDResult,
    ) -> None:
        if measurement.graphs:
            graph = measurement.graphs[0]
            graph.name = "THD+N"
            graph.x = result.frequency
            graph.y = result.percent
            graph.x_axis = AxisSpec.FREQ
            graph.y_axis = AxisSpec.THD
        else:
            graph = GraphData(
                name="THD+N",
                x=result.frequency,
                y=result.percent,
                x_axis=AxisSpec.FREQ,
                y_axis=AxisSpec.THD,
                color=measurement.color_for_graph("THD+N"),
            )
            measurement.graphs.append(graph)
        if graph.id not in self.app.app_state.visible_graph_ids:
            self.app.app_state.visible_graph_ids.append(graph.id)
        self.app.app_state.graph_data_changed = True

    def _clear_measurement_data(self, measurement: Measurement) -> None:
        measurement.remember_graph_colors()
        state = measurement.module_state
        for key in (
            "recording",
            "generator",
            "result_frequency",
            "result_ratio",
            "integrated_ratio",
            "analysis_signature",
        ):
            state[key] = None
        self._clear_level_history(measurement)
        graph_ids = {graph.id for graph in measurement.graphs}
        measurement.graphs.clear()
        self.app.app_state.visible_graph_ids = [
            graph_id
            for graph_id in self.app.app_state.visible_graph_ids
            if graph_id not in graph_ids
        ]
        self.app.app_state.graph_data_changed = True

    def _clear_level_history(self, measurement: Measurement) -> None:
        empty = np.empty(0, dtype=np.float64)
        measurement.module_state["level_time"] = empty
        measurement.module_state["level_values"] = empty.copy()
        if self._view is not None and measurement is self._active_measurement():
            self._view.update_levels(
                empty,
                empty,
                0.0,
                duration=self._recording_duration(measurement),
            )

    def _recording_duration(self, measurement: Measurement) -> float:
        return (
            THDAcquisition.LEADING_SILENCE_SECONDS
            + self.settings.fade_in_seconds
            + float(measurement.module_state["duration"])
            + self.settings.fade_out_seconds
            + THDAcquisition.RECORDING_TAIL_SECONDS
        )

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

    def _show_error(self, message: str) -> None:
        clipping = "clipping" in message.lower()
        self._set_status("THD+N stopped: clipping" if clipping else "THD+N failed")
        self.app.main_window.show_error(
            "THD+N clipping" if clipping else "THD+N error",
            message,
        )

    @classmethod
    def _ensure_state(cls, state: dict[str, Any]) -> None:
        for key, value in cls.DEFAULT_STATE.items():
            state.setdefault(key, deepcopy(value))

    @staticmethod
    def _normalize_setting(key: str, value: Any) -> Any:
        if key == "band":
            low, high = int(value[0]), int(value[1])
            low = max(1, low)
            return low, max(low + 1, high)
        if key == "duration":
            return min(600.0, max(1.0, float(value)))
        if key == "smoothing_octaves":
            return min(3.0, max(0.01, float(value)))
        raise ValueError(f"Unknown THD setting: {key}")
