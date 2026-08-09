from copy import deepcopy
from threading import Lock
from typing import TYPE_CHECKING, Any

import numpy as np

from audioanalysis import (
    ASignal,
    AnalysisMethod,
    FrequencyBand,
    ReferenceMode,
    SmoothingWindow,
    SpectrumConfig,
)

from spectrum_app.core.model import AxisSpec, GraphData, Measurement
from spectrum_app.modules.base import BaseModule
from spectrum_app.modules.spectrum.jobs import SpectrumAcquisition, SpectrumAnalyzer
from spectrum_app.modules.spectrum.settings import (
    SpectrumSettings,
    SpectrumSettingsWindow,
)
from spectrum_app.modules.spectrum.view import SpectrumView

if TYPE_CHECKING:
    from spectrum_app.application import SpectrumApplication


class SpectrumModule(BaseModule):
    id = "spectrum"
    name = "Spectrum"
    ONLINE_INTERVAL = 0.5

    DEFAULT_STATE: dict[str, Any] = {
        "band": (20, 20_000),
        "duration": 10.0,
        "reference": "channel b",
        "weighting": "none",
        "window_width": 0.1,
        "points": 1024,
        "window": SmoothingWindow.GAUSSIAN.value,
        "recording": None,
        "generator": None,
        "level_time": np.empty(0, dtype=np.float64),
        "level_values": np.empty((0, 0), dtype=np.float64),
    }

    def __init__(self) -> None:
        super().__init__()
        self._view: SpectrumView | None = None
        self._settings: SpectrumSettings | None = None
        self._settings_window: SpectrumSettingsWindow | None = None
        self._analyzer: SpectrumAnalyzer | None = None
        self._acquisition: SpectrumAcquisition | None = None
        self._runtime_lock = Lock()
        self._pending_snapshot: tuple[Measurement, ASignal, ASignal] | None = None
        self._pending_level: tuple[Measurement, np.ndarray, np.ndarray] | None = None
        self._pending_completion: (
            tuple[
                Measurement,
                ASignal | None,
                ASignal | None,
                np.ndarray,
                np.ndarray,
                str | None,
            ]
            | None
        ) = None
        self._analysis_revision = 0
        self._current_revision = 0
        self._finishing_revision: int | None = None
        self._finish_status = ""
        self._reanalysis_requested = False
        self._stop_requested = False

    def initialize(self, app: "SpectrumApplication") -> None:
        super().initialize(app)
        self._settings = SpectrumSettings(app.settings)
        self._settings_window = SpectrumSettingsWindow(app, self._settings)
        self._settings_window.build()
        self._view = SpectrumView(self)
        self._analyzer = SpectrumAnalyzer()
        self._analyzer.start()

    def activate(self, measurement: Measurement) -> None:
        super().activate(measurement)
        self._ensure_state(measurement.module_state)
        if self._view is None:
            raise RuntimeError("Spectrum module is not initialized")
        self._view.build(
            self.app.main_window.module_gui_host,
            self.app.main_window.bottom_host,
            measurement.module_state,
        )
        if measurement.module_state["recording"] is not None and not measurement.graphs:
            self._reanalysis_requested = True

    def start_measurement(self) -> None:
        if self.app.app_state.measuring:
            return
        if self._acquisition is not None and self._acquisition.is_alive():
            self._set_status("Spectrum measurement is still stopping")
            return

        state = self.measurement.module_state
        try:
            self._validate_audio_settings(state)
            band = FrequencyBand(*state["band"])
            measurement = self.measurement
            acquisition = SpectrumAcquisition(
                self.app.audio_input,
                self.app.audio_output,
                generator_mode=self.settings.generator_mode,
                band=band,
                duration=state["duration"],
                online_interval=(
                    self.ONLINE_INTERVAL if self.settings.online_welch else None
                ),
                on_level=lambda times, levels: self._receive_level(
                    measurement,
                    times,
                    levels,
                ),
                on_snapshot=lambda recording, generator: self._receive_snapshot(
                    measurement,
                    recording,
                    generator,
                ),
                on_complete=(
                    lambda recording, generator, times, levels, error: (
                        self._receive_completion(
                            measurement,
                            recording,
                            generator,
                            times,
                            levels,
                            error,
                        )
                    )
                ),
            )
            with self._runtime_lock:
                self._pending_snapshot = None
                self._pending_level = None
                self._pending_completion = None
            self._invalidate_analysis()
            self._acquisition = acquisition
            self._stop_requested = False
            self._clear_level_history(measurement)
            self.app.app_state.measuring = True
            self._set_controls_enabled(False)
            self._set_status("Spectrum measurement started")
            acquisition.start()
        except Exception as error:
            self.app.app_state.measuring = False
            self._set_controls_enabled(True)
            self._show_error(str(error))

    def stop_measurement(self) -> None:
        acquisition = self._acquisition
        if acquisition is None or not acquisition.is_alive():
            self.app.app_state.measuring = False
            self._set_controls_enabled(True)
            return
        self._stop_requested = True
        self._set_status("Stopping Spectrum measurement...")
        acquisition.request_stop()

    def update(self) -> None:
        self._process_analysis_response()
        level, completion, snapshot = self._take_worker_updates()
        if level is not None:
            self._process_level(*level)
        if completion is not None:
            self._process_completion(*completion)
        if snapshot is not None and self._finishing_revision is None:
            measurement, recording, generator = snapshot
            if measurement is self._active_measurement():
                self._submit_analysis(
                    measurement,
                    recording,
                    generator,
                    AnalysisMethod.WELCH,
                )
        if self._reanalysis_requested and not self.app.app_state.measuring:
            self._reanalysis_requested = False
            self._reanalyze_stored_recording()

    def deactivate(self) -> None:
        self.stop_measurement()
        acquisition = self._acquisition
        if acquisition is not None and acquisition.is_alive():
            acquisition.join()
        self.app.app_state.measuring = False
        self._invalidate_analysis()
        with self._runtime_lock:
            self._pending_snapshot = None
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
        if isinstance(state.get("recording"), ASignal):
            self._reanalysis_requested = True
        return normalized

    @property
    def settings(self) -> SpectrumSettings:
        if self._settings is None:
            raise RuntimeError("Spectrum settings are not initialized")
        return self._settings

    def _receive_snapshot(
        self,
        measurement: Measurement,
        recording: ASignal,
        generator: ASignal,
    ) -> None:
        with self._runtime_lock:
            self._pending_snapshot = (measurement, recording, generator)

    def _receive_level(
        self,
        measurement: Measurement,
        times: np.ndarray,
        levels: np.ndarray,
    ) -> None:
        with self._runtime_lock:
            self._pending_level = (measurement, times, levels)

    def _receive_completion(
        self,
        measurement: Measurement,
        recording: ASignal | None,
        generator: ASignal | None,
        times: np.ndarray,
        levels: np.ndarray,
        error: str | None,
    ) -> None:
        with self._runtime_lock:
            self._pending_completion = (
                measurement,
                recording,
                generator,
                times,
                levels,
                error,
            )

    def _take_worker_updates(
        self,
    ) -> tuple[
        tuple[Measurement, np.ndarray, np.ndarray] | None,
        tuple[
            Measurement,
            ASignal | None,
            ASignal | None,
            np.ndarray,
            np.ndarray,
            str | None,
        ]
        | None,
        tuple[Measurement, ASignal, ASignal] | None,
    ]:
        with self._runtime_lock:
            completion = self._pending_completion
            snapshot = self._pending_snapshot
            level = self._pending_level
            self._pending_completion = None
            self._pending_snapshot = None
            self._pending_level = None
        return level, completion, snapshot

    def _process_level(
        self,
        measurement: Measurement,
        times: np.ndarray,
        levels: np.ndarray,
    ) -> None:
        measurement.module_state["level_time"] = times
        measurement.module_state["level_values"] = levels
        if measurement is self._active_measurement() and self._view is not None:
            self._view.update_levels(
                times,
                levels,
                duration=float(measurement.module_state["duration"]),
            )

    def _process_completion(
        self,
        measurement: Measurement,
        recording: ASignal | None,
        generator: ASignal | None,
        times: np.ndarray,
        levels: np.ndarray,
        error: str | None,
    ) -> None:
        self._acquisition = None
        self._process_level(measurement, times, levels)
        if recording is not None:
            measurement.module_state["recording"] = recording
            measurement.module_state["generator"] = generator

        if measurement is not self._active_measurement():
            self.app.app_state.measuring = False
            self._set_controls_enabled(True)
            return
        if recording is None:
            self.app.app_state.measuring = False
            self._set_controls_enabled(True)
            message = error or "Measurement stopped before audio was recorded"
            if error is not None:
                self._show_error(message)
            else:
                self._set_status(message)
            return

        if error is not None:
            self.app.app_state.measuring = False
            self._set_controls_enabled(True)
            self._show_error(error)
            return
        elif self._stop_requested:
            self._finish_status = "Spectrum measurement stopped"
        else:
            self._finish_status = "Spectrum measurement completed"
        self._submit_analysis(
            measurement,
            recording,
            generator,
            AnalysisMethod.PERIODOGRAM,
            finish_measurement=True,
        )

    def _submit_analysis(
        self,
        measurement: Measurement,
        recording: ASignal,
        generator: ASignal | None,
        method: AnalysisMethod,
        *,
        finish_measurement: bool = False,
    ) -> None:
        if self._analyzer is None:
            raise RuntimeError("Spectrum analyzer is not initialized")
        try:
            signal, config = self._analysis_input(recording, generator, method)
        except Exception as error:
            self._show_error(str(error))
            if finish_measurement:
                self.app.app_state.measuring = False
                self._set_controls_enabled(True)
            return

        self._analysis_revision += 1
        revision = self._analysis_revision
        self._current_revision = revision
        if finish_measurement:
            self._finishing_revision = revision
        self._analyzer.submit(revision, finish_measurement, signal, config)

    def _process_analysis_response(self) -> None:
        if self._analyzer is None:
            return
        response = self._analyzer.poll()
        if response is None:
            return
        revision, finishing, result, error = response
        if revision != self._current_revision:
            return
        if error is not None or result is None:
            self._show_error(error or "Calculation failed")
        else:
            measurement = self._active_measurement()
            if measurement is not None:
                self._update_graph(measurement, result.frequency, result.values)

        if finishing and revision == self._finishing_revision:
            self._finishing_revision = None
            self.app.app_state.measuring = False
            self._set_controls_enabled(True)
            if error is None:
                self._set_status(self._finish_status)

    def _reanalyze_stored_recording(self) -> None:
        state = self.measurement.module_state
        recording = state.get("recording")
        generator = state.get("generator")
        if not isinstance(recording, ASignal):
            return
        self._submit_analysis(
            self.measurement,
            recording,
            generator if isinstance(generator, ASignal) else None,
            AnalysisMethod.PERIODOGRAM,
        )

    def _analysis_input(
        self,
        recording: ASignal,
        generator: ASignal | None,
        method: AnalysisMethod,
    ) -> tuple[ASignal, SpectrumConfig]:
        state = self.measurement.module_state
        reference = state["reference"]
        signal = recording
        reference_mode = ReferenceMode.NONE
        if reference == "channel b":
            if recording.channel_count < 2:
                raise ValueError("Channel B reference requires two input channels")
            reference_mode = ReferenceMode.CHANNEL_B
        elif reference == "generator":
            if generator is None:
                raise ValueError("Generator reference is not available")
            if generator.sample_rate != recording.sample_rate:
                raise ValueError(
                    "Generator reference requires equal input and output sample rates"
                )
            length = min(recording.sample_count, generator.sample_count)
            signal = ASignal(
                (
                    recording[0].trim(length),
                    generator[0].trim(length),
                )
            )
            reference_mode = ReferenceMode.CHANNEL_B
        elif reference != "none":
            raise ValueError(f"Unknown reference: {reference}")

        config = SpectrumConfig(
            method=method,
            reference=reference_mode,
            band=FrequencyBand(*state["band"]),
            points=state["points"],
            window=SmoothingWindow(state["window"]),
            window_width=state["window_width"],
            welch_samples=self.settings.welch_samples,
            pink_weighting=state["weighting"] == "pink",
        )
        return signal, config

    def _update_graph(self, measurement: Measurement, x, y) -> None:
        if measurement.graphs:
            graph = measurement.graphs[0]
            graph.name = "Spectrum"
            graph.x = x
            graph.y = y
            graph.x_axis = AxisSpec.FREQ
            graph.y_axis = AxisSpec.LEVEL
        else:
            graph = GraphData(
                name="Spectrum",
                x=x,
                y=y,
                x_axis=AxisSpec.FREQ,
                y_axis=AxisSpec.LEVEL,
            )
            measurement.graphs.append(graph)
        if graph.id not in self.app.app_state.visible_graph_ids:
            self.app.app_state.visible_graph_ids.append(graph.id)
        self.app.app_state.graph_data_changed = True

    def _validate_audio_settings(self, state: dict[str, Any]) -> None:
        input_rate = self.app.audio_input.sample_rate
        output_rate = self.app.audio_output.sample_rate
        if input_rate <= 0:
            raise ValueError("Audio input device is unavailable")
        if output_rate <= 0:
            raise ValueError("Audio output device is unavailable")
        if state["reference"] == "generator" and input_rate != output_rate:
            raise ValueError(
                "Generator reference requires equal input and output sample rates"
            )
        band = FrequencyBand(*state["band"])
        band.validate(nyquist=min(input_rate, output_rate) / 2)

    def _invalidate_analysis(self) -> None:
        self._analysis_revision += 1
        self._current_revision = self._analysis_revision
        self._finishing_revision = None

    def _active_measurement(self) -> Measurement | None:
        try:
            return self.measurement
        except RuntimeError:
            return None

    def _set_controls_enabled(self, enabled: bool) -> None:
        if self._view is not None:
            self._view.set_enabled(enabled)

    def _clear_level_history(self, measurement: Measurement) -> None:
        times = np.empty(0, dtype=np.float64)
        levels = np.empty((0, 0), dtype=np.float64)
        self._process_level(measurement, times, levels)

    def _set_status(self, text: str) -> None:
        self.app.main_window.set_status_text(text)

    def _show_error(self, message: str) -> None:
        clipping = "clipping" in message.lower()
        self._set_status("Spectrum stopped: clipping" if clipping else "Spectrum failed")
        self.app.main_window.show_error(
            "Spectrum clipping" if clipping else "Spectrum error",
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
            high = max(low + 1, high)
            return low, high
        if key == "duration":
            return min(100.0, max(1.0, float(value)))
        if key == "reference":
            if value not in ("none", "channel b", "generator"):
                raise ValueError(f"Unknown reference: {value}")
            return value
        if key == "weighting":
            if value not in ("none", "pink"):
                raise ValueError(f"Unknown weighting: {value}")
            return value
        if key == "window_width":
            return min(3.0, max(0.1, float(value)))
        if key == "points":
            return min(3000, max(100, int(value)))
        if key == "window":
            return SmoothingWindow(value).value
        raise ValueError(f"Unknown Spectrum setting: {key}")
