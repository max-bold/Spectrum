from copy import deepcopy
from threading import Lock
from time import monotonic
from typing import TYPE_CHECKING, Any

import numpy as np

from audioanalysis import (
    ASignal,
    AnalysisMethod,
    FrequencyBand,
    ReferenceMode,
    SmoothingWindow,
    SpectrumConfig,
    extend_log_sweep_band,
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

    DEFAULT_STATE: dict[str, Any] = {
        "band": (20, 20_000),
        "duration": 10.0,
        "reference": "generator",
        "weighting": "none",
        "window_width": 0.1,
        "points": 1024,
        "window": SmoothingWindow.GAUSSIAN.value,
        "recording": None,
        "generator": None,
        "recordings": [],
        "generators": [],
        "multiple": False,
        "count": 3,
        "auto": False,
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
        self._cycle_measurement: Measurement | None = None
        self._completed_recordings: list[ASignal] = []
        self._completed_generators: list[ASignal | None] = []
        self._target_count = 1
        self._automatic_repeat = False
        self._waiting_for_repeat = False
        self._next_take_at: float | None = None

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
        if self._stored_takes(measurement.module_state)[0] and not measurement.graphs:
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
            measurement = self.measurement
            with self._runtime_lock:
                self._pending_level = None
                self._pending_completion = None
            self._invalidate_analysis()
            self._stop_requested = False
            self._cycle_measurement = measurement
            self._completed_recordings = []
            self._completed_generators = []
            self._target_count = int(state["count"]) if state["multiple"] else 1
            self._automatic_repeat = bool(state["auto"])
            self._waiting_for_repeat = False
            self._next_take_at = None
            state["recording"] = None
            state["generator"] = None
            state["recordings"] = []
            state["generators"] = []
            self._clear_level_history(measurement)
            self.app.app_state.measuring = True
            self._set_controls_enabled(False)
            self._start_next_acquisition()
        except Exception as error:
            self._abort_cycle(str(error))

    def stop_measurement(self) -> None:
        if not self.app.app_state.measuring:
            self._set_controls_enabled(True)
            return
        self._stop_requested = True
        self._waiting_for_repeat = False
        self._next_take_at = None
        if self._view is not None:
            self._view.hide_repeat_dialog()
        acquisition = self._acquisition
        if acquisition is None or not acquisition.is_alive():
            self._finish_multiple_cycle("Spectrum measurement stopped")
            return
        self._set_status("Stopping Spectrum measurement...")
        acquisition.request_stop()

    def continue_multiple_measurement(
        self,
        sender=None,
        app_data=None,
        user_data=None,
    ) -> None:
        if not self.app.app_state.measuring or not self._waiting_for_repeat:
            return
        if self._view is not None:
            self._view.hide_repeat_dialog()
        self._start_next_acquisition()

    def break_multiple_measurement(
        self,
        sender=None,
        app_data=None,
        user_data=None,
    ) -> None:
        if not self.app.app_state.measuring or not self._waiting_for_repeat:
            return
        self._finish_multiple_cycle("Spectrum multiple measurement stopped")

    def update(self) -> None:
        self._process_analysis_response()
        level, completion = self._take_worker_updates()
        if level is not None:
            self._process_level(*level)
        if completion is not None:
            self._process_completion(*completion)
        if (
            self._waiting_for_repeat
            and self._next_take_at is not None
            and monotonic() >= self._next_take_at
        ):
            self._start_next_acquisition()
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
            self._pending_level = None
            self._pending_completion = None
        self._cycle_measurement = None
        self._completed_recordings = []
        self._completed_generators = []
        self._waiting_for_repeat = False
        self._next_take_at = None
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
        self._cycle_measurement = None
        self._completed_recordings = []
        self._completed_generators = []
        self._waiting_for_repeat = False
        self._next_take_at = None
        self.app.app_state.measuring = False
        super().shutdown()

    def set_setting(self, key: str, value: Any) -> Any:
        state = self.measurement.module_state
        normalized = self._normalize_setting(key, value)
        if state.get(key) == normalized:
            return normalized
        state[key] = normalized
        if self._stored_takes(state)[0]:
            self._reanalysis_requested = True
        return normalized

    @property
    def settings(self) -> SpectrumSettings:
        if self._settings is None:
            raise RuntimeError("Spectrum settings are not initialized")
        return self._settings

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
    ]:
        with self._runtime_lock:
            completion = self._pending_completion
            level = self._pending_level
            self._pending_completion = None
            self._pending_level = None
        return level, completion

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
                duration=self._total_duration(measurement.module_state),
            )

    def _start_next_acquisition(self) -> None:
        measurement = self._cycle_measurement
        if measurement is None or self._stop_requested:
            return
        state = measurement.module_state
        take = len(self._completed_recordings) + 1
        try:
            online_samples = (
                self.settings.welch_samples if self.settings.online_welch else None
            )
            online_revision = 0
            online_config: SpectrumConfig | None = None
            generator_reference = False
            running_mean = self.settings.generator_mode == "pink noise"
            if online_samples is not None:
                reference = str(state["reference"])
                reference_mode = (
                    ReferenceMode.CHANNEL_B
                    if reference in ("channel b", "generator")
                    else ReferenceMode.NONE
                )
                generator_reference = reference == "generator"
                online_config = self._spectrum_config(
                    state,
                    AnalysisMethod.WELCH,
                    reference_mode,
                )
                self._analysis_revision += 1
                online_revision = self._analysis_revision
                self._current_revision = online_revision

            def feed_online_audio(
                recording: ASignal,
                generator: ASignal,
            ) -> None:
                analyzer = self._analyzer
                if (
                    analyzer is None
                    or online_config is None
                    or online_samples is None
                    or measurement is not self._cycle_measurement
                    or self._stop_requested
                ):
                    return
                analyzer.feed_online(
                    online_revision,
                    recording,
                    online_samples,
                    generator if generator_reference else None,
                    online_config,
                    running_mean=running_mean,
                )

            acquisition = SpectrumAcquisition(
                self.app.audio_input,
                self.app.audio_output,
                generator_mode=self.settings.generator_mode,
                band=FrequencyBand(*state["band"]),
                duration=state["duration"],
                pre_silence=self.settings.pre_silence,
                post_silence=self.settings.post_silence,
                fade_in=self.settings.fade_in,
                fade_out=self.settings.fade_out,
                online_samples=online_samples,
                on_level=lambda times, levels: self._receive_level(
                    measurement,
                    times,
                    levels,
                ),
                on_snapshot=feed_online_audio,
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
                self._pending_level = None
                self._pending_completion = None
            self._waiting_for_repeat = False
            self._next_take_at = None
            self._clear_level_history(measurement)
            self._acquisition = acquisition
            self._set_status(f"Spectrum measurement {take} of {self._target_count}")
            acquisition.start()
        except Exception as error:
            self._abort_cycle(str(error))

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

        if measurement is not self._active_measurement() or measurement is not self._cycle_measurement:
            self._abort_cycle("The active Spectrum measurement changed")
            return
        if error is not None:
            self._abort_cycle(error)
            return
        if self._stop_requested:
            self._finish_multiple_cycle("Spectrum measurement stopped")
            return
        if recording is None:
            self._abort_cycle("Measurement stopped before audio was recorded")
            return

        self._completed_recordings.append(recording)
        self._completed_generators.append(generator)
        self._store_completed_takes(measurement)
        if len(self._completed_recordings) >= self._target_count:
            self._finish_multiple_cycle("Spectrum measurement completed")
            return

        self._submit_analysis(
            measurement,
            self._completed_recordings,
            self._completed_generators,
            AnalysisMethod.PERIODOGRAM,
        )
        self._waiting_for_repeat = True
        if self._automatic_repeat:
            self._next_take_at = monotonic() + self.settings.measurement_pause
            self._set_status(
                "Spectrum measurement "
                f"{len(self._completed_recordings)} of {self._target_count} completed; "
                f"next in {self.settings.measurement_pause:.1f} s"
            )
        elif self._view is not None:
            self._view.show_repeat_dialog(
                len(self._completed_recordings),
                self._target_count,
            )
            self._set_status("Move the microphone, then continue or break")

    def _submit_analysis(
        self,
        measurement: Measurement,
        recordings: ASignal | list[ASignal],
        generators: ASignal | None | list[ASignal | None],
        method: AnalysisMethod,
        *,
        finish_measurement: bool = False,
    ) -> None:
        if self._analyzer is None:
            raise RuntimeError("Spectrum analyzer is not initialized")
        try:
            recording_items = (
                recordings if isinstance(recordings, list) else [recordings]
            )
            generator_items = (
                generators if isinstance(generators, list) else [generators]
            )
            if len(generator_items) != len(recording_items):
                raise ValueError("Spectrum recordings and generators do not match")
            prepared = [
                self._analysis_input(recording, generator, method)
                for recording, generator in zip(
                    recording_items,
                    generator_items,
                    strict=True,
                )
            ]
            signals = tuple(item[0] for item in prepared)
            config = prepared[0][1]
        except Exception as error:
            self._show_error(str(error))
            if finish_measurement:
                self._complete_cycle_ui()
            return

        self._analysis_revision += 1
        revision = self._analysis_revision
        self._current_revision = revision
        if finish_measurement:
            self._finishing_revision = revision
        self._analyzer.submit(revision, finish_measurement, signals, config)

    def _process_analysis_response(self) -> None:
        if self._analyzer is None:
            return
        response = self._analyzer.poll()
        if response is None:
            return
        revision, finishing, result, error = response
        if revision != self._current_revision:
            return
        if error is not None:
            self._show_error(error)
        elif result is not None:
            measurement = self._active_measurement()
            if measurement is not None:
                self._update_graph(measurement, result.frequency, result.values)

        if finishing and revision == self._finishing_revision:
            self._finishing_revision = None
            self._complete_cycle_ui()
            if error is None:
                self._set_status(self._finish_status)

    def _reanalyze_stored_recording(self) -> None:
        state = self.measurement.module_state
        recordings, generators = self._stored_takes(state)
        if not recordings:
            return
        self._submit_analysis(
            self.measurement,
            recordings,
            generators,
            AnalysisMethod.PERIODOGRAM,
        )

    def _finish_multiple_cycle(self, status: str) -> None:
        measurement = self._cycle_measurement
        self._waiting_for_repeat = False
        self._next_take_at = None
        if self._view is not None:
            self._view.hide_repeat_dialog()
        if measurement is None or not self._completed_recordings:
            self._complete_cycle_ui()
            self._set_status("Spectrum measurement stopped without completed takes")
            return
        self._store_completed_takes(measurement)
        self._finish_status = status
        self._submit_analysis(
            measurement,
            self._completed_recordings,
            self._completed_generators,
            AnalysisMethod.PERIODOGRAM,
            finish_measurement=True,
        )

    def _abort_cycle(self, message: str) -> None:
        acquisition = self._acquisition
        if acquisition is not None and acquisition.is_alive():
            acquisition.request_stop()
        self._store_completed_takes(self._cycle_measurement)
        self._complete_cycle_ui()
        self._show_error(message)

    def _complete_cycle_ui(self) -> None:
        self.app.app_state.measuring = False
        self._set_controls_enabled(True)
        if self._view is not None:
            self._view.hide_repeat_dialog()
        self._acquisition = None
        self._cycle_measurement = None
        self._completed_recordings = []
        self._completed_generators = []
        self._waiting_for_repeat = False
        self._next_take_at = None
        self._stop_requested = False

    def _store_completed_takes(self, measurement: Measurement | None) -> None:
        if measurement is None:
            return
        state = measurement.module_state
        if self._target_count > 1:
            state["recording"] = None
            state["generator"] = None
            state["recordings"] = list(self._completed_recordings)
            state["generators"] = list(self._completed_generators)
        elif self._completed_recordings:
            state["recording"] = self._completed_recordings[0]
            state["generator"] = self._completed_generators[0]
            state["recordings"] = []
            state["generators"] = []

    @staticmethod
    def _stored_takes(
        state: dict[str, Any],
    ) -> tuple[list[ASignal], list[ASignal | None]]:
        recordings_value = state.get("recordings")
        generators_value = state.get("generators")
        if (
            isinstance(recordings_value, list)
            and recordings_value
            and all(isinstance(item, ASignal) for item in recordings_value)
        ):
            recordings: list[ASignal] = list(recordings_value)
            generators: list[ASignal | None]
            if (
                isinstance(generators_value, list)
                and len(generators_value) == len(recordings)
                and all(
                    item is None or isinstance(item, ASignal)
                    for item in generators_value
                )
            ):
                generators = list(generators_value)
            else:
                generators = [None for _ in recordings]
            return recordings, generators

        recording = state.get("recording")
        if not isinstance(recording, ASignal):
            return [], []
        generator = state.get("generator")
        return [recording], [generator if isinstance(generator, ASignal) else None]

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

        return signal, self._spectrum_config(state, method, reference_mode)

    def _spectrum_config(
        self,
        state: dict[str, Any],
        method: AnalysisMethod,
        reference_mode: ReferenceMode,
    ) -> SpectrumConfig:
        reference = state["reference"]
        return SpectrumConfig(
            method=method,
            reference=reference_mode,
            band=FrequencyBand(*state["band"]),
            points=state["points"],
            window=SmoothingWindow(state["window"]),
            window_width=state["window_width"],
            welch_samples=self.settings.welch_samples,
            pink_weighting=(reference == "none" and state["weighting"] == "pink"),
        )

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
                color=measurement.color_for_graph("Spectrum"),
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
        if self.settings.generator_mode == "log chirp":
            extend_log_sweep_band(
                band,
                float(state["duration"]),
                self.settings.fade_in,
                self.settings.fade_out,
            ).validate(nyquist=output_rate / 2)

    def _total_duration(self, state: dict[str, Any]) -> float:
        return (
            self.settings.pre_silence
            + self.settings.fade_in
            + float(state["duration"])
            + self.settings.fade_out
            + self.settings.post_silence
        )

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
        if key in ("multiple", "auto"):
            return bool(value)
        if key == "count":
            return min(100, max(2, int(value)))
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
