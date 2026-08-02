from __future__ import annotations

from copy import deepcopy
from enum import Enum
from threading import Lock, Thread
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from audioanalysis import (
    ASignal,
    FrequencyBand,
    ImpedanceConfig,
    ImpedanceResult,
    ReferenceCalibration,
    SmoothingWindow,
    SpiceTableValues,
    analyze_recording_levels,
    calculate_channel_correction,
    calculate_impedance,
    channel_calibration_config,
    fit_impedance_auto,
    format_spice_table,
    generate_channel_calibration_signal,
    generate_level_test_signal,
    generate_measurement_signal,
    require_valid_reference_calibration,
    trim_recording,
)
from audioanalysis.impedance import ChannelCalibration, estimate_reference_resistor
from audioanalysis.impedance_model import FitResult

from spectrum_app.core.model import AxisSpec, GraphData, Measurement
from spectrum_app.modules.base import BaseModule
from spectrum_app.modules.impedance.jobs import ImpedanceCapture
from spectrum_app.modules.impedance.view import ImpedanceView

if TYPE_CHECKING:
    from spectrum_app.application import SpectrumApplication


class Operation(str, Enum):
    CHANNEL_CALIBRATION = "channel_calibration"
    REFERENCE_CALIBRATION = "reference_calibration"
    MEASUREMENT = "measurement"
    REPROCESS = "reprocess"
    TEST = "test"
    SPICE = "spice"


CalculationResult = (
    ChannelCalibration
    | ReferenceCalibration
    | ImpedanceResult
    | tuple[FitResult, SpiceTableValues]
    | tuple[ChannelCalibration, ReferenceCalibration, ImpedanceResult]
)


class ImpedanceModule(BaseModule):
    id = "impedance"
    name = "Impedance"
    RECORDING_TAIL = 0.25

    DEFAULT_STATE: dict[str, Any] = {
        "workflow": "uncalibrated",
        "status": "Calibration required",
        "band": (20, 20_000),
        "duration": 20.0,
        "reference_resistor": 3.25,
        "calibration_resistor": 10.4,
        "window": SmoothingWindow.GAUSSIAN.value,
        "window_width": 0.1,
        "points": 1024,
        "calibration_signature": None,
        "channel_calibration_recording": None,
        "channel_calibration_signal": None,
        "reference_calibration_recording": None,
        "reference_calibration_signal": None,
        "measurement_recording": None,
        "measurement_signal": None,
        "channel_correction": None,
        "reference_resistor_estimated": None,
        "reference_diagnostics": None,
        "frequency": None,
        "impedance": None,
        "fit_result": None,
        "spice_values": None,
    }
    CAPTURE_SETTINGS = {
        "band",
        "duration",
        "reference_resistor",
        "calibration_resistor",
    }
    FILTER_SETTINGS = {"window", "window_width", "points"}

    def __init__(self) -> None:
        super().__init__()
        self._view: ImpedanceView | None = None
        self._capture: ImpedanceCapture | None = None
        self._calculation: Thread | None = None
        self._lock = Lock()
        self._pending_level: tuple[float, float] | None = None
        self._pending_capture: (
            tuple[int, Operation, ASignal, ASignal | None, str | None, bool]
            | None
        ) = None
        self._pending_calculation: (
            tuple[int, Operation, CalculationResult | None, str | None] | None
        ) = None
        self._operation: Operation | None = None
        self._operation_fallback = "uncalibrated"
        self._revision = 0
        self._levels = (0.0, 0.0)
        self._shown_levels: tuple[float, float] | None = None
        self._shown_status = ""
        self._calibration_stage: int | None = None

    def initialize(self, app: "SpectrumApplication") -> None:
        super().initialize(app)
        self._view = ImpedanceView(self)

    def activate(self, measurement: Measurement) -> None:
        super().activate(measurement)
        self._ensure_state(measurement.module_state)
        self._repair_transient_state(measurement.module_state)
        if self._view is None:
            raise RuntimeError("Impedance module is not initialized")
        self._view.build(
            self.app.main_window.module_gui_host,
            self.app.main_window.bottom_host,
            measurement.module_state,
        )
        self._sync_view(force=True)
        stored_result = self._stored_result()
        if stored_result is not None and not measurement.graphs:
            self._update_graphs(measurement, stored_result)

    @property
    def measurement_button_label(self) -> str:
        return (
            "MEASURE"
            if self.measurement.module_state["workflow"] in ("calibrated", "completed")
            else "Calibrate"
        )

    def start_measurement(self) -> None:
        if self._is_busy():
            return
        state = self.measurement.module_state
        if state["workflow"] not in ("calibrated", "completed"):
            self.show_calibration()
            return
        try:
            config = self._build_config()
            if state["calibration_signature"] != self._capture_signature(config):
                raise ValueError("Measurement settings changed; recalibrate first")
            signal = generate_measurement_signal(config)
            self._start_capture(Operation.MEASUREMENT, signal, config)
        except Exception as error:
            self._fail(str(error), fallback="calibrated")

    def stop_measurement(self) -> None:
        capture = self._capture
        if capture is not None and capture.is_alive():
            self._set_status("Stopping operation...")
            capture.stop()
            return
        if self._operation is not None and self._operation != Operation.SPICE:
            self._revision += 1
            fallback = self._fallback_workflow(self._operation)
            self.measurement.module_state["workflow"] = fallback
            self._finish_operation("Operation stopped")

    def update(self) -> None:
        level, capture, calculation = self._take_pending()
        if level is not None:
            self._levels = level
        if capture is not None:
            self._process_capture(*capture)
        if calculation is not None:
            self._process_calculation(*calculation)
        self._sync_view()

    def deactivate(self) -> None:
        self._revision += 1
        capture = self._capture
        if capture is not None and capture.is_alive():
            capture.stop()
            capture.join(timeout=2.0)
        self.app.audio_input.close()
        self.app.audio_output.close()
        self.app.app_state.measuring = False
        self._capture = None
        self._calculation = None
        self._operation = None
        self._calibration_stage = None
        with self._lock:
            self._pending_level = None
            self._pending_capture = None
            self._pending_calculation = None
        if self._view is not None:
            self._view.destroy()
        super().deactivate()

    def shutdown(self) -> None:
        capture = self._capture
        if capture is not None and capture.is_alive():
            capture.stop()
            capture.join(timeout=2.0)
        self._view = None
        self.app.app_state.measuring = False
        super().shutdown()

    def show_calibration(self, sender=None, app_data=None, user_data=None) -> None:
        if self._is_busy() or self._view is None:
            return
        stage = 2 if self.measurement.module_state["workflow"] == "waiting_reference" else 1
        self._show_calibration_stage(stage)

    def request_calibration(self, sender=None, app_data=None, user_data=None) -> None:
        """Open a fresh calibration workflow from the Tools menu."""
        if self._is_busy() or self._view is None:
            return
        self._show_calibration_stage(1)

    def continue_calibration(self, sender=None, app_data=None, user_data=None) -> None:
        if self._is_busy():
            return
        state = self.measurement.module_state
        try:
            config = self._build_config()
            if self._calibration_stage == 2:
                if state["calibration_signature"] != self._capture_signature(config):
                    self._clear_calibration(state)
                    raise ValueError("Measurement settings changed; restart calibration")
                signal = generate_measurement_signal(config)
                operation = Operation.REFERENCE_CALIBRATION
            else:
                self._clear_calibration(state)
                self._clear_graphs(self.measurement)
                signal = generate_channel_calibration_signal(config)
                operation = Operation.CHANNEL_CALIBRATION
            if self._view is not None:
                self._view.hide_calibration()
            self._calibration_stage = None
            self._start_capture(operation, signal, config)
        except Exception as error:
            self._fail(str(error), fallback="uncalibrated")

    def cancel_calibration(self, sender=None, app_data=None, user_data=None) -> None:
        self._calibration_stage = None
        if self._view is not None:
            self._view.hide_calibration()
        if self._operation in (
            Operation.CHANNEL_CALIBRATION,
            Operation.REFERENCE_CALIBRATION,
        ):
            self.stop_measurement()
        elif self.measurement.module_state["workflow"] == "waiting_reference":
            self._clear_calibration(self.measurement.module_state)
            self._set_status("Calibration cancelled")

    def toggle_test_signal(self, sender=None, app_data=None, user_data=None) -> None:
        if self._operation == Operation.TEST:
            self.stop_measurement()
            return
        if self._is_busy():
            return
        try:
            config = self._build_config()
            self._start_capture(
                Operation.TEST,
                generate_level_test_signal(config),
                config,
                loop=True,
            )
        except Exception as error:
            self._fail(str(error), fallback=self.measurement.module_state["workflow"])

    def request_spice_fit(self, sender=None, app_data=None, user_data=None) -> None:
        if self._view is None:
            return
        result = self._stored_result()
        if result is None:
            self._view.show_spice("Complete an impedance measurement first", None)
            return
        if self._is_busy():
            self._view.show_spice("Another operation is active", None)
            return
        stored_values = self.measurement.module_state.get("spice_values")
        if isinstance(stored_values, SpiceTableValues):
            self._view.show_spice("Stored SPICE Fit — needs testing", stored_values)
            return
        self._view.show_spice("Calculating SPICE model... (needs testing)", None)

        def calculate() -> tuple[FitResult, SpiceTableValues]:
            # needs testing: this fit is slow and may produce implausible models.
            fit, _ = fit_impedance_auto(
                result.frequency,
                result.magnitude,
                min_sections=0,
                max_sections=10,
                max_evaluations=2000,
            )
            return fit, format_spice_table(fit)

        self._start_calculation(Operation.SPICE, calculate, lock_application=False)

    def set_setting(self, key: str, value: Any) -> Any:
        state = self.measurement.module_state
        normalized = self._normalize_setting(key, value)
        if state.get(key) == normalized:
            return normalized
        state[key] = normalized
        if key in self.CAPTURE_SETTINGS:
            self._clear_calibration(state)
            self._clear_graphs(self.measurement)
            self._set_status("Measurement settings changed; calibration required")
        elif key in self.FILTER_SETTINGS and state["workflow"] == "completed":
            self._request_reprocess()
        return normalized

    def _start_capture(
        self,
        operation: Operation,
        signal: ASignal,
        config: ImpedanceConfig,
        *,
        loop: bool = False,
    ) -> None:
        self._validate_audio(config)
        self._revision += 1
        revision = self._revision
        capture = ImpedanceCapture(
            self.app.audio_input,
            self.app.audio_output,
            signal,
            recording_tail=self.RECORDING_TAIL,
            loop=loop,
            on_level=lambda levels: self._receive_level(revision, levels),
            on_complete=lambda recording, error, cancelled: self._receive_capture(
                revision, operation, signal, recording, error, cancelled
            ),
        )
        self._capture = capture
        self._operation_fallback = str(self.measurement.module_state["workflow"])
        self._operation = operation
        if operation == Operation.CHANNEL_CALIBRATION:
            self.measurement.module_state["workflow"] = "calibrating_channels"
            status = "Calibrating input channels..."
        elif operation == Operation.REFERENCE_CALIBRATION:
            self.measurement.module_state["workflow"] = "calibrating_reference"
            status = "Calibrating reference resistor..."
        elif operation == Operation.MEASUREMENT:
            self.measurement.module_state["workflow"] = "measuring"
            status = "Measuring impedance..."
        else:
            status = "Test signal running"
        self.app.app_state.measuring = True
        self._set_controls_enabled(False)
        self._set_status(status)
        capture.start()

    def _process_capture(
        self,
        revision: int,
        operation: Operation,
        signal: ASignal,
        recording: ASignal | None,
        error: str | None,
        cancelled: bool,
    ) -> None:
        if revision != self._revision:
            return
        self._capture = None
        self._levels = (0.0, 0.0)
        if error is not None:
            self._fail(error, fallback=self._fallback_workflow(operation))
            return
        if cancelled:
            self.measurement.module_state["workflow"] = self._fallback_workflow(
                operation
            )
            self._finish_operation(
                "Test signal stopped" if operation == Operation.TEST else "Operation stopped"
            )
            return
        if recording is None:
            self._fail("Audio recording is empty", fallback=self._fallback_workflow(operation))
            return
        try:
            recording = trim_recording(recording, signal.sample_count)
            analyze_recording_levels(recording, raise_on_clipping=True)
            self._store_capture(operation, recording, signal)
            config = self._build_config(sample_rate=recording.sample_rate)
            if operation == Operation.CHANNEL_CALIBRATION:
                self.measurement.module_state[
                    "calibration_signature"
                ] = self._capture_signature(config)
            self._start_operation_calculation(operation, recording, config)
        except Exception as error_value:
            self._fail(str(error_value), fallback=self._fallback_workflow(operation))

    def _start_operation_calculation(
        self,
        operation: Operation,
        recording: ASignal,
        config: ImpedanceConfig,
    ) -> None:
        state = self.measurement.module_state
        if operation == Operation.CHANNEL_CALIBRATION:
            calculate: Callable[[], CalculationResult] = lambda: calculate_channel_correction(
                recording, channel_calibration_config(config)
            )
        elif operation == Operation.REFERENCE_CALIBRATION:
            correction = np.asarray(state["channel_correction"], dtype=np.complex128)

            def calculate() -> CalculationResult:
                result = estimate_reference_resistor(recording, config, correction)
                require_valid_reference_calibration(result.diagnostics)
                return result
        elif operation == Operation.MEASUREMENT:
            correction = np.asarray(state["channel_correction"], dtype=np.complex128)
            reference = float(state["reference_resistor_estimated"])
            calculate = lambda: calculate_impedance(
                recording, config, correction, reference
            )
        else:
            raise RuntimeError(f"Unsupported capture operation: {operation}")
        self._start_calculation(operation, calculate)

    def _start_calculation(
        self,
        operation: Operation,
        calculate: Callable[[], CalculationResult],
        *,
        lock_application: bool = True,
    ) -> None:
        self._revision += 1
        revision = self._revision
        self._operation = operation
        if lock_application:
            self.app.app_state.measuring = True
            self._set_controls_enabled(False)

        def worker() -> None:
            try:
                result = calculate()
                error = None
            except Exception as exception:
                result = None
                error = str(exception) or exception.__class__.__name__
            with self._lock:
                self._pending_calculation = (revision, operation, result, error)

        thread = Thread(
            target=worker,
            name=f"impedance-{operation.value}",
            daemon=True,
        )
        self._calculation = thread
        thread.start()

    def _process_calculation(
        self,
        revision: int,
        operation: Operation,
        result: CalculationResult | None,
        error: str | None,
    ) -> None:
        if revision != self._revision:
            return
        self._calculation = None
        state = self.measurement.module_state
        if error is not None or result is None:
            if operation == Operation.SPICE:
                self._operation = None
                if self._view is not None:
                    self._view.show_spice(f"SPICE Fit failed: {error}", None)
                return
            self._fail(error or "Calculation failed", self._fallback_workflow(operation))
            return

        if operation == Operation.CHANNEL_CALIBRATION:
            assert isinstance(result, ChannelCalibration)
            state["channel_correction"] = result.correction
            state["workflow"] = "waiting_reference"
            self._finish_operation("Connect Rref and Rcal for calibration stage 2")
            self._show_calibration_stage(2)
        elif operation == Operation.REFERENCE_CALIBRATION:
            assert isinstance(result, ReferenceCalibration)
            state["reference_resistor_estimated"] = result.reference_resistor
            state["reference_diagnostics"] = result.diagnostics
            state["frequency"] = result.frequency
            state["impedance"] = result.impedance
            state["workflow"] = "calibrated"
            self._update_graphs(
                self.measurement,
                ImpedanceResult(result.frequency, result.impedance),
            )
            self._finish_operation(
                f"Calibrated, Rref = {result.reference_resistor:.4g} Ohm"
            )
        elif operation == Operation.MEASUREMENT:
            assert isinstance(result, ImpedanceResult)
            state["frequency"] = result.frequency
            state["impedance"] = result.impedance
            state["workflow"] = "completed"
            state["fit_result"] = None
            state["spice_values"] = None
            self._update_graphs(self.measurement, result)
            self._finish_operation("Measurement completed")
        elif operation == Operation.REPROCESS:
            assert isinstance(result, tuple) and len(result) == 3
            channel, reference, impedance_result = result
            assert isinstance(channel, ChannelCalibration)
            assert isinstance(reference, ReferenceCalibration)
            assert isinstance(impedance_result, ImpedanceResult)
            state["channel_correction"] = channel.correction
            state["reference_resistor_estimated"] = reference.reference_resistor
            state["reference_diagnostics"] = reference.diagnostics
            state["frequency"] = impedance_result.frequency
            state["impedance"] = impedance_result.impedance
            state["workflow"] = "completed"
            state["fit_result"] = None
            state["spice_values"] = None
            self._update_graphs(self.measurement, impedance_result)
            self._finish_operation("Measurement reprocessed")
        elif operation == Operation.SPICE:
            assert isinstance(result, tuple) and len(result) == 2
            fit, values = result
            assert isinstance(fit, FitResult)
            assert isinstance(values, SpiceTableValues)
            state["fit_result"] = fit
            state["spice_values"] = values
            self._operation = None
            if self._view is not None:
                self._view.show_spice("SPICE Fit completed", values)

    def _request_reprocess(self) -> None:
        if self._is_busy():
            return
        state = self.measurement.module_state
        recordings = (
            state.get("channel_calibration_recording"),
            state.get("reference_calibration_recording"),
            state.get("measurement_recording"),
        )
        if not all(isinstance(item, ASignal) for item in recordings):
            return
        channel_recording, reference_recording, measurement_recording = recordings
        assert isinstance(channel_recording, ASignal)
        assert isinstance(reference_recording, ASignal)
        assert isinstance(measurement_recording, ASignal)
        config = self._build_config(sample_rate=measurement_recording.sample_rate)

        def calculate() -> CalculationResult:
            channel = calculate_channel_correction(
                channel_recording,
                channel_calibration_config(config),
            )
            reference = estimate_reference_resistor(
                reference_recording,
                config,
                channel.correction,
            )
            require_valid_reference_calibration(reference.diagnostics)
            impedance_result = calculate_impedance(
                measurement_recording,
                config,
                channel.correction,
                reference.reference_resistor,
            )
            return channel, reference, impedance_result

        self._set_status("Reprocessing measurement...")
        self._start_calculation(Operation.REPROCESS, calculate)

    def _store_capture(
        self,
        operation: Operation,
        recording: ASignal,
        signal: ASignal,
    ) -> None:
        state = self.measurement.module_state
        prefix = {
            Operation.CHANNEL_CALIBRATION: "channel_calibration",
            Operation.REFERENCE_CALIBRATION: "reference_calibration",
            Operation.MEASUREMENT: "measurement",
        }[operation]
        state[f"{prefix}_recording"] = recording
        state[f"{prefix}_signal"] = signal

    def _receive_level(
        self,
        revision: int,
        levels: tuple[float, float],
    ) -> None:
        with self._lock:
            if revision == self._revision:
                self._pending_level = levels

    def _receive_capture(
        self,
        revision: int,
        operation: Operation,
        signal: ASignal,
        recording: ASignal | None,
        error: str | None,
        cancelled: bool,
    ) -> None:
        with self._lock:
            self._pending_capture = (
                revision,
                operation,
                signal,
                recording,
                error,
                cancelled,
            )

    def _take_pending(self):
        with self._lock:
            level = self._pending_level
            capture = self._pending_capture
            calculation = self._pending_calculation
            self._pending_level = None
            self._pending_capture = None
            self._pending_calculation = None
        return level, capture, calculation

    def _update_graphs(
        self,
        measurement: Measurement,
        result: ImpedanceResult,
    ) -> None:
        specs = (
            ("Impedance", result.magnitude, AxisSpec.IMPEDANCE),
            ("Phase", result.phase, AxisSpec.PHASE),
        )
        if len(measurement.graphs) > 2:
            removed_ids = {graph.id for graph in measurement.graphs[2:]}
            del measurement.graphs[2:]
            self.app.app_state.visible_graph_ids = [
                graph_id
                for graph_id in self.app.app_state.visible_graph_ids
                if graph_id not in removed_ids
            ]
        while len(measurement.graphs) < 2:
            name, y, axis = specs[len(measurement.graphs)]
            measurement.graphs.append(
                GraphData(name, result.frequency, y, AxisSpec.FREQ, axis)
            )
        for graph, (name, y, axis) in zip(measurement.graphs[:2], specs):
            graph.name = name
            graph.x = result.frequency
            graph.y = y
            graph.x_axis = AxisSpec.FREQ
            graph.y_axis = axis
            if graph.id not in self.app.app_state.visible_graph_ids:
                self.app.app_state.visible_graph_ids.append(graph.id)
        self.app.app_state.graph_data_changed = True

    def _clear_graphs(self, measurement: Measurement) -> None:
        ids = {graph.id for graph in measurement.graphs}
        measurement.graphs.clear()
        self.app.app_state.visible_graph_ids = [
            graph_id
            for graph_id in self.app.app_state.visible_graph_ids
            if graph_id not in ids
        ]
        self.app.app_state.graph_data_changed = True

    def _build_config(self, *, sample_rate: int | None = None) -> ImpedanceConfig:
        state = self.measurement.module_state
        rate = sample_rate or self.app.audio_input.sample_rate
        return ImpedanceConfig(
            sample_rate=rate,
            duration=state["duration"],
            reference_resistor=state["reference_resistor"],
            calibration_resistor=state["calibration_resistor"],
            band=FrequencyBand(*state["band"]),
            window_width=state["window_width"],
            points=state["points"],
            window=SmoothingWindow(state["window"]),
        )

    def _capture_signature(self, config: ImpedanceConfig) -> tuple[object, ...]:
        return (
            *config.capture_signature,
            self.app.settings.input_device,
            self.app.settings.output_device,
        )

    def _validate_audio(self, config: ImpedanceConfig) -> None:
        if self.app.audio_input.sample_rate <= 0:
            raise ValueError("Audio input device is unavailable")
        if self.app.audio_output.sample_rate <= 0:
            raise ValueError("Audio output device is unavailable")
        if self.app.audio_input.sample_rate != self.app.audio_output.sample_rate:
            raise ValueError("Input and output sample rates must be equal")
        config.validate()

    def _stored_result(self) -> ImpedanceResult | None:
        state = self.measurement.module_state
        frequency = state.get("frequency")
        impedance = state.get("impedance")
        if not isinstance(frequency, np.ndarray) or not isinstance(impedance, np.ndarray):
            return None
        return ImpedanceResult(
            np.asarray(frequency, dtype=np.float64),
            np.asarray(impedance, dtype=np.complex128),
        )

    def _finish_operation(self, status: str) -> None:
        self._operation = None
        self.app.app_state.measuring = False
        self._set_controls_enabled(True)
        self._set_status(status)

    def _fail(self, error: str, fallback: str) -> None:
        self.measurement.module_state["workflow"] = fallback
        self._operation = None
        self.app.app_state.measuring = False
        self._set_controls_enabled(True)
        self._set_status(f"Impedance error: {error}")

    def _set_status(self, status: str) -> None:
        self.measurement.module_state["status"] = status
        self.app.main_window.set_status_text(status)

    def _sync_view(self, *, force: bool = False) -> None:
        if self._view is None:
            return
        self._view.update()
        status = str(self.measurement.module_state["status"])
        if (
            force
            or status != self._shown_status
            or self._levels != self._shown_levels
        ):
            self._view.update_status(
                status,
                self._levels,
            )
            self._shown_status = status
            self._shown_levels = self._levels

    def _show_calibration_stage(self, stage: int) -> None:
        if self._view is None:
            return
        self._calibration_stage = stage
        self._view.show_calibration_stage(stage)

    def _set_controls_enabled(self, enabled: bool) -> None:
        if self._view is not None:
            self._view.set_enabled(enabled)

    def _is_busy(self) -> bool:
        return self._operation is not None

    def _fallback_workflow(self, operation: Operation) -> str:
        if operation == Operation.REFERENCE_CALIBRATION:
            return "waiting_reference"
        if operation in (Operation.MEASUREMENT, Operation.REPROCESS):
            return "calibrated"
        if operation == Operation.CHANNEL_CALIBRATION:
            return "uncalibrated"
        return self._operation_fallback

    @classmethod
    def _ensure_state(cls, state: dict[str, Any]) -> None:
        for key, value in cls.DEFAULT_STATE.items():
            state.setdefault(key, deepcopy(value))

    @staticmethod
    def _repair_transient_state(state: dict[str, Any]) -> None:
        if state["workflow"] in (
            "calibrating_channels",
            "calibrating_reference",
            "measuring",
        ):
            state["workflow"] = (
                "calibrated"
                if state.get("reference_resistor_estimated") is not None
                else "uncalibrated"
            )
            state["status"] = "Interrupted operation was not restored"

    @classmethod
    def _clear_calibration(cls, state: dict[str, Any]) -> None:
        for key in (
            "calibration_signature",
            "channel_calibration_recording",
            "channel_calibration_signal",
            "reference_calibration_recording",
            "reference_calibration_signal",
            "measurement_recording",
            "measurement_signal",
            "channel_correction",
            "reference_resistor_estimated",
            "reference_diagnostics",
            "frequency",
            "impedance",
            "fit_result",
            "spice_values",
        ):
            state[key] = None
        state["workflow"] = "uncalibrated"

    @staticmethod
    def _normalize_setting(key: str, value: Any) -> Any:
        if key == "band":
            low, high = int(value[0]), int(value[1])
            return max(1, low), max(max(1, low) + 1, high)
        if key == "duration":
            return min(120.0, max(0.1, float(value)))
        if key in ("reference_resistor", "calibration_resistor"):
            return max(0.001, float(value))
        if key == "window":
            return SmoothingWindow(value).value
        if key == "window_width":
            return min(3.0, max(0.01, float(value)))
        if key == "points":
            return min(3000, max(16, int(value)))
        raise ValueError(f"Unknown Impedance setting: {key}")
