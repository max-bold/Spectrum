from __future__ import annotations

import math
import warnings
from copy import deepcopy
from contextlib import ExitStack
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from threading import Event, Lock, Thread, current_thread
from time import monotonic, sleep
from typing import Callable

import numpy as np
import sounddevice as sd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares
from scipy.signal import chirp, correlate, find_peaks
from utils.windows import Windows, log_filter2

GENERATOR_AMPLITUDE = 0.9
MAX_FILTER_WINDOW_WIDTH = 3.0
LEVEL_UPDATE_RATE = 12.0
LEVEL_UPDATE_INTERVAL = 1.0 / LEVEL_UPDATE_RATE
LEVEL_SMOOTHING = 0.35
CHANNEL_CALIBRATION_DURATION = 5.0
LEVEL_TEST_LOOP_DURATION = 1.0
CHANNEL_SIMILARITY_THRESHOLD = 0.9
CHANNEL_SIMILARITY_MAX_DELAY_SECONDS = 0.02
CHANNEL_CALIBRATION_TONES = 12
CHANNEL_TONE_ENERGY_RATIO_MIN = 0.75
CHANNEL_GAIN_PROFILE_STD_MAX_DB = 3.0
CHANNEL_GAIN_PROFILE_PEAK_MAX_DB = 8.0
CHANNEL_PHASE_RESIDUAL_RMS_MAX_DEG = 35.0
REFERENCE_RESISTIVE_IMAG_RATIO_MAX = 0.05
CALIBRATION_RATIO_ERROR_MAX = 0.05


class MeasurementState(str, Enum):
    UNCALIBRATED = "uncalibrated"
    CALIBRATING = "calibrating"
    CALIBRATED = "calibrated"
    MEASURING = "measuring"
    MEASURING_COMPLETED = "measuring_completed"


class CalibrationStage(str, Enum):
    IDLE = "idle"
    CHANNELS = "channels"
    WAITING_REFERENCE = "waiting_reference"
    REFERENCE = "reference"


class WindowFunction(str, Enum):
    FLAT = "flat"
    COSINE = "cosine"
    GAUSSIAN = "gaussian"
    TRIANGULAR = "triangular"


class PhaseDisplayMode(str, Enum):
    ANGLE = "angle"
    DERIVATIVE = "derivative"


class MeasurementCancelled(Exception):
    pass


@dataclass(frozen=True)
class MeasurementConfig:
    sample_rate: int = 48000
    duration: float = 20.0
    reference_resistor: float = 3.25
    calibration_resistor: float = 10.4
    f_min: float = 20.0
    f_max: float = 20000.0
    window_width: float = 0.1
    points: int = 1024
    window_function: WindowFunction = WindowFunction.GAUSSIAN
    input_device: int | str | None = None
    output_device: int | str | None = None
    block_size: int = 1024
    recording_tail: float = 0.25
    spice_min_sections: int = 0
    spice_max_sections: int = 10
    spice_max_evaluations: int = 2000

    def validate(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("Sample rate must be positive")
        if self.duration <= 0:
            raise ValueError("Duration must be positive")
        if self.reference_resistor <= 0:
            raise ValueError("Reference resistor must be positive")
        if self.calibration_resistor <= 0:
            raise ValueError("Calibration resistor must be positive")
        if self.f_min <= 0 or self.f_max <= self.f_min:
            raise ValueError("Invalid frequency band")
        if self.f_max >= self.sample_rate / 2:
            raise ValueError("Upper frequency must be below Nyquist frequency")
        if not 0 < self.window_width <= MAX_FILTER_WINDOW_WIDTH:
            raise ValueError(
                f"Window width must be in the 0..{MAX_FILTER_WINDOW_WIDTH} range"
            )
        if self.points < 3:
            raise ValueError("Points number must be at least 3")
        if self.block_size <= 0:
            raise ValueError("Block size must be positive")
        if self.recording_tail < 0:
            raise ValueError("Recording tail must not be negative")
        if not 0 <= self.spice_min_sections <= self.spice_max_sections <= 10:
            raise ValueError("SPICE section range must be within 0..10")


@dataclass(frozen=True)
class FitResult:
    sections: int
    physical_params: np.ndarray
    rms_log_error: float
    max_abs_log_error: float
    selection_score: float = math.nan


@dataclass(frozen=True)
class SpiceTableValues:
    l1: str
    sections: tuple[tuple[str, str, str], ...]
    r1: str


@dataclass(frozen=True)
class ImpedanceProjectData:
    state: MeasurementState
    calibration_stage: CalibrationStage
    phase_mode: PhaseDisplayMode
    calibration_config: MeasurementConfig
    result_config: MeasurementConfig | None
    channel_calibration_recording: np.ndarray
    channel_calibration_recording_sample_rate: int
    channel_calibration_signal: np.ndarray
    channel_calibration_signal_sample_rate: int
    calibration_recording: np.ndarray | None
    calibration_recording_sample_rate: int | None
    calibration_signal: np.ndarray | None
    calibration_signal_sample_rate: int | None
    measurement_recording: np.ndarray | None
    measurement_recording_sample_rate: int | None
    measurement_signal: np.ndarray | None
    measurement_signal_sample_rate: int | None
    calibration_frequency: np.ndarray | None
    calibration_impedance: np.ndarray | None
    calibration_phase: np.ndarray | None
    calibration_phase_derivative: np.ndarray | None
    channel_correction: np.ndarray
    reference_resistor_estimated: float | None
    reference_diagnostics: dict[str, object] | None
    frequency: np.ndarray | None
    impedance: np.ndarray | None
    phase: np.ndarray | None
    phase_derivative: np.ndarray | None
    fit_result: FitResult | None
    spice_values: SpiceTableValues | None


@dataclass(frozen=True)
class AppSnapshot:
    state: MeasurementState
    calibration_stage: CalibrationStage
    processing: bool
    modeling: bool
    testing: bool
    status: str
    error: str | None
    levels: tuple[float, float]
    frequency: np.ndarray | None
    impedance: np.ndarray | None
    spice_values: SpiceTableValues | None
    project_available: bool
    revision: int


LevelCallback = Callable[[tuple[float, float]], None]
Recorder = Callable[
    [np.ndarray, MeasurementConfig, LevelCallback],
    np.ndarray,
]


def resolve_sample_rate(
    input_device: int | str | None = None,
    output_device: int | str | None = None,
) -> int:
    try:
        input_info = sd.query_devices(input_device, "input")
        output_info = sd.query_devices(output_device, "output")
    except (sd.PortAudioError, ValueError) as exc:
        raise ValueError(f"Cannot query audio devices: {exc}") from exc

    input_rate = int(round(float(input_info["default_samplerate"])))
    output_rate = int(round(float(output_info["default_samplerate"])))
    if input_rate != output_rate:
        raise ValueError(
            "Input and output default sample rates differ: "
            f"{input_rate} Hz != {output_rate} Hz"
        )
    return input_rate


class ImpedanceAppState:
    def __init__(self, recorder: Recorder | None = None) -> None:
        self._lock = Lock()
        self._recorder = recorder or play_and_record
        self._state = MeasurementState.UNCALIBRATED
        self._calibration_stage = CalibrationStage.IDLE
        self._status = "Calibration required"
        self._error: str | None = None
        self._levels = (0.0, 0.0)
        self._level_peaks = (0.0, 0.0)
        self._last_level_update: float | None = None
        self._channel_calibration_recording: np.ndarray | None = None
        self._channel_calibration_signal: np.ndarray | None = None
        self._calibration_recording: np.ndarray | None = None
        self._calibration_signal: np.ndarray | None = None
        self._measurement_recording: np.ndarray | None = None
        self._measurement_signal: np.ndarray | None = None
        self._channel_correction: np.ndarray | None = None
        self._reference_resistor_estimated: float | None = None
        self._reference_diagnostics: dict[str, object] | None = None
        self._calibration_frequency: np.ndarray | None = None
        self._calibration_impedance: np.ndarray | None = None
        self._calibration_config: MeasurementConfig | None = None
        self._frequency: np.ndarray | None = None
        self._impedance: np.ndarray | None = None
        self._fit_result: FitResult | None = None
        self._spice_values: SpiceTableValues | None = None
        self._result_config: MeasurementConfig | None = None
        self._revision = 0
        self._worker: Thread | None = None
        self._measurement_stop: Event | None = None
        self._reprocess_worker: Thread | None = None
        self._model_workers: set[Thread] = set()
        self._model_generation = 0
        self._modeling = False
        self._test_worker: Thread | None = None
        self._test_stop: Event | None = None
        self._testing = False
        self._pending_reprocess: MeasurementConfig | None = None
        self._processing = False

    def snapshot(self) -> AppSnapshot:
        with self._lock:
            return AppSnapshot(
                state=self._state,
                calibration_stage=self._calibration_stage,
                processing=self._processing,
                modeling=self._modeling,
                testing=self._testing,
                status=self._status,
                error=self._error,
                levels=self._levels,
                frequency=None if self._frequency is None else self._frequency.copy(),
                impedance=None if self._impedance is None else self._impedance.copy(),
                spice_values=self._spice_values,
                project_available=(
                    self._calibration_config is not None
                    and self._channel_calibration_recording is not None
                    and self._channel_calibration_signal is not None
                    and self._channel_correction is not None
                    and self._state in (
                        MeasurementState.CALIBRATING,
                        MeasurementState.CALIBRATED,
                        MeasurementState.MEASURING_COMPLETED,
                    )
                ),
                revision=self._revision,
            )

    def export_project(
        self,
        phase_mode: PhaseDisplayMode,
    ) -> ImpedanceProjectData:
        with self._lock:
            if (
                self._processing
                or self._testing
                or self._modeling
                or self._state == MeasurementState.MEASURING
                or self._calibration_stage in (
                    CalibrationStage.CHANNELS,
                    CalibrationStage.REFERENCE,
                )
            ):
                raise ValueError("Cannot save a project while an operation is active")
            if (
                self._calibration_config is None
                or self._channel_calibration_recording is None
                or self._channel_calibration_signal is None
                or self._channel_correction is None
            ):
                raise ValueError("Complete calibration stage 1 before saving")
            if self._state not in (
                MeasurementState.CALIBRATING,
                MeasurementState.CALIBRATED,
                MeasurementState.MEASURING_COMPLETED,
            ):
                raise ValueError("There is no impedance project to save")

            frequency = _copy_optional_array(self._frequency)
            impedance = _copy_optional_array(self._impedance)
            phase = (
                None
                if impedance is None
                else current_phase_angle(impedance)
            )
            derivative = (
                None
                if frequency is None or impedance is None
                else current_phase_derivative(frequency, impedance)
            )
            fit_result = _copy_fit_result(self._fit_result)
            config = self._calibration_config
            result_config = self._result_config
            project = ImpedanceProjectData(
                state=self._state,
                calibration_stage=self._calibration_stage,
                phase_mode=phase_mode,
                calibration_config=config,
                result_config=result_config,
                channel_calibration_recording=(
                    self._channel_calibration_recording.copy()
                ),
                channel_calibration_recording_sample_rate=config.sample_rate,
                channel_calibration_signal=self._channel_calibration_signal.copy(),
                channel_calibration_signal_sample_rate=config.sample_rate,
                calibration_recording=_copy_optional_array(
                    self._calibration_recording
                ),
                calibration_recording_sample_rate=(
                    None
                    if self._calibration_recording is None
                    else config.sample_rate
                ),
                calibration_signal=_copy_optional_array(self._calibration_signal),
                calibration_signal_sample_rate=(
                    None
                    if self._calibration_signal is None
                    else config.sample_rate
                ),
                measurement_recording=_copy_optional_array(
                    self._measurement_recording
                ),
                measurement_recording_sample_rate=(
                    None
                    if self._measurement_recording is None
                    else (result_config or config).sample_rate
                ),
                measurement_signal=_copy_optional_array(self._measurement_signal),
                measurement_signal_sample_rate=(
                    None
                    if self._measurement_signal is None
                    else (result_config or config).sample_rate
                ),
                calibration_frequency=_copy_optional_array(
                    self._calibration_frequency
                ),
                calibration_impedance=_copy_optional_array(
                    self._calibration_impedance
                ),
                calibration_phase=(
                    None
                    if self._calibration_impedance is None
                    else current_phase_angle(self._calibration_impedance)
                ),
                calibration_phase_derivative=(
                    None
                    if (
                        self._calibration_frequency is None
                        or self._calibration_impedance is None
                    )
                    else current_phase_derivative(
                        self._calibration_frequency,
                        self._calibration_impedance,
                    )
                ),
                channel_correction=self._channel_correction.copy(),
                reference_resistor_estimated=self._reference_resistor_estimated,
                reference_diagnostics=deepcopy(self._reference_diagnostics),
                frequency=frequency,
                impedance=impedance,
                phase=phase,
                phase_derivative=derivative,
                fit_result=fit_result,
                spice_values=self._spice_values,
            )
        validate_impedance_project_data(project)
        return project

    def restore_project(self, project: ImpedanceProjectData) -> None:
        validate_impedance_project_data(project)
        with self._lock:
            if (
                self._state == MeasurementState.MEASURING
                or self._calibration_stage in (
                    CalibrationStage.CHANNELS,
                    CalibrationStage.REFERENCE,
                )
                or self._processing
                or self._testing
                or self._modeling
                or any(worker.is_alive() for worker in self._model_workers)
            ):
                raise ValueError("Cannot open a project while an operation is active")

            self._state = project.state
            self._calibration_stage = project.calibration_stage
            self._status = _loaded_project_status(project)
            self._error = None
            self._levels = (0.0, 0.0)
            self._reset_level_filter_locked()
            self._channel_calibration_recording = (
                project.channel_calibration_recording.copy()
            )
            self._channel_calibration_signal = (
                project.channel_calibration_signal.copy()
            )
            self._calibration_recording = _copy_optional_array(
                project.calibration_recording
            )
            self._calibration_signal = _copy_optional_array(
                project.calibration_signal
            )
            self._measurement_recording = _copy_optional_array(
                project.measurement_recording
            )
            self._measurement_signal = _copy_optional_array(
                project.measurement_signal
            )
            self._channel_correction = project.channel_correction.copy()
            self._reference_resistor_estimated = (
                project.reference_resistor_estimated
            )
            self._reference_diagnostics = deepcopy(
                project.reference_diagnostics
            )
            self._calibration_frequency = _copy_optional_array(
                project.calibration_frequency
            )
            self._calibration_impedance = _copy_optional_array(
                project.calibration_impedance
            )
            self._calibration_config = project.calibration_config
            self._frequency = _copy_optional_array(project.frequency)
            self._impedance = _copy_optional_array(project.impedance)
            self._fit_result = _copy_fit_result(project.fit_result)
            self._spice_values = project.spice_values
            self._result_config = project.result_config
            self._pending_reprocess = None
            self._processing = False
            self._model_generation += 1
            self._modeling = False
            self._worker = None
            self._reprocess_worker = None
            self._model_workers.clear()
            self._revision += 1

    def invalidate_calibration(
        self,
        status: str = "Calibration required",
    ) -> bool:
        with self._lock:
            if self._state in (
                MeasurementState.CALIBRATING,
                MeasurementState.MEASURING,
            ) or self._processing or self._testing:
                return False
            self._state = MeasurementState.UNCALIBRATED
            self._calibration_stage = CalibrationStage.IDLE
            self._status = status
            self._error = None
            self._levels = (0.0, 0.0)
            self._reset_level_filter_locked()
            self._channel_calibration_recording = None
            self._channel_calibration_signal = None
            self._calibration_recording = None
            self._calibration_signal = None
            self._measurement_recording = None
            self._measurement_signal = None
            self._channel_correction = None
            self._reference_resistor_estimated = None
            self._reference_diagnostics = None
            self._calibration_frequency = None
            self._calibration_impedance = None
            self._calibration_config = None
            self._frequency = None
            self._impedance = None
            self._fit_result = None
            self._spice_values = None
            self._model_generation += 1
            self._modeling = False
            self._result_config = None
            self._pending_reprocess = None
            self._revision += 1
            return True

    def start_calibration(self, config: MeasurementConfig) -> bool:
        config.validate()
        with self._lock:
            if self._state in (
                MeasurementState.CALIBRATING,
                MeasurementState.MEASURING,
            ) or self._processing or self._testing:
                return False
            self._state = MeasurementState.CALIBRATING
            self._calibration_stage = CalibrationStage.CHANNELS
            self._status = "Calibrating input channels..."
            self._error = None
            self._channel_calibration_recording = None
            self._channel_calibration_signal = None
            self._calibration_recording = None
            self._calibration_signal = None
            self._measurement_recording = None
            self._measurement_signal = None
            self._channel_correction = None
            self._reference_resistor_estimated = None
            self._reference_diagnostics = None
            self._calibration_frequency = None
            self._calibration_impedance = None
            self._calibration_config = None
            self._frequency = None
            self._impedance = None
            self._fit_result = None
            self._spice_values = None
            self._model_generation += 1
            self._modeling = False
            self._result_config = None
            self._pending_reprocess = None
            self._processing = False
            self._levels = (0.0, 0.0)
            self._reset_level_filter_locked()
            self._revision += 1
        self._start_worker(self._channel_calibration_worker, config)
        return True

    def continue_calibration(self) -> bool:
        with self._lock:
            if (
                self._state != MeasurementState.CALIBRATING
                or self._calibration_stage
                != CalibrationStage.WAITING_REFERENCE
                or self._calibration_config is None
                or self._processing
            ):
                return False
            config = self._calibration_config
            self._calibration_stage = CalibrationStage.REFERENCE
            self._status = "Checking reference and calibration resistors..."
            self._error = None
            self._levels = (0.0, 0.0)
            self._reset_level_filter_locked()
            self._revision += 1
        self._start_worker(self._reference_calibration_worker, config)
        return True

    def cancel_calibration(self) -> bool:
        with self._lock:
            if (
                self._state != MeasurementState.CALIBRATING
                or self._calibration_stage
                != CalibrationStage.WAITING_REFERENCE
            ):
                return False
            self._state = MeasurementState.UNCALIBRATED
            self._calibration_stage = CalibrationStage.IDLE
            self._status = "Calibration cancelled"
            self._channel_calibration_recording = None
            self._channel_calibration_signal = None
            self._calibration_recording = None
            self._calibration_signal = None
            self._channel_correction = None
            self._reference_resistor_estimated = None
            self._reference_diagnostics = None
            self._calibration_config = None
            self._revision += 1
            return True

    def start_measurement(self, config: MeasurementConfig) -> bool:
        config.validate()
        with self._lock:
            if self._state not in (
                MeasurementState.CALIBRATED,
                MeasurementState.MEASURING_COMPLETED,
            ) or self._processing or self._testing:
                return False
            if (
                self._channel_correction is None
                or self._reference_resistor_estimated is None
            ):
                return False
            if (
                self._calibration_config is None
                or _capture_signature(config)
                != _capture_signature(self._calibration_config)
            ):
                raise ValueError("Measurement settings changed; recalibrate first")
            if (
                self._channel_calibration_recording is None
                or self._calibration_recording is None
            ):
                return False
            self._state = MeasurementState.MEASURING
            self._status = "Measuring..."
            self._error = None
            self._frequency = None
            self._impedance = None
            self._fit_result = None
            self._spice_values = None
            self._model_generation += 1
            self._modeling = False
            self._result_config = None
            self._measurement_recording = None
            self._measurement_signal = None
            self._pending_reprocess = None
            self._processing = False
            self._levels = (0.0, 0.0)
            self._reset_level_filter_locked()
            stop_event = Event()
            self._measurement_stop = stop_event
            self._revision += 1
        self._start_worker(
            self._measurement_worker,
            config,
            stop_event,
        )
        return True

    def stop_measurement(self) -> bool:
        with self._lock:
            stop_event = self._measurement_stop
            if self._state != MeasurementState.MEASURING or stop_event is None:
                return False
            self._status = "Stopping measurement..."
            self._revision += 1
            stop_event.set()
            return True

    def request_reprocess(self, config: MeasurementConfig) -> bool:
        config.validate()
        with self._lock:
            if self._state in (
                MeasurementState.CALIBRATING,
                MeasurementState.MEASURING,
            ) or self._testing:
                return False
            if (
                self._calibration_config is not None
                and _capture_signature(config)
                != _capture_signature(self._calibration_config)
            ):
                raise ValueError("Capture settings changed; recalibrate first")
            if (
                self._state != MeasurementState.MEASURING_COMPLETED
                or self._channel_calibration_recording is None
                or self._calibration_recording is None
                or self._measurement_recording is None
            ):
                return False
            if not self._processing and config == self._result_config:
                return False
            self._pending_reprocess = config
            self._processing = True
            self._fit_result = None
            self._spice_values = None
            self._model_generation += 1
            self._modeling = False
            self._status = "Reprocessing measurement..."
            self._error = None
            self._revision += 1
            worker = self._reprocess_worker
            if worker is not None and worker.is_alive():
                return True
            worker = Thread(target=self._reprocess_loop, daemon=True)
            self._reprocess_worker = worker
            worker.start()
            return True

    def request_spice_model(self) -> bool:
        with self._lock:
            if (
                self._state != MeasurementState.MEASURING_COMPLETED
                or self._frequency is None
                or self._impedance is None
                or self._result_config is None
            ):
                return False
            if self._modeling or self._spice_values is not None:
                return True
            frequency = self._frequency.copy()
            impedance = self._impedance.copy()
            config = self._result_config
            generation = self._model_generation
            self._status = "Calculating SPICE model..."
            self._revision += 1
        self._start_model_fit(
            frequency,
            impedance,
            config,
            generation,
        )
        return True

    def wait(self, timeout: float | None = None) -> None:
        worker = self._worker
        if worker is not None:
            worker.join(timeout)
        reprocess_worker = self._reprocess_worker
        if reprocess_worker is not None:
            reprocess_worker.join(timeout)
        for model_worker in tuple(self._model_workers):
            model_worker.join(timeout)
        test_worker = self._test_worker
        if test_worker is not None:
            test_worker.join(timeout)

    def start_test_signal(self, config: MeasurementConfig) -> bool:
        config.validate()
        with self._lock:
            if self._state in (
                MeasurementState.CALIBRATING,
                MeasurementState.MEASURING,
            ) or self._processing or self._testing:
                return False
            stop_event = Event()
            self._test_stop = stop_event
            self._testing = True
            self._status = "Test signal running"
            self._error = None
            self._levels = (0.0, 0.0)
            self._reset_level_filter_locked()
            self._revision += 1
        worker = Thread(
            target=self._test_signal_worker,
            args=(config, stop_event),
            daemon=True,
        )
        with self._lock:
            self._test_worker = worker
        worker.start()
        return True

    def stop_test_signal(self) -> bool:
        with self._lock:
            stop_event = self._test_stop
            if stop_event is None or not self._testing:
                return False
            self._status = "Stopping test signal..."
            self._revision += 1
            stop_event.set()
        return True

    def _start_worker(self, target: Callable, *args: object) -> None:
        worker = Thread(target=target, args=args, daemon=True)
        self._worker = worker
        worker.start()

    def _channel_calibration_worker(self, config: MeasurementConfig) -> None:
        try:
            channel_config = channel_calibration_config(config)
            signal = generate_channel_calibration_signal(config)
            recording = self._recorder(
                signal,
                channel_config,
                self._update_levels,
            )
            self._clear_levels()
            recording = trim_recording(recording, len(signal))
            analyze_recording_levels(recording, raise_on_clipping=True)
            frequency, channel_correction = calculate_channel_correction(
                recording[:, 0],
                recording[:, 1],
                channel_config,
            )
            validate_channel_correction(channel_correction)
            with self._lock:
                self._levels = (0.0, 0.0)
                self._reset_level_filter_locked()
                self._channel_calibration_recording = recording.copy()
                self._channel_calibration_signal = signal.copy()
                self._calibration_frequency = frequency
                self._channel_correction = channel_correction
                self._calibration_config = config
                self._calibration_stage = CalibrationStage.WAITING_REFERENCE
                self._status = "Connect Rref and Rcal for calibration stage 2"
                self._revision += 1
        except Exception as exc:
            self._fail(MeasurementState.UNCALIBRATED, exc)

    def _reference_calibration_worker(self, config: MeasurementConfig) -> None:
        try:
            signal = generate_measurement_signal(config)
            recording = self._recorder(
                signal,
                config,
                self._update_levels,
            )
            self._clear_levels()
            recording = trim_recording(recording, len(signal))
            analyze_recording_levels(recording, raise_on_clipping=True)
            with self._lock:
                channel_correction = self._channel_correction.copy()
            (
                frequency,
                reference_resistor_by_frequency,
                reference_resistor,
                diagnostics,
            ) = estimate_reference_resistor(
                recording[:, 0],
                recording[:, 1],
                config,
                channel_correction,
            )
            require_valid_reference_calibration(diagnostics)
            calibration_impedance = calculate_calibration_impedance(
                reference_resistor_by_frequency,
                reference_resistor,
                config.calibration_resistor,
            )
            with self._lock:
                self._levels = (0.0, 0.0)
                self._reset_level_filter_locked()
                self._calibration_recording = recording.copy()
                self._calibration_signal = signal.copy()
                self._calibration_frequency = frequency
                self._calibration_impedance = calibration_impedance.copy()
                self._reference_resistor_estimated = reference_resistor
                self._reference_diagnostics = diagnostics
                self._frequency = frequency
                self._impedance = calibration_impedance
                self._calibration_stage = CalibrationStage.IDLE
                self._state = MeasurementState.CALIBRATED
                self._status = (
                    f"Calibrated, Rref = {reference_resistor:.4g} Ohm; "
                    "showing measured Rcal Z/phase"
                )
                self._revision += 1
        except Exception as exc:
            self._fail(MeasurementState.UNCALIBRATED, exc)

    def _measurement_worker(
        self,
        config: MeasurementConfig,
        stop_event: Event,
    ) -> None:
        try:
            signal = generate_measurement_signal(config)
            if self._recorder is play_and_record:
                recording = self._recorder(
                    signal,
                    config,
                    self._update_levels,
                    stop_event,
                )
            else:
                recording = self._recorder(
                    signal,
                    config,
                    self._update_levels,
                )
            _raise_if_measurement_cancelled(stop_event)
            self._clear_levels()
            recording = trim_recording(recording, len(signal))
            analyze_recording_levels(recording, raise_on_clipping=True)
            _raise_if_measurement_cancelled(stop_event)
            with self._lock:
                channel_calibration_recording = (
                    self._channel_calibration_recording.copy()
                )
                calibration_recording = self._calibration_recording.copy()
                self._measurement_recording = recording.copy()
                self._measurement_signal = signal.copy()
            frequency, impedance = self._process_recordings(
                channel_calibration_recording,
                calibration_recording,
                recording,
                config,
            )
            _raise_if_measurement_cancelled(stop_event)
            with self._lock:
                self._levels = (0.0, 0.0)
                self._reset_level_filter_locked()
                self._frequency = frequency
                self._impedance = impedance
                self._result_config = config
                self._state = MeasurementState.MEASURING_COMPLETED
                self._status = "Measurement completed"
                self._revision += 1
        except MeasurementCancelled:
            with self._lock:
                self._levels = (0.0, 0.0)
                self._reset_level_filter_locked()
                self._measurement_recording = None
                self._measurement_signal = None
                self._frequency = _copy_optional_array(
                    self._calibration_frequency
                )
                self._impedance = _copy_optional_array(
                    self._calibration_impedance
                )
                self._result_config = None
                self._state = MeasurementState.CALIBRATED
                self._status = "Measurement stopped"
                self._error = None
                self._revision += 1
        except Exception as exc:
            self._fail(MeasurementState.CALIBRATED, exc)
        finally:
            with self._lock:
                if self._measurement_stop is stop_event:
                    self._measurement_stop = None

    def _reprocess_loop(self) -> None:
        while True:
            with self._lock:
                config = self._pending_reprocess
                self._pending_reprocess = None
                channel_calibration_recording = (
                    self._channel_calibration_recording.copy()
                )
                calibration_recording = self._calibration_recording.copy()
                measurement_recording = self._measurement_recording.copy()
            if config is None:
                with self._lock:
                    self._processing = False
                    self._reprocess_worker = None
                    self._status = "Measurement completed"
                    self._revision += 1
                return
            try:
                frequency, impedance = self._process_recordings(
                    channel_calibration_recording,
                    calibration_recording,
                    measurement_recording,
                    config,
                )
            except Exception as exc:
                with self._lock:
                    if self._pending_reprocess is not None:
                        continue
                    self._processing = False
                    self._reprocess_worker = None
                    self._status = "Reprocessing failed"
                    self._error = str(exc) or exc.__class__.__name__
                    self._revision += 1
                return
            with self._lock:
                if self._pending_reprocess is not None:
                    continue
                self._frequency = frequency
                self._impedance = impedance
                self._result_config = config
                self._processing = False
                self._reprocess_worker = None
                self._status = "Measurement reprocessed"
                self._revision += 1
            return

    @staticmethod
    def _process_recordings(
        channel_calibration_recording: np.ndarray,
        calibration_recording: np.ndarray,
        measurement_recording: np.ndarray,
        config: MeasurementConfig,
    ) -> tuple[np.ndarray, np.ndarray]:
        channel_config = channel_calibration_config(config)
        _, channel_correction = calculate_channel_correction(
            channel_calibration_recording[:, 0],
            channel_calibration_recording[:, 1],
            channel_config,
        )
        validate_channel_correction(channel_correction)
        _, _, reference_resistor, diagnostics = estimate_reference_resistor(
            calibration_recording[:, 0],
            calibration_recording[:, 1],
            config,
            channel_correction,
        )
        require_valid_reference_calibration(diagnostics)
        frequency, impedance = calculate_impedance(
            measurement_recording[:, 0],
            measurement_recording[:, 1],
            config,
            channel_correction,
            reference_resistor,
        )
        return frequency, impedance

    def _start_model_fit(
        self,
        frequency: np.ndarray,
        impedance: np.ndarray,
        config: MeasurementConfig,
        generation: int,
    ) -> None:
        worker = Thread(
            target=self._model_fit_worker,
            args=(
                frequency.copy(),
                impedance.copy(),
                config,
                generation,
            ),
            daemon=True,
        )
        with self._lock:
            if generation != self._model_generation:
                return
            self._modeling = True
            self._model_workers.add(worker)
            self._revision += 1
        worker.start()

    def _model_fit_worker(
        self,
        frequency: np.ndarray,
        impedance: np.ndarray,
        config: MeasurementConfig,
        generation: int,
    ) -> None:
        worker = current_thread()
        try:
            fit_result, _ = fit_impedance_auto(
                frequency,
                np.abs(impedance),
                min_sections=config.spice_min_sections,
                max_sections=config.spice_max_sections,
                max_evaluations=config.spice_max_evaluations,
            )
            spice_values = format_spice_table(fit_result)
            with self._lock:
                if generation == self._model_generation:
                    self._fit_result = fit_result
                    self._spice_values = spice_values
                    self._modeling = False
                    self._status = (
                        "Measurement completed, "
                        f"{fit_result.sections} SPICE sections"
                    )
                    self._revision += 1
        except Exception as exc:
            with self._lock:
                if generation == self._model_generation:
                    self._modeling = False
                    self._status = "SPICE model calculation failed"
                    self._error = str(exc) or exc.__class__.__name__
                    self._revision += 1
        finally:
            with self._lock:
                self._model_workers.discard(worker)

    def _test_signal_worker(
        self,
        config: MeasurementConfig,
        stop_event: Event,
    ) -> None:
        try:
            play_test_signal(config, stop_event, self._update_levels)
            with self._lock:
                if self._test_stop is stop_event:
                    self._testing = False
                    self._test_stop = None
                    self._test_worker = None
                    self._status = "Test signal stopped"
                    self._levels = (0.0, 0.0)
                    self._reset_level_filter_locked()
                    self._revision += 1
        except Exception as exc:
            with self._lock:
                if self._test_stop is stop_event:
                    self._testing = False
                    self._test_stop = None
                    self._test_worker = None
                    self._status = "Test signal failed"
                    self._error = str(exc) or exc.__class__.__name__
                    self._levels = (0.0, 0.0)
                    self._reset_level_filter_locked()
                    self._revision += 1

    def _fail(self, fallback: MeasurementState, exc: Exception) -> None:
        with self._lock:
            self._state = fallback
            self._calibration_stage = CalibrationStage.IDLE
            self._status = "Operation failed"
            self._error = str(exc) or exc.__class__.__name__
            self._processing = False
            self._levels = (0.0, 0.0)
            self._reset_level_filter_locked()
            self._revision += 1

    def _update_levels(self, levels: tuple[float, float]) -> None:
        now = monotonic()
        with self._lock:
            self._level_peaks = (
                max(self._level_peaks[0], levels[0]),
                max(self._level_peaks[1], levels[1]),
            )
            if (
                self._last_level_update is not None
                and now - self._last_level_update < LEVEL_UPDATE_INTERVAL
            ):
                return

            if self._last_level_update is None:
                smoothed = self._level_peaks
            else:
                smoothed = tuple(
                    previous + LEVEL_SMOOTHING * (peak - previous)
                    for previous, peak in zip(self._levels, self._level_peaks)
                )
            self._levels = (float(smoothed[0]), float(smoothed[1]))
            self._level_peaks = (0.0, 0.0)
            self._last_level_update = now
            self._revision += 1

    def _clear_levels(self) -> None:
        with self._lock:
            self._levels = (0.0, 0.0)
            self._reset_level_filter_locked()
            self._revision += 1

    def _reset_level_filter_locked(self) -> None:
        self._level_peaks = (0.0, 0.0)
        self._last_level_update = None


def generate_measurement_signal(config: MeasurementConfig) -> np.ndarray:
    config.validate()
    margin = 2 ** (MAX_FILTER_WINDOW_WIDTH / 2)
    f_start = max(1.0, config.f_min / margin)
    f_end = min(config.sample_rate * 0.49, config.f_max * margin)
    samples = int(round(config.sample_rate * config.duration))
    time = np.arange(samples, dtype=np.float64) / config.sample_rate
    signal = GENERATOR_AMPLITUDE * chirp(
        time,
        f0=f_start,
        f1=f_end,
        t1=config.duration,
        method="logarithmic",
    )
    fade_samples = min(int(round(0.02 * config.sample_rate)), samples)
    if fade_samples:
        signal[:fade_samples] *= np.linspace(0.0, 1.0, fade_samples)
        signal[-fade_samples:] *= np.linspace(1.0, 0.0, fade_samples)
    return signal.astype(np.float32)


def channel_calibration_config(config: MeasurementConfig) -> MeasurementConfig:
    return replace(config, duration=CHANNEL_CALIBRATION_DURATION)


def channel_calibration_frequencies(
    config: MeasurementConfig,
) -> np.ndarray:
    config.validate()
    duration = CHANNEL_CALIBRATION_DURATION
    resolution = 1.0 / duration
    low = max(config.f_min, 4.0 * resolution)
    high = min(config.f_max, config.sample_rate * 0.45)
    if high <= low:
        raise ValueError("Frequency band is too narrow for channel calibration")
    frequencies = np.geomspace(low, high, CHANNEL_CALIBRATION_TONES)
    coherent = np.rint(frequencies / resolution) * resolution
    coherent = np.unique(np.clip(coherent, low, high))
    if coherent.size < 3:
        raise ValueError(
            "At least three calibration tones are required in the frequency band"
        )
    return coherent


def generate_channel_calibration_signal(
    config: MeasurementConfig,
) -> np.ndarray:
    channel_config = channel_calibration_config(config)
    frequencies = channel_calibration_frequencies(channel_config)
    samples = int(round(channel_config.sample_rate * channel_config.duration))
    time = np.arange(samples, dtype=np.float64) / config.sample_rate
    indices = np.arange(frequencies.size, dtype=np.float64)
    phases = np.pi * indices * (indices - 1.0) / frequencies.size
    signal = np.sum(
        np.cos(
            2.0 * np.pi * frequencies[:, None] * time[None, :]
            + phases[:, None]
        ),
        axis=0,
    )
    peak = float(np.max(np.abs(signal)))
    if peak <= 0:
        raise ValueError("Could not generate channel calibration signal")
    signal *= GENERATOR_AMPLITUDE / peak
    fade_samples = min(int(round(0.02 * config.sample_rate)), samples // 4)
    if fade_samples:
        signal[:fade_samples] *= np.linspace(0.0, 1.0, fade_samples)
        signal[-fade_samples:] *= np.linspace(1.0, 0.0, fade_samples)
    return signal.astype(np.float32)


def generate_level_test_signal(config: MeasurementConfig) -> np.ndarray:
    config.validate()
    frequencies = np.asarray((100.0, 1000.0, 10000.0), dtype=np.float64)
    frequencies = frequencies[frequencies < config.sample_rate * 0.45]
    frequencies = frequencies[
        (frequencies >= config.f_min / 2.0)
        & (frequencies <= min(config.f_max * 2.0, config.sample_rate * 0.45))
    ]
    if frequencies.size == 0:
        frequencies = np.asarray(
            [min(max(config.f_min, 1000.0), config.sample_rate * 0.25)],
            dtype=np.float64,
        )
    samples = int(round(config.sample_rate * LEVEL_TEST_LOOP_DURATION))
    time = np.arange(samples, dtype=np.float64) / config.sample_rate
    signal = np.sum(
        np.sin(2.0 * np.pi * frequencies[:, None] * time[None, :]),
        axis=0,
    )
    peak = float(np.max(np.abs(signal)))
    if peak <= 0:
        raise ValueError("Could not generate level test signal")
    signal *= GENERATOR_AMPLITUDE / peak
    return signal.astype(np.float32)


def play_and_record(
    signal: np.ndarray,
    config: MeasurementConfig,
    level_callback: LevelCallback,
    stop_event: Event | None = None,
) -> np.ndarray:
    stop_event = stop_event or Event()
    mono = np.asarray(signal, dtype=np.float32).reshape(-1)
    tail_samples = int(round(config.recording_tail * config.sample_rate))
    if tail_samples:
        mono = np.pad(mono, (0, tail_samples))
    playback = np.column_stack((mono, mono)).astype(np.float32)
    recording = np.zeros_like(playback)
    input_position = 0
    output_position = 0
    input_finished = Event()
    output_finished = Event()
    input_extra_settings = _wasapi_shared_settings(config.input_device)
    output_extra_settings = _wasapi_shared_settings(config.output_device)

    try:
        sd.check_input_settings(
            device=config.input_device,
            channels=2,
            dtype="float32",
            samplerate=config.sample_rate,
            extra_settings=input_extra_settings,
        )
        sd.check_output_settings(
            device=config.output_device,
            channels=2,
            dtype="float32",
            samplerate=config.sample_rate,
            extra_settings=output_extra_settings,
        )
    except (sd.PortAudioError, ValueError) as exc:
        raise ValueError(f"Selected audio device settings are unsupported: {exc}") from exc

    def input_callback(indata, frames, time, status) -> None:
        nonlocal input_position
        if stop_event.is_set():
            raise sd.CallbackStop
        remaining = len(recording) - input_position
        active_frames = min(frames, remaining)
        if active_frames <= 0:
            raise sd.CallbackStop

        end = input_position + active_frames
        recording[input_position:end] = indata[:active_frames, :2]
        block = indata[:active_frames, :2]
        level_callback(
            (
                float(np.max(np.abs(block[:, 0]))),
                float(np.max(np.abs(block[:, 1]))),
            )
        )
        input_position = end
        if input_position >= len(recording):
            raise sd.CallbackStop

    def output_callback(outdata, frames, time, status) -> None:
        nonlocal output_position
        if stop_event.is_set():
            outdata.fill(0)
            raise sd.CallbackStop
        remaining = len(playback) - output_position
        active_frames = min(frames, remaining)
        outdata.fill(0)
        if active_frames <= 0:
            raise sd.CallbackStop

        end = output_position + active_frames
        outdata[:active_frames] = playback[output_position:end]
        output_position = end
        if output_position >= len(playback):
            raise sd.CallbackStop

    for attempt in range(2):
        input_position = 0
        output_position = 0
        input_finished.clear()
        output_finished.clear()
        recording.fill(0)
        try:
            with ExitStack() as streams:
                streams.enter_context(
                    sd.InputStream(
                        samplerate=config.sample_rate,
                        blocksize=config.block_size,
                        device=config.input_device,
                        channels=2,
                        dtype="float32",
                        callback=input_callback,
                        finished_callback=input_finished.set,
                        extra_settings=input_extra_settings,
                    )
                )
                streams.enter_context(
                    sd.OutputStream(
                        samplerate=config.sample_rate,
                        blocksize=config.block_size,
                        device=config.output_device,
                        channels=2,
                        dtype="float32",
                        callback=output_callback,
                        finished_callback=output_finished.set,
                        extra_settings=output_extra_settings,
                    )
                )
                timeout = config.duration + config.recording_tail + 5.0
                deadline = monotonic() + timeout
                if not output_finished.wait(timeout):
                    raise TimeoutError("Audio output stream did not finish in time")
                remaining = max(0.0, deadline - monotonic())
                if not input_finished.wait(remaining):
                    raise TimeoutError("Audio input stream did not finish in time")
            _raise_if_measurement_cancelled(stop_event)
            break
        except sd.PortAudioError as exc:
            if attempt == 0:
                sd._terminate()
                sd._initialize()
                sleep(0.25)
                continue
            raise ValueError(
                "Could not start the audio stream after retrying. Check the "
                "selected devices and close other applications using them.\n\n"
                f"PortAudio: {exc}"
            ) from exc

    return recording


def _raise_if_measurement_cancelled(stop_event: Event) -> None:
    if stop_event.is_set():
        raise MeasurementCancelled


def play_test_signal(
    config: MeasurementConfig,
    stop_event: Event,
    level_callback: LevelCallback,
) -> None:
    mono = generate_level_test_signal(config)
    playback = np.column_stack((mono, mono)).astype(np.float32)
    output_position = 0
    input_extra_settings = _wasapi_shared_settings(config.input_device)
    output_extra_settings = _wasapi_shared_settings(config.output_device)

    try:
        sd.check_input_settings(
            device=config.input_device,
            channels=2,
            dtype="float32",
            samplerate=config.sample_rate,
            extra_settings=input_extra_settings,
        )
        sd.check_output_settings(
            device=config.output_device,
            channels=2,
            dtype="float32",
            samplerate=config.sample_rate,
            extra_settings=output_extra_settings,
        )
    except (sd.PortAudioError, ValueError) as exc:
        raise ValueError(
            f"Selected audio device settings are unsupported: {exc}"
        ) from exc

    def input_callback(indata, frames, time, status) -> None:
        active = indata[:frames, :2]
        level_callback(
            (
                float(np.max(np.abs(active[:, 0]))),
                float(np.max(np.abs(active[:, 1]))),
            )
        )
        if stop_event.is_set():
            raise sd.CallbackStop

    def output_callback(outdata, frames, time, status) -> None:
        nonlocal output_position
        if stop_event.is_set():
            outdata.fill(0)
            raise sd.CallbackStop

        remaining = frames
        write_position = 0
        while remaining > 0:
            chunk = min(remaining, len(playback) - output_position)
            outdata[write_position : write_position + chunk] = playback[
                output_position : output_position + chunk
            ]
            write_position += chunk
            output_position = (output_position + chunk) % len(playback)
            remaining -= chunk

    for attempt in range(2):
        output_position = 0
        try:
            with ExitStack() as streams:
                streams.enter_context(
                    sd.InputStream(
                        samplerate=config.sample_rate,
                        blocksize=config.block_size,
                        device=config.input_device,
                        channels=2,
                        dtype="float32",
                        callback=input_callback,
                        extra_settings=input_extra_settings,
                    )
                )
                streams.enter_context(
                    sd.OutputStream(
                        samplerate=config.sample_rate,
                        blocksize=config.block_size,
                        device=config.output_device,
                        channels=2,
                        dtype="float32",
                        callback=output_callback,
                        extra_settings=output_extra_settings,
                    )
                )
                while not stop_event.wait(0.05):
                    pass
            break
        except sd.PortAudioError as exc:
            if stop_event.is_set():
                break
            if attempt == 0:
                sd._terminate()
                sd._initialize()
                sleep(0.25)
                continue
            raise ValueError(
                "Could not start the test signal stream after retrying. "
                "Check the selected devices and close other applications "
                "using it.\n\n"
                f"PortAudio: {exc}"
            ) from exc


def _wasapi_shared_settings(device: int | str | None):
    if device is None:
        return None
    try:
        device_info = sd.query_devices(device)
        host_api = sd.query_hostapis(device_info["hostapi"])
    except (sd.PortAudioError, TypeError, ValueError):
        return None
    if host_api["name"] != "Windows WASAPI":
        return None
    return sd.WasapiSettings(exclusive=False, auto_convert=True)


def trim_recording(
    recording: np.ndarray,
    signal_samples: int,
    threshold_ratio: float = 0.02,
) -> np.ndarray:
    data = _as_stereo_recording(recording)
    if signal_samples <= 0:
        raise ValueError("Signal length must be positive")
    peak = float(np.max(np.abs(data[:, 0]))) if data.size else 0.0
    start = 0
    if peak > 0:
        candidates = np.flatnonzero(np.abs(data[:, 0]) >= peak * threshold_ratio)
        if candidates.size:
            start = int(candidates[0])
    trimmed = data[start : start + signal_samples]
    if len(trimmed) < signal_samples:
        padding = np.zeros(
            (signal_samples - len(trimmed), data.shape[1]),
            dtype=data.dtype,
        )
        trimmed = np.vstack((trimmed, padding))
    return trimmed


def analyze_recording_levels(
    recording: np.ndarray,
    *,
    no_signal_threshold: float = 1e-8,
    quiet_threshold: float = 1e-4,
    clipping_threshold: float = 0.999,
    raise_on_clipping: bool = False,
) -> tuple[float, float]:
    data = _as_stereo_recording(recording)
    peaks = tuple(float(np.max(np.abs(data[:, index]))) for index in range(2))
    rms = tuple(
        float(np.sqrt(np.mean(np.square(data[:, index]))))
        for index in range(2)
    )
    channel_labels = ("Channel 1 (L)", "Channel 2 (R)")
    issues: list[str] = []
    for channel, peak, rms_level in zip(channel_labels, peaks, rms):
        if raise_on_clipping and peak >= clipping_threshold:
            issue = "clipping detected"
        elif rms_level <= no_signal_threshold:
            issue = "no signal"
        elif rms_level < quiet_threshold:
            issue = "signal is too quiet"
        else:
            continue
        issues.append(f"{channel}: {issue}")
    if issues:
        raise ValueError("Input level error:\n" + "\n".join(issues))
    return peaks


def calculate_fft_spectra(
    ch1: np.ndarray,
    ch2: np.ndarray,
    sample_rate: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x1 = np.asarray(ch1, dtype=np.float64).reshape(-1)
    x2 = np.asarray(ch2, dtype=np.float64).reshape(-1)
    size = min(x1.size, x2.size)
    if size < 2:
        raise ValueError("At least two samples are required")
    x1 = x1[:size] - np.mean(x1[:size])
    x2 = x2[:size] - np.mean(x2[:size])
    frequency = np.fft.rfftfreq(size, 1.0 / sample_rate)
    return frequency, np.fft.rfft(x1), np.fft.rfft(x2)


def calculate_channel_correction(
    ch1: np.ndarray,
    ch2: np.ndarray,
    config: MeasurementConfig,
) -> tuple[np.ndarray, np.ndarray]:
    tone_frequency = channel_calibration_frequencies(config)
    delay_samples, similarity = estimate_channel_delay(
        ch1,
        ch2,
        config.sample_rate,
    )
    tone_ch1 = extract_tone_amplitudes(
        ch1,
        config.sample_rate,
        tone_frequency,
    )
    tone_ch2 = extract_tone_amplitudes(
        ch2,
        config.sample_rate,
        tone_frequency,
    )
    validate_multitone_channels(
        ch1,
        ch2,
        tone_ch1,
        tone_ch2,
        tone_frequency,
        config.sample_rate,
        delay_samples,
        similarity,
    )

    tone_correction = tone_ch2 / tone_ch1
    delay_phase = (
        -2.0
        * np.pi
        * tone_frequency
        * delay_samples
        / config.sample_rate
    )
    residual_phase = np.unwrap(np.angle(tone_correction) - delay_phase)
    gain_db = 20.0 * np.log10(np.abs(tone_correction))

    frequency = np.geomspace(config.f_min, config.f_max, config.points)
    log_tones = np.log(tone_frequency)
    log_grid = np.log(frequency)
    interpolated_gain = np.interp(log_grid, log_tones, gain_db)
    interpolated_residual_phase = np.interp(
        log_grid,
        log_tones,
        residual_phase,
    )
    interpolated_delay_phase = (
        -2.0
        * np.pi
        * frequency
        * delay_samples
        / config.sample_rate
    )
    correction = (
        np.power(10.0, interpolated_gain / 20.0)
        * np.exp(
            1j
            * (interpolated_residual_phase + interpolated_delay_phase)
        )
    )
    return frequency, correction


def extract_tone_amplitudes(
    signal: np.ndarray,
    sample_rate: int,
    frequencies: np.ndarray,
) -> np.ndarray:
    values = np.asarray(signal, dtype=np.float64).reshape(-1)
    if values.size < 2:
        raise ValueError("At least two samples are required")
    values = values - np.mean(values)
    window = np.hanning(values.size)
    normalization = float(np.sum(window))
    if normalization <= 0:
        raise ValueError("Channel calibration recording is too short")
    time = np.arange(values.size, dtype=np.float64) / sample_rate
    windowed = values * window
    return np.asarray(
        [
            2.0
            * np.sum(
                windowed * np.exp(-2j * np.pi * frequency * time)
            )
            / normalization
            for frequency in frequencies
        ],
        dtype=np.complex128,
    )


def estimate_channel_delay(
    ch1: np.ndarray,
    ch2: np.ndarray,
    sample_rate: int,
    *,
    max_delay_seconds: float = CHANNEL_SIMILARITY_MAX_DELAY_SECONDS,
) -> tuple[int, float]:
    x1 = np.asarray(ch1, dtype=np.float64).reshape(-1)
    x2 = np.asarray(ch2, dtype=np.float64).reshape(-1)
    size = min(x1.size, x2.size)
    if size < 2:
        raise ValueError("At least two samples are required")
    x1 = x1[:size] - np.mean(x1[:size])
    x2 = x2[:size] - np.mean(x2[:size])
    norm = float(np.linalg.norm(x1) * np.linalg.norm(x2))
    if norm <= 1e-12:
        raise ValueError(
            "Channel calibration failed: one of the input signals is empty"
        )
    correlation = correlate(x2, x1, mode="full", method="fft")
    center = size - 1
    max_delay = min(
        size - 1,
        max(0, int(round(sample_rate * max_delay_seconds))),
    )
    active = correlation[
        center - max_delay : center + max_delay + 1
    ]
    peak_index = int(np.argmax(np.abs(active)))
    delay_samples = peak_index - max_delay
    similarity = min(float(np.abs(active[peak_index]) / norm), 1.0)
    return delay_samples, similarity


def validate_multitone_channels(
    ch1: np.ndarray,
    ch2: np.ndarray,
    tone_ch1: np.ndarray,
    tone_ch2: np.ndarray,
    frequencies: np.ndarray,
    sample_rate: int,
    delay_samples: int,
    similarity: float,
) -> None:
    tone_floor = 1e-9
    if np.any(np.abs(tone_ch1) < tone_floor) or np.any(
        np.abs(tone_ch2) < tone_floor
    ):
        raise ValueError(
            "Channel calibration failed: one or more test tones are missing"
        )

    signal_rms = (
        float(np.sqrt(np.mean(np.square(ch1)))),
        float(np.sqrt(np.mean(np.square(ch2)))),
    )
    tone_rms = (
        float(np.sqrt(np.sum(np.abs(tone_ch1) ** 2) / 2.0)),
        float(np.sqrt(np.sum(np.abs(tone_ch2) ** 2) / 2.0)),
    )
    energy_ratios = tuple(
        tone / max(total, 1e-12)
        for tone, total in zip(tone_rms, signal_rms)
    )
    if min(energy_ratios) < CHANNEL_TONE_ENERGY_RATIO_MIN:
        raise ValueError(
            "Channel calibration failed: the recorded signals are not the "
            "generated multitone signal"
        )

    correction = tone_ch2 / tone_ch1
    gain_db = 20.0 * np.log10(np.abs(correction))
    gain_profile = gain_db - np.median(gain_db)
    gain_std = float(np.std(gain_profile))
    gain_peak = float(np.max(np.abs(gain_profile)))
    if (
        gain_std > CHANNEL_GAIN_PROFILE_STD_MAX_DB
        or gain_peak > CHANNEL_GAIN_PROFILE_PEAK_MAX_DB
    ):
        raise ValueError(
            "Channel calibration failed: CH1 and CH2 have different level "
            f"profiles across the test tones (spread {gain_std:.2f} dB, "
            f"peak {gain_peak:.2f} dB)"
        )

    delay_phase = (
        -2.0
        * np.pi
        * frequencies
        * delay_samples
        / sample_rate
    )
    residual_phase = np.unwrap(np.angle(correction) - delay_phase)
    residual_phase -= np.median(residual_phase)
    phase_rms_deg = float(
        np.sqrt(np.mean(np.square(residual_phase))) * 180.0 / np.pi
    )
    if phase_rms_deg > CHANNEL_PHASE_RESIDUAL_RMS_MAX_DEG:
        raise ValueError(
            "Channel calibration failed: CH1 and CH2 have incompatible "
            f"phase responses across the test tones ({phase_rms_deg:.1f} deg RMS)"
        )
    if similarity < CHANNEL_SIMILARITY_THRESHOLD:
        raise ValueError(
            "Channel calibration failed: CH1 and CH2 contain different "
            f"signals (similarity {similarity:.1%}, required "
            f"{CHANNEL_SIMILARITY_THRESHOLD:.0%})"
        )


def validate_channel_similarity(
    ch1: np.ndarray,
    ch2: np.ndarray,
    sample_rate: int,
    *,
    threshold: float = CHANNEL_SIMILARITY_THRESHOLD,
    max_delay_seconds: float = CHANNEL_SIMILARITY_MAX_DELAY_SECONDS,
) -> float:
    _, similarity = estimate_channel_delay(
        ch1,
        ch2,
        sample_rate,
        max_delay_seconds=max_delay_seconds,
    )
    if similarity < threshold:
        raise ValueError(
            "Channel calibration failed: CH1 and CH2 contain different "
            f"signals (similarity {similarity:.1%}, required "
            f"{threshold:.0%}). Connect both inputs to the same audio_out "
            "point."
        )
    return similarity


def validate_channel_correction(channel_correction: np.ndarray) -> None:
    correction = np.asarray(channel_correction, dtype=np.complex128)
    valid = np.isfinite(correction.real) & np.isfinite(correction.imag)
    minimum_points = max(8, int(math.ceil(correction.size * 0.1)))
    if np.count_nonzero(valid) < minimum_points:
        raise ValueError(
            "Channel calibration failed: not enough valid frequency points. "
            "Connect CH1 and CH2 to the same audio_out point."
        )
    median_gain = float(np.median(np.abs(correction[valid])))
    if not 0.1 <= median_gain <= 10.0:
        raise ValueError(
            "Channel calibration failed: the input channel gain ratio is "
            "implausible. Connect CH1 and CH2 to the same audio_out point."
        )


def estimate_reference_resistor(
    ch1: np.ndarray,
    ch2: np.ndarray,
    config: MeasurementConfig,
    channel_correction: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, dict[str, object]]:
    frequency, measured_transfer = calculate_smoothed_transfer(
        ch1,
        ch2,
        config,
    )
    correction = np.asarray(channel_correction, dtype=np.complex128)
    if correction.shape != measured_transfer.shape:
        raise ValueError(
            "Channel correction does not match reference calibration settings"
        )

    valid_denominator = (
        (np.abs(measured_transfer) >= 1e-12)
        & (np.abs(correction) >= 1e-12)
        & np.isfinite(correction.real)
        & np.isfinite(correction.imag)
        & np.isfinite(measured_transfer.real)
        & np.isfinite(measured_transfer.imag)
    )
    h_cal = np.full(
        measured_transfer.shape,
        np.nan + 1j * np.nan,
        dtype=np.complex128,
    )
    h_cal[valid_denominator] = (
        measured_transfer[valid_denominator]
        / correction[valid_denominator]
    )
    rr_by_frequency = np.full(
        h_cal.shape,
        np.nan + 1j * np.nan,
        dtype=np.complex128,
    )
    valid_h = valid_denominator & (np.abs(h_cal) >= 1e-12)
    rr_by_frequency[valid_h] = (
        config.calibration_resistor
        * (1.0 - h_cal[valid_h])
        / h_cal[valid_h]
    )

    finite = (
        np.isfinite(rr_by_frequency.real)
        & np.isfinite(rr_by_frequency.imag)
    )
    if np.count_nonzero(finite) == 0:
        raise ValueError(
            "Reference resistor calibration failed: no valid frequency points"
        )

    point_imag_ratio = np.full(rr_by_frequency.shape, np.inf, dtype=np.float64)
    point_imag_ratio[finite] = np.abs(rr_by_frequency.imag[finite]) / np.maximum(
        np.abs(rr_by_frequency.real[finite]),
        1e-12,
    )
    resistive = (
        finite
        & (rr_by_frequency.real > 0)
        & (point_imag_ratio <= REFERENCE_RESISTIVE_IMAG_RATIO_MAX)
    )
    minimum_points = max(8, int(math.ceil(config.points * 0.1)))
    if np.count_nonzero(resistive) >= minimum_points:
        selected = resistive
    else:
        selected = finite

    real_values = rr_by_frequency.real[selected]
    imag_values = rr_by_frequency.imag[selected]
    rr_estimated = float(np.median(real_values))
    real_mean = float(np.mean(real_values))
    real_std = float(np.std(real_values))
    real_scale = max(abs(real_mean), 1e-12)
    real_cv = real_std / real_scale
    imag_abs_median = float(np.median(np.abs(imag_values)))
    imag_to_real_ratio = imag_abs_median / max(abs(rr_estimated), 1e-12)
    fullband_imag_ratio = float(
        np.median(point_imag_ratio[finite])
    )
    nominal_error = (
        abs(rr_estimated - config.reference_resistor)
        / config.reference_resistor
    )
    entered_ratio = config.calibration_resistor / config.reference_resistor
    measured_ratio = (
        config.calibration_resistor / rr_estimated
        if rr_estimated > 0
        else math.inf
    )
    ratio_error = abs(measured_ratio - entered_ratio) / entered_ratio

    diagnostic_warnings: list[str] = []
    fatal_warnings: list[str] = []
    if rr_estimated <= 0:
        fatal_warnings.append("estimated Rref is not positive")
    if np.count_nonzero(resistive) < minimum_points:
        fatal_warnings.append(
            f"only {np.count_nonzero(resistive)} resistive frequency points"
        )
    if imag_to_real_ratio > REFERENCE_RESISTIVE_IMAG_RATIO_MAX:
        fatal_warnings.append(
            "selected Rref points have a significant reactive component"
        )
    if fullband_imag_ratio > REFERENCE_RESISTIVE_IMAG_RATIO_MAX:
        diagnostic_warnings.append(
            "some Rref calibration frequencies have a significant reactive component"
        )
    if real_cv > 0.03:
        diagnostic_warnings.append(
            "Rref varies too much across the frequency band"
        )
    if ratio_error > CALIBRATION_RATIO_ERROR_MAX + 1e-12:
        fatal_warnings.append(
            "measured Rc/Rr differs from the entered ratio by more than 5%"
        )
    for message in [*diagnostic_warnings, *fatal_warnings]:
        warnings.warn(message, RuntimeWarning, stacklevel=2)

    diagnostics: dict[str, object] = {
        "rr_estimated": rr_estimated,
        "rr_nominal": config.reference_resistor,
        "rr_nominal_error_rel": nominal_error,
        "rc_rr_entered": entered_ratio,
        "rc_rr_measured": measured_ratio,
        "rc_rr_error_rel": ratio_error,
        "rr_real_median": rr_estimated,
        "rr_real_mean": real_mean,
        "rr_real_std": real_std,
        "rr_real_cv": real_cv,
        "rr_imag_abs_median": imag_abs_median,
        "rr_imag_to_real_ratio": imag_to_real_ratio,
        "rr_fullband_imag_to_real_ratio": fullband_imag_ratio,
        "valid_points_count": int(np.count_nonzero(finite)),
        "resistive_points_count": int(np.count_nonzero(resistive)),
        "selected_points_count": int(real_values.size),
        "warnings": diagnostic_warnings,
        "fatal_warnings": fatal_warnings,
    }
    return frequency, rr_by_frequency, rr_estimated, diagnostics


def require_valid_reference_calibration(
    diagnostics: dict[str, object],
) -> None:
    messages = list(diagnostics.get("fatal_warnings", []))
    if not messages:
        return
    details = "; ".join(str(message) for message in messages)
    estimated = float(diagnostics["rr_estimated"])
    nominal = float(diagnostics["rr_nominal"])
    cv = float(diagnostics["rr_real_cv"])
    reactive_ratio = float(diagnostics["rr_imag_to_real_ratio"])
    entered_ratio = float(diagnostics["rc_rr_entered"])
    measured_ratio = float(diagnostics["rc_rr_measured"])
    ratio_error = float(diagnostics["rc_rr_error_rel"])
    raise ValueError(
        "Calibration failed: the measured resistor network is invalid "
        f"({details}). Estimated Rref: {estimated:.4g} Ohm; "
        f"nominal: {nominal:.4g} Ohm; variation: {cv:.1%}; "
        f"reactive ratio: {reactive_ratio:.1%}. "
        f"Entered Rc/Rr: {entered_ratio:.4g}; measured Rc/Rr: "
        f"{measured_ratio:.4g}; difference: {ratio_error:.1%}. "
        "Check the entered resistor values and make sure Rref and Rcal "
        "are connected to the input."
    )


def calculate_calibration_impedance(
    reference_resistor_by_frequency: np.ndarray,
    reference_resistor: float,
    calibration_resistor: float,
) -> np.ndarray:
    reference_by_frequency = np.asarray(
        reference_resistor_by_frequency,
        dtype=np.complex128,
    )
    impedance = np.full(
        reference_by_frequency.shape,
        np.nan + 1j * np.nan,
        dtype=np.complex128,
    )
    valid = (
        np.isfinite(reference_by_frequency.real)
        & np.isfinite(reference_by_frequency.imag)
        & (np.abs(reference_by_frequency) >= 1e-12)
    )
    impedance[valid] = (
        reference_resistor
        * calibration_resistor
        / reference_by_frequency[valid]
    )
    return impedance


def calculate_impedance(
    ch1: np.ndarray,
    ch2: np.ndarray,
    config: MeasurementConfig,
    channel_correction: np.ndarray | None = None,
    reference_resistor: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    actual_reference = (
        config.reference_resistor
        if reference_resistor is None
        else float(reference_resistor)
    )
    if actual_reference <= 0:
        raise ValueError("Reference resistor must be positive")
    frequency, measured_transfer = calculate_smoothed_transfer(
        ch1,
        ch2,
        config,
    )
    valid = (
        np.isfinite(measured_transfer.real)
        & np.isfinite(measured_transfer.imag)
    )
    transfer = measured_transfer.copy()
    if channel_correction is not None:
        correction = np.asarray(channel_correction, dtype=np.complex128)
        if correction.shape != measured_transfer.shape:
            raise ValueError(
                "Channel correction does not match measurement settings"
            )
        correction_valid = (
            (np.abs(correction) >= 1e-12)
            & np.isfinite(correction.real)
            & np.isfinite(correction.imag)
        )
        valid &= correction_valid
        transfer[correction_valid] /= correction[correction_valid]

    denominator_valid = valid & (np.abs(1.0 - transfer) >= 1e-12)
    impedance = np.full(
        transfer.shape,
        np.nan + 1j * np.nan,
        dtype=np.complex128,
    )
    impedance[denominator_valid] = (
        actual_reference
        * transfer[denominator_valid]
        / (1.0 - transfer[denominator_valid])
    )
    return frequency, impedance


def calculate_smoothed_transfer(
    ch1: np.ndarray,
    ch2: np.ndarray,
    config: MeasurementConfig,
) -> tuple[np.ndarray, np.ndarray]:
    frequency, v1, v2 = calculate_fft_spectra(ch1, ch2, config.sample_rate)
    transfer = np.full(v1.shape, np.nan + 1j * np.nan, dtype=np.complex128)
    valid = np.abs(v1) >= 1e-12
    transfer[valid] = v2[valid] / v1[valid]
    window = Windows(config.window_function.value)
    filtered_frequency, filtered_transfer = log_filter2(
        frequency,
        transfer,
        band=(config.f_min, config.f_max),
        window=window,
        w=config.window_width,
        n_output=config.points,
    )
    return filtered_frequency, filtered_transfer


def smooth_fft_spectra(
    frequency: np.ndarray,
    v1: np.ndarray,
    v2: np.ndarray,
    config: MeasurementConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    window = Windows(config.window_function.value)
    band = (config.f_min, config.f_max)
    filtered_frequency, filtered_v1 = log_filter2(
        frequency,
        v1,
        band=band,
        window=window,
        w=config.window_width,
        n_output=config.points,
    )
    _, filtered_v2 = log_filter2(
        frequency,
        v2,
        band=band,
        window=window,
        w=config.window_width,
        n_output=config.points,
    )
    return filtered_frequency, filtered_v1, filtered_v2


def impedance_plot_data(
    impedance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(impedance, dtype=np.complex128)
    return np.abs(values), current_phase_angle(values)


def impedance_axis_limits(
    magnitude: np.ndarray,
    *,
    headroom: float = 0.05,
) -> tuple[float, float]:
    values = np.asarray(magnitude, dtype=np.float64)
    valid = values[np.isfinite(values) & (values >= 0)]
    if valid.size == 0:
        return 0.0, 1.0
    maximum = float(np.max(valid))
    if maximum <= 0:
        return 0.0, 1.0
    return 0.0, maximum * (1.0 + headroom)


def phase_axis_limits(
    phase: np.ndarray,
    *,
    headroom: float = 0.05,
) -> tuple[float, float]:
    values = np.asarray(phase, dtype=np.float64)
    valid = values[np.isfinite(values)]
    if valid.size == 0:
        return -1.0, 1.0
    lower = float(np.min(valid))
    upper = float(np.max(valid))
    span = upper - lower
    padding = span * headroom if span > 0 else max(abs(lower) * headroom, 1.0)
    return lower - padding, upper + padding


def current_phase_angle(impedance: np.ndarray) -> np.ndarray:
    values = np.asarray(impedance, dtype=np.complex128)
    phase = np.full(values.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(values.real) & np.isfinite(values.imag)
    if np.any(valid):
        phase[valid] = -np.rad2deg(np.unwrap(np.angle(values[valid])))
    return phase


def current_phase_derivative(
    frequency: np.ndarray,
    impedance: np.ndarray,
    *,
    smoothing_sigma: float = 2.0,
) -> np.ndarray:
    frequency = np.asarray(frequency, dtype=np.float64)
    values = np.asarray(impedance, dtype=np.complex128)
    if frequency.shape != values.shape:
        raise ValueError("Frequency and impedance arrays must have equal shapes")

    result = np.full(frequency.shape, np.nan, dtype=np.float64)
    valid = (
        (frequency > 0)
        & np.isfinite(frequency)
        & np.isfinite(values.real)
        & np.isfinite(values.imag)
    )
    if np.count_nonzero(valid) < 3:
        return result

    valid_frequency = frequency[valid]
    if np.any(np.diff(valid_frequency) <= 0):
        raise ValueError("Frequency values must be strictly increasing")
    phase = current_phase_angle(values[valid])
    if smoothing_sigma > 0:
        phase = gaussian_filter1d(
            phase,
            sigma=smoothing_sigma,
            mode="nearest",
        )
    result[valid] = np.gradient(
        phase,
        np.log10(valid_frequency),
        edge_order=2,
    )
    return result


def phase_plot_data(
    frequency: np.ndarray,
    impedance: np.ndarray,
    mode: PhaseDisplayMode,
) -> tuple[np.ndarray, str, str]:
    if mode == PhaseDisplayMode.ANGLE:
        return current_phase_angle(impedance), "Current phase (deg)", "Angle"
    if mode == PhaseDisplayMode.DERIVATIVE:
        return (
            current_phase_derivative(frequency, impedance),
            "d phase / d log10(f) (deg/decade)",
            "Phase derivative",
        )
    raise ValueError(f"Unknown phase display mode: {mode}")


def export_impedance_plot(
    path: str | Path,
    frequency: np.ndarray,
    impedance: np.ndarray,
) -> Path:
    frequency = np.asarray(frequency, dtype=np.float64)
    magnitude, phase = impedance_plot_data(impedance)
    if frequency.size == 0 or frequency.shape != magnitude.shape:
        raise ValueError("No valid impedance data to export")

    output_path = Path(path)
    if output_path.suffix.lower() not in (".png", ".jpg", ".jpeg"):
        output_path = output_path.with_suffix(".png")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure = Figure(figsize=(10, 6), dpi=150)
    FigureCanvasAgg(figure)
    impedance_axis = figure.subplots()
    phase_axis = impedance_axis.twinx()
    impedance_line = impedance_axis.semilogx(
        frequency,
        magnitude,
        label="|Z|",
    )[0]
    phase_line = phase_axis.semilogx(
        frequency,
        phase,
        color="tab:red",
        label="Phase",
    )[0]
    impedance_axis.set_title("Impedance")
    impedance_axis.set_xlabel("Frequency, Hz")
    impedance_axis.set_ylabel("Impedance, Ohm")
    phase_axis.set_ylabel("Phase, deg", color="tab:red")
    phase_axis.tick_params(axis="y", labelcolor="tab:red")
    impedance_axis.grid(True, which="both")
    impedance_axis.legend(
        handles=(impedance_line, phase_line),
        loc="best",
    )
    figure.tight_layout()
    figure.savefig(output_path)
    return output_path


def rlc_from_rf0q(
    resistance: np.ndarray,
    frequency: np.ndarray,
    quality: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    omega = 2.0 * np.pi * frequency
    inductance = resistance / (quality * omega)
    capacitance = quality / (resistance * omega)
    return inductance, capacitance


def rlc_parallel_impedance(
    omega: np.ndarray,
    resistance: float,
    frequency: float,
    quality: float,
) -> np.ndarray:
    inductance, capacitance = rlc_from_rf0q(
        np.array([resistance]),
        np.array([frequency]),
        np.array([quality]),
    )
    jw = 1j * omega
    admittance = (
        1.0 / resistance
        + 1.0 / (jw * inductance[0])
        + jw * capacitance[0]
    )
    return 1.0 / admittance


def speaker_impedance(
    frequency: np.ndarray,
    physical_params: np.ndarray,
    sections: int,
) -> np.ndarray:
    params = np.asarray(physical_params, dtype=np.float64)
    expected = 2 + sections * 3
    if params.size != expected:
        raise ValueError(f"Expected {expected} model parameters")
    omega = 2.0 * np.pi * np.asarray(frequency, dtype=np.float64)
    impedance = params[0] + 1j * omega * params[1]
    for index in range(sections):
        resistance, f0, quality = params[
            2 + index * 3 : 2 + (index + 1) * 3
        ]
        impedance += rlc_parallel_impedance(
            omega,
            resistance,
            f0,
            quality,
        )
    return impedance


def unpack_sections(
    physical_params: np.ndarray,
    sections: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    section_params = np.asarray(
        physical_params[2:],
        dtype=np.float64,
    ).reshape(sections, 3)
    resistance = section_params[:, 0]
    inductance, capacitance = rlc_from_rf0q(
        resistance,
        section_params[:, 1],
        section_params[:, 2],
    )
    return resistance, inductance, capacitance


def fit_impedance(
    frequency: np.ndarray,
    measured_magnitude: np.ndarray,
    sections: int,
    *,
    max_evaluations: int = 2000,
) -> FitResult:
    frequency, measured_magnitude = _validate_fit_data(
        frequency,
        measured_magnitude,
    )
    initial = make_initial_guess(frequency, measured_magnitude, sections)
    lower, upper = make_model_bounds(frequency, sections)
    solution = least_squares(
        model_residual,
        np.clip(np.log(initial), lower, upper),
        bounds=(lower, upper),
        args=(frequency, measured_magnitude, sections),
        loss="soft_l1",
        f_scale=0.08,
        x_scale="jac",
        max_nfev=max_evaluations,
    )
    residual = model_residual(
        solution.x,
        frequency,
        measured_magnitude,
        sections,
    )
    return FitResult(
        sections=sections,
        physical_params=np.exp(solution.x),
        rms_log_error=float(np.sqrt(np.mean(residual * residual))),
        max_abs_log_error=float(np.max(np.abs(residual))),
    )


def fit_impedance_auto(
    frequency: np.ndarray,
    measured_magnitude: np.ndarray,
    *,
    min_sections: int = 0,
    max_sections: int = 10,
    max_evaluations: int = 2000,
) -> tuple[FitResult, list[FitResult]]:
    candidates: list[FitResult] = []
    for sections in range(min_sections, max_sections + 1):
        result = fit_impedance(
            frequency,
            measured_magnitude,
            sections,
            max_evaluations=max_evaluations,
        )
        score = _bic_score(result, len(frequency))
        candidates.append(replace(result, selection_score=score))
    return min(candidates, key=lambda item: item.selection_score), candidates


def make_initial_guess(
    frequency: np.ndarray,
    measured_magnitude: np.ndarray,
    sections: int,
) -> np.ndarray:
    minimum = float(np.min(measured_magnitude))
    last = float(measured_magnitude[-1])
    re0 = float(np.clip(minimum * 0.9, 0.1, 100.0))
    omega_max = 2.0 * np.pi * float(frequency[-1])
    le0 = math.sqrt(max(last * last - re0 * re0, 1e-12)) / omega_max
    guesses: list[float] = [re0, float(np.clip(le0, 1e-7, 1e-1))]
    peaks, properties = find_peaks(
        measured_magnitude,
        prominence=max(
            0.5,
            float(np.ptp(measured_magnitude)) * 0.04,
        ),
        distance=3,
    )
    prominences = properties.get(
        "prominences",
        np.zeros_like(peaks, dtype=float),
    )
    ranked = sorted(
        zip(peaks, prominences),
        key=lambda item: item[1],
        reverse=True,
    )
    selected = sorted(int(index) for index, _ in ranked[:sections])
    for peak in selected:
        guesses.extend(
            [
                max(float(measured_magnitude[peak]) - re0, 1.0),
                float(frequency[peak]),
                3.0,
            ]
        )
    missing = sections - len(selected)
    if missing:
        filler = np.geomspace(
            max(float(frequency[0]) * 2.0, float(frequency[0])),
            max(float(frequency[0]) * 2.01, float(frequency[-1]) / 2.0),
            missing,
        )
        resistance = max(
            float(np.percentile(measured_magnitude, 75)) - re0,
            1.0,
        )
        for f0 in filler:
            guesses.extend([resistance, float(f0), 3.0])
    lower, upper = make_model_bounds(frequency, sections, physical=True)
    return np.clip(np.asarray(guesses), lower, upper)


def make_model_bounds(
    frequency: np.ndarray,
    sections: int,
    *,
    physical: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    lower = [0.1, 1e-7]
    upper = [100.0, 1e-1]
    for _ in range(sections):
        lower.extend([0.01, float(frequency[0]) / 2.0, 0.1])
        upper.extend([1000.0, float(frequency[-1]) * 2.0, 100.0])
    lower_array = np.asarray(lower)
    upper_array = np.asarray(upper)
    if physical:
        return lower_array, upper_array
    return np.log(lower_array), np.log(upper_array)


def model_residual(
    log_params: np.ndarray,
    frequency: np.ndarray,
    measured_magnitude: np.ndarray,
    sections: int,
) -> np.ndarray:
    modeled = np.abs(
        speaker_impedance(frequency, np.exp(log_params), sections)
    )
    return np.log(np.maximum(modeled, 1e-30)) - np.log(measured_magnitude)


def format_spice_table(result: FitResult) -> SpiceTableValues:
    resistance, inductance, capacitance = unpack_sections(
        result.physical_params,
        result.sections,
    )
    values = [
        (
            _format_value(inductance[index] * 1e3),
            _format_value(capacitance[index] * 1e6),
            _format_value(resistance[index]),
        )
        for index in range(result.sections)
    ]
    values.extend([("", "", "")] * (10 - len(values)))
    return SpiceTableValues(
        l1=_format_value(float(result.physical_params[1]) * 1e3),
        sections=tuple(values),
        r1=_format_value(float(result.physical_params[0])),
    )


def _validate_fit_data(
    frequency: np.ndarray,
    measured_magnitude: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    frequency = np.asarray(frequency, dtype=np.float64)
    measured = np.asarray(measured_magnitude, dtype=np.float64)
    mask = (
        np.isfinite(frequency)
        & np.isfinite(measured)
        & (frequency > 0)
        & (measured > 0)
    )
    frequency = frequency[mask]
    measured = measured[mask]
    if len(frequency) < 3:
        raise ValueError("At least three valid impedance points are required")
    order = np.argsort(frequency)
    return frequency[order], measured[order]


def _bic_score(result: FitResult, sample_count: int) -> float:
    parameter_count = 2 + result.sections * 3
    variance = max(result.rms_log_error**2, 1e-30)
    return (
        sample_count * math.log(variance)
        + parameter_count * math.log(sample_count)
    )


def _format_value(value: float) -> str:
    return f"{value:.3g}"


def _capture_signature(config: MeasurementConfig) -> tuple[object, ...]:
    return (
        config.sample_rate,
        config.duration,
        config.reference_resistor,
        config.calibration_resistor,
        config.f_min,
        config.f_max,
        config.input_device,
        config.output_device,
        config.block_size,
        config.recording_tail,
    )


def validate_impedance_project_data(project: ImpedanceProjectData) -> None:
    project.calibration_config.validate()
    if project.result_config is not None:
        project.result_config.validate()
        if _capture_signature(project.result_config) != _capture_signature(
            project.calibration_config
        ):
            raise ValueError("Project measurement settings require recalibration")
    if project.state == MeasurementState.CALIBRATING:
        if project.calibration_stage != CalibrationStage.WAITING_REFERENCE:
            raise ValueError("Only calibration stage 1 can be restored")
    elif project.state in (
        MeasurementState.CALIBRATED,
        MeasurementState.MEASURING_COMPLETED,
    ):
        if project.calibration_stage != CalibrationStage.IDLE:
            raise ValueError("Completed calibration must use the idle stage")
    else:
        raise ValueError("Unsupported impedance project state")

    _validate_project_capture(
        "channel calibration",
        project.channel_calibration_recording,
        project.channel_calibration_recording_sample_rate,
        project.channel_calibration_signal,
        project.channel_calibration_signal_sample_rate,
    )
    if (
        project.channel_calibration_recording_sample_rate
        != project.calibration_config.sample_rate
    ):
        raise ValueError("Channel calibration sample rate does not match settings")
    correction = np.asarray(project.channel_correction)
    if correction.ndim != 1 or correction.size < 3:
        raise ValueError("Invalid channel calibration correction")
    if (
        project.calibration_frequency is not None
        and np.asarray(project.calibration_frequency).size != correction.size
    ):
        raise ValueError("Channel calibration result sizes do not match")

    has_reference = project.state in (
        MeasurementState.CALIBRATED,
        MeasurementState.MEASURING_COMPLETED,
    )
    if has_reference:
        _validate_project_capture(
            "reference calibration",
            project.calibration_recording,
            project.calibration_recording_sample_rate,
            project.calibration_signal,
            project.calibration_signal_sample_rate,
        )
        if (
            project.calibration_recording_sample_rate
            != project.calibration_config.sample_rate
        ):
            raise ValueError(
                "Reference calibration sample rate does not match settings"
            )
        if (
            project.reference_resistor_estimated is None
            or project.reference_resistor_estimated <= 0
            or not isinstance(project.reference_diagnostics, dict)
            or project.calibration_frequency is None
            or project.calibration_impedance is None
            or project.calibration_phase is None
            or project.calibration_phase_derivative is None
        ):
            raise ValueError("Incomplete reference calibration results")
        calibration_sizes = {
            np.asarray(array).size
            for array in (
                project.calibration_frequency,
                project.calibration_impedance,
                project.calibration_phase,
                project.calibration_phase_derivative,
            )
        }
        if len(calibration_sizes) != 1:
            raise ValueError("Calibration result sizes do not match")

    if project.state == MeasurementState.MEASURING_COMPLETED:
        if project.result_config is None:
            raise ValueError("Measurement project has no result settings")
        _validate_project_capture(
            "measurement",
            project.measurement_recording,
            project.measurement_recording_sample_rate,
            project.measurement_signal,
            project.measurement_signal_sample_rate,
        )
        if (
            project.measurement_recording_sample_rate
            != project.result_config.sample_rate
        ):
            raise ValueError("Measurement sample rate does not match settings")

    arrays = (
        project.frequency,
        project.impedance,
        project.phase,
        project.phase_derivative,
    )
    present = [array is not None for array in arrays]
    if any(present) and not all(present):
        raise ValueError("Incomplete calculated impedance results")
    if project.state in (
        MeasurementState.CALIBRATED,
        MeasurementState.MEASURING_COMPLETED,
    ) and not all(present):
        raise ValueError("Project has no calculated impedance results")
    if all(present):
        sizes = {np.asarray(array).size for array in arrays}
        if len(sizes) != 1 or np.asarray(project.frequency).ndim != 1:
            raise ValueError("Calculated impedance result sizes do not match")

    if (project.fit_result is None) != (project.spice_values is None):
        raise ValueError("Incomplete SPICE model results")
    if project.fit_result is not None:
        expected = 2 + 3 * project.fit_result.sections
        if np.asarray(project.fit_result.physical_params).size != expected:
            raise ValueError("Invalid SPICE model parameter count")
        if len(project.spice_values.sections) != 10:
            raise ValueError("Invalid SPICE table section count")


def _validate_project_capture(
    name: str,
    recording: np.ndarray | None,
    recording_sample_rate: int | None,
    signal: np.ndarray | None,
    signal_sample_rate: int | None,
) -> None:
    if recording is None or signal is None:
        raise ValueError(f"Missing {name} audio data")
    data = np.asarray(recording)
    generator = np.asarray(signal)
    if data.ndim != 2 or data.shape[1] < 2 or len(data) == 0:
        raise ValueError(f"Invalid {name} recording")
    if generator.ndim != 1 or len(generator) == 0:
        raise ValueError(f"Invalid {name} generator signal")
    if len(data) != len(generator):
        raise ValueError(f"{name.capitalize()} signal length mismatch")
    if (
        recording_sample_rate is None
        or signal_sample_rate is None
        or recording_sample_rate <= 0
        or signal_sample_rate <= 0
    ):
        raise ValueError(f"Invalid {name} sample rate")
    if recording_sample_rate != signal_sample_rate:
        raise ValueError(f"{name.capitalize()} sample rates do not match")


def _copy_optional_array(array: np.ndarray | None) -> np.ndarray | None:
    return None if array is None else np.asarray(array).copy()


def _copy_fit_result(result: FitResult | None) -> FitResult | None:
    if result is None:
        return None
    return FitResult(
        sections=result.sections,
        physical_params=result.physical_params.copy(),
        rms_log_error=result.rms_log_error,
        max_abs_log_error=result.max_abs_log_error,
        selection_score=result.selection_score,
    )


def _loaded_project_status(project: ImpedanceProjectData) -> str:
    if project.state == MeasurementState.CALIBRATING:
        return "Project loaded; connect Rref and Rcal for calibration stage 2"
    if project.state == MeasurementState.CALIBRATED:
        return "Calibrated project loaded; showing measured Rcal Z/phase"
    return "Measurement project loaded"


def _as_stereo_recording(recording: np.ndarray) -> np.ndarray:
    data = np.asarray(recording)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError("Recording must contain at least two channels")
    return data[:, :2]
