from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray
from scipy.signal import chirp, correlate

from .smoothing import SmoothingWindow, log_smooth
from .types import ASignal, FrequencyBand


GENERATOR_AMPLITUDE = 0.9
MAX_FILTER_WINDOW_WIDTH = 3.0
CHANNEL_CALIBRATION_DURATION = 5.0
LEVEL_TEST_LOOP_DURATION = 1.0
CHANNEL_CALIBRATION_TONES = 12
CHANNEL_SIMILARITY_THRESHOLD = 0.9
CHANNEL_SIMILARITY_MAX_DELAY_SECONDS = 0.02
CHANNEL_TONE_ENERGY_RATIO_MIN = 0.75
CHANNEL_GAIN_PROFILE_STD_MAX_DB = 3.0
CHANNEL_GAIN_PROFILE_PEAK_MAX_DB = 8.0
CHANNEL_PHASE_RESIDUAL_RMS_MAX_DEG = 35.0
REFERENCE_RESISTIVE_IMAG_RATIO_MAX = 0.05
CALIBRATION_RATIO_ERROR_MAX = 0.05


@dataclass(frozen=True)
class ImpedanceConfig:
    sample_rate: int = 48_000
    duration: float = 20.0
    reference_resistor: float = 3.25
    calibration_resistor: float = 10.4
    band: FrequencyBand = FrequencyBand()
    window_width: float = 0.1
    points: int = 1024
    window: SmoothingWindow = SmoothingWindow.GAUSSIAN

    def validate(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("Sample rate must be positive")
        if self.duration <= 0:
            raise ValueError("Duration must be positive")
        if self.reference_resistor <= 0 or self.calibration_resistor <= 0:
            raise ValueError("Resistor values must be positive")
        self.band.validate(nyquist=self.sample_rate / 2)
        if not 0 < self.window_width <= MAX_FILTER_WINDOW_WIDTH:
            raise ValueError(
                f"Window width must be in the 0..{MAX_FILTER_WINDOW_WIDTH} range"
            )
        if self.points < 3:
            raise ValueError("Points number must be at least 3")

    @property
    def capture_signature(self) -> tuple[object, ...]:
        return (
            self.sample_rate,
            self.duration,
            self.reference_resistor,
            self.calibration_resistor,
            self.band,
        )


@dataclass(frozen=True)
class ImpedanceResult:
    frequency: NDArray[np.float64]
    impedance: NDArray[np.complex128]

    @property
    def magnitude(self) -> NDArray[np.float64]:
        return np.abs(self.impedance)

    @property
    def phase(self) -> NDArray[np.float64]:
        return current_phase_angle(self.impedance)


@dataclass(frozen=True)
class ChannelCalibration:
    frequency: NDArray[np.float64]
    correction: NDArray[np.complex128]


@dataclass(frozen=True)
class ReferenceCalibration:
    frequency: NDArray[np.float64]
    impedance: NDArray[np.complex128]
    reference_resistor: float
    diagnostics: dict[str, object]


def generate_measurement_signal(config: ImpedanceConfig) -> ASignal:
    config.validate()
    margin = 2 ** (MAX_FILTER_WINDOW_WIDTH / 2)
    start = max(1.0, config.band.low / margin)
    end = min(config.sample_rate * 0.49, config.band.high * margin)
    samples = int(round(config.sample_rate * config.duration))
    time = np.arange(samples, dtype=np.float64) / config.sample_rate
    data = GENERATOR_AMPLITUDE * chirp(
        time,
        f0=start,
        f1=end,
        t1=config.duration,
        method="logarithmic",
    )
    return ASignal(data, config.sample_rate).fade(
        in_=min(int(round(0.02 * config.sample_rate)), samples),
        out=min(int(round(0.02 * config.sample_rate)), samples),
    )


def channel_calibration_config(config: ImpedanceConfig) -> ImpedanceConfig:
    return replace(config, duration=CHANNEL_CALIBRATION_DURATION)


def channel_calibration_frequencies(
    config: ImpedanceConfig,
) -> NDArray[np.float64]:
    config.validate()
    resolution = 1.0 / CHANNEL_CALIBRATION_DURATION
    low = max(config.band.low, 4.0 * resolution)
    high = min(config.band.high, config.sample_rate * 0.45)
    if high <= low:
        raise ValueError("Frequency band is too narrow for channel calibration")
    frequencies = np.geomspace(low, high, CHANNEL_CALIBRATION_TONES)
    coherent = np.rint(frequencies / resolution) * resolution
    coherent = np.unique(np.clip(coherent, low, high))
    if coherent.size < 3:
        raise ValueError("At least three channel calibration tones are required")
    return coherent.astype(np.float64)


def generate_channel_calibration_signal(config: ImpedanceConfig) -> ASignal:
    calibration_config = channel_calibration_config(config)
    frequencies = channel_calibration_frequencies(calibration_config)
    samples = int(round(config.sample_rate * CHANNEL_CALIBRATION_DURATION))
    time = np.arange(samples, dtype=np.float64) / config.sample_rate
    indices = np.arange(frequencies.size, dtype=np.float64)
    phases = np.pi * indices * (indices - 1.0) / frequencies.size
    data = np.sum(
        np.cos(
            2.0 * np.pi * frequencies[:, None] * time[None, :]
            + phases[:, None]
        ),
        axis=0,
    )
    peak = float(np.max(np.abs(data)))
    if peak <= 0:
        raise ValueError("Could not generate channel calibration signal")
    data *= GENERATOR_AMPLITUDE / peak
    fade = min(int(round(0.02 * config.sample_rate)), samples // 4)
    return ASignal(data, config.sample_rate).fade(in_=fade, out=fade)


def generate_level_test_signal(config: ImpedanceConfig) -> ASignal:
    config.validate()
    frequencies = np.asarray((100.0, 1000.0, 10_000.0), dtype=np.float64)
    frequencies = frequencies[
        (frequencies < config.sample_rate * 0.45)
        & (frequencies >= config.band.low / 2.0)
        & (frequencies <= min(config.band.high * 2.0, config.sample_rate * 0.45))
    ]
    if frequencies.size == 0:
        frequencies = np.asarray(
            [min(max(config.band.low, 1000.0), config.sample_rate * 0.25)]
        )
    samples = int(round(config.sample_rate * LEVEL_TEST_LOOP_DURATION))
    time = np.arange(samples, dtype=np.float64) / config.sample_rate
    data = np.sum(
        np.sin(2.0 * np.pi * frequencies[:, None] * time[None, :]),
        axis=0,
    )
    peak = float(np.max(np.abs(data)))
    if peak <= 0:
        raise ValueError("Could not generate level test signal")
    return ASignal(data * (GENERATOR_AMPLITUDE / peak), config.sample_rate)


def trim_recording(
    recording: ASignal,
    signal_samples: int,
    threshold_ratio: float = 0.02,
) -> ASignal:
    _require_stereo(recording)
    if signal_samples <= 0:
        raise ValueError("Signal length must be positive")
    data = recording.as_array()
    peak = float(np.max(np.abs(data[:, 0]))) if data.size else 0.0
    start = 0
    if peak > 0:
        candidates = np.flatnonzero(np.abs(data[:, 0]) >= peak * threshold_ratio)
        if candidates.size:
            start = int(candidates[0])
    trimmed = recording.trim(signal_samples, start)
    missing = signal_samples - trimmed.sample_count
    return trimmed.pad(out=missing) if missing > 0 else trimmed


def analyze_recording_levels(
    recording: ASignal,
    *,
    no_signal_threshold: float = 1e-8,
    quiet_threshold: float = 1e-4,
    clipping_threshold: float = 0.999,
    raise_on_clipping: bool = False,
) -> tuple[float, float]:
    _require_stereo(recording)
    data = recording.as_array(np.float64)
    peaks = tuple(float(np.max(np.abs(data[:, index]))) for index in range(2))
    rms = tuple(
        float(np.sqrt(np.mean(np.square(data[:, index])))) for index in range(2)
    )
    issues: list[str] = []
    for label, peak, level in zip(("Channel 1 (L)", "Channel 2 (R)"), peaks, rms):
        if raise_on_clipping and peak >= clipping_threshold:
            issue = "clipping detected"
        elif level <= no_signal_threshold:
            issue = "no signal"
        elif level < quiet_threshold:
            issue = "signal is too quiet"
        else:
            continue
        issues.append(f"{label}: {issue}")
    if issues:
        raise ValueError("Input level error:\n" + "\n".join(issues))
    return peaks[0], peaks[1]


def calculate_channel_correction(
    recording: ASignal,
    config: ImpedanceConfig,
) -> ChannelCalibration:
    _validate_recording(recording, config)
    data = recording.as_array(np.float64)
    tones = channel_calibration_frequencies(config)
    delay_samples, similarity = _estimate_channel_delay(
        data[:, 0], data[:, 1], config.sample_rate
    )
    tone_ch1 = _extract_tone_amplitudes(data[:, 0], config.sample_rate, tones)
    tone_ch2 = _extract_tone_amplitudes(data[:, 1], config.sample_rate, tones)
    _validate_multitone_channels(
        data[:, 0], data[:, 1], tone_ch1, tone_ch2, tones,
        config.sample_rate, delay_samples, similarity,
    )
    tone_correction = tone_ch2 / tone_ch1
    delay_phase = -2.0 * np.pi * tones * delay_samples / config.sample_rate
    residual_phase = np.unwrap(np.angle(tone_correction) - delay_phase)
    gain_db = 20.0 * np.log10(np.abs(tone_correction))
    frequency = np.geomspace(config.band.low, config.band.high, config.points)
    log_grid = np.log(frequency)
    gain = np.interp(log_grid, np.log(tones), gain_db)
    residual = np.interp(log_grid, np.log(tones), residual_phase)
    delay = -2.0 * np.pi * frequency * delay_samples / config.sample_rate
    correction = np.power(10.0, gain / 20.0) * np.exp(1j * (residual + delay))
    validate_channel_correction(correction)
    return ChannelCalibration(frequency, correction.astype(np.complex128))


def estimate_reference_resistor(
    recording: ASignal,
    config: ImpedanceConfig,
    channel_correction: NDArray[np.complex128],
) -> ReferenceCalibration:
    frequency, measured = _calculate_smoothed_transfer(recording, config)
    correction = np.asarray(channel_correction, dtype=np.complex128)
    if correction.shape != measured.shape:
        raise ValueError("Channel correction does not match calibration settings")
    valid = (
        (np.abs(measured) >= 1e-12)
        & (np.abs(correction) >= 1e-12)
        & np.isfinite(correction.real)
        & np.isfinite(correction.imag)
        & np.isfinite(measured.real)
        & np.isfinite(measured.imag)
    )
    transfer = np.full(measured.shape, np.nan + 1j * np.nan)
    transfer[valid] = measured[valid] / correction[valid]
    reference = np.full(transfer.shape, np.nan + 1j * np.nan)
    valid &= np.abs(transfer) >= 1e-12
    reference[valid] = (
        config.calibration_resistor * (1.0 - transfer[valid]) / transfer[valid]
    )
    diagnostics = _reference_diagnostics(reference, config)
    estimated_value = diagnostics["rr_estimated"]
    if not isinstance(estimated_value, (int, float)):
        raise ValueError("Invalid reference-resistor estimate")
    estimated = float(estimated_value)
    impedance = calculate_calibration_impedance(
        reference, estimated, config.calibration_resistor
    )
    return ReferenceCalibration(frequency, impedance, estimated, diagnostics)


def require_valid_reference_calibration(diagnostics: dict[str, object]) -> None:
    raw_messages = diagnostics.get("fatal_warnings", [])
    messages = list(raw_messages) if isinstance(raw_messages, list) else []
    if not messages:
        return
    details = "; ".join(str(message) for message in messages)

    def number(key: str) -> float:
        value = diagnostics[key]
        if not isinstance(value, (int, float)):
            raise ValueError(f"Invalid calibration diagnostic: {key}")
        return float(value)

    raise ValueError(
        "Calibration failed: the measured resistor network is invalid "
        f"({details}). Estimated Rref: {number('rr_estimated'):.4g} "
        f"Ohm; nominal: {number('rr_nominal'):.4g} Ohm; "
        f"variation: {number('rr_real_cv'):.1%}; reactive ratio: "
        f"{number('rr_imag_to_real_ratio'):.1%}. Entered Rc/Rr: "
        f"{number('rc_rr_entered'):.4g}; measured Rc/Rr: "
        f"{number('rc_rr_measured'):.4g}; difference: "
        f"{number('rc_rr_error_rel'):.1%}."
    )


def calculate_calibration_impedance(
    reference_by_frequency: NDArray[np.complex128],
    reference_resistor: float,
    calibration_resistor: float,
) -> NDArray[np.complex128]:
    reference = np.asarray(reference_by_frequency, dtype=np.complex128)
    impedance = np.full(reference.shape, np.nan + 1j * np.nan)
    valid = (
        np.isfinite(reference.real)
        & np.isfinite(reference.imag)
        & (np.abs(reference) >= 1e-12)
    )
    impedance[valid] = (
        reference_resistor * calibration_resistor / reference[valid]
    )
    return impedance.astype(np.complex128)


def calculate_impedance(
    recording: ASignal,
    config: ImpedanceConfig,
    channel_correction: NDArray[np.complex128],
    reference_resistor: float,
) -> ImpedanceResult:
    if reference_resistor <= 0:
        raise ValueError("Reference resistor must be positive")
    frequency, measured = _calculate_smoothed_transfer(recording, config)
    correction = np.asarray(channel_correction, dtype=np.complex128)
    if correction.shape != measured.shape:
        raise ValueError("Channel correction does not match measurement settings")
    valid = (
        (np.abs(correction) >= 1e-12)
        & np.isfinite(correction.real)
        & np.isfinite(correction.imag)
        & np.isfinite(measured.real)
        & np.isfinite(measured.imag)
    )
    transfer = np.full(measured.shape, np.nan + 1j * np.nan)
    transfer[valid] = measured[valid] / correction[valid]
    valid &= np.abs(1.0 - transfer) >= 1e-12
    impedance = np.full(transfer.shape, np.nan + 1j * np.nan)
    impedance[valid] = reference_resistor * transfer[valid] / (1.0 - transfer[valid])
    return ImpedanceResult(frequency, impedance.astype(np.complex128))


def current_phase_angle(
    impedance: NDArray[np.complex128],
) -> NDArray[np.float64]:
    values = np.asarray(impedance, dtype=np.complex128)
    phase = np.full(values.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(values.real) & np.isfinite(values.imag)
    if np.any(valid):
        phase[valid] = -np.rad2deg(np.unwrap(np.angle(values[valid])))
    return phase


def validate_channel_correction(
    channel_correction: NDArray[np.complex128],
) -> None:
    correction = np.asarray(channel_correction, dtype=np.complex128)
    valid = np.isfinite(correction.real) & np.isfinite(correction.imag)
    if np.count_nonzero(valid) < max(8, int(math.ceil(correction.size * 0.1))):
        raise ValueError("Channel calibration failed: not enough valid points")
    median_gain = float(np.median(np.abs(correction[valid])))
    if not 0.1 <= median_gain <= 10.0:
        raise ValueError("Channel calibration failed: implausible channel gain ratio")


def _calculate_smoothed_transfer(
    recording: ASignal,
    config: ImpedanceConfig,
) -> tuple[NDArray[np.float64], NDArray[np.complex128]]:
    _validate_recording(recording, config)
    data = recording.as_array(np.float64)
    size = data.shape[0]
    ch1 = data[:, 0] - np.mean(data[:, 0])
    ch2 = data[:, 1] - np.mean(data[:, 1])
    frequency = np.fft.rfftfreq(size, 1.0 / config.sample_rate)
    v1 = np.fft.rfft(ch1)
    v2 = np.fft.rfft(ch2)
    transfer = np.full(v1.shape, np.nan + 1j * np.nan)
    valid = np.abs(v1) >= 1e-12
    transfer[valid] = v2[valid] / v1[valid]
    output_frequency, output = log_smooth(
        frequency,
        transfer,
        band=config.band.as_tuple(),
        window=config.window,
        width=config.window_width,
        points=config.points,
    )
    return output_frequency, np.asarray(output, dtype=np.complex128)


def _reference_diagnostics(
    reference: NDArray[np.complex128],
    config: ImpedanceConfig,
) -> dict[str, object]:
    finite = np.isfinite(reference.real) & np.isfinite(reference.imag)
    if not np.any(finite):
        raise ValueError("Reference calibration failed: no valid frequency points")
    ratios = np.full(reference.shape, np.inf)
    ratios[finite] = np.abs(reference.imag[finite]) / np.maximum(
        np.abs(reference.real[finite]), 1e-12
    )
    resistive = finite & (reference.real > 0) & (
        ratios <= REFERENCE_RESISTIVE_IMAG_RATIO_MAX
    )
    minimum = max(8, int(math.ceil(config.points * 0.1)))
    selected = resistive if np.count_nonzero(resistive) >= minimum else finite
    real = reference.real[selected]
    imag = reference.imag[selected]
    estimated = float(np.median(real))
    cv = float(np.std(real) / max(abs(float(np.mean(real))), 1e-12))
    reactive = float(np.median(np.abs(imag)) / max(abs(estimated), 1e-12))
    entered_ratio = config.calibration_resistor / config.reference_resistor
    measured_ratio = (
        config.calibration_resistor / estimated if estimated > 0 else math.inf
    )
    ratio_error = abs(measured_ratio - entered_ratio) / entered_ratio
    diagnostic_warnings: list[str] = []
    fatal_warnings: list[str] = []
    if estimated <= 0:
        fatal_warnings.append("estimated Rref is not positive")
    if np.count_nonzero(resistive) < minimum:
        fatal_warnings.append("not enough resistive frequency points")
    if reactive > REFERENCE_RESISTIVE_IMAG_RATIO_MAX:
        fatal_warnings.append("Rref has a significant reactive component")
    if cv > 0.03:
        diagnostic_warnings.append("Rref varies across the frequency band")
    if ratio_error > CALIBRATION_RATIO_ERROR_MAX + 1e-12:
        fatal_warnings.append("measured Rc/Rr differs from the entered ratio by more than 5%")
    for message in (*diagnostic_warnings, *fatal_warnings):
        warnings.warn(message, RuntimeWarning, stacklevel=3)
    return {
        "rr_estimated": estimated,
        "rr_nominal": config.reference_resistor,
        "rr_real_cv": cv,
        "rr_imag_to_real_ratio": reactive,
        "rc_rr_entered": entered_ratio,
        "rc_rr_measured": measured_ratio,
        "rc_rr_error_rel": ratio_error,
        "valid_points_count": int(np.count_nonzero(finite)),
        "resistive_points_count": int(np.count_nonzero(resistive)),
        "warnings": diagnostic_warnings,
        "fatal_warnings": fatal_warnings,
    }


def _extract_tone_amplitudes(
    signal: NDArray[np.float64],
    sample_rate: int,
    frequencies: NDArray[np.float64],
) -> NDArray[np.complex128]:
    values = np.asarray(signal, dtype=np.float64).reshape(-1)
    values -= np.mean(values)
    window = np.hanning(values.size)
    normalization = float(np.sum(window))
    time = np.arange(values.size, dtype=np.float64) / sample_rate
    return np.asarray(
        [
            2.0
            * np.sum(values * window * np.exp(-2j * np.pi * frequency * time))
            / normalization
            for frequency in frequencies
        ],
        dtype=np.complex128,
    )


def _estimate_channel_delay(
    ch1: NDArray[np.float64],
    ch2: NDArray[np.float64],
    sample_rate: int,
) -> tuple[int, float]:
    size = min(ch1.size, ch2.size)
    x1 = ch1[:size] - np.mean(ch1[:size])
    x2 = ch2[:size] - np.mean(ch2[:size])
    norm = float(np.linalg.norm(x1) * np.linalg.norm(x2))
    if norm <= 1e-12:
        raise ValueError("Channel calibration failed: an input signal is empty")
    correlation = correlate(x2, x1, mode="full", method="fft")
    center = size - 1
    maximum = min(size - 1, int(round(sample_rate * CHANNEL_SIMILARITY_MAX_DELAY_SECONDS)))
    active = correlation[center - maximum : center + maximum + 1]
    peak = int(np.argmax(np.abs(active)))
    return peak - maximum, min(float(np.abs(active[peak]) / norm), 1.0)


def _validate_multitone_channels(
    ch1: NDArray[np.float64],
    ch2: NDArray[np.float64],
    tone_ch1: NDArray[np.complex128],
    tone_ch2: NDArray[np.complex128],
    frequencies: NDArray[np.float64],
    sample_rate: int,
    delay_samples: int,
    similarity: float,
) -> None:
    if np.any(np.abs(tone_ch1) < 1e-9) or np.any(np.abs(tone_ch2) < 1e-9):
        raise ValueError("Channel calibration failed: calibration tones are missing")
    signal_rms = tuple(float(np.sqrt(np.mean(np.square(ch)))) for ch in (ch1, ch2))
    tone_rms = tuple(float(np.sqrt(np.sum(np.abs(ch) ** 2) / 2.0)) for ch in (tone_ch1, tone_ch2))
    if min(tone / max(total, 1e-12) for tone, total in zip(tone_rms, signal_rms)) < CHANNEL_TONE_ENERGY_RATIO_MIN:
        raise ValueError("Channel calibration failed: recording is not the generated multitone signal")
    correction = tone_ch2 / tone_ch1
    gain = 20.0 * np.log10(np.abs(correction))
    profile = gain - np.median(gain)
    if np.std(profile) > CHANNEL_GAIN_PROFILE_STD_MAX_DB or np.max(np.abs(profile)) > CHANNEL_GAIN_PROFILE_PEAK_MAX_DB:
        raise ValueError("Channel calibration failed: channels have different level profiles")
    delay_phase = -2.0 * np.pi * frequencies * delay_samples / sample_rate
    residual = np.unwrap(np.angle(correction) - delay_phase)
    residual -= np.median(residual)
    phase_rms = float(np.sqrt(np.mean(np.square(residual))) * 180.0 / np.pi)
    if phase_rms > CHANNEL_PHASE_RESIDUAL_RMS_MAX_DEG:
        raise ValueError("Channel calibration failed: incompatible channel phase responses")
    if similarity < CHANNEL_SIMILARITY_THRESHOLD:
        raise ValueError("Channel calibration failed: channels contain different signals")


def _validate_recording(recording: ASignal, config: ImpedanceConfig) -> None:
    _require_stereo(recording)
    config.validate()
    if recording.sample_rate != config.sample_rate:
        raise ValueError("Recording sample rate does not match impedance settings")
    if recording.sample_count < 2:
        raise ValueError("At least two samples are required")


def _require_stereo(recording: ASignal) -> None:
    if not isinstance(recording, ASignal):
        raise TypeError("Impedance analysis requires ASignal")
    if recording.channel_count < 2:
        raise ValueError("Impedance analysis requires two input channels")
