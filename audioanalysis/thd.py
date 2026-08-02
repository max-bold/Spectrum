from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import curve_fit

from .types import ASignal, FrequencyBand


THD_SWEEP_AMPLITUDE = 0.9
THD_FADE_SECONDS = 0.5
FALLBACK_MASK_RATIO = 5.0 / 4.0


@dataclass(frozen=True)
class THDResult:
    """Conventional harmonic estimate calculated from one magnitude spectrum."""

    fundamental_frequency: float
    fundamental_rms: float
    harmonic_rms: NDArray[np.float64]
    thd: float


@dataclass(frozen=True)
class SemiAnalogTHDConfig:
    """Settings for the moving-notch, swept-sine THD+N method."""

    sample_rate: int = 96_000
    duration: float = 30.0
    band: FrequencyBand = FrequencyBand()
    smoothing_octaves: float = 1.0 / 3.0
    segment_seconds: float = 1.0
    overlap: float = 0.9
    sweep_band_expansion: float = 1.5
    mask_expansion: float = 2.0
    points: int = 1_200

    def validate(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("Sample rate must be positive")
        if self.duration <= 0:
            raise ValueError("Duration must be positive")
        self.band.validate(nyquist=self.sample_rate / 2.0)
        if self.smoothing_octaves <= 0:
            raise ValueError("Smoothing width must be positive")
        if self.segment_seconds <= 0:
            raise ValueError("STFT window duration must be positive")
        if self.segment_seconds > self.duration:
            raise ValueError("STFT window must not be longer than the sweep")
        if not 0.0 <= self.overlap < 1.0:
            raise ValueError("STFT overlap must be in the 0..1 range")
        if self.hop_size < 1:
            raise ValueError("STFT overlap leaves no samples between frames")
        if self.sweep_band_expansion <= 1.0:
            raise ValueError("Sweep band expansion must be greater than one")
        if self.mask_expansion <= 0:
            raise ValueError("Mask expansion must be positive")
        if self.points < 3:
            raise ValueError("Point count must be at least three")
        low, high = self.sweep_band
        if low <= 0 or high <= self.band.high:
            raise ValueError("Sample rate leaves no room above the analysis band")

    @property
    def sample_count(self) -> int:
        return int(round(self.duration * self.sample_rate))

    @property
    def segment_size(self) -> int:
        return int(round(self.segment_seconds * self.sample_rate))

    @property
    def hop_size(self) -> int:
        return int(round(self.segment_size * (1.0 - self.overlap)))

    @property
    def sweep_band(self) -> tuple[float, float]:
        return (
            self.band.low / self.sweep_band_expansion,
            min(
                self.band.high * self.sweep_band_expansion,
                self.sample_rate * 0.49,
            ),
        )


@dataclass(frozen=True)
class THDMaskCalibration:
    center_frequency: NDArray[np.float64]
    left_edge: NDArray[np.float64]
    right_edge: NDArray[np.float64]


@dataclass(frozen=True)
class THDMaskFit:
    left_params: NDArray[np.float64]
    right_params: NDArray[np.float64]
    leakage_ratio: float


@dataclass(frozen=True)
class SemiAnalogTHDResult:
    frequency: NDArray[np.float64]
    ratio: NDArray[np.float64]
    integrated_ratio: float
    tracked_time: NDArray[np.float64]
    tracked_frequency: NDArray[np.float64]
    main_energy: NDArray[np.float64]
    residual_energy: NDArray[np.float64]

    @property
    def percent(self) -> NDArray[np.float64]:
        return self.ratio * 100.0

    @property
    def integrated_percent(self) -> float:
        return self.integrated_ratio * 100.0


@dataclass(frozen=True)
class _EnergySplit:
    frequency: NDArray[np.float64]
    tracked_time: NDArray[np.float64]
    tracked_frequency: NDArray[np.float64]
    main_energy: NDArray[np.float64]
    residual_energy: NDArray[np.float64]


def generate_semi_analog_thd_sweep(config: SemiAnalogTHDConfig) -> ASignal:
    """Generate the fixed-level logarithmic sweep used by the THD+N method."""
    config.validate()
    start, stop = config.sweep_band
    time = np.arange(config.sample_count, dtype=np.float64) / config.sample_rate
    sweep_rate = config.duration / math.log(stop / start)
    phase = 2.0 * np.pi * start * sweep_rate * (
        np.exp(time / sweep_rate) - 1.0
    )
    signal = ASignal(THD_SWEEP_AMPLITUDE * np.sin(phase), config.sample_rate)
    fade = min(
        int(round(THD_FADE_SECONDS * config.sample_rate)),
        config.sample_count // 4,
    )
    return signal.fade(in_=fade, out=fade).normalize(THD_SWEEP_AMPLITUDE)


def calibrate_semi_analog_thd_mask(
    config: SemiAnalogTHDConfig,
    clean_sweep: ASignal | None = None,
) -> THDMaskFit:
    """Fit a delay-independent fundamental mask from a clean digital sweep."""
    config.validate()
    sweep = clean_sweep or generate_semi_analog_thd_sweep(config)
    clean = _mono_data(sweep, config)
    calibration = _calibrate_mask(clean, config)
    left_params = _fit_mask_side(calibration, config, side="left")
    right_params = _fit_mask_side(calibration, config, side="right")
    provisional = THDMaskFit(left_params, right_params, math.nan)
    clean_split = _split_energy(clean, config, provisional)
    leakage = _integrated_ratio(
        clean_split.residual_energy,
        clean_split.main_energy,
    )
    return THDMaskFit(left_params, right_params, leakage)


def analyze_semi_analog_thd(
    recording: ASignal,
    config: SemiAnalogTHDConfig,
    *,
    mask_fit: THDMaskFit | None = None,
) -> SemiAnalogTHDResult:
    """Calculate frequency-resolved THD+N from channel 1 of a sweep recording."""
    config.validate()
    signal = _mono_data(recording, config)
    fitted_mask = mask_fit or calibrate_semi_analog_thd_mask(config)
    split = _split_energy(signal, config, fitted_mask)
    band = (
        (split.frequency >= config.band.low)
        & (split.frequency <= config.band.high)
    )
    raw_ratio = _power_ratio(
        split.residual_energy[band],
        split.main_energy[band],
    )
    frequency, ratio = _smooth_log_ratio(
        split.frequency[band],
        raw_ratio,
        config,
    )
    return SemiAnalogTHDResult(
        frequency=frequency,
        ratio=ratio,
        integrated_ratio=_integrated_ratio(
            split.residual_energy,
            split.main_energy,
        ),
        tracked_time=split.tracked_time,
        tracked_frequency=split.tracked_frequency,
        main_energy=split.main_energy,
        residual_energy=split.residual_energy,
    )


def thd_from_spectrum(
    frequency: NDArray[np.floating],
    magnitude: NDArray[np.floating],
    *,
    fundamental_frequency: float | None = None,
    harmonics: int = 5,
) -> THDResult:
    """Estimate conventional THD from a magnitude spectrum."""
    if harmonics < 2:
        raise ValueError("At least two harmonics are required")
    frequency = np.asarray(frequency, dtype=np.float64)
    magnitude = np.asarray(magnitude, dtype=np.float64)
    if frequency.shape != magnitude.shape:
        raise ValueError("Frequency and magnitude arrays must have equal shapes")
    if fundamental_frequency is None:
        positive = frequency > 0
        if not np.any(positive):
            raise ValueError("Spectrum must contain positive frequencies")
        fundamental_index = np.argmax(magnitude[positive])
        fundamental_frequency = float(frequency[positive][fundamental_index])
    fundamental_rms = _nearest_value(frequency, magnitude, fundamental_frequency)
    harmonic_values = np.array(
        [
            _nearest_value(frequency, magnitude, fundamental_frequency * order)
            for order in range(2, harmonics + 1)
            if fundamental_frequency * order <= frequency[-1]
        ],
        dtype=np.float64,
    )
    thd_value = float(
        np.sqrt(np.sum(harmonic_values * harmonic_values)) / fundamental_rms
    )
    return THDResult(
        fundamental_frequency=float(fundamental_frequency),
        fundamental_rms=float(fundamental_rms),
        harmonic_rms=harmonic_values,
        thd=thd_value,
    )


def _calibrate_mask(
    signal: NDArray[np.float64],
    config: SemiAnalogTHDConfig,
) -> THDMaskCalibration:
    frequency = np.fft.rfftfreq(config.segment_size, 1.0 / config.sample_rate)
    centers: list[float] = []
    left_edges: list[float] = []
    right_edges: list[float] = []
    for _time, power in _power_frames(signal, config):
        peak_index = 1 + int(np.argmax(power[1:]))
        threshold = float(np.mean(power))
        left_index = peak_index
        while left_index > 1 and power[left_index] > threshold:
            left_index -= 1
        right_index = peak_index
        while right_index < len(frequency) - 1 and power[right_index] > threshold:
            right_index += 1

        center = float(frequency[peak_index])
        if not config.band.low / 2.0 <= center <= config.band.high * 2.0:
            continue
        centers.append(center)
        left_edges.append(
            max(
                config.band.low,
                center
                - config.mask_expansion * (center - float(frequency[left_index])),
            )
        )
        right_edges.append(
            min(
                config.band.high,
                center
                + config.mask_expansion * (float(frequency[right_index]) - center),
            )
        )
    if len(centers) < 6:
        raise ValueError("Not enough sweep frames to calibrate the THD mask")
    return THDMaskCalibration(
        np.asarray(centers, dtype=np.float64),
        np.asarray(left_edges, dtype=np.float64),
        np.asarray(right_edges, dtype=np.float64),
    )


def _fit_mask_side(
    calibration: THDMaskCalibration,
    config: SemiAnalogTHDConfig,
    *,
    side: str,
) -> NDArray[np.float64]:
    centers = calibration.center_frequency
    if side == "left":
        ratio = centers / calibration.left_edge
        selected = (centers >= config.band.low * 2.0) & (
            centers <= config.band.high
        )
    else:
        ratio = calibration.right_edge / centers
        selected = (centers >= config.band.low) & (
            centers <= config.band.high / 2.0
        )
    if np.count_nonzero(selected) < 3:
        raise ValueError(f"Not enough sweep frames to fit the {side} THD mask")
    lower = (0.0, 1.0 / config.band.low + 1e-9, 1.0)
    upper = (10.0, 10.0, 3.0)
    params, _ = curve_fit(
        _reciprocal_log_model,
        centers[selected],
        ratio[selected],
        p0=(1.0, 1.0, 1.05),
        bounds=(lower, upper),
        maxfev=20_000,
    )
    return np.asarray(params, dtype=np.float64)


def _split_energy(
    signal: NDArray[np.float64],
    config: SemiAnalogTHDConfig,
    mask_fit: THDMaskFit,
) -> _EnergySplit:
    frequency = np.fft.rfftfreq(config.segment_size, 1.0 / config.sample_rate)
    band = (frequency >= config.band.low) & (frequency <= config.band.high)
    main = np.zeros_like(frequency)
    residual = np.zeros_like(frequency)
    times: list[float] = []
    centers: list[float] = []
    for frame_time, power in _power_frames(signal, config):
        peak_index = 1 + int(np.argmax(power[1:]))
        center = float(frequency[peak_index])
        left, right = _fitted_mask_edges(center, mask_fit, config)
        main_mask = band & (frequency >= left) & (frequency <= right)
        main[main_mask] += power[main_mask]
        residual[band & ~main_mask] += power[band & ~main_mask]
        times.append(frame_time)
        centers.append(center)
    return _EnergySplit(
        frequency,
        np.asarray(times, dtype=np.float64),
        np.asarray(centers, dtype=np.float64),
        main,
        residual,
    )


def _power_frames(
    signal: NDArray[np.float64],
    config: SemiAnalogTHDConfig,
) -> Iterator[tuple[float, NDArray[np.float64]]]:
    """Yield centered, zero-padded Hann-window power spectra one frame at a time."""
    size = config.segment_size
    hop = config.hop_size
    before = size // 2
    after = size - before
    padded = np.pad(signal, (before, after))
    window = np.hanning(size)
    for start in range(0, len(padded) - size + 1, hop):
        frame = padded[start : start + size]
        power = np.square(np.abs(np.fft.rfft(frame * window)))
        center_sample = start + size / 2.0 - before
        yield center_sample / config.sample_rate, power


def _smooth_log_ratio(
    frequency: NDArray[np.float64],
    ratio: NDArray[np.float64],
    config: SemiAnalogTHDConfig,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    output = np.geomspace(config.band.low, config.band.high, config.points)
    log_frequency = np.log2(frequency)
    half_width = config.smoothing_octaves / 2.0
    power = np.square(ratio)
    smoothed = np.full(output.shape, np.nan, dtype=np.float64)
    for index, center in enumerate(output):
        selected = (
            (log_frequency >= math.log2(center) - half_width)
            & (log_frequency <= math.log2(center) + half_width)
            & np.isfinite(power)
        )
        if np.any(selected):
            smoothed[index] = math.sqrt(float(np.mean(power[selected])))
    return output, smoothed


def _fitted_mask_edges(
    frequency: float,
    mask_fit: THDMaskFit,
    config: SemiAnalogTHDConfig,
) -> tuple[float, float]:
    if frequency <= 0:
        return config.band.low, config.band.low
    left_ratio = float(_reciprocal_log_model(frequency, *mask_fit.left_params))
    right_ratio = float(_reciprocal_log_model(frequency, *mask_fit.right_params))
    if not np.isfinite(left_ratio) or left_ratio <= 0:
        left_ratio = FALLBACK_MASK_RATIO
    if not np.isfinite(right_ratio) or right_ratio <= 0:
        right_ratio = FALLBACK_MASK_RATIO
    return (
        max(config.band.low, frequency / left_ratio),
        min(config.band.high, frequency * right_ratio),
    )


def _reciprocal_log_model(
    frequency: float | NDArray[np.float64],
    a: float,
    b: float,
    c: float,
) -> NDArray[np.float64]:
    values = np.asarray(frequency, dtype=np.float64)
    return a / np.log(b * values) + c


def _power_ratio(
    residual_energy: NDArray[np.float64],
    main_energy: NDArray[np.float64],
) -> NDArray[np.float64]:
    ratio = np.full(main_energy.shape, np.nan, dtype=np.float64)
    valid = main_energy > np.finfo(np.float64).tiny
    ratio[valid] = np.sqrt(residual_energy[valid] / main_energy[valid])
    return ratio


def _integrated_ratio(
    residual_energy: NDArray[np.float64],
    main_energy: NDArray[np.float64],
) -> float:
    main = float(np.sum(main_energy))
    if main <= np.finfo(np.float64).tiny:
        raise ValueError("The recording contains no tracked fundamental energy")
    return math.sqrt(float(np.sum(residual_energy)) / main)


def _mono_data(signal: ASignal, config: SemiAnalogTHDConfig) -> NDArray[np.float64]:
    if not isinstance(signal, ASignal):
        raise TypeError("Semi-analog THD analysis requires ASignal")
    if signal.sample_rate != config.sample_rate:
        raise ValueError("Signal sample rate does not match THD settings")
    if signal.sample_count < config.segment_size:
        raise ValueError("Signal is shorter than one STFT window")
    return signal.as_array(np.float64)[:, 0]


def _nearest_value(
    frequency: NDArray[np.float64],
    values: NDArray[np.float64],
    target: float,
) -> float:
    index = int(np.argmin(np.abs(frequency - target)))
    return float(abs(values[index]))
