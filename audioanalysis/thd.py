from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import NDArray
from scipy.signal import ShortTimeFFT

from .generators import extend_log_sweep_band
from .types import ASignal, FrequencyBand


THD_SWEEP_AMPLITUDE = 0.9


@dataclass(frozen=True)
class THDResult:
    """Conventional harmonic estimate calculated from one magnitude spectrum."""

    fundamental_frequency: float
    fundamental_rms: float
    harmonic_rms: NDArray[np.float64]
    thd: float


@dataclass(frozen=True)
class SemiAnalogTHDConfig:
    """Settings for the swept-sine residual/total THD+N method."""

    sample_rate: int = 96_000
    duration: float = 30.0
    band: FrequencyBand = FrequencyBand()
    smoothing_octaves: float = 0.1
    segment_seconds: float = 1.0
    overlap: float = 0.9
    fade_in_seconds: float = 0.5
    fade_out_seconds: float = 0.5
    notch_ratio: float = 1.5
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
        if self.fade_in_seconds < 0.0 or self.fade_out_seconds < 0.0:
            raise ValueError("Sweep fades must not be negative")
        if self.notch_ratio <= 1.0:
            raise ValueError("Rejection window ratio must be greater than one")
        if self.points < 3:
            raise ValueError("Point count must be at least three")
        if self.sweep_band[1] >= self.sample_rate / 2.0:
            raise ValueError("Sample rate leaves no room for the requested fade-out")

    @property
    def sample_count(self) -> int:
        return int(round(self.total_duration * self.sample_rate))

    @property
    def total_duration(self) -> float:
        return self.fade_in_seconds + self.duration + self.fade_out_seconds

    @property
    def segment_size(self) -> int:
        return int(round(self.segment_seconds * self.sample_rate))

    @property
    def hop_size(self) -> int:
        return int(round(self.segment_size * (1.0 - self.overlap)))

    @property
    def sweep_band(self) -> tuple[float, float]:
        return extend_log_sweep_band(
            self.band,
            self.duration,
            self.fade_in_seconds,
            self.fade_out_seconds,
        ).as_tuple()


@dataclass(frozen=True)
class SemiAnalogTHDResult:
    frequency: NDArray[np.float64]
    ratio: NDArray[np.float64]
    integrated_ratio: float
    tracked_time: NDArray[np.float64]
    tracked_frequency: NDArray[np.float64]
    total_energy: NDArray[np.float64]
    residual_energy: NDArray[np.float64]

    @property
    def percent(self) -> NDArray[np.float64]:
        return self.ratio * 100.0

    @property
    def integrated_percent(self) -> float:
        return self.integrated_ratio * 100.0


@dataclass(frozen=True)
class _FrameEnergy:
    time: NDArray[np.float64]
    center_frequency: NDArray[np.float64]
    total: NDArray[np.float64]
    residual: NDArray[np.float64]


def generate_semi_analog_thd_sweep(config: SemiAnalogTHDConfig) -> ASignal:
    """Generate the fixed-level logarithmic sweep used by the THD+N method."""
    config.validate()
    start, stop = config.sweep_band
    time = np.arange(config.sample_count, dtype=np.float64) / config.sample_rate
    sweep_rate = config.duration / math.log(config.band.high / config.band.low)
    phase = 2.0 * np.pi * start * sweep_rate * (
        np.exp(time / sweep_rate) - 1.0
    )
    signal = ASignal(THD_SWEEP_AMPLITUDE * np.sin(phase), config.sample_rate)
    fade_in = int(round(config.fade_in_seconds * config.sample_rate))
    fade_out = int(round(config.fade_out_seconds * config.sample_rate))
    return signal.fade(in_=fade_in, out=fade_out).normalize(THD_SWEEP_AMPLITUDE)


def fundamental_rejection_response(
    frequency: NDArray[np.floating],
    center_frequency: float,
    ratio: float = 1.5,
) -> NDArray[np.float64]:
    """Return a flat rejection window from ``f0 / ratio`` to ``f0 * ratio``."""
    frequency = np.asarray(frequency, dtype=np.float64)
    if frequency.ndim != 1:
        raise ValueError("Filter frequencies must be one-dimensional")
    if np.any(frequency < 0.0):
        raise ValueError("Filter frequencies must be non-negative")
    if center_frequency <= 0.0:
        raise ValueError("Center frequency must be positive")
    if ratio <= 1.0:
        raise ValueError("Rejection window ratio must be greater than one")
    response = np.ones_like(frequency)
    rejected = (frequency >= center_frequency / ratio) & (
        frequency <= center_frequency * ratio
    )
    response[rejected] = 0.0
    return response


def analyze_semi_analog_thd(
    recording: ASignal,
    config: SemiAnalogTHDConfig,
) -> SemiAnalogTHDResult:
    """Calculate IEC residual/total THD+N from logical input channel A."""
    config.validate()
    signal = _mono_data(recording, config)
    frames = _analyze_frames(signal, config)
    selected = (
        (frames.center_frequency >= config.band.low)
        & (frames.center_frequency <= config.band.high)
        & (frames.total > np.finfo(np.float64).tiny)
    )
    if np.count_nonzero(selected) < 2:
        raise ValueError("Not enough valid sweep frames in the analysis band")

    tracked_frequency = frames.center_frequency[selected]
    total_energy = frames.total[selected]
    residual_energy = frames.residual[selected]
    raw_ratio = np.sqrt(residual_energy / total_energy)
    frequency, ratio = _smooth_log_ratio(
        tracked_frequency,
        raw_ratio,
        config,
    )
    integrated_ratio = math.sqrt(
        float(np.sum(residual_energy)) / float(np.sum(total_energy))
    )
    return SemiAnalogTHDResult(
        frequency=frequency,
        ratio=ratio,
        integrated_ratio=integrated_ratio,
        tracked_time=frames.time[selected],
        tracked_frequency=tracked_frequency,
        total_energy=total_energy,
        residual_energy=residual_energy,
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


def _analyze_frames(
    signal: NDArray[np.float64],
    config: SemiAnalogTHDConfig,
) -> _FrameEnergy:
    transform = ShortTimeFFT(
        np.hanning(config.segment_size),
        hop=config.hop_size,
        fs=config.sample_rate,
        fft_mode="onesided",
        mfft=config.segment_size,
    )
    spectrum = np.asarray(transform.stft(signal).T)
    frequency = np.asarray(transform.f, dtype=np.float64)
    time = np.asarray(transform.t(len(signal)), dtype=np.float64)
    complete = (
        (time >= config.segment_seconds / 2.0)
        & (time <= len(signal) / config.sample_rate - config.segment_seconds / 2.0)
    )
    frame_indices = np.flatnonzero(complete)
    if not len(frame_indices):
        raise ValueError("The recording contains no complete STFT frames")
    selected_spectrum = spectrum[frame_indices]
    power = np.asarray(np.square(np.abs(selected_spectrum)), dtype=np.float64)
    peak_indices = 1 + np.argmax(power[:, 1:], axis=1)
    centers = frequency[peak_indices]

    # Restore the energy represented by the omitted negative-frequency bins.
    parseval_weight = np.full(len(frequency), 2.0, dtype=np.float64)
    parseval_weight[0] = 0.0
    if np.isclose(frequency[-1], config.sample_rate / 2.0):
        parseval_weight[-1] = 0.0
    power *= parseval_weight[None, :]
    total = np.sum(power, axis=1)
    residual = np.empty(len(centers), dtype=np.float64)
    for index, center in enumerate(centers):
        response = fundamental_rejection_response(
            frequency,
            float(center),
            config.notch_ratio,
        )
        residual[index] = float(np.sum(power[index] * np.square(response)))
    return _FrameEnergy(time[frame_indices], centers, total, residual)


def _smooth_log_ratio(
    frequency: NDArray[np.float64],
    ratio: NDArray[np.float64],
    config: SemiAnalogTHDConfig,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    order = np.argsort(frequency)
    sorted_frequency = frequency[order]
    sorted_power = np.square(ratio[order])
    output = np.geomspace(config.band.low, config.band.high, config.points)
    log_frequency = np.log2(sorted_frequency)
    log_output = np.log2(output)
    half_width = config.smoothing_octaves / 2.0
    smoothed_power = np.empty_like(output)
    for index, center in enumerate(log_output):
        left = int(np.searchsorted(log_frequency, center - half_width))
        right = int(np.searchsorted(log_frequency, center + half_width))
        if left == right:
            nearest = int(np.argmin(np.abs(log_frequency - center)))
            smoothed_power[index] = sorted_power[nearest]
        else:
            smoothed_power[index] = float(np.mean(sorted_power[left:right]))
    return output, np.sqrt(smoothed_power)


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
