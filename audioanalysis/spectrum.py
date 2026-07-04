from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import NDArray
from scipy.signal import periodogram, welch

from .smoothing import SmoothingWindow, log_smooth
from .types import FrequencyBand


class AnalysisMethod(str, Enum):
    PERIODOGRAM = "periodogram"
    WELCH = "welch"


class ReferenceMode(str, Enum):
    NONE = "none"
    CHANNEL_B = "channel_b"


@dataclass(frozen=True)
class SpectrumConfig:
    sample_rate: int
    method: AnalysisMethod = AnalysisMethod.PERIODOGRAM
    reference: ReferenceMode = ReferenceMode.NONE
    band: FrequencyBand = FrequencyBand()
    points: int = 1024
    window: SmoothingWindow = SmoothingWindow.GAUSSIAN
    window_width: float = 0.1
    welch_samples: int = 8192
    pink_weighting: bool = False


@dataclass(frozen=True)
class SpectrumResult:
    frequency: NDArray[np.float64]
    values: NDArray[np.float64]


def analyze_spectrum(
    record: NDArray[np.floating],
    config: SpectrumConfig,
) -> SpectrumResult:
    """Analyze ``(samples, channels)`` audio and return smoothed log-power data."""
    if config.sample_rate <= 0:
        raise ValueError("Sample rate must be positive")
    config.band.validate(nyquist=config.sample_rate / 2)
    data = np.asarray(record)
    if data.ndim != 2 or data.shape[0] == 0 or data.shape[1] == 0:
        raise ValueError("Record must have shape (samples, channels)")

    frequency, spectrum = _raw_power_spectrum(data, config)
    values = _apply_reference(spectrum, config.reference)
    if config.pink_weighting:
        values = values * frequency
    output_frequency, output_values = log_smooth(
        frequency,
        values,
        band=config.band.as_tuple(),
        window=config.window,
        width=config.window_width,
        points=config.points,
    )
    return SpectrumResult(output_frequency, np.asarray(output_values, dtype=np.float64))


def magnitude_db(
    values: NDArray[np.number],
    *,
    floor: float = 1e-20,
) -> NDArray[np.float64]:
    """Convert magnitude values to decibels."""
    return 20.0 * np.log10(np.abs(values).clip(floor))


def power_db(
    values: NDArray[np.number],
    *,
    floor: float = 1e-20,
) -> NDArray[np.float64]:
    """Convert power values to decibels."""
    return 10.0 * np.log10(np.asarray(values).clip(floor))


def phase_degrees(values: NDArray[np.number]) -> NDArray[np.float64]:
    """Return unwrapped phase in degrees."""
    return np.rad2deg(np.unwrap(np.angle(values)))


def _raw_power_spectrum(
    data: NDArray[np.floating],
    config: SpectrumConfig,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    if config.method == AnalysisMethod.PERIODOGRAM:
        return periodogram(data, config.sample_rate, axis=0)
    if config.method == AnalysisMethod.WELCH:
        samples = min(max(1, int(config.welch_samples)), len(data))
        return welch(data, config.sample_rate, window="hann", nperseg=samples, axis=0)
    raise ValueError(f"Unknown analysis method: {config.method}")


def _apply_reference(
    spectrum: NDArray[np.float64],
    reference: ReferenceMode,
) -> NDArray[np.float64]:
    if reference == ReferenceMode.NONE:
        return spectrum[:, 0]
    if reference == ReferenceMode.CHANNEL_B:
        if spectrum.shape[1] < 2:
            raise ValueError("Reference mode requires at least two channels")
        return spectrum[:, 0] / spectrum[:, 1].clip(1e-20)
    raise ValueError(f"Unknown reference mode: {reference}")
