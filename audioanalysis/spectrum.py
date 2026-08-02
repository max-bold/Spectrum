from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import NDArray
from scipy.signal import periodogram, welch

from .smoothing import SmoothingWindow, log_smooth
from .types import ASignal, FrequencyBand


class AnalysisMethod(str, Enum):
    PERIODOGRAM = "periodogram"
    WELCH = "welch"


class ReferenceMode(str, Enum):
    NONE = "none"
    CHANNEL_B = "channel_b"


@dataclass(frozen=True)
class SpectrumConfig:
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
    signal: ASignal,
    config: SpectrumConfig,
) -> SpectrumResult:
    """Analyze an audio signal and return smoothed logarithmic power data."""
    _validate_signal(signal)
    config.band.validate(nyquist=signal.sample_rate / 2)

    frequency, spectrum = calculate_power_spectrum(
        signal,
        method=config.method,
        welch_samples=config.welch_samples,
    )
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


def calculate_power_spectrum(
    signal: ASignal,
    *,
    method: AnalysisMethod = AnalysisMethod.PERIODOGRAM,
    welch_samples: int = 8192,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Calculate a periodogram or Welch PSD for every signal channel."""
    _validate_signal(signal)
    data = signal.as_array(np.float64)
    if method == AnalysisMethod.PERIODOGRAM:
        frequency, values = periodogram(data, signal.sample_rate, axis=0)
    elif method == AnalysisMethod.WELCH:
        samples = min(max(1, int(welch_samples)), signal.sample_count)
        frequency, values = welch(
            data,
            signal.sample_rate,
            window="hann",
            nperseg=samples,
            axis=0,
        )
    else:
        raise ValueError(f"Unknown analysis method: {method}")
    return (
        np.asarray(frequency, dtype=np.float64),
        np.asarray(values, dtype=np.float64),
    )


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


def _validate_signal(signal: ASignal) -> None:
    if not isinstance(signal, ASignal):
        raise TypeError("Spectrum analysis requires ASignal")
    if signal.sample_count == 0:
        raise ValueError("Signal must contain samples")


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
