from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.signal import periodogram

from .smoothing import SmoothingWindow, log_smooth
from .spectrum import power_db
from .types import ASignal, FrequencyBand

RTAWindow = Literal["hann", "blackman", "boxcar"]


@dataclass(frozen=True)
class RTAConfig:
    band: FrequencyBand = FrequencyBand()
    points: int = 31
    smoothing_width: float = 1.0 / 3.0
    smoothing_window: SmoothingWindow = SmoothingWindow.GAUSSIAN
    fft_window: RTAWindow = "hann"

    def validate(self, sample_rate: int) -> None:
        self.band.validate(nyquist=sample_rate / 2.0)
        if self.points < 2:
            raise ValueError("RTA point count must be at least two")
        if self.smoothing_width <= 0.0:
            raise ValueError("RTA smoothing width must be positive")
        if self.fft_window not in ("hann", "blackman", "boxcar"):
            raise ValueError(f"Unknown RTA FFT window: {self.fft_window}")


@dataclass(frozen=True)
class RTAResult:
    frequency: NDArray[np.float64]
    level_db: NDArray[np.float64]


def compensate_log_band_density(
    frequency: NDArray[np.floating],
    density: NDArray[np.floating],
    *,
    reference_frequency: float = 1_000.0,
) -> NDArray[np.float64]:
    """Convert spectral density to relative equal-log-band power.

    Equal logarithmic bands have bandwidth proportional to their center
    frequency. Multiplication of power density by ``f`` is equivalent to the
    ``sqrt(f)`` correction applied to an amplitude spectrum.
    """
    frequency = np.asarray(frequency, dtype=np.float64)
    density = np.asarray(density, dtype=np.float64)
    if frequency.ndim != 1:
        raise ValueError("RTA frequency must be one-dimensional")
    if density.shape[0] != len(frequency):
        raise ValueError("RTA frequency and density lengths must match")
    if reference_frequency <= 0.0:
        raise ValueError("RTA reference frequency must be positive")
    scale = frequency / float(reference_frequency)
    if density.ndim > 1:
        scale = scale.reshape((-1,) + (1,) * (density.ndim - 1))
    return np.asarray(density * scale, dtype=np.float64)


def analyze_rta(signal: ASignal, config: RTAConfig) -> RTAResult:
    """Calculate smoothed periodograms for every input channel."""
    if not isinstance(signal, ASignal):
        raise TypeError("RTA analysis requires ASignal")
    if signal.sample_count < 2:
        raise ValueError("RTA analysis requires at least two samples")
    config.validate(signal.sample_rate)

    frequency, density = periodogram(
        signal.as_array(np.float64),
        signal.sample_rate,
        window=config.fft_window,
        axis=0,
    )
    output_frequency, output_density = log_smooth(
        np.asarray(frequency, dtype=np.float64),
        np.asarray(density, dtype=np.float64),
        band=config.band.as_tuple(),
        window=config.smoothing_window,
        width=config.smoothing_width,
        points=config.points,
    )
    output_density = compensate_log_band_density(
        output_frequency,
        np.asarray(output_density, dtype=np.float64),
    )
    return RTAResult(
        frequency=np.asarray(output_frequency, dtype=np.float64),
        level_db=power_db(np.asarray(output_density, dtype=np.float64)),
    )
