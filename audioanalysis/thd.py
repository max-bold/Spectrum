from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class THDResult:
    fundamental_frequency: float
    fundamental_rms: float
    harmonic_rms: NDArray[np.float64]
    thd: float


def thd_from_spectrum(
    frequency: NDArray[np.floating],
    magnitude: NDArray[np.floating],
    *,
    fundamental_frequency: float | None = None,
    harmonics: int = 5,
) -> THDResult:
    """Estimate THD from a magnitude spectrum."""
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
    thd_value = float(np.sqrt(np.sum(harmonic_values * harmonic_values)) / fundamental_rms)
    return THDResult(
        fundamental_frequency=float(fundamental_frequency),
        fundamental_rms=float(fundamental_rms),
        harmonic_rms=harmonic_values,
        thd=thd_value,
    )


def _nearest_value(
    frequency: NDArray[np.float64],
    values: NDArray[np.float64],
    target: float,
) -> float:
    index = int(np.argmin(np.abs(frequency - target)))
    return float(abs(values[index]))
