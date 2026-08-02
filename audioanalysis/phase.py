from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter1d


def wrap_phase(phase: NDArray[np.floating]) -> NDArray[np.float64]:
    """Wrap phase angles to the ``[-180, 180)`` degree range."""
    phase = np.asarray(phase, dtype=np.float64)
    return (phase + 180.0) % 360.0 - 180.0


def break_phase_wraps(
    x: NDArray[np.floating],
    wrapped_phase: NDArray[np.floating],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Insert NaN points so a line plot does not connect phase wraps."""
    x = np.asarray(x, dtype=np.float64)
    wrapped_phase = np.asarray(wrapped_phase, dtype=np.float64)
    if x.ndim != 1 or wrapped_phase.ndim != 1:
        raise ValueError("X and phase must be one-dimensional arrays")
    if x.shape != wrapped_phase.shape:
        raise ValueError("X and phase arrays must have equal shapes")
    if len(x) < 2:
        return x.copy(), wrapped_phase.copy()

    finite_pairs = np.isfinite(wrapped_phase[:-1]) & np.isfinite(
        wrapped_phase[1:]
    )
    wrap_indices = np.flatnonzero(
        finite_pairs & (np.abs(np.diff(wrapped_phase)) > 180.0)
    )
    insert_at = wrap_indices + 1
    return (
        np.insert(x, insert_at, np.nan),
        np.insert(wrapped_phase, insert_at, np.nan),
    )


def phase_derivative(
    frequency: NDArray[np.floating],
    phase: NDArray[np.floating],
    *,
    smoothing_sigma: float = 2.0,
) -> NDArray[np.float64]:
    """Calculate phase derivative in degrees per frequency decade."""
    frequency = np.asarray(frequency, dtype=np.float64)
    phase = np.asarray(phase, dtype=np.float64)
    if frequency.ndim != 1 or phase.ndim != 1:
        raise ValueError("Frequency and phase must be one-dimensional arrays")
    if frequency.shape != phase.shape:
        raise ValueError("Frequency and phase arrays must have equal shapes")
    if smoothing_sigma < 0:
        raise ValueError("Smoothing sigma must not be negative")

    result = np.full(frequency.shape, np.nan, dtype=np.float64)
    valid = (frequency > 0) & np.isfinite(frequency) & np.isfinite(phase)
    if np.count_nonzero(valid) < 3:
        return result

    valid_frequency = frequency[valid]
    if np.any(np.diff(valid_frequency) <= 0):
        raise ValueError("Frequency values must be strictly increasing")

    valid_phase = phase[valid]
    if smoothing_sigma > 0:
        valid_phase = gaussian_filter1d(
            valid_phase,
            sigma=smoothing_sigma,
            mode="nearest",
        )
    result[valid] = np.gradient(
        valid_phase,
        np.log10(valid_frequency),
        edge_order=2,
    )
    return result
