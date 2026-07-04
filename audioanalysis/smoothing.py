from __future__ import annotations

from enum import Enum

import numpy as np
from numpy.typing import NDArray


class SmoothingWindow(str, Enum):
    FLAT = "flat"
    COSINE = "cosine"
    GAUSSIAN = "gaussian"
    TRIANGULAR = "triangular"

    @classmethod
    def list(cls) -> list[str]:
        return [item.value for item in cls]


def log_window(
    window: SmoothingWindow,
    center_frequency: float,
    frequency_step: float,
    width: float,
) -> tuple[NDArray[np.float64], int, int]:
    """Build a logarithmic smoothing window centered at a frequency."""
    if center_frequency <= 0:
        raise ValueError("Center frequency must be positive")
    if frequency_step <= 0:
        raise ValueError("Frequency step must be positive")
    if width <= 0:
        raise ValueError("Window width must be positive")

    half_width = width / 2.0
    low = center_frequency / (2.0**half_width)
    high = center_frequency * (2.0**half_width)
    frequencies = np.arange(low, high, frequency_step)
    log_frequencies = np.log2(frequencies) - np.log2(center_frequency)

    if window == SmoothingWindow.FLAT:
        weights = np.ones_like(log_frequencies)
    elif window == SmoothingWindow.GAUSSIAN:
        weights = np.exp(-((log_frequencies / half_width * 4.0) ** 2) / 2.0)
    elif window == SmoothingWindow.COSINE:
        weights = np.cos(np.pi * log_frequencies / half_width) / 2.0 + 0.5
    elif window == SmoothingWindow.TRIANGULAR:
        decay = 10.0 ** (-30.0 / half_width / 10.0)
        weights = np.power(decay, np.abs(log_frequencies))
    else:
        raise ValueError(f"Unknown smoothing window: {window}")

    start_index = int(np.rint(low / frequency_step))
    end_index = start_index + len(weights)
    return weights.astype(np.float64, copy=False), start_index, end_index


def log_smooth(
    frequency: NDArray[np.floating],
    values: NDArray[np.number],
    *,
    band: tuple[float, float] = (20.0, 20_000.0),
    window: SmoothingWindow = SmoothingWindow.GAUSSIAN,
    width: float = 1 / 3,
    points: int = 256,
) -> tuple[NDArray[np.float64], NDArray[np.number]]:
    """Smooth linearly spaced spectrum values onto a logarithmic frequency axis."""
    if points <= 0:
        raise ValueError("Point count must be positive")
    grid = np.geomspace(float(band[0]), float(band[1]), points)
    return grid, grid_smooth(frequency, values, grid, window=window, width=width)


def grid_smooth(
    frequency: NDArray[np.floating],
    values: NDArray[np.number],
    grid: NDArray[np.floating],
    *,
    window: SmoothingWindow = SmoothingWindow.GAUSSIAN,
    width: float = 1 / 3,
) -> NDArray[np.number]:
    """Smooth spectrum values using logarithmic windows centered on ``grid``."""
    frequency = np.asarray(frequency, dtype=np.float64)
    values = np.asarray(values)
    grid = np.asarray(grid, dtype=np.float64)
    if frequency.ndim != 1:
        raise ValueError("Frequency must be a one-dimensional array")
    if len(frequency) < 2:
        raise ValueError("At least two frequency points are required")
    if values.shape[0] != frequency.shape[0]:
        raise ValueError("Values must have the same first dimension as frequency")
    if np.iscomplexobj(values):
        real = _grid_smooth_real(frequency, np.real(values), grid, window, width)
        imag = _grid_smooth_real(frequency, np.imag(values), grid, window, width)
        return real + 1j * imag
    return _grid_smooth_real(frequency, values, grid, window, width)


def _grid_smooth_real(
    frequency: NDArray[np.float64],
    values: NDArray[np.number],
    grid: NDArray[np.float64],
    window: SmoothingWindow,
    width: float,
) -> NDArray[np.number]:
    dtype = np.result_type(values.dtype, np.float64)
    result = np.full(grid.shape + values.shape[1:], np.nan, dtype=dtype)
    frequency_step = float(frequency[1] - frequency[0])
    for index, center_frequency in enumerate(grid):
        weights, start_index, end_index = log_window(
            window,
            float(center_frequency),
            frequency_step,
            width,
        )
        clipped = _clip_window(start_index, end_index, len(frequency))
        if clipped is None:
            continue
        data_start, data_end, weight_start, weight_end = clipped
        result[index] = np.average(
            values[data_start:data_end],
            axis=0,
            weights=weights[weight_start:weight_end],
        )
    return result


def _clip_window(
    start_index: int,
    end_index: int,
    data_length: int,
) -> tuple[int, int, int, int] | None:
    data_start = max(0, start_index)
    data_end = min(end_index, data_length)
    if data_start >= data_end:
        return None
    weight_start = data_start - start_index
    weight_end = weight_start + (data_end - data_start)
    return data_start, data_end, weight_start, weight_end
