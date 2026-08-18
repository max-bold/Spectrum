from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def normalize_peak(
    signal: ArrayLike,
    *,
    peak: float = 0.9,
    per_channel: bool = False,
    axis: int = 0,
) -> NDArray[np.float32]:
    """Scale a signal to a requested absolute peak level.

    The input is converted to ``float64`` for the calculation and returned as
    ``float32``. Silent or empty input is returned unchanged apart from the dtype
    conversion, because there is no non-zero peak to normalize against. A
    negative ``peak`` value is allowed and inverts polarity while setting the
    absolute peak magnitude to ``abs(peak)``.

    By default, the maximum is measured across the whole array. Set
    ``per_channel=True`` when each channel or sub-array should be normalized
    independently. In that mode, ``axis`` selects the dimension over which peak
    values are measured. For example, use the default ``axis=0`` for
    ``(samples, channels)`` data and ``axis=1`` for ``(channels, samples)``
    data. For one-dimensional input, ``per_channel`` has no effect because the
    signal has only one channel.

    Args:
        signal: Any one- or multi-dimensional numeric array-like signal.
        peak: Target peak scale. Typical audio generators use ``0.9`` to leave
            a little headroom below full scale.
        per_channel: If ``True``, normalize independently across ``axis``.
            Ignored for one-dimensional input.
        axis: Axis over which to measure peaks when ``per_channel=True``.
            Defaults to ``0``.

    Returns:
        A ``float32`` array with the same shape as ``signal``.

    Raises:
        ValueError: If ``signal`` cannot be converted to a numeric array.
        ValueError: If ``per_channel=True`` and ``axis`` is invalid for the
            input shape.
    """
    data = np.asarray(signal, dtype=np.float64)
    if per_channel:
        if data.ndim == 1:
            return normalize_peak(data, peak=peak)
        if not -data.ndim <= axis < data.ndim:
            raise ValueError(
                f"Axis {axis} is out of bounds for signal shape {data.shape}"
            )
        if data.size == 0:
            return data.astype(np.float32)
        maximum = np.max(np.abs(data), axis=axis, keepdims=True)
        scale = np.divide(
            float(peak),
            maximum,
            out=np.zeros_like(maximum),
            where=maximum > 0.0,
        )
        return (data * scale).astype(np.float32)

    maximum = float(np.max(np.abs(data))) if data.size else 0.0
    if maximum <= 0.0:
        return data.astype(np.float32)
    return (data * (float(peak) / maximum)).astype(np.float32)


def as_channels(
    record: ArrayLike,
    *,
    channels: int = 2,
) -> NDArray[np.float32]:
    """Return audio data with a requested number of channels.

    One-dimensional input is treated as mono. Two-dimensional input is treated
    as ``(samples, channels)``. Mono input can be duplicated, extra channels are
    truncated, and missing channels are filled with zeros.

    Args:
        record: One-dimensional mono data or two-dimensional sample/channel
            data.
        channels: Desired channel count. Must be positive.

    Returns:
        A ``float32`` audio array with shape ``(samples, channels)``.

    Raises:
        ValueError: If ``channels`` is less than one or if ``record`` is not
            one- or two-dimensional.
    """
    if channels < 1:
        raise ValueError("Channel count must be positive")
    data = np.asarray(record, dtype=np.float32)
    if data.ndim == 1:
        data = data[:, None]
    if data.ndim != 2:
        raise ValueError("Audio data must be one- or two-dimensional")
    if data.shape[1] == channels:
        result = data
    elif data.shape[1] == 1:
        result = np.repeat(data, channels, axis=1)
    elif data.shape[1] > channels:
        result = data[:, :channels]
    else:
        pad = np.zeros((len(data), channels - data.shape[1]), dtype=np.float32)
        result = np.column_stack((data, pad))

    return result


def peak_levels(
    record: ArrayLike,
    sample_rate: int,
    *,
    time_step: float = 0.01,
    channels: int = 2,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Return per-channel absolute peak levels for fixed time chunks."""
    if sample_rate <= 0:
        raise ValueError("Sample rate must be positive")
    if time_step <= 0:
        raise ValueError("Time step must be positive")
    data = as_channels(record, channels=channels)
    if len(data) == 0:
        return np.empty(0, np.float32), np.empty((channels, 0), np.float32)

    chunk_size = max(1, int(time_step * sample_rate))
    starts = np.arange(0, len(data), chunk_size)
    timestamps = starts.astype(np.float32) / float(sample_rate)
    levels = np.zeros((len(starts), channels), dtype=np.float32)
    for index, start in enumerate(starts):
        chunk = data[start : start + chunk_size]
        levels[index] = np.max(np.abs(chunk), axis=0)
    return timestamps, levels.T
