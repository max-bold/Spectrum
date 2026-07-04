from __future__ import annotations

from dataclasses import dataclass
from typing import overload

import numpy as np
from numpy.typing import ArrayLike, DTypeLike, NDArray


class ASignal:
    """Audio signal stored in canonical ``(samples, channels)`` shape.

    Mono signals use ``(samples, 1)`` and multichannel signals use one column
    per channel.
    """

    @overload
    def __init__(self, data: ArrayLike, sample_rate: int) -> None: ...

    @overload
    def __init__(self, data: tuple[ASignal, ...]) -> None: ...

    def __init__(
        self,
        data: ArrayLike | tuple[ASignal, ...],
        sample_rate: int | None = None,
    ) -> None:
        """Create a signal from an array or combine existing signals.

        Args:
            data: Numeric array-like audio data or a tuple of ``ASignal``
                instances. One-dimensional arrays are interpreted as mono
                boundary input and stored as ``(samples, 1)``. Two-dimensional
                arrays must already be shaped as ``(samples, channels)``. A
                tuple combines channels from multiple signals with equal sample
                counts.
            sample_rate: Sampling rate in hertz for array-like input. Omit when
                combining ``ASignal`` instances; the combined signal uses their
                shared sample rate.

        Raises:
            TypeError: If tuple items are not ``ASignal`` instances.
            ValueError: If array data is not one- or two-dimensional, contains
                no channels, ``sample_rate`` is invalid, tuple signals have
                different sample counts, or tuple signals have different sample
                rates.
        """
        if isinstance(data, tuple) and (
            not data or any(isinstance(item, ASignal) for item in data)
        ):
            self._data, self.sample_rate = self._combine(data)
            return
        if sample_rate is None:
            raise ValueError("Sample rate is required for array-like ASignal data")
        self.sample_rate = self._validate_sample_rate(sample_rate)
        self._data = self._coerce_array(data)

    @property
    def sample_count(self) -> int:
        """Number of samples in the signal."""
        return int(self._data.shape[0])

    @property
    def channel_count(self) -> int:
        """Number of channels in the signal."""
        return int(self._data.shape[1])

    def __getitem__(self, channel: int) -> ASignal:
        """Return one channel as a mono ``ASignal``."""
        if not isinstance(channel, int):
            raise TypeError("Signal channel index must be an integer")
        return ASignal(self._data[:, [channel]], self.sample_rate)

    def to_channels(self, channels: int) -> ASignal:
        """Duplicate a mono signal to the requested number of channels.

        Args:
            channels: Desired output channel count. Must be positive.

        Returns:
            A new ``ASignal`` with duplicated mono data.

        Raises:
            ValueError: If ``channels`` is less than one or this signal is not
                mono.
        """
        if channels < 1:
            raise ValueError("Channel count must be positive")
        if self.channel_count != 1:
            raise ValueError("Only mono ASignal can be converted to channels")
        return ASignal(np.repeat(self._data, channels, axis=1), self.sample_rate)

    def normalize(
        self,
        max: float = 0.9,
        *,
        per_channel: bool = False,
    ) -> ASignal:
        """Normalize signal peak level.

        Args:
            max: Target absolute peak level.
            per_channel: If ``True``, normalize every channel independently.
                Otherwise, normalize the whole signal by its global absolute
                peak.

        Returns:
            A new normalized ``ASignal``. Silent channels stay silent.
        """
        if per_channel:
            peaks = self.max()[None, :]
            scale = np.divide(
                float(max),
                peaks,
                out=np.zeros_like(peaks),
                where=peaks > 0.0,
            )
            return ASignal(self._data * scale, self.sample_rate)

        peak = float(np.max(self.max())) if self.channel_count else 0.0
        if peak <= 0.0:
            return ASignal(self._data, self.sample_rate)
        return ASignal(self._data * (float(max) / peak), self.sample_rate)

    def trim(self, length: int, start: int = 0) -> ASignal:
        """Return a slice of the signal.

        Args:
            length: Number of samples to keep. Must not be negative.
            start: Start sample index. Must not be negative.

        Returns:
            A new ``ASignal`` containing samples ``start:start + length``.

        Raises:
            ValueError: If ``length`` or ``start`` is negative.
        """
        length = int(length)
        start = int(start)
        if length < 0:
            raise ValueError("Trim length must not be negative")
        if start < 0:
            raise ValueError("Trim start must not be negative")
        return ASignal(self._data[start : start + length], self.sample_rate)

    def fade(self, in_: int = 0, out: int = 0) -> ASignal:
        """Apply linear fade-in and fade-out envelopes.

        Args:
            in_: Fade-in length in samples. Must not be negative.
            out: Fade-out length in samples. Must not be negative.

        Returns:
            A new faded ``ASignal``.

        Raises:
            ValueError: If ``in_`` or ``out`` is negative.
        """
        fade_in = int(in_)
        fade_out = int(out)
        if fade_in < 0:
            raise ValueError("Fade-in length must not be negative")
        if fade_out < 0:
            raise ValueError("Fade-out length must not be negative")

        data = self._data.copy()
        if self.sample_count == 0:
            return ASignal(data, self.sample_rate)
        fade_in = min(fade_in, self.sample_count)
        fade_out = min(fade_out, self.sample_count)
        if fade_in:
            data[:fade_in] *= np.linspace(
                0.0,
                1.0,
                fade_in,
                endpoint=True,
                dtype=np.float32,
            )[:, None]
        if fade_out:
            data[-fade_out:] *= np.linspace(
                1.0,
                0.0,
                fade_out,
                endpoint=True,
                dtype=np.float32,
            )[:, None]
        return ASignal(data, self.sample_rate)

    def pad(self, in_: int = 0, out: int = 0) -> ASignal:
        """Pad the signal with zero samples.

        Args:
            in_: Number of zero samples to add before the signal. Must not be
                negative.
            out: Number of zero samples to add after the signal. Must not be
                negative.

        Returns:
            A new padded ``ASignal``.

        Raises:
            ValueError: If ``in_`` or ``out`` is negative.
        """
        pad_in = int(in_)
        pad_out = int(out)
        if pad_in < 0:
            raise ValueError("Input pad length must not be negative")
        if pad_out < 0:
            raise ValueError("Output pad length must not be negative")
        data = np.pad(self._data, ((pad_in, pad_out), (0, 0)))
        return ASignal(data, self.sample_rate)

    def as_array(self, dtype: DTypeLike = np.float32) -> NDArray:
        """Return signal data as a NumPy array shaped ``(samples, channels)``."""
        return self._data.astype(dtype, copy=True)

    def max(self) -> NDArray[np.float32]:
        """Return absolute peak value for each channel."""
        if self.sample_count == 0:
            return np.zeros(self.channel_count, dtype=np.float32)
        return np.max(np.abs(self._data), axis=0).astype(np.float32)

    def peak_levels(self, chunk_size: int) -> NDArray[np.float32]:
        """Return per-channel absolute peaks for fixed-size chunks.

        Args:
            chunk_size: Number of samples in one chunk. Must be positive.

        Returns:
            A ``float32`` array shaped ``(chunks, channels)``. The final chunk
            is included even when it contains fewer than ``chunk_size`` samples.

        Raises:
            ValueError: If ``chunk_size`` is not positive.
        """
        if chunk_size <= 0:
            raise ValueError("Chunk size must be positive")
        if self.sample_count == 0:
            return np.empty((0, self.channel_count), dtype=np.float32)

        starts = np.arange(0, self.sample_count, chunk_size)
        levels = np.zeros((len(starts), self.channel_count), dtype=np.float32)
        for index, start in enumerate(starts):
            chunk = self._data[start : start + chunk_size]
            levels[index] = np.max(np.abs(chunk), axis=0)
        return levels

    @staticmethod
    def _coerce_array(data: ArrayLike) -> NDArray[np.float32]:
        array = np.asarray(data, dtype=np.float32)
        if array.ndim == 1:
            array = array[:, None]
        if array.ndim != 2:
            raise ValueError("ASignal data must be one- or two-dimensional")
        if array.shape[1] < 1:
            raise ValueError("ASignal data must contain at least one channel")
        return array.copy()

    @staticmethod
    def _validate_sample_rate(sample_rate: int) -> int:
        sample_rate = int(sample_rate)
        if sample_rate <= 0:
            raise ValueError("Sample rate must be positive")
        return sample_rate

    @staticmethod
    def _combine(signals: tuple[object, ...]) -> tuple[NDArray[np.float32], int]:
        if not signals:
            raise ValueError("At least one ASignal is required")
        typed_signals: list[ASignal] = []
        for signal in signals:
            if not isinstance(signal, ASignal):
                raise TypeError("All combined signals must be ASignal instances")
            typed_signals.append(signal)

        sample_count = typed_signals[0].sample_count
        sample_rate = typed_signals[0].sample_rate
        if any(signal.sample_count != sample_count for signal in typed_signals):
            raise ValueError("Combined ASignal instances must have equal lengths")
        if any(signal.sample_rate != sample_rate for signal in typed_signals):
            raise ValueError("Combined ASignal instances must have equal sample rates")
        data = np.concatenate([signal.as_array() for signal in typed_signals], axis=1)
        return data, sample_rate

    def __array__(self, dtype: DTypeLike | None = None) -> NDArray:
        if dtype is None:
            return self.as_array()
        return self.as_array(dtype)

    def __len__(self) -> int:
        return self.sample_count

    def __repr__(self) -> str:
        return (
            f"ASignal(samples={self.sample_count}, "
            f"channels={self.channel_count}, "
            f"sample_rate={self.sample_rate})"
        )


@dataclass(frozen=True)
class FrequencyBand:
    """Frequency range in hertz."""

    low: float = 20.0
    high: float = 20_000.0

    def validate(self, *, nyquist: float | None = None) -> None:
        if self.low <= 0:
            raise ValueError("Low frequency must be positive")
        if self.high <= self.low:
            raise ValueError("High frequency must be greater than low frequency")
        if nyquist is not None and self.high >= nyquist:
            raise ValueError("High frequency must be below Nyquist frequency")

    def as_tuple(self) -> tuple[float, float]:
        return self.low, self.high
