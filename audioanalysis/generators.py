"""Signal generators.

The functions in this module return ``ASignal`` instances ready for audio
playback or offline analysis. Generators are deliberately stateless: callers can
pass an explicit ``numpy.random.Generator`` when deterministic noise is needed.
"""

from __future__ import annotations

from typing import overload, cast

import numpy as np
from numpy.typing import NDArray
from scipy.signal import butter, chirp, sosfilt

from .types import ASignal, FrequencyBand

PINKING_SOS = np.array(
    [
        [0.04992203, -0.00539063, 0.0, 1.0, -0.55594526, 0.0],
        [1.0, -1.81488818, 0.81786161, 1.0, -1.93901074, 0.93928204],
    ],
    dtype=np.float64,
)


def log_chirp(
    samples: int,
    sample_rate: int = 44_100,
    band: FrequencyBand | tuple[float, float] = FrequencyBand(),
    *,
    amplitude: float = 0.9,
    channels: int = 1,
    pad: int = 1024,
    fade: int = 1024,
) -> ASignal:
    """Generate a peak-normalized logarithmic chirp.

    The chirp sweeps exponentially from ``band.low`` to ``band.high`` over the
    requested number of samples. This is useful for frequency-response and
    impulse-response measurements because each octave receives comparable time
    coverage. Output always uses ``(samples, channels)`` shape, including mono
    output as ``(samples, 1)``.

    Args:
        samples: Number of samples to generate. Must be positive.
        sample_rate: Sampling rate in hertz. The upper band edge must be below
            ``sample_rate / 2``.
        band: Frequency range in hertz, either as ``FrequencyBand`` or
            ``(low, high)`` tuple.
        amplitude: Target absolute peak after normalization.
        channels: Desired channel count.
        pad: Number of zero samples to add before and after the signal.
        fade: Fade-in and fade-out length in samples before padding.

    Returns:
        An ``ASignal`` containing the generated chirp.

    Raises:
        ValueError: If ``samples`` is not positive, the frequency band is
            invalid, the high band edge reaches Nyquist, ``channels`` is not
            valid, or ``pad``/``fade`` are negative.
    """
    if samples <= 0:
        raise ValueError("Sample count must be positive")
    frequency_band = _coerce_band(band)
    frequency_band.validate(nyquist=sample_rate / 2)
    time = np.arange(samples, dtype=np.float64) / float(sample_rate)
    duration = samples / float(sample_rate)
    signal = chirp(
        time,
        frequency_band.low,
        duration,
        frequency_band.high,
        method="logarithmic",
        phi=-90,
    )
    return _shape_output(
        ASignal(signal, sample_rate).to_channels(channels).normalize(amplitude),
        pad=pad,
        fade=fade,
    )


def white_noise(
    samples: int,
    sample_rate: int = 44_100,
    *,
    amplitude: float = 0.9,
    channels: int = 1,
    rng: np.random.Generator | None = None,
    pad: int = 1024,
    fade: int = 1024,
) -> ASignal:
    """Generate peak-normalized white noise.

    Samples are drawn from a uniform distribution over ``[-1.0, 1.0]`` and then
    normalized to the requested peak amplitude. Output always uses
    ``(samples, channels)`` shape, including mono output as ``(samples, 1)``.
    Pass a seeded
    ``numpy.random.Generator`` to make the output reproducible in tests or
    measurement scripts.

    Args:
        samples: Number of samples to generate. Must be positive.
        sample_rate: Sampling rate in hertz.
        amplitude: Target absolute peak after normalization.
        channels: Desired channel count.
        rng: Optional random number generator. If omitted, a fresh default
            generator is created.
        pad: Number of zero samples to add before and after the signal.
        fade: Fade-in and fade-out length in samples before padding.

    Returns:
        An ``ASignal`` containing the generated white noise.

    Raises:
        ValueError: If ``samples`` is not positive, ``channels`` is not valid,
            or ``pad``/``fade`` are negative.
    """
    if samples <= 0:
        raise ValueError("Sample count must be positive")
    generator = rng or np.random.default_rng()
    signal = generator.uniform(-1.0, 1.0, samples)
    return _shape_output(
        ASignal(signal, sample_rate).to_channels(channels).normalize(amplitude),
        pad=pad,
        fade=fade,
    )


@overload
def pink_noise(
    samples: int,
    sample_rate: int = 44_100,
    band: FrequencyBand | tuple[float, float] = FrequencyBand(),
    *,
    amplitude: float = 0.9,
    channels: int = 1,
    rng: np.random.Generator | None = None,
    pad: int = 1024,
    fade: int = 1024,
    zi: None = None,
) -> ASignal: ...


@overload
def pink_noise(
    samples: int,
    sample_rate: int = 44_100,
    band: FrequencyBand | tuple[float, float] = FrequencyBand(),
    *,
    amplitude: float = 0.9,
    channels: int = 1,
    rng: np.random.Generator | None = None,
    pad: int = 1024,
    fade: int = 1024,
    zi: NDArray[np.float64],
) -> tuple[ASignal, NDArray[np.float64]]: ...


def pink_noise(
    samples: int,
    sample_rate: int = 44_100,
    band: FrequencyBand | tuple[float, float] = FrequencyBand(),
    *,
    amplitude: float = 0.9,
    channels: int = 1,
    rng: np.random.Generator | None = None,
    pad: int = 1024,
    fade: int = 1024,
    zi: NDArray[np.float64] | None = None,
) -> ASignal | tuple[ASignal, NDArray[np.float64]]:
    """Generate peak-normalized, band-limited pink noise.

    The function starts with uniform white noise, applies a fixed pinking filter
    cascade, then applies a Butterworth band-pass filter over ``band``. The
    result is normalized after filtering so the returned audio has the requested
    peak amplitude. Output always uses ``(samples, channels)`` shape, including
    mono output as ``(samples, 1)``.

    Args:
        samples: Number of samples to generate. Must be positive.
        sample_rate: Sampling rate in hertz. The upper band edge must be below
            ``sample_rate / 2``.
        band: Frequency range in hertz, either as ``FrequencyBand`` or
            ``(low, high)`` tuple.
        amplitude: Target absolute peak after filtering.
        channels: Desired channel count.
        rng: Optional random number generator. If omitted, a fresh default
            generator is created.
        pad: Number of zero samples to add before and after the signal.
        fade: Fade-in and fade-out length in samples before padding.
        zi: Optional SOS filter state for continuous generation. If provided,
            the returned value is ``(signal, zf)`` where ``zf`` is the state to
            pass into the next call.

    Returns:
        An ``ASignal`` containing the generated pink noise, or ``(ASignal,
        zf)`` when ``zi`` is provided.

    Raises:
        ValueError: If ``samples`` is not positive, the frequency band is
            invalid, the high band edge reaches Nyquist, ``channels`` is not
            valid, or ``pad``/``fade`` are negative.
    """
    if samples <= 0:
        raise ValueError("Sample count must be positive")
    frequency_band = _coerce_band(band)
    frequency_band.validate(nyquist=sample_rate / 2)
    generator = rng or np.random.default_rng()
    white = generator.uniform(-1.0, 1.0, samples)
    combined_sos = _pink_noise_sos(sample_rate, frequency_band)
    if zi is None:
        pink = cast(NDArray[np.float64], sosfilt(combined_sos, white))
        return _shape_output(
            ASignal(pink, sample_rate).to_channels(channels).normalize(amplitude),
            pad=pad,
            fade=fade,
        )

    pink, zf = sosfilt(combined_sos, white, zi=zi)
    signal = _shape_output(
        ASignal(pink, sample_rate).to_channels(channels).normalize(amplitude),
        pad=pad,
        fade=fade,
    )
    return signal, cast(NDArray[np.float64], zf)


def pink_noise_zi(
    sample_rate: int = 44_100,
    band: FrequencyBand | tuple[float, float] = FrequencyBand(),
) -> NDArray[np.float64]:
    """Return zero SOS state for stateful ``pink_noise`` generation."""
    frequency_band = _coerce_band(band)
    frequency_band.validate(nyquist=sample_rate / 2)
    return np.zeros(
        (_pink_noise_sos(sample_rate, frequency_band).shape[0], 2),
        dtype=np.float64,
    )


def _pink_noise_sos(
    sample_rate: int,
    band: FrequencyBand,
) -> NDArray[np.float64]:
    band_sos = cast(
        NDArray[np.float64],
        butter(
            4,
            band.as_tuple(),
            "bandpass",
            output="sos",
            fs=sample_rate,
        ),
    )
    return cast(NDArray[np.float64], np.vstack((PINKING_SOS, band_sos)))


def _shape_output(signal: ASignal, *, pad: int, fade: int) -> ASignal:
    if pad < 0:
        raise ValueError("Pad length must not be negative")
    if fade < 0:
        raise ValueError("Fade length must not be negative")
    return signal.fade(fade, fade).pad(pad, pad)


def _coerce_band(band: FrequencyBand | tuple[float, float]) -> FrequencyBand:
    """Normalize public band input into a ``FrequencyBand`` instance."""
    if isinstance(band, FrequencyBand):
        return band
    return FrequencyBand(float(band[0]), float(band[1]))
