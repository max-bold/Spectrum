from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares

from .smoothing import SmoothingWindow, grid_smooth
from .types import ASignal, FrequencyBand

SPEED_OF_SOUND_M_S = 343.0
_EPSILON = 1e-20


@dataclass(frozen=True)
class PhaseConfig:
    """Settings for a two-channel acoustic phase measurement.

    Logical input A is the measured acoustic signal and logical input B is the
    electrical reference. ``delay_correction_meters`` is the total propagation
    distance removed from the displayed phase. ``None`` uses the fitted delay.
    """

    band: FrequencyBand = FrequencyBand()
    delay_fit_band: FrequencyBand = FrequencyBand(80.0, 15_000.0)
    points: int = 1024
    smoothing_octaves: float = 0.1
    delay_correction_meters: float | None = None
    minimum_a_db: float = -60.0
    minimum_b_db: float = -60.0
    speed_of_sound: float = SPEED_OF_SOUND_M_S

    def validate(self, sample_rate: int) -> None:
        self.band.validate(nyquist=sample_rate / 2)
        self.delay_fit_band.validate(nyquist=sample_rate / 2)
        if (
            self.delay_fit_band.low < self.band.low
            or self.delay_fit_band.high > self.band.high
        ):
            raise ValueError("Delay fit range must be inside frequency range")
        if self.points < 2:
            raise ValueError("Point count must be at least two")
        if self.smoothing_octaves <= 0.0:
            raise ValueError("Smoothing width must be positive")
        if self.speed_of_sound <= 0.0:
            raise ValueError("Speed of sound must be positive")


@dataclass(frozen=True)
class PhaseResult:
    """Smoothed transfer magnitude and unwrapped compensated phase."""

    frequency: NDArray[np.float64]
    magnitude_db: NDArray[np.float64]
    phase_degrees: NDArray[np.float64]
    estimated_delay_seconds: float
    estimated_delay_meters: float
    compensation_delay_seconds: float


def analyze_phase(recording: ASignal, config: PhaseConfig) -> PhaseResult:
    """Calculate acoustic transfer A/B and compensate its linear delay.

    The returned phase is deliberately not wrapped. Presentation choices such
    as wrapping to +/-180 degrees or converting to degrees/decade belong to the
    application plot layer.
    """

    if recording.channel_count != 2:
        raise ValueError("Phase analysis requires logical input channels A and B")
    if recording.sample_count < 2:
        raise ValueError("Phase analysis requires at least two samples")
    config.validate(recording.sample_rate)

    data = recording.as_array(np.float64)
    channel_a = data[:, 0] - np.mean(data[:, 0])
    channel_b = data[:, 1] - np.mean(data[:, 1])
    frequency = np.fft.rfftfreq(len(data), d=1.0 / recording.sample_rate)
    fft_a = np.fft.rfft(channel_a)
    fft_b = np.fft.rfft(channel_b)
    transfer = np.divide(
        fft_a,
        fft_b,
        out=np.full(fft_a.shape, np.nan + 0j, dtype=np.complex128),
        where=np.abs(fft_b) > _EPSILON,
    )

    valid_transfer = np.isfinite(transfer)
    if np.count_nonzero(valid_transfer) < 2:
        raise ValueError("Not enough valid FFT bins for phase analysis")
    unwrapped_phase = np.interp(
        frequency,
        frequency[valid_transfer],
        np.unwrap(np.angle(transfer[valid_transfer])),
    )
    grid = np.geomspace(config.band.low, config.band.high, config.points)
    smoothed_phase = np.asarray(
        grid_smooth(
            frequency,
            unwrapped_phase,
            grid,
            window=SmoothingWindow.GAUSSIAN,
            width=config.smoothing_octaves,
        ),
        dtype=np.float64,
    )
    smoothed_a = np.asarray(
        grid_smooth(
            frequency,
            np.abs(fft_a),
            grid,
            window=SmoothingWindow.GAUSSIAN,
            width=config.smoothing_octaves,
        ),
        dtype=np.float64,
    )
    smoothed_b = np.asarray(
        grid_smooth(
            frequency,
            np.abs(fft_b),
            grid,
            window=SmoothingWindow.GAUSSIAN,
            width=config.smoothing_octaves,
        ),
        dtype=np.float64,
    )
    delay = _estimate_phase_delay_from_unwrapped(
        grid,
        smoothed_phase,
        smoothed_a,
        smoothed_b,
        config.delay_fit_band,
        minimum_a_db=config.minimum_a_db,
        minimum_b_db=config.minimum_b_db,
    )
    compensation_delay = (
        delay
        if config.delay_correction_meters is None
        else config.delay_correction_meters / config.speed_of_sound
    )
    compensated_phase = smoothed_phase + (
        2.0 * np.pi * grid * compensation_delay
    )
    compensated_transfer = transfer * np.exp(
        1j * 2.0 * np.pi * frequency * compensation_delay
    )
    smoothed_compensated = grid_smooth(
        frequency,
        compensated_transfer,
        grid,
        window=SmoothingWindow.GAUSSIAN,
        width=config.smoothing_octaves,
    )
    magnitude_db = 20.0 * np.log10(
        np.maximum(np.abs(smoothed_compensated), _EPSILON)
    )
    phase_degrees = np.rad2deg(compensated_phase)
    return PhaseResult(
        frequency=grid,
        magnitude_db=np.asarray(magnitude_db, dtype=np.float64),
        phase_degrees=np.asarray(phase_degrees, dtype=np.float64),
        estimated_delay_seconds=delay,
        estimated_delay_meters=delay * config.speed_of_sound,
        compensation_delay_seconds=compensation_delay,
    )


def estimate_phase_delay(
    frequency: NDArray[np.floating],
    transfer: NDArray[np.complexfloating],
    fft_a: NDArray[np.number],
    fft_b: NDArray[np.number],
    fit_band: FrequencyBand,
    *,
    minimum_a_db: float = -60.0,
    minimum_b_db: float = -60.0,
) -> float:
    """Estimate A relative to B delay from a weighted phase slope.

    ``frequency`` may be a linear FFT grid or a pre-smoothed logarithmic grid.
    The channel spectra are used only for level rejection and fit weights.
    """

    frequency = np.asarray(frequency, dtype=np.float64)
    transfer = np.asarray(transfer, dtype=np.complex128)
    magnitude_a = np.abs(np.asarray(fft_a))
    magnitude_b = np.abs(np.asarray(fft_b))
    if not (
        frequency.shape == transfer.shape == magnitude_a.shape == magnitude_b.shape
    ):
        raise ValueError("Phase delay arrays must have equal shapes")

    phase = np.full(transfer.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(transfer)
    phase[valid] = np.unwrap(np.angle(transfer[valid]))
    return _estimate_phase_delay_from_unwrapped(
        frequency,
        phase,
        magnitude_a,
        magnitude_b,
        fit_band,
        minimum_a_db=minimum_a_db,
        minimum_b_db=minimum_b_db,
    )


def _estimate_phase_delay_from_unwrapped(
    frequency: NDArray[np.floating],
    phase: NDArray[np.floating],
    magnitude_a: NDArray[np.number],
    magnitude_b: NDArray[np.number],
    fit_band: FrequencyBand,
    *,
    minimum_a_db: float = -60.0,
    minimum_b_db: float = -60.0,
) -> float:
    frequency = np.asarray(frequency, dtype=np.float64)
    phase = np.asarray(phase, dtype=np.float64)
    magnitude_a = np.abs(np.asarray(magnitude_a))
    magnitude_b = np.abs(np.asarray(magnitude_b))
    if not (
        frequency.shape == phase.shape == magnitude_a.shape == magnitude_b.shape
    ):
        raise ValueError("Phase delay arrays must have equal shapes")

    limit_a = float(np.max(magnitude_a)) * 10.0 ** (minimum_a_db / 20.0)
    limit_b = float(np.max(magnitude_b)) * 10.0 ** (minimum_b_db / 20.0)
    mask = (
        (frequency >= fit_band.low)
        & (frequency <= fit_band.high)
        & np.isfinite(phase)
        & (magnitude_a >= limit_a)
        & (magnitude_b >= limit_b)
    )
    if np.count_nonzero(mask) < 3:
        raise ValueError("Not enough valid FFT bins for delay estimation")

    fit_frequency = frequency[mask]
    fit_phase = phase[mask]
    weights = np.sqrt(magnitude_a[mask] * magnitude_b[mask])
    maximum_weight = float(np.max(weights))
    if maximum_weight <= 0.0:
        raise ValueError("Input signals are too quiet for delay estimation")
    normalized_weights = weights / maximum_weight
    frequency_center = float(np.mean(fit_frequency))
    frequency_scale = float(np.ptp(fit_frequency))
    if frequency_scale <= 0.0:
        raise ValueError("Delay fit frequencies must span a non-zero range")
    phase_center = float(np.mean(fit_phase))
    phase_scale = max(float(np.ptp(fit_phase)), 1.0)
    normalized_frequency = (fit_frequency - frequency_center) / frequency_scale
    normalized_phase = (fit_phase - phase_center) / phase_scale
    initial = np.polyfit(
        normalized_frequency,
        normalized_phase,
        1,
        w=normalized_weights,
    )
    fit = least_squares(
        _quartic_residuals,
        initial,
        args=(normalized_frequency, normalized_phase, normalized_weights),
    )
    if not fit.success or not np.all(np.isfinite(fit.x)):
        raise ValueError(f"Fourth-power delay fit failed: {fit.message}")
    slope = float(fit.x[0]) * phase_scale / frequency_scale
    return float(-slope / (2.0 * np.pi))


def _quartic_residuals(
    coefficients: NDArray[np.floating],
    frequency: NDArray[np.float64],
    phase: NDArray[np.float64],
    weights: NDArray[np.float64],
) -> NDArray[np.float64]:
    residuals = weights * (
        phase - (coefficients[0] * frequency + coefficients[1])
    )
    # least_squares squares these values, producing sum((w * error) ** 4).
    return np.asarray(residuals * np.abs(residuals), dtype=np.float64)


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

    finite_pairs = np.isfinite(wrapped_phase[:-1]) & np.isfinite(wrapped_phase[1:])
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
