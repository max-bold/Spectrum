from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares
from scipy.signal import find_peaks


@dataclass(frozen=True)
class FitResult:
    sections: int
    physical_params: NDArray[np.float64]
    rms_log_error: float
    max_abs_log_error: float
    selection_score: float = math.nan  # Legacy BIC field retained for stored results.
    inductances_refined: bool = False
    fit_method: str = "peak_anchored"
    target_log_error: float = math.nan
    stop_reason: str = ""


@dataclass(frozen=True)
class SpiceTableValues:
    l1: str
    sections: tuple[tuple[str, str, str], ...]
    r1: str


def rlc_from_rf0q(
    resistance: NDArray[np.float64],
    frequency: NDArray[np.float64],
    quality: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    omega = 2.0 * np.pi * frequency
    return resistance / (quality * omega), quality / (resistance * omega)


def speaker_impedance(
    frequency: NDArray[np.float64],
    physical_params: NDArray[np.float64],
    sections: int,
) -> NDArray[np.complex128]:
    params = np.asarray(physical_params, dtype=np.float64)
    if params.size != 2 + sections * 3:
        raise ValueError(f"Expected {2 + sections * 3} model parameters")
    omega = 2.0 * np.pi * np.asarray(frequency, dtype=np.float64)
    impedance = params[0] + 1j * omega * params[1]
    for index in range(sections):
        resistance, f0, quality = params[2 + index * 3 : 5 + index * 3]
        inductance, capacitance = rlc_from_rf0q(
            np.asarray([resistance], dtype=np.float64),
            np.asarray([f0], dtype=np.float64),
            np.asarray([quality], dtype=np.float64),
        )
        jw = 1j * omega
        impedance += 1.0 / (
            1.0 / resistance + 1.0 / (jw * inductance[0]) + jw * capacitance[0]
        )
    return np.asarray(impedance, dtype=np.complex128)


def fit_impedance(
    frequency: NDArray[np.float64],
    measured_magnitude: NDArray[np.float64],
    sections: int,
    *,
    max_evaluations: int = 2000,
    refine_inductances: bool = True,
) -> FitResult:
    """Fit detected resonances, keeping their frequencies fixed in both stages.

    First fit Re, Le and section resistances with section L = 1 mH. If
    RMS log error exceeds 2%, optionally
    fit section inductances as well, recomputing C to preserve each f0.
    """
    frequency, measured = _validate_fit_data(frequency, measured_magnitude)
    if not isinstance(sections, (int, np.integer)) or sections < 0:
        raise ValueError("Section count must be a non-negative integer")
    peaks = _resonance_peaks(measured, sections)
    if len(peaks) != sections:
        raise ValueError("Requested more sections than detected resonance peaks")
    return _fit_detected_peaks(
        frequency, measured, peaks, max_evaluations, refine_inductances
    )[0]


def fit_impedance_auto(
    frequency: NDArray[np.float64],
    measured_magnitude: NDArray[np.float64],
    *,
    min_sections: int = 0,
    max_sections: int = 10,
    max_evaluations: int = 2000,
    target_log_error: float = 0.02,
) -> tuple[FitResult, list[FitResult]]:
    """Add anchored sections until RMS log error reaches the target.

    Start with series RL, then add the most under-fitted measured peak.
    Once measured peaks are exhausted, use positive local residual maxima.
    Frequencies remain fixed during each fit. Return the best fit and one
    candidate per evaluated section count, including the initial RL model.
    """
    if (
        not isinstance(min_sections, (int, np.integer))
        or not isinstance(max_sections, (int, np.integer))
        or not 0 <= min_sections <= max_sections
    ):
        raise ValueError("Require 0 <= min_sections <= max_sections")
    if not math.isfinite(target_log_error) or target_log_error <= 0:
        raise ValueError("target_log_error must be finite and positive")
    frequency, measured = _validate_fit_data(frequency, measured_magnitude)
    measured_peaks = _resonance_peaks(measured, len(measured))
    selected: list[int] = []
    candidates: list[FitResult] = []
    best: FitResult | None = None
    stop_reason = "section_limit"
    for count in range(max_sections + 1):
        try:
            fit, _ = _fit_detected_peaks(
                frequency, measured, np.asarray(sorted(selected), dtype=np.int64),
                max_evaluations, True, target_log_error,
            )
        except RuntimeError:
            if best is None:
                raise
            stop_reason = "optimization_failed"
            break
        fit = replace(
            fit, fit_method="progressive_peak_anchored",
            target_log_error=target_log_error,
        )
        candidates.append(fit)
        if count >= min_sections and (best is None or fit.rms_log_error < best.rms_log_error):
            best = fit
        if count >= min_sections and fit.rms_log_error <= target_log_error:
            stop_reason = "target_reached"
            break
        if count == max_sections:
            break
        modeled = np.abs(speaker_impedance(frequency, fit.physical_params, count))
        residual = np.log(measured) - np.log(np.maximum(modeled, 1e-30))
        available = np.ones(len(frequency), dtype=bool)
        for peak in selected:
            available[max(0, peak - 3):peak + 4] = False
        remaining = measured_peaks[available[measured_peaks]]
        remaining = remaining[residual[remaining] > 0]
        if not len(remaining):
            residual_peaks, _ = find_peaks(residual, distance=3)
            remaining = residual_peaks[
                available[residual_peaks] & (residual[residual_peaks] > 0)
            ]
        if not len(remaining):
            stop_reason = "no_resonance"
            break
        selected.append(int(remaining[np.argmax(residual[remaining])]))
    if best is None:
        raise ValueError("Fewer resonance candidates found than min_sections")
    return replace(best, stop_reason=stop_reason), candidates


def _resonance_peaks(measured: NDArray[np.float64], limit: int) -> NDArray[np.int64]:
    peaks, properties = find_peaks(
        measured,
        prominence=max(0.5, float(np.ptp(measured)) * 0.04),
        distance=3,
    )
    ranked = np.argsort(properties["prominences"])[::-1][:limit]
    return np.sort(peaks[ranked])


def _fit_detected_peaks(
    frequency: NDArray[np.float64],
    measured: NDArray[np.float64],
    peaks: NDArray[np.int64],
    max_evaluations: int,
    refine_inductances: bool,
    target_log_error: float = 0.02,
) -> tuple[FitResult, list[FitResult]]:
    sections = len(peaks)
    # The visible maxima anchor the contours; neither optimizer can move them.
    resonance_frequency = frequency[peaks]
    re0 = float(np.clip(np.min(measured), 0.1, 100.0))
    high = frequency >= min(16_000.0, float(frequency[-1]) * 0.8)
    le0 = float(np.median(
        np.sqrt(np.maximum(measured[high] ** 2 - re0**2, 1e-12))
        / (2.0 * np.pi * frequency[high])
    ))
    initial = np.r_[
        re0, np.clip(le0, 1e-7, 1e-1), np.maximum(measured[peaks] - re0, 0.01)
    ]
    lower = np.r_[0.1, 1e-7, np.full(sections, 0.01)]
    upper = np.r_[max(0.100001, re0), 1e-1, np.full(sections, 1000.0)]

    def physical(values: NDArray[np.float64], refined: bool) -> NDArray[np.float64]:
        resistances = values[2:2 + sections]
        inductances = values[2 + sections:] if refined else np.full(sections, 1e-3)
        quality = resistances / (2.0 * np.pi * resonance_frequency * inductances)
        return np.r_[values[:2], np.column_stack(
            (resistances, resonance_frequency, quality)
        ).ravel()]

    def optimize(
        start: NDArray[np.float64],
        low: NDArray[np.float64],
        high: NDArray[np.float64],
        refined: bool,
    ) -> FitResult:
        def residual(log_values: NDArray[np.float64]) -> NDArray[np.float64]:
            params = physical(np.exp(log_values), refined)
            modeled = np.abs(speaker_impedance(frequency, params, sections))
            return np.log(np.maximum(modeled, 1e-30)) - np.log(measured)

        solution = least_squares(
            residual, np.log(np.clip(start, low, high)),
            bounds=(np.log(low), np.log(high)), loss="soft_l1",
            f_scale=0.08, x_scale="jac", max_nfev=max_evaluations,
        )
        if not solution.success:
            raise RuntimeError(f"SPICE fit did not converge: {solution.message}")
        error = residual(solution.x)
        return FitResult(
            sections, physical(np.exp(solution.x), refined),
            float(np.sqrt(np.mean(error**2))), float(np.max(np.abs(error))),
            inductances_refined=refined,
        )

    fixed = optimize(initial, lower, upper, False)
    candidates = [fixed]
    if sections and refine_inductances and (
        fixed.rms_log_error > target_log_error
    ):
        start = np.r_[
            fixed.physical_params[:2], fixed.physical_params[2::3],
            np.full(sections, 1e-3),
        ]
        refined = optimize(
            start, np.r_[lower, np.full(sections, 1e-6)],
            np.r_[upper, np.full(sections, 1.0)], True,
        )
        candidates.append(refined)
        # Keep the simpler stage unless changing widths meaningfully helps.
        if (
            refined.rms_log_error <= target_log_error
            or refined.rms_log_error < fixed.rms_log_error * 0.99
        ):
            return refined, candidates
    return fixed, candidates


def format_spice_table(result: FitResult) -> SpiceTableValues:
    section_params = result.physical_params[2:].reshape(result.sections, 3)
    resistance = section_params[:, 0]
    inductance, capacitance = rlc_from_rf0q(
        resistance, section_params[:, 1], section_params[:, 2]
    )
    values = [
        (
            _format_value(inductance[index] * 1e3),
            _format_value(capacitance[index] * 1e6),
            _format_value(resistance[index]),
        )
        for index in range(result.sections)
    ]
    return SpiceTableValues(
        l1=_format_value(float(result.physical_params[1]) * 1e3),
        sections=tuple(values),
        r1=_format_value(float(result.physical_params[0])),
    )


def _validate_fit_data(
    frequency: NDArray[np.float64],
    measured_magnitude: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    frequency = np.asarray(frequency, dtype=np.float64)
    measured = np.asarray(measured_magnitude, dtype=np.float64)
    if frequency.ndim != 1 or measured.shape != frequency.shape:
        raise ValueError("Frequency and magnitude must be matching 1-D arrays")
    mask = np.isfinite(frequency) & np.isfinite(measured) & (frequency > 0) & (measured > 0)
    frequency, measured = frequency[mask], measured[mask]
    if len(frequency) < 3:
        raise ValueError("At least three valid impedance points are required")
    order = np.argsort(frequency)
    return frequency[order], measured[order]


def _format_value(value: float) -> str:
    return f"{value:.3g}"
