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
    selection_score: float = math.nan


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
) -> FitResult:
    frequency, measured = _validate_fit_data(frequency, measured_magnitude)
    initial = _initial_guess(frequency, measured, sections)
    lower, upper = _model_bounds(frequency, sections)
    solution = least_squares(
        _model_residual,
        np.clip(np.log(initial), lower, upper),
        bounds=(lower, upper),
        args=(frequency, measured, sections),
        loss="soft_l1",
        f_scale=0.08,
        x_scale="jac",
        max_nfev=max_evaluations,
    )
    residual = _model_residual(solution.x, frequency, measured, sections)
    return FitResult(
        sections,
        np.exp(solution.x),
        float(np.sqrt(np.mean(residual * residual))),
        float(np.max(np.abs(residual))),
    )


def fit_impedance_auto(
    frequency: NDArray[np.float64],
    measured_magnitude: NDArray[np.float64],
    *,
    min_sections: int = 0,
    max_sections: int = 10,
    max_evaluations: int = 2000,
) -> tuple[FitResult, list[FitResult]]:
    # needs testing: automatic section selection is slow and can converge to
    # implausible local minima for real loudspeaker measurements.
    candidates: list[FitResult] = []
    for sections in range(min_sections, max_sections + 1):
        result = fit_impedance(
            frequency,
            measured_magnitude,
            sections,
            max_evaluations=max_evaluations,
        )
        parameter_count = 2 + sections * 3
        variance = max(result.rms_log_error**2, 1e-30)
        score = len(frequency) * math.log(variance) + parameter_count * math.log(
            len(frequency)
        )
        candidates.append(replace(result, selection_score=score))
    return min(candidates, key=lambda item: item.selection_score), candidates


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


def _initial_guess(
    frequency: NDArray[np.float64],
    measured: NDArray[np.float64],
    sections: int,
) -> NDArray[np.float64]:
    minimum = float(np.min(measured))
    re0 = float(np.clip(minimum * 0.9, 0.1, 100.0))
    omega_max = 2.0 * np.pi * float(frequency[-1])
    le0 = math.sqrt(max(float(measured[-1]) ** 2 - re0**2, 1e-12)) / omega_max
    guesses = [re0, float(np.clip(le0, 1e-7, 1e-1))]
    peaks, properties = find_peaks(
        measured,
        prominence=max(0.5, float(np.ptp(measured)) * 0.04),
        distance=3,
    )
    ranked = sorted(
        zip(peaks, properties.get("prominences", np.zeros_like(peaks))),
        key=lambda item: item[1],
        reverse=True,
    )
    selected = sorted(int(index) for index, _ in ranked[:sections])
    for peak in selected:
        guesses.extend([max(float(measured[peak]) - re0, 1.0), float(frequency[peak]), 3.0])
    missing = sections - len(selected)
    if missing:
        filler = np.geomspace(
            max(float(frequency[0]) * 2.0, float(frequency[0])),
            max(float(frequency[0]) * 2.01, float(frequency[-1]) / 2.0),
            missing,
        )
        resistance = max(float(np.percentile(measured, 75)) - re0, 1.0)
        for f0 in filler:
            guesses.extend([resistance, float(f0), 3.0])
    lower, upper = _model_bounds(frequency, sections, physical=True)
    return np.clip(np.asarray(guesses), lower, upper)


def _model_bounds(
    frequency: NDArray[np.float64],
    sections: int,
    *,
    physical: bool = False,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    lower = [0.1, 1e-7]
    upper = [100.0, 1e-1]
    for _ in range(sections):
        lower.extend([0.01, float(frequency[0]) / 2.0, 0.1])
        upper.extend([1000.0, float(frequency[-1]) * 2.0, 100.0])
    low = np.asarray(lower)
    high = np.asarray(upper)
    return (low, high) if physical else (np.log(low), np.log(high))


def _model_residual(
    log_params: NDArray[np.float64],
    frequency: NDArray[np.float64],
    measured: NDArray[np.float64],
    sections: int,
) -> NDArray[np.float64]:
    modeled = np.abs(speaker_impedance(frequency, np.exp(log_params), sections))
    return np.log(np.maximum(modeled, 1e-30)) - np.log(measured)


def _validate_fit_data(
    frequency: NDArray[np.float64],
    measured_magnitude: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    frequency = np.asarray(frequency, dtype=np.float64)
    measured = np.asarray(measured_magnitude, dtype=np.float64)
    mask = np.isfinite(frequency) & np.isfinite(measured) & (frequency > 0) & (measured > 0)
    frequency, measured = frequency[mask], measured[mask]
    if len(frequency) < 3:
        raise ValueError("At least three valid impedance points are required")
    order = np.argsort(frequency)
    return frequency[order], measured[order]


def _format_value(value: float) -> str:
    return f"{value:.3g}"
