"""Fit a SPICE-friendly RLC impedance model to measured speaker impedance.

Default usage from the repository root:

    python sandbox/impedance_model.py

This reads info/1.DAT and writes info/impedance_model.csv with columns
R_ohm,L_mH,C_uF. The first CSV row contains the fitted series Re/Le terms in
R_ohm/L_mH columns, with an empty C_uF cell. Remaining rows are the fitted
parallel RLC sections.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

SCRIPT_DIR = Path(__file__).resolve().parent
if sys.path and Path(sys.path[0]).resolve() == SCRIPT_DIR:
    sys.path.pop(0)
shadowed_platform = sys.modules.get("platform")
if shadowed_platform is not None:
    platform_file = getattr(shadowed_platform, "__file__", None)
    if platform_file and Path(platform_file).resolve() == SCRIPT_DIR / "platform.py":
        del sys.modules["platform"]

import numpy as np

from scipy.optimize import least_squares
from scipy.signal import find_peaks


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "info" / "1.DAT"
DEFAULT_OUTPUT = REPO_ROOT / "info" / "impedance_model.csv"
DEFAULT_PLOT = REPO_ROOT / "info" / "impedance_fit.png"


@dataclass(frozen=True)
class FitResult:
    sections: int
    physical_params: np.ndarray
    rms_log_error: float
    max_abs_log_error: float
    selection_score: float = math.nan


def load_impedance_file(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load FREQ and measured |Z| columns from a DAT/CSV file."""
    if not path.exists():
        raise FileNotFoundError(path)

    data = np.genfromtxt(
        path,
        delimiter=",",
        names=True,
        dtype=float,
        encoding="utf-8-sig",
        deletechars="",
        replace_space="_",
    )
    if data.size == 0:
        raise ValueError(f"No data rows found in {path}")

    if data.ndim == 0:
        data = np.array([data], dtype=data.dtype)

    names = list(data.dtype.names or ())
    freq_name = _find_column(names, ("FREQ", "freq", "frequency", "Frequency"))
    z_name = _find_column(names, ("in", "Z_ABS", "z_abs", "Z", "impedance"))
    if freq_name is None or z_name is None:
        raise ValueError(
            f"Expected frequency and impedance columns in {path}; got {names}"
        )

    freq = np.asarray(data[freq_name], dtype=np.float64)
    z_abs = np.asarray(data[z_name], dtype=np.float64)

    mask = np.isfinite(freq) & np.isfinite(z_abs) & (freq > 0.0) & (z_abs > 0.0)
    freq = freq[mask]
    z_abs = z_abs[mask]
    if freq.size < 3:
        raise ValueError("Need at least three valid positive measurement points")

    order = np.argsort(freq)
    return freq[order], z_abs[order]


def _find_column(names: Iterable[str], candidates: Iterable[str]) -> str | None:
    normalized = {name.lower().strip('"'): name for name in names}
    for candidate in candidates:
        found = normalized.get(candidate.lower())
        if found is not None:
            return found
    return None


def rlc_from_rf0q(R: np.ndarray, f0: np.ndarray, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert optimizer parameters R, f0, Q to L and C for parallel RLC."""
    w0 = 2.0 * np.pi * f0
    L = R / (Q * w0)
    C = Q / (R * w0)
    return L, C


def rlc_parallel_impedance(w: np.ndarray, R: float, f0: float, Q: float) -> np.ndarray:
    """Return impedance of a parallel RLC contour parameterized by R, f0, Q."""
    L, C = rlc_from_rf0q(np.array([R]), np.array([f0]), np.array([Q]))
    jw = 1j * w
    admittance = (1.0 / R) + (1.0 / (jw * L[0])) + (jw * C[0])
    return 1.0 / admittance


def speaker_impedance(freq: np.ndarray, physical_params: np.ndarray, sections: int) -> np.ndarray:
    """Return modeled complex speaker impedance."""
    params = np.asarray(physical_params, dtype=np.float64)
    expected = 2 + sections * 3
    if params.size != expected:
        raise ValueError(f"Expected {expected} parameters, got {params.size}")

    w = 2.0 * np.pi * np.asarray(freq, dtype=np.float64)
    z = params[0] + 1j * w * params[1]
    for i in range(sections):
        R, f0, Q = params[2 + i * 3 : 2 + (i + 1) * 3]
        z = z + rlc_parallel_impedance(w, R, f0, Q)
    return z


def unpack_sections(physical_params: np.ndarray, sections: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return arrays of R, L, C for fitted RLC sections."""
    section_params = np.asarray(physical_params[2:], dtype=np.float64).reshape(sections, 3)
    R = section_params[:, 0]
    f0 = section_params[:, 1]
    Q = section_params[:, 2]
    L, C = rlc_from_rf0q(R, f0, Q)
    return R, L, C


def make_initial_guess(freq: np.ndarray, z_measured: np.ndarray, sections: int) -> np.ndarray:
    """Build physical initial parameters: Re, Le, then R/f0/Q per section."""
    if sections < 0:
        raise ValueError("sections must be non-negative")

    z_min = float(np.min(z_measured))
    z_last = float(z_measured[-1])
    f_min = float(np.min(freq))
    f_max = float(np.max(freq))

    Re0 = np.clip(z_min * 0.9, 0.1, 100.0)
    w_max = 2.0 * np.pi * f_max
    Le0 = math.sqrt(max(z_last * z_last - Re0 * Re0, 1e-12)) / w_max
    Le0 = float(np.clip(Le0, 1e-7, 1e-1))

    peak_indices = _choose_peak_indices(freq, z_measured, sections)
    guesses: list[float] = [Re0, Le0]
    used_freqs: set[float] = set()

    for peak_index in peak_indices:
        peak_freq = float(freq[peak_index])
        peak_z = float(z_measured[peak_index])
        guesses.extend([max(peak_z - Re0, 1.0), peak_freq, 3.0])
        used_freqs.add(round(peak_freq, 9))

    missing = sections - len(peak_indices)
    if missing > 0:
        start = max(f_min * 2.0, f_min)
        stop = max(start * 1.01, f_max / 2.0)
        filler_freqs = np.geomspace(start, stop, missing)
        typical_R = max(float(np.percentile(z_measured, 75) - Re0), 1.0)
        for f0 in filler_freqs:
            if round(float(f0), 9) in used_freqs:
                f0 *= 1.05
            guesses.extend([typical_R, float(f0), 3.0])

    lower, upper = make_bounds(freq, sections, physical=True)
    return np.clip(np.asarray(guesses, dtype=np.float64), lower, upper)


def _choose_peak_indices(freq: np.ndarray, z_measured: np.ndarray, sections: int) -> list[int]:
    if sections == 0:
        return []

    z_range = float(np.max(z_measured) - np.min(z_measured))
    peaks, props = find_peaks(
        z_measured,
        prominence=max(0.5, z_range * 0.04),
        distance=3,
    )
    prominences = props.get("prominences", np.zeros_like(peaks, dtype=float))
    ranked = sorted(zip(peaks, prominences), key=lambda item: item[1], reverse=True)
    selected = sorted(int(index) for index, _ in ranked[:sections])

    if not selected:
        filler = np.geomspace(freq[0] * 2.0, max(freq[0] * 2.01, freq[-1] / 2.0), sections)
        return [int(np.argmin(np.abs(freq - f))) for f in filler]

    return selected


def make_bounds(
    freq: np.ndarray,
    sections: int,
    *,
    physical: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Return lower/upper bounds, log-transformed unless physical=True."""
    f_min = float(np.min(freq))
    f_max = float(np.max(freq))
    lower = [0.1, 1e-7]
    upper = [100.0, 1e-1]
    for _ in range(sections):
        lower.extend([0.01, f_min / 2.0, 0.1])
        upper.extend([1000.0, f_max * 2.0, 100.0])

    lower_arr = np.asarray(lower, dtype=np.float64)
    upper_arr = np.asarray(upper, dtype=np.float64)
    if physical:
        return lower_arr, upper_arr
    return np.log(lower_arr), np.log(upper_arr)


def residual(
    log_params: np.ndarray,
    freq: np.ndarray,
    z_measured: np.ndarray,
    sections: int,
) -> np.ndarray:
    """Residual in log(|Z|), suitable for least squares."""
    physical = np.exp(np.asarray(log_params, dtype=np.float64))
    z_model = np.abs(speaker_impedance(freq, physical, sections))
    return np.log(np.maximum(z_model, 1e-30)) - np.log(z_measured)


def fit_impedance(
    freq: np.ndarray,
    z_measured: np.ndarray,
    sections: int,
    *,
    max_evaluations: int = 6000,
) -> FitResult:
    """Fit model parameters using scipy.optimize.least_squares."""
    initial = make_initial_guess(freq, z_measured, sections)
    lower, upper = make_bounds(freq, sections)
    x0 = np.clip(np.log(initial), lower, upper)

    solution = least_squares(
        residual,
        x0,
        bounds=(lower, upper),
        args=(freq, z_measured, sections),
        loss="soft_l1",
        f_scale=0.08,
        x_scale="jac",
        max_nfev=max_evaluations,
    )
    best_log_params = solution.x

    final_residual = residual(best_log_params, freq, z_measured, sections)
    return FitResult(
        sections=sections,
        physical_params=np.exp(best_log_params),
        rms_log_error=float(np.sqrt(np.mean(final_residual * final_residual))),
        max_abs_log_error=float(np.max(np.abs(final_residual))),
    )


def fit_impedance_auto(
    freq: np.ndarray,
    z_measured: np.ndarray,
    *,
    min_sections: int,
    max_sections: int,
    max_evaluations: int,
) -> tuple[FitResult, list[FitResult]]:
    """Fit several section counts and choose the best BIC-scored model."""
    candidates: list[FitResult] = []
    for sections in range(min_sections, max_sections + 1):
        result = fit_impedance(
            freq,
            z_measured,
            sections,
            max_evaluations=max_evaluations,
        )
        candidates.append(
            FitResult(
                sections=result.sections,
                physical_params=result.physical_params,
                rms_log_error=result.rms_log_error,
                max_abs_log_error=result.max_abs_log_error,
                selection_score=_bic_score(result, sample_count=freq.size),
            )
        )

    best = min(candidates, key=lambda item: item.selection_score)
    return best, candidates


def _bic_score(result: FitResult, *, sample_count: int) -> float:
    parameter_count = 2 + result.sections * 3
    variance = max(result.rms_log_error * result.rms_log_error, 1e-30)
    return sample_count * math.log(variance) + parameter_count * math.log(sample_count)


def save_result_csv(
    path: Path,
    physical_params: np.ndarray,
    sections: int,
    *,
    include_series: bool = True,
) -> None:
    """Save fitted values as CSV in Ohm, mH and uF."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []

    if include_series:
        rows.append(
            {
                "R": _format_csv_value(float(physical_params[0])),
                "L": _format_csv_value(float(physical_params[1] * 1e3)),
                "C": "",
            }
        )

    R, L, C = unpack_sections(physical_params, sections)
    for r_value, l_value, c_value in zip(R, L, C):
        rows.append(
            {
                "R": _format_csv_value(float(r_value)),
                "L": _format_csv_value(float(l_value * 1e3)),
                "C": _format_csv_value(float(c_value * 1e6)),
            }
        )

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["R_ohm", "L_mH", "C_uF"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "R_ohm": row["R"],
                    "L_mH": row["L"],
                    "C_uF": row["C"],
                }
            )


def save_diagnostic_csv(path: Path, physical_params: np.ndarray, sections: int) -> None:
    """Save a richer table next to the requested CSV for debugging/reuse."""
    diagnostic_path = path.with_name(path.stem + "_details.csv")
    R, L, C = unpack_sections(physical_params, sections)
    section_params = physical_params[2:].reshape(sections, 3)

    with diagnostic_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["element", "R_ohm", "L_H", "C_F", "f0_Hz", "Q"])
        writer.writerow(["series", _format_float(physical_params[0]), _format_float(physical_params[1]), "", "", ""])
        for i, (r_value, l_value, c_value) in enumerate(zip(R, L, C), start=1):
            _, f0, q = section_params[i - 1]
            writer.writerow(
                [
                    f"RLC_{i}",
                    _format_float(float(r_value)),
                    _format_float(float(l_value)),
                    _format_float(float(c_value)),
                    _format_float(float(f0)),
                    _format_float(float(q)),
                ]
            )


def plot_fit(path: Path, freq: np.ndarray, z_measured: np.ndarray, physical_params: np.ndarray, sections: int) -> None:
    """Save a measured/model/error plot."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional feature
        raise RuntimeError("matplotlib is required for --plot") from exc

    z_model = np.abs(speaker_impedance(freq, physical_params, sections))
    log_error = np.log(z_model) - np.log(z_measured)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax_z, ax_err) = plt.subplots(
        2,
        1,
        figsize=(9.0, 6.0),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    ax_z.semilogx(freq, z_measured, "o", markersize=4, label="measured |Z|")
    ax_z.semilogx(freq, z_model, "-", linewidth=2, label="model |Z|")
    ax_z.set_ylabel("Impedance, Ohm")
    ax_z.grid(True, which="both", alpha=0.25)
    ax_z.legend()

    ax_err.semilogx(freq, log_error, "-", color="tab:red", linewidth=1.5)
    ax_err.axhline(0.0, color="black", linewidth=0.8)
    ax_err.set_xlabel("Frequency, Hz")
    ax_err.set_ylabel("log error")
    ax_err.grid(True, which="both", alpha=0.25)

    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _format_float(value: float) -> str:
    return f"{value:.12g}"


def _format_csv_value(value: float) -> str:
    return f"{value:.2f}"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fit speaker impedance with series Re/Le and parallel RLC sections.",
    )
    parser.add_argument(
        "input_file",
        nargs="?",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"DAT/CSV input file, default: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "-n",
        "--sections",
        type=int,
        default=None,
        help="number of parallel RLC sections to fit; omit for automatic selection",
    )
    parser.add_argument(
        "--auto-sections",
        action="store_true",
        help="select the number of RLC sections automatically",
    )
    parser.add_argument(
        "--min-sections",
        type=int,
        default=0,
        help="minimum section count for automatic selection, default: 0",
    )
    parser.add_argument(
        "--max-sections",
        type=int,
        default=8,
        help="maximum section count for automatic selection, default: 8",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"output CSV with R_ohm,L_mH,C_uF columns, default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--no-series",
        action="store_true",
        help="omit fitted series Re/Le from the first R,L,C row",
    )
    parser.add_argument(
        "--details",
        action="store_true",
        help="also write *_details.csv with element names, f0 and Q",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="save measured/model comparison plot",
    )
    parser.add_argument(
        "--plot-file",
        type=Path,
        default=DEFAULT_PLOT,
        help=f"plot output path, default: {DEFAULT_PLOT}",
    )
    parser.add_argument(
        "--max-evaluations",
        type=int,
        default=6000,
        help="optimizer evaluation budget, default: 6000",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.sections is not None and args.sections < 0:
        parser.error("--sections must be non-negative")
    if args.min_sections < 0:
        parser.error("--min-sections must be non-negative")
    if args.max_sections < args.min_sections:
        parser.error("--max-sections must be greater than or equal to --min-sections")
    if args.sections is not None and args.auto_sections:
        parser.error("--sections and --auto-sections are mutually exclusive")
    if args.max_evaluations < 100:
        parser.error("--max-evaluations must be at least 100")

    freq, z_measured = load_impedance_file(args.input_file)
    candidates: list[FitResult] = []
    if args.sections is None or args.auto_sections:
        result, candidates = fit_impedance_auto(
            freq,
            z_measured,
            min_sections=args.min_sections,
            max_sections=args.max_sections,
            max_evaluations=args.max_evaluations,
        )
    else:
        result = fit_impedance(
            freq,
            z_measured,
            args.sections,
            max_evaluations=args.max_evaluations,
        )

    save_result_csv(
        args.out,
        result.physical_params,
        result.sections,
        include_series=not args.no_series,
    )
    if args.details:
        save_diagnostic_csv(args.out, result.physical_params, result.sections)
    if args.plot:
        plot_fit(args.plot_file, freq, z_measured, result.physical_params, result.sections)

    Re, Le = result.physical_params[:2]
    print(f"Input: {args.input_file}")
    print(f"Output: {args.out}")
    if candidates:
        print(f"Auto sections range: {args.min_sections}..{args.max_sections}")
        print("Auto candidates: sections,rms_log_error,bic")
        for candidate in candidates:
            print(
                f"  {candidate.sections},"
                f"{_format_float(candidate.rms_log_error)},"
                f"{_format_float(candidate.selection_score)}"
            )
    print(f"Sections: {result.sections}")
    print("Optimizer: scipy.optimize.least_squares")
    print(f"Series Re: {_format_float(float(Re))} Ohm")
    print(f"Series Le: {_format_float(float(Le))} H")
    print(f"RMS log error: {_format_float(result.rms_log_error)}")
    print(f"Max abs log error: {_format_float(result.max_abs_log_error)}")
    if args.plot:
        print(f"Plot: {args.plot_file}")


if __name__ == "__main__":
    main()
