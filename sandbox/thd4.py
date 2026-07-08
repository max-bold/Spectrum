from __future__ import annotations

from pathlib import Path
import sys
from typing import TypedDict


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) in sys.path:
    sys.path.remove(str(SCRIPT_DIR))

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, NullFormatter
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import curve_fit
from scipy.signal import ShortTimeFFT


SAMPLE_RATE = 96_000
DURATION = 30.0
BAND = (20.0, 20_000.0)
SWEEP_BAND_EXPANSION = 1.5
FADE_SECONDS = 0.5
SEGMENT_SECONDS = 1.0
HOP_SECONDS = 0.1
FALLBACK_MASK_RATIO = 5.0 / 4.0
MASK_EXPANSION = 2.0
HARMONIC_COUNT = 4
HARMONIC_DECAY = 0.1
PINK_NOISE_RATIO = 0.01
SEED = 12_345
ANIMATION_STRIDE = 1
REFERENCE_SMOOTHING_OCTAVES = 1.0 / 12.0
THD_SMOOTHING_OCTAVES = 1.0 / 24.0
REFERENCE_POINTS = 1_200
THD_PERCENT_LIMITS = (0.01, 10.0)
OUTPUT_DIR = Path("artifacts")


class Metrics(TypedDict):
    """Container for scalar and per-frame THD+N metrics."""

    frequency: NDArray[np.float64]
    main_energy: NDArray[np.float64]
    residual_energy: NDArray[np.float64]
    centers: NDArray[np.float64]
    main_by_frame: NDArray[np.float64]
    residual_by_frame: NDArray[np.float64]
    measured_ratio: float
    leakage_ratio: float
    oracle_ratio: float
    injected_ratio: float
    injected_components_ratio: float


class MaskCalibration(TypedDict):
    """Per-frame frequency masks calibrated on a clean sweep."""

    centers: NDArray[np.float64]
    left_edges: NDArray[np.float64]
    right_edges: NDArray[np.float64]


class MaskFit(TypedDict):
    """Parametric left/right A-mask width functions."""

    left_params: NDArray[np.float64]
    right_params: NDArray[np.float64]


def sweep_band() -> tuple[float, float]:
    """Return the wider generation band used to keep fade outside BAND."""
    return BAND[0] / SWEEP_BAND_EXPANSION, BAND[1] * SWEEP_BAND_EXPANSION


def log_chirp(
    *,
    amplitude: float = 0.5,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Generate a logarithmic sine sweep and its instantaneous frequency."""
    time = np.arange(int(round(SAMPLE_RATE * DURATION)), dtype=np.float64) / SAMPLE_RATE
    fade_size = int(round(FADE_SECONDS * SAMPLE_RATE))
    f_start, f_stop = sweep_band()
    sweep_rate = DURATION / np.log(f_stop / f_start)
    phase = 2.0 * np.pi * f_start * sweep_rate * (np.exp(time / sweep_rate) - 1.0)
    frequency = f_start * np.exp(time / sweep_rate)
    sweep = amplitude * np.sin(phase)

    if fade_size > 0:
        fade = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, fade_size))
        sweep[:fade_size] *= fade
        sweep[-fade_size:] *= fade[::-1]

    return time, sweep, frequency


def pink_noise(
    sample_count: int,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Generate FFT-shaped pink noise with arbitrary RMS."""
    spectrum = np.fft.rfft(rng.standard_normal(sample_count))
    frequency = np.fft.rfftfreq(sample_count)
    scale = np.ones_like(frequency)
    scale[1:] = 1.0 / np.sqrt(frequency[1:])
    scale[0] = 0.0
    noise = np.fft.irfft(spectrum * scale, n=sample_count)
    return noise - np.mean(noise)


def rms(signal: NDArray[np.float64]) -> float:
    """Return RMS value of a real signal."""
    return float(np.sqrt(np.mean(np.square(signal))))


def build_test_signal() -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Build fundamental, polynomial distortion, pink noise, and their sum."""
    rng = np.random.default_rng(SEED)
    time, fundamental, _ = log_chirp()

    distortion = np.zeros_like(fundamental)
    for order in range(2, HARMONIC_COUNT + 2):
        distortion += fundamental**order * HARMONIC_DECAY ** (order - 1)

    noise = pink_noise(len(fundamental), rng)
    noise *= PINK_NOISE_RATIO * rms(fundamental) / rms(noise)

    signal = fundamental + distortion + noise
    return time, fundamental, distortion, noise, signal


def build_stft() -> ShortTimeFFT:
    """Create the STFT transform used by all frame-based calculations."""
    segment_size = int(round(SEGMENT_SECONDS * SAMPLE_RATE))
    hop_size = int(round(HOP_SECONDS * SAMPLE_RATE))
    window = np.hanning(segment_size)
    return ShortTimeFFT(
        window,
        hop=hop_size,
        fs=SAMPLE_RATE,
        fft_mode="onesided",
        mfft=segment_size,
    )


def stft_power(
    signal: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return STFT frequencies, frame center times, and per-bin power."""
    transform = build_stft()
    stft = transform.stft(signal)
    power = np.square(np.abs(stft.T))
    return (
        np.asarray(transform.f, dtype=np.float64),
        np.asarray(transform.t(len(signal)), dtype=np.float64),
        np.asarray(power, dtype=np.float64),
    )


def stft_frame_times(sample_count: int) -> NDArray[np.float64]:
    """Return frame center times produced by ShortTimeFFT for a signal length."""
    return np.asarray(build_stft().t(sample_count), dtype=np.float64)


def calibrate_masks_from_clean_sweep(
    clean_signal: NDArray[np.float64],
) -> MaskCalibration:
    """Find empirical per-frame A-mask edges from a clean sweep STFT."""
    frequency, _frame_time, spectra = stft_power(clean_signal)
    first_frequency_index = 1
    last_frequency_index = len(frequency) - 1

    centers = np.empty(len(spectra), dtype=np.float64)
    left_edges = np.empty(len(spectra), dtype=np.float64)
    right_edges = np.empty(len(spectra), dtype=np.float64)

    for frame_index, power in enumerate(spectra):
        peak_index = first_frequency_index + int(np.argmax(power[first_frequency_index:]))
        threshold = float(np.mean(power))

        left_index = peak_index
        while left_index > first_frequency_index and power[left_index] > threshold:
            left_index -= 1

        right_index = peak_index
        while right_index < last_frequency_index and power[right_index] > threshold:
            right_index += 1

        center = float(frequency[peak_index])
        left = float(frequency[left_index])
        right = float(frequency[right_index])

        centers[frame_index] = center
        left_edges[frame_index] = max(BAND[0], center - MASK_EXPANSION * (center - left))
        right_edges[frame_index] = min(BAND[1], center + MASK_EXPANSION * (right - center))

    return {
        "centers": centers,
        "left_edges": left_edges,
        "right_edges": right_edges,
    }


def reciprocal_log_model(
    frequency: NDArray[np.float64],
    a: float,
    b: float,
    c: float,
) -> NDArray[np.float64]:
    """Evaluate a / log(b*f) + c for mask-width fitting."""
    return a / np.log(b * frequency) + c


def fit_mask_width_functions(mask_calibration: MaskCalibration) -> MaskFit:
    """Fit smooth left/right mask-width functions from empirical edges."""
    centers = mask_calibration["centers"]
    left_ratio = centers / mask_calibration["left_edges"]
    right_ratio = mask_calibration["right_edges"] / centers
    left_fit_mask = (centers >= BAND[0] * 2.0) & (centers <= BAND[1])
    right_fit_mask = (centers >= BAND[0]) & (centers <= BAND[1] / 2.0)

    lower_bounds = (0.0, 1.0 / BAND[0] + 1e-9, 1.0)
    upper_bounds = (10.0, 10.0, 3.0)
    initial_guess = (1.0, 1.0, 1.05)

    left_params, _ = curve_fit(
        reciprocal_log_model,
        centers[left_fit_mask],
        left_ratio[left_fit_mask],
        p0=initial_guess,
        bounds=(lower_bounds, upper_bounds),
        maxfev=20_000,
    )
    right_params, _ = curve_fit(
        reciprocal_log_model,
        centers[right_fit_mask],
        right_ratio[right_fit_mask],
        p0=initial_guess,
        bounds=(lower_bounds, upper_bounds),
        maxfev=20_000,
    )

    return {
        "left_params": np.asarray(left_params, dtype=np.float64),
        "right_params": np.asarray(right_params, dtype=np.float64),
    }


def w_left(frequency: float | NDArray[np.float64], mask_fit: MaskFit) -> float | NDArray[np.float64]:
    """Return fitted left mask ratio center/left_edge."""
    return reciprocal_log_model(np.asarray(frequency), *mask_fit["left_params"])


def w_right(frequency: float | NDArray[np.float64], mask_fit: MaskFit) -> float | NDArray[np.float64]:
    """Return fitted right mask ratio right_edge/center."""
    return reciprocal_log_model(np.asarray(frequency), *mask_fit["right_params"])


def fitted_mask_edges(
    frequency: float,
    mask_fit: MaskFit,
) -> tuple[float, float]:
    """Return fitted left/right mask edges for a tracked frequency."""
    left_ratio = float(w_left(frequency, mask_fit))
    right_ratio = float(w_right(frequency, mask_fit))
    left_edge = max(BAND[0], frequency / left_ratio)
    right_edge = min(BAND[1], frequency * right_ratio)
    return left_edge, right_edge


def stft_frame_analysis(
    signal: NDArray[np.float64],
    *,
    center_from: NDArray[np.float64] | None = None,
    mask_fit: MaskFit | None = None,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.bool_],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Split each STFT frame into fundamental-mask energy and residual energy."""
    frequency, _frame_time, spectra = stft_power(signal)
    band_mask = (frequency >= BAND[0]) & (frequency <= BAND[1])

    if center_from is None:
        source_spectra = spectra
    else:
        _source_frequency, _source_frame_time, source_spectra = stft_power(center_from)

    masks = np.empty((len(spectra), len(frequency)), dtype=np.bool_)
    centers = np.empty(len(spectra), dtype=np.float64)
    main_by_frame = np.zeros_like(spectra)
    residual_by_frame = np.zeros_like(spectra)

    for frame_index, power in enumerate(spectra):
        source_power = source_spectra[frame_index]
        peak_index = 1 + int(np.argmax(source_power[1:]))
        center_frequency = float(frequency[peak_index])

        if mask_fit is None:
            left_edge = center_frequency / FALLBACK_MASK_RATIO
            right_edge = center_frequency * FALLBACK_MASK_RATIO
        else:
            left_edge, right_edge = fitted_mask_edges(center_frequency, mask_fit)

        main_mask = (
            band_mask
            & (frequency >= left_edge)
            & (frequency <= right_edge)
        )

        masks[frame_index] = main_mask
        centers[frame_index] = center_frequency
        main_by_frame[frame_index, main_mask] = power[main_mask]
        residual_by_frame[frame_index, band_mask & ~main_mask] = power[
            band_mask & ~main_mask
        ]

    return frequency, centers, spectra, masks, main_by_frame, residual_by_frame


def stft_energy_split(
    signal: NDArray[np.float64],
    *,
    center_from: NDArray[np.float64] | None = None,
    mask_fit: MaskFit | None = None,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Return A/B spectra plus per-frame A/B spectra."""
    frequency, centers, _spectra, _masks, main_by_frame, residual_by_frame = (
        stft_frame_analysis(
            signal,
            center_from=center_from,
            mask_fit=mask_fit,
        )
    )
    main_energy = np.sum(main_by_frame, axis=0)
    residual_energy = np.sum(residual_by_frame, axis=0)
    return (
        main_energy,
        residual_energy,
        centers,
        main_by_frame,
        residual_by_frame,
        frequency,
    )


def fft_residual_ratio(
    fundamental: NDArray[np.float64],
    residual: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return direct FFT residual/fundamental amplitude ratio."""
    frequency = np.fft.rfftfreq(len(fundamental), 1.0 / SAMPLE_RATE)
    fundamental_power = np.abs(np.fft.rfft(fundamental)) ** 2
    residual_power = np.abs(np.fft.rfft(residual)) ** 2
    ratio = np.sqrt(residual_power / np.maximum(fundamental_power, 1e-30))
    band_mask = (frequency >= BAND[0]) & (frequency <= BAND[1])
    return frequency[band_mask], ratio[band_mask]


def smooth_log_ratio(
    frequency: NDArray[np.float64],
    ratio: NDArray[np.float64],
    *,
    smoothing_octaves: float = REFERENCE_SMOOTHING_OCTAVES,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Smooth an amplitude ratio by averaging its power in log-frequency bins."""
    output_frequency = np.geomspace(BAND[0], BAND[1], REFERENCE_POINTS)
    log_frequency = np.log2(frequency)
    log_output = np.log2(output_frequency)
    half_width = smoothing_octaves / 2.0

    ratio_power = np.square(ratio)
    cumulative = np.concatenate(([0.0], np.cumsum(ratio_power)))
    left = np.searchsorted(log_frequency, log_output - half_width)
    right = np.searchsorted(log_frequency, log_output + half_width)
    right = np.maximum(right, left + 1)

    mean_power = (cumulative[right] - cumulative[left]) / (right - left)
    return output_frequency, np.sqrt(mean_power)


def ratio_db(value: float) -> float:
    """Convert an amplitude ratio to dB."""
    return 20.0 * np.log10(max(value, 1e-20))


def ratio_percent(ratio: float | NDArray[np.float64]) -> NDArray[np.float64]:
    """Return clipped ratio percent values for log-scale THD+N plots."""
    values = np.asarray(ratio, dtype=np.float64) * 100.0
    return np.clip(values, THD_PERCENT_LIMITS[0], THD_PERCENT_LIMITS[1])


def power_ratio(
    residual_energy: NDArray[np.float64],
    main_energy: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return amplitude ratio from residual and main energy vectors."""
    return np.sqrt(residual_energy / np.maximum(main_energy, 1e-30))


def set_thd_axis(axis: plt.Axes) -> None:
    """Use a fixed logarithmic percent scale for THD+N plots."""
    axis.set_yscale("log")
    axis.set_ylim(THD_PERCENT_LIMITS)
    axis.yaxis.set_major_locator(FixedLocator([0.01, 0.1, 1.0, 10.0]))
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value:g}%"))
    axis.yaxis.set_minor_locator(
        LogLocator(base=10.0, subs=(0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9))
    )
    axis.yaxis.set_minor_formatter(NullFormatter())


def print_summary(
    *,
    label: str,
    injected_ratio: float,
    injected_components_ratio: float,
    measured_ratio: float,
    leakage_ratio: float,
    oracle_ratio: float,
) -> None:
    """Print one experiment summary."""
    overlap = 1.0 - HOP_SECONDS / SEGMENT_SECONDS
    print(label)
    print(f"fs={SAMPLE_RATE} Hz, T={DURATION:g} s, band={BAND}")
    print(
        f"segment={SEGMENT_SECONDS:g} s, hop={HOP_SECONDS:g} s, "
        f"overlap={overlap:.0%}, mask ratio=adaptive"
    )
    print(
        "Injected RMS ratio, with cross terms: "
        f"{injected_ratio * 100:.3f}% ({ratio_db(injected_ratio):.2f} dB)"
    )
    print(
        "Injected component RMS ratio:        "
        f"{injected_components_ratio * 100:.3f}% "
        f"({ratio_db(injected_components_ratio):.2f} dB)"
    )
    print(
        "STFT split THD+N estimate:           "
        f"{measured_ratio * 100:.3f}% ({ratio_db(measured_ratio):.2f} dB)"
    )
    print(
        "Clean chirp leakage floor:            "
        f"{leakage_ratio * 100:.3f}% ({ratio_db(leakage_ratio):.2f} dB)"
    )
    print(
        "Oracle split of known dist+noise:     "
        f"{oracle_ratio * 100:.3f}% ({ratio_db(oracle_ratio):.2f} dB)"
    )
    print()


def compute_metrics(
    *,
    fundamental: NDArray[np.float64],
    distortion: NDArray[np.float64],
    noise: NDArray[np.float64],
    signal: NDArray[np.float64],
    mask_fit: MaskFit,
) -> Metrics:
    """Compute all scalar and per-frame values used by plots."""
    main_energy, residual_energy, centers, main_by_frame, residual_by_frame, frequency = (
        stft_energy_split(
            signal,
            mask_fit=mask_fit,
        )
    )
    clean_main, clean_residual, *_ = stft_energy_split(
        fundamental,
        mask_fit=mask_fit,
    )
    oracle_residual = stft_energy_split(
        distortion + noise,
        center_from=fundamental,
        mask_fit=mask_fit,
    )[1]
    oracle_main = stft_energy_split(
        fundamental,
        center_from=fundamental,
        mask_fit=mask_fit,
    )[0]

    injected_ratio = rms(distortion + noise) / rms(fundamental)
    injected_components_ratio = np.sqrt(rms(distortion) ** 2 + rms(noise) ** 2) / rms(
        fundamental
    )

    return {
        "frequency": frequency,
        "main_energy": main_energy,
        "residual_energy": residual_energy,
        "centers": centers,
        "main_by_frame": main_by_frame,
        "residual_by_frame": residual_by_frame,
        "measured_ratio": float(np.sqrt(np.sum(residual_energy) / np.sum(main_energy))),
        "leakage_ratio": float(np.sqrt(np.sum(clean_residual) / np.sum(clean_main))),
        "oracle_ratio": float(np.sqrt(np.sum(oracle_residual) / np.sum(oracle_main))),
        "injected_ratio": float(injected_ratio),
        "injected_components_ratio": float(injected_components_ratio),
    }


def save_summary_plot(
    *,
    output_path: Path,
    time: NDArray[np.float64],
    fundamental: NDArray[np.float64],
    distortion: NDArray[np.float64],
    noise: NDArray[np.float64],
    signal: NDArray[np.float64],
    metrics: Metrics,
) -> None:
    """Save the signal, tracked frequency, and THD+N comparison plot."""
    frequency = metrics["frequency"]
    centers = metrics["centers"]
    residual_energy = metrics["residual_energy"]
    main_energy = metrics["main_energy"]
    measured_ratio = metrics["measured_ratio"]

    frame_time = stft_frame_times(len(signal))[: len(centers)]
    band_mask = (frequency >= BAND[0]) & (frequency <= BAND[1])
    stft_ratio = power_ratio(residual_energy, main_energy)
    stft_frequency, stft_ratio = smooth_log_ratio(
        frequency[band_mask],
        stft_ratio[band_mask],
        smoothing_octaves=THD_SMOOTHING_OCTAVES,
    )
    fft_frequency, fft_ratio = fft_residual_ratio(
        fundamental,
        distortion + noise,
    )
    fft_frequency, fft_ratio = smooth_log_ratio(
        fft_frequency,
        fft_ratio,
    )

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), constrained_layout=True)

    axes[0].plot(time, fundamental, label="fundamental", linewidth=0.8)
    axes[0].plot(time, signal, label="signal + distortion + noise", linewidth=0.5, alpha=0.8)
    axes[0].set_xlim(0.0, min(1.0, DURATION))
    axes[0].set_title("First second of generated signal")
    axes[0].set_xlabel("Time, s")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper left")

    axes[1].semilogy(frame_time, centers)
    axes[1].set_title("Tracked fundamental from STFT peak")
    axes[1].set_xlabel("Time, s")
    axes[1].set_ylabel("Frequency, Hz")
    axes[1].grid(True, which="both", alpha=0.3)

    axes[2].semilogx(
        stft_frequency,
        ratio_percent(stft_ratio),
        label="accumulated STFT split",
    )
    axes[2].semilogx(
        fft_frequency,
        ratio_percent(fft_ratio),
        color="0.35",
        linewidth=0.8,
        alpha=0.65,
        label="rfft residual/fundamental",
    )
    axes[2].axhline(
        float(ratio_percent(measured_ratio)),
        color="tab:green",
        linestyle=":",
        label="integrated STFT split",
    )
    axes[2].set_title("THD+N estimate by instantaneous frequency")
    axes[2].set_xlabel("Frequency, Hz")
    axes[2].set_ylabel("THD+N, %")
    axes[2].set_xlim(BAND)
    set_thd_axis(axes[2])
    axes[2].grid(True, which="both", alpha=0.3)
    axes[2].legend(loc="upper right")

    output_path.parent.mkdir(exist_ok=True)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    print(f"Saved plot: {output_path}", flush=True)


def save_energy_plot(
    *,
    output_path: Path,
    metrics: Metrics,
) -> None:
    """Save the exact A/B energies returned by stft_energy_split."""
    frequency = metrics["frequency"]
    main_energy = metrics["main_energy"]
    residual_energy = metrics["residual_energy"]
    band_mask = (frequency >= BAND[0]) & (frequency <= BAND[1])
    band_frequency = frequency[band_mask]
    band_main = main_energy[band_mask]
    band_residual = residual_energy[band_mask]

    cumulative_main = np.cumsum(band_main)
    cumulative_residual = np.cumsum(band_residual)
    stft_ratio = power_ratio(band_residual, band_main)
    stft_frequency, stft_ratio = smooth_log_ratio(
        band_frequency,
        stft_ratio,
        smoothing_octaves=THD_SMOOTHING_OCTAVES,
    )
    cumulative_ratio = power_ratio(cumulative_residual, cumulative_main)

    total_main_energy = float(np.sum(band_main))
    total_residual_energy = float(np.sum(band_residual))
    main_peak = float(np.max(band_main))
    main_spectrum_db = 10.0 * np.log10(band_main / main_peak + 1e-14)
    residual_spectrum_db = 10.0 * np.log10(band_residual / main_peak + 1e-14)
    cumulative_main_db = 10.0 * np.log10(cumulative_main / total_main_energy + 1e-14)
    cumulative_residual_db = 10.0 * np.log10(
        cumulative_residual / total_main_energy + 1e-14
    )

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), constrained_layout=True, sharex=True)

    axes[0].semilogx(band_frequency, main_spectrum_db, color="tab:green", linewidth=1.1)
    axes[0].semilogx(band_frequency, residual_spectrum_db, color="tab:red", linewidth=1.1)
    axes[0].set_title("Accumulated A(f) and B(f) returned by stft_energy_split")
    axes[0].set_ylabel("Energy, dB rel. max A")
    axes[0].set_xlim(BAND)
    axes[0].set_ylim(-80.0, 5.0)
    axes[0].legend(["A(f)", "B(f)"], loc="upper right")
    axes[0].grid(True, which="both", alpha=0.3)

    axes[1].semilogx(band_frequency, cumulative_main_db, color="tab:green", linewidth=1.1)
    axes[1].semilogx(
        band_frequency,
        cumulative_residual_db,
        color="tab:red",
        linewidth=1.1,
    )
    axes[1].axhline(0.0, color="tab:green", linestyle=":", linewidth=0.9)
    axes[1].axhline(
        10.0 * np.log10(total_residual_energy / total_main_energy),
        color="tab:red",
        linestyle=":",
        linewidth=0.9,
    )
    axes[1].set_title("Cumulative A(f) and B(f) over frequency")
    axes[1].set_ylabel("Cumulative energy, dB rel. final A")
    axes[1].set_xlim(BAND)
    axes[1].set_ylim(-80.0, 5.0)
    axes[1].legend(["A cumulative", "B cumulative"], loc="lower right")
    axes[1].grid(True, which="both", alpha=0.3)

    axes[2].semilogx(
        stft_frequency,
        ratio_percent(stft_ratio),
        color="tab:blue",
        linewidth=1.0,
    )
    axes[2].semilogx(
        band_frequency,
        ratio_percent(cumulative_ratio),
        color="tab:purple",
        linewidth=1.2,
    )
    axes[2].axhline(
        float(ratio_percent(np.sqrt(total_residual_energy / total_main_energy))),
        color="tab:purple",
        linestyle=":",
        linewidth=0.9,
    )
    axes[2].set_title("Ratios from the same energies")
    axes[2].set_xlabel("Frequency, Hz")
    axes[2].set_ylabel("sqrt(residual/main), %")
    axes[2].set_xlim(BAND)
    set_thd_axis(axes[2])
    axes[2].legend(["B(f) / A(f)", "cumulative"], loc="upper right")
    axes[2].grid(True, which="both", alpha=0.3)

    output_path.parent.mkdir(exist_ok=True)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    print(f"Saved A/B energy plot: {output_path}", flush=True)


def save_mask_width_plot(
    *,
    output_path: Path,
    mask_calibration: MaskCalibration,
    mask_fit: MaskFit,
) -> None:
    """Save empirical A-mask width as a function of tracked frequency."""
    centers = mask_calibration["centers"]
    left_edges = mask_calibration["left_edges"]
    right_edges = mask_calibration["right_edges"]

    left_ratio = centers / left_edges
    right_ratio = right_edges / centers
    fitted_left_ratio = w_left(centers, mask_fit)
    fitted_right_ratio = w_right(centers, mask_fit)
    fitted_left_edges = centers / fitted_left_ratio
    fitted_right_edges = centers * fitted_right_ratio
    fitted_mask = (
        (centers >= BAND[0])
        & (centers <= BAND[1])
        & np.isfinite(fitted_left_edges)
        & np.isfinite(fitted_right_edges)
        & (fitted_left_edges > 0.0)
        & (fitted_right_edges > fitted_left_edges)
    )
    fitted_full_octaves = np.log2(
        fitted_right_edges[fitted_mask] / fitted_left_edges[fitted_mask]
    )
    fitted_full_percent = (
        (fitted_right_edges[fitted_mask] - fitted_left_edges[fitted_mask])
        / centers[fitted_mask]
        * 100.0
    )
    full_octaves = np.log2(right_edges / left_edges)
    full_percent = (right_edges - left_edges) / centers * 100.0

    edge_min = float(np.min(left_edges) / 1.08)
    edge_max = float(np.max(right_edges) * 1.08)
    ratio_min = float(min(np.min(left_ratio), np.min(right_ratio)) * 0.96)
    ratio_max = float(max(np.max(left_ratio), np.max(right_ratio)) * 1.04)
    octave_min = float(np.min(full_octaves) * 0.96)
    octave_max = float(np.max(full_octaves) * 1.04)
    percent_min = float(np.min(full_percent) * 0.96)
    percent_max = float(np.max(full_percent) * 1.04)

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), constrained_layout=True, sharex=True)

    axes[0].semilogx(centers, left_edges, color="tab:blue", linewidth=1.0)
    axes[0].semilogx(centers, right_edges, color="tab:orange", linewidth=1.0)
    axes[0].semilogx(
        centers[fitted_mask],
        fitted_left_edges[fitted_mask],
        color="tab:blue",
        linestyle="--",
        linewidth=1.0,
    )
    axes[0].semilogx(
        centers[fitted_mask],
        fitted_right_edges[fitted_mask],
        color="tab:orange",
        linestyle="--",
        linewidth=1.0,
    )
    axes[0].semilogx(centers, centers, color="0.25", linestyle=":", linewidth=0.9)
    axes[0].set_title("Empirical A-mask edges from clean sweep")
    axes[0].set_ylabel("Frequency, Hz")
    axes[0].set_xlim(BAND)
    axes[0].set_ylim(edge_min, edge_max)
    axes[0].set_yscale("log")
    axes[0].legend(
        ["left edge", "right edge", "left fit", "right fit", "center"],
        loc="upper left",
    )
    axes[0].grid(True, which="both", alpha=0.3)

    axes[1].semilogx(centers, left_ratio, color="tab:blue", linewidth=1.0)
    axes[1].semilogx(centers, right_ratio, color="tab:orange", linewidth=1.0)
    axes[1].semilogx(
        centers[fitted_mask],
        fitted_left_ratio[fitted_mask],
        color="tab:blue",
        linestyle="--",
        linewidth=1.0,
    )
    axes[1].semilogx(
        centers[fitted_mask],
        fitted_right_ratio[fitted_mask],
        color="tab:orange",
        linestyle="--",
        linewidth=1.0,
    )
    axes[1].set_title("Mask half-width ratios")
    axes[1].set_ylabel("Ratio")
    axes[1].set_xlim(BAND)
    axes[1].set_ylim(ratio_min, ratio_max)
    axes[1].legend(
        ["center / left", "right / center", "left fit", "right fit"],
        loc="upper right",
    )
    axes[1].grid(True, which="both", alpha=0.3)

    axes[2].semilogx(centers, full_octaves, color="tab:green", linewidth=1.0)
    axes[2].semilogx(
        centers[fitted_mask],
        fitted_full_octaves,
        color="tab:green",
        linestyle="--",
        linewidth=1.0,
    )
    axes[2].set_title("Full mask width")
    axes[2].set_xlabel("Tracked fundamental, Hz")
    axes[2].set_ylabel("Octaves")
    axes[2].set_xlim(BAND)
    axes[2].set_ylim(octave_min, octave_max)
    axes[2].grid(True, which="both", alpha=0.3)

    twin = axes[2].twinx()
    twin.semilogx(centers, full_percent, color="tab:red", alpha=0.45, linewidth=0.8)
    twin.semilogx(
        centers[fitted_mask],
        fitted_full_percent,
        color="tab:red",
        linestyle="--",
        alpha=0.6,
        linewidth=0.8,
    )
    twin.set_ylabel("Width / center, %")
    twin.set_ylim(percent_min, percent_max)

    output_path.parent.mkdir(exist_ok=True)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    print(f"Saved mask width plot: {output_path}", flush=True)


def save_energy_accumulation_animation(
    *,
    output_path: Path,
    signal: NDArray[np.float64],
    fundamental: NDArray[np.float64],
    distortion: NDArray[np.float64],
    noise: NDArray[np.float64],
    mask_fit: MaskFit,
) -> None:
    """Save a GIF of SFFT frames and A/B energy accumulation."""
    frequency, centers, spectra, _masks, main_by_frame, residual_by_frame = (
        stft_frame_analysis(signal, mask_fit=mask_fit)
    )

    band_mask = (frequency >= BAND[0]) & (frequency <= BAND[1])
    band_frequency = frequency[band_mask]
    band_spectra = spectra[:, band_mask]
    band_main_by_frame = main_by_frame[:, band_mask]
    band_residual_by_frame = residual_by_frame[:, band_mask]
    cumulative_main_spectra = np.cumsum(band_main_by_frame, axis=0)
    cumulative_residual_spectra = np.cumsum(band_residual_by_frame, axis=0)
    cumulative_ratio_spectra = power_ratio(
        cumulative_residual_spectra,
        cumulative_main_spectra,
    )
    thd_frequency = np.geomspace(BAND[0], BAND[1], REFERENCE_POINTS)
    smoothed_ratio_spectra = np.empty(
        (len(cumulative_ratio_spectra), len(thd_frequency)),
        dtype=np.float64,
    )
    for frame_index, ratio_spectrum in enumerate(cumulative_ratio_spectra):
        _frequency, smoothed_ratio_spectra[frame_index] = smooth_log_ratio(
            band_frequency,
            ratio_spectrum,
            smoothing_octaves=THD_SMOOTHING_OCTAVES,
        )

    reference_frequency, reference_ratio = fft_residual_ratio(
        fundamental,
        distortion + noise,
    )
    reference_frequency, reference_ratio = smooth_log_ratio(
        reference_frequency,
        reference_ratio,
    )

    cumulative_spectrum_correction = band_frequency / 1000.0
    final_main_display_spectrum = (
        cumulative_main_spectra[-1] * cumulative_spectrum_correction
    )
    final_main_peak = float(np.max(final_main_display_spectrum))
    final_main_energy = float(np.sum(cumulative_main_spectra[-1]))
    final_residual_energy = float(np.sum(cumulative_residual_spectra[-1]))
    final_ratio = np.sqrt(final_residual_energy / final_main_energy)
    pink_correction = band_frequency / 1000.0
    display_spectra = band_spectra * pink_correction[None, :]
    db_spectra = 10.0 * np.log10(display_spectra / np.max(display_spectra) + 1e-14)

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), constrained_layout=True)
    ax_spectrum, ax_energy, ax_ratio = axes

    (spectrum_line,) = ax_spectrum.semilogx(
        band_frequency,
        db_spectra[0],
        color="tab:blue",
        linewidth=1.1,
    )
    first_left, first_right = fitted_mask_edges(float(centers[0]), mask_fit)
    mask_span = ax_spectrum.axvspan(
        first_left,
        first_right,
        color="tab:green",
        alpha=0.28,
    )
    peak_line = ax_spectrum.axvline(centers[0], color="tab:red", linewidth=1.2)
    ax_spectrum.set_title("Current SFFT spectrum: green = A, outside = B")
    ax_spectrum.set_ylabel("Pink-corrected power, dB rel.")
    ax_spectrum.set_xlim(BAND)
    ax_spectrum.set_ylim(-95.0, 2.0)
    ax_spectrum.grid(True, which="both", alpha=0.3)

    (main_sum_line,) = ax_energy.semilogx([], [], color="tab:green", linewidth=1.1)
    (residual_sum_line,) = ax_energy.semilogx([], [], color="tab:red", linewidth=1.1)
    ax_energy.set_title("Cumulative spectra: sum(A[0:n]) and sum(B[0:n])")
    ax_energy.set_ylabel("Energy * f, dB rel. final max A")
    ax_energy.set_xlim(BAND)
    ax_energy.set_ylim(-80.0, 5.0)
    ax_energy.legend(["sum A", "sum B"], loc="upper right")
    ax_energy.grid(True, which="both", alpha=0.3)

    (reference_line,) = ax_ratio.semilogx(
        reference_frequency,
        ratio_percent(reference_ratio),
        color="0.45",
        linewidth=0.75,
        alpha=0.65,
    )
    (ratio_line,) = ax_ratio.semilogx([], [], color="tab:blue", linewidth=1.0)
    ax_ratio.axhline(
        float(ratio_percent(final_ratio)),
        color="tab:purple",
        linestyle=":",
        linewidth=0.9,
    )
    ax_ratio.set_title("Cumulative THD+N spectrum")
    ax_ratio.set_xlabel("Frequency, Hz")
    ax_ratio.set_ylabel("sqrt(sum B / sum A), %")
    ax_ratio.set_xlim(BAND)
    set_thd_axis(ax_ratio)
    ax_ratio.legend(
        [ratio_line, reference_line],
        ["sqrt(sum B / sum A)", "rfft residual/fundamental"],
        loc="upper right",
    )
    ax_ratio.grid(True, which="both", alpha=0.3)

    title = fig.suptitle("", fontsize=12)

    def update(frame_index: int):
        current_frequency = centers[frame_index]
        current_main_energy = float(np.sum(cumulative_main_spectra[frame_index]))
        current_residual_energy = float(np.sum(cumulative_residual_spectra[frame_index]))
        current_ratio = np.sqrt(
            current_residual_energy / max(current_main_energy, 1e-30)
        )
        title.set_text(
            f"frame={frame_index + 1}/{len(centers)}, "
            f"f0={current_frequency:.1f} Hz, "
            f"cumulative THD+N={current_ratio * 100.0:.3f}%"
        )

        spectrum_line.set_ydata(db_spectra[frame_index])
        peak_line.set_xdata([current_frequency, current_frequency])
        mask_left, mask_right = fitted_mask_edges(float(current_frequency), mask_fit)
        mask_span.set_x(mask_left)
        mask_span.set_width(mask_right - mask_left)

        main_sum_display = (
            cumulative_main_spectra[frame_index] * cumulative_spectrum_correction
        )
        residual_sum_display = (
            cumulative_residual_spectra[frame_index] * cumulative_spectrum_correction
        )
        main_sum_db = 10.0 * np.log10(
            main_sum_display / final_main_peak + 1e-14
        )
        residual_sum_db = 10.0 * np.log10(
            residual_sum_display / final_main_peak + 1e-14
        )
        main_sum_line.set_data(band_frequency, main_sum_db)
        residual_sum_line.set_data(band_frequency, residual_sum_db)

        ratio_line.set_data(
            thd_frequency,
            ratio_percent(smoothed_ratio_spectra[frame_index]),
        )
        return (
            title,
            spectrum_line,
            peak_line,
            mask_span,
            main_sum_line,
            residual_sum_line,
            ratio_line,
        )

    frame_indices = np.arange(0, len(centers), ANIMATION_STRIDE, dtype=int)
    if frame_indices[-1] != len(centers) - 1:
        frame_indices = np.append(frame_indices, len(centers) - 1)
    fps = max(1, round(1.0 / (HOP_SECONDS * ANIMATION_STRIDE)))
    animation = FuncAnimation(
        fig,
        update,
        frames=frame_indices,
        interval=HOP_SECONDS * ANIMATION_STRIDE * 1000.0,
        blit=False,
    )
    output_path.parent.mkdir(exist_ok=True)
    animation.save(output_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"Saved A/B accumulation animation: {output_path}", flush=True)


def save_realtime_stft_animation(
    *,
    output_path: Path,
    signal: NDArray[np.float64],
    fundamental: NDArray[np.float64],
    distortion: NDArray[np.float64],
    noise: NDArray[np.float64],
    mask_fit: MaskFit,
) -> None:
    """Save a GIF of the current STFT spectrum and frame THD+N estimate."""
    frequency, centers, spectra, _masks, main_by_frame, residual_by_frame = (
        stft_frame_analysis(
            signal,
            mask_fit=mask_fit,
        )
    )
    reference_frequency, reference_ratio = fft_residual_ratio(
        fundamental,
        distortion + noise,
    )
    reference_frequency, reference_ratio = smooth_log_ratio(
        reference_frequency,
        reference_ratio,
    )

    band_mask = (frequency >= BAND[0]) & (frequency <= BAND[1])
    band_frequency = frequency[band_mask]
    band_spectra = spectra[:, band_mask]
    band_main_by_frame = main_by_frame[:, band_mask]
    band_residual_by_frame = residual_by_frame[:, band_mask]
    cumulative_main_spectra = np.cumsum(band_main_by_frame, axis=0)
    cumulative_residual_spectra = np.cumsum(band_residual_by_frame, axis=0)
    cumulative_ratio_spectra = power_ratio(
        cumulative_residual_spectra,
        cumulative_main_spectra,
    )
    thd_frequency = np.geomspace(BAND[0], BAND[1], REFERENCE_POINTS)
    smoothed_ratio_spectra = np.empty(
        (len(cumulative_ratio_spectra), len(thd_frequency)),
        dtype=np.float64,
    )
    for frame_index, ratio_spectrum in enumerate(cumulative_ratio_spectra):
        _frequency, smoothed_ratio_spectra[frame_index] = smooth_log_ratio(
            band_frequency,
            ratio_spectrum,
            smoothing_octaves=THD_SMOOTHING_OCTAVES,
        )
    cumulative_main_total = np.sum(cumulative_main_spectra, axis=1)
    cumulative_residual_total = np.sum(cumulative_residual_spectra, axis=1)
    cumulative_integrated_ratio = power_ratio(
        cumulative_residual_total,
        cumulative_main_total,
    )
    center_indices = np.searchsorted(thd_frequency, centers)
    center_indices = np.clip(center_indices, 0, len(thd_frequency) - 1)
    pink_correction = band_frequency / 1000.0
    display_spectra = band_spectra * pink_correction[None, :]
    db_spectra = 10.0 * np.log10(display_spectra / np.max(display_spectra) + 1e-14)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), constrained_layout=True)
    ax_spectrum, ax_ratio = axes

    (spectrum_line,) = ax_spectrum.semilogx(
        band_frequency,
        db_spectra[0],
        color="tab:blue",
        linewidth=1.1,
    )
    first_left, first_right = fitted_mask_edges(float(centers[0]), mask_fit)
    mask_span = ax_spectrum.axvspan(
        first_left,
        first_right,
        color="tab:green",
        alpha=0.28,
    )
    peak_line = ax_spectrum.axvline(centers[0], color="tab:red", linewidth=1.2)
    ax_spectrum.set_xlim(BAND)
    ax_spectrum.set_ylim(-95.0, 2.0)
    ax_spectrum.set_title("Current spectrum: green = A, outside = B")
    ax_spectrum.set_xlabel("Frequency, Hz")
    ax_spectrum.set_ylabel("Pink-corrected power, dB rel.")
    ax_spectrum.grid(True, which="both", alpha=0.25)

    (reference_line,) = ax_ratio.semilogx(
        reference_frequency,
        ratio_percent(reference_ratio),
        color="0.45",
        linewidth=0.75,
        alpha=0.5,
    )
    (frame_line,) = ax_ratio.semilogx([], [], color="tab:orange", linewidth=0.9, alpha=0.65)
    (current_point,) = ax_ratio.semilogx([], [], "o", color="tab:red", markersize=4)
    ax_ratio.set_xlim(BAND)
    set_thd_axis(ax_ratio)
    ax_ratio.set_title("Frame THD+N")
    ax_ratio.set_xlabel("Tracked fundamental, Hz")
    ax_ratio.set_ylabel("THD+N, %")
    ax_ratio.grid(True, which="both", alpha=0.25)
    ax_ratio.legend(
        [reference_line, frame_line],
        ["rfft residual/fundamental", "accumulated STFT split"],
        loc="upper right",
    )

    title = fig.suptitle("", fontsize=12)
    frame_time = stft_frame_times(len(signal))[: len(centers)]

    def update(frame_index: int):
        start_time = frame_time[frame_index] - SEGMENT_SECONDS / 2.0
        stop_time = frame_time[frame_index] + SEGMENT_SECONDS / 2.0
        title.set_text(
            f"t={start_time:05.2f}-{stop_time:05.2f}s, "
            f"f0={centers[frame_index]:.1f} Hz, "
            f"THD+N={cumulative_integrated_ratio[frame_index] * 100.0:.2f}%"
        )

        spectrum_line.set_ydata(db_spectra[frame_index])
        peak_line.set_xdata([centers[frame_index], centers[frame_index]])

        mask_left, mask_right = fitted_mask_edges(float(centers[frame_index]), mask_fit)
        mask_span.set_x(mask_left)
        mask_span.set_width(mask_right - mask_left)

        frame_line.set_data(
            thd_frequency,
            ratio_percent(smoothed_ratio_spectra[frame_index]),
        )
        center_index = center_indices[frame_index]
        current_point.set_data(
            [thd_frequency[center_index]],
            [float(ratio_percent(smoothed_ratio_spectra[frame_index, center_index]))],
        )
        return title, spectrum_line, peak_line, mask_span, frame_line, current_point

    frame_indices = np.arange(0, len(centers), ANIMATION_STRIDE, dtype=int)
    if frame_indices[-1] != len(centers) - 1:
        frame_indices = np.append(frame_indices, len(centers) - 1)
    fps = max(1, round(1.0 / (HOP_SECONDS * ANIMATION_STRIDE)))
    animation = FuncAnimation(
        fig,
        update,
        frames=frame_indices,
        interval=HOP_SECONDS * ANIMATION_STRIDE * 1000.0,
        blit=False,
    )
    output_path.parent.mkdir(exist_ok=True)
    animation.save(output_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"Saved animation: {output_path}", flush=True)


if __name__ == "__main__":
    time, fundamental, distortion, noise, signal = build_test_signal()
    mask_calibration = calibrate_masks_from_clean_sweep(fundamental)
    mask_fit = fit_mask_width_functions(mask_calibration)
    normal_metrics = compute_metrics(
        fundamental=fundamental,
        distortion=distortion,
        noise=noise,
        signal=signal,
        mask_fit=mask_fit,
    )
    print_summary(
        label="Synthetic THD+N sweep experiment",
        injected_ratio=float(normal_metrics["injected_ratio"]),
        injected_components_ratio=float(normal_metrics["injected_components_ratio"]),
        measured_ratio=float(normal_metrics["measured_ratio"]),
        leakage_ratio=float(normal_metrics["leakage_ratio"]),
        oracle_ratio=float(normal_metrics["oracle_ratio"]),
    )
    save_summary_plot(
        output_path=OUTPUT_DIR / "thd4_stft_split.png",
        time=time,
        fundamental=fundamental,
        distortion=distortion,
        noise=noise,
        signal=signal,
        metrics=normal_metrics,
    )
    save_energy_plot(
        output_path=OUTPUT_DIR / "thd4_ab_breakdown.png",
        metrics=normal_metrics,
    )
    save_mask_width_plot(
        output_path=OUTPUT_DIR / "thd4_mask_width.png",
        mask_calibration=mask_calibration,
        mask_fit=mask_fit,
    )
    save_energy_accumulation_animation(
        output_path=OUTPUT_DIR / "thd4_ab_accumulation.gif",
        signal=signal,
        fundamental=fundamental,
        distortion=distortion,
        noise=noise,
        mask_fit=mask_fit,
    )

    clean_distortion = np.zeros_like(fundamental)
    clean_noise = np.zeros_like(fundamental)
    clean_signal = fundamental.copy()
    clean_metrics = compute_metrics(
        fundamental=fundamental,
        distortion=clean_distortion,
        noise=clean_noise,
        signal=clean_signal,
        mask_fit=mask_fit,
    )
    print_summary(
        label="Clean chirp experiment",
        injected_ratio=float(clean_metrics["injected_ratio"]),
        injected_components_ratio=float(clean_metrics["injected_components_ratio"]),
        measured_ratio=float(clean_metrics["measured_ratio"]),
        leakage_ratio=float(clean_metrics["leakage_ratio"]),
        oracle_ratio=float(clean_metrics["oracle_ratio"]),
    )
    save_summary_plot(
        output_path=OUTPUT_DIR / "thd4_stft_split_clean.png",
        time=time,
        fundamental=fundamental,
        distortion=clean_distortion,
        noise=clean_noise,
        signal=clean_signal,
        metrics=clean_metrics,
    )

    save_realtime_stft_animation(
        output_path=OUTPUT_DIR / "thd4_stft_realtime.gif",
        signal=signal,
        fundamental=fundamental,
        distortion=distortion,
        noise=noise,
        mask_fit=mask_fit,
    )
