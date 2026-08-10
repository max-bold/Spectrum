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
from scipy.signal import ShortTimeFFT


SAMPLE_RATE = 192000
DURATION = 30.0
BAND = (20.0, 20_000.0)
FADE_SECONDS = 0.5
SEGMENT_SECONDS = 1.0
HOP_SECONDS = 0.1

# Flat rejection window: [f0 / c, f0 * c].
NOTCH_C = 1.5

HARMONIC_COUNT = 4
HARMONIC_DECAY = 0.1
PINK_NOISE_RATIO = 0.01
SEED = 12_345
ANIMATION_STRIDE = 2
SMOOTHING_OCTAVES = 1.0 / 12.0
REFERENCE_POINTS = 1_200
THD_PERCENT_LIMITS = (0.01, 10.0)
OUTPUT_DIR = Path("artifacts")


class FrameAnalysis(TypedDict):
    """STFT data and IEC residual/total result for selected frames."""

    frequency: NDArray[np.float64]
    time: NDArray[np.float64]
    frame_indices: NDArray[np.int64]
    centers: NDArray[np.float64]
    power: NDArray[np.float64]
    filtered_power: NDArray[np.float64]
    total_rms: NDArray[np.float64]
    residual_rms: NDArray[np.float64]
    ratio: NDArray[np.float64]


class Metrics(TypedDict):
    """All values used by the report plots."""

    measured: FrameAnalysis
    clean: FrameAnalysis
    oracle: FrameAnalysis
    oracle_ratio_by_frame: NDArray[np.float64]
    measured_ratio: float
    leakage_ratio: float
    oracle_ratio: float
    injected_ratio: float
    injected_components_ratio: float


def log_chirp(
    *,
    amplitude: float = 0.5,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Generate a logarithmic sine sweep and its instantaneous frequency."""
    time = np.arange(int(round(SAMPLE_RATE * DURATION)), dtype=np.float64) / SAMPLE_RATE
    fade_size = int(round(FADE_SECONDS * SAMPLE_RATE))
    f_start, f_stop = BAND
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
    """Return the RMS of a real signal."""
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
    """Create the common STFT transform used for all frame calculations."""
    segment_size = int(round(SEGMENT_SECONDS * SAMPLE_RATE))
    hop_size = int(round(HOP_SECONDS * SAMPLE_RATE))
    return ShortTimeFFT(
        np.hanning(segment_size),
        hop=hop_size,
        fs=SAMPLE_RATE,
        fft_mode="onesided",
        mfft=segment_size,
    )


def rejection_filter(
    frequency: NDArray[np.float64],
    center_frequency: float,
    *,
    c: float = NOTCH_C,
) -> NDArray[np.float64]:
    """Return a flat moving rejection window around the fundamental.

    The response is zero from ``f0 / c`` through ``f0 * c`` and one
    everywhere else.  Thus the entire fundamental region is removed without
    retaining a frequency-dependent fraction of it.
    """
    if c <= 1.0:
        raise ValueError("c must be greater than one")
    if center_frequency <= 0.0:
        raise ValueError("center_frequency must be positive")
    if np.any(frequency < 0.0):
        raise ValueError("filter frequencies must be non-negative")

    response = np.ones_like(frequency, dtype=np.float64)
    rejected = (frequency >= center_frequency / c) & (
        frequency <= center_frequency * c
    )
    response[rejected] = 0.0
    return response


def analyze_frames(
    signal: NDArray[np.float64],
    *,
    reference: FrameAnalysis | None = None,
    keep_spectra: bool = True,
) -> FrameAnalysis:
    """Calculate one residual/total THD+N value for every complete frame.

    A is the RMS over the one-sided spectrum strictly between DC and Nyquist.
    B is the same RMS after multiplying the complex STFT by the moving
    rejection response.  The peak search uses every positive-frequency bin;
    both sums use all one-sided bins except DC and Nyquist.
    BAND is only a sweep-generation and display setting.  Consequently B/A
    is directly the IEC residual/total ratio.
    """
    transform = build_stft()
    stft = np.asarray(transform.stft(signal).T)
    all_frequency = np.asarray(transform.f, dtype=np.float64)
    all_time = np.asarray(transform.t(len(signal)), dtype=np.float64)
    frequency = all_frequency

    if reference is None:
        full_power = np.square(np.abs(stft))
        peak_indices = 1 + np.argmax(full_power[:, 1:], axis=1)
        all_centers = all_frequency[peak_indices]
        complete = (
            (all_time >= SEGMENT_SECONDS / 2.0)
            & (all_time <= len(signal) / SAMPLE_RATE - SEGMENT_SECONDS / 2.0)
        )
        frame_indices = np.flatnonzero(complete).astype(np.int64)
        centers = np.asarray(all_centers[frame_indices], dtype=np.float64)
        del full_power
    else:
        frame_indices = reference["frame_indices"]
        centers = reference["centers"]

    selected_stft = stft[frame_indices]
    power = np.asarray(np.square(np.abs(selected_stft)), dtype=np.float64)
    del selected_stft, stft

    # A one-sided real FFT contains only one bin from each +/- frequency
    # pair. Restore full-spectrum energy while excluding DC and Nyquist.
    parseval_weight = np.full(len(frequency), 2.0, dtype=np.float64)
    parseval_weight[0] = 0.0
    if np.isclose(frequency[-1], SAMPLE_RATE / 2.0):
        parseval_weight[-1] = 0.0
    power *= parseval_weight[None, :]

    filtered_power = np.empty_like(power) if keep_spectra else np.empty((0, 0))
    residual_energy = np.empty(len(centers), dtype=np.float64)

    for frame_index, center in enumerate(centers):
        response = rejection_filter(frequency, float(center))
        filtered_frame_power = power[frame_index] * np.square(response)
        residual_energy[frame_index] = float(np.sum(filtered_frame_power))
        if keep_spectra:
            filtered_power[frame_index] = filtered_frame_power

    total_rms = np.sqrt(np.sum(power, axis=1))
    residual_rms = np.sqrt(residual_energy)
    ratio = residual_rms / np.maximum(total_rms, 1e-30)
    saved_power = power if keep_spectra else np.empty((0, 0))

    return {
        "frequency": frequency,
        "time": np.asarray(all_time[frame_indices], dtype=np.float64),
        "frame_indices": frame_indices,
        "centers": centers.copy(),
        "power": saved_power,
        "filtered_power": filtered_power,
        "total_rms": np.asarray(total_rms, dtype=np.float64),
        "residual_rms": np.asarray(residual_rms, dtype=np.float64),
        "ratio": np.asarray(ratio, dtype=np.float64),
    }


def integrated_ratio(analysis: FrameAnalysis) -> float:
    """Combine frames as an energy-weighted residual/total amplitude ratio."""
    total_energy = float(np.sum(np.square(analysis["total_rms"])))
    residual_energy = float(np.sum(np.square(analysis["residual_rms"])))
    return float(np.sqrt(residual_energy / max(total_energy, 1e-30)))


def compute_metrics(
    *,
    fundamental: NDArray[np.float64],
    distortion: NDArray[np.float64],
    noise: NDArray[np.float64],
    signal: NDArray[np.float64],
) -> Metrics:
    """Compute measured, clean-floor, and known-residual frame metrics."""
    measured = analyze_frames(signal)
    clean = analyze_frames(fundamental, reference=measured, keep_spectra=False)
    oracle = analyze_frames(distortion + noise, reference=measured, keep_spectra=False)
    oracle_ratio_by_frame = oracle["residual_rms"] / np.maximum(
        measured["total_rms"], 1e-30
    )
    oracle_ratio = float(
        np.sqrt(
            np.sum(np.square(oracle["residual_rms"]))
            / np.sum(np.square(measured["total_rms"]))
        )
    )

    return {
        "measured": measured,
        "clean": clean,
        "oracle": oracle,
        "oracle_ratio_by_frame": np.asarray(oracle_ratio_by_frame, dtype=np.float64),
        "measured_ratio": integrated_ratio(measured),
        "leakage_ratio": integrated_ratio(clean),
        "oracle_ratio": oracle_ratio,
        "injected_ratio": rms(distortion + noise) / rms(signal),
        "injected_components_ratio": (
            np.sqrt(rms(distortion) ** 2 + rms(noise) ** 2) / rms(signal)
        ),
    }


def smooth_log_curve(
    frequency: NDArray[np.float64],
    ratio: NDArray[np.float64],
    *,
    smoothing_octaves: float = SMOOTHING_OCTAVES,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """RMS-smooth irregular frame ratios on a log-frequency grid."""
    order = np.argsort(frequency)
    sorted_frequency = frequency[order]
    sorted_power = np.square(ratio[order])
    output_frequency = np.geomspace(BAND[0], BAND[1], REFERENCE_POINTS)
    log_frequency = np.log2(sorted_frequency)
    log_output = np.log2(output_frequency)
    half_width = smoothing_octaves / 2.0
    output_power = np.empty_like(output_frequency)

    for index, center in enumerate(log_output):
        left = int(np.searchsorted(log_frequency, center - half_width))
        right = int(np.searchsorted(log_frequency, center + half_width))
        if left == right:
            nearest = int(np.argmin(np.abs(log_frequency - center)))
            output_power[index] = sorted_power[nearest]
        else:
            output_power[index] = float(np.mean(sorted_power[left:right]))

    return output_frequency, np.sqrt(output_power)


def ratio_db(value: float) -> float:
    """Convert an amplitude ratio to dB."""
    return 20.0 * np.log10(max(value, 1e-20))


def ratio_percent(ratio: float | NDArray[np.float64]) -> NDArray[np.float64]:
    """Return clipped percent values for a logarithmic plot."""
    values = np.asarray(ratio, dtype=np.float64) * 100.0
    return np.clip(values, THD_PERCENT_LIMITS[0], THD_PERCENT_LIMITS[1])


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


def print_summary(metrics: Metrics) -> None:
    """Print the synthetic experiment result."""
    overlap = 1.0 - HOP_SECONDS / SEGMENT_SECONDS
    print("Synthetic frame-by-frame THD+N sweep experiment")
    print(f"fs={SAMPLE_RATE} Hz, T={DURATION:g} s, fundamental band={BAND}")
    print(f"A/B integration band=(0, {SAMPLE_RATE / 2:g}) Hz, endpoints excluded")
    print(
        f"segment={SEGMENT_SECONDS:g} s, hop={HOP_SECONDS:g} s, "
        f"overlap={overlap:.0%}, notch c={NOTCH_C:g}"
    )
    print(
        "Injected residual/total RMS:         "
        f"{metrics['injected_ratio'] * 100:.3f}% "
        f"({ratio_db(metrics['injected_ratio']):.2f} dB)"
    )
    print(
        "Injected components/total RMS:       "
        f"{metrics['injected_components_ratio'] * 100:.3f}% "
        f"({ratio_db(metrics['injected_components_ratio']):.2f} dB)"
    )
    print(
        "STFT filtered residual/total:        "
        f"{metrics['measured_ratio'] * 100:.3f}% "
        f"({ratio_db(metrics['measured_ratio']):.2f} dB)"
    )
    print(
        "Clean-sweep filter leakage floor:    "
        f"{metrics['leakage_ratio'] * 100:.3f}% "
        f"({ratio_db(metrics['leakage_ratio']):.2f} dB)"
    )
    print(
        "Filtered known residual / total:     "
        f"{metrics['oracle_ratio'] * 100:.3f}% "
        f"({ratio_db(metrics['oracle_ratio']):.2f} dB)"
    )


def save_filter_plot(output_path: Path) -> None:
    """Save the normalized moving-filter response."""
    normalized_frequency = np.geomspace(0.25, 4.0, 2_000)
    response = rejection_filter(normalized_frequency, 1.0)
    response_db = 20.0 * np.log10(np.maximum(response, 1e-8))

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True, constrained_layout=True)
    axes[0].semilogx(normalized_frequency, response, linewidth=1.2)
    axes[0].axvline(1.0 / NOTCH_C, color="tab:red", linewidth=0.8, linestyle=":")
    axes[0].axvline(NOTCH_C, color="tab:red", linewidth=0.8, linestyle=":")
    axes[0].set_title(f"Flat rejection window: f0/{NOTCH_C:g} to f0*{NOTCH_C:g}")
    axes[0].set_ylabel("Amplitude response w(f)")
    axes[0].set_ylim(-0.03, 1.03)
    axes[0].grid(True, which="both", alpha=0.3)

    axes[1].semilogx(normalized_frequency, response_db, linewidth=1.2)
    axes[1].axvline(1.0 / NOTCH_C, color="tab:red", linewidth=0.8, linestyle=":")
    axes[1].axvline(NOTCH_C, color="tab:red", linewidth=0.8, linestyle=":")
    axes[1].set_xlabel("Normalized frequency f / f0")
    axes[1].set_ylabel("Amplitude, dB")
    axes[1].set_ylim(-120.0, 2.0)
    axes[1].grid(True, which="both", alpha=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    print(f"Saved filter response: {output_path}", flush=True)


def save_summary_plot(
    *,
    output_path: Path,
    time: NDArray[np.float64],
    fundamental: NDArray[np.float64],
    signal: NDArray[np.float64],
    metrics: Metrics,
) -> None:
    """Save the input, frequency tracking, and frame THD+N result."""
    measured = metrics["measured"]
    smooth_frequency, smooth_ratio = smooth_log_curve(
        measured["centers"], measured["ratio"]
    )
    oracle_frequency, oracle_ratio = smooth_log_curve(
        measured["centers"], metrics["oracle_ratio_by_frame"]
    )

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), constrained_layout=True)
    axes[0].plot(time, fundamental, label="fundamental", linewidth=0.8)
    axes[0].plot(time, signal, label="total signal", linewidth=0.5, alpha=0.8)
    axes[0].set_xlim(0.0, min(1.0, DURATION))
    axes[0].set_title("First second of generated signal")
    axes[0].set_xlabel("Time, s")
    axes[0].set_ylabel("Amplitude")
    axes[0].legend(loc="upper left")
    axes[0].grid(True, alpha=0.3)

    axes[1].semilogy(measured["time"], measured["centers"])
    axes[1].set_title("Fundamental found as the maximum STFT bin")
    axes[1].set_xlabel("Frame center, s")
    axes[1].set_ylabel("f0, Hz")
    axes[1].grid(True, which="both", alpha=0.3)

    axes[2].semilogx(
        measured["centers"],
        ratio_percent(measured["ratio"]),
        ".",
        color="tab:blue",
        markersize=2.5,
        alpha=0.35,
        label="each frame: B / A",
    )
    axes[2].semilogx(
        smooth_frequency,
        ratio_percent(smooth_ratio),
        color="tab:blue",
        linewidth=1.2,
        label="1/12-octave RMS smoothing",
    )
    axes[2].semilogx(
        oracle_frequency,
        ratio_percent(oracle_ratio),
        color="0.35",
        linewidth=0.8,
        alpha=0.7,
        label="filtered known residual / total",
    )
    axes[2].axhline(
        float(ratio_percent(metrics["measured_ratio"])),
        color="tab:purple",
        linestyle=":",
        linewidth=1.0,
        label="integrated residual / total",
    )
    axes[2].set_title("IEC 60268-3 denominator: residual / total")
    axes[2].set_xlabel("Tracked fundamental, Hz")
    axes[2].set_ylabel("THD+N, %")
    axes[2].set_xlim(BAND)
    set_thd_axis(axes[2])
    axes[2].legend(loc="upper right")
    axes[2].grid(True, which="both", alpha=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    print(f"Saved summary plot: {output_path}", flush=True)


def save_ab_plot(*, output_path: Path, metrics: Metrics) -> None:
    """Save the frame A, B, B/A, and cumulative ratio diagnostics."""
    measured = metrics["measured"]
    cumulative_ratio = np.sqrt(
        np.cumsum(np.square(measured["residual_rms"]))
        / np.cumsum(np.square(measured["total_rms"]))
    )

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True, constrained_layout=True)
    axes[0].loglog(measured["centers"], measured["total_rms"], label="A: total RMS")
    axes[0].loglog(
        measured["centers"], measured["residual_rms"], label="B: filtered RMS"
    )
    axes[0].set_title("Per-frame spectral RMS values")
    axes[0].set_ylabel("Uncalibrated RMS")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, which="both", alpha=0.3)

    axes[1].semilogx(measured["centers"], ratio_percent(measured["ratio"]), label="B / A")
    axes[1].semilogx(
        measured["centers"],
        ratio_percent(metrics["oracle_ratio_by_frame"]),
        color="0.4",
        linewidth=0.8,
        label="known residual / total",
    )
    axes[1].set_title("One IEC THD+N result per frame")
    axes[1].set_ylabel("THD+N, %")
    set_thd_axis(axes[1])
    axes[1].legend(loc="upper right")
    axes[1].grid(True, which="both", alpha=0.3)

    axes[2].semilogx(measured["centers"], ratio_percent(cumulative_ratio))
    axes[2].axhline(
        float(ratio_percent(metrics["measured_ratio"])),
        color="tab:purple",
        linestyle=":",
    )
    axes[2].set_title("Energy-weighted cumulative residual / total")
    axes[2].set_xlabel("Tracked fundamental, Hz")
    axes[2].set_ylabel("THD+N, %")
    axes[2].set_xlim(BAND)
    set_thd_axis(axes[2])
    axes[2].grid(True, which="both", alpha=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    print(f"Saved A/B plot: {output_path}", flush=True)


def save_clean_plot(*, output_path: Path, metrics: Metrics) -> None:
    """Save the clean-sweep leakage floor of the moving filter."""
    clean = metrics["clean"]
    smooth_frequency, smooth_ratio = smooth_log_curve(clean["centers"], clean["ratio"])
    fig, axis = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
    axis.semilogx(
        clean["centers"],
        ratio_percent(clean["ratio"]),
        ".",
        markersize=2.5,
        alpha=0.35,
        label="each clean frame",
    )
    axis.semilogx(
        smooth_frequency,
        ratio_percent(smooth_ratio),
        linewidth=1.2,
        label="1/12-octave RMS smoothing",
    )
    axis.axhline(
        float(ratio_percent(metrics["leakage_ratio"])),
        color="tab:purple",
        linestyle=":",
        label="integrated leakage floor",
    )
    axis.set_title("Clean-sweep leakage after the moving rejection filter")
    axis.set_xlabel("Tracked fundamental, Hz")
    axis.set_ylabel("Residual / total, %")
    axis.set_xlim(BAND)
    set_thd_axis(axis)
    axis.legend(loc="upper right")
    axis.grid(True, which="both", alpha=0.3)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    print(f"Saved clean-sweep plot: {output_path}", flush=True)


def save_frame_data(*, output_path: Path, metrics: Metrics) -> None:
    """Save the requested f, A, B, and B/A values for every frame."""
    measured = metrics["measured"]
    table = np.column_stack(
        (
            measured["time"],
            measured["centers"],
            measured["total_rms"],
            measured["residual_rms"],
            measured["ratio"],
        )
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        output_path,
        table,
        delimiter=",",
        header="frame_time_s,f_hz,A_total_rms,B_residual_rms,thd_n_residual_over_total",
        comments="",
    )
    print(f"Saved frame data: {output_path}", flush=True)


def save_realtime_animation(*, output_path: Path, metrics: Metrics) -> None:
    """Animate the current STFT, moving filter, and saved (f0, B/A) points."""
    measured = metrics["measured"]
    frequency = measured["frequency"]
    display_power = measured["power"] * (frequency / 1_000.0)[None, :]
    reference_power = float(np.max(display_power))
    spectrum_db = 10.0 * np.log10(display_power / reference_power + 1e-14)
    filtered_db = 10.0 * np.log10(
        measured["filtered_power"] * (frequency / 1_000.0)[None, :] / reference_power
        + 1e-14
    )

    fig, axes = plt.subplots(2, 1, figsize=(10, 6.5), constrained_layout=True)
    ax_spectrum, ax_ratio = axes
    (spectrum_line,) = ax_spectrum.semilogx(
        frequency, spectrum_db[0], linewidth=1.0, label="total STFT"
    )
    (filtered_line,) = ax_spectrum.semilogx(
        frequency,
        filtered_db[0],
        linewidth=0.9,
        color="tab:red",
        label="after rejection filter",
    )
    peak_line = ax_spectrum.axvline(measured["centers"][0], color="tab:green", linewidth=1.0)
    ax_spectrum.set_title("Current frame")
    ax_spectrum.set_ylabel("Power, dB rel.")
    ax_spectrum.set_xlim(BAND)
    ax_spectrum.set_ylim(-100.0, 2.0)
    ax_spectrum.legend(loc="upper right")
    ax_spectrum.grid(True, which="both", alpha=0.3)

    (ratio_line,) = ax_ratio.semilogx([], [], color="tab:blue", linewidth=1.0)
    (current_point,) = ax_ratio.semilogx([], [], "o", color="tab:red", markersize=4)
    ax_ratio.axhline(
        float(ratio_percent(metrics["measured_ratio"])),
        color="tab:purple",
        linestyle=":",
        linewidth=0.9,
    )
    ax_ratio.set_title("Saved frame results: THD+N(f0) = B / A")
    ax_ratio.set_xlabel("Tracked fundamental, Hz")
    ax_ratio.set_ylabel("Residual / total, %")
    ax_ratio.set_xlim(BAND)
    set_thd_axis(ax_ratio)
    ax_ratio.grid(True, which="both", alpha=0.3)
    title = fig.suptitle("")

    def update(frame_index: int):
        center = measured["centers"][frame_index]
        ratio = measured["ratio"][frame_index]
        title.set_text(
            f"frame={frame_index + 1}/{len(measured['centers'])}, "
            f"f0={center:.1f} Hz, B/A={ratio * 100.0:.3f}%"
        )
        spectrum_line.set_ydata(spectrum_db[frame_index])
        filtered_line.set_ydata(filtered_db[frame_index])
        peak_line.set_xdata([center, center])
        ratio_line.set_data(
            measured["centers"][: frame_index + 1],
            ratio_percent(measured["ratio"][: frame_index + 1]),
        )
        current_point.set_data([center], [float(ratio_percent(ratio))])
        return title, spectrum_line, filtered_line, peak_line, ratio_line, current_point

    frame_indices = np.arange(0, len(measured["centers"]), ANIMATION_STRIDE, dtype=int)
    if frame_indices[-1] != len(measured["centers"]) - 1:
        frame_indices = np.append(frame_indices, len(measured["centers"]) - 1)
    fps = max(1, round(1.0 / (HOP_SECONDS * ANIMATION_STRIDE)))
    animation = FuncAnimation(
        fig,
        update,
        frames=frame_indices,
        interval=HOP_SECONDS * ANIMATION_STRIDE * 1_000.0,
        blit=False,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    animation.save(output_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"Saved animation: {output_path}", flush=True)


def verify_filter() -> None:
    """Fail early if the configurable response loses its defining properties."""
    check_frequency = np.asarray(
        [0.0, 0.99 / NOTCH_C, 1.0 / NOTCH_C, 1.0, NOTCH_C, 1.01 * NOTCH_C],
        dtype=np.float64,
    )
    expected = np.asarray([1.0, 1.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    np.testing.assert_allclose(rejection_filter(check_frequency, 1.0), expected, atol=1e-14)


if __name__ == "__main__":
    verify_filter()
    time, fundamental, distortion, noise, signal = build_test_signal()
    metrics = compute_metrics(
        fundamental=fundamental,
        distortion=distortion,
        noise=noise,
        signal=signal,
    )
    print_summary(metrics)
    save_filter_plot(OUTPUT_DIR / "thd5_filter.png")
    save_summary_plot(
        output_path=OUTPUT_DIR / "thd5_stft_frames.png",
        time=time,
        fundamental=fundamental,
        signal=signal,
        metrics=metrics,
    )
    save_ab_plot(output_path=OUTPUT_DIR / "thd5_ab_frames.png", metrics=metrics)
    save_clean_plot(output_path=OUTPUT_DIR / "thd5_stft_frames_clean.png", metrics=metrics)
    save_frame_data(output_path=OUTPUT_DIR / "thd5_frames.csv", metrics=metrics)
    save_realtime_animation(
        output_path=OUTPUT_DIR / "thd5_stft_realtime.gif",
        metrics=metrics,
    )
