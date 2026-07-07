from __future__ import annotations

from pathlib import Path
import sys
from typing import TypedDict


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) in sys.path:
    sys.path.remove(str(SCRIPT_DIR))

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
from numpy.typing import NDArray


SAMPLE_RATE = 96_000
DURATION = 30.0
BAND = (20.0, 20_000.0)
SEGMENT_SECONDS = 1.0
HOP_SECONDS = 0.1
MASK_MIN_BINS = 3.0
MASK_SAFETY = 1.02
HARMONIC_COUNT = 4
HARMONIC_DECAY = 0.1
PINK_NOISE_RATIO = 0.01
SEED = 12_345
ANIMATION_STRIDE = 1
REFERENCE_SMOOTHING_OCTAVES = 1.0 / 12.0
REFERENCE_POINTS = 1_200
OUTPUT_DIR = Path("artifacts")


class Metrics(TypedDict):
    """Container for scalar and per-frame THD+N metrics."""

    main_energy: float
    residual_energy: float
    centers: NDArray[np.float64]
    main_by_frame: NDArray[np.float64]
    residual_by_frame: NDArray[np.float64]
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
    f_start, f_stop = BAND
    sweep_rate = DURATION / np.log(f_stop / f_start)
    phase = 2.0 * np.pi * f_start * sweep_rate * (np.exp(time / sweep_rate) - 1.0)
    frequency = f_start * np.exp(time / sweep_rate)
    return time, amplitude * np.sin(phase), frequency


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


def mask_ratio_for_frequency(frequency: float) -> float:
    """Return the A-mask half-width ratio for a tracked chirp frequency."""
    sweep_rate = DURATION / np.log(BAND[1] / BAND[0])
    chirp_ratio = np.exp(SEGMENT_SECONDS / (2.0 * sweep_rate))
    bin_width_hz = 1.0 / SEGMENT_SECONDS
    bin_ratio = 1.0 + MASK_MIN_BINS * bin_width_hz / max(frequency, bin_width_hz)
    return float(max(chirp_ratio, bin_ratio) * MASK_SAFETY)


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


def stft_frame_analysis(
    signal: NDArray[np.float64],
    *,
    center_from: NDArray[np.float64] | None = None,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.bool_],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Split each STFT frame into fundamental-mask energy and residual energy."""
    segment_size = int(round(SEGMENT_SECONDS * SAMPLE_RATE))
    hop_size = int(round(HOP_SECONDS * SAMPLE_RATE))
    starts = np.arange(0, len(signal) - segment_size + 1, hop_size)
    window = np.hanning(segment_size)
    frequency = np.fft.rfftfreq(segment_size, 1.0 / SAMPLE_RATE)
    band_mask = (frequency >= BAND[0]) & (frequency <= BAND[1])

    spectra = np.empty((len(starts), len(frequency)), dtype=np.float64)
    masks = np.empty((len(starts), len(frequency)), dtype=np.bool_)
    centers = np.empty(len(starts), dtype=np.float64)
    main_by_frame = np.empty(len(starts), dtype=np.float64)
    residual_by_frame = np.empty(len(starts), dtype=np.float64)

    peak_source = signal if center_from is None else center_from
    for frame_index, start in enumerate(starts):
        stop = int(start + segment_size)
        frame = signal[start:stop] * window
        source_frame = peak_source[start:stop] * window

        source_power = np.abs(np.fft.rfft(source_frame)) ** 2
        search_power = np.where(band_mask, source_power, 0.0)
        peak_index = int(np.argmax(search_power))
        center_frequency = float(frequency[peak_index])
        mask_ratio = mask_ratio_for_frequency(center_frequency)

        main_mask = (
            band_mask
            & (frequency >= center_frequency / mask_ratio)
            & (frequency <= center_frequency * mask_ratio)
        )
        power = np.abs(np.fft.rfft(frame)) ** 2

        spectra[frame_index] = power
        masks[frame_index] = main_mask
        centers[frame_index] = center_frequency
        main_by_frame[frame_index] = float(np.sum(power[main_mask]))
        residual_by_frame[frame_index] = float(np.sum(power[band_mask & ~main_mask]))

    return frequency, centers, spectra, masks, main_by_frame, residual_by_frame


def stft_energy_split(
    signal: NDArray[np.float64],
    *,
    center_from: NDArray[np.float64] | None = None,
) -> tuple[
    float,
    float,
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Return total A/B energies plus per-frame A/B energies."""
    frequency, centers, _spectra, _masks, main_by_frame, residual_by_frame = (
        stft_frame_analysis(
            signal,
            center_from=center_from,
        )
    )
    main_energy = float(np.sum(main_by_frame))
    residual_energy = float(np.sum(residual_by_frame))
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
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Smooth an amplitude ratio by averaging its power in log-frequency bins."""
    output_frequency = np.geomspace(BAND[0], BAND[1], REFERENCE_POINTS)
    log_frequency = np.log2(frequency)
    log_output = np.log2(output_frequency)
    half_width = REFERENCE_SMOOTHING_OCTAVES / 2.0

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


def ratio_percent_ylim(
    *ratios: NDArray[np.float64],
    minimum: float = 0.25,
) -> tuple[float, float]:
    """Choose a 0-based percent axis that includes all finite ratio samples."""
    values = np.concatenate([np.ravel(ratio) * 100.0 for ratio in ratios])
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0, minimum
    return 0.0, max(minimum, float(np.max(values) * 1.12))


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
) -> Metrics:
    """Compute all scalar and per-frame values used by plots."""
    main_energy, residual_energy, centers, main_by_frame, residual_by_frame, _frequency = (
        stft_energy_split(
            signal,
        )
    )
    clean_main, clean_residual, *_ = stft_energy_split(
        fundamental,
    )
    oracle_residual = stft_energy_split(
        distortion + noise,
        center_from=fundamental,
    )[1]
    oracle_main = stft_energy_split(
        fundamental,
        center_from=fundamental,
    )[0]

    injected_ratio = rms(distortion + noise) / rms(fundamental)
    injected_components_ratio = np.sqrt(rms(distortion) ** 2 + rms(noise) ** 2) / rms(
        fundamental
    )

    return {
        "main_energy": main_energy,
        "residual_energy": residual_energy,
        "centers": centers,
        "main_by_frame": main_by_frame,
        "residual_by_frame": residual_by_frame,
        "measured_ratio": float(np.sqrt(residual_energy / main_energy)),
        "leakage_ratio": float(np.sqrt(clean_residual / clean_main)),
        "oracle_ratio": float(np.sqrt(oracle_residual / oracle_main)),
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
    centers = metrics["centers"]
    residual_by_frame = metrics["residual_by_frame"]
    main_by_frame = metrics["main_by_frame"]
    measured_ratio = metrics["measured_ratio"]

    frame_time = (
        np.arange(len(centers), dtype=np.float64) * HOP_SECONDS + SEGMENT_SECONDS / 2.0
    )
    frame_ratio = np.sqrt(residual_by_frame / np.maximum(main_by_frame, 1e-30))
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

    axes[2].semilogx(centers, frame_ratio * 100.0, label="frame STFT split")
    axes[2].semilogx(
        fft_frequency,
        fft_ratio * 100.0,
        color="0.35",
        linewidth=0.8,
        alpha=0.65,
        label="rfft residual/fundamental",
    )
    axes[2].axhline(
        measured_ratio * 100.0,
        color="tab:green",
        linestyle=":",
        label="integrated STFT split",
    )
    axes[2].set_title("THD+N estimate by instantaneous frequency")
    axes[2].set_xlabel("Frequency, Hz")
    axes[2].set_ylabel("THD+N, %")
    axes[2].set_xlim(BAND)
    axes[2].set_ylim(*ratio_percent_ylim(frame_ratio, fft_ratio))
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
    centers = metrics["centers"]
    main_by_frame = metrics["main_by_frame"]
    residual_by_frame = metrics["residual_by_frame"]
    main_energy = metrics["main_energy"]
    residual_energy = metrics["residual_energy"]

    cumulative_main = np.cumsum(main_by_frame)
    cumulative_residual = np.cumsum(residual_by_frame)
    frame_ratio = np.sqrt(residual_by_frame / np.maximum(main_by_frame, 1e-30))
    cumulative_ratio = np.sqrt(cumulative_residual / np.maximum(cumulative_main, 1e-30))

    main_frame_db = 10.0 * np.log10(main_by_frame / np.max(main_by_frame) + 1e-14)
    residual_frame_db = 10.0 * np.log10(
        residual_by_frame / np.max(main_by_frame) + 1e-14
    )
    cumulative_main_db = 10.0 * np.log10(cumulative_main / main_energy + 1e-14)
    cumulative_residual_db = 10.0 * np.log10(cumulative_residual / main_energy + 1e-14)

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), constrained_layout=True, sharex=True)

    axes[0].semilogx(centers, main_frame_db, color="tab:green", linewidth=1.1)
    axes[0].semilogx(centers, residual_frame_db, color="tab:red", linewidth=1.1)
    axes[0].set_title("Frame energy returned by stft_energy_split")
    axes[0].set_ylabel("Energy, dB rel. max main frame")
    axes[0].set_xlim(BAND)
    axes[0].set_ylim(-80.0, 5.0)
    axes[0].legend(["main_by_frame", "residual_by_frame"], loc="upper right")
    axes[0].grid(True, which="both", alpha=0.3)

    axes[1].semilogx(centers, cumulative_main_db, color="tab:green", linewidth=1.1)
    axes[1].semilogx(centers, cumulative_residual_db, color="tab:red", linewidth=1.1)
    axes[1].axhline(0.0, color="tab:green", linestyle=":", linewidth=0.9)
    axes[1].axhline(
        10.0 * np.log10(residual_energy / main_energy),
        color="tab:red",
        linestyle=":",
        linewidth=0.9,
    )
    axes[1].set_title("Cumulative main_energy and residual_energy")
    axes[1].set_ylabel("Cumulative energy, dB rel. final main_energy")
    axes[1].set_xlim(BAND)
    axes[1].set_ylim(-80.0, 5.0)
    axes[1].legend(["main_energy", "residual_energy"], loc="lower right")
    axes[1].grid(True, which="both", alpha=0.3)

    axes[2].semilogx(centers, frame_ratio * 100.0, color="tab:blue", linewidth=1.0)
    axes[2].semilogx(centers, cumulative_ratio * 100.0, color="tab:purple", linewidth=1.2)
    axes[2].axhline(
        np.sqrt(residual_energy / main_energy) * 100.0,
        color="tab:purple",
        linestyle=":",
        linewidth=0.9,
    )
    axes[2].set_title("Ratios from the same energies")
    axes[2].set_xlabel("Tracked fundamental, Hz")
    axes[2].set_ylabel("sqrt(residual/main), %")
    axes[2].set_xlim(BAND)
    axes[2].set_ylim(*ratio_percent_ylim(frame_ratio, cumulative_ratio))
    axes[2].legend(["frame", "cumulative"], loc="upper right")
    axes[2].grid(True, which="both", alpha=0.3)

    output_path.parent.mkdir(exist_ok=True)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    print(f"Saved A/B energy plot: {output_path}", flush=True)


def save_energy_accumulation_animation(
    *,
    output_path: Path,
    signal: NDArray[np.float64],
) -> None:
    """Save a GIF of SFFT frames and A/B energy accumulation."""
    frequency, centers, spectra, _masks, main_by_frame, residual_by_frame = (
        stft_frame_analysis(signal)
    )
    main_energy = float(np.sum(main_by_frame))
    residual_energy = float(np.sum(residual_by_frame))

    cumulative_main = np.cumsum(main_by_frame)
    cumulative_residual = np.cumsum(residual_by_frame)
    frame_ratio = np.sqrt(residual_by_frame / np.maximum(main_by_frame, 1e-30))
    cumulative_ratio = np.sqrt(cumulative_residual / np.maximum(cumulative_main, 1e-30))

    main_frame_db = 10.0 * np.log10(main_by_frame / np.max(main_by_frame) + 1e-14)
    residual_frame_db = 10.0 * np.log10(
        residual_by_frame / np.max(main_by_frame) + 1e-14
    )
    cumulative_main_db = 10.0 * np.log10(cumulative_main / main_energy + 1e-14)
    cumulative_residual_db = 10.0 * np.log10(cumulative_residual / main_energy + 1e-14)
    final_residual_db = 10.0 * np.log10(residual_energy / main_energy)

    band_mask = (frequency >= BAND[0]) & (frequency <= BAND[1])
    band_frequency = frequency[band_mask]
    band_spectra = spectra[:, band_mask]
    pink_correction = band_frequency / 1000.0
    display_spectra = band_spectra * pink_correction[None, :]
    db_spectra = 10.0 * np.log10(display_spectra / np.max(display_spectra) + 1e-14)

    fig, axes = plt.subplots(4, 1, figsize=(10, 10), constrained_layout=True)
    ax_spectrum, ax_frame, ax_cumulative, ax_ratio = axes

    (spectrum_line,) = ax_spectrum.semilogx(
        band_frequency,
        db_spectra[0],
        color="tab:blue",
        linewidth=1.1,
    )
    mask_span = ax_spectrum.axvspan(
        max(BAND[0], centers[0] / mask_ratio_for_frequency(float(centers[0]))),
        min(BAND[1], centers[0] * mask_ratio_for_frequency(float(centers[0]))),
        color="tab:green",
        alpha=0.28,
    )
    peak_line = ax_spectrum.axvline(centers[0], color="tab:red", linewidth=1.2)
    ax_spectrum.set_title("Current SFFT spectrum: green = A, outside = B")
    ax_spectrum.set_ylabel("Pink-corrected power, dB rel.")
    ax_spectrum.set_xlim(BAND)
    ax_spectrum.set_ylim(-95.0, 2.0)
    ax_spectrum.grid(True, which="both", alpha=0.3)

    (main_frame_line,) = ax_frame.semilogx([], [], color="tab:green", linewidth=1.1)
    (residual_frame_line,) = ax_frame.semilogx([], [], color="tab:red", linewidth=1.1)
    (main_frame_point,) = ax_frame.semilogx([], [], "o", color="tab:green", markersize=4)
    (residual_frame_point,) = ax_frame.semilogx([], [], "o", color="tab:red", markersize=4)
    ax_frame.set_title("Frame A/B energy")
    ax_frame.set_ylabel("Energy, dB rel. max A")
    ax_frame.set_xlim(BAND)
    ax_frame.set_ylim(-80.0, 5.0)
    ax_frame.legend(["A frame", "B frame"], loc="upper right")
    ax_frame.grid(True, which="both", alpha=0.3)

    (main_sum_line,) = ax_cumulative.semilogx([], [], color="tab:green", linewidth=1.2)
    (residual_sum_line,) = ax_cumulative.semilogx([], [], color="tab:red", linewidth=1.2)
    (main_sum_point,) = ax_cumulative.semilogx([], [], "o", color="tab:green", markersize=4)
    (residual_sum_point,) = ax_cumulative.semilogx([], [], "o", color="tab:red", markersize=4)
    ax_cumulative.axhline(0.0, color="tab:green", linestyle=":", linewidth=0.9)
    ax_cumulative.axhline(final_residual_db, color="tab:red", linestyle=":", linewidth=0.9)
    ax_cumulative.set_title("Cumulative A/B energy")
    ax_cumulative.set_ylabel("Sum, dB rel. final A")
    ax_cumulative.set_xlim(BAND)
    ax_cumulative.set_ylim(-80.0, 5.0)
    ax_cumulative.legend(["A sum", "B sum"], loc="lower right")
    ax_cumulative.grid(True, which="both", alpha=0.3)

    (frame_ratio_line,) = ax_ratio.semilogx([], [], color="tab:blue", linewidth=1.0)
    (sum_ratio_line,) = ax_ratio.semilogx([], [], color="tab:purple", linewidth=1.2)
    (sum_ratio_point,) = ax_ratio.semilogx([], [], "o", color="tab:purple", markersize=4)
    ax_ratio.axhline(
        np.sqrt(residual_energy / main_energy) * 100.0,
        color="tab:purple",
        linestyle=":",
        linewidth=0.9,
    )
    ax_ratio.set_title("Frame and cumulative THD+N")
    ax_ratio.set_xlabel("Tracked fundamental, Hz")
    ax_ratio.set_ylabel("sqrt(B/A), %")
    ax_ratio.set_xlim(BAND)
    ax_ratio.set_ylim(*ratio_percent_ylim(frame_ratio, cumulative_ratio))
    ax_ratio.legend(["frame", "cumulative"], loc="upper right")
    ax_ratio.grid(True, which="both", alpha=0.3)

    title = fig.suptitle("", fontsize=12)

    def update(frame_index: int):
        upto = frame_index + 1
        current_frequency = centers[frame_index]
        title.set_text(
            f"frame={frame_index + 1}/{len(centers)}, "
            f"f0={current_frequency:.1f} Hz, "
            f"cumulative THD+N={cumulative_ratio[frame_index] * 100.0:.3f}%"
        )

        spectrum_line.set_ydata(db_spectra[frame_index])
        peak_line.set_xdata([current_frequency, current_frequency])
        mask_ratio = mask_ratio_for_frequency(float(current_frequency))
        mask_left = max(BAND[0], current_frequency / mask_ratio)
        mask_right = min(BAND[1], current_frequency * mask_ratio)
        mask_span.set_x(mask_left)
        mask_span.set_width(max(0.0, mask_right - mask_left))

        main_frame_line.set_data(centers[:upto], main_frame_db[:upto])
        residual_frame_line.set_data(centers[:upto], residual_frame_db[:upto])
        main_frame_point.set_data([current_frequency], [main_frame_db[frame_index]])
        residual_frame_point.set_data([current_frequency], [residual_frame_db[frame_index]])

        main_sum_line.set_data(centers[:upto], cumulative_main_db[:upto])
        residual_sum_line.set_data(centers[:upto], cumulative_residual_db[:upto])
        main_sum_point.set_data([current_frequency], [cumulative_main_db[frame_index]])
        residual_sum_point.set_data(
            [current_frequency],
            [cumulative_residual_db[frame_index]],
        )

        frame_ratio_line.set_data(centers[:upto], frame_ratio[:upto] * 100.0)
        sum_ratio_line.set_data(centers[:upto], cumulative_ratio[:upto] * 100.0)
        sum_ratio_point.set_data(
            [current_frequency],
            [cumulative_ratio[frame_index] * 100.0],
        )
        return (
            title,
            spectrum_line,
            peak_line,
            mask_span,
            main_frame_line,
            residual_frame_line,
            main_frame_point,
            residual_frame_point,
            main_sum_line,
            residual_sum_line,
            main_sum_point,
            residual_sum_point,
            frame_ratio_line,
            sum_ratio_line,
            sum_ratio_point,
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
) -> None:
    """Save a GIF of the current STFT spectrum and frame THD+N estimate."""
    frequency, centers, spectra, _masks, main_by_frame, residual_by_frame = (
        stft_frame_analysis(
            signal,
        )
    )
    frame_ratio = np.sqrt(residual_by_frame / np.maximum(main_by_frame, 1e-30))
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
    mask_span = ax_spectrum.axvspan(
        max(BAND[0], centers[0] / mask_ratio_for_frequency(float(centers[0]))),
        min(BAND[1], centers[0] * mask_ratio_for_frequency(float(centers[0]))),
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
        reference_ratio * 100.0,
        color="0.45",
        linewidth=0.75,
        alpha=0.5,
    )
    (frame_line,) = ax_ratio.semilogx([], [], color="tab:orange", linewidth=0.9, alpha=0.65)
    (current_point,) = ax_ratio.semilogx([], [], "o", color="tab:red", markersize=4)
    ax_ratio.set_xlim(BAND)
    ax_ratio.set_ylim(*ratio_percent_ylim(frame_ratio, reference_ratio))
    ax_ratio.set_title("Frame THD+N")
    ax_ratio.set_xlabel("Tracked fundamental, Hz")
    ax_ratio.set_ylabel("THD+N, %")
    ax_ratio.grid(True, which="both", alpha=0.25)
    ax_ratio.legend(
        [reference_line, frame_line],
        ["rfft residual/fundamental", "frame"],
        loc="upper right",
    )

    title = fig.suptitle("", fontsize=12)

    def update(frame_index: int):
        start_time = frame_index * HOP_SECONDS
        stop_time = start_time + SEGMENT_SECONDS
        title.set_text(
            f"t={start_time:05.2f}-{stop_time:05.2f}s, "
            f"f0={centers[frame_index]:.1f} Hz, "
            f"THD+N={frame_ratio[frame_index] * 100.0:.2f}%"
        )

        spectrum_line.set_ydata(db_spectra[frame_index])
        peak_line.set_xdata([centers[frame_index], centers[frame_index]])

        mask_ratio = mask_ratio_for_frequency(float(centers[frame_index]))
        mask_left = max(BAND[0], centers[frame_index] / mask_ratio)
        mask_right = min(BAND[1], centers[frame_index] * mask_ratio)
        mask_span.set_x(mask_left)
        mask_span.set_width(max(0.0, mask_right - mask_left))

        frame_line.set_data(centers[: frame_index + 1], frame_ratio[: frame_index + 1] * 100.0)
        current_point.set_data([centers[frame_index]], [frame_ratio[frame_index] * 100.0])
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
    normal_metrics = compute_metrics(
        fundamental=fundamental,
        distortion=distortion,
        noise=noise,
        signal=signal,
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
    save_energy_accumulation_animation(
        output_path=OUTPUT_DIR / "thd4_ab_accumulation.gif",
        signal=signal,
    )

    clean_distortion = np.zeros_like(fundamental)
    clean_noise = np.zeros_like(fundamental)
    clean_signal = fundamental.copy()
    clean_metrics = compute_metrics(
        fundamental=fundamental,
        distortion=clean_distortion,
        noise=clean_noise,
        signal=clean_signal,
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
    )
