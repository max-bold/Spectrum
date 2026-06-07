"""Proof-of-concept impedance meter using a sound card.

Measurement circuit:

    audio_out -- Rr -- p1 -- Rl -- gnd

CH1 measures audio_out relative to gnd.
CH2 measures p1 relative to gnd.

The impedance is calculated from complex FFT spectra:

    H = V2 / V1
    Z = Rr * H / (1 - H)
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
from matplotlib.ticker import ScalarFormatter
from scipy.signal import chirp

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.windows import Windows, grid_filter


def generate_chirp(
    fs: int,
    duration: float,
    f_start: float,
    f_end: float,
    amplitude: float,
    fade_in: float = 0.02,
    fade_out: float = 0.02,
) -> np.ndarray:
    """Generate a mono float32 logarithmic chirp."""
    if fs <= 0:
        raise ValueError("fs must be positive")
    if duration <= 0:
        raise ValueError("duration must be positive")
    if f_start <= 0 or f_end <= 0:
        raise ValueError("f_start and f_end must be positive")
    if amplitude < 0:
        raise ValueError("amplitude must be non-negative")

    samples = int(round(fs * duration))
    t = np.arange(samples, dtype=np.float64) / fs
    signal = amplitude * chirp(
        t,
        f0=f_start,
        f1=f_end,
        t1=duration,
        method="logarithmic",
    )

    fade_in_samples = min(int(round(fade_in * fs)), samples)
    fade_out_samples = min(int(round(fade_out * fs)), samples)

    if fade_in_samples > 0:
        signal[:fade_in_samples] *= np.linspace(0.0, 1.0, fade_in_samples)
    if fade_out_samples > 0:
        signal[-fade_out_samples:] *= np.linspace(1.0, 0.0, fade_out_samples)

    return signal.astype(np.float32)


def play_and_record(
    signal: np.ndarray,
    fs: int,
    input_device: int | str | None = None,
    output_device: int | str | None = None,
    input_channels: int = 2,
    output_channels: int = 2,
    block_size: int = 1024,
    recording_tail: float = 0.25,
) -> np.ndarray:
    """Play a chirp and record it with trailing silence for latency alignment."""
    if input_channels < 2:
        raise ValueError("input_channels must be at least 2")
    if output_channels not in (1, 2):
        raise ValueError("output_channels must be 1 or 2")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if recording_tail < 0.0:
        raise ValueError("recording_tail must be non-negative")

    mono = np.asarray(signal, dtype=np.float32)
    if mono.ndim != 1:
        raise ValueError("signal must be a mono 1-D array")

    tail_samples = int(round(recording_tail * fs))
    if tail_samples:
        mono = np.pad(mono, (0, tail_samples))

    if output_channels == 1:
        playback = mono
    else:
        playback = np.column_stack([mono, mono]).astype(np.float32)

    recording = sd.playrec(
        playback,
        samplerate=fs,
        channels=input_channels,
        dtype="float32",
        device=(input_device, output_device),
        blocksize=block_size,
        blocking=True,
    )
    return np.asarray(recording, dtype=np.float32)


def trim_recording(
    recording: np.ndarray,
    chirp_samples: int,
    threshold_ratio: float = 0.02,
) -> np.ndarray:
    """Trim the recording to the chirp length using a simple CH1 threshold."""
    data = _as_2d_recording(recording)
    if chirp_samples <= 0:
        raise ValueError("chirp_samples must be positive")

    ch1 = data[:, 0]
    peak = float(np.max(np.abs(ch1))) if ch1.size else 0.0
    start = 0

    if peak > 0.0:
        threshold = peak * threshold_ratio
        candidates = np.flatnonzero(np.abs(ch1) >= threshold)
        if candidates.size:
            start = int(candidates[0])
        else:
            warnings.warn(
                "Signal start was not found, using the beginning of recording",
                RuntimeWarning,
                stacklevel=2,
            )
    else:
        warnings.warn(
            "CH1 is silent, using the beginning of recording",
            RuntimeWarning,
            stacklevel=2,
        )

    trimmed = data[start : start + chirp_samples]
    if len(trimmed) < chirp_samples:
        warnings.warn(
            "Aligned recording is shorter than chirp length; "
            "increase recording_tail. Padding with zeros",
            RuntimeWarning,
            stacklevel=2,
        )
        pad = np.zeros((chirp_samples - len(trimmed), data.shape[1]), dtype=data.dtype)
        trimmed = np.vstack([trimmed, pad])

    return trimmed


def analyze_recording_levels(
    recording: np.ndarray,
    quiet_threshold: float = 1e-4,
    clipping_threshold: float = 1.0,
    raise_on_clipping: bool = False,
) -> list[dict[str, Any]]:
    """Return peak/RMS/clipping/quiet status for every recorded channel."""
    data = _as_2d_recording(recording)
    levels: list[dict[str, Any]] = []

    for channel_idx in range(data.shape[1]):
        x = data[:, channel_idx]
        peak = float(np.max(np.abs(x))) if x.size else 0.0
        rms = float(np.sqrt(np.mean(np.square(x)))) if x.size else 0.0
        has_clipping = bool(np.any(np.abs(x) >= clipping_threshold))
        is_too_quiet = bool(rms < quiet_threshold)

        levels.append(
            {
                "channel": channel_idx + 1,
                "peak": peak,
                "rms": rms,
                "has_clipping": has_clipping,
                "is_too_quiet": is_too_quiet,
            }
        )

    clipped_channels = [item["channel"] for item in levels if item["has_clipping"]]
    if clipped_channels:
        message = f"Clipping detected on channel(s): {clipped_channels}"
        if raise_on_clipping:
            raise ValueError(message)
        warnings.warn(message, RuntimeWarning, stacklevel=2)

    return levels


def calculate_fft_spectra(
    ch1: np.ndarray,
    ch2: np.ndarray,
    fs: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate complex rFFT spectra for CH1 and CH2."""
    if fs <= 0:
        raise ValueError("fs must be positive")

    x1 = np.asarray(ch1, dtype=np.float64).reshape(-1)
    x2 = np.asarray(ch2, dtype=np.float64).reshape(-1)
    n = min(x1.size, x2.size)
    if n < 2:
        raise ValueError("ch1 and ch2 must contain at least two samples")

    x1 = x1[:n] - np.mean(x1[:n])
    x2 = x2[:n] - np.mean(x2[:n])

    # The excitation already has short fades. A full Hann window would attenuate
    # most of the logarithmic sweep near both frequency-range boundaries.
    V1 = np.fft.rfft(x1)
    V2 = np.fft.rfft(x2)
    freq = np.fft.rfftfreq(n, d=1.0 / fs)
    return freq, V1, V2


def calculate_calibration_from_known_resistor(
    ch1_cal: np.ndarray,
    ch2_cal: np.ndarray,
    fs: int,
    reference_resistor: float,
    calibration_resistor: float,
    f_min: float | None = None,
    f_max: float | None = None,
    smoothing: bool = True,
    eps: float = 1e-12,
    filter_window_width: float = 1 / 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate complex channel calibration from a known load resistor."""
    if reference_resistor <= 0:
        raise ValueError("reference_resistor must be positive")
    if calibration_resistor <= 0:
        raise ValueError("calibration_resistor must be positive")

    freq, V1_cal, V2_cal = calculate_fft_spectra(ch1_cal, ch2_cal, fs)
    freq, H_measured_cal = _calculate_transfer_function(
        freq,
        V1_cal,
        V2_cal,
        f_min,
        f_max,
        smoothing,
        filter_window_width,
        eps,
    )

    H_expected = calibration_resistor / (reference_resistor + calibration_resistor)
    calibration = H_measured_cal / H_expected
    calibration = calibration.astype(np.complex128, copy=False)
    calibration = np.where(np.abs(calibration) < eps, eps + 0j, calibration)

    freq, calibration = _apply_frequency_range(freq, calibration, f_min, f_max)
    return freq, calibration


def calculate_impedance(
    ch1: np.ndarray,
    ch2: np.ndarray,
    fs: int,
    reference_resistor: float,
    calibration: np.ndarray | None = None,
    f_min: float | None = None,
    f_max: float | None = None,
    smoothing: bool = True,
    eps: float = 1e-12,
    filter_window_width: float = 1 / 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate complex load impedance from CH1 and CH2 recordings."""
    if reference_resistor <= 0:
        raise ValueError("reference_resistor must be positive")

    freq, V1, V2 = calculate_fft_spectra(ch1, ch2, fs)
    freq, H_measured = _calculate_transfer_function(
        freq,
        V1,
        V2,
        f_min,
        f_max,
        smoothing,
        filter_window_width,
        eps,
    )

    range_already_applied = False
    if calibration is None:
        H_corrected = H_measured
    else:
        calibration_array = np.asarray(calibration, dtype=np.complex128)
        if calibration_array.shape != H_measured.shape:
            ranged_freq, ranged_H = _apply_frequency_range(
                freq, H_measured, f_min, f_max
            )
            if calibration_array.shape != ranged_H.shape:
                raise ValueError(
                    "calibration and measured transfer function sizes differ"
                )
            freq = ranged_freq
            H_measured = ranged_H
            range_already_applied = True
        H_corrected = H_measured / _safe_denominator(calibration_array, eps)

    denominator = _safe_denominator(1.0 - H_corrected, eps)
    impedance = reference_resistor * H_corrected / denominator

    if not range_already_applied:
        freq, impedance = _apply_frequency_range(freq, impedance, f_min, f_max)
    return freq, impedance


def measure_impedance_with_inline_calibration(
    fs: int,
    duration: float,
    reference_resistor: float,
    calibration_resistor: float,
    f_start: float = 20.0,
    f_end: float = 20000.0,
    amplitude: float = 0.2,
    input_device: int | str | None = None,
    output_device: int | str | None = None,
    input_channels: int = 2,
    output_channels: int = 2,
    block_size: int = 1024,
    f_min: float | None = None,
    f_max: float | None = None,
    smoothing: bool = True,
    raise_on_clipping: bool = True,
    filter_window_width: float = 1 / 3,
    recording_tail: float = 0.25,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Calibrate once, then measure any number of unknown loads."""
    if smoothing and f_min is not None and f_max is not None:
        half_window = filter_window_width / 2
        required_start = f_min / (2**half_window)
        required_end = f_max * (2**half_window)
        if f_start > required_start or f_end < required_end:
            warnings.warn(
                "The chirp does not cover the full smoothing windows at the "
                "analysis boundaries; extend f_start/f_end or reduce "
                "filter_window_width",
                RuntimeWarning,
                stacklevel=2,
            )

    test_signal = generate_chirp(
        fs=fs,
        duration=duration,
        f_start=f_start,
        f_end=f_end,
        amplitude=amplitude,
    )
    chirp_samples = len(test_signal)

    input("Connect the calibration resistor Rcal, then press Enter...")
    cal_recording = play_and_record(
        test_signal,
        fs,
        input_device=input_device,
        output_device=output_device,
        input_channels=input_channels,
        output_channels=output_channels,
        block_size=block_size,
        recording_tail=recording_tail,
    )
    cal_recording = trim_recording(cal_recording, chirp_samples)
    analyze_recording_levels(cal_recording, raise_on_clipping=raise_on_clipping)
    _, calibration = calculate_calibration_from_known_resistor(
        cal_recording[:, 0],
        cal_recording[:, 1],
        fs=fs,
        reference_resistor=reference_resistor,
        calibration_resistor=calibration_resistor,
        f_min=f_min,
        f_max=f_max,
        smoothing=smoothing,
        filter_window_width=filter_window_width,
    )

    measurements: list[tuple[np.ndarray, np.ndarray]] = []
    while True:
        measurement_number = len(measurements) + 1
        command = input(
            f"Connect unknown load for measurement {measurement_number}. "
            "Press Enter to measure or type q to finish: "
        ).strip().lower()
        if command == "q":
            break

        measurement_recording = play_and_record(
            test_signal,
            fs,
            input_device=input_device,
            output_device=output_device,
            input_channels=input_channels,
            output_channels=output_channels,
            block_size=block_size,
            recording_tail=recording_tail,
        )
        measurement_recording = trim_recording(
            measurement_recording,
            chirp_samples,
        )
        analyze_recording_levels(
            measurement_recording,
            raise_on_clipping=raise_on_clipping,
        )
        measurement = calculate_impedance(
            measurement_recording[:, 0],
            measurement_recording[:, 1],
            fs=fs,
            reference_resistor=reference_resistor,
            calibration=calibration,
            f_min=f_min,
            f_max=f_max,
            smoothing=smoothing,
            filter_window_width=filter_window_width,
        )
        measurements.append(measurement)

        freq, impedance = measurement
        _, ax = plot_impedance_magnitude(freq, impedance)
        ax.set_title(
            f"Load impedance and current phase, measurement {measurement_number}"
        )
        plt.show(block=False)
        plt.pause(0.1)

        print(f"Measurement {measurement_number} completed.")

    return measurements


def plot_impedance_magnitude(
    freq: np.ndarray,
    impedance: np.ndarray,
    log_x: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot impedance magnitude and current phase relative to voltage."""
    fig, ax = plt.subplots()
    current_phase = -np.angle(impedance, deg=True)

    if log_x:
        magnitude_line = ax.semilogx(
            freq,
            np.abs(impedance),
            label="|Z|",
        )[0]
    else:
        magnitude_line = ax.plot(
            freq,
            np.abs(impedance),
            label="|Z|",
        )[0]

    ax.set_xlabel("Frequency, Hz")
    ax.set_ylabel("|Z|, Ohm")
    ax.set_title("Load impedance magnitude and current phase")
    ax.grid(True, which="both")
    y_formatter = ScalarFormatter(useOffset=False)
    y_formatter.set_scientific(False)
    ax.yaxis.set_major_formatter(y_formatter)

    phase_ax = ax.twinx()
    phase_line = phase_ax.plot(
        freq,
        current_phase,
        color="tab:red",
        linestyle="--",
        label="Current phase",
    )[0]
    phase_ax.axhline(0.0, color="tab:red", linewidth=0.8, alpha=0.35)
    phase_ax.set_ylabel("Current phase relative to voltage, deg", color="tab:red")
    phase_ax.tick_params(axis="y", labelcolor="tab:red")
    phase_formatter = ScalarFormatter(useOffset=False)
    phase_formatter.set_scientific(False)
    phase_ax.yaxis.set_major_formatter(phase_formatter)

    ax.legend(handles=[magnitude_line, phase_line], loc="best")
    return fig, ax


def _as_2d_recording(recording: np.ndarray) -> np.ndarray:
    data = np.asarray(recording)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    if data.ndim != 2:
        raise ValueError("recording must have shape (samples, channels)")
    if data.shape[1] < 1:
        raise ValueError("recording must contain at least one channel")
    return data


def _safe_denominator(values: np.ndarray, eps: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.complex128)
    return np.where(np.abs(values) < eps, eps + 0j, values)


def _log_frequency_grid(
    freq: np.ndarray,
    f_min: float | None,
    f_max: float | None,
    n_output: int = 256,
) -> np.ndarray:
    """Build the single log-frequency grid used by all smoothed spectra."""
    if n_output <= 0:
        raise ValueError("n_output must be positive")

    positive = freq[freq > 0.0]
    if positive.size == 0:
        raise ValueError("frequency array has no positive bins")

    low = float(f_min) if f_min is not None else float(positive[0])
    high = float(f_max) if f_max is not None else float(positive[-1])
    if low <= 0.0:
        raise ValueError("f_min must be positive when smoothing is enabled")
    if high <= low:
        raise ValueError("f_max must be greater than f_min")
    return np.geomspace(low, high, n_output)


def _calculate_transfer_function(
    freq: np.ndarray,
    V1: np.ndarray,
    V2: np.ndarray,
    f_min: float | None,
    f_max: float | None,
    smoothing: bool,
    filter_window_width: float,
    eps: float,
) -> tuple[np.ndarray, np.ndarray]:
    if not smoothing:
        return freq, V2 / _safe_denominator(V1, eps)
    if filter_window_width <= 0.0:
        raise ValueError("filter_window_width must be positive")

    log_freq = _log_frequency_grid(freq, f_min, f_max)
    cross_spectrum = np.conj(V1) * V2
    input_power = np.abs(V1) ** 2
    smoothed_cross_spectrum = grid_filter(
        freq,
        cross_spectrum,
        log_freq,
        window=Windows.GAUSSIAN,
        w=filter_window_width,
    )
    smoothed_input_power = grid_filter(
        freq,
        input_power,
        log_freq,
        window=Windows.GAUSSIAN,
        w=filter_window_width,
    )
    transfer_function = smoothed_cross_spectrum / _safe_denominator(
        smoothed_input_power,
        eps,
    )
    return log_freq, transfer_function


def _apply_frequency_range(
    freq: np.ndarray,
    values: np.ndarray,
    f_min: float | None,
    f_max: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    mask = np.ones_like(freq, dtype=bool)
    if f_min is not None:
        mask &= freq >= f_min
    if f_max is not None:
        mask &= freq <= f_max
    return freq[mask], values[mask]


def _self_test() -> None:
    test_freq = np.linspace(0.0, 24000.0, 24001)
    common_phase = np.exp(1j * 0.003 * np.square(test_freq))
    expected_h = 0.4 + 0.1j
    transfer_freq, transfer = _calculate_transfer_function(
        test_freq,
        common_phase,
        expected_h * common_phase,
        f_min=20.0,
        f_max=20000.0,
        smoothing=True,
        filter_window_width=0.9,
        eps=1e-12,
    )
    assert transfer_freq.shape == transfer.shape
    assert np.allclose(transfer, expected_h, rtol=1e-10, atol=1e-10)

    fs = 48000
    t = np.arange(fs, dtype=np.float64) / fs
    ch1 = np.sin(2.0 * np.pi * 1000.0 * t)
    expected_z = 8.0
    reference_resistor = 10.0
    h = expected_z / (reference_resistor + expected_z)
    ch2 = ch1 * h

    freq, z = calculate_impedance(
        ch1,
        ch2,
        fs,
        reference_resistor,
        smoothing=False,
        f_min=900.0,
        f_max=1100.0,
    )
    idx = int(np.argmin(np.abs(freq - 1000.0)))
    assert np.iscomplexobj(z)
    assert np.isclose(z[idx].real, expected_z, rtol=1e-6, atol=1e-6)

    freq_cal, calibration = calculate_calibration_from_known_resistor(
        ch1,
        ch2,
        fs,
        reference_resistor=reference_resistor,
        calibration_resistor=expected_z,
        smoothing=False,
        f_min=900.0,
        f_max=1100.0,
    )
    freq_z, z_calibrated = calculate_impedance(
        ch1,
        ch2,
        fs,
        reference_resistor=reference_resistor,
        calibration=calibration,
        smoothing=False,
        f_min=900.0,
        f_max=1100.0,
    )
    assert freq_cal.shape == freq_z.shape
    assert np.iscomplexobj(z_calibrated)


def main() -> None:
    fs = 192000
    duration = 10.0
    f_start = 10.0
    f_end = 30000.0
    amplitude = 0.9

    reference_resistor = 3.25
    calibration_resistor = 10.4

    input_device = 25
    output_device = 22

    input_channels = 2
    output_channels = 2
    block_size = 1024

    f_min = 20.0
    f_max = 20000.0

    smoothing = True
    filter_window_width = 0.33
    recording_tail = 0.25

    raise_on_clipping = True

    measurements = measure_impedance_with_inline_calibration(
        fs=fs,
        duration=duration,
        reference_resistor=reference_resistor,
        calibration_resistor=calibration_resistor,
        f_start=f_start,
        f_end=f_end,
        amplitude=amplitude,
        input_device=input_device,
        output_device=output_device,
        input_channels=input_channels,
        output_channels=output_channels,
        block_size=block_size,
        f_min=f_min,
        f_max=f_max,
        smoothing=smoothing,
        raise_on_clipping=raise_on_clipping,
        filter_window_width=filter_window_width,
        recording_tail=recording_tail,
    )

    if measurements:
        plt.show()


if __name__ == "__main__":
    main()
