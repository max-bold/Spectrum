from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
from scipy.signal import chirp

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.windows import Windows, grid_filter


# Device selection. Use `print_devices()` once if you need PortAudio indices.
INPUT_DEVICE: int | str | None = None
OUTPUT_DEVICE: int | str | None = None

# Channel numbers are zero-based here: 0 = channel A / first channel,
# 1 = channel B / second channel. PortAudio mappings are one-based internally.
MIC_CHANNEL = 0
REFERENCE_CHANNEL = 1
OUTPUT_CHANNELS = (0,)

SWEEP_LENGTH = 5.0
PRE_SILENCE = 0.25
POST_SILENCE = 0.50
BAND = (20.0, 20_000.0)
AMPLITUDE = 0.30
FADE = 0.02
BLOCK_SIZE = 2048

SMOOTHING_POINTS = 1000
SMOOTHING_WIDTH_OCT = 1 / 6
DELAY_FIT_BAND = (80.0, 15_000.0)
WRAP_PHASE_TO_180 = True
REMOVE_CONSTANT_PHASE_OFFSET = False
MIN_REFERENCE_DB = -80.0
MIN_MIC_DB = -90.0
EPS = 1e-20


def print_devices() -> None:
    """Print sounddevice device list for manually filling the variables above."""
    print(sd.query_devices())


def device_default_sample_rate(
    device: int | str | None,
    kind: str,
) -> float | None:
    try:
        device_info = sd.query_devices(device, kind)
    except (sd.PortAudioError, ValueError):
        return None
    if not isinstance(device_info, dict):
        return None

    sample_rate = float(device_info.get("default_samplerate", 0.0))
    if sample_rate <= 0.0:
        return None
    return sample_rate


def default_sample_rate() -> int:
    input_fs = device_default_sample_rate(INPUT_DEVICE, "input")
    output_fs = device_default_sample_rate(OUTPUT_DEVICE, "output")
    rates = [rate for rate in (input_fs, output_fs) if rate is not None]

    if not rates:
        raise RuntimeError("Could not read default samplerate from audio devices")

    if len(rates) == 2 and abs(rates[0] - rates[1]) > 1.0:
        warnings.warn(
            "Input and output devices have different default sample rates "
            f"({rates[0]:.0f} Hz vs {rates[1]:.0f} Hz); using the lower one",
            RuntimeWarning,
            stacklevel=2,
        )

    return int(round(min(rates)))


def make_log_sweep(
    fs: int,
    length: float,
    band: tuple[float, float],
    amplitude: float,
    fade: float,
) -> np.ndarray:
    if fs <= 0:
        raise ValueError("sample rate must be positive")
    if length <= 0.0:
        raise ValueError("SWEEP_LENGTH must be positive")
    if band[0] <= 0.0 or band[1] <= band[0]:
        raise ValueError("BAND must be (positive_low, higher_high)")
    if band[1] >= fs / 2:
        raise ValueError("BAND high edge must be below Nyquist")
    if amplitude <= 0.0 or amplitude > 1.0:
        raise ValueError("AMPLITUDE must be in the range (0, 1]")

    n = int(round(length * fs))
    t = np.arange(n, dtype=np.float64) / fs
    sweep = amplitude * chirp(
        t,
        f0=band[0],
        t1=length,
        f1=band[1],
        method="logarithmic",
        phi=-90,
    )

    fade_samples = min(int(round(fade * fs)), n // 2)
    if fade_samples > 0:
        fade_in = np.linspace(0.0, 1.0, fade_samples)
        sweep[:fade_samples] *= fade_in
        sweep[-fade_samples:] *= fade_in[::-1]

    return sweep.astype(np.float32)


def make_playback_signal(sweep: np.ndarray, fs: int) -> np.ndarray:
    pre = np.zeros(int(round(PRE_SILENCE * fs)), dtype=np.float32)
    post = np.zeros(int(round(POST_SILENCE * fs)), dtype=np.float32)
    mono = np.concatenate([pre, sweep, post])

    if len(OUTPUT_CHANNELS) == 1:
        return mono.reshape(-1, 1)
    return np.repeat(mono.reshape(-1, 1), len(OUTPUT_CHANNELS), axis=1)


def play_and_record(playback: np.ndarray, fs: int) -> np.ndarray:
    input_mapping = [MIC_CHANNEL + 1, REFERENCE_CHANNEL + 1]
    output_mapping = [channel + 1 for channel in OUTPUT_CHANNELS]

    return sd.playrec(
        playback,
        samplerate=fs,
        dtype="float32",
        device=(INPUT_DEVICE, OUTPUT_DEVICE),
        input_mapping=input_mapping,
        output_mapping=output_mapping,
        blocksize=BLOCK_SIZE,
        blocking=True,
    )


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(np.asarray(x, dtype=np.float64)))))


def print_levels(recording: np.ndarray) -> None:
    names = (f"mic input {MIC_CHANNEL}", f"reference input {REFERENCE_CHANNEL}")
    for channel, name in enumerate(names):
        x = recording[:, channel]
        peak = float(np.max(np.abs(x)))
        level_rms = rms(x)
        clipping = " CLIPPING" if np.any(np.abs(x) >= 0.999) else ""
        print(f"{name}: peak={peak:.4f}, rms={level_rms:.4f}{clipping}")


def transfer_function(
    mic: np.ndarray,
    reference: np.ndarray,
    fs: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = min(len(mic), len(reference))
    mic = np.asarray(mic[:n], dtype=np.float64)
    reference = np.asarray(reference[:n], dtype=np.float64)

    mic -= np.mean(mic)
    reference -= np.mean(reference)

    mic_fft = np.fft.rfft(mic)
    ref_fft = np.fft.rfft(reference)
    freq = np.fft.rfftfreq(n, d=1.0 / fs)
    H = mic_fft / np.where(np.abs(ref_fft) > EPS, ref_fft, np.nan)
    return freq, H, mic_fft, ref_fft


def usable_band_mask(
    freq: np.ndarray,
    mic_fft: np.ndarray,
    ref_fft: np.ndarray,
    band: tuple[float, float],
) -> np.ndarray:
    ref_mag = np.abs(ref_fft)
    mic_mag = np.abs(mic_fft)
    ref_limit = np.max(ref_mag) * 10 ** (MIN_REFERENCE_DB / 20)
    mic_limit = np.max(mic_mag) * 10 ** (MIN_MIC_DB / 20)

    return (
        (freq >= band[0])
        & (freq <= band[1])
        & np.isfinite(ref_mag)
        & np.isfinite(mic_mag)
        & (ref_mag >= ref_limit)
        & (mic_mag >= mic_limit)
    )


def estimate_delay(
    freq: np.ndarray,
    H: np.ndarray,
    mic_fft: np.ndarray,
    ref_fft: np.ndarray,
    fit_band: tuple[float, float],
) -> float:
    mask = usable_band_mask(freq, mic_fft, ref_fft, fit_band) & np.isfinite(H)
    if np.count_nonzero(mask) < 3:
        raise ValueError("Not enough valid FFT bins for delay estimation")

    ff = freq[mask]
    phase = np.unwrap(np.angle(H[mask]))

    weights = np.sqrt(np.abs(mic_fft[mask]) * np.abs(ref_fft[mask]))
    weights /= np.max(weights)

    slope, _ = np.polyfit(ff, phase, 1, w=weights)
    return float(-slope / (2 * np.pi))


def compensated_phase_response(
    freq: np.ndarray,
    H: np.ndarray,
    delay: float,
    band: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    compensated = H * np.exp(1j * 2 * np.pi * freq * delay)
    grid = np.geomspace(band[0], band[1], SMOOTHING_POINTS)
    smoothed = grid_filter(
        freq,
        compensated,
        grid,
        window=Windows.GAUSSIAN,
        w=SMOOTHING_WIDTH_OCT,
    )
    phase_deg = np.rad2deg(np.unwrap(np.angle(smoothed)))
    if REMOVE_CONSTANT_PHASE_OFFSET:
        phase_deg -= np.nanmedian(phase_deg)
    if WRAP_PHASE_TO_180:
        phase_deg = (phase_deg + 180.0) % 360.0 - 180.0
    magnitude_db = 20.0 * np.log10(np.maximum(np.abs(smoothed), EPS))
    return grid, magnitude_db, phase_deg


def plot_response(
    grid: np.ndarray,
    magnitude_db: np.ndarray,
    phase_deg: np.ndarray,
    delay: float,
) -> None:
    fig, (mag_ax, phase_ax) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(11, 7),
        constrained_layout=True,
    )

    mag_ax.semilogx(grid, magnitude_db, color="tab:blue")
    mag_ax.set_ylabel("Mic / reference, dB")
    mag_ax.grid(True, which="both", alpha=0.3)

    phase_ax.semilogx(grid, phase_deg, color="tab:red")
    phase_ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.4)
    phase_ax.set_xlabel("Frequency, Hz")
    phase_ax.set_ylabel("Phase, deg")
    phase_ax.grid(True, which="both", alpha=0.3)

    fig.suptitle(
        f"Acoustic phase relative to amplifier output; "
        f"delay compensation {delay * 1000:.3f} ms"
    )
    plt.show()


def run_measurement() -> None:
    fs = default_sample_rate()
    sweep = make_log_sweep(fs, SWEEP_LENGTH, BAND, AMPLITUDE, FADE)
    playback = make_playback_signal(sweep, fs)

    print(
        f"Playing {SWEEP_LENGTH:.2f} s log sweep, "
        f"{BAND[0]:.0f}-{BAND[1]:.0f} Hz at {fs} Hz..."
    )
    recording = play_and_record(playback, fs)
    print_levels(recording)

    if rms(recording[:, 1]) < 1e-5:
        warnings.warn(
            "Reference channel is very quiet; check amplifier output to input B",
            RuntimeWarning,
            stacklevel=2,
        )
    if rms(recording[:, 0]) < 1e-5:
        warnings.warn(
            "Mic channel is very quiet; check microphone input A",
            RuntimeWarning,
            stacklevel=2,
        )

    freq, H, mic_fft, ref_fft = transfer_function(
        recording[:, 0],
        recording[:, 1],
        fs,
    )
    delay = estimate_delay(freq, H, mic_fft, ref_fft, DELAY_FIT_BAND)
    print(f"delay: {delay * 1000:.3f} ms ({delay * fs:.1f} samples)")

    grid, magnitude_db, phase_deg = compensated_phase_response(freq, H, delay, BAND)
    plot_response(grid, magnitude_db, phase_deg, delay)


if __name__ == "__main__":
    # Uncomment this while choosing INPUT_DEVICE and OUTPUT_DEVICE.
    # print_devices()
    run_measurement()
