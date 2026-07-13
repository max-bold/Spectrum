from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal, cast

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIRECTORY = Path(__file__).resolve().parent
sys.path = [
    entry
    for entry in sys.path
    if Path(entry or ".").resolve() != SCRIPT_DIRECTORY
]
sys.path.insert(0, str(REPOSITORY_ROOT))

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.fft import rfft, rfftfreq
import sounddevice as sd

from audioanalysis import FrequencyBand, PinkNoiseThread  # noqa: E402
from audioanalysis.smoothing import log_smooth  # noqa: E402


DURATION = 30
BAND = (100, 10e3)
SMOOTHING_WIDTH = 1 / 3
OUTPUT_PATH = REPOSITORY_ROOT / "artifacts" / "pink_noise_fft_2.png"


def find_wasapi_device(
    name: str,
    kind: Literal["input", "output"],
) -> int:
    """Return the index of a matching Windows WASAPI audio device."""
    channel_key = f"max_{kind}_channels"
    matches: list[int] = []
    for index, device in enumerate(sd.query_devices()):
        host_api = sd.query_hostapis(int(device["hostapi"]))
        if (
            str(host_api["name"]) == "Windows WASAPI"
            and name.casefold() in str(device["name"]).casefold()
            and int(device[channel_key]) >= 2
        ):
            matches.append(index)
    if len(matches) != 1:
        raise ValueError(
            f"Expected one WASAPI {kind} device matching {name!r}, found {matches}"
        )
    return matches[0]


def record_pink_noise() -> tuple[NDArray[np.float32], int]:
    """Play pink noise through UMC outputs 1-2 and record UMC inputs 1-2."""
    output_device = find_wasapi_device("OUT 1-2 (BEHRINGER UMC", "output")
    input_device = find_wasapi_device("IN 1-2 (BEHRINGER UMC", "input")
    output_info = sd.query_devices(output_device, kind="output")
    input_info = sd.query_devices(input_device, kind="input")
    sample_rate = int(round(float(output_info["default_samplerate"])))
    input_sample_rate = int(round(float(input_info["default_samplerate"])))
    if input_sample_rate != sample_rate:
        raise ValueError(
            f"Input and output sample rates differ: {input_sample_rate} != {sample_rate}"
        )

    frame_count = int(round(DURATION * sample_rate))
    noise = PinkNoiseThread(
        device=output_device,
        band=FrequencyBand(*BAND),
    )
    print(
        f"Recording {DURATION:.1f} s at {sample_rate} Hz: "
        f"output={output_device}, input={input_device}",
        flush=True,
    )
    recording = sd.rec(
        frame_count,
        samplerate=sample_rate,
        channels=2,
        dtype="float32",
        device=input_device,
        blocking=False,
    )
    try:
        noise.start()
        sd.wait()
        noise.raise_if_failed()
    finally:
        noise.close(timeout=10.0)
    noise.raise_if_failed()
    return np.asarray(recording, dtype=np.float32), sample_rate


def corrected_spectrum(
    recording: NDArray[np.float32],
    sample_rate: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Calculate, smooth, and pink-correct the two-channel RFFT magnitude."""
    frequency = rfftfreq(recording.shape[0], d=1.0 / sample_rate)
    spectrum = cast(NDArray[np.complex64], rfft(recording, axis=0))
    magnitude = np.abs(spectrum) / float(recording.shape[0])
    smooth_frequency, smooth_magnitude = log_smooth(
        frequency,
        magnitude,
        width=SMOOTHING_WIDTH,
        points=1024,
    )
    smoothed = np.asarray(smooth_magnitude, dtype=np.float64)
    corrected = smoothed * np.sqrt(smooth_frequency[:, None])
    corrected_db = 20.0 * np.log10(np.maximum(corrected, 1e-20))
    return smooth_frequency, corrected_db


def plot_spectrum(
    frequency: NDArray[np.float64],
    values_db: NDArray[np.float64],
) -> None:
    """Save the corrected spectra for UMC input channels 1 and 2."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(12, 7), constrained_layout=True)
    axis.semilogx(frequency, values_db[:, 0], label="Input 1", linewidth=1.4)
    axis.semilogx(frequency, values_db[:, 1], label="Input 2", linewidth=1.4)
    axis.set(
        title="BEHRINGER UMC 204HD pink-noise loopback",
        xlabel="Frequency, Hz",
        ylabel="Corrected magnitude, dB (RFFT x sqrt(f))",
        xlim=(20,20e3),
    )
    axis.grid(True, which="both", alpha=0.3)
    axis.legend()
    figure.savefig(OUTPUT_PATH, dpi=160)
    plt.close(figure)


def main() -> None:
    recording, sample_rate = record_pink_noise()
    frequency, values_db = corrected_spectrum(recording, sample_rate)
    plot_spectrum(frequency, values_db)
    print(f"Saved {OUTPUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
